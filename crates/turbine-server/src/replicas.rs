//! Data-parallel replica choice for the API (P5 S-7): which replica's engine a request goes to.
//!
//! [`ReplicaRouter::pick`] builds one [`ReplicaView`] per ready replica — the pressure state and
//! circuit of its controller, its outstanding tokens (what its engine published plus what is in
//! transit to it) and whether it holds the request's prefix — and asks
//! `turbine_distributed::router::route`, which counts `turbine_dp_routed_total{replica,reason}`
//! and logs the decision.
//!
//! Prefix affinity: the replica that served a request remembers the key of its first full KV
//! block (the cache salt and that block's token ids, the same content the Phase 4 directory keys
//! blocks by). A later request with the same first block goes where that block was computed, as
//! long as that replica is eligible. The table is bounded (oldest entries leave first), so it
//! never outgrows [`AFFINITY_ENTRIES`]; an entry whose block the replica has since evicted only
//! costs one cold prefill there, exactly as a directory miss would.

use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use turbine_core::config::DpRouterPolicy;
use turbine_core::types::{CircuitState, PressureState, ReplicaId};
use turbine_distributed::router::{DpRouteReason, ReplicaView, RouterMetrics, route};
use turbine_observability::MetricsRegistry;

/// Prefix-affinity entries kept (first-block keys), across replicas.
pub const AFFINITY_ENTRIES: usize = 16_384;

/// What the router needs of one replica right now.
#[derive(Clone, Copy, Debug)]
pub struct ReplicaLoad {
    pub state: PressureState,
    pub circuit: CircuitState,
    /// Tokens the replica's engine published as still to process.
    pub engine_outstanding: u64,
}

/// The replica choice for one process: policy, metrics, tokens in transit and the affinity table.
pub struct ReplicaRouter {
    policy: DpRouterPolicy,
    metrics: RouterMetrics,
    /// Per replica: tokens of submissions sent to its engine and not yet counted by it.
    in_transit: Vec<AtomicU64>,
    affinity: Mutex<Affinity>,
    block_tokens: usize,
}

#[derive(Default)]
struct Affinity {
    by_key: HashMap<u64, u32>,
    order: VecDeque<u64>,
}

impl ReplicaRouter {
    pub fn new(
        replicas: usize,
        policy: DpRouterPolicy,
        block_tokens: u32,
        reg: &MetricsRegistry,
    ) -> ReplicaRouter {
        ReplicaRouter {
            policy,
            metrics: RouterMetrics::register(reg),
            in_transit: (0..replicas).map(|_| AtomicU64::new(0)).collect(),
            affinity: Mutex::new(Affinity::default()),
            block_tokens: block_tokens.max(1) as usize,
        }
    }

    /// The first-block key of a prompt, `None` when it has no full block.
    pub fn prefix_key(&self, cache_salt: Option<&str>, prompt: &[u32]) -> Option<u64> {
        let block = prompt.get(..self.block_tokens)?;
        let mut h = DefaultHasher::new();
        cache_salt.unwrap_or_default().hash(&mut h);
        block.hash(&mut h);
        Some(h.finish())
    }

    /// Picks the replica for a request of `key` (its first-block key) among `loads` (index =
    /// replica); `None` loads are replicas that are not ready and are never picked.
    pub fn pick(
        &self,
        loads: &[Option<ReplicaLoad>],
        key: Option<u64>,
    ) -> Option<(usize, DpRouteReason)> {
        let holder = key.and_then(|k| {
            self.affinity
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .by_key
                .get(&k)
                .copied()
        });
        let views: Vec<ReplicaView> = loads
            .iter()
            .enumerate()
            .filter_map(|(r, load)| {
                let load = load.as_ref()?;
                Some(ReplicaView {
                    replica: ReplicaId(r as u32),
                    state: load.state,
                    circuit: load.circuit,
                    outstanding_tokens: load.engine_outstanding
                        + self.in_transit[r].load(Ordering::Acquire),
                    has_prefix: holder == Some(r as u32),
                })
            })
            .collect();
        if views.is_empty() {
            return None;
        }
        let (replica, reason) = route(&views, self.policy);
        self.metrics.routed(replica, reason);
        tracing::debug!(
            event = "dp_route",
            replica = replica.0,
            reason = reason.as_str(),
            "request routed to a data-parallel replica"
        );
        Some((replica.0 as usize, reason))
    }

    /// `tokens` are on their way to `replica`'s engine (until [`Self::delivered`]).
    pub fn sending(&self, replica: usize, tokens: u64) {
        self.in_transit[replica].fetch_add(tokens, Ordering::AcqRel);
    }

    /// The engine has taken (or refused) the submission: it counts the tokens itself now.
    pub fn delivered(&self, replica: usize, tokens: u64) {
        let _ = self.in_transit[replica].fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
            Some(v.saturating_sub(tokens))
        });
    }

    /// `replica` computed the prefix of `key` (it served that request).
    pub fn remember(&self, key: Option<u64>, replica: usize) {
        let Some(key) = key else {
            return;
        };
        let mut a = self.affinity.lock().unwrap_or_else(|p| p.into_inner());
        if a.by_key.insert(key, replica as u32).is_none() {
            a.order.push_back(key);
            while a.order.len() > AFFINITY_ENTRIES {
                if let Some(old) = a.order.pop_front() {
                    a.by_key.remove(&old);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(state: PressureState, outstanding: u64) -> Option<ReplicaLoad> {
        Some(ReplicaLoad {
            state,
            circuit: CircuitState::Healthy,
            engine_outstanding: outstanding,
        })
    }

    /// Least-loaded counts tokens in transit; a remembered first block pulls its request back to
    /// the replica that computed it unless that replica is under pressure; replicas that are not
    /// ready are skipped; the affinity table stays bounded.
    #[test]
    fn routes_by_load_affinity_and_readiness() {
        let reg = MetricsRegistry::new();
        let r = ReplicaRouter::new(2, DpRouterPolicy::PrefixAffinity, 4, &reg);
        let green = [
            load(PressureState::Green, 100),
            load(PressureState::Green, 50),
        ];
        assert_eq!(r.pick(&green, None).map(|p| p.0), Some(1));
        r.sending(1, 200);
        assert_eq!(r.pick(&green, None).map(|p| p.0), Some(0));
        r.delivered(1, 200);

        let key = r.prefix_key(Some("s"), &[1, 2, 3, 4, 5]);
        assert!(key.is_some());
        assert_eq!(r.prefix_key(None, &[1, 2, 3]), None, "no full block");
        assert_ne!(
            key,
            r.prefix_key(None, &[1, 2, 3, 4]),
            "the salt is part of the key"
        );
        r.remember(key, 0);
        assert_eq!(
            r.pick(&green, key),
            Some((0, DpRouteReason::PrefixAffinity))
        );
        let orange0 = [
            load(PressureState::Orange, 0),
            load(PressureState::Green, 50),
        ];
        assert_eq!(
            r.pick(&orange0, key),
            Some((1, DpRouteReason::PressureAvoidance))
        );
        assert_eq!(
            r.pick(&[None, load(PressureState::Green, 9)], key)
                .map(|p| p.0),
            Some(1)
        );
        assert_eq!(r.pick(&[None, None], None), None);

        for i in 0..(AFFINITY_ENTRIES as u64 + 10) {
            r.remember(Some(i), 1);
        }
        let a = r.affinity.lock().unwrap();
        assert_eq!(a.by_key.len(), AFFINITY_ENTRIES);
        assert_eq!(a.order.len(), AFFINITY_ENTRIES);
        drop(a);
        let text = reg.render().unwrap();
        assert!(text.contains("turbine_dp_routed_total"), "{text}");
    }
}
