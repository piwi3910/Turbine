//! Eviction policies (P4 S-7, TS §8; contract §24 `eviction_policy`): pluggable block scoring,
//! higher score = keep.
//!
//! A policy is one file in this directory plus one entry in [`registry`]; `kv.policy` selects
//! it by name (validated against the registry before any port is bound). Policies are pure
//! functions of the block's facts, tier estimates, the configured weights and a timestamp, so
//! they are deterministic under the simulator and benchmarkable against the `lru` baseline.
//! Which blocks may be scored at all (unreferenced, leaf-first) is the directory's call, and
//! the victim order's tie-break is [`order_victims`]'s — never the policy's.

#[cfg(test)]
pub(crate) mod conformance;
mod cost_aware;
mod lru;

pub use cost_aware::CostAwarePolicy;
pub use lru::LruPolicy;

use std::time::Duration;

use turbine_core::config::KvPolicyWeights;
use turbine_core::registry::{Module, Registry, UnknownModule};
use turbine_core::types::PressureState;

use crate::directory::{CostEstimate, KvBlock, KvPriority, Timestamp};
use crate::identity::KvKey;
use crate::tier::TierId;

/// The per-block facts a policy scores.
#[derive(Clone, Debug)]
pub struct KvBlockSummary {
    pub key: KvKey,
    pub size_bytes: u64,
    pub tokens: u32,
    pub last_access: Timestamp,
    pub decayed_hits: f64,
    pub decayed_at: Timestamp,
    pub child_count: u32,
    pub priority: KvPriority,
}

impl KvBlockSummary {
    pub fn of(b: &KvBlock) -> Self {
        KvBlockSummary {
            key: b.key,
            size_bytes: b.size_bytes,
            tokens: b.token_range.end - b.token_range.start,
            last_access: b.last_access,
            decayed_hits: b.decayed_hits,
            decayed_at: b.decayed_at,
            child_count: b.child_count,
            priority: b.priority,
        }
    }
}

/// Everything a policy needs to value one copy of a block in one tier.
#[derive(Clone, Debug)]
pub struct BlockScoreInputs {
    pub block: KvBlockSummary,
    pub tier: TierId,
    pub tier_capacity: u64,
    pub tier_pressure: PressureState,
    /// Phase 3 prefill-throughput EWMA (tokens/s).
    pub prefill_tps: f64,
    /// Seconds to bring the block back to L0 from the tier it would be demoted to (its
    /// recompute cost when it would be dropped).
    pub retrieval: CostEstimate,
    pub session_hot: bool,
    /// Token depth of the block's end (`token_range.end`).
    pub depth_tokens: u32,
}

/// `kv.policy_weights`: the tunables every policy receives (a policy may ignore them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyWeights {
    /// Reuse boost of a block whose session is hot (0..1).
    pub session_active: f64,
    /// Half-life of a block's decayed hit count.
    pub hit_half_life: Duration,
}

impl PolicyWeights {
    pub fn from_config(w: &KvPolicyWeights) -> Self {
        PolicyWeights {
            session_active: w.session_active,
            hit_half_life: w.hit_half_life.0,
        }
    }
}

impl Default for PolicyWeights {
    fn default() -> Self {
        PolicyWeights::from_config(&KvPolicyWeights::default())
    }
}

/// An eviction policy (contract §24 `eviction_policy`). Stateless: the same inputs always give
/// the same score, with no clock, randomness or interior state.
pub trait EvictionPolicy: Module {
    /// Value of keeping the copy where it is; higher = keep. Finite and ≥ 0.
    fn score(&self, b: &BlockScoreInputs, w: &PolicyWeights, now: Timestamp) -> f64;
}

static REGISTRY: Registry<dyn EvictionPolicy> =
    Registry::new("eviction_policy", &[&CostAwarePolicy, &LruPolicy]);

/// Every registered eviction policy; `cost_aware` (first) is the configuration default.
pub fn registry() -> &'static Registry<dyn EvictionPolicy> {
    &REGISTRY
}

/// A registered policy with the configured weights: what the hierarchy scores with.
#[derive(Clone, Copy)]
pub struct SelectedPolicy {
    pub policy: &'static dyn EvictionPolicy,
    pub weights: PolicyWeights,
}

impl SelectedPolicy {
    pub fn name(&self) -> &'static str {
        self.policy.name()
    }

    pub fn score(&self, b: &BlockScoreInputs, now: Timestamp) -> f64 {
        self.policy.score(b, &self.weights, now)
    }
}

impl std::fmt::Debug for SelectedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedPolicy")
            .field("policy", &self.name())
            .field("weights", &self.weights)
            .finish()
    }
}

/// The policy `kv.policy` names, weighted by `kv.policy_weights`. No log: the server logs the
/// selection once (`registry().select`); simulators build many hierarchies.
pub fn make_policy(name: &str, w: &KvPolicyWeights) -> Result<SelectedPolicy, UnknownModule> {
    let policy = registry()
        .get(name)
        .ok_or_else(|| registry().unknown(name))?;
    Ok(SelectedPolicy {
        policy,
        weights: PolicyWeights::from_config(w),
    })
}

/// Seconds to prefill a block's tokens at its depth (attention cost grows with depth):
/// `tokens / prefill_tps × (1 + depth / 8192)`.
pub fn recompute_seconds(tokens: u32, depth_tokens: u32, prefill_tps: f64) -> f64 {
    f64::from(tokens) / prefill_tps.max(1.0) * (1.0 + f64::from(depth_tokens) / 8192.0)
}

/// Candidates scored and ordered lowest value first (the first victim first). Ties: older last
/// access first, then the deeper block first (leaves before their ancestors).
pub fn order_victims(
    policy: &SelectedPolicy,
    candidates: &[BlockScoreInputs],
    now: Timestamp,
) -> Vec<(KvKey, f64)> {
    let mut scored: Vec<(&BlockScoreInputs, f64)> = candidates
        .iter()
        .map(|c| (c, policy.score(c, now)))
        .collect();
    scored.sort_by(|(a, va), (b, vb)| {
        va.total_cmp(vb)
            .then_with(|| a.block.last_access.cmp(&b.block.last_access))
            .then_with(|| b.depth_tokens.cmp(&a.depth_tokens))
    });
    scored.into_iter().map(|(c, v)| (c.block.key, v)).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::directory::KvDirectory;
    use crate::directory::tests::{block, l0};
    use crate::tier::KvLocation;

    /// 1 MiB, 16 tokens, 2 hits, 1 child, in a YELLOW 1 GiB L0, 10,000 tok/s, 1 ms retrieval.
    pub(crate) fn base() -> BlockScoreInputs {
        BlockScoreInputs {
            block: KvBlockSummary {
                key: KvKey([1; 16]),
                size_bytes: 1 << 20,
                tokens: 16,
                last_access: Duration::from_secs(10),
                decayed_hits: 2.0,
                decayed_at: Duration::from_secs(10),
                child_count: 1,
                priority: KvPriority(1.0),
            },
            tier: TierId::L0,
            tier_capacity: 1 << 30,
            tier_pressure: PressureState::Yellow,
            prefill_tps: 10_000.0,
            retrieval: CostEstimate { seconds: 0.001 },
            session_hot: false,
            depth_tokens: 1024,
        }
    }

    /// A one-term change to the base inputs.
    type Tweak = fn(&mut BlockScoreInputs);

    /// Scores every block the same, so only the tie-break orders them.
    struct Flat;

    impl Module for Flat {
        fn name(&self) -> &'static str {
            "flat"
        }
    }

    impl EvictionPolicy for Flat {
        fn score(&self, _b: &BlockScoreInputs, _w: &PolicyWeights, _now: Timestamp) -> f64 {
            1.0
        }
    }

    fn weights() -> PolicyWeights {
        PolicyWeights {
            session_active: 0.5,
            hit_half_life: Duration::from_secs(60),
        }
    }

    fn selected(policy: &'static dyn EvictionPolicy) -> SelectedPolicy {
        SelectedPolicy {
            policy,
            weights: weights(),
        }
    }

    #[test]
    fn cost_aware_ordering() {
        let p = selected(&CostAwarePolicy);
        assert_eq!(p.name(), "cost_aware");
        let now = Duration::from_secs(20);
        let v0 = p.score(&base(), now);
        let raise: [(&str, Tweak); 5] = [
            ("more hits", |b| b.block.decayed_hits = 20.0),
            ("a hot session", |b| b.session_hot = true),
            ("a deeper block (recompute cost)", |b| b.depth_tokens = 8192),
            ("slower prefill (recompute cost)", |b| {
                b.prefill_tps = 1_000.0
            }),
            ("high priority", |b| b.block.priority = KvPriority(2.0)),
        ];
        for (what, f) in raise {
            let mut b = base();
            f(&mut b);
            assert!(p.score(&b, now) > v0, "{what} must raise the value");
        }
        let lower: [(&str, Tweak); 4] = [
            ("a larger size", |b| b.block.size_bytes = 4 << 20),
            ("higher tier pressure", |b| {
                b.tier_pressure = PressureState::Red
            }),
            ("cheaper retrieval", |b| {
                b.retrieval = CostEstimate { seconds: 0.0001 }
            }),
            ("low priority", |b| b.block.priority = KvPriority(0.5)),
        ];
        for (what, f) in lower {
            let mut b = base();
            f(&mut b);
            assert!(p.score(&b, now) < v0, "{what} must lower the value");
        }
        assert!(
            p.score(&base(), Duration::from_secs(600)) < v0,
            "hits decay with the half-life"
        );
        let mut hot = base();
        hot.session_hot = true;
        hot.block.decayed_hits = 1e9;
        assert_eq!(
            CostAwarePolicy.reuse_probability(&hot, &weights(), now),
            1.0,
            "capped at 1"
        );
        let mut zero = base();
        zero.retrieval = CostEstimate { seconds: 0.0 };
        assert!(p.score(&zero, now) > 0.0, "retrieval is floored at 1 µs");

        // Protected blocks are never candidates: referenced ones, and parents of a child cached
        // in the same tier.
        let (k1, k2, k3) = (KvKey([1; 16]), KvKey([2; 16]), KvKey([3; 16]));
        let mut dir = KvDirectory::new(8);
        dir.insert(block(k1, None, 0, &[0; 16], &[l0(0)])).unwrap();
        dir.insert(block(k2, Some(k1), 1, &[1; 16], &[l0(1)]))
            .unwrap();
        dir.insert(block(k3, None, 0, &[3; 16], &[l0(2)])).unwrap();
        dir.get_mut(&k3).unwrap().ref_count = 1;
        let none = HashSet::new();
        let keys = |dir: &KvDirectory| -> Vec<KvKey> {
            dir.candidates(TierId::L0, &none)
                .iter()
                .map(|b| b.key)
                .collect()
        };
        assert_eq!(keys(&dir), [k2], "only the unreferenced leaf is evictable");
        // Once the child lives only in L1, the parent may leave L0.
        let l1 = KvLocation {
            tier: TierId::L1,
            slot: 0,
        };
        dir.add_location(&k2, l1);
        dir.remove_location(&k2, TierId::L0);
        assert_eq!(keys(&dir), [k1]);

        // Scoring a directory block and ordering victims: lowest value first.
        let summary = KvBlockSummary::of(dir.get(&k1).unwrap());
        assert_eq!((summary.tokens, summary.child_count), (16, 1));
        let cheap = BlockScoreInputs {
            retrieval: CostEstimate { seconds: 1e-5 },
            ..base()
        };
        let mut dear = base();
        dear.block.key = KvKey([9; 16]);
        let order = order_victims(&p, &[dear.clone(), cheap.clone()], now);
        assert_eq!(
            order[0].0, cheap.block.key,
            "cheap retrieval is evicted first"
        );
        assert!(order[0].1 < order[1].1);

        // Ties: older last access first, then the deeper block (a leaf) first.
        let mut older = base();
        older.block.key = KvKey([10; 16]);
        older.block.last_access = Duration::from_secs(5);
        let mut deeper = base();
        deeper.block.key = KvKey([11; 16]);
        deeper.depth_tokens = 4096;
        let order: Vec<KvKey> = order_victims(&selected(&Flat), &[base(), deeper, older], now)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(order, [KvKey([10; 16]), KvKey([11; 16]), KvKey([1; 16])]);

        // LRU: the benchmark baseline scores recency only.
        let lru = selected(&LruPolicy);
        assert_eq!(lru.name(), "lru");
        let order = order_victims(&lru, &[base(), older_by(3)], now);
        assert_eq!(order[0].0, KvKey([12; 16]));
        assert_eq!(recompute_seconds(16, 8192, 8.0), 4.0);
    }

    /// `kv.policy` and `kv.policy_weights` select and parameterise the policy; an unregistered
    /// name is refused naming the registered ones.
    #[test]
    fn make_policy_follows_config() {
        use turbine_core::config::{HumanDuration, KvPolicyWeights};

        let now = Duration::from_secs(20);
        let w = KvPolicyWeights {
            session_active: 0.25,
            hit_half_life: HumanDuration::from_secs(30),
        };
        let lru = make_policy("lru", &w).unwrap();
        assert_eq!(lru.name(), "lru");
        assert_eq!(
            lru.score(&base(), now),
            LruPolicy.score(&base(), &weights(), now)
        );

        let cost = make_policy("cost_aware", &w).unwrap();
        assert_eq!(cost.name(), "cost_aware");
        let expected = PolicyWeights {
            session_active: 0.25,
            hit_half_life: Duration::from_secs(30),
        };
        assert_eq!(cost.weights, expected);
        let mut hot = base();
        hot.session_hot = true;
        for b in [base(), hot] {
            assert_eq!(
                cost.score(&b, now),
                CostAwarePolicy.score(&b, &expected, now)
            );
        }
        let err = make_policy("lfu", &w).unwrap_err();
        assert_eq!(err.point, "eviction_policy");
        assert_eq!(err.registered, "cost_aware, lru");
    }

    fn older_by(secs: u64) -> BlockScoreInputs {
        let mut b = base();
        b.block.key = KvKey([12; 16]);
        b.block.last_access = Duration::from_secs(10 - secs);
        b
    }
}
