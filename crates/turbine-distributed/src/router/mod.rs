//! Data-parallel replica router (P5 S-7): the [`RouterPolicy`] extension point
//! (`dp_router_policy`), selected by `parallel.router`.
//!
//! Eligible replicas are below ORANGE with a circuit that is neither `CIRCUIT_OPEN` nor
//! `DRAINING`; when none is eligible every replica is a candidate ([`candidates`]). A policy
//! picks among the candidates: `prefix_affinity` prefers a candidate holding the prompt's first
//! KV blocks, else (like `least_loaded`) the candidate with the fewest outstanding tokens, ties
//! to the lowest index. Skipping a prefix holder for pressure is reported as
//! `pressure_avoidance`. The server fills `has_prefix` from each replica's KV directory lookup of
//! the prompt's first block keys.

pub mod conformance;
pub mod least_loaded;
pub mod prefix_affinity;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use turbine_core::registry::{Module, Registry, UnknownModule};
use turbine_core::types::{CircuitState, PressureState, ReplicaId};
use turbine_observability::MetricsRegistry;

pub use least_loaded::LeastLoaded;
pub use prefix_affinity::PrefixAffinity;

/// Why a replica was chosen; `as_str` is the `reason` label value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DpRouteReason {
    PrefixAffinity,
    LeastLoaded,
    PressureAvoidance,
    OnlyCandidate,
}

impl DpRouteReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DpRouteReason::PrefixAffinity => "prefix_affinity",
            DpRouteReason::LeastLoaded => "least_loaded",
            DpRouteReason::PressureAvoidance => "pressure_avoidance",
            DpRouteReason::OnlyCandidate => "only_candidate",
        }
    }
}

/// What the router knows about one replica at routing time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicaView {
    pub replica: ReplicaId,
    pub state: PressureState,
    pub circuit: CircuitState,
    pub outstanding_tokens: u64,
    pub has_prefix: bool,
}

impl ReplicaView {
    /// Below ORANGE, with a circuit that is neither `CIRCUIT_OPEN` nor `DRAINING`.
    pub fn eligible(&self) -> bool {
        self.state < PressureState::Orange
            && !matches!(
                self.circuit,
                CircuitState::CircuitOpen | CircuitState::Draining
            )
    }
}

/// The replicas a policy may pick: the eligible ones, or every replica when none is eligible.
pub fn candidates(views: &[ReplicaView]) -> impl Iterator<Item = &ReplicaView> {
    let any_eligible = views.iter().any(ReplicaView::eligible);
    views.iter().filter(move |v| !any_eligible || v.eligible())
}

/// The least-loaded view, ties to the lowest replica index.
pub fn least_loaded<'a>(views: impl Iterator<Item = &'a ReplicaView>) -> Option<&'a ReplicaView> {
    views.min_by_key(|v| (v.outstanding_tokens, v.replica))
}

/// A registered DP routing policy (extension point `dp_router_policy`, contract §24).
pub trait RouterPolicy: Module {
    /// Picks the replica for one request among `views` (at least two; [`route`] answers a single
    /// replica itself with `only_candidate`). Must pick from [`candidates`], be deterministic,
    /// and break ties by the lowest replica index.
    fn choose(&self, views: &[ReplicaView]) -> (ReplicaId, DpRouteReason);
}

static PREFIX_AFFINITY: PrefixAffinity = PrefixAffinity;
static LEAST_LOADED: LeastLoaded = LeastLoaded;

static DP_ROUTER_POLICIES: Registry<dyn RouterPolicy> =
    Registry::new("dp_router_policy", &[&PREFIX_AFFINITY, &LEAST_LOADED]);

/// The registered DP router policies, in registration order.
pub fn registry() -> &'static Registry<dyn RouterPolicy> {
    &DP_ROUTER_POLICIES
}

/// The policy named by `parallel.router`, logging `event="module_selected"`.
pub fn select(name: &str) -> Result<&'static dyn RouterPolicy, UnknownModule> {
    registry().select(name, "parallel.router")
}

/// Picks the replica for one request. Panics on an empty slice (a plan has at least one
/// replica).
pub fn route(views: &[ReplicaView], policy: &dyn RouterPolicy) -> (ReplicaId, DpRouteReason) {
    assert!(!views.is_empty(), "routing needs at least one replica");
    if views.len() == 1 {
        return (views[0].replica, DpRouteReason::OnlyCandidate);
    }
    policy.choose(views)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RouteLabels {
    replica: u32,
    reason: &'static str,
}

/// `turbine_dp_routed_total{replica,reason}`.
#[derive(Clone)]
pub struct RouterMetrics {
    routed: Family<RouteLabels, Counter>,
}

impl RouterMetrics {
    pub fn register(reg: &MetricsRegistry) -> Self {
        let routed = reg.register(
            "turbine_dp_routed",
            "Requests routed to a data-parallel replica, by reason",
            Family::<RouteLabels, Counter>::default(),
        );
        RouterMetrics { routed }
    }

    /// Counts one routing decision and logs it with its reason code.
    pub fn routed(&self, replica: ReplicaId, reason: DpRouteReason) {
        self.routed
            .get_or_create(&RouteLabels {
                replica: replica.0,
                reason: reason.as_str(),
            })
            .inc();
        tracing::debug!(
            event = "dp_routed",
            replica = replica.0,
            reason = reason.as_str(),
            "request routed"
        );
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::types::{CircuitState, PressureState, ReplicaId};
    use turbine_observability::MetricsRegistry;

    use super::*;

    fn view(i: u32, state: PressureState, outstanding: u64, prefix: bool) -> ReplicaView {
        ReplicaView {
            replica: ReplicaId(i),
            state,
            circuit: CircuitState::Healthy,
            outstanding_tokens: outstanding,
            has_prefix: prefix,
        }
    }

    #[test]
    fn routing_policy() {
        use PressureState::{Green, Orange, Red, Yellow};
        let affinity = select("prefix_affinity").expect("registered");
        let least = select("least_loaded").expect("registered");

        // The prefix holder wins even when busier.
        let v = [view(0, Green, 10, false), view(1, Green, 500, true)];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(1), DpRouteReason::PrefixAffinity)
        );
        // least_loaded ignores the prefix.
        assert_eq!(route(&v, least), (ReplicaId(0), DpRouteReason::LeastLoaded));
        // No prefix anywhere: fewest outstanding tokens, ties to the lowest index.
        let v = [
            view(0, Green, 300, false),
            view(1, Yellow, 100, false),
            view(2, Green, 100, false),
        ];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(1), DpRouteReason::LeastLoaded)
        );
        // The prefix holder is ORANGE while another is GREEN: pressure wins.
        let v = [view(0, Orange, 0, true), view(1, Green, 900, false)];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(1), DpRouteReason::PressureAvoidance)
        );
        // An open circuit is skipped like pressure.
        let mut open = view(0, Green, 0, true);
        open.circuit = CircuitState::CircuitOpen;
        let v = [open, view(1, Green, 900, false)];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(1), DpRouteReason::PressureAvoidance)
        );
        let mut draining = view(1, Green, 0, false);
        draining.circuit = CircuitState::Draining;
        let v = [view(0, Yellow, 50, false), draining];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(0), DpRouteReason::LeastLoaded)
        );
        // Every replica RED: the least-loaded RED one.
        let v = [
            view(0, Red, 70, false),
            view(1, Red, 20, false),
            view(2, Red, 40, false),
        ];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(1), DpRouteReason::LeastLoaded)
        );
        // One replica.
        let v = [view(3, Red, 0, false)];
        assert_eq!(
            route(&v, affinity),
            (ReplicaId(3), DpRouteReason::OnlyCandidate)
        );

        // Each decision counts once in turbine_dp_routed_total{replica,reason}.
        let reg = MetricsRegistry::new();
        let m = RouterMetrics::register(&reg);
        m.routed(ReplicaId(1), DpRouteReason::PrefixAffinity);
        m.routed(ReplicaId(1), DpRouteReason::PrefixAffinity);
        m.routed(ReplicaId(0), DpRouteReason::PressureAvoidance);
        let text = reg.render().expect("renders");
        assert!(
            text.contains("turbine_dp_routed_total{replica=\"1\",reason=\"prefix_affinity\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("turbine_dp_routed_total{replica=\"0\",reason=\"pressure_avoidance\"} 1"),
            "{text}"
        );
    }
}
