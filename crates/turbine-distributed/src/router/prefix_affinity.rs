//! `prefix_affinity` (the default): the least-loaded candidate holding the prompt's first KV
//! blocks, else the least-loaded candidate — `pressure_avoidance` when a holder was skipped for
//! pressure or its circuit, `least_loaded` otherwise.

use turbine_core::registry::Module;
use turbine_core::types::ReplicaId;

use super::{DpRouteReason, ReplicaView, RouterPolicy, candidates, least_loaded};

/// The `prefix_affinity` module of the `dp_router_policy` registry.
pub struct PrefixAffinity;

impl Module for PrefixAffinity {
    fn name(&self) -> &'static str {
        "prefix_affinity"
    }
}

impl RouterPolicy for PrefixAffinity {
    fn choose(&self, views: &[ReplicaView]) -> (ReplicaId, DpRouteReason) {
        if let Some(v) = least_loaded(candidates(views).filter(|v| v.has_prefix)) {
            return (v.replica, DpRouteReason::PrefixAffinity);
        }
        let chosen = least_loaded(candidates(views))
            .expect("at least one candidate: all replicas are candidates when none is eligible");
        // No candidate holds the prefix, so any holder was skipped for pressure or its circuit.
        let reason = if views.iter().any(|v| v.has_prefix) {
            DpRouteReason::PressureAvoidance
        } else {
            DpRouteReason::LeastLoaded
        };
        (chosen.replica, reason)
    }
}
