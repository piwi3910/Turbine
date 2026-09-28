//! `least_loaded`: the candidate with the fewest outstanding tokens, ties to the lowest index;
//! prefixes are ignored.

use turbine_core::registry::Module;
use turbine_core::types::ReplicaId;

use super::{DpRouteReason, ReplicaView, RouterPolicy, candidates, least_loaded};

/// The `least_loaded` module of the `dp_router_policy` registry.
pub struct LeastLoaded;

impl Module for LeastLoaded {
    fn name(&self) -> &'static str {
        "least_loaded"
    }
}

impl RouterPolicy for LeastLoaded {
    fn choose(&self, views: &[ReplicaView]) -> (ReplicaId, DpRouteReason) {
        let chosen = least_loaded(candidates(views))
            .expect("at least one candidate: all replicas are candidates when none is eligible");
        (chosen.replica, DpRouteReason::LeastLoaded)
    }
}
