//! `lru`: the benchmark baseline (TS §8 "never LRU alone" — selectable, never the default).

use turbine_core::registry::Module;

use super::{BlockScoreInputs, EvictionPolicy, PolicyWeights};
use crate::directory::Timestamp;

/// Recency only: the most recently accessed block is kept longest. Takes the default
/// [`EvictionPolicy::action`] (Phase 4: demote or drop, never compress), so the ladder is a
/// `cost_aware` behaviour.
#[derive(Clone, Copy, Debug)]
pub struct LruPolicy;

impl Module for LruPolicy {
    fn name(&self) -> &'static str {
        "lru"
    }
}

impl EvictionPolicy for LruPolicy {
    fn score(&self, b: &BlockScoreInputs, _w: &PolicyWeights, _now: Timestamp) -> f64 {
        b.block.last_access.as_secs_f64()
    }
}
