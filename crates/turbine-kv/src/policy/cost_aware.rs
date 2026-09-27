//! `cost_aware` (the default, P4 §Cost-aware value terms): value = reuse × recompute × priority
//! × retrieval / memory.

use turbine_core::registry::Module;

use super::{BlockScoreInputs, EvictionPolicy, PolicyWeights, recompute_seconds};
use crate::directory::{Timestamp, decay};

/// Retrieval costs below this count as this (a copy is never free).
const MIN_RETRIEVAL_SECONDS: f64 = 1e-6;

/// The default policy (`kv.policy: cost_aware`), weighted by `kv.policy_weights`.
#[derive(Clone, Copy, Debug)]
pub struct CostAwarePolicy;

impl CostAwarePolicy {
    /// min(1, session_active·[hot] + h/(1+h) + 0.25·c/(1+c)): h = hits decayed to `now` with
    /// the `hit_half_life` weight, c = child count (prefix popularity).
    pub fn reuse_probability(
        &self,
        b: &BlockScoreInputs,
        w: &PolicyWeights,
        now: Timestamp,
    ) -> f64 {
        let h = decay(
            b.block.decayed_hits,
            b.block.decayed_at,
            now,
            w.hit_half_life,
        );
        let c = f64::from(b.block.child_count);
        let session = if b.session_hot { w.session_active } else { 0.0 };
        (session + h / (1.0 + h) + 0.25 * c / (1.0 + c)).min(1.0)
    }
}

impl Module for CostAwarePolicy {
    fn name(&self) -> &'static str {
        "cost_aware"
    }
}

impl EvictionPolicy for CostAwarePolicy {
    /// value = reuse × recompute × priority × retrieval / memory, where
    /// memory = size / tier capacity × (1 + pressure level 0..4).
    ///
    /// Retrieval multiplies (user decision 2026-09-25, amending TS §8 which divides by it):
    /// a block that is cheap to bring back loses little by leaving, so it is evicted first.
    fn score(&self, b: &BlockScoreInputs, w: &PolicyWeights, now: Timestamp) -> f64 {
        let reuse = self.reuse_probability(b, w, now);
        let recompute = recompute_seconds(b.block.tokens, b.depth_tokens, b.prefill_tps);
        let retrieval = b.retrieval.seconds.max(MIN_RETRIEVAL_SECONDS);
        let memory = b.block.size_bytes.max(1) as f64 / b.tier_capacity.max(1) as f64
            * (1.0 + f64::from(b.tier_pressure.as_u8()));
        reuse * recompute * f64::from(b.block.priority.0) * retrieval / memory
    }
}
