//! The simulated executor: a per-token cost model standing in for the GPU.

use std::time::Duration;

use serde::Serialize;

use crate::scheduler::IterationPlan;

/// Virtual seconds per unit of work. An iteration costs `per_decode_step_s` when it decodes
/// at least one sequence, plus `per_prefill_token_s` per prefill token, plus `per_seq_s` per
/// sequence in the batch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CostModel {
    pub per_prefill_token_s: f64,
    pub per_decode_step_s: f64,
    pub per_seq_s: f64,
}

impl CostModel {
    pub fn iteration_seconds(&self, prefill_tokens: u32, decode_seqs: u32, seqs: u32) -> f64 {
        let decode = if decode_seqs > 0 {
            self.per_decode_step_s
        } else {
            0.0
        };
        decode
            + self.per_prefill_token_s * f64::from(prefill_tokens)
            + self.per_seq_s * f64::from(seqs)
    }
}

/// Executes plans in virtual time only.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SimExecutor {
    pub cost: CostModel,
}

impl SimExecutor {
    /// Virtual duration of `plan`.
    pub fn duration(&self, plan: &IterationPlan) -> Duration {
        let secs = self.cost.iteration_seconds(
            plan.prefill_tokens(),
            plan.decode_tokens(),
            plan.items.len() as u32,
        );
        Duration::from_secs_f64(secs.max(0.0))
    }
}
