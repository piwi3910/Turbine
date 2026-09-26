//! `default`: the Phase 2 scheduling policy, unchanged. Admission by priority (lower value
//! first), then arrival, then submission order, with preempted requests ahead of every
//! never-admitted one (the most recently preempted first); the victim is the worst-ranked
//! running request (largest priority value, then the most recent admission); prefill chunks
//! of `prefill_chunk_tokens` (the whole batch budget without chunked prefill), capped by the
//! iteration's throttle.

use std::time::Duration;

use turbine_core::registry::Module;
use turbine_core::types::{Priority, RequestId};

use super::{AdmissionInfo, AdmissionKey, PreemptionRank, RunningInfo, SchedulingPolicy};
use crate::scheduler::{IterationLimits, SchedulerParams};

/// The Phase 2 policy (`scheduler.policy: default`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultPolicy;

impl Module for DefaultPolicy {
    fn name(&self) -> &'static str {
        "default"
    }
}

impl SchedulingPolicy for DefaultPolicy {
    fn admission_key(&self, req: &AdmissionInfo) -> AdmissionKey {
        if req.preempted {
            AdmissionKey {
                tier: 0,
                priority: Priority(0),
                arrival: Duration::ZERO,
                order: u64::MAX - req.push_no,
            }
        } else {
            AdmissionKey {
                tier: 1,
                priority: req.priority,
                arrival: req.arrival,
                order: req.submit_no,
            }
        }
    }

    fn preemption_rank(&self, running: &RunningInfo) -> PreemptionRank {
        PreemptionRank(running.priority, running.admitted.unwrap_or(u64::MAX))
    }

    fn pick_victim(&self, candidates: &[(RequestId, PreemptionRank)]) -> Option<RequestId> {
        candidates
            .iter()
            .max_by_key(|(_, rank)| *rank)
            .map(|(id, _)| *id)
    }

    fn chunk_cap(&self, params: &SchedulerParams, limits: &IterationLimits) -> u32 {
        let cap = if params.chunked_prefill {
            params.prefill_chunk_tokens
        } else {
            params.max_batch_tokens
        };
        limits
            .prefill_chunk_tokens
            .map_or(cap, |c| c.min(cap))
            .max(1)
    }
}
