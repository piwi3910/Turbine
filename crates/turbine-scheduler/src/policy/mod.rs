//! Scheduling policies (Phase 2m S-9, contract §24 `scheduling_policy`): the choices the
//! scheduler's mechanism delegates — admission order of the waiting queue, the preemption
//! victim, and the prefill chunk size. The mechanism stays in [`crate::Scheduler`]: the stage
//! order of `plan()` (drop cancelled → decode → forks → continuing prefills → admit), block
//! accounting, the eligibility rules of preemption (a prefill or fork preempts only requests
//! ranked below its own, never one already in the plan) and every bound.
//!
//! A policy is one file in this directory plus one entry in [`registry`]; `scheduler.policy`
//! selects it, and the simulator runs every invariant test over every registered policy
//! (`sim::tests`). Policies are consulted per decision, so they must be cheap and
//! deterministic: no clock, no randomness, no state.

#[cfg(test)]
pub(crate) mod conformance;
mod default;

pub use default::DefaultPolicy;

use std::time::Duration;

use turbine_core::registry::{Module, Registry};
use turbine_core::types::{Priority, RequestId};

use crate::scheduler::{IterationLimits, SchedulerParams};

/// What a policy may use to order a waiting request.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionInfo {
    pub priority: Priority,
    /// Arrival time of the request (virtual time in the simulator).
    pub arrival: Duration,
    /// Submission number: the queue's counter when the request was submitted.
    pub submit_no: u64,
    /// The request was admitted before and is queued again after a preemption.
    pub preempted: bool,
    /// The queue's counter when this push happened (equals `submit_no` on submission).
    pub push_no: u64,
}

/// Admission order of the waiting queue: the smallest key is admitted first. The queue
/// compares whole keys, so a policy keeps them unique by putting a counter in `order`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct AdmissionKey {
    pub tier: u8,
    pub priority: Priority,
    pub arrival: Duration,
    pub order: u64,
}

/// What a policy may use to rank a running request for preemption.
#[derive(Clone, Copy, Debug)]
pub struct RunningInfo {
    pub id: RequestId,
    pub priority: Priority,
    /// Admission number of the current admission (`None` only for a request not running).
    pub admitted: Option<u64>,
}

/// Preemption rank: the larger rank is dropped first. The mechanism lets a prefill or fork
/// preempt only requests whose rank is larger than its own.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PreemptionRank(pub Priority, pub u64);

/// A scheduling policy (contract §24).
pub trait SchedulingPolicy: Module {
    /// The queue position of a waiting request.
    fn admission_key(&self, req: &AdmissionInfo) -> AdmissionKey;
    /// The rank of a running request; larger is preempted first.
    fn preemption_rank(&self, running: &RunningInfo) -> PreemptionRank;
    /// The victim among the eligible `candidates` (the mechanism already filtered them), in
    /// admission order; `None` preempts nothing.
    fn pick_victim(&self, candidates: &[(RequestId, PreemptionRank)]) -> Option<RequestId>;
    /// The largest prefill chunk of this iteration, at least 1. The mechanism checks it
    /// against nothing, so a policy keeps it within `prefill_chunk_tokens` (chunked prefill)
    /// or `max_batch_tokens` and within `limits.prefill_chunk_tokens`.
    fn chunk_cap(&self, params: &SchedulerParams, limits: &IterationLimits) -> u32;
}

static REGISTRY: Registry<dyn SchedulingPolicy> =
    Registry::new("scheduling_policy", &[&DefaultPolicy]);

/// Every registered scheduling policy (`scheduling_policy`).
pub fn registry() -> &'static Registry<dyn SchedulingPolicy> {
    &REGISTRY
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn id(n: u128) -> RequestId {
        RequestId(uuid::Uuid::from_u128(n))
    }

    fn waiting(priority: i32, arrival_s: u64, submit_no: u64) -> AdmissionInfo {
        AdmissionInfo {
            priority: Priority(priority),
            arrival: Duration::from_secs(arrival_s),
            submit_no,
            preempted: false,
            push_no: submit_no,
        }
    }

    fn preempted(priority: i32, submit_no: u64, push_no: u64) -> AdmissionInfo {
        AdmissionInfo {
            preempted: true,
            push_no,
            ..waiting(priority, 0, submit_no)
        }
    }

    fn params(chunked_prefill: bool) -> SchedulerParams {
        SchedulerParams {
            max_running_requests: 8,
            max_batch_tokens: 512,
            prefill_chunk_tokens: 128,
            max_queued_requests: 16,
            chunked_prefill,
            block_tokens: 16,
            free_watermark: 0.01,
            max_seq_len: 4096,
            queue_timeout: Duration::from_secs(60),
        }
    }

    /// The Phase 2 `Scheduler::chunk_cap` body, as it was on main.
    fn main_chunk_cap(p: &SchedulerParams, limits: &IterationLimits) -> u32 {
        let cap = if p.chunked_prefill {
            p.prefill_chunk_tokens
        } else {
            p.max_batch_tokens
        };
        limits
            .prefill_chunk_tokens
            .map_or(cap, |c| c.min(cap))
            .max(1)
    }

    /// Catches a default policy that orders admissions or victims differently from main:
    /// priority, then arrival, then submission; preempted requests first, the most recently
    /// preempted at the head; the worst (largest priority value, then latest admission) victim.
    #[test]
    fn default_policy_orders_like_main() {
        let p = &DefaultPolicy;
        assert_eq!(p.name(), "default");

        // Priority −1 before 0 before 1, regardless of arrival.
        let mut keys = [
            (p.admission_key(&waiting(1, 0, 0)), "p1"),
            (p.admission_key(&waiting(0, 5, 1)), "p0"),
            (p.admission_key(&waiting(-1, 9, 2)), "p-1"),
        ];
        keys.sort();
        assert_eq!(keys.map(|k| k.1), ["p-1", "p0", "p1"]);
        // Equal priority: arrival, then submission order.
        assert!(p.admission_key(&waiting(0, 1, 7)) < p.admission_key(&waiting(0, 2, 3)));
        assert!(p.admission_key(&waiting(0, 1, 3)) < p.admission_key(&waiting(0, 1, 4)));
        // Every preempted key before every new key, whatever the priorities.
        let new_best = p.admission_key(&waiting(-100, 0, 0));
        let preempted_worst = p.admission_key(&preempted(100, 50, 60));
        assert!(preempted_worst < new_best);
        // Two preempted requests: the later push comes first.
        assert!(
            p.admission_key(&preempted(0, 1, 20)) < p.admission_key(&preempted(0, 2, 10)),
            "most recently pushed first"
        );

        // Victims: the largest priority value first, then the most recent admission.
        let info = |n: u128, priority: i32, admitted: u64| RunningInfo {
            id: id(n),
            priority: Priority(priority),
            admitted: Some(admitted),
        };
        let candidates: Vec<(RequestId, PreemptionRank)> =
            [info(1, 0, 0), info(2, 1, 1), info(3, 1, 2), info(4, 0, 3)]
                .iter()
                .map(|r| (r.id, p.preemption_rank(r)))
                .collect();
        assert_eq!(p.pick_victim(&candidates), Some(id(3)));
        assert_eq!(p.pick_victim(&candidates[..2]), Some(id(2)));
        assert_eq!(
            p.pick_victim(&[candidates[0], candidates[3]]),
            Some(id(4)),
            "equal priority: the latest admission"
        );
        assert_eq!(p.pick_victim(&[]), None);
        let unadmitted = RunningInfo {
            admitted: None,
            ..info(5, 0, 0)
        };
        assert_eq!(
            p.preemption_rank(&unadmitted),
            PreemptionRank(Priority(0), u64::MAX)
        );

        // Chunk cap: the old formula, chunked prefill on and off, with and without a limit.
        for chunked in [true, false] {
            for limit in [None, Some(64), Some(0), Some(100_000)] {
                let limits = IterationLimits {
                    prefill_chunk_tokens: limit,
                    ..IterationLimits::default()
                };
                let params = params(chunked);
                assert_eq!(
                    p.chunk_cap(&params, &limits),
                    main_chunk_cap(&params, &limits),
                    "chunked {chunked}, limit {limit:?}"
                );
            }
        }
        assert_eq!(
            p.chunk_cap(
                &params(true),
                &IterationLimits {
                    prefill_chunk_tokens: Some(64),
                    ..IterationLimits::default()
                }
            ),
            64
        );
    }
}
