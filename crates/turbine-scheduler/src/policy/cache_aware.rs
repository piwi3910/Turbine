//! `cache_aware` (Phase 5p S-7): cache-affinity admission with a windowed starvation bound.
//! Waiting requests order by arrival window, then by cached-prefix length (the longest
//! attached prefix first), then by submission order: within one window the requests whose KV
//! is already resident go first, while every earlier window empties before a later one. The
//! bound: a waiting request is never admitted after an equal-priority, never-preempted
//! request that arrived at least one `window` later (the simulator's `admission_inversions`
//! judges it over the real scheduler). Preempted requests, the victim and the chunk cap are
//! [`DefaultPolicy`]'s. Stateless and clock-free like every policy: the window arrives on the
//! `AdmissionInfo`.

use std::time::Duration;

use turbine_core::registry::Module;
use turbine_core::types::RequestId;

use super::{
    AdmissionInfo, AdmissionKey, DefaultPolicy, PreemptionRank, RunningInfo, SchedulingPolicy,
};
use crate::scheduler::{IterationLimits, SchedulerParams};

/// The cache-aware policy (`scheduler.policy: cache_aware`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheAwarePolicy;

impl Module for CacheAwarePolicy {
    fn name(&self) -> &'static str {
        "cache_aware"
    }
}

impl SchedulingPolicy for CacheAwarePolicy {
    fn admission_key(&self, req: &AdmissionInfo) -> AdmissionKey {
        if req.preempted {
            return DefaultPolicy.admission_key(req);
        }
        // Bucket the arrival into its window; a zero window degenerates to identity bucketing
        // rather than dividing by zero.
        let window = req.window.max(Duration::from_nanos(1));
        let window_nanos = window.as_nanos();
        let bucket = (req.arrival.as_nanos() / window_nanos) * window_nanos;
        AdmissionKey {
            tier: 1,
            priority: req.priority,
            arrival: Duration::from_nanos(u64::try_from(bucket).unwrap_or(u64::MAX)),
            affinity: u64::from(u32::MAX - req.cached_prefix_tokens),
            order: req.submit_no,
        }
    }

    fn preemption_rank(&self, running: &RunningInfo) -> PreemptionRank {
        DefaultPolicy.preemption_rank(running)
    }

    fn pick_victim(&self, candidates: &[(RequestId, PreemptionRank)]) -> Option<RequestId> {
        DefaultPolicy.pick_victim(candidates)
    }

    fn chunk_cap(&self, params: &SchedulerParams, limits: &IterationLimits) -> u32 {
        DefaultPolicy.chunk_cap(params, limits)
    }
}
