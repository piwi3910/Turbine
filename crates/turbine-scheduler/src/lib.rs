//! Turbine scheduler (contract §12): request lifecycle, the waiting queue, per-iteration batch
//! planning with chunked prefill and preemption by recompute, and a deterministic simulator.
//! GPU-free: it sees requests, token counts, block counts and a cost model only.

pub mod metrics;
pub mod policy;
pub mod queue;
pub mod request;
pub mod scheduler;
pub mod sim;

pub use metrics::SchedulerMetrics;
pub use queue::WaitingQueue;
pub use request::{CancelReason, PreemptReason, RequestState, SchedError, SchedRequest};
pub use scheduler::{
    BatchItem, BatchKind, ForkOp, IterationFailure, IterationLimits, IterationOutcome,
    IterationPlan, Scheduler, SchedulerParams, SchedulerSnapshot, SubmitError,
};
pub use turbine_core::types::{BlockId, Priority, RequestId, SeqId};

/// Phase 2m S-1 / S-13: every registry of this crate passes the shared conformance check.
#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::{Module, conformance};

    use crate::policy::{self, DefaultPolicy};

    /// Catches a duplicate or malformed policy name, an empty registry, or a `default` that
    /// is not the Phase 2 policy.
    #[test]
    fn policies() {
        conformance::check(policy::registry()).unwrap();
        let default = policy::registry()
            .get("default")
            .expect("default registered");
        assert_eq!(default.name(), DefaultPolicy.name());
        assert_eq!(policy::registry().point(), "scheduling_policy");
    }
}
