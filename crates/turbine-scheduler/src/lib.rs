//! Turbine scheduler (contract §12): request lifecycle, the waiting queue, per-iteration batch
//! planning with chunked prefill and preemption by recompute, and a deterministic simulator.
//! GPU-free: it sees requests, token counts, block counts and a cost model only.

pub mod gate;
pub mod metrics;
pub mod pipeline;
pub mod policy;
pub mod queue;
pub mod request;
pub mod scheduler;
pub mod sim;

pub use gate::{AdmissionGate, GateOutcome};
pub use metrics::SchedulerMetrics;
pub use pipeline::{MicroBatchPlan, PipelineMetrics, StageTimeline};
pub use queue::WaitingQueue;
pub use request::{CancelReason, PreemptReason, RequestState, SchedError, SchedRequest};
pub use scheduler::{
    BatchItem, BatchKind, ForkOp, IterationFailure, IterationLimits, IterationOutcome,
    IterationPlan, PipelineSnapshot, Scheduler, SchedulerParams, SchedulerSnapshot, StageSnapshot,
    SubmitError,
};
pub use turbine_core::types::{BlockId, Priority, RequestId, SeqId};
/// The reason carried by `SubmitError::Rejected`.
pub use turbine_reliability::admission::RejectionReason;

/// Phase 2m S-1 / S-13: every registry of this crate passes the shared conformance check and
/// every registered module its extension point's suite, run over the registry itself.
#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::Module;

    use crate::policy::conformance::policies_suite;
    use crate::policy::{self, DefaultPolicy};

    /// Every policy keeps the simulator invariants (`policies_suite`: decode never starved,
    /// chunk budget respected, preemption by recompute); `default` is the Phase 2 policy.
    /// Catches a duplicate or malformed policy name, an empty registry, or a policy that
    /// breaks an invariant.
    #[test]
    fn policies() {
        let reg = policy::registry();
        assert_eq!(reg.point(), "scheduling_policy");
        let default = reg.get("default").expect("default registered");
        assert_eq!(default.name(), DefaultPolicy.name());
        if let Err(failures) = policies_suite(reg) {
            panic!("policy conformance failures:\n{}", failures.join("\n"));
        }
    }
}
