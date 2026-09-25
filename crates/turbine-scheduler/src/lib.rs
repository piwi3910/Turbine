//! Turbine scheduler (contract §12): request lifecycle, the waiting queue, per-iteration batch
//! planning with chunked prefill and preemption by recompute, and a deterministic simulator.
//! GPU-free: it sees requests, token counts, block counts and a cost model only.

pub mod metrics;
pub mod queue;
pub mod request;
pub mod scheduler;

pub use metrics::SchedulerMetrics;
pub use queue::WaitingQueue;
pub use request::{CancelReason, PreemptReason, RequestState, SchedError, SchedRequest};
pub use scheduler::{
    BatchItem, BatchKind, ForkOp, IterationFailure, IterationLimits, IterationOutcome,
    IterationPlan, Scheduler, SchedulerParams, SchedulerSnapshot, SubmitError,
};
pub use turbine_core::types::{BlockId, Priority, RequestId, SeqId};
