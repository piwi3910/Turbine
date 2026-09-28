//! Backend-neutral collectives (P5 S-2, contract §15.1): the [`Collective`] trait every backend
//! implements, its error type and the `turbine_collective_*` metrics.
//!
//! Buffers are phase-1 [`DeviceSlice`]s and ordering is the phase-1 [`StreamRef`]; a backend
//! never allocates model memory (`hostmem` allocates only its own mapped exchange region).
//! Supported element types: BF16 and FP32.

pub mod conformance;
// The only module of this crate allowed to contain `unsafe` (contract §1.3, CONFLICT C-19).
#[allow(unsafe_code)]
pub mod ffi;
pub mod host;
pub mod hostmem;
pub mod nccl_api;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_core::clock::Clock;
use turbine_core::config::ParallelConfig;
use turbine_core::registry::{Module, Registry};
use turbine_core::types::Vendor;
use turbine_observability::MetricsRegistry;
use turbine_tensor::{DType, DeviceMemory, DeviceSlice, StreamRef};

pub use ffi::NcclApi;
pub use host::{HostBackend, HostCollective};
pub use hostmem::{HostmemBackend, HostmemCollective};
pub use nccl_api::{NcclApiBackend, NcclFlavor};

/// Element-wise reduction of all-reduce and reduce-scatter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
}

/// One communicator rank. Every rank of a group must issue the same operations in the same
/// order with matching sizes; no call blocks longer than the backend's op timeout.
pub trait Collective: Send + Sync {
    /// The registered backend (extension point `collective_backend`) this communicator runs on;
    /// also the `backend` metric label.
    fn backend(&self) -> &'static str;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    /// Reduces `buf` (elements of `dtype`, BF16 or FP32) across ranks in place.
    fn all_reduce(
        &self,
        buf: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError>;
    /// `recv` (world × `send.len()` bytes) receives every rank's `send`, in rank order (byte-wise:
    /// any element type).
    fn all_gather(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError>;
    /// Reduces `send` (world × `recv.len()` bytes of `dtype`) across ranks; rank r receives
    /// chunk r.
    fn reduce_scatter(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError>;
    /// Every rank's `buf` receives rank `root`'s (byte-wise).
    fn broadcast(
        &self,
        buf: &mut DeviceSlice,
        root: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError>;
    fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError>;
    /// Point-to-point (pipeline stages hand activations on): `buf` (byte-wise) goes to rank
    /// `peer`, whose matching [`Self::recv`] receives it. The two ranks of a pair issue their
    /// sends and receives to each other in the same order and with the same lengths; a send may
    /// wait (on the device or the host, bounded by the op timeout) until the peer receives, so a
    /// schedule where both ranks send first may time out.
    fn send(
        &self,
        buf: &DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError>;
    /// Receives into `buf` what rank `peer`'s matching [`Self::send`] sent (same length).
    fn recv(
        &self,
        buf: &mut DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError>;
    /// Starts bounding one step: every collective enqueued until [`Self::step_end`] must
    /// complete on the device within the op timeout, or the communicator is aborted (a peer
    /// that never arrives otherwise leaves a device kernel waiting forever). Default: nothing
    /// (backends whose calls complete before they return).
    fn step_begin(&self) {}
    /// The step's work has completed (the caller synchronised its stream); `Err` when the
    /// communicator was aborted meanwhile.
    fn step_end(&self) -> Result<(), CollectiveError> {
        Ok(())
    }
    /// Aborts the communicator: every pending and later call on any rank fails.
    fn abort(&self);
}

/// A collective backend as a registered module (Phase 2m convention, extension point
/// `collective_backend`, contract §24): what it can reduce over and how its library loads.
/// Adding a backend is one file plus one registry entry; nothing matches on backend names.
pub trait CollectiveBackend: Module {
    /// The device vendors whose memory its communicators reduce over; empty for a host-memory
    /// reference backend (which serves only plans without tensor parallelism on GPUs).
    fn vendors(&self) -> &'static [Vendor];
    /// The operator-configured library path in `cfg` (`parallel.rccl_library`, …): when set,
    /// a failure to load it is fatal (exit 1). `None` for a backend without a library.
    fn configured_library<'a>(&self, cfg: &'a ParallelConfig) -> Option<&'a Path> {
        let _ = cfg;
        None
    }
    /// Loads the backend: `explicit` is the configured library, else the default search.
    fn load(&self, explicit: Option<&Path>) -> Result<Arc<dyn CollectiveLibrary>, CollectiveError>;
    /// True for a backend whose ranks must be threads of one process (it exchanges through
    /// memory of that process, e.g. `hostmem`): the planner refuses it in `static` rank mode and
    /// `auto` skips it there. Default: false.
    fn one_process_only(&self) -> bool {
        false
    }
}

/// A loaded backend: makes the group's unique id (on one rank) and opens each rank's
/// communicator from it.
pub trait CollectiveLibrary: Send + Sync {
    /// The registered name (the `backend` label).
    fn backend(&self) -> &'static str;
    /// The library's version (`2.30.4`), `None` for a backend without one.
    fn version(&self) -> Option<String>;
    /// A fresh id for one communicator group; every rank opens with the same id.
    fn unique_id(&self) -> Result<[u8; UNIQUE_ID_BYTES], CollectiveError>;
    /// Opens rank `init.rank` of the group `init.unique_id`, bounded by `init.init_timeout`
    /// (on expiry the communicator is aborted and `Timeout { op: "comm_init" }` returned).
    fn open(self: Arc<Self>, init: CollectiveInit) -> Result<Arc<dyn Collective>, CollectiveError>;
}

/// Bytes of a group id (`ncclUniqueId`).
pub const UNIQUE_ID_BYTES: usize = 128;

/// What [`CollectiveLibrary::open`] needs for one rank.
pub struct CollectiveInit {
    pub rank: usize,
    pub world: usize,
    pub unique_id: [u8; UNIQUE_ID_BYTES],
    pub init_timeout: Duration,
    pub op_timeout: Duration,
    pub clock: Arc<dyn Clock>,
    /// `turbine_collective_*`; `None` records nothing (tests).
    pub metrics: Option<CollectiveMetrics>,
    /// The rank's device memory (its kernel-library context), for a backend that allocates its
    /// communication buffers and runs its kernels through the kernel library (`hostmem`);
    /// `None` where the rank has none (the host backend) or the backend needs none.
    pub memory: Option<Arc<dyn DeviceMemory>>,
    /// `parallel.collective.hostmem_max_bytes` for a backend that routes large messages to a
    /// delegate (`hostmem` → `rccl`): the largest message (nccl-tests bytes) it keeps; `None`
    /// is `auto`, the backend's measured per-op default; `Some(u64::MAX)` keeps every call on
    /// the backend and opens no delegate. Other backends ignore it.
    pub route_max_bytes: Option<u64>,
}

static HOST: HostBackend = HostBackend;
/// Also `hostmem`'s delegate for large messages.
pub(crate) static RCCL: NcclApiBackend = NcclApiBackend::new(&nccl_api::RCCL_FLAVOR);
static NCCL: NcclApiBackend = NcclApiBackend::new(&nccl_api::NCCL_FLAVOR);
static HOSTMEM: HostmemBackend = HostmemBackend;

// `hostmem` comes before `rccl`: `auto` picks it for AMD plans in `local` rank mode (it
// routes large messages to RCCL itself, and hands the whole communicator to RCCL on a kernel
// library without ABI v2.7); in `static` mode `auto` skips it (`one_process_only`) and takes
// `rccl`.
static COLLECTIVE_BACKENDS: Registry<dyn CollectiveBackend> =
    Registry::new("collective_backend", &[&HOST, &HOSTMEM, &RCCL, &NCCL]);

/// The registered collective backends, in registration order (`auto` takes the first one
/// serving the plan's vendor).
pub fn registry() -> &'static Registry<dyn CollectiveBackend> {
    &COLLECTIVE_BACKENDS
}

#[derive(Debug, thiserror::Error)]
pub enum CollectiveError {
    #[error("{op} timed out after {after:?}")]
    Timeout { op: &'static str, after: Duration },
    #[error("rank {rank} aborted")]
    RemoteAbort { rank: usize },
    #[error("backend error {code}: {message}")]
    Backend { code: i32, message: String },
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("{library} unavailable: {detail}")]
    Unavailable { library: String, detail: String },
}

impl CollectiveError {
    /// The `kind` label of `turbine_collective_errors_total`, for errors raised by a running
    /// communicator (`None` for shape and availability errors).
    pub fn kind(&self) -> Option<CollectiveErrorKind> {
        match self {
            CollectiveError::Timeout { .. } => Some(CollectiveErrorKind::Timeout),
            CollectiveError::RemoteAbort { .. } => Some(CollectiveErrorKind::RemoteAbort),
            CollectiveError::Backend { .. } => Some(CollectiveErrorKind::Backend),
            CollectiveError::ShapeMismatch | CollectiveError::Unavailable { .. } => None,
        }
    }
}

/// `op` label values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CollectiveOp {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    Barrier,
    Send,
    Recv,
}

impl CollectiveOp {
    pub fn as_str(self) -> &'static str {
        match self {
            CollectiveOp::AllReduce => "all_reduce",
            CollectiveOp::AllGather => "all_gather",
            CollectiveOp::ReduceScatter => "reduce_scatter",
            CollectiveOp::Broadcast => "broadcast",
            CollectiveOp::Barrier => "barrier",
            CollectiveOp::Send => "send",
            CollectiveOp::Recv => "recv",
        }
    }
}

/// `kind` label values of `turbine_collective_errors_total`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CollectiveErrorKind {
    Timeout,
    RemoteAbort,
    Backend,
}

impl CollectiveErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CollectiveErrorKind::Timeout => "timeout",
            CollectiveErrorKind::RemoteAbort => "remote_abort",
            CollectiveErrorKind::Backend => "backend",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct OpLabels {
    op: &'static str,
    backend: &'static str,
}

/// Why a routing backend sent one call where it did (`reason` of
/// `turbine_collective_route_total`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RouteReason {
    /// The message is at most the threshold: the backend's own path.
    BelowThreshold,
    /// Above the threshold: the delegate.
    AboveThreshold,
    /// The backend's own path cannot run the call: the delegate.
    OpUnsupported,
    /// Above the threshold, but the delegate cannot run the call between these devices
    /// (point-to-point without a peer path): the backend's own path.
    NoPeerAccess,
    /// At least the backend's copy-engine threshold: its copy-engine path (`hostmem`).
    CopyEngine,
}

impl RouteReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RouteReason::BelowThreshold => "below_threshold",
            RouteReason::AboveThreshold => "above_threshold",
            RouteReason::OpUnsupported => "op_unsupported",
            RouteReason::NoPeerAccess => "no_peer_access",
            RouteReason::CopyEngine => "copy_engine",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RouteLabels {
    op: &'static str,
    backend: &'static str,
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ErrorLabels {
    backend: &'static str,
    kind: &'static str,
}

/// `turbine_collective_duration_seconds{op,backend}`, `turbine_collective_bytes_total{op,backend}`
/// and `turbine_collective_errors_total{backend,kind}`; every label comes from a closed enum.
#[derive(Clone)]
pub struct CollectiveMetrics {
    duration: Family<OpLabels, Histogram>,
    bytes: Family<OpLabels, Counter>,
    errors: Family<ErrorLabels, Counter>,
    route: Family<RouteLabels, Counter>,
}

impl CollectiveMetrics {
    pub fn register(reg: &MetricsRegistry) -> Self {
        let duration = reg.register(
            "turbine_collective_duration_seconds",
            "Wall time of one collective operation",
            Family::<OpLabels, Histogram>::new_with_constructor(|| {
                // 10 µs … ~42 s
                Histogram::new(exponential_buckets(1e-5, 4.0, 12))
            }),
        );
        let bytes = reg.register(
            "turbine_collective_bytes",
            "Bytes contributed by this rank to collective operations",
            Family::<OpLabels, Counter>::default(),
        );
        let errors = reg.register(
            "turbine_collective_errors",
            "Collective operations that failed, by kind",
            Family::<ErrorLabels, Counter>::default(),
        );
        let route = reg.register(
            "turbine_collective_route",
            "Calls a routing collective backend sent to itself or to its delegate, by reason",
            Family::<RouteLabels, Counter>::default(),
        );
        CollectiveMetrics {
            duration,
            bytes,
            errors,
            route,
        }
    }

    /// One call of `op` sent to `backend` (the routing backend or its delegate) for `reason`.
    pub fn route(&self, op: CollectiveOp, backend: &'static str, reason: RouteReason) {
        self.route
            .get_or_create(&RouteLabels {
                op: op.as_str(),
                backend,
                reason: reason.as_str(),
            })
            .inc();
    }

    pub fn observe(&self, op: CollectiveOp, backend: &'static str, bytes: u64, seconds: f64) {
        let labels = OpLabels {
            op: op.as_str(),
            backend,
        };
        self.duration.get_or_create(&labels).observe(seconds);
        self.bytes.get_or_create(&labels).inc_by(bytes);
    }

    pub fn error(&self, backend: &'static str, kind: CollectiveErrorKind) {
        self.errors
            .get_or_create(&ErrorLabels {
                backend,
                kind: kind.as_str(),
            })
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_render_with_closed_labels() {
        let reg = MetricsRegistry::new();
        let m = CollectiveMetrics::register(&reg);
        m.observe(CollectiveOp::AllReduce, "rccl", 4096, 0.002);
        m.error("rccl", CollectiveErrorKind::Timeout);
        m.route(
            CollectiveOp::AllReduce,
            "hostmem",
            RouteReason::BelowThreshold,
        );
        let text = reg.render().expect("renders");
        assert!(
            text.contains(
                "turbine_collective_bytes_total{op=\"all_reduce\",backend=\"rccl\"} 4096"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "turbine_collective_duration_seconds_count{op=\"all_reduce\",backend=\"rccl\"} 1"
            ),
            "{text}"
        );
        assert!(
            text.contains("turbine_collective_errors_total{backend=\"rccl\",kind=\"timeout\"} 1"),
            "{text}"
        );
        assert!(
            text.contains(
                "turbine_collective_route_total{op=\"all_reduce\",backend=\"hostmem\",reason=\"below_threshold\"} 1"
            ),
            "{text}"
        );
        let timeout = CollectiveError::Timeout {
            op: "all_reduce",
            after: Duration::from_millis(500),
        };
        assert_eq!(timeout.kind(), Some(CollectiveErrorKind::Timeout));
        assert_eq!(CollectiveError::ShapeMismatch.kind(), None);
    }
}
