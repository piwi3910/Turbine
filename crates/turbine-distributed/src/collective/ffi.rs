//! One runtime-loaded binding to the NCCL C API (P5 S-2). RCCL implements the same API, so the
//! same table loads `librccl.so.1` (backend `rccl`) or `libnccl.so.2` (backend `nccl`); nothing
//! links either library at build time.
//!
//! This is the only module of `turbine-distributed` allowed to contain `unsafe` (contract §1.3).
//! Ownership rules: [`NcclApi`] owns the loaded library and never unloads it (RCCL and NCCL
//! start helper threads that outlive their communicators), so every function pointer copied
//! out of it stays valid for the life of the process. Communicators, device buffers and streams
//! are never created here; callers pass the phase-1 handles they own.
//!
//! Communicator init runs on a helper thread (`turbine-collective-init-<rank>`) that the opening
//! rank gives up after `init_timeout` + [`INIT_GRACE`]: the init watchdog aborts a communicator
//! whose peers never arrive, but on RCCL 2.30.4 the init or abort call itself may not return
//! (measured on novanas: a lone rank 0 came back only through this bound). A thread given up
//! that way is leaked together with whatever RCCL holds for it (its half-initialised
//! communicator, its sockets); nothing of it is reachable from Turbine afterwards. That is
//! acceptable because such a failure ends the process: at startup the open fails and the server
//! exits, and at runtime a communicator is only re-created on the circuit path, which restarts
//! the process on a fatal collective failure.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use libloading::Library;
use turbine_core::clock::Clock;
use turbine_tensor::{DType, DeviceBuffer, DeviceSlice, StreamRef};

use super::nccl_api::{FLAVORS, NcclFlavor};
use super::{
    Collective, CollectiveError, CollectiveInit, CollectiveLibrary, CollectiveMetrics,
    CollectiveOp, ReduceOp, UNIQUE_ID_BYTES,
};

/// The 15 symbols the binding resolves (contract §15.1, plus `ncclSend` / `ncclRecv` for
/// point-to-point); a library missing any is refused.
pub(crate) const SYMBOLS: [&str; 15] = [
    "ncclGetVersion",
    "ncclGetUniqueId",
    "ncclCommInitRankConfig",
    "ncclCommGetAsyncError",
    "ncclCommAbort",
    "ncclCommDestroy",
    "ncclAllReduce",
    "ncclAllGather",
    "ncclReduceScatter",
    "ncclBroadcast",
    "ncclSend",
    "ncclRecv",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclGetErrorString",
];

/// `ncclResult_t`.
type NcclResult = c_int;
/// `ncclComm_t` (opaque).
type NcclComm = *mut c_void;
/// `cudaStream_t` / `hipStream_t` (opaque), from `StreamRef::native_handle`.
type NativeStream = *mut c_void;

/// `ncclUniqueId`: 128 opaque bytes, passed by value to `ncclCommInitRankConfig`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NcclUniqueId {
    pub(crate) internal: [c_char; UNIQUE_ID_BYTES],
}

type GetVersionFn = unsafe extern "C" fn(version: *mut c_int) -> NcclResult;
type GetUniqueIdFn = unsafe extern "C" fn(id: *mut NcclUniqueId) -> NcclResult;
/// `config` is an `ncclConfig_t*` built by the communicator (P5 Task 7).
type CommInitRankConfigFn = unsafe extern "C" fn(
    comm: *mut NcclComm,
    nranks: c_int,
    id: NcclUniqueId,
    rank: c_int,
    config: *mut c_void,
) -> NcclResult;
type CommGetAsyncErrorFn =
    unsafe extern "C" fn(comm: NcclComm, async_error: *mut NcclResult) -> NcclResult;
type CommFn = unsafe extern "C" fn(comm: NcclComm) -> NcclResult;
type AllReduceFn = unsafe extern "C" fn(
    send: *const c_void,
    recv: *mut c_void,
    count: usize,
    datatype: c_int,
    op: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
type AllGatherFn = unsafe extern "C" fn(
    send: *const c_void,
    recv: *mut c_void,
    sendcount: usize,
    datatype: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
type BroadcastFn = unsafe extern "C" fn(
    send: *const c_void,
    recv: *mut c_void,
    count: usize,
    datatype: c_int,
    root: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
/// `ncclSend` and `ncclRecv` (the buffer is `const void*` for send, `void*` for recv; the same
/// ABI).
type P2pFn = unsafe extern "C" fn(
    buf: *mut c_void,
    count: usize,
    datatype: c_int,
    peer: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
type GroupFn = unsafe extern "C" fn() -> NcclResult;
type GetErrorStringFn = unsafe extern "C" fn(result: NcclResult) -> *const c_char;

/// The resolved entry points (all 15 are required so a library is refused as a whole).
#[expect(
    dead_code,
    reason = "ncclGroupStart/ncclGroupEnd are resolved but no grouped call is issued yet"
)]
pub(crate) struct NcclFns {
    pub(crate) get_version: GetVersionFn,
    pub(crate) get_unique_id: GetUniqueIdFn,
    pub(crate) comm_init_rank_config: CommInitRankConfigFn,
    pub(crate) comm_get_async_error: CommGetAsyncErrorFn,
    pub(crate) comm_abort: CommFn,
    pub(crate) comm_destroy: CommFn,
    pub(crate) all_reduce: AllReduceFn,
    pub(crate) all_gather: AllGatherFn,
    /// Same signature as all-reduce, with `count` = elements received per rank.
    pub(crate) reduce_scatter: AllReduceFn,
    pub(crate) broadcast: BroadcastFn,
    pub(crate) send: P2pFn,
    pub(crate) recv: P2pFn,
    pub(crate) group_start: GroupFn,
    pub(crate) group_end: GroupFn,
    pub(crate) get_error_string: GetErrorStringFn,
}

/// One loaded NCCL-API library and its binding table.
pub struct NcclApi {
    path: PathBuf,
    flavor: &'static NcclFlavor,
    version: i32,
    pub(crate) fns: NcclFns,
    /// Never unloaded; see the module ownership rules.
    _lib: ManuallyDrop<Library>,
}

impl std::fmt::Debug for NcclApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclApi")
            .field("path", &self.path)
            .field("backend", &self.flavor.name)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl NcclApi {
    /// Loads the library of `flavor`: `explicit` when given (a failure is fatal and names the
    /// path), else the flavor's default locations in order (RCCL under the ROCm tree, then the
    /// loader path; NCCL from the loader path).
    pub fn load(
        flavor: &'static NcclFlavor,
        explicit: Option<&Path>,
    ) -> Result<Arc<NcclApi>, CollectiveError> {
        apply_env_defaults(flavor);
        if let Some(path) = explicit {
            return NcclApi::load_from(path, flavor);
        }
        let mut tried = Vec::new();
        let rooted = flavor.root_env.and_then(|(var, file)| {
            let root = std::env::var_os(var).filter(|v| !v.is_empty())?;
            Some(PathBuf::from(root).join("lib").join(file))
        });
        let candidates = rooted
            .into_iter()
            .chain(flavor.defaults.iter().map(PathBuf::from));
        for candidate in candidates {
            match open(&candidate) {
                // A library that loads but fails the binding checks is reported as such,
                // not skipped in favour of the next candidate.
                Ok(lib) => return bind(lib, &candidate, flavor),
                Err(e) => tried.push(e),
            }
        }
        let library = flavor
            .defaults
            .last()
            .map_or_else(|| flavor.name.to_string(), |d| d.to_string());
        Err(CollectiveError::Unavailable {
            library,
            detail: tried.join("; "),
        })
    }

    /// Loads exactly `path`. A file named for another flavor (`libnccl*` asked as `rccl`) is
    /// refused; other names are taken as `flavor`.
    pub fn load_from(
        path: &Path,
        flavor: &'static NcclFlavor,
    ) -> Result<Arc<NcclApi>, CollectiveError> {
        let lib = open(path).map_err(|detail| CollectiveError::Unavailable {
            library: path.display().to_string(),
            detail,
        })?;
        bind(lib, path, flavor)
    }

    /// The registered backend name (`rccl`, `nccl`).
    pub fn backend_name(&self) -> &'static str {
        self.flavor.name
    }

    /// `ncclGetVersion` of the loaded library (`NCCL_VERSION_CODE` encoding).
    pub fn version_code(&self) -> i32 {
        self.version
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `ncclGetErrorString(code)`.
    pub fn error_string(&self, code: i32) -> String {
        // SAFETY: `get_error_string` comes from the library `self` keeps loaded; it takes a
        // plain integer and returns a pointer to a static NUL-terminated string (or NULL).
        let s = unsafe { (self.fns.get_error_string)(code) };
        if s.is_null() {
            return format!("unknown NCCL result {code}");
        }
        // SAFETY: non-NULL results of ncclGetErrorString point at static NUL-terminated text
        // owned by the library, which is never unloaded.
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

impl CollectiveLibrary for NcclApi {
    fn backend(&self) -> &'static str {
        self.flavor.name
    }
    fn version(&self) -> Option<String> {
        Some(version_text(self.version))
    }
    /// `ncclGetUniqueId`.
    fn unique_id(&self) -> Result<[u8; UNIQUE_ID_BYTES], CollectiveError> {
        let mut id = NcclUniqueId {
            internal: [0; UNIQUE_ID_BYTES],
        };
        // SAFETY: `get_unique_id` comes from the library `self` keeps loaded; `id` is an
        // exclusively borrowed local of the exact C type (128 bytes) the call writes.
        let rc = unsafe { (self.fns.get_unique_id)(&mut id) };
        if rc != 0 {
            return Err(CollectiveError::Backend {
                code: rc,
                message: format!("ncclGetUniqueId: {}", self.error_string(rc)),
            });
        }
        Ok(id.internal.map(|c| c as u8))
    }
    /// [`NcclCollective::init`] on a helper thread bound to the rank's device, given up after
    /// `init_timeout` + [`INIT_GRACE`] (see the module's ownership rules).
    fn open(self: Arc<Self>, init: CollectiveInit) -> Result<Arc<dyn Collective>, CollectiveError> {
        let (rank, world, init_timeout) = (init.rank, init.world, init.init_timeout);
        let metrics = init.metrics.clone();
        let backend = self.backend_name();
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name(format!("turbine-collective-init-{rank}"))
            .spawn(move || {
                // The library initialises on the calling thread's current device: a kernel-library
                // call makes the rank's device current on this thread (every call of the shim
                // selects its context's device first).
                if let Some(mem) = &init.memory
                    && let Err(e) = mem.mem_info()
                {
                    let _ = tx.send(Err(CollectiveError::Backend {
                        code: -1,
                        message: format!("selecting the rank's device for communicator init: {e}"),
                    }));
                    return;
                }
                let _ = tx.send(NcclCollective::init(self, init));
            });
        if let Err(e) = spawned {
            return Err(CollectiveError::Backend {
                code: -1,
                message: format!("cannot start the communicator init thread: {e}"),
            });
        }
        match rx.recv_timeout(init_timeout + INIT_GRACE) {
            Ok(result) => Ok(Arc::new(result?)),
            Err(_) => {
                tracing::warn!(
                    event = "collective_init_failed",
                    backend,
                    reason = "init_timeout",
                    rank,
                    world,
                    after_ms = (init_timeout + INIT_GRACE).as_millis() as u64,
                    "communicator init did not return; the rank gives it up (its thread is leaked)"
                );
                if let Some(m) = &metrics {
                    m.error(backend, super::CollectiveErrorKind::Timeout);
                }
                Err(CollectiveError::Timeout {
                    op: "comm_init",
                    after: init_timeout,
                })
            }
        }
    }
}

/// How long past its `init_timeout` a communicator init may take before the opening rank gives
/// up its helper thread.
pub const INIT_GRACE: Duration = Duration::from_secs(5);

/// The flavor whose file-name prefix `path` carries (`librccl*` → rccl, `libnccl*` → nccl).
/// Sets the flavor's library settings the operator did not set (`NcclFlavor::env_defaults`),
/// before the library is loaded and reads them; each applied default is logged once.
fn apply_env_defaults(flavor: &NcclFlavor) {
    static APPLIED: std::sync::Once = std::sync::Once::new();
    APPLIED.call_once(|| {
        for &(key, value) in flavor.env_defaults {
            if std::env::var_os(key).is_some() {
                continue;
            }
            // SAFETY: called once, from startup (the parallel plan's collective setup or
            // turbine-collbench's main) before any communicator exists and before the collective
            // library is loaded, so no thread of that library reads the environment concurrently;
            // no other Turbine code reads or writes this variable.
            unsafe { std::env::set_var(key, value) };
            tracing::info!(
                event = "collective_env_default",
                backend = flavor.name,
                key,
                value,
                "collective library setting applied (set it in the environment to override)"
            );
        }
    });
}

fn flavor_from_name(path: &Path) -> Option<&'static NcclFlavor> {
    let name = path.file_name()?.to_str()?;
    FLAVORS
        .iter()
        .copied()
        .find(|f| name.starts_with(f.file_prefix))
}

/// `2.27.3 (22703)` for an `NCCL_VERSION_CODE`.
fn version_text(code: i32) -> String {
    format!(
        "{}.{}.{} ({code})",
        code / 10_000,
        code / 100 % 100,
        code % 100
    )
}

fn open(path: &Path) -> Result<Library, String> {
    // SAFETY: loading runs the library's initialisers; RCCL and NCCL have none with
    // preconditions. The returned Library is either owned by an NcclApi (never unloaded) or
    // dropped before any symbol is resolved from it.
    unsafe { Library::new(path) }.map_err(|e| format!("{}: {}", path.display(), loader_error(&e)))
}

fn bind(
    lib: Library,
    path: &Path,
    flavor: &'static NcclFlavor,
) -> Result<Arc<NcclApi>, CollectiveError> {
    let unavailable = |detail: String| CollectiveError::Unavailable {
        library: path.display().to_string(),
        detail,
    };
    if let Some(named) = flavor_from_name(path)
        && !std::ptr::eq(named, flavor)
    {
        return Err(unavailable(format!(
            "the file name names backend {} but {} was requested",
            named.name, flavor.name
        )));
    }
    // Report every missing symbol at once rather than the first one only.
    let missing: Vec<&str> = SYMBOLS
        .iter()
        .copied()
        .filter(|name| {
            // SAFETY: only the symbol's presence is checked; the returned address is never
            // dereferenced or called.
            unsafe { lib.get::<*const c_void>(name.as_bytes()) }.is_err()
        })
        .collect();
    if !missing.is_empty() {
        return Err(unavailable(format!(
            "missing symbol(s) {} of the {} the binding needs",
            missing.join(", "),
            SYMBOLS.len()
        )));
    }
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            // SAFETY: `$ty` is the C signature nccl.h / rccl.h declare for `$name`; the pointer
            // is copied out and only used while `lib`, moved into the NcclApi below and never
            // unloaded, is loaded.
            let s = unsafe { lib.get::<$ty>($name.as_bytes()) }.map_err(|e| {
                unavailable(format!("missing symbol {}: {}", $name, loader_error(&e)))
            })?;
            *s
        }};
    }
    let fns = NcclFns {
        get_version: sym!("ncclGetVersion", GetVersionFn),
        get_unique_id: sym!("ncclGetUniqueId", GetUniqueIdFn),
        comm_init_rank_config: sym!("ncclCommInitRankConfig", CommInitRankConfigFn),
        comm_get_async_error: sym!("ncclCommGetAsyncError", CommGetAsyncErrorFn),
        comm_abort: sym!("ncclCommAbort", CommFn),
        comm_destroy: sym!("ncclCommDestroy", CommFn),
        all_reduce: sym!("ncclAllReduce", AllReduceFn),
        all_gather: sym!("ncclAllGather", AllGatherFn),
        reduce_scatter: sym!("ncclReduceScatter", AllReduceFn),
        broadcast: sym!("ncclBroadcast", BroadcastFn),
        send: sym!("ncclSend", P2pFn),
        recv: sym!("ncclRecv", P2pFn),
        group_start: sym!("ncclGroupStart", GroupFn),
        group_end: sym!("ncclGroupEnd", GroupFn),
        get_error_string: sym!("ncclGetErrorString", GetErrorStringFn),
    };
    let mut version: c_int = 0;
    // SAFETY: `get_version` was resolved from `lib`, still loaded here; `version` is an
    // exclusively borrowed local int, the out-parameter ncclGetVersion writes.
    let rc = unsafe { (fns.get_version)(&mut version) };
    if rc != 0 {
        return Err(unavailable(format!("ncclGetVersion returned {rc}")));
    }
    let minimum = flavor.min_version;
    if version < minimum {
        return Err(unavailable(format!(
            "{} version {} is below the minimum {}",
            flavor.name,
            version_text(version),
            version_text(minimum)
        )));
    }
    tracing::info!(
        event = "collective_library_loaded",
        backend = flavor.name,
        path = %path.display(),
        version = %version_text(version),
        "NCCL-API library loaded"
    );
    Ok(Arc::new(NcclApi {
        path: path.to_path_buf(),
        flavor,
        version,
        fns,
        _lib: ManuallyDrop::new(lib),
    }))
}

impl NcclApi {
    fn backend_error(&self, code: NcclResult) -> CollectiveError {
        CollectiveError::Backend {
            code,
            message: self.error_string(code),
        }
    }
}

const NCCL_SUCCESS: NcclResult = 0;
/// `ncclInProgress`: a non-blocking communicator call has not finished yet.
const NCCL_IN_PROGRESS: NcclResult = 7;
/// `ncclUint8`, `ncclFloat32`, `ncclBfloat16` (`ncclDataType_t`).
const NCCL_UINT8: c_int = 1;
const NCCL_FLOAT32: c_int = 7;
const NCCL_BFLOAT16: c_int = 9;
/// `ncclSum`, `ncclMax` (`ncclRedOp_t`).
const NCCL_SUM: c_int = 0;
const NCCL_MAX: c_int = 2;

/// `NCCL_API_MAGIC`.
const NCCL_API_MAGIC: c_uint = 0xcafe_beef;
/// `NCCL_CONFIG_UNDEF_INT` (`INT_MIN`).
const NCCL_CONFIG_UNDEF_INT: c_int = c_int::MIN;
/// `NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH)` of the header this layout is copied from.
const NCCL_CONFIG_VERSION: c_uint = 23_004;

/// `ncclConfig_t` exactly as `/opt/rocm/rocm/include/rccl/rccl.h` of ROCm 7.14.1 declares it
/// (`struct ncclConfig_v22800`, RCCL `NCCL_VERSION_CODE` 23004; read 2026-09-26). The library
/// reads `size` and `version` to know which fields exist; NCCL appends fields release by
/// release, so an older library reads the leading fields it knows.
#[repr(C)]
struct NcclConfig {
    size: usize,
    magic: c_uint,
    version: c_uint,
    blocking: c_int,
    cga_cluster_size: c_int,
    min_ctas: c_int,
    max_ctas: c_int,
    net_name: *const c_char,
    split_share: c_int,
    traffic_class: c_int,
    comm_name: *const c_char,
    collnet_enable: c_int,
    cta_policy: c_int,
    shrink_share: c_int,
    nvls_ctas: c_int,
    n_channels_per_net_peer: c_int,
    nvlink_centric_sched: c_int,
    graph_usage_mode: c_int,
    num_rma_ctx: c_int,
    max_p2p_peers: c_int,
}

impl NcclConfig {
    /// `NCCL_CONFIG_INITIALIZER` with `blocking = 0` (every call returns promptly, possibly
    /// with `ncclInProgress`, so a hung peer can never block a Turbine thread).
    fn non_blocking() -> Self {
        let undef = NCCL_CONFIG_UNDEF_INT;
        NcclConfig {
            size: std::mem::size_of::<NcclConfig>(),
            magic: NCCL_API_MAGIC,
            version: NCCL_CONFIG_VERSION,
            blocking: 0,
            cga_cluster_size: undef,
            min_ctas: undef,
            max_ctas: undef,
            net_name: std::ptr::null(),
            split_share: undef,
            traffic_class: undef,
            comm_name: std::ptr::null(),
            collnet_enable: undef,
            cta_policy: undef,
            shrink_share: undef,
            nvls_ctas: undef,
            n_channels_per_net_peer: undef,
            nvlink_centric_sched: undef,
            graph_usage_mode: undef,
            num_rma_ctx: undef,
            max_p2p_peers: undef,
        }
    }
}

/// An `ncclComm_t` shared between the calling thread and the watchdog.
#[derive(Clone, Copy)]
struct CommPtr(NcclComm);

// SAFETY: an ncclComm_t is an opaque handle that NCCL/RCCL allow to be used from any thread
// (ncclCommGetAsyncError and ncclCommAbort are documented to be called from a thread other
// than the one issuing collectives). Its end of life (abort/destroy) is serialised against
// every call that uses it by `Shared::gate`.
unsafe impl Send for CommPtr {}
// SAFETY: see `Send` above; the handle itself is never mutated after init.
unsafe impl Sync for CommPtr {}

/// Why a communicator was aborted by Turbine (reported to the call that was in flight).
#[derive(Clone, Copy, Debug)]
enum Failure {
    Timeout { op: &'static str, after: Duration },
    Backend { code: NcclResult },
}

/// A deadline the watchdog enforces.
#[derive(Clone, Copy, Debug)]
struct Armed {
    op: &'static str,
    /// `Clock::now_mono` value after which the communicator is aborted.
    deadline: Duration,
    after: Duration,
}

/// State shared by the rank's calling thread and its watchdog thread.
struct Shared {
    api: Arc<NcclApi>,
    rank: usize,
    /// `Some` while the communicator is usable, `None` once aborted or destroyed. Read-held
    /// only across non-blocking NCCL calls (the communicator is created with `blocking = 0`),
    /// so an abort never waits behind a hung peer.
    gate: RwLock<Option<CommPtr>>,
    /// The enqueue in flight.
    armed: Mutex<Option<Armed>>,
    /// The step in flight ([`Collective::step_begin`] .. `step_end`): bounds the device-side
    /// completion of every collective the step enqueued, which the enqueue watchdog cannot
    /// see (a peer that never arrives leaves the kernel spinning on the device).
    step: Mutex<Option<Armed>>,
    failure: Mutex<Option<Failure>>,
    stop: AtomicBool,
    clock: Arc<dyn Clock>,
    metrics: Option<CollectiveMetrics>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Shared {
    fn read_gate(&self) -> RwLockReadGuard<'_, Option<CommPtr>> {
        self.gate
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn error_metric(&self, e: &CollectiveError) {
        if let (Some(m), Some(kind)) = (&self.metrics, e.kind()) {
            m.error(self.api.backend_name(), kind);
        }
    }

    /// Aborts the communicator once; `why` is kept for the call in flight. Returns whether
    /// this call performed the abort.
    fn abort(&self, why: Option<Failure>) -> bool {
        let mut gate = self
            .gate
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(comm) = gate.take() else {
            return false;
        };
        if let Some(why) = why {
            lock(&self.failure).get_or_insert(why);
        }
        // SAFETY: `comm` came from ncclCommInitRankConfig of this library and was just taken
        // out of the gate under its write lock, so no other call is using it and nothing can
        // use it afterwards; ncclCommAbort releases it.
        let rc = unsafe { (self.api.fns.comm_abort)(comm.0) };
        tracing::warn!(
            event = "collective_aborted",
            backend = self.api.backend_name(),
            rank = self.rank,
            reason = ?why,
            abort_result = rc,
            "communicator aborted"
        );
        true
    }

    /// `ncclCommGetAsyncError`: the communicator's pending state (`ncclSuccess`,
    /// `ncclInProgress` or an error code).
    fn async_state(&self) -> Result<NcclResult, CollectiveError> {
        let gate = self.read_gate();
        let Some(comm) = *gate else {
            return Err(self.after_abort());
        };
        let mut state: NcclResult = NCCL_SUCCESS;
        // SAFETY: `comm` is live while the read guard is held (abort/destroy need the write
        // lock); `state` is an exclusively borrowed local the call writes.
        let rc = unsafe { (self.api.fns.comm_get_async_error)(comm.0, &mut state) };
        if rc != NCCL_SUCCESS {
            return Ok(rc);
        }
        Ok(state)
    }

    /// The error a call reports once the communicator is gone.
    fn after_abort(&self) -> CollectiveError {
        match *lock(&self.failure) {
            Some(Failure::Timeout { op, after }) => CollectiveError::Timeout { op, after },
            Some(Failure::Backend { code }) => self.api.backend_error(code),
            None => CollectiveError::RemoteAbort { rank: self.rank },
        }
    }

    /// Polls a non-blocking call to completion, aborting at `deadline`.
    fn wait_complete(&self, armed: Armed) -> Result<(), CollectiveError> {
        loop {
            match self.async_state()? {
                NCCL_SUCCESS => return Ok(()),
                NCCL_IN_PROGRESS => {
                    if self.clock.now_mono() >= armed.deadline {
                        self.abort(Some(Failure::Timeout {
                            op: armed.op,
                            after: armed.after,
                        }));
                        return Err(self.after_abort());
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                code => {
                    self.abort(Some(Failure::Backend { code }));
                    return Err(self.after_abort());
                }
            }
        }
    }

    /// Watchdog body: every 10 ms, abort on an asynchronous error or a passed deadline (of
    /// the enqueue or of the step).
    fn watch(&self) {
        while !self.stop.load(Ordering::Acquire) {
            match self.async_state() {
                Err(_) => return,
                Ok(NCCL_SUCCESS | NCCL_IN_PROGRESS) => {}
                Ok(code) => {
                    if self.abort(Some(Failure::Backend { code })) {
                        self.error_metric(&CollectiveError::Backend {
                            code,
                            message: String::new(),
                        });
                    }
                    return;
                }
            }
            let now = self.clock.now_mono();
            let passed = [*lock(&self.armed), *lock(&self.step)]
                .into_iter()
                .flatten()
                .find(|a| now >= a.deadline);
            if let Some(a) = passed {
                let failure = Failure::Timeout {
                    op: a.op,
                    after: a.after,
                };
                if self.abort(Some(failure)) {
                    self.error_metric(&CollectiveError::Timeout {
                        op: a.op,
                        after: a.after,
                    });
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// One rank of an NCCL-API communicator (RCCL or NCCL) with an init watchdog and a watchdog
/// thread. Created non-blocking, so no call into the library can hang a Turbine thread: every
/// enqueue is bounded by `op_timeout`, and so is a step between `step_begin` and `step_end`;
/// on expiry the communicator is aborted and every later call fails.
///
/// Buffers are the caller's device memory ([`DeviceSlice`]) and ordering is its stream
/// ([`StreamRef::native_handle`], the vendor stream of the kernel library's context).
pub struct NcclCollective {
    shared: Arc<Shared>,
    world: usize,
    op_timeout: Duration,
    watchdog: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for NcclCollective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclCollective")
            .field("backend", &self.shared.api.backend_name())
            .field("rank", &self.shared.rank)
            .field("world", &self.world)
            .finish_non_exhaustive()
    }
}

impl NcclCollective {
    /// `ncclCommInitRankConfig` (non-blocking) for `init.rank` of `init.world`, polled every
    /// 1 ms until ready; after `init.init_timeout` the communicator is aborted and
    /// `Timeout { op: "comm_init" }` returned. Waits sleep in real time and compare against
    /// `init.clock`.
    pub fn init(api: Arc<NcclApi>, init: CollectiveInit) -> Result<Self, CollectiveError> {
        let CollectiveInit {
            rank,
            world,
            unique_id,
            init_timeout,
            op_timeout,
            clock,
            metrics,
            memory: _,
            route_max_bytes: _,
        } = init;
        let backend = api.backend_name();
        let (Ok(nranks), Ok(rank_c)) = (c_int::try_from(world), c_int::try_from(rank)) else {
            return Err(CollectiveError::ShapeMismatch);
        };
        if world == 0 || rank >= world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let mut comm: NcclComm = std::ptr::null_mut();
        let mut config = NcclConfig::non_blocking();
        let id = NcclUniqueId {
            internal: unique_id.map(|b| b as c_char),
        };
        // SAFETY: `comm_init_rank_config` comes from the library `api` keeps loaded; `comm` and
        // `config` are exclusively borrowed locals of the layouts the header declares
        // (`ncclComm_t*`, `ncclConfig_t*`), and `id` is passed by value.
        let rc = unsafe {
            (api.fns.comm_init_rank_config)(
                &mut comm,
                nranks,
                id,
                rank_c,
                (&raw mut config).cast::<c_void>(),
            )
        };
        let shared = Arc::new(Shared {
            api: Arc::clone(&api),
            rank,
            gate: RwLock::new((!comm.is_null()).then_some(CommPtr(comm))),
            armed: Mutex::new(None),
            step: Mutex::new(None),
            failure: Mutex::new(None),
            stop: AtomicBool::new(false),
            clock: Arc::clone(&clock),
            metrics,
        });
        let fail = |e: CollectiveError| {
            shared.error_metric(&e);
            tracing::warn!(
                event = "collective_init_failed",
                backend,
                rank,
                world,
                error = %e,
                "communicator init failed"
            );
            Err(e)
        };
        if rc != NCCL_SUCCESS && rc != NCCL_IN_PROGRESS {
            shared.abort(Some(Failure::Backend { code: rc }));
            return fail(api.backend_error(rc));
        }
        if comm.is_null() {
            return fail(CollectiveError::Backend {
                code: rc,
                message: "ncclCommInitRankConfig returned no communicator".into(),
            });
        }
        let armed = Armed {
            op: "comm_init",
            deadline: clock.now_mono() + init_timeout,
            after: init_timeout,
        };
        if let Err(e) = shared.wait_complete(armed) {
            return fail(e);
        }
        let watched = Arc::clone(&shared);
        let watchdog = std::thread::Builder::new()
            .name(format!("turbine-collective-watchdog-{rank}"))
            .spawn(move || watched.watch())
            .map_err(|e| CollectiveError::Backend {
                code: -1,
                message: format!("cannot start the watchdog thread: {e}"),
            });
        let watchdog = match watchdog {
            Ok(handle) => handle,
            Err(e) => {
                shared.abort(None);
                return fail(e);
            }
        };
        tracing::info!(
            event = "collective_init",
            backend,
            rank,
            world,
            "communicator ready"
        );
        Ok(NcclCollective {
            shared,
            world,
            op_timeout,
            watchdog: Some(watchdog),
        })
    }

    fn arm(&self, op: &'static str) -> Armed {
        let armed = Armed {
            op,
            deadline: self.shared.clock.now_mono() + self.op_timeout,
            after: self.op_timeout,
        };
        *lock(&self.shared.armed) = Some(armed);
        armed
    }

    fn disarm(&self) {
        *lock(&self.shared.armed) = None;
    }

    /// `Err` (nothing enqueued) when `stream` belongs to a GPU context but has no native handle:
    /// the call would run on the legacy default stream ([`super::RouteReason::NoNativeStream`]).
    fn native_stream_ok(
        &self,
        op: CollectiveOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        if stream.native_handle() != 0 || stream.memory().as_host().is_some() {
            return Ok(());
        }
        let reason = super::RouteReason::NoNativeStream;
        if let Some(m) = &self.shared.metrics {
            m.route(op, self.shared.api.backend_name(), reason);
        }
        tracing::warn!(
            event = "collective_refused",
            op = op.as_str(),
            backend = self.shared.api.backend_name(),
            reason = reason.as_str(),
            "the stream has no native handle (kernel library without ABI v2.6): the call would \
             run unordered with the compute stream"
        );
        Err(CollectiveError::Unavailable {
            library: self.shared.api.backend_name().into(),
            detail: format!(
                "{}: {} on a stream without a native handle",
                reason.as_str(),
                op.as_str()
            ),
        })
    }

    /// Issues one non-blocking call under the op watchdog and waits for it to be enqueued.
    fn issue(
        &self,
        armed: Armed,
        call: impl FnOnce(NcclComm) -> NcclResult,
    ) -> Result<(), CollectiveError> {
        let rc = {
            let gate = self.shared.read_gate();
            let Some(comm) = *gate else {
                return Err(self.shared.after_abort());
            };
            call(comm.0)
        };
        match rc {
            NCCL_SUCCESS => Ok(()),
            NCCL_IN_PROGRESS => self.shared.wait_complete(armed),
            code => {
                self.shared.abort(Some(Failure::Backend { code }));
                Err(self.shared.after_abort())
            }
        }
    }

    /// Runs `call` as operation `op` of `bytes`: watchdog, metrics and error accounting.
    fn run(
        &self,
        op: CollectiveOp,
        bytes: usize,
        call: impl FnOnce(NcclComm) -> NcclResult,
    ) -> Result<(), CollectiveError> {
        let started = Instant::now();
        let armed = self.arm(op.as_str());
        let result = self.issue(armed, call);
        self.disarm();
        self.account(op, bytes, started, &result);
        result
    }

    fn account(
        &self,
        op: CollectiveOp,
        bytes: usize,
        started: Instant,
        result: &Result<(), CollectiveError>,
    ) {
        match result {
            Ok(()) => {
                if let Some(m) = &self.shared.metrics {
                    m.observe(
                        op,
                        self.shared.api.backend_name(),
                        bytes as u64,
                        started.elapsed().as_secs_f64(),
                    );
                }
            }
            Err(e) => self.shared.error_metric(e),
        }
    }
}

/// `ncclDataType_t` and element count of a reduction over `bytes` of `dtype` (BF16 or FP32).
fn reduce_type(dtype: DType, bytes: usize) -> Result<(c_int, usize), CollectiveError> {
    let code = match dtype {
        DType::BF16 => NCCL_BFLOAT16,
        DType::F32 => NCCL_FLOAT32,
        _ => return Err(CollectiveError::ShapeMismatch),
    };
    let size = dtype.size_bytes();
    if !bytes.is_multiple_of(size) {
        return Err(CollectiveError::ShapeMismatch);
    }
    Ok((code, bytes / size))
}

fn reduce_op(op: ReduceOp) -> c_int {
    match op {
        ReduceOp::Sum => NCCL_SUM,
        ReduceOp::Max => NCCL_MAX,
    }
}

fn device_ptr(slice: &DeviceSlice) -> *mut c_void {
    slice.ptr().addr() as *mut c_void
}

fn native_stream(stream: &StreamRef) -> NativeStream {
    stream.native_handle() as NativeStream
}

impl Collective for NcclCollective {
    fn backend(&self) -> &'static str {
        self.shared.api.backend_name()
    }

    fn rank(&self) -> usize {
        self.shared.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce(
        &self,
        buf: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let (dt, count) = reduce_type(dtype, buf.len())?;
        let (ptr, s, redop) = (device_ptr(buf), native_stream(stream), reduce_op(op));
        self.native_stream_ok(CollectiveOp::AllReduce, stream)?;
        let f = self.shared.api.fns.all_reduce;
        self.run(CollectiveOp::AllReduce, buf.len(), |comm| {
            // SAFETY: `f` comes from the library the NcclApi keeps loaded; `comm` is live under
            // the gate's read guard; `ptr` is `count` elements of the caller's device buffer,
            // borrowed mutably for this call and in place (send == recv is allowed by NCCL);
            // `s` is the caller's stream, which outlives the enqueue.
            unsafe { f(ptr, ptr, count, dt, redop, comm, s) }
        })
    }

    fn all_gather(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let count = send.len();
        if recv.len() != count * self.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let (src, dst, s) = (device_ptr(send), device_ptr(recv), native_stream(stream));
        self.native_stream_ok(CollectiveOp::AllGather, stream)?;
        let f = self.shared.api.fns.all_gather;
        self.run(CollectiveOp::AllGather, count, |comm| {
            // SAFETY: as in all_reduce; `src` holds `count` bytes and `dst` (borrowed mutably)
            // world × `count`, both caller-owned device memory.
            unsafe { f(src, dst, count, NCCL_UINT8, comm, s) }
        })
    }

    fn reduce_scatter(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let (dt, count) = reduce_type(dtype, recv.len())?;
        if send.len() != recv.len() * self.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let (src, dst, s, redop) = (
            device_ptr(send),
            device_ptr(recv),
            native_stream(stream),
            reduce_op(op),
        );
        self.native_stream_ok(CollectiveOp::ReduceScatter, stream)?;
        let f = self.shared.api.fns.reduce_scatter;
        self.run(CollectiveOp::ReduceScatter, send.len(), |comm| {
            // SAFETY: as in all_reduce; `src` holds world × `count` elements and `dst`
            // (borrowed mutably) `count`, both caller-owned device memory.
            unsafe { f(src, dst, count, dt, redop, comm, s) }
        })
    }

    fn broadcast(
        &self,
        buf: &mut DeviceSlice,
        root: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let Ok(root_c) = c_int::try_from(root) else {
            return Err(CollectiveError::ShapeMismatch);
        };
        if root >= self.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let (ptr, count, s) = (device_ptr(buf), buf.len(), native_stream(stream));
        self.native_stream_ok(CollectiveOp::Broadcast, stream)?;
        let f = self.shared.api.fns.broadcast;
        self.run(CollectiveOp::Broadcast, count, |comm| {
            // SAFETY: as in all_reduce; in place on `count` bytes of the caller's buffer.
            unsafe { f(ptr, ptr, count, NCCL_UINT8, root_c, comm, s) }
        })
    }

    fn send(
        &self,
        buf: &DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let Ok(peer_c) = c_int::try_from(peer) else {
            return Err(CollectiveError::ShapeMismatch);
        };
        if peer >= self.world || peer == self.shared.rank {
            return Err(CollectiveError::ShapeMismatch);
        }
        let (ptr, count, s) = (device_ptr(buf), buf.len(), native_stream(stream));
        self.native_stream_ok(CollectiveOp::Send, stream)?;
        let f = self.shared.api.fns.send;
        self.run(CollectiveOp::Send, count, |comm| {
            // SAFETY: as in all_reduce; ncclSend only reads `count` bytes of the caller's
            // buffer (declared `const void*`; the pointer type differs only in constness).
            unsafe { f(ptr, count, NCCL_UINT8, peer_c, comm, s) }
        })
    }

    fn recv(
        &self,
        buf: &mut DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let Ok(peer_c) = c_int::try_from(peer) else {
            return Err(CollectiveError::ShapeMismatch);
        };
        if peer >= self.world || peer == self.shared.rank {
            return Err(CollectiveError::ShapeMismatch);
        }
        let (ptr, count, s) = (device_ptr(buf), buf.len(), native_stream(stream));
        self.native_stream_ok(CollectiveOp::Recv, stream)?;
        let f = self.shared.api.fns.recv;
        self.run(CollectiveOp::Recv, count, |comm| {
            // SAFETY: as in all_reduce; ncclRecv writes `count` bytes of the caller's buffer,
            // borrowed mutably for this call.
            unsafe { f(ptr, count, NCCL_UINT8, peer_c, comm, s) }
        })
    }

    /// An all-reduce of one FP32 on `stream`, then a synchronize of the stream's context, all
    /// under the op watchdog (a peer that never arrives is aborted at `op_timeout`).
    fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError> {
        if self.shared.read_gate().is_none() {
            return Err(self.shared.after_abort());
        }
        let started = Instant::now();
        let mem = stream.memory();
        let scratch = DeviceBuffer::alloc(mem, 4).map_err(|e| CollectiveError::Backend {
            code: -1,
            message: format!("barrier scratch allocation: {e}"),
        })?;
        let ptr = device_ptr(&scratch.whole());
        let s = native_stream(stream);
        let f = self.shared.api.fns.all_reduce;
        let armed = self.arm(CollectiveOp::Barrier.as_str());
        let result = self
            .issue(armed, |comm| {
                // SAFETY: as in all_reduce; `ptr` is the 4-byte scratch buffer owned by this
                // call, which outlives the synchronize below.
                unsafe { f(ptr, ptr, 1, NCCL_FLOAT32, NCCL_SUM, comm, s) }
            })
            .and_then(|()| {
                mem.synchronize().map_err(|e| {
                    // A synchronize released by the watchdog's abort reports the abort reason.
                    if self.shared.read_gate().is_none() {
                        self.shared.after_abort()
                    } else {
                        CollectiveError::Backend {
                            code: -1,
                            message: format!("barrier synchronize: {e}"),
                        }
                    }
                })
            })
            .and_then(|()| {
                // Aborted by the watchdog while the synchronize waited.
                if self.shared.read_gate().is_none() {
                    Err(self.shared.after_abort())
                } else {
                    Ok(())
                }
            });
        self.disarm();
        drop(scratch);
        self.account(CollectiveOp::Barrier, 4, started, &result);
        result
    }

    fn step_begin(&self) {
        *lock(&self.shared.step) = Some(Armed {
            op: "step",
            deadline: self.shared.clock.now_mono() + self.op_timeout,
            after: self.op_timeout,
        });
    }

    fn step_end(&self) -> Result<(), CollectiveError> {
        *lock(&self.shared.step) = None;
        if self.shared.read_gate().is_none() {
            Err(self.shared.after_abort())
        } else {
            Ok(())
        }
    }

    fn abort(&self) {
        self.shared.abort(None);
    }
}

impl Drop for NcclCollective {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(handle) = self.watchdog.take() {
            let _ = handle.join();
        }
        let mut gate = self
            .shared
            .gate
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(comm) = gate.take() {
            // SAFETY: the watchdog has exited and the gate's write lock is held, so no call is
            // using `comm`; it is taken out of the gate, so it is destroyed exactly once.
            let rc = unsafe { (self.shared.api.fns.comm_destroy)(comm.0) };
            if rc != NCCL_SUCCESS && rc != NCCL_IN_PROGRESS {
                tracing::warn!(
                    event = "collective_destroy_failed",
                    backend = self.shared.api.backend_name(),
                    rank = self.shared.rank,
                    code = rc,
                    "ncclCommDestroy failed"
                );
            }
        }
    }
}

/// libloading's message plus the platform loader's (`dlerror`) text.
fn loader_error(e: &libloading::Error) -> String {
    match std::error::Error::source(e) {
        Some(source) => format!("{e}: {source}"),
        None => e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::collective::CollectiveError;
    use crate::collective::nccl_api::{NCCL_FLAVOR, RCCL_FLAVOR, RCCL_MIN_VERSION};

    fn stub(variant: &str, file: &str) -> PathBuf {
        Path::new(env!("TURBINE_NCCL_STUB_DIR"))
            .join(variant)
            .join(file)
    }

    fn unavailable(e: CollectiveError) -> (String, String) {
        match e {
            CollectiveError::Unavailable { library, detail } => (library, detail),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn missing_library() {
        #[cfg(target_os = "macos")]
        for flavor in [&RCCL_FLAVOR, &NCCL_FLAVOR] {
            let (library, detail) =
                unavailable(NcclApi::load(flavor, None).expect_err("no NCCL-API library on macOS"));
            assert!(
                library.contains(flavor.name),
                "{}: library {library}",
                flavor.name
            );
            // The platform loader's message (dlopen) is carried through.
            assert!(detail.contains("dlopen"), "{}: {detail}", flavor.name);
        }
        let explicit = Path::new("/nonexistent/librccl.so.1");
        let err =
            NcclApi::load(&RCCL_FLAVOR, Some(explicit)).expect_err("explicit path does not exist");
        assert!(
            err.to_string().contains("/nonexistent/librccl.so.1"),
            "{err}"
        );
    }

    #[test]
    fn one_binding_both_libraries() {
        let expected: i32 = env!("TURBINE_NCCL_STUB_VERSION").parse().expect("version");
        let rccl = NcclApi::load_from(&stub("rccl", "librccl.so.1"), &RCCL_FLAVOR)
            .expect("rccl stub loads");
        assert_eq!(rccl.backend_name(), "rccl");
        assert_eq!(rccl.version_code(), expected);
        assert_eq!(rccl.path(), stub("rccl", "librccl.so.1"));

        // The same binding type serves NCCL; the file name decides the backend.
        let nccl: Arc<NcclApi> = NcclApi::load_from(&stub("nccl", "libnccl.so.2"), &NCCL_FLAVOR)
            .expect("nccl stub loads");
        assert_eq!(nccl.backend_name(), "nccl");
        assert_eq!(nccl.version_code(), expected);
        assert_eq!(SYMBOLS.len(), 15);
        assert_eq!(nccl.error_string(4), "invalid argument");
        // Through the registry's trait: the name, the version text and a group id.
        let lib: &dyn CollectiveLibrary = &*nccl;
        assert_eq!(lib.backend(), "nccl");
        assert_eq!(lib.version().as_deref(), Some("2.30.4 (23004)"));
        assert!(lib.unique_id().is_ok());

        // An explicit path goes through the same loader.
        let explicit = NcclApi::load(&NCCL_FLAVOR, Some(&stub("nccl", "libnccl.so.2")))
            .expect("explicit stub path loads");
        assert_eq!(explicit.backend_name(), "nccl");

        // A file named for the other vendor is refused.
        let (_, detail) = unavailable(
            NcclApi::load_from(&stub("rccl", "librccl.so.1"), &NCCL_FLAVOR)
                .expect_err("rccl file for an nccl request"),
        );
        assert!(detail.contains("rccl"), "{detail}");

        let (library, detail) = unavailable(
            NcclApi::load_from(&stub("low", "librccl.so.1"), &RCCL_FLAVOR)
                .expect_err("version below the minimum"),
        );
        assert!(library.ends_with("low/librccl.so.1"), "{library}");
        assert!(detail.contains("21800"), "{detail}");
        assert!(detail.contains(&RCCL_MIN_VERSION.to_string()), "{detail}");

        let (_, detail) = unavailable(
            NcclApi::load_from(&stub("nogroupend", "libnccl.so.2"), &NCCL_FLAVOR)
                .expect_err("ncclGroupEnd missing"),
        );
        assert!(detail.contains("ncclGroupEnd"), "{detail}");
    }

    /// `stub_abort_calls()` of the stub loaded from `path` (the loader returns the handle the
    /// NcclApi already holds, so the counter is the one it increments).
    fn stub_abort_calls(path: &Path) -> i32 {
        // SAFETY: test-only; the stub library is never unloaded while the NcclApi under test
        // holds it, and `stub_abort_calls` is `int (void)` in tests/stub/nccl_stub.c.
        unsafe {
            let lib = Library::new(path).expect("stub reloads");
            let f = lib
                .get::<unsafe extern "C" fn() -> c_int>(b"stub_abort_calls")
                .expect("test hook");
            f()
        }
    }

    fn init(
        rank: usize,
        world: usize,
        id: [u8; UNIQUE_ID_BYTES],
        init_timeout: Duration,
        op_timeout: Duration,
        metrics: Option<CollectiveMetrics>,
    ) -> CollectiveInit {
        CollectiveInit {
            rank,
            world,
            unique_id: id,
            init_timeout,
            op_timeout,
            clock: Arc::new(turbine_core::clock::SystemClock::new()),
            metrics,
            memory: None,
            route_max_bytes: None,
        }
    }

    #[test]
    fn stub_init_times_out() {
        let path = stub("inithang", "librccl.so.1");
        let api = NcclApi::load_from(&path, &RCCL_FLAVOR).expect("stub loads");
        let id = api.unique_id().expect("unique id");
        assert_eq!(id[5], 5, "the stub fills the id with 0..128");
        let before = stub_abort_calls(&path);
        let reg = turbine_observability::MetricsRegistry::new();
        let started = Instant::now();
        let err = Arc::clone(&api)
            .open(init(
                0,
                2,
                id,
                Duration::from_millis(200),
                Duration::from_secs(1),
                Some(CollectiveMetrics::register(&reg)),
            ))
            .map(|_| ())
            .expect_err("init never completes");
        let waited = started.elapsed();
        assert!(
            matches!(err, CollectiveError::Timeout { op: "comm_init", after } if after == Duration::from_millis(200)),
            "{err:?}"
        );
        assert!(
            waited >= Duration::from_millis(200) && waited < Duration::from_secs(1),
            "{waited:?}"
        );
        assert_eq!(stub_abort_calls(&path), before + 1, "one ncclCommAbort");
        let text = reg.render().expect("renders");
        assert!(
            text.contains("turbine_collective_errors_total{backend=\"rccl\",kind=\"timeout\"} 1"),
            "{text}"
        );
    }

    #[test]
    fn stub_world_one_abort_fails_later_calls() {
        use turbine_tensor::host::HostMemory;
        use turbine_tensor::{DeviceId, DeviceMemory};

        let path = stub("nccl", "libnccl.so.2");
        let api = NcclApi::load_from(&path, &NCCL_FLAVOR).expect("stub loads");
        let id = api.unique_id().expect("unique id");
        let t = Duration::from_secs(1);
        let comm = Arc::clone(&api)
            .open(init(0, 1, id, t, t, None))
            .expect("a world of one initialises");
        assert_eq!(comm.backend(), "nccl");
        assert_eq!((comm.rank(), comm.world_size()), (0, 1));

        let before = stub_abort_calls(&path);
        comm.abort();
        comm.abort();
        assert_eq!(stub_abort_calls(&path), before + 1, "aborted exactly once");

        // Every later call fails before reaching the library.
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 10);
        let stream = mem.compute_stream();
        let buf = DeviceBuffer::alloc(&mem, 16).expect("alloc");
        let mut slice = buf.whole();
        let err = comm
            .all_reduce(&mut slice, DType::F32, ReduceOp::Sum, &stream)
            .expect_err("aborted");
        assert!(
            matches!(err, CollectiveError::RemoteAbort { rank: 0 }),
            "{err:?}"
        );
        assert!(matches!(
            comm.barrier(&stream),
            Err(CollectiveError::RemoteAbort { .. })
        ));
        drop(comm);
        assert_eq!(stub_abort_calls(&path), before + 1, "no destroy-time abort");
    }

    /// A GPU context's stream without a native handle (a kernel library without ABI v2.6) is
    /// refused before the library is called (`Unavailable`, reason `no_native_stream`, counted),
    /// and the communicator stays usable: the same call on a context with native handles runs.
    /// Breaks if a null stream handle reaches the NCCL-API library (it would run on the legacy
    /// default stream, unordered with the context's compute stream).
    #[test]
    fn stub_refuses_a_gpu_stream_without_a_native_handle() {
        use turbine_kernels::test_support::stub_mapped_context_minor;
        use turbine_tensor::DeviceMemory;

        let path = stub("nccl", "libnccl.so.2");
        let api = NcclApi::load_from(&path, &NCCL_FLAVOR).expect("stub loads");
        let id = api.unique_id().expect("unique id");
        let t = Duration::from_secs(1);
        let reg = turbine_observability::MetricsRegistry::new();
        let metrics = CollectiveMetrics::register(&reg);
        let comm = Arc::clone(&api)
            .open(CollectiveInit {
                metrics: Some(metrics),
                ..init(0, 1, id, t, t, None)
            })
            .expect("a world of one initialises");
        let old: Arc<dyn DeviceMemory> = stub_mapped_context_minor(0, 5);
        assert_eq!(
            old.compute_stream().native_handle(),
            0,
            "v2.5: no native handle"
        );
        let buf = DeviceBuffer::alloc(&old, 16).expect("alloc");
        let err = comm
            .all_reduce(
                &mut buf.whole(),
                DType::F32,
                ReduceOp::Sum,
                &old.compute_stream(),
            )
            .expect_err("refused");
        assert!(
            matches!(&err, CollectiveError::Unavailable { detail, .. } if detail.contains("no_native_stream")),
            "{err:?}"
        );
        let text = reg.render().expect("renders");
        assert!(text.contains("reason=\"no_native_stream\""), "{text}");
        let new: Arc<dyn DeviceMemory> = stub_mapped_context_minor(1, 8);
        assert_ne!(new.compute_stream().native_handle(), 0);
        let buf = DeviceBuffer::alloc(&new, 16).expect("alloc");
        comm.all_reduce(
            &mut buf.whole(),
            DType::F32,
            ReduceOp::Sum,
            &new.compute_stream(),
        )
        .expect("a stream with a native handle runs");
    }

    /// A step whose device work never completes (nobody calls `step_end`) is aborted by the
    /// watchdog at the op timeout, and the late `step_end` reports the step timeout.
    #[test]
    fn stub_step_deadline_aborts() {
        let path = stub("rccl", "librccl.so.1");
        let api = NcclApi::load_from(&path, &RCCL_FLAVOR).expect("stub loads");
        let id = api.unique_id().expect("unique id");
        let comm = Arc::clone(&api)
            .open(init(
                0,
                1,
                id,
                Duration::from_secs(1),
                Duration::from_millis(100),
                None,
            ))
            .expect("a world of one initialises");
        // A step that ends in time leaves the communicator usable.
        comm.step_begin();
        comm.step_end().expect("step ended in time");
        let before = stub_abort_calls(&path);
        comm.step_begin();
        std::thread::sleep(Duration::from_millis(400));
        let err = comm.step_end().expect_err("the watchdog aborted the step");
        assert!(
            matches!(err, CollectiveError::Timeout { op: "step", .. }),
            "{err:?}"
        );
        assert_eq!(stub_abort_calls(&path), before + 1);
    }
}
