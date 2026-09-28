//! The runtime-loaded kernel shim library (`libturbine_hip.so`, later `libturbine_cuda.so`):
//! loading with ABI version, backend and build-arch checks, the device context (a
//! `DeviceMemory` backend), and the kernel provider that forwards every op through the C ABI.
//!
//! Ownership rules (TS §21 rule 10, contract §9.2):
//! - every device pointer is allocated by the shim (`turbine_malloc`) and owned by exactly one
//!   `DeviceBuffer`, which frees it through this context on `Drop`;
//! - the shim never retains a caller pointer beyond the call; host buffers passed to the copy
//!   functions are kept borrowed until the context stream is synchronized;
//! - the context owns its streams and workspace and is destroyed exactly once, in
//!   `ShimContext::drop`; every `DeviceBuffer` holds an `Arc` of its context, so no allocation
//!   outlives it, and the context holds an `Arc` of its `ShimLibrary`, so the code stays loaded;
//! - a captured graph (ABI v2.1) records the device pointers of the ops captured into it: the
//!   caller keeps those buffers alive while the `GraphHandle` lives, and the handle (which holds
//!   an `Arc` of its context) destroys the graph exactly once, in `GraphHandle::drop`;
//! - a host staging buffer (ABI v2.3) is page-locked memory from `turbine_host_alloc_pinned` plus one
//!   event, both owned by the context's staging table and released exactly once, in
//!   `staging_free` (called by `HostStaging::drop`, which holds an `Arc` of the context). Every
//!   staged copy re-records the buffer's event after it is enqueued; every host access to the
//!   buffer, and its release, first waits on that event, so the host never touches bytes a copy
//!   in flight reads or writes. The table's lock serialises those accesses.
//!
//! ABI v2.1 is optional: `ShimLibrary::abi_minor` is 0 for a v2.0 library, whose provider then
//! has no `add_rmsnorm`/`logits_reduce` family and whose context answers the option and graph
//! calls with `KernelError::Unsupported`. ABI v2.6 is optional the same way: without it the
//! provider has no `sharded_norm` family and `ShimContext::compute_stream` carries native handle
//! 0 (`ShimContext::has_native_streams` is false), so collectives cannot be ordered with the ops.
//! The native handle is owned by the context (it is the context's compute stream) and is valid
//! while the context lives; every `StreamRef` holds an `Arc` of the context.
use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use libloading::Library;
use turbine_core::types::{DType, DeviceId};
use turbine_device::DeviceInfo;
use turbine_tensor::tensor::contiguous_strides;
use turbine_tensor::{
    DeviceMemory, DevicePtr, DeviceSlice, MappedCollectives, MemInfo, MemoryError, StagingId,
    StreamRef, TensorView,
};

use crate::cards::CardProfile;
use crate::ffi::{
    self, AddDesc, AddRmsnormDesc, AttentionDesc, AttentionPagedDesc, CopyBlocksDesc, CtxInfo,
    EmbeddingDesc, GemmDesc, LogitsReduceDesc, MOE_ROUTE_BF16_LOGITS, MOE_ROUTE_RENORMALIZE,
    MoeExpertsDesc, MoeRouteDesc, OpTrio, RmsnormDesc, RmsnormShardedDesc, RopeDesc, RowSumsqDesc,
    ShimSymbols, SiluMulDesc, StagingFns, TurbineCtx, TurbineEvent, TurbineGraph,
};
use crate::ops::{
    ActivationConfig, ActivationContext, ActivationKernel, AddRmsnormConfig, AddRmsnormContext,
    AddRmsnormKernel, AttentionConfig, AttentionContext, AttentionKernel, AttentionKind,
    ElementwiseConfig, ElementwiseContext, ElementwiseKernel, EmbeddingConfig, EmbeddingContext,
    EmbeddingKernel, GemmConfig, GemmContext, GemmKernel, ImplChoice, ImplInfo, KernelProvider,
    KvCopyConfig, KvCopyContext, KvCopyKernel, LogitsReduceConfig, LogitsReduceContext,
    LogitsReduceKernel, MoeExpertsConfig, MoeExpertsContext, MoeKernel, MoeRouteConfig,
    MoeRouteContext, NormConfig, NormContext, NormKernel, OpKind, PagedAttentionContext,
    ProviderId, RmsnormShardedConfig, RmsnormShardedContext, RopeConfig, RopeContext, RopeKernel,
    RowSumsqConfig, RowSumsqContext, ShardedNormKernel,
};
use crate::pinned::PinnedState;
use crate::registry::OpConfig;
use crate::{KernelError, TURBINE_KERNELS_ABI_VERSION};

/// Environment variable naming the shim library when `execution.kernel_library` is null.
const KERNEL_LIBRARY_VAR: &str = "TURBINE_KERNEL_LIBRARY";

/// Context option (ABI v2.1) `TURBINE_OPTION_GEMM_AUTOTUNE`: 1 (the library default) runs the
/// pinned algorithm of the library's tuned GEMM table for the card wherever the table has a row
/// for the shape, 0 takes the first heuristic answer for every shape (`execution.gemm_autotune`).
pub const TURBINE_OPTION_GEMM_AUTOTUNE: i32 = 1;
/// Read-only context option (ABI v2.1) `TURBINE_OPTION_GEMM_TUNED_SHAPES`: GEMM shapes run so far
/// on the context that use a pinned algorithm of the tuned table.
pub const TURBINE_OPTION_GEMM_TUNED_SHAPES: i32 = 2;
/// Context option `TURBINE_OPTION_GEMM_PREFILL` (additive in ABI v2.5): 1 marks the following
/// GEMMs as a prefill step's, so shapes with row-invariant table rows run them
/// ([`crate::ops::GemmContext::prefill`]); 0 (the library default) a decode step's. The shim sets
/// it when a GEMM's step kind differs from the previous one.
pub const TURBINE_OPTION_GEMM_PREFILL: i32 = 3;
/// `ShimContext::gemm_prefill` once the library refused `TURBINE_OPTION_GEMM_PREFILL` (a library
/// older than the option: every GEMM runs as a decode step's).
const GEMM_PREFILL_UNSUPPORTED: u8 = 2;

/// A loaded kernel shim library whose ABI version and backend name have been checked.
pub struct ShimLibrary {
    path: PathBuf,
    syms: ShimSymbols,
    /// The configured backend's name, equal to `turbine_backend_name()`; the provider id.
    backend: &'static str,
    archs: Vec<String>,
    // Declared last so it is dropped after the symbol table copied out of it.
    _lib: Library,
}

impl fmt::Debug for ShimLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShimLibrary")
            .field("path", &self.path)
            .field("backend", &self.backend)
            .field("archs", &self.archs)
            .finish_non_exhaustive()
    }
}

/// The `&'static` form of a backend name, the provider id of the library's kernels. Each
/// distinct name is leaked once (there are as many as registered backends), however many
/// libraries are loaded.
fn intern_backend(name: &str) -> &'static str {
    static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&known) = names.iter().find(|&&n| n == name) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.push(leaked);
    leaked
}

impl ShimLibrary {
    /// Candidate library paths in search order (P1 §Configuration): `explicit`
    /// (`execution.kernel_library`) alone when set; otherwise `TURBINE_KERNEL_LIBRARY`, then
    /// `libturbine_<backend>.so` beside the executable, then the bare file name (resolved by the
    /// dynamic loader's search path).
    pub fn search_paths(backend: &str, explicit: Option<&Path>) -> Vec<PathBuf> {
        if let Some(path) = explicit {
            return vec![path.to_path_buf()];
        }
        let file_name = format!("libturbine_{backend}.so");
        let mut paths = Vec::with_capacity(3);
        if let Some(env) = std::env::var_os(KERNEL_LIBRARY_VAR).filter(|v| !v.is_empty()) {
            paths.push(PathBuf::from(env));
        }
        if let Some(dir) = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
        {
            paths.push(dir.join(&file_name));
        }
        paths.push(PathBuf::from(file_name));
        paths
    }

    /// Loads the library at `path`. The ABI version is checked before any other symbol is
    /// resolved (a library of another ABI may lack them), then every ABI v2 symbol is resolved
    /// and the backend name compared with `expected_backend` (the `execution.backend` name).
    pub fn load(path: &Path, expected_backend: &str) -> Result<Arc<ShimLibrary>, KernelError> {
        // SAFETY: loading a shared library runs its initialisers. The kernel shim is Turbine's
        // own C ABI library (kernels/), whose initialisers only register device code; it stays
        // loaded for the lifetime of the returned `ShimLibrary`.
        let lib = unsafe { Library::new(path) }.map_err(|e| KernelError::Load {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })?;
        let abi_version: unsafe extern "C" fn() -> u32 =
            ffi::resolve(&lib, path, "turbine_abi_version")?;
        // SAFETY: `turbine_abi_version` takes no arguments and returns an integer; `lib` is
        // loaded for the duration of the call.
        let found = unsafe { abi_version() };
        if found != TURBINE_KERNELS_ABI_VERSION {
            return Err(KernelError::AbiMismatch {
                expected: TURBINE_KERNELS_ABI_VERSION,
                found,
            });
        }
        let syms = ShimSymbols::resolve_all(&lib, path)?;
        // SAFETY: identity function without arguments returning a static string (see c_str).
        let backend = ffi::c_str(unsafe { (syms.backend_name)() });
        if backend != expected_backend {
            return Err(KernelError::BackendMismatch {
                expected: expected_backend.to_string(),
                found: backend,
            });
        }
        // SAFETY: identity function without arguments returning a static string (see c_str).
        let archs = ffi::c_str(unsafe { (syms.build_archs)() })
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(str::to_string)
            .collect();
        Ok(Arc::new(ShimLibrary {
            path: path.to_path_buf(),
            syms,
            backend: intern_backend(&backend),
            archs,
            _lib: lib,
        }))
    }

    /// The resolved ABI symbols (crate-internal: `pinned` reads the optional groups).
    pub(crate) fn syms(&self) -> &ShimSymbols {
        &self.syms
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn abi_version(&self) -> u32 {
        // SAFETY: no arguments, returns an integer; the library is loaded while `self` lives.
        unsafe { (self.syms.abi_version)() }
    }

    /// The ABI minor revision (`turbine_abi_minor`); 0 when the library does not export it (a
    /// v2.0 library such as the Phase 2 HIP build or the Phase 2b CUDA shim).
    pub fn abi_minor(&self) -> u32 {
        self.syms.v21.minor
    }

    /// True when the library exports the ABI v2.1 graph functions (`turbine_graph_*`), so its
    /// contexts can capture and replay graphs.
    pub fn supports_graphs(&self) -> bool {
        self.syms.v21.graph.is_some()
    }

    /// The `KernelError::Unsupported` for a v2.1 function this library lacks.
    fn lacks(&self, what: &str) -> KernelError {
        KernelError::Unsupported {
            message: format!(
                "{} does not export {what} (kernel ABI minor {})",
                self.path.display(),
                self.syms.v21.minor
            ),
        }
    }

    pub fn backend_name(&self) -> &str {
        self.backend
    }

    /// The device architectures the library was compiled for (`turbine_build_archs`).
    pub fn build_archs(&self) -> &[String] {
        &self.archs
    }

    /// True when the library exports the ABI v2.6 group: the native handle of its streams (so a
    /// collective library can enqueue on the compute stream) and the sharded RMSNorm ops. Without
    /// it tensor parallelism cannot run on this library.
    pub fn exports_native_streams(&self) -> bool {
        self.syms.v21.tensor_parallel.is_some()
    }

    /// True when the library exports the ABI v2.4 group (implementation enumeration, explicit
    /// runs and the card profile); otherwise it chooses the implementation of every call itself.
    pub fn enumerates_implementations(&self) -> bool {
        self.syms.v21.impls.is_some()
    }

    /// The implementations of `op` the library contains, in its order (ABI v2.4
    /// `turbine_impl_count` / `turbine_impl_info`); empty when it does not enumerate, predates
    /// the op (a minor below [`OpKind::abi_minor`]) or reports an error for `op`.
    pub fn implementations(&self, op: OpKind) -> Vec<ImplInfo> {
        let Some(fns) = self.syms.v21.impls else {
            return Vec::new();
        };
        if self.syms.v21.minor < op.abi_minor() {
            return Vec::new();
        }
        let code = op.abi_code();
        // SAFETY: `turbine_impl_count` takes an integer and returns one; no pointer is involved.
        let count = unsafe { (fns.count)(code) };
        let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        for index in 0..count.max(0) {
            let mut entry = ffi::ImplEntry::zeroed();
            // SAFETY: `entry` is a live, writable `turbine_impl_entry` on this stack frame; the
            // library fills it with pointers to static strings and keeps no pointer to it.
            let rc = unsafe { (fns.info)(code, index, &mut entry) };
            if rc != 0 {
                return Vec::new();
            }
            out.push(ImplInfo {
                index: index as u32,
                name: ffi::c_str(entry.name),
                provider: ffi::c_str(entry.provider),
                needs_host_offsets: entry.flags & ffi::IMPL_NEEDS_HOST_OFFSETS != 0,
            });
        }
        out
    }

    /// `turbine_impl_supports(op, index, desc)` (ABI v2.4): false when the library does not
    /// enumerate. `desc` must be `op`'s descriptor type.
    fn impl_supports<D>(&self, op: OpKind, index: u32, desc: &D) -> bool {
        let Some(fns) = self.syms.v21.impls else {
            return false;
        };
        let Ok(index) = i32::try_from(index) else {
            return false;
        };
        // SAFETY: `desc` is a fully initialised descriptor of `op`'s type (the caller's
        // contract); the library reads only its shape and dtype fields, ignores its pointers and
        // needs no context (header v2.4, like `_supported`).
        unsafe { (fns.supports)(op.abi_code(), index, std::ptr::from_ref(desc).cast()) == 1 }
    }

    /// Creates a context on `device`. A device whose architecture is not in `build_archs` is
    /// refused before `turbine_ctx_create(vendor_index)` is called.
    pub fn create_context(
        self: &Arc<Self>,
        device: &DeviceInfo,
    ) -> Result<Arc<ShimContext>, KernelError> {
        let device_arch = device.arch.as_deref().unwrap_or("unknown");
        if !self.archs.iter().any(|a| a == device_arch) {
            return Err(KernelError::ArchMismatch {
                device_arch: device_arch.to_string(),
                build_archs: self.archs.join(","),
            });
        }
        let ordinal =
            i32::try_from(device.vendor_index).map_err(|_| KernelError::InvalidArgument {
                message: format!(
                    "vendor_index {} does not fit the C ABI",
                    device.vendor_index
                ),
            })?;
        let mut raw: *mut TurbineCtx = std::ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer on this stack frame. On success the shim stores a
        // context it allocated; ownership passes to the `ShimContext` below, which destroys it
        // exactly once. On failure `turbine_last_error(NULL)` reports this thread's message.
        let code = unsafe { (self.syms.ctx_create)(ordinal, &mut raw) };
        ffi::check(code, &self.syms, std::ptr::null_mut())?;
        if raw.is_null() {
            return Err(KernelError::Library {
                message: "turbine_ctx_create succeeded but returned a null context".into(),
            });
        }
        let mut info = CtxInfo::zeroed();
        // SAFETY: `raw` is the live context created above and `info` a live, writable
        // `turbine_ctx_info` on this stack frame; the shim fills it and keeps no pointer to it.
        let code = unsafe { (self.syms.ctx_get_info)(raw, &mut info) };
        if let Err(e) = ffi::check(code, &self.syms, raw) {
            // SAFETY: `raw` came from `turbine_ctx_create` above, was never shared, and is
            // destroyed exactly once, here, because no `ShimContext` took ownership of it.
            unsafe { (self.syms.ctx_destroy)(raw) };
            return Err(e);
        }
        let info = ContextInfo::from_raw(&info);
        let mut native_compute: *mut c_void = std::ptr::null_mut();
        if let Some(tp) = self.syms.v21.tensor_parallel {
            // SAFETY: `raw` is the live context created above; a null stream names its compute
            // stream, and `native_compute` is a live out-pointer on this stack frame. The handle
            // stays owned by the context (header v2.6) and is only carried as an integer.
            let code = unsafe {
                (tp.stream_native_handle)(raw, std::ptr::null_mut(), &mut native_compute)
            };
            if let Err(e) = ffi::check(code, &self.syms, raw) {
                // SAFETY: as above: `raw` was never shared and no `ShimContext` owns it yet, so it
                // is destroyed exactly once, here.
                unsafe { (self.syms.ctx_destroy)(raw) };
                return Err(e);
            }
        }
        Ok(Arc::new_cyclic(|weak| ShimContext {
            raw,
            native_compute: native_compute as u64,
            lib: Arc::clone(self),
            device: device.index,
            info,
            staging: Mutex::new(StagingTable::default()),
            self_ref: weak.clone(),
            card: OnceLock::new(),
            pinned: PinnedState::default(),
            gemm_prefill: AtomicU8::new(0),
        }))
    }
}

/// Fixed properties of a shim context (`turbine_ctx_get_info`), read once at creation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ContextInfo {
    /// Kernel workspace the context allocated at creation.
    pub workspace_bytes: u64,
    /// `(major, minor)` compute capability; `None` where the vendor has none (AMD).
    pub compute_capability: Option<(i32, i32)>,
    /// Device architecture as the runtime names it, e.g. `gfx1201` or `sm_121`.
    pub device_arch: String,
}

impl ContextInfo {
    fn from_raw(raw: &CtxInfo) -> ContextInfo {
        let bytes: Vec<u8> = raw.device_arch.iter().map(|&c| c as u8).collect();
        let device_arch = match CStr::from_bytes_until_nul(&bytes) {
            Ok(s) => s.to_string_lossy().into_owned(),
            // No NUL within the 32 bytes: take all of them.
            Err(_) => String::from_utf8_lossy(&bytes).into_owned(),
        };
        let compute_capability = (raw.compute_major >= 0 && raw.compute_minor >= 0)
            .then_some((raw.compute_major, raw.compute_minor));
        ContextInfo {
            workspace_bytes: raw.workspace_bytes,
            compute_capability,
            device_arch,
        }
    }
}

/// One shim context (`turbine_ctx*`) on one device: owns the device's compute stream, library
/// handles and workspace. The `DeviceMemory` backend for that device.
pub struct ShimContext {
    raw: *mut TurbineCtx,
    /// The device runtime's handle of the compute stream (ABI v2.6
    /// `turbine_stream_native_handle`), 0 when the library lacks the v2.6 group. Owned by the
    /// context; only ever handed out as an integer inside a `StreamRef`, which holds the context.
    native_compute: u64,
    lib: Arc<ShimLibrary>,
    device: DeviceId,
    info: ContextInfo,
    /// Host staging buffers (ABI v2.3) by id; see the module's ownership rules.
    staging: Mutex<StagingTable>,
    /// Lets `compute_stream` hand out an owning `Arc` of this context.
    self_ref: Weak<ShimContext>,
    /// The card profile of `set_profile` (ABI v2.4).
    card: OnceLock<&'static CardProfile>,
    /// Phase 4 pinned buffers, the copy stream and copy events (ABI v2.3 + v2.5; `pinned`).
    pinned: PinnedState,
    /// The step kind last handed to the library as `TURBINE_OPTION_GEMM_PREFILL` (0 decode, the
    /// library default; 1 prefill), or `GEMM_PREFILL_UNSUPPORTED`.
    gemm_prefill: AtomicU8,
}

/// The staging buffers of one context and the next id.
#[derive(Default)]
struct StagingTable {
    next: u64,
    bufs: HashMap<u64, Staged>,
}

/// One page-locked host buffer and the event recorded after its latest staged copy.
struct Staged {
    host: *mut u8,
    len: usize,
    event: *mut TurbineEvent,
    /// A staged copy was enqueued since the host last waited on `event`.
    pending: bool,
}

impl fmt::Debug for ShimContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShimContext")
            .field("device", &self.device)
            .field("library", &self.lib.path)
            .finish_non_exhaustive()
    }
}

// SAFETY: the context pointer is never dereferenced on the Rust side; it is only passed to the
// shim, whose entry points may be called from any thread (contract §9.4: one engine thread issues
// the work of a context, other threads only query it). All shared state lives in the shim.
unsafe impl Send for ShimContext {}
// SAFETY: every method takes `&self` and forwards to shim entry points with the same pointer;
// see the `Send` impl.
unsafe impl Sync for ShimContext {}

impl Drop for ShimContext {
    fn drop(&mut self) {
        // The copy stream (waiting for its copies) and copy events go first: they belong to
        // this context. No `PinnedBuffer` is alive: each holds an `Arc` of the context.
        self.pinned.release();
        // SAFETY: `raw` came from `turbine_ctx_create` of this library and is destroyed exactly
        // once, here. No `DeviceBuffer` or `StreamRef` of this context is alive: each holds an
        // `Arc` of it. `self.lib` keeps the library loaded until after this call.
        unsafe { (self.lib.syms.ctx_destroy)(self.raw) };
    }
}

impl ShimContext {
    pub fn library(&self) -> &Arc<ShimLibrary> {
        &self.lib
    }

    /// The raw context pointer, for the crate's other FFI modules (`pinned`).
    pub(crate) fn raw_ctx(&self) -> *mut TurbineCtx {
        self.raw
    }

    pub(crate) fn pinned_state(&self) -> &PinnedState {
        &self.pinned
    }

    /// An owning `Arc` of this context (every context lives in the `Arc` `create_context`
    /// returns).
    pub(crate) fn owning_arc(&self) -> Arc<ShimContext> {
        self.self_ref
            .upgrade()
            .expect("a ShimContext only exists inside the Arc create_context returns")
    }

    /// Workspace size, compute capability and device arch of this context.
    pub fn info(&self) -> ContextInfo {
        self.info.clone()
    }

    /// True when [`DeviceMemory::compute_stream`] carries the device runtime's own stream handle
    /// (the library exports the ABI v2.6 group), so a collective library can enqueue on the
    /// compute stream in order with the ops; false (handle 0) otherwise, and tensor parallelism
    /// is refused on this library.
    pub fn has_native_streams(&self) -> bool {
        self.lib.exports_native_streams()
    }

    fn check(&self, code: i32) -> Result<(), KernelError> {
        ffi::check(code, &self.lib.syms, self.raw)
    }

    /// The v2.3 staging functions, or `Unsupported` naming the library.
    fn staging_fns(&self) -> Result<StagingFns, MemoryError> {
        self.lib.syms.v21.staging.ok_or_else(|| {
            MemoryError::Unsupported(format!(
                "{} does not export the kernel ABI v2.3 staging functions (minor {})",
                self.lib.path.display(),
                self.lib.syms.v21.minor
            ))
        })
    }

    /// Runs `f` on staging buffer `id` after checking `[offset, offset + len)` lies inside it,
    /// holding the table lock throughout.
    fn with_staged<R>(
        &self,
        id: StagingId,
        offset: usize,
        len: usize,
        f: impl FnOnce(&mut Staged) -> Result<R, MemoryError>,
    ) -> Result<R, MemoryError> {
        let mut table = self.staging.lock().unwrap_or_else(PoisonError::into_inner);
        let staged = table.bufs.get_mut(&id.0).ok_or_else(|| {
            MemoryError::InvalidArgument(format!("staging buffer {} is not allocated", id.0))
        })?;
        if offset.checked_add(len).is_none_or(|end| end > staged.len) {
            return Err(MemoryError::InvalidArgument(format!(
                "range {offset}+{len} outside staging buffer of {} bytes",
                staged.len
            )));
        }
        f(staged)
    }

    /// Waits until every staged copy of `staged` has completed (the event recorded after the
    /// latest one), and no longer.
    fn wait_staged(&self, fns: &StagingFns, staged: &mut Staged) -> Result<(), MemoryError> {
        if staged.pending {
            // SAFETY: `event` came from `turbine_event_create` on this context and lives until
            // `staging_free`; `raw` is a live context.
            let code = unsafe { (fns.event_synchronize)(self.raw, staged.event) };
            self.check(code)?;
            staged.pending = false;
        }
        Ok(())
    }

    /// Re-records `staged`'s event after the staged copy just enqueued.
    fn mark_staged(&self, fns: &StagingFns, staged: &mut Staged) -> Result<(), MemoryError> {
        // SAFETY: as in `wait_staged`; a null stream is the compute stream the copy went to.
        let code = unsafe { (fns.event_record)(self.raw, staged.event, std::ptr::null_mut()) };
        self.check(code)?;
        staged.pending = true;
        Ok(())
    }

    /// The device address of `v` for a descriptor, after checking the view's memory is this
    /// context (a host-backend or other-device pointer must never reach the shim).
    fn device_ptr(&self, name: &str, v: &TensorView<'_>) -> Result<*mut c_void, KernelError> {
        self.slice_ptr(name, &v.slice)
    }

    /// The device address of `s`, after the same ownership check as `device_ptr`.
    fn slice_ptr(&self, name: &str, s: &DeviceSlice<'_>) -> Result<*mut c_void, KernelError> {
        let owner = Arc::as_ptr(s.memory());
        if !std::ptr::addr_eq(owner, std::ptr::from_ref(self)) {
            return Err(KernelError::InvalidArgument {
                message: format!(
                    "{name} is not memory of this {} context on device {}",
                    self.lib.backend, self.device.0
                ),
            });
        }
        Ok(s.ptr().addr() as *mut c_void)
    }

    /// Sets context option `option` (ABI v2.1 `turbine_ctx_set_option`, e.g.
    /// `TURBINE_OPTION_GEMM_AUTOTUNE`). `Unsupported` when the library has no options or does
    /// not know `option`.
    pub fn set_option(&self, option: i32, value: i64) -> Result<(), KernelError> {
        let Some(options) = self.lib.syms.v21.options else {
            return Err(self.lib.lacks("turbine_ctx_set_option"));
        };
        // SAFETY: `raw` is a live context of this library; the call takes plain integers.
        let code = unsafe { (options.set)(self.raw, option, value) };
        self.check(code)
    }

    /// Hands a GEMM's step kind to the library (`TURBINE_OPTION_GEMM_PREFILL`) when it differs
    /// from the previous GEMM's. A library without the option (or without options) is told
    /// nothing again: its GEMMs all run as decode steps'.
    fn gemm_step(&self, prefill: bool) -> Result<(), KernelError> {
        let want = u8::from(prefill);
        let current = self.gemm_prefill.load(Ordering::Relaxed);
        if current == want || current == GEMM_PREFILL_UNSUPPORTED {
            return Ok(());
        }
        match self.set_option(TURBINE_OPTION_GEMM_PREFILL, i64::from(want)) {
            Ok(()) => self.gemm_prefill.store(want, Ordering::Relaxed),
            Err(KernelError::Unsupported { .. }) => self
                .gemm_prefill
                .store(GEMM_PREFILL_UNSUPPORTED, Ordering::Relaxed),
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// Reads context option `option` (ABI v2.1 `turbine_ctx_get_option`). `Unsupported` when the
    /// library has no options or does not know `option`.
    pub fn get_option(&self, option: i32) -> Result<i64, KernelError> {
        let Some(options) = self.lib.syms.v21.options else {
            return Err(self.lib.lacks("turbine_ctx_get_option"));
        };
        let mut value = 0i64;
        // SAFETY: `raw` is a live context of this library and `value` a live, writable host
        // integer on this stack frame; the shim writes it and keeps no pointer to it.
        let code = unsafe { (options.get)(self.raw, option, &mut value) };
        self.check(code)?;
        Ok(value)
    }

    /// Hands `profile`'s thresholds to the library (ABI v2.4 `turbine_ctx_set_profile`), with
    /// this context's device architecture; the defaults of the `turbine_<op>` entry points read
    /// them. A no-op `Ok` when the library does not enumerate implementations (it keeps its own
    /// choice); `Unsupported` when the library refuses the profile (wave size, LDS or arch).
    /// The first profile set is the context's [`ShimContext::card_profile`].
    pub fn set_profile(&self, profile: &'static CardProfile) -> Result<(), KernelError> {
        let Some(fns) = self.lib.syms.v21.impls else {
            let _ = self.card.set(profile);
            return Ok(());
        };
        let arch = std::ffi::CString::new(self.info.device_arch.as_str()).map_err(|_| {
            invalid(format!(
                "device arch {:?} contains a NUL byte",
                self.info.device_arch
            ))
        })?;
        let desc = ffi::CardProfileDesc {
            struct_bytes: std::mem::size_of::<ffi::CardProfileDesc>() as u32,
            arch: arch.as_ptr(),
            wave_size: to_i32("wave_size", profile.capabilities.wave_size)?,
            lds_bytes: to_i32("lds_bytes", profile.capabilities.lds_bytes)?,
            moe_small_max_rows: i64::from(profile.thresholds.moe_small_max_rows),
            paged_page_multiple: to_i32(
                "paged_page_multiple",
                profile.thresholds.paged_page_multiple,
            )?,
        };
        // SAFETY: `raw` is a live context of this library; `desc` and the string `arch` points
        // to live on this stack frame for the call, and the library copies what it keeps.
        let code = unsafe { (fns.set_profile)(self.raw, &desc) };
        self.check(code)?;
        let _ = self.card.set(profile);
        Ok(())
    }

    /// The card profile set on this context (`None` before [`ShimContext::set_profile`]).
    pub fn card_profile(&self) -> Option<&'static CardProfile> {
        self.card.get().copied()
    }

    fn graph_fns(&self) -> Result<ffi::GraphFns, KernelError> {
        self.lib
            .syms
            .v21
            .graph
            .ok_or_else(|| self.lib.lacks("the turbine_graph functions"))
    }

    /// Starts capturing this context's compute stream into a graph (ABI v2.1). Until
    /// `graph_end`, only op calls may be issued: the shim refuses copies, syncs and allocations.
    pub fn graph_begin(&self) -> Result<(), KernelError> {
        let fns = self.graph_fns()?;
        // SAFETY: `raw` is a live context of this library.
        let code = unsafe { (fns.begin)(self.raw) };
        self.check(code)
    }

    /// Stops the capture begun by `graph_begin` and instantiates it. The graph records the
    /// device pointers of the captured ops: the caller keeps those buffers alive until the
    /// returned handle is dropped.
    pub fn graph_end(&self) -> Result<GraphHandle, KernelError> {
        let fns = self.graph_fns()?;
        let mut raw: *mut TurbineGraph = std::ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer on this stack frame and `self.raw` a live context.
        // On success the shim stores a graph it allocated; ownership passes to the `GraphHandle`
        // below, which destroys it exactly once.
        let code = unsafe { (fns.end)(self.raw, &mut raw) };
        self.check(code)?;
        if raw.is_null() {
            return Err(KernelError::Library {
                message: "turbine_graph_end succeeded but returned a null graph".into(),
            });
        }
        let ctx = self
            .self_ref
            .upgrade()
            .expect("a ShimContext only exists inside the Arc create_context returns");
        Ok(GraphHandle { raw, ctx })
    }

    /// Enqueues the captured graph `g` on this context's compute stream.
    pub fn graph_launch(&self, g: &GraphHandle) -> Result<(), KernelError> {
        let fns = self.graph_fns()?;
        if !std::ptr::addr_eq(Arc::as_ptr(&g.ctx), std::ptr::from_ref(self)) {
            return Err(KernelError::InvalidArgument {
                message: format!(
                    "graph was captured on another context than this {} context on device {}",
                    self.lib.backend, self.device.0
                ),
            });
        }
        // SAFETY: `g.raw` was created by `turbine_graph_end` on this context (checked above) and
        // is not destroyed while `g` is borrowed; `raw` is a live context.
        let code = unsafe { (fns.launch)(self.raw, g.raw) };
        self.check(code)
    }
}

/// A graph captured on a shim context (ABI v2.1), destroyed through that context on drop.
pub struct GraphHandle {
    raw: *mut TurbineGraph,
    /// Keeps the context (and its library) alive until the graph is destroyed.
    ctx: Arc<ShimContext>,
}

impl fmt::Debug for GraphHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphHandle")
            .field("device", &self.ctx.device)
            .finish_non_exhaustive()
    }
}

// SAFETY: the graph pointer is never dereferenced on the Rust side; it is only passed to the shim
// together with its context, whose entry points may be called from any thread (see `ShimContext`).
unsafe impl Send for GraphHandle {}
// SAFETY: no method mutates through `&self`; the pointer is only handed to the shim (see `Send`).
unsafe impl Sync for GraphHandle {}

impl Drop for GraphHandle {
    fn drop(&mut self) {
        let Some(fns) = self.ctx.lib.syms.v21.graph else {
            // Unreachable: a handle only comes from `graph_end`, which needs the graph functions.
            return;
        };
        // SAFETY: `raw` came from `turbine_graph_end` on `ctx` and is destroyed exactly once,
        // here; `self.ctx` keeps the context and its library alive through the call.
        let code = unsafe { (fns.destroy)(self.ctx.raw, self.raw) };
        if let Err(e) = self.ctx.check(code) {
            tracing::warn!(
                event = "graph_destroy_failed",
                device = self.ctx.device.0,
                error = %e,
                "turbine_graph_destroy failed"
            );
        }
    }
}

impl DeviceMemory for ShimContext {
    fn device(&self) -> DeviceId {
        self.device
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError> {
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: `out` is a live out-pointer; the shim allocates on this context's device and
        // the pointer becomes owned by exactly one `DeviceBuffer`, which frees it through `free`.
        let code = unsafe { (self.lib.syms.malloc)(self.raw, bytes, &mut out) };
        match self.check(code) {
            Ok(()) => Ok(DevicePtr::from_addr(out as u64)),
            Err(KernelError::OutOfMemory { .. }) => Err(MemoryError::OutOfMemory {
                requested: bytes as u64,
            }),
            Err(e) => Err(e.into()),
        }
    }

    fn free(&self, ptr: DevicePtr) {
        // SAFETY: `ptr` came from `turbine_malloc` on this context and is freed once, by the
        // `DeviceBuffer` that owns it (the only caller of `free`).
        let code = unsafe { (self.lib.syms.free)(self.raw, ptr.addr() as *mut c_void) };
        if let Err(e) = self.check(code) {
            tracing::warn!(
                event = "device_free_failed",
                device = self.device.0,
                error = %e,
                "turbine_free failed"
            );
        }
    }

    fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError> {
        // SAFETY: `src` is a live host slice of `src.len()` bytes and `dst` a device range the
        // caller (`DeviceBuffer`/`DeviceSlice`) bounds-checked. The copy may be asynchronous, so
        // the stream is synchronized below before `src` stops being borrowed.
        let code = unsafe {
            (self.lib.syms.memcpy_h2d)(
                self.raw,
                dst.addr() as *mut c_void,
                src.as_ptr().cast(),
                src.len(),
            )
        };
        self.check(code)?;
        self.synchronize()
    }

    fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError> {
        // SAFETY: `dst` is a live, exclusively borrowed host slice of `dst.len()` bytes and `src`
        // a bounds-checked device range. The stream is synchronized below, so the write completes
        // while `dst` is still borrowed.
        let code = unsafe {
            (self.lib.syms.memcpy_d2h)(
                self.raw,
                dst.as_mut_ptr().cast(),
                src.addr() as *const c_void,
                dst.len(),
            )
        };
        self.check(code)?;
        self.synchronize()
    }

    /// Kernel ABI v2.5: `turbine_memcpy_async` of kind D2D on the compute stream (the null
    /// stream argument), so the copy is ordered with the ops around it and nothing waits for
    /// it (the tensor-parallel logits reorder, Phase 5). A library without the v2.5 copy group
    /// answers `Unsupported`; the library refuses it while the compute stream is captured.
    fn copy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<(), MemoryError> {
        let copies = self.lib.syms.v21.copies.ok_or_else(|| {
            MemoryError::Unsupported(format!(
                "device-to-device copy needs the kernel ABI v2.5 copy group, which {} does not \
                 export (minor {})",
                self.lib.path.display(),
                self.lib.syms.v21.minor
            ))
        })?;
        // SAFETY: `dst` and `src` are device addresses of `bytes`-long ranges inside live
        // allocations of this context (callers derive them from bounds-checked `DeviceSlice`s
        // that outlive the call); the copy is enqueued on the compute stream, so it completes
        // before any later work on that stream and before the buffers can be freed (freeing
        // synchronises through the same context).
        let code = unsafe {
            (copies.memcpy_async)(
                self.raw,
                std::ptr::null_mut(),
                dst.addr() as *mut c_void,
                src.addr() as *const c_void,
                bytes,
                ffi::COPY_D2D,
            )
        };
        Ok(self.check(code)?)
    }

    fn synchronize(&self) -> Result<(), MemoryError> {
        // SAFETY: `raw` is a live context of this library.
        let code = unsafe { (self.lib.syms.stream_sync)(self.raw) };
        Ok(self.check(code)?)
    }

    fn mem_info(&self) -> Result<MemInfo, MemoryError> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both out-pointers point at live stack integers; `raw` is a live context.
        let code = unsafe { (self.lib.syms.mem_info)(self.raw, &mut free, &mut total) };
        self.check(code)?;
        Ok(MemInfo {
            free_bytes: free as u64,
            total_bytes: total as u64,
        })
    }

    fn mapped_collectives(&self) -> Option<&dyn MappedCollectives> {
        self.has_mapped_collectives()
            .then_some(self as &dyn MappedCollectives)
    }

    fn compute_stream(&self) -> StreamRef {
        let owner: Arc<dyn DeviceMemory> = self
            .self_ref
            .upgrade()
            .expect("a ShimContext only exists inside the Arc create_context returns");
        StreamRef::new(self.native_compute, self.device, owner)
    }

    fn staging_alloc(&self, bytes: usize) -> Result<StagingId, MemoryError> {
        let fns = self.staging_fns()?;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` is a live out-pointer; on success the shim stores page-locked memory it
        // allocated, owned from here by this context's staging table.
        let code = unsafe { (fns.host_alloc)(self.raw, bytes, &mut host) };
        self.check(code)?;
        let mut event: *mut TurbineEvent = std::ptr::null_mut();
        // SAFETY: `event` is a live out-pointer; on success the shim stores an event it created,
        // owned from here by the staging table together with `host`.
        let code = unsafe { (fns.event_create)(self.raw, &mut event) };
        if let Err(e) = self.check(code).and_then(|()| {
            if host.is_null() || event.is_null() {
                Err(KernelError::Library {
                    message: "turbine_host_alloc_pinned or turbine_event_create returned null"
                        .into(),
                })
            } else {
                Ok(())
            }
        }) {
            // SAFETY: `host` (when not null) came from `turbine_host_alloc_pinned` above, no copy used
            // it, and it is freed exactly once, here, because no table entry took it; likewise
            // `event` from `turbine_event_create`.
            unsafe {
                if !event.is_null() {
                    (fns.event_destroy)(self.raw, event);
                }
                (fns.host_free)(self.raw, host);
            }
            return Err(e.into());
        }
        let mut table = self.staging.lock().unwrap_or_else(PoisonError::into_inner);
        table.next += 1;
        let id = table.next;
        table.bufs.insert(
            id,
            Staged {
                host: host.cast(),
                len: bytes,
                event,
                pending: false,
            },
        );
        Ok(StagingId(id))
    }

    fn staging_free(&self, id: StagingId) {
        let Ok(fns) = self.staging_fns() else {
            return;
        };
        let removed = self
            .staging
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .bufs
            .remove(&id.0);
        let Some(mut staged) = removed else {
            return;
        };
        if let Err(e) = self.wait_staged(&fns, &mut staged) {
            tracing::warn!(event = "staging_free_failed", device = self.device.0, error = %e, "waiting for a staging buffer's copies failed");
        }
        // SAFETY: `host` and `event` came from `turbine_host_alloc_pinned` / `turbine_event_create` on
        // this context, were removed from the table above (so this is their only release), and
        // no staged copy still uses them (waited above; after a device error the context is
        // unusable and the stream no longer runs).
        let codes = unsafe {
            [
                (fns.event_destroy)(self.raw, staged.event),
                (fns.host_free)(self.raw, staged.host.cast()),
            ]
        };
        for code in codes {
            if let Err(e) = self.check(code) {
                tracing::warn!(event = "staging_free_failed", device = self.device.0, error = %e, "releasing a staging buffer failed");
            }
        }
    }

    fn staging_write(&self, id: StagingId, offset: usize, src: &[u8]) -> Result<(), MemoryError> {
        let fns = self.staging_fns()?;
        self.with_staged(id, offset, src.len(), |staged| {
            self.wait_staged(&fns, staged)?;
            // SAFETY: `staged.host` is a live allocation of `staged.len` bytes and
            // `[offset, offset + src.len())` lies inside it (checked by `with_staged`); no staged
            // copy uses it (waited above) and the table lock excludes every other host access.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), staged.host.add(offset), src.len());
            }
            Ok(())
        })
    }

    fn staging_read(
        &self,
        id: StagingId,
        offset: usize,
        dst: &mut [u8],
    ) -> Result<(), MemoryError> {
        let fns = self.staging_fns()?;
        self.with_staged(id, offset, dst.len(), |staged| {
            self.wait_staged(&fns, staged)?;
            // SAFETY: as in `staging_write`, reading into the exclusively borrowed `dst`.
            unsafe {
                std::ptr::copy_nonoverlapping(staged.host.add(offset), dst.as_mut_ptr(), dst.len());
            }
            Ok(())
        })
    }

    fn copy_h2d_staged(
        &self,
        dst: DevicePtr,
        id: StagingId,
        offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let fns = self.staging_fns()?;
        self.with_staged(id, offset, bytes, |staged| {
            // SAFETY: the source range lies inside the live page-locked buffer (checked by
            // `with_staged`) and `dst` is a device range the caller (`HostStaging::upload`)
            // bounds-checked. The copy runs asynchronously: the event recorded next marks its
            // completion, and every host access to the buffer waits on it first.
            let code = unsafe {
                (self.lib.syms.memcpy_h2d)(
                    self.raw,
                    dst.addr() as *mut c_void,
                    staged.host.add(offset).cast(),
                    bytes,
                )
            };
            self.check(code)?;
            self.mark_staged(&fns, staged)
        })
    }

    fn copy_d2h_staged(
        &self,
        id: StagingId,
        offset: usize,
        src: DevicePtr,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let fns = self.staging_fns()?;
        self.with_staged(id, offset, bytes, |staged| {
            // SAFETY: as in `copy_h2d_staged`, with the buffer as the destination: no host access
            // reads it before the event recorded next has completed.
            let code = unsafe {
                (self.lib.syms.memcpy_d2h)(
                    self.raw,
                    staged.host.add(offset).cast(),
                    src.addr() as *const c_void,
                    bytes,
                )
            };
            self.check(code)?;
            self.mark_staged(&fns, staged)
        })
    }
}

/// The kernel provider of a shim context: `supports` → `turbine_<op>_supported` (null
/// pointers), `implementation` → `turbine_<op>_impl`, `execute` → `turbine_<op>`, where the
/// library chooses the implementation. A provider bound to one op config
/// ([`KernelProvider::bind`], kernel ABI v2.4) runs that op through
/// `turbine_impl_run(index)` with the index the kernel registry chose instead: one branch per
/// call (plus one comparison per routed-row tier for `moe_experts`), no lookup by name.
pub struct ShimProvider {
    ctx: Arc<ShimContext>,
    bound: Option<Bound>,
}

/// What a bound provider runs for its op.
struct Bound {
    op: OpKind,
    choice: ImplChoice,
    /// The library's implementations of `op`, indexed as `choice` indexes them.
    impls: Vec<ImplInfo>,
}

impl Bound {
    /// The implementation a call of `rows` routed rows runs.
    fn info(&self, rows: usize) -> &ImplInfo {
        &self.impls[self.choice.index_for(rows) as usize]
    }
}

/// The provider of `ctx`; its id is the library's backend name (`hip`, `cuda`).
pub fn shim_provider(ctx: Arc<ShimContext>) -> Arc<dyn KernelProvider> {
    Arc::new(ShimProvider { ctx, bound: None })
}

fn invalid(message: String) -> KernelError {
    KernelError::InvalidArgument { message }
}

fn to_i64(name: &str, v: impl TryInto<i64> + Copy + fmt::Display) -> Result<i64, KernelError> {
    v.try_into()
        .map_err(|_| invalid(format!("{name} = {v} does not fit int64_t")))
}

fn to_i32(name: &str, v: impl TryInto<i32> + Copy + fmt::Display) -> Result<i32, KernelError> {
    v.try_into()
        .map_err(|_| invalid(format!("{name} = {v} does not fit int32_t")))
}

/// Checks that `v` has `rank` dimensions and that every dimension after the first is dense
/// (the C ABI describes a view by its row stride only), and returns that row stride.
fn row_stride(name: &str, v: &TensorView<'_>, rank: usize) -> Result<i64, KernelError> {
    // A rank-0 view has no rows; reject it rather than index `strides[0]`.
    if rank == 0 || v.shape.len() != rank || v.strides.len() != rank {
        return Err(invalid(format!(
            "{name} has shape {:?}, expected rank {rank} (at least 1)",
            v.shape.as_slice()
        )));
    }
    let dense = contiguous_strides(&v.shape);
    if v.strides[1..] != dense[1..] {
        return Err(invalid(format!(
            "{name} has strides {:?}; only the first dimension may be strided",
            v.strides.as_slice()
        )));
    }
    to_i64(&format!("{name} row stride"), v.strides[0])
}

/// Checks that `v` is a dense view of exactly `shape` and `dtype` (for descriptor fields that
/// carry no strides).
fn dense(name: &str, v: &TensorView<'_>, shape: &[usize], dtype: DType) -> Result<(), KernelError> {
    if v.shape.as_slice() != shape || v.dtype != dtype || v.strides != contiguous_strides(shape) {
        return Err(invalid(format!(
            "{name} must be a dense {} view of shape {shape:?}, has {} shape {:?} strides {:?}",
            dtype.as_str(),
            v.dtype.as_str(),
            v.shape.as_slice(),
            v.strides.as_slice()
        )));
    }
    Ok(())
}

fn null() -> *mut c_void {
    std::ptr::null_mut()
}

impl ShimProvider {
    fn syms(&self) -> &ShimSymbols {
        &self.ctx.lib.syms
    }

    fn supported<D>(trio: &OpTrio<D>, d: &D) -> bool {
        // SAFETY: `d` is a fully initialised descriptor; `_supported` reads only its shape and
        // dtype fields, ignores its (null) pointers and needs no context (contract §9.2).
        unsafe { (trio.supported)(d) == 1 }
    }

    /// The implementation `op` runs at `d`: the bound one (its first tier's), else the
    /// library's `turbine_<op>_impl`.
    fn implementation_of<D>(&self, op: OpKind, trio: &OpTrio<D>, d: &D) -> String {
        if let Some(bound) = self.bound.as_ref().filter(|b| b.op == op) {
            return bound.info(1).name.clone();
        }
        // SAFETY: as for `_supported`; the returned string is static and owned by the library.
        ffi::c_str(unsafe { (trio.implementation)(d) })
    }

    /// Enqueues `op` with descriptor `d`: through `turbine_impl_run` with the bound index when
    /// this provider is bound to `op` (`rows`: the routed rows of a `moe_experts` call, else
    /// ignored), else through the library's entry point `trio.run`.
    fn run<D>(&self, op: OpKind, trio: &OpTrio<D>, d: &D, rows: usize) -> Result<(), KernelError> {
        if let Some(bound) = self.bound.as_ref().filter(|b| b.op == op) {
            let fns = self
                .syms()
                .v21
                .impls
                .ok_or_else(|| self.ctx.lib.lacks("turbine_impl_run"))?;
            let index = to_i32("implementation index", bound.choice.index_for(rows))?;
            // SAFETY: as for the entry point below; `d` is `op`'s descriptor type, and `index`
            // is one of the library's implementations of `op` (the registry chose it from
            // `turbine_impl_info`). The library runs exactly that implementation.
            let code = unsafe {
                (fns.run)(
                    self.ctx.raw,
                    op.abi_code(),
                    index,
                    std::ptr::from_ref(d).cast(),
                )
            };
            return self.ctx.check(code);
        }
        // SAFETY: every pointer in `d` comes from `ShimContext::device_ptr`, so it is memory of
        // this context whose view covers the shape and strides encoded in `d` (the views were
        // bounds-checked when built). The shim enqueues on the context stream and keeps no
        // pointer beyond the call; the views borrow their buffers for the whole call.
        let code = unsafe { (trio.run)(self.ctx.raw, d) };
        self.ctx.check(code)
    }

    /// The contiguous entry point for `kind`; `None` for the paged kinds. The match lists every
    /// kind so a new one cannot silently fall into another kind's entry point.
    fn attention_trio(&self, kind: AttentionKind) -> Option<&OpTrio<AttentionDesc>> {
        match kind {
            AttentionKind::Prefill => Some(&self.syms().attention_prefill),
            AttentionKind::Decode => Some(&self.syms().attention_decode),
            AttentionKind::PrefillPaged | AttentionKind::DecodePaged => None,
        }
    }

    /// The paged entry point for `kind`; `None` for the contiguous kinds.
    fn paged_trio(&self, kind: AttentionKind) -> Option<&OpTrio<AttentionPagedDesc>> {
        match kind {
            AttentionKind::PrefillPaged => Some(&self.syms().attention_prefill_paged),
            AttentionKind::DecodePaged => Some(&self.syms().attention_decode_paged),
            AttentionKind::Prefill | AttentionKind::Decode => None,
        }
    }
}

// Probe descriptors for `supports`/`implementation`: the config's shapes and dtypes with null
// pointers and dense strides; the per-call dimension (tokens, rows) is 1.

fn gemm_probe(cfg: &GemmConfig) -> GemmDesc {
    let (n, k) = (cfg.n as i64, cfg.k as i64);
    GemmDesc {
        a: null(),
        b: null(),
        c: null(),
        m: 1,
        n,
        k,
        lda: k,
        ldb: if cfg.trans_b { k } else { n },
        ldc: n,
        trans_b: i32::from(cfg.trans_b),
        a_dtype: cfg.a_dtype.abi_code(),
        b_dtype: cfg.b_dtype.abi_code(),
        c_dtype: cfg.c_dtype.abi_code(),
        alpha: 1.0,
        beta: 0.0,
    }
}

fn attention_probe(cfg: &AttentionConfig) -> AttentionDesc {
    let (hq, hkv, d) = (
        i64::from(cfg.num_q_heads),
        i64::from(cfg.num_kv_heads),
        i64::from(cfg.head_dim),
    );
    AttentionDesc {
        q: null(),
        k_cache: null(),
        v_cache: null(),
        out: null(),
        q_len: 1,
        q_start: 0,
        num_q_heads: cfg.num_q_heads as i32,
        num_kv_heads: cfg.num_kv_heads as i32,
        head_dim: cfg.head_dim as i32,
        q_stride_token: hq * d,
        kv_stride_token: hkv * d,
        out_stride_token: hq * d,
        scale: 1.0 / (cfg.head_dim.max(1) as f32).sqrt(),
        causal: i32::from(cfg.causal),
        dtype: cfg.dtype.abi_code(),
    }
}

fn rmsnorm_probe(cfg: &NormConfig) -> RmsnormDesc {
    let dim = cfg.dim as i64;
    RmsnormDesc {
        x: null(),
        weight: null(),
        out: null(),
        rows: 1,
        dim,
        x_stride_row: dim,
        out_stride_row: dim,
        eps: 1e-5,
        dtype: cfg.dtype.abi_code(),
    }
}

fn rope_probe(cfg: &RopeConfig) -> RopeDesc {
    let d = i64::from(cfg.head_dim);
    RopeDesc {
        q: null(),
        k: null(),
        positions: std::ptr::null(),
        inv_freq: std::ptr::null(),
        num_tokens: 1,
        num_q_heads: cfg.num_q_heads as i32,
        num_kv_heads: cfg.num_kv_heads as i32,
        head_dim: cfg.head_dim as i32,
        rotary_dim: cfg.rotary_dim as i32,
        q_stride_token: i64::from(cfg.num_q_heads) * d,
        k_stride_token: i64::from(cfg.num_kv_heads) * d,
        style: 0,
        dtype: cfg.dtype.abi_code(),
    }
}

fn silu_mul_probe(cfg: &ActivationConfig) -> SiluMulDesc {
    let cols = cfg.cols as i64;
    SiluMulDesc {
        gate: null(),
        up: null(),
        out: null(),
        rows: 1,
        cols,
        gate_stride_row: cols,
        up_stride_row: cols,
        out_stride_row: cols,
        dtype: cfg.dtype.abi_code(),
    }
}

fn embedding_probe(cfg: &EmbeddingConfig) -> EmbeddingDesc {
    let hidden = cfg.hidden as i64;
    EmbeddingDesc {
        ids: std::ptr::null(),
        table: null(),
        out: null(),
        num_tokens: 1,
        hidden,
        vocab_offset: 0,
        vocab_rows: cfg.vocab_rows as i64,
        out_stride_row: hidden,
        dtype: cfg.dtype.abi_code(),
    }
}

fn paged_probe(cfg: &AttentionConfig) -> AttentionPagedDesc {
    let (hq, hkv, d) = (
        i64::from(cfg.num_q_heads),
        i64::from(cfg.num_kv_heads),
        i64::from(cfg.head_dim),
    );
    AttentionPagedDesc {
        q: null(),
        k_new: null(),
        v_new: null(),
        out: null(),
        kv_layer: null(),
        block_table: std::ptr::null(),
        q_indptr: std::ptr::null(),
        kv_lens: std::ptr::null(),
        num_seqs: 1,
        total_q: 1,
        max_q_len: 1,
        max_kv_len: 1,
        max_blocks_per_seq: 1,
        num_blocks: 1,
        block_tokens: cfg.block_tokens.unwrap_or(0) as i32,
        num_q_heads: cfg.num_q_heads as i32,
        num_kv_heads: cfg.num_kv_heads as i32,
        head_dim: cfg.head_dim as i32,
        q_stride_token: hq * d,
        new_stride_token: hkv * d,
        out_stride_token: hq * d,
        scale: 1.0 / (cfg.head_dim.max(1) as f32).sqrt(),
        causal: i32::from(cfg.causal),
        dtype: cfg.dtype.abi_code(),
    }
}

fn copy_blocks_probe(cfg: &KvCopyConfig) -> CopyBlocksDesc {
    CopyBlocksDesc {
        pool: null(),
        layer_stride_bytes: cfg.block_bytes as i64,
        block_bytes: cfg.block_bytes as i64,
        num_layers: cfg.num_layers as i32,
        src_blocks: std::ptr::null(),
        dst_blocks: std::ptr::null(),
        count: 1,
    }
}

/// The `turbine_moe_route_desc` `flags` of `cfg`.
fn moe_route_flags(cfg: &MoeRouteConfig) -> i32 {
    let mut flags = 0;
    if cfg.renormalize {
        flags |= MOE_ROUTE_RENORMALIZE;
    }
    if cfg.bf16_logits {
        flags |= MOE_ROUTE_BF16_LOGITS;
    }
    flags
}

fn moe_route_probe(cfg: &MoeRouteConfig) -> MoeRouteDesc {
    MoeRouteDesc {
        router_logits: std::ptr::null(),
        num_tokens: 1,
        num_experts: cfg.num_experts as i32,
        top_k: cfg.top_k as i32,
        flags: moe_route_flags(cfg),
        topk_ids: std::ptr::null_mut(),
        topk_weights: std::ptr::null_mut(),
        sorted_rows: std::ptr::null_mut(),
        expert_offsets: std::ptr::null_mut(),
    }
}

fn moe_experts_probe(cfg: &MoeExpertsConfig) -> MoeExpertsDesc {
    moe_experts_probe_tokens(cfg, 1)
}

/// The probe of `moe_experts` for a call over `num_tokens` tokens.
fn moe_experts_probe_tokens(cfg: &MoeExpertsConfig, num_tokens: i32) -> MoeExpertsDesc {
    MoeExpertsDesc {
        x: null(),
        w_gate: null(),
        w_up: null(),
        w_down: null(),
        sorted_rows: std::ptr::null(),
        expert_offsets: std::ptr::null(),
        topk_weights: std::ptr::null(),
        host_expert_offsets: std::ptr::null(),
        out: null(),
        workspace: null(),
        workspace_bytes: 0,
        num_tokens,
        hidden: cfg.hidden as i32,
        inter: cfg.inter as i32,
        top_k: cfg.top_k as i32,
        num_experts: cfg.num_experts as i32,
        expert_begin: cfg.expert_begin as i32,
        expert_end: cfg.expert_end as i32,
        dtype: cfg.dtype.abi_code(),
    }
}

fn add_probe(cfg: &ElementwiseConfig) -> AddDesc {
    AddDesc {
        a: null(),
        b: null(),
        out: null(),
        n: 1,
        dtype: cfg.dtype.abi_code(),
    }
}

fn add_rmsnorm_probe(cfg: &AddRmsnormConfig) -> AddRmsnormDesc {
    let dim = i64::from(cfg.dim);
    AddRmsnormDesc {
        residual: null(),
        x: null(),
        weight: null(),
        out: null(),
        rows: 1,
        dim,
        residual_stride_row: dim,
        x_stride_row: dim,
        out_stride_row: dim,
        eps: 1e-5,
        dtype: cfg.dtype.abi_code(),
    }
}

fn row_sumsq_probe(cfg: &RowSumsqConfig) -> RowSumsqDesc {
    let dim = i64::from(cfg.dim);
    RowSumsqDesc {
        x: std::ptr::null(),
        sumsq: std::ptr::null_mut(),
        rows: 1,
        dim,
        x_stride_row: dim,
        dtype: cfg.dtype.abi_code(),
    }
}

fn rmsnorm_sharded_probe(cfg: &RmsnormShardedConfig) -> RmsnormShardedDesc {
    let dim = i64::from(cfg.dim);
    RmsnormShardedDesc {
        x: std::ptr::null(),
        weight: std::ptr::null(),
        sumsq: std::ptr::null(),
        out: null(),
        rows: 1,
        dim,
        full_dim: i64::from(cfg.full_dim),
        x_stride_row: dim,
        out_stride_row: dim,
        eps: 1e-5,
        dtype: cfg.dtype.abi_code(),
    }
}

fn logits_reduce_probe(cfg: &LogitsReduceConfig) -> LogitsReduceDesc {
    let vocab = i64::from(cfg.vocab);
    LogitsReduceDesc {
        logits: std::ptr::null(),
        rows: 1,
        vocab,
        stride_row: vocab,
        temperature: std::ptr::null(),
        uniform: std::ptr::null(),
        mode: std::ptr::null(),
        top_n: cfg.top_n as i32,
        top_ids: std::ptr::null_mut(),
        top_values: std::ptr::null_mut(),
        lse: std::ptr::null_mut(),
        sampled: std::ptr::null_mut(),
        sampled_logit: std::ptr::null_mut(),
        top_p: std::ptr::null(),
    }
}

/// Checks that `v` is a dense `dtype` view of rank `1 + cols.is_some()` with at least `rows` rows
/// (and exactly `cols` columns): the per-row arrays of `logits_reduce`, which may be sized for
/// more rows than one call reduces.
fn dense_rows(
    name: &str,
    v: &TensorView<'_>,
    rows: usize,
    cols: Option<usize>,
    dtype: DType,
) -> Result<(), KernelError> {
    let rank = 1 + usize::from(cols.is_some());
    if v.dtype != dtype
        || v.shape.len() != rank
        || v.shape[0] < rows
        || cols.is_some_and(|c| v.shape[1] != c)
        || v.strides != contiguous_strides(&v.shape)
    {
        return Err(invalid(format!(
            "{name} must be a dense {} view of at least {rows} rows{}, has {} shape {:?} strides {:?}",
            dtype.as_str(),
            cols.map_or(String::new(), |c| format!(" of {c} columns")),
            v.dtype.as_str(),
            v.shape.as_slice(),
            v.strides.as_slice()
        )));
    }
    Ok(())
}

impl ShimProvider {
    /// The `add_rmsnorm` trio; `KernelProvider::add_rmsnorm` is `Some` exactly when it exists.
    fn add_rmsnorm_trio(&self) -> Result<&OpTrio<AddRmsnormDesc>, KernelError> {
        self.syms()
            .v21
            .add_rmsnorm
            .as_ref()
            .ok_or_else(|| self.ctx.lib.lacks("turbine_add_rmsnorm"))
    }

    /// The `logits_reduce` trio; `KernelProvider::logits_reduce` is `Some` exactly when it exists.
    fn logits_reduce_trio(&self) -> Result<&OpTrio<LogitsReduceDesc>, KernelError> {
        self.syms()
            .v21
            .logits_reduce
            .as_ref()
            .ok_or_else(|| self.ctx.lib.lacks("turbine_logits_reduce"))
    }
}

impl ShimProvider {
    /// The v2.6 group; `KernelProvider::sharded_norm` is `Some` exactly when it exists.
    fn tensor_parallel_fns(&self) -> Result<&ffi::TensorParallelFns, KernelError> {
        self.syms().v21.tensor_parallel.as_ref().ok_or_else(|| {
            self.ctx
                .lib
                .lacks("the ABI v2.6 row_sumsq / rmsnorm_sharded group")
        })
    }
}

/// Checks that `v` is a dense F32 view of shape `[rows]` (the per-row sums of squares).
fn sumsq_view(v: &TensorView<'_>, rows: usize) -> Result<(), KernelError> {
    dense("sumsq", v, &[rows], DType::F32)
}

impl ShardedNormKernel for ShimProvider {
    fn supports_row_sumsq(&self, cfg: &RowSumsqConfig) -> bool {
        self.tensor_parallel_fns()
            .is_ok_and(|f| Self::supported(&f.row_sumsq, &row_sumsq_probe(cfg)))
    }

    fn supports_rmsnorm_sharded(&self, cfg: &RmsnormShardedConfig) -> bool {
        cfg.full_dim >= cfg.dim
            && self
                .tensor_parallel_fns()
                .is_ok_and(|f| Self::supported(&f.rmsnorm_sharded, &rmsnorm_sharded_probe(cfg)))
    }

    fn implementation_row_sumsq(&self, cfg: &RowSumsqConfig) -> String {
        self.tensor_parallel_fns()
            .map(|f| self.implementation_of(OpKind::RowSumsq, &f.row_sumsq, &row_sumsq_probe(cfg)))
            .unwrap_or_default()
    }

    fn implementation_rmsnorm_sharded(&self, cfg: &RmsnormShardedConfig) -> String {
        self.tensor_parallel_fns()
            .map(|f| {
                self.implementation_of(
                    OpKind::RmsnormSharded,
                    &f.rmsnorm_sharded,
                    &rmsnorm_sharded_probe(cfg),
                )
            })
            .unwrap_or_default()
    }

    fn row_sumsq(&self, ctx: &mut RowSumsqContext<'_>) -> Result<(), KernelError> {
        let fns = self.tensor_parallel_fns()?;
        let x_stride_row = row_stride("x", &ctx.x, 2)?;
        let (rows, dim) = (ctx.x.shape[0], ctx.x.shape[1]);
        sumsq_view(&ctx.sumsq, rows)?;
        let d = RowSumsqDesc {
            x: self.ctx.device_ptr("x", &ctx.x)?,
            sumsq: self.ctx.device_ptr("sumsq", &ctx.sumsq)?.cast(),
            rows: to_i64("rows", rows)?,
            dim: to_i64("dim", dim)?,
            x_stride_row,
            dtype: ctx.x.dtype.abi_code(),
        };
        self.run(OpKind::RowSumsq, &fns.row_sumsq, &d, 0)
    }

    fn rmsnorm_sharded(&self, ctx: &mut RmsnormShardedContext<'_>) -> Result<(), KernelError> {
        let fns = self.tensor_parallel_fns()?;
        let x_stride_row = row_stride("x", &ctx.x, 2)?;
        let (rows, dim) = (ctx.x.shape[0], ctx.x.shape[1]);
        if ctx.out.shape.as_slice() != [rows, dim] || ctx.out.dtype != ctx.x.dtype {
            return Err(invalid(format!(
                "out must be a {} view of shape [{rows}, {dim}], has {} shape {:?}",
                ctx.x.dtype.as_str(),
                ctx.out.dtype.as_str(),
                ctx.out.shape.as_slice()
            )));
        }
        dense("weight", &ctx.weight, &[dim], ctx.x.dtype)?;
        sumsq_view(&ctx.sumsq, rows)?;
        let d = RmsnormShardedDesc {
            out_stride_row: row_stride("out", &ctx.out, 2)?,
            x: self.ctx.device_ptr("x", &ctx.x)?,
            weight: self.ctx.device_ptr("weight", &ctx.weight)?,
            sumsq: self.ctx.device_ptr("sumsq", &ctx.sumsq)? as *const f32,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            rows: to_i64("rows", rows)?,
            dim: to_i64("dim", dim)?,
            full_dim: i64::from(ctx.full_dim),
            x_stride_row,
            eps: ctx.eps,
            dtype: ctx.x.dtype.abi_code(),
        };
        self.run(OpKind::RmsnormSharded, &fns.rmsnorm_sharded, &d, 0)
    }
}

impl AddRmsnormKernel for ShimProvider {
    fn supports(&self, cfg: &AddRmsnormConfig) -> bool {
        self.add_rmsnorm_trio()
            .is_ok_and(|t| Self::supported(t, &add_rmsnorm_probe(cfg)))
    }

    fn implementation(&self, cfg: &AddRmsnormConfig) -> String {
        self.add_rmsnorm_trio()
            .map(|t| self.implementation_of(OpKind::AddRmsnorm, t, &add_rmsnorm_probe(cfg)))
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &mut AddRmsnormContext<'_>) -> Result<(), KernelError> {
        let trio = self.add_rmsnorm_trio()?;
        let residual_stride_row = row_stride("residual", &ctx.residual, 2)?;
        let (rows, dim) = (ctx.residual.shape[0], ctx.residual.shape[1]);
        for (name, v) in [("x", &ctx.x), ("out", &ctx.out)] {
            if v.shape.as_slice() != [rows, dim] || v.dtype != ctx.residual.dtype {
                return Err(invalid(format!(
                    "{name} must be a {} view of shape [{rows}, {dim}], has {} shape {:?}",
                    ctx.residual.dtype.as_str(),
                    v.dtype.as_str(),
                    v.shape.as_slice()
                )));
            }
        }
        dense("weight", &ctx.weight, &[dim], ctx.residual.dtype)?;
        let d = AddRmsnormDesc {
            residual_stride_row,
            x_stride_row: row_stride("x", &ctx.x, 2)?,
            out_stride_row: row_stride("out", &ctx.out, 2)?,
            residual: self.ctx.device_ptr("residual", &ctx.residual)?,
            x: self.ctx.device_ptr("x", &ctx.x)?,
            weight: self.ctx.device_ptr("weight", &ctx.weight)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            rows: to_i64("rows", rows)?,
            dim: to_i64("dim", dim)?,
            eps: ctx.eps,
            dtype: ctx.residual.dtype.abi_code(),
        };
        self.run(OpKind::AddRmsnorm, trio, &d, 0)
    }
}

impl LogitsReduceKernel for ShimProvider {
    fn supports(&self, cfg: &LogitsReduceConfig) -> bool {
        cfg.top_n <= LogitsReduceConfig::MAX_TOP_N
            && self
                .logits_reduce_trio()
                .is_ok_and(|t| Self::supported(t, &logits_reduce_probe(cfg)))
    }

    fn implementation(&self, cfg: &LogitsReduceConfig) -> String {
        self.logits_reduce_trio()
            .map(|t| self.implementation_of(OpKind::LogitsReduce, t, &logits_reduce_probe(cfg)))
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &mut LogitsReduceContext<'_>) -> Result<(), KernelError> {
        let trio = self.logits_reduce_trio()?;
        let rows = ctx.rows as usize;
        let stride_row = row_stride("logits", &ctx.logits, 2)?;
        if ctx.logits.dtype != DType::F32 || ctx.logits.shape[0] < rows {
            return Err(invalid(format!(
                "logits must be an f32 view of at least {rows} rows, has {} shape {:?}",
                ctx.logits.dtype.as_str(),
                ctx.logits.shape.as_slice()
            )));
        }
        let top_n = ctx.top_ids.shape.get(1).copied().unwrap_or(0);
        dense_rows("top_ids", &ctx.top_ids, rows, Some(top_n), DType::I32)?;
        dense_rows("top_values", &ctx.top_values, rows, Some(top_n), DType::F32)?;
        for (name, v, dtype) in [
            ("temperature", &ctx.temperature, DType::F32),
            ("uniform", &ctx.uniform, DType::F32),
            ("top_p", &ctx.top_p, DType::F32),
            ("mode", &ctx.mode, DType::I32),
            ("lse", &ctx.lse, DType::F32),
            ("sampled", &ctx.sampled, DType::I32),
            ("sampled_logit", &ctx.sampled_logit, DType::F32),
        ] {
            dense_rows(name, v, rows, None, dtype)?;
        }
        let d = LogitsReduceDesc {
            logits: self.ctx.device_ptr("logits", &ctx.logits)? as *const f32,
            rows: to_i64("rows", rows)?,
            vocab: to_i64("vocab", ctx.logits.shape[1])?,
            stride_row,
            temperature: self.ctx.device_ptr("temperature", &ctx.temperature)? as *const f32,
            uniform: self.ctx.device_ptr("uniform", &ctx.uniform)? as *const f32,
            mode: self.ctx.device_ptr("mode", &ctx.mode)? as *const i32,
            top_n: to_i32("top_n", top_n)?,
            top_ids: self.ctx.device_ptr("top_ids", &ctx.top_ids)?.cast(),
            top_values: self.ctx.device_ptr("top_values", &ctx.top_values)?.cast(),
            lse: self.ctx.device_ptr("lse", &ctx.lse)?.cast(),
            sampled: self.ctx.device_ptr("sampled", &ctx.sampled)?.cast(),
            sampled_logit: self
                .ctx
                .device_ptr("sampled_logit", &ctx.sampled_logit)?
                .cast(),
            top_p: self.ctx.device_ptr("top_p", &ctx.top_p)? as *const f32,
        };
        self.run(OpKind::LogitsReduce, trio, &d, 0)
    }
}

impl GemmKernel for ShimProvider {
    fn supports(&self, cfg: &GemmConfig) -> bool {
        Self::supported(&self.syms().gemm, &gemm_probe(cfg))
    }

    fn implementation(&self, cfg: &GemmConfig) -> String {
        self.implementation_of(OpKind::Gemm, &self.syms().gemm, &gemm_probe(cfg))
    }

    fn execute(&self, ctx: &mut GemmContext<'_>) -> Result<(), KernelError> {
        let d = GemmDesc {
            a: self.ctx.device_ptr("a", &ctx.a)?,
            b: self.ctx.device_ptr("b", &ctx.b)?,
            c: self.ctx.device_ptr("c", &ctx.c)?,
            m: to_i64("m", ctx.a.shape.first().copied().unwrap_or(0))?,
            n: to_i64("n", ctx.c.shape.get(1).copied().unwrap_or(0))?,
            k: to_i64("k", ctx.a.shape.get(1).copied().unwrap_or(0))?,
            lda: row_stride("a", &ctx.a, 2)?,
            ldb: row_stride("b", &ctx.b, 2)?,
            ldc: row_stride("c", &ctx.c, 2)?,
            trans_b: i32::from(ctx.trans_b),
            a_dtype: ctx.a.dtype.abi_code(),
            b_dtype: ctx.b.dtype.abi_code(),
            c_dtype: ctx.c.dtype.abi_code(),
            alpha: ctx.alpha,
            beta: ctx.beta,
        };
        self.ctx.gemm_step(ctx.prefill)?;
        self.run(OpKind::Gemm, &self.syms().gemm, &d, 0)
    }
}

impl AttentionKernel for ShimProvider {
    fn supports(&self, cfg: &AttentionConfig) -> bool {
        if let Some(trio) = self.paged_trio(cfg.kind) {
            cfg.block_tokens.is_some_and(|b| b > 0) && Self::supported(trio, &paged_probe(cfg))
        } else if let Some(trio) = self.attention_trio(cfg.kind) {
            cfg.block_tokens.is_none() && Self::supported(trio, &attention_probe(cfg))
        } else {
            false
        }
    }

    fn implementation(&self, cfg: &AttentionConfig) -> String {
        if let Some(trio) = self.paged_trio(cfg.kind) {
            self.implementation_of(cfg.op(), trio, &paged_probe(cfg))
        } else if let Some(trio) = self.attention_trio(cfg.kind) {
            self.implementation_of(cfg.op(), trio, &attention_probe(cfg))
        } else {
            format!("unsupported attention kind {:?}", cfg.kind)
        }
    }

    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
        let Some(trio) = self.attention_trio(ctx.cfg.kind) else {
            return Err(invalid(format!(
                "{} attention runs through execute_paged",
                ctx.cfg.op()
            )));
        };
        let kv_stride_token = row_stride("k_cache", &ctx.k_cache, 3)?;
        if row_stride("v_cache", &ctx.v_cache, 3)? != kv_stride_token {
            return Err(invalid(
                "k_cache and v_cache must share one token stride".into(),
            ));
        }
        let d = AttentionDesc {
            q: self.ctx.device_ptr("q", &ctx.q)?,
            k_cache: self.ctx.device_ptr("k_cache", &ctx.k_cache)?,
            v_cache: self.ctx.device_ptr("v_cache", &ctx.v_cache)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            q_stride_token: row_stride("q", &ctx.q, 3)?,
            kv_stride_token,
            out_stride_token: row_stride("out", &ctx.out, 3)?,
            q_len: to_i32("q_len", ctx.q.shape[0])?,
            q_start: to_i32("q_start", ctx.q_start)?,
            num_q_heads: to_i32("num_q_heads", ctx.cfg.num_q_heads)?,
            num_kv_heads: to_i32("num_kv_heads", ctx.cfg.num_kv_heads)?,
            head_dim: to_i32("head_dim", ctx.cfg.head_dim)?,
            scale: ctx.scale,
            causal: i32::from(ctx.cfg.causal),
            dtype: ctx.cfg.dtype.abi_code(),
        };
        self.run(ctx.cfg.op(), trio, &d, 0)
    }

    fn execute_paged(&self, ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        let (Some(trio), Some(block_tokens)) = (self.paged_trio(cfg.kind), cfg.block_tokens) else {
            return Err(invalid(format!(
                "{} attention with block_tokens {:?} is not paged",
                cfg.op(),
                cfg.block_tokens
            )));
        };
        let (hkv, d) = (cfg.num_kv_heads as usize, cfg.head_dim as usize);
        let new_stride_token = row_stride("k_new", &ctx.k_new, 3)?;
        if row_stride("v_new", &ctx.v_new, 3)? != new_stride_token {
            return Err(invalid(
                "k_new and v_new must share one token stride".into(),
            ));
        }
        let num_blocks = ctx.kv_layer.shape.first().copied().unwrap_or(0);
        dense(
            "kv_layer",
            &ctx.kv_layer,
            &[num_blocks, 2, block_tokens as usize, hkv, d],
            cfg.dtype,
        )?;
        let num_seqs = ctx.kv_lens.shape.first().copied().unwrap_or(0);
        let max_blocks = ctx.max_blocks_per_seq as usize;
        dense("kv_lens", &ctx.kv_lens, &[num_seqs], DType::I32)?;
        dense("q_indptr", &ctx.q_indptr, &[num_seqs + 1], DType::I32)?;
        dense(
            "block_table",
            &ctx.block_table,
            &[num_seqs, max_blocks],
            DType::I32,
        )?;
        let d = AttentionPagedDesc {
            q_stride_token: row_stride("q", &ctx.q, 3)?,
            new_stride_token,
            out_stride_token: row_stride("out", &ctx.out, 3)?,
            q: self.ctx.device_ptr("q", &ctx.q)?,
            k_new: self.ctx.device_ptr("k_new", &ctx.k_new)?,
            v_new: self.ctx.device_ptr("v_new", &ctx.v_new)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            kv_layer: self.ctx.device_ptr("kv_layer", &ctx.kv_layer)?,
            block_table: self.ctx.device_ptr("block_table", &ctx.block_table)? as *const i32,
            q_indptr: self.ctx.device_ptr("q_indptr", &ctx.q_indptr)? as *const i32,
            kv_lens: self.ctx.device_ptr("kv_lens", &ctx.kv_lens)? as *const i32,
            num_seqs: to_i32("num_seqs", num_seqs)?,
            total_q: to_i32("total_q", ctx.q.shape[0])?,
            max_q_len: to_i32("max_q_len", ctx.max_q_len)?,
            max_kv_len: to_i32("max_kv_len", ctx.max_kv_len)?,
            max_blocks_per_seq: to_i32("max_blocks_per_seq", ctx.max_blocks_per_seq)?,
            num_blocks: to_i32("num_blocks", num_blocks)?,
            block_tokens: to_i32("block_tokens", block_tokens)?,
            num_q_heads: to_i32("num_q_heads", cfg.num_q_heads)?,
            num_kv_heads: to_i32("num_kv_heads", cfg.num_kv_heads)?,
            head_dim: to_i32("head_dim", cfg.head_dim)?,
            scale: ctx.scale,
            causal: i32::from(cfg.causal),
            dtype: cfg.dtype.abi_code(),
        };
        self.run(cfg.op(), trio, &d, 0)
    }
}

impl KvCopyKernel for ShimProvider {
    fn supports(&self, cfg: &KvCopyConfig) -> bool {
        Self::supported(&self.syms().copy_blocks, &copy_blocks_probe(cfg))
    }

    fn implementation(&self, cfg: &KvCopyConfig) -> String {
        self.implementation_of(
            OpKind::CopyBlocks,
            &self.syms().copy_blocks,
            &copy_blocks_probe(cfg),
        )
    }

    fn execute(&self, ctx: &mut KvCopyContext<'_>) -> Result<(), KernelError> {
        if ctx.block_bytes == 0 || ctx.layer_stride_bytes < ctx.block_bytes {
            return Err(invalid(format!(
                "block_bytes {} must be positive and at most layer_stride_bytes {}",
                ctx.block_bytes, ctx.layer_stride_bytes
            )));
        }
        let pool_bytes = u64::from(ctx.num_layers) * ctx.layer_stride_bytes;
        if pool_bytes > ctx.pool.len() as u64 {
            return Err(invalid(format!(
                "{} layers of {} bytes exceed the pool of {} bytes",
                ctx.num_layers,
                ctx.layer_stride_bytes,
                ctx.pool.len()
            )));
        }
        // Host arrays, read by the shim during the call only.
        let blocks_per_layer = ctx.layer_stride_bytes / ctx.block_bytes;
        let mut src = Vec::with_capacity(ctx.pairs.len());
        let mut dst = Vec::with_capacity(ctx.pairs.len());
        for &(s, d) in ctx.pairs {
            for b in [s, d] {
                if u64::from(b.0) >= blocks_per_layer {
                    return Err(invalid(format!(
                        "block id {} is outside the {blocks_per_layer} blocks of a layer",
                        b.0
                    )));
                }
            }
            src.push(to_i32("src block", s.0)?);
            dst.push(to_i32("dst block", d.0)?);
        }
        let d = CopyBlocksDesc {
            pool: self.ctx.slice_ptr("pool", &ctx.pool)?,
            layer_stride_bytes: to_i64("layer_stride_bytes", ctx.layer_stride_bytes)?,
            block_bytes: to_i64("block_bytes", ctx.block_bytes)?,
            num_layers: to_i32("num_layers", ctx.num_layers)?,
            src_blocks: src.as_ptr(),
            dst_blocks: dst.as_ptr(),
            count: to_i32("count", ctx.pairs.len())?,
        };
        self.run(OpKind::CopyBlocks, &self.syms().copy_blocks, &d, 0)
    }
}

impl MoeKernel for ShimProvider {
    /// BF16 router logits need kernel ABI minor 2: an older library may not check `flags`.
    fn supports_route(&self, cfg: &MoeRouteConfig) -> bool {
        (!cfg.bf16_logits || self.syms().v21.minor >= 2)
            && Self::supported(&self.syms().moe_route, &moe_route_probe(cfg))
    }

    fn supports_experts(&self, cfg: &MoeExpertsConfig) -> bool {
        Self::supported(&self.syms().moe_experts, &moe_experts_probe(cfg))
    }

    fn implementation_route(&self, cfg: &MoeRouteConfig) -> String {
        self.implementation_of(
            OpKind::MoeRoute,
            &self.syms().moe_route,
            &moe_route_probe(cfg),
        )
    }

    fn implementation_experts(&self, cfg: &MoeExpertsConfig) -> String {
        self.implementation_of(
            OpKind::MoeExperts,
            &self.syms().moe_experts,
            &moe_experts_probe(cfg),
        )
    }

    fn route(&self, ctx: &mut MoeRouteContext<'_>) -> Result<(), KernelError> {
        let (experts, k) = (ctx.cfg.num_experts as usize, ctx.cfg.top_k as usize);
        let tokens = ctx.router_logits.shape.first().copied().unwrap_or(0);
        dense(
            "router_logits",
            &ctx.router_logits,
            &[tokens, experts],
            DType::F32,
        )?;
        dense("topk_ids", &ctx.topk_ids, &[tokens, k], DType::I32)?;
        dense("topk_weights", &ctx.topk_weights, &[tokens, k], DType::F32)?;
        dense("sorted_rows", &ctx.sorted_rows, &[tokens * k], DType::I32)?;
        dense(
            "expert_offsets",
            &ctx.expert_offsets,
            &[experts + 1],
            DType::I32,
        )?;
        let d = MoeRouteDesc {
            router_logits: self.ctx.device_ptr("router_logits", &ctx.router_logits)? as *const f32,
            num_tokens: to_i32("num_tokens", tokens)?,
            num_experts: to_i32("num_experts", ctx.cfg.num_experts)?,
            top_k: to_i32("top_k", ctx.cfg.top_k)?,
            flags: moe_route_flags(&ctx.cfg),
            topk_ids: self.ctx.device_ptr("topk_ids", &ctx.topk_ids)?.cast(),
            topk_weights: self
                .ctx
                .device_ptr("topk_weights", &ctx.topk_weights)?
                .cast(),
            sorted_rows: self.ctx.device_ptr("sorted_rows", &ctx.sorted_rows)?.cast(),
            expert_offsets: self
                .ctx
                .device_ptr("expert_offsets", &ctx.expert_offsets)?
                .cast(),
        };
        self.run(OpKind::MoeRoute, &self.syms().moe_route, &d, 0)
    }

    fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        let (h, inter) = (cfg.hidden as usize, cfg.inter as usize);
        let (experts, k) = (cfg.num_experts as usize, cfg.top_k as usize);
        let local = cfg.num_local_experts() as usize;
        let tokens = ctx.x.shape.first().copied().unwrap_or(0);
        dense("x", &ctx.x, &[tokens, h], cfg.dtype)?;
        dense("out", &ctx.out, &[tokens, h], cfg.dtype)?;
        dense("w_gate", &ctx.w_gate, &[local, inter, h], cfg.dtype)?;
        dense("w_up", &ctx.w_up, &[local, inter, h], cfg.dtype)?;
        dense("w_down", &ctx.w_down, &[local, h, inter], cfg.dtype)?;
        dense("sorted_rows", &ctx.sorted_rows, &[tokens * k], DType::I32)?;
        dense(
            "expert_offsets",
            &ctx.expert_offsets,
            &[experts + 1],
            DType::I32,
        )?;
        dense("topk_weights", &ctx.topk_weights, &[tokens, k], DType::F32)?;
        // Empty host offsets are passed as NULL: the caller asked `needs_host_offsets`, and the
        // library rejects a NULL it needs.
        let host_expert_offsets = match ctx.host_expert_offsets.len() {
            0 => std::ptr::null(),
            n if n == experts + 1 => ctx.host_expert_offsets.as_ptr(),
            n => {
                return Err(invalid(format!(
                    "host_expert_offsets has {n} entries, expected {} (or none)",
                    experts + 1
                )));
            }
        };
        let (workspace, workspace_bytes) = match &ctx.workspace {
            Some(ws) => (self.ctx.slice_ptr("workspace", ws)?, ws.len()),
            None => (null(), 0),
        };
        let d = MoeExpertsDesc {
            x: self.ctx.device_ptr("x", &ctx.x)?,
            w_gate: self.ctx.device_ptr("w_gate", &ctx.w_gate)?,
            w_up: self.ctx.device_ptr("w_up", &ctx.w_up)?,
            w_down: self.ctx.device_ptr("w_down", &ctx.w_down)?,
            sorted_rows: self.ctx.device_ptr("sorted_rows", &ctx.sorted_rows)? as *const i32,
            expert_offsets: self.ctx.device_ptr("expert_offsets", &ctx.expert_offsets)?
                as *const i32,
            topk_weights: self.ctx.device_ptr("topk_weights", &ctx.topk_weights)? as *const f32,
            host_expert_offsets,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            workspace,
            workspace_bytes,
            num_tokens: to_i32("num_tokens", tokens)?,
            hidden: to_i32("hidden", cfg.hidden)?,
            inter: to_i32("inter", cfg.inter)?,
            top_k: to_i32("top_k", cfg.top_k)?,
            num_experts: to_i32("num_experts", cfg.num_experts)?,
            expert_begin: to_i32("expert_begin", cfg.expert_begin)?,
            expert_end: to_i32("expert_end", cfg.expert_end)?,
            dtype: cfg.dtype.abi_code(),
        };
        self.run(OpKind::MoeExperts, &self.syms().moe_experts, &d, tokens * k)
    }

    /// A provider bound to `moe_experts` answers with the flag of the implementation its tier
    /// runs at `routed_rows`; otherwise the library's `turbine_moe_experts_needs_host_offsets`.
    fn needs_host_offsets(&self, cfg: &MoeExpertsConfig, routed_rows: usize) -> bool {
        if let Some(bound) = self.bound.as_ref().filter(|b| b.op == OpKind::MoeExperts) {
            return bound.info(routed_rows).needs_host_offsets;
        }
        let Some(needs) = self.syms().moe_experts_needs_host_offsets else {
            return true;
        };
        let top_k = cfg.top_k.max(1) as usize;
        let Ok(tokens) = i32::try_from(routed_rows.div_ceil(top_k)) else {
            return true;
        };
        // SAFETY: the probe is a fully initialised descriptor with null pointers; the function
        // reads only its shape fields and needs no context (header: like `_supported`).
        unsafe { needs(&moe_experts_probe_tokens(cfg, tokens)) != 0 }
    }
}

impl NormKernel for ShimProvider {
    fn supports(&self, cfg: &NormConfig) -> bool {
        Self::supported(&self.syms().rmsnorm, &rmsnorm_probe(cfg))
    }

    fn implementation(&self, cfg: &NormConfig) -> String {
        self.implementation_of(OpKind::Rmsnorm, &self.syms().rmsnorm, &rmsnorm_probe(cfg))
    }

    fn execute(&self, ctx: &mut NormContext<'_>) -> Result<(), KernelError> {
        let d = RmsnormDesc {
            x_stride_row: row_stride("x", &ctx.x, 2)?,
            out_stride_row: row_stride("out", &ctx.out, 2)?,
            x: self.ctx.device_ptr("x", &ctx.x)?,
            weight: self.ctx.device_ptr("weight", &ctx.weight)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            rows: to_i64("rows", ctx.x.shape[0])?,
            dim: to_i64("dim", ctx.x.shape[1])?,
            eps: ctx.eps,
            dtype: ctx.x.dtype.abi_code(),
        };
        self.run(OpKind::Rmsnorm, &self.syms().rmsnorm, &d, 0)
    }
}

impl RopeKernel for ShimProvider {
    fn supports(&self, cfg: &RopeConfig) -> bool {
        Self::supported(&self.syms().rope, &rope_probe(cfg))
    }

    fn implementation(&self, cfg: &RopeConfig) -> String {
        self.implementation_of(OpKind::Rope, &self.syms().rope, &rope_probe(cfg))
    }

    fn execute(&self, ctx: &mut RopeContext<'_>) -> Result<(), KernelError> {
        let mut d = rope_probe(&ctx.cfg);
        d.q_stride_token = row_stride("q", &ctx.q, 3)?;
        d.k_stride_token = row_stride("k", &ctx.k, 3)?;
        d.num_tokens = to_i32("num_tokens", ctx.q.shape[0])?;
        d.q = self.ctx.device_ptr("q", &ctx.q)?;
        d.k = self.ctx.device_ptr("k", &ctx.k)?;
        d.positions = self.ctx.device_ptr("positions", &ctx.positions)? as *const i32;
        d.inv_freq = self.ctx.device_ptr("inv_freq", &ctx.inv_freq)? as *const f32;
        self.run(OpKind::Rope, &self.syms().rope, &d, 0)
    }
}

impl ActivationKernel for ShimProvider {
    fn supports(&self, cfg: &ActivationConfig) -> bool {
        Self::supported(&self.syms().silu_mul, &silu_mul_probe(cfg))
    }

    fn implementation(&self, cfg: &ActivationConfig) -> String {
        self.implementation_of(OpKind::SiluMul, &self.syms().silu_mul, &silu_mul_probe(cfg))
    }

    fn execute(&self, ctx: &mut ActivationContext<'_>) -> Result<(), KernelError> {
        let d = SiluMulDesc {
            gate_stride_row: row_stride("gate", &ctx.gate, 2)?,
            up_stride_row: row_stride("up", &ctx.up, 2)?,
            out_stride_row: row_stride("out", &ctx.out, 2)?,
            gate: self.ctx.device_ptr("gate", &ctx.gate)?,
            up: self.ctx.device_ptr("up", &ctx.up)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            rows: to_i64("rows", ctx.gate.shape[0])?,
            cols: to_i64("cols", ctx.gate.shape[1])?,
            dtype: ctx.out.dtype.abi_code(),
        };
        self.run(OpKind::SiluMul, &self.syms().silu_mul, &d, 0)
    }
}

impl EmbeddingKernel for ShimProvider {
    fn supports(&self, cfg: &EmbeddingConfig) -> bool {
        Self::supported(&self.syms().embedding, &embedding_probe(cfg))
    }

    fn implementation(&self, cfg: &EmbeddingConfig) -> String {
        self.implementation_of(
            OpKind::Embedding,
            &self.syms().embedding,
            &embedding_probe(cfg),
        )
    }

    fn execute(&self, ctx: &mut EmbeddingContext<'_>) -> Result<(), KernelError> {
        row_stride("table", &ctx.table, 2)?;
        let d = EmbeddingDesc {
            out_stride_row: row_stride("out", &ctx.out, 2)?,
            ids: self.ctx.device_ptr("ids", &ctx.ids)? as *const i32,
            table: self.ctx.device_ptr("table", &ctx.table)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            num_tokens: to_i64("num_tokens", ctx.ids.numel())?,
            hidden: to_i64("hidden", ctx.table.shape[1])?,
            vocab_offset: ctx.vocab_offset,
            vocab_rows: to_i64("vocab_rows", ctx.table.shape[0])?,
            dtype: ctx.table.dtype.abi_code(),
        };
        self.run(OpKind::Embedding, &self.syms().embedding, &d, 0)
    }
}

impl ElementwiseKernel for ShimProvider {
    fn supports(&self, cfg: &ElementwiseConfig) -> bool {
        Self::supported(&self.syms().add, &add_probe(cfg))
    }

    fn implementation(&self, cfg: &ElementwiseConfig) -> String {
        self.implementation_of(OpKind::Add, &self.syms().add, &add_probe(cfg))
    }

    fn execute(&self, ctx: &mut ElementwiseContext<'_>) -> Result<(), KernelError> {
        let n = ctx.out.numel();
        for (name, v) in [("a", &ctx.a), ("b", &ctx.b), ("out", &ctx.out)] {
            if v.numel() != n || v.strides != contiguous_strides(&v.shape) {
                return Err(invalid(format!(
                    "add needs contiguous views of {n} elements; {name} has shape {:?} strides {:?}",
                    v.shape.as_slice(),
                    v.strides.as_slice()
                )));
            }
        }
        let d = AddDesc {
            a: self.ctx.device_ptr("a", &ctx.a)?,
            b: self.ctx.device_ptr("b", &ctx.b)?,
            out: self.ctx.device_ptr("out", &ctx.out)?,
            n: to_i64("n", n)?,
            dtype: ctx.out.dtype.abi_code(),
        };
        self.run(OpKind::Add, &self.syms().add, &d, 0)
    }
}

impl KernelProvider for ShimProvider {
    fn id(&self) -> ProviderId {
        ProviderId(self.ctx.lib.backend)
    }
    fn gemm(&self) -> Option<&dyn GemmKernel> {
        Some(self)
    }
    fn attention(&self) -> Option<&dyn AttentionKernel> {
        Some(self)
    }
    fn norm(&self) -> Option<&dyn NormKernel> {
        Some(self)
    }
    fn rope(&self) -> Option<&dyn RopeKernel> {
        Some(self)
    }
    fn activation(&self) -> Option<&dyn ActivationKernel> {
        Some(self)
    }
    fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
        Some(self)
    }
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
        Some(self)
    }
    fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
        Some(self)
    }
    fn moe(&self) -> Option<&dyn MoeKernel> {
        Some(self)
    }
    fn add_rmsnorm(&self) -> Option<&dyn AddRmsnormKernel> {
        self.syms()
            .v21
            .add_rmsnorm
            .is_some()
            .then_some(self as &dyn AddRmsnormKernel)
    }
    fn logits_reduce(&self) -> Option<&dyn LogitsReduceKernel> {
        self.syms()
            .v21
            .logits_reduce
            .is_some()
            .then_some(self as &dyn LogitsReduceKernel)
    }
    fn sharded_norm(&self) -> Option<&dyn ShardedNormKernel> {
        self.syms()
            .v21
            .tensor_parallel
            .is_some()
            .then_some(self as &dyn ShardedNormKernel)
    }

    fn implementations(&self, op: OpKind) -> Vec<ImplInfo> {
        self.ctx.lib.implementations(op)
    }

    fn card_profile(&self) -> Option<&'static CardProfile> {
        self.ctx.card_profile()
    }

    /// A provider on the same context bound to `choice` for `spec`'s op; `None` when the
    /// library does not enumerate or `choice` names an index it does not have.
    fn bind(&self, spec: &OpConfig, choice: &ImplChoice) -> Option<Arc<dyn KernelProvider>> {
        let op = spec.op();
        let impls = self.ctx.lib.implementations(op);
        let indices: Vec<u32> = match choice {
            ImplChoice::Single(index) => vec![*index],
            ImplChoice::ByRows(tiers) => tiers.iter().map(|t| t.index).collect(),
        };
        if impls.is_empty()
            || indices.is_empty()
            || indices.iter().any(|&i| i as usize >= impls.len())
        {
            return None;
        }
        Some(Arc::new(ShimProvider {
            ctx: Arc::clone(&self.ctx),
            bound: Some(Bound {
                op,
                choice: choice.clone(),
                impls,
            }),
        }))
    }

    /// `turbine_impl_supports` on the probe descriptor of `spec` (a `moe_experts` probe covers
    /// `rows` routed rows), after the same config checks as the family's `supports`.
    fn implementation_supports(&self, spec: &OpConfig, index: u32, rows: Option<u32>) -> bool {
        let lib = &self.ctx.lib;
        match spec {
            OpConfig::Gemm(cfg) => lib.impl_supports(OpKind::Gemm, index, &gemm_probe(cfg)),
            OpConfig::Attention(cfg) if cfg.kind.is_paged() => {
                cfg.block_tokens.is_some_and(|b| b > 0)
                    && lib.impl_supports(cfg.op(), index, &paged_probe(cfg))
            }
            OpConfig::Attention(cfg) => {
                cfg.block_tokens.is_none()
                    && lib.impl_supports(cfg.op(), index, &attention_probe(cfg))
            }
            OpConfig::Rmsnorm(cfg) => {
                lib.impl_supports(OpKind::Rmsnorm, index, &rmsnorm_probe(cfg))
            }
            OpConfig::Rope(cfg) => lib.impl_supports(OpKind::Rope, index, &rope_probe(cfg)),
            OpConfig::SiluMul(cfg) => {
                lib.impl_supports(OpKind::SiluMul, index, &silu_mul_probe(cfg))
            }
            OpConfig::Embedding(cfg) => {
                lib.impl_supports(OpKind::Embedding, index, &embedding_probe(cfg))
            }
            OpConfig::Add(cfg) => lib.impl_supports(OpKind::Add, index, &add_probe(cfg)),
            OpConfig::CopyBlocks(cfg) => {
                lib.impl_supports(OpKind::CopyBlocks, index, &copy_blocks_probe(cfg))
            }
            OpConfig::MoeRoute(cfg) => {
                (!cfg.bf16_logits || self.syms().v21.minor >= 2)
                    && lib.impl_supports(OpKind::MoeRoute, index, &moe_route_probe(cfg))
            }
            OpConfig::MoeExperts(cfg) => {
                let top_k = cfg.top_k.max(1);
                let tokens = rows.map_or(1, |r| r.div_ceil(top_k));
                let Ok(tokens) = i32::try_from(tokens) else {
                    return false;
                };
                lib.impl_supports(
                    OpKind::MoeExperts,
                    index,
                    &moe_experts_probe_tokens(cfg, tokens),
                )
            }
            OpConfig::AddRmsnorm(cfg) => {
                lib.impl_supports(OpKind::AddRmsnorm, index, &add_rmsnorm_probe(cfg))
            }
            OpConfig::LogitsReduce(cfg) => {
                cfg.top_n <= LogitsReduceConfig::MAX_TOP_N
                    && lib.impl_supports(OpKind::LogitsReduce, index, &logits_reduce_probe(cfg))
            }
            OpConfig::RowSumsq(cfg) => {
                lib.exports_native_streams()
                    && lib.impl_supports(OpKind::RowSumsq, index, &row_sumsq_probe(cfg))
            }
            OpConfig::RmsnormSharded(cfg) => {
                lib.exports_native_streams()
                    && cfg.full_dim >= cfg.dim
                    && lib.impl_supports(OpKind::RmsnormSharded, index, &rmsnorm_sharded_probe(cfg))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use turbine_core::types::{BlockId, DType, DeviceId, MemoryKind, Vendor};
    use turbine_device::{DeviceInfo, DeviceMemoryInfo};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, HostStaging, Tensor};

    use super::*;
    use crate::KernelError;

    fn mocked_device(arch: &str) -> DeviceInfo {
        DeviceInfo {
            index: DeviceId(0),
            vendor: Vendor::Amd,
            vendor_index: 0,
            name: "AMD Radeon AI PRO R9700".into(),
            uuid: None,
            pci_bus_id: None,
            arch: Some(arch.into()),
            driver_version: None,
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 32 << 30,
                shared_with_host: false,
            },
        }
    }

    /// Calls a test hook the stub exports besides the ABI (not part of turbine_kernels.h).
    fn stub_hook<T: Copy, R>(lib: &ShimLibrary, name: &str, call: impl FnOnce(T) -> R) -> R {
        let f: T = ffi::resolve(&lib._lib, &lib.path, name).expect("stub test hook");
        call(f)
    }

    /// Serializes the tests that create contexts in the gfx942 stub: its live-context counter is
    /// process-global, so a parallel test's context would shift `live_contexts`.
    static STUB_CONTEXTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn live_contexts(lib: &ShimLibrary) -> i32 {
        stub_hook(
            lib,
            "stub_live_contexts",
            |f: unsafe extern "C" fn() -> i32| {
                // SAFETY: the stub defines `int32_t stub_live_contexts(void)`; the library is loaded.
                unsafe { f() }
            },
        )
    }

    #[test]
    fn abi_and_arch_mismatch_are_fatal() {
        let err = ShimLibrary::load(Path::new(env!("TURBINE_STUB_ABI999")), "hip")
            .expect_err("ABI 999 must be refused");
        assert!(
            matches!(
                err,
                KernelError::AbiMismatch {
                    expected: 2,
                    found: 999
                }
            ),
            "{err:?}"
        );
        assert_eq!(
            err.to_string(),
            "kernel ABI version mismatch: library 999, expected 2"
        );

        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip")
            .expect("the gfx942 stub has the right ABI and backend");
        assert_eq!(lib.build_archs(), ["gfx942".to_string()]);
        let err = lib
            .create_context(&mocked_device("gfx1201"))
            .expect_err("a gfx1201 device must be refused");
        assert_eq!(
            err.to_string(),
            "device arch gfx1201 not in library build archs gfx942"
        );

        let err = ShimLibrary::load(Path::new("/nonexistent/libturbine_hip.so"), "hip")
            .expect_err("missing file");
        assert!(
            err.to_string()
                .starts_with("cannot load /nonexistent/libturbine_hip.so"),
            "{err}"
        );
    }

    #[test]
    fn last_error_with_null_buffer_only_returns_the_length() {
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip").expect("load");
        let mut buf = [0u8; 64];
        // SAFETY: a null context reads this thread's create message; `buf` is live and writable
        // for its full length.
        let with_buf =
            unsafe { (lib.syms.last_error)(std::ptr::null_mut(), buf.as_mut_ptr().cast(), 64) };
        // SAFETY: the header allows a null buffer: the library writes nothing and returns the
        // length.
        let without_buf =
            unsafe { (lib.syms.last_error)(std::ptr::null_mut(), std::ptr::null_mut(), buf.len()) };
        assert_eq!(without_buf, with_buf);
    }

    #[test]
    fn backend_mismatch_is_fatal() {
        let err = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "cuda")
            .expect_err("a hip library under backend cuda");
        assert_eq!(
            err.to_string(),
            "kernel library backend hip, configured cuda"
        );
    }

    #[test]
    fn context_memory_provider_and_destroy_once() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip").expect("load");
        assert_eq!(lib.abi_version(), 2);
        assert_eq!(lib.backend_name(), "hip");
        let before = live_contexts(&lib);
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert_eq!(live_contexts(&lib), before + 1);
        assert!(Arc::ptr_eq(ctx.library(), &lib));
        assert_eq!(
            ctx.info(),
            ContextInfo {
                workspace_bytes: 1 << 20,
                compute_capability: None,
                device_arch: "gfx942".into(),
            }
        );

        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        assert_eq!(mem.device(), DeviceId(0));
        assert_eq!(mem.mem_info().expect("mem_info").total_bytes, 1 << 30);
        let mut buf = DeviceBuffer::alloc(&mem, 16).expect("alloc");
        buf.copy_from_host(4, &[1, 2, 3]).expect("h2d");
        let mut back = [0u8; 3];
        buf.copy_to_host(4, &mut back).expect("d2h");
        assert_eq!(back, [1, 2, 3]);
        assert!(matches!(
            DeviceBuffer::alloc(&mem, 2 << 30),
            Err(MemoryError::OutOfMemory { requested }) if requested == 2 << 30
        ));
        assert!(matches!(
            mem.copy_d2d(buf.ptr(), buf.ptr(), 1),
            Err(MemoryError::Unsupported(_))
        ));
        assert_eq!(mem.compute_stream().device(), DeviceId(0));

        let provider = shim_provider(Arc::clone(&ctx));
        assert_eq!(provider.id(), ProviderId("hip"));
        let add = provider.elementwise().expect("add family");
        let cfg = ElementwiseConfig { dtype: DType::BF16 };
        assert!(!add.supports(&cfg));
        assert_eq!(add.implementation(&cfg), "stub_add");
        let a = Tensor::empty(&mem, &[4], DType::BF16).expect("a");
        let err = add
            .execute(&mut ElementwiseContext {
                a: a.view(),
                b: a.view(),
                out: a.view(),
            })
            .expect_err("the stub implements no op");
        assert!(
            matches!(&err, KernelError::Unsupported { message } if message == "stub: add is not implemented"),
            "{err:?}"
        );

        // Host-backend memory never reaches the shim, even on the same device id.
        let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 10);
        let foreign = Tensor::empty(&host, &[4], DType::BF16).expect("host tensor");
        let err = add
            .execute(&mut ElementwiseContext {
                a: foreign.view(),
                b: a.view(),
                out: a.view(),
            })
            .expect_err("foreign memory");
        assert!(
            err.to_string()
                .contains("a is not memory of this hip context"),
            "{err}"
        );

        // Buffers, streams and the provider hold the context; it is destroyed once, after the last.
        drop((provider, a, buf, mem, ctx));
        assert_eq!(live_contexts(&lib), before);
    }

    #[test]
    fn v2_ops_forward_through_the_abi() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip").expect("load");
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let provider = shim_provider(Arc::clone(&ctx));

        let paged = AttentionConfig {
            kind: AttentionKind::DecodePaged,
            num_q_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: Some(16),
            causal: true,
        };
        let attn = provider.attention().expect("attention family");
        assert!(!attn.supports(&paged));
        assert_eq!(attn.implementation(&paged), "stub_attention_decode_paged");
        let prefill = AttentionConfig {
            kind: AttentionKind::PrefillPaged,
            ..paged
        };
        assert_eq!(
            attn.implementation(&prefill),
            "stub_attention_prefill_paged"
        );
        // A paged kind without a page size is never supported.
        assert!(!attn.supports(&AttentionConfig {
            block_tokens: None,
            ..paged
        }));
        // Every kind binds its own entry point; none falls through to another kind's.
        for (kind, entry) in [
            (AttentionKind::Prefill, "stub_attention_prefill"),
            (AttentionKind::Decode, "stub_attention_decode"),
            (AttentionKind::PrefillPaged, "stub_attention_prefill_paged"),
            (AttentionKind::DecodePaged, "stub_attention_decode_paged"),
        ] {
            let block_tokens = kind.is_paged().then_some(16);
            let cfg = AttentionConfig {
                kind,
                block_tokens,
                ..paged
            };
            assert_eq!(attn.implementation(&cfg), entry, "{kind:?}");
        }
        // A paged config handed to the contiguous entry point is refused before any FFI call.
        let t = Tensor::empty(&mem, &[1, 16, 128], DType::BF16).expect("q");
        let err = attn
            .execute(&mut AttentionContext {
                cfg: paged,
                q: t.view(),
                k_cache: t.view(),
                v_cache: t.view(),
                out: t.view(),
                q_start: 0,
                scale: 1.0,
            })
            .expect_err("paged kind on the contiguous path");
        assert!(
            matches!(&err, KernelError::InvalidArgument { message } if message.contains("execute_paged")),
            "{err:?}"
        );

        let copy = KvCopyConfig {
            num_layers: 2,
            block_bytes: 64,
        };
        let kv_copy = provider.kv_copy().expect("kv_copy family");
        assert_eq!(kv_copy.implementation(&copy), "stub_copy_blocks");
        let pool = DeviceBuffer::alloc(&mem, 2 * 4 * 64).expect("pool");
        let err = kv_copy
            .execute(&mut KvCopyContext {
                pool: pool.whole(),
                layer_stride_bytes: 4 * 64,
                block_bytes: 64,
                num_layers: 2,
                pairs: &[(BlockId(1), BlockId(3))],
            })
            .expect_err("the stub implements no op");
        assert!(
            matches!(&err, KernelError::Unsupported { message } if message == "stub: copy_blocks is not implemented"),
            "{err:?}"
        );
        let err = kv_copy
            .execute(&mut KvCopyContext {
                pool: pool.whole(),
                layer_stride_bytes: 4 * 64,
                block_bytes: 64,
                num_layers: 2,
                pairs: &[(BlockId(1), BlockId(4))],
            })
            .expect_err("block 4 is outside a 4-block layer");
        assert!(err.to_string().contains("block id 4"), "{err}");

        let moe = provider.moe().expect("moe family");
        let route = MoeRouteConfig {
            num_experts: 64,
            top_k: 8,
            renormalize: false,
            bf16_logits: false,
        };
        assert!(!moe.supports_route(&route));
        assert_eq!(moe.implementation_route(&route), "stub_moe_route");
        let experts = MoeExpertsConfig {
            hidden: 2048,
            inter: 1024,
            num_experts: 64,
            top_k: 8,
            expert_begin: 0,
            expert_end: 64,
            dtype: DType::BF16,
        };
        assert!(!moe.supports_experts(&experts));
        assert_eq!(moe.implementation_experts(&experts), "stub_moe_experts");
        // The stub exports no `turbine_moe_experts_needs_host_offsets`: a library without it
        // always gets the host offsets (the Phase 2 contract).
        for rows in [8, 512, 4096] {
            assert!(moe.needs_host_offsets(&experts, rows), "{rows} routed rows");
        }
    }

    /// ABI v2.1 is optional. The plain (v2.0) stub loads with minor 0, its provider has no
    /// `add_rmsnorm`/`logits_reduce` family and its context answers the option and graph calls
    /// with `Unsupported`; the `TURBINE_STUB_V21` stub reports the header's minor (3), exposes both op families
    /// through the ABI, keeps options, and captures, launches and destroys graphs. Breaks if a
    /// v2.1 symbol becomes required, if a v2.0 library reports v2.1 support, or if a graph is not
    /// destroyed exactly once.
    #[test]
    fn v21_symbols_optional() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let plain = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip")
            .expect("a v2.0 library still loads");
        assert_eq!(plain.abi_minor(), 0);
        let ctx = plain
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let provider = shim_provider(Arc::clone(&ctx));
        assert!(provider.add_rmsnorm().is_none());
        assert!(provider.logits_reduce().is_none());
        for result in [
            ctx.graph_begin(),
            ctx.graph_end().map(drop),
            ctx.set_option(TURBINE_OPTION_GEMM_AUTOTUNE, 1),
            ctx.get_option(TURBINE_OPTION_GEMM_AUTOTUNE).map(drop),
        ] {
            let err = result.expect_err("a v2.0 library has no v2.1 functions");
            assert!(
                matches!(&err, KernelError::Unsupported { message } if message.contains("kernel ABI minor 0")),
                "{err:?}"
            );
        }
        drop((provider, ctx));

        let v21 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V21")), "hip")
            .expect("a v2.1 library loads");
        assert_eq!((v21.abi_version(), v21.abi_minor()), (2, 3));
        let ctx = v21
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let provider = shim_provider(Arc::clone(&ctx));
        let fused_cfg = AddRmsnormConfig {
            dtype: DType::BF16,
            dim: 3072,
        };
        let fused = provider.add_rmsnorm().expect("v2.1 add_rmsnorm family");
        assert!(!fused.supports(&fused_cfg), "the stub supports no config");
        assert_eq!(fused.implementation(&fused_cfg), "stub_add_rmsnorm");
        let reduce_cfg = LogitsReduceConfig {
            vocab: 128_256,
            top_n: 20,
        };
        let reduce = provider.logits_reduce().expect("v2.1 logits_reduce family");
        assert!(!reduce.supports(&reduce_cfg));
        assert_eq!(reduce.implementation(&reduce_cfg), "stub_logits_reduce");

        // The op forwards through the ABI (the stub reports it unimplemented).
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let rows = Tensor::empty(&mem, &[2, 8], DType::BF16).expect("rows");
        let weight = Tensor::empty(&mem, &[8], DType::BF16).expect("weight");
        let err = fused
            .execute(&mut AddRmsnormContext {
                residual: rows.view(),
                x: rows.view(),
                weight: weight.view(),
                out: rows.view(),
                eps: 1e-5,
            })
            .expect_err("the stub implements no op");
        assert!(
            matches!(&err, KernelError::Unsupported { message } if message == "stub: add_rmsnorm is not implemented"),
            "{err:?}"
        );

        ctx.set_option(TURBINE_OPTION_GEMM_AUTOTUNE, 1)
            .expect("set autotune");
        assert_eq!(
            ctx.get_option(TURBINE_OPTION_GEMM_AUTOTUNE).expect("get"),
            1
        );
        ctx.set_option(TURBINE_OPTION_GEMM_AUTOTUNE, 0)
            .expect("clear autotune");
        assert_eq!(
            ctx.get_option(TURBINE_OPTION_GEMM_AUTOTUNE).expect("get"),
            0
        );
        assert_eq!(
            ctx.get_option(TURBINE_OPTION_GEMM_TUNED_SHAPES)
                .expect("tuned shapes"),
            0
        );
        assert!(matches!(
            ctx.set_option(TURBINE_OPTION_GEMM_TUNED_SHAPES, 5),
            Err(KernelError::InvalidArgument { .. })
        ));
        assert!(matches!(
            ctx.get_option(99),
            Err(KernelError::Unsupported { .. })
        ));

        let live_graphs = || {
            stub_hook(
                &v21,
                "stub_live_graphs",
                |f: unsafe extern "C" fn() -> i32| {
                    // SAFETY: the v2.1 stub defines `int32_t stub_live_graphs(void)`; the library
                    // is loaded.
                    unsafe { f() }
                },
            )
        };
        assert!(
            matches!(ctx.graph_end(), Err(KernelError::InvalidArgument { .. })),
            "graph_end without graph_begin"
        );
        ctx.graph_begin().expect("begin capture");
        let graph = ctx.graph_end().expect("end capture");
        assert_eq!(live_graphs(), 1);
        ctx.graph_launch(&graph).expect("launch");
        ctx.graph_launch(&graph).expect("replay");
        // A graph only launches on the context it was captured on.
        let other = v21
            .create_context(&mocked_device("gfx942"))
            .expect("second context");
        assert!(matches!(
            other.graph_launch(&graph),
            Err(KernelError::InvalidArgument { .. })
        ));
        // The graph holds its context: dropping every other handle first is safe.
        drop((provider, rows, weight, mem, ctx, other));
        assert_eq!(live_graphs(), 1);
        drop(graph);
        assert_eq!(live_graphs(), 0, "the graph is destroyed exactly once");
    }

    /// ABI v2.3 host staging: a v2.0 library has none (`Unsupported`, so the executor keeps the
    /// synchronous copies); the stub's staged copies round-trip through page-locked memory, every
    /// host access after a staged copy waits on the buffer's event (and one without a pending
    /// copy does not), and dropping the handle frees the memory and the event exactly once.
    /// Breaks if staging is reported without the symbols, if a host access skips the wait, or
    /// if a buffer or event leaks.
    #[test]
    fn v22_host_staging() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let plain =
            ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip").expect("load v2.0");
        let mem: Arc<dyn DeviceMemory> = plain
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(matches!(
            HostStaging::alloc(&mem, 16),
            Err(MemoryError::Unsupported(message)) if message.contains("v2.3")
        ));
        drop(mem);

        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V21")), "hip")
            .expect("load v2.3");
        let hook = |name: &str| {
            stub_hook(&lib, name, |f: unsafe extern "C" fn() -> i32| {
                // SAFETY: the v2.1 stub defines `int32_t <name>(void)` for the three v2.3
                // counters; the library is loaded.
                unsafe { f() }
            })
        };
        let (buffers, events) = (hook("stub_live_host_buffers"), hook("stub_live_events"));
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let staging = HostStaging::alloc(&mem, 8).expect("staging");
        assert_eq!(hook("stub_live_host_buffers"), buffers + 1);
        assert_eq!(hook("stub_live_events"), events + 1);
        let dev = DeviceBuffer::alloc(&mem, 8).expect("device buffer");

        let syncs = hook("stub_event_syncs");
        staging.write(0, &[1, 2, 3, 4]).expect("write");
        assert_eq!(hook("stub_event_syncs"), syncs, "no copy pending: no wait");
        staging.upload(0, dev.slice(4, 4)).expect("upload");
        staging.download(4, dev.slice(4, 4)).expect("download");
        let mut out = [0u8; 8];
        staging.read(0, &mut out).expect("read");
        assert_eq!(out, [1, 2, 3, 4, 1, 2, 3, 4]);
        assert_eq!(
            hook("stub_event_syncs"),
            syncs + 1,
            "one wait covers both copies"
        );
        staging.read(0, &mut out).expect("read again");
        assert_eq!(hook("stub_event_syncs"), syncs + 1);
        assert!(matches!(
            staging.upload(6, dev.slice(0, 4)),
            Err(MemoryError::InvalidArgument(_))
        ));
        let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 10);
        let foreign = DeviceBuffer::alloc(&host, 4).expect("host buffer");
        assert!(matches!(
            staging.upload(0, foreign.whole()),
            Err(MemoryError::InvalidArgument(_))
        ));

        staging
            .upload(0, dev.slice(0, 4))
            .expect("upload before drop");
        let syncs = hook("stub_event_syncs");
        drop((staging, dev, mem, ctx));
        assert_eq!(
            hook("stub_event_syncs"),
            syncs + 1,
            "freeing waits for the copy"
        );
        assert_eq!(hook("stub_live_host_buffers"), buffers);
        assert_eq!(hook("stub_live_events"), events);
    }

    #[test]
    fn descriptors_match_the_c_layout() {
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942")), "hip").expect("load");
        let c_size = |which: i32| {
            stub_hook(
                &lib,
                "stub_desc_size",
                |f: unsafe extern "C" fn(i32) -> usize| {
                    // SAFETY: the stub defines `size_t stub_desc_size(int32_t)`; the library is loaded.
                    unsafe { f(which) }
                },
            )
        };
        let rust_sizes = [
            size_of::<GemmDesc>(),
            size_of::<AttentionDesc>(),
            size_of::<RmsnormDesc>(),
            size_of::<RopeDesc>(),
            size_of::<SiluMulDesc>(),
            size_of::<EmbeddingDesc>(),
            size_of::<AddDesc>(),
            size_of::<CtxInfo>(),
            size_of::<AttentionPagedDesc>(),
            size_of::<CopyBlocksDesc>(),
            size_of::<MoeRouteDesc>(),
            size_of::<MoeExpertsDesc>(),
            size_of::<AddRmsnormDesc>(),
            size_of::<LogitsReduceDesc>(),
            size_of::<RowSumsqDesc>(),
            size_of::<RmsnormShardedDesc>(),
            size_of::<ffi::MappedCollectiveDesc>(),
        ];
        for (which, rust) in rust_sizes.into_iter().enumerate() {
            assert_eq!(c_size(which as i32), rust, "descriptor {which}");
        }
    }

    #[test]
    fn row_stride_rejects_rank_zero() {
        // A rank-0 view passes the shape/stride length check vacuously; it must be an error, not
        // an out-of-bounds index on `strides[0]`.
        let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 10);
        let scalar = Tensor::empty(&host, &[], DType::BF16).expect("scalar tensor");
        let err = row_stride("x", &scalar.view(), 0).expect_err("rank 0 has no row stride");
        assert!(
            matches!(&err, KernelError::InvalidArgument { message } if message.contains("rank")),
            "{err:?}"
        );
        let matrix = Tensor::empty(&host, &[2, 3], DType::BF16).expect("matrix");
        assert_eq!(row_stride("m", &matrix.view(), 2).expect("rank 2"), 3);
    }

    #[test]
    fn search_order() {
        let explicit = Path::new("/opt/k/libturbine_hip.so");
        assert_eq!(
            ShimLibrary::search_paths("hip", Some(explicit)),
            [explicit.to_path_buf()]
        );
        let paths = ShimLibrary::search_paths("hip", None);
        let exe_dir = std::env::current_exe()
            .expect("exe")
            .parent()
            .expect("dir")
            .to_path_buf();
        let n = paths.len();
        assert_eq!(paths[n - 2], exe_dir.join("libturbine_hip.so"));
        assert_eq!(paths[n - 1], PathBuf::from("libturbine_hip.so"));
        match std::env::var_os("TURBINE_KERNEL_LIBRARY").filter(|v| !v.is_empty()) {
            Some(env) => assert_eq!(
                paths,
                [PathBuf::from(env), paths[1].clone(), paths[2].clone()]
            ),
            None => assert_eq!(n, 2),
        }
        let cuda = ShimLibrary::search_paths("cuda", None);
        assert_eq!(cuda.last(), Some(&PathBuf::from("libturbine_cuda.so")));
    }

    fn profile_small_rows(lib: &ShimLibrary, ctx: &ShimContext) -> i64 {
        stub_hook(
            lib,
            "stub_profile_small_rows",
            |f: unsafe extern "C" fn(*const TurbineCtx) -> i64| {
                // SAFETY: the stub defines `int64_t stub_profile_small_rows(const turbine_ctx *)`;
                // `ctx.raw` is a live context of this library, only read by the hook.
                unsafe { f(ctx.raw) }
            },
        )
    }

    /// ABI v2.4: the V24 stub (minor 4) enumerates two `rmsnorm` implementations with their
    /// providers and one for every other op, and accepts the card profile with its device arch
    /// (the thresholds reach the context). Breaks if the group is not resolved on a minor-4
    /// library, if the enumeration drops or reorders an implementation, or if `set_profile` does
    /// not reach the library.
    #[test]
    fn v24_enumeration_resolved() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V24")), "hip")
            .expect("a v2.4 library loads");
        assert_eq!((lib.abi_version(), lib.abi_minor()), (2, 4));
        assert!(lib.enumerates_implementations());
        assert_eq!(
            lib.implementations(OpKind::Rmsnorm),
            [
                ImplInfo {
                    index: 0,
                    name: "stub_a".into(),
                    provider: "stub".into(),
                    needs_host_offsets: false,
                },
                ImplInfo {
                    index: 1,
                    name: "stub_b".into(),
                    provider: "stub_alt".into(),
                    needs_host_offsets: false,
                },
            ]
        );
        for &op in OpKind::ALL {
            let impls = lib.implementations(op);
            if op.abi_minor() > 4 {
                // A v2.4 library predates the op: its code is not asked for.
                assert!(impls.is_empty(), "{op}");
            } else if op != OpKind::Rmsnorm {
                let names: Vec<&str> = impls.iter().map(|i| i.name.as_str()).collect();
                assert_eq!(names, [format!("stub_{op}")], "{op}");
            }
        }
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert_eq!(profile_small_rows(&lib, &ctx), -1);
        ctx.set_profile(&crate::cards::GFX1201)
            .expect("the stub accepts a profile on its build arch");
        assert_eq!(profile_small_rows(&lib, &ctx), 512);

        // The provider asks `turbine_impl_supports` per implementation: stub_b refuses dim 4096.
        let provider = shim_provider(Arc::clone(&ctx));
        assert_eq!(provider.implementations(OpKind::Rmsnorm).len(), 2);
        let norm = |dim| {
            OpConfig::Rmsnorm(NormConfig {
                dim,
                dtype: DType::BF16,
            })
        };
        assert!(provider.implementation_supports(&norm(2048), 0, None));
        assert!(provider.implementation_supports(&norm(2048), 1, None));
        assert!(provider.implementation_supports(&norm(4096), 0, None));
        assert!(!provider.implementation_supports(&norm(4096), 1, None));
        assert!(!provider.implementation_supports(&norm(2048), 2, None));
    }

    /// A v2.3 library (the V21 stub, minor 3) has no v2.4 group: nothing is enumerated and
    /// `set_profile` is a no-op, so the library keeps choosing its implementations itself.
    #[test]
    fn v23_library_has_no_enumeration() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V21")), "hip")
            .expect("a v2.3 library loads");
        assert_eq!(lib.abi_minor(), 3);
        assert!(!lib.enumerates_implementations());
        for &op in OpKind::ALL {
            assert!(lib.implementations(op).is_empty(), "{op}");
        }
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        ctx.set_profile(&crate::cards::GFX1201)
            .expect("no v2.4 group: set_profile does nothing");
    }

    fn last_impl_run(lib: &ShimLibrary) -> i32 {
        stub_hook(
            lib,
            "stub_last_impl_run",
            |f: unsafe extern "C" fn() -> i32| {
                // SAFETY: the stub defines `int32_t stub_last_impl_run(void)`; the library is
                // loaded.
                unsafe { f() }
            },
        )
    }

    /// Phase 2m S-5: a provider bound to an implementation runs the op through
    /// `turbine_impl_run` with that index (the V24 stub records op · 100 + index), reports its
    /// name, and leaves the unbound provider on the library's entry point (which the stub
    /// refuses). A library without the v2.4 group, or an index it lacks, cannot be bound.
    #[test]
    fn bound_provider_runs_the_chosen_implementation() {
        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V24")), "hip")
            .expect("a v2.4 library loads");
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let provider = shim_provider(Arc::clone(&ctx));
        let cfg = NormConfig {
            dim: 2048,
            dtype: DType::BF16,
        };
        let spec = OpConfig::Rmsnorm(cfg);
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let x = Tensor::empty(&mem, &[1, 2048], DType::BF16).expect("x");
        let w = Tensor::empty(&mem, &[2048], DType::BF16).expect("w");
        let out = Tensor::empty(&mem, &[1, 2048], DType::BF16).expect("out");
        let run = |p: &dyn KernelProvider| {
            p.norm().expect("rmsnorm family").execute(&mut NormContext {
                x: x.view(),
                weight: w.view(),
                out: out.view(),
                eps: 1e-5,
            })
        };

        let err = run(provider.as_ref()).expect_err("the stub's turbine_rmsnorm refuses");
        assert!(
            err.to_string().contains("rmsnorm is not implemented"),
            "{err}"
        );
        let bound = provider
            .bind(&spec, &ImplChoice::Single(1))
            .expect("an enumerating library binds");
        assert_eq!(
            bound.norm().expect("rmsnorm").implementation(&cfg),
            "stub_b"
        );
        run(bound.as_ref()).expect("stub_b runs through turbine_impl_run");
        assert_eq!(last_impl_run(&lib), OpKind::Rmsnorm.abi_code() * 100 + 1);
        // Another op on the bound provider still takes the library's entry point.
        let add = ElementwiseConfig { dtype: DType::BF16 };
        assert_eq!(
            bound.elementwise().expect("add").implementation(&add),
            "stub_add"
        );
        assert!(provider.bind(&spec, &ImplChoice::Single(2)).is_none());

        let v23 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V21")), "hip")
            .expect("a v2.3 library loads");
        let old = shim_provider(
            v23.create_context(&mocked_device("gfx942"))
                .expect("context"),
        );
        assert!(old.bind(&spec, &ImplChoice::Single(0)).is_none());
    }

    fn stub_count(lib: &ShimLibrary, name: &str) -> i32 {
        stub_hook(lib, name, |f: unsafe extern "C" fn() -> i32| {
            // SAFETY: every stub counter hook is `int32_t <name>(void)`; the library is loaded.
            unsafe { f() }
        })
    }

    /// ABI v2.5 (Phase 4): a v2.4 library has no copy engine (`Unsupported`, the server then
    /// runs L0 only); the v2.5 stub allocates pinned buffers, copies through one copy stream,
    /// reports a ticket complete only once its event signals, orders a device-source copy after
    /// the compute stream, bounds-checks every end and frees everything with its owner. Breaks
    /// if a copy completes before its event, a buffer or stream leaks, or a v2.4 library is
    /// driven through missing symbols.
    #[test]
    fn pinned_memory_and_copies_through_the_abi() {
        use turbine_tensor::{CopyEngine, CopyTarget, PinnedMemory};

        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let v24 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V24")), "hip")
            .expect("load v2.4");
        let old = v24
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(!old.has_copy_engine());
        assert!(matches!(
            old.alloc_pinned(64),
            Err(MemoryError::Unsupported(message)) if message.contains("v2.5")
        ));
        drop(old);

        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V25")), "hip")
            .expect("load v2.5");
        assert_eq!((lib.abi_version(), lib.abi_minor()), (2, 5));
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(ctx.has_copy_engine());
        let pinned_before = stub_count(&lib, "stub_live_host_buffers");
        let hold = |on: bool| {
            stub_hook(&lib, "stub_hold_events", |f: unsafe extern "C" fn(i32)| {
                // SAFETY: the stub defines `void stub_hold_events(int32_t)`; the library is
                // loaded.
                unsafe { f(i32::from(on)) }
            })
        };

        let src = ctx.alloc_pinned(64).expect("pinned buffer");
        assert_eq!(src.len(), 64);
        assert_eq!(
            stub_count(&lib, "stub_live_host_buffers"),
            pinned_before + 1
        );
        let pattern: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(7)).collect();
        src.with_bytes_mut(|b| b.copy_from_slice(&pattern));

        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let dev = DeviceBuffer::alloc(&mem, 64).expect("device buffer");
        let pinned = |b: &turbine_tensor::PinnedBuffer, offset| CopyTarget::Pinned {
            buffer_id: b.id(),
            offset,
        };

        // Host → device on the copy stream; a ticket is complete only once its event signals.
        hold(true);
        let t = ctx
            .copy_async(CopyTarget::Device(dev.ptr()), pinned(&src, 0), 64)
            .expect("h2d");
        assert_eq!(t.bytes, 64);
        assert!(!ctx.poll(&t).expect("poll"), "the event has not signalled");
        hold(false);
        assert!(ctx.poll(&t).expect("poll"));
        assert!(ctx.poll(&t).expect("a finished ticket stays finished"));
        assert_eq!(stub_count(&lib, "stub_live_streams"), 1, "one copy stream");
        let waits = stub_count(&lib, "stub_stream_waits");

        // Device → host into another buffer at an offset, waited on; the copy of a device
        // source first waits for the compute stream.
        let back = ctx.alloc_pinned(96).expect("pinned buffer");
        let t = ctx
            .copy_async(pinned(&back, 32), CopyTarget::Device(dev.ptr()), 64)
            .expect("d2h");
        assert_eq!(stub_count(&lib, "stub_stream_waits"), waits + 1);
        ctx.wait(&t).expect("wait");
        back.with_bytes(|b| assert_eq!(&b[32..], pattern.as_slice()));

        // Bounds, unknown buffers and host-to-host copies are refused before the shim.
        let err = ctx
            .copy_async(pinned(&back, 64), CopyTarget::Device(dev.ptr()), 64)
            .expect_err("past the end");
        assert!(matches!(err, MemoryError::InvalidArgument(_)), "{err:?}");
        let err = ctx
            .copy_async(
                CopyTarget::Pinned {
                    buffer_id: 9999,
                    offset: 0,
                },
                CopyTarget::Device(dev.ptr()),
                1,
            )
            .expect_err("unknown buffer");
        assert!(matches!(err, MemoryError::InvalidArgument(_)), "{err:?}");
        let err = ctx
            .copy_async(pinned(&back, 0), pinned(&src, 0), 8)
            .expect_err("host to host");
        assert!(matches!(err, MemoryError::Unsupported(_)), "{err:?}");
        assert!(matches!(
            ctx.alloc_pinned(2 << 30),
            Err(MemoryError::OutOfMemory { requested }) if requested == 2 << 30
        ));

        // Buffers free on Drop; the copy stream and every copy event go with the context.
        drop((src, back));
        assert_eq!(stub_count(&lib, "stub_live_host_buffers"), pinned_before);
        let contexts = live_contexts(&lib);
        drop((dev, mem, ctx));
        assert_eq!(live_contexts(&lib), contexts - 1);
        assert_eq!(stub_count(&lib, "stub_live_streams"), 0);
    }

    /// ABI v2.6 (Phase 5 Task 6): a v2.5 library has no native stream handle (the compute
    /// stream's `StreamRef` carries 0, `has_native_streams` is false) and no sharded RMSNorm
    /// family, and is not asked for the v2.6 op codes; the v2.6 stub resolves the whole group:
    /// the compute stream carries the library's own handle for the context, the provider has the
    /// `sharded_norm` family forwarding to `turbine_row_sumsq` / `turbine_rmsnorm_sharded`, and
    /// the new ops are enumerated. Breaks if the handle is not fetched from a v2.6 library, is
    /// invented for an older one, or the group resolves on a library without it.
    #[test]
    fn v26_native_stream_and_sharded_norm_group() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let sum_cfg = RowSumsqConfig {
            dim: 1024,
            dtype: DType::BF16,
        };
        let norm_cfg = RmsnormShardedConfig {
            dim: 1024,
            full_dim: 2048,
            dtype: DType::BF16,
        };

        let v25 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V25")), "hip")
            .expect("a v2.5 library loads");
        assert_eq!(v25.abi_minor(), 5);
        assert!(!v25.exports_native_streams());
        let old = v25
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(!old.has_native_streams());
        assert_eq!(old.compute_stream().native_handle(), 0);
        let old_provider = shim_provider(Arc::clone(&old));
        assert!(old_provider.sharded_norm().is_none());
        assert!(!OpConfig::RowSumsq(sum_cfg).supported_by(old_provider.as_ref()));
        for op in [OpKind::RowSumsq, OpKind::RmsnormSharded] {
            assert!(v25.implementations(op).is_empty(), "{op}");
        }
        drop((old_provider, old));

        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V26")), "hip")
            .expect("a v2.6 library loads");
        assert_eq!((lib.abi_version(), lib.abi_minor()), (2, 6));
        assert!(lib.exports_native_streams());
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(ctx.has_native_streams());
        let want = stub_hook(
            &lib,
            "stub_compute_stream",
            |f: unsafe extern "C" fn(*mut TurbineCtx) -> *mut c_void| {
                // SAFETY: the stub defines `void *stub_compute_stream(turbine_ctx *)`; `ctx.raw`
                // is a live context of this library, and the hook only takes a field's address.
                unsafe { f(ctx.raw) }
            },
        ) as u64;
        assert_ne!(want, 0);
        let stream = ctx.compute_stream();
        assert_eq!(stream.native_handle(), want);
        assert_eq!(stream.device(), DeviceId(0));

        let provider = shim_provider(Arc::clone(&ctx));
        let sharded = provider.sharded_norm().expect("the v2.6 family");
        // The stub supports no op; the names come from the library's `_impl`.
        assert!(!sharded.supports_row_sumsq(&sum_cfg));
        assert!(!sharded.supports_rmsnorm_sharded(&norm_cfg));
        assert_eq!(sharded.implementation_row_sumsq(&sum_cfg), "stub_row_sumsq");
        assert_eq!(
            sharded.implementation_rmsnorm_sharded(&norm_cfg),
            "stub_rmsnorm_sharded"
        );
        for op in [OpKind::RowSumsq, OpKind::RmsnormSharded] {
            let names: Vec<String> = lib
                .implementations(op)
                .into_iter()
                .map(|i| i.name)
                .collect();
            assert_eq!(names, [format!("stub_{op}")], "{op}");
        }

        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let x = Tensor::empty(&mem, &[3, 1024], DType::BF16).expect("x");
        let w = Tensor::empty(&mem, &[1024], DType::BF16).expect("w");
        let sumsq = Tensor::empty(&mem, &[3], DType::F32).expect("sumsq");
        let out = Tensor::empty(&mem, &[3, 1024], DType::BF16).expect("out");
        let err = sharded
            .row_sumsq(&mut RowSumsqContext {
                x: x.view(),
                sumsq: sumsq.view(),
            })
            .expect_err("the stub refuses");
        assert!(
            matches!(&err, KernelError::Unsupported { message } if message == "stub: row_sumsq is not implemented"),
            "{err:?}"
        );
        let err = sharded
            .rmsnorm_sharded(&mut RmsnormShardedContext {
                x: x.view(),
                sumsq: sumsq.view(),
                weight: w.view(),
                out: out.view(),
                full_dim: 2048,
                eps: 1e-5,
            })
            .expect_err("the stub refuses");
        assert!(
            err.to_string()
                .contains("rmsnorm_sharded is not implemented"),
            "{err}"
        );
        // A sums view of the wrong dtype is refused before the library.
        let err = sharded
            .row_sumsq(&mut RowSumsqContext {
                x: x.view(),
                sumsq: w.view(),
            })
            .expect_err("bf16 sums");
        assert!(
            matches!(err, KernelError::InvalidArgument { .. }),
            "{err:?}"
        );
    }

    /// ABI v2.7 (Phase 5 hostmem): a v2.6 library has no host-mapped group
    /// (`DeviceMemory::mapped_collectives` is `None`); the v2.7 stub resolves the whole group:
    /// a zeroed mapped region whose words the host reads and writes, the device address the
    /// library reports, a one-rank all-reduce step run through the library, a refused step, and
    /// the allocation freed exactly once when the last region handle drops. Breaks if the group
    /// resolves without the symbols, the region is not freed or freed twice, or the descriptor
    /// does not reach the library as the header lays it out.
    #[test]
    fn v27_host_mapped_group() {
        use std::time::Duration;

        use turbine_tensor::{MappedKind, MappedReduce, MappedStep};

        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let v26 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V26")), "hip")
            .expect("a v2.6 library loads");
        let old = v26
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(!old.has_mapped_collectives());
        assert!(old.mapped_collectives().is_none());
        drop(old);

        let lib = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V27")), "hip")
            .expect("a v2.7 library loads");
        assert_eq!((lib.abi_version(), lib.abi_minor()), (2, 7));
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert!(ctx.has_mapped_collectives());
        let live = || {
            stub_hook(
                &lib,
                "stub_live_mapped",
                |f: unsafe extern "C" fn() -> i32| {
                    // SAFETY: the stub defines `int32_t stub_live_mapped(void)`; loaded.
                    unsafe { f() }
                },
            )
        };
        let before = live();
        let mc = ctx.mapped_collectives().expect("the v2.7 group");
        let region = mc.alloc_mapped(4096 + 2 * 64).expect("mapped region");
        assert_eq!(live(), before + 1);
        assert_eq!(region.len(), 4096 + 128);
        assert_eq!(region.load_u32(0), 0, "zeroed");
        region.store_u32(8, 0xabcd);
        assert_eq!(region.load_u32(8), 0xabcd);
        region.store_u32(8, 0);
        let base = mc.mapped_device_addr(&region).expect("device address");
        // The stub's device memory is host memory.
        assert_eq!(base.addr(), region.host_addr());

        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let buf = turbine_tensor::DeviceBuffer::alloc(&mem, 8).expect("buffer");
        let input: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        buf.whole().write_bytes(&input).expect("write");
        let mut step = MappedStep {
            kind: MappedKind::AllReduce,
            reduce: MappedReduce::Sum,
            dtype: DType::F32,
            rank: 0,
            world: 1,
            send: buf.whole().ptr(),
            recv: buf.whole().ptr(),
            bytes: 8,
            send_stride: 0,
            recv_stride: 0,
            flags: base,
            max_blocks: 1,
            abort_word: base.offset(64),
            slots: base.offset(128),
            slot_bytes: 64,
            seq: 1,
            timeout: Duration::from_secs(1),
        };
        assert!(mc.mapped_step_supported(&step));
        mc.enqueue_mapped_step(&step).expect("one-rank all-reduce");
        assert_eq!(buf.whole().read_bytes().expect("read"), input);
        // Published this rank's flag (tag seq << 24 | 1) through the flags address.
        assert_eq!(region.load_u32(0), (1 << 24) | 1);
        assert_eq!(region.load_u32(4), 0);

        step.bytes = 6; // not whole F32 elements
        assert!(!mc.mapped_step_supported(&step));
        let err = mc.enqueue_mapped_step(&step).expect_err("refused");
        assert!(
            matches!(&err, MemoryError::Unsupported(m) if m.contains("mapped_collective")),
            "{err:?}"
        );

        let copy = region.clone();
        drop(region);
        assert_eq!(live(), before + 1, "a handle still holds it");
        drop(copy);
        assert_eq!(live(), before, "freed once with the last handle");
        drop((buf, mem, ctx));
    }

    /// ABI v2.8 (P5 Task 32): a v2.7 library has no device-sequenced step; on the v2.8 stub
    /// each step reads the device counter, runs with seq = counter + 1 (its published tag) and
    /// stores it back, so two steps publish tags of seq 1 and 2 with no host sequence number.
    /// Breaks if the symbol resolves on a v2.7 library, the counter address does not reach the
    /// library, or the descriptor's own seq is used instead of the counter.
    #[test]
    fn v28_device_sequenced_step() {
        use std::time::Duration;

        use turbine_tensor::{MappedKind, MappedReduce, MappedStep};

        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let v27 = crate::test_support::stub_mapped_context_minor(0, 7);
        let mc27 = v27.mapped_collectives().expect("the v2.7 group");
        assert!(!mc27.mapped_dseq_supported());
        drop(v27);

        let ctx = crate::test_support::stub_mapped_context_minor(0, 8);
        assert_eq!(ctx.library().abi_minor(), 8);
        let mc = ctx.mapped_collectives().expect("the v2.7 group");
        assert!(mc.mapped_dseq_supported());
        let region = mc.alloc_mapped(4096 + 2 * 64).expect("mapped region");
        let base = mc.mapped_device_addr(&region).expect("device address");
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let buf = turbine_tensor::DeviceBuffer::alloc(&mem, 8).expect("buffer");
        buf.whole().write_bytes(&[0u8; 8]).expect("write");
        let counter = turbine_tensor::DeviceBuffer::alloc(&mem, 16).expect("counter");
        counter.whole().write_bytes(&[0u8; 16]).expect("zero");
        let step = MappedStep {
            kind: MappedKind::AllReduce,
            reduce: MappedReduce::Sum,
            dtype: DType::F32,
            rank: 0,
            world: 1,
            send: buf.whole().ptr(),
            recv: buf.whole().ptr(),
            bytes: 8,
            send_stride: 0,
            recv_stride: 0,
            flags: base,
            max_blocks: 1,
            abort_word: base.offset(64),
            slots: base.offset(128),
            slot_bytes: 64,
            // Ignored: the counter sequences the step.
            seq: 77,
            timeout: Duration::from_secs(1),
        };
        for want in 1u32..=2 {
            mc.enqueue_mapped_step_dseq(&step, counter.whole().ptr())
                .expect("device-sequenced step");
            let words = counter.whole().read_bytes().expect("read");
            assert_eq!(
                u64::from_le_bytes(words[..8].try_into().unwrap()),
                u64::from(want)
            );
            // The tag seq << 24 | 1, low word then high word.
            assert_eq!(region.load_u32(0), (want << 24) | 1);
        }
        drop((region, buf, counter, mem, ctx));
    }
}
