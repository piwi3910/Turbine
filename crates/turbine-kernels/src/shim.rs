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
//!   an `Arc` of its context) destroys the graph exactly once, in `GraphHandle::drop`.
//!
//! ABI v2.1 is optional: `ShimLibrary::abi_minor` is 0 for a v2.0 library, whose provider then
//! has no `add_rmsnorm`/`logits_reduce` family and whose context answers the option and graph
//! calls with `KernelError::Unsupported`.
use std::ffi::{CStr, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use libloading::Library;
use turbine_core::types::{DType, DeviceId, ExecutionBackend};
use turbine_device::DeviceInfo;
use turbine_tensor::tensor::contiguous_strides;
use turbine_tensor::{
    DeviceMemory, DevicePtr, DeviceSlice, MemInfo, MemoryError, StreamRef, TensorView,
};

use crate::ffi::{
    self, AddDesc, AddRmsnormDesc, AttentionDesc, AttentionPagedDesc, CopyBlocksDesc, CtxInfo,
    EmbeddingDesc, GemmDesc, LogitsReduceDesc, MOE_ROUTE_BF16_LOGITS, MOE_ROUTE_RENORMALIZE,
    MoeExpertsDesc, MoeRouteDesc, OpTrio, RmsnormDesc, RopeDesc, ShimSymbols, SiluMulDesc,
    TurbineCtx, TurbineGraph,
};
use crate::ops::{
    ActivationConfig, ActivationContext, ActivationKernel, AddRmsnormConfig, AddRmsnormContext,
    AddRmsnormKernel, AttentionConfig, AttentionContext, AttentionKernel, AttentionKind,
    ElementwiseConfig, ElementwiseContext, ElementwiseKernel, EmbeddingConfig, EmbeddingContext,
    EmbeddingKernel, GemmConfig, GemmContext, GemmKernel, KernelProvider, KvCopyConfig,
    KvCopyContext, KvCopyKernel, LogitsReduceConfig, LogitsReduceContext, LogitsReduceKernel,
    MoeExpertsConfig, MoeExpertsContext, MoeKernel, MoeRouteConfig, MoeRouteContext, NormConfig,
    NormContext, NormKernel, PagedAttentionContext, ProviderId, RopeConfig, RopeContext,
    RopeKernel,
};
use crate::{KernelError, TURBINE_KERNELS_ABI_VERSION};

/// Environment variable naming the shim library when `execution.kernel_library` is null.
const KERNEL_LIBRARY_VAR: &str = "TURBINE_KERNEL_LIBRARY";

/// Context option (ABI v2.1) `TURBINE_OPTION_GEMM_AUTOTUNE`: 1 times the GEMM algorithm
/// candidates per shape at first use, 0 takes the first heuristic answer.
pub const TURBINE_OPTION_GEMM_AUTOTUNE: i32 = 1;
/// Read-only context option (ABI v2.1) `TURBINE_OPTION_GEMM_TUNED_SHAPES`: GEMM shapes tuned so
/// far on the context.
pub const TURBINE_OPTION_GEMM_TUNED_SHAPES: i32 = 2;

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

impl ShimLibrary {
    /// Candidate library paths in search order (P1 §Configuration): `explicit`
    /// (`execution.kernel_library`) alone when set; otherwise `TURBINE_KERNEL_LIBRARY`, then
    /// `libturbine_<backend>.so` beside the executable, then the bare file name (resolved by the
    /// dynamic loader's search path).
    pub fn search_paths(backend: ExecutionBackend, explicit: Option<&Path>) -> Vec<PathBuf> {
        if let Some(path) = explicit {
            return vec![path.to_path_buf()];
        }
        let file_name = format!("libturbine_{}.so", backend.as_str());
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
    /// and the backend name compared with `expected_backend`.
    pub fn load(
        path: &Path,
        expected_backend: ExecutionBackend,
    ) -> Result<Arc<ShimLibrary>, KernelError> {
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
        if backend != expected_backend.as_str() {
            return Err(KernelError::BackendMismatch {
                expected: expected_backend.as_str().to_string(),
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
            backend: expected_backend.as_str(),
            archs,
            _lib: lib,
        }))
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
        Ok(Arc::new_cyclic(|weak| ShimContext {
            raw,
            lib: Arc::clone(self),
            device: device.index,
            info,
            self_ref: weak.clone(),
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
    lib: Arc<ShimLibrary>,
    device: DeviceId,
    info: ContextInfo,
    /// Lets `compute_stream` hand out an owning `Arc` of this context.
    self_ref: Weak<ShimContext>,
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

    /// Workspace size, compute capability and device arch of this context.
    pub fn info(&self) -> ContextInfo {
        self.info.clone()
    }

    fn check(&self, code: i32) -> Result<(), KernelError> {
        ffi::check(code, &self.lib.syms, self.raw)
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

    fn copy_d2d(&self, _dst: DevicePtr, _src: DevicePtr, _bytes: usize) -> Result<(), MemoryError> {
        Err(MemoryError::Unsupported(
            "device-to-device copy needs kernel ABI v3".into(),
        ))
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

    fn compute_stream(&self) -> StreamRef {
        let owner: Arc<dyn DeviceMemory> = self
            .self_ref
            .upgrade()
            .expect("a ShimContext only exists inside the Arc create_context returns");
        StreamRef::new(0, self.device, owner)
    }
}

/// The kernel provider of a shim context: `supports` → `turbine_<op>_supported` (null
/// pointers), `implementation` → `turbine_<op>_impl`, `execute` → `turbine_<op>`.
pub struct ShimProvider {
    ctx: Arc<ShimContext>,
}

/// The provider of `ctx`; its id is the library's backend name (`hip`, `cuda`).
pub fn shim_provider(ctx: Arc<ShimContext>) -> Arc<dyn KernelProvider> {
    Arc::new(ShimProvider { ctx })
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

    fn implementation_of<D>(trio: &OpTrio<D>, d: &D) -> String {
        // SAFETY: as for `_supported`; the returned string is static and owned by the library.
        ffi::c_str(unsafe { (trio.implementation)(d) })
    }

    fn run<D>(&self, trio: &OpTrio<D>, d: &D) -> Result<(), KernelError> {
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

impl AddRmsnormKernel for ShimProvider {
    fn supports(&self, cfg: &AddRmsnormConfig) -> bool {
        self.add_rmsnorm_trio()
            .is_ok_and(|t| Self::supported(t, &add_rmsnorm_probe(cfg)))
    }

    fn implementation(&self, cfg: &AddRmsnormConfig) -> String {
        self.add_rmsnorm_trio()
            .map(|t| Self::implementation_of(t, &add_rmsnorm_probe(cfg)))
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
        self.run(trio, &d)
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
            .map(|t| Self::implementation_of(t, &logits_reduce_probe(cfg)))
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
        self.run(trio, &d)
    }
}

impl GemmKernel for ShimProvider {
    fn supports(&self, cfg: &GemmConfig) -> bool {
        Self::supported(&self.syms().gemm, &gemm_probe(cfg))
    }

    fn implementation(&self, cfg: &GemmConfig) -> String {
        Self::implementation_of(&self.syms().gemm, &gemm_probe(cfg))
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
        self.run(&self.syms().gemm, &d)
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
            Self::implementation_of(trio, &paged_probe(cfg))
        } else if let Some(trio) = self.attention_trio(cfg.kind) {
            Self::implementation_of(trio, &attention_probe(cfg))
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
        self.run(trio, &d)
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
        self.run(trio, &d)
    }
}

impl KvCopyKernel for ShimProvider {
    fn supports(&self, cfg: &KvCopyConfig) -> bool {
        Self::supported(&self.syms().copy_blocks, &copy_blocks_probe(cfg))
    }

    fn implementation(&self, cfg: &KvCopyConfig) -> String {
        Self::implementation_of(&self.syms().copy_blocks, &copy_blocks_probe(cfg))
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
        self.run(&self.syms().copy_blocks, &d)
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
        Self::implementation_of(&self.syms().moe_route, &moe_route_probe(cfg))
    }

    fn implementation_experts(&self, cfg: &MoeExpertsConfig) -> String {
        Self::implementation_of(&self.syms().moe_experts, &moe_experts_probe(cfg))
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
        self.run(&self.syms().moe_route, &d)
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
        self.run(&self.syms().moe_experts, &d)
    }

    fn needs_host_offsets(&self, cfg: &MoeExpertsConfig, routed_rows: usize) -> bool {
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
        Self::implementation_of(&self.syms().rmsnorm, &rmsnorm_probe(cfg))
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
        self.run(&self.syms().rmsnorm, &d)
    }
}

impl RopeKernel for ShimProvider {
    fn supports(&self, cfg: &RopeConfig) -> bool {
        Self::supported(&self.syms().rope, &rope_probe(cfg))
    }

    fn implementation(&self, cfg: &RopeConfig) -> String {
        Self::implementation_of(&self.syms().rope, &rope_probe(cfg))
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
        self.run(&self.syms().rope, &d)
    }
}

impl ActivationKernel for ShimProvider {
    fn supports(&self, cfg: &ActivationConfig) -> bool {
        Self::supported(&self.syms().silu_mul, &silu_mul_probe(cfg))
    }

    fn implementation(&self, cfg: &ActivationConfig) -> String {
        Self::implementation_of(&self.syms().silu_mul, &silu_mul_probe(cfg))
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
        self.run(&self.syms().silu_mul, &d)
    }
}

impl EmbeddingKernel for ShimProvider {
    fn supports(&self, cfg: &EmbeddingConfig) -> bool {
        Self::supported(&self.syms().embedding, &embedding_probe(cfg))
    }

    fn implementation(&self, cfg: &EmbeddingConfig) -> String {
        Self::implementation_of(&self.syms().embedding, &embedding_probe(cfg))
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
        self.run(&self.syms().embedding, &d)
    }
}

impl ElementwiseKernel for ShimProvider {
    fn supports(&self, cfg: &ElementwiseConfig) -> bool {
        Self::supported(&self.syms().add, &add_probe(cfg))
    }

    fn implementation(&self, cfg: &ElementwiseConfig) -> String {
        Self::implementation_of(&self.syms().add, &add_probe(cfg))
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
        self.run(&self.syms().add, &d)
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
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use turbine_core::types::{BlockId, DType, DeviceId, ExecutionBackend, MemoryKind, Vendor};
    use turbine_device::{DeviceInfo, DeviceMemoryInfo};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, Tensor};

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
        let err = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_ABI999")),
            ExecutionBackend::Hip,
        )
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

        let lib = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Hip,
        )
        .expect("the gfx942 stub has the right ABI and backend");
        assert_eq!(lib.build_archs(), ["gfx942".to_string()]);
        let err = lib
            .create_context(&mocked_device("gfx1201"))
            .expect_err("a gfx1201 device must be refused");
        assert_eq!(
            err.to_string(),
            "device arch gfx1201 not in library build archs gfx942"
        );

        let err = ShimLibrary::load(
            Path::new("/nonexistent/libturbine_hip.so"),
            ExecutionBackend::Hip,
        )
        .expect_err("missing file");
        assert!(
            err.to_string()
                .starts_with("cannot load /nonexistent/libturbine_hip.so"),
            "{err}"
        );
    }

    #[test]
    fn backend_mismatch_is_fatal() {
        let err = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Cuda,
        )
        .expect_err("a hip library under backend cuda");
        assert_eq!(
            err.to_string(),
            "kernel library backend hip, configured cuda"
        );
    }

    #[test]
    fn context_memory_provider_and_destroy_once() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let lib = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Hip,
        )
        .expect("load");
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
        let lib = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Hip,
        )
        .expect("load");
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
    /// with `Unsupported`; the `TURBINE_STUB_V21` stub reports the header's minor (2), exposes both op families
    /// through the ABI, keeps options, and captures, launches and destroys graphs. Breaks if a
    /// v2.1 symbol becomes required, if a v2.0 library reports v2.1 support, or if a graph is not
    /// destroyed exactly once.
    #[test]
    fn v21_symbols_optional() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let plain = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Hip,
        )
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

        let v21 = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942_V21")),
            ExecutionBackend::Hip,
        )
        .expect("a v2.1 library loads");
        assert_eq!((v21.abi_version(), v21.abi_minor()), (2, 2));
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

    #[test]
    fn descriptors_match_the_c_layout() {
        let lib = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Hip,
        )
        .expect("load");
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
            ShimLibrary::search_paths(ExecutionBackend::Hip, Some(explicit)),
            [explicit.to_path_buf()]
        );
        let paths = ShimLibrary::search_paths(ExecutionBackend::Hip, None);
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
        let cuda = ShimLibrary::search_paths(ExecutionBackend::Cuda, None);
        assert_eq!(cuda.last(), Some(&PathBuf::from("libturbine_cuda.so")));
    }
}
