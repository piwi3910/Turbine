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
//!   outlives it, and the context holds an `Arc` of its `ShimLibrary`, so the code stays loaded.
use std::ffi::c_void;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use libloading::Library;
use turbine_core::types::{DeviceId, ExecutionBackend};
use turbine_device::DeviceInfo;
use turbine_tensor::tensor::contiguous_strides;
use turbine_tensor::{DeviceMemory, DevicePtr, MemInfo, MemoryError, StreamRef, TensorView};

use crate::ffi::{
    self, AddDesc, AttentionDesc, EmbeddingDesc, GemmDesc, OpTrio, RmsnormDesc, RopeDesc,
    ShimSymbols, SiluMulDesc, TurbineCtx,
};
use crate::ops::{
    ActivationConfig, ActivationContext, ActivationKernel, AttentionConfig, AttentionContext,
    AttentionKernel, AttentionKind, ElementwiseConfig, ElementwiseContext, ElementwiseKernel,
    EmbeddingConfig, EmbeddingContext, EmbeddingKernel, GemmConfig, GemmContext, GemmKernel,
    KernelProvider, NormConfig, NormContext, NormKernel, ProviderId, RopeConfig, RopeContext,
    RopeKernel,
};
use crate::{KernelError, TURBINE_KERNELS_ABI_VERSION};

/// Environment variable naming the shim library when `execution.kernel_library` is null.
const KERNEL_LIBRARY_VAR: &str = "TURBINE_KERNEL_LIBRARY";

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
    /// resolved (a library of another ABI may lack them), then every ABI v1 symbol is resolved
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
        Ok(Arc::new_cyclic(|weak| ShimContext {
            raw,
            lib: Arc::clone(self),
            device: device.index,
            self_ref: weak.clone(),
        }))
    }
}

/// One shim context (`turbine_ctx*`) on one device: owns the device's compute stream, library
/// handles and workspace. The `DeviceMemory` backend for that device.
pub struct ShimContext {
    raw: *mut TurbineCtx,
    lib: Arc<ShimLibrary>,
    device: DeviceId,
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

    fn check(&self, code: i32) -> Result<(), KernelError> {
        ffi::check(code, &self.lib.syms, self.raw)
    }

    /// The device address of `v` for a descriptor, after checking the view's memory is this
    /// context (a host-backend or other-device pointer must never reach the shim).
    fn device_ptr(&self, name: &str, v: &TensorView<'_>) -> Result<*mut c_void, KernelError> {
        let owner = Arc::as_ptr(v.slice.memory());
        if !std::ptr::addr_eq(owner, std::ptr::from_ref(self)) {
            return Err(KernelError::InvalidArgument {
                message: format!(
                    "{name} is not memory of this {} context on device {}",
                    self.lib.backend, self.device.0
                ),
            });
        }
        Ok(v.slice.ptr().addr() as *mut c_void)
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
    if v.shape.len() != rank || v.strides.len() != rank {
        return Err(invalid(format!(
            "{name} has shape {:?}, expected rank {rank}",
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

    fn attention_trio(&self, kind: AttentionKind) -> &OpTrio<AttentionDesc> {
        match kind {
            AttentionKind::Decode => &self.syms().attention_decode,
            _ => &self.syms().attention_prefill,
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

fn add_probe(cfg: &ElementwiseConfig) -> AddDesc {
    AddDesc {
        a: null(),
        b: null(),
        out: null(),
        n: 1,
        dtype: cfg.dtype.abi_code(),
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
        // ABI v1 has contiguous per-sequence KV only; paged attention arrives with v2.
        cfg.block_tokens.is_none()
            && Self::supported(self.attention_trio(cfg.kind), &attention_probe(cfg))
    }

    fn implementation(&self, cfg: &AttentionConfig) -> String {
        Self::implementation_of(self.attention_trio(cfg.kind), &attention_probe(cfg))
    }

    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
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
        self.run(self.attention_trio(ctx.cfg.kind), &d)
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
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use turbine_core::types::{DType, DeviceId, ExecutionBackend, MemoryKind, Vendor};
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
                    expected: 1,
                    found: 999
                }
            ),
            "{err:?}"
        );
        assert_eq!(
            err.to_string(),
            "kernel ABI version mismatch: library 999, expected 1"
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
        let lib = ShimLibrary::load(
            Path::new(env!("TURBINE_STUB_GFX942")),
            ExecutionBackend::Hip,
        )
        .expect("load");
        assert_eq!(lib.abi_version(), 1);
        assert_eq!(lib.backend_name(), "hip");
        let before = live_contexts(&lib);
        let ctx = lib
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        assert_eq!(live_contexts(&lib), before + 1);
        assert!(Arc::ptr_eq(ctx.library(), &lib));

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
        ];
        for (which, rust) in rust_sizes.into_iter().enumerate() {
            assert_eq!(c_size(which as i32), rust, "descriptor {which}");
        }
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
