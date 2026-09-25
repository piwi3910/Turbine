//! `#[repr(C)]` mirrors of `kernels/include/turbine_kernels.h` (ABI v1), the symbol table
//! resolved once per loaded library, and the status-code mapping (contract §9.4).
//!
//! Descriptor field order and types match the header field for field. Pointer fields carry
//! device pointers (except where the header says "host"); the shim never retains them beyond
//! the call.
use std::ffi::{CStr, c_char, c_void};
use std::path::Path;

use libloading::Library;

use crate::KernelError;

/// Opaque `turbine_ctx`; only ever handled by pointer.
#[repr(C)]
pub(crate) struct TurbineCtx {
    _private: [u8; 0],
}

/// `turbine_gemm_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmDesc {
    pub a: *const c_void,
    pub b: *const c_void,
    pub c: *mut c_void,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub lda: i64,
    pub ldb: i64,
    pub ldc: i64,
    pub trans_b: i32,
    pub a_dtype: i32,
    pub b_dtype: i32,
    pub c_dtype: i32,
    pub alpha: f32,
    pub beta: f32,
}

/// `turbine_attention_desc` (prefill and decode share it).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionDesc {
    pub q: *const c_void,
    pub k_cache: *const c_void,
    pub v_cache: *const c_void,
    pub out: *mut c_void,
    pub q_len: i32,
    pub q_start: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub q_stride_token: i64,
    pub kv_stride_token: i64,
    pub out_stride_token: i64,
    pub scale: f32,
    pub causal: i32,
    pub dtype: i32,
}

/// `turbine_rmsnorm_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsnormDesc {
    pub x: *const c_void,
    pub weight: *const c_void,
    pub out: *mut c_void,
    pub rows: i64,
    pub dim: i64,
    pub x_stride_row: i64,
    pub out_stride_row: i64,
    pub eps: f32,
    pub dtype: i32,
}

/// `turbine_rope_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeDesc {
    pub q: *mut c_void,
    pub k: *mut c_void,
    pub positions: *const i32,
    pub inv_freq: *const f32,
    pub num_tokens: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub q_stride_token: i64,
    pub k_stride_token: i64,
    pub style: i32,
    pub dtype: i32,
}

/// `turbine_silu_mul_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SiluMulDesc {
    pub gate: *const c_void,
    pub up: *const c_void,
    pub out: *mut c_void,
    pub rows: i64,
    pub cols: i64,
    pub gate_stride_row: i64,
    pub up_stride_row: i64,
    pub out_stride_row: i64,
    pub dtype: i32,
}

/// `turbine_embedding_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddingDesc {
    pub ids: *const i32,
    pub table: *const c_void,
    pub out: *mut c_void,
    pub num_tokens: i64,
    pub hidden: i64,
    pub vocab_offset: i64,
    pub vocab_rows: i64,
    pub out_stride_row: i64,
    pub dtype: i32,
}

/// `turbine_add_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddDesc {
    pub a: *const c_void,
    pub b: *const c_void,
    pub out: *mut c_void,
    pub n: i64,
    pub dtype: i32,
}

/// `turbine_<op>`: enqueue on the context's compute stream.
pub(crate) type OpFn<D> = unsafe extern "C" fn(*mut TurbineCtx, *const D) -> i32;
/// `turbine_<op>_supported`: 1 / 0 (negative on an internal error); pointers may be null.
pub(crate) type SupportedFn<D> = unsafe extern "C" fn(*const D) -> i32;
/// `turbine_<op>_impl`: a static string owned by the library.
pub(crate) type ImplFn<D> = unsafe extern "C" fn(*const D) -> *const c_char;

/// The three entry points of one op.
pub(crate) struct OpTrio<D> {
    pub run: OpFn<D>,
    pub supported: SupportedFn<D>,
    pub implementation: ImplFn<D>,
}

// Manual impls: the derives would require `D: Clone`/`D: Copy`, but fn pointers are always
// copyable.
impl<D> Clone for OpTrio<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D> Copy for OpTrio<D> {}

/// Every function of ABI v1, resolved once in `ShimLibrary::load`. The pointers stay valid while
/// the `libloading::Library` they came from is loaded; `ShimLibrary` owns both.
#[derive(Clone, Copy)]
pub(crate) struct ShimSymbols {
    pub abi_version: unsafe extern "C" fn() -> u32,
    pub backend_name: unsafe extern "C" fn() -> *const c_char,
    pub build_archs: unsafe extern "C" fn() -> *const c_char,
    pub ctx_create: unsafe extern "C" fn(i32, *mut *mut TurbineCtx) -> i32,
    pub ctx_destroy: unsafe extern "C" fn(*mut TurbineCtx),
    pub malloc: unsafe extern "C" fn(*mut TurbineCtx, usize, *mut *mut c_void) -> i32,
    pub free: unsafe extern "C" fn(*mut TurbineCtx, *mut c_void) -> i32,
    pub memcpy_h2d: unsafe extern "C" fn(*mut TurbineCtx, *mut c_void, *const c_void, usize) -> i32,
    pub memcpy_d2h: unsafe extern "C" fn(*mut TurbineCtx, *mut c_void, *const c_void, usize) -> i32,
    pub stream_sync: unsafe extern "C" fn(*mut TurbineCtx) -> i32,
    pub mem_info: unsafe extern "C" fn(*mut TurbineCtx, *mut usize, *mut usize) -> i32,
    pub last_error: unsafe extern "C" fn(*mut TurbineCtx, *mut c_char, usize) -> usize,
    pub gemm: OpTrio<GemmDesc>,
    pub attention_prefill: OpTrio<AttentionDesc>,
    pub attention_decode: OpTrio<AttentionDesc>,
    pub rmsnorm: OpTrio<RmsnormDesc>,
    pub rope: OpTrio<RopeDesc>,
    pub silu_mul: OpTrio<SiluMulDesc>,
    pub embedding: OpTrio<EmbeddingDesc>,
    pub add: OpTrio<AddDesc>,
}

/// Resolves the function `name` from `lib` as the fn-pointer type `T`; a missing symbol is a
/// `KernelError::Load` naming the library and the symbol.
///
/// Callers must pick `T` as the `unsafe extern "C" fn` type the header declares for `name`.
pub(crate) fn resolve<T: Copy>(lib: &Library, path: &Path, name: &str) -> Result<T, KernelError> {
    // SAFETY: every `T` this module requests is the `unsafe extern "C" fn` pointer type matching
    // the header's declaration of `name`. The pointer is copied out of the `Symbol` and stays
    // valid while `lib` is loaded; `ShimLibrary` keeps `lib` alive as long as the symbol table.
    let symbol = unsafe { lib.get::<T>(name) }.map_err(|e| KernelError::Load {
        path: path.to_path_buf(),
        detail: format!("missing symbol {name}: {e}"),
    })?;
    Ok(*symbol)
}

fn trio<D>(lib: &Library, path: &Path, op: &str) -> Result<OpTrio<D>, KernelError> {
    Ok(OpTrio {
        run: resolve(lib, path, &format!("turbine_{op}"))?,
        supported: resolve(lib, path, &format!("turbine_{op}_supported"))?,
        implementation: resolve(lib, path, &format!("turbine_{op}_impl"))?,
    })
}

impl ShimSymbols {
    /// Resolves every ABI v1 function; the first missing one fails the load.
    pub(crate) fn resolve_all(lib: &Library, path: &Path) -> Result<ShimSymbols, KernelError> {
        Ok(ShimSymbols {
            abi_version: resolve(lib, path, "turbine_abi_version")?,
            backend_name: resolve(lib, path, "turbine_backend_name")?,
            build_archs: resolve(lib, path, "turbine_build_archs")?,
            ctx_create: resolve(lib, path, "turbine_ctx_create")?,
            ctx_destroy: resolve(lib, path, "turbine_ctx_destroy")?,
            malloc: resolve(lib, path, "turbine_malloc")?,
            free: resolve(lib, path, "turbine_free")?,
            memcpy_h2d: resolve(lib, path, "turbine_memcpy_h2d")?,
            memcpy_d2h: resolve(lib, path, "turbine_memcpy_d2h")?,
            stream_sync: resolve(lib, path, "turbine_stream_sync")?,
            mem_info: resolve(lib, path, "turbine_mem_info")?,
            last_error: resolve(lib, path, "turbine_last_error")?,
            gemm: trio(lib, path, "gemm")?,
            attention_prefill: trio(lib, path, "attention_prefill")?,
            attention_decode: trio(lib, path, "attention_decode")?,
            rmsnorm: trio(lib, path, "rmsnorm")?,
            rope: trio(lib, path, "rope")?,
            silu_mul: trio(lib, path, "silu_mul")?,
            embedding: trio(lib, path, "embedding")?,
            add: trio(lib, path, "add")?,
        })
    }
}

/// Copies a NUL-terminated string returned by the shim (identity and `_impl` functions); null
/// becomes the empty string.
pub(crate) fn c_str(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the shim returns pointers to static NUL-terminated strings it owns (header,
    // "a static string owned by the library"); `p` is non-null and only read here.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Longest `turbine_last_error` message kept; longer messages are truncated.
const LAST_ERROR_CAPACITY: usize = 1024;

/// Maps a status code to `Ok` (0) or the `KernelError` of codes −1…−5, with the message of
/// `turbine_last_error(ctx)`; `ctx` is null for a failed `turbine_ctx_create`.
pub(crate) fn check(
    code: i32,
    syms: &ShimSymbols,
    ctx: *mut TurbineCtx,
) -> Result<(), KernelError> {
    if code == 0 {
        return Ok(());
    }
    let mut buf = vec![0u8; LAST_ERROR_CAPACITY];
    // SAFETY: `buf` is a live, writable buffer of exactly `buf.len()` bytes; the shim writes at
    // most that many bytes including the NUL and keeps no pointer to it. `ctx` is null or a
    // context created by this library and not yet destroyed (the caller holds it).
    let full_len = unsafe { (syms.last_error)(ctx, buf.as_mut_ptr().cast(), buf.len()) };
    buf.truncate(full_len.min(buf.len() - 1));
    let message = String::from_utf8_lossy(&buf).into_owned();
    Err(match code {
        -1 => KernelError::InvalidArgument { message },
        -2 => KernelError::Unsupported { message },
        -3 => KernelError::OutOfMemory { message },
        -4 => KernelError::Device { message },
        -5 => KernelError::Library { message },
        other => KernelError::Library {
            message: format!("unknown status {other}: {message}"),
        },
    })
}
