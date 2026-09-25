//! `#[repr(C)]` mirrors of `kernels/include/turbine_kernels.h` (ABI v2), the symbol table
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

/// `turbine_ctx_info` (v2): fixed properties of a context.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CtxInfo {
    pub workspace_bytes: u64,
    /// -1 when not applicable (AMD).
    pub compute_major: i32,
    pub compute_minor: i32,
    /// NUL-terminated device architecture name.
    pub device_arch: [c_char; 32],
}

impl CtxInfo {
    pub(crate) fn zeroed() -> CtxInfo {
        CtxInfo {
            workspace_bytes: 0,
            compute_major: -1,
            compute_minor: -1,
            device_arch: [0; 32],
        }
    }
}

/// `turbine_attention_paged_desc` (v2; prefill and decode share it).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionPagedDesc {
    pub q: *const c_void,
    pub k_new: *const c_void,
    pub v_new: *const c_void,
    pub out: *mut c_void,
    pub kv_layer: *mut c_void,
    pub block_table: *const i32,
    pub q_indptr: *const i32,
    pub kv_lens: *const i32,
    pub num_seqs: i32,
    pub total_q: i32,
    pub max_q_len: i32,
    pub max_kv_len: i32,
    pub max_blocks_per_seq: i32,
    pub num_blocks: i32,
    pub block_tokens: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub q_stride_token: i64,
    pub new_stride_token: i64,
    pub out_stride_token: i64,
    pub scale: f32,
    pub causal: i32,
    pub dtype: i32,
}

/// `turbine_copy_blocks_desc` (v2). `src_blocks`/`dst_blocks` are host arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CopyBlocksDesc {
    pub pool: *mut c_void,
    pub layer_stride_bytes: i64,
    pub block_bytes: i64,
    pub num_layers: i32,
    pub src_blocks: *const i32,
    pub dst_blocks: *const i32,
    pub count: i32,
}

/// `turbine_moe_route_desc` (v2).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeRouteDesc {
    pub router_logits: *const f32,
    pub num_tokens: i32,
    pub num_experts: i32,
    pub top_k: i32,
    pub renormalize: i32,
    pub topk_ids: *mut i32,
    pub topk_weights: *mut f32,
    pub sorted_rows: *mut i32,
    pub expert_offsets: *mut i32,
}

/// `turbine_moe_experts_desc` (v2). `host_expert_offsets` is a host array.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeExpertsDesc {
    pub x: *const c_void,
    pub w_gate: *const c_void,
    pub w_up: *const c_void,
    pub w_down: *const c_void,
    pub sorted_rows: *const i32,
    pub expert_offsets: *const i32,
    pub topk_weights: *const f32,
    pub host_expert_offsets: *const i32,
    pub out: *mut c_void,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub num_tokens: i32,
    pub hidden: i32,
    pub inter: i32,
    pub top_k: i32,
    pub num_experts: i32,
    pub expert_begin: i32,
    pub expert_end: i32,
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

/// Every function of ABI v2, resolved once in `ShimLibrary::load`. The pointers stay valid while
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
    pub ctx_get_info: unsafe extern "C" fn(*mut TurbineCtx, *mut CtxInfo) -> i32,
    pub attention_prefill_paged: OpTrio<AttentionPagedDesc>,
    pub attention_decode_paged: OpTrio<AttentionPagedDesc>,
    pub copy_blocks: OpTrio<CopyBlocksDesc>,
    pub moe_route: OpTrio<MoeRouteDesc>,
    pub moe_experts: OpTrio<MoeExpertsDesc>,
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
    /// Resolves every ABI v2 function; the first missing one fails the load.
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
            ctx_get_info: resolve(lib, path, "turbine_ctx_get_info")?,
            attention_prefill_paged: trio(lib, path, "attention_prefill_paged")?,
            attention_decode_paged: trio(lib, path, "attention_decode_paged")?,
            copy_blocks: trio(lib, path, "copy_blocks")?,
            moe_route: trio(lib, path, "moe_route")?,
            moe_experts: trio(lib, path, "moe_experts")?,
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
