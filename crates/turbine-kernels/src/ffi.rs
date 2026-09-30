//! `#[repr(C)]` mirrors of `kernels/include/turbine_kernels.h` (ABI v2.7), the symbol table
//! resolved once per loaded library (the minor groups v2.1–v2.7 optionally), and the status-code
//! mapping (contract §9.4).
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
    /// v2.10: YaRN's attention factor on cos/sin (1.0 = none); a library below minor 10 does
    /// not read it and is only ever handed 1.0 ([`V21Symbols::rope_attn_factor`]).
    pub attn_factor: f32,
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
    /// v2.9: per-layer K / V scales of FP8 pages (read only by a library of minor ≥ 9; 1.0
    /// otherwise).
    pub k_scale: f32,
    pub v_scale: f32,
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

/// `TURBINE_MOE_ROUTE_RENORMALIZE`: `MoeRouteDesc::flags` bit dividing the selected weights by
/// their sum (v2).
pub(crate) const MOE_ROUTE_RENORMALIZE: i32 = 1;
/// `TURBINE_MOE_ROUTE_BF16_LOGITS`: `MoeRouteDesc::flags` bit rounding each logit to BF16 before
/// the softmax (v2.2; a library of an earlier minor reports such a descriptor unsupported).
pub(crate) const MOE_ROUTE_BF16_LOGITS: i32 = 2;

/// `turbine_moe_route_desc` (v2; `flags` was `renormalize`, whose values 0 and 1 keep their
/// meaning).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeRouteDesc {
    pub router_logits: *const f32,
    pub num_tokens: i32,
    pub num_experts: i32,
    pub top_k: i32,
    pub flags: i32,
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

/// Opaque `turbine_graph` (v2.1); only ever handled by pointer.
#[repr(C)]
pub(crate) struct TurbineGraph {
    _private: [u8; 0],
}

/// Opaque `turbine_event` (v2.3); only ever handled by pointer.
#[repr(C)]
pub(crate) struct TurbineEvent {
    _private: [u8; 0],
}

/// Opaque `turbine_stream` (v2.3: only null, the compute stream, is passed; v2.5: copy streams).
#[repr(C)]
pub(crate) struct TurbineStream {
    _private: [u8; 0],
}

/// `turbine_add_rmsnorm_desc` (v2.1).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddRmsnormDesc {
    pub residual: *mut c_void,
    pub x: *const c_void,
    pub weight: *const c_void,
    pub out: *mut c_void,
    pub rows: i64,
    pub dim: i64,
    pub residual_stride_row: i64,
    pub x_stride_row: i64,
    pub out_stride_row: i64,
    pub eps: f32,
    pub dtype: i32,
}

/// `turbine_logits_reduce_desc` (v2.1).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogitsReduceDesc {
    pub logits: *const f32,
    pub rows: i64,
    pub vocab: i64,
    pub stride_row: i64,
    pub temperature: *const f32,
    pub uniform: *const f32,
    pub mode: *const i32,
    pub top_n: i32,
    pub top_ids: *mut i32,
    pub top_values: *mut f32,
    pub lse: *mut f32,
    pub sampled: *mut i32,
    pub sampled_logit: *mut f32,
    /// `[rows]`, or null for 1 on every row.
    pub top_p: *const f32,
}

/// `turbine_row_sumsq_desc` (v2.6).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowSumsqDesc {
    pub x: *const c_void,
    pub sumsq: *mut f32,
    pub rows: i64,
    pub dim: i64,
    pub x_stride_row: i64,
    pub dtype: i32,
}

/// `turbine_qgemm_desc` (v2.9).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct QGemmDesc {
    pub a: *const c_void,
    pub a_scales: *const f32,
    pub b: *const c_void,
    pub b_scales: *const c_void,
    pub b_zeros: *const u8,
    pub c: *mut c_void,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub lda: i64,
    pub ldc: i64,
    pub scheme: i32,
    pub act_quant: i32,
    pub a_dtype: i32,
    pub c_dtype: i32,
    pub group_size: i32,
    pub block_n: i32,
    pub block_k: i32,
    pub alpha: f32,
    pub prefill: i32,
}

/// `turbine_quantize_act_desc` (v2.9).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuantizeActDesc {
    pub x: *const c_void,
    pub out: *mut c_void,
    pub scales: *mut f32,
    pub rows: i64,
    pub cols: i64,
    pub x_stride_row: i64,
    pub out_stride_row: i64,
    pub mode: i32,
    pub static_scale: f32,
    pub x_dtype: i32,
    pub out_dtype: i32,
}

/// `turbine_kv_transcode_desc` (v2.11).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct KvTranscodeDesc {
    pub pages: *const *mut c_void,
    pub k_scales: *const f32,
    pub v_scales: *const f32,
    pub coded: *mut c_void,
    pub coded_block_bytes: i64,
    pub seed: u64,
    pub num_blocks: i32,
    pub layers: i32,
    pub block_tokens: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub page_dtype: i32,
    pub format: i32,
    pub direction: i32,
}

/// `turbine_rmsnorm_sharded_desc` (v2.6).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsnormShardedDesc {
    pub x: *const c_void,
    pub weight: *const c_void,
    pub sumsq: *const f32,
    pub out: *mut c_void,
    pub rows: i64,
    pub dim: i64,
    pub full_dim: i64,
    pub x_stride_row: i64,
    pub out_stride_row: i64,
    pub eps: f32,
    pub dtype: i32,
}

/// `turbine_mapped_collective_desc` (v2.7).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MappedCollectiveDesc {
    pub send: *const c_void,
    pub recv: *mut c_void,
    pub bytes: i64,
    pub send_stride: i64,
    pub recv_stride: i64,
    pub slots: *mut c_void,
    pub slot_bytes: i64,
    pub flags: *mut u64,
    pub abort_word: *mut u32,
    pub seq: u64,
    pub timeout_ns: i64,
    pub kind: i32,
    pub reduce_op: i32,
    pub dtype: i32,
    pub rank: i32,
    pub world: i32,
    pub root: i32,
    pub max_blocks: i32,
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

/// `turbine_ctx_set_option` / `turbine_ctx_get_option` (v2.1).
#[derive(Clone, Copy)]
pub(crate) struct OptionFns {
    pub set: unsafe extern "C" fn(*mut TurbineCtx, i32, i64) -> i32,
    pub get: unsafe extern "C" fn(*mut TurbineCtx, i32, *mut i64) -> i32,
}

/// `turbine_graph_{begin,end,launch,destroy}` (v2.1).
#[derive(Clone, Copy)]
pub(crate) struct GraphFns {
    pub begin: unsafe extern "C" fn(*mut TurbineCtx) -> i32,
    pub end: unsafe extern "C" fn(*mut TurbineCtx, *mut *mut TurbineGraph) -> i32,
    pub launch: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineGraph) -> i32,
    pub destroy: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineGraph) -> i32,
}

/// `turbine_host_{alloc,free}_pinned` and `turbine_event_{create,record,synchronize,destroy}`
/// (v2.3, the compute-stream subset of the v3 functions).
#[derive(Clone, Copy)]
pub(crate) struct StagingFns {
    pub host_alloc: unsafe extern "C" fn(*mut TurbineCtx, usize, *mut *mut c_void) -> i32,
    pub host_free: unsafe extern "C" fn(*mut TurbineCtx, *mut c_void) -> i32,
    pub event_create: unsafe extern "C" fn(*mut TurbineCtx, *mut *mut TurbineEvent) -> i32,
    pub event_record:
        unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineEvent, *mut TurbineStream) -> i32,
    pub event_synchronize: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineEvent) -> i32,
    pub event_destroy: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineEvent) -> i32,
}

/// `TURBINE_COPY_*` (v2.5): the direction of `turbine_memcpy_async`.
pub(crate) const COPY_H2D: i32 = 0;
pub(crate) const COPY_D2H: i32 = 1;
pub(crate) const COPY_D2D: i32 = 2;

/// The v2.5 copy-stream group: `turbine_copy_stream_{create,destroy}`, `turbine_memcpy_async`,
/// `turbine_event_query` and `turbine_stream_wait_event` (used with the v2.3 staging group).
#[derive(Clone, Copy)]
pub(crate) struct CopyFns {
    pub stream_create: unsafe extern "C" fn(*mut TurbineCtx, *mut *mut TurbineStream) -> i32,
    pub stream_destroy: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineStream) -> i32,
    pub memcpy_async: unsafe extern "C" fn(
        *mut TurbineCtx,
        *mut TurbineStream,
        *mut c_void,
        *const c_void,
        usize,
        i32,
    ) -> i32,
    pub event_query: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineEvent) -> i32,
    pub stream_wait_event:
        unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineStream, *mut TurbineEvent) -> i32,
}

/// The v2.6 tensor-parallel group: `turbine_stream_native_handle` and the `row_sumsq` and
/// `rmsnorm_sharded` trios.
#[derive(Clone, Copy)]
pub(crate) struct TensorParallelFns {
    pub stream_native_handle:
        unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineStream, *mut *mut c_void) -> i32,
    pub row_sumsq: OpTrio<RowSumsqDesc>,
    pub rmsnorm_sharded: OpTrio<RmsnormShardedDesc>,
}

/// The v2.9 quantization group: the `qgemm` and `quantize_act` trios.
#[derive(Clone, Copy)]
pub(crate) struct QuantFns {
    pub qgemm: OpTrio<QGemmDesc>,
    pub quantize_act: OpTrio<QuantizeActDesc>,
}

/// The v2.7 host-mapped group: `turbine_host_{alloc,free}_mapped`,
/// `turbine_host_mapped_device_ptr` and the `mapped_collective` trio.
#[derive(Clone, Copy)]
pub(crate) struct MappedFns {
    pub alloc: unsafe extern "C" fn(*mut TurbineCtx, usize, *mut *mut c_void) -> i32,
    pub device_ptr: unsafe extern "C" fn(*mut TurbineCtx, *mut c_void, *mut *mut c_void) -> i32,
    pub free: unsafe extern "C" fn(*mut TurbineCtx, *mut c_void) -> i32,
    pub collective: OpTrio<MappedCollectiveDesc>,
}

/// `turbine_mapped_collective_dseq` (v2.8): a mapped collective step sequenced by a device
/// counter (graph-capturable).
pub(crate) type MappedDseqFn =
    unsafe extern "C" fn(*mut TurbineCtx, *const MappedCollectiveDesc, *mut u64) -> i32;

/// `turbine_mapped_dma_desc` (v2.8).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MappedDmaDesc {
    pub copy: *mut TurbineStream,
    pub event: *mut TurbineEvent,
    pub scratch: *mut c_void,
    pub chunk_bytes: i64,
    pub seq_counter: *mut u64,
    pub flags: i32,
}

/// `turbine_mapped_all_reduce_dma` (v2.8): the copy-engine all-reduce.
pub(crate) type MappedDmaFn =
    unsafe extern "C" fn(*mut TurbineCtx, *const MappedCollectiveDesc, *const MappedDmaDesc) -> i32;

/// The v2.8 copy-engine group: `turbine_mapped_all_reduce_dma` and `turbine_host_alloc_dma`
/// (freed with the v2.7 `turbine_host_free_mapped`).
#[derive(Clone, Copy)]
pub(crate) struct MappedDmaFns {
    pub run: MappedDmaFn,
    pub alloc: unsafe extern "C" fn(*mut TurbineCtx, usize, *mut *mut c_void) -> i32,
}

/// `TURBINE_IMPL_NEEDS_HOST_OFFSETS` (v2.4): the implementation reads `host_expert_offsets`.
pub(crate) const IMPL_NEEDS_HOST_OFFSETS: u32 = 1;

/// `turbine_impl_entry` (v2.4): one implementation of an op; both strings are static storage
/// owned by the library.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImplEntry {
    pub name: *const c_char,
    pub provider: *const c_char,
    pub flags: u32,
}

impl ImplEntry {
    pub(crate) fn zeroed() -> ImplEntry {
        ImplEntry {
            name: std::ptr::null(),
            provider: std::ptr::null(),
            flags: 0,
        }
    }
}

/// `turbine_card_profile` (v2.4): the card profile's thresholds, copied into the context by
/// `turbine_ctx_set_profile`. `arch` is a host string the library does not retain.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CardProfileDesc {
    pub struct_bytes: u32,
    pub arch: *const c_char,
    pub wave_size: i32,
    pub lds_bytes: i32,
    pub moe_small_max_rows: i64,
    pub paged_page_multiple: i32,
}

/// The five v2.4 functions: implementation enumeration, explicit runs and the card profile.
#[derive(Clone, Copy)]
pub(crate) struct ImplFns {
    pub count: unsafe extern "C" fn(i32) -> i32,
    pub info: unsafe extern "C" fn(i32, i32, *mut ImplEntry) -> i32,
    pub supports: unsafe extern "C" fn(i32, i32, *const c_void) -> i32,
    pub run: unsafe extern "C" fn(*mut TurbineCtx, i32, i32, *const c_void) -> i32,
    pub set_profile: unsafe extern "C" fn(*mut TurbineCtx, *const CardProfileDesc) -> i32,
}

/// The optional ABI v2.1–v2.11 functions. `minor` is `turbine_abi_minor()` (0 when the library
/// lacks it); every v2.1 group is `None` unless `minor` ≥ 1, `staging` unless `minor` ≥ 3,
/// `impls` unless `minor` ≥ 4, `copies` unless `minor` ≥ 5 and `staging` is resolved,
/// `tensor_parallel` unless `minor` ≥ 6, `mapped` unless `minor` ≥ 7, and each only when the
/// library exports the whole group; `rope_attn_factor` (a descriptor field, no symbol) is set
/// from minor 10, and `kv_transcode` needs minor ≥ 11 with the `quant` (v2.9) and `copies`
/// (v2.5) groups resolved.
#[derive(Clone, Copy, Default)]
pub(crate) struct V21Symbols {
    pub minor: u32,
    pub options: Option<OptionFns>,
    pub add_rmsnorm: Option<OpTrio<AddRmsnormDesc>>,
    pub logits_reduce: Option<OpTrio<LogitsReduceDesc>>,
    pub graph: Option<GraphFns>,
    /// v2.3 host staging memory and events.
    pub staging: Option<StagingFns>,
    /// v2.4 implementation enumeration and the card profile.
    pub impls: Option<ImplFns>,
    /// v2.5 copy streams and asynchronous copies (Phase 4 KV tiers).
    pub copies: Option<CopyFns>,
    /// v2.6 native stream handles and the sharded RMSNorm ops (Phase 5 tensor parallelism).
    pub tensor_parallel: Option<TensorParallelFns>,
    /// v2.7 host-mapped memory and one-shot collectives (the `hostmem` collective backend).
    pub mapped: Option<MappedFns>,
    /// v2.8 device-sequenced mapped collective steps (needs the v2.7 group).
    pub mapped_dseq: Option<MappedDseqFn>,
    /// v2.8 copy-engine all-reduce (needs the v2.7 group and the v2.3 / v2.5 copy functions).
    pub mapped_dma: Option<MappedDmaFns>,
    /// v2.9 quantized GEMM and activation quantization (Phase 6a).
    pub quant: Option<QuantFns>,
    /// v2.10: the library reads `turbine_rope_desc.attn_factor` (minor ≥ 10; no new symbol).
    pub rope_attn_factor: bool,
    /// v2.11 KV transcode (Phase 6b); needs the v2.9 and v2.5 groups.
    pub kv_transcode: Option<OpTrio<KvTranscodeDesc>>,
}

impl V21Symbols {
    /// Resolves the v2.1 functions of `lib`. Missing symbols are not errors: a v2.0 library (no
    /// `turbine_abi_minor`, or minor 0) gets no v2.1 group at all, and a group the library exports
    /// only in part is left out, so callers fall back to the ABI v2 paths.
    pub(crate) fn resolve(lib: &Library) -> V21Symbols {
        let Some(abi_minor) = optional::<unsafe extern "C" fn() -> u32>(lib, "turbine_abi_minor")
        else {
            return V21Symbols::default();
        };
        // SAFETY: `turbine_abi_minor` takes no arguments and returns an integer (header v2.1);
        // `lib` is loaded for the duration of the call.
        let minor = unsafe { abi_minor() };
        if minor == 0 {
            return V21Symbols::default();
        }
        let options = (|| {
            Some(OptionFns {
                set: optional(lib, "turbine_ctx_set_option")?,
                get: optional(lib, "turbine_ctx_get_option")?,
            })
        })();
        let graph = (|| {
            Some(GraphFns {
                begin: optional(lib, "turbine_graph_begin")?,
                end: optional(lib, "turbine_graph_end")?,
                launch: optional(lib, "turbine_graph_launch")?,
                destroy: optional(lib, "turbine_graph_destroy")?,
            })
        })();
        let staging = (|| {
            if minor < 3 {
                return None;
            }
            Some(StagingFns {
                host_alloc: optional(lib, "turbine_host_alloc_pinned")?,
                host_free: optional(lib, "turbine_host_free_pinned")?,
                event_create: optional(lib, "turbine_event_create")?,
                event_record: optional(lib, "turbine_event_record")?,
                event_synchronize: optional(lib, "turbine_event_synchronize")?,
                event_destroy: optional(lib, "turbine_event_destroy")?,
            })
        })();
        let impls = (|| {
            if minor < 4 {
                return None;
            }
            Some(ImplFns {
                count: optional(lib, "turbine_impl_count")?,
                info: optional(lib, "turbine_impl_info")?,
                supports: optional(lib, "turbine_impl_supports")?,
                run: optional(lib, "turbine_impl_run")?,
                set_profile: optional(lib, "turbine_ctx_set_profile")?,
            })
        })();
        let copies = (|| {
            if minor < 5 || staging.is_none() {
                return None;
            }
            Some(CopyFns {
                stream_create: optional(lib, "turbine_copy_stream_create")?,
                stream_destroy: optional(lib, "turbine_copy_stream_destroy")?,
                memcpy_async: optional(lib, "turbine_memcpy_async")?,
                event_query: optional(lib, "turbine_event_query")?,
                stream_wait_event: optional(lib, "turbine_stream_wait_event")?,
            })
        })();
        let tensor_parallel = (|| {
            if minor < 6 {
                return None;
            }
            Some(TensorParallelFns {
                stream_native_handle: optional(lib, "turbine_stream_native_handle")?,
                row_sumsq: optional_trio(lib, "row_sumsq")?,
                rmsnorm_sharded: optional_trio(lib, "rmsnorm_sharded")?,
            })
        })();
        let mapped = (|| {
            if minor < 7 {
                return None;
            }
            Some(MappedFns {
                alloc: optional(lib, "turbine_host_alloc_mapped")?,
                device_ptr: optional(lib, "turbine_host_mapped_device_ptr")?,
                free: optional(lib, "turbine_host_free_mapped")?,
                collective: optional_trio(lib, "mapped_collective")?,
            })
        })();
        let mapped_dseq = if minor >= 8 && mapped.is_some() {
            optional(lib, "turbine_mapped_collective_dseq")
        } else {
            None
        };
        let mapped_dma = (|| {
            if minor < 8 || mapped.is_none() || copies.is_none() {
                return None;
            }
            Some(MappedDmaFns {
                run: optional(lib, "turbine_mapped_all_reduce_dma")?,
                alloc: optional(lib, "turbine_host_alloc_dma")?,
            })
        })();
        let quant = (|| {
            if minor < 9 {
                return None;
            }
            Some(QuantFns {
                qgemm: optional_trio(lib, "qgemm")?,
                quantize_act: optional_trio(lib, "quantize_act")?,
            })
        })();
        let kv_transcode = if minor >= 11 && quant.is_some() && copies.is_some() {
            optional_trio(lib, "kv_transcode")
        } else {
            None
        };
        V21Symbols {
            minor,
            rope_attn_factor: minor >= 10,
            kv_transcode,
            mapped_dseq,
            mapped_dma,
            quant,
            options,
            add_rmsnorm: optional_trio(lib, "add_rmsnorm"),
            logits_reduce: optional_trio(lib, "logits_reduce"),
            graph,
            staging,
            impls,
            copies,
            tensor_parallel,
            mapped,
        }
    }
}

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
    /// Optional `turbine_moe_experts_needs_host_offsets` (same signature as `_supported`);
    /// `None` when the library does not export it, and then `host_expert_offsets` is always
    /// passed.
    pub moe_experts_needs_host_offsets: Option<SupportedFn<MoeExpertsDesc>>,
    /// The optional v2.1 additions.
    pub v21: V21Symbols,
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

/// Resolves the function `name` from `lib` as `T`, or `None` when the library does not export
/// it (an optional v2.1 symbol).
///
/// Callers must pick `T` as the `unsafe extern "C" fn` type the header declares for `name`.
fn optional<T: Copy>(lib: &Library, name: &str) -> Option<T> {
    // SAFETY: as in `resolve`: `T` matches the header's declaration of `name`, and the copied
    // pointer stays valid while `lib` is loaded (`ShimLibrary` keeps it alive with the table).
    unsafe { lib.get::<T>(name) }.ok().map(|symbol| *symbol)
}

/// The three entry points of the optional op `op`, or `None` unless all three are exported.
fn optional_trio<D>(lib: &Library, op: &str) -> Option<OpTrio<D>> {
    Some(OpTrio {
        run: optional(lib, &format!("turbine_{op}"))?,
        supported: optional(lib, &format!("turbine_{op}_supported"))?,
        implementation: optional(lib, &format!("turbine_{op}_impl"))?,
    })
}

fn trio<D>(lib: &Library, path: &Path, op: &str) -> Result<OpTrio<D>, KernelError> {
    Ok(OpTrio {
        run: resolve(lib, path, &format!("turbine_{op}"))?,
        supported: resolve(lib, path, &format!("turbine_{op}_supported"))?,
        implementation: resolve(lib, path, &format!("turbine_{op}_impl"))?,
    })
}

impl ShimSymbols {
    /// Resolves every ABI v2 function (the first missing one fails the load) and the optional
    /// v2.1 functions (never a failure).
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
            moe_experts_needs_host_offsets: optional(lib, "turbine_moe_experts_needs_host_offsets"),
            v21: V21Symbols::resolve(lib),
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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use turbine_core::types::DType;
    use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor};

    use crate::KernelError;
    use crate::ops::{
        KernelProvider, KvCodecFns, KvTranscodeConfig, KvTranscodeContext, KvTranscodeFormat,
        OpKind, RopeConfig, RopeContext,
    };
    use crate::registry::OpConfig;
    use crate::shim::tests::{STUB_CONTEXTS, mocked_device, stub_hook};
    use crate::shim::{ShimLibrary, shim_provider};

    /// Calls of `turbine_rope` the stub has seen and the `attn_factor` of the last one.
    fn rope_calls(lib: &ShimLibrary) -> (i32, f32) {
        let calls = stub_hook(
            lib,
            "stub_rope_calls",
            |f: unsafe extern "C" fn() -> i32| {
                // SAFETY: the stub defines `int32_t stub_rope_calls(void)`; the library is loaded.
                unsafe { f() }
            },
        );
        let last = stub_hook(
            lib,
            "stub_rope_last_attn_factor",
            |f: unsafe extern "C" fn() -> f32| {
                // SAFETY: the stub defines `float stub_rope_last_attn_factor(void)`.
                unsafe { f() }
            },
        );
        (calls, last)
    }

    /// Runs the provider's rope on 1-token device tensors with `attn_factor`.
    fn run_rope(
        provider: &Arc<dyn KernelProvider>,
        mem: &Arc<dyn DeviceMemory>,
        attn_factor: f32,
    ) -> Result<(), KernelError> {
        let cfg = RopeConfig {
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 4,
            dtype: DType::BF16,
        };
        let q = Tensor::empty(mem, &[1, 1, 4], DType::BF16).expect("q");
        let k = Tensor::empty(mem, &[1, 1, 4], DType::BF16).expect("k");
        let positions = Tensor::empty(mem, &[1], DType::I32).expect("positions");
        let inv_freq = Tensor::empty(mem, &[2], DType::F32).expect("inv_freq");
        provider
            .rope()
            .expect("rope family")
            .execute(&mut RopeContext {
                cfg,
                q: q.view(),
                k: k.view(),
                positions: positions.view(),
                inv_freq: inv_freq.view(),
                attn_factor,
            })
    }

    /// ABI v2.10 (Phase 6a Task 28a): `turbine_rope_desc.attn_factor` is handed only to a
    /// library at minor ≥ 10. A minor-9 library reports no support, is refused a factor ≠ 1
    /// with `rope_attn_factor_unavailable` before the library is called, and gets 1.0
    /// otherwise; a minor-10 library reports support and receives the factor as passed.
    /// Breaks if an older library is handed a factor it would ignore (a silent YaRN error) or
    /// the factor does not reach a v2.10 library.
    #[test]
    fn optional_groups_v210_rope() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let m = 1.277_258_9_f32;

        let v29 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V29")), "hip")
            .expect("a v2.9 library loads");
        assert_eq!(v29.abi_minor(), 9);
        assert!(!v29.rope_attn_factor());
        let ctx = v29
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let provider = shim_provider(Arc::clone(&ctx));
        assert!(!provider.rope().expect("rope").attn_factor_supported());
        let before = rope_calls(&v29).0;
        let err = run_rope(&provider, &mem, m).expect_err("a factor on a minor-9 library");
        assert!(
            err.to_string().contains("rope_attn_factor_unavailable"),
            "{err}"
        );
        assert_eq!(rope_calls(&v29).0, before, "the library must not be called");
        // Factor 1: the call reaches the library (the stub's rope is unsupported) with 1.0.
        let err = run_rope(&provider, &mem, 1.0).expect_err("the stub rope fails");
        assert!(err.to_string().contains("stub: rope"), "{err}");
        assert_eq!(rope_calls(&v29), (before + 1, 1.0));
        drop((provider, mem, ctx));

        let v210 = ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V210")), "hip")
            .expect("a v2.10 library loads");
        assert_eq!((v210.abi_version(), v210.abi_minor()), (2, 10));
        assert!(v210.rope_attn_factor());
        let ctx = v210
            .create_context(&mocked_device("gfx942"))
            .expect("context");
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let provider = shim_provider(Arc::clone(&ctx));
        assert!(provider.rope().expect("rope").attn_factor_supported());
        let before = rope_calls(&v210).0;
        let err = run_rope(&provider, &mem, m).expect_err("the stub rope fails");
        assert!(err.to_string().contains("stub: rope"), "{err}");
        assert_eq!(rope_calls(&v210), (before + 1, m));
        drop((provider, mem, ctx));
    }
    /// A codec table the stub never calls.
    struct NoCodecs;

    impl KvCodecFns for NoCodecs {
        fn encode(
            &self,
            _: &KvTranscodeConfig,
            _: u64,
            _: (&[f32], &[f32]),
            _: &[u8],
            _: &mut [u8],
        ) -> Result<(), String> {
            Err("no codecs".into())
        }

        fn decode(
            &self,
            _: &KvTranscodeConfig,
            _: u64,
            _: (&[f32], &[f32]),
            _: &[u8],
            _: &mut [u8],
        ) -> Result<(), String> {
            Err("no codecs".into())
        }
    }

    /// ABI v2.11 (Phase 6b Task 5): the KV transcode group resolves only on a library of minor
    /// 11 or later that exports the whole trio and the v2.9 and v2.5 groups it depends on. A
    /// minor-10 library has no `kv_transcode` (and is not asked for its op code); minor 11
    /// resolves it, and the stub names its implementation; a library missing one symbol of the
    /// trio, or the v2.9 group, gets none of it. A well-formed call reaches the library, a
    /// malformed one is refused before. Breaks if the group resolves partially or on an older
    /// library (a demotion would call a symbol that is not there), or is missing on a v2.11 one.
    #[test]
    fn optional_groups_v211() {
        let _serial = STUB_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        let cfg = KvTranscodeConfig {
            src_format: KvTranscodeFormat::L0,
            dst_format: KvTranscodeFormat::Fp8E4m3,
            page_dtype: DType::BF16,
            head_dim: 128,
            num_kv_heads: 8,
            block_tokens: 128,
            layers: 28,
        };
        let provider_of = |path: &str| {
            let lib = ShimLibrary::load(Path::new(path), "hip").expect("stub library loads");
            let ctx = lib
                .create_context(&mocked_device("gfx942"))
                .expect("context");
            (lib, ctx)
        };

        let (v210, ctx) = provider_of(env!("TURBINE_STUB_GFX942_V210"));
        assert_eq!(v210.abi_minor(), 10);
        let provider = shim_provider(Arc::clone(&ctx));
        assert!(provider.kv_transcode().is_none());
        assert!(!OpConfig::KvTranscode(cfg).supported_by(provider.as_ref()));
        assert!(v210.implementations(OpKind::KvTranscode).is_empty());
        drop((provider, ctx));

        for (path, what) in [
            (env!("TURBINE_STUB_GFX942_V211_PARTIAL"), "missing _impl"),
            (env!("TURBINE_STUB_GFX942_V211_NOQUANT"), "no v2.9 group"),
        ] {
            let (lib, ctx) = provider_of(path);
            assert_eq!(lib.abi_minor(), 11, "{what}");
            let provider = shim_provider(Arc::clone(&ctx));
            assert!(provider.kv_transcode().is_none(), "{what}");
            assert!(!OpConfig::KvTranscode(cfg).supported_by(provider.as_ref()));
            drop((provider, ctx));
        }

        let (v211, ctx) = provider_of(env!("TURBINE_STUB_GFX942_V211"));
        assert_eq!((v211.abi_version(), v211.abi_minor()), (2, 11));
        let provider = shim_provider(Arc::clone(&ctx));
        let kernel = provider.kv_transcode().expect("the v2.11 family");
        assert!(!kernel.supports(&cfg), "the stub supports nothing");
        assert_eq!(kernel.implementation(&cfg), "stub_kv_transcode");
        let impls = v211.implementations(OpKind::KvTranscode);
        assert_eq!(impls.len(), 1);
        assert_eq!(impls[0].name, "stub_kv_transcode");

        // A call shaped like a demotion of one block: the stub refuses it, having been called.
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let page = cfg.page_bytes();
        let pages: Vec<DeviceBuffer> = (0..cfg.layers)
            .map(|_| DeviceBuffer::alloc(&mem, page).expect("page"))
            .collect();
        let slots: Vec<_> = pages.iter().map(DeviceBuffer::whole).collect();
        let slot = cfg.layers as usize * (8 + page / 2);
        let coded = DeviceBuffer::alloc(&mem, slot).expect("coded");
        let call = |pages: &[turbine_tensor::DeviceSlice<'_>], bytes: usize| {
            kernel.execute(&mut KvTranscodeContext {
                cfg,
                pages,
                coded: coded.slice(0, bytes),
                coded_block_bytes: slot,
                seed: 0,
                k_scales: None,
                v_scales: None,
                codecs: &NoCodecs,
            })
        };
        let err = call(&slots, slot).expect_err("the stub kv_transcode fails");
        assert!(err.to_string().contains("stub: kv_transcode"), "{err}");
        // Malformed: a page missing, a coded buffer that is not whole slots.
        let err = call(&slots[1..], slot).expect_err("27 pages for 28 layers");
        assert!(err.to_string().contains("pages for 1 blocks"), "{err}");
        let err = call(&slots, slot - 1).expect_err("a partial slot");
        assert!(err.to_string().contains("whole slots"), "{err}");
        drop((pages, coded, provider, mem, ctx));
    }
}
