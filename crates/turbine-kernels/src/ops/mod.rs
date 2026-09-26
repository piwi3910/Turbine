//! Op families (TS §6, P1 S-6). Each family has a config (shapes and dtypes only, what
//! `supports` and `implementation` look at), a context (the tensor views `execute` reads and
//! writes) and a capability trait every provider implements. Nothing here names a vendor type,
//! so every backend provider implements the same traits.
//!
//! The compute stream is implicit: every provider owns its context and enqueues on that
//! context's compute stream (contract §9.2).
use std::fmt;

use turbine_core::types::{BlockId, DType};
use turbine_tensor::{DeviceSlice, TensorView};

use crate::KernelError;

/// One entry point of the kernel C ABI; `as_str` is the `turbine_<op>` suffix and the `op`
/// metric label.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum OpKind {
    Gemm,
    AttentionPrefill,
    AttentionDecode,
    Rmsnorm,
    Rope,
    SiluMul,
    Embedding,
    Add,
    AttentionPrefillPaged,
    AttentionDecodePaged,
    CopyBlocks,
    MoeRoute,
    MoeExperts,
    /// ABI v2.1 (optional in a shim library).
    AddRmsnorm,
    /// ABI v2.1 (optional in a shim library).
    LogitsReduce,
}

impl OpKind {
    /// Every op of the current ABI version (v2.1 included), in header order.
    pub const ALL: &'static [OpKind] = &[
        OpKind::Gemm,
        OpKind::AttentionPrefill,
        OpKind::AttentionDecode,
        OpKind::Rmsnorm,
        OpKind::Rope,
        OpKind::SiluMul,
        OpKind::Embedding,
        OpKind::Add,
        OpKind::AttentionPrefillPaged,
        OpKind::AttentionDecodePaged,
        OpKind::CopyBlocks,
        OpKind::MoeRoute,
        OpKind::MoeExperts,
        OpKind::AddRmsnorm,
        OpKind::LogitsReduce,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            OpKind::Gemm => "gemm",
            OpKind::AttentionPrefill => "attention_prefill",
            OpKind::AttentionDecode => "attention_decode",
            OpKind::Rmsnorm => "rmsnorm",
            OpKind::Rope => "rope",
            OpKind::SiluMul => "silu_mul",
            OpKind::Embedding => "embedding",
            OpKind::Add => "add",
            OpKind::AttentionPrefillPaged => "attention_prefill_paged",
            OpKind::AttentionDecodePaged => "attention_decode_paged",
            OpKind::CopyBlocks => "copy_blocks",
            OpKind::MoeRoute => "moe_route",
            OpKind::MoeExperts => "moe_experts",
            OpKind::AddRmsnorm => "add_rmsnorm",
            OpKind::LogitsReduce => "logits_reduce",
        }
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A provider's name: `cpu-reference`, or the shim library's backend name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProviderId(pub &'static str);

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

// ---------------------------------------------------------------------------------- configs
// `Display` renders the form used in selection logs and startup failures, e.g.
// `no kernel provider supports attention_prefill head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1`.

/// `c[m,n] = alpha · a[m,k] · op(b) + beta · c`; `m` (the token count) varies per call and is
/// not part of the config.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GemmConfig {
    pub n: u64,
    pub k: u64,
    /// `b` is `[n, k]` (an HF Linear weight) when true, `[k, n]` otherwise.
    pub trans_b: bool,
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub c_dtype: DType,
}

impl fmt::Display for GemmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={} k={} trans_b={} a_dtype={} b_dtype={} c_dtype={}",
            self.n,
            self.k,
            u8::from(self.trans_b),
            self.a_dtype.as_str(),
            self.b_dtype.as_str(),
            self.c_dtype.as_str()
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum AttentionKind {
    /// Contiguous per-sequence KV (Phase 1).
    Prefill,
    Decode,
    /// Ragged batch over the paged KV pool; the config carries `block_tokens`.
    PrefillPaged,
    DecodePaged,
}

impl AttentionKind {
    /// True for the kinds that append to and read the paged KV pool.
    pub fn is_paged(self) -> bool {
        matches!(
            self,
            AttentionKind::PrefillPaged | AttentionKind::DecodePaged
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AttentionConfig {
    pub kind: AttentionKind,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
    /// KV page size in tokens for paged attention; `None` for the contiguous per-sequence KV of
    /// Phase 1.
    pub block_tokens: Option<u32>,
    pub causal: bool,
}

impl AttentionConfig {
    /// The C ABI entry point this config binds to.
    pub fn op(&self) -> OpKind {
        match self.kind {
            AttentionKind::Prefill => OpKind::AttentionPrefill,
            AttentionKind::Decode => OpKind::AttentionDecode,
            AttentionKind::PrefillPaged => OpKind::AttentionPrefillPaged,
            AttentionKind::DecodePaged => OpKind::AttentionDecodePaged,
        }
    }
}

impl fmt::Display for AttentionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "head_dim={} kv_heads={} dtype={} q_heads={} causal={}",
            self.head_dim,
            self.num_kv_heads,
            self.dtype.as_str(),
            self.num_q_heads,
            u8::from(self.causal)
        )?;
        if let Some(block_tokens) = self.block_tokens {
            write!(f, " block_tokens={block_tokens}")?;
        }
        Ok(())
    }
}

/// RMSNorm over rows of `dim` elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NormConfig {
    pub dim: u64,
    pub dtype: DType,
}

impl fmt::Display for NormConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dim={} dtype={}", self.dim, self.dtype.as_str())
    }
}

/// Half-split rotary embedding (HF `rotate_half`) over the first `rotary_dim` of `head_dim`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RopeConfig {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub dtype: DType,
}

impl fmt::Display for RopeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "head_dim={} rotary_dim={} q_heads={} kv_heads={} dtype={}",
            self.head_dim,
            self.rotary_dim,
            self.num_q_heads,
            self.num_kv_heads,
            self.dtype.as_str()
        )
    }
}

/// `silu(gate) · up` over rows of `cols` elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ActivationConfig {
    pub cols: u64,
    pub dtype: DType,
}

impl fmt::Display for ActivationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cols={} dtype={}", self.cols, self.dtype.as_str())
    }
}

/// Row gather from a `[vocab_rows, hidden]` table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EmbeddingConfig {
    pub hidden: u64,
    pub vocab_rows: u64,
    pub dtype: DType,
}

impl fmt::Display for EmbeddingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hidden={} vocab_rows={} dtype={}",
            self.hidden,
            self.vocab_rows,
            self.dtype.as_str()
        )
    }
}

/// Elementwise `a + b`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ElementwiseConfig {
    pub dtype: DType,
}

impl fmt::Display for ElementwiseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dtype={}", self.dtype.as_str())
    }
}

/// Block fork across every layer of the KV pool (`copy_blocks`, `n > 1`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KvCopyConfig {
    pub num_layers: u32,
    /// Bytes of one block within one layer.
    pub block_bytes: u64,
}

impl fmt::Display for KvCopyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "num_layers={} block_bytes={}",
            self.num_layers, self.block_bytes
        )
    }
}

/// MoE router: softmax over `num_experts` F32 logits, top-`top_k`, optional renormalisation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MoeRouteConfig {
    pub num_experts: u32,
    pub top_k: u32,
    /// Divide the selected weights by their sum (`norm_topk_prob`; false for OLMoE-1B-7B).
    pub renormalize: bool,
}

impl fmt::Display for MoeRouteConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "experts={} top_k={} renormalize={}",
            self.num_experts,
            self.top_k,
            u8::from(self.renormalize)
        )
    }
}

/// SwiGLU experts `down(silu(gate(x)) · up(x))` of width `inter` over the local expert range
/// `[expert_begin, expert_end)` of `num_experts`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MoeExpertsConfig {
    pub hidden: u32,
    pub inter: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub expert_begin: u32,
    pub expert_end: u32,
    pub dtype: DType,
}

impl MoeExpertsConfig {
    /// Experts whose weights one call holds (`expert_end − expert_begin`).
    pub fn num_local_experts(&self) -> u32 {
        self.expert_end.saturating_sub(self.expert_begin)
    }
}

impl fmt::Display for MoeExpertsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hidden={} inter={} experts={} top_k={} local={}..{} dtype={}",
            self.hidden,
            self.inter,
            self.num_experts,
            self.top_k,
            self.expert_begin,
            self.expert_end,
            self.dtype.as_str()
        )
    }
}

/// Residual add fused with RMSNorm (ABI v2.1) over rows of `dim` elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddRmsnormConfig {
    pub dtype: DType,
    pub dim: u32,
}

impl fmt::Display for AddRmsnormConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dim={} dtype={}", self.dim, self.dtype.as_str())
    }
}

/// Per-row reduction of F32 logits rows of `vocab` values to their log-sum-exp, their `top_n`
/// (at most 64) largest values and, per row on request, one categorical draw (ABI v2.1).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LogitsReduceConfig {
    pub vocab: u32,
    pub top_n: u32,
}

impl LogitsReduceConfig {
    /// The largest `top_n` any provider must accept (the ABI bound).
    pub const MAX_TOP_N: u32 = 64;
}

impl fmt::Display for LogitsReduceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vocab={} top_n={}", self.vocab, self.top_n)
    }
}

// --------------------------------------------------------------------------------- contexts

/// `a`: `[m, k]`; `b`: `[n, k]` when `trans_b`, else `[k, n]`; `c`: `[m, n]`. Row strides are
/// the leading dimensions.
pub struct GemmContext<'a> {
    pub a: TensorView<'a>,
    pub b: TensorView<'a>,
    pub c: TensorView<'a>,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
}

/// A ragged batch of sequences over one layer of the paged KV pool. Sequence `s` owns query rows
/// `q_indptr[s]..q_indptr[s + 1]` of `q`/`k_new`/`v_new`/`out` and, after this call's append,
/// `kv_lens[s]` tokens of KV; its new tokens sit at positions `kv_lens[s] − q_len..kv_lens[s]`.
/// Token `p` of sequence `s` lives in block `block_table[s][p / block_tokens]` at slot
/// `p % block_tokens`. The op first writes `k_new`/`v_new` into their slots, then attends
/// (causal: new token `i` sees keys `0..=kv_lens[s] − q_len + i`).
///
/// - `q`/`out`: `[total_q, num_q_heads, head_dim]`; `k_new`/`v_new`:
///   `[total_q, num_kv_heads, head_dim]`
/// - `kv_layer`: this layer's pool `[num_blocks, 2, block_tokens, num_kv_heads, head_dim]`
///   (K then V within a block)
/// - `block_table`: `[num_seqs, max_blocks_per_seq]` I32; `q_indptr`: `[num_seqs + 1]` I32;
///   `kv_lens`: `[num_seqs]` I32
/// - `max_q_len`, `max_kv_len`: upper bounds of the per-sequence query and KV lengths
pub struct PagedAttentionContext<'a> {
    pub cfg: AttentionConfig,
    pub q: TensorView<'a>,
    pub k_new: TensorView<'a>,
    pub v_new: TensorView<'a>,
    pub out: TensorView<'a>,
    pub kv_layer: TensorView<'a>,
    pub block_table: TensorView<'a>,
    pub q_indptr: TensorView<'a>,
    pub kv_lens: TensorView<'a>,
    pub max_q_len: u32,
    pub max_kv_len: u32,
    pub max_blocks_per_seq: u32,
    /// Usually `1 / sqrt(head_dim)`.
    pub scale: f32,
}

/// Copies block `src` to block `dst` in every layer, for each `(src, dst)` of `pairs` in order.
/// Layer `l`'s block `b` is the `block_bytes` bytes at `l · layer_stride_bytes + b · block_bytes`
/// of `pool`.
pub struct KvCopyContext<'a> {
    pub pool: DeviceSlice<'a>,
    pub layer_stride_bytes: u64,
    pub block_bytes: u64,
    pub num_layers: u32,
    pub pairs: &'a [(BlockId, BlockId)],
}

/// MoE routing of `num_tokens` tokens.
///
/// - `router_logits`: `[num_tokens, num_experts]` F32
/// - `topk_ids` (I32) / `topk_weights` (F32): `[num_tokens, top_k]`, per token in descending
///   weight, ties to the lower expert id
/// - `sorted_rows`: `[num_tokens · top_k]` I32, every row `token · top_k + slot` grouped by
///   expert (ascending expert, ascending row within an expert)
/// - `expert_offsets`: `[num_experts + 1]` I32; expert `e` owns
///   `sorted_rows[expert_offsets[e]..expert_offsets[e + 1]]`
pub struct MoeRouteContext<'a> {
    pub cfg: MoeRouteConfig,
    pub router_logits: TensorView<'a>,
    pub topk_ids: TensorView<'a>,
    pub topk_weights: TensorView<'a>,
    pub sorted_rows: TensorView<'a>,
    pub expert_offsets: TensorView<'a>,
}

/// `out[t] += Σ w[t, slot] · down(silu(gate(x[t])) · up(x[t]))` over the slots routed to a local
/// expert, accumulated expert by expert in ascending id (and in `sorted_rows` order within an
/// expert), so the result does not depend on how a provider batches the work.
///
/// - `x`/`out`: `[num_tokens, hidden]`
/// - `w_gate`/`w_up`: `[num_local_experts, inter, hidden]`; `w_down`:
///   `[num_local_experts, hidden, inter]`
/// - `sorted_rows`, `expert_offsets`, `topk_weights`: the outputs of `moe_route`
/// - `host_expert_offsets`: a host copy of `expert_offsets` (group sizes without a device read)
/// - `workspace`: provider scratch (gathered rows, intermediates); `None` when the provider needs
///   none
pub struct MoeExpertsContext<'a> {
    pub cfg: MoeExpertsConfig,
    pub x: TensorView<'a>,
    pub w_gate: TensorView<'a>,
    pub w_up: TensorView<'a>,
    pub w_down: TensorView<'a>,
    pub sorted_rows: TensorView<'a>,
    pub expert_offsets: TensorView<'a>,
    pub topk_weights: TensorView<'a>,
    pub host_expert_offsets: &'a [i32],
    pub out: TensorView<'a>,
    pub workspace: Option<DeviceSlice<'a>>,
}

/// `q`/`out`: `[q_len, num_q_heads, head_dim]`; `k_cache`/`v_cache`:
/// `[kv_capacity, num_kv_heads, head_dim]` with rows `[0, q_start + q_len)` valid. Query `i` sits
/// at absolute position `q_start + i`; decode has `q_len = 1`.
pub struct AttentionContext<'a> {
    pub cfg: AttentionConfig,
    pub q: TensorView<'a>,
    pub k_cache: TensorView<'a>,
    pub v_cache: TensorView<'a>,
    pub out: TensorView<'a>,
    pub q_start: u32,
    /// Usually `1 / sqrt(head_dim)`.
    pub scale: f32,
}

/// `x`/`out`: `[rows, dim]`; `weight`: `[dim]`.
pub struct NormContext<'a> {
    pub x: TensorView<'a>,
    pub weight: TensorView<'a>,
    pub out: TensorView<'a>,
    pub eps: f32,
}

/// In place on `q` `[tokens, num_q_heads, head_dim]` and `k` `[tokens, num_kv_heads, head_dim]`;
/// `positions`: `[tokens]` I32; `inv_freq`: `[rotary_dim / 2]` F32.
pub struct RopeContext<'a> {
    pub cfg: RopeConfig,
    pub q: TensorView<'a>,
    pub k: TensorView<'a>,
    pub positions: TensorView<'a>,
    pub inv_freq: TensorView<'a>,
}

/// `out = silu(gate) · up`, all `[rows, cols]`.
pub struct ActivationContext<'a> {
    pub gate: TensorView<'a>,
    pub up: TensorView<'a>,
    pub out: TensorView<'a>,
}

/// `out[t] = table[ids[t] − vocab_offset]`, zero when that row is outside `[0, vocab_rows)`;
/// `ids`: `[tokens]` I32; `table`: `[vocab_rows, hidden]`; `out`: `[tokens, hidden]`.
pub struct EmbeddingContext<'a> {
    pub ids: TensorView<'a>,
    pub table: TensorView<'a>,
    pub out: TensorView<'a>,
    pub vocab_offset: i64,
}

/// `out = a + b` over the same number of contiguous elements.
pub struct ElementwiseContext<'a> {
    pub a: TensorView<'a>,
    pub b: TensorView<'a>,
    pub out: TensorView<'a>,
}

/// `residual = round(residual + x)` in place (rounded to the residual's dtype), then
/// `out = rmsnorm(residual) · weight` — numerically `add` followed by `rmsnorm` on the rounded sum.
/// `residual`/`x`/`out`: `[rows, dim]` (rows may be strided); `weight`: `[dim]`.
pub struct AddRmsnormContext<'a> {
    pub residual: TensorView<'a>,
    pub x: TensorView<'a>,
    pub weight: TensorView<'a>,
    pub out: TensorView<'a>,
    pub eps: f32,
}

/// Reduces the first `rows` rows of `logits` (every view may hold more rows, e.g. buffers sized
/// for the largest batch). Per row `r`:
///
/// - `lse[r]`: log-sum-exp of the raw logits, NaN ignored;
/// - `top_ids[r]`/`top_values[r]`: the `top_n` largest raw logits and their ids, descending, ties
///   to the lower id, NaN last;
/// - `mode[r]` = 1: `sampled[r]` is the smallest id whose cumulative sum of
///   `exp(logit / temperature[r] − max)` in id order exceeds `uniform[r] · total` (the argmax when
///   `temperature[r]` ≤ 0 or no logit is finite) and `sampled_logit[r]` its raw logit; `mode[r]` =
///   0: `sampled[r]` = −1 and `sampled_logit[r]` = NaN.
///
/// Views: `logits` `[≥ rows, vocab]` F32 (row-strided); `temperature`, `uniform`, `lse`,
/// `sampled_logit` `[≥ rows]` F32; `mode`, `sampled` `[≥ rows]` I32; `top_ids` I32 and
/// `top_values` F32 `[≥ rows, top_n]`. All but `logits` are dense.
pub struct LogitsReduceContext<'a> {
    pub logits: TensorView<'a>,
    pub temperature: TensorView<'a>,
    pub uniform: TensorView<'a>,
    pub mode: TensorView<'a>,
    pub top_ids: TensorView<'a>,
    pub top_values: TensorView<'a>,
    pub lse: TensorView<'a>,
    pub sampled: TensorView<'a>,
    pub sampled_logit: TensorView<'a>,
    pub rows: u32,
}

// ----------------------------------------------------------------------------------- traits
// `supports` and `implementation` depend on the config only and are called at startup by the
// registry; `execute` enqueues the op on the provider's compute stream.

pub trait GemmKernel: Send + Sync {
    fn supports(&self, cfg: &GemmConfig) -> bool;
    fn implementation(&self, cfg: &GemmConfig) -> String;
    fn execute(&self, ctx: &mut GemmContext<'_>) -> Result<(), KernelError>;
}

/// Prefill and decode attention (`cfg.kind` selects the entry point): `execute` runs the
/// contiguous kinds, `execute_paged` the paged kinds.
pub trait AttentionKernel: Send + Sync {
    fn supports(&self, cfg: &AttentionConfig) -> bool;
    fn implementation(&self, cfg: &AttentionConfig) -> String;
    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError>;
    fn execute_paged(&self, ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError>;
}

/// RMSNorm.
pub trait NormKernel: Send + Sync {
    fn supports(&self, cfg: &NormConfig) -> bool;
    fn implementation(&self, cfg: &NormConfig) -> String;
    fn execute(&self, ctx: &mut NormContext<'_>) -> Result<(), KernelError>;
}

/// Rotary embedding.
pub trait RopeKernel: Send + Sync {
    fn supports(&self, cfg: &RopeConfig) -> bool;
    fn implementation(&self, cfg: &RopeConfig) -> String;
    fn execute(&self, ctx: &mut RopeContext<'_>) -> Result<(), KernelError>;
}

/// SiLU-and-multiply.
pub trait ActivationKernel: Send + Sync {
    fn supports(&self, cfg: &ActivationConfig) -> bool;
    fn implementation(&self, cfg: &ActivationConfig) -> String;
    fn execute(&self, ctx: &mut ActivationContext<'_>) -> Result<(), KernelError>;
}

/// Embedding gather.
pub trait EmbeddingKernel: Send + Sync {
    fn supports(&self, cfg: &EmbeddingConfig) -> bool;
    fn implementation(&self, cfg: &EmbeddingConfig) -> String;
    fn execute(&self, ctx: &mut EmbeddingContext<'_>) -> Result<(), KernelError>;
}

/// Residual add.
pub trait ElementwiseKernel: Send + Sync {
    fn supports(&self, cfg: &ElementwiseConfig) -> bool;
    fn implementation(&self, cfg: &ElementwiseConfig) -> String;
    fn execute(&self, ctx: &mut ElementwiseContext<'_>) -> Result<(), KernelError>;
}

/// KV block fork (`copy_blocks`).
pub trait KvCopyKernel: Send + Sync {
    fn supports(&self, cfg: &KvCopyConfig) -> bool;
    fn implementation(&self, cfg: &KvCopyConfig) -> String;
    fn execute(&self, ctx: &mut KvCopyContext<'_>) -> Result<(), KernelError>;
}

/// Mixture-of-experts routing (`moe_route`) and expert compute (`moe_experts`).
pub trait MoeKernel: Send + Sync {
    fn supports_route(&self, cfg: &MoeRouteConfig) -> bool;
    fn supports_experts(&self, cfg: &MoeExpertsConfig) -> bool;
    fn implementation_route(&self, cfg: &MoeRouteConfig) -> String;
    fn implementation_experts(&self, cfg: &MoeExpertsConfig) -> String;
    fn route(&self, ctx: &mut MoeRouteContext<'_>) -> Result<(), KernelError>;
    fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError>;
}

/// Residual add fused with RMSNorm (ABI v2.1).
pub trait AddRmsnormKernel: Send + Sync {
    fn supports(&self, cfg: &AddRmsnormConfig) -> bool;
    fn implementation(&self, cfg: &AddRmsnormConfig) -> String;
    fn execute(&self, ctx: &mut AddRmsnormContext<'_>) -> Result<(), KernelError>;
}

/// Device-side logits reduction (ABI v2.1).
pub trait LogitsReduceKernel: Send + Sync {
    fn supports(&self, cfg: &LogitsReduceConfig) -> bool;
    fn implementation(&self, cfg: &LogitsReduceConfig) -> String;
    fn execute(&self, ctx: &mut LogitsReduceContext<'_>) -> Result<(), KernelError>;
}

/// One implementation source (`cpu-reference`, a loaded shim library). A family the provider
/// does not implement at all returns `None`; per-config support is `supports`. The ABI v2.1
/// families default to `None`, so a provider (or a shim library) without them is a fallback, not
/// an error.
pub trait KernelProvider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn gemm(&self) -> Option<&dyn GemmKernel>;
    fn attention(&self) -> Option<&dyn AttentionKernel>;
    fn norm(&self) -> Option<&dyn NormKernel>;
    fn rope(&self) -> Option<&dyn RopeKernel>;
    fn activation(&self) -> Option<&dyn ActivationKernel>;
    fn embedding(&self) -> Option<&dyn EmbeddingKernel>;
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel>;
    fn kv_copy(&self) -> Option<&dyn KvCopyKernel>;
    fn moe(&self) -> Option<&dyn MoeKernel>;
    fn add_rmsnorm(&self) -> Option<&dyn AddRmsnormKernel> {
        None
    }
    fn logits_reduce(&self) -> Option<&dyn LogitsReduceKernel> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_names_are_the_abi_suffixes() {
        let names: Vec<&str> = OpKind::ALL.iter().map(OpKind::as_str).collect();
        assert_eq!(
            names,
            [
                "gemm",
                "attention_prefill",
                "attention_decode",
                "rmsnorm",
                "rope",
                "silu_mul",
                "embedding",
                "add",
                "attention_prefill_paged",
                "attention_decode_paged",
                "copy_blocks",
                "moe_route",
                "moe_experts",
                "add_rmsnorm",
                "logits_reduce"
            ]
        );
        assert_eq!(OpKind::SiluMul.to_string(), "silu_mul");
        assert_eq!(ProviderId("cpu-reference").to_string(), "cpu-reference");
    }

    #[test]
    fn configs_render_the_failure_message_form() {
        let mut attn = AttentionConfig {
            kind: AttentionKind::Prefill,
            num_q_heads: 24,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: None,
            causal: true,
        };
        assert_eq!(
            attn.to_string(),
            "head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1"
        );
        assert_eq!(attn.op(), OpKind::AttentionPrefill);
        attn.kind = AttentionKind::Decode;
        attn.block_tokens = Some(16);
        assert_eq!(attn.op(), OpKind::AttentionDecode);
        assert!(attn.to_string().ends_with(" block_tokens=16"));
        attn.kind = AttentionKind::PrefillPaged;
        assert_eq!(attn.op(), OpKind::AttentionPrefillPaged);
        attn.kind = AttentionKind::DecodePaged;
        assert_eq!(attn.op(), OpKind::AttentionDecodePaged);
        assert!(attn.kind.is_paged() && !AttentionKind::Decode.is_paged());
        assert_eq!(
            KvCopyConfig {
                num_layers: 16,
                block_bytes: 65536
            }
            .to_string(),
            "num_layers=16 block_bytes=65536"
        );
        assert_eq!(
            MoeRouteConfig {
                num_experts: 64,
                top_k: 8,
                renormalize: false
            }
            .to_string(),
            "experts=64 top_k=8 renormalize=0"
        );
        let experts = MoeExpertsConfig {
            hidden: 2048,
            inter: 1024,
            num_experts: 64,
            top_k: 8,
            expert_begin: 0,
            expert_end: 64,
            dtype: DType::BF16,
        };
        assert_eq!(
            experts.to_string(),
            "hidden=2048 inter=1024 experts=64 top_k=8 local=0..64 dtype=bf16"
        );
        assert_eq!(experts.num_local_experts(), 64);

        let gemm = GemmConfig {
            n: 3072,
            k: 8192,
            trans_b: true,
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype: DType::F32,
        };
        assert_eq!(
            gemm.to_string(),
            "n=3072 k=8192 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=f32"
        );
        assert_eq!(
            NormConfig {
                dim: 3072,
                dtype: DType::BF16
            }
            .to_string(),
            "dim=3072 dtype=bf16"
        );
        assert_eq!(
            RopeConfig {
                num_q_heads: 24,
                num_kv_heads: 8,
                head_dim: 128,
                rotary_dim: 128,
                dtype: DType::BF16
            }
            .to_string(),
            "head_dim=128 rotary_dim=128 q_heads=24 kv_heads=8 dtype=bf16"
        );
        assert_eq!(
            ActivationConfig {
                cols: 8192,
                dtype: DType::BF16
            }
            .to_string(),
            "cols=8192 dtype=bf16"
        );
        assert_eq!(
            EmbeddingConfig {
                hidden: 3072,
                vocab_rows: 128256,
                dtype: DType::BF16
            }
            .to_string(),
            "hidden=3072 vocab_rows=128256 dtype=bf16"
        );
        assert_eq!(
            ElementwiseConfig { dtype: DType::F32 }.to_string(),
            "dtype=f32"
        );
        assert_eq!(
            AddRmsnormConfig {
                dtype: DType::BF16,
                dim: 3072
            }
            .to_string(),
            "dim=3072 dtype=bf16"
        );
        assert_eq!(
            LogitsReduceConfig {
                vocab: 128_256,
                top_n: 20
            }
            .to_string(),
            "vocab=128256 top_n=20"
        );
    }
}
