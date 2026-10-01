//! Op families (TS §6, P1 S-6). Each family has a config (shapes and dtypes only, what
//! `supports` and `implementation` look at), a context (the tensor views `execute` reads and
//! writes) and a capability trait every provider implements. Nothing here names a vendor type,
//! so every backend provider implements the same traits.
//!
//! The compute stream is implicit: every provider owns its context and enqueues on that
//! context's compute stream (contract §9.2).
use std::fmt;
use std::sync::Arc;

use turbine_core::types::{BlockId, DType};
use turbine_tensor::{DevicePtr, DeviceSlice, TensorView};

use crate::KernelError;
use crate::cards::CardProfile;
use crate::quant::{ActQuantDesc, QuantSchemeDesc};
use crate::registry::OpConfig;

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
    /// ABI v2.6 (optional in a shim library): per-row FP32 sum of squares of a tensor-parallel
    /// rank's slice.
    RowSumsq,
    /// ABI v2.6 (optional in a shim library): RMSNorm of a slice with the all-reduced sum of
    /// squares of the full row.
    RmsnormSharded,
    /// ABI v2.9 (optional in a shim library): GEMM against a quantized weight (Phase 6a).
    QGemm,
    /// ABI v2.9 (optional in a shim library): activation quantization before a quantized GEMM.
    QuantizeAct,
    /// ABI v2.11 (optional in a shim library): encode KV blocks into a lower tier's codec and
    /// decode them back (Phase 6b).
    KvTranscode,
}

impl OpKind {
    /// Every op of the current ABI version (the optional minor groups included), in header order:
    /// the position is the op's `TURBINE_OP_*` code, so new ops are appended.
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
        OpKind::RowSumsq,
        OpKind::RmsnormSharded,
        OpKind::QGemm,
        OpKind::QuantizeAct,
        OpKind::KvTranscode,
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
            OpKind::RowSumsq => "row_sumsq",
            OpKind::RmsnormSharded => "rmsnorm_sharded",
            OpKind::QGemm => "qgemm",
            OpKind::QuantizeAct => "quantize_act",
            OpKind::KvTranscode => "kv_transcode",
        }
    }

    /// The `TURBINE_OP_*` code of kernel ABI v2.4: the op's position in [`OpKind::ALL`].
    pub fn abi_code(self) -> i32 {
        match self {
            OpKind::Gemm => 0,
            OpKind::AttentionPrefill => 1,
            OpKind::AttentionDecode => 2,
            OpKind::Rmsnorm => 3,
            OpKind::Rope => 4,
            OpKind::SiluMul => 5,
            OpKind::Embedding => 6,
            OpKind::Add => 7,
            OpKind::AttentionPrefillPaged => 8,
            OpKind::AttentionDecodePaged => 9,
            OpKind::CopyBlocks => 10,
            OpKind::MoeRoute => 11,
            OpKind::MoeExperts => 12,
            OpKind::AddRmsnorm => 13,
            OpKind::LogitsReduce => 14,
            OpKind::RowSumsq => 15,
            OpKind::RmsnormSharded => 16,
            OpKind::QGemm => 17,
            OpKind::QuantizeAct => 18,
            OpKind::KvTranscode => 19,
        }
    }

    /// The kernel ABI minor revision that added the op (0: part of ABI v2 proper). A library of
    /// an earlier minor has no such op and does not know its `TURBINE_OP_*` code.
    pub fn abi_minor(self) -> u32 {
        match self {
            OpKind::AddRmsnorm | OpKind::LogitsReduce => 1,
            OpKind::RowSumsq | OpKind::RmsnormSharded => 6,
            OpKind::QGemm | OpKind::QuantizeAct => 9,
            OpKind::KvTranscode => 11,
            _ => 0,
        }
    }
}

/// One implementation of an op a provider enumerates (kernel ABI v2.4 `turbine_impl_info`):
/// `index` is its position in the provider's order, `name` the name `implementation()` reports
/// when it runs, `provider` its family (`hipblaslt`, `ck`, `turbine_hip`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplInfo {
    pub index: u32,
    pub name: String,
    pub provider: String,
    /// `moe_experts`: the implementation reads `host_expert_offsets`.
    pub needs_host_offsets: bool,
}

/// The implementation(s) the kernel registry chose for one op config: one index, or one per
/// routed-row tier (`moe_experts`), each an index into the provider's
/// [`KernelProvider::implementations`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImplChoice {
    Single(u32),
    /// Tiers in ascending `max_rows`, the open tier (`max_rows: None`) last.
    ByRows(Vec<RowTier>),
}

/// One routed-row tier of an [`ImplChoice::ByRows`]: up to `max_rows` rows (`None` = no upper
/// bound) run implementation `index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowTier {
    pub max_rows: Option<u32>,
    pub index: u32,
}

impl ImplChoice {
    /// The implementation index a call of `rows` routed rows runs: the first tier whose bound
    /// holds `rows` (the last tier when none does). One comparison per tier (at most two).
    #[inline]
    pub fn index_for(&self, rows: usize) -> u32 {
        match self {
            ImplChoice::Single(index) => *index,
            ImplChoice::ByRows(tiers) => tiers
                .iter()
                .find(|t| t.max_rows.is_none_or(|max| rows <= max as usize))
                .or(tiers.last())
                .map_or(0, |t| t.index),
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
    /// Element type of Q, the new K/V rows, the output and the KV. A paged kind may name
    /// [`DType::F8E4M3`] (Phase 6a S-13, `kv.dtype: fp8_e4m3`): the pool pages are then OCP
    /// e4m3fn bytes with one K and one V scale per layer ([`PagedAttentionContext::k_scale`]),
    /// while Q, the new rows and the output stay BF16 (the C ABI's `dtype` names the pages the
    /// same way).
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

/// MoE router: softmax over `num_experts` F32 logits, top-`top_k` selected as PyTorch's CPU
/// `torch.topk` selects ([`crate::torch_topk`]), optional renormalisation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MoeRouteConfig {
    pub num_experts: u32,
    pub top_k: u32,
    /// Divide the selected weights by their sum (`norm_topk_prob`; false for OLMoE-1B-7B).
    pub renormalize: bool,
    /// Round each logit to BF16 (round to nearest even) before the softmax: the router GEMM
    /// output of a BF16 model in transformers (`self.gate(hidden_states)` is a BF16 linear).
    /// Kernel ABI v2.2 (`TURBINE_MOE_ROUTE_BF16_LOGITS`).
    pub bf16_logits: bool,
}

impl fmt::Display for MoeRouteConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "experts={} top_k={} renormalize={} bf16_logits={}",
            self.num_experts,
            self.top_k,
            u8::from(self.renormalize),
            u8::from(self.bf16_logits)
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

    /// Routed rows (`tokens · top_k`) of a call over `tokens` tokens: the size a provider's
    /// [`MoeKernel::needs_host_offsets`] answer depends on. Known on the host before routing.
    pub fn routed_rows(&self, tokens: usize) -> usize {
        tokens.saturating_mul(self.top_k as usize)
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

/// Per-row FP32 sum of squares over rows of `dim` elements (ABI v2.6 `row_sumsq`): the partial
/// sum a tensor-parallel rank contributes for its slice of a row normalised across ranks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RowSumsqConfig {
    pub dim: u32,
    pub dtype: DType,
}

impl fmt::Display for RowSumsqConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dim={} dtype={}", self.dim, self.dtype.as_str())
    }
}

/// RMSNorm of a `dim`-wide slice of rows `full_dim` wide, from the all-reduced FP32 sum of
/// squares of the full rows (ABI v2.6 `rmsnorm_sharded`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RmsnormShardedConfig {
    pub dim: u32,
    pub full_dim: u32,
    pub dtype: DType,
}

impl fmt::Display for RmsnormShardedConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dim={} full_dim={} dtype={}",
            self.dim,
            self.full_dim,
            self.dtype.as_str()
        )
    }
}

/// GEMM against a quantized weight (ABI v2.9 `qgemm`, Phase 6a):
/// `c[m, n] = alpha · a[m, k] · dequant(b)[n, k]ᵀ`, F32 accumulation. `a_dtype` is BF16 for
/// weight-only schemes (and MXFP4-emulated activations) or F8E4M3 for the FP8 activation
/// modes; `c_dtype` BF16 or F32.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QGemmConfig {
    pub n: u32,
    pub k: u32,
    pub scheme: QuantSchemeDesc,
    pub act_quant: ActQuantDesc,
    pub a_dtype: DType,
    pub c_dtype: DType,
}

impl fmt::Display for QGemmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={} k={} scheme={} act={} a={} c={}",
            self.n,
            self.k,
            self.scheme.as_str(),
            self.act_quant.as_str(),
            self.a_dtype.as_str(),
            self.c_dtype.as_str()
        )
    }
}

/// Activation quantization before a quantized GEMM (ABI v2.9 `quantize_act`): `cols`-wide rows
/// of `x_dtype` into `out_dtype` (F8E4M3 for the FP8 modes, BF16 for MXFP4 emulation).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuantizeActConfig {
    pub cols: u32,
    pub mode: ActQuantDesc,
    pub x_dtype: DType,
    pub out_dtype: DType,
}

impl fmt::Display for QuantizeActConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cols={} mode={} x={} out={}",
            self.cols,
            self.mode.as_str(),
            self.x_dtype.as_str(),
            self.out_dtype.as_str()
        )
    }
}

/// The format of one side of a KV transcode (ABI v2.11 `TURBINE_KVFMT_*`, registry names of
/// `turbine_kv::codec`): `L0` is the page format itself, the others a lower tier's codec.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KvTranscodeFormat {
    L0,
    Fp8E4m3,
    Tq4,
    Tq2,
}

impl KvTranscodeFormat {
    /// The `TURBINE_KVFMT_*` code.
    pub fn abi_code(self) -> i32 {
        match self {
            KvTranscodeFormat::L0 => 0,
            KvTranscodeFormat::Fp8E4m3 => 1,
            KvTranscodeFormat::Tq4 => 2,
            KvTranscodeFormat::Tq2 => 3,
        }
    }

    /// The codec's registry name (`turbine_kv::codec`).
    pub fn as_str(self) -> &'static str {
        match self {
            KvTranscodeFormat::L0 => "l0",
            KvTranscodeFormat::Fp8E4m3 => "fp8_e4m3",
            KvTranscodeFormat::Tq4 => "tq4",
            KvTranscodeFormat::Tq2 => "tq2",
        }
    }
}

/// Which way a transcode runs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KvTranscodeDirection {
    /// L0 pages into coded slots (a demotion).
    Encode,
    /// Coded slots into L0 pages (a promotion).
    Decode,
}

/// A KV transcode (ABI v2.11 `kv_transcode`, Phase 6b): `layers` pages of
/// `[2, block_tokens, num_kv_heads, head_dim]` elements of `page_dtype` per block, between the
/// page format (`L0`) and a codec. Exactly one of `src_format` and `dst_format` is `L0`: the
/// source is the pages when encoding, the coded slots when decoding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KvTranscodeConfig {
    pub src_format: KvTranscodeFormat,
    pub dst_format: KvTranscodeFormat,
    pub page_dtype: DType,
    pub head_dim: u32,
    pub num_kv_heads: u32,
    pub block_tokens: u32,
    pub layers: u32,
}

impl KvTranscodeConfig {
    /// The direction, or `None` unless exactly one side is `L0`.
    pub fn direction(&self) -> Option<KvTranscodeDirection> {
        match (self.src_format, self.dst_format) {
            (KvTranscodeFormat::L0, KvTranscodeFormat::L0) => None,
            (KvTranscodeFormat::L0, _) => Some(KvTranscodeDirection::Encode),
            (_, KvTranscodeFormat::L0) => Some(KvTranscodeDirection::Decode),
            _ => None,
        }
    }

    /// The codec side (the one that is not `L0`), `None` unless [`Self::direction`] is.
    pub fn codec(&self) -> Option<KvTranscodeFormat> {
        self.direction().map(|d| match d {
            KvTranscodeDirection::Encode => self.dst_format,
            KvTranscodeDirection::Decode => self.src_format,
        })
    }

    /// Elements of one layer's K (or V) of a block.
    pub fn half_layer_elems(&self) -> usize {
        self.block_tokens as usize * self.num_kv_heads as usize * self.head_dim as usize
    }

    /// Bytes of one layer's page of a block (K and V).
    pub fn page_bytes(&self) -> usize {
        2 * self.half_layer_elems() * self.page_dtype.size_bytes()
    }
}

impl fmt::Display for KvTranscodeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}->{} page={} layers={} block_tokens={} kv_heads={} head_dim={}",
            self.src_format.as_str(),
            self.dst_format.as_str(),
            self.page_dtype.as_str(),
            self.layers,
            self.block_tokens,
            self.num_kv_heads,
            self.head_dim
        )
    }
}

// --------------------------------------------------------------------------------- contexts

/// `a`: `[m, k]` (`a_dtype`); `a_scales`: the activation scales of the FP8 modes (F32, dense:
/// `[1]`, `[m]` or `[m, k / 128]`), else `None`; `b`: the packed weight bytes `[n, row bytes]`
/// (U8 or F8E4M3) in the scheme's layout; `b_scales`: F32 scales, or E8M0 bytes (U8) for MXFP4;
/// `b_zeros`: U8 zero points (INT4 with zero points only); `c`: `[m, n]`.
pub struct QGemmContext<'a> {
    pub cfg: QGemmConfig,
    pub a: TensorView<'a>,
    pub a_scales: Option<TensorView<'a>>,
    pub b: TensorView<'a>,
    pub b_scales: TensorView<'a>,
    pub b_zeros: Option<TensorView<'a>>,
    pub c: TensorView<'a>,
    pub alpha: f32,
    /// As [`GemmContext::prefill`].
    pub prefill: bool,
}

/// `x`: `[rows, cols]`; `out`: `[rows, cols]` (F8E4M3 or BF16); `scales`: F32, dense, as many
/// as the mode has for `rows × cols` (`ActQuantDesc::scale_count`); `static_scale`: the
/// checkpoint's `input_scale` for [`ActQuantDesc::Fp8Tensor`].
pub struct QuantizeActContext<'a> {
    pub cfg: QuantizeActConfig,
    pub x: TensorView<'a>,
    pub out: TensorView<'a>,
    pub scales: TensorView<'a>,
    pub static_scale: f32,
}

/// The host codecs a transcode runs on the CPU (the `cpu-reference` provider; a library
/// ignores it). `turbine-kernels` does not depend on `turbine-kv`: the server hands in a table
/// over `turbine_kv::codec`, and the kernel tests one over the same codecs, so the reference a
/// GPU transcode is judged against is the codec itself.
pub trait KvCodecFns: Send + Sync {
    /// Encodes one block (`layers` pages, `layers × page_bytes` bytes, layer order) into one
    /// coded slot; `k_scales` / `v_scales` are the per-layer scales of FP8 pages (empty = 1.0).
    fn encode(
        &self,
        cfg: &KvTranscodeConfig,
        seed: u64,
        scales: (&[f32], &[f32]),
        block: &[u8],
        slot: &mut [u8],
    ) -> Result<(), String>;

    /// Decodes one coded slot into one block (`layers × page_bytes` bytes).
    fn decode(
        &self,
        cfg: &KvTranscodeConfig,
        seed: u64,
        scales: (&[f32], &[f32]),
        slot: &[u8],
        block: &mut [u8],
    ) -> Result<(), String>;
}

/// `pages`: `num_blocks × layers` device addresses, block-major (`pages[b * layers + l]` is
/// layer `l` of block `b`), each the start of `cfg.page_bytes()` bytes of the same device as
/// `coded` (read when encoding, written when decoding; like the addresses of
/// `CopyEngine::copy_async`, the caller guarantees they address live pool pages); `coded`:
/// `num_blocks` consecutive slots of `coded_block_bytes` (written when encoding, read when
/// decoding); `k_scales` / `v_scales`: F32 `[layers]` scales of FP8 pages (`None` = 1.0; unused
/// with BF16 pages); `seed`: the TurboQuant rotation seed; `codecs`: read by the CPU provider
/// only.
pub struct KvTranscodeContext<'a> {
    pub cfg: KvTranscodeConfig,
    pub pages: &'a [DevicePtr],
    pub coded: DeviceSlice<'a>,
    pub coded_block_bytes: usize,
    pub seed: u64,
    pub k_scales: Option<TensorView<'a>>,
    pub v_scales: Option<TensorView<'a>>,
    pub codecs: &'a dyn KvCodecFns,
}

impl KvTranscodeContext<'_> {
    /// Blocks of the batch (`coded` slots).
    pub fn num_blocks(&self) -> usize {
        self.coded
            .len()
            .checked_div(self.coded_block_bytes)
            .unwrap_or(0)
    }
}

/// The TurboQuant tables of a `tq4` / `tq2` transcode in device memory (ABI v2.11
/// `turbine_tq_params`), built on the host from the codec (`turbine_kv::codec::turboquant`) for
/// the context's seed; a library never regenerates them. `codebooks[bits − 1]`: F32 `[2^bits]`,
/// the unit-variance Lloyd–Max centroids; `tables`: F32 `[layers, num_kv_heads, 2·head_dim +
/// head_dim²]`, per (layer, KV head) the K signs, the V signs and the QJL projection `S`
/// (row-major), as [`TqHeadTables`] holds them.
#[derive(Clone, Debug)]
pub struct KvTranscodeTables<'a> {
    pub codebooks: [TensorView<'a>; 4],
    pub tables: TensorView<'a>,
}

impl KvTranscodeTables<'_> {
    /// F32 elements of one (layer, KV head)'s entry of `tables`.
    pub fn head_elems(head_dim: u32) -> usize {
        let d = head_dim as usize;
        2 * d + d * d
    }
}

/// `a`: `[m, k]`; `b`: `[n, k]` when `trans_b`, else `[k, n]`; `c`: `[m, n]`. Row strides are
/// the leading dimensions.
pub struct GemmContext<'a> {
    pub a: TensorView<'a>,
    pub b: TensorView<'a>,
    pub c: TensorView<'a>,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
    /// The call belongs to a step that prefills prompt tokens (every step but a decode-only
    /// one). A provider whose fastest algorithms make a row's result depend on how many rows
    /// share the call must then compute each row as any other prefill step would: prefix
    /// reuse prefills only a prompt's uncached suffix and must reproduce the whole-prompt
    /// prefill bit for bit (Phase 4). Decode steps (`false`) may run batch-dependent algorithms.
    pub prefill: bool,
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
    /// FP8 pages (`cfg.dtype` F8E4M3): a K element is stored as `e4m3(k / k_scale)` and read as
    /// `e4m3 · k_scale` rounded to the activation dtype, likewise V with `v_scale` (this layer's scales, > 0). Ignored (1.0 by
    /// convention) for BF16 / F16 / F32 pages.
    pub k_scale: f32,
    pub v_scale: f32,
    /// P6b S-5 (ABI v2.11): `[num_seqs, max_blocks_per_seq]` U8 in the provider's memory
    /// (device memory on a GPU, so a decode graph can capture the call), laid out like
    /// `block_table`: one format code per entry ([`KV_FMT_BF16`], [`KV_FMT_FP8_E4M3`],
    /// [`KV_FMT_TQ4`], [`KV_FMT_TQ2`]); `None` = every block in `cfg.dtype`. A block's bytes are
    /// the first page bytes of its format in slot `b` of `kv_layer` (slots of `cfg.dtype`'s
    /// page; the ladder's L0 step, S-7, adds class-page addressing). A table is passed only
    /// when it can hold a block of another format: a GPU provider cannot read it on the host,
    /// so `Some` always takes the mixed-format implementations.
    pub block_formats: Option<TensorView<'a>>,
    /// TurboQuant tables of this layer and the codec that encodes appended rows; required
    /// when `cfg.dtype` is TurboQuant or `block_formats` is `Some` (on a GPU provider with
    /// [`TqPaged::device`]).
    pub tq: Option<TqPaged<'a>>,
}

/// Block format code of a BF16 page (the v2.11 `block_formats` byte).
pub const KV_FMT_BF16: u8 = 0;
/// Block format code of an FP8 e4m3 page.
pub const KV_FMT_FP8_E4M3: u8 = 1;
/// Block format code of a TurboQuant `tq4` page (K 3 + 1 bits, V 4 bits).
pub const KV_FMT_TQ4: u8 = 2;
/// Block format code of a TurboQuant `tq2` page (K 1 + 1 bits, V 2 bits).
pub const KV_FMT_TQ2: u8 = 3;

/// The block format code of KV pages of `dtype` (`None`: not a KV page dtype).
pub fn kv_format_code(dtype: DType) -> Option<u8> {
    match dtype {
        DType::BF16 => Some(KV_FMT_BF16),
        DType::F8E4M3 => Some(KV_FMT_FP8_E4M3),
        DType::Tq4 => Some(KV_FMT_TQ4),
        DType::Tq2 => Some(KV_FMT_TQ2),
        _ => None,
    }
}

/// The TurboQuant tables of one KV head of a layer (P6b S-5).
#[derive(Clone, Debug, PartialEq)]
pub struct TqHeadTables {
    /// ±1 signs of the K rotation (head_dim 128).
    pub k_signs: Vec<f32>,
    /// ±1 signs of the V rotation.
    pub v_signs: Vec<f32>,
    /// The QJL projection `S`, `128 × 128` row-major.
    pub qjl: Vec<f32>,
}

/// TurboQuant parameters of one layer (P6b S-5; the host side of the v2.11 `tq_params`).
#[derive(Clone, Debug, PartialEq)]
pub struct TqParams {
    /// Per KV head of the layer.
    pub heads: Vec<TqHeadTables>,
    /// Unit-variance Lloyd–Max codebooks of 1, 2, 3 and 4 bits (index `bits − 1`).
    pub codebooks: [Vec<f32>; 4],
}

/// Encodes one token-head K/V pair (head_dim values each) of TurboQuant format `fmt`
/// ([`KV_FMT_TQ4`] / [`KV_FMT_TQ2`]) into `record` (the format's record bytes) with `head`'s
/// tables. The caller supplies the codec (`turbine-kv`'s TurboQuant `encode_record`), so this
/// crate carries no copy of the encoder.
pub type TqEncodeFn = fn(fmt: u8, k: &[f32], v: &[f32], head: &TqHeadTables, record: &mut [u8]);

/// The TurboQuant side of a paged attention call.
#[derive(Clone)]
pub struct TqPaged<'a> {
    /// Host tables of the layer (the CPU provider reads these).
    pub params: &'a TqParams,
    pub encode: TqEncodeFn,
    /// The rotation seed the tables were built from.
    pub seed: u64,
    /// The same tables in device memory, for a GPU provider (ABI v2.11 `tq_params`):
    /// `codebooks` as for the transcode and `tables` THIS layer's `[num_kv_heads, head_elems]`
    /// slice of the model's tables (the caller offsets it, user decision 2026-10-01 "6b Task
    /// 12: how per-layer TurboQuant tables reach the paged-attention call", A). `None` on the
    /// CPU provider.
    pub device: Option<KvTranscodeTables<'a>>,
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
/// - `topk_ids` (I32) / `topk_weights` (F32): `[num_tokens, top_k]`, the experts
///   `torch.topk` selects (among tied weights not "the lower id first": see
///   [`crate::torch_topk`]), per token in descending weight, ties to the lower expert id
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
/// - `host_expert_offsets`: a host copy of `expert_offsets` (group sizes without a device read);
///   empty when the provider's [`MoeKernel::needs_host_offsets`] is false for this call, so the
///   caller skips the device-to-host read
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
    /// YaRN's attention factor `m` (Phase 6a Task 28a, kernel ABI v2.10): cos and sin are
    /// multiplied by it in F32 before they are rounded to the dtype, so the rotated q and k
    /// carry it as transformers' do. 1.0 = none; a kernel whose
    /// [`RopeKernel::attn_factor_supported`] is false refuses any other value.
    pub attn_factor: f32,
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

/// `sumsq[r] = Σ_j x[r, j]²`, accumulated in f32 in an order that depends only on the row's
/// length (never on the number of rows). `x`: `[rows, dim]` (rows may be strided); `sumsq`:
/// `[rows]` F32, dense.
pub struct RowSumsqContext<'a> {
    pub x: TensorView<'a>,
    pub sumsq: TensorView<'a>,
}

/// `out[r, j] = round(round(x[r, j] · inv) · weight[j])` with
/// `inv = 1 / sqrt(sumsq[r] / full_dim + eps)` in f32: `rmsnorm` of the full row restricted to
/// this slice when `sumsq[r]` is the sum of squares of the whole row (the ranks' `row_sumsq`
/// all-reduced). `x`/`out`: `[rows, dim]` (rows may be strided); `weight`: `[dim]`, this slice of
/// the norm weight; `sumsq`: `[rows]` F32, dense.
pub struct RmsnormShardedContext<'a> {
    pub x: TensorView<'a>,
    pub sumsq: TensorView<'a>,
    pub weight: TensorView<'a>,
    pub out: TensorView<'a>,
    pub full_dim: u32,
    pub eps: f32,
}

/// Reduces the first `rows` rows of `logits` (every view may hold more rows, e.g. buffers sized
/// for the largest batch). Per row `r`:
///
/// - `lse[r]`: log-sum-exp of the raw logits, NaN ignored;
/// - `top_ids[r]`/`top_values[r]`: the `top_n` largest raw logits and their ids, descending, ties
///   to the lower id, NaN last;
/// - `mode[r]` = 1: `sampled[r]` is one categorical draw at `temperature[r]` with `uniform[r]`
///   (the argmax when `temperature[r]` ≤ 0 or no logit is finite) and `sampled_logit[r]` its raw
///   logit; `mode[r]` = 0: `sampled[r]` = −1 and `sampled_logit[r]` = NaN. With the weights
///   `w = exp(logit / temperature[r] − max)` (NaN weighs 0) the draw is:
///   - `top_p[r]` ≥ 1: the smallest id whose cumulative `w` in id order exceeds
///     `uniform[r] · total`;
///   - `top_p[r]` < 1 (nucleus): over the ids in descending logit order (ties to the lower id),
///     the shortest prefix whose `w` reaches `top_p[r] · total` (at least one id), and in it the
///     first id whose cumulative `w` exceeds `uniform[r]` × the prefix's sum — the host
///     sampler's top-p draw.
///
/// Views: `logits` `[≥ rows, vocab]` F32 (row-strided); `temperature`, `uniform`, `top_p`, `lse`,
/// `sampled_logit` `[≥ rows]` F32; `mode`, `sampled` `[≥ rows]` I32; `top_ids` I32 and
/// `top_values` F32 `[≥ rows, top_n]`. All but `logits` are dense.
pub struct LogitsReduceContext<'a> {
    pub logits: TensorView<'a>,
    pub temperature: TensorView<'a>,
    pub uniform: TensorView<'a>,
    pub top_p: TensorView<'a>,
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
    /// True when `execute` applies a [`RopeContext::attn_factor`] other than 1.0 (a kernel
    /// library at ABI minor ≥ 10, the cpu-reference provider). False by default, so a kernel
    /// that does not read the factor is never handed one: a YaRN model whose factor is not 1 is
    /// refused at startup (`rope_attn_factor_unavailable`).
    fn attn_factor_supported(&self) -> bool {
        false
    }
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

    /// Whether `experts` needs `host_expert_offsets` for a call of `routed_rows` rows
    /// ([`MoeExpertsConfig::routed_rows`]). When false the caller passes an empty slice and
    /// reads nothing back from the device after routing. The answer depends only on host-known
    /// sizes, never on routing results. Default true (the Phase 2 contract).
    fn needs_host_offsets(&self, cfg: &MoeExpertsConfig, routed_rows: usize) -> bool {
        let _ = (cfg, routed_rows);
        true
    }
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

/// The tensor-parallel sharded RMSNorm (ABI v2.6): `row_sumsq` of a rank's slice, then, after
/// the caller all-reduced the sums across ranks, `rmsnorm_sharded` of the slice.
pub trait ShardedNormKernel: Send + Sync {
    fn supports_row_sumsq(&self, cfg: &RowSumsqConfig) -> bool;
    fn supports_rmsnorm_sharded(&self, cfg: &RmsnormShardedConfig) -> bool;
    fn implementation_row_sumsq(&self, cfg: &RowSumsqConfig) -> String;
    fn implementation_rmsnorm_sharded(&self, cfg: &RmsnormShardedConfig) -> String;
    fn row_sumsq(&self, ctx: &mut RowSumsqContext<'_>) -> Result<(), KernelError>;
    fn rmsnorm_sharded(&self, ctx: &mut RmsnormShardedContext<'_>) -> Result<(), KernelError>;
}

/// GEMM against a quantized weight (ABI v2.9).
pub trait QGemmKernel: Send + Sync {
    fn supports(&self, cfg: &QGemmConfig) -> bool;
    fn implementation(&self, cfg: &QGemmConfig) -> String;
    fn execute(&self, ctx: &mut QGemmContext<'_>) -> Result<(), KernelError>;
}

/// Activation quantization (ABI v2.9).
pub trait QuantizeActKernel: Send + Sync {
    fn supports(&self, cfg: &QuantizeActConfig) -> bool;
    fn implementation(&self, cfg: &QuantizeActConfig) -> String;
    fn execute(&self, ctx: &mut QuantizeActContext<'_>) -> Result<(), KernelError>;
}

/// KV transcode (ABI v2.11).
pub trait KvTranscodeKernel: Send + Sync {
    fn supports(&self, cfg: &KvTranscodeConfig) -> bool;
    fn implementation(&self, cfg: &KvTranscodeConfig) -> String;
    /// Runs a transcode that needs no tables (`fp8_e4m3`); a library refuses `tq4` / `tq2`
    /// here, since they need [`Self::execute_with_tables`].
    fn execute(&self, ctx: &mut KvTranscodeContext<'_>) -> Result<(), KernelError>;

    /// Runs a transcode with the TurboQuant tables of `ctx.seed` (`tables` is required by
    /// `tq4` / `tq2` on a library and ignored by `fp8_e4m3`; the cpu-reference provider runs
    /// `ctx.codecs` and ignores it). The default knows no tables: it runs [`Self::execute`]
    /// when `tables` is `None` and refuses otherwise.
    fn execute_with_tables(
        &self,
        ctx: &mut KvTranscodeContext<'_>,
        tables: Option<&KvTranscodeTables<'_>>,
    ) -> Result<(), KernelError> {
        match tables {
            None => self.execute(ctx),
            Some(_) => Err(KernelError::Unsupported {
                message: format!("kv_transcode: {} takes no TurboQuant tables", ctx.cfg),
            }),
        }
    }
}

/// One implementation source (`cpu-reference`, a loaded shim library). A family the provider
/// does not implement at all returns `None`; per-config support is `supports`. The ABI v2.1 and
/// v2.6 families default to `None`, so a provider (or a shim library) without them is a
/// fallback, not an error.
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
    /// ABI v2.6: `None` (the default) for a provider without the sharded RMSNorm ops.
    fn sharded_norm(&self) -> Option<&dyn ShardedNormKernel> {
        None
    }
    /// ABI v2.9: `None` (the default) for a provider without the quantized GEMM.
    fn qgemm(&self) -> Option<&dyn QGemmKernel> {
        None
    }
    /// ABI v2.9: `None` (the default) for a provider without activation quantization.
    fn quantize_act(&self) -> Option<&dyn QuantizeActKernel> {
        None
    }
    /// ABI v2.11: `None` (the default) for a provider without the KV transcode.
    fn kv_transcode(&self) -> Option<&dyn KvTranscodeKernel> {
        None
    }

    /// The implementations of `op` the provider enumerates (kernel ABI v2.4), in its own order.
    /// Empty (the default) when the provider chooses the implementation of every call itself.
    fn implementations(&self, op: OpKind) -> Vec<ImplInfo> {
        let _ = op;
        Vec::new()
    }

    /// Whether implementation `index` of [`KernelProvider::implementations`] supports `spec`;
    /// `rows` is the routed-row count a `moe_experts` call is sized for (`None` for other ops).
    /// False (the default) for a provider that does not enumerate.
    fn implementation_supports(&self, spec: &OpConfig, index: u32, rows: Option<u32>) -> bool {
        let _ = (spec, index, rows);
        false
    }

    /// A provider that runs `choice` for `spec` on every call (the kernel registry binds one per
    /// op config at startup); `None` (the default) runs this provider as it is, i.e. with its
    /// own choice.
    fn bind(&self, spec: &OpConfig, choice: &ImplChoice) -> Option<Arc<dyn KernelProvider>> {
        let _ = (spec, choice);
        None
    }

    /// The card profile the provider's device context runs with (a kernel library context after
    /// `set_profile`), the `card` a caller passes to `KernelRegistry::build`; `None` (the
    /// default) for a provider without one.
    fn card_profile(&self) -> Option<&'static CardProfile> {
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
                "logits_reduce",
                "row_sumsq",
                "rmsnorm_sharded",
                "qgemm",
                "quantize_act",
                "kv_transcode"
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
                renormalize: false,
                bf16_logits: true
            }
            .to_string(),
            "experts=64 top_k=8 renormalize=0 bf16_logits=1"
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
        assert_eq!(
            RowSumsqConfig {
                dim: 1024,
                dtype: DType::BF16
            }
            .to_string(),
            "dim=1024 dtype=bf16"
        );
        assert_eq!(
            RmsnormShardedConfig {
                dim: 1024,
                full_dim: 2048,
                dtype: DType::BF16
            }
            .to_string(),
            "dim=1024 full_dim=2048 dtype=bf16"
        );
    }

    /// The minor that introduced each op: the v2 ops 0, the v2.1 fused ops 1, the v2.6 sharded
    /// RMSNorm ops 6, the v2.9 quantization ops 9, the v2.11 KV transcode 11. Breaks if a new op
    /// is appended without saying which minor group adds it (an older library would be asked for
    /// an op code it does not know).
    #[test]
    fn op_minor_revisions() {
        for &op in OpKind::ALL {
            let want = match op.abi_code() {
                0..=12 => 0,
                13 | 14 => 1,
                15 | 16 => 6,
                17 | 18 => 9,
                _ => 11,
            };
            assert_eq!(op.abi_minor(), want, "{op}");
        }
    }
}
