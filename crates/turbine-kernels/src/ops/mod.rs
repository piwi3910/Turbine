//! Op families (TS §6, P1 S-6). Each family has a config (shapes and dtypes only, what
//! `supports` and `implementation` look at), a context (the tensor views `execute` reads and
//! writes) and a capability trait every provider implements. Nothing here names a vendor type,
//! so every backend provider implements the same traits.
//!
//! The compute stream is implicit: every provider owns its context and enqueues on that
//! context's compute stream (contract §9.2).
use std::fmt;

use turbine_core::types::DType;
use turbine_tensor::TensorView;

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
}

impl OpKind {
    /// Every op of the current ABI version, in header order.
    pub const ALL: &'static [OpKind] = &[
        OpKind::Gemm,
        OpKind::AttentionPrefill,
        OpKind::AttentionDecode,
        OpKind::Rmsnorm,
        OpKind::Rope,
        OpKind::SiluMul,
        OpKind::Embedding,
        OpKind::Add,
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
    Prefill,
    Decode,
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

// ----------------------------------------------------------------------------------- traits
// `supports` and `implementation` depend on the config only and are called at startup by the
// registry; `execute` enqueues the op on the provider's compute stream.

pub trait GemmKernel: Send + Sync {
    fn supports(&self, cfg: &GemmConfig) -> bool;
    fn implementation(&self, cfg: &GemmConfig) -> String;
    fn execute(&self, ctx: &mut GemmContext<'_>) -> Result<(), KernelError>;
}

/// Prefill and decode attention (`cfg.kind` selects the entry point).
pub trait AttentionKernel: Send + Sync {
    fn supports(&self, cfg: &AttentionConfig) -> bool;
    fn implementation(&self, cfg: &AttentionConfig) -> String;
    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError>;
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

/// One implementation source (`cpu-reference`, a loaded shim library). A family the provider
/// does not implement at all returns `None`; per-config support is `supports`.
pub trait KernelProvider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn gemm(&self) -> Option<&dyn GemmKernel>;
    fn attention(&self) -> Option<&dyn AttentionKernel>;
    fn norm(&self) -> Option<&dyn NormKernel>;
    fn rope(&self) -> Option<&dyn RopeKernel>;
    fn activation(&self) -> Option<&dyn ActivationKernel>;
    fn embedding(&self) -> Option<&dyn EmbeddingKernel>;
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel>;
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
                "add"
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
    }
}
