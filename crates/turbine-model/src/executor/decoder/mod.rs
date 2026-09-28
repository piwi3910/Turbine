//! The shared decoder skeleton (Phase 2m S-3): one executor for every pre-norm decoder family
//! over a ragged batch of sequences on the kernel registry (P1 S-8, P2 S-5/S-9/S-16):
//! embedding → per layer (RMSNorm → Q/K/V projections → the attention hook
//! ([`AttentionHook::after_projections`], e.g. OLMoE's Q/K RMSNorm) → RoPE on Q and the new K
//! rows → paged causal GQA attention, which appends the new K/V rows into each sequence's pool
//! blocks → O projection → residual add → RMSNorm → the FFN hook ([`FfnHook::forward`]: dense
//! SwiGLU or a mixture of experts) → residual add) → final RMSNorm on each sequence's last row
//! → LM head (`embed_tokens` when tied) with FP32 output → the device logits reduction. A family
//! is a [`DecoderSpec`]: its attention and FFN hook, both `&'static` stateless values called
//! once per layer (never per element).
//!
//! Fused projections (Phase 2c, [`ExecutorOptions::fused_projections`]): the loader lays each
//! layer's Q/K/V weights out as one `[q; k; v]` matrix, so one GEMM computes all three
//! projections into one `[tokens, q + 2·kv]` buffer, and the hooks, RoPE and attention read
//! their operands as row-strided column blocks of it ([`Split`]). Unfused (or with a hook whose
//! [`AttentionHook::fuses_qkv`] is false), one GEMM per projection runs over row views of the
//! same weights into dense regions of the same buffer: the Phase 2 op sequence, bit for bit.
//!
//! Residual add + RMSNorm (Phase 2c): with `fused_ops` and a provider for the kernel ABI v2.1
//! `add_rmsnorm`, each residual add that a norm follows (after attention: the FFN norm; after
//! the FFN: the next layer's input norm) is one fused op, so layer `i + 1`'s input norm runs at
//! the end of layer `i`; otherwise `add` then `rmsnorm`, in the same stream order. The fused op
//! rounds the sum to the activation dtype and normalises that, so both paths give the same
//! numbers.
//!
//! Every token of every sequence goes through the GEMMs as one `[total_tokens, hidden]` batch;
//! only attention looks at sequence boundaries (`q_indptr`, `kv_lens`, block tables). The KV
//! lives in the caller's pool (`KvPoolView`); activations live in buffers allocated once for
//! `max_batch_tokens` tokens and `max_seqs` sequences (the FFN hook's own buffers included,
//! [`FfnHook::alloc`]). Every kernel lookup uses a config [`DecoderExecutor::requirements`]
//! lists, so the registry built from it at startup serves every call.
//!
//! Decode graphs (Phase 2c, [`super::graphs`]): with [`ModelExecutor::set_decode_graphs`], a
//! decode-only iteration's ops (embedding through the LM head and the logits reduction) are
//! captured into a graph on the second iteration of their shape and replayed afterwards; the
//! batch metadata and reduction inputs are uploaded into the same device buffers first. Never
//! while tracing or profiling, nor when the FFN hook says the batch cannot be captured
//! ([`FfnHook::graph_capturable`]: MoE with a provider reading the expert offsets on the host).
//!
//! Diagnostics: [`DecoderExecutor::set_trace`] makes each forward record every intermediate
//! tensor ([`TraceTensor`]) with a blocking device read after the op that wrote it; comparing
//! two providers' traces locates the first op where their numerics part. Off by default and
//! free when off (one branch per trace point). [`DecoderExecutor::set_profile`] times every op
//! instead ([`OpProfile`]): each op goes through [`LayerRun::op`] (or the skeleton's own
//! wrapper), which synchronises the stream after it while profiling and costs one branch
//! otherwise.
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use turbine_core::types::{BlockId, DType, KvLayout, ModelShape};
use turbine_distributed::collective::ReduceOp;
use turbine_distributed::tp::{ShardSpec, vocab_shard};
use turbine_kernels::{
    AddRmsnormConfig, AddRmsnormContext, AttentionConfig, AttentionKind, ElementwiseConfig,
    ElementwiseContext, EmbeddingConfig, EmbeddingContext, GemmConfig, GemmContext, KernelError,
    KernelRegistry, NormConfig, NormContext, OpConfig, OpRequirement, PagedAttentionContext,
    RopeConfig, RopeContext,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView, Tensor, TensorView};

use super::batch::{self, BatchLimits, DeviceBatch, HostBatch, Packed};
use super::graphs::{self, DecodeGraphs, GraphCounters, GraphKey, PoolId};
use super::logits::{self, LogitsHead};
use super::profile::{self, OpProfile, Profiler};
use super::{
    BatchInput, ExecutorLimits, ExecutorOptions, ForwardTimings, Logits, ModelExecutor, Split,
    TokenFeed, feed_runs, rope,
};
use crate::ModelError;
use crate::config::{ModelArchConfig, MoeConfig};
use crate::loader::{LM_HEAD, LoadedWeights, qkv_proj_name};
use crate::tp::{TpContext, rank_config};

pub mod hooks;
mod trace;

pub use hooks::{MOE, PLAIN_ATTENTION, QK_NORM_FULL, QK_NORM_PER_HEAD, SWIGLU};
pub use trace::TraceTensor;
use trace::Tracer;

pub(crate) fn invalid(message: String) -> ModelError {
    ModelError::Kernel(KernelError::InvalidArgument { message })
}

/// A decoder family's variant points: the attention hook (what runs between the Q/K/V
/// projections and RoPE) and the FFN hook (the block between the FFN norm and its residual
/// add).
#[derive(Clone, Copy)]
pub struct DecoderSpec {
    pub attention: &'static dyn AttentionHook,
    pub ffn: &'static dyn FfnHook,
}

impl std::fmt::Debug for DecoderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderSpec")
            .field("attention", &self.attention.name())
            .field("ffn", &self.ffn.name())
            .finish()
    }
}

/// Model dimensions in elements, and what the hooks read of the configuration.
#[derive(Clone, Copy, Debug)]
pub struct DecoderDims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `heads · head_dim`.
    pub q_dim: usize,
    /// `kv_heads · head_dim`.
    pub kv_dim: usize,
    /// The dense MLP's `intermediate_size` (a MoE hook reads its experts' width from `moe`).
    pub inter: usize,
    pub vocab: usize,
    pub layers: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Weights, activations and KV dtype (the weight format's; logits are F32).
    pub act: DType,
    /// The LM head is `embed_tokens` (`tie_word_embeddings`).
    pub tied_lm_head: bool,
    /// The mixture of experts, when the configuration has one.
    pub moe: Option<MoeConfig>,
    /// Rows of the embedding table and the LM head this executor holds: `vocab`, or a
    /// tensor-parallel rank's padded vocabulary shard.
    pub vocab_rows: usize,
    /// A tensor-parallel rank's place in its group (the widths above are then the rank's);
    /// `None` on one device.
    pub tp: Option<TpDims>,
}

/// What a tensor-parallel rank's dims add ([`DecoderDims::for_shard`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpDims {
    pub rank: u32,
    pub world: u32,
    /// The width a full-projection Q norm normalises over: the model's `heads · head_dim`.
    pub q_norm_dim: usize,
    /// The width a full-projection K norm's all-reduced sum of squares counts: the model's
    /// `kv_heads · head_dim`, times the replication factor when KV heads are replicated (each
    /// rank of a replicated head adds its squares), so the mean is the model's.
    pub k_norm_dim: usize,
    /// The first vocabulary id of the rank's shard.
    pub vocab_offset: usize,
}

impl DecoderDims {
    pub fn of(cfg: &ModelArchConfig) -> DecoderDims {
        let head_dim = cfg.head_dim as usize;
        DecoderDims {
            hidden: cfg.hidden as usize,
            heads: cfg.num_attention_heads as usize,
            kv_heads: cfg.num_kv_heads as usize,
            head_dim,
            q_dim: cfg.num_attention_heads as usize * head_dim,
            kv_dim: cfg.num_kv_heads as usize * head_dim,
            inter: cfg.intermediate as usize,
            vocab: cfg.vocab_size as usize,
            layers: cfg.num_layers as usize,
            eps: cfg.rms_norm_eps,
            act: cfg.weight_format.0.activation_dtype(),
            tied_lm_head: cfg.tie_word_embeddings,
            moe: cfg.moe,
            vocab_rows: cfg.vocab_size as usize,
            tp: None,
        }
    }

    /// The dims tensor-parallel rank `shard` of `cfg` runs with ([`crate::tp::rank_config`]:
    /// its heads, KV heads and intermediate widths, its vocabulary shard); `None` is
    /// [`DecoderDims::of`]. Refuses a split the rules cannot make.
    pub fn for_shard(
        cfg: &ModelArchConfig,
        shard: Option<ShardSpec>,
    ) -> Result<DecoderDims, ModelError> {
        let Some(s) = shard else {
            return Ok(DecoderDims::of(cfg));
        };
        let mut d = DecoderDims::of(&rank_config(cfg, s)?);
        let (offset, _, padded) = vocab_shard(cfg.vocab_size, s);
        let kv_full = cfg.num_kv_heads as usize * d.head_dim;
        // Ranks per KV head: `world / kv_heads` when replicated, else 1.
        let replicas = if s.world > cfg.num_kv_heads {
            (s.world / cfg.num_kv_heads) as usize
        } else {
            1
        };
        d.vocab_rows = padded as usize;
        d.tp = Some(TpDims {
            rank: s.rank,
            world: s.world,
            q_norm_dim: cfg.num_attention_heads as usize * d.head_dim,
            k_norm_dim: kv_full * replicas,
            vocab_offset: offset as usize,
        });
        Ok(d)
    }

    /// `c = a · wᵀ` for an HF Linear weight `w` of `[n, k]` (activation-dtype operands).
    pub fn gemm(&self, n: usize, k: usize, c_dtype: DType) -> GemmConfig {
        GemmConfig {
            n: n as u64,
            k: k as u64,
            trans_b: true,
            a_dtype: self.act,
            b_dtype: self.act,
            c_dtype,
        }
    }

    /// RMSNorm over rows of `dim` elements.
    pub fn norm(&self, dim: usize) -> NormConfig {
        NormConfig {
            dim: dim as u64,
            dtype: self.act,
        }
    }

    /// The elementwise add (residual adds, the MoE accumulator reset).
    pub fn add(&self) -> ElementwiseConfig {
        ElementwiseConfig { dtype: self.act }
    }

    /// The residual add fused with the RMSNorm over `hidden` that follows it (kernel ABI v2.1).
    pub fn add_norm(&self) -> AddRmsnormConfig {
        AddRmsnormConfig {
            dtype: self.act,
            dim: self.hidden as u32,
        }
    }

    fn embedding(&self) -> EmbeddingConfig {
        EmbeddingConfig {
            hidden: self.hidden as u64,
            vocab_rows: self.vocab_rows as u64,
            dtype: self.act,
        }
    }

    fn rope(&self) -> RopeConfig {
        RopeConfig {
            num_q_heads: self.heads as u32,
            num_kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            rotary_dim: self.head_dim as u32,
            dtype: self.act,
        }
    }

    /// Paged causal GQA attention over blocks of `block_tokens` tokens.
    fn attention(&self, kind: AttentionKind, block_tokens: u32) -> AttentionConfig {
        AttentionConfig {
            kind,
            num_q_heads: self.heads as u32,
            num_kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            dtype: self.act,
            block_tokens: Some(block_tokens),
            causal: true,
        }
    }

    /// Parameter `name` taken from `weights`, checked to be the `shape` matrix (or stack of
    /// matrices) in the activation dtype.
    pub fn take_weight(
        &self,
        weights: &mut LoadedWeights,
        name: &str,
        shape: &[usize],
    ) -> Result<Tensor, ModelError> {
        let w = weights.take(name)?;
        if w.shape.as_slice() != shape || w.dtype != self.act {
            return Err(invalid(format!(
                "{name} is {:?} {}, expected {shape:?} {}",
                w.shape.as_slice(),
                w.dtype.as_str(),
                self.act.as_str()
            )));
        }
        Ok(w)
    }
}

/// One layer's parameters of a hook, in the order the hook took them.
pub struct HookWeights(pub Vec<Tensor>);

/// A hook's activation buffers, allocated once for `max_batch_tokens` tokens
/// ([`FfnHook::alloc`]): tensors and raw provider scratch, in the order the hook allocated them.
#[derive(Default)]
pub struct HookBuffers {
    pub tensors: Vec<Tensor>,
    pub scratch: Vec<DeviceBuffer>,
}

/// The attention variant: what runs on the new rows' Q and K projections before RoPE.
pub trait AttentionHook: Send + Sync {
    fn name(&self) -> &'static str;
    /// The op configs the hook runs, listed after the Q/K/V projection GEMMs.
    fn requirements(&self, d: &DecoderDims) -> Vec<OpConfig>;
    /// Whether the Q/K/V projections may run as one GEMM into the row-interleaved buffer (the
    /// hook's ops then read row-strided views); `false` keeps them dense.
    fn fuses_qkv(&self) -> bool;
    /// Takes the hook's parameters of the layer whose names start with `prefix`
    /// (`model.layers.<i>`).
    fn load_layer(
        &self,
        d: &DecoderDims,
        prefix: &str,
        weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError>;
    /// Runs on the projections `q` (`[tokens, q_dim]`) and `k` (`[tokens, kv_dim]`) in place,
    /// after the `q`, `k`, `v` trace points and before RoPE.
    fn after_projections(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
    ) -> Result<(), ModelError>;
}

/// The FFN variant: the block from the normalised rows ([`LayerRun::normed`]) to its output
/// ([`LayerRun::ffn_out`]), before the residual add.
pub trait FfnHook: Send + Sync {
    fn name(&self) -> &'static str;
    /// Refuses a configuration the hook cannot run (e.g. MoE without experts).
    fn check(&self, d: &DecoderDims) -> Result<(), ModelError> {
        let _ = d;
        Ok(())
    }
    /// The op configs the hook runs with `opts`, listed after the residual ops.
    fn requirements(&self, d: &DecoderDims, opts: ExecutorOptions) -> Vec<OpConfig>;
    /// Device bytes of the hook's buffers for `max_batch_tokens` tokens.
    fn workspace_bytes(&self, d: &DecoderDims, max_batch_tokens: usize) -> u64;
    /// Allocates (and initialises) the hook's buffers for `max_batch_tokens` tokens; the
    /// skeleton synchronises `mem` afterwards.
    fn alloc(
        &self,
        d: &DecoderDims,
        max_batch_tokens: usize,
        mem: &Arc<dyn DeviceMemory>,
    ) -> Result<HookBuffers, ModelError>;
    /// Takes layer `layer`'s FFN parameters (names starting with `prefix`) for `opts`.
    fn load_layer(
        &self,
        d: &DecoderDims,
        layer: u32,
        prefix: &str,
        weights: &mut LoadedWeights,
        opts: ExecutorOptions,
    ) -> Result<HookWeights, ModelError>;
    /// The block on `run.tokens` rows: reads [`LayerRun::normed`], writes
    /// [`LayerRun::ffn_out`].
    fn forward(&self, run: &LayerRun<'_>, w: &HookWeights) -> Result<(), ModelError>;
    /// Whether a decode batch of `rows` tokens may be captured into a decode graph (no host
    /// round trip mid-forward).
    fn graph_capturable(&self, d: &DecoderDims, registry: &KernelRegistry, rows: usize) -> bool {
        let _ = (d, registry, rows);
        true
    }
}

/// One decoder layer's forward, as the hooks see it: its dimensions, the registry, the row
/// count and the op wrappers (profile and trace) every op goes through.
pub struct LayerRun<'a> {
    pub layer: usize,
    pub dims: &'a DecoderDims,
    pub registry: &'a KernelRegistry,
    /// Rows of the batch (tokens).
    pub tokens: usize,
    pub opts: ExecutorOptions,
    /// The FFN hook's buffers ([`FfnHook::alloc`]).
    pub ffn_buffers: &'a HookBuffers,
    exec: &'a DecoderExecutor,
}

impl LayerRun<'_> {
    /// Runs the registry op `spec` through `f`: while profiling, the stream is synchronised
    /// after it and the time recorded under the op and its selected implementation.
    pub fn op(
        &self,
        spec: OpConfig,
        f: impl FnOnce() -> Result<(), KernelError>,
    ) -> Result<(), ModelError> {
        self.exec.op(spec, f)
    }

    /// Runs the executor transfer `op` (implementation `imp`, e.g. a host read) through `f`,
    /// timed like [`LayerRun::op`].
    pub fn step<R>(
        &self,
        op: &'static str,
        imp: &'static str,
        f: impl FnOnce() -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        self.exec.profiler.step(self.exec.mem.as_ref(), op, imp, f)
    }

    /// `c = a · wᵀ` for a `[n, k]` weight view `w`.
    pub fn linear(
        &self,
        a: TensorView<'_>,
        w: TensorView<'_>,
        c: TensorView<'_>,
    ) -> Result<(), ModelError> {
        self.exec.linear(a, w, c)
    }

    /// RMSNorm of `x` with weight `w` over rows of `w.len()` elements into `out` (may be `x`).
    pub fn rmsnorm(
        &self,
        x: TensorView<'_>,
        w: &Tensor,
        out: TensorView<'_>,
    ) -> Result<(), ModelError> {
        self.exec.rmsnorm(x, w, out)
    }

    /// The FFN input: the normalised rows `[tokens, hidden]`.
    pub fn normed(&self) -> TensorView<'_> {
        rows(&self.exec.bufs.h, self.tokens)
    }

    /// The FFN output rows `[tokens, hidden]`, which the residual add reads.
    pub fn ffn_out(&self) -> TensorView<'_> {
        rows(&self.exec.bufs.proj, self.tokens)
    }

    /// Records `view` under the layer and `name` when tracing.
    pub fn trace(&self, name: &'static str, view: &TensorView<'_>) -> Result<(), ModelError> {
        self.exec.trace.record(Some(self.layer), name, view)
    }

    /// Tensor parallelism: sums the contiguous `view` across the group's ranks in place (its
    /// dtype, BF16 or F32); nothing on one device.
    pub fn all_reduce(&self, view: &TensorView<'_>) -> Result<(), ModelError> {
        self.exec.all_reduce(view)
    }

    /// Tensor parallelism: the F32 `[2 · max_batch_tokens]` scratch a sharded norm keeps its
    /// partial sums of squares in (the Q rows' then the K rows', so one all-reduce covers
    /// both); `None` on one device.
    pub fn sumsq(&self) -> Option<&Tensor> {
        self.exec.tp_bufs.as_ref().map(|b| &b.sumsq)
    }
}

/// Rows `[0, t)` of an activation buffer as `[t, cols]`.
pub fn rows(buf: &Tensor, t: usize) -> TensorView<'_> {
    buf.view().rows(0, t)
}

/// Refuses a configuration whose weight format stores weights, activations and KV in different
/// dtypes (the skeleton runs one dtype for all three).
fn check_weight_format(cfg: &ModelArchConfig) -> Result<(), ModelError> {
    let format = cfg.weight_format.0;
    let act = format.activation_dtype();
    if format.weight_dtype() == act && format.kv_dtype() == act {
        Ok(())
    } else {
        Err(invalid(format!(
            "the decoder executor runs weights, activations and KV in one dtype; weight format \
             {} is not supported",
            format.name()
        )))
    }
}

fn batch_limits(cfg: &ModelArchConfig, limits: ExecutorLimits) -> BatchLimits {
    BatchLimits {
        vocab: cfg.vocab_size,
        max_batch_tokens: limits.max_batch_tokens,
        max_seqs: limits.max_seqs,
        max_positions: cfg.max_position_embeddings,
        layout: cfg.kv_layout(limits.block_tokens),
    }
}

/// The skeleton's parameters of one layer, and the hooks'.
struct Layer {
    input_norm: Tensor,
    /// `[q_dim + 2·kv_dim, hidden]`: Q rows, then K, then V.
    w_qkv: Tensor,
    wo: Tensor,
    post_norm: Tensor,
    attention: HookWeights,
    ffn: HookWeights,
}

/// Activation buffers, allocated once: `[max_batch_tokens, cols]` per token row, `[max_seqs,
/// cols]` per sequence row.
struct Buffers {
    inv_freq: Tensor,
    /// Residual stream.
    x: Tensor,
    /// Normalised input of the attention and FFN blocks.
    h: Tensor,
    /// Q, K and V projections of the new rows (Q and K rewritten in place by the attention
    /// hook and RoPE; K and V appended by attention), laid out by `DecoderExecutor::qkv`.
    qkv: Tensor,
    attn: Tensor,
    /// O projection and FFN outputs before the residual add.
    proj: Tensor,
    /// Final-norm output of each sequence's last row.
    last: Tensor,
}

/// A tensor-parallel rank's extra buffers: the sharded norms' partial sums and the LM head's
/// shard and gathered logits.
struct TpBuffers {
    /// F32 `[2 · max_batch_tokens]`.
    sumsq: Tensor,
    /// F32 `[max_seqs, vocab_rows]`: this rank's LM-head shard, the all-gather's input.
    shard_logits: Tensor,
    /// F32 `[world · max_seqs, vocab_rows]`: every rank's shard, rank-major.
    gathered: Tensor,
}

/// The decoder executor: ragged batches of up to `max_seqs` sequences and `max_batch_tokens`
/// tokens over the paged KV pool, on one device (or as one rank of a tensor-parallel group,
/// [`DecoderExecutor::new_tp`]), for any [`DecoderSpec`].
pub struct DecoderExecutor {
    /// Decode graphs, when on. First field: the graphs record pointers into the buffers below
    /// and are destroyed before them.
    graphs: Option<DecodeGraphs>,
    /// The model's configuration (unsharded).
    cfg: ModelArchConfig,
    /// The rank's collectives, when tensor-parallel.
    tp: Option<TpContext>,
    tp_bufs: Option<TpBuffers>,
    spec: DecoderSpec,
    dims: DecoderDims,
    shape: ModelShape,
    kv_layout: KvLayout,
    limits: BatchLimits,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    /// The op sequence ([`DecoderExecutor::requirements`] of these options).
    opts: ExecutorOptions,
    /// Q/K/V (`[q_dim, kv_dim, kv_dim]`) in their buffer, fused or not.
    qkv: Split<3>,
    /// Residual adds followed by a norm run as one `add_rmsnorm` (`fused_ops` and a provider
    /// selected for it).
    add_norm: bool,
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    /// `None` when the model ties its LM head to `embed`.
    lm_head: Option<Tensor>,
    bufs: Buffers,
    ffn_buffers: HookBuffers,
    /// The LM head output, its device reduction and the logits copy.
    head: LogitsHead,
    meta: DeviceBatch,
    /// Host bytes of the last packed batch.
    host: HostBatch,
    /// Host launch time of each uncollected launch, oldest first.
    launches: VecDeque<Duration>,
    /// Timings of the last collected step ([`ModelExecutor::last_timings`]).
    timings: ForwardTimings,
    trace: Tracer,
    profiler: Profiler,
    /// The batch being enqueued prefills prompt tokens (not decode-only): its GEMMs ask for
    /// rows independent of the batch ([`GemmContext::prefill`]), so a prefix-reused prefill of a
    /// suffix reproduces the whole-prompt prefill.
    step_prefill: AtomicBool,
}

impl DecoderExecutor {
    /// Every distinct op config the forward pass of `spec` executes with `opts`, in first-use
    /// order, plus the `copy_blocks` fork over KV blocks of `block_tokens` tokens. The registry
    /// is built from this list at startup, so a config no provider supports fails before any
    /// weight is read. With `fused_ops` the list has the optional `add_rmsnorm`, which
    /// [`super::available_requirements`] drops when no provider has it.
    pub fn requirements(
        cfg: &ModelArchConfig,
        spec: &DecoderSpec,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        Self::requirements_of(cfg, &DecoderDims::of(cfg), spec, block_tokens, opts)
    }

    /// [`DecoderExecutor::requirements`] of tensor-parallel rank `shard` (`None`: one device):
    /// the rank's GEMM, attention, RoPE and FFN widths, its vocabulary shard's embedding and LM
    /// head, its KV layout's block copy, and a full-projection Q/K norm as `row_sumsq` and
    /// `rmsnorm_sharded`.
    pub fn requirements_for(
        cfg: &ModelArchConfig,
        spec: &DecoderSpec,
        block_tokens: u32,
        opts: ExecutorOptions,
        shard: Option<ShardSpec>,
    ) -> Result<Vec<OpRequirement>, ModelError> {
        let d = DecoderDims::for_shard(cfg, shard)?;
        let rank = match shard {
            Some(s) => rank_config(cfg, s)?,
            None => cfg.clone(),
        };
        Ok(Self::requirements_of(&rank, &d, spec, block_tokens, opts))
    }

    /// The requirements of dims `d` over the KV layout of `cfg` (the rank's configuration).
    fn requirements_of(
        cfg: &ModelArchConfig,
        d: &DecoderDims,
        spec: &DecoderSpec,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        let d = *d;
        let act = d.act;
        let mut specs = vec![
            OpConfig::Embedding(d.embedding()),
            OpConfig::Rmsnorm(d.norm(d.hidden)),
        ];
        if opts.fused_projections && spec.attention.fuses_qkv() {
            specs.push(OpConfig::Gemm(d.gemm(
                d.q_dim + 2 * d.kv_dim,
                d.hidden,
                act,
            )));
        } else {
            specs.extend([
                OpConfig::Gemm(d.gemm(d.q_dim, d.hidden, act)),
                OpConfig::Gemm(d.gemm(d.kv_dim, d.hidden, act)),
            ]);
        }
        specs.extend(spec.attention.requirements(&d));
        specs.extend([
            OpConfig::Rope(d.rope()),
            OpConfig::Attention(d.attention(AttentionKind::PrefillPaged, block_tokens)),
            OpConfig::Attention(d.attention(AttentionKind::DecodePaged, block_tokens)),
            OpConfig::Gemm(d.gemm(d.hidden, d.q_dim, act)),
            OpConfig::Add(d.add()),
        ]);
        if opts.fused_ops {
            specs.push(OpConfig::AddRmsnorm(d.add_norm()));
        }
        specs.extend(spec.ffn.requirements(&d, opts));
        specs.extend([
            OpConfig::Gemm(d.gemm(d.vocab_rows, d.hidden, DType::F32)),
            OpConfig::CopyBlocks(batch::copy_config(&cfg.kv_layout(block_tokens))),
        ]);
        let mut unique: Vec<OpConfig> = Vec::with_capacity(specs.len());
        for spec in specs {
            if !unique.contains(&spec) {
                unique.push(spec);
            }
        }
        unique.into_iter().map(OpRequirement::from).collect()
    }

    /// Device bytes of the executor's buffers (the budget's workspace term): per token the I32
    /// id and position and the rows of `x`, `h`, `proj` (hidden), `qkv` (q_dim + 2·kv_dim) and
    /// `attn` (q_dim) in the activation dtype; per sequence its last normalised row, its F32
    /// logits row and reduction ([`LogitsHead::bytes`]), its `q_indptr` and `kv_lens` entries
    /// and a block table for `max_position_embeddings` tokens (I32); one `q_indptr` entry and
    /// the F32 `inv_freq`; plus the FFN hook's buffers ([`FfnHook::workspace_bytes`]).
    pub fn workspace_bytes(
        cfg: &ModelArchConfig,
        spec: &DecoderSpec,
        limits: ExecutorLimits,
    ) -> u64 {
        Self::workspace_of(cfg, &DecoderDims::of(cfg), spec, limits)
    }

    /// [`DecoderExecutor::workspace_bytes`] of tensor-parallel rank `shard` (`None`: one
    /// device): the rank's widths, plus the F32 partial sums of squares (2 per token) and the
    /// LM head's shard and gathered logits (`(1 + world) · max_seqs · vocab_rows`).
    pub fn workspace_bytes_for(
        cfg: &ModelArchConfig,
        spec: &DecoderSpec,
        limits: ExecutorLimits,
        shard: Option<ShardSpec>,
    ) -> Result<u64, ModelError> {
        let d = DecoderDims::for_shard(cfg, shard)?;
        let Some(s) = shard else {
            return Ok(Self::workspace_bytes(cfg, spec, limits));
        };
        let rank = rank_config(cfg, s)?;
        let tp = DType::F32.size_bytes() as u64
            * (2 * u64::from(limits.max_batch_tokens)
                + (1 + u64::from(s.world)) * u64::from(limits.max_seqs) * d.vocab_rows as u64);
        Ok(Self::workspace_of(&rank, &d, spec, limits) + tp)
    }

    fn workspace_of(
        cfg: &ModelArchConfig,
        d: &DecoderDims,
        spec: &DecoderSpec,
        limits: ExecutorLimits,
    ) -> u64 {
        let d = *d;
        let es = d.act.size_bytes() as u64;
        let f32 = DType::F32.size_bytes() as u64;
        let per_token = es * (3 * d.hidden + 2 * d.q_dim + 2 * d.kv_dim) as u64;
        let per_seq = es * d.hidden as u64;
        let t = limits.max_batch_tokens;
        u64::from(t) * per_token
            + spec.ffn.workspace_bytes(&d, t as usize)
            + u64::from(limits.max_seqs) * per_seq
            + LogitsHead::bytes(d.vocab, limits.max_seqs as usize)
            + f32 * (d.head_dim / 2) as u64
            + DeviceBatch::bytes(&batch_limits(cfg, limits))
    }

    /// Takes the parameters out of `weights` (Q/K/V fused by the loader; the hooks take
    /// theirs), allocates the activation and batch buffers for `limits` on `mem`, and uploads
    /// the rotary inverse frequencies. The KV is not the executor's: every forward names its
    /// pool, laid out as `cfg.kv_layout(limits.block_tokens)`. `registry` must have been built
    /// from [`DecoderExecutor::requirements`] of `cfg`, `spec`, the block size and `opts`.
    pub fn new(
        cfg: &ModelArchConfig,
        spec: DecoderSpec,
        weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        limits: ExecutorLimits,
        opts: ExecutorOptions,
    ) -> Result<DecoderExecutor, ModelError> {
        Self::new_tp(cfg, spec, weights, registry, mem, limits, opts, None)
    }

    /// [`DecoderExecutor::new`] as rank `tp.rank` of a tensor-parallel group (`None`: one
    /// device, exactly `new`): `weights` are the rank's shard ([`crate::tp::weight_slots`]),
    /// `registry` was built from [`DecoderExecutor::requirements_for`] of the same shard, and
    /// every forward's pool is laid out as the rank's KV layout ([`crate::tp::kv_layout`]).
    /// Every rank of the group must be fed the same batches in the same order; each returns
    /// the full logits. Tensor-parallel ranks never capture decode graphs nor overlap launches.
    #[allow(clippy::too_many_arguments)]
    pub fn new_tp(
        cfg: &ModelArchConfig,
        spec: DecoderSpec,
        mut weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        limits: ExecutorLimits,
        opts: ExecutorOptions,
        tp: Option<TpContext>,
    ) -> Result<DecoderExecutor, ModelError> {
        let shard = tp.as_ref().map(TpContext::shard);
        if let Some(t) = &tp
            && (t.collective.world_size() != t.world as usize
                || t.collective.rank() != t.rank as usize)
        {
            return Err(invalid(format!(
                "tensor-parallel rank {} of {} over a communicator of rank {} of {}",
                t.rank,
                t.world,
                t.collective.rank(),
                t.collective.world_size()
            )));
        }
        let rank_cfg = match shard {
            Some(s) => rank_config(cfg, s)?,
            None => cfg.clone(),
        };
        let ExecutorLimits {
            block_tokens,
            max_batch_tokens,
            max_seqs,
        } = limits;
        if block_tokens == 0 || max_batch_tokens == 0 || max_seqs == 0 {
            return Err(invalid(format!(
                "block_tokens {block_tokens}, max_batch_tokens {max_batch_tokens} and max_seqs \
                 {max_seqs} must be positive"
            )));
        }
        check_weight_format(cfg)?;
        let d = DecoderDims::for_shard(cfg, shard)?;
        spec.ffn.check(&d)?;
        let qkv = Split {
            cols: [d.q_dim, d.kv_dim, d.kv_dim],
            fused: opts.fused_projections && spec.attention.fuses_qkv(),
        };
        let mut layers = Vec::with_capacity(d.layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let w_qkv = d.take_weight(&mut weights, &qkv_proj_name(i), &[qkv.width(), d.hidden])?;
            let mut take = |s: &str| weights.take(&format!("{p}.{s}.weight"));
            let (input_norm, wo, post_norm) = (
                take("input_layernorm")?,
                take("self_attn.o_proj")?,
                take("post_attention_layernorm")?,
            );
            let attention = spec.attention.load_layer(&d, &p, &mut weights)?;
            let ffn = spec.ffn.load_layer(&d, i, &p, &mut weights, opts)?;
            layers.push(Layer {
                input_norm,
                w_qkv,
                wo,
                post_norm,
                attention,
                ffn,
            });
        }
        let embed = weights.take("model.embed_tokens.weight")?;
        let final_norm = weights.take("model.norm.weight")?;
        let lm_head = if d.tied_lm_head {
            None
        } else {
            Some(weights.take(LM_HEAD)?)
        };

        let add_norm = opts.fused_ops && registry.is_selected(&OpConfig::AddRmsnorm(d.add_norm()));
        let limits = batch_limits(&rank_cfg, limits);
        let t = max_batch_tokens as usize;
        let n = max_seqs as usize;
        let act = |cols: usize| Tensor::empty(&mem, &[t, cols], d.act);
        let mut inv_freq = Tensor::empty(&mem, &[d.head_dim / 2], DType::F32)?;
        let freqs = rope::inv_freq(cfg.rope_theta, cfg.head_dim, cfg.rope_scaling.as_ref());
        let freq_bytes: Vec<u8> = freqs.iter().flat_map(|f| f.to_le_bytes()).collect();
        inv_freq.storage.copy_from_host(0, &freq_bytes)?;
        let ffn_buffers = spec.ffn.alloc(&d, t, &mem)?;
        mem.synchronize()?;
        let bufs = Buffers {
            inv_freq,
            x: act(d.hidden)?,
            h: act(d.hidden)?,
            qkv: act(qkv.width())?,
            attn: act(d.q_dim)?,
            proj: act(d.hidden)?,
            last: Tensor::empty(&mem, &[n, d.hidden], d.act)?,
        };
        let head = LogitsHead::new(cfg, &registry, &mem, n)?;
        let meta = DeviceBatch::alloc(&mem, &limits)?;
        let tp_bufs = match &tp {
            Some(tc) => Some(TpBuffers {
                sumsq: Tensor::empty(&mem, &[2 * t], DType::F32)?,
                shard_logits: Tensor::empty(&mem, &[n, d.vocab_rows], DType::F32)?,
                gathered: Tensor::empty(&mem, &[tc.world as usize * n, d.vocab_rows], DType::F32)?,
            }),
            None => None,
        };
        Ok(DecoderExecutor {
            graphs: None,
            cfg: cfg.clone(),
            tp,
            tp_bufs,
            spec,
            dims: d,
            shape: cfg.shape(),
            kv_layout: limits.layout,
            limits,
            registry,
            mem,
            opts,
            qkv,
            add_norm,
            embed,
            layers,
            final_norm,
            lm_head,
            bufs,
            ffn_buffers,
            head,
            meta,
            host: HostBatch::default(),
            launches: VecDeque::with_capacity(2),
            timings: ForwardTimings::default(),
            trace: Tracer::default(),
            profiler: Profiler::default(),
            step_prefill: AtomicBool::new(false),
        })
    }

    /// The family's hooks.
    pub fn spec(&self) -> DecoderSpec {
        self.spec
    }

    /// Diagnostics: while enabled every forward records its intermediate tensors (one blocking
    /// device read per trace point), returned by [`DecoderExecutor::take_trace`]. Disabling
    /// drops them.
    pub fn set_trace(&mut self, enabled: bool) {
        self.trace.set(enabled);
    }

    /// The tensors recorded since tracing was enabled or last taken, in execution order; empty
    /// when tracing is off.
    pub fn take_trace(&mut self) -> Vec<TraceTensor> {
        self.trace.take()
    }

    /// Diagnostics: while enabled every op of every forward is followed by a stream
    /// synchronisation and timed into the profile returned by
    /// [`DecoderExecutor::take_profile`]. Disabling drops what was recorded. Off by default.
    pub fn set_profile(&mut self, on: bool) {
        let mut reqs = Self::requirements_for(
            &self.cfg,
            &self.spec,
            self.kv_layout.block_tokens,
            self.opts,
            self.tp.as_ref().map(TpContext::shard),
        )
        .unwrap_or_default();
        reqs.push(logits::reduce_requirement(&self.cfg));
        self.profiler.set(on, &self.registry, &reqs);
    }

    /// The op profile recorded since profiling was enabled or last taken; empty when it is off.
    pub fn take_profile(&mut self) -> OpProfile {
        self.profiler.take()
    }

    /// The decode graph key [`ModelExecutor::launch`] would run `batch` with `feeds` under
    /// (graphs on, no tracing or profiling, the FFN hook allowing capture): `None` when the
    /// batch is not decode-only, mixes reduced and full rows or feeds tokens other than one
    /// run over every token. Changes nothing; fails as `launch` does on an invalid batch or
    /// feed.
    pub fn graph_key(
        &self,
        batch: &BatchInput<'_>,
        feeds: &[TokenFeed],
    ) -> Result<Option<GraphKey>, ModelError> {
        let runs = feed_runs(feeds, batch.tokens.len(), |slot| self.head.feed_word(slot))?;
        let mut host = HostBatch::default();
        let mut p = host.pack(batch, &self.limits)?;
        Ok(self
            .head
            .graph_top_n_of(batch.seqs)
            .and_then(|top_n| graphs::decode_key(&mut p, &mut host, &self.limits, top_n, &runs)))
    }

    /// Runs the registry op `spec` through `f`: every op of the forward goes through here, so
    /// profile mode times each one.
    fn op(
        &self,
        spec: OpConfig,
        f: impl FnOnce() -> Result<(), KernelError>,
    ) -> Result<(), ModelError> {
        self.profiler
            .op(self.mem.as_ref(), spec, || f().map_err(ModelError::from))
    }

    /// Records `view` under `layer` and `name` when tracing.
    fn record(
        &self,
        layer: Option<usize>,
        name: &'static str,
        view: TensorView<'_>,
    ) -> Result<(), ModelError> {
        self.trace.record(layer, name, &view)
    }

    /// Rows `[0, t)` of an activation buffer as `[t, heads, head_dim]`.
    fn heads<'b>(&self, buf: &'b Tensor, t: usize) -> TensorView<'b> {
        let d = &self.dims;
        TensorView::contiguous(buf.storage.whole(), 0, &[t, d.heads, d.head_dim], d.act)
    }

    /// `c = a · wᵀ` for a `[n, k]` weight view `w`.
    fn linear(
        &self,
        a: TensorView<'_>,
        w: TensorView<'_>,
        c: TensorView<'_>,
    ) -> Result<(), ModelError> {
        let cfg = self.dims.gemm(w.shape[0], w.shape[1], c.dtype);
        self.op(OpConfig::Gemm(cfg), || {
            self.registry.gemm(&cfg).execute(&mut GemmContext {
                a,
                b: w,
                c,
                trans_b: true,
                alpha: 1.0,
                beta: 0.0,
                prefill: self.step_prefill.load(Ordering::Relaxed),
            })
        })
    }

    /// Tensor parallelism: sums the contiguous `view` (BF16 or F32) across the group in place,
    /// on the rank's stream; nothing on one device.
    fn all_reduce(&self, view: &TensorView<'_>) -> Result<(), ModelError> {
        let Some(tp) = &self.tp else {
            return Ok(());
        };
        if view.slice.len() != view.numel() * view.dtype.size_bytes() {
            return Err(invalid(format!(
                "all-reduce of a non-contiguous view {:?} strides {:?}",
                view.shape.as_slice(),
                view.strides.as_slice()
            )));
        }
        let (mut slice, dtype) = (view.slice, view.dtype);
        self.profiler.step(
            self.mem.as_ref(),
            profile::TP_ALL_REDUCE,
            tp.collective.backend(),
            || {
                Ok(tp
                    .collective
                    .all_reduce(&mut slice, dtype, ReduceOp::Sum, &tp.stream)?)
            },
        )
    }

    /// Tensor parallelism: the LM head of the `n` final-norm rows over this rank's vocabulary
    /// shard, all-gathered from every rank (rank-major `[world][n][vocab_rows]`) and reordered
    /// by device-to-device copies into the logits head's row-major `[n, vocab]` rows, each
    /// rank's real rows only (an uneven last shard's padding columns stay behind).
    fn tp_lm_head(
        &self,
        tp: &TpContext,
        bufs: &TpBuffers,
        head: &Tensor,
        n: usize,
    ) -> Result<(), ModelError> {
        let d = &self.dims;
        let (vr, vocab, world) = (d.vocab_rows, d.vocab, tp.world as usize);
        let f32 = DType::F32.size_bytes();
        let shard =
            TensorView::contiguous(bufs.shard_logits.storage.whole(), 0, &[n, vr], DType::F32);
        self.linear(rows(&self.bufs.last, n), head.view(), shard.clone())?;
        let mut gathered = bufs.gathered.storage.whole().sub(0, world * n * vr * f32);
        self.profiler.step(
            self.mem.as_ref(),
            profile::TP_ALL_GATHER,
            tp.collective.backend(),
            || {
                Ok(tp
                    .collective
                    .all_gather(&shard.slice, &mut gathered, &tp.stream)?)
            },
        )?;
        let out = self.head.rows(n).slice;
        self.profiler.step(
            self.mem.as_ref(),
            profile::TP_LOGITS_REORDER,
            profile::D2D,
            || {
                if n == 1 {
                    // One row: the ranks' shards are already consecutive.
                    self.mem.copy_d2d(out.ptr(), gathered.ptr(), vocab * f32)?;
                    return Ok(());
                }
                for r in 0..world {
                    let (offset, real, _) = vocab_shard(
                        vocab as u32,
                        ShardSpec {
                            rank: r as u32,
                            world: tp.world,
                        },
                    );
                    let (offset, real) = (offset as usize, real as usize);
                    if real == 0 {
                        continue;
                    }
                    for i in 0..n {
                        let src = gathered.sub((r * n + i) * vr * f32, real * f32);
                        let dst = out.sub((i * vocab + offset) * f32, real * f32);
                        self.mem.copy_d2d(dst.ptr(), src.ptr(), real * f32)?;
                    }
                }
                Ok(())
            },
        )
    }

    /// RMSNorm over rows of `w.numel()` elements.
    fn rmsnorm(
        &self,
        x: TensorView<'_>,
        w: &Tensor,
        out: TensorView<'_>,
    ) -> Result<(), ModelError> {
        let cfg = self.dims.norm(w.numel());
        self.op(OpConfig::Rmsnorm(cfg), || {
            self.registry.norm(&cfg).execute(&mut NormContext {
                x,
                weight: w.view(),
                out,
                eps: self.dims.eps,
            })
        })
    }

    /// `x[0..t] += proj[0..t]` (the residual add), then with `norm`
    /// `h[0..t] = rmsnorm(x[0..t]) · norm`: one `add_rmsnorm` when `add_norm`, else `add` and
    /// `rmsnorm`.
    fn residual_add_norm(&self, t: usize, norm: Option<&Tensor>) -> Result<(), ModelError> {
        let b = &self.bufs;
        let (x, h, proj) = (rows(&b.x, t), rows(&b.h, t), rows(&b.proj, t));
        match norm {
            Some(w) if self.add_norm => {
                let cfg = self.dims.add_norm();
                self.op(OpConfig::AddRmsnorm(cfg), || {
                    self.registry
                        .add_rmsnorm(&cfg)
                        .execute(&mut AddRmsnormContext {
                            residual: x,
                            x: proj,
                            weight: w.view(),
                            out: h,
                            eps: self.dims.eps,
                        })
                })?;
            }
            _ => {
                let add = self.dims.add();
                self.op(OpConfig::Add(add), || {
                    self.registry
                        .elementwise(&add)
                        .execute(&mut ElementwiseContext {
                            a: x.clone(),
                            b: proj,
                            out: x.clone(),
                        })
                })?;
                if let Some(w) = norm {
                    self.rmsnorm(x, w, h)?;
                }
            }
        }
        Ok(())
    }

    /// One decoder layer on the batch's `p.total_q` rows, whose normalised input is already in
    /// `h` (the embedding's norm, or the previous layer's last op); ends with the next layer's
    /// input norm in `h`, if there is a next layer. Attention appends to and reads layer `i` of
    /// `kv`.
    fn layer(&self, i: usize, p: &Packed, kv: &KvPoolView<'_>) -> Result<(), ModelError> {
        let l = &self.layers[i];
        let b = &self.bufs;
        let d = &self.dims;
        let t = p.total_q;
        let li = Some(i);
        let run = LayerRun {
            layer: i,
            dims: d,
            registry: &self.registry,
            tokens: t,
            opts: self.opts,
            ffn_buffers: &self.ffn_buffers,
            exec: self,
        };
        let (q, k, v) = (
            || self.qkv.part(&b.qkv, 0, t, &[d.q_dim]),
            || self.qkv.part(&b.qkv, 1, t, &[d.kv_dim]),
            || self.qkv.part(&b.qkv, 2, t, &[d.kv_dim]),
        );

        // Attention block.
        self.record(li, "attn_norm", rows(&b.h, t))?;
        if self.qkv.fused {
            let w = l.w_qkv.view();
            self.linear(rows(&b.h, t), w, self.qkv.whole(&b.qkv, t))?;
        } else {
            let w = |r, n| l.w_qkv.view().rows(r, n);
            let h = || rows(&b.h, t);
            self.linear(h(), w(0, d.q_dim), q())?;
            self.linear(h(), w(d.q_dim, d.kv_dim), k())?;
            self.linear(h(), w(d.q_dim + d.kv_dim, d.kv_dim), v())?;
        }
        self.record(li, "q", q())?;
        self.record(li, "k", k())?;
        self.record(li, "v", v())?;
        self.spec
            .attention
            .after_projections(&run, &l.attention, &q(), &k())?;
        let q_heads = || self.qkv.part(&b.qkv, 0, t, &[d.heads, d.head_dim]);
        let k_heads = || self.qkv.part(&b.qkv, 1, t, &[d.kv_heads, d.head_dim]);
        let v_heads = || self.qkv.part(&b.qkv, 2, t, &[d.kv_heads, d.head_dim]);
        let rope = d.rope();
        self.op(OpConfig::Rope(rope), || {
            self.registry.rope(&rope).execute(&mut RopeContext {
                cfg: rope,
                q: q_heads(),
                k: k_heads(),
                positions: self.meta.positions_view(p),
                inv_freq: b.inv_freq.view(),
            })
        })?;
        self.record(li, "q_rope", q())?;
        self.record(li, "k_rope", k())?;
        let kind = if p.is_decode() {
            AttentionKind::DecodePaged
        } else {
            AttentionKind::PrefillPaged
        };
        let attn = d.attention(kind, self.kv_layout.block_tokens);
        self.op(OpConfig::Attention(attn), || {
            self.registry
                .attention(&attn)
                .execute_paged(&mut PagedAttentionContext {
                    cfg: attn,
                    q: q_heads(),
                    k_new: k_heads(),
                    v_new: v_heads(),
                    out: self.heads(&b.attn, t),
                    kv_layer: batch::kv_layer(kv, i),
                    block_table: self.meta.block_table_view(p),
                    q_indptr: self.meta.q_indptr_view(p),
                    kv_lens: self.meta.kv_lens_view(p),
                    max_q_len: p.max_q_len,
                    max_kv_len: p.max_kv_len,
                    max_blocks_per_seq: p.max_blocks_per_seq,
                    scale: 1.0 / (d.head_dim as f32).sqrt(),
                })
        })?;
        self.record(li, "attn", rows(&b.attn, t))?;
        self.linear(rows(&b.attn, t), l.wo.view(), rows(&b.proj, t))?;
        self.all_reduce(&rows(&b.proj, t))?;
        self.record(li, "o_proj", rows(&b.proj, t))?;
        self.residual_add_norm(t, Some(&l.post_norm))?;
        self.record(li, "resid_attn", rows(&b.x, t))?;

        // FFN block.
        self.record(li, "mlp_norm", rows(&b.h, t))?;
        self.spec.ffn.forward(&run, &l.ffn)?;
        self.all_reduce(&rows(&b.proj, t))?;
        let next_norm = self.layers.get(i + 1).map(|next| &next.input_norm);
        self.residual_add_norm(t, next_norm)?;
        self.record(li, "resid_mlp", rows(&b.x, t))
    }

    /// Final RMSNorm of each sequence's last row into `last[0..n]`, destination row `p` holding
    /// sequence `order[p]` ([`LogitsHead::plan`]): one call per run of consecutive rows (a
    /// decode-only batch is one call).
    fn final_norm_last_rows(&self, p: &Packed, order: &[usize]) -> Result<(), ModelError> {
        let b = &self.bufs;
        let x = rows(&b.x, p.total_q);
        for (src, dst, len) in logits::norm_runs(&p.last_rows, order) {
            self.rmsnorm(
                x.rows(src, len),
                &self.final_norm,
                b.last.view().rows(dst, len),
            )?;
        }
        Ok(())
    }

    /// Every op of a forward over the uploaded batch `p`, rows of the logits in `order`
    /// ([`LogitsHead::plan`]): embedding, the layers, the final norm, the LM head and the
    /// logits reduction. No host copy or synchronisation unless a hook needs one (which then
    /// refuses capture, [`FfnHook::graph_capturable`]), so a decode graph can capture it.
    fn enqueue(
        &self,
        p: &Packed,
        kv: &KvPoolView<'_>,
        order: &[usize],
        runs: &[(usize, usize, usize)],
    ) -> Result<(), ModelError> {
        let b = &self.bufs;
        let (t, n) = (p.total_q, p.num_seqs);
        self.step_prefill.store(!p.is_decode(), Ordering::Relaxed);
        let embedding = self.dims.embedding();
        let vocab_offset = self.dims.tp.map_or(0, |tp| tp.vocab_offset as i64);
        self.op(OpConfig::Embedding(embedding), || {
            self.registry
                .embedding(&embedding)
                .execute(&mut EmbeddingContext {
                    ids: self.meta.ids_view(p),
                    table: self.embed.view(),
                    out: rows(&b.x, t),
                    vocab_offset,
                })
        })?;
        for &(token, word, len) in runs {
            self.op(OpConfig::Embedding(embedding), || {
                self.registry
                    .embedding(&embedding)
                    .execute(&mut EmbeddingContext {
                        ids: self.head.ids_at(word, len),
                        table: self.embed.view(),
                        out: rows(&b.x, t).rows(token, len),
                        vocab_offset,
                    })
            })?;
        }
        // Each id was embedded by the rank whose shard holds it, zeros elsewhere.
        self.all_reduce(&rows(&b.x, t))?;
        self.record(None, "embed", rows(&b.x, t))?;
        if let Some(first) = self.layers.first() {
            self.rmsnorm(rows(&b.x, t), &first.input_norm, rows(&b.h, t))?;
        }
        for i in 0..self.layers.len() {
            self.layer(i, p, kv)?;
        }
        self.final_norm_last_rows(p, order)?;
        self.record(None, "final_norm", rows(&b.last, n))?;
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        match (&self.tp, &self.tp_bufs) {
            (Some(tp), Some(bufs)) => self.tp_lm_head(tp, bufs, head, n)?,
            _ => self.linear(rows(&b.last, n), head.view(), self.head.rows(n))?,
        }
        self.record(None, "logits", self.head.rows(n))?;
        self.head
            .reduce(&self.registry, &self.profiler, self.mem.as_ref())
    }
}

impl ModelExecutor for DecoderExecutor {
    fn shape(&self) -> &ModelShape {
        &self.shape
    }

    fn kv_layout(&self) -> &KvLayout {
        &self.kv_layout
    }

    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
        if self.head.pending() > 0 {
            return Err(invalid(
                "forward while a launched step is not collected".into(),
            ));
        }
        self.launch(batch, &[])?;
        self.collect()
    }

    /// Never for a tensor-parallel rank: its collectives run inside the launch.
    fn overlaps(&self) -> bool {
        self.tp.is_none() && self.head.overlaps() && self.meta.is_async()
    }

    fn launch(&mut self, batch: &BatchInput<'_>, feeds: &[TokenFeed]) -> Result<(), ModelError> {
        let in_flight = if self.overlaps() { 2 } else { 1 };
        if self.head.pending() >= in_flight {
            return Err(invalid(format!(
                "{in_flight} launched steps are not collected yet"
            )));
        }
        if !feeds.is_empty() && self.head.pending() == 0 && !self.overlaps() {
            return Err(invalid(
                "token feeds need an executor that overlaps steps".into(),
            ));
        }
        // Feeds read the previous launch's choices: resolve them before this launch replans.
        let runs = feed_runs(feeds, batch.tokens.len(), |slot| self.head.feed_word(slot))?;
        let order = self.head.plan(batch.seqs).to_vec();
        let graph_top_n = self
            .graphs
            .as_ref()
            .filter(|g| g.is_enabled() && !self.profiler.is_on() && !self.trace.is_on())
            // E.g. a MoE provider reading the group sizes on the host copies them back
            // mid-forward.
            .filter(|_| {
                self.spec
                    .ffn
                    .graph_capturable(&self.dims, &self.registry, batch.tokens.len())
            })
            .and_then(|_| self.head.graph_top_n());
        let (host, meta, limits) = (&mut self.host, &mut self.meta, &self.limits);
        let (p, key) = self.profiler.step(
            self.mem.as_ref(),
            profile::BATCH_UPLOAD,
            profile::HOST,
            || {
                let mut p = host.pack(batch, limits)?;
                let key = graph_top_n
                    .and_then(|top_n| graphs::decode_key(&mut p, host, limits, top_n, &runs));
                meta.upload(host)?;
                Ok((p, key))
            },
        )?;
        self.head.upload_inputs()?;
        let launch_started = Instant::now();
        if let Some(tp) = &self.tp {
            // Bounds the step's collectives until `collect` has synchronised the stream.
            tp.collective.step_begin();
        }
        let mut graphs = self.graphs.take();
        let enqueued = match graphs.as_mut() {
            Some(g) => g.run(key, PoolId::of(batch.kv), || {
                self.enqueue(&p, batch.kv, &order, &runs)
            }),
            None => self.enqueue(&p, batch.kv, &order, &runs),
        };
        self.graphs = graphs;
        if let (Err(_), Some(tp)) = (&enqueued, &self.tp) {
            // The step is abandoned: close its bound (the error that ended it is returned).
            let _ = tp.collective.step_end();
        }
        enqueued?;
        // The step's one device-to-host read, after the device reduction of the rows that
        // asked for one; collected by `collect`.
        self.head.launch_read()?;
        self.launches.push_back(launch_started.elapsed());
        Ok(())
    }

    fn collect(&mut self) -> Result<Logits, ModelError> {
        let wait_started = Instant::now();
        let launch = self.launches.pop_front().unwrap_or_default();
        let logits = self.head.collect(&self.profiler, self.mem.as_ref());
        if let Some(tp) = &self.tp {
            // The read synchronised the stream: the step's collectives are complete.
            let ended = tp.collective.step_end();
            if logits.is_ok() {
                ended?;
            }
        }
        let logits = logits?;
        self.timings = ForwardTimings {
            launch,
            device_wait: wait_started.elapsed(),
        };
        Ok(logits)
    }

    /// Ignored (graphs stay off) for a tensor-parallel rank: its collectives are not captured.
    fn set_decode_graphs(&mut self, graphs: Option<DecodeGraphs>) {
        if self.tp.is_some() && graphs.is_some() {
            tracing::info!(
                event = "decode_graphs_off",
                reason = "tensor_parallel",
                "decode graphs are not captured under tensor parallelism"
            );
            return;
        }
        self.graphs = graphs;
    }

    fn graph_counters(&self) -> GraphCounters {
        self.graphs
            .as_ref()
            .map(DecodeGraphs::counters)
            .unwrap_or_default()
    }

    fn reduces_logits(&self) -> bool {
        self.head.reduces()
    }

    fn last_timings(&self) -> ForwardTimings {
        self.timings
    }

    fn copy_blocks(
        &mut self,
        kv: &KvPoolView<'_>,
        src: &[BlockId],
        dst: &[BlockId],
    ) -> Result<(), ModelError> {
        batch::copy_blocks(&self.registry, &self.kv_layout, kv, src, dst)
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::types::{DeviceId, SeqId};
    use turbine_kernels::{KernelMetrics, cpu_reference_provider};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::executor::{RowReduce, SeqSlice};
    use crate::families;
    use crate::testing::TempDir;
    use crate::testing::tiny::write_tiny_llama;
    use crate::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader};

    const BLOCK_TOKENS: u32 = 16;
    const MAX_SEQS: u32 = 16;
    const TABLE: u32 = 20;

    /// The GraphKey formula of main's `LlamaExecutor::launch` / `OlmoeExecutor::launch`
    /// (Phase 2c), written out: `seqs` one-token sequences whose block tables need `blocks`
    /// entries, rows all reduced at `top_n` candidates (0: none reduced), tokens fed from
    /// `feed_word` of the previous launch.
    fn main_key(
        seqs: u32,
        top_n: u8,
        blocks: u32,
        max_blocks: u32,
        feed_word: Option<u64>,
    ) -> GraphKey {
        GraphKey {
            seqs,
            reduce_top_n: top_n,
            table_width: blocks.max(1).next_power_of_two().min(max_blocks.max(1)),
            feed_word,
        }
    }

    /// The decode graph key of batches of 1, 5 and 16 one-token sequences whose block tables
    /// have 3, 7 and 20 entries, with full rows, with greedy reduced rows, and with every token
    /// fed from the previous launch's device choices, is main's formula. Breaks if the
    /// skeleton changes a decode-graph key (replays would then run another shape's graph, or
    /// capture counts would change).
    #[test]
    fn graph_key_matches_main_formula() {
        let tmp = TempDir::new("decoder-graph-key");
        let spec = write_tiny_llama(tmp.path(), 7);
        let cfg = &spec.config;
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
        let weights = WeightLoader::load(
            &index,
            &cfg.family.0.weight_slots(cfg),
            &mem,
            MAX_STAGING_BYTES,
        )
        .expect("load");
        let provider = cpu_reference_provider();
        let order = [provider.id()];
        let opts = ExecutorOptions::default();
        let mut reqs = super::super::requirements(cfg, BLOCK_TOKENS, opts);
        reqs.push(logits::reduce_requirement(cfg));
        let registry = KernelRegistry::build(
            vec![provider],
            &order,
            &reqs,
            &KernelMetrics::register(&MetricsRegistry::new()),
            None,
        )
        .expect("every op has a provider");
        let limits = ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: 64,
            max_seqs: MAX_SEQS,
        };
        let mut exec = DecoderExecutor::new(
            cfg,
            families::llama::decoder_spec(),
            weights,
            Arc::new(registry),
            Arc::clone(&mem),
            limits,
            opts,
        )
        .expect("executor");
        assert!(exec.reduces_logits(), "the CPU provider reduces rows");
        let max_blocks = exec.limits.max_blocks_per_seq();
        let layout = *exec.kv_layout();
        let blocks = MAX_SEQS * TABLE;
        let storage =
            DeviceBuffer::alloc(&mem, (layout.block_bytes() * u64::from(blocks)) as usize)
                .expect("pool");
        let kv = KvPoolView {
            storage: &storage,
            layout,
            num_blocks: blocks,
            layer_stride_bytes: layout.block_bytes() / u64::from(layout.num_layers)
                * u64::from(blocks),
        };
        let greedy = RowReduce {
            top_n: 1,
            temperature: 0.0,
            uniform: None,
            top_p: 1.0,
        };
        let top4 = RowReduce { top_n: 4, ..greedy };

        for n in [1u32, 5, 16] {
            for entries in [3u32, 7, 20] {
                let tables: Vec<Vec<BlockId>> = (0..n)
                    .map(|s| (0..entries).map(|b| BlockId(s * TABLE + b)).collect())
                    .collect();
                let kv_len = (entries - 1) * BLOCK_TOKENS + 1;
                let tokens: Vec<u32> = (0..n).map(|s| 3 + s).collect();
                let positions = vec![kv_len - 1; n as usize];
                let seqs = |reduce: Option<RowReduce>| -> Vec<SeqSlice<'_>> {
                    tables
                        .iter()
                        .enumerate()
                        .map(|(s, table)| SeqSlice {
                            seq: SeqId(s as u64 + 1),
                            q_start: s as u32,
                            q_len: 1,
                            kv_len,
                            block_table: table,
                            reduce,
                        })
                        .collect()
                };
                let case = format!("{n} seqs, {entries} blocks");
                let (full, top4_rows, greedy_rows) =
                    (seqs(None), seqs(Some(top4)), seqs(Some(greedy)));
                let key = |exec: &DecoderExecutor, seqs: &[SeqSlice<'_>], feeds: &[TokenFeed]| {
                    let batch = BatchInput {
                        tokens: &tokens,
                        positions: &positions,
                        seqs,
                        kv: &kv,
                    };
                    exec.graph_key(&batch, feeds).expect("valid batch")
                };
                assert_eq!(
                    key(&exec, &full, &[]),
                    Some(main_key(n, 0, entries, max_blocks, None)),
                    "{case}, full rows"
                );
                assert_eq!(
                    key(&exec, &top4_rows, &[]),
                    Some(main_key(n, 4, entries, max_blocks, None)),
                    "{case}, reduced rows"
                );
                if n > 1 {
                    // A batch mixing reduced and full rows runs eagerly.
                    let mut mixed = seqs(None);
                    mixed[0].reduce = Some(greedy);
                    assert_eq!(key(&exec, &mixed, &[]), None, "{case}, mixed rows");
                }

                // Every token fed from the previous launch's greedy choices: one run.
                let previous = BatchInput {
                    tokens: &tokens,
                    positions: &positions,
                    seqs: &greedy_rows,
                    kv: &kv,
                };
                exec.launch(&previous, &[]).expect("previous launch");
                exec.collect().expect("collect");
                let word = exec
                    .head
                    .feed_word(0)
                    .expect("greedy rows have a device choice");
                let feeds: Vec<TokenFeed> = (0..n)
                    .map(|s| TokenFeed {
                        token: s,
                        prev_slot: s,
                    })
                    .collect();
                assert_eq!(
                    key(&exec, &greedy_rows, &feeds),
                    Some(main_key(n, 1, entries, max_blocks, Some(word as u64))),
                    "{case}, fed"
                );
                if n > 1 {
                    // Feeding only some tokens is not one run over every token: eager.
                    assert_eq!(
                        key(&exec, &greedy_rows, &feeds[1..]),
                        None,
                        "{case}, partly fed"
                    );
                }
            }
        }

        // A prefill is never a decode graph.
        let table: Vec<BlockId> = (0..3).map(BlockId).collect();
        let seqs = [SeqSlice {
            seq: SeqId(1),
            q_start: 0,
            q_len: 2,
            kv_len: 2,
            block_table: &table,
            reduce: None,
        }];
        let prefill = BatchInput {
            tokens: &[1, 2],
            positions: &[0, 1],
            seqs: &seqs,
            kv: &kv,
        };
        assert_eq!(exec.graph_key(&prefill, &[]).expect("prefill"), None);
    }
}
