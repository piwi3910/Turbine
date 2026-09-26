//! Llama forward pass over a ragged batch of sequences (P1 S-8, P2 S-5/S-9) on the kernel
//! registry: embedding → per layer (RMSNorm → Q/K/V projections → Llama-3 RoPE on Q and the new
//! K rows → paged causal GQA attention, which appends the new K/V rows into each sequence's pool
//! blocks → O projection → residual add → RMSNorm → gate/up → SiLU·up → down → residual add) →
//! final RMSNorm on each sequence's last row → LM head (`embed_tokens` when tied) with FP32
//! output.
//!
//! Fused projections (Phase 2c, [`ExecutorOptions::fused_projections`]): the loader lays each layer's
//! Q/K/V and gate/up weights out as one `[q; k; v]` and one `[gate; up]` matrix, so one GEMM
//! computes all three (two) projections into one `[tokens, q + 2·kv]` (`[tokens, 2·inter]`)
//! buffer, and RoPE, attention and SiLU·up read their operands as row-strided column blocks of
//! it. Unfused, one GEMM per projection runs over row views of the same weights into dense
//! regions of the same buffer: the Phase 2 op sequence, bit for bit (every output element is the
//! same dot product either way).
//!
//! Every token of every sequence goes through the GEMMs as one `[total_tokens, hidden]` batch;
//! only attention looks at sequence boundaries (`q_indptr`, `kv_lens`, block tables). The KV
//! lives in the caller's pool (`KvPoolView`); activations live in buffers allocated once for
//! `max_batch_tokens` tokens and `max_seqs` sequences. Every kernel lookup uses a config
//! `requirements` lists, so the registry built from it at startup serves every call.
//!
//! Diagnostics: [`LlamaExecutor::set_trace`] makes each forward record every intermediate
//! tensor ([`TraceTensor`]) with a blocking device read after the op that wrote it; comparing
//! two providers' traces locates the first op where their numerics part. Off by default and
//! free when off (one branch per op).
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use turbine_core::types::{BlockId, DType, KvLayout, ModelShape};
use turbine_kernels::{
    ActivationConfig, ActivationContext, AttentionConfig, AttentionKind, ElementwiseConfig,
    ElementwiseContext, EmbeddingConfig, EmbeddingContext, GemmConfig, GemmContext, KernelError,
    KernelRegistry, NormConfig, NormContext, OpConfig, OpRequirement, PagedAttentionContext,
    RopeConfig, RopeContext,
};
use turbine_tensor::{DeviceMemory, KvPoolView, Tensor, TensorView};

use super::batch::{self, BatchLimits, DeviceBatch, HostBatch, Packed};
use super::{BatchInput, ExecutorOptions, ForwardTimings, Logits, ModelExecutor, Split, rope};
use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::loader::{LM_HEAD, LoadedWeights, gate_up_proj_name, qkv_proj_name};

/// Weights, activations and KV are BF16; logits are F32.
pub(super) const ACT: DType = DType::BF16;

/// Model dimensions in elements.
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    inter: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Dims {
    fn of(cfg: &ModelArchConfig) -> Dims {
        let head_dim = cfg.head_dim as usize;
        Dims {
            hidden: cfg.hidden as usize,
            q_dim: cfg.num_attention_heads as usize * head_dim,
            kv_dim: cfg.num_kv_heads as usize * head_dim,
            inter: cfg.intermediate as usize,
            vocab: cfg.vocab_size as usize,
            heads: cfg.num_attention_heads as usize,
            kv_heads: cfg.num_kv_heads as usize,
            head_dim,
        }
    }
}

/// `c = a · wᵀ` for an HF Linear weight `w` of `[n, k]`.
pub(super) fn gemm_cfg(n: usize, k: usize, c_dtype: DType) -> GemmConfig {
    GemmConfig {
        n: n as u64,
        k: k as u64,
        trans_b: true,
        a_dtype: ACT,
        b_dtype: ACT,
        c_dtype,
    }
}

/// Paged causal GQA attention over blocks of `block_tokens` tokens.
pub(super) fn attention_cfg(
    cfg: &ModelArchConfig,
    kind: AttentionKind,
    block_tokens: u32,
) -> AttentionConfig {
    AttentionConfig {
        kind,
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        dtype: ACT,
        block_tokens: Some(block_tokens),
        causal: true,
    }
}

pub(super) fn rope_cfg(cfg: &ModelArchConfig) -> RopeConfig {
    RopeConfig {
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        rotary_dim: cfg.head_dim,
        dtype: ACT,
    }
}

fn norm_cfg(d: &Dims) -> NormConfig {
    NormConfig {
        dim: d.hidden as u64,
        dtype: ACT,
    }
}

fn embedding_cfg(d: &Dims) -> EmbeddingConfig {
    EmbeddingConfig {
        hidden: d.hidden as u64,
        vocab_rows: d.vocab as u64,
        dtype: ACT,
    }
}

fn activation_cfg(d: &Dims) -> ActivationConfig {
    ActivationConfig {
        cols: d.inter as u64,
        dtype: ACT,
    }
}

pub(super) const ADD_CFG: ElementwiseConfig = ElementwiseConfig { dtype: ACT };

pub(super) fn invalid(message: String) -> ModelError {
    ModelError::Kernel(KernelError::InvalidArgument { message })
}

/// One intermediate tensor of a traced forward ([`LlamaExecutor::set_trace`]), widened to
/// f32 from its stored dtype (BF16 activations, F32 logits).
#[derive(Clone, Debug, PartialEq)]
pub struct TraceTensor {
    /// Decoder layer; `None` for `embed`, `final_norm` and `logits`.
    pub layer: Option<usize>,
    /// The op output: `embed`; per layer `attn_norm`, `q`, `k`, `v` (projections of the new
    /// rows), `q_rope`, `k_rope`, `attn`, `o_proj`, `resid_attn`, `mlp_norm`, `gate`, `up`,
    /// `act`, `down`, `resid_mlp`; then `final_norm` (each sequence's last row) and `logits`.
    pub name: &'static str,
    /// `[rows, cols]`: rows are the forward's tokens (one per sequence for `final_norm` and
    /// `logits`).
    pub shape: [usize; 2],
    pub data: Vec<f32>,
}

/// Decodes a BF16 or F32 view read from the device into f32 values, row-major (a row-strided
/// view's rows are gathered; its other dimensions must be dense).
fn read_f32(view: &TensorView<'_>) -> Result<Vec<f32>, ModelError> {
    let raw = view.slice.read_bytes()?;
    let es = view.dtype.size_bytes();
    let rows = view.shape.first().copied().unwrap_or(1);
    let row_bytes = view.numel() / rows.max(1) * es;
    let stride_bytes = view.strides.first().map_or(row_bytes, |s| s * es);
    let bytes: Vec<u8> = if stride_bytes == row_bytes {
        raw
    } else {
        (0..rows)
            .flat_map(|r| &raw[r * stride_bytes..r * stride_bytes + row_bytes])
            .copied()
            .collect()
    };
    Ok(match view.dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => return Err(invalid(format!("trace of a {other:?} tensor"))),
    })
}

struct Layer {
    input_norm: Tensor,
    /// `[q_dim + 2·kv_dim, hidden]`: Q rows, then K, then V.
    w_qkv: Tensor,
    wo: Tensor,
    post_norm: Tensor,
    /// `[2·inter, hidden]`: gate rows, then up.
    w_gate_up: Tensor,
    w_down: Tensor,
}

/// Parameter `name` taken from `weights`, checked to be the `shape` BF16 matrix.
pub(super) fn take_matrix(
    weights: &mut LoadedWeights,
    name: &str,
    shape: [usize; 2],
) -> Result<Tensor, ModelError> {
    let w = weights.take(name)?;
    if w.shape.as_slice() != shape || w.dtype != ACT {
        return Err(invalid(format!(
            "{name} is {:?} {}, expected {shape:?} {}",
            w.shape.as_slice(),
            w.dtype.as_str(),
            ACT.as_str()
        )));
    }
    Ok(w)
}

/// Activation buffers, allocated once: `[max_batch_tokens, cols]` per token row, `[max_seqs,
/// cols]` per sequence row.
struct Buffers {
    inv_freq: Tensor,
    /// Residual stream.
    x: Tensor,
    /// Normalised input of the attention and MLP blocks.
    h: Tensor,
    /// Q, K and V projections of the new rows (Q and K rotated in place; K and V appended by
    /// attention), laid out by `LlamaExecutor::qkv`.
    qkv: Tensor,
    attn: Tensor,
    /// O and down projection outputs before the residual add.
    proj: Tensor,
    /// Gate and up projections, laid out by `LlamaExecutor::gate_up`.
    gate_up: Tensor,
    act: Tensor,
    /// Final-norm output of each sequence's last row.
    last: Tensor,
    logits: Tensor,
}

/// The Llama executor: ragged batches of up to `max_seqs` sequences and `max_batch_tokens`
/// tokens over the paged KV pool, on one device.
pub struct LlamaExecutor {
    cfg: ModelArchConfig,
    dims: Dims,
    shape: ModelShape,
    kv_layout: KvLayout,
    limits: BatchLimits,
    registry: Arc<KernelRegistry>,
    /// Q/K/V (`[q_dim, kv_dim, kv_dim]`) and gate/up (`[inter, inter]`) in their buffers, fused
    /// or not per [`ExecutorOptions::fused_projections`].
    qkv: Split<3>,
    gate_up: Split<2>,
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    /// `None` when the model ties its LM head to `embed`.
    lm_head: Option<Tensor>,
    bufs: Buffers,
    meta: DeviceBatch,
    /// Host bytes of the last upload; they stay valid until the next synchronization, which the
    /// logits read performs (contract §9.2).
    host: HostBatch,
    /// Timings of the last forward ([`ModelExecutor::last_timings`]).
    timings: ForwardTimings,
    /// `Some` while tracing: the recorded tensors not yet taken.
    trace: RefCell<Option<Vec<TraceTensor>>>,
}

pub(super) fn limits(
    cfg: &ModelArchConfig,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
) -> BatchLimits {
    BatchLimits {
        vocab: cfg.vocab_size,
        max_batch_tokens,
        max_seqs,
        max_positions: cfg.max_position_embeddings,
        layout: cfg.kv_layout(block_tokens),
    }
}

impl LlamaExecutor {
    /// Every distinct op config the forward pass executes with `opts`, in first-use order, plus
    /// the `copy_blocks` fork over KV blocks of `block_tokens` tokens. The registry is built from
    /// this list at startup, so a config no provider supports fails before any weight is read.
    pub fn requirements(
        cfg: &ModelArchConfig,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        let d = Dims::of(cfg);
        let mut specs = vec![
            OpConfig::Embedding(embedding_cfg(&d)),
            OpConfig::Rmsnorm(norm_cfg(&d)),
        ];
        if opts.fused_projections {
            specs.push(OpConfig::Gemm(gemm_cfg(
                d.q_dim + 2 * d.kv_dim,
                d.hidden,
                ACT,
            )));
        } else {
            specs.extend([
                OpConfig::Gemm(gemm_cfg(d.q_dim, d.hidden, ACT)),
                OpConfig::Gemm(gemm_cfg(d.kv_dim, d.hidden, ACT)),
            ]);
        }
        specs.extend([
            OpConfig::Rope(rope_cfg(cfg)),
            OpConfig::Attention(attention_cfg(
                cfg,
                AttentionKind::PrefillPaged,
                block_tokens,
            )),
            OpConfig::Attention(attention_cfg(cfg, AttentionKind::DecodePaged, block_tokens)),
            OpConfig::Gemm(gemm_cfg(d.hidden, d.q_dim, ACT)),
            OpConfig::Add(ADD_CFG),
            OpConfig::Gemm(gemm_cfg(
                if opts.fused_projections {
                    2 * d.inter
                } else {
                    d.inter
                },
                d.hidden,
                ACT,
            )),
            OpConfig::SiluMul(activation_cfg(&d)),
            OpConfig::Gemm(gemm_cfg(d.hidden, d.inter, ACT)),
            OpConfig::Gemm(gemm_cfg(d.vocab, d.hidden, DType::F32)),
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

    /// Device bytes of the executor's buffers (the budget's workspace term): per token the
    /// I32 id and position and the BF16 rows of `x`, `h`, `proj` (hidden), `q`, `attn`
    /// (heads · head_dim), `k`, `v` (kv_heads · head_dim) and `gate`, `up`, `act`
    /// (intermediate); per sequence its last normalised row (BF16), its F32 logits row, its
    /// `q_indptr` and `kv_lens` entries and a block table for `max_position_embeddings` tokens
    /// (I32); plus one `q_indptr` entry and the F32 `inv_freq`.
    pub fn workspace_bytes(
        cfg: &ModelArchConfig,
        block_tokens: u32,
        max_batch_tokens: u32,
        max_seqs: u32,
    ) -> u64 {
        let d = Dims::of(cfg);
        let bf16 = ACT.size_bytes() as u64;
        let f32 = DType::F32.size_bytes() as u64;
        let per_token = bf16 * (3 * d.hidden + 2 * d.q_dim + 2 * d.kv_dim + 3 * d.inter) as u64;
        let per_seq = bf16 * d.hidden as u64 + f32 * d.vocab as u64;
        let limits = limits(cfg, block_tokens, max_batch_tokens, max_seqs);
        u64::from(max_batch_tokens) * per_token
            + u64::from(max_seqs) * per_seq
            + f32 * (d.head_dim / 2) as u64
            + DeviceBatch::bytes(&limits)
    }

    /// Takes the parameters out of `weights` (Q/K/V and gate/up fused by the loader), allocates
    /// the activation and batch buffers for `max_batch_tokens` tokens of up to `max_seqs`
    /// sequences on `mem`, and uploads the rotary inverse frequencies. The KV is not the
    /// executor's: every forward names its pool, laid out as `cfg.kv_layout(block_tokens)`.
    /// `registry` must have been built from [`LlamaExecutor::requirements`] of `cfg`,
    /// `block_tokens` and `opts`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &ModelArchConfig,
        mut weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        block_tokens: u32,
        max_batch_tokens: u32,
        max_seqs: u32,
        opts: ExecutorOptions,
    ) -> Result<LlamaExecutor, ModelError> {
        if block_tokens == 0 || max_batch_tokens == 0 || max_seqs == 0 {
            return Err(invalid(format!(
                "block_tokens {block_tokens}, max_batch_tokens {max_batch_tokens} and max_seqs \
                 {max_seqs} must be positive"
            )));
        }
        let d = Dims::of(cfg);
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        let qkv = Split {
            cols: [d.q_dim, d.kv_dim, d.kv_dim],
            fused: opts.fused_projections,
        };
        let gate_up = Split {
            cols: [d.inter, d.inter],
            fused: opts.fused_projections,
        };
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let w_qkv = take_matrix(&mut weights, &qkv_proj_name(i), [qkv.width(), d.hidden])?;
            let w_gate_up = take_matrix(
                &mut weights,
                &gate_up_proj_name(i),
                [gate_up.width(), d.hidden],
            )?;
            let mut take = |s: &str| weights.take(&format!("{p}.{s}.weight"));
            layers.push(Layer {
                input_norm: take("input_layernorm")?,
                w_qkv,
                wo: take("self_attn.o_proj")?,
                post_norm: take("post_attention_layernorm")?,
                w_gate_up,
                w_down: take("mlp.down_proj")?,
            });
        }
        let embed = weights.take("model.embed_tokens.weight")?;
        let final_norm = weights.take("model.norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(weights.take(LM_HEAD)?)
        };

        let limits = limits(cfg, block_tokens, max_batch_tokens, max_seqs);
        let t = max_batch_tokens as usize;
        let n = max_seqs as usize;
        let act = |cols: usize| Tensor::empty(&mem, &[t, cols], ACT);
        let mut inv_freq = Tensor::empty(&mem, &[d.head_dim / 2], DType::F32)?;
        let freqs = rope::inv_freq(cfg.rope_theta, cfg.head_dim, cfg.rope_scaling.as_ref());
        let freq_bytes: Vec<u8> = freqs.iter().flat_map(|f| f.to_le_bytes()).collect();
        inv_freq.storage.copy_from_host(0, &freq_bytes)?;
        mem.synchronize()?;
        let bufs = Buffers {
            inv_freq,
            x: act(d.hidden)?,
            h: act(d.hidden)?,
            qkv: act(qkv.width())?,
            attn: act(d.q_dim)?,
            proj: act(d.hidden)?,
            gate_up: act(gate_up.width())?,
            act: act(d.inter)?,
            last: Tensor::empty(&mem, &[n, d.hidden], ACT)?,
            logits: Tensor::empty(&mem, &[n, d.vocab], DType::F32)?,
        };
        let meta = DeviceBatch::alloc(&mem, &limits)?;
        Ok(LlamaExecutor {
            cfg: cfg.clone(),
            dims: d,
            shape: cfg.shape(),
            kv_layout: limits.layout,
            limits,
            registry,
            qkv,
            gate_up,
            embed,
            layers,
            final_norm,
            lm_head,
            bufs,
            meta,
            host: HostBatch::default(),
            timings: ForwardTimings::default(),
            trace: RefCell::new(None),
        })
    }

    /// Diagnostics: while enabled every forward records its intermediate tensors (one blocking
    /// device read per op), returned by [`LlamaExecutor::take_trace`]. Disabling drops them.
    pub fn set_trace(&mut self, enabled: bool) {
        *self.trace.get_mut() = enabled.then(Vec::new);
    }

    /// The tensors recorded since tracing was enabled or last taken, in execution order; empty
    /// when tracing is off.
    pub fn take_trace(&mut self) -> Vec<TraceTensor> {
        self.trace
            .get_mut()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Records `view` (`[rows, cols]` contiguous) under `name` when tracing.
    fn record(
        &self,
        layer: Option<usize>,
        name: &'static str,
        view: TensorView<'_>,
    ) -> Result<(), ModelError> {
        if self.trace.borrow().is_none() {
            return Ok(());
        }
        let rows = view.shape[0];
        let cols = view.numel() / rows.max(1);
        let data = read_f32(&view)?;
        if let Some(trace) = self.trace.borrow_mut().as_mut() {
            trace.push(TraceTensor {
                layer,
                name,
                shape: [rows, cols],
                data,
            });
        }
        Ok(())
    }

    /// Rows `[0, t)` of an activation buffer as `[t, cols]`.
    fn rows(buf: &Tensor, t: usize) -> TensorView<'_> {
        buf.view().rows(0, t)
    }

    /// Rows `[0, t)` of an activation buffer as `[t, heads, head_dim]`.
    fn heads(buf: &Tensor, t: usize, heads: usize, head_dim: usize) -> TensorView<'_> {
        TensorView::contiguous(buf.storage.whole(), 0, &[t, heads, head_dim], ACT)
    }

    /// `c = a · wᵀ` for a `[n, k]` weight view `w`.
    fn linear(
        &self,
        a: TensorView<'_>,
        w: TensorView<'_>,
        c: TensorView<'_>,
    ) -> Result<(), ModelError> {
        let cfg = gemm_cfg(w.shape[0], w.shape[1], c.dtype);
        self.registry.gemm(&cfg).execute(&mut GemmContext {
            a,
            b: w,
            c,
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
        })?;
        Ok(())
    }

    fn rmsnorm(
        &self,
        x: TensorView<'_>,
        w: &Tensor,
        out: TensorView<'_>,
    ) -> Result<(), ModelError> {
        self.registry
            .norm(&norm_cfg(&self.dims))
            .execute(&mut NormContext {
                x,
                weight: w.view(),
                out,
                eps: self.cfg.rms_norm_eps,
            })?;
        Ok(())
    }

    /// `x[0..t] += proj[0..t]` (the residual add).
    fn residual_add(&self, t: usize) -> Result<(), ModelError> {
        let x = Self::rows(&self.bufs.x, t);
        self.registry
            .elementwise(&ADD_CFG)
            .execute(&mut ElementwiseContext {
                a: x.clone(),
                b: Self::rows(&self.bufs.proj, t),
                out: x,
            })?;
        Ok(())
    }

    /// One decoder layer on the batch's `p.total_q` rows; attention appends to and reads
    /// layer `i` of `kv`.
    fn layer(&self, i: usize, p: &Packed, kv: &KvPoolView<'_>) -> Result<(), ModelError> {
        let l = &self.layers[i];
        let b = &self.bufs;
        let d = &self.dims;
        let t = p.total_q;

        let li = Some(i);
        let (q, k, v) = (
            |t| self.qkv.part(&b.qkv, 0, t, &[d.q_dim]),
            |t| self.qkv.part(&b.qkv, 1, t, &[d.kv_dim]),
            |t| self.qkv.part(&b.qkv, 2, t, &[d.kv_dim]),
        );
        // Attention block.
        self.rmsnorm(Self::rows(&b.x, t), &l.input_norm, Self::rows(&b.h, t))?;
        self.record(li, "attn_norm", Self::rows(&b.h, t))?;
        if self.qkv.fused {
            let w = l.w_qkv.view();
            self.linear(Self::rows(&b.h, t), w, self.qkv.whole(&b.qkv, t))?;
        } else {
            let w = |r, n| l.w_qkv.view().rows(r, n);
            let h = || Self::rows(&b.h, t);
            self.linear(h(), w(0, d.q_dim), q(t))?;
            self.linear(h(), w(d.q_dim, d.kv_dim), k(t))?;
            self.linear(h(), w(d.q_dim + d.kv_dim, d.kv_dim), v(t))?;
        }
        self.record(li, "q", q(t))?;
        self.record(li, "k", k(t))?;
        self.record(li, "v", v(t))?;
        let q_heads = |t| self.qkv.part(&b.qkv, 0, t, &[d.heads, d.head_dim]);
        let k_heads = |t| self.qkv.part(&b.qkv, 1, t, &[d.kv_heads, d.head_dim]);
        let v_heads = |t| self.qkv.part(&b.qkv, 2, t, &[d.kv_heads, d.head_dim]);
        let rope = rope_cfg(&self.cfg);
        self.registry.rope(&rope).execute(&mut RopeContext {
            cfg: rope,
            q: q_heads(t),
            k: k_heads(t),
            positions: self.meta.positions_view(p),
            inv_freq: b.inv_freq.view(),
        })?;
        self.record(li, "q_rope", q(t))?;
        self.record(li, "k_rope", k(t))?;
        let kind = if p.is_decode() {
            AttentionKind::DecodePaged
        } else {
            AttentionKind::PrefillPaged
        };
        let attn = attention_cfg(&self.cfg, kind, self.kv_layout.block_tokens);
        self.registry
            .attention(&attn)
            .execute_paged(&mut PagedAttentionContext {
                cfg: attn,
                q: q_heads(t),
                k_new: k_heads(t),
                v_new: v_heads(t),
                out: Self::heads(&b.attn, t, d.heads, d.head_dim),
                kv_layer: batch::kv_layer(kv, i),
                block_table: self.meta.block_table_view(p),
                q_indptr: self.meta.q_indptr_view(p),
                kv_lens: self.meta.kv_lens_view(p),
                max_q_len: p.max_q_len,
                max_kv_len: p.max_kv_len,
                max_blocks_per_seq: p.max_blocks_per_seq,
                scale: 1.0 / (d.head_dim as f32).sqrt(),
            })?;
        self.record(li, "attn", Self::rows(&b.attn, t))?;
        self.linear(Self::rows(&b.attn, t), l.wo.view(), Self::rows(&b.proj, t))?;
        self.record(li, "o_proj", Self::rows(&b.proj, t))?;
        self.residual_add(t)?;
        self.record(li, "resid_attn", Self::rows(&b.x, t))?;

        // MLP block.
        self.rmsnorm(Self::rows(&b.x, t), &l.post_norm, Self::rows(&b.h, t))?;
        self.record(li, "mlp_norm", Self::rows(&b.h, t))?;
        let gate = |t| self.gate_up.part(&b.gate_up, 0, t, &[d.inter]);
        let up = |t| self.gate_up.part(&b.gate_up, 1, t, &[d.inter]);
        if self.gate_up.fused {
            let w = l.w_gate_up.view();
            self.linear(Self::rows(&b.h, t), w, self.gate_up.whole(&b.gate_up, t))?;
        } else {
            let w = |r| l.w_gate_up.view().rows(r, d.inter);
            self.linear(Self::rows(&b.h, t), w(0), gate(t))?;
            self.linear(Self::rows(&b.h, t), w(d.inter), up(t))?;
        }
        self.record(li, "gate", gate(t))?;
        self.record(li, "up", up(t))?;
        self.registry
            .activation(&activation_cfg(d))
            .execute(&mut ActivationContext {
                gate: gate(t),
                up: up(t),
                out: Self::rows(&b.act, t),
            })?;
        self.record(li, "act", Self::rows(&b.act, t))?;
        self.linear(
            Self::rows(&b.act, t),
            l.w_down.view(),
            Self::rows(&b.proj, t),
        )?;
        self.record(li, "down", Self::rows(&b.proj, t))?;
        self.residual_add(t)?;
        self.record(li, "resid_mlp", Self::rows(&b.x, t))
    }

    /// Final RMSNorm of each sequence's last row into `last[0..n]`: one call per run of
    /// consecutive last rows (a decode-only batch is one call).
    fn final_norm_last_rows(&self, p: &Packed) -> Result<(), ModelError> {
        let b = &self.bufs;
        let x = Self::rows(&b.x, p.total_q);
        let mut s = 0;
        while s < p.num_seqs {
            let mut len = 1;
            while s + len < p.num_seqs && p.last_rows[s + len] == p.last_rows[s] + len {
                len += 1;
            }
            self.rmsnorm(
                x.rows(p.last_rows[s], len),
                &self.final_norm,
                b.last.view().rows(s, len),
            )?;
            s += len;
        }
        Ok(())
    }
}

impl ModelExecutor for LlamaExecutor {
    fn shape(&self) -> &ModelShape {
        &self.shape
    }

    fn kv_layout(&self) -> &KvLayout {
        &self.kv_layout
    }

    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
        let p = self.host.pack(batch, &self.limits)?;
        self.meta.upload(&self.host)?;
        let launch_started = Instant::now();

        let b = &self.bufs;
        let d = &self.dims;
        let (t, n) = (p.total_q, p.num_seqs);
        self.registry
            .embedding(&embedding_cfg(d))
            .execute(&mut EmbeddingContext {
                ids: self.meta.ids_view(&p),
                table: self.embed.view(),
                out: Self::rows(&b.x, t),
                vocab_offset: 0,
            })?;
        self.record(None, "embed", Self::rows(&b.x, t))?;
        for i in 0..self.layers.len() {
            self.layer(i, &p, batch.kv)?;
        }
        self.final_norm_last_rows(&p)?;
        self.record(None, "final_norm", Self::rows(&b.last, n))?;
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        self.linear(
            Self::rows(&b.last, n),
            head.view(),
            Self::rows(&b.logits, n),
        )?;
        self.record(None, "logits", Self::rows(&b.logits, n))?;
        let wait_started = Instant::now();
        // The iteration's one device-to-host copy (it synchronizes the stream).
        let raw = b
            .logits
            .storage
            .slice(0, n * d.vocab * DType::F32.size_bytes())
            .read_bytes()?;
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.timings = ForwardTimings {
            launch: wait_started - launch_started,
            device_wait: wait_started.elapsed(),
        };
        Ok(Logits {
            rows: n,
            vocab: d.vocab,
            data,
        })
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
