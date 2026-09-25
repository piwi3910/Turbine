//! Llama forward pass for one sequence (P1 S-8) over the kernel registry: embedding → per layer
//! (RMSNorm → Q/K/V projections → Llama-3 RoPE on Q and the new K rows → causal GQA attention →
//! O projection → residual add → RMSNorm → gate/up → SiLU·up → down → residual add) → final
//! RMSNorm on the last row → LM head (`embed_tokens` when tied) with FP32 output.
//!
//! K and V projections are written straight into the contiguous per-request cache
//! `[layers, 2, max_seq_len, kv_heads, head_dim]` BF16; activations live in buffers allocated
//! once for `max_forward_tokens` tokens. Every kernel lookup uses a config `requirements`
//! lists, so the registry built from it at startup serves every call.
use std::sync::Arc;

use turbine_core::types::{DType, KvLayout, ModelShape};
use turbine_kernels::{
    ActivationConfig, ActivationContext, AttentionConfig, AttentionContext, AttentionKind,
    ElementwiseConfig, ElementwiseContext, EmbeddingConfig, EmbeddingContext, GemmConfig,
    GemmContext, KernelError, KernelRegistry, NormConfig, NormContext, OpConfig, OpRequirement,
    RopeConfig, RopeContext,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor, TensorView};

use super::{BatchInput, Logits, ModelExecutor, rope};
use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::loader::{LM_HEAD, LoadedWeights};

/// Weights, activations and KV are BF16 in Phase 1; logits are F32.
const ACT: DType = DType::BF16;

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
fn gemm_cfg(n: usize, k: usize, c_dtype: DType) -> GemmConfig {
    GemmConfig {
        n: n as u64,
        k: k as u64,
        trans_b: true,
        a_dtype: ACT,
        b_dtype: ACT,
        c_dtype,
    }
}

fn attention_cfg(cfg: &ModelArchConfig, kind: AttentionKind) -> AttentionConfig {
    AttentionConfig {
        kind,
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        dtype: ACT,
        block_tokens: None,
        causal: true,
    }
}

fn rope_cfg(cfg: &ModelArchConfig) -> RopeConfig {
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

const ADD_CFG: ElementwiseConfig = ElementwiseConfig { dtype: ACT };

fn invalid(message: String) -> ModelError {
    ModelError::Kernel(KernelError::InvalidArgument { message })
}

/// Little-endian I32 bytes of `values` (token ids and positions, which fit: both are below the
/// vocabulary size and `max_seq_len`, checked before the conversion).
fn i32_bytes(values: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.extend(values.iter().flat_map(|&v| (v as i32).to_le_bytes()));
}

struct Layer {
    input_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    post_norm: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

/// Activation buffers, allocated once for `max_forward_tokens` tokens (`[t, cols]` each).
struct Buffers {
    ids: DeviceBuffer,
    positions: DeviceBuffer,
    inv_freq: Tensor,
    /// Residual stream.
    x: Tensor,
    /// Normalised input of the attention and MLP blocks.
    h: Tensor,
    q: Tensor,
    attn: Tensor,
    /// O and down projection outputs before the residual add.
    proj: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
    /// Final-norm output of the last row.
    last: Tensor,
    logits: Tensor,
}

/// The Phase 1 Llama executor: one sequence at a time on one device.
pub struct LlamaExecutor {
    cfg: ModelArchConfig,
    dims: Dims,
    shape: ModelShape,
    kv_layout: KvLayout,
    registry: Arc<KernelRegistry>,
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    /// `None` when the model ties its LM head to `embed`.
    lm_head: Option<Tensor>,
    kv: DeviceBuffer,
    bufs: Buffers,
    /// Host bytes of the last ids/positions upload; they stay valid until the next
    /// synchronization, which the logits read performs (contract §9.2).
    host_ids: Vec<u8>,
    host_positions: Vec<u8>,
    max_seq_len: u32,
    max_forward_tokens: u32,
    /// Positions `[0, cached_len)` of the current sequence hold valid K/V.
    cached_len: u32,
}

impl LlamaExecutor {
    /// Every distinct op config the forward pass executes, in first-use order. The registry is
    /// built from this list at startup, so a config no provider supports fails before any
    /// weight is read.
    pub fn requirements(cfg: &ModelArchConfig) -> Vec<OpRequirement> {
        let d = Dims::of(cfg);
        let specs = [
            OpConfig::Embedding(embedding_cfg(&d)),
            OpConfig::Rmsnorm(norm_cfg(&d)),
            OpConfig::Gemm(gemm_cfg(d.q_dim, d.hidden, ACT)),
            OpConfig::Gemm(gemm_cfg(d.kv_dim, d.hidden, ACT)),
            OpConfig::Rope(rope_cfg(cfg)),
            OpConfig::Attention(attention_cfg(cfg, AttentionKind::Prefill)),
            OpConfig::Attention(attention_cfg(cfg, AttentionKind::Decode)),
            OpConfig::Gemm(gemm_cfg(d.hidden, d.q_dim, ACT)),
            OpConfig::Add(ADD_CFG),
            OpConfig::Gemm(gemm_cfg(d.inter, d.hidden, ACT)),
            OpConfig::SiluMul(activation_cfg(&d)),
            OpConfig::Gemm(gemm_cfg(d.hidden, d.inter, ACT)),
            OpConfig::Gemm(gemm_cfg(d.vocab, d.hidden, DType::F32)),
        ];
        let mut unique: Vec<OpConfig> = Vec::with_capacity(specs.len());
        for spec in specs {
            if !unique.contains(&spec) {
                unique.push(spec);
            }
        }
        unique.into_iter().map(OpRequirement::from).collect()
    }

    /// Bytes of the activation buffers for `max_tokens` tokens per forward (the budget's
    /// workspace term): per token the I32 id and position and the BF16 rows of `x`, `h`, `proj`
    /// (hidden), `q`, `attn` (heads · head_dim) and `gate`, `up`, `act` (intermediate); plus the
    /// last normalised row (BF16), the F32 logits row and the F32 `inv_freq`.
    pub fn workspace_bytes(cfg: &ModelArchConfig, max_tokens: u32) -> u64 {
        let d = Dims::of(cfg);
        let bf16 = ACT.size_bytes() as u64;
        let f32 = DType::F32.size_bytes() as u64;
        let i32 = DType::I32.size_bytes() as u64;
        let per_token = 2 * i32 + bf16 * (3 * d.hidden + 2 * d.q_dim + 3 * d.inter) as u64;
        let fixed = bf16 * d.hidden as u64 + f32 * d.vocab as u64 + f32 * (d.head_dim / 2) as u64;
        u64::from(max_tokens) * per_token + fixed
    }

    /// Takes the parameters out of `weights`, allocates the KV cache for `max_seq_len` tokens
    /// and the activation buffers for `max_forward_tokens` tokens on `mem`, and uploads the
    /// rotary inverse frequencies. `registry` must have been built from
    /// [`LlamaExecutor::requirements`] of `cfg`.
    pub fn new(
        cfg: &ModelArchConfig,
        mut weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        max_seq_len: u32,
        max_forward_tokens: u32,
    ) -> Result<LlamaExecutor, ModelError> {
        if max_seq_len == 0 || max_forward_tokens == 0 {
            return Err(invalid(format!(
                "max_seq_len {max_seq_len} and max_forward_tokens {max_forward_tokens} must be \
                 positive"
            )));
        }
        let d = Dims::of(cfg);
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let mut take = |s: &str| weights.take(&format!("{p}.{s}.weight"));
            layers.push(Layer {
                input_norm: take("input_layernorm")?,
                wq: take("self_attn.q_proj")?,
                wk: take("self_attn.k_proj")?,
                wv: take("self_attn.v_proj")?,
                wo: take("self_attn.o_proj")?,
                post_norm: take("post_attention_layernorm")?,
                w_gate: take("mlp.gate_proj")?,
                w_up: take("mlp.up_proj")?,
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

        // The contiguous cache is one "block" of max_seq_len tokens.
        let kv_layout = cfg.kv_layout(max_seq_len);
        let kv = DeviceBuffer::alloc(&mem, kv_layout.block_bytes() as usize)?;

        let t = max_forward_tokens as usize;
        let act = |cols: usize| Tensor::empty(&mem, &[t, cols], ACT);
        let mut inv_freq = Tensor::empty(&mem, &[d.head_dim / 2], DType::F32)?;
        let freqs = rope::inv_freq(cfg.rope_theta, cfg.head_dim, cfg.rope_scaling.as_ref());
        let freq_bytes: Vec<u8> = freqs.iter().flat_map(|f| f.to_le_bytes()).collect();
        inv_freq.storage.copy_from_host(0, &freq_bytes)?;
        mem.synchronize()?;
        let bufs = Buffers {
            ids: DeviceBuffer::alloc(&mem, t * DType::I32.size_bytes())?,
            positions: DeviceBuffer::alloc(&mem, t * DType::I32.size_bytes())?,
            inv_freq,
            x: act(d.hidden)?,
            h: act(d.hidden)?,
            q: act(d.q_dim)?,
            attn: act(d.q_dim)?,
            proj: act(d.hidden)?,
            gate: act(d.inter)?,
            up: act(d.inter)?,
            act: act(d.inter)?,
            last: Tensor::empty(&mem, &[1, d.hidden], ACT)?,
            logits: Tensor::empty(&mem, &[1, d.vocab], DType::F32)?,
        };
        Ok(LlamaExecutor {
            cfg: cfg.clone(),
            dims: d,
            shape: cfg.shape(),
            kv_layout,
            registry,
            embed,
            layers,
            final_norm,
            lm_head,
            kv,
            bufs,
            host_ids: Vec::with_capacity(t * 4),
            host_positions: Vec::with_capacity(t * 4),
            max_seq_len,
            max_forward_tokens,
            cached_len: 0,
        })
    }

    /// Checks one Phase 1 batch; returns its first position.
    fn validate(&self, batch: &BatchInput<'_>) -> Result<u32, ModelError> {
        let t = batch.tokens.len();
        if t == 0 {
            return Err(invalid("empty batch: no tokens".into()));
        }
        if batch.positions.len() != t {
            return Err(invalid(format!(
                "{t} tokens but {} positions",
                batch.positions.len()
            )));
        }
        if t > self.max_forward_tokens as usize {
            return Err(invalid(format!(
                "{t} tokens exceed max_forward_tokens {}",
                self.max_forward_tokens
            )));
        }
        if let Some(&bad) = batch
            .tokens
            .iter()
            .find(|&&id| id as usize >= self.dims.vocab)
        {
            return Err(invalid(format!(
                "token id {bad} outside the vocabulary of {}",
                self.dims.vocab
            )));
        }
        let p0 = batch.positions[0];
        if let Some(i) =
            (1..t).find(|&i| batch.positions[i] != batch.positions[i - 1].wrapping_add(1))
        {
            return Err(invalid(format!(
                "positions must be consecutive: {} follows {}",
                batch.positions[i],
                batch.positions[i - 1]
            )));
        }
        if p0 > self.cached_len {
            return Err(invalid(format!(
                "batch starts at position {p0} but only {} positions are cached",
                self.cached_len
            )));
        }
        let last = u64::from(p0) + t as u64 - 1;
        if last >= u64::from(self.max_seq_len) {
            return Err(invalid(format!(
                "position {last} is not below max_seq_len {}",
                self.max_seq_len
            )));
        }
        Ok(p0)
    }

    /// Rows `[0, t)` of an activation buffer as `[t, cols]`.
    fn rows(buf: &Tensor, t: usize) -> TensorView<'_> {
        buf.view().rows(0, t)
    }

    /// Rows `[0, t)` of an activation buffer as `[t, heads, head_dim]`.
    fn heads<'a>(&self, buf: &'a Tensor, t: usize) -> TensorView<'a> {
        let d = &self.dims;
        TensorView::contiguous(buf.storage.whole(), 0, &[t, d.heads, d.head_dim], ACT)
    }

    /// K (`which` = 0) or V (1) rows `[start, start + len)` of `layer`'s cache, as
    /// `[len, kv_heads, head_dim]`.
    fn cache(&self, layer: usize, which: usize, start: usize, len: usize) -> TensorView<'_> {
        let d = &self.dims;
        let offset = ((layer * 2 + which) * self.max_seq_len as usize + start) * d.kv_dim;
        TensorView::contiguous(self.kv.whole(), offset, &[len, d.kv_heads, d.head_dim], ACT)
    }

    /// `c = a · wᵀ`.
    fn linear(&self, a: TensorView<'_>, w: &Tensor, c: TensorView<'_>) -> Result<(), ModelError> {
        let cfg = gemm_cfg(w.shape[0], w.shape[1], c.dtype);
        self.registry.gemm(&cfg).execute(&mut GemmContext {
            a,
            b: w.view(),
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

    /// One decoder layer on rows `[0, t)` sitting at positions `p0..p0 + t`.
    fn layer(&self, i: usize, t: usize, p0: usize) -> Result<(), ModelError> {
        let l = &self.layers[i];
        let b = &self.bufs;
        let d = &self.dims;

        // Attention block. K and V land in the cache rows of the new positions.
        self.rmsnorm(Self::rows(&b.x, t), &l.input_norm, Self::rows(&b.h, t))?;
        self.linear(Self::rows(&b.h, t), &l.wq, Self::rows(&b.q, t))?;
        let k_new = self.cache(i, 0, p0, t);
        let v_new = self.cache(i, 1, p0, t);
        let k_rows = TensorView::contiguous(k_new.slice, 0, &[t, d.kv_dim], ACT);
        let v_rows = TensorView::contiguous(v_new.slice, 0, &[t, d.kv_dim], ACT);
        self.linear(Self::rows(&b.h, t), &l.wk, k_rows)?;
        self.linear(Self::rows(&b.h, t), &l.wv, v_rows)?;
        let rope = rope_cfg(&self.cfg);
        self.registry.rope(&rope).execute(&mut RopeContext {
            cfg: rope,
            q: self.heads(&b.q, t),
            k: k_new,
            positions: TensorView::contiguous(b.positions.whole(), 0, &[t], DType::I32),
            inv_freq: b.inv_freq.view(),
        })?;
        let kind = if t == 1 {
            AttentionKind::Decode
        } else {
            AttentionKind::Prefill
        };
        let attn = attention_cfg(&self.cfg, kind);
        self.registry
            .attention(&attn)
            .execute(&mut AttentionContext {
                cfg: attn,
                q: self.heads(&b.q, t),
                k_cache: self.cache(i, 0, 0, p0 + t),
                v_cache: self.cache(i, 1, 0, p0 + t),
                out: self.heads(&b.attn, t),
                q_start: p0 as u32,
                scale: 1.0 / (d.head_dim as f32).sqrt(),
            })?;
        self.linear(Self::rows(&b.attn, t), &l.wo, Self::rows(&b.proj, t))?;
        self.residual_add(t)?;

        // MLP block.
        self.rmsnorm(Self::rows(&b.x, t), &l.post_norm, Self::rows(&b.h, t))?;
        self.linear(Self::rows(&b.h, t), &l.w_gate, Self::rows(&b.gate, t))?;
        self.linear(Self::rows(&b.h, t), &l.w_up, Self::rows(&b.up, t))?;
        self.registry
            .activation(&activation_cfg(d))
            .execute(&mut ActivationContext {
                gate: Self::rows(&b.gate, t),
                up: Self::rows(&b.up, t),
                out: Self::rows(&b.act, t),
            })?;
        self.linear(Self::rows(&b.act, t), &l.w_down, Self::rows(&b.proj, t))?;
        self.residual_add(t)
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
        let p0 = self.validate(batch)?;
        let t = batch.tokens.len();
        i32_bytes(batch.tokens, &mut self.host_ids);
        i32_bytes(batch.positions, &mut self.host_positions);
        self.bufs.ids.copy_from_host(0, &self.host_ids)?;
        self.bufs
            .positions
            .copy_from_host(0, &self.host_positions)?;
        // Positions from p0 on are rewritten by this step; until it completes only the prefix
        // before p0 is valid.
        self.cached_len = p0;

        let b = &self.bufs;
        let d = &self.dims;
        self.registry
            .embedding(&embedding_cfg(d))
            .execute(&mut EmbeddingContext {
                ids: TensorView::contiguous(b.ids.whole(), 0, &[t], DType::I32),
                table: self.embed.view(),
                out: Self::rows(&b.x, t),
                vocab_offset: 0,
            })?;
        for i in 0..self.layers.len() {
            self.layer(i, t, p0 as usize)?;
        }
        self.rmsnorm(
            Self::rows(&b.x, t).rows(t - 1, 1),
            &self.final_norm,
            b.last.view(),
        )?;
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        self.linear(b.last.view(), head, b.logits.view())?;
        let raw = b.logits.storage.whole().read_bytes()?;
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.cached_len = p0 + t as u32;
        Ok(Logits {
            rows: 1,
            vocab: d.vocab,
            data,
        })
    }
}
