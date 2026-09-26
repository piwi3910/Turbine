//! OLMoE forward pass over a ragged batch of sequences (P2 S-16, S-5, S-9) on the kernel
//! registry: embedding → per layer (RMSNorm → Q/K/V projections → RMSNorm over the full Q and
//! K projections (`q_norm`, `k_norm`) → standard RoPE → paged causal attention, which appends
//! the new K/V rows into each sequence's pool blocks → O projection → residual add → RMSNorm →
//! router GEMM with F32 logits → `moe_route` (softmax, top-k, renormalised only with
//! `norm_topk_prob`) → `moe_experts` into a BF16 accumulator zeroed by adding two zero buffers
//! (one elementwise launch; kernel ABI v2 has no device-to-device copy or memset) → residual
//! add) → final
//! RMSNorm on each sequence's last row → untied LM head with FP32 output.
//!
//! This is transformers' `OlmoeSparseMoeBlock` numerics: the experts' weighted outputs are
//! summed in BF16 in ascending expert order starting from zero, then added to the residual.
//! Batch packing, the KV pool and the block fork are shared with the Llama executor
//! ([`super::batch`]).
//!
//! The loader uploads each expert's weights straight into its layer's stacked
//! `[experts, inter, hidden]` (gate, up) and `[experts, hidden, inter]` (down) tensors
//! ([`crate::loader::stacked_experts_name`]), the layout the `moe_experts` op takes, so loading
//! needs no device-to-device copy. When the selected `moe_experts` provider needs the group
//! sizes on the host for the batch's routed rows
//! ([`turbine_kernels::MoeKernel::needs_host_offsets`], decided from the host-known row count),
//! every layer reads the `[experts + 1]` expert offsets back after routing (a small blocking
//! copy); otherwise (the HIP small-m path, up to 512 routed rows: every decode-only batch of up
//! to 64 sequences) the forward makes no device-to-host copy until the logits, its only bulk
//! device-to-host copy.
use std::sync::Arc;
use std::time::Instant;

use turbine_core::types::{BlockId, DType, KvLayout, ModelShape};
use turbine_kernels::{
    AttentionKind, ElementwiseContext, EmbeddingConfig, EmbeddingContext, GemmContext,
    KernelRegistry, MoeExpertsConfig, MoeExpertsContext, MoeRouteConfig, MoeRouteContext,
    NormConfig, NormContext, OpConfig, OpRequirement, PagedAttentionContext, RopeContext,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView, Tensor, TensorView};

use super::batch::{self, BatchLimits, DeviceBatch, HostBatch, Packed};
use super::llama::{ACT, ADD_CFG, attention_cfg, gemm_cfg, invalid, limits, rope_cfg};
use super::{BatchInput, ForwardTimings, Logits, ModelExecutor, rope};
use crate::ModelError;
use crate::config::{Architecture, ModelArchConfig, MoeConfig};
use crate::loader::{LM_HEAD, LoadedWeights, stacked_experts_name};

const I32: usize = 4;
const F32: usize = 4;
/// Alignment slack the shims may spend carving the `moe_experts` workspace into its five
/// regions (gathered rows, gate, up, activation, down).
const MOE_WORKSPACE_ALIGN_BYTES: u64 = 5 * 256;

/// Model dimensions in elements.
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    experts: usize,
    top_k: usize,
    inter: usize,
}

impl Dims {
    fn of(cfg: &ModelArchConfig, moe: &MoeConfig) -> Dims {
        let head_dim = cfg.head_dim as usize;
        Dims {
            hidden: cfg.hidden as usize,
            q_dim: cfg.num_attention_heads as usize * head_dim,
            kv_dim: cfg.num_kv_heads as usize * head_dim,
            vocab: cfg.vocab_size as usize,
            heads: cfg.num_attention_heads as usize,
            kv_heads: cfg.num_kv_heads as usize,
            head_dim,
            experts: moe.num_experts as usize,
            top_k: moe.experts_per_token as usize,
            inter: moe.expert_intermediate as usize,
        }
    }

    /// Bytes of the `moe_experts` workspace for `tokens` tokens: every routed row
    /// (`tokens · top_k`) gathered with its gate, up, activation and down rows in BF16
    /// (`2·hidden + 3·inter` elements), an 8-byte inverse-permutation entry per row, plus
    /// alignment slack.
    fn moe_workspace_bytes(&self, tokens: usize) -> u64 {
        let rows = (tokens * self.top_k) as u64;
        let per_row = ((2 * self.hidden + 3 * self.inter) * ACT.size_bytes()) as u64 + 8;
        rows * per_row + MOE_WORKSPACE_ALIGN_BYTES
    }
}

fn norm_cfg(dim: usize) -> NormConfig {
    NormConfig {
        dim: dim as u64,
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

fn route_cfg(moe: &MoeConfig) -> MoeRouteConfig {
    MoeRouteConfig {
        num_experts: moe.num_experts,
        top_k: moe.experts_per_token,
        renormalize: moe.norm_topk_prob,
    }
}

/// Every expert of the layer is local (no expert parallelism before Phase 7).
fn experts_cfg(cfg: &ModelArchConfig, moe: &MoeConfig) -> MoeExpertsConfig {
    MoeExpertsConfig {
        hidden: cfg.hidden,
        inter: moe.expert_intermediate,
        num_experts: moe.num_experts,
        top_k: moe.experts_per_token,
        expert_begin: 0,
        expert_end: moe.num_experts,
        dtype: ACT,
    }
}

/// The MoE settings of an OLMoE config, or why it is not one.
fn moe_of(cfg: &ModelArchConfig) -> Result<MoeConfig, ModelError> {
    match (cfg.architecture, cfg.moe) {
        (Architecture::Olmoe, Some(moe)) if cfg.qk_norm => Ok(moe),
        _ => Err(invalid(format!(
            "the OLMoE executor needs an OlmoeForCausalLM config with experts and Q/K norm, got \
             {} (experts: {}, qk_norm: {})",
            cfg.architecture.as_str(),
            cfg.moe.is_some(),
            cfg.qk_norm
        ))),
    }
}

struct Layer {
    input_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_norm: Tensor,
    /// `[experts, hidden]`.
    router: Tensor,
    /// `[experts, inter, hidden]`.
    w_gate: Tensor,
    w_up: Tensor,
    /// `[experts, hidden, inter]`.
    w_down: Tensor,
}

/// Activation and routing buffers, allocated once for `max_batch_tokens` tokens and
/// `max_seqs` sequences.
struct Buffers {
    inv_freq: Tensor,
    /// Residual stream.
    x: Tensor,
    /// Normalised input of the attention and MoE blocks.
    h: Tensor,
    /// Q after its norm (rotated in place); the raw Q projection goes through `attn`.
    q: Tensor,
    /// K and V of the new rows (K normalised, then rotated in place), appended by attention.
    k: Tensor,
    v: Tensor,
    /// The raw K projection before its norm.
    k_raw: Tensor,
    /// The raw Q projection, then the attention output.
    attn: Tensor,
    /// O projection output, then the MoE accumulator, before each residual add.
    proj: Tensor,
    /// BF16 zeros; `zeros + zeros` resets the MoE accumulator.
    zeros: Tensor,
    /// `[tokens, experts]` F32.
    router_logits: Tensor,
    /// `[tokens, top_k]` I32 / F32.
    topk_ids: Tensor,
    topk_weights: Tensor,
    /// `[tokens · top_k]` I32.
    sorted_rows: Tensor,
    /// `[experts + 1]` I32.
    expert_offsets: Tensor,
    /// Provider scratch of `moe_experts`.
    moe_workspace: DeviceBuffer,
    /// Final-norm output of each sequence's last row.
    last: Tensor,
    logits: Tensor,
}

/// The OLMoE executor: ragged batches of up to `max_seqs` sequences and `max_batch_tokens`
/// tokens over the paged KV pool, on one device.
pub struct OlmoeExecutor {
    cfg: ModelArchConfig,
    moe: MoeConfig,
    dims: Dims,
    shape: ModelShape,
    kv_layout: KvLayout,
    limits: BatchLimits,
    registry: Arc<KernelRegistry>,
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    lm_head: Tensor,
    bufs: Buffers,
    meta: DeviceBatch,
    /// Host bytes of the last upload; valid until the next synchronization (contract §9.2).
    host: HostBatch,
    /// Timings of the last forward ([`ModelExecutor::last_timings`]).
    timings: ForwardTimings,
}

/// Layer `layer`'s stacked expert projection `proj`, as the loader filled it, checked to be
/// `shape` BF16.
fn stacked_experts(
    weights: &mut LoadedWeights,
    layer: u32,
    proj: &str,
    shape: [usize; 3],
) -> Result<Tensor, ModelError> {
    let name = stacked_experts_name(layer, proj);
    let stacked = weights.take(&name)?;
    if stacked.shape.as_slice() != shape || stacked.dtype != ACT {
        return Err(invalid(format!(
            "{name} is {:?} {}, expected {shape:?} {}",
            stacked.shape.as_slice(),
            stacked.dtype.as_str(),
            ACT.as_str()
        )));
    }
    Ok(stacked)
}

impl OlmoeExecutor {
    /// Every distinct op config the forward pass executes, in first-use order, plus the
    /// `copy_blocks` fork over KV blocks of `block_tokens` tokens: the Llama attention ops, RMSNorm
    /// over `hidden`, `heads·head_dim` and `kv_heads·head_dim` (Q/K norm), the F32-output router
    /// GEMM, `moe_route` and `moe_experts`, and no dense MLP. A config without experts lists no
    /// MoE ops ([`OlmoeExecutor::new`] refuses it).
    pub fn requirements(cfg: &ModelArchConfig, block_tokens: u32) -> Vec<OpRequirement> {
        let head_dim = cfg.head_dim as usize;
        let hidden = cfg.hidden as usize;
        let q_dim = cfg.num_attention_heads as usize * head_dim;
        let kv_dim = cfg.num_kv_heads as usize * head_dim;
        let vocab = cfg.vocab_size as usize;
        let mut specs = vec![
            OpConfig::Embedding(EmbeddingConfig {
                hidden: hidden as u64,
                vocab_rows: vocab as u64,
                dtype: ACT,
            }),
            OpConfig::Rmsnorm(norm_cfg(hidden)),
            OpConfig::Gemm(gemm_cfg(q_dim, hidden, ACT)),
            OpConfig::Gemm(gemm_cfg(kv_dim, hidden, ACT)),
            OpConfig::Rmsnorm(norm_cfg(q_dim)),
            OpConfig::Rmsnorm(norm_cfg(kv_dim)),
            OpConfig::Rope(rope_cfg(cfg)),
            OpConfig::Attention(attention_cfg(
                cfg,
                AttentionKind::PrefillPaged,
                block_tokens,
            )),
            OpConfig::Attention(attention_cfg(cfg, AttentionKind::DecodePaged, block_tokens)),
            OpConfig::Gemm(gemm_cfg(hidden, q_dim, ACT)),
            OpConfig::Add(ADD_CFG),
        ];
        if let Some(moe) = cfg.moe {
            specs.extend([
                OpConfig::Gemm(gemm_cfg(moe.num_experts as usize, hidden, DType::F32)),
                OpConfig::MoeRoute(route_cfg(&moe)),
                OpConfig::MoeExperts(experts_cfg(cfg, &moe)),
            ]);
        }
        specs.extend([
            OpConfig::Gemm(gemm_cfg(vocab, hidden, DType::F32)),
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

    /// Device bytes of the executor's buffers (the budget's workspace term), including the MoE
    /// permutation buffers: per token the I32 id and position, the BF16 rows of `x`, `h`,
    /// `proj`, `zeros` (hidden), `q`, `attn` (heads · head_dim), `k`, `v`, `k_raw`
    /// (kv_heads · head_dim), the F32 router logits (experts) and top-k weights, the I32 top-k
    /// ids and sorted rows (top_k each) and the `moe_experts` workspace share; per sequence its
    /// last normalised row (BF16), its F32 logits row, its `q_indptr` and `kv_lens` entries and a
    /// block table for `max_position_embeddings` tokens (I32); plus one `q_indptr` entry, the
    /// I32 expert offsets, the F32 `inv_freq` and the workspace alignment slack. Zero for a
    /// config without experts.
    pub fn workspace_bytes(
        cfg: &ModelArchConfig,
        block_tokens: u32,
        max_batch_tokens: u32,
        max_seqs: u32,
    ) -> u64 {
        let Ok(moe) = moe_of(cfg) else {
            return 0;
        };
        let d = Dims::of(cfg, &moe);
        let t = max_batch_tokens as usize;
        let bf16 = ACT.size_bytes() as u64;
        let per_token = bf16 * (4 * d.hidden + 2 * d.q_dim + 3 * d.kv_dim) as u64
            + (F32 * (d.experts + d.top_k)) as u64
            + (I32 * 2 * d.top_k) as u64;
        let per_seq = bf16 * d.hidden as u64 + (F32 * d.vocab) as u64;
        let limits = limits(cfg, block_tokens, max_batch_tokens, max_seqs);
        t as u64 * per_token
            + d.moe_workspace_bytes(t)
            + u64::from(max_seqs) * per_seq
            + (I32 * (d.experts + 1)) as u64
            + (F32 * (d.head_dim / 2)) as u64
            + DeviceBatch::bytes(&limits)
    }

    /// Takes the parameters out of `weights` (the expert weights stacked by the loader), allocates the
    /// activation, routing and batch buffers for `max_batch_tokens` tokens of up to `max_seqs`
    /// sequences on `mem`, and uploads the rotary inverse frequencies. Every forward names its
    /// KV pool, laid out as `cfg.kv_layout(block_tokens)`. `registry` must have been built from
    /// [`OlmoeExecutor::requirements`] of `cfg` and `block_tokens`.
    pub fn new(
        cfg: &ModelArchConfig,
        mut weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        block_tokens: u32,
        max_batch_tokens: u32,
        max_seqs: u32,
    ) -> Result<OlmoeExecutor, ModelError> {
        let moe = moe_of(cfg)?;
        if cfg.tie_word_embeddings {
            return Err(invalid(
                "the OLMoE executor needs an untied LM head (tie_word_embeddings false)".into(),
            ));
        }
        if block_tokens == 0 || max_batch_tokens == 0 || max_seqs == 0 {
            return Err(invalid(format!(
                "block_tokens {block_tokens}, max_batch_tokens {max_batch_tokens} and max_seqs \
                 {max_seqs} must be positive"
            )));
        }
        let d = Dims::of(cfg, &moe);
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let mut take = |s: &str| weights.take(&format!("{p}.{s}.weight"));
            let (input_norm, wq, wk, wv, wo) = (
                take("input_layernorm")?,
                take("self_attn.q_proj")?,
                take("self_attn.k_proj")?,
                take("self_attn.v_proj")?,
                take("self_attn.o_proj")?,
            );
            let (q_norm, k_norm, post_norm, router) = (
                take("self_attn.q_norm")?,
                take("self_attn.k_norm")?,
                take("post_attention_layernorm")?,
                take("mlp.gate")?,
            );
            let gate_shape = [d.experts, d.inter, d.hidden];
            layers.push(Layer {
                input_norm,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                post_norm,
                router,
                w_gate: stacked_experts(&mut weights, i, "gate_proj", gate_shape)?,
                w_up: stacked_experts(&mut weights, i, "up_proj", gate_shape)?,
                w_down: stacked_experts(
                    &mut weights,
                    i,
                    "down_proj",
                    [d.experts, d.hidden, d.inter],
                )?,
            });
        }
        let embed = weights.take("model.embed_tokens.weight")?;
        let final_norm = weights.take("model.norm.weight")?;
        let lm_head = weights.take(LM_HEAD)?;

        let limits = limits(cfg, block_tokens, max_batch_tokens, max_seqs);
        let t = max_batch_tokens as usize;
        let n = max_seqs as usize;
        let act = |cols: usize| Tensor::empty(&mem, &[t, cols], ACT);
        let mut inv_freq = Tensor::empty(&mem, &[d.head_dim / 2], DType::F32)?;
        let freqs = rope::inv_freq(cfg.rope_theta, cfg.head_dim, cfg.rope_scaling.as_ref());
        let freq_bytes: Vec<u8> = freqs.iter().flat_map(|f| f.to_le_bytes()).collect();
        inv_freq.storage.copy_from_host(0, &freq_bytes)?;
        let mut zeros = act(d.hidden)?;
        zeros
            .storage
            .copy_from_host(0, &vec![0u8; t * d.hidden * ACT.size_bytes()])?;
        mem.synchronize()?;
        let bufs = Buffers {
            inv_freq,
            x: act(d.hidden)?,
            h: act(d.hidden)?,
            q: act(d.q_dim)?,
            k: act(d.kv_dim)?,
            v: act(d.kv_dim)?,
            k_raw: act(d.kv_dim)?,
            attn: act(d.q_dim)?,
            proj: act(d.hidden)?,
            zeros,
            router_logits: Tensor::empty(&mem, &[t, d.experts], DType::F32)?,
            topk_ids: Tensor::empty(&mem, &[t, d.top_k], DType::I32)?,
            topk_weights: Tensor::empty(&mem, &[t, d.top_k], DType::F32)?,
            sorted_rows: Tensor::empty(&mem, &[t * d.top_k], DType::I32)?,
            expert_offsets: Tensor::empty(&mem, &[d.experts + 1], DType::I32)?,
            moe_workspace: DeviceBuffer::alloc(&mem, d.moe_workspace_bytes(t) as usize)?,
            last: Tensor::empty(&mem, &[n, d.hidden], ACT)?,
            logits: Tensor::empty(&mem, &[n, d.vocab], DType::F32)?,
        };
        let meta = DeviceBatch::alloc(&mem, &limits)?;
        Ok(OlmoeExecutor {
            cfg: cfg.clone(),
            moe,
            dims: d,
            shape: cfg.shape(),
            kv_layout: limits.layout,
            limits,
            registry,
            embed,
            layers,
            final_norm,
            lm_head,
            bufs,
            meta,
            host: HostBatch::default(),
            timings: ForwardTimings::default(),
        })
    }

    /// Rows `[0, t)` of a per-token buffer as `[t, cols]`.
    fn rows(buf: &Tensor, t: usize) -> TensorView<'_> {
        buf.view().rows(0, t)
    }

    /// Rows `[0, t)` of an activation buffer as `[t, heads, head_dim]`.
    fn heads(buf: &Tensor, t: usize, heads: usize, head_dim: usize) -> TensorView<'_> {
        TensorView::contiguous(buf.storage.whole(), 0, &[t, heads, head_dim], ACT)
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

    /// RMSNorm over rows of `w.len()` elements.
    fn rmsnorm(
        &self,
        x: TensorView<'_>,
        w: &Tensor,
        out: TensorView<'_>,
    ) -> Result<(), ModelError> {
        self.registry
            .norm(&norm_cfg(w.numel()))
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

    /// The attention block of layer `i` on the batch's `p.total_q` rows; attention appends to
    /// and reads layer `i` of `kv`.
    fn attention(&self, i: usize, p: &Packed, kv: &KvPoolView<'_>) -> Result<(), ModelError> {
        let l = &self.layers[i];
        let b = &self.bufs;
        let d = &self.dims;
        let t = p.total_q;
        self.rmsnorm(Self::rows(&b.x, t), &l.input_norm, Self::rows(&b.h, t))?;
        self.linear(Self::rows(&b.h, t), &l.wq, Self::rows(&b.attn, t))?;
        self.linear(Self::rows(&b.h, t), &l.wk, Self::rows(&b.k_raw, t))?;
        self.linear(Self::rows(&b.h, t), &l.wv, Self::rows(&b.v, t))?;
        self.rmsnorm(Self::rows(&b.attn, t), &l.q_norm, Self::rows(&b.q, t))?;
        self.rmsnorm(Self::rows(&b.k_raw, t), &l.k_norm, Self::rows(&b.k, t))?;
        let rope = rope_cfg(&self.cfg);
        self.registry.rope(&rope).execute(&mut RopeContext {
            cfg: rope,
            q: Self::heads(&b.q, t, d.heads, d.head_dim),
            k: Self::heads(&b.k, t, d.kv_heads, d.head_dim),
            positions: self.meta.positions_view(p),
            inv_freq: b.inv_freq.view(),
        })?;
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
                q: Self::heads(&b.q, t, d.heads, d.head_dim),
                k_new: Self::heads(&b.k, t, d.kv_heads, d.head_dim),
                v_new: Self::heads(&b.v, t, d.kv_heads, d.head_dim),
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
        self.linear(Self::rows(&b.attn, t), &l.wo, Self::rows(&b.proj, t))?;
        self.residual_add(t)
    }

    /// The sparse MoE block of layer `i` on `t` rows.
    fn moe_block(&self, i: usize, t: usize) -> Result<(), ModelError> {
        let l = &self.layers[i];
        let b = &self.bufs;
        let d = &self.dims;
        self.rmsnorm(Self::rows(&b.x, t), &l.post_norm, Self::rows(&b.h, t))?;
        self.linear(
            Self::rows(&b.h, t),
            &l.router,
            Self::rows(&b.router_logits, t),
        )?;
        let route = route_cfg(&self.moe);
        let sorted_rows =
            TensorView::contiguous(b.sorted_rows.storage.whole(), 0, &[t * d.top_k], DType::I32);
        self.registry
            .moe_route(&route)
            .route(&mut MoeRouteContext {
                cfg: route,
                router_logits: Self::rows(&b.router_logits, t),
                topk_ids: Self::rows(&b.topk_ids, t),
                topk_weights: Self::rows(&b.topk_weights, t),
                sorted_rows: sorted_rows.clone(),
                expert_offsets: b.expert_offsets.view(),
            })?;
        let experts = experts_cfg(&self.cfg, &self.moe);
        let kernel = self.registry.moe_experts(&experts);
        // The group sizes on the host (a blocking read of experts + 1 ints), only when the
        // provider needs them for this many routed rows.
        let host_offsets: Vec<i32> = if kernel.needs_host_offsets(&experts, experts.routed_rows(t))
        {
            b.expert_offsets
                .storage
                .whole()
                .read_bytes()?
                .chunks_exact(I32)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            Vec::new()
        };
        // The accumulator starts at zero, as transformers' `final_hidden_states`: exactly
        // `0 + 0` in BF16 (no device-to-device copy under kernel ABI v2).
        self.registry
            .elementwise(&ADD_CFG)
            .execute(&mut ElementwiseContext {
                a: Self::rows(&b.zeros, t),
                b: Self::rows(&b.zeros, t),
                out: Self::rows(&b.proj, t),
            })?;
        let workspace_len = d.moe_workspace_bytes(t) as usize;
        kernel.experts(&mut MoeExpertsContext {
            cfg: experts,
            x: Self::rows(&b.h, t),
            w_gate: l.w_gate.view(),
            w_up: l.w_up.view(),
            w_down: l.w_down.view(),
            sorted_rows,
            expert_offsets: b.expert_offsets.view(),
            topk_weights: Self::rows(&b.topk_weights, t),
            host_expert_offsets: &host_offsets,
            out: Self::rows(&b.proj, t),
            workspace: Some(b.moe_workspace.slice(0, workspace_len)),
        })?;
        self.residual_add(t)
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

impl ModelExecutor for OlmoeExecutor {
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
        for i in 0..self.layers.len() {
            self.attention(i, &p, batch.kv)?;
            self.moe_block(i, t)?;
        }
        self.final_norm_last_rows(&p)?;
        self.linear(
            Self::rows(&b.last, n),
            &self.lm_head,
            Self::rows(&b.logits, n),
        )?;
        let wait_started = Instant::now();
        // The iteration's logits copy (it synchronizes the stream).
        let raw = b
            .logits
            .storage
            .slice(0, n * d.vocab * DType::F32.size_bytes())
            .read_bytes()?;
        let data: Vec<f32> = raw
            .chunks_exact(F32)
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
