//! The sparse mixture-of-experts block (OLMoE): router GEMM with F32 logits → `moe_route`
//! (logits rounded to the activation dtype as transformers' BF16 router linear leaves them,
//! softmax, the top-k set `torch.topk` selects, renormalised only with `norm_topk_prob`) →
//! `moe_experts` into an accumulator zeroed by adding two zero buffers (one elementwise launch;
//! kernel ABI v2 has no device-to-device copy or memset).
//!
//! This is transformers' `OlmoeSparseMoeBlock` numerics: the experts' weighted outputs are
//! summed in the activation dtype in ascending expert order starting from zero; the skeleton
//! then adds the block's output to the residual.
//!
//! The loader uploads each expert's weights straight into its layer's stacked
//! `[experts, inter, hidden]` (gate, up) and `[experts, hidden, inter]` (down) tensors
//! ([`crate::loader::stacked_experts_name`]), the layout the `moe_experts` op takes, so loading
//! needs no device-to-device copy. When the selected `moe_experts` provider needs the group
//! sizes on the host for the batch's routed rows
//! ([`turbine_kernels::MoeKernel::needs_host_offsets`], decided from the host-known row count),
//! every layer reads the `[experts + 1]` expert offsets back after routing (a small blocking
//! copy, timed as [`profile::MOE_OFFSETS_READ`]) and the batch is not captured into a decode
//! graph ([`FfnHook::graph_capturable`]); otherwise (the HIP small-m path, up to 512 routed rows:
//! every decode-only batch of up to 64 sequences; above it the HIP grouped WMMA path when hidden
//! and inter are multiples of 64, as for OLMoE) the forward makes no device-to-host copy until
//! the logits.
//!
//! On an expert-parallel rank ([`crate::ep`]; `DecoderDims::ep`) the layer's stacks hold only
//! the rank's experts, `moe_experts` runs once per run of consecutive expert ids the rank holds
//! (the local expert range of the op, the weights of that run), each layer routes into its own
//! slot of `[layers, experts + 1]` expert offsets, the rank's output is combined across the
//! group ([`LayerRun::ep_combine`]) and, after each collected step, the offsets are read back
//! into the rank's [`crate::ep::ExpertTokenCounts`].
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::{
    ElementwiseContext, KernelRegistry, MoeExpertsConfig, MoeExpertsContext, MoeRouteConfig,
    MoeRouteContext, OpConfig,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor, TensorView};

use crate::ModelError;
use crate::config::MoeConfig;
use crate::executor::ExecutorOptions;
use crate::executor::decoder::{
    DecoderDims, FfnHook, HookBuffers, HookWeights, LayerRun, invalid, rows,
};
use crate::executor::profile;
use crate::loader::{LoadedWeights, stacked_experts_name};
use crate::weights::Bf16;

const I32: usize = 4;
const F32: usize = 4;
/// Alignment slack the shims may spend carving the `moe_experts` workspace into its five
/// regions (gathered rows, gate, up, activation, down).
const MOE_WORKSPACE_ALIGN_BYTES: u64 = 5 * 256;

/// Parameter indices of [`HookWeights`]: the router `[experts, hidden]`, the stacked gate and up
/// `[experts, inter, hidden]` and down `[experts, hidden, inter]`.
const ROUTER: usize = 0;
const W_GATE: usize = 1;
const W_UP: usize = 2;
const W_DOWN: usize = 3;
/// Buffer indices of [`HookBuffers::tensors`]: zeros `[tokens, hidden]` (activation dtype;
/// `zeros + zeros` resets the accumulator), router logits `[tokens, experts]` F32, top-k ids
/// (I32) and weights (F32) `[tokens, top_k]`, sorted rows `[tokens · top_k]` I32 and expert
/// offsets `[experts + 1]` I32; [`HookBuffers::scratch`] holds the `moe_experts` workspace.
const ZEROS: usize = 0;
const ROUTER_LOGITS: usize = 1;
const TOPK_IDS: usize = 2;
const TOPK_WEIGHTS: usize = 3;
const SORTED_ROWS: usize = 4;
const EXPERT_OFFSETS: usize = 5;
const WORKSPACE: usize = 0;

/// The sparse mixture of SwiGLU experts.
pub struct Moe;

/// The [`Moe`] hook.
pub const MOE: &dyn FfnHook = &Moe;

/// The configuration's mixture of experts, or why it has none.
fn moe_of(d: &DecoderDims) -> Result<MoeConfig, ModelError> {
    d.moe
        .ok_or_else(|| invalid("the moe FFN hook needs a config with experts (num_experts)".into()))
}

fn route_cfg(d: &DecoderDims, moe: &MoeConfig) -> MoeRouteConfig {
    MoeRouteConfig {
        num_experts: moe.num_experts,
        top_k: moe.experts_per_token,
        renormalize: moe.norm_topk_prob,
        // transformers' router is a BF16 linear: its logits are BF16 before the F32 softmax.
        bf16_logits: d.act == Bf16::DTYPE,
    }
}

/// The `moe_experts` call over the local experts `[begin, end)`.
fn experts_cfg(d: &DecoderDims, moe: &MoeConfig, (begin, end): (u32, u32)) -> MoeExpertsConfig {
    MoeExpertsConfig {
        hidden: d.hidden as u32,
        inter: moe.expert_intermediate,
        num_experts: moe.num_experts,
        top_k: moe.experts_per_token,
        expert_begin: begin,
        expert_end: end,
        dtype: d.act,
    }
}

/// The runs of consecutive expert ids layer `layer` holds: every expert on one device, the
/// rank's on an expert-parallel rank.
fn local_runs(d: &DecoderDims, moe: &MoeConfig, layer: usize) -> Vec<(u32, u32)> {
    match &d.ep {
        Some(ep) => ep.runs[layer].clone(),
        None => vec![(0, moe.num_experts)],
    }
}

/// Layers whose expert offsets are kept apart (each its own `[experts + 1]` slot): every layer
/// on an expert-parallel rank (read back after the step), else one slot all layers share.
fn offset_slots(d: &DecoderDims) -> usize {
    if d.ep.is_some() { d.layers } else { 1 }
}

/// Bytes of the `moe_experts` workspace for `tokens` tokens: every routed row
/// (`tokens · top_k`) gathered with its gate, up, activation and down rows in the activation
/// dtype (`2·hidden + 3·inter` elements), an 8-byte inverse-permutation entry per row, plus
/// alignment slack.
fn experts_workspace_bytes(d: &DecoderDims, moe: &MoeConfig, tokens: usize) -> u64 {
    let rows = (tokens * moe.experts_per_token as usize) as u64;
    let inter = moe.expert_intermediate as usize;
    let per_row = ((2 * d.hidden + 3 * inter) * d.act.size_bytes()) as u64 + 8;
    rows * per_row + MOE_WORKSPACE_ALIGN_BYTES
}

impl FfnHook for Moe {
    fn name(&self) -> &'static str {
        "moe"
    }

    fn check(&self, d: &DecoderDims) -> Result<(), ModelError> {
        moe_of(d).map(drop)
    }

    /// The F32-output router GEMM, `moe_route` and `moe_experts`; nothing for a config without
    /// experts (which [`FfnHook::check`] refuses).
    fn requirements(&self, d: &DecoderDims, _opts: ExecutorOptions) -> Vec<OpConfig> {
        let Some(moe) = d.moe else {
            return Vec::new();
        };
        let mut ops = vec![
            OpConfig::Gemm(d.gemm(moe.num_experts as usize, d.hidden, DType::F32)),
            OpConfig::MoeRoute(route_cfg(d, &moe)),
        ];
        for layer in 0..d.layers {
            for run in local_runs(d, &moe, layer) {
                let op = OpConfig::MoeExperts(experts_cfg(d, &moe, run));
                if !ops.contains(&op) {
                    ops.push(op);
                }
            }
        }
        ops
    }

    /// Per token the zeros row (hidden, activation dtype), the F32 router logits (experts) and
    /// top-k weights, the I32 top-k ids and sorted rows (top_k each) and the `moe_experts`
    /// workspace share; plus the I32 expert offsets (one `[experts + 1]` slot per layer on an
    /// expert-parallel rank) and the workspace alignment slack. Zero for a config without
    /// experts.
    fn workspace_bytes(&self, d: &DecoderDims, max_batch_tokens: usize) -> u64 {
        let Some(moe) = d.moe else {
            return 0;
        };
        let (experts, top_k) = (moe.num_experts as usize, moe.experts_per_token as usize);
        let per_token = (d.act.size_bytes() * d.hidden) as u64
            + (F32 * (experts + top_k)) as u64
            + (I32 * 2 * top_k) as u64;
        max_batch_tokens as u64 * per_token
            + experts_workspace_bytes(d, &moe, max_batch_tokens)
            + (I32 * (experts + 1) * offset_slots(d)) as u64
    }

    fn alloc(
        &self,
        d: &DecoderDims,
        max_batch_tokens: usize,
        mem: &Arc<dyn DeviceMemory>,
    ) -> Result<HookBuffers, ModelError> {
        let moe = moe_of(d)?;
        let t = max_batch_tokens;
        let (experts, top_k) = (moe.num_experts as usize, moe.experts_per_token as usize);
        let mut zeros = Tensor::empty(mem, &[t, d.hidden], d.act)?;
        zeros
            .storage
            .copy_from_host(0, &vec![0u8; t * d.hidden * d.act.size_bytes()])?;
        Ok(HookBuffers {
            tensors: vec![
                zeros,
                Tensor::empty(mem, &[t, experts], DType::F32)?,
                Tensor::empty(mem, &[t, top_k], DType::I32)?,
                Tensor::empty(mem, &[t, top_k], DType::F32)?,
                Tensor::empty(mem, &[t * top_k], DType::I32)?,
                Tensor::empty(mem, &[offset_slots(d), experts + 1], DType::I32)?,
            ],
            scratch: vec![DeviceBuffer::alloc(
                mem,
                experts_workspace_bytes(d, &moe, t) as usize,
            )?],
        })
    }

    fn load_layer(
        &self,
        d: &DecoderDims,
        layer: u32,
        prefix: &str,
        weights: &mut LoadedWeights,
        _opts: ExecutorOptions,
    ) -> Result<HookWeights, ModelError> {
        let moe = moe_of(d)?;
        let inter = moe.expert_intermediate as usize;
        let experts = local_runs(d, &moe, layer as usize)
            .iter()
            .map(|(b, e)| (e - b) as usize)
            .sum();
        let gate_shape = [experts, inter, d.hidden];
        let mut stacked = |proj: &str, shape: &[usize]| {
            d.take_weight(weights, &stacked_experts_name(layer, proj), shape)
        };
        let (w_gate, w_up, w_down) = (
            stacked("gate_proj", &gate_shape)?,
            stacked("up_proj", &gate_shape)?,
            stacked("down_proj", &[experts, d.hidden, inter])?,
        );
        let router = weights.take(&format!("{prefix}.mlp.gate.weight"))?;
        Ok(HookWeights(vec![router, w_gate, w_up, w_down]))
    }

    fn forward(&self, run: &LayerRun<'_>, w: &HookWeights) -> Result<(), ModelError> {
        let d = run.dims;
        let t = run.tokens;
        let moe = moe_of(d)?;
        let b = &run.ffn_buffers.tensors;
        let top_k = moe.experts_per_token as usize;
        run.linear(run.normed(), w.0[ROUTER].view(), rows(&b[ROUTER_LOGITS], t))?;
        run.trace("router_logits", &rows(&b[ROUTER_LOGITS], t))?;
        let route = route_cfg(d, &moe);
        let slot = if d.ep.is_some() { run.layer } else { 0 };
        let offsets = TensorView::contiguous(
            b[EXPERT_OFFSETS].storage.whole(),
            slot * (moe.num_experts as usize + 1),
            &[moe.num_experts as usize + 1],
            DType::I32,
        );
        let sorted_rows =
            TensorView::contiguous(b[SORTED_ROWS].storage.whole(), 0, &[t * top_k], DType::I32);
        run.op(OpConfig::MoeRoute(route), || {
            run.registry.moe_route(&route).route(&mut MoeRouteContext {
                cfg: route,
                router_logits: rows(&b[ROUTER_LOGITS], t),
                topk_ids: rows(&b[TOPK_IDS], t),
                topk_weights: rows(&b[TOPK_WEIGHTS], t),
                sorted_rows: sorted_rows.clone(),
                expert_offsets: offsets.clone(),
            })
        })?;
        if d.ep.is_some() {
            run.trace("topk_ids", &rows(&b[TOPK_IDS], t))?;
            run.trace("topk_weights", &rows(&b[TOPK_WEIGHTS], t))?;
        }
        let calls: Vec<MoeExpertsConfig> = local_runs(d, &moe, run.layer)
            .into_iter()
            .map(|r| experts_cfg(d, &moe, r))
            .collect();
        // The group sizes on the host (a blocking read of experts + 1 ints), only when the
        // provider of a call needs them for this many routed rows.
        let needs_host = calls.iter().any(|c| {
            run.registry
                .moe_experts(c)
                .needs_host_offsets(c, c.routed_rows(t))
        });
        let host_offsets: Vec<i32> = if needs_host {
            run.step(profile::MOE_OFFSETS_READ, profile::D2H, || {
                Ok(offsets
                    .slice
                    .read_bytes()?
                    .chunks_exact(I32)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            })?
        } else {
            Vec::new()
        };
        // The accumulator starts at zero, as transformers' `final_hidden_states`: exactly
        // `0 + 0` (no device-to-device copy under kernel ABI v2).
        let add = d.add();
        run.step(profile::MOE_ZERO, profile::ZERO_ADD, || {
            Ok(run
                .registry
                .elementwise(&add)
                .execute(&mut ElementwiseContext {
                    a: rows(&b[ZEROS], t),
                    b: rows(&b[ZEROS], t),
                    out: run.ffn_out(),
                })?)
        })?;
        let workspace_len = experts_workspace_bytes(d, &moe, t) as usize;
        // Each call adds its experts' weighted outputs in ascending expert order, so the calls
        // in ascending order add a token's experts in ascending order (one device's).
        let mut local = 0;
        for experts in calls {
            let n = experts.num_local_experts() as usize;
            let weight = |i: usize| w.0[i].view().rows(local, n);
            run.op(OpConfig::MoeExperts(experts), || {
                run.registry
                    .moe_experts(&experts)
                    .experts(&mut MoeExpertsContext {
                        cfg: experts,
                        x: run.normed(),
                        w_gate: weight(W_GATE),
                        w_up: weight(W_UP),
                        w_down: weight(W_DOWN),
                        sorted_rows: sorted_rows.clone(),
                        expert_offsets: offsets.clone(),
                        topk_weights: rows(&b[TOPK_WEIGHTS], t),
                        host_expert_offsets: &host_offsets,
                        out: run.ffn_out(),
                        workspace: Some(run.ffn_buffers.scratch[WORKSPACE].slice(0, workspace_len)),
                    })
            })?;
            local += n;
        }
        if d.ep.is_some() {
            run.trace("moe_partial", &run.ffn_out())?;
            run.ep_combine(&run.ffn_out())?;
        }
        run.trace("moe_out", &run.ffn_out())
    }

    /// Main's OLMoE filter: not when the provider reads the group sizes on the host for the
    /// batch's routed rows (it copies them back mid-forward).
    fn graph_capturable(&self, d: &DecoderDims, registry: &KernelRegistry, rows: usize) -> bool {
        let Some(moe) = d.moe else {
            return true;
        };
        (0..d.layers)
            .flat_map(|layer| local_runs(d, &moe, layer))
            .all(|run| {
                let c = experts_cfg(d, &moe, run);
                !registry
                    .moe_experts(&c)
                    .needs_host_offsets(&c, c.routed_rows(rows))
            })
    }

    /// Expert parallelism: every layer's expert offsets of the collected step into the rank's
    /// token counts (one `[layers, experts + 1]` read; the stream is synchronised).
    fn collected(&self, d: &DecoderDims, buffers: &HookBuffers) -> Result<(), ModelError> {
        let Some(ep) = &d.ep else {
            return Ok(());
        };
        let raw = buffers.tensors[EXPERT_OFFSETS]
            .storage
            .whole()
            .read_bytes()?;
        let all: Vec<i32> = raw
            .chunks_exact(I32)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let per_layer = all.len() / d.layers.max(1);
        let layers: Vec<&[i32]> = ep
            .placement
            .layers
            .iter()
            .map(|(l, _)| &all[*l as usize * per_layer..(*l as usize + 1) * per_layer])
            .collect();
        ep.counts.record(&ep.placement, &layers);
        Ok(())
    }
}
