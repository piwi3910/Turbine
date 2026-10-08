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
    ElementwiseContext, ExpertActivation, ExpertDtype, KernelRegistry, MoeExpertsConfig,
    MoeExpertsContext, MoeExpertsExtra, MoeRouteConfig, MoeRouteContext, OpConfig, RouteMode,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor, TensorView};

use super::attention_sinks_window::bf16_as_f32;
use crate::ModelError;
use crate::config::MoeConfig;
use crate::executor::ExecutorOptions;
use crate::executor::decoder::linear::LinearQuant;
use crate::executor::decoder::{
    DecoderDims, FfnHook, HookBuffers, HookWeights, LayerRun, invalid, rows,
};
use crate::executor::profile;
use crate::loader::{LoadedWeights, stacked_experts_name};
use crate::weights::{ActivationQuant, Bf16, QuantScheme};

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
/// A quantized layer's further parameters: the weight scales of gate, up and down (`[experts,
/// blocks…]` for FP8 block, `[experts]` for FP8 tensor, E8M0 `[experts, n, k / 32]` for MXFP4),
/// then (FP8 tensor, W8A8) the per-expert static activation scales of gate/up (the larger of the
/// two) and of down, each `[experts]` F32.
const S_GATE: usize = 4;
const S_UP: usize = 5;
const S_DOWN: usize = 6;
const ACT_GATE_UP: usize = 7;
const ACT_DOWN: usize = 8;
/// Parameter indices of [`HookWeights`] of a gpt-oss layer ([`Style::GptOss`]): the router
/// `[experts, hidden]` (index 0 as above) and its F32 bias `[experts]`, then per fused
/// projection its U8 MXFP4 data `[experts, n, k/2]`, E8M0 scales `[experts, n, k/32]` and bias
/// `[experts, n]` (`n` = `2 · inter` for gate/up, `hidden` for down).
const G_ROUTER_BIAS: usize = 1;
const G_GATE_UP: usize = 2;
const G_GATE_UP_SCALE: usize = 3;
const G_GATE_UP_BIAS: usize = 4;
const G_DOWN: usize = 5;
const G_DOWN_SCALE: usize = 6;
const G_DOWN_BIAS: usize = 7;
/// The clamp of gpt-oss's activation (`swiglu_limit` of every released checkpoint).
pub const GPT_OSS_SWIGLU_LIMIT: f32 = 7.0;
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

/// Which expert block a [`Moe`] hook runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Style {
    /// SwiGLU experts behind a softmax-then-top-k router (OLMoE, Qwen3-MoE, Mixtral), in the
    /// decoder's weight format.
    SwiGlu,
    /// gpt-oss (P7 S-8): MXFP4 experts stored fused (`gate_up_proj_*` with interleaved gate and
    /// up rows, `down_proj_*`), biases on both, the clamped gated activation, behind a router
    /// with a bias that takes the top-k of the logits and then a softmax over those.
    GptOss,
}

/// The sparse mixture of experts.
pub struct Moe {
    style: Style,
}

/// The [`Moe`] hook: SwiGLU experts.
pub const MOE: &dyn FfnHook = &Moe {
    style: Style::SwiGlu,
};

/// The [`Moe`] hook of gpt-oss ([`Style::GptOss`]).
pub const MOE_GPT_OSS: &dyn FfnHook = &Moe {
    style: Style::GptOss,
};

impl Style {
    /// The expert weight format of a decoder with `quant`: gpt-oss's experts are MXFP4
    /// whatever its linear layers are; `Err` for a format without a kernel row.
    fn weight_dtype(self, quant: Option<LinearQuant>) -> Result<ExpertDtype, ()> {
        match self {
            Style::SwiGlu => expert_dtype(quant),
            Style::GptOss => Ok(ExpertDtype::Mxfp4),
        }
    }
}

/// The configuration's mixture of experts, or why it has none.
fn moe_of(d: &DecoderDims) -> Result<MoeConfig, ModelError> {
    d.moe
        .ok_or_else(|| invalid("the moe FFN hook needs a config with experts (num_experts)".into()))
}

/// The expert weight format of a decoder quantization (`None`: BF16 experts), or `None` inside
/// the `Err` for a format without a kernel row (`quant_moe_phase7`).
fn expert_dtype(quant: Option<LinearQuant>) -> Result<ExpertDtype, ()> {
    match quant {
        None => Ok(ExpertDtype::Bf16),
        Some(LinearQuant {
            scheme: QuantScheme::Fp8Block { n: 128, k: 128 },
            act: ActivationQuant::None,
        }) => Ok(ExpertDtype::Fp8Block),
        // Per-tensor scales are stored per row: one expert scale needs equal rows (checked at
        // load); W8A8 with static activation scales only (W8A16 has no kernel).
        Some(LinearQuant {
            scheme: QuantScheme::Fp8Channel,
            act: ActivationQuant::Fp8PerTensorStatic,
        }) => Ok(ExpertDtype::Fp8Tensor),
        Some(LinearQuant {
            scheme: QuantScheme::Mxfp4,
            act: ActivationQuant::None,
        }) => Ok(ExpertDtype::Mxfp4),
        Some(_) => Err(()),
    }
}

/// Whether the quantized experts of a decoder with `quant` have a kernel row (the check
/// `quant_moe_phase7` refuses the rest with).
pub fn experts_served(quant: &LinearQuant) -> bool {
    expert_dtype(Some(*quant)).is_ok()
}

fn route_cfg(d: &DecoderDims, moe: &MoeConfig, style: Style) -> MoeRouteConfig {
    MoeRouteConfig {
        num_experts: moe.num_experts,
        top_k: moe.experts_per_token,
        renormalize: moe.norm_topk_prob,
        // transformers' router is a BF16 linear: its logits are BF16 before the F32 softmax.
        bf16_logits: d.act == Bf16::DTYPE,
        mode: match style {
            Style::SwiGlu => RouteMode::SoftmaxThenTopk,
            Style::GptOss => RouteMode::TopkThenSoftmax,
        },
        bias: style == Style::GptOss,
    }
}

/// The `moe_experts` call over the local experts `[begin, end)`.
fn experts_cfg(
    d: &DecoderDims,
    moe: &MoeConfig,
    (begin, end): (u32, u32),
    style: Style,
) -> MoeExpertsConfig {
    let gpt_oss = style == Style::GptOss;
    MoeExpertsConfig {
        hidden: d.hidden as u32,
        inter: moe.expert_intermediate,
        num_experts: moe.num_experts,
        top_k: moe.experts_per_token,
        expert_begin: begin,
        expert_end: end,
        dtype: d.act,
        // A format without a row never gets here (`check_weight_format`).
        weight_dtype: style.weight_dtype(d.quant).unwrap_or(ExpertDtype::Bf16),
        activation: if gpt_oss {
            ExpertActivation::GptOssClamp {
                limit: GPT_OSS_SWIGLU_LIMIT,
            }
        } else {
            ExpertActivation::SwiGlu
        },
        biases: gpt_oss,
        gate_up_interleaved: gpt_oss,
        ..Default::default()
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
fn experts_workspace_bytes(d: &DecoderDims, moe: &MoeConfig, tokens: usize, style: Style) -> u64 {
    let rows = (tokens * moe.experts_per_token as usize) as u64;
    let inter = moe.expert_intermediate as usize;
    let per_row = ((2 * d.hidden + 3 * inter) * d.act.size_bytes()) as u64 + 8;
    // The quantized per-expert loop keeps, per routed row: its position (4), the gathered row
    // (2·hidden) and its FP8 copy (hidden), gate and up (4·inter), the activation (2·inter) and
    // its FP8 copy (inter), a dynamic scale (4) and the down output (2·hidden); an upper bound
    // over the three formats, 16 regions of alignment slack.
    let quantized = style == Style::GptOss
        || d.quant.is_some() && expert_dtype(d.quant).is_ok_and(|t| t != ExpertDtype::Bf16);
    if quantized {
        let per_row_quant = (8 + 5 * d.hidden + 7 * inter) as u64;
        return rows * per_row.max(per_row_quant) + 16 * 256;
    }
    rows * per_row + MOE_WORKSPACE_ALIGN_BYTES
}

/// The `quant_moe_phase7` refusal text for `what`.
fn refused(d: &DecoderDims, what: &str) -> ModelError {
    invalid(format!(
        "quant_moe_phase7: {what} experts have no supported row (scheme {:?})",
        d.quant
    ))
}

/// One expert projection stack as the kernels take it: the weights `[experts, n, k]` (packed
/// `k / 2` for MXFP4), their scales and, for FP8 tensor experts, the per-expert static
/// activation scale (F32 per expert, host values: the loader's per-row copies of one scalar).
struct ExpertStack {
    w: Tensor,
    scale: Option<Tensor>,
    input_scale: Option<Vec<f32>>,
}

fn take_experts(
    d: &DecoderDims,
    weights: &mut LoadedWeights,
    layer: u32,
    proj: &str,
    dtype: ExpertDtype,
    [experts, n, k]: [usize; 3],
) -> Result<ExpertStack, ModelError> {
    let name = stacked_experts_name(layer, proj);
    if dtype == ExpertDtype::Bf16 {
        let w = d.take_weight(weights, &name, &[experts, n, k])?;
        return Ok(ExpertStack {
            w,
            scale: None,
            input_scale: None,
        });
    }
    let (wdt, cols) = if dtype == ExpertDtype::Mxfp4 {
        (DType::U8, k / 2)
    } else {
        (DType::F8E4M3, k)
    };
    let w = weights.take(&name)?;
    if w.dtype == d.act {
        return Err(refused(
            d,
            &format!("{name} decoded to BF16 at load (no kernel for its shape): quantized"),
        ));
    }
    if w.dtype != wdt || w.shape.as_slice() != [experts, n, cols] {
        return Err(invalid(format!(
            "{name} is {:?} {}, expected {:?} {}",
            w.shape.as_slice(),
            w.dtype.as_str(),
            [experts, n, cols],
            wdt.as_str()
        )));
    }
    let scale = weights.take(&format!("{name}_scale"))?;
    let check = |t: &Tensor, what: &str, shape: &[usize], dt: DType| {
        if t.shape.as_slice() == shape && t.dtype == dt {
            Ok(())
        } else {
            Err(invalid(format!(
                "{name}_{what} is {:?} {}, expected {shape:?} {}",
                t.shape.as_slice(),
                t.dtype.as_str(),
                dt.as_str()
            )))
        }
    };
    match dtype {
        ExpertDtype::Fp8Block => {
            let blocks = [experts, n.div_ceil(128), k.div_ceil(128)];
            check(&scale, "scale", &blocks, DType::F32)?;
            Ok(ExpertStack {
                w,
                scale: Some(scale),
                input_scale: None,
            })
        }
        ExpertDtype::Mxfp4 => {
            check(&scale, "scale", &[experts, n, k / 32], DType::U8)?;
            Ok(ExpertStack {
                w,
                scale: Some(scale),
                input_scale: None,
            })
        }
        _ => {
            // FP8 tensor: the loader stores a per-tensor scale once per row; the kernel takes
            // one per expert, so the rows of an expert must agree (a per-channel checkpoint
            // has no row).
            check(&scale, "scale", &[experts, n], DType::F32)?;
            let per_expert = |t: &Tensor| -> Result<Vec<f32>, ModelError> {
                let rows = read_f32(t)?;
                rows.chunks_exact(n)
                    .map(|r| {
                        if r.iter().all(|v| *v == r[0]) {
                            Ok(r[0])
                        } else {
                            Err(refused(d, &format!("{name}: per-channel FP8 scales,")))
                        }
                    })
                    .collect()
            };
            let scales = per_expert(&scale)?;
            let input = weights
                .tensors
                .remove(&format!("{name}_input_scale"))
                .ok_or_else(|| ModelError::MissingTensor(format!("{name}_input_scale")))?;
            check(&input, "input_scale", &[experts, n], DType::F32)?;
            let input_scale = read_f32(&input)?
                .chunks_exact(n)
                .map(|r| r.iter().copied().fold(f32::MIN, f32::max))
                .collect();
            let scale = upload_f32(w.storage.memory(), &scales)?;
            Ok(ExpertStack {
                w,
                scale: Some(scale),
                input_scale: Some(input_scale),
            })
        }
    }
}

fn read_f32(t: &Tensor) -> Result<Vec<f32>, ModelError> {
    let mut bytes = vec![0u8; t.shape.iter().product::<usize>() * F32];
    t.storage.copy_to_host(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(F32)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// A `[values.len()]` F32 tensor on `mem`.
fn upload_f32(mem: &Arc<dyn DeviceMemory>, values: &[f32]) -> Result<Tensor, ModelError> {
    let mut t = Tensor::empty(mem, &[values.len()], DType::F32)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    t.storage.copy_from_host(0, &bytes)?;
    Ok(t)
}

/// A fused gpt-oss expert tensor `name` of the checkpoint's stored `shape` as the kernels take
/// it: the same bytes, shaped `to` (`[experts, n, k/32, 16]` blocks as `[experts, n, k/2]`).
fn take_fused(
    weights: &mut LoadedWeights,
    name: &str,
    dtype: DType,
    shape: &[usize],
    to: &[usize],
) -> Result<Tensor, ModelError> {
    let mut t = weights.take(name)?;
    if t.dtype != dtype || t.shape.as_slice() != shape {
        return Err(invalid(format!(
            "{name} is {:?} {}, expected {shape:?} {}",
            t.shape.as_slice(),
            t.dtype.as_str(),
            dtype.as_str()
        )));
    }
    // Row-major strides of `to`.
    let mut strides = vec![1usize; to.len()];
    for i in (0..to.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * to[i + 1];
    }
    t.shape = to.into();
    t.strides = strides.as_slice().into();
    Ok(t)
}

/// Layer `prefix`'s gpt-oss parameters (the [`Style::GptOss`] layout): the router and its F32
/// bias, and the fused MXFP4 `gate_up_proj` and `down_proj` stacks with their biases, as the
/// OpenAI checkpoint stores them (`model.layers.<i>.mlp.experts.<proj>_{blocks,scales,bias}`;
/// row `2j` of `gate_up_proj` is gate column `j`, row `2j + 1` is up column `j`).
fn load_gpt_oss(
    d: &DecoderDims,
    moe: &MoeConfig,
    prefix: &str,
    weights: &mut LoadedWeights,
) -> Result<HookWeights, ModelError> {
    if d.ep.is_some() {
        return Err(invalid(
            "gpt-oss experts are not implemented under expert parallelism".into(),
        ));
    }
    let (e, h, inter) = (
        moe.num_experts as usize,
        d.hidden,
        moe.expert_intermediate as usize,
    );
    let p = format!("{prefix}.mlp.experts");
    let router_name = format!("{prefix}.mlp.router.weight");
    let router = d.take_weight(weights, &router_name, &[e, h])?;
    let bias_name = format!("{prefix}.mlp.router.bias");
    let router_bias = bf16_as_f32(weights.take(&bias_name)?, &[e], &bias_name)?;
    let mut proj = |name: &str, n: usize, k: usize| -> Result<[Tensor; 3], ModelError> {
        let data = take_fused(
            weights,
            &format!("{p}.{name}_blocks"),
            DType::U8,
            &[e, n, k / 32, 16],
            &[e, n, k / 2],
        )?;
        let scales = take_fused(
            weights,
            &format!("{p}.{name}_scales"),
            DType::U8,
            &[e, n, k / 32],
            &[e, n, k / 32],
        )?;
        let bias = d.take_weight(weights, &format!("{p}.{name}_bias"), &[e, n])?;
        Ok([data, scales, bias])
    };
    let [gu, gu_scale, gu_bias] = proj("gate_up_proj", 2 * inter, h)?;
    let [down, down_scale, down_bias] = proj("down_proj", h, inter)?;
    Ok(HookWeights::tensors(vec![
        router,
        router_bias,
        gu,
        gu_scale,
        gu_bias,
        down,
        down_scale,
        down_bias,
    ]))
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
            OpConfig::MoeRoute(route_cfg(d, &moe, self.style)),
        ];
        for layer in 0..d.layers {
            for run in local_runs(d, &moe, layer) {
                let op = OpConfig::MoeExperts(experts_cfg(d, &moe, run, self.style));
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
            + experts_workspace_bytes(d, &moe, max_batch_tokens, self.style)
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
                experts_workspace_bytes(d, &moe, t, self.style) as usize,
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
        if self.style == Style::GptOss {
            return load_gpt_oss(d, &moe, prefix, weights);
        }
        let inter = moe.expert_intermediate as usize;
        let experts = local_runs(d, &moe, layer as usize)
            .iter()
            .map(|(b, e)| (e - b) as usize)
            .sum();
        let dtype = expert_dtype(d.quant).map_err(|()| refused(d, "quantized"))?;
        let mut stack = |proj: &str, n: usize, k: usize| {
            take_experts(d, weights, layer, proj, dtype, [experts, n, k])
        };
        let (gate, up, down) = (
            stack("gate_proj", inter, d.hidden)?,
            stack("up_proj", inter, d.hidden)?,
            stack("down_proj", d.hidden, inter)?,
        );
        let router = weights.take(&format!("{prefix}.mlp.gate.weight"))?;
        if router.dtype != d.act {
            return Err(refused(
                d,
                &format!(
                    "{prefix}.mlp.gate (the router must stay BF16, add it to the ignore list): quantized"
                ),
            ));
        }
        let mut tensors = vec![router, gate.w, up.w, down.w];
        if dtype != ExpertDtype::Bf16 {
            tensors.extend([gate.scale, up.scale, down.scale].into_iter().flatten());
        }
        if dtype == ExpertDtype::Fp8Tensor {
            // Static activation scales: the gate and up projections read the same input, so one
            // scale per expert, the larger of the two (as a fused projection's).
            let (g, u) = (
                gate.input_scale.expect("fp8 tensor input scale"),
                up.input_scale.expect("fp8 tensor input scale"),
            );
            let both: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g.max(*u)).collect();
            let mem = Arc::clone(tensors[W_GATE].storage.memory());
            tensors.push(upload_f32(&mem, &both)?);
            let down = down.input_scale.expect("fp8 tensor input scale");
            tensors.push(upload_f32(&mem, &down)?);
        }
        Ok(HookWeights::tensors(tensors))
    }

    fn forward(&self, run: &LayerRun<'_>, w: &HookWeights) -> Result<(), ModelError> {
        let d = run.dims;
        let t = run.tokens;
        let moe = moe_of(d)?;
        let b = &run.ffn_buffers.tensors;
        let top_k = moe.experts_per_token as usize;
        run.linear(run.normed(), w.0[ROUTER].view(), rows(&b[ROUTER_LOGITS], t))?;
        run.trace("router_logits", &rows(&b[ROUTER_LOGITS], t))?;
        let route = route_cfg(d, &moe, self.style);
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
                router_bias: (self.style == Style::GptOss).then(|| w.0[G_ROUTER_BIAS].view()),
            })
        })?;
        run.trace_routing(
            &rows(&b[ROUTER_LOGITS], t),
            &rows(&b[TOPK_IDS], t),
            &rows(&b[TOPK_WEIGHTS], t),
            route.bf16_logits,
        )?;
        // gpt-oss's trace carries the routing too (the reference trace of the tiny checkpoint
        // records the selected experts).
        if d.ep.is_some() || self.style == Style::GptOss {
            run.trace("topk_ids", &rows(&b[TOPK_IDS], t))?;
            run.trace("topk_weights", &rows(&b[TOPK_WEIGHTS], t))?;
        }
        let calls: Vec<MoeExpertsConfig> = local_runs(d, &moe, run.layer)
            .into_iter()
            .map(|r| experts_cfg(d, &moe, r, self.style))
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
        let workspace_len = experts_workspace_bytes(d, &moe, t, self.style) as usize;
        // Each call adds its experts' weighted outputs in ascending expert order, so the calls
        // in ascending order add a token's experts in ascending order (one device's).
        let mut local = 0;
        for experts in calls {
            let n = experts.num_local_experts() as usize;
            let weight = |i: usize| w.0[i].view().rows(local, n);
            let quantized = |i: usize| w.0.get(i).map(|t| t.view().rows(local, n));
            // gpt-oss: the fused gate/up stack stands for both (`w_up` is unused when the gate
            // and up rows are interleaved).
            let (w_gate, w_up, w_down, extra) = match self.style {
                Style::SwiGlu => (
                    weight(W_GATE),
                    weight(W_UP),
                    weight(W_DOWN),
                    MoeExpertsExtra {
                        w_gate_scale: quantized(S_GATE),
                        w_up_scale: quantized(S_UP),
                        w_down_scale: quantized(S_DOWN),
                        act_static_scale: quantized(ACT_GATE_UP),
                        act_static_scale_down: quantized(ACT_DOWN),
                        ..Default::default()
                    },
                ),
                Style::GptOss => (
                    weight(G_GATE_UP),
                    weight(G_GATE_UP),
                    weight(G_DOWN),
                    MoeExpertsExtra {
                        w_gate_scale: Some(weight(G_GATE_UP_SCALE)),
                        w_down_scale: Some(weight(G_DOWN_SCALE)),
                        b_gate: Some(weight(G_GATE_UP_BIAS)),
                        b_down: Some(weight(G_DOWN_BIAS)),
                        ..Default::default()
                    },
                ),
            };
            run.op(OpConfig::MoeExperts(experts), || {
                run.registry
                    .moe_experts(&experts)
                    .experts(&mut MoeExpertsContext {
                        cfg: experts,
                        x: run.normed(),
                        w_gate,
                        w_up,
                        w_down,
                        sorted_rows: sorted_rows.clone(),
                        expert_offsets: offsets.clone(),
                        topk_weights: rows(&b[TOPK_WEIGHTS], t),
                        host_expert_offsets: &host_offsets,
                        out: run.ffn_out(),
                        workspace: Some(run.ffn_buffers.scratch[WORKSPACE].slice(0, workspace_len)),
                        extra,
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
                let c = experts_cfg(d, &moe, run, self.style);
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
