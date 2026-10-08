//! Mixture-of-experts reference ops: `moe_route` (logits optionally rounded to BF16, F32
//! softmax, the top-k set `torch.topk` selects, optional renormalisation, rows grouped by
//! expert) and `moe_experts` (per-expert SwiGLU with the Hugging Face BF16 roundings, weighted
//! accumulation in ascending expert order).
use turbine_core::types::DType;

use super::quant;
use super::topk::topk_then_softmax;
use super::{
    CpuReference, expect_rank, expect_shape, invalid, is_float, load, load_bytes, load_i32, math,
    round_to, store, store_i32, torch_topk,
};
use crate::KernelError;
use crate::ops::{
    ExpertActivation, ExpertDtype, MoeExpertsConfig, MoeExpertsContext, MoeKernel, MoeRouteConfig,
    MoeRouteContext, RouteMode,
};
use crate::quant::{ActQuantDesc, QuantSchemeDesc};

/// Softmax of one row of router logits in f32 (max-subtracted, sequential sum).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let mut sum = 0f32;
    for &v in &p {
        sum += v;
    }
    for v in &mut p {
        *v /= sum;
    }
    p
}

/// Checks `v` is a dense I32 or F32 view of `want` shape.
fn expect_typed(
    name: &str,
    v: &turbine_tensor::TensorView<'_>,
    want: &[usize],
    dtype: DType,
) -> Result<(), KernelError> {
    expect_shape(name, v, want)?;
    if v.dtype != dtype {
        return Err(invalid(format!(
            "{name} must be {}, got {}",
            dtype.as_str(),
            v.dtype.as_str()
        )));
    }
    Ok(())
}

/// Validates `expert_offsets` (`[num_experts + 1]`, from 0 to `rows`, non-decreasing).
fn check_offsets(offsets: &[i32], rows: usize) -> Result<(), KernelError> {
    let ends_ok = offsets.first() == Some(&0)
        && offsets
            .last()
            .is_some_and(|&l| usize::try_from(l).is_ok_and(|l| l == rows));
    if !ends_ok || offsets.windows(2).any(|w| w[1] < w[0]) {
        return Err(invalid(format!(
            "expert_offsets must rise from 0 to {rows}, got {offsets:?}"
        )));
    }
    Ok(())
}

impl MoeKernel for CpuReference {
    fn supports_route(&self, cfg: &MoeRouteConfig) -> bool {
        cfg.top_k > 0 && cfg.top_k <= cfg.num_experts
    }

    fn supports_experts(&self, cfg: &MoeExpertsConfig) -> bool {
        cfg.hidden > 0
            && cfg.inter > 0
            && cfg.top_k > 0
            && cfg.top_k <= cfg.num_experts
            && cfg.expert_begin < cfg.expert_end
            && cfg.expert_end <= cfg.num_experts
            && is_float(cfg.dtype)
            && (cfg.weight_dtype != ExpertDtype::Mxfp4
                || (cfg.hidden.is_multiple_of(32) && cfg.inter.is_multiple_of(32)))
            && match cfg.activation {
                ExpertActivation::SwiGlu => true,
                ExpertActivation::GptOssClamp { limit } => limit > 0.0,
            }
            && cfg.shared_inter != Some(0)
    }

    fn implementation_route(&self, _cfg: &MoeRouteConfig) -> String {
        "cpu_moe_route".into()
    }

    fn implementation_experts(&self, _cfg: &MoeExpertsConfig) -> String {
        "cpu_moe_experts_f32acc".into()
    }

    fn route(&self, ctx: &mut MoeRouteContext<'_>) -> Result<(), KernelError> {
        if !self.supports_route(&ctx.cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference moe_route {}", ctx.cfg),
            });
        }
        let experts = ctx.cfg.num_experts as usize;
        let k = ctx.cfg.top_k as usize;
        expect_rank("router_logits", &ctx.router_logits, 2)?;
        let tokens = ctx.router_logits.shape[0];
        expect_typed(
            "router_logits",
            &ctx.router_logits,
            &[tokens, experts],
            DType::F32,
        )?;
        expect_typed("topk_ids", &ctx.topk_ids, &[tokens, k], DType::I32)?;
        expect_typed("topk_weights", &ctx.topk_weights, &[tokens, k], DType::F32)?;
        expect_typed("sorted_rows", &ctx.sorted_rows, &[tokens * k], DType::I32)?;
        expect_typed(
            "expert_offsets",
            &ctx.expert_offsets,
            &[experts + 1],
            DType::I32,
        )?;

        let bias = match (&ctx.router_bias, ctx.cfg.bias) {
            (Some(b), true) => {
                expect_typed("router_bias", b, &[experts], DType::F32)?;
                Some(load(b)?)
            }
            (None, false) => None,
            _ => {
                return Err(invalid(
                    "router_bias must be passed exactly when the config has bias".into(),
                ));
            }
        };
        let logits = load(&ctx.router_logits)?;
        let mut ids = Vec::with_capacity(tokens * k);
        let mut weights = Vec::with_capacity(tokens * k);
        for row in logits.chunks_exact(experts) {
            // The router linear's output: the bias is part of it, so it is rounded with it.
            let mut row: Vec<f32> = match &bias {
                Some(b) => row.iter().zip(b).map(|(&l, &b)| l + b).collect(),
                None => row.to_vec(),
            };
            if ctx.cfg.bf16_logits {
                for l in &mut row {
                    *l = round_to(DType::BF16, *l);
                }
            }
            if ctx.cfg.mode == RouteMode::TopkThenSoftmax {
                let (chosen, w) = topk_then_softmax(&row, None, k);
                ids.extend(chosen);
                weights.extend(w);
                continue;
            }
            let p = softmax(&row);
            let chosen = torch_topk(&p, k);
            let mut sum = 0f32;
            for &e in &chosen {
                sum += p[e];
            }
            for &e in &chosen {
                ids.push(e);
                weights.push(if ctx.cfg.renormalize {
                    p[e] / sum
                } else {
                    p[e]
                });
            }
        }

        // Counting sort of rows `token · k + slot` by expert, stable in row order.
        let mut offsets = vec![0usize; experts + 1];
        for &e in &ids {
            offsets[e + 1] += 1;
        }
        for e in 0..experts {
            offsets[e + 1] += offsets[e];
        }
        let mut cursor = offsets.clone();
        let mut sorted = vec![0i32; ids.len()];
        for (row, &e) in ids.iter().enumerate() {
            sorted[cursor[e]] = row as i32;
            cursor[e] += 1;
        }

        let as_i32 =
            |v: usize| i32::try_from(v).map_err(|_| invalid(format!("{v} does not fit int32_t")));
        let ids = ids.into_iter().map(as_i32).collect::<Result<Vec<_>, _>>()?;
        let offsets = offsets
            .into_iter()
            .map(as_i32)
            .collect::<Result<Vec<_>, _>>()?;
        store_i32(&ctx.topk_ids, &ids)?;
        store(&ctx.topk_weights, &weights)?;
        store_i32(&ctx.sorted_rows, &sorted)?;
        store_i32(&ctx.expert_offsets, &offsets)
    }

    /// The reference reads the group sizes from `expert_offsets` itself.
    fn needs_host_offsets(&self, _cfg: &MoeExpertsConfig, _routed_rows: usize) -> bool {
        false
    }

    fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        if !self.supports_experts(&cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference moe_experts {cfg}"),
            });
        }
        let (h, inter) = (cfg.hidden as usize, cfg.inter as usize);
        let (experts, k) = (cfg.num_experts as usize, cfg.top_k as usize);
        let dt = cfg.dtype;
        expect_rank("x", &ctx.x, 2)?;
        let tokens = ctx.x.shape[0];
        expect_shape("x", &ctx.x, &[tokens, h])?;
        expect_shape("out", &ctx.out, &[tokens, h])?;
        for (name, view, shape, dtype) in ctx.weight_operands()? {
            expect_typed(name, view, &shape, dtype)?;
        }
        expect_typed("sorted_rows", &ctx.sorted_rows, &[tokens * k], DType::I32)?;
        expect_typed(
            "expert_offsets",
            &ctx.expert_offsets,
            &[experts + 1],
            DType::I32,
        )?;
        expect_typed("topk_weights", &ctx.topk_weights, &[tokens, k], DType::F32)?;

        let offsets = load_i32(&ctx.expert_offsets)?;
        check_offsets(&offsets, tokens * k)?;
        // The host copy is optional here (`needs_host_offsets` is false); one that is passed
        // must agree with the device offsets.
        if !ctx.host_expert_offsets.is_empty() && ctx.host_expert_offsets != offsets.as_slice() {
            return Err(invalid(format!(
                "host_expert_offsets {:?} differ from expert_offsets {offsets:?}",
                ctx.host_expert_offsets
            )));
        }
        let sorted = load_i32(&ctx.sorted_rows)?;
        let weights = load(&ctx.topk_weights)?;
        let x = load(&ctx.x)?;
        let mut out = load(&ctx.out)?;
        let round = |v: f32| round_to(dt, v);
        let gate_rows = if cfg.gate_up_interleaved {
            2 * inter
        } else {
            inter
        };
        let ex = &ctx.extra;
        let act_scales = ex.act_static_scale.as_ref().map(load).transpose()?;
        let act_scales_down = ex.act_static_scale_down.as_ref().map(load).transpose()?;

        for (li, e) in (cfg.expert_begin as usize..cfg.expert_end as usize).enumerate() {
            let group = &sorted[offsets[e] as usize..offsets[e + 1] as usize];
            if group.is_empty() {
                continue;
            }
            // Only the weights of experts that received rows are read.
            let w_gate = expert_weight(
                &cfg,
                &ctx.w_gate,
                ex.w_gate_scale.as_ref(),
                li,
                gate_rows,
                h,
            )?;
            let w_up = if cfg.gate_up_interleaved {
                Vec::new()
            } else {
                expert_weight(&cfg, &ctx.w_up, ex.w_up_scale.as_ref(), li, inter, h)?
            };
            let w_down = expert_weight(&cfg, &ctx.w_down, ex.w_down_scale.as_ref(), li, h, inter)?;
            let bias = |b: &Option<turbine_tensor::TensorView<'_>>| {
                b.as_ref().map(|v| load(&v.rows(li, 1))).transpose()
            };
            let (b_gate, b_up, b_down) = (bias(&ex.b_gate)?, bias(&ex.b_up)?, bias(&ex.b_down)?);
            let mut rows = Vec::with_capacity(group.len());
            let mut xs = Vec::with_capacity(group.len() * h);
            for &r in group {
                let r = usize::try_from(r)
                    .ok()
                    .filter(|&r| r < tokens * k)
                    .ok_or_else(|| invalid(format!("sorted_rows entry {r} is out of range")))?;
                rows.push(r);
                xs.extend_from_slice(&x[(r / k) * h..(r / k + 1) * h]);
            }
            let n = rows.len();
            // W8A8: the inputs of the quantized projections are FP8 round trips.
            let w8a8 = cfg.weight_dtype == ExpertDtype::Fp8Tensor && act_scales.is_some();
            if let (true, Some(s)) = (w8a8, &act_scales) {
                quant::quantize_dequantize_activations(
                    &mut xs,
                    n,
                    h,
                    ActQuantDesc::Fp8Tensor,
                    s[li],
                );
            }
            let proj = |w: &[f32], bias: Option<&[f32]>, cols: usize, depth: usize, a: &[f32]| {
                let shape = math::GemmShape {
                    m: n,
                    n: cols,
                    k: depth,
                    trans_b: true,
                };
                let mut y = math::gemm(a, w, &[], &shape, 1.0, 0.0);
                if let Some(b) = bias {
                    for row in y.chunks_exact_mut(cols) {
                        for (v, &b) in row.iter_mut().zip(b) {
                            *v += b;
                        }
                    }
                }
                y.into_iter().map(round).collect::<Vec<f32>>()
            };
            let (gate, up) = if cfg.gate_up_interleaved {
                let gu = proj(&w_gate, b_gate.as_deref(), gate_rows, h, &xs);
                let gate = gu.chunks_exact(2).map(|p| p[0]).collect::<Vec<f32>>();
                let up = gu.chunks_exact(2).map(|p| p[1]).collect::<Vec<f32>>();
                (gate, up)
            } else {
                (
                    proj(&w_gate, b_gate.as_deref(), inter, h, &xs),
                    proj(&w_up, b_up.as_deref(), inter, h, &xs),
                )
            };
            let mut act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| match cfg.activation {
                    ExpertActivation::GptOssClamp { limit } => gptoss_glu(round, g, u, limit),
                    _ => round(round(math::silu(g)) * u),
                })
                .collect();
            if w8a8 {
                let (mode, scale) = match &act_scales_down {
                    Some(s) => (ActQuantDesc::Fp8Tensor, s[li]),
                    None => (ActQuantDesc::Fp8Token, 0.0),
                };
                quant::quantize_dequantize_activations(&mut act, n, inter, mode, scale);
            }
            let down = proj(&w_down, b_down.as_deref(), h, inter, &act);
            for (i, &r) in rows.iter().enumerate() {
                let t = r / k;
                let w = round(weights[r]);
                for (o, &y) in out[t * h..(t + 1) * h]
                    .iter_mut()
                    .zip(&down[i * h..(i + 1) * h])
                {
                    *o = round(*o + round(y * w));
                }
            }
        }

        if let Some(si) = cfg.shared_inter {
            let si = si as usize;
            let shared = |v: &Option<turbine_tensor::TensorView<'_>>| {
                v.as_ref()
                    .map(load)
                    .transpose()?
                    .ok_or_else(|| invalid("shared expert weight missing".into()))
            };
            let (sg, su, sd) = (
                shared(&ex.shared_w_gate)?,
                shared(&ex.shared_w_up)?,
                shared(&ex.shared_w_down)?,
            );
            let gate_w = ex.shared_gate_w.as_ref().map(load).transpose()?;
            let proj = |w: &[f32], cols: usize, depth: usize, a: &[f32]| {
                let shape = math::GemmShape {
                    m: tokens,
                    n: cols,
                    k: depth,
                    trans_b: true,
                };
                math::gemm(a, w, &[], &shape, 1.0, 0.0)
                    .into_iter()
                    .map(round)
                    .collect::<Vec<f32>>()
            };
            let gate = proj(&sg, si, h, &x);
            let up = proj(&su, si, h, &x);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| round(round(math::silu(g)) * u))
                .collect();
            let down = proj(&sd, h, si, &act);
            for t in 0..tokens {
                let scale = gate_w.as_ref().map_or(1.0, |gw| {
                    let mut dot = 0f32;
                    for (a, b) in x[t * h..(t + 1) * h].iter().zip(gw) {
                        dot += a * b;
                    }
                    round(1.0 / (1.0 + (-round(dot)).exp()))
                });
                for (o, &y) in out[t * h..(t + 1) * h]
                    .iter_mut()
                    .zip(&down[t * h..(t + 1) * h])
                {
                    *o = round(*o + round(y * scale));
                }
            }
        }
        store(&ctx.out, &out)
    }
}

/// The gpt-oss gated activation of one element: gate clamped to `≤ limit`, up to `±limit`,
/// `(up + 1) · gate · σ(1.702 · gate)`, rounded after each step as the BF16 model's ops are.
fn gptoss_glu(round: impl Fn(f32) -> f32, gate: f32, up: f32, limit: f32) -> f32 {
    let gate = gate.min(limit);
    let up = up.clamp(-limit, limit);
    let sig = round(1.0 / (1.0 + (-round(gate * 1.702)).exp()));
    let glu = round(gate * sig);
    round(round(up + 1.0) * glu)
}

/// One expert's `[n, k]` weight as F32: BF16 weights as stored, quantized ones dequantized with
/// the expert's scales (the 6a layouts, [`quant::dequantize`]).
fn expert_weight(
    cfg: &MoeExpertsConfig,
    w: &turbine_tensor::TensorView<'_>,
    scale: Option<&turbine_tensor::TensorView<'_>>,
    li: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, KernelError> {
    let one = w.rows(li, 1);
    let scale = |what: &str| {
        scale
            .map(|s| s.rows(li, 1))
            .ok_or_else(|| invalid(format!("{what} experts need their scales")))
    };
    let scheme = match cfg.weight_dtype {
        ExpertDtype::Bf16 => return load(&one),
        ExpertDtype::Fp8Block => QuantSchemeDesc::Fp8Block {
            block_n: 128,
            block_k: 128,
        },
        ExpertDtype::Fp8Tensor => QuantSchemeDesc::Fp8Tensor,
        ExpertDtype::Mxfp4 => QuantSchemeDesc::Mxfp4,
    };
    let data = load_bytes(&one)?;
    let sv = scale(cfg.weight_dtype.as_str())?;
    let scales = if scheme == QuantSchemeDesc::Mxfp4 {
        quant::mxfp4_scales_as_f32(&load_bytes(&sv)?)
    } else {
        load(&sv)?
    };
    Ok(quant::dequantize(scheme, &data, &scales, None, n, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::test_util::*;
    use crate::ops::MoeExpertsExtra;

    /// Ties at the k-th place go where `torch.topk` puts them (here experts 2 and 3 of four
    /// equal weights, not the lower ids 0 and 1); the chosen set is listed by descending weight,
    /// lower id first among equals.
    #[test]
    fn top_k_breaks_ties_as_torch() {
        assert_eq!(torch_topk(&[0.25, 0.25, 0.25, 0.25], 2), [2, 3]);
        assert_eq!(torch_topk(&[0.1, 0.3, 0.3, 0.3], 3), [1, 2, 3]);
        let p = softmax(&[0.0, 0.0]);
        assert_eq!(p, [0.5, 0.5]);
    }

    #[test]
    fn moe_experts_accumulate_weighted_expert_outputs() {
        let mem = host();
        let moe = CpuReference.moe().expect("moe family");
        let (h, inter, experts, top_k) = (3usize, 2usize, 3usize, 2usize);
        // Two tokens: token 0 → experts (2, 0), token 1 → experts (0, 1); weights per slot.
        let sorted = [1, 2, 3, 0]; // e0 {1, 2}, e1 {3}, e2 {0}
        let offsets = [0, 2, 3, 4];
        let weights = [0.5f32, 0.25, 0.75, 0.125];
        let x_vals = [1.0f32, -2.0, 0.5, 0.25, 1.0, -1.0];
        let wg = seeded(1, experts * inter * h);
        let wu = seeded(2, experts * inter * h);
        let wd = seeded(3, experts * h * inter);
        let cfg = MoeExpertsConfig {
            hidden: h as u32,
            inter: inter as u32,
            num_experts: experts as u32,
            top_k: top_k as u32,
            expert_begin: 0,
            expert_end: experts as u32,
            dtype: DType::F32,
            ..Default::default()
        };
        assert!(moe.supports_experts(&cfg));
        assert_eq!(moe.implementation_experts(&cfg), "cpu_moe_experts_f32acc");
        let x = tensor(&mem, &[2, h], DType::F32, &x_vals);
        let out = tensor(&mem, &[2, h], DType::F32, &[10.0; 6]);
        let w_gate = tensor(&mem, &[experts, inter, h], DType::F32, &wg);
        let w_up = tensor(&mem, &[experts, inter, h], DType::F32, &wu);
        let w_down = tensor(&mem, &[experts, h, inter], DType::F32, &wd);
        let sorted_rows = i32_tensor(&mem, &sorted);
        let expert_offsets = i32_tensor(&mem, &offsets);
        let topk_weights = tensor(&mem, &[2, top_k], DType::F32, &weights);
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: w_gate.view(),
            w_up: w_up.view(),
            w_down: w_down.view(),
            sorted_rows: sorted_rows.view(),
            expert_offsets: expert_offsets.view(),
            topk_weights: topk_weights.view(),
            host_expert_offsets: &offsets,
            out: out.view(),
            workspace: None,
            extra: Default::default(),
        })
        .expect("experts");

        // Naive: out[t] = 10 + Σ over experts in ascending id of w · expert_e(x[t]).
        let expert = |e: usize, xt: &[f32]| -> Vec<f32> {
            let act: Vec<f32> = (0..inter)
                .map(|j| {
                    let dot = |w: &[f32]| (0..h).map(|c| xt[c] * w[(e * inter + j) * h + c]).sum();
                    let (g, u): (f32, f32) = (dot(&wg), dot(&wu));
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            (0..h)
                .map(|c| {
                    (0..inter)
                        .map(|j| act[j] * wd[(e * h + c) * inter + j])
                        .sum()
                })
                .collect()
        };
        let routes = [[(2usize, 0.5f32), (0, 0.25)], [(0, 0.75), (1, 0.125)]];
        let mut want = Vec::new();
        for (t, route) in routes.iter().enumerate() {
            let xt = &x_vals[t * h..(t + 1) * h];
            let mut acc = [10.0f32; 3];
            let mut by_expert = route.to_vec();
            by_expert.sort_by_key(|&(e, _)| e);
            for (e, w) in by_expert {
                for (a, y) in acc.iter_mut().zip(expert(e, xt)) {
                    *a += w * y;
                }
            }
            want.extend_from_slice(&acc);
        }
        assert_close(&load(&out.view()).expect("out"), &want, 1e-5);

        // The host copy of the offsets must agree with the device offsets.
        let err = moe
            .experts(&mut MoeExpertsContext {
                cfg,
                x: x.view(),
                w_gate: w_gate.view(),
                w_up: w_up.view(),
                w_down: w_down.view(),
                sorted_rows: sorted_rows.view(),
                expert_offsets: expert_offsets.view(),
                topk_weights: topk_weights.view(),
                host_expert_offsets: &[0, 1, 3, 4],
                out: out.view(),
                workspace: None,
                extra: Default::default(),
            })
            .expect_err("mismatched host offsets");
        assert!(matches!(err, KernelError::InvalidArgument { .. }), "{err}");

        // The reference reads the offsets from `expert_offsets` itself: it never needs the host
        // copy, and an empty one gives the same result as the full one.
        for rows in [0, 8, 512, 513, 1 << 20] {
            assert!(!moe.needs_host_offsets(&cfg, rows), "{rows} routed rows");
        }
        assert_eq!(cfg.routed_rows(3), 3 * top_k);
        let with_host = load(&out.view()).expect("out");
        let again = tensor(&mem, &[2, h], DType::F32, &[10.0; 6]);
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: w_gate.view(),
            w_up: w_up.view(),
            w_down: w_down.view(),
            sorted_rows: sorted_rows.view(),
            expert_offsets: expert_offsets.view(),
            topk_weights: topk_weights.view(),
            host_expert_offsets: &[],
            out: again.view(),
            workspace: None,
            extra: Default::default(),
        })
        .expect("experts without host offsets");
        assert_eq!(load(&again.view()).expect("out"), with_host);
    }
    fn bytes_tensor(
        mem: &Arc<dyn DeviceMemory>,
        shape: &[usize],
        dtype: DType,
        b: &[u8],
    ) -> Tensor {
        let mut t = Tensor::empty(mem, shape, dtype).expect("alloc");
        t.storage.copy_from_host(0, b).expect("upload");
        t
    }

    /// Routes `tokens` seeded logit rows (softmax then top-k, renormalised) over `experts`.
    fn route(
        mem: &Arc<dyn DeviceMemory>,
        tokens: usize,
        experts: usize,
        top_k: usize,
        seed: u64,
    ) -> (Tensor, Tensor, Tensor) {
        let moe = CpuReference.moe().expect("moe family");
        let logits = tensor(
            mem,
            &[tokens, experts],
            DType::F32,
            &seeded(seed, tokens * experts),
        );
        let ids = Tensor::empty(mem, &[tokens, top_k], DType::I32).expect("alloc");
        let weights = Tensor::empty(mem, &[tokens, top_k], DType::F32).expect("alloc");
        let sorted = Tensor::empty(mem, &[tokens * top_k], DType::I32).expect("alloc");
        let offsets = Tensor::empty(mem, &[experts + 1], DType::I32).expect("alloc");
        moe.route(&mut MoeRouteContext {
            cfg: MoeRouteConfig {
                num_experts: experts as u32,
                top_k: top_k as u32,
                renormalize: true,
                ..Default::default()
            },
            router_logits: logits.view(),
            topk_ids: ids.view(),
            topk_weights: weights.view(),
            sorted_rows: sorted.view(),
            expert_offsets: offsets.view(),
            router_bias: None,
        })
        .expect("route");
        (weights, sorted, offsets)
    }

    /// Quantizes an `n × k` matrix: `(data, f32 scales, E8M0 scales)`.
    fn quantize(dt: ExpertDtype, w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
        let mut data = vec![
            0u8;
            if dt == ExpertDtype::Mxfp4 {
                n * k / 2
            } else {
                n * k
            }
        ];
        match dt {
            ExpertDtype::Fp8Block | ExpertDtype::Fp8Tensor => {
                let (bn, bk) = if dt == ExpertDtype::Fp8Block {
                    (128, 128)
                } else {
                    (n, k)
                };
                let (nb, kb) = (n.div_ceil(bn), k.div_ceil(bk));
                let mut scales = vec![0f32; nb * kb];
                for (i, s) in scales.iter_mut().enumerate() {
                    let (bi, bj) = (i / kb, i % kb);
                    let mut amax = 0f32;
                    for r in bi * bn..((bi + 1) * bn).min(n) {
                        for c in bj * bk..((bj + 1) * bk).min(k) {
                            amax = amax.max(w[r * k + c].abs());
                        }
                    }
                    *s = (amax / quant::FP8_E4M3_MAX).max(quant::FP8_MIN_SCALE);
                }
                for r in 0..n {
                    for c in 0..k {
                        let s = scales[(r / bn) * kb + c / bk];
                        data[r * k + c] = quant::fp8_e4m3_round(w[r * k + c] / s);
                    }
                }
                (data, scales, Vec::new())
            }
            ExpertDtype::Mxfp4 => {
                let mut e8 = vec![0u8; n * k / 32];
                for r in 0..n {
                    for g in 0..k / 32 {
                        let mut grp = [0f32; 32];
                        grp.copy_from_slice(&w[r * k + g * 32..r * k + (g + 1) * 32]);
                        let (packed, e) = quant::mxfp4_quantize_group(&grp);
                        let at = (r * k + g * 32) / 2;
                        data[at..at + 16].copy_from_slice(&packed);
                        e8[r * (k / 32) + g] = e;
                    }
                }
                (data, Vec::new(), e8)
            }
            _ => unreachable!("quantized dtypes only"),
        }
    }

    /// Runs `experts` over the shared routing with `out` starting at zero.
    #[allow(clippy::too_many_arguments)]
    fn run_experts(
        mem: &Arc<dyn DeviceMemory>,
        cfg: MoeExpertsConfig,
        x: &Tensor,
        w: [&Tensor; 3],
        routing: &(Tensor, Tensor, Tensor),
        extra: MoeExpertsExtra<'_>,
    ) -> Result<Vec<f32>, KernelError> {
        let moe = CpuReference.moe().expect("moe family");
        let (tokens, h) = (x.shape[0], x.shape[1]);
        let out = tensor(mem, &[tokens, h], DType::F32, &vec![0.0; tokens * h]);
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: w[0].view(),
            w_up: w[1].view(),
            w_down: w[2].view(),
            sorted_rows: routing.1.view(),
            expert_offsets: routing.2.view(),
            topk_weights: routing.0.view(),
            host_expert_offsets: &[],
            out: out.view(),
            workspace: None,
            extra,
        })?;
        load(&out.view())
    }

    /// Every expert dtype on a seeded 8-expert problem equals the BF16-weight reference over the
    /// dequantized weights; with `act_static_scale` the FP8-tensor form is W8A8 and stays within
    /// the FP8 activation error of the reference over FP8-round-tripped inputs.
    #[test]
    fn quant_experts_match_dequantized() {
        let mem = host();
        let (h, inter, experts, top_k, tokens) = (128usize, 64usize, 8usize, 2usize, 5usize);
        let x = tensor(&mem, &[tokens, h], DType::F32, &seeded(7, tokens * h));
        let routing = route(&mem, tokens, experts, top_k, 11);
        let shapes = [(inter, h), (inter, h), (h, inter)];
        for dt in [
            ExpertDtype::Fp8Block,
            ExpertDtype::Fp8Tensor,
            ExpertDtype::Mxfp4,
        ] {
            let mx = dt == ExpertDtype::Mxfp4;
            let (mut data, mut scale_t, mut deq) = (Vec::new(), Vec::new(), Vec::new());
            for (pi, &(n, k)) in shapes.iter().enumerate() {
                let (mut d, mut s_f32, mut s_e8, mut dq) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for e in 0..experts {
                    let w: Vec<f32> = seeded(100 * pi as u64 + e as u64 + 1, n * k)
                        .iter()
                        .map(|v| v * 0.5)
                        .collect();
                    let (bytes, sf, se) = quantize(dt, &w, n, k);
                    let scales = if mx {
                        quant::mxfp4_scales_as_f32(&se)
                    } else {
                        sf.clone()
                    };
                    let scheme = match dt {
                        ExpertDtype::Fp8Block => QuantSchemeDesc::Fp8Block {
                            block_n: 128,
                            block_k: 128,
                        },
                        ExpertDtype::Fp8Tensor => QuantSchemeDesc::Fp8Tensor,
                        _ => QuantSchemeDesc::Mxfp4,
                    };
                    dq.extend(quant::dequantize(scheme, &bytes, &scales, None, n, k));
                    d.extend(bytes);
                    s_f32.extend(sf);
                    s_e8.extend(se);
                }
                let sshape: Vec<usize> = match dt {
                    ExpertDtype::Fp8Block => vec![experts, n.div_ceil(128), k.div_ceil(128)],
                    ExpertDtype::Fp8Tensor => vec![experts],
                    _ => vec![experts, n, k / 32],
                };
                let (wdt, kk) = if mx {
                    (DType::U8, k / 2)
                } else {
                    (DType::F8E4M3, k)
                };
                data.push(bytes_tensor(&mem, &[experts, n, kk], wdt, &d));
                scale_t.push(if mx {
                    bytes_tensor(&mem, &sshape, DType::U8, &s_e8)
                } else {
                    tensor(&mem, &sshape, DType::F32, &s_f32)
                });
                deq.push(tensor(&mem, &[experts, n, k], DType::F32, &dq));
            }
            let base = MoeExpertsConfig {
                hidden: h as u32,
                inter: inter as u32,
                num_experts: experts as u32,
                top_k: top_k as u32,
                expert_begin: 0,
                expert_end: experts as u32,
                dtype: DType::F32,
                ..Default::default()
            };
            let want = run_experts(
                &mem,
                base,
                &x,
                [&deq[0], &deq[1], &deq[2]],
                &routing,
                MoeExpertsExtra::default(),
            )
            .expect("reference");
            assert!(
                want.iter().any(|v| v.abs() > 1e-3),
                "{dt:?}: degenerate reference"
            );
            let cfg = MoeExpertsConfig {
                weight_dtype: dt,
                ..base
            };
            let extra = || MoeExpertsExtra {
                w_gate_scale: Some(scale_t[0].view()),
                w_up_scale: Some(scale_t[1].view()),
                w_down_scale: Some(scale_t[2].view()),
                ..Default::default()
            };
            let w = [&data[0], &data[1], &data[2]];
            let got = run_experts(&mem, cfg, &x, w, &routing, extra()).expect("quantized");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-2 * w.abs() + 1e-4,
                    "{dt:?} element {i}: got {g}, want {w}"
                );
            }
            // A missing scale is refused, not read.
            let mut missing = extra();
            missing.w_up_scale = None;
            let err = run_experts(&mem, cfg, &x, w, &routing, missing).expect_err("missing scale");
            assert!(matches!(err, KernelError::InvalidArgument { .. }), "{err}");

            if dt == ExpertDtype::Fp8Tensor {
                // W8A8 with a static scale of 0.05: x goes through FP8 with it, the input of
                // down per token. Against the reference over the pre-rounded x the only
                // difference is the down input's FP8 round trip.
                let act = tensor(&mem, &[experts], DType::F32, &[0.05; 8]);
                let mut w8 = extra();
                w8.act_static_scale = Some(act.view());
                let got8 = run_experts(&mem, cfg, &x, w, &routing, w8).expect("w8a8");
                assert!(
                    got8.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1e-6),
                    "the static scale changes the result"
                );
                let mut xq = load(&x.view()).expect("x");
                quant::quantize_dequantize_activations(
                    &mut xq,
                    tokens,
                    h,
                    ActQuantDesc::Fp8Tensor,
                    0.05,
                );
                let xq_t = tensor(&mem, &[tokens, h], DType::F32, &xq);
                let want8 = run_experts(
                    &mem,
                    base,
                    &xq_t,
                    [&deq[0], &deq[1], &deq[2]],
                    &routing,
                    MoeExpertsExtra::default(),
                )
                .expect("reference over rounded x");
                let rms = (want8.iter().map(|v| v * v).sum::<f32>() / want8.len() as f32).sqrt();
                for (g, w) in got8.iter().zip(&want8) {
                    assert!(
                        (g - w).abs() <= 0.1 * rms + 0.05 * w.abs(),
                        "W8A8: got {g}, want {w}"
                    );
                }

                // A static scale for down's input changes the result against the dynamic one
                // and stays within the same bound; down's scale alone is refused.
                let down = tensor(&mem, &[experts], DType::F32, &[0.5; 8]);
                let mut w8s = extra();
                w8s.act_static_scale = Some(act.view());
                w8s.act_static_scale_down = Some(down.view());
                let got8s = run_experts(&mem, cfg, &x, w, &routing, w8s).expect("static down");
                assert!(
                    got8s.iter().zip(&got8).any(|(a, b)| (a - b).abs() > 1e-6),
                    "the static down scale changes the result"
                );
                for (g, w) in got8s.iter().zip(&want8) {
                    assert!(
                        (g - w).abs() <= 0.2 * rms + 0.1 * w.abs(),
                        "W8A8 static down: got {g}, want {w}"
                    );
                }
                let mut only_down = extra();
                only_down.act_static_scale_down = Some(down.view());
                let err = run_experts(&mem, cfg, &x, w, &routing, only_down).expect_err("alone");
                assert!(matches!(err, KernelError::InvalidArgument { .. }), "{err}");
            }
        }
    }

    /// One token, one expert of weight 1, `hidden = 1`, `x = 1`: the expert's gate/up columns are
    /// the weights themselves. Gate 8.0 clamps to 7 (only the upper side clamps: −8.0 stays and
    /// 0.5 stays); up 9.0 clamps to 7, −9.0 to −7, 2.0 stays. Interleaved columns, biases on.
    #[test]
    fn gptoss_clamp_activation_matches_formula() {
        let mem = host();
        let moe = CpuReference.moe().expect("moe family");
        let (gate, up) = ([8.0f32, -8.0, 0.5], [9.0f32, -9.0, 2.0]);
        let mut w_gate = Vec::new();
        for j in 0..3 {
            w_gate.extend([gate[j], up[j]]);
        }
        let glu = |g: f64, u: f64| (u + 1.0) * g * (1.0 / (1.0 + (-1.702 * g).exp()));
        let want_sum: f64 = glu(7.0, 7.0) + glu(-8.0, -7.0) + glu(0.5, 2.0);
        // Hand-computed anchors: gate 7 saturates the sigmoid (8 · 7 · 0.999993 = 55.9996) and
        // gate 0.5 with up 2 gives 3 · 0.5 · σ(0.851) = 1.5 · 0.70076.
        assert!((glu(7.0, 7.0) - 55.9996).abs() < 1e-3, "{}", glu(7.0, 7.0));
        assert!(
            (glu(0.5, 2.0) - 1.5 * 0.700_76).abs() < 1e-3,
            "{}",
            glu(0.5, 2.0)
        );
        let w_gate_t = tensor(&mem, &[1, 6, 1], DType::F32, &w_gate);
        let w_up_t = tensor(&mem, &[1, 3, 1], DType::F32, &[0.0; 3]); // unused: interleaved
        let w_down_t = tensor(&mem, &[1, 1, 3], DType::F32, &[1.0; 3]);
        let b_gate_t = tensor(&mem, &[1, 6], DType::F32, &[0.0; 6]);
        let b_down_t = tensor(&mem, &[1, 1], DType::F32, &[0.25]);
        let x = tensor(&mem, &[1, 1], DType::F32, &[1.0]);
        let out = tensor(&mem, &[1, 1], DType::F32, &[0.0]);
        let sorted = i32_tensor(&mem, &[0]);
        let offsets = i32_tensor(&mem, &[0, 1]);
        let weights = tensor(&mem, &[1, 1], DType::F32, &[1.0]);
        let cfg = MoeExpertsConfig {
            hidden: 1,
            inter: 3,
            num_experts: 1,
            top_k: 1,
            expert_begin: 0,
            expert_end: 1,
            dtype: DType::F32,
            activation: ExpertActivation::GptOssClamp { limit: 7.0 },
            biases: true,
            gate_up_interleaved: true,
            ..Default::default()
        };
        assert!(moe.supports_experts(&cfg));
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: w_gate_t.view(),
            w_up: w_up_t.view(),
            w_down: w_down_t.view(),
            sorted_rows: sorted.view(),
            expert_offsets: offsets.view(),
            topk_weights: weights.view(),
            host_expert_offsets: &[],
            out: out.view(),
            workspace: None,
            extra: MoeExpertsExtra {
                b_gate: Some(b_gate_t.view()),
                b_down: Some(b_down_t.view()),
                ..Default::default()
            },
        })
        .expect("experts");
        let got = f64::from(load(&out.view()).expect("out")[0]);
        assert!(
            (got - (want_sum + 0.25)).abs() < 1e-3,
            "got {got}, want {}",
            want_sum + 0.25
        );
    }

    /// The shared expert adds `σ(x · shared_gate_w) · swiglu_expert(x)` to every token.
    #[test]
    fn shared_expert_gate_adds_sigmoid_scaled_output() {
        let mem = host();
        let (h, inter, experts, tokens, si) = (3usize, 2usize, 2usize, 2usize, 2usize);
        let x_vals = [1.0f32, -2.0, 0.5, 0.25, 1.0, -1.0];
        let x = tensor(&mem, &[tokens, h], DType::F32, &x_vals);
        let mk = |seed, shape: &[usize]| {
            tensor(
                &mem,
                shape,
                DType::F32,
                &seeded(seed, shape.iter().product()),
            )
        };
        let wg = mk(1, &[experts, inter, h]);
        let wu = mk(2, &[experts, inter, h]);
        let wd = mk(3, &[experts, h, inter]);
        let (sg, su, sd) = (seeded(4, si * h), seeded(5, si * h), seeded(6, h * si));
        let gate_w = [0.3f32, -0.2, 0.7];
        let sg_t = tensor(&mem, &[si, h], DType::F32, &sg);
        let su_t = tensor(&mem, &[si, h], DType::F32, &su);
        let sd_t = tensor(&mem, &[h, si], DType::F32, &sd);
        let gw_t = tensor(&mem, &[h], DType::F32, &gate_w);
        let routing = (
            tensor(&mem, &[tokens, 1], DType::F32, &[0.5, 0.25]),
            i32_tensor(&mem, &[0, 1]),
            i32_tensor(&mem, &[0, 1, 2]),
        );
        let base = MoeExpertsConfig {
            hidden: h as u32,
            inter: inter as u32,
            num_experts: experts as u32,
            top_k: 1,
            expert_begin: 0,
            expert_end: experts as u32,
            dtype: DType::F32,
            ..Default::default()
        };
        let w = [&wg, &wu, &wd];
        let without =
            run_experts(&mem, base, &x, w, &routing, MoeExpertsExtra::default()).expect("without");
        let with = run_experts(
            &mem,
            MoeExpertsConfig {
                shared_inter: Some(si as u32),
                ..base
            },
            &x,
            w,
            &routing,
            MoeExpertsExtra {
                shared_w_gate: Some(sg_t.view()),
                shared_w_up: Some(su_t.view()),
                shared_w_down: Some(sd_t.view()),
                shared_gate_w: Some(gw_t.view()),
                ..Default::default()
            },
        )
        .expect("with shared expert");
        for t in 0..tokens {
            let xt = &x_vals[t * h..(t + 1) * h];
            let dot = |w: &[f32]| (0..h).map(|c| xt[c] * w[c]).sum::<f32>();
            let act: Vec<f32> = (0..si)
                .map(|j| {
                    let (g, u) = (dot(&sg[j * h..(j + 1) * h]), dot(&su[j * h..(j + 1) * h]));
                    g * sigmoid(g) * u
                })
                .collect();
            let scale = sigmoid(dot(&gate_w));
            for c in 0..h {
                let y: f32 = (0..si).map(|j| act[j] * sd[c * si + j]).sum();
                let (a, b) = (with[t * h + c], without[t * h + c]);
                assert!(
                    (a - b - scale * y).abs() < 1e-5,
                    "token {t} col {c}: {a} - {b} vs {}",
                    scale * y
                );
            }
        }
    }
}
