//! Mixture-of-experts reference ops: `moe_route` (F32 softmax, top-k with ties to the lower
//! expert id, optional renormalisation, rows grouped by expert) and `moe_experts` (per-expert
//! SwiGLU with the Hugging Face BF16 roundings, weighted accumulation in ascending expert order).
use turbine_core::types::DType;

use super::{
    CpuReference, expect_rank, expect_shape, invalid, is_float, load, load_i32, math, round_to,
    store, store_i32,
};
use crate::KernelError;
use crate::ops::{MoeExpertsConfig, MoeExpertsContext, MoeKernel, MoeRouteConfig, MoeRouteContext};

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

/// The `top_k` largest entries of `p` in descending order; among equal values the lower index
/// comes first.
fn top_k(p: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..p.len()).collect();
    // Stable sort by descending value keeps ascending ids among ties.
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    order.truncate(k);
    order
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

        let logits = load(&ctx.router_logits)?;
        let mut ids = Vec::with_capacity(tokens * k);
        let mut weights = Vec::with_capacity(tokens * k);
        for row in logits.chunks_exact(experts) {
            let p = softmax(row);
            let chosen = top_k(&p, k);
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

    fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        if !self.supports_experts(&cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference moe_experts {cfg}"),
            });
        }
        let (h, inter) = (cfg.hidden as usize, cfg.inter as usize);
        let (experts, k) = (cfg.num_experts as usize, cfg.top_k as usize);
        let local = cfg.num_local_experts() as usize;
        let dt = cfg.dtype;
        expect_rank("x", &ctx.x, 2)?;
        let tokens = ctx.x.shape[0];
        expect_shape("x", &ctx.x, &[tokens, h])?;
        expect_shape("out", &ctx.out, &[tokens, h])?;
        expect_shape("w_gate", &ctx.w_gate, &[local, inter, h])?;
        expect_shape("w_up", &ctx.w_up, &[local, inter, h])?;
        expect_shape("w_down", &ctx.w_down, &[local, h, inter])?;
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
        if ctx.host_expert_offsets != offsets.as_slice() {
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

        for (li, e) in (cfg.expert_begin as usize..cfg.expert_end as usize).enumerate() {
            let group = &sorted[offsets[e] as usize..offsets[e + 1] as usize];
            if group.is_empty() {
                continue;
            }
            // Only the weights of experts that received rows are read.
            let w_gate = load(&ctx.w_gate.rows(li, 1))?;
            let w_up = load(&ctx.w_up.rows(li, 1))?;
            let w_down = load(&ctx.w_down.rows(li, 1))?;
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
            let proj = |w: &[f32], cols: usize, depth: usize, a: &[f32]| {
                let shape = math::GemmShape {
                    m: n,
                    n: cols,
                    k: depth,
                    trans_b: true,
                };
                math::gemm(a, w, &[], &shape, 1.0, 0.0)
                    .into_iter()
                    .map(round)
                    .collect::<Vec<f32>>()
            };
            let gate = proj(&w_gate, inter, h, &xs);
            let up = proj(&w_up, inter, h, &xs);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| round(round(math::silu(g)) * u))
                .collect();
            let down = proj(&w_down, h, inter, &act);
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
        store(&ctx.out, &out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_breaks_ties_toward_the_lower_id() {
        assert_eq!(top_k(&[0.25, 0.25, 0.25, 0.25], 2), [0, 1]);
        assert_eq!(top_k(&[0.1, 0.3, 0.3, 0.3], 3), [1, 2, 3]);
        let p = softmax(&[0.0, 0.0]);
        assert_eq!(p, [0.5, 0.5]);
    }
}
