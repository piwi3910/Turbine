//! Gated DeltaNet and causal conv1d on the cpu-reference provider (Phase 7 Task 30, ABI v2.18):
//! the numerical reference the HIP kernels are judged against.
//!
//! GDN is transformers' `torch_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel`:
//! `q` and `k` are L2-normalised per head (`x · 1 / sqrt(Σx² + 1e-6)`), `q` is scaled by
//! `1 / sqrt(k_dim)`, and per token the state is decayed by `exp(g)`, corrected by
//! `(v − Sᵀk) · beta` along `k` and read with `q`, all in f32. The state is `[v_dim, k_dim]` per
//! value head (k contiguous). The conv accumulates `bias + w0·x0 + w1·x1 + w2·x2 + w3·x3` left
//! to right in f32 and applies `x / (1 + exp(−x))`, rounding to the dtype once, for prefill and
//! decode alike, so both are bit-equal.
use std::ops::Range;

use turbine_tensor::TensorView;

use super::{CpuReference, expect_rank, expect_shape, invalid, is_float, load, load_i32, store};
use crate::KernelError;
use crate::ops::{
    Conv1dConfig, Conv1dContext, Conv1dKernel, GdnConfig, GdnContext, GdnKernel, GdnMode,
};

/// `1e-6`, transformers' `l2norm` epsilon.
const L2_EPS: f32 = 1e-6;

/// The token ranges of the sequences of a call: `cu_seqlens` (prefill) or one token per
/// sequence (decode).
fn sequence_ranges(
    mode: GdnMode,
    tokens: usize,
    seqs: usize,
    cu_seqlens: Option<&TensorView<'_>>,
) -> Result<Vec<Range<usize>>, KernelError> {
    match (mode, cu_seqlens) {
        (GdnMode::Decode, None) => {
            if tokens != seqs {
                return Err(invalid(format!(
                    "decode takes one token per sequence, got {tokens} tokens for {seqs} sequences"
                )));
            }
            Ok((0..seqs).map(|s| s..s + 1).collect())
        }
        (GdnMode::Decode, Some(_)) => Err(invalid("decode takes no cu_seqlens".into())),
        (GdnMode::Prefill, None) => Err(invalid("prefill needs cu_seqlens".into())),
        (GdnMode::Prefill, Some(cu)) => {
            expect_rank("cu_seqlens", cu, 1)?;
            expect_shape("cu_seqlens", cu, &[seqs + 1])?;
            let cu = load_i32(cu)?;
            let mut ranges = Vec::with_capacity(seqs);
            for w in cu.windows(2) {
                let (a, b) = (usize::try_from(w[0]), usize::try_from(w[1]));
                match (a, b) {
                    (Ok(a), Ok(b)) if a <= b && b <= tokens => ranges.push(a..b),
                    _ => return Err(invalid(format!("cu_seqlens {cu:?} for {tokens} tokens"))),
                }
            }
            Ok(ranges)
        }
    }
}

/// The slots of the sequences, each below `slots`.
fn slots_of(
    state_slots: &TensorView<'_>,
    seqs_expected: Option<usize>,
    slots: usize,
) -> Result<Vec<usize>, KernelError> {
    expect_rank("state_slots", state_slots, 1)?;
    let raw = load_i32(state_slots)?;
    if seqs_expected.is_some_and(|n| n != raw.len()) {
        return Err(invalid(format!(
            "{} state slots for {} sequences",
            raw.len(),
            seqs_expected.unwrap_or(0)
        )));
    }
    raw.iter()
        .map(|&s| {
            usize::try_from(s)
                .ok()
                .filter(|&s| s < slots)
                .ok_or_else(|| invalid(format!("state slot {s} outside {slots} slots")))
        })
        .collect()
}

/// Where each sequence's boundary rows start, and the total: sequence `s` owns
/// `len / boundary_tokens` rows.
fn boundary_rows(
    ranges: &[Range<usize>],
    boundary_tokens: u32,
    present: bool,
) -> Result<(Vec<usize>, usize), KernelError> {
    if !present {
        return Ok((vec![0; ranges.len()], 0));
    }
    if boundary_tokens == 0 {
        return Err(invalid("boundary buffer without boundary_tokens".into()));
    }
    let b = boundary_tokens as usize;
    let mut starts = Vec::with_capacity(ranges.len());
    let mut total = 0;
    for r in ranges {
        starts.push(total);
        total += r.len() / b;
    }
    Ok((starts, total))
}

fn l2_scale(x: &[f32]) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() + L2_EPS).sqrt()
}

impl GdnKernel for CpuReference {
    fn supports(&self, cfg: &GdnConfig) -> bool {
        is_float(cfg.dtype) && cfg.k_dim > 0 && cfg.v_dim > 0 && cfg.heads_per_key().is_some()
    }

    fn implementation(&self, _cfg: &GdnConfig) -> String {
        "cpu_gdn".into()
    }

    fn execute(&self, ctx: &mut GdnContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        if !GdnKernel::supports(self, &cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference does not run gdn {cfg}"),
            });
        }
        let (kh, vh, kd, vd) = (
            cfg.k_heads as usize,
            cfg.v_heads as usize,
            cfg.k_dim as usize,
            cfg.v_dim as usize,
        );
        let per_key = vh / kh;
        let tokens = ctx.q.shape.first().copied().unwrap_or(0);
        expect_shape("q", &ctx.q, &[tokens, kh, kd])?;
        expect_shape("k", &ctx.k, &[tokens, kh, kd])?;
        expect_shape("v", &ctx.v, &[tokens, vh, vd])?;
        expect_shape("g", &ctx.g, &[tokens, vh])?;
        expect_shape("beta", &ctx.beta, &[tokens, vh])?;
        expect_shape("out", &ctx.out, &[tokens, vh, vd])?;
        expect_rank("state", &ctx.state, 4)?;
        let slots = ctx.state.shape[0];
        expect_shape("state", &ctx.state, &[slots, vh, vd, kd])?;
        if ctx.state.dtype != turbine_core::types::DType::F32 {
            return Err(invalid("gdn state must be f32".into()));
        }
        let seqs = ctx.state_slots.shape.first().copied().unwrap_or(0);
        let ranges = sequence_ranges(cfg.mode, tokens, seqs, ctx.cu_seqlens.as_ref())?;
        let slot_of = slots_of(&ctx.state_slots, Some(ranges.len()), slots)?;
        let mut seen = slot_of.clone();
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return Err(invalid("two sequences share a state slot".into()));
        }
        let b = ctx.boundary_tokens as usize;
        if cfg.mode == GdnMode::Decode && ctx.boundary_states.is_some() {
            return Err(invalid("decode writes no boundary states".into()));
        }
        let (row0, nrows) =
            boundary_rows(&ranges, ctx.boundary_tokens, ctx.boundary_states.is_some())?;
        let state_elems = vh * vd * kd;
        if let Some(bs) = &ctx.boundary_states {
            expect_shape("boundary_states", bs, &[nrows, vh, vd, kd])?;
        }

        let (q, k, v) = (load(&ctx.q)?, load(&ctx.k)?, load(&ctx.v)?);
        let (g, beta) = (load(&ctx.g)?, load(&ctx.beta)?);
        let mut state = load(&ctx.state)?;
        let mut out = vec![0f32; tokens * vh * vd];
        let mut boundary = vec![0f32; nrows * state_elems];
        let scale = 1.0 / (kd as f32).sqrt();

        for (s, range) in ranges.iter().enumerate() {
            let base = slot_of[s] * state_elems;
            for h in 0..vh {
                let kv_head = h / per_key;
                let st = &mut state[base + h * vd * kd..base + (h + 1) * vd * kd];
                for t in range.clone() {
                    let q_t = &q[(t * kh + kv_head) * kd..][..kd];
                    let k_t = &k[(t * kh + kv_head) * kd..][..kd];
                    let (qs, ks) = (l2_scale(q_t) * scale, l2_scale(k_t));
                    let qn: Vec<f32> = q_t.iter().map(|x| x * qs).collect();
                    let kn: Vec<f32> = k_t.iter().map(|x| x * ks).collect();
                    let decay = g[t * vh + h].exp();
                    let beta_t = beta[t * vh + h];
                    for col in 0..vd {
                        let row = &mut st[col * kd..(col + 1) * kd];
                        let mut kv = 0f32;
                        for (sv, kx) in row.iter_mut().zip(&kn) {
                            *sv *= decay;
                            kv += *sv * kx;
                        }
                        let delta = (v[(t * vh + h) * vd + col] - kv) * beta_t;
                        let mut acc = 0f32;
                        for ((sv, kx), qx) in row.iter_mut().zip(&kn).zip(&qn) {
                            *sv += kx * delta;
                            acc += *sv * qx;
                        }
                        out[(t * vh + h) * vd + col] = acc;
                    }
                    let done = t - range.start + 1;
                    if b > 0 && nrows > 0 && done % b == 0 {
                        let j = done / b - 1;
                        let dst = (row0[s] + j) * state_elems + h * vd * kd;
                        boundary[dst..dst + vd * kd].copy_from_slice(st);
                    }
                }
            }
        }
        store(&ctx.out, &out)?;
        store(&ctx.state, &state)?;
        if let Some(bs) = &ctx.boundary_states {
            store(bs, &boundary)?;
        }
        Ok(())
    }
}

/// The ABI's only window.
const CONV_KERNEL: usize = 4;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Conv1dKernel for CpuReference {
    fn supports(&self, cfg: &Conv1dConfig) -> bool {
        is_float(cfg.dtype) && cfg.channels > 0 && cfg.kernel as usize == CONV_KERNEL
    }

    fn implementation(&self, _cfg: &Conv1dConfig) -> String {
        "cpu_conv1d".into()
    }

    fn execute(&self, ctx: &mut Conv1dContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        if !Conv1dKernel::supports(self, &cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference does not run conv1d {cfg}"),
            });
        }
        if ctx.activation != 0 && ctx.activation != 1 {
            return Err(invalid(format!("activation {}", ctx.activation)));
        }
        let (ch, tail) = (cfg.channels as usize, CONV_KERNEL - 1);
        let tokens = ctx.x.shape.first().copied().unwrap_or(0);
        expect_shape("x", &ctx.x, &[tokens, ch])?;
        expect_shape("weight", &ctx.weight, &[ch, CONV_KERNEL])?;
        expect_shape("out", &ctx.out, &[tokens, ch])?;
        expect_rank("conv_state", &ctx.conv_state, 3)?;
        let slots = ctx.conv_state.shape[0];
        expect_shape("conv_state", &ctx.conv_state, &[slots, ch, tail])?;
        let seqs = ctx.state_slots.shape.first().copied().unwrap_or(0);
        let ranges = sequence_ranges(cfg.mode, tokens, seqs, ctx.cu_seqlens.as_ref())?;
        let slot_of = slots_of(&ctx.state_slots, Some(ranges.len()), slots)?;
        if cfg.mode == GdnMode::Decode && ctx.boundary_conv_states.is_some() {
            return Err(invalid("decode writes no boundary states".into()));
        }
        let b = ctx.boundary_tokens as usize;
        if ctx.boundary_conv_states.is_some() && b < tail {
            return Err(invalid(format!(
                "boundary_tokens {b} below the {tail}-input conv tail"
            )));
        }
        let (row0, nrows) = boundary_rows(
            &ranges,
            ctx.boundary_tokens,
            ctx.boundary_conv_states.is_some(),
        )?;
        if let Some(bs) = &ctx.boundary_conv_states {
            expect_shape("boundary_conv_states", bs, &[nrows, ch, tail])?;
        }

        let x = load(&ctx.x)?;
        let w = load(&ctx.weight)?;
        let bias = match &ctx.bias {
            Some(bias) => {
                expect_shape("bias", bias, &[ch])?;
                load(bias)?
            }
            None => vec![0f32; ch],
        };
        let mut state = load(&ctx.conv_state)?;
        let mut out = vec![0f32; tokens * ch];
        let mut boundary = vec![0f32; nrows * ch * tail];
        for (s, range) in ranges.iter().enumerate() {
            for c in 0..ch {
                let sb = (slot_of[s] * ch + c) * tail;
                // The window's inputs, oldest first: the stored tail, then the sequence.
                let mut win: Vec<f32> = state[sb..sb + tail].to_vec();
                win.extend(range.clone().map(|t| x[t * ch + c]));
                for i in 0..range.len() {
                    let wc = &w[c * CONV_KERNEL..(c + 1) * CONV_KERNEL];
                    let mut acc = bias[c];
                    for j in 0..CONV_KERNEL {
                        acc += wc[j] * win[i + j];
                    }
                    if ctx.activation == 1 {
                        acc = silu(acc);
                    }
                    // Rounded once, when stored.
                    out[(range.start + i) * ch + c] = acc;
                    let done = i + 1;
                    if b > 0 && nrows > 0 && done % b == 0 {
                        let dst = ((row0[s] + done / b - 1) * ch + c) * tail;
                        boundary[dst..dst + tail].copy_from_slice(&win[done..done + tail]);
                    }
                }
                let end = win.len();
                state[sb..sb + tail].copy_from_slice(&win[end - tail..end]);
            }
        }
        store(&ctx.out, &out)?;
        store(&ctx.conv_state, &state)?;
        if let Some(bs) = &ctx.boundary_conv_states {
            store(bs, &boundary)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    const KH: usize = 2;
    const VH: usize = 4;
    const KD: usize = 16;
    const VD: usize = 16;

    struct Inputs {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        g: Vec<f32>,
        beta: Vec<f32>,
    }

    fn inputs(seed: u64, tokens: usize) -> Inputs {
        // bf16-exact q/k/v, decays in (-1, 0), betas in (0, 1).
        let bf = |x: f32| round_to(DType::BF16, x);
        Inputs {
            q: seeded(seed, tokens * KH * KD).into_iter().map(bf).collect(),
            k: seeded(seed + 1, tokens * KH * KD)
                .into_iter()
                .map(bf)
                .collect(),
            v: seeded(seed + 2, tokens * VH * VD)
                .into_iter()
                .map(bf)
                .collect(),
            g: seeded(seed + 3, tokens * VH)
                .into_iter()
                .map(|x| -0.5 * (x + 1.0) * 0.5)
                .collect(),
            beta: seeded(seed + 4, tokens * VH)
                .into_iter()
                .map(|x| (x + 1.0) * 0.5)
                .collect(),
        }
    }

    /// The textbook recurrence in f64 (transformers' `torch_recurrent_gated_delta_rule`, state
    /// `S[i][col]` with `i` along k), one head at a time.
    fn naive(
        inp: &Inputs,
        tokens: usize,
        state: &mut [f64], // [VH][KD][VD]
    ) -> Vec<f64> {
        let mut out = vec![0f64; tokens * VH * VD];
        for h in 0..VH {
            let kv_head = h / (VH / KH);
            let s = &mut state[h * KD * VD..(h + 1) * KD * VD];
            for t in 0..tokens {
                let norm = |x: &[f32]| -> Vec<f64> {
                    let n = x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() + 1e-6;
                    x.iter().map(|&v| f64::from(v) / n.sqrt()).collect()
                };
                let qn: Vec<f64> = norm(&inp.q[(t * KH + kv_head) * KD..][..KD])
                    .iter()
                    .map(|x| x / (KD as f64).sqrt())
                    .collect();
                let kn = norm(&inp.k[(t * KH + kv_head) * KD..][..KD]);
                let decay = f64::from(inp.g[t * VH + h]).exp();
                for x in s.iter_mut() {
                    *x *= decay;
                }
                let beta = f64::from(inp.beta[t * VH + h]);
                for col in 0..VD {
                    let kv: f64 = (0..KD).map(|i| s[i * VD + col] * kn[i]).sum();
                    let delta = (f64::from(inp.v[(t * VH + h) * VD + col]) - kv) * beta;
                    for i in 0..KD {
                        s[i * VD + col] += kn[i] * delta;
                    }
                    out[(t * VH + h) * VD + col] = (0..KD).map(|i| s[i * VD + col] * qn[i]).sum();
                }
            }
        }
        out
    }

    struct Run {
        out: Vec<f32>,
        state: Vec<f32>,
        boundary: Vec<f32>,
    }

    /// One prefill call over `inp[tokens]` as sequences of `lens`, sequence `s` in slot `s`
    /// of `slots` slots whose current content is `state` (`None` = zeros).
    fn run_prefill(
        inp: &Inputs,
        lens: &[usize],
        state: Option<&[f32]>,
        boundary_tokens: u32,
        mode: GdnMode,
    ) -> Run {
        let mem = host();
        let tokens: usize = lens.iter().sum();
        let slots = lens.len();
        let per_slot = VH * VD * KD;
        let zeros = vec![0f32; slots * per_slot];
        let state_t = tensor(
            &mem,
            &[slots, VH, VD, KD],
            DType::F32,
            state.unwrap_or(&zeros),
        );
        let mut cu = vec![0i32];
        for l in lens {
            cu.push(cu.last().copied().unwrap_or(0) + *l as i32);
        }
        let nrows: usize = if boundary_tokens > 0 {
            lens.iter().map(|l| l / boundary_tokens as usize).sum()
        } else {
            0
        };
        let q = tensor(
            &mem,
            &[tokens, KH, KD],
            DType::BF16,
            &inp.q[..tokens * KH * KD],
        );
        let k = tensor(
            &mem,
            &[tokens, KH, KD],
            DType::BF16,
            &inp.k[..tokens * KH * KD],
        );
        let v = tensor(
            &mem,
            &[tokens, VH, VD],
            DType::BF16,
            &inp.v[..tokens * VH * VD],
        );
        let g = tensor(&mem, &[tokens, VH], DType::F32, &inp.g[..tokens * VH]);
        let beta = tensor(&mem, &[tokens, VH], DType::F32, &inp.beta[..tokens * VH]);
        let out = Tensor::empty(&mem, &[tokens, VH, VD], DType::F32).expect("alloc");
        let slot_ids = i32_tensor(&mem, &(0..slots as i32).collect::<Vec<_>>());
        let cu_t = i32_tensor(&mem, &cu);
        let bs = (nrows > 0)
            .then(|| Tensor::empty(&mem, &[nrows, VH, VD, KD], DType::F32).expect("alloc"));
        CpuReference
            .gdn()
            .expect("gdn family")
            .execute(&mut GdnContext {
                cfg: GdnConfig {
                    k_heads: KH as u32,
                    v_heads: VH as u32,
                    k_dim: KD as u32,
                    v_dim: VD as u32,
                    mode,
                    dtype: DType::F32,
                },
                q: q.view(),
                k: k.view(),
                v: v.view(),
                g: g.view(),
                beta: beta.view(),
                state: state_t.view(),
                state_slots: slot_ids.view(),
                cu_seqlens: (mode == GdnMode::Prefill).then(|| cu_t.view()),
                out: out.view(),
                boundary_states: bs.as_ref().map(Tensor::view),
                boundary_tokens,
            })
            .expect("gdn");
        Run {
            out: load(&out.view()).expect("load"),
            state: load(&state_t.view()).expect("load"),
            boundary: bs.map_or_else(Vec::new, |t| load(&t.view()).expect("load")),
        }
    }

    /// The state of the kernel (`[VH][VD][KD]`) against the textbook one (`[VH][KD][VD]`).
    fn state_close(got: &[f32], want: &[f64], tol: f64) {
        for h in 0..VH {
            for col in 0..VD {
                for i in 0..KD {
                    let g = f64::from(got[(h * VD + col) * KD + i]);
                    let w = want[(h * KD + i) * VD + col];
                    assert!((g - w).abs() <= tol, "state h{h} col{col} i{i}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn recurrence_matches_naive() {
        for tokens in [1usize, 5, 40] {
            let inp = inputs(7, tokens);
            let mut want_state = vec![0f64; VH * KD * VD];
            let want = naive(&inp, tokens, &mut want_state);
            let run = run_prefill(&inp, &[tokens], None, 0, GdnMode::Prefill);
            for (i, (g, w)) in run.out.iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(*g) - w).abs() <= 2e-5,
                    "tokens {tokens} out {i}: {g} vs {w}"
                );
            }
            state_close(&run.state, &want_state, 2e-5);
        }
    }

    #[test]
    fn decode_steps_equal_prefill() {
        // 6 tokens one at a time through decode (one sequence, slot 0) equal one prefill.
        let inp = inputs(11, 6);
        let whole = run_prefill(&inp, &[6], None, 0, GdnMode::Prefill);
        let mut state = vec![0f32; VH * VD * KD];
        let mut outs = Vec::new();
        for t in 0..6 {
            let one = Inputs {
                q: inp.q[t * KH * KD..(t + 1) * KH * KD].to_vec(),
                k: inp.k[t * KH * KD..(t + 1) * KH * KD].to_vec(),
                v: inp.v[t * VH * VD..(t + 1) * VH * VD].to_vec(),
                g: inp.g[t * VH..(t + 1) * VH].to_vec(),
                beta: inp.beta[t * VH..(t + 1) * VH].to_vec(),
            };
            let run = run_prefill(&one, &[1], Some(&state), 0, GdnMode::Decode);
            state = run.state;
            outs.extend(run.out);
        }
        assert_eq!(outs, whole.out);
        assert_eq!(state, whole.state);
    }

    #[test]
    fn split_prefill_equals_whole() {
        let tokens = 70;
        let inp = inputs(3, tokens);
        let whole = run_prefill(&inp, &[tokens], None, 0, GdnMode::Prefill);
        let split = 33;
        let first = run_prefill(&inp, &[split], None, 0, GdnMode::Prefill);
        let rest = Inputs {
            q: inp.q[split * KH * KD..].to_vec(),
            k: inp.k[split * KH * KD..].to_vec(),
            v: inp.v[split * VH * VD..].to_vec(),
            g: inp.g[split * VH..].to_vec(),
            beta: inp.beta[split * VH..].to_vec(),
        };
        let second = run_prefill(
            &rest,
            &[tokens - split],
            Some(&first.state),
            0,
            GdnMode::Prefill,
        );
        let mut out = first.out.clone();
        out.extend(&second.out);
        assert_eq!(
            out, whole.out,
            "a carried state continues the recurrence exactly"
        );
        assert_eq!(second.state, whole.state);
    }

    #[test]
    fn batch_of_sequences_is_independent() {
        let inp = inputs(5, 30);
        let lens = [10usize, 1, 19];
        let batch = run_prefill(&inp, &lens, None, 0, GdnMode::Prefill);
        let mut at = 0;
        for (s, &len) in lens.iter().enumerate() {
            let one = Inputs {
                q: inp.q[at * KH * KD..(at + len) * KH * KD].to_vec(),
                k: inp.k[at * KH * KD..(at + len) * KH * KD].to_vec(),
                v: inp.v[at * VH * VD..(at + len) * VH * VD].to_vec(),
                g: inp.g[at * VH..(at + len) * VH].to_vec(),
                beta: inp.beta[at * VH..(at + len) * VH].to_vec(),
            };
            let alone = run_prefill(&one, &[len], None, 0, GdnMode::Prefill);
            assert_eq!(batch.out[at * VH * VD..(at + len) * VH * VD], alone.out[..]);
            let per_slot = VH * VD * KD;
            assert_eq!(
                batch.state[s * per_slot..(s + 1) * per_slot],
                alone.state[..]
            );
            at += len;
        }
    }

    #[test]
    fn boundary_states_equal_stopped_prefill() {
        let inp = inputs(9, 21);
        let lens = [13usize, 8];
        let b = 4u32;
        let run = run_prefill(&inp, &lens, None, b, GdnMode::Prefill);
        let per = VH * VD * KD;
        // Rows: sequence 0 has 3 boundaries (4, 8, 12), sequence 1 has 2 (4, 8).
        assert_eq!(run.boundary.len(), 5 * per);
        let mut at = 0;
        let mut row = 0;
        for &len in &lens {
            for j in 0..len / b as usize {
                let stop = (j + 1) * b as usize;
                let part = Inputs {
                    q: inp.q[at * KH * KD..(at + stop) * KH * KD].to_vec(),
                    k: inp.k[at * KH * KD..(at + stop) * KH * KD].to_vec(),
                    v: inp.v[at * VH * VD..(at + stop) * VH * VD].to_vec(),
                    g: inp.g[at * VH..(at + stop) * VH].to_vec(),
                    beta: inp.beta[at * VH..(at + stop) * VH].to_vec(),
                };
                let stopped = run_prefill(&part, &[stop], None, 0, GdnMode::Prefill);
                assert_eq!(run.boundary[row * per..(row + 1) * per], stopped.state[..]);
                row += 1;
            }
            at += len;
        }
    }

    const CH: usize = 6;

    fn conv_run(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        lens: &[usize],
        state: &[f32],
        boundary_tokens: u32,
        mode: GdnMode,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mem = host();
        let tokens: usize = lens.iter().sum();
        let slots = lens.len();
        let xt = tensor(&mem, &[tokens, CH], DType::BF16, x);
        let wt = tensor(&mem, &[CH, 4], DType::BF16, w);
        let bt = bias.map(|b| tensor(&mem, &[CH], DType::BF16, b));
        let st = tensor(&mem, &[slots, CH, 3], DType::BF16, state);
        let out = Tensor::empty(&mem, &[tokens, CH], DType::BF16).expect("alloc");
        let ids = i32_tensor(&mem, &(0..slots as i32).collect::<Vec<_>>());
        let mut cu = vec![0i32];
        for l in lens {
            cu.push(cu.last().copied().unwrap_or(0) + *l as i32);
        }
        let cu_t = i32_tensor(&mem, &cu);
        let nrows: usize = if boundary_tokens > 0 {
            lens.iter().map(|l| l / boundary_tokens as usize).sum()
        } else {
            0
        };
        let bs =
            (nrows > 0).then(|| Tensor::empty(&mem, &[nrows, CH, 3], DType::BF16).expect("alloc"));
        CpuReference
            .conv1d()
            .expect("conv1d family")
            .execute(&mut Conv1dContext {
                cfg: Conv1dConfig {
                    channels: CH as u32,
                    kernel: 4,
                    mode,
                    dtype: DType::BF16,
                },
                x: xt.view(),
                weight: wt.view(),
                bias: bt.as_ref().map(Tensor::view),
                conv_state: st.view(),
                state_slots: ids.view(),
                cu_seqlens: (mode == GdnMode::Prefill).then(|| cu_t.view()),
                out: out.view(),
                boundary_conv_states: bs.as_ref().map(Tensor::view),
                boundary_tokens,
                activation: 1,
            })
            .expect("conv1d");
        (
            load(&out.view()).expect("load"),
            load(&st.view()).expect("load"),
            bs.map_or_else(Vec::new, |t| load(&t.view()).expect("load")),
        )
    }

    #[test]
    fn conv1d_matches_naive_and_updates_state() {
        let bf = |x: f32| round_to(DType::BF16, x);
        let x: Vec<f32> = seeded(1, 9 * CH).into_iter().map(bf).collect();
        let w: Vec<f32> = seeded(2, CH * 4).into_iter().map(bf).collect();
        let bias: Vec<f32> = seeded(3, CH).into_iter().map(bf).collect();
        let state: Vec<f32> = seeded(4, CH * 3).into_iter().map(bf).collect();
        let (out, new_state, _) = conv_run(&x, &w, Some(&bias), &[9], &state, 0, GdnMode::Prefill);
        for c in 0..CH {
            let mut win: Vec<f64> = state[c * 3..c * 3 + 3]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            win.extend((0..9).map(|t| f64::from(x[t * CH + c])));
            for t in 0..9 {
                let acc: f64 = f64::from(bias[c])
                    + (0..4)
                        .map(|j| f64::from(w[c * 4 + j]) * win[t + j])
                        .sum::<f64>();
                let want = acc / (1.0 + (-acc).exp());
                let got = f64::from(out[t * CH + c]);
                assert!(
                    (got - want).abs() <= 0.01 * want.abs() + 1e-3,
                    "t{t} c{c}: {got} vs {want}"
                );
            }
            for j in 0..3 {
                assert_eq!(f64::from(new_state[c * 3 + j]), win[9 + j]);
            }
        }
        // Decode token by token equals the prefill, bit for bit, and so does the state.
        let mut st = state.clone();
        let mut outs = Vec::new();
        for t in 0..9 {
            let (o, s, _) = conv_run(
                &x[t * CH..(t + 1) * CH],
                &w,
                Some(&bias),
                &[1],
                &st,
                0,
                GdnMode::Decode,
            );
            outs.extend(o);
            st = s;
        }
        assert_eq!(outs, out);
        assert_eq!(st, new_state);
    }

    #[test]
    fn conv1d_boundary_states_are_the_tail_at_each_boundary() {
        let bf = |x: f32| round_to(DType::BF16, x);
        let x: Vec<f32> = seeded(21, 10 * CH).into_iter().map(bf).collect();
        let w: Vec<f32> = seeded(22, CH * 4).into_iter().map(bf).collect();
        let zero = vec![0f32; CH * 3];
        let (_, _, rows) = conv_run(&x, &w, None, &[10], &zero, 4, GdnMode::Prefill);
        assert_eq!(rows.len(), 2 * CH * 3);
        for (j, stop) in [4usize, 8].into_iter().enumerate() {
            for c in 0..CH {
                for i in 0..3 {
                    assert_eq!(rows[(j * CH + c) * 3 + i], x[(stop - 3 + i) * CH + c]);
                }
            }
        }
    }
}
