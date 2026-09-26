//! Device-style logits reduction on the cpu-reference provider: log-sum-exp, top-n and the
//! categorical / nucleus draws of the host sampler.
use turbine_core::types::DType;
use turbine_tensor::TensorView;

use super::{CpuReference, expect_rank, invalid, load, load_i32, math, store, store_i32};
use crate::KernelError;
use crate::ops::{LogitsReduceConfig, LogitsReduceContext, LogitsReduceKernel};

/// Checks that `v` is a dense `dtype` view of rank `1 + cols.is_some()` with at least `rows` rows
/// (and exactly `cols` columns), returning its first `rows` rows.
fn leading_rows<'a>(
    name: &str,
    v: &TensorView<'a>,
    rows: usize,
    cols: Option<usize>,
    dtype: DType,
) -> Result<TensorView<'a>, KernelError> {
    let rank = 1 + usize::from(cols.is_some());
    let dense = turbine_tensor::tensor::contiguous_strides(&v.shape);
    if v.dtype != dtype
        || v.shape.len() != rank
        || v.shape[0] < rows
        || cols.is_some_and(|c| v.shape[1] != c)
        || v.strides != dense
    {
        return Err(invalid(format!(
            "{name} must be a dense {} view of at least {rows} rows{}, has {} shape {:?} strides {:?}",
            dtype.as_str(),
            cols.map_or(String::new(), |c| format!(" of {c} columns")),
            v.dtype.as_str(),
            v.shape.as_slice(),
            v.strides.as_slice()
        )));
    }
    Ok(v.rows(0, rows))
}

impl LogitsReduceKernel for CpuReference {
    fn supports(&self, cfg: &LogitsReduceConfig) -> bool {
        cfg.vocab > 0 && cfg.top_n <= LogitsReduceConfig::MAX_TOP_N && cfg.top_n <= cfg.vocab
    }

    fn implementation(&self, _cfg: &LogitsReduceConfig) -> String {
        "cpu_logits_reduce".into()
    }

    fn execute(&self, ctx: &mut LogitsReduceContext<'_>) -> Result<(), KernelError> {
        let rows = ctx.rows as usize;
        expect_rank("logits", &ctx.logits, 2)?;
        expect_rank("top_ids", &ctx.top_ids, 2)?;
        let vocab = ctx.logits.shape[1];
        let top_n = ctx.top_ids.shape[1];
        let cfg = LogitsReduceConfig {
            vocab: u32::try_from(vocab).map_err(|_| invalid(format!("vocab {vocab}")))?,
            top_n: u32::try_from(top_n).map_err(|_| invalid(format!("top_n {top_n}")))?,
        };
        if !LogitsReduceKernel::supports(self, &cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference logits_reduce {cfg}"),
            });
        }
        if ctx.logits.dtype != DType::F32 || ctx.logits.shape[0] < rows {
            return Err(invalid(format!(
                "logits must be an f32 view of at least {rows} rows, has {} shape {:?}",
                ctx.logits.dtype.as_str(),
                ctx.logits.shape.as_slice()
            )));
        }
        let f32_rows = |name, v| leading_rows(name, v, rows, None, DType::F32);
        let temperature = load(&f32_rows("temperature", &ctx.temperature)?)?;
        let uniform = load(&f32_rows("uniform", &ctx.uniform)?)?;
        let top_p = load(&f32_rows("top_p", &ctx.top_p)?)?;
        let mode = load_i32(&leading_rows("mode", &ctx.mode, rows, None, DType::I32)?)?;
        let top_ids_view = leading_rows("top_ids", &ctx.top_ids, rows, Some(top_n), DType::I32)?;
        let top_values_view =
            leading_rows("top_values", &ctx.top_values, rows, Some(top_n), DType::F32)?;
        let lse_view = f32_rows("lse", &ctx.lse)?;
        let sampled_view = leading_rows("sampled", &ctx.sampled, rows, None, DType::I32)?;
        let sampled_logit_view = f32_rows("sampled_logit", &ctx.sampled_logit)?;
        let logits = load(&ctx.logits.rows(0, rows))?;

        let mut top_ids = Vec::with_capacity(rows * top_n);
        let mut top_values = Vec::with_capacity(rows * top_n);
        let mut lse = Vec::with_capacity(rows);
        let mut sampled = Vec::with_capacity(rows);
        let mut sampled_logit = Vec::with_capacity(rows);
        for (r, row) in logits.chunks_exact(vocab).enumerate() {
            for (id, value) in math::top_n(row, top_n) {
                top_ids.push(id as i32);
                top_values.push(value);
            }
            lse.push(math::log_sum_exp(row));
            match mode[r] {
                0 => {
                    sampled.push(-1);
                    sampled_logit.push(f32::NAN);
                }
                1 => {
                    let id = if top_p[r] < 1.0 {
                        math::nucleus(row, temperature[r], top_p[r], uniform[r])
                    } else {
                        math::categorical(row, temperature[r], uniform[r])
                    };
                    sampled.push(id as i32);
                    sampled_logit.push(row[id as usize]);
                }
                other => {
                    return Err(invalid(format!("mode[{r}] = {other}, expected 0 or 1")));
                }
            }
        }
        store_i32(&top_ids_view, &top_ids)?;
        store(&top_values_view, &top_values)?;
        store(&lse_view, &lse)?;
        store_i32(&sampled_view, &sampled)?;
        store(&sampled_logit_view, &sampled_logit)
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    /// The host sampler's candidate order: descending, ties to the lower id, NaN last (a full
    /// sort, independent of the provider's partial selection).
    fn reference_top(row: &[f32], n: usize) -> Vec<(u32, f32)> {
        let mut pairs: Vec<(u32, f32)> = (0..row.len() as u32).zip(row.iter().copied()).collect();
        pairs.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => a.0.cmp(&b.0),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)),
        });
        pairs.truncate(n);
        pairs
    }

    /// Log-sum-exp over the non-NaN values, entirely in f64.
    fn reference_lse(row: &[f32]) -> f64 {
        let finite = row.iter().filter(|v| !v.is_nan()).map(|&v| f64::from(v));
        let max = finite.clone().fold(f64::NEG_INFINITY, f64::max);
        max + finite.map(|v| (v - max).exp()).sum::<f64>().ln()
    }

    /// The host sampler's draw with `top_k` −1 and `top_p` 1 (candidates in id order): scaled
    /// logits `v · (1/T)`, weights `exp(scaled − max)` summed sequentially in f64, and the first id
    /// whose cumulative weight exceeds `u · total`.
    fn reference_draw(row: &[f32], temperature: f32, u: f32) -> u32 {
        let inv_t = 1.0 / temperature;
        let scaled: Vec<f32> = row.iter().map(|&v| v * inv_t).collect();
        let max = scaled
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = scaled
            .iter()
            .map(|&s| {
                if s.is_nan() {
                    0.0
                } else {
                    f64::from(s - max).exp()
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let target = f64::from(u) * total;
        let mut cum = 0.0;
        for (id, w) in weights.iter().enumerate() {
            cum += w;
            if target < cum {
                return id as u32;
            }
        }
        weights.iter().rposition(|&w| w > 0.0).unwrap_or(0) as u32
    }

    /// On seeded rows of both model vocabularies (tied maxima, a tied run inside the top-20, NaN
    /// and −∞ entries, a padded row stride) the reference `logits_reduce` returns the sampler's
    /// top-20 in order with the raw values, its log-sum-exp within 1e-6 (relative) and, in
    /// categorical mode at temperature 0.7 with ChaCha8 uniforms, the sampler's inverse-CDF token;
    /// rows with `top_p` < 1 (a broad row at 0.9 and 0.5, a peaked one whose nucleus ends inside
    /// the tied run) return the sampler's top-p token. Breaks if ties go to the higher id, NaN
    /// ranks first, the lse ignores part of the row, the draw is taken in another order or at
    /// another temperature, or the nucleus is cut at another mass or drawn in id order.
    #[test]
    fn logits_reduce_matches_sampler() {
        use rand_chacha::ChaCha8Rng;
        use rand_chacha::rand_core::{RngCore, SeedableRng};

        let mem = HostMemory::new(DeviceId(0), 1 << 24) as Arc<dyn DeviceMemory>;
        let reduce = CpuReference.logits_reduce().expect("logits_reduce family");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let mut uniform = || (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let (rows, top_n, pad) = (7usize, 20usize, 5usize);
        // Row 0 reduces only; rows 1..7 also draw at temperature 0.7 (row 6 at 0.4), rows 4..7
        // from a nucleus.
        let modes = [0, 1, 1, 1, 1, 1, 1];
        let temperatures = [0.0f32, 0.7, 0.7, 0.7, 0.7, 0.7, 0.4];
        let top_ps = [1.0f32, 1.0, 1.0, 1.0, 0.9, 0.5, 0.8];
        let mut draws_below_the_top = 0;
        let mut nucleus_draws_below_the_top = 0;
        for vocab in [50_304usize, 128_256] {
            let cfg = LogitsReduceConfig {
                vocab: vocab as u32,
                top_n: top_n as u32,
            };
            assert!(reduce.supports(&cfg));
            assert_eq!(reduce.implementation(&cfg), "cpu_logits_reduce");
            assert!(!reduce.supports(&LogitsReduceConfig {
                vocab: 100,
                top_n: 65
            }));

            let mut logits = Vec::with_capacity(rows);
            for r in 0..rows {
                // Row 6's bulk is narrow (±2), the others' ±8.
                let spread = if r == 6 { 2.0 } else { 8.0 };
                let mut row: Vec<f32> = seeded((vocab + r) as u64, vocab)
                    .iter()
                    .map(|v| v * spread)
                    .collect();
                // Three tied maxima, a tied run just below them, NaN and −∞ entries.
                for id in [7, 1000 + r, vocab - 3] {
                    row[id] = 9.0;
                }
                for value in &mut row[100..110] {
                    *value = 8.5;
                }
                row[3] = f32::NAN;
                row[vocab / 2] = f32::NAN;
                row[11] = f32::NEG_INFINITY;
                if r == 6 {
                    // Peaked: the three maxima and the tied run hold nearly all the mass at
                    // T = 0.4, so the 0.8 nucleus ends inside the run (at its eighth id).
                    for value in &mut row[100..110] {
                        *value = 8.9;
                    }
                }
                logits.push(row);
            }
            // Logits rows padded to a stride of vocab + pad elements; the padding must never
            // be read (it would win every ranking).
            let padded = Tensor::empty(&mem, &[rows, vocab + pad], DType::F32).expect("logits");
            let flat: Vec<f32> = logits
                .iter()
                .flat_map(|row| row.iter().copied().chain([1e30; 5]))
                .collect();
            store(&padded.view(), &flat).expect("fill");
            let logits_view = TensorView {
                slice: padded.storage.whole(),
                shape: (&[rows, vocab][..]).into(),
                strides: (&[vocab + pad, 1][..]).into(),
                dtype: DType::F32,
            };
            let uniforms: Vec<f32> = (0..rows).map(|_| uniform()).collect();
            // Outputs hold one spare row, which the op must leave alone.
            let spare = rows + 1;
            let temperature = tensor(&mem, &[rows], DType::F32, &temperatures);
            let uniform_t = tensor(&mem, &[rows], DType::F32, &uniforms);
            let top_p = tensor(&mem, &[rows], DType::F32, &top_ps);
            let mode = i32_tensor(&mem, &modes);
            let top_ids = i32_tensor_2d(&mem, spare, &vec![-7; spare * top_n]);
            let top_values = Tensor::empty(&mem, &[spare, top_n], DType::F32).expect("values");
            let lse = Tensor::empty(&mem, &[spare], DType::F32).expect("lse");
            let sampled = i32_tensor(&mem, &vec![-7; spare]);
            let sampled_logit = Tensor::empty(&mem, &[spare], DType::F32).expect("logit");
            reduce
                .execute(&mut LogitsReduceContext {
                    logits: logits_view,
                    temperature: temperature.view(),
                    uniform: uniform_t.view(),
                    top_p: top_p.view(),
                    mode: mode.view(),
                    top_ids: top_ids.view(),
                    top_values: top_values.view(),
                    lse: lse.view(),
                    sampled: sampled.view(),
                    sampled_logit: sampled_logit.view(),
                    rows: rows as u32,
                })
                .expect("logits_reduce");

            let got_ids = load_i32(&top_ids.view()).expect("ids");
            let got_values = load(&top_values.view()).expect("values");
            let got_lse = load(&lse.view()).expect("lse");
            let got_sampled = load_i32(&sampled.view()).expect("sampled");
            let got_logit = load(&sampled_logit.view()).expect("logit");
            for (r, row) in logits.iter().enumerate() {
                let want = reference_top(row, top_n);
                let mut maxima = vec![7, 1000 + r as u32, vocab as u32 - 3];
                maxima.sort_unstable();
                let tied: Vec<u32> = want[..3].iter().map(|c| c.0).collect();
                assert_eq!(tied, maxima, "tied maxima rank by id");
                let ids = &got_ids[r * top_n..(r + 1) * top_n];
                let values = &got_values[r * top_n..(r + 1) * top_n];
                assert_eq!(
                    ids,
                    want.iter().map(|c| c.0 as i32).collect::<Vec<_>>(),
                    "vocab {vocab} row {r}"
                );
                for (got, want) in values.iter().zip(&want) {
                    assert_eq!(got.to_bits(), want.1.to_bits(), "vocab {vocab} row {r}");
                }
                let want_lse = reference_lse(row);
                assert!(
                    (f64::from(got_lse[r]) - want_lse).abs() <= 1e-6 * want_lse.abs().max(1.0),
                    "vocab {vocab} row {r}: lse {} vs {want_lse}",
                    got_lse[r]
                );
                if modes[r] == 0 {
                    assert_eq!(got_sampled[r], -1);
                    assert!(got_logit[r].is_nan());
                } else if top_ps[r] < 1.0 {
                    let want = reference_nucleus(row, temperatures[r], top_ps[r], uniforms[r]);
                    assert_eq!(got_sampled[r], want as i32, "vocab {vocab} row {r} (top_p)");
                    assert_eq!(got_logit[r].to_bits(), row[want as usize].to_bits());
                    if !maxima.contains(&want) {
                        nucleus_draws_below_the_top += 1;
                    }
                } else {
                    let want = reference_draw(row, temperatures[r], uniforms[r]);
                    assert_eq!(got_sampled[r], want as i32, "vocab {vocab} row {r}");
                    assert_eq!(got_logit[r].to_bits(), row[want as usize].to_bits());
                    if want != maxima[0] {
                        draws_below_the_top += 1;
                    }
                }
            }
            assert_eq!(&got_ids[rows * top_n..], [-7; 20], "spare row written");
            assert_eq!(got_sampled[rows], -7, "spare row written");
        }
        assert!(
            draws_below_the_top > 0,
            "every draw returned the argmax: the categorical path is not exercised"
        );
        assert!(
            nucleus_draws_below_the_top > 0,
            "every nucleus draw returned a maximum: the top-p path is not exercised"
        );
    }

    /// The host sampler's seeded top-p draw (`top_k` −1, `top_p` < 1), written out: all ids
    /// sorted descending (ties by id, NaN last), f64 weights summed in that order, the shortest
    /// prefix reaching `top_p · total`, then `u` × the prefix's sum scanned in the same order.
    fn reference_nucleus(row: &[f32], temperature: f32, top_p: f32, u: f32) -> u32 {
        let inv_t = 1.0 / temperature;
        let sorted = reference_top(row, row.len());
        let max = sorted
            .iter()
            .map(|c| c.1 * inv_t)
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = sorted
            .iter()
            .map(|c| {
                let s = c.1 * inv_t;
                if s.is_nan() {
                    0.0
                } else {
                    f64::from(s - max).exp()
                }
            })
            .collect();
        let target = f64::from(top_p) * weights.iter().sum::<f64>();
        let mut cum = 0.0;
        let keep = weights
            .iter()
            .position(|w| {
                cum += w;
                cum >= target
            })
            .map_or(weights.len(), |i| i + 1);
        let target = f64::from(u) * weights[..keep].iter().sum::<f64>();
        let mut cum = 0.0;
        let i = weights[..keep]
            .iter()
            .position(|w| {
                cum += w;
                target < cum
            })
            .expect("u < 1 falls inside the nucleus");
        sorted[i].0
    }
}
