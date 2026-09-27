//! The windowed decode step time behind `step_time_drift` and the circuit's latency drift
//! (P3 S-6, S-12). The engine and the overload simulator feed it the same way, so the drift the
//! simulator tests is the drift the server reports.
//!
//! A decode step's time depends on its work: the weights are read once per step and every
//! running sequence's KV once, so a full batch of long contexts takes longer than a few short
//! ones on a healthy device. The window therefore keeps a work-cost model,
//! `time ≈ a + b · rows + c · context tokens`, fitted by exponentially weighted least squares
//! on calm steps only (GREEN + HEALTHY, no probe), and reports each step's time relative to the
//! model's prediction for that step. Drift is then "slower than this work costs", not "more
//! work than at calibration".

use std::collections::VecDeque;

/// Pure decode iterations the window keeps.
pub const STEP_WINDOW: usize = 64;
/// Calm steps the cost model needs before it predicts (and the window reports).
pub const MIN_CALM_STEPS: u32 = 32;
/// Per-step decay of the cost model's weights (an effective memory of about 500 calm steps).
const DECAY: f64 = 0.998;
/// Ridge on the rows and context coefficients (in their feature units), relative to the total
/// weight: a coefficient the calm steps do not identify stays at 0 instead of fitting noise.
const RIDGE: f64 = 1e-3;
/// Context tokens are fitted per 1,024 to keep the normal equations well conditioned.
const CONTEXT_UNIT: f64 = 1024.0;

/// One executed iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepSample {
    pub prefill_tokens: u32,
    /// Sequences decoding in the step.
    pub rows: u32,
    /// Context tokens those sequences attend over (the sum of their context lengths).
    pub context_tokens: u64,
    pub secs: f64,
}

/// `time ≈ a + b · rows + c · context/1024`, weighted least squares with decay and ridge.
#[derive(Clone, Debug, Default)]
struct CostModel {
    /// Σ w·x·xᵀ over x = (1, rows, context/1024).
    xx: [[f64; 3]; 3],
    /// Σ w·x·y.
    xy: [f64; 3],
    calm_steps: u32,
    coef: Option<[f64; 3]>,
}

impl CostModel {
    fn features(rows: u32, context_tokens: u64) -> [f64; 3] {
        [1.0, f64::from(rows), context_tokens as f64 / CONTEXT_UNIT]
    }

    fn learn(&mut self, rows: u32, context_tokens: u64, secs: f64) {
        let x = Self::features(rows, context_tokens);
        for (i, row) in self.xx.iter_mut().enumerate() {
            self.xy[i] = DECAY * self.xy[i] + x[i] * secs;
            for (v, xj) in row.iter_mut().zip(x) {
                *v = DECAY * *v + x[i] * xj;
            }
        }
        self.calm_steps = self.calm_steps.saturating_add(1);
        if self.calm_steps >= MIN_CALM_STEPS {
            self.coef = self.solve();
        }
    }

    fn solve(&self) -> Option<[f64; 3]> {
        let weight = self.xx[0][0];
        if weight <= 0.0 {
            return None;
        }
        let mut a = self.xx;
        a[1][1] += RIDGE * weight;
        a[2][2] += RIDGE * weight;
        let mut coef = solve3(a, self.xy)?;
        // A healthy step never gets faster with more rows or context; a negative coefficient
        // is noise, and it would read heavier steps as drift. Refit the rest without it.
        for k in [1, 2] {
            if coef[k] < 0.0 {
                let mut a = a;
                let mut b = self.xy;
                for row in a.iter_mut() {
                    row[k] = 0.0;
                }
                a[k] = [0.0; 3];
                a[k][k] = 1.0;
                b[k] = 0.0;
                coef = solve3(a, b)?;
            }
        }
        let [c0, c1, c2] = coef;
        Some([c0, c1.max(0.0), c2.max(0.0)])
    }

    /// Predicted step time; `None` until [`MIN_CALM_STEPS`] calm steps were seen.
    fn expected(&self, rows: u32, context_tokens: u64) -> Option<f64> {
        let coef = self.coef?;
        let x = Self::features(rows, context_tokens);
        let t: f64 = coef.iter().zip(x).map(|(c, x)| c * x).sum();
        // Never below a quarter of the mean calm step: a tiny prediction would turn noise into
        // drift.
        let mean = self.xy[0] / self.xx[0][0];
        Some(t.max(0.25 * mean)).filter(|t| *t > 0.0)
    }
}

/// Gaussian elimination with partial pivoting; `None` when singular.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col];
        for row in col + 1..3 {
            let f = a[row][col] / pivot_row[col];
            for (v, p) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *v -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let s: f64 = (row + 1..3).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

/// The last [`STEP_WINDOW`] pure decode iterations (no prefill in the batch), each as its time
/// relative to the work-cost model's prediction.
#[derive(Clone, Debug, Default)]
pub struct DecodeStepWindow {
    steps: VecDeque<f64>,
    model: CostModel,
}

impl DecodeStepWindow {
    pub fn new() -> Self {
        DecodeStepWindow {
            steps: VecDeque::with_capacity(STEP_WINDOW),
            model: CostModel::default(),
        }
    }

    /// One executed iteration. Iterations with a prefill are skipped, so a prefill is not
    /// mistaken for slowing down. `calm` (GREEN + HEALTHY, no circuit probe): the cost model
    /// learns from this step. The step time is the iteration's, the time every sequence in it
    /// waits for its token.
    pub fn observe(&mut self, s: StepSample, calm: bool) {
        if s.rows == 0 || s.prefill_tokens > 0 {
            return;
        }
        if calm {
            self.model.learn(s.rows, s.context_tokens, s.secs);
        }
        let Some(expected) = self.model.expected(s.rows, s.context_tokens) else {
            return;
        };
        if self.steps.len() == STEP_WINDOW {
            self.steps.pop_front();
        }
        self.steps.push_back(s.secs / expected);
    }

    /// The window's p95 of step time relative to the cost model (about 1 while the device
    /// performs as in calm steps); `None` before the model is ready.
    pub fn p95(&self) -> Option<f64> {
        if self.steps.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.steps.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        Some(sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A memory-bound 3B decode step on an R9700: 11 ms of weights, 0.2 ms per row,
    /// 0.19 ms per 1,024 context tokens (112 KiB of KV per token at ~600 GB/s).
    fn healthy(rows: u32, context_tokens: u64) -> f64 {
        0.011 + 0.0002 * f64::from(rows) + 0.00019 * context_tokens as f64 / 1024.0
    }

    fn step(rows: u32, context_tokens: u64, secs: f64) -> StepSample {
        StepSample {
            prefill_tokens: 0,
            rows,
            context_tokens,
            secs,
        }
    }

    /// The calibration of the 2026-09-27 soak: up to 4 sequences of up to 7,000 tokens.
    fn calibrate(w: &mut DecodeStepWindow) {
        for i in 0..400u64 {
            let rows = 1 + (i % 4) as u32;
            let context = u64::from(rows) * (200 + (i * 997) % 6800);
            w.observe(step(rows, context, healthy(rows, context)), true);
        }
    }

    /// The fourth soak's end: 20 s after the queue emptied, the last long-context sequences
    /// drained as a full batch; raw step time was about 2.5 × the calibration's, drift crossed
    /// 2.0 and the circuit went DEGRADED (`latency_drift`), so GREEN + HEALTHY came at 61 s.
    /// Catches: more work per step (rows, context) read as device degradation.
    #[test]
    fn long_context_full_batch_is_not_drift() {
        let mut w = DecodeStepWindow::new();
        calibrate(&mut w);
        let calm = w.p95().unwrap();
        assert!((calm - 1.0).abs() < 0.02, "calm p95 {calm}");
        // Under pressure (not calm): 23 rows of ~4,000 tokens each.
        let (rows, context) = (23, 23 * 4000);
        assert!(
            healthy(rows, context) / healthy(2, 2 * 3500) > 2.0,
            "raw time doubles"
        );
        for _ in 0..STEP_WINDOW {
            w.observe(step(rows, context, healthy(rows, context)), false);
        }
        let p95 = w.p95().unwrap();
        assert!(p95 < 1.1, "a healthy full batch reads {p95}");
        // A device that really is twice as slow on the same work still reads as drift.
        for _ in 0..STEP_WINDOW {
            w.observe(step(rows, context, 2.0 * healthy(rows, context)), false);
        }
        let slow = w.p95().unwrap();
        assert!(slow > 1.9, "a 2× slower device reads {slow}");
    }

    /// Catches: a batch-size change read as drift (the soak's calibration went DEGRADED when
    /// the step time was divided by the batch size), and a prefill counted as a decode step.
    #[test]
    fn batch_size_is_not_drift() {
        let mut w = DecodeStepWindow::new();
        assert_eq!(w.p95(), None);
        for i in 0..MIN_CALM_STEPS - 1 {
            let rows = 1 + i % 16;
            w.observe(step(rows, 1000, healthy(rows, 1000)), true);
        }
        assert_eq!(w.p95(), None, "no drift before the model is ready");
        calibrate(&mut w);
        for _ in 0..STEP_WINDOW {
            w.observe(step(1, 500, healthy(1, 500)), false);
        }
        assert!((w.p95().unwrap() - 1.0).abs() < 0.05);
        w.observe(
            StepSample {
                prefill_tokens: 2048,
                ..step(1, 500, 0.5)
            },
            false,
        );
        assert!((w.p95().unwrap() - 1.0).abs() < 0.05, "prefill skipped");
    }

    /// Calm steps that never vary the rows (one sequence at a time) leave the rows coefficient
    /// unidentified: the ridge keeps it at 0 and the prediction stays sane.
    #[test]
    fn unidentified_coefficients_stay_sane() {
        let mut w = DecodeStepWindow::new();
        for i in 0..200u64 {
            let context = 500 + (i * 131) % 7000;
            w.observe(step(1, context, healthy(1, context)), true);
        }
        let expected = w.model.expected(1, 3000).unwrap();
        assert!((expected / healthy(1, 3000) - 1.0).abs() < 0.02);
        let coef = w.model.coef.unwrap();
        assert!(coef[1] >= 0.0 && coef[2] >= 0.0, "{coef:?}");
    }
}
