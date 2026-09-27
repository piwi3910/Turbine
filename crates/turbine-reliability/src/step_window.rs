//! The windowed decode step time behind `step_time_drift` and the circuit's latency drift
//! (P3 S-6, S-12). The engine and the overload simulator feed it the same way, so the drift the
//! simulator tests is the drift the server reports.
//!
//! A decode step's time depends on its shape: a dense model reads its weights once per step
//! whatever the batch, an MoE model's expert GEMMs grow with the rows, and every model reads
//! each sequence's KV. So the window keeps one baseline per shape bucket — the exact number of
//! decoding rows × the total context tokens in half-powers of two — learned from calm steps
//! (GREEN + HEALTHY, no circuit probe), and judges a step only against its own bucket once that
//! bucket has [`MIN_BUCKET_SAMPLES`] calm steps. Shapes never seen while calm are not judged:
//! no extrapolation across buckets.

use std::collections::{HashMap, VecDeque};

/// Pure decode iterations the window keeps.
pub const STEP_WINDOW: usize = 64;
/// Calm steps a bucket needs before its steps are judged.
pub const MIN_BUCKET_SAMPLES: u32 = 8;
/// Smoothing of a bucket's baseline.
const BASELINE_ALPHA: f64 = 0.1;

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

/// (rows, context bucket): context in half-powers of two of 1,024 tokens.
type Bucket = (u32, u32);

fn bucket(rows: u32, context_tokens: u64) -> Bucket {
    let k = context_tokens as f64 / 1024.0;
    (rows, (2.0 * (k + 1.0).log2()).floor() as u32)
}

#[derive(Clone, Copy, Debug, Default)]
struct Baseline {
    secs: f64,
    samples: u32,
}

/// The last [`STEP_WINDOW`] judged pure decode iterations (no prefill in the batch), each as its
/// time over its shape bucket's calm baseline.
#[derive(Clone, Debug, Default)]
pub struct DecodeStepWindow {
    steps: VecDeque<f64>,
    baselines: HashMap<Bucket, Baseline>,
}

impl DecodeStepWindow {
    pub fn new() -> Self {
        DecodeStepWindow {
            steps: VecDeque::with_capacity(STEP_WINDOW),
            baselines: HashMap::new(),
        }
    }

    /// One executed iteration. Iterations with a prefill are skipped, so a prefill is not
    /// mistaken for slowing down. The step is judged against its bucket's baseline (before this
    /// step updates it); `calm` steps then update that baseline.
    pub fn observe(&mut self, s: StepSample, calm: bool) {
        if s.rows == 0 || s.prefill_tokens > 0 {
            return;
        }
        let key = bucket(s.rows, s.context_tokens);
        let base = self.baselines.get(&key).copied().unwrap_or_default();
        if base.samples >= MIN_BUCKET_SAMPLES && base.secs > 0.0 {
            if self.steps.len() == STEP_WINDOW {
                self.steps.pop_front();
            }
            self.steps.push_back(s.secs / base.secs);
        }
        if calm {
            let b = self.baselines.entry(key).or_default();
            b.secs = if b.samples == 0 {
                s.secs
            } else {
                BASELINE_ALPHA * s.secs + (1.0 - BASELINE_ALPHA) * b.secs
            };
            b.samples = b.samples.saturating_add(1);
        }
    }

    /// Forget every baseline and judged step (the circuit came back from PROBING: the device
    /// may have changed, P3 S-12).
    pub fn reset(&mut self) {
        self.steps.clear();
        self.baselines.clear();
    }

    /// The window's p95 of step time over its bucket's baseline (about 1 while the device
    /// performs as in calm steps); `None` before any step could be judged.
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

    fn step(rows: u32, context_tokens: u64, secs: f64) -> StepSample {
        StepSample {
            prefill_tokens: 0,
            rows,
            context_tokens,
            secs,
        }
    }

    /// An MoE-like step: expert GEMMs grow with the rows (OLMoE, 2026-09-27 regression).
    fn moe(rows: u32, context_tokens: u64) -> f64 {
        0.008 + 0.0009 * f64::from(rows) + 0.0001 * context_tokens as f64 / 1024.0
    }

    /// The landing regression: OLMoE at 16 concurrent requests read a full batch against the
    /// light-load baseline as drift, pressure went ORANGE on `step_time_drift` and throughput
    /// fell 12 %. Catches: steps of one shape judged against a baseline of another.
    #[test]
    fn moe_full_batch_is_not_drift() {
        let mut w = DecodeStepWindow::new();
        // Calm ramp-up: batches of 1..16 rows, each shape seen often enough.
        for _ in 0..20 {
            for rows in 1..=16 {
                let ctx = u64::from(rows) * 700;
                w.observe(step(rows, ctx, moe(rows, ctx)), true);
            }
        }
        // Steady full batch, twice the time of a 4-row step.
        assert!(moe(16, 16 * 700) / moe(4, 4 * 700) > 1.9);
        for _ in 0..STEP_WINDOW {
            w.observe(step(16, 16 * 700, moe(16, 16 * 700)), false);
        }
        let p95 = w.p95().unwrap();
        assert!((p95 - 1.0).abs() < 0.01, "full batch reads {p95}");
    }

    /// Catches: a real slowdown hidden by the bucketing. The same shape twice as slow reads 2.
    #[test]
    fn same_bucket_slowdown_is_drift() {
        let mut w = DecodeStepWindow::new();
        for _ in 0..MIN_BUCKET_SAMPLES {
            w.observe(step(8, 8 * 2000, moe(8, 8 * 2000)), true);
        }
        for _ in 0..STEP_WINDOW {
            w.observe(step(8, 8 * 2000, 2.0 * moe(8, 8 * 2000)), false);
        }
        assert!((w.p95().unwrap() - 2.0).abs() < 1e-9);
        w.reset();
        assert_eq!(w.p95(), None, "reset forgets baselines and steps");
    }

    /// A shape never seen while calm is not judged (no extrapolation), a bucket needs
    /// `MIN_BUCKET_SAMPLES` calm steps, and a prefill in the batch is not a decode step.
    #[test]
    fn unseen_shapes_and_prefills_are_not_judged() {
        let mut w = DecodeStepWindow::new();
        for _ in 0..MIN_BUCKET_SAMPLES - 1 {
            w.observe(step(2, 1000, 0.01), true);
        }
        w.observe(step(2, 1000, 0.05), false);
        assert_eq!(w.p95(), None, "bucket not ready");
        w.observe(step(2, 1000, 0.01), true);
        w.observe(step(23, 92_000, 0.2), false);
        assert_eq!(w.p95(), None, "unseen shape");
        w.observe(
            StepSample {
                prefill_tokens: 2048,
                ..step(2, 1000, 0.5)
            },
            false,
        );
        assert_eq!(w.p95(), None, "prefill skipped");
        w.observe(step(2, 1000, 0.01), false);
        assert!((w.p95().unwrap() - 1.0).abs() < 1e-9);
    }

    /// Context in half-powers of two: 1,100 and 1,300 tokens share a bucket, 1,100 and 3,000 do
    /// not; every row count is its own bucket.
    #[test]
    fn context_buckets() {
        assert_eq!(bucket(4, 1100), bucket(4, 1300));
        assert_ne!(bucket(4, 1100), bucket(4, 3000));
        assert_ne!(bucket(4, 1100), bucket(5, 1100));
    }
}
