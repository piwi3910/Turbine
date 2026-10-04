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
//!
//! Judged steps older than [`STEP_MAX_AGE`] leave the window: drift describes the decoding that
//! is happening. Above GREEN no shape learns a baseline, so once the judged shapes stop running
//! (a RED queue admits one request at a time, at contexts never decoded while calm) the window
//! would otherwise keep its last spike and hold the level for as long as anything runs.
//!
//! A decode step that overlapped an in-flight KV tier copy ([`StepSample::copy_overlap`]) is
//! neither judged nor learned (decision "6b: step-time drift during KV promotions", C): the
//! hierarchy's own copies slow decode 1.5–2.4× (OLMoE `l0` promotions), and drift is meant to
//! measure the device, not our transfers. Such steps still count for throughput (the engine's
//! step EWMAs); the window counts them ([`DecodeStepWindow::kv_copy_excluded`]).

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Pure decode iterations the window keeps.
pub const STEP_WINDOW: usize = 64;
/// Calm steps a bucket needs before its steps are judged.
pub const MIN_BUCKET_SAMPLES: u32 = 8;
/// Smoothing of a bucket's baseline.
const BASELINE_ALPHA: f64 = 0.1;
/// How long a judged step stays in the window (the default `deescalate_dwell`).
pub const STEP_MAX_AGE: Duration = Duration::from_secs(10);
/// Judged steps (within [`STEP_MAX_AGE`]) the window needs before it has a p95: with 20 the p95
/// is the second-largest, so one slow step is not drift.
pub const MIN_JUDGED_STEPS: usize = 20;

/// One executed iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepSample {
    pub prefill_tokens: u32,
    /// Sequences decoding in the step.
    pub rows: u32,
    /// Context tokens those sequences attend over (the sum of their context lengths).
    pub context_tokens: u64,
    pub secs: f64,
    /// When the step completed (the engine's monotonic clock).
    pub at: Duration,
    /// A KV tier copy was in flight at some point while the step ran (the engine's test:
    /// `CopyMark` in `turbine-server`'s KV orchestrator).
    pub copy_overlap: bool,
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
/// completion time and its time over its shape bucket's calm baseline.
#[derive(Clone, Debug, Default)]
pub struct DecodeStepWindow {
    steps: VecDeque<(Duration, f64)>,
    baselines: HashMap<Bucket, Baseline>,
    kv_copy_excluded: u64,
}

impl DecodeStepWindow {
    pub fn new() -> Self {
        DecodeStepWindow {
            steps: VecDeque::with_capacity(STEP_WINDOW),
            baselines: HashMap::new(),
            kv_copy_excluded: 0,
        }
    }

    /// One executed iteration. Iterations with a prefill are skipped, so a prefill is not
    /// mistaken for slowing down, and so are decode steps that overlapped a KV tier copy
    /// (counted in [`DecodeStepWindow::kv_copy_excluded`]). The step is judged against its bucket's baseline (before this
    /// step updates it); `calm` steps then update that baseline. Returns the step's judged ratio
    /// (`None`: skipped or not judged), for the engine's `decode_step` debug trace.
    pub fn observe(&mut self, s: StepSample, calm: bool) -> Option<f64> {
        if s.rows == 0 || s.prefill_tokens > 0 {
            return None;
        }
        if s.copy_overlap {
            self.kv_copy_excluded += 1;
            return None;
        }
        let key = bucket(s.rows, s.context_tokens);
        let base = self.baselines.get(&key).copied().unwrap_or_default();
        let mut judged = None;
        if base.samples >= MIN_BUCKET_SAMPLES && base.secs > 0.0 {
            if self.steps.len() == STEP_WINDOW {
                self.steps.pop_front();
            }
            let ratio = s.secs / base.secs;
            self.steps.push_back((s.at, ratio));
            judged = Some(ratio);
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
        judged
    }

    /// Pure decode steps not judged (nor learned) because they overlapped a KV tier copy.
    pub fn kv_copy_excluded(&self) -> u64 {
        self.kv_copy_excluded
    }

    /// Forget every baseline and judged step (the circuit came back from PROBING: the device
    /// may have changed, P3 S-12).
    pub fn reset(&mut self) {
        self.steps.clear();
        self.baselines.clear();
    }

    /// The window's p95 of step time over its bucket's baseline (about 1 while the device
    /// performs as in calm steps) over the judged steps of the last [`STEP_MAX_AGE`] before
    /// `now`; `None` with fewer than [`MIN_JUDGED_STEPS`] of them.
    pub fn p95(&self, now: Duration) -> Option<f64> {
        let mut sorted: Vec<f64> = self
            .steps
            .iter()
            .filter(|(at, _)| now.saturating_sub(*at) <= STEP_MAX_AGE)
            .map(|&(_, r)| r)
            .collect();
        if sorted.len() < MIN_JUDGED_STEPS {
            return None;
        }
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
            at: Duration::ZERO,
            copy_overlap: false,
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
        let p95 = w.p95(Duration::ZERO).unwrap();
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
        assert!((w.p95(Duration::ZERO).unwrap() - 2.0).abs() < 1e-9);
        w.reset();
        assert_eq!(
            w.p95(Duration::ZERO),
            None,
            "reset forgets baselines and steps"
        );
    }

    /// A shape never seen while calm is not judged (no extrapolation), a bucket needs
    /// `MIN_BUCKET_SAMPLES` calm steps, and a prefill in the batch is not a decode step.
    #[test]
    fn unseen_shapes_and_prefills_are_not_judged() {
        let mut w = DecodeStepWindow::new();
        for _ in 0..MIN_BUCKET_SAMPLES - 1 {
            w.observe(step(2, 1000, 0.01), true);
        }
        assert_eq!(
            w.observe(step(2, 1000, 0.05), false),
            None,
            "bucket not ready"
        );
        assert_eq!(w.p95(Duration::ZERO), None, "bucket not ready");
        w.observe(step(2, 1000, 0.01), true);
        w.observe(step(23, 92_000, 0.2), false);
        assert_eq!(w.p95(Duration::ZERO), None, "unseen shape");
        w.observe(
            StepSample {
                prefill_tokens: 2048,
                ..step(2, 1000, 0.5)
            },
            false,
        );
        assert_eq!(w.p95(Duration::ZERO), None, "prefill skipped");
        for _ in 0..MIN_JUDGED_STEPS {
            let judged = w.observe(step(2, 1000, 0.02), false);
            assert!(judged.is_some_and(|r| (r - 2.0).abs() < 1e-9), "{judged:?}");
        }
        assert!((w.p95(Duration::ZERO).unwrap() - 2.0).abs() < 1e-9);
    }

    /// The window's p95 needs [`MIN_JUDGED_STEPS`] judged steps: with fewer it is the window's
    /// maximum, so one slow step (a first launch, a graph capture; one in ~11,500 Llama
    /// multi-turn decode steps took 7.3× its baseline) reads as sustained drift. That is how a
    /// server 2.4 s after ready went RED. Catches: a p95 taken over a handful of steps.
    #[test]
    fn one_slow_step_in_a_short_window_is_not_drift() {
        let mut w = DecodeStepWindow::new();
        for _ in 0..MIN_BUCKET_SAMPLES {
            w.observe(step(16, 55_000, 0.017), true);
        }
        w.observe(step(16, 55_000, 0.017), true);
        w.observe(step(16, 55_000, 0.124), true);
        assert_eq!(w.p95(Duration::ZERO), None, "two judged steps");
        for _ in 2..MIN_JUDGED_STEPS {
            w.observe(step(16, 55_000, 0.019), true);
        }
        let p95 = w.p95(Duration::ZERO).unwrap();
        assert!(
            p95 < 1.5,
            "one slow step among {MIN_JUDGED_STEPS} reads {p95}"
        );
    }

    /// Catches: a judged step kept past [`STEP_MAX_AGE`], so a spike latches the drift after its
    /// shape stopped running (the 2026-10-02 stuck RED).
    #[test]
    fn judged_steps_age_out() {
        let mut w = DecodeStepWindow::new();
        let at = |secs: u64, mut s: StepSample| {
            s.at = Duration::from_secs(secs);
            s
        };
        for _ in 0..MIN_BUCKET_SAMPLES {
            w.observe(at(0, step(16, 55_000, 0.017)), true);
        }
        for _ in 0..MIN_JUDGED_STEPS {
            w.observe(at(1, step(16, 55_000, 0.034)), false);
        }
        assert!((w.p95(Duration::from_secs(1)).unwrap() - 2.0).abs() < 1e-9);
        assert!(w.p95(Duration::from_secs(1) + STEP_MAX_AGE).is_some());
        // Unseen shapes are not judged and do not refresh the window.
        w.observe(at(5, step(1, 3_500, 0.015)), false);
        assert_eq!(w.p95(Duration::from_secs(2) + STEP_MAX_AGE), None);
        // Newer judged steps count.
        for _ in 0..MIN_JUDGED_STEPS {
            w.observe(at(20, step(16, 55_000, 0.017)), false);
        }
        assert!((w.p95(Duration::from_secs(21)).unwrap() - 1.0).abs() < 1e-9);
    }

    /// A decode step that overlapped an in-flight KV tier copy is neither judged nor learned
    /// (decision "6b: step-time drift during KV promotions", C): OLMoE `l0` decode steps run
    /// 1.5–2.4× slower while an L1 → L0 promotion is in flight, and the controller read the
    /// hierarchy's own copy traffic as device drift (YELLOW 24 s into every A/B run). Catches:
    /// overlapped steps judged (the p95 reads 2), or calm overlapped steps raising the baseline
    /// (a later real 2× slowdown then reads under 2), or excluded steps not counted.
    #[test]
    fn steps_overlapping_a_kv_copy_are_not_judged() {
        let mut w = DecodeStepWindow::new();
        let copy = |mut s: StepSample| {
            s.copy_overlap = true;
            s
        };
        for _ in 0..MIN_BUCKET_SAMPLES {
            w.observe(step(16, 55_000, 0.017), true);
        }
        // Calm, but each one overlapped a promotion: not learned.
        for _ in 0..4 * MIN_BUCKET_SAMPLES {
            assert_eq!(w.observe(copy(step(16, 55_000, 0.034)), true), None);
        }
        for _ in 0..STEP_WINDOW {
            assert_eq!(w.observe(copy(step(16, 55_000, 0.034)), false), None);
        }
        assert_eq!(w.p95(Duration::ZERO), None, "no step was judged");
        assert_eq!(
            w.kv_copy_excluded(),
            (4 * MIN_BUCKET_SAMPLES as usize + STEP_WINDOW) as u64
        );
        // A prefill that overlapped a copy is skipped as a prefill, not counted as excluded.
        w.observe(
            StepSample {
                prefill_tokens: 512,
                ..copy(step(16, 55_000, 0.5))
            },
            false,
        );
        assert_eq!(
            w.kv_copy_excluded(),
            (4 * MIN_BUCKET_SAMPLES as usize + STEP_WINDOW) as u64
        );
        // The baseline is still the copy-free 0.017 s: a real 2× slowdown reads 2.
        for _ in 0..MIN_JUDGED_STEPS {
            let judged = w.observe(step(16, 55_000, 0.034), false);
            assert!(judged.is_some_and(|r| (r - 2.0).abs() < 1e-9), "{judged:?}");
        }
        assert!((w.p95(Duration::ZERO).unwrap() - 2.0).abs() < 1e-9);
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
