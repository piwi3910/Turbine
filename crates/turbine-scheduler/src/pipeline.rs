//! Pipeline-parallel micro-batches (Phase 5 S-10): what the engine needs beside
//! `Scheduler::with_micro_batches` — the sequences of a micro-batch, the per-stage busy
//! timeline behind `turbine_pipeline_bubble_ratio` and the pipeline metric families.
//!
//! Engine loop with `m` micro-batches and `s` stages: whenever stage 0 is free and fewer than
//! `m` plans are in flight, `Scheduler::plan` the next micro-batch (an empty plan is completed
//! at once) and send it into stage 0; a micro-batch leaving stage `k` enters stage `k + 1`
//! when that stage is free; when it leaves the last stage its tokens are sampled and
//! `Scheduler::complete` gets its outcome (any order). Every stage execution is recorded with
//! [`StageTimeline::record`] and [`PipelineMetrics::observe_stage`].

use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_core::types::SeqId;
use turbine_observability::MetricsRegistry;

use crate::scheduler::IterationPlan;

/// One micro-batch: the plan's slot in the pipeline and the sequences it carries (its batch
/// items, then the destinations of its forks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicroBatchPlan {
    pub micro_batch: u32,
    pub seqs: Vec<SeqId>,
}

impl MicroBatchPlan {
    pub fn new(micro_batch: u32, plan: &IterationPlan) -> MicroBatchPlan {
        MicroBatchPlan {
            micro_batch,
            seqs: plan
                .items
                .iter()
                .map(|i| i.seq)
                .chain(plan.forks.iter().map(|f| f.dst))
                .collect(),
        }
    }
}

/// Busy intervals of each pipeline stage over a sliding window (default 10 s), on the
/// caller's monotonic clock. The bubble ratio is the idle share of stage time:
/// `1 − Σ busy / (stages × span)`.
#[derive(Clone, Debug)]
pub struct StageTimeline {
    stages: Vec<VecDeque<(Duration, Duration)>>,
    window: Duration,
    /// Start of the first recorded interval: a younger pipeline's span starts there.
    first: Option<Duration>,
}

impl StageTimeline {
    /// The window of `turbine_pipeline_bubble_ratio` (spec §Metrics).
    pub const WINDOW: Duration = Duration::from_secs(10);

    pub fn new(stages: usize) -> StageTimeline {
        StageTimeline::with_window(stages, StageTimeline::WINDOW)
    }

    pub fn with_window(stages: usize, window: Duration) -> StageTimeline {
        StageTimeline {
            stages: vec![VecDeque::new(); stages.max(1)],
            window,
            first: None,
        }
    }

    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Stage `stage` executed a micro-batch over `[start, end)`. Intervals of one stage are
    /// recorded in time order; those that ended a window before `end` are dropped.
    pub fn record(&mut self, stage: usize, start: Duration, end: Duration) {
        let Some(q) = self.stages.get_mut(stage) else {
            return;
        };
        q.push_back((start, end.max(start)));
        self.first = Some(self.first.map_or(start, |f| f.min(start)));
        if let Some(horizon) = end.checked_sub(self.window) {
            for q in &mut self.stages {
                while q.front().is_some_and(|(_, e)| *e < horizon) {
                    q.pop_front();
                }
            }
        }
    }

    /// Time stage `stage` was busy within `[from, to)`.
    pub fn busy(&self, stage: usize, from: Duration, to: Duration) -> Duration {
        self.stages.get(stage).map_or(Duration::ZERO, |q| {
            q.iter()
                .map(|&(s, e)| e.min(to).saturating_sub(s.max(from)))
                .sum()
        })
    }

    /// Idle share of stage time within `[from, to)` (0 for an empty span).
    pub fn bubble_ratio_between(&self, from: Duration, to: Duration) -> f64 {
        let span = to.saturating_sub(from);
        if span.is_zero() {
            return 0.0;
        }
        let busy: f64 = (0..self.stages.len())
            .map(|s| self.busy(s, from, to).as_secs_f64())
            .sum();
        (1.0 - busy / (span.as_secs_f64() * self.stages.len() as f64)).clamp(0.0, 1.0)
    }

    /// `turbine_pipeline_bubble_ratio` at `now`: the idle share over the last window, or since
    /// the first recorded execution when that is more recent.
    pub fn bubble_ratio(&self, now: Duration) -> f64 {
        self.bubble_ratio_between(self.window_start(now), now)
    }

    /// The `busy_ratio` of stage `stage` at `now`, over the same span as [`Self::bubble_ratio`].
    pub fn busy_ratio(&self, stage: usize, now: Duration) -> f64 {
        let from = self.window_start(now);
        let span = now.saturating_sub(from);
        if span.is_zero() {
            return 0.0;
        }
        (self.busy(stage, from, now).as_secs_f64() / span.as_secs_f64()).clamp(0.0, 1.0)
    }

    fn window_start(&self, now: Duration) -> Duration {
        let start = now.saturating_sub(self.window);
        self.first.map_or(now, |f| f.max(start))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StageLabels {
    stage: u32,
}

fn stage_histogram() -> Histogram {
    // 50 µs … ~1.6 s.
    Histogram::new(exponential_buckets(0.000_05, 2.0, 16))
}

/// `turbine_pipeline_stage_duration_seconds{stage}` and `turbine_pipeline_bubble_ratio`
/// (Phase 5 S-10). Registered once by the engine that runs the pipeline.
#[derive(Clone)]
pub struct PipelineMetrics {
    stage_duration: Family<StageLabels, Histogram, fn() -> Histogram>,
    bubble_ratio: Gauge<f64, AtomicU64>,
}

impl PipelineMetrics {
    pub fn register(reg: &MetricsRegistry) -> PipelineMetrics {
        PipelineMetrics {
            stage_duration: reg.register(
                "turbine_pipeline_stage_duration_seconds",
                "Duration of one micro-batch on one pipeline stage",
                Family::new_with_constructor(stage_histogram as fn() -> Histogram),
            ),
            bubble_ratio: reg.register(
                "turbine_pipeline_bubble_ratio",
                "Fraction of pipeline stage time idle over the last 10 s",
                Gauge::default(),
            ),
        }
    }

    pub fn observe_stage(&self, stage: u32, seconds: f64) {
        self.stage_duration
            .get_or_create(&StageLabels { stage })
            .observe(seconds);
    }

    pub fn set_bubble_ratio(&self, ratio: f64) {
        self.bubble_ratio.set(ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Two stages, each busy half of 40 ms: bubble 0.5; the window drops old intervals; a
    /// full pipeline has no bubble. Catches a wrong span, double counting or no pruning.
    #[test]
    fn bubble_ratio_over_the_window() {
        let mut t = StageTimeline::with_window(2, ms(100));
        t.record(0, ms(0), ms(10));
        t.record(1, ms(10), ms(20));
        t.record(0, ms(20), ms(30));
        t.record(1, ms(30), ms(40));
        assert!((t.bubble_ratio(ms(40)) - 0.5).abs() < 1e-9);
        assert!((t.busy_ratio(0, ms(40)) - 0.5).abs() < 1e-9);
        // Both stages busy back to back from 40 ms to 240 ms.
        for k in 0..20 {
            t.record(0, ms(40 + 10 * k), ms(50 + 10 * k));
            t.record(1, ms(40 + 10 * k), ms(50 + 10 * k));
        }
        assert!(t.bubble_ratio(ms(240)).abs() < 1e-9);
        assert!(
            t.stages[0].len() <= 11,
            "intervals older than the window are dropped"
        );
    }

    /// The pipeline families render under their spec names.
    #[test]
    fn metrics_render() {
        let reg = MetricsRegistry::new();
        let m = PipelineMetrics::register(&reg);
        m.observe_stage(1, 0.01);
        m.set_bubble_ratio(0.25);
        let text = reg.render().unwrap();
        assert!(
            text.contains("turbine_pipeline_stage_duration_seconds_count{stage=\"1\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("turbine_pipeline_bubble_ratio 0.25"),
            "{text}"
        );
    }
}
