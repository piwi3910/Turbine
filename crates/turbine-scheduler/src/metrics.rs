//! Scheduler metrics (P2 §Metrics). Every automatic decision (reject, preempt, pause,
//! cancel) is counted here and logged with its reason by the scheduler.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_observability::MetricsRegistry;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StateLabels {
    state: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct AdmissionLabels {
    outcome: &'static str,
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct PhaseLabels {
    phase: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ReasonLabels {
    reason: &'static str,
}

fn token_histogram() -> Histogram {
    // 1 … 32768 tokens.
    Histogram::new(exponential_buckets(1.0, 2.0, 16))
}

/// Handles to the Phase 2 scheduler metric families.
#[derive(Clone)]
pub struct SchedulerMetrics {
    requests_active: Family<StateLabels, Gauge>,
    requests_queued: Gauge,
    admission: Family<AdmissionLabels, Counter>,
    queue_wait_seconds: Histogram,
    iteration_seconds: Histogram,
    iteration_tokens: Family<PhaseLabels, Histogram, fn() -> Histogram>,
    batch_requests: Histogram,
    preemptions: Family<ReasonLabels, Counter>,
    cancelled: Family<ReasonLabels, Counter>,
}

impl SchedulerMetrics {
    pub fn register(reg: &MetricsRegistry) -> SchedulerMetrics {
        let m = SchedulerMetrics {
            requests_active: reg.register(
                "turbine_requests_active",
                "Running requests by state",
                Family::default(),
            ),
            requests_queued: reg.register(
                "turbine_requests_queued",
                "Requests in the waiting queue",
                Gauge::default(),
            ),
            admission: reg.register(
                "turbine_admission",
                "Submissions by outcome and reason",
                Family::default(),
            ),
            queue_wait_seconds: reg.register(
                "turbine_queue_wait_seconds",
                "Time from submission to first admission",
                Histogram::new(exponential_buckets(0.001, 2.0, 18)),
            ),
            iteration_seconds: reg.register(
                "turbine_iteration_seconds",
                "Duration of one scheduler iteration (plan to complete)",
                Histogram::new(exponential_buckets(0.000_25, 2.0, 16)),
            ),
            iteration_tokens: reg.register(
                "turbine_iteration_tokens",
                "Tokens per iteration by phase",
                Family::new_with_constructor(token_histogram as fn() -> Histogram),
            ),
            batch_requests: reg.register(
                "turbine_batch_requests",
                "Sequences per iteration batch",
                Histogram::new(exponential_buckets(1.0, 2.0, 12)),
            ),
            preemptions: reg.register(
                "turbine_preemptions",
                "Preempted requests by reason",
                Family::default(),
            ),
            cancelled: reg.register(
                "turbine_requests_cancelled",
                "Cancelled requests by reason",
                Family::default(),
            ),
        };
        // Every state series exists from startup, at 0.
        for state in ["prefilling", "decoding", "paused"] {
            m.set_active(state, 0);
        }
        m
    }

    pub(crate) fn set_active(&self, state: &'static str, n: u32) {
        self.requests_active
            .get_or_create(&StateLabels { state })
            .set(i64::from(n));
    }

    pub(crate) fn set_queued(&self, n: u32) {
        self.requests_queued.set(i64::from(n));
    }

    pub(crate) fn admission(&self, outcome: &'static str, reason: &'static str) {
        self.admission
            .get_or_create(&AdmissionLabels { outcome, reason })
            .inc();
    }

    /// A submission refused before it reaches the scheduler — `invalid_json_schema`, whose
    /// grammar the HTTP side compiles before queueing — counted as
    /// `turbine_admission_total{outcome="rejected",reason}`.
    pub fn record_rejection(&self, reason: &'static str) {
        self.admission("rejected", reason);
    }

    pub(crate) fn queue_wait(&self, seconds: f64) {
        self.queue_wait_seconds.observe(seconds);
    }

    pub(crate) fn iteration(
        &self,
        seconds: f64,
        prefill_tokens: u32,
        decode_tokens: u32,
        seqs: u32,
    ) {
        self.iteration_seconds.observe(seconds);
        self.iteration_tokens
            .get_or_create(&PhaseLabels { phase: "prefill" })
            .observe(f64::from(prefill_tokens));
        self.iteration_tokens
            .get_or_create(&PhaseLabels { phase: "decode" })
            .observe(f64::from(decode_tokens));
        self.batch_requests.observe(f64::from(seqs));
    }

    pub(crate) fn preempted(&self, reason: &'static str) {
        self.preemptions
            .get_or_create(&ReasonLabels { reason })
            .inc();
    }

    pub(crate) fn cancelled(&self, reason: &'static str) {
        self.cancelled.get_or_create(&ReasonLabels { reason }).inc();
    }
}
