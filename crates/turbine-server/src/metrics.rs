//! Server-owned metrics (contract §17): request outcomes, per-request latency and token
//! counts. Label values come from closed sets rendered by `as_str()`.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_core::request::Endpoint;
use turbine_observability::MetricsRegistry;

use crate::engine::stages::{IterationStages, Stage};

/// `turbine_requests_total{outcome}` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// The request finished (`stop` or `length`).
    Ok,
    /// Cancelled before it finished: the client went away, a timeout fired or shutdown.
    Cancelled,
    /// Refused before generation started (validation, context length, full queue, queue timeout).
    Rejected,
    /// Generation failed (kernel or device error).
    Failed,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Cancelled => "cancelled",
            Outcome::Rejected => "rejected",
            Outcome::Failed => "failed",
        }
    }
}

/// `turbine_tokens_total{kind}` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Prompt,
    Generated,
}

impl TokenKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TokenKind::Prompt => "prompt",
            TokenKind::Generated => "generated",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabels {
    pub endpoint: &'static str,
    pub outcome: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TokenLabels {
    pub kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct StageLabels {
    pub stage: &'static str,
}

/// 50 µs doubling to ~0.82 s (15 buckets, then `+Inf`).
fn stage_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.000_05, 2.0, 15))
}

/// Handles to the server's metrics; clones share the registered series.
#[derive(Clone, Debug)]
pub struct ServerMetrics {
    /// `turbine_requests_total{endpoint,outcome}`.
    pub requests: Family<RequestLabels, Counter>,
    /// `turbine_request_ttft_seconds`: request arrival to the first generated token.
    pub ttft: Histogram,
    /// `turbine_request_itl_seconds`: gap between consecutive generated tokens.
    pub itl: Histogram,
    /// `turbine_request_e2e_seconds`: request arrival to the end of generation.
    pub e2e: Histogram,
    /// `turbine_tokens_total{kind}`.
    pub tokens: Family<TokenLabels, Counter>,
    /// `turbine_stream_paused_total`: requests paused because their output channel was full.
    pub stream_paused: Counter,
    /// `turbine_engine_iteration_seconds{stage}`: one engine iteration's time per stage.
    pub engine_iteration: Family<StageLabels, Histogram, fn() -> Histogram>,
}

impl ServerMetrics {
    pub fn register(reg: &MetricsRegistry) -> ServerMetrics {
        ServerMetrics {
            // prometheus-client appends `_total` to counter names.
            requests: reg.register(
                "turbine_requests",
                "Inference requests by endpoint and outcome",
                Family::default(),
            ),
            ttft: reg.register(
                "turbine_request_ttft_seconds",
                "Time from request arrival to the first generated token",
                Histogram::new(exponential_buckets(0.001, 2.0, 16)),
            ),
            itl: reg.register(
                "turbine_request_itl_seconds",
                "Time between consecutive generated tokens of one request",
                Histogram::new(exponential_buckets(0.0005, 2.0, 16)),
            ),
            e2e: reg.register(
                "turbine_request_e2e_seconds",
                "Time from request arrival to the end of generation",
                Histogram::new(exponential_buckets(0.001, 2.0, 20)),
            ),
            tokens: reg.register(
                "turbine_tokens",
                "Prompt and generated tokens",
                Family::default(),
            ),
            stream_paused: reg.register(
                "turbine_stream_paused",
                "Requests paused because their output channel was full",
                Counter::default(),
            ),
            engine_iteration: reg.register(
                "turbine_engine_iteration_seconds",
                "Time of one engine iteration spent in each stage",
                Family::<StageLabels, Histogram, fn() -> Histogram>::new_with_constructor(
                    stage_histogram,
                ),
            ),
        }
    }

    /// Observes every stage of one executed iteration.
    pub fn observe_stages(&self, stages: &IterationStages) {
        for stage in Stage::ALL {
            self.engine_iteration
                .get_or_create(&StageLabels {
                    stage: stage.as_str(),
                })
                .observe(stages.get(stage).as_secs_f64());
        }
    }

    pub fn request(&self, endpoint: Endpoint, outcome: Outcome) {
        self.requests
            .get_or_create(&RequestLabels {
                endpoint: endpoint.as_str(),
                outcome: outcome.as_str(),
            })
            .inc();
    }

    pub fn add_tokens(&self, kind: TokenKind, n: u64) {
        self.tokens
            .get_or_create(&TokenLabels {
                kind: kind.as_str(),
            })
            .inc_by(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_phase1_names_and_labels() {
        let reg = MetricsRegistry::new();
        let m = ServerMetrics::register(&reg);
        m.request(Endpoint::Completions, Outcome::Ok);
        m.request(Endpoint::ChatCompletions, Outcome::Cancelled);
        m.add_tokens(TokenKind::Prompt, 7);
        m.add_tokens(TokenKind::Generated, 3);
        m.ttft.observe(0.01);
        m.itl.observe(0.001);
        m.e2e.observe(0.1);
        m.stream_paused.inc();
        let mut stages = IterationStages::default();
        stages.0[Stage::Launch as usize] = std::time::Duration::from_micros(70);
        m.observe_stages(&stages);
        let text = reg.render().expect("render");
        for line in [
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/chat/completions",outcome="cancelled"} 1"#,
            r#"turbine_tokens_total{kind="prompt"} 7"#,
            r#"turbine_tokens_total{kind="generated"} 3"#,
            "turbine_request_ttft_seconds_count 1",
            "turbine_request_itl_seconds_count 1",
            "turbine_request_e2e_seconds_count 1",
            "turbine_stream_paused_total 1",
            r#"turbine_engine_iteration_seconds_count{stage="device_wait"} 1"#,
            r#"turbine_engine_iteration_seconds_bucket{le="0.00005",stage="launch"} 0"#,
            r#"turbine_engine_iteration_seconds_bucket{le="0.0001",stage="launch"} 1"#,
            r#"turbine_engine_iteration_seconds_bucket{le="0.8192",stage="complete"} 1"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }
}
