//! Server-owned Phase 1 metrics (contract §17): request outcomes, per-request latency and token
//! counts. Label values come from closed sets rendered by `as_str()`.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_core::request::Endpoint;
use turbine_observability::MetricsRegistry;

/// `turbine_requests_total{outcome}` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// The request finished (`stop` or `length`).
    Ok,
    /// The client went away before the request finished.
    Cancelled,
    /// Refused before generation started (validation, context length, busy slot).
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
        let text = reg.render().expect("render");
        for line in [
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/chat/completions",outcome="cancelled"} 1"#,
            r#"turbine_tokens_total{kind="prompt"} 7"#,
            r#"turbine_tokens_total{kind="generated"} 3"#,
            "turbine_request_ttft_seconds_count 1",
            "turbine_request_itl_seconds_count 1",
            "turbine_request_e2e_seconds_count 1",
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }
}
