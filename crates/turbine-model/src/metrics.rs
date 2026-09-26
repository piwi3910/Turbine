//! Model-layer metrics (P1 S-14, P2 S-17, S-18): load time and weight bytes of the served model,
//! the duration of every forward pass by phase, grammar compilation, token-mask computation and
//! tool-call parser outcomes.
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_observability::MetricsRegistry;

/// Which kind of forward pass a duration belongs to: a prompt (prefill) or a one-token step
/// (decode).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ForwardPhase {
    Prefill,
    Decode,
}

impl ForwardPhase {
    /// The metric label value.
    pub fn as_str(self) -> &'static str {
        match self {
            ForwardPhase::Prefill => "prefill",
            ForwardPhase::Decode => "decode",
        }
    }
}

/// Labels of `turbine_forward_seconds`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PhaseLabels {
    phase: &'static str,
}

/// Labels of `turbine_model_weight_bytes`; `format` is `bf16` in Phase 1.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct WeightFormatLabels {
    pub format: String,
}

/// Labels of `turbine_grammar_compile_seconds`: `json_object`, `json_schema` or `tool_call`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct GrammarKindLabels {
    kind: &'static str,
}

/// 1 ms doubling to ~33 s: a trivial grammar up to a compile hitting its timeout.
fn grammar_compile_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.001, 2.0, 16))
}

/// Whether a tool-call parser turned a choice's output into calls (P2 S-18).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ToolCallOutcome {
    Parsed,
    ParseFailed,
}

impl ToolCallOutcome {
    /// The metric label value.
    pub fn as_str(self) -> &'static str {
        match self {
            ToolCallOutcome::Parsed => "parsed",
            ToolCallOutcome::ParseFailed => "parse_failed",
        }
    }
}

/// Labels of `turbine_tool_calls_total`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ToolCallLabels {
    parser: &'static str,
    outcome: &'static str,
}

/// 0.5 ms doubling to ~16 s: a tiny-model decode step up to a long prefill.
fn forward_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.0005, 2.0, 16))
}

/// `turbine_model_load_seconds`, `turbine_model_weight_bytes{format}`,
/// `turbine_forward_seconds{phase}`, `turbine_grammar_compile_seconds{kind}` and
/// `turbine_token_mask_seconds`. Cloning shares the registered metrics.
#[derive(Clone, Debug)]
pub struct ModelMetrics {
    /// Wall time of the model load (weights to device, executor built).
    pub load_seconds: Gauge<f64, AtomicU64>,
    /// Bytes of loaded weights by storage format.
    pub weight_bytes: Family<WeightFormatLabels, Gauge>,
    /// Duration of each forward pass by phase.
    pub forward_seconds: Family<PhaseLabels, Histogram, fn() -> Histogram>,
    /// Duration of each successful grammar compilation by constraint kind.
    pub grammar_compile_seconds: Family<GrammarKindLabels, Histogram, fn() -> Histogram>,
    /// Duration of each per-step allowed-token mask computation.
    pub token_mask_seconds: Histogram,
    /// Tool-call parser results by parser and outcome (P2 S-18).
    pub tool_calls: Family<ToolCallLabels, Counter>,
}

impl ModelMetrics {
    /// Registers the model metric families in `reg`.
    pub fn register(reg: &MetricsRegistry) -> ModelMetrics {
        ModelMetrics {
            load_seconds: reg.register(
                "turbine_model_load_seconds",
                "Wall time of the model load in seconds",
                Gauge::<f64, AtomicU64>::default(),
            ),
            weight_bytes: reg.register(
                "turbine_model_weight_bytes",
                "Bytes of loaded model weights by storage format",
                Family::default(),
            ),
            forward_seconds: reg.register(
                "turbine_forward_seconds",
                "Duration of one model forward pass by phase",
                Family::<PhaseLabels, Histogram, fn() -> Histogram>::new_with_constructor(
                    forward_histogram,
                ),
            ),
            grammar_compile_seconds: reg.register(
                "turbine_grammar_compile_seconds",
                "Duration of one constrained-decoding grammar compilation by kind",
                Family::<GrammarKindLabels, Histogram, fn() -> Histogram>::new_with_constructor(
                    grammar_compile_histogram,
                ),
            ),
            token_mask_seconds: reg.register(
                "turbine_token_mask_seconds",
                "Duration of one allowed-token mask computation",
                // 10 µs doubling to ~0.3 s
                Histogram::new(exponential_buckets(0.000_01, 2.0, 16)),
            ),
            tool_calls: reg.register(
                // prometheus-client appends `_total` to counters
                "turbine_tool_calls",
                "Tool-call parser results by parser and outcome",
                Family::default(),
            ),
        }
    }

    /// Counts one tool-call parse of a choice's output by `parser` (e.g. `llama3_json`).
    pub fn record_tool_call(&self, parser: &'static str, outcome: ToolCallOutcome) {
        self.tool_calls
            .get_or_create(&ToolCallLabels {
                parser,
                outcome: outcome.as_str(),
            })
            .inc();
    }

    /// Records a finished model load: its duration and the weight bytes in `format`.
    pub fn record_load(&self, seconds: f64, format: &str, weight_bytes: u64) {
        self.load_seconds.set(seconds);
        self.weight_bytes
            .get_or_create(&WeightFormatLabels {
                format: format.to_string(),
            })
            .set(i64::try_from(weight_bytes).unwrap_or(i64::MAX));
    }

    /// Adds one forward-pass duration to `turbine_forward_seconds{phase}`.
    pub fn observe_forward(&self, phase: ForwardPhase, seconds: f64) {
        self.forward_seconds
            .get_or_create(&PhaseLabels {
                phase: phase.as_str(),
            })
            .observe(seconds);
    }

    /// Adds one grammar compilation to `turbine_grammar_compile_seconds{kind}`.
    pub fn observe_grammar_compile(&self, kind: &'static str, seconds: f64) {
        self.grammar_compile_seconds
            .get_or_create(&GrammarKindLabels { kind })
            .observe(seconds);
    }

    /// Adds one mask computation to `turbine_token_mask_seconds`.
    pub fn observe_token_mask(&self, seconds: f64) {
        self.token_mask_seconds.observe(seconds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_model_metrics_with_labels() {
        let reg = MetricsRegistry::new();
        let m = ModelMetrics::register(&reg);
        m.record_load(1.5, "bf16", 6_425_499_648);
        m.observe_forward(ForwardPhase::Prefill, 0.25);
        m.observe_forward(ForwardPhase::Decode, 0.01);
        m.observe_forward(ForwardPhase::Decode, 0.02);
        m.observe_grammar_compile("json_schema", 0.2);
        m.observe_token_mask(0.0001);
        let text = reg.render().expect("render");
        assert!(text.contains("turbine_model_load_seconds 1.5"), "{text}");
        assert!(
            text.contains("turbine_model_weight_bytes{format=\"bf16\"} 6425499648"),
            "{text}"
        );
        assert!(
            text.contains("turbine_forward_seconds_count{phase=\"prefill\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("turbine_forward_seconds_count{phase=\"decode\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("turbine_grammar_compile_seconds_count{kind=\"json_schema\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("turbine_token_mask_seconds_count 1"),
            "{text}"
        );
    }

    #[test]
    fn counts_tool_calls_by_parser_and_outcome() {
        let reg = MetricsRegistry::new();
        let m = ModelMetrics::register(&reg);
        m.record_tool_call("llama3_json", ToolCallOutcome::Parsed);
        m.record_tool_call("llama3_json", ToolCallOutcome::Parsed);
        m.record_tool_call("llama3_json", ToolCallOutcome::ParseFailed);
        let text = reg.render().expect("render");
        assert!(
            text.contains("turbine_tool_calls_total{parser=\"llama3_json\",outcome=\"parsed\"} 2"),
            "{text}"
        );
        assert!(
            text.contains(
                "turbine_tool_calls_total{parser=\"llama3_json\",outcome=\"parse_failed\"} 1"
            ),
            "{text}"
        );
    }
}
