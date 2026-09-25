//! Model-layer metrics (P1 S-14): load time and weight bytes of the served model, and the
//! duration of every forward pass by phase.
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::EncodeLabelSet;
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

/// 0.5 ms doubling to ~16 s: a tiny-model decode step up to a long prefill.
fn forward_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.0005, 2.0, 16))
}

/// `turbine_model_load_seconds`, `turbine_model_weight_bytes{format}` and
/// `turbine_forward_seconds{phase}`. Cloning shares the registered metrics.
#[derive(Clone, Debug)]
pub struct ModelMetrics {
    /// Wall time of the model load (weights to device, executor built).
    pub load_seconds: Gauge<f64, AtomicU64>,
    /// Bytes of loaded weights by storage format.
    pub weight_bytes: Family<WeightFormatLabels, Gauge>,
    /// Duration of each forward pass by phase.
    pub forward_seconds: Family<PhaseLabels, Histogram, fn() -> Histogram>,
}

impl ModelMetrics {
    /// Registers the three model metric families in `reg`.
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
        }
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
    }
}
