//! Shared Prometheus registry rendered as OpenMetrics text at `GET /metrics`.

use std::sync::{Arc, Mutex, MutexGuard};

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::{Metric, Registry};

use crate::ObservabilityError;

/// Content type of the `GET /metrics` response.
pub const OPENMETRICS_CONTENT_TYPE: &str =
    "application/openmetrics-text; version=1.0.0; charset=utf-8";

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct BuildInfoLabels {
    version: String,
}

/// Cloneable handle to the process-wide registry; every crate registers its metrics here.
#[derive(Clone)]
pub struct MetricsRegistry(Arc<Mutex<Registry>>);

impl MetricsRegistry {
    /// A fresh registry already holding `turbine_build_info{version="<crate version>"} 1`.
    pub fn new() -> Self {
        let reg = MetricsRegistry(Arc::new(Mutex::new(Registry::default())));
        let build_info = reg.register(
            "turbine_build_info",
            "Turbine build information; always 1",
            Family::<BuildInfoLabels, Gauge>::default(),
        );
        build_info
            .get_or_create(&BuildInfoLabels {
                version: env!("CARGO_PKG_VERSION").to_string(),
            })
            .set(1);
        reg
    }

    /// Register `metric` under `name` and return the same shared handle.
    pub fn register<M: Metric + Clone>(&self, name: &str, help: &str, metric: M) -> M {
        self.lock().register(name, help, metric.clone());
        metric
    }

    /// Render every registered metric as OpenMetrics text (ending with `# EOF`).
    pub fn render(&self) -> Result<String, ObservabilityError> {
        let mut out = String::new();
        prometheus_client::encoding::text::encode(&mut out, &self.lock())
            .map_err(|e| ObservabilityError::Render(e.to_string()))?;
        Ok(out)
    }

    // Registration and encoding never leave the registry half-updated, so a poisoned
    // lock (a panic elsewhere while holding it) is safe to keep using.
    fn lock(&self) -> MutexGuard<'_, Registry> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use prometheus_client::metrics::counter::Counter;

    use super::*;

    #[test]
    fn build_info_and_registered_metrics_render() {
        let reg = MetricsRegistry::new();
        let events = reg.register(
            "turbine_test_events",
            "test counter",
            Counter::<u64>::default(),
        );
        events.inc();
        // A clone shares the same underlying registry.
        let text = reg.clone().render().unwrap();
        assert!(
            text.contains("turbine_build_info{version=\"0.1.0\"} 1"),
            "{text}"
        );
        assert!(text.contains("turbine_test_events_total 1"), "{text}");
        assert!(text.ends_with("# EOF\n"), "{text}");
    }
}
