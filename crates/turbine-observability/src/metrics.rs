//! Shared Prometheus registry rendered as OpenMetrics text at `GET /metrics`.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicI64, Ordering};
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

/// One contributor's share of a gauge that several contributors add up, e.g. a queue-length
/// gauge that every data-parallel replica's engine sets for itself (P5 S-7): each share
/// remembers the value it last set and moves the gauge by the difference, so the gauge is the
/// sum of the shares. A clone is the same share; [`GaugeShare::another`] is a new one at 0.
#[derive(Clone, Debug, Default)]
pub struct GaugeShare {
    gauge: Gauge,
    last: Arc<AtomicI64>,
}

impl GaugeShare {
    /// The first share of `gauge`, at 0.
    pub fn new(gauge: Gauge) -> GaugeShare {
        GaugeShare {
            gauge,
            last: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Another contributor to the same gauge, starting at 0.
    pub fn another(&self) -> GaugeShare {
        GaugeShare::new(self.gauge.clone())
    }

    /// Sets this share to `value`; the gauge moves by the change.
    pub fn set(&self, value: i64) {
        let previous = self.last.swap(value, Ordering::AcqRel);
        self.gauge.inc_by(value - previous);
    }

    /// The gauge, for registration.
    pub fn gauge(&self) -> &Gauge {
        &self.gauge
    }

    /// The gauge's total over every share.
    pub fn total(&self) -> i64 {
        self.gauge.get()
    }
}

/// [`GaugeShare`] for every label set of a gauge family: one contributor's share of each
/// series, so several data-parallel replicas add up per series.
#[derive(Clone, Debug)]
pub struct GaugeFamilyShare<L: EncodeLabelSet + Clone + Eq + Hash + Debug + Send + Sync + 'static> {
    family: Family<L, Gauge>,
    last: Arc<Mutex<HashMap<L, i64>>>,
}

impl<L: EncodeLabelSet + Clone + Eq + Hash + Debug + Send + Sync + 'static> GaugeFamilyShare<L> {
    /// The first share of `family`, at 0 for every series.
    pub fn new(family: Family<L, Gauge>) -> GaugeFamilyShare<L> {
        GaugeFamilyShare {
            family,
            last: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Another contributor to the same family, at 0 for every series.
    pub fn another(&self) -> GaugeFamilyShare<L> {
        GaugeFamilyShare::new(self.family.clone())
    }

    /// The family, for registration.
    pub fn family(&self) -> &Family<L, Gauge> {
        &self.family
    }

    /// Creates the series of `labels` (so it renders from startup) without changing it.
    pub fn touch(&self, labels: &L) {
        let _ = self.family.get_or_create(labels);
    }

    /// Sets this share of the series `labels` to `value`; the series moves by the change.
    pub fn set(&self, labels: &L, value: i64) {
        let previous = {
            let mut last = self.last.lock().unwrap_or_else(|p| p.into_inner());
            last.insert(labels.clone(), value).unwrap_or(0)
        };
        self.family.get_or_create(labels).inc_by(value - previous);
    }
}

#[cfg(test)]
mod tests {
    use prometheus_client::metrics::counter::Counter;

    use super::*;

    /// Shares add up: two contributors at 3 and 4 make 7; one dropping to 1 makes 5; a clone
    /// is the same share, not a third one. The same per series of a family.
    #[test]
    fn gauge_shares_sum() {
        let gauge = Gauge::<i64>::default();
        let a = GaugeShare::new(gauge.clone());
        let b = a.another();
        a.set(3);
        b.set(4);
        assert_eq!(gauge.get(), 7);
        a.set(1);
        assert_eq!(gauge.get(), 5);
        a.clone().set(2);
        assert_eq!(gauge.get(), 6);
        b.set(0);
        a.set(0);
        assert_eq!(gauge.get(), 0);

        let family = Family::<Vec<(&'static str, &'static str)>, Gauge>::default();
        let x = GaugeFamilyShare::new(family.clone());
        let y = x.another();
        let used = vec![("state", "used")];
        let free = vec![("state", "free")];
        x.set(&used, 10);
        y.set(&used, 5);
        x.set(&free, 2);
        assert_eq!(family.get_or_create(&used).get(), 15);
        y.set(&used, 1);
        assert_eq!(family.get_or_create(&used).get(), 11);
        assert_eq!(family.get_or_create(&free).get(), 2);
    }

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
