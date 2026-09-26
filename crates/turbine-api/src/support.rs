//! `turbine_support_matrix_status` (phase 8 S-2): registered by `turbine-server` at startup
//! with the resolved status of the running configuration.
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use turbine_core::support::SupportStatus;
use turbine_observability::MetricsRegistry;

/// Label set of `turbine_support_matrix_status`; `status` ∈ supported, experimental, unsupported.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SupportStatusLabels {
    pub status: &'static str,
}

const STATUSES: [&str; 3] = ["supported", "experimental", "unsupported"];

#[derive(Clone)]
pub struct SupportMetrics {
    status: Family<SupportStatusLabels, Gauge>,
}

impl SupportMetrics {
    /// Registers `turbine_support_matrix_status` with all three label values at 0.
    pub fn register(reg: &MetricsRegistry) -> Self {
        let status = reg.register(
            "turbine_support_matrix_status",
            "Support-matrix status of the running configuration (1 for the resolved status)",
            Family::<SupportStatusLabels, Gauge>::default(),
        );
        for s in STATUSES {
            status
                .get_or_create(&SupportStatusLabels { status: s })
                .set(0);
        }
        Self { status }
    }

    /// Sets the resolved status to 1 and the other two to 0.
    pub fn set(&self, resolved: &SupportStatus) {
        for s in STATUSES {
            let v = i64::from(s == resolved.as_str());
            self.status
                .get_or_create(&SupportStatusLabels { status: s })
                .set(v);
        }
    }
}
