//! Tracing subscriber, Prometheus registry and HTTP request-id / metrics layers (contract §4).

pub mod http;
mod metrics;
mod tracing;

pub use metrics::{GaugeFamilyShare, GaugeShare, MetricsRegistry, OPENMETRICS_CONTENT_TYPE};
pub use tracing::init_tracing;

/// The crate's single error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ObservabilityError {
    /// The log filter (`RUST_LOG` or `logging.level`) does not parse.
    #[error("invalid log filter: {0}")]
    Filter(String),
    /// Encoding the registry as OpenMetrics text failed; `/metrics` answers 500.
    #[error("metrics render failed: {0}")]
    Render(String),
    /// A global tracing subscriber is already installed.
    #[error("cannot install the tracing subscriber: {0}")]
    Init(String),
}
