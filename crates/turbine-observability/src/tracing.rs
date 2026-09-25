//! Global `tracing` subscriber: text or JSON to stderr, `RUST_LOG` overriding `logging.level`.

use tracing_subscriber::EnvFilter;
use turbine_core::config::{LogFormat, LoggingConfig};

use crate::ObservabilityError;

/// Install the global subscriber. `RUST_LOG`, when set, replaces `cfg.level`.
///
/// Fails with [`ObservabilityError::Filter`] when the directive does not parse and with
/// [`ObservabilityError::Init`] when a global subscriber is already installed.
pub fn init_tracing(cfg: &LoggingConfig) -> Result<(), ObservabilityError> {
    let directive = std::env::var("RUST_LOG").unwrap_or_else(|_| cfg.level.clone());
    let filter = EnvFilter::try_new(&directive)
        .map_err(|e| ObservabilityError::Filter(format!("{directive}: {e}")))?;
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    let result = match cfg.format {
        LogFormat::Json => builder.json().try_init(),
        _ => builder.try_init(),
    };
    result.map_err(|e| ObservabilityError::Init(e.to_string()))
}
