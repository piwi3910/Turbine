//! `quality.*` (phase 8 umbrella, S-3/S-4): thresholds used by `turbine-golden eval-compare`.
use serde::{Deserialize, Serialize};

use super::ConfigError;

/// Default for `quality.max_accuracy_drop` (P8 §Interfaces).
pub const DEFAULT_MAX_ACCURACY_DROP: f64 = 0.01;

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct QualityConfig {
    /// Largest accuracy drop (absolute, 0..=0.1) a lossy format may show against its baseline.
    pub max_accuracy_drop: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            max_accuracy_drop: DEFAULT_MAX_ACCURACY_DROP,
        }
    }
}

impl QualityConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let v = self.max_accuracy_drop;
        if !v.is_finite() || !(0.0..=0.1).contains(&v) {
            return Err(ConfigError::Invalid {
                key: "quality.max_accuracy_drop".into(),
                reason: format!("must be between 0 and 0.1, got {v}"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_config_validation() {
        assert_eq!(QualityConfig::default().max_accuracy_drop, 0.01);
        assert!(QualityConfig::default().validate().is_ok());
        for ok in [0.0, 0.05, 0.1] {
            assert!(
                QualityConfig {
                    max_accuracy_drop: ok
                }
                .validate()
                .is_ok(),
                "{ok}"
            );
        }
        for bad in [-0.001, 0.1001, f64::NAN, f64::INFINITY] {
            let err = QualityConfig {
                max_accuracy_drop: bad,
            }
            .validate()
            .unwrap_err();
            assert_eq!(err.key(), Some("quality.max_accuracy_drop"), "{bad}");
        }
    }
}
