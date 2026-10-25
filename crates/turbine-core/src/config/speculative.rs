//! `speculative.*`: the Phase 8 draft-model setting. The umbrella owns `method` only, because the
//! support matrix (S-2) keys on it; `num_tokens`, `draft_model_path` and `min_acceptance` are the
//! Phase 8 track's keys (`.procoder/specs/phase-8-speculative-decoding.md` S-1).
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::ConfigError;

/// The longest greedy proposal a draft may make (`speculative.num_tokens`).
pub const MAX_SPECULATIVE_TOKENS: u32 = 8;

#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum SpeculativeMethod {
    #[default]
    None,
    Draft,
}

impl SpeculativeMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            SpeculativeMethod::None => "none",
            SpeculativeMethod::Draft => "draft",
        }
    }
}

/// `speculative.*`: off by default; `draft` loads one small draft model of the target's family.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct SpeculativeConfig {
    pub method: SpeculativeMethod,
    /// Greedy proposals per round (`speculative.num_tokens`, 1..=8).
    pub num_tokens: u32,
    /// The draft checkpoint directory; required with `method: draft`, unread with `none`.
    pub draft_model_path: Option<PathBuf>,
    /// Disable a request's speculation below this accepted fraction (`speculative.min_acceptance`).
    pub min_acceptance: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        SpeculativeConfig {
            method: SpeculativeMethod::None,
            num_tokens: 4,
            draft_model_path: None,
            min_acceptance: 0.4,
        }
    }
}

impl SpeculativeConfig {
    /// Bounds every key; the draft path presence and the row capacity are draft-only, so a
    /// `method: none` configuration keeps the other keys inert.
    pub fn validate(&self, effective_max_running: u32) -> Result<(), ConfigError> {
        if !(1..=MAX_SPECULATIVE_TOKENS).contains(&self.num_tokens) {
            return Err(ConfigError::Invalid {
                key: "speculative.num_tokens".to_string(),
                reason: format!(
                    "must be between 1 and {MAX_SPECULATIVE_TOKENS}, got {}",
                    self.num_tokens
                ),
            });
        }
        if !(0.0..=1.0).contains(&self.min_acceptance) {
            return Err(ConfigError::Invalid {
                key: "speculative.min_acceptance".to_string(),
                reason: format!("must be between 0 and 1, got {}", self.min_acceptance),
            });
        }
        if self.method != SpeculativeMethod::Draft {
            return Ok(());
        }
        if self
            .draft_model_path
            .as_ref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err(ConfigError::Invalid {
                key: "speculative.draft_model_path".to_string(),
                reason: "is required when speculative.method is draft".to_string(),
            });
        }
        let rows = self.num_tokens.saturating_add(1);
        if rows > effective_max_running {
            return Err(ConfigError::Invalid {
                key: "speculative.num_tokens".to_string(),
                reason: format!(
                    "speculative_rows: num_tokens + 1 ({rows}) exceeds the plan's logits-row \
                     capacity scheduler.effective_max_running() ({effective_max_running})"
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_method_parses() {
        let cfg: SpeculativeConfig = serde_norway::from_str("method: draft").unwrap();
        assert_eq!(cfg.method, SpeculativeMethod::Draft);
        assert_eq!(SpeculativeConfig::default().method, SpeculativeMethod::None);
        assert!(serde_norway::from_str::<SpeculativeConfig>("method: eagle").is_err());
        assert!(serde_norway::from_str::<SpeculativeConfig>("proposer: draft").is_err());
    }

    #[test]
    fn speculative_config_defaults_and_bounds() {
        let cfg: SpeculativeConfig = serde_norway::from_str("{}").unwrap();
        assert_eq!(cfg.method, SpeculativeMethod::None);
        assert_eq!(cfg.num_tokens, 4);
        assert_eq!(cfg.draft_model_path, None);
        assert_eq!(cfg.min_acceptance, 0.4);
        cfg.validate(64).unwrap();

        for (yaml, key) in [
            ("num_tokens: 0", "speculative.num_tokens"),
            ("num_tokens: 9", "speculative.num_tokens"),
            ("min_acceptance: -0.1", "speculative.min_acceptance"),
            ("min_acceptance: 1.1", "speculative.min_acceptance"),
            ("method: draft", "speculative.draft_model_path"),
        ] {
            let cfg: SpeculativeConfig = serde_norway::from_str(yaml).unwrap();
            assert_eq!(cfg.validate(64).unwrap_err().key(), Some(key), "{yaml}");
        }

        // The row capacity is a draft rule; `none` leaves num_tokens inert even at one row.
        let cfg = SpeculativeConfig {
            num_tokens: 8,
            ..Default::default()
        };
        cfg.validate(1).unwrap();

        let cfg = SpeculativeConfig {
            method: SpeculativeMethod::Draft,
            num_tokens: 8,
            draft_model_path: Some("/nonexistent".into()),
            ..Default::default()
        };
        let err = cfg.validate(8).unwrap_err();
        assert!(err.to_string().contains("speculative_rows:"), "{err}");
        cfg.validate(16).unwrap();
    }
}
