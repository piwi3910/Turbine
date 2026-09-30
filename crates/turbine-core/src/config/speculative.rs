//! `speculative.*`: the umbrella owns `method` only, because the support matrix (S-2) keys on it;
//! `num_tokens`, `draft_model_path` and `min_acceptance` are added by the phase-8-speculative-decoding plan.
use serde::{Deserialize, Serialize};

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

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct SpeculativeConfig {
    pub method: SpeculativeMethod,
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
        assert!(serde_norway::from_str::<SpeculativeConfig>("num_tokens: 4").is_err());
    }
}
