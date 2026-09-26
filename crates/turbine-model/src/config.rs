//! Hugging Face `config.json` / `generation_config.json` parsing and the allowlist (P1 S-3):
//! `architectures[0]` must name a registered model family (`crate::families::resolve`, Phase 2m
//! S-2), whose own keys it parses; the keys every decoder shares are parsed here. Anything else
//! is refused with the offending field named and the supported set listed.
//!
//! The weight-format part of the allowlist is the checkpoint's [`crate::weights::WeightFormat`] (Phase 2m S-10,
//! `crate::weights`): `detect` picks it from `config.json`, and
//! `ModelArchConfig::check_supported_weights` holds every tensor the architecture loads to it.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use smallvec::SmallVec;
use turbine_core::types::{KvLayout, ModelShape};

use crate::ModelError;
use crate::families::{FamilyRef, resolve};
use crate::safetensors::SafetensorsIndex;
use crate::weights::{WeightFormatRef, detect};

/// `config.json` `rope_scaling` (absent or `rope_type: default` → `None`).
#[derive(Clone, Copy, PartialEq, Debug)]
#[non_exhaustive]
pub enum RopeScaling {
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: u32,
    },
}

/// The architecture description of a checkpoint, as far as Turbine uses it.
#[derive(Clone, PartialEq, Debug)]
pub struct ModelArchConfig {
    /// The model family `config.json` `architectures[0]` resolved to.
    pub family: FamilyRef,
    /// `architectures[0]` as `config.json` spells it (one of the family's HF names).
    pub hf_architecture: String,
    pub num_layers: u32,
    pub hidden: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub tie_word_embeddings: bool,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    /// From `generation_config.json`, else `config.json`; never empty.
    pub eos_token_ids: SmallVec<[u32; 4]>,
    /// The expert layout of a mixture-of-experts model; `None` for a dense MLP.
    pub moe: Option<MoeConfig>,
    /// RMSNorm over the full Q and K projections before RoPE (`q_norm` / `k_norm`, OLMoE).
    pub qk_norm: bool,
    /// How the weights are stored and which dtypes weights, activations and KV use.
    pub weight_format: WeightFormatRef,
}

/// Per-layer mixture of experts: a router picks `experts_per_token` of `num_experts` SwiGLU
/// experts of width `expert_intermediate` per token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeConfig {
    pub num_experts: u32,
    pub experts_per_token: u32,
    /// `intermediate_size` of each expert.
    pub expert_intermediate: u32,
    /// Renormalise the selected top-k softmax weights to sum to 1 (`false` for OLMoE).
    pub norm_topk_prob: bool,
}

/// `generation_config.json`: EOS ids, BOS and the model's suggested sampling defaults.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct GenerationConfig {
    pub eos_token_ids: SmallVec<[u32; 4]>,
    pub bos_token_id: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
}

const DEFAULT_ROPE_THETA: f64 = 10_000.0;

impl ModelArchConfig {
    /// The description budget and planners consume. `weight_bytes` is the stored size (in the
    /// weight format) of every parameter the executor loads (no `lm_head` when tied).
    pub fn shape(&self) -> ModelShape {
        ModelShape {
            architecture: self.hf_architecture.clone(),
            num_layers: self.num_layers,
            hidden: self.hidden,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate: self.intermediate,
            vocab: self.vocab_size,
            num_experts: self.moe.map_or(0, |m| m.num_experts),
            experts_per_token: self.moe.map_or(0, |m| m.experts_per_token),
            tied_embeddings: self.tie_word_embeddings,
            weight_bytes: self.param_count() * self.weight_format.0.bytes_per_param(),
            max_position_embeddings: self.max_position_embeddings,
        }
    }

    /// Per-token K and V of every layer, in the weight format's KV dtype, in blocks of
    /// `block_tokens` tokens.
    pub fn kv_layout(&self, block_tokens: u32) -> KvLayout {
        KvLayout {
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.weight_format.0.kv_dtype(),
            block_tokens,
        }
    }

    /// The dtype half of the allowlist: every checkpoint tensor this architecture loads must be
    /// stored in the weight format (BF16: `unsupported tensor dtype = F8_E4M3 (<tensor>);
    /// supported: BF16` otherwise). Missing tensors are the loader's to report.
    pub fn check_supported_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelError> {
        for slot in &self.family.0.weight_slots(self) {
            if let Some(entry) = index.get(&slot.name) {
                self.weight_format.0.check_tensor(entry)?;
            }
        }
        Ok(())
    }

    /// Parameters of every slot the executor loads, so the count cannot drift from the loader.
    fn param_count(&self) -> u64 {
        self.family
            .0
            .weight_slots(self)
            .iter()
            .map(|s| s.shape.iter().map(|&d| d as u64).product::<u64>())
            .sum()
    }
}

/// `eos_token_id` is an int or a list of ints in HF configs.
#[derive(Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIds {
    fn into_ids(self) -> SmallVec<[u32; 4]> {
        match self {
            TokenIds::One(id) => SmallVec::from_slice(&[id]),
            TokenIds::Many(ids) => SmallVec::from_vec(ids),
        }
    }
}

/// The subset of `config.json` Turbine reads; other keys are ignored (HF configs carry many).
#[derive(Deserialize)]
struct RawConfig {
    num_hidden_layers: u32,
    hidden_size: u32,
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    intermediate_size: u32,
    rms_norm_eps: f32,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    vocab_size: u32,
    max_position_embeddings: u32,
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
}

#[derive(Deserialize)]
struct RawGenerationConfig {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
}

#[derive(Deserialize)]
struct RawLlama3Scaling {
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: u32,
}

pub(crate) fn unsupported(field: &str, value: impl Into<String>, supported: &str) -> ModelError {
    ModelError::Unsupported {
        field: field.to_string(),
        value: value.into(),
        supported: supported.to_string(),
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ModelError> {
    let io = |detail: String| ModelError::Io {
        path: path.to_path_buf(),
        detail,
    };
    let bytes = fs::read(path).map_err(|e| io(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| io(format!("invalid JSON: {e}")))
}

/// Parses `generation_config.json` in `dir`; a missing file is an `Io` error naming it.
pub fn load_generation_config(dir: &Path) -> Result<GenerationConfig, ModelError> {
    let raw: RawGenerationConfig = read_json(&dir.join("generation_config.json"))?;
    Ok(GenerationConfig {
        eos_token_ids: raw.eos_token_id.map(TokenIds::into_ids).unwrap_or_default(),
        bos_token_id: raw.bos_token_id,
        temperature: raw.temperature,
        top_p: raw.top_p,
        top_k: raw.top_k,
    })
}

/// Parses `config.json` (and `generation_config.json` for EOS ids) in `dir` and applies the
/// model-family and weight-format allowlist.
pub fn load_model_config(dir: &Path) -> Result<ModelArchConfig, ModelError> {
    let config_path = dir.join("config.json");
    let top: serde_json::Value = read_json(&config_path)?;
    let invalid = |detail: String| ModelError::Io {
        path: config_path.clone(),
        detail,
    };

    let (family, text) = resolve(&top)?;
    let hf_architecture = text["architectures"][0]
        .as_str()
        .expect("resolve matched a string")
        .to_string();
    let raw = RawConfig::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
    let weight_format = WeightFormatRef(detect(&top)?);

    let heads = raw.num_attention_heads;
    let num_kv_heads = raw.num_key_value_heads.unwrap_or(heads);
    if heads == 0 || num_kv_heads == 0 || !heads.is_multiple_of(num_kv_heads) {
        return Err(invalid(format!(
            "num_attention_heads {heads} must be a non-zero multiple of num_key_value_heads \
             {num_kv_heads}"
        )));
    }
    let head_dim = match raw.head_dim {
        Some(d) => d,
        None if raw.hidden_size.is_multiple_of(heads) => raw.hidden_size / heads,
        None => {
            return Err(invalid(format!(
                "no head_dim and hidden_size {} is not divisible by num_attention_heads {heads}",
                raw.hidden_size
            )));
        }
    };
    for (name, value) in [
        ("num_hidden_layers", raw.num_hidden_layers),
        ("hidden_size", raw.hidden_size),
        ("head_dim", head_dim),
        ("intermediate_size", raw.intermediate_size),
        ("vocab_size", raw.vocab_size),
        ("max_position_embeddings", raw.max_position_embeddings),
    ] {
        if value == 0 {
            return Err(invalid(format!("{name} must be non-zero")));
        }
    }
    // The inverse frequencies are theta^(-2i/d): a zero or negative base gives inf or NaN.
    let rope_theta = raw.rope_theta.unwrap_or(DEFAULT_ROPE_THETA);
    if !(rope_theta.is_finite() && rope_theta > 0.0) {
        return Err(invalid(format!(
            "rope_theta must be a positive number (got {rope_theta})"
        )));
    }

    // The family's own keys; a malformed value names this file.
    let family_cfg = family.parse_config(text).map_err(|e| match e {
        ModelError::Io { detail, .. } => invalid(detail),
        other => other,
    })?;

    let rope_scaling = parse_rope_scaling(raw.rope_scaling, &config_path)?;
    let eos_token_ids = load_eos(dir, raw.eos_token_id, &config_path)?;

    Ok(ModelArchConfig {
        family: FamilyRef(family),
        hf_architecture,
        num_layers: raw.num_hidden_layers,
        hidden: raw.hidden_size,
        num_attention_heads: heads,
        num_kv_heads,
        head_dim,
        intermediate: raw.intermediate_size,
        rms_norm_eps: raw.rms_norm_eps,
        rope_theta,
        rope_scaling,
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        vocab_size: raw.vocab_size,
        max_position_embeddings: raw.max_position_embeddings,
        eos_token_ids,
        moe: family_cfg.moe,
        qk_norm: family_cfg.qk_norm,
        weight_format,
    })
}

fn parse_rope_scaling(
    value: Option<serde_json::Value>,
    config_path: &Path,
) -> Result<Option<RopeScaling>, ModelError> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    // Older configs spell the key `type`.
    let rope_type = value
        .get("rope_type")
        .or_else(|| value.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("<missing>")
        .to_string();
    match rope_type.as_str() {
        "default" => Ok(None),
        "llama3" => {
            let raw: RawLlama3Scaling =
                serde_json::from_value(value).map_err(|e| ModelError::Io {
                    path: config_path.to_path_buf(),
                    detail: format!("rope_scaling: {e}"),
                })?;
            // The interpolated band divides by `high − low`: equal factors give NaN frequencies.
            let valid = raw.factor > 0.0
                && raw.low_freq_factor > 0.0
                && raw.high_freq_factor > raw.low_freq_factor
                && raw.original_max_position_embeddings > 0;
            if !valid {
                return Err(ModelError::Io {
                    path: config_path.to_path_buf(),
                    detail: format!(
                        "rope_scaling: llama3 requires factor > 0, 0 < low_freq_factor < \
                         high_freq_factor and original_max_position_embeddings > 0 (got factor \
                         {}, low_freq_factor {}, high_freq_factor {}, \
                         original_max_position_embeddings {})",
                        raw.factor,
                        raw.low_freq_factor,
                        raw.high_freq_factor,
                        raw.original_max_position_embeddings
                    ),
                });
            }
            Ok(Some(RopeScaling::Llama3 {
                factor: raw.factor,
                low_freq_factor: raw.low_freq_factor,
                high_freq_factor: raw.high_freq_factor,
                original_max_position_embeddings: raw.original_max_position_embeddings,
            }))
        }
        _ => Err(unsupported(
            "rope_scaling.rope_type",
            rope_type,
            "default, llama3",
        )),
    }
}

/// EOS ids from `generation_config.json` when it names any, else from `config.json`.
fn load_eos(
    dir: &Path,
    from_config: Option<TokenIds>,
    config_path: &Path,
) -> Result<SmallVec<[u32; 4]>, ModelError> {
    let generation_path: PathBuf = dir.join("generation_config.json");
    let from_generation = if generation_path.exists() {
        load_generation_config(dir)?.eos_token_ids
    } else {
        SmallVec::new()
    };
    let ids = if from_generation.is_empty() {
        from_config.map(TokenIds::into_ids).unwrap_or_default()
    } else {
        from_generation
    };
    if ids.is_empty() {
        return Err(ModelError::Io {
            path: config_path.to_path_buf(),
            detail: "no eos_token_id in generation_config.json or config.json".to_string(),
        });
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use turbine_core::types::DType;

    use super::*;
    use crate::ModelError;
    use crate::testing::TempDir;
    use crate::testing::tiny::write_tiny_llama;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-3.2-3b-instruct")
    }

    /// A per-process scratch directory under the system temp dir, emptied first.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "turbine-model-config-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The fixture config with `edit` applied, written to a scratch directory (no
    /// generation_config.json, so EOS comes from config.json).
    fn edited_config(name: &str, edit: impl FnOnce(&mut serde_json::Value)) -> PathBuf {
        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture_dir().join("config.json")).unwrap()).unwrap();
        edit(&mut v);
        let dir = scratch(name);
        fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
        dir
    }

    fn unsupported(err: ModelError) -> (String, String, String) {
        match err {
            ModelError::Unsupported {
                field,
                value,
                supported,
            } => (field, value, supported),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn parses_target_config() {
        let cfg = load_model_config(&fixture_dir()).unwrap();
        assert_eq!(cfg.family.0.name(), "llama");
        assert_eq!(cfg.hf_architecture, "LlamaForCausalLM");
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.hidden, 3072);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate, 8192);
        assert_eq!(cfg.vocab_size, 128_256);
        assert_eq!(cfg.max_position_embeddings, 131_072);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.rope_theta, 500_000.0);
        assert_eq!(
            cfg.rope_scaling,
            Some(RopeScaling::Llama3 {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            })
        );
        // generation_config.json's list wins over config.json's single 128009.
        assert_eq!(cfg.eos_token_ids.as_slice(), &[128_001, 128_008, 128_009]);

        let layout = cfg.kv_layout(16);
        assert_eq!(layout.dtype, DType::BF16);
        assert_eq!(layout.bytes_per_token(), 114_688);
        assert_eq!(layout.block_bytes(), 1_835_008);

        let shape = cfg.shape();
        assert_eq!(shape.architecture, "LlamaForCausalLM");
        assert_eq!(
            (shape.num_layers, shape.hidden, shape.vocab, shape.head_dim),
            (28, 3072, 128_256, 128)
        );
        assert_eq!((shape.num_experts, shape.experts_per_token), (0, 0));
        assert!(shape.tied_embeddings);
        // 3 212 749 824 BF16 parameters (tied: no separate lm_head).
        assert_eq!(shape.weight_bytes, 6_425_499_648);

        let generation = load_generation_config(&fixture_dir()).unwrap();
        assert_eq!(
            generation.eos_token_ids.as_slice(),
            &[128_001, 128_008, 128_009]
        );
        assert_eq!(generation.bos_token_id, Some(128_000));
        assert_eq!(generation.temperature, Some(0.6));
        assert_eq!(generation.top_p, Some(0.9));
        assert_eq!(generation.top_k, None);
    }

    fn olmoe_fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe-1b-7b-0125-instruct")
    }

    #[test]
    fn parses_olmoe_config() {
        let cfg = load_model_config(&olmoe_fixture_dir()).unwrap();
        assert_eq!(cfg.family.0.name(), "olmoe");
        assert_eq!(cfg.hf_architecture, "OlmoeForCausalLM");
        assert_eq!(cfg.num_layers, 16);
        assert_eq!(cfg.hidden, 2048);
        assert_eq!((cfg.num_attention_heads, cfg.num_kv_heads), (16, 16));
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(
            cfg.moe,
            Some(MoeConfig {
                num_experts: 64,
                experts_per_token: 8,
                expert_intermediate: 1024,
                norm_topk_prob: false,
            })
        );
        assert!(cfg.qk_norm);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.rope_scaling, None);
        assert_eq!(cfg.vocab_size, 50_304);
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.eos_token_ids.as_slice(), &[50_279]);

        // Block bytes as in the spec: 16 x 2 x 16 x 16 x 128 x 2.
        assert_eq!(cfg.kv_layout(16).block_bytes(), 2_097_152);

        let shape = cfg.shape();
        assert_eq!(shape.architecture, "OlmoeForCausalLM");
        assert_eq!((shape.num_experts, shape.experts_per_token), (64, 8));
        assert_eq!(shape.intermediate, 1024);
        assert!(!shape.tied_embeddings);
        // 6 919 161 856 BF16 parameters (untied, Q/K norm, 64 experts per layer).
        assert_eq!(shape.weight_bytes, 13_838_323_712);

        // The Llama checkpoint is dense with no Q/K norm.
        let llama = load_model_config(&fixture_dir()).unwrap();
        assert_eq!(llama.moe, None);
        assert!(!llama.qk_norm);
    }

    #[test]
    fn rejects_non_positive_rope_theta() {
        // theta <= 0 gives NaN (negative base) or infinite (zero base) inverse frequencies.
        for (name, theta) in [("rope-theta-zero", 0.0), ("rope-theta-negative", -10_000.0)] {
            let dir = edited_config(name, |v| v["rope_theta"] = serde_json::json!(theta));
            match load_model_config(&dir).unwrap_err() {
                ModelError::Io { detail, .. } => assert_eq!(
                    detail,
                    format!("rope_theta must be a positive number (got {theta})")
                ),
                other => panic!("expected Io, got {other:?}"),
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn rejects_degenerate_llama3_rope_scaling() {
        let detail = |dir: &Path| match load_model_config(dir).unwrap_err() {
            ModelError::Io { detail, .. } => detail,
            other => panic!("expected Io, got {other:?}"),
        };
        // Equal band factors would divide by zero in the interpolated band (a NaN frequency);
        // inverted ones, a zero factor or a zero original length make no valid band either.
        for (name, key, value) in [
            (
                "rope-equal-bands",
                "high_freq_factor",
                serde_json::json!(1.0),
            ),
            (
                "rope-inverted-bands",
                "high_freq_factor",
                serde_json::json!(0.5),
            ),
            ("rope-zero-low", "low_freq_factor", serde_json::json!(0.0)),
            ("rope-zero-factor", "factor", serde_json::json!(0.0)),
            (
                "rope-zero-original",
                "original_max_position_embeddings",
                serde_json::json!(0),
            ),
        ] {
            let dir = edited_config(name, |v| v["rope_scaling"][key] = value.clone());
            let detail = detail(&dir);
            assert!(
                detail.starts_with("rope_scaling: llama3 requires factor > 0, 0 < low_freq_factor < high_freq_factor and original_max_position_embeddings > 0"),
                "{name}: {detail}"
            );
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn rejects_malformed_olmoe() {
        let edited = |name: &str, edit: &dyn Fn(&mut serde_json::Value)| {
            let mut v: serde_json::Value =
                serde_json::from_slice(&fs::read(olmoe_fixture_dir().join("config.json")).unwrap())
                    .unwrap();
            edit(&mut v);
            let dir = scratch(name);
            fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
            dir
        };
        let detail = |dir: &Path| match load_model_config(dir).unwrap_err() {
            ModelError::Io { detail, .. } => detail,
            other => panic!("expected Io, got {other:?}"),
        };

        let dir = edited("olmoe-no-experts", &|v| {
            v.as_object_mut().unwrap().remove("num_experts");
        });
        assert_eq!(detail(&dir), "OlmoeForCausalLM requires num_experts");
        fs::remove_dir_all(dir).unwrap();

        let dir = edited("olmoe-topk", &|v| {
            v["num_experts_per_tok"] = serde_json::json!(65);
        });
        assert_eq!(
            detail(&dir),
            "num_experts_per_tok 65 must be between 1 and num_experts 64"
        );
        fs::remove_dir_all(dir).unwrap();

        let dir = edited("olmoe-clip", &|v| {
            v["clip_qkv"] = serde_json::json!(8.0);
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("clip_qkv", "8.0", "null")
        );
        fs::remove_dir_all(dir).unwrap();

        // norm_topk_prob defaults to false when absent (transformers' OlmoeConfig default).
        let dir = edited("olmoe-norm-default", &|v| {
            v.as_object_mut().unwrap().remove("norm_topk_prob");
        });
        assert!(!load_model_config(&dir).unwrap().moe.unwrap().norm_topk_prob);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn head_dim_and_eos_fallbacks() {
        // No explicit head_dim → hidden / heads; no generation_config.json → config.json EOS.
        let dir = edited_config("fallbacks", |v| {
            v.as_object_mut().unwrap().remove("head_dim");
        });
        let cfg = load_model_config(&dir).unwrap();
        assert_eq!(cfg.head_dim, 3072 / 24);
        assert_eq!(cfg.eos_token_ids.as_slice(), &[128_009]);
        fs::remove_dir_all(dir).unwrap();

        let missing = std::env::temp_dir().join("turbine-model-config-does-not-exist");
        match load_model_config(&missing).unwrap_err() {
            ModelError::Io { path, .. } => assert_eq!(path, missing.join("config.json")),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    /// Re-serializes the checkpoint with `name` stored as F8_E4M3 (same shape, one byte per
    /// element).
    fn rewrite_as_f8(dir: &Path, index: &SafetensorsIndex, name: &str) {
        use ::safetensors::Dtype;
        use ::safetensors::tensor::TensorView;
        let file = fs::read(dir.join("model.safetensors")).unwrap();
        let tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = index
            .entries()
            .map(|e| {
                let bytes = &file[e.range.start as usize..e.range.end as usize];
                if e.name == name {
                    let f8 = bytes.chunks(2).map(|b| b[1]).collect();
                    (e.name.clone(), Dtype::F8_E4M3, e.shape.clone(), f8)
                } else {
                    (e.name.clone(), e.dtype, e.shape.clone(), bytes.to_vec())
                }
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(n, dtype, shape, data)| {
                (
                    n.clone(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        ::safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    }

    #[test]
    fn rejects_unsupported() {
        let dir = edited_config("qwen", |v| {
            v["architectures"] = serde_json::json!(["Qwen3MoeForCausalLM"]);
        });
        let err = load_model_config(&dir).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported architectures = Qwen3MoeForCausalLM; supported: registered families: \
             llama (LlamaForCausalLM), olmoe (OlmoeForCausalLM)"
        );
        fs::remove_dir_all(dir).unwrap();

        let dir = edited_config("quantized", |v| {
            v["quantization_config"] = serde_json::json!({"quant_method": "fp8"});
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(field, "quantization_config");
        assert!(value.contains("fp8"), "{value}");
        assert_eq!(supported, "none");
        fs::remove_dir_all(dir).unwrap();

        let dir = edited_config("fp16", |v| {
            v["torch_dtype"] = serde_json::json!("float16");
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("torch_dtype", "float16", "bfloat16")
        );
        fs::remove_dir_all(dir).unwrap();

        // A tiny checkpoint whose safetensors holds an F8_E4M3 tensor.
        let tiny = TempDir::new("config-f8");
        let spec = write_tiny_llama(tiny.path(), 5);
        let index = SafetensorsIndex::open(tiny.path()).unwrap();
        spec.config.check_supported_weights(&index).unwrap();
        let f8 = "model.layers.0.mlp.down_proj.weight";
        rewrite_as_f8(tiny.path(), &index, f8);
        let index = SafetensorsIndex::open(tiny.path()).unwrap();
        assert_eq!(index.get(f8).unwrap().dtype, ::safetensors::Dtype::F8_E4M3);
        let err = spec.config.check_supported_weights(&index).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("unsupported tensor dtype = F8_E4M3 ({f8}); supported: BF16")
        );
        let (field, value, supported) = unsupported(err);
        assert_eq!(field, "tensor dtype");
        assert!(value.contains(f8), "{value}");
        assert_eq!(supported, "BF16");

        let dir = edited_config("yarn", |v| {
            v["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0});
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("rope_scaling.rope_type", "yarn", "default, llama3")
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
