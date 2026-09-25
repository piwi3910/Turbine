//! Hugging Face `config.json` / `generation_config.json` parsing and the Phase 1 allowlist
//! (P1 S-3): architecture `LlamaForCausalLM`, BF16, no `quantization_config`. Anything else is
//! refused with the offending field named and the supported set listed.
//!
//! The weight-dtype part of the allowlist (`check_supported_weights`, every tensor BF16) needs the
//! safetensors index and lands with the weight loader.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use smallvec::SmallVec;
use turbine_core::types::{DType, KvLayout, ModelShape};

use crate::ModelError;

/// Model architectures this build can execute (`config.json` `architectures[0]`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Architecture {
    Llama,
}

impl Architecture {
    /// Every supported architecture, in the order error messages list them.
    pub const ALL: &'static [Architecture] = &[Architecture::Llama];

    /// The Hugging Face class name.
    pub fn as_str(self) -> &'static str {
        match self {
            Architecture::Llama => "LlamaForCausalLM",
        }
    }

    pub fn from_hf_name(name: &str) -> Option<Architecture> {
        Architecture::ALL
            .iter()
            .copied()
            .find(|a| a.as_str() == name)
    }

    fn supported_list() -> String {
        let names: Vec<&str> = Architecture::ALL.iter().map(|a| a.as_str()).collect();
        names.join(", ")
    }
}

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
    pub architecture: Architecture,
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

/// Weights and KV are BF16 in Phase 1 (`torch_dtype` must say so when present).
const SUPPORTED_TORCH_DTYPE: &str = "bfloat16";
const DEFAULT_ROPE_THETA: f64 = 10_000.0;

impl ModelArchConfig {
    /// The description budget and planners consume. `weight_bytes` is the BF16 size of every
    /// parameter the executor loads (no `lm_head` when tied).
    pub fn shape(&self) -> ModelShape {
        ModelShape {
            architecture: self.architecture.as_str().to_string(),
            num_layers: self.num_layers,
            hidden: self.hidden,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate: self.intermediate,
            vocab: self.vocab_size,
            num_experts: 0,
            experts_per_token: 0,
            tied_embeddings: self.tie_word_embeddings,
            weight_bytes: self.param_count() * DType::BF16.size_bytes() as u64,
            max_position_embeddings: self.max_position_embeddings,
        }
    }

    /// Per-token K and V of every layer, BF16, in blocks of `block_tokens` tokens.
    pub fn kv_layout(&self, block_tokens: u32) -> KvLayout {
        KvLayout {
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: DType::BF16,
            block_tokens,
        }
    }

    fn param_count(&self) -> u64 {
        let hidden = u64::from(self.hidden);
        let q = u64::from(self.num_attention_heads) * u64::from(self.head_dim);
        let kv = u64::from(self.num_kv_heads) * u64::from(self.head_dim);
        let embed = u64::from(self.vocab_size) * hidden;
        // q, k, v, o projections + gate, up, down + the two RMSNorm weights.
        let per_layer = hidden * q * 2
            + hidden * kv * 2
            + 3 * hidden * u64::from(self.intermediate)
            + 2 * hidden;
        let lm_head = if self.tie_word_embeddings { 0 } else { embed };
        embed + u64::from(self.num_layers) * per_layer + hidden + lm_head
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
    #[serde(default)]
    architectures: Option<Vec<String>>,
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
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
    #[serde(default)]
    torch_dtype: Option<String>,
    /// transformers ≥ 4.56 writes `dtype` instead of `torch_dtype`.
    #[serde(default)]
    dtype: Option<String>,
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

fn unsupported(field: &str, value: impl Into<String>, supported: &str) -> ModelError {
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
/// architecture, dtype and quantization allowlist.
pub fn load_model_config(dir: &Path) -> Result<ModelArchConfig, ModelError> {
    let config_path = dir.join("config.json");
    let raw: RawConfig = read_json(&config_path)?;
    let invalid = |detail: String| ModelError::Io {
        path: config_path.clone(),
        detail,
    };

    let architectures = raw.architectures.unwrap_or_default();
    let architecture = match architectures.as_slice() {
        [one] => Architecture::from_hf_name(one),
        _ => None,
    }
    .ok_or_else(|| {
        let value = if architectures.is_empty() {
            "<missing>".to_string()
        } else {
            architectures.join(", ")
        };
        unsupported("architectures", value, &Architecture::supported_list())
    })?;

    if let Some(q) = raw.quantization_config.filter(|q| !q.is_null()) {
        return Err(unsupported("quantization_config", q.to_string(), "none"));
    }
    for (field, value) in [("torch_dtype", &raw.torch_dtype), ("dtype", &raw.dtype)] {
        if let Some(value) = value.as_deref().filter(|v| *v != SUPPORTED_TORCH_DTYPE) {
            return Err(unsupported(field, value, SUPPORTED_TORCH_DTYPE));
        }
    }

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

    let rope_scaling = parse_rope_scaling(raw.rope_scaling, &config_path)?;
    let eos_token_ids = load_eos(dir, raw.eos_token_id, &config_path)?;

    Ok(ModelArchConfig {
        architecture,
        num_layers: raw.num_hidden_layers,
        hidden: raw.hidden_size,
        num_attention_heads: heads,
        num_kv_heads,
        head_dim,
        intermediate: raw.intermediate_size,
        rms_norm_eps: raw.rms_norm_eps,
        rope_theta: raw.rope_theta.unwrap_or(DEFAULT_ROPE_THETA),
        rope_scaling,
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        vocab_size: raw.vocab_size,
        max_position_embeddings: raw.max_position_embeddings,
        eos_token_ids,
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
        assert_eq!(cfg.architecture, Architecture::Llama);
        assert_eq!(cfg.architecture.as_str(), "LlamaForCausalLM");
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

    #[test]
    fn rejects_unsupported() {
        let dir = edited_config("qwen", |v| {
            v["architectures"] = serde_json::json!(["Qwen3MoeForCausalLM"]);
        });
        let err = load_model_config(&dir).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported architectures = Qwen3MoeForCausalLM; supported: LlamaForCausalLM"
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
