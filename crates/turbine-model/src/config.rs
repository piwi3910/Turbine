//! Hugging Face `config.json` / `generation_config.json` parsing and the allowlist (P1 S-3):
//! `architectures[0]` must name a registered model family (`crate::families::resolve`, Phase 2m
//! S-2), whose own keys it parses; the keys every decoder shares are parsed here (and
//! `attention_bias: true`, which no executor implements, is refused for every family). Anything
//! else is refused with the offending field named and the supported set listed.
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
use crate::executor::rope::yarn_attention_factor;
use crate::families::{FamilyRef, resolve};
use crate::safetensors::SafetensorsIndex;
use crate::weights::{WeightFormatRef, detect};

/// `config.json` `rope_scaling`, or the `model.rope_scaling` override that replaces it (absent
/// or `rope_type: default` → `None`).
#[derive(Clone, Copy, PartialEq, Debug)]
#[non_exhaustive]
pub enum RopeScaling {
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: u32,
    },
    /// Static YaRN (P6a S-15), every field resolved as transformers 4.57.1 resolves it:
    /// `original_max_position_embeddings` defaults to `max_position_embeddings`, the betas to
    /// 32 / 1, `truncate` to true, and `attention_factor` to
    /// [`crate::executor::rope::yarn_attention_factor`] of `factor`, `mscale` and
    /// `mscale_all_dim`. The attention factor multiplies cos and sin inside the rope op
    /// ([`ModelArchConfig::rope_attention_factor`], kernel ABI v2.10), not the rotary table and
    /// not the softmax scale (user decision 2026-09-29, superseding Q19's fold).
    Yarn {
        factor: f64,
        original_max_position_embeddings: u32,
        beta_fast: f64,
        beta_slow: f64,
        attention_factor: f64,
        truncate: bool,
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
    /// RMSNorm over each Q and K head (`q_norm` / `k_norm` of `[head_dim]`) before RoPE
    /// (Qwen3, Qwen3-MoE); never together with `qk_norm`.
    pub qk_norm_per_head: bool,
    /// How the weights are stored and which dtypes weights and activations use.
    pub weight_format: WeightFormatRef,
    /// How the paged KV pool stores K and V (`kv.dtype`, Phase 6a S-13): BF16 as parsed; the
    /// server sets FP8 and its per-layer scales before the executor is built.
    pub kv_cache: crate::kv_scales::KvCache,
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
            weight_bytes: self.weight_format.0.weight_bytes(self),
            max_position_embeddings: self.max_positions(),
        }
    }

    /// The longest sequence the model serves: `max_position_embeddings`, or with YaRN
    /// `factor × original_max_position_embeddings` when that is larger and the original
    /// context is below `max_position_embeddings` (with `original ≥ max_position_embeddings`
    /// there is nothing to extend; P6a S-15).
    pub fn max_positions(&self) -> u32 {
        match self.rope_scaling {
            Some(RopeScaling::Yarn {
                factor,
                original_max_position_embeddings: original,
                ..
            }) if original < self.max_position_embeddings => {
                let extended = (factor * f64::from(original))
                    .floor()
                    .min(f64::from(u32::MAX));
                self.max_position_embeddings.max(extended as u32)
            }
            _ => self.max_position_embeddings,
        }
    }

    /// The attention softmax scale: exactly `1 / sqrt(head_dim)` in FP32, with or without
    /// YaRN — YaRN's attention factor is applied to cos and sin instead
    /// ([`ModelArchConfig::rope_attention_factor`]; user decision 2026-09-29: folding its
    /// square into this scale rounded q·k before the factor and measured a p16 golden tail of
    /// 1.49 against 0.39 for transformers' placement).
    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// The factor the rope op multiplies cos and sin by before rounding (kernel ABI v2.10
    /// `turbine_rope_desc.attn_factor`, P6a S-15): YaRN's resolved `attention_factor` in FP32
    /// (transformers' `attention_scaling`), 1.0 for every other RoPE.
    pub fn rope_attention_factor(&self) -> f32 {
        match self.rope_scaling {
            Some(RopeScaling::Yarn {
                attention_factor, ..
            }) => attention_factor as f32,
            _ => 1.0,
        }
    }

    /// Canonical JSON of the resolved RoPE parameters (`theta` and every scaling field, keys
    /// sorted), for the prefix-cache namespace (P6a S-16): blocks cached under one RoPE
    /// configuration must never be reused under another.
    pub fn rope_identity(&self) -> String {
        let scaling = match self.rope_scaling {
            None => serde_json::Value::Null,
            Some(RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => serde_json::json!({
                "type": "llama3",
                "factor": factor,
                "low_freq_factor": low_freq_factor,
                "high_freq_factor": high_freq_factor,
                "original_max_position_embeddings": original_max_position_embeddings,
            }),
            Some(RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                attention_factor,
                truncate,
            }) => serde_json::json!({
                "type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_position_embeddings,
                "beta_fast": beta_fast,
                "beta_slow": beta_slow,
                "attention_factor": attention_factor,
                "truncate": truncate,
            }),
        };
        let value = serde_json::json!({
            "theta": self.rope_theta,
            "rotary_dim": self.head_dim,
            "scaling": scaling,
        });
        canonical_json(&value)
    }

    /// Per-token K and V of every layer in the [`ModelArchConfig::kv_cache`] dtype (BF16, or
    /// FP8 e4m3 under `kv.dtype: fp8_e4m3`), in blocks of `block_tokens` tokens.
    pub fn kv_layout(&self, block_tokens: u32) -> KvLayout {
        KvLayout {
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.kv_cache.dtype,
            block_tokens,
        }
    }

    /// The executor's activation dtype (BF16 for every weight format).
    pub fn activation_dtype(&self) -> turbine_core::types::DType {
        crate::weights::ACTIVATION_DTYPE
    }

    /// Every linear layer the executor loads: the family's 2-D slots except the token embedding
    /// (a gather, never a GEMM), by checkpoint name with `n × k` from the slot shape.
    pub fn linear_slots(&self) -> Vec<crate::weights::LinearSlot> {
        self.family
            .0
            .weight_slots(self)
            .into_iter()
            .filter(|s| s.shape.len() == 2 && !s.name.ends_with("embed_tokens.weight"))
            .map(|s| crate::weights::LinearSlot {
                name: s.name,
                n: s.shape[0] as u32,
                k: s.shape[1] as u32,
            })
            .collect()
    }

    /// The dtype half of the allowlist: every checkpoint tensor this architecture loads must be
    /// stored in the weight format (BF16: `unsupported tensor dtype = F8_E4M3 (<tensor>);
    /// supported: BF16` otherwise), a quantized layer's scales included
    /// ([`crate::weights::WeightFormat::slots`]). Missing tensors are the loader's to report.
    pub fn check_supported_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelError> {
        let format = self.weight_format.get();
        for base in &self.family.0.weight_slots(self) {
            for slot in format.slots(base) {
                if let Some(entry) = index.get(&slot.name) {
                    format.check_tensor(entry)?;
                }
            }
        }
        Ok(())
    }
}

/// Compact JSON with every object's keys sorted, whatever `serde_json`'s map order is.
fn canonical_json(value: &serde_json::Value) -> String {
    fn sorted(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = serde_json::Map::new();
                for k in keys {
                    out.insert(k.clone(), sorted(&map[k]));
                }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(sorted).collect())
            }
            other => other.clone(),
        }
    }
    sorted(value).to_string()
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
    /// Biased Q/K/V/O projections; no executor implements them (refused for every family).
    #[serde(default)]
    attention_bias: Option<bool>,
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
    load_model_config_with(dir, None)
}

/// [`load_model_config`] with the configuration key `model.rope_scaling` (a mapping with
/// Hugging Face's field names): when given, it replaces `config.json`'s `rope_scaling`
/// wholesale (P6a S-15, user decision 2026-09-28, Q19), and its refusals name
/// `model.rope_scaling`.
pub fn load_model_config_with(
    dir: &Path,
    rope_scaling: Option<&serde_json::Value>,
) -> Result<ModelArchConfig, ModelError> {
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

    if raw.attention_bias == Some(true) {
        return Err(unsupported("attention_bias", "true", "false"));
    }

    // The family's own keys; a malformed value names this file.
    let family_cfg = family.parse_config(text).map_err(|e| match e {
        ModelError::Io { detail, .. } => invalid(detail),
        other => other,
    })?;

    // A mixture of experts reports its experts' width as the model's intermediate size
    // (Qwen3-MoE's dense `intermediate_size` is unused; OLMoE's and Mixtral's are the experts').
    let intermediate = family_cfg
        .moe
        .map_or(raw.intermediate_size, |m| m.expert_intermediate);
    // Every weight size and offset (slots, stacks, buffers, the total in `shape`) is computed in
    // `usize` from these dimensions: refuse a config whose weights could not be addressed at 4
    // bytes per element (an upper bound on every model's parameters, so nothing below can wrap).
    let (hidden, inter) = (u128::from(raw.hidden_size), u128::from(intermediate));
    let (q_dim, kv_dim) = (
        u128::from(heads) * u128::from(head_dim),
        u128::from(num_kv_heads) * u128::from(head_dim),
    );
    let experts = family_cfg.moe.map_or(1, |m| u128::from(m.num_experts));
    // Each term fits u128 (at most 2^99 from u32 inputs); the layer product saturates.
    let per_layer = (2 * q_dim + 2 * kv_dim) * hidden + experts * (3 * inter + 1) * hidden;
    let elements = u128::from(raw.num_hidden_layers)
        .saturating_mul(per_layer)
        .saturating_add(2 * u128::from(raw.vocab_size) * hidden);
    if elements.saturating_mul(4) > usize::MAX as u128 {
        return Err(invalid(format!(
            "the dimensions give {elements} weight elements, too large to address"
        )));
    }
    let rope_scaling = match rope_scaling {
        Some(over) => parse_rope_scaling(
            Some(over),
            raw.max_position_embeddings,
            "model.rope_scaling",
            &config_path,
        )?,
        None => parse_rope_scaling(
            raw.rope_scaling.as_ref(),
            raw.max_position_embeddings,
            "rope_scaling",
            &config_path,
        )?,
    };
    let eos_token_ids = load_eos(dir, raw.eos_token_id, &config_path)?;

    Ok(ModelArchConfig {
        family: FamilyRef(family),
        hf_architecture,
        num_layers: raw.num_hidden_layers,
        hidden: raw.hidden_size,
        num_attention_heads: heads,
        num_kv_heads,
        head_dim,
        intermediate,
        rms_norm_eps: raw.rms_norm_eps,
        rope_theta,
        rope_scaling,
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        vocab_size: raw.vocab_size,
        max_position_embeddings: raw.max_position_embeddings,
        eos_token_ids,
        moe: family_cfg.moe,
        qk_norm: family_cfg.qk_norm,
        qk_norm_per_head: family_cfg.qk_norm_per_head,
        weight_format,
        kv_cache: crate::kv_scales::KvCache::bf16(),
    })
}

/// The RoPE scaling types Turbine implements; `dynamic`, `linear`, `longrope` and the rest are
/// refused (dynamic scaling would change cached keys mid-sequence; P6a S-15).
const SUPPORTED_ROPE_TYPES: &str = "default, llama3, yarn";

/// `rope_scaling` of `config.json` (`key` = `rope_scaling`) or the `model.rope_scaling`
/// override that replaces it (`key` = `model.rope_scaling`); refusals name `key`.
fn parse_rope_scaling(
    value: Option<&serde_json::Value>,
    max_position_embeddings: u32,
    key: &str,
    config_path: &Path,
) -> Result<Option<RopeScaling>, ModelError> {
    let Some(value) = value.filter(|v| !v.is_null()).cloned() else {
        return Ok(None);
    };
    let invalid = |detail: String| ModelError::Io {
        path: config_path.to_path_buf(),
        detail: format!("{key}: {detail}"),
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
        "yarn" => parse_yarn(&value, max_position_embeddings, key, &invalid).map(Some),
        "llama3" => {
            let raw: RawLlama3Scaling =
                serde_json::from_value(value).map_err(|e| invalid(e.to_string()))?;
            // The interpolated band divides by `high − low`: equal factors give NaN frequencies.
            let valid = raw.factor > 0.0
                && raw.low_freq_factor > 0.0
                && raw.high_freq_factor > raw.low_freq_factor
                && raw.original_max_position_embeddings > 0;
            if !valid {
                return Err(invalid(format!(
                    "llama3 requires factor > 0, 0 < low_freq_factor < high_freq_factor and \
                     original_max_position_embeddings > 0 (got factor {}, low_freq_factor {}, \
                     high_freq_factor {}, original_max_position_embeddings {})",
                    raw.factor,
                    raw.low_freq_factor,
                    raw.high_freq_factor,
                    raw.original_max_position_embeddings
                )));
            }
            Ok(Some(RopeScaling::Llama3 {
                factor: raw.factor,
                low_freq_factor: raw.low_freq_factor,
                high_freq_factor: raw.high_freq_factor,
                original_max_position_embeddings: raw.original_max_position_embeddings,
            }))
        }
        _ => Err(unsupported(
            &format!("{key}.rope_type"),
            rope_type,
            SUPPORTED_ROPE_TYPES,
        )),
    }
}

/// A `yarn` entry, resolved as transformers 4.57.1's `_compute_yarn_parameters` resolves it:
/// a missing, null or zero `original_max_position_embeddings`, `beta_fast` or `beta_slow`
/// takes its default (Python `or`: `max_position_embeddings`, 32, 1); `truncate` defaults to
/// true; a missing `attention_factor` is [`yarn_attention_factor`]. `dynamic: true` is refused;
/// other keys are ignored, as transformers ignores them.
fn parse_yarn(
    value: &serde_json::Value,
    max_position_embeddings: u32,
    key: &str,
    invalid: &dyn Fn(String) -> ModelError,
) -> Result<RopeScaling, ModelError> {
    let number = |name: &str| -> Result<Option<f64>, ModelError> {
        match value.get(name) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| invalid(format!("yarn {name} must be a number (got {v})"))),
        }
    };
    let flag = |name: &str, default: bool| -> Result<bool, ModelError> {
        match value.get(name) {
            None | Some(serde_json::Value::Null) => Ok(default),
            Some(v) => v
                .as_bool()
                .ok_or_else(|| invalid(format!("yarn {name} must be a boolean (got {v})"))),
        }
    };
    if flag("dynamic", false)? {
        return Err(unsupported(&format!("{key}.dynamic"), "true", "false"));
    }
    let factor = number("factor")?.ok_or_else(|| invalid("yarn requires factor".to_string()))?;
    let original_max_position_embeddings = match value.get("original_max_position_embeddings") {
        None | Some(serde_json::Value::Null) => max_position_embeddings,
        Some(v) => match v.as_u64() {
            Some(0) => max_position_embeddings,
            Some(n) => u32::try_from(n).map_err(|_| {
                invalid(format!(
                    "yarn original_max_position_embeddings {n} is too large"
                ))
            })?,
            None => {
                return Err(invalid(format!(
                    "yarn original_max_position_embeddings must be a positive integer (got {v})"
                )));
            }
        },
    };
    let or = |v: Option<f64>, default: f64| v.filter(|&x| x != 0.0).unwrap_or(default);
    let beta_fast = or(number("beta_fast")?, 32.0);
    let beta_slow = or(number("beta_slow")?, 1.0);
    let attention_factor = match number("attention_factor")? {
        Some(a) => a,
        None => yarn_attention_factor(factor, number("mscale")?, number("mscale_all_dim")?),
    };
    let truncate = flag("truncate", true)?;
    // transformers only warns on these; a factor below 1, a non-positive attention factor or
    // inverted betas give no meaningful extension, so Turbine refuses them.
    let valid = factor.is_finite()
        && factor >= 1.0
        && attention_factor.is_finite()
        && attention_factor > 0.0
        && beta_slow.is_finite()
        && beta_fast.is_finite()
        && beta_slow > 0.0
        && beta_slow <= beta_fast;
    if !valid {
        return Err(invalid(format!(
            "yarn requires factor >= 1, attention_factor > 0 and 0 < beta_slow <= beta_fast \
             (got factor {factor}, attention_factor {attention_factor}, beta_fast {beta_fast}, \
             beta_slow {beta_slow})"
        )));
    }
    Ok(RopeScaling::Yarn {
        factor,
        original_max_position_embeddings,
        beta_fast,
        beta_slow,
        attention_factor,
        truncate,
    })
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

    /// Dimensions whose weights cannot be addressed are refused at load, before any slot offset
    /// or buffer size is computed from them (Scout 99ababd5).
    #[test]
    fn rejects_dimensions_whose_weights_overflow() {
        let dir = edited_config("dims-overflow", |v| {
            v["vocab_size"] = serde_json::json!(u32::MAX);
            v["hidden_size"] = serde_json::json!(u32::MAX);
            v["head_dim"] = serde_json::json!(128);
        });
        match load_model_config(&dir).unwrap_err() {
            ModelError::Io { detail, .. } => {
                assert!(detail.contains("too large to address"), "{detail}")
            }
            other => panic!("expected Io, got {other:?}"),
        }
        fs::remove_dir_all(dir).unwrap();
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
        let dir = edited_config("gpt-oss", |v| {
            v["architectures"] = serde_json::json!(["GptOssForCausalLM"]);
        });
        let err = load_model_config(&dir).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported architectures = GptOssForCausalLM; supported: registered families: \
             llama (LlamaForCausalLM), olmoe (OlmoeForCausalLM), qwen3 (Qwen3ForCausalLM), \
             qwen3_moe (Qwen3MoeForCausalLM), mistral (MistralForCausalLM), mixtral \
             (MixtralForCausalLM)"
        );
        fs::remove_dir_all(dir).unwrap();

        // Biased projections: no executor implements them, for any family.
        let dir = edited_config("bias", |v| {
            v["attention_bias"] = serde_json::json!(true);
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("attention_bias", "true", "false")
        );
        fs::remove_dir_all(dir).unwrap();

        let dir = edited_config("quantized", |v| {
            // A packaging no registered weight format claims (Phase 6a serves fp8 and others).
            v["quantization_config"] = serde_json::json!({"quant_method": "gguf"});
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(field, "quantization_config");
        assert!(value.contains("gguf"), "{value}");
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

        // Dynamic scaling would change cached keys mid-sequence (P6a S-15).
        let dir = edited_config("dynamic", |v| {
            v["rope_scaling"] = serde_json::json!({"rope_type": "dynamic", "factor": 4.0});
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("rope_scaling.rope_type", "dynamic", "default, llama3, yarn")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// The committed transformers values (`tests/fixtures/yarn_params.json`, written by
    /// `scripts/golden/yarn_params.py` with transformers 4.57.1).
    fn yarn_fixture() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/yarn_params.json");
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    /// The target config with a fixture entry's RoPE keys (`rope_theta`, `head_dim`,
    /// `max_position_embeddings`, `rope_scaling`).
    fn yarn_config(name: &str, entry: &serde_json::Value) -> PathBuf {
        edited_config(&format!("yarn-{name}"), |v| {
            for key in [
                "rope_theta",
                "head_dim",
                "max_position_embeddings",
                "rope_scaling",
            ] {
                v[key] = entry[key].clone();
            }
        })
    }

    /// One FP32 unit in the last place of `x`.
    fn ulp(x: f32) -> f64 {
        f64::from(f32::from_bits(x.to_bits() + 1) - x)
    }

    /// P6a S-15: `inv_freq` and the attention factor of four YaRN configurations equal
    /// transformers' (`LlamaRotaryEmbedding`, i.e. `_compute_yarn_parameters`). transformers
    /// evaluates the blend in FP32 with a 1-ulp `powf`; Turbine evaluates it in FP64 and rounds
    /// once, so each frequency must lie within 2 FP32 ulps of transformers' (≈ 1.5e-7
    /// relative at worst). The attention factor is FP64 on both sides and must be equal to
    /// 1e-12 relative; the attention scale is `head_dim^-0.5 · attention_factor²`.
    #[test]
    fn yarn_parameters_match_transformers() {
        let fixture = yarn_fixture();
        assert_eq!(fixture["transformers"], "4.57.1");
        let configs = fixture["configs"].as_array().unwrap();
        let names: Vec<&str> = configs
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "llama-yarn16",
                "qwen3-yarn4",
                "gpt-oss-yarn32",
                "mscale-pair"
            ]
        );
        for c in configs {
            let name = c["name"].as_str().unwrap();
            let dir = yarn_config(name, &c["config"]);
            let cfg = load_model_config(&dir).unwrap();
            fs::remove_dir_all(dir).unwrap();
            let Some(RopeScaling::Yarn {
                attention_factor, ..
            }) = cfg.rope_scaling
            else {
                panic!("{name}: expected YaRN, got {:?}", cfg.rope_scaling);
            };
            let want: Vec<f64> = c["inv_freq"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let got = crate::executor::rope::inv_freq(
                cfg.rope_theta,
                cfg.head_dim,
                cfg.rope_scaling.as_ref(),
            );
            assert_eq!(got.len(), want.len(), "{name}");
            let mut max_rel = 0f64;
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let diff = (f64::from(g) - w).abs();
                max_rel = max_rel.max(diff / w);
                assert!(
                    diff <= 2.0 * ulp(w as f32),
                    "{name}: inv_freq[{i}] = {g:e}, transformers {w:e}"
                );
            }
            let want_af = c["attention_factor"].as_f64().unwrap();
            assert!(
                (attention_factor - want_af).abs() <= 1e-12 * want_af,
                "{name}: attention factor {attention_factor}, transformers {want_af}"
            );
            // The factor goes to cos/sin; the softmax scale stays head_dim^-0.5 (Task 28a).
            assert_eq!(
                cfg.attention_scale(),
                1.0 / (cfg.head_dim as f32).sqrt(),
                "{name}"
            );
            assert_eq!(
                cfg.rope_attention_factor(),
                want_af as f32,
                "{name}: the rope factor is transformers' attention_scaling in f32"
            );
            eprintln!(
                "{name}: inv_freq max relative error {max_rel:.3e}, attention factor \
                 {attention_factor}"
            );
        }
        // Without YaRN the scale is exactly the Phase 1 one and the rope factor is 1.
        let plain = load_model_config(&fixture_dir()).unwrap();
        assert_eq!(plain.attention_scale(), 1.0 / (128f32).sqrt());
        assert_eq!(plain.rope_attention_factor(), 1.0);
    }

    /// The configuration key `model.rope_scaling` replaces `config.json`'s entry wholesale
    /// (P6a S-15), here Llama-3.2's `llama3` scaling by the factor-16 YaRN of the proof.
    #[test]
    fn yarn_override_replaces_config_entry() {
        let fixture = yarn_fixture();
        let llama = &fixture["configs"][0];
        let over = &llama["config"]["rope_scaling"];
        let cfg = load_model_config_with(&fixture_dir(), Some(over)).unwrap();
        assert_eq!(
            cfg.rope_scaling,
            Some(RopeScaling::Yarn {
                factor: 16.0,
                original_max_position_embeddings: 8192,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: llama["attention_factor"].as_f64().unwrap(),
                truncate: true,
            })
        );
        // `default` turns scaling off; no override keeps config.json's.
        let off = serde_json::json!({"rope_type": "default"});
        assert_eq!(
            load_model_config_with(&fixture_dir(), Some(&off))
                .unwrap()
                .rope_scaling,
            None
        );
        assert!(matches!(
            load_model_config_with(&fixture_dir(), None)
                .unwrap()
                .rope_scaling,
            Some(RopeScaling::Llama3 { .. })
        ));
        // A refused override names the configuration key.
        let dynamic = serde_json::json!({"rope_type": "dynamic", "factor": 2.0});
        let (field, value, supported) =
            unsupported(load_model_config_with(&fixture_dir(), Some(&dynamic)).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            (
                "model.rope_scaling.rope_type",
                "dynamic",
                "default, llama3, yarn"
            )
        );
    }

    /// YaRN defaults (transformers' `or` rules), and what is refused: a `dynamic` flag, the
    /// other scaling types, and values that give no valid ramp or scale.
    #[test]
    fn yarn_defaults_and_refusals() {
        // `type` spelling; only the factor given: original = max_position_embeddings, betas
        // 32 / 1, truncate true, attention factor 0.1·ln(factor) + 1.
        let dir = edited_config("yarn-defaults", |v| {
            v["rope_scaling"] = serde_json::json!({"type": "yarn", "factor": 4.0});
        });
        let cfg = load_model_config(&dir).unwrap();
        fs::remove_dir_all(dir).unwrap();
        assert_eq!(
            cfg.rope_scaling,
            Some(RopeScaling::Yarn {
                factor: 4.0,
                original_max_position_embeddings: 131_072,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: 0.1 * 4f64.ln() + 1.0,
                truncate: true,
            })
        );
        // An explicit attention factor wins over mscale; `truncate: false` is kept.
        let dir = edited_config("yarn-explicit", |v| {
            v["rope_scaling"] = serde_json::json!({
                "rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096,
                "attention_factor": 1.5, "mscale": 2.0, "mscale_all_dim": 1.0,
                "truncate": false, "dynamic": false
            });
        });
        let cfg = load_model_config(&dir).unwrap();
        fs::remove_dir_all(dir).unwrap();
        assert!(matches!(
            cfg.rope_scaling,
            Some(RopeScaling::Yarn { attention_factor, truncate: false, .. })
                if attention_factor == 1.5
        ));

        for (name, scaling) in [
            (
                "linear",
                serde_json::json!({"rope_type": "linear", "factor": 2.0}),
            ),
            (
                "longrope",
                serde_json::json!({"rope_type": "longrope", "factor": 2.0}),
            ),
        ] {
            let dir = edited_config(name, |v| v["rope_scaling"] = scaling);
            let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
            assert_eq!(
                (field.as_str(), value.as_str(), supported.as_str()),
                ("rope_scaling.rope_type", name, "default, llama3, yarn")
            );
            fs::remove_dir_all(dir).unwrap();
        }
        let dir = edited_config("yarn-dynamic", |v| {
            v["rope_scaling"] =
                serde_json::json!({"rope_type": "yarn", "factor": 4.0, "dynamic": true});
        });
        let (field, value, supported) = unsupported(load_model_config(&dir).unwrap_err());
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("rope_scaling.dynamic", "true", "false")
        );
        fs::remove_dir_all(dir).unwrap();

        for (name, key, value) in [
            ("yarn-small-factor", "factor", serde_json::json!(0.5)),
            ("yarn-zero-af", "attention_factor", serde_json::json!(0.0)),
            ("yarn-inverted-betas", "beta_slow", serde_json::json!(64.0)),
        ] {
            let dir = edited_config(name, |v| {
                v["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0});
                v["rope_scaling"][key] = value;
            });
            match load_model_config(&dir).unwrap_err() {
                ModelError::Io { detail, .. } => assert!(
                    detail.starts_with("rope_scaling: yarn requires factor >= 1"),
                    "{name}: {detail}"
                ),
                other => panic!("{name}: expected Io, got {other:?}"),
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    /// `model.max_seq_len` may extend to `factor × original_max_position_embeddings` when
    /// `max_position_embeddings` is smaller; with nothing to extend (original ≥ max) or no YaRN
    /// the model's own `max_position_embeddings` stays the bound (P6a S-15).
    #[test]
    fn yarn_extends_max_positions() {
        let qwen3 = yarn_fixture()["configs"][1]["config"].clone();
        let dir = yarn_config("qwen3-ext", &qwen3);
        let cfg = load_model_config(&dir).unwrap();
        fs::remove_dir_all(dir).unwrap();
        assert_eq!(cfg.max_position_embeddings, 40_960);
        assert_eq!(cfg.max_positions(), 131_072);
        assert_eq!(cfg.shape().max_position_embeddings, 131_072);

        let dir = edited_config("yarn-no-ext", |v| {
            v["max_position_embeddings"] = serde_json::json!(4096);
            v["rope_scaling"] = serde_json::json!({
                "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192
            });
        });
        let cfg = load_model_config(&dir).unwrap();
        fs::remove_dir_all(dir).unwrap();
        assert_eq!(cfg.max_positions(), 4096);

        let plain = load_model_config(&fixture_dir()).unwrap();
        assert_eq!(plain.max_positions(), plain.max_position_embeddings);
    }

    /// The canonical RoPE description the prefix namespace will carry (Task 27): equal for
    /// equal parameters, different when any of them differs.
    #[test]
    fn rope_identity_names_every_parameter() {
        let plain = load_model_config(&fixture_dir()).unwrap();
        let fixture = yarn_fixture();
        let over = fixture["configs"][0]["config"]["rope_scaling"].clone();
        let yarn = load_model_config_with(&fixture_dir(), Some(&over)).unwrap();
        let mut other = over.clone();
        other["beta_fast"] = serde_json::json!(16);
        let yarn_other = load_model_config_with(&fixture_dir(), Some(&other)).unwrap();
        assert_eq!(plain.rope_identity(), plain.clone().rope_identity());
        assert_ne!(plain.rope_identity(), yarn.rope_identity());
        assert_ne!(yarn.rope_identity(), yarn_other.rope_identity());
        let v: serde_json::Value = serde_json::from_str(&yarn.rope_identity()).unwrap();
        assert_eq!(v["theta"], 500_000.0);
        assert_eq!(v["scaling"]["type"], "yarn");
        assert_eq!(v["scaling"]["factor"], 16.0);
        assert_eq!(v["scaling"]["truncate"], true);
        let v: serde_json::Value = serde_json::from_str(&plain.rope_identity()).unwrap();
        assert_eq!(v["scaling"]["type"], "llama3");
    }

    /// The `config.json` / `generation_config.json` of the Phase 8 checkpoints, copied verbatim
    /// from Hugging Face (`tests/fixtures/<slug>/`).
    fn family_fixture(slug: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(slug)
    }

    /// The four Phase 8 families parse from their real configs, with the parameter counts the
    /// model cards state (Qwen3-0.6B 0.6B, Qwen3-30B-A3B 30.5B, Mistral-7B 7.25B, Mixtral-8x7B
    /// 46.7B).
    #[test]
    fn parses_family_configs() {
        let qwen3 = load_model_config(&family_fixture("qwen3-0.6b")).unwrap();
        assert_eq!(qwen3.family.0.name(), "qwen3");
        assert_eq!(qwen3.hf_architecture, "Qwen3ForCausalLM");
        assert_eq!((qwen3.num_layers, qwen3.hidden), (28, 1024));
        assert_eq!((qwen3.num_attention_heads, qwen3.num_kv_heads), (16, 8));
        // head_dim is explicit: 128, not 1024 / 16.
        assert_eq!(qwen3.head_dim, 128);
        assert!(qwen3.qk_norm_per_head && !qwen3.qk_norm);
        assert_eq!(qwen3.moe, None);
        assert!(qwen3.tie_word_embeddings);
        assert_eq!(qwen3.rope_theta, 1_000_000.0);
        assert_eq!(qwen3.rope_scaling, None);
        assert_eq!(qwen3.rms_norm_eps, 1e-6);
        assert_eq!(qwen3.eos_token_ids.as_slice(), &[151_645, 151_643]);
        assert_eq!(qwen3.shape().weight_bytes, 2 * 596_049_920);

        let qwen3_moe = load_model_config(&family_fixture("qwen3-30b-a3b")).unwrap();
        assert_eq!(qwen3_moe.family.0.name(), "qwen3_moe");
        assert_eq!(
            qwen3_moe.moe,
            Some(MoeConfig {
                num_experts: 128,
                experts_per_token: 8,
                expert_intermediate: 768,
                norm_topk_prob: true,
            })
        );
        // The experts' width, not the unused dense intermediate_size (6144).
        assert_eq!(qwen3_moe.intermediate, 768);
        assert!(qwen3_moe.qk_norm_per_head && !qwen3_moe.qk_norm);
        assert!(!qwen3_moe.tie_word_embeddings);
        assert_eq!(
            (qwen3_moe.num_attention_heads, qwen3_moe.num_kv_heads),
            (32, 4)
        );
        assert_eq!(qwen3_moe.shape().weight_bytes, 2 * 30_532_122_624);

        let mistral = load_model_config(&family_fixture("mistral-7b-instruct-v0.3")).unwrap();
        assert_eq!(mistral.family.0.name(), "mistral");
        assert_eq!((mistral.head_dim, mistral.vocab_size), (128, 32_768));
        assert!(!mistral.qk_norm_per_head && !mistral.qk_norm && mistral.moe.is_none());
        assert_eq!(mistral.eos_token_ids.as_slice(), &[2]);
        assert_eq!(mistral.shape().weight_bytes, 2 * 7_248_023_552);

        let mixtral = load_model_config(&family_fixture("mixtral-8x7b-instruct-v0.1")).unwrap();
        assert_eq!(mixtral.family.0.name(), "mixtral");
        assert_eq!(
            mixtral.moe,
            Some(MoeConfig {
                num_experts: 8,
                experts_per_token: 2,
                expert_intermediate: 14_336,
                norm_topk_prob: true,
            })
        );
        assert!(!mixtral.qk_norm_per_head && !mixtral.qk_norm);
        assert_eq!(mixtral.shape().weight_bytes, 2 * 46_702_792_704);
        assert_eq!(
            mixtral.kv_layout(128).block_bytes(),
            128 * 2 * 8 * 128 * 2 * 32
        );
    }

    /// Features no executor implements are refused naming the key (open Phase 8 decisions,
    /// kept as the run-ahead has them): sliding windows (Mistral, Mixtral, Qwen3), partially
    /// dense Qwen3-MoE layers; missing expert keys are malformed.
    #[test]
    fn rejects_unsupported_family_features() {
        let edited = |slug: &str, name: &str, edit: &dyn Fn(&mut serde_json::Value)| {
            let fixture = family_fixture(slug);
            let mut v: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.join("config.json")).unwrap()).unwrap();
            edit(&mut v);
            let dir = scratch(name);
            fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
            dir
        };
        let refused = |dir: PathBuf| {
            let got = unsupported(load_model_config(&dir).unwrap_err());
            fs::remove_dir_all(dir).unwrap();
            got
        };

        // Mistral-7B-v0.1's 4096-token window is refused; a window covering every position is
        // no window at all.
        let (field, value, _) = refused(edited("mistral-7b-instruct-v0.3", "mistral-sw", &|v| {
            v["sliding_window"] = serde_json::json!(4096);
        }));
        assert_eq!((field.as_str(), value.as_str()), ("sliding_window", "4096"));
        let dir = edited("mistral-7b-instruct-v0.3", "mistral-sw-wide", &|v| {
            v["sliding_window"] = serde_json::json!(32768);
        });
        assert_eq!(load_model_config(&dir).unwrap().family.0.name(), "mistral");
        fs::remove_dir_all(dir).unwrap();
        let (field, _, _) = refused(edited("mixtral-8x7b-instruct-v0.1", "mixtral-sw", &|v| {
            v["sliding_window"] = serde_json::json!(4096);
        }));
        assert_eq!(field, "sliding_window");
        let (field, value, supported) = refused(edited("qwen3-0.6b", "qwen3-sw", &|v| {
            v["use_sliding_window"] = serde_json::json!(true);
        }));
        assert_eq!(
            (field.as_str(), value.as_str(), supported.as_str()),
            ("use_sliding_window", "true", "false")
        );
        let (field, value, _) = refused(edited("qwen3-30b-a3b", "qwen3-moe-dense", &|v| {
            v["mlp_only_layers"] = serde_json::json!([0, 47]);
        }));
        assert_eq!(
            (field.as_str(), value.as_str()),
            ("mlp_only_layers", "[0, 47]")
        );
        let (field, value, _) = refused(edited("qwen3-30b-a3b", "qwen3-moe-step", &|v| {
            v["decoder_sparse_step"] = serde_json::json!(2);
        }));
        assert_eq!(
            (field.as_str(), value.as_str()),
            ("decoder_sparse_step", "2")
        );

        let dir = edited("qwen3-30b-a3b", "qwen3-moe-no-inter", &|v| {
            v.as_object_mut().unwrap().remove("moe_intermediate_size");
        });
        match load_model_config(&dir).unwrap_err() {
            ModelError::Io { detail, .. } => {
                assert_eq!(detail, "Qwen3MoeForCausalLM requires moe_intermediate_size")
            }
            other => panic!("expected Io, got {other:?}"),
        }
        fs::remove_dir_all(dir).unwrap();
        let dir = edited("mixtral-8x7b-instruct-v0.1", "mixtral-topk", &|v| {
            v["num_experts_per_tok"] = serde_json::json!(9);
        });
        match load_model_config(&dir).unwrap_err() {
            ModelError::Io { detail, .. } => assert_eq!(
                detail,
                "num_experts_per_tok 9 must be between 1 and num_local_experts 8"
            ),
            other => panic!("expected Io, got {other:?}"),
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
