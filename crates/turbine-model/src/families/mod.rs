//! Model families (Phase 2m S-2, contract §24 `model_family`): everything that differs between
//! decoder architectures — the Hugging Face class names, the family's `config.json` keys, the
//! checkpoint's weight slots, the op requirements, the workspace, the default tool-call format
//! and the executor. One file per family and one entry in [`registry`]; [`resolve`] picks the
//! family of a `config.json` by `architectures[0]`.
//!
//! The keys every decoder shares (layers, widths, heads, RoPE, EOS) are parsed by
//! `crate::config`; [`ModelFamily::parse_config`] reads only the family's own keys.

use std::path::PathBuf;
use std::sync::Arc;

use turbine_core::registry::{Module, Registry};
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use crate::ModelError;
use crate::config::{ModelArchConfig, MoeConfig, unsupported};
use crate::executor::{ExecutorLimits, ExecutorOptions, ModelExecutor};
use crate::loader::{LoadedWeights, WeightSlot};

pub mod llama;
pub mod olmoe;

pub use llama::{Llama, llama_slots};
pub use olmoe::{Olmoe, olmoe_slots};

/// One decoder family.
pub trait ModelFamily: Module {
    /// The `config.json` `architectures[0]` values this family serves.
    fn hf_architectures(&self) -> &'static [&'static str];
    /// Reads and checks the family's own keys of the object holding the text-model fields (the
    /// top level, or a wrapper's `text_config`). A malformed value is [`ModelError::Io`] naming
    /// `config.json` ([`invalid`]); a refused feature is [`ModelError::Unsupported`].
    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError>;
    /// Every parameter slot the executor loads: checkpoint names mapped onto its parameters.
    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot>;
    /// Every op config the forward runs over KV blocks of `block_tokens` tokens with `opts`.
    fn requirements(
        &self,
        cfg: &ModelArchConfig,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement>;
    /// Device bytes of the executor's buffers for batches within `limits`.
    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64;
    /// The tool-call format a null `model.tool_call_parser` resolves to (when the chat template
    /// renders tools).
    fn default_tool_format(&self) -> Option<&'static str>;
    /// The family's executor over `weights` for batches within `limits`; `registry` must have
    /// been built from [`ModelFamily::requirements`] of the same `cfg`, block size and `opts`.
    fn build_executor(
        &self,
        cfg: &ModelArchConfig,
        weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        limits: ExecutorLimits,
        opts: ExecutorOptions,
    ) -> Result<Box<dyn ModelExecutor>, ModelError>;
}

impl std::fmt::Debug for dyn ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// What a family's own keys add to `ModelArchConfig`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct FamilyConfig {
    /// The expert layout of a mixture-of-experts family; `None` for a dense MLP.
    pub moe: Option<MoeConfig>,
    /// RMSNorm over the full Q and K projections before RoPE (OLMoE).
    pub qk_norm: bool,
}

/// A registered family as a value of `ModelArchConfig`: equal by name, printed as its name.
#[derive(Clone, Copy)]
pub struct FamilyRef(pub &'static dyn ModelFamily);

impl PartialEq for FamilyRef {
    fn eq(&self, other: &FamilyRef) -> bool {
        self.0.name() == other.0.name()
    }
}

impl Eq for FamilyRef {}

impl std::fmt::Debug for FamilyRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.name())
    }
}

static FAMILIES: Registry<dyn ModelFamily> = Registry::new("model_family", &[&Llama, &Olmoe]);

/// Every model family, in the order error messages list them.
pub fn registry() -> &'static Registry<dyn ModelFamily> {
    &FAMILIES
}

/// A malformed `config.json` value (the path is filled in by `crate::config::load_model_config`).
pub(crate) fn invalid(detail: String) -> ModelError {
    ModelError::Io {
        path: PathBuf::from("config.json"),
        detail,
    }
}

/// `name (HfName), …` for every registered family.
fn registered_list() -> String {
    let entries: Vec<String> = registry()
        .iter()
        .map(|f| format!("{} ({})", f.name(), f.hf_architectures().join(", ")))
        .collect();
    format!("registered families: {}", entries.join(", "))
}

/// The one `architectures` entry of `obj`, when it has exactly one.
fn single_architecture(obj: &serde_json::Value) -> Option<&str> {
    match obj.get("architectures")?.as_array()?.as_slice() {
        [one] => one.as_str(),
        _ => None,
    }
}

/// The family registered for `name` (an `architectures[0]` value); two families claiming it is
/// refused naming both.
fn family_for(name: &str) -> Result<Option<&'static dyn ModelFamily>, ModelError> {
    let claims: Vec<&'static dyn ModelFamily> = registry()
        .iter()
        .filter(|f| f.hf_architectures().contains(&name))
        .collect();
    match claims.as_slice() {
        [] => Ok(None),
        [one] => Ok(Some(*one)),
        many => {
            let names: Vec<&str> = many.iter().map(|f| f.name()).collect();
            Err(unsupported(
                "architectures",
                format!("{name} (claimed by families {})", names.join(", ")),
                &registered_list(),
            ))
        }
    }
}

/// The family of a parsed top-level `config.json` and the object its text-model keys are read
/// from: `architectures[0]` of the top level (the object is the top level), else of a wrapper's
/// `text_config` (the object is `text_config`). No registered family, or two claiming the name,
/// is [`ModelError::Unsupported`] naming the top-level `architectures` and the registered
/// families.
pub fn resolve(
    top: &serde_json::Value,
) -> Result<(&'static dyn ModelFamily, &serde_json::Value), ModelError> {
    if let Some(name) = single_architecture(top)
        && let Some(family) = family_for(name)?
    {
        return Ok((family, top));
    }
    if let Some(text) = top.get("text_config").filter(|t| t.is_object())
        && let Some(name) = single_architecture(text)
        && let Some(family) = family_for(name)?
    {
        return Ok((family, text));
    }
    let names: Vec<&str> = top
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let value = if names.is_empty() {
        "<missing>".to_string()
    } else {
        names.join(", ")
    };
    Err(unsupported("architectures", value, &registered_list()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::loader::WeightSlot;
    use crate::testing::TempDir;
    use crate::testing::tiny::{write_tiny_llama, write_tiny_olmoe};

    fn config_json(dir: &std::path::Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap()
    }

    #[test]
    fn resolve_by_hf_name_and_refuse_unknown() {
        let tmp = TempDir::new("families-resolve");
        let llama = write_tiny_llama(&tmp.path().join("llama"), 1);
        let olmoe = write_tiny_olmoe(&tmp.path().join("olmoe"), 1);
        for (spec, name, hf) in [
            (&llama, "llama", "LlamaForCausalLM"),
            (&olmoe, "olmoe", "OlmoeForCausalLM"),
        ] {
            let top = config_json(&spec.dir);
            let (family, text) = resolve(&top).unwrap();
            assert_eq!(family.name(), name);
            assert_eq!(family.hf_architectures(), [hf]);
            assert!(std::ptr::eq(text, &top));
            assert_eq!(spec.config.family, FamilyRef(family));
            assert_eq!(spec.config.hf_architecture, hf);
        }

        // A wrapper whose language model is only named under `text_config`.
        let wrapped = json!({
            "architectures": ["LlavaForConditionalGeneration"],
            "text_config": {"architectures": ["LlamaForCausalLM"], "hidden_size": 64},
        });
        let (family, text) = resolve(&wrapped).unwrap();
        assert_eq!(family.name(), "llama");
        assert_eq!(text["hidden_size"], 64);
        let nested_only = json!({"text_config": {"architectures": ["LlamaForCausalLM"]}});
        assert_eq!(resolve(&nested_only).unwrap().0.name(), "llama");

        let err = resolve(&json!({"architectures": ["MistralForCausalLM"]}))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("registered families: llama (LlamaForCausalLM), olmoe (OlmoeForCausalLM)"),
            "{err}"
        );
        assert!(err.contains("MistralForCausalLM"), "{err}");
        let err = resolve(&json!({})).unwrap_err().to_string();
        assert!(
            err.starts_with("unsupported architectures = <missing>;"),
            "{err}"
        );
    }

    /// One line per slot: name, shape and, for a slice of a stacked parameter, the stack's name,
    /// shape and the slot's offset.
    fn lines(slots: &[WeightSlot]) -> Vec<String> {
        slots
            .iter()
            .map(|s| match &s.stack {
                None => format!("{} {:?}", s.name, s.shape),
                Some(p) => format!(
                    "{} {:?} -> {} {:?} @{}",
                    s.name, s.shape, p.name, p.shape, p.offset
                ),
            })
            .collect()
    }

    /// The slots main's `llama_slots` / `olmoe_slots` produced for the tiny checkpoints
    /// (seed-independent), recorded before the move into the families.
    #[test]
    fn slots_match_main() {
        let tmp = TempDir::new("families-slots");
        let llama = write_tiny_llama(&tmp.path().join("llama"), 1);
        let olmoe = write_tiny_olmoe(&tmp.path().join("olmoe"), 1);
        let got = |cfg: &ModelArchConfig| lines(&cfg.family.0.weight_slots(cfg));
        assert_eq!(got(&llama.config), MAIN_LLAMA_SLOTS);
        assert_eq!(got(&olmoe.config), MAIN_OLMOE_SLOTS);
    }

    const MAIN_LLAMA_SLOTS: &[&str] = &[
        "model.embed_tokens.weight [263, 64]",
        "model.layers.0.input_layernorm.weight [64]",
        "model.layers.0.self_attn.q_proj.weight [64, 64] -> model.layers.0.self_attn.qkv_proj.weight [128, 64] @0",
        "model.layers.0.self_attn.k_proj.weight [32, 64] -> model.layers.0.self_attn.qkv_proj.weight [128, 64] @4096",
        "model.layers.0.self_attn.v_proj.weight [32, 64] -> model.layers.0.self_attn.qkv_proj.weight [128, 64] @6144",
        "model.layers.0.self_attn.o_proj.weight [64, 64]",
        "model.layers.0.post_attention_layernorm.weight [64]",
        "model.layers.0.mlp.gate_proj.weight [128, 64] -> model.layers.0.mlp.gate_up_proj.weight [256, 64] @0",
        "model.layers.0.mlp.up_proj.weight [128, 64] -> model.layers.0.mlp.gate_up_proj.weight [256, 64] @8192",
        "model.layers.0.mlp.down_proj.weight [64, 128]",
        "model.layers.1.input_layernorm.weight [64]",
        "model.layers.1.self_attn.q_proj.weight [64, 64] -> model.layers.1.self_attn.qkv_proj.weight [128, 64] @0",
        "model.layers.1.self_attn.k_proj.weight [32, 64] -> model.layers.1.self_attn.qkv_proj.weight [128, 64] @4096",
        "model.layers.1.self_attn.v_proj.weight [32, 64] -> model.layers.1.self_attn.qkv_proj.weight [128, 64] @6144",
        "model.layers.1.self_attn.o_proj.weight [64, 64]",
        "model.layers.1.post_attention_layernorm.weight [64]",
        "model.layers.1.mlp.gate_proj.weight [128, 64] -> model.layers.1.mlp.gate_up_proj.weight [256, 64] @0",
        "model.layers.1.mlp.up_proj.weight [128, 64] -> model.layers.1.mlp.gate_up_proj.weight [256, 64] @8192",
        "model.layers.1.mlp.down_proj.weight [64, 128]",
        "model.norm.weight [64]",
    ];

    const MAIN_OLMOE_SLOTS: &[&str] = &[
        "model.embed_tokens.weight [263, 64]",
        "model.layers.0.input_layernorm.weight [64]",
        "model.layers.0.self_attn.q_proj.weight [64, 64] -> model.layers.0.self_attn.qkv_proj.weight [192, 64] @0",
        "model.layers.0.self_attn.k_proj.weight [64, 64] -> model.layers.0.self_attn.qkv_proj.weight [192, 64] @4096",
        "model.layers.0.self_attn.v_proj.weight [64, 64] -> model.layers.0.self_attn.qkv_proj.weight [192, 64] @8192",
        "model.layers.0.self_attn.o_proj.weight [64, 64]",
        "model.layers.0.self_attn.q_norm.weight [64]",
        "model.layers.0.self_attn.k_norm.weight [64]",
        "model.layers.0.post_attention_layernorm.weight [64]",
        "model.layers.0.mlp.gate.weight [8, 64]",
        "model.layers.0.mlp.experts.0.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @0",
        "model.layers.0.mlp.experts.0.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @0",
        "model.layers.0.mlp.experts.0.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @0",
        "model.layers.0.mlp.experts.1.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @2048",
        "model.layers.0.mlp.experts.1.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @2048",
        "model.layers.0.mlp.experts.1.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @2048",
        "model.layers.0.mlp.experts.2.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @4096",
        "model.layers.0.mlp.experts.2.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @4096",
        "model.layers.0.mlp.experts.2.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @4096",
        "model.layers.0.mlp.experts.3.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @6144",
        "model.layers.0.mlp.experts.3.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @6144",
        "model.layers.0.mlp.experts.3.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @6144",
        "model.layers.0.mlp.experts.4.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @8192",
        "model.layers.0.mlp.experts.4.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @8192",
        "model.layers.0.mlp.experts.4.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @8192",
        "model.layers.0.mlp.experts.5.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @10240",
        "model.layers.0.mlp.experts.5.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @10240",
        "model.layers.0.mlp.experts.5.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @10240",
        "model.layers.0.mlp.experts.6.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @12288",
        "model.layers.0.mlp.experts.6.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @12288",
        "model.layers.0.mlp.experts.6.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @12288",
        "model.layers.0.mlp.experts.7.gate_proj.weight [32, 64] -> model.layers.0.mlp.experts.gate_proj.weight [8, 32, 64] @14336",
        "model.layers.0.mlp.experts.7.up_proj.weight [32, 64] -> model.layers.0.mlp.experts.up_proj.weight [8, 32, 64] @14336",
        "model.layers.0.mlp.experts.7.down_proj.weight [64, 32] -> model.layers.0.mlp.experts.down_proj.weight [8, 64, 32] @14336",
        "model.layers.1.input_layernorm.weight [64]",
        "model.layers.1.self_attn.q_proj.weight [64, 64] -> model.layers.1.self_attn.qkv_proj.weight [192, 64] @0",
        "model.layers.1.self_attn.k_proj.weight [64, 64] -> model.layers.1.self_attn.qkv_proj.weight [192, 64] @4096",
        "model.layers.1.self_attn.v_proj.weight [64, 64] -> model.layers.1.self_attn.qkv_proj.weight [192, 64] @8192",
        "model.layers.1.self_attn.o_proj.weight [64, 64]",
        "model.layers.1.self_attn.q_norm.weight [64]",
        "model.layers.1.self_attn.k_norm.weight [64]",
        "model.layers.1.post_attention_layernorm.weight [64]",
        "model.layers.1.mlp.gate.weight [8, 64]",
        "model.layers.1.mlp.experts.0.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @0",
        "model.layers.1.mlp.experts.0.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @0",
        "model.layers.1.mlp.experts.0.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @0",
        "model.layers.1.mlp.experts.1.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @2048",
        "model.layers.1.mlp.experts.1.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @2048",
        "model.layers.1.mlp.experts.1.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @2048",
        "model.layers.1.mlp.experts.2.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @4096",
        "model.layers.1.mlp.experts.2.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @4096",
        "model.layers.1.mlp.experts.2.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @4096",
        "model.layers.1.mlp.experts.3.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @6144",
        "model.layers.1.mlp.experts.3.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @6144",
        "model.layers.1.mlp.experts.3.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @6144",
        "model.layers.1.mlp.experts.4.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @8192",
        "model.layers.1.mlp.experts.4.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @8192",
        "model.layers.1.mlp.experts.4.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @8192",
        "model.layers.1.mlp.experts.5.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @10240",
        "model.layers.1.mlp.experts.5.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @10240",
        "model.layers.1.mlp.experts.5.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @10240",
        "model.layers.1.mlp.experts.6.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @12288",
        "model.layers.1.mlp.experts.6.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @12288",
        "model.layers.1.mlp.experts.6.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @12288",
        "model.layers.1.mlp.experts.7.gate_proj.weight [32, 64] -> model.layers.1.mlp.experts.gate_proj.weight [8, 32, 64] @14336",
        "model.layers.1.mlp.experts.7.up_proj.weight [32, 64] -> model.layers.1.mlp.experts.up_proj.weight [8, 32, 64] @14336",
        "model.layers.1.mlp.experts.7.down_proj.weight [64, 32] -> model.layers.1.mlp.experts.down_proj.weight [8, 64, 32] @14336",
        "model.norm.weight [64]",
        "lm_head.weight [263, 64]",
    ];
}
