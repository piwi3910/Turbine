//! Model families (Phase 2m S-2, contract §24 `model_family`): everything that differs between
//! decoder architectures — the Hugging Face class names, the family's `config.json` keys, the
//! checkpoint's weight slots, the op requirements, the workspace, the default tool-call format
//! and the executor. One file per family and one entry in [`registry`]; [`resolve`] picks the
//! family of a `config.json` by `architectures[0]`.
//!
//! The keys every decoder shares (layers, widths, heads, RoPE, EOS) are parsed by
//! `crate::config`; [`ModelFamily::parse_config`] reads only the family's own keys.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbine_core::registry::{Module, Registry};
use turbine_core::types::StateLayout;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use crate::ModelError;
use crate::config::{ModelArchConfig, MoeConfig, unsupported};
use crate::executor::{DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor};
use crate::loader::{LoadedWeights, WeightSlot};
use crate::testing::tiny::TinySpec;

pub mod gpt_oss;
pub mod llama;
pub mod mistral;
pub mod mixtral;
pub mod olmoe;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_moe;

pub use gpt_oss::GptOss;
pub use llama::{Llama, llama_slots};
pub use mistral::Mistral;
pub use mixtral::{Mixtral, mixtral_slots};
pub use olmoe::{Olmoe, olmoe_slots};
pub use qwen3::Qwen3;
pub use qwen3_5::Qwen3_5;
pub use qwen3_moe::Qwen3Moe;

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
    /// The reasoning parser `model.reasoning_parser: auto` resolves to (Phase 7 S-10); `None`
    /// for a family without marked reasoning.
    fn default_reasoning_parser(&self) -> Option<&'static str> {
        None
    }
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
    /// The decoder hooks a tensor-parallel rank runs this family with (P5 S-6, [`crate::tp`]):
    /// heads, KV heads, intermediate columns and vocabulary split across ranks by the
    /// `turbine_distributed::tp` rules. `None` (the default): the family serves one device per
    /// model, and a tensor-parallel size above 1 is refused naming it.
    fn tp_decoder_spec(&self) -> Option<DecoderSpec> {
        None
    }
    /// A chat template (`(name, Jinja source)`) the family substitutes when the checkpoint's own
    /// template renders neither `tools` nor an assistant turn's `tool_calls` and the operator
    /// set no `model.chat_template` ([`crate::ChatTemplate::resolve_for`]); `None` (the default):
    /// the checkpoint's template is always used.
    fn chat_template_override(&self) -> Option<(&'static str, &'static str)> {
        None
    }
    /// Writes a tiny synthetic checkpoint of this family into `dir` (weights from a ChaCha8
    /// stream seeded by `seed`; the tiny tokenizer): the checkpoint the family's conformance
    /// suite ([`crate::conformance::families_suite`]) runs on the CPU provider. A test
    /// utility: panics on I/O errors.
    fn write_tiny(&self, dir: &Path, seed: u64) -> TinySpec;
    /// One kind per layer of `cfg` (P7 S-1). The default: every layer full attention.
    fn layer_kinds(&self, cfg: &ModelArchConfig) -> Vec<LayerKind> {
        vec![LayerKind::FullAttention; cfg.num_layers as usize]
    }
    /// Whether the family's checkpoints carry Q/K/V/O projection biases (`attention_bias:
    /// true` in `config.json`): the shared config parser refuses `attention_bias: true` for a
    /// family that does not (P7 S-8: gpt-oss does). The default: no.
    fn attention_bias(&self) -> bool {
        false
    }
    /// The per-sequence state `cfg` keeps beside its paged KV (P7 S-1). The default: none.
    fn state_layout(&self, cfg: &ModelArchConfig) -> StateLayout {
        let _ = cfg;
        StateLayout::default()
    }
    /// Checkpoint tensor prefixes this family's executor does not load but that the checkpoint
    /// legitimately ships (e.g. a vision tower or an MTP head the text path ignores): the loader
    /// counts the tensors and bytes under each prefix and logs one `event="weights_skipped"`
    /// each, instead of warning `unexpected_tensor` per tensor. The default: none.
    fn skipped_prefixes(&self) -> &'static [&'static str] {
        &[]
    }
    /// The checkpoint prefix of the text model's parameters (`model`, or Qwen3.5's
    /// `model.language_model` under the multimodal wrapper): the executor builds the embedding,
    /// per-layer and final-norm names from it. Slots stacks the family names itself
    /// ([`ModelFamily::weight_slots`]) are unaffected. The default: `model`.
    fn text_prefix(&self) -> &'static str {
        "model"
    }
}

/// How one decoder layer mixes tokens (P7 S-1): full causal attention, attention over the last
/// `window` positions, or a linear-attention (recurrent) layer with per-sequence state and no
/// paged KV.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum LayerKind {
    FullAttention,
    SlidingWindow { window: u32 },
    LinearAttention,
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
    /// RMSNorm over each Q and K head before RoPE (Qwen3, Qwen3-MoE).
    pub qk_norm_per_head: bool,
    /// The window of the family's sliding-attention layers ([`LayerKind::SlidingWindow`]):
    /// `sliding_window` of `config.json`, read by a family whose layers alternate it with full
    /// attention (gpt-oss); `None` for every other family.
    pub sliding_window: Option<u32>,
    /// The Gated DeltaNet dimensions of the family's linear-attention layers
    /// ([`LayerKind::LinearAttention`], P7 S-11); `None` for every attention-only family.
    pub gdn: Option<crate::config::GdnDims>,
    /// Every full-attention layer has a per-head output gate (Qwen3.5, P7 S-11).
    pub attn_output_gate: bool,
    /// A fraction of `head_dim` the RoPE rotates (`partial_rotary_factor`, Qwen3.5's 0.25);
    /// `None` rotates the whole head.
    pub partial_rotary_factor: Option<f32>,
    /// The family's RMSNorm weights follow transformers' `output · (1 + weight)` form (Qwen3.5).
    pub unit_offset_norms: bool,
    /// A hybrid family's full-attention interval (Qwen3.5's `full_attention_interval`): every
    /// `n`-th layer is [`LayerKind::FullAttention`] (1-indexed: layers `n − 1`, `2n − 1`, …),
    /// the rest [`LayerKind::LinearAttention`]. `None` for every attention-only family.
    pub full_attention_interval: Option<u32>,
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

static FAMILIES: Registry<dyn ModelFamily> = Registry::new(
    "model_family",
    &[
        &Llama, &Olmoe, &Qwen3, &Qwen3Moe, &Mistral, &Mixtral, &GptOss, &Qwen3_5,
    ],
);

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

/// The expert layout of a mixture-of-experts family's keys: `num_experts` (under the key
/// `experts.0`) of width `inter` (under `inter.0`), `per_token` of them per token, each value
/// required (`<hf> requires <key>`: [`invalid`]) and in range.
pub(crate) fn experts(
    hf: &str,
    experts: (&str, Option<u32>),
    per_token: Option<u32>,
    inter: (&str, Option<u32>),
    norm_topk_prob: bool,
) -> Result<MoeConfig, ModelError> {
    let required = |name: &str, value: Option<u32>| {
        value.ok_or_else(|| invalid(format!("{hf} requires {name}")))
    };
    let num_experts = required(experts.0, experts.1)?;
    let experts_per_token = required("num_experts_per_tok", per_token)?;
    if experts_per_token == 0 || experts_per_token > num_experts {
        return Err(invalid(format!(
            "num_experts_per_tok {experts_per_token} must be between 1 and {} {num_experts}",
            experts.0
        )));
    }
    let expert_intermediate = required(inter.0, inter.1)?;
    if expert_intermediate == 0 {
        return Err(invalid(format!("{} must be non-zero", inter.0)));
    }
    Ok(MoeConfig {
        num_experts,
        experts_per_token,
        expert_intermediate,
        norm_topk_prob,
    })
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
/// from: `architectures[0]` of the top level, else of a wrapper's `text_config`; the object is
/// `text_config` whenever the top level has one (P7 S-1), else the top level. No registered
/// family, or two claiming the name,
/// is [`ModelError::Unsupported`] naming the top-level `architectures` and the registered
/// families.
pub fn resolve(
    top: &serde_json::Value,
) -> Result<(&'static dyn ModelFamily, &serde_json::Value), ModelError> {
    let text_config = top.get("text_config").filter(|t| t.is_object());
    if let Some(name) = single_architecture(top)
        && let Some(family) = family_for(name)?
    {
        return Ok((family, text_config.unwrap_or(top)));
    }
    if let Some(text) = text_config
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

    /// An architecture no family registers.
    const UNREGISTERED_ARCH: &str = "ExampleUnregisteredForCausalLM";

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

        // Every registered family resolves by its HF name, top level or nested.
        for (name, hf) in [
            ("llama", "LlamaForCausalLM"),
            ("olmoe", "OlmoeForCausalLM"),
            ("qwen3", "Qwen3ForCausalLM"),
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
            ("mistral", "MistralForCausalLM"),
            ("mixtral", "MixtralForCausalLM"),
            ("gpt_oss", "GptOssForCausalLM"),
            ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ] {
            assert_eq!(
                resolve(&json!({"architectures": [hf]})).unwrap().0.name(),
                name
            );
            let nested = json!({"architectures": ["W"], "text_config": {"architectures": [hf]}});
            assert_eq!(resolve(&nested).unwrap().0.name(), name);
        }

        let err = resolve(&json!({"architectures": [UNREGISTERED_ARCH]}))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(
                "registered families: llama (LlamaForCausalLM), olmoe (OlmoeForCausalLM), qwen3 \
                 (Qwen3ForCausalLM), qwen3_moe (Qwen3MoeForCausalLM), mistral \
                 (MistralForCausalLM), mixtral (MixtralForCausalLM), gpt_oss (GptOssForCausalLM), \
                 qwen3_5 (Qwen3_5ForConditionalGeneration)"
            ),
            "{err}"
        );
        assert!(err.contains(UNREGISTERED_ARCH), "{err}");
        let err = resolve(&json!({})).unwrap_err().to_string();
        assert!(
            err.starts_with("unsupported architectures = <missing>;"),
            "{err}"
        );
    }

    /// The layer kinds and per-sequence state of a raw `config.json`, derived independently of
    /// the families: `layer_types` when present, else `full_attention_interval` (every n-th layer
    /// full, the rest linear), else one window (`sliding_window`, unless `use_sliding_window` is
    /// false) or full attention for every layer. Families without linear layers carry no state.
    fn expected_kinds_and_state(
        family: &str,
        config: &serde_json::Value,
    ) -> (Vec<LayerKind>, StateLayout) {
        let text = config
            .get("text_config")
            .filter(|t| t.is_object())
            .unwrap_or(config);
        let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
        let window = text
            .get("sliding_window")
            .and_then(|w| w.as_u64())
            .filter(|_| text.get("use_sliding_window") != Some(&json!(false)))
            .map(|w| w as u32);
        let kinds: Vec<LayerKind> = if let Some(types) = text["layer_types"].as_array() {
            types
                .iter()
                .map(|t| match t.as_str().unwrap() {
                    "full_attention" => LayerKind::FullAttention,
                    "sliding_attention" => LayerKind::SlidingWindow {
                        window: window.expect("sliding_attention needs sliding_window"),
                    },
                    "linear_attention" => LayerKind::LinearAttention,
                    other => panic!("{family}: unknown layer type {other}"),
                })
                .collect()
        } else if let Some(n) = text["full_attention_interval"].as_u64() {
            (0..layers)
                .map(|i| {
                    if (i as u64 + 1).is_multiple_of(n) {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect()
        } else if let Some(window) = window {
            vec![LayerKind::SlidingWindow { window }; layers]
        } else {
            vec![LayerKind::FullAttention; layers]
        };
        let linear = kinds
            .iter()
            .filter(|k| **k == LayerKind::LinearAttention)
            .count();
        let state = if linear == 0 {
            StateLayout::default()
        } else {
            // The Gated DeltaNet state of a hybrid family: the linear layers' dimensions come
            // from the raw config, the state bytes from them.
            let k_heads = text["linear_num_key_heads"].as_u64().unwrap() as u64;
            let v_heads = text["linear_num_value_heads"].as_u64().unwrap() as u64;
            let k_dim = text["linear_key_head_dim"].as_u64().unwrap();
            let v_dim = text["linear_value_head_dim"].as_u64().unwrap();
            let conv_kernel = text["linear_conv_kernel_dim"].as_u64().unwrap();
            let conv_dim = 2 * k_heads * k_dim + v_heads * v_dim;
            StateLayout {
                linear_layers: linear as u32,
                recurrent_bytes_per_layer: v_heads * v_dim * k_dim * 4,
                conv_bytes_per_layer: conv_dim * (conv_kernel - 1) * 2,
                dtype: Some(turbine_core::types::StateDtype::F32),
            }
        };
        (kinds, state)
    }

    #[test]
    fn layer_kinds_and_state_layout() {
        let tmp = TempDir::new("families-layer-kinds");
        for family in registry().iter() {
            let dir = tmp.path().join(family.name());
            let spec = family.write_tiny(&dir, 7);
            let cfg = crate::config::load_model_config(&dir).unwrap();
            assert_eq!(cfg, spec.config, "{}", family.name());
            let expect = expected_kinds_and_state(family.name(), &config_json(&dir));
            assert_eq!(cfg.layer_kinds, expect.0, "{}", family.name());
            assert_eq!(cfg.state, expect.1, "{}", family.name());
            assert_eq!(
                cfg.kv_layout(128).num_layers,
                cfg.attention_layers(),
                "{}",
                family.name()
            );
            let mut next = 0;
            for layer in 0..cfg.num_layers {
                let slot = cfg.kv_layer_index(layer);
                if cfg.layer_kinds[layer as usize] == LayerKind::LinearAttention {
                    assert_eq!(slot, None, "{} layer {layer}", family.name());
                } else {
                    assert_eq!(slot, Some(next), "{} layer {layer}", family.name());
                    next += 1;
                }
            }
        }

        // A `text_config`-wrapped copy of the tiny Qwen3 config (`architectures` stays at the
        // top level) parses to the same `ModelArchConfig`.
        let dir = tmp.path().join("qwen3");
        let plain = crate::config::load_model_config(&dir).unwrap();
        let mut text = config_json(&dir);
        let architectures = text
            .as_object_mut()
            .unwrap()
            .remove("architectures")
            .unwrap();
        let wrapped = json!({
            "architectures": architectures,
            "model_type": "qwen3_wrapper",
            "text_config": text,
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&wrapped).unwrap(),
        )
        .unwrap();
        assert_eq!(crate::config::load_model_config(&dir).unwrap(), plain);
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
