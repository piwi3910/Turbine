//! The Qwen3.5 hybrid family (`qwen3_5`, `Qwen3_5ForConditionalGeneration`; Phase 7 S-11):
//! `Qwen/Qwen3.5-9B`, text only. Its layers alternate three Gated DeltaNet (linear-attention)
//! layers with one full-attention layer (`full_attention_interval: 4`); the text model lives
//! under `model.language_model` in the multimodal wrapper and the vision tower (`model.visual.*`)
//! and the MTP head (`mtp.*`) are skipped at load (one `event="weights_skipped"` each).
//!
//! What differs from the other families:
//!
//! - The full-attention layers carry a per-head output gate (`attn_output_gate`): `q_proj` is
//!   per-head interleaved `[q_j | g_j]`, de-interleaved at load time by
//!   [`crate::loader::gated_qkv_slots_at`] into a `[q_all | gate_all | k | v]` stack, and the
//!   attention output is multiplied by `sigmoid(gate)` before the O projection
//!   ([`DecoderExecutor`]'s gated path, kernel ABI v2.19 `TURBINE_OP_SIGMOID_MUL`). RoPE is
//!   partial (`partial_rotary_factor: 0.25` → `rotary_dim = head_dim / 4`) and the RMSNorm
//!   weights use transformers' `output · (1 + weight)` form, folded into the weights at load.
//! - The linear-attention layers are Gated DeltaNet (`cfg.gdn`): `in_proj_qkv` → causal
//!   depthwise conv1d → split `q|k|v` → `in_proj_z`/`in_proj_a`/`in_proj_b` → the delta-rule
//!   recurrence (per-sequence F32 state, [`crate::config::GdnDims`]) → gated RMSNorm → `out_proj`.
//!   The executor holds a private per-sequence slot map for now (Task 32 replaces it with
//!   [`turbine_kv::state::StateSlotPool`]).
//!
//! Reasoning defaults to the `qwen_think` parser (` thinking…</think>`) and tool calls to the
//! `qwen3_xml` format (the checkpoint's own XML `<function=…>` form, S-18). One device only: no
//! tensor, pipeline or expert parallelism (Task 34 adds Qwen3.6's TP/EP).

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_core::types::{StateDtype, StateLayout};
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::{FamilyConfig, LayerKind, ModelFamily, invalid};
use crate::ModelError;
use crate::config::{GdnDims, ModelArchConfig, unsupported};
use crate::executor::decoder::{QK_NORM_PER_HEAD, SWIGLU};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::formats::qwen3_xml::QWEN3_XML;
use crate::loader::{
    LM_HEAD, LoadedWeights, WeightSlot, gated_qkv_slots_at, language_model_layer, row_concat,
};

/// `Qwen3_5ForConditionalGeneration` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct Qwen3_5;

/// The HF class name, as error messages name it.
const HF_NAME: &str = "Qwen3_5ForConditionalGeneration";

/// The ABI supports a causal conv window of 4 only.
const CONV_KERNEL: u32 = 4;

/// The Qwen3.5 keys of the text configuration.
#[derive(Deserialize)]
struct Qwen3_5Keys {
    num_hidden_layers: u32,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    full_attention_interval: Option<u32>,
    #[serde(default)]
    attn_output_gate: Option<bool>,
    #[serde(default)]
    linear_num_key_heads: Option<u32>,
    #[serde(default)]
    linear_num_value_heads: Option<u32>,
    #[serde(default)]
    linear_key_head_dim: Option<u32>,
    #[serde(default)]
    linear_value_head_dim: Option<u32>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<u32>,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
}

impl Module for Qwen3_5 {
    fn name(&self) -> &'static str {
        "qwen3_5"
    }
}

/// The shared Qwen3.5/3.6 hybrid keys of `config.json`: the Gated DeltaNet dimensions, the
/// attention output gate, the layer kinds and the partial RoPE. `hf` names the family in error
/// messages. The Qwen3.6 MoE variant ([`super::qwen3_5_moe`]) reuses this and adds its `moe`.
pub(crate) fn qwen3_5_family_config(
    hf: &str,
    text: &serde_json::Value,
) -> Result<FamilyConfig, ModelError> {
    let keys = Qwen3_5Keys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
    if keys.attn_output_gate == Some(false) {
        return Err(unsupported("attn_output_gate", "false", "true"));
    }
    // The full-attention interval is what `layer_kinds` reads; `layer_types`, when present,
    // must agree with it.
    let interval = keys.full_attention_interval.unwrap_or(1);
    if interval == 0 {
        return Err(invalid(
            "full_attention_interval must be non-zero".to_string(),
        ));
    }
    let kinds: Vec<LayerKind> = (0..keys.num_hidden_layers)
        .map(|i| {
            if (i as u64 + 1).is_multiple_of(u64::from(interval)) {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        })
        .collect();
    if let Some(types) = &keys.layer_types {
        if types.len() != keys.num_hidden_layers as usize {
            return Err(invalid(format!(
                "layer_types has {} entries for num_hidden_layers {}",
                types.len(),
                keys.num_hidden_layers
            )));
        }
        for (i, (t, kind)) in types.iter().zip(&kinds).enumerate() {
            let want = match kind {
                LayerKind::FullAttention => "full_attention",
                LayerKind::LinearAttention => "linear_attention",
                _ => unreachable!(),
            };
            if t != want {
                return Err(unsupported(
                    "layer_types",
                    format!("layer {i} {t}"),
                    &format!("{want} (full_attention_interval {interval})"),
                ));
            }
        }
    }
    let required = |name: &str, value: Option<u32>| {
        value.ok_or_else(|| invalid(format!("{hf} requires {name}")))
    };
    let conv_kernel = required("linear_conv_kernel_dim", keys.linear_conv_kernel_dim)?;
    if conv_kernel != CONV_KERNEL {
        return Err(unsupported(
            "linear_conv_kernel_dim",
            conv_kernel.to_string(),
            &CONV_KERNEL.to_string(),
        ));
    }
    let gdn = GdnDims {
        k_heads: required("linear_num_key_heads", keys.linear_num_key_heads)?,
        v_heads: required("linear_num_value_heads", keys.linear_num_value_heads)?,
        k_dim: required("linear_key_head_dim", keys.linear_key_head_dim)?,
        v_dim: required("linear_value_head_dim", keys.linear_value_head_dim)?,
        conv_kernel,
    };
    if gdn.k_heads == 0 || gdn.v_heads == 0 || gdn.k_dim == 0 || gdn.v_dim == 0 {
        return Err(invalid(
            "Gated DeltaNet dimensions must be non-zero".to_string(),
        ));
    }
    if !gdn.v_heads.is_multiple_of(gdn.k_heads) {
        return Err(invalid(format!(
            "linear_num_value_heads {} must be a multiple of linear_num_key_heads {}",
            gdn.v_heads, gdn.k_heads
        )));
    }
    Ok(FamilyConfig {
        moe: None,
        qk_norm: false,
        qk_norm_per_head: true,
        sliding_window: None,
        gdn: Some(gdn),
        attn_output_gate: true,
        partial_rotary_factor: partial_rotary_factor(&keys)?,
        unit_offset_norms: true,
        full_attention_interval: Some(interval),
    })
}

/// Every `n`-th layer full attention, the rest linear (`full_attention_interval`), shared by the
/// Qwen3.5 dense and Qwen3.6 MoE hybrids.
pub(crate) fn hybrid_layer_kinds(cfg: &ModelArchConfig) -> Vec<LayerKind> {
    let n = cfg.full_attention_interval.unwrap_or(1).max(1);
    (0..cfg.num_layers)
        .map(|i| {
            if (i as u64 + 1).is_multiple_of(u64::from(n)) {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        })
        .collect()
}

/// The per-sequence Gated DeltaNet state: one F32 recurrent state and one activation-dtype conv
/// state per linear layer, shared by the Qwen3.5 dense and Qwen3.6 MoE hybrids.
pub(crate) fn gdn_state_layout(cfg: &ModelArchConfig) -> StateLayout {
    let Some(gdn) = cfg.gdn else {
        return StateLayout::default();
    };
    StateLayout {
        linear_layers: cfg
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::LinearAttention)
            .count() as u32,
        recurrent_bytes_per_layer: gdn.recurrent_bytes(),
        conv_bytes_per_layer: gdn.conv_bytes(),
        dtype: Some(StateDtype::F32),
    }
}

/// The `partial_rotary_factor` of `keys`: the top-level key, else
/// `rope_parameters.partial_rotary_factor`.
fn partial_rotary_factor(keys: &Qwen3_5Keys) -> Result<Option<f32>, ModelError> {
    if let Some(f) = keys.partial_rotary_factor {
        return Ok(Some(f));
    }
    match keys
        .rope_parameters
        .as_ref()
        .and_then(|p| p.get("partial_rotary_factor"))
    {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(|f| Some(f as f32))
            .ok_or_else(|| invalid(format!("partial_rotary_factor must be a number (got {v})"))),
    }
}

impl ModelFamily for Qwen3_5 {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &[HF_NAME]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        qwen3_5_family_config(HF_NAME, text)
    }

    /// Every `n`-th layer full attention, the rest linear (`full_attention_interval`).
    fn layer_kinds(&self, cfg: &ModelArchConfig) -> Vec<LayerKind> {
        hybrid_layer_kinds(cfg)
    }

    /// The per-sequence Gated DeltaNet state: one F32 recurrent state and one activation-dtype
    /// conv state per linear layer.
    fn state_layout(&self, cfg: &ModelArchConfig) -> StateLayout {
        gdn_state_layout(cfg)
    }

    /// The checkpoint tensor prefixes the text path ignores: the vision tower and the MTP head.
    fn skipped_prefixes(&self) -> &'static [&'static str] {
        &["model.visual.", "mtp."]
    }

    /// The text model lives under the multimodal wrapper's `model.language_model`.
    fn text_prefix(&self) -> &'static str {
        "model.language_model"
    }

    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        qwen3_5_slots(cfg)
    }

    /// Tensor parallel is allowed (P7 Task 34, S-11): the full-attention layers shard by heads
    /// ([`crate::tp`]'s Q/K/V/O rules) and the Gated DeltaNet layers by value head — the key
    /// heads they read shrink with them, and `linear_attn.in_proj_qkv` / `conv1d` take a
    /// multi-part row slice.
    fn tp_decoder_spec(&self) -> Option<DecoderSpec> {
        Some(decoder_spec())
    }

    fn requirements(
        &self,
        cfg: &ModelArchConfig,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        DecoderExecutor::requirements(cfg, &decoder_spec(), block_tokens, opts)
    }

    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        DecoderExecutor::workspace_bytes(cfg, &decoder_spec(), limits)
    }

    fn default_tool_format(&self) -> Option<&'static str> {
        Some(QWEN3_XML)
    }

    fn default_reasoning_parser(&self) -> Option<&'static str> {
        Some("qwen_think")
    }

    fn write_tiny(&self, dir: &std::path::Path, seed: u64) -> crate::testing::tiny::TinySpec {
        crate::testing::tiny::write_tiny_qwen3_5(dir, seed)
    }

    fn build_executor(
        &self,
        cfg: &ModelArchConfig,
        weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        limits: ExecutorLimits,
        opts: ExecutorOptions,
    ) -> Result<Box<dyn ModelExecutor>, ModelError> {
        Ok(Box::new(DecoderExecutor::new(
            cfg,
            decoder_spec(),
            weights,
            registry,
            mem,
            limits,
            opts,
        )?))
    }
}

/// The Qwen3.5 decoder: per-head Q/K norm on the full-attention layers (the Gated DeltaNet
/// layers and the output gate are the executor's, keyed on `cfg.gdn` / `cfg.attn_output_gate`).
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: QK_NORM_PER_HEAD,
        ffn: SWIGLU,
    }
}

/// Every parameter slot of a Qwen3.5 text model, in load order. The text prefix is
/// `model.language_model`; the embedding and final norm sit under it, `lm_head.weight` at the
/// top level (untied). A full-attention layer's `q_proj` is de-interleaved into the
/// `[q_all | gate_all | k | v]` stack ([`gated_qkv_slots_at`]); a linear layer holds the
/// Gated DeltaNet projections.
pub fn qwen3_5_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q = heads * head_dim;
    let inter = cfg.intermediate as usize;
    let vocab = cfg.vocab_size as usize;
    let gdn = cfg
        .gdn
        .expect("a Qwen3.5 configuration carries Gated DeltaNet dims");
    // The Gated DeltaNet gated-norm weight, `A_log` and `dt_bias` are F32 in the released
    // checkpoint (transformers casts them to the model dtype); accept either and convert.
    // Every RMSNorm weight, `A_log` and `dt_bias` is F32 in the released checkpoint
    // (transformers casts them to the model dtype); accept either and convert on load.
    let slot = |name: String, shape: Vec<usize>| WeightSlot {
        float32_stored: crate::executor::decoder::is_norm_weight(&name)
            || name.ends_with("A_log")
            || name.ends_with("dt_bias"),
        name,
        shape,
        stack: None,
        source: None,
    };

    let mut slots = vec![slot(
        format!("{}.embed_tokens.weight", cfg.family.0.text_prefix()),
        vec![vocab, hidden],
    )];
    for i in 0..cfg.num_layers {
        let p = language_model_layer(i);
        slots.push(slot(format!("{p}.input_layernorm.weight"), vec![hidden]));
        match cfg.layer_kinds[i as usize] {
            LayerKind::LinearAttention => {
                let a = format!("{p}.linear_attn");
                slots.extend([
                    slot(
                        format!("{a}.in_proj_qkv.weight"),
                        vec![gdn.conv_dim(), hidden],
                    ),
                    slot(
                        format!("{a}.in_proj_z.weight"),
                        vec![gdn.value_dim(), hidden],
                    ),
                    slot(
                        format!("{a}.in_proj_a.weight"),
                        vec![gdn.v_heads as usize, hidden],
                    ),
                    slot(
                        format!("{a}.in_proj_b.weight"),
                        vec![gdn.v_heads as usize, hidden],
                    ),
                    slot(
                        format!("{a}.conv1d.weight"),
                        vec![gdn.conv_dim(), 1, gdn.conv_kernel as usize],
                    ),
                    slot(format!("{a}.norm.weight"), vec![gdn.v_dim as usize]),
                    slot(format!("{a}.A_log"), vec![gdn.v_heads as usize]),
                    slot(format!("{a}.dt_bias"), vec![gdn.v_heads as usize]),
                    slot(
                        format!("{a}.out_proj.weight"),
                        vec![hidden, gdn.value_dim()],
                    ),
                ]);
            }
            _ => {
                slots.extend(gated_qkv_slots_at(&p, heads, kv_heads, head_dim, hidden));
                slots.push(slot(
                    format!("{p}.self_attn.o_proj.weight"),
                    vec![hidden, q],
                ));
                slots.extend([
                    slot(format!("{p}.self_attn.q_norm.weight"), vec![head_dim]),
                    slot(format!("{p}.self_attn.k_norm.weight"), vec![head_dim]),
                ]);
            }
        }
        slots.push(slot(
            format!("{p}.post_attention_layernorm.weight"),
            vec![hidden],
        ));
        slots.extend(row_concat(
            format!("{p}.mlp.gate_up_proj.weight"),
            &[
                (format!("{p}.mlp.gate_proj.weight"), inter),
                (format!("{p}.mlp.up_proj.weight"), inter),
            ],
            hidden,
        ));
        slots.push(slot(
            format!("{p}.mlp.down_proj.weight"),
            vec![hidden, inter],
        ));
    }
    slots.push(slot(
        format!("{}.norm.weight", cfg.family.0.text_prefix()),
        vec![hidden],
    ));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}

/// The checkpoint's own tensor layout of a Qwen3.5 text model, with unique names and the
/// per-head interleaved `q_proj` `[2·q_dim, hidden]` (the form the tiny writer emits; the
/// loader's [`qwen3_5_slots`] de-interleaves it). Every tensor name is distinct.
pub fn qwen3_5_checkpoint_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q = heads * head_dim;
    let kv = kv_heads * head_dim;
    let inter = cfg.intermediate as usize;
    let vocab = cfg.vocab_size as usize;
    let gdn = cfg
        .gdn
        .expect("a Qwen3.5 configuration carries Gated DeltaNet dims");
    // The Gated DeltaNet gated-norm weight, `A_log` and `dt_bias` are F32 in the released
    // checkpoint (transformers casts them to the model dtype); accept either and convert.
    // Every RMSNorm weight, `A_log` and `dt_bias` is F32 in the released checkpoint
    // (transformers casts them to the model dtype); accept either and convert on load.
    let slot = |name: String, shape: Vec<usize>| WeightSlot {
        float32_stored: crate::executor::decoder::is_norm_weight(&name)
            || name.ends_with("A_log")
            || name.ends_with("dt_bias"),
        name,
        shape,
        stack: None,
        source: None,
    };

    let mut slots = vec![slot(
        format!("{}.embed_tokens.weight", cfg.family.0.text_prefix()),
        vec![vocab, hidden],
    )];
    for i in 0..cfg.num_layers {
        let p = language_model_layer(i);
        slots.push(slot(format!("{p}.input_layernorm.weight"), vec![hidden]));
        match cfg.layer_kinds[i as usize] {
            LayerKind::LinearAttention => {
                let a = format!("{p}.linear_attn");
                slots.extend([
                    slot(
                        format!("{a}.in_proj_qkv.weight"),
                        vec![gdn.conv_dim(), hidden],
                    ),
                    slot(
                        format!("{a}.in_proj_z.weight"),
                        vec![gdn.value_dim(), hidden],
                    ),
                    slot(
                        format!("{a}.in_proj_a.weight"),
                        vec![gdn.v_heads as usize, hidden],
                    ),
                    slot(
                        format!("{a}.in_proj_b.weight"),
                        vec![gdn.v_heads as usize, hidden],
                    ),
                    slot(
                        format!("{a}.conv1d.weight"),
                        vec![gdn.conv_dim(), 1, gdn.conv_kernel as usize],
                    ),
                    slot(format!("{a}.norm.weight"), vec![gdn.v_dim as usize]),
                    slot(format!("{a}.A_log"), vec![gdn.v_heads as usize]),
                    slot(format!("{a}.dt_bias"), vec![gdn.v_heads as usize]),
                    slot(
                        format!("{a}.out_proj.weight"),
                        vec![hidden, gdn.value_dim()],
                    ),
                ]);
            }
            _ => {
                slots.extend([
                    slot(format!("{p}.self_attn.q_proj.weight"), vec![2 * q, hidden]),
                    slot(format!("{p}.self_attn.k_proj.weight"), vec![kv, hidden]),
                    slot(format!("{p}.self_attn.v_proj.weight"), vec![kv, hidden]),
                    slot(format!("{p}.self_attn.o_proj.weight"), vec![hidden, q]),
                    slot(format!("{p}.self_attn.q_norm.weight"), vec![head_dim]),
                    slot(format!("{p}.self_attn.k_norm.weight"), vec![head_dim]),
                ]);
            }
        }
        slots.push(slot(
            format!("{p}.post_attention_layernorm.weight"),
            vec![hidden],
        ));
        slots.extend([
            slot(format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]),
            slot(format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]),
            slot(format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]),
        ]);
    }
    slots.push(slot(
        format!("{}.norm.weight", cfg.family.0.text_prefix()),
        vec![hidden],
    ));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}
