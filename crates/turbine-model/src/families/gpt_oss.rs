//! The gpt-oss family (`gpt_oss`, `GptOssForCausalLM`; Phase 7 S-8): `openai/gpt-oss-20b`.
//!
//! What differs from the other families:
//!
//! - Attention ([`ATTENTION_SINKS_WINDOW`]): head_dim 64, biased Q/K/V/O projections, one learned
//!   sink logit per query head, and layers alternating a sliding window of `sliding_window`
//!   positions (even layers; [`LayerKind::SlidingWindow`]) with full attention (odd layers).
//!   `config.json` must say so: `layer_types` strictly alternating `sliding_attention` /
//!   `full_attention` from layer 0 (or absent, transformers' default) and a positive
//!   `sliding_window`; any other pattern is refused. YaRN is the checkpoint's own
//!   `rope_scaling` (landed in Phase 6a).
//! - The block ([`MOE_GPT_OSS`]): a router with a bias (`mlp.router.{weight,bias}`) that takes the
//!   top `num_experts_per_tok` of its logits and a softmax over those, and MXFP4 experts stored
//!   fused per layer: `mlp.experts.gate_up_proj_{blocks,scales,bias}` (gate and up rows
//!   interleaved) and `down_proj_{blocks,scales,bias}`, with the clamped activation
//!   `(up + 1) · gate · σ(1.702 · gate)`, `gate ≤ 7`, `|up| ≤ 7` (`swiglu_limit`, which must be
//!   7.0). The experts are MXFP4 whatever the checkpoint's other layers are: its
//!   `quantization_config` (`quant_method: mxfp4`, [`crate::weights`] `openai_mxfp4`) leaves
//!   attention, the router, the embedding and the LM head in BF16.
//!
//! Tool calls and reasoning default to the Harmony formats (registered with Phase 7 Task 25).
//! One device only: no tensor, pipeline or expert parallelism.

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::{FamilyConfig, LayerKind, ModelFamily, experts, invalid};
use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::decoder::{ATTENTION_SINKS_WINDOW, MOE_GPT_OSS};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::loader::{LM_HEAD, LoadedWeights, WeightSlot, qkv_slots};

/// `GptOssForCausalLM` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct GptOss;

/// The HF class name, as error messages name it.
const HF_NAME: &str = "GptOssForCausalLM";

/// The `sliding_attention` / `full_attention` entries of `layer_types`.
const SLIDING: &str = "sliding_attention";
const FULL: &str = "full_attention";

/// The gpt-oss keys of `config.json`.
#[derive(Deserialize)]
struct GptOssKeys {
    num_hidden_layers: u32,
    #[serde(default)]
    num_local_experts: Option<u32>,
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    sliding_window: Option<u32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    swiglu_limit: Option<f32>,
    #[serde(default)]
    attention_bias: Option<bool>,
}

impl Module for GptOss {
    fn name(&self) -> &'static str {
        "gpt_oss"
    }
}

impl ModelFamily for GptOss {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &[HF_NAME]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        let keys =
            GptOssKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        if keys.attention_bias == Some(false) {
            return Err(unsupported("attention_bias", "false", "true"));
        }
        if let Some(limit) = keys.swiglu_limit.filter(|l| *l != 7.0) {
            return Err(unsupported("swiglu_limit", limit.to_string(), "7.0"));
        }
        let window = keys
            .sliding_window
            .filter(|w| *w > 0)
            .ok_or_else(|| invalid(format!("{HF_NAME} requires a positive sliding_window")))?;
        let alternating = |i: usize| if i.is_multiple_of(2) { SLIDING } else { FULL };
        if let Some(types) = &keys.layer_types {
            let fits = types.len() == keys.num_hidden_layers as usize
                && types.iter().enumerate().all(|(i, t)| t == alternating(i));
            if !fits {
                return Err(unsupported(
                    "layer_types",
                    format!("{types:?}"),
                    "sliding_attention and full_attention alternating, from layer 0 \
                     sliding_attention, one per layer",
                ));
            }
        }
        let moe = experts(
            HF_NAME,
            ("num_local_experts", keys.num_local_experts),
            keys.num_experts_per_tok,
            ("intermediate_size", keys.intermediate_size),
            false,
        )?;
        Ok(FamilyConfig {
            moe: Some(moe),
            sliding_window: Some(window),
            ..FamilyConfig::default()
        })
    }

    fn attention_bias(&self) -> bool {
        true
    }

    /// Even layers slide over `sliding_window` positions, odd layers see every position.
    fn layer_kinds(&self, cfg: &ModelArchConfig) -> Vec<LayerKind> {
        let window = cfg.sliding_window.unwrap_or(u32::MAX);
        (0..cfg.num_layers)
            .map(|i| {
                if i.is_multiple_of(2) {
                    LayerKind::SlidingWindow { window }
                } else {
                    LayerKind::FullAttention
                }
            })
            .collect()
    }

    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        gpt_oss_slots(cfg)
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

    /// The Harmony tool-call format (registered with Task 25).
    fn default_tool_format(&self) -> Option<&'static str> {
        Some("harmony")
    }

    /// The Harmony channel parser (registered with Task 25).
    fn default_reasoning_parser(&self) -> Option<&'static str> {
        Some("harmony")
    }

    /// The tiny gpt-oss checkpoint (`testing::tiny`, window 128).
    fn write_tiny(&self, dir: &std::path::Path, seed: u64) -> crate::testing::tiny::TinySpec {
        crate::testing::tiny::write_tiny_gpt_oss(dir, seed, 128)
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

/// The gpt-oss decoder: biased attention with sinks and the MXFP4 mixture of experts.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: ATTENTION_SINKS_WINDOW,
        ffn: MOE_GPT_OSS,
    }
}

/// Every parameter slot of a gpt-oss model, in load order: embedding, per layer the input norm,
/// the attention weights (Q/K/V as rows of the fused [`crate::loader::qkv_proj_name`]
/// parameter), their biases, the O projection and its bias, the sinks `[heads]`, the
/// post-attention norm, the router `[experts, hidden]` with its bias, and the fused experts as
/// the checkpoint stores them (`mlp.experts.<proj>_blocks` `[experts, n, k/32, 16]`, `_scales`
/// `[experts, n, k/32]`, both U8 under the `openai_mxfp4` format, and `_bias` `[experts, n]`
/// for `gate_up_proj` with `n = 2 · inter` and `down_proj` with `n = hidden`), then the final
/// norm and `lm_head.weight` when untied.
pub fn gpt_oss_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
    let kv = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let (experts, inter) = cfg.moe.map_or((0, 0), |m| {
        (m.num_experts as usize, m.expert_intermediate as usize)
    });
    let slot = |name: String, shape: Vec<usize>| WeightSlot {
        name,
        shape,
        stack: None,
        source: None,
    };

    let mut slots = vec![slot(
        "model.embed_tokens.weight".into(),
        vec![vocab, hidden],
    )];
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        slots.push(slot(format!("{p}.input_layernorm.weight"), vec![hidden]));
        slots.extend(qkv_slots(i, q, kv, hidden));
        slots.extend([
            slot(format!("{p}.self_attn.q_proj.bias"), vec![q]),
            slot(format!("{p}.self_attn.k_proj.bias"), vec![kv]),
            slot(format!("{p}.self_attn.v_proj.bias"), vec![kv]),
            slot(format!("{p}.self_attn.o_proj.weight"), vec![hidden, q]),
            slot(format!("{p}.self_attn.o_proj.bias"), vec![hidden]),
            slot(
                format!("{p}.self_attn.sinks"),
                vec![cfg.num_attention_heads as usize],
            ),
            slot(format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            slot(format!("{p}.mlp.router.weight"), vec![experts, hidden]),
            slot(format!("{p}.mlp.router.bias"), vec![experts]),
        ]);
        for (proj, n, k) in [
            ("gate_up_proj", 2 * inter, hidden),
            ("down_proj", hidden, inter),
        ] {
            slots.extend([
                slot(
                    format!("{p}.mlp.experts.{proj}_blocks"),
                    vec![experts, n, k / 32, 16],
                ),
                slot(
                    format!("{p}.mlp.experts.{proj}_scales"),
                    vec![experts, n, k / 32],
                ),
                slot(format!("{p}.mlp.experts.{proj}_bias"), vec![experts, n]),
            ]);
        }
    }
    slots.push(slot("model.norm.weight".into(), vec![hidden]));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}
