//! The OLMoE family (`olmoe`, `OlmoeForCausalLM`; Phase 2 S-16): a mixture-of-experts decoder
//! with RMSNorm over the full Q and K projections. Its keys: `num_experts`,
//! `num_experts_per_tok`, `norm_topk_prob` (default false), `intermediate_size` as each
//! expert's width, and `clip_qkv`, which must be null. No default tool-call format (its chat
//! template renders no tools).

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::{FamilyConfig, ModelFamily, invalid};
use crate::ModelError;
use crate::config::{ModelArchConfig, MoeConfig, unsupported};
use crate::executor::{ExecutorLimits, ExecutorOptions, ModelExecutor, OlmoeExecutor};
use crate::loader::{
    LM_HEAD, LoadedWeights, StackPlace, WeightSlot, qkv_slots, stacked_experts_name,
};

/// `OlmoeForCausalLM` on the [`OlmoeExecutor`].
pub struct Olmoe;

/// The HF class name, as error messages name it.
const HF_NAME: &str = "OlmoeForCausalLM";

/// The OLMoE keys of `config.json`.
#[derive(Deserialize)]
struct OlmoeKeys {
    intermediate_size: u32,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    /// OLMoE clamps Q/K/V to ±`clip_qkv` when set; no executor implements that.
    #[serde(default)]
    clip_qkv: Option<serde_json::Value>,
}

impl Module for Olmoe {
    fn name(&self) -> &'static str {
        "olmoe"
    }
}

impl ModelFamily for Olmoe {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &[HF_NAME]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        let keys =
            OlmoeKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        if let Some(clip) = keys.clip_qkv.as_ref().filter(|c| !c.is_null()) {
            return Err(unsupported("clip_qkv", clip.to_string(), "null"));
        }
        let required = |name: &str, value: Option<u32>| {
            value.ok_or_else(|| invalid(format!("{HF_NAME} requires {name}")))
        };
        let num_experts = required("num_experts", keys.num_experts)?;
        let experts_per_token = required("num_experts_per_tok", keys.num_experts_per_tok)?;
        if experts_per_token == 0 || experts_per_token > num_experts {
            return Err(invalid(format!(
                "num_experts_per_tok {experts_per_token} must be between 1 and num_experts \
                 {num_experts}"
            )));
        }
        Ok(FamilyConfig {
            moe: Some(MoeConfig {
                num_experts,
                experts_per_token,
                expert_intermediate: keys.intermediate_size,
                norm_topk_prob: keys.norm_topk_prob.unwrap_or(false),
            }),
            qk_norm: true,
        })
    }

    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        olmoe_slots(cfg)
    }

    fn requirements(
        &self,
        cfg: &ModelArchConfig,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        OlmoeExecutor::requirements(cfg, block_tokens, opts)
    }

    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        OlmoeExecutor::workspace_bytes(
            cfg,
            limits.block_tokens,
            limits.max_batch_tokens,
            limits.max_seqs,
        )
    }

    fn default_tool_format(&self) -> Option<&'static str> {
        None
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
        Ok(Box::new(OlmoeExecutor::new(
            cfg,
            weights,
            registry,
            mem,
            limits.block_tokens,
            limits.max_batch_tokens,
            limits.max_seqs,
            opts,
        )?))
    }
}

/// Every parameter slot of an OLMoE model, in load order: embedding, per layer the attention
/// weights (Q/K/V as rows of the fused [`crate::loader::qkv_proj_name`] parameter) with the Q/K
/// norms (over the full `heads·head_dim` and `kv_heads·head_dim` projections), the router
/// `mlp.gate.weight` `[experts, hidden]`, every expert's SwiGLU weights, the two layer norms, then the final norm and `lm_head.weight` only when untied.
/// Each expert projection is entry `e` of the layer's stacked `[experts, rows, cols]` parameter
/// [`stacked_experts_name`] (the `moe_experts` layout). A dense config (`moe: None`) has no
/// experts and yields only the non-MLP slots.
pub fn olmoe_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
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
            slot(format!("{p}.self_attn.o_proj.weight"), vec![hidden, q]),
            slot(format!("{p}.self_attn.q_norm.weight"), vec![q]),
            slot(format!("{p}.self_attn.k_norm.weight"), vec![kv]),
            slot(format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            slot(format!("{p}.mlp.gate.weight"), vec![experts, hidden]),
        ]);
        let expert = |e: usize, proj: &str, shape: Vec<usize>| {
            let stack = StackPlace {
                name: stacked_experts_name(i, proj),
                shape: [vec![experts], shape.clone()].concat(),
                offset: e * shape.iter().product::<usize>(),
            };
            WeightSlot {
                name: format!("{p}.mlp.experts.{e}.{proj}.weight"),
                shape,
                stack: Some(stack),
            }
        };
        for e in 0..experts {
            slots.extend([
                expert(e, "gate_proj", vec![inter, hidden]),
                expert(e, "up_proj", vec![inter, hidden]),
                expert(e, "down_proj", vec![hidden, inter]),
            ]);
        }
    }
    slots.push(slot("model.norm.weight".into(), vec![hidden]));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}
