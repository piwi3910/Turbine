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

use super::{FamilyConfig, ModelFamily, experts, invalid};
use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::decoder::{MOE, QK_NORM_FULL};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::loader::{
    LM_HEAD, LoadedWeights, StackPlace, WeightSlot, qkv_slots, row_concat, stacked_experts_name,
};

/// `OlmoeForCausalLM` on the shared [`DecoderExecutor`] with [`decoder_spec`].
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
        let moe = experts(
            HF_NAME,
            ("num_experts", keys.num_experts),
            keys.num_experts_per_tok,
            ("intermediate_size", Some(keys.intermediate_size)),
            keys.norm_topk_prob.unwrap_or(false),
        )?;
        Ok(FamilyConfig {
            moe: Some(moe),
            qk_norm: true,
            qk_norm_per_head: false,
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
        DecoderExecutor::requirements(cfg, &decoder_spec(), block_tokens, opts)
    }

    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        DecoderExecutor::workspace_bytes(cfg, &decoder_spec(), limits)
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

/// The OLMoE decoder: RMSNorm over the full Q and K projections and the mixture of experts.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: QK_NORM_FULL,
        ffn: MOE,
    }
}

/// Checkpoint names of a mixture-of-experts layer's router and experts, relative to the layer.
pub(crate) struct MoeNames {
    /// The router; loaded as the MoE hook's `mlp.gate.weight` whatever the checkpoint calls it.
    pub router: &'static str,
    /// Expert `e`'s projections are `<experts>.<e>.<gate|up|down>.weight`.
    pub experts: &'static str,
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
}

/// OLMoE's checkpoint names (Qwen3-MoE uses them too).
pub(crate) const OLMOE_NAMES: MoeNames = MoeNames {
    router: "mlp.gate",
    experts: "mlp.experts",
    gate: "gate_proj",
    up: "up_proj",
    down: "down_proj",
};

/// Every parameter slot of an OLMoE model, in load order: embedding, per layer the attention
/// weights (Q/K/V as rows of the fused [`crate::loader::qkv_proj_name`] parameter) with the Q/K
/// norms (over the full `heads·head_dim` and `kv_heads·head_dim` projections), the router
/// `mlp.gate.weight` `[experts, hidden]`, every expert's SwiGLU weights, the two layer norms,
/// then the final norm and `lm_head.weight` only when untied. Each expert projection is entry
/// `e` of the layer's stacked `[experts, rows, cols]` parameter [`stacked_experts_name`] (the
/// `moe_experts` layout). A dense config (`moe: None`) has no experts and yields only the
/// non-MLP slots.
pub fn olmoe_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
    let kv = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    moe_slots(cfg, &OLMOE_NAMES, Some((q, kv)))
}

/// The slots of a mixture-of-experts decoder whose checkpoint uses `names`, in
/// [`olmoe_slots`]' order, with Q and K norm weights of `qk_norm` elements each (`None`: no Q/K
/// norm). A router not named `mlp.gate` lands as `model.layers.<i>.mlp.gate.weight` through a
/// one-part stack, and the experts land in the [`stacked_experts_name`] stacks of `gate_proj`,
/// `up_proj` and `down_proj` whatever the checkpoint calls them.
pub(crate) fn moe_slots(
    cfg: &ModelArchConfig,
    names: &MoeNames,
    qk_norm: Option<(usize, usize)>,
) -> Vec<WeightSlot> {
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
        slots.push(slot(
            format!("{p}.self_attn.o_proj.weight"),
            vec![hidden, q],
        ));
        if let Some((qn, kn)) = qk_norm {
            slots.extend([
                slot(format!("{p}.self_attn.q_norm.weight"), vec![qn]),
                slot(format!("{p}.self_attn.k_norm.weight"), vec![kn]),
            ]);
        }
        slots.push(slot(
            format!("{p}.post_attention_layernorm.weight"),
            vec![hidden],
        ));
        let router = format!("{p}.{}.weight", names.router);
        let hook_router = format!("{p}.mlp.gate.weight");
        if router == hook_router {
            slots.push(slot(router, vec![experts, hidden]));
        } else {
            slots.extend(row_concat(hook_router, &[(router, experts)], hidden));
        }
        let expert = |e: usize, proj: &str, stack: &str, shape: Vec<usize>| {
            let place = StackPlace {
                name: stacked_experts_name(i, stack),
                shape: [vec![experts], shape.clone()].concat(),
                offset: e * shape.iter().product::<usize>(),
            };
            WeightSlot {
                name: format!("{p}.{}.{e}.{proj}.weight", names.experts),
                shape,
                stack: Some(place),
            }
        };
        for e in 0..experts {
            slots.extend([
                expert(e, names.gate, "gate_proj", vec![inter, hidden]),
                expert(e, names.up, "up_proj", vec![inter, hidden]),
                expert(e, names.down, "down_proj", vec![hidden, inter]),
            ]);
        }
    }
    slots.push(slot("model.norm.weight".into(), vec![hidden]));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}
