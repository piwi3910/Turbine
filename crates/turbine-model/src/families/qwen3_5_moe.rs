//! The Qwen3.6 MoE hybrid family (`qwen3_5_moe`, `Qwen3_5MoeForConditionalGeneration`; Phase 7
//! S-11): `Qwen/Qwen3.6-35B-A3B-FP8` and its siblings, text only. It is [`crate::families::
//! qwen3_5`]'s layer stack — Gated DeltaNet layers alternating with full-attention layers
//! carrying a per-head output gate, per-head Q/K norm, partial RoPE and `(1 + weight)` RMSNorm —
//! with a mixture-of-experts FFN in place of the dense MLP.
//!
//! The FFN: a BF16 router `mlp.gate` over `num_experts` SwiGLU experts of width
//! `moe_intermediate_size`, top-`num_experts_per_tok`, softmax then renormalise (always on: the
//! config class has no `norm_topk_prob`), plus a shared expert (`mlp.shared_expert.*`, width
//! `shared_expert_intermediate_size`) scaled by `sigmoid(x · shared_expert_gate)` and added to
//! every token's routed output ([`crate::executor::decoder::MOE_SHARED`]). The released
//! checkpoint stores the experts FP8-block (`weight_scale_inv`), the router, the shared-expert
//! gate and the norms BF16; the Gated DeltaNet and attention projections are FP8-block too.
//!
//! One device only for now: no tensor or expert parallelism (Task 34's TP2/EP2 legs follow with
//! the Gated DeltaNet value-head sharding).

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::olmoe::OLMOE_NAMES;
use super::{FamilyConfig, LayerKind, ModelFamily, experts, invalid};
use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::executor::decoder::{MOE_SHARED, QK_NORM_PER_HEAD};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::formats::qwen3_xml::QWEN3_XML;
use crate::loader::{
    LM_HEAD, LoadedWeights, WeightSlot, gated_qkv_slots_at, language_model_layer,
    stacked_experts_name,
};

/// `Qwen3_5MoeForConditionalGeneration` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct Qwen3_5Moe;

/// The HF class name, as error messages name it.
const HF_NAME: &str = "Qwen3_5MoeForConditionalGeneration";

/// The Qwen3.6 MoE keys beyond the shared hybrid ones ([`super::qwen3_5::qwen3_5_family_config`]).
#[derive(Deserialize)]
struct Qwen3_5MoeKeys {
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<u32>,
}

impl Module for Qwen3_5Moe {
    fn name(&self) -> &'static str {
        "qwen3_5_moe"
    }
}

impl ModelFamily for Qwen3_5Moe {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &[HF_NAME]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        let keys =
            Qwen3_5MoeKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        // The hybrid layer stack is the dense family's.
        let mut cfg = super::qwen3_5::qwen3_5_family_config(HF_NAME, text)?;
        // `norm_topk_prob` is absent from the config class: the softmax top-k is always
        // renormalised.
        let mut moe = experts(
            HF_NAME,
            ("num_experts", keys.num_experts),
            keys.num_experts_per_tok,
            ("moe_intermediate_size", keys.moe_intermediate_size),
            true,
        )?;
        let shared = keys.shared_expert_intermediate_size.ok_or_else(|| {
            invalid(format!(
                "{HF_NAME} requires shared_expert_intermediate_size"
            ))
        })?;
        if shared == 0 {
            return Err(invalid(
                "shared_expert_intermediate_size must be non-zero".to_string(),
            ));
        }
        moe.shared_inter = Some(shared);
        cfg.moe = Some(moe);
        Ok(cfg)
    }

    /// Every `n`-th layer full attention, the rest Gated DeltaNet (`full_attention_interval`).
    fn layer_kinds(&self, cfg: &ModelArchConfig) -> Vec<LayerKind> {
        super::qwen3_5::hybrid_layer_kinds(cfg)
    }

    /// The per-sequence Gated DeltaNet state: one F32 recurrent state and one activation-dtype
    /// conv state per linear layer.
    fn state_layout(&self, cfg: &ModelArchConfig) -> turbine_core::types::StateLayout {
        super::qwen3_5::gdn_state_layout(cfg)
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
        qwen3_5_moe_slots(cfg)
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
        crate::testing::tiny::write_tiny_qwen3_5_moe(dir, seed)
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

/// The Qwen3.6 MoE decoder: the dense hybrid's hooks with the shared-expert mixture of experts.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: QK_NORM_PER_HEAD,
        ffn: MOE_SHARED,
    }
}

/// Every parameter slot of a Qwen3.6 MoE text model, in load order: [`super::qwen3_5::
/// qwen3_5_slots`]'s attention half (the Gated DeltaNet projections, or the de-interleaved
/// `[q_all | gate_all | k | v]` stack with per-head Q/K norms of a full-attention layer) and,
/// in place of the dense MLP, the MoE FFN: the router `mlp.gate.weight` `[experts, hidden]`,
/// every expert's SwiGLU weights (entry `e` of the layer's stacked parameter
/// [`stacked_experts_name`]), and the shared expert's `mlp.shared_expert.{gate,up,down}_proj`
/// plus the `[1, hidden]` `mlp.shared_expert_gate`.
pub fn qwen3_5_moe_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q = heads * head_dim;
    let vocab = cfg.vocab_size as usize;
    let gdn = cfg
        .gdn
        .expect("a Qwen3.6 MoE configuration carries Gated DeltaNet dims");
    let (experts, inter, shared) = match cfg.moe {
        Some(m) => (
            m.num_experts as usize,
            m.expert_intermediate as usize,
            m.shared_inter
                .expect("a Qwen3.6 MoE configuration carries a shared expert") as usize,
        ),
        None => (0, 0, 0),
    };
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
        slots.push(slot(
            format!("{p}.{}.weight", OLMOE_NAMES.router),
            vec![experts, hidden],
        ));
        let expert = |e: usize, proj: &str, stack: &str, shape: Vec<usize>| {
            let place = crate::loader::StackPlace {
                name: stacked_experts_name(i, stack),
                shape: [vec![experts], shape.clone()].concat(),
                offset: e * shape.iter().product::<usize>(),
            };
            WeightSlot {
                float32_stored: false,
                name: format!("{p}.{}.{e}.{proj}.weight", OLMOE_NAMES.experts),
                shape,
                stack: Some(place),
                source: None,
            }
        };
        for e in 0..experts {
            slots.extend([
                expert(e, OLMOE_NAMES.gate, "gate_proj", vec![inter, hidden]),
                expert(e, OLMOE_NAMES.up, "up_proj", vec![inter, hidden]),
                expert(e, OLMOE_NAMES.down, "down_proj", vec![hidden, inter]),
            ]);
        }
        let sp = format!("{p}.mlp.shared_expert");
        slots.extend([
            slot(format!("{sp}.gate_proj.weight"), vec![shared, hidden]),
            slot(format!("{sp}.up_proj.weight"), vec![shared, hidden]),
            slot(format!("{sp}.down_proj.weight"), vec![hidden, shared]),
            slot(
                format!("{p}.mlp.shared_expert_gate.weight"),
                vec![1, hidden],
            ),
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

/// The checkpoint's own tensor layout of a Qwen3.6 MoE text model, with unique names and the
/// per-head interleaved `q_proj` `[2·q_dim, hidden]` (the form the tiny writer emits; the
/// loader's [`qwen3_5_moe_slots`] de-interleaves it). The MoE FFN is written per expert and as
/// the shared expert, as the released checkpoint stores it.
pub fn qwen3_5_moe_checkpoint_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q = heads * head_dim;
    let kv = kv_heads * head_dim;
    let vocab = cfg.vocab_size as usize;
    let gdn = cfg
        .gdn
        .expect("a Qwen3.6 MoE configuration carries Gated DeltaNet dims");
    let (experts, inter, shared) = match cfg.moe {
        Some(m) => (
            m.num_experts as usize,
            m.expert_intermediate as usize,
            m.shared_inter
                .expect("a Qwen3.6 MoE configuration carries a shared expert") as usize,
        ),
        None => (0, 0, 0),
    };
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
        slots.push(slot(
            format!("{p}.{}.weight", OLMOE_NAMES.router),
            vec![experts, hidden],
        ));
        for e in 0..experts {
            slots.extend([
                slot(
                    format!(
                        "{p}.{}.{e}.{}.weight",
                        OLMOE_NAMES.experts, OLMOE_NAMES.gate
                    ),
                    vec![inter, hidden],
                ),
                slot(
                    format!("{p}.{}.{e}.{}.weight", OLMOE_NAMES.experts, OLMOE_NAMES.up),
                    vec![inter, hidden],
                ),
                slot(
                    format!(
                        "{p}.{}.{e}.{}.weight",
                        OLMOE_NAMES.experts, OLMOE_NAMES.down
                    ),
                    vec![hidden, inter],
                ),
            ]);
        }
        let sp = format!("{p}.mlp.shared_expert");
        slots.extend([
            slot(format!("{sp}.gate_proj.weight"), vec![shared, hidden]),
            slot(format!("{sp}.up_proj.weight"), vec![shared, hidden]),
            slot(format!("{sp}.down_proj.weight"), vec![hidden, shared]),
            slot(
                format!("{p}.mlp.shared_expert_gate.weight"),
                vec![1, hidden],
            ),
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
