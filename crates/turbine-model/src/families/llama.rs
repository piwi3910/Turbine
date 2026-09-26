//! The Llama family (`llama`, `LlamaForCausalLM`; Phase 1): a dense SwiGLU decoder with
//! grouped-query attention and no family keys of its own. Tool calls default to `llama3_json`.

use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::{FamilyConfig, ModelFamily};
use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::executor::{ExecutorLimits, ExecutorOptions, LlamaExecutor, ModelExecutor};
use crate::loader::{LM_HEAD, LoadedWeights, WeightSlot, gate_up_proj_name, qkv_slots, row_concat};
use crate::tools::LLAMA3_JSON;

/// `LlamaForCausalLM` on the [`LlamaExecutor`].
pub struct Llama;

impl Module for Llama {
    fn name(&self) -> &'static str {
        "llama"
    }
}

impl ModelFamily for Llama {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &["LlamaForCausalLM"]
    }

    fn parse_config(&self, _text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        Ok(FamilyConfig::default())
    }

    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        llama_slots(cfg)
    }

    fn requirements(
        &self,
        cfg: &ModelArchConfig,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        LlamaExecutor::requirements(cfg, block_tokens, opts)
    }

    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        LlamaExecutor::workspace_bytes(
            cfg,
            limits.block_tokens,
            limits.max_batch_tokens,
            limits.max_seqs,
        )
    }

    fn default_tool_format(&self) -> Option<&'static str> {
        Some(LLAMA3_JSON)
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
        Ok(Box::new(LlamaExecutor::new(
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

/// Every parameter slot of a Llama model, in load order: embedding, per layer the attention
/// and MLP weights with their norms, the final norm, and `lm_head.weight` only when untied.
/// Q/K/V land as rows of the layer's fused [`crate::loader::qkv_proj_name`] parameter and
/// gate/up as rows of its [`gate_up_proj_name`] parameter, so the executor runs one GEMM for each (or one per
/// projection over row views of the same memory).
pub fn llama_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
    let kv = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    let inter = cfg.intermediate as usize;
    let vocab = cfg.vocab_size as usize;
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
            slot(format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
        ]);
        slots.extend(row_concat(
            gate_up_proj_name(i),
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
    slots.push(slot("model.norm.weight".into(), vec![hidden]));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}
