//! The Qwen3-MoE family (`qwen3_moe`, `Qwen3MoeForCausalLM`; Phase 8 run-ahead, Phase 2m S-11):
//! the Qwen3 attention ([`QK_NORM_PER_HEAD`]) and a sparse mixture of SwiGLU experts ([`MOE`]).
//! Its keys: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` as each expert's
//! width (the dense `intermediate_size` is unused), `norm_topk_prob` (default false, as
//! transformers' `Qwen3MoeConfig`); every layer must be sparse (`decoder_sparse_step` 1, no
//! `mlp_only_layers`), and `use_sliding_window` must not be true. Checkpoint names are OLMoE's
//! (`mlp.gate`, `mlp.experts.<e>.{gate,up,down}_proj`). Tool calls default to `hermes`. CPU
//! backend only: the support matrix refuses it on GPU vendors until its Phase 8 track closes.

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::olmoe::{OLMOE_NAMES, moe_slots};
use super::qwen3::qwen3_attention;
use super::{FamilyConfig, ModelFamily, experts, invalid};
use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::decoder::{MOE, QK_NORM_PER_HEAD};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::formats::hermes::HERMES;
use crate::loader::{LoadedWeights, WeightSlot};

/// `Qwen3MoeForCausalLM` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct Qwen3Moe;

/// The HF class name, as error messages name it.
const HF_NAME: &str = "Qwen3MoeForCausalLM";

/// The Qwen3-MoE expert keys of `config.json`.
#[derive(Deserialize)]
struct Qwen3MoeKeys {
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    /// Every `decoder_sparse_step`-th layer is sparse.
    #[serde(default)]
    decoder_sparse_step: Option<u32>,
    /// Layers with a dense MLP instead of experts.
    #[serde(default)]
    mlp_only_layers: Option<Vec<u32>>,
}

impl Module for Qwen3Moe {
    fn name(&self) -> &'static str {
        "qwen3_moe"
    }
}

impl ModelFamily for Qwen3Moe {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &[HF_NAME]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        qwen3_attention(text)?;
        let keys =
            Qwen3MoeKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        // Every layer sparse: step 1 and no dense layers (every released Qwen3-MoE checkpoint).
        if let Some(step) = keys.decoder_sparse_step.filter(|&s| s != 1) {
            return Err(unsupported("decoder_sparse_step", step.to_string(), "1"));
        }
        if let Some(dense) = keys.mlp_only_layers.as_ref().filter(|l| !l.is_empty()) {
            return Err(unsupported("mlp_only_layers", format!("{dense:?}"), "[]"));
        }
        let moe = experts(
            HF_NAME,
            ("num_experts", keys.num_experts),
            keys.num_experts_per_tok,
            ("moe_intermediate_size", keys.moe_intermediate_size),
            keys.norm_topk_prob.unwrap_or(false),
        )?;
        Ok(FamilyConfig {
            moe: Some(moe),
            qk_norm: false,
            qk_norm_per_head: true,
        })
    }

    /// [`crate::families::olmoe_slots`] with per-head `q_norm` / `k_norm` (`[head_dim]`).
    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        let head_dim = cfg.head_dim as usize;
        moe_slots(cfg, &OLMOE_NAMES, Some((head_dim, head_dim)))
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
        Some(HERMES)
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

/// The Qwen3-MoE decoder: RMSNorm over each Q and K head and the mixture of experts.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: QK_NORM_PER_HEAD,
        ffn: MOE,
    }
}
