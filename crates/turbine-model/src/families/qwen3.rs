//! The Qwen3 family (`qwen3`, `Qwen3ForCausalLM`; Phase 8 run-ahead, Phase 2m S-11): the
//! Llama layer plus an RMSNorm over each Q and K head (`self_attn.q_norm` / `k_norm` of
//! `[head_dim]`) before RoPE, as the [`QK_NORM_PER_HEAD`] attention hook, which keeps the
//! Q/K/V projections unfused (gate/up still fuse). Its keys: `use_sliding_window` must not be
//! true (sliding windows are refused, an open Phase 8 decision; `sliding_window` itself is then
//! unused). Tool calls default to `hermes`. CPU backend only: the support matrix refuses it on
//! GPU vendors until its Phase 8 track closes.

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::llama::dense_slots;
use super::{FamilyConfig, ModelFamily, invalid};
use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::decoder::{QK_NORM_PER_HEAD, SWIGLU};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::formats::hermes::HERMES;
use crate::loader::{LoadedWeights, WeightSlot};

/// `Qwen3ForCausalLM` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct Qwen3;

/// The Qwen3 attention keys (shared with Qwen3-MoE).
#[derive(Deserialize)]
struct Qwen3AttentionKeys {
    #[serde(default)]
    use_sliding_window: Option<bool>,
}

/// Qwen3 attention: sliding-window attention off (`use_sliding_window` absent or false).
pub(crate) fn qwen3_attention(text: &serde_json::Value) -> Result<(), ModelError> {
    let keys =
        Qwen3AttentionKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
    if keys.use_sliding_window == Some(true) {
        return Err(unsupported("use_sliding_window", "true", "false"));
    }
    Ok(())
}

impl Module for Qwen3 {
    fn name(&self) -> &'static str {
        "qwen3"
    }
}

impl ModelFamily for Qwen3 {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &["Qwen3ForCausalLM"]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        qwen3_attention(text)?;
        Ok(FamilyConfig {
            moe: None,
            qk_norm: false,
            qk_norm_per_head: true,
        })
    }

    /// [`crate::families::llama_slots`] with the per-head `q_norm` / `k_norm` (`[head_dim]`)
    /// after the O projection.
    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        let head_dim = cfg.head_dim as usize;
        dense_slots(cfg, Some((head_dim, head_dim)))
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

/// The Qwen3 decoder: RMSNorm over each Q and K head and the dense SwiGLU MLP.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: QK_NORM_PER_HEAD,
        ffn: SWIGLU,
    }
}
