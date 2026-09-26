//! The Mistral family (`mistral`, `MistralForCausalLM`; Phase 8 run-ahead, Phase 2m S-11): the
//! Llama layer as is ([`PLAIN_ATTENTION`], [`SWIGLU`], Llama's weight slots). Its key:
//! `sliding_window` must be null or at least `max_position_embeddings` (it then never limits
//! attention); a real window is refused (an open Phase 8 decision). Tool calls default to
//! `mistral`. CPU backend only: the support matrix refuses it on GPU vendors until its Phase 8
//! track closes.

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::llama::llama_slots;
use super::{FamilyConfig, ModelFamily, invalid};
use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::decoder::{PLAIN_ATTENTION, SWIGLU};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::formats::mistral::MISTRAL;
use crate::loader::{LoadedWeights, WeightSlot};

/// `MistralForCausalLM` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct Mistral;

/// The Mistral attention keys (shared with Mixtral).
#[derive(Deserialize)]
struct WindowKeys {
    max_position_embeddings: u32,
    #[serde(default)]
    sliding_window: Option<serde_json::Value>,
}

/// Mistral-family attention: `sliding_window` null, or at least `max_position_embeddings`.
pub(crate) fn full_attention_only(text: &serde_json::Value) -> Result<(), ModelError> {
    let keys = WindowKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
    match keys.sliding_window.as_ref().filter(|w| !w.is_null()) {
        None => Ok(()),
        Some(w)
            if w.as_u64()
                .is_some_and(|w| w >= u64::from(keys.max_position_embeddings)) =>
        {
            Ok(())
        }
        Some(w) => Err(unsupported(
            "sliding_window",
            w.to_string(),
            "null, or at least max_position_embeddings",
        )),
    }
}

impl Module for Mistral {
    fn name(&self) -> &'static str {
        "mistral"
    }
}

impl ModelFamily for Mistral {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &["MistralForCausalLM"]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        full_attention_only(text)?;
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
        DecoderExecutor::requirements(cfg, &decoder_spec(), block_tokens, opts)
    }

    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        DecoderExecutor::workspace_bytes(cfg, &decoder_spec(), limits)
    }

    fn default_tool_format(&self) -> Option<&'static str> {
        Some(MISTRAL)
    }

    /// The family's tiny checkpoint (`testing::tiny`).
    fn write_tiny(&self, dir: &std::path::Path, seed: u64) -> crate::testing::tiny::TinySpec {
        crate::testing::tiny::write_tiny_family(dir, self.name(), seed)
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

/// The Mistral decoder: Llama's plain attention and dense SwiGLU MLP.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: PLAIN_ATTENTION,
        ffn: SWIGLU,
    }
}
