//! The Mixtral family (`mixtral`, `MixtralForCausalLM`; Phase 8 run-ahead, Phase 2m S-11):
//! Mistral's plain attention ([`PLAIN_ATTENTION`]) and a sparse mixture of SwiGLU experts
//! ([`MOE`]) that always renormalises its top-k weights. Its keys: `num_local_experts`,
//! `num_experts_per_tok`, `intermediate_size` as each expert's width, and `sliding_window` as
//! Mistral's (null or covering every position). Its checkpoint names
//! (`block_sparse_moe.gate`, experts `w1` / `w3` / `w2`) are mapped onto the MoE hook's
//! parameters by [`mixtral_slots`]. Tool calls default to `mistral`. CPU backend only: the
//! support matrix refuses it on GPU vendors until its Phase 8 track closes.

use std::sync::Arc;

use serde::Deserialize;
use turbine_core::registry::Module;
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::DeviceMemory;

use super::mistral::full_attention_only;
use super::olmoe::{MoeNames, moe_slots};
use super::{FamilyConfig, ModelFamily, experts, invalid};
use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::executor::decoder::{MOE, PLAIN_ATTENTION};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::formats::mistral::MISTRAL;
use crate::loader::{LoadedWeights, WeightSlot};

/// `MixtralForCausalLM` on the shared [`DecoderExecutor`] with [`decoder_spec`].
pub struct Mixtral;

/// The HF class name, as error messages name it.
const HF_NAME: &str = "MixtralForCausalLM";

/// Mixtral's checkpoint names: `block_sparse_moe`, with `w1` (gate), `w3` (up), `w2` (down).
const MIXTRAL_NAMES: MoeNames = MoeNames {
    router: "block_sparse_moe.gate",
    experts: "block_sparse_moe.experts",
    gate: "w1",
    up: "w3",
    down: "w2",
};

/// The Mixtral expert keys of `config.json`.
#[derive(Deserialize)]
struct MixtralKeys {
    intermediate_size: u32,
    #[serde(default)]
    num_local_experts: Option<u32>,
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
}

impl Module for Mixtral {
    fn name(&self) -> &'static str {
        "mixtral"
    }
}

impl ModelFamily for Mixtral {
    fn hf_architectures(&self) -> &'static [&'static str] {
        &[HF_NAME]
    }

    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        full_attention_only(text)?;
        let keys =
            MixtralKeys::deserialize(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        // Mixtral always renormalises the top-k routing weights (no `norm_topk_prob` key).
        let moe = experts(
            HF_NAME,
            ("num_local_experts", keys.num_local_experts),
            keys.num_experts_per_tok,
            ("intermediate_size", Some(keys.intermediate_size)),
            true,
        )?;
        Ok(FamilyConfig {
            moe: Some(moe),
            qk_norm: false,
            qk_norm_per_head: false,
        })
    }

    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        mixtral_slots(cfg)
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

/// The Mixtral decoder: plain attention and the mixture of experts.
pub fn decoder_spec() -> DecoderSpec {
    DecoderSpec {
        attention: PLAIN_ATTENTION,
        ffn: MOE,
    }
}

/// [`crate::families::olmoe_slots`] for Mixtral's checkpoint names and without Q/K norm: the
/// router `block_sparse_moe.gate.weight` lands as `model.layers.<i>.mlp.gate.weight` (a
/// one-part stack), experts `w1` / `w3` / `w2` in the
/// [`crate::loader::stacked_experts_name`] stacks of `gate_proj`, `up_proj` and `down_proj`.
pub fn mixtral_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    moe_slots(cfg, &MIXTRAL_NAMES, None)
}
