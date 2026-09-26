//! Model layer (Phase 1): the Hugging Face config and its architecture allowlist, safetensors
//! loading, tokenizer and chat template, the Llama executor, sampling and the single-request
//! generation loop; Phase 2 adds sampler penalties and constrained decoding (`structured`).
use std::path::PathBuf;

use turbine_kernels::KernelError;
use turbine_tensor::MemoryError;

pub mod budget;
pub mod chat_template;
pub mod config;
pub mod executor;
pub mod families;
pub mod formats;
pub mod generate;
pub mod loader;
pub mod metrics;
pub mod registries;
pub mod safetensors;
pub mod sampling;
pub mod structured;
pub mod testing;
pub mod tokenizer;
pub mod tools;
pub mod weights;

/// The sampler at its Phase 1 path, so `turbine_model::sampler::…` keeps resolving.
pub use sampling::sampler;

pub use crate::safetensors::{SafetensorsIndex, TensorEntry};
pub use budget::{BudgetTerms, available_bytes, check_budget, host_mem_available};
pub use chat_template::ChatTemplate;
pub use config::MoeConfig;
pub use config::{
    GenerationConfig, ModelArchConfig, RopeScaling, load_generation_config, load_model_config,
};
pub use families::{FamilyConfig, FamilyRef, ModelFamily, llama_slots, mixtral_slots, olmoe_slots};
pub use formats::{BoundToolFormat, HermesParser, Llama3JsonParser, MistralParser, ToolFormat};
pub use generate::{GenerateOptions, Generation, generate};
pub use loader::{LoadedWeights, MAX_STAGING_BYTES, WeightLoader, WeightSlot};
pub use loader::{StackPlace, gate_up_proj_name, qkv_proj_name, stacked_experts_name};
pub use metrics::{ForwardPhase, ModelMetrics, ToolCallOutcome};
pub use sampler::{SampleJob, SampledToken, Sampler, SamplerState, sample_rows};
pub use structured::{
    GrammarCompiler, GrammarLimits, JSON_MAX_WHITESPACE, TokenMask, TokenMatcher, constraint_kind,
    json_options, step_mask,
};
pub use tokenizer::{IncrementalDetokenizer, Tokenizer};
pub use tools::{ToolCallParser, ToolChoice, ToolParse, new_call_id, tool_call_grammar};

/// Every failure of the model layer (contract §10). Messages name the offending file, field or
/// tensor so a startup failure is actionable from the log line alone.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    #[error("{}: {detail}", path.display())]
    Io { path: PathBuf, detail: String },
    #[error("{}: pickle formats are not supported", path.display())]
    Pickle { path: PathBuf },
    #[error("unsupported {field} = {value}; supported: {supported}")]
    Unsupported {
        field: String,
        value: String,
        supported: String,
    },
    #[error("{}: tensor {tensor}: {rule}", file.display())]
    Safetensors {
        file: PathBuf,
        tensor: String,
        rule: String,
    },
    #[error("missing tensor {0}")]
    MissingTensor(String),
    /// Lists weights, KV reservation, workspace, emergency reserve and available bytes.
    #[error("memory budget exceeded: {0}")]
    Budget(String),
    #[error("template: {0}")]
    Template(String),
    /// Constrained decoding (P2 S-17/S-18): a grammar that does not compile or exceeds its
    /// bounds (naming the keyword or bound), or a matcher failing mid-generation.
    #[error("constraint: {0}")]
    Constraint(String),
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

impl From<MemoryError> for ModelError {
    fn from(e: MemoryError) -> ModelError {
        ModelError::Kernel(e.into())
    }
}
