//! Model layer (Phase 1): the Hugging Face config and its architecture allowlist, safetensors
//! loading, tokenizer and chat template, the Llama executor, sampling and the single-request
//! generation loop.
use std::path::PathBuf;

use turbine_kernels::KernelError;
use turbine_tensor::MemoryError;

pub mod budget;
pub mod chat_template;
pub mod config;
pub mod loader;
pub mod safetensors;
pub mod testing;
pub mod tokenizer;

pub use crate::safetensors::{SafetensorsIndex, TensorEntry};
pub use budget::{BudgetTerms, available_bytes, check_budget, host_mem_available};
pub use chat_template::ChatTemplate;
pub use config::{
    Architecture, GenerationConfig, ModelArchConfig, RopeScaling, load_generation_config,
    load_model_config,
};
pub use loader::{LoadedWeights, MAX_STAGING_BYTES, WeightLoader, WeightSlot, llama_slots};
pub use tokenizer::{IncrementalDetokenizer, Tokenizer};

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
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

impl From<MemoryError> for ModelError {
    fn from(e: MemoryError) -> ModelError {
        ModelError::Kernel(e.into())
    }
}
