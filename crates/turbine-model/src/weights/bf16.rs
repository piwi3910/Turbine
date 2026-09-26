//! BF16 weights, activations and KV (`bf16`): the only format through Phase 2. A checkpoint is
//! BF16 when `config.json` has no `quantization_config` and its `torch_dtype` / `dtype` (when
//! present) is `bfloat16`, and every tensor the executor loads is stored as BF16.

use turbine_core::registry::Module;
use turbine_core::types::DType;

use super::WeightFormat;
use crate::ModelError;
use crate::config::unsupported;
use crate::safetensors::{Dtype, TensorEntry};

/// The `torch_dtype` / `dtype` value that declares BF16 weights.
const TORCH_DTYPE: &str = "bfloat16";

/// BF16 weights, activations and KV cache.
pub struct Bf16;

impl Bf16 {
    /// The one dtype of the format, for code that is BF16-only by construction (the Llama and
    /// OLMoE executors, until the shared decoder takes the dtype from the configuration).
    pub const DTYPE: DType = DType::BF16;
}

impl Module for Bf16 {
    fn name(&self) -> &'static str {
        "bf16"
    }
}

impl WeightFormat for Bf16 {
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError> {
        if let Some(q) = top.get("quantization_config").filter(|q| !q.is_null()) {
            return Err(unsupported("quantization_config", q.to_string(), "none"));
        }
        // transformers ≥ 4.56 writes `dtype` instead of `torch_dtype`.
        for field in ["torch_dtype", "dtype"] {
            if let Some(value) = top.get(field).filter(|v| !v.is_null()) {
                let text = value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_string);
                if text != TORCH_DTYPE {
                    return Err(unsupported(field, text, TORCH_DTYPE));
                }
            }
        }
        Ok(())
    }

    fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError> {
        if entry.dtype == Dtype::BF16 {
            Ok(())
        } else {
            Err(ModelError::Unsupported {
                field: "tensor dtype".to_string(),
                value: format!("{} ({})", entry.dtype, entry.name),
                supported: "BF16".to_string(),
            })
        }
    }

    fn weight_dtype(&self) -> DType {
        Bf16::DTYPE
    }

    fn activation_dtype(&self) -> DType {
        Bf16::DTYPE
    }

    fn kv_dtype(&self) -> DType {
        Bf16::DTYPE
    }

    fn bytes_per_param(&self) -> u64 {
        Bf16::DTYPE.size_bytes() as u64
    }
}
