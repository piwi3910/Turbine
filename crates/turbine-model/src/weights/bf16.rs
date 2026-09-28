//! BF16 checkpoints (`bf16`): no `quantization_config`, `torch_dtype` / `dtype` (when present)
//! `bfloat16`, and every tensor the executor loads stored as BF16. Every linear layer is
//! [`super::QuantScheme::Bf16`] (the trait's defaults).

use turbine_core::registry::Module;
use turbine_core::support::WeightFormatColumn;
use turbine_core::types::DType;

use super::WeightFormat;
use crate::ModelError;
use crate::config::unsupported;
use crate::safetensors::{Dtype, TensorEntry};

/// The `torch_dtype` / `dtype` value that declares BF16 weights.
const TORCH_DTYPE: &str = "bfloat16";

/// BF16 weights.
pub struct Bf16;

impl Bf16 {
    /// BF16: the stored dtype of this format, the executor's activation dtype
    /// ([`super::ACTIVATION_DTYPE`]) and the default KV dtype.
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

    fn column(&self) -> WeightFormatColumn {
        WeightFormatColumn::Bf16
    }
}
