//! Weight formats (Phase 2m S-10, contract §24 `weight_format`): how a checkpoint's weights
//! are stored and which dtypes the executor, the loader and the KV cache use for them. One file
//! per format and one entry in [`registry`]; [`detect`] picks the format of a `config.json`.

use turbine_core::registry::{Module, Registry};
use turbine_core::types::DType;

use crate::ModelError;
use crate::safetensors::TensorEntry;

pub mod bf16;

pub use bf16::Bf16;

/// A checkpoint weight format: the `config.json` keys that declare it, the tensors it accepts
/// and the dtypes weights, activations and the KV cache use.
pub trait WeightFormat: Module {
    /// `Ok` when the top-level `config.json` declares this format; else the refusal naming the
    /// offending key.
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError>;
    /// `Ok` when a checkpoint tensor is stored in this format; else the refusal naming it.
    fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError>;
    /// The dtype of the loaded parameters.
    fn weight_dtype(&self) -> DType;
    /// The dtype of the executor's activations.
    fn activation_dtype(&self) -> DType;
    /// The dtype of the KV cache.
    fn kv_dtype(&self) -> DType;
    /// Stored bytes per parameter (the budget's weight term).
    fn bytes_per_param(&self) -> u64;
}

/// A registered weight format as a value of `ModelArchConfig`: equal by name, printed as its
/// name.
#[derive(Clone, Copy)]
pub struct WeightFormatRef(pub &'static dyn WeightFormat);

impl PartialEq for WeightFormatRef {
    fn eq(&self, other: &WeightFormatRef) -> bool {
        self.0.name() == other.0.name()
    }
}

impl Eq for WeightFormatRef {}

impl std::fmt::Debug for dyn WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Debug for WeightFormatRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.name())
    }
}

static WEIGHT_FORMATS: Registry<dyn WeightFormat> = Registry::new("weight_format", &[&Bf16]);

/// Every weight format, in detection order.
pub fn registry() -> &'static Registry<dyn WeightFormat> {
    &WEIGHT_FORMATS
}

/// The format of a top-level `config.json`: the first registered format whose
/// [`WeightFormat::check_config`] accepts it, else the first format's refusal.
pub fn detect(top: &serde_json::Value) -> Result<&'static dyn WeightFormat, ModelError> {
    let mut first_err = None;
    for format in registry().iter() {
        match format.check_config(top) {
            Ok(()) => return Ok(format),
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    Err(first_err.expect("the weight-format registry is not empty"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;
    use crate::safetensors::{Dtype, TensorEntry};

    fn entry(dtype: Dtype) -> TensorEntry {
        TensorEntry {
            name: "model.layers.0.mlp.down_proj.weight".to_string(),
            dtype,
            shape: vec![4, 4],
            file: PathBuf::from("model.safetensors"),
            range: 0..32,
        }
    }

    /// The refusals main's `load_model_config` and `check_supported_weights` produced, kept as
    /// literals so the move cannot change a message.
    #[test]
    fn bf16_refusals_match_main() {
        let format = registry().get("bf16").expect("bf16 is registered");
        let err = |top: serde_json::Value| format.check_config(&top).unwrap_err().to_string();
        assert_eq!(
            err(json!({"quantization_config": {"quant_method": "fp8"}})),
            r#"unsupported quantization_config = {"quant_method":"fp8"}; supported: none"#
        );
        assert_eq!(
            err(json!({"torch_dtype": "float16"})),
            "unsupported torch_dtype = float16; supported: bfloat16"
        );
        assert_eq!(
            err(json!({"dtype": "float32"})),
            "unsupported dtype = float32; supported: bfloat16"
        );
        // A null quantization_config is no quantization.
        format
            .check_config(&json!({"quantization_config": null}))
            .unwrap();
        assert_eq!(
            format
                .check_tensor(&entry(Dtype::F16))
                .unwrap_err()
                .to_string(),
            "unsupported tensor dtype = F16 (model.layers.0.mlp.down_proj.weight); supported: BF16"
        );
        format.check_tensor(&entry(Dtype::BF16)).unwrap();
        // The same refusal through `detect`: the first format's error.
        assert_eq!(
            detect(&json!({"torch_dtype": "float16"}))
                .unwrap_err()
                .to_string(),
            "unsupported torch_dtype = float16; supported: bfloat16"
        );
    }

    #[test]
    fn detect_picks_bf16() {
        for top in [
            json!({}),
            json!({"torch_dtype": "bfloat16"}),
            json!({"dtype": "bfloat16"}),
        ] {
            let format = detect(&top).unwrap();
            assert_eq!(format.name(), "bf16", "{top}");
            assert_eq!(format.bytes_per_param(), 2);
            assert_eq!(format.weight_dtype().size_bytes(), 2);
            assert_eq!(format.activation_dtype(), format.weight_dtype());
            assert_eq!(format.kv_dtype(), format.weight_dtype());
        }
        let a = WeightFormatRef(detect(&json!({})).unwrap());
        assert_eq!(a, WeightFormatRef(registry().get("bf16").unwrap()));
        assert_eq!(format!("{a:?}"), "bf16");
    }
}
