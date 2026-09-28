//! transformers FP8 checkpoints (`hf_fp8`, Phase 6a S-3): `quant_method: fp8`, `fmt: e4m3`.
//! With `weight_block_size: [128, 128]` (Qwen3-FP8 / DeepSeek style, column `fp8_block`) the
//! weights carry `X.weight_scale_inv` per 128 × 128 block and dynamic activations are quantized
//! per 128-element group; without it (column `fp8`) `X.weight_scale` is per tensor and dynamic
//! activations are quantized per token (vLLM quantizes them per tensor, dynamically; per token
//! is at least as accurate — builder decision 2026-09-29, provisional). `activation_scheme:
//! static` reads `X.input_scale`. `modules_to_not_convert` (or `ignored_layers`) names the
//! modules left in BF16 ([`super::fp8`]).

use serde_json::{Value, json};

use super::ActivationQuant;
use super::common::{pair, scheme_unsupported, string_list};
use super::fp8::{FP8_BLOCK, Fp8Format, Fp8Layout, Fp8Packaging, Fp8Weights, check_block};
use crate::ModelError;
use crate::config::unsupported;

/// The transformers `quant_method: fp8` container.
pub struct HfFp8;

/// The registry entry.
pub static HF_FP8: Fp8Format<HfFp8> = Fp8Format::ENTRY;

/// `obj[key]` as a string, `""` when absent.
fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

impl Fp8Packaging for HfFp8 {
    const NAME: &'static str = "hf_fp8";
    /// Per-tensor scales: every registered family's tiny checkpoint (hidden 64) can hold them;
    /// block scales need dimensions in multiples of 128.
    const DEFAULT: Fp8Layout = Fp8Layout {
        weights: Fp8Weights::Tensor,
        act: ActivationQuant::Fp8PerTokenDynamic,
        ignore: Vec::new(),
        scale_suffix: "weight_scale",
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        if text(q, "quant_method") == "fp8" {
            Ok(())
        } else {
            Err(unsupported(
                "quantization_config.quant_method",
                text(q, "quant_method"),
                "fp8",
            ))
        }
    }

    fn parse(q: &Value) -> Result<Fp8Layout, ModelError> {
        let fmt = q.get("fmt").and_then(Value::as_str).unwrap_or("e4m3");
        if fmt != "e4m3" {
            return Err(scheme_unsupported("fmt", fmt, "e4m3"));
        }
        let block = match q.get("weight_block_size").filter(|b| !b.is_null()) {
            None => None,
            Some(b) => Some(check_block(
                "weight_block_size",
                pair(b).ok_or_else(|| {
                    scheme_unsupported("weight_block_size", b.to_string(), "[128, 128]")
                })?,
            )?),
        };
        let act = match (text(q, "activation_scheme"), block) {
            ("static", _) => ActivationQuant::Fp8PerTensorStatic,
            ("dynamic", Some(_)) => ActivationQuant::Fp8PerGroupDynamic { group: FP8_BLOCK },
            ("dynamic", None) => ActivationQuant::Fp8PerTokenDynamic,
            (other, _) => {
                return Err(scheme_unsupported(
                    "activation_scheme",
                    other,
                    "dynamic, static",
                ));
            }
        };
        let mut ignore = string_list(q, "modules_to_not_convert");
        ignore.extend(string_list(q, "ignored_layers"));
        Ok(Fp8Layout {
            weights: block.unwrap_or(Fp8Weights::Tensor),
            act,
            ignore,
            scale_suffix: if block.is_some() {
                "weight_scale_inv"
            } else {
                "weight_scale"
            },
        })
    }

    fn to_json(layout: &Fp8Layout) -> Value {
        let block = match layout.weights {
            Fp8Weights::Block { n, k } => json!([n, k]),
            _ => Value::Null,
        };
        let scheme = if layout.act == ActivationQuant::Fp8PerTensorStatic {
            "static"
        } else {
            "dynamic"
        };
        json!({
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": scheme,
            "weight_block_size": block,
            "modules_to_not_convert": layout.ignore,
        })
    }
}
