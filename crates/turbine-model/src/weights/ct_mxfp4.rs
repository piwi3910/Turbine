//! compressed-tensors MXFP4 checkpoints (`ct_mxfp4`, Phase 6a S-3, column `mxfp4`):
//! `quant_method: compressed-tensors`, `format: mxfp4-pack-quantized`, one config group whose
//! `weights` are 4-bit `float` with strategy `group` 32 and E8M0 (`uint8`) scales, `actorder`
//! null or `static` (static groups store the weights in their order); no input activations;
//! the `ignore` list honoured. Tensors: `X.weight_packed` U8 `[n, k/2]`, `X.weight_scale` U8
//! `[n, k/32]` ([`super::mxfp4`]).

use serde_json::{Value, json};

use super::ActivationQuant;
use super::common::{scheme_unsupported, string_list};
use super::mxfp4::{Mxfp4Format, Mxfp4Kind, Mxfp4Layout, Mxfp4Packaging};
use crate::ModelError;
use crate::config::unsupported;

/// The compressed-tensors MXFP4 container.
pub struct CtMxfp4;

/// The registry entry.
pub static CT_MXFP4: Mxfp4Format<CtMxfp4> = Mxfp4Format::ENTRY;

const FORMAT: &str = "mxfp4-pack-quantized";

fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

impl Mxfp4Packaging for CtMxfp4 {
    const NAME: &'static str = "ct_mxfp4";
    const DEFAULT: Mxfp4Layout = Mxfp4Layout {
        kind: Mxfp4Kind::CompressedTensors,
        act: ActivationQuant::None,
        ignore: Vec::new(),
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        if text(q, "quant_method") != "compressed-tensors" {
            return Err(unsupported(
                "quantization_config.quant_method",
                text(q, "quant_method"),
                "compressed-tensors",
            ));
        }
        if text(q, "format") != FORMAT {
            return Err(unsupported(
                "quantization_config.format",
                text(q, "format"),
                FORMAT,
            ));
        }
        Ok(())
    }

    fn parse(q: &Value) -> Result<Mxfp4Layout, ModelError> {
        let groups = q
            .get("config_groups")
            .and_then(Value::as_object)
            .filter(|g| g.len() == 1)
            .ok_or_else(|| scheme_unsupported("config_groups", "not one group", "one group"))?;
        let g = groups.values().next().expect("one group");
        if string_list(g, "targets") != ["Linear"] {
            return Err(scheme_unsupported(
                "targets",
                format!("{:?}", string_list(g, "targets")),
                "[\"Linear\"]",
            ));
        }
        if g.get("input_activations").is_some_and(|a| !a.is_null()) {
            return Err(scheme_unsupported(
                "input_activations",
                "set",
                "null (weight-only)",
            ));
        }
        let w = g.get("weights").unwrap_or(&Value::Null);
        let bits = w.get("num_bits").and_then(Value::as_u64);
        let group = w.get("group_size").and_then(Value::as_u64);
        if bits != Some(4) || text(w, "type") != "float" || text(w, "strategy") != "group" {
            return Err(scheme_unsupported(
                "weights",
                format!("{bits:?}-bit {} {}", text(w, "type"), text(w, "strategy")),
                "4-bit float, strategy group",
            ));
        }
        if group != Some(32) {
            return Err(scheme_unsupported(
                "weights.group_size",
                format!("{group:?}"),
                "32",
            ));
        }
        match w.get("actorder").filter(|a| !a.is_null()) {
            None => {}
            Some(a) if a.as_str() == Some("static") => {}
            Some(a) => {
                return Err(unsupported(
                    "weights.actorder",
                    a.to_string(),
                    "gptq_act_order: null or static",
                ));
            }
        }
        Ok(Mxfp4Layout {
            kind: Mxfp4Kind::CompressedTensors,
            act: ActivationQuant::None,
            ignore: string_list(q, "ignore"),
        })
    }

    fn to_json(layout: &Mxfp4Layout) -> Value {
        json!({
            "config_groups": {
                "group_0": {
                    "format": FORMAT,
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4, "type": "float", "symmetric": true, "strategy": "group",
                        "group_size": 32, "dynamic": false, "actorder": "static",
                        "scale_dtype": "torch.uint8", "zp_dtype": null,
                    },
                },
            },
            "format": FORMAT,
            "ignore": layout.ignore,
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        })
    }
}
