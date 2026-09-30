//! compressed-tensors `pack-quantized` W4A16 checkpoints (`ct_pack_int4`, Phase 6a S-3, column
//! `gptq_int4`): `quant_method: compressed-tensors`, `format: pack-quantized`, one config group
//! whose `weights` are 4-bit symmetric `int` with strategy `group` (32 / 64 / 128) and
//! `actorder` null or `static`; no input activations; the `ignore` list honoured. Tensors and repack:
//! [`super::int4`].

use serde_json::{Value, json};

use super::common::{
    ct_check_kv_scheme, ct_check_output_activations, scheme_unsupported, string_list,
};
use super::int4::{Int4Format, Int4Kind, Int4Layout, Int4Packaging, check_group};
use crate::ModelError;
use crate::config::unsupported;
use turbine_core::support::WeightFormatColumn;

/// The compressed-tensors packed INT4 container.
pub struct CtPackInt4;

/// The registry entry.
pub static CT_PACK_INT4: Int4Format<CtPackInt4> = Int4Format::ENTRY;

fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

impl Int4Packaging for CtPackInt4 {
    const NAME: &'static str = "ct_pack_int4";
    const COLUMN: WeightFormatColumn = WeightFormatColumn::GptqInt4;
    const DEFAULT: Int4Layout = Int4Layout {
        kind: Int4Kind::CtPack,
        group: 32,
        zero_points: false,
        zeros_offset: 0,
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
        if text(q, "format") != "pack-quantized" {
            return Err(unsupported(
                "quantization_config.format",
                text(q, "format"),
                "pack-quantized",
            ));
        }
        Ok(())
    }

    fn parse(q: &Value) -> Result<Int4Layout, ModelError> {
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
        ct_check_kv_scheme(q)?;
        ct_check_output_activations(g)?;
        if g.get("input_activations").is_some_and(|a| !a.is_null()) {
            return Err(scheme_unsupported(
                "input_activations",
                "set",
                "null (weight-only)",
            ));
        }
        let w = g.get("weights").unwrap_or(&Value::Null);
        let bits = w.get("num_bits").and_then(Value::as_u64);
        if bits != Some(4) || text(w, "type") != "int" || text(w, "strategy") != "group" {
            return Err(scheme_unsupported(
                "weights",
                format!("{bits:?}-bit {} {}", text(w, "type"), text(w, "strategy")),
                "4-bit int, strategy group",
            ));
        }
        if !w.get("symmetric").and_then(Value::as_bool).unwrap_or(false) {
            return Err(scheme_unsupported("weights.symmetric", "false", "true"));
        }
        // `static` act order quantizes groups in order: nothing to permute at run time.
        if let Some(a) = w
            .get("actorder")
            .filter(|a| !a.is_null() && a.as_str() != Some("static"))
        {
            return Err(unsupported(
                "weights.actorder",
                a.to_string(),
                "gptq_act_order: null or static",
            ));
        }
        Ok(Int4Layout {
            kind: Int4Kind::CtPack,
            group: check_group(
                "weights.group_size",
                w.get("group_size").and_then(Value::as_u64),
            )?,
            zero_points: false,
            zeros_offset: 0,
            ignore: string_list(q, "ignore"),
        })
    }

    fn to_json(layout: &Int4Layout) -> Value {
        json!({
            "config_groups": {
                "group_0": {
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4, "type": "int", "symmetric": true, "strategy": "group",
                        "group_size": layout.group, "dynamic": false, "actorder": null,
                        "block_structure": null, "observer": "minmax", "observer_kwargs": {},
                    },
                },
            },
            "format": "pack-quantized",
            "ignore": layout.ignore,
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        })
    }
}
