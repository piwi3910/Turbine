//! AutoGPTQ checkpoints (`gptq`, Phase 6a S-3, column `gptq_int4`): `quant_method: gptq`,
//! `bits: 4`, `group_size` 32 / 64 / 128, `desc_act: false` (act order is refused,
//! `gptq_act_order`), `sym` true (implicit zero point 8) or false (stored zero points),
//! `checkpoint_format` `gptq` (zeros stored minus one) or `gptq_v2`. Tensors and repack:
//! [`super::int4`].

use serde_json::{Value, json};

use super::common::scheme_unsupported;
use super::int4::{Int4Format, Int4Kind, Int4Layout, Int4Packaging, check_group};
use crate::ModelError;
use crate::config::unsupported;
use turbine_core::support::WeightFormatColumn;

/// The AutoGPTQ container.
pub struct Gptq;

/// The registry entry.
pub static GPTQ: Int4Format<Gptq> = Int4Format::ENTRY;

fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

impl Int4Packaging for Gptq {
    const NAME: &'static str = "gptq";
    const COLUMN: WeightFormatColumn = WeightFormatColumn::GptqInt4;
    const DEFAULT: Int4Layout = Int4Layout {
        kind: Int4Kind::Gptq,
        group: 32,
        zero_points: false,
        zeros_offset: 1,
        ignore: Vec::new(),
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        if text(q, "quant_method") == "gptq" {
            Ok(())
        } else {
            Err(unsupported(
                "quantization_config.quant_method",
                text(q, "quant_method"),
                "gptq",
            ))
        }
    }

    fn parse(q: &Value) -> Result<Int4Layout, ModelError> {
        let bits = q.get("bits").and_then(Value::as_u64);
        if bits != Some(4) {
            return Err(scheme_unsupported("bits", format!("{bits:?}"), "4"));
        }
        let group = check_group("group_size", q.get("group_size").and_then(Value::as_u64))?;
        if q.get("desc_act").and_then(Value::as_bool).unwrap_or(false) {
            return Err(unsupported(
                "desc_act",
                "true",
                "gptq_act_order: false (act-order GPTQ is not served)",
            ));
        }
        if q.get("is_marlin_format")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(scheme_unsupported("is_marlin_format", "true", "false"));
        }
        let zeros_offset = match text(q, "checkpoint_format") {
            "" | "gptq" => 1,
            "gptq_v2" => 0,
            other => {
                return Err(scheme_unsupported(
                    "checkpoint_format",
                    other,
                    "gptq, gptq_v2",
                ));
            }
        };
        Ok(Int4Layout {
            kind: Int4Kind::Gptq,
            group,
            zero_points: !q.get("sym").and_then(Value::as_bool).unwrap_or(true),
            zeros_offset,
            ignore: Vec::new(),
        })
    }

    fn to_json(layout: &Int4Layout) -> Value {
        json!({
            "quant_method": "gptq",
            "bits": 4,
            "group_size": layout.group,
            "desc_act": false,
            "sym": !layout.zero_points,
            "checkpoint_format": if layout.zeros_offset == 1 { "gptq" } else { "gptq_v2" },
            "damp_percent": 0.01,
            "static_groups": false,
            "true_sequential": true,
        })
    }
}
