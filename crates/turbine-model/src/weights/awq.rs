//! AutoAWQ GEMM checkpoints (`awq`, Phase 6a S-3, column `awq_int4`): `quant_method: awq`,
//! `bits: 4`, `group_size` 32 / 64 / 128, `zero_point: true`, `version: gemm`;
//! `modules_to_not_convert` names modules left unquantized (a substring of the module name, as
//! AutoAWQ and vLLM match it). Tensors and repack: [`super::int4`].

use serde_json::{Value, json};

use super::common::{scheme_unsupported, string_list};
use super::int4::{Int4Format, Int4Kind, Int4Layout, Int4Packaging, check_group};
use crate::ModelError;
use crate::config::unsupported;
use turbine_core::support::WeightFormatColumn;

/// The AutoAWQ container.
pub struct Awq;

/// The registry entry.
pub static AWQ: Int4Format<Awq> = Int4Format::ENTRY;

fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

impl Int4Packaging for Awq {
    const NAME: &'static str = "awq";
    const COLUMN: WeightFormatColumn = WeightFormatColumn::AwqInt4;
    const DEFAULT: Int4Layout = Int4Layout {
        kind: Int4Kind::Awq,
        group: 32,
        zero_points: true,
        zeros_offset: 0,
        ignore: Vec::new(),
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        if text(q, "quant_method") == "awq" {
            Ok(())
        } else {
            Err(unsupported(
                "quantization_config.quant_method",
                text(q, "quant_method"),
                "awq",
            ))
        }
    }

    fn parse(q: &Value) -> Result<Int4Layout, ModelError> {
        let bits = q
            .get("bits")
            .or_else(|| q.get("w_bit"))
            .and_then(Value::as_u64);
        if bits != Some(4) {
            return Err(scheme_unsupported("bits", format!("{bits:?}"), "4"));
        }
        let group = check_group(
            "group_size",
            q.get("group_size")
                .or_else(|| q.get("q_group_size"))
                .and_then(Value::as_u64),
        )?;
        if q.get("zero_point").and_then(Value::as_bool) != Some(true) {
            return Err(scheme_unsupported(
                "zero_point",
                q.get("zero_point").map_or("none".into(), Value::to_string),
                "true",
            ));
        }
        let version = text(q, "version").to_ascii_lowercase();
        if version != "gemm" {
            return Err(scheme_unsupported("version", version, "gemm"));
        }
        Ok(Int4Layout {
            kind: Int4Kind::Awq,
            group,
            zero_points: true,
            zeros_offset: 0,
            // A substring of the module name, as AutoAWQ matches it.
            ignore: string_list(q, "modules_to_not_convert")
                .into_iter()
                .map(|m| format!("re:.*{m}"))
                .collect(),
        })
    }

    fn to_json(layout: &Int4Layout) -> Value {
        let ignore: Vec<&str> = layout
            .ignore
            .iter()
            .map(|p| p.strip_prefix("re:.*").unwrap_or(p))
            .collect();
        json!({
            "quant_method": "awq",
            "bits": 4,
            "group_size": layout.group,
            "zero_point": true,
            "version": "gemm",
            "modules_to_not_convert": if ignore.is_empty() { Value::Null } else { json!(ignore) },
        })
    }
}
