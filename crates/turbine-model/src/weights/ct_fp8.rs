//! compressed-tensors FP8 checkpoints (`ct_fp8`, Phase 6a S-3): `quant_method:
//! compressed-tensors`, `format: float-quantized`, one config group whose `weights` are 8-bit
//! `float` with strategy `tensor` or `channel` (column `fp8`) or `block` with `block_structure`
//! `[128, 128]` (column `fp8_block`); `input_activations` null (weight-only), `tensor` static,
//! `token` dynamic or `group` 128 dynamic; the `ignore` list honoured (llm-compressor's
//! RedHatAI `*-FP8-dynamic` / `*-FP8` checkpoints). Tensors: `X.weight` (F8_E4M3),
//! `X.weight_scale` and, static, `X.input_scale` ([`super::fp8`]).

use serde_json::{Value, json};

use super::ActivationQuant;
use super::common::{pair, scheme_unsupported, string_list};
use super::fp8::{Fp8Format, Fp8Layout, Fp8Packaging, Fp8Weights, check_block};
use crate::ModelError;
use crate::config::unsupported;

/// The compressed-tensors FP8 container.
pub struct CtFp8;

/// The registry entry.
pub static CT_FP8: Fp8Format<CtFp8> = Fp8Format::ENTRY;

const SCALE: &str = "weight_scale";

/// `obj[key]` as a string, `""` when absent.
fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The one config group's `weights` and `input_activations`.
fn group(q: &Value) -> Result<&Value, ModelError> {
    let groups = q
        .get("config_groups")
        .and_then(Value::as_object)
        .filter(|g| !g.is_empty())
        .ok_or_else(|| scheme_unsupported("config_groups", "none", "one group"))?;
    if groups.len() != 1 {
        return Err(scheme_unsupported(
            "config_groups",
            format!("{} groups", groups.len()),
            "one group",
        ));
    }
    Ok(groups.values().next().expect("one group"))
}

fn weights_of(w: &Value) -> Result<Fp8Weights, ModelError> {
    let bits = w.get("num_bits").and_then(Value::as_u64).unwrap_or(0);
    if bits != 8 || text(w, "type") != "float" {
        return Err(scheme_unsupported(
            "weights",
            format!("{bits}-bit {}", text(w, "type")),
            "8-bit float",
        ));
    }
    match text(w, "strategy") {
        "tensor" => Ok(Fp8Weights::Tensor),
        "channel" => Ok(Fp8Weights::Channel),
        "block" => {
            let block = w.get("block_structure").and_then(pair).ok_or_else(|| {
                scheme_unsupported("weights.block_structure", "none", "[128, 128]")
            })?;
            check_block("weights.block_structure", block)
        }
        other => Err(scheme_unsupported(
            "weights.strategy",
            other,
            "tensor, channel, block",
        )),
    }
}

fn activations_of(a: Option<&Value>) -> Result<ActivationQuant, ModelError> {
    let Some(a) = a.filter(|a| !a.is_null()) else {
        return Ok(ActivationQuant::None);
    };
    let bits = a.get("num_bits").and_then(Value::as_u64).unwrap_or(0);
    let dynamic = a.get("dynamic").and_then(Value::as_bool).unwrap_or(false);
    let group = a.get("group_size").and_then(Value::as_u64);
    match (bits, text(a, "type"), text(a, "strategy"), dynamic, group) {
        (8, "float", "tensor", false, _) => Ok(ActivationQuant::Fp8PerTensorStatic),
        (8, "float", "token", true, _) => Ok(ActivationQuant::Fp8PerTokenDynamic),
        (8, "float", "group", true, Some(128)) => {
            Ok(ActivationQuant::Fp8PerGroupDynamic { group: 128 })
        }
        (bits, ty, strategy, dynamic, group) => Err(scheme_unsupported(
            "input_activations",
            format!(
                "{bits}-bit {ty} {strategy} {} group {group:?}",
                if dynamic { "dynamic" } else { "static" }
            ),
            "8-bit float: tensor static, token dynamic, group 128 dynamic",
        )),
    }
}

/// `kv_cache_scheme`: null, or static FP8 per tensor (its scales are read by `kv.dtype`).
fn check_kv_scheme(q: &Value) -> Result<(), ModelError> {
    match q.get("kv_cache_scheme").filter(|k| !k.is_null()) {
        None => Ok(()),
        Some(k)
            if k.get("num_bits").and_then(Value::as_u64) == Some(8)
                && text(k, "type") == "float"
                && text(k, "strategy") == "tensor"
                && !k.get("dynamic").and_then(Value::as_bool).unwrap_or(false) =>
        {
            Ok(())
        }
        Some(k) => Err(scheme_unsupported(
            "kv_cache_scheme",
            k.to_string(),
            "null or static 8-bit float per tensor",
        )),
    }
}

impl Fp8Packaging for CtFp8 {
    const NAME: &'static str = "ct_fp8";
    const DEFAULT: Fp8Layout = Fp8Layout {
        weights: Fp8Weights::Channel,
        act: ActivationQuant::Fp8PerTokenDynamic,
        ignore: Vec::new(),
        scale_suffix: SCALE,
        decoded: Vec::new(),
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        if text(q, "quant_method") != "compressed-tensors" {
            return Err(unsupported(
                "quantization_config.quant_method",
                text(q, "quant_method"),
                "compressed-tensors",
            ));
        }
        if text(q, "format") != "float-quantized" {
            return Err(unsupported(
                "quantization_config.format",
                text(q, "format"),
                "float-quantized",
            ));
        }
        Ok(())
    }

    fn parse(q: &Value) -> Result<Fp8Layout, ModelError> {
        let g = group(q)?;
        let targets = string_list(g, "targets");
        if targets != ["Linear"] {
            return Err(scheme_unsupported(
                "targets",
                format!("{targets:?}"),
                "[\"Linear\"]",
            ));
        }
        let weights = g
            .get("weights")
            .ok_or_else(|| scheme_unsupported("weights", "none", "8-bit float"))?;
        check_kv_scheme(q)?;
        Ok(Fp8Layout {
            weights: weights_of(weights)?,
            act: activations_of(g.get("input_activations"))?,
            ignore: string_list(q, "ignore"),
            scale_suffix: SCALE,
            decoded: Vec::new(),
        })
    }

    fn to_json(layout: &Fp8Layout) -> Value {
        let (strategy, block) = match layout.weights {
            Fp8Weights::Tensor => ("tensor", Value::Null),
            Fp8Weights::Channel => ("channel", Value::Null),
            Fp8Weights::Block { n, k } => ("block", json!([n, k])),
        };
        let act = |strategy: &str, dynamic: bool, group: Value| {
            json!({
                "num_bits": 8, "type": "float", "symmetric": true, "strategy": strategy,
                "dynamic": dynamic, "group_size": group, "block_structure": null,
                "actorder": null, "observer": null, "observer_kwargs": {},
            })
        };
        let input = match layout.act {
            ActivationQuant::Fp8PerTensorStatic => act("tensor", false, Value::Null),
            ActivationQuant::Fp8PerTokenDynamic => act("token", true, Value::Null),
            ActivationQuant::Fp8PerGroupDynamic { group } => act("group", true, json!(group)),
            _ => Value::Null,
        };
        json!({
            "config_groups": {
                "group_0": {
                    "input_activations": input,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8, "type": "float", "symmetric": true, "strategy": strategy,
                        "dynamic": false, "group_size": null, "block_structure": block,
                        "actorder": null, "observer": "minmax", "observer_kwargs": {},
                    },
                },
            },
            "format": "float-quantized",
            "ignore": layout.ignore,
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        })
    }
}
