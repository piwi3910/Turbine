//! AMD Quark MXFP4 checkpoints (`quark_mxfp4`, Phase 6a S-3, S-10): `quant_method: quark`, a
//! `global_quant_config` whose `weight` is `fp4` `per_group` 32 with `e8m0` scales, exported
//! `real_quantized`. Without `input_tensors` the column is `mxfp4` (weight-only); with `fp4`
//! dynamic `per_group` 32 input tensors (`round_method: half_even`, `scale_calculation_mode:
//! even`) it is `mxfp4_a4` and activations are quantize-dequantized to MXFP4 before each layer
//! (`ActivationQuant::Mxfp4Emulated`, user decision 2026-09-28, Q6). Per-layer overrides
//! (`layer_quant_config`, `layer_type_quant_config`) and output quantization are refused;
//! `exclude` names modules left in BF16 (globs). A `kv_cache_quant_config` is ignored with a WARN
//! (`kv_cache_quant_ignored`): the KV cache follows `kv.dtype`; so are the `layer_quant_config`
//! entries that repeat it for the K/V projections (same weight and input spec as the global one),
//! and their `output_scale` tensors load as unexpected. GPTQ calibration with `desc_act` needs
//! `static_groups` (the weights stay in order). Tensors: `X.weight` U8 `[n, k/2]`,
//! `X.weight_scale` U8 `[n, k/32]` ([`super::mxfp4`]).

use serde_json::{Value, json};

use super::ActivationQuant;
use super::common::{scheme_unsupported, string_list};
use super::mxfp4::{Mxfp4Format, Mxfp4Kind, Mxfp4Layout, Mxfp4Packaging, glob_ignore};
use crate::ModelError;
use crate::config::unsupported;

/// The Quark container.
pub struct QuarkMxfp4;

/// The registry entry.
pub static QUARK_MXFP4: Mxfp4Format<QuarkMxfp4> = Mxfp4Format::ENTRY;

fn text<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

/// An MX `fp4` per-group-32 `e8m0` spec (`weight` or `input_tensors`) with `dynamic` as given.
fn check_mx(field: &str, spec: &Value, dynamic: bool) -> Result<(), ModelError> {
    let ok = text(spec, "dtype") == "fp4"
        && text(spec, "qscheme") == "per_group"
        && spec.get("group_size").and_then(Value::as_u64) == Some(32)
        && text(spec, "scale_format") == "e8m0"
        && spec.get("is_dynamic").and_then(Value::as_bool) == Some(dynamic)
        && text(spec, "round_method") == "half_even"
        && text(spec, "scale_calculation_mode") == "even";
    if ok {
        Ok(())
    } else {
        Err(scheme_unsupported(
            field,
            spec.to_string(),
            &format!(
                "fp4 per_group 32, e8m0 scales, {} half_even rounding, even scales",
                if dynamic { "dynamic" } else { "static" }
            ),
        ))
    }
}

/// An empty (absent, null, `{}` or `[]`) value.
fn empty(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => true,
        Some(Value::Object(m)) => m.is_empty(),
        Some(Value::Array(a)) => a.is_empty(),
        _ => false,
    }
}

impl Mxfp4Packaging for QuarkMxfp4 {
    const NAME: &'static str = "quark_mxfp4";
    const DEFAULT: Mxfp4Layout = Mxfp4Layout {
        kind: Mxfp4Kind::Quark,
        act: ActivationQuant::Mxfp4Emulated,
        ignore: Vec::new(),
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        if text(q, "quant_method") != "quark" {
            return Err(unsupported(
                "quantization_config.quant_method",
                text(q, "quant_method"),
                "quark",
            ));
        }
        let dtype = text(&q["global_quant_config"]["weight"], "dtype");
        if dtype != "fp4" {
            return Err(unsupported(
                "global_quant_config.weight.dtype",
                dtype,
                "fp4",
            ));
        }
        Ok(())
    }

    fn parse(q: &Value) -> Result<Mxfp4Layout, ModelError> {
        let global = &q["global_quant_config"];
        check_mx("global_quant_config.weight", &global["weight"], false)?;
        let act = match global.get("input_tensors").filter(|v| !v.is_null()) {
            None => ActivationQuant::None,
            Some(input) => {
                check_mx("global_quant_config.input_tensors", input, true)?;
                ActivationQuant::Mxfp4Emulated
            }
        };
        for field in ["output_tensors", "bias"] {
            if !empty(global.get(field)) {
                return Err(scheme_unsupported(
                    &format!("global_quant_config.{field}"),
                    global[field].to_string(),
                    "null",
                ));
            }
        }
        if !empty(q.get("layer_type_quant_config")) {
            return Err(scheme_unsupported(
                "layer_type_quant_config",
                q["layer_type_quant_config"].to_string(),
                "empty",
            ));
        }
        // Quark exports its KV recipe twice: in `kv_cache_quant_config` and as per-layer
        // overrides of the K/V projections that differ from the global spec only in
        // `output_tensors`. Such an override is part of the ignored KV recipe; any other one
        // changes the layer's weight or activation scheme and is refused.
        if let Some(layers) = q.get("layer_quant_config").and_then(Value::as_object) {
            for (pattern, spec) in layers {
                let kv_mirror = q["kv_cache_quant_config"].get(pattern) == Some(spec)
                    && spec["weight"] == global["weight"]
                    && spec["input_tensors"] == global["input_tensors"]
                    && empty(spec.get("bias"));
                if !kv_mirror {
                    return Err(scheme_unsupported(
                        &format!("layer_quant_config.{pattern}"),
                        spec.to_string(),
                        "only the K/V projections' kv_cache_quant_config entries",
                    ));
                }
            }
        } else if !empty(q.get("layer_quant_config")) {
            return Err(scheme_unsupported(
                "layer_quant_config",
                q["layer_quant_config"].to_string(),
                "empty",
            ));
        }
        // A quantized KV cache in the checkpoint's recipe (e.g. fp4 K/V projection outputs) is
        // not applied: the KV stays at the configured `kv.dtype` (user decision 2026-09-29).
        if !empty(q.get("kv_cache_quant_config")) {
            // Once per process: the server parses config.json more than once while starting.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    event = "kv_cache_quant_ignored",
                    "the checkpoint's kv_cache_quant_config is ignored; the KV cache uses kv.dtype"
                );
            });
        }
        let export = &q["export"];
        if text(export, "weight_format") != "real_quantized" {
            return Err(scheme_unsupported(
                "export.weight_format",
                text(export, "weight_format"),
                "real_quantized",
            ));
        }
        if !matches!(text(export, "pack_method"), "reorder" | "order") {
            return Err(scheme_unsupported(
                "export.pack_method",
                text(export, "pack_method"),
                "reorder, order",
            ));
        }
        for algo in q
            .get("algo_config")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let desc_act = algo
                .get("desc_act")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let static_groups = algo
                .get("static_groups")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if desc_act && !static_groups {
                return Err(unsupported(
                    "algo_config.static_groups",
                    "false with desc_act",
                    "gptq_act_order: static_groups true",
                ));
            }
        }
        Ok(Mxfp4Layout {
            kind: Mxfp4Kind::Quark,
            act,
            ignore: string_list(q, "exclude")
                .iter()
                .map(|g| glob_ignore(g))
                .collect(),
        })
    }

    fn to_json(layout: &Mxfp4Layout) -> Value {
        let mx = |dynamic: bool| {
            json!({
                "dtype": "fp4", "qscheme": "per_group", "group_size": 32, "ch_axis": -1,
                "scale_format": "e8m0", "scale_type": "float", "is_dynamic": dynamic,
                "round_method": "half_even", "scale_calculation_mode": "even",
                "observer_cls": "PerBlockMXObserver", "symmetric": null,
            })
        };
        let input = if layout.act == ActivationQuant::Mxfp4Emulated {
            mx(true)
        } else {
            Value::Null
        };
        json!({
            "quant_method": "quark",
            "global_quant_config": {
                "weight": mx(false),
                "input_tensors": input,
                "output_tensors": null,
                "bias": null,
            },
            "exclude": layout.ignore,
            "export": {"weight_format": "real_quantized", "pack_method": "reorder",
                       "kv_cache_group": []},
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "kv_cache_quant_config": {},
        })
    }
}
