//! Writes one tiny quantized checkpoint per packaging variant, plus its exact BF16 twin, into
//! `<out>/<case>/{quantized,twin}`. It is the fixture set for
//! `scripts/golden/dequantize_checkpoint.py --check-tiny` (Phase 6a Task 11), which decodes each
//! `quantized/` and must equal `twin/` bit for bit. The configurations are those of
//! `tests/tiny_model.rs` `quantized_matches_dequantized_bf16`.
//!
//! usage: cargo run -p turbine-model --example dump_dequant -- <out-dir>

use serde_json::{Value, json};
use turbine_model::testing::tiny::write_tiny_quantized;

const SEED: u64 = 6;

fn cases() -> Vec<(&'static str, Value, u32)> {
    let ct = |strategy: &str, block: Value, input: Value| {
        json!({
            "config_groups": {"group_0": {
                "input_activations": input,
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "type": "float", "strategy": strategy,
                            "dynamic": false, "symmetric": true, "block_structure": block},
            }},
            "format": "float-quantized",
            "ignore": ["lm_head"],
            "quant_method": "compressed-tensors",
        })
    };
    let act = |strategy: &str, dynamic: bool, group: Value| {
        json!({"num_bits": 8, "type": "float", "strategy": strategy,
               "dynamic": dynamic, "group_size": group})
    };
    let quark = |a4: bool| {
        let mx = |dynamic: bool| {
            json!({"dtype": "fp4", "qscheme": "per_group", "group_size": 32,
                   "scale_format": "e8m0", "is_dynamic": dynamic,
                   "round_method": "half_even", "scale_calculation_mode": "even"})
        };
        json!({
            "quant_method": "quark",
            "global_quant_config": {"weight": mx(false),
                                    "input_tensors": if a4 { mx(true) } else { Value::Null }},
            "exclude": ["lm_head"],
            "export": {"weight_format": "real_quantized", "pack_method": "reorder"},
        })
    };
    let null = Value::Null;
    vec![
        (
            "ct_fp8-tensor-static",
            ct("tensor", null.clone(), act("tensor", false, null.clone())),
            64,
        ),
        (
            "ct_fp8-channel-token",
            ct("channel", null.clone(), act("token", true, null.clone())),
            64,
        ),
        (
            "ct_fp8-channel-weight-only",
            ct("channel", null.clone(), null.clone()),
            64,
        ),
        (
            "ct_fp8-block-group128",
            ct("block", json!([128, 128]), act("group", true, json!(128))),
            128,
        ),
        (
            "hf_fp8-block-dynamic",
            json!({"quant_method": "fp8", "activation_scheme": "dynamic",
                   "weight_block_size": [128, 128]}),
            128,
        ),
        (
            "hf_fp8-tensor-static",
            json!({"quant_method": "fp8", "activation_scheme": "static"}),
            64,
        ),
        (
            "awq",
            json!({"quant_method": "awq", "bits": 4, "group_size": 128,
                   "zero_point": true, "version": "gemm"}),
            128,
        ),
        (
            "gptq-sym",
            json!({"quant_method": "gptq", "bits": 4, "group_size": 128,
                   "desc_act": false, "sym": true}),
            128,
        ),
        (
            "gptq-asym-v1-g64",
            json!({"quant_method": "gptq", "bits": 4, "group_size": 64,
                   "desc_act": false, "sym": false}),
            128,
        ),
        (
            "ct_pack_int4",
            json!({"quant_method": "compressed-tensors", "format": "pack-quantized",
                   "ignore": ["lm_head"],
                   "config_groups": {"group_0": {"targets": ["Linear"],
                       "weights": {"num_bits": 4, "type": "int", "symmetric": true,
                                   "strategy": "group", "group_size": 128}}}}),
            128,
        ),
        (
            "ct_mxfp4",
            json!({"quant_method": "compressed-tensors",
                   "format": "mxfp4-pack-quantized", "ignore": ["lm_head"],
                   "config_groups": {"group_0": {"targets": ["Linear"],
                       "weights": {"num_bits": 4, "type": "float",
                                   "strategy": "group", "group_size": 32}}}}),
            128,
        ),
        ("quark_mxfp4-w4a16", quark(false), 128),
        ("quark_mxfp4-w4a4", quark(true), 128),
        (
            "openai_mxfp4",
            json!({"quant_method": "mxfp4", "modules_to_not_convert": ["lm_head"]}),
            128,
        ),
    ]
}

fn main() {
    let Some(out) = std::env::args_os().nth(1) else {
        eprintln!("usage: dump_dequant <out-dir>");
        std::process::exit(2);
    };
    let out = std::path::PathBuf::from(out);
    for (name, config, hidden) in cases() {
        let dir = out.join(name);
        let _ = std::fs::remove_dir_all(&dir);
        write_tiny_quantized(&dir, SEED, &config, hidden, 128);
        println!("{name}: {}", dir.display());
    }
}
