//! Pins the W4A4 8B accuracy bug (`.procoder/handoff/p6a-w4a4-numerics.md`): transformers-5
//! exports (e.g. `amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ` @ 00b0d018) spell the
//! RoPE settings only as `rope_parameters` (`rope_theta` moved inside it, no top-level
//! `rope_theta` / `rope_scaling`). `load_model_config` reads neither and silently falls back to
//! theta 10 000 with no llama3 scaling, so the model runs with the wrong rotary table.
//!
//! The fix (`config.rs` reads `rope_parameters`, no silent `rope_theta` default) landed with
//! `p6a-rope-parameters`; this test guards it.

use std::fs;
use std::path::{Path, PathBuf};

use turbine_model::{RopeScaling, load_model_config};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-3.2-3b-instruct")
}

/// The Llama-3.2-3B fixture config rewritten the transformers-5 way: `rope_theta` and
/// `rope_scaling` folded into one `rope_parameters` object.
fn transformers5_config_dir() -> PathBuf {
    let mut v: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_dir().join("config.json")).unwrap()).unwrap();
    let obj = v.as_object_mut().unwrap();
    let theta = obj.remove("rope_theta").unwrap();
    let mut params = obj.remove("rope_scaling").unwrap();
    params
        .as_object_mut()
        .unwrap()
        .insert("rope_theta".into(), theta);
    obj.insert("rope_parameters".into(), params);
    let dir = std::env::temp_dir().join(format!(
        "turbine-model-rope-parameters-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    dir
}

#[test]
fn rope_parameters_is_read_like_rope_theta_and_rope_scaling() {
    let cfg = load_model_config(&transformers5_config_dir()).unwrap();
    assert_eq!(cfg.rope_theta, 500_000.0, "rope_theta from rope_parameters");
    assert_eq!(
        cfg.rope_scaling,
        Some(RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        }),
        "llama3 scaling from rope_parameters"
    );
}
