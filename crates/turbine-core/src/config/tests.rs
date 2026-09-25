use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use super::*;

/// TS §15 example, verbatim.
const TS15_EXAMPLE: &str = r#"server:
  listen: 0.0.0.0:8000
model:
  path: /models/qwen
  dtype: bf16
kv:
  block_tokens: 16
  gpu: { enabled: true }
  cpu:
    enabled: true
    max_bytes: 64GiB
  nvme:
    enabled: false
    path: /var/lib/turbine/kv
reliability:
  enabled: true
  emergency_vram_reserve: 2GiB
  adaptive_admission: true
scheduler:
  continuous_batching: true
  chunked_prefill: true
distributed:
  enabled: false
"#;

fn parse(yaml: &str, sets: &[&str]) -> Result<Config, ConfigError> {
    let overrides: Vec<Override> = sets.iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    load_from_str(yaml, Path::new("<test>"), &overrides)
}

fn assert_rejected(yaml: &str, sets: &[&str], key: &str) {
    match parse(yaml, sets) {
        Ok(cfg) => panic!("expected rejection naming {key}, got {cfg:?}"),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains(key), "error {msg:?} does not name {key}");
        }
    }
}

#[test]
fn example_config_loads() {
    let c = parse(TS15_EXAMPLE, &[]).expect("TS §15 example must load");
    assert_eq!(
        c.server.listen,
        "0.0.0.0:8000".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(c.server.max_request_bytes, ByteSize(8 * 1024 * 1024));
    assert_eq!(c.model.path, PathBuf::from("/models/qwen"));
    assert_eq!(c.model.dtype, ModelDtype::Bf16);
    assert_eq!(c.kv.block_tokens, 16);
    assert!(c.kv.gpu.enabled);
    assert!(c.kv.cpu.enabled);
    assert_eq!(c.kv.cpu.max_bytes, ByteSize(68_719_476_736));
    assert!(!c.kv.nvme.enabled);
    assert_eq!(c.kv.nvme.path, PathBuf::from("/var/lib/turbine/kv"));
    assert!(c.reliability.enabled);
    assert_eq!(
        c.reliability.emergency_vram_reserve,
        ByteSize(2 * 1024 * 1024 * 1024)
    );
    assert!(c.reliability.adaptive_admission);
    assert!(c.scheduler.continuous_batching);
    assert!(c.scheduler.chunked_prefill);
    assert!(!c.distributed.enabled);
    assert_eq!(c.logging.format, LogFormat::Text);
    assert_eq!(c.logging.level, "info");
    assert_eq!(c.devices.nvml_library, None);
    assert_eq!(c.devices.amd_smi_library, None);

    // Defaults alone (only the required key) equal the documented values too.
    let d = parse("model:\n  path: /m\n", &[]).unwrap();
    assert_eq!(d.server.listen, c.server.listen);
    assert_eq!(d.kv.cpu.max_bytes, c.kv.cpu.max_bytes);
    assert_eq!(
        d.reliability.emergency_vram_reserve,
        c.reliability.emergency_vram_reserve
    );
    assert_eq!(d.kv.nvme.path, c.kv.nvme.path);
}

#[test]
fn byte_size_parsing() {
    assert_eq!("64GiB".parse::<ByteSize>(), Ok(ByteSize(68_719_476_736)));
    assert_eq!("8MiB".parse::<ByteSize>(), Ok(ByteSize(8_388_608)));
    assert_eq!("1000".parse::<ByteSize>(), Ok(ByteSize(1000)));
    assert_eq!("2GB".parse::<ByteSize>(), Ok(ByteSize(2_000_000_000)));

    let with = |v: &str| format!("model:\n  path: /m\nkv:\n  cpu:\n    max_bytes: {v}\n");
    assert_eq!(
        parse(&with("64GiB"), &[]).unwrap().kv.cpu.max_bytes,
        ByteSize(68_719_476_736)
    );
    assert_eq!(
        parse(&with("8MiB"), &[]).unwrap().kv.cpu.max_bytes,
        ByteSize(8_388_608)
    );
    assert_eq!(
        parse(&with("1000"), &[]).unwrap().kv.cpu.max_bytes,
        ByteSize(1000)
    );
    assert_eq!(
        parse(&with("2GB"), &[]).unwrap().kv.cpu.max_bytes,
        ByteSize(2_000_000_000)
    );

    for bad in ["64 GiB", "64gib", "-1", "1.5GiB", "99999999999TiB"] {
        assert_rejected(&with(bad), &[], "kv.cpu.max_bytes");
    }
}

#[test]
fn impossible_configs_rejected() {
    let base = "model:\n  path: /m\n";
    assert_rejected(
        "model:\n  path: /m\nkv:\n  cpu:\n    max_byte: 1GiB\n",
        &[],
        "kv.cpu.max_byte",
    );
    assert_rejected("kv:\n  block_tokens: 16\n", &[], "model.path");
    assert_rejected("model:\n  path: /m\n  dtype: fp16\n", &[], "model.dtype");
    assert_rejected(
        &format!("{base}kv:\n  block_tokens: 0\n"),
        &[],
        "kv.block_tokens",
    );
    assert_rejected(
        &format!("{base}kv:\n  block_tokens: 1025\n"),
        &[],
        "kv.block_tokens",
    );
    assert_rejected(
        &format!("{base}kv:\n  cpu:\n    enabled: true\n    max_bytes: 0\n"),
        &[],
        "kv.cpu.max_bytes",
    );
    assert_rejected(
        &format!("{base}kv:\n  nvme:\n    enabled: true\n    path: relative/kv\n"),
        &[],
        "kv.nvme.path",
    );
    assert_rejected(
        &format!("{base}distributed:\n  enabled: true\n"),
        &[],
        "distributed.enabled",
    );
    let err = parse(&format!("{base}distributed:\n  enabled: true\n"), &[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("distributed mode is not supported in this build"));
    assert_rejected(
        &format!("{base}server:\n  max_request_bytes: 512\n"),
        &[],
        "server.max_request_bytes",
    );
}

#[test]
fn set_overrides_apply() {
    let c = parse(
        TS15_EXAMPLE,
        &["kv.cpu.enabled=false", "server.listen=127.0.0.1:9000"],
    )
    .unwrap();
    assert!(!c.kv.cpu.enabled);
    assert_eq!(
        c.server.listen,
        "127.0.0.1:9000".parse::<SocketAddr>().unwrap()
    );

    // Overrides are applied before validation: an invalid file value fixed by --set loads.
    let fixed = parse(
        "model:\n  path: /m\nkv:\n  block_tokens: 0\n",
        &["kv.block_tokens=32"],
    )
    .unwrap();
    assert_eq!(fixed.kv.block_tokens, 32);
    // ...and an override that breaks a valid file is rejected.
    assert_rejected(TS15_EXAMPLE, &["kv.block_tokens=0"], "kv.block_tokens");

    assert_rejected(TS15_EXAMPLE, &["nope=1"], "nope");
    assert_rejected(TS15_EXAMPLE, &["kv.block_tokens=abc"], "kv.block_tokens");
    assert!(matches!(
        "no-equals-sign".parse::<Override>(),
        Err(ConfigError::BadOverride { .. })
    ));
}
