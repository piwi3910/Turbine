use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use super::*;
use crate::types::{DeviceId, ExecutionBackend};

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

#[test]
fn execution_and_model_keys() {
    let c = parse(
        "model:\n  path: /m\n  served_name: meta-llama/Llama-3.2-3B-Instruct\n  max_seq_len: 4096\n\
         execution:\n  backend: cpu\n  device: 1\n  kernel_library: /opt/k/libturbine_hip.so\n",
        &[],
    )
    .expect("execution and model keys must load");
    assert_eq!(
        c.model.served_name.as_deref(),
        Some("meta-llama/Llama-3.2-3B-Instruct")
    );
    assert_eq!(c.model.max_seq_len, Some(4096));
    assert_eq!(c.execution.backend, ExecutionBackend::Cpu);
    assert_eq!(c.execution.device, DeviceId(1));
    assert_eq!(
        c.execution.kernel_library,
        Some(PathBuf::from("/opt/k/libturbine_hip.so"))
    );

    // Defaults: hip on device 0, no explicit shim, every new model key unset.
    let d = parse("model:\n  path: /m\n", &[]).unwrap();
    assert_eq!(d.execution.backend, ExecutionBackend::Hip);
    assert_eq!(d.execution.device, DeviceId(0));
    assert_eq!(d.execution.kernel_library, None);
    assert_eq!(d.model.served_name, None);
    assert_eq!(d.model.tokenizer, None);
    assert_eq!(d.model.chat_template, None);
    assert_eq!(d.model.max_seq_len, None);

    let base = "model:\n  path: /m\n";
    assert_rejected(base, &["execution.backend=cuda"], "execution.backend");
    let err = parse(base, &["execution.backend=cuda"]).unwrap_err();
    assert_eq!(err.key(), Some("execution.backend"));
    assert!(err.to_string().contains("phase-2b-nvidia"), "{err}");

    assert_rejected(base, &["model.served_name=\"\""], "model.served_name");
    let long = "x".repeat(257);
    assert_rejected(
        base,
        &[&format!("model.served_name={long}")],
        "model.served_name",
    );
    assert!(parse(base, &[&format!("model.served_name={}", "x".repeat(256))]).is_ok());
    assert_rejected(base, &["model.max_seq_len=0"], "model.max_seq_len");
    assert_rejected(base, &["execution.backend=rocm"], "execution.backend");
}

#[test]
fn phase2_keys() {
    use std::time::Duration;

    let base = "model:\n  path: /m\n";
    // Defaults equal the P2 §Configuration table (and C-8 for kv.gpu.max_bytes).
    let d = parse(base, &[]).unwrap();
    assert_eq!(
        d.server.request_timeout,
        HumanDuration(Duration::from_secs(600))
    );
    assert_eq!(
        d.server.slow_client_timeout,
        HumanDuration(Duration::from_secs(30))
    );
    assert_eq!(
        d.server.shutdown_grace,
        HumanDuration(Duration::from_secs(30))
    );
    assert_eq!(d.scheduler.max_running_requests, 64);
    assert_eq!(d.scheduler.max_batch_tokens, 8192);
    assert_eq!(d.scheduler.prefill_chunk_tokens, 2048);
    assert_eq!(d.scheduler.max_queued_requests, 256);
    assert_eq!(
        d.scheduler.queue_timeout,
        HumanDuration(Duration::from_secs(60))
    );
    assert_eq!(d.kv.gpu.max_bytes, Some(ByteSize::gib(8)));
    assert_eq!(d.model.tool_call_parser, None);
    assert_eq!(d.structured_output.max_schema_bytes, ByteSize::kib(64));
    assert_eq!(
        d.structured_output.compile_timeout,
        HumanDuration(Duration::from_secs(5))
    );
    assert_eq!(d.effective_max_running(), 64);

    // Durations: <integer><ms|s|m|h>, no space, exact case (C-14).
    for bad in ["1.5s", "\"2 s\"", "10d", "5", "10S", "s", "-1s"] {
        assert_rejected(
            base,
            &[&format!("scheduler.queue_timeout={bad}")],
            "scheduler.queue_timeout",
        );
    }
    let t = |v: &str| {
        parse(base, &[&format!("scheduler.queue_timeout={v}")])
            .unwrap()
            .scheduler
            .queue_timeout
            .0
    };
    assert_eq!(t("250ms"), Duration::from_millis(250));
    assert_eq!(t("1h"), Duration::from_secs(3600));
    assert_eq!(t("2m"), Duration::from_secs(120));
    assert_eq!(t("45s"), Duration::from_secs(45));
    assert_eq!("10m".parse::<HumanDuration>().unwrap().to_string(), "10m");
    assert_eq!(
        "1500ms".parse::<HumanDuration>().unwrap().to_string(),
        "1500ms"
    );
    assert_rejected(
        base,
        &["scheduler.queue_timeout=0s"],
        "scheduler.queue_timeout",
    );
    assert_rejected(
        base,
        &["server.request_timeout=0ms"],
        "server.request_timeout",
    );
    assert_rejected(
        base,
        &["server.slow_client_timeout=0s"],
        "server.slow_client_timeout",
    );
    assert!(parse(base, &["server.shutdown_grace=0s"]).is_ok());
    assert_rejected(
        base,
        &["structured_output.compile_timeout=0s"],
        "structured_output.compile_timeout",
    );

    // Scheduler bounds.
    assert_rejected(
        base,
        &[
            "scheduler.max_batch_tokens=32",
            "scheduler.max_running_requests=64",
        ],
        "scheduler.max_batch_tokens",
    );
    assert_rejected(
        base,
        &[
            "scheduler.max_batch_tokens=8",
            "scheduler.max_running_requests=4",
        ],
        "scheduler.max_batch_tokens",
    );
    assert_rejected(
        base,
        &["scheduler.max_running_requests=0"],
        "scheduler.max_running_requests",
    );
    assert_rejected(
        base,
        &["scheduler.max_running_requests=1025"],
        "scheduler.max_running_requests",
    );
    assert_rejected(
        base,
        &["scheduler.prefill_chunk_tokens=0"],
        "scheduler.prefill_chunk_tokens",
    );
    assert_rejected(
        base,
        &["scheduler.prefill_chunk_tokens=8193"],
        "scheduler.prefill_chunk_tokens",
    );
    assert_rejected(
        base,
        &["scheduler.max_queued_requests=0"],
        "scheduler.max_queued_requests",
    );
    assert_rejected(
        base,
        &["scheduler.max_queued_requests=65537"],
        "scheduler.max_queued_requests",
    );

    // Structured output bounds: 1KiB..=1MiB.
    assert_rejected(
        base,
        &["structured_output.max_schema_bytes=2MiB"],
        "structured_output.max_schema_bytes",
    );
    assert_rejected(
        base,
        &["structured_output.max_schema_bytes=512"],
        "structured_output.max_schema_bytes",
    );
    assert!(parse(base, &["structured_output.max_schema_bytes=1MiB"]).is_ok());

    // continuous_batching: false forces one running request.
    let single = parse(base, &["scheduler.continuous_batching=false"]).unwrap();
    assert_eq!(single.effective_max_running(), 1);

    let tools = parse(base, &["model.tool_call_parser=llama3_json"]).unwrap();
    assert_eq!(
        tools.model.tool_call_parser,
        Some(ToolCallParserKind::Llama3Json)
    );
    let none = parse(base, &["model.tool_call_parser=none"]).unwrap();
    assert_eq!(none.model.tool_call_parser, Some(ToolCallParserKind::None));
    assert_rejected(
        base,
        &["model.tool_call_parser=hermes"],
        "model.tool_call_parser",
    );

    // kv.gpu.max_bytes is optional (C-8).
    let unset = parse(base, &["kv.gpu.max_bytes=null"]).unwrap();
    assert_eq!(unset.kv.gpu.max_bytes, None);
    let four = parse(base, &["kv.gpu.max_bytes=4GiB"]).unwrap();
    assert_eq!(four.kv.gpu.max_bytes, Some(ByteSize::gib(4)));
}

#[test]
fn phase2c_execution_keys() {
    let base = "model:\n  path: /m\n";
    // Defaults: every optimised path on, four sampler threads (P2c §Configuration additions).
    let d = parse(base, &[]).unwrap();
    assert!(d.execution.gemm_autotune);
    assert!(d.execution.decode_graphs);
    assert!(d.execution.device_sampling);
    assert!(d.execution.fused_ops);
    assert_eq!(d.execution.sampler_threads, 4);
    assert!(!d.execution.overlap_scheduling);

    for bad in ["0", "65"] {
        assert_rejected(
            base,
            &[&format!("execution.sampler_threads={bad}")],
            "execution.sampler_threads",
        );
    }
    for ok in ["1", "64"] {
        assert!(parse(base, &[&format!("execution.sampler_threads={ok}")]).is_ok());
    }

    let c = parse(
        base,
        &[
            "execution.decode_graphs=false",
            "execution.gemm_autotune=false",
            "execution.device_sampling=false",
            "execution.fused_ops=false",
            "execution.overlap_scheduling=true",
        ],
    )
    .unwrap();
    assert!(c.execution.overlap_scheduling);
    assert!(!c.execution.decode_graphs);
    assert!(!c.execution.gemm_autotune);
    assert!(!c.execution.device_sampling);
    assert!(!c.execution.fused_ops);

    // min(sampler_threads, available parallelism − 1), at least 1.
    assert_eq!(d.execution.effective_sampler_threads(2), 1);
    assert_eq!(d.execution.effective_sampler_threads(16), 4);
    assert_eq!(d.execution.effective_sampler_threads(1), 1);
    assert_eq!(d.execution.effective_sampler_threads(0), 1);
    assert_eq!(d.execution.effective_sampler_threads(4), 3);

    // The shipped example spells the keys out at their defaults and still loads.
    let example = include_str!("../../../../examples/turbine.yaml");
    let e = parse(example, &[]).expect("examples/turbine.yaml must load");
    assert!(e.execution.gemm_autotune && e.execution.decode_graphs);
    assert!(e.execution.device_sampling && e.execution.fused_ops);
    assert_eq!(e.execution.sampler_threads, 4);
    assert!(!e.execution.overlap_scheduling);
    for key in [
        "gemm_autotune:",
        "decode_graphs:",
        "device_sampling:",
        "fused_ops:",
        "sampler_threads:",
        "overlap_scheduling:",
    ] {
        assert!(example.contains(key), "examples/turbine.yaml lacks {key}");
    }
}

/// The default KV page is 128 tokens, the page size CK's `fmha_fwd_pagedkv` requires, so paged
/// attention on HIP runs on CK unless a config asks for another size.
#[test]
fn default_block_tokens_is_128() {
    let base = "model:\n  path: /m\n";
    assert_eq!(KvConfig::default().block_tokens, 128);
    assert_eq!(parse(base, &[]).unwrap().kv.block_tokens, 128);
    for n in [16, 1024] {
        let c = parse(base, &[&format!("kv.block_tokens={n}")]).unwrap();
        assert_eq!(c.kv.block_tokens, n);
    }
    for n in [0, 1025] {
        assert_rejected(base, &[&format!("kv.block_tokens={n}")], "kv.block_tokens");
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for file in [
        "examples/turbine.yaml",
        "scripts/lab/phase2-novanas-llama.yaml",
        "scripts/lab/phase2-novanas-olmoe.yaml",
    ] {
        let c = load(&root.join(file), &[]).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(c.kv.block_tokens, 128, "{file}");
    }
}
