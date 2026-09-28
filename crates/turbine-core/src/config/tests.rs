use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use super::*;
use crate::types::DeviceId;

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
    assert_eq!(c.execution.backend.as_str(), "cpu");
    assert_eq!(c.execution.device, DeviceId(1));
    assert_eq!(
        c.execution.kernel_library,
        Some(PathBuf::from("/opt/k/libturbine_hip.so"))
    );

    // Defaults: hip on device 0, no explicit shim, every new model key unset.
    let d = parse("model:\n  path: /m\n", &[]).unwrap();
    assert_eq!(d.execution.backend.as_str(), "hip");
    assert_eq!(d.execution.device, DeviceId(0));
    assert_eq!(d.execution.kernel_library, None);
    assert_eq!(d.model.served_name, None);
    assert_eq!(d.model.tokenizer, None);
    assert_eq!(d.model.chat_template, None);
    assert_eq!(d.model.max_seq_len, None);

    let base = "model:\n  path: /m\n";
    assert_rejected(base, &["model.served_name=\"\""], "model.served_name");
    let long = "x".repeat(257);
    assert_rejected(
        base,
        &[&format!("model.served_name={long}")],
        "model.served_name",
    );
    assert!(parse(base, &[&format!("model.served_name={}", "x".repeat(256))]).is_ok());
    assert_rejected(base, &["model.max_seq_len=0"], "model.max_seq_len");
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
    // Removed in Phase 3 (C-1): reliability.admission.queue_timeout bounds queue wait.
    assert!(d.scheduler.queue_timeout.is_none());
    // Phase 3 (C-8): null, the kv pool is the remainder of the memory budget.
    assert_eq!(d.kv.gpu.max_bytes, None);
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
            &[&format!("server.request_timeout={bad}")],
            "server.request_timeout",
        );
    }
    let t = |v: &str| {
        parse(base, &[&format!("server.request_timeout={v}")])
            .unwrap()
            .server
            .request_timeout
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
        tools
            .model
            .tool_call_parser
            .as_ref()
            .map(ModuleName::as_str),
        Some("llama3_json")
    );
    let none = parse(base, &["model.tool_call_parser=none"]).unwrap();
    assert_eq!(
        none.model.tool_call_parser.as_ref().map(ModuleName::as_str),
        Some("none")
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
        "scripts/lab/phase2c-novanas-llama.yaml",
        "scripts/lab/phase2c-novanas-olmoe.yaml",
        "scripts/lab/phase4-novanas.yaml",
    ] {
        let c = load(&root.join(file), &[]).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(c.kv.block_tokens, 128, "{file}");
    }
}

/// Phase 2m: the module keys take any well-formed name; which names exist is the registries'
/// business (`Config::validate_modules`, run by the server before any port is bound).
#[test]
fn module_names_are_open() {
    let base = "model:\n  path: /m\n";
    let c = parse(
        base,
        &[
            "model.tool_call_parser=hermes",
            "execution.backend=cuda",
            "scheduler.policy=fifo",
            "execution.card_profile=gfx942",
        ],
    )
    .expect("well-formed module names parse");
    assert_eq!(
        c.model.tool_call_parser.as_ref().map(ModuleName::as_str),
        Some("hermes")
    );
    assert_eq!(c.execution.backend.as_str(), "cuda");
    assert_eq!(c.scheduler.policy.as_str(), "fifo");
    assert_eq!(c.execution.card_profile.as_str(), "gfx942");

    assert_rejected(base, &["execution.backend=Hip!"], "execution.backend");
    assert_rejected(base, &["scheduler.policy=\"\""], "scheduler.policy");
    assert_rejected(
        base,
        &[&format!("execution.card_profile={}", "x".repeat(65))],
        "execution.card_profile",
    );

    let d = parse(base, &[]).unwrap();
    assert_eq!(d.execution.backend.as_str(), "hip");
    assert_eq!(d.execution.card_profile.as_str(), "auto");
    assert_eq!(d.scheduler.policy.as_str(), "default");
    assert_eq!(d.model.tool_call_parser, None);

    assert!(ModuleName::new("llama3_json").is_ok());
    assert!(ModuleName::new("Hip").is_err());
    assert!(ModuleName::new("").is_err());
    assert_eq!(
        serde_norway::to_value(ModuleName::new("hip").unwrap()).unwrap(),
        serde_norway::Value::String("hip".into())
    );

    // The shipped example spells the new keys out at their defaults.
    let example = include_str!("../../../../examples/turbine.yaml");
    assert!(example.contains("  policy: default\n"));
    assert!(example.contains("  card_profile: auto\n"));

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = vec![root.join("examples/turbine.yaml")];
    for entry in std::fs::read_dir(root.join("scripts/lab")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // The k3s Job templates in the same directory are not Turbine configurations.
        if name.ends_with(".yaml") && !name.ends_with("-job.yaml") {
            files.push(path);
        }
    }
    assert!(files.len() > 1, "{files:?}");
    for file in files {
        load(&file, &[]).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    }
}

/// The names the server's registries hold in Phase 2m Task 1.
fn task1_modules() -> ModuleNames<'static> {
    ModuleNames {
        tool_formats: &["llama3_json"],
        backends: &["cpu", "hip"],
        card_profiles: &["gfx1201"],
        scheduling_policies: &["default"],
        eviction_policies: &["cost_aware", "lru"],
        collective_backends: &["host", "nccl", "rccl"],
        rank_transports: &["tcp"],
    }
}

#[test]
fn validate_modules_names_registries() {
    let base = "model:\n  path: /m\n";
    let known = task1_modules();
    for (set, key, registered) in [
        (
            "model.tool_call_parser=hermes",
            "model.tool_call_parser",
            "llama3_json",
        ),
        ("execution.backend=cuda", "execution.backend", "cpu, hip"),
        ("scheduler.policy=fifo", "scheduler.policy", "default"),
        ("kv.policy=lfu", "kv.policy", "cost_aware, lru"),
        (
            "execution.card_profile=gfx942",
            "execution.card_profile",
            "gfx1201",
        ),
    ] {
        let c = parse(base, &[set]).unwrap();
        let err = c.validate_modules(&known).unwrap_err();
        assert_eq!(err.key(), Some(key), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("is not registered"), "{msg}");
        assert!(
            msg.contains(&format!("(registered: {registered})")),
            "{msg}"
        );
    }
    for set in [
        "model.tool_call_parser=none",
        "model.tool_call_parser=llama3_json",
        "execution.backend=cpu",
        "execution.backend=hip",
        "execution.card_profile=auto",
        "execution.card_profile=gfx1201",
        "scheduler.policy=default",
    ] {
        let c = parse(base, &[set]).unwrap();
        c.validate_modules(&known)
            .unwrap_or_else(|e| panic!("{set}: {e}"));
    }
    parse(base, &[]).unwrap().validate_modules(&known).unwrap();
}

/// The dotted key `load` names when rejecting `yaml`.
fn rejected_key(yaml: &str) -> String {
    match parse(yaml, &[]) {
        Ok(cfg) => panic!("expected {yaml:?} to be rejected, got {cfg:?}"),
        Err(e) => e
            .key()
            .unwrap_or_else(|| panic!("error {e} names no key"))
            .to_string(),
    }
}

#[test]
fn reliability_config_validation() {
    use crate::pressure::PressureSignal;

    // Defaults: the P3 configuration table.
    let r = ReliabilityConfig::default();
    assert!(r.enabled && r.adaptive_admission);
    assert_eq!(r.emergency_vram_reserve, ByteSize::gib(2));
    assert_eq!(r.memory.workspace_bytes, ByteSize::gib(1));
    assert_eq!(r.memory.runtime_overhead_bytes, ByteSize::gib(1));
    assert_eq!(r.memory.device_budget_bytes, None);
    assert_eq!(r.memory.host_reserve_bytes, ByteSize::gib(8));
    assert_eq!(r.telemetry.interval, HumanDuration::from_millis(100));
    assert_eq!(r.telemetry.vendor_interval, HumanDuration::from_secs(1));
    assert_eq!(r.telemetry.call_timeout, HumanDuration::from_millis(500));
    assert_eq!(r.telemetry.stale_after, HumanDuration::from_secs(5));
    assert_eq!(r.pressure.escalate_samples, 2);
    assert_eq!(r.pressure.deescalate_dwell, HumanDuration::from_secs(10));
    assert_eq!(r.pressure.exit_margin, 0.05);
    assert!(r.pressure.thresholds.is_empty(), "overrides only");
    assert_eq!(r.admission.max_queue, 256);
    assert_eq!(r.admission.queue_timeout, HumanDuration::from_secs(30));
    assert_eq!(r.admission.large_prefill_tokens, 2048);
    assert_eq!(r.admission.max_bypass, 8);
    assert_eq!(r.recovery.max_retries, 3);
    assert_eq!(r.recovery.backoff, HumanDuration::from_millis(50));
    assert_eq!(
        r.recovery.survival_liveness,
        SurvivalLiveness::RequeueUnstarted
    );
    let c = &r.circuit;
    assert_eq!(
        (
            c.oom_recoveries_to_open,
            c.window,
            c.latency_drift_degraded,
            c.latency_drift_open,
            c.cooldown,
            c.drain_timeout,
            c.probe_successes
        ),
        (
            3,
            HumanDuration::from_secs(60),
            2.0,
            4.0,
            HumanDuration::from_secs(30),
            HumanDuration::from_secs(120),
            3
        )
    );
    let base = "model:\n  path: /m\n";
    parse(base, &[]).expect("defaults are valid");

    // Durations in several units and a threshold override with an unused level parse.
    let ok = parse(
        &format!(
            "{base}reliability:\n  telemetry:\n    interval: 100ms\n    vendor_interval: 10s\n    stale_after: 1h\n  admission:\n    queue_timeout: 1h\n  pressure:\n    thresholds:\n      kv_utilization: [0.6, 0.8, 0.9, null]\n      host_available: [8.0, 4.0, 2.0, 1.0]\n"
        ),
        &[],
    )
    .expect("valid overrides accepted");
    let rel = &ok.reliability;
    let b = parse(
        &format!("{base}reliability:\n  recovery:\n    survival_liveness: continue_prefills\n"),
        &[],
    )
    .expect("option B is a valid choice");
    assert_eq!(
        b.reliability.recovery.survival_liveness,
        SurvivalLiveness::ContinuePrefills
    );
    assert_eq!(
        rejected_key(&format!(
            "{base}reliability:\n  recovery:\n    survival_liveness: sometimes\n"
        )),
        "reliability.recovery.survival_liveness"
    );
    assert_eq!(rel.telemetry.vendor_interval, HumanDuration::from_secs(10));
    assert_eq!(rel.telemetry.stale_after, HumanDuration::from_secs(3600));
    assert_eq!(rel.admission.queue_timeout, HumanDuration::from_secs(3600));
    assert_eq!(
        rel.pressure.thresholds[&PressureSignal::KvUtilization],
        [Some(0.6), Some(0.8), Some(0.9), None]
    );
    assert_eq!(rel.pressure.thresholds.len(), 2);

    // Every impossible configuration is rejected naming its full dotted key.
    let cases = [
        (
            "reliability:\n  pressure:\n    thresholds:\n      kv_utilization: [0.9, 0.8, 0.95, 0.97]\n",
            "reliability.pressure.thresholds.kv_utilization",
        ),
        (
            "reliability:\n  pressure:\n    thresholds:\n      host_available: [1.0, 2.0, 4.0, 8.0]\n",
            "reliability.pressure.thresholds.host_available",
        ),
        (
            "reliability:\n  circuit:\n    latency_drift_open: 1.5\n    latency_drift_degraded: 2.0\n",
            "reliability.circuit.latency_drift_open",
        ),
        (
            "reliability:\n  pressure:\n    deescalate_dwell: 10ms\n",
            "reliability.pressure.deescalate_dwell",
        ),
        (
            "reliability:\n  telemetry:\n    vendor_interval: 50ms\n    interval: 100ms\n",
            "reliability.telemetry.vendor_interval",
        ),
        (
            "reliability:\n  telemetry:\n    stale_after: 1s\n    vendor_interval: 1s\n",
            "reliability.telemetry.stale_after",
        ),
        (
            "reliability:\n  admission:\n    max_queue: 0\n",
            "reliability.admission.max_queue",
        ),
        (
            "reliability:\n  admission:\n    kv_overcommit: 1.5\n",
            "reliability.admission.kv_overcommit",
        ),
        (
            "scheduler:\n  queue_timeout: 60s\n",
            "scheduler.queue_timeout",
        ),
        (
            "reliability:\n  telemetry:\n    interval: 1.5s\n",
            "reliability.telemetry.interval",
        ),
        (
            "reliability:\n  pressure:\n    thresholds:\n      no_such_signal: [1, 2, 3, 4]\n",
            "reliability.pressure.thresholds.no_such_signal",
        ),
    ];
    for (yaml, key) in cases {
        assert_eq!(rejected_key(&format!("{base}{yaml}")), key, "for {yaml:?}");
    }
    let fault = format!("{base}reliability:\n  fault_injection:\n    alloc_fail_every: 5\n");
    #[cfg(not(feature = "fault-injection"))]
    assert_eq!(rejected_key(&fault), "reliability.fault_injection");
    #[cfg(feature = "fault-injection")]
    {
        let cfg = parse(&fault, &[]).expect("the fault-injection build accepts the section");
        let fi = cfg.reliability.fault_injection.expect("section present");
        assert_eq!(fi.alloc_fail_every, Some(5));
    }
}

#[test]
fn kv_config_validation() {
    const GIB: u64 = 1 << 30;
    // Defaults equal the P4 §Configuration table (kv.gpu.max_bytes null from Phase 3, C-8).
    let d = KvConfig::default();
    assert!(d.gpu.enabled);
    assert_eq!(d.gpu.max_bytes, None);
    assert!(d.cpu.enabled);
    assert_eq!(d.cpu.max_bytes, ByteSize::gib(64));
    assert!(!d.nvme.enabled);
    assert_eq!(d.nvme.path, PathBuf::from("/var/lib/turbine/kv"));
    assert_eq!(d.nvme.max_bytes, ByteSize::gib(64));
    assert_eq!(d.nvme.slab_bytes, ByteSize::gib(1));
    assert_eq!((d.nvme.max_queue_depth, d.nvme.io_threads), (64, 4));
    assert_eq!(d.policy.as_str(), "cost_aware");
    assert_eq!(d.demote_min_value, 0.0);
    assert!(d.prefix_sharing);
    assert_eq!(d.transfer.max_inflight_bytes, ByteSize::gib(1));
    assert_eq!(d.session.max_sessions, 10_000);
    assert_eq!(d.session.hot_ttl, HumanDuration::from_secs(60));
    assert_eq!(d.session.warm_ttl, HumanDuration::from_secs(600));
    assert_eq!(d.session.max_idle, HumanDuration::from_secs(3600));
    assert_eq!(d.prefetch.lead_time, HumanDuration::from_secs(2));
    assert_eq!(d.prefetch.max_queue, 256);
    assert_eq!(d.policy_weights.session_active, 0.5);
    assert_eq!(d.policy_weights.hit_half_life, HumanDuration::from_secs(60));

    let base = "model:\n  path: /m\n";
    let lru = parse(&format!("{base}kv:\n  policy: lru\n"), &[]).expect("lru is valid");
    assert_eq!(lru.kv.policy.as_str(), "lru");
    // kv.policy names an `eviction_policy` registry module: an unregistered name is refused
    // naming the key by Config::validate_modules (before any port is bound).
    let lfu = parse(&format!("{base}kv:\n  policy: lfu\n"), &[]).expect("well-formed name");
    let err = lfu.validate_modules(&task1_modules()).unwrap_err();
    assert_eq!(err.key(), Some("kv.policy"), "{err}");
    for (yaml, key) in [
        ("kv:\n  gpu:\n    enabled: false\n", "kv.gpu.enabled"),
        ("kv:\n  policy: LFU\n", "kv.policy"),
        ("kv:\n  nvme:\n    max_bytes: 0\n", "kv.nvme.max_bytes"),
        (
            "kv:\n  session:\n    hot_ttl: 60s\n    warm_ttl: 30s\n",
            "kv.session.warm_ttl",
        ),
        (
            "kv:\n  nvme:\n    enabled: true\n    path: rel/kv\n",
            "kv.nvme.path",
        ),
        (
            "kv:\n  transfer:\n    max_inflight_bytes: 1\n",
            "kv.transfer.max_inflight_bytes",
        ),
        ("kv:\n  nvme:\n    slab_bytes: 5000\n", "kv.nvme.slab_bytes"),
        (
            "kv:\n  nvme:\n    max_queue_depth: 0\n",
            "kv.nvme.max_queue_depth",
        ),
        ("kv:\n  nvme:\n    io_threads: 65\n", "kv.nvme.io_threads"),
        (
            "kv:\n  session:\n    max_sessions: 0\n",
            "kv.session.max_sessions",
        ),
        (
            "kv:\n  policy_weights:\n    session_active: 1.5\n",
            "kv.policy_weights.session_active",
        ),
        ("kv:\n  demote_min_value: -1.0\n", "kv.demote_min_value"),
    ] {
        assert_rejected(&format!("{base}{yaml}"), &[], key);
    }

    // Host facts: 32 GiB of RAM with an 8 GiB reserve cannot hold a 64 GiB L1; 50 GiB of free
    // disk cannot hold a 64 GiB L2 (free space minus 10 %), and the error names the free space.
    let small_host = HostFacts {
        mem_total_bytes: Some(32 * GIB),
        disk_free_bytes: Some(50 * GIB),
    };
    let err = KvConfig::default()
        .validate_host(&small_host, ByteSize::gib(8))
        .unwrap_err();
    assert_eq!(err.key(), Some("kv.cpu.max_bytes"));
    // Config::validate_host passes reliability.memory.host_reserve_bytes (contract §3.2).
    let cfg = parse(base, &[]).expect("valid");
    let err = cfg.validate_host(&small_host).unwrap_err();
    assert_eq!(err.key(), Some("kv.cpu.max_bytes"));
    let roomy = HostFacts {
        mem_total_bytes: Some(125 * GIB),
        disk_free_bytes: None,
    };
    cfg.validate_host(&roomy)
        .expect("64 GiB L1 fits 125 GiB minus an 8 GiB reserve");
    let mut nvme = KvConfig::default();
    nvme.cpu.max_bytes = ByteSize::gib(8);
    nvme.nvme.enabled = true;
    let err = nvme
        .validate_host(&small_host, ByteSize::gib(8))
        .unwrap_err();
    assert_eq!(err.key(), Some("kv.nvme.max_bytes"));
    assert!(
        err.to_string().contains(&(50 * GIB).to_string()),
        "names the free space: {err}"
    );
    let unknown = HostFacts {
        mem_total_bytes: None,
        disk_free_bytes: None,
    };
    nvme.validate_host(&unknown, ByteSize::gib(8))
        .expect("unknown host facts are not checked");

    // Once the model layout is known: in-flight bytes hold one block, a slab holds one slot.
    let block = 1_835_008;
    KvConfig::default()
        .validate_block_bytes(block)
        .expect("defaults fit a Llama-3.2-3B block");
    let err = KvConfig::default()
        .validate_block_bytes(2 * GIB)
        .unwrap_err();
    assert_eq!(err.key(), Some("kv.transfer.max_inflight_bytes"));
    let mut small_slab = KvConfig::default();
    small_slab.nvme.enabled = true;
    small_slab.nvme.slab_bytes = ByteSize::mib(1);
    let err = small_slab.validate_block_bytes(block).unwrap_err();
    assert_eq!(err.key(), Some("kv.nvme.slab_bytes"));
    assert_eq!(KV_IO_ALIGN, 4096);
}

#[test]
fn parallel_rejections() {
    use std::time::Duration;

    use crate::types::Vendor;

    let base = "model:\n  path: /m\n";
    // Defaults (P5 §Configuration).
    let d = parse(base, &[]).unwrap().parallel;
    assert_eq!(d.tensor_parallel_size, SizeOrAuto::Size(1));
    assert_eq!(d.data_parallel_size, SizeOrAuto::Size(1));
    assert_eq!(d.devices, DeviceSelection::Auto);
    assert_eq!(d.collective_backend.as_str(), "auto");
    assert_eq!(d.nccl_library, None);
    assert_eq!(d.rccl_library, None);
    assert!(!d.allow_device_sharing);
    assert_eq!(d.plan_queue_depth, 2);
    assert_eq!(d.router, DpRouterPolicy::PrefixAffinity);
    assert_eq!(d.collective.init_timeout.0, Duration::from_secs(120));
    assert_eq!(d.collective.op_timeout.0, Duration::from_secs(30));
    assert_eq!(d.ranks.mode, RankMode::Local);
    assert_eq!(d.ranks.rank, 0);
    assert_eq!(d.ranks.leader, None);
    assert_eq!(d.ranks.local_devices, vec![DeviceId(0)]);

    // Values the static rules accept.
    let ok = parse(
        base,
        &[
            "parallel.tensor_parallel_size=auto",
            "parallel.data_parallel_size=auto",
            "parallel.devices=[0, 1]",
            "parallel.router=least_loaded",
            "parallel.collective_backend=rccl",
        ],
    )
    .unwrap()
    .parallel;
    assert_eq!(ok.tensor_parallel_size, SizeOrAuto::Auto);
    assert_eq!(ok.data_parallel_size, SizeOrAuto::Auto);
    assert_eq!(
        ok.devices,
        DeviceSelection::List(vec![DeviceId(0), DeviceId(1)])
    );
    assert_eq!(ok.router, DpRouterPolicy::LeastLoaded);
    assert_eq!(ok.collective_backend.as_str(), "rccl");
    let stat = parse(
        base,
        &[
            "parallel.tensor_parallel_size=2",
            "parallel.ranks.mode=static",
            "parallel.ranks.rank=1",
            "parallel.ranks.leader=127.0.0.1:18100",
            "parallel.ranks.local_devices=[1]",
        ],
    )
    .unwrap()
    .parallel;
    assert_eq!(stat.ranks.mode, RankMode::Static);
    assert_eq!(
        stat.ranks.leader,
        Some("127.0.0.1:18100".parse::<SocketAddr>().unwrap())
    );

    // Static rules, each naming its key.
    let tp = "parallel.tensor_parallel_size";
    assert_rejected(base, &["parallel.tensor_parallel_size=3"], tp);
    assert_rejected(base, &["parallel.tensor_parallel_size=16"], tp);
    assert_rejected(base, &["parallel.tensor_parallel_size=0"], tp);
    assert_rejected(base, &["parallel.tensor_parallel_size=many"], tp);
    let dp = "parallel.data_parallel_size";
    assert_rejected(base, &["parallel.data_parallel_size=0"], dp);
    assert_rejected(base, &["parallel.data_parallel_size=65"], dp);
    assert_rejected(
        base,
        &["parallel.tensor_parallel_size=2", "parallel.devices=[0, 0]"],
        "parallel.devices",
    );
    assert_rejected(
        base,
        &["parallel.plan_queue_depth=0"],
        "parallel.plan_queue_depth",
    );
    assert_rejected(
        base,
        &["parallel.plan_queue_depth=17"],
        "parallel.plan_queue_depth",
    );
    assert_rejected(
        base,
        &[
            "parallel.ranks.mode=static",
            "parallel.ranks.leader=127.0.0.1:18100",
        ],
        "parallel.ranks.mode",
    );
    assert_rejected(
        base,
        &[
            "parallel.tensor_parallel_size=2",
            "parallel.ranks.mode=static",
        ],
        "parallel.ranks.leader",
    );
    assert_rejected(
        base,
        &[
            "parallel.tensor_parallel_size=2",
            "parallel.data_parallel_size=2",
            "parallel.ranks.mode=static",
            "parallel.ranks.leader=127.0.0.1:18100",
        ],
        "parallel.ranks.mode",
    );
    assert_rejected(
        base,
        &[
            "parallel.tensor_parallel_size=2",
            "parallel.ranks.mode=static",
            "parallel.ranks.leader=127.0.0.1:18100",
            "parallel.ranks.rank=2",
        ],
        "parallel.ranks.rank",
    );
    assert_rejected(
        base,
        &[
            "parallel.tensor_parallel_size=2",
            "parallel.ranks.mode=static",
            "parallel.ranks.leader=127.0.0.1:18100",
            "parallel.ranks.local_devices=[0, 1]",
        ],
        "parallel.ranks.local_devices",
    );
    let op = "parallel.collective.op_timeout";
    assert_rejected(base, &["parallel.collective.op_timeout=50ms"], op);
    assert_rejected(base, &["parallel.collective.op_timeout=11m"], op);
    assert_rejected(base, &["parallel.collective.op_timeout=\"30 s\""], op);
    let init = "parallel.collective.init_timeout";
    assert_rejected(base, &["parallel.collective.init_timeout=999ms"], init);
    assert_rejected(base, &["parallel.collective.init_timeout=31m"], init);
    // The backend is a registry name: malformed at parse, unregistered in validate_modules.
    assert_rejected(
        base,
        &["parallel.collective_backend=Gloo!"],
        "parallel.collective_backend",
    );
    let gloo = parse(base, &["parallel.collective_backend=gloo"]).unwrap();
    let err = gloo.validate_modules(&task1_modules()).unwrap_err();
    assert_eq!(err.key(), Some("parallel.collective_backend"), "{err}");
    // So is the rank transport (default `tcp`).
    assert_eq!(
        parse(base, &[]).unwrap().parallel.ranks.transport.as_str(),
        "tcp"
    );
    assert_rejected(
        base,
        &["parallel.ranks.transport=RDMA"],
        "parallel.ranks.transport",
    );
    let rdma = parse(base, &["parallel.ranks.transport=rdma"]).unwrap();
    let err = rdma.validate_modules(&task1_modules()).unwrap_err();
    assert_eq!(err.key(), Some("parallel.ranks.transport"), "{err}");
    assert!(err.to_string().contains("(registered: tcp)"), "{err}");
    assert!(
        parse(base, &["parallel.ranks.transport=tcp"])
            .unwrap()
            .validate_modules(&task1_modules())
            .is_ok()
    );
    assert!(
        parse(base, &[])
            .unwrap()
            .validate_modules(&task1_modules())
            .is_ok()
    );

    // Inventory rules (after device discovery).
    // What the registered modules serve (turbine-distributed): host memory, one vendor each.
    const HOST: Option<&[Vendor]> = Some(&[]);
    const NCCL: Option<&[Vendor]> = Some(&[Vendor::Nvidia]);
    const RCCL: Option<&[Vendor]> = Some(&[Vendor::Amd]);
    let gpu = |i: u32, v: Vendor| (DeviceId(i), v, Some("gfx1201".to_string()));
    let novanas = [gpu(0, Vendor::Amd), gpu(1, Vendor::Amd)];
    let with = |sets: &[&str]| parse(base, sets).unwrap().parallel;
    let names = |r: Result<(), ConfigError>, key: &str| match r {
        Ok(()) => panic!("expected an inventory rejection naming {key}"),
        Err(e) => assert!(e.to_string().contains(key), "{e} does not name {key}"),
    };
    let host_tp2 = with(&[
        "parallel.tensor_parallel_size=2",
        "parallel.collective_backend=host",
    ]);
    names(
        host_tp2.validate_devices(&novanas, HOST),
        "parallel.collective_backend",
    );
    // The host backend serves a plan without tensor parallelism, even on GPUs.
    let host_tp1 = with(&["parallel.collective_backend=host"]);
    assert!(host_tp1.validate_devices(&novanas[..1], HOST).is_ok());
    let tp2 = with(&["parallel.tensor_parallel_size=2"]);
    assert!(
        with(&[
            "parallel.collective_backend=rccl",
            "parallel.tensor_parallel_size=2"
        ])
        .validate_devices(&novanas, RCCL)
        .is_ok()
    );
    assert!(tp2.validate_devices(&novanas, None).is_ok());
    // Index 2 does not exist.
    names(
        with(&["parallel.devices=[0, 2]", "parallel.data_parallel_size=2"])
            .validate_devices(&novanas, None),
        "parallel.devices",
    );
    // Two devices for tp 1 × dp 1.
    names(
        with(&["parallel.devices=[0, 1]"]).validate_devices(&novanas, None),
        "parallel.devices",
    );
    // Mixed vendors.
    names(
        with(&["parallel.tensor_parallel_size=2", "parallel.devices=[0, 1]"])
            .validate_devices(&[gpu(0, Vendor::Amd), gpu(1, Vendor::Nvidia)], None),
        "parallel.devices",
    );
    names(
        with(&["parallel.collective_backend=nccl"]).validate_devices(&novanas, NCCL),
        "parallel.collective_backend",
    );
    names(
        with(&["parallel.collective_backend=rccl", "parallel.devices=[0]"])
            .validate_devices(&[gpu(0, Vendor::Nvidia)], RCCL),
        "parallel.collective_backend",
    );
    let sharing = with(&[
        "parallel.data_parallel_size=2",
        "parallel.devices=[0, 0]",
        "parallel.allow_device_sharing=true",
    ]);
    assert!(sharing.validate_devices(&novanas, None).is_ok());
}
