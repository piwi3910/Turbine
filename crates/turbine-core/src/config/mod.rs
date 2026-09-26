//! Configuration model (contract §3.2): YAML file → `--set` overrides → typed `Config` →
//! static validation. Every error names the dotted key path.

mod byte_size;
mod duration;
mod overrides;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_norway::{Mapping, Value};

pub use byte_size::ByteSize;
pub use duration::HumanDuration;

use crate::types::{DeviceId, ExecutionBackend};
pub use overrides::Override;

/// Configuration errors. Every variant maps to exit code 2 in `turbine-server`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: not valid YAML: {detail}")]
    Syntax { path: PathBuf, detail: String },
    #[error("{key}: unknown key")]
    UnknownKey { key: String },
    #[error("{key}: {reason}")]
    Invalid { key: String, reason: String },
    #[error("--set {arg}: {reason}")]
    BadOverride { arg: String, reason: String },
}

impl ConfigError {
    /// The dotted key path this error is about, when there is one.
    pub fn key(&self) -> Option<&str> {
        match self {
            ConfigError::UnknownKey { key } | ConfigError::Invalid { key, .. } => Some(key),
            _ => None,
        }
    }
}

fn invalid(key: &str, reason: impl Into<String>) -> ConfigError {
    ConfigError::Invalid {
        key: key.to_string(),
        reason: reason.into(),
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub kv: KvConfig,
    pub reliability: ReliabilityConfig,
    pub scheduler: SchedulerConfig,
    pub distributed: DistributedConfig,
    pub logging: LoggingConfig,
    pub devices: DevicesConfig,
    pub execution: ExecutionConfig,
    pub structured_output: StructuredOutputConfig,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub max_request_bytes: ByteSize,
    /// Total request deadline (Phase 2).
    pub request_timeout: HumanDuration,
    /// How long a request may stay paused on a full output channel (Phase 2).
    pub slow_client_timeout: HumanDuration,
    /// Drain time for running requests after SIGINT/SIGTERM (Phase 2).
    pub shutdown_grace: HumanDuration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            listen: SocketAddr::from(([0, 0, 0, 0], 8000)),
            max_request_bytes: ByteSize::mib(8),
            request_timeout: HumanDuration::from_secs(600),
            slow_client_timeout: HumanDuration::from_secs(30),
            shutdown_grace: HumanDuration::from_secs(30),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    /// Required; the empty default is rejected by `validate`.
    pub path: PathBuf,
    pub dtype: ModelDtype,
    /// Default: `<org>/<name>` for a Hugging Face snapshot path, else the last path component.
    pub served_name: Option<String>,
    /// Default: `<model.path>/tokenizer.json`.
    pub tokenizer: Option<PathBuf>,
    /// Default: `<model.path>/chat_template.jinja`, else `<model.path>/tokenizer_config.json`.
    pub chat_template: Option<PathBuf>,
    /// Default: min(32768, max_position_embeddings); the upper bound is checked at startup.
    pub max_seq_len: Option<u32>,
    /// Tool-call output parser (Phase 2). Null → `llama3_json` for `LlamaForCausalLM` whose
    /// template renders `tools`, else `none` (resolved at startup).
    pub tool_call_parser: Option<ToolCallParserKind>,
}

/// `model.tool_call_parser` values.
#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolCallParserKind {
    /// Llama-3.x JSON calls: optional `<|python_tag|>`, then `;`-separated call objects.
    Llama3Json,
    None,
}

#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelDtype {
    #[default]
    Bf16,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvConfig {
    pub block_tokens: u32,
    pub gpu: KvGpuConfig,
    pub cpu: KvCpuConfig,
    pub nvme: KvNvmeConfig,
}

impl Default for KvConfig {
    fn default() -> Self {
        KvConfig {
            block_tokens: 128,
            gpu: KvGpuConfig::default(),
            cpu: KvCpuConfig::default(),
            nvme: KvNvmeConfig::default(),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvGpuConfig {
    pub enabled: bool,
    /// L0 block-pool size (Phase 2, default 8GiB). Null is allowed; from Phase 3 null means
    /// the `kv` pool remainder of the budget (CONFLICT C-8).
    pub max_bytes: Option<ByteSize>,
}

impl Default for KvGpuConfig {
    fn default() -> Self {
        KvGpuConfig {
            enabled: true,
            max_bytes: Some(ByteSize::gib(8)),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvCpuConfig {
    pub enabled: bool,
    pub max_bytes: ByteSize,
}

impl Default for KvCpuConfig {
    fn default() -> Self {
        KvCpuConfig {
            enabled: true,
            max_bytes: ByteSize::gib(64),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvNvmeConfig {
    pub enabled: bool,
    pub path: PathBuf,
}

impl Default for KvNvmeConfig {
    fn default() -> Self {
        KvNvmeConfig {
            enabled: false,
            path: PathBuf::from("/var/lib/turbine/kv"),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ReliabilityConfig {
    pub enabled: bool,
    pub emergency_vram_reserve: ByteSize,
    pub adaptive_admission: bool,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        ReliabilityConfig {
            enabled: true,
            emergency_vram_reserve: ByteSize::gib(2),
            adaptive_admission: true,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct SchedulerConfig {
    /// `false` forces one running request (`Config::effective_max_running`).
    pub continuous_batching: bool,
    /// `false` rejects prompts longer than `max_batch_tokens` at submission.
    pub chunked_prefill: bool,
    pub max_running_requests: u32,
    /// Per-iteration token budget (decodes + prefill chunks).
    pub max_batch_tokens: u32,
    pub prefill_chunk_tokens: u32,
    /// Waiting-queue bound and HTTP→engine submission-channel capacity (Phase 2, C-1).
    pub max_queued_requests: u32,
    /// Longest wait in the queue before 503 `queue_timeout` (Phase 2 only, C-1).
    pub queue_timeout: HumanDuration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            continuous_batching: true,
            chunked_prefill: true,
            max_running_requests: 64,
            max_batch_tokens: 8192,
            prefill_chunk_tokens: 2048,
            max_queued_requests: 256,
            queue_timeout: HumanDuration::from_secs(60),
        }
    }
}

/// `structured_output` section (Phase 2): bounds on constrained-decoding grammars.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct StructuredOutputConfig {
    /// Largest accepted schema or tool-definition JSON.
    pub max_schema_bytes: ByteSize,
    /// Longest grammar compilation before 400 `invalid_json_schema`.
    pub compile_timeout: HumanDuration,
}

impl Default for StructuredOutputConfig {
    fn default() -> Self {
        StructuredOutputConfig {
            max_schema_bytes: ByteSize::kib(64),
            compile_timeout: HumanDuration::from_secs(5),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct DistributedConfig {
    pub enabled: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct LoggingConfig {
    pub format: LogFormat,
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            format: LogFormat::Text,
            level: "info".to_string(),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LogFormat {
    #[default]
    Text,
    Json,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct DevicesConfig {
    pub nvml_library: Option<PathBuf>,
    pub amd_smi_library: Option<PathBuf>,
}

/// `execution` section (Phase 1): which kernel backend runs the model, on which device; the
/// Phase 2c switches, each defaulting to the optimised path and each able to restore the
/// Phase 2 one (P2c S-14).
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ExecutionConfig {
    pub backend: ExecutionBackend,
    /// Global index from the device inventory.
    pub device: DeviceId,
    /// Explicit kernel shim path; null → TURBINE_KERNEL_LIBRARY, beside the executable, loader path.
    pub kernel_library: Option<PathBuf>,
    /// Tune the GEMM algorithm per shape at first use (false: the first heuristic answer).
    pub gemm_autotune: bool,
    /// Capture decode-only iterations into graphs and replay them (false: always eager).
    pub decode_graphs: bool,
    /// Reduce eligible logits rows on the device (false: every row copied to the host).
    pub device_sampling: bool,
    /// Fused QKV and gate/up projections and `add_rmsnorm` (false: separate ops).
    pub fused_ops: bool,
    /// Threads sampling rows in parallel, 1..=64 (1: serial);
    /// see [`ExecutionConfig::effective_sampler_threads`].
    pub sampler_threads: u32,
    /// Launch each engine iteration before the host work of the previous one, its decodes
    /// taking their tokens on the device (false: one iteration at a time). Needs a kernel
    /// library with host staging (ABI v2.3) and device sampling; otherwise ignored. Off by
    /// default: it measured no faster on novanas and delayed OLMoE's first tokens.
    pub overlap_scheduling: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            backend: ExecutionBackend::Hip,
            device: DeviceId(0),
            kernel_library: None,
            gemm_autotune: true,
            decode_graphs: true,
            device_sampling: true,
            fused_ops: true,
            sampler_threads: 4,
            overlap_scheduling: false,
        }
    }
}

impl ExecutionConfig {
    /// Sampler threads on a host with `available` parallelism: `sampler_threads`, at most
    /// `available − 1` (the engine thread keeps a core), at least 1.
    pub fn effective_sampler_threads(&self, available: usize) -> usize {
        (self.sampler_threads as usize)
            .min(available.saturating_sub(1))
            .max(1)
    }
}

/// Read `path`, apply `overrides` in order, deserialize and validate.
pub fn load(path: &Path, overrides: &[Override]) -> Result<Config, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    load_from_str(&text, path, overrides)
}

fn load_from_str(text: &str, origin: &Path, overrides: &[Override]) -> Result<Config, ConfigError> {
    let mut root: Value = serde_norway::from_str(text).map_err(|e| ConfigError::Syntax {
        path: origin.to_path_buf(),
        detail: e.to_string(),
    })?;
    match root {
        Value::Null => root = Value::Mapping(Mapping::new()),
        Value::Mapping(_) => {}
        _ => {
            return Err(ConfigError::Syntax {
                path: origin.to_path_buf(),
                detail: "the top level must be a mapping".to_string(),
            });
        }
    }
    for o in overrides {
        overrides::set_path(&mut root, &o.key, o.value.clone());
    }
    let config = deserialize(&root)?;
    config.validate()?;
    Ok(config)
}

fn deserialize(root: &Value) -> Result<Config, ConfigError> {
    let known = serde_norway::to_value(Config::default())
        .map_err(|e| invalid("<root>", format!("cannot build the key schema: {e}")))?;
    if let Some(key) = overrides::first_unknown_key(root, &known, "") {
        return Err(ConfigError::UnknownKey { key });
    }
    match serde_norway::from_value::<Config>(root.clone()) {
        Ok(config) => Ok(config),
        Err(whole) => {
            // Name the key: re-deserialize the defaults with one user leaf at a time.
            let mut user_leaves = Vec::new();
            overrides::leaves(root, &known, "", &mut user_leaves);
            for (key, value) in user_leaves {
                let mut probe = known.clone();
                overrides::set_path(&mut probe, &key, value);
                if let Err(e) = serde_norway::from_value::<Config>(probe) {
                    return Err(invalid(&key, e.to_string()));
                }
            }
            Err(invalid("<root>", whole.to_string()))
        }
    }
}

impl Config {
    /// Static validation rules (exit 2). Runs before device discovery and before binding.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mrb = self.server.max_request_bytes;
        if mrb < ByteSize::kib(1) || mrb > ByteSize::mib(256) {
            return Err(invalid(
                "server.max_request_bytes",
                format!("must be between 1KiB and 256MiB, got {mrb}"),
            ));
        }
        if self.model.path.as_os_str().is_empty() {
            return Err(invalid("model.path", "is required and must be non-empty"));
        }
        if !(1..=1024).contains(&self.kv.block_tokens) {
            return Err(invalid(
                "kv.block_tokens",
                format!("must be between 1 and 1024, got {}", self.kv.block_tokens),
            ));
        }
        if self.kv.cpu.enabled && self.kv.cpu.max_bytes.0 == 0 {
            return Err(invalid(
                "kv.cpu.max_bytes",
                "must be greater than 0 when kv.cpu.enabled is true",
            ));
        }
        if self.kv.nvme.enabled && !self.kv.nvme.path.is_absolute() {
            return Err(invalid(
                "kv.nvme.path",
                format!(
                    "must be an absolute path when kv.nvme.enabled is true, got {}",
                    self.kv.nvme.path.display()
                ),
            ));
        }
        if self.distributed.enabled {
            return Err(invalid(
                "distributed.enabled",
                "distributed mode is not supported in this build",
            ));
        }
        if let Some(name) = &self.model.served_name
            && (name.is_empty() || name.chars().count() > 256)
        {
            return Err(invalid(
                "model.served_name",
                "must be between 1 and 256 characters",
            ));
        }
        if self.model.max_seq_len == Some(0) {
            return Err(invalid("model.max_seq_len", "must be at least 1"));
        }
        if self.execution.backend == ExecutionBackend::Cuda {
            return Err(invalid(
                "execution.backend",
                "cuda is not available in this build; NVIDIA execution arrives with phase-2b-nvidia",
            ));
        }
        if let Err(e) = tracing_subscriber::EnvFilter::try_new(&self.logging.level) {
            return Err(invalid(
                "logging.level",
                format!("not a valid tracing filter directive: {e}"),
            ));
        }
        self.validate_phase2()
    }

    /// Running-request bound actually applied: `scheduler.continuous_batching: false` forces 1.
    pub fn effective_max_running(&self) -> u32 {
        if self.scheduler.continuous_batching {
            self.scheduler.max_running_requests
        } else {
            1
        }
    }

    /// Phase 2 keys: timeouts, scheduler bounds and structured-output bounds.
    fn validate_phase2(&self) -> Result<(), ConfigError> {
        for (key, d) in [
            ("server.request_timeout", self.server.request_timeout),
            (
                "server.slow_client_timeout",
                self.server.slow_client_timeout,
            ),
            ("scheduler.queue_timeout", self.scheduler.queue_timeout),
            (
                "structured_output.compile_timeout",
                self.structured_output.compile_timeout,
            ),
        ] {
            if d.is_zero() {
                return Err(invalid(key, "must be greater than 0"));
            }
        }
        let s = &self.scheduler;
        if !(1..=1024).contains(&s.max_running_requests) {
            return Err(invalid(
                "scheduler.max_running_requests",
                format!("must be between 1 and 1024, got {}", s.max_running_requests),
            ));
        }
        if s.max_batch_tokens < s.max_running_requests {
            return Err(invalid(
                "scheduler.max_batch_tokens",
                format!(
                    "must be at least scheduler.max_running_requests ({}), got {}",
                    s.max_running_requests, s.max_batch_tokens
                ),
            ));
        }
        if s.max_batch_tokens < self.kv.block_tokens {
            return Err(invalid(
                "scheduler.max_batch_tokens",
                format!(
                    "must be at least kv.block_tokens ({}), got {}",
                    self.kv.block_tokens, s.max_batch_tokens
                ),
            ));
        }
        if !(1..=s.max_batch_tokens).contains(&s.prefill_chunk_tokens) {
            return Err(invalid(
                "scheduler.prefill_chunk_tokens",
                format!(
                    "must be between 1 and scheduler.max_batch_tokens ({}), got {}",
                    s.max_batch_tokens, s.prefill_chunk_tokens
                ),
            ));
        }
        if !(1..=65536).contains(&s.max_queued_requests) {
            return Err(invalid(
                "scheduler.max_queued_requests",
                format!("must be between 1 and 65536, got {}", s.max_queued_requests),
            ));
        }
        if !(1..=64).contains(&self.execution.sampler_threads) {
            return Err(invalid(
                "execution.sampler_threads",
                format!(
                    "must be between 1 and 64, got {}",
                    self.execution.sampler_threads
                ),
            ));
        }
        let msb = self.structured_output.max_schema_bytes;
        if msb < ByteSize::kib(1) || msb > ByteSize::mib(1) {
            return Err(invalid(
                "structured_output.max_schema_bytes",
                format!("must be between 1KiB and 1MiB, got {msb}"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
