//! Configuration model (contract §3.2): YAML file → `--set` overrides → typed `Config` →
//! static validation. Every error names the dotted key path.

mod byte_size;
mod duration;
mod kv;
mod overrides;
mod reliability;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_norway::{Mapping, Value};

pub use byte_size::ByteSize;
pub use duration::HumanDuration;
pub use kv::{
    HostFacts, KV_IO_ALIGN, KvConfig, KvCpuConfig, KvGpuConfig, KvNvmeConfig, KvPolicyKind,
    KvPolicyWeights, KvPrefetchConfig, KvSessionConfig, KvTransferConfig,
};
pub use reliability::*;

use crate::registry::valid_name;
use crate::types::DeviceId;
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

/// The name of a module in a registry (Phase 2m S-1): `^[a-z0-9_]{1,64}$`. The configuration
/// only checks the form; whether a module of that name exists is checked against the
/// registries by [`Config::validate_modules`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModuleName(String);

impl ModuleName {
    pub fn new(s: &str) -> Result<ModuleName, String> {
        if valid_name(s) {
            Ok(ModuleName(s.to_string()))
        } else {
            Err(format!(
                "`{s}` is not a module name (lowercase letters, digits and `_`, 1 to 64 characters)"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A default name written in this file; always well-formed.
    fn fixed(s: &'static str) -> ModuleName {
        debug_assert!(valid_name(s), "{s}");
        ModuleName(s.to_string())
    }
}

impl TryFrom<String> for ModuleName {
    type Error = String;
    fn try_from(s: String) -> Result<ModuleName, String> {
        ModuleName::new(&s)
    }
}

impl From<ModuleName> for String {
    fn from(name: ModuleName) -> String {
        name.0
    }
}

impl std::fmt::Display for ModuleName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The module names the registries hold, per configuration key, for
/// [`Config::validate_modules`].
#[derive(Clone, Copy, Debug)]
pub struct ModuleNames<'a> {
    /// `model.tool_call_parser` (`none` is always accepted).
    pub tool_formats: &'a [&'a str],
    /// `execution.backend`.
    pub backends: &'a [&'a str],
    /// `execution.card_profile` (`auto` is always accepted).
    pub card_profiles: &'a [&'a str],
    /// `scheduler.policy`.
    pub scheduling_policies: &'a [&'a str],
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
    /// Tool-call format (Phase 2; a `tool_format` registry name from Phase 2m). Null →
    /// `llama3_json` for `LlamaForCausalLM` whose template renders `tools`, else none (resolved
    /// at startup); `none` turns tool calling off.
    pub tool_call_parser: Option<ModuleName>,
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
pub struct SchedulerConfig {
    /// `false` forces one running request (`Config::effective_max_running`).
    pub continuous_batching: bool,
    /// `false` rejects prompts longer than `max_batch_tokens` at submission.
    pub chunked_prefill: bool,
    pub max_running_requests: u32,
    /// Per-iteration token budget (decodes + prefill chunks).
    pub max_batch_tokens: u32,
    pub prefill_chunk_tokens: u32,
    /// HTTP→engine submission-channel capacity (from Phase 3 only that, C-1).
    pub max_queued_requests: u32,
    /// Removed in Phase 3 (CONFLICT C-1): the admission queue's
    /// `reliability.admission.queue_timeout` bounds queue wait. Parsed only so `validate`
    /// rejects it naming the key.
    pub queue_timeout: Option<Value>,
    /// Scheduling policy (Phase 2m): a `scheduling_policy` registry name.
    pub policy: ModuleName,
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
            queue_timeout: None,
            policy: ModuleName::fixed("default"),
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
    /// An `execution_backend` registry name (`hip`, `cpu`).
    pub backend: ModuleName,
    /// Global index from the device inventory.
    pub device: DeviceId,
    /// Card profile (Phase 2m): a `card_profile` registry name, or `auto` for the profile of
    /// the device's architecture.
    pub card_profile: ModuleName,
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
            backend: ModuleName::fixed("hip"),
            device: DeviceId(0),
            card_profile: ModuleName::fixed("auto"),
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
        self.kv.validate()?;
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
        if self.scheduler.queue_timeout.is_some() {
            return Err(invalid(
                "scheduler.queue_timeout",
                "removed in phase 3: use reliability.admission.queue_timeout",
            ));
        }
        self.reliability.validate(self.kv.block_tokens)?;
        if let Err(e) = tracing_subscriber::EnvFilter::try_new(&self.logging.level) {
            return Err(invalid(
                "logging.level",
                format!("not a valid tracing filter directive: {e}"),
            ));
        }
        self.validate_phase2()
    }

    /// Rules that need host facts (contract §3.2): `kv.cpu.max_bytes` against `MemTotal` minus
    /// `reliability.memory.host_reserve_bytes`, `kv.nvme.max_bytes` against free disk. The
    /// server maps an error on a `kv.nvme.*` key to exit 1 and any other key to exit 2.
    pub fn validate_host(&self, host: &HostFacts) -> Result<(), ConfigError> {
        self.kv
            .validate_host(host, self.reliability.memory.host_reserve_bytes)
    }

    /// Checks every module key against the registries' names (`known`): exit 2, before device
    /// discovery and before binding. `none` (`model.tool_call_parser`) and `auto`
    /// (`execution.card_profile`) are always accepted.
    pub fn validate_modules(&self, known: &ModuleNames<'_>) -> Result<(), ConfigError> {
        fn check(
            key: &str,
            name: &ModuleName,
            registered: &[&str],
            always: Option<&str>,
        ) -> Result<(), ConfigError> {
            let name = name.as_str();
            if always == Some(name) || registered.contains(&name) {
                return Ok(());
            }
            Err(invalid(
                key,
                format!(
                    "`{name}` is not registered (registered: {})",
                    registered.join(", ")
                ),
            ))
        }
        if let Some(parser) = &self.model.tool_call_parser {
            check(
                "model.tool_call_parser",
                parser,
                known.tool_formats,
                Some("none"),
            )?;
        }
        check(
            "execution.backend",
            &self.execution.backend,
            known.backends,
            None,
        )?;
        check(
            "execution.card_profile",
            &self.execution.card_profile,
            known.card_profiles,
            Some("auto"),
        )?;
        check(
            "scheduler.policy",
            &self.scheduler.policy,
            known.scheduling_policies,
            None,
        )
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
