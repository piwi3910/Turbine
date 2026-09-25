//! Configuration model (contract §3.2): YAML file → `--set` overrides → typed `Config` →
//! static validation. Every error names the dotted key path.

mod byte_size;
mod overrides;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_norway::{Mapping, Value};

pub use byte_size::ByteSize;
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
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub max_request_bytes: ByteSize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            listen: SocketAddr::from(([0, 0, 0, 0], 8000)),
            max_request_bytes: ByteSize::mib(8),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    /// Required; the empty default is rejected by `validate`.
    pub path: PathBuf,
    pub dtype: ModelDtype,
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
            block_tokens: 16,
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
}

impl Default for KvGpuConfig {
    fn default() -> Self {
        KvGpuConfig { enabled: true }
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
    pub continuous_batching: bool,
    pub chunked_prefill: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            continuous_batching: true,
            chunked_prefill: true,
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
        if let Err(e) = tracing_subscriber::EnvFilter::try_new(&self.logging.level) {
            return Err(invalid(
                "logging.level",
                format!("not a valid tracing filter directive: {e}"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
