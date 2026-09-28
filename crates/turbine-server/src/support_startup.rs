//! The startup support-matrix decision (Phase 2m S-11, from the Phase 8 run-ahead): the key is
//! built from the module registries — `vendor` from the configured execution backend
//! (`ExecutionBackend::vendor`), `architecture` from the model's `config.json` (the Hugging Face
//! name a `ModelFamily` claims), `arch` from the discovered device, and the format columns from
//! [`format_columns`] (the checkpoint's weight format and the configured L0 KV dtype; no
//! speculation). It runs twice: [`before_discovery`] right after the configuration is read
//! (also under `--check-config`; `arch` unknown except on the host backend), and
//! [`after_discovery`] once the device's architecture is known. An `unsupported` resolution is a
//! configuration error (exit 2, before any port is bound).

use std::path::Path;

use turbine_core::config::{Config, ConfigError, KvDtypeChoice};
use turbine_core::support::{
    self, HOST_VENDOR, KvFormatColumn, SupportDecision, SupportKey, SupportStatus,
    WeightFormatColumn,
};
use turbine_device::DeviceInventory;

/// The support-matrix vendor column of `execution.backend` (already validated against the
/// backend registry by `Config::validate_modules`).
pub fn vendor(cfg: &Config) -> &'static str {
    turbine_kernels::backends::registry()
        .get(cfg.execution.backend.as_str())
        .map_or(support::WILDCARD, |b| b.vendor())
}

/// The architecture column of the model: `architectures[0]` of `<model_dir>/config.json`, or of
/// its `text_config` when the family registry resolves the model from there; `None` when the
/// file is missing or has no architecture (the model-config step reports that later).
pub fn read_architecture(model_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let top: serde_json::Value = serde_json::from_str(&text).ok()?;
    let first = |obj: &serde_json::Value| {
        obj.get("architectures")?
            .get(0)?
            .as_str()
            .map(str::to_string)
    };
    // A wrapper checkpoint names its text model in `text_config`: the registry resolved it there.
    if let Ok((_, obj)) = turbine_model::families::resolve(&top) {
        return first(obj);
    }
    first(&top)
}

/// The device architecture column of `execution.device` on a backend of `vendor`: the host
/// backend's is its vendor word; `None` when the device is missing or of another vendor (the
/// backend reports that when it opens).
pub fn device_arch(cfg: &Config, vendor: &str, inventory: &DeviceInventory) -> Option<String> {
    if vendor == HOST_VENDOR {
        return Some(HOST_VENDOR.to_string());
    }
    inventory
        .devices
        .iter()
        .find(|d| d.index == cfg.execution.device && d.vendor.as_str() == vendor)
        .and_then(|d| d.arch.clone())
}

/// The format columns of the configured model: `(weight_format, kv_format)`. Every checkpoint
/// the model loader accepts today is BF16 (the Phase 6a weight-format registry, plan Task 6,
/// supplies the detected column); the KV column is `kv.dtype` (Phase 6a S-13).
pub fn format_columns(cfg: &Config) -> (WeightFormatColumn, KvFormatColumn) {
    (WeightFormatColumn::Bf16, kv_column(cfg.kv.dtype))
}

/// The support-matrix KV column of `kv.dtype`: the column of the same spelling (a `kv.dtype`
/// value lands together with its column).
pub fn kv_column(dtype: KvDtypeChoice) -> KvFormatColumn {
    KvFormatColumn::ALL
        .into_iter()
        .find(|c| c.as_str() == dtype.as_str())
        .expect("every kv.dtype value has a support-matrix KV column")
}

/// The key before device discovery for `cfg` with the given format columns.
pub fn key_before_discovery(
    cfg: &Config,
    weight: WeightFormatColumn,
    kv: KvFormatColumn,
) -> SupportKey {
    let architecture = read_architecture(&cfg.model.path);
    SupportKey::before_discovery(vendor(cfg), architecture.as_deref(), weight, kv)
}

/// The decision before device discovery (startup and `--check-config`).
pub fn before_discovery(cfg: &Config) -> Result<SupportDecision, ConfigError> {
    let (weight, kv) = format_columns(cfg);
    support::check(key_before_discovery(cfg, weight, kv))
}

/// The decision once the device is known; `first` is kept when the device arch is unknown.
pub fn after_discovery(
    cfg: &Config,
    inventory: &DeviceInventory,
    first: SupportDecision,
) -> Result<SupportDecision, ConfigError> {
    let vendor = vendor(cfg);
    let Some(arch) = device_arch(cfg, vendor, inventory) else {
        return Ok(first);
    };
    if arch == first.key.arch {
        return Ok(first);
    }
    let key = SupportKey::for_model(
        vendor,
        &arch,
        &first.key.architecture,
        first.key.weight_format,
        first.key.kv_format,
        first.key.speculative,
    );
    support::check(key)
}

/// Logs the decision as `event="support_matrix"`: INFO when supported, WARN when experimental.
pub fn log(decision: &SupportDecision) {
    let row = decision.key.to_string();
    match decision.status {
        SupportStatus::Experimental => tracing::warn!(
            event = "support_matrix",
            status = decision.status.as_str(),
            row = %row,
            "support matrix: {row} is experimental"
        ),
        _ => tracing::info!(
            event = "support_matrix",
            status = decision.status.as_str(),
            row = %row,
            "support matrix: {row} is {}",
            decision.status.as_str()
        ),
    }
}

/// Logs a refusal as `event="support_matrix"` at ERROR (tracing may not be installed yet, so
/// the caller also prints it).
pub fn log_refusal(error: &ConfigError) {
    tracing::error!(
        event = "support_matrix",
        status = "unsupported",
        error = %error,
        "unsupported configuration"
    );
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use turbine_core::config::ModuleName;
    use turbine_core::support::{VENDORS, WILDCARD};
    use turbine_core::types::{DeviceId, MemoryKind, Vendor};
    use turbine_device::{DeviceInfo, DeviceMemoryInfo};

    /// A per-test directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "turbine-support-startup-{name}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn model_dir(name: &str, config: serde_json::Value) -> TempDir {
        let dir = TempDir::new(name);
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        dir
    }

    fn config(backend: &str, model: &Path) -> Config {
        let mut cfg = Config::default();
        cfg.execution.backend = ModuleName::new(backend).unwrap();
        cfg.model.path = model.to_path_buf();
        cfg
    }

    fn inventory(arch: &str) -> DeviceInventory {
        DeviceInventory {
            devices: vec![DeviceInfo {
                index: DeviceId(0),
                vendor: Vendor::Amd,
                vendor_index: 0,
                name: "test".into(),
                uuid: None,
                pci_bus_id: None,
                arch: Some(arch.into()),
                driver_version: None,
                memory: DeviceMemoryInfo {
                    kind: MemoryKind::Dedicated,
                    total_bytes: 1 << 30,
                    shared_with_host: false,
                },
            }],
            backends: Vec::new(),
        }
    }

    /// Spec edge case (the matrix and the family registry disagree): every registered family's
    /// every HF name has a row (not the "no support-matrix row" fallback) on every vendor, and
    /// every registered backend's vendor is a matrix vendor.
    #[test]
    fn every_registered_family_has_a_row_per_vendor() {
        for family in turbine_model::families::registry().iter() {
            for architecture in family.hf_architectures() {
                for vendor in VENDORS {
                    let key = SupportKey::before_discovery(
                        vendor,
                        Some(architecture),
                        WeightFormatColumn::Bf16,
                        KvFormatColumn::Bf16,
                    );
                    let status = support::resolve_partial_in(support::SUPPORT_MATRIX, &key);
                    assert_ne!(
                        status.reason(),
                        Some("no support-matrix row"),
                        "{} ({architecture}) on {vendor}",
                        family.name()
                    );
                }
            }
        }
        for backend in turbine_kernels::backends::registry().iter() {
            assert!(VENDORS.contains(&backend.vendor()), "{}", backend.name());
        }
    }

    /// Phase 6a S-13: the KV column is the configured `kv.dtype`; FP8 KV is experimental on
    /// `amd` gfx1201 Llama / OLMoE during its lab proof, refused elsewhere on `amd` (exit 2
    /// naming `kv.dtype`) until its gate passes, and experimental on the cpu backend.
    /// Breaks if the KV column is still hard-coded BF16.
    #[test]
    fn kv_column_follows_kv_dtype() {
        let llama = model_dir(
            "llama-fp8-kv",
            serde_json::json!({"architectures": ["LlamaForCausalLM"]}),
        );
        let mut cfg = config("hip", llama.path());
        assert_eq!(format_columns(&cfg).1, KvFormatColumn::Bf16);
        cfg.kv.dtype = turbine_core::config::KvDtypeChoice::Fp8E4m3;
        assert_eq!(format_columns(&cfg).1, KvFormatColumn::Fp8E4m3);
        // Before discovery the arch is unknown: the gfx1201 row (experimental during the lab
        // proof) is the best any card could give.
        let d = before_discovery(&cfg).unwrap();
        assert_eq!(d.status.as_str(), "experimental", "{:?}", d.key);
        // A key on another architecture family is refused naming `kv.dtype`.
        let qwen = model_dir(
            "qwen-fp8-kv",
            serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}),
        );
        let mut other = config("hip", qwen.path());
        other.kv.dtype = turbine_core::config::KvDtypeChoice::Fp8E4m3;
        let err = before_discovery(&other).unwrap_err();
        assert_eq!(err.key(), Some("kv.dtype"), "{err}");
        let mut cpu = config("cpu", llama.path());
        cpu.kv.dtype = turbine_core::config::KvDtypeChoice::Fp8E4m3;
        let d = before_discovery(&cpu).unwrap();
        assert_eq!(
            d.key.to_string(),
            "cpu/cpu/LlamaForCausalLM/bf16/fp8_e4m3/none"
        );
        assert_eq!(d.status, SupportStatus::Experimental);
    }

    #[test]
    fn decisions_before_and_after_discovery() {
        let llama = model_dir(
            "llama",
            serde_json::json!({"architectures": ["LlamaForCausalLM"]}),
        );
        let cfg = config("hip", llama.path());
        assert_eq!(vendor(&cfg), "amd");
        let first = before_discovery(&cfg).unwrap();
        assert_eq!(
            first.key.to_string(),
            "amd/*/LlamaForCausalLM/bf16/bf16/none"
        );
        assert_eq!(first.status, SupportStatus::Supported);

        // The discovered gfx1201 resolves the exact row; an unknown device keeps the first.
        let d = after_discovery(&cfg, &inventory("gfx1201"), first.clone()).unwrap();
        assert_eq!(
            d.key.to_string(),
            "amd/gfx1201/LlamaForCausalLM/bf16/bf16/none"
        );
        assert_eq!(d.status, SupportStatus::Supported);
        let none = DeviceInventory {
            devices: Vec::new(),
            backends: Vec::new(),
        };
        assert_eq!(after_discovery(&cfg, &none, first.clone()).unwrap(), first);
        // A device architecture without a row is refused, naming the row.
        let err = after_discovery(&cfg, &inventory("gfx942"), first).unwrap_err();
        assert!(err.to_string().contains("no support-matrix row"), "{err}");
        assert!(err.to_string().contains("amd/gfx942/"), "{err}");

        // The CPU backend runs on the host: its key is complete before discovery.
        let cpu = config("cpu", llama.path());
        let d = before_discovery(&cpu).unwrap();
        assert_eq!(d.key.to_string(), "cpu/cpu/LlamaForCausalLM/bf16/bf16/none");
        assert_eq!(d.status, SupportStatus::Experimental);
        assert_eq!(after_discovery(&cpu, &none, d.clone()).unwrap(), d);

        // A Phase 7 family is refused on the GPU vendor before discovery, served on the host.
        let qwen = model_dir(
            "qwen",
            serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}),
        );
        let err = before_discovery(&config("hip", qwen.path())).unwrap_err();
        assert_eq!(err.key(), Some("execution.backend"));
        assert!(err.to_string().contains("phase-7-model-families"), "{err}");
        assert!(before_discovery(&config("cpu", qwen.path())).is_ok());

        // No config.json: the architecture is unknown.
        let empty = TempDir::new("empty");
        let d = before_discovery(&config("hip", empty.path())).unwrap();
        assert_eq!(d.key.architecture, WILDCARD);
    }

    /// Phase 6a S-1: the key carries the checkpoint's weight format and the L0 KV dtype instead
    /// of hard-coded BF16, before and after discovery. Breaks if either column is dropped.
    #[test]
    fn key_uses_detected_format() {
        let llama = model_dir(
            "fmt",
            serde_json::json!({"architectures": ["LlamaForCausalLM"]}),
        );
        let cfg = config("hip", llama.path());
        // Nothing configured: BF16 weights (the only loadable format so far) and BF16 KV.
        assert_eq!(
            format_columns(&cfg),
            (WeightFormatColumn::Bf16, KvFormatColumn::Bf16)
        );
        let key = key_before_discovery(&cfg, WeightFormatColumn::Fp8, KvFormatColumn::Bf16);
        assert_eq!(key.to_string(), "amd/*/LlamaForCausalLM/fp8/bf16/none");
        let err = support::check(key).unwrap_err();
        assert_eq!(err.key(), Some("model.path"));
        assert!(err.to_string().contains("phase-6a-quantization"), "{err}");

        // After discovery the format columns are kept.
        let first = SupportDecision {
            key: key_before_discovery(&cfg, WeightFormatColumn::Fp8, KvFormatColumn::Fp8E4m3),
            status: SupportStatus::Experimental,
        };
        let err = after_discovery(&cfg, &inventory("gfx1201"), first).unwrap_err();
        assert!(
            err.to_string()
                .contains("amd/gfx1201/LlamaForCausalLM/fp8/fp8_e4m3/none"),
            "{err}"
        );
    }

    #[test]
    fn architecture_from_text_config_when_the_family_resolves_there() {
        let wrapped = model_dir(
            "wrapped",
            serde_json::json!({
                "architectures": ["SomeWrapperForConditionalGeneration"],
                "text_config": {"architectures": ["LlamaForCausalLM"]}
            }),
        );
        assert_eq!(
            read_architecture(wrapped.path()).as_deref(),
            Some("LlamaForCausalLM")
        );
        let unknown = model_dir(
            "unknown",
            serde_json::json!({"architectures": ["GptOssForCausalLM"]}),
        );
        assert_eq!(
            read_architecture(unknown.path()).as_deref(),
            Some("GptOssForCausalLM")
        );
    }
}
