//! The startup support-matrix decision (Phase 2m S-11, from the Phase 8 run-ahead): the key is
//! built from the module registries — `vendor` from the configured execution backend
//! (`ExecutionBackend::vendor`), `architecture` from the model's `config.json` (the Hugging Face
//! name a `ModelFamily` claims), `arch` from the discovered device, and the format columns from
//! [`format_columns`] (the checkpoint's weight format and the configured L0 KV dtype; no
//! speculation). It runs twice: [`before_discovery`] right after the configuration is read
//! (also under `--check-config`; `arch` unknown except on the host backend), and
//! [`after_discovery`] once the device's architecture is known. An `unsupported` resolution is a
//! configuration error (exit 2, before any port is bound). The lower-tier KV formats (P6b
//! S-2) are resolved beside the row, against `TIER_FORMAT_REFUSALS` ([`tier_formats`]), and
//! [`kv_format_availability`] refuses, before binding, the KV formats no kernel provider can
//! run yet (exit 1).

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

/// The format columns of the configured model: `(weight_format, kv_format)`. The weight column
/// is the one of the checkpoint's packaging (`turbine_model::weights::detect` on `config.json`,
/// Phase 6a S-3); BF16 when `config.json` cannot be read or detection refuses it (the model load
/// then reports why). The KV column is `kv.dtype` (Phase 6a S-13).
pub fn format_columns(cfg: &Config) -> (WeightFormatColumn, KvFormatColumn) {
    (weight_column(&cfg.model.path), kv_column(cfg.kv.dtype))
}

/// The support-matrix weight column of the checkpoint in `model_dir`.
pub fn weight_column(model_dir: &Path) -> WeightFormatColumn {
    std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|top| turbine_model::weights::detect(&top).ok())
        .map_or(WeightFormatColumn::Bf16, |format| format.column())
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

/// The decision before device discovery (startup and `--check-config`); the lower-tier KV
/// formats are checked first ([`tier_formats`]).
pub fn before_discovery(cfg: &Config) -> Result<SupportDecision, ConfigError> {
    tier_formats(cfg)?;
    let (weight, kv) = format_columns(cfg);
    support::check(key_before_discovery(cfg, weight, kv))
}

/// The lower-tier KV formats in use, `(key, format)`: `kv.cpu.format` and `kv.nvme.format` of
/// the enabled tiers, from L1 down, and, with the ladder on, `kv.ladder.max_format` (the
/// lossiest format a tier can reach).
fn tier_format_keys(cfg: &Config) -> Vec<(&'static str, &str)> {
    let kv = &cfg.kv;
    let mut keys = Vec::new();
    if kv.cpu.enabled {
        keys.push(("kv.cpu.format", kv.cpu.format.as_str()));
    }
    if kv.nvme.enabled {
        keys.push(("kv.nvme.format", kv.nvme.format.as_str()));
    }
    if kv.ladder.enabled {
        keys.push(("kv.ladder.max_format", kv.ladder.max_format.as_str()));
    }
    keys
}

fn invalid(key: &str, reason: String) -> ConfigError {
    ConfigError::Invalid {
        key: key.to_string(),
        reason,
    }
}

/// The rung of `format` below the configured L0 (`turbine_kv::codec::tier_rung`); `key` names
/// the setting in the error for an unregistered codec.
fn rung(cfg: &Config, key: &str, format: &str) -> Result<usize, ConfigError> {
    turbine_kv::codec::tier_rung(format, cfg.kv.dtype.as_str()).ok_or_else(|| {
        invalid(
            key,
            format!(
                "`{format}` is not registered (registered: {})",
                turbine_kv::codec::registry().names().join(", ")
            ),
        )
    })
}

/// The tier ordering of P6b S-2, from the `kv_format` registry's order (exit 2): each enabled
/// lower tier no more precise than the tier above it (L1 against `kv.dtype`, L2 against L1
/// when it is enabled, else `kv.dtype`), naming both keys; `kv.ladder.max_format` a lossy
/// codec (not the first, lossless `l0`).
fn tier_ordering(cfg: &Config) -> Result<(), ConfigError> {
    let kv = &cfg.kv;
    let l0 = kv.dtype.as_str();
    let mut above = (
        "kv.dtype",
        l0,
        turbine_kv::codec::tier_rung("l0", l0).unwrap_or(0),
    );
    for (enabled, key, format) in [
        (kv.cpu.enabled, "kv.cpu.format", kv.cpu.format.as_str()),
        (kv.nvme.enabled, "kv.nvme.format", kv.nvme.format.as_str()),
    ] {
        if !enabled {
            continue;
        }
        let r = rung(cfg, key, format)?;
        if r < above.2 {
            return Err(invalid(
                key,
                format!(
                    "{format} is more precise than {} ({}): a tier may not be more precise than \
                     the tier above it",
                    above.0, above.1
                ),
            ));
        }
        above = (key, format, r);
    }
    let max = kv.ladder.max_format.as_str();
    if turbine_kv::codec::rung_index(max) == Some(0) {
        return Err(invalid(
            "kv.ladder.max_format",
            format!("must be a lossy codec, got the lossless {max}"),
        ));
    }
    Ok(())
}

/// Checks the lower-tier KV formats in use (P6b S-2): first their ordering ([`tier_ordering`]),
/// then each against `TIER_FORMAT_REFUSALS`: an unsupported one is a configuration error naming
/// its key (exit 2, also under `--check-config`); the rest are returned with their status.
pub fn tier_formats(cfg: &Config) -> Result<Vec<(&'static str, &str, SupportStatus)>, ConfigError> {
    tier_ordering(cfg)?;
    tier_format_keys(cfg)
        .into_iter()
        .map(|(key, format)| {
            let status = support::check_tier_format(key, format)?;
            Ok((key, format, status))
        })
        .collect()
}

/// The planner penalty of codec `name` (P6b S-3): `kv.lossy_penalty.<name>` when set, else the
/// codec's own default (0 for an unregistered name, which startup has already refused).
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the reuse planner reads it from 6b Task 4")
)]
pub fn lossy_penalty(cfg: &Config, name: &str) -> f64 {
    cfg.kv.lossy_penalty_override(name).unwrap_or_else(|| {
        turbine_kv::codec::registry()
            .get(name)
            .map_or(0.0, |c| c.default_lossy_penalty())
    })
}

/// Logs every experimental lower-tier KV format as `event="support_matrix"` at WARN.
pub fn log_tier_formats(cfg: &Config) {
    for (key, format, status) in tier_formats(cfg).unwrap_or_default() {
        if status == SupportStatus::Experimental {
            tracing::warn!(
                event = "support_matrix",
                status = status.as_str(),
                key,
                format,
                "support matrix: tier format {format} ({key}) is experimental"
            );
        }
    }
}

/// A KV format no kernel provider can run yet: the reason code and the message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvFormatUnavailable {
    pub code: &'static str,
    pub message: String,
}

/// Refuses, before binding (exit 1), the KV formats whose kernel group the provider lacks (P6b
/// S-1, S-5, S-7): TurboQuant L0 pages (`kv.dtype: tq4|tq2`) on a GPU backend and the ladder's
/// L0 step need the ABI v2.10 mixed-format paged attention, which no GPU provider implements
/// yet (`kv_tq_unavailable`; plan Task 12 replaces that refusal); a lower tier not stored at
/// the L0 format needs the ABI v2.11 KV transcode (`kv_transcode_unavailable`). `transcode` is
/// `library`: whether the loaded kernel library has it (`None` before the library is loaded: the
/// transcode checks wait for the second call); the ladder's L1/L2 rungs stay refused until
/// their rewrites run on the device transcode too.
pub fn kv_format_availability(
    cfg: &Config,
    library: Option<bool>,
) -> Result<(), KvFormatUnavailable> {
    let kv = &cfg.kv;
    let tq = |message: String| KvFormatUnavailable {
        code: "kv_tq_unavailable",
        message,
    };
    // The CPU reference provider reads TurboQuant pages (`cpu::tq_attention`, P6b S-5); a GPU
    // provider needs the v2.10 mixed-format paged attention (plan Task 12).
    if kv.dtype.is_turboquant() && vendor(cfg) != "cpu" {
        return Err(tq(format!(
            "kv.dtype {} needs the ABI v2.10 mixed-format paged attention, which no GPU kernel \
             provider implements yet (the cpu backend runs it)",
            kv.dtype.as_str()
        )));
    }
    if kv.ladder.enabled && kv.ladder.l0 {
        return Err(tq(
            "kv.ladder.l0 (with kv.ladder.enabled) needs the ABI v2.10 mixed-format paged \
             attention, which no kernel provider implements yet; set kv.ladder.l0: false"
                .to_string(),
        ));
    }
    let transcode = |what: String| KvFormatUnavailable {
        code: "kv_transcode_unavailable",
        message: format!(
            "{what} needs the ABI v2.11 KV transcode, which the kernel library does not provide"
        ),
    };
    let l0 = kv.dtype.as_str();
    let l0_rung = turbine_kv::codec::tier_rung("l0", l0);
    for (key, format) in tier_format_keys(cfg)
        .into_iter()
        .filter(|(key, _)| *key != "kv.ladder.max_format")
    {
        // Below TurboQuant L0 pages a tier stores them as they are (`l0`): the tier codecs
        // encode from BF16 / FP8 pages only. Any other lossier tier needs the library's
        // transcode, once it is known.
        if kv.dtype.is_turboquant() && format != "l0"
            || library == Some(false) && turbine_kv::codec::tier_rung(format, l0) != l0_rung
        {
            return Err(transcode(format!("{key} {format}")));
        }
    }
    if kv.ladder.enabled {
        return Err(KvFormatUnavailable {
            code: "kv_transcode_unavailable",
            message: "kv.ladder.enabled: the ladder's L1/L2 rewrites are not wired to the device \
                      transcode yet"
                .to_string(),
        });
    }
    Ok(())
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

    /// Phase 6a S-13: the KV column is the configured `kv.dtype`; FP8 KV is supported on `amd`
    /// gfx1201 Llama / OLMoE (Task 24 proof), refused elsewhere on `amd` (exit 2 naming
    /// `kv.dtype`), and experimental on the cpu backend.
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
        // Before discovery the arch is unknown: the gfx1201 row (supported after the Task 24
        // proof) is the best any card could give.
        let d = before_discovery(&cfg).unwrap();
        assert_eq!(d.status.as_str(), "supported", "{:?}", d.key);
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

    /// P6b S-2: the lower-tier formats in use are ordered by the `kv_format` registry (a tier
    /// never more precise than the tier above it, exit 2 naming both keys; the ladder's
    /// `max_format` lossy) and resolved against `TIER_FORMAT_REFUSALS` (TurboQuant refused
    /// naming its key, exit 2, also under `--check-config`); the KV formats that need the ABI
    /// v2.10 group are refused before binding with their reason code; penalties default from
    /// the codecs. Breaks if a tier format goes unchecked or a lossy format is silently served
    /// as `l0`.
    #[test]
    fn tier_formats_and_availability() {
        use turbine_core::config::{KvDtypeChoice, ModuleName};
        let name = |s: &str| ModuleName::new(s).unwrap();
        let llama = model_dir(
            "llama-tier-formats",
            serde_json::json!({"architectures": ["LlamaForCausalLM"]}),
        );
        let cfg = config("hip", llama.path());
        assert_eq!(
            tier_formats(&cfg).unwrap(),
            vec![("kv.cpu.format", "l0", SupportStatus::Supported)]
        );
        assert_eq!(kv_format_availability(&cfg, None), Ok(()));
        assert_eq!(lossy_penalty(&cfg, "tq4"), 0.5);
        assert_eq!(lossy_penalty(&cfg, "l0"), 0.0);
        let mut over = config("hip", llama.path());
        over.kv.lossy_penalty = Some([(name("tq4"), 0.7)].into_iter().collect());
        assert_eq!(lossy_penalty(&over, "tq4"), 0.7);
        assert_eq!(lossy_penalty(&over, "tq2"), 1.0);

        // Ordering: L2 more precise than L1 (spec AC: nvme fp8_e4m3 under cpu tq4) names both
        // keys, before the TurboQuant refusal of kv.cpu.format.
        let mut order = config("hip", llama.path());
        order.kv.cpu.format = name("tq4");
        order.kv.nvme.enabled = true;
        order.kv.nvme.format = name("fp8_e4m3");
        let err = before_discovery(&order).unwrap_err();
        assert_eq!(err.key(), Some("kv.nvme.format"), "{err}");
        assert!(err.to_string().contains("kv.cpu.format"), "{err}");
        // With L1 off, L2 is compared with L0 (kv.dtype).
        order.kv.cpu.enabled = false;
        assert!(tier_formats(&order).is_ok());
        for (dtype, cpu) in [
            (KvDtypeChoice::Tq4, "fp8_e4m3"),
            (KvDtypeChoice::Tq2, "tq4"),
        ] {
            let mut c = config("cpu", llama.path());
            c.kv.dtype = dtype;
            c.kv.cpu.format = name(cpu);
            let err = tier_formats(&c).unwrap_err();
            assert_eq!(err.key(), Some("kv.cpu.format"), "{err}");
            assert!(err.to_string().contains("kv.dtype"), "{err}");
        }
        // `fp8_e4m3` below an FP8 L0 is the L0 format; `l0` is always accepted.
        for (dtype, cpu) in [
            (KvDtypeChoice::Fp8E4m3, "fp8_e4m3"),
            (KvDtypeChoice::Fp8E4m3, "l0"),
            (KvDtypeChoice::Tq2, "l0"),
        ] {
            let mut c = config("cpu", llama.path());
            c.kv.dtype = dtype;
            c.kv.cpu.format = name(cpu);
            assert!(tier_formats(&c).is_ok(), "{dtype:?} {cpu}");
        }

        let mut fp8 = config("hip", llama.path());
        fp8.kv.cpu.format = name("fp8_e4m3");
        assert_eq!(
            tier_formats(&fp8).unwrap(),
            vec![("kv.cpu.format", "fp8_e4m3", SupportStatus::Experimental)]
        );
        assert!(before_discovery(&fp8).is_ok());
        // Before the library is loaded nothing is known; a library with the v2.11 transcode
        // (or the cpu reference) accepts it, one without refuses it.
        assert_eq!(kv_format_availability(&fp8, None), Ok(()));
        assert_eq!(kv_format_availability(&fp8, Some(true)), Ok(()));
        let err = kv_format_availability(&fp8, Some(false)).unwrap_err();
        assert_eq!(err.code, "kv_transcode_unavailable", "{err:?}");
        assert!(err.message.contains("kv.cpu.format fp8_e4m3"), "{err:?}");
        // fp8_e4m3 below an FP8 L0 is the L0 format: nothing to transcode.
        fp8.kv.dtype = KvDtypeChoice::Fp8E4m3;
        assert_eq!(kv_format_availability(&fp8, Some(false)), Ok(()));

        let mut tq = config("hip", llama.path());
        tq.kv.nvme.enabled = true;
        tq.kv.nvme.format = name("tq4");
        let err = before_discovery(&tq).unwrap_err();
        assert_eq!(err.key(), Some("kv.nvme.format"), "{err}");
        assert!(err.to_string().contains("phase-6b-kv-compression"), "{err}");
        // A disabled tier's format is not in use.
        tq.kv.nvme.enabled = false;
        assert!(before_discovery(&tq).is_ok());

        let mut ladder = config("hip", llama.path());
        ladder.kv.ladder.enabled = true;
        let err = before_discovery(&ladder).unwrap_err();
        assert_eq!(err.key(), Some("kv.ladder.max_format"), "{err}");
        ladder.kv.ladder.max_format = name("l0");
        let err = before_discovery(&ladder).unwrap_err();
        assert_eq!(err.key(), Some("kv.ladder.max_format"), "{err}");
        assert!(err.to_string().contains("lossy"), "{err}");
        ladder.kv.ladder.max_format = name("fp8_e4m3");
        assert!(before_discovery(&ladder).is_ok());
        assert_eq!(
            kv_format_availability(&ladder, None).unwrap_err().code,
            "kv_tq_unavailable"
        );
        ladder.kv.ladder.l0 = false;
        assert_eq!(
            kv_format_availability(&ladder, None).unwrap_err().code,
            "kv_transcode_unavailable"
        );

        // TurboQuant L0 pages (P6b S-5): the cpu backend runs them (`experimental`), a GPU
        // backend is refused (`kv_tq_unavailable`; the support matrix first, naming kv.dtype),
        // and a lower tier below them stores them as they are.
        for dtype in [KvDtypeChoice::Tq4, KvDtypeChoice::Tq2] {
            let mut l0 = config("cpu", llama.path());
            l0.kv.dtype = dtype;
            assert_eq!(kv_format_availability(&l0, None), Ok(()));
            assert_eq!(format_columns(&l0).1.as_str(), dtype.as_str());
            let first = before_discovery(&l0).unwrap();
            assert_eq!(first.status, SupportStatus::Experimental, "{}", first.key);
            l0.kv.cpu.format = name(dtype.as_str());
            let err = kv_format_availability(&l0, None).unwrap_err();
            assert_eq!(err.code, "kv_transcode_unavailable", "{err:?}");

            let mut hip = config("hip", llama.path());
            hip.kv.dtype = dtype;
            let err = kv_format_availability(&hip, None).unwrap_err();
            assert_eq!(err.code, "kv_tq_unavailable", "{err:?}");
            let err = before_discovery(&hip).unwrap_err();
            assert_eq!(err.key(), Some("kv.dtype"), "{err}");
        }
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
        // No `quantization_config`: BF16 weights; nothing configured: BF16 KV.
        assert_eq!(
            format_columns(&cfg),
            (WeightFormatColumn::Bf16, KvFormatColumn::Bf16)
        );
        // A compressed-tensors FP8 checkpoint: the detected column.
        let fp8 = model_dir(
            "fmt-fp8",
            serde_json::json!({"architectures": ["LlamaForCausalLM"],
                "quantization_config": {"quant_method": "compressed-tensors",
                    "format": "float-quantized", "ignore": ["lm_head"],
                    "config_groups": {"group_0": {"targets": ["Linear"],
                        "weights": {"num_bits": 8, "type": "float", "strategy": "channel"}}}}}),
        );
        assert_eq!(
            format_columns(&config("hip", fp8.path())),
            (WeightFormatColumn::Fp8, KvFormatColumn::Bf16)
        );
        let key = key_before_discovery(&cfg, WeightFormatColumn::Fp8, KvFormatColumn::Bf16);
        assert_eq!(key.to_string(), "amd/*/LlamaForCausalLM/fp8/bf16/none");
        // Supported on gfx1201 Llama after its gate; another family is refused.
        assert_eq!(support::check(key).unwrap().status.as_str(), "supported");
        let qwen = model_dir(
            "fmt-qwen",
            serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}),
        );
        let key = key_before_discovery(
            &config("hip", qwen.path()),
            WeightFormatColumn::Fp8,
            KvFormatColumn::Bf16,
        );
        let err = support::check(key).unwrap_err();
        assert_eq!(err.key(), Some("model.path"));

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
