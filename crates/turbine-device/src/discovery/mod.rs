//! Discovery orchestration: every registered [`DiscoveryKind`] builds a backend that runs on
//! its own thread with a deadline, results are merged into one inventory with a stable global
//! index, and each outcome is logged.

pub(crate) mod amd_smi;
mod nvml;

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use turbine_core::config::DevicesConfig;
use turbine_core::registry::{Module, Registry};
use turbine_core::types::{DeviceId, Vendor};

use crate::inventory::{BackendReport, BackendStatus, DeviceInfo, DeviceInventory, DiscoveryError};
use crate::telemetry::VendorTelemetry;

/// Library names searched by the platform loader when no explicit path is configured.
const NVML_DEFAULT_LIBRARY: &str = "libnvidia-ml.so.1";
const AMD_SMI_DEFAULT_LIBRARY: &str = "libamd_smi.so";
/// Environment variable naming an explicit amd-smi library (fatal when it fails to load).
pub const AMD_SMI_LIBRARY_ENV: &str = "TURBINE_AMD_SMI_LIBRARY";

#[derive(Clone, Debug)]
pub struct DiscoveryOptions {
    /// Explicit NVML path; `None` = loader search for `libnvidia-ml.so.1`.
    pub nvml_library: Option<PathBuf>,
    /// Explicit amd-smi path; `None` = loader search for `libamd_smi.so`.
    pub amd_smi_library: Option<PathBuf>,
    /// Per-backend deadline (10 s).
    pub deadline: Duration,
    /// Host memory source for unified-memory devices.
    pub meminfo_path: PathBuf,
}

impl Default for DiscoveryOptions {
    fn default() -> Self {
        DiscoveryOptions {
            nvml_library: None,
            amd_smi_library: None,
            deadline: Duration::from_secs(10),
            meminfo_path: PathBuf::from("/proc/meminfo"),
        }
    }
}

impl DiscoveryOptions {
    /// Options from `devices.*`; `TURBINE_AMD_SMI_LIBRARY` applies when `devices.amd_smi_library` is null.
    pub fn from_config(cfg: &DevicesConfig) -> Self {
        let env_amd = std::env::var_os(AMD_SMI_LIBRARY_ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        DiscoveryOptions {
            nvml_library: cfg.nvml_library.clone(),
            amd_smi_library: cfg.amd_smi_library.clone().or(env_amd),
            ..DiscoveryOptions::default()
        }
    }
}

/// One vendor's discovery. Runs on a dedicated thread; may block (the caller enforces the deadline).
pub trait DiscoveryBackend: Send {
    fn vendor(&self) -> Vendor;
    /// Devices in vendor order (`index` is reassigned by [`run_backends`]), or why the backend is unavailable.
    fn discover(&mut self) -> Result<Vec<DeviceInfo>, String>;
}

/// One vendor's discovery as a registered module (`device_discovery`, contract §24): which
/// library it loads and how it builds the [`DiscoveryBackend`] that runs it. Every registered
/// kind runs, in registration order (which fixes the global device index order).
pub trait DiscoveryKind: Module {
    fn vendor(&self) -> Vendor;
    /// The operator-configured library path in `opts` (fatal when it cannot be loaded).
    fn configured_library<'a>(&self, opts: &'a DiscoveryOptions) -> Option<&'a Path>;
    /// The library name the platform loader searches when none is configured.
    fn default_library(&self) -> &'static str;
    fn build(&self, library: PathBuf, opts: &DiscoveryOptions) -> Box<dyn DiscoveryBackend>;
    /// The live telemetry backend (P3 vendor tick) over the same library, or why it cannot
    /// load (its devices then stay `unavailable`).
    fn telemetry(&self, opts: &DiscoveryOptions) -> Result<Box<dyn VendorTelemetry>, String>;
}

struct NvmlKind;

impl Module for NvmlKind {
    fn name(&self) -> &'static str {
        "nvml"
    }
}

impl DiscoveryKind for NvmlKind {
    fn vendor(&self) -> Vendor {
        Vendor::Nvidia
    }
    fn configured_library<'a>(&self, opts: &'a DiscoveryOptions) -> Option<&'a Path> {
        opts.nvml_library.as_deref()
    }
    fn default_library(&self) -> &'static str {
        NVML_DEFAULT_LIBRARY
    }
    fn build(&self, library: PathBuf, opts: &DiscoveryOptions) -> Box<dyn DiscoveryBackend> {
        Box::new(nvml::NvmlBackend::new(library, opts.meminfo_path.clone()))
    }
    fn telemetry(&self, opts: &DiscoveryOptions) -> Result<Box<dyn VendorTelemetry>, String> {
        let t = crate::telemetry::nvml::NvmlTelemetry::open(self.configured_library(opts))?;
        Ok(Box::new(t))
    }
}

struct AmdSmiKind;

impl Module for AmdSmiKind {
    fn name(&self) -> &'static str {
        "amd_smi"
    }
}

impl DiscoveryKind for AmdSmiKind {
    fn vendor(&self) -> Vendor {
        Vendor::Amd
    }
    fn configured_library<'a>(&self, opts: &'a DiscoveryOptions) -> Option<&'a Path> {
        opts.amd_smi_library.as_deref()
    }
    fn default_library(&self) -> &'static str {
        AMD_SMI_DEFAULT_LIBRARY
    }
    fn build(&self, library: PathBuf, _opts: &DiscoveryOptions) -> Box<dyn DiscoveryBackend> {
        Box::new(amd_smi::AmdSmiBackend::new(library))
    }
    fn telemetry(&self, opts: &DiscoveryOptions) -> Result<Box<dyn VendorTelemetry>, String> {
        let t = crate::telemetry::amd_smi::AmdSmiTelemetry::open(self.configured_library(opts))?;
        Ok(Box::new(t))
    }
}

static DISCOVERY: Registry<dyn DiscoveryKind> =
    Registry::new("device_discovery", &[&NvmlKind, &AmdSmiKind]);

/// The registered discovery kinds: `nvml`, then `amd_smi`.
pub fn registry() -> &'static Registry<dyn DiscoveryKind> {
    &DISCOVERY
}

/// Discover the devices of every registered kind (NVIDIA then AMD). `Err` only when an
/// explicitly configured library cannot be loaded.
pub fn discover(opts: &DiscoveryOptions) -> Result<DeviceInventory, DiscoveryError> {
    discover_with_defaults(opts, |kind| PathBuf::from(kind.default_library()))
}

fn discover_with_defaults(
    opts: &DiscoveryOptions,
    default_library: impl Fn(&dyn DiscoveryKind) -> PathBuf,
) -> Result<DeviceInventory, DiscoveryError> {
    for explicit in registry().iter().filter_map(|k| k.configured_library(opts)) {
        check_loadable(explicit)?;
    }
    let backends: Vec<Box<dyn DiscoveryBackend>> = registry()
        .iter()
        .map(|kind| {
            let library = kind
                .configured_library(opts)
                .map_or_else(|| default_library(kind), Path::to_path_buf);
            kind.build(library, opts)
        })
        .collect();
    Ok(run_backends(backends, opts.deadline))
}

/// Fail fast when an operator-configured library path cannot be opened.
fn check_loadable(path: &Path) -> Result<(), DiscoveryError> {
    // SAFETY: loading runs the library's initialisers; NVML and amd-smi have none with
    // preconditions. No symbol is resolved and the handle is dropped (dlclose) immediately,
    // so no pointer into the library outlives this call.
    let loaded = unsafe { libloading::Library::new(path) };
    loaded
        .map(drop)
        .map_err(|e| DiscoveryError::ExplicitLibrary {
            path: path.to_path_buf(),
            detail: loader_error(&e),
        })
}

/// libloading's message plus the platform loader's (`dlerror`) text.
pub(crate) fn loader_error(e: &libloading::Error) -> String {
    match std::error::Error::source(e) {
        Some(source) => format!("{e}: {source}"),
        None => e.to_string(),
    }
}

/// Run every backend on its own thread and wait at most `deadline` (measured from the start)
/// for all of them. A backend that misses the deadline is reported `timeout`; its thread is
/// left detached. Global indices follow backend order, then vendor order.
pub fn run_backends(
    backends: Vec<Box<dyn DiscoveryBackend>>,
    deadline: Duration,
) -> DeviceInventory {
    let started = Instant::now();
    let (tx, rx) = mpsc::channel::<(usize, Result<Vec<DeviceInfo>, String>)>();
    let vendors: Vec<Vendor> = backends.iter().map(|b| b.vendor()).collect();
    let mut results: Vec<Option<Result<Vec<DeviceInfo>, String>>> =
        vendors.iter().map(|_| None).collect();
    let mut pending = 0usize;
    for (slot, mut backend) in backends.into_iter().enumerate() {
        let tx = tx.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("discovery-{}", vendors[slot].as_str()))
            .spawn(move || {
                let outcome = backend.discover();
                let _ = tx.send((slot, outcome));
            });
        match spawned {
            Ok(_) => pending += 1,
            Err(e) => results[slot] = Some(Err(format!("cannot spawn discovery thread: {e}"))),
        }
    }
    drop(tx);
    while pending > 0 {
        let remaining = deadline.saturating_sub(started.elapsed());
        match rx.recv_timeout(remaining) {
            Ok((slot, outcome)) => {
                results[slot] = Some(outcome);
                pending -= 1;
            }
            Err(_) => break,
        }
    }

    let mut devices = Vec::new();
    let mut reports = Vec::new();
    for (vendor, result) in vendors.into_iter().zip(results) {
        let report = match result {
            Some(Ok(found)) => {
                let detail = format!("{} device(s)", found.len());
                for mut d in found {
                    d.index = DeviceId(u32::try_from(devices.len()).unwrap_or(u32::MAX));
                    devices.push(d);
                }
                BackendReport {
                    vendor,
                    status: BackendStatus::Ok,
                    detail,
                }
            }
            Some(Err(detail)) => BackendReport {
                vendor,
                status: BackendStatus::Unavailable,
                detail,
            },
            None => BackendReport {
                vendor,
                status: BackendStatus::Timeout,
                detail: format!(
                    "discovery did not finish within {} s",
                    deadline.as_secs_f64()
                ),
            },
        };
        if report.status == BackendStatus::Ok {
            tracing::info!(
                event = "device_discovery",
                vendor = vendor.as_str(),
                status = report.status.as_str(),
                detail = %report.detail,
                "device discovery backend finished"
            );
        } else {
            tracing::warn!(
                event = "device_discovery",
                vendor = vendor.as_str(),
                status = report.status.as_str(),
                detail = %report.detail,
                "device discovery backend contributed no devices"
            );
        }
        reports.push(report);
    }
    DeviceInventory {
        devices,
        backends: reports,
    }
}

#[cfg(test)]
mod tests;
