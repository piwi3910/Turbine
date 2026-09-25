//! Discovery orchestration: every backend runs on its own thread with a deadline, results
//! are merged into one inventory with a stable global index, and each outcome is logged.

mod amd_smi;
mod nvml;

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use turbine_core::config::DevicesConfig;
use turbine_core::types::{DeviceId, Vendor};

use crate::inventory::{BackendReport, BackendStatus, DeviceInfo, DeviceInventory, DiscoveryError};

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

/// Discover NVIDIA then AMD devices. `Err` only when an explicitly configured library cannot be loaded.
pub fn discover(opts: &DiscoveryOptions) -> Result<DeviceInventory, DiscoveryError> {
    discover_with_defaults(opts, NVML_DEFAULT_LIBRARY, AMD_SMI_DEFAULT_LIBRARY)
}

fn discover_with_defaults(
    opts: &DiscoveryOptions,
    nvml_default: &str,
    amd_smi_default: &str,
) -> Result<DeviceInventory, DiscoveryError> {
    for explicit in [&opts.nvml_library, &opts.amd_smi_library]
        .into_iter()
        .flatten()
    {
        check_loadable(explicit)?;
    }
    let nvml_lib = opts
        .nvml_library
        .clone()
        .unwrap_or_else(|| PathBuf::from(nvml_default));
    let amd_lib = opts
        .amd_smi_library
        .clone()
        .unwrap_or_else(|| PathBuf::from(amd_smi_default));
    let backends: Vec<Box<dyn DiscoveryBackend>> = vec![
        Box::new(nvml::NvmlBackend::new(nvml_lib, opts.meminfo_path.clone())),
        Box::new(amd_smi::AmdSmiBackend::new(amd_lib)),
    ];
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
