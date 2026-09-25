//! NVIDIA discovery through NVML (`nvml-wrapper`, which loads the library at runtime).

use std::path::{Path, PathBuf};

use nvml_wrapper::Nvml;
use nvml_wrapper::error::NvmlError;
use turbine_core::types::{DeviceId, MemoryKind, Vendor};

use super::DiscoveryBackend;
use crate::inventory::{DeviceInfo, DeviceMemoryInfo};

pub(crate) struct NvmlBackend {
    library: PathBuf,
    meminfo_path: PathBuf,
}

impl NvmlBackend {
    pub(crate) fn new(library: PathBuf, meminfo_path: PathBuf) -> Self {
        NvmlBackend {
            library,
            meminfo_path,
        }
    }
}

impl DiscoveryBackend for NvmlBackend {
    fn vendor(&self) -> Vendor {
        Vendor::Nvidia
    }

    fn discover(&mut self) -> Result<Vec<DeviceInfo>, String> {
        let nvml = Nvml::builder()
            .lib_path(self.library.as_os_str())
            .init()
            .map_err(|e| format!("{}: {e}", self.library.display()))?;
        let driver_version = nvml.sys_driver_version().ok();
        let count = nvml
            .device_count()
            .map_err(|e| format!("nvmlDeviceGetCount_v2 failed: {e}"))?;
        let mut devices = Vec::new();
        for i in 0..count {
            let device = match nvml.device_by_index(i) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(vendor = "nvidia", vendor_index = i, error = %e, "cannot open NVML device handle");
                    devices.push(DeviceInfo {
                        index: DeviceId(0),
                        vendor: Vendor::Nvidia,
                        vendor_index: i,
                        name: "unknown NVIDIA device".to_string(),
                        uuid: None,
                        pci_bus_id: None,
                        arch: None,
                        driver_version: driver_version.clone(),
                        memory: memory_info(Err(e), &self.meminfo_path),
                    });
                    continue;
                }
            };
            let arch = device
                .cuda_compute_capability()
                .ok()
                .map(|cc| format!("sm_{}{}", cc.major, cc.minor));
            devices.push(DeviceInfo {
                index: DeviceId(0),
                vendor: Vendor::Nvidia,
                vendor_index: i,
                name: device
                    .name()
                    .unwrap_or_else(|_| "unknown NVIDIA device".to_string()),
                uuid: device.uuid().ok(),
                pci_bus_id: device.pci_info().ok().map(|p| normalize_bus_id(&p.bus_id)),
                arch,
                driver_version: driver_version.clone(),
                memory: memory_info(device.memory_info().map(|m| m.total), &self.meminfo_path),
            });
        }
        Ok(devices)
    }
}

/// Map NVML's memory answer to the inventory record. `NotSupported` means the device has no
/// dedicated framebuffer (GB10 unified memory): total = host `MemTotal`, shared with the host.
pub(crate) fn memory_info(total: Result<u64, NvmlError>, meminfo_path: &Path) -> DeviceMemoryInfo {
    match total {
        Ok(total_bytes) => DeviceMemoryInfo {
            kind: MemoryKind::Dedicated,
            total_bytes,
            shared_with_host: false,
        },
        Err(NvmlError::NotSupported) => {
            let total_bytes = std::fs::read_to_string(meminfo_path)
                .ok()
                .and_then(|text| mem_total_bytes(&text))
                .unwrap_or_else(|| {
                    tracing::warn!(path = %meminfo_path.display(), "cannot read MemTotal for a unified-memory device");
                    0
                });
            DeviceMemoryInfo {
                kind: MemoryKind::Unified,
                total_bytes,
                shared_with_host: true,
            }
        }
        Err(e) => {
            tracing::warn!(vendor = "nvidia", error = %e, "NVML memory query failed");
            DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 0,
                shared_with_host: false,
            }
        }
    }
}

/// `MemTotal:  126877932 kB` → bytes.
pub(crate) fn mem_total_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let mut parts = line["MemTotal:".len()..].split_whitespace();
    let kib: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => kib.checked_mul(1024),
        _ => None,
    }
}

/// NVML `00000000:0F:01.0` → `0000:0f:01.0` (same form as amd-smi).
pub(crate) fn normalize_bus_id(raw: &str) -> String {
    match raw.split_once(':') {
        Some((domain, rest)) => match u32::from_str_radix(domain, 16) {
            Ok(d) => format!("{d:04x}:{}", rest.to_ascii_lowercase()),
            Err(_) => raw.to_ascii_lowercase(),
        },
        None => raw.to_ascii_lowercase(),
    }
}
