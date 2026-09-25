//! Inventory types serialized at `GET /turbine/v1/devices`, and the `turbine_devices` gauge.

use std::path::PathBuf;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use serde::Serialize;
use turbine_core::types::{DeviceId, MemoryKind, Vendor};
use turbine_observability::MetricsRegistry;

#[derive(Serialize, Clone, Debug)]
pub struct DeviceInventory {
    pub devices: Vec<DeviceInfo>,
    pub backends: Vec<BackendReport>,
}

impl DeviceInventory {
    pub fn count(&self, vendor: Vendor) -> usize {
        self.devices.iter().filter(|d| d.vendor == vendor).count()
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct DeviceInfo {
    /// Global index: NVIDIA devices first (NVML order), then AMD (amd-smi order).
    pub index: DeviceId,
    pub vendor: Vendor,
    pub vendor_index: u32,
    pub name: String,
    pub uuid: Option<String>,
    pub pci_bus_id: Option<String>,
    /// `sm_<major><minor>` (NVIDIA) or `gfx<target>` (AMD).
    pub arch: Option<String>,
    pub driver_version: Option<String>,
    pub memory: DeviceMemoryInfo,
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct DeviceMemoryInfo {
    pub kind: MemoryKind,
    pub total_bytes: u64,
    pub shared_with_host: bool,
}

#[derive(Serialize, Clone, Debug)]
pub struct BackendReport {
    pub vendor: Vendor,
    pub status: BackendStatus,
    pub detail: String,
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum BackendStatus {
    Ok,
    Unavailable,
    Timeout,
}

impl BackendStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendStatus::Ok => "ok",
            BackendStatus::Unavailable => "unavailable",
            BackendStatus::Timeout => "timeout",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("cannot load {path}: {detail}")]
    ExplicitLibrary { path: PathBuf, detail: String },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct VendorLabels {
    vendor: String,
}

/// `turbine_devices{vendor}`: devices discovered per vendor.
#[derive(Clone, Debug)]
pub struct DeviceMetrics {
    devices: Family<VendorLabels, Gauge>,
}

impl DeviceMetrics {
    pub fn register(reg: &MetricsRegistry) -> Self {
        let devices = reg.register(
            "turbine_devices",
            "GPU devices discovered at startup, per vendor",
            Family::<VendorLabels, Gauge>::default(),
        );
        DeviceMetrics { devices }
    }

    /// Set the gauge for every vendor (0 when none were found).
    pub fn record(&self, inventory: &DeviceInventory) {
        for vendor in [Vendor::Nvidia, Vendor::Amd] {
            let n = i64::try_from(inventory.count(vendor)).unwrap_or(i64::MAX);
            self.devices
                .get_or_create(&VendorLabels {
                    vendor: vendor.as_str().to_string(),
                })
                .set(n);
        }
    }
}
