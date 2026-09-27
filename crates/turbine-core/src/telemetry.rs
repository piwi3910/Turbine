//! Telemetry vocabulary (contract §3.6, P3 S-5) shared by `turbine-device` (producer) and
//! `turbine-reliability` (consumer), so the reliability crate never depends on the device crate.

use serde::Serialize;

use crate::types::DeviceId;

/// `Unavailable`: the source never worked (library missing, file absent) — its signals are
/// omitted. `Stale`: the source worked and then missed its deadline — its signals are flagged.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceStatus {
    #[default]
    Ok,
    Unavailable,
    Stale,
}

/// Host readings from `/proc` taken on the fast tick.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct HostSample {
    pub mem_available_bytes: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub swap_free_bytes: Option<u64>,
    /// Cumulative `pswpin` from `/proc/vmstat`; the rate is derived from deltas.
    pub pswpin_total: Option<u64>,
    pub psi_memory_some_avg10: Option<f64>,
    pub status: SourceStatus,
}

/// Throttle / clock-event reasons reported by the vendor library.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ThrottleReasons {
    pub thermal: bool,
    pub power: bool,
    pub other: bool,
}

/// Per-device vendor readings taken on the vendor tick.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeviceSample {
    pub device: DeviceId,
    /// Dedicated-memory devices only.
    pub memory_used_bytes: Option<u64>,
    /// Dedicated-memory devices only.
    pub memory_free_bytes: Option<u64>,
    pub temperature_c: Option<f64>,
    pub slowdown_temperature_c: Option<f64>,
    pub clock_mhz: Option<u32>,
    pub power_watts: Option<f64>,
    pub utilization: Option<f64>,
    pub throttle: ThrottleReasons,
    pub status: SourceStatus,
}

impl DeviceSample {
    /// A sample with no readings: before the first vendor tick, or for an unavailable or stale
    /// device.
    pub fn empty(device: DeviceId, status: SourceStatus) -> Self {
        DeviceSample {
            device,
            memory_used_bytes: None,
            memory_free_bytes: None,
            temperature_c: None,
            slowdown_temperature_c: None,
            clock_mhz: None,
            power_watts: None,
            utilization: None,
            throttle: ThrottleReasons::default(),
            status,
        }
    }
}

/// Readings from the reservation ledger and admission queue, taken on every fast tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct LedgerSample {
    pub kv_utilization: f64,
    pub queue_fill: f64,
}

/// NVMe tier readings (P4).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StorageSample {
    pub queue_depth: u32,
    pub max_queue_depth: u32,
    pub p99_latency_s: f64,
    pub calibration_latency_s: f64,
}

/// One published telemetry sample (the lock-free latest value the controller reads).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TelemetrySample {
    /// Monotonic time of the fast tick that produced this sample, from the injected `Clock`.
    pub at_mono_ns: u64,
    pub host: HostSample,
    pub devices: Vec<DeviceSample>,
    pub ledger: LedgerSample,
    pub storage: Option<StorageSample>,
}

/// Read on the fast tick; implemented over the reliability ledger and admission queue.
pub trait LedgerProbe: Send + Sync {
    fn kv_utilization(&self) -> f64;
    fn queue_fill(&self) -> f64;
}

/// Read on the fast tick (P4): the L2 NVMe tier's queue depth and latency as the KV orchestrator
/// last published them; `None` while no L2 tier exists.
pub trait StorageProbe: Send + Sync {
    fn storage(&self) -> Option<StorageSample>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_device_sample_carries_only_status() {
        let s = DeviceSample::empty(DeviceId(3), SourceStatus::Unavailable);
        assert_eq!(s.device, DeviceId(3));
        assert_eq!(s.status, SourceStatus::Unavailable);
        assert!(s.temperature_c.is_none() && s.memory_free_bytes.is_none());
        assert_eq!(s.throttle, ThrottleReasons::default());
        assert_eq!(
            serde_json::to_string(&SourceStatus::Unavailable).unwrap(),
            "\"unavailable\""
        );
    }
}
