//! NVML telemetry (the `nvml` discovery kind's vendor tick), loaded at runtime through
//! `nvml-wrapper`.

use std::path::Path;

use nvml_wrapper::Nvml;
use turbine_core::telemetry::{DeviceSample, SourceStatus, ThrottleReasons};
use turbine_core::types::Vendor;

use super::VendorTelemetry;
use crate::inventory::DeviceInfo;

/// NVML telemetry for every NVIDIA device (`vendor_index` = NVML index).
pub struct NvmlTelemetry {
    nvml: Nvml,
}

impl NvmlTelemetry {
    /// `path` = `devices.nvml_library`; `None` = the loader's `libnvidia-ml.so.1`.
    pub fn open(path: Option<&Path>) -> Result<Self, String> {
        let nvml = match path {
            Some(p) => Nvml::builder().lib_path(p.as_os_str()).init(),
            None => Nvml::init(),
        };
        nvml.map(|nvml| NvmlTelemetry { nvml })
            .map_err(|e| format!("NVML: {e}"))
    }
}

impl VendorTelemetry for NvmlTelemetry {
    fn vendor(&self) -> Vendor {
        Vendor::Nvidia
    }

    fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String> {
        use nvml_wrapper::bitmasks::device::ThrottleReasons as R;
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor, TemperatureThreshold};
        let d = self
            .nvml
            .device_by_index(device.vendor_index)
            .map_err(|e| format!("nvmlDeviceGetHandleByIndex({}): {e}", device.vendor_index))?;
        let temperature_c = d.temperature(TemperatureSensor::Gpu).ok().map(f64::from);
        let clock_mhz = d.clock_info(Clock::SM).ok();
        if temperature_c.is_none() && clock_mhz.is_none() {
            return Err("NVML returned neither temperature nor clock".into());
        }
        // GB10 (unified memory) answers NotSupported; the sampler drops memory for unified
        // devices anyway.
        let memory = d.memory_info().ok();
        let reasons = d.current_throttle_reasons().unwrap_or(R::empty());
        Ok(DeviceSample {
            device: device.index,
            memory_used_bytes: memory.as_ref().map(|m| m.used),
            memory_free_bytes: memory.as_ref().map(|m| m.free),
            temperature_c,
            slowdown_temperature_c: d
                .temperature_threshold(TemperatureThreshold::Slowdown)
                .ok()
                .map(f64::from),
            clock_mhz,
            // NVML reports milliwatts.
            power_watts: d.power_usage().ok().map(|mw| f64::from(mw) / 1000.0),
            utilization: d.utilization_rates().ok().map(|u| f64::from(u.gpu) / 100.0),
            throttle: ThrottleReasons {
                thermal: reasons.intersects(R::SW_THERMAL_SLOWDOWN | R::HW_THERMAL_SLOWDOWN),
                power: reasons.intersects(R::SW_POWER_CAP | R::HW_POWER_BRAKE_SLOWDOWN),
                other: reasons.intersects(R::HW_SLOWDOWN),
            },
            status: SourceStatus::Ok,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_yields_error_not_panic() {
        assert!(NvmlTelemetry::open(Some(Path::new("/nonexistent/libnvidia-ml.so.1"))).is_err());
    }
}
