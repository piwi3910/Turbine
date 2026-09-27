//! Vendor telemetry backends. Each registered `device_discovery` kind (Phase 2m registry,
//! [`crate::discovery::registry`]) contributes the telemetry backend of its vendor library
//! ([`super::nvml::NvmlTelemetry`], [`super::amd_smi::AmdSmiTelemetry`]), both loaded at
//! runtime. Adding a vendor is a new discovery kind plus its telemetry file, never a branch here.

use crate::discovery::{DiscoveryOptions, registry};

use super::VendorTelemetry;

/// One backend per registered discovery kind whose vendor library loads, in registry order; a
/// missing library contributes nothing (its devices stay `unavailable`, logged once here at
/// WARN).
pub fn vendor_backends(opts: &DiscoveryOptions) -> Vec<Box<dyn VendorTelemetry>> {
    let mut out: Vec<Box<dyn VendorTelemetry>> = Vec::new();
    for kind in registry().iter() {
        match kind.telemetry(opts) {
            Ok(backend) => out.push(backend),
            Err(e) => {
                tracing::warn!(
                    event = "telemetry_stale",
                    reason = "unavailable",
                    vendor = kind.vendor().as_str(),
                    kind = kind.name(),
                    error = %e
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn missing_libraries_yield_no_backends() {
        let opts = DiscoveryOptions {
            nvml_library: Some(PathBuf::from("/nonexistent/a")),
            amd_smi_library: Some(PathBuf::from("/nonexistent/libamd_smi.so")),
            ..DiscoveryOptions::default()
        };
        assert!(vendor_backends(&opts).is_empty());
    }
}
