//! GPU discovery and the static device inventory (contract §5.1).
//!
//! NVIDIA devices are discovered through NVML (`nvml-wrapper`), AMD devices through a minimal
//! runtime-loaded FFI to `libamd_smi.so`. Nothing links a GPU library at build time. Each
//! vendor's discovery is a [`DiscoveryKind`] in the `device_discovery` registry (Phase 2m,
//! contract §24).
//! [`telemetry`] (P3) samples host `/proc` files, the reservation ledger and the vendor
//! libraries on two cadences.

pub mod discovery;
mod host;
mod inventory;
pub mod telemetry;
pub mod topology;

pub use discovery::{DiscoveryBackend, DiscoveryKind, DiscoveryOptions, discover, run_backends};
pub use host::host_mem_available;
pub use inventory::{
    BackendReport, BackendStatus, DeviceInfo, DeviceInventory, DeviceMemoryInfo, DeviceMetrics,
    DiscoveryError,
};

/// Every registry of this crate passes the shared conformance checks (Phase 2m S-1).
#[cfg(test)]
mod registry_conformance {
    #[test]
    fn discovery() {
        let reg = crate::discovery::registry();
        turbine_core::registry::conformance::check(reg).unwrap();
        assert_eq!(reg.names(), ["nvml", "amd_smi"]);
    }
}
