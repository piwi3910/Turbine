//! GPU discovery and the static device inventory (contract §5.1).
//!
//! NVIDIA devices are discovered through NVML (`nvml-wrapper`), AMD devices through a minimal
//! runtime-loaded FFI to `libamd_smi.so`. Nothing links a GPU library at build time.

pub mod discovery;
mod inventory;

pub use discovery::{DiscoveryBackend, DiscoveryOptions, discover, run_backends};
pub use inventory::{
    BackendReport, BackendStatus, DeviceInfo, DeviceInventory, DeviceMemoryInfo, DeviceMetrics,
    DiscoveryError,
};
