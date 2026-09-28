//! Helpers for the `#[ignore]`d GPU and weights tests (contract §7 `test_support`). Always
//! compiled so every crate's lab tests use the same gating:
//!
//! ```no_run
//! if !turbine_kernels::test_support::require_backend("hip") { return; }
//! let dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MODEL_DIR");
//! ```
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbine_core::types::DeviceId;

use crate::backends::{self, BackendRequest, OpenedBackend};
use crate::{KernelError, ShimContext};

/// The variable naming the backend a lab run tests (`hip` or `cuda`).
const BACKEND_VAR: &str = "TURBINE_TEST_BACKEND";

/// Opens the registered execution backend `name` on the first discovered device of its vendor
/// (device 0 when there is none), with the kernel library `TURBINE_KERNEL_LIBRARY` names (else
/// the backend's own search order). Panics when the backend is not registered or cannot open.
pub fn open_backend(name: &str) -> OpenedBackend {
    let backend = backends::registry()
        .get(name)
        .unwrap_or_else(|| panic!("{}", backends::registry().unknown(name)));
    let inventory = turbine_device::discover(&turbine_device::DiscoveryOptions::default())
        .expect("device discovery");
    let device = inventory
        .devices
        .iter()
        .find(|d| d.vendor.as_str() == backend.vendor())
        .map_or(DeviceId(0), |d| d.index);
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    backend
        .open(&BackendRequest {
            device,
            kernel_library: library.as_deref(),
            inventory: &inventory,
            meminfo: Path::new("/proc/meminfo"),
            card_profile: "auto",
        })
        .unwrap_or_else(|e| panic!("open execution backend {name}: {e}"))
}

/// The kernel-library context of [`open_backend`]`(name)`; panics on a backend without one.
pub fn open_context(name: &str) -> Arc<ShimContext> {
    open_backend(name)
        .context
        .unwrap_or_else(|| panic!("execution backend {name} has no kernel-library context"))
}

/// A context on mocked device `index` (an AMD `gfx942`, vendor index `index`) of the test stub
/// kernel library that exports every optional group up to ABI v2.7 (`stub/stub_shim.c`, built
/// by this crate's build script with the host C compiler). Its "device" memory is host memory
/// and its host-mapped collective steps run on the calling thread, so the `hostmem` collective
/// backend can be tested without a GPU. Only usable on the host that built this crate.
pub fn stub_mapped_context(index: u32) -> Arc<ShimContext> {
    use turbine_core::types::{MemoryKind, Vendor};
    use turbine_device::{DeviceInfo, DeviceMemoryInfo};

    let lib = crate::ShimLibrary::load(Path::new(env!("TURBINE_STUB_GFX942_V27")), "hip")
        .expect("the v2.7 stub library loads");
    let device = DeviceInfo {
        index: DeviceId(index),
        vendor: Vendor::Amd,
        vendor_index: index,
        name: "stub".into(),
        uuid: None,
        pci_bus_id: None,
        arch: Some("gfx942".into()),
        driver_version: None,
        memory: DeviceMemoryInfo {
            kind: MemoryKind::Dedicated,
            total_bytes: 1 << 30,
            shared_with_host: false,
        },
    };
    lib.create_context(&device)
        .expect("a stub context on a mocked device")
}

/// A device error a registered backend classifies as sticky (the context is corrupted).
pub fn sticky_device_error(detail: &str) -> KernelError {
    let name = backends::registry()
        .iter()
        .find_map(|b| b.sticky_error_prefixes().first())
        .expect("a registered backend with sticky device errors");
    KernelError::Device {
        message: format!("{name}: {detail}"),
    }
}

/// A device error no registered backend classifies as sticky.
pub fn plain_device_error(detail: &str) -> KernelError {
    KernelError::Device {
        message: format!("device error: {detail}"),
    }
}

/// `TURBINE_TEST_BACKEND` equal to `backend` → true; a different value → prints
/// `SKIP backend=<value>` and returns false; unset → panics naming the variable (a lab run must
/// say which backend it tests, so an ignored test never passes by accident).
pub fn require_backend(backend: &str) -> bool {
    backend_matches(std::env::var(BACKEND_VAR).ok().as_deref(), backend)
}

fn backend_matches(value: Option<&str>, backend: &str) -> bool {
    match value {
        Some(v) if v == backend => true,
        Some(v) => {
            println!("SKIP backend={v}");
            false
        }
        None => panic!("{BACKEND_VAR} is not set; set it to the backend under test (hip or cuda)"),
    }
}

/// The directory named by the environment variable `var` (e.g. `TURBINE_TEST_MODEL_DIR`).
/// Panics when the variable is unset or the directory does not exist: a weights test fails, it
/// never skips and never downloads.
pub fn require_env_dir(var: &str) -> PathBuf {
    let Some(value) = std::env::var_os(var) else {
        panic!("{var} is not set; point it at the provisioned model directory");
    };
    let dir = PathBuf::from(value);
    assert!(
        dir.is_dir(),
        "{var}={} does not exist or is not a directory",
        dir.display()
    );
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_match_skip_and_unset() {
        assert!(backend_matches(Some("hip"), "hip"));
        assert!(!backend_matches(Some("cuda"), "hip"));
        let unset = std::panic::catch_unwind(|| backend_matches(None, "hip"))
            .expect_err("unset must panic");
        let message = unset
            .downcast_ref::<String>()
            .expect("formatted panic message");
        assert!(
            message.contains("TURBINE_TEST_BACKEND is not set"),
            "{message}"
        );
    }

    #[test]
    #[should_panic(expected = "TURBINE_TEST_UNSET_MODEL_DIR_FOR_TESTS is not set")]
    fn unset_model_dir_fails() {
        require_env_dir("TURBINE_TEST_UNSET_MODEL_DIR_FOR_TESTS");
    }
}
