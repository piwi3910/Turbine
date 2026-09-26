//! Execution backends (Phase 2m S-7, contract §24): one file per backend, each opening the
//! device memory and the kernel providers `execution.backend` names. Everything backend- or
//! vendor-specific (library search, device matching, sticky device errors, notes on the
//! kernels it selected) lives behind [`ExecutionBackend`]; the server only looks the backend up
//! in [`registry`] and opens it.

use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::{Module, Registry};
use turbine_core::types::{DeviceId, MemoryKind};
use turbine_device::{DeviceInfo, DeviceInventory};
use turbine_tensor::DeviceMemory;

use crate::cards::CardProfile;
use crate::{KernelError, KernelProvider, ProviderId, Selection, ShimContext};

pub mod cpu;
pub mod hip;

pub use cpu::CpuBackend;
pub use hip::HipBackend;

/// What opening a backend needs from the configuration and the host.
pub struct BackendRequest<'a> {
    /// `execution.device`: the global inventory index.
    pub device: DeviceId,
    /// `execution.kernel_library`; `None` = the backend's own search order.
    pub kernel_library: Option<&'a Path>,
    pub inventory: &'a DeviceInventory,
    /// Host memory figures (`/proc/meminfo` in production, a fixture in tests).
    pub meminfo: &'a Path,
    /// `execution.card_profile`: `auto` (the device architecture's profile) or a profile name.
    /// Backends without card profiles (`cpu`) ignore it.
    pub card_profile: &'a str,
}

/// An opened backend: device memory, kernel providers in selection order, and the context
/// decode graphs capture on.
pub struct OpenedBackend {
    pub mem: Arc<dyn DeviceMemory>,
    pub providers: Vec<Arc<dyn KernelProvider>>,
    pub order: Vec<ProviderId>,
    pub memory_kind: MemoryKind,
    /// The kernel-library context of a shim backend (`None` on `cpu`).
    pub context: Option<Arc<ShimContext>>,
    /// The context decode graphs capture on: `Some` when the kernel library exports the ABI v2.1
    /// graph functions.
    pub graphs: Option<Arc<ShimContext>>,
    /// The device the backend runs on (`None` on `cpu`, which runs on the host).
    pub device: Option<DeviceInfo>,
    /// The card profile of the device (`None` on `cpu`).
    pub card: Option<&'static CardProfile>,
}

/// Why a backend could not be opened: exit 1 with this message.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("{0}")]
    Startup(String),
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

/// Something a backend has to say about the kernels selected on it, logged at INFO by the
/// server as `event=<event>` with `fields` and `message`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendNote {
    pub event: &'static str,
    pub fields: Vec<(&'static str, String)>,
    pub message: &'static str,
}

/// One execution backend (`execution.backend`).
pub trait ExecutionBackend: Module {
    /// The support-matrix vendor column: `cpu` for the host, else the device vendor (`amd`).
    fn vendor(&self) -> &'static str;
    /// Opens device memory and the kernel providers on `req.device`.
    fn open(&self, req: &BackendRequest<'_>) -> Result<OpenedBackend, BackendError>;
    /// Device error names after which the context is corrupted and every later call fails
    /// (P3 "sticky" errors); device messages start with the runtime's error name.
    fn sticky_error_prefixes(&self) -> &'static [&'static str] {
        &[]
    }
    /// Notes on the kernel selections made on this backend with the opened `card` profile
    /// (e.g. a slower fallback path than the profile prefers).
    fn selection_notes(
        &self,
        _card: Option<&CardProfile>,
        _selections: &[Selection],
    ) -> Vec<BackendNote> {
        Vec::new()
    }
}

static BACKENDS: Registry<dyn ExecutionBackend> =
    Registry::new("execution_backend", &[&CpuBackend, &HipBackend]);

/// The registered execution backends: `cpu`, `hip`.
pub fn registry() -> &'static Registry<dyn ExecutionBackend> {
    &BACKENDS
}

/// True when `message` starts with a sticky device error name of any registered backend.
pub(crate) fn is_sticky_message(message: &str) -> bool {
    registry().iter().any(|b| {
        b.sticky_error_prefixes()
            .iter()
            .any(|name| message.starts_with(name))
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use turbine_device::DeviceInventory;

    use super::*;

    fn meminfo_fixture(name: &str, text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "turbine-kernels-backends-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meminfo");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn cpu_and_hip_registered() {
        let reg = registry();
        assert_eq!(reg.point(), "execution_backend");
        assert_eq!(reg.names(), ["cpu", "hip"]);
        let vendors: Vec<&str> = reg.iter().map(|b| b.vendor()).collect();
        assert_eq!(vendors, ["cpu", "amd"]);

        let meminfo = meminfo_fixture(
            "cpu",
            "MemTotal:       2097152 kB\nMemAvailable:   1048576 kB\n",
        );
        let inventory = DeviceInventory {
            devices: Vec::new(),
            backends: Vec::new(),
        };
        let opened = reg
            .get("cpu")
            .unwrap()
            .open(&BackendRequest {
                device: DeviceId(0),
                kernel_library: None,
                inventory: &inventory,
                meminfo: &meminfo,
                card_profile: "auto",
            })
            .unwrap();
        let ids: Vec<&str> = opened.providers.iter().map(|p| p.id().0).collect();
        assert_eq!(ids, ["cpu-reference"]);
        assert_eq!(opened.order, [ProviderId("cpu-reference")]);
        assert_eq!(opened.mem.mem_info().unwrap().total_bytes, 1 << 30);
        assert_eq!(opened.memory_kind, MemoryKind::Dedicated);
        assert!(opened.context.is_none() && opened.graphs.is_none() && opened.device.is_none());
        assert!(opened.card.is_none());
    }
}
