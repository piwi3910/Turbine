//! The `rccl` and `nccl` modules of the `collective_backend` registry: one runtime-loaded
//! binding to the NCCL C API ([`super::ffi::NcclApi`]) serves both, because RCCL implements the
//! same API. A [`NcclFlavor`] is everything that differs between them: the name, the vendor,
//! the library file names searched and the oldest version accepted.

use std::path::Path;
use std::sync::Arc;

use turbine_core::config::ParallelConfig;
use turbine_core::registry::Module;
use turbine_core::types::Vendor;

use super::ffi::NcclApi;
use super::{CollectiveBackend, CollectiveError, CollectiveLibrary};

/// What distinguishes one NCCL-API library from another.
#[derive(Debug)]
pub struct NcclFlavor {
    /// The registered name and `backend` label.
    pub name: &'static str,
    pub vendor: Vendor,
    /// File-name prefix identifying this library (`librccl`); a file named for another flavor
    /// is refused.
    pub file_prefix: &'static str,
    /// An environment variable naming the vendor install root (`TURBINE_ROCM_PATH`): when set,
    /// `<root>/lib/<file>` is searched first. ROCm keeps RCCL's device code in `<root>/.kpack`
    /// next to the real `lib` directory, so a library opened through a symlinked or bind-mounted
    /// `lib` (as the lab containers mount `/opt/rocm/rocm/lib`) finds no kernel for the GPU
    /// ("invalid device function").
    pub root_env: Option<(&'static str, &'static str)>,
    /// Library settings applied before the library loads, each only when the operator has not set
    /// it in the environment (the library reads them once, at its first call).
    pub env_defaults: &'static [(&'static str, &'static str)],
    /// Default locations, searched in order after the root, when no explicit path is
    /// configured.
    pub defaults: &'static [&'static str],
    /// The explicit library key of the `parallel` section (`parallel.rccl_library`).
    pub configured: fn(&ParallelConfig) -> Option<&Path>,
    /// `NCCL_VERSION_CODE` (major × 10000 + minor × 100 + patch) of the oldest release accepted.
    pub min_version: i32,
}

/// `NCCL_VERSION_CODE` of the oldest RCCL accepted: the RCCL of ROCm 7.14.1
/// (`/opt/rocm/rocm/include/rccl/rccl.h` on novanas defines `NCCL_VERSION_CODE 23004`, read
/// 2026-09-26).
pub const RCCL_MIN_VERSION: i32 = 23_004;
/// `NCCL_VERSION_CODE` of the oldest NCCL accepted (2.27.0).
pub const NCCL_MIN_VERSION: i32 = 22_700;

/// AMD RCCL. ROCm installs it under `/opt/rocm/lib` (on novanas the tree lives at
/// `/opt/rocm/rocm`), then the loader path.
pub static RCCL_FLAVOR: NcclFlavor = NcclFlavor {
    name: "rccl",
    vendor: Vendor::Amd,
    file_prefix: "librccl",
    root_env: Some(("TURBINE_ROCM_PATH", "librccl.so.1")),
    // The LL protocol RCCL picks for small messages costs about 1 ms per all-reduce on the
    // host-staged path of GPUs without peer access (novanas: 8 B 1,034 µs, 64 KiB 1,409 µs);
    // LL128 and Simple take 46–50 µs (`scripts/lab-cluster.sh collbench-sweep-novanas`,
    // 2026-09-28). Excluding LL leaves RCCL the choice between the other two.
    env_defaults: &[("NCCL_PROTO", "^LL")],
    defaults: &[
        "/opt/rocm/lib/librccl.so.1",
        "/opt/rocm/rocm/lib/librccl.so.1",
        "librccl.so.1",
    ],
    min_version: RCCL_MIN_VERSION,
    configured: |cfg| cfg.rccl_library.as_deref(),
};

/// NVIDIA NCCL, from the loader path.
pub static NCCL_FLAVOR: NcclFlavor = NcclFlavor {
    name: "nccl",
    vendor: Vendor::Nvidia,
    file_prefix: "libnccl",
    root_env: None,
    env_defaults: &[],
    defaults: &["libnccl.so.2"],
    min_version: NCCL_MIN_VERSION,
    configured: |cfg| cfg.nccl_library.as_deref(),
};

/// Every flavor, for recognising a library by its file name.
pub(crate) static FLAVORS: [&NcclFlavor; 2] = [&RCCL_FLAVOR, &NCCL_FLAVOR];

/// A registered NCCL-API backend: the binding loaded with its flavor.
pub struct NcclApiBackend {
    flavor: &'static NcclFlavor,
}

impl NcclApiBackend {
    pub const fn new(flavor: &'static NcclFlavor) -> NcclApiBackend {
        NcclApiBackend { flavor }
    }
}

impl Module for NcclApiBackend {
    fn name(&self) -> &'static str {
        self.flavor.name
    }
}

impl CollectiveBackend for NcclApiBackend {
    fn vendors(&self) -> &'static [Vendor] {
        std::slice::from_ref(&self.flavor.vendor)
    }
    fn configured_library<'a>(&self, cfg: &'a ParallelConfig) -> Option<&'a Path> {
        (self.flavor.configured)(cfg)
    }
    fn load(&self, explicit: Option<&Path>) -> Result<Arc<dyn CollectiveLibrary>, CollectiveError> {
        let api: Arc<dyn CollectiveLibrary> = NcclApi::load(self.flavor, explicit)?;
        Ok(api)
    }
}
