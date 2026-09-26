//! `hip`: the HIP kernel library (_libturbine_hip.so_) on an AMD device. The library is the
//! first loadable one of [`ShimLibrary::search_paths`] (an ABI, backend or architecture
//! mismatch is fatal, a missing file moves on); the context is created on the configured device,
//! which must be an AMD device.

use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_core::types::Vendor;
use turbine_device::{DeviceInfo, DeviceInventory};

use super::{BackendError, BackendNote, BackendRequest, ExecutionBackend, OpenedBackend};
use crate::cards;
use crate::{
    KernelError, KernelProvider, OpKind, Selection, ShimContext, ShimLibrary, shim_provider,
};

/// The implementation name the HIP shim reports for Composable Kernel paged attention.
const CK_PAGED_ATTENTION: &str = "ck_tile_fmha_pagedkv";

pub struct HipBackend;

impl HipBackend {
    /// Device error names after which a HIP context is corrupted (contract §9.2).
    pub const STICKY_ERRORS: &'static [&'static str] = &[
        "hipErrorIllegalAddress",
        "hipErrorLaunchFailure",
        "hipErrorAssert",
    ];
}

impl Module for HipBackend {
    fn name(&self) -> &'static str {
        "hip"
    }
}

impl ExecutionBackend for HipBackend {
    fn vendor(&self) -> &'static str {
        Vendor::Amd.as_str()
    }

    /// The library, the AMD device, its card profile (a device no profile describes is refused
    /// before any context exists), then a context on the device.
    fn open(&self, req: &BackendRequest<'_>) -> Result<OpenedBackend, BackendError> {
        let lib = load_shim(self.name(), req.kernel_library)?;
        let device = amd_device(req.inventory, req.device.0)?;
        let card = cards::select(req.card_profile, device)
            .map_err(|e| BackendError::Startup(format!("execution.card_profile: {e}")))?;
        let ctx: Arc<ShimContext> = lib
            .create_context(device)
            .map_err(|e| BackendError::Startup(format!("kernel library context: {e}")))?;
        tracing::info!(
            event = "kernel_library_loaded",
            path = %lib.path().display(),
            backend = lib.backend_name(),
            abi_version = lib.abi_version(),
            build_archs = %lib.build_archs().join(","),
            device_arch = device.arch.as_deref().unwrap_or("unknown"),
            driver_version = device.driver_version.as_deref().unwrap_or("unknown"),
            "kernel library loaded"
        );
        let provider: Arc<dyn KernelProvider> = shim_provider(Arc::clone(&ctx));
        let id = provider.id();
        let graphs = lib.supports_graphs().then(|| Arc::clone(&ctx));
        Ok(OpenedBackend {
            mem: ctx.clone(),
            providers: vec![provider],
            order: vec![id],
            memory_kind: device.memory.kind,
            context: Some(ctx),
            graphs,
            device: Some(device.clone()),
            card: Some(card),
        })
    }

    fn sticky_error_prefixes(&self) -> &'static [&'static str] {
        Self::STICKY_ERRORS
    }

    /// `paged_attention_fallback` when paged attention is not on Composable Kernel's
    /// `fmha_fwd_pagedkv` (CK serves only pages of a multiple of 128 tokens; other sizes run the
    /// slower Turbine kernel).
    fn selection_notes(&self, selections: &[Selection]) -> Vec<BackendNote> {
        selections
            .iter()
            .filter(|s| {
                matches!(
                    s.op,
                    OpKind::AttentionPrefillPaged | OpKind::AttentionDecodePaged
                )
            })
            .find(|s| s.implementation != CK_PAGED_ATTENTION)
            .map(|s| BackendNote {
                event: "paged_attention_fallback",
                fields: vec![("impl", s.implementation.clone())],
                message: "paged attention is not on CK fmha_fwd_pagedkv: kv.block_tokens is not \
                          a multiple of 128",
            })
            .into_iter()
            .collect()
    }
}

/// The first loadable `libturbine_<backend>.so` of the search order.
fn load_shim(backend: &str, explicit: Option<&Path>) -> Result<Arc<ShimLibrary>, BackendError> {
    let paths = ShimLibrary::search_paths(backend, explicit);
    if explicit.is_none() {
        let order: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        tracing::info!(search_order = %order.join(", "), "kernel library search order");
    }
    let mut misses = Vec::new();
    for path in &paths {
        match ShimLibrary::load(path, backend) {
            Ok(lib) => return Ok(lib),
            Err(e @ KernelError::Load { .. }) => misses.push(e.to_string()),
            Err(e) => return Err(BackendError::Startup(format!("kernel library: {e}"))),
        }
    }
    Err(BackendError::Startup(format!(
        "kernel library: no loadable libturbine_{backend}.so: {}",
        misses.join("; ")
    )))
}

/// The inventory device `index`, which must be an AMD device.
fn amd_device(inventory: &DeviceInventory, index: u32) -> Result<&DeviceInfo, BackendError> {
    let device = inventory
        .devices
        .iter()
        .find(|d| d.index.0 == index)
        .ok_or_else(|| {
            BackendError::Startup(format!(
                "execution.device {index} is not in the device inventory ({} devices)",
                inventory.devices.len()
            ))
        })?;
    if device.vendor != Vendor::Amd {
        return Err(BackendError::Startup(format!(
            "execution.device {index} is a {} device; backend hip needs an AMD device",
            device.vendor.as_str()
        )));
    }
    Ok(device)
}

#[cfg(test)]
mod tests {
    use turbine_core::types::{DeviceId, MemoryKind};
    use turbine_device::DeviceMemoryInfo;

    use super::*;
    use crate::ProviderId;
    use crate::test_support::{plain_device_error, sticky_device_error};

    fn device(vendor: Vendor, arch: &str) -> DeviceInfo {
        DeviceInfo {
            index: DeviceId(0),
            vendor,
            vendor_index: 0,
            name: "test device".into(),
            uuid: None,
            pci_bus_id: None,
            arch: Some(arch.into()),
            driver_version: None,
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 1 << 30,
                shared_with_host: false,
            },
        }
    }

    fn sel(op: OpKind, implementation: &str) -> Selection {
        Selection {
            op,
            config: String::new(),
            provider: ProviderId("hip"),
            implementation: implementation.to_string(),
            reason: String::new(),
        }
    }

    #[test]
    fn sticky_prefixes_and_fallback_note() {
        assert!(sticky_device_error("x").is_sticky());
        assert!(!plain_device_error("x").is_sticky());
        assert!(
            KernelError::Device {
                message: "hipErrorIllegalAddress: an illegal memory access".into()
            }
            .is_sticky()
        );
        assert_eq!(
            HipBackend.sticky_error_prefixes(),
            HipBackend::STICKY_ERRORS
        );

        let fallback = [
            sel(OpKind::Gemm, "hipblaslt"),
            sel(OpKind::AttentionPrefillPaged, "turbine_hip"),
            sel(OpKind::AttentionDecodePaged, "turbine_hip"),
        ];
        let notes = HipBackend.selection_notes(&fallback);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].event, "paged_attention_fallback");
        assert_eq!(notes[0].fields, [("impl", "turbine_hip".to_string())]);
        let on_ck = [
            sel(OpKind::Gemm, "hipblaslt"),
            sel(OpKind::AttentionPrefillPaged, CK_PAGED_ATTENTION),
            sel(OpKind::AttentionDecodePaged, CK_PAGED_ATTENTION),
        ];
        assert!(HipBackend.selection_notes(&on_ck).is_empty());
    }

    #[test]
    fn refuses_a_non_amd_device() {
        let inventory = DeviceInventory {
            devices: vec![device(Vendor::Nvidia, "sm_121")],
            backends: Vec::new(),
        };
        let err = HipBackend
            .open(&BackendRequest {
                device: DeviceId(0),
                kernel_library: Some(Path::new(env!("TURBINE_STUB_GFX942"))),
                inventory: &inventory,
                meminfo: Path::new("/nonexistent"),
                card_profile: "auto",
            })
            .err()
            .expect("an NVIDIA device under backend hip");
        assert_eq!(
            err.to_string(),
            "execution.device 0 is a nvidia device; backend hip needs an AMD device"
        );
    }

    /// An AMD device whose architecture no card profile lists is refused before a context is
    /// created (the stub library is built for `gfx942`, which has no profile).
    #[test]
    fn refuses_device_without_profile() {
        let inventory = DeviceInventory {
            devices: vec![device(Vendor::Amd, "gfx942")],
            backends: Vec::new(),
        };
        let err = HipBackend
            .open(&BackendRequest {
                device: DeviceId(0),
                kernel_library: Some(Path::new(env!("TURBINE_STUB_GFX942"))),
                inventory: &inventory,
                meminfo: Path::new("/nonexistent"),
                card_profile: "auto",
            })
            .err()
            .expect("a gfx942 device without a card profile");
        assert_eq!(
            err.to_string(),
            "execution.card_profile: no card profile for device architecture gfx942 \
             (profiles: gfx1201)"
        );
    }
}
