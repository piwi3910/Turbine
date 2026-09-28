//! Vendor topology sources: GPU↔GPU link type, hop count and peer access from amd-smi or
//! NVML, and the coherent host link of an integrated GPU (NVML C2C mode). Each is loaded at
//! runtime like discovery; a call that fails makes the caller report `unknown` for that value.

use std::ffi::{c_uint, c_void};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr;

use libloading::Library;
use nvml_wrapper::Nvml;
use turbine_core::types::Vendor;

use crate::discovery::amd_smi::{session_begin, session_end};
use crate::discovery::loader_error;

use super::{EdgeKind, P2pStatus, PathClass, TopologyVendor, VendorLink};
use crate::inventory::DeviceInfo;

/// No vendor library: every query fails, so every vendor-derived value stays `unknown`.
pub struct NoVendorTopology;

impl TopologyVendor for NoVendorTopology {
    fn link(&mut self, _: &DeviceInfo, _: &DeviceInfo) -> Result<VendorLink, String> {
        Err("no vendor topology library loaded".into())
    }
    fn gpu_nic_p2p(&mut self, _: &DeviceInfo) -> Result<Option<P2pStatus>, String> {
        Err("no vendor topology library loaded".into())
    }
    fn coherent_host_link(&mut self, _: &DeviceInfo) -> Result<Option<String>, String> {
        Err("no vendor topology library loaded".into())
    }
}

type AmdsmiStatus = u32;
type AmdsmiHandle = *mut c_void;

const AMDSMI_STATUS_SUCCESS: AmdsmiStatus = 0;
const AMDSMI_INIT_AMD_GPUS: u64 = 1 << 1;
const AMDSMI_PROCESSOR_TYPE_AMD_GPU: u32 = 1;
/// `amdsmi_link_type_t` (ROCm 7.14.1 amdsmi.h).
const AMDSMI_LINK_TYPE_INTERNAL: u32 = 0;
const AMDSMI_LINK_TYPE_PCIE: u32 = 1;
const AMDSMI_LINK_TYPE_XGMI: u32 = 2;

type InitFn = unsafe extern "C" fn(init_flags: u64) -> AmdsmiStatus;
type ShutDownFn = unsafe extern "C" fn() -> AmdsmiStatus;
type GetSocketHandlesFn =
    unsafe extern "C" fn(socket_count: *mut u32, socket_handles: *mut AmdsmiHandle) -> AmdsmiStatus;
type GetProcessorHandlesFn = unsafe extern "C" fn(
    socket_handle: AmdsmiHandle,
    processor_count: *mut u32,
    processor_handles: *mut AmdsmiHandle,
) -> AmdsmiStatus;
type GetProcessorTypeFn =
    unsafe extern "C" fn(processor_handle: AmdsmiHandle, processor_type: *mut u32) -> AmdsmiStatus;
type GetBdfFn = unsafe extern "C" fn(processor_handle: AmdsmiHandle, bdf: *mut u64) -> AmdsmiStatus;
/// `amdsmi_topo_get_link_type(src, dst, uint64_t* hops, amdsmi_link_type_t* type)`.
type TopoGetLinkTypeFn = unsafe extern "C" fn(
    src: AmdsmiHandle,
    dst: AmdsmiHandle,
    hops: *mut u64,
    link_type: *mut u32,
) -> AmdsmiStatus;
/// `amdsmi_is_P2P_accessible(src, dst, bool* accessible)`.
type IsP2pAccessibleFn = unsafe extern "C" fn(
    src: AmdsmiHandle,
    dst: AmdsmiHandle,
    accessible: *mut bool,
) -> AmdsmiStatus;

/// GPU↔GPU links from amd-smi. Holds an initialised amd-smi session (shut down on drop) and
/// the processor handle of every AMD GPU keyed by PCI bus id.
///
/// amd-smi reports a PCIe link without the PCIe level it crosses, so a PCIe link is classed
/// `sys`, the most conservative class; xGMI is classed `xgmi`.
pub struct AmdSmiTopology {
    topo_get_link_type: TopoGetLinkTypeFn,
    is_p2p_accessible: IsP2pAccessibleFn,
    shut_down: ShutDownFn,
    /// (lower-case BDF, processor handle); handles stay valid until `amdsmi_shut_down`.
    gpus: Vec<(String, AmdsmiHandle)>,
    /// Never unloaded: ROCm libraries start helper threads that outlive `amdsmi_shut_down`.
    _lib: ManuallyDrop<Library>,
}

impl AmdSmiTopology {
    /// Loads `library` (e.g. `libamd_smi.so`), initialises it and enumerates the AMD GPUs.
    pub fn open(library: &Path) -> Result<AmdSmiTopology, String> {
        // SAFETY: loading runs amd-smi's static initialisers, which have no preconditions. The
        // Library is kept (never unloaded) by the returned value, so every fn pointer copied
        // out below stays valid for the life of the process.
        let lib = unsafe { Library::new(library) }
            .map_err(|e| format!("{}: {}", library.display(), loader_error(&e)))?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // SAFETY: `$ty` is the exact C signature declared in amdsmi.h for `$name`; the
                // pointer is only used while `lib` (kept by AmdSmiTopology) is loaded.
                let s = unsafe { lib.get::<$ty>($name) }.map_err(|e| {
                    format!(
                        "{}: missing symbol {}: {}",
                        library.display(),
                        $name,
                        loader_error(&e)
                    )
                })?;
                *s
            }};
        }
        let init: InitFn = sym!("amdsmi_init", InitFn);
        let shut_down: ShutDownFn = sym!("amdsmi_shut_down", ShutDownFn);
        let get_socket_handles: GetSocketHandlesFn =
            sym!("amdsmi_get_socket_handles", GetSocketHandlesFn);
        let get_processor_handles: GetProcessorHandlesFn =
            sym!("amdsmi_get_processor_handles", GetProcessorHandlesFn);
        let get_processor_type: GetProcessorTypeFn =
            sym!("amdsmi_get_processor_type", GetProcessorTypeFn);
        let get_bdf: GetBdfFn = sym!("amdsmi_get_gpu_device_bdf", GetBdfFn);
        let topo_get_link_type: TopoGetLinkTypeFn =
            sym!("amdsmi_topo_get_link_type", TopoGetLinkTypeFn);
        let is_p2p_accessible: IsP2pAccessibleFn =
            sym!("amdsmi_is_P2P_accessible", IsP2pAccessibleFn);

        // Joins the process-wide amd-smi session shared with discovery and telemetry (see
        // `discovery::amd_smi::SESSIONS`): an init / shut-down pair of its own would tear the
        // library down under a concurrent discovery (novanas, two GPUs: discoveries beside a
        // topology discovery saw 0 devices, and its own links fell back to nominal).
        // SAFETY: amdsmi_init takes no pointers; the session is left by Drop (also on the error
        // below, since `topo` owns it by then).
        session_begin(|| unsafe { init(AMDSMI_INIT_AMD_GPUS) })?;
        let mut topo = AmdSmiTopology {
            topo_get_link_type,
            is_p2p_accessible,
            shut_down,
            gpus: Vec::new(),
            _lib: ManuallyDrop::new(lib),
        };
        topo.gpus = enumerate_gpus(
            get_socket_handles,
            get_processor_handles,
            get_processor_type,
            get_bdf,
        )?;
        Ok(topo)
    }

    fn handle(&self, d: &DeviceInfo) -> Result<AmdsmiHandle, String> {
        if d.vendor != Vendor::Amd {
            return Err(format!("device {} is not an AMD GPU", d.index.0));
        }
        let bdf = d
            .pci_bus_id
            .as_deref()
            .ok_or_else(|| format!("device {} has no PCI bus id", d.index.0))?
            .to_ascii_lowercase();
        self.gpus
            .iter()
            .find(|(b, _)| *b == bdf)
            .map(|(_, h)| *h)
            .ok_or_else(|| format!("amd-smi reports no GPU at {bdf}"))
    }
}

/// Every AMD GPU processor handle with its BDF.
fn enumerate_gpus(
    get_socket_handles: GetSocketHandlesFn,
    get_processor_handles: GetProcessorHandlesFn,
    get_processor_type: GetProcessorTypeFn,
    get_bdf: GetBdfFn,
) -> Result<Vec<(String, AmdsmiHandle)>, String> {
    let mut socket_count: u32 = 0;
    // SAFETY: a NULL handle array returns only the count in `socket_count`, a valid local.
    let st = unsafe { get_socket_handles(&mut socket_count, ptr::null_mut()) };
    check(st, "amdsmi_get_socket_handles")?;
    let mut sockets: Vec<AmdsmiHandle> = vec![ptr::null_mut(); socket_count as usize];
    // SAFETY: `sockets` has room for exactly `socket_count` handles, the limit passed in.
    let st = unsafe { get_socket_handles(&mut socket_count, sockets.as_mut_ptr()) };
    check(st, "amdsmi_get_socket_handles")?;
    sockets.truncate(socket_count as usize);
    let mut gpus = Vec::new();
    for socket in sockets {
        let mut n: u32 = 0;
        // SAFETY: `socket` is a live socket handle of this session; NULL output = count only.
        let st = unsafe { get_processor_handles(socket, &mut n, ptr::null_mut()) };
        check(st, "amdsmi_get_processor_handles")?;
        let mut procs: Vec<AmdsmiHandle> = vec![ptr::null_mut(); n as usize];
        // SAFETY: `procs` has room for exactly `n` handles, the limit passed in.
        let st = unsafe { get_processor_handles(socket, &mut n, procs.as_mut_ptr()) };
        check(st, "amdsmi_get_processor_handles")?;
        procs.truncate(n as usize);
        for p in procs {
            let mut kind: u32 = 0;
            // SAFETY: `p` is a live processor handle; `kind` is a valid out-parameter.
            let st = unsafe { get_processor_type(p, &mut kind) };
            if st != AMDSMI_STATUS_SUCCESS || kind != AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                continue;
            }
            let mut bdf: u64 = 0;
            // SAFETY: amdsmi_bdf_t is a 64-bit union; `bdf` is an exclusively borrowed u64.
            if unsafe { get_bdf(p, &mut bdf) } == AMDSMI_STATUS_SUCCESS {
                gpus.push((format_bdf(bdf), p));
            }
        }
    }
    Ok(gpus)
}

impl TopologyVendor for AmdSmiTopology {
    fn link(&mut self, a: &DeviceInfo, b: &DeviceInfo) -> Result<VendorLink, String> {
        let (ha, hb) = (self.handle(a)?, self.handle(b)?);
        let mut hops: u64 = 0;
        let mut link_type: u32 = 0;
        // SAFETY: both handles are live processor handles of this session; `hops` and
        // `link_type` are exclusively borrowed locals of the C types the call writes.
        let st = unsafe { (self.topo_get_link_type)(ha, hb, &mut hops, &mut link_type) };
        check(st, "amdsmi_topo_get_link_type")?;
        let (kind, path) = match link_type {
            AMDSMI_LINK_TYPE_XGMI => (EdgeKind::Xgmi, PathClass::Xgmi),
            AMDSMI_LINK_TYPE_PCIE => (EdgeKind::Pcie, PathClass::Sys),
            AMDSMI_LINK_TYPE_INTERNAL => (EdgeKind::Pcie, PathClass::SelfPath),
            other => return Err(format!("amdsmi_topo_get_link_type returned type {other}")),
        };
        let mut accessible = false;
        // SAFETY: as above; `accessible` is an exclusively borrowed C `bool`.
        let st = unsafe { (self.is_p2p_accessible)(ha, hb, &mut accessible) };
        let p2p = match (st, accessible) {
            (AMDSMI_STATUS_SUCCESS, true) => P2pStatus::Enabled,
            (AMDSMI_STATUS_SUCCESS, false) => P2pStatus::Disabled,
            _ => P2pStatus::Unknown,
        };
        Ok(VendorLink {
            kind,
            path,
            hops: u32::try_from(hops).ok(),
            p2p,
        })
    }

    /// amd-smi has no GPU↔NIC peer-access query.
    fn gpu_nic_p2p(&mut self, gpu: &DeviceInfo) -> Result<Option<P2pStatus>, String> {
        self.handle(gpu).map(|_| None)
    }

    /// Discrete AMD GPUs have no coherent host link.
    fn coherent_host_link(&mut self, gpu: &DeviceInfo) -> Result<Option<String>, String> {
        self.handle(gpu).map(|_| None)
    }
}

impl Drop for AmdSmiTopology {
    fn drop(&mut self) {
        let shut_down = self.shut_down;
        // SAFETY: amdsmi_shut_down takes no arguments; it runs only when this was the last open
        // session in the process, and no handle of this value is used afterwards.
        session_end(|| unsafe { shut_down() });
    }
}

/// `nvmlC2cModeInfo_v1_t`.
#[repr(C)]
struct NvmlC2cModeInfo {
    is_c2c_enabled: c_uint,
}

/// `nvmlDeviceGetC2cModeInfoV(nvmlDevice_t, nvmlC2cModeInfo_v1_t*)`.
type C2cModeInfoFn =
    unsafe extern "C" fn(device: *mut c_void, info: *mut NvmlC2cModeInfo) -> c_uint;

/// GPU↔GPU links from NVML (common ancestor, NVLink and read P2P status) and the C2C host link
/// of integrated GPUs such as GB10.
pub struct NvmlTopology {
    nvml: Nvml,
    /// Resolved from a second handle on the same library (the loader returns the already
    /// mapped image); absent on drivers without C2C mode reporting.
    c2c_mode: Option<C2cModeInfoFn>,
    _lib: ManuallyDrop<Library>,
}

impl NvmlTopology {
    /// Initialises NVML from `library` (e.g. `libnvidia-ml.so.1`).
    pub fn open(library: &Path) -> Result<NvmlTopology, String> {
        let nvml = Nvml::builder()
            .lib_path(library.as_os_str())
            .init()
            .map_err(|e| format!("{}: {e}", library.display()))?;
        // SAFETY: the library is already loaded and initialised by `Nvml`; this only takes a
        // second reference to the same image, which is never unloaded (ManuallyDrop), so the
        // fn pointer copied out below stays valid while `nvml` is alive.
        let lib = unsafe { Library::new(library) }
            .map_err(|e| format!("{}: {}", library.display(), loader_error(&e)))?;
        // SAFETY: `C2cModeInfoFn` is the C signature of nvmlDeviceGetC2cModeInfoV in nvml.h.
        let c2c_mode = unsafe { lib.get::<C2cModeInfoFn>(b"nvmlDeviceGetC2cModeInfoV") }
            .ok()
            .map(|s| *s);
        Ok(NvmlTopology {
            nvml,
            c2c_mode,
            _lib: ManuallyDrop::new(lib),
        })
    }
}

fn nvidia(d: &DeviceInfo) -> Result<u32, String> {
    if d.vendor == Vendor::Nvidia {
        Ok(d.vendor_index)
    } else {
        Err(format!("device {} is not an NVIDIA GPU", d.index.0))
    }
}

fn p2p_from_nvml(status: nvml_wrapper::enum_wrappers::device::P2pStatus) -> P2pStatus {
    use nvml_wrapper::enum_wrappers::device::P2pStatus as N;
    match status {
        N::Ok => P2pStatus::Enabled,
        N::Unknown => P2pStatus::Unknown,
        _ => P2pStatus::Disabled,
    }
}

#[cfg(target_os = "linux")]
fn common_ancestor(
    a: &nvml_wrapper::Device<'_>,
    b: nvml_wrapper::Device<'_>,
) -> Result<PathClass, String> {
    use nvml_wrapper::enum_wrappers::device::TopologyLevel as L;
    let level = a
        .topology_common_ancestor(b)
        .map_err(|e| format!("nvmlDeviceGetTopologyCommonAncestor: {e}"))?;
    Ok(match level {
        L::Internal => PathClass::SelfPath,
        L::Single => PathClass::Pix,
        L::Multiple => PathClass::Pxb,
        L::HostBridge => PathClass::Phb,
        L::Node => PathClass::Node,
        L::System => PathClass::Sys,
    })
}

#[cfg(not(target_os = "linux"))]
fn common_ancestor(
    _: &nvml_wrapper::Device<'_>,
    _: nvml_wrapper::Device<'_>,
) -> Result<PathClass, String> {
    Err("nvmlDeviceGetTopologyCommonAncestor is Linux-only".into())
}

impl TopologyVendor for NvmlTopology {
    fn link(&mut self, a: &DeviceInfo, b: &DeviceInfo) -> Result<VendorLink, String> {
        use nvml_wrapper::enum_wrappers::device::{P2pCapabilitiesIndex, P2pStatus as N};
        let (ia, ib) = (nvidia(a)?, nvidia(b)?);
        let open = |i: u32| {
            self.nvml
                .device_by_index(i)
                .map_err(|e| format!("nvmlDeviceGetHandleByIndex({i}): {e}"))
        };
        let (da, db) = (open(ia)?, open(ib)?);
        let nvlink = matches!(da.p2p_status(&db, P2pCapabilitiesIndex::NvLink), Ok(N::Ok));
        let (kind, path) = if nvlink {
            (EdgeKind::Nvlink, PathClass::Nvlink)
        } else {
            (EdgeKind::Pcie, common_ancestor(&da, db)?)
        };
        let db = open(ib)?;
        let p2p = da
            .p2p_status(&db, P2pCapabilitiesIndex::Read)
            .map_or(P2pStatus::Unknown, p2p_from_nvml);
        Ok(VendorLink {
            kind,
            path,
            hops: None,
            p2p,
        })
    }

    /// NVML has no GPU↔NIC peer-access query; the edge stays `unknown`.
    fn gpu_nic_p2p(&mut self, gpu: &DeviceInfo) -> Result<Option<P2pStatus>, String> {
        nvidia(gpu).map(|_| None)
    }

    fn coherent_host_link(&mut self, gpu: &DeviceInfo) -> Result<Option<String>, String> {
        let index = nvidia(gpu)?;
        let f = self
            .c2c_mode
            .ok_or("nvmlDeviceGetC2cModeInfoV is not exported by this driver")?;
        let device = self
            .nvml
            .device_by_index(index)
            .map_err(|e| format!("nvmlDeviceGetHandleByIndex({index}): {e}"))?;
        let mut info = NvmlC2cModeInfo { is_c2c_enabled: 0 };
        // SAFETY: `device.handle()` is a live nvmlDevice_t of the initialised `self.nvml`,
        // valid while `device` borrows it; `info` is an exclusively borrowed value of the
        // exact C struct the call writes, and `f` came from the library `self` keeps loaded.
        let rc = unsafe { f(device.handle().cast::<c_void>(), &mut info) };
        if rc != 0 {
            return Err(format!("nvmlDeviceGetC2cModeInfoV returned {rc}"));
        }
        Ok((info.is_c2c_enabled != 0).then(|| "nvlink_c2c".to_string()))
    }
}

fn check(status: AmdsmiStatus, call: &str) -> Result<(), String> {
    if status == AMDSMI_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{call} failed with amdsmi_status_t {status}"))
    }
}

/// `amdsmi_bdf_t` bitfields: function[0:3], device[3:8], bus[8:16], domain[16:64].
fn format_bdf(bdf: u64) -> String {
    let function = bdf & 0x7;
    let device = (bdf >> 3) & 0x1f;
    let bus = (bdf >> 8) & 0xff;
    let domain = bdf >> 16;
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn missing_libraries_are_errors_not_panics() {
        let missing = PathBuf::from("/nonexistent/libamd_smi.so");
        let err = AmdSmiTopology::open(&missing)
            .err()
            .expect("missing amd-smi fails");
        assert!(err.contains("/nonexistent/libamd_smi.so"), "{err}");
        let err = NvmlTopology::open(Path::new("/nonexistent/libnvidia-ml.so.1"))
            .err()
            .expect("missing NVML fails");
        assert!(err.contains("/nonexistent/libnvidia-ml.so.1"), "{err}");
        assert_eq!(format_bdf(0x0300), "0000:03:00.0");
    }
}
