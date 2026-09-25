//! AMD discovery through a minimal runtime-loaded FFI to `libamd_smi.so`.
//!
//! Layouts and values are copied from ROCm 7.14.1 `include/amd_smi/amdsmi.h`
//! (AMDSMI_LIB_VERSION 26.5.0); the layout tests pin the sizes the C compiler reports.

use std::ffi::{c_char, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use libloading::Library;
use turbine_core::types::{DeviceId, MemoryKind, Vendor};

use super::DiscoveryBackend;
use crate::inventory::{DeviceInfo, DeviceMemoryInfo};

type AmdsmiStatus = u32; // amdsmi_status_t (C enum)
type AmdsmiHandle = *mut c_void; // amdsmi_socket_handle / amdsmi_processor_handle

const AMDSMI_STATUS_SUCCESS: AmdsmiStatus = 0;
const AMDSMI_INIT_AMD_GPUS: u64 = 1 << 1;
const AMDSMI_PROCESSOR_TYPE_AMD_GPU: u32 = 1;
const AMDSMI_MEM_TYPE_VRAM: u32 = 0;
const AMDSMI_MAX_STRING_LENGTH: usize = 256;
const AMDSMI_GPU_UUID_SIZE: usize = 38;
const NOT_SUPPORTED_U64: u64 = u64::MAX;

/// `amdsmi_asic_info_t` (896 bytes). Only some fields are read; the rest fix the layout.
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct AmdsmiAsicInfo {
    market_name: [c_char; AMDSMI_MAX_STRING_LENGTH],
    vendor_id: u32,
    vendor_name: [c_char; AMDSMI_MAX_STRING_LENGTH],
    subvendor_id: u32,
    device_id: u64,
    rev_id: u32,
    asic_serial: [c_char; AMDSMI_MAX_STRING_LENGTH],
    oam_id: u32,
    num_of_compute_units: u32,
    target_graphics_version: u64,
    subsystem_id: u32,
    flags: u64,
    reserved: [u32; 18],
}

/// `amdsmi_driver_info_t` (768 bytes).
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct AmdsmiDriverInfo {
    driver_version: [c_char; AMDSMI_MAX_STRING_LENGTH],
    driver_date: [c_char; AMDSMI_MAX_STRING_LENGTH],
    driver_name: [c_char; AMDSMI_MAX_STRING_LENGTH],
}

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
type GetAsicInfoFn =
    unsafe extern "C" fn(processor_handle: AmdsmiHandle, info: *mut AmdsmiAsicInfo) -> AmdsmiStatus;
type GetUuidFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    uuid_length: *mut c_uint,
    uuid: *mut c_char,
) -> AmdsmiStatus;
type GetBdfFn = unsafe extern "C" fn(processor_handle: AmdsmiHandle, bdf: *mut u64) -> AmdsmiStatus;
type GetMemoryTotalFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    mem_type: u32,
    total: *mut u64,
) -> AmdsmiStatus;
type GetDriverInfoFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    info: *mut AmdsmiDriverInfo,
) -> AmdsmiStatus;

/// Resolved entry points. Plain fn pointers: valid only while `_lib` is loaded, which this
/// struct guarantees by owning it.
struct Api {
    init: InitFn,
    shut_down: ShutDownFn,
    get_socket_handles: GetSocketHandlesFn,
    get_processor_handles: GetProcessorHandlesFn,
    get_processor_type: GetProcessorTypeFn,
    get_asic_info: GetAsicInfoFn,
    get_uuid: GetUuidFn,
    get_bdf: GetBdfFn,
    get_memory_total: GetMemoryTotalFn,
    get_driver_info: GetDriverInfoFn,
    _lib: Library,
}

impl Api {
    fn load(path: &Path) -> Result<Api, String> {
        // SAFETY: loading runs amd-smi's static initialisers, which have no preconditions.
        // The Library is stored in `Api::_lib`, so every fn pointer resolved below stays valid
        // for as long as the Api value lives.
        let lib = unsafe { Library::new(path) }
            .map_err(|e| format!("{}: {}", path.display(), super::loader_error(&e)))?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // SAFETY: `$ty` is the exact C signature declared in amdsmi.h for `$name`;
                // the pointer is copied out and only used while `lib` (moved into Api) is loaded.
                let s = unsafe { lib.get::<$ty>($name) }.map_err(|e| {
                    format!(
                        "{}: missing symbol {}: {}",
                        path.display(),
                        $name,
                        super::loader_error(&e)
                    )
                })?;
                *s
            }};
        }
        Ok(Api {
            init: sym!("amdsmi_init", InitFn),
            shut_down: sym!("amdsmi_shut_down", ShutDownFn),
            get_socket_handles: sym!("amdsmi_get_socket_handles", GetSocketHandlesFn),
            get_processor_handles: sym!("amdsmi_get_processor_handles", GetProcessorHandlesFn),
            get_processor_type: sym!("amdsmi_get_processor_type", GetProcessorTypeFn),
            get_asic_info: sym!("amdsmi_get_gpu_asic_info", GetAsicInfoFn),
            get_uuid: sym!("amdsmi_get_gpu_device_uuid", GetUuidFn),
            get_bdf: sym!("amdsmi_get_gpu_device_bdf", GetBdfFn),
            get_memory_total: sym!("amdsmi_get_gpu_memory_total", GetMemoryTotalFn),
            get_driver_info: sym!("amdsmi_get_gpu_driver_info", GetDriverInfoFn),
            _lib: lib,
        })
    }

    /// All AMD GPU processor handles, socket by socket (amd-smi order).
    fn gpu_handles(&self) -> Result<Vec<AmdsmiHandle>, String> {
        let mut socket_count: u32 = 0;
        // SAFETY: a NULL handle array is documented to return only the count in `socket_count`,
        // a valid, exclusively borrowed u32.
        let st = unsafe { (self.get_socket_handles)(&mut socket_count, ptr::null_mut()) };
        check(st, "amdsmi_get_socket_handles")?;
        let mut sockets: Vec<AmdsmiHandle> = vec![ptr::null_mut(); socket_count as usize];
        // SAFETY: `sockets` has room for exactly `socket_count` handles, the limit passed in.
        let st = unsafe { (self.get_socket_handles)(&mut socket_count, sockets.as_mut_ptr()) };
        check(st, "amdsmi_get_socket_handles")?;
        sockets.truncate(socket_count as usize);

        let mut gpus = Vec::new();
        for socket in sockets {
            let mut n: u32 = 0;
            // SAFETY: `socket` came from amdsmi_get_socket_handles in this session; NULL output = count only.
            let st = unsafe { (self.get_processor_handles)(socket, &mut n, ptr::null_mut()) };
            check(st, "amdsmi_get_processor_handles")?;
            let mut procs: Vec<AmdsmiHandle> = vec![ptr::null_mut(); n as usize];
            // SAFETY: `procs` has room for exactly `n` handles, the limit passed in.
            let st = unsafe { (self.get_processor_handles)(socket, &mut n, procs.as_mut_ptr()) };
            check(st, "amdsmi_get_processor_handles")?;
            procs.truncate(n as usize);
            for p in procs {
                let mut kind: u32 = 0;
                // SAFETY: `p` is a live processor handle; `kind` is a valid out-parameter.
                let st = unsafe { (self.get_processor_type)(p, &mut kind) };
                if st == AMDSMI_STATUS_SUCCESS && kind == AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                    gpus.push(p);
                }
            }
        }
        Ok(gpus)
    }

    fn device_info(&self, handle: AmdsmiHandle, vendor_index: u32) -> DeviceInfo {
        // SAFETY: an all-zero bit pattern is a valid AmdsmiAsicInfo (integers and char arrays).
        let mut asic: AmdsmiAsicInfo = unsafe { std::mem::zeroed() };
        // SAFETY: `handle` is a live processor handle from gpu_handles(); `asic` is an exclusively
        // borrowed value of the exact C type amdsmi_get_gpu_asic_info writes.
        let asic_ok = unsafe { (self.get_asic_info)(handle, &mut asic) } == AMDSMI_STATUS_SUCCESS;

        let mut uuid_buf = [0 as c_char; AMDSMI_GPU_UUID_SIZE];
        let mut uuid_len = AMDSMI_GPU_UUID_SIZE as c_uint;
        // SAFETY: `uuid_len` tells the library the buffer holds AMDSMI_GPU_UUID_SIZE chars,
        // the documented minimum; both pointers are exclusively borrowed locals.
        let uuid_ok = unsafe { (self.get_uuid)(handle, &mut uuid_len, uuid_buf.as_mut_ptr()) }
            == AMDSMI_STATUS_SUCCESS;

        let mut bdf: u64 = 0;
        // SAFETY: amdsmi_bdf_t is a 64-bit union; `bdf` is an exclusively borrowed u64.
        let bdf_ok = unsafe { (self.get_bdf)(handle, &mut bdf) } == AMDSMI_STATUS_SUCCESS;

        let mut vram: u64 = 0;
        // SAFETY: `vram` is an exclusively borrowed u64; AMDSMI_MEM_TYPE_VRAM is a valid enum value.
        let vram_ok = unsafe { (self.get_memory_total)(handle, AMDSMI_MEM_TYPE_VRAM, &mut vram) }
            == AMDSMI_STATUS_SUCCESS;

        // SAFETY: an all-zero bit pattern is a valid AmdsmiDriverInfo (char arrays).
        let mut driver: AmdsmiDriverInfo = unsafe { std::mem::zeroed() };
        // SAFETY: `driver` is an exclusively borrowed value of the exact C type the call writes.
        let driver_ok =
            unsafe { (self.get_driver_info)(handle, &mut driver) } == AMDSMI_STATUS_SUCCESS;

        let name = Some(c_string(&asic.market_name))
            .filter(|n| asic_ok && !n.is_empty())
            .unwrap_or_else(|| "unknown AMD device".to_string());
        DeviceInfo {
            index: DeviceId(0),
            vendor: Vendor::Amd,
            vendor_index,
            name,
            uuid: Some(c_string(&uuid_buf)).filter(|u| uuid_ok && !u.is_empty()),
            pci_bus_id: bdf_ok.then(|| format_bdf(bdf)),
            arch: asic_ok
                .then(|| gfx_arch(asic.target_graphics_version))
                .flatten(),
            driver_version: Some(c_string(&driver.driver_version))
                .filter(|v| driver_ok && !v.is_empty() && v != "N/A"),
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: if vram_ok { vram } else { 0 },
                shared_with_host: false,
            },
        }
    }
}

fn check(status: AmdsmiStatus, call: &str) -> Result<(), String> {
    if status == AMDSMI_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{call} failed with amdsmi_status_t {status}"))
    }
}

/// NUL-terminated C char buffer → String (lossy UTF-8), without reading past the buffer.
fn c_string(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// `amdsmi_bdf_t` bitfields: function[0:3], device[3:8], bus[8:16], domain[16:64].
pub(crate) fn format_bdf(bdf: u64) -> String {
    let function = bdf & 0x7;
    let device = (bdf >> 3) & 0x1f;
    let bus = (bdf >> 8) & 0xff;
    let domain = bdf >> 16;
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// `target_graphics_version` is printed by amd-smi as `"gfx" + hex` (0x1201 → gfx1201).
pub(crate) fn gfx_arch(target_graphics_version: u64) -> Option<String> {
    (target_graphics_version != NOT_SUPPORTED_U64 && target_graphics_version != 0)
        .then(|| format!("gfx{target_graphics_version:x}"))
}

pub(crate) struct AmdSmiBackend {
    library: PathBuf,
}

impl AmdSmiBackend {
    pub(crate) fn new(library: PathBuf) -> Self {
        AmdSmiBackend { library }
    }
}

impl DiscoveryBackend for AmdSmiBackend {
    fn vendor(&self) -> Vendor {
        Vendor::Amd
    }

    fn discover(&mut self) -> Result<Vec<DeviceInfo>, String> {
        let api = Api::load(&self.library)?;
        // SAFETY: amdsmi_init has no pointer arguments; it is paired with amdsmi_shut_down below.
        check(unsafe { (api.init)(AMDSMI_INIT_AMD_GPUS) }, "amdsmi_init")?;
        let result = api.gpu_handles().map(|handles| {
            handles
                .into_iter()
                .enumerate()
                .map(|(i, h)| api.device_info(h, u32::try_from(i).unwrap_or(u32::MAX)))
                .collect()
        });
        // SAFETY: called once after a successful amdsmi_init; no handle is used afterwards.
        let _ = unsafe { (api.shut_down)() };
        // Keep the library mapped for the process lifetime: unloading ROCm libraries that
        // started helper threads is unsafe, and Phase 3 telemetry reloads it anyway.
        std::mem::forget(api);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_amdsmi_h() {
        // Values printed by the C compiler for ROCm 7.14.1 amdsmi.h.
        assert_eq!(std::mem::size_of::<AmdsmiAsicInfo>(), 896);
        assert_eq!(std::mem::offset_of!(AmdsmiAsicInfo, vendor_id), 256);
        assert_eq!(std::mem::offset_of!(AmdsmiAsicInfo, device_id), 520);
        assert_eq!(
            std::mem::offset_of!(AmdsmiAsicInfo, target_graphics_version),
            800
        );
        assert_eq!(std::mem::size_of::<AmdsmiDriverInfo>(), 768);
    }

    #[test]
    fn bdf_and_arch_format() {
        assert_eq!(format_bdf(0x0300), "0000:03:00.0");
        assert_eq!(
            format_bdf((0x0001 << 16) | (0x0f << 8) | (0x01 << 3) | 0x2),
            "0001:0f:01.2"
        );
        assert_eq!(gfx_arch(0x1201).as_deref(), Some("gfx1201"));
        assert_eq!(gfx_arch(u64::MAX), None);
    }
}
