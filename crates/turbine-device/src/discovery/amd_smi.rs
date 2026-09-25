//! AMD discovery through a minimal runtime-loaded FFI to `libamd_smi.so`.
//!
//! Layouts and values are copied from ROCm 7.14.1 `include/amd_smi/amdsmi.h`
//! (AMDSMI_LIB_VERSION 26.5.0); the layout tests pin the sizes the C compiler reports.

use std::ffi::{c_char, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Mutex, PoisonError};

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

/// One amd-smi session: `init`, the device queries, `shut_down`. A seam so the session
/// discipline in [`run_session`] is testable without the library.
trait Session {
    fn init(&self) -> AmdsmiStatus;
    fn devices(&self) -> Result<Vec<DeviceInfo>, String>;
    fn shut_down(&self) -> AmdsmiStatus;
}

impl Session for Api {
    fn init(&self) -> AmdsmiStatus {
        // SAFETY: amdsmi_init has no pointer arguments; run_session pairs it with amdsmi_shut_down.
        unsafe { (self.init)(AMDSMI_INIT_AMD_GPUS) }
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>, String> {
        self.gpu_handles().map(|handles| {
            handles
                .into_iter()
                .enumerate()
                .map(|(i, h)| self.device_info(h, u32::try_from(i).unwrap_or(u32::MAX)))
                .collect()
        })
    }

    fn shut_down(&self) -> AmdsmiStatus {
        // SAFETY: run_session calls this once after a successful amdsmi_init; no handle from
        // the session is used afterwards.
        unsafe { (self.shut_down)() }
    }
}

/// amd-smi keeps one process-wide session ("singleton design": inits and shut-downs are
/// reference counted), but overlapping sessions are not safe: an `amdsmi_init` that finds the
/// library already initialising returns success before the first caller has enumerated the
/// sockets, and the first `amdsmi_shut_down` tears the shared state down under the others. On
/// novanas, 8 concurrent discoveries all reported `Ok` with 0 devices (2 × R9700 present).
/// Every user in this process therefore shares one session counted here: the first
/// `session_begin` initialises amd-smi completely under the lock before anyone queries, and
/// only the last `session_end` shuts it down. Discovery sessions and the Phase 3 live telemetry
/// (which keeps its session open for the life of the sampler) use it alike — an `amdsmi_init`
/// / `amdsmi_shut_down` pair of one never tears the library down under the other (novanas lab:
/// a discovery beside running telemetry saw 0 devices, and the telemetry went `unavailable`).
/// A call stuck inside amd-smi's init keeps the lock held; later discoveries then block and
/// report `timeout` at their deadline instead of an empty inventory.
static SESSIONS: Mutex<u32> = Mutex::new(0);

/// Joins the process-wide amd-smi session, running `init` when none is open. A failed `init`
/// leaves nothing open (and is never shut down).
pub(crate) fn session_begin(init: impl FnOnce() -> AmdsmiStatus) -> Result<(), String> {
    let mut open = SESSIONS.lock().unwrap_or_else(PoisonError::into_inner);
    if *open == 0 {
        check(init(), "amdsmi_init")?;
    }
    *open += 1;
    Ok(())
}

/// Leaves the process-wide amd-smi session; the last user runs `shut_down`.
pub(crate) fn session_end(shut_down: impl FnOnce() -> AmdsmiStatus) {
    let mut open = SESSIONS.lock().unwrap_or_else(PoisonError::into_inner);
    *open = open.saturating_sub(1);
    if *open == 0 {
        let _ = shut_down();
    }
}

/// Run one complete init → query → shut_down session within the process-wide session.
fn run_session(session: &impl Session) -> Result<Vec<DeviceInfo>, String> {
    session_begin(|| session.init())?;
    let result = session.devices();
    session_end(|| session.shut_down());
    result
}

impl DiscoveryBackend for AmdSmiBackend {
    fn vendor(&self) -> Vendor {
        Vendor::Amd
    }

    fn discover(&mut self) -> Result<Vec<DeviceInfo>, String> {
        let api = Api::load(&self.library)?;
        let result = run_session(&api);
        // Keep the library mapped for the process lifetime: unloading ROCm libraries that
        // started helper threads is unsafe, and Phase 3 telemetry reloads it anyway.
        std::mem::forget(api);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The session tests share the process-wide session count.
    static SESSION_TESTS: Mutex<()> = Mutex::new(());

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

    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    /// The process-wide amd-smi state as the real library behaves: inits and shut-downs are
    /// reference counted, only the first init enumerates the sockets (slowly), a nested init
    /// returns success at once, and the last shut-down clears the sockets.
    #[derive(Default)]
    struct FakeLibrary {
        refs: Mutex<u32>,
        enumerated: AtomicBool,
    }

    struct FakeSession(Arc<FakeLibrary>);

    const FAKE_GPUS: u32 = 2;

    impl Session for FakeSession {
        fn init(&self) -> AmdsmiStatus {
            let first = {
                let mut refs = self.0.refs.lock().unwrap();
                *refs += 1;
                *refs == 1
            };
            if first {
                std::thread::sleep(Duration::from_millis(20));
                self.0.enumerated.store(true, Ordering::SeqCst);
            }
            AMDSMI_STATUS_SUCCESS
        }

        fn devices(&self) -> Result<Vec<DeviceInfo>, String> {
            let n = if self.0.enumerated.load(Ordering::SeqCst) {
                FAKE_GPUS
            } else {
                0
            };
            std::thread::sleep(Duration::from_millis(5));
            Ok((0..n)
                .map(|i| DeviceInfo {
                    index: DeviceId(0),
                    vendor: Vendor::Amd,
                    vendor_index: i,
                    name: "fake R9700".into(),
                    uuid: None,
                    pci_bus_id: None,
                    arch: Some("gfx1201".into()),
                    driver_version: None,
                    memory: DeviceMemoryInfo {
                        kind: MemoryKind::Dedicated,
                        total_bytes: 1,
                        shared_with_host: false,
                    },
                })
                .collect())
        }

        fn shut_down(&self) -> AmdsmiStatus {
            let mut refs = self.0.refs.lock().unwrap();
            *refs -= 1;
            if *refs == 0 {
                self.0.enumerated.store(false, Ordering::SeqCst);
            }
            AMDSMI_STATUS_SUCCESS
        }
    }

    /// Catches: concurrent discoveries overlapping amd-smi sessions, so a nested init sees no
    /// sockets or a shut-down clears them under another thread (novanas: 0 of 2 R9700s).
    #[test]
    fn concurrent_sessions_each_see_every_device() {
        let _serial = SESSION_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        const THREADS: usize = 8;
        let lib = Arc::new(FakeLibrary::default());
        let barrier = Barrier::new(THREADS);
        let short = AtomicU32::new(0);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    let session = FakeSession(Arc::clone(&lib));
                    barrier.wait();
                    let found = run_session(&session).expect("session");
                    if found.len() != FAKE_GPUS as usize {
                        short.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(
            short.load(Ordering::SeqCst),
            0,
            "discoveries that saw fewer than {FAKE_GPUS} devices"
        );
        assert_eq!(*lib.refs.lock().unwrap(), 0, "every init is shut down");
    }

    /// Catches: a failed init still being shut down (amd-smi's reference count would underflow).
    #[test]
    fn failed_init_is_not_shut_down() {
        struct Failing(AtomicU32);
        impl Session for Failing {
            fn init(&self) -> AmdsmiStatus {
                8
            }
            fn devices(&self) -> Result<Vec<DeviceInfo>, String> {
                panic!("queried without a session")
            }
            fn shut_down(&self) -> AmdsmiStatus {
                self.0.fetch_add(1, Ordering::SeqCst);
                AMDSMI_STATUS_SUCCESS
            }
        }
        let _serial = SESSION_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let failing = Failing(AtomicU32::new(0));
        let err = run_session(&failing).expect_err("init failure is reported");
        assert!(err.contains("amdsmi_init"), "{err}");
        assert_eq!(failing.0.load(Ordering::SeqCst), 0);
        assert_eq!(
            *SESSIONS.lock().unwrap(),
            0,
            "a failed init leaves no session open"
        );
    }

    /// Catches (P3 lab, novanas): discovery running beside the live telemetry, whose amd-smi
    /// session stays open for the sampler's life — a discovery's shut-down tore the shared
    /// session down under the telemetry (then `unavailable`), and a discovery that joined an
    /// open session was mistaken for a fresh one. Every discovery sees every device, the held
    /// session survives them, and the library is shut down once, when the holder leaves.
    #[test]
    fn a_held_session_survives_discoveries() {
        let _serial = SESSION_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let lib = Arc::new(FakeLibrary::default());
        let held = FakeSession(Arc::clone(&lib));
        session_begin(|| held.init()).expect("telemetry session");
        for _ in 0..3 {
            let found = run_session(&FakeSession(Arc::clone(&lib))).expect("discovery");
            assert_eq!(found.len(), FAKE_GPUS as usize);
        }
        assert!(
            lib.enumerated.load(Ordering::SeqCst),
            "a discovery shut the held session down"
        );
        assert_eq!(*lib.refs.lock().unwrap(), 1, "one init for every user");
        session_end(|| held.shut_down());
        assert_eq!(*lib.refs.lock().unwrap(), 0);
        assert_eq!(*SESSIONS.lock().unwrap(), 0);
    }
}
