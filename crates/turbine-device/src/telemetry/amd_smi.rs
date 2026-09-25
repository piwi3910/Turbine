//! amd-smi telemetry (the `amd_smi` discovery kind's vendor tick), loaded at runtime through
//! `libloading`. Layouts and values are copied from ROCm 7.14.1 `include/amd_smi/amdsmi.h`
//! (lines 3158, 3215, 3280, 4302, 4331, 7623, 7641, 7661, 7699, 7737); the layout test pins the
//! sizes the C compiler reports.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::ptr;

use libloading::Library;
use turbine_core::telemetry::{DeviceSample, SourceStatus, ThrottleReasons};
use turbine_core::types::Vendor;

use super::VendorTelemetry;
use crate::discovery::amd_smi::{session_begin, session_end};
use crate::inventory::DeviceInfo;

/// Loader search name when `devices.amd_smi_library` is not set (same as discovery).
const AMD_SMI_DEFAULT_LIBRARY: &str = "libamd_smi.so";

type AmdsmiStatus = u32; // amdsmi_status_t (C enum)
type AmdsmiHandle = *mut c_void; // amdsmi_socket_handle / amdsmi_processor_handle

const AMDSMI_STATUS_SUCCESS: AmdsmiStatus = 0;
const AMDSMI_INIT_AMD_GPUS: u64 = 1 << 1;
const AMDSMI_PROCESSOR_TYPE_AMD_GPU: u32 = 1;
const AMDSMI_TEMPERATURE_TYPE_HOTSPOT: u32 = 1;
const AMDSMI_TEMP_CURRENT: u32 = 0;
const AMDSMI_TEMP_CRITICAL: u32 = 5;
const AMDSMI_CLK_TYPE_GFX: u32 = 0;
const AMDSMI_MEM_TYPE_VRAM: u32 = 0;
/// `amdsmi_violation_status_t` is 6,016 bytes with `AMDSMI_MAX_NUM_XCP = AMDSMI_MAX_NUM_XCC = 8`;
/// the 8 KiB buffer leaves room for a newer library that grows the struct.
const VIOLATION_WORDS: usize = 1024;
/// Byte offsets of `active_prochot_thrm`, `active_ppt_pwr`, `active_socket_thrm` (after 15
/// `uint64_t` fields).
const ACTIVE_PROCHOT_THRM: usize = 120;
const ACTIVE_PPT_PWR: usize = 121;
const ACTIVE_SOCKET_THRM: usize = 122;

/// `amdsmi_clk_info_t` (32 bytes).
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)]
struct AmdsmiClkInfo {
    clk: u32,
    min_clk: u32,
    max_clk: u32,
    clk_locked: u8,
    clk_deep_sleep: u8,
    reserved: [u32; 4],
}

/// `amdsmi_power_info_t` (192 bytes); unsupported members are `UINT32_MAX`.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)]
struct AmdsmiPowerInfo {
    socket_power: u64,
    current_socket_power: u32,
    average_socket_power: u32,
    gfx_voltage: u64,
    soc_voltage: u64,
    mem_voltage: u64,
    power_limit: u32,
    ubb_power: u32,
    reserved: [u64; 18],
}

/// `amdsmi_engine_usage_t` (64 bytes).
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)]
struct AmdsmiEngineUsage {
    gfx_activity: u32,
    umc_activity: u32,
    mm_activity: u32,
    reserved: [u32; 13],
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
type GetTempMetricFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    sensor_type: u32,
    metric: u32,
    temperature: *mut i64,
) -> AmdsmiStatus;
type GetClockInfoFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    clk_type: u32,
    info: *mut AmdsmiClkInfo,
) -> AmdsmiStatus;
type GetPowerInfoFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    info: *mut AmdsmiPowerInfo,
) -> AmdsmiStatus;
type GetGpuActivityFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    info: *mut AmdsmiEngineUsage,
) -> AmdsmiStatus;
type GetMemoryFn = unsafe extern "C" fn(
    processor_handle: AmdsmiHandle,
    mem_type: u32,
    value: *mut u64,
) -> AmdsmiStatus;
/// `amdsmi_get_violation_status(handle, amdsmi_violation_status_t*)`, written into an 8-byte
/// aligned buffer larger than the struct.
type GetViolationStatusFn =
    unsafe extern "C" fn(processor_handle: AmdsmiHandle, info: *mut u64) -> AmdsmiStatus;

/// amd-smi telemetry for every AMD GPU (`vendor_index` = amd-smi GPU enumeration order, as in
/// discovery).
pub struct AmdSmiTelemetry {
    shut_down: ShutDownFn,
    temp_metric: GetTempMetricFn,
    clock_info: GetClockInfoFn,
    power_info: GetPowerInfoFn,
    gpu_activity: GetGpuActivityFn,
    memory_usage: GetMemoryFn,
    memory_total: GetMemoryFn,
    violation_status: Option<GetViolationStatusFn>,
    handles: Vec<AmdsmiHandle>,
    /// Never unloaded: unloading ROCm libraries that started helper threads is unsafe (the same
    /// rule as discovery). Owning it keeps every fn pointer above valid.
    _lib: ManuallyDrop<Library>,
}

// SAFETY: amd-smi processor handles are opaque identifiers valid process-wide between
// `amdsmi_init` and `amdsmi_shut_down`, and amd-smi's query functions are thread-safe. The value
// is moved to one vendor worker thread and only used there, one call at a time (`&mut self`).
unsafe impl Send for AmdSmiTelemetry {}

fn amd_check(status: AmdsmiStatus, call: &str) -> Result<(), String> {
    if status == AMDSMI_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{call} failed with amdsmi_status_t {status}"))
    }
}

impl AmdSmiTelemetry {
    /// `path` = `devices.amd_smi_library` / `TURBINE_AMD_SMI_LIBRARY`; `None` = the loader's
    /// `libamd_smi.so`.
    pub fn open(path: Option<&Path>) -> Result<Self, String> {
        let path = path.map_or_else(|| PathBuf::from(AMD_SMI_DEFAULT_LIBRARY), Path::to_path_buf);
        // SAFETY: loading amd-smi runs its static initialisers, which have no preconditions. The
        // Library is stored (never unloaded) in the returned value, so every fn pointer resolved
        // from it stays valid for the value's lifetime.
        let lib = unsafe { Library::new(&path) }.map_err(|e| format!("{}: {e}", path.display()))?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // SAFETY: `$ty` is the exact C signature declared in amdsmi.h for `$name`; the
                // pointer is copied out and only used while `lib` (kept by the value) is loaded.
                let s = unsafe { lib.get::<$ty>($name) }
                    .map_err(|e| format!("{}: missing symbol {}: {e}", path.display(), $name))?;
                *s
            }};
        }
        let init = sym!("amdsmi_init", InitFn);
        let get_socket_handles = sym!("amdsmi_get_socket_handles", GetSocketHandlesFn);
        let get_processor_handles = sym!("amdsmi_get_processor_handles", GetProcessorHandlesFn);
        let get_processor_type = sym!("amdsmi_get_processor_type", GetProcessorTypeFn);
        let shut_down = sym!("amdsmi_shut_down", ShutDownFn);
        let temp_metric = sym!("amdsmi_get_temp_metric", GetTempMetricFn);
        let clock_info = sym!("amdsmi_get_clock_info", GetClockInfoFn);
        let power_info = sym!("amdsmi_get_power_info", GetPowerInfoFn);
        let gpu_activity = sym!("amdsmi_get_gpu_activity", GetGpuActivityFn);
        let memory_usage = sym!("amdsmi_get_gpu_memory_usage", GetMemoryFn);
        let memory_total = sym!("amdsmi_get_gpu_memory_total", GetMemoryFn);
        // SAFETY: optional symbol (older libraries lack it); same signature rule as `sym!`.
        let violation_status =
            unsafe { lib.get::<GetViolationStatusFn>("amdsmi_get_violation_status") }
                .ok()
                .map(|s| *s);

        // Joins the process-wide amd-smi session shared with discovery; left in Drop (or below
        // on failure).
        // SAFETY: amdsmi_init takes a flag word and no pointers.
        session_begin(|| unsafe { init(AMDSMI_INIT_AMD_GPUS) })?;
        let handles = match gpu_handles(
            get_socket_handles,
            get_processor_handles,
            get_processor_type,
        ) {
            Ok(h) => h,
            Err(e) => {
                // SAFETY: amdsmi_shut_down takes no arguments; it runs only when this was the
                // last open session, and no handle escapes.
                session_end(|| unsafe { shut_down() });
                std::mem::forget(lib);
                return Err(e);
            }
        };
        Ok(AmdSmiTelemetry {
            shut_down,
            temp_metric,
            clock_info,
            power_info,
            gpu_activity,
            memory_usage,
            memory_total,
            violation_status,
            handles,
            _lib: ManuallyDrop::new(lib),
        })
    }

    fn throttle(&self, h: AmdsmiHandle) -> ThrottleReasons {
        let mut throttle = ThrottleReasons::default();
        let Some(violation_status) = self.violation_status else {
            return throttle;
        };
        let mut buf = vec![0u64; VIOLATION_WORDS];
        // SAFETY: `h` is a live processor handle; `buf` is an exclusively borrowed, 8-byte
        // aligned, zeroed 8 KiB buffer, larger than amdsmi_violation_status_t (6,016 bytes).
        let st = unsafe { violation_status(h, buf.as_mut_ptr()) };
        if st == AMDSMI_STATUS_SUCCESS {
            let bytes: Vec<u8> = buf.iter().take(16).flat_map(|w| w.to_ne_bytes()).collect();
            // 1 = active, 0 = inactive, UINT8_MAX = unsupported.
            let active = |i: usize| bytes[i] == 1;
            throttle.thermal = active(ACTIVE_PROCHOT_THRM) || active(ACTIVE_SOCKET_THRM);
            throttle.power = active(ACTIVE_PPT_PWR);
        }
        throttle
    }
}

/// All AMD GPU processor handles, socket by socket (amd-smi order, as in discovery).
fn gpu_handles(
    get_socket_handles: GetSocketHandlesFn,
    get_processor_handles: GetProcessorHandlesFn,
    get_processor_type: GetProcessorTypeFn,
) -> Result<Vec<AmdsmiHandle>, String> {
    let mut socket_count: u32 = 0;
    // SAFETY: a NULL handle array asks only for the count, written to an exclusively borrowed u32.
    let st = unsafe { get_socket_handles(&mut socket_count, ptr::null_mut()) };
    amd_check(st, "amdsmi_get_socket_handles")?;
    let mut sockets: Vec<AmdsmiHandle> = vec![ptr::null_mut(); socket_count as usize];
    // SAFETY: `sockets` has room for exactly `socket_count` handles, the limit passed in.
    let st = unsafe { get_socket_handles(&mut socket_count, sockets.as_mut_ptr()) };
    amd_check(st, "amdsmi_get_socket_handles")?;
    sockets.truncate(socket_count as usize);

    let mut gpus = Vec::new();
    for socket in sockets {
        let mut n: u32 = 0;
        // SAFETY: `socket` came from amdsmi_get_socket_handles in this session; NULL = count only.
        let st = unsafe { get_processor_handles(socket, &mut n, ptr::null_mut()) };
        amd_check(st, "amdsmi_get_processor_handles")?;
        let mut procs: Vec<AmdsmiHandle> = vec![ptr::null_mut(); n as usize];
        // SAFETY: `procs` has room for exactly `n` handles, the limit passed in.
        let st = unsafe { get_processor_handles(socket, &mut n, procs.as_mut_ptr()) };
        amd_check(st, "amdsmi_get_processor_handles")?;
        procs.truncate(n as usize);
        for p in procs {
            let mut kind: u32 = 0;
            // SAFETY: `p` is a live processor handle; `kind` is an exclusively borrowed u32.
            let st = unsafe { get_processor_type(p, &mut kind) };
            if st == AMDSMI_STATUS_SUCCESS && kind == AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                gpus.push(p);
            }
        }
    }
    Ok(gpus)
}

impl Drop for AmdSmiTelemetry {
    fn drop(&mut self) {
        let shut_down = self.shut_down;
        // SAFETY: amdsmi_shut_down takes no arguments; it runs only when this was the last open
        // session, and no handle of this value is used afterwards.
        session_end(|| unsafe { shut_down() });
    }
}

impl VendorTelemetry for AmdSmiTelemetry {
    fn vendor(&self) -> Vendor {
        Vendor::Amd
    }

    fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String> {
        let h = *self
            .handles
            .get(device.vendor_index as usize)
            .ok_or_else(|| format!("no amd-smi GPU handle {}", device.vendor_index))?;
        let ok = |st: AmdsmiStatus| st == AMDSMI_STATUS_SUCCESS;

        let mut temp: i64 = 0;
        // SAFETY: `h` is a live processor handle from gpu_handles(); `temp` is an exclusively
        // borrowed int64_t; the enum arguments are valid amdsmi.h values.
        let temp_ok = ok(unsafe {
            (self.temp_metric)(
                h,
                AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                AMDSMI_TEMP_CURRENT,
                &mut temp,
            )
        });
        let mut crit: i64 = 0;
        // SAFETY: as above, for the critical limit of the same sensor.
        let crit_ok = ok(unsafe {
            (self.temp_metric)(
                h,
                AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                AMDSMI_TEMP_CRITICAL,
                &mut crit,
            )
        });
        let mut clk = AmdsmiClkInfo::default();
        // SAFETY: `clk` is an exclusively borrowed value of the exact C type the call writes.
        let clk_ok = ok(unsafe { (self.clock_info)(h, AMDSMI_CLK_TYPE_GFX, &mut clk) });
        let mut power = AmdsmiPowerInfo::default();
        // SAFETY: `power` is an exclusively borrowed value of the exact C type the call writes.
        let power_ok = ok(unsafe { (self.power_info)(h, &mut power) });
        let mut usage = AmdsmiEngineUsage::default();
        // SAFETY: `usage` is an exclusively borrowed value of the exact C type the call writes.
        let usage_ok = ok(unsafe { (self.gpu_activity)(h, &mut usage) });
        let mut used: u64 = 0;
        // SAFETY: `used` is an exclusively borrowed uint64_t; VRAM is a valid enum value.
        let used_ok = ok(unsafe { (self.memory_usage)(h, AMDSMI_MEM_TYPE_VRAM, &mut used) });
        let mut total: u64 = 0;
        // SAFETY: `total` is an exclusively borrowed uint64_t; VRAM is a valid enum value.
        let total_ok = ok(unsafe { (self.memory_total)(h, AMDSMI_MEM_TYPE_VRAM, &mut total) });

        if !temp_ok && !clk_ok {
            return Err("amd-smi returned neither temperature nor clock".into());
        }
        let valid = |w: u32| w != 0 && w != u32::MAX;
        let watts = if valid(power.current_socket_power) {
            Some(power.current_socket_power)
        } else if valid(power.average_socket_power) {
            Some(power.average_socket_power)
        } else {
            None
        };
        Ok(DeviceSample {
            device: device.index,
            memory_used_bytes: used_ok.then_some(used),
            memory_free_bytes: (used_ok && total_ok).then(|| total.saturating_sub(used)),
            temperature_c: temp_ok.then_some(temp as f64),
            // The vendor slowdown point: amd-smi's critical limit of the hotspot sensor.
            slowdown_temperature_c: crit_ok.then_some(crit as f64),
            clock_mhz: clk_ok.then_some(clk.clk),
            power_watts: power_ok.then_some(watts).flatten().map(f64::from),
            utilization: usage_ok.then(|| f64::from(usage.gfx_activity.min(100)) / 100.0),
            throttle: self.throttle(h),
            status: SourceStatus::Ok,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_amdsmi_h() {
        assert_eq!(std::mem::size_of::<AmdsmiClkInfo>(), 32);
        assert_eq!(std::mem::offset_of!(AmdsmiClkInfo, reserved), 16);
        assert_eq!(std::mem::size_of::<AmdsmiPowerInfo>(), 192);
        assert_eq!(
            std::mem::offset_of!(AmdsmiPowerInfo, average_socket_power),
            12
        );
        assert_eq!(std::mem::size_of::<AmdsmiEngineUsage>(), 64);
        const { assert!(VIOLATION_WORDS * 8 >= 6016) };
    }

    #[test]
    fn missing_library_yields_error_not_panic() {
        assert!(AmdSmiTelemetry::open(Some(Path::new("/nonexistent/libamd_smi.so"))).is_err());
    }
}
