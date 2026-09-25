//! Live telemetry sampler (P3 S-5): a fast tick (default every 100 ms) reads the host `/proc`
//! files and the reservation ledger, a vendor tick (default every 1 s) reads each device through
//! NVML or amd-smi with a deadline per call, and every fast tick publishes one
//! [`TelemetrySample`] through a lock-free latest-value cell ([`LatestSample`]).
//!
//! Vendor calls run on one worker thread per vendor library, so a hung driver call can never
//! block the sampler: a missed deadline marks the device `stale` and later ticks skip the busy
//! worker until it answers.

pub mod amd_smi;
pub mod nvml;
pub mod proc;
pub mod vendor;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;

use arc_swap::ArcSwap;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_core::clock::Clock;
use turbine_core::config::ReliabilityTelemetryConfig;
use turbine_core::telemetry::{
    DeviceSample, HostSample, LedgerProbe, LedgerSample, SourceStatus, TelemetrySample,
};
use turbine_core::types::{MemoryKind, Vendor};
use turbine_observability::MetricsRegistry;

use crate::inventory::{DeviceInfo, DeviceInventory};
use proc::{ProcFile, ProcSource, parse_meminfo, parse_psi, parse_vmstat};

/// After a stall longer than this many intervals, missed ticks are skipped instead of replayed.
const MAX_CATCH_UP_INTERVALS: u32 = 10;

/// Sampler timing, resolved from `reliability.telemetry.*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelemetryConfig {
    /// Fast tick: `/proc` files and the ledger.
    pub interval: Duration,
    /// Vendor tick: NVML / amd-smi.
    pub vendor_interval: Duration,
    /// Deadline per vendor call.
    pub call_timeout: Duration,
    /// How long a source may stay stale before the `telemetry_stale` signal fires (read by the
    /// pressure controller).
    pub stale_after: Duration,
}

impl TelemetryConfig {
    pub fn from_config(cfg: &ReliabilityTelemetryConfig) -> Self {
        TelemetryConfig {
            interval: cfg.interval.0,
            vendor_interval: cfg.vendor_interval.0,
            call_timeout: cfg.call_timeout.0,
            stale_after: cfg.stale_after.0,
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        TelemetryConfig::from_config(&ReliabilityTelemetryConfig::default())
    }
}

/// One vendor library (NVML or amd-smi) serving every device of its vendor; fakes in tests.
/// `sample` may block; the sampler calls it on a dedicated worker thread with a deadline.
pub trait VendorTelemetry: Send {
    fn vendor(&self) -> Vendor;
    fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String>;
}

/// Lock-free latest-value cell: readers never block the sampler and always see a whole sample.
#[derive(Clone)]
pub struct LatestSample(Arc<ArcSwap<TelemetrySample>>);

impl LatestSample {
    /// A cell holding an empty sample (every source unavailable) until the first fast tick.
    pub fn new() -> Self {
        LatestSample(Arc::new(ArcSwap::from_pointee(TelemetrySample {
            at_mono_ns: 0,
            host: HostSample {
                status: SourceStatus::Unavailable,
                ..HostSample::default()
            },
            devices: Vec::new(),
            ledger: LedgerSample::default(),
            storage: None,
        })))
    }

    pub fn load(&self) -> Arc<TelemetrySample> {
        self.0.load_full()
    }

    fn store(&self, sample: TelemetrySample) {
        self.0.store(Arc::new(sample));
    }
}

impl Default for LatestSample {
    fn default() -> Self {
        LatestSample::new()
    }
}

/// One fast-tick host read. `sample.status` follows `/proc/meminfo`; `vmstat` and `psi` report
/// their own files (a kernel without PSI leaves only `psi` unavailable).
#[derive(Clone, Debug, PartialEq)]
pub struct HostRead {
    pub sample: HostSample,
    pub vmstat: SourceStatus,
    pub psi: SourceStatus,
}

/// Read and parse the three host files; never fails and never panics.
pub fn read_host(proc: &dyn ProcSource) -> HostRead {
    let mut sample = HostSample {
        status: SourceStatus::Unavailable,
        ..HostSample::default()
    };
    if let Some(m) = proc
        .read(ProcFile::Meminfo)
        .ok()
        .and_then(|s| parse_meminfo(&s).ok())
    {
        sample.mem_available_bytes = Some(m.mem_available_bytes);
        sample.swap_total_bytes = m.swap_total_bytes;
        sample.swap_free_bytes = m.swap_free_bytes;
        sample.status = SourceStatus::Ok;
    }
    let vmstat = match proc
        .read(ProcFile::Vmstat)
        .ok()
        .and_then(|s| parse_vmstat(&s).ok())
    {
        Some(v) => {
            sample.pswpin_total = Some(v.pswpin);
            SourceStatus::Ok
        }
        None => SourceStatus::Unavailable,
    };
    let psi = match proc
        .read(ProcFile::PressureMemory)
        .ok()
        .and_then(|s| parse_psi(&s).ok())
    {
        Some(p) => {
            sample.psi_memory_some_avg10 = Some(p.some_avg10);
            SourceStatus::Ok
        }
        None => SourceStatus::Unavailable,
    };
    HostRead {
        sample,
        vmstat,
        psi,
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct DeviceLabel {
    device: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct DeviceKindLabels {
    device: u32,
    /// `used` or `free`.
    kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct DeviceReasonLabels {
    device: u32,
    /// `thermal`, `power` or `other`.
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct SourceLabel {
    /// `host`, `storage` or the Phase 0 device index.
    source: String,
}

type FloatGauge = Gauge<f64, AtomicU64>;
type CallHistogram = Family<SourceLabel, Histogram, fn() -> Histogram>;

fn call_histogram() -> Histogram {
    // 0.5 ms .. ~4 s: covers healthy calls and every configurable deadline.
    Histogram::new(exponential_buckets(0.0005, 2.0, 14))
}

fn gauge_i64(v: u64) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// GPU and host telemetry metric families (P3 §Interfaces "Metrics", owner `turbine-device`).
#[derive(Clone)]
pub struct TelemetryMetrics {
    gpu_temperature_celsius: Family<DeviceLabel, FloatGauge>,
    gpu_clock_mhz: Family<DeviceLabel, Gauge>,
    gpu_power_watts: Family<DeviceLabel, FloatGauge>,
    gpu_utilization_ratio: Family<DeviceLabel, FloatGauge>,
    gpu_memory_bytes: Family<DeviceKindLabels, Gauge>,
    gpu_throttle_active: Family<DeviceReasonLabels, Gauge>,
    host_memory_available_bytes: Gauge,
    host_psi_memory_some_avg10: FloatGauge,
    host_swap_in_pages_per_second: FloatGauge,
    telemetry_stale: Family<SourceLabel, Gauge>,
    telemetry_call_duration_seconds: CallHistogram,
}

impl TelemetryMetrics {
    pub fn register(reg: &MetricsRegistry) -> Self {
        TelemetryMetrics {
            gpu_temperature_celsius: reg.register(
                "turbine_gpu_temperature_celsius",
                "GPU temperature (vendor tick)",
                Family::default(),
            ),
            gpu_clock_mhz: reg.register(
                "turbine_gpu_clock_mhz",
                "GPU SM/GFX clock (vendor tick)",
                Family::default(),
            ),
            gpu_power_watts: reg.register(
                "turbine_gpu_power_watts",
                "GPU power draw (vendor tick)",
                Family::default(),
            ),
            gpu_utilization_ratio: reg.register(
                "turbine_gpu_utilization_ratio",
                "GPU utilisation, 0 to 1 (vendor tick)",
                Family::default(),
            ),
            gpu_memory_bytes: reg.register(
                "turbine_gpu_memory_bytes",
                "GPU memory used and free, dedicated-memory devices only",
                Family::default(),
            ),
            gpu_throttle_active: reg.register(
                "turbine_gpu_throttle_active",
                "1 while the vendor reports a throttle reason active",
                Family::default(),
            ),
            host_memory_available_bytes: reg.register(
                "turbine_host_memory_available_bytes",
                "Host MemAvailable from /proc/meminfo",
                Gauge::default(),
            ),
            host_psi_memory_some_avg10: reg.register(
                "turbine_host_psi_memory_some_avg10",
                "Host PSI memory some avg10 from /proc/pressure/memory",
                FloatGauge::default(),
            ),
            host_swap_in_pages_per_second: reg.register(
                "turbine_host_swap_in_pages_per_second",
                "Host swap-in rate from /proc/vmstat pswpin deltas",
                FloatGauge::default(),
            ),
            telemetry_stale: reg.register(
                "turbine_telemetry_stale",
                "1 while a telemetry source is stale",
                Family::default(),
            ),
            telemetry_call_duration_seconds: reg.register(
                "turbine_telemetry_call_duration_seconds",
                "Time the sampler waited for one vendor telemetry call",
                Family::new_with_constructor(call_histogram as fn() -> Histogram),
            ),
        }
    }

    /// Record one device's vendor readings; device memory only for dedicated-memory devices.
    pub fn record_device(&self, d: &DeviceSample, kind: MemoryKind) {
        let device = d.device.0;
        let label = DeviceLabel { device };
        if let Some(t) = d.temperature_c {
            self.gpu_temperature_celsius.get_or_create(&label).set(t);
        }
        if let Some(c) = d.clock_mhz {
            self.gpu_clock_mhz.get_or_create(&label).set(i64::from(c));
        }
        if let Some(p) = d.power_watts {
            self.gpu_power_watts.get_or_create(&label).set(p);
        }
        if let Some(u) = d.utilization {
            self.gpu_utilization_ratio.get_or_create(&label).set(u);
        }
        if kind == MemoryKind::Dedicated {
            for (k, v) in [("used", d.memory_used_bytes), ("free", d.memory_free_bytes)] {
                if let Some(v) = v {
                    self.gpu_memory_bytes
                        .get_or_create(&DeviceKindLabels { device, kind: k })
                        .set(gauge_i64(v));
                }
            }
        }
        for (reason, on) in [
            ("thermal", d.throttle.thermal),
            ("power", d.throttle.power),
            ("other", d.throttle.other),
        ] {
            self.gpu_throttle_active
                .get_or_create(&DeviceReasonLabels { device, reason })
                .set(i64::from(on));
        }
        self.set_stale(&device.to_string(), d.status == SourceStatus::Stale);
    }

    fn set_stale(&self, source: &str, stale: bool) {
        self.telemetry_stale
            .get_or_create(&SourceLabel {
                source: source.to_string(),
            })
            .set(i64::from(stale));
    }

    fn observe_call(&self, source: &str, seconds: f64) {
        self.telemetry_call_duration_seconds
            .get_or_create(&SourceLabel {
                source: source.to_string(),
            })
            .observe(seconds);
    }
}

type VendorReply = Result<DeviceSample, String>;

/// The sampler's end of one vendor worker thread.
struct VendorWorker {
    vendor: Vendor,
    requests: SyncSender<DeviceInfo>,
    replies: Receiver<VendorReply>,
    /// A call missed its deadline and has not answered yet.
    busy: bool,
}

struct DeviceSlot {
    info: DeviceInfo,
    worker: Option<usize>,
    last: DeviceSample,
    ever_ok: bool,
    warned_unavailable: bool,
}

/// The sampler state machine, driven by [`SamplerCore::poll`]; [`TelemetrySampler`] runs it on
/// a thread, tests drive it directly with a fake clock.
pub struct SamplerCore {
    cfg: TelemetryConfig,
    clock: Arc<dyn Clock>,
    proc: Box<dyn ProcSource>,
    ledger: Arc<dyn LedgerProbe>,
    workers: Vec<VendorWorker>,
    devices: Vec<DeviceSlot>,
    host_warned: bool,
    last_pswpin: Option<(u64, Duration)>,
    next_fast: Duration,
    next_vendor: Duration,
    latest: LatestSample,
    metrics: Option<TelemetryMetrics>,
    fast_ticks: u64,
    vendor_ticks: u64,
}

impl SamplerCore {
    /// One worker thread per vendor backend; a device whose vendor has no backend stays
    /// `unavailable` (logged once).
    pub fn new(
        cfg: TelemetryConfig,
        inventory: &DeviceInventory,
        vendor: Vec<Box<dyn VendorTelemetry>>,
        proc: Box<dyn ProcSource>,
        ledger: Arc<dyn LedgerProbe>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let workers: Vec<VendorWorker> = vendor.into_iter().filter_map(spawn_worker).collect();
        let devices = inventory
            .devices
            .iter()
            .map(|d| {
                let worker = workers.iter().position(|w| w.vendor == d.vendor);
                if worker.is_none() {
                    tracing::warn!(
                        event = "telemetry_stale",
                        reason = "unavailable",
                        source = d.index.0,
                        vendor = d.vendor.as_str(),
                        "no vendor telemetry library for this device"
                    );
                }
                DeviceSlot {
                    info: d.clone(),
                    worker,
                    last: DeviceSample::empty(d.index, SourceStatus::Unavailable),
                    ever_ok: false,
                    warned_unavailable: worker.is_none(),
                }
            })
            .collect();
        let now = clock.now_mono();
        SamplerCore {
            cfg,
            clock,
            proc,
            ledger,
            workers,
            devices,
            host_warned: false,
            last_pswpin: None,
            next_fast: now,
            next_vendor: now,
            latest: LatestSample::new(),
            metrics: None,
            fast_ticks: 0,
            vendor_ticks: 0,
        }
    }

    pub fn with_metrics(mut self, metrics: TelemetryMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// The cell every fast tick publishes into.
    pub fn latest(&self) -> LatestSample {
        self.latest.clone()
    }

    pub fn fast_ticks(&self) -> u64 {
        self.fast_ticks
    }

    pub fn vendor_ticks(&self) -> u64 {
        self.vendor_ticks
    }

    /// Run every tick due at the clock's current time and return the time until the next one.
    /// Blocks at most `call_timeout` per busy-free vendor worker, never on a hung call.
    pub fn poll(&mut self) -> Duration {
        let now = self.clock.now_mono();
        if now >= self.next_vendor {
            self.vendor_tick();
            self.next_vendor = next_due(self.next_vendor, self.cfg.vendor_interval, now);
        }
        if now >= self.next_fast {
            // The vendor tick may have waited on a deadline: stamp the sample with fresh time.
            self.fast_tick(self.clock.now_mono());
            self.next_fast = next_due(self.next_fast, self.cfg.interval, now);
        }
        self.next_fast
            .min(self.next_vendor)
            .saturating_sub(self.clock.now_mono())
    }

    fn fast_tick(&mut self, now: Duration) {
        self.fast_ticks += 1;
        let read = read_host(self.proc.as_ref());
        if read.sample.status == SourceStatus::Unavailable && !self.host_warned {
            tracing::warn!(
                event = "telemetry_stale",
                reason = "unavailable",
                source = "host",
                "cannot read /proc/meminfo; host memory signals are omitted"
            );
            self.host_warned = true;
        }
        let sample = TelemetrySample {
            at_mono_ns: u64::try_from(now.as_nanos()).unwrap_or(u64::MAX),
            host: read.sample,
            devices: self.devices.iter().map(|d| d.last.clone()).collect(),
            ledger: LedgerSample {
                kv_utilization: self.ledger.kv_utilization(),
                queue_fill: self.ledger.queue_fill(),
            },
            storage: None,
        };
        if let Some(m) = &self.metrics {
            if let Some(avail) = sample.host.mem_available_bytes {
                m.host_memory_available_bytes.set(gauge_i64(avail));
            }
            if let Some(psi) = sample.host.psi_memory_some_avg10 {
                m.host_psi_memory_some_avg10.set(psi);
            }
            m.set_stale("host", sample.host.status == SourceStatus::Stale);
        }
        if let Some(pswpin) = sample.host.pswpin_total {
            if let Some((prev, at)) = self.last_pswpin
                && now > at
                && let Some(m) = &self.metrics
            {
                let pages = pswpin.saturating_sub(prev) as f64;
                m.host_swap_in_pages_per_second
                    .set(pages / (now - at).as_secs_f64());
            }
            self.last_pswpin = Some((pswpin, now));
        }
        self.latest.store(sample);
    }

    fn vendor_tick(&mut self) {
        self.vendor_ticks += 1;
        for i in 0..self.devices.len() {
            let Some(w) = self.devices[i].worker else {
                continue;
            };
            let started = self.clock.now_mono();
            let reply = self.call(w, i);
            let waited = self.clock.now_mono().saturating_sub(started);
            let call_timeout_ms = u64::try_from(self.cfg.call_timeout.as_millis()).unwrap_or(0);
            let slot = &mut self.devices[i];
            let device = slot.info.index;
            match reply {
                Some(Ok(mut s)) => {
                    s.device = device;
                    s.status = SourceStatus::Ok;
                    if slot.info.memory.kind == MemoryKind::Unified {
                        // Unified devices share host memory: the host signals stand in.
                        s.memory_used_bytes = None;
                        s.memory_free_bytes = None;
                    }
                    slot.last = s;
                    slot.ever_ok = true;
                }
                Some(Err(e)) if !slot.ever_ok => {
                    if !slot.warned_unavailable {
                        tracing::warn!(event = "telemetry_stale", reason = "unavailable", source = device.0, error = %e);
                        slot.warned_unavailable = true;
                    }
                    slot.last = DeviceSample::empty(device, SourceStatus::Unavailable);
                }
                Some(Err(e)) => {
                    if slot.last.status != SourceStatus::Stale {
                        tracing::warn!(event = "telemetry_stale", reason = "call_failed", source = device.0, error = %e);
                    }
                    slot.last.status = SourceStatus::Stale;
                }
                None => {
                    if slot.last.status != SourceStatus::Stale {
                        tracing::warn!(
                            event = "telemetry_stale",
                            reason = "deadline_missed",
                            source = device.0,
                            call_timeout_ms
                        );
                    }
                    slot.last.status = SourceStatus::Stale;
                }
            }
            if let Some(m) = &self.metrics {
                let slot = &self.devices[i];
                m.observe_call(&device.0.to_string(), waited.as_secs_f64());
                m.record_device(&slot.last, slot.info.memory.kind);
            }
        }
    }

    /// One vendor call with a deadline. `None`: the deadline passed, or the worker is still
    /// hung on an earlier call (checked without blocking).
    fn call(&mut self, w: usize, device: usize) -> Option<VendorReply> {
        let worker = &mut self.workers[w];
        if worker.busy {
            match worker.replies.try_recv() {
                // The late answer is out of date; ask again below.
                Ok(_late) => worker.busy = false,
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => {
                    return Some(Err("vendor telemetry worker exited".into()));
                }
            }
        }
        if worker
            .requests
            .try_send(self.devices[device].info.clone())
            .is_err()
        {
            return Some(Err("vendor telemetry worker exited".into()));
        }
        match worker.replies.recv_timeout(self.cfg.call_timeout) {
            Ok(reply) => Some(reply),
            Err(RecvTimeoutError::Timeout) => {
                worker.busy = true;
                None
            }
            Err(RecvTimeoutError::Disconnected) => {
                Some(Err("vendor telemetry worker exited".into()))
            }
        }
    }
}

/// The next due time after `prev`; after a stall of more than [`MAX_CATCH_UP_INTERVALS`] the
/// missed ticks are skipped instead of run back to back.
fn next_due(prev: Duration, interval: Duration, now: Duration) -> Duration {
    let next = prev + interval;
    if next + interval * MAX_CATCH_UP_INTERVALS < now {
        now + interval
    } else {
        next
    }
}

/// Start the worker thread serving one vendor library. The request channel holds one call, so
/// the sampler can never queue work behind a hung call. `None` if the OS refuses a thread.
fn spawn_worker(mut backend: Box<dyn VendorTelemetry>) -> Option<VendorWorker> {
    let vendor = backend.vendor();
    let (req_tx, req_rx) = sync_channel::<DeviceInfo>(1);
    let (rep_tx, rep_rx) = sync_channel::<VendorReply>(1);
    let spawned = std::thread::Builder::new()
        .name(format!("turbine-telemetry-{}", vendor.as_str()))
        .spawn(move || {
            while let Ok(device) = req_rx.recv() {
                if rep_tx.send(backend.sample(&device)).is_err() {
                    break;
                }
            }
        });
    match spawned {
        Ok(_detached) => Some(VendorWorker {
            vendor,
            requests: req_tx,
            replies: rep_rx,
            busy: false,
        }),
        Err(e) => {
            tracing::warn!(event = "telemetry_stale", reason = "unavailable", vendor = vendor.as_str(), error = %e, "cannot start vendor telemetry thread");
            None
        }
    }
}

/// Runs a [`SamplerCore`] on its own thread; dropping it stops and joins the thread. Vendor
/// worker threads are detached: a worker hung in a driver call exits once that call returns.
pub struct TelemetrySampler {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl TelemetrySampler {
    pub fn spawn(
        cfg: TelemetryConfig,
        inventory: &DeviceInventory,
        vendor: Vec<Box<dyn VendorTelemetry>>,
        proc: Box<dyn ProcSource>,
        ledger: Arc<dyn LedgerProbe>,
        clock: Arc<dyn Clock>,
    ) -> (TelemetrySampler, LatestSample) {
        TelemetrySampler::spawn_core(SamplerCore::new(
            cfg, inventory, vendor, proc, ledger, clock,
        ))
    }

    /// # Panics
    /// If the OS cannot start the sampler thread (resource exhaustion at startup).
    pub fn spawn_core(mut core: SamplerCore) -> (TelemetrySampler, LatestSample) {
        let latest = core.latest();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("turbine-telemetry".into())
            .spawn(move || {
                while !flag.load(Ordering::Acquire) {
                    let wait = core.poll();
                    std::thread::sleep(
                        wait.clamp(Duration::from_millis(1), Duration::from_millis(50)),
                    );
                }
            })
            .expect("the OS starts the telemetry sampler thread");
        (
            TelemetrySampler {
                stop,
                thread: Some(thread),
            },
            latest,
        )
    }
}

impl Drop for TelemetrySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use turbine_core::clock::{FakeClock, SystemClock};
    use turbine_core::telemetry::ThrottleReasons;
    use turbine_core::types::DeviceId;
    use turbine_observability::MetricsRegistry;

    use super::proc::{FsProc, MemInfo, ParseError, PsiMemory, VmStat};
    use super::*;
    use crate::inventory::DeviceMemoryInfo;

    const MEMINFO: &str = include_str!("../../tests/fixtures/proc/meminfo");
    const VMSTAT_A: &str = include_str!("../../tests/fixtures/proc/vmstat_a");
    const VMSTAT_B: &str = include_str!("../../tests/fixtures/proc/vmstat_b");
    const PSI: &str = include_str!("../../tests/fixtures/proc/pressure_memory");

    /// Fixture-backed `/proc`; `psi: None` = the file is absent (kernel without PSI). Counts reads.
    struct FakeProc {
        psi: Option<&'static str>,
        reads: Arc<AtomicUsize>,
    }

    impl ProcSource for FakeProc {
        fn read(&self, file: ProcFile) -> std::io::Result<String> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            match file {
                ProcFile::Meminfo => Ok(MEMINFO.to_string()),
                ProcFile::Vmstat => Ok(VMSTAT_A.to_string()),
                ProcFile::PressureMemory => self.psi.map(str::to_string).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, file.relative_path())
                }),
            }
        }
    }

    fn fake_proc(psi: Option<&'static str>) -> (FakeProc, Arc<AtomicUsize>) {
        let reads = Arc::new(AtomicUsize::new(0));
        (
            FakeProc {
                psi,
                reads: Arc::clone(&reads),
            },
            reads,
        )
    }

    struct FakeLedger(Mutex<f64>);

    impl LedgerProbe for FakeLedger {
        fn kv_utilization(&self) -> f64 {
            *self.0.lock().unwrap()
        }
        fn queue_fill(&self) -> f64 {
            0.25
        }
    }

    /// Vendor backend that counts calls and sleeps `block` inside each one.
    struct Counting {
        calls: Arc<AtomicUsize>,
        block: Duration,
    }

    impl VendorTelemetry for Counting {
        fn vendor(&self) -> Vendor {
            Vendor::Amd
        }
        fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(self.block);
            Ok(DeviceSample {
                temperature_c: Some(55.0),
                slowdown_temperature_c: Some(100.0),
                clock_mhz: Some(2350),
                power_watts: Some(180.5),
                utilization: Some(0.5),
                memory_used_bytes: Some(1 << 30),
                memory_free_bytes: Some(31 << 30),
                throttle: ThrottleReasons {
                    thermal: true,
                    ..ThrottleReasons::default()
                },
                ..DeviceSample::empty(device.index, SourceStatus::Ok)
            })
        }
    }

    /// Device 0.. are dedicated R9700s; the last one is unified when `unified_last` is set.
    fn inventory(n: u32, unified_last: bool) -> DeviceInventory {
        DeviceInventory {
            devices: (0..n)
                .map(|i| {
                    let unified = unified_last && i + 1 == n;
                    DeviceInfo {
                        index: DeviceId(i),
                        vendor: Vendor::Amd,
                        vendor_index: i,
                        name: "AMD Radeon AI PRO R9700".into(),
                        uuid: None,
                        pci_bus_id: None,
                        arch: Some("gfx1201".into()),
                        driver_version: None,
                        memory: DeviceMemoryInfo {
                            kind: if unified {
                                MemoryKind::Unified
                            } else {
                                MemoryKind::Dedicated
                            },
                            total_bytes: 32 << 30,
                            shared_with_host: unified,
                        },
                    }
                })
                .collect(),
            backends: Vec::new(),
        }
    }

    #[test]
    fn proc_parsers() {
        assert_eq!(
            parse_meminfo(MEMINFO),
            Ok(MemInfo {
                mem_available_bytes: 44_950_556 * 1024,
                swap_total_bytes: Some(16_777_212 * 1024),
                swap_free_bytes: Some(16_777_212 * 1024),
            }),
            "meminfo kB values are scaled to bytes"
        );
        assert_eq!(
            parse_meminfo("MemTotal:       127535336 kB\n"),
            Err(ParseError::Missing {
                file: "meminfo",
                field: "MemAvailable",
            })
        );
        assert!(matches!(
            parse_meminfo("MemAvailable: lots kB\n"),
            Err(ParseError::Malformed { .. })
        ));
        let a = parse_vmstat(VMSTAT_A).unwrap();
        let b = parse_vmstat(VMSTAT_B).unwrap();
        assert_eq!(
            a,
            VmStat {
                pswpin: 1200,
                pswpout: 3400,
            }
        );
        assert_eq!(b.pswpin - a.pswpin, 250, "pswpin delta between two reads");
        assert_eq!(
            parse_psi(PSI).unwrap(),
            PsiMemory {
                some_avg10: 12.5,
                full_avg10: Some(4.02),
            }
        );

        // A kernel without PSI: only the PSI source is unavailable, the host source stays ok.
        let (no_psi, _) = fake_proc(None);
        let read = read_host(&no_psi);
        assert_eq!(read.psi, SourceStatus::Unavailable);
        assert_eq!(read.vmstat, SourceStatus::Ok);
        assert_eq!(read.sample.status, SourceStatus::Ok);
        assert_eq!(read.sample.psi_memory_some_avg10, None);
        assert_eq!(read.sample.pswpin_total, Some(1200));
        assert_eq!(read.sample.mem_available_bytes, Some(44_950_556 * 1024));

        // No /proc at all (macOS): the host source is unavailable and nothing panics.
        let absent = FsProc {
            root: "/nonexistent-proc-root".into(),
        };
        let read = read_host(&absent);
        assert_eq!(read.sample.status, SourceStatus::Unavailable);
        assert_eq!(read.psi, SourceStatus::Unavailable);
        assert_eq!(read.vmstat, SourceStatus::Unavailable);
        assert_eq!(read.sample.mem_available_bytes, None);
    }

    #[test]
    fn two_cadences() {
        let clock = FakeClock::new(Duration::ZERO);
        let calls = Arc::new(AtomicUsize::new(0));
        let ledger = Arc::new(FakeLedger(Mutex::new(0.10)));
        let (proc, reads) = fake_proc(Some(PSI));
        let vendor: Vec<Box<dyn VendorTelemetry>> = vec![Box::new(Counting {
            calls: Arc::clone(&calls),
            block: Duration::ZERO,
        })];
        let reg = MetricsRegistry::new();
        let mut core = SamplerCore::new(
            TelemetryConfig::default(),
            &inventory(2, true),
            vendor,
            Box::new(proc),
            ledger.clone(),
            Arc::new(clock.clone()),
        )
        .with_metrics(TelemetryMetrics::register(&reg));
        let latest = core.latest();
        let step = Duration::from_millis(10);
        let mut changed_at = None;
        let mut seen_after = None;
        for i in 0..1000 {
            if i == 537 {
                *ledger.0.lock().unwrap() = 0.75;
                changed_at = Some(clock.now_mono());
            }
            core.poll();
            if let Some(t) = changed_at
                && seen_after.is_none()
                && latest.load().ledger.kv_utilization == 0.75
            {
                seen_after = Some(clock.now_mono() - t);
            }
            clock.advance(step);
        }
        let seen_after = seen_after.expect("the ledger change was published");
        assert!(
            seen_after <= Duration::from_millis(100),
            "ledger change visible after {seen_after:?}"
        );
        assert!(
            (99..=101).contains(&core.fast_ticks()),
            "fast ticks {}",
            core.fast_ticks()
        );
        let per_device = calls.load(Ordering::SeqCst) / 2;
        assert!(
            (9..=11).contains(&per_device),
            "vendor calls per device {per_device}"
        );
        assert!((9..=11).contains(&core.vendor_ticks()));
        assert_eq!(
            reads.load(Ordering::SeqCst) as u64,
            3 * core.fast_ticks(),
            "only the fast tick reads /proc"
        );

        let s = latest.load();
        assert_eq!(s.ledger.queue_fill, 0.25);
        assert_eq!(s.host.mem_available_bytes, Some(44_950_556 * 1024));
        assert_eq!(s.host.psi_memory_some_avg10, Some(12.5));
        assert_eq!(s.devices.len(), 2);
        assert!(
            s.devices
                .iter()
                .all(|d| d.status == SourceStatus::Ok && d.temperature_c == Some(55.0))
        );
        assert_eq!(s.devices[0].memory_free_bytes, Some(31 << 30));
        assert_eq!(
            (
                s.devices[1].memory_used_bytes,
                s.devices[1].memory_free_bytes
            ),
            (None, None),
            "unified devices carry no device memory"
        );

        let text = reg.render().unwrap();
        for line in [
            "turbine_gpu_temperature_celsius{device=\"0\"} 55.0",
            "turbine_gpu_clock_mhz{device=\"1\"} 2350",
            "turbine_gpu_power_watts{device=\"0\"} 180.5",
            "turbine_gpu_utilization_ratio{device=\"0\"} 0.5",
            "turbine_gpu_memory_bytes{device=\"0\",kind=\"free\"} 33285996544",
            "turbine_gpu_throttle_active{device=\"0\",reason=\"thermal\"} 1",
            "turbine_gpu_throttle_active{device=\"0\",reason=\"power\"} 0",
            "turbine_host_memory_available_bytes 46029369344",
            "turbine_host_psi_memory_some_avg10 12.5",
            "turbine_host_swap_in_pages_per_second 0.0",
            "turbine_telemetry_stale{source=\"host\"} 0",
            "turbine_telemetry_stale{source=\"1\"} 0",
            "turbine_telemetry_call_duration_seconds_count{source=\"0\"} 10",
        ] {
            assert!(text.contains(line), "missing `{line}` in\n{text}");
        }
        assert!(
            !text.contains("turbine_gpu_memory_bytes{device=\"1\""),
            "no device memory gauge for a unified device"
        );
    }

    #[test]
    fn hung_call_marks_stale() {
        let cfg = TelemetryConfig {
            call_timeout: Duration::from_millis(500),
            ..TelemetryConfig::default()
        };
        let (proc, _) = fake_proc(Some(PSI));
        let vendor: Vec<Box<dyn VendorTelemetry>> = vec![Box::new(Counting {
            calls: Arc::new(AtomicUsize::new(0)),
            block: Duration::from_secs(5),
        })];
        let mut core = SamplerCore::new(
            cfg,
            &inventory(1, false),
            vendor,
            Box::new(proc),
            Arc::new(FakeLedger(Mutex::new(0.0))),
            Arc::new(SystemClock::new()),
        );
        let latest = core.latest();

        let started = Instant::now();
        core.poll(); // first vendor tick: the call hangs past its deadline
        assert!(
            started.elapsed() <= Duration::from_millis(600),
            "first sample returned after {:?}",
            started.elapsed()
        );
        let first = latest.load();
        assert_eq!(first.devices[0].status, SourceStatus::Stale);
        assert_eq!(first.host.status, SourceStatus::Ok);

        // Later samples keep coming on schedule while the vendor call is still hung.
        let ticks = core.fast_ticks();
        let until = Instant::now() + Duration::from_millis(1500);
        while Instant::now() < until {
            let t = Instant::now();
            let wait = core.poll();
            assert!(
                t.elapsed() <= Duration::from_millis(600),
                "a poll blocked for {:?}",
                t.elapsed()
            );
            std::thread::sleep(wait.clamp(Duration::from_millis(1), Duration::from_millis(20)));
        }
        assert!(
            core.fast_ticks() >= ticks + 10,
            "fast ticks continued: {ticks} -> {}",
            core.fast_ticks()
        );
        assert!(core.vendor_ticks() >= 2);
        let last = latest.load();
        assert!(
            last.at_mono_ns > first.at_mono_ns,
            "the host sample keeps updating"
        );
        assert_eq!(last.devices[0].status, SourceStatus::Stale);
        assert!(last.host.mem_available_bytes.is_some());
    }

    #[test]
    fn spawned_sampler_publishes_and_stops() {
        let (proc, _) = fake_proc(Some(PSI));
        let vendor: Vec<Box<dyn VendorTelemetry>> = vec![Box::new(Counting {
            calls: Arc::new(AtomicUsize::new(0)),
            block: Duration::ZERO,
        })];
        let cfg = TelemetryConfig {
            interval: Duration::from_millis(50),
            vendor_interval: Duration::from_millis(50),
            ..TelemetryConfig::default()
        };
        let (sampler, latest) = TelemetrySampler::spawn(
            cfg,
            &inventory(1, false),
            vendor,
            Box::new(proc),
            Arc::new(FakeLedger(Mutex::new(0.4))),
            Arc::new(SystemClock::new()),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while latest.load().devices.first().map(|d| d.status) != Some(SourceStatus::Ok) {
            assert!(Instant::now() < deadline, "no sample published");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(latest.load().ledger.kv_utilization, 0.4);
        let t = Instant::now();
        drop(sampler);
        assert!(t.elapsed() < Duration::from_secs(1), "drop stops and joins");
    }
}
