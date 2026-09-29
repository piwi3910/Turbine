//! L2 NVMe tier (P4 S-11): fixed-size slab files of 4 KiB-aligned block slots, a CRC32C per
//! block verified on every read, a bounded I/O queue, and `O_DIRECT` where the filesystem allows
//! it. Nothing persists across restarts: opening the tier deletes every `turbine-kv-*.slab`.
//!
//! Slab file layout: a 4 KiB header (magic `TKVSLAB1`, format version u32 LE, namespace key,
//! slot size u64 LE, slot count u64 LE; version 2 adds the slots' `kv_format` codec name,
//! 16 bytes NUL-padded, and its rotation seed u64 LE — P6b S-1) followed by the slots, each one
//! block padded to 4 KiB.
//!
//! Formats (P6b S-1): each slab file holds blocks of one codec — `kv.nvme.format`, or `l0` for
//! the lossless tail — in slots sized by the first block of that codec, never larger than the
//! `l0` slab file (so `kv.nvme.max_bytes` still bounds the files).

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use turbine_core::clock::Clock;
use turbine_core::types::PressureState;

use super::{KvTier, TierBlockMut, TierBlockRef, TierError, TierHealth, TierId, TierSlot};
use crate::identity::{KvKey, NamespaceKey};
use crate::metrics::{EvictReason, KvMetrics};

pub const SLAB_MAGIC: &[u8; 8] = b"TKVSLAB1";
const SLAB_VERSION: u32 = 2;
/// Bytes of the codec name in a version 2 header.
const FORMAT_NAME_BYTES: usize = 16;
/// Slot and header alignment (the `O_DIRECT` requirement on the lab NVMe drives).
const ALIGN: usize = 4096;
const HEADER_BYTES: u64 = 4096;
/// A degraded L2 is probed again after this long (P4 Failure modes).
const PROBE_AFTER: Duration = Duration::from_secs(300);
/// I/O latencies kept for the p99.
const LATENCY_WINDOW: usize = 1024;
/// Bandwidth estimate before calibration: the conservative L1↔L2 fallback (P4 Failure modes).
const DEFAULT_BANDWIDTH: f64 = 1e9;
/// Latency estimate before calibration.
const DEFAULT_LATENCY: Duration = Duration::from_micros(100);
/// Key of the block written and read back by a health probe.
const PROBE_KEY: KvKey = KvKey([0xff; 16]);
/// `EINVAL`: what Linux returns when a filesystem refuses `O_DIRECT` on open.
const EINVAL: i32 = 22;

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "x86")))]
const O_DIRECT: Option<i32> = Some(0o040000);
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "arm")))]
const O_DIRECT: Option<i32> = Some(0o200000);
#[cfg(not(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "arm"
    )
)))]
const O_DIRECT: Option<i32> = None;

#[derive(Clone, Debug)]
pub struct L2Config {
    /// `kv.nvme.path`; created if absent.
    pub path: PathBuf,
    /// `kv.nvme.max_bytes`: bound on the total size of the slab files.
    pub max_bytes: u64,
    /// `kv.nvme.slab_bytes`: slot bytes per slab file (rounded down to whole slots).
    pub slab_bytes: u64,
    /// `kv.nvme.max_queue_depth`: concurrent I/O operations.
    pub max_queue_depth: u32,
    pub block_bytes: u64,
    /// Salt-free namespace key written into each slab header.
    pub namespace: NamespaceKey,
}

/// One slab file and the codec its slots hold.
struct Slab {
    file: File,
    format: &'static str,
    slot_bytes: usize,
    /// Slots in use (stored, or being written); an unused slab may take another codec.
    taken: usize,
}

#[derive(Clone, Copy)]
struct SlotEntry {
    slab: usize,
    slot: usize,
    crc: u32,
    len: usize,
}

struct L2State {
    slabs: Vec<Slab>,
    index: HashMap<KvKey, SlotEntry>,
    /// Free slots per codec name.
    free: HashMap<&'static str, Vec<(usize, usize)>>,
    /// Bytes of the stored blocks.
    used: u64,
    health: TierHealth,
    latencies: VecDeque<f64>,
    calibration_p99: Option<f64>,
    bandwidth: f64,
    direct: bool,
}

pub struct L2NvmeTier {
    cfg: L2Config,
    /// Slot bytes of an `l0` block (the largest slot).
    slot_bytes: usize,
    /// Slots of an `l0` slab file.
    slots_per_slab: usize,
    max_slabs: usize,
    clock: Arc<dyn Clock>,
    metrics: KvMetrics,
    state: Mutex<L2State>,
    /// In-flight I/O count, bounded by `max_queue_depth`.
    queue: (Mutex<u32>, Condvar),
}

/// A zeroed heap buffer whose window starts on a 4 KiB boundary (`O_DIRECT`), without `unsafe`.
struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedBuf {
    fn zeroed(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGN];
        let addr = raw.as_ptr() as usize;
        let off = (ALIGN - addr % ALIGN) % ALIGN;
        AlignedBuf { raw, off, len }
    }

    fn bytes(&self) -> &[u8] {
        &self.raw[self.off..self.off + self.len]
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

fn is_slab_file(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("turbine-kv-") && n.ends_with(".slab"))
}

fn open_slab(path: &Path, direct: bool) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true).write(true).create(true).truncate(true);
    if let (true, Some(flag)) = (direct, O_DIRECT) {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(flag);
    }
    o.open(path)
}

fn io_err(path: &Path, e: impl std::fmt::Display) -> TierError {
    TierError::Io(format!("{}: {e}", path.display()))
}

impl L2NvmeTier {
    /// Creates `cfg.path` if absent, deletes only `turbine-kv-*.slab` files in it, and checks
    /// that it is writable. Slab files are created lazily as blocks arrive.
    pub fn open(
        cfg: L2Config,
        clock: Arc<dyn Clock>,
        metrics: KvMetrics,
    ) -> Result<Self, TierError> {
        std::fs::create_dir_all(&cfg.path).map_err(|e| io_err(&cfg.path, e))?;
        for entry in std::fs::read_dir(&cfg.path).map_err(|e| io_err(&cfg.path, e))? {
            let p = entry.map_err(|e| io_err(&cfg.path, e))?.path();
            if is_slab_file(&p) {
                std::fs::remove_file(&p).map_err(|e| io_err(&p, e))?;
            }
        }
        let probe = cfg.path.join(".turbine-kv-write-probe");
        std::fs::write(&probe, b"ok")
            .map_err(|e| io_err(&cfg.path, format!("not writable: {e}")))?;
        std::fs::remove_file(&probe).map_err(|e| io_err(&probe, e))?;

        let slot_bytes = usize::try_from(cfg.block_bytes)
            .map_err(|_| TierError::Io(format!("block of {} bytes", cfg.block_bytes)))?
            .div_ceil(ALIGN)
            * ALIGN;
        let slots_per_slab = usize::try_from(cfg.slab_bytes).unwrap_or(usize::MAX) / slot_bytes;
        if slots_per_slab == 0 {
            return Err(TierError::Io(format!(
                "kv.nvme.slab_bytes {} holds no {slot_bytes}-byte slot",
                cfg.slab_bytes
            )));
        }
        let file_bytes = HEADER_BYTES + (slots_per_slab * slot_bytes) as u64;
        let max_slabs = usize::try_from(cfg.max_bytes / file_bytes).unwrap_or(usize::MAX);
        if max_slabs == 0 {
            return Err(TierError::Io(format!(
                "kv.nvme.max_bytes {} holds no {file_bytes}-byte slab file",
                cfg.max_bytes
            )));
        }
        Ok(L2NvmeTier {
            slot_bytes,
            slots_per_slab,
            max_slabs,
            clock,
            metrics,
            state: Mutex::new(L2State {
                slabs: Vec::new(),
                index: HashMap::new(),
                free: HashMap::new(),
                used: 0,
                health: TierHealth::new(Some(PROBE_AFTER)),
                latencies: VecDeque::with_capacity(LATENCY_WINDOW),
                calibration_p99: None,
                bandwidth: DEFAULT_BANDWIDTH,
                direct: O_DIRECT.is_some(),
            }),
            queue: (Mutex::new(0), Condvar::new()),
            cfg,
        })
    }

    pub fn slab_path(&self, slab: usize) -> PathBuf {
        self.cfg.path.join(format!("turbine-kv-{slab:04}.slab"))
    }

    /// Size of one `l0` slab file (the largest): header plus its slots.
    pub fn slab_file_bytes(&self) -> u64 {
        HEADER_BYTES + (self.slots_per_slab * self.slot_bytes) as u64
    }

    /// I/O operations in flight (`turbine_storage_queue_depth`).
    pub fn queue_depth(&self) -> u32 {
        *lock(&self.queue.0)
    }

    /// False once the filesystem refused `O_DIRECT` (and always off Linux).
    pub fn uses_direct_io(&self) -> bool {
        self.lock().direct
    }

    /// Startup calibration: p99 latency and bandwidth of this device.
    pub fn set_calibration(&self, p99_seconds: f64, bandwidth: f64) {
        let mut s = self.lock();
        s.calibration_p99 = Some(p99_seconds);
        s.bandwidth = bandwidth;
    }

    /// p99 of the last 1,024 I/O latencies in seconds (0 before any I/O).
    pub fn p99_latency(&self) -> f64 {
        let mut v: Vec<f64> = self.lock().latencies.iter().copied().collect();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(f64::total_cmp);
        v[(v.len() * 99 / 100).min(v.len() - 1)]
    }

    pub fn calibration_p99(&self) -> Option<f64> {
        self.lock().calibration_p99
    }

    /// Slow-storage rule: a p99 above 10× the calibration p99 degrades the tier. Returns true
    /// when this call degraded it.
    pub fn check_slow(&self) -> bool {
        let p99 = self.p99_latency();
        let now = self.clock.now_mono();
        let mut s = self.lock();
        let calibration = s.calibration_p99;
        match calibration {
            Some(c) if c > 0.0 && p99 > 10.0 * c && s.health.mark_degraded(now) => {
                self.metrics.set_degraded(TierId::L2, true);
                tracing::warn!(
                    event = "kv_tier_degraded",
                    tier = "l2",
                    p99,
                    calibration_p99 = c,
                    "storage slower than 10x its calibration"
                );
                true
            }
            _ => false,
        }
    }

    /// Once a degraded period has lasted 5 minutes, one probe write/read decides whether the
    /// tier recovers. Returns true when the tier is healthy again.
    pub fn probe(&self) -> bool {
        let now = self.clock.now_mono();
        if !self.lock().health.probe_due(now) {
            return false;
        }
        // Let the probe's own I/O through; the outcome below decides the state.
        self.lock().health.probe_result(true, now);
        let data = vec![0x5a_u8; self.cfg.block_bytes as usize];
        let ok = self.put(PROBE_KEY, TierBlockRef::Host(&data)).is_ok() && {
            let mut back = vec![0u8; data.len()];
            self.get(&PROBE_KEY, TierBlockMut::Host(&mut back)).is_ok() && back == data
        };
        // The probe block may be absent after a failed put; either way it must not stay.
        let _ = self.evict(&PROBE_KEY);
        self.lock().health.probe_result(ok, now);
        self.metrics.set_degraded(TierId::L2, !ok);
        ok
    }

    // Updates keep the state consistent at every unlock, so a poisoned lock is safe to reuse.
    fn lock(&self) -> MutexGuard<'_, L2State> {
        lock(&self.state)
    }

    /// Waits for a queue slot (bounded queue depth).
    fn enter(&self) {
        let (m, cv) = &self.queue;
        let mut depth = lock(m);
        while *depth >= self.cfg.max_queue_depth {
            depth = cv.wait(depth).unwrap_or_else(|e| e.into_inner());
        }
        *depth += 1;
        self.metrics.storage_queue_depth.set(i64::from(*depth));
    }

    /// Releases the queue slot and records the operation's latency (the only wall-clock use).
    fn leave(&self, started: Instant) {
        let secs = started.elapsed().as_secs_f64();
        self.metrics.storage_latency.observe(secs);
        {
            let mut s = self.lock();
            if s.latencies.len() == LATENCY_WINDOW {
                s.latencies.pop_front();
            }
            s.latencies.push_back(secs);
        }
        let (m, cv) = &self.queue;
        let mut depth = lock(m);
        *depth -= 1;
        self.metrics.storage_queue_depth.set(i64::from(*depth));
        cv.notify_one();
    }

    /// Runs one positioned I/O inside the queue gate.
    fn io<R>(&self, f: impl FnOnce() -> std::io::Result<R>) -> std::io::Result<R> {
        self.enter();
        let started = Instant::now();
        let r = f();
        self.leave(started);
        r
    }

    /// Formats slab file `idx` for blocks of codec `format`, `slot_bytes` per slot: a new file
    /// when `idx` is `s.slabs.len()`, else an unused slab of another codec, rewritten in place
    /// (so blocks of one codec never starve for slabs the other codecs hold empty).
    fn format_slab(
        &self,
        s: &mut L2State,
        idx: usize,
        format: &'static str,
        slot_bytes: usize,
    ) -> Result<(), TierError> {
        let slots = (self.slots_per_slab * self.slot_bytes) / slot_bytes;
        if format.len() > FORMAT_NAME_BYTES || slots == 0 {
            return Err(TierError::Io(format!(
                "no {slot_bytes}-byte `{format}` slot fits an L2 slab"
            )));
        }
        let path = self.slab_path(idx);
        let reused = match s.slabs.get(idx) {
            Some(old) => Some(
                old.file
                    .try_clone()
                    .map_err(|e| TierError::Io(e.to_string()))?,
            ),
            None => None,
        };
        let file = match reused {
            Some(f) => f,
            None => self.open_new_slab(s, &path)?,
        };
        file.set_len(HEADER_BYTES + (slots * slot_bytes) as u64)
            .map_err(|e| io_err(&path, e))?;
        let mut header = AlignedBuf::zeroed(HEADER_BYTES as usize);
        let h = header.bytes_mut();
        h[0..8].copy_from_slice(SLAB_MAGIC);
        h[8..12].copy_from_slice(&SLAB_VERSION.to_le_bytes());
        h[12..44].copy_from_slice(&self.cfg.namespace.0);
        h[44..52].copy_from_slice(&(slot_bytes as u64).to_le_bytes());
        h[52..60].copy_from_slice(&(slots as u64).to_le_bytes());
        let name = 60 + FORMAT_NAME_BYTES;
        h[60..60 + format.len()].copy_from_slice(format.as_bytes());
        h[name..name + 8].copy_from_slice(&self.codec_seed().to_le_bytes());
        file.write_all_at(header.bytes(), 0)
            .map_err(|e| io_err(&path, e))?;
        let slab = Slab {
            file,
            format,
            slot_bytes,
            taken: 0,
        };
        if idx < s.slabs.len() {
            let old = std::mem::replace(&mut s.slabs[idx], slab);
            if let Some(free) = s.free.get_mut(old.format) {
                free.retain(|&(i, _)| i != idx);
            }
        } else {
            s.slabs.push(slab);
        }
        // Popped from the end: slot 0 first.
        s.free
            .entry(format)
            .or_default()
            .extend((0..slots).rev().map(|slot| (idx, slot)));
        Ok(())
    }

    /// Opens a new slab file, falling back to buffered I/O where `O_DIRECT` is refused.
    fn open_new_slab(&self, s: &mut L2State, path: &Path) -> Result<File, TierError> {
        let path = path.to_path_buf();
        let file = match open_slab(&path, s.direct) {
            Ok(f) => f,
            Err(e) if s.direct && e.raw_os_error() == Some(EINVAL) => {
                tracing::warn!(
                    event = "kv_nvme_buffered_io",
                    path = %path.display(),
                    "the filesystem refuses O_DIRECT; using buffered I/O"
                );
                s.direct = false;
                open_slab(&path, false).map_err(|e| io_err(&path, e))?
            }
            Err(e) => return Err(io_err(&path, e)),
        };
        Ok(file)
    }

    /// The codecs' rotation seed (`CodecParams::seed`): the first 8 bytes of the namespace key,
    /// little-endian.
    fn codec_seed(&self) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.cfg.namespace.0[..8]);
        u64::from_le_bytes(b)
    }

    /// Stores `bytes` under `key` in a slot of codec `format`.
    fn store(&self, key: KvKey, format: &'static str, bytes: &[u8]) -> Result<TierSlot, TierError> {
        if bytes.is_empty() || bytes.len() > self.slot_bytes {
            return Err(TierError::Io(format!(
                "block of {} bytes does not fit the {}-byte slot",
                bytes.len(),
                self.slot_bytes
            )));
        }
        let (slab, slot, slot_bytes, file) = {
            let mut s = self.lock();
            if s.health.is_degraded() {
                return Err(TierError::Degraded);
            }
            // A replaced block of the same codec is rewritten in its own slot; it is not
            // readable meanwhile. Another codec's block moves to a slot of its own codec.
            let old = s.index.remove(&key);
            if let Some(e) = old {
                s.used -= e.len as u64;
            }
            let fits = |s: &L2State, slab: usize| {
                s.slabs[slab].format == format && bytes.len() <= s.slabs[slab].slot_bytes
            };
            let reuse = old.filter(|e| fits(&s, e.slab));
            if let Some(e) = old.filter(|e| !fits(&s, e.slab)) {
                self.free_slot(&mut s, e.slab, e.slot);
            }
            let (slab, slot) = match reuse {
                Some(e) => (e.slab, e.slot),
                None => match self.pop_free(&mut s, format, bytes.len()) {
                    Some(free) => free,
                    None => {
                        let slot_bytes = if format == crate::tier::L0_FORMAT {
                            self.slot_bytes
                        } else {
                            bytes.len().div_ceil(ALIGN) * ALIGN
                        };
                        let idx = if s.slabs.len() < self.max_slabs {
                            s.slabs.len()
                        } else {
                            s.slabs
                                .iter()
                                .position(|x| x.taken == 0)
                                .ok_or(TierError::Full)?
                        };
                        self.format_slab(&mut s, idx, format, slot_bytes)?;
                        self.pop_free(&mut s, format, bytes.len())
                            .ok_or(TierError::Full)?
                    }
                },
            };
            let file = s.slabs[slab]
                .file
                .try_clone()
                .map_err(|e| TierError::Io(e.to_string()))?;
            (slab, slot, s.slabs[slab].slot_bytes, file)
        };
        let mut buf = AlignedBuf::zeroed(slot_bytes);
        buf.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        let res = self.io(|| file.write_all_at(buf.bytes(), Self::offset(slot_bytes, slot)));
        let mut s = self.lock();
        match res {
            Ok(()) => {
                let entry = SlotEntry {
                    slab,
                    slot,
                    crc: crc32c::crc32c(bytes),
                    len: bytes.len(),
                };
                s.used += bytes.len() as u64;
                s.index.insert(key, entry);
                Ok(TierSlot(((slab as u64) << 32) | slot as u64))
            }
            Err(e) => {
                self.free_slot(&mut s, slab, slot);
                Err(self.io_error(&mut s, e))
            }
        }
    }

    /// A free slot of codec `format` holding `len` bytes.
    fn pop_free(
        &self,
        s: &mut L2State,
        format: &'static str,
        len: usize,
    ) -> Option<(usize, usize)> {
        let slabs = &s.slabs;
        let free = s.free.get_mut(format)?;
        let i = free
            .iter()
            .rposition(|&(slab, _)| len <= slabs[slab].slot_bytes)?;
        let (slab, slot) = free.remove(i);
        s.slabs[slab].taken += 1;
        Some((slab, slot))
    }

    fn free_slot(&self, s: &mut L2State, slab: usize, slot: usize) {
        let x = &mut s.slabs[slab];
        x.taken -= 1;
        let format = x.format;
        s.free.entry(format).or_default().push((slab, slot));
    }

    /// Counts one I/O error toward degradation and converts it.
    fn io_error(&self, s: &mut L2State, e: impl std::fmt::Display) -> TierError {
        let now = self.clock.now_mono();
        if s.health.record_error(now) {
            self.metrics.set_degraded(TierId::L2, true);
            tracing::warn!(
                event = "kv_tier_degraded",
                tier = "l2",
                error = %e,
                "3 L2 errors within 60 s"
            );
        }
        TierError::Io(e.to_string())
    }

    fn offset(slot_bytes: usize, slot: usize) -> u64 {
        HEADER_BYTES + (slot * slot_bytes) as u64
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl KvTier for L2NvmeTier {
    fn id(&self) -> TierId {
        TierId::L2
    }

    fn enabled(&self) -> bool {
        true
    }

    fn capacity_bytes(&self) -> u64 {
        (self.max_slabs * self.slots_per_slab) as u64 * self.cfg.block_bytes
    }

    /// Bytes of the stored blocks (each at its codec's size).
    fn used_bytes(&self) -> u64 {
        self.lock().used
    }

    /// Storage queue fill on the contract thresholds (0.50 / 0.75 / 0.90).
    fn pressure(&self) -> PressureState {
        match f64::from(self.queue_depth()) / f64::from(self.cfg.max_queue_depth.max(1)) {
            f if f >= 0.90 => PressureState::Red,
            f if f >= 0.75 => PressureState::Orange,
            f if f >= 0.50 => PressureState::Yellow,
            _ => PressureState::Green,
        }
    }

    fn est_latency(&self) -> Duration {
        self.lock()
            .calibration_p99
            .map_or(DEFAULT_LATENCY, Duration::from_secs_f64)
    }

    fn est_bandwidth(&self) -> Option<f64> {
        Some(self.lock().bandwidth)
    }

    fn contains(&self, key: &KvKey) -> bool {
        self.lock().index.contains_key(key)
    }

    /// Stores a block at the L0 format (`l0`).
    fn put(&self, key: KvKey, src: TierBlockRef<'_>) -> Result<TierSlot, TierError> {
        let TierBlockRef::Host(bytes) = src;
        self.store(key, crate::tier::L0_FORMAT, bytes)
    }

    /// Stores a block in a slot of codec `format` (its slab header names the codec).
    fn put_as(
        &self,
        key: KvKey,
        format: &'static str,
        _bytes: u64,
        src: TierBlockRef<'_>,
    ) -> Result<TierSlot, TierError> {
        let TierBlockRef::Host(bytes) = src;
        self.store(key, format, bytes)
    }

    fn get(&self, key: &KvKey, dst: TierBlockMut<'_>) -> Result<(), TierError> {
        let TierBlockMut::Host(out) = dst;
        let (entry, file) = {
            let s = self.lock();
            if s.health.is_degraded() {
                return Err(TierError::Degraded);
            }
            let e = *s.index.get(key).ok_or(TierError::Missing)?;
            let slab = &s.slabs[e.slab];
            let file = slab
                .file
                .try_clone()
                .map_err(|e| TierError::Io(e.to_string()))?;
            (e, (file, slab.slot_bytes))
        };
        let (file, slot_bytes) = file;
        if out.len() != entry.len {
            return Err(TierError::Io(format!(
                "block is {} bytes, buffer {}",
                entry.len,
                out.len()
            )));
        }
        let mut buf = AlignedBuf::zeroed(slot_bytes);
        let res =
            self.io(|| file.read_exact_at(buf.bytes_mut(), Self::offset(slot_bytes, entry.slot)));
        let mut s = self.lock();
        if let Err(e) = res {
            return Err(self.io_error(&mut s, e));
        }
        let data = &buf.bytes()[..entry.len];
        if crc32c::crc32c(data) != entry.crc {
            // Only drop the entry if it still names this slot (no concurrent replacement).
            if s.index
                .get(key)
                .is_some_and(|e| (e.slab, e.slot) == (entry.slab, entry.slot))
            {
                s.index.remove(key);
                s.used -= entry.len as u64;
                self.free_slot(&mut s, entry.slab, entry.slot);
            }
            self.metrics.eviction(TierId::L2, EvictReason::Checksum);
            tracing::warn!(
                event = "kv_checksum_mismatch",
                tier = "l2",
                key = %key,
                "L2 block failed its checksum; dropped, the request recomputes"
            );
            let _ = self.io_error(&mut s, "checksum mismatch");
            return Err(TierError::Checksum);
        }
        out.copy_from_slice(data);
        Ok(())
    }

    fn evict(&self, key: &KvKey) -> Result<(), TierError> {
        let mut s = self.lock();
        let e = s.index.remove(key).ok_or(TierError::Missing)?;
        s.used -= e.len as u64;
        self.free_slot(&mut s, e.slab, e.slot);
        Ok(())
    }

    fn degraded(&self) -> bool {
        self.lock().health.is_degraded()
    }
}
