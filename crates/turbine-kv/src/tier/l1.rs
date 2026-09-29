//! L1 pinned host tier (P4 S-5): page-locked slabs allocated lazily through
//! [`turbine_tensor::PinnedMemory`], released slab by slab when empty under host RED. Disabled on
//! unified-memory devices, where it would copy between two budgets of the same memory.
//!
//! Copy-stream protocol: the GPU copy stream writes a block straight into a slot obtained with
//! [`L1PinnedTier::reserve`]; the slot is invisible to `contains`/`get` until
//! [`L1PinnedTier::commit`], and [`L1PinnedTier::abort_reservation`] frees it after a failed copy.
//! A slot's bytes are never read or written through this tier while a copy on it is outstanding.
//!
//! Slot sizes (P6b S-1): a block is stored in the format of its tier's codec (`kv.cpu.format`)
//! or, for the lossless tail, at the L0 format, so each slab holds slots of one size — the
//! size of the first block that needed it — and a block goes to a slab of its own size.
//! `L1Config.block_bytes` is the L0-format size, the largest a slot can be; the copy-stream
//! protocol reserves slots of that size.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use turbine_core::clock::Clock;
use turbine_core::types::{MemoryKind, PressureState};
use turbine_tensor::{PinnedBuffer, PinnedMemory};

use super::{
    KvTier, TierBlockMut, TierBlockRef, TierError, TierHealth, TierId, TierSlot,
    utilization_pressure,
};
use crate::identity::KvKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L1Config {
    /// `kv.cpu.enabled`; ignored (with one WARN) on unified-memory devices.
    pub enabled: bool,
    /// `kv.cpu.max_bytes`.
    pub max_bytes: u64,
    /// Bytes per pinned slab (1 GiB in production).
    pub slab_bytes: u64,
    /// Bytes of one block at the L0 format: the largest slot and the capacity unit.
    pub block_bytes: u64,
    pub memory_kind: MemoryKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Being written by a copy; not yet visible.
    Reserved,
    Stored,
}

struct Slab {
    buf: PinnedBuffer,
    /// Bytes of each slot of this slab.
    slot_bytes: usize,
    slots: Vec<SlotState>,
    occupied: usize,
}

type Loc = (usize, usize);

struct L1State {
    /// Slab positions stay stable (a slot encodes its slab index); released slabs leave `None`.
    slabs: Vec<Option<Slab>>,
    index: HashMap<KvKey, Loc>,
    pending: HashMap<KvKey, Loc>,
    host_pressure: PressureState,
    health: TierHealth,
    latency: Duration,
    bandwidth: f64,
    /// Slot bytes of the stored and reserved blocks.
    used: u64,
}

pub struct L1PinnedTier {
    cfg: L1Config,
    enabled: bool,
    slots_per_slab: usize,
    max_slabs: usize,
    alloc: Arc<dyn PinnedMemory>,
    clock: Arc<dyn Clock>,
    state: Mutex<L1State>,
}

fn encode(loc: Loc) -> TierSlot {
    TierSlot(((loc.0 as u64) << 32) | loc.1 as u64)
}

impl L1PinnedTier {
    /// Builds the tier without allocating; slabs are allocated by the first puts that need them.
    pub fn new(cfg: L1Config, alloc: Arc<dyn PinnedMemory>, clock: Arc<dyn Clock>) -> Self {
        let unified = cfg.memory_kind == MemoryKind::Unified;
        if cfg.enabled && unified {
            tracing::warn!(
                event = "kv_l1_disabled_unified",
                tier = "l1",
                "L1 pinned tier disabled on a unified-memory device; kv.cpu.* ignored, L0 demotes to L2"
            );
        }
        // A zero block or slab size leaves the tier without slots (disabled) instead of panicking.
        let slots_per_slab = cfg.slab_bytes.checked_div(cfg.block_bytes).unwrap_or(0) as usize;
        let max_slabs = cfg.max_bytes.checked_div(cfg.slab_bytes).unwrap_or(0) as usize;
        L1PinnedTier {
            cfg,
            enabled: cfg.enabled && !unified && slots_per_slab > 0 && max_slabs > 0,
            slots_per_slab,
            max_slabs,
            alloc,
            clock,
            state: Mutex::new(L1State {
                slabs: Vec::new(),
                index: HashMap::new(),
                pending: HashMap::new(),
                host_pressure: PressureState::Green,
                health: TierHealth::new(None),
                latency: Duration::from_micros(20),
                bandwidth: 8e9,
                used: 0,
            }),
        }
    }

    /// Live (allocated) slabs.
    pub fn slab_count(&self) -> usize {
        self.lock().slabs.iter().filter(|s| s.is_some()).count()
    }

    /// Phase 3 host-memory pressure. At RED and above no slab is allocated and every empty slab
    /// is released (and further slabs as they empty).
    pub fn set_host_pressure(&self, p: PressureState) {
        let mut s = self.lock();
        s.host_pressure = p;
        if p >= PressureState::Red {
            let mut released = 0usize;
            for slab in s.slabs.iter_mut() {
                if slab.as_ref().is_some_and(|x| x.occupied == 0) {
                    *slab = None;
                    released += 1;
                }
            }
            if released > 0 {
                tracing::info!(
                    event = "kv_l1_slab_released",
                    tier = "l1",
                    released,
                    host_pressure = p.as_str(),
                    "empty L1 slabs released under host memory pressure"
                );
            }
        }
    }

    /// Calibrated L1 → L0 copy estimates.
    pub fn set_estimates(&self, latency: Duration, bandwidth: f64) {
        let mut s = self.lock();
        s.latency = latency;
        s.bandwidth = bandwidth;
    }

    /// Pinned buffer id and byte offset of a stored block (the copy-stream end of L0 ↔ L1).
    pub fn locate(&self, key: &KvKey) -> Option<(u64, usize)> {
        let s = self.lock();
        let loc = *s.index.get(key)?;
        self.address(&s, loc)
    }

    /// Takes a free slot for `key`, invisible until [`commit`](Self::commit); returns the buffer id
    /// and offset the copy stream writes to.
    pub fn reserve(&self, key: KvKey) -> Result<(u64, usize), TierError> {
        if !self.enabled {
            return Err(TierError::Full);
        }
        let mut s = self.lock();
        if s.health.is_degraded() {
            return Err(TierError::Degraded);
        }
        if let Some(&loc) = s.pending.get(&key) {
            return self.address(&s, loc).ok_or(TierError::Missing);
        }
        let loc = self.take_slot(&mut s, SlotState::Reserved, self.cfg.block_bytes as usize)?;
        s.pending.insert(key, loc);
        self.address(&s, loc).ok_or(TierError::Missing)
    }

    /// Makes a reserved block visible, replacing any stored copy of `key`. Idempotent for a key
    /// already stored. Panics when `key` was neither reserved nor stored (a caller bug).
    pub fn commit(&self, key: &KvKey) -> TierSlot {
        let mut s = self.lock();
        let Some(loc) = s.pending.remove(key) else {
            let loc = *s.index.get(key).expect("L1 commit without a reservation");
            return encode(loc);
        };
        Self::slot_mut(&mut s, loc, SlotState::Stored);
        if let Some(old) = s.index.insert(*key, loc) {
            Self::release_slot(&mut s, old);
        }
        encode(loc)
    }

    /// Frees the reservation of `key` after its copy failed; false when there is none.
    pub fn abort_reservation(&self, key: &KvKey) -> bool {
        let mut s = self.lock();
        match s.pending.remove(key) {
            Some(loc) => {
                Self::release_slot(&mut s, loc);
                true
            }
            None => false,
        }
    }

    /// Records one copy-stream error; returns true when it degrades the tier (3 within 60 s).
    pub fn record_copy_error(&self) -> bool {
        let now = self.clock.now_mono();
        let degraded = self.lock().health.record_error(now);
        if degraded {
            tracing::warn!(
                event = "kv_tier_degraded",
                tier = "l1",
                "3 L1 copy errors within 60 s"
            );
        }
        degraded
    }

    fn address(&self, s: &L1State, (slab, slot): Loc) -> Option<(u64, usize)> {
        let slab = s.slabs.get(slab)?.as_ref()?;
        Some((slab.buf.id(), slot * slab.slot_bytes))
    }

    fn slot_mut(s: &mut L1State, (slab, slot): Loc, state: SlotState) {
        let slab = s.slabs[slab].as_mut().expect("an indexed slab is live");
        slab.slots[slot] = state;
    }

    /// Frees one slot; the emptied slab is released at once under host RED.
    fn release_slot(s: &mut L1State, (slab_idx, slot): Loc) {
        let release = s.host_pressure >= PressureState::Red;
        let slab = s.slabs[slab_idx].as_mut().expect("an indexed slab is live");
        slab.slots[slot] = SlotState::Free;
        slab.occupied -= 1;
        let freed = slab.slot_bytes as u64;
        let empty = slab.occupied == 0;
        s.used -= freed;
        if release && empty {
            s.slabs[slab_idx] = None;
        }
    }

    /// A free `slot_bytes` slot in a live slab of that slot size, else a slot in a newly
    /// allocated slab when the size limit and host pressure allow one.
    fn take_slot(
        &self,
        s: &mut L1State,
        state: SlotState,
        slot_bytes: usize,
    ) -> Result<Loc, TierError> {
        let free = s.slabs.iter().enumerate().find_map(|(i, slab)| {
            let slab = slab.as_ref()?;
            if slab.slot_bytes != slot_bytes {
                return None;
            }
            let j = slab.slots.iter().position(|x| *x == SlotState::Free)?;
            Some((i, j))
        });
        let loc = match free {
            Some(loc) => loc,
            None => {
                let live = s.slabs.iter().filter(|x| x.is_some()).count();
                if live >= self.max_slabs || s.host_pressure >= PressureState::Red {
                    return Err(TierError::Full);
                }
                let buf = match self.alloc.alloc_pinned(self.cfg.slab_bytes as usize) {
                    Ok(buf) => buf,
                    Err(e) => {
                        tracing::warn!(
                            event = "kv_l1_slab_alloc_failed",
                            tier = "l1",
                            slabs = live,
                            slab_bytes = self.cfg.slab_bytes,
                            error = %e,
                            "pinned slab allocation failed; L1 stays at its current size"
                        );
                        return Err(TierError::Full);
                    }
                };
                let slab = Slab {
                    buf,
                    slot_bytes,
                    slots: vec![SlotState::Free; self.cfg.slab_bytes as usize / slot_bytes],
                    occupied: 0,
                };
                let idx = match s.slabs.iter().position(Option::is_none) {
                    Some(i) => {
                        s.slabs[i] = Some(slab);
                        i
                    }
                    None => {
                        s.slabs.push(Some(slab));
                        s.slabs.len() - 1
                    }
                };
                (idx, 0)
            }
        };
        let slab = s.slabs[loc.0].as_mut().expect("the chosen slab is live");
        slab.slots[loc.1] = state;
        slab.occupied += 1;
        s.used += slot_bytes as u64;
        Ok(loc)
    }

    // Every update leaves the state consistent, so a lock poisoned by a panicking thread is
    // safe to keep using.
    fn lock(&self) -> MutexGuard<'_, L1State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A block fits a slot when it is not empty and no larger than an L0-format block.
    fn check_len(&self, len: usize) -> Result<(), TierError> {
        if len > 0 && len as u64 <= self.cfg.block_bytes {
            Ok(())
        } else {
            Err(TierError::Io(format!(
                "block is {len} bytes, L1 slots hold 1 to {} bytes",
                self.cfg.block_bytes
            )))
        }
    }
}

impl KvTier for L1PinnedTier {
    fn id(&self) -> TierId {
        TierId::L1
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    /// Usable bytes: whole slots in whole slabs up to `max_bytes` (0 when disabled).
    fn capacity_bytes(&self) -> u64 {
        if self.enabled {
            (self.max_slabs * self.slots_per_slab) as u64 * self.cfg.block_bytes
        } else {
            0
        }
    }

    /// Slot bytes of the stored and reserved blocks.
    fn used_bytes(&self) -> u64 {
        self.lock().used
    }

    /// The worse of utilisation pressure and Phase 3 host-memory pressure.
    fn pressure(&self) -> PressureState {
        let host = self.lock().host_pressure;
        utilization_pressure(self.used_bytes(), self.capacity_bytes()).max(host)
    }

    fn est_latency(&self) -> Duration {
        self.lock().latency
    }

    fn est_bandwidth(&self) -> Option<f64> {
        Some(self.lock().bandwidth)
    }

    fn contains(&self, key: &KvKey) -> bool {
        self.lock().index.contains_key(key)
    }

    fn put(&self, key: KvKey, src: TierBlockRef<'_>) -> Result<TierSlot, TierError> {
        let TierBlockRef::Host(bytes) = src;
        if !self.enabled {
            return Err(TierError::Full);
        }
        self.check_len(bytes.len())?;
        let mut s = self.lock();
        if s.health.is_degraded() {
            return Err(TierError::Degraded);
        }
        let same_size = |s: &L1State, loc: Loc| {
            s.slabs[loc.0]
                .as_ref()
                .is_some_and(|x| x.slot_bytes == bytes.len())
        };
        let loc = match s.index.get(&key).copied() {
            Some(loc) if same_size(&s, loc) => loc,
            old => {
                // A replacement in another format moves to a slot of its own size.
                let loc = self.take_slot(&mut s, SlotState::Stored, bytes.len())?;
                if let Some(old) = old {
                    Self::release_slot(&mut s, old);
                }
                loc
            }
        };
        let off = loc.1 * bytes.len();
        let slab = s.slabs[loc.0].as_ref().expect("the chosen slab is live");
        slab.buf
            .with_bytes_mut(|b| b[off..off + bytes.len()].copy_from_slice(bytes));
        s.index.insert(key, loc);
        Ok(encode(loc))
    }

    fn get(&self, key: &KvKey, dst: TierBlockMut<'_>) -> Result<(), TierError> {
        let TierBlockMut::Host(out) = dst;
        let s = self.lock();
        if s.health.is_degraded() {
            return Err(TierError::Degraded);
        }
        let &(slab, slot) = s.index.get(key).ok_or(TierError::Missing)?;
        let slab = s.slabs[slab].as_ref().expect("an indexed slab is live");
        if out.len() != slab.slot_bytes {
            return Err(TierError::Io(format!(
                "block is {} bytes, buffer {}",
                slab.slot_bytes,
                out.len()
            )));
        }
        let off = slot * out.len();
        slab.buf
            .with_bytes(|b| out.copy_from_slice(&b[off..off + out.len()]));
        Ok(())
    }

    fn evict(&self, key: &KvKey) -> Result<(), TierError> {
        let mut s = self.lock();
        let loc = s.index.remove(key).ok_or(TierError::Missing)?;
        Self::release_slot(&mut s, loc);
        Ok(())
    }

    fn degraded(&self) -> bool {
        self.lock().health.is_degraded()
    }
}
