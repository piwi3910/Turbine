//! In-memory tier: the test double for L1/L2 with injectable latency, bandwidth and faults, and
//! (payload-free) the offline simulator's tier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use turbine_core::clock::Clock;
use turbine_core::types::PressureState;

use super::{
    KvTier, TierBlockMut, TierBlockRef, TierError, TierHealth, TierId, TierSlot,
    utilization_pressure,
};
use crate::identity::KvKey;

pub struct MemTier {
    id: TierId,
    capacity_bytes: u64,
    /// `Some(block_bytes)`: payload-free mode (a block accounts `block_bytes`, or the logical
    /// bytes [`KvTier::put_as`] names; none are stored).
    nominal_block: Option<u64>,
    clock: Arc<dyn Clock>,
    state: Mutex<MemState>,
}

/// One stored block: its bytes (none when payload-free) and the bytes it accounts.
struct Stored {
    bytes: Vec<u8>,
    size: u64,
}

struct MemState {
    blocks: HashMap<KvKey, Stored>,
    used: u64,
    latency: Duration,
    bandwidth: f64,
    fail_reads: u32,
    fail_writes: u32,
    /// Stores still to refuse with `Full` ([`MemTier::inject_full`]).
    fail_full: u32,
    /// [`KvTier::room_epoch`]: bumped whenever stored bytes are released.
    room_epoch: u64,
    health: TierHealth,
    next_slot: u64,
}

impl MemTier {
    /// A tier storing block bytes, with pinned-copy-like default estimates (20 µs, 8 GB/s).
    pub fn new(id: TierId, capacity_bytes: u64, clock: Arc<dyn Clock>) -> Self {
        Self::build(id, capacity_bytes, None, clock)
    }

    /// A tier that accounts `block_bytes` per block (or the logical bytes a
    /// [`KvTier::put_as`] names) and stores nothing; `get` fills zeros.
    pub fn payload_free(
        id: TierId,
        capacity_bytes: u64,
        block_bytes: u64,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self::build(id, capacity_bytes, Some(block_bytes), clock)
    }

    fn build(
        id: TierId,
        capacity_bytes: u64,
        nominal_block: Option<u64>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        MemTier {
            id,
            capacity_bytes,
            nominal_block,
            clock,
            state: Mutex::new(MemState {
                blocks: HashMap::new(),
                used: 0,
                latency: Duration::from_micros(20),
                bandwidth: 8e9,
                fail_reads: 0,
                fail_writes: 0,
                fail_full: 0,
                room_epoch: 0,
                health: TierHealth::new(None),
                next_slot: 0,
            }),
        }
    }

    pub fn set_estimates(&self, latency: Duration, bandwidth: f64) {
        let mut s = self.lock();
        s.latency = latency;
        s.bandwidth = bandwidth;
    }

    /// The next `n` `get` calls fail with `TierError::Io` (each counts toward degradation).
    pub fn inject_read_errors(&self, n: u32) {
        self.lock().fail_reads = n;
    }

    /// The next `n` `put` calls fail with `TierError::Io`.
    pub fn inject_write_errors(&self, n: u32) {
        self.lock().fail_writes = n;
    }

    /// The next `n` `put` / `put_as` calls find no room (`TierError::Full`), as a slab tier
    /// without a free slot of the block's size does; the stored copy stays.
    pub fn inject_full(&self, n: u32) {
        self.lock().fail_full = n;
    }

    pub fn len(&self) -> usize {
        self.lock().blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keys of the stored blocks with the bytes each accounts.
    pub fn sizes(&self) -> Vec<(KvKey, u64)> {
        self.lock()
            .blocks
            .iter()
            .map(|(k, b)| (*k, b.size))
            .collect()
    }

    fn store(&self, key: KvKey, bytes: &[u8], logical: Option<u64>) -> Result<TierSlot, TierError> {
        let now = self.clock.now_mono();
        let mut s = self.lock();
        if s.health.is_degraded() {
            return Err(TierError::Degraded);
        }
        if s.fail_writes > 0 {
            s.fail_writes -= 1;
            s.health.record_error(now);
            return Err(TierError::Io("injected write error".into()));
        }
        if s.fail_full > 0 {
            s.fail_full -= 1;
            return Err(TierError::Full);
        }
        let size = match self.nominal_block {
            Some(nominal) => logical.unwrap_or(nominal),
            None => bytes.len() as u64,
        };
        let replaced = s.blocks.get(&key).map_or(0, |b| b.size);
        if s.used - replaced + size > self.capacity_bytes {
            return Err(TierError::Full);
        }
        let stored = if self.nominal_block.is_some() {
            Vec::new()
        } else {
            bytes.to_vec()
        };
        if size < replaced {
            s.room_epoch += 1;
        }
        s.used = s.used - replaced + size;
        s.blocks.insert(
            key,
            Stored {
                bytes: stored,
                size,
            },
        );
        s.next_slot += 1;
        Ok(TierSlot(s.next_slot))
    }

    // The state is updated in single assignments that leave it consistent, so a poisoned lock
    // (a panic in another test thread) is safe to keep using.
    fn lock(&self) -> MutexGuard<'_, MemState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl KvTier for MemTier {
    fn id(&self) -> TierId {
        self.id
    }

    fn enabled(&self) -> bool {
        true
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    fn used_bytes(&self) -> u64 {
        self.lock().used
    }

    fn pressure(&self) -> PressureState {
        utilization_pressure(self.used_bytes(), self.capacity_bytes)
    }

    fn est_latency(&self) -> Duration {
        self.lock().latency
    }

    fn est_bandwidth(&self) -> Option<f64> {
        Some(self.lock().bandwidth)
    }

    fn contains(&self, key: &KvKey) -> bool {
        self.lock().blocks.contains_key(key)
    }

    fn put(&self, key: KvKey, src: TierBlockRef<'_>) -> Result<TierSlot, TierError> {
        let TierBlockRef::Host(bytes) = src;
        self.store(key, bytes, None)
    }

    /// Payload-free: accounts `bytes`; otherwise stores `src` as `put` does.
    fn put_as(
        &self,
        key: KvKey,
        _format: &'static str,
        bytes: u64,
        src: TierBlockRef<'_>,
    ) -> Result<TierSlot, TierError> {
        let TierBlockRef::Host(data) = src;
        self.store(key, data, Some(bytes))
    }

    fn get(&self, key: &KvKey, dst: TierBlockMut<'_>) -> Result<(), TierError> {
        let TierBlockMut::Host(out) = dst;
        let now = self.clock.now_mono();
        let mut s = self.lock();
        if s.health.is_degraded() {
            return Err(TierError::Degraded);
        }
        if s.fail_reads > 0 {
            s.fail_reads -= 1;
            s.health.record_error(now);
            return Err(TierError::Io("injected read error".into()));
        }
        let b = s.blocks.get(key).ok_or(TierError::Missing)?;
        if self.nominal_block.is_some() {
            out.fill(0);
            return Ok(());
        }
        if b.bytes.len() != out.len() {
            return Err(TierError::Io(format!(
                "block is {} bytes, buffer {}",
                b.bytes.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&b.bytes);
        Ok(())
    }

    fn evict(&self, key: &KvKey) -> Result<(), TierError> {
        let mut s = self.lock();
        let b = s.blocks.remove(key).ok_or(TierError::Missing)?;
        s.used -= b.size;
        s.room_epoch += 1;
        Ok(())
    }

    fn degraded(&self) -> bool {
        self.lock().health.is_degraded()
    }

    fn room_epoch(&self) -> u64 {
        self.lock().room_epoch
    }

    /// No slot sizes: bytes are the only limit.
    fn free_slots(&self, _format: &'static str, _bytes: u64) -> u64 {
        u64::MAX
    }
}
