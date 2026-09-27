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
    /// `Some(block_bytes)`: payload-free mode (every block accounts `block_bytes`, none stored).
    nominal_block: Option<u64>,
    clock: Arc<dyn Clock>,
    state: Mutex<MemState>,
}

struct MemState {
    blocks: HashMap<KvKey, Vec<u8>>,
    used: u64,
    latency: Duration,
    bandwidth: f64,
    fail_reads: u32,
    fail_writes: u32,
    health: TierHealth,
    next_slot: u64,
}

impl MemTier {
    /// A tier storing block bytes, with pinned-copy-like default estimates (20 µs, 8 GB/s).
    pub fn new(id: TierId, capacity_bytes: u64, clock: Arc<dyn Clock>) -> Self {
        Self::build(id, capacity_bytes, None, clock)
    }

    /// A tier that accounts `block_bytes` per block and stores nothing; `get` fills zeros.
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

    pub fn len(&self) -> usize {
        self.lock().blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn size_of(&self, stored: &[u8]) -> u64 {
        self.nominal_block.unwrap_or(stored.len() as u64)
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
        let size = self.size_of(bytes);
        let replaced = s.blocks.get(&key).map_or(0, |b| self.size_of(b));
        if s.used - replaced + size > self.capacity_bytes {
            return Err(TierError::Full);
        }
        let stored = if self.nominal_block.is_some() {
            Vec::new()
        } else {
            bytes.to_vec()
        };
        s.used = s.used - replaced + size;
        s.blocks.insert(key, stored);
        s.next_slot += 1;
        Ok(TierSlot(s.next_slot))
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
        if b.len() != out.len() {
            return Err(TierError::Io(format!(
                "block is {} bytes, buffer {}",
                b.len(),
                out.len()
            )));
        }
        out.copy_from_slice(b);
        Ok(())
    }

    fn evict(&self, key: &KvKey) -> Result<(), TierError> {
        let mut s = self.lock();
        let b = s.blocks.remove(key).ok_or(TierError::Missing)?;
        s.used -= self.size_of(&b);
        Ok(())
    }

    fn degraded(&self) -> bool {
        self.lock().health.is_degraded()
    }
}
