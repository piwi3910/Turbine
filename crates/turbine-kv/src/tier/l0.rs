//! `L0Tier` (P4 S-4): the GPU pool seen through the `KvTier` trait for accounting only.
//! Capacity, usage and pressure come from the last `BlockPool` snapshot the hierarchy
//! published; `contains` follows the hierarchy's residency updates. L0 bytes move only on the
//! device copy stream (the transfer engine), so `put`, `get` and `evict` are refused.

use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use turbine_core::types::PressureState;

use super::{
    KvTier, TierBlockMut, TierBlockRef, TierError, TierId, TierSlot, utilization_pressure,
};
use crate::identity::KvKey;
use crate::pool::BlockPool;

const L0_BYTES_REFUSED: &str = "L0 blocks move through the transfer engine";

#[derive(Default)]
struct L0State {
    total_blocks: u32,
    used_blocks: u32,
    resident: HashSet<KvKey>,
}

pub struct L0Tier {
    block_bytes: u64,
    state: Mutex<L0State>,
}

impl L0Tier {
    /// An empty view; capacity is 0 until the first [`L0Tier::refresh`].
    pub fn new(block_bytes: u64) -> Self {
        L0Tier {
            block_bytes,
            state: Mutex::new(L0State::default()),
        }
    }

    /// Publishes the pool's current block counts.
    pub fn refresh(&self, pool: &BlockPool) {
        let mut s = self.lock();
        s.total_blocks = pool.total_blocks();
        s.used_blocks = pool.used_blocks();
    }

    /// Records whether `key` has a copy in L0.
    pub fn set_resident(&self, key: KvKey, resident: bool) {
        let mut s = self.lock();
        if resident {
            s.resident.insert(key);
        } else {
            s.resident.remove(&key);
        }
    }

    // Counts and a key set cannot be left half-updated, so a poisoned lock is safe to reuse.
    fn lock(&self) -> MutexGuard<'_, L0State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl KvTier for L0Tier {
    fn id(&self) -> TierId {
        TierId::L0
    }

    fn enabled(&self) -> bool {
        true
    }

    fn capacity_bytes(&self) -> u64 {
        u64::from(self.lock().total_blocks) * self.block_bytes
    }

    fn used_bytes(&self) -> u64 {
        u64::from(self.lock().used_blocks) * self.block_bytes
    }

    fn pressure(&self) -> PressureState {
        let s = self.lock();
        utilization_pressure(u64::from(s.used_blocks), u64::from(s.total_blocks))
    }

    fn est_latency(&self) -> Duration {
        Duration::ZERO
    }

    fn est_bandwidth(&self) -> Option<f64> {
        None
    }

    fn contains(&self, key: &KvKey) -> bool {
        self.lock().resident.contains(key)
    }

    fn put(&self, _key: KvKey, _src: TierBlockRef<'_>) -> Result<TierSlot, TierError> {
        Err(TierError::Io(L0_BYTES_REFUSED.into()))
    }

    fn get(&self, _key: &KvKey, _dst: TierBlockMut<'_>) -> Result<(), TierError> {
        Err(TierError::Io(L0_BYTES_REFUSED.into()))
    }

    fn evict(&self, _key: &KvKey) -> Result<(), TierError> {
        Err(TierError::Io(L0_BYTES_REFUSED.into()))
    }

    fn degraded(&self) -> bool {
        false
    }
}
