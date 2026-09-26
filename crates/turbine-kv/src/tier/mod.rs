//! Tier abstraction (P4 S-4): the `KvTier` trait over block slots, tier ids and locations, and
//! the error-window health every tier shares. Tiers own bytes; the directory owns metadata.
//! Attaching L0 blocks to a batch is a synchronous pool operation and never goes through this
//! trait (TS §8); the trait serves copies, capacity and accounting.

use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use turbine_core::types::{PressureSignal, PressureState};
use turbine_reliability::signals::{SignalThresholds, default_thresholds};

use crate::identity::KvKey;

mod l0;
mod l1;
mod l2;
mod mem;

pub use l0::L0Tier;
pub use l1::{L1Config, L1PinnedTier};
pub use l2::{L2Config, L2NvmeTier, SLAB_MAGIC};
pub use mem::MemTier;

/// A KV tier, fastest first. Serde and label spelling `"l0"`..`"l3"` everywhere (CONFLICT C-4);
/// `L3` (cluster) arrives with Phase 6.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierId {
    L0,
    L1,
    L2,
    L3,
}

impl TierId {
    /// Local tiers of Phase 4, fastest first (metric label sets).
    pub const LOCAL: [TierId; 3] = [TierId::L0, TierId::L1, TierId::L2];

    pub fn as_str(self) -> &'static str {
        match self {
            TierId::L0 => "l0",
            TierId::L1 => "l1",
            TierId::L2 => "l2",
            TierId::L3 => "l3",
        }
    }
}

/// Where one copy of a block lives. `slot` is the L0 `BlockId`; for L1/L2 `slab << 32 | slot`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct KvLocation {
    pub tier: TierId,
    pub slot: u64,
}

/// The slot a `put` stored a block in (same encoding as [`KvLocation::slot`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TierSlot(pub u64);

/// Source bytes of one block.
pub enum TierBlockRef<'a> {
    Host(&'a [u8]),
}

/// Destination of one block; must be exactly the stored block's length.
pub enum TierBlockMut<'a> {
    Host(&'a mut [u8]),
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum TierError {
    #[error("tier full")]
    Full,
    #[error("not found")]
    Missing,
    #[error("checksum mismatch")]
    Checksum,
    #[error("tier I/O: {0}")]
    Io(String),
    #[error("tier degraded")]
    Degraded,
}

/// One KV tier (contract §11). Methods take `&self`: tiers synchronise internally so the
/// transfer engine's I/O threads can share them.
pub trait KvTier: Send + Sync {
    fn id(&self) -> TierId;
    fn enabled(&self) -> bool;
    fn capacity_bytes(&self) -> u64;
    fn used_bytes(&self) -> u64;
    fn pressure(&self) -> PressureState;
    /// Latency of bringing one block from this tier to L0.
    fn est_latency(&self) -> Duration;
    /// Bytes per second from this tier to L0 (`None` for L0 itself).
    fn est_bandwidth(&self) -> Option<f64>;
    fn contains(&self, key: &KvKey) -> bool;
    /// Stores a block, replacing any block stored under `key`.
    fn put(&self, key: KvKey, src: TierBlockRef<'_>) -> Result<TierSlot, TierError>;
    fn get(&self, key: &KvKey, dst: TierBlockMut<'_>) -> Result<(), TierError>;
    fn evict(&self, key: &KvKey) -> Result<(), TierError>;
    fn degraded(&self) -> bool;
}

/// Pressure of a tier from its utilisation, on the Phase 3 `kv_utilization` thresholds
/// (0.70 / 0.82 / 0.90 / 0.97).
pub fn utilization_pressure(used: u64, capacity: u64) -> PressureState {
    if capacity == 0 {
        return PressureState::Green;
    }
    kv_utilization_thresholds().level(used as f64 / capacity as f64)
}

/// Phase 3's documented `kv_utilization` thresholds (contract §8.3), shared so a tier's level
/// and the pressure controller's never disagree.
fn kv_utilization_thresholds() -> SignalThresholds {
    static T: OnceLock<SignalThresholds> = OnceLock::new();
    *T.get_or_init(|| default_thresholds()[&PressureSignal::KvUtilization])
}

/// The tier a block leaving `from` is copied to: L0 → L1 when L1 is enabled (discrete VRAM),
/// otherwise L2 (unified memory); L1 → L2. `None` means the block is dropped.
pub fn demotion_target(from: TierId, l1_enabled: bool, l2_enabled: bool) -> Option<TierId> {
    match from {
        TierId::L0 if l1_enabled => Some(TierId::L1),
        TierId::L0 | TierId::L1 if l2_enabled => Some(TierId::L2),
        _ => None,
    }
}

/// Error-window health (P4 Failure modes): 3 errors within 60 s mark a tier degraded; with a
/// probe interval (L2: 5 minutes) one probe then decides whether it recovers.
#[derive(Debug)]
pub struct TierHealth {
    errors: VecDeque<Duration>,
    degraded_since: Option<Duration>,
    probe_after: Option<Duration>,
}

impl TierHealth {
    pub const ERRORS_TO_DEGRADE: usize = 3;
    pub const WINDOW: Duration = Duration::from_secs(60);

    pub fn new(probe_after: Option<Duration>) -> Self {
        TierHealth {
            errors: VecDeque::with_capacity(Self::ERRORS_TO_DEGRADE + 1),
            degraded_since: None,
            probe_after,
        }
    }

    /// Records one I/O or copy error; returns true when this error degrades the tier.
    pub fn record_error(&mut self, now: Duration) -> bool {
        while self
            .errors
            .front()
            .is_some_and(|t| now.saturating_sub(*t) > Self::WINDOW)
        {
            self.errors.pop_front();
        }
        self.errors.push_back(now);
        while self.errors.len() > Self::ERRORS_TO_DEGRADE {
            self.errors.pop_front();
        }
        if self.degraded_since.is_none() && self.errors.len() >= Self::ERRORS_TO_DEGRADE {
            self.degraded_since = Some(now);
            return true;
        }
        false
    }

    /// Degrades the tier for a reason other than errors (L2: p99 above 10× calibration).
    /// Returns true when it was healthy.
    pub fn mark_degraded(&mut self, now: Duration) -> bool {
        let was_healthy = self.degraded_since.is_none();
        if was_healthy {
            self.degraded_since = Some(now);
        }
        was_healthy
    }

    pub fn is_degraded(&self) -> bool {
        self.degraded_since.is_some()
    }

    /// Whether the degraded period has lasted `probe_after` (never without a probe interval).
    pub fn probe_due(&self, now: Duration) -> bool {
        match (self.degraded_since, self.probe_after) {
            (Some(since), Some(after)) => now.saturating_sub(since) >= after,
            _ => false,
        }
    }

    /// Probe outcome: success clears the state; failure restarts the degraded period.
    pub fn probe_result(&mut self, ok: bool, now: Duration) {
        if ok {
            self.errors.clear();
            self.degraded_since = None;
        } else {
            self.degraded_since = Some(now);
        }
    }
}

#[cfg(test)]
mod tests;
