//! Reservation ledger (P3 S-3): `reserve → commit → release`, with RAII guards that release
//! on drop, so cancellation, request failure and preemption never leak pool bytes.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use turbine_core::types::DeviceId;

use crate::budget::{DeviceBudget, PoolKind};
use crate::metrics::{DevicePoolLabels, PoolLabels, ReliabilityMetrics};

/// One pool's bytes. Invariant: `used + reserved <= capacity`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolUsage {
    pub capacity: u64,
    /// Committed bytes.
    pub used: u64,
    /// Reserved but not yet committed bytes.
    pub reserved: u64,
}

impl PoolUsage {
    pub fn available(&self) -> u64 {
        self.capacity
            .saturating_sub(self.used)
            .saturating_sub(self.reserved)
    }

    /// (used + reserved) / capacity; 0 for an empty pool.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.used + self.reserved) as f64 / self.capacity as f64
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LedgerError {
    #[error("pool {} exhausted: requested {requested} bytes, {available} available", pool.as_str())]
    Exhausted {
        pool: PoolKind,
        requested: u64,
        available: u64,
    },
    #[error("injected allocation failure (fault injection)")]
    Injected,
}

/// Per-device pools with their capacity, committed and reserved bytes.
#[derive(Default)]
pub struct Ledger {
    pools: Mutex<BTreeMap<(DeviceId, PoolKind), PoolUsage>>,
    metrics: OnceLock<ReliabilityMetrics>,
    #[cfg(feature = "fault-injection")]
    injector: OnceLock<Arc<crate::fault::FaultInjector>>,
}

impl fmt::Debug for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ledger")
            .field("pools", &*self.lock())
            .finish_non_exhaustive()
    }
}

impl Ledger {
    /// One pool per `budget.pools` entry, all empty.
    pub fn new(budget: &DeviceBudget) -> Arc<Ledger> {
        let ledger = Ledger::default();
        {
            let mut pools = ledger.lock();
            for (kind, bytes) in &budget.pools {
                pools.insert(
                    (budget.device, *kind),
                    PoolUsage {
                        capacity: *bytes,
                        ..PoolUsage::default()
                    },
                );
            }
        }
        Arc::new(ledger)
    }

    /// Attach metrics (first call wins); publishes every pool, then every change.
    pub fn set_metrics(&self, metrics: ReliabilityMetrics) {
        // A second call keeps the first metrics: the families are process-wide anyway.
        let _ = self.metrics.set(metrics);
        let pools = self.lock();
        for ((device, pool), usage) in pools.iter() {
            self.publish(*device, *pool, usage);
        }
    }

    /// Attach the allocation fault injector (first call wins).
    #[cfg(feature = "fault-injection")]
    pub fn set_fault_injector(&self, injector: Arc<crate::fault::FaultInjector>) {
        let _ = self.injector.set(injector);
    }

    /// Reserve `bytes` of `pool` on `device`; refused when the pool cannot hold them.
    pub fn reserve(
        self: &Arc<Self>,
        device: DeviceId,
        pool: PoolKind,
        bytes: u64,
    ) -> Result<Reservation, LedgerError> {
        #[cfg(feature = "fault-injection")]
        if self.injector.get().is_some_and(|f| f.should_fail_alloc()) {
            self.count_failure(device, pool, bytes, None);
            return Err(LedgerError::Injected);
        }
        let mut pools = self.lock();
        let Some(usage) = pools.get_mut(&(device, pool)) else {
            drop(pools);
            self.count_failure(device, pool, bytes, Some(0));
            return Err(LedgerError::Exhausted {
                pool,
                requested: bytes,
                available: 0,
            });
        };
        let available = usage.available();
        if bytes > available {
            drop(pools);
            self.count_failure(device, pool, bytes, Some(available));
            return Err(LedgerError::Exhausted {
                pool,
                requested: bytes,
                available,
            });
        }
        usage.reserved += bytes;
        let snapshot = *usage;
        self.publish(device, pool, &snapshot);
        drop(pools);
        Ok(Reservation {
            ledger: Arc::clone(self),
            device,
            pool,
            bytes,
            committed: 0,
            members: Vec::new(),
        })
    }

    /// Current usage of one pool (all zero for an unknown pool).
    pub fn usage(&self, device: DeviceId, pool: PoolKind) -> PoolUsage {
        self.lock()
            .get(&(device, pool))
            .copied()
            .unwrap_or_default()
    }

    // Every mutation is a single in-place update, so a poisoned lock never holds a
    // half-applied change and is safe to keep using.
    fn lock(&self) -> MutexGuard<'_, BTreeMap<(DeviceId, PoolKind), PoolUsage>> {
        self.pools.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn count_failure(
        &self,
        device: DeviceId,
        pool: PoolKind,
        requested: u64,
        available: Option<u64>,
    ) {
        if let Some(m) = self.metrics.get() {
            m.allocation_failures
                .get_or_create(&DevicePoolLabels {
                    device: device.0,
                    pool: pool.as_str(),
                })
                .inc();
        }
        let reason = if available.is_some() {
            "pool_exhausted"
        } else {
            "injected"
        };
        tracing::warn!(
            event = "allocation_failure",
            reason,
            device = device.0,
            pool = pool.as_str(),
            requested,
            available,
        );
    }

    /// Called with the pool lock held so gauge updates from concurrent threads stay ordered.
    fn publish(&self, device: DeviceId, pool: PoolKind, usage: &PoolUsage) {
        let Some(m) = self.metrics.get() else { return };
        for (kind, v) in [
            ("capacity", usage.capacity),
            ("used", usage.used),
            ("reserved", usage.reserved),
        ] {
            m.memory_pool_bytes
                .get_or_create(&PoolLabels {
                    device: device.0,
                    pool: pool.as_str(),
                    kind,
                })
                .set(i64::try_from(v).unwrap_or(i64::MAX));
        }
    }

    fn update(&self, device: DeviceId, pool: PoolKind, f: impl FnOnce(&mut PoolUsage)) {
        let mut pools = self.lock();
        // A Reservation exists only for a pool that `reserve` found.
        if let Some(usage) = pools.get_mut(&(device, pool)) {
            f(usage);
            let snapshot = *usage;
            self.publish(device, pool, &snapshot);
        }
    }
}

/// A pool reservation. Uncommitted bytes count as `reserved`, committed bytes as `used`;
/// dropping the guard returns both to the pool.
///
/// A tensor-parallel group's KV reservation (P5 S-8) is rank 0's reservation carrying every
/// other rank's as `members` ([`Reservation::with_members`]): committing on it commits the same
/// share of each member, and dropping it releases every rank's part, so the scheduler and the
/// block pool keep handling one guard per request.
pub struct Reservation {
    ledger: Arc<Ledger>,
    device: DeviceId,
    pool: PoolKind,
    bytes: u64,
    committed: u64,
    members: Vec<Reservation>,
}

impl Reservation {
    /// Commit every remaining reserved byte.
    pub fn commit(&mut self) {
        self.commit_bytes(self.bytes - self.committed);
    }

    /// Commit `bytes` more (e.g. KV blocks as a sequence grows); saturates at the total. Each
    /// member commits the same fraction of its own bytes (rounded up, so a fully committed
    /// reservation leaves every member fully committed).
    pub fn commit_bytes(&mut self, bytes: u64) {
        let n = bytes.min(self.bytes - self.committed);
        if n == 0 {
            return;
        }
        self.committed += n;
        self.ledger.update(self.device, self.pool, |u| {
            u.reserved -= n;
            u.used += n;
        });
        let (committed, total) = (u128::from(self.committed), u128::from(self.bytes));
        for m in &mut self.members {
            let target = (committed * u128::from(m.bytes)).div_ceil(total);
            let target = u64::try_from(target).unwrap_or(m.bytes);
            m.commit_bytes(target.saturating_sub(m.committed));
        }
    }

    /// This reservation carrying `members` (the other ranks' parts of a group reservation):
    /// they commit with it and are released when it drops.
    pub fn with_members(mut self, members: Vec<Reservation>) -> Reservation {
        self.members.extend(members);
        self
    }

    /// The other ranks' parts of a group reservation (empty on one device).
    pub fn members(&self) -> &[Reservation] {
        &self.members
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn committed(&self) -> u64 {
        self.committed
    }

    pub fn pool(&self) -> PoolKind {
        self.pool
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let (committed, reserved) = (self.committed, self.bytes - self.committed);
        self.ledger.update(self.device, self.pool, |u| {
            u.used -= committed;
            u.reserved -= reserved;
        });
    }
}

impl fmt::Debug for Reservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reservation")
            .field("device", &self.device)
            .field("pool", &self.pool)
            .field("bytes", &self.bytes)
            .field("committed", &self.committed)
            .field("members", &self.members)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbine_core::types::MemoryKind;

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn reservations_never_exceed_capacity() {
        const CAPACITY: u64 = 1_000_000;
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: CAPACITY,
            pools: vec![(PoolKind::Kv, CAPACITY)],
        };
        let ledger = Ledger::new(&budget);
        let metrics = ReliabilityMetrics::unregistered();
        ledger.set_metrics(metrics.clone());
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let ledger = Arc::clone(&ledger);
                std::thread::spawn(move || {
                    let mut rng = 0xC0FFEE ^ t;
                    let mut held: Vec<Reservation> = Vec::new();
                    for _ in 0..1250 {
                        match splitmix(&mut rng) % 4 {
                            0 | 1 => {
                                let bytes = 1 + splitmix(&mut rng) % 50_000;
                                if let Ok(r) = ledger.reserve(DeviceId(0), PoolKind::Kv, bytes) {
                                    held.push(r);
                                }
                            }
                            2 if !held.is_empty() => {
                                let i = (splitmix(&mut rng) as usize) % held.len();
                                let part = splitmix(&mut rng) % (held[i].bytes() + 1);
                                held[i].commit_bytes(part);
                            }
                            _ if !held.is_empty() => {
                                let i = (splitmix(&mut rng) as usize) % held.len();
                                drop(held.swap_remove(i));
                            }
                            _ => {}
                        }
                        let u = ledger.usage(DeviceId(0), PoolKind::Kv);
                        assert!(u.used + u.reserved <= u.capacity, "over-commit: {u:?}");
                    }
                    held.len()
                })
            })
            .collect();
        for t in threads {
            t.join().expect("worker thread");
        }
        let u = ledger.usage(DeviceId(0), PoolKind::Kv);
        assert_eq!(
            (u.used, u.reserved),
            (0, 0),
            "every dropped guard released its bytes"
        );
        let err = ledger
            .reserve(DeviceId(0), PoolKind::Kv, CAPACITY + 1)
            .unwrap_err();
        assert_eq!(
            err,
            LedgerError::Exhausted {
                pool: PoolKind::Kv,
                requested: CAPACITY + 1,
                available: CAPACITY
            }
        );
        let gauge = |kind| {
            metrics
                .memory_pool_bytes
                .get_or_create(&PoolLabels {
                    device: 0,
                    pool: "kv",
                    kind,
                })
                .get()
        };
        assert_eq!(
            (gauge("capacity"), gauge("used"), gauge("reserved")),
            (1_000_000, 0, 0)
        );
        let refused = metrics
            .allocation_failures
            .get_or_create(&DevicePoolLabels {
                device: 0,
                pool: "kv",
            })
            .get();
        assert!(refused >= 1, "the oversize request is counted");

        // Commit moves bytes from reserved to used; an unknown pool refuses.
        let mut r = ledger.reserve(DeviceId(0), PoolKind::Kv, 300).unwrap();
        r.commit_bytes(100);
        assert_eq!(
            ledger.usage(DeviceId(0), PoolKind::Kv),
            PoolUsage {
                capacity: CAPACITY,
                used: 100,
                reserved: 200
            }
        );
        r.commit();
        assert_eq!((r.committed(), r.bytes()), (300, 300));
        drop(r);
        assert_eq!(
            ledger.usage(DeviceId(0), PoolKind::Kv).available(),
            CAPACITY
        );
        assert!(matches!(
            ledger.reserve(DeviceId(0), PoolKind::Workspace, 1),
            Err(LedgerError::Exhausted { available: 0, .. })
        ));
    }
}
