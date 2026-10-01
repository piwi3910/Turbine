//! Reservation ledger (P3 S-3): `reserve → commit → release`, with RAII guards that release
//! on drop, so cancellation, request failure and preemption never leak pool bytes.
//!
//! Mirror ledgers (P5 Task 33, decision "P5: group KV admission in static rank mode" B): in
//! `static` rank mode the leader keeps an exact copy of each worker rank's ledger
//! ([`Ledger::mirror`], built from the budget the worker reports once it loaded) and admits
//! through it like through a `local` rank's ledger. A mirror journals every change
//! ([`LedgerOp`], [`Ledger::drain_journal`]); the leader sends the journal to the worker with its
//! step plans, and the worker replays it on its real ledger in the same order
//! ([`LedgerReplica`]). Both sides compare [`Ledger::digest`]s of the pool to catch a divergence.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use turbine_core::types::DeviceId;

use crate::budget::{DeviceBudget, PoolKind};
use crate::metrics::{DevicePoolLabels, PoolLabels, ReliabilityMetrics};

/// One pool's bytes. Reservations alone never take `used + reserved` past `capacity`; bytes
/// `held` outside any reservation ([`Ledger::set_held`]) can, when the pool's owner already
/// holds more than the reservations account for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolUsage {
    pub capacity: u64,
    /// Committed bytes.
    pub used: u64,
    /// Reserved but not yet committed bytes.
    pub reserved: u64,
    /// Bytes in use that no reservation covers: the KV pool's referenced blocks beyond the
    /// committed bytes, e.g. cached prefix blocks a request attached (P4 S-3), which its
    /// reservation leaves out. Set by [`Ledger::set_held`]; never journaled.
    pub held: u64,
}

impl PoolUsage {
    /// Bytes in use: committed plus held outside any reservation.
    pub fn in_use(&self) -> u64 {
        self.used.saturating_add(self.held)
    }

    pub fn available(&self) -> u64 {
        self.capacity
            .saturating_sub(self.in_use())
            .saturating_sub(self.reserved)
    }

    /// (used + held + reserved) / capacity; 0 for an empty pool. Above 1 when the pool's owner
    /// holds more than the reservations left room for.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.in_use() + self.reserved) as f64 / self.capacity as f64
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

/// One change of a mirror ledger ([`Ledger::mirror`]), in the order it happened. `id` names the
/// reservation within its ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerOp {
    Reserve { id: u64, pool: PoolKind, bytes: u64 },
    Commit { id: u64, bytes: u64 },
    Release { id: u64 },
}

/// Per-device pools with their capacity, committed and reserved bytes.
#[derive(Default)]
pub struct Ledger {
    pools: Mutex<BTreeMap<(DeviceId, PoolKind), PoolUsage>>,
    /// The id of the next reservation handed out.
    next_id: AtomicU64,
    /// A mirror's journal of every change, appended under the pools lock.
    journal: Option<Mutex<Vec<LedgerOp>>>,
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

    /// A mirror of another process's ledger over the same `budget`: like [`Ledger::new`], and
    /// every reservation, commit and release is journaled for [`Ledger::drain_journal`].
    pub fn mirror(budget: &DeviceBudget) -> Arc<Ledger> {
        let mut ledger = Arc::into_inner(Ledger::new(budget)).expect("a new ledger is unshared");
        ledger.journal = Some(Mutex::new(Vec::new()));
        Arc::new(ledger)
    }

    /// The changes since the last call, oldest first (always empty unless a mirror).
    pub fn drain_journal(&self) -> Vec<LedgerOp> {
        self.journal.as_ref().map_or_else(Vec::new, |j| {
            std::mem::take(&mut *j.lock().unwrap_or_else(PoisonError::into_inner))
        })
    }

    /// [`Ledger::drain_journal`] and the [`Ledger::digest`] of `pool` right after the last
    /// drained change, read together (no change can land between them).
    pub fn drain_journal_with_digest(
        &self,
        device: DeviceId,
        pool: PoolKind,
    ) -> (Vec<LedgerOp>, u64) {
        let pools = self.lock();
        let usage = pools.get(&(device, pool)).copied().unwrap_or_default();
        let ops = self.drain_journal();
        drop(pools);
        (ops, digest_of(device, pool, &usage))
    }

    /// A digest of one pool's state (device, pool, capacity, committed and reserved bytes):
    /// equal digests of a mirror and the ledger it mirrors mean the same state.
    pub fn digest(&self, device: DeviceId, pool: PoolKind) -> u64 {
        digest_of(device, pool, &self.usage(device, pool))
    }

    /// Appends `op` to a mirror's journal; called with the pools lock held, so the journal keeps
    /// the ledger's order.
    fn record(&self, op: LedgerOp) {
        if let Some(j) = &self.journal {
            j.lock().unwrap_or_else(PoisonError::into_inner).push(op);
        }
    }
}

/// FNV-1a over a pool's identity and bytes ([`Ledger::digest`]).
fn digest_of(device: DeviceId, pool: PoolKind, u: &PoolUsage) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for word in [
        u64::from(device.0),
        pool as u64,
        u.capacity,
        u.used,
        u.reserved,
    ] {
        for b in word.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Ledger {
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.record(LedgerOp::Reserve { id, pool, bytes });
        drop(pools);
        Ok(Reservation {
            ledger: Arc::clone(self),
            id,
            device,
            pool,
            bytes,
            committed: 0,
            members: Vec::new(),
        })
    }

    /// The pool's owner reports `in_use` bytes it holds now (the KV block pool: its referenced
    /// blocks). What the committed bytes do not cover counts as `held`, so admission and
    /// `kv_utilization` see blocks taken without a reservation (P6b: attached cached prefixes
    /// filled L0 while the ledger showed 70 %). Not journaled: a mirror's owner reports its own.
    pub fn set_held(&self, device: DeviceId, pool: PoolKind, in_use: u64) {
        let mut pools = self.lock();
        if let Some(usage) = pools.get_mut(&(device, pool)) {
            let held = in_use.saturating_sub(usage.used);
            if usage.held != held {
                usage.held = held;
                let snapshot = *usage;
                self.publish(device, pool, &snapshot);
            }
        }
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
            // Everything in use, held outside reservations included (P6b).
            ("used", usage.in_use()),
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

    /// Applies `f` to one pool and journals `op` (a mirror), both under the pools lock.
    fn update(
        &self,
        device: DeviceId,
        pool: PoolKind,
        op: LedgerOp,
        f: impl FnOnce(&mut PoolUsage),
    ) {
        let mut pools = self.lock();
        // A Reservation exists only for a pool that `reserve` found.
        if let Some(usage) = pools.get_mut(&(device, pool)) {
            f(usage);
            let snapshot = *usage;
            self.publish(device, pool, &snapshot);
            self.record(op);
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
    /// Unique within its ledger ([`LedgerOp`]).
    id: u64,
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
        let op = LedgerOp::Commit {
            id: self.id,
            bytes: n,
        };
        self.ledger.update(self.device, self.pool, op, |u| {
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
        let op = LedgerOp::Release { id: self.id };
        self.ledger.update(self.device, self.pool, op, |u| {
            u.used -= committed;
            u.reserved -= reserved;
        });
    }
}

/// A worker rank's real ledger following the leader's mirror of it (P5 Task 33): replays the
/// mirror's journal in order, holding one reservation of its own per live mirror reservation,
/// so both ledgers pass through the same states.
pub struct LedgerReplica {
    ledger: Arc<Ledger>,
    device: DeviceId,
    /// The replica's reservation of each live mirror reservation id.
    held: BTreeMap<u64, Reservation>,
}

impl LedgerReplica {
    /// Replays onto `device`'s pools of `ledger`.
    pub fn new(ledger: Arc<Ledger>, device: DeviceId) -> LedgerReplica {
        LedgerReplica {
            ledger,
            device,
            held: BTreeMap::new(),
        }
    }

    /// Applies `ops` in order. Every op that can be applied is; the first that cannot (the real
    /// ledger refuses a reservation the mirror took, or an unknown id) is returned as the reason
    /// once the rest are applied.
    pub fn apply(&mut self, ops: &[LedgerOp]) -> Result<(), String> {
        let mut first = None;
        for op in ops {
            let failed = match *op {
                LedgerOp::Reserve { id, pool, bytes } => {
                    match self.ledger.reserve(self.device, pool, bytes) {
                        Ok(r) => {
                            self.held.insert(id, r);
                            None
                        }
                        Err(e) => Some(format!("reservation {id}: {e}")),
                    }
                }
                LedgerOp::Commit { id, bytes } => match self.held.get_mut(&id) {
                    Some(r) => {
                        r.commit_bytes(bytes);
                        None
                    }
                    None => Some(format!("commit of unknown reservation {id}")),
                },
                LedgerOp::Release { id } => match self.held.remove(&id) {
                    Some(_) => None,
                    None => Some(format!("release of unknown reservation {id}")),
                },
            };
            if first.is_none() {
                first = failed;
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// [`Ledger::digest`] of `pool` on the replica's device.
    pub fn digest(&self, pool: PoolKind) -> u64 {
        self.ledger.digest(self.device, pool)
    }

    /// Live reservations replayed from the mirror.
    pub fn held(&self) -> usize {
        self.held.len()
    }

    /// The real ledger.
    pub fn ledger(&self) -> &Arc<Ledger> {
        &self.ledger
    }

    /// The device whose pools it replays onto.
    pub fn device(&self) -> DeviceId {
        self.device
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
                reserved: 200,
                held: 0,
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

    /// P6b: blocks the KV pool holds outside any reservation (attached cached prefixes) count
    /// against admission and utilisation: with 100 of 300 reserved bytes committed and the pool
    /// holding 400, 400 + 200 = 60 % is in use or promised and a reservation of 500 is refused;
    /// held bytes the commits already cover add nothing; reporting 0 clears them; a mirror's
    /// digest ignores them. Breaks if `set_held` is ignored, double-counts committed bytes, or
    /// enters the digest.
    #[test]
    fn held_bytes_outside_reservations_count() {
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 1000,
            pools: vec![(PoolKind::Kv, 1000)],
        };
        let ledger = Ledger::new(&budget);
        let metrics = ReliabilityMetrics::unregistered();
        ledger.set_metrics(metrics.clone());
        let (d, kv) = (DeviceId(0), PoolKind::Kv);
        let mut r = ledger.reserve(d, kv, 300).unwrap();
        r.commit_bytes(100);
        let digest = ledger.digest(d, kv);
        ledger.set_held(d, kv, 400);
        let u = ledger.usage(d, kv);
        assert_eq!((u.used, u.held, u.reserved), (100, 300, 200), "{u:?}");
        assert!((u.utilization() - 0.6).abs() < 1e-12, "{u:?}");
        assert_eq!(u.available(), 400);
        assert!(matches!(
            ledger.reserve(d, kv, 500),
            Err(LedgerError::Exhausted { available: 400, .. })
        ));
        let used_gauge = metrics
            .memory_pool_bytes
            .get_or_create(&PoolLabels {
                device: 0,
                pool: "kv",
                kind: "used",
            })
            .get();
        assert_eq!(used_gauge, 400, "the used gauge shows what is in use");
        assert_eq!(
            ledger.digest(d, kv),
            digest,
            "held bytes stay out of the digest"
        );
        // Held bytes the commits cover add nothing.
        ledger.set_held(d, kv, 50);
        assert_eq!(ledger.usage(d, kv).held, 0);
        assert!((ledger.usage(d, kv).utilization() - 0.3).abs() < 1e-12);
        // Holding more than the capacity reads above 1 and leaves nothing available.
        ledger.set_held(d, kv, 1100);
        let u = ledger.usage(d, kv);
        assert!(u.utilization() > 1.0, "{u:?}");
        assert_eq!(u.available(), 0);
        ledger.set_held(d, kv, 0);
        drop(r);
        assert_eq!(
            ledger.usage(d, kv),
            PoolUsage {
                capacity: 1000,
                ..PoolUsage::default()
            }
        );
    }

    /// P5 Task 33: a mirror journals every reservation, commit and release (group members
    /// included); replaying the journal in chunks on another ledger over the same budget reaches
    /// the same state (equal digests) after every chunk, a replica with a skewed ledger differs,
    /// and an unknown id is reported without stopping the replay. Breaks if a change escapes the
    /// journal or the replay reorders it.
    #[test]
    fn mirror_journal_replays_to_the_same_state() {
        let budget = DeviceBudget {
            device: DeviceId(1),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 1000,
            pools: vec![(PoolKind::Kv, 1000)],
        };
        let mirror = Ledger::mirror(&budget);
        let mut replica = LedgerReplica::new(Ledger::new(&budget), DeviceId(1));
        let leader = Ledger::new(&DeviceBudget {
            device: DeviceId(0),
            ..budget.clone()
        });
        let mut rng = 7u64;
        let mut held: Vec<Reservation> = Vec::new();
        for step in 0..400 {
            match splitmix(&mut rng) % 3 {
                0 => {
                    let bytes = 1 + splitmix(&mut rng) % 200;
                    // A group reservation: rank 0's carrying the mirror's as a member.
                    if let Ok(m) = mirror.reserve(DeviceId(1), PoolKind::Kv, bytes)
                        && let Ok(r) = leader.reserve(DeviceId(0), PoolKind::Kv, bytes)
                    {
                        held.push(r.with_members(vec![m]));
                    }
                }
                1 if !held.is_empty() => {
                    let i = (splitmix(&mut rng) as usize) % held.len();
                    let part = splitmix(&mut rng) % 64;
                    held[i].commit_bytes(part);
                }
                _ if !held.is_empty() => {
                    let i = (splitmix(&mut rng) as usize) % held.len();
                    drop(held.swap_remove(i));
                }
                _ => {}
            }
            if step % 7 == 0 {
                replica.apply(&mirror.drain_journal()).expect("replay");
                assert_eq!(
                    replica.digest(PoolKind::Kv),
                    mirror.digest(DeviceId(1), PoolKind::Kv),
                    "step {step}"
                );
                assert_eq!(
                    replica.ledger().usage(DeviceId(1), PoolKind::Kv),
                    mirror.usage(DeviceId(1), PoolKind::Kv)
                );
            }
        }
        drop(held);
        replica.apply(&mirror.drain_journal()).expect("replay");
        assert_eq!(replica.held(), 0);
        assert_eq!(replica.ledger().usage(DeviceId(1), PoolKind::Kv).used, 0);
        assert!(
            Ledger::new(&budget).drain_journal().is_empty(),
            "not a mirror"
        );

        // A skewed real ledger: the digests differ.
        let _skew = replica
            .ledger()
            .reserve(DeviceId(1), PoolKind::Kv, 3)
            .unwrap();
        assert_ne!(
            replica.digest(PoolKind::Kv),
            mirror.digest(DeviceId(1), PoolKind::Kv)
        );
        let err = replica
            .apply(&[
                LedgerOp::Commit { id: 99, bytes: 1 },
                LedgerOp::Reserve {
                    id: 100,
                    pool: PoolKind::Kv,
                    bytes: 5,
                },
            ])
            .unwrap_err();
        assert!(err.contains("unknown reservation 99"), "{err}");
        assert_eq!(replica.held(), 1, "the later op still applied");
    }
}
