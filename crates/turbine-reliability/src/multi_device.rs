//! Multi-device pressure accounting (P5 S-8): a tensor-parallel group is only as healthy as
//! its worst device, a replica's KV reservation holds on every rank of its group or on none,
//! and data-parallel replicas that share a device (`parallel.allow_device_sharing`) split that
//! device's budget while reserving from its one ledger.

use std::sync::Arc;

use turbine_core::types::{DeviceId, PressureState};

use crate::admission::{AdmissionDecision, PressureReason};
use crate::budget::{DeviceBudget, PoolKind};
use crate::ledger::{Ledger, Reservation};

/// A TP group's pressure: the worst member state and the first device at that state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupState {
    pub state: PressureState,
    pub limiting_device: DeviceId,
}

/// The group state of `members` (device, its pressure state): the worst state, limited by the
/// first member at that state. An empty group is GREEN, limited by device 0.
pub fn group_state(members: &[(DeviceId, PressureState)]) -> (PressureState, DeviceId) {
    let worst = members
        .iter()
        .map(|(_, s)| *s)
        .max()
        .unwrap_or(PressureState::Green);
    let limiting = members
        .iter()
        .find(|(_, s)| *s == worst)
        .map_or(DeviceId(0), |(d, _)| *d);
    (worst, limiting)
}

impl GroupState {
    pub fn of(members: &[(DeviceId, PressureState)]) -> GroupState {
        let (state, limiting_device) = group_state(members);
        GroupState {
            state,
            limiting_device,
        }
    }
}

/// A replica's KV reservation on every rank of its group; dropping it releases every rank's
/// part.
#[derive(Debug)]
pub struct GroupReservation {
    pub reservations: Vec<Reservation>,
}

impl From<Vec<Reservation>> for GroupReservation {
    fn from(reservations: Vec<Reservation>) -> GroupReservation {
        GroupReservation { reservations }
    }
}

/// Reserves `blocks` KV blocks of `block_bytes_per_rank` on every rank's ledger, all or
/// nothing: the first rank that cannot hold them releases what the earlier ranks took and the
/// request queues with `KvReservation` (CONFLICT C-11).
pub fn reserve_group(
    ledgers: &[(DeviceId, Arc<Ledger>)],
    blocks: u32,
    block_bytes_per_rank: u64,
) -> Result<Vec<Reservation>, AdmissionDecision> {
    let bytes = u64::from(blocks).saturating_mul(block_bytes_per_rank);
    let mut taken = Vec::with_capacity(ledgers.len());
    for (device, ledger) in ledgers {
        match ledger.reserve(*device, PoolKind::Kv, bytes) {
            Ok(r) => taken.push(r),
            Err(e) => {
                tracing::debug!(
                    event = "group_reservation_queued",
                    device = device.0,
                    bytes,
                    error = %e,
                    "a rank cannot hold the group's KV reservation; nothing is held"
                );
                // `taken` drops here: every earlier rank's reservation is released.
                return Err(AdmissionDecision::Queue {
                    reason: PressureReason::KvReservation,
                });
            }
        }
    }
    Ok(taken)
}

/// One physical device shared by several data-parallel replicas: each replica sees its share of
/// the device budget, and every replica reserves from the device's single ledger.
#[derive(Debug, Clone)]
pub struct SharedDeviceBudget {
    budget: DeviceBudget,
    replicas: u32,
    ledger: Arc<Ledger>,
}

impl SharedDeviceBudget {
    /// The device's one ledger over its whole `budget`, shared by `replicas` replicas.
    pub fn new(budget: &DeviceBudget, replicas: u32) -> SharedDeviceBudget {
        SharedDeviceBudget {
            budget: budget.clone(),
            replicas: replicas.max(1),
            ledger: Ledger::new(budget),
        }
    }

    /// Each replica's view: the device budget and every pool divided by the replicas on it.
    pub fn split(budget: &DeviceBudget, replicas: u32) -> Vec<DeviceBudget> {
        let n = u64::from(replicas.max(1));
        (0..n)
            .map(|_| DeviceBudget {
                device: budget.device,
                memory_kind: budget.memory_kind,
                budget_bytes: budget.budget_bytes / n,
                pools: budget.pools.iter().map(|(k, b)| (*k, b / n)).collect(),
            })
            .collect()
    }

    /// The per-replica budgets of this device ([`SharedDeviceBudget::split`]).
    pub fn replica_budgets(&self) -> Vec<DeviceBudget> {
        SharedDeviceBudget::split(&self.budget, self.replicas)
    }

    /// The device's ledger, the same one for every replica on it.
    pub fn ledger(&self) -> &Arc<Ledger> {
        &self.ledger
    }

    /// KV bytes still available on `device` for any replica on it (0 for another device).
    pub fn available(&self, device: DeviceId) -> u64 {
        if device != self.budget.device {
            return 0;
        }
        self.ledger.usage(device, PoolKind::Kv).available()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use turbine_core::types::MemoryKind;

    use super::*;
    use crate::signals::PressureSignal;
    use crate::state::Gates;
    use crate::state::tests::{TICK, machine, sig, tick};

    const GIB: u64 = 1 << 30;

    fn budget(device: u32, kv: u64) -> DeviceBudget {
        DeviceBudget {
            device: DeviceId(device),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 32 * GIB,
            pools: vec![
                (PoolKind::Weights, 7 * GIB),
                (PoolKind::Kv, kv),
                (PoolKind::Workspace, GIB),
                (PoolKind::Collective, GIB / 2),
                (PoolKind::Runtime, GIB),
                (PoolKind::Reserve, 2 * GIB),
            ],
        }
    }

    /// Device 0 GREEN and device 1 ORANGE make the group ORANGE, limited by device 1; the group
    /// follows device 1's hysteresis down (dwell per level) and returns stepwise to GREEN.
    #[test]
    fn group_state_is_worst_member() {
        let dwell = Duration::from_secs(10);
        let (mut m0, c0, _) = machine(1, dwell);
        let (mut m1, c1, _) = machine(1, dwell);
        let green = [sig(PressureSignal::KvUtilization, 0.10)];
        let orange = [sig(PressureSignal::KvUtilization, 0.83)];
        let low = [sig(PressureSignal::KvUtilization, 0.10)];
        tick(&mut m0, &c0, &green, Gates::default());
        tick(&mut m1, &c1, &orange, Gates::default());
        let group = |m0: &crate::state::PressureMachine, m1: &crate::state::PressureMachine| {
            group_state(&[(DeviceId(0), m0.state()), (DeviceId(1), m1.state())])
        };
        assert_eq!(group(&m0, &m1), (PressureState::Orange, DeviceId(1)));
        assert_eq!(
            GroupState::of(&[(DeviceId(0), m0.state()), (DeviceId(1), m1.state())]),
            GroupState {
                state: PressureState::Orange,
                limiting_device: DeviceId(1)
            }
        );

        // Device 1's samples drop below ORANGE: the group holds ORANGE for the whole dwell.
        let ticks_per_dwell = (dwell.as_millis() / TICK.as_millis()) as usize;
        let mut seen = vec![group(&m0, &m1).0];
        for i in 0..4 * ticks_per_dwell {
            tick(&mut m0, &c0, &green, Gates::default());
            tick(&mut m1, &c1, &low, Gates::default());
            let (state, limiting) = group(&m0, &m1);
            if i + 1 < ticks_per_dwell {
                assert_eq!(state, PressureState::Orange, "tick {i}: inside the dwell");
            }
            if state != PressureState::Green {
                assert_eq!(limiting, DeviceId(1));
            }
            if seen.last() != Some(&state) {
                seen.push(state);
            }
        }
        assert_eq!(
            seen,
            [
                PressureState::Orange,
                PressureState::Yellow,
                PressureState::Green
            ],
            "one level per dwell"
        );
        // A tie limits by the first member at the worst state.
        assert_eq!(
            group_state(&[
                (DeviceId(3), PressureState::Red),
                (DeviceId(1), PressureState::Red)
            ]),
            (PressureState::Red, DeviceId(3))
        );
    }

    /// Rank 1's KV pool is one block short: the group reservation queues with KvReservation and
    /// neither ledger holds anything.
    #[test]
    fn atomic_group_reservation() {
        let block = 1 << 20;
        let n = 64u32;
        let l0 = Ledger::new(&budget(0, u64::from(n) * block));
        let l1 = Ledger::new(&budget(1, u64::from(n - 1) * block));
        let ledgers = [
            (DeviceId(0), Arc::clone(&l0)),
            (DeviceId(1), Arc::clone(&l1)),
        ];
        let before = (
            l0.usage(DeviceId(0), PoolKind::Kv),
            l1.usage(DeviceId(1), PoolKind::Kv),
        );
        match reserve_group(&ledgers, n, block) {
            Err(AdmissionDecision::Queue {
                reason: PressureReason::KvReservation,
            }) => {}
            other => panic!("expected Queue(KvReservation), got {other:?}"),
        }
        assert_eq!(
            (
                l0.usage(DeviceId(0), PoolKind::Kv),
                l1.usage(DeviceId(1), PoolKind::Kv)
            ),
            before,
            "nothing held on either rank"
        );
        // One block fewer fits on both, held until the group reservation drops.
        let held: GroupReservation = reserve_group(&ledgers, n - 1, block).unwrap().into();
        assert_eq!(held.reservations.len(), 2);
        assert_eq!(
            l0.usage(DeviceId(0), PoolKind::Kv).reserved,
            u64::from(n - 1) * block
        );
        assert_eq!(l1.usage(DeviceId(1), PoolKind::Kv).available(), 0);
        drop(held);
        assert_eq!(l0.usage(DeviceId(0), PoolKind::Kv), before.0);
        assert_eq!(l1.usage(DeviceId(1), PoolKind::Kv), before.1);
    }

    /// Two replicas on one 32 GiB device each see half the budget; a KV reservation by replica
    /// 0 lowers replica 1's headroom by the same bytes (one ledger per device).
    #[test]
    fn shared_device_budget() {
        let device = budget(0, 20 * GIB);
        let halves = SharedDeviceBudget::split(&device, 2);
        assert_eq!(halves.len(), 2);
        for h in &halves {
            assert_eq!(h.budget_bytes, 16 * GIB);
            assert_eq!(h.pool(PoolKind::Kv), 10 * GIB);
            assert_eq!(
                h.pool(PoolKind::Weights),
                device.pool(PoolKind::Weights) / 2
            );
        }
        let shared = SharedDeviceBudget::new(&device, 2);
        assert_eq!(shared.replica_budgets(), halves);
        let replica0 = shared.clone();
        let replica1 = shared.clone();
        let before = replica1.available(DeviceId(0));
        assert_eq!(before, 20 * GIB);
        let r = replica0
            .ledger()
            .reserve(DeviceId(0), PoolKind::Kv, 3 * GIB)
            .unwrap();
        assert_eq!(replica1.available(DeviceId(0)), before - 3 * GIB);
        assert!(Arc::ptr_eq(replica0.ledger(), replica1.ledger()));
        assert_eq!(replica1.available(DeviceId(1)), 0, "another device");
        drop(r);
        assert_eq!(replica1.available(DeviceId(0)), before);
    }
}
