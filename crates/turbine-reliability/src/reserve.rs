//! Emergency reserve (P3 S-4): device memory held outside normal scheduling, released only
//! by the recovery controller in SURVIVAL and re-acquired before the state may drop below RED.

use std::sync::Arc;

use turbine_core::types::{DeviceId, PressureState};

use crate::budget::PoolKind;
use crate::ledger::{Ledger, Reservation};
use crate::metrics::{DeviceLabel, ReliabilityMetrics};

/// The real device allocation behind the reserve (implemented by the server over a device
/// buffer; tests use a fake).
pub trait ReserveAllocator: Send {
    fn allocate(&mut self, bytes: u64) -> Result<(), String>;
    fn free(&mut self);
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ReserveError {
    #[error("the emergency reserve may only be released in SURVIVAL (state {})", .0.as_str())]
    NotSurvival(PressureState),
    #[error("the emergency reserve is not held")]
    NotHeld,
    #[error("emergency reserve allocation of {bytes} bytes failed: {detail}")]
    Allocation { bytes: u64, detail: String },
}

pub struct EmergencyReserve {
    device: DeviceId,
    bytes: u64,
    ledger: Arc<Ledger>,
    allocator: Box<dyn ReserveAllocator>,
    /// The committed `reserve`-pool reservation while held.
    reservation: Option<Reservation>,
    releases: u64,
    /// A re-acquisition failed since the last success (logs once per failing run).
    reacquire_failing: bool,
    metrics: ReliabilityMetrics,
}

impl std::fmt::Debug for EmergencyReserve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmergencyReserve")
            .field("device", &self.device)
            .field("bytes", &self.bytes)
            .field("held", &self.held())
            .field("releases", &self.releases)
            .finish_non_exhaustive()
    }
}

impl EmergencyReserve {
    /// Allocate the reserve at startup. `bytes == 0` disables it: it then always counts as
    /// held (the caller logs that at WARN).
    pub fn acquire(
        device: DeviceId,
        bytes: u64,
        ledger: &Arc<Ledger>,
        allocator: Box<dyn ReserveAllocator>,
        metrics: ReliabilityMetrics,
    ) -> Result<Self, ReserveError> {
        let mut reserve = EmergencyReserve {
            device,
            bytes,
            ledger: Arc::clone(ledger),
            allocator,
            reservation: None,
            releases: 0,
            reacquire_failing: false,
            metrics,
        };
        if bytes > 0 {
            reserve.allocate()?;
        }
        reserve.publish();
        Ok(reserve)
    }

    pub fn held(&self) -> bool {
        self.bytes == 0 || self.reservation.is_some()
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn releases(&self) -> u64 {
        self.releases
    }

    /// Hand the reserve back to the allocator; only in SURVIVAL (the recovery controller).
    pub fn release_for_recovery(&mut self, state: PressureState) -> Result<u64, ReserveError> {
        if state != PressureState::Survival {
            return Err(ReserveError::NotSurvival(state));
        }
        let Some(reservation) = self.reservation.take() else {
            return Err(ReserveError::NotHeld);
        };
        self.allocator.free();
        drop(reservation);
        self.releases += 1;
        self.metrics
            .emergency_reserve_releases
            .get_or_create(&DeviceLabel {
                device: self.device.0,
            })
            .inc();
        self.publish();
        tracing::warn!(
            event = "emergency_reserve",
            reason = "released",
            device = self.device.0,
            bytes = self.bytes,
            state = state.as_str(),
        );
        Ok(self.bytes)
    }

    /// Re-allocate after a release (called every tick while not held). False when the
    /// memory is not available (e.g. another process took it): the state then stays at RED.
    pub fn try_reacquire(&mut self) -> bool {
        if self.held() {
            return true;
        }
        match self.allocate() {
            Ok(()) => {
                self.reacquire_failing = false;
                self.publish();
                true
            }
            Err(e) => {
                if !self.reacquire_failing {
                    tracing::warn!(
                        event = "emergency_reserve",
                        reason = "reacquire_failed",
                        device = self.device.0,
                        bytes = self.bytes,
                        detail = %e,
                    );
                }
                self.reacquire_failing = true;
                false
            }
        }
    }

    /// Ledger reservation first (never over the pool), then the real allocation.
    fn allocate(&mut self) -> Result<(), ReserveError> {
        let mut reservation = self
            .ledger
            .reserve(self.device, PoolKind::Reserve, self.bytes)
            .map_err(|e| ReserveError::Allocation {
                bytes: self.bytes,
                detail: e.to_string(),
            })?;
        self.allocator
            .allocate(self.bytes)
            .map_err(|detail| ReserveError::Allocation {
                bytes: self.bytes,
                detail,
            })?;
        reservation.commit();
        self.reservation = Some(reservation);
        tracing::warn!(
            event = "emergency_reserve",
            reason = "acquired",
            device = self.device.0,
            bytes = self.bytes,
        );
        Ok(())
    }

    fn publish(&self) {
        self.metrics
            .emergency_reserve_held
            .get_or_create(&DeviceLabel {
                device: self.device.0,
            })
            .set(i64::from(self.held()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::DeviceBudget;
    use crate::signals::PressureSignal;
    use crate::state::Gates;
    use crate::state::tests::{machine, sig, tick};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use turbine_core::types::MemoryKind;

    struct FakeAlloc {
        available: Arc<AtomicBool>,
    }
    impl ReserveAllocator for FakeAlloc {
        fn allocate(&mut self, _bytes: u64) -> Result<(), String> {
            if self.available.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err("memory taken by another process".into())
            }
        }
        fn free(&mut self) {}
    }

    #[test]
    fn reserve_only_released_in_survival() {
        const GIB: u64 = 1 << 30;
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 4 * GIB,
            pools: vec![(PoolKind::Reserve, 2 * GIB)],
        };
        let ledger = Ledger::new(&budget);
        let available = Arc::new(AtomicBool::new(true));
        let alloc = Box::new(FakeAlloc {
            available: Arc::clone(&available),
        });
        let metrics = ReliabilityMetrics::unregistered();
        let held_gauge = || {
            metrics
                .emergency_reserve_held
                .get_or_create(&DeviceLabel { device: 0 })
                .get()
        };
        let mut reserve =
            EmergencyReserve::acquire(DeviceId(0), 2 * GIB, &ledger, alloc, metrics.clone())
                .unwrap();
        assert_eq!(held_gauge(), 1);
        assert!(reserve.held());
        assert_eq!(ledger.usage(DeviceId(0), PoolKind::Reserve).used, 2 * GIB);
        // Normal scheduling cannot reach the reserve: the pool is fully used and GREEN..RED refuse release.
        assert!(ledger.reserve(DeviceId(0), PoolKind::Reserve, 1).is_err());
        for state in [
            PressureState::Green,
            PressureState::Yellow,
            PressureState::Orange,
            PressureState::Red,
        ] {
            assert_eq!(
                reserve.release_for_recovery(state),
                Err(ReserveError::NotSurvival(state))
            );
            assert!(reserve.held());
        }
        // Recovery enters SURVIVAL (allocation failure) and releases the reserve.
        let (mut m, clock, _) = machine(2, Duration::from_secs(10));
        tick(
            &mut m,
            &clock,
            &[sig(PressureSignal::AllocationFailure, 1.0)],
            Gates::default(),
        );
        assert_eq!(m.state(), PressureState::Survival);
        assert_eq!(reserve.release_for_recovery(m.state()), Ok(2 * GIB));
        assert!(!reserve.held());
        assert_eq!(reserve.releases(), 1);
        assert_eq!(held_gauge(), 0);
        assert_eq!(
            metrics
                .emergency_reserve_releases
                .get_or_create(&DeviceLabel { device: 0 })
                .get(),
            1
        );
        assert_eq!(ledger.usage(DeviceId(0), PoolKind::Reserve).used, 0);
        assert_eq!(
            reserve.release_for_recovery(PressureState::Survival),
            Err(ReserveError::NotHeld)
        );
        // While the reserve is out (and another process holds the memory) the state stops at RED.
        available.store(false, Ordering::SeqCst);
        let clear = [sig(PressureSignal::AllocationFailure, 0.0)];
        for _ in 0..600 {
            assert!(!reserve.try_reacquire());
            let gates = Gates {
                reserve_held: reserve.held(),
                ..Gates::default()
            };
            tick(&mut m, &clock, &clear, gates);
            assert!(
                m.state() >= PressureState::Red,
                "no drop below RED without the reserve"
            );
        }
        assert_eq!(m.state(), PressureState::Red);
        // Re-acquisition succeeds → descent continues after one more dwell.
        available.store(true, Ordering::SeqCst);
        assert!(reserve.try_reacquire());
        assert_eq!(held_gauge(), 1);
        assert_eq!(ledger.usage(DeviceId(0), PoolKind::Reserve).used, 2 * GIB);
        let mut next = None;
        for _ in 0..101 {
            next = next.or(tick(
                &mut m,
                &clock,
                &clear,
                Gates {
                    reserve_held: reserve.held(),
                    ..Gates::default()
                },
            ));
        }
        assert_eq!(
            next.map(|t| (t.from, t.to)),
            Some((PressureState::Red, PressureState::Orange))
        );

        // A zero-byte reserve is disabled: always held, never touches the ledger.
        let disabled = EmergencyReserve::acquire(
            DeviceId(0),
            0,
            &ledger,
            Box::new(FakeAlloc {
                available: Arc::new(AtomicBool::new(false)),
            }),
            ReliabilityMetrics::unregistered(),
        )
        .unwrap();
        assert!(disabled.held());
        // A reserve that cannot be allocated at startup is an error.
        let err = EmergencyReserve::acquire(
            DeviceId(1),
            GIB,
            &ledger,
            Box::new(FakeAlloc {
                available: Arc::new(AtomicBool::new(true)),
            }),
            ReliabilityMetrics::unregistered(),
        );
        assert!(matches!(err, Err(ReserveError::Allocation { bytes, .. }) if bytes == GIB));
    }
}
