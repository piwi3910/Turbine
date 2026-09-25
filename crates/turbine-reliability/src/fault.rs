//! Fault injection (Cargo feature `fault-injection`, P3 S-16): fail every Nth pool
//! reservation and raise device OOM / kernel errors at chosen engine iterations.

use std::sync::atomic::{AtomicU64, Ordering};

use turbine_core::config::FaultInjectionConfig;

/// A fault the executor raises at a configured iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InjectedIterationFault {
    DeviceOom,
    KernelError { sticky: bool },
}

#[derive(Debug)]
pub struct FaultInjector {
    cfg: FaultInjectionConfig,
    /// 1-based count of reservation attempts seen so far.
    allocations: AtomicU64,
}

impl FaultInjector {
    pub fn new(cfg: FaultInjectionConfig) -> Self {
        FaultInjector {
            cfg,
            allocations: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> &FaultInjectionConfig {
        &self.cfg
    }

    /// Reservation attempts counted so far.
    pub fn allocations(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// True for every `alloc_fail_every`-th pool reservation.
    pub fn should_fail_alloc(&self) -> bool {
        let n = self.allocations.fetch_add(1, Ordering::Relaxed) + 1;
        matches!(self.cfg.alloc_fail_every, Some(every) if every > 0 && n.is_multiple_of(u64::from(every)))
    }

    /// The fault to raise at engine iteration `iteration`, if any (OOM wins a tie).
    pub fn iteration_fault(&self, iteration: u64) -> Option<InjectedIterationFault> {
        if self.cfg.oom_at_iteration == Some(iteration) {
            return Some(InjectedIterationFault::DeviceOom);
        }
        (self.cfg.kernel_error_at_iteration == Some(iteration)).then_some(
            InjectedIterationFault::KernelError {
                sticky: self.cfg.kernel_error_sticky,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::types::{DeviceId, MemoryKind};

    use super::*;
    use crate::budget::{DeviceBudget, PoolKind};
    use crate::ledger::{Ledger, LedgerError};
    use crate::metrics::{DevicePoolLabels, ReliabilityMetrics};

    #[test]
    fn every_fifth_reservation_fails() {
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 1 << 20,
            pools: vec![(PoolKind::Workspace, 1 << 20)],
        };
        let ledger = Ledger::new(&budget);
        let metrics = ReliabilityMetrics::unregistered();
        ledger.set_metrics(metrics.clone());
        let injector = Arc::new(FaultInjector::new(FaultInjectionConfig {
            alloc_fail_every: Some(5),
            oom_at_iteration: Some(7),
            kernel_error_at_iteration: Some(9),
            kernel_error_sticky: true,
            ..FaultInjectionConfig::default()
        }));
        ledger.set_fault_injector(Arc::clone(&injector));
        let failed: Vec<u32> = (1..=20u32)
            .filter(|_| {
                matches!(
                    ledger.reserve(DeviceId(0), PoolKind::Workspace, 1),
                    Err(LedgerError::Injected)
                )
            })
            .collect();
        assert_eq!(failed, [5, 10, 15, 20]);
        assert_eq!(injector.allocations(), 20);
        let counted = metrics
            .allocation_failures
            .get_or_create(&DevicePoolLabels {
                device: 0,
                pool: "workspace",
            })
            .get();
        assert_eq!(counted, 4);
        assert_eq!(injector.iteration_fault(6), None);
        assert_eq!(
            injector.iteration_fault(7),
            Some(InjectedIterationFault::DeviceOom)
        );
        assert_eq!(
            injector.iteration_fault(9),
            Some(InjectedIterationFault::KernelError { sticky: true })
        );
    }
}
