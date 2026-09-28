//! Reliability metric families (P3 §Interfaces "Metrics"). Every label value comes from a
//! closed enum's `as_str()` or the Phase 0 device index. Counters are registered without
//! `_total`; prometheus-client appends it when rendering OpenMetrics.

use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_observability::MetricsRegistry;

/// A gauge holding an `f64` (signal values, horizon, plan fields).
pub type FloatGauge = Gauge<f64, AtomicU64>;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct StateLabel {
    pub state: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TransitionLabels {
    pub from: &'static str,
    pub to: &'static str,
    pub signal: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SignalLabel {
    pub signal: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DecisionLabels {
    pub decision: &'static str,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct FieldLabel {
    pub field: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ActionLabel {
    pub action: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PoolLabels {
    pub device: u32,
    pub pool: &'static str,
    /// `capacity`, `used` or `reserved`.
    pub kind: &'static str,
}

/// `turbine_device_budget_bytes{device,component}` (P5; `component` from `PoolKind`).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DeviceComponentLabels {
    pub device: u32,
    pub component: &'static str,
}

/// `rank` label (decimal rank index of a tensor-parallel group).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RankLabel {
    pub rank: u32,
}

/// `replica` label (decimal index, CONFLICT C-5).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ReplicaLabel {
    pub replica: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DeviceLabel {
    pub device: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DevicePoolLabels {
    pub device: u32,
    pub pool: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct OutcomeLabel {
    pub outcome: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CircuitTransitionLabels {
    pub from: &'static str,
    pub to: &'static str,
    pub reason: &'static str,
}

/// Every P3 reliability family; cloning shares the underlying metrics.
#[derive(Clone, Debug)]
pub struct ReliabilityMetrics {
    pub pressure_state: Family<StateLabel, Gauge>,
    pub pressure_transitions: Family<TransitionLabels, Counter>,
    pub pressure_signal: Family<SignalLabel, FloatGauge>,
    pub pressure_signal_level: Family<SignalLabel, Gauge>,
    pub exhaustion_horizon_seconds: FloatGauge,
    pub admission_decisions: Family<DecisionLabels, Counter>,
    pub admission_queue_depth: Gauge,
    pub admission_queue_wait_seconds: Histogram,
    pub throttle_plan: Family<FieldLabel, FloatGauge>,
    pub reclaim_bytes: Family<ActionLabel, Counter>,
    pub memory_pool_bytes: Family<PoolLabels, Gauge>,
    pub emergency_reserve_held: Family<DeviceLabel, Gauge>,
    pub emergency_reserve_releases: Family<DeviceLabel, Counter>,
    pub allocation_failures: Family<DevicePoolLabels, Counter>,
    pub recoveries: Family<OutcomeLabel, Counter>,
    pub recovery_retries: Counter,
    pub circuit_state: Family<StateLabel, Gauge>,
    pub circuit_transitions: Family<CircuitTransitionLabels, Counter>,
    /// P5: per-device budget by component.
    pub device_budget_bytes: Family<DeviceComponentLabels, Gauge>,
    /// P5: TP group state per replica, 0 GREEN … 4 SURVIVAL.
    pub group_pressure_state: Family<ReplicaLabel, Gauge>,
    /// P5: the device index limiting each replica's group.
    pub group_limiting_device: Family<ReplicaLabel, Gauge>,
    /// P5 Task 33: a `static` worker rank's real ledger differed from the leader's mirror of it.
    pub ledger_mirror_divergence: Family<RankLabel, Counter>,
}

impl ReliabilityMetrics {
    /// Registers every reliability family on `reg`.
    pub fn register(reg: &MetricsRegistry) -> Self {
        let m = Self::unregistered();
        reg.register(
            "turbine_pressure_state",
            "1 for the current pressure state, 0 otherwise",
            m.pressure_state.clone(),
        );
        reg.register(
            "turbine_pressure_transitions",
            "Pressure state transitions by dominant signal",
            m.pressure_transitions.clone(),
        );
        reg.register(
            "turbine_pressure_signal",
            "Current value of each pressure signal",
            m.pressure_signal.clone(),
        );
        reg.register(
            "turbine_pressure_signal_level",
            "Level 0-4 (GREEN..SURVIVAL) of each pressure signal",
            m.pressure_signal_level.clone(),
        );
        reg.register(
            "turbine_pressure_exhaustion_horizon_seconds",
            "Predicted seconds until the KV pool is exhausted (+Inf when not growing)",
            m.exhaustion_horizon_seconds.clone(),
        );
        reg.register(
            "turbine_admission_decisions",
            "Admission decisions by decision and reason",
            m.admission_decisions.clone(),
        );
        reg.register(
            "turbine_admission_queue_depth",
            "Requests waiting in the admission queue",
            m.admission_queue_depth.clone(),
        );
        reg.register(
            "turbine_admission_queue_wait_seconds",
            "Time requests spent in the admission queue",
            m.admission_queue_wait_seconds.clone(),
        );
        reg.register(
            "turbine_throttle_plan",
            "Current throttle plan fields (batch growth -1 = unlimited)",
            m.throttle_plan.clone(),
        );
        reg.register(
            "turbine_reclaim_bytes",
            "Bytes reclaimed by action",
            m.reclaim_bytes.clone(),
        );
        reg.register(
            "turbine_memory_pool_bytes",
            "Memory pool capacity, used and reserved bytes",
            m.memory_pool_bytes.clone(),
        );
        reg.register(
            "turbine_emergency_reserve_held",
            "1 while the emergency reserve is held",
            m.emergency_reserve_held.clone(),
        );
        reg.register(
            "turbine_emergency_reserve_releases",
            "Emergency reserve releases",
            m.emergency_reserve_releases.clone(),
        );
        reg.register(
            "turbine_allocation_failures",
            "Pool allocation failures",
            m.allocation_failures.clone(),
        );
        reg.register(
            "turbine_recoveries",
            "OOM recovery outcomes",
            m.recoveries.clone(),
        );
        reg.register(
            "turbine_recovery_retries",
            "OOM recovery retries",
            m.recovery_retries.clone(),
        );
        reg.register(
            "turbine_circuit_state",
            "1 for the current circuit state, 0 otherwise",
            m.circuit_state.clone(),
        );
        reg.register(
            "turbine_circuit_transitions",
            "Circuit breaker transitions by reason",
            m.circuit_transitions.clone(),
        );
        reg.register(
            "turbine_device_budget_bytes",
            "Per-device memory budget by component",
            m.device_budget_bytes.clone(),
        );
        reg.register(
            "turbine_group_pressure_state",
            "Pressure state of each replica's TP group (0 GREEN .. 4 SURVIVAL): its worst member",
            m.group_pressure_state.clone(),
        );
        reg.register(
            "turbine_group_limiting_device",
            "Device index whose state sets each replica's group state",
            m.group_limiting_device.clone(),
        );
        reg.register(
            "turbine_ledger_mirror_divergence",
            "Static-mode worker rank ledgers found different from the leader's mirror of them",
            m.ledger_mirror_divergence.clone(),
        );
        m
    }

    /// Publishes `budget` as `turbine_device_budget_bytes{device,component}`, every component.
    pub fn record_device_budget(&self, budget: &crate::budget::DeviceBudget) {
        for kind in crate::budget::PoolKind::ALL {
            self.device_budget_bytes
                .get_or_create(&DeviceComponentLabels {
                    device: budget.device.0,
                    component: kind.as_str(),
                })
                .set(i64::try_from(budget.pool(kind)).unwrap_or(i64::MAX));
        }
    }

    /// Publishes replica `replica`'s group state and limiting device.
    pub fn record_group(&self, replica: u32, group: crate::multi_device::GroupState) {
        let label = ReplicaLabel { replica };
        self.group_pressure_state
            .get_or_create(&label)
            .set(i64::from(group.state.as_u8()));
        self.group_limiting_device
            .get_or_create(&label)
            .set(i64::from(group.limiting_device.0));
    }

    /// Families attached to no registry (components under unit test that read values back).
    pub fn unregistered() -> Self {
        ReliabilityMetrics {
            pressure_state: Family::default(),
            pressure_transitions: Family::default(),
            pressure_signal: Family::default(),
            pressure_signal_level: Family::default(),
            exhaustion_horizon_seconds: FloatGauge::default(),
            admission_decisions: Family::default(),
            admission_queue_depth: Gauge::default(),
            // 10 ms .. ~82 s.
            admission_queue_wait_seconds: Histogram::new(exponential_buckets(0.01, 2.0, 14)),
            throttle_plan: Family::default(),
            reclaim_bytes: Family::default(),
            memory_pool_bytes: Family::default(),
            emergency_reserve_held: Family::default(),
            emergency_reserve_releases: Family::default(),
            allocation_failures: Family::default(),
            recoveries: Family::default(),
            recovery_retries: Counter::default(),
            circuit_state: Family::default(),
            circuit_transitions: Family::default(),
            device_budget_bytes: Family::default(),
            group_pressure_state: Family::default(),
            group_limiting_device: Family::default(),
            ledger_mirror_divergence: Family::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_renders_with_counter_suffix() {
        let reg = MetricsRegistry::new();
        let m = ReliabilityMetrics::register(&reg);
        m.pressure_transitions
            .get_or_create(&TransitionLabels {
                from: "GREEN",
                to: "ORANGE",
                signal: "kv_utilization",
            })
            .inc();
        m.memory_pool_bytes
            .get_or_create(&PoolLabels {
                device: 0,
                pool: "kv",
                kind: "capacity",
            })
            .set(42);
        m.recovery_retries.inc();
        // Families render only once they hold a series.
        let state = StateLabel { state: "GREEN" };
        m.pressure_state.get_or_create(&state).set(1);
        m.circuit_state.get_or_create(&state).set(0);
        let signal = SignalLabel {
            signal: "kv_utilization",
        };
        m.pressure_signal.get_or_create(&signal).set(0.5);
        m.pressure_signal_level.get_or_create(&signal).set(0);
        m.admission_decisions
            .get_or_create(&DecisionLabels {
                decision: "admit",
                reason: "none",
            })
            .inc();
        m.throttle_plan
            .get_or_create(&FieldLabel {
                field: "batch_growth_limit",
            })
            .set(-1.0);
        m.reclaim_bytes
            .get_or_create(&ActionLabel {
                action: "free_cached",
            })
            .inc();
        m.emergency_reserve_held
            .get_or_create(&DeviceLabel { device: 0 })
            .set(1);
        m.emergency_reserve_releases
            .get_or_create(&DeviceLabel { device: 0 })
            .inc();
        m.allocation_failures
            .get_or_create(&DevicePoolLabels {
                device: 0,
                pool: "kv",
            })
            .inc();
        m.recoveries
            .get_or_create(&OutcomeLabel {
                outcome: "recovered",
            })
            .inc();
        m.circuit_transitions
            .get_or_create(&CircuitTransitionLabels {
                from: "HEALTHY",
                to: "DEGRADED",
                reason: "thermal_throttle",
            })
            .inc();
        let text = reg.render().unwrap();
        assert!(
            text.contains(
                "turbine_pressure_transitions_total{from=\"GREEN\",to=\"ORANGE\",signal=\"kv_utilization\"} 1"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "turbine_memory_pool_bytes{device=\"0\",pool=\"kv\",kind=\"capacity\"} 42"
            ),
            "{text}"
        );
        assert!(text.contains("turbine_recovery_retries_total 1"), "{text}");
        for family in [
            "turbine_pressure_state",
            "turbine_pressure_signal",
            "turbine_pressure_signal_level",
            "turbine_pressure_exhaustion_horizon_seconds",
            "turbine_admission_decisions",
            "turbine_admission_queue_depth",
            "turbine_admission_queue_wait_seconds",
            "turbine_throttle_plan",
            "turbine_reclaim_bytes",
            "turbine_emergency_reserve_held",
            "turbine_emergency_reserve_releases",
            "turbine_allocation_failures",
            "turbine_recoveries",
            "turbine_circuit_state",
            "turbine_circuit_transitions",
        ] {
            assert!(
                text.contains(&format!("# TYPE {family} ")),
                "{family} missing: {text}"
            );
        }
    }
}
