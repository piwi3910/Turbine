//! Body of `GET /turbine/v1/pressure` (P3 §Data, keys verbatim).
use crate::admission::{PressureReason, RejectionReason};
use crate::budget::{DeviceBudget, PoolKind};
use crate::circuit::CircuitReason;
use crate::ledger::Ledger;
use crate::metrics::{DecisionLabels, ReliabilityMetrics};
use crate::signals::{PressureSignal, SignalValue};
use crate::state::Transition;
use crate::throttle::{AdmissionMode, ThrottlePlan};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};
use turbine_core::types::{CircuitState, DeviceId, MemoryKind, PressureState};

use crate::multi_device::GroupState;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PressureDocument {
    pub enabled: bool,
    pub state: PressureState,
    pub since: String,
    pub dominant_signal: Option<PressureSignal>,
    /// `null` when not growing (+∞ has no JSON form; the metric carries +Inf).
    pub exhaustion_horizon_seconds: Option<f64>,
    pub signals: Vec<SignalValue>,
    pub throttle: ThrottleDoc,
    pub memory: Vec<MemoryDoc>,
    pub admission: AdmissionDoc,
    pub circuit: CircuitDoc,
    pub transitions: Vec<TransitionDoc>,
    /// P5 S-8: per-device budget by component and pressure state.
    pub devices: Vec<DeviceDoc>,
    /// P5 S-8: per TP group (one per replica) the worst member state and its device.
    pub groups: Vec<GroupDoc>,
    /// P5 S-7: whether the data-parallel router may pick each replica, and why not.
    pub replicas: Vec<ReplicaDoc>,
}

/// `devices[]` entry: a device's state and its budget by component (pool).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DeviceDoc {
    pub device: u32,
    pub state: PressureState,
    /// Component → bytes, every [`PoolKind`] present (0 when the budget has no such pool).
    pub budget: BTreeMap<&'static str, u64>,
}

/// The `devices[]` entry of `budget` in `state`.
pub fn device_doc(budget: &DeviceBudget, state: PressureState) -> DeviceDoc {
    DeviceDoc {
        device: budget.device.0,
        state,
        budget: PoolKind::ALL
            .iter()
            .map(|&k| (k.as_str(), budget.pool(k)))
            .collect(),
    }
}

/// `groups[]` entry: a replica's TP group state (worst member) and the device limiting it.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct GroupDoc {
    pub replica: u32,
    pub state: PressureState,
    pub limiting_device: u32,
}

impl GroupDoc {
    /// The group of `replica` whose members are (device, state).
    pub fn of(replica: u32, members: &[(DeviceId, PressureState)]) -> GroupDoc {
        let g = GroupState::of(members);
        GroupDoc {
            replica,
            state: g.state,
            limiting_device: g.limiting_device.0,
        }
    }
}

/// `replicas[]` entry: router eligibility (below ORANGE, circuit neither open nor draining).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ReplicaDoc {
    pub replica: u32,
    pub eligible: bool,
    /// `eligible`, `pressure` (ORANGE or worse) or `circuit` (CIRCUIT_OPEN / DRAINING).
    pub reason: &'static str,
}

impl ReplicaDoc {
    pub fn of(replica: u32, state: PressureState, circuit: CircuitState) -> ReplicaDoc {
        let reason = if matches!(circuit, CircuitState::CircuitOpen | CircuitState::Draining) {
            "circuit"
        } else if state >= PressureState::Orange {
            "pressure"
        } else {
            "eligible"
        };
        ReplicaDoc {
            replica,
            eligible: reason == "eligible",
            reason,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ThrottleDoc {
    pub batch_growth_limit: Option<u32>,
    pub prefill_budget_fraction: f64,
    pub prefill_chunk_tokens: Option<u32>,
    pub admission: AdmissionMode,
}
impl From<&ThrottlePlan> for ThrottleDoc {
    fn from(p: &ThrottlePlan) -> Self {
        Self {
            batch_growth_limit: p.batch_growth_limit,
            prefill_budget_fraction: p.prefill_budget_fraction,
            prefill_chunk_tokens: p.prefill_chunk_tokens,
            admission: p.admission,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MemoryDoc {
    pub device: u32,
    pub memory_kind: MemoryKind,
    pub budget_bytes: u64,
    pub pools: Vec<PoolDoc>,
    pub emergency_reserve_held: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PoolDoc {
    pub name: PoolKind,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub reserved_bytes: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Default)]
pub struct AdmissionDoc {
    pub queued: u32,
    pub max_queue: u32,
    pub decisions: DecisionsDoc,
}

#[derive(Clone, Debug, Serialize, PartialEq, Default)]
pub struct DecisionsDoc {
    pub admit: u64,
    pub queue: BTreeMap<&'static str, u64>,
    pub reject: BTreeMap<&'static str, u64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CircuitDoc {
    pub state: CircuitState,
    pub since: String,
    pub last_reason: Option<CircuitReason>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TransitionDoc {
    pub at: String,
    pub from: PressureState,
    pub to: PressureState,
    pub signal: PressureSignal,
    pub value: f64,
    pub threshold: f64,
}
impl From<&Transition> for TransitionDoc {
    fn from(t: &Transition) -> Self {
        Self {
            at: rfc3339_millis(t.at_wall),
            from: t.from,
            to: t.to,
            signal: t.signal,
            value: t.value,
            threshold: t.threshold,
        }
    }
}

pub fn memory_doc(budget: &DeviceBudget, ledger: &Ledger, reserve_held: bool) -> MemoryDoc {
    MemoryDoc {
        device: budget.device.0,
        memory_kind: budget.memory_kind,
        budget_bytes: budget.budget_bytes,
        pools: PoolKind::ALL
            .iter()
            .map(|&name| {
                let u = ledger.usage(budget.device, name);
                PoolDoc {
                    name,
                    capacity_bytes: u.capacity,
                    used_bytes: u.used,
                    reserved_bytes: u.reserved,
                }
            })
            .collect(),
        emergency_reserve_held: reserve_held,
    }
}

/// Decision counts read back from `turbine_admission_decisions_total` (non-zero reasons only).
pub fn decisions_doc(metrics: &ReliabilityMetrics) -> DecisionsDoc {
    let get = |decision, reason| {
        metrics
            .admission_decisions
            .get_or_create(&DecisionLabels { decision, reason })
            .get()
    };
    let mut doc = DecisionsDoc {
        admit: get("admit", "none"),
        ..DecisionsDoc::default()
    };
    for r in PressureReason::ALL {
        let n = get("queue", r.as_str());
        if n > 0 {
            doc.queue.insert(r.as_str(), n);
        }
    }
    for r in RejectionReason::ALL {
        let n = get("reject", r.as_str());
        if n > 0 {
            doc.reject.insert(r.as_str(), n);
        }
    }
    doc
}

/// `2026-09-25T18:02:11.482Z` without a date-time dependency (civil-from-days, UTC).
pub fn rfc3339_millis(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn rfc3339_formats_utc_millis() {
        let t = UNIX_EPOCH + Duration::from_millis(1_790_359_331_482);
        assert_eq!(rfc3339_millis(t), "2026-09-25T18:02:11.482Z");
        assert_eq!(rfc3339_millis(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            rfc3339_millis(UNIX_EPOCH + Duration::from_secs(951_782_400)),
            "2000-02-29T00:00:00.000Z"
        );
    }
}
