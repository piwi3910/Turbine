//! Throttle planner (P3 S-10): the TS §9 degradation ladder "reduce batch growth → throttle
//! prefill → demote KV → shrink chunks → queue requests → stop admission → drain/recover" as
//! one plan per pressure state, read by the scheduler once per iteration.

use serde::Serialize;
use turbine_core::config::SurvivalLiveness;
use turbine_core::types::PressureState;

use crate::metrics::{ActionLabel, FieldLabel, ReliabilityMetrics};
use crate::signals::SignalThresholds;

/// Which new requests admission lets through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionMode {
    Open,
    /// Requests whose new prefill exceeds `large_prefill_tokens` queue (`pressure_orange`).
    ExpensiveQueued,
    /// Every new request queues (`pressure_red`); queued requests only refill running slots
    /// freed by finished sequences.
    AllQueued,
    /// New requests are rejected `503 overloaded`; the queue is kept but not drained.
    Stopped,
}

impl AdmissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AdmissionMode::Open => "open",
            AdmissionMode::ExpensiveQueued => "expensive_queued",
            AdmissionMode::AllQueued => "all_queued",
            AdmissionMode::Stopped => "stopped",
        }
    }
}

/// The reclaim step of a state's plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReclaimAction {
    None,
    /// `KvReclaimer::demote` of idle blocks down to the `kv_utilization` YELLOW threshold.
    DemoteIdle,
    /// Free unreferenced cached blocks, then demote, down to the ORANGE threshold.
    FreeCachedToOrange,
    /// Free every unreferenced cached block and the optional buffers.
    FreeAllCachedAndOptional,
    /// The recovery controller releases the emergency reserve; preempt (recompute) the most
    /// recently admitted sequence only if the next decode step cannot allocate.
    ReleaseReserveAndPreemptIfNeeded,
}

/// What the scheduler may do this iteration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ThrottlePlan {
    pub state: PressureState,
    /// `None` = unlimited; `Some(0)` = frozen at the current running count.
    pub batch_growth_limit: Option<u32>,
    /// Finished sequences are not replaced (SURVIVAL).
    pub shrink_only: bool,
    pub prefill_budget_fraction: f64,
    /// `None` = no prefill at all (SURVIVAL).
    pub prefill_chunk_tokens: Option<u32>,
    /// False in SURVIVAL: no prefill starts. In RED new prefills start only in slots freed by
    /// finished sequences (batch growth 0).
    pub start_new_prefills: bool,
    pub admission: AdmissionMode,
    pub reclaim: ReclaimAction,
    /// SURVIVAL under `survival_liveness: requeue_unstarted` (option A): admitted requests that
    /// have not started go back to the admission queue and drop their KV reservations.
    pub requeue_unstarted: bool,
}

/// The scheduler configuration the plan scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerLimits {
    pub prefill_chunk_tokens: u32,
    pub block_tokens: u32,
}

/// The P3 "Throttle plan per state" table. The chunk floor is 4 × block tokens (never above
/// the configured chunk).
pub fn plan_for(state: PressureState, cfg: &SchedulerLimits) -> ThrottlePlan {
    let configured = cfg.prefill_chunk_tokens;
    let floor = cfg.block_tokens.saturating_mul(4).min(configured);
    let open = ThrottlePlan {
        state,
        batch_growth_limit: None,
        shrink_only: false,
        prefill_budget_fraction: 1.0,
        prefill_chunk_tokens: Some(configured),
        start_new_prefills: true,
        admission: AdmissionMode::Open,
        reclaim: ReclaimAction::None,
        requeue_unstarted: false,
    };
    match state {
        PressureState::Green => open,
        PressureState::Yellow => ThrottlePlan {
            batch_growth_limit: Some(1),
            reclaim: ReclaimAction::DemoteIdle,
            ..open
        },
        PressureState::Orange => ThrottlePlan {
            batch_growth_limit: Some(0),
            prefill_budget_fraction: 0.5,
            prefill_chunk_tokens: Some((configured / 2).max(floor)),
            admission: AdmissionMode::ExpensiveQueued,
            reclaim: ReclaimAction::FreeCachedToOrange,
            ..open
        },
        // RED refills finished slots from the queue (user decision 2026-09-26): the running
        // count never grows, and every refill holds its worst-case KV reservation.
        PressureState::Red => ThrottlePlan {
            batch_growth_limit: Some(0),
            prefill_budget_fraction: 0.5,
            prefill_chunk_tokens: Some(floor),
            admission: AdmissionMode::AllQueued,
            reclaim: ReclaimAction::FreeAllCachedAndOptional,
            ..open
        },
        PressureState::Survival => ThrottlePlan {
            batch_growth_limit: Some(0),
            shrink_only: true,
            prefill_budget_fraction: 0.0,
            prefill_chunk_tokens: None,
            start_new_prefills: false,
            admission: AdmissionMode::Stopped,
            reclaim: ReclaimAction::ReleaseReserveAndPreemptIfNeeded,
            requeue_unstarted: true,
            ..open
        },
    }
}

/// [`plan_for`] with the SURVIVAL row of `survival` (`reliability.recovery.survival_liveness`,
/// [`crate::recovery::survival_plan`]); every other state is the table's.
pub fn plan_with(
    state: PressureState,
    cfg: &SchedulerLimits,
    survival: SurvivalLiveness,
) -> ThrottlePlan {
    crate::recovery::survival_plan(plan_for(state, cfg), cfg, survival)
}

/// Reclaim hook. Phase 3 frees unreferenced cached GPU blocks; phase 4 implements demotion.
pub trait KvReclaimer: Send + Sync {
    /// Demote idle blocks to lower tiers until KV utilisation ≤ `target_utilization`;
    /// returns the bytes demoted.
    fn demote(&self, target_utilization: f64) -> u64;
    /// Free unreferenced cached blocks until KV utilisation ≤ `target_utilization`;
    /// returns the bytes freed.
    fn free_unreferenced(&self, target_utilization: f64) -> u64;
    /// Free optional buffers; returns the bytes freed.
    fn free_optional(&self) -> u64 {
        0
    }
}

/// Run the plan's reclaim step with the `kv_utilization` thresholds as targets (a step whose
/// threshold is unset by an override is skipped). Each non-zero action adds to
/// `turbine_reclaim_bytes_total{action}` and logs `reclaim` (INFO). SURVIVAL's reserve release
/// is the recovery controller's, not done here.
pub fn apply_reclaim(
    plan: &ThrottlePlan,
    reclaimer: &dyn KvReclaimer,
    kv: &SignalThresholds,
    metrics: &ReliabilityMetrics,
) -> Vec<(&'static str, u64)> {
    let yellow = kv.threshold(PressureState::Yellow);
    let orange = kv.threshold(PressureState::Orange);
    let mut done = Vec::new();
    match plan.reclaim {
        ReclaimAction::None | ReclaimAction::ReleaseReserveAndPreemptIfNeeded => {}
        ReclaimAction::DemoteIdle => {
            if let Some(target) = yellow {
                done.push(("demote", reclaimer.demote(target)));
            }
        }
        ReclaimAction::FreeCachedToOrange => {
            if let Some(target) = orange {
                done.push(("free_cached", reclaimer.free_unreferenced(target)));
                done.push(("demote", reclaimer.demote(target)));
            }
        }
        ReclaimAction::FreeAllCachedAndOptional => {
            done.push(("free_cached", reclaimer.free_unreferenced(0.0)));
            done.push(("free_optional", reclaimer.free_optional()));
        }
    }
    for &(action, bytes) in &done {
        if bytes > 0 {
            metrics
                .reclaim_bytes
                .get_or_create(&ActionLabel { action })
                .inc_by(bytes);
            tracing::info!(
                event = "reclaim",
                reason = action,
                bytes,
                state = plan.state.as_str(),
            );
        }
    }
    done
}

/// Publish a new plan: `turbine_throttle_plan{field}` gauges (unlimited batch growth as -1,
/// no prefill chunk as 0) and a `throttle_plan_changed` (INFO) event. Call on change.
pub fn publish_plan(plan: &ThrottlePlan, metrics: &ReliabilityMetrics) {
    let growth = plan.batch_growth_limit.map_or(-1.0, f64::from);
    let chunk = plan.prefill_chunk_tokens.map_or(0.0, f64::from);
    for (field, v) in [
        ("batch_growth_limit", growth),
        ("prefill_budget_fraction", plan.prefill_budget_fraction),
        ("prefill_chunk_tokens", chunk),
    ] {
        metrics
            .throttle_plan
            .get_or_create(&FieldLabel { field })
            .set(v);
    }
    tracing::info!(
        event = "throttle_plan_changed",
        reason = plan.state.as_str(),
        batch_growth_limit = growth,
        shrink_only = plan.shrink_only,
        prefill_budget_fraction = plan.prefill_budget_fraction,
        prefill_chunk_tokens = chunk,
        start_new_prefills = plan.start_new_prefills,
        admission = plan.admission.as_str(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::{PressureSignal, default_thresholds};
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recording(Mutex<Vec<(&'static str, f64)>>);
    impl KvReclaimer for Recording {
        fn demote(&self, target: f64) -> u64 {
            self.0.lock().unwrap().push(("demote", target));
            0
        }
        fn free_unreferenced(&self, target: f64) -> u64 {
            self.0.lock().unwrap().push(("free_unreferenced", target));
            4096
        }
        fn free_optional(&self) -> u64 {
            self.0.lock().unwrap().push(("free_optional", 0.0));
            0
        }
    }

    #[test]
    fn plan_per_state() {
        use AdmissionMode::*;
        let limits = SchedulerLimits {
            prefill_chunk_tokens: 2048,
            block_tokens: 16,
        };
        let p = |s| plan_for(s, &limits);
        let row = |pl: ThrottlePlan| {
            (
                pl.batch_growth_limit,
                pl.shrink_only,
                pl.prefill_budget_fraction,
                pl.prefill_chunk_tokens,
                pl.start_new_prefills,
                pl.admission,
                pl.reclaim,
            )
        };
        assert_eq!(
            row(p(PressureState::Green)),
            (
                None,
                false,
                1.0,
                Some(2048),
                true,
                Open,
                ReclaimAction::None
            )
        );
        assert_eq!(
            row(p(PressureState::Yellow)),
            (
                Some(1),
                false,
                1.0,
                Some(2048),
                true,
                Open,
                ReclaimAction::DemoteIdle
            )
        );
        assert_eq!(
            row(p(PressureState::Orange)),
            (
                Some(0),
                false,
                0.5,
                Some(1024),
                true,
                ExpensiveQueued,
                ReclaimAction::FreeCachedToOrange
            )
        );
        assert_eq!(
            row(p(PressureState::Red)),
            (
                Some(0),
                false,
                0.5,
                Some(64),
                true,
                AllQueued,
                ReclaimAction::FreeAllCachedAndOptional
            )
        );
        assert_eq!(
            row(p(PressureState::Survival)),
            (
                Some(0),
                true,
                0.0,
                None,
                false,
                Stopped,
                ReclaimAction::ReleaseReserveAndPreemptIfNeeded
            )
        );
        // Chunk floor: 4 × block tokens even when halving goes below it.
        let small = SchedulerLimits {
            prefill_chunk_tokens: 96,
            block_tokens: 16,
        };
        assert_eq!(
            plan_for(PressureState::Orange, &small).prefill_chunk_tokens,
            Some(64)
        );
        // KvReclaimer calls per state.
        let kv = default_thresholds()[&PressureSignal::KvUtilization];
        let metrics = ReliabilityMetrics::unregistered();
        let calls = |s| {
            let r = Recording::default();
            apply_reclaim(&p(s), &r, &kv, &metrics);
            r.0.into_inner().unwrap()
        };
        assert!(calls(PressureState::Green).is_empty());
        assert_eq!(calls(PressureState::Yellow), vec![("demote", 0.70)]);
        assert_eq!(
            calls(PressureState::Orange),
            vec![("free_unreferenced", 0.82), ("demote", 0.82)]
        );
        assert_eq!(
            calls(PressureState::Red),
            vec![("free_unreferenced", 0.0), ("free_optional", 0.0)]
        );
        assert!(
            calls(PressureState::Survival).is_empty(),
            "SURVIVAL reclaim is the recovery controller's reserve release"
        );
        assert_eq!(
            metrics
                .reclaim_bytes
                .get_or_create(&ActionLabel {
                    action: "free_cached"
                })
                .get(),
            2 * 4096
        );

        // Plan gauges: unlimited growth is -1, no chunk (SURVIVAL) is 0.
        let field = |field| {
            metrics
                .throttle_plan
                .get_or_create(&FieldLabel { field })
                .get()
        };
        publish_plan(&p(PressureState::Green), &metrics);
        assert_eq!(field("batch_growth_limit"), -1.0);
        assert_eq!(field("prefill_chunk_tokens"), 2048.0);
        publish_plan(&p(PressureState::Survival), &metrics);
        assert_eq!(
            (
                field("batch_growth_limit"),
                field("prefill_budget_fraction"),
                field("prefill_chunk_tokens")
            ),
            (0.0, 0.0, 0.0)
        );
        // A configured chunk below the floor is never raised.
        let tiny = SchedulerLimits {
            prefill_chunk_tokens: 32,
            block_tokens: 16,
        };
        assert_eq!(
            plan_for(PressureState::Red, &tiny).prefill_chunk_tokens,
            Some(32)
        );
    }
}
