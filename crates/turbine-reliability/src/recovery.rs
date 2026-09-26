//! Bounded recovery from device out-of-memory (P3 S-11): shrink the failed batch and retry with
//! exponential backoff up to `max_retries`, then give up (the batch's requests fail `resource_exhausted`).
use crate::metrics::{OutcomeLabel, ReliabilityMetrics};
use std::time::Duration;
use turbine_core::config::{RecoveryConfig, SurvivalLiveness};
use turbine_core::types::PressureState;

use crate::throttle::{SchedulerLimits, ThrottlePlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryOutcome {
    Recovered { retries: u32 },
    Failed,
}
impl RecoveryOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            RecoveryOutcome::Recovered { .. } => "recovered",
            RecoveryOutcome::Failed => "failed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryStep {
    /// Retry the iteration with at most `batch_limit` sequences after sleeping `backoff`.
    Retry {
        attempt: u32,
        backoff: Duration,
        batch_limit: usize,
    },
    /// Retries exhausted: fail the batch's requests with `resource_exhausted`; the worker keeps running.
    GiveUp,
}

pub struct RecoveryController {
    max_retries: u32,
    backoff: Duration,
    attempt: u32,
    metrics: ReliabilityMetrics,
}

impl RecoveryController {
    pub fn new(cfg: &RecoveryConfig, metrics: ReliabilityMetrics) -> Self {
        Self {
            max_retries: cfg.max_retries,
            backoff: cfg.backoff.0,
            attempt: 0,
            metrics,
        }
    }

    pub fn in_recovery(&self) -> bool {
        self.attempt > 0
    }

    /// Called after an iteration attempt failed with a device OOM; `batch_len` = sequences in that attempt.
    pub fn on_oom(&mut self, batch_len: usize) -> RecoveryStep {
        if self.attempt >= self.max_retries {
            let retries = self.attempt;
            self.attempt = 0;
            self.finish(RecoveryOutcome::Failed, retries);
            return RecoveryStep::GiveUp;
        }
        self.attempt += 1;
        self.metrics.recovery_retries.inc();
        let backoff = self.backoff * 2u32.saturating_pow(self.attempt - 1);
        let batch_limit = (batch_len / 2).max(1);
        tracing::warn!(
            event = "recovery_attempt",
            reason = "device_oom",
            attempt = self.attempt,
            backoff_ms = backoff.as_millis() as u64,
            batch_limit,
            batch_len
        );
        RecoveryStep::Retry {
            attempt: self.attempt,
            backoff,
            batch_limit,
        }
    }

    /// Called after an iteration attempt succeeded; `Some` when it ended a recovery.
    pub fn on_success(&mut self) -> Option<RecoveryOutcome> {
        if self.attempt == 0 {
            return None;
        }
        let retries = self.attempt;
        self.attempt = 0;
        let outcome = RecoveryOutcome::Recovered { retries };
        self.finish(outcome, retries);
        Some(outcome)
    }

    fn finish(&self, outcome: RecoveryOutcome, retries: u32) {
        self.metrics
            .recoveries
            .get_or_create(&OutcomeLabel {
                outcome: outcome.as_str(),
            })
            .inc();
        tracing::warn!(
            event = "recovery_outcome",
            reason = outcome.as_str(),
            retries
        );
    }
}

/// The SURVIVAL row of the throttle plan under `survival_liveness` (decision "Phase 3:
/// SURVIVAL liveness fix", provisional A pending user review). `plan` is [`plan_for`]'s; other
/// states pass through unchanged.
///
/// - `requeue_unstarted` (A, default): the table's row — no prefill at all — with
///   `requeue_unstarted` set, so the scheduler returns admitted requests that have not started
///   to the admission queue and drops their KV reservations.
/// - `continue_prefills` (B): prefills already in progress continue with RED's budget (half the
///   batch) and chunk floor; nothing new starts, nothing is requeued.
///
/// [`plan_for`]: crate::throttle::plan_for
pub fn survival_plan(
    plan: ThrottlePlan,
    cfg: &SchedulerLimits,
    survival: SurvivalLiveness,
) -> ThrottlePlan {
    if plan.state != PressureState::Survival {
        return plan;
    }
    match survival {
        SurvivalLiveness::RequeueUnstarted => ThrottlePlan {
            requeue_unstarted: true,
            ..plan
        },
        SurvivalLiveness::ContinuePrefills => ThrottlePlan {
            prefill_budget_fraction: 0.5,
            prefill_chunk_tokens: Some(
                cfg.block_tokens
                    .saturating_mul(4)
                    .min(cfg.prefill_chunk_tokens),
            ),
            start_new_prefills: false,
            requeue_unstarted: false,
            ..plan
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decision "Phase 3: SURVIVAL liveness fix" (provisional A, switchable to B): A keeps the
    /// table's SURVIVAL row (no prefill) and requeues admitted requests that have not started;
    /// B lets prefills in progress continue at RED's budget and chunk floor and requeues
    /// nothing; no other state changes. Breaks if the switch does not reach the plan.
    #[test]
    fn survival_plan_switch() {
        use crate::throttle::{plan_for, plan_with};
        let limits = SchedulerLimits {
            prefill_chunk_tokens: 2048,
            block_tokens: 16,
        };
        let a = plan_with(
            PressureState::Survival,
            &limits,
            SurvivalLiveness::RequeueUnstarted,
        );
        assert_eq!(a, plan_for(PressureState::Survival, &limits));
        assert!(a.requeue_unstarted && a.shrink_only && !a.start_new_prefills);
        assert_eq!(
            (a.prefill_budget_fraction, a.prefill_chunk_tokens),
            (0.0, None)
        );
        let b = plan_with(
            PressureState::Survival,
            &limits,
            SurvivalLiveness::ContinuePrefills,
        );
        assert!(!b.requeue_unstarted && b.shrink_only && !b.start_new_prefills);
        assert_eq!(
            (b.prefill_budget_fraction, b.prefill_chunk_tokens),
            (0.5, Some(64))
        );
        assert_eq!(b.admission, a.admission);
        for state in [
            PressureState::Green,
            PressureState::Yellow,
            PressureState::Orange,
            PressureState::Red,
        ] {
            for mode in [
                SurvivalLiveness::RequeueUnstarted,
                SurvivalLiveness::ContinuePrefills,
            ] {
                let p = plan_with(state, &limits, mode);
                assert_eq!(p, plan_for(state, &limits), "{state:?}");
                assert!(!p.requeue_unstarted);
            }
        }
    }

    #[test]
    fn retries_are_bounded_and_back_off() {
        let metrics = ReliabilityMetrics::unregistered();
        let mut r = RecoveryController::new(&RecoveryConfig::default(), metrics.clone());
        let ms = Duration::from_millis;
        assert_eq!(
            r.on_oom(8),
            RecoveryStep::Retry {
                attempt: 1,
                backoff: ms(50),
                batch_limit: 4
            }
        );
        assert_eq!(
            r.on_oom(4),
            RecoveryStep::Retry {
                attempt: 2,
                backoff: ms(100),
                batch_limit: 2
            }
        );
        assert_eq!(
            r.on_success(),
            Some(RecoveryOutcome::Recovered { retries: 2 })
        );
        assert_eq!(r.on_success(), None);
        for (i, len) in [8usize, 4, 2].into_iter().enumerate() {
            assert!(
                matches!(r.on_oom(len), RecoveryStep::Retry { attempt, .. } if attempt == i as u32 + 1)
            );
        }
        assert_eq!(r.on_oom(1), RecoveryStep::GiveUp);
        assert!(
            !r.in_recovery(),
            "a new failure starts a new bounded recovery"
        );
        assert_eq!(metrics.recovery_retries.get(), 5);
        assert_eq!(
            metrics
                .recoveries
                .get_or_create(&OutcomeLabel { outcome: "failed" })
                .get(),
            1
        );
        let mut none = RecoveryController::new(
            &RecoveryConfig {
                max_retries: 0,
                ..RecoveryConfig::default()
            },
            metrics,
        );
        assert_eq!(
            none.on_oom(8),
            RecoveryStep::GiveUp,
            "max_retries: 0 gives up at once"
        );
    }
}
