//! Request lifecycle (P2 S-2): states, legal transitions, cancellation and preemption reasons,
//! and the request record the engine submits.

use std::time::Duration;

use serde::Serialize;
use smallvec::SmallVec;
use turbine_core::request::{CancelFlag, ResourceEstimate};
use turbine_core::types::{Priority, RequestId, SeqId};

/// Lifecycle state of a sequence (and of a request, aggregated over its sequences).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestState {
    Waiting,
    Prefilling,
    Decoding,
    /// Decoding suspended because the client's output channel is full; keeps its KV.
    Paused,
    Finished,
    Cancelled,
    Failed,
}

impl RequestState {
    /// Waiting, prefilling, decoding or paused.
    pub fn is_live(self) -> bool {
        matches!(
            self,
            RequestState::Waiting
                | RequestState::Prefilling
                | RequestState::Decoding
                | RequestState::Paused
        )
    }

    /// The lifecycle table: waiting → prefilling → decoding ↔ paused; a running state →
    /// waiting on preemption; any live state → finished, cancelled or failed.
    pub fn can_transition(self, to: RequestState) -> bool {
        use RequestState::*;
        match (self, to) {
            (Waiting, Prefilling)
            | (Prefilling, Decoding)
            | (Decoding, Paused)
            | (Paused, Decoding) => true,
            (Prefilling | Decoding | Paused, Waiting) => true,
            (from, Finished | Cancelled | Failed) => from.is_live(),
            _ => false,
        }
    }

    /// Moves to `to` when the edge is legal; otherwise leaves the state unchanged.
    pub fn transition(&mut self, to: RequestState) -> Result<(), SchedError> {
        if self.can_transition(to) {
            *self = to;
            Ok(())
        } else {
            Err(SchedError::IllegalTransition { from: *self, to })
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RequestState::Waiting => "waiting",
            RequestState::Prefilling => "prefilling",
            RequestState::Decoding => "decoding",
            RequestState::Paused => "paused",
            RequestState::Finished => "finished",
            RequestState::Cancelled => "cancelled",
            RequestState::Failed => "failed",
        }
    }
}

/// Why a request was cancelled; `as_str` is the metric label and log `reason`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CancelReason {
    ClientDisconnect,
    RequestTimeout,
    SlowClient,
    Shutdown,
    /// Waited longer than the queue timeout without being admitted (Phase 2
    /// `scheduler.queue_timeout`, from Phase 3 `reliability.admission.queue_timeout`).
    QueueTimeout,
    /// Still queued when the circuit opened (P3): answered `503 circuit_open`.
    CircuitOpen,
    /// Admitted but not started when SURVIVAL requeued it, with the admission queue full
    /// (P3, `survival_liveness: requeue_unstarted`): answered `503 overloaded`.
    Overloaded,
}

impl CancelReason {
    pub fn as_str(self) -> &'static str {
        match self {
            CancelReason::ClientDisconnect => "client_disconnect",
            CancelReason::RequestTimeout => "request_timeout",
            CancelReason::SlowClient => "slow_client",
            CancelReason::Shutdown => "shutdown",
            CancelReason::QueueTimeout => "queue_timeout",
            CancelReason::CircuitOpen => "circuit_open",
            CancelReason::Overloaded => "overloaded",
        }
    }
}

/// Why a running request was preempted (P2 S-6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PreemptReason {
    /// The pool could not supply the blocks the next iteration needs.
    KvExhausted,
    /// SURVIVAL only (P3): the next decode step could not allocate, so the most recently
    /// admitted sequence is recomputed later.
    SurvivalDecodeAlloc,
}

impl PreemptReason {
    pub fn as_str(self) -> &'static str {
        match self {
            PreemptReason::KvExhausted => "kv_exhausted",
            PreemptReason::SurvivalDecodeAlloc => "survival_decode_alloc",
        }
    }
}

/// What the engine submits for one request.
#[derive(Clone, Debug)]
pub struct SchedRequest {
    pub id: RequestId,
    /// One sequence per choice (`n`); `seqs[0]` runs the shared prefill.
    pub seqs: SmallVec<[SeqId; 1]>,
    pub prompt_len: u32,
    pub max_new_tokens: u32,
    pub priority: Priority,
    pub estimate: ResourceEstimate,
    /// Submission time on the scheduler's clock.
    pub arrival: Duration,
    pub cancel: CancelFlag,
    /// Has a `response_format` or tool grammar (counted in the scheduler snapshot).
    pub constrained: bool,
}

impl SchedRequest {
    /// A request with default priority, no constraint, arrival 0 and its Phase 2 estimate
    /// (`ResourceEstimate::for_request`); callers set the other fields directly.
    pub fn new(
        id: RequestId,
        seqs: SmallVec<[SeqId; 1]>,
        prompt_len: u32,
        max_new_tokens: u32,
        block_tokens: u32,
    ) -> SchedRequest {
        SchedRequest {
            id,
            seqs,
            prompt_len,
            max_new_tokens,
            priority: Priority::default(),
            estimate: ResourceEstimate::for_request(prompt_len, max_new_tokens, block_tokens),
            arrival: Duration::ZERO,
            cancel: CancelFlag::default(),
            constrained: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SchedError {
    #[error("illegal transition {from:?} → {to:?}")]
    IllegalTransition {
        from: RequestState,
        to: RequestState,
    },
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    const ALL: [RequestState; 7] = [
        RequestState::Waiting,
        RequestState::Prefilling,
        RequestState::Decoding,
        RequestState::Paused,
        RequestState::Finished,
        RequestState::Cancelled,
        RequestState::Failed,
    ];

    #[test]
    fn lifecycle_transitions() {
        use RequestState::*;
        let allowed = [
            (Waiting, Prefilling),
            (Prefilling, Decoding),
            (Decoding, Paused),
            (Paused, Decoding),
            // Preemption by recompute returns a running request to the queue.
            (Prefilling, Waiting),
            (Decoding, Waiting),
            (Paused, Waiting),
        ];
        let live = [Waiting, Prefilling, Decoding, Paused];
        for from in ALL {
            for to in ALL {
                let expected = allowed.contains(&(from, to))
                    || (live.contains(&from) && matches!(to, Finished | Cancelled | Failed));
                assert_eq!(from.can_transition(to), expected, "{from:?} → {to:?}");
                let mut state = from;
                match state.transition(to) {
                    Ok(()) => {
                        assert!(expected, "{from:?} → {to:?} accepted");
                        assert_eq!(state, to);
                    }
                    Err(SchedError::IllegalTransition { from: f, to: t }) => {
                        assert!(!expected, "{from:?} → {to:?} rejected");
                        assert_eq!((f, t), (from, to));
                        assert_eq!(state, from, "a rejected transition changed the state");
                    }
                }
            }
        }
        // The named illegal edges of the acceptance criterion.
        assert!(!Finished.can_transition(Decoding));
        assert!(!Cancelled.can_transition(Waiting));
        assert!(!Failed.can_transition(Prefilling));
        assert_eq!(
            serde_json::to_value(Prefilling).unwrap(),
            serde_json::json!("prefilling")
        );

        // 100-token prompt, max_tokens 60, 16-token blocks → 10 KV blocks at completion.
        let r = SchedRequest::new(RequestId::new_v4(), smallvec![SeqId(1)], 100, 60, 16);
        assert_eq!(r.estimate.projected_kv_blocks, 10);
        assert_eq!(r.estimate.prompt_tokens, 100);
        assert_eq!(r.priority, Priority(0));
        assert!(!r.constrained);

        assert_eq!(CancelReason::ClientDisconnect.as_str(), "client_disconnect");
        assert_eq!(CancelReason::RequestTimeout.as_str(), "request_timeout");
        assert_eq!(CancelReason::SlowClient.as_str(), "slow_client");
        assert_eq!(CancelReason::Shutdown.as_str(), "shutdown");
        assert_eq!(CancelReason::QueueTimeout.as_str(), "queue_timeout");
        assert_eq!(CancelReason::CircuitOpen.as_str(), "circuit_open");
        assert_eq!(CancelReason::Overloaded.as_str(), "overloaded");
        assert_eq!(PreemptReason::KvExhausted.as_str(), "kv_exhausted");
        assert_eq!(
            PreemptReason::SurvivalDecodeAlloc.as_str(),
            "survival_decode_alloc"
        );
    }
}
