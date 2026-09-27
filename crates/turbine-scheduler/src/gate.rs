//! The admission gate (P3 S-9): predictive admission in front of the scheduler, owning the
//! single waiting queue from Phase 3 on (CONFLICT C-1). Every admitted request carries its
//! worst-case KV reservation; queued requests hold none. The queue is ordered by the
//! scheduling policy's admission key (Phase 2m `scheduling_policy`), which the scheduler
//! computes at submission; the gate adds the mechanism around it (decisions, reservations, the
//! `max_bypass` starvation guard, the `max_queue` and `queue_timeout` bounds).

use std::collections::HashMap;
use std::sync::Arc;

use turbine_core::clock::Clock;
use turbine_core::request::ResourceEstimate;
use turbine_core::types::RequestId;
use turbine_reliability::admission::{
    Admission, AdmissionDecision, AdmissionQueue, PressureReason, Queued, RejectionReason,
};
use turbine_reliability::controller::ControllerHandle;
use turbine_reliability::ledger::{LedgerError, Reservation};
use turbine_reliability::metrics::DecisionLabels;

use crate::policy::AdmissionKey;
use crate::request::SchedRequest;
use crate::scheduler::SubmitError;

/// Result of offering a request to the gate.
#[derive(Debug)]
pub enum GateOutcome {
    /// Admitted with its worst-case KV reservation; the scheduler may start it.
    Admitted(SchedRequest, Reservation),
    /// Waiting in the admission queue.
    Queued,
}

/// A request waiting in the gate: the request and its submission number (the scheduling
/// policy's `AdmissionInfo::submit_no`, kept so the admitted request keeps its order).
pub struct Waiting {
    pub req: SchedRequest,
    pub submit_no: u64,
}

/// The gate's queue: ordered by the scheduling policy's admission key.
pub type GateQueue = AdmissionQueue<Waiting, AdmissionKey>;

/// An admitted request leaving the gate's queue: the request, its KV reservation and its
/// submission number.
pub type Pumped = (SchedRequest, Reservation, u64);

/// Admission decisions against the controller's current snapshot, the KV reservation of
/// admitted requests and the bounded admission queue.
pub struct AdmissionGate {
    admission: Admission,
    queue: GateQueue,
    controller: ControllerHandle,
    clock: Arc<dyn Clock>,
    max_running: u32,
}

impl AdmissionGate {
    pub fn new(
        admission: Admission,
        queue: GateQueue,
        controller: ControllerHandle,
        clock: Arc<dyn Clock>,
        max_running: u32,
    ) -> AdmissionGate {
        let gate = AdmissionGate {
            admission,
            queue,
            controller,
            clock,
            max_running,
        };
        gate.publish_depth();
        gate
    }

    /// Decide `r` now. `Reject` → `SubmitError::Rejected`; `Admit` reserves the KV and admits
    /// only while the queue is empty (queued requests keep their turn) and `slot_free` (the
    /// scheduler has an admitted slot within `max_running_requests` and the throttle plan's
    /// batch growth); otherwise the request waits (`QueueFull` when the queue is at
    /// `max_queue`). `key` is the policy's admission key for `submit_no`.
    pub fn offer(
        &mut self,
        mut r: SchedRequest,
        key: AdmissionKey,
        submit_no: u64,
        slot_free: bool,
    ) -> Result<GateOutcome, SubmitError> {
        r.estimate = self.admission.with_timing(r.estimate);
        let snap = self.controller.snapshot();
        let decision =
            self.admission
                .decide(&r.estimate, snap.state, snap.circuit, self.queue.len());
        let reason = match decision {
            AdmissionDecision::Reject { reason } => {
                return Err(SubmitError::Rejected {
                    reason,
                    retry_after_secs: self.retry_after_secs(reason),
                });
            }
            AdmissionDecision::Queue { reason } => reason,
            AdmissionDecision::Admit if slot_free && self.queue.is_empty() => {
                match self.admission.reserve_kv(&r.estimate) {
                    Ok(reservation) => {
                        self.started(&r.estimate);
                        return Ok(GateOutcome::Admitted(r, reservation));
                    }
                    // Only an injected allocation failure (or a racing reservation) gets here.
                    Err(LedgerError::Exhausted { .. } | LedgerError::Injected) => {
                        PressureReason::KvReservation
                    }
                }
            }
            // Admissible, but earlier requests wait (FIFO order within the priority) or every
            // admitted slot is taken: the pump admits it into a freed slot.
            AdmissionDecision::Admit => PressureReason::KvReservation,
        };
        let now = self.clock.now_mono();
        let (id, priority, estimate) = (r.id, r.priority, r.estimate);
        match self.queue.push_keyed(
            id,
            priority,
            key,
            estimate,
            reason,
            now,
            Waiting { req: r, submit_no },
        ) {
            Ok(()) => {
                self.publish_depth();
                Ok(GateOutcome::Queued)
            }
            Err(_) => Err(SubmitError::Rejected {
                reason: RejectionReason::QueueFull,
                retry_after_secs: self.retry_after_secs(RejectionReason::QueueFull),
            }),
        }
    }

    /// Queued requests older than `queue_timeout` (answered `queue_timeout`).
    pub fn expire(&mut self) -> Vec<SchedRequest> {
        let expired = self.queue.expire(self.clock.now_mono());
        self.finish_waits(expired, RejectionReason::QueueTimeout)
    }

    /// Admit up to `max_new` queued requests in queue order (with the `max_bypass` starvation
    /// guard), each re-decided against the current snapshot with `Admission::evaluate_refill`
    /// (in RED queued requests refill finished slots) and given its KV reservation. `idle`:
    /// nothing is admitted, so the work-conserving floor (`Admission::evaluate_idle`) decides.
    pub fn pump(&mut self, max_new: usize, idle: bool) -> Vec<Pumped> {
        if max_new == 0 || self.queue.is_empty() {
            return Vec::new();
        }
        let snap = self.controller.snapshot();
        let admission = &self.admission;
        let mut reservations: HashMap<RequestId, Reservation> = HashMap::new();
        let admitted = self.queue.pump(|q| {
            let decision = if idle {
                admission.evaluate_idle(&q.estimate, snap.state, snap.circuit)
            } else {
                admission.evaluate_refill(&q.estimate, snap.state, snap.circuit)
            };
            if reservations.len() >= max_new || decision != AdmissionDecision::Admit {
                return false;
            }
            match admission.reserve_kv(&q.estimate) {
                Ok(r) => {
                    reservations.insert(q.id, r);
                    true
                }
                Err(_) => false,
            }
        });
        let now = self.clock.now_mono();
        let metrics = self.admission.metrics().clone();
        let mut out = Vec::with_capacity(admitted.len());
        for q in admitted {
            metrics
                .admission_queue_wait_seconds
                .observe(now.saturating_sub(q.enqueued_at).as_secs_f64());
            let reservation = reservations
                .remove(&q.id)
                .expect("every admitted entry took a reservation");
            if idle {
                tracing::info!(
                    event = "admission_decision",
                    request_id = %q.id.0,
                    decision = "admit",
                    reason = "idle_floor",
                    state = snap.state.as_str(),
                    "nothing admitted: the queue head is admitted despite pressure"
                );
            }
            self.started(&q.estimate);
            out.push((q.payload.req, reservation, q.payload.submit_no));
        }
        self.publish_depth();
        out
    }

    /// SURVIVAL, option A: an admitted request that had not started comes back without its
    /// KV reservation (the caller dropped it), at the policy's `key` for its original
    /// `submit_no` and with its original arrival as the start of its queue wait — so it keeps
    /// its turn and its `queue_timeout` bound. Recorded as a `queue` decision with reason
    /// `survival_requeue`. Returns false when the queue is full: the request is then recorded
    /// as rejected `survival` and the caller answers it `overloaded`.
    pub fn requeue(&mut self, r: SchedRequest, key: AdmissionKey, submit_no: u64) -> bool {
        let (id, priority, estimate, arrival) = (r.id, r.priority, r.estimate, r.arrival);
        match self.queue.push_keyed(
            id,
            priority,
            key,
            estimate,
            PressureReason::SurvivalRequeue,
            arrival,
            Waiting { req: r, submit_no },
        ) {
            Ok(()) => {
                self.admission.record_requeue(id, &estimate);
                self.publish_depth();
                true
            }
            Err(w) => {
                let waiting = Queued {
                    id,
                    priority,
                    estimate,
                    reason: PressureReason::SurvivalRequeue,
                    enqueued_at: arrival,
                    bypassed: 0,
                    key,
                    payload: w,
                };
                self.finish_waits(vec![waiting], RejectionReason::Survival);
                false
            }
        }
    }

    /// Take `id` out of the queue (client disconnect): it never took a reservation.
    pub fn remove(&mut self, id: RequestId) -> Option<SchedRequest> {
        let q = self.queue.remove(id)?;
        self.publish_depth();
        Some(q.payload.req)
    }

    /// Queued requests whose client has gone (their `CancelFlag` is set).
    pub fn take_cancelled(&mut self) -> Vec<SchedRequest> {
        let ids: Vec<RequestId> = self
            .queue
            .iter()
            .filter(|q| q.payload.req.cancel.is_cancelled())
            .map(|q| q.id)
            .collect();
        ids.into_iter().filter_map(|id| self.remove(id)).collect()
    }

    /// Everything queued (the circuit opened: answered `circuit_open`).
    pub fn reject_all(&mut self) -> Vec<SchedRequest> {
        let all = self.queue.drain_all();
        self.finish_waits(all, RejectionReason::CircuitOpen)
    }

    /// `Retry-After` of a rejection: the remaining cooldown for `circuit_open` (at least 1),
    /// the estimated queue drain time (1..=60) for the overload reasons, 0 otherwise.
    pub fn retry_after_secs(&self, reason: RejectionReason) -> u64 {
        match reason {
            RejectionReason::ContextExceedsKvCapacity => 0,
            RejectionReason::CircuitOpen => {
                self.controller.snapshot().circuit_retry_after_secs.max(1)
            }
            _ => self.queue.estimated_drain_seconds(self.max_running),
        }
    }

    /// A worst-case KV reservation outside the queue (circuit probes bypass admission).
    pub fn reserve_direct(&self, est: &ResourceEstimate) -> Result<Reservation, LedgerError> {
        self.admission.reserve_kv(est)
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn max_queue(&self) -> usize {
        self.queue.max_queue()
    }

    pub fn contains(&self, id: RequestId) -> bool {
        self.queue.iter().any(|q| q.id == id)
    }

    /// Queued request ids in queue order.
    pub fn queued_ids(&self) -> Vec<RequestId> {
        self.queue.iter().map(|q| q.id).collect()
    }

    pub fn controller(&self) -> &ControllerHandle {
        &self.controller
    }

    pub fn admission_mut(&mut self) -> &mut Admission {
        &mut self.admission
    }

    /// A request whose new prefill exceeds `large_prefill_tokens`; the scheduler calls
    /// `admission_mut().expensive_prefill_finished()` when that prefill ends.
    pub fn is_expensive(&self, est: &ResourceEstimate) -> bool {
        est.new_prefill_tokens > self.admission.params().large_prefill_tokens
    }

    fn started(&mut self, est: &ResourceEstimate) {
        if self.is_expensive(est) {
            self.admission.expensive_prefill_started();
        }
    }

    /// Queued requests leaving without admission: each is a `reject` decision with `reason`
    /// (`turbine_admission_decisions_total`, `admission_decision` INFO) and a queue wait.
    fn finish_waits(
        &self,
        gone: Vec<Queued<Waiting, AdmissionKey>>,
        reason: RejectionReason,
    ) -> Vec<SchedRequest> {
        let now = self.clock.now_mono();
        let metrics = self.admission.metrics();
        let (decision, label) = AdmissionDecision::Reject { reason }.labels();
        let out = gone
            .into_iter()
            .map(|q| {
                let waited = now.saturating_sub(q.enqueued_at).as_secs_f64();
                metrics.admission_queue_wait_seconds.observe(waited);
                metrics
                    .admission_decisions
                    .get_or_create(&DecisionLabels {
                        decision,
                        reason: label,
                    })
                    .inc();
                tracing::info!(
                    event = "admission_decision",
                    request_id = %q.id.0,
                    decision,
                    reason = label,
                    waited_s = waited,
                    queued_reason = q.reason.as_str(),
                );
                q.payload.req
            })
            .collect();
        self.publish_depth();
        out
    }

    fn publish_depth(&self) {
        self.admission
            .metrics()
            .admission_queue_depth
            .set(i64::try_from(self.queue.len()).unwrap_or(i64::MAX));
    }
}

impl std::fmt::Debug for AdmissionGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmissionGate")
            .field("queued", &self.queue.len())
            .field("max_running", &self.max_running)
            .finish_non_exhaustive()
    }
}
