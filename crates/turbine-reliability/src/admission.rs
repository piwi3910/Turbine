//! Predictive admission (P3 S-9): estimate → Admit | Queue{reason} | Reject{reason}, worst-case KV
//! reservation, and the bounded admission queue with its starvation guard.
use crate::budget::PoolKind;
use crate::ledger::{Ledger, LedgerError, Reservation};
use crate::metrics::{DecisionLabels, ReliabilityMetrics};
use crate::signals::SignalThresholds;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use turbine_core::request::ResourceEstimate;
use turbine_core::types::{CircuitState, DeviceId, KvLayout, PressureState, Priority, RequestId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PressureReason {
    KvReservation,
    PressureOrange,
    PressureRed,
    CircuitDegraded,
    PrefillBudget,
    /// An admitted request that had not started went back to the queue on entering SURVIVAL
    /// (`reliability.recovery.survival_liveness: requeue_unstarted`), dropping its KV
    /// reservation.
    SurvivalRequeue,
}
impl PressureReason {
    pub const ALL: [PressureReason; 6] = [
        PressureReason::KvReservation,
        PressureReason::PressureOrange,
        PressureReason::PressureRed,
        PressureReason::CircuitDegraded,
        PressureReason::PrefillBudget,
        PressureReason::SurvivalRequeue,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            PressureReason::KvReservation => "kv_reservation",
            PressureReason::PressureOrange => "pressure_orange",
            PressureReason::PressureRed => "pressure_red",
            PressureReason::CircuitDegraded => "circuit_degraded",
            PressureReason::PrefillBudget => "prefill_budget",
            PressureReason::SurvivalRequeue => "survival_requeue",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RejectionReason {
    ContextExceedsKvCapacity,
    QueueFull,
    QueueTimeout,
    Survival,
    CircuitOpen,
}
impl RejectionReason {
    pub const ALL: [RejectionReason; 5] = [
        RejectionReason::ContextExceedsKvCapacity,
        RejectionReason::QueueFull,
        RejectionReason::QueueTimeout,
        RejectionReason::Survival,
        RejectionReason::CircuitOpen,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            RejectionReason::ContextExceedsKvCapacity => "context_exceeds_kv_capacity",
            RejectionReason::QueueFull => "queue_full",
            RejectionReason::QueueTimeout => "queue_timeout",
            RejectionReason::Survival => "survival",
            RejectionReason::CircuitOpen => "circuit_open",
        }
    }
    /// (HTTP status, OpenAI error `type`, error `code`) — the P3 reject table.
    pub fn http(&self) -> (u16, &'static str, &'static str) {
        match self {
            RejectionReason::ContextExceedsKvCapacity => {
                (400, "invalid_request_error", "context_exceeds_kv_capacity")
            }
            RejectionReason::QueueFull => (429, "rate_limit_error", "queue_full"),
            RejectionReason::QueueTimeout => (503, "service_unavailable", "queue_timeout"),
            RejectionReason::Survival => (503, "service_unavailable", "overloaded"),
            RejectionReason::CircuitOpen => (503, "service_unavailable", "circuit_open"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Queue { reason: PressureReason },
    Reject { reason: RejectionReason },
}
impl AdmissionDecision {
    /// (`decision`, `reason`) metric labels; `reason="none"` for `Admit`.
    pub fn labels(&self) -> (&'static str, &'static str) {
        match self {
            AdmissionDecision::Admit => ("admit", "none"),
            AdmissionDecision::Queue { reason } => ("queue", reason.as_str()),
            AdmissionDecision::Reject { reason } => ("reject", reason.as_str()),
        }
    }
}

/// Throughput figures used before the EWMAs are seeded (the soak's calibration step, or engine warm-up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub prefill_tokens_per_s: f64,
    pub decode_step_s: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AdmissionParams {
    pub device: DeviceId,
    pub adaptive: bool,
    pub max_queue: u32,
    pub large_prefill_tokens: u32,
    pub block_bytes: u64,
    pub prefill_chunk_tokens: u32,
    pub workspace_bytes_per_token: u64,
    pub calibration: Calibration,
}

const EWMA_ALPHA: f64 = 0.1;
const SEED_ITERATIONS: u32 = 32;

pub struct Admission {
    params: AdmissionParams,
    ledger: Arc<Ledger>,
    metrics: ReliabilityMetrics,
    prefill_tps: f64,
    decode_step_s: f64,
    observed: u32,
    seed_prefill: (f64, f64),
    seed_decode: (f64, u32),
    expensive_in_flight: u32,
    /// `kv_utilization` thresholds for the KV headroom rule; `None` = no headroom rule.
    kv_headroom: Option<SignalThresholds>,
}

impl Admission {
    pub fn new(params: AdmissionParams, ledger: Arc<Ledger>, metrics: ReliabilityMetrics) -> Self {
        Self {
            prefill_tps: params.calibration.prefill_tokens_per_s,
            decode_step_s: params.calibration.decode_step_s,
            params,
            ledger,
            metrics,
            observed: 0,
            seed_prefill: (0.0, 0.0),
            seed_decode: (0.0, 0),
            expensive_in_flight: 0,
            kv_headroom: None,
        }
    }

    /// The KV headroom rule (P3 S-9, amendment 2026-09-27): with adaptive admission, in
    /// YELLOW, ORANGE and RED an admission (or a queued request's refill) waits
    /// (`kv_reservation`) when its worst-case reservation would lift `kv_utilization` (used +
    /// reserved) past the `kv` threshold of the next state up — so admissions alone never
    /// escalate the state, and after a load stops the queued backlog cannot push a
    /// de-escalating engine back up (the overload simulation's seed 1 went YELLOW → RED on its
    /// backlog and recovered 64 s after the load stopped). GREEN has no headroom rule.
    pub fn with_kv_headroom(mut self, kv: SignalThresholds) -> Self {
        self.kv_headroom = Some(kv);
        self
    }

    pub fn params(&self) -> &AdmissionParams {
        &self.params
    }

    /// `max_output_tokens` = `max_tokens`, else the remaining context; blocks = ceil((prompt + max_output) /
    /// block tokens) − cached full blocks (worst case, P3 §Data).
    pub fn estimate(
        &self,
        prompt_tokens: u32,
        cached_prefix_tokens: u32,
        max_tokens: Option<u32>,
        layout: &KvLayout,
        max_seq_len: u32,
    ) -> ResourceEstimate {
        let bt = u64::from(layout.block_tokens.max(1));
        let max_output = max_tokens.unwrap_or(max_seq_len.saturating_sub(prompt_tokens));
        let new_prefill = prompt_tokens.saturating_sub(cached_prefix_tokens);
        let blocks = (u64::from(prompt_tokens) + u64::from(max_output))
            .div_ceil(bt)
            .saturating_sub(u64::from(cached_prefix_tokens) / bt);
        self.with_timing(ResourceEstimate {
            prompt_tokens,
            cached_prefix_tokens,
            new_prefill_tokens: new_prefill,
            max_output_tokens: max_output,
            projected_kv_blocks: u32::try_from(blocks).unwrap_or(u32::MAX),
            ..ResourceEstimate::default()
        })
    }

    /// Fills the workspace and time fields of `est` (e.g. the Phase 2
    /// `ResourceEstimate::for_request` the scheduler receives) from the current throughput
    /// figures; `estimated` stays false until the EWMAs are seeded.
    pub fn with_timing(&self, mut est: ResourceEstimate) -> ResourceEstimate {
        est.workspace_bytes =
            u64::from(est.new_prefill_tokens.min(self.params.prefill_chunk_tokens))
                * self.params.workspace_bytes_per_token;
        est.est_prefill_seconds = f64::from(est.new_prefill_tokens) / self.prefill_tps.max(1e-9);
        est.est_decode_seconds = f64::from(est.max_output_tokens) * self.decode_step_s;
        est.estimated = self.observed >= SEED_ITERATIONS;
        est
    }

    pub fn metrics(&self) -> &ReliabilityMetrics {
        &self.metrics
    }

    /// The decision `decide` would take, without counting or logging it (re-checks of queued
    /// requests while the queue is pumped).
    pub fn evaluate(
        &self,
        est: &ResourceEstimate,
        state: PressureState,
        circuit: CircuitState,
        queue_len: usize,
    ) -> AdmissionDecision {
        self.decide_inner(est, state, state, circuit, queue_len)
    }

    /// The decision for a queued request offered a running slot the throttle plan opened (the
    /// admission gate's pump). RED refills finished slots (user decision 2026-09-26): its
    /// `pressure_red` queueing binds new arrivals only, and a queued request is judged by
    /// ORANGE's rules — expensive prefills keep waiting (`pressure_orange`). Hard capacity,
    /// the circuit, SURVIVAL and the worst-case KV reservation apply unchanged; `queue_full`
    /// never does (the request is already queued). Records nothing.
    pub fn evaluate_refill(
        &self,
        est: &ResourceEstimate,
        state: PressureState,
        circuit: CircuitState,
    ) -> AdmissionDecision {
        let rules = match state {
            PressureState::Red => PressureState::Orange,
            other => other,
        };
        self.decide_inner(est, rules, state, circuit, 0)
    }

    fn kv_bytes(&self, est: &ResourceEstimate) -> u64 {
        est.projected_kv_blocks as u64 * self.params.block_bytes
    }

    /// An admitted request that had not started goes back to the queue (SURVIVAL, option A):
    /// a `queue` decision with reason `survival_requeue` in `turbine_admission_decisions_total`
    /// and an `admission_decision` INFO event.
    pub fn record_requeue(&self, request_id: RequestId, est: &ResourceEstimate) {
        let (decision, reason) = AdmissionDecision::Queue {
            reason: PressureReason::SurvivalRequeue,
        }
        .labels();
        self.metrics
            .admission_decisions
            .get_or_create(&DecisionLabels { decision, reason })
            .inc();
        tracing::info!(
            event = "admission_decision",
            request_id = %request_id.0,
            decision,
            reason,
            released_kv_blocks = est.projected_kv_blocks,
            "admitted request returned to the queue in SURVIVAL"
        );
    }

    /// Pure decision; records `turbine_admission_decisions_total` and an `admission_decision` log event.
    pub fn decide(
        &mut self,
        est: &ResourceEstimate,
        state: PressureState,
        circuit: CircuitState,
        queue_len: usize,
    ) -> AdmissionDecision {
        let d = self.decide_inner(est, state, state, circuit, queue_len);
        let (decision, reason) = d.labels();
        self.metrics
            .admission_decisions
            .get_or_create(&DecisionLabels { decision, reason })
            .inc();
        if d == AdmissionDecision::Admit {
            tracing::debug!(
                event = "admission_decision",
                decision,
                reason,
                projected_kv_blocks = est.projected_kv_blocks,
                new_prefill_tokens = est.new_prefill_tokens
            );
        } else {
            tracing::info!(
                event = "admission_decision",
                decision,
                reason,
                projected_kv_blocks = est.projected_kv_blocks,
                new_prefill_tokens = est.new_prefill_tokens,
                state = state.as_str(),
                queue_len
            );
        }
        d
    }

    /// `state` selects the pressure rules, `actual` (the controller's state) the headroom.
    fn decide_inner(
        &self,
        est: &ResourceEstimate,
        state: PressureState,
        actual: PressureState,
        circuit: CircuitState,
        queue_len: usize,
    ) -> AdmissionDecision {
        let kv = self.ledger.usage(self.params.device, PoolKind::Kv);
        let need = self.kv_bytes(est);
        if need > kv.capacity {
            return AdmissionDecision::Reject {
                reason: RejectionReason::ContextExceedsKvCapacity,
            };
        }
        if circuit.blocks_readiness() {
            return AdmissionDecision::Reject {
                reason: RejectionReason::CircuitOpen,
            };
        }
        let expensive = est.new_prefill_tokens > self.params.large_prefill_tokens;
        let pressure = if !self.params.adaptive {
            None
        } else {
            match state {
                PressureState::Survival => {
                    return AdmissionDecision::Reject {
                        reason: RejectionReason::Survival,
                    };
                }
                PressureState::Red => Some(PressureReason::PressureRed),
                PressureState::Orange if expensive => Some(PressureReason::PressureOrange),
                _ if circuit == CircuitState::Degraded && expensive => {
                    Some(PressureReason::CircuitDegraded)
                }
                PressureState::Yellow if expensive && self.expensive_in_flight > 0 => {
                    Some(PressureReason::PrefillBudget)
                }
                _ => None,
            }
        };
        let short =
            need > kv.available() || (self.params.adaptive && !self.within_headroom(need, actual));
        let reason = pressure.or(short.then_some(PressureReason::KvReservation));
        match reason {
            None => AdmissionDecision::Admit,
            Some(_) if queue_len >= self.params.max_queue as usize => AdmissionDecision::Reject {
                reason: RejectionReason::QueueFull,
            },
            Some(reason) => AdmissionDecision::Queue { reason },
        }
    }

    /// Whether `need` more reserved bytes keep `kv_utilization` at or below the `kv` threshold
    /// of the state above `state` (the headroom rule of [`Admission::with_kv_headroom`]).
    fn within_headroom(&self, need: u64, state: PressureState) -> bool {
        let Some(kv) = &self.kv_headroom else {
            return true;
        };
        let next = match state {
            PressureState::Yellow => PressureState::Orange,
            PressureState::Orange => PressureState::Red,
            PressureState::Red => PressureState::Survival,
            _ => return true,
        };
        let Some(limit) = kv.threshold(next) else {
            return true;
        };
        let usage = self.ledger.usage(self.params.device, PoolKind::Kv);
        if usage.capacity == 0 {
            return true;
        }
        let after = usage
            .used
            .saturating_add(usage.reserved)
            .saturating_add(need);
        (after as f64 / usage.capacity as f64) <= limit
    }

    /// Takes the worst-case KV reservation of an admitted request.
    pub fn reserve_kv(&self, est: &ResourceEstimate) -> Result<Reservation, LedgerError> {
        self.ledger
            .reserve(self.params.device, PoolKind::Kv, self.kv_bytes(est))
    }

    pub fn expensive_prefill_started(&mut self) {
        self.expensive_in_flight += 1;
    }
    pub fn expensive_prefill_finished(&mut self) {
        self.expensive_in_flight = self.expensive_in_flight.saturating_sub(1);
    }

    /// Feed one iteration's observed throughput. The first 32 iterations seed the EWMAs by their mean.
    pub fn observe_iteration(
        &mut self,
        prefill_tokens: u32,
        prefill_seconds: f64,
        decode_step_seconds: Option<f64>,
    ) {
        self.observed = self.observed.saturating_add(1);
        if self.observed <= SEED_ITERATIONS {
            self.seed_prefill.0 += prefill_tokens as f64;
            self.seed_prefill.1 += prefill_seconds;
            if let Some(d) = decode_step_seconds {
                self.seed_decode.0 += d;
                self.seed_decode.1 += 1;
            }
            if self.observed == SEED_ITERATIONS {
                if self.seed_prefill.1 > 0.0 {
                    self.prefill_tps = self.seed_prefill.0 / self.seed_prefill.1;
                }
                if self.seed_decode.1 > 0 {
                    self.decode_step_s = self.seed_decode.0 / self.seed_decode.1 as f64;
                }
            }
            return;
        }
        if prefill_tokens > 0 && prefill_seconds > 0.0 {
            self.prefill_tps = EWMA_ALPHA * (prefill_tokens as f64 / prefill_seconds)
                + (1.0 - EWMA_ALPHA) * self.prefill_tps;
        }
        if let Some(d) = decode_step_seconds {
            self.decode_step_s = EWMA_ALPHA * d + (1.0 - EWMA_ALPHA) * self.decode_step_s;
        }
    }
}

/// One waiting request (payload = the engine's own request object). `key` orders the queue
/// (smallest first): `(priority, arrival sequence)` by default, or the scheduling policy's
/// admission key when the scheduler's gate supplies one (Phase 2m `scheduling_policy`).
pub struct Queued<T, K = (Priority, u64)> {
    pub id: RequestId,
    pub priority: Priority,
    pub estimate: ResourceEstimate,
    pub reason: PressureReason,
    pub enqueued_at: Duration,
    pub bypassed: u32,
    pub key: K,
    pub payload: T,
}

/// Ordered by `K` (by default FIFO within priority, lower `Priority` first), bounded by
/// `max_queue` and `queue_timeout`; a request that fits may overtake a head that does not fit
/// at most `max_bypass` times, then the head blocks the queue.
pub struct AdmissionQueue<T, K = (Priority, u64)> {
    entries: Vec<Queued<T, K>>,
    max_queue: usize,
    queue_timeout: Duration,
    max_bypass: u32,
    next_seq: u64,
}

impl<T> AdmissionQueue<T, (Priority, u64)> {
    /// Queue by `(priority, arrival)`. Err(payload) when the queue is full (the caller
    /// answers `queue_full`).
    pub fn push(
        &mut self,
        id: RequestId,
        priority: Priority,
        estimate: ResourceEstimate,
        reason: PressureReason,
        now: Duration,
        payload: T,
    ) -> Result<(), T> {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.push_keyed(
            id,
            priority,
            (priority, seq),
            estimate,
            reason,
            now,
            payload,
        )
    }
}

impl<T, K: Ord + Copy> AdmissionQueue<T, K> {
    pub fn new(max_queue: u32, queue_timeout: Duration, max_bypass: u32) -> Self {
        Self {
            entries: Vec::new(),
            max_queue: max_queue as usize,
            queue_timeout,
            max_bypass,
            next_seq: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn max_queue(&self) -> usize {
        self.max_queue
    }
    pub fn iter(&self) -> impl Iterator<Item = &Queued<T, K>> {
        self.entries.iter()
    }

    /// Queue at `key` (unique per entry; equal keys keep insertion order). `enqueued_at` starts
    /// the `queue_timeout` clock. Err(payload) when the queue is full.
    #[allow(clippy::too_many_arguments)]
    pub fn push_keyed(
        &mut self,
        id: RequestId,
        priority: Priority,
        key: K,
        estimate: ResourceEstimate,
        reason: PressureReason,
        enqueued_at: Duration,
        payload: T,
    ) -> Result<(), T> {
        if self.entries.len() >= self.max_queue {
            return Err(payload);
        }
        let at = self.entries.partition_point(|e| e.key <= key);
        self.entries.insert(
            at,
            Queued {
                id,
                priority,
                estimate,
                reason,
                enqueued_at,
                bypassed: 0,
                key,
                payload,
            },
        );
        Ok(())
    }

    /// Client disconnect: removed without ever taking a reservation.
    pub fn remove(&mut self, id: RequestId) -> Option<Queued<T, K>> {
        let i = self.entries.iter().position(|e| e.id == id)?;
        Some(self.entries.remove(i))
    }

    /// Entries older than `queue_timeout` (the caller answers `queue_timeout`).
    pub fn expire(&mut self, now: Duration) -> Vec<Queued<T, K>> {
        let (expired, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.entries)
            .into_iter()
            .partition(|e| now.saturating_sub(e.enqueued_at) >= self.queue_timeout);
        self.entries = kept;
        expired
    }

    /// Everything, in order (circuit open rejects the whole queue).
    pub fn drain_all(&mut self) -> Vec<Queued<T, K>> {
        std::mem::take(&mut self.entries)
    }

    /// Admits in queue order; `try_admit` returns true when the entry was admitted (it takes the reservation).
    pub fn pump(&mut self, mut try_admit: impl FnMut(&Queued<T, K>) -> bool) -> Vec<Queued<T, K>> {
        let mut admitted = Vec::new();
        while let Some(head) = self.entries.first() {
            if try_admit(head) {
                admitted.push(self.entries.remove(0));
                continue;
            }
            let mut j = 1;
            while j < self.entries.len() && self.entries[0].bypassed < self.max_bypass {
                if try_admit(&self.entries[j]) {
                    admitted.push(self.entries.remove(j));
                    self.entries[0].bypassed += 1;
                } else {
                    j += 1;
                }
            }
            break;
        }
        admitted
    }

    /// Estimated seconds to drain the queue with `max_running` concurrent sequences, clamped to 1..=60
    /// (the `Retry-After` of `queue_full`, `queue_timeout` and `overloaded`).
    pub fn estimated_drain_seconds(&self, max_running: u32) -> u64 {
        let work: f64 = self
            .entries
            .iter()
            .map(|e| e.estimate.est_prefill_seconds + e.estimate.est_decode_seconds)
            .sum();
        ((work / max_running.max(1) as f64).ceil() as u64).clamp(1, 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::DeviceBudget;
    use turbine_core::types::{DType, MemoryKind, PressureSignal};

    const BLOCK_BYTES: u64 = 1_835_008; // Llama-3.2-3B, 16-token blocks
    const LAYOUT: KvLayout = KvLayout {
        num_layers: 28,
        num_kv_heads: 8,
        head_dim: 128,
        dtype: DType::BF16,
        block_tokens: 16,
    };
    const MAX_SEQ: u32 = 8192;

    fn setup(pool_blocks: u64, adaptive: bool) -> (Admission, Arc<Ledger>) {
        let pool = pool_blocks * BLOCK_BYTES;
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: pool,
            pools: vec![(PoolKind::Kv, pool)],
        };
        let ledger = Ledger::new(&budget);
        let params = AdmissionParams {
            device: DeviceId(0),
            adaptive,
            max_queue: 4,
            large_prefill_tokens: 2048,
            block_bytes: BLOCK_BYTES,
            prefill_chunk_tokens: 2048,
            workspace_bytes_per_token: 0,
            calibration: Calibration {
                prefill_tokens_per_s: 10_000.0,
                decode_step_s: 0.02,
            },
        };
        (
            Admission::new(
                params,
                Arc::clone(&ledger),
                ReliabilityMetrics::unregistered(),
            ),
            ledger,
        )
    }

    /// RED refills finished slots (user decision 2026-09-26): a queued request offered a slot
    /// the throttle plan opened is not held back by `pressure_red`, but keeps ORANGE's
    /// expensive-prefill rule, the KV reservation and SURVIVAL's rejection.
    #[test]
    fn red_refill_from_queue() {
        use AdmissionDecision::*;
        use PressureState::*;
        let (mut a, ledger) = setup(1024, true);
        let h = CircuitState::Healthy;
        let cheap = a.estimate(100, 0, Some(100), &LAYOUT, MAX_SEQ);
        let expensive = a.estimate(4000, 0, Some(100), &LAYOUT, MAX_SEQ);
        let red = Queue {
            reason: PressureReason::PressureRed,
        };
        let orange = Queue {
            reason: PressureReason::PressureOrange,
        };
        // New arrivals still queue in RED; a queued one refills a finished slot.
        assert_eq!(a.decide(&cheap, Red, h, 0), red);
        assert_eq!(a.evaluate(&cheap, Red, h, 0), red);
        assert_eq!(a.evaluate_refill(&cheap, Red, h), Admit);
        // Expensive prefills wait in RED as in ORANGE; cheap ones refill in ORANGE too.
        assert_eq!(a.evaluate_refill(&expensive, Red, h), orange);
        assert_eq!(a.evaluate_refill(&expensive, Orange, h), orange);
        assert_eq!(a.evaluate_refill(&cheap, Orange, h), Admit);
        assert_eq!(a.evaluate_refill(&expensive, Green, h), Admit);
        // SURVIVAL never drains the queue; an open circuit rejects.
        assert_eq!(
            a.evaluate_refill(&cheap, Survival, h),
            Reject {
                reason: RejectionReason::Survival
            }
        );
        assert_eq!(
            a.evaluate_refill(&cheap, Red, CircuitState::CircuitOpen),
            Reject {
                reason: RejectionReason::CircuitOpen
            }
        );
        // The worst-case KV reservation still bounds a refill.
        let held = ledger
            .reserve(DeviceId(0), PoolKind::Kv, 1020 * BLOCK_BYTES)
            .unwrap();
        assert_eq!(
            a.evaluate_refill(&cheap, Red, h),
            Queue {
                reason: PressureReason::KvReservation
            }
        );
        drop(held);
        // A refill is not a queue insertion: `queue_full` never applies.
        let before = a
            .metrics()
            .admission_decisions
            .get_or_create(&DecisionLabels {
                decision: "admit",
                reason: "none",
            })
            .get();
        assert_eq!(a.evaluate_refill(&cheap, Red, h), Admit);
        assert_eq!(
            a.metrics()
                .admission_decisions
                .get_or_create(&DecisionLabels {
                    decision: "admit",
                    reason: "none",
                })
                .get(),
            before,
            "evaluate_refill records nothing"
        );
    }

    #[test]
    fn decision_table() {
        use AdmissionDecision::*;
        use PressureState::*;
        let (mut a, ledger) = setup(1024, true); // 16,384 tokens of KV
        let h = CircuitState::Healthy;
        let cheap = a.estimate(100, 0, Some(100), &LAYOUT, MAX_SEQ);
        let expensive = a.estimate(4000, 0, Some(100), &LAYOUT, MAX_SEQ);
        let huge = a.estimate(4000, 0, Some(13_000), &LAYOUT, 32_768);
        assert_eq!(
            a.decide(&huge, Green, h, 0),
            Reject {
                reason: RejectionReason::ContextExceedsKvCapacity
            }
        );
        assert_eq!(a.decide(&cheap, Green, h, 0), Admit);
        assert_eq!(
            a.decide(&expensive, Orange, h, 0),
            Queue {
                reason: PressureReason::PressureOrange
            }
        );
        assert_eq!(
            a.decide(&cheap, Orange, h, 0),
            Admit,
            "cheap prefills are still admitted in ORANGE"
        );
        assert_eq!(
            a.decide(&cheap, Red, h, 0),
            Queue {
                reason: PressureReason::PressureRed
            }
        );
        let counted = |a: &Admission| {
            a.metrics()
                .admission_decisions
                .get_or_create(&DecisionLabels {
                    decision: "queue",
                    reason: "pressure_red",
                })
                .get()
        };
        let before = counted(&a);
        assert_eq!(
            a.evaluate(&cheap, Red, h, 0),
            Queue {
                reason: PressureReason::PressureRed
            },
            "evaluate takes the same decision"
        );
        assert_eq!(counted(&a), before, "evaluate records nothing");
        let timed = a.with_timing(ResourceEstimate::for_request(100, 100, 16));
        assert_eq!(timed.est_prefill_seconds, 100.0 / 10_000.0);
        assert_eq!(timed.est_decode_seconds, 100.0 * 0.02);
        assert!(
            !timed.estimated,
            "calibration figures until the EWMAs are seeded"
        );
        assert_eq!(
            a.decide(&cheap, Red, h, 4),
            Reject {
                reason: RejectionReason::QueueFull
            }
        );
        assert_eq!(
            a.decide(&cheap, Survival, h, 0),
            Reject {
                reason: RejectionReason::Survival
            }
        );
        assert_eq!(
            a.decide(&cheap, Green, CircuitState::CircuitOpen, 0),
            Reject {
                reason: RejectionReason::CircuitOpen
            }
        );
        assert_eq!(
            a.decide(&expensive, Green, CircuitState::Degraded, 0),
            Queue {
                reason: PressureReason::CircuitDegraded
            }
        );
        a.expensive_prefill_started();
        assert_eq!(
            a.decide(&expensive, Yellow, h, 0),
            Queue {
                reason: PressureReason::PrefillBudget
            }
        );
        a.expensive_prefill_finished();
        // Only free capacity is short: another request holds 1,020 of the 1,024 blocks.
        let held = ledger
            .reserve(DeviceId(0), PoolKind::Kv, 1020 * BLOCK_BYTES)
            .unwrap();
        assert_eq!(
            a.decide(&cheap, Green, h, 0),
            Queue {
                reason: PressureReason::KvReservation
            }
        );
        drop(held);
        // adaptive_admission: false checks only hard capacity and queue bounds.
        let (mut plain, _) = setup(1024, false);
        assert_eq!(plain.decide(&cheap, Red, h, 0), Admit);
        assert_eq!(plain.decide(&cheap, Survival, h, 0), Admit);
        assert_eq!(
            plain.decide(&huge, Green, h, 0),
            Reject {
                reason: RejectionReason::ContextExceedsKvCapacity
            }
        );
        // Worst-case reservation: no max_tokens → the whole remaining context.
        let no_max = a.estimate(100, 0, None, &LAYOUT, MAX_SEQ);
        assert_eq!(no_max.projected_kv_blocks, MAX_SEQ.div_ceil(16));
        assert_eq!(a.reserve_kv(&no_max).unwrap().bytes(), 512 * BLOCK_BYTES);
        assert_eq!(
            a.reserve_kv(&cheap).unwrap().bytes(),
            200u64.div_ceil(16) * BLOCK_BYTES
        );
        // The HTTP mapping of every reject reason (P3 reject table).
        let http: Vec<_> = RejectionReason::ALL.iter().map(|r| r.http()).collect();
        assert_eq!(
            http,
            vec![
                (400, "invalid_request_error", "context_exceeds_kv_capacity"),
                (429, "rate_limit_error", "queue_full"),
                (503, "service_unavailable", "queue_timeout"),
                (503, "service_unavailable", "overloaded"),
                (503, "service_unavailable", "circuit_open"),
            ]
        );
    }

    fn est(blocks: u32) -> ResourceEstimate {
        ResourceEstimate {
            projected_kv_blocks: blocks,
            ..ResourceEstimate::default()
        }
    }

    /// P3 S-9 amendment 2026-09-27 (KV headroom): in YELLOW, ORANGE and RED an admission or a
    /// refill waits (`kv_reservation`) when its worst-case reservation would lift
    /// `kv_utilization` past the next state's `kv` threshold (0.82 / 0.90 / 0.97 by default);
    /// GREEN has no headroom rule, and without adaptive admission it never applies. Breaks if
    /// admissions alone can escalate the pressure state.
    #[test]
    fn kv_headroom() {
        let (a, ledger) = setup(1024, true);
        let thresholds = crate::signals::default_thresholds()[&PressureSignal::KvUtilization];
        let a = a.with_kv_headroom(thresholds);
        let h = CircuitState::Healthy;
        // 800 of 1,024 blocks held (0.781); a 40-block request would reach 0.820.
        let _held = ledger
            .reserve(DeviceId(0), PoolKind::Kv, 800 * BLOCK_BYTES)
            .unwrap();
        let est40 = est(40);
        let est20 = est(20);
        assert_eq!(
            a.evaluate(&est40, PressureState::Green, h, 0),
            AdmissionDecision::Admit
        );
        assert_eq!(
            a.evaluate(&est40, PressureState::Yellow, h, 0),
            AdmissionDecision::Queue {
                reason: PressureReason::KvReservation
            },
            "0.820 > ORANGE 0.82"
        );
        assert_eq!(
            a.evaluate(&est20, PressureState::Yellow, h, 0),
            AdmissionDecision::Admit
        );
        // ORANGE admits up to RED's 0.90 (921 blocks), a RED refill up to SURVIVAL's 0.97.
        assert_eq!(
            a.evaluate(&est40, PressureState::Orange, h, 0),
            AdmissionDecision::Admit
        );
        let est130 = est(130);
        assert_eq!(
            a.evaluate(&est130, PressureState::Orange, h, 0),
            AdmissionDecision::Queue {
                reason: PressureReason::KvReservation
            }
        );
        assert_eq!(
            a.evaluate_refill(&est130, PressureState::Red, h),
            AdmissionDecision::Admit,
            "0.908 <= SURVIVAL 0.97"
        );
        assert_eq!(
            a.evaluate_refill(&est(200), PressureState::Red, h),
            AdmissionDecision::Queue {
                reason: PressureReason::KvReservation
            },
            "0.977 > SURVIVAL 0.97"
        );
        // Without adaptive admission only hard capacity counts.
        let (plain, ledger2) = setup(1024, false);
        let plain = plain.with_kv_headroom(thresholds);
        let _held2 = ledger2
            .reserve(DeviceId(0), PoolKind::Kv, 800 * BLOCK_BYTES)
            .unwrap();
        assert_eq!(
            plain.evaluate(&est40, PressureState::Yellow, h, 0),
            AdmissionDecision::Admit
        );
    }

    #[test]
    fn bypass_bounded() {
        let mut q: AdmissionQueue<u32> = AdmissionQueue::new(64, Duration::from_secs(30), 8);
        let id = |_: u32| RequestId::new_v4();
        q.push(
            id(0),
            Priority(0),
            est(1000),
            PressureReason::KvReservation,
            Duration::ZERO,
            0,
        )
        .unwrap();
        for n in 1..=20 {
            q.push(
                id(n),
                Priority(0),
                est(1),
                PressureReason::KvReservation,
                Duration::ZERO,
                n,
            )
            .unwrap();
        }
        // Free capacity: 10 blocks. The head (1000 blocks) never fits; small ones do.
        let fits = |e: &Queued<u32>| e.estimate.projected_kv_blocks <= 10;
        let first: Vec<u32> = q.pump(fits).into_iter().map(|e| e.payload).collect();
        assert_eq!(
            first,
            (1..=8).collect::<Vec<_>>(),
            "exactly max_bypass overtakes, in FIFO order"
        );
        assert!(
            q.pump(fits).is_empty(),
            "after max_bypass the head blocks the queue"
        );
        assert_eq!(
            q.iter().next().map(|e| (e.payload, e.bypassed)),
            Some((0, 8))
        );
        // Once the head fits it goes first, and the rest follow.
        let all: Vec<u32> = q.pump(|_| true).into_iter().map(|e| e.payload).collect();
        assert_eq!(all, [0].into_iter().chain(9..=20).collect::<Vec<_>>());
        // Priority orders the queue; FIFO within a priority.
        let mut p: AdmissionQueue<u32> = AdmissionQueue::new(2, Duration::from_secs(30), 8);
        p.push(
            id(1),
            Priority(0),
            est(1),
            PressureReason::PressureRed,
            Duration::ZERO,
            1,
        )
        .unwrap();
        p.push(
            id(2),
            Priority(-1),
            est(1),
            PressureReason::PressureRed,
            Duration::ZERO,
            2,
        )
        .unwrap();
        assert_eq!(
            p.push(
                id(3),
                Priority(-5),
                est(1),
                PressureReason::PressureRed,
                Duration::ZERO,
                3
            ),
            Err(3),
            "bounded by max_queue"
        );
        assert_eq!(p.iter().map(|e| e.payload).collect::<Vec<_>>(), vec![2, 1]);
        assert_eq!(
            p.expire(Duration::from_secs(30)).len(),
            2,
            "queue_timeout bounds the wait"
        );
    }
}
