//! P3 overload acceptance tests (S-3, S-9, S-10, S-11, S-17): the real scheduler, admission
//! gate, pressure controller and recovery controller on a simulated executor, a host-memory
//! KV pool linked to the reservation ledger and a fake clock.

use std::time::Duration;

use turbine_core::types::{CircuitState, PressureState, RequestId};
use turbine_reliability::recovery::RecoveryOutcome;
use turbine_scheduler::sim::overload::{Outcome, OverloadConfig, OverloadSim};

fn secs(s: u64) -> Duration {
    Duration::from_secs(s)
}

/// Reject-table codes (P3 §Admission decisions and HTTP mapping).
const REJECT_CODES: [&str; 5] = [
    "context_exceeds_kv_capacity",
    "queue_full",
    "queue_timeout",
    "overloaded",
    "circuit_open",
];

#[test]
fn cancellation_releases_reservations() {
    let mut cfg = OverloadConfig {
        pool_blocks: 1024,
        ..OverloadConfig::default()
    };
    cfg.params.max_running_requests = 128;
    let mut sim = OverloadSim::new(cfg);

    // 100 running: 16 + 96 tokens = 7 blocks each, 700 of the 1,024 blocks reserved.
    let running: Vec<RequestId> = (0..100).map(|_| sim.submit_now(16, Some(96))).collect();
    for _ in 0..3 {
        sim.step_iteration();
    }
    let now_running = sim.running_ids();
    assert!(
        running.iter().all(|id| now_running.contains(id)),
        "all 100 small requests run"
    );
    // 100 queued: 5,000 + 1,000 tokens = 375 blocks each, more than the 324 unreserved.
    let queued: Vec<RequestId> = (0..100).map(|_| sim.submit_now(5000, Some(1000))).collect();
    // 100 short ones behind them (a few overtake the blocked head, the rest wait).
    let short: Vec<RequestId> = (0..100).map(|_| sim.submit_now(8, Some(8))).collect();
    sim.step_iteration();
    assert!(
        queued.iter().all(|id| sim.is_queued(*id)),
        "large requests wait for KV"
    );
    assert!(short.iter().any(|id| sim.is_queued(*id)));
    assert!(sim.kv_usage().reserved + sim.kv_usage().used > 0);
    assert!(sim.pool_used_blocks() > 0);

    for id in running.iter().chain(&queued).chain(&short) {
        sim.cancel(*id);
    }
    sim.step_iteration();
    let kv = sim.kv_usage();
    assert_eq!(kv.used + kv.reserved, 0, "KV ledger back to idle: {kv:?}");
    let ws = sim.workspace_usage();
    assert_eq!(ws.used + ws.reserved, 0, "workspace ledger back to idle");
    assert_eq!(sim.pool_used_blocks(), 0, "every block back in the pool");
    assert!(sim.ledger_idle());
    for id in running.iter().chain(&queued) {
        assert_eq!(sim.outcome(*id), Some(&Outcome::Cancelled));
    }
    assert!(sim.queued_ids().is_empty() && sim.running_ids().is_empty());
}

/// An admitted request holds a running slot (P3 S-10: batch growth counts admitted requests):
/// with every slot taken, a new request waits in the admission queue — holding no KV — and is
/// bounded by `reliability.admission.queue_timeout` there, instead of being admitted behind the
/// running ones with a reservation that never times out.
#[test]
fn admission_waits_for_a_running_slot() {
    let mut cfg = OverloadConfig::default();
    cfg.params.max_running_requests = 2;
    cfg.reliability.admission.queue_timeout = turbine_core::config::HumanDuration::from_secs(2);
    let mut sim = OverloadSim::new(cfg);
    // 2 × (16 + 4000 tokens) = 2 × 251 blocks: far from the 4,096-block pool.
    let running: Vec<RequestId> = (0..2).map(|_| sim.submit_now(16, Some(4000))).collect();
    let waiting: Vec<RequestId> = (0..3).map(|_| sim.submit_now(16, Some(16))).collect();
    sim.step_iteration();
    assert_eq!(sim.running_ids(), running);
    for id in &waiting {
        assert!(sim.is_queued(*id), "{id:?} waits in the admission queue");
    }
    let kv = sim.kv_usage();
    assert_eq!(
        kv.used + kv.reserved,
        2 * 251 * 16_384,
        "only the running requests hold KV: {kv:?}"
    );
    // The running decodes take 4,000 steps of 20 ms; the waiting ones time out after 2 s.
    sim.run_for(secs(3));
    for id in &waiting {
        assert_eq!(
            sim.outcome(*id),
            Some(&Outcome::Rejected("queue_timeout".into()))
        );
    }
    assert_eq!(sim.running_ids(), running);
}

/// Runs over every registered scheduling policy (Phase 2m: the simulator runs any registered
/// policy, and every policy must keep the overload invariants).
#[test]
fn ten_x_overload() {
    for policy in turbine_scheduler::policy::registry().iter() {
        ten_x_overload_under(policy.name());
    }
}

fn ten_x_overload_under(policy: &'static str) {
    let cfg = OverloadConfig {
        seed: 7,
        rate_multiple: 10.0,
        prompt_range: (64, 6000),
        max_tokens_range: (16, 1024),
        policy,
        ..OverloadConfig::default()
    };
    let mut sim = OverloadSim::new(cfg);
    let capacity = sim.config().service_rate() * 600.0;
    let r = sim.run_load(secs(600), secs(120));
    let count = |code: &str| {
        r.outcomes
            .iter()
            .filter(|o| matches!(o, Outcome::Rejected(c) if c == code))
            .count()
    };
    eprintln!(
        "ten_x_overload [{policy}]: {} requests, {} completed (capacity bound {capacity:.0}), {} queue_full, {} queue_timeout, {} overloaded, GREEN+HEALTHY {:?} after stop, states {:?}",
        r.outcomes.len(),
        r.outcomes
            .iter()
            .filter(|o| **o == Outcome::Completed)
            .count(),
        count("queue_full"),
        count("queue_timeout"),
        count("overloaded"),
        r.green_after_stop,
        r.states
    );

    assert!(
        r.max_kv_committed_plus_reserved <= r.kv_capacity_bytes,
        "KV {} > capacity {}",
        r.max_kv_committed_plus_reserved,
        r.kv_capacity_bytes
    );
    assert_eq!(r.preempted_below_survival, 0);
    assert_eq!(r.unfinished, 0, "every request reached an outcome");
    assert!(r.outcomes.len() > 1000, "{} requests", r.outcomes.len());
    let mut overload_codes = 0;
    let mut completed = 0;
    for o in &r.outcomes {
        match o {
            Outcome::Completed => completed += 1,
            Outcome::Cancelled => {}
            Outcome::Rejected(code) => {
                assert!(REJECT_CODES.contains(&code.as_str()), "code {code}");
                if ["queue_full", "queue_timeout", "overloaded"].contains(&code.as_str()) {
                    overload_codes += 1;
                }
            }
            Outcome::Failed(code) => panic!("request failed with {code} under plain overload"),
        }
    }
    // Goodput floor (user decision 2026-09-26: RED refills finished slots): at least half of
    // the completions the analytic service rate allows in the loaded 600 s. Admit-nothing RED
    // completed 16 (2 %); freezing the running count instead of the admitted one, 166 (24 %).
    assert!(
        completed as f64 >= 0.5 * capacity,
        "{completed} completions < 50 % of the capacity bound {capacity:.0}"
    );
    assert_eq!(r.red_growth, 0, "the admitted count never grows in RED");
    assert!(overload_codes > 0, "overload surfaces as reject codes");
    assert!(r.max_queue_len <= 256, "queue length {}", r.max_queue_len);
    assert!(
        r.states.contains(&PressureState::Red),
        "states seen: {:?}",
        r.states
    );
    let back = r
        .green_after_stop
        .expect("GREEN + HEALTHY after the load stopped");
    assert!(
        back <= secs(60),
        "GREEN + HEALTHY {back:?} after the load stopped"
    );
    assert_eq!(r.final_circuit, CircuitState::Healthy);
}

#[test]
fn active_generations_protected() {
    let mut sim = OverloadSim::new(OverloadConfig::default());
    let decodes: Vec<RequestId> = (0..8).map(|_| sim.submit_now(100, Some(4000))).collect();
    let flood_at = sim.now() + secs(5);
    sim.run_for(secs(5));
    let pre = sim
        .mean_step_time(&decodes, Duration::ZERO, flood_at)
        .expect("decode steps before the flood");

    for _ in 0..2000 {
        sim.submit_now(6000, Some(16));
    }
    sim.run_until_done(secs(600));
    let post = sim
        .mean_step_time(&decodes, flood_at, sim.now())
        .expect("decode steps after the flood");
    assert!(
        post <= 1.5 * pre,
        "decode step {post:.4}s after the flood vs {pre:.4}s before"
    );
    for id in &decodes {
        assert_eq!(sim.outcome(*id), Some(&Outcome::Completed));
    }
    assert_eq!(sim.report().preempted_below_survival, 0);
}

#[test]
fn oom_recovery_bounded() {
    let mut sim = OverloadSim::new(OverloadConfig::default());
    let batch: Vec<RequestId> = (0..8).map(|_| sim.submit_now(16, Some(200))).collect();
    for _ in 0..3 {
        sim.step_iteration();
    }

    // Device OOM on the next two attempts: the third, smaller attempt succeeds.
    sim.inject_oom_attempts(2);
    let it = sim.step_iteration();
    assert_eq!(it.recovery, Some(RecoveryOutcome::Recovered { retries: 2 }));
    assert_eq!(
        it.attempt_batch_sizes.len(),
        3,
        "{:?}",
        it.attempt_batch_sizes
    );
    assert!(
        it.attempt_batch_sizes.windows(2).all(|w| w[1] < w[0]),
        "attempts shrink: {:?}",
        it.attempt_batch_sizes
    );
    assert!(it.failed_requests.is_empty());

    // Persistent OOM: the original attempt plus max_retries (3) retries, then give up.
    sim.inject_oom_attempts(4);
    let it = sim.step_iteration();
    assert_eq!(
        it.attempt_batch_sizes.len(),
        4,
        "{:?}",
        it.attempt_batch_sizes
    );
    assert_eq!(it.recovery, Some(RecoveryOutcome::Failed));
    assert_eq!(it.failed_requests.len(), batch.len());
    for id in &batch {
        assert_eq!(
            sim.outcome(*id),
            Some(&Outcome::Failed("resource_exhausted".to_string()))
        );
        let tail = sim.stream_tail(*id);
        assert_eq!(tail.len(), 2, "{tail:?}");
        let err: serde_json::Value =
            serde_json::from_str(tail[0].strip_prefix("data: ").expect("an SSE data line"))
                .expect("JSON error event");
        assert_eq!(err["error"]["type"], "server_error");
        assert_eq!(err["error"]["code"], "resource_exhausted");
        assert!(err["error"]["message"].is_string());
        assert_eq!(tail[1], "data: [DONE]");
    }

    // The worker keeps serving: once the circuit has drained and probed, a new request completes.
    sim.run_for(secs(120));
    let fresh = sim.submit_now(16, Some(32));
    sim.run_until_done(secs(60));
    assert_eq!(sim.outcome(fresh), Some(&Outcome::Completed));
    assert_eq!(sim.handle().circuit(), CircuitState::Healthy);
}

/// OPEN (P3 SURVIVAL liveness gap, reported 2026-09-26; the fix is the user's decision): the
/// ten-times overload of `ten_x_overload` with seed 6 enters SURVIVAL and never leaves it. In
/// SURVIVAL the prefill budget is 0, so prefills already in progress cannot finish, while the
/// worst-case KV reservations they hold keep `kv_utilization` above SURVIVAL's exit threshold
/// (0.97 − 5 %); nothing else releases KV, so the state is stuck even after every arrival has
/// stopped. Ignored until a fix is chosen; `--ignored` reproduces the failure.
#[test]
#[ignore = "SURVIVAL liveness gap: reproduces an open defect until a fix is chosen"]
fn survival_liveness_gap_seed_6() {
    let cfg = OverloadConfig {
        seed: 6,
        rate_multiple: 10.0,
        prompt_range: (64, 6000),
        max_tokens_range: (16, 1024),
        ..OverloadConfig::default()
    };
    let mut sim = OverloadSim::new(cfg);
    let r = sim.run_load(secs(600), secs(120));
    let kv = sim.kv_usage();
    eprintln!(
        "seed 6: states {:?}, final {:?}/{:?}, kv used {} + reserved {} of {} ({:.3}), running {}, queued {}, unfinished {}",
        r.states,
        sim.handle().state(),
        r.final_circuit,
        kv.used,
        kv.reserved,
        kv.capacity,
        kv.utilization(),
        sim.running_ids().len(),
        sim.queued_ids().len(),
        r.unfinished,
    );
    assert!(
        r.states.contains(&PressureState::Survival),
        "seed 6 reaches SURVIVAL: {:?}",
        r.states
    );
    // Liveness: the load stopped 120 s ago; the state must have returned to GREEN.
    assert_eq!(
        sim.handle().state(),
        PressureState::Green,
        "still {:?} 120 s after the load stopped",
        sim.handle().state()
    );
    assert!(r.green_after_stop.is_some_and(|t| t <= secs(60)));
}
