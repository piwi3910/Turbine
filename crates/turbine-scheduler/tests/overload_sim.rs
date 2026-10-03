//! P3 overload acceptance tests (S-3, S-9, S-10, S-11, S-17): the real scheduler, admission
//! gate, pressure controller and recovery controller on a simulated executor, a host-memory
//! KV pool linked to the reservation ledger and a fake clock.

use std::time::Duration;

use turbine_core::config::SurvivalLiveness;
use turbine_core::types::{CircuitState, PressureState, RequestId};
use turbine_reliability::recovery::RecoveryOutcome;
use turbine_scheduler::sim::overload::{Outcome, OverloadConfig, OverloadReport, OverloadSim};

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
    assert!(
        r.max_idle_with_queue <= Duration::from_secs(1),
        "idle {:?} with requests queued",
        r.max_idle_with_queue
    );
}

/// The soak's shape (`scripts/overload-soak.sh`): 4× the service rate, prompts of 64..6,000
/// tokens (three quarters above `large_prefill_tokens`), 16..1,024 new tokens.
fn soak_config(seed: u64) -> OverloadConfig {
    OverloadConfig {
        seed,
        rate_multiple: 4.0,
        prompt_range: (64, 6000),
        max_tokens_range: (16, 1024),
        ..OverloadConfig::default()
    }
}

/// Serving continues through the overload and ends GREEN + HEALTHY; the device never idles
/// while requests wait (below SURVIVAL, circuit admitting).
fn assert_keeps_serving(name: &str, r: &OverloadReport, capacity: f64) {
    let completed = r
        .outcomes
        .iter()
        .filter(|o| **o == Outcome::Completed)
        .count();
    eprintln!(
        "{name}: {} requests, {completed} completed (capacity bound {capacity:.0}), states {:?}, idle with queue {:?}, GREEN+HEALTHY {:?} after stop, final circuit {:?}",
        r.outcomes.len(),
        r.states,
        r.max_idle_with_queue,
        r.green_after_stop,
        r.final_circuit
    );
    assert!(
        r.max_idle_with_queue <= Duration::from_secs(1),
        "{name}: idle {:?} with requests queued",
        r.max_idle_with_queue
    );
    for o in &r.outcomes {
        if let Outcome::Rejected(code) = o {
            assert!(REJECT_CODES.contains(&code.as_str()), "{name}: code {code}");
        }
        assert!(!matches!(o, Outcome::Failed(_)), "{name}: {o:?}");
    }
    assert!(
        completed as f64 >= 0.5 * capacity,
        "{name}: {completed} completions < 50 % of the capacity bound {capacity:.0}"
    );
    let back = r
        .green_after_stop
        .expect("GREEN + HEALTHY after the load stopped");
    assert!(
        back <= secs(60),
        "{name}: GREEN + HEALTHY {back:?} after stop"
    );
    assert_eq!(r.final_circuit, CircuitState::Healthy, "{name}");
}

/// The 2026-09-27 soak on novanas: the calibration's shrinking batches read as latency drift
/// (step time divided by the batch), the circuit went DEGRADED, and the overload stalled with
/// nothing running and the queue full. Catches: drift from batch size, and any path where
/// pressure rules idle the device while requests wait.
#[test]
fn soak_workload_keeps_serving() {
    for seed in [1, 7] {
        let mut sim = OverloadSim::new(soak_config(seed));
        let capacity = sim.config().service_rate() * 600.0;
        let r = sim.run_load(secs(600), secs(120));
        assert_keeps_serving(&format!("soak seed {seed}"), &r, capacity);
    }
}

/// A DEGRADED circuit queues expensive prefills; with three quarters of the arrivals expensive
/// the queue head blocks, the admitted count drains to 0 and `queue_fill` holds ORANGE, whose
/// frozen batch then admits nothing. Catches: a pressure or circuit rule that leaves the engine
/// idle with requests queued (P3 S-10: sustained overload keeps serving).
#[test]
fn degraded_circuit_keeps_serving() {
    let mut sim = OverloadSim::new(OverloadConfig {
        degraded_during_load: true,
        ..soak_config(1)
    });
    let capacity = sim.config().service_rate() * 600.0;
    let r = sim.run_load(secs(600), secs(120));
    eprintln!(
        "degraded: {} requests, capacity bound {capacity:.0}, states {:?}, idle with queue {:?}, GREEN+HEALTHY {:?} after stop",
        r.outcomes.len(),
        r.states,
        r.max_idle_with_queue,
        r.green_after_stop
    );
    assert!(r.states.contains(&PressureState::Yellow));
    assert!(
        r.max_idle_with_queue <= Duration::from_secs(1),
        "idle {:?} with requests queued",
        r.max_idle_with_queue
    );
    // DEGRADED lasts until `reliability.circuit.window` (60 s) passes without a trigger after
    // the load stops, so only the end state is checked here.
    assert_eq!(r.unfinished, 0);
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

/// P3 SURVIVAL, option A (`survival_liveness: requeue_unstarted`): admitted requests that
/// have not started go back to the admission queue on entering SURVIVAL and give back their
/// worst-case KV reservations; the running ones keep theirs and finish; out of SURVIVAL the
/// requeued ones are admitted again and complete. Breaks if SURVIVAL keeps a reservation of a
/// request that wrote no KV, or loses a requeued request.
#[test]
fn survival_requeues_unstarted_admitted() {
    let mut cfg = OverloadConfig::default();
    // SURVIVAL after an OOM lasts two dwells; the requeued requests wait through it.
    cfg.reliability.admission.queue_timeout = turbine_core::config::HumanDuration::from_secs(120);
    let mut sim = OverloadSim::new(cfg);
    // 16 requests of 2,000 + 100 tokens (132 blocks each, 2,112 of 4,096 reserved): all are
    // admitted at once, but the 8,192-token budget starts only four prefills per iteration.
    let ids: Vec<RequestId> = (0..16).map(|_| sim.submit_now(2000, Some(100))).collect();
    sim.step_iteration();
    let started = sim.running_ids();
    assert!(
        !started.is_empty() && started.len() < ids.len(),
        "{started:?}"
    );
    let before = sim.kv_usage();
    assert_eq!(before.used + before.reserved, 16 * 132 * 16_384);

    // A device OOM (recovered on the retry) enters SURVIVAL: the next plan requeues.
    sim.inject_oom_attempts(1);
    sim.step_iteration();
    assert_eq!(sim.handle().state(), PressureState::Survival);
    // The recovered iteration may have started more prefills before SURVIVAL.
    let started = sim.running_ids();
    assert!(started.len() < ids.len(), "{started:?}");
    sim.step_iteration();
    let unstarted: Vec<RequestId> = ids
        .iter()
        .copied()
        .filter(|id| !started.contains(id))
        .collect();
    for id in &unstarted {
        assert!(sim.is_queued(*id), "{id:?} is back in the admission queue");
    }
    let after = sim.kv_usage();
    assert_eq!(
        after.used + after.reserved,
        started.len() as u64 * 132 * 16_384,
        "only the started requests hold KV: {after:?}"
    );

    // SURVIVAL descends once the pool drains; every request completes.
    sim.run_until_done(secs(300));
    for id in &ids {
        assert_eq!(sim.outcome(*id), Some(&Outcome::Completed), "{id:?}");
    }
    assert!(sim.ledger_idle());
}

/// One `ten_x_overload`-shaped run (10 × the service rate, prompts 64–6000, max tokens
/// 16–1024, 600 s of load then 120 s of silence) with `seed` under `survival`, and its
/// completions; checks every overload invariant of `ten_x_overload` except the goodput floor
/// and the RED requirement (both seed-dependent), plus the spec's recovery criterion: GREEN
/// and HEALTHY within 60 s of the load stopping.
fn survival_case(seed: u64, survival: SurvivalLiveness) -> (OverloadReport, usize) {
    survival_case_inner(seed, survival, None)
}

/// `survival_case` with one device OOM injected once the overload is `oom_at` under way: S-11's
/// own SURVIVAL trigger (the GREEN admission cap of S-9, amendment 2026-10-02, keeps the
/// admissions-only path below SURVIVAL).
fn survival_case_with_oom(
    seed: u64,
    survival: SurvivalLiveness,
    oom_at: Duration,
) -> (OverloadReport, usize) {
    survival_case_inner(seed, survival, Some(oom_at))
}

fn survival_case_inner(
    seed: u64,
    survival: SurvivalLiveness,
    oom_at: Option<Duration>,
) -> (OverloadReport, usize) {
    let mut cfg = OverloadConfig {
        seed,
        rate_multiple: 10.0,
        prompt_range: (64, 6000),
        max_tokens_range: (16, 1024),
        oom_once_at: oom_at,
        ..OverloadConfig::default()
    };
    cfg.reliability.recovery.survival_liveness = survival;
    let mut sim = OverloadSim::new(cfg);
    let r = sim.run_load(secs(600), secs(120));
    let completed = r
        .outcomes
        .iter()
        .filter(|o| **o == Outcome::Completed)
        .count();
    let kv = sim.kv_usage();
    eprintln!(
        "survival {} seed {seed}: {} requests, {completed} completed, states {:?}, GREEN+HEALTHY {:?} after stop, final {:?}/{:?}, kv used {} + reserved {} of {}",
        survival.as_str(),
        r.outcomes.len(),
        r.states,
        r.green_after_stop,
        sim.handle().state(),
        r.final_circuit,
        kv.used,
        kv.reserved,
        kv.capacity,
    );
    (r, completed)
}

/// The invariants every SURVIVAL liveness case must keep.
fn assert_survival_case(seed: u64, survival: SurvivalLiveness, r: &OverloadReport) {
    let label = format!("seed {seed}, {}", survival.as_str());
    assert!(
        r.max_kv_committed_plus_reserved <= r.kv_capacity_bytes,
        "{label}: KV {} > capacity {}",
        r.max_kv_committed_plus_reserved,
        r.kv_capacity_bytes
    );
    assert_eq!(r.preempted_below_survival, 0, "{label}");
    assert_eq!(r.red_growth, 0, "{label}: the admitted count grew in RED");
    assert_eq!(r.unfinished, 0, "{label}: every request reached an outcome");
    assert!(r.max_queue_len <= 256, "{label}: queue {}", r.max_queue_len);
    for o in &r.outcomes {
        match o {
            Outcome::Completed | Outcome::Cancelled => {}
            Outcome::Rejected(code) => {
                assert!(
                    REJECT_CODES.contains(&code.as_str()),
                    "{label}: code {code}"
                )
            }
            Outcome::Failed(code) => panic!("{label}: a request failed with {code}"),
        }
    }
    let back = r
        .green_after_stop
        .unwrap_or_else(|| panic!("{label}: never GREEN + HEALTHY after the load stopped"));
    assert!(
        back <= secs(60),
        "{label}: GREEN + HEALTHY {back:?} after the load stopped (criterion: 60 s)"
    );
    assert_eq!(r.final_circuit, CircuitState::Healthy, "{label}");
}

/// P3 SURVIVAL liveness regression (decision "Phase 3: SURVIVAL liveness fix", provisional A:
/// `survival_liveness: requeue_unstarted`). With seed 6 the ten-times overload once jumped
/// GREEN → SURVIVAL during the ramp and stuck there: SURVIVAL runs no prefill, so the 18
/// admitted requests that had not started and the 3 prefills in progress held their worst-case
/// KV reservations and kept `kv_utilization` at 0.934, above SURVIVAL's exit threshold
/// (0.97 − 5 %), with nothing left to release them. Option A returns the unstarted requests to
/// the admission queue without their reservations, so the pool drains and the state descends.
/// Since the GREEN admission cap (S-9, amendment 2026-10-02) bounds held + reserved at RED's
/// 0.90 — below SURVIVAL's 0.9215 exit threshold — admissions alone no longer reach SURVIVAL
/// at all (the seed 1–12 sweep peaks at RED and recovers in 43–50 s under either option), so
/// the case enters SURVIVAL the way the engine does under a real fault: one device OOM at 60 s
/// into the overload, with the pool at the cap and a full admission queue. Breaks if SURVIVAL
/// can hold reservations that nothing releases, or recovery is slower than the criterion.
#[test]
fn survival_liveness_seed_6() {
    let (r, _) = survival_case_with_oom(6, SurvivalLiveness::RequeueUnstarted, secs(60));
    assert!(
        r.states.contains(&PressureState::Survival),
        "seed 6 reaches SURVIVAL: {:?}",
        r.states
    );
    assert_survival_case(6, SurvivalLiveness::RequeueUnstarted, &r);
}

/// P3 SURVIVAL liveness regression, seed 1 (provisional A): before the fix this run recovered
/// only 64 s after the load stopped, past the spec's 60 s criterion. Breaks if recovery from
/// the overload is slower than GREEN + HEALTHY within 60 s.
#[test]
fn survival_liveness_seed_1() {
    let (r, _) = survival_case(1, SurvivalLiveness::RequeueUnstarted);
    assert_survival_case(1, SurvivalLiveness::RequeueUnstarted, &r);
}

/// The switch to option B (`survival_liveness: continue_prefills`) is live: the same two seeds
/// keep every overload invariant and recover within the criterion, and the same OOM-triggered
/// SURVIVAL entry as `survival_liveness_seed_6` works under B too. Breaks if B cannot be
/// switched in.
#[test]
fn survival_liveness_option_b() {
    for seed in [6, 1] {
        let (r, _) = survival_case(seed, SurvivalLiveness::ContinuePrefills);
        assert_survival_case(seed, SurvivalLiveness::ContinuePrefills, &r);
    }
    // The OOM-triggered entry of `survival_liveness_seed_6` under B as well: prefills in
    // progress continue in SURVIVAL at RED's budget and the state still descends.
    let (r, _) = survival_case_with_oom(6, SurvivalLiveness::ContinuePrefills, secs(60));
    assert!(
        r.states.contains(&PressureState::Survival),
        "seed 6 reaches SURVIVAL under option B: {:?}",
        r.states
    );
    assert_survival_case(6, SurvivalLiveness::ContinuePrefills, &r);
}

/// Measurement, not a gate: seeds 1–12 under both SURVIVAL liveness options, printed per seed
/// (`-- --ignored --nocapture`).
#[test]
#[ignore = "measurement: prints the SURVIVAL liveness sweep"]
fn survival_liveness_sweep() {
    for survival in [
        SurvivalLiveness::RequeueUnstarted,
        SurvivalLiveness::ContinuePrefills,
    ] {
        for seed in 1..=12 {
            let (r, completed) = survival_case(seed, survival);
            eprintln!(
                "sweep {} seed {seed}: completed {completed}, survival {}, back {:?}",
                survival.as_str(),
                r.states.contains(&PressureState::Survival),
                r.green_after_stop
            );
        }
    }
}

/// P5 S-8, decision "P5: KV admission across tensor-parallel ranks" (B), plan Task 31: a
/// 2-rank group whose rank 1 `kv` pool (2,048 blocks) is half of rank 0's (4,096; the block
/// pool holds the agreed 2,048). Under 10× overload admission never reserves beyond rank 1's
/// capacity (rank 0's ledger alone would admit up to 4,096 blocks and run the pool dry), both
/// ranks' ledgers hold the same bytes after every plan and completion (no partial reservation
/// leaks), every admitted request completes (none fails or is preempted below SURVIVAL), and
/// both ledgers are idle at the end. A client cancellation mid-flight releases on both ranks.
/// Breaks if admission reads only rank 0's ledger or a reservation, commit or release misses
/// a rank.
#[test]
fn group_reservation_unequal_pools() {
    const BLOCK_BYTES: u64 = 16_384;
    let cfg = OverloadConfig {
        seed: 7,
        rate_multiple: 10.0,
        pool_blocks: 4096,
        group_kv_blocks: vec![2048],
        ..OverloadConfig::default()
    };
    let mut sim = OverloadSim::new(cfg);
    let rank1 = sim.rank_kv_usage(1).capacity;
    assert_eq!(rank1, 2048 * BLOCK_BYTES);
    assert_eq!(sim.rank_kv_usage(0).capacity, 4096 * BLOCK_BYTES);
    let r = sim.run_load(secs(300), secs(120));
    let completed = r
        .outcomes
        .iter()
        .filter(|o| **o == Outcome::Completed)
        .count();
    eprintln!(
        "group_reservation_unequal_pools: {} requests, {completed} completed, max KV {} of rank 1's {rank1}, states {:?}",
        r.outcomes.len(),
        r.max_kv_committed_plus_reserved,
        r.states
    );
    assert!(
        r.max_kv_committed_plus_reserved <= rank1,
        "rank 0 held {} > rank 1's capacity {rank1}",
        r.max_kv_committed_plus_reserved
    );
    assert!(
        r.max_kv_committed_plus_reserved > rank1 / 2,
        "the load fills rank 1's pool: {}",
        r.max_kv_committed_plus_reserved
    );
    assert_eq!(sim.group_kv_mismatches(), 0, "both ranks hold the same KV");
    assert_eq!(r.preempted_below_survival, 0);
    assert_eq!(r.unfinished, 0, "every request reached an outcome");
    assert!(r.outcomes.len() > 500, "{} requests", r.outcomes.len());
    assert!(completed > 0);
    for o in &r.outcomes {
        match o {
            Outcome::Completed => {}
            Outcome::Rejected(code) => {
                assert!(REJECT_CODES.contains(&code.as_str()), "code {code}")
            }
            other => panic!("an admitted request did not complete: {other:?}"),
        }
    }
    assert!(sim.ledger_idle(), "both ledgers balance to zero");

    // Cancellation mid-flight: the cancelled requests' KV leaves both ranks at once.
    let ids: Vec<RequestId> = (0..8).map(|_| sim.submit_now(512, Some(512))).collect();
    for _ in 0..3 {
        sim.step_iteration();
    }
    let running = sim.running_ids();
    assert!(ids.iter().all(|id| running.contains(id)), "all 8 run");
    let held = |sim: &OverloadSim, rank| {
        let u = sim.rank_kv_usage(rank);
        (u.used, u.reserved)
    };
    let all8 = held(&sim, 0);
    assert!(all8.0 > 0 && all8.1 > 0, "{all8:?}");
    assert_eq!(held(&sim, 1), all8);
    for id in &ids[..4] {
        sim.cancel(*id);
    }
    sim.step_iteration();
    let four = held(&sim, 0);
    assert_eq!(held(&sim, 1), four);
    // 4 × (512 + 512 tokens) = 4 × 64 blocks released.
    let total = |(u, r): (u64, u64)| u + r;
    assert_eq!(total(all8) - total(four), 4 * 64 * BLOCK_BYTES);
    for id in &ids[..4] {
        assert_eq!(sim.outcome(*id), Some(&Outcome::Cancelled));
    }
    sim.run_until_done(secs(120));
    for id in &ids[4..] {
        assert_eq!(sim.outcome(*id), Some(&Outcome::Completed));
    }
    assert!(sim.ledger_idle());
    assert_eq!(sim.group_kv_mismatches(), 0);
}
