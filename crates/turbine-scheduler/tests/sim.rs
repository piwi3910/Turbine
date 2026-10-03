//! Pipeline-parallel micro-batches in the deterministic simulator (Phase 5 S-10): overlap of
//! micro-batches across stages, cancellation while a micro-batch is in flight, and one
//! micro-batch planning exactly like the serial loop.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use smallvec::smallvec;
use turbine_core::clock::FakeClock;
use turbine_core::request::FinishReason;
use turbine_core::types::{DType, DeviceId, KvLayout};
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_scheduler::sim::{
    ArrivalProcess, CostModel, LengthMix, PipelineReport, SimArrival, SimExecutor, Simulation,
};
use turbine_scheduler::{
    BatchKind, CancelReason, IterationLimits, IterationOutcome, IterationPlan, RequestId,
    SchedRequest, Scheduler, SchedulerParams, SeqId,
};
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

/// Accounting-only pool of `n` 16-token blocks.
fn pool(n: u32) -> BlockPool {
    let layout = KvLayout {
        num_layers: 0,
        num_kv_heads: 0,
        head_dim: 0,
        dtype: DType::BF16,
        block_tokens: 16,
    };
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 0);
    BlockPool::new(
        BlockPoolConfig {
            layout,
            num_blocks: n,
        },
        mem,
    )
    .unwrap()
}

fn params() -> SchedulerParams {
    SchedulerParams {
        max_running_requests: 32,
        max_batch_tokens: 512,
        prefill_chunk_tokens: 128,
        max_queued_requests: 1024,
        chunked_prefill: true,
        block_tokens: 16,
        free_watermark: 0.01,
        max_seq_len: 4096,
        queue_timeout: Duration::from_secs(3600),
        recent_window: None,
    }
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s)
}

/// 16 long-running requests (16-token prompts, 2,000 tokens each: none finishes in the run)
/// through 2 stages with `m` micro-batches for 3 virtual seconds.
fn closed_loop_run(m: u32, cost: CostModel) -> PipelineReport {
    let arrivals = ArrivalProcess::closed_loop(SimArrival::new(Duration::ZERO, 16, 2000), 16, 16);
    let mut sim =
        Simulation::new(params(), pool(4096), SimExecutor { cost }, arrivals).with_micro_batches(m);
    sim.run_pipelined(secs(3.0), 2)
}

/// Every sequence is in at most one micro-batch in flight, and its next micro-batch enters
/// stage 0 only after the previous one left the last stage (its token was sampled).
fn assert_sequences_wait_for_their_tokens(r: &PipelineReport) {
    let mut last_exit: HashMap<u64, (u64, f64)> = HashMap::new();
    let mut by_entry = r.completed_micro_batches.clone();
    by_entry.sort_by(|a, b| {
        a.enter_s
            .total_cmp(&b.enter_s)
            .then(a.iteration.cmp(&b.iteration))
    });
    for b in &by_entry {
        for &s in &b.seqs {
            if let Some(&(it, exit)) = last_exit.get(&s) {
                assert!(
                    b.enter_s >= exit,
                    "seq {s}: iteration {} entered stage 0 at {} before iteration {it} left the \
                     last stage at {exit}",
                    b.iteration,
                    b.enter_s
                );
            }
            last_exit.insert(s, (b.iteration, b.exit_s));
        }
    }
}

/// S-10: 2 stages at 10 ms per stage-step with 2 micro-batches overlap — stage 1 runs one
/// micro-batch while stage 0 runs the other — so steady-state decode throughput is ≥ 1.8× the
/// one-micro-batch run and the bubble ratio ≤ 0.1, and no sequence runs ahead of its own
/// token. The stage cost is compute-bound (2.5 ms per sequence for the whole model): an
/// 8-sequence micro-batch costs 10 ms per stage, the 16-sequence single batch 20 ms. With a
/// batch-size-independent stage cost the split buys nothing (checked last: ratio 1.0).
/// Catches micro-batches that serialise, a sequence planned into two micro-batches in flight,
/// or a sequence's step planned before its previous token was sampled.
#[test]
fn pipeline_micro_batches_overlap() {
    let compute_bound = CostModel {
        per_prefill_token_s: 0.000_1,
        per_decode_step_s: 0.0,
        per_seq_s: 0.002_5,
    };
    let one = closed_loop_run(1, compute_bound);
    let two = closed_loop_run(2, compute_bound);
    for r in [&one, &two] {
        assert!(r.sim.violations.is_empty(), "{:?}", r.sim.violations);
        assert_sequences_wait_for_their_tokens(r);
    }
    assert_eq!(one.max_micro_batches_in_flight, 1);
    assert_eq!(two.max_micro_batches_in_flight, 2);

    // Steady state: every request is decoding; 10 ms per stage-step with two micro-batches.
    let (from, to) = (1.0, 3.0);
    let steady: Vec<_> = two
        .completed_micro_batches
        .iter()
        .filter(|b| b.enter_s >= from && b.exit_s < to)
        .collect();
    assert!(!steady.is_empty());
    for b in &steady {
        assert_eq!(b.seqs.len(), 8, "the running set splits in half: {b:?}");
    }
    let stage_steps: Vec<f64> = two.stage_busy[0]
        .iter()
        .filter(|(a, _)| *a >= from && *a < to)
        .map(|(a, b)| b - a)
        .collect();
    assert!(
        stage_steps.iter().all(|d| (d - 0.010).abs() < 1e-6),
        "{stage_steps:?}"
    );

    let ratio = two.tokens_per_s(from, to) / one.tokens_per_s(from, to);
    let bubble_two = two.bubble_ratio(from, to);
    let bubble_one = one.bubble_ratio(from, to);
    println!(
        "pipeline_micro_batches_overlap: m=1 {:.1} tok/s bubble {bubble_one:.3}; m=2 {:.1} tok/s \
         bubble {bubble_two:.3}; ratio {ratio:.3}",
        one.tokens_per_s(from, to),
        two.tokens_per_s(from, to)
    );
    assert!(ratio >= 1.8, "throughput ratio {ratio}");
    assert!(bubble_two <= 0.1, "bubble ratio {bubble_two}");
    assert!(
        bubble_one >= 0.45,
        "one micro-batch leaves each stage idle half the time"
    );

    // A stage cost that does not depend on the batch size (10 ms per stage-step, whatever the
    // micro-batch holds): halving the batch halves the tokens per step, so the throughput is
    // unchanged although the bubble is gone.
    let flat = CostModel {
        per_prefill_token_s: 0.000_1,
        per_decode_step_s: 0.020,
        per_seq_s: 0.0,
    };
    let one = closed_loop_run(1, flat);
    let two = closed_loop_run(2, flat);
    let flat_ratio = two.tokens_per_s(from, to) / one.tokens_per_s(from, to);
    println!(
        "pipeline_micro_batches_overlap (flat stage cost): ratio {flat_ratio:.3}, bubble m=2 {:.3}",
        two.bubble_ratio(from, to)
    );
    assert!(
        (0.95..=1.05).contains(&flat_ratio),
        "flat ratio {flat_ratio}"
    );
    assert!(two.bubble_ratio(from, to) <= 0.1);
}

fn rid(n: u128) -> RequestId {
    RequestId(uuid::Uuid::from_u128(n))
}

fn kinds(plan: &IterationPlan) -> Vec<(u64, BatchKind)> {
    plan.items.iter().map(|i| (i.seq.0, i.kind)).collect()
}

/// Completes `plan`: every decode and every finished prefill samples one token.
fn complete(s: &mut Scheduler, p: &mut BlockPool, plan: &IterationPlan) {
    let appended = plan
        .items
        .iter()
        .filter(|i| match i.kind {
            BatchKind::Decode => true,
            BatchKind::Prefill { start, len } => s.prefill_target(i.seq) == Some(start + len),
        })
        .map(|i| (i.seq, 1))
        .collect();
    s.complete(
        p,
        IterationOutcome {
            iteration: plan.iteration,
            appended,
            ..IterationOutcome::default()
        },
    );
}

/// S-10: a request cancelled while its micro-batch is in flight keeps its blocks until that
/// micro-batch completed — the first plan after it drops the request and frees them — and
/// the other micro-batch carries on untouched; in the simulator, a cancellation under load
/// leaves no violation and no leaked block. Catches blocks freed under an executing plan, a
/// cancellation lost, or the other micro-batch disturbed.
#[test]
fn pipeline_cancel_in_flight() {
    let clock = FakeClock::new(Duration::ZERO);
    let mut s = Scheduler::new(params(), Arc::new(clock)).with_micro_batches(2);
    let mut p = pool(64);
    for n in 1..=2u64 {
        let r = SchedRequest::new(rid(u128::from(n)), smallvec![SeqId(n)], 8, 50, 16);
        s.submit(r, p.total_blocks()).unwrap();
    }
    let lim = IterationLimits::default();
    let a = s.plan(&mut p, &lim);
    let b = s.plan(&mut p, &lim);
    assert_eq!(kinds(&a), [(1, BatchKind::Prefill { start: 0, len: 8 })]);
    assert_eq!(kinds(&b), [(2, BatchKind::Prefill { start: 0, len: 8 })]);
    complete(&mut s, &mut p, &a);
    complete(&mut s, &mut p, &b);
    let c = s.plan(&mut p, &lim);
    let d = s.plan(&mut p, &lim);
    assert_eq!(kinds(&c), [(1, BatchKind::Decode)]);
    assert_eq!(kinds(&d), [(2, BatchKind::Decode)]);

    // Request 1 is cancelled while `c` executes; `d` completes first and request 2 goes on.
    s.cancel(rid(1), CancelReason::ClientDisconnect);
    complete(&mut s, &mut p, &d);
    let e = s.plan(&mut p, &lim);
    assert!(
        e.dropped.is_empty(),
        "not dropped while its plan is in flight"
    );
    assert_eq!(kinds(&e), [(2, BatchKind::Decode)]);
    assert_eq!(p.used_blocks(), 2, "request 1 keeps its block under `c`");

    // `c` completes; the next plan drops request 1 and frees its block.
    complete(&mut s, &mut p, &c);
    assert_eq!(p.used_blocks(), 2);
    let f = s.plan(&mut p, &lim);
    assert_eq!(f.dropped, [(rid(1), CancelReason::ClientDisconnect)]);
    assert!(f.is_empty(), "request 2 is in `e`");
    assert_eq!(p.used_blocks(), 1);
    s.complete(
        &mut p,
        IterationOutcome {
            iteration: f.iteration,
            ..IterationOutcome::default()
        },
    );

    // Request 2 is untouched: it decodes position after position.
    complete(&mut s, &mut p, &e);
    let g = s.plan(&mut p, &lim);
    assert_eq!(kinds(&g), [(2, BatchKind::Decode)]);
    assert_eq!(g.items[0].block_table.tokens, 11, "8 prompt + 3 decodes");
    s.complete(
        &mut p,
        IterationOutcome {
            iteration: g.iteration,
            appended: vec![(SeqId(2), 1)],
            finished: vec![(SeqId(2), FinishReason::Stop)],
            failed: None,
        },
    );
    assert_eq!(p.used_blocks(), 0);
    assert!(s.is_idle());

    // Under load in the simulator: 8 requests over 2 micro-batches, request 0 cancelled at
    // plan 6, while the micro-batch of plan 5 (which holds it) is in flight.
    let arrivals = ArrivalProcess::closed_loop(SimArrival::new(Duration::ZERO, 16, 40), 8, 8);
    let cost = CostModel {
        per_prefill_token_s: 0.000_1,
        per_decode_step_s: 0.0,
        per_seq_s: 0.002_5,
    };
    let mut sim =
        Simulation::new(params(), pool(256), SimExecutor { cost }, arrivals).with_micro_batches(2);
    let victim = Simulation::request_id(0);
    sim.cancel_at(6, victim, CancelReason::ClientDisconnect);
    let r = sim.run_pipelined(secs(60.0), 2);
    assert!(r.sim.violations.is_empty(), "{:?}", r.sim.violations);
    assert_sequences_wait_for_their_tokens(&r);
    let seq0 = Simulation::seq_id(0, 0).0;
    let holder = r
        .completed_micro_batches
        .iter()
        .find(|b| b.iteration == 5)
        .expect("plan 5 ran");
    assert!(holder.seqs.contains(&seq0), "{holder:?}");
    let dropped_at = r
        .sim
        .iterations
        .iter()
        .find(|it| it.dropped.iter().any(|(id, _)| *id == victim))
        .expect("the cancelled request is dropped");
    assert!(
        dropped_at.iteration > 6 && dropped_at.start_s >= holder.exit_s,
        "dropped by plan {} at {}, its micro-batch left at {}",
        dropped_at.iteration,
        dropped_at.start_s,
        holder.exit_s
    );
    assert_eq!(r.sim.completed, 7);
    assert_eq!(sim.pool().used_blocks(), 0, "every block returned");
    assert!(sim.scheduler().is_idle());
}

/// A seeded Poisson mix with long prompts, preemptions (a small pool) and a cancellation.
fn serial_workload(seed: u64) -> Simulation {
    let mix = LengthMix {
        short_prompt: (8, 128),
        long_prompt: (400, 1200),
        long_fraction: 0.15,
        output: (4, 160),
        max_new_tokens: 192,
    };
    let arrivals = ArrivalProcess::poisson(30.0, seed)
        .with_mix(mix)
        .with_limit(200);
    let cost = CostModel {
        per_prefill_token_s: 0.000_2,
        per_decode_step_s: 0.02,
        per_seq_s: 0.000_5,
    };
    let mut sim = Simulation::new(params(), pool(320), SimExecutor { cost }, arrivals);
    sim.cancel_at(
        40,
        Simulation::request_id(3),
        CancelReason::ClientDisconnect,
    );
    sim
}

/// S-10 regression: with one micro-batch the pipelined driver plans exactly what the serial
/// loop plans — same batches, preemptions, drops and virtual times — whatever the stage
/// count. Catches the micro-batch machinery changing single-plan scheduling.
#[test]
fn pipeline_m1_matches_serial() {
    for seed in [1, 7, 42] {
        let serial = serial_workload(seed).run(secs(120.0));
        assert!(serial.violations.is_empty(), "{:?}", serial.violations);
        assert!(
            serial.iterations.iter().any(|it| !it.preempted.is_empty()),
            "seed {seed}: the workload preempts"
        );
        for stages in [1, 2, 3] {
            let piped = serial_workload(seed)
                .with_micro_batches(1)
                .run_pipelined(secs(120.0), stages);
            assert_eq!(
                piped.sim.plan_digest(),
                serial.plan_digest(),
                "seed {seed}, {stages} stages"
            );
            assert_eq!(piped.sim, serial, "seed {seed}, {stages} stages");
        }
    }
}
