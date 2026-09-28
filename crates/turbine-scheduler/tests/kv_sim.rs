//! Deterministic KV-tier simulations (P4 S-3, S-8, S-12): the real `Scheduler`, `BlockPool`
//! and `KvHierarchy` driven by `KvSimDriver` on virtual time, with payload-free memory tiers.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::{Clock, FakeClock};
use turbine_core::config::KvConfig;
use turbine_core::types::{
    DType, DeviceId, KvDtype, KvLayout, MemoryKind, ModelIdentity, PressureState, RequestId,
};
use turbine_kv::hierarchy::{HierarchyConfig, KvHierarchy, PrefetchTarget};
use turbine_kv::identity::KvFormat;
use turbine_kv::metrics::KvMetrics;
use turbine_kv::tier::{KvTier, MemTier, TierId};
use turbine_kv::transfer::SimTransferBackend;
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_observability::MetricsRegistry;
use turbine_scheduler::sim::KvSimDriver;
use turbine_scheduler::{Scheduler, SchedulerParams};
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

/// Llama-3.2-3B BF16: 1,835,008-byte blocks of 16 tokens.
fn format() -> KvFormat {
    KvFormat {
        dtype: KvDtype::Bf16,
        layout: KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        },
        shards: 1,
    }
}

fn block_bytes() -> u64 {
    format().layout.block_bytes()
}

struct Setup {
    driver: KvSimDriver,
    l1: Arc<MemTier>,
    l2: Arc<MemTier>,
    reg: MetricsRegistry,
    clock: FakeClock,
}

/// L0 of `l0` blocks (accounting only), payload-free L1/L2 of `l1`/`l2` blocks.
fn setup(l0: u32, l1: u64, l2: u64, kv: KvConfig, kind: MemoryKind) -> Setup {
    setup_with(l0, l1, l2, kv, kind, |_| {})
}

/// [`setup`] with `tune` applied to the simulated copy backend (path costs).
fn setup_with(
    l0: u32,
    l1: u64,
    l2: u64,
    mut kv: KvConfig,
    kind: MemoryKind,
    tune: impl FnOnce(&mut SimTransferBackend),
) -> Setup {
    // These mechanism tests page at 16 tokens (`format()`), not the 128-token default.
    kv.block_tokens = format().layout.block_tokens;
    let clock = FakeClock::new(Duration::ZERO);
    let arc: Arc<dyn Clock> = Arc::new(clock.clone());
    let bb = block_bytes();
    let l1t = Arc::new(MemTier::payload_free(TierId::L1, l1 * bb, bb, arc.clone()));
    let l2t = Arc::new(MemTier::payload_free(TierId::L2, l2 * bb, bb, arc.clone()));
    let tiers = |t: &Arc<MemTier>, n: u64| (n > 0).then(|| t.clone() as Arc<dyn KvTier>);
    let (l1d, l2d) = (tiers(&l1t, l1), tiers(&l2t, l2));
    let reg = MetricsRegistry::new();
    let kv = KvHierarchy::new(
        HierarchyConfig::from_config(&kv, bb, kind).expect("a registered eviction policy"),
        ModelIdentity {
            config_hash: [3; 32],
            weights_index_hash: [4; 32],
        },
        format(),
        l0,
        l1d.clone(),
        l2d.clone(),
        arc.clone(),
        KvMetrics::register(&reg),
    );
    // The scheduler's pool counts blocks only; tier bytes use the hierarchy's block size.
    let accounting = KvLayout {
        num_layers: 0,
        num_kv_heads: 0,
        head_dim: 0,
        dtype: DType::BF16,
        block_tokens: 16,
    };
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 0);
    let pool = BlockPool::new(
        BlockPoolConfig {
            layout: accounting,
            num_blocks: l0,
        },
        mem,
    )
    .unwrap();
    let params = SchedulerParams {
        max_running_requests: 64,
        max_batch_tokens: 2048,
        prefill_chunk_tokens: 512,
        max_queued_requests: 1024,
        chunked_prefill: true,
        block_tokens: 16,
        free_watermark: 0.01,
        max_seq_len: 8192,
        queue_timeout: Duration::from_secs(3600),
    };
    let sched = Scheduler::new(params, arc.clone());
    let mut backend = SimTransferBackend::new(arc, l1d, l2d, bb as usize);
    tune(&mut backend);
    Setup {
        driver: KvSimDriver::new(sched, pool, kv, backend, clock.clone()),
        l1: l1t,
        l2: l2t,
        reg,
        clock,
    }
}

fn metric(reg: &MetricsRegistry, series: &str) -> f64 {
    let text = reg.render().unwrap();
    text.lines()
        .find_map(|l| l.strip_prefix(series)?.trim().parse().ok())
        .unwrap_or_else(|| panic!("{series} missing from:\n{text}"))
}

fn rid(n: u128) -> RequestId {
    RequestId(uuid::Uuid::from_u128(n))
}

/// Steps until the driver is idle (bounded).
fn drain(d: &mut KvSimDriver) {
    for _ in 0..10_000 {
        if d.is_idle() {
            return;
        }
        d.step();
    }
    panic!("the simulation did not drain");
}

#[test]
fn prefix_reuse_refcounts() {
    let mut s = setup(256, 0, 0, KvConfig::default(), MemoryKind::Dedicated);
    let d = &mut s.driver;
    let prefix: Vec<u32> = (0..1024).collect();
    let prompt = |i: u32| -> Vec<u32> {
        let mut p = prefix.clone();
        p.extend(10_000 + i * 20..10_000 + i * 20 + 20);
        p
    };
    for i in 0..50 {
        d.submit(rid(u128::from(i) + 1), prompt(i), 4);
    }
    let mut max_rc = 0;
    for _ in 0..10_000 {
        if d.is_idle() {
            break;
        }
        d.step();
        let pool = d.pool();
        let rc = (0..pool.total_blocks())
            .map(|b| pool.refcount(turbine_core::types::BlockId(b)))
            .max()
            .unwrap_or(0);
        max_rc = max_rc.max(rc);
    }
    assert!(d.is_idle());
    assert_eq!(
        d.violations(),
        &[] as &[String],
        "no shared block is ever written"
    );
    let prefilled: u32 = (0..50).map(|i| d.prefilled_tokens(rid(i + 1))).sum();
    assert_eq!(
        prefilled,
        1024 + 50 * 20,
        "the shared prefix is prefilled once"
    );
    assert_eq!(
        max_rc, 50,
        "shared blocks are referenced by every running sharer"
    );
    assert_eq!(d.pool().referenced_blocks(), 0, "references return to 0");
    assert!(
        d.pool().cached_unreferenced() >= 64,
        "the prefix stays cached"
    );

    // A fully cached prompt still prefills its last block.
    d.submit(rid(100), prefix.clone(), 1);
    drain(d);
    assert_eq!(d.prefilled_tokens(rid(100)), 16);
    assert_eq!(
        d.last_estimate(rid(100)).unwrap().cached_prefix_tokens,
        1008
    );

    // A later request sharing the prefix is estimated with its cached tokens.
    d.submit(rid(101), prompt(77), 1);
    drain(d);
    let est = d.last_estimate(rid(101)).unwrap();
    assert_eq!(est.cached_prefix_tokens, 1024);
    assert_eq!(est.new_prefill_tokens, 20);
    assert_eq!(d.prefilled_tokens(rid(101)), 20);
    assert_eq!(d.violations(), &[] as &[String]);
}

/// Each prompt sent again with two more tokens (prefix reuse always leaves two prompt tokens
/// to prefill): all its full blocks are attached, a hit — the reuse evidence demotion needs
/// (one-off blocks are never copied down, and pressure reclaim drops them).
fn reuse(d: &mut KvSimDriver, prompts: &[Vec<u32>], first_id: u128) {
    for (i, p) in prompts.iter().enumerate() {
        let mut again = p.clone();
        again.extend([7, 8]);
        // One at a time: the pool is nearly full of the cached blocks being re-used.
        d.submit(rid(first_id + i as u128), again, 1);
        drain(d);
    }
}

/// Five distinct 96-token prompts (6 full blocks each) run to completion, each re-used once:
/// 30 cached blocks with reuse evidence.
fn fill_l0(d: &mut KvSimDriver) -> Vec<Vec<u32>> {
    let prompts: Vec<Vec<u32>> = (0..5u32)
        .map(|p| (p * 1000..p * 1000 + 96).collect())
        .collect();
    for (i, p) in prompts.iter().enumerate() {
        d.submit(rid(i as u128 + 1), p.clone(), 1);
    }
    drain(d);
    reuse(d, &prompts, 50);
    assert_eq!(d.pool().cached_unreferenced(), 30);
    prompts
}

#[test]
fn demotion_under_pressure() {
    let mut s = setup(32, 8, 16, KvConfig::default(), MemoryKind::Dedicated);
    let prompts = fill_l0(&mut s.driver);
    let d = &mut s.driver;

    // ORANGE (Phase 3 throttle plan: free cached blocks and demote down to the ORANGE
    // `kv_utilization` threshold, 0.82 of 32 blocks): unreferenced blocks are copied to L1
    // first and freed only when the copy is done.
    d.set_pressure(PressureState::Orange);
    d.step();
    assert_eq!(
        d.pool().used_blocks(),
        30,
        "nothing freed before its copy completes"
    );
    for _ in 0..3 {
        d.step();
    }
    assert_eq!(s.l1.len(), 4, "L1 takes the demoted blocks");
    assert_eq!(d.pool().used_blocks(), 26);
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l1"}"#),
        4.0
    );

    // RED: everything unreferenced leaves L0; L1 victims move on to L2.
    d.set_pressure(PressureState::Red);
    for _ in 0..40 {
        d.step();
    }
    assert!(!s.l2.is_empty(), "L1 victims moved to L2");
    assert!(metric(&s.reg, r#"turbine_kv_demotions_total{from="l1",to="l2"}"#) > 0.0);
    assert!(d.pool().used_blocks() < 26);

    // Back to GREEN: a request sharing a demoted prefix promotes it before its first chunk.
    d.set_pressure(PressureState::Green);
    let mut promoted = 0;
    for (i, p) in prompts.iter().enumerate() {
        let id = rid(100 + i as u128);
        let mut prompt = p.clone();
        prompt.extend(50_000..50_008);
        d.submit(id, prompt, 1);
        drain(d);
        let cached = d.last_estimate(id).unwrap().cached_prefix_tokens;
        assert_eq!(
            d.prefilled_tokens(id),
            104 - cached,
            "prefill starts after the prefix"
        );
        promoted += u32::from(cached > 0);
    }
    assert!(promoted > 0, "at least one demoted prefix came back");
    assert!(d.kv().stats().promotions > 0);
    assert_eq!(d.violations(), &[] as &[String]);

    // Blocks worth less than kv.demote_min_value are dropped instead of demoted.
    let cfg = KvConfig {
        demote_min_value: 1e9,
        ..KvConfig::default()
    };
    let mut s = setup(32, 8, 16, cfg, MemoryKind::Dedicated);
    fill_l0(&mut s.driver);
    s.driver.set_pressure(PressureState::Orange);
    for _ in 0..3 {
        s.driver.step();
    }
    assert_eq!(s.l1.len(), 0);
    assert_eq!(s.driver.pool().used_blocks(), 26);
    assert_eq!(
        metric(
            &s.reg,
            r#"turbine_kv_drops_total{reason="below_min_value"}"#
        ),
        4.0
    );

    // Unified memory: L1 is never used; L0 demotes straight to L2.
    let mut s = setup(32, 8, 16, KvConfig::default(), MemoryKind::Unified);
    fill_l0(&mut s.driver);
    s.driver.set_pressure(PressureState::Orange);
    for _ in 0..4 {
        s.driver.step();
    }
    assert_eq!(s.l1.len(), 0, "L1 untouched on unified memory");
    assert_eq!(s.l2.len(), 4);
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l2"}"#),
        4.0
    );
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l1"}"#),
        0.0
    );
}

#[test]
fn cancellation_releases_kv() {
    // At most 4 blocks in flight, so most copies are still queued when the requests go away.
    let mut cfg = KvConfig::default();
    cfg.transfer.max_inflight_bytes = turbine_core::config::ByteSize(4 * block_bytes());
    let mut s = setup(64, 8, 64, cfg, MemoryKind::Dedicated);
    let d = &mut s.driver;
    // Five 64-token prefixes, computed once and then demoted out of L0 into L1 and L2.
    let prefixes: Vec<Vec<u32>> = (0..5u32)
        .map(|p| (p * 1000..p * 1000 + 64).collect())
        .collect();
    for (i, p) in prefixes.iter().enumerate() {
        let mut prompt = p.clone();
        prompt.push(9);
        d.submit(rid(i as u128 + 1), prompt, 1);
    }
    drain(d);
    reuse(d, &prefixes, 50);
    d.set_pressure(PressureState::Red);
    for _ in 0..20 {
        d.step();
    }
    d.set_pressure(PressureState::Green);
    assert_eq!(d.pool().used_blocks(), 0, "every prefix left L0");
    assert!(
        !s.l1.is_empty() && !s.l2.is_empty(),
        "prefixes live in L1 and L2"
    );
    let baseline = d.pool().referenced_blocks();
    assert_eq!(baseline, 0);

    // 100 requests over the first four prefixes, and a prefetch of the fifth.
    let ids: Vec<RequestId> = (0..100).map(|i| rid(1000 + i)).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut prompt = prefixes[i % 4].clone();
        prompt.extend(20_000 + i as u32 * 8..20_000 + i as u32 * 8 + 8);
        d.submit(*id, prompt, 16);
    }
    d.step(); // attaches: promotions queued
    let tokens = prefixes[4].clone();
    let accepted = d
        .prefetch(PrefetchTarget::Tokens {
            prompt: &tokens,
            cache_salt: "",
        })
        .expect("the prefetch is accepted");
    assert_eq!(accepted.blocks_queued, 4);
    d.step(); // the first copies start
    let t = d.kv().transfer();
    assert!(t.inflight_bytes() > 0, "copies in flight");
    assert!(t.queued() > 0, "copies still queued");
    assert!(
        d.scheduler().is_idle(),
        "every request is still waiting on its prefix"
    );

    let cancelled: HashSet<RequestId> = ids.iter().copied().collect();
    for id in &ids {
        d.cancel(*id);
    }
    let mut scheduled = d.step().items;
    // Only copies already running on the copy stream (their target blocks free on completion)
    // and the prefetch, which no request owns, may still hold L0 blocks.
    let prefetch_held = d
        .kv()
        .document(d.pool(), (0, 0))
        .summary
        .unwrap()
        .prefetch
        .queued;
    let copying = d.kv().transfer().inflight_bytes() / block_bytes();
    assert!(
        u64::from(d.pool().referenced_blocks()) <= u64::from(baseline) + prefetch_held + copying,
        "every request reference is dropped within one iteration: {} referenced, {prefetch_held} \
         prefetching, {copying} copying",
        d.pool().referenced_blocks()
    );
    // The copies in flight land as cached blocks, never in a cancelled sequence (L2 copies
    // take a few iterations of virtual time).
    for _ in 0..10 {
        if d.pool().cached_unreferenced() > 0 {
            break;
        }
        scheduled.extend(d.step().items);
    }
    assert!(
        d.pool().cached_unreferenced() > 0,
        "copies in flight at the cancellation land as cached blocks"
    );
    for _ in 0..50 {
        scheduled.extend(d.step().items);
    }
    assert!(
        scheduled
            .iter()
            .all(|i| !cancelled.contains(&d.request_of(i.seq).unwrap_or(rid(0)))),
        "no cancelled sequence is ever scheduled"
    );
    assert!(d.is_idle());
    assert!(d.kv().transfer().is_idle());
    assert_eq!(d.pool().referenced_blocks(), baseline);
    assert_eq!(
        d.kv().prefetch_stats().outstanding(),
        4,
        "the prefetch completed"
    );
    assert_eq!(d.violations(), &[] as &[String]);
}

/// What one overload run of [`one_off_overload`] did.
#[derive(Debug)]
struct OverloadRun {
    /// L0 → lower-tier copies started.
    demotions: u64,
    /// Blocks dropped outright by pressure reclaim.
    pressure_drops: f64,
    /// Most L0 blocks pinned by in-flight demotions at one iteration boundary.
    peak_pinned: usize,
    /// Iterations that submitted at least one demotion, and the most in one iteration.
    demoting_turns: u32,
    peak_demotions_per_turn: u64,
    /// Virtual time until every request finished.
    drain: Duration,
}

/// The overload soak's shape on the simulator: a stream of one-off prompts (every prompt
/// distinct, as `turbine-bench --prompt-words-range` draws them) at a fixed pressure state,
/// with L1 copies as slow relative to a decode step as on the R9700 (a 14.7 MB Llama block at
/// ~3.8 GB/s ≈ 3.9 ms, against a 5 ms step here).
fn one_off_overload(l1_blocks: u64, state: PressureState) -> OverloadRun {
    let mut s = setup_with(
        64,
        l1_blocks,
        0,
        KvConfig::default(),
        MemoryKind::Dedicated,
        |b| {
            b.set_cost(
                turbine_kv::transfer::TransferPath::L0ToL1,
                turbine_kv::planner::PathCost {
                    latency_s: 20e-6,
                    bandwidth_bps: block_bytes() as f64 / 3.9e-3,
                },
            )
        },
    );
    let d = &mut s.driver;
    d.set_pressure(state);
    let start = s.clock.now_mono();
    let mut run = OverloadRun {
        demotions: 0,
        pressure_drops: 0.0,
        peak_pinned: 0,
        demoting_turns: 0,
        peak_demotions_per_turn: 0,
        drain: Duration::ZERO,
    };
    let mut next = 1u128;
    for step in 0..20_000 {
        // Three arrivals every other step for 120 steps: more than the 64-block pool holds.
        if step < 240 && step % 2 == 0 {
            for _ in 0..3 {
                let len = 40 + (next as u32 * 37) % 120;
                let base = next as u32 * 1_000;
                d.submit(rid(next), (base..base + len).collect(), 8);
                next += 1;
            }
        }
        if step >= 240 && d.is_idle() {
            break;
        }
        let before = d.kv().stats().demotions;
        d.step();
        let submitted = d.kv().stats().demotions - before;
        run.peak_pinned = run.peak_pinned.max(d.kv().l0_demotions_in_flight());
        if submitted > 0 {
            run.demoting_turns += 1;
            run.peak_demotions_per_turn = run.peak_demotions_per_turn.max(submitted);
        }
    }
    assert!(d.is_idle(), "the overload drains");
    assert_eq!(d.violations(), &[] as &[String]);
    run.demotions = d.kv().stats().demotions;
    run.pressure_drops = metric(&s.reg, r#"turbine_kv_drops_total{reason="pressure"}"#);
    run.drain = s.clock.now_mono() - start;
    run
}

/// Overload soak regression (provisional decision "Phase 4: pressure reclaim copies only
/// blocks with reuse evidence, bounded in flight"): at ORANGE and RED the Phase 3 controller
/// asks for every unreferenced cached block on each tick. One-off blocks carry no reuse
/// evidence, so they are dropped, as with L1 off, instead of each costing an L0 → L1 copy that
/// pins its block until done; the run takes as long as without L1.
#[test]
fn one_off_overload_does_not_demote() {
    let runs: Vec<_> = [PressureState::Orange, PressureState::Red]
        .into_iter()
        .map(|state| {
            (
                state,
                one_off_overload(4_096, state),
                one_off_overload(0, state),
            )
        })
        .collect();
    for (state, with_l1, without) in &runs {
        eprintln!("{state:?} L1 on:  {with_l1:?}");
        eprintln!("{state:?} L1 off: {without:?}");
    }
    for (state, with_l1, without) in &runs {
        assert_eq!(
            with_l1.demotions, 0,
            "{state:?}: no copy for one-off blocks"
        );
        assert_eq!(with_l1.peak_pinned, 0);
        assert_eq!(with_l1.drain, without.drain, "{state:?}: L1 costs nothing");
    }
    assert!(
        runs[1].1.pressure_drops > 0.0,
        "RED frees one-off blocks now"
    );
}
