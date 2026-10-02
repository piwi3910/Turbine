//! Deterministic KV-tier simulations (P4 S-3, S-8, S-12): the real `Scheduler`, `BlockPool`
//! and `KvHierarchy` driven by `KvSimDriver` on virtual time, with payload-free memory tiers.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::{Clock, FakeClock};
use turbine_core::config::{KvConfig, ModuleName};
use turbine_core::types::{
    DType, DeviceId, KvDtype, KvLayout, MemoryKind, ModelIdentity, PressureState, RequestId,
};
use turbine_kv::directory::KvBlock;
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
        scales: None,
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
    kv: KvConfig,
    kind: MemoryKind,
    tune: impl FnOnce(&mut SimTransferBackend),
) -> Setup {
    setup_cfg(l0, l1, l2, kv, kind, tune, |_| {})
}

/// [`setup_with`] with `tune_cfg` applied to the hierarchy's settings.
fn setup_cfg(
    l0: u32,
    l1: u64,
    l2: u64,
    mut kv: KvConfig,
    kind: MemoryKind,
    tune: impl FnOnce(&mut SimTransferBackend),
    tune_cfg: impl FnOnce(&mut HierarchyConfig),
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
    let mut cfg =
        HierarchyConfig::from_config(&kv, bb, kind).expect("a registered eviction policy");
    tune_cfg(&mut cfg);
    let kv = KvHierarchy::new(
        cfg,
        ModelIdentity {
            config_hash: [3; 32],
            weights_index_hash: [4; 32],
            rope_hash: [0; 32],
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

/// P6b S-1, S-2: L1 stores blocks in `kv.cpu.format` and L2 in `kv.nvme.format`, except the
/// last full block of each finished sequence (`kv.lossless_tail_blocks: 1`), which keeps the L0
/// format through both tiers; each tier accounts a copy at its codec's size, and a promoted
/// block comes back at the L0 format.
#[test]
fn per_tier_formats() {
    let mut kv = KvConfig::default();
    kv.cpu.format = ModuleName::new("fp8_e4m3").unwrap();
    kv.nvme.format = ModuleName::new("tq4").unwrap();
    kv.lossless_tail_blocks = 1;
    let mut s = setup(32, 8, 16, kv, MemoryKind::Dedicated);
    let prompts = fill_l0(&mut s.driver);
    let d = &mut s.driver;
    let codec_bytes = |name: &str| {
        turbine_kv::codec::registry()
            .get(name)
            .unwrap()
            .bytes_per_block(&format().layout)
    };
    let (fp8, tq4) = (codec_bytes("fp8_e4m3"), codec_bytes("tq4"));
    assert!(tq4 < fp8 && fp8 < block_bytes());

    // RED: everything unreferenced leaves L0 for L1; L1 (8 L0-format blocks) fills and its
    // victims move on to L2.
    d.set_pressure(PressureState::Red);
    for _ in 0..60 {
        d.step();
    }
    // Every sequence ran 6 full blocks (96- and 98-token prompts): block 5 is its tail.
    let tail = |b: &KvBlock| b.token_range.start == 80;
    let mut seen = std::collections::HashMap::new();
    for b in d.kv().directory().iter() {
        for loc in &b.locations {
            let want = match loc.tier {
                TierId::L0 => "l0",
                _ if tail(b) => "l0",
                TierId::L1 => "fp8_e4m3",
                _ => "tq4",
            };
            assert_eq!(
                loc.format,
                want,
                "{:?} copy of block {:?} (tail {})",
                loc.tier,
                b.token_range,
                tail(b)
            );
            *seen.entry((loc.tier, loc.format)).or_insert(0) += 1;
        }
    }
    for (tier, format) in [
        (TierId::L1, "fp8_e4m3"),
        (TierId::L2, "tq4"),
        (TierId::L2, "l0"),
    ] {
        assert!(
            seen.get(&(tier, format)).is_some_and(|n| *n > 0),
            "no {format} copy in {tier:?}: {seen:?}"
        );
    }
    assert!(
        seen.get(&(TierId::L1, "fp8_e4m3")).unwrap() + seen.get(&(TierId::L1, "l0")).unwrap_or(&0)
            > 8,
        "compressed copies let L1 hold more than 8 blocks: {seen:?}"
    );

    // Each copy is accounted at its format's size, and the tiers stay within capacity.
    for (tier, mem) in [(TierId::L1, &s.l1), (TierId::L2, &s.l2)] {
        let mut total = 0;
        for (key, size) in mem.sizes() {
            let loc = d.kv().directory().get(&key).and_then(|b| b.location(tier));
            let format = loc.expect("a stored copy is in the directory").format;
            let want = match format {
                "l0" => block_bytes(),
                f => codec_bytes(f),
            };
            assert_eq!(size, want, "{tier:?} copy in {format}");
            total += size;
        }
        assert_eq!(mem.used_bytes(), total);
        assert!(total <= mem.capacity_bytes());
    }

    // GREEN: the prefixes come back, promoted into L0 at the L0 format.
    d.set_pressure(PressureState::Green);
    let before = d.kv().stats().promotions;
    for (i, p) in prompts.iter().enumerate() {
        let id = rid(200 + i as u128);
        let mut prompt = p.clone();
        prompt.extend(60_000..60_008);
        d.submit(id, prompt, 1);
        drain(d);
    }
    assert!(
        d.kv().stats().promotions > before,
        "demoted blocks promoted"
    );
    for b in d.kv().directory().iter() {
        if let Some(loc) = b.location(TierId::L0) {
            assert_eq!(loc.format, "l0");
        }
    }
    assert_eq!(d.violations(), &[] as &[String]);
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

/// Exact keys of every full block of `prompt` under the simulation's unsalted namespace.
fn exact_keys(prompt: &[u32]) -> Vec<turbine_kv::identity::KvKey> {
    use turbine_kv::identity::{Blake3Hasher, namespace_key, prefix_keys};
    let id = ModelIdentity {
        config_hash: [3; 32],
        weights_index_hash: [4; 32],
        rope_hash: [0; 32],
    };
    let h = Blake3Hasher(namespace_key(&id, &format(), ""));
    prefix_keys(&h, prompt, 16)
}

/// P6b S-3: over an L1 that stores `tq4` (every block lossy, no lossless tail), a seeded mix
/// of opted-in and opted-out (`x-turbine-kv-lossy: deny`) requests. No opted-out request is
/// ever attached a block of lossy lineage or a lossy copy; the first one on a prompt recomputes
/// from its first lossy block and publishes an exact chain later exact lookups take; blocks
/// prefilled over a lossy prefix carry lossy keys; `lossy_tokens` counts exactly the tokens of
/// lossy blocks; and the planner retrieves a lossy block only while retrieval × (1 + penalty)
/// is cheaper than recomputing it. Breaks if a lookup ignores the opt-out.
#[test]
fn lossy_lineage_never_reaches_opted_out() {
    use turbine_kv::directory::Lineage;
    use turbine_kv::transfer::TransferPath;

    let lossy_setup = |penalty: Option<f64>| {
        let mut kv = KvConfig::default();
        kv.cpu.format = ModuleName::new("tq4").unwrap();
        kv.lossless_tail_blocks = 0;
        if let Some(p) = penalty {
            kv.lossy_penalty = Some([(ModuleName::new("tq4").unwrap(), p)].into());
        }
        let mut s = setup(128, 16, 0, kv, MemoryKind::Dedicated);
        let prompts = fill_l0(&mut s.driver);
        // RED: every unreferenced block leaves L0 for L1, where it is stored lossy.
        s.driver.set_pressure(PressureState::Red);
        for _ in 0..60 {
            s.driver.step();
        }
        s.driver.set_pressure(PressureState::Green);
        assert_eq!(s.driver.pool().cached_unreferenced(), 0);
        for b in s.driver.kv().directory().iter() {
            let formats: Vec<_> = b.locations.iter().map(|l| (l.tier, l.format)).collect();
            assert_eq!(formats, [(TierId::L1, "tq4")], "{:?}", b.token_range);
        }
        (s, prompts)
    };
    // One more full block past each prompt, and the two tokens reuse always recomputes.
    let extended = |p: &Vec<u32>, i: usize| -> Vec<u32> {
        let mut q = p.clone();
        q.extend(70_000 + i as u32 * 100..70_000 + i as u32 * 100 + 16);
        q.extend([7, 8]);
        q
    };

    let tq4_bytes = turbine_kv::codec::registry()
        .get("tq4")
        .unwrap()
        .bytes_per_block(&format().layout);
    let recompute_block = 16.0 / 8_000.0;
    let (mut s, prompts) = lossy_setup(None);
    let d = &mut s.driver;
    let mut exact_chain = [false; 5];
    let mut lossy_first = [false; 5];
    let mut seen = [false; 5];
    let mut lossy_sum = 0u64;
    let mut rng = 0x2545_f491_4f6c_dd1d_u64;
    for n in 0..30u128 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let (i, opted_out) = ((rng % 5) as usize, (rng >> 8).is_multiple_of(2));
        let id = rid(1000 + n);
        d.submit_with(id, extended(&prompts[i], i), 1, Some(!opted_out));
        drain(d);
        let rec = d.attached(id).expect("scheduled").clone();
        let est = d.kv().transfer().estimate(TransferPath::L1ToL0);
        assert!(
            est.block_seconds(tq4_bytes) * 1.5 < recompute_block,
            "tq4 (penalty 0.5) stays worth retrieving: {est:?}"
        );
        let lossy_blocks = rec
            .entries
            .iter()
            .filter(|e| e.is_some_and(|(_, l)| l.is_lossy()))
            .count() as u32;
        assert_eq!(
            rec.lossy_tokens,
            16 * lossy_blocks,
            "request {n}: lossy tokens are the lossy blocks' tokens: {rec:?}"
        );
        assert!(rec.entries.iter().all(Option::is_some), "{rec:?}");
        lossy_sum += u64::from(rec.lossy_tokens);
        if opted_out {
            assert_eq!(rec.lossy_tokens, 0, "request {n}: {rec:?}");
            assert!(
                rec.entries
                    .iter()
                    .all(|e| e.is_some_and(|(_, l)| l == Lineage::Exact)),
                "request {n}: an opted-out request got a lossy block: {rec:?}"
            );
            if !exact_chain[i] {
                assert_eq!(
                    rec.cached_tokens, 0,
                    "request {n}: recomputes from the first lossy block"
                );
                exact_chain[i] = true;
            } else {
                assert_eq!(rec.cached_tokens, 112, "request {n}: the exact chain");
            }
        } else if !exact_chain[i] {
            assert!(rec.lossy_tokens >= 96, "request {n}: reuses lossy: {rec:?}");
        } else {
            assert_eq!(
                (rec.cached_tokens, rec.lossy_tokens),
                (112, 0),
                "request {n}: an exact chain goes first"
            );
        }
        if !seen[i] {
            seen[i] = true;
            lossy_first[i] = !opted_out;
        }
    }
    assert!(
        exact_chain.iter().any(|x| *x) && lossy_first.iter().any(|x| *x),
        "the seed mixes both first requests: {exact_chain:?} {lossy_first:?}"
    );

    // The published exact chains, and the lossy lineage of blocks prefilled over lossy prefixes.
    for (i, p) in prompts.iter().enumerate() {
        let q = extended(p, i);
        let keys = exact_keys(&q);
        let dir = d.kv().directory();
        if exact_chain[i] {
            for k in &keys {
                let b = dir.get(k).expect("the exact chain is published");
                assert_eq!(b.lineage, Lineage::Exact);
                assert!(b.location(TierId::L0).is_some());
            }
        }
        if lossy_first[i] {
            let ext = &q[96..112];
            let lossy_ext: Vec<_> = dir.iter().filter(|b| *b.tokens == *ext).collect();
            assert!(
                lossy_ext
                    .iter()
                    .any(|b| b.lineage == (Lineage::Lossy { format: "tq4" }) && b.key != keys[6]),
                "block 6 of prompt {i} prefilled over a lossy prefix has a lossy key"
            );
        }
        for b in dir.iter().filter(|b| b.lineage.is_lossy()) {
            assert!(!keys.contains(&b.key), "a lossy block under an exact key");
        }
    }
    assert!(lossy_sum > 0);
    assert_eq!(
        metric(&s.reg, "turbine_kv_lossy_cached_tokens_total"),
        lossy_sum as f64
    );
    assert!(metric(&s.reg, "turbine_kv_lossy_denied_total") >= 1.0);
    assert_eq!(s.driver.violations(), &[] as &[String]);

    // At the maximum penalty a lossy retrieval costs more than recomputing: never used.
    let (mut s, prompts) = lossy_setup(Some(100.0));
    let est = s.driver.kv().transfer().estimate(TransferPath::L1ToL0);
    assert!(est.block_seconds(tq4_bytes) * 101.0 > recompute_block);
    let d = &mut s.driver;
    d.submit_with(rid(2000), extended(&prompts[0], 0), 1, Some(true));
    drain(d);
    let rec = d.attached(rid(2000)).unwrap();
    assert_eq!((rec.cached_tokens, rec.lossy_tokens), (0, 0), "{rec:?}");
    assert_eq!(metric(&s.reg, "turbine_kv_lossy_cached_tokens_total"), 0.0);
}

/// The pinned pressure trace of [`ladder_under_pinned_pressure`]
/// (`fixtures/ladder_pressure_trace.json`).
#[derive(serde::Deserialize)]
struct LadderTrace {
    /// Seeds the session each arrival belongs to.
    seed: u64,
    /// `reliability.pressure.deescalate_dwell` of the run.
    dwell_ms: u64,
    segments: Vec<LadderSegment>,
}

#[derive(serde::Deserialize)]
struct LadderSegment {
    state: PressureState,
    steps: u32,
    /// Steps between two arrivals during the segment.
    arrival_every: u32,
}

/// One change of a tier's rung for new demotions (`fixtures/ladder_expected_rungs.json`).
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct RungChange {
    step: u32,
    tier: String,
    from: String,
    to: String,
}

/// What one run of [`ladder_run`] did.
#[derive(Debug)]
struct LadderRun {
    changes: Vec<RungChange>,
    /// Virtual time (ms) of every change, in `changes` order.
    change_ms: Vec<u64>,
    /// Pressure state at every change.
    change_state: Vec<PressureState>,
    recompute_tokens: u64,
    rewrites: u64,
}

/// Sessions of the ladder workload and their turns before a session restarts.
const LADDER_SESSIONS: u32 = 6;
const LADDER_TURNS: u32 = 10;

/// The seeded multi-turn workload of [`ladder_under_pinned_pressure`] over the pinned pressure
/// trace, with the ladder on or off: every `arrival_every` steps a seeded session sends its next
/// turn (its previous prompt plus 32 tokens; 48-token session bases, three bases per session
/// taking turns, so older prefixes come back after they left L0). L1 (32 blocks) and L2 (150
/// blocks) are too small for the workload's prefixes at the L0 format: both tiers pass low
/// water within the YELLOW stretch, and both fill under ORANGE and RED. Checks, every step,
/// that each ladder tick window starts ≥ 50 ms after the previous one and holds at most 32
/// rewrites, and that no rewrite starts on a block a running request references.
fn ladder_run(trace: &LadderTrace, enabled: bool) -> LadderRun {
    let mut kv = KvConfig::default();
    kv.ladder.enabled = enabled;
    // The whole rung order down to `tq2`, so the trace exercises every rung and the steady-YELLOW
    // drift check means something (the default floor is `tq4` since 6b Task 16).
    kv.ladder.max_format = ModuleName::new("tq2").unwrap();
    let dwell = Duration::from_millis(trace.dwell_ms);
    let mut s = setup_cfg(
        64,
        32,
        150,
        kv,
        MemoryKind::Dedicated,
        |_| {},
        |cfg| {
            if let Some(l) = cfg.ladder.as_mut() {
                l.dwell = dwell;
            }
        },
    );
    let d = &mut s.driver;
    let mut rng = trace.seed;
    let mut next = |bound: u32| {
        // SplitMix64: a fixed, dependency-free sequence.
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % u64::from(bound)) as u32
    };
    let mut turns = vec![0u32; LADDER_SESSIONS as usize];
    let mut epochs = vec![0u32; LADDER_SESSIONS as usize];
    let mut id = 1u128;
    let rung = |d: &KvSimDriver, t: TierId| d.kv().ladder_rung(t).unwrap_or("-");
    let mut rungs = [rung(d, TierId::L1), rung(d, TierId::L2)];
    let mut run = LadderRun {
        changes: Vec::new(),
        change_ms: Vec::new(),
        change_state: Vec::new(),
        recompute_tokens: 0,
        rewrites: 0,
    };
    let (mut ticks, mut window_rewrites) = (d.kv().stats().ladder_ticks, 0u64);
    let mut last_tick_ms: Option<u64> = None;
    let mut in_flight: HashSet<turbine_kv::identity::KvKey> = HashSet::new();
    let mut step = 0u32;
    for seg in &trace.segments {
        d.set_pressure(seg.state);
        for _ in 0..seg.steps {
            if step.is_multiple_of(seg.arrival_every) {
                let sess = next(LADDER_SESSIONS) as usize;
                let base = (sess as u32 * 3 + epochs[sess] % 3) * 10_000;
                let len = 48 + 32 * turns[sess];
                d.submit(rid(id), (base..base + len).collect(), 2);
                id += 1;
                turns[sess] += 1;
                if turns[sess] == LADDER_TURNS {
                    turns[sess] = 0;
                    epochs[sess] += 1;
                }
            }
            let rewrites_before = d.kv().stats().compressions;
            d.step();
            step += 1;
            let now_ms = s.clock.now_mono().as_millis() as u64;
            let stats = d.kv().stats();
            if stats.ladder_ticks > ticks {
                assert_eq!(stats.ladder_ticks, ticks + 1, "one ladder tick per step");
                if let Some(t) = last_tick_ms {
                    assert!(
                        now_ms - t >= 50,
                        "ladder ticks ≥ 50 ms apart: {t} → {now_ms}"
                    );
                }
                last_tick_ms = Some(now_ms);
                ticks = stats.ladder_ticks;
                window_rewrites = 0;
            }
            window_rewrites += stats.compressions - rewrites_before;
            assert!(
                window_rewrites <= 32,
                "at most 32 rewrites per ladder tick (step {step}: {window_rewrites})"
            );
            // A rewrite never starts on a block a running request references.
            let now_in_flight: HashSet<_> = d.kv().ladder_in_flight().map(|(k, _)| *k).collect();
            for key in now_in_flight.difference(&in_flight) {
                let b = d
                    .kv()
                    .directory()
                    .get(key)
                    .expect("a rewritten block is known");
                let referenced = b.location(TierId::L0).is_some_and(|l| {
                    d.pool()
                        .refcount(turbine_core::types::BlockId(l.slot as u32))
                        > 0
                });
                assert!(!referenced, "step {step}: rewrite of a referenced block");
            }
            in_flight = now_in_flight;
            for (i, t) in [TierId::L1, TierId::L2].into_iter().enumerate() {
                let now = rung(d, t);
                if now != rungs[i] {
                    run.changes.push(RungChange {
                        step,
                        tier: t.as_str().into(),
                        from: rungs[i].into(),
                        to: now.into(),
                    });
                    run.change_ms.push(now_ms);
                    run.change_state.push(seg.state);
                    rungs[i] = now;
                }
            }
        }
    }
    drain(d);
    assert_eq!(d.violations(), &[] as &[String]);
    run.recompute_tokens = d.kv().stats().recompute_tokens;
    run.rewrites = d.kv().stats().compressions;
    run
}

/// P6b S-6 (`ladder_under_pinned_pressure`): the committed pressure trace (GREEN → YELLOW →
/// ORANGE → RED → GREEN) over the seeded multi-turn workload with small L1/L2 yields exactly
/// the committed sequence of rung changes; at most 32 rewrites per ladder tick, ticks ≥ 50 ms
/// apart (checked every step by [`ladder_run`]); a steady YELLOW never walks a tier to `tq2`;
/// back at GREEN no rewrite starts; a rung steps back up only while the pinned state is GREEN
/// and only after `deescalate_dwell` below low water (no oscillation within the dwell; user
/// decision 2026-09-30, option A); no referenced block is rewritten; and the ladder
/// recomputes fewer prompt tokens than the same run with `kv.ladder.enabled: false`, which never
/// changes a rung. Set `TURBINE_LADDER_BLESS=1` to rewrite the expected sequence where the test
/// runs (review it; `scripts/remote-cargo.sh` does not forward it, the printed sequence does).
///
/// Breaks if the ladder's behaviour depends on anything but the pinned inputs, if a rung is
/// stepped up before its dwell or while the pressure is not GREEN, if YELLOW drifts to `tq2`, or if rewrites continue at GREEN.
#[test]
fn ladder_under_pinned_pressure() {
    let trace: LadderTrace =
        serde_json::from_str(include_str!("fixtures/ladder_pressure_trace.json"))
            .expect("the committed pressure trace parses");
    let on = ladder_run(&trace, true);
    let off = ladder_run(&trace, false);
    eprintln!(
        "ladder on: {} rewrites, {} recomputed tokens; off: {} recomputed tokens",
        on.rewrites, on.recompute_tokens, off.recompute_tokens
    );
    for (c, ms) in on.changes.iter().zip(&on.change_ms) {
        eprintln!("  {c:?} at {ms} ms");
    }

    let expected_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ladder_expected_rungs.json"
    );
    if std::env::var_os("TURBINE_LADDER_BLESS").is_some() {
        let json = serde_json::to_string_pretty(&on.changes).unwrap();
        std::fs::write(expected_path, json + "\n").unwrap();
    }
    let expected: Vec<RungChange> =
        serde_json::from_str(include_str!("fixtures/ladder_expected_rungs.json"))
            .expect("the committed rung sequence parses");
    assert_eq!(
        on.changes, expected,
        "the rung sequence of the pinned trace"
    );

    assert!(on.rewrites > 0, "the ladder rewrote copies");
    assert!(
        off.changes.is_empty() && off.rewrites == 0,
        "ladder off: no rung"
    );
    assert!(
        on.recompute_tokens < off.recompute_tokens,
        "the ladder keeps more reuse: {} vs {} recomputed tokens",
        on.recompute_tokens,
        off.recompute_tokens
    );

    let rung_index = |f: &str| turbine_kv::codec::rung_index(f).unwrap_or(0);
    let mut last_change: std::collections::HashMap<&str, u64> = Default::default();
    let mut stepped_up = false;
    for ((c, ms), state) in on.changes.iter().zip(&on.change_ms).zip(&on.change_state) {
        let down = rung_index(&c.to) > rung_index(&c.from);
        match state {
            PressureState::Yellow => {
                assert_ne!(c.to, "tq2", "a steady YELLOW never drifts to tq2: {c:?}")
            }
            PressureState::Green => {
                assert!(!down, "no rung goes down at GREEN: {c:?}")
            }
            _ => {}
        }
        if !down {
            // User decision 2026-09-30 (option A): a rung relaxes only at GREEN.
            assert_eq!(
                *state,
                PressureState::Green,
                "{c:?}: rung_step_up while the pinned state is not GREEN"
            );
            stepped_up = true;
            if let Some(prev) = last_change.get(c.tier.as_str()) {
                assert!(
                    ms - prev >= trace.dwell_ms,
                    "{c:?}: stepped up {} ms after the previous change (dwell {} ms)",
                    ms - prev,
                    trace.dwell_ms
                );
            }
        }
        last_change.insert(c.tier.as_str(), *ms);
    }
    assert!(stepped_up, "back at GREEN a rung steps back up");
    assert!(
        on.changes.iter().any(|c| c.tier == "l1") && on.changes.iter().any(|c| c.tier == "l2"),
        "both tiers take part: {:?}",
        on.changes
    );
}
