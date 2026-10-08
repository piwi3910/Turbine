//! Deterministic KV-tier simulations (P4 S-3, S-8, S-12): the real `Scheduler`, `BlockPool`
//! and `KvHierarchy` driven by `KvSimDriver` on virtual time, with payload-free memory tiers.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::{Clock, FakeClock};
use turbine_core::config::{KvConfig, ModuleName};
use turbine_core::types::{
    BlockId, DType, DeviceId, KvDtype, KvLayout, MemoryKind, ModelIdentity, PressureState,
    RequestId, StateDtype, StateLayout,
};
use turbine_kv::directory::KvBlock;
use turbine_kv::hierarchy::{HierarchyConfig, KvHierarchy, PrefetchTarget};
use turbine_kv::identity::{KvFormat, KvKey};
use turbine_kv::metrics::KvMetrics;
use turbine_kv::tier::{KvTier, MemTier, TierId};
use turbine_kv::transfer::SimTransferBackend;
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_observability::MetricsRegistry;
use turbine_scheduler::policy::{
    AdmissionInfo, AdmissionKey, CacheAwarePolicy, PreemptionRank, RunningInfo, SchedulingPolicy,
};
use turbine_scheduler::scheduler::BatchKind;
use turbine_scheduler::scheduler::IterationLimits;
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
    kv: KvConfig,
    kind: MemoryKind,
    tune: impl FnOnce(&mut SimTransferBackend),
    tune_cfg: impl FnOnce(&mut HierarchyConfig),
) -> Setup {
    setup_full(l0, l1, l2, kv, kind, tune, tune_cfg, |_| {})
}

/// [`setup_cfg`] with `tune_params` applied to the scheduler's parameters.
#[allow(clippy::too_many_arguments)]
fn setup_full(
    l0: u32,
    l1: u64,
    l2: u64,
    mut kv: KvConfig,
    kind: MemoryKind,
    tune: impl FnOnce(&mut SimTransferBackend),
    tune_cfg: impl FnOnce(&mut HierarchyConfig),
    tune_params: impl FnOnce(&mut SchedulerParams),
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
    let l0_classes = kv.ladder.enabled && kv.ladder.l0;
    let tq4_pool = kv.dtype == turbine_core::config::KvDtypeChoice::Tq4;
    let recent_window = (kv.dtype == turbine_core::config::KvDtypeChoice::Tq4
        && kv.recent_window_blocks > 0)
        .then_some(turbine_scheduler::RecentWindow {
            blocks: kv.recent_window_blocks,
            format: "bf16",
        });
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
    let mut accounting = KvLayout {
        num_layers: 0,
        num_kv_heads: 0,
        head_dim: 0,
        dtype: DType::BF16,
        block_tokens: 16,
    };
    // The recent window (S-5) needs the pool's base format to be the configured `kv.dtype`
    // (the BF16 window class is then a real class); the sim stays payload-free.
    if tq4_pool {
        // A base page above the BF16 window class's page (TurboQuant pages are record-sized,
        // 144 bytes a token and head), and the class page splits over the pool's layers; the
        // pool counts pages only.
        accounting.dtype = DType::Tq4;
        accounting.num_layers = 112;
        accounting.num_kv_heads = 8;
        accounting.head_dim = 128;
    }
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    // The L0 ladder rewrites base pages into page classes of the lossier codecs (S-7): the
    // simulated pool hosts them at the real Llama layout's page sizes.
    let layout = format().layout;
    let window_class = recent_window
        .is_some()
        .then(|| turbine_kv::pool::PageClass {
            format: "bf16",
            page_bytes: {
                let bf16 = KvLayout {
                    dtype: DType::BF16,
                    ..layout
                };
                bf16.block_bytes()
            },
        });
    let mut classes: Vec<turbine_kv::pool::PageClass> = window_class.into_iter().collect();
    let pool = if l0_classes || !classes.is_empty() {
        if l0_classes {
            classes.extend(
                turbine_kv::codec::registry()
                    .iter()
                    .filter(|c| c.name() != "l0")
                    .map(|c| turbine_kv::pool::PageClass {
                        format: c.name(),
                        page_bytes: c.bytes_per_block(&layout),
                    }),
            );
        }
        BlockPool::new(
            BlockPoolConfig {
                // A TurboQuant pool counts pages of the tq4 accounting layout (its base page
                // is the larger BF16 class's page); otherwise the real Llama layout.
                layout: if tq4_pool { accounting } else { layout },
                num_blocks: l0,
            },
            mem,
        )
        .unwrap()
        .with_page_classes(&classes, 1)
        .unwrap()
    } else {
        BlockPool::new(
            BlockPoolConfig {
                layout: accounting,
                num_blocks: l0,
            },
            mem,
        )
        .unwrap()
    };
    let mut params = SchedulerParams {
        max_running_requests: 64,
        max_batch_tokens: 2048,
        prefill_chunk_tokens: 512,
        max_queued_requests: 1024,
        chunked_prefill: true,
        block_tokens: 16,
        free_watermark: 0.01,
        max_seq_len: 8192,
        queue_timeout: Duration::from_secs(3600),
        cache_aware_window: Duration::from_secs(1),
        recent_window,
    };
    tune_params(&mut params);
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

    // A fully cached prompt prefills only `MIN_RECOMPUTE_TOKENS`: the last matched block is
    // split (P5p S-4, the amended S-4 AC — 1024 = 63 full blocks + 14 partial tokens + 2).
    d.submit(rid(100), prefix.clone(), 1);
    drain(d);
    assert_eq!(d.prefilled_tokens(rid(100)), 2);
    assert_eq!(
        d.last_estimate(rid(100)).unwrap().cached_prefix_tokens,
        1022
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
/// to prefill) and 15 generated, so the written KV ends on a block boundary and no partial
/// tail is published: all its full blocks are attached, a hit — the reuse evidence demotion
/// needs (one-off blocks are never copied down, and pressure reclaim drops them).
fn reuse(d: &mut KvSimDriver, prompts: &[Vec<u32>], first_id: u128) {
    for (i, p) in prompts.iter().enumerate() {
        let mut again = p.clone();
        again.extend([7, 8]);
        // One at a time: the pool is nearly full of the cached blocks being re-used. The same
        // session as [`fill_l0`]'s first wave, so the block each re-use commits carries the
        // reuse evidence too.
        d.submit_session(
            rid(first_id + i as u128),
            again,
            15,
            turbine_core::request::SessionHints {
                session_id: "fill-l0".into(),
                resume_within_secs: Some(3600),
                end: false,
            },
        );
        drain(d);
    }
}

/// Four distinct 96-token prompts (6 full blocks each) run to completion, each re-used once:
/// 28 cached blocks with reuse evidence (the re-use commits one more full block each — it ends
/// block-aligned on purpose, two extra tokens and 15 generated). The published-tail behaviour
/// is `partial_tail_reuse`'s subject; these pressure tests want whole blocks only, and 28
/// blocks leave headroom under the callers' 32-block pools.
fn fill_l0(d: &mut KvSimDriver) -> Vec<Vec<u32>> {
    let prompts: Vec<Vec<u32>> = (0..4u32)
        .map(|p| (p * 1000..p * 1000 + 96).collect())
        .collect();
    let hints = || turbine_core::request::SessionHints {
        session_id: "fill-l0".into(),
        resume_within_secs: Some(3600),
        end: false,
    };
    for (i, p) in prompts.iter().enumerate() {
        // Session blocks: the reuse evidence the pressure tests' demotions need (the re-use
        // pass alone no longer covers every block — see `reuse` below).
        d.submit_session(rid(i as u128 + 1), p.clone(), 1, hints());
    }
    drain(d);
    reuse(d, &prompts, 50);
    assert_eq!(d.pool().cached_unreferenced(), 28);
    prompts
}

#[test]
fn demotion_under_pressure() {
    let mut s = setup(32, 8, 16, KvConfig::default(), MemoryKind::Dedicated);
    let prompts = fill_l0(&mut s.driver);
    let d = &mut s.driver;

    // ORANGE (Phase 3 throttle plan: free cached blocks and demote down to the ORANGE
    // `kv_utilization` threshold, 0.82 of 32 blocks): unreferenced blocks are copied to L1
    // first and freed only when the copy is done. 28 cached blocks − 0.82 × 32 ≈ 1.8 → 2
    // victims per pass.
    d.set_pressure(PressureState::Orange);
    d.step();
    assert!(
        d.pool().used_blocks() <= 28,
        "the ORANGE demand never grows the pool"
    );
    for _ in 0..3 {
        d.step();
    }
    assert_eq!(s.l1.len(), 2, "L1 takes the demoted blocks");
    assert_eq!(d.pool().used_blocks(), 26);
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l1"}"#),
        2.0
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
        2.0
    );

    // Unified memory: L1 is never used; L0 demotes straight to L2.
    let mut s = setup(32, 8, 16, KvConfig::default(), MemoryKind::Unified);
    fill_l0(&mut s.driver);
    s.driver.set_pressure(PressureState::Orange);
    for _ in 0..4 {
        s.driver.step();
    }
    assert_eq!(s.l1.len(), 0, "L1 untouched on unified memory");
    assert_eq!(s.l2.len(), 2);
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l2"}"#),
        2.0
    );
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l1"}"#),
        0.0
    );
}

/// P6b S-1, S-2: L1 stores blocks in `kv.cpu.format` and L2 in `kv.nvme.format`, except the
/// last full block of the latest finished sequence (`kv.lossless_tail_blocks: 1`), which keeps
/// the L0 format through both tiers while its tag is held — the tags expire with a later finish
/// (user decision 2026-10-03, "6b: stale lossless-tail tags and the tq4 lab bound", 1 A), so an
/// earlier sequence's tail demotes encoded. Each tier accounts a copy at its codec's size, and
/// a promoted block comes back at the L0 format.
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
    // Every sequence ran 6 full blocks (96- and 98-token prompts): block 5 is its tail. Expiry
    // (user decision 2026-10-03, "6b: stale lossless-tail tags and the tq4 lab bound", 1 A):
    // only the latest finished sequence's tail keeps its tag — at most one block carries the
    // L0 format below L0; every earlier sequence's tail demoted encoded.
    let tail = |b: &KvBlock| b.token_range.start == 80;
    let mut seen = std::collections::HashMap::new();
    let mut raw: HashSet<KvKey> = HashSet::new();
    for b in d.kv().directory().iter() {
        for loc in &b.locations {
            if loc.tier != TierId::L0 {
                if loc.format == "l0" {
                    raw.insert(b.key);
                }
            } else if !tail(b) {
                assert_eq!(
                    loc.format, "l0",
                    "{:?} copy of block {:?}",
                    loc.tier, b.token_range
                );
            }
            *seen.entry((loc.tier, loc.format)).or_insert(0) += 1;
        }
    }
    assert!(
        raw.len() <= 1,
        "only the latest finished sequence's tail keeps its tag: {raw:?}"
    );
    let encoded_tail = d
        .kv()
        .directory()
        .iter()
        .filter(|b| tail(b))
        .filter(|b| {
            b.locations
                .iter()
                .any(|l| l.tier != TierId::L0 && l.format != "l0")
        })
        .count();
    assert!(
        encoded_tail > 0,
        "an expired tail demoted encoded: {seen:?}"
    );
    for (tier, format) in [(TierId::L1, "fp8_e4m3"), (TierId::L2, "tq4")] {
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
    let mut exact_chain = [false; 4];
    let mut lossy_first = [false; 4];
    let mut seen = [false; 4];
    let mut lossy_sum = 0u64;
    let mut rng = 0x2545_f491_4f6c_dd1d_u64;
    for n in 0..30u128 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let (i, opted_out) = ((rng % 4) as usize, (rng >> 8).is_multiple_of(2));
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

/// Sessions of [`lossy_multi_turn_run`] and their turns.
const MT_SESSIONS: u32 = 16;
const MT_TURNS: u32 = 8;

/// The OLMoE multi-turn shape of the 6b lower-tier gate (spec Lab: one shared prefix, short
/// turns, `--session-hints`), scaled to 16-token blocks: `MT_SESSIONS` sessions behind one
/// 96-token prefix take seeded turns (each its previous prompt plus 14–33 tokens, with 0–5
/// idle steps of think time) at YELLOW, over a 64-block L0 too small for their histories and an
/// L1 in `format` that holds all of them. Idle histories leave L0 leaf-first, a few blocks at a
/// time. Returns the cached and the lossy-cached prompt tokens over all requests.
fn lossy_multi_turn_run(format: &str, seed: u64) -> (u64, u64) {
    lossy_multi_turn_run_at(format, seed, 64, PressureState::Yellow)
}

/// [`lossy_multi_turn_run`] over an `l0_blocks`-block L0 at `pressure`. The token-granular
/// tail reuse is off: the 6b comparison prices lower-tier `tq4` against `bf16` L0 whole-block
/// reuse — a partial tail is exact-class only at `bf16` (P5p S-4), so leaving it on would
/// compare two different features.
fn lossy_multi_turn_run_at(
    format: &str,
    seed: u64,
    l0_blocks: u32,
    pressure: PressureState,
) -> (u64, u64) {
    let mut kv = KvConfig::default();
    kv.cpu.format = ModuleName::new(format).unwrap();
    let mut s = setup_cfg(
        l0_blocks,
        1024,
        0,
        kv,
        MemoryKind::Dedicated,
        |_| {},
        |cfg| {
            cfg.partial_prefix_reuse = false;
        },
    );
    let d = &mut s.driver;
    d.set_pressure(pressure);
    let prefix: Vec<u32> = (0..96).collect();
    let mut history: Vec<Vec<u32>> = vec![prefix; MT_SESSIONS as usize];
    let mut turns = vec![0u32; MT_SESSIONS as usize];
    let (mut cached, mut lossy) = (0u64, 0u64);
    let mut rng = seed | 1;
    let mut n = 0u128;
    while turns.iter().any(|t| *t < MT_TURNS) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let si = (rng % u64::from(MT_SESSIONS)) as usize;
        if turns[si] >= MT_TURNS {
            continue;
        }
        let t = turns[si];
        turns[si] += 1;
        let h = &mut history[si];
        let len = 14 + (rng >> 20) as u32 % 20;
        let base = 100_000 + si as u32 * 10_000 + t * 100;
        h.extend(base..base + len);
        n += 1;
        let id = rid(5_000 + n);
        d.submit_session(
            id,
            h.clone(),
            2,
            turbine_core::request::SessionHints {
                session_id: format!("s{si}"),
                resume_within_secs: Some(60),
                end: false,
            },
        );
        for _ in 0..((rng >> 40) % 6) {
            d.step();
        }
        drain(d);
        let rec = d.attached(id).expect("scheduled");
        cached += u64::from(rec.cached_tokens);
        lossy += u64::from(rec.lossy_tokens);
    }
    assert_eq!(d.violations(), &[] as &[String]);
    (cached, lossy)
}

/// The OLMoE multi-turn shortfall of lower-tier `tq4` (decision "6b: lower-tier tq4 after the
/// promotion fix"): with an L1 that holds every session's history, a lossy L1 caches as many
/// prompt tokens as an `l0` one. Before the fix tq4 cached 640–1,328 fewer tokens on these
/// seeds: a session's history leaves L0 leaf-first, so the first lossy block of its prefix
/// moves earlier from turn to turn, and a lookup whose lossy chain started earlier than the
/// chain of the turn that computed the later blocks (over a lossy prefix) missed them. Breaks
/// if a lossy-lineage block is reachable only through the request's own chain.
#[test]
fn lossy_multi_turn_matches_l0_reuse() {
    for seed in [1u64, 2, 4] {
        let (l0, _) = lossy_multi_turn_run("l0", seed);
        let (tq4, tq4_lossy) = lossy_multi_turn_run("tq4", seed);
        eprintln!("seed {seed}: l0 cached {l0}; tq4 cached {tq4} (lossy {tq4_lossy})");
        assert!(
            tq4_lossy > 0,
            "seed {seed}: the tq4 run reused lossy blocks"
        );
        assert!(
            tq4 >= l0,
            "seed {seed}: tq4 cached {tq4} prompt tokens, l0 {l0}"
        );
    }
}

/// The OLMoE multi-turn shortfall left after the lossy-chain fix (decision "6b: OLMoE tq4 —
/// lossless last block in eviction order", 1 A): at GREEN, where L0 is reclaimed only on
/// demand, a sequence's lossless last block was scored at the L0-format bytes it is demoted at,
/// about 3.5× its history's `tq4` retrieval cost, so it outranked every history block and whole
/// histories of other sessions drained first (tq4 up to 944 tokens short of `l0`). Scored like
/// its history for eviction order, `tq4` caches as many prompt tokens as `l0` on every L0 size
/// and seed. Breaks if the last block's eviction score prices it at its own L0-format bytes.
#[test]
fn lossy_multi_turn_green_matches_l0_reuse() {
    for l0_blocks in [64u32, 96, 128, 192] {
        for seed in [1u64, 2, 4] {
            let (l0, _) = lossy_multi_turn_run_at("l0", seed, l0_blocks, PressureState::Green);
            let (tq4, tq4_lossy) =
                lossy_multi_turn_run_at("tq4", seed, l0_blocks, PressureState::Green);
            eprintln!(
                "L0 {l0_blocks} seed {seed}: l0 cached {l0}; tq4 cached {tq4} (lossy {tq4_lossy})"
            );
            assert!(
                tq4 >= l0,
                "L0 {l0_blocks} seed {seed}: tq4 cached {tq4} prompt tokens, l0 {l0}"
            );
        }
    }
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
#[allow(clippy::too_many_arguments)]
fn ladder_run(
    trace: &LadderTrace,
    enabled: bool,
    l0: bool,
    l0_blocks: u32,
    high_water: f64,
    low_water: f64,
) -> LadderRun {
    let mut kv = KvConfig::default();
    kv.ladder.enabled = enabled;
    // The whole rung order down to `tq2`, so the trace exercises every rung and the steady-YELLOW
    // drift check means something (the default floor is `tq4` since 6b Task 16).
    kv.ladder.max_format = ModuleName::new("tq2").unwrap();
    kv.ladder.l0 = l0;
    kv.ladder.high_water = high_water;
    kv.ladder.low_water = low_water;
    let dwell = Duration::from_millis(trace.dwell_ms);
    let mut s = setup_cfg(
        l0_blocks,
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
    let mut rungs = [
        rung(d, TierId::L0),
        rung(d, TierId::L1),
        rung(d, TierId::L2),
    ];
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
            for (i, t) in [TierId::L0, TierId::L1, TierId::L2].into_iter().enumerate() {
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
    let on = ladder_run(&trace, true, false, 64, 0.95, 0.85);
    let off = ladder_run(&trace, false, false, 64, 0.95, 0.85);
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

/// P6b S-7 (`ladder_l0_under_pinned_pressure`): the same pinned trace with `kv.ladder.l0` on:
/// L0 joins the ladder as its top tier — its rung changes are pinned in
/// `fixtures/ladder_l0_expected_rungs.json`, L1/L2 keep the S-6 sequence, L0 compresses only
/// unreferenced cached blocks outside each sequence's lossless tail and recent window (the
/// referenced-block check of [`ladder_run`]), one base page is freed per compressed block,
/// and with `kv.ladder.l0: false` the run equals the S-6 run exactly (the committed rung
/// sequence and the recomputed-token count). Set `TURBINE_LADDER_BLESS=1` to rewrite the
/// expected sequence where the test runs.
#[test]
fn ladder_l0_under_pinned_pressure() {
    let trace: LadderTrace =
        serde_json::from_str(include_str!("fixtures/ladder_pressure_trace.json"))
            .expect("the committed pressure trace parses");
    // A high water of 0.80 (below the controller's deescalation target, above `low_water`
    // 0.70): at the default 0.95 the controller's reclaim keeps L0 below it, and the top tier
    // never triggers.
    let on = ladder_run(&trace, true, true, 64, 0.80, 0.70);
    let off = ladder_run(&trace, false, false, 64, 0.80, 0.70);
    eprintln!(
        "ladder l0 on: {} rewrites, {} recomputed tokens; l0 off: {} recomputed tokens",
        on.rewrites, on.recompute_tokens, off.recompute_tokens
    );
    for (c, ms) in on.changes.iter().zip(&on.change_ms) {
        eprintln!("  {c:?} at {ms} ms");
    }

    let expected_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ladder_l0_expected_rungs.json"
    );
    if std::env::var_os("TURBINE_LADDER_BLESS").is_some() {
        let json = serde_json::to_string_pretty(&on.changes).unwrap();
        std::fs::write(expected_path, json + "\n").unwrap();
    }
    // The L0 step-down is pinned exactly; L1 and L2 keep taking part. (The suite's sim attach
    // order varies per process, which shifts the L1/L2 steps and, with them, whether L0's
    // GREEN step-up lands inside the trace's GREEN tail; the step-up rule itself is pinned by
    // `ladder_under_pinned_pressure` for L1/L2 and was observed for L0 at step 5128 in the
    // blessed runs.)
    let expected: Vec<RungChange> =
        serde_json::from_str(include_str!("fixtures/ladder_l0_expected_rungs.json"))
            .expect("the committed L0 rung sequence parses");
    let l0_changes: Vec<RungChange> = on
        .changes
        .iter()
        .filter(|c| c.tier == "l0")
        .cloned()
        .collect();
    assert_eq!(
        l0_changes.first(),
        expected.first(),
        "the L0 ladder's first rung change of the pinned trace"
    );
    for c in &l0_changes {
        let rank = |f: &str| turbine_kv::codec::tier_rung(f, "bf16").unwrap_or(0);
        let d = rank(&c.to) as isize - rank(&c.from) as isize;
        assert_eq!(d.abs(), 1, "one rung at a time: {c:?}");
    }
    assert!(on.changes.len() > l0_changes.len(), "L1/L2 take part too");
    assert!(on.rewrites > 0, "the ladder rewrote copies");
    // `kv.ladder.l0: false` never touches L0: its rung stays at the base, and a rerun is
    // exactly this run (the pinned-input determinism the S-6 test asserts for L1/L2).
    assert!(
        !off.changes.iter().any(|c| c.tier == "l0"),
        "l0: false never changes the L0 rung"
    );
    let again = ladder_run(&trace, false, false, 64, 0.80, 0.70);
    assert_eq!(again.changes, off.changes);
    assert_eq!(again.recompute_tokens, off.recompute_tokens);
}

/// P6b S-5 (`recent_window_holds_newest_blocks_at_bf16`): with `kv.dtype: tq4` and
/// `kv.recent_window_blocks: 1`, a sequence's newest full block holds its BF16 window-class
/// pages while every block that left the window (nothing references it any more, and it is
/// not the lossless tail) is recompressed into the L0 base format; the pages freed and taken
/// are the two classes' (`pool.class_usage`), and a referenced block is never rewritten
/// (the dir's ref_count is checked before the rewrite starts). Fails if a window block is
/// recompressed early, if an out-of-window block stays BF16, or if a page leaks.
#[test]
fn recent_window_holds_newest_blocks_at_bf16() {
    let kv = KvConfig {
        dtype: turbine_core::config::KvDtypeChoice::Tq4,
        recent_window_blocks: 1,
        ..KvConfig::default()
    };
    let mut s = setup_cfg(8, 0, 0, kv, MemoryKind::Dedicated, |_| {}, |_| {});
    let d = &mut s.driver;
    // 48 prompt tokens = 3 full blocks; 64 generated tokens append up to 4 more.
    d.submit(rid(1), (0..48).collect(), 64);
    drain(d);
    // The window's exit conversions run after the request released its blocks; idle does not
    // wait for them.
    while d.kv().ladder_in_flight().next().is_some() {
        d.step();
    }
    assert_eq!(d.violations(), &[] as &[String]);
    let dir = d.kv().directory();
    let mut formats: Vec<(String, u32)> = Vec::new();
    for b in dir.iter() {
        // A published partial entry is L0-only and never recompressed; it holds class pages of
        // its own and is not part of the window's full-block census (P5p S-5). The sim pages at
        // 16 tokens (`format()`).
        if b.is_partial(format().layout.block_tokens) {
            continue;
        }
        if let Some(l) = b.location(TierId::L0) {
            formats.push((l.format.to_string(), b.token_range.end));
        }
    }
    formats.sort();
    eprintln!("L0 formats after the run: {formats:?}");
    let bf16 = formats.iter().filter(|(f, _)| f == "bf16").count();
    let base = formats.iter().filter(|(f, _)| f != "bf16").count();
    // P5p S-5: the run's last block now ends at token 111 (the newest token is never fed), so
    // it is a published partial entry and stays out of the full-block census — one full block
    // fewer than before.
    assert!(
        base >= 3,
        "out-of-window blocks are recompressed into the base format: {formats:?}"
    );
    assert!(
        bf16 <= 1,
        "only the lossless tail (the newest block) stays BF16: {formats:?}"
    );
    assert_eq!(bf16 + base, formats.len());
    // P6b exit (the ladder soak's `kv_idle` failure): at idle nothing is referenced, even with
    // the window's BF16 class page cached — `referenced_blocks` counts classed pages too, and
    // the pressure document's kv held bytes read it (`sync_kv_held`). Breaks if a cached
    // classed page is reported as referenced and an idle server keeps `kv_utilization` up.
    assert_eq!(
        d.pool().referenced_blocks(),
        0,
        "idle with a classed page cached: nothing referenced"
    );
}

/// `setup_cfg` with `tune_params` applied to the scheduler's parameters.
fn setup_params(
    l0: u32,
    l1: u64,
    l2: u64,
    kv: KvConfig,
    kind: MemoryKind,
    tune_params: impl FnOnce(&mut SchedulerParams),
) -> Setup {
    setup_full(l0, l1, l2, kv, kind, |_| {}, |_| {}, tune_params)
}

/// P5p S-5 (`partial_tail_reuse`): a multi-turn session whose later turns reuse the previous
/// turn's partial tail through one copy-on-write tail copy. Turn 1 (96 prompt tokens, 10
/// generated) writes 105 tokens of KV: six keyed full blocks and a 9-token partial entry.
/// Turn 2 (previous prompt + generated + a new message) attaches the six blocks by reference
/// and copies the entry's 9 tokens into one private block, planned in its first iteration:
/// `cached_prefix_tokens` is token-granular (105), only 5 prompt tokens prefill, the private
/// block counts against the reservation (only the six full blocks are excluded), shared blocks
/// are never written, and every reference returns to 0. Under RED the published entries leave
/// with `partial_l0_only` and are never demoted (L1 on). With `partial_prefix_reuse` off the
/// same run is Phase 4: block-granular attaches, no tail copies, no partial entries.
#[test]
fn partial_tail_reuse() {
    let mut s = setup(64, 8, 0, KvConfig::default(), MemoryKind::Dedicated);
    let bt = s.driver.scheduler().params().block_tokens;
    let prompt: Vec<u32> = (0..96).collect();
    let d = &mut s.driver;

    // Turn 1: 96 + 10 tokens, KV written = 105 → a 9-token partial entry under block 5.
    d.submit(rid(1), prompt.clone(), 10);
    drain(d);
    let gen_1 = d.generated_of(rid(1)).expect("turn 1 sampled").to_vec();
    assert_eq!(gen_1.len(), 10);
    let dir = d.kv().directory();
    let partials: Vec<&KvBlock> = dir.iter().filter(|b| b.is_partial(bt)).collect();
    assert_eq!(partials.len(), 1, "turn 1 published its partial tail");
    assert_eq!(partials[0].tokens.len(), 9, "the newest token is never fed");
    assert_eq!(partials[0].locations.len(), 1, "a partial entry is L0-only");
    let src = turbine_core::types::BlockId(partials[0].location(TierId::L0).unwrap().slot as u32);

    // Turn 2: previous prompt + generated + a 4-token message = 110 prompt tokens.
    let mut turn2 = prompt.clone();
    turn2.extend(&gen_1);
    turn2.extend(9_000..9_004);
    let seq2 = turbine_core::types::SeqId(1);
    d.submit(rid(2), turn2.clone(), 4);

    // One tail copy for the reusing request, in its first iteration, before its prefill;
    // no prefill of the reusing request ever starts below the reused tail.
    let mut copies = 0;
    let mut first_copy_seen = false;
    let mut wrote_below_tail = false;
    for _ in 0..10_000 {
        if d.is_idle() {
            break;
        }
        let plan = d.step();
        let seq_copies: Vec<(BlockId, BlockId)> = plan
            .tail_copies
            .iter()
            .filter(|(s, _, _)| *s == seq2)
            .map(|(_, src, dst)| (*src, *dst))
            .collect();
        copies += seq_copies.len();
        if !seq_copies.is_empty() {
            // The source reference went back with the copy's iteration — before the request
            // finished (`tail_copied`, not `request_done`).
            assert_eq!(
                d.pool().refcount(src),
                0,
                "the source reference is released after the copy's iteration"
            );
            first_copy_seen = true;
            assert!(
                plan.items
                    .iter()
                    .any(|i| i.seq == seq2 && matches!(i.kind, BatchKind::Prefill { .. })),
                "the tail copy runs with the request's first prefill chunk"
            );
            for c in &seq_copies {
                assert_eq!(c.0, src, "the tail copy reads the published entry");
            }
        }
        for item in plan.items.iter().filter(|i| i.seq == seq2) {
            if let BatchKind::Prefill { start, .. } = item.kind {
                wrote_below_tail |= start < 105;
            }
        }
    }
    assert!(first_copy_seen, "the tail copy was planned");
    assert_eq!(copies, 1, "exactly one tail copy per reusing request");
    assert!(
        !wrote_below_tail,
        "the warm prefill starts at the reused tail"
    );

    let est = d.last_estimate(rid(2)).unwrap();
    assert_eq!(
        est.cached_prefix_tokens, 105,
        "token-granular cached tokens: six blocks + the 9-token tail"
    );
    assert_eq!(est.new_prefill_tokens, 5);
    assert_eq!(d.prefilled_tokens(rid(2)), 5);
    // prompt 110 + max 4 = 114 tokens → 8 blocks, minus the six attached full blocks. The
    // private tail block is not excluded: the reservation covers it like any written block.
    assert_eq!(est.projected_kv_blocks, 8 - 6);
    assert_eq!(
        d.generated_of(rid(2)).unwrap().len(),
        4,
        "the turn completed"
    );
    assert_eq!(
        d.violations(),
        &[] as &[String],
        "shared blocks never written"
    );
    assert_eq!(d.pool().referenced_blocks(), 0, "references return to 0");
    assert_eq!(
        metric(
            &s.reg,
            r#"turbine_kv_partial_reuse_total{outcome="reused"}"#
        ),
        1.0
    );
    assert_eq!(
        metric(&s.reg, "turbine_kv_partial_reused_tokens_total"),
        9.0
    );

    // Turn 2 published its own tail; a RED reclaim drops the entries (`partial_l0_only`),
    // never demotes them.
    drain(d);
    let partials = d
        .kv()
        .directory()
        .iter()
        .filter(|b| b.is_partial(bt))
        .count();
    assert_eq!(partials, 2, "both turns published");
    let demotions = metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l1"}"#);
    d.set_pressure(PressureState::Red);
    for _ in 0..40 {
        d.step();
    }
    assert_eq!(
        metric(
            &s.reg,
            r#"turbine_kv_evictions_total{tier="l0",reason="partial_l0_only"}"#
        ),
        2.0,
        "both entries reclaimed with their reason"
    );
    assert_eq!(
        metric(&s.reg, r#"turbine_kv_demotions_total{from="l0",to="l1"}"#) - demotions,
        6.0,
        "exactly the six shared full blocks demote; the partial entries never do"
    );
    assert_eq!(s.l1.len(), 6, "L1 holds the full blocks only");
    assert!(
        !d.kv().directory().iter().any(|b| b.is_partial(bt)),
        "the entries are gone"
    );

    // With the switch off the same run is Phase 4: block-granular attach, no tail copy, no
    // partial entries. Two independently built drivers behave identically (determinism), and
    // no attach is ever partial.
    let run = |seed: u128| {
        let mut s2 = setup_cfg(
            64,
            8,
            0,
            KvConfig::default(),
            MemoryKind::Dedicated,
            |_| {},
            |cfg| cfg.partial_prefix_reuse = false,
        );
        let turn = |d: &mut KvSimDriver, prompt: &[u32], out: u32| {
            d.submit(rid(seed), prompt.to_vec(), out);
            drain(d);
            d.generated_of(rid(seed)).unwrap().to_vec()
        };
        let g1 = turn(&mut s2.driver, &prompt, 10);
        let mut t2 = prompt.clone();
        t2.extend(&g1);
        t2.extend(9_000..9_004);
        s2.driver.submit(rid(seed), t2, 4);
        let mut copies = 0;
        for _ in 0..10_000 {
            if s2.driver.is_idle() {
                break;
            }
            copies += s2.driver.step().tail_copies.len();
        }
        let est = s2.driver.last_estimate(rid(seed)).unwrap();
        let partials = s2
            .driver
            .kv()
            .directory()
            .iter()
            .filter(|b| b.is_partial(bt))
            .count();
        (copies, est.cached_prefix_tokens, partials)
    };
    let (c1, cached1, p1) = run(31);
    let (c2, cached2, p2) = run(32);
    assert_eq!(c1, 0, "no tail copies with the switch off");
    assert_eq!(c2, 0);
    assert_eq!(cached1, 96, "the six full blocks only, block-granular");
    assert_eq!(cached2, cached1, "deterministic");
    assert_eq!(p1, 0, "no partial entries published with the switch off");
    assert_eq!(p2, 0, "deterministic");
}

/// P5p S-5 (`cancellation_before_tail_copy`): a request attached with a pending tail copy but
/// never planned (the running cap is full) is cancelled — its source reference and its private
/// block are back within one iteration, and the simulation stays violation-free.
#[test]
fn cancellation_before_tail_copy() {
    let mut s = setup_params(64, 0, 0, KvConfig::default(), MemoryKind::Dedicated, |p| {
        p.max_running_requests = 1;
    });
    let bt = s.driver.scheduler().params().block_tokens;
    let prompt: Vec<u32> = (0..96).collect();
    let d = &mut s.driver;
    d.submit(rid(1), prompt.clone(), 10);
    drain(d);
    let gen_1 = d.generated_of(rid(1)).unwrap().to_vec();
    let dir = d.kv().directory();
    let partials: Vec<&KvBlock> = dir.iter().filter(|b| b.is_partial(bt)).collect();
    assert_eq!(partials.len(), 1);
    let src = turbine_core::types::BlockId(partials[0].location(TierId::L0).unwrap().slot as u32);
    let used_before = d.pool().used_blocks();

    // Turn 3 goes first and takes the single running slot; turn 2 then attaches (its tail copy
    // pending) and queues behind it — attached, never planned.
    let mut turn2 = prompt.clone();
    turn2.extend(&gen_1);
    turn2.extend(9_000..9_004);
    // An unrelated prompt: turn 3 only occupies the running slot, so turn 2's attach sees
    // turn 1's partial entry alone.
    let turn3: Vec<u32> = (50_000..50_114).collect();
    d.submit(rid(3), turn3, 4); // takes the running slot
    d.step(); // turn 3 plans and runs
    d.submit(rid(2), turn2, 4); // attaches and queues
    d.step();
    let seq2 = d.seq_of(rid(2)).expect("turn 2 tracked");
    let attached = d.attached(rid(2)).expect("turn 2 attached");
    assert_eq!(
        attached.cached_tokens, 105,
        "turn 2 attached with its tail copy pending"
    );
    d.cancel(rid(2));
    d.step();
    assert_eq!(
        d.pool().refcount(src),
        0,
        "the source reference is released within one iteration"
    );
    assert_eq!(d.seq_of(rid(2)), None, "turn 2 is gone");
    // Turn 3 runs out; nothing of turn 2 leaks: its private block was freed (a referenced-only
    // block would keep `referenced_blocks` up) and the shared blocks are back to cache only.
    drain(d);
    assert_eq!(d.violations(), &[] as &[String]);
    assert_eq!(
        d.pool().referenced_blocks(),
        0,
        "the private block went back; no reference leaked"
    );
    // Turn 3 finished: the unrelated 114-token prompt left its seven full blocks and its
    // published partial entry — nothing of turn 2 stayed.
    assert_eq!(
        d.pool().used_blocks(),
        used_before + 8,
        "only turn 3's own blocks stayed"
    );
    assert_ne!(seq2, turbine_core::types::SeqId(0), "sanity");
}

#[test]
fn partial_match_falls_back_to_block_reuse() {
    let mut s = setup(256, 0, 0, KvConfig::default(), MemoryKind::Dedicated);
    let d = &mut s.driver;
    let prefix: Vec<u32> = (0..1024).collect();
    d.submit(rid(1), prefix.clone(), 1);
    drain(d);

    // A prompt that diverges at token 500 (inside cached block 31): the full blocks before
    // it are attached, everything from the divergence is prefilled.
    let mut diverged = prefix.clone();
    diverged[500] += 1;
    d.submit(rid(2), diverged, 1);
    drain(d);
    let est = d.last_estimate(rid(2)).unwrap();
    assert_eq!(
        est.cached_prefix_tokens, 500,
        "31 full blocks + the 4 matched tokens of block 31 (divergence at 500)"
    );
    assert_eq!(d.prefilled_tokens(rid(2)), 1024 - 500);
    assert_eq!(d.violations(), &[] as &[String]);

    // A prompt shorter than the cached sequence, ending inside a cached block, reuses the
    // matched tokens of that block too: 62 full blocks + 6 of block 62, 2 recomputed.
    let short: Vec<u32> = (0..1000).collect();
    d.submit(rid(3), short, 1);
    drain(d);
    let est = d.last_estimate(rid(3)).unwrap();
    assert_eq!(est.cached_prefix_tokens, 998);
    assert_eq!(d.prefilled_tokens(rid(3)), 2);
    assert_eq!(d.violations(), &[] as &[String]);
    assert_eq!(d.pool().referenced_blocks(), 0, "references return to 0");
}

/// The test-only unbounded variant of `cache_aware` (S-8): every arrival shares one window,
/// so the affinity term orders the whole queue and the uncached requests starve behind the
/// cached flood. Not registered; the control leg of `cache_aware_starvation_bound`.
#[derive(Clone, Copy, Debug, Default)]
struct UnboundedCacheAware;

impl turbine_core::registry::Module for UnboundedCacheAware {
    fn name(&self) -> &'static str {
        "cache_aware_unbounded"
    }
}

impl turbine_scheduler::policy::SchedulingPolicy for UnboundedCacheAware {
    fn admission_key(&self, req: &AdmissionInfo) -> AdmissionKey {
        CacheAwarePolicy.admission_key(&AdmissionInfo {
            arrival: Duration::ZERO,
            window: Duration::MAX,
            ..*req
        })
    }
    fn preemption_rank(&self, running: &RunningInfo) -> PreemptionRank {
        CacheAwarePolicy.preemption_rank(running)
    }
    fn pick_victim(&self, candidates: &[(RequestId, PreemptionRank)]) -> Option<RequestId> {
        CacheAwarePolicy.pick_victim(candidates)
    }
    fn chunk_cap(&self, params: &SchedulerParams, limits: &IterationLimits) -> u32 {
        CacheAwarePolicy.chunk_cap(params, limits)
    }
}

static UNBOUNDED: UnboundedCacheAware = UnboundedCacheAware;

/// S-8: the `cache_aware` starvation bound over a saturated flood. 2,000 seeded arrivals of
/// 128-token prompts sharing a fully cached 128-token primer, one uncached request every 50
/// arrivals, `max_running_requests` 4, window 1 s: `cache_aware` completes every request with
/// an empty `admission_inversions`; the unbounded variant (all arrivals in one window)
/// accumulates inversions and a larger maximum wait for the uncached requests.
#[test]
fn cache_aware_starvation_bound() {
    struct Leg {
        inversions: Vec<(RequestId, RequestId)>,
        completed: usize,
        max_uncached_wait_s: f64,
        max_cached_wait_s: f64,
    }
    let leg = |policy: &'static dyn SchedulingPolicy| -> Leg {
        let Setup { driver, clock, .. } = setup_full(
            8192,
            0,
            0,
            KvConfig::default(),
            MemoryKind::Dedicated,
            |_| {},
            |_| {},
            |p| {
                p.max_running_requests = 4;
                p.max_queued_requests = 4096;
                p.cache_aware_window = Duration::from_secs(1);
            },
        );
        let d = &mut driver.with_policy(policy);
        const REQUESTS: u128 = 2_000;
        const MAX_TOKENS: u32 = 16;

        // The primer: a 128-token prompt every flood request shares, prefilled and finished
        // before the flood arrives, so every flood request is fully cached.
        let primer: Vec<u32> = (9_000..9_128).collect();
        let primer_id = rid(1);
        d.submit(primer_id, primer.clone(), MAX_TOKENS);
        for _ in 0..1_000 {
            if d.is_idle() {
                break;
            }
            d.step();
        }
        assert!(
            d.generated_of(primer_id).is_some(),
            "the primer completed before the flood"
        );

        // Seeded arrivals: 3-6 ms apart, ~8 s of arrivals against ~40 s of service.
        use rand_chacha::rand_core::{RngCore as _, SeedableRng as _};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(6);
        let mut dues = Vec::new();
        let mut due = Duration::ZERO;
        for _ in 0..REQUESTS {
            dues.push(due);
            due += Duration::from_millis(3 + (rng.next_u64() % 4));
        }

        let start = clock.now_mono();
        let mut submitted = 0;
        while submitted < REQUESTS as usize {
            d.step();
            let now = clock.now_mono().saturating_sub(start);
            while submitted < REQUESTS as usize && dues[submitted] <= now {
                let k = submitted as u128;
                let (prompt, _uncached) = if k % 50 == 49 {
                    let unique: Vec<u32> =
                        (20_000..20_128).map(|t| t + (k as u32) * 1_000).collect();
                    (unique, true)
                } else {
                    (primer.clone(), false)
                };
                d.submit(rid(10_000 + k), prompt, MAX_TOKENS);
                submitted += 1;
            }
        }
        for _ in 0..200_000 {
            if d.is_idle() {
                break;
            }
            d.step();
        }
        assert!(
            d.is_idle(),
            "the flood drained: queued {} running {} driver-reqs {} free blocks {}",
            d.scheduler().queue_len(),
            d.scheduler().running_ids().len(),
            d.scheduler().snapshot().waiting + d.scheduler().running_ids().len() as u32,
            d.pool().free_blocks(),
        );

        let mut completed = 0;
        for k in 0..REQUESTS {
            if d.generated_of(rid(10_000 + k)).is_some() {
                completed += 1;
            }
        }
        let mut max_uncached = 0.0f64;
        let mut max_cached = 0.0f64;
        for k in 0..REQUESTS {
            let Some((arrival, admitted)) = d.admission_span(rid(10_000 + k)) else {
                continue;
            };
            let wait = admitted.saturating_sub(arrival).as_secs_f64();
            if k % 50 == 49 {
                max_uncached = max_uncached.max(wait);
            } else {
                max_cached = max_cached.max(wait);
            }
        }
        assert!(
            d.violations().is_empty(),
            "violations: {:?}",
            d.violations()
        );
        Leg {
            inversions: d.admission_inversions(),
            completed,
            max_uncached_wait_s: max_uncached,
            max_cached_wait_s: max_cached,
        }
    };

    let cache_aware: &'static dyn turbine_scheduler::policy::SchedulingPolicy =
        turbine_scheduler::policy::registry()
            .get("cache_aware")
            .unwrap();
    let ca = leg(cache_aware);
    let un = leg(&UNBOUNDED);

    // The policy's bound: no equal-priority, never-preempted pair where a request that
    // arrived at least one window later was admitted first, and everything completes.
    assert!(
        ca.inversions.is_empty(),
        "cache_aware had {} inversions, first {:?}",
        ca.inversions.len(),
        ca.inversions.first()
    );
    assert_eq!(
        ca.completed, 2_000,
        "every request completes under cache_aware"
    );
    assert!(
        un.max_uncached_wait_s > ca.max_uncached_wait_s,
        "the unbounded variant must starve the uncached requests longer: \
         unbounded max wait {:.2}s vs cache_aware {:.2}s",
        un.max_uncached_wait_s,
        ca.max_uncached_wait_s
    );
    assert!(
        !un.inversions.is_empty(),
        "the unbounded variant must report inversions"
    );
    // Informational: the legs' shape, for the perf log.
    println!(
        "cache_aware_starvation_bound: cache_aware max uncached wait {:.2}s / max cached \
         wait {:.2}s; unbounded {:.2}s / {:.2}s; unbounded inversions {}",
        ca.max_uncached_wait_s,
        ca.max_cached_wait_s,
        un.max_uncached_wait_s,
        un.max_cached_wait_s,
        un.inversions.len()
    );
}

/// No emergency reserve to allocate (the hybrid reservation test).
struct NoReserve;

impl turbine_reliability::reserve::ReserveAllocator for NoReserve {
    fn allocate(&mut self, _bytes: u64) -> Result<(), String> {
        Ok(())
    }
    fn free(&mut self) {}
}

/// A scheduler behind an admission gate whose ledger holds `kv_pages` pages of 1 MiB and whose
/// model keeps `state_bytes` of per-sequence state (P7 S-1).
fn gated_with_state(
    kv_pages: u64,
    state_bytes: u64,
) -> (
    Scheduler,
    Arc<turbine_reliability::ledger::Ledger>,
    turbine_reliability::controller::PressureController,
) {
    use turbine_core::config::{ByteSize, ReliabilityConfig};
    use turbine_reliability::admission::{Admission, AdmissionParams, AdmissionQueue, Calibration};
    use turbine_reliability::budget::{DeviceBudget, PoolKind};
    use turbine_reliability::controller::PressureController;
    use turbine_reliability::ledger::Ledger;
    use turbine_reliability::metrics::ReliabilityMetrics;
    use turbine_reliability::reserve::EmergencyReserve;
    use turbine_reliability::throttle::SchedulerLimits;

    let page = 1u64 << 20;
    let clock: Arc<dyn Clock> = Arc::new(FakeClock::new(Duration::ZERO));
    let kv_bytes = kv_pages * page;
    let budget = DeviceBudget {
        device: DeviceId(0),
        memory_kind: MemoryKind::Dedicated,
        budget_bytes: kv_bytes,
        pools: vec![(PoolKind::Kv, kv_bytes), (PoolKind::Reserve, 0)],
    };
    let metrics = ReliabilityMetrics::unregistered();
    let ledger = Ledger::new(&budget);
    let reserve = EmergencyReserve::acquire(
        DeviceId(0),
        0,
        &ledger,
        Box::new(NoReserve),
        metrics.clone(),
    )
    .unwrap();
    let cfg = ReliabilityConfig {
        emergency_vram_reserve: ByteSize(0),
        ..ReliabilityConfig::default()
    };
    let (controller, handle) = PressureController::new(
        &cfg,
        SchedulerLimits {
            prefill_chunk_tokens: 64,
            block_tokens: 16,
        },
        budget,
        Arc::clone(&ledger),
        reserve,
        Arc::new(turbine_kv::L0Reclaimer),
        metrics.clone(),
        Arc::clone(&clock),
    );
    let admission = Admission::new(
        AdmissionParams {
            device: DeviceId(0),
            adaptive: false,
            max_queue: 16,
            large_prefill_tokens: 1 << 20,
            block_bytes: page,
            prefill_chunk_tokens: 64,
            workspace_bytes_per_token: 0,
            calibration: Calibration {
                prefill_tokens_per_s: 1000.0,
                decode_step_s: 0.01,
            },
        },
        Arc::clone(&ledger),
        metrics,
    );
    let queue = AdmissionQueue::new(16, Duration::from_secs(60), 0);
    let gate =
        turbine_scheduler::AdmissionGate::new(admission, queue, handle, Arc::clone(&clock), 4);
    let params = SchedulerParams {
        max_running_requests: 4,
        max_batch_tokens: 64,
        prefill_chunk_tokens: 64,
        max_queued_requests: 16,
        chunked_prefill: true,
        block_tokens: 16,
        free_watermark: 0.0,
        max_seq_len: 1024,
        queue_timeout: Duration::from_secs(60),
        recent_window: None,
        cache_aware_window: Duration::from_secs(1),
    };
    let state = StateLayout {
        linear_layers: 1,
        recurrent_bytes_per_layer: state_bytes,
        conv_bytes_per_layer: 0,
        dtype: Some(StateDtype::F32),
    };
    let sched = Scheduler::new(params, clock)
        .with_state_layout(state)
        .with_gate(gate);
    (sched, ledger, controller)
}

/// P7 S-1 (`hybrid_reservation_includes_state`): a request's worst-case reservation is its KV
/// pages plus its per-sequence state — 4 pages of 1 MiB and 3 MiB of state reserve exactly
/// 7 MiB (4 MiB of pages in the ledger, 3 MiB in the state allocation) — a pool too small for
/// the pages refuses it with the existing KV-capacity reason, and finishing or cancelling returns the ledger to 0. Breaks if the
/// state is left out of the reservation, the admission check or the release.
#[test]
fn hybrid_reservation_includes_state() {
    use smallvec::smallvec;
    use turbine_core::request::FinishReason;
    use turbine_reliability::budget::PoolKind;
    use turbine_scheduler::{
        CancelReason, IterationOutcome, RejectionReason, SchedRequest, SeqId, SubmitError,
    };

    let held = |l: &turbine_reliability::ledger::Ledger| {
        let u = l.usage(DeviceId(0), PoolKind::Kv);
        u.used + u.reserved
    };
    let req = |n: u64| SchedRequest::new(rid(u128::from(n)), smallvec![SeqId(n)], 32, 32, 16);

    let (mut s, ledger, _c) = gated_with_state(64, 3 << 20);
    let mut pool = BlockPool::new(
        BlockPoolConfig {
            layout: format().layout,
            num_blocks: 64,
        },
        HostMemory::new(DeviceId(0), 64 * block_bytes()) as Arc<dyn DeviceMemory>,
    )
    .unwrap();
    s.submit(req(1), 64).unwrap();
    assert_eq!(
        held(&ledger),
        4 << 20,
        "4 pages; the state is charged apart"
    );
    let res = s.reservation(rid(1)).expect("admitted with a reservation");
    assert_eq!(res.bytes(), 4 << 20);
    assert_eq!(res.state_bytes(), 3 << 20);
    // Finishing releases KV and state.
    let plan = s.plan(&mut pool, &IterationLimits::default());
    assert_eq!(plan.items.len(), 1);
    s.complete(
        &mut pool,
        IterationOutcome {
            iteration: plan.iteration,
            finished: vec![(SeqId(1), FinishReason::Stop)],
            appended: vec![(SeqId(1), 1)],
            rolled_back: Vec::new(),
            failed: None,
        },
    );
    assert_eq!(held(&ledger), 0, "finish releases the state");
    // Cancelling releases it too.
    s.submit(req(2), 64).unwrap();
    assert_eq!(held(&ledger), 4 << 20);
    s.cancel(rid(2), CancelReason::ClientDisconnect);
    let _ = s.plan(&mut pool, &IterationLimits::default());
    assert_eq!(held(&ledger), 0, "cancel releases the state");

    // 3 pages of KV do not hold the request's 4 pages.
    let (mut small, ledger, _c) = gated_with_state(3, 3 << 20);
    match small.submit(req(3), 64) {
        Err(SubmitError::Rejected { reason, .. }) => {
            assert_eq!(reason, RejectionReason::ContextExceedsKvCapacity)
        }
        other => panic!("expected a KV-capacity rejection, got {other:?}"),
    }
    assert_eq!(held(&ledger), 0);
}

/// P7 review fix 2: the ledger's `kv` capacity is the physical KV pool, and the per-sequence
/// state is charged against its own fixed allocation (`max_running × state`), not against the
/// KV pool. A 16-page pool with 3 MiB of state a sequence and 4 running slots: two 8-page
/// requests fill the pool exactly (16 MiB held, 6 MiB of state), the third waits for pages.
/// Breaks if admission counts the state against the KV pool (the pool then holds one request).
#[test]
fn state_is_charged_beside_the_kv_pool() {
    use smallvec::smallvec;
    use turbine_reliability::budget::PoolKind;
    use turbine_scheduler::{SchedRequest, SeqId};

    let (mut s, ledger, _c) = gated_with_state(16, 3 << 20);
    let req = |n: u64| SchedRequest::new(rid(u128::from(n)), smallvec![SeqId(n)], 64, 64, 16);
    for n in 1..=3 {
        s.submit(req(n), 16).unwrap();
    }
    let u = ledger.usage(DeviceId(0), PoolKind::Kv);
    assert_eq!(u.used + u.reserved, 16 << 20, "KV pages only");
    assert!(s.reservation(rid(1)).is_some());
    assert!(
        s.reservation(rid(2)).is_some(),
        "the pool holds two requests"
    );
    assert!(s.reservation(rid(3)).is_none(), "the third waits for pages");
    assert_eq!(s.gate().unwrap().queue_len(), 1);
    assert_eq!(s.reservation(rid(1)).unwrap().bytes(), 8 << 20);
    assert_eq!(s.reservation(rid(1)).unwrap().state_bytes(), 3 << 20);
}

/// P7 review fix 3: a request's state covers every sequence (`n` choices); the allocation holds
/// `max_running_requests` (4) sequences. n = 4 takes all 12 MiB, n = 5 is refused at submit with
/// `state_exceeds_capacity`, and a request that only lacks free state waits and starts once the
/// state is back. Breaks if the estimate charges one sequence's state for `n` choices.
#[test]
fn state_covers_every_sequence() {
    use turbine_scheduler::{CancelReason, SchedRequest, SeqId, SubmitError};

    let (mut s, _ledger, _c) = gated_with_state(64, 3 << 20);
    let mut pool = BlockPool::new(
        BlockPoolConfig {
            layout: format().layout,
            num_blocks: 64,
        },
        HostMemory::new(DeviceId(0), 64 * block_bytes()) as Arc<dyn DeviceMemory>,
    )
    .unwrap();
    let req = |id: u64, n: u64| {
        let seqs = (0..n).map(|i| SeqId(id * 10 + i)).collect();
        SchedRequest::new(rid(u128::from(id)), seqs, 16, 16, 16)
    };
    assert_eq!(
        s.submit(req(9, 5), 64),
        Err(SubmitError::StateExceedsCapacity)
    );
    s.submit(req(1, 4), 64).unwrap();
    assert_eq!(s.reservation(rid(1)).unwrap().state_bytes(), 12 << 20);
    s.submit(req(2, 1), 64).unwrap();
    assert!(s.reservation(rid(2)).is_none(), "no free state: it waits");
    s.cancel(rid(1), CancelReason::ClientDisconnect);
    let _ = s.plan(&mut pool, &IterationLimits::default());
    let _ = s.plan(&mut pool, &IterationLimits::default());
    assert_eq!(s.reservation(rid(2)).unwrap().state_bytes(), 3 << 20);
}
