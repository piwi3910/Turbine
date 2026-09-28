//! `turbine-bench kv-sim` (P4 S-7, S-16): replays seeded synthetic workloads through the real
//! `turbine_kv` identity, directory, policy, planner and transfer engine (via `KvHierarchy`)
//! against payload-free in-memory tiers on a fake clock. No GPU, no sleeps; equal seeds give
//! equal reports.

use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use serde::Serialize;
use turbine_core::clock::{Clock, FakeClock};
use turbine_core::config::{KvConfig, ModuleName};
use turbine_core::request::SessionHints;
use turbine_core::types::{
    DType, DeviceId, KvDtype, KvLayout, MemoryKind, ModelIdentity, Priority, RequestId,
};
use turbine_kv::hierarchy::{
    AttachOutcome, AttachRequest, HierarchyConfig, KvHierarchy, PrefixAttach,
};
use turbine_kv::identity::KvFormat;
use turbine_kv::metrics::{EvictReason, KvMetrics};
use turbine_kv::tier::{KvTier, MemTier, TierId};
use turbine_kv::transfer::SimTransferBackend;
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Workload {
    MultiTurn,
    SharedSystem,
    Mixed,
}

/// `--policy`: any registered eviction policy (`turbine_kv::policy::registry`), so a policy
/// added under `docs/extending/eviction-policy.md` is comparable here without a code change.
fn policy_arg(name: &str) -> Result<&'static str, String> {
    let reg = turbine_kv::policy::registry();
    reg.get(name)
        .map(|p| turbine_core::registry::Module::name(p))
        .ok_or_else(|| reg.unknown(name).to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SimOutput {
    Text,
    Json,
}

/// `turbine-bench kv-sim …` (contract §19).
#[derive(Clone, Debug, Parser)]
#[command(
    name = "turbine-bench kv-sim",
    about = "Offline KV policy simulator: replays a synthetic workload through the KV hierarchy"
)]
pub struct KvSimArgs {
    #[arg(long, value_enum)]
    pub workload: Workload,
    /// A registered eviction policy (`cost_aware`, `lru`, …).
    #[arg(long, value_parser = policy_arg, default_value = "cost_aware")]
    pub policy: &'static str,
    /// L0 (GPU) blocks.
    #[arg(long)]
    pub l0_blocks: u32,
    /// L1 (pinned host) blocks; 0 disables L1.
    #[arg(long, default_value_t = 0)]
    pub l1_blocks: u64,
    /// L2 (NVMe) blocks; 0 disables L2.
    #[arg(long, default_value_t = 0)]
    pub l2_blocks: u64,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    #[arg(long, value_enum, default_value_t = SimOutput::Text)]
    pub output: SimOutput,
}

/// Share of looked-up prompt blocks served by each tier, or missed.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct HitRateByTier {
    pub l0: f64,
    pub l1: f64,
    pub l2: f64,
    pub miss: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct KvSimReport {
    pub workload: Workload,
    pub policy: &'static str,
    pub requests: u64,
    pub hit_rate_by_tier: HitRateByTier,
    pub recompute_tokens: u64,
    pub transfer_bytes: u64,
    pub evictions: u64,
    /// Recomputed prompt tokens at 8,000 tok/s plus time spent waiting for promotions.
    pub simulated_prefill_seconds: f64,
}

impl KvSimReport {
    pub fn to_text(&self) -> String {
        let h = &self.hit_rate_by_tier;
        format!(
            "workload {:?} policy {} requests {}\n\
             hit rate l0 {:.3} l1 {:.3} l2 {:.3} miss {:.3}\n\
             recompute_tokens {}\ntransfer_bytes {}\nevictions {}\n\
             simulated_prefill_seconds {:.3}\n",
            self.workload,
            self.policy,
            self.requests,
            h.l0,
            h.l1,
            h.l2,
            h.miss,
            self.recompute_tokens,
            self.transfer_bytes,
            self.evictions,
            self.simulated_prefill_seconds
        )
    }
}

/// xorshift64*: deterministic on every platform.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn tokens(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| 1000 + self.below(120_000) as u32).collect()
    }
}

/// One synthetic request arriving at virtual time `at`.
pub struct SimRequest {
    pub at: Duration,
    pub prompt: Vec<u32>,
    pub output: Vec<u32>,
    pub session: Option<SessionHints>,
}

/// Prefill rate of the simulated device.
const PREFILL_TPS: f64 = 8_000.0;
/// Decode time per output token.
const DECODE_STEP: Duration = Duration::from_millis(5);
/// Reclaim to this L0 utilisation once it reaches `RECLAIM_AT` (Phase 3 YELLOW / ORANGE).
const RECLAIM_TO: f64 = 0.70;
const RECLAIM_AT: f64 = 0.82;
/// Virtual time step while waiting for promotions.
const WAIT_STEP: Duration = Duration::from_micros(100);

/// Agentic sessions: `concurrent` sessions live at once, each runs 6–12 turns with 20–180 s of
/// think time and session hints (resume-within 200 s, end on the last turn); a finished session
/// is replaced until `total` sessions ran. A turn is the shared prefix + history + 200–1,200 new
/// tokens, answered with 100–300 tokens.
fn multi_turn(rng: &mut Rng, total: usize, concurrent: usize, shared: &[u32]) -> Vec<SimRequest> {
    let mut out = Vec::new();
    let mut slot_free = vec![Duration::ZERO; concurrent];
    for s in 0..total {
        let slot = (0..concurrent)
            .min_by_key(|i| slot_free[*i])
            .expect("at least one concurrent session");
        let mut at = slot_free[slot] + Duration::from_millis(rng.below(5_000));
        let turns = 6 + rng.below(7) as usize;
        let mut history = shared.to_vec();
        for t in 0..turns {
            let n = 200 + rng.below(1000) as usize;
            let mut prompt = history.clone();
            prompt.extend(rng.tokens(n));
            let n = 100 + rng.below(200) as usize;
            let output = rng.tokens(n);
            history.clone_from(&prompt);
            history.extend(&output);
            let session = SessionHints {
                session_id: format!("s{s}"),
                resume_within_secs: Some(200),
                end: t + 1 == turns,
            };
            out.push(SimRequest {
                at,
                prompt,
                output,
                session: Some(session),
            });
            at += Duration::from_secs(20 + rng.below(160));
        }
        slot_free[slot] = at;
    }
    out
}

/// `n` requests over Zipf-weighted 1,024-token system prompts plus a 32–128-token suffix.
fn shared_system(
    rng: &mut Rng,
    n: usize,
    systems: &[Vec<u32>],
    spacing_ms: u64,
) -> Vec<SimRequest> {
    let weights: Vec<f64> = (0..systems.len()).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let sum: f64 = weights.iter().sum();
    let mut at = Duration::ZERO;
    (0..n)
        .map(|_| {
            let r = rng.below(1_000_000) as f64 / 1_000_000.0 * sum;
            let mut acc = 0.0;
            let mut pick = systems.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                acc += w;
                if r < acc {
                    pick = i;
                    break;
                }
            }
            let mut prompt = systems[pick].clone();
            let n = 32 + rng.below(96) as usize;
            prompt.extend(rng.tokens(n));
            at += Duration::from_millis(spacing_ms / 2 + rng.below(spacing_ms));
            SimRequest {
                at,
                prompt,
                output: rng.tokens(64),
                session: None,
            }
        })
        .collect()
}

/// One-off 2,048–4,096-token documents, never reused (scan traffic).
fn scans(rng: &mut Rng, n: usize, spacing_ms: u64) -> Vec<SimRequest> {
    let mut at = Duration::ZERO;
    (0..n)
        .map(|_| {
            at += Duration::from_millis(spacing_ms / 2 + rng.below(spacing_ms));
            let n = 2048 + rng.below(2048) as usize;
            SimRequest {
                at,
                prompt: rng.tokens(n),
                output: rng.tokens(32),
                session: None,
            }
        })
        .collect()
}

/// The requests of a workload, merged by arrival time.
pub fn workload(w: Workload, seed: u64) -> Vec<SimRequest> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let shared = rng.tokens(512);
    let systems: Vec<Vec<u32>> = (0..8).map(|_| rng.tokens(1024)).collect();
    let mut out = match w {
        Workload::MultiTurn => multi_turn(&mut rng, 160, 24, &shared),
        Workload::SharedSystem => shared_system(&mut rng, 1500, &systems, 400),
        Workload::Mixed => {
            let mut v = multi_turn(&mut rng, 80, 12, &shared);
            v.extend(shared_system(&mut rng, 800, &systems, 2_000));
            v.extend(scans(&mut rng, 300, 5_000));
            v
        }
    };
    out.sort_by_key(|r| r.at);
    out
}

/// The simulated engine: the hierarchy over an accounting-only L0 pool.
struct Engine {
    clock: FakeClock,
    h: KvHierarchy,
    pool: BlockPool,
    backend: SimTransferBackend,
}

impl Engine {
    fn new(args: &KvSimArgs) -> Engine {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let kv = KvConfig {
            policy: ModuleName::new(args.policy).expect("a registered name is a module name"),
            ..KvConfig::default()
        };
        // Llama-3.2-3B BF16 blocks at the default page size (`kv.block_tokens`, 128) for tier
        // sizes and copy times.
        let layout = KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: kv.block_tokens,
        };
        let bb = layout.block_bytes();
        let tier = |id, blocks: u64| {
            (blocks > 0).then(|| {
                Arc::new(MemTier::payload_free(id, blocks * bb, bb, arc.clone())) as Arc<dyn KvTier>
            })
        };
        let (l1, l2) = (
            tier(TierId::L1, args.l1_blocks),
            tier(TierId::L2, args.l2_blocks),
        );
        let mut h = KvHierarchy::new(
            HierarchyConfig::from_config(&kv, bb, MemoryKind::Dedicated)
                .expect("--policy names a registered eviction policy"),
            ModelIdentity {
                config_hash: [7; 32],
                weights_index_hash: [9; 32],
            },
            KvFormat {
                dtype: KvDtype::Bf16,
                layout,
                shards: 1,
                scales: None,
            },
            args.l0_blocks,
            l1.clone(),
            l2.clone(),
            arc.clone(),
            KvMetrics::unregistered(),
        );
        h.set_prefill_tps(PREFILL_TPS);
        // Zero bytes per block: the pool only counts blocks.
        let accounting = KvLayout {
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            ..layout
        };
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 0);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: accounting,
                num_blocks: args.l0_blocks,
            },
            mem,
        )
        .expect("a zero-byte pool always fits");
        let backend = SimTransferBackend::new(arc, l1, l2, 0);
        Engine {
            clock,
            h,
            pool,
            backend,
        }
    }

    /// Attaches `req`, advancing virtual time while it waits for promotions or for a block
    /// already on its way into L0; returns the attach and the time waited.
    fn attach(&mut self, req: &AttachRequest<'_>) -> (PrefixAttach, Duration) {
        let started = self.clock.now_mono();
        loop {
            match self.h.attach_prefix(&mut self.pool, req) {
                AttachOutcome::Ready(a) => return (a, self.clock.now_mono() - started),
                AttachOutcome::Promoting => loop {
                    self.clock.advance(WAIT_STEP);
                    let ready = self.h.poll(&mut self.pool, &mut self.backend);
                    if let Some((_, a)) = ready.into_iter().find(|(r, _)| *r == req.request) {
                        return (a, self.clock.now_mono() - started);
                    }
                },
                AttachOutcome::WaitForPrefix => {
                    self.clock.advance(WAIT_STEP);
                    self.h.poll(&mut self.pool, &mut self.backend);
                }
            }
        }
    }
}

/// Replays the workload one request at a time: attach, prefill the rest, decode, commit, then
/// reclaim when L0 reaches `RECLAIM_AT`.
pub fn run(args: &KvSimArgs) -> KvSimReport {
    let mut e = Engine::new(args);
    let bt = e.pool.layout().block_tokens as usize;
    let mut prefill_seconds = 0.0;
    let reqs = workload(args.workload, args.seed);
    for (i, r) in reqs.iter().enumerate() {
        let id = RequestId(uuid::Uuid::from_u128(i as u128 + 1));
        if e.clock.now_mono() < r.at {
            e.clock.set(r.at);
        }
        e.h.poll(&mut e.pool, &mut e.backend);
        e.h.tick(&mut e.pool);
        let req = AttachRequest {
            request: id,
            prompt: &r.prompt,
            cache_salt: "",
            session: r.session.as_ref(),
            priority: Priority::default(),
        };
        let (attach, waited) = e.attach(&req);
        let recompute = f64::from(attach.plan.recompute_tokens) / PREFILL_TPS;
        prefill_seconds += waited.as_secs_f64() + recompute;

        let need = ((r.prompt.len() + r.output.len()).div_ceil(bt) - attach.blocks.len()) as u32;
        if e.pool.free_blocks() < need {
            e.h.refresh_reclaim_order(&mut e.pool);
        }
        let mut table = attach.blocks.to_vec();
        match e.pool.allocate(need) {
            Ok(ids) => table.extend(ids),
            Err(_) => {
                // Larger than what L0 can hold now: the request is shed.
                e.pool.release(&table);
                e.h.request_done(&mut e.pool, id, true);
                continue;
            }
        }
        e.h.after_plan(&mut e.pool);
        let mut tokens = r.prompt.clone();
        tokens.extend(&r.output);
        e.h.commit_progress(&mut e.pool, id, &table, &tokens);
        e.pool.release(&table);
        e.h.request_done(&mut e.pool, id, false);
        e.clock
            .advance(Duration::from_secs_f64(recompute) + DECODE_STEP * r.output.len() as u32);
        e.h.poll(&mut e.pool, &mut e.backend);
        if f64::from(e.pool.used_blocks()) >= RECLAIM_AT * f64::from(e.pool.total_blocks()) {
            e.h.demote_to(&mut e.pool, RECLAIM_TO, EvictReason::Pressure);
        }
        e.h.poll(&mut e.pool, &mut e.backend);
    }
    let s = e.h.stats();
    let total = s.lookups.iter().sum::<u64>().max(1) as f64;
    KvSimReport {
        workload: args.workload,
        policy: args.policy,
        requests: reqs.len() as u64,
        hit_rate_by_tier: HitRateByTier {
            l0: s.lookups[0] as f64 / total,
            l1: s.lookups[1] as f64 / total,
            l2: s.lookups[2] as f64 / total,
            miss: s.lookups[3] as f64 / total,
        },
        recompute_tokens: s.recompute_tokens,
        transfer_bytes: s.transfer_bytes,
        evictions: s.evictions,
        simulated_prefill_seconds: prefill_seconds,
    }
}
