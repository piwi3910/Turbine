//! Lab only, two GPUs (P7 Task 12, S-3): the OLMoE expert × tensor parallel drift, traced per
//! position. OLMoE-1B-7B-0125-Instruct is teacher-forced on the transformers reference tokens of
//! the golden prompts p08 and p10 (their prompt prefilled, then one decode step per reference
//! token, as the server runs a request) in four placements: one GPU, tp 2, ep 2 (attention
//! replicated) and ep 2 × tp 2, the multi-GPU ones as in-process groups over RCCL on both
//! R9700s (the served `collective_backend`). With the routing trace
//! ([`DecoderExecutor::set_routing_trace`]) and the router logits of the op trace on, the test
//! prints per placement and prompt:
//!
//! - per predicted token: the argmax, the golden metric (max |Δ logprob| of the reference's
//!   top-5 candidates, likely and tail tiers of `tolerance.json`) and, against one GPU, the
//!   layers whose top-k expert set differs, with one GPU's routing margin and the
//!   placement's there;
//! - per placement: the |Δ| of the router logits against one GPU (all layers and positions; the
//!   partial-sum rounding the router sees), every position where a top-k set differs (layer,
//!   both id sets, one GPU's margin, the |Δ router logit| of the flipped experts) and whether
//!   the ranks of the group routed identically.
//!
//! The ep 2 × tp 2 refusal (`olmoe_ep_tp_drift`) is a server startup check; the executors are
//! built here directly, so it needs no bypass. Run by `scripts/lab-test.sh novanas --gpus 2 --
//! -p turbine-model --test olmoe_ep_tp_lab -- olmoe_routing_trace` (a `TWO_GPU_TESTS` entry).
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use turbine_core::clock::SystemClock;
use turbine_distributed::collective::{self, Collective, CollectiveInit};
use turbine_kernels::test_support::{open_context_nth, require_backend, require_env_dir};
use turbine_kernels::{KernelMetrics, KernelProvider, KernelRegistry, ShimContext, shim_provider};
use turbine_model::config::ModelArchConfig;
use turbine_model::ep::{self, EpContext, ExpertPlacement};
use turbine_model::executor::{self, DecoderExecutor, ExecutorLimits, ExecutorOptions, SequenceKv};
use turbine_model::families;
use turbine_model::testing::trace::RoutingRow;
use turbine_model::tp;
use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, load_model_config};
use turbine_observability::MetricsRegistry;
use turbine_tensor::DeviceMemory;

const BLOCK_TOKENS: u32 = 128;
const PROMPTS: [&str; 2] = ["p08", "p10"];

#[derive(Deserialize)]
struct Reference {
    id: String,
    prompt_token_ids: Vec<u32>,
    tokens: Vec<u32>,
    top_logprobs: Vec<Vec<(u32, f32)>>,
}

#[derive(Deserialize)]
struct Tolerance {
    likely_logprob_floor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    Tp2,
    Ep2,
    Ep2Tp2,
}

impl Placement {
    fn name(self) -> &'static str {
        match self {
            Placement::Tp2 => "tp2",
            Placement::Ep2 => "ep2",
            Placement::Ep2Tp2 => "ep2xtp2",
        }
    }
}

/// One predicted token of a teacher-forced run.
#[derive(Clone, Debug)]
struct Predicted {
    argmax: u32,
    /// The argmax's logprob minus the runner-up's.
    top2_gap: f32,
    likely: f32,
    tail: f32,
}

/// One placement's teacher-forced run of one prompt.
#[derive(Clone, Debug)]
struct Run {
    predicted: Vec<Predicted>,
    /// (layer, position) → that row's routing.
    routing: HashMap<(u32, u32), RoutingRow>,
    /// (layer, position) → the router logits (F32, as the router GEMM wrote them).
    router: HashMap<(u32, u32), Vec<f32>>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn references() -> (Vec<Reference>, Tolerance) {
    let dir = repo_root().join("tests/golden/olmoe-1b-7b-0125-instruct");
    let text = std::fs::read_to_string(dir.join("reference.jsonl")).expect("reference.jsonl");
    let refs = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Reference>(l).expect("reference row"))
        .filter(|r| PROMPTS.contains(&r.id.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(refs.len(), PROMPTS.len(), "p08 and p10 in the reference");
    let tol = serde_json::from_str(
        &std::fs::read_to_string(dir.join("tolerance.json")).expect("tolerance.json"),
    )
    .expect("tolerance");
    (refs, tol)
}

/// FP32 log-softmax of one logits row (accumulated in f64).
fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&v| f64::from(v - max).exp()).sum();
    let lse = f64::from(max) + sum.ln();
    row.iter().map(|&v| (f64::from(v) - lse) as f32).collect()
}

fn predicted(row: &[f32], reference: &[(u32, f32)], floor: f32) -> Predicted {
    let lp = log_softmax(row);
    let mut order: Vec<usize> = (0..lp.len()).collect();
    order.select_nth_unstable_by(1, |a, b| lp[*b].total_cmp(&lp[*a]));
    let (a, b) = if lp[order[0]] >= lp[order[1]] {
        (order[0], order[1])
    } else {
        (order[1], order[0])
    };
    let (mut likely, mut tail) = (0f32, 0f32);
    for &(id, want) in reference {
        let d = (lp[id as usize] - want).abs();
        if want > floor {
            likely = likely.max(d);
        } else {
            tail = tail.max(d);
        }
    }
    Predicted {
        argmax: a as u32,
        top2_gap: lp[a] - lp[b],
        likely,
        tail,
    }
}

/// Teacher-forces `r` on `exec`: the prompt in one prefill, then each reference token but the
/// last as one decode step; every step traced.
fn teacher_forced(
    exec: &mut DecoderExecutor,
    mem: &Arc<dyn DeviceMemory>,
    r: &Reference,
    floor: f32,
) -> Run {
    let layout = *executor::ModelExecutor::kv_layout(exec);
    let max_seq = (r.prompt_token_ids.len() + r.tokens.len()) as u32;
    let mut kv = SequenceKv::new(mem, layout, max_seq).expect("kv");
    exec.set_routing_trace(true);
    exec.set_trace(true);
    let mut run = Run {
        predicted: Vec::new(),
        routing: HashMap::new(),
        router: HashMap::new(),
    };
    let plen = r.prompt_token_ids.len() as u32;
    let mut steps: Vec<(Vec<u32>, Vec<u32>)> =
        vec![(r.prompt_token_ids.clone(), (0..plen).collect())];
    for (i, &t) in r.tokens[..r.tokens.len() - 1].iter().enumerate() {
        steps.push((vec![t], vec![plen + i as u32]));
    }
    for (i, (tokens, positions)) in steps.iter().enumerate() {
        let logits = kv.forward(exec, tokens, positions).expect("forward");
        run.predicted
            .push(predicted(logits.row(0), &r.top_logprobs[i], floor));
        for row in exec.take_routing_trace() {
            run.routing.insert((row.layer, row.position), row);
        }
        for t in exec.take_trace() {
            if t.name != "router_logits" {
                continue;
            }
            let layer = t.layer.expect("router logits are per layer") as u32;
            let experts = t.shape[1];
            for (j, &p) in positions.iter().enumerate() {
                run.router
                    .insert((layer, p), t.data[j * experts..(j + 1) * experts].to_vec());
            }
        }
    }
    exec.set_routing_trace(false);
    exec.set_trace(false);
    run
}

fn registry_of(
    provider: Arc<dyn KernelProvider>,
    reqs: &[turbine_kernels::OpRequirement],
) -> Arc<KernelRegistry> {
    let order = [provider.id()];
    let card = provider.card_profile();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    Arc::new(
        KernelRegistry::build(vec![provider], &order, reqs, &metrics, card)
            .expect("every op has a provider"),
    )
}

fn limits(max_seq: u32) -> ExecutorLimits {
    ExecutorLimits {
        block_tokens: BLOCK_TOKENS,
        max_batch_tokens: max_seq,
        max_seqs: 1,
    }
}

/// One GPU's executor (device 0).
fn one_gpu(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<ShimContext>,
    max_seq: u32,
) -> DecoderExecutor {
    let opts = ExecutorOptions::default();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(dir).expect("open safetensors");
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let provider = shim_provider(ctx.clone());
    let reqs =
        executor::available_requirements(cfg, BLOCK_TOKENS, opts, std::slice::from_ref(&provider));
    DecoderExecutor::new(
        cfg,
        families::olmoe::decoder_spec(),
        weights,
        registry_of(provider, &reqs),
        mem,
        limits(max_seq),
        opts,
    )
    .expect("one-GPU executor")
}

/// Rank `rank` of a two-rank group of `placement` on device `rank`, over `collective`.
fn rank_exec(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<ShimContext>,
    placement: Placement,
    rank: u32,
    collective: Arc<dyn Collective>,
    max_seq: u32,
) -> DecoderExecutor {
    let opts = ExecutorOptions::default();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(dir).expect("open safetensors");
    let provider = shim_provider(ctx.clone());
    let providers = std::slice::from_ref(&provider);
    let moe = cfg.moe.expect("OLMoE has experts");
    let spec = cfg
        .family
        .0
        .tp_decoder_spec()
        .expect("olmoe parallel hooks");
    let tp_ctx = |world| tp::TpContext {
        rank,
        world,
        collective: Arc::clone(&collective),
        stream: mem.compute_stream(),
    };
    if placement == Placement::Tp2 {
        let s = tp::ShardSpec { rank, world: 2 };
        let slots = tp::weight_slots(cfg, s).expect("tp slots");
        let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
        let reqs = tp::available_requirements(cfg, s, BLOCK_TOKENS, opts, providers).expect("reqs");
        return DecoderExecutor::new_tp(
            cfg,
            spec,
            weights,
            registry_of(provider.clone(), &reqs),
            Arc::clone(&mem),
            limits(max_seq),
            opts,
            Some(tp_ctx(2)),
        )
        .expect("tp rank");
    }
    let attention = if placement == Placement::Ep2Tp2 {
        ep::EpAttention::TensorParallel
    } else {
        ep::EpAttention::Replicated
    };
    let s = ep::EpShard {
        rank,
        world: 2,
        attention,
    };
    let experts = Arc::new(
        ExpertPlacement::contiguous(moe.num_experts, &ep::moe_layers(cfg), 2).expect("placement"),
    );
    let slots = ep::weight_slots(cfg, s, &experts).expect("ep slots");
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let reqs =
        ep::available_requirements(cfg, s, &experts, BLOCK_TOKENS, opts, providers).expect("reqs");
    let ep_ctx = EpContext {
        rank,
        world: 2,
        placement: experts,
        collective: Arc::clone(&collective),
        stream: mem.compute_stream(),
        counts: Arc::new(ep::ExpertTokenCounts::default()),
    };
    DecoderExecutor::new_parallel(
        cfg,
        spec,
        weights,
        registry_of(provider.clone(), &reqs),
        Arc::clone(&mem),
        limits(max_seq),
        opts,
        s.tp_shard().map(|_| tp_ctx(2)),
        Some(ep_ctx),
    )
    .expect("ep rank")
}

/// Every rank's runs of `refs` in a two-rank group of `placement` over RCCL, one thread per
/// rank on its own GPU.
fn group_runs(
    cfg: &ModelArchConfig,
    dir: &Path,
    placement: Placement,
    refs: &[Reference],
    floor: f32,
    max_seq: u32,
) -> Vec<Vec<Run>> {
    let lib = collective::registry()
        .get("rccl")
        .expect("registered")
        .load(None)
        .expect("load rccl");
    let id = lib.unique_id().expect("id");
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2u32)
            .map(|rank| {
                let lib = Arc::clone(&lib);
                scope.spawn(move || {
                    let ctx = open_context_nth("hip", rank as usize);
                    let mem: Arc<dyn DeviceMemory> = ctx.clone();
                    let collective = lib
                        .open(CollectiveInit {
                            rank: rank as usize,
                            world: 2,
                            unique_id: id,
                            init_timeout: Duration::from_secs(120),
                            op_timeout: Duration::from_secs(120),
                            clock: Arc::new(SystemClock::new()),
                            metrics: None,
                            memory: Some(Arc::clone(&mem)),
                            route_max_bytes: None,
                        })
                        .expect("open rccl");
                    let mut exec = rank_exec(cfg, dir, &ctx, placement, rank, collective, max_seq);
                    refs.iter()
                        .map(|r| teacher_forced(&mut exec, &mem, r, floor))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    })
}

fn sorted(ids: &[u32]) -> Vec<u32> {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v
}

fn percentile(v: &mut [f32], q: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Prints `got`'s table against one GPU's `want` for prompt `r`.
fn report(name: &str, r: &Reference, plen: u32, want: &Run, got: &Run, ranks_agree: bool) {
    let mut keys: Vec<(u32, u32)> = want.routing.keys().copied().collect();
    keys.sort_by_key(|&(l, p)| (p, l));
    // |Δ router logit| over every layer and position.
    let mut deltas: Vec<f32> = Vec::new();
    let mut flips: Vec<String> = Vec::new();
    let mut flips_at: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut flip_margins: Vec<f32> = Vec::new();
    for &(l, p) in &keys {
        let (a, b) = (&want.router[&(l, p)], &got.router[&(l, p)]);
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        deltas.push(d);
        let (wa, ga) = (&want.routing[&(l, p)], &got.routing[&(l, p)]);
        if sorted(&wa.topk_ids) != sorted(&ga.topk_ids) {
            let (sa, sb) = (sorted(&wa.topk_ids), sorted(&ga.topk_ids));
            let only_one: Vec<u32> = sa.iter().filter(|e| !sb.contains(e)).copied().collect();
            let only_got: Vec<u32> = sb.iter().filter(|e| !sa.contains(e)).copied().collect();
            let dl: Vec<String> = only_one
                .iter()
                .chain(&only_got)
                .map(|&e| format!("{}:{:+.5}", e, b[e as usize] - a[e as usize]))
                .collect();
            flips.push(format!(
                "  flip pos {p:>3} ({}) layer {l:>2}: one-gpu {:?} margin {:.6} / {name} {:?} margin {:.6}; \
                 out {:?} in {:?}; Δlogit {} (row max |Δ| {:.5})",
                if p < plen { "prompt" } else { "decode" },
                sa,
                wa.margin,
                sb,
                ga.margin,
                only_one,
                only_got,
                dl.join(" "),
                d
            ));
            flips_at.entry(p).or_default().push(l);
            flip_margins.push(wa.margin);
        }
    }
    let n = deltas.len();
    let mut sorted_d = deltas.clone();
    println!(
        "routing-trace {name} {}: router-logit |Δ| vs one-gpu over {n} (layer, position) rows: \
         p50 {:.5} p90 {:.5} p99 {:.5} max {:.5}; top-k set flips {} ({} positions); one-gpu margin \
         at flips max {:.6}; ranks route identically: {ranks_agree}",
        r.id,
        percentile(&mut sorted_d, 0.5),
        percentile(&mut sorted_d, 0.9),
        percentile(&mut sorted_d, 0.99),
        percentile(&mut sorted_d, 1.0),
        flips.len(),
        flips_at.len(),
        flip_margins.iter().copied().fold(0f32, f32::max),
    );
    // How often a one-GPU margin is as small as the router-logit noise: the near-ties at risk.
    let mut margins: Vec<f32> = keys.iter().map(|k| want.routing[k].margin).collect();
    let noise = percentile(&mut sorted_d, 0.99);
    let at_risk = margins.iter().filter(|&&m| m <= noise).count();
    println!(
        "routing-trace {name} {}: one-gpu margins p01 {:.6} p10 {:.6} p50 {:.6}; rows with margin <= \
         p99 |Δ| ({noise:.5}): {at_risk}",
        r.id,
        percentile(&mut margins, 0.01),
        percentile(&mut margins, 0.1),
        percentile(&mut margins, 0.5),
    );
    println!(
        "routing-trace {name} {}: tok ref_tok | one-gpu argmax gap likely tail | {name} argmax gap likely tail | flipped layers",
        r.id
    );
    for (i, (w, g)) in want.predicted.iter().zip(&got.predicted).enumerate() {
        // Predicted token i comes from position plen - 1 + i; its routing at every earlier
        // position contributed through the KV.
        let pos = plen - 1 + i as u32;
        let mut layers = flips_at.get(&pos).cloned().unwrap_or_default();
        layers.sort_unstable();
        let mark = if w.argmax != g.argmax {
            " ARGMAX-DIFFERS"
        } else {
            ""
        };
        println!(
            "routing-trace {name} {} {i:>3} {:>6} | {:>6} {:>7.4} {:>7.4} {:>7.4} | {:>6} {:>7.4} {:>7.4} {:>7.4} | {layers:?}{mark}",
            r.id,
            r.tokens[i],
            w.argmax,
            w.top2_gap,
            w.likely,
            w.tail,
            g.argmax,
            g.top2_gap,
            g.likely,
            g.tail,
        );
    }
    for f in &flips {
        println!("routing-trace {name} {}{f}", r.id);
    }
}

#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so, RCCL and TURBINE_TEST_MOE_MODEL_DIR (lab-test --gpus 2)"]
fn olmoe_routing_trace_ep2_tp2() {
    if !require_backend("hip") {
        return;
    }
    let dir = require_env_dir("TURBINE_TEST_MOE_MODEL_DIR");
    let cfg = load_model_config(&dir).expect("config.json");
    let (refs, tol) = references();
    let floor = tol.likely_logprob_floor;
    let max_seq = refs
        .iter()
        .map(|r| (r.prompt_token_ids.len() + r.tokens.len()) as u32)
        .max()
        .expect("prompts");
    let one: Vec<Run> = {
        let ctx = open_context_nth("hip", 0);
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let mut exec = one_gpu(&cfg, &dir, &ctx, max_seq);
        refs.iter()
            .map(|r| teacher_forced(&mut exec, &mem, r, floor))
            .collect()
    };
    for (r, run) in refs.iter().zip(&one) {
        println!(
            "routing-trace one-gpu {}: worst likely {:.4} tail {:.4}",
            r.id,
            run.predicted.iter().map(|p| p.likely).fold(0f32, f32::max),
            run.predicted.iter().map(|p| p.tail).fold(0f32, f32::max),
        );
    }
    for placement in [Placement::Tp2, Placement::Ep2, Placement::Ep2Tp2] {
        let ranks = group_runs(&cfg, &dir, placement, &refs, floor, max_seq);
        for (i, r) in refs.iter().enumerate() {
            let agree = ranks[1][i].routing == ranks[0][i].routing;
            let got = &ranks[0][i];
            report(
                placement.name(),
                r,
                r.prompt_token_ids.len() as u32,
                &one[i],
                got,
                agree,
            );
            println!(
                "routing-trace {} {}: worst likely {:.4} tail {:.4}; argmax differs from one-gpu at {:?}",
                placement.name(),
                r.id,
                got.predicted.iter().map(|p| p.likely).fold(0f32, f32::max),
                got.predicted.iter().map(|p| p.tail).fold(0f32, f32::max),
                got.predicted
                    .iter()
                    .zip(&one[i].predicted)
                    .enumerate()
                    .filter(|(_, (g, w))| g.argmax != w.argmax)
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>(),
            );
        }
    }
}
