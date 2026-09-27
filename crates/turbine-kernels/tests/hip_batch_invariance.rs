//! Lab only (novanas R9700): batch invariance of the HIP ops the OLMoE-1B-7B executor runs. One
//! target row (or sequence) is computed alone and then inside batches of other sizes, at other
//! positions and next to other rows; its output bits must not change. A row's logits then do not
//! depend on which requests share its batch, so greedy decoding is the same at any concurrency
//! (decision 2026-09-27, "Before Phase 5": the OLMoE golden flip at concurrency 16). Llama's
//! GEMMs are row-invariant in prefill steps only (`llama_prefill_gemm_rows_are_batch_invariant`):
//! prefix reuse must reproduce the whole-prompt prefill, decode keeps its speed-tuned rows.
//!
//! Every case prints one line per batch shape (`invariance: <op> <shape> max_abs=<Δ>
//! differing=<n>/<numel> impl=<name>`) before the test asserts, so one run reports every op.
use std::sync::Arc;

use half::bf16;
use turbine_core::types::DType;
use turbine_kernels::test_support::{open_context, require_backend};
use turbine_kernels::{
    AttentionConfig, AttentionKind, GemmConfig, GemmContext, ImplChoice, KernelMetrics,
    KernelProvider, KernelRegistry, MoeExpertsConfig, MoeExpertsContext, MoeKernel, MoeRouteConfig,
    MoeRouteContext, OpConfig, OpRequirement, PagedAttentionContext, shim_provider,
};
use turbine_observability::MetricsRegistry;
use turbine_tensor::{DeviceMemory, Tensor};

// OLMoE-1B-7B shapes.
const HIDDEN: usize = 2048;
const HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const VOCAB: usize = 50304;
const INTER: usize = 1024;
const EXPERTS: usize = 64;
const TOP_K: usize = 8;
const BLOCK_TOKENS: usize = 128;

struct Hip {
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
}

fn setup() -> Hip {
    let ctx = open_context("hip");
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    Hip {
        provider: shim_provider(ctx),
        mem,
    }
}

impl Hip {
    /// The registry the executor would build for `specs` on this card: the card profile's
    /// implementation order and row tiers (e.g. `moe_experts` small-m up to 512 routed rows,
    /// grouped WMMA above).
    fn registry(&self, specs: &[OpConfig]) -> KernelRegistry {
        let reqs: Vec<OpRequirement> = specs.iter().copied().map(OpRequirement::from).collect();
        KernelRegistry::build(
            vec![Arc::clone(&self.provider)],
            &[self.provider.id()],
            &reqs,
            &KernelMetrics::register(&MetricsRegistry::new()),
            self.provider.card_profile(),
        )
        .expect("every op has an implementation")
    }
}

/// splitmix64 (reproducible inputs, no dependency).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let (u1, u2) = (self.unit(), self.unit());
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                z as f32 * scale
            })
            .collect()
    }
}

fn encode(dtype: DType, v: &[f32]) -> Vec<u8> {
    match dtype {
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect(),
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::I32 => v.iter().flat_map(|x| (*x as i32).to_le_bytes()).collect(),
        other => panic!("unsupported test dtype {}", other.as_str()),
    }
}

fn decode(dtype: DType, b: &[u8]) -> Vec<f32> {
    match dtype {
        DType::I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::BF16 => b
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DType::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported test dtype {}", other.as_str()),
    }
}

fn upload(h: &Hip, shape: &[usize], dtype: DType, data: &[f32]) -> Tensor {
    let mut t = Tensor::empty(&h.mem, shape, dtype).expect("HIP alloc");
    t.storage
        .copy_from_host(0, &encode(dtype, data))
        .expect("copy to HIP");
    t
}

fn read(t: &Tensor) -> Vec<f32> {
    decode(t.dtype, &t.view().slice.read_bytes().expect("read back"))
}

/// One comparison of a target's output in a batch against the same target alone.
struct Outcome {
    what: String,
    max_abs: f32,
    differing: usize,
    numel: usize,
}

impl Outcome {
    fn of(what: String, alone: &[f32], batched: &[f32], impl_name: &str) -> Outcome {
        assert_eq!(alone.len(), batched.len(), "{what}: length");
        let mut o = Outcome {
            what: format!("{what} impl={impl_name}"),
            max_abs: 0.0,
            differing: 0,
            numel: alone.len(),
        };
        for (a, b) in alone.iter().zip(batched) {
            if a.to_bits() != b.to_bits() {
                o.differing += 1;
                o.max_abs = o.max_abs.max((a - b).abs());
            }
        }
        println!(
            "invariance: {} max_abs={:.3e} differing={}/{}",
            o.what, o.max_abs, o.differing, o.numel
        );
        o
    }
}

fn assert_invariant(outcomes: &[Outcome]) {
    let bad: Vec<&str> = outcomes
        .iter()
        .filter(|o| o.differing > 0)
        .map(|o| o.what.as_str())
        .collect();
    assert!(
        bad.is_empty(),
        "{} batch shapes change the target's bits:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// Rows `[pos, pos + len)` of a row-major `[rows, cols]` host copy.
fn row_slice(v: &[f32], cols: usize, pos: usize, len: usize) -> &[f32] {
    &v[pos * cols..(pos + len) * cols]
}

// ------------------------------------------------------------------------------------ GEMM

/// Positions of a target row in an `m`-row batch: first, second, around the 16-row WMMA tile
/// edge, and last (deduplicated, in range).
fn positions(m: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [0, 1, 15, 16, 17, m / 2, m - 1]
        .into_iter()
        .filter(|&p| p < m)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// How [`gemm_case`] calls the GEMM and builds its target row.
#[derive(Clone, Copy)]
struct GemmCall {
    /// [`GemmContext::prefill`]: every call belongs to a prefill step.
    prefill: bool,
    /// A heavy-tailed target row (every 97th element 30× larger, as real hidden states have
    /// outlier channels, so two summation orders part far more often than on normal data).
    heavy_target: bool,
}

/// OLMoE: every step's rows are batch-invariant, decode steps included.
const ANY_STEP: GemmCall = GemmCall {
    prefill: false,
    heavy_target: false,
};

/// `c = a · wᵀ` of one target row alone (m = 1) against the same row at [`positions`] of
/// `m`-row batches.
fn gemm_case(
    h: &Hip,
    rng: &mut Rng,
    n: usize,
    k: usize,
    c_dtype: DType,
    ms: &[usize],
    call: GemmCall,
) -> Vec<Outcome> {
    let cfg = GemmConfig {
        n: n as u64,
        k: k as u64,
        trans_b: true,
        a_dtype: DType::BF16,
        b_dtype: DType::BF16,
        c_dtype,
    };
    let registry = h.registry(&[OpConfig::Gemm(cfg)]);
    let kernel = registry.gemm(&cfg);
    let impl_name = kernel.implementation(&cfg);
    let w = upload(
        h,
        &[n, k],
        DType::BF16,
        &rng.normal(n * k, 1.0 / (k as f32).sqrt()),
    );
    let mut target = rng.normal(k, 1.0);
    if call.heavy_target {
        for v in target.iter_mut().step_by(97) {
            *v *= 30.0;
        }
    }
    let run = |a: &[f32], m: usize| -> Vec<f32> {
        let a = upload(h, &[m, k], DType::BF16, a);
        let c = Tensor::empty(&h.mem, &[m, n], c_dtype).expect("c");
        kernel
            .execute(&mut GemmContext {
                a: a.view(),
                b: w.view(),
                c: c.view(),
                trans_b: true,
                alpha: 1.0,
                beta: 0.0,
                prefill: call.prefill,
            })
            .expect("gemm");
        read(&c)
    };
    let alone = run(&target, 1);
    let mut out = Vec::new();
    for &m in ms {
        for pos in positions(m) {
            let mut a = rng.normal(m * k, 1.0);
            a[pos * k..(pos + 1) * k].copy_from_slice(&target);
            let c = run(&a, m);
            out.push(Outcome::of(
                format!("gemm n={n} k={k} c={} m={m} row={pos}", c_dtype.as_str()),
                &alone,
                row_slice(&c, n, pos, 1),
                &impl_name,
            ));
        }
    }
    out
}

/// The OLMoE GEMMs (fused Q/K/V, O projection, router with F32 logits, LM head with F32
/// logits) at the row counts a serving batch reaches: decode batches of 1–64 sequences and
/// mixed or prefill batches of up to 2,048 tokens.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn gemm_rows_are_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    let h = setup();
    let mut rng = Rng(11);
    let tokens = [
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 128, 129, 256, 700, 2048,
    ];
    let seqs = [1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 33, 64];
    let mut out = Vec::new();
    out.extend(gemm_case(
        &h,
        &mut rng,
        3 * HIDDEN,
        HIDDEN,
        DType::BF16,
        &tokens,
        ANY_STEP,
    ));
    out.extend(gemm_case(
        &h,
        &mut rng,
        HIDDEN,
        HIDDEN,
        DType::BF16,
        &tokens,
        ANY_STEP,
    ));
    out.extend(gemm_case(
        &h,
        &mut rng,
        EXPERTS,
        HIDDEN,
        DType::F32,
        &tokens,
        ANY_STEP,
    ));
    out.extend(gemm_case(
        &h,
        &mut rng,
        VOCAB,
        HIDDEN,
        DType::F32,
        &seqs,
        ANY_STEP,
    ));
    let (pending, checked): (Vec<Outcome>, Vec<Outcome>) = out
        .into_iter()
        .partition(|o| PENDING_GEMM_TABLE.iter().any(|p| o.what.starts_with(p)));
    for o in pending.iter().filter(|o| o.differing > 0) {
        println!("invariance pending the per-card GEMM table: {}", o.what);
    }
    assert_invariant(&checked);
}

/// The Llama-3.2-3B GEMMs in steps that prefill prompt tokens ([`GemmContext::prefill`]): fused
/// Q/K/V, Q or O projection, K or V, fused and unfused gate/up, down (all BF16 out) at 1 to
/// 2,048 rows, and the LM head (F32 logits) at 1 to 64 sequences. Prefix reuse prefills only a
/// prompt's uncached suffix, so a row must get the bits it gets inside the whole-prompt prefill
/// at any row count (Phase 4 warm = cold; decision 2026-09-27, "Pre-Phase-5 #1 follow-up").
/// Decode steps run the speed-tuned rows, which may depend on the batch (option (c) of #1). A
/// heavy-tailed target row: hipBLASLt's summation orders often agree on normal data.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn llama_prefill_gemm_rows_are_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    const L_HIDDEN: usize = 3072;
    const L_KV: usize = 1024;
    const L_INTER: usize = 8192;
    const L_VOCAB: usize = 128_256;
    let h = setup();
    let mut rng = Rng(13);
    let tokens = [
        1, 2, 3, 7, 16, 17, 33, 64, 65, 100, 128, 129, 256, 400, 700, 1024, 2048,
    ];
    let seqs = [1, 2, 3, 16, 17, 64];
    let prefill = GemmCall {
        prefill: true,
        heavy_target: true,
    };
    let mut out = Vec::new();
    for (n, k) in [
        (L_HIDDEN + 2 * L_KV, L_HIDDEN),
        (L_HIDDEN, L_HIDDEN),
        (L_KV, L_HIDDEN),
        (2 * L_INTER, L_HIDDEN),
        (L_INTER, L_HIDDEN),
        (L_HIDDEN, L_INTER),
    ] {
        out.extend(gemm_case(&h, &mut rng, n, k, DType::BF16, &tokens, prefill));
    }
    out.extend(gemm_case(
        &h,
        &mut rng,
        L_VOCAB,
        L_HIDDEN,
        DType::F32,
        &seqs,
        prefill,
    ));
    assert_invariant(&out);
}

/// GEMM shapes whose rows hipBLASLt does not compute batch-invariantly today, reported but not
/// asserted: the first heuristic answer per (m, n, k) changes with m, and for these shapes the
/// rows past the first 16 of a batch of more than 16 rows (router: m ≥ 17) or of large batches
/// (O projection: m ≥ 129) get other F32 sums than the same row alone (measured on the R9700,
/// 2026-09-27: router |Δ| ≤ 3.3e-6 on ~59 of 64 logits, O projection one BF16 ulp on 1–2 of 2,048
/// outputs). The fix is the per-card GEMM table (one algorithm per shape for every m, a
/// data-parallel one without split-k); when it lands, drop the entries it covers so the test
/// asserts them.
const PENDING_GEMM_TABLE: &[&str] = &[];

// ------------------------------------------------------------------------------------- MoE

/// Heavy-tailed activations: every 97th channel 30× larger, as real hidden states have outlier
/// channels. The partial sums then cancel, so two summation orders part far more often than on
/// plain normal data (where small-m and grouped WMMA agreed on every output of a row).
fn heavy(mut x: Vec<f32>) -> Vec<f32> {
    for (i, v) in x.iter_mut().enumerate() {
        if i % HIDDEN % 97 == 3 {
            *v *= 30.0;
        }
    }
    x
}

/// `moe_route` then `moe_experts` (through the registry the card profile builds, whose routed-row
/// tiers switch from small-m to prefill WMMA above 512 rows, as the executor calls them) for one
/// target token alone and at the first and last position of `tokens`-token batches; the target's
/// routing and its output row must not change. Breaks if the tiers (or the prefill kernels' two
/// down-projection tiles) stop computing a row with the same WMMA chain.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn moe_rows_are_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    let h = setup();
    let mut rng = Rng(12);
    let route_cfg = MoeRouteConfig {
        num_experts: EXPERTS as u32,
        top_k: TOP_K as u32,
        renormalize: false,
        bf16_logits: true,
    };
    let cfg = MoeExpertsConfig {
        hidden: HIDDEN as u32,
        inter: INTER as u32,
        num_experts: EXPERTS as u32,
        top_k: TOP_K as u32,
        expert_begin: 0,
        expert_end: EXPERTS as u32,
        dtype: DType::BF16,
    };
    let registry = h.registry(&[OpConfig::MoeRoute(route_cfg), OpConfig::MoeExperts(cfg)]);
    let (router, moe) = (registry.moe_route(&route_cfg), registry.moe_experts(&cfg));
    let small_max = h
        .provider
        .card_profile()
        .map_or(0, |c| c.thresholds.moe_small_max_rows as usize);
    let n = EXPERTS * INTER * HIDDEN;
    let up = 1.0 / (HIDDEN as f32).sqrt();
    let w_gate = upload(
        &h,
        &[EXPERTS, INTER, HIDDEN],
        DType::BF16,
        &rng.normal(n, up),
    );
    let w_up = upload(
        &h,
        &[EXPERTS, INTER, HIDDEN],
        DType::BF16,
        &rng.normal(n, up),
    );
    let w_down = upload(
        &h,
        &[EXPERTS, HIDDEN, INTER],
        DType::BF16,
        &rng.normal(n, 1.0 / (INTER as f32).sqrt()),
    );
    let target_x = heavy(rng.normal(HIDDEN, 1.0));
    let target_logits = rng.normal(EXPERTS, 2.0);

    // (topk ids, topk weights, output row) of the target at `pos` of a batch.
    let run_with = |moe: &dyn MoeKernel, x: &[f32], logits: &[f32], t: usize, pos: usize| {
        let rows = t * TOP_K;
        let l = upload(&h, &[t, EXPERTS], DType::F32, logits);
        let ids = Tensor::empty(&h.mem, &[t, TOP_K], DType::I32).expect("ids");
        let tw = Tensor::empty(&h.mem, &[t, TOP_K], DType::F32).expect("weights");
        let sorted = Tensor::empty(&h.mem, &[rows], DType::I32).expect("sorted");
        let offsets = Tensor::empty(&h.mem, &[EXPERTS + 1], DType::I32).expect("offsets");
        router
            .route(&mut MoeRouteContext {
                cfg: route_cfg,
                router_logits: l.view(),
                topk_ids: ids.view(),
                topk_weights: tw.view(),
                sorted_rows: sorted.view(),
                expert_offsets: offsets.view(),
            })
            .expect("moe_route");
        let host: Vec<i32> = if moe.needs_host_offsets(&cfg, rows) {
            read(&offsets).iter().map(|&o| o as i32).collect()
        } else {
            Vec::new()
        };
        let xs = upload(&h, &[t, HIDDEN], DType::BF16, x);
        let o = upload(&h, &[t, HIDDEN], DType::BF16, &vec![0.0; t * HIDDEN]);
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: xs.view(),
            w_gate: w_gate.view(),
            w_up: w_up.view(),
            w_down: w_down.view(),
            sorted_rows: sorted.view(),
            expert_offsets: offsets.view(),
            topk_weights: tw.view(),
            host_expert_offsets: &host,
            out: o.view(),
            workspace: None,
        })
        .expect("moe_experts");
        (
            row_slice(&read(&ids), TOP_K, pos, 1).to_vec(),
            row_slice(&read(&tw), TOP_K, pos, 1).to_vec(),
            read(&o),
        )
    };
    let run = |x: &[f32], logits: &[f32], t: usize, pos: usize| {
        let (ids, w, o) = run_with(moe, x, logits, t, pos);
        (ids, w, row_slice(&o, HIDDEN, pos, 1).to_vec())
    };
    // The device-offset moe_experts implementations (small-m, prefill WMMA and grouped WMMA;
    // the card profile picks between the first two by routed rows, and the prefill kernels
    // change their down-projection tile at 12,288 rows) on the same batch with heavy-tailed
    // activations: bitwise equal on every row, at 65 tokens and at 1,536 (12,288 routed rows).
    // Before the small-m tier ran the WMMA chain, 426 of 133,120 outputs differed at 65 tokens
    // (up to 0.125).
    let mut ab = Vec::new();
    for (t, pos) in [(65, 17), (1536, 700)] {
        let mut x = heavy(rng.normal(t * HIDDEN, 1.0));
        let mut logits = rng.normal(t * EXPERTS, 2.0);
        x[pos * HIDDEN..(pos + 1) * HIDDEN].copy_from_slice(&target_x);
        logits[pos * EXPERTS..(pos + 1) * EXPERTS].copy_from_slice(&target_logits);
        let spec = OpConfig::MoeExperts(cfg);
        let impls = h.provider.implementations(spec.op());
        let mut first: Option<(String, Vec<f32>)> = None;
        for info in impls
            .iter()
            .filter(|i| i.name.starts_with("turbine_hip_moe_"))
        {
            let Some(bound) = h.provider.bind(&spec, &ImplChoice::Single(info.index)) else {
                continue;
            };
            let kernel = bound.moe().expect("moe");
            if !kernel.supports_experts(&cfg) {
                continue;
            }
            let got = run_with(kernel, &x, &logits, t, pos).2;
            match &first {
                None => first = Some((info.name.clone(), got)),
                Some((name, want)) => ab.push(Outcome::of(
                    format!("moe_experts {} vs {name} tokens={t} all rows", info.name),
                    want,
                    &got,
                    &info.name,
                )),
            }
        }
    }
    assert_eq!(
        ab.len(),
        4,
        "small-m, prefill WMMA and grouped WMMA all ran at both sizes"
    );
    let alone = run(&target_x, &target_logits, 1, 0);
    let mut out = ab;
    for t in [1usize, 2, 4, 8, 16, 32, 63, 64, 65, 66, 128, 256, 700, 2048] {
        for pos in [0, t - 1] {
            let mut x = heavy(rng.normal(t * HIDDEN, 1.0));
            let mut logits = rng.normal(t * EXPERTS, 2.0);
            x[pos * HIDDEN..(pos + 1) * HIDDEN].copy_from_slice(&target_x);
            logits[pos * EXPERTS..(pos + 1) * EXPERTS].copy_from_slice(&target_logits);
            let got = run(&x, &logits, t, pos);
            let rows = t * TOP_K;
            let impl_name = if rows <= small_max {
                "small-m tier"
            } else {
                "above the small-m tier"
            };
            out.push(Outcome::of(
                format!("moe_route ids tokens={t} row={pos}"),
                &alone.0,
                &got.0,
                "route",
            ));
            out.push(Outcome::of(
                format!("moe_route weights tokens={t} row={pos}"),
                &alone.1,
                &got.1,
                "route",
            ));
            out.push(Outcome::of(
                format!("moe_experts tokens={t} routed_rows={rows} row={pos}"),
                &alone.2,
                &got.2,
                impl_name,
            ));
            if t == 1 {
                break;
            }
        }
    }
    assert_invariant(&out);
}

// ------------------------------------------------------------------------------- attention

/// One sequence of a paged attention batch: its new query rows and its K/V length.
#[derive(Clone, Copy)]
struct Seq {
    q_len: usize,
    kv_len: usize,
}

/// Paged attention (append + attend) over `seqs`, sequence `target` holding the target's data:
/// its query/new K/V rows, its blocks (always blocks `0..`) and the pool history in them are the
/// same in every batch. Returns the target's output rows.
fn paged_run(
    h: &Hip,
    kind: AttentionKind,
    seqs: &[Seq],
    target: usize,
    target_rows: &[Vec<f32>; 3],
    pool_seed: u64,
) -> (Vec<f32>, String) {
    let cfg = AttentionConfig {
        kind,
        num_q_heads: HEADS as u32,
        num_kv_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        dtype: DType::BF16,
        block_tokens: Some(BLOCK_TOKENS as u32),
        causal: true,
    };
    let kernel = h.provider.attention().expect("hip attention");
    assert!(kernel.supports(&cfg), "{cfg}");
    let impl_name = kernel.implementation(&cfg);
    let row = HEADS * HEAD_DIM;
    let blocks_of: Vec<usize> = seqs
        .iter()
        .map(|s| s.kv_len.div_ceil(BLOCK_TOKENS))
        .collect();
    let max_blocks = blocks_of.iter().copied().max().expect("a sequence");
    // The target owns blocks 0..; the others follow in batch order.
    let mut next = blocks_of[target];
    let mut table = vec![0f32; seqs.len() * max_blocks];
    for (s, &n) in blocks_of.iter().enumerate() {
        for b in 0..n {
            table[s * max_blocks + b] = if s == target {
                b as f32
            } else {
                let id = next;
                next += 1;
                id as f32
            };
        }
    }
    let num_blocks = next;
    let pool_len = num_blocks * 2 * BLOCK_TOKENS * row;
    // The pool history: the same seeded values in every run, so the target's blocks match.
    let pool = upload(
        h,
        &[num_blocks, 2, BLOCK_TOKENS, HEADS, HEAD_DIM],
        DType::BF16,
        &Rng(pool_seed).normal(pool_len, 1.0),
    );
    let total_q: usize = seqs.iter().map(|s| s.q_len).sum();
    let mut rng = Rng(pool_seed ^ 0x5eed);
    let mut data: [Vec<f32>; 3] = [
        rng.normal(total_q * row, 1.0),
        rng.normal(total_q * row, 1.0),
        rng.normal(total_q * row, 1.0),
    ];
    let start: usize = seqs[..target].iter().map(|s| s.q_len).sum();
    for (d, t) in data.iter_mut().zip(target_rows) {
        d[start * row..(start + seqs[target].q_len) * row].copy_from_slice(t);
    }
    let shape = [total_q, HEADS, HEAD_DIM];
    let q = upload(h, &shape, DType::BF16, &data[0]);
    let k = upload(h, &shape, DType::BF16, &data[1]);
    let v = upload(h, &shape, DType::BF16, &data[2]);
    let o = upload(h, &shape, DType::BF16, &vec![0.0; total_q * row]);
    let mut indptr = vec![0f32];
    for s in seqs {
        indptr.push(indptr.last().copied().unwrap_or(0.0) + s.q_len as f32);
    }
    let kv: Vec<f32> = seqs.iter().map(|s| s.kv_len as f32).collect();
    let bt = upload(h, &[seqs.len(), max_blocks], DType::I32, &table);
    let ip = upload(h, &[seqs.len() + 1], DType::I32, &indptr);
    let kl = upload(h, &[seqs.len()], DType::I32, &kv);
    kernel
        .execute_paged(&mut PagedAttentionContext {
            cfg,
            q: q.view(),
            k_new: k.view(),
            v_new: v.view(),
            out: o.view(),
            kv_layer: pool.view(),
            block_table: bt.view(),
            q_indptr: ip.view(),
            kv_lens: kl.view(),
            max_q_len: seqs.iter().map(|s| s.q_len).max().unwrap_or(0) as u32,
            max_kv_len: seqs.iter().map(|s| s.kv_len).max().unwrap_or(0) as u32,
            max_blocks_per_seq: max_blocks as u32,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        })
        .expect("paged attention");
    let out = read(&o);
    (
        row_slice(&out, row, start, seqs[target].q_len).to_vec(),
        impl_name,
    )
}

/// A decode row (45 cached tokens) and a 44-token prefill, alone and next to other decodes,
/// prefills and a 2,004-token prefill, at the first, a middle and the last position, under both
/// attention kinds the executor picks (decode-only batches run `DecodePaged`, any batch with a
/// prefill `PrefillPaged`).
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn paged_attention_rows_are_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    let h = setup();
    let mut rng = Rng(13);
    let row = HEADS * HEAD_DIM;
    let decode = Seq {
        q_len: 1,
        kv_len: 45,
    };
    let prefill = Seq {
        q_len: 44,
        kv_len: 44,
    };
    let chunk = Seq {
        q_len: 20,
        kv_len: 300,
    };
    let d = |kv_len| Seq { q_len: 1, kv_len };
    let p = |q_len| Seq {
        q_len,
        kv_len: q_len,
    };
    let mut out = Vec::new();
    for (name, target) in [("decode", decode), ("prefill", prefill), ("chunk", chunk)] {
        let rows: [Vec<f32>; 3] = [
            rng.normal(target.q_len * row, 1.0),
            rng.normal(target.q_len * row, 1.0),
            rng.normal(target.q_len * row, 1.0),
        ];
        let alone_kind = if target.q_len == 1 {
            AttentionKind::DecodePaged
        } else {
            AttentionKind::PrefillPaged
        };
        let (alone, impl_name) = paged_run(&h, alone_kind, &[target], 0, &rows, 99);
        let decodes: Vec<Seq> = (0..15).map(|i| d(20 + 37 * i)).collect();
        let prefills = vec![p(25), p(56), p(70)];
        let mut cases: Vec<(String, AttentionKind, Vec<Seq>, usize)> = Vec::new();
        let with = |others: &[Seq], at: usize| {
            let mut v = others.to_vec();
            v.insert(at, target);
            v
        };
        if target.q_len == 1 {
            cases.push((
                "alone as prefill kind".into(),
                AttentionKind::PrefillPaged,
                vec![target],
                0,
            ));
            for at in [0, 7, 15] {
                cases.push((
                    format!("16 decodes at {at}"),
                    AttentionKind::DecodePaged,
                    with(&decodes, at),
                    at,
                ));
            }
            cases.push((
                "2 decodes at 1".into(),
                AttentionKind::DecodePaged,
                with(&decodes[..1], 1),
                1,
            ));
        }
        for at in [0, 3] {
            cases.push((
                format!("with prefills at {at}"),
                AttentionKind::PrefillPaged,
                with(&prefills, at),
                at,
            ));
        }
        cases.push((
            "with decodes and prefills at 15".into(),
            AttentionKind::PrefillPaged,
            with(&[decodes.clone(), prefills.clone()].concat(), 15),
            15,
        ));
        cases.push((
            "with a 2004-token prefill at 1".into(),
            AttentionKind::PrefillPaged,
            with(&[p(2004)], 1),
            1,
        ));
        for (case, kind, seqs, at) in cases {
            let (got, _) = paged_run(&h, kind, &seqs, at, &rows, 99);
            out.push(Outcome::of(
                format!(
                    "paged_attention {name} (q_len {} kv_len {}) {case}",
                    target.q_len, target.kv_len
                ),
                &alone,
                &got,
                &impl_name,
            ));
        }
    }
    assert_invariant(&out);
}

/// Lab timing (perf tier): the device-offset `moe_experts` implementations (small-m, prefill WMMA
/// and grouped WMMA; the card profile's row tiers pick the first two) bound one at a time, at
/// 1–128 tokens with
/// uniform routing over 64 experts whose 805 MB of weights stream from memory: one line per
/// implementation and size with the mean time per call and the weight bandwidth. No bound; the
/// numbers compare kernel changes and place the tier boundary (`moe_small_max_rows`) on GPU 0.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas --tier perf)"]
fn moe_decode_tier_timings() {
    if !require_backend("hip") {
        return;
    }
    let h = setup();
    let mut rng = Rng(21);
    let route_cfg = MoeRouteConfig {
        num_experts: EXPERTS as u32,
        top_k: TOP_K as u32,
        renormalize: false,
        bf16_logits: true,
    };
    let cfg = MoeExpertsConfig {
        hidden: HIDDEN as u32,
        inter: INTER as u32,
        num_experts: EXPERTS as u32,
        top_k: TOP_K as u32,
        expert_begin: 0,
        expert_end: EXPERTS as u32,
        dtype: DType::BF16,
    };
    let registry = h.registry(&[OpConfig::MoeRoute(route_cfg)]);
    let router = registry.moe_route(&route_cfg);
    let spec = OpConfig::MoeExperts(cfg);
    let bound: Vec<(String, Arc<dyn KernelProvider>)> = h
        .provider
        .implementations(spec.op())
        .into_iter()
        .filter(|i| i.name.starts_with("turbine_hip_moe_"))
        .filter_map(|i| {
            let p = h.provider.bind(&spec, &ImplChoice::Single(i.index))?;
            Some((i.name, p))
        })
        .collect();
    assert_eq!(bound.len(), 3, "small-m, prefill WMMA and grouped WMMA");
    let n = EXPERTS * INTER * HIDDEN;
    let up = 1.0 / (HIDDEN as f32).sqrt();
    let w_gate = upload(
        &h,
        &[EXPERTS, INTER, HIDDEN],
        DType::BF16,
        &rng.normal(n, up),
    );
    let w_up = upload(
        &h,
        &[EXPERTS, INTER, HIDDEN],
        DType::BF16,
        &rng.normal(n, up),
    );
    let w_down = upload(
        &h,
        &[EXPERTS, HIDDEN, INTER],
        DType::BF16,
        &rng.normal(n, 0.03),
    );
    for t in [1usize, 2, 4, 8, 16, 32, 48, 64, 96, 128] {
        let rows = t * TOP_K;
        let l = upload(&h, &[t, EXPERTS], DType::F32, &rng.normal(t * EXPERTS, 2.0));
        let ids = Tensor::empty(&h.mem, &[t, TOP_K], DType::I32).expect("ids");
        let tw = Tensor::empty(&h.mem, &[t, TOP_K], DType::F32).expect("weights");
        let sorted = Tensor::empty(&h.mem, &[rows], DType::I32).expect("sorted");
        let offsets = Tensor::empty(&h.mem, &[EXPERTS + 1], DType::I32).expect("offsets");
        router
            .route(&mut MoeRouteContext {
                cfg: route_cfg,
                router_logits: l.view(),
                topk_ids: ids.view(),
                topk_weights: tw.view(),
                sorted_rows: sorted.view(),
                expert_offsets: offsets.view(),
            })
            .expect("moe_route");
        let off = read(&offsets);
        let active = (0..EXPERTS).filter(|&e| off[e + 1] > off[e]).count();
        let xs = upload(&h, &[t, HIDDEN], DType::BF16, &rng.normal(t * HIDDEN, 1.0));
        let o = upload(&h, &[t, HIDDEN], DType::BF16, &vec![0.0; t * HIDDEN]);
        for (name, provider) in &bound {
            let moe = provider.moe().expect("moe");
            let run = || {
                moe.experts(&mut MoeExpertsContext {
                    cfg,
                    x: xs.view(),
                    w_gate: w_gate.view(),
                    w_up: w_up.view(),
                    w_down: w_down.view(),
                    sorted_rows: sorted.view(),
                    expert_offsets: offsets.view(),
                    topk_weights: tw.view(),
                    host_expert_offsets: &[],
                    out: o.view(),
                    workspace: None,
                })
                .expect("moe_experts");
            };
            run();
            run();
            h.mem.synchronize().expect("synchronize");
            let iters = 64;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run();
            }
            h.mem.synchronize().expect("synchronize");
            let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
            let gb_s = (active * 3 * INTER * HIDDEN * 2) as f64 / us / 1e3;
            println!(
                "moe_decode_timing impl={name} tokens={t} rows={rows} active_experts={active} \
                 us={us:.1} weight_gb_s={gb_s:.0}"
            );
        }
    }
}
