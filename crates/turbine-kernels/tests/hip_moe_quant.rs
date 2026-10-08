//! Lab only (novanas R9700, Phase 7 S-4): the quantized MoE experts of `libturbine_hip.so` (ABI
//! v2.16, impl `turbine_hip_moe_quant`: FP8 block, FP8 tensor W8A8 with static per-linear scales,
//! MXFP4; clamp, biases, interleaved gate_up, shared expert) and the v2.16 router (top-k then
//! softmax, router bias) match the `cpu-reference` provider on the three Task 14 shapes. Run by
//! `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_moe_quant`.
//!
//! Tolerance of the experts: the 6a per-GEMM bound, an output element agrees within
//! `|want| / 256 + rms(want) / 128`. The reference models the two documented roundings the
//! kernels share with the BF16 model path (FP8-block weights as `bf16(e4m3(q) * s)`, a projection
//! bias added to the already rounded matmul output); what is left is summation order. The worst
//! ratio of every case is printed (`moe_quant: ... ratio=`).
use std::sync::Arc;

use half::bf16;
use turbine_core::types::{DType, DeviceId};
use turbine_kernels::test_support::{open_context, require_backend};
use turbine_kernels::{
    ExpertActivation, ExpertDtype, KernelProvider, MoeExpertsConfig, MoeExpertsContext,
    MoeExpertsExtra, MoeRouteConfig, MoeRouteContext, RouteMode, cpu_reference_provider,
    shim_provider,
};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

/// One test at a time per process (one device context).
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Pair {
    hip: Arc<dyn KernelProvider>,
    cpu: Arc<dyn KernelProvider>,
    hip_mem: Arc<dyn DeviceMemory>,
    cpu_mem: Arc<dyn DeviceMemory>,
}

fn setup() -> Pair {
    let ctx = open_context("hip");
    let hip_mem: Arc<dyn DeviceMemory> = ctx.clone();
    Pair {
        hip: shim_provider(ctx),
        cpu: cpu_reference_provider(),
        hip_mem,
        cpu_mem: HostMemory::new(DeviceId(u32::MAX), 24 << 30),
    }
}

/// splitmix64 plus Box-Muller (reproducible inputs).
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

    /// `n` uniform values in `[lo, hi)`.
    fn uniform(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| lo + (hi - lo) * (self.unit() as f32))
            .collect()
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn tensor(mem: &Arc<dyn DeviceMemory>, shape: &[usize], dtype: DType, raw: &[u8]) -> Tensor {
    let mut t = Tensor::empty(mem, shape, dtype).expect("alloc");
    t.storage.copy_from_host(0, raw).expect("copy in");
    t
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    let raw = t.view().slice.read_bytes().expect("read back");
    match t.dtype {
        DType::BF16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DType::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::I32 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        other => panic!("unsupported test dtype {}", other.as_str()),
    }
}

/// The expert layer shape and weight format of one case.
#[derive(Clone, Copy)]
struct Spec {
    name: &'static str,
    experts: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    dt: ExpertDtype,
}

/// The three Task 14 shapes.
const QWEN3: Spec = Spec {
    name: "qwen3_30b_a3b",
    experts: 128,
    top_k: 8,
    hidden: 2048,
    inter: 768,
    dt: ExpertDtype::Fp8Block,
};
const MIXTRAL: Spec = Spec {
    name: "mixtral_8x7b",
    experts: 8,
    top_k: 2,
    hidden: 4096,
    inter: 14336,
    dt: ExpertDtype::Fp8Tensor,
};
const GPT_OSS: Spec = Spec {
    name: "gpt_oss_20b",
    experts: 32,
    top_k: 4,
    hidden: 2880,
    inter: 2880,
    dt: ExpertDtype::Mxfp4,
};

/// What a case adds to the plain SwiGLU experts.
#[derive(Clone, Copy, Default)]
struct Variant {
    /// gpt-oss: interleaved gate_up, biases and the clamped activation.
    gptoss: bool,
    /// Shared expert of this width (with its sigmoid gate).
    shared: Option<usize>,
    /// FP8 tensor: a static scale for down's input too.
    static_down: bool,
}

/// The host bytes of one projection of every expert: `n x k`.
struct HostProj {
    data: Vec<u8>,
    scales: Vec<u8>,
    scale_shape: Vec<usize>,
}

/// Random e4m3 codes (magnitude code < 0x58: no NaN, |w| <= 15, rms ~ 4) or random E2M1 nibbles
/// (rms ~ 2.9) with scales that make a projection's output about unit-variance.
fn host_proj(rng: &mut Rng, dt: ExpertDtype, local: usize, n: usize, k: usize) -> HostProj {
    let target = 1.0 / (k as f32).sqrt();
    match dt {
        ExpertDtype::Fp8Block | ExpertDtype::Fp8Tensor => {
            let data: Vec<u8> = (0..local * n * k)
                .map(|_| {
                    let r = rng.next_u64();
                    (((r >> 8) % 0x58) as u8) | (((r & 0x80) as u8) & 0x80)
                })
                .collect();
            let (count, scale_shape) = if dt == ExpertDtype::Fp8Block {
                let (nb, kb) = (n.div_ceil(128), k.div_ceil(128));
                (local * nb * kb, vec![local, nb, kb])
            } else {
                (local, vec![local])
            };
            let scales = rng.uniform(count, 0.6 * target / 4.0, 1.4 * target / 4.0);
            HostProj {
                data,
                scales: f32_bytes(&scales),
                scale_shape,
            }
        }
        ExpertDtype::Mxfp4 => {
            let data: Vec<u8> = (0..local * n * k / 2)
                .map(|_| (rng.next_u64() >> 11) as u8)
                .collect();
            // 2^(e - 127) * 2.9 ~ target: e = 120 for k = 2880, a notch either way.
            let base = 127 + (target / 2.9).log2().round() as i32;
            let scales: Vec<u8> = (0..local * n * k / 32)
                .map(|_| (base + (rng.next_u64() % 2) as i32) as u8)
                .collect();
            HostProj {
                data,
                scales,
                scale_shape: vec![local, n, k / 32],
            }
        }
        _ => unreachable!("quantized experts only"),
    }
}

/// Every host-side array of one case, uploaded to a memory by [`Host::upload`].
struct Host {
    spec: Spec,
    var: Variant,
    gate: HostProj,
    up: HostProj,
    down: HostProj,
    act_scale: Option<Vec<f32>>,
    act_scale_down: Option<Vec<f32>>,
    b_gate: Option<Vec<f32>>,
    b_down: Option<Vec<f32>>,
    shared: Option<[Vec<f32>; 3]>,
    shared_gate_w: Option<Vec<f32>>,
}

impl Host {
    fn new(rng: &mut Rng, spec: Spec, var: Variant) -> Host {
        let (e, h, i) = (spec.experts, spec.hidden, spec.inter);
        let gate_rows = if var.gptoss { 2 * i } else { i };
        let w8a8 = spec.dt == ExpertDtype::Fp8Tensor;
        Host {
            spec,
            var,
            gate: host_proj(rng, spec.dt, e, gate_rows, h),
            up: if var.gptoss {
                HostProj {
                    data: Vec::new(),
                    scales: Vec::new(),
                    scale_shape: vec![0],
                }
            } else {
                host_proj(rng, spec.dt, e, i, h)
            },
            down: host_proj(rng, spec.dt, e, h, i),
            act_scale: w8a8.then(|| rng.uniform(e, 0.03, 0.06)),
            act_scale_down: (w8a8 && var.static_down).then(|| rng.uniform(e, 0.01, 0.02)),
            b_gate: var.gptoss.then(|| rng.normal(e * gate_rows, 3.0)),
            b_down: var.gptoss.then(|| rng.normal(e * h, 0.5)),
            shared: var.shared.map(|si| {
                [
                    rng.normal(si * h, 1.0 / (h as f32).sqrt()),
                    rng.normal(si * h, 1.0 / (h as f32).sqrt()),
                    rng.normal(h * si, 1.0 / (si as f32).sqrt()),
                ]
            }),
            shared_gate_w: var.shared.map(|_| rng.normal(h, 1.0 / (h as f32).sqrt())),
        }
    }

    fn config(&self, begin: usize, end: usize) -> MoeExpertsConfig {
        MoeExpertsConfig {
            hidden: self.spec.hidden as u32,
            inter: self.spec.inter as u32,
            num_experts: self.spec.experts as u32,
            top_k: self.spec.top_k as u32,
            expert_begin: begin as u32,
            expert_end: end as u32,
            dtype: DType::BF16,
            weight_dtype: self.spec.dt,
            activation: if self.var.gptoss {
                ExpertActivation::GptOssClamp { limit: 7.0 }
            } else {
                ExpertActivation::SwiGlu
            },
            biases: self.var.gptoss,
            gate_up_interleaved: self.var.gptoss,
            shared_inter: self.var.shared.map(|s| s as u32),
        }
    }

    fn upload(&self, mem: &Arc<dyn DeviceMemory>) -> Side {
        let (e, h, i) = (self.spec.experts, self.spec.hidden, self.spec.inter);
        let mx = self.spec.dt == ExpertDtype::Mxfp4;
        let gate_rows = if self.var.gptoss { 2 * i } else { i };
        let (wdt, pack) = if mx {
            (DType::U8, 2)
        } else {
            (DType::F8E4M3, 1)
        };
        let sdt = if mx { DType::U8 } else { DType::F32 };
        let w = |p: &HostProj, n: usize, k: usize| tensor(mem, &[e, n, k / pack], wdt, &p.data);
        let s = |p: &HostProj| tensor(mem, &p.scale_shape, sdt, &p.scales);
        let f32t = |v: &Vec<f32>, shape: &[usize]| tensor(mem, shape, DType::F32, &f32_bytes(v));
        let bft = |v: &Vec<f32>, shape: &[usize]| tensor(mem, shape, DType::BF16, &bf16_bytes(v));
        Side {
            w_gate: w(&self.gate, gate_rows, h),
            w_up: if self.var.gptoss {
                // Unused with interleaved gate_up; the shape is still validated.
                tensor(mem, &[e, i, h / pack], wdt, &vec![0u8; e * i * h / pack])
            } else {
                w(&self.up, i, h)
            },
            w_down: w(&self.down, h, i),
            s_gate: s(&self.gate),
            s_up: (!self.var.gptoss).then(|| s(&self.up)),
            s_down: s(&self.down),
            act: self.act_scale.as_ref().map(|v| f32t(v, &[e])),
            act_down: self.act_scale_down.as_ref().map(|v| f32t(v, &[e])),
            b_gate: self.b_gate.as_ref().map(|v| bft(v, &[e, gate_rows])),
            b_down: self.b_down.as_ref().map(|v| bft(v, &[e, h])),
            shared: self.shared.as_ref().map(|[g, u, d]| {
                let si = self.var.shared.expect("shared width");
                [bft(g, &[si, h]), bft(u, &[si, h]), bft(d, &[h, si])]
            }),
            shared_gate_w: self.shared_gate_w.as_ref().map(|v| bft(v, &[h])),
        }
    }
}

/// One memory's copy of a [`Host`].
struct Side {
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    s_gate: Tensor,
    s_up: Option<Tensor>,
    s_down: Tensor,
    act: Option<Tensor>,
    act_down: Option<Tensor>,
    b_gate: Option<Tensor>,
    b_down: Option<Tensor>,
    shared: Option<[Tensor; 3]>,
    shared_gate_w: Option<Tensor>,
}

impl Side {
    fn extra(&self, begin: usize, local: usize) -> MoeExpertsExtra<'_> {
        fn r(t: &Tensor, (begin, local): (usize, usize)) -> TensorView<'_> {
            t.view().rows(begin, local)
        }
        let sel = (begin, local);
        MoeExpertsExtra {
            w_gate_scale: Some(r(&self.s_gate, sel)),
            w_up_scale: self.s_up.as_ref().map(|t| r(t, sel)),
            w_down_scale: Some(r(&self.s_down, sel)),
            act_static_scale: self.act.as_ref().map(|t| r(t, sel)),
            act_static_scale_down: self.act_down.as_ref().map(|t| r(t, sel)),
            b_gate: self.b_gate.as_ref().map(|t| r(t, sel)),
            b_up: None,
            b_down: self.b_down.as_ref().map(|t| r(t, sel)),
            shared_w_gate: self.shared.as_ref().map(|s| s[0].view()),
            shared_w_up: self.shared.as_ref().map(|s| s[1].view()),
            shared_w_down: self.shared.as_ref().map(|s| s[2].view()),
            shared_gate_w: self.shared_gate_w.as_ref().map(|t| t.view()),
        }
    }
}

/// The routing of seeded logits (CPU reference, softmax then top-k, renormalized): per-token
/// weights, rows grouped by expert and the offsets.
struct Routing {
    weights: Vec<f32>,
    sorted_rows: Vec<i32>,
    offsets: Vec<i32>,
}

fn route(p: &Pair, rng: &mut Rng, spec: Spec, tokens: usize) -> Routing {
    let (e, k) = (spec.experts, spec.top_k);
    let logits = rng.normal(tokens * e, 2.0);
    let l = tensor(&p.cpu_mem, &[tokens, e], DType::F32, &f32_bytes(&logits));
    let ids = Tensor::empty(&p.cpu_mem, &[tokens, k], DType::I32).expect("alloc");
    let w = Tensor::empty(&p.cpu_mem, &[tokens, k], DType::F32).expect("alloc");
    let s = Tensor::empty(&p.cpu_mem, &[tokens * k], DType::I32).expect("alloc");
    let off = Tensor::empty(&p.cpu_mem, &[e + 1], DType::I32).expect("alloc");
    p.cpu
        .moe()
        .expect("cpu moe")
        .route(&mut MoeRouteContext {
            cfg: MoeRouteConfig {
                num_experts: e as u32,
                top_k: k as u32,
                renormalize: true,
                bf16_logits: true,
                ..Default::default()
            },
            router_logits: l.view(),
            topk_ids: ids.view(),
            topk_weights: w.view(),
            sorted_rows: s.view(),
            expert_offsets: off.view(),
            router_bias: None,
        })
        .expect("route");
    Routing {
        weights: read_f32(&w),
        sorted_rows: read_f32(&s).iter().map(|&v| v as i32).collect(),
        offsets: read_f32(&off).iter().map(|&v| v as i32).collect(),
    }
}

/// `out` (starting at `out0`) after one `moe_experts` call of `provider` over the local experts
/// `[begin, end)` of `host`'s weights on `side`, with `x` and the routing `r`.
#[allow(clippy::too_many_arguments)]
fn run(
    provider: &Arc<dyn KernelProvider>,
    mem: &Arc<dyn DeviceMemory>,
    host: &Host,
    side: &Side,
    x: &[f32],
    out0: &[f32],
    r: &Routing,
    (begin, end): (usize, usize),
) -> Vec<f32> {
    let spec = host.spec;
    let tokens = x.len() / spec.hidden;
    let cfg = host.config(begin, end);
    let moe = provider.moe().expect("moe");
    assert!(moe.supports_experts(&cfg), "must support moe_experts {cfg}");
    let rows = cfg.routed_rows(tokens);
    let needs_host = moe.needs_host_offsets(&cfg, rows);
    let host_offsets: &[i32] = if needs_host { &r.offsets } else { &[] };
    let xt = tensor(mem, &[tokens, spec.hidden], DType::BF16, &bf16_bytes(x));
    let ot = tensor(mem, &[tokens, spec.hidden], DType::BF16, &bf16_bytes(out0));
    let st = tensor(mem, &[rows], DType::I32, &i32_bytes(&r.sorted_rows));
    let off = tensor(mem, &[spec.experts + 1], DType::I32, &i32_bytes(&r.offsets));
    let tw = tensor(
        mem,
        &[tokens, spec.top_k],
        DType::F32,
        &f32_bytes(&r.weights),
    );
    let local = end - begin;
    moe.experts(&mut MoeExpertsContext {
        cfg,
        x: xt.view(),
        w_gate: side.w_gate.view().rows(begin, local),
        w_up: side.w_up.view().rows(begin, local),
        w_down: side.w_down.view().rows(begin, local),
        sorted_rows: st.view(),
        expert_offsets: off.view(),
        topk_weights: tw.view(),
        host_expert_offsets: host_offsets,
        out: ot.view(),
        workspace: None,
        extra: side.extra(begin, local),
    })
    .expect("moe_experts");
    read_f32(&ot)
}

/// Worst `|got - want| / (|want| / 256 + rms(want) / 128)`.
fn worst_ratio(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    assert!(rms > 1e-3, "degenerate reference (rms {rms})");
    let mut worst = (0f32, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ratio = (g - w).abs() / (w.abs() / 256.0 + rms / 128.0);
        if ratio.is_nan() || ratio > worst.0 {
            worst = (ratio, i);
        }
    }
    worst
}

/// HIP against the CPU reference for `tokens` tokens of `host`'s layer.
fn match_case(
    p: &Pair,
    rng: &mut Rng,
    host: &Host,
    sides: &(Side, Side),
    tokens: usize,
    tag: &str,
) {
    let spec = host.spec;
    let r = route(p, rng, spec, tokens);
    let x = rng.normal(tokens * spec.hidden, 1.0);
    let out0 = rng.normal(tokens * spec.hidden, 0.1);
    let range = (0, spec.experts);
    let got = run(&p.hip, &p.hip_mem, host, &sides.0, &x, &out0, &r, range);
    let want = run(&p.cpu, &p.cpu_mem, host, &sides.1, &x, &out0, &r, range);
    let (ratio, at) = worst_ratio(&got, &want);
    let cfg = host.config(0, spec.experts);
    let moe = p.hip.moe().expect("hip moe");
    println!(
        "moe_quant: {} {tag} tokens={tokens} impl={} ratio={ratio:.3} (element {at}: hip {} vs cpu {})",
        spec.name,
        moe.implementation_experts(&cfg),
        got[at],
        want[at]
    );
    assert_eq!(moe.implementation_experts(&cfg), "turbine_hip_moe_quant");
    if ratio > 1.0 {
        FAILURES.with(|f| {
            f.borrow_mut().push(format!(
                "{} {tag} tokens={tokens}: element {at}: hip {} vs cpu {} (ratio {ratio})",
                spec.name, got[at], want[at]
            ))
        });
    }
}

thread_local! {
    /// Cases over the bound, collected so one run reports every case's ratio.
    static FAILURES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Fails the test with every case that exceeded the bound.
fn finish() {
    let failures = FAILURES.with(|f| std::mem::take(&mut *f.borrow_mut()));
    assert!(
        failures.is_empty(),
        "cases over the 6a bound:\n{}",
        failures.join("\n")
    );
}

fn sides(p: &Pair, host: &Host) -> (Side, Side) {
    (host.upload(&p.hip_mem), host.upload(&p.cpu_mem))
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn fp8_block_experts_match_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x7e57_0001);
    let host = Host::new(&mut rng, QWEN3, Variant::default());
    let s = sides(&p, &host);
    for tokens in [1, 16, 256] {
        match_case(&p, &mut rng, &host, &s, tokens, "plain");
    }
    // A shared expert (Qwen2-MoE style: width 512, sigmoid-gated) on the same layer family.
    let shared = Host::new(
        &mut rng,
        QWEN3,
        Variant {
            shared: Some(512),
            ..Default::default()
        },
    );
    let s = sides(&p, &shared);
    match_case(&p, &mut rng, &shared, &s, 16, "shared");
    finish();
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn fp8_tensor_experts_match_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x7e57_0002);
    let host = Host::new(&mut rng, MIXTRAL, Variant::default());
    let s = sides(&p, &host);
    for tokens in [1, 16, 256] {
        match_case(&p, &mut rng, &host, &s, tokens, "w8a8 dynamic down");
    }
    drop(s);
    let st = Host::new(
        &mut rng,
        MIXTRAL,
        Variant {
            static_down: true,
            ..Default::default()
        },
    );
    let s = sides(&p, &st);
    match_case(&p, &mut rng, &st, &s, 16, "w8a8 static down");
    finish();
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn mxfp4_experts_match_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x7e57_0003);
    // The gpt-oss layer: interleaved gate_up, biases, the clamped activation.
    let host = Host::new(
        &mut rng,
        GPT_OSS,
        Variant {
            gptoss: true,
            ..Default::default()
        },
    );
    let s = sides(&p, &host);
    for tokens in [1, 16, 256] {
        match_case(&p, &mut rng, &host, &s, tokens, "gptoss clamp+bias");
    }
    drop(s);
    let plain = Host::new(&mut rng, GPT_OSS, Variant::default());
    let s = sides(&p, &plain);
    match_case(&p, &mut rng, &plain, &s, 16, "swiglu");
    finish();
}

/// A small layer of each format (8 experts, top-2, hidden 512, inter 256, no routing structure
/// beyond the seeded logits): a token's output row run alone and inside batches of 1 to 64
/// tokens is the same bits, whichever other rows share its experts.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn quant_experts_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x7e57_0004);
    let small = |dt| Spec {
        name: "small",
        experts: 8,
        top_k: 2,
        hidden: 512,
        inter: 256,
        dt,
    };
    let mut failures = Vec::new();
    for (dt, var) in [
        (ExpertDtype::Fp8Block, Variant::default()),
        (ExpertDtype::Fp8Tensor, Variant::default()),
        (
            ExpertDtype::Fp8Tensor,
            Variant {
                static_down: true,
                ..Default::default()
            },
        ),
        (
            ExpertDtype::Mxfp4,
            Variant {
                gptoss: true,
                ..Default::default()
            },
        ),
    ] {
        let spec = small(dt);
        let host = Host::new(&mut rng, spec, var);
        let side = host.upload(&p.hip_mem);
        let (h, k) = (spec.hidden, spec.top_k);
        let max_tokens = 64;
        // One seeded set of per-token logits and activations; routing is per token.
        let logits = rng.normal(max_tokens * spec.experts, 2.0);
        let x = rng.normal(max_tokens * h, 1.0);
        let routing = |tokens: usize| {
            let l = tensor(
                &p.cpu_mem,
                &[tokens, spec.experts],
                DType::F32,
                &f32_bytes(&logits[..tokens * spec.experts]),
            );
            let ids = Tensor::empty(&p.cpu_mem, &[tokens, k], DType::I32).expect("alloc");
            let w = Tensor::empty(&p.cpu_mem, &[tokens, k], DType::F32).expect("alloc");
            let s = Tensor::empty(&p.cpu_mem, &[tokens * k], DType::I32).expect("alloc");
            let off = Tensor::empty(&p.cpu_mem, &[spec.experts + 1], DType::I32).expect("alloc");
            p.cpu
                .moe()
                .expect("cpu moe")
                .route(&mut MoeRouteContext {
                    cfg: MoeRouteConfig {
                        num_experts: spec.experts as u32,
                        top_k: k as u32,
                        renormalize: true,
                        bf16_logits: true,
                        ..Default::default()
                    },
                    router_logits: l.view(),
                    topk_ids: ids.view(),
                    topk_weights: w.view(),
                    sorted_rows: s.view(),
                    expert_offsets: off.view(),
                    router_bias: None,
                })
                .expect("route");
            Routing {
                weights: read_f32(&w),
                sorted_rows: read_f32(&s).iter().map(|&v| v as i32).collect(),
                offsets: read_f32(&off).iter().map(|&v| v as i32).collect(),
            }
        };
        let zeros = vec![0f32; max_tokens * h];
        let full = run(
            &p.hip,
            &p.hip_mem,
            &host,
            &side,
            &x,
            &zeros,
            &routing(max_tokens),
            (0, spec.experts),
        );
        for tokens in [1usize, 2, 3, 7, 16, 33, 63] {
            let got = run(
                &p.hip,
                &p.hip_mem,
                &host,
                &side,
                &x[..tokens * h],
                &zeros[..tokens * h],
                &routing(tokens),
                (0, spec.experts),
            );
            let differing = got
                .iter()
                .zip(&full[..tokens * h])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "invariance: moe_quant {} static_down={} batch={tokens} vs 64 differing={differing}/{}",
                dt.as_str(),
                var.static_down,
                tokens * h
            );
            if differing != 0 {
                failures.push(format!("{} batch {tokens}: {differing}", dt.as_str()));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "rows depend on the batch: {failures:?}"
    );
}

/// gpt-oss routing: the top-k of the biased logits, then a softmax over those only; the ids and
/// grouping are exact, the weights close.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn route_topk_then_softmax_with_bias_matches_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x7e57_0005);
    let (e, k) = (32usize, 4usize);
    let bias = rng.normal(e, 2.0);
    for (tokens, bf16_logits) in [(1usize, true), (7, true), (64, false), (130, true)] {
        let logits = rng.normal(tokens * e, 2.0);
        let cfg = MoeRouteConfig {
            num_experts: e as u32,
            top_k: k as u32,
            bf16_logits,
            mode: RouteMode::TopkThenSoftmax,
            bias: true,
            ..Default::default()
        };
        let hip = p.hip.moe().expect("hip moe");
        assert!(hip.supports_route(&cfg), "must support moe_route {cfg}");
        let mut outs = Vec::new();
        for (kernel, mem) in [
            (hip, &p.hip_mem),
            (p.cpu.moe().expect("cpu moe"), &p.cpu_mem),
        ] {
            let l = tensor(mem, &[tokens, e], DType::F32, &f32_bytes(&logits));
            let b = tensor(mem, &[e], DType::F32, &f32_bytes(&bias));
            let ids = Tensor::empty(mem, &[tokens, k], DType::I32).expect("alloc");
            let w = Tensor::empty(mem, &[tokens, k], DType::F32).expect("alloc");
            let s = Tensor::empty(mem, &[tokens * k], DType::I32).expect("alloc");
            let off = Tensor::empty(mem, &[e + 1], DType::I32).expect("alloc");
            kernel
                .route(&mut MoeRouteContext {
                    cfg,
                    router_logits: l.view(),
                    topk_ids: ids.view(),
                    topk_weights: w.view(),
                    sorted_rows: s.view(),
                    expert_offsets: off.view(),
                    router_bias: Some(b.view()),
                })
                .expect("moe_route");
            outs.push((read_f32(&ids), read_f32(&w), read_f32(&s), read_f32(&off)));
        }
        let (hip_out, cpu_out) = (&outs[0], &outs[1]);
        assert_eq!(hip_out.0, cpu_out.0, "topk_ids, tokens={tokens}");
        assert_eq!(hip_out.2, cpu_out.2, "sorted_rows, tokens={tokens}");
        assert_eq!(hip_out.3, cpu_out.3, "expert_offsets, tokens={tokens}");
        for (i, (g, w)) in hip_out.1.iter().zip(&cpu_out.1).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
                "tokens={tokens} weight {i}: hip {g} vs cpu {w}"
            );
        }
        for row in hip_out.1.chunks_exact(k) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{row:?}");
        }
        println!("moe_route topk_then_softmax+bias tokens={tokens} bf16_logits={bf16_logits} ok");
    }
}
