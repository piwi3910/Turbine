//! Lab only (novanas R9700, Phase 7 S-4): the quantized MoE experts of `libturbine_hip.so` (ABI
//! v2.16, impl `turbine_hip_moe_quant`: FP8 block, FP8 tensor W8A8 with static per-linear scales,
//! MXFP4; clamp, biases, interleaved gate_up, shared expert) and the v2.16 router (top-k then
//! softmax, router bias) match the `cpu-reference` provider on the three Task 14 shapes. Run by
//! `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_moe_quant`.
//!
//! Tolerances (user decision 2026-10-08 C, "P7 quantized expert test bound"):
//!
//! - Every stage alone, on the same BF16 input, under the 6a per-GEMM bound: an element agrees
//!   within `|want| / 256 + rms(want) / 128`. `qgemm_stages_match_reference` (gate_up and down
//!   `qgemm`, through the public `qgemm` entry the loop calls with `prefill = 1`, on one expert's
//!   slice of the case's weights), `activation_stage_matches_reference` (biases and the gated
//!   activation) and `scatter_stage_matches_reference`. The kernel has no per-stage entry and no
//!   test-only hook was added: the activation and scatter stages run through `moe_experts` on
//!   layers whose projections are exact identities (one 1.0 per row, so both GEMMs are exact
//!   whatever the summation order), and the scatter is checked against its specification
//!   applied to the activation rows HIP itself produced. FP8-block stages are referenced by the
//!   same BF16 GEMM over `bf16(e4m3(q) * s)` weights (the documented staging, as the
//!   `moe_experts` reference does); the CPU `qgemm` keeps the products unrounded and reads
//!   1.1 to 1.3 of the bound on random scales for that reason alone.
//!   Measured worst ratios: qgemm FP8 block 0.95 (m = 64), FP8 tensor W8A8 0.86, MXFP4 0.59,
//!   activation 0.000 (bit-exact, every format, clamp and biases included), scatter 0.000.
//! - The chained end-to-end expert output (`*_experts_match_reference`) under 4x that bound,
//!   `|want| / 64 + rms(want) / 32`. Measured worst ratios: Qwen3 FP8 block 0.58 (256 tokens),
//!   Mixtral FP8 tensor W8A8 0.43, gpt-oss MXFP4 0.39; every 1-token case is below 0.2. Against
//!   the 6a bound the same runs read up to 2.33 (run 1008112147-0f4907dc). Each stage is
//!   within the 6a bound, so the excess is not a defect of one kernel: it is the FP32
//!   accumulation order of the WMMA tiles against the CPU's sequential GEMM, which flips a BF16
//!   rounding in 1 to 2 ulps of some elements, compounding through three chained GEMMs (with
//!   the BF16 rounding of the gate/up values and the activation between them). The model-level
//!   golden and GSM8K gates stay the bar for the chain.
//!
//! The worst ratio of every case is printed (`moe_quant: ... ratio=`).
use std::sync::Arc;

use half::bf16;
use turbine_core::types::{DType, DeviceId};
use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};
use turbine_kernels::test_support::{open_context, require_backend};
use turbine_kernels::{
    ExpertActivation, ExpertDtype, KernelProvider, MoeExpertsConfig, MoeExpertsContext,
    MoeExpertsExtra, MoeRouteConfig, MoeRouteContext, QGemmConfig, QGemmContext, QuantizeActConfig,
    QuantizeActContext, RouteMode, cpu_reference_provider, shim_provider,
};
use turbine_kernels::{ImplChoice, OpConfig, OpKind};
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

/// A tolerance: an element agrees within `|want| * rel + rms(want) * rms`.
#[derive(Clone, Copy)]
struct Bound {
    rel: f32,
    rms: f32,
    name: &'static str,
}

/// The 6a per-GEMM bound, `|want| / 256 + rms(want) / 128`: every stage alone.
const BOUND_6A: Bound = Bound {
    rel: 1.0 / 256.0,
    rms: 1.0 / 128.0,
    name: "6a",
};

/// Four times the 6a bound, `|want| / 64 + rms(want) / 32`: the chained end-to-end output.
const BOUND_CHAINED: Bound = Bound {
    rel: 1.0 / 64.0,
    rms: 1.0 / 32.0,
    name: "chained 4x",
};

/// Worst `|got - want| / (|want| * rel + rms(want) * rms)` and where.
fn worst_ratio(got: &[f32], want: &[f32], bound: Bound) -> (f32, usize) {
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    assert!(rms > 1e-3, "degenerate reference (rms {rms})");
    let mut worst = (0f32, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ratio = (g - w).abs() / (w.abs() * bound.rel + rms * bound.rms);
        if ratio.is_nan() || ratio > worst.0 {
            worst = (ratio, i);
        }
    }
    worst
}

/// Prints the worst ratio of `got` against `want` and records the case when it is over `bound`.
fn judge(what: &str, got: &[f32], want: &[f32], bound: Bound) {
    let (ratio, at) = worst_ratio(got, want, bound);
    println!(
        "moe_quant: {what} bound={} ratio={ratio:.3} (element {at}: hip {} vs ref {})",
        bound.name, got[at], want[at]
    );
    if ratio.is_nan() || ratio > 1.0 {
        FAILURES.with(|f| {
            f.borrow_mut().push(format!(
                "{what}: element {at}: hip {} vs ref {} (ratio {ratio} over the {} bound)",
                got[at], want[at], bound.name
            ))
        });
    }
}

/// The two paths of `turbine_hip_moe_quant`, each forced through its own implementation (P7
/// Task 16 step 2): the per-expert qgemm loop and the grouped prefill kernel. The default
/// implementation runs the grouped kernel from 64 routed rows for the formats it wins on.
const PATHS: [&str; 2] = [
    "turbine_hip_moe_quant_loop",
    "turbine_hip_moe_quant_grouped",
];

/// The HIP provider bound to the implementation `name` for `host`'s layer.
fn forced(p: &Pair, host: &Host, name: &str) -> Arc<dyn KernelProvider> {
    let cfg = host.config(0, host.spec.experts);
    let index = p
        .hip
        .implementations(OpKind::MoeExperts)
        .into_iter()
        .find(|i| i.name == name)
        .unwrap_or_else(|| panic!("the library has no {name}"))
        .index;
    p.hip
        .bind(&OpConfig::MoeExperts(cfg), &ImplChoice::Single(index))
        .unwrap_or_else(|| panic!("cannot bind {name}"))
}

/// HIP (each path) against the CPU reference for `tokens` tokens of `host`'s layer, end to end.
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
    let want = run(&p.cpu, &p.cpu_mem, host, &sides.1, &x, &out0, &r, range);
    let moe = p.hip.moe().expect("hip moe");
    assert_eq!(
        moe.implementation_experts(&host.config(0, spec.experts)),
        "turbine_hip_moe_quant"
    );
    for path in PATHS {
        let prov = forced(p, host, path);
        let got = run(&prov, &p.hip_mem, host, &sides.0, &x, &out0, &r, range);
        judge(
            &format!("chained {} {tag} tokens={tokens} {path}", spec.name),
            &got,
            &want,
            BOUND_CHAINED,
        );
    }
    // The default implementation: the same bits as the path it dispatches to.
    let got = run(&p.hip, &p.hip_mem, host, &sides.0, &x, &out0, &r, range);
    judge(
        &format!("chained {} {tag} tokens={tokens} default", spec.name),
        &got,
        &want,
        BOUND_CHAINED,
    );
}

thread_local! {
    /// Cases over their bound, collected so one run reports every case's ratio.
    static FAILURES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Fails the test with every case that exceeded its bound.
fn finish() {
    let failures = FAILURES.with(|f| std::mem::take(&mut *f.borrow_mut()));
    assert!(
        failures.is_empty(),
        "cases over their bound:\n{}",
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
        for path in PATHS {
            let prov = forced(&p, &host, path);
            let full = run(
                &prov,
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
                    &prov,
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
                    "invariance: moe_quant {path} {} static_down={} batch={tokens} vs 64 differing={differing}/{}",
                    dt.as_str(),
                    var.static_down,
                    tokens * h
                );
                if differing != 0 {
                    failures.push(format!(
                        "{path} {} batch {tokens}: {differing}",
                        dt.as_str()
                    ));
                }
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

// ---------------------------------------------------------------------------------------------
// Stage tests (user decision 2026-10-08 C): every stage of the per-expert loop alone against the
// CPU reference on the same BF16 input, under the 6a bound. The kernel has no per-stage entry
// and gets no test-only hook: the two qgemm stages run through the public `qgemm` entry the
// loop calls (`turbine_qgemm`, `prefill = 1`, the same config the loop builds) on one expert's
// slice of the case's random weights; the activation stage (biases and the gated activation)
// and the scatter stage run through `moe_experts` on layers whose projections are exact
// identities, so the two GEMMs add no rounding of their own.
// ---------------------------------------------------------------------------------------------

fn bytes(t: &Tensor) -> Vec<u8> {
    t.view().slice.read_bytes().expect("read back")
}

fn rb(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

fn scheme_of(dt: ExpertDtype) -> QuantSchemeDesc {
    match dt {
        ExpertDtype::Fp8Block => QuantSchemeDesc::Fp8Block {
            block_n: 128,
            block_k: 128,
        },
        ExpertDtype::Fp8Tensor => QuantSchemeDesc::Fp8Tensor,
        ExpertDtype::Mxfp4 => QuantSchemeDesc::Mxfp4,
        _ => unreachable!("quantized experts only"),
    }
}

/// Expert `e`'s code bytes and scale bytes of a projection stacked over `local` experts.
fn expert_slice(proj: &HostProj, local: usize, e: usize) -> (&[u8], &[u8]) {
    let (d, s) = (proj.data.len() / local, proj.scales.len() / local);
    (
        &proj.data[e * d..(e + 1) * d],
        &proj.scales[e * s..(e + 1) * s],
    )
}

/// One qgemm of the loop (`n x k` weight of expert 1, `m` rows of N(0, 1) BF16 input, the FP8
/// activation `mode` of the format) on HIP and on the CPU reference, both fed the same input
/// (the HIP's own FP8 codes and scales where the activations are quantized).
#[allow(clippy::too_many_arguments)]
fn gemm_stage(
    p: &Pair,
    rng: &mut Rng,
    host: &Host,
    proj: &HostProj,
    (n, k): (usize, usize),
    m: usize,
    mode: ActQuantDesc,
    tag: &str,
) {
    let dt = host.spec.dt;
    let scheme = scheme_of(dt);
    let (wdt, pack, sdt) = if dt == ExpertDtype::Mxfp4 {
        (DType::U8, 2, DType::U8)
    } else {
        (DType::F8E4M3, 1, DType::F32)
    };
    let a_dtype = if mode == ActQuantDesc::None {
        DType::BF16
    } else {
        DType::F8E4M3
    };
    let cfg = QGemmConfig {
        n: n as u32,
        k: k as u32,
        scheme,
        act_quant: mode,
        a_dtype,
        c_dtype: DType::BF16,
    };
    let (wb, sb) = expert_slice(proj, host.spec.experts, 1);
    let twin = |shape: &[usize], dtype: DType, raw: &[u8]| {
        (
            tensor(&p.hip_mem, shape, dtype, raw),
            tensor(&p.cpu_mem, shape, dtype, raw),
        )
    };
    let (b_hip, b_cpu) = twin(&[n, k / pack], wdt, wb);
    let (bs_hip, bs_cpu) = twin(&[scheme.scale_count(n, k)], sdt, sb);

    let x = rng.normal(m * k, 1.0);
    let x_hip = tensor(&p.hip_mem, &[m, k], DType::BF16, &bf16_bytes(&x));
    let (a_hip, a_cpu, s_hip, s_cpu) = if mode == ActQuantDesc::None {
        let a_cpu = tensor(&p.cpu_mem, &[m, k], DType::BF16, &bytes(&x_hip));
        (x_hip, a_cpu, None, None)
    } else {
        let count = mode.scale_count(m, k);
        let codes = Tensor::empty(&p.hip_mem, &[m, k], DType::F8E4M3).expect("alloc");
        let scales = Tensor::empty(&p.hip_mem, &[count], DType::F32).expect("alloc");
        p.hip
            .quantize_act()
            .expect("hip quantize_act")
            .execute(&mut QuantizeActContext {
                cfg: QuantizeActConfig {
                    cols: k as u32,
                    mode,
                    x_dtype: DType::BF16,
                    out_dtype: DType::F8E4M3,
                },
                x: x_hip.view(),
                out: codes.view(),
                scales: scales.view(),
                static_scale: 0.04,
            })
            .expect("quantize_act");
        let c_cpu = tensor(&p.cpu_mem, &[m, k], DType::F8E4M3, &bytes(&codes));
        let s_cpu = tensor(&p.cpu_mem, &[count], DType::F32, &bytes(&scales));
        (codes, c_cpu, Some(scales), Some(s_cpu))
    };
    let run = |prov: &Arc<dyn KernelProvider>,
               mem: &Arc<dyn DeviceMemory>,
               a: &Tensor,
               a_scales: Option<&Tensor>,
               b: &Tensor,
               bs: &Tensor,
               prefill: bool| {
        let c = Tensor::empty(mem, &[m, n], DType::BF16).expect("alloc");
        let kernel = prov.qgemm().expect("qgemm");
        assert!(kernel.supports(&cfg), "must support qgemm {cfg}");
        kernel
            .execute(&mut QGemmContext {
                cfg,
                a: a.view(),
                a_scales: a_scales.map(Tensor::view),
                b: b.view(),
                b_scales: bs.view(),
                b_zeros: None,
                c: c.view(),
                alpha: 1.0,
                prefill,
                act_static_scale: 1.0,
            })
            .expect("qgemm");
        read_f32(&c)
    };
    // The loop's calls are prefill calls (`qgemm_for`).
    let got = run(
        &p.hip,
        &p.hip_mem,
        &a_hip,
        s_hip.as_ref(),
        &b_hip,
        &bs_hip,
        true,
    );
    let want = if dt == ExpertDtype::Fp8Block {
        // The documented contract of FP8-block weights (qgemm_fp8_block's staging, the BF16
        // checkpoint the model path matches): `bf16(e4m3(q) * s)`, then a BF16 GEMM with F32
        // accumulation. The CPU `qgemm` keeps the products unrounded, so the reference is the
        // same GEMM over those rounded weights (as the `moe_experts` reference does).
        let scales: Vec<f32> = sb
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let w: Vec<f32> = turbine_kernels::cpu::quant::dequantize(scheme, wb, &scales, None, n, k)
            .into_iter()
            .map(rb)
            .collect();
        let xr: Vec<f32> = x.iter().copied().map(rb).collect();
        (0..m)
            .flat_map(|i| {
                let (xr, w) = (&xr, &w);
                (0..n).map(move |j| {
                    let mut acc = 0f32;
                    for c in 0..k {
                        acc += xr[i * k + c] * w[j * k + c];
                    }
                    rb(acc)
                })
            })
            .collect()
    } else {
        run(
            &p.cpu,
            &p.cpu_mem,
            &a_cpu,
            s_cpu.as_ref(),
            &b_cpu,
            &bs_cpu,
            false,
        )
    };
    judge(
        &format!("stage qgemm {} {tag} n={n} k={k} m={m}", host.spec.name),
        &got,
        &want,
        BOUND_6A,
    );
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_stages_match_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x57a6_0001);
    for spec in [QWEN3, MIXTRAL, GPT_OSS] {
        // Two experts keep the weights small; the stage reads expert 1's slice.
        let spec = Spec {
            experts: 2,
            top_k: 2,
            ..spec
        };
        let (h, i) = (spec.hidden, spec.inter);
        let w8a8 = spec.dt == ExpertDtype::Fp8Tensor;
        let ms: &[usize] = if w8a8 { &[1, 16] } else { &[1, 16, 64] };
        let host = Host::new(&mut rng, spec, Variant::default());
        // gate_up (separate gate and up as the loop runs them) with the static tensor mode of
        // W8A8, down with its dynamic token mode (and the static one below).
        let (gu_mode, down_mode) = if w8a8 {
            (ActQuantDesc::Fp8Tensor, ActQuantDesc::Fp8Token)
        } else {
            (ActQuantDesc::None, ActQuantDesc::None)
        };
        for &m in ms {
            gemm_stage(&p, &mut rng, &host, &host.gate, (i, h), m, gu_mode, "gate");
            gemm_stage(&p, &mut rng, &host, &host.up, (i, h), m, gu_mode, "up");
            gemm_stage(
                &p,
                &mut rng,
                &host,
                &host.down,
                (h, i),
                m,
                down_mode,
                "down",
            );
        }
        if w8a8 {
            gemm_stage(
                &p,
                &mut rng,
                &host,
                &host.down,
                (h, i),
                16,
                ActQuantDesc::Fp8Tensor,
                "down static",
            );
        }
        if spec.dt == ExpertDtype::Mxfp4 {
            // The gpt-oss interleaved gate_up: one qgemm of 2 * inter rows.
            let gu = Host::new(
                &mut rng,
                spec,
                Variant {
                    gptoss: true,
                    ..Default::default()
                },
            );
            for &m in ms {
                gemm_stage(
                    &p,
                    &mut rng,
                    &gu,
                    &gu.gate,
                    (2 * i, h),
                    m,
                    gu_mode,
                    "gate_up",
                );
            }
        }
    }
    finish();
}

/// A projection of every expert whose row `r` has 1.0 at column `col(r)` and 0 elsewhere
/// (e4m3 0x38 / E2M1 code 2 with scale 1.0 and E8M0 127).
fn ident_proj(
    dt: ExpertDtype,
    local: usize,
    rows: usize,
    k: usize,
    col: impl Fn(usize) -> usize,
) -> HostProj {
    match dt {
        ExpertDtype::Fp8Block | ExpertDtype::Fp8Tensor => {
            let mut data = vec![0u8; local * rows * k];
            for e in 0..local {
                for r in 0..rows {
                    data[(e * rows + r) * k + col(r)] = 0x38;
                }
            }
            let (count, scale_shape) = if dt == ExpertDtype::Fp8Block {
                let (nb, kb) = (rows.div_ceil(128), k.div_ceil(128));
                (local * nb * kb, vec![local, nb, kb])
            } else {
                (local, vec![local])
            };
            HostProj {
                data,
                scales: f32_bytes(&vec![1.0; count]),
                scale_shape,
            }
        }
        ExpertDtype::Mxfp4 => {
            let mut data = vec![0u8; local * rows * k / 2];
            for e in 0..local {
                for r in 0..rows {
                    let at = (e * rows + r) * k + col(r);
                    data[at / 2] |= if at.is_multiple_of(2) { 2 } else { 2 << 4 };
                }
            }
            HostProj {
                data,
                scales: vec![127u8; local * rows * k / 32],
                scale_shape: vec![local, rows, k / 32],
            }
        }
        _ => unreachable!("quantized experts only"),
    }
}

/// Decodes expert 0 of `proj` with the reference's own dequantizer and checks that it is the
/// identity pattern `col` (so the layer's two GEMMs are exact whatever the summation order).
fn assert_identity(
    dt: ExpertDtype,
    proj: &HostProj,
    local: usize,
    rows: usize,
    k: usize,
    col: impl Fn(usize) -> usize,
) {
    let (data, scale_bytes) = expert_slice(proj, local, 0);
    let scales = if dt == ExpertDtype::Mxfp4 {
        turbine_kernels::cpu::quant::mxfp4_scales_as_f32(scale_bytes)
    } else {
        scale_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let w = turbine_kernels::cpu::quant::dequantize(scheme_of(dt), data, &scales, None, rows, k);
    for r in 0..rows {
        for c in 0..k {
            let want = if c == col(r) { 1.0 } else { 0.0 };
            assert_eq!(w[r * k + c], want, "identity weight at row {r} column {c}");
        }
    }
}

/// The identity layer: 4 experts, hidden = inter = 512, every expert the same weights, and
/// (W8A8) the same static scales, so a token's activation row is the same whichever expert
/// computes it. `gptoss` adds the interleaved gate_up, the biases and the clamped activation.
fn identity_host(dt: ExpertDtype, gptoss: bool, top_k: usize, static_down: bool) -> Host {
    let (e, h, i) = (4usize, 512usize, 512usize);
    let spec = Spec {
        name: "identity",
        experts: e,
        top_k,
        hidden: h,
        inter: i,
        dt,
    };
    let gate_rows = if gptoss { 2 * i } else { i };
    let gate_col = |r: usize| if gptoss { r / 2 } else { r };
    let w8a8 = dt == ExpertDtype::Fp8Tensor;
    let gate = ident_proj(dt, e, gate_rows, h, gate_col);
    let down = ident_proj(dt, e, h, i, |r| r);
    assert_identity(dt, &gate, e, gate_rows, h, gate_col);
    assert_identity(dt, &down, e, h, i, |r| r);
    let up = if gptoss {
        HostProj {
            data: Vec::new(),
            scales: Vec::new(),
            scale_shape: vec![0],
        }
    } else {
        ident_proj(dt, e, i, h, |r| r)
    };
    let mut rng = Rng(0x1de7);
    Host {
        spec,
        var: Variant {
            gptoss,
            shared: None,
            static_down: w8a8 && static_down,
        },
        gate,
        up,
        down,
        act_scale: w8a8.then(|| vec![0.04; e]),
        act_scale_down: (w8a8 && static_down).then(|| vec![0.03; e]),
        b_gate: gptoss.then(|| rng.normal(e * gate_rows, 3.0)),
        b_down: gptoss.then(|| rng.normal(e * h, 0.5)),
        shared: None,
        shared_gate_w: None,
    }
}

/// The activation stage (with the biases of gpt-oss): every token goes to one expert with
/// weight 1, the projections are identities, so the output row is the activation (plus the down
/// bias) of the gate/up values `x` and `x`, compared with the reference's.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn activation_stage_matches_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x57a6_0002);
    let cases = [
        (ExpertDtype::Fp8Block, false, false, "swiglu"),
        (
            ExpertDtype::Fp8Tensor,
            false,
            false,
            "swiglu w8a8 dynamic down",
        ),
        (
            ExpertDtype::Fp8Tensor,
            false,
            true,
            "swiglu w8a8 static down",
        ),
        (ExpertDtype::Mxfp4, false, false, "swiglu"),
        (ExpertDtype::Mxfp4, true, false, "gptoss clamp+bias"),
    ];
    for (dt, gptoss, static_down, tag) in cases {
        let host = identity_host(dt, gptoss, 1, static_down);
        let s = sides(&p, &host);
        for tokens in [1, 16, 256] {
            let r = route(&p, &mut rng, host.spec, tokens);
            let x = rng.normal(tokens * host.spec.hidden, 1.0);
            let out0 = vec![0f32; tokens * host.spec.hidden];
            let range = (0, host.spec.experts);
            let got = run(&p.hip, &p.hip_mem, &host, &s.0, &x, &out0, &r, range);
            let want = run(&p.cpu, &p.cpu_mem, &host, &s.1, &x, &out0, &r, range);
            judge(
                &format!("stage activation {dt:?} {tag} tokens={tokens}"),
                &got,
                &want,
                BOUND_6A,
            );
        }
    }
    finish();
}

/// The scatter stage: the same identity experts, every token to all four experts with seeded
/// weights. The input of the scatter is the activation rows HIP itself produced (a first run
/// with one expert per token and weight 1); the expected output is the scatter's specification
/// applied to them: experts in ascending id order, per element
/// `out = bf16(out + bf16(bf16(y) * bf16(w)))`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn scatter_stage_matches_reference() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let p = setup();
    let mut rng = Rng(0x57a6_0003);
    for dt in [
        ExpertDtype::Fp8Block,
        ExpertDtype::Fp8Tensor,
        ExpertDtype::Mxfp4,
    ] {
        let one = identity_host(dt, false, 1, false);
        let all = identity_host(dt, false, 4, false);
        let (e, h) = (all.spec.experts, all.spec.hidden);
        let one_sides = sides(&p, &one);
        let all_sides = sides(&p, &all);
        for tokens in [1, 16, 256] {
            let x = rng.normal(tokens * h, 1.0);
            // Every token to expert 0 with weight 1: the output is bf16(0 + bf16(y * 1)) = y.
            let mut offsets = vec![0i32; e + 1];
            offsets[1..].fill(tokens as i32);
            let first = Routing {
                weights: vec![1.0; tokens],
                sorted_rows: (0..tokens as i32).collect(),
                offsets,
            };
            let zeros = vec![0f32; tokens * h];
            let act = run(
                &p.hip,
                &p.hip_mem,
                &one,
                &one_sides.0,
                &x,
                &zeros,
                &first,
                (0, e),
            );
            let r = route(&p, &mut rng, all.spec, tokens);
            // The output buffer holds BF16: the expected chain starts from its rounded values.
            let out0: Vec<f32> = rng.normal(tokens * h, 0.1).into_iter().map(rb).collect();
            let got = run(
                &p.hip,
                &p.hip_mem,
                &all,
                &all_sides.0,
                &x,
                &out0,
                &r,
                (0, e),
            );
            let mut want = out0.clone();
            for ex in 0..e {
                let (lo, hi) = (r.offsets[ex] as usize, r.offsets[ex + 1] as usize);
                for &row in &r.sorted_rows[lo..hi] {
                    let (t, w) = (row as usize / all.spec.top_k, rb(r.weights[row as usize]));
                    for c in 0..h {
                        want[t * h + c] = rb(want[t * h + c] + rb(act[t * h + c] * w));
                    }
                }
            }
            judge(
                &format!("stage scatter {dt:?} tokens={tokens}"),
                &got,
                &want,
                BOUND_6A,
            );
        }
    }
    finish();
}
