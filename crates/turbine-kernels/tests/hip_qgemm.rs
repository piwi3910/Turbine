//! Lab only (novanas R9700, Phase 6a S-7): the ABI v2.9 quantized GEMM and activation
//! quantization of `libturbine_hip.so`, run through the Rust shim bindings, match the
//! `cpu-reference` provider (`cpu::qgemm`, `cpu::quant`) on seeded random inputs at the
//! Llama-3.2-3B and Llama-3.1-8B linear shapes. Run by
//! `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_qgemm`, which sets
//! `TURBINE_TEST_BACKEND=hip`, `TURBINE_KERNEL_LIBRARY` and `TURBINE_AMD_SMI_LIBRARY`.
//!
//! Tolerance: `quantize_act` is bit-exact (codes and scales). `qgemm`: e4m3 × e4m3 products are
//! exact in F32, so GPU and reference differ only by the F32 summation order before the one
//! rounding to the output dtype: the Phase 1 GEMM tolerance (BF16 |Δ| ≤ 1e-2, or one BF16 ulp
//! of the reference where its magnitude is 2 or more; F32 |Δ| ≤ 1e-4) at outputs of unit scale.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use half::bf16;
use turbine_core::config::DevicesConfig;
use turbine_core::types::{DType, DeviceId, Vendor};
use turbine_device::{DiscoveryOptions, discover};
use turbine_kernels::cpu::quant::{fp8_e4m3_round, fp8_e4m3_value};
use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};
use turbine_kernels::test_support::require_backend;
use turbine_kernels::{
    GemmConfig, GemmContext, ImplChoice, KernelProvider, OpConfig, OpKind, QGemmConfig,
    QGemmContext, QuantizeActConfig, QuantizeActContext, RowTier, ShimLibrary,
    cpu_reference_provider, shim_provider,
};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

/// (name, n, k) of the dense linear layers (fused Q/K/V and gate/up as the executor runs them).
const SHAPES_3B: [(&str, usize, usize); 4] = [
    ("3b.qkv", 5120, 3072),
    ("3b.o", 3072, 3072),
    ("3b.gate_up", 16384, 3072),
    ("3b.down", 3072, 8192),
];
const SHAPES_8B: [(&str, usize, usize); 4] = [
    ("8b.qkv", 6144, 4096),
    ("8b.o", 4096, 4096),
    ("8b.gate_up", 28672, 4096),
    ("8b.down", 4096, 14336),
];
const MS: [usize; 5] = [1, 7, 16, 128, 513];

/// The HIP provider and the CPU reference, each with the memory its tensors live in.
struct Pair {
    hip: Arc<dyn KernelProvider>,
    cpu: Arc<dyn KernelProvider>,
    hip_mem: Arc<dyn DeviceMemory>,
    cpu_mem: Arc<dyn DeviceMemory>,
}

impl Pair {
    /// The same pair with `hip` in place of the HIP provider (a bound implementation).
    fn with_hip(&self, hip: Arc<dyn KernelProvider>) -> Pair {
        Pair {
            hip,
            cpu: Arc::clone(&self.cpu),
            hip_mem: Arc::clone(&self.hip_mem),
            cpu_mem: Arc::clone(&self.cpu_mem),
        }
    }
}

/// The FP8 W8A8 (FP8_TENSOR / FP8_CHANNEL × FP8_TENSOR / FP8_TOKEN) implementations.
const FP8_W8A8_IMPLS: [&str; 2] = ["hipblaslt_fp8", "turbine_hip_fp8_wmma"];

/// One test at a time per process: device discovery (amd-smi) and the per-thread context.
static GPU: Mutex<()> = Mutex::new(());

fn lock_gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn setup() -> Pair {
    let inventory =
        discover(&DiscoveryOptions::from_config(&DevicesConfig::default())).expect("discovery");
    let device = inventory
        .devices
        .iter()
        .find(|d| d.vendor == Vendor::Amd)
        .unwrap_or_else(|| panic!("no AMD device in the inventory: {:?}", inventory.backends));
    let path = PathBuf::from(
        std::env::var_os("TURBINE_KERNEL_LIBRARY")
            .filter(|v| !v.is_empty())
            .expect("TURBINE_KERNEL_LIBRARY is not set; point it at libturbine_hip.so"),
    );
    let lib = ShimLibrary::load(&path, "hip").expect("load libturbine_hip.so");
    let ctx = lib.create_context(device).expect("HIP context");
    let hip_mem: Arc<dyn DeviceMemory> = ctx.clone();
    Pair {
        hip: shim_provider(ctx),
        cpu: cpu_reference_provider(),
        hip_mem,
        cpu_mem: HostMemory::new(DeviceId(u32::MAX), 16 << 30),
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
}

fn encode(dtype: DType, v: &[f32]) -> Vec<u8> {
    match dtype {
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect(),
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        other => panic!("unsupported test dtype {}", other.as_str()),
    }
}

fn decode(dtype: DType, b: &[u8]) -> Vec<f32> {
    match dtype {
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

fn tensor(mem: &Arc<dyn DeviceMemory>, shape: &[usize], dtype: DType, raw: &[u8]) -> Tensor {
    let mut t = Tensor::empty(mem, shape, dtype).expect("alloc");
    t.storage.copy_from_host(0, raw).expect("copy in");
    t
}

/// The same bytes in HIP memory and in host memory.
fn twin(p: &Pair, shape: &[usize], dtype: DType, raw: &[u8]) -> (Tensor, Tensor) {
    (
        tensor(&p.hip_mem, shape, dtype, raw),
        tensor(&p.cpu_mem, shape, dtype, raw),
    )
}

/// Raw contents of `v` (a HIP read synchronizes the compute stream first).
fn bytes(v: TensorView<'_>) -> Vec<u8> {
    v.slice.read_bytes().expect("read back")
}

/// Columns `[0, cols)` of every row of the `[rows, stride]` tensor `t` (row-strided view).
fn column_prefix(t: &Tensor, cols: usize) -> TensorView<'_> {
    let (rows, stride) = (t.shape[0], t.shape[1]);
    let es = t.dtype.size_bytes();
    TensorView {
        slice: t.storage.whole().sub(0, ((rows - 1) * stride + cols) * es),
        shape: (&[rows, cols][..]).into(),
        strides: (&[stride, 1][..]).into(),
        dtype: t.dtype,
    }
}

/// |Δ| ≤ 1e-2, or one BF16 ulp of `want` where |want| ≥ 2.
fn bf16_close(got: f32, want: f32) -> bool {
    let d = (got - want).abs();
    if d <= 1e-2 {
        return true;
    }
    if want.abs() >= 2.0 {
        let b = bf16::from_f32(want.abs());
        let ulp = bf16::from_bits(b.to_bits() + 1).to_f32() - b.to_f32();
        return d <= ulp;
    }
    false
}

fn assert_close(what: &str, got: &[f32], want: &[f32], dtype: DType) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        let ok = match dtype {
            DType::F32 => d <= 1e-4,
            _ => bf16_close(g, w),
        };
        assert!(ok, "{what}: element {i}: hip {g} vs cpu {w} (|Δ| {d})");
        if d > worst.1 {
            worst = (i, d);
        }
    }
    println!("{what}: max |Δ| {:.3e} at {} ok", worst.1, worst.0);
}

/// e4m3 weights `[n, k]` of a unit-variance output layer quantized per `scheme` (per-row amax or
/// the whole tensor's), and their F32 scales.
fn fp8_weights(rng: &mut Rng, scheme: QuantSchemeDesc, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let w = rng.normal(n * k, 1.0 / (k as f32).sqrt());
    let row_scale = |r: &[f32]| r.iter().fold(0f32, |m, v| m.max(v.abs())) / 448.0;
    let scales: Vec<f32> = match scheme {
        QuantSchemeDesc::Fp8Tensor => vec![row_scale(&w)],
        QuantSchemeDesc::Fp8Channel => w.chunks_exact(k).map(row_scale).collect(),
        other => panic!("not an FP8 per-tensor/channel scheme: {other:?}"),
    };
    let q = w
        .chunks_exact(k)
        .enumerate()
        .flat_map(|(r, row)| {
            let s = scales[if scales.len() == 1 { 0 } else { r }];
            row.iter().map(move |v| fp8_e4m3_round(v / s))
        })
        .collect();
    (q, scales)
}

/// Quantizes `x` (`[rows, cols]` of `x_dtype`, row stride `x_stride`) on both providers and
/// asserts the codes and scales are bitwise equal. Returns the HIP output (`[rows, cols]` e4m3)
/// and scales.
#[allow(clippy::too_many_arguments)]
fn quantize_case(
    p: &Pair,
    what: &str,
    x: &[f32],
    rows: usize,
    cols: usize,
    x_stride: usize,
    x_dtype: DType,
    mode: ActQuantDesc,
    static_scale: f32,
) -> (Tensor, Tensor) {
    let cfg = QuantizeActConfig {
        cols: cols as u32,
        mode,
        x_dtype,
        out_dtype: DType::F8E4M3,
    };
    let hip = p.hip.quantize_act().expect("hip quantize_act (ABI v2.9)");
    assert!(hip.supports(&cfg), "hip must support quantize_act {cfg}");
    let impl_name = hip.implementation(&cfg);
    let raw = encode(x_dtype, x);
    let (x_hip, x_cpu) = twin(p, &[rows, x_stride], x_dtype, &raw);
    let n_scales = mode.scale_count(rows, cols);
    let zeros = vec![0u8; rows * cols];
    let (o_hip, o_cpu) = twin(p, &[rows, cols], DType::F8E4M3, &zeros);
    let (s_hip, s_cpu) = twin(p, &[n_scales], DType::F32, &vec![0u8; 4 * n_scales]);
    let runs: [(
        &dyn turbine_kernels::QuantizeActKernel,
        &Tensor,
        &Tensor,
        &Tensor,
    ); 2] = [
        (hip, &x_hip, &o_hip, &s_hip),
        (
            p.cpu.quantize_act().expect("cpu quantize_act"),
            &x_cpu,
            &o_cpu,
            &s_cpu,
        ),
    ];
    for (kernel, x, o, s) in runs {
        let mut ctx = QuantizeActContext {
            cfg,
            x: column_prefix(x, cols),
            out: o.view(),
            scales: s.view(),
            static_scale,
        };
        kernel.execute(&mut ctx).expect("quantize_act");
    }
    let (got, want) = (bytes(o_hip.view()), bytes(o_cpu.view()));
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{what} ({impl_name}): code {i} (row {}, col {}): hip {:#04x} vs cpu {:#04x}",
            i / cols,
            i % cols,
            got[i],
            want[i]
        );
    }
    let (gs, ws) = (bytes(s_hip.view()), bytes(s_cpu.view()));
    assert_eq!(gs, ws, "{what} ({impl_name}): scales differ");
    println!(
        "{what}: impl={impl_name} bit-exact ({} codes, {n_scales} scales)",
        got.len()
    );
    (o_hip, s_hip)
}

/// Rows the reference computes for an `m`-row product (the CPU GEMM is too slow for every row
/// of the largest shapes).
fn sample_rows(m: usize) -> Vec<usize> {
    let mut rows = vec![0, m / 7, m / 3, m / 2, (2 * m) / 3, m - 1];
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// One quantized GEMM case: BF16 activations quantized on the GPU per `act`, then `qgemm` on the
/// GPU at `m` rows vs the CPU reference on the sampled rows (fed the same e4m3 codes and scales).
#[allow(clippy::too_many_arguments)]
fn qgemm_case(
    p: &Pair,
    rng: &mut Rng,
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    scheme: QuantSchemeDesc,
    act: ActQuantDesc,
    c_dtype: DType,
    weights: &(Vec<u8>, Vec<f32>),
) {
    let cfg = QGemmConfig {
        n: n as u32,
        k: k as u32,
        scheme,
        act_quant: act,
        a_dtype: DType::F8E4M3,
        c_dtype,
    };
    let what = format!("qgemm {name} m={m} {cfg}");
    let hip = p.hip.qgemm().expect("hip qgemm (ABI v2.9)");
    assert!(hip.supports(&cfg), "hip must support {what}");
    let impl_name = hip.implementation(&cfg);

    let x = rng.normal(m * k, 1.0);
    let static_scale = x.iter().fold(0f32, |a, v| a.max(v.abs())) / 448.0 * 1.25;
    let (a_hip, a_scales_hip) = quantize_case(
        p,
        &format!("quantize_act for {what}"),
        &x,
        m,
        k,
        k,
        DType::BF16,
        act,
        static_scale,
    );
    let (wq, ws) = weights;
    let ws_raw = encode(DType::F32, ws);
    let (b_hip, b_cpu) = twin(p, &[n, k], DType::F8E4M3, wq);
    let (bs_hip, bs_cpu) = twin(p, &[ws.len()], DType::F32, &ws_raw);
    let c_hip = tensor(
        &p.hip_mem,
        &[m, n],
        c_dtype,
        &vec![0u8; m * n * c_dtype.size_bytes()],
    );
    let mut ctx = QGemmContext {
        cfg,
        a: a_hip.view(),
        a_scales: Some(a_scales_hip.view()),
        b: b_hip.view(),
        b_scales: bs_hip.view(),
        b_zeros: None,
        c: c_hip.view(),
        alpha: 1.0,
        // Decode-sized calls take the pinned FP8 table rows (m <= 64), larger ones the prefill
        // path (the first heuristic answer).
        prefill: m > 64,
        act_static_scale: 1.0,
    };
    hip.execute(&mut ctx).expect("hip qgemm");
    let got_all = decode(c_dtype, &bytes(c_hip.view()));

    // The reference on the sampled rows, fed the GPU's own e4m3 codes and scales (quantize_act
    // was just checked bit-exact against the CPU's).
    let codes = bytes(a_hip.view());
    let a_scales = decode(DType::F32, &bytes(a_scales_hip.view()));
    let rows = sample_rows(m);
    let s = rows.len();
    let sampled_codes: Vec<u8> = rows
        .iter()
        .flat_map(|&r| codes[r * k..(r + 1) * k].iter().copied())
        .collect();
    let sampled_scales: Vec<f32> = match act {
        ActQuantDesc::Fp8Tensor => vec![a_scales[0]],
        ActQuantDesc::Fp8Token => rows.iter().map(|&r| a_scales[r]).collect(),
        other => panic!("not an FP8 tensor/token mode: {other:?}"),
    };
    let a_cpu = tensor(&p.cpu_mem, &[s, k], DType::F8E4M3, &sampled_codes);
    let as_cpu = tensor(
        &p.cpu_mem,
        &[sampled_scales.len()],
        DType::F32,
        &encode(DType::F32, &sampled_scales),
    );
    let c_cpu = tensor(
        &p.cpu_mem,
        &[s, n],
        c_dtype,
        &vec![0u8; s * n * c_dtype.size_bytes()],
    );
    let mut ctx = QGemmContext {
        cfg,
        a: a_cpu.view(),
        a_scales: Some(as_cpu.view()),
        b: b_cpu.view(),
        b_scales: bs_cpu.view(),
        b_zeros: None,
        c: c_cpu.view(),
        alpha: 1.0,
        prefill: false,
        act_static_scale: 1.0,
    };
    p.cpu
        .qgemm()
        .expect("cpu qgemm")
        .execute(&mut ctx)
        .expect("cpu qgemm");
    let want = decode(c_dtype, &bytes(c_cpu.view()));
    let got: Vec<f32> = rows
        .iter()
        .flat_map(|&r| got_all[r * n..(r + 1) * n].iter().copied())
        .collect();
    assert_close(
        &format!("{what} ({impl_name}) rows {rows:?}"),
        &got,
        &want,
        c_dtype,
    );
}

/// The v2.9 implementations the library enumerates on gfx1201 (decision "P6: FP8 GEMM —
/// provider evaluation (kernel reuse rule)").
fn assert_implementations(p: &Pair) {
    let names = |op| -> Vec<String> {
        p.hip
            .implementations(op)
            .into_iter()
            .map(|i| i.name)
            .collect()
    };
    assert_eq!(
        names(OpKind::QGemm),
        [
            "hipblaslt_fp8",
            "turbine_hip_int4_wmma",
            "turbine_hip_int4_dequant",
            "turbine_hip_mxfp4",
            "turbine_hip_fp8_block",
            "turbine_hip_fp8_wmma"
        ]
    );
    assert_eq!(
        names(OpKind::QuantizeAct),
        ["turbine_hip", "turbine_hip_mxfp4"]
    );
}

/// Phase 6a S-7 (and 5p Task 14): `hipblaslt_fp8` and `turbine_hip_fp8_wmma`, each bound alone,
/// match the CPU reference for FP8_TENSOR and FP8_CHANNEL
/// weights × FP8_TENSOR (static) and FP8_TOKEN (dynamic) activations, M ∈ {1, 7, 16, 128, 513},
/// every Llama-3.2-3B and Llama-3.1-8B linear shape, BF16 out (F32 out is refused by
/// `supports`).
/// Breaks if a scale is applied to the wrong operand or axis (per-token vs per-channel
/// swapped), if the weight is read untransposed, or if the output rounding differs.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    assert_implementations(&p);
    let mut rng = Rng(61);
    let combos = [
        (QuantSchemeDesc::Fp8Channel, ActQuantDesc::Fp8Token),
        (QuantSchemeDesc::Fp8Tensor, ActQuantDesc::Fp8Tensor),
        (QuantSchemeDesc::Fp8Channel, ActQuantDesc::Fp8Tensor),
        (QuantSchemeDesc::Fp8Tensor, ActQuantDesc::Fp8Token),
    ];
    // Each FP8 W8A8 implementation bound alone (hipblaslt_fp8 at every m; turbine_hip_fp8_wmma
    // on its kernel up to 16 decode rows, on hipblaslt_fp8 above).
    let providers: Vec<(String, Pair)> = FP8_W8A8_IMPLS
        .iter()
        .map(|&wanted| {
            let info = p
                .hip
                .implementations(OpKind::QGemm)
                .into_iter()
                .find(|i| i.name == wanted)
                .unwrap_or_else(|| panic!("{wanted} is not in the library"));
            let spec = OpConfig::QGemm(QGemmConfig {
                n: 3072,
                k: 3072,
                scheme: QuantSchemeDesc::Fp8Channel,
                act_quant: ActQuantDesc::Fp8Token,
                a_dtype: DType::F8E4M3,
                c_dtype: DType::BF16,
            });
            let hip = p
                .hip
                .bind(&spec, &ImplChoice::Single(info.index))
                .unwrap_or_else(|| panic!("bind {wanted}"));
            (wanted.to_string(), p.with_hip(hip))
        })
        .collect();
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        for (scheme, act) in combos {
            let weights = fp8_weights(&mut rng, scheme, n, k);
            for m in MS {
                for (bound, pair) in &providers {
                    println!("qgemm_matches_cpu: {bound}");
                    qgemm_case(
                        pair,
                        &mut rng,
                        name,
                        m,
                        n,
                        k,
                        scheme,
                        act,
                        DType::BF16,
                        &weights,
                    );
                }
            }
        }
    }
    // F32 out is refused up front (gfx1201's hipBLASLt has no FP8 kernel with vector scales and
    // an F32 D; quantized linear layers write BF16), never failed at run time.
    let (_, n, k) = SHAPES_3B[0];
    let f32_out = QGemmConfig {
        n: n as u32,
        k: k as u32,
        scheme: QuantSchemeDesc::Fp8Channel,
        act_quant: ActQuantDesc::Fp8Token,
        a_dtype: DType::F8E4M3,
        c_dtype: DType::F32,
    };
    assert!(!p.hip.qgemm().expect("hip qgemm").supports(&f32_out));
}

/// Rows of a BF16 activation matrix that exercise the e4m3 rounding edges, cycled over `m`
/// rows: unit-variance rows, an all-zero row (the 1 / (448 · 512) token-scale floor), a row
/// 10⁴ times larger (saturation at ±448 under a static scale), a row in the e4m3 subnormal
/// range, and a row on the e4m3 grid and its halfway points (ties).
fn edge_rows(rng: &mut Rng, m: usize, k: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(m * k);
    for r in 0..m {
        let row: Vec<f32> = match r % 5 {
            0 => rng.normal(k, 1.0),
            1 => vec![0.0; k],
            2 => rng.normal(k, 1e4),
            3 => rng.normal(k, 1e-4),
            _ => (0..k)
                .map(|c| (1.0 + (c % 16) as f32 / 16.0) * if c % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        };
        x.extend(row.into_iter().map(|v| bf16::from_f32(v).to_f32()));
    }
    x
}

/// Phase 5p (kernel ABI v2.13, user decision "5p Task 14" 1 A): `turbine_hip_fp8_wmma` handed the
/// BF16 activations (`a_dtype` BF16, the FP8 act mode and static scale) gives the same bits as
/// `quantize_act` followed by the same implementation on its e4m3 codes, for FP8_TENSOR /
/// FP8_CHANNEL weights × FP8_TENSOR (static) / FP8_TOKEN (dynamic) activations, every 3B and 8B
/// shape, the single-row decode call it takes that form for, on random rows and rounding-edge
/// rows (zero, saturating, subnormal, ties). Bit-equal outputs over every
/// column mean bit-equal codes and scales (an e4m3 code off by one moves the sum). Also: the
/// form is offered exactly for single-row decode calls (not prefill, not 2 or 17 rows), and
/// through a row-tier binding only in the tier the kernel serves. Breaks if the
/// in-kernel quantization divides by another scale, reduces the amax differently (a NaN or
/// signed zero kept), rounds differently from `fp8_e4m3_round`, or sums in another order.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_fused_act_quant_matches_two_launches() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let find = |wanted: &str| {
        p.hip
            .implementations(OpKind::QGemm)
            .into_iter()
            .find(|i| i.name == wanted)
            .unwrap_or_else(|| panic!("{wanted} is not in the library"))
    };
    let wmma = find("turbine_hip_fp8_wmma");
    let hipblaslt = find("hipblaslt_fp8");
    let quantize = p.hip.quantize_act().expect("hip quantize_act (ABI v2.9)");
    let mut rng = Rng(97);
    let combos = [
        (QuantSchemeDesc::Fp8Channel, ActQuantDesc::Fp8Token),
        (QuantSchemeDesc::Fp8Tensor, ActQuantDesc::Fp8Tensor),
        (QuantSchemeDesc::Fp8Channel, ActQuantDesc::Fp8Tensor),
        (QuantSchemeDesc::Fp8Tensor, ActQuantDesc::Fp8Token),
    ];
    let mut cases = 0usize;
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        for (scheme, act) in combos {
            let cfg = QGemmConfig {
                n: n as u32,
                k: k as u32,
                scheme,
                act_quant: act,
                a_dtype: DType::F8E4M3,
                c_dtype: DType::BF16,
            };
            let spec = OpConfig::QGemm(cfg);
            let bound = p
                .hip
                .bind(&spec, &ImplChoice::Single(wmma.index))
                .expect("bind turbine_hip_fp8_wmma");
            let kernel = bound.qgemm().expect("qgemm");
            // gfx1201's tiers: the kernel up to 16 rows, hipBLASLt above (no BF16 form there).
            let tiered = p
                .hip
                .bind(
                    &spec,
                    &ImplChoice::ByRows(vec![
                        RowTier {
                            max_rows: Some(16),
                            index: wmma.index,
                        },
                        RowTier {
                            max_rows: None,
                            index: hipblaslt.index,
                        },
                    ]),
                )
                .expect("bind the row tiers");
            assert!(
                !kernel.quantizes_activations(&cfg, 1, true),
                "{name} prefill"
            );
            assert!(
                !kernel.quantizes_activations(&cfg, 17, false),
                "{name} m=17"
            );
            assert!(!kernel.quantizes_activations(&cfg, 2, false), "{name} m=2");
            let tiered_q = tiered.qgemm().expect("qgemm");
            assert!(
                tiered_q.quantizes_activations(&cfg, 1, false),
                "{name} tier"
            );
            assert!(
                !tiered_q.quantizes_activations(&cfg, 32, false),
                "{name} open tier"
            );

            let (wq, ws) = fp8_weights(&mut rng, scheme, n, k);
            let b = tensor(&p.hip_mem, &[n, k], DType::F8E4M3, &wq);
            let bs = tensor(
                &p.hip_mem,
                &[ws.len()],
                DType::F32,
                &encode(DType::F32, &ws),
            );
            let m = 1usize;
            assert!(kernel.quantizes_activations(&cfg, m, false), "{name} m=1");
            let edges = edge_rows(&mut rng, 5, k);
            for (kind, x) in edges.chunks_exact(k).enumerate() {
                let static_scale =
                    rng.normal(k, 1.0).iter().fold(0f32, |a, v| a.max(v.abs())) / 448.0;
                let a = tensor(&p.hip_mem, &[m, k], DType::BF16, &encode(DType::BF16, x));
                let codes = tensor(&p.hip_mem, &[m, k], DType::F8E4M3, &vec![0u8; m * k]);
                let count = act.scale_count(m, k);
                let a_scales = tensor(&p.hip_mem, &[count], DType::F32, &vec![0u8; 4 * count]);
                quantize
                    .execute(&mut QuantizeActContext {
                        cfg: QuantizeActConfig {
                            cols: k as u32,
                            mode: act,
                            x_dtype: DType::BF16,
                            out_dtype: DType::F8E4M3,
                        },
                        x: a.view(),
                        out: codes.view(),
                        scales: a_scales.view(),
                        static_scale,
                    })
                    .expect("quantize_act");
                let run = |cfg: QGemmConfig, a: &Tensor, a_scales: Option<&Tensor>| {
                    let c = tensor(&p.hip_mem, &[m, n], DType::BF16, &vec![0xffu8; m * n * 2]);
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
                            prefill: false,
                            act_static_scale: static_scale,
                        })
                        .expect("qgemm");
                    bytes(c.view())
                };
                let two = run(cfg, &codes, Some(&a_scales));
                let fused = QGemmConfig {
                    a_dtype: DType::BF16,
                    ..cfg
                };
                let one = run(fused, &a, None);
                if let Some(i) =
                    (0..m * n).find(|&i| one[2 * i..2 * i + 2] != two[2 * i..2 * i + 2])
                {
                    panic!(
                        "{name} {cfg} row kind {kind}: element {i} (row {}, col {}): fused {:?} vs \
                         two launches {:?}",
                        i / n,
                        i % n,
                        &one[2 * i..2 * i + 2],
                        &two[2 * i..2 * i + 2]
                    );
                }
                cases += 1;
            }
            println!("qgemm_fused_act_quant {name} {cfg}: bit-identical on 5 row kinds");
        }
    }
    assert!(cases > 0);
}

/// Phase 6a S-7: the HIP `quantize_act` is bit-exact with the CPU reference (codes and scales)
/// for FP8_TENSOR (static), FP8_TOKEN and FP8_GROUP128 at 1, 7, 16, 128 and 513 rows of every
/// 3B / 8B input width, from BF16 and from F32, on dense and row-strided input, including an
/// all-zero row (the 1 / (448 · 512) floor), a row of huge values (saturation at ±448), exact
/// ties of the e4m3 grid and values in the subnormal range. Breaks on a rounding that is not to
/// nearest-even, a scale computed as a reciprocal product, or a missing floor or saturation.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn quantize_act_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    assert_implementations(&p);
    let mut rng = Rng(67);
    let modes = [
        ActQuantDesc::Fp8Token,
        ActQuantDesc::Fp8Group { group: 128 },
        ActQuantDesc::Fp8Tensor,
    ];
    for mode in modes {
        for cols in [3072, 8192, 4096, 14336] {
            for rows in MS {
                let mut x = rng.normal(rows * cols, 1.0);
                if rows >= 3 {
                    x[..cols].fill(0.0);
                    for v in &mut x[cols..2 * cols] {
                        *v *= 1e4;
                    }
                    // Row 2: exact e4m3 grid points and midpoints relative to a unit scale,
                    // and subnormal-range values, bracketed by ±448 so the dynamic scale is 1.
                    let row = &mut x[2 * cols..3 * cols];
                    for (i, v) in row.iter_mut().enumerate() {
                        let code = (i % 126) as u8;
                        let lo = fp8_e4m3_value(code);
                        let hi = fp8_e4m3_value(code + 1);
                        *v = match i % 3 {
                            0 => lo,
                            1 => (lo + hi) / 2.0,
                            _ => lo / 3.0,
                        };
                        if i % 2 == 1 {
                            *v = -*v;
                        }
                    }
                    row[0] = 448.0;
                }
                let static_scale = if rows >= 3 { 1.0 } else { 0.0173 };
                quantize_case(
                    &p,
                    &format!("quantize_act {} rows={rows} cols={cols}", mode.as_str()),
                    &x,
                    rows,
                    cols,
                    cols,
                    DType::BF16,
                    mode,
                    static_scale,
                );
            }
        }
        // F32 input and a row-strided view (a column prefix of wider rows).
        let (rows, cols, stride) = (7, 3072, 3072 + 128);
        let x = rng.normal(rows * stride, 2.0);
        quantize_case(
            &p,
            &format!("quantize_act {} f32 strided", mode.as_str()),
            &x,
            rows,
            cols,
            stride,
            DType::F32,
            mode,
            0.02,
        );
    }
}

/// Rows `[first, first + count)` of the dense `[rows, cols]` (or `[rows]`) tensor `t`.
fn row_range(t: &Tensor, first: usize, count: usize) -> TensorView<'_> {
    let cols = t.shape.get(1).copied().unwrap_or(1);
    let es = t.dtype.size_bytes();
    let (shape, strides): (Vec<usize>, Vec<usize>) = if t.shape.len() == 2 {
        (vec![count, cols], vec![cols, 1])
    } else {
        (vec![count], vec![1])
    };
    TensorView {
        slice: t.storage.whole().sub(first * cols * es, count * cols * es),
        shape: shape.as_slice().into(),
        strides: strides.as_slice().into(),
        dtype: t.dtype,
    }
}

/// Phase 6a S-7 with Phase 4 prefix reuse: in a prefill step (`QGemmContext::prefill`) a row's
/// `hipblaslt_fp8` result does not depend on how many rows the call has or where the row sits,
/// so the prefill of a prompt's suffix (after a reused prefix) computes each row bit for bit as
/// the whole-prompt prefill does. For every 3B shape and scale pairing, rows of a 513-row prefill
/// are compared bitwise with the same rows run as calls of 1, 7, 128 and 213 rows (a suffix at
/// row 300). Breaks if the prefill algorithm depends on m or runs with split-K.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_prefill_rows_do_not_depend_on_m() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(71);
    let m = 513;
    let hip = p.hip.qgemm().expect("hip qgemm (ABI v2.9)");
    for (name, n, k) in SHAPES_3B {
        for (scheme, act) in [
            (QuantSchemeDesc::Fp8Channel, ActQuantDesc::Fp8Token),
            (QuantSchemeDesc::Fp8Channel, ActQuantDesc::Fp8Tensor),
        ] {
            let cfg = QGemmConfig {
                n: n as u32,
                k: k as u32,
                scheme,
                act_quant: act,
                a_dtype: DType::F8E4M3,
                c_dtype: DType::BF16,
            };
            let x = rng.normal(m * k, 1.0);
            let (a, a_scales) = quantize_case(
                &p,
                &format!("quantize_act for prefill {name}"),
                &x,
                m,
                k,
                k,
                DType::BF16,
                act,
                0.02,
            );
            let (wq, ws) = fp8_weights(&mut rng, scheme, n, k);
            let b = tensor(&p.hip_mem, &[n, k], DType::F8E4M3, &wq);
            let bs = tensor(
                &p.hip_mem,
                &[ws.len()],
                DType::F32,
                &encode(DType::F32, &ws),
            );
            let run = |first: usize, rows: usize| -> Vec<u8> {
                let c = tensor(
                    &p.hip_mem,
                    &[rows, n],
                    DType::BF16,
                    &vec![0u8; rows * n * 2],
                );
                let scales = match act {
                    ActQuantDesc::Fp8Token => row_range(&a_scales, first, rows),
                    _ => a_scales.view(),
                };
                let mut ctx = QGemmContext {
                    cfg,
                    a: row_range(&a, first, rows),
                    a_scales: Some(scales),
                    b: b.view(),
                    b_scales: bs.view(),
                    b_zeros: None,
                    c: c.view(),
                    alpha: 1.0,
                    prefill: true,
                    act_static_scale: 1.0,
                };
                hip.execute(&mut ctx).expect("hip qgemm prefill");
                bytes(c.view())
            };
            let whole = run(0, m);
            for (first, rows) in [(0, 1), (0, 7), (0, 128), (300, 213)] {
                let part = run(first, rows);
                let want = &whole[first * n * 2..(first + rows) * n * 2];
                assert!(
                    part.as_slice() == want,
                    "qgemm prefill {name} {cfg}: rows {first}..{} differ between a {rows}-row \
                     call and the {m}-row call",
                    first + rows
                );
            }
            println!("qgemm prefill {name} {cfg}: rows independent of m ok");
        }
    }
}

const FP8_BLOCK: QuantSchemeDesc = QuantSchemeDesc::Fp8Block {
    block_n: 128,
    block_k: 128,
};

/// e4m3 weights `[n, k]` with one F32 scale per 128 × 128 block (`[n/128, k/128]`, per-block
/// amax / 448) of a unit-variance output layer.
fn fp8_block_weights(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let w = rng.normal(n * k, 1.0 / (k as f32).sqrt());
    let (nb, kb) = (n.div_ceil(128), k.div_ceil(128));
    let mut scales = vec![0f32; nb * kb];
    for r in 0..n {
        for c in 0..k {
            let s = &mut scales[(r / 128) * kb + c / 128];
            *s = s.max(w[r * k + c].abs());
        }
    }
    for s in &mut scales {
        *s /= 448.0;
    }
    let q = (0..n * k)
        .map(|i| fp8_e4m3_round(w[i] / scales[(i / k / 128) * kb + (i % k) / 128]))
        .collect();
    (q, scales)
}

/// Rows up to which `turbine_hip_fp8_block` runs a decode step on its fused kernel
/// (`kFusedMaxRows` in `qgemm_fp8_block.hip`); prefill steps and larger decode batches take the
/// dequantize path.
const FP8_BLOCK_FUSED_MAX_ROWS: usize = 64;

/// One FP8_BLOCK weight in both of the forms the two paths are judged against: the codes and
/// block scales (the CPU `qgemm` reference, exact `q · s` products), and the BF16 weight
/// `bf16(e4m3(q) · s)` the dequantize path stages (the golden reference's dequantized checkpoint
/// holds the same values), for the CPU BF16 `gemm` reference.
struct Fp8BlockWeight {
    codes: Vec<u8>,
    scales: Vec<f32>,
    dequantized_bf16: Vec<u8>,
}

impl Fp8BlockWeight {
    fn new(codes: Vec<u8>, scales: Vec<f32>, k: usize) -> Self {
        let kb = k.div_ceil(128);
        let dequantized: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(i, &q)| fp8_e4m3_value(q) * scales[(i / k / 128) * kb + (i % k) / 128])
            .collect();
        Self {
            codes,
            scales,
            dequantized_bf16: encode(DType::BF16, &dequantized),
        }
    }
}

/// One FP8_BLOCK case: BF16 activations (W8A16) × block-scaled e4m3 weights on the GPU at `m`
/// rows (`prefill` picks the step kind), judged on the sampled rows with the Phase 1 BF16 GEMM
/// tolerance against the reference of the path the implementation takes: the fused kernel against
/// the CPU `qgemm` (exact `q · s`; the kernel applies each block's scale to its exact partial sum,
/// so only the F32 summation order differs), the dequantize path against the CPU BF16 `gemm` of
/// the BF16-rounded weight `bf16(e4m3(q) · s)` (what it multiplies; off the exact products by up
/// to 2^-9 relative per weight, which the Phase 1 bound does not cover at k = 3072).
#[allow(clippy::too_many_arguments)]
fn fp8_block_case(
    p: &Pair,
    rng: &mut Rng,
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    prefill: bool,
    weight: &Fp8BlockWeight,
) {
    let cfg = QGemmConfig {
        n: n as u32,
        k: k as u32,
        scheme: FP8_BLOCK,
        act_quant: ActQuantDesc::None,
        a_dtype: DType::BF16,
        c_dtype: DType::BF16,
    };
    let fused = !prefill && m <= FP8_BLOCK_FUSED_MAX_ROWS;
    let path = if fused { "fused" } else { "dequantize" };
    let what = format!("qgemm {name} m={m} prefill={prefill} {cfg} [{path}]");
    let hip = p.hip.qgemm().expect("hip qgemm (ABI v2.9)");
    assert!(hip.supports(&cfg), "hip must support {what}");
    let impl_name = hip.implementation(&cfg);
    let x = rng.normal(m * k, 1.0);
    let a_raw = encode(DType::BF16, &x);
    let a_hip = tensor(&p.hip_mem, &[m, k], DType::BF16, &a_raw);
    let b_hip = tensor(&p.hip_mem, &[n, k], DType::F8E4M3, &weight.codes);
    let bs_hip = tensor(
        &p.hip_mem,
        &[weight.scales.len()],
        DType::F32,
        &encode(DType::F32, &weight.scales),
    );
    let c_hip = tensor(&p.hip_mem, &[m, n], DType::BF16, &vec![0u8; m * n * 2]);
    let mut ctx = QGemmContext {
        cfg,
        a: a_hip.view(),
        a_scales: None,
        b: b_hip.view(),
        b_scales: bs_hip.view(),
        b_zeros: None,
        c: c_hip.view(),
        alpha: 1.0,
        prefill,
        act_static_scale: 1.0,
    };
    hip.execute(&mut ctx).expect("hip qgemm");
    let got_all = decode(DType::BF16, &bytes(c_hip.view()));

    let rows = sample_rows(m);
    let s = rows.len();
    let sampled: Vec<u8> = rows
        .iter()
        .flat_map(|&r| a_raw[r * k * 2..(r + 1) * k * 2].iter().copied())
        .collect();
    let a_cpu = tensor(&p.cpu_mem, &[s, k], DType::BF16, &sampled);
    let c_cpu = tensor(&p.cpu_mem, &[s, n], DType::BF16, &vec![0u8; s * n * 2]);
    if fused {
        let b_cpu = tensor(&p.cpu_mem, &[n, k], DType::F8E4M3, &weight.codes);
        let bs_cpu = tensor(
            &p.cpu_mem,
            &[weight.scales.len()],
            DType::F32,
            &encode(DType::F32, &weight.scales),
        );
        let mut ctx = QGemmContext {
            cfg,
            a: a_cpu.view(),
            a_scales: None,
            b: b_cpu.view(),
            b_scales: bs_cpu.view(),
            b_zeros: None,
            c: c_cpu.view(),
            alpha: 1.0,
            prefill: false,
            act_static_scale: 1.0,
        };
        p.cpu
            .qgemm()
            .expect("cpu qgemm")
            .execute(&mut ctx)
            .expect("cpu qgemm");
    } else {
        let b_cpu = tensor(&p.cpu_mem, &[n, k], DType::BF16, &weight.dequantized_bf16);
        let mut ctx = GemmContext {
            a: a_cpu.view(),
            b: b_cpu.view(),
            c: c_cpu.view(),
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
            prefill: false,
        };
        p.cpu
            .gemm()
            .expect("cpu gemm")
            .execute(&mut ctx)
            .expect("cpu gemm");
    }
    let want = decode(DType::BF16, &bytes(c_cpu.view()));
    let got: Vec<f32> = rows
        .iter()
        .flat_map(|&r| got_all[r * n..(r + 1) * n].iter().copied())
        .collect();
    assert_close(
        &format!("{what} ({impl_name}) rows {rows:?}"),
        &got,
        &want,
        DType::BF16,
    );
}

/// Phase 6a S-8 (user decision: own kernel): `turbine_hip_fp8_block` matches the CPU reference
/// for 128 × 128 block-scaled e4m3 weights × BF16 activations, M ∈ {1, 7, 16, 128, 513}, every
/// Llama-3.2-3B and Llama-3.1-8B linear shape, in decode steps (the fused kernel up to 64 rows,
/// dequantize + BF16 GEMM above) and prefill steps (dequantize + the tuned table's invariant BF16
/// GEMM). Breaks if a block scale is applied to the wrong block, a code decodes wrongly (e.g. an
/// FNUZ instead of an OCP e4m3 conversion), or the k order of the two operands disagrees.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_fp8_block_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(73);
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        let (codes, scales) = fp8_block_weights(&mut rng, n, k);
        let weight = Fp8BlockWeight::new(codes, scales, k);
        for m in MS {
            for prefill in [false, true] {
                fp8_block_case(&p, &mut rng, name, m, n, k, prefill, &weight);
            }
        }
    }
}

/// Phase 6a S-8 with Phase 4 prefix reuse: `turbine_hip_fp8_block`'s rows do not depend on the
/// call. Prefill steps: rows of a 513-row call equal the same rows computed as calls of 1, 7, 128
/// and 213 (from row 300) rows. Decode steps (the fused kernel): rows of a 64-row call equal
/// calls of 1, 7 and 16 (from row 30) rows. Every Llama-3.2-3B shape (the tuned table has their
/// invariant rows). Breaks if a path's summation order depends on m or on a row's position.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_fp8_block_rows_do_not_depend_on_m() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(79);
    let hip = p.hip.qgemm().expect("hip qgemm (ABI v2.9)");
    for (name, n, k) in SHAPES_3B {
        let cfg = QGemmConfig {
            n: n as u32,
            k: k as u32,
            scheme: FP8_BLOCK,
            act_quant: ActQuantDesc::None,
            a_dtype: DType::BF16,
            c_dtype: DType::BF16,
        };
        let (wq, ws) = fp8_block_weights(&mut rng, n, k);
        let b = tensor(&p.hip_mem, &[n, k], DType::F8E4M3, &wq);
        let bs = tensor(
            &p.hip_mem,
            &[ws.len()],
            DType::F32,
            &encode(DType::F32, &ws),
        );
        for (prefill, m, parts) in [
            (true, 513, &[(0, 1), (0, 7), (0, 128), (300, 213)][..]),
            (false, 64, &[(0, 1), (0, 7), (30, 16)][..]),
        ] {
            let x = rng.normal(m * k, 1.0);
            let a = tensor(&p.hip_mem, &[m, k], DType::BF16, &encode(DType::BF16, &x));
            let run = |first: usize, rows: usize| -> Vec<u8> {
                let c = tensor(
                    &p.hip_mem,
                    &[rows, n],
                    DType::BF16,
                    &vec![0u8; rows * n * 2],
                );
                let mut ctx = QGemmContext {
                    cfg,
                    a: row_range(&a, first, rows),
                    a_scales: None,
                    b: b.view(),
                    b_scales: bs.view(),
                    b_zeros: None,
                    c: c.view(),
                    alpha: 1.0,
                    prefill,
                    act_static_scale: 1.0,
                };
                hip.execute(&mut ctx).expect("hip qgemm fp8 block");
                bytes(c.view())
            };
            let whole = run(0, m);
            for &(first, rows) in parts {
                let part = run(first, rows);
                assert!(
                    part.as_slice() == &whole[first * n * 2..(first + rows) * n * 2],
                    "qgemm fp8 block {name} prefill={prefill}: rows {first}..{} differ between a \
                     {rows}-row call and the {m}-row call",
                    first + rows
                );
            }
            println!("qgemm fp8 block {name} prefill={prefill}: rows independent of m ok");
        }
    }
}

/// Host-timed µs per call of `op` over `iters` back-to-back calls, after `warm` (at least two)
/// warm-up calls: a pass over every cycled weight copy, so no timing pays first-touch costs.
fn time_us(p: &Pair, warm: u32, iters: u32, mut op: impl FnMut()) -> f64 {
    for _ in 0..warm.max(2) {
        op();
    }
    p.hip_mem.synchronize().expect("synchronize");
    let start = std::time::Instant::now();
    for _ in 0..iters {
        op();
    }
    p.hip_mem.synchronize().expect("synchronize");
    start.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
}

/// Lab microbenchmark (Phase 5p Task 14, S-15; no assertion on speed): the decode-step linear
/// layers of Llama-3.2-3B at m ∈ {1, 2, 4, 8, 16} rows, per weight format the FP8 checkpoints
/// serve — the BF16 GEMM (the tuned table), `qgemm` FP8_TENSOR × FP8_TENSOR (per-tensor
/// checkpoint), FP8_CHANNEL × FP8_TOKEN (FP8-dynamic) and FP8_BLOCK × BF16 (FP8-block) — and the
/// `quantize_act` before them, each timed alone over back-to-back calls with the weights cycled
/// through ≥ 512 MiB of copies (never cache-resident, as in a forward). Prints one `qgemm_timing`
/// line per (shape, format, m) with the weight-read bandwidth. Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_qgemm -- qgemm_decode_timings --nocapture`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_decode_timings() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(83);
    const ITERS: u32 = 100;
    const TIMING_MS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
    let max_m = TIMING_MS[TIMING_MS.len() - 1];
    let gemm = p.hip.gemm().expect("hip gemm");
    let quantize = p.hip.quantize_act().expect("hip quantize_act (ABI v2.9)");
    let report = |shape: &str, format: &str, impl_name: &str, m: usize, us: f64, bytes: usize| {
        println!(
            "qgemm_timing {shape:<11} {format:<22} m={m:<2} impl={impl_name:<24} {us:>8.1} us \
             {:>5.0} GB/s",
            bytes as f64 / (us * 1e3)
        );
    };
    for (name, n, k) in SHAPES_3B {
        let a_bf16 = tensor(
            &p.hip_mem,
            &[max_m, k],
            DType::BF16,
            &encode(DType::BF16, &rng.normal(max_m * k, 1.0)),
        );
        let a_fp8 = tensor(
            &p.hip_mem,
            &[max_m, k],
            DType::F8E4M3,
            &vec![0x38u8; max_m * k],
        );
        let a_scales = tensor(
            &p.hip_mem,
            &[max_m],
            DType::F32,
            &encode(DType::F32, &vec![0.01f32; max_m]),
        );
        let c = tensor(
            &p.hip_mem,
            &[max_m, n],
            DType::BF16,
            &vec![0u8; max_m * n * 2],
        );

        // BF16 GEMM on the tuned table (the BF16 checkpoint's decode layer).
        {
            let bytes = n * k * 2;
            let raw = encode(DType::BF16, &rng.normal(n * k, 1.0 / (k as f32).sqrt()));
            let copies = (512usize << 20).div_ceil(bytes).max(2);
            let ws: Vec<Tensor> = (0..copies)
                .map(|_| tensor(&p.hip_mem, &[n, k], DType::BF16, &raw))
                .collect();
            let cfg = GemmConfig {
                n: n as u64,
                k: k as u64,
                trans_b: true,
                a_dtype: DType::BF16,
                b_dtype: DType::BF16,
                c_dtype: DType::BF16,
                gemv: false,
            };
            for m in TIMING_MS {
                let mut next = 0usize;
                let us = time_us(&p, copies as u32, ITERS, || {
                    next = (next + 1) % copies;
                    gemm.execute(&mut GemmContext {
                        a: row_range(&a_bf16, 0, m),
                        b: ws[next].view(),
                        c: row_range(&c, 0, m),
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("gemm");
                });
                report(name, "bf16", &gemm.implementation(&cfg), m, us, bytes);
            }
        }

        // The FP8 formats: e4m3 weights, n·k bytes (the scales are negligible).
        let formats = [
            (
                "fp8 tensor x tensor",
                QuantSchemeDesc::Fp8Tensor,
                ActQuantDesc::Fp8Tensor,
            ),
            (
                "fp8 channel x token",
                QuantSchemeDesc::Fp8Channel,
                ActQuantDesc::Fp8Token,
            ),
            ("fp8 block x bf16", FP8_BLOCK, ActQuantDesc::None),
        ];
        for (format, scheme, act) in formats {
            let block = matches!(scheme, QuantSchemeDesc::Fp8Block { .. });
            let (codes, scales) = if block {
                fp8_block_weights(&mut rng, n, k)
            } else {
                fp8_weights(&mut rng, scheme, n, k)
            };
            let bytes = n * k;
            let copies = (512usize << 20).div_ceil(bytes).max(2);
            let ws: Vec<(Tensor, Tensor)> = (0..copies)
                .map(|_| {
                    (
                        tensor(&p.hip_mem, &[n, k], DType::F8E4M3, &codes),
                        tensor(
                            &p.hip_mem,
                            &[scales.len()],
                            DType::F32,
                            &encode(DType::F32, &scales),
                        ),
                    )
                })
                .collect();
            let cfg = QGemmConfig {
                n: n as u32,
                k: k as u32,
                scheme,
                act_quant: act,
                a_dtype: if block { DType::BF16 } else { DType::F8E4M3 },
                c_dtype: DType::BF16,
            };
            // Every implementation that takes the format, each bound alone.
            let spec = OpConfig::QGemm(cfg);
            let runs: Vec<(String, Arc<dyn KernelProvider>)> = p
                .hip
                .implementations(OpKind::QGemm)
                .into_iter()
                .filter(|info| p.hip.implementation_supports(&spec, info.index, None))
                .map(|info| {
                    let bound = p
                        .hip
                        .bind(&spec, &ImplChoice::Single(info.index))
                        .expect("bind a qgemm implementation");
                    (info.name, bound)
                })
                .collect();
            for (impl_name, provider) in &runs {
                let kernel = provider.qgemm().expect("qgemm");
                for m in TIMING_MS {
                    let mut next = 0usize;
                    let us = time_us(&p, copies as u32, ITERS, || {
                        next = (next + 1) % copies;
                        let (a, a_s) = if block {
                            (row_range(&a_bf16, 0, m), None)
                        } else {
                            (row_range(&a_fp8, 0, m), Some(row_range(&a_scales, 0, m)))
                        };
                        kernel
                            .execute(&mut QGemmContext {
                                cfg,
                                a,
                                a_scales: a_s,
                                b: ws[next].0.view(),
                                b_scales: ws[next].1.view(),
                                b_zeros: None,
                                c: row_range(&c, 0, m),
                                alpha: 1.0,
                                prefill: false,
                                act_static_scale: 1.0,
                            })
                            .expect("qgemm");
                    });
                    report(name, format, impl_name, m, us, bytes);
                    // The same call on BF16 activations, quantized in the kernel (v2.13).
                    if !block && kernel.quantizes_activations(&cfg, m, false) {
                        let fused = QGemmConfig {
                            a_dtype: DType::BF16,
                            ..cfg
                        };
                        let mut next = 0usize;
                        let us = time_us(&p, copies as u32, ITERS, || {
                            next = (next + 1) % copies;
                            kernel
                                .execute(&mut QGemmContext {
                                    cfg: fused,
                                    a: row_range(&a_bf16, 0, m),
                                    a_scales: None,
                                    b: ws[next].0.view(),
                                    b_scales: ws[next].1.view(),
                                    b_zeros: None,
                                    c: row_range(&c, 0, m),
                                    alpha: 1.0,
                                    prefill: false,
                                    act_static_scale: 0.01,
                                })
                                .expect("qgemm (BF16 activations)");
                        });
                        report(name, &format!("{format} +q"), impl_name, m, us, bytes);
                    }
                }
            }
        }
    }

    // The activation quantization before each W8A8 layer.
    for k in [3072usize, 8192] {
        let x = tensor(
            &p.hip_mem,
            &[max_m, k],
            DType::BF16,
            &encode(DType::BF16, &rng.normal(max_m * k, 1.0)),
        );
        let out = tensor(
            &p.hip_mem,
            &[max_m, k],
            DType::F8E4M3,
            &vec![0u8; max_m * k],
        );
        let scales = tensor(&p.hip_mem, &[max_m], DType::F32, &vec![0u8; 4 * max_m]);
        for mode in [ActQuantDesc::Fp8Tensor, ActQuantDesc::Fp8Token] {
            let cfg = QuantizeActConfig {
                cols: k as u32,
                mode,
                x_dtype: DType::BF16,
                out_dtype: DType::F8E4M3,
            };
            for m in TIMING_MS {
                let us = time_us(&p, 2, ITERS, || {
                    quantize
                        .execute(&mut QuantizeActContext {
                            cfg,
                            x: row_range(&x, 0, m),
                            out: row_range(&out, 0, m),
                            scales: row_range(&scales, 0, mode.scale_count(m, k).max(1)),
                            static_scale: 0.01,
                        })
                        .expect("quantize_act");
                });
                report(
                    &format!("k={k}"),
                    &format!("quantize {}", mode.as_str()),
                    &quantize.implementation(&cfg),
                    m,
                    us,
                    m * k * 2,
                );
            }
        }
    }
}
