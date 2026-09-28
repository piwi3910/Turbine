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
    KernelProvider, OpKind, QGemmConfig, QGemmContext, QuantizeActConfig, QuantizeActContext,
    ShimLibrary, cpu_reference_provider, shim_provider,
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
        prefill: m > 1,
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
            "turbine_hip_int4_dequant"
        ]
    );
    assert_eq!(names(OpKind::QuantizeAct), ["turbine_hip"]);
}

/// Phase 6a S-7: `hipblaslt_fp8` matches the CPU reference for FP8_TENSOR and FP8_CHANNEL
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
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        for (scheme, act) in combos {
            let weights = fp8_weights(&mut rng, scheme, n, k);
            for m in MS {
                qgemm_case(
                    &p,
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
