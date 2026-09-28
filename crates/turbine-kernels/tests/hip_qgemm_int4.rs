//! Lab only (novanas R9700, Phase 6a S-9): the INT4 group-quantized GEMM implementations of
//! `libturbine_hip.so` (`turbine_hip_int4_wmma`, `turbine_hip_int4_dequant`; decision "P6: INT4
//! group GEMM — provider evaluation (kernel reuse rule)"), run through the Rust shim bindings,
//! match the `cpu-reference` provider (`cpu::qgemm`, `cpu::quant::dequantize`) on seeded random
//! inputs at the Llama-3.2-3B and Llama-3.1-8B linear shapes, for AWQ-style zero points
//! (`INT4_GROUP_ZP`) and GPTQ-style symmetric groups (`INT4_GROUP_SYM`, implicit 8), group 128.
//! Run by `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_qgemm_int4`, which sets
//! `TURBINE_TEST_BACKEND=hip`, `TURBINE_KERNEL_LIBRARY` and `TURBINE_AMD_SMI_LIBRARY`.
//!
//! Tolerance: `turbine_hip_int4_wmma` computes `a · (q − z) · s` in F32 (the codes enter the WMMA
//! as exact BF16 integers and the scale is applied per group), so against the CPU reference (an
//! exact F32 sum rounded once) it differs only by the summation order. `turbine_hip_int4_dequant`
//! rounds the dequantized weight to BF16 first and runs the BF16 GEMM, exactly what the golden
//! reference's dequantized checkpoint computes; that rounding is systematic within a group (one
//! scale times small integers), so it is compared with the CPU BF16 GEMM of the same
//! BF16-rounded weight (`cpu::quant::dequantize`, rounded to nearest even) instead. Both use the
//! Phase 1 GEMM tolerance: BF16 |Δ| ≤ 1e-2, or one BF16 ulp of the reference where its magnitude
//! is 2 or more; F32 |Δ| ≤ 1e-4.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use half::bf16;
use turbine_core::config::DevicesConfig;
use turbine_core::types::{DType, DeviceId, Vendor};
use turbine_device::{DiscoveryOptions, discover};
use turbine_kernels::cpu::quant::dequantize;
use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};
use turbine_kernels::test_support::require_backend;
use turbine_kernels::{
    GemmContext, ImplChoice, KernelProvider, OpConfig, OpKind, QGemmConfig, QGemmContext,
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
const GROUP: u32 = 128;
/// The INT4 implementations the library enumerates, in library order.
const INT4_IMPLS: [&str; 2] = ["turbine_hip_int4_wmma", "turbine_hip_int4_dequant"];

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

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
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

/// Raw contents of `v` (a HIP read synchronizes the compute stream first).
fn bytes(v: TensorView<'_>) -> Vec<u8> {
    v.slice.read_bytes().expect("read back")
}

/// Rows `[start, start + count)` of the `[rows, cols]` tensor `t`.
fn rows_of(t: &Tensor, start: usize, count: usize) -> TensorView<'_> {
    t.view().rows(start, count)
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

/// An INT4 weight `[n, k]` in the v2.9 layout: codes `[n, k / 2]` (low nibble = even column),
/// F32 scales and U8 zero points `[n, k / group]` (zero points 8 for the symmetric scheme). The
/// scales give outputs of about unit scale for unit-variance activations.
struct Weight {
    scheme: QuantSchemeDesc,
    n: usize,
    k: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
    zeros: Vec<u8>,
    /// `bf16((q − z) · s)`, `[n, k]` BF16 bytes: the weight the dequant path multiplies.
    dequant_bf16: Vec<u8>,
}

fn int4_weight(rng: &mut Rng, zp: bool, n: usize, k: usize) -> Weight {
    let groups = k / GROUP as usize;
    let codes = rng.bytes(n * k / 2);
    // (q − z) has a standard deviation of about 4.6 for uniform codes; spread the scales over
    // [0.5, 1.5] of the unit-output value so each group differs.
    let base = 1.0 / (4.6 * (k as f32).sqrt());
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| base * (0.5 + rng.unit() as f32))
        .collect();
    let zeros: Vec<u8> = if zp {
        rng.bytes(n * groups).into_iter().map(|b| b & 15).collect()
    } else {
        vec![8; n * groups]
    };
    let scheme = if zp {
        QuantSchemeDesc::Int4GroupZp { group: GROUP }
    } else {
        QuantSchemeDesc::Int4GroupSym { group: GROUP }
    };
    let zp_view = zp.then_some(zeros.as_slice());
    let dequant = dequantize(scheme, &codes, &scales, zp_view, n, k);
    let dequant_bf16 = encode(DType::BF16, &dequant);
    Weight {
        scheme,
        n,
        k,
        codes,
        scales,
        zeros,
        dequant_bf16,
    }
}

/// The weight's tensors on one device: data, scales and (ZP only) zero points.
struct WeightTensors {
    b: Tensor,
    scales: Tensor,
    zeros: Option<Tensor>,
}

fn weight_on(mem: &Arc<dyn DeviceMemory>, w: &Weight) -> WeightTensors {
    let groups = w.k / GROUP as usize;
    WeightTensors {
        b: tensor(mem, &[w.n, w.k / 2], DType::U8, &w.codes),
        scales: tensor(
            mem,
            &[w.n, groups],
            DType::F32,
            &encode(DType::F32, &w.scales),
        ),
        zeros: matches!(w.scheme, QuantSchemeDesc::Int4GroupZp { .. })
            .then(|| tensor(mem, &[w.n, groups], DType::U8, &w.zeros)),
    }
}

fn config(w: &Weight, c_dtype: DType) -> QGemmConfig {
    QGemmConfig {
        n: w.n as u32,
        k: w.k as u32,
        scheme: w.scheme,
        act_quant: ActQuantDesc::None,
        a_dtype: DType::BF16,
        c_dtype,
    }
}

/// `provider` bound to the implementation `name` of `cfg` (explicit index, as the kernel
/// registry binds it).
fn bound(p: &Pair, cfg: &QGemmConfig, name: &str) -> Arc<dyn KernelProvider> {
    let info = p
        .hip
        .implementations(OpKind::QGemm)
        .into_iter()
        .find(|i| i.name == name)
        .unwrap_or_else(|| panic!("the library has no qgemm implementation {name}"));
    let spec = OpConfig::QGemm(*cfg);
    assert!(
        p.hip.implementation_supports(&spec, info.index, None),
        "{name} must support {cfg}"
    );
    p.hip
        .bind(&spec, &ImplChoice::Single(info.index))
        .unwrap_or_else(|| panic!("bind {name}"))
}

/// Runs `provider`'s qgemm of `a` (`[m, k]` BF16 on the provider's memory) against `wt` into a
/// fresh `[m, n]` output and returns it decoded.
fn run(
    provider: &dyn KernelProvider,
    mem: &Arc<dyn DeviceMemory>,
    cfg: QGemmConfig,
    a: TensorView<'_>,
    wt: &WeightTensors,
    m: usize,
) -> Vec<f32> {
    let n = cfg.n as usize;
    let c = tensor(
        mem,
        &[m, n],
        cfg.c_dtype,
        &vec![0u8; m * n * cfg.c_dtype.size_bytes()],
    );
    let mut ctx = QGemmContext {
        cfg,
        a,
        a_scales: None,
        b: wt.b.view(),
        b_scales: wt.scales.view(),
        b_zeros: wt.zeros.as_ref().map(Tensor::view),
        c: c.view(),
        alpha: 1.0,
        prefill: m > 1,
    };
    provider
        .qgemm()
        .expect("qgemm")
        .execute(&mut ctx)
        .expect("qgemm");
    decode(cfg.c_dtype, &bytes(c.view()))
}

/// Rows the reference computes for an `m`-row product (the CPU GEMM is too slow for every row
/// of the largest shapes).
fn sample_rows(m: usize) -> Vec<usize> {
    let mut rows = vec![0, m / 7, m / 3, m / 2, (2 * m) / 3, m - 1];
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The CPU BF16 GEMM of `a` (`[rows, k]` BF16 on host memory) against the BF16-rounded
/// dequantized weight: the dequant path's arithmetic.
fn reference_bf16_weight(
    p: &Pair,
    w: &Weight,
    a: TensorView<'_>,
    rows: usize,
    c_dtype: DType,
) -> Vec<f32> {
    let b = tensor(&p.cpu_mem, &[w.n, w.k], DType::BF16, &w.dequant_bf16);
    let c = tensor(
        &p.cpu_mem,
        &[rows, w.n],
        c_dtype,
        &vec![0u8; rows * w.n * c_dtype.size_bytes()],
    );
    let mut ctx = GemmContext {
        a,
        b: b.view(),
        c: c.view(),
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
    decode(c_dtype, &bytes(c.view()))
}

/// One INT4 GEMM case: every INT4 implementation at `m` rows vs its CPU reference on the sampled
/// rows (the exact product for the fused kernel, the BF16-rounded weight's for the dequant path).
fn qgemm_case(p: &Pair, rng: &mut Rng, name: &str, m: usize, w: &Weight, c_dtype: DType) {
    let cfg = config(w, c_dtype);
    let k = w.k;
    let x = rng.normal(m * k, 1.0);
    let raw = encode(DType::BF16, &x);
    let a_hip = tensor(&p.hip_mem, &[m, k], DType::BF16, &raw);
    let hip_w = weight_on(&p.hip_mem, w);

    let rows = sample_rows(m);
    let sampled: Vec<u8> = rows
        .iter()
        .flat_map(|&r| raw[r * k * 2..(r + 1) * k * 2].iter().copied())
        .collect();
    let a_cpu = tensor(&p.cpu_mem, &[rows.len(), k], DType::BF16, &sampled);
    let cpu_w = weight_on(&p.cpu_mem, w);
    let exact = run(
        p.cpu.as_ref(),
        &p.cpu_mem,
        cfg,
        a_cpu.view(),
        &cpu_w,
        rows.len(),
    );
    let rounded = reference_bf16_weight(p, w, a_cpu.view(), rows.len(), c_dtype);
    for impl_name in INT4_IMPLS {
        let want = if impl_name == INT4_IMPLS[1] {
            &rounded
        } else {
            &exact
        };
        let provider = bound(p, &cfg, impl_name);
        let got_all = run(provider.as_ref(), &p.hip_mem, cfg, a_hip.view(), &hip_w, m);
        let n = w.n;
        let got: Vec<f32> = rows
            .iter()
            .flat_map(|&r| got_all[r * n..(r + 1) * n].iter().copied())
            .collect();
        assert_close(
            &format!("qgemm {name} m={m} {cfg} ({impl_name}) rows {rows:?}"),
            &got,
            want,
            c_dtype,
        );
    }
}

/// The v2.9 qgemm implementations the library enumerates on gfx1201: the FP8 one, then the two
/// INT4 ones (decision "P6: INT4 group GEMM — provider evaluation (kernel reuse rule)").
fn assert_implementations(p: &Pair) {
    let names: Vec<String> = p
        .hip
        .implementations(OpKind::QGemm)
        .into_iter()
        .map(|i| i.name)
        .collect();
    for name in INT4_IMPLS {
        assert!(
            names.iter().any(|n| n == name),
            "qgemm implementations {names:?} lack {name}"
        );
    }
}

/// Phase 6a S-9: `turbine_hip_int4_wmma` and `turbine_hip_int4_dequant` match the CPU reference
/// for INT4_GROUP_ZP (random zero points) and INT4_GROUP_SYM weights, group 128, M ∈ {1, 7, 16,
/// 128, 513}, every Llama-3.2-3B and Llama-3.1-8B linear shape, BF16 out, plus F32 out on one
/// shape. Breaks if a nibble is read in the wrong order, a zero point or scale is taken from the
/// wrong group or row, the symmetric zero is not 8, or the weight is read untransposed.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_int4_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    assert_implementations(&p);
    let mut rng = Rng(71);
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        for zp in [true, false] {
            let w = int4_weight(&mut rng, zp, n, k);
            for m in MS {
                qgemm_case(&p, &mut rng, name, m, &w, DType::BF16);
            }
        }
    }
    let (name, n, k) = SHAPES_3B[3];
    let w = int4_weight(&mut rng, true, n, k);
    for m in [1, 16, 128] {
        qgemm_case(&p, &mut rng, name, m, &w, DType::F32);
    }
    // Groups the kernels do not take are refused up front, never failed at run time.
    for group in [16, 48] {
        let odd = QGemmConfig {
            scheme: QuantSchemeDesc::Int4GroupSym { group },
            ..config(&w, DType::BF16)
        };
        let spec = OpConfig::QGemm(odd);
        for info in p.hip.implementations(OpKind::QGemm) {
            if INT4_IMPLS.contains(&info.name.as_str()) {
                assert!(
                    !p.hip.implementation_supports(&spec, info.index, None),
                    "{} must refuse group {group}",
                    info.name
                );
            }
        }
    }
}

/// Phase 6a S-9: a row's `turbine_hip_int4_wmma` result is bitwise the same whether it is
/// computed alone or inside a batch of 7, 16, 64 or 513 rows (the fused kernel's chain depends
/// on k and the group only). Breaks if the k split or the wave reduction order depends on m, or
/// if rows of a tile leak into each other.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn int4_rows_are_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(73);
    for (name, n, k) in [SHAPES_3B[0], SHAPES_3B[3]] {
        for zp in [true, false] {
            let w = int4_weight(&mut rng, zp, n, k);
            let cfg = config(&w, DType::BF16);
            let provider = bound(&p, &cfg, INT4_IMPLS[0]);
            let hip_w = weight_on(&p.hip_mem, &w);
            let max_m = 513;
            let x = rng.normal(max_m * k, 1.0);
            let a = tensor(
                &p.hip_mem,
                &[max_m, k],
                DType::BF16,
                &encode(DType::BF16, &x),
            );
            let batch_513 = run(provider.as_ref(), &p.hip_mem, cfg, a.view(), &hip_w, max_m);
            for m in [7, 16, 64] {
                let batch = run(
                    provider.as_ref(),
                    &p.hip_mem,
                    cfg,
                    rows_of(&a, 0, m),
                    &hip_w,
                    m,
                );
                assert_eq!(
                    batch.to_bits(),
                    batch_513[..m * n].to_bits(),
                    "{name} zp={zp}: rows of a {m}-row batch differ from the 513-row batch"
                );
            }
            for r in sample_rows(max_m) {
                let alone = run(
                    provider.as_ref(),
                    &p.hip_mem,
                    cfg,
                    rows_of(&a, r, 1),
                    &hip_w,
                    1,
                );
                assert_eq!(
                    alone.to_bits(),
                    batch_513[r * n..(r + 1) * n].to_bits(),
                    "{name} zp={zp}: row {r} alone differs from the 513-row batch"
                );
            }
            println!("{name} zp={zp}: rows batch-invariant");
        }
    }
}

trait Bits {
    fn to_bits(&self) -> Vec<u32>;
}

impl Bits for [f32] {
    fn to_bits(&self) -> Vec<u32> {
        self.iter().map(|v| v.to_bits()).collect()
    }
}
