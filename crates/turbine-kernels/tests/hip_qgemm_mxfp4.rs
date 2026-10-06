//! Lab only (novanas R9700, Phase 6a S-10): the ABI v2.9 MXFP4 quantized GEMM and the MXFP4
//! activation emulation of `libturbine_hip.so` (`turbine_hip_mxfp4`), run through the Rust shim
//! bindings, match the `cpu-reference` provider (`cpu::qgemm`, `cpu::quant`) on seeded random
//! inputs at the Llama-3.2-3B and Llama-3.1-8B linear shapes. Run by
//! `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_qgemm_mxfp4`, which sets
//! `TURBINE_TEST_BACKEND=hip`, `TURBINE_KERNEL_LIBRARY` and `TURBINE_AMD_SMI_LIBRARY`.
//!
//! Tolerance: `quantize_act` (MXFP4_EMULATED) is bit-exact (values and group scales). `qgemm`:
//! every E2M1 × 2^(E8M0 − 127) weight and every product with a BF16 activation is exact in F32,
//! so GPU and reference differ only by the F32 summation order before the one rounding to the
//! output dtype: the Phase 1 GEMM tolerance (BF16 |Δ| ≤ 1e-2, or one BF16 ulp of the reference
//! where its magnitude is 2 or more; F32 |Δ| ≤ 1e-4 plus 1e-6 relative) at outputs of unit
//! scale.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use half::bf16;
use turbine_core::config::DevicesConfig;
use turbine_core::types::{DType, DeviceId, Vendor};
use turbine_device::{DiscoveryOptions, discover};
use turbine_kernels::cpu::quant::{e2m1_value, mxfp4_quantize_group};
use turbine_kernels::quant::{ActQuantDesc, MX_BLOCK, QuantSchemeDesc};
use turbine_kernels::test_support::require_backend;
use turbine_kernels::{
    GemmConfig, GemmContext, KernelProvider, OpKind, QGemmConfig, QGemmContext, QuantizeActConfig,
    QuantizeActContext, ShimLibrary, cpu_reference_provider, shim_provider,
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
/// Every tile of `turbine_hip_mxfp4` (≤ 16, ≤ 32, ≤ 64, above 64 and from 1,024 rows) and partial
/// tiles.
const MS: [usize; 8] = [1, 7, 16, 24, 40, 128, 513, 1031];

/// The implementation name of both ops (decision "P6: MXFP4 GEMM — provider evaluation").
const IMPL: &str = "turbine_hip_mxfp4";

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
            DType::F32 => d <= 1e-4 + 1e-6 * w.abs(),
            _ => bf16_close(g, w),
        };
        assert!(ok, "{what}: element {i}: hip {g} vs cpu {w} (|Δ| {d})");
        if d > worst.1 {
            worst = (i, d);
        }
    }
    println!("{what}: max |Δ| {:.3e} at {} ok", worst.1, worst.0);
}

/// MXFP4 weights `[n, k]` of a unit-variance output layer: N(0, 1/k) values quantized per
/// 32-column group by the reference quantizer (Quark `even` exponents). Returns the packed codes
/// (`[n, k/2]`) and E8M0 exponents (`[n, k/32]`).
fn mxfp4_weights(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let w = rng.normal(n * k, 1.0 / (k as f32).sqrt());
    let mut codes = Vec::with_capacity(n * k / 2);
    let mut exps = Vec::with_capacity(n * k / MX_BLOCK);
    for g in w.chunks_exact(MX_BLOCK) {
        let (packed, e) = mxfp4_quantize_group(g.try_into().expect("one group"));
        codes.extend_from_slice(&packed);
        exps.push(e);
    }
    (codes, exps)
}

/// Runs `quantize_act` (MXFP4_EMULATED) on both providers over `x` (`[rows, cols]` of `x_dtype`,
/// row stride `x_stride`, written to a `[rows, cols]` `out_dtype` tensor) and asserts the values
/// and scales are bitwise equal. Returns the HIP output.
#[allow(clippy::too_many_arguments)]
fn quantize_case(
    p: &Pair,
    what: &str,
    x: &[f32],
    rows: usize,
    cols: usize,
    x_stride: usize,
    x_dtype: DType,
    out_dtype: DType,
) -> Tensor {
    let mode = ActQuantDesc::Mxfp4Emulated;
    let cfg = QuantizeActConfig {
        cols: cols as u32,
        mode,
        x_dtype,
        out_dtype,
    };
    let hip = p.hip.quantize_act().expect("hip quantize_act (ABI v2.9)");
    assert!(hip.supports(&cfg), "hip must support quantize_act {cfg}");
    assert_eq!(hip.implementation(&cfg), IMPL, "{what}");
    let raw = encode(x_dtype, x);
    let (x_hip, x_cpu) = twin(p, &[rows, x_stride], x_dtype, &raw);
    let n_scales = mode.scale_count(rows, cols);
    let zeros = vec![0u8; rows * cols * out_dtype.size_bytes()];
    let (o_hip, o_cpu) = twin(p, &[rows, cols], out_dtype, &zeros);
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
            static_scale: 1.0,
        };
        kernel.execute(&mut ctx).expect("quantize_act");
    }
    let (got, want) = (bytes(o_hip.view()), bytes(o_cpu.view()));
    let es = out_dtype.size_bytes();
    if let Some(i) =
        (0..got.len() / es).find(|&i| got[i * es..(i + 1) * es] != want[i * es..(i + 1) * es])
    {
        let (g, w) = (
            decode(out_dtype, &got[i * es..(i + 1) * es])[0],
            decode(out_dtype, &want[i * es..(i + 1) * es])[0],
        );
        panic!(
            "{what}: value {i} (row {}, col {}): hip {g:e} vs cpu {w:e} (input {:e})",
            i / cols,
            i % cols,
            x[(i / cols) * x_stride + i % cols]
        );
    }
    let (gs, ws) = (bytes(s_hip.view()), bytes(s_cpu.view()));
    assert_eq!(gs, ws, "{what}: scales differ");
    println!(
        "{what}: bit-exact ({} values, {n_scales} scales)",
        got.len() / es
    );
    o_hip
}

/// Rows the reference computes for an `m`-row product (the CPU GEMM is too slow for every row
/// of the largest shapes).
fn sample_rows(m: usize) -> Vec<usize> {
    let mut rows = vec![0, m / 7, m / 3, m / 2, (2 * m) / 3, m - 1];
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// One MXFP4 GEMM case: BF16 activations (quantize-dequantized to MXFP4 on the GPU first when
/// `act` is MXFP4_EMULATED, as the decoder runs Quark W4A4 layers), then `qgemm` on the GPU at
/// `m` rows vs the CPU reference on the sampled rows fed the same activations. From 1,024 rows
/// every call runs the Prefill tile; below, a decode-step call (`prefill: false`) picks its
/// tile by m and a prefill-step call (`prefill: true`) runs the Large tile, which gives a row
/// the Prefill tile's bits (`qgemm_mxfp4_prefill_rows_are_batch_invariant` checks it).
#[allow(clippy::too_many_arguments)]
fn qgemm_case(
    p: &Pair,
    rng: &mut Rng,
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    act: ActQuantDesc,
    c_dtype: DType,
    weights: &(Vec<u8>, Vec<u8>),
    prefill: bool,
) {
    let cfg = QGemmConfig {
        n: n as u32,
        k: k as u32,
        scheme: QuantSchemeDesc::Mxfp4,
        act_quant: act,
        a_dtype: DType::BF16,
        c_dtype,
    };
    let what = format!("qgemm {name} m={m} prefill={prefill} {cfg}");
    let hip = p.hip.qgemm().expect("hip qgemm (ABI v2.9)");
    assert!(hip.supports(&cfg), "hip must support {what}");
    assert_eq!(hip.implementation(&cfg), IMPL, "{what}");

    let x = rng.normal(m * k, 1.0);
    let a_hip = if act == ActQuantDesc::Mxfp4Emulated {
        quantize_case(
            p,
            &format!("quantize_act for {what}"),
            &x,
            m,
            k,
            k,
            DType::BF16,
            DType::BF16,
        )
    } else {
        tensor(&p.hip_mem, &[m, k], DType::BF16, &encode(DType::BF16, &x))
    };
    let (codes, exps) = weights;
    let (b_hip, b_cpu) = twin(p, &[n, k / 2], DType::U8, codes);
    let (bs_hip, bs_cpu) = twin(p, &[exps.len()], DType::U8, exps);
    let c_hip = tensor(
        &p.hip_mem,
        &[m, n],
        c_dtype,
        &vec![0u8; m * n * c_dtype.size_bytes()],
    );
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
    let got_all = decode(c_dtype, &bytes(c_hip.view()));

    // The reference on the sampled rows, fed the GPU's own activations (quantize_act was just
    // checked bit-exact against the CPU's).
    let a_all = bytes(a_hip.view());
    let rows = sample_rows(m);
    let s = rows.len();
    let sampled: Vec<u8> = rows
        .iter()
        .flat_map(|&r| a_all[r * k * 2..(r + 1) * k * 2].iter().copied())
        .collect();
    let a_cpu = tensor(&p.cpu_mem, &[s, k], DType::BF16, &sampled);
    let c_cpu = tensor(
        &p.cpu_mem,
        &[s, n],
        c_dtype,
        &vec![0u8; s * n * c_dtype.size_bytes()],
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
    let want = decode(c_dtype, &bytes(c_cpu.view()));
    let got: Vec<f32> = rows
        .iter()
        .flat_map(|&r| got_all[r * n..(r + 1) * n].iter().copied())
        .collect();
    assert_close(
        &format!("{what} ({IMPL}) rows {rows:?}"),
        &got,
        &want,
        c_dtype,
    );
}

/// The MXFP4 implementations the library enumerates on gfx1201 (decision "P6: MXFP4 GEMM —
/// provider evaluation (kernel reuse rule)").
fn assert_implementations(p: &Pair) {
    for op in [OpKind::QGemm, OpKind::QuantizeAct] {
        let names: Vec<String> = p
            .hip
            .implementations(op)
            .into_iter()
            .map(|i| i.name)
            .collect();
        assert!(
            names.iter().any(|n| n == IMPL),
            "{op}: {IMPL} not among {names:?}"
        );
    }
}

/// Phase 6a S-10: `turbine_hip_mxfp4` matches the CPU reference for MXFP4 weights × BF16
/// activations (act NONE, the W4A16 checkpoints) and × MXFP4-emulated activations (Quark W4A4),
/// M ∈ {1, 7, 16, 24, 40, 128, 513, 1031} (every tile, full and partial), every Llama-3.2-3B and
/// Llama-3.1-8B linear shape, BF16 out, plus F32 out and alpha ≠ 1 on one shape.
/// Breaks if a nibble is read in the wrong order, a group's scale is applied to the wrong
/// columns, the half-wave code exchange pairs the wrong k values with the activations, or a
/// tile drops or duplicates rows or columns.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_mxfp4_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    assert_implementations(&p);
    let mut rng = Rng(71);
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        let weights = mxfp4_weights(&mut rng, n, k);
        for m in MS {
            for act in [ActQuantDesc::None, ActQuantDesc::Mxfp4Emulated] {
                for prefill in [false, true] {
                    qgemm_case(
                        &p,
                        &mut rng,
                        name,
                        m,
                        n,
                        k,
                        act,
                        DType::BF16,
                        &weights,
                        prefill,
                    );
                }
            }
        }
    }
    // F32 out, a non-multiple-of-16 n and odd m.
    let (n, k) = (1000, 1024);
    let weights = mxfp4_weights(&mut rng, n, k);
    for m in [3, 33, 200, 300] {
        for prefill in [false, true] {
            qgemm_case(
                &p,
                &mut rng,
                "odd",
                m,
                n,
                k,
                ActQuantDesc::None,
                DType::F32,
                &weights,
                prefill,
            );
        }
    }
}

/// Phase 6a S-10: `turbine_hip_mxfp4`'s MXFP4_EMULATED `quantize_act` is bit-exact with the CPU
/// reference (values and group scales) at 1, 7, 16, 24, 40, 128, 513 and 1,031 rows of every 3B / 8B input
/// width, from BF16 and from F32, to BF16 and F32, on dense and row-strided input and a ragged
/// last group, including all-zero groups (scale 2^-127), groups of exact E2M1 grid points and
/// midpoints (ties to the even code), groups whose maximum rounds the scale up (Quark `even`:
/// mantissa ≥ 1.75), saturating values, huge and subnormal-range values. Breaks on OCP's floor
/// scale rule, rounding ties away from zero, a signed zero, or a missing saturation.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn quantize_act_mxfp4_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    assert_implementations(&p);
    let mut rng = Rng(73);
    for cols in [3072, 8192, 4096, 14336] {
        for rows in MS {
            let mut x = rng.normal(rows * cols, 1.0);
            if rows >= 4 {
                x[..cols].fill(0.0);
                // Row 1: grid points and midpoints at scale 2^-3, the group maximum 6 or 7
                // times the scale (7 rounds the `even` exponent up).
                for (i, v) in x[cols..2 * cols].iter_mut().enumerate() {
                    let code = (i % 8) as u8;
                    let lo = e2m1_value(code);
                    let hi = e2m1_value((code + 1).min(7));
                    *v = if i % MX_BLOCK == 0 {
                        if (i / MX_BLOCK).is_multiple_of(2) {
                            6.0
                        } else {
                            7.0
                        }
                    } else if i % 3 == 0 {
                        (lo + hi) / 2.0
                    } else {
                        lo
                    } * 0.125;
                    if i % 2 == 1 {
                        *v = -*v;
                    }
                }
                // Row 2: huge values; row 3: values in F32's subnormal range after scaling.
                for v in &mut x[2 * cols..3 * cols] {
                    *v *= 1e30;
                }
                for v in &mut x[3 * cols..4 * cols] {
                    *v *= 1e-36;
                }
            }
            quantize_case(
                &p,
                &format!("quantize_act mxfp4 rows={rows} cols={cols}"),
                &x,
                rows,
                cols,
                cols,
                DType::BF16,
                DType::BF16,
            );
        }
    }
    // F32 in and out, a row-strided view with a ragged last group.
    let (rows, cols, stride) = (7, 3072 + 20, 3072 + 128);
    let x = rng.normal(rows * stride, 2.0);
    for (x_dtype, out_dtype) in [
        (DType::F32, DType::F32),
        (DType::F32, DType::BF16),
        (DType::BF16, DType::F32),
    ] {
        quantize_case(
            &p,
            &format!(
                "quantize_act mxfp4 {} -> {} strided ragged",
                x_dtype.as_str(),
                out_dtype.as_str()
            ),
            &x,
            rows,
            cols,
            stride,
            x_dtype,
            out_dtype,
        );
    }
}

/// The HIP `qgemm` of `a` (`[m, k]` BF16) with the MXFP4 weights `b` / `bs`, BF16 out, as a
/// prefill-step call; the raw output bytes.
fn prefill_call(p: &Pair, cfg: QGemmConfig, a: TensorView<'_>, b: &Tensor, bs: &Tensor) -> Vec<u8> {
    let (m, n) = (a.shape[0], cfg.n as usize);
    let c = tensor(&p.hip_mem, &[m, n], DType::BF16, &vec![0u8; m * n * 2]);
    let mut ctx = QGemmContext {
        cfg,
        a,
        a_scales: None,
        b: b.view(),
        b_scales: bs.view(),
        b_zeros: None,
        c: c.view(),
        alpha: 1.0,
        prefill: true,
        act_static_scale: 1.0,
    };
    p.hip
        .qgemm()
        .expect("hip qgemm (ABI v2.9)")
        .execute(&mut ctx)
        .expect("hip qgemm");
    bytes(c.view())
}

/// Rows `[first, first + rows)` of the `[m, k]` tensor `t` as an `[rows, k]` view.
fn row_range(t: &Tensor, first: usize, rows: usize) -> TensorView<'_> {
    let k = t.shape[1];
    let es = t.dtype.size_bytes();
    TensorView {
        slice: t.storage.whole().sub(first * k * es, rows * k * es),
        shape: (&[rows, k][..]).into(),
        strides: (&[k, 1][..]).into(),
        dtype: t.dtype,
    }
}

/// Phase 4 prefix reuse under MXFP4 weights: a prefill-step `qgemm` gives every row the same
/// bits whatever the call's row count and the row's position in it — rows of a 1,031-row call
/// equal the same rows computed in calls of 1, 7 and 128 rows, and in a 128-row call starting at
/// row 200 — on a 3B and an 8B shape; the 1,031-row call runs the Prefill tile, the shorter ones
/// the Large tile. Breaks if a prefill call picks a tile with another summation order by m
/// (e.g. the k-split decode tiles for a short prefill chunk), or if the Prefill and Large tiles
/// sum a row differently.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_mxfp4_prefill_rows_are_batch_invariant() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(79);
    let m_full = 1031;
    for (name, n, k) in [SHAPES_3B[0], SHAPES_8B[3]] {
        let cfg = QGemmConfig {
            n: n as u32,
            k: k as u32,
            scheme: QuantSchemeDesc::Mxfp4,
            act_quant: ActQuantDesc::None,
            a_dtype: DType::BF16,
            c_dtype: DType::BF16,
        };
        let (codes, exps) = mxfp4_weights(&mut rng, n, k);
        let b = tensor(&p.hip_mem, &[n, k / 2], DType::U8, &codes);
        let bs = tensor(&p.hip_mem, &[exps.len()], DType::U8, &exps);
        let x = rng.normal(m_full * k, 1.0);
        let a = tensor(
            &p.hip_mem,
            &[m_full, k],
            DType::BF16,
            &encode(DType::BF16, &x),
        );
        let full = prefill_call(&p, cfg, a.view(), &b, &bs);
        let row_bytes = n * 2;
        for (first, rows) in [(0, 1), (0, 7), (0, 128), (200, 128)] {
            let part = prefill_call(&p, cfg, row_range(&a, first, rows), &b, &bs);
            let want = &full[first * row_bytes..(first + rows) * row_bytes];
            let differs = |i: &usize| part[2 * i..2 * i + 2] != want[2 * i..2 * i + 2];
            if let Some(i) = (0..part.len() / 2).find(differs) {
                panic!(
                    "{name}: row {} col {} of a {rows}-row prefill call at row {first} differs \
                     from the {m_full}-row call",
                    first + i / n,
                    i % n
                );
            }
        }
        println!(
            "{name}: prefill rows bitwise equal at 1, 7, 128 (at 0 and 200) and {m_full} rows"
        );
    }
}

/// Microseconds per call of `op`: two warm-up calls, then the median of `rounds` rounds of
/// `iters` calls each, timed on the host around stream synchronizations.
fn time_us(p: &Pair, rounds: usize, iters: u32, mut op: impl FnMut()) -> f64 {
    op();
    op();
    let mut per_round = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        p.hip_mem.synchronize().expect("synchronize");
        let start = std::time::Instant::now();
        for _ in 0..iters {
            op();
        }
        p.hip_mem.synchronize().expect("synchronize");
        per_round.push(start.elapsed().as_secs_f64() * 1e6 / f64::from(iters));
    }
    per_round.sort_by(f64::total_cmp);
    per_round[rounds / 2]
}

/// Phase 5p S-16: a prefill-step MXFP4 `qgemm` (W4A16, BF16 out) at m = 2,048 takes at most
/// 1.45 × the BF16 GEMM a BF16 checkpoint runs for the same layer in a prefill step (`gemm`
/// with `prefill`: hipBLASLt through the tuned table's invariant rows where the shape has some,
/// the heuristic otherwise), both measured in this run, on every Llama-3.2-3B and Llama-3.1-8B
/// linear shape; m = 512 is printed for information. The bound is a no-regression bound (user
/// decision 2026-10-06 A: the Prefill tile measured 1.0–1.44 ×, the 6a tile up to 1.60 ×; the
/// original 1.10 × target was re-bound). Breaks if prefill calls lose the
/// shared-decode tile (e.g. run a decode tile or the per-wave-decode LDS tile).
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn qgemm_mxfp4_prefill_within_hipblaslt_bound() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(83);
    const BOUND: f64 = 1.45;
    const GATED_M: usize = 2048;
    let max_k = 14336;
    let x = rng.normal(GATED_M * max_k, 1.0);
    let a = tensor(
        &p.hip_mem,
        &[GATED_M, max_k],
        DType::BF16,
        &encode(DType::BF16, &x),
    );
    let gemm = p.hip.gemm().expect("hip gemm");
    let qgemm = p.hip.qgemm().expect("hip qgemm (ABI v2.9)");
    let mut misses = Vec::new();
    for (name, n, k) in SHAPES_3B.into_iter().chain(SHAPES_8B) {
        let cfg = QGemmConfig {
            n: n as u32,
            k: k as u32,
            scheme: QuantSchemeDesc::Mxfp4,
            act_quant: ActQuantDesc::None,
            a_dtype: DType::BF16,
            c_dtype: DType::BF16,
        };
        let (codes, exps) = mxfp4_weights(&mut rng, n, k);
        let b = tensor(&p.hip_mem, &[n, k / 2], DType::U8, &codes);
        let bs = tensor(&p.hip_mem, &[exps.len()], DType::U8, &exps);
        let w = tensor(
            &p.hip_mem,
            &[n, k],
            DType::BF16,
            &encode(DType::BF16, &rng.normal(n * k, 1.0 / (k as f32).sqrt())),
        );
        let gemm_cfg = GemmConfig {
            n: n as u64,
            k: k as u64,
            trans_b: true,
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype: DType::BF16,
            gemv: false,
        };
        let c = tensor(
            &p.hip_mem,
            &[GATED_M, n],
            DType::BF16,
            &vec![0u8; GATED_M * n * 2],
        );
        for m in [512, GATED_M] {
            let a_m = TensorView {
                slice: a.storage.whole().sub(0, ((m - 1) * max_k + k) * 2),
                shape: (&[m, k][..]).into(),
                strides: (&[max_k, 1][..]).into(),
                dtype: DType::BF16,
            };
            let c_m = row_range(&c, 0, m);
            let bf16_us = time_us(&p, 5, 10, || {
                gemm.execute(&mut GemmContext {
                    a: a_m.clone(),
                    b: w.view(),
                    c: c_m.clone(),
                    trans_b: true,
                    alpha: 1.0,
                    beta: 0.0,
                    prefill: true,
                })
                .expect("gemm");
            });
            let mx_us = time_us(&p, 5, 10, || {
                qgemm
                    .execute(&mut QGemmContext {
                        cfg,
                        a: a_m.clone(),
                        a_scales: None,
                        b: b.view(),
                        b_scales: bs.view(),
                        b_zeros: None,
                        c: c_m.clone(),
                        alpha: 1.0,
                        prefill: true,
                        act_static_scale: 1.0,
                    })
                    .expect("qgemm");
            });
            let ratio = mx_us / bf16_us;
            let gated = m == GATED_M;
            println!(
                "mxfp4_prefill_timing {name:<10} m={m:<5} bf16 {bf16_us:>8.1} us ({}) mxfp4 \
                 {mx_us:>8.1} us ({}) ratio {ratio:.3} {}",
                gemm.implementation(&gemm_cfg),
                qgemm.implementation(&cfg),
                if gated { "(gated)" } else { "(info)" }
            );
            if gated && ratio > BOUND {
                misses.push(format!("{name}: {ratio:.3}"));
            }
        }
    }
    assert!(
        misses.is_empty(),
        "MXFP4 prefill above {BOUND} x the BF16 prefill GEMM: {misses:?}"
    );
}
