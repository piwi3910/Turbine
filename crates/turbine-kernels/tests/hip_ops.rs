//! Lab only (novanas R9700, P1 AC S-7/S-13, P2 S-5/S-16): every op the Llama and OLMoE
//! executors use, run through `libturbine_hip.so` via the Rust shim bindings, matches the
//! `cpu-reference` provider on seeded random inputs at the Llama-3.2-3B and OLMoE-1B-7B shapes.
//! Run by `scripts/lab-test.sh novanas`, which sets `TURBINE_TEST_BACKEND=hip`,
//! `TURBINE_KERNEL_LIBRARY` and `TURBINE_AMD_SMI_LIBRARY`.
//!
//! Tolerance: BF16 outputs |Δ| ≤ 1e-2, or one BF16 ulp of the reference where its magnitude
//! exceeds 2 (a rounding flip after a different f32 summation order); F32 outputs |Δ| ≤ 1e-4;
//! copies (the paged K/V append, `copy_blocks`) and `moe_route` selections are exact.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use half::bf16;
use turbine_core::config::DevicesConfig;
use turbine_core::types::{BlockId, DType, DeviceId, ExecutionBackend, Vendor};
use turbine_device::{DiscoveryOptions, discover};
use turbine_kernels::test_support::require_backend;
use turbine_kernels::{
    ActivationConfig, ActivationContext, AddRmsnormConfig, AddRmsnormContext, AttentionConfig,
    AttentionContext, AttentionKind, ElementwiseConfig, ElementwiseContext, EmbeddingConfig,
    EmbeddingContext, GemmConfig, GemmContext, KernelProvider, KvCopyConfig, KvCopyContext,
    MoeExpertsConfig, MoeExpertsContext, MoeRouteConfig, MoeRouteContext, NormConfig, NormContext,
    PagedAttentionContext, RopeConfig, RopeContext, ShimLibrary, cpu_reference_provider,
    shim_provider,
};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

// Llama-3.2-3B shapes.
const HIDDEN: usize = 3072;
const INTERMEDIATE: usize = 8192;
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB: usize = 128256;
const ROPE_THETA: f64 = 500_000.0;

// OLMoE-1B-7B shapes.
const MOE_HIDDEN: usize = 2048;
const MOE_INTER: usize = 1024;
const MOE_EXPERTS: usize = 64;
const MOE_TOP_K: usize = 8;
/// Routed rows up to which the HIP `moe_experts` takes the small-m path (P2c S-11).
const MOE_SMALL_M_MAX_ROWS: usize = 512;

/// The HIP provider and the CPU reference, each with the memory its tensors live in.
struct Pair {
    hip: Arc<dyn KernelProvider>,
    cpu: Arc<dyn KernelProvider>,
    hip_mem: Arc<dyn DeviceMemory>,
    cpu_mem: Arc<dyn DeviceMemory>,
}

/// Serializes the tests of this binary: device discovery (amd-smi) runs once at a time per
/// process, the vocabulary GEMM reference needs ~3 GB of host memory, and each context is driven
/// by one thread (contract §9.4).
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
    let lib = ShimLibrary::load(&path, ExecutionBackend::Hip).expect("load libturbine_hip.so");
    println!(
        "loaded {} abi={} backend={} archs={}",
        path.display(),
        lib.abi_version(),
        lib.backend_name(),
        lib.build_archs().join(",")
    );
    let ctx = lib.create_context(device).expect("HIP context");
    let hip_mem: Arc<dyn DeviceMemory> = ctx.clone();
    let cpu_mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(u32::MAX), 16 << 30);
    Pair {
        hip: shim_provider(ctx),
        cpu: cpu_reference_provider(),
        hip_mem,
        cpu_mem,
    }
}

/// splitmix64: a seeded, dependency-free generator (the inputs only need to be reproducible).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// `n` normal samples with standard deviation `scale` (Box-Muller).
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

/// The same data in HIP memory and in host memory.
fn twin(p: &Pair, shape: &[usize], dtype: DType, data: &[f32]) -> (Tensor, Tensor) {
    let raw = encode(dtype, data);
    let mut hip = Tensor::empty(&p.hip_mem, shape, dtype).expect("HIP alloc");
    let mut cpu = Tensor::empty(&p.cpu_mem, shape, dtype).expect("host alloc");
    hip.storage.copy_from_host(0, &raw).expect("copy to HIP");
    cpu.storage.copy_from_host(0, &raw).expect("copy to host");
    (hip, cpu)
}

/// Contents of `t`; for HIP memory the read synchronizes the compute stream first.
fn read(t: &Tensor) -> Vec<f32> {
    decode(t.dtype, &t.view().slice.read_bytes().expect("read back"))
}

/// |Δ| ≤ 1e-2, or one BF16 ulp of `want` where |want| > 2.
fn bf16_close(got: f32, want: f32) -> bool {
    let d = (got - want).abs();
    if d <= 1e-2 {
        return true;
    }
    if want.abs() > 2.0 {
        let b = bf16::from_f32(want.abs());
        let ulp = bf16::from_bits(b.to_bits() + 1).to_f32() - b.to_f32();
        return d <= ulp;
    }
    false
}

fn assert_close(what: &str, impl_name: &str, got: &[f32], want: &[f32], dtype: DType) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        let ok = match dtype {
            DType::F32 => d <= 1e-4,
            _ => bf16_close(g, w),
        };
        assert!(
            ok,
            "{what} ({impl_name}): element {i}: hip {g} vs cpu {w} (|Δ| {d})"
        );
        if d > worst.1 {
            worst = (i, d);
        }
    }
    println!(
        "{what}: impl={impl_name} max |Δ| {:.3e} at {} ok",
        worst.1, worst.0
    );
}

fn gemm_case(p: &Pair, rng: &mut Rng, m: usize, n: usize, k: usize, c_dtype: DType) {
    let cfg = GemmConfig {
        n: n as u64,
        k: k as u64,
        trans_b: true,
        a_dtype: DType::BF16,
        b_dtype: DType::BF16,
        c_dtype,
    };
    let hip = p.hip.gemm().expect("hip gemm");
    assert!(hip.supports(&cfg), "hip must support gemm {cfg}");
    let impl_name = hip.implementation(&cfg);
    let (a_hip, a_cpu) = twin(p, &[m, k], DType::BF16, &rng.normal(m * k, 1.0));
    let b_scale = 1.0 / (k as f32).sqrt();
    let (b_hip, b_cpu) = twin(p, &[n, k], DType::BF16, &rng.normal(n * k, b_scale));
    let (c_hip, c_cpu) = twin(p, &[m, n], c_dtype, &vec![0.0; m * n]);
    let runs = [
        (hip, &a_hip, &b_hip, &c_hip),
        (p.cpu.gemm().expect("cpu gemm"), &a_cpu, &b_cpu, &c_cpu),
    ];
    for (kernel, a, b, c) in runs {
        let mut ctx = GemmContext {
            a: a.view(),
            b: b.view(),
            c: c.view(),
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
        };
        kernel.execute(&mut ctx).expect("gemm");
    }
    assert_close(
        &format!("gemm m={m} {cfg}"),
        &impl_name,
        &read(&c_hip),
        &read(&c_cpu),
        c_dtype,
    );
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn gemm_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(1);
    for m in [1, 17] {
        for (n, k) in [
            (HIDDEN, HIDDEN),
            (1024, HIDDEN),
            (INTERMEDIATE, HIDDEN),
            (HIDDEN, INTERMEDIATE),
        ] {
            gemm_case(&p, &mut rng, m, n, k, DType::BF16);
        }
        gemm_case(&p, &mut rng, m, VOCAB, HIDDEN, DType::F32);
    }
}

/// One attention call on HIP over the whole query, compared with the CPU reference on row
/// blocks (the full 4096-row reference is too slow for a test).
fn attention_case(p: &Pair, rng: &mut Rng, kind: AttentionKind, q_len: usize, q_start: usize) {
    let cfg = AttentionConfig {
        kind,
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        dtype: DType::BF16,
        block_tokens: None,
        causal: true,
    };
    let hip = p.hip.attention().expect("hip attention");
    assert!(hip.supports(&cfg), "hip must support {} {cfg}", cfg.op());
    let impl_name = hip.implementation(&cfg);
    let kv_len = q_start + q_len;
    let (q_rows, kv_rows) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q_shape = [q_len, Q_HEADS, HEAD_DIM];
    let kv_shape = [kv_len, KV_HEADS, HEAD_DIM];
    let (q_hip, q_cpu) = twin(p, &q_shape, DType::BF16, &rng.normal(q_len * q_rows, 1.0));
    let (k_hip, k_cpu) = twin(
        p,
        &kv_shape,
        DType::BF16,
        &rng.normal(kv_len * kv_rows, 1.0),
    );
    let (v_hip, v_cpu) = twin(
        p,
        &kv_shape,
        DType::BF16,
        &rng.normal(kv_len * kv_rows, 1.0),
    );
    let (o_hip, o_cpu) = twin(p, &q_shape, DType::BF16, &vec![0.0; q_len * q_rows]);

    let mut ctx = AttentionContext {
        cfg,
        q: q_hip.view(),
        k_cache: k_hip.view(),
        v_cache: v_hip.view(),
        out: o_hip.view(),
        q_start: q_start as u32,
        scale,
    };
    hip.execute(&mut ctx).expect("hip attention");
    let hip_out = read(&o_hip);

    let blocks: Vec<(usize, usize)> = if q_len <= 64 {
        vec![(0, q_len)]
    } else {
        vec![(0, 16), (q_len / 2, 16), (q_len - 16, 16)]
    };
    let cpu = p.cpu.attention().expect("cpu attention");
    for (start, count) in blocks {
        let mut ctx = AttentionContext {
            cfg,
            q: q_cpu.view().rows(start, count),
            k_cache: k_cpu.view(),
            v_cache: v_cpu.view(),
            out: o_cpu.view().rows(start, count),
            q_start: (q_start + start) as u32,
            scale,
        };
        cpu.execute(&mut ctx).expect("cpu attention");
        let cpu_out = read(&o_cpu);
        let range = start * q_rows..(start + count) * q_rows;
        assert_close(
            &format!(
                "{} q_len={q_len} q_start={q_start} rows {start}..{}",
                cfg.op(),
                start + count
            ),
            &impl_name,
            &hip_out[range.clone()],
            &cpu_out[range],
            DType::BF16,
        );
    }
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn attention_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(2);
    for s in [1usize, 17, 512, 4096] {
        attention_case(&p, &mut rng, AttentionKind::Prefill, s, 0);
        attention_case(&p, &mut rng, AttentionKind::Decode, 1, s - 1);
    }
    // A prefill chunk after cached context.
    attention_case(&p, &mut rng, AttentionKind::Prefill, 17, 100);
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn norm_rope_silu_embedding_add_match_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(3);
    let t = 17usize;

    // RMSNorm
    let cfg = NormConfig {
        dim: HIDDEN as u64,
        dtype: DType::BF16,
    };
    let hip = p.hip.norm().expect("hip norm");
    assert!(hip.supports(&cfg), "hip must support rmsnorm {cfg}");
    let impl_name = hip.implementation(&cfg);
    let (x_hip, x_cpu) = twin(&p, &[t, HIDDEN], DType::BF16, &rng.normal(t * HIDDEN, 1.0));
    let (w_hip, w_cpu) = twin(&p, &[HIDDEN], DType::BF16, &rng.normal(HIDDEN, 1.0));
    let (o_hip, o_cpu) = twin(&p, &[t, HIDDEN], DType::BF16, &vec![0.0; t * HIDDEN]);
    let runs = [
        (hip, &x_hip, &w_hip, &o_hip),
        (p.cpu.norm().expect("cpu norm"), &x_cpu, &w_cpu, &o_cpu),
    ];
    for (kernel, x, w, o) in runs {
        let mut ctx = NormContext {
            x: x.view(),
            weight: w.view(),
            out: o.view(),
            eps: 1e-5,
        };
        kernel.execute(&mut ctx).expect("rmsnorm");
    }
    let what = format!("rmsnorm rows={t} {cfg}");
    assert_close(&what, &impl_name, &read(&o_hip), &read(&o_cpu), DType::BF16);

    // RoPE (in place on q and k), Llama-3 theta, positions spread over [0, 4096).
    let cfg = RopeConfig {
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        rotary_dim: HEAD_DIM as u32,
        dtype: DType::BF16,
    };
    let hip = p.hip.rope().expect("hip rope");
    assert!(hip.supports(&cfg), "hip must support rope {cfg}");
    let impl_name = hip.implementation(&cfg);
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / ROPE_THETA.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let positions: Vec<f32> = (0..t).map(|i| ((i * 257) % 4096) as f32).collect();
    let q_n = t * Q_HEADS * HEAD_DIM;
    let k_n = t * KV_HEADS * HEAD_DIM;
    let (q_hip, q_cpu) = twin(
        &p,
        &[t, Q_HEADS, HEAD_DIM],
        DType::BF16,
        &rng.normal(q_n, 1.0),
    );
    let (k_hip, k_cpu) = twin(
        &p,
        &[t, KV_HEADS, HEAD_DIM],
        DType::BF16,
        &rng.normal(k_n, 1.0),
    );
    let (pos_hip, pos_cpu) = twin(&p, &[t], DType::I32, &positions);
    let (f_hip, f_cpu) = twin(&p, &[half], DType::F32, &inv_freq);
    let runs = [
        (hip, &q_hip, &k_hip, &pos_hip, &f_hip),
        (
            p.cpu.rope().expect("cpu rope"),
            &q_cpu,
            &k_cpu,
            &pos_cpu,
            &f_cpu,
        ),
    ];
    for (kernel, q, k, pos, f) in runs {
        let mut ctx = RopeContext {
            cfg,
            q: q.view(),
            k: k.view(),
            positions: pos.view(),
            inv_freq: f.view(),
        };
        kernel.execute(&mut ctx).expect("rope");
    }
    let what = format!("rope q tokens={t} {cfg}");
    assert_close(&what, &impl_name, &read(&q_hip), &read(&q_cpu), DType::BF16);
    let what = format!("rope k tokens={t} {cfg}");
    assert_close(&what, &impl_name, &read(&k_hip), &read(&k_cpu), DType::BF16);

    // SiLU · up
    let cfg = ActivationConfig {
        cols: INTERMEDIATE as u64,
        dtype: DType::BF16,
    };
    let hip = p.hip.activation().expect("hip silu_mul");
    assert!(hip.supports(&cfg), "hip must support silu_mul {cfg}");
    let impl_name = hip.implementation(&cfg);
    let n = t * INTERMEDIATE;
    let shape = [t, INTERMEDIATE];
    let (g_hip, g_cpu) = twin(&p, &shape, DType::BF16, &rng.normal(n, 1.0));
    let (u_hip, u_cpu) = twin(&p, &shape, DType::BF16, &rng.normal(n, 1.0));
    let (o_hip, o_cpu) = twin(&p, &shape, DType::BF16, &vec![0.0; n]);
    let runs = [
        (hip, &g_hip, &u_hip, &o_hip),
        (
            p.cpu.activation().expect("cpu silu_mul"),
            &g_cpu,
            &u_cpu,
            &o_cpu,
        ),
    ];
    for (kernel, g, u, o) in runs {
        let mut ctx = ActivationContext {
            gate: g.view(),
            up: u.view(),
            out: o.view(),
        };
        kernel.execute(&mut ctx).expect("silu_mul");
    }
    let what = format!("silu_mul rows={t} {cfg}");
    assert_close(&what, &impl_name, &read(&o_hip), &read(&o_cpu), DType::BF16);

    // Embedding over the full vocabulary, ids spread across it (last row included).
    let cfg = EmbeddingConfig {
        hidden: HIDDEN as u64,
        vocab_rows: VOCAB as u64,
        dtype: DType::BF16,
    };
    let hip = p.hip.embedding().expect("hip embedding");
    assert!(hip.supports(&cfg), "hip must support embedding {cfg}");
    let impl_name = hip.implementation(&cfg);
    let mut ids: Vec<f32> = (0..t - 1).map(|i| ((i * 7919) % VOCAB) as f32).collect();
    ids.push((VOCAB - 1) as f32);
    let table = rng.normal(VOCAB * HIDDEN, 1.0);
    let (tab_hip, tab_cpu) = twin(&p, &[VOCAB, HIDDEN], DType::BF16, &table);
    let (id_hip, id_cpu) = twin(&p, &[t], DType::I32, &ids);
    let (o_hip, o_cpu) = twin(&p, &[t, HIDDEN], DType::BF16, &vec![0.0; t * HIDDEN]);
    let runs = [
        (hip, &id_hip, &tab_hip, &o_hip),
        (
            p.cpu.embedding().expect("cpu embedding"),
            &id_cpu,
            &tab_cpu,
            &o_cpu,
        ),
    ];
    for (kernel, ids, table, o) in runs {
        let mut ctx = EmbeddingContext {
            ids: ids.view(),
            table: table.view(),
            out: o.view(),
            vocab_offset: 0,
        };
        kernel.execute(&mut ctx).expect("embedding");
    }
    let what = format!("embedding tokens={t} {cfg}");
    assert_close(&what, &impl_name, &read(&o_hip), &read(&o_cpu), DType::BF16);

    // Residual add
    let cfg = ElementwiseConfig { dtype: DType::BF16 };
    let hip = p.hip.elementwise().expect("hip add");
    assert!(hip.supports(&cfg), "hip must support add {cfg}");
    let impl_name = hip.implementation(&cfg);
    let n = t * HIDDEN;
    let (a_hip, a_cpu) = twin(&p, &[n], DType::BF16, &rng.normal(n, 1.0));
    let (b_hip, b_cpu) = twin(&p, &[n], DType::BF16, &rng.normal(n, 1.0));
    let (o_hip, o_cpu) = twin(&p, &[n], DType::BF16, &vec![0.0; n]);
    let runs = [
        (hip, &a_hip, &b_hip, &o_hip),
        (
            p.cpu.elementwise().expect("cpu add"),
            &a_cpu,
            &b_cpu,
            &o_cpu,
        ),
    ];
    for (kernel, a, b, o) in runs {
        let mut ctx = ElementwiseContext {
            a: a.view(),
            b: b.view(),
            out: o.view(),
        };
        kernel.execute(&mut ctx).expect("add");
    }
    let what = format!("add n={n} {cfg}");
    assert_close(&what, &impl_name, &read(&o_hip), &read(&o_cpu), DType::BF16);
}

/// The first `cols` columns of the `[rows, stride]` tensor `t`, as a row-strided `[rows, cols]`
/// view.
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

/// Phase 2c S-9: the fused residual add + RMSNorm of kernel ABI v2.1 (`turbine_add_rmsnorm`)
/// matches the cpu-reference `add` then `rmsnorm`: the updated residual exactly (the same
/// round-to-nearest-even BF16 sum); the normalised output within the RMSNorm tolerance, except
/// that on 2048-row cases (4–6 M elements) a few elements may sit one BF16 rounding step of the
/// intermediate `x·inv_rms` away (a different f32 reduction order crossing a BF16 midpoint,
/// then multiplied by γ): those must be exactly such a step, and at most 1 in 1,000. At
/// Llama's 3072 the fused op is also bitwise HIP `add` then HIP `rmsnorm` (the same ck_tile
/// pipeline). Rows 1/16/2048 at OLMoE's 2048 and Llama's 3072 (the two ck_tile buckets), plus
/// the Turbine kernel for a row stride CK's 8-wide loads cannot take and for a dimension outside
/// the buckets.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn add_rmsnorm_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let hip = p
        .hip
        .add_rmsnorm()
        .expect("libturbine_hip.so exports the ABI v2.1 add_rmsnorm trio");
    let cpu = p.cpu.add_rmsnorm().expect("cpu add_rmsnorm");
    let mut rng = Rng(21);
    // (rows, dim, row stride of x).
    let cases = [
        (1, MOE_HIDDEN, MOE_HIDDEN),
        (16, MOE_HIDDEN, MOE_HIDDEN),
        (2048, MOE_HIDDEN, MOE_HIDDEN),
        (1, HIDDEN, HIDDEN),
        (16, HIDDEN, HIDDEN),
        (2048, HIDDEN, HIDDEN),
        (16, HIDDEN, HIDDEN + 4),
        (7, 64, 64),
    ];
    for (rows, dim, x_stride) in cases {
        let cfg = AddRmsnormConfig {
            dtype: DType::BF16,
            dim: dim as u32,
        };
        assert!(hip.supports(&cfg), "hip must support add_rmsnorm {cfg}");
        let impl_name = hip.implementation(&cfg);
        let n = rows * dim;
        let before = rng.normal(n, 1.0);
        let (r_hip, r_cpu) = twin(&p, &[rows, dim], DType::BF16, &before);
        let x = rng.normal(rows * x_stride, 1.0);
        let (x_hip, x_cpu) = twin(&p, &[rows, x_stride], DType::BF16, &x);
        let (w_hip, w_cpu) = twin(&p, &[dim], DType::BF16, &rng.normal(dim, 1.0));
        let (o_hip, o_cpu) = twin(&p, &[rows, dim], DType::BF16, &vec![0.0; n]);
        let runs = [
            (hip, &r_hip, &x_hip, &w_hip, &o_hip),
            (cpu, &r_cpu, &x_cpu, &w_cpu, &o_cpu),
        ];
        for (kernel, r, x, w, o) in runs {
            let mut ctx = AddRmsnormContext {
                residual: r.view(),
                x: column_prefix(x, dim),
                weight: w.view(),
                out: o.view(),
                eps: 1e-5,
            };
            kernel.execute(&mut ctx).expect("add_rmsnorm");
        }
        let what = format!("add_rmsnorm rows={rows} x_stride={x_stride} {cfg}");
        assert_exact(
            &format!("{what} residual"),
            &impl_name,
            &read(&r_hip),
            &read(&r_cpu),
        );
        let residual = read(&r_cpu);
        let weight = read(&w_cpu);
        let (out_hip, out_cpu) = (read(&o_hip), read(&o_cpu));
        let mut flips = 0usize;
        for (i, (&g, &w)) in out_hip.iter().zip(&out_cpu).enumerate() {
            if bf16_close(g, w) {
                continue;
            }
            let row = &residual[i / dim * dim..(i / dim + 1) * dim];
            let normed = rms_normalised(row, 1e-5, i % dim);
            let allowed = one_rounding_step(normed, weight[i % dim]);
            assert!(
                allowed.contains(&g),
                "{what} out ({impl_name}): element {i}: hip {g} vs cpu {w}, not within one BF16 \
                 rounding step of x·inv_rms = {normed} (allowed {allowed:?})"
            );
            flips += 1;
        }
        // A rounding flip of x·inv_rms needs x·inv_rms within f32 noise of a BF16 midpoint: rare.
        assert!(
            flips * 1000 <= n,
            "{what} out ({impl_name}): {flips} of {n} elements need a rounding flip"
        );
        println!(
            "{what} out: impl={impl_name} {flips} of {n} elements one intermediate rounding \
             step from cpu, the rest within the BF16 tolerance ok"
        );

        // At Llama's 3072, rmsnorm runs the same ck_tile pipeline without the fused add: the
        // fused op is bitwise HIP add followed by HIP rmsnorm.
        if dim == HIDDEN && x_stride == dim {
            let (r2, _) = twin(&p, &[rows, dim], DType::BF16, &before);
            let (o2, _) = twin(&p, &[rows, dim], DType::BF16, &vec![0.0; n]);
            let add = p.hip.elementwise().expect("hip add");
            add.execute(&mut ElementwiseContext {
                a: r2.view(),
                b: x_hip.view(),
                out: r2.view(),
            })
            .expect("hip add");
            let norm = p.hip.norm().expect("hip rmsnorm");
            norm.execute(&mut NormContext {
                x: r2.view(),
                weight: w_hip.view(),
                out: o2.view(),
                eps: 1e-5,
            })
            .expect("hip rmsnorm");
            assert_exact(
                &format!("{what} vs hip add + rmsnorm"),
                &impl_name,
                &out_hip,
                &read(&o2),
            );
        }
    }
}

/// `x[j] · inv_rms(x)` computed exactly enough (f64) to bracket any f32 implementation's value.
fn rms_normalised(x: &[f32], eps: f64, j: usize) -> f64 {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    f64::from(x[j]) / (ss / x.len() as f64 + eps).sqrt()
}

/// The BF16 outputs `round(round(t) · γ)` an RMSNorm may give when its f32 `t = x · inv_rms`
/// rounds to the BF16 value nearest `normed` or to either neighbour of it (a different f32
/// summation order moves `t` by a few f32 ulps, which can cross a BF16 midpoint); the product
/// with `γ` is f32 and rounded to nearest even, as every provider does.
fn one_rounding_step(normed: f64, gamma: f32) -> Vec<f32> {
    let nearest = bf16::from_f64(normed).to_bits();
    [nearest.wrapping_sub(1), nearest, nearest.wrapping_add(1)]
        .into_iter()
        .map(|bits| bf16::from_f32(bf16::from_bits(bits).to_f32() * gamma).to_f32())
        .collect()
}

/// `0..n` in a seeded random order (Fisher–Yates).
fn shuffled(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// Bit-for-bit equality (copies and integer outputs).
fn assert_exact(what: &str, impl_name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what} ({impl_name}): element {i}: hip {g} vs cpu {w}"
        );
    }
    println!("{what}: impl={impl_name} exact ok");
}

/// One ragged batch through `attention_{prefill,decode}_paged` on both providers with
/// `heads = (q_heads, kv_heads)` of `HEAD_DIM`: sequence `s` has `q_lens[s]` new tokens and
/// `kv_lens[s]` tokens after the append, its pages spread over a shuffled block table of a pool
/// with two blocks nobody owns. Compares the outputs and the pool after the append; returns the
/// HIP implementation that ran.
fn paged_case(
    p: &Pair,
    rng: &mut Rng,
    kind: AttentionKind,
    heads: (usize, usize),
    block_tokens: usize,
    q_lens: &[usize],
    kv_lens: &[usize],
) -> String {
    let (q_heads, kv_heads) = heads;
    let cfg = AttentionConfig {
        kind,
        num_q_heads: q_heads as u32,
        num_kv_heads: kv_heads as u32,
        head_dim: HEAD_DIM as u32,
        dtype: DType::BF16,
        block_tokens: Some(block_tokens as u32),
        causal: true,
    };
    let hip = p.hip.attention().expect("hip attention");
    assert!(hip.supports(&cfg), "hip must support {} {cfg}", cfg.op());
    let impl_name = hip.implementation(&cfg);

    let seqs = q_lens.len();
    let blocks_of: Vec<usize> = kv_lens
        .iter()
        .map(|&kv| kv.div_ceil(block_tokens))
        .collect();
    let max_blocks = blocks_of.iter().copied().max().expect("a sequence");
    let num_blocks = blocks_of.iter().sum::<usize>() + 2;
    let order = shuffled(rng, num_blocks);
    let mut table = vec![-1f32; seqs * max_blocks];
    let mut next = 0;
    for (s, &n) in blocks_of.iter().enumerate() {
        for b in 0..n {
            table[s * max_blocks + b] = order[next] as f32;
            next += 1;
        }
    }
    let mut indptr = vec![0f32];
    for &q in q_lens {
        indptr.push(indptr.last().copied().unwrap_or(0.0) + q as f32);
    }
    let kv: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
    let total_q: usize = q_lens.iter().sum();
    let (q_rows, kv_rows) = (q_heads * HEAD_DIM, kv_heads * HEAD_DIM);
    let pool_shape = [num_blocks, 2, block_tokens, kv_heads, HEAD_DIM];
    let pool_len = num_blocks * 2 * block_tokens * kv_rows;

    // The pool starts with random history in every slot, including the unowned blocks.
    let (pool_hip, pool_cpu) = twin(p, &pool_shape, DType::BF16, &rng.normal(pool_len, 1.0));
    let q_shape = [total_q, q_heads, HEAD_DIM];
    let new_shape = [total_q, kv_heads, HEAD_DIM];
    let (q_hip, q_cpu) = twin(p, &q_shape, DType::BF16, &rng.normal(total_q * q_rows, 1.0));
    let (k_hip, k_cpu) = twin(
        p,
        &new_shape,
        DType::BF16,
        &rng.normal(total_q * kv_rows, 1.0),
    );
    let (v_hip, v_cpu) = twin(
        p,
        &new_shape,
        DType::BF16,
        &rng.normal(total_q * kv_rows, 1.0),
    );
    let (o_hip, o_cpu) = twin(p, &q_shape, DType::BF16, &vec![0.0; total_q * q_rows]);
    let (bt_hip, bt_cpu) = twin(p, &[seqs, max_blocks], DType::I32, &table);
    let (ip_hip, ip_cpu) = twin(p, &[seqs + 1], DType::I32, &indptr);
    let (kv_hip, kv_cpu) = twin(p, &[seqs], DType::I32, &kv);

    let runs = [
        (
            hip,
            [&q_hip, &k_hip, &v_hip, &o_hip, &pool_hip],
            [&bt_hip, &ip_hip, &kv_hip],
        ),
        (
            p.cpu.attention().expect("cpu attention"),
            [&q_cpu, &k_cpu, &v_cpu, &o_cpu, &pool_cpu],
            [&bt_cpu, &ip_cpu, &kv_cpu],
        ),
    ];
    for (kernel, [q, k, v, o, pool], [bt, ip, kv]) in runs {
        let mut ctx = PagedAttentionContext {
            cfg,
            q: q.view(),
            k_new: k.view(),
            v_new: v.view(),
            out: o.view(),
            kv_layer: pool.view(),
            block_table: bt.view(),
            q_indptr: ip.view(),
            kv_lens: kv.view(),
            max_q_len: q_lens.iter().copied().max().unwrap_or(0) as u32,
            max_kv_len: kv_lens.iter().copied().max().unwrap_or(0) as u32,
            max_blocks_per_seq: max_blocks as u32,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };
        kernel.execute_paged(&mut ctx).expect("paged attention");
    }
    let what = format!(
        "{} heads={q_heads}/{kv_heads} block_tokens={block_tokens} q_lens={q_lens:?} \
         kv_lens={kv_lens:?}",
        cfg.op()
    );
    assert_close(
        &format!("{what} out"),
        &impl_name,
        &read(&o_hip),
        &read(&o_cpu),
        DType::BF16,
    );
    assert_exact(
        &format!("{what} pool after append"),
        &impl_name,
        &read(&pool_hip),
        &read(&pool_cpu),
    );
    impl_name
}

/// The CPU outputs of one `moe_route` call.
struct Routing {
    weights: Vec<f32>,
    sorted_rows: Vec<f32>,
    expert_offsets: Vec<f32>,
}

/// `moe_route` on both providers; the selections, grouping and offsets must be identical.
fn route_case(p: &Pair, cfg: MoeRouteConfig, tokens: usize, logits: &[f32]) -> Routing {
    let hip = p.hip.moe().expect("hip moe");
    assert!(hip.supports_route(&cfg), "hip must support moe_route {cfg}");
    let impl_name = hip.implementation_route(&cfg);
    let (e, k) = (cfg.num_experts as usize, cfg.top_k as usize);
    let (l_hip, l_cpu) = twin(p, &[tokens, e], DType::F32, logits);
    let (id_hip, id_cpu) = twin(p, &[tokens, k], DType::I32, &vec![0.0; tokens * k]);
    let (w_hip, w_cpu) = twin(p, &[tokens, k], DType::F32, &vec![0.0; tokens * k]);
    let (s_hip, s_cpu) = twin(p, &[tokens * k], DType::I32, &vec![0.0; tokens * k]);
    let (off_hip, off_cpu) = twin(p, &[e + 1], DType::I32, &vec![0.0; e + 1]);
    let runs = [
        (hip, [&l_hip, &id_hip, &w_hip, &s_hip, &off_hip]),
        (
            p.cpu.moe().expect("cpu moe"),
            [&l_cpu, &id_cpu, &w_cpu, &s_cpu, &off_cpu],
        ),
    ];
    for (kernel, [l, id, w, s, off]) in runs {
        let mut ctx = MoeRouteContext {
            cfg,
            router_logits: l.view(),
            topk_ids: id.view(),
            topk_weights: w.view(),
            sorted_rows: s.view(),
            expert_offsets: off.view(),
        };
        kernel.route(&mut ctx).expect("moe_route");
    }
    let what = format!("moe_route tokens={tokens} {cfg}");
    let routing = Routing {
        weights: read(&w_cpu),
        sorted_rows: read(&s_cpu),
        expert_offsets: read(&off_cpu),
    };
    assert_exact(
        &format!("{what} topk_ids"),
        &impl_name,
        &read(&id_hip),
        &read(&id_cpu),
    );
    assert_exact(
        &format!("{what} sorted_rows"),
        &impl_name,
        &read(&s_hip),
        &routing.sorted_rows,
    );
    assert_exact(
        &format!("{what} expert_offsets"),
        &impl_name,
        &read(&off_hip),
        &routing.expert_offsets,
    );
    assert_close(
        &format!("{what} topk_weights"),
        &impl_name,
        &read(&w_hip),
        &routing.weights,
        DType::F32,
    );
    routing
}

/// The OLMoE expert weights in both memories: gate/up `[E, inter, hidden]`, down
/// `[E, hidden, inter]`.
struct ExpertWeights {
    hip: [Tensor; 3],
    cpu: [Tensor; 3],
}

fn expert_weights(p: &Pair, rng: &mut Rng) -> ExpertWeights {
    let n = MOE_EXPERTS * MOE_INTER * MOE_HIDDEN;
    let up_scale = 1.0 / (MOE_HIDDEN as f32).sqrt();
    let (g_hip, g_cpu) = twin(
        p,
        &[MOE_EXPERTS, MOE_INTER, MOE_HIDDEN],
        DType::BF16,
        &rng.normal(n, up_scale),
    );
    let (u_hip, u_cpu) = twin(
        p,
        &[MOE_EXPERTS, MOE_INTER, MOE_HIDDEN],
        DType::BF16,
        &rng.normal(n, up_scale),
    );
    let (d_hip, d_cpu) = twin(
        p,
        &[MOE_EXPERTS, MOE_HIDDEN, MOE_INTER],
        DType::BF16,
        &rng.normal(n, 1.0 / (MOE_INTER as f32).sqrt()),
    );
    ExpertWeights {
        hip: [g_hip, u_hip, d_hip],
        cpu: [g_cpu, u_cpu, d_cpu],
    }
}

/// One `moe_experts` batch: `tokens` tokens routed by `logits` (on both providers, which must
/// agree) over the local experts `local`; the HIP run uses a caller workspace when `workspace`,
/// and gets the host copy of the offsets only when its `needs_host_offsets` asks for it. With
/// `repeat`, the HIP call runs a second time on the same inputs, which must give the same bits,
/// and then `repeat` more times to print its mean time per call.
struct ExpertsCase<'a> {
    tokens: usize,
    logits: &'a [f32],
    local: (usize, usize),
    workspace: bool,
    repeat: usize,
}

fn experts_case(p: &Pair, rng: &mut Rng, w: &ExpertWeights, case: &ExpertsCase<'_>) {
    let (tokens, (begin, end)) = (case.tokens, case.local);
    let route_cfg = MoeRouteConfig {
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        renormalize: false,
    };
    let routing = route_case(p, route_cfg, tokens, case.logits);
    let host_offsets: Vec<i32> = routing.expert_offsets.iter().map(|&o| o as i32).collect();
    let empty = (begin..end)
        .filter(|&e| host_offsets[e] == host_offsets[e + 1])
        .count();

    let cfg = MoeExpertsConfig {
        hidden: MOE_HIDDEN as u32,
        inter: MOE_INTER as u32,
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        expert_begin: begin as u32,
        expert_end: end as u32,
        dtype: DType::BF16,
    };
    let hip = p.hip.moe().expect("hip moe");
    assert!(
        hip.supports_experts(&cfg),
        "hip must support moe_experts {cfg}"
    );
    let rows = cfg.routed_rows(tokens);
    // P2c S-11: the HIP library reads the offsets on the device for up to 512 routed rows.
    let needs_host = hip.needs_host_offsets(&cfg, rows);
    assert_eq!(
        needs_host,
        rows > MOE_SMALL_M_MAX_ROWS,
        "needs_host_offsets for {rows} routed rows"
    );
    let impl_name = format!(
        "{} host_offsets={needs_host}",
        hip.implementation_experts(&cfg)
    );
    let hip_offsets: &[i32] = if needs_host { &host_offsets } else { &[] };
    let (x_hip, x_cpu) = twin(
        p,
        &[tokens, MOE_HIDDEN],
        DType::BF16,
        &rng.normal(tokens * MOE_HIDDEN, 1.0),
    );
    let out0 = rng.normal(tokens * MOE_HIDDEN, 0.1);
    let (o_hip, o_cpu) = twin(p, &[tokens, MOE_HIDDEN], DType::BF16, &out0);
    let (s_hip, s_cpu) = twin(p, &[rows], DType::I32, &routing.sorted_rows);
    let (off_hip, off_cpu) = twin(p, &[MOE_EXPERTS + 1], DType::I32, &routing.expert_offsets);
    let (tw_hip, tw_cpu) = twin(p, &[tokens, MOE_TOP_K], DType::F32, &routing.weights);
    // A generous caller workspace: row positions, gathered rows and the three intermediates.
    let ws = case.workspace.then(|| {
        let bytes = rows * 4 + rows * (2 * MOE_HIDDEN + 2 * MOE_INTER) * 2 + 4096;
        Tensor::empty(&p.hip_mem, &[bytes.div_ceil(2)], DType::BF16).expect("workspace")
    });
    let local = end - begin;
    let runs = [
        (
            hip,
            &w.hip,
            [&x_hip, &o_hip, &s_hip, &off_hip, &tw_hip],
            hip_offsets,
            ws.as_ref(),
        ),
        (
            p.cpu.moe().expect("cpu moe"),
            &w.cpu,
            [&x_cpu, &o_cpu, &s_cpu, &off_cpu, &tw_cpu],
            &host_offsets[..],
            None,
        ),
    ];
    for (kernel, [wg, wu, wd], [x, o, s, off, tw], offsets, ws) in runs {
        let mut ctx = MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: wg.view().rows(begin, local),
            w_up: wu.view().rows(begin, local),
            w_down: wd.view().rows(begin, local),
            sorted_rows: s.view(),
            expert_offsets: off.view(),
            topk_weights: tw.view(),
            host_expert_offsets: offsets,
            out: o.view(),
            workspace: ws.map(|t| t.view().slice),
        };
        kernel.experts(&mut ctx).expect("moe_experts");
    }
    let what = format!(
        "moe_experts tokens={tokens} rows={rows} local=[{begin},{end}) empty_local_experts={empty} caller_workspace={} {cfg}",
        case.workspace
    );
    let got = read(&o_hip);
    assert_close(&what, &impl_name, &got, &read(&o_cpu), DType::BF16);
    if case.repeat == 0 {
        return;
    }

    // The same inputs again: bitwise the same output. Then the mean time of one call.
    let mut again = Tensor::empty(&p.hip_mem, &[tokens, MOE_HIDDEN], DType::BF16).expect("out");
    let raw = encode(DType::BF16, &out0);
    let run = |out: &Tensor| {
        hip.experts(&mut MoeExpertsContext {
            cfg,
            x: x_hip.view(),
            w_gate: w.hip[0].view().rows(begin, local),
            w_up: w.hip[1].view().rows(begin, local),
            w_down: w.hip[2].view().rows(begin, local),
            sorted_rows: s_hip.view(),
            expert_offsets: off_hip.view(),
            topk_weights: tw_hip.view(),
            host_expert_offsets: hip_offsets,
            out: out.view(),
            workspace: ws.as_ref().map(|t| t.view().slice),
        })
    };
    again.storage.copy_from_host(0, &raw).expect("copy to HIP");
    run(&again).expect("moe_experts again");
    assert_exact(&format!("{what} rerun"), &impl_name, &read(&again), &got);
    p.hip_mem.synchronize().expect("sync");
    let start = std::time::Instant::now();
    for _ in 0..case.repeat {
        run(&again).expect("moe_experts timing");
    }
    p.hip_mem.synchronize().expect("sync");
    let us = start.elapsed().as_secs_f64() * 1e6 / case.repeat as f64;
    println!(
        "moe_experts_timing: tokens={tokens} rows={rows} impl={impl_name} us_per_call={us:.1}"
    );
}

/// The default 128-token page runs paged attention on CK `fmha_fwd_pagedkv`, a 16-token page on
/// the Turbine kernel, both against the CPU reference: ragged prefill batches of 1-, 17-, 512-
/// and 2,048-token chunks after 0, 100 and 1,000 cached tokens with three decode rows riding
/// along, then a decode batch, at the Llama-3.2-3B (24/8) and OLMoE (16/16) head shapes.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn paged_prefill_ck_128_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(12);

    // (new tokens, cached tokens before them) per sequence.
    let prefill_batches: [&[(usize, usize)]; 2] = [
        &[(2048, 0), (17, 100), (1, 1000), (1, 1), (1, 130), (1, 700)],
        &[(512, 1000), (1, 0), (17, 0), (2048, 100), (1, 255), (1, 1)],
    ];
    let decode_lens = [1, 128, 129, 1001];
    for heads in [(Q_HEADS, KV_HEADS), (16, 16)] {
        for (block_tokens, want) in [(128, "ck_tile_fmha_pagedkv"), (16, "turbine_hip")] {
            for batch in prefill_batches {
                let q_lens: Vec<usize> = batch.iter().map(|&(q, _)| q).collect();
                let kv_lens: Vec<usize> = batch.iter().map(|&(q, c)| q + c).collect();
                let got = paged_case(
                    &p,
                    &mut rng,
                    AttentionKind::PrefillPaged,
                    heads,
                    block_tokens,
                    &q_lens,
                    &kv_lens,
                );
                assert_eq!(
                    got, want,
                    "prefill heads={heads:?} block_tokens={block_tokens}"
                );
            }
            let got = paged_case(
                &p,
                &mut rng,
                AttentionKind::DecodePaged,
                heads,
                block_tokens,
                &[1; 4],
                &decode_lens,
            );
            assert_eq!(
                got, want,
                "decode heads={heads:?} block_tokens={block_tokens}"
            );
        }
    }
}

#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn paged_and_moe_ops() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(4);

    // Paged attention at the Llama-3.2-3B shapes: a 16-token page (Turbine kernel) and the
    // 128-token default page (Composable Kernel pagedkv). A prefill batch mixing a fresh prompt,
    // a single-token row and a chunk after cached context (more than one query tile), then a
    // decode batch.
    for block_tokens in [16, 128] {
        paged_case(
            &p,
            &mut rng,
            AttentionKind::PrefillPaged,
            (Q_HEADS, KV_HEADS),
            block_tokens,
            &[37, 1, 70],
            &[37, 300, 200],
        );
        paged_case(
            &p,
            &mut rng,
            AttentionKind::DecodePaged,
            (Q_HEADS, KV_HEADS),
            block_tokens,
            &[1, 1, 1, 1],
            &[1, 17, 129, 513],
        );
    }

    // copy_blocks over 3 layers of 8 sixteen-token Llama blocks; the third pair reads a block
    // the first pair wrote, so the copy order matters.
    let block_elems = 2 * 16 * KV_HEADS * HEAD_DIM;
    let block_bytes = (block_elems * 2) as u64;
    let (layers, per_layer) = (3usize, 8usize);
    let copy_cfg = KvCopyConfig {
        num_layers: layers as u32,
        block_bytes,
    };
    let hip = p.hip.kv_copy().expect("hip copy_blocks");
    assert!(
        hip.supports(&copy_cfg),
        "hip must support copy_blocks {copy_cfg}"
    );
    let impl_name = hip.implementation(&copy_cfg);
    let (pool_hip, pool_cpu) = twin(
        &p,
        &[layers, per_layer, block_elems],
        DType::BF16,
        &rng.normal(layers * per_layer * block_elems, 1.0),
    );
    let pairs = [
        (BlockId(3), BlockId(7)),
        (BlockId(0), BlockId(5)),
        (BlockId(7), BlockId(1)),
    ];
    let runs = [
        (hip, &pool_hip),
        (p.cpu.kv_copy().expect("cpu copy_blocks"), &pool_cpu),
    ];
    for (kernel, pool) in runs {
        let mut ctx = KvCopyContext {
            pool: pool.view().slice,
            layer_stride_bytes: per_layer as u64 * block_bytes,
            block_bytes,
            num_layers: layers as u32,
            pairs: &pairs,
        };
        kernel.execute(&mut ctx).expect("copy_blocks");
    }
    assert_exact(
        &format!("copy_blocks {copy_cfg} pairs={pairs:?}"),
        &impl_name,
        &read(&pool_hip),
        &read(&pool_cpu),
    );

    // moe_route: OLMoE (64 experts, top-8, no renormalisation) on random logits; exact ties
    // (they go to the lower expert id); a renormalising 8-expert top-2 router.
    let olmoe = MoeRouteConfig {
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        renormalize: false,
    };
    route_case(&p, olmoe, 37, &rng.normal(37 * MOE_EXPERTS, 2.0));
    let mut ties = vec![0f32; 3 * MOE_EXPERTS];
    ties[0] = 1.0;
    ties[1] = 1.0;
    ties[MOE_EXPERTS + 40] = 2.0;
    route_case(&p, olmoe, 3, &ties);
    let small = MoeRouteConfig {
        num_experts: 8,
        top_k: 2,
        renormalize: true,
    };
    route_case(&p, small, 13, &rng.normal(13 * 8, 1.0));

    // moe_experts at the OLMoE shapes.
    let w = expert_weights(&p, &mut rng);
    // Random routing in which expert 63 never wins, so one expert receives no token.
    let mut logits = rng.normal(37 * MOE_EXPERTS, 1.0);
    for row in logits.chunks_exact_mut(MOE_EXPERTS) {
        row[MOE_EXPERTS - 1] = -30.0;
    }
    let case = ExpertsCase {
        tokens: 37,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: false,
        repeat: 0,
    };
    experts_case(&p, &mut rng, &w, &case);
    // Every token selects the same 8 experts (3..=10), through a caller workspace.
    let mut logits = rng.normal(21 * MOE_EXPERTS, 0.1);
    for row in logits.chunks_exact_mut(MOE_EXPERTS) {
        for v in &mut row[3..11] {
            *v += 10.0;
        }
    }
    let case = ExpertsCase {
        tokens: 21,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: true,
        repeat: 0,
    };
    experts_case(&p, &mut rng, &w, &case);
    // A shard of the experts: the local range [16, 48).
    let logits = rng.normal(21 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 21,
        logits: &logits,
        local: (16, 48),
        workspace: false,
        repeat: 0,
    };
    experts_case(&p, &mut rng, &w, &case);
}

/// P2c S-11: `moe_experts` at the OLMoE shapes (64 experts, top-8, hidden 2048, inter 1024) for
/// decode batches of 1, 16 and 64 tokens runs the small-m path without host offsets
/// (`needs_host_offsets` false, `host_expert_offsets` empty) and matches the CPU reference,
/// bitwise identical across two runs: random routing (experts without rows), every token on the
/// same 8 experts (64 rows on one expert), a local shard and a caller workspace. Above 512 routed
/// rows (65 tokens) the library asks for the host offsets and the hipBLASLt path still matches.
/// Prints each case's mean time per call (`moe_experts_timing:`).
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn moe_experts_small_m_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(13);
    let w = expert_weights(&p, &mut rng);
    for tokens in [1, 16, 64] {
        let logits = rng.normal(tokens * MOE_EXPERTS, 1.0);
        experts_case(
            &p,
            &mut rng,
            &w,
            &ExpertsCase {
                tokens,
                logits: &logits,
                local: (0, MOE_EXPERTS),
                workspace: tokens == 16,
                repeat: 20,
            },
        );
    }
    // Every token on experts 3..=10: 64 rows per selected expert, 56 experts without rows.
    let mut logits = rng.normal(64 * MOE_EXPERTS, 0.1);
    for row in logits.chunks_exact_mut(MOE_EXPERTS) {
        for v in &mut row[3..11] {
            *v += 10.0;
        }
    }
    let case = ExpertsCase {
        tokens: 64,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: false,
        repeat: 20,
    };
    experts_case(&p, &mut rng, &w, &case);
    // A shard of the experts, [16, 48).
    let logits = rng.normal(16 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 16,
        logits: &logits,
        local: (16, 48),
        workspace: false,
        repeat: 1,
    };
    experts_case(&p, &mut rng, &w, &case);
    // 65 tokens = 520 routed rows: the host-offset (hipBLASLt) path.
    let logits = rng.normal(65 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 65,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: false,
        repeat: 20,
    };
    experts_case(&p, &mut rng, &w, &case);
}

/// `n` copies of a short seeded random pattern (the timing inputs only need realistic values,
/// and generating the vocabulary-sized tensors element by element is slow).
fn pattern(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    let base = rng.normal(4099, scale);
    (0..n).map(|i| base[i % base.len()]).collect()
}

/// A HIP tensor holding `data`.
fn on_hip(p: &Pair, shape: &[usize], dtype: DType, data: &[f32]) -> Tensor {
    let mut t = Tensor::empty(&p.hip_mem, shape, dtype).expect("HIP alloc");
    t.storage
        .copy_from_host(0, &encode(dtype, data))
        .expect("copy to HIP");
    t
}

/// Mean wall time of `op` in microseconds over `iters` back-to-back calls after two warm-up
/// calls, the stream drained before and after: the device time of the op, or its host launch
/// time when that is longer (as in the executor, which enqueues without synchronizing).
fn time_us(p: &Pair, iters: u32, mut op: impl FnMut()) -> f64 {
    op();
    op();
    p.hip_mem.synchronize().expect("synchronize");
    let start = std::time::Instant::now();
    for _ in 0..iters {
        op();
    }
    p.hip_mem.synchronize().expect("synchronize");
    start.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
}

/// Accumulates per-op timings into a decode-forward estimate.
#[derive(Default)]
struct Timings {
    per_forward_us: f64,
}

impl Timings {
    /// Prints one op's time per call and its share of a forward (`calls` per forward).
    fn report(&mut self, name: &str, impl_name: &str, us: f64, calls: f64) {
        self.per_forward_us += us * calls;
        println!(
            "timing {name:<40} impl={impl_name:<22} {us:>9.1} us/call x{calls:>3} = {:>8.3} ms/forward",
            us * calls / 1e3
        );
    }
}

/// Lab microbenchmark (no assertion on speed): the time of every op of one Llama-3.2-3B decode
/// step for a batch of 16 sequences with ~600 cached tokens each (16-token pages), per call and
/// per forward (28 layers plus the embedding, the final norm and the LM head). Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_ops -- decode_op_timings`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn decode_op_timings() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(5);
    const LAYERS: f64 = 28.0;
    const ITERS: u32 = 50;
    let m = 16usize;
    let (q_rows, kv_rows) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let mut t = Timings::default();

    // GEMMs: every projection of a layer, then the LM head (F32 logits).
    let gemm = p.hip.gemm().expect("hip gemm");
    let act = on_hip(
        &p,
        &[m, INTERMEDIATE],
        DType::BF16,
        &pattern(&mut rng, m * INTERMEDIATE, 1.0),
    );
    let gemms = [
        ("q_proj/o_proj", HIDDEN, HIDDEN, DType::BF16, 2.0 * LAYERS),
        ("k_proj/v_proj", 1024, HIDDEN, DType::BF16, 2.0 * LAYERS),
        ("gate/up", INTERMEDIATE, HIDDEN, DType::BF16, 2.0 * LAYERS),
        ("down", HIDDEN, INTERMEDIATE, DType::BF16, LAYERS),
        ("lm_head", VOCAB, HIDDEN, DType::F32, 1.0),
    ];
    for (name, n, k, c_dtype, calls) in gemms {
        let cfg = GemmConfig {
            n: n as u64,
            k: k as u64,
            trans_b: true,
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype,
        };
        let scale = 1.0 / (k as f32).sqrt();
        // Enough weight copies (cycled) that the weights never stay in the 64 MB L2/MALL
        // caches between calls, as in a forward where every layer has its own weights.
        let bytes = n * k * 2;
        let raw = encode(DType::BF16, &pattern(&mut rng, n * k, scale));
        let copies = (512usize << 20).div_ceil(bytes).max(1);
        let ws: Vec<Tensor> = (0..copies)
            .map(|_| raw_on_hip(&p, &[n, k], DType::BF16, &raw))
            .collect();
        let c = on_hip(&p, &[m, n], c_dtype, &vec![0.0; m * n]);
        let a = TensorView::contiguous(act.storage.whole(), 0, &[m, k], DType::BF16);
        let mut next = 0usize;
        let us = time_us(&p, ITERS, || {
            next = (next + 1) % copies;
            gemm.execute(&mut GemmContext {
                a: a.clone(),
                b: ws[next].view(),
                c: c.view(),
                trans_b: true,
                alpha: 1.0,
                beta: 0.0,
            })
            .expect("gemm");
        });
        t.report(
            &format!(
                "gemm {name} m={m} n={n} k={k} ({:.0} GB/s)",
                bytes as f64 / (us * 1e3)
            ),
            &gemm.implementation(&cfg),
            us,
            calls,
        );
    }

    // RMSNorm (2 per layer + the final norm).
    let cfg = NormConfig {
        dim: HIDDEN as u64,
        dtype: DType::BF16,
    };
    let norm = p.hip.norm().expect("hip norm");
    let x = on_hip(
        &p,
        &[m, HIDDEN],
        DType::BF16,
        &pattern(&mut rng, m * HIDDEN, 1.0),
    );
    let w = on_hip(&p, &[HIDDEN], DType::BF16, &pattern(&mut rng, HIDDEN, 1.0));
    let o = on_hip(&p, &[m, HIDDEN], DType::BF16, &vec![0.0; m * HIDDEN]);
    let us = time_us(&p, ITERS, || {
        norm.execute(&mut NormContext {
            x: x.view(),
            weight: w.view(),
            out: o.view(),
            eps: 1e-5,
        })
        .expect("rmsnorm");
    });
    t.report(
        "rmsnorm",
        &norm.implementation(&cfg),
        us,
        2.0 * LAYERS + 1.0,
    );

    // Residual add (2 per layer).
    let cfg = ElementwiseConfig { dtype: DType::BF16 };
    let add = p.hip.elementwise().expect("hip add");
    let us = time_us(&p, ITERS, || {
        add.execute(&mut ElementwiseContext {
            a: x.view(),
            b: o.view(),
            out: x.view(),
        })
        .expect("add");
    });
    t.report("add", &add.implementation(&cfg), us, 2.0 * LAYERS);

    // SiLU · up.
    let cfg = ActivationConfig {
        cols: INTERMEDIATE as u64,
        dtype: DType::BF16,
    };
    let silu = p.hip.activation().expect("hip silu_mul");
    let g = on_hip(
        &p,
        &[m, INTERMEDIATE],
        DType::BF16,
        &pattern(&mut rng, m * INTERMEDIATE, 1.0),
    );
    let u = on_hip(
        &p,
        &[m, INTERMEDIATE],
        DType::BF16,
        &pattern(&mut rng, m * INTERMEDIATE, 1.0),
    );
    let us = time_us(&p, ITERS, || {
        silu.execute(&mut ActivationContext {
            gate: g.view(),
            up: u.view(),
            out: act.view(),
        })
        .expect("silu_mul");
    });
    t.report("silu_mul", &silu.implementation(&cfg), us, LAYERS);

    // RoPE on the batch's q and k rows.
    let cfg = RopeConfig {
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        rotary_dim: HEAD_DIM as u32,
        dtype: DType::BF16,
    };
    let rope = p.hip.rope().expect("hip rope");
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / ROPE_THETA.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let positions: Vec<f32> = (0..m).map(|i| (590 + i) as f32).collect();
    let q_shape = [m, Q_HEADS, HEAD_DIM];
    let kv_shape = [m, KV_HEADS, HEAD_DIM];
    let q = on_hip(
        &p,
        &q_shape,
        DType::BF16,
        &pattern(&mut rng, m * q_rows, 1.0),
    );
    let k = on_hip(
        &p,
        &kv_shape,
        DType::BF16,
        &pattern(&mut rng, m * kv_rows, 1.0),
    );
    let v = on_hip(
        &p,
        &kv_shape,
        DType::BF16,
        &pattern(&mut rng, m * kv_rows, 1.0),
    );
    let pos = on_hip(&p, &[m], DType::I32, &positions);
    let freq = on_hip(&p, &[half], DType::F32, &inv_freq);
    let us = time_us(&p, ITERS, || {
        rope.execute(&mut RopeContext {
            cfg,
            q: q.view(),
            k: k.view(),
            positions: pos.view(),
            inv_freq: freq.view(),
        })
        .expect("rope");
    });
    t.report("rope", &rope.implementation(&cfg), us, LAYERS);

    // Embedding of the batch's tokens.
    let cfg = EmbeddingConfig {
        hidden: HIDDEN as u64,
        vocab_rows: VOCAB as u64,
        dtype: DType::BF16,
    };
    let emb = p.hip.embedding().expect("hip embedding");
    let table = on_hip(
        &p,
        &[VOCAB, HIDDEN],
        DType::BF16,
        &pattern(&mut rng, VOCAB * HIDDEN, 1.0),
    );
    let ids: Vec<f32> = (0..m).map(|i| ((i * 7919) % VOCAB) as f32).collect();
    let ids = on_hip(&p, &[m], DType::I32, &ids);
    let us = time_us(&p, ITERS, || {
        emb.execute(&mut EmbeddingContext {
            ids: ids.view(),
            table: table.view(),
            out: x.view(),
            vocab_offset: 0,
        })
        .expect("embedding");
    });
    t.report("embedding", &emb.implementation(&cfg), us, 1.0);

    // Paged decode attention (append + attend), 16 sequences: 16-token pages (the Turbine
    // kernel, the default page) with ~600 cached tokens is the forward estimate; 128-token pages
    // (Composable Kernel pagedkv) and the longer contexts are for comparison.
    let attn = p.hip.attention().expect("hip attention");
    let o = on_hip(&p, &q_shape, DType::BF16, &vec![0.0; m * q_rows]);
    let cases = [
        (16usize, 600usize, LAYERS),
        (16, 2048, 0.0),
        (16, 8192, 0.0),
        (128, 600, 0.0),
        (128, 2048, 0.0),
        (128, 8192, 0.0),
    ];
    for (block_tokens, base, calls) in cases {
        let cfg = AttentionConfig {
            kind: AttentionKind::DecodePaged,
            num_q_heads: Q_HEADS as u32,
            num_kv_heads: KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            dtype: DType::BF16,
            block_tokens: Some(block_tokens as u32),
            causal: true,
        };
        let kv_lens: Vec<usize> = (0..m).map(|s| base - 40 + 5 * s).collect();
        let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
        let max_blocks = max_kv.div_ceil(block_tokens);
        let num_blocks = m * max_blocks;
        let table: Vec<f32> = shuffled(&mut rng, num_blocks)
            .iter()
            .map(|&b| b as f32)
            .collect();
        let pool_len = num_blocks * 2 * block_tokens * kv_rows;
        let pool = on_hip(
            &p,
            &[num_blocks, 2, block_tokens, KV_HEADS, HEAD_DIM],
            DType::BF16,
            &pattern(&mut rng, pool_len, 1.0),
        );
        let bt = on_hip(&p, &[m, max_blocks], DType::I32, &table);
        let indptr: Vec<f32> = (0..=m).map(|i| i as f32).collect();
        let ip = on_hip(&p, &[m + 1], DType::I32, &indptr);
        let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
        let kl = on_hip(&p, &[m], DType::I32, &lens);
        let us = time_us(&p, ITERS, || {
            attn.execute_paged(&mut PagedAttentionContext {
                cfg,
                q: q.view(),
                k_new: k.view(),
                v_new: v.view(),
                out: o.view(),
                kv_layer: pool.view(),
                block_table: bt.view(),
                q_indptr: ip.view(),
                kv_lens: kl.view(),
                max_q_len: 1,
                max_kv_len: max_kv as u32,
                max_blocks_per_seq: max_blocks as u32,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            })
            .expect("paged decode attention");
        });
        t.report(
            &format!("attention_decode_paged bt={block_tokens} kv~{base}"),
            &attn.implementation(&cfg),
            us,
            calls,
        );
    }
    println!(
        "timing decode forward estimate (sum of the ops): {:.3} ms",
        t.per_forward_us / 1e3
    );
}

/// A HIP tensor holding the already encoded `raw` bytes.
fn raw_on_hip(p: &Pair, shape: &[usize], dtype: DType, raw: &[u8]) -> Tensor {
    let mut t = Tensor::empty(&p.hip_mem, shape, dtype).expect("HIP alloc");
    t.storage.copy_from_host(0, raw).expect("copy to HIP");
    t
}

/// One synthetic Llama-3.2-3B decoder layer: its own weights and its own KV pool.
struct BenchLayer {
    norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    pool: Tensor,
}

/// Lab microbenchmark (no assertion on speed): one whole synthetic Llama-3.2-3B decode forward
/// (28 layers with distinct weights and KV pools, so nothing stays in the GPU caches between
/// layers) for 16 sequences with ~600 cached tokens each, with 16- and 128-token pages. Prints
/// the host enqueue time and the time until the stream drains. Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_ops -- decode_forward_timing`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn decode_forward_timing() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(6);
    const LAYERS: usize = 28;
    let m = 16usize;
    let (q_rows, kv_rows) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let kv_lens: Vec<usize> = (0..m).map(|s| 560 + 5 * s).collect();
    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);

    let weight = |rng: &mut Rng, n: usize, k: usize| {
        encode(DType::BF16, &pattern(rng, n * k, 1.0 / (k as f32).sqrt()))
    };
    let (w_qo, w_kv) = (
        weight(&mut rng, HIDDEN, HIDDEN),
        weight(&mut rng, 1024, HIDDEN),
    );
    let (w_gu, w_d) = (
        weight(&mut rng, INTERMEDIATE, HIDDEN),
        weight(&mut rng, HIDDEN, INTERMEDIATE),
    );
    let norm_raw = encode(DType::BF16, &pattern(&mut rng, HIDDEN, 1.0));
    let table_raw = encode(DType::BF16, &pattern(&mut rng, VOCAB * HIDDEN, 1.0));
    let embed = raw_on_hip(&p, &[VOCAB, HIDDEN], DType::BF16, &table_raw);
    drop(table_raw);
    let final_norm = raw_on_hip(&p, &[HIDDEN], DType::BF16, &norm_raw);

    // Activations.
    let zeros = |cols: usize| on_hip(&p, &[m, cols], DType::BF16, &vec![0.0; m * cols]);
    let (x, h, proj) = (zeros(HIDDEN), zeros(HIDDEN), zeros(HIDDEN));
    let (q, attn_out) = (zeros(q_rows), zeros(q_rows));
    let (k, v) = (zeros(kv_rows), zeros(kv_rows));
    let (gate, up, act) = (
        zeros(INTERMEDIATE),
        zeros(INTERMEDIATE),
        zeros(INTERMEDIATE),
    );
    let logits = on_hip(&p, &[m, VOCAB], DType::F32, &vec![0.0; m * VOCAB]);
    let ids: Vec<f32> = (0..m).map(|i| ((i * 7919) % VOCAB) as f32).collect();
    let ids = on_hip(&p, &[m], DType::I32, &ids);
    let positions: Vec<f32> = kv_lens.iter().map(|&k| (k - 1) as f32).collect();
    let positions = on_hip(&p, &[m], DType::I32, &positions);
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / ROPE_THETA.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let inv_freq = on_hip(&p, &[half], DType::F32, &inv_freq);
    let indptr: Vec<f32> = (0..=m).map(|i| i as f32).collect();
    let q_indptr = on_hip(&p, &[m + 1], DType::I32, &indptr);
    let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
    let kv_lens_t = on_hip(&p, &[m], DType::I32, &lens);

    let gemm = p.hip.gemm().expect("hip gemm");
    let norm = p.hip.norm().expect("hip norm");
    let add = p.hip.elementwise().expect("hip add");
    let silu = p.hip.activation().expect("hip silu_mul");
    let rope = p.hip.rope().expect("hip rope");
    let emb = p.hip.embedding().expect("hip embedding");
    let attn = p.hip.attention().expect("hip attention");
    let rope_cfg = RopeConfig {
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        rotary_dim: HEAD_DIM as u32,
        dtype: DType::BF16,
    };
    let linear = |a: &Tensor, w: &Tensor, c: &Tensor| {
        gemm.execute(&mut GemmContext {
            a: a.view(),
            b: w.view(),
            c: c.view(),
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
        })
        .expect("gemm");
    };
    let rmsnorm = |x: &Tensor, w: &Tensor, out: &Tensor| {
        norm.execute(&mut NormContext {
            x: x.view(),
            weight: w.view(),
            out: out.view(),
            eps: 1e-5,
        })
        .expect("rmsnorm");
    };
    let residual = || {
        add.execute(&mut ElementwiseContext {
            a: x.view(),
            b: proj.view(),
            out: x.view(),
        })
        .expect("add");
    };
    fn heads(t: &Tensor, n: usize) -> TensorView<'_> {
        let rows = t.shape[0];
        TensorView::contiguous(t.storage.whole(), 0, &[rows, n, HEAD_DIM], DType::BF16)
    }

    for block_tokens in [16usize, 128] {
        let max_blocks = max_kv.div_ceil(block_tokens);
        let num_blocks = m * max_blocks;
        let table: Vec<f32> = shuffled(&mut rng, num_blocks)
            .iter()
            .map(|&b| b as f32)
            .collect();
        let block_table = on_hip(&p, &[m, max_blocks], DType::I32, &table);
        let pool_shape = [num_blocks, 2, block_tokens, KV_HEADS, HEAD_DIM];
        let pool_raw = encode(
            DType::BF16,
            &pattern(&mut rng, num_blocks * 2 * block_tokens * kv_rows, 1.0),
        );
        let layers: Vec<BenchLayer> = (0..LAYERS)
            .map(|_| BenchLayer {
                norm: raw_on_hip(&p, &[HIDDEN], DType::BF16, &norm_raw),
                wq: raw_on_hip(&p, &[HIDDEN, HIDDEN], DType::BF16, &w_qo),
                wk: raw_on_hip(&p, &[1024, HIDDEN], DType::BF16, &w_kv),
                wv: raw_on_hip(&p, &[1024, HIDDEN], DType::BF16, &w_kv),
                wo: raw_on_hip(&p, &[HIDDEN, HIDDEN], DType::BF16, &w_qo),
                w_gate: raw_on_hip(&p, &[INTERMEDIATE, HIDDEN], DType::BF16, &w_gu),
                w_up: raw_on_hip(&p, &[INTERMEDIATE, HIDDEN], DType::BF16, &w_gu),
                w_down: raw_on_hip(&p, &[HIDDEN, INTERMEDIATE], DType::BF16, &w_d),
                pool: raw_on_hip(&p, &pool_shape, DType::BF16, &pool_raw),
            })
            .collect();
        let attn_cfg = AttentionConfig {
            kind: AttentionKind::DecodePaged,
            num_q_heads: Q_HEADS as u32,
            num_kv_heads: KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            dtype: DType::BF16,
            block_tokens: Some(block_tokens as u32),
            causal: true,
        };
        let attn_impl = attn.implementation(&attn_cfg);
        let forward = || {
            emb.execute(&mut EmbeddingContext {
                ids: ids.view(),
                table: embed.view(),
                out: x.view(),
                vocab_offset: 0,
            })
            .expect("embedding");
            for l in &layers {
                rmsnorm(&x, &l.norm, &h);
                linear(&h, &l.wq, &q);
                linear(&h, &l.wk, &k);
                linear(&h, &l.wv, &v);
                rope.execute(&mut RopeContext {
                    cfg: rope_cfg,
                    q: heads(&q, Q_HEADS),
                    k: heads(&k, KV_HEADS),
                    positions: positions.view(),
                    inv_freq: inv_freq.view(),
                })
                .expect("rope");
                attn.execute_paged(&mut PagedAttentionContext {
                    cfg: attn_cfg,
                    q: heads(&q, Q_HEADS),
                    k_new: heads(&k, KV_HEADS),
                    v_new: heads(&v, KV_HEADS),
                    out: heads(&attn_out, Q_HEADS),
                    kv_layer: l.pool.view(),
                    block_table: block_table.view(),
                    q_indptr: q_indptr.view(),
                    kv_lens: kv_lens_t.view(),
                    max_q_len: 1,
                    max_kv_len: max_kv as u32,
                    max_blocks_per_seq: max_blocks as u32,
                    scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                })
                .expect("paged decode attention");
                linear(&attn_out, &l.wo, &proj);
                residual();
                rmsnorm(&x, &l.norm, &h);
                linear(&h, &l.w_gate, &gate);
                linear(&h, &l.w_up, &up);
                silu.execute(&mut ActivationContext {
                    gate: gate.view(),
                    up: up.view(),
                    out: act.view(),
                })
                .expect("silu_mul");
                linear(&act, &l.w_down, &proj);
                residual();
            }
            rmsnorm(&x, &final_norm, &h);
            linear(&h, &embed, &logits);
        };
        forward();
        p.hip_mem.synchronize().expect("synchronize");
        let runs = 5u32;
        let (mut host, mut total) = (0.0f64, 0.0f64);
        for _ in 0..runs {
            let start = std::time::Instant::now();
            forward();
            host += start.elapsed().as_secs_f64();
            p.hip_mem.synchronize().expect("synchronize");
            total += start.elapsed().as_secs_f64();
        }
        println!(
            "timing decode forward m={m} kv~{max_kv} block_tokens={block_tokens} attention={attn_impl}: \
             {:.3} ms (host enqueue {:.3} ms)",
            total * 1e3 / f64::from(runs),
            host * 1e3 / f64::from(runs)
        );
    }
}
