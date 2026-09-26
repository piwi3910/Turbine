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
    ActivationConfig, ActivationContext, AttentionConfig, AttentionContext, AttentionKind,
    ElementwiseConfig, ElementwiseContext, EmbeddingConfig, EmbeddingContext, GemmConfig,
    GemmContext, KernelProvider, KvCopyConfig, KvCopyContext, MoeExpertsConfig, MoeExpertsContext,
    MoeRouteConfig, MoeRouteContext, NormConfig, NormContext, PagedAttentionContext, RopeConfig,
    RopeContext, ShimLibrary, cpu_reference_provider, shim_provider,
};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, Tensor};

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

/// One ragged batch through `attention_{prefill,decode}_paged` on both providers: sequence `s`
/// has `q_lens[s]` new tokens and `kv_lens[s]` tokens after the append, its pages spread over a
/// shuffled block table of a pool with two blocks nobody owns. Compares the outputs and the pool
/// after the append.
fn paged_case(
    p: &Pair,
    rng: &mut Rng,
    kind: AttentionKind,
    block_tokens: usize,
    q_lens: &[usize],
    kv_lens: &[usize],
) {
    let cfg = AttentionConfig {
        kind,
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
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
    let (q_rows, kv_rows) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let pool_shape = [num_blocks, 2, block_tokens, KV_HEADS, HEAD_DIM];
    let pool_len = num_blocks * 2 * block_tokens * kv_rows;

    // The pool starts with random history in every slot, including the unowned blocks.
    let (pool_hip, pool_cpu) = twin(p, &pool_shape, DType::BF16, &rng.normal(pool_len, 1.0));
    let q_shape = [total_q, Q_HEADS, HEAD_DIM];
    let new_shape = [total_q, KV_HEADS, HEAD_DIM];
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
        "{} block_tokens={block_tokens} q_lens={q_lens:?} kv_lens={kv_lens:?}",
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
/// agree) over the local experts `local`; the HIP run uses a caller workspace when `workspace`.
struct ExpertsCase<'a> {
    tokens: usize,
    logits: &'a [f32],
    local: (usize, usize),
    workspace: bool,
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
    let impl_name = hip.implementation_experts(&cfg);
    let rows = tokens * MOE_TOP_K;
    let (x_hip, x_cpu) = twin(
        p,
        &[tokens, MOE_HIDDEN],
        DType::BF16,
        &rng.normal(tokens * MOE_HIDDEN, 1.0),
    );
    let (o_hip, o_cpu) = twin(
        p,
        &[tokens, MOE_HIDDEN],
        DType::BF16,
        &rng.normal(tokens * MOE_HIDDEN, 0.1),
    );
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
            ws.as_ref(),
        ),
        (
            p.cpu.moe().expect("cpu moe"),
            &w.cpu,
            [&x_cpu, &o_cpu, &s_cpu, &off_cpu, &tw_cpu],
            None,
        ),
    ];
    for (kernel, [wg, wu, wd], [x, o, s, off, tw], ws) in runs {
        let mut ctx = MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: wg.view().rows(begin, local),
            w_up: wu.view().rows(begin, local),
            w_down: wd.view().rows(begin, local),
            sorted_rows: s.view(),
            expert_offsets: off.view(),
            topk_weights: tw.view(),
            host_expert_offsets: &host_offsets,
            out: o.view(),
            workspace: ws.map(|t| t.view().slice),
        };
        kernel.experts(&mut ctx).expect("moe_experts");
    }
    assert_close(
        &format!(
            "moe_experts tokens={tokens} local=[{begin},{end}) empty_local_experts={empty} caller_workspace={} {cfg}",
            case.workspace
        ),
        &impl_name,
        &read(&o_hip),
        &read(&o_cpu),
        DType::BF16,
    );
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

    // Paged attention at the Llama-3.2-3B shapes: the 16-token default page (Turbine kernel)
    // and a 128-token page (Composable Kernel pagedkv). A prefill batch mixing a fresh prompt,
    // a single-token row and a chunk after cached context (more than one query tile), then a
    // decode batch.
    for block_tokens in [16, 128] {
        paged_case(
            &p,
            &mut rng,
            AttentionKind::PrefillPaged,
            block_tokens,
            &[37, 1, 70],
            &[37, 300, 200],
        );
        paged_case(
            &p,
            &mut rng,
            AttentionKind::DecodePaged,
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
    };
    experts_case(&p, &mut rng, &w, &case);
    // A shard of the experts: the local range [16, 48).
    let logits = rng.normal(21 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 21,
        logits: &logits,
        local: (16, 48),
        workspace: false,
    };
    experts_case(&p, &mut rng, &w, &case);
}
