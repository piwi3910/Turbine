//! Lab only (novanas R9700, P1 AC S-7/S-13, P2 S-5/S-16): every op the Llama and OLMoE
//! executors use, run through `libturbine_hip.so` via the Rust shim bindings, matches the
//! `cpu-reference` provider on seeded random inputs at the Llama-3.2-3B and OLMoE-1B-7B shapes.
//! Run by `scripts/lab-test.sh novanas`, which sets `TURBINE_TEST_BACKEND=hip`,
//! `TURBINE_KERNEL_LIBRARY` and `TURBINE_AMD_SMI_LIBRARY`.
//!
//! Tolerance: BF16 outputs |Δ| ≤ 1e-2, or one BF16 ulp of the reference where its magnitude
//! is 2 or more (a rounding flip after a different f32 summation order); F32 outputs |Δ| ≤ 1e-4;
//! copies (the paged K/V append, `copy_blocks`) and `moe_route` selections are exact;
//! `logits_reduce` (ABI v2.1) top-n ids are exact, its lse within 1e-5 relative and its draws
//! identical except within 1e-6 of a CDF boundary; the sharded RMSNorm (ABI v2.6) as stated at
//! `sharded_norm_ops`.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use half::bf16;
use turbine_core::config::DevicesConfig;
use turbine_core::types::{BlockId, DType, DeviceId, Vendor};
use turbine_device::{DiscoveryOptions, discover};
use turbine_kernels::cards::GFX1201;
use turbine_kernels::cpu::quant::fp8_e4m3_round;
use turbine_kernels::test_support::require_backend;
use turbine_kernels::{
    ActivationConfig, ActivationContext, AddRmsnormConfig, AddRmsnormContext, AttentionConfig,
    AttentionContext, AttentionKind, ElementwiseConfig, ElementwiseContext, EmbeddingConfig,
    EmbeddingContext, GemmConfig, GemmContext, ImplChoice, ImplInfo, KernelProvider, KvCopyConfig,
    KvCopyContext, LogitsReduceConfig, LogitsReduceContext, LogitsReduceKernel, MoeExpertsConfig,
    MoeExpertsContext, MoeRouteConfig, MoeRouteContext, NormConfig, NormContext, OpConfig, OpKind,
    PagedAttentionContext, RmsnormShardedConfig, RmsnormShardedContext, RopeConfig, RopeContext,
    RowSumsqConfig, RowSumsqContext, ShimContext, ShimLibrary, TURBINE_OPTION_GEMM_AUTOTUNE,
    TURBINE_OPTION_GEMM_TUNED_SHAPES, cpu_reference_provider, shim_provider,
};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, HostStaging, Tensor, TensorView};

// Llama-3.2-3B shapes.
const HIDDEN: usize = 3072;
const INTERMEDIATE: usize = 8192;
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB: usize = 128256;
const ROPE_THETA: f64 = 500_000.0;
/// YaRN's attention factor of the factor-16 Llama override (`0.1 · ln 16 + 1`, ABI v2.10).
const YARN16_ATTN_FACTOR: f32 = 1.277_258_9;

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
    /// The HIP provider takes the library's own choice per call (`false`: bound to one
    /// implementation by `every_implementation_matches_cpu`).
    legacy: bool,
    /// The HIP context (its options) and the device architecture.
    ctx: Arc<ShimContext>,
    arch: String,
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
    let lib = ShimLibrary::load(&path, "hip").expect("load libturbine_hip.so");
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
        hip: shim_provider(Arc::clone(&ctx)),
        cpu: cpu_reference_provider(),
        hip_mem,
        cpu_mem,
        legacy: true,
        ctx,
        arch: device
            .arch
            .clone()
            .expect("the AMD device reports its architecture"),
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

/// |Δ| ≤ 1e-2, or one BF16 ulp of `want` where |want| ≥ 2 (at exactly 2 the ulp above, 2^-6, exceeds 1e-2).
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
            prefill: false,
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
fn bf16_gemm_matches_cpu() {
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

/// RMSNorm over `t` rows of Llama's hidden size.
fn norm_case(p: &Pair, rng: &mut Rng, t: usize) {
    // RMSNorm
    let cfg = NormConfig {
        dim: HIDDEN as u64,
        dtype: DType::BF16,
    };
    let hip = p.hip.norm().expect("hip norm");
    assert!(hip.supports(&cfg), "hip must support rmsnorm {cfg}");
    let impl_name = hip.implementation(&cfg);
    let (x_hip, x_cpu) = twin(p, &[t, HIDDEN], DType::BF16, &rng.normal(t * HIDDEN, 1.0));
    let (w_hip, w_cpu) = twin(p, &[HIDDEN], DType::BF16, &rng.normal(HIDDEN, 1.0));
    let (o_hip, o_cpu) = twin(p, &[t, HIDDEN], DType::BF16, &vec![0.0; t * HIDDEN]);
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
}

/// RoPE over `t` tokens at the Llama head shapes, cos and sin scaled by `attn_factor` (YaRN's
/// attention factor, kernel ABI v2.10; 1.0 = none).
fn rope_case(p: &Pair, rng: &mut Rng, t: usize, attn_factor: f32) {
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
        p,
        &[t, Q_HEADS, HEAD_DIM],
        DType::BF16,
        &rng.normal(q_n, 1.0),
    );
    let (k_hip, k_cpu) = twin(
        p,
        &[t, KV_HEADS, HEAD_DIM],
        DType::BF16,
        &rng.normal(k_n, 1.0),
    );
    let (pos_hip, pos_cpu) = twin(p, &[t], DType::I32, &positions);
    let (f_hip, f_cpu) = twin(p, &[half], DType::F32, &inv_freq);
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
            attn_factor,
        };
        kernel.execute(&mut ctx).expect("rope");
    }
    let what = format!("rope q tokens={t} attn_factor={attn_factor} {cfg}");
    assert_close(&what, &impl_name, &read(&q_hip), &read(&q_cpu), DType::BF16);
    let what = format!("rope k tokens={t} attn_factor={attn_factor} {cfg}");
    assert_close(&what, &impl_name, &read(&k_hip), &read(&k_cpu), DType::BF16);
}

/// SiLU · up over `t` rows of Llama's intermediate size.
fn silu_mul_case(p: &Pair, rng: &mut Rng, t: usize) {
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
    let (g_hip, g_cpu) = twin(p, &shape, DType::BF16, &rng.normal(n, 1.0));
    let (u_hip, u_cpu) = twin(p, &shape, DType::BF16, &rng.normal(n, 1.0));
    let (o_hip, o_cpu) = twin(p, &shape, DType::BF16, &vec![0.0; n]);
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
}

/// Embedding of `t` tokens from the full Llama vocabulary.
fn embedding_case(p: &Pair, rng: &mut Rng, t: usize) {
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
    let (tab_hip, tab_cpu) = twin(p, &[VOCAB, HIDDEN], DType::BF16, &table);
    let (id_hip, id_cpu) = twin(p, &[t], DType::I32, &ids);
    let (o_hip, o_cpu) = twin(p, &[t, HIDDEN], DType::BF16, &vec![0.0; t * HIDDEN]);
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
}

/// Residual add of `t` rows of Llama's hidden size.
fn add_case(p: &Pair, rng: &mut Rng, t: usize) {
    // Residual add
    let cfg = ElementwiseConfig { dtype: DType::BF16 };
    let hip = p.hip.elementwise().expect("hip add");
    assert!(hip.supports(&cfg), "hip must support add {cfg}");
    let impl_name = hip.implementation(&cfg);
    let n = t * HIDDEN;
    let (a_hip, a_cpu) = twin(p, &[n], DType::BF16, &rng.normal(n, 1.0));
    let (b_hip, b_cpu) = twin(p, &[n], DType::BF16, &rng.normal(n, 1.0));
    let (o_hip, o_cpu) = twin(p, &[n], DType::BF16, &vec![0.0; n]);
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
    norm_case(&p, &mut rng, t);
    rope_case(&p, &mut rng, t, 1.0);
    rope_case(&p, &mut rng, t, YARN16_ATTN_FACTOR);
    silu_mul_case(&p, &mut rng, t);
    embedding_case(&p, &mut rng, t);
    add_case(&p, &mut rng, t);
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

/// One fused residual add + RMSNorm case of `add_rmsnorm_matches_cpu` (rows, dim, row stride
/// of x).
fn add_rmsnorm_case(p: &Pair, rng: &mut Rng, rows: usize, dim: usize, x_stride: usize) {
    let hip = p
        .hip
        .add_rmsnorm()
        .expect("libturbine_hip.so exports the ABI v2.1 add_rmsnorm trio");
    let cpu = p.cpu.add_rmsnorm().expect("cpu add_rmsnorm");
    let cfg = AddRmsnormConfig {
        dtype: DType::BF16,
        dim: dim as u32,
    };
    assert!(hip.supports(&cfg), "hip must support add_rmsnorm {cfg}");
    let impl_name = hip.implementation(&cfg);
    let n = rows * dim;
    let before = rng.normal(n, 1.0);
    let (r_hip, r_cpu) = twin(p, &[rows, dim], DType::BF16, &before);
    let x = rng.normal(rows * x_stride, 1.0);
    let (x_hip, x_cpu) = twin(p, &[rows, x_stride], DType::BF16, &x);
    let (w_hip, w_cpu) = twin(p, &[dim], DType::BF16, &rng.normal(dim, 1.0));
    let (o_hip, o_cpu) = twin(p, &[rows, dim], DType::BF16, &vec![0.0; n]);
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
    if p.legacy && dim == HIDDEN && x_stride == dim {
        let (r2, _) = twin(p, &[rows, dim], DType::BF16, &before);
        let (o2, _) = twin(p, &[rows, dim], DType::BF16, &vec![0.0; n]);
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
        add_rmsnorm_case(&p, &mut rng, rows, dim, x_stride);
    }
}

/// Columns `[start, start + cols)` of every row of the `[rows, stride]` tensor `t`, as a
/// row-strided `[rows, cols]` view (one tensor-parallel rank's slice of the row).
fn column_slice(t: &Tensor, start: usize, cols: usize) -> TensorView<'_> {
    let (rows, stride) = (t.shape[0], t.shape[1]);
    let es = t.dtype.size_bytes();
    TensorView {
        slice: t
            .storage
            .whole()
            .sub(start * es, ((rows - 1) * stride + cols) * es),
        shape: (&[rows, cols][..]).into(),
        strides: (&[stride, 1][..]).into(),
        dtype: t.dtype,
    }
}

/// Rows `[first, first + count)` of the `[rows, cols]` tensor `t` (dense rows).
fn row_range(t: &Tensor, first: usize, count: usize) -> TensorView<'_> {
    let cols = t.shape.get(1).copied().unwrap_or(1);
    let es = t.dtype.size_bytes();
    let shape: Vec<usize> = if t.shape.len() == 2 {
        vec![count, cols]
    } else {
        vec![count]
    };
    let strides: Vec<usize> = if t.shape.len() == 2 {
        vec![cols, 1]
    } else {
        vec![1]
    };
    TensorView {
        slice: t.storage.whole().sub(first * cols * es, count * cols * es),
        shape: shape.as_slice().into(),
        strides: strides.as_slice().into(),
        dtype: t.dtype,
    }
}

/// One tensor-parallel sharded RMSNorm case of `sharded_norm_ops`: `rows` BF16 rows of `full`
/// split into `tp` equal slices. Per rank, HIP `row_sumsq` of the slice vs the CPU reference (F32,
/// |Δ| ≤ 1e-5 relative: only the summation order differs); the per-rank sums are added on the host
/// (the all-reduce) and the same sums go to both providers' `rmsnorm_sharded`, whose BF16 outputs
/// must be bitwise equal on at least 99.9 % of the elements and within the BF16 tolerance on the
/// rest (a 1-ulp f32 difference of `inv_rms` flipping a rounding of `x·inv_rms`). Returns the HIP
/// implementation names (`row_sumsq`, `rmsnorm_sharded`).
fn sharded_norm_case(
    p: &Pair,
    rng: &mut Rng,
    rows: usize,
    full: usize,
    tp: usize,
) -> (String, String) {
    let hip = p
        .hip
        .sharded_norm()
        .expect("libturbine_hip.so exports the ABI v2.6 sharded RMSNorm group");
    let cpu = p.cpu.sharded_norm().expect("cpu sharded norm");
    let dim = full / tp;
    let bf16 = DType::BF16;
    let sum_cfg = RowSumsqConfig {
        dim: dim as u32,
        dtype: bf16,
    };
    let norm_cfg = RmsnormShardedConfig {
        dim: dim as u32,
        full_dim: full as u32,
        dtype: bf16,
    };
    assert!(hip.supports_row_sumsq(&sum_cfg), "hip row_sumsq {sum_cfg}");
    assert!(
        hip.supports_rmsnorm_sharded(&norm_cfg),
        "hip rmsnorm_sharded {norm_cfg}"
    );
    let names = (
        hip.implementation_row_sumsq(&sum_cfg),
        hip.implementation_rmsnorm_sharded(&norm_cfg),
    );
    let (x_hip, x_cpu) = twin(p, &[rows, full], bf16, &rng.normal(rows * full, 1.0));
    let w: Vec<f32> = rng.normal(full, 0.5).iter().map(|v| 1.0 + v).collect();

    let mut total = vec![0f32; rows];
    for rank in 0..tp {
        let (s_hip, s_cpu) = twin(p, &[rows], DType::F32, &vec![0.0; rows]);
        for (kernel, x, sumsq) in [(hip, &x_hip, &s_hip), (cpu, &x_cpu, &s_cpu)] {
            kernel
                .row_sumsq(&mut RowSumsqContext {
                    x: column_slice(x, rank * dim, dim),
                    sumsq: sumsq.view(),
                })
                .expect("row_sumsq");
        }
        let (got, want) = (read(&s_hip), read(&s_cpu));
        for (r, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 * w.abs(),
                "row_sumsq rows={rows} {sum_cfg} rank {rank} row {r}: hip {g} vs cpu {w}"
            );
        }
        for (t, v) in total.iter_mut().zip(&want) {
            *t += v;
        }
    }
    println!(
        "row_sumsq rows={rows} {sum_cfg} tp={tp}: impl={} within 1e-5 relative ok",
        names.0
    );

    let (s_hip, s_cpu) = twin(p, &[rows], DType::F32, &total);
    let (o_hip, o_cpu) = twin(p, &[rows, full], bf16, &vec![0.0; rows * full]);
    for rank in 0..tp {
        let (w_hip, w_cpu) = twin(p, &[dim], bf16, &w[rank * dim..(rank + 1) * dim]);
        for (kernel, x, sumsq, weight, out) in [
            (hip, &x_hip, &s_hip, &w_hip, &o_hip),
            (cpu, &x_cpu, &s_cpu, &w_cpu, &o_cpu),
        ] {
            kernel
                .rmsnorm_sharded(&mut RmsnormShardedContext {
                    x: column_slice(x, rank * dim, dim),
                    sumsq: sumsq.view(),
                    weight: weight.view(),
                    out: column_slice(out, rank * dim, dim),
                    full_dim: full as u32,
                    eps: 1e-5,
                })
                .expect("rmsnorm_sharded");
        }
    }
    let (got, want) = (read(&o_hip), read(&o_cpu));
    let what = format!("rmsnorm_sharded rows={rows} {norm_cfg} tp={tp}");
    let mut differ = 0usize;
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        if g.to_bits() == w.to_bits() {
            continue;
        }
        assert!(
            bf16_close(g, w),
            "{what} ({}): element {i}: hip {g} vs cpu {w}",
            names.1
        );
        differ += 1;
    }
    assert!(
        differ * 1000 <= got.len(),
        "{what} ({}): {differ} of {} elements not bitwise equal",
        names.1,
        got.len()
    );
    println!(
        "{what}: impl={} {differ} of {} elements one rounding step off, the rest bitwise ok",
        names.1,
        got.len()
    );
    names
}

/// Phase 5 Task 6 (kernel ABI v2.6, decision "P5 T6", answer B): the tensor-parallel sharded
/// RMSNorm of OLMoE's QK-norm (a row of 16 heads × 128 = 2,048 split by heads over 2 and 4
/// ranks) matches the CPU reference (`sharded_norm_case`: sums within 1e-5 relative, outputs
/// bitwise except ≤ 0.1 % one-rounding-step flips). Batch invariance: a row's `row_sumsq` and
/// `rmsnorm_sharded` alone are bit-identical to the same row inside a 37-row call. One slice
/// spanning the whole row with its own sum is bitwise the HIP `rmsnorm` (the Turbine kernel at
/// 2,048: the same reduction order). `turbine_stream_native_handle` gives the compute stream a
/// non-null native handle. Breaks if a kernel reduces in a row-count-dependent order, divides by
/// the slice width, or the library does not resolve the v2.6 group.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn sharded_norm_ops() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    assert!(
        p.ctx.library().exports_native_streams(),
        "libturbine_hip.so exports the ABI v2.6 group (minor {})",
        p.ctx.library().abi_minor()
    );
    assert!(p.ctx.has_native_streams());
    let stream = p.hip_mem.compute_stream();
    assert_ne!(
        stream.native_handle(),
        0,
        "the compute stream's native handle"
    );
    println!(
        "turbine_stream_native_handle: compute stream 0x{:x} ok",
        stream.native_handle()
    );

    let mut rng = Rng(61);
    let full = MOE_HIDDEN;
    for (rows, tp) in [(1, 2), (37, 2), (37, 4), (2048, 2)] {
        let names = sharded_norm_case(&p, &mut rng, rows, full, tp);
        assert_eq!(names, ("turbine_hip".into(), "turbine_hip".into()));
    }

    // Row invariance: row 17 alone vs inside the 37-row call, both ops, tp = 2.
    let hip = p.hip.sharded_norm().expect("v2.6 group");
    let (rows, dim) = (37usize, full / 2);
    let x = on_hip(
        &p,
        &[rows, full],
        DType::BF16,
        &rng.normal(rows * full, 1.0),
    );
    let w = on_hip(&p, &[dim], DType::BF16, &rng.normal(dim, 1.0));
    let all_sums = zeros_on_hip(&p, &[rows], DType::F32);
    let one_sum = zeros_on_hip(&p, &[1], DType::F32);
    let x_slice = column_slice(&x, dim, dim);
    let run_sumsq = |x: TensorView<'_>, sumsq: TensorView<'_>| {
        hip.row_sumsq(&mut RowSumsqContext { x, sumsq })
            .expect("row_sumsq")
    };
    run_sumsq(x_slice.clone(), all_sums.view());
    let row = 17usize;
    let x_row = TensorView {
        slice: x.storage.whole().sub((row * full + dim) * 2, dim * 2),
        shape: (&[1usize, dim][..]).into(),
        strides: (&[full, 1][..]).into(),
        dtype: DType::BF16,
    };
    run_sumsq(x_row.clone(), one_sum.view());
    let (all, one) = (read(&all_sums), read(&one_sum));
    assert_eq!(
        all[row].to_bits(),
        one[0].to_bits(),
        "row_sumsq of row {row} alone vs in a {rows}-row call"
    );
    let all_out = zeros_on_hip(&p, &[rows, dim], DType::BF16);
    let one_out = zeros_on_hip(&p, &[1, dim], DType::BF16);
    for (xv, sums, out) in [
        (x_slice, all_sums.view(), all_out.view()),
        (x_row, row_range(&all_sums, row, 1), one_out.view()),
    ] {
        hip.rmsnorm_sharded(&mut RmsnormShardedContext {
            x: xv,
            sumsq: sums,
            weight: w.view(),
            out,
            full_dim: full as u32,
            eps: 1e-5,
        })
        .expect("rmsnorm_sharded");
    }
    let (all, one) = (read(&all_out), read(&one_out));
    assert!(
        all[row * dim..(row + 1) * dim]
            .iter()
            .zip(&one)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "rmsnorm_sharded of row {row} alone vs in a {rows}-row call"
    );
    println!("sharded norm row invariance: row {row} alone == in {rows} rows (bitwise) ok");

    // One full-width slice with its own sum: bitwise the HIP rmsnorm (Turbine kernel at 2048).
    let norm_cfg = NormConfig {
        dim: full as u64,
        dtype: DType::BF16,
    };
    let norm = p.hip.norm().expect("hip rmsnorm");
    assert_eq!(norm.implementation(&norm_cfg), "turbine_hip");
    let wf = on_hip(&p, &[full], DType::BF16, &rng.normal(full, 1.0));
    let sums = zeros_on_hip(&p, &[rows], DType::F32);
    let (sharded_out, norm_out) = (
        zeros_on_hip(&p, &[rows, full], DType::BF16),
        zeros_on_hip(&p, &[rows, full], DType::BF16),
    );
    run_sumsq(x.view(), sums.view());
    hip.rmsnorm_sharded(&mut RmsnormShardedContext {
        x: x.view(),
        sumsq: sums.view(),
        weight: wf.view(),
        out: sharded_out.view(),
        full_dim: full as u32,
        eps: 1e-5,
    })
    .expect("rmsnorm_sharded");
    norm.execute(&mut NormContext {
        x: x.view(),
        weight: wf.view(),
        out: norm_out.view(),
        eps: 1e-5,
    })
    .expect("rmsnorm");
    assert_exact(
        "rmsnorm_sharded (tp = 1) vs hip rmsnorm",
        "turbine_hip",
        &read(&sharded_out),
        &read(&norm_out),
    );
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
    paged_case_pages(
        p,
        rng,
        kind,
        heads,
        block_tokens,
        q_lens,
        kv_lens,
        Pages::Bf16,
    )
}

/// The KV pages of a paged case: BF16, or FP8 e4m3 with the layer's K and V scales.
#[derive(Clone, Copy, Debug)]
enum Pages {
    Bf16,
    Fp8 { k_scale: f32, v_scale: f32 },
}

/// FP8 history bytes: the e4m3 codes of normal samples (as K/V of scale 1 would be stored), so
/// scores stay in the range real pages give (uniform codes up to ±448 would make every softmax
/// row one-hot and turn rounding-order noise into large differences).
fn fp8_history(rng: &mut Rng, n: usize) -> Vec<u8> {
    rng.normal(4099, 1.0)
        .iter()
        .map(|&x| fp8_e4m3_round(x))
        .cycle()
        .take(n)
        .collect()
}

/// [`paged_case`] over `pages`: FP8 pages start with random e4m3 history and are compared byte
/// for byte after the append.
#[allow(clippy::too_many_arguments)]
fn paged_case_pages(
    p: &Pair,
    rng: &mut Rng,
    kind: AttentionKind,
    heads: (usize, usize),
    block_tokens: usize,
    q_lens: &[usize],
    kv_lens: &[usize],
    pages: Pages,
) -> String {
    let (q_heads, kv_heads) = heads;
    let (page_dtype, (k_scale, v_scale)) = match pages {
        Pages::Bf16 => (DType::BF16, (1.0, 1.0)),
        Pages::Fp8 { k_scale, v_scale } => (DType::F8E4M3, (k_scale, v_scale)),
    };
    let cfg = AttentionConfig {
        kind,
        num_q_heads: q_heads as u32,
        num_kv_heads: kv_heads as u32,
        head_dim: HEAD_DIM as u32,
        dtype: page_dtype,
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
    let (pool_hip, pool_cpu) = match pages {
        Pages::Bf16 => twin(p, &pool_shape, DType::BF16, &rng.normal(pool_len, 1.0)),
        Pages::Fp8 { .. } => {
            let raw = fp8_history(rng, pool_len);
            let mut hip = Tensor::empty(&p.hip_mem, &pool_shape, DType::F8E4M3).expect("HIP");
            let mut cpu = Tensor::empty(&p.cpu_mem, &pool_shape, DType::F8E4M3).expect("host");
            hip.storage.copy_from_host(0, &raw).expect("copy to HIP");
            cpu.storage.copy_from_host(0, &raw).expect("copy to host");
            (hip, cpu)
        }
    };
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
            k_scale,
            v_scale,
        };
        kernel.execute_paged(&mut ctx).expect("paged attention");
    }
    let what = format!(
        "{} heads={q_heads}/{kv_heads} block_tokens={block_tokens} pages={pages:?} \
         q_lens={q_lens:?} kv_lens={kv_lens:?}",
        cfg.op()
    );
    assert_close(
        &format!("{what} out"),
        &impl_name,
        &read(&o_hip),
        &read(&o_cpu),
        DType::BF16,
    );
    if matches!(pages, Pages::Bf16) {
        assert_exact(
            &format!("{what} pool after append"),
            &impl_name,
            &read(&pool_hip),
            &read(&pool_cpu),
        );
    } else {
        let bytes = |t: &Tensor| t.view().slice.read_bytes().expect("read back");
        let (hip_bytes, cpu_bytes) = (bytes(&pool_hip), bytes(&pool_cpu));
        let first = hip_bytes.iter().zip(&cpu_bytes).position(|(h, c)| h != c);
        assert!(
            first.is_none(),
            "{what} pool after append ({impl_name}): byte {first:?} differs"
        );
        println!("{what} pool after append: impl={impl_name} exact ok");
    }
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

/// Expert weights in both memories: gate/up `[E, inter, hidden]`, down `[E, hidden, inter]`.
struct ExpertWeights {
    hidden: usize,
    inter: usize,
    hip: [Tensor; 3],
    cpu: [Tensor; 3],
}

/// The OLMoE expert weights (64 experts, hidden 2048, inter 1024).
fn expert_weights(p: &Pair, rng: &mut Rng) -> ExpertWeights {
    expert_weights_of(p, rng, MOE_HIDDEN, MOE_INTER)
}

/// [`MOE_EXPERTS`] experts of the given hidden and intermediate sizes.
fn expert_weights_of(p: &Pair, rng: &mut Rng, hidden: usize, inter: usize) -> ExpertWeights {
    let n = MOE_EXPERTS * inter * hidden;
    let up_scale = 1.0 / (hidden as f32).sqrt();
    let (g_hip, g_cpu) = twin(
        p,
        &[MOE_EXPERTS, inter, hidden],
        DType::BF16,
        &rng.normal(n, up_scale),
    );
    let (u_hip, u_cpu) = twin(
        p,
        &[MOE_EXPERTS, inter, hidden],
        DType::BF16,
        &rng.normal(n, up_scale),
    );
    let (d_hip, d_cpu) = twin(
        p,
        &[MOE_EXPERTS, hidden, inter],
        DType::BF16,
        &rng.normal(n, 1.0 / (inter as f32).sqrt()),
    );
    ExpertWeights {
        hidden,
        inter,
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
        bf16_logits: true,
    };
    let routing = route_case(p, route_cfg, tokens, case.logits);
    let host_offsets: Vec<i32> = routing.expert_offsets.iter().map(|&o| o as i32).collect();
    let empty = (begin..end)
        .filter(|&e| host_offsets[e] == host_offsets[e + 1])
        .count();

    let (hidden, inter) = (w.hidden, w.inter);
    let cfg = MoeExpertsConfig {
        hidden: hidden as u32,
        inter: inter as u32,
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
    // P2c S-11: the HIP library reads the offsets on the device for up to 512 routed rows
    // (small-m) and, above, whenever hidden and inter are multiples of 64 (grouped WMMA).
    let needs_host = hip.needs_host_offsets(&cfg, rows);
    if p.legacy {
        assert_eq!(
            needs_host,
            rows > MOE_SMALL_M_MAX_ROWS && (hidden % 64 != 0 || inter % 64 != 0),
            "needs_host_offsets for {rows} routed rows, hidden {hidden}, inter {inter}"
        );
    }
    let impl_name = format!(
        "{} host_offsets={needs_host}",
        hip.implementation_experts(&cfg)
    );
    let hip_offsets: &[i32] = if needs_host { &host_offsets } else { &[] };
    let (x_hip, x_cpu) = twin(
        p,
        &[tokens, hidden],
        DType::BF16,
        &rng.normal(tokens * hidden, 1.0),
    );
    let out0 = rng.normal(tokens * hidden, 0.1);
    let (o_hip, o_cpu) = twin(p, &[tokens, hidden], DType::BF16, &out0);
    let (s_hip, s_cpu) = twin(p, &[rows], DType::I32, &routing.sorted_rows);
    let (off_hip, off_cpu) = twin(p, &[MOE_EXPERTS + 1], DType::I32, &routing.expert_offsets);
    let (tw_hip, tw_cpu) = twin(p, &[tokens, MOE_TOP_K], DType::F32, &routing.weights);
    // A generous caller workspace: row positions, gathered rows and the three intermediates.
    let ws = case.workspace.then(|| {
        let bytes = rows * 4 + rows * (2 * hidden + 2 * inter) * 2 + 4096;
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
    let mut again = Tensor::empty(&p.hip_mem, &[tokens, hidden], DType::BF16).expect("out");
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

/// The default 128-token page runs paged attention on CK `fmha_fwd_pagedkv` (decode of grouped
/// heads on CK `fmha_fwd_splitkv`), a 16-token page on the Turbine kernel, all against the CPU
/// reference: ragged prefill batches of 1-, 17-, 512- and 2,048-token chunks after 0, 100 and
/// 1,000 cached tokens with three decode rows riding along, then a decode batch, at the
/// Llama-3.2-3B (24/8) and OLMoE (16/16) head shapes.
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
            let want_decode = if block_tokens == 128 && heads.0 > heads.1 {
                "ck_tile_fmha_splitkv"
            } else {
                want
            };
            assert_eq!(
                got, want_decode,
                "decode heads={heads:?} block_tokens={block_tokens}"
            );
        }
    }
}

/// Perf item #3: paged decode of grouped query heads at 128-token pages runs CK
/// `fmha_fwd_splitkv` (the query heads of a KV head merged into one tile) and matches the CPU
/// reference: one sequence of ~8k tokens, sequences of 1-129 tokens around the page edge, a
/// short and a long sequence together, a ragged batch of 16 around the served ~768 tokens, 64
/// sequences, and the GQA ratios 3 (Llama 3.2 3B), 4 (Llama 3.1 8B) and 7 (Qwen2.5 7B). Breaks
/// if the head merge maps a query head to the wrong KV head or row, or the combine loses a
/// key.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn paged_decode_splitkv_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(13);
    let ragged16: Vec<usize> = (0..16).map(|s| 740 + (37 * s) % 90).collect();
    let many: Vec<usize> = (0..64).map(|s| 1 + (97 * s) % 1500).collect();
    let batches: [&[usize]; 5] = [
        &[8000],
        &[1, 2, 127, 128, 129],
        &[1, 3000, 5],
        &ragged16,
        &many,
    ];
    for heads in [(Q_HEADS, KV_HEADS), (32, 8), (28, 4)] {
        for kv_lens in batches {
            let got = paged_case(
                &p,
                &mut rng,
                AttentionKind::DecodePaged,
                heads,
                128,
                &vec![1; kv_lens.len()],
                kv_lens,
            );
            assert_eq!(
                got, "ck_tile_fmha_splitkv",
                "heads={heads:?} kv_lens={kv_lens:?}"
            );
        }
    }
}

/// Phase 6a S-13 (plan Task 23): paged attention over FP8 e4m3 pages matches the CPU reference
/// (pages written as `e4m3(x / scale)`, read as `bf16(e4m3 · scale)`) for every registered FP8
/// implementation, each run alone as the registry binds it: prefill batches (ragged chunks after
/// cached tokens with decode rows riding along; 64 sequences whose staged pages need two CK
/// groups; one sequence too long to stage, which runs the Turbine kernel) and decode batches, at
/// 128- and 16-token pages, the Llama (24/8), OLMoE (16/16) and GQA-4 (32/8) head shapes, unit
/// and non-unit K/V scales; the append's page bytes are compared exactly. The library's own
/// choice is the staged CK prefill at 128-token pages, the Turbine FP8 kernel at 16, and the FP8
/// decode kernel. Breaks if a page is written or read with the wrong scale or byte, or a group,
/// staged table or head mapping is wrong.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn paged_fp8_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(23);
    // (new tokens, cached tokens before them) per sequence.
    let mixed: &[(usize, usize)] = &[(512, 1000), (17, 0), (1, 255), (300, 100), (1, 1)];
    let many: Vec<(usize, usize)> = (0..64).map(|s| (1, 1060 + (37 * s) % 60)).collect();
    let decode_lens = [1usize, 17, 128, 129, 1001];
    let ragged16: Vec<usize> = (0..16).map(|s| 740 + (37 * s) % 90).collect();
    let scales = [
        Pages::Fp8 {
            k_scale: 1.0,
            v_scale: 1.0,
        },
        Pages::Fp8 {
            k_scale: 0.07,
            v_scale: 0.11,
        },
    ];
    for heads in [(Q_HEADS, KV_HEADS), (16, 16), (32, 8)] {
        for block_tokens in [128usize, 16] {
            let cfg = |kind| AttentionConfig {
                kind,
                num_q_heads: heads.0 as u32,
                num_kv_heads: heads.1 as u32,
                head_dim: HEAD_DIM as u32,
                dtype: DType::F8E4M3,
                block_tokens: Some(block_tokens as u32),
                causal: true,
            };
            let hip = p.hip.attention().expect("hip attention");
            let prefill_default = hip.implementation(&cfg(AttentionKind::PrefillPaged));
            let want = if block_tokens == 128 {
                "ck_tile_fmha_pagedkv_fp8_staged"
            } else {
                "turbine_hip_fp8"
            };
            assert_eq!(
                prefill_default, want,
                "prefill heads={heads:?} {block_tokens}"
            );
            assert_eq!(
                hip.implementation(&cfg(AttentionKind::DecodePaged)),
                "turbine_hip_fp8_decode",
                "decode heads={heads:?} {block_tokens}"
            );
            for kind in [AttentionKind::PrefillPaged, AttentionKind::DecodePaged] {
                let spec = OpConfig::Attention(cfg(kind));
                let impls: Vec<ImplInfo> = p
                    .hip
                    .implementations(spec.op())
                    .into_iter()
                    .filter(|i| p.hip.implementation_supports(&spec, i.index, None))
                    .collect();
                assert!(
                    impls.iter().all(|i| i.name.contains("fp8")),
                    "only FP8 implementations take FP8 pages: {impls:?}"
                );
                assert!(!impls.is_empty(), "{kind:?}: {impls:?}");
                for info in &impls {
                    let b = bound(&p, &spec, info.index);
                    for pages in scales {
                        let run = |rng: &mut Rng, batch: &[(usize, usize)]| {
                            let q_lens: Vec<usize> = batch.iter().map(|&(q, _)| q).collect();
                            let kv_lens: Vec<usize> = batch.iter().map(|&(q, c)| q + c).collect();
                            let got = paged_case_pages(
                                &b,
                                rng,
                                kind,
                                heads,
                                block_tokens,
                                &q_lens,
                                &kv_lens,
                                pages,
                            );
                            assert_eq!(got, info.name);
                        };
                        match kind {
                            AttentionKind::PrefillPaged => {
                                run(&mut rng, mixed);
                                if heads == (Q_HEADS, KV_HEADS) {
                                    run(&mut rng, &many);
                                }
                            }
                            _ => {
                                for batch in [&decode_lens[..], &ragged16] {
                                    let batch: Vec<(usize, usize)> =
                                        batch.iter().map(|&k| (1, k - 1)).collect();
                                    run(&mut rng, &batch);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // One sequence whose staged pages exceed the staging bound (256 MiB: 513 Llama pages of
    // 128 tokens) runs the Turbine FP8 kernel inside the staged implementation.
    let spec = OpConfig::Attention(AttentionConfig {
        kind: AttentionKind::PrefillPaged,
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        dtype: DType::F8E4M3,
        block_tokens: Some(128),
        causal: true,
    });
    let staged = p
        .hip
        .implementations(spec.op())
        .into_iter()
        .find(|i| i.name == "ck_tile_fmha_pagedkv_fp8_staged")
        .expect("the staged implementation");
    let got = paged_case_pages(
        &bound(&p, &spec, staged.index),
        &mut rng,
        AttentionKind::PrefillPaged,
        (Q_HEADS, KV_HEADS),
        128,
        &[3, 1],
        &[66_000, 40],
        scales[1],
    );
    assert_eq!(got, "ck_tile_fmha_pagedkv_fp8_staged");
}

/// Review r13 P1: the staged FP8 prefill finishes its single-query rows with the FP8 decode
/// kernel, whose LDS holds at most 8 query heads per KV head. At a GQA group of 16 (32/2 heads)
/// the staged implementation does not serve, the library routes a prefill batch with a
/// single-query row to the Turbine FP8 kernel, and the output matches the CPU reference; the
/// decode kind likewise falls back from the FP8 decode kernel. Breaks if the staged path (or
/// the decode kernel) is chosen for a group it would read past its LDS for.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn paged_fp8_group16_avoids_the_decode_kernel() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(29);
    let heads = (32usize, 2usize);
    let pages = Pages::Fp8 {
        k_scale: 0.07,
        v_scale: 0.11,
    };
    let cfg = |kind| AttentionConfig {
        kind,
        num_q_heads: heads.0 as u32,
        num_kv_heads: heads.1 as u32,
        head_dim: HEAD_DIM as u32,
        dtype: DType::F8E4M3,
        block_tokens: Some(128),
        causal: true,
    };
    let hip = p.hip.attention().expect("hip attention");
    for kind in [AttentionKind::PrefillPaged, AttentionKind::DecodePaged] {
        let spec = OpConfig::Attention(cfg(kind));
        for info in p.hip.implementations(spec.op()) {
            if matches!(
                info.name.as_str(),
                "ck_tile_fmha_pagedkv_fp8_staged" | "turbine_hip_fp8_decode"
            ) {
                assert!(
                    !p.hip.implementation_supports(&spec, info.index, None),
                    "{} must not serve GQA group 16 ({kind:?})",
                    info.name
                );
            }
        }
        assert_eq!(
            hip.implementation(&cfg(kind)),
            "turbine_hip_fp8",
            "{kind:?}"
        );
    }
    // A prefill batch with single-query rows riding along.
    let got = paged_case_pages(
        &p,
        &mut rng,
        AttentionKind::PrefillPaged,
        heads,
        128,
        &[300, 1, 17, 1],
        &[400, 256, 17, 2],
        pages,
    );
    assert_eq!(got, "turbine_hip_fp8");
    let got = paged_case_pages(
        &p,
        &mut rng,
        AttentionKind::DecodePaged,
        heads,
        128,
        &[1, 1, 1],
        &[1, 129, 1001],
        pages,
    );
    assert_eq!(got, "turbine_hip_fp8");
}

/// FP8 KV paged decode timings (plan Task 23 provider evaluation): `attention_decode_paged`
/// (append + attend) of every implementation that takes BF16 pages, then every one that takes
/// FP8 pages, bound as the registry binds it, for Llama (24/8) and OLMoE (16/16) at 128-token
/// pages, batches 1, 16 and 64 at ~768 tokens and 16 at ~2,048 / ~8,192; pools cycled past the
/// MALL as in `decode_attention_timings`. Prints `decode_attention_fp8` lines with the KV
/// bandwidth of the pool's own bytes (2 per element for BF16, 1 for FP8).
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn decode_attention_timings_fp8() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(11);
    const ITERS: u32 = 50;
    const BLOCK_TOKENS: usize = 128;
    let cases: [(usize, usize); 5] = [(1, 768), (16, 768), (64, 768), (16, 2048), (16, 8192)];
    for model in BENCH_MODELS {
        for page_dtype in [DType::BF16, DType::F8E4M3] {
            let cfg = AttentionConfig {
                dtype: page_dtype,
                ..model.attention_cfg(BLOCK_TOKENS)
            };
            let spec = OpConfig::Attention(cfg);
            let impls: Vec<ImplInfo> = p
                .hip
                .implementations(OpKind::AttentionDecodePaged)
                .into_iter()
                .filter(|i| p.hip.implementation_supports(&spec, i.index, None))
                .collect();
            let es = page_dtype.size_bytes();
            let max_m = cases.iter().map(|&(m, _)| m).max().unwrap_or(1);
            let q = on_hip(
                &p,
                &[max_m, model.q_heads, HEAD_DIM],
                DType::BF16,
                &rng.normal(max_m * model.q_rows(), 1.0),
            );
            let kv_new = on_hip(
                &p,
                &[max_m, model.kv_heads, HEAD_DIM],
                DType::BF16,
                &rng.normal(max_m * model.kv_rows(), 1.0),
            );
            let out = zeros_on_hip(&p, &[max_m, model.q_heads, HEAD_DIM], DType::BF16);
            for &(m, ctx) in &cases {
                let kv_lens = bench_kv_lens(m, ctx);
                let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
                let max_blocks = max_kv.div_ceil(BLOCK_TOKENS);
                let num_blocks = m * max_blocks;
                let pool_elems = num_blocks * 2 * BLOCK_TOKENS * model.kv_rows();
                let copies = (512usize << 20).div_ceil(pool_elems * es).max(1);
                let raw = if page_dtype == DType::F8E4M3 {
                    fp8_history(&mut rng, pool_elems)
                } else {
                    encode(DType::BF16, &pattern(&mut rng, pool_elems, 1.0))
                };
                let pool_shape = [num_blocks, 2, BLOCK_TOKENS, model.kv_heads, HEAD_DIM];
                let pools: Vec<Tensor> = (0..copies)
                    .map(|_| raw_on_hip(&p, &pool_shape, page_dtype, &raw))
                    .collect();
                let table: Vec<f32> = shuffled(&mut rng, num_blocks)
                    .iter()
                    .map(|&b| b as f32)
                    .collect();
                let bt = on_hip(&p, &[m, max_blocks], DType::I32, &table);
                let indptr: Vec<f32> = (0..=m).map(|i| i as f32).collect();
                let ip = on_hip(&p, &[m + 1], DType::I32, &indptr);
                let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
                let kl = on_hip(&p, &[m], DType::I32, &lens);
                let kv_bytes = kv_lens.iter().sum::<usize>() * model.kv_rows() * 2 * es;
                for info in &impls {
                    let b = bound(&p, &spec, info.index);
                    let attn = b.hip.attention().expect("hip attention");
                    let mut next = 0usize;
                    let us = time_us(&b, ITERS, || {
                        next = (next + 1) % copies;
                        attn.execute_paged(&mut PagedAttentionContext {
                            cfg,
                            q: q.view().rows(0, m),
                            k_new: kv_new.view().rows(0, m),
                            v_new: kv_new.view().rows(0, m),
                            out: out.view().rows(0, m),
                            kv_layer: pools[next].view(),
                            block_table: bt.view(),
                            q_indptr: ip.view(),
                            kv_lens: kl.view(),
                            max_q_len: 1,
                            max_kv_len: max_kv as u32,
                            max_blocks_per_seq: max_blocks as u32,
                            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                            k_scale: 1.0,
                            v_scale: 1.0,
                        })
                        .expect("paged decode attention");
                    });
                    println!(
                        "decode_attention_fp8 {} heads={}/{} pages={} impl={} b={m} kv~{ctx}: \
                         {us:.1} us ({:.0} GB/s of KV)",
                        model.name,
                        model.q_heads,
                        model.kv_heads,
                        page_dtype.as_str(),
                        info.name,
                        kv_bytes as f64 / us / 1e3,
                    );
                }
            }
        }
    }
}

/// FP8 KV paged prefill timings (plan Task 23 provider evaluation): `attention_prefill_paged`
/// (append + attend) of every implementation taking BF16 pages, then FP8 pages, bound as the
/// registry binds it, at 128-token pages: 16 prompts of 512 new tokens after 0 and after 1,024
/// cached tokens, and one prompt of 2,048 new tokens, for Llama and OLMoE heads. Prints
/// `prefill_attention_fp8` lines.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn prefill_op_timings_fp8() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(12);
    const ITERS: u32 = 10;
    const BLOCK_TOKENS: usize = 128;
    // (sequences, new tokens, cached tokens) per case.
    let cases: [(usize, usize, usize); 3] = [(16, 512, 0), (16, 512, 1024), (1, 2048, 0)];
    for model in BENCH_MODELS {
        for page_dtype in [DType::BF16, DType::F8E4M3] {
            let cfg = AttentionConfig {
                kind: AttentionKind::PrefillPaged,
                dtype: page_dtype,
                ..model.attention_cfg(BLOCK_TOKENS)
            };
            let spec = OpConfig::Attention(cfg);
            let impls: Vec<ImplInfo> = p
                .hip
                .implementations(OpKind::AttentionPrefillPaged)
                .into_iter()
                .filter(|i| p.hip.implementation_supports(&spec, i.index, None))
                .collect();
            for &(m, new, cached) in &cases {
                let total_q = m * new;
                let kv = new + cached;
                let max_blocks = kv.div_ceil(BLOCK_TOKENS);
                let num_blocks = m * max_blocks;
                let pool_elems = num_blocks * 2 * BLOCK_TOKENS * model.kv_rows();
                let raw = if page_dtype == DType::F8E4M3 {
                    fp8_history(&mut rng, pool_elems)
                } else {
                    encode(DType::BF16, &pattern(&mut rng, pool_elems, 1.0))
                };
                let pool_shape = [num_blocks, 2, BLOCK_TOKENS, model.kv_heads, HEAD_DIM];
                let pool = raw_on_hip(&p, &pool_shape, page_dtype, &raw);
                let q = on_hip(
                    &p,
                    &[total_q, model.q_heads, HEAD_DIM],
                    DType::BF16,
                    &pattern(&mut rng, total_q * model.q_rows(), 1.0),
                );
                let kv_new = on_hip(
                    &p,
                    &[total_q, model.kv_heads, HEAD_DIM],
                    DType::BF16,
                    &pattern(&mut rng, total_q * model.kv_rows(), 1.0),
                );
                let out = zeros_on_hip(&p, &[total_q, model.q_heads, HEAD_DIM], DType::BF16);
                let table: Vec<f32> = shuffled(&mut rng, num_blocks)
                    .iter()
                    .map(|&b| b as f32)
                    .collect();
                let bt = on_hip(&p, &[m, max_blocks], DType::I32, &table);
                let indptr: Vec<f32> = (0..=m).map(|i| (i * new) as f32).collect();
                let ip = on_hip(&p, &[m + 1], DType::I32, &indptr);
                let kl = on_hip(&p, &[m], DType::I32, &vec![kv as f32; m]);
                for info in &impls {
                    let b = bound(&p, &spec, info.index);
                    let attn = b.hip.attention().expect("hip attention");
                    let us = time_us(&b, ITERS, || {
                        attn.execute_paged(&mut PagedAttentionContext {
                            cfg,
                            q: q.view(),
                            k_new: kv_new.view(),
                            v_new: kv_new.view(),
                            out: out.view(),
                            kv_layer: pool.view(),
                            block_table: bt.view(),
                            q_indptr: ip.view(),
                            kv_lens: kl.view(),
                            max_q_len: new as u32,
                            max_kv_len: kv as u32,
                            max_blocks_per_seq: max_blocks as u32,
                            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                            k_scale: 1.0,
                            v_scale: 1.0,
                        })
                        .expect("paged prefill attention");
                    });
                    println!(
                        "prefill_attention_fp8 {} heads={}/{} pages={} impl={} seqs={m} \
                         new={new} cached={cached}: {us:.1} us",
                        model.name,
                        model.q_heads,
                        model.kv_heads,
                        page_dtype.as_str(),
                        info.name,
                    );
                }
            }
        }
    }
}

/// `copy_blocks` over 3 layers of 8 sixteen-token Llama blocks; the third pair reads a
/// block the first pair wrote, so the copy order matters.
fn copy_blocks_case(p: &Pair, rng: &mut Rng) {
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
        p,
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

    copy_blocks_case(&p, &mut rng);

    // moe_route: OLMoE (64 experts, top-8, no renormalisation, BF16 router logits) on random
    // logits; exact ties (the set torch.topk keeps, not always the lower ids); logits that
    // differ in F32 but tie in BF16 (with and without the BF16 rounding); a renormalising
    // 8-expert top-2 router; a 256-expert top-4 router (the heap-select path).
    let olmoe = MoeRouteConfig {
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        renormalize: false,
        bf16_logits: true,
    };
    route_case(&p, olmoe, 37, &rng.normal(37 * MOE_EXPERTS, 2.0));
    let mut ties = vec![0f32; 3 * MOE_EXPERTS];
    ties[0] = 1.0;
    ties[1] = 1.0;
    ties[MOE_EXPERTS + 40] = 2.0;
    route_case(&p, olmoe, 3, &ties);
    // Few distinct levels plus sub-BF16 jitter: ties at the 8th place in most rows.
    let near_ties: Vec<f32> = rng
        .normal(64 * MOE_EXPERTS, 1.0)
        .iter()
        .map(|v| (v * 2.0).round() / 2.0 + v * 1e-4)
        .collect();
    for bf16_logits in [true, false] {
        let cfg = MoeRouteConfig {
            bf16_logits,
            ..olmoe
        };
        route_case(&p, cfg, 64, &near_ties);
    }
    let small = MoeRouteConfig {
        num_experts: 8,
        top_k: 2,
        renormalize: true,
        bf16_logits: false,
    };
    route_case(&p, small, 13, &rng.normal(13 * 8, 1.0));
    let wide = MoeRouteConfig {
        num_experts: 256,
        top_k: 4,
        renormalize: true,
        bf16_logits: true,
    };
    let wide_ties: Vec<f32> = rng
        .normal(9 * 256, 1.0)
        .iter()
        .map(|v| (v * 2.0).round() / 2.0)
        .collect();
    route_case(&p, wide, 9, &wide_ties);

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
/// rows see `moe_experts_grouped_matches_cpu`.
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
}

/// P2c (prefill): above 512 routed rows `moe_experts` at the OLMoE shapes runs the prefill WMMA
/// path (the grouped WMMA path in a library without it), also without host offsets, and matches
/// the CPU reference, bitwise identical across two runs: 65 tokens (520 rows, the first size past
/// small-m), 300 tokens through a caller workspace, 256 tokens all on experts 40..=47 (256 rows
/// each: four full row tiles, 56 experts without rows) and a shard of the experts at 200 tokens.
/// A shape whose hidden or intermediate size is not a multiple of 64 still takes the host-offset
/// (hipBLASLt) path and matches; hidden and inter 256 at 1,536 tokens take the prefill kernels'
/// wide down-projection tile.
/// Prints each case's mean time per call (`moe_experts_timing:`).
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn moe_experts_grouped_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(19);
    let w = expert_weights(&p, &mut rng);
    for (tokens, workspace) in [(65, false), (300, true)] {
        let logits = rng.normal(tokens * MOE_EXPERTS, 1.0);
        let case = ExpertsCase {
            tokens,
            logits: &logits,
            local: (0, MOE_EXPERTS),
            workspace,
            repeat: 20,
        };
        experts_case(&p, &mut rng, &w, &case);
    }
    let mut logits = rng.normal(256 * MOE_EXPERTS, 0.1);
    for row in logits.chunks_exact_mut(MOE_EXPERTS) {
        for v in &mut row[40..48] {
            *v += 10.0;
        }
    }
    let case = ExpertsCase {
        tokens: 256,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: false,
        repeat: 5,
    };
    experts_case(&p, &mut rng, &w, &case);
    let logits = rng.normal(200 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 200,
        logits: &logits,
        local: (16, 48),
        workspace: false,
        repeat: 1,
    };
    experts_case(&p, &mut rng, &w, &case);
    drop(w);

    // hidden 136 and inter 72 (multiples of 8, not of 64): the hipBLASLt path with host offsets.
    let w = expert_weights_of(&p, &mut rng, 136, 72);
    let logits = rng.normal(80 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 80,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: false,
        repeat: 1,
    };
    experts_case(&p, &mut rng, &w, &case);
    drop(w);

    // hidden and inter 256 at 1,536 tokens (12,288 routed rows, the average expert filling three
    // 64-row tiles): the prefill kernels' wide down-projection tile.
    let w = expert_weights_of(&p, &mut rng, 256, 256);
    let logits = rng.normal(1536 * MOE_EXPERTS, 1.0);
    let case = ExpertsCase {
        tokens: 1536,
        logits: &logits,
        local: (0, MOE_EXPERTS),
        workspace: false,
        repeat: 2,
    };
    experts_case(&p, &mut rng, &w, &case);
}

/// Distance of `u · total` from the nearest CDF boundary of the id-order draw at `temperature`
/// over `row`, as a fraction of the total (the cpu-reference arithmetic: f32 scaled logits,
/// f64 exponentials summed in id order).
fn cdf_boundary_distance(row: &[f32], temperature: f32, u: f32) -> f64 {
    let inv_t = 1.0 / temperature;
    let max = row
        .iter()
        .map(|&v| v * inv_t)
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = row
        .iter()
        .map(|&v| {
            let s = v * inv_t;
            if s.is_nan() {
                0.0
            } else {
                f64::from(s - max).exp()
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let target = f64::from(u) * total;
    let mut cum = 0.0;
    let mut best = f64::INFINITY;
    for w in weights {
        cum += w;
        best = best.min((cum - target).abs());
    }
    best / total
}

/// Distance of the nucleus draw's two targets from the nearest prefix boundary, as a fraction
/// of the mass they are taken of: `top_p · total` against the prefix sums of all ids in
/// descending order (where the cut falls), and `u · kept` against the kept prefix's sums
/// (where the draw falls), in the cpu-reference arithmetic.
fn nucleus_boundary_distance(row: &[f32], temperature: f32, top_p: f32, u: f32) -> f64 {
    let inv_t = 1.0 / temperature;
    let mut ids: Vec<usize> = (0..row.len()).collect();
    let key = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
    ids.sort_by(|&a, &b| key(row[b]).total_cmp(&key(row[a])).then(a.cmp(&b)));
    let max = row
        .iter()
        .map(|&v| v * inv_t)
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = ids
        .iter()
        .map(|&i| {
            let s = row[i] * inv_t;
            if s.is_nan() {
                0.0
            } else {
                f64::from(s - max).exp()
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let cut = f64::from(top_p) * total;
    let mut cum = 0.0;
    let mut keep = weights.len();
    let mut d_cut = f64::INFINITY;
    for (i, w) in weights.iter().enumerate() {
        cum += w;
        d_cut = d_cut.min((cum - cut).abs() / total);
        if cum >= cut && keep == weights.len() {
            keep = i + 1;
        }
    }
    let kept: f64 = weights[..keep].iter().sum();
    let target = f64::from(u) * kept;
    let mut cum = 0.0;
    let mut d_draw = f64::INFINITY;
    for w in &weights[..keep] {
        cum += w;
        d_draw = d_draw.min((cum - target).abs() / kept);
    }
    d_cut.min(d_draw)
}

/// One `logits_reduce` comparison: `rows` rows of `vocab` logits (row stride `vocab + 5`),
/// seeded normal values with planted ties, NaN and −∞ entries, an all-NaN and an all-−∞ row,
/// every fifth row quantized to steps of 0.25 (long runs of tied values), every seventh flat
/// (one value but the planted ones: the top bucket holds the whole row), mixed modes,
/// temperatures and `top_p` (1, 0.9, 0.5, 0.2, 0.95: nuclei inside the planted ties, inside
/// the sorted candidates and far past them).
fn logits_reduce_case(p: &Pair, rng: &mut Rng, rows: usize, vocab: usize, top_n: usize) {
    let stride = vocab + 5;
    let mut logits = rng.normal(rows * stride, 3.0);
    for r in 0..rows {
        let row = &mut logits[r * stride..r * stride + vocab];
        match r {
            0 => row.fill(f32::NAN),
            1 => row.fill(f32::NEG_INFINITY),
            _ => {
                if r % 5 == 4 {
                    for v in row.iter_mut() {
                        *v = (*v * 4.0).round() / 4.0;
                    }
                }
                // A flat row: every key in round 0's top bucket, far more than the kernel
                // gathers into shared memory (its radix-round path).
                if r % 7 == 3 {
                    row.fill(2.5);
                }
                // Ties at the top (a value repeated at spread-out ids), a NaN and a −∞.
                let peak = 14.0 + (r % 3) as f32;
                for i in 0..6 {
                    row[(i * 7919 + r * 131) % vocab] = peak;
                }
                row[(r * 977) % vocab] = f32::NAN;
                row[(r * 1543 + 11) % vocab] = f32::NEG_INFINITY;
            }
        }
    }
    let temperatures: Vec<f32> = (0..rows).map(|r| [0.7, 1.0, 1.3, 0.0][r % 4]).collect();
    let uniforms: Vec<f32> = (0..rows).map(|_| rng.unit() as f32).collect();
    let modes: Vec<f32> = (0..rows).map(|r| (r % 3 != 2) as i32 as f32).collect();
    let top_ps: Vec<f32> = (0..rows)
        .map(|r| [1.0, 0.9, 0.5, 0.2, 0.95][(r / 2) % 5])
        .collect();
    let cfg = LogitsReduceConfig {
        vocab: vocab as u32,
        top_n: top_n as u32,
    };
    let hip = p.hip.logits_reduce().expect("hip logits_reduce (ABI v2.1)");
    assert!(hip.supports(&cfg), "hip must support logits_reduce {cfg}");
    let impl_name = hip.implementation(&cfg);
    let cpu = p.cpu.logits_reduce().expect("cpu logits_reduce");
    let (l_hip, l_cpu) = twin(p, &[rows, stride], DType::F32, &logits);
    let (t_hip, t_cpu) = twin(p, &[rows], DType::F32, &temperatures);
    let (u_hip, u_cpu) = twin(p, &[rows], DType::F32, &uniforms);
    let (tp_hip, tp_cpu) = twin(p, &[rows], DType::F32, &top_ps);
    let (m_hip, m_cpu) = twin(p, &[rows], DType::I32, &modes);
    let outputs = |mem: &Arc<dyn DeviceMemory>| {
        (
            Tensor::empty(mem, &[rows, top_n], DType::I32).expect("alloc"),
            Tensor::empty(mem, &[rows, top_n], DType::F32).expect("alloc"),
            Tensor::empty(mem, &[rows], DType::F32).expect("alloc"),
            Tensor::empty(mem, &[rows], DType::I32).expect("alloc"),
            Tensor::empty(mem, &[rows], DType::F32).expect("alloc"),
        )
    };
    let o_hip = outputs(&p.hip_mem);
    let o_cpu = outputs(&p.cpu_mem);
    let run = |kernel: &dyn LogitsReduceKernel,
               l: &Tensor,
               t: &Tensor,
               (u, tp): (&Tensor, &Tensor),
               m: &Tensor,
               o: &(Tensor, Tensor, Tensor, Tensor, Tensor)| {
        let logits_view = TensorView {
            shape: [rows, vocab].into_iter().collect(),
            strides: [stride, 1].into_iter().collect(),
            ..l.view()
        };
        kernel
            .execute(&mut LogitsReduceContext {
                logits: logits_view,
                temperature: t.view(),
                uniform: u.view(),
                top_p: tp.view(),
                mode: m.view(),
                top_ids: o.0.view(),
                top_values: o.1.view(),
                lse: o.2.view(),
                sampled: o.3.view(),
                sampled_logit: o.4.view(),
                rows: rows as u32,
            })
            .expect("logits_reduce");
    };
    run(hip, &l_hip, &t_hip, (&u_hip, &tp_hip), &m_hip, &o_hip);
    run(cpu, &l_cpu, &t_cpu, (&u_cpu, &tp_cpu), &m_cpu, &o_cpu);
    let what = format!("logits_reduce {cfg} rows={rows}");
    let (ids_h, ids_c) = (read(&o_hip.0), read(&o_cpu.0));
    assert_eq!(ids_h, ids_c, "{what} ({impl_name}): top ids");
    let (vals_h, vals_c) = (read(&o_hip.1), read(&o_cpu.1));
    for (i, (h, c)) in vals_h.iter().zip(&vals_c).enumerate() {
        assert!(
            h.to_bits() == c.to_bits() || (h.is_nan() && c.is_nan()),
            "{what}: top value {i}: hip {h} vs cpu {c}"
        );
    }
    let (lse_h, lse_c) = (read(&o_hip.2), read(&o_cpu.2));
    let mut worst_lse = 0f32;
    for r in 0..rows {
        let (h, c) = (lse_h[r], lse_c[r]);
        if c.is_finite() {
            let rel = (h - c).abs() / c.abs().max(1.0);
            assert!(rel <= 1e-5, "{what}: row {r} lse hip {h} vs cpu {c}");
            worst_lse = worst_lse.max(rel);
        } else {
            assert_eq!(h, c, "{what}: row {r} lse");
        }
    }
    let (s_h, s_c) = (read(&o_hip.3), read(&o_cpu.3));
    let (sl_h, sl_c) = (read(&o_hip.4), read(&o_cpu.4));
    let mut boundary = 0;
    let mut nucleus_rows = 0;
    for r in 0..rows {
        if modes[r] == 0.0 {
            assert_eq!((s_h[r], s_c[r]), (-1.0, -1.0), "{what}: row {r} mode 0");
            assert!(sl_h[r].is_nan(), "{what}: row {r} mode 0 sampled_logit");
            continue;
        }
        let row = &logits[r * stride..r * stride + vocab];
        let id = s_h[r] as usize;
        assert!(
            sl_h[r].to_bits() == row[id].to_bits() || (sl_h[r].is_nan() && row[id].is_nan()),
            "{what}: row {r} sampled_logit"
        );
        let nucleus = temperatures[r] > 0.0 && top_ps[r] < 1.0 && row.iter().any(|v| v.is_finite());
        nucleus_rows += usize::from(nucleus);
        if s_h[r] != s_c[r] {
            let d = if nucleus {
                nucleus_boundary_distance(row, temperatures[r], top_ps[r], uniforms[r])
            } else {
                cdf_boundary_distance(row, temperatures[r], uniforms[r])
            };
            assert!(
                temperatures[r] > 0.0 && d <= 1e-6,
                "{what}: row {r} drew {} (cpu {}), {d:e} from a CDF boundary",
                s_h[r],
                s_c[r]
            );
            boundary += 1;
        } else {
            assert_eq!(sl_h[r].to_bits(), sl_c[r].to_bits(), "{what}: row {r}");
        }
    }
    println!(
        "{what}: impl={impl_name} top ids exact, max lse rel |Δ| {worst_lse:.3e}, \
         draws identical ({nucleus_rows} from a top_p nucleus) but {boundary} at a CDF \
         boundary ok"
    );
}

/// Lab only (P2c S-4/S-9): the HIP `logits_reduce` (ABI v2.1) matches the cpu reference over
/// 64 rows of the OLMoE (50,304) and Llama-3 (128,256) vocabularies: top-n ids identical, lse
/// within 1e-5 relative, categorical draws — id order and `top_p` nucleus — identical except
/// within 1e-6 of a CDF or nucleus boundary. Also prints the device time of a 16-row decode
/// batch's reduction (id-order draws, and nuclei at the Llama defaults T 0.6 / top_p 0.9 over
/// peaked and over broad rows) against the full-row copy it replaces.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn logits_reduce_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(21);
    for vocab in [50_304, VOCAB] {
        logits_reduce_case(&p, &mut rng, 64, vocab, 64);
        logits_reduce_case(&p, &mut rng, 7, vocab, 5);
        // One candidate (the served default without logprobs): the argmax shortcut.
        logits_reduce_case(&p, &mut rng, 16, vocab, 1);
    }

    // Timing at the Llama decode shape: 16 rows reduced to top-20 and a draw, versus copying
    // the 16 full rows to the host (what the device reduction removes).
    let rows = 16;
    let hip = p.hip.logits_reduce().expect("hip logits_reduce");
    let mut logits = Tensor::empty(&p.hip_mem, &[rows, VOCAB], DType::F32).expect("alloc");
    let data = rng.normal(rows * VOCAB, 3.0);
    logits
        .storage
        .copy_from_host(0, &encode(DType::F32, &data))
        .expect("upload");
    let (t, _) = twin(&p, &[rows], DType::F32, &[1.0; 16]);
    let (u, _) = twin(&p, &[rows], DType::F32, &[0.5; 16]);
    let (no_nucleus, _) = twin(&p, &[rows], DType::F32, &[1.0; 16]);
    let (m, _) = twin(&p, &[rows], DType::I32, &[1.0; 16]);
    let ids = Tensor::empty(&p.hip_mem, &[rows, 20], DType::I32).expect("alloc");
    let vals = Tensor::empty(&p.hip_mem, &[rows, 20], DType::F32).expect("alloc");
    let small: Vec<Tensor> = [DType::F32, DType::I32, DType::F32]
        .into_iter()
        .map(|d| Tensor::empty(&p.hip_mem, &[rows], d).expect("alloc"))
        .collect();
    let reduce_with = |logits: &Tensor, t: &Tensor, tp: &Tensor| {
        hip.execute(&mut LogitsReduceContext {
            logits: logits.view(),
            temperature: t.view(),
            uniform: u.view(),
            top_p: tp.view(),
            mode: m.view(),
            top_ids: ids.view(),
            top_values: vals.view(),
            lse: small[0].view(),
            sampled: small[1].view(),
            sampled_logit: small[2].view(),
            rows: rows as u32,
        })
        .expect("logits_reduce");
    };
    const ITERS: u32 = 50;
    let time = |logits: &Tensor, t: &Tensor, tp: &Tensor| {
        reduce_with(logits, t, tp);
        p.hip_mem.synchronize().expect("sync");
        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            reduce_with(logits, t, tp);
        }
        p.hip_mem.synchronize().expect("sync");
        started.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS)
    };
    let reduce_us = time(&logits, &t, &no_nucleus);
    // Nuclei at the Llama-3.2 generation defaults: over the broad rows (a nucleus of tens of
    // thousands of ids: the mass searches) and over peaked rows (a few confident ids over a
    // bulk far below: the nucleus is inside the sorted candidates).
    let (t_llama, _) = twin(&p, &[rows], DType::F32, &[0.6; 16]);
    let (tp_llama, _) = twin(&p, &[rows], DType::F32, &[0.9; 16]);
    let broad_us = time(&logits, &t_llama, &tp_llama);
    let mut peaked_data = data.clone();
    for (r, row) in peaked_data.chunks_exact_mut(VOCAB).enumerate() {
        for (i, v) in [(r * 11, 24.0), (r * 11 + 5000, 23.0), (r * 11 + 9000, 22.5)] {
            row[i] = v;
        }
    }
    let mut peaked = Tensor::empty(&p.hip_mem, &[rows, VOCAB], DType::F32).expect("alloc");
    peaked
        .storage
        .copy_from_host(0, &encode(DType::F32, &peaked_data))
        .expect("upload");
    let peaked_us = time(&peaked, &t_llama, &tp_llama);
    let started = std::time::Instant::now();
    for _ in 0..ITERS {
        let _ = read(&ids);
    }
    let small_us = started.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
    let started = std::time::Instant::now();
    for _ in 0..10 {
        let _ = logits.view().slice.read_bytes().expect("read");
    }
    let full_us = started.elapsed().as_secs_f64() * 1e6 / 10.0;
    println!(
        "logits_reduce timing: 16 x {VOCAB} rows reduce {reduce_us:.1} us (T 0.6 top_p 0.9 \
         nucleus: peaked rows {peaked_us:.1} us, broad rows {broad_us:.1} us), read of the \
         reduction {small_us:.1} us, full-row copy {full_us:.1} us ({} bytes)",
        rows * VOCAB * 4
    );
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

/// A HIP tensor holding the already encoded `raw` bytes.
fn raw_on_hip(p: &Pair, shape: &[usize], dtype: DType, raw: &[u8]) -> Tensor {
    let mut t = Tensor::empty(&p.hip_mem, shape, dtype).expect("HIP alloc");
    t.storage.copy_from_host(0, raw).expect("copy to HIP");
    t
}

/// A zeroed HIP tensor.
fn zeros_on_hip(p: &Pair, shape: &[usize], dtype: DType) -> Tensor {
    let n: usize = shape.iter().product();
    raw_on_hip(p, shape, dtype, &vec![0u8; n * dtype.size_bytes()])
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

// ------------------------------------------------------ decode timings (lab, P2c S-2)

/// The MLP of a benchmarked model.
#[derive(Clone, Copy)]
enum Mlp {
    Dense {
        inter: usize,
    },
    Moe {
        experts: usize,
        top_k: usize,
        inter: usize,
    },
}

/// The decode shapes of one benchmarked model: BF16, head_dim 128, `q_heads · 128 = hidden`.
#[derive(Clone, Copy)]
struct BenchModel {
    name: &'static str,
    layers: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    vocab: usize,
    rope_theta: f64,
    /// OLMoE normalises the full Q and K projections before RoPE.
    qk_norm: bool,
    mlp: Mlp,
}

/// Llama-3.2-3B-Instruct and OLMoE-1B-7B-0125-Instruct (their `config.json`).
const BENCH_MODELS: [BenchModel; 2] = [
    BenchModel {
        name: "llama-3.2-3b-instruct",
        layers: 28,
        hidden: HIDDEN,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        vocab: VOCAB,
        rope_theta: ROPE_THETA,
        qk_norm: false,
        mlp: Mlp::Dense {
            inter: INTERMEDIATE,
        },
    },
    BenchModel {
        name: "olmoe-1b-7b-0125-instruct",
        layers: 16,
        hidden: MOE_HIDDEN,
        q_heads: 16,
        kv_heads: 16,
        vocab: 50304,
        rope_theta: 10_000.0,
        qk_norm: true,
        mlp: Mlp::Moe {
            experts: MOE_EXPERTS,
            top_k: MOE_TOP_K,
            inter: MOE_INTER,
        },
    },
];

/// Decode batch sizes of the timing benchmarks.
const BENCH_BATCHES: [usize; 3] = [1, 16, 64];
/// Mean context of every benchmarked sequence (the new token included).
const BENCH_CTX: usize = 768;
/// The Phase 2c default KV page size.
const BENCH_BLOCK_TOKENS: usize = 128;

impl BenchModel {
    fn q_rows(&self) -> usize {
        self.q_heads * HEAD_DIM
    }

    fn kv_rows(&self) -> usize {
        self.kv_heads * HEAD_DIM
    }

    fn rope_cfg(&self) -> RopeConfig {
        RopeConfig {
            num_q_heads: self.q_heads as u32,
            num_kv_heads: self.kv_heads as u32,
            head_dim: HEAD_DIM as u32,
            rotary_dim: HEAD_DIM as u32,
            dtype: DType::BF16,
        }
    }

    fn attention_cfg(&self, block_tokens: usize) -> AttentionConfig {
        AttentionConfig {
            kind: AttentionKind::DecodePaged,
            num_q_heads: self.q_heads as u32,
            num_kv_heads: self.kv_heads as u32,
            head_dim: HEAD_DIM as u32,
            dtype: DType::BF16,
            block_tokens: Some(block_tokens as u32),
            causal: true,
        }
    }

    fn inv_freq(&self) -> Vec<f32> {
        (0..HEAD_DIM / 2)
            .map(|i| (1.0 / self.rope_theta.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
            .collect()
    }

    /// RMSNorm calls per forward: two per layer (four with Q/K norm, whose dimension equals
    /// `hidden` for OLMoE) plus the final norm.
    fn norm_calls(&self) -> usize {
        let per_layer = if self.qk_norm { 4 } else { 2 };
        per_layer * self.layers + 1
    }

    /// The GEMMs of one decode forward, grouped by shape: (names, n, k, output dtype, calls
    /// per forward).
    fn gemms(&self) -> Vec<(String, usize, usize, DType, usize)> {
        let (l, h) = (self.layers, self.hidden);
        let mut all = vec![
            ("q_proj", self.q_rows(), h, DType::BF16, l),
            ("k_proj", self.kv_rows(), h, DType::BF16, l),
            ("v_proj", self.kv_rows(), h, DType::BF16, l),
            ("o_proj", h, self.q_rows(), DType::BF16, l),
        ];
        match self.mlp {
            Mlp::Dense { inter } => all.extend([
                ("gate_proj", inter, h, DType::BF16, l),
                ("up_proj", inter, h, DType::BF16, l),
                ("down_proj", h, inter, DType::BF16, l),
            ]),
            Mlp::Moe { experts, .. } => all.push(("router", experts, h, DType::F32, l)),
        }
        all.push(("lm_head", self.vocab, h, DType::F32, 1));
        let mut grouped: Vec<(String, usize, usize, DType, usize)> = Vec::new();
        for (name, n, k, dtype, calls) in all {
            match grouped
                .iter_mut()
                .find(|g| (g.1, g.2, g.3) == (n, k, dtype))
            {
                Some(g) => {
                    g.0 = format!("{}/{name}", g.0);
                    g.4 += calls;
                }
                None => grouped.push((name.to_string(), n, k, dtype, calls)),
            }
        }
        grouped
    }
}

fn gemm_config(n: usize, k: usize, c_dtype: DType) -> GemmConfig {
    GemmConfig {
        n: n as u64,
        k: k as u64,
        trans_b: true,
        a_dtype: DType::BF16,
        b_dtype: DType::BF16,
        c_dtype,
    }
}

/// Context lengths of a benchmark batch: `m` sequences spread around `ctx` tokens.
fn bench_kv_lens(m: usize, ctx: usize) -> Vec<usize> {
    (0..m).map(|s| ctx - 28 + (5 * s) % 57).collect()
}

/// A JSON string literal.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Accumulates per-op timings into a decode-forward estimate.
#[derive(Default)]
struct Timings {
    per_forward_us: f64,
}

impl Timings {
    /// Prints one op's time per call and its share of a forward (`calls` per forward).
    fn report(&mut self, label: &str, name: &str, impl_name: &str, us: f64, calls: usize) {
        self.per_forward_us += us * calls as f64;
        println!(
            "timing {label} {name:<44} impl={impl_name:<24} {us:>9.1} us/call x{calls:>3} = {:>8.3} ms/forward",
            us * calls as f64 / 1e3
        );
    }
}

/// Lab microbenchmark (no assertion on speed): every op of one decode step of Llama-3.2-3B and
/// OLMoE-1B-7B, each timed alone over back-to-back calls, for batches of 1, 16 and 64
/// sequences with ~768 cached tokens each (128-token pages), per call and per forward (every
/// layer plus the embedding, the final norm and the LM head). GEMM weights are cycled through
/// enough copies that they never stay in the GPU caches, as in a forward where every layer has
/// its own. Paged decode attention also runs at 16-token pages (the Turbine kernel, x0: for
/// comparison) and, at batch 16, at 2,048- and 8,192-token contexts. Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_ops -- decode_op_timings --nocapture`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn decode_op_timings() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(5);
    const ITERS: u32 = 50;
    let max_m = BENCH_BATCHES[BENCH_BATCHES.len() - 1];
    for model in BENCH_MODELS {
        let mut t: Vec<Timings> = BENCH_BATCHES.iter().map(|_| Timings::default()).collect();
        let label = |m: usize| format!("{} b={m:<2}", model.name);
        let (h, l) = (model.hidden, model.layers);
        let widest = match model.mlp {
            Mlp::Dense { inter } => inter.max(h),
            Mlp::Moe { .. } => h,
        };
        let act = on_hip(
            &p,
            &[max_m, widest],
            DType::BF16,
            &pattern(&mut rng, max_m * widest, 1.0),
        );

        // GEMMs, one per distinct shape.
        let gemm = p.hip.gemm().expect("hip gemm");
        for (name, n, k, c_dtype, calls) in model.gemms() {
            let cfg = gemm_config(n, k, c_dtype);
            let bytes = n * k * 2;
            let raw = encode(
                DType::BF16,
                &pattern(&mut rng, n * k, 1.0 / (k as f32).sqrt()),
            );
            // Enough weight copies (cycled) that the weights never stay in the 64 MB L2/MALL.
            let copies = (512usize << 20).div_ceil(bytes).max(1);
            let ws: Vec<Tensor> = (0..copies)
                .map(|_| raw_on_hip(&p, &[n, k], DType::BF16, &raw))
                .collect();
            let c = zeros_on_hip(&p, &[max_m, n], c_dtype);
            for (bi, &m) in BENCH_BATCHES.iter().enumerate() {
                let a = TensorView::contiguous(act.storage.whole(), 0, &[m, k], DType::BF16);
                let mut next = 0usize;
                let us = time_us(&p, ITERS, || {
                    next = (next + 1) % copies;
                    gemm.execute(&mut GemmContext {
                        a: a.clone(),
                        b: ws[next].view(),
                        c: c.view().rows(0, m),
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("gemm");
                });
                t[bi].report(
                    &label(m),
                    &format!(
                        "gemm {name} n={n} k={k} ({:.0} GB/s)",
                        bytes as f64 / (us * 1e3)
                    ),
                    &gemm.implementation(&cfg),
                    us,
                    calls,
                );
            }
        }

        // RMSNorm, residual add, SiLU · up, RoPE and the embedding.
        let norm_cfg = NormConfig {
            dim: h as u64,
            dtype: DType::BF16,
        };
        let norm = p.hip.norm().expect("hip norm");
        let add = p.hip.elementwise().expect("hip add");
        let silu = p.hip.activation().expect("hip silu_mul");
        let rope = p.hip.rope().expect("hip rope");
        let emb = p.hip.embedding().expect("hip embedding");
        let x = on_hip(
            &p,
            &[max_m, h],
            DType::BF16,
            &pattern(&mut rng, max_m * h, 1.0),
        );
        let w = on_hip(&p, &[h], DType::BF16, &pattern(&mut rng, h, 1.0));
        let o = zeros_on_hip(&p, &[max_m, h], DType::BF16);
        let (q_rows, kv_rows) = (model.q_rows(), model.kv_rows());
        let q = on_hip(
            &p,
            &[max_m, model.q_heads, HEAD_DIM],
            DType::BF16,
            &pattern(&mut rng, max_m * q_rows, 1.0),
        );
        let k = on_hip(
            &p,
            &[max_m, model.kv_heads, HEAD_DIM],
            DType::BF16,
            &pattern(&mut rng, max_m * kv_rows, 1.0),
        );
        let v = on_hip(
            &p,
            &[max_m, model.kv_heads, HEAD_DIM],
            DType::BF16,
            &pattern(&mut rng, max_m * kv_rows, 1.0),
        );
        let freq = on_hip(&p, &[HEAD_DIM / 2], DType::F32, &model.inv_freq());
        let table = on_hip(
            &p,
            &[model.vocab, h],
            DType::BF16,
            &pattern(&mut rng, model.vocab * h, 1.0),
        );
        let ids: Vec<f32> = (0..max_m)
            .map(|i| ((i * 7919) % model.vocab) as f32)
            .collect();
        let ids = on_hip(&p, &[max_m], DType::I32, &ids);
        let dense = match model.mlp {
            Mlp::Dense { inter } => Some((
                inter,
                on_hip(
                    &p,
                    &[max_m, inter],
                    DType::BF16,
                    &pattern(&mut rng, max_m * inter, 1.0),
                ),
                on_hip(
                    &p,
                    &[max_m, inter],
                    DType::BF16,
                    &pattern(&mut rng, max_m * inter, 1.0),
                ),
            )),
            Mlp::Moe { .. } => None,
        };
        for (bi, &m) in BENCH_BATCHES.iter().enumerate() {
            let us = time_us(&p, ITERS, || {
                norm.execute(&mut NormContext {
                    x: x.view().rows(0, m),
                    weight: w.view(),
                    out: o.view().rows(0, m),
                    eps: 1e-5,
                })
                .expect("rmsnorm");
            });
            t[bi].report(
                &label(m),
                "rmsnorm",
                &norm.implementation(&norm_cfg),
                us,
                model.norm_calls(),
            );
            let us = time_us(&p, ITERS, || {
                add.execute(&mut ElementwiseContext {
                    a: x.view().rows(0, m),
                    b: o.view().rows(0, m),
                    out: x.view().rows(0, m),
                })
                .expect("add");
            });
            let add_cfg = ElementwiseConfig { dtype: DType::BF16 };
            t[bi].report(&label(m), "add", &add.implementation(&add_cfg), us, 2 * l);
            if let Some((inter, g, u)) = &dense {
                let cfg = ActivationConfig {
                    cols: *inter as u64,
                    dtype: DType::BF16,
                };
                let out = TensorView::contiguous(act.storage.whole(), 0, &[m, *inter], DType::BF16);
                let us = time_us(&p, ITERS, || {
                    silu.execute(&mut ActivationContext {
                        gate: g.view().rows(0, m),
                        up: u.view().rows(0, m),
                        out: out.clone(),
                    })
                    .expect("silu_mul");
                });
                t[bi].report(&label(m), "silu_mul", &silu.implementation(&cfg), us, l);
            }
            let positions: Vec<f32> = bench_kv_lens(m, BENCH_CTX)
                .iter()
                .map(|&kl| (kl - 1) as f32)
                .collect();
            let pos = on_hip(&p, &[m], DType::I32, &positions);
            let rope_cfg = model.rope_cfg();
            let us = time_us(&p, ITERS, || {
                rope.execute(&mut RopeContext {
                    cfg: rope_cfg,
                    q: q.view().rows(0, m),
                    k: k.view().rows(0, m),
                    positions: pos.view(),
                    inv_freq: freq.view(),
                    attn_factor: 1.0,
                })
                .expect("rope");
            });
            t[bi].report(&label(m), "rope", &rope.implementation(&rope_cfg), us, l);
            let emb_cfg = EmbeddingConfig {
                hidden: h as u64,
                vocab_rows: model.vocab as u64,
                dtype: DType::BF16,
            };
            let us = time_us(&p, ITERS, || {
                emb.execute(&mut EmbeddingContext {
                    ids: ids.view().rows(0, m),
                    table: table.view(),
                    out: x.view().rows(0, m),
                    vocab_offset: 0,
                })
                .expect("embedding");
            });
            t[bi].report(&label(m), "embedding", &emb.implementation(&emb_cfg), us, 1);
        }
        drop(table);

        // Paged decode attention (append + attend): 128-token pages (CK pagedkv, the default)
        // at ~768 tokens is the forward estimate; 16-token pages (the Turbine kernel) and the
        // longer contexts at batch 16 are for comparison.
        let attn = p.hip.attention().expect("hip attention");
        let attn_out = zeros_on_hip(&p, &[max_m, model.q_heads, HEAD_DIM], DType::BF16);
        for (bi, &m) in BENCH_BATCHES.iter().enumerate() {
            let mut cases = vec![(BENCH_BLOCK_TOKENS, BENCH_CTX, l), (16, BENCH_CTX, 0)];
            if m == 16 {
                cases.extend([(16, 2048, 0), (16, 8192, 0), (128, 2048, 0), (128, 8192, 0)]);
            }
            for (block_tokens, ctx, calls) in cases {
                let cfg = model.attention_cfg(block_tokens);
                let kv_lens = bench_kv_lens(m, ctx);
                let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
                let max_blocks = max_kv.div_ceil(block_tokens);
                let num_blocks = m * max_blocks;
                let table: Vec<f32> = shuffled(&mut rng, num_blocks)
                    .iter()
                    .map(|&b| b as f32)
                    .collect();
                let pool = on_hip(
                    &p,
                    &[num_blocks, 2, block_tokens, model.kv_heads, HEAD_DIM],
                    DType::BF16,
                    &pattern(&mut rng, num_blocks * 2 * block_tokens * kv_rows, 1.0),
                );
                let bt = on_hip(&p, &[m, max_blocks], DType::I32, &table);
                let indptr: Vec<f32> = (0..=m).map(|i| i as f32).collect();
                let ip = on_hip(&p, &[m + 1], DType::I32, &indptr);
                let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
                let kl = on_hip(&p, &[m], DType::I32, &lens);
                let us = time_us(&p, ITERS, || {
                    attn.execute_paged(&mut PagedAttentionContext {
                        cfg,
                        q: q.view().rows(0, m),
                        k_new: k.view().rows(0, m),
                        v_new: v.view().rows(0, m),
                        out: attn_out.view().rows(0, m),
                        kv_layer: pool.view(),
                        block_table: bt.view(),
                        q_indptr: ip.view(),
                        kv_lens: kl.view(),
                        max_q_len: 1,
                        max_kv_len: max_kv as u32,
                        max_blocks_per_seq: max_blocks as u32,
                        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                        k_scale: 1.0,
                        v_scale: 1.0,
                    })
                    .expect("paged decode attention");
                });
                t[bi].report(
                    &label(m),
                    &format!("attention_decode_paged bt={block_tokens} kv~{ctx}"),
                    &attn.implementation(&cfg),
                    us,
                    calls,
                );
            }
        }

        // The MoE block: routing, the expert-offset read-back `moe_experts` needs on the host,
        // the accumulator reset and the grouped expert GEMMs (one layer's stacked experts,
        // larger than the GPU caches).
        if let Mlp::Moe {
            experts,
            top_k,
            inter,
        } = model.mlp
        {
            let moe = p.hip.moe().expect("hip moe");
            let route_cfg = MoeRouteConfig {
                num_experts: experts as u32,
                top_k: top_k as u32,
                renormalize: false,
                bf16_logits: true,
            };
            let experts_cfg = MoeExpertsConfig {
                hidden: h as u32,
                inter: inter as u32,
                num_experts: experts as u32,
                top_k: top_k as u32,
                expert_begin: 0,
                expert_end: experts as u32,
                dtype: DType::BF16,
            };
            let up_scale = 1.0 / (h as f32).sqrt();
            let gate_raw = encode(
                DType::BF16,
                &pattern(&mut rng, experts * inter * h, up_scale),
            );
            let w_gate = raw_on_hip(&p, &[experts, inter, h], DType::BF16, &gate_raw);
            let w_up = raw_on_hip(&p, &[experts, inter, h], DType::BF16, &gate_raw);
            drop(gate_raw);
            let w_down = on_hip(
                &p,
                &[experts, h, inter],
                DType::BF16,
                &pattern(&mut rng, experts * inter * h, 1.0 / (inter as f32).sqrt()),
            );
            let logits = on_hip(
                &p,
                &[max_m, experts],
                DType::F32,
                &rng.normal(max_m * experts, 1.0),
            );
            let topk_ids = zeros_on_hip(&p, &[max_m, top_k], DType::I32);
            let topk_w = zeros_on_hip(&p, &[max_m, top_k], DType::F32);
            let sorted = zeros_on_hip(&p, &[max_m * top_k], DType::I32);
            let offsets = zeros_on_hip(&p, &[experts + 1], DType::I32);
            let acc_zeros = zeros_on_hip(&p, &[max_m, h], DType::BF16);
            let ws_bytes = max_m * top_k * ((2 * h + 3 * inter) * 2 + 8) + 5 * 256;
            let workspace = zeros_on_hip(&p, &[ws_bytes.div_ceil(2)], DType::BF16);
            for (bi, &m) in BENCH_BATCHES.iter().enumerate() {
                let sorted_rows =
                    TensorView::contiguous(sorted.storage.whole(), 0, &[m * top_k], DType::I32);
                let route = || {
                    moe.route(&mut MoeRouteContext {
                        cfg: route_cfg,
                        router_logits: logits.view().rows(0, m),
                        topk_ids: topk_ids.view().rows(0, m),
                        topk_weights: topk_w.view().rows(0, m),
                        sorted_rows: sorted_rows.clone(),
                        expert_offsets: offsets.view(),
                    })
                    .expect("moe_route");
                };
                let us = time_us(&p, ITERS, route);
                t[bi].report(
                    &label(m),
                    "moe_route",
                    &moe.implementation_route(&route_cfg),
                    us,
                    l,
                );
                let read_offsets = || -> Vec<i32> {
                    offsets
                        .storage
                        .whole()
                        .read_bytes()
                        .expect("read offsets")
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                };
                let us = time_us(&p, ITERS, || {
                    read_offsets();
                });
                t[bi].report(&label(m), "moe_offsets_read", "d2h", us, l);
                // Kernel ABI v2 has no device-to-device copy or memset: `0 + 0` resets it.
                let add_cfg = ElementwiseConfig { dtype: DType::BF16 };
                let us = time_us(&p, ITERS, || {
                    add.execute(&mut ElementwiseContext {
                        a: acc_zeros.view().rows(0, m),
                        b: acc_zeros.view().rows(0, m),
                        out: o.view().rows(0, m),
                    })
                    .expect("zero the accumulator");
                });
                t[bi].report(
                    &label(m),
                    "moe_zero (add)",
                    &add.implementation(&add_cfg),
                    us,
                    l,
                );
                let host_offsets = read_offsets();
                let ws_len = m * top_k * ((2 * h + 3 * inter) * 2 + 8) + 5 * 256;
                let us = time_us(&p, ITERS, || {
                    moe.experts(&mut MoeExpertsContext {
                        cfg: experts_cfg,
                        x: x.view().rows(0, m),
                        w_gate: w_gate.view(),
                        w_up: w_up.view(),
                        w_down: w_down.view(),
                        sorted_rows: sorted_rows.clone(),
                        expert_offsets: offsets.view(),
                        topk_weights: topk_w.view().rows(0, m),
                        host_expert_offsets: &host_offsets,
                        out: o.view().rows(0, m),
                        workspace: Some(workspace.storage.slice(0, ws_len)),
                    })
                    .expect("moe_experts");
                });
                let active = host_offsets.windows(2).filter(|w| w[1] > w[0]).count();
                t[bi].report(
                    &label(m),
                    &format!("moe_experts ({active} active experts)"),
                    &moe.implementation_experts(&experts_cfg),
                    us,
                    l,
                );
            }
        }
        for (bi, &m) in BENCH_BATCHES.iter().enumerate() {
            println!(
                "timing {} decode forward estimate (sum of the ops): {:.3} ms",
                label(m),
                t[bi].per_forward_us / 1e3
            );
        }
    }
}

/// The MLP weights of one synthetic decoder layer.
enum BenchMlp {
    Dense {
        w_gate: Tensor,
        w_up: Tensor,
        w_down: Tensor,
    },
    Moe {
        router: Tensor,
        /// `[experts, inter, hidden]`, `[experts, inter, hidden]`, `[experts, hidden, inter]`.
        w_gate: Tensor,
        w_up: Tensor,
        w_down: Tensor,
    },
}

/// One synthetic decoder layer: its own weights (its KV pool lives beside it).
struct BenchLayer {
    norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    mlp: BenchMlp,
}

/// One op of a profiled synthetic forward.
struct OpTiming {
    op: &'static str,
    name: &'static str,
    imp: String,
    calls: usize,
    total_us: f64,
}

/// Times each op of a synthetic forward from its launch until the stream drains (as the
/// executors' profile mode does) while `on`; runs it untimed otherwise.
struct OpClock<'a> {
    mem: &'a dyn DeviceMemory,
    on: bool,
    ops: Vec<OpTiming>,
}

impl OpClock<'_> {
    fn op<R>(
        &mut self,
        op: &'static str,
        name: &'static str,
        imp: &str,
        f: impl FnOnce() -> R,
    ) -> R {
        if !self.on {
            return f();
        }
        let start = std::time::Instant::now();
        let out = f();
        self.mem.synchronize().expect("synchronize");
        let us = start.elapsed().as_secs_f64() * 1e6;
        match self.ops.iter_mut().find(|o| o.op == op && o.name == name) {
            Some(o) => {
                o.calls += 1;
                o.total_us += us;
            }
            None => self.ops.push(OpTiming {
                op,
                name,
                imp: imp.to_string(),
                calls: 1,
                total_us: us,
            }),
        }
        out
    }
}

/// The implementation name of every op of a synthetic forward.
struct BenchImpls {
    gemm: [String; 4],
    router: String,
    lm_head: String,
    mlp: [String; 3],
    norm: String,
    add: String,
    silu: String,
    rope: String,
    embedding: String,
    route: String,
    experts: String,
}

/// Rows `[0, m)` of `t`.
fn rows(t: &Tensor, m: usize) -> TensorView<'_> {
    t.view().rows(0, m)
}

/// Rows `[0, m)` of a `[rows, n · 128]` activation as `[m, n, 128]`.
fn heads(t: &Tensor, m: usize, n: usize) -> TensorView<'_> {
    TensorView::contiguous(t.storage.whole(), 0, &[m, n, HEAD_DIM], DType::BF16)
}

/// Lab microbenchmark (no assertion on speed): one whole synthetic decode forward of
/// Llama-3.2-3B and of OLMoE-1B-7B (every layer with its own weights and KV pool, so nothing
/// stays in the GPU caches between layers) for batches of 1, 16 and 64 sequences with ~768
/// cached tokens each. At 128-token pages (the default) it prints one line per model and batch,
/// `op_timings: {"model","batch","block_tokens","forward_ms","host_enqueue_ms","ops":[{"op",
/// "name","impl","calls","us_per_call"}]}`: the forward time (mean of 5 unprofiled runs) and
/// each op's time per call and calls per forward, measured from its launch until the stream
/// drained (the executors' profile mode, so `perf forward_profile` compares like with like).
/// At 16-token pages it prints the forward time only. Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_ops -- decode_forward_timing --nocapture`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn decode_forward_timing() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(6);
    const RUNS: u32 = 5;
    const PROFILED_RUNS: usize = 3;
    let max_m = BENCH_BATCHES[BENCH_BATCHES.len() - 1];
    let gemm = p.hip.gemm().expect("hip gemm");
    let norm = p.hip.norm().expect("hip norm");
    let add = p.hip.elementwise().expect("hip add");
    let silu = p.hip.activation().expect("hip silu_mul");
    let rope = p.hip.rope().expect("hip rope");
    let emb = p.hip.embedding().expect("hip embedding");
    let attn = p.hip.attention().expect("hip attention");
    let moe = p.hip.moe().expect("hip moe");
    for model in BENCH_MODELS {
        let (h, layers) = (model.hidden, model.layers);
        let (q_rows, kv_rows) = (model.q_rows(), model.kv_rows());
        let weight = |rng: &mut Rng, n: usize, k: usize| {
            encode(DType::BF16, &pattern(rng, n * k, 1.0 / (k as f32).sqrt()))
        };
        let (w_q, w_kv, w_o) = (
            weight(&mut rng, q_rows, h),
            weight(&mut rng, kv_rows, h),
            weight(&mut rng, h, q_rows),
        );
        let norm_raw = encode(DType::BF16, &pattern(&mut rng, h, 1.0));
        let mlp_raw: [Vec<u8>; 2] = match model.mlp {
            Mlp::Dense { inter } => [weight(&mut rng, inter, h), weight(&mut rng, h, inter)],
            Mlp::Moe { experts, inter, .. } => [
                weight(&mut rng, experts * inter, h),
                encode(
                    DType::BF16,
                    &pattern(&mut rng, experts * h * inter, 1.0 / (inter as f32).sqrt()),
                ),
            ],
        };
        let router_raw = match model.mlp {
            Mlp::Moe { experts, .. } => weight(&mut rng, experts, h),
            Mlp::Dense { .. } => Vec::new(),
        };
        let layer_weights: Vec<BenchLayer> = (0..layers)
            .map(|_| BenchLayer {
                norm: raw_on_hip(&p, &[h], DType::BF16, &norm_raw),
                wq: raw_on_hip(&p, &[q_rows, h], DType::BF16, &w_q),
                wk: raw_on_hip(&p, &[kv_rows, h], DType::BF16, &w_kv),
                wv: raw_on_hip(&p, &[kv_rows, h], DType::BF16, &w_kv),
                wo: raw_on_hip(&p, &[h, q_rows], DType::BF16, &w_o),
                mlp: match model.mlp {
                    Mlp::Dense { inter } => BenchMlp::Dense {
                        w_gate: raw_on_hip(&p, &[inter, h], DType::BF16, &mlp_raw[0]),
                        w_up: raw_on_hip(&p, &[inter, h], DType::BF16, &mlp_raw[0]),
                        w_down: raw_on_hip(&p, &[h, inter], DType::BF16, &mlp_raw[1]),
                    },
                    Mlp::Moe { experts, inter, .. } => BenchMlp::Moe {
                        router: raw_on_hip(&p, &[experts, h], DType::BF16, &router_raw),
                        w_gate: raw_on_hip(&p, &[experts, inter, h], DType::BF16, &mlp_raw[0]),
                        w_up: raw_on_hip(&p, &[experts, inter, h], DType::BF16, &mlp_raw[0]),
                        w_down: raw_on_hip(&p, &[experts, h, inter], DType::BF16, &mlp_raw[1]),
                    },
                },
            })
            .collect();
        drop(mlp_raw);
        let table_raw = encode(DType::BF16, &pattern(&mut rng, model.vocab * h, 1.0));
        let embed = raw_on_hip(&p, &[model.vocab, h], DType::BF16, &table_raw);
        let lm_head = raw_on_hip(&p, &[model.vocab, h], DType::BF16, &table_raw);
        drop(table_raw);
        let final_norm = raw_on_hip(&p, &[h], DType::BF16, &norm_raw);

        // Activations and routing buffers for the largest batch.
        let zeros = |cols: usize| zeros_on_hip(&p, &[max_m, cols], DType::BF16);
        let (x, hn, proj, acc_zeros) = (zeros(h), zeros(h), zeros(h), zeros(h));
        let (q, q_raw, attn_out) = (zeros(q_rows), zeros(q_rows), zeros(q_rows));
        let (k, k_raw, v) = (zeros(kv_rows), zeros(kv_rows), zeros(kv_rows));
        let dense_inter = match model.mlp {
            Mlp::Dense { inter } => inter,
            Mlp::Moe { .. } => 1,
        };
        let (gate, up, act) = (zeros(dense_inter), zeros(dense_inter), zeros(dense_inter));
        let (experts, top_k, moe_inter) = match model.mlp {
            Mlp::Moe {
                experts,
                top_k,
                inter,
            } => (experts, top_k, inter),
            Mlp::Dense { .. } => (1, 1, 1),
        };
        let router_logits = zeros_on_hip(&p, &[max_m, experts], DType::F32);
        let topk_ids = zeros_on_hip(&p, &[max_m, top_k], DType::I32);
        let topk_w = zeros_on_hip(&p, &[max_m, top_k], DType::F32);
        let sorted = zeros_on_hip(&p, &[max_m * top_k], DType::I32);
        let offsets = zeros_on_hip(&p, &[experts + 1], DType::I32);
        let moe_ws_len = |m: usize| m * top_k * ((2 * h + 3 * moe_inter) * 2 + 8) + 5 * 256;
        let workspace = zeros_on_hip(&p, &[moe_ws_len(max_m).div_ceil(2)], DType::BF16);
        let logits = zeros_on_hip(&p, &[max_m, model.vocab], DType::F32);
        let inv_freq = on_hip(&p, &[HEAD_DIM / 2], DType::F32, &model.inv_freq());

        let kv_lens = bench_kv_lens(max_m, BENCH_CTX);
        let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
        let ids: Vec<f32> = (0..max_m)
            .map(|i| ((i * 7919) % model.vocab) as f32)
            .collect();
        let ids = on_hip(&p, &[max_m], DType::I32, &ids);
        let positions: Vec<f32> = kv_lens.iter().map(|&k| (k - 1) as f32).collect();
        let positions = on_hip(&p, &[max_m], DType::I32, &positions);
        let indptr: Vec<f32> = (0..=max_m).map(|i| i as f32).collect();
        let q_indptr = on_hip(&p, &[max_m + 1], DType::I32, &indptr);
        let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
        let kv_lens_t = on_hip(&p, &[max_m], DType::I32, &lens);

        let rope_cfg = model.rope_cfg();
        let norm_cfg = NormConfig {
            dim: h as u64,
            dtype: DType::BF16,
        };
        let route_cfg = MoeRouteConfig {
            num_experts: experts as u32,
            top_k: top_k as u32,
            renormalize: false,
            bf16_logits: true,
        };
        let experts_cfg = MoeExpertsConfig {
            hidden: h as u32,
            inter: moe_inter as u32,
            num_experts: experts as u32,
            top_k: top_k as u32,
            expert_begin: 0,
            expert_end: experts as u32,
            dtype: DType::BF16,
        };
        let gi = |n: usize, k: usize, c: DType| gemm.implementation(&gemm_config(n, k, c));
        let impls = BenchImpls {
            gemm: [
                gi(q_rows, h, DType::BF16),
                gi(kv_rows, h, DType::BF16),
                gi(kv_rows, h, DType::BF16),
                gi(h, q_rows, DType::BF16),
            ],
            router: gi(experts, h, DType::F32),
            lm_head: gi(model.vocab, h, DType::F32),
            mlp: [
                gi(dense_inter, h, DType::BF16),
                gi(dense_inter, h, DType::BF16),
                gi(h, dense_inter, DType::BF16),
            ],
            norm: norm.implementation(&norm_cfg),
            add: add.implementation(&ElementwiseConfig { dtype: DType::BF16 }),
            silu: silu.implementation(&ActivationConfig {
                cols: dense_inter as u64,
                dtype: DType::BF16,
            }),
            rope: rope.implementation(&rope_cfg),
            embedding: emb.implementation(&EmbeddingConfig {
                hidden: h as u64,
                vocab_rows: model.vocab as u64,
                dtype: DType::BF16,
            }),
            route: moe.implementation_route(&route_cfg),
            experts: moe.implementation_experts(&experts_cfg),
        };

        for block_tokens in [BENCH_BLOCK_TOKENS, 16] {
            let max_blocks = max_kv.div_ceil(block_tokens);
            let num_blocks = max_m * max_blocks;
            let table: Vec<f32> = shuffled(&mut rng, num_blocks)
                .iter()
                .map(|&b| b as f32)
                .collect();
            let block_table = on_hip(&p, &[max_m, max_blocks], DType::I32, &table);
            let pool_shape = [num_blocks, 2, block_tokens, model.kv_heads, HEAD_DIM];
            let pool_raw = encode(
                DType::BF16,
                &pattern(&mut rng, num_blocks * 2 * block_tokens * kv_rows, 1.0),
            );
            let pools: Vec<Tensor> = (0..layers)
                .map(|_| raw_on_hip(&p, &pool_shape, DType::BF16, &pool_raw))
                .collect();
            drop(pool_raw);
            let attn_cfg = model.attention_cfg(block_tokens);
            let attn_impl = attn.implementation(&attn_cfg);
            let ip = &impls;
            let forward = |m: usize, clock: &mut OpClock<'_>| {
                let linear = |a: TensorView<'_>, w: &Tensor, c: TensorView<'_>| {
                    gemm.execute(&mut GemmContext {
                        a,
                        b: w.view(),
                        c,
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("gemm");
                };
                let rmsnorm = |x: TensorView<'_>, w: &Tensor, out: TensorView<'_>| {
                    norm.execute(&mut NormContext {
                        x,
                        weight: w.view(),
                        out,
                        eps: 1e-5,
                    })
                    .expect("rmsnorm");
                };
                let residual = || {
                    add.execute(&mut ElementwiseContext {
                        a: rows(&x, m),
                        b: rows(&proj, m),
                        out: rows(&x, m),
                    })
                    .expect("add");
                };
                clock.op("embedding", "embedding", &ip.embedding, || {
                    emb.execute(&mut EmbeddingContext {
                        ids: rows(&ids, m),
                        table: embed.view(),
                        out: rows(&x, m),
                        vocab_offset: 0,
                    })
                    .expect("embedding")
                });
                for (l, pool) in layer_weights.iter().zip(&pools) {
                    clock.op("rmsnorm", "input_norm", &ip.norm, || {
                        rmsnorm(rows(&x, m), &l.norm, rows(&hn, m))
                    });
                    // OLMoE projects Q and K into raw buffers and normalises them into q, k.
                    let (q_dst, k_dst) = if model.qk_norm {
                        (&q_raw, &k_raw)
                    } else {
                        (&q, &k)
                    };
                    clock.op("gemm", "q_proj", &ip.gemm[0], || {
                        linear(rows(&hn, m), &l.wq, rows(q_dst, m))
                    });
                    clock.op("gemm", "k_proj", &ip.gemm[1], || {
                        linear(rows(&hn, m), &l.wk, rows(k_dst, m))
                    });
                    clock.op("gemm", "v_proj", &ip.gemm[2], || {
                        linear(rows(&hn, m), &l.wv, rows(&v, m))
                    });
                    if model.qk_norm {
                        clock.op("rmsnorm", "q_norm", &ip.norm, || {
                            rmsnorm(rows(&q_raw, m), &l.norm, rows(&q, m))
                        });
                        clock.op("rmsnorm", "k_norm", &ip.norm, || {
                            rmsnorm(rows(&k_raw, m), &l.norm, rows(&k, m))
                        });
                    }
                    clock.op("rope", "rope", &ip.rope, || {
                        rope.execute(&mut RopeContext {
                            cfg: rope_cfg,
                            q: heads(&q, m, model.q_heads),
                            k: heads(&k, m, model.kv_heads),
                            positions: rows(&positions, m),
                            inv_freq: inv_freq.view(),
                            attn_factor: 1.0,
                        })
                        .expect("rope")
                    });
                    clock.op("attention_decode_paged", "attention", &attn_impl, || {
                        attn.execute_paged(&mut PagedAttentionContext {
                            cfg: attn_cfg,
                            q: heads(&q, m, model.q_heads),
                            k_new: heads(&k, m, model.kv_heads),
                            v_new: heads(&v, m, model.kv_heads),
                            out: heads(&attn_out, m, model.q_heads),
                            kv_layer: pool.view(),
                            block_table: rows(&block_table, m),
                            q_indptr: q_indptr.view().rows(0, m + 1),
                            kv_lens: rows(&kv_lens_t, m),
                            max_q_len: 1,
                            max_kv_len: max_kv as u32,
                            max_blocks_per_seq: max_blocks as u32,
                            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                            k_scale: 1.0,
                            v_scale: 1.0,
                        })
                        .expect("paged decode attention")
                    });
                    clock.op("gemm", "o_proj", &ip.gemm[3], || {
                        linear(rows(&attn_out, m), &l.wo, rows(&proj, m))
                    });
                    clock.op("add", "residual", &ip.add, residual);
                    clock.op("rmsnorm", "post_norm", &ip.norm, || {
                        rmsnorm(rows(&x, m), &l.norm, rows(&hn, m))
                    });
                    match &l.mlp {
                        BenchMlp::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        } => {
                            clock.op("gemm", "gate_proj", &ip.mlp[0], || {
                                linear(rows(&hn, m), w_gate, rows(&gate, m))
                            });
                            clock.op("gemm", "up_proj", &ip.mlp[1], || {
                                linear(rows(&hn, m), w_up, rows(&up, m))
                            });
                            clock.op("silu_mul", "silu_mul", &ip.silu, || {
                                silu.execute(&mut ActivationContext {
                                    gate: rows(&gate, m),
                                    up: rows(&up, m),
                                    out: rows(&act, m),
                                })
                                .expect("silu_mul")
                            });
                            clock.op("gemm", "down_proj", &ip.mlp[2], || {
                                linear(rows(&act, m), w_down, rows(&proj, m))
                            });
                        }
                        BenchMlp::Moe {
                            router,
                            w_gate,
                            w_up,
                            w_down,
                        } => {
                            clock.op("gemm", "router", &ip.router, || {
                                linear(rows(&hn, m), router, rows(&router_logits, m))
                            });
                            let sorted_rows = TensorView::contiguous(
                                sorted.storage.whole(),
                                0,
                                &[m * top_k],
                                DType::I32,
                            );
                            clock.op("moe_route", "moe_route", &ip.route, || {
                                moe.route(&mut MoeRouteContext {
                                    cfg: route_cfg,
                                    router_logits: rows(&router_logits, m),
                                    topk_ids: rows(&topk_ids, m),
                                    topk_weights: rows(&topk_w, m),
                                    sorted_rows: sorted_rows.clone(),
                                    expert_offsets: offsets.view(),
                                })
                                .expect("moe_route")
                            });
                            let host_offsets: Vec<i32> =
                                clock.op("moe_offsets_read", "moe_offsets_read", "d2h", || {
                                    offsets
                                        .storage
                                        .whole()
                                        .read_bytes()
                                        .expect("read the expert offsets")
                                        .chunks_exact(4)
                                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                        .collect()
                                });
                            // `0 + 0`: kernel ABI v2 has no device-to-device copy or memset.
                            clock.op("add", "moe_zero", &ip.add, || {
                                add.execute(&mut ElementwiseContext {
                                    a: rows(&acc_zeros, m),
                                    b: rows(&acc_zeros, m),
                                    out: rows(&proj, m),
                                })
                                .expect("zero the accumulator")
                            });
                            clock.op("moe_experts", "moe_experts", &ip.experts, || {
                                moe.experts(&mut MoeExpertsContext {
                                    cfg: experts_cfg,
                                    x: rows(&hn, m),
                                    w_gate: w_gate.view(),
                                    w_up: w_up.view(),
                                    w_down: w_down.view(),
                                    sorted_rows: sorted_rows.clone(),
                                    expert_offsets: offsets.view(),
                                    topk_weights: rows(&topk_w, m),
                                    host_expert_offsets: &host_offsets,
                                    out: rows(&proj, m),
                                    workspace: Some(workspace.storage.slice(0, moe_ws_len(m))),
                                })
                                .expect("moe_experts")
                            });
                        }
                    }
                    clock.op("add", "residual", &ip.add, residual);
                }
                clock.op("rmsnorm", "final_norm", &ip.norm, || {
                    rmsnorm(rows(&x, m), &final_norm, rows(&hn, m))
                });
                clock.op("gemm", "lm_head", &ip.lm_head, || {
                    linear(rows(&hn, m), &lm_head, rows(&logits, m))
                });
            };

            for &m in &BENCH_BATCHES {
                let mut clock = OpClock {
                    mem: p.hip_mem.as_ref(),
                    on: false,
                    ops: Vec::new(),
                };
                forward(m, &mut clock);
                p.hip_mem.synchronize().expect("synchronize");
                let (mut host, mut total) = (0.0f64, 0.0f64);
                for _ in 0..RUNS {
                    let start = std::time::Instant::now();
                    forward(m, &mut clock);
                    host += start.elapsed().as_secs_f64();
                    p.hip_mem.synchronize().expect("synchronize");
                    total += start.elapsed().as_secs_f64();
                }
                let forward_ms = total * 1e3 / f64::from(RUNS);
                let host_ms = host * 1e3 / f64::from(RUNS);
                println!(
                    "timing decode forward {} b={m} kv~{BENCH_CTX} block_tokens={block_tokens} attention={}: \
                     {forward_ms:.3} ms (host enqueue {host_ms:.3} ms)",
                    model.name, attn_impl
                );
                if block_tokens != BENCH_BLOCK_TOKENS {
                    continue;
                }
                clock.on = true;
                for _ in 0..PROFILED_RUNS {
                    forward(m, &mut clock);
                }
                let ops: Vec<String> = clock
                    .ops
                    .iter()
                    .map(|o| {
                        format!(
                            "{{\"op\":{},\"name\":{},\"impl\":{},\"calls\":{},\"us_per_call\":{:.1}}}",
                            json_str(o.op),
                            json_str(o.name),
                            json_str(&o.imp),
                            o.calls / PROFILED_RUNS,
                            o.total_us / o.calls as f64
                        )
                    })
                    .collect();
                let profiled_ms: f64 =
                    clock.ops.iter().map(|o| o.total_us).sum::<f64>() / 1e3 / PROFILED_RUNS as f64;
                println!(
                    "op_timings: {{\"model\":{},\"batch\":{m},\"block_tokens\":{block_tokens},\"forward_ms\":{forward_ms:.3},\"host_enqueue_ms\":{host_ms:.3},\"profiled_ms\":{profiled_ms:.3},\"ops\":[{}]}}",
                    json_str(model.name),
                    ops.join(",")
                );
            }
        }
    }
}

/// `m` rows of `inner` elements of `t`'s storage, starting `col` elements into each row of
/// `row_stride` elements: a column block of a fused projection output, as the executors view it.
fn column_block<'a>(
    t: &'a Tensor,
    col: usize,
    row_stride: usize,
    m: usize,
    inner: &[usize],
) -> TensorView<'a> {
    let es = t.dtype.size_bytes();
    let row: usize = inner.iter().product();
    let mut shape = vec![m];
    shape.extend_from_slice(inner);
    let mut strides = vec![row_stride];
    let mut s = row;
    for &d in inner {
        s /= d;
        strides.push(s);
    }
    TensorView {
        slice: t
            .storage
            .whole()
            .sub(col * es, ((m - 1) * row_stride + row) * es),
        shape: shape.as_slice().into(),
        strides: strides.as_slice().into(),
        dtype: t.dtype,
    }
}

/// Lab microbenchmark (no assertion on speed; perf item #3): every enumerated implementation of
/// `attention_decode_paged` (append + attend), bound through `turbine_impl_run` as the kernel
/// registry binds it, for both served models' heads (Llama 24/8, OLMoE 16/16) at 128-token pages:
/// batches 1, 4, 8, 16 and 64 at a ~768-token context, batch 16 at ~2,048 and ~8,192, and the
/// small batches where splitting the context matters (1 at ~8,192, 4 at ~2,048). The KV pool is
/// cycled through enough copies (≥ 512 MB) that it never stays in the 64 MB L2/MALL, as in a
/// served step where every layer has its own pool (the single pool of `decode_op_timings` fits the
/// cache at batch 16 and reads faster than served). Wall time per call, host launches included.
/// Prints `decode_attention` lines with the achieved KV bandwidth. Run natively on GPU 0 with
/// `cargo test --release -p turbine-kernels --test hip_ops -- --ignored decode_attention_timings --nocapture`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn decode_attention_timings() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(11);
    const ITERS: u32 = 50;
    const BLOCK_TOKENS: usize = 128;
    let cases: [(usize, usize); 9] = [
        (1, 768),
        (4, 768),
        (8, 768),
        (16, 768),
        (64, 768),
        (1, 8192),
        (4, 2048),
        (16, 2048),
        (16, 8192),
    ];
    for model in BENCH_MODELS {
        let cfg = model.attention_cfg(BLOCK_TOKENS);
        let spec = OpConfig::Attention(cfg);
        let impls: Vec<ImplInfo> = p
            .hip
            .implementations(OpKind::AttentionDecodePaged)
            .into_iter()
            .filter(|i| p.hip.implementation_supports(&spec, i.index, None))
            .collect();
        let max_m = cases.iter().map(|&(m, _)| m).max().unwrap_or(1);
        let q = on_hip(
            &p,
            &[max_m, model.q_heads, HEAD_DIM],
            DType::BF16,
            &rng.normal(max_m * model.q_rows(), 1.0),
        );
        let kv_new = on_hip(
            &p,
            &[max_m, model.kv_heads, HEAD_DIM],
            DType::BF16,
            &rng.normal(max_m * model.kv_rows(), 1.0),
        );
        let out = zeros_on_hip(&p, &[max_m, model.q_heads, HEAD_DIM], DType::BF16);
        for &(m, ctx) in &cases {
            let kv_lens = bench_kv_lens(m, ctx);
            let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
            let max_blocks = max_kv.div_ceil(BLOCK_TOKENS);
            let num_blocks = m * max_blocks;
            let pool_elems = num_blocks * 2 * BLOCK_TOKENS * model.kv_rows();
            let copies = (512usize << 20).div_ceil(pool_elems * 2).max(1);
            let raw = encode(DType::BF16, &pattern(&mut rng, pool_elems, 1.0));
            let pools: Vec<Tensor> = (0..copies)
                .map(|_| {
                    raw_on_hip(
                        &p,
                        &[num_blocks, 2, BLOCK_TOKENS, model.kv_heads, HEAD_DIM],
                        DType::BF16,
                        &raw,
                    )
                })
                .collect();
            let table: Vec<f32> = shuffled(&mut rng, num_blocks)
                .iter()
                .map(|&b| b as f32)
                .collect();
            let bt = on_hip(&p, &[m, max_blocks], DType::I32, &table);
            let indptr: Vec<f32> = (0..=m).map(|i| i as f32).collect();
            let ip = on_hip(&p, &[m + 1], DType::I32, &indptr);
            let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
            let kl = on_hip(&p, &[m], DType::I32, &lens);
            // K and V bytes the attention reads (the appended token included).
            let kv_bytes = kv_lens.iter().sum::<usize>() * model.kv_rows() * 2 * 2;
            for info in &impls {
                let b = bound(&p, &spec, info.index);
                let attn = b.hip.attention().expect("hip attention");
                let mut next = 0usize;
                let us = time_us(&b, ITERS, || {
                    next = (next + 1) % copies;
                    attn.execute_paged(&mut PagedAttentionContext {
                        cfg,
                        q: q.view().rows(0, m),
                        k_new: kv_new.view().rows(0, m),
                        v_new: kv_new.view().rows(0, m),
                        out: out.view().rows(0, m),
                        kv_layer: pools[next].view(),
                        block_table: bt.view(),
                        q_indptr: ip.view(),
                        kv_lens: kl.view(),
                        max_q_len: 1,
                        max_kv_len: max_kv as u32,
                        max_blocks_per_seq: max_blocks as u32,
                        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                        k_scale: 1.0,
                        v_scale: 1.0,
                    })
                    .expect("paged decode attention");
                });
                println!(
                    "decode_attention {} heads={}/{} impl={} b={m} kv~{ctx}: {us:.1} us \
                     ({:.0} GB/s of KV)",
                    model.name,
                    model.q_heads,
                    model.kv_heads,
                    info.name,
                    kv_bytes as f64 / us / 1e3,
                );
            }
        }
    }
}

/// Lab microbenchmark (no assertion on speed, P2c Task 10): Llama-3.2-3B's projections run
/// fused (one `[q+2kv]` and one `[2·inter]` GEMM) against separate (one GEMM per projection over
/// row views of the same fused weight, as the unfused executor runs them), and RoPE, paged
/// decode attention and SiLU·up on row-strided column blocks of the fused outputs against dense
/// operands, for m ∈ {1, 16, 64} (cold-cache weights, ~768-token contexts, 128-token pages).
/// The GEMMs are also timed for every m in 1..=24, 32, 48 and 64, on the first call at a new
/// prefill-sized m (the shim's heuristic query included) and in steady state there; and the
/// residual add + RMSNorm as `add` then `rmsnorm` against the ABI v2.1 `add_rmsnorm`.
/// Prints `fused_timing` lines. Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_ops -- fused_projection_timings --nocapture`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn fused_projection_timings() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(7);
    const ITERS: u32 = 100;
    let model = BENCH_MODELS[0];
    let (h, inter) = (model.hidden, INTERMEDIATE);
    let (q_rows, kv_rows) = (model.q_rows(), model.kv_rows());
    let qkv_w = q_rows + 2 * kv_rows;
    let gu_w = 2 * inter;
    let max_m = BENCH_BATCHES[BENCH_BATCHES.len() - 1];
    let gemm = p.hip.gemm().expect("hip gemm");
    let x = on_hip(
        &p,
        &[max_m, h],
        DType::BF16,
        &pattern(&mut rng, max_m * h, 1.0),
    );
    let report = |what: &str, m: usize, sep: f64, fused: f64| {
        println!(
            "fused_timing m={m:<2} {what:<40} separate {sep:>8.1} us  fused {fused:>8.1} us  delta {:>+8.1} us",
            fused - sep
        );
    };

    // GEMMs: separate row views of the fused weight vs one fused GEMM, cold-cache weights.
    for (what, parts) in [
        ("qkv", vec![q_rows, kv_rows, kv_rows]),
        ("gate_up", vec![inter, inter]),
    ] {
        let n: usize = parts.iter().sum();
        let raw = encode(
            DType::BF16,
            &pattern(&mut rng, n * h, 1.0 / (h as f32).sqrt()),
        );
        let copies = (512usize << 20).div_ceil(n * h * 2).max(1);
        let ws: Vec<Tensor> = (0..copies)
            .map(|_| raw_on_hip(&p, &[n, h], DType::BF16, &raw))
            .collect();
        let out = zeros_on_hip(&p, &[max_m, n], DType::BF16);
        for m in (1..=24).chain([32, 48, 64]) {
            let a = x.view().rows(0, m);
            let mut next = 0usize;
            let separate = time_us(&p, ITERS, || {
                next = (next + 1) % copies;
                let mut row = 0usize;
                for &rows in &parts {
                    let c = TensorView::contiguous(
                        out.storage.whole(),
                        max_m * row,
                        &[m, rows],
                        DType::BF16,
                    );
                    gemm.execute(&mut GemmContext {
                        a: a.clone(),
                        b: ws[next].view().rows(row, rows),
                        c,
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("separate gemm");
                    row += rows;
                }
            });
            let fused = time_us(&p, ITERS, || {
                next = (next + 1) % copies;
                gemm.execute(&mut GemmContext {
                    a: a.clone(),
                    b: ws[next].view(),
                    c: out.view().rows(0, m),
                    trans_b: true,
                    alpha: 1.0,
                    beta: 0.0,
                    prefill: false,
                })
                .expect("fused gemm");
            });
            report(&format!("gemm {what} n={n} k={h}"), m, separate, fused);
            // Each part alone, for the per-shape bandwidth.
            let mut row = 0usize;
            for &rows in &parts {
                let us = time_us(&p, ITERS, || {
                    next = (next + 1) % copies;
                    gemm.execute(&mut GemmContext {
                        a: a.clone(),
                        b: ws[next].view().rows(row, rows),
                        c: TensorView::contiguous(out.storage.whole(), 0, &[m, rows], DType::BF16),
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("part gemm");
                });
                println!(
                    "fused_timing m={m:<2} gemm part n={rows} k={h}: {us:.1} us ({:.0} GB/s); fused n={n}: {:.0} GB/s",
                    (rows * h * 2) as f64 / (us * 1e3),
                    (n * h * 2) as f64 / (fused * 1e3)
                );
                row += rows;
            }
        }
        // First call at a new m (the shim's heuristic query and algorithm cache miss), and the
        // steady state at prefill-sized m.
        let big = zeros_on_hip(&p, &[2100, n], DType::BF16);
        let xa = zeros_on_hip(&p, &[2100, h], DType::BF16);
        for m in [101usize, 257, 700, 1403, 2048, 2071] {
            let a = xa.view().rows(0, m);
            let once = |fused: bool| {
                p.hip_mem.synchronize().expect("synchronize");
                let start = std::time::Instant::now();
                if fused {
                    gemm.execute(&mut GemmContext {
                        a: a.clone(),
                        b: ws[0].view(),
                        c: big.view().rows(0, m),
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("fused gemm");
                } else {
                    let mut row = 0usize;
                    for &rows in &parts {
                        gemm.execute(&mut GemmContext {
                            a: a.clone(),
                            b: ws[0].view().rows(row, rows),
                            c: TensorView::contiguous(
                                big.storage.whole(),
                                2100 * row,
                                &[m, rows],
                                DType::BF16,
                            ),
                            trans_b: true,
                            alpha: 1.0,
                            beta: 0.0,
                            prefill: false,
                        })
                        .expect("separate gemm");
                        row += rows;
                    }
                }
                p.hip_mem.synchronize().expect("synchronize");
                start.elapsed().as_secs_f64() * 1e6
            };
            let first_sep = once(false);
            let first_fused = once(true);
            let sep = time_us(&p, 10, || {
                once(false);
            });
            let fused = time_us(&p, 10, || {
                once(true);
            });
            report(
                &format!("gemm {what} first call (new m)"),
                m,
                first_sep,
                first_fused,
            );
            report(&format!("gemm {what} steady (warm weights)"), m, sep, fused);
        }
    }

    // Consumers: dense operands vs column blocks of the fused outputs.
    let rope = p.hip.rope().expect("hip rope");
    let attn = p.hip.attention().expect("hip attention");
    let silu = p.hip.activation().expect("hip silu_mul");
    let qkv = on_hip(
        &p,
        &[max_m, qkv_w],
        DType::BF16,
        &pattern(&mut rng, max_m * qkv_w, 1.0),
    );
    let dense_q = on_hip(
        &p,
        &[max_m, q_rows],
        DType::BF16,
        &pattern(&mut rng, max_m * q_rows, 1.0),
    );
    let dense_k = on_hip(
        &p,
        &[max_m, kv_rows],
        DType::BF16,
        &pattern(&mut rng, max_m * kv_rows, 1.0),
    );
    let dense_v = on_hip(
        &p,
        &[max_m, kv_rows],
        DType::BF16,
        &pattern(&mut rng, max_m * kv_rows, 1.0),
    );
    let gu = on_hip(
        &p,
        &[max_m, gu_w],
        DType::BF16,
        &pattern(&mut rng, max_m * gu_w, 1.0),
    );
    let dense_g = on_hip(
        &p,
        &[max_m, inter],
        DType::BF16,
        &pattern(&mut rng, max_m * inter, 1.0),
    );
    let dense_u = on_hip(
        &p,
        &[max_m, inter],
        DType::BF16,
        &pattern(&mut rng, max_m * inter, 1.0),
    );
    let act_out = zeros_on_hip(&p, &[max_m, inter], DType::BF16);
    let attn_out = zeros_on_hip(&p, &[max_m, model.q_heads, HEAD_DIM], DType::BF16);
    let freq = on_hip(&p, &[HEAD_DIM / 2], DType::F32, &model.inv_freq());
    let rope_cfg = model.rope_cfg();
    let (qh, kh) = (model.q_heads, model.kv_heads);
    for &m in &BENCH_BATCHES {
        let dq = || column_block(&dense_q, 0, q_rows, m, &[qh, HEAD_DIM]);
        let dk = || column_block(&dense_k, 0, kv_rows, m, &[kh, HEAD_DIM]);
        let dv = || column_block(&dense_v, 0, kv_rows, m, &[kh, HEAD_DIM]);
        let fq = || column_block(&qkv, 0, qkv_w, m, &[qh, HEAD_DIM]);
        let fk = || column_block(&qkv, q_rows, qkv_w, m, &[kh, HEAD_DIM]);
        let fv = || column_block(&qkv, q_rows + kv_rows, qkv_w, m, &[kh, HEAD_DIM]);
        let positions: Vec<f32> = bench_kv_lens(m, BENCH_CTX)
            .iter()
            .map(|&kl| (kl - 1) as f32)
            .collect();
        let pos = on_hip(&p, &[m], DType::I32, &positions);
        let run_rope = |q: TensorView<'_>, k: TensorView<'_>| {
            rope.execute(&mut RopeContext {
                cfg: rope_cfg,
                q,
                k,
                positions: pos.view(),
                inv_freq: freq.view(),
                attn_factor: 1.0,
            })
            .expect("rope");
        };
        let sep = time_us(&p, ITERS, || run_rope(dq(), dk()));
        let fused = time_us(&p, ITERS, || run_rope(fq(), fk()));
        report(
            &format!("rope impl={}", rope.implementation(&rope_cfg)),
            m,
            sep,
            fused,
        );

        let cfg = model.attention_cfg(BENCH_BLOCK_TOKENS);
        let kv_lens = bench_kv_lens(m, BENCH_CTX);
        let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
        let max_blocks = max_kv.div_ceil(BENCH_BLOCK_TOKENS);
        let num_blocks = m * max_blocks;
        let table: Vec<f32> = shuffled(&mut rng, num_blocks)
            .iter()
            .map(|&b| b as f32)
            .collect();
        let pool = on_hip(
            &p,
            &[num_blocks, 2, BENCH_BLOCK_TOKENS, kh, HEAD_DIM],
            DType::BF16,
            &pattern(&mut rng, num_blocks * 2 * BENCH_BLOCK_TOKENS * kv_rows, 1.0),
        );
        let bt = on_hip(&p, &[m, max_blocks], DType::I32, &table);
        let indptr: Vec<f32> = (0..=m).map(|i| i as f32).collect();
        let ip = on_hip(&p, &[m + 1], DType::I32, &indptr);
        let lens: Vec<f32> = kv_lens.iter().map(|&k| k as f32).collect();
        let kl = on_hip(&p, &[m], DType::I32, &lens);
        let run_attn = |q: TensorView<'_>, k: TensorView<'_>, v: TensorView<'_>| {
            attn.execute_paged(&mut PagedAttentionContext {
                cfg,
                q,
                k_new: k,
                v_new: v,
                out: attn_out.view().rows(0, m),
                kv_layer: pool.view(),
                block_table: bt.view(),
                q_indptr: ip.view(),
                kv_lens: kl.view(),
                max_q_len: 1,
                max_kv_len: max_kv as u32,
                max_blocks_per_seq: max_blocks as u32,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                k_scale: 1.0,
                v_scale: 1.0,
            })
            .expect("paged decode attention");
        };
        let sep = time_us(&p, ITERS, || run_attn(dq(), dk(), dv()));
        let fused = time_us(&p, ITERS, || run_attn(fq(), fk(), fv()));
        report(
            &format!("attention_decode_paged impl={}", attn.implementation(&cfg)),
            m,
            sep,
            fused,
        );

        let act_cfg = ActivationConfig {
            cols: inter as u64,
            dtype: DType::BF16,
        };
        let run_silu = |gate: TensorView<'_>, up: TensorView<'_>| {
            silu.execute(&mut ActivationContext {
                gate,
                up,
                out: act_out.view().rows(0, m),
            })
            .expect("silu_mul");
        };
        let sep = time_us(&p, ITERS, || {
            run_silu(dense_g.view().rows(0, m), dense_u.view().rows(0, m))
        });
        let fused = time_us(&p, ITERS, || {
            run_silu(
                column_block(&gu, 0, gu_w, m, &[inter]),
                column_block(&gu, inter, gu_w, m, &[inter]),
            )
        });
        report(
            &format!("silu_mul impl={}", silu.implementation(&act_cfg)),
            m,
            sep,
            fused,
        );

        // Residual add + RMSNorm: `add` then `rmsnorm` against the ABI v2.1 `add_rmsnorm`.
        if let Some(add_norm) = p.hip.add_rmsnorm() {
            let add = p.hip.elementwise().expect("hip add");
            let norm = p.hip.norm().expect("hip norm");
            let an_cfg = AddRmsnormConfig {
                dtype: DType::BF16,
                dim: h as u32,
            };
            let resid = on_hip(
                &p,
                &[max_m, h],
                DType::BF16,
                &pattern(&mut rng, max_m * h, 1.0),
            );
            let w = on_hip(&p, &[h], DType::BF16, &pattern(&mut rng, h, 1.0));
            let normed = zeros_on_hip(&p, &[max_m, h], DType::BF16);
            let sep = time_us(&p, ITERS, || {
                add.execute(&mut ElementwiseContext {
                    a: rows(&resid, m),
                    b: rows(&x, m),
                    out: rows(&resid, m),
                })
                .expect("add");
                norm.execute(&mut NormContext {
                    x: rows(&resid, m),
                    weight: w.view(),
                    out: rows(&normed, m),
                    eps: 1e-5,
                })
                .expect("rmsnorm");
            });
            let fused = time_us(&p, ITERS, || {
                add_norm
                    .execute(&mut AddRmsnormContext {
                        residual: rows(&resid, m),
                        x: rows(&x, m),
                        weight: w.view(),
                        out: rows(&normed, m),
                        eps: 1e-5,
                    })
                    .expect("add_rmsnorm");
            });
            report(
                &format!("add+rmsnorm impl={}", add_norm.implementation(&an_cfg)),
                m,
                sep,
                fused,
            );
        }
    }
}

/// Kernel ABI v2.3 on the R9700: staged copies through page-locked memory round-trip exactly,
/// and they do not wait for earlier work on the stream — with ≈ 100 ms of GEMMs queued, writing
/// and uploading a second staging buffer and enqueueing a download behind the GEMMs take a
/// fraction of that, while reading the download back waits for the GEMMs. Breaks if a staged copy
/// synchronizes the stream (the engine could not overlap host work with the forward pass) or if
/// a read does not wait for its copy.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn host_staging_does_not_wait_for_the_stream() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let staging = HostStaging::alloc(&p.hip_mem, 1 << 20)
        .expect("libturbine_hip.so exports the v2.3 staging functions");
    let dev = Tensor::empty(&p.hip_mem, &[1 << 18], DType::I32).expect("device buffer");
    let pattern: Vec<u8> = (0..1u32 << 18)
        .flat_map(|i| (i * 7 + 3).to_le_bytes())
        .collect();
    staging.write(0, &pattern).expect("write");
    staging.upload(0, dev.storage.whole()).expect("upload");
    assert_eq!(dev.view().slice.read_bytes().expect("read back"), pattern);

    // Queue ≈ 100 ms of GEMMs, then time host work on a second staging buffer.
    let (m, n, k) = (4096usize, 4096usize, 4096usize);
    let gemm = p.hip.gemm().expect("hip gemm");
    let a = zeros_on_hip(&p, &[m, k], DType::BF16);
    let b = zeros_on_hip(&p, &[n, k], DType::BF16);
    let c = zeros_on_hip(&p, &[m, n], DType::BF16);
    let run = || {
        gemm.execute(&mut GemmContext {
            a: a.view(),
            b: b.view(),
            c: c.view(),
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
            prefill: false,
        })
        .expect("gemm");
    };
    let one = time_us(&p, 3, run);
    let queued = ((100_000.0 / one).ceil() as u32).clamp(4, 400);
    let other = HostStaging::alloc(&p.hip_mem, 4096).expect("second staging");
    let small = Tensor::empty(&p.hip_mem, &[1024], DType::I32).expect("small");
    let started = std::time::Instant::now();
    for _ in 0..queued {
        run();
    }
    let enqueued = started.elapsed();
    other.write(0, &[5u8; 4096]).expect("write while busy");
    other
        .upload(0, small.storage.whole())
        .expect("upload while busy");
    staging
        .download(0, dev.storage.whole())
        .expect("download behind the GEMMs");
    let host_side = started.elapsed() - enqueued;
    let mut back = vec![0u8; pattern.len()];
    staging
        .read(0, &mut back)
        .expect("read waits for the download");
    let total = started.elapsed();
    println!(
        "host staging: {queued} GEMMs of {one:.0} us queued in {enqueued:?}; the staged write \
         and copies took {host_side:?}; the read returned after {total:?}"
    );
    assert_eq!(
        back, pattern,
        "the staged download reads the uploaded bytes"
    );
    assert!(
        host_side.as_secs_f64() * 4.0 < total.as_secs_f64(),
        "staged copies waited for the queued work: {host_side:?} of {total:?}"
    );
    let expected = f64::from(queued) * one * 1e-6;
    assert!(
        total.as_secs_f64() >= 0.5 * expected,
        "the read did not wait for the queued GEMMs: {total:?} < half of {expected} s"
    );
    assert_eq!(
        small.view().slice.read_bytes().expect("small"),
        vec![5u8; 4096]
    );
}

// ------------------------------------------------------ prefill (lab, P2c TTFT)

/// P2c (prefill): the ops at prefill-sized token counts, where the HIP library switches to its
/// many-token kernels, match the CPU reference — RoPE over whole tokens (from 128 tokens, BF16)
/// on the q and k column blocks of a fused QKV output at the Llama and OLMoE shapes, positions
/// past the first chunk included; SiLU·up with 16-byte slices (from 64 rows) on the column
/// blocks of a fused gate-up output; `moe_route` (exact) at 680 and 2,048 tokens.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn prefill_shapes_match_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(17);

    // RoPE: (q heads, kv heads, theta, tokens, first position, attention factor); the YaRN
    // factor (ABI v2.10) on both kernels (below and from 128 tokens), past 8192 positions.
    let rope_cases = [
        (Q_HEADS, KV_HEADS, ROPE_THETA, 127, 0, 1.0),
        (Q_HEADS, KV_HEADS, ROPE_THETA, 128, 0, 1.0),
        (Q_HEADS, KV_HEADS, ROPE_THETA, 300, 5000, 1.0),
        (Q_HEADS, KV_HEADS, ROPE_THETA, 2048, 0, 1.0),
        (16, 16, 10_000.0, 680, 2048, 1.0),
        (Q_HEADS, KV_HEADS, ROPE_THETA, 17, 9000, YARN16_ATTN_FACTOR),
        (
            Q_HEADS,
            KV_HEADS,
            ROPE_THETA,
            300,
            11_900,
            YARN16_ATTN_FACTOR,
        ),
    ];
    let hip = p.hip.rope().expect("hip rope");
    assert!(
        hip.attn_factor_supported(),
        "the HIP library reads attn_factor (v2.10)"
    );
    let half = HEAD_DIM / 2;
    for (qh, kh, theta, t, first, attn_factor) in rope_cases {
        let cfg = RopeConfig {
            num_q_heads: qh as u32,
            num_kv_heads: kh as u32,
            head_dim: HEAD_DIM as u32,
            rotary_dim: HEAD_DIM as u32,
            dtype: DType::BF16,
        };
        assert!(hip.supports(&cfg), "hip must support rope {cfg}");
        let impl_name = hip.implementation(&cfg);
        let (q_rows, kv_rows) = (qh * HEAD_DIM, kh * HEAD_DIM);
        let width = q_rows + 2 * kv_rows;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
            .collect();
        let positions: Vec<f32> = (0..t).map(|i| (first + i) as f32).collect();
        let (x_hip, x_cpu) = twin(&p, &[t, width], DType::BF16, &rng.normal(t * width, 1.0));
        let (pos_hip, pos_cpu) = twin(&p, &[t], DType::I32, &positions);
        let (f_hip, f_cpu) = twin(&p, &[half], DType::F32, &inv_freq);
        let runs = [
            (hip, &x_hip, &pos_hip, &f_hip),
            (p.cpu.rope().expect("cpu rope"), &x_cpu, &pos_cpu, &f_cpu),
        ];
        for (kernel, x, pos, f) in runs {
            let mut ctx = RopeContext {
                cfg,
                q: column_block(x, 0, width, t, &[qh, HEAD_DIM]),
                k: column_block(x, q_rows, width, t, &[kh, HEAD_DIM]),
                positions: pos.view(),
                inv_freq: f.view(),
                attn_factor,
            };
            kernel.execute(&mut ctx).expect("rope");
        }
        let what = format!(
            "rope fused qkv tokens={t} first_position={first} attn_factor={attn_factor} {cfg}"
        );
        assert_close(&what, &impl_name, &read(&x_hip), &read(&x_cpu), DType::BF16);
    }

    // SiLU · up on the column blocks of a fused [rows, 2 · inter] gate-up output.
    let cfg = ActivationConfig {
        cols: INTERMEDIATE as u64,
        dtype: DType::BF16,
    };
    let hip = p.hip.activation().expect("hip silu_mul");
    assert!(hip.supports(&cfg), "hip must support silu_mul {cfg}");
    let impl_name = hip.implementation(&cfg);
    for rows in [63, 64, 680] {
        let width = 2 * INTERMEDIATE;
        let (gu_hip, gu_cpu) = twin(
            &p,
            &[rows, width],
            DType::BF16,
            &rng.normal(rows * width, 2.0),
        );
        let n = rows * INTERMEDIATE;
        let (o_hip, o_cpu) = twin(&p, &[rows, INTERMEDIATE], DType::BF16, &vec![0.0; n]);
        let runs = [
            (hip, &gu_hip, &o_hip),
            (p.cpu.activation().expect("cpu silu_mul"), &gu_cpu, &o_cpu),
        ];
        for (kernel, gu, o) in runs {
            let mut ctx = ActivationContext {
                gate: column_block(gu, 0, width, rows, &[INTERMEDIATE]),
                up: column_block(gu, INTERMEDIATE, width, rows, &[INTERMEDIATE]),
                out: o.view(),
            };
            kernel.execute(&mut ctx).expect("silu_mul");
        }
        let what = format!("silu_mul fused gate-up rows={rows} {cfg}");
        assert_close(&what, &impl_name, &read(&o_hip), &read(&o_cpu), DType::BF16);
    }

    // moe_route at prefill sizes: selections, grouping and offsets identical.
    let olmoe = MoeRouteConfig {
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        renormalize: false,
        bf16_logits: true,
    };
    for tokens in [680, 2048] {
        route_case(&p, olmoe, tokens, &rng.normal(tokens * MOE_EXPERTS, 2.0));
    }
    // Every token on the same 8 experts: 2,048 rows per expert.
    let mut skewed = rng.normal(256 * MOE_EXPERTS, 0.1);
    for row in skewed.chunks_exact_mut(MOE_EXPERTS) {
        for v in &mut row[40..48] {
            *v += 10.0;
        }
    }
    route_case(&p, olmoe, 256, &skewed);
}

/// Prefill chunk sizes of the prefill benchmarks: a ~680-token prompt, the default
/// `prefill_chunk_tokens` and a full default `max_batch_tokens` iteration.
const PREFILL_TOKENS: [usize; 3] = [680, 2048, 8192];

/// Lab microbenchmark (no assertion on speed): the ops whose cost grows with the prefill chunk,
/// per call and per forward, at m ∈ [`PREFILL_TOKENS`] — every GEMM shape of the fused-projection
/// executors (TFLOPS printed), RoPE and SiLU·up on row-strided column blocks of the fused
/// outputs (as the executors run them), and for OLMoE `moe_route` and `moe_experts` (whichever
/// path the library picks at that many routed rows; the host offsets are read once outside the
/// timing). Prints `prefill_timing` lines. Run with
/// `scripts/lab-test.sh novanas -- --release -p turbine-kernels --test hip_ops -- prefill_op_timings --nocapture`.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn prefill_op_timings() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(11);
    const ITERS: u32 = 10;
    let max_m = PREFILL_TOKENS[PREFILL_TOKENS.len() - 1];
    let line = |model: &str, m: usize, what: &str, imp: &str, us: f64, calls: usize| {
        println!(
            "prefill_timing {model} m={m:<5} {what:<44} impl={imp:<26} {us:>10.1} us/call x{calls:>3} = {:>8.3} ms/forward",
            us * calls as f64 / 1e3
        );
    };
    for model in BENCH_MODELS {
        let (h, l) = (model.hidden, model.layers);
        let (q_rows, kv_rows) = (model.q_rows(), model.kv_rows());
        let qkv_w = q_rows + 2 * kv_rows;
        let mut shapes = vec![
            ("qkv", qkv_w, h, DType::BF16, l),
            ("o_proj", h, q_rows, DType::BF16, l),
        ];
        match model.mlp {
            Mlp::Dense { inter } => shapes.extend([
                ("gate_up", 2 * inter, h, DType::BF16, l),
                ("down", h, inter, DType::BF16, l),
            ]),
            Mlp::Moe { experts, .. } => shapes.push(("router", experts, h, DType::F32, l)),
        }
        let widest = match model.mlp {
            Mlp::Dense { inter } => (2 * inter).max(qkv_w),
            Mlp::Moe { .. } => qkv_w.max(h),
        };
        let act = on_hip(
            &p,
            &[max_m, widest],
            DType::BF16,
            &pattern(&mut rng, max_m * widest, 1.0),
        );
        let gemm = p.hip.gemm().expect("hip gemm");
        for (name, n, k, c_dtype, calls) in shapes {
            let cfg = gemm_config(n, k, c_dtype);
            let w = on_hip(
                &p,
                &[n, k],
                DType::BF16,
                &pattern(&mut rng, n * k, 1.0 / (k as f32).sqrt()),
            );
            let c = zeros_on_hip(&p, &[max_m, n], c_dtype);
            for m in PREFILL_TOKENS {
                let a = TensorView::contiguous(act.storage.whole(), 0, &[m, k], DType::BF16);
                let us = time_us(&p, ITERS, || {
                    gemm.execute(&mut GemmContext {
                        a: a.clone(),
                        b: w.view(),
                        c: c.view().rows(0, m),
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                        prefill: false,
                    })
                    .expect("gemm");
                });
                let tflops = 2.0 * (m * n * k) as f64 / (us * 1e6);
                line(
                    model.name,
                    m,
                    &format!("gemm {name} n={n} k={k} ({tflops:.1} TFLOPS)"),
                    &gemm.implementation(&cfg),
                    us,
                    calls,
                );
            }
        }

        // RoPE on the q and k column blocks of a fused QKV output.
        let rope = p.hip.rope().expect("hip rope");
        let rope_cfg = model.rope_cfg();
        let freq = on_hip(&p, &[HEAD_DIM / 2], DType::F32, &model.inv_freq());
        let qkv = on_hip(
            &p,
            &[max_m, qkv_w],
            DType::BF16,
            &pattern(&mut rng, max_m * qkv_w, 1.0),
        );
        let (qh, kh) = (model.q_heads, model.kv_heads);
        for m in PREFILL_TOKENS {
            let positions: Vec<f32> = (0..m).map(|i| i as f32).collect();
            let pos = on_hip(&p, &[m], DType::I32, &positions);
            let us = time_us(&p, ITERS, || {
                rope.execute(&mut RopeContext {
                    cfg: rope_cfg,
                    q: column_block(&qkv, 0, qkv_w, m, &[qh, HEAD_DIM]),
                    k: column_block(&qkv, q_rows, qkv_w, m, &[kh, HEAD_DIM]),
                    positions: pos.view(),
                    inv_freq: freq.view(),
                    attn_factor: 1.0,
                })
                .expect("rope");
            });
            line(
                model.name,
                m,
                "rope (fused qkv columns)",
                &rope.implementation(&rope_cfg),
                us,
                l,
            );
        }

        match model.mlp {
            Mlp::Dense { inter } => {
                // SiLU·up on the gate and up column blocks of a fused gate-up output.
                let silu = p.hip.activation().expect("hip silu_mul");
                let cfg = ActivationConfig {
                    cols: inter as u64,
                    dtype: DType::BF16,
                };
                let out = zeros_on_hip(&p, &[max_m, inter], DType::BF16);
                for m in PREFILL_TOKENS {
                    let us = time_us(&p, ITERS, || {
                        silu.execute(&mut ActivationContext {
                            gate: column_block(&act, 0, 2 * inter, m, &[inter]),
                            up: column_block(&act, inter, 2 * inter, m, &[inter]),
                            out: out.view().rows(0, m),
                        })
                        .expect("silu_mul");
                    });
                    line(
                        model.name,
                        m,
                        "silu_mul (fused gate-up columns)",
                        &silu.implementation(&cfg),
                        us,
                        l,
                    );
                }
            }
            Mlp::Moe {
                experts,
                top_k,
                inter,
            } => {
                let moe = p.hip.moe().expect("hip moe");
                let route_cfg = MoeRouteConfig {
                    num_experts: experts as u32,
                    top_k: top_k as u32,
                    renormalize: false,
                    bf16_logits: true,
                };
                let experts_cfg = MoeExpertsConfig {
                    hidden: h as u32,
                    inter: inter as u32,
                    num_experts: experts as u32,
                    top_k: top_k as u32,
                    expert_begin: 0,
                    expert_end: experts as u32,
                    dtype: DType::BF16,
                };
                let up_scale = 1.0 / (h as f32).sqrt();
                let gate_raw = encode(
                    DType::BF16,
                    &pattern(&mut rng, experts * inter * h, up_scale),
                );
                let w_gate = raw_on_hip(&p, &[experts, inter, h], DType::BF16, &gate_raw);
                let w_up = raw_on_hip(&p, &[experts, inter, h], DType::BF16, &gate_raw);
                drop(gate_raw);
                let w_down = on_hip(
                    &p,
                    &[experts, h, inter],
                    DType::BF16,
                    &pattern(&mut rng, experts * inter * h, 1.0 / (inter as f32).sqrt()),
                );
                let x = on_hip(
                    &p,
                    &[max_m, h],
                    DType::BF16,
                    &pattern(&mut rng, max_m * h, 1.0),
                );
                let o = zeros_on_hip(&p, &[max_m, h], DType::BF16);
                let logits = on_hip(
                    &p,
                    &[max_m, experts],
                    DType::F32,
                    &rng.normal(max_m * experts, 1.0),
                );
                let topk_ids = zeros_on_hip(&p, &[max_m, top_k], DType::I32);
                let topk_w = zeros_on_hip(&p, &[max_m, top_k], DType::F32);
                let sorted = zeros_on_hip(&p, &[max_m * top_k], DType::I32);
                let offsets = zeros_on_hip(&p, &[experts + 1], DType::I32);
                for m in [64, 128, 256].into_iter().chain(PREFILL_TOKENS) {
                    let sorted_rows =
                        TensorView::contiguous(sorted.storage.whole(), 0, &[m * top_k], DType::I32);
                    let route = || {
                        moe.route(&mut MoeRouteContext {
                            cfg: route_cfg,
                            router_logits: logits.view().rows(0, m),
                            topk_ids: topk_ids.view().rows(0, m),
                            topk_weights: topk_w.view().rows(0, m),
                            sorted_rows: sorted_rows.clone(),
                            expert_offsets: offsets.view(),
                        })
                        .expect("moe_route");
                    };
                    let us = time_us(&p, ITERS, route);
                    line(
                        model.name,
                        m,
                        "moe_route",
                        &moe.implementation_route(&route_cfg),
                        us,
                        l,
                    );
                    let host_offsets: Vec<i32> = offsets
                        .storage
                        .whole()
                        .read_bytes()
                        .expect("read offsets")
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let us = time_us(&p, ITERS, || {
                        moe.experts(&mut MoeExpertsContext {
                            cfg: experts_cfg,
                            x: x.view().rows(0, m),
                            w_gate: w_gate.view(),
                            w_up: w_up.view(),
                            w_down: w_down.view(),
                            sorted_rows: sorted_rows.clone(),
                            expert_offsets: offsets.view(),
                            topk_weights: topk_w.view().rows(0, m),
                            host_expert_offsets: &host_offsets,
                            out: o.view().rows(0, m),
                            workspace: None,
                        })
                        .expect("moe_experts");
                    });
                    let tflops = 6.0 * (m * top_k * h * inter) as f64 / (us * 1e6);
                    line(
                        model.name,
                        m,
                        &format!("moe_experts ({tflops:.1} TFLOPS)"),
                        &moe.implementation_experts(&experts_cfg),
                        us,
                        l,
                    );
                }
            }
        }
    }
}

// ------------------------------------------ implementation enumeration (lab, Phase 2m S-5)

/// The `dtype=` value of a rendered config.
fn parse_dtype(v: &str) -> DType {
    match v {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "f32" => DType::F32,
        other => panic!("unknown dtype {other}"),
    }
}

/// The `OpConfig` of one recorded kernel choice (`op` and its rendered `config`), which must
/// render back to `config`.
fn parse_choice(op: &str, config: &str) -> OpConfig {
    let kv: std::collections::HashMap<&str, &str> = config
        .split_whitespace()
        .map(|p| p.split_once('=').expect("key=value"))
        .collect();
    let num = |k: &str| -> u64 {
        kv.get(k)
            .unwrap_or_else(|| panic!("{op} {config}: no {k}"))
            .parse()
            .unwrap_or_else(|e| panic!("{op} {config}: {k}: {e}"))
    };
    let flag = |k: &str| num(k) == 1;
    let dtype = |k: &str| parse_dtype(kv[k]);
    let attention = |kind| {
        OpConfig::Attention(AttentionConfig {
            kind,
            num_q_heads: num("q_heads") as u32,
            num_kv_heads: num("kv_heads") as u32,
            head_dim: num("head_dim") as u32,
            dtype: dtype("dtype"),
            block_tokens: kv
                .get("block_tokens")
                .map(|b| b.parse().expect("block_tokens")),
            causal: flag("causal"),
        })
    };
    let spec = match op {
        "gemm" => OpConfig::Gemm(GemmConfig {
            n: num("n"),
            k: num("k"),
            trans_b: flag("trans_b"),
            a_dtype: dtype("a_dtype"),
            b_dtype: dtype("b_dtype"),
            c_dtype: dtype("c_dtype"),
        }),
        "attention_prefill" => attention(AttentionKind::Prefill),
        "attention_decode" => attention(AttentionKind::Decode),
        "attention_prefill_paged" => attention(AttentionKind::PrefillPaged),
        "attention_decode_paged" => attention(AttentionKind::DecodePaged),
        "rmsnorm" => OpConfig::Rmsnorm(NormConfig {
            dim: num("dim"),
            dtype: dtype("dtype"),
        }),
        "add_rmsnorm" => OpConfig::AddRmsnorm(AddRmsnormConfig {
            dim: num("dim") as u32,
            dtype: dtype("dtype"),
        }),
        "rope" => OpConfig::Rope(RopeConfig {
            num_q_heads: num("q_heads") as u32,
            num_kv_heads: num("kv_heads") as u32,
            head_dim: num("head_dim") as u32,
            rotary_dim: num("rotary_dim") as u32,
            dtype: dtype("dtype"),
        }),
        "silu_mul" => OpConfig::SiluMul(ActivationConfig {
            cols: num("cols"),
            dtype: dtype("dtype"),
        }),
        "embedding" => OpConfig::Embedding(EmbeddingConfig {
            hidden: num("hidden"),
            vocab_rows: num("vocab_rows"),
            dtype: dtype("dtype"),
        }),
        "add" => OpConfig::Add(ElementwiseConfig {
            dtype: dtype("dtype"),
        }),
        "copy_blocks" => OpConfig::CopyBlocks(KvCopyConfig {
            num_layers: num("num_layers") as u32,
            block_bytes: num("block_bytes"),
        }),
        "moe_route" => OpConfig::MoeRoute(MoeRouteConfig {
            num_experts: num("experts") as u32,
            top_k: num("top_k") as u32,
            renormalize: flag("renormalize"),
            bf16_logits: flag("bf16_logits"),
        }),
        "moe_experts" => {
            let (begin, end) = kv["local"].split_once("..").expect("local=a..b");
            OpConfig::MoeExperts(MoeExpertsConfig {
                hidden: num("hidden") as u32,
                inter: num("inter") as u32,
                num_experts: num("experts") as u32,
                top_k: num("top_k") as u32,
                expert_begin: begin.parse().expect("expert_begin"),
                expert_end: end.parse().expect("expert_end"),
                dtype: dtype("dtype"),
            })
        }
        "logits_reduce" => OpConfig::LogitsReduce(LogitsReduceConfig {
            vocab: num("vocab") as u32,
            top_n: num("top_n") as u32,
        }),
        other => panic!("unknown op {other}"),
    };
    assert_eq!(spec.op().as_str(), op);
    assert_eq!(spec.render(), config, "{op} config round trip");
    spec
}

/// `(model, spec, implementation)` of every entry of
/// `tests/lab/kernel-choices-{llama,olmoe}.json`: the implementations the served models ran on
/// main (recorded from `/turbine/v1/status`).
fn recorded_choices() -> Vec<(&'static str, OpConfig, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/lab");
    let mut out = Vec::new();
    for model in ["llama", "olmoe"] {
        let path = dir.join(format!("kernel-choices-{model}.json"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let entries: serde_json::Value = serde_json::from_str(&text).expect("kernel choices JSON");
        for e in entries.as_array().expect("an array of choices") {
            let field = |k: &str| e[k].as_str().unwrap_or_else(|| panic!("{k} in {e}"));
            let spec = parse_choice(field("op"), field("config"));
            out.push((model, spec, field("implementation").to_string()));
        }
    }
    out
}

/// Phase 2m Task 10 (kernel ABI v2.4): `libturbine_hip.so` enumerates exactly its implementation
/// table per op (names, provider families, the host-offsets flag), and for every op config the
/// served Llama and OLMoE models use (`tests/lab/kernel-choices-*.json`) the first
/// implementation in library order that `turbine_impl_supports` accepts (for `moe_experts` at
/// the first row tier of the gfx1201 profile) is the one the library's own `turbine_<op>_impl`
/// names and the one main served. Breaks if the table drifts from the contract (§9.1) or if
/// enumeration and the library's own choice disagree.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn implementations_enumerated() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    const HOST: bool = true;
    let one = |name: &'static str, provider: &'static str| vec![(name, provider, false)];
    let norm = vec![
        ("ck_tile_rmsnorm2d", "ck", false),
        ("turbine_hip", "turbine_hip", false),
    ];
    let paged = vec![
        ("ck_tile_fmha_pagedkv", "ck", false),
        ("turbine_hip", "turbine_hip", false),
        ("ck_tile_fmha_pagedkv_fp8_staged", "ck", false),
        ("turbine_hip_fp8", "turbine_hip", false),
    ];
    let paged_decode = vec![
        ("ck_tile_fmha_splitkv", "ck", false),
        ("ck_tile_fmha_pagedkv", "ck", false),
        ("turbine_hip", "turbine_hip", false),
        ("turbine_hip_fp8_decode", "turbine_hip", false),
        ("turbine_hip_fp8", "turbine_hip", false),
    ];
    // (name, provider, needs host offsets) in library order.
    type Impls = Vec<(&'static str, &'static str, bool)>;
    let table: Vec<(OpKind, Impls)> = vec![
        (OpKind::Gemm, one("hipblaslt", "hipblaslt")),
        (OpKind::AttentionPrefill, one("ck_tile_fmha_fwd", "ck")),
        (OpKind::AttentionDecode, one("ck_tile_fmha_fwd", "ck")),
        (OpKind::Rmsnorm, norm.clone()),
        (OpKind::Rope, one("turbine_hip", "turbine_hip")),
        (OpKind::SiluMul, one("turbine_hip", "turbine_hip")),
        (OpKind::Embedding, one("turbine_hip", "turbine_hip")),
        (OpKind::Add, one("turbine_hip", "turbine_hip")),
        (OpKind::AttentionPrefillPaged, paged),
        (OpKind::AttentionDecodePaged, paged_decode),
        (OpKind::CopyBlocks, one("hip_memcpy_d2d", "turbine_hip")),
        (OpKind::MoeRoute, one("turbine_hip", "turbine_hip")),
        (
            OpKind::MoeExperts,
            vec![
                ("turbine_hip_moe_small_m", "turbine_hip", false),
                ("turbine_hip_moe_wmma_prefill", "turbine_hip", false),
                ("turbine_hip_moe_wmma", "turbine_hip", false),
                ("hipblaslt_grouped", "hipblaslt", HOST),
                ("hipblaslt_per_expert", "hipblaslt", HOST),
            ],
        ),
        (OpKind::AddRmsnorm, norm),
        (OpKind::LogitsReduce, one("turbine_hip", "turbine_hip")),
        (OpKind::RowSumsq, one("turbine_hip", "turbine_hip")),
        (OpKind::RmsnormSharded, one("turbine_hip", "turbine_hip")),
        // ABI v2.9 (Phase 6a).
        (OpKind::QGemm, one("hipblaslt_fp8", "hipblaslt")),
        (OpKind::QuantizeAct, one("turbine_hip", "turbine_hip")),
    ];
    assert_eq!(table.len(), OpKind::ALL.len());
    for (op, want) in &table {
        let want: Vec<ImplInfo> = want
            .iter()
            .enumerate()
            .map(|(i, &(name, provider, host))| ImplInfo {
                index: i as u32,
                name: name.into(),
                provider: provider.into(),
                needs_host_offsets: host,
            })
            .collect();
        assert_eq!(p.hip.implementations(*op), want, "{op}");
        let names: Vec<&str> = want.iter().map(|i| i.name.as_str()).collect();
        println!("implementations {op}: {}", names.join(", "));
    }

    let first_tier = GFX1201.thresholds.moe_small_max_rows;
    for (model, spec, recorded) in recorded_choices() {
        let what = format!("{model} {} {}", spec.op(), spec.render());
        let legacy = spec
            .probe(p.hip.as_ref())
            .unwrap_or_else(|| panic!("{what}: unsupported"));
        let rows = matches!(spec, OpConfig::MoeExperts(_)).then_some(first_tier);
        let enumerated = p
            .hip
            .implementations(spec.op())
            .into_iter()
            .find(|i| p.hip.implementation_supports(&spec, i.index, rows))
            .unwrap_or_else(|| panic!("{what}: no implementation supports it"));
        assert_eq!(legacy, recorded, "{what}: the library's own choice");
        assert_eq!(
            enumerated.name, recorded,
            "{what}: first supporting implementation"
        );
        println!("{what}: {recorded} ok");
    }
}

/// A copy of `p` whose HIP provider runs implementation `index` of `spec`'s op on every call
/// (bound the way the kernel registry binds the implementation it chose).
fn bound(p: &Pair, spec: &OpConfig, index: u32) -> Pair {
    Pair {
        hip: p
            .hip
            .bind(spec, &ImplChoice::Single(index))
            .unwrap_or_else(|| panic!("bind {} implementation {index}", spec.op())),
        cpu: Arc::clone(&p.cpu),
        hip_mem: Arc::clone(&p.hip_mem),
        cpu_mem: Arc::clone(&p.cpu_mem),
        legacy: false,
        ctx: Arc::clone(&p.ctx),
        arch: p.arch.clone(),
    }
}

/// One case of `every_implementation_matches_cpu`: the config an implementation must support
/// (`moe_experts` at `rows` routed rows) and the existing `hip_ops` case that runs it against the
/// CPU reference with that op's tolerance.
/// A case body run with a (bound) pair.
type CaseFn<'a> = Box<dyn Fn(&Pair, &mut Rng) + 'a>;

struct ImplCase<'a> {
    spec: OpConfig,
    rows: Option<u32>,
    run: CaseFn<'a>,
}

/// Phase 2m S-5 / S-13: every implementation the library enumerates for every op, run alone
/// through `turbine_impl_run` (a provider bound to it) on the shapes of the existing `hip_ops`
/// cases of that op it supports (`moe_experts` at 8 and 1,024 routed rows, paged attention at 16-
/// and 128-token pages), matches the CPU reference under that op's tolerance. Every enumerated
/// implementation runs at least once, except `hipblaslt_grouped` where hipBLASLt has no grouped
/// solution. Breaks if an implementation the registry can choose disagrees with the reference.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn every_implementation_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let mut rng = Rng(31);
    let bf16 = DType::BF16;
    let gemm = |n: usize, k: usize, c_dtype| {
        OpConfig::Gemm(GemmConfig {
            n: n as u64,
            k: k as u64,
            trans_b: true,
            a_dtype: bf16,
            b_dtype: bf16,
            c_dtype,
        })
    };
    let attention = |kind, block_tokens: Option<usize>, (q, kv): (usize, usize)| {
        OpConfig::Attention(AttentionConfig {
            kind,
            num_q_heads: q as u32,
            num_kv_heads: kv as u32,
            head_dim: HEAD_DIM as u32,
            dtype: bf16,
            block_tokens: block_tokens.map(|b| b as u32),
            causal: true,
        })
    };
    let w = expert_weights(&p, &mut rng);
    let experts_cfg = MoeExpertsConfig {
        hidden: MOE_HIDDEN as u32,
        inter: MOE_INTER as u32,
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        expert_begin: 0,
        expert_end: MOE_EXPERTS as u32,
        dtype: bf16,
    };
    let olmoe_route = MoeRouteConfig {
        num_experts: MOE_EXPERTS as u32,
        top_k: MOE_TOP_K as u32,
        renormalize: false,
        bf16_logits: true,
    };
    fn case<'a>(spec: OpConfig, run: CaseFn<'a>) -> ImplCase<'a> {
        ImplCase {
            spec,
            rows: None,
            run,
        }
    }

    let mut cases: Vec<ImplCase<'_>> = vec![
        case(
            gemm(HIDDEN, HIDDEN, bf16),
            Box::new(|p, rng| {
                gemm_case(p, rng, 1, HIDDEN, HIDDEN, DType::BF16);
                gemm_case(p, rng, 17, INTERMEDIATE, HIDDEN, DType::BF16);
                gemm_case(p, rng, 17, 1024, HIDDEN, DType::F32);
            }),
        ),
        case(
            attention(AttentionKind::Prefill, None, (Q_HEADS, KV_HEADS)),
            Box::new(|p, rng| {
                attention_case(p, rng, AttentionKind::Prefill, 17, 0);
                attention_case(p, rng, AttentionKind::Prefill, 17, 100);
            }),
        ),
        case(
            attention(AttentionKind::Decode, None, (Q_HEADS, KV_HEADS)),
            Box::new(|p, rng| attention_case(p, rng, AttentionKind::Decode, 1, 511)),
        ),
        case(
            OpConfig::Rmsnorm(NormConfig {
                dim: HIDDEN as u64,
                dtype: bf16,
            }),
            // The inputs of `norm_rope_silu_embedding_add_match_cpu` (seed 3): `norm_case`'s
            // BF16 tolerance has no allowance for a rounding flip of `x·inv_rms`, which other
            // seeds hit on the ck_tile pipeline (see `add_rmsnorm_matches_cpu`).
            Box::new(|p, _rng| norm_case(p, &mut Rng(3), 17)),
        ),
        case(
            OpConfig::Rope(RopeConfig {
                num_q_heads: Q_HEADS as u32,
                num_kv_heads: KV_HEADS as u32,
                head_dim: HEAD_DIM as u32,
                rotary_dim: HEAD_DIM as u32,
                dtype: bf16,
            }),
            Box::new(|p, rng| rope_case(p, rng, 17, 1.0)),
        ),
        case(
            OpConfig::SiluMul(ActivationConfig {
                cols: INTERMEDIATE as u64,
                dtype: bf16,
            }),
            Box::new(|p, rng| silu_mul_case(p, rng, 17)),
        ),
        case(
            OpConfig::Embedding(EmbeddingConfig {
                hidden: HIDDEN as u64,
                vocab_rows: VOCAB as u64,
                dtype: bf16,
            }),
            Box::new(|p, rng| embedding_case(p, rng, 17)),
        ),
        case(
            OpConfig::Add(ElementwiseConfig { dtype: bf16 }),
            Box::new(|p, rng| add_case(p, rng, 17)),
        ),
        case(
            OpConfig::CopyBlocks(KvCopyConfig {
                num_layers: 3,
                block_bytes: (2 * 16 * KV_HEADS * HEAD_DIM * 2) as u64,
            }),
            Box::new(copy_blocks_case),
        ),
        case(
            OpConfig::MoeRoute(olmoe_route),
            Box::new(move |p, rng| {
                let logits = rng.normal(37 * MOE_EXPERTS, 2.0);
                route_case(p, olmoe_route, 37, &logits);
            }),
        ),
        case(
            OpConfig::LogitsReduce(LogitsReduceConfig {
                vocab: 50_304,
                top_n: 64,
            }),
            Box::new(|p, rng| {
                logits_reduce_case(p, rng, 64, 50_304, 64);
                logits_reduce_case(p, rng, 7, 50_304, 5);
            }),
        ),
    ];
    cases.push(case(
        OpConfig::RowSumsq(RowSumsqConfig {
            dim: (MOE_HIDDEN / 2) as u32,
            dtype: bf16,
        }),
        Box::new(|p, rng| {
            sharded_norm_case(p, rng, 37, MOE_HIDDEN, 2);
        }),
    ));
    cases.push(case(
        OpConfig::RmsnormSharded(RmsnormShardedConfig {
            dim: (MOE_HIDDEN / 2) as u32,
            full_dim: MOE_HIDDEN as u32,
            dtype: bf16,
        }),
        Box::new(|p, rng| {
            sharded_norm_case(p, rng, 37, MOE_HIDDEN, 2);
        }),
    ));
    for (rows, dim, x_stride) in [
        (16, HIDDEN, HIDDEN),
        (16, MOE_HIDDEN, MOE_HIDDEN),
        (7, 64, 64),
    ] {
        // (The row-strided x case of `add_rmsnorm_matches_cpu` is left out: a stride is not
        // part of the config an implementation is chosen for, and ck_tile refuses one that is
        // not a multiple of 8 at run time rather than switching implementation.)
        cases.push(case(
            OpConfig::AddRmsnorm(AddRmsnormConfig {
                dim: dim as u32,
                dtype: bf16,
            }),
            Box::new(move |p, rng| add_rmsnorm_case(p, rng, rows, dim, x_stride)),
        ));
    }
    for kind in [AttentionKind::PrefillPaged, AttentionKind::DecodePaged] {
        for block_tokens in [16, 128] {
            let OpConfig::Attention(base) =
                attention(kind, Some(block_tokens), (Q_HEADS, KV_HEADS))
            else {
                unreachable!("an attention config")
            };
            let fp8 = AttentionConfig {
                dtype: DType::F8E4M3,
                ..base
            };
            cases.push(case(
                OpConfig::Attention(fp8),
                Box::new(move |p, rng| {
                    let (q_lens, kv_lens): (&[usize], &[usize]) = match kind {
                        AttentionKind::PrefillPaged => (&[37, 1, 70], &[37, 300, 200]),
                        _ => (&[1, 1, 1, 1], &[1, 17, 129, 513]),
                    };
                    let pages = Pages::Fp8 {
                        k_scale: 0.07,
                        v_scale: 0.11,
                    };
                    paged_case_pages(
                        p,
                        rng,
                        kind,
                        (Q_HEADS, KV_HEADS),
                        block_tokens,
                        q_lens,
                        kv_lens,
                        pages,
                    );
                }),
            ));
            cases.push(case(
                attention(kind, Some(block_tokens), (Q_HEADS, KV_HEADS)),
                Box::new(move |p, rng| {
                    let (q_lens, kv_lens): (&[usize], &[usize]) = match kind {
                        AttentionKind::PrefillPaged => (&[37, 1, 70], &[37, 300, 200]),
                        _ => (&[1, 1, 1, 1], &[1, 17, 129, 513]),
                    };
                    paged_case(
                        p,
                        rng,
                        kind,
                        (Q_HEADS, KV_HEADS),
                        block_tokens,
                        q_lens,
                        kv_lens,
                    );
                }),
            ));
        }
    }
    for tokens in [1usize, 128] {
        let w = &w;
        cases.push(ImplCase {
            spec: OpConfig::MoeExperts(experts_cfg),
            rows: Some((tokens * MOE_TOP_K) as u32),
            run: Box::new(move |p, rng| {
                let logits = rng.normal(tokens * MOE_EXPERTS, 1.0);
                let case = ExpertsCase {
                    tokens,
                    logits: &logits,
                    local: (0, MOE_EXPERTS),
                    workspace: false,
                    repeat: 0,
                };
                experts_case(p, rng, w, &case);
            }),
        });
    }

    let mut ran: Vec<(OpKind, String)> = Vec::new();
    for c in &cases {
        let op = c.spec.op();
        let impls = p.hip.implementations(op);
        assert!(!impls.is_empty(), "{op}: the library enumerates nothing");
        for info in impls {
            let what = format!(
                "{op} {} rows={:?} impl={}",
                c.spec.render(),
                c.rows,
                info.name
            );
            if !p.hip.implementation_supports(&c.spec, info.index, c.rows) {
                println!("every_implementation {what}: not supported, skipped");
                continue;
            }
            (c.run)(&bound(&p, &c.spec, info.index), &mut rng);
            println!("every_implementation {what}: ok");
            ran.push((op, info.name));
        }
    }
    for &op in OpKind::ALL {
        // The v2.9 quantized ops are run against the CPU reference by their own suite
        // (`tests/hip_qgemm.rs`), implementation by implementation as their builders add them.
        if matches!(op, OpKind::QGemm | OpKind::QuantizeAct) {
            continue;
        }
        for info in p.hip.implementations(op) {
            if ran.iter().any(|(o, n)| *o == op && *n == info.name) {
                continue;
            }
            assert_eq!(
                info.name, "hipblaslt_grouped",
                "{op} implementation {} never ran",
                info.name
            );
            println!(
                "every_implementation {op} {}: no case supports it",
                info.name
            );
        }
    }
}

/// One row of a tuned GEMM table (`kernels/rocm/tuning/<arch>/gemm.tsv`): the shape and the
/// largest m it serves.
struct TunedRow {
    n: usize,
    k: usize,
    c_dtype: DType,
    m_max: usize,
    /// The row pins a solution (`false`: `heuristic`, hipBLASLt's own answer per call).
    pinned: bool,
    /// Mode `invariant` (prefill steps run it when the shape also has `speed` rows).
    invariant: bool,
}

/// The rows of the tuned GEMM table of `arch`, in file order; empty when the card has none.
fn tuned_rows(arch: &str) -> Vec<TunedRow> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/rocm/tuning")
        .join(arch)
        .join("gemm.tsv");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            let num = |i: usize| c[i].parse::<usize>().expect("numeric column");
            TunedRow {
                n: num(0),
                k: num(1),
                c_dtype: if c[3] == "f32" {
                    DType::F32
                } else {
                    DType::BF16
                },
                m_max: num(4),
                pinned: c[6] != "heuristic",
                invariant: c[7] == "invariant",
            }
        })
        .collect()
}

/// A GEMM of `m` rows on HIP (the whole batch, through the library's own algorithm choice for a
/// prefill or a decode step) checked against the CPU reference on a few of its rows: a row's
/// result does not depend on the other rows, and the full reference of an 8,192-row prefill GEMM
/// is too slow for a test.
fn gemm_rows_case(
    p: &Pair,
    rng: &mut Rng,
    m: usize,
    (b_hip, b_cpu): (&Tensor, &Tensor),
    c_dtype: DType,
    prefill: bool,
) {
    let (n, k) = (b_hip.shape[0], b_hip.shape[1]);
    let a = rng.normal(m * k, 1.0);
    let a_hip = on_hip(p, &[m, k], DType::BF16, &a);
    let c_hip = zeros_on_hip(p, &[m, n], c_dtype);
    let hip = p.hip.gemm().expect("hip gemm");
    let mut ctx = GemmContext {
        a: a_hip.view(),
        b: b_hip.view(),
        c: c_hip.view(),
        trans_b: true,
        alpha: 1.0,
        beta: 0.0,
        prefill,
    };
    hip.execute(&mut ctx).expect("hip gemm");
    let mut sample = vec![0, m / 3, 2 * m / 3, m - 1];
    sample.dedup();
    let rows: Vec<f32> = sample
        .iter()
        .flat_map(|&r| a[r * k..(r + 1) * k].iter().copied())
        .collect();
    let (_, a_cpu) = twin(p, &[sample.len(), k], DType::BF16, &rows);
    let (_, c_cpu) = twin(p, &[sample.len(), n], c_dtype, &vec![0.0; sample.len() * n]);
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
    let want = read(&c_cpu);
    for (i, &r) in sample.iter().enumerate() {
        let got = decode(
            c_dtype,
            &c_hip
                .view()
                .rows(r, 1)
                .slice
                .read_bytes()
                .expect("read row"),
        );
        assert_close(
            &format!("tuned gemm m={m} n={n} k={k} row {r}"),
            "hipblaslt",
            &got,
            &want[i * n..(i + 1) * n],
            c_dtype,
        );
    }
}

/// The card's tuned GEMM table (kernels/rocm/tuning/<arch>/gemm.tsv, compiled into the library):
/// every row, at the smallest and the largest m it serves, runs as written — a pinned row's
/// solution (the context's `TURBINE_OPTION_GEMM_TUNED_SHAPES` grows by one per new shape, so no
/// row fell back with `gemm_table_unavailable` / `_unsupported`), a `heuristic` row hipBLASLt's
/// own answer (the count stays) — and matches the CPU reference; an `invariant` row runs as a
/// prefill step's GEMM (`GemmContext::prefill`), a `speed` row as a decode step's,
/// with the Phase 1 GEMM tolerance. With `TURBINE_OPTION_GEMM_AUTOTUNE` 0 the same shapes run the
/// heuristic's choice (the count stays 0). Breaks if a pinned solution is missing from the
/// installed hipBLASLt, rejects its shape, or computes a wrong result.
#[test]
#[ignore = "needs an R9700 and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn gemm_table_matches_cpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let p = setup();
    let rows = tuned_rows(&p.arch);
    assert!(
        !rows.is_empty(),
        "no tuned GEMM table for {} (kernels/rocm/tuning/{}/gemm.tsv)",
        p.arch,
        p.arch
    );
    assert_eq!(
        p.ctx
            .get_option(TURBINE_OPTION_GEMM_AUTOTUNE)
            .expect("option"),
        1,
        "the tuned table is on by default"
    );
    let tuned = || {
        p.ctx
            .get_option(TURBINE_OPTION_GEMM_TUNED_SHAPES)
            .expect("option")
    };
    let mut rng = Rng(7);
    let mut weights: Option<((usize, usize), Tensor, Tensor)> = None;
    let mut prev: Option<((usize, usize, DType, bool), usize)> = None;
    let mut cases = 0;
    for row in &rows {
        let shape = (row.n, row.k, row.c_dtype, row.invariant);
        let m_min = match prev {
            Some((s, m)) if s == shape => m + 1,
            _ => 1,
        };
        prev = Some((shape, row.m_max));
        if weights.as_ref().is_none_or(|w| w.0 != (row.n, row.k)) {
            let b_scale = 1.0 / (row.k as f32).sqrt();
            let (b_hip, b_cpu) = twin(
                &p,
                &[row.n, row.k],
                DType::BF16,
                &rng.normal(row.n * row.k, b_scale),
            );
            weights = Some(((row.n, row.k), b_hip, b_cpu));
        }
        let (_, b_hip, b_cpu) = weights.as_ref().expect("weights");
        let mut ms = vec![m_min, row.m_max];
        ms.dedup();
        for m in ms {
            let before = tuned();
            gemm_rows_case(&p, &mut rng, m, (b_hip, b_cpu), row.c_dtype, row.invariant);
            assert_eq!(
                tuned(),
                before + i64::from(row.pinned),
                "gemm m={m} n={} k={} {}: the tuned table's row did not run as written (see \
                 the library's gemm_table_fallback line on stderr)",
                row.n,
                row.k,
                row.c_dtype.as_str()
            );
            cases += usize::from(row.pinned);
        }
    }
    println!(
        "gemm_table_matches_cpu: {} rows, {cases} shapes on pinned algorithms",
        rows.len()
    );

    // Off: the heuristic's first answer for the same shapes; the cached choices were dropped.
    p.ctx
        .set_option(TURBINE_OPTION_GEMM_AUTOTUNE, 0)
        .expect("table off");
    assert_eq!(tuned(), 0, "turning the table off drops the pinned choices");
    let row = rows.iter().find(|r| r.pinned).expect("a pinned row");
    let b_scale = 1.0 / (row.k as f32).sqrt();
    let (b_hip, b_cpu) = twin(
        &p,
        &[row.n, row.k],
        DType::BF16,
        &rng.normal(row.n * row.k, b_scale),
    );
    gemm_rows_case(
        &p,
        &mut rng,
        row.m_max,
        (&b_hip, &b_cpu),
        row.c_dtype,
        row.invariant,
    );
    assert_eq!(
        tuned(),
        0,
        "with the table off no shape runs a pinned algorithm"
    );
    p.ctx
        .set_option(TURBINE_OPTION_GEMM_AUTOTUNE, 1)
        .expect("table on");
}
