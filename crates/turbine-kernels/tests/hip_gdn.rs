//! Lab only (novanas R9700, Phase 7 Task 30, S-11): Gated DeltaNet and the causal conv1d of kernel
//! ABI v2.18, run through `libturbine_hip.so`, match the `cpu-reference` provider
//! (`cpu/gdn.rs`, transformers' recurrence) on seeded inputs at Qwen3.5-9B's shapes (16 key heads,
//! 32 value heads, head dims 128, 8192 conv channels), and keep the properties the hybrid state
//! cache relies on:
//!
//! * a prefill split in two calls with the state carried equals the whole prefill, bit for bit;
//! * `boundary_states` / `boundary_conv_states` at every `B` tokens equal the state of a prefill
//!   stopped there, bit for bit;
//! * a decode step equals a one-token prefill, and the conv update equals the conv prefill token
//!   by token, bit for bit;
//! * a sequence's output and final state do not depend on the other sequences of the batch.
//!
//! Tolerance against the CPU reference: BF16 outputs one BF16 ulp (2^-7 relative) plus 2e-5 (a
//! different f32 summation order flips a BF16 rounding; the conv outputs go through the
//! hardware exp2 / rcp SiLU), F32 states |Δ| ≤ 1e-3. Run by `scripts/lab-test.sh novanas`.
use std::sync::{Arc, Mutex, MutexGuard};

use half::bf16;
use turbine_core::types::{DType, DeviceId};
use turbine_kernels::test_support::{open_context, require_backend};
use turbine_kernels::{
    Conv1dConfig, Conv1dContext, GdnConfig, GdnContext, GdnMode, KernelProvider, OpConfig, OpKind,
    cpu_reference_provider, shim_provider,
};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

/// A GDN shape: `k_heads`, `v_heads`, head dim.
#[derive(Clone, Copy)]
struct Shape {
    kh: usize,
    vh: usize,
    d: usize,
}

/// Qwen3.5-9B's linear-attention layers.
const QWEN: Shape = Shape {
    kh: 16,
    vh: 32,
    d: 128,
};
/// A smaller one for the long CPU comparisons.
const SMALL: Shape = Shape {
    kh: 4,
    vh: 8,
    d: 128,
};
const CHANNELS: usize = 8192;

static GPU: Mutex<()> = Mutex::new(());

fn lock_gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|p| p.into_inner())
}

struct Side {
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
}

fn setup() -> (Side, Side) {
    let ctx = open_context("hip");
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let hip = Side {
        provider: shim_provider(ctx),
        mem,
    };
    let cpu = Side {
        provider: cpu_reference_provider(),
        mem: HostMemory::new(DeviceId(u32::MAX), 4 << 30),
    };
    (hip, cpu)
}

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

    fn uniform(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| lo + (hi - lo) * self.unit() as f32)
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

impl Side {
    fn tensor(&self, shape: &[usize], dtype: DType, data: &[f32]) -> Tensor {
        let mut t = Tensor::empty(&self.mem, shape, dtype).expect("alloc");
        t.storage
            .copy_from_host(0, &encode(dtype, data))
            .expect("upload");
        t
    }

    fn zeros(&self, shape: &[usize], dtype: DType) -> Tensor {
        let n: usize = shape.iter().product();
        self.tensor(shape, dtype, &vec![0.0; n])
    }
}

fn read(t: &Tensor) -> Vec<f32> {
    decode(t.dtype, &t.view().slice.read_bytes().expect("read back"))
}

/// A view of `t` (rank 2, `[rows, width]`) reduced to columns `[col, col + cols)` as the
/// `[rows, heads, dim]` operand of a GDN call: the row stride stays `width`.
fn columns<'a>(t: &'a Tensor, col: usize, heads: usize, dim: usize) -> TensorView<'a> {
    let rows = t.shape[0];
    let width = t.shape[1];
    let es = t.dtype.size_bytes();
    let len = ((rows - 1) * width + heads * dim) * es;
    TensorView {
        slice: t.storage.whole().sub(col * es, len),
        shape: (&[rows, heads, dim][..]).into(),
        strides: (&[width, dim, 1][..]).into(),
        dtype: t.dtype,
    }
}

/// One BF16 ulp of `want` (at most 2^-7 of it) plus a floor far below the outputs' scale (the
/// GDN outputs are ~1e-2): a different f32 summation order may flip one rounding.
fn bf16_close(got: f32, want: f32) -> bool {
    (got - want).abs() <= want.abs() / 128.0 + 2e-5
}

fn assert_bf16_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            bf16_close(g, w),
            "{what}: element {i}: hip {g} vs cpu {w} (|Δ| {})",
            (g - w).abs()
        );
        if (g - w).abs() > worst.1 {
            worst = (i, (g - w).abs());
        }
    }
    println!("{what}: max |Δ| {:.3e} at {} ok", worst.1, worst.0);
}

fn assert_f32_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= tol,
            "{what}: element {i}: hip {g} vs cpu {w} (|Δ| {d})"
        );
        if d > worst.1 {
            worst = (i, d);
        }
    }
    println!("{what}: max |Δ| {:.3e} at {} ok", worst.1, worst.0);
}

fn assert_same(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let differing = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!("{what}: differing={differing}/{}", a.len());
    assert_eq!(differing, 0, "{what}: results must be bit-equal");
}

/// Host data of `tokens` tokens for a GDN call (every value already BF16-exact where it is BF16).
struct GdnData {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

impl GdnData {
    fn new(seed: u64, sh: Shape, tokens: usize) -> GdnData {
        let mut rng = Rng(seed);
        let bf = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect()
        };
        GdnData {
            q: bf(rng.normal(tokens * sh.kh * sh.d, 1.0)),
            k: bf(rng.normal(tokens * sh.kh * sh.d, 1.0)),
            v: bf(rng.normal(tokens * sh.vh * sh.d, 1.0)),
            // A log decay of a typical Qwen3.5 layer: exp(g) in (0.22, 1).
            g: rng.uniform(tokens * sh.vh, -1.5, -0.001),
            beta: rng.uniform(tokens * sh.vh, 0.05, 0.95),
        }
    }

    /// Tokens `[a, b)`.
    fn slice(&self, sh: Shape, a: usize, b: usize) -> GdnData {
        let (kw, vw) = (sh.kh * sh.d, sh.vh * sh.d);
        GdnData {
            q: self.q[a * kw..b * kw].to_vec(),
            k: self.k[a * kw..b * kw].to_vec(),
            v: self.v[a * vw..b * vw].to_vec(),
            g: self.g[a * sh.vh..b * sh.vh].to_vec(),
            beta: self.beta[a * sh.vh..b * sh.vh].to_vec(),
        }
    }

    /// The sequences of `lens` (consecutive tokens from `at`) concatenated in `order`.
    fn gather(&self, sh: Shape, starts: &[usize], lens: &[usize], order: &[usize]) -> GdnData {
        let mut out = GdnData {
            q: vec![],
            k: vec![],
            v: vec![],
            g: vec![],
            beta: vec![],
        };
        for &s in order {
            let part = self.slice(sh, starts[s], starts[s] + lens[s]);
            out.q.extend(part.q);
            out.k.extend(part.k);
            out.v.extend(part.v);
            out.g.extend(part.g);
            out.beta.extend(part.beta);
        }
        out
    }
}

struct GdnOut {
    out: Vec<f32>,
    state: Vec<f32>,
    boundary: Vec<f32>,
}

fn gdn_cfg(sh: Shape, mode: GdnMode) -> GdnConfig {
    GdnConfig {
        k_heads: sh.kh as u32,
        v_heads: sh.vh as u32,
        k_dim: sh.d as u32,
        v_dim: sh.d as u32,
        mode,
        dtype: DType::BF16,
    }
}

/// One GDN call on `side`: sequences of `lens` over `data`, sequence `s` in `slots[s]` of a state
/// holding `state0` (`[slots, v_heads, d, d]`), snapshots every `boundary` tokens (0 = none).
#[allow(clippy::too_many_arguments)]
fn gdn_call(
    side: &Side,
    sh: Shape,
    mode: GdnMode,
    data: &GdnData,
    lens: &[usize],
    slots: &[i32],
    state0: &[f32],
    boundary: u32,
) -> GdnOut {
    let tokens: usize = lens.iter().sum();
    let nslots = state0.len() / (sh.vh * sh.d * sh.d);
    let q = side.tensor(&[tokens, sh.kh, sh.d], DType::BF16, &data.q);
    let k = side.tensor(&[tokens, sh.kh, sh.d], DType::BF16, &data.k);
    let v = side.tensor(&[tokens, sh.vh, sh.d], DType::BF16, &data.v);
    let g = side.tensor(&[tokens, sh.vh], DType::F32, &data.g);
    let beta = side.tensor(&[tokens, sh.vh], DType::F32, &data.beta);
    let state = side.tensor(&[nslots, sh.vh, sh.d, sh.d], DType::F32, state0);
    let ids = side.tensor(
        &[slots.len()],
        DType::I32,
        &slots.iter().map(|&s| s as f32).collect::<Vec<_>>(),
    );
    let mut cu = vec![0f32];
    for l in lens {
        cu.push(cu.last().copied().unwrap_or(0.0) + *l as f32);
    }
    let cu_t = side.tensor(&[cu.len()], DType::I32, &cu);
    let out = side.zeros(&[tokens, sh.vh, sh.d], DType::BF16);
    let rows: usize = if boundary > 0 {
        lens.iter().map(|l| l / boundary as usize).sum()
    } else {
        0
    };
    let bs = (rows > 0).then(|| side.zeros(&[rows, sh.vh, sh.d, sh.d], DType::F32));
    side.provider
        .gdn()
        .expect("gdn family")
        .execute(&mut GdnContext {
            cfg: gdn_cfg(sh, mode),
            q: q.view(),
            k: k.view(),
            v: v.view(),
            g: g.view(),
            beta: beta.view(),
            state: state.view(),
            state_slots: ids.view(),
            cu_seqlens: (mode == GdnMode::Prefill).then(|| cu_t.view()),
            out: out.view(),
            boundary_states: bs.as_ref().map(Tensor::view),
            boundary_tokens: boundary,
        })
        .expect("gdn");
    GdnOut {
        out: read(&out),
        state: read(&state),
        boundary: bs.as_ref().map(read).unwrap_or_default(),
    }
}

fn state_len(sh: Shape) -> usize {
    sh.vh * sh.d * sh.d
}

fn require_hip() -> bool {
    require_backend("hip")
}

#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn gdn_support_and_implementations() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, _) = setup();
    for mode in [GdnMode::Prefill, GdnMode::Decode] {
        let kernel = hip.provider.gdn().expect("the library exports v2.18");
        let cfg = gdn_cfg(QWEN, mode);
        assert!(kernel.supports(&cfg), "{cfg}");
        println!("gdn {cfg}: impl={}", kernel.implementation(&cfg));
        let conv = Conv1dConfig {
            channels: CHANNELS as u32,
            kernel: 4,
            mode,
            dtype: DType::BF16,
        };
        let kernel = hip.provider.conv1d().expect("the library exports v2.18");
        assert!(kernel.supports(&conv), "{conv}");
        println!("conv1d {conv}: impl={}", kernel.implementation(&conv));
        assert!(OpConfig::Gdn(cfg).supported_by(hip.provider.as_ref()));
    }
    // Head dims outside {32, 64, 128}, F32 operands and a non-dividing head ratio are refused.
    let kernel = hip.provider.gdn().expect("gdn");
    let mut cfg = gdn_cfg(QWEN, GdnMode::Prefill);
    cfg.k_dim = 96;
    cfg.v_dim = 96;
    assert!(!kernel.supports(&cfg));
    cfg = gdn_cfg(QWEN, GdnMode::Prefill);
    cfg.dtype = DType::F32;
    assert!(!kernel.supports(&cfg));
    cfg = gdn_cfg(QWEN, GdnMode::Prefill);
    cfg.v_heads = 24;
    assert!(!kernel.supports(&cfg));
    assert_eq!(OpKind::GdnPrefill.abi_code(), 20);
}

#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn gdn_prefill_matches_reference() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, cpu) = setup();
    // Lengths 1, 63, 64, 65 at Qwen3.5-9B's shape, 2048 at the smaller one (the CPU reference
    // runs the whole recurrence in f32).
    for (sh, len) in [
        (QWEN, 1usize),
        (QWEN, 63),
        (QWEN, 64),
        (QWEN, 65),
        (SMALL, 2048),
    ] {
        let data = GdnData::new(100 + len as u64, sh, len);
        let zero = vec![0f32; state_len(sh)];
        let got = gdn_call(&hip, sh, GdnMode::Prefill, &data, &[len], &[0], &zero, 0);
        let want = gdn_call(&cpu, sh, GdnMode::Prefill, &data, &[len], &[0], &zero, 0);
        assert_bf16_close(&format!("gdn prefill len {len} out"), &got.out, &want.out);
        assert_f32_close(
            &format!("gdn prefill len {len} state"),
            &got.state,
            &want.state,
            1e-3,
        );
    }
    split_and_boundary_checks(&hip);
}

/// A prefill split at 1000 equals the whole one, and the snapshots every 128 tokens equal the
/// state of a prefill stopped there (one sequence, then three with unaligned lengths).
fn split_and_boundary_checks(hip: &Side) {
    let sh = QWEN;
    let len = 2048;
    let data = GdnData::new(7, sh, len);
    let zero = vec![0f32; state_len(sh)];
    let whole = gdn_call(hip, sh, GdnMode::Prefill, &data, &[len], &[0], &zero, 0);

    // A chunked prefill split at 1000 (not a multiple of any chunk size), the state carried.
    let first = gdn_call(
        hip,
        sh,
        GdnMode::Prefill,
        &data.slice(sh, 0, 1000),
        &[1000],
        &[0],
        &zero,
        0,
    );
    let second = gdn_call(
        hip,
        sh,
        GdnMode::Prefill,
        &data.slice(sh, 1000, len),
        &[len - 1000],
        &[0],
        &first.state,
        0,
    );
    let mut out = first.out.clone();
    out.extend(&second.out);
    assert_same("split at 1000: out", &out, &whole.out);
    assert_same("split at 1000: state", &second.state, &whole.state);

    // Snapshots every 128 tokens equal the state of a prefill stopped there.
    let with = gdn_call(hip, sh, GdnMode::Prefill, &data, &[len], &[0], &zero, 128);
    assert_same("boundary run: out", &with.out, &whole.out);
    assert_same("boundary run: state", &with.state, &whole.state);
    assert_eq!(with.boundary.len(), 16 * state_len(sh));
    for j in 0..16 {
        let stop = (j + 1) * 128;
        let stopped = gdn_call(
            hip,
            sh,
            GdnMode::Prefill,
            &data.slice(sh, 0, stop),
            &[stop],
            &[0],
            &zero,
            0,
        );
        assert_same(
            &format!("boundary state after {stop} tokens"),
            &with.boundary[j * state_len(sh)..(j + 1) * state_len(sh)],
            &stopped.state,
        );
    }

    // Several sequences: boundaries are counted from each sequence's start, in sequence order.
    let lens = [300usize, 127, 513];
    let total: usize = lens.iter().sum();
    let starts = [0usize, 300, 427];
    let multi = data.slice(sh, 0, total);
    let zero3 = vec![0f32; 3 * state_len(sh)];
    let run = gdn_call(
        hip,
        sh,
        GdnMode::Prefill,
        &multi,
        &lens,
        &[0, 1, 2],
        &zero3,
        128,
    );
    let mut row = 0;
    for (s, &l) in lens.iter().enumerate() {
        for j in 0..l / 128 {
            let stop = (j + 1) * 128;
            let part = multi.slice(sh, starts[s], starts[s] + stop);
            let stopped = gdn_call(hip, sh, GdnMode::Prefill, &part, &[stop], &[0], &zero, 0);
            assert_same(
                &format!("multi-sequence boundary seq {s} after {stop}"),
                &run.boundary[row * state_len(sh)..(row + 1) * state_len(sh)],
                &stopped.state,
            );
            row += 1;
        }
    }
    assert_eq!(run.boundary.len(), row * state_len(sh));
}

#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn gdn_decode_matches_reference() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, cpu) = setup();
    let sh = QWEN;
    let seqs = 5;
    let data = GdnData::new(21, sh, seqs);
    // Random current states, sequences in non-adjacent slots of a pool of 8.
    let mut rng = Rng(22);
    let pool = rng.normal(8 * state_len(sh), 0.3);
    let slots = [6i32, 1, 4, 0, 7];
    let got = gdn_call(&hip, sh, GdnMode::Decode, &data, &[1; 5], &slots, &pool, 0);
    let want = gdn_call(&cpu, sh, GdnMode::Decode, &data, &[1; 5], &slots, &pool, 0);
    assert_bf16_close("gdn decode out", &got.out, &want.out);
    assert_f32_close("gdn decode state", &got.state, &want.state, 1e-3);
    // The slots no sequence names stay untouched.
    for slot in [2usize, 3, 5] {
        let r = slot * state_len(sh)..(slot + 1) * state_len(sh);
        assert_same("untouched slot", &got.state[r.clone()], &pool[r]);
    }

    // A decode step is a one-token prefill, bit for bit.
    for s in 0..seqs {
        let one = data.slice(sh, s, s + 1);
        let st = pool[slots[s] as usize * state_len(sh)..(slots[s] as usize + 1) * state_len(sh)]
            .to_vec();
        let pre = gdn_call(&hip, sh, GdnMode::Prefill, &one, &[1], &[0], &st, 0);
        let o = s * sh.vh * sh.d..(s + 1) * sh.vh * sh.d;
        assert_same("decode vs 1-token prefill out", &got.out[o], &pre.out);
        let r = slots[s] as usize * state_len(sh)..(slots[s] as usize + 1) * state_len(sh);
        assert_same("decode vs 1-token prefill state", &got.state[r], &pre.state);
    }
}

#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn gdn_batch_invariant() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, _) = setup();
    let sh = QWEN;
    let lens = [70usize, 1, 333];
    let starts = [0usize, 70, 71];
    let total: usize = lens.iter().sum();
    let data = GdnData::new(31, sh, total);
    let mut rng = Rng(32);
    let pool = rng.normal(3 * state_len(sh), 0.2);
    let slot_state = |s: usize| pool[s * state_len(sh)..(s + 1) * state_len(sh)].to_vec();
    // Each sequence alone, from its own state.
    let alone: Vec<GdnOut> = (0..3)
        .map(|s| {
            gdn_call(
                &hip,
                sh,
                GdnMode::Prefill,
                &data.slice(sh, starts[s], starts[s] + lens[s]),
                &[lens[s]],
                &[0],
                &slot_state(s),
                0,
            )
        })
        .collect();
    // The same sequences in other batch compositions, orders and slots.
    for order in [vec![0usize, 1, 2], vec![2, 0, 1], vec![1, 2], vec![2, 1, 0]] {
        let batch = data.gather(sh, &starts, &lens, &order);
        let batch_lens: Vec<usize> = order.iter().map(|&s| lens[s]).collect();
        let slots: Vec<i32> = order.iter().map(|&s| s as i32).collect();
        let run = gdn_call(
            &hip,
            sh,
            GdnMode::Prefill,
            &batch,
            &batch_lens,
            &slots,
            &pool,
            0,
        );
        let mut at = 0;
        for &s in &order {
            let w = sh.vh * sh.d;
            assert_same(
                &format!("batch {order:?}: seq {s} out"),
                &run.out[at * w..(at + lens[s]) * w],
                &alone[s].out,
            );
            assert_same(
                &format!("batch {order:?}: seq {s} state"),
                &run.state[s * state_len(sh)..(s + 1) * state_len(sh)],
                &alone[s].state,
            );
            at += lens[s];
        }
    }
}

/// Host data of a conv call: `x` `[tokens, CHANNELS]` BF16-exact, the weights and the bias.
struct ConvData {
    w: Vec<f32>,
    bias: Vec<f32>,
}

fn conv_data(seed: u64) -> ConvData {
    let mut rng = Rng(seed);
    let bf =
        |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect() };
    ConvData {
        w: bf(rng.normal(CHANNELS * 4, 0.5)),
        bias: bf(rng.normal(CHANNELS, 0.1)),
    }
}

fn conv_x(seed: u64, tokens: usize) -> Vec<f32> {
    Rng(seed)
        .normal(tokens * CHANNELS, 1.0)
        .into_iter()
        .map(|x| bf16::from_f32(x).to_f32())
        .collect()
}

struct ConvOut {
    out: Vec<f32>,
    state: Vec<f32>,
    boundary: Vec<f32>,
}

fn conv_cfg(mode: GdnMode) -> Conv1dConfig {
    Conv1dConfig {
        channels: CHANNELS as u32,
        kernel: 4,
        mode,
        dtype: DType::BF16,
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_call(
    side: &Side,
    mode: GdnMode,
    cd: &ConvData,
    x: &[f32],
    lens: &[usize],
    slots: &[i32],
    state0: &[f32],
    boundary: u32,
) -> ConvOut {
    let tokens: usize = lens.iter().sum();
    let nslots = state0.len() / (CHANNELS * 3);
    let xt = side.tensor(&[tokens, CHANNELS], DType::BF16, x);
    let w = side.tensor(&[CHANNELS, 4], DType::BF16, &cd.w);
    let bias = side.tensor(&[CHANNELS], DType::BF16, &cd.bias);
    let state = side.tensor(&[nslots, CHANNELS, 3], DType::BF16, state0);
    let ids = side.tensor(
        &[slots.len()],
        DType::I32,
        &slots.iter().map(|&s| s as f32).collect::<Vec<_>>(),
    );
    let mut cu = vec![0f32];
    for l in lens {
        cu.push(cu.last().copied().unwrap_or(0.0) + *l as f32);
    }
    let cu_t = side.tensor(&[cu.len()], DType::I32, &cu);
    let out = side.zeros(&[tokens, CHANNELS], DType::BF16);
    let rows: usize = if boundary > 0 {
        lens.iter().map(|l| l / boundary as usize).sum()
    } else {
        0
    };
    let bs = (rows > 0).then(|| side.zeros(&[rows, CHANNELS, 3], DType::BF16));
    side.provider
        .conv1d()
        .expect("conv1d family")
        .execute(&mut Conv1dContext {
            cfg: conv_cfg(mode),
            x: xt.view(),
            weight: w.view(),
            bias: Some(bias.view()),
            conv_state: state.view(),
            state_slots: ids.view(),
            cu_seqlens: (mode == GdnMode::Prefill).then(|| cu_t.view()),
            out: out.view(),
            boundary_conv_states: bs.as_ref().map(Tensor::view),
            boundary_tokens: boundary,
            activation: 1,
        })
        .expect("conv1d");
    ConvOut {
        out: read(&out),
        state: read(&state),
        boundary: bs.as_ref().map(read).unwrap_or_default(),
    }
}

#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn conv1d_state_update_matches_reference() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, cpu) = setup();
    let cd = conv_data(41);
    // Prefill lengths around the state size (1, 2, 3), a chunk edge (16, 17) and a long one, from
    // a nonzero stored tail, in one batch.
    let lens = [1usize, 2, 3, 5, 16, 17, 64, 257];
    let total: usize = lens.iter().sum();
    let x = conv_x(42, total);
    // [slots, CHANNELS, 3] BF16-exact random tails.
    let mut state0 = Rng(43).normal(lens.len() * CHANNELS * 3, 1.0);
    state0
        .iter_mut()
        .for_each(|v| *v = bf16::from_f32(*v).to_f32());
    let slots: Vec<i32> = (0..lens.len() as i32).rev().collect();
    let got = conv_call(&hip, GdnMode::Prefill, &cd, &x, &lens, &slots, &state0, 0);
    let want = conv_call(&cpu, GdnMode::Prefill, &cd, &x, &lens, &slots, &state0, 0);
    assert_bf16_close("conv1d prefill out", &got.out, &want.out);
    // The new tails are copies of inputs and of the old tails: exact.
    assert_same("conv1d prefill state", &got.state, &want.state);

    // The update op, one token per sequence, equals the CPU and the prefill token by token.
    let steps = 9;
    let xs = conv_x(44, steps);
    let mut st_hip = state0[..3 * CHANNELS].to_vec();
    let mut st_cpu = st_hip.clone();
    let mut pre_out = Vec::new();
    for t in 0..steps {
        let xt = &xs[t * CHANNELS..(t + 1) * CHANNELS];
        let a = conv_call(&hip, GdnMode::Decode, &cd, xt, &[1], &[0], &st_hip, 0);
        let b = conv_call(&cpu, GdnMode::Decode, &cd, xt, &[1], &[0], &st_cpu, 0);
        assert_bf16_close(&format!("conv1d update step {t}"), &a.out, &b.out);
        assert_same(&format!("conv1d update state {t}"), &a.state, &b.state);
        pre_out.extend(a.out);
        st_hip = a.state;
        st_cpu = b.state;
    }
    let pre = conv_call(
        &hip,
        GdnMode::Prefill,
        &cd,
        &xs[..steps * CHANNELS],
        &[steps],
        &[0],
        &state0[..3 * CHANNELS],
        0,
    );
    assert_same("conv1d update vs prefill out", &pre_out, &pre.out);
    assert_same("conv1d update vs prefill state", &st_hip, &pre.state);

    // Snapshots every 64 tokens equal the tail of a prefill stopped there.
    let n = 300;
    let xb = conv_x(45, n);
    let zero = vec![0f32; 3 * CHANNELS];
    let with = conv_call(&hip, GdnMode::Prefill, &cd, &xb, &[n], &[0], &zero, 64);
    assert_eq!(with.boundary.len(), 4 * CHANNELS * 3);
    for j in 0..4 {
        let stop = (j + 1) * 64;
        let stopped = conv_call(
            &hip,
            GdnMode::Prefill,
            &cd,
            &xb[..stop * CHANNELS],
            &[stop],
            &[0],
            &zero,
            0,
        );
        assert_same(
            &format!("conv boundary after {stop}"),
            &with.boundary[j * CHANNELS * 3..(j + 1) * CHANNELS * 3],
            &stopped.state,
        );
    }
}

#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn conv1d_batch_invariant() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, _) = setup();
    let cd = conv_data(51);
    let lens = [37usize, 1, 130];
    let starts = [0usize, 37, 38];
    let total: usize = lens.iter().sum();
    let x = conv_x(52, total);
    let mut pool = Rng(53).normal(3 * CHANNELS * 3, 1.0);
    pool.iter_mut()
        .for_each(|v| *v = bf16::from_f32(*v).to_f32());
    let alone: Vec<ConvOut> = (0..3)
        .map(|s| {
            let st = pool[s * CHANNELS * 3..(s + 1) * CHANNELS * 3].to_vec();
            conv_call(
                &hip,
                GdnMode::Prefill,
                &cd,
                &x[starts[s] * CHANNELS..(starts[s] + lens[s]) * CHANNELS],
                &[lens[s]],
                &[0],
                &st,
                0,
            )
        })
        .collect();
    for order in [vec![0usize, 1, 2], vec![2, 1, 0], vec![1, 2]] {
        let mut bx = Vec::new();
        for &s in &order {
            bx.extend(&x[starts[s] * CHANNELS..(starts[s] + lens[s]) * CHANNELS]);
        }
        let batch_lens: Vec<usize> = order.iter().map(|&s| lens[s]).collect();
        let slots: Vec<i32> = order.iter().map(|&s| s as i32).collect();
        let run = conv_call(
            &hip,
            GdnMode::Prefill,
            &cd,
            &bx,
            &batch_lens,
            &slots,
            &pool,
            0,
        );
        let mut at = 0;
        for &s in &order {
            assert_same(
                &format!("conv batch {order:?}: seq {s} out"),
                &run.out[at * CHANNELS..(at + lens[s]) * CHANNELS],
                &alone[s].out,
            );
            assert_same(
                &format!("conv batch {order:?}: seq {s} state"),
                &run.state[s * CHANNELS * 3..(s + 1) * CHANNELS * 3],
                &alone[s].state,
            );
            at += lens[s];
        }
    }
}

/// The conv output feeds the GDN in place: `q | k | v` are column ranges of the packed
/// `[tokens, 8192]` row (16 · 128 + 16 · 128 + 32 · 128), read with a row stride, equal to the
/// same data copied into dense tensors.
#[test]
#[ignore = "needs the HIP backend (scripts/lab-test.sh)"]
fn gdn_reads_the_conv_output_in_place() {
    if !require_hip() {
        return;
    }
    let _gpu = lock_gpu();
    let (hip, cpu) = setup();
    let sh = QWEN;
    let len = 90;
    let cd = conv_data(61);
    let x = conv_x(62, len);
    let zero_conv = vec![0f32; 3 * CHANNELS];
    let conv = conv_call(&hip, GdnMode::Prefill, &cd, &x, &[len], &[0], &zero_conv, 0);
    let (qw, kw) = (sh.kh * sh.d, sh.kh * sh.d);
    // Dense copies of the three column ranges.
    let mut dense = GdnData::new(63, sh, len);
    dense.q.clear();
    dense.k.clear();
    dense.v.clear();
    for t in 0..len {
        let row = &conv.out[t * CHANNELS..(t + 1) * CHANNELS];
        dense.q.extend(&row[..qw]);
        dense.k.extend(&row[qw..qw + kw]);
        dense.v.extend(&row[qw + kw..]);
    }
    let zero = vec![0f32; state_len(sh)];
    let want = gdn_call(&hip, sh, GdnMode::Prefill, &dense, &[len], &[0], &zero, 0);
    let on_cpu = gdn_call(&cpu, sh, GdnMode::Prefill, &dense, &[len], &[0], &zero, 0);
    assert_bf16_close("dense q|k|v vs cpu", &want.out, &on_cpu.out);

    // In place on the device: the conv output tensor itself.
    let packed = hip.tensor(&[len, CHANNELS], DType::BF16, &conv.out);
    let g = hip.tensor(&[len, sh.vh], DType::F32, &dense.g);
    let beta = hip.tensor(&[len, sh.vh], DType::F32, &dense.beta);
    let state = hip.tensor(&[1, sh.vh, sh.d, sh.d], DType::F32, &zero);
    let ids = hip.tensor(&[1], DType::I32, &[0.0]);
    let cu = hip.tensor(&[2], DType::I32, &[0.0, len as f32]);
    let out = hip.zeros(&[len, sh.vh, sh.d], DType::BF16);
    hip.provider
        .gdn()
        .expect("gdn")
        .execute(&mut GdnContext {
            cfg: gdn_cfg(sh, GdnMode::Prefill),
            q: columns(&packed, 0, sh.kh, sh.d),
            k: columns(&packed, qw, sh.kh, sh.d),
            v: columns(&packed, qw + kw, sh.vh, sh.d),
            g: g.view(),
            beta: beta.view(),
            state: state.view(),
            state_slots: ids.view(),
            cu_seqlens: Some(cu.view()),
            out: out.view(),
            boundary_states: None,
            boundary_tokens: 0,
        })
        .expect("gdn on strided views");
    assert_same("in-place q|k|v out", &read(&out), &want.out);
    assert_same("in-place q|k|v state", &read(&state), &want.state);
}
