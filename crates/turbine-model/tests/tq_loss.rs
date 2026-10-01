//! defect? Lab only (HIP device and weights): one prefill of the long golden prompt `p09` is
//! traced, and every layer's post-RoPE K (`k_rope`), V and post-RoPE q (`q_rope`) are run through
//! the CPU reference codec (bit-exact to the GPU transcode). Per layer it prints:
//!
//! - `nmse` of K and V (Σ‖x̂ − x‖² / Σ‖x‖², decoded values rounded to BF16 as the L0 write does),
//!   against the paper's D_mse (√3·π/2·4^−b bound; measured 0.36 / 0.117 / 0.03 / 0.009 for
//!   b = 1 … 4) and, for K, the QJL variant's ≈ (π/2)·D_mse(b − 1);
//! - `rot²`: mean of the rotated, √d/‖x‖-scaled coordinates squared (1 for a correct rotation)
//!   and their kurtosis (3 for the Gaussian the codebook assumes);
//! - `mean`: ‖mean of the head's K‖² / mean ‖k‖² — the share of K that is common to every token;
//! - the score error `e = scale·q·(k̂ − k)` of the last 128 queries over the prefix keys (every
//!   key before the last 128, the lossy part of a reused prefix): its bias and standard
//!   deviation in logits, the score's own spread, and `prod·d` = E[(q·(k̂ − k))²]·d /
//!   E[‖q‖²‖k‖²] against the paper's D_prod·d (1.57, 0.56, 0.18, 0.047 for b = 1 … 4);
//! - the attention output error of those queries (prefix keys and values decoded, the last 128
//!   exact): mean ‖ô − o‖ / ‖o‖ and the mean total variation of the probabilities.
//!
//! Besides the shipped formats it measures K with all its bits in the MSE stage (`k4mse`, `k2mse`:
//! the V path applied to K, no QJL) — the alternative TurboQuant_mse construction for K.
//!
//! Asserts that each format's K and V nmse on real K/V stays within the codec's documented
//! `nmse_bound` and that the rotation is normalised; the table is the evidence.
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use turbine_kernels::{KernelMetrics, KernelRegistry, shim_provider};
use turbine_kv::codec::turboquant::codebook::TQ_DIM;
use turbine_kv::codec::turboquant::hadamard::{SignKind, rademacher, rotate};
use turbine_kv::codec::turboquant::{
    Tq2Codec, Tq4Codec, TqWidths, decode_record, decode_v, encode_record, encode_v, qjl,
};
use turbine_kv::codec::{KvCodec, bf16_to_f32, f32_to_bf16};
use turbine_model::executor::{
    self, DecoderExecutor, ExecutorLimits, ExecutorOptions, SequenceKv, TraceTensor,
};
use turbine_model::{
    MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, families, load_model_config,
};
use turbine_observability::MetricsRegistry;
use turbine_tensor::DeviceMemory;

const BLOCK_TOKENS: u32 = 128;
/// Queries measured (and keys kept exact): the last block of the prompt, like a lossless tail.
const TAIL: usize = 128;
/// The namespace seed of the codec (any value: the loss does not depend on it).
const SEED: u64 = 0x7451_5eed;

#[derive(Deserialize)]
struct ReferenceRecord {
    id: String,
    prompt_token_ids: Vec<u32>,
}

fn prompt(slug: &str, id: &str) -> Vec<u32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(slug)
        .join("reference.jsonl");
    std::fs::read_to_string(&path)
        .expect("reference.jsonl")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<ReferenceRecord>(l).expect("reference line"))
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("no prompt {id} in {}", path.display()))
        .prompt_token_ids
}

fn bf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// How one format turns a K or V vector into its decoded (BF16-rounded) value.
#[derive(Clone, Copy, Debug)]
enum Variant {
    /// A shipped record (`encode_record` / `decode_record`).
    Record(&'static str, TqWidths),
    /// K with every bit in the MSE stage (the V path, no QJL); V as in the record.
    KMse(&'static str, TqWidths, u32),
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Record(n, _) | Variant::KMse(n, _, _) => n,
        }
    }
}

/// Decoded K and V of one (layer, head), `[tokens, d]` each.
fn decode_head(
    v: Variant,
    layer: usize,
    head: usize,
    k: &[f32],
    val: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let d = TQ_DIM;
    let ks = rademacher(SEED, layer as u32, head as u32, SignKind::K, d);
    let vs = rademacher(SEED, layer as u32, head as u32, SignKind::V, d);
    let s = qjl::projection(SEED, layer as u32, head as u32, d);
    let w = match v {
        Variant::Record(_, w) | Variant::KMse(_, w, _) => w,
    };
    let mut rec = vec![0u8; w.record_bytes()];
    let (mut kh, mut vh) = (Vec::with_capacity(k.len()), Vec::with_capacity(val.len()));
    for (kx, vx) in k.chunks_exact(d).zip(val.chunks_exact(d)) {
        encode_record(w, kx, vx, &ks, &vs, &s, &mut rec);
        let (kd, vd) = decode_record(w, &rec, SEED, layer, head);
        let kd = match v {
            Variant::Record(..) => kd,
            Variant::KMse(_, _, bits) => decode_v(&encode_v(kx, bits, &ks), bits, &ks),
        };
        kh.extend(kd.into_iter().map(bf));
        vh.extend(vd.into_iter().map(bf));
    }
    (kh, vh)
}

#[derive(Default, Clone, Copy)]
struct Acc {
    n: f64,
    s: f64,
    s2: f64,
}

impl Acc {
    fn add(&mut self, x: f64) {
        self.n += 1.0;
        self.s += x;
        self.s2 += x * x;
    }
    fn mean(&self) -> f64 {
        self.s / self.n
    }
    fn sd(&self) -> f64 {
        (self.s2 / self.n - self.mean().powi(2)).max(0.0).sqrt()
    }
}

/// A layer's format-independent statistics and its row per variant.
type Row = ((f64, f64, f64, f64), Vec<LayerRow>);

#[derive(Clone, Default)]
struct LayerRow {
    k_nmse: f64,
    v_nmse: f64,
    bias: f64,
    sd: f64,
    prod_d: f64,
    out_rel: f64,
    tv: f64,
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn softmax(s: &[f64]) -> Vec<f64> {
    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

struct Shapes {
    t: usize,
    hq: usize,
    hkv: usize,
    scale: f64,
}

/// Head `h` of `[t, heads·d]` rows, `[t, d]`.
fn head_rows(x: &[f32], heads: usize, h: usize) -> Vec<f32> {
    let d = TQ_DIM;
    x.chunks_exact(heads * d)
        .flat_map(|row| row[h * d..(h + 1) * d].iter().copied())
        .collect()
}

/// One layer, one variant: the row of the table.
fn measure(v: Variant, layer: usize, q: &[f32], k: &[f32], val: &[f32], sh: &Shapes) -> LayerRow {
    let d = TQ_DIM;
    let prefix = sh.t - TAIL;
    let group = sh.hq / sh.hkv;
    let (mut ke, mut kn, mut ve, mut vn) = (0f64, 0f64, 0f64, 0f64);
    let mut err = Acc::default();
    let (mut prod_num, mut prod_den) = (0f64, 0f64);
    let (mut out_rel, mut tv, mut cnt) = (0f64, 0f64, 0f64);
    for h in 0..sh.hkv {
        let kx = head_rows(k, sh.hkv, h);
        let vx = head_rows(val, sh.hkv, h);
        let (mut kh, mut vh) = decode_head(v, layer, h, &kx[..prefix * d], &vx[..prefix * d]);
        for (a, b) in kh.iter().zip(&kx) {
            ke += f64::from(a - b).powi(2);
            kn += f64::from(*b).powi(2);
        }
        for (a, b) in vh.iter().zip(&vx) {
            ve += f64::from(a - b).powi(2);
            vn += f64::from(*b).powi(2);
        }
        // The tail stays exact.
        kh.extend_from_slice(&kx[prefix * d..]);
        vh.extend_from_slice(&vx[prefix * d..]);
        for g in 0..group {
            let qh = head_rows(q, sh.hq, h * group + g);
            for i in prefix..sh.t {
                let qi = &qh[i * d..(i + 1) * d];
                let qn = dot(qi, qi);
                let visible = i + 1;
                let mut s = Vec::with_capacity(visible);
                let mut sh_ = Vec::with_capacity(visible);
                for j in 0..visible {
                    let kj = &kx[j * d..(j + 1) * d];
                    let kjh = &kh[j * d..(j + 1) * d];
                    let (a, b) = (dot(qi, kj), dot(qi, kjh));
                    if j < prefix {
                        err.add(sh.scale * (b - a));
                        prod_num += (b - a).powi(2);
                        prod_den += qn * dot(kj, kj);
                    }
                    s.push(sh.scale * a);
                    sh_.push(sh.scale * b);
                }
                let (p, ph) = (softmax(&s), softmax(&sh_));
                let mut o = vec![0f64; d];
                let mut oh = vec![0f64; d];
                for j in 0..visible {
                    for c in 0..d {
                        o[c] += p[j] * f64::from(vx[j * d + c]);
                        oh[c] += ph[j] * f64::from(vh[j * d + c]);
                    }
                }
                let num: f64 = o.iter().zip(&oh).map(|(a, b)| (a - b).powi(2)).sum();
                let den: f64 = o.iter().map(|a| a * a).sum();
                out_rel += (num / den).sqrt();
                tv += 0.5 * p.iter().zip(&ph).map(|(a, b)| (a - b).abs()).sum::<f64>();
                cnt += 1.0;
            }
        }
    }
    LayerRow {
        k_nmse: ke / kn,
        v_nmse: ve / vn,
        bias: err.mean(),
        sd: err.sd(),
        prod_d: prod_num / prod_den * d as f64,
        out_rel: out_rel / cnt,
        tv: tv / cnt,
    }
}

/// Layer statistics independent of the format: rotated coordinates of K (second moment and
/// kurtosis), the common-mean share of K, and the exact score spread of the measured queries.
fn describe(layer: usize, q: &[f32], k: &[f32], sh: &Shapes) -> (f64, f64, f64, f64) {
    let d = TQ_DIM;
    let (mut m2, mut m4, mut n) = (0f64, 0f64, 0f64);
    let mut mean_share = 0f64;
    let mut spread = Acc::default();
    let group = sh.hq / sh.hkv;
    for h in 0..sh.hkv {
        let kx = head_rows(k, sh.hkv, h);
        let ks = rademacher(SEED, layer as u32, h as u32, SignKind::K, d);
        let mut mu = vec![0f64; d];
        let mut norm2 = 0f64;
        for x in kx.chunks_exact(d) {
            let nx = dot(x, x).sqrt();
            if nx == 0.0 {
                continue;
            }
            for y in rotate(x, &ks) {
                let u = f64::from(y) * (d as f64).sqrt() / nx;
                m2 += u * u;
                m4 += u.powi(4);
                n += 1.0;
            }
            mu.iter_mut().zip(x).for_each(|(m, v)| *m += f64::from(*v));
            norm2 += nx * nx;
        }
        let tokens = (kx.len() / d) as f64;
        let mu2: f64 = mu.iter().map(|m| (m / tokens).powi(2)).sum();
        mean_share += mu2 / (norm2 / tokens);
        for g in 0..group {
            let qh = head_rows(q, sh.hq, h * group + g);
            for i in sh.t - TAIL..sh.t {
                let qi = &qh[i * d..(i + 1) * d];
                let s: Vec<f64> = (0..=i)
                    .map(|j| sh.scale * dot(qi, &kx[j * d..(j + 1) * d]))
                    .collect();
                let m = s.iter().sum::<f64>() / s.len() as f64;
                let v = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / s.len() as f64;
                spread.add(v.sqrt());
            }
        }
    }
    (
        m2 / n,
        (m4 / n) / (m2 / n).powi(2),
        mean_share / sh.hkv as f64,
        spread.mean(),
    )
}

fn get<'a>(trace: &'a [TraceTensor], layer: usize, name: &str) -> &'a TraceTensor {
    trace
        .iter()
        .find(|e| e.layer == Some(layer) && e.name == name)
        .unwrap_or_else(|| panic!("trace lacks {name} of layer {layer}"))
}

fn run(model_dir: &Path, slug: &str) {
    let ctx = turbine_kernels::test_support::open_context("hip");
    let tokens = prompt(slug, "p09");
    let t = tokens.len();
    let cfg = load_model_config(model_dir).expect("config.json");
    assert_eq!(cfg.head_dim as usize, TQ_DIM);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let slots = cfg.family.0.weight_slots(&cfg);
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let provider = shim_provider(ctx.clone());
    let order = [provider.id()];
    let card = provider.card_profile();
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&cfg, BLOCK_TOKENS, ExecutorOptions::default()),
        &KernelMetrics::register(&MetricsRegistry::new()),
        card,
    )
    .expect("every op has a provider");
    let spec = match cfg.family.0.name() {
        "llama" => families::llama::decoder_spec(),
        _ => families::olmoe::decoder_spec(),
    };
    let mut exec = DecoderExecutor::new(
        &cfg,
        spec,
        weights,
        Arc::new(registry),
        mem.clone(),
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: t as u32,
            max_seqs: 1,
        },
        ExecutorOptions::default(),
    )
    .expect("executor");
    let mut kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), t as u32).expect("kv");
    exec.set_trace(true);
    let positions: Vec<u32> = (0..t as u32).collect();
    kv.forward(&mut exec, &tokens, &positions).expect("forward");
    let trace = exec.take_trace();

    let sh = Shapes {
        t,
        hq: cfg.num_attention_heads as usize,
        hkv: cfg.num_kv_heads as usize,
        scale: f64::from(cfg.attention_scale()),
    };
    let variants = [
        Variant::Record("tq4", Tq4Codec::WIDTHS),
        Variant::KMse("tq4/k4mse", Tq4Codec::WIDTHS, 4),
        Variant::Record("tq2", Tq2Codec::WIDTHS),
        Variant::KMse("tq2/k2mse", Tq2Codec::WIDTHS, 2),
    ];
    let layers = cfg.num_layers as usize;
    let rows: Vec<Row> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..layers)
            .map(|l| {
                let (q, k, val) = (
                    &get(&trace, l, "q_rope").data,
                    &get(&trace, l, "k_rope").data,
                    &get(&trace, l, "v").data,
                );
                let sh = &sh;
                s.spawn(move || {
                    let desc = describe(l, q, k, sh);
                    let per: Vec<LayerRow> = variants
                        .iter()
                        .map(|v| measure(*v, l, q, k, val, sh))
                        .collect();
                    (desc, per)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("layer"))
            .collect()
    });
    print_table(slug, t, &variants, &rows);

    let bounds = [Tq4Codec.nmse_bound(), Tq2Codec.nmse_bound()];
    for (l, (desc, per)) in rows.iter().enumerate() {
        assert!(
            (desc.0 - 1.0).abs() < 0.01,
            "layer {l}: rotated second moment {}",
            desc.0
        );
        for (vi, b) in [(0, bounds[0]), (2, bounds[1])] {
            let r = &per[vi];
            assert!(
                r.k_nmse <= b && r.v_nmse <= b,
                "layer {l} {}: K nmse {:.4}, V nmse {:.4} above the codec bound {b}",
                variants[vi].name(),
                r.k_nmse,
                r.v_nmse
            );
        }
    }
}

fn print_table(slug: &str, t: usize, variants: &[Variant], rows: &[Row]) {
    println!(
        "TurboQuant on real K/V: {slug} p09, {t} tokens, last {TAIL} queries over {} lossy prefix keys",
        t - TAIL
    );
    println!(
        "layer  rot2   kurt  mean  s_sd   | per format: K nmse  V nmse  e_bias  e_sd  prod*d  out_rel  tv"
    );
    let mut avg = vec![LayerRow::default(); variants.len()];
    for (l, (d, per)) in rows.iter().enumerate() {
        print!(
            "{l:>5} {:>5.3} {:>5.2} {:>5.3} {:>6.3} |",
            d.0, d.1, d.2, d.3
        );
        for (i, r) in per.iter().enumerate() {
            print!(
                " {}: {:.4} {:.4} {:+.4} {:.3} {:.3} {:.4} {:.4} |",
                variants[i].name(),
                r.k_nmse,
                r.v_nmse,
                r.bias,
                r.sd,
                r.prod_d,
                r.out_rel,
                r.tv
            );
            let a = &mut avg[i];
            a.k_nmse += r.k_nmse;
            a.v_nmse += r.v_nmse;
            a.bias += r.bias;
            a.sd += r.sd;
            a.prod_d += r.prod_d;
            a.out_rel += r.out_rel;
            a.tv += r.tv;
        }
        println!();
    }
    let n = rows.len() as f64;
    for (i, a) in avg.iter().enumerate() {
        println!(
            "mean {:<10} K nmse {:.4}  V nmse {:.4}  e_bias {:+.4}  e_sd {:.3}  prod*d {:.3}  out_rel {:.4}  tv {:.4}",
            variants[i].name(),
            a.k_nmse / n,
            a.v_nmse / n,
            a.bias / n,
            a.sd / n,
            a.prod_d / n,
            a.out_rel / n,
            a.tv / n
        );
    }
}

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and TURBINE_TEST_MODEL_DIR"]
fn tq_loss_on_real_kv_llama() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MODEL_DIR");
    run(&dir, "llama-3.2-3b-instruct");
}

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and TURBINE_TEST_MOE_MODEL_DIR"]
fn tq_loss_on_real_kv_olmoe() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MOE_MODEL_DIR");
    run(&dir, "olmoe-1b-7b-0125-instruct");
}
