//! `codec::turboquant::tests` (P6b S-4 acceptance): rotation, codebooks, the K inner-product
//! estimator, the V MSE against the paper's D_mse and the packed layout.

use super::codebook::*;
use super::hadamard::*;
use super::*;
use crate::codec::tests::{Rng, block, layout};
use turbine_core::types::DType;

fn gaussian(rng: &mut Rng) -> Vec<f32> {
    (0..TQ_DIM).map(|_| rng.normal() as f32).collect()
}

fn unit(x: Vec<f32>) -> Vec<f32> {
    let n = norm(&x);
    x.into_iter().map(|v| v / n).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// Signs of pair / vector `i`: every one its own (seed, layer, head), as blocks of different
/// namespaces, layers and heads are.
fn signs_for(i: usize) -> HeadSigns {
    HeadSigns::of(0x7e57 ^ (i as u64 * 0x9e37_79b9), i % 28, i % 8)
}

/// Mean, standard deviation and standard error of `v`.
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt(), (var / n).sqrt())
}

/// `tq4` and `tq2` K estimates `⟨q, k̂⟩` of `⟨q, k⟩` are unbiased over 10,000 seeded pairs of unit
/// vectors with ⟨q, k⟩ ≈ 0.71 (|mean error| < 3 standard errors), and their variance stays
/// within the paper's D_prod bound (√3·π²/d)·4^−b. The pairs are correlated so that an
/// MSE-only estimate (no QJL term), which shrinks every inner product by ≈ D_mse, is far
/// outside 3 standard errors. Breaks if the QJL residual is dropped or mis-scaled.
#[test]
fn k_inner_product_unbiased() {
    for (name, w) in [("tq4", Tq4Codec::WIDTHS), ("tq2", Tq2Codec::WIDTHS)] {
        let mut rng = Rng(0xbad5eed);
        let mut err = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            let k = unit(gaussian(&mut rng));
            let noise = gaussian(&mut rng);
            let q = unit(k.iter().zip(&noise).map(|(a, b)| a + b / 11.3).collect());
            let s = signs_for(i);
            let enc = encode_k(&k, w.k_bits, &s.k, &s.qjl);
            let khat = decode_k(&enc, w.k_bits, &s.k, &s.qjl);
            err.push(dot(&q, &khat) - dot(&q, &k));
        }
        let (mean, sd, se) = stats(&err);
        let b = w.k_bits as i32 + 1;
        let bound = 3f64.sqrt() * std::f64::consts::PI.powi(2) / TQ_DIM as f64 / 4f64.powi(b);
        println!(
            "{name} K: mean error {mean:.3e}, 3·SE {:.3e}, variance {:.3e} (D_prod bound {bound:.3e})",
            3.0 * se,
            sd * sd
        );
        assert!(
            mean.abs() < 3.0 * se,
            "{name}: biased, mean {mean} vs SE {se}"
        );
        assert!(
            sd * sd <= bound,
            "{name}: variance {} above {bound}",
            sd * sd
        );
    }
}

/// V reconstruction MSE of unit-normalised vectors, ‖x − x̂‖²/‖x‖² averaged over 4,000 seeded
/// Gaussian vectors of random norms, for `bits`: within 10 % of the paper's D_mse and below
/// the theorem's bound (√3·π/2)·4^−b.
fn v_mse(bits: u32, paper: f64) {
    let mut rng = Rng(0x5eed_0000 + u64::from(bits));
    let mut e = Vec::new();
    for i in 0..4_000 {
        let scale = (rng.normal() * 2.0).exp() as f32;
        let x: Vec<f32> = gaussian(&mut rng).into_iter().map(|v| v * scale).collect();
        let s = signs_for(i);
        let xhat = decode_v(&encode_v(&x, bits, &s.v), bits, &s.v);
        let d: Vec<f32> = x.iter().zip(&xhat).map(|(a, b)| a - b).collect();
        e.push(dot(&d, &d) / dot(&x, &x));
    }
    let (mean, _, se) = stats(&e);
    let theorem = 3f64.sqrt() * std::f64::consts::PI / 2.0 / 4f64.powi(bits as i32);
    println!("V {bits}-bit: MSE {mean:.5} ± {se:.5} (paper {paper}, theorem bound {theorem:.5})");
    assert!(
        (mean - paper).abs() <= 0.1 * paper,
        "{bits}-bit MSE {mean} vs paper {paper}"
    );
    assert!(mean <= theorem);
}

/// `tq4`'s V (4 bits): MSE within 10 % of the paper's 0.009. Breaks with a wrong codebook,
/// norm or rotation.
#[test]
fn v_mse_bound_4bit() {
    v_mse(4, 0.009);
}

/// `tq2`'s V (2 bits): MSE within 10 % of the paper's 0.117.
#[test]
fn v_mse_bound_2bit() {
    v_mse(2, 0.117);
}

/// The packed block holds exactly the per-vector encodings at the documented offsets: records
/// of 144 (`tq4`) / 80 (`tq2`) bytes, 16-byte aligned, per layer, per head, per token; zero
/// padding; codes packed LSB first; decoding the block equals decoding each vector. Breaks if a
/// field moves, the record order changes or padding carries data.
#[test]
fn layout_round_trip() {
    assert_eq!(Tq4Codec::WIDTHS.record_bytes(), 144);
    assert_eq!(Tq2Codec::WIDTHS.record_bytes(), 80);
    let mut rng = Rng(99);
    for bits in 1..=4u32 {
        let codes: Vec<u8> = (0..TQ_DIM)
            .map(|_| (rng.next_u64() % (1 << bits)) as u8)
            .collect();
        let packed = pack(&codes, bits);
        assert_eq!(packed.len(), TQ_DIM * bits as usize / 8);
        assert_eq!(unpack(&packed, bits, TQ_DIM), codes);
    }
    assert_eq!(pack(&[1, 2, 3, 0], 2), [0b0011_1001]);

    let p = crate::codec::CodecParams {
        seed: 0xfeed,
        ..Default::default()
    };
    let l0 = layout(DType::BF16);
    let g = crate::codec::L0Geometry::of(&l0);
    let src = block(&l0, 5, true, &p);
    for codec in [&Tq4Codec as &dyn KvCodec, &Tq2Codec] {
        let w = if codec.name() == "tq4" {
            Tq4Codec::WIDTHS
        } else {
            Tq2Codec::WIDTHS
        };
        let rec = w.record_bytes();
        assert_eq!(
            codec.bytes_per_block(&l0),
            (g.layers * g.heads * g.tokens * rec) as u64
        );
        let mut enc = vec![0u8; codec.bytes_per_block(&l0) as usize];
        codec.encode_cpu(&src, &l0, &mut enc, &p).unwrap();
        let mut dec = vec![0u8; l0.block_bytes() as usize];
        codec.decode_cpu(&enc, &l0, &mut dec, &p).unwrap();
        let f = Fields::of(w);
        let vec_of = |buf: &[u8], layer, kind, t, h| -> Vec<f32> {
            let base = g.vector_elem(layer, kind, t, h);
            (base..base + TQ_DIM)
                .map(|i| read_l0(buf, l0.dtype, i, 1.0))
                .collect()
        };
        for layer in 0..g.layers {
            for h in 0..g.heads {
                let s = HeadSigns::of(p.seed, layer, h);
                for t in 0..g.tokens {
                    let at = ((layer * g.heads + h) * g.tokens + t) * rec;
                    assert_eq!(at % 16, 0);
                    let r = &enc[at..at + rec];
                    let k = encode_k(&vec_of(&src, layer, 0, t, h), w.k_bits, &s.k, &s.qjl);
                    let v = encode_v(&vec_of(&src, layer, 1, t, h), w.v_bits, &s.v);
                    assert_eq!(&r[..f.k_norm], &k.codes[..]);
                    assert_eq!(get_u16(r, f.k_norm), k.norm);
                    assert_eq!(&r[f.qjl..f.r_norm], &k.qjl[..]);
                    assert_eq!(get_u16(r, f.r_norm), k.residual_norm);
                    assert_eq!(&r[f.v_codes..f.v_norm], &v.codes[..]);
                    assert_eq!(get_u16(r, f.v_norm), v.norm);
                    assert!(r[f.v_norm + 2..].iter().all(|b| *b == 0), "padding");
                    for (kind, x) in [
                        (0, decode_k(&k, w.k_bits, &s.k, &s.qjl)),
                        (1, decode_v(&v, w.v_bits, &s.v)),
                    ] {
                        let want: Vec<f32> =
                            x.iter().map(|v| bf16_to_f32(f32_to_bf16(*v))).collect();
                        assert_eq!(vec_of(&dec, layer, kind, t, h), want);
                    }
                }
            }
        }
    }
    // Capacity against a BF16 L0 (512 bytes per token-head K+V): ≥ 3.5× and ≥ 6× (targets).
    let ratio = |c: &dyn KvCodec| l0.block_bytes() as f64 / c.bytes_per_block(&l0) as f64;
    assert!(ratio(&Tq4Codec) >= 3.5, "{}", ratio(&Tq4Codec));
    assert!(ratio(&Tq2Codec) >= 6.0, "{}", ratio(&Tq2Codec));
}

/// The rotation's matrix is orthonormal (R·Rᵀ = I within F32 rounding), `unrotate` inverts
/// it, norms are kept, and the signs are a pure function of (seed, layer, head, kind) that
/// differs when any of them does. Breaks if the transform is unnormalised, the inverse
/// misses the signs, or the sign stream ignores an input.
#[test]
fn hadamard_orthonormal() {
    let d = 128;
    let s = rademacher(42, 3, 5, SignKind::K, d);
    assert_eq!(s.len(), d);
    assert!(s.iter().all(|v| *v == 1.0 || *v == -1.0));
    let plus = s.iter().filter(|v| **v > 0.0).count();
    assert!((40..=88).contains(&plus), "{plus} of 128 signs are +1");
    // Columns of R are the rotations of the basis vectors.
    let cols: Vec<Vec<f32>> = (0..d)
        .map(|i| {
            let mut e = vec![0f32; d];
            e[i] = 1.0;
            rotate(&e, &s)
        })
        .collect();
    for i in 0..d {
        for j in 0..d {
            let dot: f64 = (0..d)
                .map(|k| f64::from(cols[i][k]) * f64::from(cols[j][k]))
                .sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-5, "R col {i}·col {j} = {dot}");
        }
    }
    let x: Vec<f32> = (0..d).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
    let y = rotate(&x, &s);
    let back = unrotate(&y, &s);
    for (a, b) in x.iter().zip(&back) {
        assert!((a - b).abs() < 1e-5);
    }
    let n = |v: &[f32]| v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>();
    assert!((n(&x) - n(&y)).abs() < 1e-4 * n(&x));

    // Deterministic, and every input matters.
    assert_eq!(s, rademacher(42, 3, 5, SignKind::K, d));
    for other in [
        rademacher(43, 3, 5, SignKind::K, d),
        rademacher(42, 4, 5, SignKind::K, d),
        rademacher(42, 3, 6, SignKind::K, d),
        rademacher(42, 3, 5, SignKind::V, d),
    ] {
        assert_ne!(s, other);
    }
    // Pinned: the first eight signs of seed 0, layer 0, head 0, K (the GPU codec's anchor).
    let first: Vec<i8> = rademacher(0, 0, 0, SignKind::K, d)[..8]
        .iter()
        .map(|v| *v as i8)
        .collect();
    assert_eq!(first, PINNED_FIRST_SIGNS);
}

const PINNED_FIRST_SIGNS: [i8; 8] = [-1, 1, 1, 1, -1, 1, 1, -1];

/// The committed codebooks are the generator's output for d = 128 (to F32 rounding); they
/// are ascending and symmetric, the 1- and 2-bit ones sit near the paper's large-d values
/// (±√(2/π); ±0.453, ±1.51) and the distortions near its D_mse (0.36, 0.117, 0.03, 0.009).
/// Breaks if a constant is edited, the density or the Lloyd iteration is wrong.
#[test]
fn codebooks_reproduce() {
    let paper = [0.36, 0.117, 0.03, 0.009];
    let mut generated = String::new();
    let mut mismatch = false;
    for bits in 1..=4u32 {
        let lm = lloyd_max(bits, TQ_DIM);
        generated += &format!("{bits}: {:?} D={}\n", lm.centroids, lm.distortion);
        let cb = codebook(bits);
        assert_eq!(cb.len(), 1 << bits);
        for (a, b) in cb.iter().zip(&lm.centroids) {
            if (f64::from(*a) - b).abs() > 1e-6 {
                mismatch = true;
            }
        }
        if mismatch {
            continue;
        }
        let rel = (lm.distortion - paper[bits as usize - 1]).abs() / paper[bits as usize - 1];
        assert!(
            rel < 0.16,
            "{bits}-bit distortion {} vs {}",
            lm.distortion,
            paper[bits as usize - 1]
        );
        assert!(
            !mismatch,
            "committed codebooks differ from the generator:\n{generated}"
        );
        assert!(cb.windows(2).all(|w| w[0] < w[1]), "{bits}-bit ascending");
        for (a, b) in cb.iter().zip(cb.iter().rev()) {
            assert!((a + b).abs() < 1e-6, "{bits}-bit symmetric");
        }
    }
    assert!(
        !mismatch,
        "committed codebooks differ from the generator:\n{generated}"
    );
    assert!((f64::from(CODEBOOK_1[1]) - (2.0 / std::f64::consts::PI).sqrt()).abs() < 0.01);
    assert!((f64::from(CODEBOOK_2[2]) - 0.453).abs() < 0.01);
    assert!((f64::from(CODEBOOK_2[3]) - 1.51).abs() < 0.01);
    assert_eq!(
        nearest(&CODEBOOK_2, 0.0),
        1,
        "a tie takes the lower centroid"
    );
    assert_eq!(nearest(&CODEBOOK_2, 9.0), 3);
    assert_eq!(nearest(&CODEBOOK_2, -9.0), 0);
}
