//! Lloyd–Max codebooks for TurboQuant's rotated coordinates (P6b S-4).
//!
//! After the random rotation, each coordinate `u` of a unit vector in `d` dimensions follows
//! `f(u) ∝ (1 − u²)^((d−3)/2)` on [−1, 1] (a scaled Beta, near Gaussian at d = 128). Codes
//! quantize the coordinate scaled by √d (unit variance), `x = √d·u`, whose density is
//! `∝ (1 − x²/d)^((d−3)/2)` on [−√d, √d]. The codebook of `b` bits holds the `2^b` centroids
//! minimising E[(x − c(x))²] ([`lloyd_max`]); the constants below are its output for d = 128,
//! rounded to F32, and `codebooks_reproduce` regenerates them.

/// The head dimension the committed codebooks are for.
pub const TQ_DIM: usize = 128;

/// 1-bit codebook (the K MSE stage of `tq2`).
pub const CODEBOOK_1: [f32; 2] = [-0.79944444, 0.79944444];
/// 2-bit codebook (V of `tq2`).
pub const CODEBOOK_2: [f32; 4] = [-1.505193, -0.45245326, 0.45245326, 1.505193];
/// 3-bit codebook (the K MSE stage of `tq4`).
pub const CODEBOOK_3: [f32; 8] = [
    -2.131471,
    -1.3365989,
    -0.7533302,
    -0.24442486,
    0.24442486,
    0.7533302,
    1.3365989,
    2.131471,
];
/// 4-bit codebook (V of `tq4`).
pub const CODEBOOK_4: [f32; 16] = [
    -2.6888592,
    -2.0459254,
    -1.6043426,
    -1.2477703,
    -0.93709695,
    -0.6536189,
    -0.38638088,
    -0.12787314,
    0.12787314,
    0.38638088,
    0.6536189,
    0.93709695,
    1.2477703,
    1.6043426,
    2.0459254,
    2.6888592,
];

/// The committed codebook of `bits` (1 ..= 4) for [`TQ_DIM`].
pub fn codebook(bits: u32) -> &'static [f32] {
    match bits {
        1 => &CODEBOOK_1,
        2 => &CODEBOOK_2,
        3 => &CODEBOOK_3,
        4 => &CODEBOOK_4,
        _ => panic!("no {bits}-bit TurboQuant codebook"),
    }
}

/// Index of the nearest centroid of an ascending codebook; a value exactly between two
/// centroids takes the lower one.
pub fn nearest(cb: &[f32], x: f32) -> u8 {
    let mut code = 0u8;
    for w in cb.windows(2) {
        if x > (w[0] + w[1]) * 0.5 {
            code += 1;
        } else {
            break;
        }
    }
    code
}

/// Grid intervals of the numerical integration.
const GRID: usize = 1 << 18;

/// A Lloyd–Max codebook and its distortion.
#[derive(Clone, Debug)]
pub struct LloydMax {
    /// Ascending centroids of the √d-scaled coordinate.
    pub centroids: Vec<f64>,
    /// E[(x − c(x))²] of the scaled coordinate (its variance is 1, so this is the MSE per
    /// coordinate of a unit vector × d, the paper's D_mse).
    pub distortion: f64,
}

/// The `2^bits`-level Lloyd–Max quantizer of the √d-scaled rotated coordinate in `d`
/// dimensions: density tabulated on a uniform grid of [−√d, √d], cumulative moments by the
/// trapezoid rule (linear interpolation between grid points), Lloyd iterations from uniform
/// quantiles until no centroid moves by more than 1e-13.
pub fn lloyd_max(bits: u32, d: usize) -> LloydMax {
    let levels = 1usize << bits;
    let lim = (d as f64).sqrt();
    let h = 2.0 * lim / GRID as f64;
    let e = (d as f64 - 3.0) / 2.0;
    let x = |i: usize| -lim + i as f64 * h;
    let f: Vec<f64> = (0..=GRID)
        .map(|i| (1.0 - x(i) * x(i) / d as f64).max(0.0).powf(e))
        .collect();
    // Cumulative ∫f, ∫x·f, ∫x²·f at every grid point.
    let (mut m0, mut m1, mut m2) = (
        vec![0f64; GRID + 1],
        vec![0f64; GRID + 1],
        vec![0f64; GRID + 1],
    );
    for i in 1..=GRID {
        let (a, b) = (x(i - 1), x(i));
        m0[i] = m0[i - 1] + h * (f[i - 1] + f[i]) / 2.0;
        m1[i] = m1[i - 1] + h * (a * f[i - 1] + b * f[i]) / 2.0;
        m2[i] = m2[i - 1] + h * (a * a * f[i - 1] + b * b * f[i]) / 2.0;
    }
    // A cumulative moment at any point: linear between grid points.
    let at = |m: &[f64], t: f64| -> f64 {
        let p = ((t + lim) / h).clamp(0.0, GRID as f64);
        let i = (p.floor() as usize).min(GRID - 1);
        let frac = p - i as f64;
        m[i] + frac * (m[i + 1] - m[i])
    };
    let total = m0[GRID];
    // Start from the centroids of equal-probability cells.
    let mut c: Vec<f64> = (0..levels)
        .map(|k| {
            let target = (k as f64 + 0.5) / levels as f64 * total;
            let i = m0.partition_point(|v| *v < target).min(GRID);
            x(i)
        })
        .collect();
    for _ in 0..200_000 {
        let mut bounds = vec![-lim];
        bounds.extend(c.windows(2).map(|w| (w[0] + w[1]) / 2.0));
        bounds.push(lim);
        let next: Vec<f64> = (0..levels)
            .map(|k| {
                let (a, b) = (bounds[k], bounds[k + 1]);
                (at(&m1, b) - at(&m1, a)) / (at(&m0, b) - at(&m0, a))
            })
            .collect();
        let moved = next
            .iter()
            .zip(&c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        c = next;
        if moved < 1e-13 {
            break;
        }
    }
    let mut bounds = vec![-lim];
    bounds.extend(c.windows(2).map(|w| (w[0] + w[1]) / 2.0));
    bounds.push(lim);
    let mut err = 0.0;
    for k in 0..levels {
        let (a, b) = (bounds[k], bounds[k + 1]);
        let (p0, p1, p2) = (
            at(&m0, b) - at(&m0, a),
            at(&m1, b) - at(&m1, a),
            at(&m2, b) - at(&m2, a),
        );
        err += p2 - 2.0 * c[k] * p1 + c[k] * c[k] * p0;
    }
    LloydMax {
        centroids: c,
        distortion: err / total,
    }
}
