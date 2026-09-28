//! The 1-bit QJL residual of TurboQuant's inner-product variant (P6b S-4; arXiv 2504.19874,
//! TurboQuant_prod), as the paper builds it: `S ∈ ℝ^{d×d}` with i.i.d. N(0, 1) entries, stored
//! signs `sign(S·r)`, dequantization `r̂ = √(π/2)/d · ‖r‖ · Sᵀ·sign(S·r)`, which is unbiased in
//! every inner product (E_S ⟨q, r̂⟩ = ⟨q, r⟩).
//!
//! A randomized Hadamard projection was measured first (plan Task 7 wrote `H·(s' ⊙ r)`): its
//! ±1/√d entries bias the estimate by ≈ 2.7 % of ⟨q, r⟩ for `tq4` (6.5e-4 against 3 standard
//! errors of 3.7e-4 over 10,000 pairs), so `k_inner_product_unbiased` failed; the Gaussian `S`
//! passes (provisional, pending user review).
//!
//! `S` of (seed, layer, head): row-major F32, from the SplitMix64 stream started at
//! `splitmix64(seed ^ splitmix64(layer << 24 | head << 8 | 2))`, Box–Muller in F64 on pairs of
//! 53-bit uniforms `(u + 0.5)/2^53`, `√(−2 ln u1)·cos(2π u2)` then `·sin(2π u2)`, rounded to F32.
//! The GPU codec receives the same table from the host (`tq_params`), never regenerates it.
//!
//! Sign bits: bit `i` (byte `i / 8`, bit `i % 8`) is 1 when `(S·r)_i ≥ 0`.

use super::hadamard::splitmix64;

const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// The `d × d` Gaussian projection of (seed, layer, head), row-major.
pub fn projection(seed: u64, layer: u32, head: u32, d: usize) -> Vec<f32> {
    let tag = (u64::from(layer) << 24) | (u64::from(head) << 8) | 2;
    let mut state = splitmix64(seed ^ splitmix64(tag));
    let mut uniform = move || {
        let v = splitmix64(state);
        state = state.wrapping_add(GAMMA);
        ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(d * d);
    while out.len() < d * d {
        let (u1, u2) = (uniform(), uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        out.push((r * t.cos()) as f32);
        if out.len() < d * d {
            out.push((r * t.sin()) as f32);
        }
    }
    out
}

/// Sign bits of `S·r` (`r.len() / 8` bytes).
pub fn encode(r: &[f32], s: &[f32]) -> Vec<u8> {
    let d = r.len();
    let mut bits = vec![0u8; d.div_ceil(8)];
    for (i, row) in s.chunks_exact(d).enumerate() {
        let z: f32 = row.iter().zip(r).map(|(a, b)| a * b).sum();
        if z >= 0.0 {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    bits
}

/// `r̂ = √(π/2)/d · norm · Sᵀ·z` with `z` the ±1 vector of `bits`.
pub fn decode(bits: &[u8], norm: f32, s: &[f32]) -> Vec<f32> {
    let d = (s.len() as f64).sqrt() as usize;
    let mut out = vec![0f32; d];
    for (i, row) in s.chunks_exact(d).enumerate() {
        let z = if bits[i / 8] >> (i % 8) & 1 == 1 {
            1.0
        } else {
            -1.0
        };
        out.iter_mut().zip(row).for_each(|(o, a)| *o += z * a);
    }
    let k = ((std::f64::consts::PI / 2.0).sqrt() / d as f64) as f32 * norm;
    out.iter_mut().for_each(|v| *v *= k);
    out
}
