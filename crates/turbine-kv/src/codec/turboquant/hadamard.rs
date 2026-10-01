//! The randomized Hadamard rotation of TurboQuant (P6b S-4): `y = H·(s ⊙ x) / √d` with `H` the
//! (unnormalised, Sylvester) Walsh–Hadamard matrix and `s` a vector of ±1 signs drawn from a
//! seed. `H·H = d·I`, so the rotation is orthonormal and its inverse is `x = s ⊙ (H·y) / √d`.
//!
//! Sign derivation (part of the codec, versioned by its name): the signs of (seed, layer, head,
//! kind) are the bits of successive SplitMix64 outputs, least-significant bit first, of the
//! stream started at `splitmix64(seed ^ splitmix64(layer << 24 | head << 8 | kind))`; bit 1 is
//! −1. `kind` separates the K rotation (0) and the V rotation (1) (2 tagged the QJL projection of
//! the former inner-product K and stays unused). The signs depend only on the namespace seed, the layer and the head, so one
//! rotation of q serves every TurboQuant block of a sequence (S-5).

/// Which rotation of a (layer, head) the signs are for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignKind {
    K = 0,
    V = 1,
}

const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64's output function of state `z` (the state is advanced by [`GAMMA`] before it).
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The ±1 signs of one rotation, `d` of them.
pub fn rademacher(seed: u64, layer: u32, head: u32, kind: SignKind, d: usize) -> Vec<f32> {
    let tag = (u64::from(layer) << 24) | (u64::from(head) << 8) | kind as u64;
    let mut state = splitmix64(seed ^ splitmix64(tag));
    let mut out = Vec::with_capacity(d);
    while out.len() < d {
        let bits = splitmix64(state);
        state = state.wrapping_add(GAMMA);
        for b in 0..64 {
            if out.len() == d {
                break;
            }
            out.push(if (bits >> b) & 1 == 1 { -1.0 } else { 1.0 });
        }
    }
    out
}

/// In-place unnormalised fast Walsh–Hadamard transform (Sylvester order, butterflies of span
/// 1, 2, 4, …). `x.len()` must be a power of two.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

fn inv_sqrt(d: usize) -> f32 {
    (1.0 / (d as f64).sqrt()) as f32
}

/// `y = H·(s ⊙ x) / √d`.
pub fn rotate(x: &[f32], signs: &[f32]) -> Vec<f32> {
    let mut y: Vec<f32> = x.iter().zip(signs).map(|(v, s)| v * s).collect();
    fwht(&mut y);
    let k = inv_sqrt(y.len());
    y.iter_mut().for_each(|v| *v *= k);
    y
}

/// The inverse rotation, `x = s ⊙ (H·y) / √d`.
pub fn unrotate(y: &[f32], signs: &[f32]) -> Vec<f32> {
    let mut x = y.to_vec();
    fwht(&mut x);
    let k = inv_sqrt(x.len());
    x.iter_mut().zip(signs).for_each(|(v, s)| *v *= k * s);
    x
}
