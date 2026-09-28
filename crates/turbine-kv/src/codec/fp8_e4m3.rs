//! `fp8_e4m3`: OCP e4m3fn elements (P6b S-1).
//!
//! From an FP8 L0 (`kv.dtype: fp8_e4m3`, 6a S-13) the slot is the L0 page unchanged: the L0
//! scales are the namespace's, so the copy is lossless. From a BF16 L0 the block is quantized
//! with per-block, per-layer K and V scales `max(absmax / 448, 1 / (448 × 512))` (no calibration
//! needed), stored in a slot header.
//!
//! Slot layout from a BF16 L0: a header of `num_layers × [k_scale, v_scale]` F32 little-endian,
//! then the L0 element order (per layer `[2, block_tokens, kv_heads, head_dim]`), one FP8 byte
//! per element.

use turbine_core::registry::Module;
use turbine_core::types::{DType, KvLayout};

use super::l0::copy;
use super::{
    CodecError, CodecParams, FP8_E4M3_MAX, KvCodec, L0Geometry, check_size, fp8_e4m3_round,
    fp8_e4m3_value, read_l0, write_l0,
};

/// Smallest per-block scale (vLLM's dynamic-FP8 floor `1 / (448 × 512)`): a zero block stays
/// finite.
pub const FP8_MIN_SCALE: f32 = 1.0 / (FP8_E4M3_MAX * 512.0);

/// The FP8 e4m3 codec.
#[derive(Clone, Copy, Debug)]
pub struct Fp8E4m3Codec;

impl Module for Fp8E4m3Codec {
    fn name(&self) -> &'static str {
        "fp8_e4m3"
    }
}

impl KvCodec for Fp8E4m3Codec {
    /// Lossless from an FP8 L0 (the page bytes are kept), lossy from BF16.
    fn lossy(&self, l0: &KvLayout) -> bool {
        l0.dtype != DType::F8E4M3
    }

    fn abi_code(&self) -> u8 {
        1
    }

    /// e4m3 keeps 3 mantissa bits: a normal element's rounding error is ≤ 2⁻⁴ of it, so the
    /// block's error energy is ≤ 1/256 of its energy (subnormal errors, ≤ scale × 2⁻¹⁰, are far
    /// below that at absmax scaling).
    fn nmse_bound(&self) -> f64 {
        1.0 / 256.0
    }

    fn bytes_per_block(&self, l0: &KvLayout) -> u64 {
        match l0.dtype {
            DType::F8E4M3 => l0.block_bytes(),
            _ => header_bytes(l0) + elements(l0) as u64,
        }
    }

    fn encode_cpu(
        &self,
        src: &[u8],
        l0: &KvLayout,
        dst: &mut [u8],
        _params: &CodecParams,
    ) -> Result<(), CodecError> {
        self.supports(l0)?;
        if l0.dtype == DType::F8E4M3 {
            return copy(NAME, src, l0, dst);
        }
        check_size(NAME, "source", src.len(), l0.block_bytes())?;
        check_size(NAME, "destination", dst.len(), self.bytes_per_block(l0))?;
        let g = L0Geometry::of(l0);
        let (header, body) = dst.split_at_mut(header_bytes(l0) as usize);
        for layer in 0..g.layers {
            for kind in 0..2 {
                let start = g.vector_elem(layer, kind, 0, 0);
                let range = start..start + g.half_layer_elems();
                let amax = range
                    .clone()
                    .map(|i| read_l0(src, l0.dtype, i, 1.0).abs())
                    .fold(0f32, f32::max);
                let scale = (amax / FP8_E4M3_MAX).max(FP8_MIN_SCALE);
                let h = (layer * 2 + kind) * 4;
                header[h..h + 4].copy_from_slice(&scale.to_le_bytes());
                for i in range {
                    body[i] = fp8_e4m3_round(read_l0(src, l0.dtype, i, 1.0) / scale);
                }
            }
        }
        Ok(())
    }

    fn decode_cpu(
        &self,
        src: &[u8],
        l0: &KvLayout,
        dst: &mut [u8],
        _params: &CodecParams,
    ) -> Result<(), CodecError> {
        self.supports(l0)?;
        if l0.dtype == DType::F8E4M3 {
            return copy(NAME, src, l0, dst);
        }
        check_size(NAME, "source", src.len(), self.bytes_per_block(l0))?;
        check_size(NAME, "destination", dst.len(), l0.block_bytes())?;
        let g = L0Geometry::of(l0);
        let (header, body) = src.split_at(header_bytes(l0) as usize);
        for layer in 0..g.layers {
            for kind in 0..2 {
                let h = (layer * 2 + kind) * 4;
                let scale = f32::from_le_bytes(header[h..h + 4].try_into().expect("4 bytes"));
                let start = g.vector_elem(layer, kind, 0, 0);
                let codes = &body[start..start + g.half_layer_elems()];
                for (i, &code) in (start..).zip(codes) {
                    write_l0(dst, l0.dtype, i, fp8_e4m3_value(code) * scale, 1.0);
                }
            }
        }
        Ok(())
    }
}

const NAME: &str = "fp8_e4m3";

/// The slot header: one F32 K and one F32 V scale per layer.
fn header_bytes(l0: &KvLayout) -> u64 {
    u64::from(l0.num_layers) * 8
}

/// K and V elements of one block.
fn elements(l0: &KvLayout) -> usize {
    let g = L0Geometry::of(l0);
    g.layers * 2 * g.half_layer_elems()
}
