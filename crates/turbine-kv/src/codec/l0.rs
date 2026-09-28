//! `l0`: the L0 page bytes unchanged (lossless; the default of every tier).

use turbine_core::registry::Module;
use turbine_core::types::KvLayout;

use super::{CodecError, CodecParams, KvCodec, check_l0_dtype, check_size};

/// The identity codec: a slot holds the L0 block as is.
#[derive(Clone, Copy, Debug)]
pub struct L0Codec;

impl Module for L0Codec {
    fn name(&self) -> &'static str {
        "l0"
    }
}

impl KvCodec for L0Codec {
    fn lossy(&self, _l0: &KvLayout) -> bool {
        false
    }

    fn abi_code(&self) -> u8 {
        0
    }

    fn nmse_bound(&self) -> f64 {
        0.0
    }

    fn bytes_per_block(&self, l0: &KvLayout) -> u64 {
        l0.block_bytes()
    }

    fn encode_cpu(
        &self,
        src: &[u8],
        l0: &KvLayout,
        dst: &mut [u8],
        _params: &CodecParams,
    ) -> Result<(), CodecError> {
        copy("l0", src, l0, dst)
    }

    fn decode_cpu(
        &self,
        src: &[u8],
        l0: &KvLayout,
        dst: &mut [u8],
        _params: &CodecParams,
    ) -> Result<(), CodecError> {
        copy("l0", src, l0, dst)
    }
}

/// Copies one whole L0 block (both buffers `l0.block_bytes()` long).
pub(super) fn copy(
    codec: &'static str,
    src: &[u8],
    l0: &KvLayout,
    dst: &mut [u8],
) -> Result<(), CodecError> {
    check_l0_dtype(codec, l0)?;
    check_size(codec, "source", src.len(), l0.block_bytes())?;
    check_size(codec, "destination", dst.len(), l0.block_bytes())?;
    dst.copy_from_slice(src);
    Ok(())
}
