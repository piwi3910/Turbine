//! Quantized-weight and activation-quantization descriptors (Phase 6a S-5, S-6): the vocabulary
//! shared by the CPU reference (`cpu::quant`), the quantized GEMM op and the kernel C ABI v2.9
//! (`TURBINE_QSCHEME_*`, `TURBINE_ACTQ_*`). Vendor-neutral: layouts only, no provider types.
//!
//! Layouts (row-major, `n` output rows × `k` input columns, the layout the loader repacks every
//! checkpoint packaging into):
//! - FP8 schemes: `data` = `n × k` e4m3 bytes; `scales` (F32) = `[1]` (tensor), `[n]` (channel) or
//!   `[ceil(n / block_n) × ceil(k / block_k)]` (block). Value = `e4m3(q) × scale`.
//! - INT4 group schemes: `data` = `n × k/2` bytes, the low nibble holding the even column;
//!   `scales` (F32) = `[n × k/group]`; zero points (`Int4GroupZp` only) = `[n × k/group]` bytes
//!   0..=15. Value = `(q − z) × s`, with `z = 8` for `Int4GroupSym`.
//! - MXFP4: `data` = `n × k/2` bytes of E2M1 codes, low nibble first; `scales` = `[n × k/32]` E8M0
//!   bytes. Value = `e2m1(q) × 2^(e − 127)`.

/// A quantized linear layer's weight scheme (`TURBINE_QSCHEME_*`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum QuantSchemeDesc {
    /// FP8 e4m3 with one scale for the whole tensor.
    Fp8Tensor,
    /// FP8 e4m3 with one scale per output row.
    Fp8Channel,
    /// FP8 e4m3 with one scale per `block_n × block_k` block.
    Fp8Block { block_n: u32, block_k: u32 },
    /// INT4 groups of `group` columns with a scale and a zero point each (AWQ).
    Int4GroupZp { group: u32 },
    /// INT4 groups of `group` columns with a scale each, zero point 8 (GPTQ symmetric).
    Int4GroupSym { group: u32 },
    /// OCP MXFP4: E2M1 elements, one E8M0 exponent per 32 columns.
    Mxfp4,
}

impl QuantSchemeDesc {
    /// `group_size`, `block_n`, `block_k` of the C descriptor.
    pub fn abi_shape(self) -> (i32, i32, i32) {
        match self {
            QuantSchemeDesc::Fp8Block { block_n, block_k } => (0, block_n as i32, block_k as i32),
            QuantSchemeDesc::Int4GroupZp { group } | QuantSchemeDesc::Int4GroupSym { group } => {
                (group as i32, 0, 0)
            }
            QuantSchemeDesc::Mxfp4 => (MX_BLOCK as i32, 0, 0),
            QuantSchemeDesc::Fp8Tensor | QuantSchemeDesc::Fp8Channel => (0, 0, 0),
        }
    }

    /// `TURBINE_QSCHEME_*` (0 is BF16, never passed to the quantized GEMM).
    pub fn abi_code(self) -> i32 {
        match self {
            QuantSchemeDesc::Fp8Tensor => 1,
            QuantSchemeDesc::Fp8Channel => 2,
            QuantSchemeDesc::Fp8Block { .. } => 3,
            QuantSchemeDesc::Int4GroupZp { .. } => 4,
            QuantSchemeDesc::Int4GroupSym { .. } => 5,
            QuantSchemeDesc::Mxfp4 => 6,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            QuantSchemeDesc::Fp8Tensor => "fp8_tensor",
            QuantSchemeDesc::Fp8Channel => "fp8_channel",
            QuantSchemeDesc::Fp8Block { .. } => "fp8_block",
            QuantSchemeDesc::Int4GroupZp { .. } => "int4_group_zp",
            QuantSchemeDesc::Int4GroupSym { .. } => "int4_group_sym",
            QuantSchemeDesc::Mxfp4 => "mxfp4",
        }
    }

    /// Bytes of packed weight data for an `n × k` layer.
    pub fn data_bytes(self, n: usize, k: usize) -> usize {
        match self {
            QuantSchemeDesc::Fp8Tensor
            | QuantSchemeDesc::Fp8Channel
            | QuantSchemeDesc::Fp8Block { .. } => n * k,
            QuantSchemeDesc::Int4GroupZp { .. }
            | QuantSchemeDesc::Int4GroupSym { .. }
            | QuantSchemeDesc::Mxfp4 => n * k / 2,
        }
    }

    /// Number of scale entries for an `n × k` layer.
    pub fn scale_count(self, n: usize, k: usize) -> usize {
        match self {
            QuantSchemeDesc::Fp8Tensor => 1,
            QuantSchemeDesc::Fp8Channel => n,
            QuantSchemeDesc::Fp8Block { block_n, block_k } => {
                n.div_ceil(block_n as usize) * k.div_ceil(block_k as usize)
            }
            QuantSchemeDesc::Int4GroupZp { group } | QuantSchemeDesc::Int4GroupSym { group } => {
                n * k.div_ceil(group as usize)
            }
            QuantSchemeDesc::Mxfp4 => n * k.div_ceil(MX_BLOCK),
        }
    }
}

/// Elements sharing one E8M0 exponent in MXFP4.
pub const MX_BLOCK: usize = 32;

/// How activations are quantized before a quantized GEMM (`TURBINE_ACTQ_*`). The static
/// per-tensor scale of [`ActQuantDesc::Fp8Tensor`] travels with the call, not the mode, so a
/// mode is a registry key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum ActQuantDesc {
    /// Activations stay BF16 (weight-only schemes).
    None,
    /// FP8 e4m3 with a static per-tensor scale from the checkpoint (`input_scale`).
    Fp8Tensor,
    /// FP8 e4m3 with a dynamic scale per row (token): `amax / 448`.
    Fp8Token,
    /// FP8 e4m3 with a dynamic scale per row and group of `group` columns.
    Fp8Group { group: u32 },
    /// MXFP4 quantize-dequantize of each 32-column group (Quark W4A4 emulation; the result is
    /// BF16-representable and feeds the weight-only GEMM).
    Mxfp4Emulated,
}

impl ActQuantDesc {
    pub fn as_str(self) -> &'static str {
        match self {
            ActQuantDesc::None => "none",
            ActQuantDesc::Fp8Tensor => "fp8_tensor",
            ActQuantDesc::Fp8Token => "fp8_token",
            ActQuantDesc::Fp8Group { .. } => "fp8_group",
            ActQuantDesc::Mxfp4Emulated => "mxfp4_emulated",
        }
    }

    /// True for the FP8 modes (activations arrive as e4m3 with F32 scales).
    pub fn is_fp8(self) -> bool {
        matches!(
            self,
            ActQuantDesc::Fp8Tensor | ActQuantDesc::Fp8Token | ActQuantDesc::Fp8Group { .. }
        )
    }

    /// Number of activation scales for an `m × k` activation matrix.
    pub fn scale_count(self, m: usize, k: usize) -> usize {
        match self {
            ActQuantDesc::None => 0,
            ActQuantDesc::Fp8Tensor => 1,
            ActQuantDesc::Fp8Token => m,
            ActQuantDesc::Fp8Group { group } => m * k.div_ceil(group as usize),
            ActQuantDesc::Mxfp4Emulated => m * k.div_ceil(MX_BLOCK),
        }
    }

    /// `TURBINE_ACTQ_*`.
    pub fn abi_code(self) -> i32 {
        match self {
            ActQuantDesc::None => 0,
            ActQuantDesc::Fp8Tensor => 1,
            ActQuantDesc::Fp8Token => 2,
            ActQuantDesc::Fp8Group { .. } => 3,
            ActQuantDesc::Mxfp4Emulated => 4,
        }
    }
}
