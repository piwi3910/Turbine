//! Shared vocabulary types (contract §3.4).

use serde::{Deserialize, Serialize};

pub use crate::model_identity::{KvDtype, ModelFingerprint, ModelIdentity};
pub use crate::pressure::{CircuitState, PressureSignal, PressureState};

/// Global device index: the Phase 0 inventory index, stable for the process lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeviceId(pub u32);

/// GPU vendor. Serialized as `"nvidia"` / `"amd"`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Vendor {
    Nvidia,
    Amd,
}

impl Vendor {
    /// Metric label / log value.
    pub fn as_str(self) -> &'static str {
        match self {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
        }
    }
}

/// How a device's memory relates to host memory. Serialized as `"dedicated"` / `"unified"`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryKind {
    Dedicated,
    Unified,
}

/// Element type of a tensor (P1 S-5). `abi_code` equals the `TURBINE_DTYPE_*` C ABI codes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    BF16,
    F16,
    F32,
    I32,
    I64,
    /// FP8 e4m3 (OCP `e4m3fn`: no infinities, NaN = 0x7f / 0xff, max ±448) — FP8 weights and
    /// the FP8 KV cache (Phase 6a).
    F8E4M3,
    /// Raw bytes: the storage of packed formats (INT4 / FP4 nibbles, E8M0 exponents), whose
    /// meaning a quantization scheme describes (Phase 6a).
    U8,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F8E4M3 | DType::U8 => 1,
            DType::BF16 | DType::F16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
        }
    }
    pub fn abi_code(self) -> i32 {
        match self {
            DType::BF16 => 0,
            DType::F16 => 1,
            DType::F32 => 2,
            DType::I32 => 3,
            DType::I64 => 4,
            DType::F8E4M3 => 16,
            DType::U8 => 17,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F8E4M3 => "f8e4m3",
            DType::U8 => "u8",
        }
    }
    /// True for raw storage whose element meaning comes from a quantization scheme.
    pub fn is_packed_storage(self) -> bool {
        matches!(self, DType::U8)
    }
}

/// Engine request id; displayed as `cmpl-<uuid>` / `chatcmpl-<uuid>` by the API.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct RequestId(pub uuid::Uuid);

impl RequestId {
    pub fn new_v4() -> Self {
        RequestId(uuid::Uuid::new_v4())
    }
}

/// Per-token KV description (budget, pool, wire format).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct KvLayout {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
    pub block_tokens: u32,
}

impl KvLayout {
    /// K and V of every layer for one token (Llama-3.2-3B BF16: 114 688). Saturates at
    /// `u64::MAX` for dimensions no device could hold, so a budget refuses it.
    pub fn bytes_per_token(&self) -> u64 {
        u64::from(self.num_layers)
            .saturating_mul(2)
            .saturating_mul(u64::from(self.num_kv_heads))
            .saturating_mul(u64::from(self.head_dim))
            .saturating_mul(self.dtype.size_bytes() as u64)
    }
    /// One block of `block_tokens` tokens (Llama-3.2-3B: 1 835 008 at 16 tokens, 14 680 064 at
    /// the default 128). Saturates like [`KvLayout::bytes_per_token`].
    pub fn block_bytes(&self) -> u64 {
        self.bytes_per_token()
            .saturating_mul(u64::from(self.block_tokens))
    }
}

/// Model description consumed by budget and planners.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ModelShape {
    pub architecture: String,
    pub num_layers: u32,
    pub hidden: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub tied_embeddings: bool,
    pub weight_bytes: u64,
    pub max_position_embeddings: u32,
}

/// One sequence (= one choice of a request); an engine-local counter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SeqId(pub u64);

/// Logical L0 (GPU) KV block id: the block's index in the preallocated pool.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BlockId(pub u32);

/// Data-parallel replica index (P5); rendered as a decimal `replica` label.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReplicaId(pub u32);

/// Request priority (the vLLM `priority` extension): lower is served first; default 0
/// (CONFLICT C-10).
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Priority(pub i32);

impl Priority {
    /// `< 0` High, `0` Normal, `> 0` Low (CONFLICT C-10).
    pub fn class(self) -> PriorityClass {
        match self.0 {
            i32::MIN..=-1 => PriorityClass::High,
            0 => PriorityClass::Normal,
            _ => PriorityClass::Low,
        }
    }
}

/// Coarse priority class derived from `Priority` (P4 weights, P6 routing).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PriorityClass {
    High,
    Normal,
    Low,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_codes_and_kv_layout_sizes() {
        let codes: Vec<(i32, usize, &str)> =
            [DType::BF16, DType::F16, DType::F32, DType::I32, DType::I64]
                .iter()
                .map(|d| (d.abi_code(), d.size_bytes(), d.as_str()))
                .collect();
        assert_eq!(
            codes,
            [
                (0, 2, "bf16"),
                (1, 2, "f16"),
                (2, 4, "f32"),
                (3, 4, "i32"),
                (4, 8, "i64")
            ]
        );
        // Llama-3.2-3B BF16: 28 layers, 8 KV heads, head_dim 128, 16-token blocks.
        let layout = KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        };
        assert_eq!(layout.bytes_per_token(), 114_688);
        assert_eq!(layout.block_bytes(), 1_835_008);
        let default_page = KvLayout {
            block_tokens: 128,
            ..layout
        };
        assert_eq!(default_page.block_bytes(), 14_680_064);
        // A config whose dimensions overflow u64 saturates (and so fails any budget) instead of
        // wrapping to a small size.
        let huge = KvLayout {
            num_layers: u32::MAX,
            num_kv_heads: u32::MAX,
            head_dim: u32::MAX,
            dtype: DType::BF16,
            block_tokens: u32::MAX,
        };
        assert_eq!(huge.bytes_per_token(), u64::MAX);
        assert_eq!(huge.block_bytes(), u64::MAX);
        assert_ne!(RequestId::new_v4(), RequestId::new_v4());
    }

    /// Phase 6a S-2: FP8 e4m3 and raw packed bytes take ABI codes 16 and 17 (the header's
    /// reserved 16..63 range), one byte each; an FP8 KV page is half a BF16 one. Neither is a
    /// float the CPU GEMM computes in, and `U8` alone is packed storage (INT4 / FP4 / E8M0).
    /// Breaks if a code collides with the Phase 1 codes or a size is wrong.
    #[test]
    fn quant_dtypes() {
        assert_eq!(
            (
                DType::F8E4M3.abi_code(),
                DType::F8E4M3.size_bytes(),
                DType::F8E4M3.as_str()
            ),
            (16, 1, "f8e4m3")
        );
        assert_eq!(
            (
                DType::U8.abi_code(),
                DType::U8.size_bytes(),
                DType::U8.as_str()
            ),
            (17, 1, "u8")
        );
        assert!(DType::U8.is_packed_storage());
        assert!(!DType::F8E4M3.is_packed_storage());
        assert!(!DType::BF16.is_packed_storage());
        let bf16 = KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 128,
        };
        let fp8 = KvLayout {
            dtype: DType::F8E4M3,
            ..bf16
        };
        assert_eq!(fp8.bytes_per_token() * 2, bf16.bytes_per_token());
        assert_eq!(fp8.block_bytes(), 7_340_032);
    }
}
