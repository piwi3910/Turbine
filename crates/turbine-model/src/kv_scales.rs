//! The L0 KV cache element format of a model (Phase 6a S-13, `kv.dtype`): BF16 (the exact
//! default), or FP8 e4m3 pages with one K and one V scale per layer (user decision 2026-09-28,
//! Q10: the checkpoint's `k_scale` / `v_scale` tensors when present, else 1.0). A K element is
//! stored as `e4m3(k / k_scale)` and read back as `e4m3 · k_scale`, likewise V.

use std::sync::Arc;

use turbine_core::types::DType;

/// How the paged KV pool stores K and V ([`crate::ModelArchConfig::kv_cache`]).
#[derive(Clone, PartialEq, Debug)]
pub struct KvCache {
    /// Element type of the pool pages: the activation dtype (BF16) or [`DType::F8E4M3`].
    pub dtype: DType,
    /// Per-layer K dequantization scales (FP8 only; empty for BF16).
    pub k_scales: Arc<[f32]>,
    /// Per-layer V dequantization scales (FP8 only; empty for BF16).
    pub v_scales: Arc<[f32]>,
}

impl KvCache {
    /// BF16 pages: the exact, default format.
    pub fn bf16() -> KvCache {
        KvCache {
            dtype: DType::BF16,
            k_scales: Arc::from(Vec::new()),
            v_scales: Arc::from(Vec::new()),
        }
    }

    /// FP8 e4m3 pages with these per-layer scales.
    pub fn fp8_e4m3(k_scales: Vec<f32>, v_scales: Vec<f32>) -> KvCache {
        KvCache {
            dtype: DType::F8E4M3,
            k_scales: Arc::from(k_scales),
            v_scales: Arc::from(v_scales),
        }
    }

    /// True when the pages are quantized (FP8).
    pub fn is_fp8(&self) -> bool {
        self.dtype == DType::F8E4M3
    }

    /// Layer `layer`'s K scale (model layer index; 1.0 when none is stored).
    pub fn k_scale(&self, layer: usize) -> f32 {
        self.k_scales.get(layer).copied().unwrap_or(1.0)
    }

    /// Layer `layer`'s V scale (model layer index; 1.0 when none is stored).
    pub fn v_scale(&self, layer: usize) -> f32 {
        self.v_scales.get(layer).copied().unwrap_or(1.0)
    }
}

impl Default for KvCache {
    fn default() -> Self {
        KvCache::bf16()
    }
}
