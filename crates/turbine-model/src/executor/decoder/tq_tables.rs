//! TurboQuant tables in device memory for the paged attention over TurboQuant L0 pages (P6b
//! S-5, kernel ABI v2.11 `tq_params`).
//!
//! A GPU provider reads the rotation signs, the QJL projection and the codebooks from device
//! memory; the host tables ([`crate::kv_scales::TqKv`]) serve the CPU provider. The executor
//! holds one copy of every model layer's tables in the transcode's layout and hands each
//! attention call its layer's slice (the caller offsets `tables`, user decision 2026-10-01 "6b
//! Task 12: how per-layer TurboQuant tables reach the paged-attention call", A). The server
//! uploads them at startup (or shares the copy it uploads for the TurboQuant transcode) and
//! passes them with [`super::DecoderExecutor::set_tq_device_tables`] before the first forward,
//! so decode graphs capture the final pointers.

use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::{KvTranscodeTables, TqParams};
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

use super::invalid;
use crate::ModelError;
use crate::kv_scales::TqKv;

/// Every model layer's TurboQuant tables in device memory: `codebooks[bits − 1]` F32
/// `[2^bits]` (the unit-variance Lloyd–Max centroids) and `tables` F32
/// `[layers · num_kv_heads · head_elems]`, per (layer, KV head) the K rotation signs, the V
/// rotation signs and the QJL projection `S` row-major — the layout of
/// [`KvTranscodeTables`], so one upload can serve the transcode and the attention.
pub struct TqDeviceTables {
    pub codebooks: [Tensor; 4],
    pub tables: Tensor,
    /// Layers and KV heads `tables` holds.
    pub layers: usize,
    pub kv_heads: usize,
}

impl TqDeviceTables {
    /// Uploads `tq`'s host tables (every layer, the heads each layer holds) to `mem`.
    pub fn upload(mem: &Arc<dyn DeviceMemory>, tq: &TqKv) -> Result<TqDeviceTables, ModelError> {
        let first = tq
            .layers
            .first()
            .ok_or_else(|| invalid("TurboQuant tables of no layer".into()))?;
        let kv_heads = first.heads.len();
        let dim = first.heads.first().map_or(0, |h| h.k_signs.len());
        let per_head = KvTranscodeTables::head_elems(dim as u32);
        let mut flat: Vec<u8> = Vec::with_capacity(tq.layers.len() * kv_heads * per_head * 4);
        for (l, layer) in tq.layers.iter().enumerate() {
            if layer.heads.len() != kv_heads {
                return Err(invalid(format!(
                    "TurboQuant tables of layer {l} hold {} KV heads, layer 0 {kv_heads}",
                    layer.heads.len()
                )));
            }
            for h in &layer.heads {
                if h.k_signs.len() != dim || h.v_signs.len() != dim || h.qjl.len() != dim * dim {
                    return Err(invalid(format!(
                        "TurboQuant tables of layer {l} are not of head_dim {dim}"
                    )));
                }
                for v in h.k_signs.iter().chain(&h.v_signs).chain(&h.qjl) {
                    flat.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let tables = upload_f32(mem, &flat)?;
        let codebooks = codebooks_of(mem, first)?;
        mem.synchronize()?;
        Ok(TqDeviceTables {
            codebooks,
            tables,
            layers: tq.layers.len(),
            kv_heads,
        })
    }

    /// Layer `layer`'s slice: the codebooks and the `[num_kv_heads · head_elems]` tables of
    /// that layer (`None` past the last layer).
    pub fn layer(&self, layer: usize, head_dim: u32) -> Option<KvTranscodeTables<'_>> {
        if layer >= self.layers {
            return None;
        }
        let n = self.kv_heads * KvTranscodeTables::head_elems(head_dim);
        Some(KvTranscodeTables {
            codebooks: [0, 1, 2, 3].map(|i| self.codebooks[i].view()),
            tables: TensorView::contiguous(
                self.tables.storage.whole(),
                layer * n,
                &[n],
                DType::F32,
            ),
        })
    }
}

/// A dense F32 tensor holding the little-endian bytes `bytes`.
fn upload_f32(mem: &Arc<dyn DeviceMemory>, bytes: &[u8]) -> Result<Tensor, ModelError> {
    let mut t = Tensor::empty(mem, &[bytes.len() / 4], DType::F32)?;
    t.storage.copy_from_host(0, bytes)?;
    Ok(t)
}

fn codebooks_of(mem: &Arc<dyn DeviceMemory>, p: &TqParams) -> Result<[Tensor; 4], ModelError> {
    let one = |bits: usize| -> Result<Tensor, ModelError> {
        let cb = &p.codebooks[bits - 1];
        if cb.len() != 1 << bits {
            return Err(invalid(format!(
                "the {bits}-bit TurboQuant codebook has {} centroids",
                cb.len()
            )));
        }
        let bytes: Vec<u8> = cb.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_f32(mem, &bytes)
    };
    Ok([one(1)?, one(2)?, one(3)?, one(4)?])
}
