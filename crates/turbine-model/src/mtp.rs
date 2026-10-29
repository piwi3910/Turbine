//! P8 MTP: the Qwen3.5/3.6 checkpoint's own multi-token-prediction head (`mtp.*`) as a
//! one-layer model. The head is the target checkpoint's single full-attention (MoE) layer plus
//! a `fc`/pre-fc-norm front-matter; the one-layer executor loads it through
//! [`crate::families::qwen3_5::mtp_aliases`] over the target's checkpoint.
//!
//! The front matter (`fc(pre_fc_norm_embedding(e) ‖ pre_fc_norm_hidden(h))`) and the injected
//! input run **on device** ([`DecoderExecutor::set_mtp_front`] / `mtp_step`): the head never
//! reads the hidden rows or the embeddings back to the host, so a speculative round has no
//! per-step device↔host sync.
use std::path::Path;
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::{KernelProvider, KernelRegistry};
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::executor::decoder::invalid;
use crate::executor::{ExecutorLimits, ExecutorOptions, LogitsSlot, ModelExecutor};
use crate::families::qwen3_5;
use crate::loader::{MAX_STAGING_BYTES, WeightLoader};
use crate::safetensors::SafetensorsIndex;
use crate::weights::{WeightFormatRef, detect, resolve_for_providers};

/// The loaded MTP head: the one-layer executor (carrying the on-device front-matter weights)
/// and its hidden width.
pub struct MtpHead {
    pub executor: Box<dyn ModelExecutor>,
    dim: usize,
}

/// Load the one-layer MTP head from the target checkpoint `dir`, over `mem` and `registry`.
/// The head's `mtp.layers.0.*` tensors are aliased into the executor's own layer 0 and
/// `mtp.norm.weight` into the layer stack's final norm ([`qwen3_5::mtp_aliases`]); the
/// front-matter weights are installed on device with the `1 + w` offset baked into the norms.
/// Checkpoint tensors naming no slot are logged `unexpected_tensor` (the target's other
/// layers), which the loader does not treat as an error.
pub fn load(
    cfg: &ModelArchConfig,
    dir: &Path,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    limits: ExecutorLimits,
    opts: ExecutorOptions,
    providers: &[Arc<dyn KernelProvider>],
) -> Result<MtpHead, ModelError> {
    let mut mcfg = qwen3_5::mtp_arch_config(cfg);
    let mut index = SafetensorsIndex::open(dir)?;
    let head: Vec<String> = index
        .entries()
        .filter(|e| e.name.starts_with("mtp.layers.0."))
        .map(|e| e.name.clone())
        .collect();
    for (from, to) in qwen3_5::mtp_aliases(cfg, &head) {
        index.alias(&from, &to);
    }
    let mut slots = mcfg.family.0.weight_slots(&mcfg);
    slots.extend(qwen3_5::mtp_front_slots(cfg));
    let config_path = dir.join("config.json");
    let text = std::fs::read(&config_path).map_err(|e| ModelError::Io {
        path: config_path.clone(),
        detail: e.to_string(),
    })?;
    let top: serde_json::Value = serde_json::from_slice(&text).map_err(|e| ModelError::Io {
        path: config_path,
        detail: e.to_string(),
    })?;
    let format = detect(&top)?;
    mcfg.weight_format = WeightFormatRef(format);
    // Like a rank load: resolve the head's format for the providers actually selected, so its
    // shared experts (the MoE path reads them BF16) are decoded, not left block-scaled FP8.
    resolve_for_providers(&mut mcfg, &slots, providers);
    let mut weights = WeightLoader::load_format(
        mcfg.weight_format.get(),
        &index,
        &slots,
        &mem,
        MAX_STAGING_BYTES,
    )?;
    let dim = cfg.hidden as usize;
    let act = mcfg.activation_dtype();
    // The pre-fc norms are `mtp.pre_fc_norm_*` (`float32_stored` false for those names), so the
    // loader's `(1 + w)` offset never touched them: fold it in now and upload in the activation
    // dtype, so the front matter needs no host round trip.
    let raw_hidden = host_f32(&weights.take("mtp.pre_fc_norm_hidden.weight")?)?;
    let raw_embedding = host_f32(&weights.take("mtp.pre_fc_norm_embedding.weight")?)?;
    let fc = weights.take("mtp.fc.weight")?;
    let pre_hidden = act_rows(&mem, act, &raw_hidden, 1.0)?;
    let pre_embedding = act_rows(&mem, act, &raw_embedding, 1.0)?;
    let mut executor = mcfg
        .family
        .0
        .build_executor(&mcfg, weights, registry, mem, limits, opts)?;
    // The MTP recurrence needs the head's own final residual rows each step.
    executor.set_hidden_capture(true, limits.max_batch_tokens as usize)?;
    executor.set_mtp_front(pre_hidden, pre_embedding, fc)?;
    Ok(MtpHead { executor, dim })
}

impl MtpHead {
    /// The head's hidden width.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// A fresh `[rows, hidden]` activation-dtype device buffer on the head's memory.
    pub fn rows(&self, rows: usize) -> Result<Tensor, ModelError> {
        self.executor.mtp_rows(rows)
    }

    /// A device view of the head executor's last `forward_mtp` output rows.
    pub fn capture(&self, rows: usize) -> Option<TensorView<'_>> {
        self.executor.mtp_capture_view(rows)
    }

    /// `dst[dst_row ..] = src[src_row ..]` rows on device.
    pub fn copy_rows(
        &self,
        dst: &Tensor,
        dst_row: usize,
        src: TensorView<'_>,
        src_row: usize,
        rows: usize,
    ) -> Result<(), ModelError> {
        self.executor
            .mtp_copy_rows(dst, dst_row, src, src_row, rows)
    }

    /// Run the MTP layer over `rows` positions, all on device: the front matter combines
    /// `hidden` (`[rows, hidden]`, the target's final rows or the head's own output) with the
    /// embedding of `tokens` (the token at each position), then the shared layer stack + final
    /// norm + LM head. `positions` are the overall positions (the MTP row for position `p` sits
    /// at KV slot `p - 1`; RoPE is shift-invariant, so the offset is exact) and `kv_len` is the
    /// MTP row count after this step's append. Returns the last position's logits row.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mtp(
        &mut self,
        kv: &turbine_tensor::KvPoolView<'_>,
        seq: turbine_core::types::SeqId,
        table: &[turbine_core::types::BlockId],
        block_formats: &[u8],
        positions: &[u32],
        kv_len: u32,
        tokens: &[u32],
        hidden: TensorView<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let logits = self.executor.mtp_step(
            kv,
            tokens,
            hidden,
            positions,
            kv_len,
            table,
            block_formats,
            seq,
        )?;
        match logits.slot(0) {
            LogitsSlot::Full(row) => Ok(row.to_vec()),
            LogitsSlot::Reduced(_) => {
                Err(invalid("the mtp head's logits were device-reduced".into()))
            }
        }
    }
}

/// A `[values.len()]` activation-dtype device tensor of `values + add`.
fn act_rows(
    mem: &Arc<dyn DeviceMemory>,
    act: DType,
    values: &[f32],
    add: f32,
) -> Result<Tensor, ModelError> {
    let t = Tensor::empty(mem, &[values.len()], act)?;
    let mut bytes = Vec::with_capacity(values.len() * act.size_bytes());
    match act {
        DType::F32 => {
            for v in values {
                bytes.extend_from_slice(&(v + add).to_le_bytes());
            }
        }
        DType::BF16 => {
            for v in values {
                bytes.extend_from_slice(&half::bf16::from_f32(v + add).to_le_bytes());
            }
        }
        DType::F16 => {
            for v in values {
                bytes.extend_from_slice(&half::f16::from_f32(v + add).to_le_bytes());
            }
        }
        other => {
            return Err(invalid(format!(
                "the mtp front matter is stored as {}",
                other.as_str()
            )));
        }
    }
    t.storage.whole().write_bytes(&bytes)?;
    Ok(t)
}

/// Read a device tensor's values back to the host as F32 (BF16, F16 or F32 storage).
fn host_f32(t: &Tensor) -> Result<Vec<f32>, ModelError> {
    let bytes = t.storage.whole().read_bytes()?;
    Ok(match t.dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => {
            return Err(invalid(format!(
                "the mtp front matter is stored as {}",
                other.as_str()
            )));
        }
    })
}
