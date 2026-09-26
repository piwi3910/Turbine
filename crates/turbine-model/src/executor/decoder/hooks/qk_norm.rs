//! Q/K RMSNorm over the full projections (OLMoE): before RoPE, each new row's whole Q
//! projection (`heads·head_dim` elements) is RMS-normalised with `self_attn.q_norm.weight` and
//! its K projection (`kv_heads·head_dim`) with `self_attn.k_norm.weight`, in place (each row is
//! read whole before it is written, so the row-strided fused layout works too).
//!
//! Q/K RMSNorm per head (Qwen3): [`QkNormPerHead`], below.
use turbine_kernels::OpConfig;
use turbine_tensor::TensorView;

use crate::ModelError;
use crate::executor::decoder::{AttentionHook, DecoderDims, HookWeights, LayerRun, invalid};
use crate::loader::LoadedWeights;

/// Parameter indices of [`HookWeights`].
const Q_NORM: usize = 0;
const K_NORM: usize = 1;

/// RMSNorm over the full Q and K projections.
pub struct QkNormFull;

/// The [`QkNormFull`] hook.
pub const QK_NORM_FULL: &dyn AttentionHook = &QkNormFull;

impl AttentionHook for QkNormFull {
    fn name(&self) -> &'static str {
        "qk_norm_full"
    }

    fn requirements(&self, d: &DecoderDims) -> Vec<OpConfig> {
        vec![
            OpConfig::Rmsnorm(d.norm(d.q_dim)),
            OpConfig::Rmsnorm(d.norm(d.kv_dim)),
        ]
    }

    fn fuses_qkv(&self) -> bool {
        true
    }

    fn load_layer(
        &self,
        _d: &DecoderDims,
        prefix: &str,
        weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError> {
        Ok(HookWeights(vec![
            weights.take(&format!("{prefix}.self_attn.q_norm.weight"))?,
            weights.take(&format!("{prefix}.self_attn.k_norm.weight"))?,
        ]))
    }

    fn after_projections(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        run.rmsnorm(q.clone(), &w.0[Q_NORM], q.clone())?;
        run.trace("q_norm", q)?;
        run.rmsnorm(k.clone(), &w.0[K_NORM], k.clone())?;
        run.trace("k_norm", k)
    }
}

/// RMSNorm over each Q and K head (Qwen3, Qwen3-MoE): before RoPE, every `head_dim` elements
/// of each new row's Q projection are RMS-normalised with `self_attn.q_norm.weight` and every
/// `head_dim` elements of its K projection with `self_attn.k_norm.weight` (both `[head_dim]`),
/// in place, as `[tokens · heads, head_dim]` rows. The norm op takes one row stride, and in the
/// fused Q/K/V layout a token's heads are strided by the fused width, so this hook keeps the
/// projections dense ([`AttentionHook::fuses_qkv`] false: one GEMM per projection); rows never
/// mix tokens, so ragged batches stay independent.
pub struct QkNormPerHead;

/// The [`QkNormPerHead`] hook.
pub const QK_NORM_PER_HEAD: &dyn AttentionHook = &QkNormPerHead;

/// A dense `[tokens, heads · head_dim]` projection view as `[tokens · heads, head_dim]` rows.
fn head_rows<'a>(view: &TensorView<'a>, head_dim: usize) -> Result<TensorView<'a>, ModelError> {
    let (tokens, cols) = match view.shape.as_slice() {
        &[tokens, cols] => (tokens, cols),
        other => {
            return Err(invalid(format!(
                "per-head Q/K norm needs a [tokens, heads·head_dim] view, got {other:?}"
            )));
        }
    };
    if view.strides.as_slice() != [cols, 1] || !cols.is_multiple_of(head_dim) {
        return Err(invalid(format!(
            "per-head Q/K norm needs a dense projection of whole heads of {head_dim}, got shape \
             {:?} strides {:?}",
            view.shape.as_slice(),
            view.strides.as_slice()
        )));
    }
    Ok(TensorView::contiguous(
        view.slice,
        0,
        &[tokens * cols / head_dim, head_dim],
        view.dtype,
    ))
}

impl AttentionHook for QkNormPerHead {
    fn name(&self) -> &'static str {
        "qk_norm_per_head"
    }

    fn requirements(&self, d: &DecoderDims) -> Vec<OpConfig> {
        vec![OpConfig::Rmsnorm(d.norm(d.head_dim))]
    }

    fn fuses_qkv(&self) -> bool {
        false
    }

    fn load_layer(
        &self,
        d: &DecoderDims,
        prefix: &str,
        weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError> {
        let mut take = |proj: &str| {
            d.take_weight(
                weights,
                &format!("{prefix}.self_attn.{proj}.weight"),
                &[d.head_dim],
            )
        };
        Ok(HookWeights(vec![take("q_norm")?, take("k_norm")?]))
    }

    fn after_projections(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        let head_dim = run.dims.head_dim;
        let (qr, kr) = (head_rows(q, head_dim)?, head_rows(k, head_dim)?);
        run.rmsnorm(qr.clone(), &w.0[Q_NORM], qr)?;
        run.trace("q_norm", q)?;
        run.rmsnorm(kr.clone(), &w.0[K_NORM], kr)?;
        run.trace("k_norm", k)
    }
}
