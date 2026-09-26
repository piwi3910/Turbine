//! Q/K RMSNorm over the full projections (OLMoE): before RoPE, each new row's whole Q
//! projection (`heads·head_dim` elements) is RMS-normalised with `self_attn.q_norm.weight` and
//! its K projection (`kv_heads·head_dim`) with `self_attn.k_norm.weight`, in place (each row is
//! read whole before it is written, so the row-strided fused layout works too).
use turbine_kernels::OpConfig;
use turbine_tensor::TensorView;

use crate::ModelError;
use crate::executor::decoder::{AttentionHook, DecoderDims, HookWeights, LayerRun};
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
