//! Plain attention (Llama): RoPE runs straight on the Q and K projections.
use turbine_kernels::OpConfig;
use turbine_tensor::TensorView;

use crate::ModelError;
use crate::executor::decoder::{AttentionHook, DecoderDims, HookWeights, LayerRun};
use crate::loader::LoadedWeights;

/// No op and no parameter between the projections and RoPE.
pub struct PlainAttention;

/// The [`PlainAttention`] hook.
pub const PLAIN_ATTENTION: &dyn AttentionHook = &PlainAttention;

impl AttentionHook for PlainAttention {
    fn name(&self) -> &'static str {
        "plain_attention"
    }

    fn requirements(&self, _d: &DecoderDims) -> Vec<OpConfig> {
        Vec::new()
    }

    fn fuses_qkv(&self) -> bool {
        true
    }

    fn load_layer(
        &self,
        _d: &DecoderDims,
        _prefix: &str,
        _weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError> {
        Ok(HookWeights(Vec::new()))
    }

    fn after_projections(
        &self,
        _run: &LayerRun<'_>,
        _w: &HookWeights,
        _q: &TensorView<'_>,
        _k: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        Ok(())
    }
}
