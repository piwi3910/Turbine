//! gpt-oss attention (Phase 7 S-8): biased Q/K/V/O projections and a learned sink logit per query
//! head. The per-layer sliding window comes from the family's [`crate::families::LayerKind`]s
//! (`DecoderDims::window_left`), not from this hook.
//!
//! - Biases: every projection adds its `[out]` bias row to each output row, in the activation
//!   dtype after the GEMM rounded its output (transformers' BF16 linear rounds once; the second
//!   rounding differs by at most one BF16 ulp). There is no broadcast add in the kernel ABI, so
//!   each row runs one `add` over the row and the bias (the HIP path of Task 26 fuses the bias
//!   into the GEMM where its descriptor takes one).
//! - Sinks: `self_attn.sinks` (BF16 `[heads]`) is converted to F32 at load, the dtype of
//!   [`turbine_kernels::PagedAttentionContext::sinks`]; the logit joins the softmax denominator
//!   in the scaled-score domain (transformers' eager attention concatenates it to the scaled
//!   scores) and adds nothing to the output.
//!
//! One device only: a tensor-parallel rank would have to bias after the all-reduce and slice the
//! sinks, which no family asks for (`ModelFamily::tp_decoder_spec` is `None` for gpt-oss).
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::{ElementwiseContext, OpConfig};
use turbine_tensor::{DeviceMemory, Tensor, TensorView};

use crate::ModelError;
use crate::executor::decoder::{AttentionHook, DecoderDims, HookWeights, LayerRun, invalid};
use crate::loader::LoadedWeights;
use crate::weights::Bf16;

/// Parameter indices of [`HookWeights`]: the Q, K, V and O biases (`[q_dim]`, `[kv_dim]`,
/// `[kv_dim]`, `[hidden]`, activation dtype) and the F32 `[heads]` sinks.
const Q_BIAS: usize = 0;
const K_BIAS: usize = 1;
const V_BIAS: usize = 2;
const O_BIAS: usize = 3;
const SINKS: usize = 4;

/// Biased projections and attention sinks.
pub struct AttentionSinksWindow;

/// The [`AttentionSinksWindow`] hook.
pub const ATTENTION_SINKS_WINDOW: &dyn AttentionHook = &AttentionSinksWindow;

/// `x[r] += bias` for every row of the `[tokens, n]` view `x` (rows may be strided).
fn add_bias(run: &LayerRun<'_>, x: &TensorView<'_>, bias: &Tensor) -> Result<(), ModelError> {
    let add = run.dims.add();
    if x.strides.last() != Some(&1) {
        return Err(invalid(format!(
            "add_bias needs a last-dim-contiguous view; {:?} strides {:?}",
            x.shape.as_slice(),
            x.strides.as_slice()
        )));
    }
    let n: usize = x.shape[1..].iter().product();
    for r in 0..run.tokens {
        let row = x.rows(r, 1);
        // The row's elements are contiguous (last-dim stride 1); present the row flat so the
        // ABI's flat n-element add takes it despite the strided parent (the q slice of a fused
        // QKV projection's [tokens, q + kv + kv] output).
        let flat = TensorView::contiguous(row.slice, 0, &[n], row.dtype);
        run.op(OpConfig::Add(add), || {
            run.registry
                .elementwise(&add)
                .execute(&mut ElementwiseContext {
                    a: flat.clone(),
                    b: bias.view(),
                    out: flat.clone(),
                })
        })?;
    }
    Ok(())
}

/// The BF16 tensor `t` (`what`, expected `shape`) as an F32 one on the same memory.
pub(super) fn bf16_as_f32(t: Tensor, shape: &[usize], what: &str) -> Result<Tensor, ModelError> {
    if t.dtype != Bf16::DTYPE || t.shape.as_slice() != shape {
        return Err(invalid(format!(
            "{what} is {:?} {}, expected {shape:?} bf16",
            t.shape.as_slice(),
            t.dtype.as_str()
        )));
    }
    let numel: usize = shape.iter().product();
    let mut bytes = vec![0u8; numel * 2];
    t.storage.copy_to_host(0, &mut bytes)?;
    let values: Vec<u8> = bytes
        .chunks_exact(2)
        .flat_map(|b| {
            half::bf16::from_le_bytes([b[0], b[1]])
                .to_f32()
                .to_le_bytes()
        })
        .collect();
    let mem: &Arc<dyn DeviceMemory> = t.storage.memory();
    let mut out = Tensor::empty(mem, shape, DType::F32)?;
    out.storage.copy_from_host(0, &values)?;
    Ok(out)
}

impl AttentionHook for AttentionSinksWindow {
    fn name(&self) -> &'static str {
        "attention_sinks_window"
    }

    fn requirements(&self, d: &DecoderDims) -> Vec<OpConfig> {
        vec![OpConfig::Add(d.add())]
    }

    fn fuses_qkv(&self) -> bool {
        true
    }

    fn load_layer(
        &self,
        d: &DecoderDims,
        prefix: &str,
        weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError> {
        if d.tp.is_some() {
            return Err(invalid(
                "attention sinks and biases are not implemented under tensor parallelism".into(),
            ));
        }
        let mut bias = |proj: &str, n: usize| {
            d.take_weight(weights, &format!("{prefix}.self_attn.{proj}.bias"), &[n])
        };
        let (q, k, v, o) = (
            bias("q_proj", d.q_dim)?,
            bias("k_proj", d.kv_dim)?,
            bias("v_proj", d.kv_dim)?,
            bias("o_proj", d.hidden)?,
        );
        let name = format!("{prefix}.self_attn.sinks");
        let sinks = bf16_as_f32(weights.take(&name)?, &[d.heads], &name)?;
        Ok(HookWeights::tensors(vec![q, k, v, o, sinks]))
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

    fn after_linear(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
        v: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        add_bias(run, q, &w.0[Q_BIAS])?;
        add_bias(run, k, &w.0[K_BIAS])?;
        add_bias(run, v, &w.0[V_BIAS])
    }

    fn after_o_proj(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        proj: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        add_bias(run, proj, &w.0[O_BIAS])
    }

    fn has_sinks(&self) -> bool {
        true
    }

    fn sinks<'a>(&self, w: &'a HookWeights) -> Option<TensorView<'a>> {
        Some(w.0[SINKS].view())
    }

    fn allows_fused_decode(&self) -> bool {
        false
    }
}
