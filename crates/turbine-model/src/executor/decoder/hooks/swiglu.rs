//! The dense SwiGLU MLP (Llama): `down(silu(gate(h)) · up(h))`.
//!
//! Fused projections (Phase 2c, [`ExecutorOptions::fused_projections`]): the loader lays each
//! layer's gate/up weights out as one `[gate; up]` matrix
//! ([`crate::loader::gate_up_proj_name`]), so one GEMM computes both projections into one
//! `[tokens, 2·inter]` buffer and SiLU·up reads its operands as row-strided column blocks of it;
//! unfused, one GEMM per projection over row views of the same weights writes dense regions of
//! the same buffer.
use std::sync::Arc;

use turbine_kernels::{ActivationConfig, ActivationContext, OpConfig};
use turbine_tensor::{DeviceMemory, Tensor};

use crate::ModelError;
use crate::executor::decoder::{DecoderDims, FfnHook, HookBuffers, HookWeights, LayerRun, rows};
use crate::executor::{ExecutorOptions, Split};
use crate::loader::{LoadedWeights, gate_up_proj_name};

/// Linear-layer indices of [`HookWeights`] (BF16 or quantized, [`DecoderDims::take_linear`]):
/// `[2·inter, hidden]` gate rows then up rows, and `[hidden, inter]` down.
const GATE_UP: usize = 0;
const DOWN: usize = 1;
/// Buffer indices of [`HookBuffers::tensors`]: the gate/up projections (`[tokens, 2·inter]`,
/// laid out by [`split`]) and SiLU·up (`[tokens, inter]`).
const GATE_UP_BUF: usize = 0;
const ACT_BUF: usize = 1;

/// The dense SwiGLU MLP.
pub struct SwiGlu;

/// The [`SwiGlu`] hook.
pub const SWIGLU: &dyn FfnHook = &SwiGlu;

/// Gate and up (`[inter, inter]`) in their buffer, fused or not.
fn split(d: &DecoderDims, opts: ExecutorOptions) -> Split<2> {
    Split {
        cols: [d.inter, d.inter],
        fused: opts.fused_projections,
    }
}

fn activation_cfg(d: &DecoderDims) -> ActivationConfig {
    ActivationConfig {
        cols: d.inter as u64,
        dtype: d.act,
    }
}

impl FfnHook for SwiGlu {
    fn name(&self) -> &'static str {
        "swiglu"
    }

    fn requirements(&self, d: &DecoderDims, opts: ExecutorOptions) -> Vec<OpConfig> {
        // Fused: one GEMM over `[gate; up]`; unfused: one per projection, both `[inter, hidden]`.
        let gate_up_rows = if split(d, opts).fused {
            2 * d.inter
        } else {
            d.inter
        };
        let mut ops = d.linear_ops(gate_up_rows, d.hidden);
        ops.push(OpConfig::SiluMul(activation_cfg(d)));
        ops.extend(d.linear_ops(d.hidden, d.inter));
        ops
    }

    /// Per token the gate, up and SiLU·up rows (`3·inter` activation-dtype elements).
    fn workspace_bytes(&self, d: &DecoderDims, max_batch_tokens: usize) -> u64 {
        (max_batch_tokens * 3 * d.inter * d.act.size_bytes()) as u64
    }

    fn alloc(
        &self,
        d: &DecoderDims,
        max_batch_tokens: usize,
        mem: &Arc<dyn DeviceMemory>,
    ) -> Result<HookBuffers, ModelError> {
        let t = max_batch_tokens;
        Ok(HookBuffers {
            tensors: vec![
                Tensor::empty(mem, &[t, 2 * d.inter], d.act)?,
                Tensor::empty(mem, &[t, d.inter], d.act)?,
            ],
            scratch: Vec::new(),
        })
    }

    fn load_layer(
        &self,
        d: &DecoderDims,
        layer: u32,
        prefix: &str,
        weights: &mut LoadedWeights,
        _opts: ExecutorOptions,
    ) -> Result<HookWeights, ModelError> {
        Ok(HookWeights(
            Vec::new(),
            vec![
                d.take_linear(weights, &gate_up_proj_name(layer), &[2 * d.inter, d.hidden])?,
                d.take_linear(
                    weights,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[d.hidden, d.inter],
                )?,
            ],
        ))
    }

    fn forward(&self, run: &LayerRun<'_>, w: &HookWeights) -> Result<(), ModelError> {
        let d = run.dims;
        let t = run.tokens;
        let (gate_up, act) = (
            &run.ffn_buffers.tensors[GATE_UP_BUF],
            &run.ffn_buffers.tensors[ACT_BUF],
        );
        let split = split(d, run.opts);
        let gate = || split.part(gate_up, 0, t, &[d.inter]);
        let up = || split.part(gate_up, 1, t, &[d.inter]);
        if split.fused {
            run.linear(run.normed(), w.1[GATE_UP].view(), split.whole(gate_up, t))?;
        } else {
            let w = |r| w.1[GATE_UP].view().rows(r, d.inter);
            run.linear(run.normed(), w(0), gate())?;
            run.linear(run.normed(), w(d.inter), up())?;
        }
        run.trace("gate", &gate())?;
        run.trace("up", &up())?;
        let silu = activation_cfg(d);
        run.op(OpConfig::SiluMul(silu), || {
            run.registry
                .activation(&silu)
                .execute(&mut ActivationContext {
                    gate: gate(),
                    up: up(),
                    out: rows(act, t),
                })
        })?;
        run.trace("act", &rows(act, t))?;
        run.linear(rows(act, t), w.1[DOWN].view(), run.ffn_out())?;
        run.trace("down", &run.ffn_out())
    }
}
