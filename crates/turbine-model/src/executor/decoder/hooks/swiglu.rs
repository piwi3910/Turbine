//! The dense SwiGLU MLP (Llama): `down(silu(gate(h)) · up(h))`.
//!
//! Fused projections (Phase 2c, [`ExecutorOptions::fused_projections`]): the loader lays each
//! layer's gate/up weights out as one `[gate; up]` matrix
//! ([`crate::loader::gate_up_proj_name`]), so one GEMM computes both projections into one
//! `[tokens, 2·inter]` buffer and SiLU·up reads its operands as row-strided column blocks of it;
//! unfused, one GEMM per projection over row views of the same weights writes dense regions of
//! the same buffer.
use std::sync::Arc;

use turbine_kernels::{ActivationConfig, ActivationContext, GemmFusion, KernelRegistry, OpConfig};
use turbine_tensor::{DeviceMemory, Tensor};

use crate::ModelError;
use crate::executor::decoder::{DecoderDims, FfnHook, HookBuffers, HookWeights, LayerRun, rows};
use crate::executor::{ExecutorOptions, Split};
use crate::loader::LoadedWeights;

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
        max_batch_tokens as u64 * 3 * d.inter as u64 * d.act.size_bytes() as u64
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
        _layer: u32,
        prefix: &str,
        weights: &mut LoadedWeights,
        _opts: ExecutorOptions,
    ) -> Result<HookWeights, ModelError> {
        Ok(HookWeights(
            Vec::new(),
            vec![
                d.take_linear(
                    weights,
                    &format!("{prefix}.mlp.gate_up_proj.weight"),
                    &[2 * d.inter, d.hidden],
                )?,
                d.take_linear(
                    weights,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[d.hidden, d.inter],
                )?,
            ],
        ))
    }

    /// The fused gate/up projection normalises its input as it loads it and the down projection
    /// takes SiLU·up and adds into the residual (kernel ABI v2.15): BF16 weights and the card's
    /// GEMV fusing both single-row calls.
    fn fuses_norm_residual(
        &self,
        d: &DecoderDims,
        registry: &KernelRegistry,
        w: &HookWeights,
        opts: ExecutorOptions,
    ) -> bool {
        let (gate_up, down) = (w.1[GATE_UP].view(), w.1[DOWN].view());
        if !split(d, opts).fused || gate_up.q.is_some() || down.q.is_some() {
            return false;
        }
        let fuses = |n, k, fusion| {
            let cfg = d.gemm(n, k, d.act);
            registry.gemm(&cfg).fuses(&cfg, 1, false, fusion)
        };
        let norm = GemmFusion {
            rmsnorm: true,
            ..GemmFusion::default()
        };
        let silu_residual = GemmFusion {
            silu_mul: true,
            residual: true,
            ..GemmFusion::default()
        };
        fuses(2 * d.inter, d.hidden, norm) && fuses(d.hidden, d.inter, silu_residual)
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
            run.linear_normed(w.1[GATE_UP].view(), split.whole(gate_up, t))?;
        } else {
            let w = |r| w.1[GATE_UP].view().rows(r, d.inter);
            let normed = run.normed_input()?;
            run.linear(normed.clone(), w(0), gate())?;
            run.linear(normed, w(d.inter), up())?;
        }
        run.trace("gate", &gate())?;
        run.trace("up", &up())?;
        // Decode steps on the card's GEMV (kernel ABI v2.14): the down projection computes
        // SiLU·up as it loads its input, bit for bit the two ops below, one launch fewer.
        if run.linear_silu_mul(gate(), up(), w.1[DOWN].view(), run.ffn_out())? {
            return run.trace("down", &run.ffn_out());
        }
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
