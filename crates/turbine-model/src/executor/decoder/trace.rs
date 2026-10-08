//! The decoder's trace recorder ([`super::DecoderExecutor::set_trace`]): with tracing on, every
//! op output the skeleton or a hook names is read back from the device after the op (one
//! blocking read each) and kept as a [`TraceTensor`]; off, each trace point costs one branch.
use std::cell::RefCell;

use turbine_core::types::DType;
use turbine_tensor::TensorView;

use super::invalid;
use crate::ModelError;
use crate::testing::trace::RoutingRow;
use crate::weights::Bf16;

/// One intermediate tensor of a traced forward ([`super::DecoderExecutor::set_trace`]),
/// widened to f32 from its stored dtype (BF16 activations, F32 router logits and logits, I32
/// ids).
#[derive(Clone, Debug, PartialEq)]
pub struct TraceTensor {
    /// Decoder layer; `None` for `embed`, `final_norm` and `logits`.
    pub layer: Option<usize>,
    /// The op output: `embed`; per layer `attn_norm`, `q`, `k`, `v` (projections of the new
    /// rows), `q_norm`, `k_norm` (Q/K-norm attention hooks only), `q_rope`, `k_rope`, `attn`,
    /// `o_proj`, `resid_attn`, `mlp_norm`, then the FFN hook's outputs (SwiGLU: `gate`, `up`,
    /// `act`, `down`; MoE: `router_logits`, on an expert-parallel rank `topk_ids`,
    /// `topk_weights` and its own experts' `moe_partial`, then `moe_out`), then `resid_mlp`; then
    /// `final_norm` (each sequence's last row) and `logits`.
    pub name: &'static str,
    /// `[rows, cols]`: rows are the forward's tokens (one per sequence for `final_norm` and
    /// `logits`).
    pub shape: [usize; 2],
    pub data: Vec<f32>,
}

/// The recorded tensors while tracing is on (`Some`), not yet taken.
#[derive(Default)]
pub(super) struct Tracer(RefCell<Option<Vec<TraceTensor>>>);

impl Tracer {
    /// Turns tracing on (dropping anything recorded) or off.
    pub fn set(&mut self, enabled: bool) {
        *self.0.get_mut() = enabled.then(Vec::new);
    }

    pub fn is_on(&self) -> bool {
        self.0.borrow().is_some()
    }

    /// The tensors recorded since tracing was enabled or last taken; empty when it is off.
    pub fn take(&mut self) -> Vec<TraceTensor> {
        self.0
            .get_mut()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Records `view` (`[rows, …]`, rows possibly strided) under `layer` and `name` when
    /// tracing.
    pub fn record(
        &self,
        layer: Option<usize>,
        name: &'static str,
        view: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        if !self.is_on() {
            return Ok(());
        }
        let rows = view.shape[0];
        let cols = view.numel() / rows.max(1);
        let data = read_f32(view)?;
        if let Some(trace) = self.0.borrow_mut().as_mut() {
            trace.push(TraceTensor {
                layer,
                name,
                shape: [rows, cols],
                data,
            });
        }
        Ok(())
    }
}

/// The MoE routing trace ([`super::DecoderExecutor::set_routing_trace`]): while on, every MoE
/// layer reads its router logits and top-k choices back after routing (three blocking reads)
/// and keeps one [`RoutingRow`] per token; off, each MoE layer costs one branch.
#[derive(Default)]
pub(super) struct RoutingTracer(RefCell<Option<RoutingState>>);

#[derive(Default)]
struct RoutingState {
    /// The positions of the forward being run (one per batch row).
    positions: Vec<u32>,
    rows: Vec<RoutingRow>,
}

impl RoutingTracer {
    /// Turns the routing trace on (dropping anything recorded) or off.
    pub fn set(&mut self, on: bool) {
        *self.0.get_mut() = on.then(RoutingState::default);
    }

    pub fn is_on(&self) -> bool {
        self.0.borrow().is_some()
    }

    /// The rows recorded since the trace was enabled or last taken; empty when it is off.
    pub fn take(&mut self) -> Vec<RoutingRow> {
        self.0
            .get_mut()
            .as_mut()
            .map(|s| std::mem::take(&mut s.rows))
            .unwrap_or_default()
    }

    /// The positions of the batch about to run (its rows' positions).
    pub fn begin(&self, positions: &[u32]) {
        if let Some(s) = self.0.borrow_mut().as_mut() {
            s.positions = positions.to_vec();
        }
    }

    /// Records layer `layer`'s routing of the rows `[row0, row0 + tokens)`: the router logits
    /// `[tokens, experts]` (F32; rounded to BF16 before the margin when `bf16_logits`, as the
    /// router selects on them), the top-k ids (I32) and gates (F32) `[tokens, top_k]`.
    pub fn record(
        &self,
        layer: usize,
        row0: usize,
        logits: &TensorView<'_>,
        ids: &TensorView<'_>,
        gates: &TensorView<'_>,
        bf16_logits: bool,
    ) -> Result<(), ModelError> {
        if !self.is_on() {
            return Ok(());
        }
        let (tokens, experts) = (logits.shape[0], logits.numel() / logits.shape[0].max(1));
        let k = ids.numel() / tokens.max(1);
        let (logits, ids, gates) = (read_f32(logits)?, read_f32(ids)?, read_f32(gates)?);
        let mut state = self.0.borrow_mut();
        let Some(s) = state.as_mut() else {
            return Ok(());
        };
        for t in 0..tokens {
            let row: Vec<f32> = logits[t * experts..(t + 1) * experts]
                .iter()
                .map(|&l| {
                    if bf16_logits {
                        half::bf16::from_f32(l).to_f32()
                    } else {
                        l
                    }
                })
                .collect();
            let topk_ids: Vec<u32> = ids[t * k..(t + 1) * k].iter().map(|&e| e as u32).collect();
            s.rows.push(RoutingRow {
                layer: layer as u32,
                position: s
                    .positions
                    .get(row0 + t)
                    .copied()
                    .unwrap_or((row0 + t) as u32),
                margin: routing_margin(&row, &topk_ids),
                topk_ids,
                topk_gates: gates[t * k..(t + 1) * k].to_vec(),
            });
        }
        Ok(())
    }
}

/// The smallest selected logit minus the largest unselected one (`+inf` when every expert is
/// selected): how far the router's top-k set is from flipping.
fn routing_margin(row: &[f32], chosen: &[u32]) -> f32 {
    let selected = |e: usize| chosen.contains(&(e as u32));
    let low = chosen
        .iter()
        .map(|&e| row[e as usize])
        .fold(f32::INFINITY, f32::min);
    let high = (0..row.len())
        .filter(|&e| !selected(e))
        .map(|e| row[e])
        .fold(f32::NEG_INFINITY, f32::max);
    low - high
}

/// Decodes a BF16 or F32 view read from the device into f32 values, row-major (a row-strided
/// view's rows are gathered; its other dimensions must be dense).
fn read_f32(view: &TensorView<'_>) -> Result<Vec<f32>, ModelError> {
    let raw = view.slice.read_bytes()?;
    let es = view.dtype.size_bytes();
    let rows = view.shape.first().copied().unwrap_or(1);
    let row_bytes = view.numel() / rows.max(1) * es;
    let stride_bytes = view.strides.first().map_or(row_bytes, |s| s * es);
    let bytes: Vec<u8> = if stride_bytes == row_bytes {
        raw
    } else {
        (0..rows)
            .flat_map(|r| &raw[r * stride_bytes..r * stride_bytes + row_bytes])
            .copied()
            .collect()
    };
    Ok(match view.dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Bf16::DTYPE => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        // Ids (e.g. the MoE router's top-k choices), exact below 2^24.
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        other => return Err(invalid(format!("trace of a {other:?} tensor"))),
    })
}
