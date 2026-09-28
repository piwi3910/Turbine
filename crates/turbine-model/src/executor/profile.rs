//! Op-level forward profile (P2c S-2): with profile mode on, an executor synchronises the stream
//! after every registry op and accumulates the wall time of each (op kind, implementation) pair
//! into an [`OpProfile`]. The executor's own device transfers — the batch metadata upload, the
//! OLMoE expert-offset read-back and accumulator reset, the logits copy — are timed the same way
//! under their own names ([`BATCH_UPLOAD`], [`MOE_OFFSETS_READ`], [`MOE_ZERO`], [`LOGITS_READ`]),
//! so the profile accounts for the whole forward. Off by default; when off an op costs one
//! branch and no timer or synchronisation.
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Serialize;
use turbine_kernels::{KernelRegistry, OpConfig, OpRequirement};
use turbine_tensor::DeviceMemory;

use crate::ModelError;

/// Packing the batch on the host and uploading its metadata (ids, positions, `q_indptr`,
/// `kv_lens`, block tables); implementation [`HOST`].
pub const BATCH_UPLOAD: &str = "batch_upload";
/// The FP32 logits copy to the host at the end of the forward; implementation [`D2H`].
pub const LOGITS_READ: &str = "logits_read";
/// OLMoE: the blocking read of the `[experts + 1]` expert offsets `moe_experts` needs on the
/// host; implementation [`D2H`].
pub const MOE_OFFSETS_READ: &str = "moe_offsets_read";
/// OLMoE: resetting the BF16 expert accumulator to zero (`0 + 0` through the registry's
/// `add`: kernel ABI v2 has no device-to-device copy); implementation [`ZERO_ADD`].
pub const MOE_ZERO: &str = "moe_zero";
/// Tensor parallelism: an all-reduce of the rank's partial sums (embedding, O and down
/// projections, sharded norms); implementation: the collective backend.
pub const TP_ALL_REDUCE: &str = "tp_all_reduce";
/// Expert parallelism at tp = 1: the all-reduce combining the ranks' MoE outputs;
/// implementation: the collective backend.
pub const EP_COMBINE: &str = "ep_combine";
/// Tensor parallelism: the all-gather of the ranks' LM-head shards; implementation: the
/// collective backend.
pub const TP_ALL_GATHER: &str = "tp_all_gather";
/// Tensor parallelism: the device-to-device copies reordering the gathered shards into
/// row-major logits rows; implementation [`D2D`].
pub const TP_LOGITS_REORDER: &str = "tp_logits_reorder";
/// Pipeline parallelism: sending the residual rows to the next stage; implementation: the
/// collective backend.
pub const PP_SEND: &str = "pp_send";
/// Pipeline parallelism: receiving the residual rows from the stage before; implementation: the
/// collective backend.
pub const PP_RECV: &str = "pp_recv";
/// Implementation name of device-to-device copies.
pub const D2D: &str = "d2d";
/// Implementation name of host work followed by a host-to-device copy.
pub const HOST: &str = "host";
/// Implementation name of a device-to-host copy.
pub const D2H: &str = "d2h";
/// Implementation name of the accumulator reset as an `add` of two zero buffers.
pub const ZERO_ADD: &str = "add";

/// One (op, implementation) pair of a profile: how often it ran and its accumulated wall time,
/// each call measured from its launch until the stream drained.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpProfileEntry {
    /// The op kind (`gemm`, `attention_decode_paged`, …) or one of the executor's transfer
    /// names ([`BATCH_UPLOAD`], [`LOGITS_READ`], [`MOE_OFFSETS_READ`], [`MOE_ZERO`]).
    pub op: String,
    /// The selected implementation (`hipblaslt`, `ck_tile_fmha_pagedkv`, …), or [`HOST`],
    /// [`D2H`], [`ZERO_ADD`] for the executor's own steps.
    pub r#impl: String,
    pub calls: u32,
    pub total_ms: f64,
}

/// The per-op wall times of the forwards run since profiling was enabled or last taken, in
/// first-call order.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpProfile {
    pub entries: Vec<OpProfileEntry>,
}

impl OpProfile {
    /// The sum of every entry's `total_ms`.
    pub fn total_ms(&self) -> f64 {
        self.entries.iter().map(|e| e.total_ms).sum()
    }

    /// The entry of `op` with implementation `imp`, if it ran.
    pub fn entry(&self, op: &str, imp: &str) -> Option<&OpProfileEntry> {
        self.entries.iter().find(|e| e.op == op && e.r#impl == imp)
    }

    /// Calls of `op` summed over its implementations.
    pub fn calls(&self, op: &str) -> u32 {
        self.entries
            .iter()
            .filter(|e| e.op == op)
            .map(|e| e.calls)
            .sum()
    }

    /// Wall time of `op` summed over its implementations, in milliseconds.
    pub fn op_ms(&self, op: &str) -> f64 {
        self.entries
            .iter()
            .filter(|e| e.op == op)
            .map(|e| e.total_ms)
            .sum()
    }

    fn add(&mut self, op: &str, imp: &str, elapsed: Duration) {
        let ms = elapsed.as_secs_f64() * 1e3;
        match self
            .entries
            .iter_mut()
            .find(|e| e.op == op && e.r#impl == imp)
        {
            Some(e) => {
                e.calls += 1;
                e.total_ms += ms;
            }
            None => self.entries.push(OpProfileEntry {
                op: op.to_string(),
                r#impl: imp.to_string(),
                calls: 1,
                total_ms: ms,
            }),
        }
    }
}

/// An executor's profile state: `None` while profiling is off.
#[derive(Default)]
pub(super) struct Profiler {
    state: RefCell<Option<Recording>>,
}

struct Recording {
    /// The implementation the registry selected for every op config the executor runs.
    impls: HashMap<OpConfig, String>,
    profile: OpProfile,
}

impl Profiler {
    /// True while profiling (every op then synchronises the stream, so nothing may be captured).
    pub(super) fn is_on(&self) -> bool {
        self.state.borrow().is_some()
    }

    /// Turns profiling on (keeping what was recorded if it already was) or off (dropping it).
    /// `reqs` are the executor's requirements, which `registry` was built from.
    pub(super) fn set(&self, on: bool, registry: &KernelRegistry, reqs: &[OpRequirement]) {
        let mut state = self.state.borrow_mut();
        if !on {
            *state = None;
            return;
        }
        if state.is_some() {
            return;
        }
        let impls = reqs
            .iter()
            .filter_map(|req| {
                registry
                    .selections()
                    .iter()
                    .find(|s| s.op == req.op && s.config == req.config)
                    .map(|s| (req.spec, s.implementation.clone()))
            })
            .collect();
        *state = Some(Recording {
            impls,
            profile: OpProfile::default(),
        });
    }

    /// The profile recorded since profiling was enabled or last taken; empty when it is off.
    pub(super) fn take(&self) -> OpProfile {
        self.state
            .borrow_mut()
            .as_mut()
            .map(|r| std::mem::take(&mut r.profile))
            .unwrap_or_default()
    }

    /// Runs the registry op `spec` through `f`; when profiling, synchronises `mem` afterwards
    /// and records the wall time under the op kind and its selected implementation.
    pub(super) fn op<R>(
        &self,
        mem: &dyn DeviceMemory,
        spec: OpConfig,
        f: impl FnOnce() -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        if self.state.borrow().is_none() {
            return f();
        }
        let start = Instant::now();
        let out = f()?;
        mem.synchronize()?;
        let elapsed = start.elapsed();
        if let Some(Recording { impls, profile }) = self.state.borrow_mut().as_mut() {
            let imp = impls.get(&spec).map_or("unselected", String::as_str);
            profile.add(spec.op().as_str(), imp, elapsed);
        }
        Ok(out)
    }

    /// Runs the executor transfer `op` (implementation `imp`) through `f`, timed like
    /// [`Profiler::op`].
    pub(super) fn step<R>(
        &self,
        mem: &dyn DeviceMemory,
        op: &'static str,
        imp: &'static str,
        f: impl FnOnce() -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        if self.state.borrow().is_none() {
            return f();
        }
        let start = Instant::now();
        let out = f()?;
        mem.synchronize()?;
        let elapsed = start.elapsed();
        if let Some(r) = self.state.borrow_mut().as_mut() {
            r.profile.add(op, imp, elapsed);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_accumulate_per_op_and_impl() {
        let mut p = OpProfile::default();
        p.add("gemm", "a", Duration::from_millis(2));
        p.add("add", "a", Duration::from_millis(1));
        p.add("gemm", "a", Duration::from_millis(3));
        p.add("gemm", "b", Duration::from_millis(4));
        assert_eq!(p.entries.len(), 3);
        assert_eq!(p.entry("gemm", "a").map(|e| e.calls), Some(2));
        assert_eq!(p.calls("gemm"), 3);
        assert!((p.op_ms("gemm") - 9.0).abs() < 1e-9);
        assert!((p.total_ms() - 10.0).abs() < 1e-9);
        let json = serde_json::to_value(&p).expect("serialize");
        assert_eq!(json["entries"][0]["impl"], "a");
        assert_eq!(json["entries"][0]["calls"], 2);
    }
}
