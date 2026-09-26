//! The LM head's output and its one device-to-host copy (P2 S-5, P2c S-4), shared by both
//! executors.
//!
//! Sequences whose slice asks for a reduction ([`SeqSlice::reduce`]) get their logits row
//! reduced on the device by the registry's `logits_reduce` op — top-n, log-sum-exp and, when
//! asked, a categorical draw — instead of copied whole. The final norm writes the reduced
//! sequences' rows first, so after the LM head the logits buffer holds `r` reduced rows, then
//! the `n − r` full rows; the reduction writes its results right after row `n`, and one copy
//! of the bytes from row `r` to the end of the results returns both. A 128k-entry F32 row is
//! 512 KB; its reduction is at most 524 bytes.
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::{
    KernelRegistry, LogitsReduceConfig, LogitsReduceContext, OpConfig, OpKind, OpRequirement,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, TensorView};

use super::{Logits, ReducedRow, RowReduce, SeqSlice};
use crate::ModelError;
use crate::config::ModelArchConfig;

/// The most candidates one reduced row returns (the ABI bound).
pub const MAX_TOP_N: usize = LogitsReduceConfig::MAX_TOP_N as usize;

/// 4-byte words per row of the result block at `top_n`: ids and values, then lse, sampled id
/// and sampled logit.
fn result_words(top_n: usize) -> usize {
    2 * top_n + 3
}

/// The `logits_reduce` config an executor over `cfg` runs every reduction at: the whole
/// vocabulary and [`MAX_TOP_N`] (a batch asking for fewer candidates passes fewer columns).
pub fn reduce_config(cfg: &ModelArchConfig) -> LogitsReduceConfig {
    LogitsReduceConfig {
        vocab: cfg.vocab_size,
        top_n: LogitsReduceConfig::MAX_TOP_N,
    }
}

/// The optional requirement that lets an executor over `cfg` reduce logits rows on the device
/// (not part of [`super::requirements`]: a registry without it keeps the full-row path).
pub fn reduce_requirement(cfg: &ModelArchConfig) -> OpRequirement {
    OpRequirement::from(OpConfig::LogitsReduce(reduce_config(cfg)))
}

/// Contiguous runs of the final norm: `(source token row, destination row, length)` so that
/// destination row `p` receives token row `last_rows[order[p]]`. The identity order of a batch
/// whose last rows are consecutive (every decode batch) is one run.
pub(super) fn norm_runs(last_rows: &[usize], order: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut p = 0;
    while p < order.len() {
        let src = last_rows[order[p]];
        let mut len = 1;
        while p + len < order.len() && last_rows[order[p + len]] == src + len {
            len += 1;
        }
        runs.push((src, p, len));
        p += len;
    }
    runs
}

/// The LM head's output buffer, the reduction's inputs and the per-forward row order.
pub(super) struct LogitsHead {
    vocab: usize,
    /// `Some` when the registry selected `logits_reduce` for [`reduce_config`].
    reduce: Option<LogitsReduceConfig>,
    /// F32 `[max_seqs, vocab]` rows, then the result block for `max_seqs` rows at
    /// [`MAX_TOP_N`].
    out: DeviceBuffer,
    /// Per reduced row: temperature (F32), uniform (F32), mode (I32), as three arrays.
    inputs: DeviceBuffer,
    /// The bytes of the last `inputs` upload; valid until the logits copy synchronizes.
    host_inputs: Vec<u8>,
    /// Destination row → sequence of the current forward: reduced sequences first.
    order: Vec<usize>,
    /// Sequences of the current forward, and how many of them are reduced.
    seqs: usize,
    reduced: usize,
    /// Per reduced row (in `order`), what it asked for.
    requests: Vec<RowReduce>,
}

impl LogitsHead {
    /// Device bytes for `max_seqs` rows of `vocab` logits plus the reduction's inputs and
    /// results.
    pub fn bytes(vocab: usize, max_seqs: usize) -> u64 {
        4 * (max_seqs * (vocab + result_words(MAX_TOP_N)) + 3 * max_seqs) as u64
    }

    /// Allocates the buffers for `max_seqs` rows. Rows are reduced when `registry` selected
    /// `logits_reduce` at [`reduce_config`] of `cfg` and the vocabulary is larger than
    /// [`MAX_TOP_N`].
    pub fn new(
        cfg: &ModelArchConfig,
        registry: &KernelRegistry,
        mem: &Arc<dyn DeviceMemory>,
        max_seqs: usize,
    ) -> Result<LogitsHead, ModelError> {
        let vocab = cfg.vocab_size as usize;
        let rc = reduce_config(cfg);
        let rendered = rc.to_string();
        let selected = registry
            .selections()
            .iter()
            .any(|s| s.op == OpKind::LogitsReduce && s.config == rendered);
        let out_bytes = 4 * max_seqs * (vocab + result_words(MAX_TOP_N));
        Ok(LogitsHead {
            vocab,
            reduce: (selected && vocab > MAX_TOP_N).then_some(rc),
            out: DeviceBuffer::alloc(mem, out_bytes)?,
            inputs: DeviceBuffer::alloc(mem, 12 * max_seqs)?,
            host_inputs: Vec::with_capacity(12 * max_seqs),
            order: Vec::with_capacity(max_seqs),
            seqs: 0,
            reduced: 0,
            requests: Vec::with_capacity(max_seqs),
        })
    }

    pub fn reduces(&self) -> bool {
        self.reduce.is_some()
    }

    /// Orders the batch's rows (reduced sequences first, each group in sequence order) and
    /// returns destination row → sequence, for the final norm.
    pub fn plan(&mut self, seqs: &[SeqSlice<'_>]) -> &[usize] {
        let wants = |s: &SeqSlice<'_>| self.reduce.is_some() && s.reduce.is_some();
        self.order.clear();
        self.requests.clear();
        for (i, s) in seqs.iter().enumerate() {
            if let Some(r) = s.reduce.filter(|_| wants(s)) {
                self.order.push(i);
                self.requests.push(r);
            }
        }
        self.reduced = self.order.len();
        for (i, s) in seqs.iter().enumerate() {
            if !wants(s) {
                self.order.push(i);
            }
        }
        self.seqs = seqs.len();
        &self.order
    }

    /// The LM head's output `[n, vocab]` (F32), rows in [`LogitsHead::plan`] order.
    pub fn rows(&self, n: usize) -> TensorView<'_> {
        TensorView::contiguous(self.out.whole(), 0, &[n, self.vocab], DType::F32)
    }

    /// Candidates the current batch's reduction returns per row: the most any row asked for.
    fn batch_top_n(&self) -> usize {
        self.requests
            .iter()
            .map(|r| usize::from(r.top_n).clamp(1, MAX_TOP_N))
            .max()
            .unwrap_or(1)
    }

    /// Enqueues the reduction of the planned rows (when any) and makes the iteration's one
    /// device-to-host copy (it synchronizes the stream): the full rows and the reductions.
    pub fn finish(&mut self, registry: &KernelRegistry) -> Result<Logits, ModelError> {
        let (n, r, vocab) = (self.seqs, self.reduced, self.vocab);
        let top_n = self.batch_top_n();
        let base = n * vocab;
        if let (Some(cfg), true) = (self.reduce, r > 0) {
            self.host_inputs.clear();
            for q in &self.requests {
                self.host_inputs
                    .extend_from_slice(&q.temperature.to_le_bytes());
            }
            for q in &self.requests {
                let u = q.uniform.unwrap_or(0.0);
                self.host_inputs.extend_from_slice(&u.to_le_bytes());
            }
            for q in &self.requests {
                let mode = i32::from(q.uniform.is_some());
                self.host_inputs.extend_from_slice(&mode.to_le_bytes());
            }
            self.inputs.copy_from_host(0, &self.host_inputs)?;
            let (inp, out) = (self.inputs.whole(), self.out.whole());
            let rows_f32 = |off: usize| TensorView::contiguous(inp, off, &[r], DType::F32);
            let block = |off: usize, shape: &[usize], dtype| {
                TensorView::contiguous(out, base + off, shape, dtype)
            };
            registry
                .logits_reduce(&cfg)
                .execute(&mut LogitsReduceContext {
                    logits: TensorView::contiguous(out, 0, &[r, vocab], DType::F32),
                    temperature: rows_f32(0),
                    uniform: rows_f32(r),
                    mode: TensorView::contiguous(inp, 2 * r, &[r], DType::I32),
                    top_ids: block(0, &[r, top_n], DType::I32),
                    top_values: block(r * top_n, &[r, top_n], DType::F32),
                    lse: block(2 * r * top_n, &[r], DType::F32),
                    sampled: block(2 * r * top_n + r, &[r], DType::I32),
                    sampled_logit: block(2 * r * top_n + 2 * r, &[r], DType::F32),
                    rows: r as u32,
                })?;
        }
        let result = if r > 0 { r * result_words(top_n) } else { 0 };
        let start = r * vocab;
        let raw = self
            .out
            .slice(4 * start, 4 * (base + result - start))
            .read_bytes()?;
        let words: Vec<[u8; 4]> = raw
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let full = (n - r) * vocab;
        let data: Vec<f32> = words[..full]
            .iter()
            .map(|&w| f32::from_le_bytes(w))
            .collect();
        let res = &words[full..];
        let f = |i: usize| f32::from_le_bytes(res[i]);
        let id = |i: usize| i32::from_le_bytes(res[i]);
        let reduced: Vec<ReducedRow> = self
            .requests
            .iter()
            .enumerate()
            .map(|(k, q)| {
                let want = usize::from(q.top_n).clamp(1, MAX_TOP_N);
                let top = (0..want)
                    .map(|j| (id(k * top_n + j) as u32, f(r * top_n + k * top_n + j)))
                    .collect();
                let sampled = id(2 * r * top_n + r + k);
                ReducedRow {
                    lse: f(2 * r * top_n + k),
                    top,
                    sampled: (q.uniform.is_some() && sampled >= 0)
                        .then(|| (sampled as u32, f(2 * r * top_n + 2 * r + k))),
                }
            })
            .collect();
        let mut is_reduced = vec![false; n];
        for &s in &self.order[..r] {
            is_reduced[s] = true;
        }
        Ok(Logits::mixed(vocab, data, reduced, is_reduced))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_runs_follow_the_order() {
        // Decode batch, identity order: one run.
        assert_eq!(norm_runs(&[0, 1, 2, 3], &[0, 1, 2, 3]), vec![(0, 0, 4)]);
        // Sequences 1 and 3 reduced first.
        assert_eq!(
            norm_runs(&[0, 1, 2, 3], &[1, 3, 0, 2]),
            vec![(1, 0, 1), (3, 1, 1), (0, 2, 1), (2, 3, 1)]
        );
        // A prefill chunk ends at row 9, decodes follow.
        assert_eq!(
            norm_runs(&[9, 10, 11], &[1, 2, 0]),
            vec![(10, 0, 2), (9, 2, 1)]
        );
    }
}
