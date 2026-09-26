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
use std::collections::VecDeque;
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::{
    KernelError, KernelRegistry, LogitsReduceConfig, LogitsReduceContext, OpConfig, OpKind,
    OpRequirement,
};
use turbine_tensor::{DeviceBuffer, DeviceMemory, TensorView};

use super::batch::StagingPair;
use super::profile::{self, Profiler};
use super::{Logits, ReducedRow, RowReduce, SeqSlice};
use crate::ModelError;
use crate::config::ModelArchConfig;

/// The most candidates one reduced row returns (the ABI bound).
pub const MAX_TOP_N: usize = LogitsReduceConfig::MAX_TOP_N as usize;

fn invalid(message: String) -> ModelError {
    ModelError::Kernel(KernelError::InvalidArgument { message })
}

/// 4-byte words per row of the reduction's inputs: temperature, uniform, top_p and mode.
const INPUT_WORDS: usize = 4;

/// 4-byte words per row of the result block at `top_n`: ids and values, then lse, sampled id
/// and sampled logit.
fn result_words(top_n: usize) -> usize {
    2 * top_n + 3
}

/// Candidates a reduction of rows asking for `requests` returns per row: the most any row asked
/// for (1 when none did).
fn top_n_of(requests: &[RowReduce]) -> usize {
    requests
        .iter()
        .map(|r| usize::from(r.top_n).clamp(1, MAX_TOP_N))
        .max()
        .unwrap_or(1)
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

/// The LM head's output buffer, the reduction's inputs, the per-forward row order and the
/// logits reads not yet collected.
///
/// A forward's read is enqueued at the end of its launch ([`LogitsHead::launch_read`]) and
/// taken by [`LogitsHead::collect`]. With host staging (kernel ABI v2.3) the reduction inputs and
/// the read go through double-buffered staging and never wait for the stream, so up to two
/// forwards may be in flight: the second launch's feeds read the first one's chosen tokens on
/// the device ([`LogitsHead::feed_word`]). Without staging the read is a synchronous copy at
/// collect time and one forward at a time is in flight.
pub(super) struct LogitsHead {
    vocab: usize,
    /// `Some` when the registry selected `logits_reduce` for [`reduce_config`].
    reduce: Option<LogitsReduceConfig>,
    /// F32 `[max_seqs, vocab]` rows, then the result block for `max_seqs` rows at
    /// [`MAX_TOP_N`].
    out: DeviceBuffer,
    /// Per reduced row: temperature (F32), uniform (F32), top_p (F32), mode (I32), as four
    /// arrays.
    inputs: DeviceBuffer,
    /// The bytes of the last `inputs` upload.
    host_inputs: Vec<u8>,
    /// Staging for the `inputs` upload and for the reads; `None` without host staging.
    input_staging: Option<StagingPair>,
    read_staging: Option<StagingPair>,
    /// Destination row → sequence of the current forward: reduced sequences first.
    order: Vec<usize>,
    /// Sequences of the current forward, and how many of them are reduced.
    seqs: usize,
    reduced: usize,
    /// Per reduced row (in `order`), what it asked for.
    requests: Vec<RowReduce>,
    /// Reads enqueued and not yet collected, oldest first.
    pending: VecDeque<PendingRead>,
    /// Per sequence of the last launched forward: the word of `out` holding the token the device
    /// chose for it, when the device's choice is final ([`device_choice_word`]).
    last_tokens: Vec<Option<usize>>,
}

/// One forward's logits read: what `collect` needs to decode it.
struct PendingRead {
    n: usize,
    /// Reduced rows and the candidates each returns.
    r: usize,
    top_n: usize,
    requests: Vec<RowReduce>,
    is_reduced: Vec<bool>,
    /// Words of `out` read, from row `r` on: the full rows, then the results.
    words: usize,
    /// The staging buffer holding the read; `None`: read synchronously at collect.
    staged: Option<usize>,
}

/// The word of the result block (at word `base`, `r` rows of `top_n` candidates) holding the
/// token the device chose for reduced row `k` asking for `q`, when that choice is final: the
/// categorical draw when one was asked for, the best candidate for a greedy row. A `top_k` row
/// (host draw over the candidates) has none. The host takes these same tokens
/// ([`crate::Sampler::finish_reduced`]) unless no logit of the row is finite.
fn device_choice_word(
    base: usize,
    r: usize,
    top_n: usize,
    k: usize,
    q: &RowReduce,
) -> Option<usize> {
    if q.uniform.is_some() {
        Some(base + 2 * r * top_n + r + k)
    } else if q.temperature <= 0.0 {
        Some(base + k * top_n)
    } else {
        None
    }
}

impl LogitsHead {
    /// Device bytes for `max_seqs` rows of `vocab` logits plus the reduction's inputs and
    /// results.
    pub fn bytes(vocab: usize, max_seqs: usize) -> u64 {
        4 * (max_seqs * (vocab + result_words(MAX_TOP_N)) + INPUT_WORDS * max_seqs) as u64
    }

    /// Allocates the buffers for `max_seqs` rows (and, when `mem` has staging, the staging for
    /// the inputs and for reads of the result block; a read of full rows grows it). Rows are
    /// reduced when `registry` selected `logits_reduce` at [`reduce_config`] of `cfg` and the
    /// vocabulary is larger than [`MAX_TOP_N`].
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
        let input_bytes = 4 * INPUT_WORDS * max_seqs;
        Ok(LogitsHead {
            vocab,
            reduce: (selected && vocab > MAX_TOP_N).then_some(rc),
            out: DeviceBuffer::alloc(mem, out_bytes)?,
            inputs: DeviceBuffer::alloc(mem, input_bytes)?,
            host_inputs: Vec::with_capacity(input_bytes),
            input_staging: StagingPair::alloc(mem, input_bytes)?,
            read_staging: StagingPair::alloc(mem, 4 * max_seqs * result_words(MAX_TOP_N))?,
            order: Vec::with_capacity(max_seqs),
            seqs: 0,
            reduced: 0,
            requests: Vec::with_capacity(max_seqs),
            pending: VecDeque::with_capacity(2),
            last_tokens: Vec::with_capacity(max_seqs),
        })
    }

    pub fn reduces(&self) -> bool {
        self.reduce.is_some()
    }

    /// The reads never wait for the stream, so two forwards may be in flight.
    pub fn overlaps(&self) -> bool {
        self.input_staging.is_some() && self.read_staging.is_some()
    }

    /// Reads enqueued and not collected.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// The word of `out` holding the token the device chose for sequence `slot` of the last
    /// launched forward; `None` when that row was not reduced or its choice is the host's.
    pub fn feed_word(&self, slot: usize) -> Option<usize> {
        self.last_tokens.get(slot).copied().flatten()
    }

    /// `len` token ids from word `word` of `out`, as the embedding's I32 ids.
    pub fn ids_at(&self, word: usize, len: usize) -> TensorView<'_> {
        TensorView::contiguous(self.out.whole(), word, &[len], DType::I32)
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
        top_n_of(&self.requests)
    }

    /// [`LogitsHead::graph_top_n`] of the batch `seqs` as [`LogitsHead::plan`] would plan it,
    /// without planning it.
    pub fn graph_top_n_of(&self, seqs: &[SeqSlice<'_>]) -> Option<u8> {
        let requests: Vec<RowReduce> = seqs
            .iter()
            .filter_map(|s| s.reduce.filter(|_| self.reduce.is_some()))
            .collect();
        match requests.len() {
            0 => Some(0),
            r if r == seqs.len() => Some(top_n_of(&requests) as u8),
            _ => None,
        }
    }

    /// The reduction's candidates per row when the planned batch can run as a decode graph
    /// ([`super::graphs::GraphKey::reduce_top_n`]): 0 when no row is reduced, the batch's
    /// candidate count when every row is; `None` for a batch mixing both (its final norm runs
    /// in several pieces, so it runs eagerly).
    pub fn graph_top_n(&self) -> Option<u8> {
        match self.reduced {
            0 => Some(0),
            r if r == self.seqs => Some(self.batch_top_n() as u8),
            _ => None,
        }
    }

    /// Uploads the planned rows' reduction inputs (when any row is reduced), before the forward's
    /// first kernel so that nothing after the last kernel waits for the host; the reduction
    /// reads them from the same device buffer every iteration, so a captured decode graph
    /// replays with this iteration's values.
    pub fn upload_inputs(&mut self) -> Result<(), ModelError> {
        if self.reduce.is_none() || self.reduced == 0 {
            return Ok(());
        }
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
            let top_p = if q.uniform.is_some() { q.top_p } else { 1.0 };
            self.host_inputs.extend_from_slice(&top_p.to_le_bytes());
        }
        for q in &self.requests {
            let mode = i32::from(q.uniform.is_some());
            self.host_inputs.extend_from_slice(&mode.to_le_bytes());
        }
        let mem = Arc::clone(self.inputs.memory());
        match self.input_staging.as_mut() {
            Some(pair) => {
                let i = pair.next(&mem, self.host_inputs.len(), self.inputs.len())?;
                let staging = pair.buf(i);
                staging.write(0, &self.host_inputs)?;
                staging.upload(0, self.inputs.slice(0, self.host_inputs.len()))?;
            }
            None => self.inputs.copy_from_host(0, &self.host_inputs)?,
        }
        Ok(())
    }

    /// Enqueues the reduction of the planned rows (when any), after the LM head, through
    /// `profiler` (`logits_reduce`) on `mem`; its inputs were uploaded by
    /// [`LogitsHead::upload_inputs`]. Only an op call, so a decode graph can capture it.
    pub fn reduce(
        &self,
        registry: &KernelRegistry,
        profiler: &Profiler,
        mem: &dyn DeviceMemory,
    ) -> Result<(), ModelError> {
        let (n, r, vocab) = (self.seqs, self.reduced, self.vocab);
        let top_n = self.batch_top_n();
        let base = n * vocab;
        if let (Some(cfg), true) = (self.reduce, r > 0) {
            let (inp, out) = (self.inputs.whole(), self.out.whole());
            let rows_f32 = |off: usize| TensorView::contiguous(inp, off, &[r], DType::F32);
            let block = |off: usize, shape: &[usize], dtype| {
                TensorView::contiguous(out, base + off, shape, dtype)
            };
            profiler.op(mem, OpConfig::LogitsReduce(cfg), || {
                registry
                    .logits_reduce(&cfg)
                    .execute(&mut LogitsReduceContext {
                        logits: TensorView::contiguous(out, 0, &[r, vocab], DType::F32),
                        temperature: rows_f32(0),
                        uniform: rows_f32(r),
                        top_p: rows_f32(2 * r),
                        mode: TensorView::contiguous(inp, 3 * r, &[r], DType::I32),
                        top_ids: block(0, &[r, top_n], DType::I32),
                        top_values: block(r * top_n, &[r, top_n], DType::F32),
                        lse: block(2 * r * top_n, &[r], DType::F32),
                        sampled: block(2 * r * top_n + r, &[r], DType::I32),
                        sampled_logit: block(2 * r * top_n + 2 * r, &[r], DType::F32),
                        rows: r as u32,
                    })
                    .map_err(ModelError::from)
            })?;
        }
        Ok(())
    }

    /// After [`LogitsHead::reduce`]: enqueues the forward's one device-to-host read of the full
    /// rows and the reductions (through staging; without it the read happens at
    /// [`LogitsHead::collect`]), and records which words hold the device's token choices for
    /// the next forward's feeds.
    pub fn launch_read(&mut self) -> Result<(), ModelError> {
        let (n, r, vocab) = (self.seqs, self.reduced, self.vocab);
        let top_n = self.batch_top_n();
        let base = n * vocab;
        let result = if r > 0 { r * result_words(top_n) } else { 0 };
        let words = (n - r) * vocab + result;
        let src = self.out.slice(4 * r * vocab, 4 * words);
        let staged = match self.read_staging.as_mut() {
            Some(pair) => {
                let i = pair.next(self.out.memory(), 4 * words, self.out.len())?;
                pair.buf(i).download(0, src)?;
                Some(i)
            }
            None => None,
        };
        let mut is_reduced = vec![false; n];
        for &s in &self.order[..r] {
            is_reduced[s] = true;
        }
        self.last_tokens.clear();
        self.last_tokens.resize(n, None);
        if self.reduce.is_some() {
            for (k, q) in self.requests.iter().enumerate() {
                self.last_tokens[self.order[k]] = device_choice_word(base, r, top_n, k, q);
            }
        }
        self.pending.push_back(PendingRead {
            n,
            r,
            top_n,
            requests: self.requests.clone(),
            is_reduced,
            words,
            staged,
        });
        Ok(())
    }

    /// The oldest launched forward's logits: waits for its read (only its own; later forwards
    /// keep running) and decodes the full rows and the reductions. The read is timed through
    /// `profiler` ([`profile::LOGITS_READ`]) on `mem`.
    pub fn collect(
        &mut self,
        profiler: &Profiler,
        mem: &dyn DeviceMemory,
    ) -> Result<Logits, ModelError> {
        let p = self
            .pending
            .pop_front()
            .ok_or_else(|| invalid("collect without a launched forward".into()))?;
        let (n, r, top_n, vocab) = (p.n, p.r, p.top_n, self.vocab);
        let full = (n - r) * vocab;
        // The read and its decoding into host words and F32 rows, timed together.
        let (words, data) = profiler.step(mem, profile::LOGITS_READ, profile::D2H, || {
            let raw = match (p.staged, self.read_staging.as_ref()) {
                (Some(i), Some(pair)) => {
                    let mut raw = vec![0u8; 4 * p.words];
                    pair.buf(i).read(0, &mut raw)?;
                    raw
                }
                _ => self.out.slice(4 * r * vocab, 4 * p.words).read_bytes()?,
            };
            let words: Vec<[u8; 4]> = raw
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect();
            let data: Vec<f32> = words[..full]
                .iter()
                .map(|&w| f32::from_le_bytes(w))
                .collect();
            Ok((words, data))
        })?;
        let res = &words[full..];
        let f = |i: usize| f32::from_le_bytes(res[i]);
        let id = |i: usize| i32::from_le_bytes(res[i]);
        let reduced: Vec<ReducedRow> = p
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
        Ok(Logits::mixed(vocab, data, reduced, p.is_reduced))
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
