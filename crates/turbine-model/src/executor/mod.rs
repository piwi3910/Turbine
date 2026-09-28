//! Model executors: the forward pass over kernel-registry ops (TS §6). Phase 2: ragged batches
//! of sequences whose KV lives in the paged L0 block pool (`KvPoolView`), one FP32 logits row
//! per sequence and one device-to-host copy per iteration (P2 S-5, S-9).
use std::sync::Arc;
use std::time::Duration;

use turbine_core::types::{BlockId, KvLayout, ModelShape, SeqId};
use turbine_kernels::{KernelError, KernelProvider, KernelRegistry, OpConfig, OpRequirement};
use turbine_tensor::tensor::contiguous_strides;
use turbine_tensor::{DeviceMemory, KvPoolView, Tensor, TensorView};

use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::loader::LoadedWeights;

pub mod batch;
pub mod decoder;
pub mod graphs;
pub mod logits;
pub mod profile;
pub mod rope;

pub use batch::SequenceKv;
pub use decoder::{
    AttentionHook, DecoderDims, DecoderExecutor, DecoderSpec, FfnHook, HookBuffers, HookWeights,
    LayerRun, TpDims, TraceTensor,
};
pub use graphs::{DecodeGraphs, GraphBackend, GraphCache, GraphCounters, GraphKey, GraphStep};
pub use profile::{OpProfile, OpProfileEntry};

/// How an executor runs its forward pass (Phase 2c). `Default` is what the server runs with
/// `execution.fused_ops: true` ([`ExecutorOptions::from_fused_ops`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutorOptions {
    /// Each residual add a norm follows runs as one `add_rmsnorm` (kernel ABI v2.1) when a
    /// provider has it; `false` runs `add` then `rmsnorm` (the Phase 2 op sequence). On by
    /// default.
    pub fused_ops: bool,
    /// One GEMM for the Q/K/V projections and (Llama) one for gate/up, over the fused weights the
    /// loader lays out; `false` runs one GEMM per projection over row views of the same weights
    /// (the Phase 2 op sequence, kept as the reference path).
    ///
    /// On by default ([`FUSED_PROJECTIONS_DEFAULT`]).
    pub fused_projections: bool,
}

/// Whether projection fusion is on by default. On: on the R9700 the fused GEMMs are faster
/// alone (Llama-3.2-3B, 16 rows: Q/K/V 56 µs against 86 µs for the three GEMMs; gate/up 173
/// against 171 µs; RoPE, attention and SiLU·up as fast on the strided views) and save about
/// 0.7 ms per decode step in a server-like executor loop (`perf serving_mix`). An earlier
/// end-to-end run measured 556.7 tok/s fused against 623.6 unfused (2026-09-26), unexplained;
/// if the baseline workload regresses again, set this back to `false`. There is no
/// configuration key for it (spec S-14 lists the switches); `execution.fused_ops: false` keeps
/// it off regardless.
pub const FUSED_PROJECTIONS_DEFAULT: bool = true;

impl Default for ExecutorOptions {
    fn default() -> ExecutorOptions {
        ExecutorOptions {
            fused_ops: true,
            fused_projections: FUSED_PROJECTIONS_DEFAULT,
        }
    }
}

impl ExecutorOptions {
    /// The options `execution.fused_ops` selects: `true` the defaults, `false` every fusion off
    /// (the Phase 2 op sequence).
    pub fn from_fused_ops(fused_ops: bool) -> ExecutorOptions {
        if fused_ops {
            ExecutorOptions::default()
        } else {
            ExecutorOptions {
                fused_ops: false,
                fused_projections: false,
            }
        }
    }
}

/// The batch bounds an executor is built for: KV blocks of `block_tokens` tokens, ragged
/// batches of up to `max_batch_tokens` tokens and `max_seqs` sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutorLimits {
    pub block_tokens: u32,
    pub max_batch_tokens: u32,
    pub max_seqs: u32,
}

/// The op requirements of `cfg`'s family over KV blocks of `block_tokens` tokens run with
/// `opts` ([`crate::families::ModelFamily::requirements`]): the registry the executor runs on
/// is built from this list.
pub fn requirements(
    cfg: &ModelArchConfig,
    block_tokens: u32,
    opts: ExecutorOptions,
) -> Vec<OpRequirement> {
    cfg.family.0.requirements(cfg, block_tokens, opts)
}

/// [`requirements`] without the optional ops (kernel ABI v2.1 `add_rmsnorm`) that none of
/// `providers` supports: the executor then runs their ABI v2 equivalent (`add`, then `rmsnorm`),
/// so a kernel library without them (minor 0, e.g. the Phase 2b CUDA shim) still serves the model.
pub fn available_requirements(
    cfg: &ModelArchConfig,
    block_tokens: u32,
    opts: ExecutorOptions,
    providers: &[Arc<dyn KernelProvider>],
) -> Vec<OpRequirement> {
    requirements(cfg, block_tokens, opts)
        .into_iter()
        .filter(|r| {
            !matches!(r.spec, OpConfig::AddRmsnorm(_))
                || providers.iter().any(|p| r.spec.supported_by(p.as_ref()))
        })
        .collect()
}

/// Where the outputs of projections of one input live in their shared buffer `[rows, Σ cols]`:
/// side by side in each row (`fused`: one GEMM writes them all; each part is a row-strided view)
/// or as consecutive dense `[rows, cols_i]` matrices (one GEMM per projection). Either way the
/// buffer holds the same bytes, so the workspace does not depend on the choice.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Split<const N: usize> {
    pub cols: [usize; N],
    pub fused: bool,
}

impl<const N: usize> Split<N> {
    pub fn width(&self) -> usize {
        self.cols.iter().sum()
    }

    /// Part `i` of the first `t` rows of `buf` (a `[rows, width]` buffer), as `[t, inner…]` with
    /// `inner` multiplying to `cols[i]`.
    pub fn part<'a>(&self, buf: &'a Tensor, i: usize, t: usize, inner: &[usize]) -> TensorView<'a> {
        debug_assert_eq!(inner.iter().product::<usize>(), self.cols[i]);
        let before: usize = self.cols[..i].iter().sum();
        let (offset, row_stride) = if self.fused {
            (before, self.width())
        } else {
            (buf.shape[0] * before, self.cols[i])
        };
        strided_rows(buf, offset, row_stride, t, inner)
    }

    /// The first `t` rows of the whole fused buffer as `[t, width]` (the fused GEMM's output).
    pub fn whole<'a>(&self, buf: &'a Tensor, t: usize) -> TensorView<'a> {
        debug_assert!(self.fused);
        strided_rows(buf, 0, self.width(), t, &[self.width()])
    }
}

/// `t` rows of `[inner…]` elements of `buf`'s storage, the first at element `offset` and each
/// `row_stride` elements after the previous one.
fn strided_rows<'a>(
    buf: &'a Tensor,
    offset: usize,
    row_stride: usize,
    t: usize,
    inner: &[usize],
) -> TensorView<'a> {
    let es = buf.dtype.size_bytes();
    let row: usize = inner.iter().product();
    let len = if t == 0 {
        0
    } else {
        (t - 1) * row_stride + row
    };
    let shape: Vec<usize> = std::iter::once(t).chain(inner.iter().copied()).collect();
    let mut strides = contiguous_strides(&shape);
    strides[0] = row_stride;
    TensorView {
        slice: buf.storage.whole().sub(offset * es, len * es),
        shape: shape.as_slice().into(),
        strides,
        dtype: buf.dtype,
    }
}

/// Device bytes of the executor's buffers for `cfg`'s family (the budget's workspace term):
/// [`crate::families::ModelFamily::workspace_bytes`].
pub fn workspace_bytes(
    cfg: &ModelArchConfig,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
) -> u64 {
    cfg.family.0.workspace_bytes(
        cfg,
        ExecutorLimits {
            block_tokens,
            max_batch_tokens,
            max_seqs,
        },
    )
}

/// The executor of `cfg`'s family ([`crate::families::ModelFamily::build_executor`]) over
/// `weights`, for ragged batches of up to
/// `max_batch_tokens` tokens and `max_seqs` sequences whose KV pool is laid out as
/// `cfg.kv_layout(block_tokens)`, run with `opts`. `registry` must have been built from
/// [`requirements`] of `cfg`, `block_tokens` and `opts`.
#[allow(clippy::too_many_arguments)]
pub fn build_executor(
    cfg: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
    opts: ExecutorOptions,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    let limits = ExecutorLimits {
        block_tokens,
        max_batch_tokens,
        max_seqs,
    };
    cfg.family
        .0
        .build_executor(cfg, weights, registry, mem, limits, opts)
}

/// One sequence of a ragged batch: its new tokens are `tokens[q_start..q_start + q_len]` of the
/// [`BatchInput`], at positions `kv_len − q_len .. kv_len`.
#[derive(Clone, Copy, Debug)]
pub struct SeqSlice<'a> {
    pub seq: SeqId,
    /// Offset of the sequence's first new token in `BatchInput::tokens`.
    pub q_start: u32,
    /// New tokens this step (a prefill chunk, or 1 for a decode); at least 1.
    pub q_len: u32,
    /// Tokens holding K/V after this step's append (cached prefix + `q_len`).
    pub kv_len: u32,
    /// The pool blocks holding tokens `0..kv_len` in token order: token `p` lives in
    /// `block_table[p / block_tokens]` at slot `p % block_tokens`. May be longer than needed.
    pub block_table: &'a [BlockId],
    /// Reduce this sequence's logits row on the device (`logits_reduce`, P2c S-4) instead of
    /// copying it whole; ignored when the executor does not reduce
    /// ([`ModelExecutor::reduces_logits`]).
    pub reduce: Option<RowReduce>,
}

/// What the device reduction of one logits row computes (P2c S-4): the `top_n` largest raw
/// logits with their ids and the row's log-sum-exp, and, when `uniform` is set, one
/// categorical draw at `temperature`: by inverse CDF in id order when `top_p` is 1, else over
/// the `top_p` nucleus in descending logit order (the host sampler's top-p draw).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowReduce {
    /// Candidates to return, 1..=[`logits::MAX_TOP_N`].
    pub top_n: u8,
    pub temperature: f32,
    /// The draw's uniform in `[0, 1)`; `None` = no draw on the device.
    pub uniform: Option<f32>,
    /// The draw's nucleus mass in `(0, 1]` (1 = the whole vocabulary); unused without a draw.
    pub top_p: f32,
}

/// One logits row reduced on the device: its raw log-sum-exp, its `top_n` largest raw logits
/// (descending, ties to the lower id, NaN last) and the categorical draw `(id, raw logit)` when
/// one was asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedRow {
    pub lse: f32,
    pub top: Vec<(u32, f32)>,
    pub sampled: Option<(u32, f32)>,
}

/// One token of a launched batch taken on the device from the previous launch (P2c overlap
/// scheduling): batch token `token` becomes the token the device chose — its categorical draw,
/// or its best candidate for a greedy row — for sequence `prev_slot` of the previous
/// [`ModelExecutor::launch`], without a host round trip. That row must have been reduced with a
/// draw or greedily (`RowReduce::uniform` set, or `temperature` ≤ 0). The batch's own id at
/// `token` is ignored (any in-vocabulary placeholder).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenFeed {
    pub token: u32,
    pub prev_slot: u32,
}

/// Groups `feeds` into runs `(first batch token, first word, length)` over consecutive batch
/// tokens whose device words (`word(prev_slot)`) are consecutive, after checking that every
/// token is inside a batch of `tokens` tokens and fed once and that every slot has a device
/// choice.
pub(crate) fn feed_runs(
    feeds: &[TokenFeed],
    tokens: usize,
    word: impl Fn(usize) -> Option<usize>,
) -> Result<Vec<(usize, usize, usize)>, ModelError> {
    let invalid = |message: String| ModelError::Kernel(KernelError::InvalidArgument { message });
    let mut resolved: Vec<(usize, usize)> = Vec::with_capacity(feeds.len());
    for f in feeds {
        let token = f.token as usize;
        if token >= tokens {
            return Err(invalid(format!(
                "token feed into token {token} of a {tokens}-token batch"
            )));
        }
        let w = word(f.prev_slot as usize).ok_or_else(|| {
            invalid(format!(
                "token feed from sequence {} of the previous launch, which has no device-chosen \
                 token",
                f.prev_slot
            ))
        })?;
        resolved.push((token, w));
    }
    resolved.sort_unstable();
    if let Some(w) = resolved.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(invalid(format!("token {} is fed twice", w[0].0)));
    }
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (token, w) in resolved {
        match runs.last_mut() {
            Some((t0, w0, len)) if *t0 + *len == token && *w0 + *len == w => *len += 1,
            _ => runs.push((token, w, 1)),
        }
    }
    Ok(runs)
}

/// One forward step over a ragged batch: `tokens[i]` sits at absolute position `positions[i]`,
/// `seqs` tile `tokens` in order (sequence `s + 1` starts where `s` ends), and every sequence's
/// K/V lives in `kv`.
#[derive(Clone, Copy, Debug)]
pub struct BatchInput<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
    pub seqs: &'a [SeqSlice<'a>],
    pub kv: &'a KvPoolView<'a>,
}

/// The logits of one forward, one slot per sequence (its last position) in `seqs` order: a full
/// FP32 row of `data` (`rows × vocab`, row-major, in sequence order) or, for a sequence whose
/// row was reduced on the device, an entry of `reduced` (in sequence order).
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    /// Full rows in `data`.
    pub rows: usize,
    pub vocab: usize,
    pub data: Vec<f32>,
    pub reduced: Vec<ReducedRow>,
    /// Per sequence: its full row (`Full(i)`: row `i` of `data`) or reduced row.
    slots: Vec<SlotIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotIndex {
    Full(usize),
    Reduced(usize),
}

/// One sequence's logits: its full row or its device reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LogitsSlot<'a> {
    Full(&'a [f32]),
    Reduced(&'a ReducedRow),
}

impl Logits {
    /// `rows` full rows, one per sequence.
    pub fn full(rows: usize, vocab: usize, data: Vec<f32>) -> Logits {
        assert_eq!(data.len(), rows * vocab, "{rows} logits rows of {vocab}");
        Logits {
            rows,
            vocab,
            data,
            reduced: Vec::new(),
            slots: (0..rows).map(SlotIndex::Full).collect(),
        }
    }

    /// Full rows `data` and `reduced` rows interleaved as `is_reduced` says, one flag per
    /// sequence: the full rows belong to the sequences with `false` in order, the reduced rows
    /// to those with `true`.
    pub fn mixed(
        vocab: usize,
        data: Vec<f32>,
        reduced: Vec<ReducedRow>,
        is_reduced: impl IntoIterator<Item = bool>,
    ) -> Logits {
        let (mut full, mut red) = (0, 0);
        let slots: Vec<SlotIndex> = is_reduced
            .into_iter()
            .map(|r| {
                if r {
                    red += 1;
                    SlotIndex::Reduced(red - 1)
                } else {
                    full += 1;
                    SlotIndex::Full(full - 1)
                }
            })
            .collect();
        assert_eq!(
            data.len(),
            full * vocab,
            "{full} full logits rows of {vocab}"
        );
        assert_eq!(reduced.len(), red, "{red} reduced rows");
        Logits {
            rows: full,
            vocab,
            data,
            reduced,
            slots,
        }
    }

    /// Full row `r` of `data`; panics when `r >= rows`.
    pub fn row(&self, r: usize) -> &[f32] {
        assert!(r < self.rows, "logits row {r} of {}", self.rows);
        &self.data[r * self.vocab..(r + 1) * self.vocab]
    }

    /// Sequences covered (full plus reduced rows).
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Sequence `i`'s logits; panics when `i >= slots()`.
    pub fn slot(&self, i: usize) -> LogitsSlot<'_> {
        match self.slots[i] {
            SlotIndex::Full(r) => LogitsSlot::Full(self.row(r)),
            SlotIndex::Reduced(r) => LogitsSlot::Reduced(&self.reduced[r]),
        }
    }

    /// The row of `data` holding sequence `i`'s full logits; `None` when it was reduced.
    pub fn full_row_index(&self, i: usize) -> Option<usize> {
        match self.slots[i] {
            SlotIndex::Full(r) => Some(r),
            SlotIndex::Reduced(_) => None,
        }
    }
}

/// Where the host time of the last [`ModelExecutor::forward`] went (P2c S-1); the rest of the
/// call (packing and uploading the batch metadata) is the engine's `prepare` stage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ForwardTimings {
    /// From the uploaded batch metadata until the last kernel of the pass is enqueued (the
    /// op calls' host time; OLMoE's per-layer routing reads are included).
    pub launch: Duration,
    /// The synchronising device-to-host logits copy and its conversion into [`Logits`].
    pub device_wait: Duration,
}

/// A model's forward pass over the kernel registry.
pub trait ModelExecutor: Send {
    fn shape(&self) -> &ModelShape;
    /// The KV layout the executor reads and writes; the pool of every batch must match it.
    fn kv_layout(&self) -> &KvLayout;
    /// Runs one step: appends every sequence's new K/V into its pool blocks and returns one
    /// logits slot per sequence (its last position), in `seqs` order — the full FP32 row, or
    /// its device reduction when the slice asks for one and the executor reduces — with one
    /// device-to-host copy.
    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError>;
    /// Timings of the last successful `forward` (or `collect`); zeros when the executor does not
    /// measure.
    fn last_timings(&self) -> ForwardTimings {
        ForwardTimings::default()
    }
    /// True when [`ModelExecutor::launch`] can enqueue a step while the previous one is still
    /// uncollected, with [`TokenFeed`]s from it (P2c overlap scheduling: the kernel library has
    /// host staging, ABI v2.3).
    fn overlaps(&self) -> bool {
        false
    }
    /// Enqueues one step like [`ModelExecutor::forward`] without waiting for it; its logits come
    /// from the matching [`ModelExecutor::collect`], in launch order. `feeds` take tokens the
    /// previous launch chose on the device. At most two launches are uncollected when the
    /// executor [`overlaps`](ModelExecutor::overlaps), one otherwise. Unsupported by default.
    fn launch(&mut self, batch: &BatchInput<'_>, feeds: &[TokenFeed]) -> Result<(), ModelError> {
        let _ = (batch, feeds);
        Err(ModelError::Kernel(KernelError::Unsupported {
            message: "this executor does not launch steps ahead".into(),
        }))
    }
    /// The logits of the oldest uncollected [`ModelExecutor::launch`]; waits for that step
    /// only. Unsupported by default.
    fn collect(&mut self) -> Result<Logits, ModelError> {
        Err(ModelError::Kernel(KernelError::Unsupported {
            message: "this executor does not launch steps ahead".into(),
        }))
    }
    /// Decode graphs (P2c S-10, [`graphs`]): with `Some`, decode-only iterations are captured
    /// into graphs and replayed through them; `None` (the default) launches every op. An
    /// executor that cannot run graphs ignores the call.
    fn set_decode_graphs(&mut self, graphs: Option<DecodeGraphs>) {
        let _ = graphs;
    }
    /// Tensor-parallel prefill overlap (P5 Task 32, `parallel.tp_prefill_overlap`): prefills of
    /// at least `min_tokens` rows (`None`: off; the server passes
    /// [`decoder::PREFILL_OVERLAP_MIN_TOKENS`]) split in two halves whose all-reduces overlap
    /// the other half's work, with the same results. Returns whether it is on; executors
    /// without it ignore the call.
    fn set_prefill_overlap(&mut self, min_tokens: Option<usize>) -> bool {
        let _ = min_tokens;
        false
    }
    /// Decode graph outcomes since graphs were set; zeros without them.
    fn graph_counters(&self) -> GraphCounters {
        GraphCounters::default()
    }
    /// True when `forward` honours [`SeqSlice::reduce`] (a `logits_reduce` provider was
    /// selected at startup); otherwise every sequence gets its full row.
    fn reduces_logits(&self) -> bool {
        false
    }
    /// Tensor or expert parallelism (P5 Task 28): replaces the rank's communicator with `c`, a
    /// fresh one of the same group and rank (the old one failed and was aborted). Unsupported by
    /// an executor without one.
    fn set_collective(
        &mut self,
        c: Arc<dyn turbine_distributed::collective::Collective>,
    ) -> Result<(), ModelError> {
        let _ = c;
        Err(ModelError::Kernel(KernelError::Unsupported {
            message: "this executor has no communicator".into(),
        }))
    }
    /// Copies block `src[i]` to `dst[i]` in every layer of `kv` (the `n > 1` fork), ordered
    /// before the next forward on the same stream.
    fn copy_blocks(
        &mut self,
        kv: &KvPoolView<'_>,
        src: &[BlockId],
        dst: &[BlockId],
    ) -> Result<(), ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `execution.fused_ops: true` runs the defaults (projection fusion only when
    /// [`FUSED_PROJECTIONS_DEFAULT`]); `false` turns every fusion off.
    /// Feeds group into runs over consecutive tokens and words; a token outside the batch, a
    /// token fed twice and a slot without a device choice are refused. Breaks if a run joins
    /// non-adjacent words (the embedding would read the wrong ids).
    #[test]
    fn feed_runs_group_and_validate() {
        let feed = |token, prev_slot| TokenFeed { token, prev_slot };
        // Slot s's choice sits at word 100 + s, except slot 3 (a host draw).
        let word = |s: usize| (s != 3).then_some(100 + s);
        let runs = feed_runs(
            &[
                feed(2, 1),
                feed(0, 0),
                feed(1, 1 + 1),
                feed(3, 5),
                feed(4, 6),
            ],
            8,
            word,
        )
        .expect("runs");
        // Tokens 0..3 read words 100, 102, 101: 0 and 1 are adjacent, 2 starts a new run.
        assert_eq!(
            runs,
            vec![(0, 100, 1), (1, 102, 1), (2, 101, 1), (3, 105, 2)]
        );
        let consecutive = feed_runs(&[feed(1, 0), feed(2, 1), feed(3, 2)], 4, word).expect("one");
        assert_eq!(consecutive, vec![(1, 100, 3)]);
        assert!(feed_runs(&[], 1, word).expect("none").is_empty());
        for (feeds, what) in [
            (vec![feed(8, 0)], "token 8 of a 8-token"),
            (vec![feed(1, 0), feed(1, 1)], "fed twice"),
            (vec![feed(0, 3)], "no device-chosen"),
        ] {
            let err = feed_runs(&feeds, 8, word).expect_err(what).to_string();
            assert!(err.contains(what), "{err}");
        }
    }

    #[test]
    fn fused_ops_selects_defaults_or_nothing() {
        let on = ExecutorOptions::from_fused_ops(true);
        assert_eq!(on, ExecutorOptions::default());
        assert!(on.fused_ops);
        assert_eq!(on.fused_projections, FUSED_PROJECTIONS_DEFAULT);
        let off = ExecutorOptions::from_fused_ops(false);
        assert!(!off.fused_ops && !off.fused_projections);
    }
}
