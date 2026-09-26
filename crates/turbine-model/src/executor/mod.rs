//! Model executors: the forward pass over kernel-registry ops (TS §6). Phase 2: ragged batches
//! of sequences whose KV lives in the paged L0 block pool (`KvPoolView`), one FP32 logits row
//! per sequence and one device-to-host copy per iteration (P2 S-5, S-9).
use turbine_core::types::{BlockId, KvLayout, ModelShape, SeqId};
use turbine_tensor::KvPoolView;

use crate::ModelError;

pub mod batch;
pub mod llama;
pub mod rope;

pub use batch::SequenceKv;
pub use llama::{LlamaExecutor, TraceTensor};

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

/// FP32 logits, `rows × vocab`, row-major: one row per sequence (its last position).
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    pub rows: usize,
    pub vocab: usize,
    pub data: Vec<f32>,
}

impl Logits {
    /// Row `r`; panics when `r >= rows`.
    pub fn row(&self, r: usize) -> &[f32] {
        assert!(r < self.rows, "logits row {r} of {}", self.rows);
        &self.data[r * self.vocab..(r + 1) * self.vocab]
    }
}

/// A model's forward pass over the kernel registry.
pub trait ModelExecutor: Send {
    fn shape(&self) -> &ModelShape;
    /// The KV layout the executor reads and writes; the pool of every batch must match it.
    fn kv_layout(&self) -> &KvLayout;
    /// Runs one step: appends every sequence's new K/V into its pool blocks and returns one
    /// FP32 logits row per sequence (its last position), in `seqs` order, with one
    /// device-to-host copy.
    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError>;
    /// Copies block `src[i]` to `dst[i]` in every layer of `kv` (the `n > 1` fork), ordered
    /// before the next forward on the same stream.
    fn copy_blocks(
        &mut self,
        kv: &KvPoolView<'_>,
        src: &[BlockId],
        dst: &[BlockId],
    ) -> Result<(), ModelError>;
}
