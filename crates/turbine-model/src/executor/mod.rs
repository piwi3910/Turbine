//! Model executors: the forward pass over kernel-registry ops (TS §6). Phase 2: ragged batches
//! of sequences whose KV lives in the paged L0 block pool (`KvPoolView`), one FP32 logits row
//! per sequence and one device-to-host copy per iteration (P2 S-5, S-9).
use std::sync::Arc;

use turbine_core::types::{BlockId, KvLayout, ModelShape, SeqId};
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_tensor::{DeviceMemory, KvPoolView};

use crate::ModelError;
use crate::config::{Architecture, ModelArchConfig};
use crate::loader::LoadedWeights;

pub mod batch;
pub mod llama;
pub mod olmoe;
pub mod rope;

pub use batch::SequenceKv;
pub use llama::{LlamaExecutor, TraceTensor};
pub use olmoe::OlmoeExecutor;

/// The op requirements of `cfg`'s architecture over KV blocks of `block_tokens` tokens
/// ([`LlamaExecutor::requirements`] or [`OlmoeExecutor::requirements`]): the registry the
/// executor runs on is built from this list.
pub fn requirements(cfg: &ModelArchConfig, block_tokens: u32) -> Vec<OpRequirement> {
    match cfg.architecture {
        Architecture::Llama => LlamaExecutor::requirements(cfg, block_tokens),
        Architecture::Olmoe => OlmoeExecutor::requirements(cfg, block_tokens),
    }
}

/// Device bytes of the executor's buffers for `cfg`'s architecture (the budget's workspace
/// term): [`LlamaExecutor::workspace_bytes`] or [`OlmoeExecutor::workspace_bytes`].
pub fn workspace_bytes(
    cfg: &ModelArchConfig,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
) -> u64 {
    match cfg.architecture {
        Architecture::Llama => {
            LlamaExecutor::workspace_bytes(cfg, block_tokens, max_batch_tokens, max_seqs)
        }
        Architecture::Olmoe => {
            OlmoeExecutor::workspace_bytes(cfg, block_tokens, max_batch_tokens, max_seqs)
        }
    }
}

/// The executor of `cfg`'s architecture over `weights`, for ragged batches of up to
/// `max_batch_tokens` tokens and `max_seqs` sequences whose KV pool is laid out as
/// `cfg.kv_layout(block_tokens)`. `registry` must have been built from [`requirements`] of
/// `cfg` and `block_tokens`.
pub fn build_executor(
    cfg: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    Ok(match cfg.architecture {
        Architecture::Llama => Box::new(LlamaExecutor::new(
            cfg,
            weights,
            registry,
            mem,
            block_tokens,
            max_batch_tokens,
            max_seqs,
        )?),
        Architecture::Olmoe => Box::new(OlmoeExecutor::new(
            cfg,
            weights,
            registry,
            mem,
            block_tokens,
            max_batch_tokens,
            max_seqs,
        )?),
    })
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
