//! Model executors: the forward pass over kernel-registry ops (TS §6). Phase 1: Llama, one
//! sequence, a contiguous per-request KV cache.
use turbine_core::types::{KvLayout, ModelShape};

use crate::ModelError;

pub mod llama;
pub mod rope;

pub use llama::LlamaExecutor;

/// One forward step of a single sequence (Phase 1): `tokens[i]` sits at absolute position
/// `positions[i]`. Positions are consecutive and start at or before the cached length; starting
/// at 0 begins a new sequence and overwrites the cache.
#[derive(Clone, Copy, Debug)]
pub struct BatchInput<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
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
    fn kv_layout(&self) -> &KvLayout;
    /// Runs one step: writes the step's K/V into the cache and returns the last position's
    /// FP32 logits (one device-to-host copy).
    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError>;
}
