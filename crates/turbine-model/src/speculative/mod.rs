//! Phase 8 S-2: the speculative `Proposer` extension point. A proposer owns a small draft
//! model (its executor, its private one-sequence KV pool and its slot) and returns greedy
//! proposals the target verifies in one forward (`turbine_scheduler::speculative`). `draft_model`
//! is the only entry this track ships; `speculative.method: draft` selects it.
//!
//! The registry is compiled in and covered by `conformance`, which the integration test
//! `crates/turbine-model/tests/conformance.rs` runs over the real registries; the page is
//! `docs/extending/speculative-proposer.md`.
use std::sync::Arc;

use turbine_core::types::SeqId;
use turbine_kv::pool::{BlockPool, PoolError};

use crate::ModelError;
use crate::Tokenizer;
use crate::config::ModelArchConfig;
use crate::executor::ModelExecutor;

pub mod draft;
pub mod k_policy;
pub mod mtp;

/// What `Proposer::propose` asks for: the sequence, the last accepted token, its position and k.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProposeRequest {
    pub seq: SeqId,
    pub last_token: u32,
    pub accepted_len: u32,
    pub num_tokens: u32,
}

/// The greedy proposals, in order; at most `num_tokens`, fewer only at the draft's context end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposals {
    pub tokens: Vec<u32>,
}

/// A proposer's failure; `reason` is the bounded code in logs and fallbacks.
#[derive(Debug, thiserror::Error)]
pub enum ProposeError {
    #[error("the draft slot is held by another sequence")]
    SlotBusy,
    #[error("the draft sequence has not begun")]
    NotBegun,
    #[error("draft KV: {0}")]
    Kv(#[from] PoolError),
    #[error("draft model: {0}")]
    Model(#[from] ModelError),
    #[error("draft unsupported: {0}")]
    Unsupported(&'static str),
}

impl ProposeError {
    /// The reason code (`draft_slot_busy`, `draft_not_begun`, `draft_kv`, `draft_model`,
    /// `draft_unsupported`).
    pub fn reason(&self) -> &'static str {
        match self {
            ProposeError::SlotBusy => "draft_slot_busy",
            ProposeError::NotBegun => "draft_not_begun",
            ProposeError::Kv(_) => "draft_kv",
            ProposeError::Model(_) => "draft_model",
            ProposeError::Unsupported(_) => "draft_unsupported",
        }
    }
}

/// Phase 8 S-2: a draft proposer. One slot per proposer: `begin` takes it, `release` frees it.
pub trait Proposer: Send {
    fn name(&self) -> &'static str;

    /// Validate the target/draft pair before serving; errors are startup refusals.
    fn validate(
        &self,
        target: &ModelArchConfig,
        draft: &ModelArchConfig,
        tokenizer: &Tokenizer,
    ) -> Result<(), ProposeError>;

    /// Take the one full-context slot for `seq` and prefill `prompt`.
    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<(), ProposeError>;

    /// Propose greedy tokens from `last_token` at `accepted_len`; the slot must hold `seq`.
    fn propose(&mut self, request: ProposeRequest) -> Result<Proposals, ProposeError>;

    /// Drop the draft KV past `len` (the target's accepted length after a partial accept).
    fn truncate(&mut self, seq: SeqId, len: u32) -> Result<(), ProposeError>;

    /// Free the slot (finish, cancel, preempt or permanent disable).
    fn release(&mut self, seq: SeqId);

    /// P8 MTP: append the MTP head's KV rows for `positions` (a run of positions `1..` that
    /// share a sequence), each built from the target's final hidden row at `position − 1`
    /// (`hidden_prev`, `positions.len() × hidden` F32) and the embedding of `tokens` (the token
    /// ids at `positions`). The engine calls it after each target forward that advances the
    /// accepted prefix; the `draft_model` proposer ignores it.
    fn observe(
        &mut self,
        seq: SeqId,
        positions: &[u32],
        tokens: &[u32],
        hidden_prev: turbine_tensor::TensorView<'_>,
    ) -> Result<(), ProposeError> {
        let _ = (seq, positions, tokens, hidden_prev);
        Ok(())
    }

    /// P8 MTP: the target's final hidden row at the last accepted position (`hidden` F32), the
    /// next propose round's first query input. The `draft_model` proposer ignores it.
    fn seed_last_hidden(
        &mut self,
        seq: SeqId,
        hidden: turbine_tensor::TensorView<'_>,
    ) -> Result<(), ProposeError> {
        let _ = (seq, hidden);
        Ok(())
    }
}

/// What `ProposerFactory::build` needs: the draft's prepared executor and pool, the target and
/// draft configs, the block geometry and the bounds the executor was sized for.
pub struct DraftProposerConfig {
    pub executor: Box<dyn ModelExecutor>,
    pub pool: BlockPool,
    pub tokenizer: Arc<Tokenizer>,
    pub block_tokens: u32,
    pub max_seq_len: u32,
    pub max_batch_tokens: u32,
    pub target: ModelArchConfig,
    pub draft: ModelArchConfig,
}

/// A compiled-in proposer implementation.
pub trait ProposerFactory: Send + Sync {
    fn name(&self) -> &'static str;
    fn build(&self, cfg: DraftProposerConfig) -> Result<Box<dyn Proposer>, ProposeError>;
}

/// The `draft_model` factory (the one registered entry).
pub struct DraftModelFactory;

impl ProposerFactory for DraftModelFactory {
    fn name(&self) -> &'static str {
        "draft_model"
    }

    fn build(&self, cfg: DraftProposerConfig) -> Result<Box<dyn Proposer>, ProposeError> {
        Ok(Box::new(draft::DraftProposer::new(cfg)?))
    }
}

/// Every proposer implementation this build ships.
pub static PROPOSERS: &[&dyn ProposerFactory] = &[&DraftModelFactory];

/// The registered factory named `name`.
pub fn proposer(name: &str) -> Option<&'static dyn ProposerFactory> {
    PROPOSERS.iter().copied().find(|f| f.name() == name)
}

/// The extension point's conformance: build the factory, validate the pair, take the slot,
/// propose, truncate, release and take the slot again. Returns the first broken contract.
pub fn conformance(factory: &dyn ProposerFactory, cfg: DraftProposerConfig) -> Result<(), String> {
    if factory.name().is_empty() {
        return Err("empty proposer name".into());
    }
    let mut proposer = factory
        .build(cfg)
        .map_err(|e| format!("build: {}", e.reason()))?;
    let seq = SeqId(1);
    let prompt: Vec<u32> = (0..4).collect();
    let error = proposer.propose(ProposeRequest {
        seq,
        last_token: 0,
        accepted_len: 4,
        num_tokens: 2,
    });
    if !matches!(error, Err(ProposeError::NotBegun)) {
        return Err("propose before begin must fail with NotBegun".into());
    }
    proposer
        .begin(seq, &prompt)
        .map_err(|e| format!("begin: {}", e.reason()))?;
    let proposals = proposer
        .propose(ProposeRequest {
            seq,
            last_token: 0,
            accepted_len: 4,
            num_tokens: 4,
        })
        .map_err(|e| format!("propose: {}", e.reason()))?;
    if proposals.tokens.len() != 4 {
        return Err(format!(
            "proposed {} tokens, expected 4",
            proposals.tokens.len()
        ));
    }
    if !matches!(
        proposer.begin(SeqId(2), &prompt),
        Err(ProposeError::SlotBusy)
    ) {
        return Err("a second begin on the held slot must fail with SlotBusy".into());
    }
    proposer
        .truncate(seq, 4)
        .map_err(|e| format!("truncate: {}", e.reason()))?;
    proposer.release(seq);
    proposer
        .begin(SeqId(3), &prompt)
        .map_err(|e| format!("begin after release: {}", e.reason()))?;
    proposer.release(SeqId(3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use turbine_core::types::{BlockId, DeviceId, KvLayout, ModelShape};
    use turbine_kv::pool::BlockPoolConfig;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use turbine_tensor::KvPoolView;

    use crate::executor::{BatchInput, Logits, ModelExecutor};
    use crate::testing::TempDir;
    use crate::testing::tiny::write_tiny_llama;

    /// A deterministic executor: the row's argmax is `(last position + 1) % vocab`, so the
    /// proposals are predictable without weights.
    struct MockExecutor {
        shape: ModelShape,
        layout: KvLayout,
        vocab: usize,
    }

    impl ModelExecutor for MockExecutor {
        fn shape(&self) -> &ModelShape {
            &self.shape
        }

        fn kv_layout(&self) -> &KvLayout {
            &self.layout
        }

        fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            let last = *batch.positions.last().unwrap_or(&0) as usize;
            let mut row = vec![0.0f32; self.vocab];
            row[(last + 1) % self.vocab] = 1.0;
            Ok(Logits::full(1, self.vocab, row))
        }

        fn copy_blocks(
            &mut self,
            _kv: &KvPoolView<'_>,
            _src: &[BlockId],
            _dst: &[BlockId],
        ) -> Result<(), ModelError> {
            Ok(())
        }
    }

    fn config(tmp: &TempDir) -> DraftProposerConfig {
        let spec = write_tiny_llama(tmp.path(), 3);
        let block_tokens = 4;
        let layout = spec.config.kv_layout(block_tokens);
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout,
                num_blocks: 8,
            },
            mem,
        )
        .unwrap();
        let vocab = spec.config.shape().vocab as usize;
        let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
        DraftProposerConfig {
            executor: Box::new(MockExecutor {
                shape: spec.config.shape(),
                layout,
                vocab,
            }),
            pool,
            tokenizer,
            block_tokens,
            max_seq_len: 64,
            max_batch_tokens: 8,
            target: spec.config.clone(),
            draft: spec.config,
        }
    }

    #[test]
    fn draft_model_conformance() {
        let tmp = TempDir::new("turbine-proposer-conformance");
        conformance(&DraftModelFactory, config(&tmp)).expect("the draft_model entry conforms");
    }

    #[test]
    fn proposals_are_greedy_and_truncate_keeps_the_prefix() {
        let tmp = TempDir::new("turbine-proposer-greedy");
        let mut proposer = DraftModelFactory.build(config(&tmp)).unwrap();
        let seq = SeqId(1);
        proposer.begin(seq, &[0, 1, 2, 3]).unwrap();
        let first = proposer
            .propose(ProposeRequest {
                seq,
                last_token: 5,
                accepted_len: 4,
                num_tokens: 4,
            })
            .unwrap();
        assert_eq!(first.tokens, [5, 6, 7, 8]);
        proposer.truncate(seq, 6).unwrap();
        let second = proposer
            .propose(ProposeRequest {
                seq,
                last_token: 7,
                accepted_len: 6,
                num_tokens: 2,
            })
            .unwrap();
        assert_eq!(second.tokens, [7, 8]);
        assert!(!matches!(proposer.truncate(SeqId(2), 0), Ok(())));
        proposer.release(seq);
        assert!(matches!(
            proposer.propose(ProposeRequest {
                seq,
                last_token: 0,
                accepted_len: 0,
                num_tokens: 1,
            }),
            Err(ProposeError::NotBegun)
        ));
    }

    #[test]
    fn slot_is_not_shared() {
        let tmp = TempDir::new("turbine-proposer-slot");
        let mut proposer = DraftModelFactory.build(config(&tmp)).unwrap();
        proposer.begin(SeqId(1), &[0, 1, 2, 3]).unwrap();
        assert!(matches!(
            proposer.begin(SeqId(2), &[0]),
            Err(ProposeError::SlotBusy)
        ));
        proposer.release(SeqId(1));
        proposer.begin(SeqId(2), &[0]).unwrap();
    }
}
