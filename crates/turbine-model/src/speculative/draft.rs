//! The `draft_model` proposer: one small same-family model, one private one-sequence KV pool,
//! greedy proposals from the draft's own executor. See `super` for the trait and registry.
use std::sync::Arc;

use turbine_core::types::{BlockId, SeqId};
use turbine_kv::pool::BlockPool;
use turbine_kv::{BlockTable, blocks_for_tokens};

use super::{DraftProposerConfig, Proposals, ProposeError, ProposeRequest, Proposer};
use crate::Tokenizer;
use crate::config::ModelArchConfig;
use crate::executor::{BatchInput, LogitsSlot, ModelExecutor, SeqSlice};

/// The single slot a proposer holds.
struct Slot {
    seq: SeqId,
    table: BlockTable,
}

/// Phase 8 S-2: the draft over its executor and private pool.
pub struct DraftProposer {
    executor: Box<dyn ModelExecutor>,
    pool: BlockPool,
    block_tokens: u32,
    max_seq_len: u32,
    max_batch_tokens: u32,
    slot: Option<Slot>,
}

impl DraftProposer {
    /// Build the proposer; the executor's KV layout must be the draft's.
    pub fn new(cfg: DraftProposerConfig) -> Result<DraftProposer, ProposeError> {
        let expected = cfg.draft.kv_layout(cfg.block_tokens);
        if *cfg.executor.kv_layout() != expected {
            return Err(ProposeError::Unsupported(
                "the draft executor's KV layout differs from the draft model's",
            ));
        }
        if cfg.target.hf_architecture != cfg.draft.hf_architecture {
            return Err(ProposeError::Unsupported(
                "the draft architecture differs from the target's",
            ));
        }
        // The tokenizer belongs to the server's pair check; the proposer works on token ids.
        let _: Arc<Tokenizer> = cfg.tokenizer;
        Ok(DraftProposer {
            executor: cfg.executor,
            pool: cfg.pool,
            block_tokens: cfg.block_tokens,
            max_seq_len: cfg.max_seq_len,
            max_batch_tokens: cfg.max_batch_tokens.max(1),
            slot: None,
        })
    }

    /// Prefill `prompt` in `max_batch_tokens` chunks into `table`.
    fn fill(
        &mut self,
        seq: SeqId,
        prompt: &[u32],
        table: &mut BlockTable,
    ) -> Result<(), ProposeError> {
        let chunk = self.max_batch_tokens as usize;
        for (i, tokens) in prompt.chunks(chunk).enumerate() {
            let start = (i * chunk) as u32;
            ensure_room(
                &mut self.pool,
                table,
                start + tokens.len() as u32,
                self.block_tokens,
            )?;
            let _ = run_row(&mut *self.executor, &self.pool, seq, tokens, start, table)?;
        }
        Ok(())
    }
}

impl Proposer for DraftProposer {
    fn name(&self) -> &'static str {
        "draft_model"
    }

    fn validate(
        &self,
        target: &ModelArchConfig,
        draft: &ModelArchConfig,
        _tokenizer: &Tokenizer,
    ) -> Result<(), ProposeError> {
        if target.hf_architecture != draft.hf_architecture {
            return Err(ProposeError::Unsupported(
                "the draft architecture differs from the target's",
            ));
        }
        Ok(())
    }

    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<(), ProposeError> {
        if prompt.is_empty() {
            return Err(ProposeError::Unsupported("empty draft prompt"));
        }
        if prompt.len() as u32 > self.max_seq_len {
            return Err(ProposeError::Unsupported("draft context overflow"));
        }
        if self.slot.as_ref().is_some_and(|slot| slot.seq != seq) {
            return Err(ProposeError::SlotBusy);
        }
        if let Some(old) = self.slot.take() {
            self.pool.release(&old.table.blocks);
        }
        let mut table = BlockTable::new();
        let filled = self.fill(seq, prompt, &mut table);
        if filled.is_err() {
            self.pool.release(&table.blocks);
        }
        filled?;
        self.slot = Some(Slot { seq, table });
        Ok(())
    }

    fn propose(&mut self, request: ProposeRequest) -> Result<Proposals, ProposeError> {
        let DraftProposer {
            executor,
            pool,
            block_tokens,
            max_seq_len,
            slot,
            ..
        } = self;
        let slot = slot
            .as_mut()
            .filter(|slot| slot.seq == request.seq)
            .ok_or(ProposeError::NotBegun)?;
        let available = max_seq_len.saturating_sub(request.accepted_len.saturating_add(1));
        let k = request.num_tokens.min(8).min(available);
        if k == 0 {
            return Ok(Proposals { tokens: Vec::new() });
        }
        truncate_table(pool, &mut slot.table, request.accepted_len, *block_tokens);
        let mut tokens = Vec::with_capacity(k as usize);
        let mut next = request.last_token;
        for i in 0..k {
            let position = request.accepted_len + i;
            ensure_room(pool, &mut slot.table, position + 1, *block_tokens)?;
            let row = run_row(
                &mut **executor,
                pool,
                slot.seq,
                &[next],
                position,
                &slot.table,
            )?;
            let token = argmax(&row);
            tokens.push(token);
            next = token;
        }
        Ok(Proposals { tokens })
    }

    fn truncate(&mut self, seq: SeqId, len: u32) -> Result<(), ProposeError> {
        let DraftProposer {
            pool,
            block_tokens,
            slot,
            ..
        } = self;
        let slot = slot
            .as_mut()
            .filter(|slot| slot.seq == seq)
            .ok_or(ProposeError::NotBegun)?;
        truncate_table(pool, &mut slot.table, len, *block_tokens);
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        if self.slot.as_ref().is_some_and(|slot| slot.seq == seq)
            && let Some(slot) = self.slot.take()
        {
            self.pool.release(&slot.table.blocks);
        }
    }
}

/// Grows `table` to hold `new_tokens` positions.
fn ensure_room(
    pool: &mut BlockPool,
    table: &mut BlockTable,
    new_tokens: u32,
    block_tokens: u32,
) -> Result<(), ProposeError> {
    if new_tokens <= table.tokens {
        return Ok(());
    }
    let needed = table.blocks_needed(new_tokens - table.tokens, block_tokens);
    if needed > 0 {
        let blocks = pool.allocate(needed)?;
        table.blocks.extend(blocks);
    }
    table.tokens = new_tokens;
    Ok(())
}

/// Drops `table` past `len` positions and frees the blocks that become unneeded.
fn truncate_table(pool: &mut BlockPool, table: &mut BlockTable, len: u32, block_tokens: u32) {
    if len >= table.tokens {
        return;
    }
    let keep = blocks_for_tokens(len, block_tokens) as usize;
    let tail: Vec<BlockId> = table.blocks.drain(keep..).collect();
    if !tail.is_empty() {
        pool.release(&tail);
    }
    table.tokens = len;
}

/// One draft forward of `tokens` at `start`, returning the last row.
fn run_row(
    executor: &mut dyn ModelExecutor,
    pool: &BlockPool,
    seq: SeqId,
    tokens: &[u32],
    start: u32,
    table: &BlockTable,
) -> Result<Vec<f32>, ProposeError> {
    let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
    let kv_len = start + tokens.len() as u32;
    let seqs = [SeqSlice {
        seq,
        q_start: 0,
        q_len: tokens.len() as u32,
        kv_len,
        block_table: &table.blocks,
        block_formats: &[],
        reduce: None,
        logits_rows: 1,
        state_slot: None,
        snapshot_blocks: &[],
    }];
    let view = pool.view();
    let logits = executor.forward(&BatchInput {
        tokens,
        positions: &positions,
        seqs: &seqs,
        kv: &view,
    })?;
    match logits.slot(0) {
        LogitsSlot::Full(row) => Ok(row.to_vec()),
        LogitsSlot::Reduced(_) => Err(ProposeError::Unsupported(
            "the draft logits were device-reduced",
        )),
    }
}

/// The greedy token of `row`.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut value = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > value {
            value = v;
            best = i as u32;
        }
    }
    best
}
