//! P8 MTP: the `mtp` proposer over the checkpoint's own `mtp.*` head ([`crate::mtp::MtpHead`]).
//! The engine feeds the head the target's final hidden rows as **device** views
//! (`observe`/`seed_last_hidden`) and the head drafts greedily through its single layer; the
//! shared verifier and rollback apply unchanged. See `super` for the trait and registry.
use std::collections::HashMap;

use turbine_core::types::{BlockId, SeqId};
use turbine_kv::pool::BlockPool;
use turbine_kv::{BlockTable, blocks_for_tokens};
use turbine_tensor::{Tensor, TensorView};

use super::{Proposals, ProposeError, ProposeRequest, Proposer};
use crate::config::ModelArchConfig;
use crate::mtp::MtpHead;

/// One sequence's MTP KV and the last accepted hidden row (its own device buffer, so the next
/// draft step reads it without a host round trip).
struct Slot {
    table: BlockTable,
    /// The highest MTP KV position filled (0 = none); positions `1..=kv_rows` hold rows.
    kv_rows: u32,
    /// The target's final hidden row at the last accepted position (`[1, hidden]` on device).
    pending: Option<Tensor>,
}

impl Slot {
    fn new() -> Slot {
        Slot {
            table: BlockTable::new(),
            kv_rows: 0,
            pending: None,
        }
    }
}

/// P8 MTP: the draft over the checkpoint's own MTP head.
pub struct MtpProposer {
    head: MtpHead,
    pool: BlockPool,
    block_tokens: u32,
    max_seq_len: u32,
    max_batch_tokens: u32,
    dim: usize,
    slots: HashMap<SeqId, Slot>,
}

impl MtpProposer {
    /// Build the proposer over a loaded [`MtpHead`] and its KV pool; the pool's layout must be
    /// the head executor's, and it must hold one full-context slot per sequence that may
    /// speculate.
    pub fn new(
        head: MtpHead,
        pool: BlockPool,
        block_tokens: u32,
        max_seq_len: u32,
        max_batch_tokens: u32,
    ) -> Result<MtpProposer, ProposeError> {
        if *head.executor.kv_layout() != pool.layout() {
            return Err(ProposeError::Unsupported(
                "the mtp executor's KV layout differs from its pool's",
            ));
        }
        let dim = head.dim();
        Ok(MtpProposer {
            head,
            pool,
            block_tokens,
            max_seq_len,
            max_batch_tokens: max_batch_tokens.max(1),
            dim,
            slots: HashMap::new(),
        })
    }
}

impl Proposer for MtpProposer {
    fn name(&self) -> &'static str {
        "mtp"
    }

    fn validate(
        &self,
        target: &ModelArchConfig,
        draft: &ModelArchConfig,
        _tokenizer: &crate::Tokenizer,
    ) -> Result<(), ProposeError> {
        if target.hf_architecture != draft.hf_architecture {
            return Err(ProposeError::Unsupported(
                "the mtp head differs from the target's architecture",
            ));
        }
        Ok(())
    }

    /// MTP needs no prompt prefill here (the engine observes the head's rows): `begin` just
    /// allocates the sequence's slot.
    fn begin(&mut self, seq: SeqId, _prompt: &[u32]) -> Result<(), ProposeError> {
        self.slots.entry(seq).or_insert_with(Slot::new);
        Ok(())
    }

    fn observe(
        &mut self,
        seq: SeqId,
        positions: &[u32],
        tokens: &[u32],
        hidden_prev: TensorView<'_>,
    ) -> Result<(), ProposeError> {
        if positions.is_empty() {
            return Ok(());
        }
        let dim = self.dim;
        if positions.len() != tokens.len()
            || hidden_prev.shape.first().copied() != Some(positions.len())
        {
            return Err(ProposeError::Unsupported(
                "the mtp observation rows are inconsistent",
            ));
        }
        let MtpProposer {
            head,
            pool,
            block_tokens,
            max_batch_tokens,
            slots,
            ..
        } = self;
        let chunk = (*max_batch_tokens).max(1) as usize;
        let start_pos = positions.first().copied().unwrap_or(1);
        let s = slots.entry(seq).or_insert_with(Slot::new);
        // A gap (or overlap) from the filled end means the accepted prefix changed: drop past the
        // previous position and resync.
        if start_pos != s.kv_rows + 1 {
            truncate_table(
                pool,
                &mut s.table,
                start_pos.saturating_sub(1),
                *block_tokens,
            );
            s.kv_rows = start_pos.saturating_sub(1);
        }
        for i in (0..positions.len()).step_by(chunk) {
            let j = (i + chunk).min(positions.len());
            let (pos, tok) = (&positions[i..j], &tokens[i..j]);
            let prev = hidden_prev.rows(i, j - i);
            // The row for overall position p lands at MTP index (and KV slot) p - 1: the whole
            // MTP sequence starts at index 0, and RoPE is shift-invariant so the offset is
            // exact. `kv_len` is the MTP row count (the overall highest position).
            let kv_len = pos.last().copied().unwrap_or(0);
            let kv_pos: Vec<u32> = pos.iter().map(|p| p - 1).collect();
            ensure_room(pool, &mut s.table, kv_len, *block_tokens)?;
            let table = s.table.blocks.clone();
            let view = pool.view();
            head.forward_mtp(&view, seq, &table, &[], &kv_pos, kv_len, tok, prev)?;
            s.kv_rows = kv_len;
        }
        let _ = dim;
        Ok(())
    }

    fn seed_last_hidden(&mut self, seq: SeqId, hidden: TensorView<'_>) -> Result<(), ProposeError> {
        if hidden.shape.first().copied() != Some(1) {
            return Err(ProposeError::Unsupported(
                "the mtp last-hidden row has the wrong width",
            ));
        }
        let MtpProposer { head, slots, .. } = self;
        let s = slots.entry(seq).or_insert_with(Slot::new);
        if s.pending.is_none() {
            s.pending = Some(head.rows(1)?);
        }
        head.copy_rows(s.pending.as_ref().expect("set"), 0, hidden, 0, 1)?;
        Ok(())
    }

    fn propose(&mut self, request: ProposeRequest) -> Result<Proposals, ProposeError> {
        let MtpProposer {
            head,
            pool,
            block_tokens,
            max_seq_len,
            slots,
            ..
        } = self;
        let k = request
            .num_tokens
            .min(8)
            .min(max_seq_len.saturating_sub(request.accepted_len.saturating_add(1)));
        if k == 0 {
            return Ok(Proposals { tokens: Vec::new() });
        }
        let seq = request.seq;
        let n = request.accepted_len;
        let s = slots.get_mut(&seq).ok_or(ProposeError::NotBegun)?;
        if s.kv_rows + 1 < n || s.pending.is_none() {
            return Err(ProposeError::NotBegun);
        }
        truncate_table(pool, &mut s.table, n.saturating_sub(1), *block_tokens);
        s.kv_rows = n.saturating_sub(1);
        let mut tokens = Vec::with_capacity(k as usize);
        let mut next = request.last_token;
        for i in 0..k {
            let position = n + i;
            let klen = position;
            let kv_pos = position - 1;
            ensure_room(pool, &mut s.table, klen, *block_tokens)?;
            let table = s.table.blocks.clone();
            let view = pool.view();
            let hidden = s.pending.as_ref().expect("checked").view();
            let logits =
                head.forward_mtp(&view, seq, &table, &[], &[kv_pos], klen, &[next], hidden)?;
            tokens.push(argmax(&logits));
            s.kv_rows = position;
            let captured = head.capture(1).ok_or_else(|| {
                ProposeError::Model(crate::ModelError::Kernel(
                    turbine_kernels::KernelError::InvalidArgument {
                        message: "the mtp head captured no hidden row".into(),
                    },
                ))
            })?;
            head.copy_rows(s.pending.as_ref().expect("checked"), 0, captured, 0, 1)?;
            next = *tokens.last().expect("pushed");
        }
        Ok(Proposals { tokens })
    }

    fn truncate(&mut self, seq: SeqId, len: u32) -> Result<(), ProposeError> {
        let MtpProposer {
            pool,
            block_tokens,
            slots,
            ..
        } = self;
        let s = slots.get_mut(&seq).ok_or(ProposeError::NotBegun)?;
        truncate_table(pool, &mut s.table, len.saturating_sub(1), *block_tokens);
        s.kv_rows = len.saturating_sub(1);
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        if let Some(slot) = self.slots.remove(&seq) {
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
