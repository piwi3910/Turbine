//! Per-sequence block tables: the ordered L0 blocks holding one sequence's KV.

use smallvec::SmallVec;
use turbine_core::types::BlockId;

/// Blocks needed to hold `tokens` tokens: `ceil(tokens / block_tokens)`.
/// `block_tokens` must be at least 1 (config validation guarantees it).
pub fn blocks_for_tokens(tokens: u32, block_tokens: u32) -> u32 {
    tokens.div_ceil(block_tokens.max(1))
}

/// The ordered blocks of one sequence and the number of tokens they hold.
/// Invariant kept by its users: `blocks.len() == blocks_for_tokens(tokens, block_tokens)`
/// once the tokens are written; token `i` lives in `blocks[i / block_tokens]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockTable {
    pub blocks: SmallVec<[BlockId; 16]>,
    pub tokens: u32,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Additional blocks to allocate before appending `extra_tokens` tokens.
    pub fn blocks_needed(&self, extra_tokens: u32, block_tokens: u32) -> u32 {
        let have = u32::try_from(self.blocks.len()).unwrap_or(u32::MAX);
        blocks_for_tokens(self.tokens.saturating_add(extra_tokens), block_tokens)
            .saturating_sub(have)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_counts() {
        assert_eq!(blocks_for_tokens(0, 16), 0);
        assert_eq!(blocks_for_tokens(1, 16), 1);
        assert_eq!(blocks_for_tokens(16, 16), 1);
        assert_eq!(blocks_for_tokens(17, 16), 2);
        let mut t = BlockTable::new();
        assert_eq!(t.blocks_needed(16, 16), 1);
        t.blocks.push(BlockId(0));
        t.tokens = 15;
        assert_eq!(t.blocks_needed(1, 16), 0);
        assert_eq!(t.blocks_needed(2, 16), 1);
        // 15 + 33 = 48 tokens → 3 blocks, one already held.
        assert_eq!(t.blocks_needed(33, 16), 2);
        assert_eq!(t.blocks_needed(34, 16), 3);
    }
}
