//! Tier abstraction (P4 S-4): tier ids and block locations.

use serde::{Deserialize, Serialize};

/// A KV tier, fastest first. Serde and label spelling `"l0"`..`"l3"` everywhere (CONFLICT C-4);
/// `L3` (cluster) arrives with Phase 6.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierId {
    L0,
    L1,
    L2,
    L3,
}

impl TierId {
    /// Local tiers of Phase 4, fastest first (metric label sets).
    pub const LOCAL: [TierId; 3] = [TierId::L0, TierId::L1, TierId::L2];

    pub fn as_str(self) -> &'static str {
        match self {
            TierId::L0 => "l0",
            TierId::L1 => "l1",
            TierId::L2 => "l2",
            TierId::L3 => "l3",
        }
    }
}

/// Where one copy of a block lives. `slot` is the L0 `BlockId`; for L1/L2 `slab << 32 | slot`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct KvLocation {
    pub tier: TierId,
    pub slot: u64,
}
