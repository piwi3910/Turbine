//! Turbine KV cache (contract §11, TS §8). Each module is a separate boundary with its own
//! public API — there is no catch-all "KV manager" (TS §21 rule 6).

pub mod directory;
pub mod document;
pub mod identity;
pub mod metrics;
pub mod planner;
pub mod pool;
pub mod reclaim;
pub mod table;
pub mod tier;
pub mod transfer;

pub use document::{KvDocument, KvTierDocument};
pub use metrics::KvMetrics;
pub use pool::{BlockPool, BlockPoolConfig, PoolError};
pub use reclaim::L0Reclaimer;
pub use table::{BlockTable, blocks_for_tokens};

#[cfg(test)]
mod test_log;
