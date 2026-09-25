//! Turbine KV cache (contract §11). Each module is a separate boundary — there is no
//! catch-all "KV manager" (TS §21 rule 6).

pub mod pool;
pub mod table;

pub use pool::{BlockPool, BlockPoolConfig, PoolError};
pub use table::{BlockTable, blocks_for_tokens};
