//! Turbine KV cache (contract §11, TS §8). Each module is a separate boundary with its own
//! public API — there is no catch-all "KV manager" (TS §21 rule 6).

pub mod directory;
pub mod document;
pub mod identity;
pub mod metrics;
pub mod planner;
pub mod policy;
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

/// Contract §24: every registry of this crate passes the shared conformance check and every
/// registered module its extension point's suite, run over the registry itself.
#[cfg(test)]
mod registry_conformance {
    use crate::policy::{self, conformance::eviction_policies_suite};

    /// Every eviction policy scores finitely and deterministically and orders every candidate
    /// (`eviction_policies_suite`); `cost_aware` is the configuration default and comes first.
    /// Catches a duplicate or malformed policy name, an empty registry, or a policy that breaks
    /// a property.
    #[test]
    fn eviction_policies() {
        let reg = policy::registry();
        assert_eq!(reg.point(), "eviction_policy");
        assert_eq!(reg.names(), ["cost_aware", "lru"]);
        if let Err(failures) = eviction_policies_suite(reg) {
            panic!(
                "eviction policy conformance failures:\n{}",
                failures.join("\n")
            );
        }
    }
}
