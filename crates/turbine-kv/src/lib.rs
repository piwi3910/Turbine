//! Turbine KV cache (contract §11, TS §8). Each module is a separate boundary with its own
//! public API — there is no catch-all "KV manager" (TS §21 rule 6).

pub mod codec;
pub mod directory;
pub mod document;
pub mod hierarchy;
pub mod identity;
pub mod metrics;
pub mod planner;
pub mod policy;
pub mod pool;
pub mod reclaim;
pub mod session;
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
    use crate::codec::{self, conformance::kv_codecs_suite};
    use crate::policy::{self, conformance::eviction_policies_suite};

    /// Every KV codec fits its slot, round-trips exactly (lossless) or within its documented
    /// bound (lossy) on seeded Gaussian and outlier-heavy blocks, encodes deterministically and
    /// refuses short buffers (`kv_codecs_suite`); `l0` is first and slot sizes never grow along
    /// the lossiness order. Catches a codec that breaks the contract.
    #[test]
    fn kv_codecs() {
        let reg = codec::registry();
        assert_eq!(reg.point(), "kv_format");
        if let Err(failures) = kv_codecs_suite(reg) {
            panic!("kv codec conformance failures:\n{}", failures.join("\n"));
        }
    }

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
