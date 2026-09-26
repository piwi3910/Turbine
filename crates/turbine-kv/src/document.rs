//! Body of `GET /turbine/v1/kv` (P2 §Data; tier names per CONFLICT C-4). Built from the
//! pool's public counters only.

use serde::Serialize;

use crate::metrics::TIER_L0;
use crate::pool::BlockPool;

/// `{"tiers": [...]}`; Phase 2 has the one GPU tier `l0`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvDocument {
    pub tiers: Vec<KvTierDocument>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvTierDocument {
    pub tier: &'static str,
    pub dtype: &'static str,
    pub block_tokens: u32,
    pub block_bytes: u64,
    pub blocks_total: u32,
    pub blocks_used: u32,
    pub blocks_free: u32,
}

impl KvDocument {
    pub fn from_pool(pool: &BlockPool) -> KvDocument {
        let layout = pool.layout();
        KvDocument {
            tiers: vec![KvTierDocument {
                tier: TIER_L0,
                dtype: layout.dtype.as_str(),
                block_tokens: layout.block_tokens,
                block_bytes: layout.block_bytes(),
                blocks_total: pool.total_blocks(),
                blocks_used: pool.used_blocks(),
                blocks_free: pool.free_blocks(),
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::config::ByteSize;
    use turbine_core::types::{DType, DeviceId, KvLayout};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::metrics::KvMetrics;
    use crate::pool::{BlockPool, BlockPoolConfig};

    #[test]
    fn document_and_metrics_agree() {
        // Llama-3.2-3B BF16, 16-token blocks, the default 8 GiB pool.
        let layout = KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        };
        let cfg = BlockPoolConfig::for_bytes(layout, ByteSize::gib(8).0);
        assert_eq!(cfg.num_blocks, 4681);
        // Host "device" pages are zero-filled lazily, so this reserves no real memory.
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), ByteSize::gib(9).0);
        let mut pool = BlockPool::new(cfg, mem).unwrap();

        let reg = MetricsRegistry::new();
        let metrics = KvMetrics::register(&reg);
        metrics.record(&pool);
        let text = reg.render().unwrap();
        assert!(
            text.contains("turbine_kv_blocks{tier=\"l0\",state=\"free\"} 4681"),
            "{text}"
        );

        let _held = pool.allocate(1024).unwrap();
        metrics.record(&pool);
        let doc = KvDocument::from_pool(&pool);
        assert_eq!(
            serde_json::to_value(&doc).unwrap(),
            serde_json::json!({
                "tiers": [{
                    "tier": "l0",
                    "dtype": "bf16",
                    "block_tokens": 16,
                    "block_bytes": 1_835_008,
                    "blocks_total": 4681,
                    "blocks_used": 1024,
                    "blocks_free": 3657
                }]
            })
        );
        let text = reg.render().unwrap();
        assert!(
            text.contains("turbine_kv_blocks{tier=\"l0\",state=\"used\"} 1024"),
            "{text}"
        );
        assert!(
            text.contains("turbine_kv_blocks{tier=\"l0\",state=\"free\"} 3657"),
            "{text}"
        );

        // OLMoE-1B-7B: 2 097 152-byte blocks, 4 096 in 8 GiB.
        let olmoe = KvLayout {
            num_layers: 16,
            num_kv_heads: 16,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        };
        assert_eq!(olmoe.block_bytes(), 2_097_152);
        assert_eq!(
            BlockPoolConfig::for_bytes(olmoe, ByteSize::gib(8).0).num_blocks,
            4096
        );
    }

    #[test]
    fn default_page_document() {
        // Llama-3.2-3B BF16 at the default 128-token page: 585 blocks of 14 680 064 bytes in
        // 8 GiB (the remainder is left unused); OLMoE-1B-7B: 512 of 16 777 216.
        let layout = KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 128,
        };
        let cfg = BlockPoolConfig::for_bytes(layout, ByteSize::gib(8).0);
        assert_eq!(cfg.num_blocks, 585);
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), ByteSize::gib(9).0);
        let mut pool = BlockPool::new(cfg, mem).unwrap();
        let _held = pool.allocate(100).unwrap();
        assert_eq!(
            serde_json::to_value(KvDocument::from_pool(&pool)).unwrap(),
            serde_json::json!({
                "tiers": [{
                    "tier": "l0",
                    "dtype": "bf16",
                    "block_tokens": 128,
                    "block_bytes": 14_680_064,
                    "blocks_total": 585,
                    "blocks_used": 100,
                    "blocks_free": 485
                }]
            })
        );

        let olmoe = KvLayout {
            num_layers: 16,
            num_kv_heads: 16,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 128,
        };
        assert_eq!(olmoe.block_bytes(), 16_777_216);
        assert_eq!(
            BlockPoolConfig::for_bytes(olmoe, ByteSize::gib(8).0).num_blocks,
            512
        );
    }
}
