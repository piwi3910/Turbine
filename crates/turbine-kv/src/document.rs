//! Body of `GET /turbine/v1/kv`. Phase 2 (§Data; tier names per CONFLICT C-4) built it from the
//! pool's public counters; Phase 4 (§Data) adds the policy, per-tier state, hit rate, sessions,
//! prefetch and transfer objects. The Phase 4 parts are optional and flattened, so
//! `KvDocument::from_pool` still renders exactly the Phase 2 JSON while `KvHierarchy::document`
//! renders the full Phase 4 document (every P2 tier field kept inside each tier object).

use serde::Serialize;
use turbine_core::types::PressureState;

use crate::metrics::TIER_L0;
use crate::pool::BlockPool;

/// `{"tiers": [...]}` plus, from Phase 4, the summary keys.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvDocument {
    pub tiers: Vec<KvTierDocument>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub summary: Option<KvSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvTierDocument {
    pub tier: &'static str,
    pub dtype: &'static str,
    pub block_tokens: u32,
    pub block_bytes: u64,
    pub blocks_total: u32,
    pub blocks_used: u32,
    pub blocks_free: u32,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub state: Option<TierState>,
}

/// Phase 4 per-tier keys (P4 §Data).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TierState {
    pub enabled: bool,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub blocks: u64,
    pub referenced_blocks: u64,
    pub pressure: PressureState,
    pub degraded: bool,
    pub est_latency_seconds: f64,
    pub est_bandwidth_bytes_per_second: Option<f64>,
}

/// Phase 4 top-level keys (P4 §Data).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvSummary {
    pub policy: &'static str,
    pub prefix_sharing: bool,
    pub block_tokens: u32,
    pub block_bytes: u64,
    pub unified_memory: bool,
    pub hit_rate: HitRate,
    pub sessions: Sessions,
    pub prefetch: Prefetch,
    pub transfers: Transfers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct HitRate {
    pub window_seconds: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Sessions {
    pub active: u64,
    pub max: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Prefetch {
    pub queued: u64,
    pub used: u64,
    pub wasted: u64,
    pub cancelled: u64,
}

/// Local tier copies in flight. CONFLICT C-18: Phase 7 extends this object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Transfers {
    pub inflight_bytes: u64,
    pub max_inflight_bytes: u64,
}

impl KvDocument {
    /// The Phase 2 document: the one GPU tier `l0`, from the pool's counters.
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
                state: None,
            }],
            summary: None,
        }
    }
}

/// Prompt and cached token counts over a sliding window of 300 s in 30 buckets of 10 s.
#[derive(Clone, Debug)]
pub struct HitWindow {
    /// (bucket index = seconds / 10, prompt tokens, cached tokens); `u64::MAX` marks unused.
    buckets: [(u64, u64, u64); HitWindow::BUCKETS],
}

impl Default for HitWindow {
    fn default() -> Self {
        HitWindow {
            buckets: [(u64::MAX, 0, 0); HitWindow::BUCKETS],
        }
    }
}

impl HitWindow {
    pub const WINDOW_SECONDS: u64 = 300;
    const BUCKETS: usize = 30;
    const BUCKET_SECONDS: u64 = Self::WINDOW_SECONDS / Self::BUCKETS as u64;

    pub fn record(&mut self, now_secs: u64, prompt_tokens: u64, cached_tokens: u64) {
        let slot = now_secs / Self::BUCKET_SECONDS;
        let b = &mut self.buckets[(slot % Self::BUCKETS as u64) as usize];
        if b.0 != slot {
            *b = (slot, 0, 0);
        }
        b.1 += prompt_tokens;
        b.2 += cached_tokens;
    }

    /// (prompt tokens, cached tokens) of the last 300 s.
    pub fn totals(&self, now_secs: u64) -> (u64, u64) {
        let slot = now_secs / Self::BUCKET_SECONDS;
        self.buckets
            .iter()
            .filter(|b| b.0 != u64::MAX && slot.saturating_sub(b.0) < Self::BUCKETS as u64)
            .fold((0, 0), |acc, b| (acc.0 + b.1, acc.1 + b.2))
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

    #[test]
    fn hit_window_rolls_over_300_seconds() {
        let mut w = HitWindow::default();
        assert_eq!(w.totals(0), (0, 0));
        w.record(5, 100, 60);
        w.record(9, 50, 50);
        w.record(15, 10, 0);
        assert_eq!(w.totals(20), (160, 110));
        assert_eq!(
            w.totals(299),
            (160, 110),
            "bucket 0 still inside the window"
        );
        assert_eq!(w.totals(300), (10, 0), "bucket 0 left the window");
        // A reused bucket starts from zero.
        w.record(305, 7, 7);
        assert_eq!(w.totals(305), (17, 7));
        assert_eq!(w.totals(10_000), (0, 0));
    }
}
