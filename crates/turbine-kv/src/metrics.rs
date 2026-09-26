//! KV occupancy metrics (P2 §Metrics; tier label `l0` per CONFLICT C-4).

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use turbine_observability::MetricsRegistry;

use crate::pool::BlockPool;

/// The Phase 2 tier label of the GPU pool (CONFLICT C-4).
pub const TIER_L0: &str = "l0";

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct KvBlockLabels {
    tier: &'static str,
    state: &'static str,
}

/// `turbine_kv_blocks{tier,state}`; updated by the engine after every iteration.
#[derive(Clone)]
pub struct KvMetrics {
    used: Gauge,
    free: Gauge,
}

impl KvMetrics {
    pub fn register(reg: &MetricsRegistry) -> KvMetrics {
        let blocks = reg.register(
            "turbine_kv_blocks",
            "KV blocks per tier and state",
            Family::<KvBlockLabels, Gauge>::default(),
        );
        let gauge = |state| {
            blocks
                .get_or_create(&KvBlockLabels {
                    tier: TIER_L0,
                    state,
                })
                .clone()
        };
        KvMetrics {
            used: gauge("used"),
            free: gauge("free"),
        }
    }

    /// Publish the pool's current occupancy.
    pub fn record(&self, pool: &BlockPool) {
        self.used.set(i64::from(pool.used_blocks()));
        self.free.set(i64::from(pool.free_blocks()));
    }
}

/// The startup INFO line (`event="kv_pool"`) naming the block size and count.
pub fn log_pool_startup(pool: &BlockPool) {
    let layout = pool.layout();
    tracing::info!(
        event = "kv_pool",
        tier = TIER_L0,
        dtype = layout.dtype.as_str(),
        block_tokens = layout.block_tokens,
        block_bytes = layout.block_bytes(),
        num_blocks = pool.total_blocks(),
        "KV block pool allocated"
    );
}
