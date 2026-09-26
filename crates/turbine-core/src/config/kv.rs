//! `kv` configuration section: the Phase 0 tier switches, the Phase 2 L0 pool size and the
//! Phase 4 tier, policy, transfer, session and prefetch keys (P4 §Configuration, S-15).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{ByteSize, ConfigError, HumanDuration, invalid};

/// L2 slots and slab-file headers are aligned to this many bytes (P4 §Data).
pub const KV_IO_ALIGN: u64 = 4096;

/// `kv.policy`: the eviction policy. Serialized as `"cost_aware"` / `"lru"`.
#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KvPolicyKind {
    CostAware,
    Lru,
}

impl KvPolicyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            KvPolicyKind::CostAware => "cost_aware",
            KvPolicyKind::Lru => "lru",
        }
    }
}

/// Facts about the host that some `kv` rules need (contract §3.2); `None` skips the rule.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostFacts {
    /// `MemTotal` of `/proc/meminfo`.
    pub mem_total_bytes: Option<u64>,
    /// Free space of the filesystem holding `kv.nvme.path`.
    pub disk_free_bytes: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvConfig {
    pub block_tokens: u32,
    pub gpu: KvGpuConfig,
    pub cpu: KvCpuConfig,
    pub nvme: KvNvmeConfig,
    pub policy: KvPolicyKind,
    /// Blocks scoring below this are dropped instead of demoted.
    pub demote_min_value: f64,
    /// `false`: no prefix lookup and no reuse (A/B switch).
    pub prefix_sharing: bool,
    pub transfer: KvTransferConfig,
    pub session: KvSessionConfig,
    pub prefetch: KvPrefetchConfig,
    pub policy_weights: KvPolicyWeights,
}

impl Default for KvConfig {
    fn default() -> Self {
        KvConfig {
            block_tokens: 128,
            gpu: KvGpuConfig::default(),
            cpu: KvCpuConfig::default(),
            nvme: KvNvmeConfig::default(),
            policy: KvPolicyKind::CostAware,
            demote_min_value: 0.0,
            prefix_sharing: true,
            transfer: KvTransferConfig::default(),
            session: KvSessionConfig::default(),
            prefetch: KvPrefetchConfig::default(),
            policy_weights: KvPolicyWeights::default(),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvGpuConfig {
    /// `false` is rejected from Phase 4: L0 is required.
    pub enabled: bool,
    /// L0 block-pool cap. Null (the default from Phase 3; Phase 2 defaulted to 8GiB) means
    /// the `kv` pool remainder of the budget (CONFLICT C-8).
    pub max_bytes: Option<ByteSize>,
}

impl Default for KvGpuConfig {
    fn default() -> Self {
        KvGpuConfig {
            enabled: true,
            max_bytes: None,
        }
    }
}

/// L1 pinned host tier; ignored with a WARN on unified-memory devices.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvCpuConfig {
    pub enabled: bool,
    pub max_bytes: ByteSize,
}

impl Default for KvCpuConfig {
    fn default() -> Self {
        KvCpuConfig {
            enabled: true,
            max_bytes: ByteSize::gib(64),
        }
    }
}

/// L2 NVMe slab-file tier.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvNvmeConfig {
    pub enabled: bool,
    pub path: PathBuf,
    pub max_bytes: ByteSize,
    pub slab_bytes: ByteSize,
    pub max_queue_depth: u32,
    pub io_threads: u32,
}

impl Default for KvNvmeConfig {
    fn default() -> Self {
        KvNvmeConfig {
            enabled: false,
            path: PathBuf::from("/var/lib/turbine/kv"),
            max_bytes: ByteSize::gib(64),
            slab_bytes: ByteSize::gib(1),
            max_queue_depth: 64,
            io_threads: 4,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvTransferConfig {
    /// Bound on bytes of block copies in flight across all local paths.
    pub max_inflight_bytes: ByteSize,
}

impl Default for KvTransferConfig {
    fn default() -> Self {
        KvTransferConfig {
            max_inflight_bytes: ByteSize::gib(1),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvSessionConfig {
    pub max_sessions: u32,
    /// Idle time after which a session's blocks demote out of L0.
    pub hot_ttl: HumanDuration,
    /// Idle time after which a session's blocks demote out of L1; greater than `hot_ttl`.
    pub warm_ttl: HumanDuration,
    /// Idle time after which the session's metadata is dropped.
    pub max_idle: HumanDuration,
}

impl Default for KvSessionConfig {
    fn default() -> Self {
        KvSessionConfig {
            max_sessions: 10_000,
            hot_ttl: HumanDuration::from_secs(60),
            warm_ttl: HumanDuration::from_secs(600),
            max_idle: HumanDuration::from_secs(3600),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvPrefetchConfig {
    /// How long before a predicted resume the prefetch starts.
    pub lead_time: HumanDuration,
    pub max_queue: u32,
}

impl Default for KvPrefetchConfig {
    fn default() -> Self {
        KvPrefetchConfig {
            lead_time: HumanDuration::from_secs(2),
            max_queue: 256,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct KvPolicyWeights {
    /// Reuse-probability boost of a hot session's blocks (0..1).
    pub session_active: f64,
    /// Half-life of the decayed hit count.
    pub hit_half_life: HumanDuration,
}

impl Default for KvPolicyWeights {
    fn default() -> Self {
        KvPolicyWeights {
            session_active: 0.5,
            hit_half_life: HumanDuration::from_secs(60),
        }
    }
}

impl KvConfig {
    /// Static rules (exit 2), called from `Config::validate`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(1..=1024).contains(&self.block_tokens) {
            return Err(invalid(
                "kv.block_tokens",
                format!("must be between 1 and 1024, got {}", self.block_tokens),
            ));
        }
        if !self.gpu.enabled {
            return Err(invalid("kv.gpu.enabled", "L0 is required"));
        }
        if self.cpu.enabled && self.cpu.max_bytes.0 == 0 {
            return Err(invalid(
                "kv.cpu.max_bytes",
                "must be greater than 0 when kv.cpu.enabled is true",
            ));
        }
        self.validate_nvme()?;
        if !(self.demote_min_value >= 0.0 && self.demote_min_value.is_finite()) {
            return Err(invalid(
                "kv.demote_min_value",
                format!(
                    "must be a finite number >= 0, got {}",
                    self.demote_min_value
                ),
            ));
        }
        if self.transfer.max_inflight_bytes.0 < KV_IO_ALIGN {
            return Err(invalid(
                "kv.transfer.max_inflight_bytes",
                format!(
                    "must be at least one block (and at least 4KiB), got {}",
                    self.transfer.max_inflight_bytes
                ),
            ));
        }
        let s = &self.session;
        if !(1..=1_000_000).contains(&s.max_sessions) {
            return Err(invalid(
                "kv.session.max_sessions",
                format!("must be between 1 and 1000000, got {}", s.max_sessions),
            ));
        }
        if s.warm_ttl <= s.hot_ttl {
            return Err(invalid(
                "kv.session.warm_ttl",
                format!(
                    "must be greater than kv.session.hot_ttl ({}), got {}",
                    s.hot_ttl, s.warm_ttl
                ),
            ));
        }
        let active = self.policy_weights.session_active;
        if !(0.0..=1.0).contains(&active) {
            return Err(invalid(
                "kv.policy_weights.session_active",
                format!("must be between 0 and 1, got {active}"),
            ));
        }
        Ok(())
    }

    fn validate_nvme(&self) -> Result<(), ConfigError> {
        let n = &self.nvme;
        if n.enabled && !n.path.is_absolute() {
            return Err(invalid(
                "kv.nvme.path",
                format!(
                    "must be an absolute path when kv.nvme.enabled is true, got {}",
                    n.path.display()
                ),
            ));
        }
        if n.max_bytes.0 == 0 {
            return Err(invalid("kv.nvme.max_bytes", "must be greater than 0"));
        }
        if n.slab_bytes.0 == 0 || !n.slab_bytes.0.is_multiple_of(KV_IO_ALIGN) {
            return Err(invalid(
                "kv.nvme.slab_bytes",
                format!("must be a non-zero multiple of 4KiB, got {}", n.slab_bytes),
            ));
        }
        if !(1..=1024).contains(&n.max_queue_depth) {
            return Err(invalid(
                "kv.nvme.max_queue_depth",
                format!("must be between 1 and 1024, got {}", n.max_queue_depth),
            ));
        }
        if !(1..=64).contains(&n.io_threads) {
            return Err(invalid(
                "kv.nvme.io_threads",
                format!("must be between 1 and 64, got {}", n.io_threads),
            ));
        }
        Ok(())
    }

    /// Rules that need the model's KV block size, checked at startup once the model config is
    /// parsed (exit 2): the in-flight bound holds one block and an enabled L2 slab holds one
    /// 4 KiB-rounded slot. Slots per slab are `slab_bytes / slot_bytes`, rounded down.
    pub fn validate_block_bytes(&self, block_bytes: u64) -> Result<(), ConfigError> {
        if self.transfer.max_inflight_bytes.0 < block_bytes {
            return Err(invalid(
                "kv.transfer.max_inflight_bytes",
                format!(
                    "must be at least one KV block ({block_bytes} bytes), got {}",
                    self.transfer.max_inflight_bytes
                ),
            ));
        }
        let slot = block_bytes.div_ceil(KV_IO_ALIGN) * KV_IO_ALIGN;
        if self.nvme.enabled && self.nvme.slab_bytes.0 < slot {
            return Err(invalid(
                "kv.nvme.slab_bytes",
                format!(
                    "must hold at least one {slot}-byte slot, got {}",
                    self.nvme.slab_bytes
                ),
            ));
        }
        Ok(())
    }

    /// Rules that need host facts. `host_reserve` is `reliability.memory.host_reserve_bytes`.
    /// `turbine-server` exits 1 for an error on a `kv.nvme.*` key and 2 for any other key.
    pub fn validate_host(
        &self,
        host: &HostFacts,
        host_reserve: ByteSize,
    ) -> Result<(), ConfigError> {
        if self.cpu.enabled
            && let Some(total) = host.mem_total_bytes
        {
            let limit = total.saturating_sub(host_reserve.0);
            if self.cpu.max_bytes.0 > limit {
                return Err(invalid(
                    "kv.cpu.max_bytes",
                    format!(
                        "{} exceeds MemTotal ({total} bytes) minus \
                         reliability.memory.host_reserve_bytes ({})",
                        self.cpu.max_bytes, host_reserve
                    ),
                ));
            }
        }
        if self.nvme.enabled
            && let Some(free) = host.disk_free_bytes
        {
            let limit = free - free / 10;
            if self.nvme.max_bytes.0 > limit {
                return Err(invalid(
                    "kv.nvme.max_bytes",
                    format!(
                        "{} exceeds the free space of {} ({free} bytes) minus 10 %",
                        self.nvme.max_bytes,
                        self.nvme.path.display()
                    ),
                ));
            }
        }
        Ok(())
    }
}
