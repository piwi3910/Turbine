//! `kv` configuration section: the Phase 0 tier switches, the Phase 2 L0 pool size and the
//! Phase 4 tier, policy, transfer, session and prefetch keys (P4 §Configuration, S-15).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{ByteSize, ConfigError, HumanDuration, ModuleName, invalid};

/// L2 slots and slab-file headers are aligned to this many bytes (P4 §Data).
pub const KV_IO_ALIGN: u64 = 4096;

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
    /// Element format of the L0 pages (Phase 6a S-13); `bf16`, the exact default, or the
    /// lossy `fp8_e4m3`, which is only ever an explicit setting.
    pub dtype: KvDtypeChoice,
    pub gpu: KvGpuConfig,
    pub cpu: KvCpuConfig,
    pub nvme: KvNvmeConfig,
    /// Eviction policy (Phase 4; an `eviction_policy` registry name, validated with the other
    /// module keys by `Config::validate_modules`).
    pub policy: ModuleName,
    /// Blocks scoring below this are dropped instead of demoted. Absolute, on the eviction
    /// policy's scale: the policy prices a block's return trip at the lower tier's encoded size
    /// (P6b), so with a compressing lower tier an L0 block scores lower and a set threshold drops
    /// more blocks the cheaper that tier is (`docs/extending/eviction-policy.md`, Pitfalls).
    pub demote_min_value: f64,
    /// `false`: no prefix lookup and no reuse (A/B switch).
    pub prefix_sharing: bool,
    pub transfer: KvTransferConfig,
    pub session: KvSessionConfig,
    pub prefetch: KvPrefetchConfig,
    pub policy_weights: KvPolicyWeights,
    /// The last N full blocks of a sequence at demotion time leave L0 at the L0 format,
    /// whatever the tier's format (P6b S-2, user decision 2026-09-28, Q13); 0..=64.
    pub lossless_tail_blocks: u32,
    /// Whether requests without `x-turbine-kv-lossy` may reuse lossy cached blocks (P6b S-3,
    /// Q15).
    pub lossy_reuse: LossyReuse,
    /// Planner retrieval-cost penalty per codec name (P6b S-3, Q16): a lossy block's retrieval
    /// cost is multiplied by `1 + penalty` (0..=100). Null: every codec's own default
    /// (`KvCodec::default_lossy_penalty`); an entry overrides one codec and the others keep
    /// theirs. The names are checked against the `kv_format` registry at startup.
    pub lossy_penalty: Option<BTreeMap<ModuleName, f64>>,
    /// The pressure-driven compression ladder (P6b S-6, S-7).
    pub ladder: KvLadderConfig,
}

impl Default for KvConfig {
    fn default() -> Self {
        KvConfig {
            block_tokens: 128,
            dtype: KvDtypeChoice::Bf16,
            gpu: KvGpuConfig::default(),
            cpu: KvCpuConfig::default(),
            nvme: KvNvmeConfig::default(),
            policy: ModuleName::fixed("cost_aware"),
            demote_min_value: 0.0,
            prefix_sharing: true,
            transfer: KvTransferConfig::default(),
            session: KvSessionConfig::default(),
            prefetch: KvPrefetchConfig::default(),
            policy_weights: KvPolicyWeights::default(),
            lossless_tail_blocks: 1,
            lossy_reuse: LossyReuse::Allow,
            lossy_penalty: None,
            ladder: KvLadderConfig::default(),
        }
    }
}

/// `kv.dtype`: how the L0 pool stores K and V (Phase 6a S-13; `tq4` / `tq2` arrive with
/// `phase-6b-kv-compression`).
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KvDtypeChoice {
    /// BF16 pages: exact.
    #[default]
    Bf16,
    /// OCP e4m3fn pages with one K and one V scale per layer (the checkpoint's `k_scale` /
    /// `v_scale` when present, else 1.0): half the bytes, lossy.
    Fp8E4m3,
    /// TurboQuant 4-bit pages (P6b S-5): `experimental` on the CPU reference provider and on
    /// gfx1201 Llama / OLMoE (the ABI v2.11 mixed-format attention reads the tables the server
    /// uploads at startup); a library without it is refused with `kv_tq_unavailable`.
    Tq4,
    /// TurboQuant 2-bit pages (P6b S-5): parsed, but refused at startup on every backend
    /// (`kv_tq2_l0_refused`, user decision 2026-10-02); `tq2` stays a lower-tier format.
    Tq2,
}

impl KvDtypeChoice {
    /// The configuration spelling, also the support-matrix KV column.
    pub fn as_str(self) -> &'static str {
        match self {
            KvDtypeChoice::Bf16 => "bf16",
            KvDtypeChoice::Fp8E4m3 => "fp8_e4m3",
            KvDtypeChoice::Tq4 => "tq4",
            KvDtypeChoice::Tq2 => "tq2",
        }
    }

    /// True for a lossy (quantized) format.
    pub fn is_lossy(self) -> bool {
        self != KvDtypeChoice::Bf16
    }

    /// True for a TurboQuant format (`tq4`, `tq2`).
    pub fn is_turboquant(self) -> bool {
        matches!(self, KvDtypeChoice::Tq4 | KvDtypeChoice::Tq2)
    }
}

/// `kv.lossy_reuse`: the default for requests without `x-turbine-kv-lossy`.
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum LossyReuse {
    /// Lookups continue on lossy blocks (the planner still weighs their penalty).
    #[default]
    Allow,
    /// Lookups stop at the first lossy block; the request recomputes from there.
    Deny,
}

/// `kv.ladder`: the pressure-driven compression ladder (P6b S-6, S-7; user decision
/// 2026-09-28, Q17, Q18). Off by default.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct KvLadderConfig {
    /// Needs L1 or L2 enabled.
    pub enabled: bool,
    /// Whether L0 joins the ladder (S-7); false keeps it in L1/L2.
    pub l0: bool,
    /// The lossiest rung: a lossy codec of the `kv_format` registry (checked at startup).
    /// Default `tq4`: `tq2` failed its lower-tier GSM8K gate (user decision 2026-10-02).
    pub max_format: ModuleName,
    /// Tier fill above which an upper tier compresses (and the floor may drop); 0.5 < v ≤ 1.0.
    pub high_water: f64,
    /// At YELLOW the lowest tier compresses only while `fill + demand` exceeds this (GREEN
    /// headroom; user decision 2026-09-29); 0.5 ≤ v < `high_water`.
    pub low_water: f64,
}

impl Default for KvLadderConfig {
    fn default() -> Self {
        KvLadderConfig {
            enabled: false,
            l0: true,
            max_format: ModuleName::fixed("tq4"),
            high_water: 0.95,
            low_water: 0.85,
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
    /// Codec of the blocks L1 holds (P6b S-2): a `kv_format` registry name, checked at startup
    /// with the tier ordering (not more precise than L0).
    pub format: ModuleName,
}

impl Default for KvCpuConfig {
    fn default() -> Self {
        KvCpuConfig {
            enabled: true,
            max_bytes: ByteSize::gib(64),
            format: ModuleName::fixed("l0"),
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
    /// Codec of the blocks L2 holds (P6b S-2): a `kv_format` registry name, not more precise
    /// than L1 when L1 is enabled, else than L0 (checked at startup).
    pub format: ModuleName,
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
            format: ModuleName::fixed("l0"),
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
        self.validate_formats()
    }

    /// P6b S-2, S-3, S-6: the lossless tail, the penalties and the ladder. Codec names and the
    /// tier ordering need the `kv_format` registry: `Config::validate_modules` and the server's
    /// startup check them.
    fn validate_formats(&self) -> Result<(), ConfigError> {
        if self.lossless_tail_blocks > 64 {
            return Err(invalid(
                "kv.lossless_tail_blocks",
                format!(
                    "must be between 0 and 64, got {}",
                    self.lossless_tail_blocks
                ),
            ));
        }
        for (name, &value) in self.lossy_penalty.iter().flatten() {
            if !(0.0..=100.0).contains(&value) {
                return Err(invalid(
                    &format!("kv.lossy_penalty.{name}"),
                    format!("must be between 0 and 100, got {value}"),
                ));
            }
        }
        let ladder = &self.ladder;
        if !(0.5..1.0).contains(&ladder.low_water) {
            return Err(invalid(
                "kv.ladder.low_water",
                format!(
                    "must be at least 0.5 and below 1.0, got {}",
                    ladder.low_water
                ),
            ));
        }
        if !(ladder.high_water > 0.5 && ladder.high_water <= 1.0) {
            return Err(invalid(
                "kv.ladder.high_water",
                format!(
                    "must be above 0.5 and at most 1.0, got {}",
                    ladder.high_water
                ),
            ));
        }
        if ladder.high_water <= ladder.low_water {
            return Err(invalid(
                "kv.ladder.high_water",
                format!(
                    "must be greater than kv.ladder.low_water ({}), got {}",
                    ladder.low_water, ladder.high_water
                ),
            ));
        }
        if ladder.enabled && !self.cpu.enabled && !self.nvme.enabled {
            return Err(invalid(
                "kv.ladder.enabled",
                "needs a lower tier: kv.cpu.enabled or kv.nvme.enabled",
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

    /// The configured `kv.lossy_penalty` of codec `name`, if any (else the codec's default).
    pub fn lossy_penalty_override(&self, name: &str) -> Option<f64> {
        self.lossy_penalty
            .as_ref()?
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, &v)| v)
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn parse(yaml: &str) -> Result<super::super::Config, ConfigError> {
        super::super::load_from_str(yaml, Path::new("test.yaml"), &[])
    }

    /// `kv.dtype` defaults to the exact `bf16`, accepts `fp8_e4m3`, and refuses anything else
    /// naming the key. Breaks if the default turns lossy or the key is not wired.
    #[test]
    fn kv_dtype_key() {
        let base = "model: {path: /models/m}\n";
        let default = parse(base).expect("defaults");
        assert_eq!(default.kv.dtype, KvDtypeChoice::Bf16);
        assert!(!default.kv.dtype.is_lossy());
        let fp8 = parse(&format!("{base}kv: {{dtype: fp8_e4m3}}\n")).expect("fp8_e4m3");
        assert_eq!(fp8.kv.dtype, KvDtypeChoice::Fp8E4m3);
        assert_eq!(fp8.kv.dtype.as_str(), "fp8_e4m3");
        assert!(fp8.kv.dtype.is_lossy());
        for bad in ["int8", "fp8", "tq3"] {
            let err = parse(&format!("{base}kv: {{dtype: {bad}}}\n")).expect_err(bad);
            assert_eq!(err.key(), Some("kv.dtype"), "{bad}: {err}");
        }
    }
}
