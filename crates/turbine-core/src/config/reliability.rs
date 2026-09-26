//! `reliability` section: the P0 keys `enabled`, `emergency_vram_reserve`,
//! `adaptive_admission` plus the P3 S-15 sub-sections, with static validation (exit 2).

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{ByteSize, ConfigError, HumanDuration, invalid};
use crate::pressure::PressureSignal;

/// Per-signal thresholds for YELLOW, ORANGE, RED, SURVIVAL; `None` = level unused.
pub type ThresholdLevels = [Option<f64>; 4];

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ReliabilityConfig {
    /// false: state fixed at GREEN; recovery and queue bounds remain.
    pub enabled: bool,
    /// Held outside normal scheduling; 0 disables the reserve.
    pub emergency_vram_reserve: ByteSize,
    /// false: admission checks only hard capacity and queue bounds.
    pub adaptive_admission: bool,
    pub memory: ReliabilityMemoryConfig,
    pub telemetry: ReliabilityTelemetryConfig,
    pub pressure: PressureConfig,
    pub admission: AdmissionConfig,
    pub recovery: RecoveryConfig,
    pub circuit: CircuitConfig,
    #[cfg(feature = "fault-injection")]
    pub fault_injection: Option<FaultInjectionConfig>,
    /// Without the `fault-injection` build the section is parsed only so `validate` can reject
    /// it naming the key (P3 S-16).
    #[cfg(not(feature = "fault-injection"))]
    pub fault_injection: Option<serde_norway::Value>,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        ReliabilityConfig {
            enabled: true,
            emergency_vram_reserve: ByteSize::gib(2),
            adaptive_admission: true,
            memory: ReliabilityMemoryConfig::default(),
            telemetry: ReliabilityTelemetryConfig::default(),
            pressure: PressureConfig::default(),
            admission: AdmissionConfig::default(),
            recovery: RecoveryConfig::default(),
            circuit: CircuitConfig::default(),
            fault_injection: None,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ReliabilityMemoryConfig {
    /// Execution workspace pool.
    pub workspace_bytes: ByteSize,
    /// Context, streams, graphs, allocator slack.
    pub runtime_overhead_bytes: ByteSize,
    /// Optional hard cap on everything Turbine claims on a device (co-tenancy).
    pub device_budget_bytes: Option<ByteSize>,
    /// Host memory Turbine never plans to use; on unified devices the budget is
    /// `MemAvailable` at startup minus this.
    pub host_reserve_bytes: ByteSize,
}

impl Default for ReliabilityMemoryConfig {
    fn default() -> Self {
        ReliabilityMemoryConfig {
            workspace_bytes: ByteSize::gib(1),
            runtime_overhead_bytes: ByteSize::gib(1),
            device_budget_bytes: None,
            host_reserve_bytes: ByteSize::gib(8),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ReliabilityTelemetryConfig {
    /// Fast tick: `/proc` files and the pool ledger.
    pub interval: HumanDuration,
    /// Vendor tick: amd-smi / NVML.
    pub vendor_interval: HumanDuration,
    /// Deadline per vendor call.
    pub call_timeout: HumanDuration,
    pub stale_after: HumanDuration,
}

impl Default for ReliabilityTelemetryConfig {
    fn default() -> Self {
        ReliabilityTelemetryConfig {
            interval: HumanDuration::from_millis(100),
            vendor_interval: HumanDuration::from_secs(1),
            call_timeout: HumanDuration::from_millis(500),
            stale_after: HumanDuration::from_secs(5),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct PressureConfig {
    pub escalate_samples: u32,
    pub deescalate_dwell: HumanDuration,
    /// Relative to the threshold: de-escalation needs value < threshold × (1 − margin)
    /// (× (1 + margin) for "lower is worse" signals).
    pub exit_margin: f64,
    /// Overrides of the P3 signal table only; signals absent here keep their defaults.
    #[serde(
        serialize_with = "serialize_thresholds",
        deserialize_with = "deserialize_thresholds"
    )]
    pub thresholds: BTreeMap<PressureSignal, ThresholdLevels>,
}

impl Default for PressureConfig {
    fn default() -> Self {
        PressureConfig {
            escalate_samples: 2,
            deescalate_dwell: HumanDuration::from_secs(10),
            exit_margin: 0.05,
            thresholds: BTreeMap::new(),
        }
    }
}

/// Every signal appears (null when not overridden) so the loader's key schema knows each
/// `reliability.pressure.thresholds.<signal>` key and names unknown signals.
fn serialize_thresholds<S: Serializer>(
    map: &BTreeMap<PressureSignal, ThresholdLevels>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let all: BTreeMap<&'static str, Option<&ThresholdLevels>> = PressureSignal::ALL
        .iter()
        .map(|s| (s.as_str(), map.get(s)))
        .collect();
    all.serialize(serializer)
}

/// Null entries (as produced by `serialize_thresholds`) are not overrides.
fn deserialize_thresholds<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<PressureSignal, ThresholdLevels>, D::Error> {
    let raw = BTreeMap::<PressureSignal, Option<ThresholdLevels>>::deserialize(deserializer)?;
    Ok(raw.into_iter().filter_map(|(s, l)| Some((s, l?))).collect())
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct AdmissionConfig {
    pub max_queue: u32,
    pub queue_timeout: HumanDuration,
    /// Prefills above this many new tokens are "expensive" (TS §9 ORANGE).
    pub large_prefill_tokens: u32,
    /// Smaller requests that may overtake a non-fitting queue head before it blocks the queue.
    pub max_bypass: u32,
    /// Removed key (KV overcommit is not offered in Phase 3): parsed only so `validate` can
    /// reject it naming the key.
    pub kv_overcommit: Option<serde_norway::Value>,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        AdmissionConfig {
            max_queue: 256,
            queue_timeout: HumanDuration::from_secs(30),
            large_prefill_tokens: 2048,
            max_bypass: 8,
            kv_overcommit: None,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct RecoveryConfig {
    pub max_retries: u32,
    /// Doubled per retry.
    pub backoff: HumanDuration,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        RecoveryConfig {
            max_retries: 3,
            backoff: HumanDuration::from_millis(50),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct CircuitConfig {
    /// OOM recoveries within `window` that open the circuit.
    pub oom_recoveries_to_open: u32,
    pub window: HumanDuration,
    /// Ratio of windowed p95 step time to baseline that degrades the circuit.
    pub latency_drift_degraded: f64,
    /// Ratio that opens the circuit.
    pub latency_drift_open: f64,
    pub cooldown: HumanDuration,
    pub drain_timeout: HumanDuration,
    pub probe_successes: u32,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        CircuitConfig {
            oom_recoveries_to_open: 3,
            window: HumanDuration::from_secs(60),
            latency_drift_degraded: 2.0,
            latency_drift_open: 4.0,
            cooldown: HumanDuration::from_secs(30),
            drain_timeout: HumanDuration::from_secs(120),
            probe_successes: 3,
        }
    }
}

/// `reliability.fault_injection` (P3 S-16), only in the `fault-injection` build.
#[cfg(feature = "fault-injection")]
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct FaultInjectionConfig {
    /// Fail every Nth pool reservation (absent = never).
    pub alloc_fail_every: Option<u32>,
    /// Raise a device out-of-memory error at this engine iteration.
    pub oom_at_iteration: Option<u64>,
    /// Raise a non-OOM kernel error at this engine iteration.
    pub kernel_error_at_iteration: Option<u64>,
    /// Make the injected kernel error context-corrupting (sticky → exit 3).
    pub kernel_error_sticky: bool,
    /// Override every vendor temperature reading (°C).
    pub telemetry_temperature_c: Option<f64>,
    /// Delay every vendor telemetry call by this long.
    pub telemetry_delay: Option<HumanDuration>,
}

impl ReliabilityConfig {
    /// Static rules (exit 2, P3 §Interfaces "Startup validation"); `block_tokens` is
    /// `kv.block_tokens`.
    pub fn validate(&self, block_tokens: u32) -> Result<(), ConfigError> {
        let t = &self.telemetry;
        let (interval, vendor) = (t.interval.0, t.vendor_interval.0);
        if !(Duration::from_millis(50)..=Duration::from_secs(10)).contains(&interval) {
            return Err(invalid(
                "reliability.telemetry.interval",
                format!("must be between 50ms and 10s, got {}", t.interval),
            ));
        }
        if vendor < interval || vendor > Duration::from_secs(60) {
            return Err(invalid(
                "reliability.telemetry.vendor_interval",
                format!(
                    "must be at least reliability.telemetry.interval ({}) and at most 60s, got {}",
                    t.interval, t.vendor_interval
                ),
            ));
        }
        if t.call_timeout.0.is_zero() || t.call_timeout.0 >= Duration::from_secs(10) {
            return Err(invalid(
                "reliability.telemetry.call_timeout",
                format!(
                    "must be greater than 0 and less than 10s, got {}",
                    t.call_timeout
                ),
            ));
        }
        if t.stale_after.0 <= vendor {
            return Err(invalid(
                "reliability.telemetry.stale_after",
                format!(
                    "must be longer than reliability.telemetry.vendor_interval ({}), got {}",
                    t.vendor_interval, t.stale_after
                ),
            ));
        }

        let p = &self.pressure;
        if !(1..=20).contains(&p.escalate_samples) {
            return Err(invalid(
                "reliability.pressure.escalate_samples",
                format!("must be between 1 and 20, got {}", p.escalate_samples),
            ));
        }
        if p.deescalate_dwell.0 < interval {
            return Err(invalid(
                "reliability.pressure.deescalate_dwell",
                format!(
                    "must be at least reliability.telemetry.interval ({}), got {}",
                    t.interval, p.deescalate_dwell
                ),
            ));
        }
        if !(0.0..0.5).contains(&p.exit_margin) {
            return Err(invalid(
                "reliability.pressure.exit_margin",
                format!("must be in [0, 0.5), got {}", p.exit_margin),
            ));
        }
        for (signal, levels) in &p.thresholds {
            validate_thresholds(*signal, levels)?;
        }

        let a = &self.admission;
        if a.kv_overcommit.is_some() {
            return Err(invalid(
                "reliability.admission.kv_overcommit",
                "removed: KV overcommit is not offered; admission reserves the worst case",
            ));
        }
        if !(1..=65536).contains(&a.max_queue) {
            return Err(invalid(
                "reliability.admission.max_queue",
                format!("must be between 1 and 65536, got {}", a.max_queue),
            ));
        }
        if !(Duration::from_secs(1)..=Duration::from_secs(3600)).contains(&a.queue_timeout.0) {
            return Err(invalid(
                "reliability.admission.queue_timeout",
                format!("must be between 1s and 1h, got {}", a.queue_timeout),
            ));
        }
        if a.large_prefill_tokens < block_tokens {
            return Err(invalid(
                "reliability.admission.large_prefill_tokens",
                format!(
                    "must be at least kv.block_tokens ({block_tokens}), got {}",
                    a.large_prefill_tokens
                ),
            ));
        }

        if self.recovery.max_retries > 10 {
            return Err(invalid(
                "reliability.recovery.max_retries",
                format!(
                    "must be between 0 and 10, got {}",
                    self.recovery.max_retries
                ),
            ));
        }
        if self.memory.workspace_bytes.0 == 0 {
            return Err(invalid(
                "reliability.memory.workspace_bytes",
                "must be greater than 0",
            ));
        }

        let c = &self.circuit;
        if c.latency_drift_degraded.is_nan() || c.latency_drift_degraded <= 1.0 {
            return Err(invalid(
                "reliability.circuit.latency_drift_degraded",
                format!("must be greater than 1, got {}", c.latency_drift_degraded),
            ));
        }
        if c.latency_drift_open.is_nan() || c.latency_drift_open <= c.latency_drift_degraded {
            return Err(invalid(
                "reliability.circuit.latency_drift_open",
                format!(
                    "must be greater than reliability.circuit.latency_drift_degraded ({}), got {}",
                    c.latency_drift_degraded, c.latency_drift_open
                ),
            ));
        }
        if c.oom_recoveries_to_open == 0 {
            return Err(invalid(
                "reliability.circuit.oom_recoveries_to_open",
                "must be at least 1",
            ));
        }
        if c.probe_successes == 0 {
            return Err(invalid(
                "reliability.circuit.probe_successes",
                "must be at least 1",
            ));
        }

        #[cfg(not(feature = "fault-injection"))]
        if self.fault_injection.is_some() {
            return Err(invalid(
                "reliability.fault_injection",
                "requires a turbine-server built with --features fault-injection",
            ));
        }
        Ok(())
    }
}

/// Non-null values must be finite and strictly ascending (descending for "lower is worse").
fn validate_thresholds(
    signal: PressureSignal,
    levels: &ThresholdLevels,
) -> Result<(), ConfigError> {
    let key = format!("reliability.pressure.thresholds.{}", signal.as_str());
    let set: Vec<f64> = levels.iter().flatten().copied().collect();
    if set.is_empty() || set.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            &key,
            "needs at least one threshold and every threshold must be a finite number",
        ));
    }
    let lower_is_worse = signal.lower_is_worse();
    let monotonic = set.windows(2).all(|w| {
        if lower_is_worse {
            w[1] < w[0]
        } else {
            w[1] > w[0]
        }
    });
    if !monotonic {
        let dir = if lower_is_worse {
            "descending (lower is worse)"
        } else {
            "ascending"
        };
        return Err(invalid(
            &key,
            format!(
                "thresholds for YELLOW, ORANGE, RED, SURVIVAL must be strictly {dir}, got {levels:?}"
            ),
        ));
    }
    Ok(())
}
