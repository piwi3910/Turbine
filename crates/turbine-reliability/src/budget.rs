//! Per-device memory budget split into pools (P3 S-2; P5 S-8 adds the `collective` pool:
//! communicator buffers, measured as the drop in free device memory across communicator init).

use serde::Serialize;
use turbine_core::config::{ByteSize, ReliabilityConfig};
use turbine_core::types::{DeviceId, MemoryKind};

/// The pools a device budget is split into.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PoolKind {
    Weights,
    Kv,
    Workspace,
    /// Collective communicator buffers (P5); 0 without tensor parallelism.
    Collective,
    Runtime,
    Reserve,
}

impl PoolKind {
    pub const ALL: [PoolKind; 6] = [
        PoolKind::Weights,
        PoolKind::Kv,
        PoolKind::Workspace,
        PoolKind::Collective,
        PoolKind::Runtime,
        PoolKind::Reserve,
    ];

    /// Metric label / JSON name.
    pub fn as_str(self) -> &'static str {
        match self {
            PoolKind::Weights => "weights",
            PoolKind::Kv => "kv",
            PoolKind::Workspace => "workspace",
            PoolKind::Collective => "collective",
            PoolKind::Runtime => "runtime",
            PoolKind::Reserve => "reserve",
        }
    }
}

/// Metric label / JSON name of a memory kind (`dedicated` / `unified`).
pub fn memory_kind_str(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Dedicated => "dedicated",
        MemoryKind::Unified => "unified",
        _ => "other",
    }
}

/// What startup measured, after weights load.
#[derive(Clone, Debug)]
pub struct BudgetInputs {
    pub device: DeviceId,
    pub memory_kind: MemoryKind,
    /// Dedicated devices: device memory measured free. Ignored on unified devices.
    pub measured_free_bytes: Option<u64>,
    /// Bytes Turbine already held on the device when `measured_free_bytes` was taken (weights).
    pub already_held_bytes: u64,
    /// Host `MemAvailable` at startup: the budget source on unified devices.
    pub host_mem_available_bytes: Option<u64>,
    pub weights_bytes: u64,
    /// From the model's KV layout (`KvLayout::bytes_per_token`), never assumed dimensions.
    pub kv_bytes_per_token: u64,
    pub max_seq_len: u32,
    pub block_bytes: u64,
    /// Communicator buffers on this device (P5): the drop in free device memory across
    /// communicator init, 0 without a communicator.
    pub collective_bytes: u64,
}

/// A device budget and its pools; the pools never exceed the budget.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeviceBudget {
    pub device: DeviceId,
    pub memory_kind: MemoryKind,
    pub budget_bytes: u64,
    pub pools: Vec<(PoolKind, u64)>,
}

impl DeviceBudget {
    /// Capacity of `kind` (0 when absent).
    pub fn pool(&self, kind: PoolKind) -> u64 {
        self.pools
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or(0, |(_, b)| *b)
    }

    /// `device_memory` is computed on dedicated devices only (P3 Constraints).
    pub fn tracks_device_memory(&self) -> bool {
        self.memory_kind == MemoryKind::Dedicated
    }
}

/// The budget cannot hold the model (startup exit 1); `breakdown` names every pool and its bytes.
#[derive(Debug, thiserror::Error)]
#[error("memory budget cannot hold the model: {breakdown}")]
pub struct BudgetError {
    pub breakdown: String,
}

const GIB: f64 = (1u64 << 30) as f64;

fn show(bytes: u64) -> String {
    format!("{bytes} bytes ({:.2} GiB)", bytes as f64 / GIB)
}

/// Dedicated: budget = measured free + already held; unified: `MemAvailable` −
/// `host_reserve_bytes`; both capped by `reliability.memory.device_budget_bytes`. The `kv` pool
/// is budget − weights − workspace − collective − runtime − reserve, then capped by `kv_cap`
/// (`kv.gpu.max_bytes`, CONFLICT C-8); it must hold one full-context sequence.
pub fn compute_budget(
    inp: &BudgetInputs,
    cfg: &ReliabilityConfig,
    kv_cap: Option<ByteSize>,
) -> Result<DeviceBudget, BudgetError> {
    let device = inp.device.0;
    let (source, raw) = match inp.memory_kind {
        MemoryKind::Dedicated => {
            let free = inp.measured_free_bytes.ok_or_else(|| BudgetError {
                breakdown: format!("device {device}: measured free device memory unavailable"),
            })?;
            (
                format!(
                    "measured_free={} + already_held={}",
                    show(free),
                    show(inp.already_held_bytes)
                ),
                free.saturating_add(inp.already_held_bytes),
            )
        }
        MemoryKind::Unified => {
            let avail = inp.host_mem_available_bytes.ok_or_else(|| BudgetError {
                breakdown: format!(
                    "device {device}: unified memory but host MemAvailable unavailable"
                ),
            })?;
            let reserve = cfg.memory.host_reserve_bytes.0;
            (
                format!(
                    "MemAvailable={} - host_reserve_bytes={}",
                    show(avail),
                    show(reserve)
                ),
                avail.saturating_sub(reserve),
            )
        }
        other => {
            return Err(BudgetError {
                breakdown: format!("device {device}: unsupported memory kind {other:?}"),
            });
        }
    };
    let budget = cfg
        .memory
        .device_budget_bytes
        .map_or(raw, |cap| raw.min(cap.0));
    let workspace = cfg.memory.workspace_bytes.0;
    let runtime = cfg.memory.runtime_overhead_bytes.0;
    let reserve = cfg.emergency_vram_reserve.0;
    let collective = inp.collective_bytes;
    let fixed = inp
        .weights_bytes
        .saturating_add(workspace)
        .saturating_add(collective)
        .saturating_add(runtime)
        .saturating_add(reserve);
    let block = inp.block_bytes.max(1);
    let seq_bytes = u64::from(inp.max_seq_len).saturating_mul(inp.kv_bytes_per_token);
    let min_kv = seq_bytes.div_ceil(block).saturating_mul(block);
    let remainder = budget.saturating_sub(fixed);
    let kv = kv_cap.map_or(remainder, |cap| remainder.min(cap.0));
    if reserve >= budget || fixed > budget || kv < min_kv {
        let cap = cfg
            .memory
            .device_budget_bytes
            .map(|c| format!(", capped by device_budget_bytes={}", show(c.0)))
            .unwrap_or_default();
        let kv_capped = kv_cap
            .filter(|c| c.0 < remainder)
            .map(|c| format!(", capped by kv.gpu.max_bytes={}", show(c.0)))
            .unwrap_or_default();
        return Err(BudgetError {
            breakdown: format!(
                "device {device} {}: budget={} ({source}{cap}); weights={}; workspace={}; collective={}; runtime={}; reserve={}; kv={}{kv_capped} (minimum {} for one {}-token sequence)",
                memory_kind_str(inp.memory_kind),
                show(budget),
                show(inp.weights_bytes),
                show(workspace),
                show(collective),
                show(runtime),
                show(reserve),
                show(kv),
                show(min_kv),
                inp.max_seq_len,
            ),
        });
    }
    Ok(DeviceBudget {
        device: inp.device,
        memory_kind: inp.memory_kind,
        budget_bytes: budget,
        pools: vec![
            (PoolKind::Weights, inp.weights_bytes),
            (PoolKind::Kv, kv),
            (PoolKind::Workspace, workspace),
            (PoolKind::Collective, collective),
            (PoolKind::Runtime, runtime),
            (PoolKind::Reserve, reserve),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::active_signals;
    use turbine_core::types::PressureSignal;

    const GIB: u64 = 1 << 30;
    /// Llama-3.2-3B BF16: 28 layers × 2 × 8 KV heads × 128 × 2 bytes.
    const PER_TOKEN: u64 = 114_688;

    fn inputs(kind: MemoryKind, free: Option<u64>, avail: Option<u64>) -> BudgetInputs {
        BudgetInputs {
            device: DeviceId(0),
            memory_kind: kind,
            measured_free_bytes: free,
            already_held_bytes: 0,
            host_mem_available_bytes: avail,
            weights_bytes: 6 * GIB,
            kv_bytes_per_token: PER_TOKEN,
            max_seq_len: 32_768,
            block_bytes: PER_TOKEN * 16,
            collective_bytes: 0,
        }
    }

    fn cfg_with_cap(cap: Option<u64>) -> ReliabilityConfig {
        let mut cfg = ReliabilityConfig::default();
        cfg.memory.device_budget_bytes = cap.map(ByteSize);
        cfg
    }

    fn pool_sum(b: &DeviceBudget) -> u64 {
        b.pools.iter().map(|(_, v)| v).sum()
    }

    #[test]
    fn pools_partition_free_memory() {
        let b = compute_budget(
            &inputs(MemoryKind::Dedicated, Some(30 * GIB), None),
            &cfg_with_cap(None),
            None,
        )
        .unwrap();
        assert_eq!(b.budget_bytes, 30 * GIB);
        assert_eq!(b.pool(PoolKind::Weights), 6 * GIB);
        assert_eq!(b.pool(PoolKind::Workspace), GIB);
        assert_eq!(b.pool(PoolKind::Runtime), GIB);
        assert_eq!(b.pool(PoolKind::Reserve), 2 * GIB);
        assert_eq!(b.pool(PoolKind::Kv), 20 * GIB);
        assert_eq!(b.pool(PoolKind::Collective), 0);
        assert_eq!(pool_sum(&b), b.budget_bytes, "pools partition the budget");

        // P5: communicator buffers come out of the kv pool and still partition the budget.
        let mut tp = inputs(MemoryKind::Dedicated, Some(30 * GIB), None);
        tp.collective_bytes = GIB / 2;
        let with_collective = compute_budget(&tp, &cfg_with_cap(None), None).unwrap();
        assert_eq!(with_collective.pool(PoolKind::Collective), GIB / 2);
        assert_eq!(with_collective.pool(PoolKind::Kv), 20 * GIB - GIB / 2);
        assert_eq!(pool_sum(&with_collective), with_collective.budget_bytes);

        // Measured after weights load: 24 GiB free + 6 GiB already held is the same budget.
        let mut after = inputs(MemoryKind::Dedicated, Some(24 * GIB), None);
        after.already_held_bytes = 6 * GIB;
        assert_eq!(
            compute_budget(&after, &cfg_with_cap(None), None).unwrap(),
            b
        );

        // device_budget_bytes caps the budget.
        let capped = compute_budget(
            &inputs(MemoryKind::Dedicated, Some(30 * GIB), None),
            &cfg_with_cap(Some(20 * GIB)),
            None,
        )
        .unwrap();
        assert_eq!(capped.budget_bytes, 20 * GIB);
        assert_eq!(capped.pool(PoolKind::Kv), 10 * GIB);
        assert_eq!(pool_sum(&capped), 20 * GIB);

        // kv.gpu.max_bytes caps only the kv pool (CONFLICT C-8).
        let kv_capped = compute_budget(
            &inputs(MemoryKind::Dedicated, Some(30 * GIB), None),
            &cfg_with_cap(None),
            Some(ByteSize::gib(4)),
        )
        .unwrap();
        assert_eq!(kv_capped.budget_bytes, 30 * GIB);
        assert_eq!(kv_capped.pool(PoolKind::Kv), 4 * GIB);
        assert!(pool_sum(&kv_capped) <= 30 * GIB);
    }

    #[test]
    fn unified_budget_from_mem_available() {
        // The measured device figure is never an input on unified memory.
        let unified = inputs(MemoryKind::Unified, Some(100 * GIB), Some(42 * GIB));
        let b = compute_budget(&unified, &cfg_with_cap(None), None).unwrap();
        assert_eq!(
            b.budget_bytes,
            34 * GIB,
            "MemAvailable 42 GiB - host reserve 8 GiB"
        );
        assert_eq!(pool_sum(&b), b.budget_bytes);
        assert_eq!(
            compute_budget(&unified, &cfg_with_cap(Some(24 * GIB)), None)
                .unwrap()
                .budget_bytes,
            24 * GIB
        );
        assert_eq!(
            compute_budget(&unified, &cfg_with_cap(Some(64 * GIB)), None)
                .unwrap()
                .budget_bytes,
            34 * GIB,
            "a larger cap never raises the budget"
        );
        assert!(!b.tracks_device_memory());
        assert!(!active_signals(b.memory_kind).contains(&PressureSignal::DeviceMemory));
        assert!(active_signals(MemoryKind::Dedicated).contains(&PressureSignal::DeviceMemory));
    }

    #[test]
    fn impossible_budget_rejected() {
        // 10 GiB free: 6 + 1 + 1 + 2 leaves no KV, but one 32,768-token sequence needs 3.5 GiB.
        let err = compute_budget(
            &inputs(MemoryKind::Dedicated, Some(10 * GIB), None),
            &cfg_with_cap(None),
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        for part in [
            "budget=",
            "weights=",
            "workspace=",
            "collective=",
            "runtime=",
            "reserve=",
            "kv=",
            "measured_free=",
            "32768-token",
        ] {
            assert!(msg.contains(part), "breakdown names {part}: {msg}");
        }
        // 13 GiB leaves 3 GiB of KV: still below one full-context sequence (3.5 GiB).
        let err13 = compute_budget(
            &inputs(MemoryKind::Dedicated, Some(13 * GIB), None),
            &cfg_with_cap(None),
            None,
        );
        assert!(err13.is_err());
        let ok = compute_budget(
            &inputs(MemoryKind::Dedicated, Some(14 * GIB), None),
            &cfg_with_cap(None),
            None,
        )
        .unwrap();
        assert_eq!(ok.pool(PoolKind::Kv), 4 * GIB);
        // A missing measurement is an error too, never a guess.
        assert!(
            compute_budget(
                &inputs(MemoryKind::Dedicated, None, None),
                &cfg_with_cap(None),
                None
            )
            .is_err()
        );
    }
}
