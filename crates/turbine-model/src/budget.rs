//! Startup memory budget (P1 S-5): weights + KV reservation (`model.max_seq_len` tokens) +
//! workspace + `reliability.emergency_vram_reserve` must fit in the available memory, checked
//! before any weight byte is read. Pure functions: nothing here touches weight files or devices;
//! the caller supplies the device's free memory and, on unified devices, host `MemAvailable`.
use turbine_core::types::MemoryKind;

use crate::ModelError;

/// The terms of the startup budget, in bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BudgetTerms {
    pub weights: u64,
    pub kv_reservation: u64,
    pub workspace: u64,
    pub emergency_reserve: u64,
    pub available: u64,
    /// The model's per-sequence state ([`turbine_core::types::StateLayout::bytes_per_sequence`],
    /// P7 S-1); 0 for attention-only families.
    pub state_bytes_per_sequence: u64,
    /// `scheduler.max_running_requests`: each running sequence holds one state.
    pub max_running_requests: u32,
}

impl BudgetTerms {
    /// The state of every running sequence: `max_running_requests × state_bytes_per_sequence`
    /// (saturating).
    pub fn state_reservation(&self) -> u64 {
        u64::from(self.max_running_requests).saturating_mul(self.state_bytes_per_sequence)
    }

    /// Everything the model needs (saturating: an overflow can never fit).
    pub fn required(&self) -> u64 {
        self.weights
            .saturating_add(self.kv_reservation)
            .saturating_add(self.state_reservation())
            .saturating_add(self.workspace)
            .saturating_add(self.emergency_reserve)
    }
}

/// Memory the model may use: the device's free memory on a `dedicated` device; on a device that
/// shares host memory (`unified`), the smaller of that and host `MemAvailable`, falling back to
/// the device figure when the host figure is unknown.
pub fn available_bytes(kind: MemoryKind, device_free: u64, host_mem_available: Option<u64>) -> u64 {
    match kind {
        MemoryKind::Dedicated => device_free,
        // Unified, and any later kind: host memory can bound it, so take the minimum.
        _ => host_mem_available.map_or(device_free, |host| host.min(device_free)),
    }
}

/// Refuses a budget whose terms exceed `available`, listing every term. Logs `memory_budget`
/// (INFO) with every term either way.
pub fn check_budget(terms: &BudgetTerms) -> Result<(), ModelError> {
    let required = terms.required();
    let fits = required <= terms.available;
    let state = terms.state_reservation();
    tracing::info!(
        event = "memory_budget",
        weights = terms.weights,
        kv_reservation = terms.kv_reservation,
        state_reservation = state,
        workspace = terms.workspace,
        emergency_reserve = terms.emergency_reserve,
        required,
        available_bytes = terms.available,
        fits,
    );
    if fits {
        return Ok(());
    }
    // The state term is named only when the model keeps state (the P1 message is unchanged).
    let state = if state == 0 {
        String::new()
    } else {
        format!(
            " + state {} × {} B",
            terms.max_running_requests, terms.state_bytes_per_sequence
        )
    };
    Err(ModelError::Budget(format!(
        "weights {} B + kv_reservation {} B{state} + workspace {} B + emergency_reserve {} B = \
         {required} B > available {} B",
        terms.weights,
        terms.kv_reservation,
        terms.workspace,
        terms.emergency_reserve,
        terms.available
    )))
}

/// Host `MemAvailable` (moved to `turbine_device` in Phase 2m Task 4, re-exported here).
pub use turbine_device::host_mem_available;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::types::{DeviceId, MemoryKind, StateDtype, StateLayout};
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::ModelError;
    use crate::families::llama_slots;
    use crate::loader::WeightLoader;
    use crate::safetensors::SafetensorsIndex;
    use crate::testing::TempDir;
    use crate::testing::tiny::write_tiny_llama;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn refuses_before_loading() {
        let dir = TempDir::new("budget-tiny");
        let spec = write_tiny_llama(dir.path(), 11);
        let index = SafetensorsIndex::open(dir.path()).unwrap();

        // Truncate the weights file to its header: any weight read would now fail.
        let file = dir.path().join("model.safetensors");
        let bytes = std::fs::read(&file).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        std::fs::write(&file, &bytes[..8 + header_len as usize]).unwrap();

        let terms = BudgetTerms {
            weights: index.total_bytes(),
            kv_reservation: spec.config.kv_layout(1).bytes_per_token() * 512,
            workspace: MIB,
            emergency_reserve: MIB,
            available: 1024,
            state_bytes_per_sequence: spec.config.state.bytes_per_sequence(),
            max_running_requests: 4,
        };
        let err = check_budget(&terms).expect_err("budget must be refused");
        let text = err.to_string();
        let sum = terms.weights + terms.kv_reservation + terms.workspace + terms.emergency_reserve;
        let expected = format!(
            "weights {} B + kv_reservation {} B + workspace {} B + emergency_reserve {} B = {sum} B \
             > available 1024 B",
            terms.weights, terms.kv_reservation, terms.workspace, terms.emergency_reserve
        );
        match err {
            ModelError::Budget(msg) => assert_eq!(msg, expected),
            other => panic!("expected Budget, got {other:?}"),
        }
        assert!(text.starts_with("memory budget exceeded: "), "{text}");

        // The budget never touched the weights: the truncated file still fails on a real load.
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), GIB);
        let load = WeightLoader::load(&index, &llama_slots(&spec.config), &mem, 1 << 20);
        assert!(
            matches!(load, Err(ModelError::Io { ref path, .. }) if *path == file),
            "{load:?}"
        );

        // Exactly fitting is allowed.
        let fits = BudgetTerms {
            available: sum,
            ..terms
        };
        check_budget(&fits).unwrap();
    }

    /// A model with per-sequence state (P7 S-1) is charged `max_running_requests` states beside
    /// its KV: KV + state for 4 running sequences over the budget is refused, naming the term.
    #[test]
    fn charges_state_per_running_sequence() {
        let state = StateLayout {
            linear_layers: 3,
            recurrent_bytes_per_layer: 1 << 20,
            conv_bytes_per_layer: 4096,
            dtype: Some(StateDtype::F32),
        };
        let per_seq = state.bytes_per_sequence();
        assert_eq!(per_seq, 3 * (MIB + 4096));
        let terms = BudgetTerms {
            weights: 64 * MIB,
            kv_reservation: 16 * MIB,
            workspace: MIB,
            emergency_reserve: MIB,
            available: 0,
            state_bytes_per_sequence: per_seq,
            max_running_requests: 4,
        };
        let without_state = 64 * MIB + 16 * MIB + MIB + MIB;
        assert_eq!(terms.state_reservation(), 4 * per_seq);
        assert_eq!(terms.required(), without_state + 4 * per_seq);

        // Room for the weights, KV and three states: refused.
        let short = BudgetTerms {
            available: without_state + 3 * per_seq,
            ..terms
        };
        match check_budget(&short) {
            Err(ModelError::Budget(msg)) => {
                assert!(msg.contains(&format!("+ state 4 × {per_seq} B")), "{msg}")
            }
            other => panic!("expected Budget, got {other:?}"),
        }
        // The same terms without state fit.
        check_budget(&BudgetTerms {
            state_bytes_per_sequence: 0,
            ..short
        })
        .unwrap();
        // Room for all four states fits.
        check_budget(&BudgetTerms {
            available: terms.required(),
            ..terms
        })
        .unwrap();
    }

    #[test]
    fn available_memory_by_kind() {
        let free = 10 * GIB;
        assert_eq!(
            available_bytes(MemoryKind::Dedicated, free, Some(5 * GIB)),
            10 * GIB
        );
        assert_eq!(available_bytes(MemoryKind::Dedicated, free, None), 10 * GIB);
        assert_eq!(
            available_bytes(MemoryKind::Unified, free, Some(5 * GIB)),
            5 * GIB
        );
        assert_eq!(
            available_bytes(MemoryKind::Unified, free, Some(20 * GIB)),
            10 * GIB
        );
        assert_eq!(available_bytes(MemoryKind::Unified, free, None), 10 * GIB);

        let dir = TempDir::new("budget-meminfo");
        let meminfo = dir.path().join("meminfo");
        std::fs::write(
            &meminfo,
            "MemTotal:       131072000 kB\nMemFree:         2048000 kB\n\
             MemAvailable:   126877932 kB\nBuffers:          123456 kB\n",
        )
        .unwrap();
        // 126 877 932 kB × 1024 (the plan text says 129922002368, an arithmetic slip).
        assert_eq!(host_mem_available(&meminfo), Some(129_923_002_368));
        std::fs::write(&meminfo, "MemTotal:       131072000 kB\n").unwrap();
        assert_eq!(host_mem_available(&meminfo), None);
        assert_eq!(host_mem_available(&dir.path().join("absent")), None);
    }
}
