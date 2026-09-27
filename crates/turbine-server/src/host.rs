//! Host facts for the Phase 4 `kv` startup rules (contract §3.2 `HostFacts`, P4 §Configuration):
//! the host's total memory (`MemTotal` through Phase 3's `turbine_device::telemetry::proc`
//! parser; `sysctl -n hw.memsize` on macOS) and the free space of the filesystem holding
//! `kv.nvme.path` (`proc::disk_free_bytes`). `Config::validate_host` checks them against
//! `reliability.memory.host_reserve_bytes`.

use std::process::Command;

use turbine_core::config::{HostFacts, KvConfig};
use turbine_device::telemetry::proc::{disk_free_bytes, parse_meminfo};

/// Total host memory: `MemTotal` on Linux, `sysctl -n hw.memsize` on macOS; `None` when
/// neither answers (the rule is then skipped).
pub fn mem_total_bytes() -> Option<u64> {
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        return parse_meminfo(&text).ok()?.mem_total_bytes;
    }
    let out = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// The facts `Config::validate_host` needs: free disk only when L2 is enabled. A fact that
/// cannot be read is `None` (logged), so its rule is skipped rather than failing startup.
pub fn facts(kv: &KvConfig) -> HostFacts {
    let disk_free_bytes = if kv.nvme.enabled {
        match disk_free_bytes(&kv.nvme.path) {
            Ok(free) => Some(free),
            Err(e) => {
                tracing::warn!(path = %kv.nvme.path.display(), error = %e, "cannot read the free space of kv.nvme.path");
                None
            }
        }
    } else {
        None
    };
    HostFacts {
        mem_total_bytes: mem_total_bytes(),
        disk_free_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Breaks if the host memory is not read (the `kv.cpu.max_bytes` rule would be skipped).
    #[test]
    fn host_memory_is_read() {
        assert!(mem_total_bytes().is_some_and(|b| b > 0));
        let facts = facts(&KvConfig::default());
        assert!(facts.mem_total_bytes.is_some());
        assert_eq!(facts.disk_free_bytes, None, "L2 disabled: no disk rule");
    }
}
