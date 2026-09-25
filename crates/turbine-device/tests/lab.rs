//! Lab-only checks against real GPUs. Run by `scripts/lab-test.sh <host>` with `--include-ignored`.
//!
//! Expectations come from the environment and each one is skipped when unset, so the test also
//! passes on a machine without GPUs: `TURBINE_EXPECT_NVIDIA`, `TURBINE_EXPECT_AMD` (device counts),
//! `TURBINE_EXPECT_NVIDIA_MEMORY` (`unified` | `dedicated`), `TURBINE_EXPECT_AMD_ARCH` (`gfx…`).

use turbine_core::config::DevicesConfig;
use turbine_core::types::{MemoryKind, Vendor};
use turbine_device::{DiscoveryOptions, discover};

fn expected(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

fn expected_count(var: &str) -> Option<usize> {
    expected(var).map(|v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{var} must be a device count, got {v:?}"))
    })
}

/// The serialized name (`dedicated` / `unified`), so a future kind is never mislabelled.
fn kind_name(kind: MemoryKind) -> String {
    match serde_json::to_value(kind).expect("MemoryKind serializes") {
        serde_json::Value::String(s) => s,
        other => panic!("MemoryKind serialized as {other}"),
    }
}

#[test]
#[ignore = "needs lab GPUs; run via scripts/lab-test.sh"]
fn inventory_matches_expectation() {
    let inv =
        discover(&DiscoveryOptions::from_config(&DevicesConfig::default())).expect("discovery");
    println!(
        "{}",
        serde_json::to_string_pretty(&inv).expect("inventory serializes")
    );
    for d in &inv.devices {
        println!(
            "lab-inventory: index={} vendor={} name={:?} arch={} memory.kind={} total_bytes={}",
            d.index.0,
            d.vendor.as_str(),
            d.name,
            d.arch.as_deref().unwrap_or("null"),
            kind_name(d.memory.kind),
            d.memory.total_bytes
        );
    }

    if let Some(n) = expected_count("TURBINE_EXPECT_NVIDIA") {
        assert_eq!(
            inv.count(Vendor::Nvidia),
            n,
            "NVIDIA device count; backends: {:?}",
            inv.backends
        );
    }
    if let Some(kind) = expected("TURBINE_EXPECT_NVIDIA_MEMORY") {
        let want = match kind.as_str() {
            "unified" => MemoryKind::Unified,
            "dedicated" => MemoryKind::Dedicated,
            other => {
                panic!("TURBINE_EXPECT_NVIDIA_MEMORY must be unified or dedicated, got {other:?}")
            }
        };
        for d in inv.devices.iter().filter(|d| d.vendor == Vendor::Nvidia) {
            assert_eq!(d.memory.kind, want, "{d:?}");
            assert!(d.memory.total_bytes > 0, "{d:?}");
            assert_eq!(
                d.memory.shared_with_host,
                want == MemoryKind::Unified,
                "{d:?}"
            );
        }
    }
    if let Some(n) = expected_count("TURBINE_EXPECT_AMD") {
        assert_eq!(
            inv.count(Vendor::Amd),
            n,
            "AMD device count; backends: {:?}",
            inv.backends
        );
    }
    if let Some(arch) = expected("TURBINE_EXPECT_AMD_ARCH") {
        for d in inv.devices.iter().filter(|d| d.vendor == Vendor::Amd) {
            assert_eq!(d.arch.as_deref(), Some(arch.as_str()), "{d:?}");
            assert_eq!(d.memory.kind, MemoryKind::Dedicated, "{d:?}");
            assert!(d.memory.total_bytes > 0, "{d:?}");
        }
    }
}
