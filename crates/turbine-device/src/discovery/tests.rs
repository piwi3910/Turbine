use std::path::PathBuf;
use std::time::{Duration, Instant};

use nvml_wrapper::error::NvmlError;
use turbine_core::types::{MemoryKind, Vendor};

use super::*;
use crate::inventory::BackendStatus;

/// A library name no loader search can satisfy, so the test behaves the same on lab hosts that
/// do have NVML / amd-smi installed.
fn missing_default(kind: &dyn DiscoveryKind) -> PathBuf {
    PathBuf::from(format!("libturbine-test-missing-{}.so", kind.name()))
}

#[test]
fn no_libraries_means_empty_inventory() {
    let inv = discover_with_defaults(&DiscoveryOptions::default(), missing_default)
        .expect("default search must never be fatal");
    assert!(inv.devices.is_empty());
    assert_eq!(inv.backends.len(), 2);
    assert_eq!(inv.backends[0].vendor, Vendor::Nvidia);
    assert_eq!(inv.backends[1].vendor, Vendor::Amd);
    for b in &inv.backends {
        assert_eq!(b.status, BackendStatus::Unavailable, "{b:?}");
        assert!(!b.detail.is_empty());
    }
}

#[test]
fn explicit_missing_library_is_fatal() {
    let opts = DiscoveryOptions {
        amd_smi_library: Some(PathBuf::from("/nonexistent/libamd_smi.so")),
        ..DiscoveryOptions::default()
    };
    let err = discover(&opts).expect_err("an explicit library that cannot load must be fatal");
    assert!(
        err.to_string().contains("/nonexistent/libamd_smi.so"),
        "{err}"
    );

    let opts = DiscoveryOptions {
        nvml_library: Some(PathBuf::from("/nonexistent/libnvidia-ml.so.1")),
        ..DiscoveryOptions::default()
    };
    let err = discover(&opts).expect_err("explicit NVML path must be fatal too");
    assert!(
        err.to_string().contains("/nonexistent/libnvidia-ml.so.1"),
        "{err}"
    );
}

struct Sleeper;

impl DiscoveryBackend for Sleeper {
    fn vendor(&self) -> Vendor {
        Vendor::Amd
    }
    fn discover(&mut self) -> Result<Vec<DeviceInfo>, String> {
        std::thread::sleep(Duration::from_secs(30));
        Ok(Vec::new())
    }
}

struct Instant1;

impl DiscoveryBackend for Instant1 {
    fn vendor(&self) -> Vendor {
        Vendor::Nvidia
    }
    fn discover(&mut self) -> Result<Vec<DeviceInfo>, String> {
        Ok(vec![DeviceInfo {
            index: DeviceId(99),
            vendor: Vendor::Nvidia,
            vendor_index: 0,
            name: "fake".into(),
            uuid: None,
            pci_bus_id: None,
            arch: None,
            driver_version: None,
            memory: nvml::memory_info(Ok(1), std::path::Path::new("/nonexistent")),
        }])
    }
}

#[test]
fn backend_timeout() {
    let deadline = DiscoveryOptions::default().deadline;
    assert_eq!(deadline, Duration::from_secs(10));
    let started = Instant::now();
    let inv = run_backends(vec![Box::new(Instant1), Box::new(Sleeper)], deadline);
    assert!(
        started.elapsed() < Duration::from_secs(11),
        "took {:?}",
        started.elapsed()
    );
    assert_eq!(inv.backends[0].status, BackendStatus::Ok);
    assert_eq!(inv.backends[1].vendor, Vendor::Amd);
    assert_eq!(inv.backends[1].status, BackendStatus::Timeout);
    assert_eq!(inv.devices.len(), 1);
    assert_eq!(
        inv.devices[0].index,
        DeviceId(0),
        "global index is reassigned"
    );
}

#[test]
fn unified_memory_uses_host_total() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/proc/meminfo-gb10");
    let mem = nvml::memory_info(Err(NvmlError::NotSupported), &fixture);
    assert_eq!(mem.kind, MemoryKind::Unified);
    assert_eq!(mem.total_bytes, 129_923_002_368);
    assert!(mem.shared_with_host);

    let dedicated = nvml::memory_info(Ok(34_208_743_424), &fixture);
    assert_eq!(dedicated.kind, MemoryKind::Dedicated);
    assert!(!dedicated.shared_with_host);

    let json = serde_json::to_value(&mem).unwrap();
    assert_eq!(json["kind"], "unified");
    assert_eq!(nvml::normalize_bus_id("00000000:0F:01.0"), "0000:0f:01.0");
}

#[test]
fn registry_lists_nvml_then_amd_smi() {
    let reg = registry();
    assert_eq!(reg.point(), "device_discovery");
    assert_eq!(reg.names(), ["nvml", "amd_smi"]);
    let vendors: Vec<Vendor> = reg.iter().map(|k| k.vendor()).collect();
    assert_eq!(vendors, [Vendor::Nvidia, Vendor::Amd]);
    let defaults: Vec<&str> = reg.iter().map(|k| k.default_library()).collect();
    assert_eq!(defaults, ["libnvidia-ml.so.1", "libamd_smi.so"]);
    let opts = DiscoveryOptions {
        amd_smi_library: Some(PathBuf::from("/x/libamd_smi.so")),
        ..DiscoveryOptions::default()
    };
    let configured: Vec<Option<&std::path::Path>> =
        reg.iter().map(|k| k.configured_library(&opts)).collect();
    assert_eq!(
        configured,
        [None, Some(std::path::Path::new("/x/libamd_smi.so"))]
    );
}
