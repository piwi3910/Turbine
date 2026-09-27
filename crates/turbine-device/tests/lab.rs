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
    let opts = DiscoveryOptions::from_config(&DevicesConfig::default());
    let inv = discover(&opts).expect("discovery");
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

/// Discovery is called from several threads of one process (parallel tests, a server probing
/// devices while a test discovers): every concurrent call must see the full inventory.
#[test]
#[ignore = "needs lab GPUs; run via scripts/lab-test.sh"]
fn concurrent_discovery_sees_every_device() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 4;
    let Some(want) = expected_count("TURBINE_EXPECT_AMD") else {
        return;
    };
    let opts = DiscoveryOptions::from_config(&DevicesConfig::default());
    for round in 0..ROUNDS {
        let barrier = std::sync::Barrier::new(THREADS);
        let seen: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        discover(&opts).expect("discovery")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("discovery thread"))
                .collect()
        });
        for (thread, inv) in seen.iter().enumerate() {
            println!(
                "lab-concurrent-discovery: round={round} thread={thread} amd={} backends={:?}",
                inv.count(Vendor::Amd),
                inv.backends
            );
        }
        for inv in &seen {
            assert_eq!(
                inv.count(Vendor::Amd),
                want,
                "AMD device count under {THREADS} concurrent discoveries; backends: {:?}",
                inv.backends
            );
        }
    }
}

/// No ledger in this test: telemetry reads only the host and the vendor libraries.
struct IdleLedger;

impl turbine_core::telemetry::LedgerProbe for IdleLedger {
    fn kv_utilization(&self) -> f64 {
        0.0
    }
    fn queue_fill(&self) -> f64 {
        0.0
    }
}

/// P3 S-5, S-14: ten vendor ticks of the real sampler on the host's devices (read-only: no GPU
/// memory is allocated). Every device reports a fresh temperature and clock; AMD (dedicated)
/// devices report used and free memory, unified (GB10) devices none; the host has
/// `MemAvailable`. Fails on a host without GPUs.
#[test]
#[ignore = "needs lab GPUs; run via scripts/lab-test.sh"]
fn live_telemetry() {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use turbine_core::clock::SystemClock;
    use turbine_core::telemetry::SourceStatus;
    use turbine_device::telemetry::proc::FsProc;
    use turbine_device::telemetry::vendor::vendor_backends;
    use turbine_device::telemetry::{SamplerCore, TelemetryConfig};

    let opts = DiscoveryOptions::from_config(&DevicesConfig::default());
    let inv = discover(&opts).expect("discovery");
    assert!(
        !inv.devices.is_empty(),
        "no GPU on this host; backends: {:?}",
        inv.backends
    );
    let mut core = SamplerCore::new(
        TelemetryConfig::default(),
        &inv,
        vendor_backends(&opts),
        Box::new(FsProc::default()),
        Arc::new(IdleLedger),
        Arc::new(SystemClock::new()),
    );
    let started = Instant::now();
    while core.vendor_ticks() < 10 {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "10 vendor ticks took over a minute"
        );
        let wait = core.poll();
        std::thread::sleep(wait.clamp(Duration::from_millis(1), Duration::from_millis(50)));
    }
    let sample = core.latest().load();
    assert!(
        sample.host.mem_available_bytes.is_some_and(|b| b > 0),
        "host: {:?}",
        sample.host
    );
    assert_eq!(sample.devices.len(), inv.devices.len());
    for (d, s) in inv.devices.iter().zip(&sample.devices) {
        println!(
            "live-telemetry: device={} vendor={} status={:?} temperature_c={:?} clock_mhz={:?} \
             used={:?} free={:?} power_w={:?} util={:?}",
            d.index.0,
            d.vendor.as_str(),
            s.status,
            s.temperature_c,
            s.clock_mhz,
            s.memory_used_bytes,
            s.memory_free_bytes,
            s.power_watts,
            s.utilization
        );
        assert_eq!(s.device, d.index);
        assert_eq!(s.status, SourceStatus::Ok, "device {}: {s:?}", d.index.0);
        assert!(s.temperature_c.is_some(), "device {}: {s:?}", d.index.0);
        assert!(s.clock_mhz.is_some(), "device {}: {s:?}", d.index.0);
        match d.memory.kind {
            MemoryKind::Unified => {
                assert!(
                    s.memory_used_bytes.is_none() && s.memory_free_bytes.is_none(),
                    "unified device {} reports device memory: {s:?}",
                    d.index.0
                );
            }
            _ if d.vendor == Vendor::Amd => {
                assert!(
                    s.memory_used_bytes.is_some() && s.memory_free_bytes.is_some(),
                    "AMD device {} lacks used/free memory: {s:?}",
                    d.index.0
                );
            }
            _ => {}
        }
    }
}
