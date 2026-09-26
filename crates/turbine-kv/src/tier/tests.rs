use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::{Clock, FakeClock};
use turbine_core::types::PressureState;

use super::*;
use crate::identity::{KvKey, NamespaceKey};
use crate::metrics::{EvictReason, KvMetrics};
use crate::test_log;

/// Deliberately not 4 KiB-aligned: L2 slots round it up to 8 KiB.
const BLOCK: u64 = 6000;

fn clock() -> Arc<dyn Clock> {
    Arc::new(FakeClock::new(Duration::ZERO))
}

fn bytes(seed: u8) -> Vec<u8> {
    (0..BLOCK as usize)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn key(i: u8) -> KvKey {
    KvKey([i; 16])
}

/// An L2 tier on `dir` holding exactly `blocks` blocks in one slab.
fn l2(dir: &std::path::Path, blocks: u64, metrics: KvMetrics) -> L2NvmeTier {
    let cfg = L2Config {
        path: dir.to_path_buf(),
        max_bytes: 4096 + blocks * 8192,
        slab_bytes: blocks * 8192,
        max_queue_depth: 8,
        block_bytes: BLOCK,
        namespace: NamespaceKey([3; 32]),
    };
    L2NvmeTier::open(cfg, clock(), metrics).unwrap()
}

/// The one suite every byte-holding tier must pass.
fn suite(t: &dyn KvTier, capacity_blocks: u8) {
    assert!(t.enabled());
    assert_eq!(t.used_bytes(), 0);
    assert!(!t.contains(&key(1)));
    let mut out = vec![0u8; BLOCK as usize];
    assert_eq!(
        t.get(&key(1), TierBlockMut::Host(&mut out)),
        Err(TierError::Missing)
    );
    assert_eq!(t.evict(&key(1)), Err(TierError::Missing));
    for i in 0..capacity_blocks {
        t.put(key(i), TierBlockRef::Host(&bytes(i))).unwrap();
    }
    assert_eq!(t.used_bytes(), u64::from(capacity_blocks) * BLOCK);
    assert_eq!(
        t.put(key(200), TierBlockRef::Host(&bytes(200))),
        Err(TierError::Full),
        "capacity enforced"
    );
    for i in 0..capacity_blocks {
        assert!(t.contains(&key(i)));
        t.get(&key(i), TierBlockMut::Host(&mut out)).unwrap();
        assert_eq!(out, bytes(i), "byte-identical round trip of block {i}");
    }
    t.put(key(0), TierBlockRef::Host(&bytes(99))).unwrap();
    t.get(&key(0), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, bytes(99), "put over an existing key replaces it");
    assert_eq!(
        t.used_bytes(),
        u64::from(capacity_blocks) * BLOCK,
        "replacing does not grow used bytes"
    );
    t.evict(&key(0)).unwrap();
    assert!(!t.contains(&key(0)));
    assert_eq!(t.used_bytes(), u64::from(capacity_blocks - 1) * BLOCK);
    t.put(key(201), TierBlockRef::Host(&bytes(201))).unwrap();
    assert!(!t.degraded());
}

#[test]
fn contract_suite() {
    let mem = MemTier::new(TierId::L1, 4 * BLOCK, clock());
    assert_eq!(mem.capacity_bytes(), 4 * BLOCK);
    suite(&mem, 4);
    assert_eq!(
        mem.pressure(),
        PressureState::Survival,
        "a full memory tier"
    );

    let dir = tempfile::tempdir().unwrap();
    let t = l2(dir.path(), 4, KvMetrics::unregistered());
    assert_eq!(t.id(), TierId::L2);
    assert_eq!(t.capacity_bytes(), 4 * BLOCK);
    suite(&t, 4);
    assert_eq!(t.queue_depth(), 0, "every I/O left the queue");
    assert_eq!(
        t.pressure(),
        PressureState::Green,
        "L2 pressure is its queue fill"
    );
    assert!(t.p99_latency() > 0.0, "I/O latency is measured");
}

#[test]
fn nvme_checksum_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("keep.txt"), b"operator file").unwrap();
    std::fs::write(dir.path().join("turbine-kv-9999.slab"), b"stale slab").unwrap();
    let metrics = KvMetrics::unregistered();
    let t = l2(dir.path(), 4, metrics.clone());
    assert!(
        !dir.path().join("turbine-kv-9999.slab").exists(),
        "stale slabs are wiped at startup"
    );
    assert!(dir.path().join("keep.txt").exists());
    for i in 0..3u8 {
        t.put(key(i), TierBlockRef::Host(&bytes(i))).unwrap();
    }
    let slab = t.slab_path(0);
    let mut raw = std::fs::read(&slab).unwrap();
    assert_eq!(&raw[..8], SLAB_MAGIC);
    assert_eq!(&raw[8..12], &1u32.to_le_bytes(), "format version 1");
    assert_eq!(&raw[12..44], &[3u8; 32], "namespace key");
    // Slots are 8 KiB after the 4 KiB header and fill from slot 0: corrupt one byte of slot 1.
    raw[4096 + 8192 + 17] ^= 0xff;
    std::fs::write(&slab, &raw).unwrap();

    let mut out = vec![0u8; BLOCK as usize];
    let (res, logs) = test_log::capture(|| t.get(&key(1), TierBlockMut::Host(&mut out)));
    assert_eq!(
        res,
        Err(TierError::Checksum),
        "a corrupted block is never returned"
    );
    assert!(logs.contains("kv_checksum_mismatch"), "{logs}");
    assert!(!t.contains(&key(1)), "the corrupted block is dropped");
    assert_eq!(
        metrics.evictions_value(TierId::L2, EvictReason::Checksum),
        1
    );
    t.get(&key(2), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, bytes(2), "other slots still round-trip");

    drop(t);
    let _restarted = l2(dir.path(), 4, KvMetrics::unregistered());
    assert!(!slab.exists(), "a restart deletes turbine-kv-*.slab");
    assert!(
        dir.path().join("keep.txt").exists(),
        "unrelated files are left alone"
    );
}

#[test]
fn health_window_degrades_and_probes() {
    let s = Duration::from_secs;
    let mut h = TierHealth::new(Some(s(300)));
    assert!(!h.record_error(s(0)));
    assert!(!h.record_error(s(50)));
    assert!(
        !h.record_error(s(61)),
        "the first error left the 60 s window"
    );
    assert!(h.record_error(s(62)), "3 errors within 60 s degrade");
    assert!(h.is_degraded());
    assert!(!h.probe_due(s(361)));
    assert!(h.probe_due(s(362)), "probed 5 minutes after degrading");
    h.probe_result(false, s(362));
    assert!(
        h.is_degraded() && !h.probe_due(s(600)),
        "a failed probe restarts the period"
    );
    h.probe_result(true, s(662));
    assert!(!h.is_degraded());

    // An in-memory tier with injected read errors degrades and then refuses I/O.
    let fake = FakeClock::new(Duration::ZERO);
    let mem = MemTier::new(TierId::L1, 4 * BLOCK, Arc::new(fake.clone()));
    mem.put(key(1), TierBlockRef::Host(&bytes(1))).unwrap();
    mem.inject_read_errors(3);
    let mut out = vec![0u8; BLOCK as usize];
    for _ in 0..3 {
        fake.advance(s(1));
        assert!(matches!(
            mem.get(&key(1), TierBlockMut::Host(&mut out)),
            Err(TierError::Io(_))
        ));
    }
    assert!(mem.degraded());
    assert_eq!(
        mem.get(&key(1), TierBlockMut::Host(&mut out)),
        Err(TierError::Degraded)
    );
}

#[test]
fn pressure_thresholds_and_demotion_targets() {
    let p = |used| utilization_pressure(used, 100);
    assert_eq!(
        [0, 69, 70, 82, 90, 97].map(p),
        [
            PressureState::Green,
            PressureState::Green,
            PressureState::Yellow,
            PressureState::Orange,
            PressureState::Red,
            PressureState::Survival
        ]
    );
    assert_eq!(utilization_pressure(5, 0), PressureState::Green);
    assert_eq!(demotion_target(TierId::L0, true, true), Some(TierId::L1));
    assert_eq!(
        demotion_target(TierId::L0, false, true),
        Some(TierId::L2),
        "unified memory: L1 disabled, L0 demotes straight to L2"
    );
    assert_eq!(demotion_target(TierId::L0, false, false), None, "or drops");
    assert_eq!(demotion_target(TierId::L1, true, true), Some(TierId::L2));
    assert_eq!(demotion_target(TierId::L1, true, false), None);
    assert_eq!(demotion_target(TierId::L2, true, true), None);
}

#[test]
fn nvme_slow_storage_degrades_until_a_probe_passes() {
    let dir = tempfile::tempdir().unwrap();
    let fake = FakeClock::new(Duration::ZERO);
    let metrics = KvMetrics::unregistered();
    let cfg = L2Config {
        path: dir.path().join("kv"),
        max_bytes: 4096 + 4 * 8192,
        slab_bytes: 4 * 8192,
        max_queue_depth: 8,
        block_bytes: BLOCK,
        namespace: NamespaceKey([3; 32]),
    };
    let t = L2NvmeTier::open(cfg, Arc::new(fake.clone()), metrics.clone()).unwrap();
    assert!(dir.path().join("kv").is_dir(), "the path is created");
    assert!(!t.check_slow(), "no calibration, no slow verdict");
    t.put(key(1), TierBlockRef::Host(&bytes(1))).unwrap();
    // A calibration p99 far below any real I/O makes the measured p99 more than 10x slower.
    t.set_calibration(1e-12, 2e9);
    assert_eq!(t.est_bandwidth(), Some(2e9));
    let (slow, logs) = test_log::capture(|| t.check_slow());
    assert!(
        slow && t.degraded(),
        "p99 above 10x calibration degrades L2"
    );
    assert!(logs.contains("kv_tier_degraded"), "{logs}");
    assert!(!t.check_slow(), "degrading is reported once");
    assert_eq!(
        t.put(key(2), TierBlockRef::Host(&bytes(2))),
        Err(TierError::Degraded)
    );
    assert!(!t.probe(), "no probe before 5 minutes");
    fake.advance(Duration::from_secs(300));
    assert!(t.probe(), "one probe write/read decides");
    assert!(!t.degraded() && !t.contains(&KvKey([0xff; 16])));
    t.put(key(2), TierBlockRef::Host(&bytes(2))).unwrap();
    let reg = turbine_observability::MetricsRegistry::new();
    reg.register(
        "turbine_kv_tier_degraded",
        "",
        metrics.tier_degraded.clone(),
    );
    let text = reg.render().unwrap();
    assert!(
        text.contains(r#"turbine_kv_tier_degraded{tier="l2"} 0"#),
        "{text}"
    );
}

/// L0 is an accounting view: capacity, usage and pressure come from the last pool snapshot and
/// `contains` from the hierarchy's residency updates; bytes never move through the trait.
#[test]
fn l0_tier_is_an_accounting_view() {
    use crate::pool::{BlockPool, BlockPoolConfig};
    use turbine_core::types::{DType, DeviceId, KvLayout};
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    let layout = KvLayout {
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
        dtype: DType::BF16,
        block_tokens: 16,
    };
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
    let mut pool = BlockPool::new(
        BlockPoolConfig {
            layout,
            num_blocks: 10,
        },
        mem,
    )
    .unwrap();
    let block = layout.block_bytes();
    let l0 = L0Tier::new(block);
    assert_eq!(l0.id(), TierId::L0);
    assert!(l0.enabled());
    assert_eq!(
        (l0.capacity_bytes(), l0.used_bytes()),
        (0, 0),
        "no snapshot yet"
    );

    let held = pool.allocate(9).unwrap();
    l0.refresh(&pool);
    assert_eq!(l0.capacity_bytes(), 10 * block);
    assert_eq!(l0.used_bytes(), 9 * block);
    assert_eq!(l0.pressure(), PressureState::Red, "90 % used");
    assert_eq!(l0.est_latency(), Duration::ZERO);
    assert_eq!(l0.est_bandwidth(), None);
    assert!(!l0.degraded());

    assert!(!l0.contains(&key(1)));
    l0.set_resident(key(1), true);
    assert!(l0.contains(&key(1)));
    l0.set_resident(key(1), false);
    assert!(!l0.contains(&key(1)));

    let refused = TierError::Io("L0 blocks move through the transfer engine".into());
    assert_eq!(
        l0.put(key(2), TierBlockRef::Host(&[0; 4])),
        Err(refused.clone())
    );
    let mut out = [0u8; 4];
    assert_eq!(
        l0.get(&key(2), TierBlockMut::Host(&mut out)),
        Err(refused.clone())
    );
    assert_eq!(l0.evict(&key(2)), Err(refused));

    pool.release(&held);
    l0.refresh(&pool);
    assert_eq!(l0.used_bytes(), 0);
    assert_eq!(l0.pressure(), PressureState::Green);
}
