use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::{Clock, FakeClock};
use turbine_core::types::PressureState;

use super::*;
use crate::identity::{KvKey, NamespaceKey};
use crate::metrics::{EvictReason, KvMetrics};
use crate::test_log;
use turbine_core::types::MemoryKind;
use turbine_tensor::host::HostPinned;

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
    assert_eq!(&raw[8..12], &2u32.to_le_bytes(), "format version 2");
    assert_eq!(
        &raw[60..76],
        b"l0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        "the slots' codec"
    );
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

/// An L1 tier of 2-block slabs over the host fake allocator.
fn l1_cfg(max_slabs: u64, memory_kind: MemoryKind) -> L1Config {
    L1Config {
        enabled: true,
        max_bytes: max_slabs * 2 * BLOCK,
        slab_bytes: 2 * BLOCK,
        block_bytes: BLOCK,
        memory_kind,
    }
}

#[test]
fn l1_grows_and_shrinks() {
    let alloc = Arc::new(HostPinned::new(u64::MAX));
    let l1 = L1PinnedTier::new(l1_cfg(3, MemoryKind::Dedicated), alloc.clone(), clock());
    assert!(l1.enabled());
    assert_eq!(l1.id(), TierId::L1);
    assert_eq!(l1.capacity_bytes(), 6 * BLOCK);
    assert_eq!(alloc.live_buffers(), 0, "no preallocation");
    assert_eq!(l1.slab_count(), 0);

    for i in 0..3u8 {
        l1.put(key(i), TierBlockRef::Host(&bytes(i))).unwrap();
    }
    assert_eq!(l1.slab_count(), 2, "slabs allocated on demand");
    assert_eq!(alloc.allocated_bytes(), 4 * BLOCK);
    // Fills the free slot of slab 1 without allocating.
    l1.put(key(3), TierBlockRef::Host(&bytes(3))).unwrap();
    assert_eq!(l1.slab_count(), 2);

    alloc.fail_next(1);
    let (res, logs) = test_log::capture(|| l1.put(key(9), TierBlockRef::Host(&bytes(9))));
    assert_eq!(
        res,
        Err(TierError::Full),
        "failed slab allocation leaves L1 at its size"
    );
    assert!(logs.contains("kv_l1_slab_alloc_failed"), "{logs}");
    assert!(logs.contains("WARN"), "{logs}");
    assert_eq!(l1.slab_count(), 2);
    let mut out = vec![0u8; BLOCK as usize];
    for i in 0..4u8 {
        l1.get(&key(i), TierBlockMut::Host(&mut out)).unwrap();
        assert_eq!(
            out,
            bytes(i),
            "L1 stays usable after a failed slab allocation"
        );
    }

    l1.put(key(9), TierBlockRef::Host(&bytes(9))).unwrap();
    assert_eq!(l1.slab_count(), 3, "grows again after the failure");
    l1.put(key(10), TierBlockRef::Host(&bytes(10))).unwrap();
    assert_eq!(
        l1.put(key(11), TierBlockRef::Host(&bytes(11))),
        Err(TierError::Full),
        "never beyond kv.cpu.max_bytes"
    );
    assert_eq!(l1.slab_count(), 3);
    assert_eq!(alloc.live_buffers(), 3);
    assert_eq!(l1.used_bytes(), 6 * BLOCK);

    // Slab 0 holds blocks 0 and 1; emptying it keeps it while host pressure is below RED.
    for i in [0u8, 1] {
        l1.evict(&key(i)).unwrap();
    }
    l1.set_host_pressure(PressureState::Orange);
    assert_eq!(l1.slab_count(), 3, "empty slabs kept below RED");
    l1.set_host_pressure(PressureState::Red);
    assert_eq!(l1.slab_count(), 2, "the empty slab is released at host RED");
    assert_eq!(alloc.live_buffers(), 2);
    assert_eq!(
        l1.pressure(),
        PressureState::Red,
        "host pressure bounds the tier's pressure"
    );
    assert_eq!(
        l1.put(key(12), TierBlockRef::Host(&bytes(12))),
        Err(TierError::Full),
        "no slab is allocated at host RED"
    );
    l1.set_host_pressure(PressureState::Green);
    l1.put(key(12), TierBlockRef::Host(&bytes(12))).unwrap();
    assert_eq!(l1.slab_count(), 3, "grows again once host pressure falls");
    for i in [2u8, 3, 9, 10, 12] {
        l1.get(&key(i), TierBlockMut::Host(&mut out)).unwrap();
        assert_eq!(out, bytes(i));
    }
    assert_eq!(l1.used_bytes(), 5 * BLOCK);

    // A reserved slot is written by the copy stream before it becomes visible.
    let (buffer_id, offset) = l1.reserve(key(13)).unwrap();
    assert!(!l1.contains(&key(13)), "a reservation is invisible");
    assert_eq!(l1.locate(&key(13)), None);
    assert_eq!(
        l1.used_bytes(),
        6 * BLOCK,
        "a reservation occupies its slot"
    );
    assert_eq!(l1.reserve(key(14)), Err(TierError::Full));
    let slot = l1.commit(&key(13));
    assert!(l1.contains(&key(13)));
    assert_eq!(l1.locate(&key(13)), Some((buffer_id, offset)));
    assert_eq!(offset % BLOCK as usize, 0);
    assert_eq!(
        slot.0 & 0xffff_ffff,
        (offset / BLOCK as usize) as u64,
        "slot encodes slab << 32 | slot"
    );
    l1.evict(&key(13)).unwrap();
    l1.reserve(key(14)).unwrap();
    assert!(
        l1.abort_reservation(&key(14)),
        "a failed copy frees its slot"
    );
    assert!(!l1.abort_reservation(&key(14)));
    assert!(!l1.contains(&key(14)));
    assert_eq!(l1.used_bytes(), 5 * BLOCK);

    // Three copy errors within 60 s degrade the tier.
    assert!(!l1.record_copy_error());
    assert!(!l1.record_copy_error());
    assert!(l1.record_copy_error());
    assert!(l1.degraded());
    assert_eq!(
        l1.get(&key(3), TierBlockMut::Host(&mut out)),
        Err(TierError::Degraded)
    );

    let (tier, logs) = test_log::capture(|| {
        L1PinnedTier::new(l1_cfg(3, MemoryKind::Unified), alloc.clone(), clock())
    });
    assert!(!tier.enabled());
    assert_eq!(tier.capacity_bytes(), 0);
    assert_eq!(
        logs.matches("kv_l1_disabled_unified").count(),
        1,
        "exactly one WARN: {logs}"
    );
    assert!(logs.contains("WARN"), "{logs}");
    assert_eq!(
        tier.put(key(1), TierBlockRef::Host(&bytes(1))),
        Err(TierError::Full)
    );
    assert_eq!(
        alloc.live_buffers(),
        3,
        "a unified device never allocates L1"
    );
    assert_eq!(
        demotion_target(TierId::L0, tier.enabled(), true),
        Some(TierId::L2),
        "L0 demotes straight to L2"
    );
    assert_eq!(
        demotion_target(TierId::L0, tier.enabled(), false),
        None,
        "or drops without L2"
    );
}

/// P6b S-6: at its size limit L1 gives an empty slab of another slot size to a block of a new
/// size (a ladder rewrite into another format), as L2 reformats an empty slab; a slab that still
/// holds a block keeps its size. Breaks if a rewrite into a new format is refused while a slab
/// is empty (the ladder could then never step an L1 of few slabs past its second format).
#[test]
fn l1_reuses_an_empty_slab_for_another_slot_size() {
    let alloc = Arc::new(HostPinned::new(u64::MAX));
    let l1 = L1PinnedTier::new(l1_cfg(2, MemoryKind::Dedicated), alloc.clone(), clock());
    let half = (BLOCK / 2) as usize;
    let quarter = (BLOCK / 4) as usize;
    l1.put(key(0), TierBlockRef::Host(&bytes(0))).unwrap();
    l1.put(key(1), TierBlockRef::Host(&bytes(1)[..half]))
        .unwrap();
    assert_eq!(l1.slab_count(), 2);
    // Key 0 is rewritten at half size: it moves to slab 1, slab 0 is empty.
    l1.put(key(0), TierBlockRef::Host(&bytes(0)[..half]))
        .unwrap();
    // A quarter-size block takes the empty slab.
    l1.put(key(1), TierBlockRef::Host(&bytes(1)[..quarter]))
        .unwrap();
    assert_eq!(l1.slab_count(), 2);
    assert_eq!(alloc.live_buffers(), 2, "no slab beyond the limit");
    let mut out = vec![0u8; quarter];
    l1.get(&key(1), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, bytes(1)[..quarter]);
    let mut out = vec![0u8; half];
    l1.get(&key(0), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, bytes(0)[..half]);
    // Both slabs hold blocks now: a third size is refused.
    let eighth = (BLOCK / 8) as usize;
    assert_eq!(
        l1.put(key(2), TierBlockRef::Host(&bytes(2)[..eighth])),
        Err(TierError::Full)
    );
}

#[test]
fn l1_passes_the_contract_suite() {
    let alloc = Arc::new(HostPinned::new(u64::MAX));
    let l1 = L1PinnedTier::new(l1_cfg(2, MemoryKind::Dedicated), alloc, clock());
    suite(&l1, 4);
}

/// Two rank shards of `BLOCK / 2` bytes each, 2-slot slabs, `max_slabs` slabs per shard, each
/// shard over its own allocator (a rank's own kernel-library context).
fn sharded_l1(max_slabs: u64) -> (ShardedL1Tier, [Arc<HostPinned>; 2]) {
    let half = BLOCK / 2;
    let allocs = [
        Arc::new(HostPinned::new(u64::MAX)),
        Arc::new(HostPinned::new(u64::MAX)),
    ];
    let shards = allocs
        .iter()
        .map(|a| {
            Arc::new(L1PinnedTier::new(
                L1Config {
                    enabled: true,
                    max_bytes: max_slabs * 2 * half,
                    slab_bytes: 2 * half,
                    block_bytes: half,
                    memory_kind: MemoryKind::Dedicated,
                },
                a.clone() as Arc<dyn turbine_tensor::PinnedMemory>,
                clock(),
            ))
        })
        .collect();
    (ShardedL1Tier::new(shards, half), allocs)
}

#[test]
fn sharded_l1_passes_the_contract_suite() {
    let (l1, allocs) = sharded_l1(2);
    assert_eq!(l1.capacity_bytes(), 4 * BLOCK, "capacity sums the shards");
    suite(&l1, 4);
    assert!(
        allocs.iter().all(|a| a.live_buffers() == 2),
        "each shard pins its slabs through its own allocator"
    );
}

#[test]
fn sharded_l1_splits_blocks_by_rank() {
    let (l1, _allocs) = sharded_l1(2);
    let half = (BLOCK / 2) as usize;
    let block = bytes(7);
    l1.put(key(1), TierBlockRef::Host(&block)).unwrap();
    for (rank, shard) in l1.shards().iter().enumerate() {
        let mut out = vec![0u8; half];
        shard.get(&key(1), TierBlockMut::Host(&mut out)).unwrap();
        assert_eq!(
            out,
            block[rank * half..(rank + 1) * half],
            "rank {rank} holds its slice"
        );
    }

    // A block one shard lacks is not stored.
    l1.shards()[1].evict(&key(1)).unwrap();
    assert!(!l1.contains(&key(1)));
    let mut out = vec![0u8; BLOCK as usize];
    assert_eq!(
        l1.get(&key(1), TierBlockMut::Host(&mut out)),
        Err(TierError::Missing)
    );
    assert_eq!(
        l1.evict(&key(1)),
        Ok(()),
        "evicts the shard that still held it"
    );
    assert_eq!(l1.evict(&key(1)), Err(TierError::Missing));

    // Reservations are all-or-nothing and commit / abort on every shard.
    let slots = l1.reserve(key(2)).unwrap();
    assert_eq!(slots.len(), 2);
    assert!(l1.locate(&key(2)).is_none(), "invisible until committed");
    l1.commit(&key(2));
    assert_eq!(l1.locate(&key(2)).unwrap(), slots);
    l1.reserve(key(3)).unwrap();
    assert!(l1.abort_reservation(&key(3)));
    assert_eq!(
        l1.used_bytes(),
        BLOCK,
        "the aborted reservation is freed on both shards"
    );

    // A put that fails on one shard leaves no partial copy on the others.
    let (full, _a) = sharded_l1(1);
    full.shards()[1]
        .put(key(8), TierBlockRef::Host(&vec![0u8; half]))
        .unwrap();
    full.shards()[1]
        .put(key(9), TierBlockRef::Host(&vec![0u8; half]))
        .unwrap();
    assert_eq!(
        full.put(key(1), TierBlockRef::Host(&block)),
        Err(TierError::Full)
    );
    assert!(!full.shards()[0].contains(&key(1)));
    assert_eq!(full.shards()[0].used_bytes(), 0);
    // A reservation that fails on one shard frees the other's.
    assert_eq!(full.reserve(key(1)), Err(TierError::Full));
    assert_eq!(full.shards()[0].used_bytes(), 0);
}

/// Phase 5 S-10: a pipeline's stage shards differ in size (a quarter and three quarters of a
/// block here): each shard stores its own part, in order, and a whole block round-trips; a
/// block of the wrong length is refused. Breaks if the split assumes equal shards.
#[test]
fn sharded_l1_with_uneven_shards() {
    let sizes = [BLOCK / 4, BLOCK - BLOCK / 4];
    let shards = sizes
        .iter()
        .map(|&n| {
            Arc::new(L1PinnedTier::new(
                L1Config {
                    enabled: true,
                    max_bytes: 4 * n,
                    slab_bytes: 2 * n,
                    block_bytes: n,
                    memory_kind: MemoryKind::Dedicated,
                },
                Arc::new(HostPinned::new(u64::MAX)) as Arc<dyn turbine_tensor::PinnedMemory>,
                clock(),
            ))
        })
        .collect();
    let l1 = ShardedL1Tier::with_sizes(shards, sizes.to_vec());
    let block = bytes(5);
    l1.put(key(1), TierBlockRef::Host(&block)).unwrap();
    let mut start = 0usize;
    for (shard, &n) in l1.shards().iter().zip(&sizes) {
        let mut out = vec![0u8; n as usize];
        shard.get(&key(1), TierBlockMut::Host(&mut out)).unwrap();
        assert_eq!(out, block[start..start + n as usize]);
        start += n as usize;
    }
    let mut out = vec![0u8; BLOCK as usize];
    l1.get(&key(1), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, block);
    assert!(l1.put(key(2), TierBlockRef::Host(&block[1..])).is_err());
}

/// A tier whose reads keep failing degrades, and every attach that needed it ends ready with
/// reason `tier_degraded` (recompute), never failed or lost.
#[test]
fn faulty_tier_degrades_to_recompute() {
    use crate::hierarchy::AttachOutcome;
    use crate::hierarchy::tests::{Rig, attach, prefill_and_finish, rig};
    use crate::planner::PlanReason;
    use turbine_core::types::RequestId;

    let fake = FakeClock::new(Duration::ZERO);
    let arc: Arc<dyn Clock> = Arc::new(fake.clone());
    let bb = crate::directory::tests::fmt16().layout.block_bytes();
    let mem = Arc::new(MemTier::new(TierId::L1, 64 * bb, arc));
    let mut r = rig(64, Some(mem.clone()), None, fake);
    let tick = |r: &mut Rig| {
        let mut ready = Vec::new();
        for _ in 0..3 {
            ready.extend(r.h.poll(&mut r.pool, &mut r.backend));
            r.clock.advance(Duration::from_millis(5));
        }
        ready
    };
    let prompt = |p: u32| -> Vec<u32> { (p * 1000..p * 1000 + 33).collect() };
    // Four prompts of 2 full blocks + 1 token, each sent twice (the hit is the reuse evidence
    // demotion needs), then demoted to L1.
    for p in (0..4).chain(0..4) {
        let id = RequestId::new_v4();
        let AttachOutcome::Ready(a) = attach(&mut r, id, &prompt(p)) else {
            panic!("an L0-resident or cold prompt attaches at once");
        };
        prefill_and_finish(&mut r, id, &prompt(p), &a);
    }
    r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
    tick(&mut r);
    assert_eq!(mem.len(), 8);
    assert_eq!(r.pool.used_blocks(), 0);

    // Three read errors within 60 s: promotions fail, the tier degrades, requests recompute.
    mem.inject_read_errors(3);
    let mut reasons = Vec::new();
    for p in 0..4 {
        let id = RequestId::new_v4();
        let a = match attach(&mut r, id, &prompt(p)) {
            AttachOutcome::Ready(a) => a,
            AttachOutcome::Promoting => {
                let (_, a) = tick(&mut r)
                    .into_iter()
                    .find(|(rid, _)| *rid == id)
                    .expect("a request is never failed or lost");
                a
            }
            AttachOutcome::WaitForPrefix => panic!("no concurrent prefix"),
        };
        reasons.push(a.plan.reason);
        r.pool.release(&a.blocks);
        r.h.request_done(&mut r.pool, id, false);
    }
    assert!(mem.degraded(), "3 errors in 60 s mark the tier degraded");
    assert!(
        reasons.iter().all(|r| *r == PlanReason::TierDegraded),
        "{reasons:?}"
    );
    assert_eq!(r.pool.referenced_blocks(), 0, "no reference leaked");
}

/// P6b S-1: L2 keeps each codec's blocks in slab files of their own (header version 2 names
/// the codec and slot size), accounts a block at its own bytes, and rewrites a slab that no
/// block uses for another codec once every slab file exists.
#[test]
fn nvme_slabs_per_codec() {
    let dir = tempfile::tempdir().unwrap();
    let t = l2(dir.path(), 4, KvMetrics::unregistered());
    for i in 0..4u8 {
        t.put(key(i), TierBlockRef::Host(&bytes(i))).unwrap();
    }
    assert_eq!(
        t.put_as(key(9), "fp8_e4m3", 3000, TierBlockRef::Host(&[1u8; 3000])),
        Err(TierError::Full),
        "the only slab file holds l0 blocks"
    );
    for i in 0..4u8 {
        t.evict(&key(i)).unwrap();
    }
    // The emptied slab takes 4 KiB fp8 slots now: twice as many blocks.
    for i in 0..8u8 {
        let block = vec![i; 3000];
        t.put_as(key(i), "fp8_e4m3", 3000, TierBlockRef::Host(&block))
            .unwrap();
    }
    assert_eq!(t.used_bytes(), 8 * 3000);
    let raw = std::fs::read(t.slab_path(0)).unwrap();
    assert_eq!(&raw[44..52], &4096u64.to_le_bytes(), "slot size");
    assert_eq!(&raw[52..60], &8u64.to_le_bytes(), "slot count");
    assert_eq!(&raw[60..68], b"fp8_e4m3");
    let mut out = vec![0u8; 3000];
    t.get(&key(5), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, vec![5u8; 3000]);
    assert_eq!(
        t.put(key(20), TierBlockRef::Host(&bytes(20))),
        Err(TierError::Full),
        "no slab is free for an l0 block"
    );
}

/// An L2 tier on `dir` with `slabs` slab files of `blocks` `l0` blocks each.
fn l2_slabs(dir: &std::path::Path, blocks: u64, slabs: u64) -> L2NvmeTier {
    let cfg = L2Config {
        path: dir.to_path_buf(),
        max_bytes: slabs * (4096 + blocks * 8192),
        slab_bytes: blocks * 8192,
        max_queue_depth: 8,
        block_bytes: BLOCK,
        namespace: NamespaceKey([3; 32]),
    };
    L2NvmeTier::open(cfg, clock(), KvMetrics::unregistered()).unwrap()
}

/// P6b S-6 edge case (user decision "6b Task 16: ladder proof results — four open points", 2 A):
/// a rewrite of a block into another codec finds its new slot before it frees the old one, so a
/// rewrite that finds no room (`Full`) leaves the old copy stored, readable and accounted. Breaks
/// if the store frees the replaced slot (or drops its index entry) before it has the new one.
#[test]
fn nvme_rewrite_without_room_keeps_the_copy() {
    let dir = tempfile::tempdir().unwrap();
    let t = l2(dir.path(), 4, KvMetrics::unregistered());
    t.put(key(0), TierBlockRef::Host(&bytes(0))).unwrap();
    t.put(key(1), TierBlockRef::Host(&bytes(1))).unwrap();
    assert_eq!(
        t.put_as(key(0), "fp8_e4m3", 3000, TierBlockRef::Host(&[7u8; 3000])),
        Err(TierError::Full),
        "the only slab file holds l0 blocks"
    );
    assert!(t.contains(&key(0)), "the old copy stays");
    let mut out = vec![0u8; BLOCK as usize];
    t.get(&key(0), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, bytes(0), "at its old format");
    assert_eq!(t.used_bytes(), 2 * BLOCK);
    // Its slot is still taken: two more l0 blocks fill the slab, a fifth finds none.
    t.put(key(2), TierBlockRef::Host(&bytes(2))).unwrap();
    t.put(key(3), TierBlockRef::Host(&bytes(3))).unwrap();
    assert_eq!(
        t.put(key(4), TierBlockRef::Host(&bytes(4))),
        Err(TierError::Full)
    );
}

/// The other half of [`nvme_rewrite_without_room_keeps_the_copy`]: a rewrite that finds room
/// stores the new copy and only then frees the old slot, whose emptied slab takes `l0` blocks
/// again. Breaks if the old slot leaks or the old copy stays indexed.
#[test]
fn nvme_rewrite_with_room_frees_the_old_slot() {
    let dir = tempfile::tempdir().unwrap();
    let t = l2_slabs(dir.path(), 4, 2);
    t.put(key(0), TierBlockRef::Host(&bytes(0))).unwrap();
    t.put_as(key(0), "fp8_e4m3", 3000, TierBlockRef::Host(&[7u8; 3000]))
        .unwrap();
    assert_eq!(t.used_bytes(), 3000);
    let mut out = vec![0u8; 3000];
    t.get(&key(0), TierBlockMut::Host(&mut out)).unwrap();
    assert_eq!(out, vec![7u8; 3000]);
    // Slab 0 is empty again: it holds four l0 blocks.
    for i in 1..5u8 {
        t.put(key(i), TierBlockRef::Host(&bytes(i))).unwrap();
    }
    assert_eq!(
        t.put(key(5), TierBlockRef::Host(&bytes(5))),
        Err(TierError::Full)
    );
}

/// `room_epoch` (P6b S-6, user decision "6b Task 16: ladder proof results — four open points",
/// 3 A): L1 changes it when a slab empties or is released and when host pressure drops below RED,
/// never for a freed slot of a slab that still holds blocks. Breaks if the ladder's back-off
/// could resume while no slab freed, or wait forever after one did.
#[test]
fn l1_room_epoch_changes_when_a_slab_frees() {
    let alloc = Arc::new(HostPinned::new(u64::MAX));
    let l1 = L1PinnedTier::new(l1_cfg(2, MemoryKind::Dedicated), alloc, clock());
    let e0 = l1.room_epoch();
    for i in 0..3u8 {
        l1.put(key(i), TierBlockRef::Host(&bytes(i))).unwrap();
    }
    // Slab 0 holds keys 0 and 1, slab 1 key 2.
    l1.evict(&key(0)).unwrap();
    assert_eq!(l1.room_epoch(), e0, "slab 0 still holds key 1");
    l1.evict(&key(1)).unwrap();
    let e1 = l1.room_epoch();
    assert_ne!(e1, e0, "slab 0 emptied");
    l1.set_host_pressure(PressureState::Red);
    let e2 = l1.room_epoch();
    assert_ne!(e2, e1, "the empty slab is released");
    l1.set_host_pressure(PressureState::Red);
    assert_eq!(l1.room_epoch(), e2, "nothing more to release");
    l1.set_host_pressure(PressureState::Orange);
    assert_ne!(l1.room_epoch(), e2, "slabs may be allocated again");
}

/// L2's `room_epoch` changes when a slab's last slot frees, not for a slot of a slab still in
/// use (see [`l1_room_epoch_changes_when_a_slab_frees`]).
#[test]
fn nvme_room_epoch_changes_when_a_slab_frees() {
    let dir = tempfile::tempdir().unwrap();
    let t = l2(dir.path(), 4, KvMetrics::unregistered());
    let e0 = t.room_epoch();
    t.put(key(0), TierBlockRef::Host(&bytes(0))).unwrap();
    t.put(key(1), TierBlockRef::Host(&bytes(1))).unwrap();
    t.evict(&key(0)).unwrap();
    assert_eq!(t.room_epoch(), e0, "the slab still holds key 1");
    t.evict(&key(1)).unwrap();
    assert_ne!(t.room_epoch(), e0, "the slab emptied");
}
