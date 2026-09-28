//! Lab only (novanas R9700, P4 S-5/S-6/S-17, CONFLICT C-6): kernel C ABI v2.3 + v2.5 pinned host
//! memory and copy streams through `libturbine_hip.so`. Run by `scripts/lab-test.sh novanas`,
//! which sets `TURBINE_TEST_BACKEND=hip` and `TURBINE_KERNEL_LIBRARY`.
use std::sync::Arc;
use std::time::Instant;

use turbine_kernels::test_support::{open_context, require_backend};
use turbine_tensor::{CopyEngine, CopyTarget, DeviceBuffer, DeviceMemory, PinnedMemory};

/// One Llama-3.2-3B BF16 KV block (28 layers × 8 KV heads × 128 × 16 tokens × K+V × 2 bytes).
const BLOCK: usize = 1_835_008;
const BLOCKS: usize = 1000;
const SLAB: usize = 1 << 30;

/// xorshift64*: a reproducible fill that differs per block.
fn fill(seed: u64, out: &mut [u8]) {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for chunk in out.chunks_mut(8) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let v = x.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes();
        chunk.copy_from_slice(&v[..chunk.len()]);
    }
}

#[test]
#[ignore = "needs a HIP device and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn pinned_round_trip() {
    if !require_backend("hip") {
        return;
    }
    let ctx = open_context("hip");
    let lib = ctx.library();
    assert_eq!(lib.abi_version(), 2);
    assert!(
        lib.abi_minor() >= 5,
        "minor {} lacks the v2.5 copy streams",
        lib.abi_minor()
    );
    assert!(ctx.has_copy_engine(), "the v2.3 and v2.5 groups resolve");
    let mem: Arc<dyn DeviceMemory> = ctx.clone();

    // Two 1 GiB slabs of page-locked host memory: blocks go down into the first and come back
    // from the second.
    let down = ctx.alloc_pinned(SLAB).expect("first pinned slab");
    let up = ctx.alloc_pinned(SLAB).expect("second pinned slab");
    let per_slab = SLAB / BLOCK;
    let mut gpu = DeviceBuffer::alloc(&mem, BLOCKS * BLOCK).expect("GPU blocks");
    let mut block = vec![0u8; BLOCK];
    for b in 0..BLOCKS {
        fill(b as u64, &mut block);
        gpu.copy_from_host(b * BLOCK, &block)
            .expect("seed GPU block");
        // The host buffer is reused: the enqueued copy must finish first.
        mem.synchronize().expect("seeded");
    }

    // Device → pinned in rounds of one slab; each ticket is complete only once polled true.
    let (mut d2h_bytes, mut d2h_secs) = (0usize, 0.0f64);
    let (mut h2d_bytes, mut h2d_secs) = (0usize, 0.0f64);
    let back = DeviceBuffer::alloc(&mem, BLOCKS * BLOCK).expect("GPU landing blocks");
    for round in 0..BLOCKS.div_ceil(per_slab) {
        let first = round * per_slab;
        let n = per_slab.min(BLOCKS - first);
        let started = Instant::now();
        let tickets: Vec<_> = (0..n)
            .map(|i| {
                let b = first + i;
                ctx.copy_async(
                    CopyTarget::Pinned {
                        buffer_id: down.id(),
                        offset: i * BLOCK,
                    },
                    CopyTarget::Device(gpu.ptr().offset((b * BLOCK) as u64)),
                    BLOCK,
                )
                .expect("d2h copy")
            })
            .collect();
        for t in &tickets {
            ctx.wait(t).expect("d2h wait");
            assert!(ctx.poll(t).expect("poll"), "a waited ticket is complete");
        }
        d2h_secs += started.elapsed().as_secs_f64();
        d2h_bytes += n * BLOCK;

        // The pinned copy is byte-identical; move it to the second slab and back up.
        down.with_bytes(|src| {
            up.with_bytes_mut(|dst| dst[..n * BLOCK].copy_from_slice(&src[..n * BLOCK]))
        });
        let started = Instant::now();
        let tickets: Vec<_> = (0..n)
            .map(|i| {
                let b = first + i;
                ctx.copy_async(
                    CopyTarget::Device(back.ptr().offset((b * BLOCK) as u64)),
                    CopyTarget::Pinned {
                        buffer_id: up.id(),
                        offset: i * BLOCK,
                    },
                    BLOCK,
                )
                .expect("h2d copy")
            })
            .collect();
        // A ticket never reports complete before its event: poll until each signals.
        for t in &tickets {
            while !ctx.poll(t).expect("h2d poll") {
                std::hint::spin_loop();
            }
        }
        h2d_secs += started.elapsed().as_secs_f64();
        h2d_bytes += n * BLOCK;
    }
    for b in 0..BLOCKS {
        fill(b as u64, &mut block);
        let mut got = vec![0u8; BLOCK];
        back.copy_to_host(b * BLOCK, &mut got).expect("read back");
        mem.synchronize().expect("read back sync");
        assert!(
            got == block,
            "block {b} differs after the pinned round trip"
        );
    }
    println!(
        "pinned_round_trip d2h_gbps={:.2} h2d_gbps={:.2}",
        d2h_bytes as f64 / d2h_secs / 1e9,
        h2d_bytes as f64 / h2d_secs / 1e9
    );
}

/// Engine-thread cost of demoting Llama-3.2-3B blocks of the serving default (128 tokens,
/// 14.7 MB) to pinned memory the way the KV orchestrator does: one `copy_async` per layer
/// (28 per block, each with its compute-stream fence and completion event), then the per-turn
/// `poll` sweep over every in-flight ticket until all are done. Reports the host time to
/// enqueue a block, the host time of one `poll`, and the same enqueue as one copy per block
/// for comparison (overload soak investigation, provisional decision "Phase 4: pressure
/// reclaim copies only blocks with reuse evidence, bounded in flight").
#[test]
#[ignore = "needs a HIP device and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn demotion_host_cost() {
    const LAYERS: usize = 28;
    const LAYER_BYTES: usize = 524_288;
    const BLOCK_128: usize = LAYERS * LAYER_BYTES;
    const N: usize = 32;
    if !require_backend("hip") {
        return;
    }
    let ctx = open_context("hip");
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let host = ctx.alloc_pinned(N * BLOCK_128).expect("pinned slab");
    let gpu = DeviceBuffer::alloc(&mem, N * BLOCK_128).expect("GPU blocks");
    mem.synchronize().expect("allocated");

    let per_layer = |round: usize| {
        let started = Instant::now();
        let mut tickets = Vec::with_capacity(N * LAYERS);
        for b in 0..N {
            for l in 0..LAYERS {
                let off = b * BLOCK_128 + l * LAYER_BYTES;
                tickets.push(
                    ctx.copy_async(
                        CopyTarget::Pinned {
                            buffer_id: host.id(),
                            offset: off,
                        },
                        CopyTarget::Device(gpu.ptr().offset(off as u64)),
                        LAYER_BYTES,
                    )
                    .expect("d2h copy"),
                );
            }
        }
        let enqueue = started.elapsed().as_secs_f64();
        let (mut polls, mut poll_secs) = (0u64, 0.0f64);
        let mut pending = tickets;
        while !pending.is_empty() {
            let sweep = Instant::now();
            let n = pending.len() as u64;
            pending.retain(|t| !ctx.poll(t).expect("poll"));
            poll_secs += sweep.elapsed().as_secs_f64();
            polls += n;
        }
        let total = started.elapsed().as_secs_f64();
        println!(
            "demotion_host_cost round={round} per_layer_copies enqueue_us_per_block={:.0} \
             poll_us={:.2} polls={polls} d2h_gbps={:.2}",
            enqueue / N as f64 * 1e6,
            poll_secs / polls as f64 * 1e6,
            (N * BLOCK_128) as f64 / total / 1e9
        );
    };
    per_layer(0);
    per_layer(1);

    let started = Instant::now();
    let tickets: Vec<_> = (0..N)
        .map(|b| {
            ctx.copy_async(
                CopyTarget::Pinned {
                    buffer_id: host.id(),
                    offset: b * BLOCK_128,
                },
                CopyTarget::Device(gpu.ptr().offset((b * BLOCK_128) as u64)),
                BLOCK_128,
            )
            .expect("d2h copy")
        })
        .collect();
    let enqueue = started.elapsed().as_secs_f64();
    for t in &tickets {
        ctx.wait(t).expect("wait");
    }
    println!(
        "demotion_host_cost one_copy_per_block enqueue_us_per_block={:.0} d2h_gbps={:.2}",
        enqueue / N as f64 * 1e6,
        (N * BLOCK_128) as f64 / started.elapsed().as_secs_f64() / 1e9
    );
}

/// P5 S-13: the startup host-link probe on every visible GPU of the backend's vendor, printed as
/// `host_link device=<i> h2d_gbps=… d2h_gbps=…`; each direction must measure > 1 GB/s. With
/// `TURBINE_EXPECT_AMD=2` (a `--gpus 2` run on novanas) GPU0's slot (Gen5 x8) must measure no
/// more than 10 % below GPU1's (Gen4 x8) host-to-device. A 64 MiB pinned copy measures about the
/// same on both, below either link's rate (2026-09-28: GPU0 13.08 / 12.74 GB/s, GPU1 12.47 /
/// 12.29 GB/s h2d / d2h), so a strict order would flake. Breaks if the probe fails on a real device
/// or GPU0 measures clearly slower than GPU1.
#[test]
#[ignore = "needs a HIP device and libturbine_hip.so (scripts/lab-test.sh novanas)"]
fn host_link_probe() {
    use turbine_device::topology::{LinkProbe, PROBE_BYTES};
    use turbine_kernels::backends::{self, BackendRequest};
    use turbine_kernels::link_probe::{CopyLinkProbe, ProbeTarget};

    if !require_backend("hip") {
        return;
    }
    let backend = backends::registry().get("hip").expect("hip backend");
    let inventory =
        turbine_device::discover(&turbine_device::DiscoveryOptions::default()).expect("discovery");
    let devices: Vec<_> = inventory
        .devices
        .iter()
        .filter(|d| d.vendor.as_str() == backend.vendor())
        .map(|d| d.index)
        .collect();
    assert!(!devices.is_empty(), "a visible HIP device");
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY").map(std::path::PathBuf::from);
    let mut probe = CopyLinkProbe::new(|device| {
        let opened = backend
            .open(&BackendRequest {
                device,
                kernel_library: library.as_deref(),
                inventory: &inventory,
                meminfo: std::path::Path::new("/proc/meminfo"),
                card_profile: "auto",
            })
            .map_err(|e| e.to_string())?;
        let ctx = opened.context.ok_or("no kernel-library context")?;
        Ok(ProbeTarget {
            memory: opened.mem,
            copies: ctx.clone(),
            pinned: ctx,
        })
    });
    let mut h2d = Vec::new();
    for &d in &devices {
        let bw = probe.host_link(d, PROBE_BYTES).expect("probe");
        println!(
            "host_link device={} h2d_gbps={:.2} d2h_gbps={:.2}",
            d.0, bw.h2d_gbps, bw.d2h_gbps
        );
        assert!(
            bw.h2d_gbps > 1.0 && bw.d2h_gbps > 1.0,
            "device {}: {bw:?}",
            d.0
        );
        h2d.push(bw.h2d_gbps);
    }
    let expect_amd = std::env::var("TURBINE_EXPECT_AMD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if expect_amd == Some(2) {
        assert_eq!(h2d.len(), 2, "two GPUs visible");
        assert!(
            h2d[0] >= 0.9 * h2d[1],
            "GPU0 (Gen5 x8) {:.2} GB/s is more than 10 % below GPU1 (Gen4 x8) {:.2} GB/s",
            h2d[0],
            h2d[1]
        );
    }
}
