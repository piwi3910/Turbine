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
    assert_eq!((lib.abi_version(), lib.abi_minor()), (2, 5));
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
