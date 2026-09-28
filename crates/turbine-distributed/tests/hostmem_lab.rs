//! Lab only (novanas, both R9700s): the `hostmem` collective backend on real devices through
//! `libturbine_hip.so` (kernel ABI v2.7). Run by
//! `scripts/lab-test.sh novanas --gpus 2 -- -p turbine-distributed --test hostmem_lab`, which sets
//! `TURBINE_TEST_BACKEND=hip` and `TURBINE_KERNEL_LIBRARY`.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use half::bf16;
use turbine_core::clock::SystemClock;
use turbine_distributed::collective::{
    self, Collective, CollectiveError, CollectiveInit, CollectiveLibrary, HostCollective, ReduceOp,
};
use turbine_kernels::backends::BackendRequest;
use turbine_kernels::test_support::require_backend;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DType, DeviceBuffer, DeviceId, DeviceMemory};

/// Aborts the whole test process when the test holding the guard runs longer than `limit`:
/// a hang here would otherwise hold both GPUs of the lab Job until someone deletes it.
struct Watchdog(Arc<std::sync::atomic::AtomicBool>);

fn watchdog(name: &'static str, limit: Duration) -> Watchdog {
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = Arc::clone(&done);
    std::thread::spawn(move || {
        let started = Instant::now();
        while !seen.load(std::sync::atomic::Ordering::Acquire) {
            if started.elapsed() > limit {
                eprintln!("hostmem_lab: {name} ran longer than {limit:?}; aborting the process");
                std::process::abort();
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    Watchdog(done)
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// The device memory (kernel-library context) of every AMD device, in index order.
fn devices() -> Vec<Arc<dyn DeviceMemory>> {
    let inventory = turbine_device::discover(&turbine_device::DiscoveryOptions::default())
        .expect("device discovery");
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let backend = turbine_kernels::backends::registry()
        .get("hip")
        .expect("the hip execution backend");
    inventory
        .devices
        .iter()
        .filter(|d| d.vendor.as_str() == backend.vendor())
        .map(|d| {
            backend
                .open(&BackendRequest {
                    device: d.index,
                    kernel_library: library.as_deref(),
                    inventory: &inventory,
                    meminfo: Path::new("/proc/meminfo"),
                    card_profile: "auto",
                })
                .unwrap_or_else(|e| panic!("open device {}: {e}", d.index.0))
                .mem
        })
        .collect()
}

fn init(
    rank: usize,
    world: usize,
    id: [u8; 128],
    timeout: Duration,
    memory: Arc<dyn DeviceMemory>,
) -> CollectiveInit {
    CollectiveInit {
        rank,
        world,
        unique_id: id,
        init_timeout: timeout,
        op_timeout: timeout,
        clock: Arc::new(SystemClock::new()),
        metrics: None,
        memory: Some(memory),
        // The kernels at every size (no routing to RCCL).
        route_max_bytes: Some(u64::MAX),
    }
}

/// splitmix64 values in [-8, 8) with a fractional part, so sums round.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 16.0 - 8.0
        })
        .collect()
}

fn encode(dtype: DType, v: &[f32]) -> Vec<u8> {
    match dtype {
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// One rank's results: all-reduce sum, all-reduce max, all-gather, reduce-scatter (sum),
/// broadcast from the last rank. Every op is enqueued back to back inside one step and read
/// after one synchronize.
fn ops(
    comm: &dyn Collective,
    mem: &Arc<dyn DeviceMemory>,
    dtype: DType,
    input: &[u8],
    scatter_input: &[u8],
) -> Vec<Vec<u8>> {
    let world = comm.world_size();
    let stream = mem.compute_stream();
    let n = input.len();
    let sum = DeviceBuffer::alloc(mem, n).expect("alloc");
    let max = DeviceBuffer::alloc(mem, n).expect("alloc");
    let send = DeviceBuffer::alloc(mem, n).expect("alloc");
    let gathered = DeviceBuffer::alloc(mem, n * world).expect("alloc");
    let ssend = DeviceBuffer::alloc(mem, n * world).expect("alloc");
    let srecv = DeviceBuffer::alloc(mem, n).expect("alloc");
    let bcast = DeviceBuffer::alloc(mem, n).expect("alloc");
    for b in [&sum, &max, &send, &bcast] {
        b.whole().write_bytes(input).expect("write");
    }
    ssend.whole().write_bytes(scatter_input).expect("write");
    mem.synchronize().expect("inputs");
    comm.step_begin();
    comm.all_reduce(&mut sum.whole(), dtype, ReduceOp::Sum, &stream)
        .expect("all_reduce sum");
    comm.all_reduce(&mut max.whole(), dtype, ReduceOp::Max, &stream)
        .expect("all_reduce max");
    comm.all_gather(&send.whole(), &mut gathered.whole(), &stream)
        .expect("all_gather");
    comm.reduce_scatter(
        &ssend.whole(),
        &mut srecv.whole(),
        dtype,
        ReduceOp::Sum,
        &stream,
    )
    .expect("reduce_scatter");
    comm.broadcast(&mut bcast.whole(), world - 1, &stream)
        .expect("broadcast");
    // Point to point around the ring: even ranks send first, odd ranks receive first.
    let got = DeviceBuffer::alloc(mem, n).expect("alloc");
    let rank = comm.rank();
    let (next, prev) = ((rank + 1) % world, (rank + world - 1) % world);
    if rank.is_multiple_of(2) {
        comm.send(&send.whole(), next, &stream).expect("send");
        comm.recv(&mut got.whole(), prev, &stream).expect("recv");
    } else {
        comm.recv(&mut got.whole(), prev, &stream).expect("recv");
        comm.send(&send.whole(), next, &stream).expect("send");
    }
    mem.synchronize().expect("synchronize");
    comm.step_end().expect("healthy step");
    [sum, max, gathered, srecv, bcast, got]
        .iter()
        .map(|b| b.whole().read_bytes().expect("read"))
        .collect()
}

/// The host reference backend's results for every rank.
fn reference(
    world: usize,
    dtype: DType,
    inputs: &[Vec<u8>],
    scatter: &[Vec<u8>],
) -> Vec<Vec<Vec<u8>>> {
    let group = HostCollective::group(world, Duration::from_secs(60));
    std::thread::scope(|s| {
        let handles: Vec<_> = group
            .into_iter()
            .enumerate()
            .map(|(r, c)| {
                let (i, si) = (&inputs[r], &scatter[r]);
                s.spawn(move || {
                    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(r as u32), 1 << 34);
                    ops(&c, &mem, dtype, i, si)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank"))
            .collect()
    })
}

/// Bit for bit the host backend's results, identical on both ranks: all-reduce (sum, max),
/// all-gather, reduce-scatter, broadcast and send/recv, BF16 and FP32, odd sizes up to one that spans
/// several 32 MiB slots (so steps alternate slot parities within one call).
#[test]
#[ignore = "needs two HIP devices and libturbine_hip.so (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_matches_host_backend_on_two_gpus() {
    if !require_backend("hip") {
        return;
    }
    let _watchdog = watchdog(
        "hostmem_matches_host_backend_on_two_gpus",
        Duration::from_secs(900),
    );
    let mems = devices();
    assert!(mems.len() >= 2, "two AMD devices, found {}", mems.len());
    let mems = &mems[..2];
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load");
    for dtype in [DType::BF16, DType::F32] {
        for elems in [1usize, 7, 4099, 65_537, 1_000_003, 20_000_001] {
            let ctx = format!("{dtype:?} {elems} elements");
            let inputs: Vec<Vec<u8>> = (0..2)
                .map(|r| encode(dtype, &values(r as u64 * 7919 + elems as u64, elems)))
                .collect();
            let scatter: Vec<Vec<u8>> = (0..2)
                .map(|r| encode(dtype, &values(r as u64 * 104_729 + 3, elems * 2)))
                .collect();
            let want = reference(2, dtype, &inputs, &scatter);
            let id = lib.unique_id().expect("id");
            let got: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..2)
                    .map(|r| {
                        let (lib, mem, i, si) =
                            (Arc::clone(&lib), &mems[r], &inputs[r], &scatter[r]);
                        s.spawn(move || {
                            let comm = lib
                                .open(init(r, 2, id, Duration::from_secs(30), Arc::clone(mem)))
                                .expect("open");
                            ops(comm.as_ref(), mem, dtype, i, si)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("rank"))
                    .collect()
            });
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                for (k, name) in [
                    "all_reduce sum",
                    "all_reduce max",
                    "all_gather",
                    "reduce_scatter",
                    "broadcast",
                    "send/recv",
                ]
                .iter()
                .enumerate()
                {
                    assert!(
                        g[k] == w[k],
                        "{ctx} rank {r} {name} differs from the host backend"
                    );
                }
            }
            assert_eq!(
                got[0][0], got[1][0],
                "{ctx}: both ranks hold the same all-reduce bits"
            );
            println!("hostmem {ctx}: every op bitwise equal to the host backend on both ranks");
        }
    }
}

/// A peer that never arrives: the kernel gives up after the op timeout (the device is not
/// hung: the synchronize returns), `step_end` reports `Timeout`, and a later step of the
/// aborted group ends at once.
#[test]
#[ignore = "needs a HIP device and libturbine_hip.so (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_missing_peer_times_out_on_gpu() {
    if !require_backend("hip") {
        return;
    }
    let _watchdog = watchdog(
        "hostmem_missing_peer_times_out_on_gpu",
        Duration::from_secs(60),
    );
    let mems = devices();
    let mem = &mems[0];
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load");
    let id = lib.unique_id().expect("id");
    let comm = Arc::clone(&lib)
        .open(init(0, 2, id, Duration::from_millis(500), Arc::clone(mem)))
        .expect("open");
    let stream = mem.compute_stream();
    let buf = DeviceBuffer::alloc(mem, 1 << 20).expect("alloc");
    let started = Instant::now();
    comm.step_begin();
    comm.all_reduce(&mut buf.whole(), DType::BF16, ReduceOp::Sum, &stream)
        .expect("enqueued");
    mem.synchronize().expect("the kernel gives up by itself");
    let waited = started.elapsed();
    println!("hostmem missing peer: the step ended after {waited:?}");
    assert!(
        waited >= Duration::from_millis(450) && waited < Duration::from_secs(5),
        "{waited:?}"
    );
    let err = comm.step_end().expect_err("timed out");
    assert!(
        matches!(
            err,
            CollectiveError::Timeout {
                op: "all_reduce",
                ..
            }
        ),
        "{err:?}"
    );
    // The group is aborted: a later call fails before enqueuing anything.
    let later = comm
        .all_reduce(&mut buf.whole(), DType::BF16, ReduceOp::Sum, &stream)
        .expect_err("aborted");
    assert!(
        matches!(later, CollectiveError::Timeout { .. }),
        "{later:?}"
    );
}

/// `abort` from the host releases a peer kernel spinning on the device long before its op
/// timeout.
#[test]
#[ignore = "needs two HIP devices and libturbine_hip.so (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_abort_releases_a_spinning_gpu_peer() {
    if !require_backend("hip") {
        return;
    }
    let _watchdog = watchdog(
        "hostmem_abort_releases_a_spinning_gpu_peer",
        Duration::from_secs(60),
    );
    let mems = devices();
    assert!(mems.len() >= 2, "two AMD devices");
    let lib: Arc<dyn CollectiveLibrary> = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load");
    let id = lib.unique_id().expect("id");
    let comms: Vec<Arc<dyn Collective>> = (0..2)
        .map(|r| {
            Arc::clone(&lib)
                .open(init(
                    r,
                    2,
                    id,
                    Duration::from_secs(60),
                    Arc::clone(&mems[r]),
                ))
                .expect("open")
        })
        .collect();
    let mem = &mems[1];
    let buf = DeviceBuffer::alloc(mem, 4096).expect("alloc");
    let started = Instant::now();
    comms[1]
        .all_reduce(
            &mut buf.whole(),
            DType::F32,
            ReduceOp::Sum,
            &mem.compute_stream(),
        )
        .expect("enqueued");
    std::thread::sleep(Duration::from_millis(200));
    comms[0].abort();
    mem.synchronize().expect("released");
    let waited = started.elapsed();
    println!("hostmem abort: the spinning peer ended after {waited:?}");
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    assert!(matches!(
        comms[1].step_end(),
        Err(CollectiveError::RemoteAbort { rank: 0 })
    ));
}

/// Production path with RCCL as the delegate (`auto` threshold): only rank 0 opens, so RCCL's
/// communicator init waits for a rank that never comes. The open must fail with
/// `Timeout { op: "comm_init" }` within the init timeout (plus the delegate-init grace), never
/// hang.
#[test]
#[ignore = "needs a HIP device, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_delegate_init_with_a_missing_peer_fails_in_time() {
    if !require_backend("hip") {
        return;
    }
    let _watchdog = watchdog(
        "hostmem_delegate_init_with_a_missing_peer_fails_in_time",
        Duration::from_secs(60),
    );
    let mems = devices();
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load (with RCCL as the delegate)");
    let id = lib.unique_id().expect("id");
    let started = Instant::now();
    let result = Arc::clone(&lib).open(CollectiveInit {
        route_max_bytes: None,
        ..init(0, 2, id, Duration::from_secs(3), Arc::clone(&mems[0]))
    });
    let waited = started.elapsed();
    println!(
        "hostmem delegate init without its peer: {:?} after {waited:?}",
        result.as_ref().map(|_| ()).map_err(|e| e.to_string())
    );
    let err = result
        .map(|_| ())
        .expect_err("the delegate's peer never opens");
    assert!(
        matches!(
            err,
            CollectiveError::Timeout {
                op: "comm_init",
                ..
            }
        ),
        "{err:?}"
    );
    assert!(waited < Duration::from_secs(10), "{waited:?}");
}

/// Routing on the GPUs: both ranks open with RCCL as the delegate and a 1 KiB threshold; a
/// 64-byte all-reduce stays on hostmem, a 1 MiB one goes to RCCL, and both equal the host
/// backend on both ranks.
#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_routes_large_messages_to_rccl_on_two_gpus() {
    if !require_backend("hip") {
        return;
    }
    let _watchdog = watchdog(
        "hostmem_routes_large_messages_to_rccl_on_two_gpus",
        Duration::from_secs(120),
    );
    let mems = devices();
    assert!(mems.len() >= 2, "two AMD devices");
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load (with RCCL as the delegate)");
    let id = lib.unique_id().expect("id");
    let sizes = [16usize, 262_144];
    let inputs: Vec<Vec<Vec<u8>>> = (0..2)
        .map(|r| {
            sizes
                .iter()
                .map(|&n| encode(DType::F32, &values(r as u64 * 31 + n as u64, n)))
                .collect()
        })
        .collect();
    let got: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|r| {
                let (lib, mem, inputs) = (Arc::clone(&lib), &mems[r], &inputs[r]);
                s.spawn(move || {
                    let comm = lib
                        .open(CollectiveInit {
                            route_max_bytes: Some(1024),
                            ..init(r, 2, id, Duration::from_secs(60), Arc::clone(mem))
                        })
                        .expect("open with the RCCL delegate");
                    let stream = mem.compute_stream();
                    inputs
                        .iter()
                        .map(|input| {
                            let buf = DeviceBuffer::alloc(mem, input.len()).expect("alloc");
                            buf.whole().write_bytes(input).expect("write");
                            comm.step_begin();
                            comm.all_reduce(&mut buf.whole(), DType::F32, ReduceOp::Sum, &stream)
                                .expect("all_reduce");
                            mem.synchronize().expect("sync");
                            comm.step_end().expect("healthy");
                            buf.whole().read_bytes().expect("read")
                        })
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank"))
            .collect()
    });
    for (k, n) in sizes.iter().enumerate() {
        let f = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let sum: Vec<f32> = f(&inputs[0][k])
            .iter()
            .zip(f(&inputs[1][k]))
            .map(|(a, b)| a + b)
            .collect();
        let want = encode(DType::F32, &sum);
        for (r, g) in got.iter().enumerate() {
            assert!(g[k] == want, "{n} F32 elements rank {r} differs");
        }
        println!(
            "hostmem routed all-reduce of {} bytes: equal to the host backend on both ranks",
            n * 4
        );
    }
}

/// Plain `rccl`: only rank 0 opens, so RCCL's communicator init waits for a rank that never
/// comes. The open must fail with `Timeout { op: "comm_init" }` within the init timeout plus
/// the init grace (`ffi::INIT_GRACE`), never hang.
#[test]
#[ignore = "needs a HIP device, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn rccl_init_with_a_missing_peer_fails_in_time() {
    if !require_backend("hip") {
        return;
    }
    let _watchdog = watchdog(
        "rccl_init_with_a_missing_peer_fails_in_time",
        Duration::from_secs(60),
    );
    let mems = devices();
    let lib = collective::registry()
        .get("rccl")
        .expect("registered")
        .load(None)
        .expect("RCCL loads");
    let id = lib.unique_id().expect("id");
    let started = Instant::now();
    let result =
        Arc::clone(&lib).open(init(0, 2, id, Duration::from_secs(3), Arc::clone(&mems[0])));
    let waited = started.elapsed();
    println!(
        "rccl init without its peer: {:?} after {waited:?}",
        result.as_ref().map(|_| ()).map_err(|e| e.to_string())
    );
    let err = result.map(|_| ()).expect_err("the peer never opens");
    assert!(
        matches!(
            err,
            CollectiveError::Timeout {
                op: "comm_init",
                ..
            }
        ),
        "{err:?}"
    );
    assert!(waited < Duration::from_secs(10), "{waited:?}");
}
