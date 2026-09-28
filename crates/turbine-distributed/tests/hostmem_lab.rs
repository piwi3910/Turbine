//! Lab only (novanas, both R9700s): the `hostmem` collective backend on real devices through
//! `libturbine_hip.so` (kernel ABI v2.7). Run by
//! `scripts/lab-test.sh novanas --gpus 2 -- -p turbine-distributed --test hostmem_lab`, which sets
//! `TURBINE_TEST_BACKEND=hip` and `TURBINE_KERNEL_LIBRARY`.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use half::bf16;
use turbine_core::clock::SystemClock;
use turbine_distributed::collective::{
    self, Collective, CollectiveError, CollectiveInit, CollectiveLibrary, HostCollective, ReduceOp,
};
use turbine_kernels::ShimContext;
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

/// Serialises every test of this file: each loads the hostmem library (which reads the
/// process-global copy-engine threshold, `hostmem::set_dma_min_bytes`, that
/// `hostmem_dma_matches_host_backend_on_two_gpus` sets) and several open RCCL communicators,
/// whose concurrent inits in one process fail on novanas with "unhandled system error". libtest
/// runs tests on parallel threads; each test holds this lock from its start, before its
/// watchdog, so a test waiting for the lock is not timed.
static LAB_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lab_serial() -> std::sync::MutexGuard<'static, ()> {
    LAB_TESTS.lock().unwrap_or_else(|p| p.into_inner())
}

/// The device memory (kernel-library context) of every AMD device, in index order.
fn devices() -> Vec<Arc<dyn DeviceMemory>> {
    contexts().into_iter().map(|(mem, _)| mem).collect()
}

/// A device's memory and the context its graphs capture on (kernel ABI v2.1).
type DeviceContext = (Arc<dyn DeviceMemory>, Option<Arc<ShimContext>>);

/// Every AMD device's [`DeviceContext`], in index order.
fn contexts() -> Vec<DeviceContext> {
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
        })
        .map(|opened| (opened.mem, opened.graphs))
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
    if matches!(comm.backend(), "rccl" | "nccl") {
        // NCCL-API point-to-point needs a peer path, which novanas lacks: the input stands in.
        mem.synchronize().expect("synchronize");
        comm.step_end().expect("healthy step");
        let mut out: Vec<Vec<u8>> = [sum, max, gathered, srecv, bcast]
            .iter()
            .map(|b| b.whole().read_bytes().expect("read"))
            .collect();
        out.push(Vec::new());
        return out;
    }
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
    matches_host_backend_on_two_gpus("hostmem_matches_host_backend_on_two_gpus", None);
}

/// P5 Task 32: [`hostmem_matches_host_backend_on_two_gpus`] with the copy-engine all-reduce
/// (kernel ABI v2.8) for every all-reduce of at least 64 KiB — several pipelined chunks from
/// 1 MiB up and several copy-engine calls for 20,000,001 elements (both slot parities) — among
/// the one-shot steps of the other ops on the same device step counter: bit for bit the host
/// backend's results on both ranks.
#[test]
#[ignore = "needs two HIP devices and libturbine_hip.so (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_dma_matches_host_backend_on_two_gpus() {
    matches_host_backend_on_two_gpus(
        "hostmem_dma_matches_host_backend_on_two_gpus",
        Some(64 << 10),
    );
}

fn matches_host_backend_on_two_gpus(name: &'static str, dma_min: Option<u64>) {
    if !require_backend("hip") {
        return;
    }
    let _serial = lab_serial();
    let _watchdog = watchdog(name, Duration::from_secs(900));
    let mems = devices();
    assert!(mems.len() >= 2, "two AMD devices, found {}", mems.len());
    let mems = &mems[..2];
    collective::hostmem::set_dma_min_bytes(dma_min);
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load");
    collective::hostmem::set_dma_min_bytes(None);
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
            println!(
                "{name} {ctx}: every op bitwise equal to the host backend on both ranks \
                 (copy engine from {dma_min:?} bytes)"
            );
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
    let _serial = lab_serial();
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
    let _serial = lab_serial();
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
    let _serial = lab_serial();
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
    let _serial = lab_serial();
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
    let _serial = lab_serial();
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

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// One round's inputs of rank `r`: a BF16 all-reduce of a decode step's hidden rows (16 × 3,072,
/// 96 KiB), an FP32 all-reduce of 64 sums of squares and an FP32 all-gather of `gather` bytes
/// per rank.
fn graph_inputs(round: u64, r: usize, gather: usize) -> [Vec<u8>; 3] {
    let seed = round * 1_000 + r as u64 * 17;
    [
        encode(DType::BF16, &values(seed + 1, 16 * 3072)),
        encode(DType::F32, &values(seed + 2, 64)),
        encode(DType::F32, &values(seed + 3, gather / 4)),
    ]
}

/// Both ranks' expected results of one round of [`graph_inputs`]: the rank-order sums (BF16
/// rounded once) and the rank-major gather.
fn graph_want(inputs: &[[Vec<u8>; 3]]) -> [Vec<u8>; 3] {
    let bsum: Vec<f32> = bf16s(&inputs[0][0])
        .iter()
        .zip(bf16s(&inputs[1][0]))
        .map(|(a, b)| a + b)
        .collect();
    let fsum: Vec<f32> = f32s(&inputs[0][1])
        .iter()
        .zip(f32s(&inputs[1][1]))
        .map(|(a, b)| a + b)
        .collect();
    [
        encode(DType::BF16, &bsum),
        encode(DType::F32, &fsum),
        [inputs[0][2].clone(), inputs[1][2].clone()].concat(),
    ]
}

/// P5 Task 32 (tensor-parallel decode graphs): both ranks capture a step's collectives — a BF16
/// and an FP32 all-reduce on hostmem's kernels and an FP32 all-gather of `gather` bytes per rank
/// (routed by the `auto` thresholds: to the RCCL delegate above 256 KiB) — into a graph once,
/// then replay it for several rounds with fresh inputs, an eager all-reduce between replays.
/// Every replay and every eager step must equal the reference bit for bit on both ranks: the
/// device step counter (kernel ABI v2.8) advances per replay, so no replay reuses a stale
/// sequence number (it would pass its waits early or read another round's slot).
fn graph_replays(name: &'static str, gather: usize, route_max_bytes: Option<u64>) {
    if !require_backend("hip") {
        return;
    }
    let _serial = lab_serial();
    let _watchdog = watchdog(name, Duration::from_secs(180));
    let ctxs = contexts();
    assert!(ctxs.len() >= 2, "two AMD devices, found {}", ctxs.len());
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load (with RCCL as the delegate)");
    let id = lib.unique_id().expect("id");
    const ROUNDS: usize = 20;
    let inputs: Vec<Vec<[Vec<u8>; 3]>> = (0..=ROUNDS as u64)
        .map(|round| (0..2).map(|r| graph_inputs(round, r, gather)).collect())
        .collect();
    let got: Vec<Vec<[Vec<u8>; 3]>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|r| {
                let (lib, (mem, ctx), inputs) = (Arc::clone(&lib), &ctxs[r], &inputs);
                s.spawn(move || {
                    let ctx = ctx
                        .as_ref()
                        .expect("the library exports the graph functions");
                    let comm = lib
                        .open(CollectiveInit {
                            route_max_bytes,
                            ..init(r, 2, id, Duration::from_secs(60), Arc::clone(mem))
                        })
                        .expect("open");
                    let stream = mem.compute_stream();
                    let bufs: Vec<DeviceBuffer> = inputs[0][r]
                        .iter()
                        .map(|i| DeviceBuffer::alloc(mem, i.len()).expect("alloc"))
                        .collect();
                    let gathered = DeviceBuffer::alloc(mem, 2 * gather).expect("alloc");
                    let enqueue = || {
                        comm.all_reduce(&mut bufs[0].whole(), DType::BF16, ReduceOp::Sum, &stream)
                            .expect("bf16 all_reduce");
                        comm.all_reduce(&mut bufs[1].whole(), DType::F32, ReduceOp::Sum, &stream)
                            .expect("f32 all_reduce");
                        comm.all_gather(&bufs[2].whole(), &mut gathered.whole(), &stream)
                            .expect("all_gather");
                    };
                    let write = |round: &[Vec<u8>; 3]| {
                        for (b, i) in bufs.iter().zip(round) {
                            b.whole().write_bytes(i).expect("write");
                        }
                    };
                    let read = || {
                        [
                            bufs[0].whole().read_bytes().expect("read"),
                            bufs[1].whole().read_bytes().expect("read"),
                            gathered.whole().read_bytes().expect("read"),
                        ]
                    };
                    let mut out = Vec::new();
                    // Round 0 eagerly, then the capture (nothing runs), then the replays.
                    write(&inputs[0][r]);
                    comm.step_begin();
                    enqueue();
                    mem.synchronize().expect("sync");
                    comm.step_end().expect("healthy eager step");
                    out.push(read());
                    ctx.graph_begin().expect("begin capture");
                    enqueue();
                    let graph = ctx.graph_end().expect("the collectives are capturable");
                    for round in 1..=ROUNDS {
                        write(&inputs[round][r]);
                        comm.step_begin();
                        ctx.graph_launch(&graph).expect("replay");
                        mem.synchronize().expect("sync");
                        comm.step_end().expect("healthy replay");
                        out.push(read());
                        // An eager all-reduce between replays shares the step counter.
                        write(&inputs[0][r]);
                        comm.step_begin();
                        comm.all_reduce(&mut bufs[0].whole(), DType::BF16, ReduceOp::Sum, &stream)
                            .expect("eager all_reduce");
                        mem.synchronize().expect("sync");
                        comm.step_end().expect("healthy eager step");
                        let eager = bufs[0].whole().read_bytes().expect("read");
                        out.push([eager, Vec::new(), Vec::new()]);
                    }
                    drop(graph);
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank"))
            .collect()
    });
    let first = graph_want(&inputs[0]);
    for (r, rank) in got.iter().enumerate() {
        assert!(rank[0] == first, "{name}: rank {r} eager round 0 differs");
        for round in 1..=ROUNDS {
            let want = graph_want(&inputs[round]);
            let replay = &rank[2 * round - 1];
            for (k, what) in ["bf16 all_reduce", "f32 all_reduce", "all_gather"]
                .iter()
                .enumerate()
            {
                assert!(
                    replay[k] == want[k],
                    "{name}: rank {r} replay {round} {what} differs from the reference"
                );
            }
            assert!(
                rank[2 * round][0] == first[0],
                "{name}: rank {r} eager all_reduce after replay {round} differs"
            );
        }
    }
    println!(
        "{name}: {ROUNDS} graph replays (gather {gather} B per rank) and the eager steps between \
         them bitwise equal to the reference on both ranks"
    );
}

/// [`graph_replays`] with every collective on hostmem's kernels (a 32 KiB all-gather).
#[test]
#[ignore = "needs two HIP devices and libturbine_hip.so (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_graph_replays_match_reference_on_two_gpus() {
    graph_replays(
        "hostmem_graph_replays_match_reference_on_two_gpus",
        32 << 10,
        // No RCCL delegate: hostmem's kernels only.
        Some(u64::MAX),
    );
}

/// [`graph_replays`] with the logits all-gather of a 16-sequence Llama decode step (16 × 64,128
/// FP32 per rank, 4 MiB): routed to the RCCL delegate and captured with the rest.
#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_graph_replays_with_rccl_delegate_on_two_gpus() {
    graph_replays(
        "hostmem_graph_replays_with_rccl_delegate_on_two_gpus",
        16 * 64_128 * 4,
        None,
    );
}

/// Per-word classification of a wrong result `got` against the expected `want`: counts of the
/// words that hold `overwrite` (a later write to the source), `prev` / `next` (a neighbouring
/// round's value) or nothing recognisable (garbage), and the range of differing word indices.
fn classify(
    got: &[u8],
    want: &[u8],
    overwrite: &[u8],
    prev: &[u8],
    next: &[u8],
    w: usize,
) -> String {
    let (mut ow, mut pv, mut nx, mut garbage, mut lo, mut hi, mut n) =
        (0usize, 0usize, 0usize, 0usize, usize::MAX, 0usize, 0usize);
    for (i, (g, x)) in got.chunks_exact(w).zip(want.chunks_exact(w)).enumerate() {
        if g == x {
            continue;
        }
        n += 1;
        lo = lo.min(i);
        hi = hi.max(i);
        let at = |v: &[u8]| v.len() >= (i + 1) * w && &v[i * w..(i + 1) * w] == g;
        if at(overwrite) {
            ow += 1;
        } else if at(prev) {
            pv += 1;
        } else if at(next) {
            nx += 1;
        } else {
            garbage += 1;
        }
    }
    format!(
        "{n} words differ in [{lo}, {hi}]: overwrite {ow}, previous round {pv}, next round {nx}, \
         garbage {garbage}"
    )
}

/// Diagnosis child (does nothing unless `TURBINE_DIAG_CHILD` names a part; run by
/// [`hostmem_diag_rccl_protocols`] in processes of its own, because RCCL reads `NCCL_PROTO` once
/// per process).
///
/// `rccl`: on both GPUs, three modes of 40 rounds of an RCCL (`rccl` backend) FP32 all-gather of
/// 4 MiB per rank with fresh inputs — `overwrite`: each rank overwrites its send buffer right
/// after its own synchronize (as the next step's GEMM would); `barrier`: both ranks synchronize
/// and meet at a host barrier before overwriting; `fresh`: a new send buffer every round, never
/// overwritten. `hostmem`: 30 rounds of a hostmem one-shot BF16 all-reduce (every message on its
/// kernels) at 128 KiB, 1 MiB, 1,000,003 elements and 4 MiB. Prints one `diag …` line per case,
/// classifying every wrong word ([`classify`]).
#[test]
#[ignore = "diagnosis child of hostmem_diag_rccl_protocols"]
fn hostmem_diag_child() {
    let Some(part) = std::env::var_os("TURBINE_DIAG_CHILD") else {
        return;
    };
    if !require_backend("hip") {
        return;
    }
    let mems = devices();
    assert!(mems.len() >= 2, "two AMD devices");
    let mems = &mems[..2];
    let proto = std::env::var("NCCL_PROTO").ok();
    if part == "rccl_ops" {
        diag_rccl_ops(mems);
        return;
    }
    if part == "read_paths" {
        diag_read_paths(mems);
        return;
    }
    if part == "gather_reuse" {
        diag_gather_reuse(mems);
        return;
    }
    if part == "hostmem_fresh_group" {
        diag_hostmem_fresh_group(mems);
        return;
    }
    if part == "rccl" {
        const ROUNDS: u64 = 40;
        const GATHER: usize = 1 << 20; // F32 elements per rank (4 MiB)
        let rccl = collective::registry()
            .get("rccl")
            .expect("registered")
            .load(None)
            .expect("RCCL loads");
        let id = rccl.unique_id().expect("id");
        let data = |mode: u64, round: u64, r: usize| {
            encode(
                DType::F32,
                &values(mode * 100_003 + round * 977 + r as u64, GATHER),
            )
        };
        let junk = |r: usize| encode(DType::F32, &values(0xdead + r as u64, GATHER));
        let modes = ["overwrite", "barrier", "fresh"];
        static WRITE_BAD: AtomicUsize = AtomicUsize::new(0);
        static READ_UNSTABLE: AtomicUsize = AtomicUsize::new(0);
        let fill_of = |round: u64| -> Vec<u8> {
            (0..2 * GATHER)
                .flat_map(|_| (0xa5a5_a500u32 | round as u32).to_le_bytes())
                .collect()
        };
        let barrier = std::sync::Barrier::new(2);
        let gathers: Vec<Vec<Vec<Vec<u8>>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|r| {
                    let (rccl, mem, barrier) = (Arc::clone(&rccl), &mems[r], &barrier);
                    s.spawn(move || {
                        let comm = rccl
                            .open(init(r, 2, id, Duration::from_secs(60), Arc::clone(mem)))
                            .expect("open rccl");
                        let stream = mem.compute_stream();
                        let recv = DeviceBuffer::alloc(mem, 2 * GATHER * 4).expect("alloc");
                        let mut send = DeviceBuffer::alloc(mem, GATHER * 4).expect("alloc");
                        (0..modes.len() as u64)
                            .map(|mode| {
                                (0..ROUNDS)
                                    .map(|round| {
                                        if modes[mode as usize] == "fresh" {
                                            send = DeviceBuffer::alloc(mem, GATHER * 4)
                                                .expect("alloc");
                                        }
                                        let sent = data(mode, round, r);
                                        send.whole().write_bytes(&sent).expect("write");
                                        // Host-copy checks: the write reads back as written,
                                        // and the receive buffer holds this round's fill.
                                        if send.whole().read_bytes().expect("read") != sent {
                                            WRITE_BAD.fetch_add(1, Ordering::Relaxed);
                                        }
                                        let fill = fill_of(round);
                                        recv.whole().write_bytes(&fill).expect("fill");
                                        if recv.whole().read_bytes().expect("read") != fill {
                                            WRITE_BAD.fetch_add(1, Ordering::Relaxed);
                                        }
                                        comm.step_begin();
                                        comm.all_gather(&send.whole(), &mut recv.whole(), &stream)
                                            .expect("all_gather");
                                        mem.synchronize().expect("sync");
                                        comm.step_end().expect("healthy");
                                        match modes[mode as usize] {
                                            "overwrite" => {
                                                send.whole().write_bytes(&junk(r)).expect("junk")
                                            }
                                            "barrier" => {
                                                barrier.wait();
                                                send.whole().write_bytes(&junk(r)).expect("junk")
                                            }
                                            _ => {}
                                        }
                                        let first = recv.whole().read_bytes().expect("read");
                                        let second = recv.whole().read_bytes().expect("read");
                                        if first != second {
                                            READ_UNSTABLE.fetch_add(1, Ordering::Relaxed);
                                        }
                                        first
                                    })
                                    .collect()
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
        let part_bytes = GATHER * 4;
        for (mode, name) in modes.iter().enumerate() {
            let mut bad = Vec::new();
            for (r, rank) in gathers.iter().enumerate() {
                for (round, got) in rank[mode].iter().enumerate() {
                    for q in 0..2 {
                        let g = &got[q * part_bytes..(q + 1) * part_bytes];
                        let want = data(mode as u64, round as u64, q);
                        if g != want.as_slice() {
                            let prev = if round > 0 {
                                data(mode as u64, round as u64 - 1, q)
                            } else {
                                Vec::new()
                            };
                            let next = data(mode as u64, round as u64 + 1, q);
                            let fill = fill_of(round as u64);
                            let prev_fill = if round > 0 {
                                fill_of(round as u64 - 1)
                            } else {
                                Vec::new()
                            };
                            bad.push(format!(
                                "rank {r} round {round} part {q}: {} | vs this round's fill: {}",
                                classify(g, &want, &junk(q), &prev, &next, 4),
                                classify(g, &want, &fill[q * part_bytes..], &prev_fill, &[], 4)
                            ));
                        }
                    }
                }
            }
            println!(
                "diag rccl_all_gather proto={proto:?} mode={name} bad={}/{} write_bad={} \
                 read_unstable={} {:?}",
                bad.len(),
                2 * ROUNDS,
                WRITE_BAD.load(Ordering::Relaxed),
                READ_UNSTABLE.load(Ordering::Relaxed),
                bad.iter().take(4).collect::<Vec<_>>()
            );
        }
        return;
    }

    const ROUNDS: u64 = 30;
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load");
    for elems in [64usize << 10, 512 << 10, 1_000_003, 2 << 20] {
        let input = |round: u64, r: usize| {
            encode(
                DType::BF16,
                &values(elems as u64 + round * 31 + r as u64, elems),
            )
        };
        let sum = |round: u64| {
            let (a, b) = (bf16s(&input(round, 0)), bf16s(&input(round, 1)));
            encode(
                DType::BF16,
                &a.iter().zip(&b).map(|(x, y)| x + y).collect::<Vec<f32>>(),
            )
        };
        let hid = lib.unique_id().expect("id");
        let sums: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|r| {
                    let (lib, mem) = (Arc::clone(&lib), &mems[r]);
                    s.spawn(move || {
                        let comm = lib
                            .open(init(r, 2, hid, Duration::from_secs(60), Arc::clone(mem)))
                            .expect("open hostmem");
                        let stream = mem.compute_stream();
                        let buf = DeviceBuffer::alloc(mem, elems * 2).expect("alloc");
                        (0..ROUNDS)
                            .map(|round| {
                                buf.whole().write_bytes(&input(round, r)).expect("write");
                                comm.step_begin();
                                comm.all_reduce(
                                    &mut buf.whole(),
                                    DType::BF16,
                                    ReduceOp::Sum,
                                    &stream,
                                )
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
        let mut bad = Vec::new();
        for round in 0..ROUNDS {
            let want = sum(round);
            let prev = if round > 0 {
                sum(round - 1)
            } else {
                Vec::new()
            };
            let next = sum(round + 1);
            for (r, rank) in sums.iter().enumerate() {
                let got = &rank[round as usize];
                if *got != want {
                    // "overwrite": the rank's own input left in place (the sum never landed).
                    bad.push(format!(
                        "rank {r} round {round}: {}",
                        classify(got, &want, &input(round, r), &prev, &next, 2)
                    ));
                }
            }
        }
        println!(
            "diag hostmem_all_reduce bytes={} bad={}/{} {:?}",
            elems * 2,
            bad.len(),
            2 * ROUNDS,
            bad.iter().take(4).collect::<Vec<_>>()
        );
    }
}

/// Diagnosis (P5 Task 32 follow-up, collective corruption): runs [`hostmem_diag_child`] in child
/// processes of this test binary — the RCCL part with `NCCL_PROTO` left to the backend's default
/// (`^LL`), forced to `Simple` and to `LL128` (`NCCL_DEBUG=INFO`, `COLL,TUNING`: the first
/// all-gather lines show the algorithm and protocol RCCL picked), then the hostmem part — and
/// prints their `diag` lines. Never fails on wrong bits (it reports them); each child is bounded
/// by the file's watchdog.
#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_diag_rccl_protocols() {
    if !require_backend("hip") {
        return;
    }
    let _serial = lab_serial();
    let _watchdog = watchdog("hostmem_diag_rccl_protocols", Duration::from_secs(1500));
    let exe = std::env::current_exe().expect("test binary");
    // (child part, extra environment)
    let runs: [(&str, &[(&str, &str)]); 2] = [("read_paths", &[]), ("read_paths", &[])];
    for (part, env) in runs {
        let proto = env.first().map(|(k, v)| format!("{k}={v}"));
        let mut cmd = std::process::Command::new(&exe);
        cmd.args([
            "--exact",
            "hostmem_diag_child",
            "--include-ignored",
            "--nocapture",
        ])
        .env("TURBINE_DIAG_CHILD", part);
        for (k, v) in env {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("run the child");
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        for line in stdout.lines().filter(|l| l.starts_with("diag ")) {
            println!(
                "{line} (child {part} {proto:?}, exit {:?})",
                out.status.code()
            );
        }
        for line in stdout
            .lines()
            .chain(stderr.lines())
            .filter(|l| l.contains("AllGather") && (l.contains("proto") || l.contains("algo")))
            .take(2)
        {
            println!("diag rccl-log {proto:?}: {line}");
        }
        if !out.status.success() {
            println!(
                "diag child {part} {proto:?} failed: {}",
                stderr.lines().rev().take(8).collect::<Vec<_>>().join(" | ")
            );
        }
    }
}

/// Diagnosis child `rccl_ops` part: on both GPUs, 40 rounds of each RCCL op with fresh inputs —
/// FP32 all-gather of 4 MiB per rank, BF16 all-reduce of 12 MiB (a 2,048-token Llama tp 2
/// prefill all-reduce), BF16 reduce-scatter of 8 MiB, broadcast of 4 MiB from rank 1 — every
/// output a slice in the middle of a buffer whose 64 KiB guards before and after hold a pattern.
/// Prints per op the rounds with wrong bits (the differing word range) and whether a guard was
/// written (an out-of-bounds writer) or not (wrong output inside the slice).
fn diag_rccl_ops(mems: &[Arc<dyn DeviceMemory>]) {
    const ROUNDS: u64 = 40;
    const GUARD: usize = 64 << 10;
    let rccl = collective::registry()
        .get("rccl")
        .expect("registered")
        .load(None)
        .expect("RCCL loads");
    let id = rccl.unique_id().expect("id");
    let guard = vec![0xa5u8; GUARD];
    // (op, bytes of one rank's input, bytes of the output)
    let ops: [(&str, usize, usize); 4] = [
        ("all_gather", 4 << 20, 8 << 20),
        ("all_reduce", 12 << 20, 12 << 20),
        ("reduce_scatter", 8 << 20, 4 << 20),
        ("broadcast", 4 << 20, 4 << 20),
    ];
    let input = |op: usize, round: u64, r: usize, bytes: usize| {
        encode(
            DType::BF16,
            &values(op as u64 * 1_000_003 + round * 977 + r as u64, bytes / 2),
        )
    };
    let bf = |b: &[u8]| bf16s(b);
    let want = |op: usize, round: u64, rank: usize| -> Vec<u8> {
        let (name, inb, _) = ops[op];
        let (a, b) = (input(op, round, 0, inb), input(op, round, 1, inb));
        let sum = |x: &[u8], y: &[u8]| {
            encode(
                DType::BF16,
                &bf(x)
                    .iter()
                    .zip(bf(y))
                    .map(|(p, q)| p + q)
                    .collect::<Vec<f32>>(),
            )
        };
        match name {
            "all_gather" => [a, b].concat(),
            "all_reduce" => sum(&a, &b),
            "reduce_scatter" => {
                let h = inb / 2;
                sum(&a[rank * h..(rank + 1) * h], &b[rank * h..(rank + 1) * h])
            }
            _ => b,
        }
    };
    // Per rank, per op, per round: the output slice and whether both guards held.
    type Rounds = Vec<Vec<(Vec<u8>, bool)>>;
    let results: Vec<Rounds> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|r| {
                let (rccl, mem, guard) = (Arc::clone(&rccl), &mems[r], &guard);
                s.spawn(move || {
                    let comm = rccl
                        .open(init(r, 2, id, Duration::from_secs(60), Arc::clone(mem)))
                        .expect("open rccl");
                    let stream = mem.compute_stream();
                    (0..ops.len())
                        .map(|op| {
                            let (name, inb, outb) = ops[op];
                            (0..ROUNDS)
                                .map(|round| {
                                    let out =
                                        DeviceBuffer::alloc(mem, outb + 2 * GUARD).expect("alloc");
                                    out.slice(0, GUARD).write_bytes(guard).expect("g");
                                    out.slice(GUARD + outb, GUARD)
                                        .write_bytes(guard)
                                        .expect("g");
                                    let mut slice = out.slice(GUARD, outb);
                                    let send = DeviceBuffer::alloc(mem, inb).expect("alloc");
                                    let data = input(op, round, r, inb);
                                    send.whole().write_bytes(&data).expect("write");
                                    if name == "all_reduce" || name == "broadcast" {
                                        slice.write_bytes(&data).expect("write");
                                    }
                                    comm.step_begin();
                                    match name {
                                        "all_gather" => {
                                            comm.all_gather(&send.whole(), &mut slice, &stream)
                                        }
                                        "all_reduce" => comm.all_reduce(
                                            &mut slice,
                                            DType::BF16,
                                            ReduceOp::Sum,
                                            &stream,
                                        ),
                                        "reduce_scatter" => comm.reduce_scatter(
                                            &send.whole(),
                                            &mut slice,
                                            DType::BF16,
                                            ReduceOp::Sum,
                                            &stream,
                                        ),
                                        _ => comm.broadcast(&mut slice, 1, &stream),
                                    }
                                    .expect("op");
                                    mem.synchronize().expect("sync");
                                    comm.step_end().expect("healthy");
                                    let all = out.whole().read_bytes().expect("read");
                                    let guards_ok = all[..GUARD] == guard[..]
                                        && all[GUARD + outb..] == guard[..];
                                    (all[GUARD..GUARD + outb].to_vec(), guards_ok)
                                })
                                .collect()
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
    for (op, (name, _, _)) in ops.iter().enumerate() {
        let mut bad = Vec::new();
        for (r, rank) in results.iter().enumerate() {
            for (round, (got, guards_ok)) in rank[op].iter().enumerate() {
                let w = want(op, round as u64, r);
                if *got != w || !guards_ok {
                    bad.push(format!(
                        "rank {r} round {round}: guards {}; {}",
                        if *guards_ok { "intact" } else { "WRITTEN" },
                        classify(got, &w, &[], &[], &[], 4)
                    ));
                }
            }
        }
        println!(
            "diag rccl_op op={name} proto={:?} bad={}/{} {:?}",
            std::env::var("NCCL_PROTO").ok(),
            bad.len(),
            2 * ROUNDS,
            bad.iter().take(5).collect::<Vec<_>>()
        );
    }
}

/// Diagnosis child `gather_reuse`: 60 rounds of an RCCL FP32 all-gather of 4 MiB per rank into
/// ONE receive buffer reused every round (the corrupting pattern), a fresh send buffer each round
/// (the old one freed). With `TURBINE_DIAG_GUARD` the receive slice sits between 64 KiB pattern
/// guards. Prints the buffers' device addresses of the first rounds, then per bad round the
/// classification of the wrong words (also against the other part of the same round) and
/// whether a guard was written.
fn diag_gather_reuse(mems: &[Arc<dyn DeviceMemory>]) {
    const ROUNDS: u64 = 60;
    const GATHER: usize = 1 << 20; // F32 words per rank
    const GUARD: usize = 64 << 10;
    let guarded = std::env::var_os("TURBINE_DIAG_GUARD").is_some();
    let rccl = collective::registry()
        .get("rccl")
        .expect("registered")
        .load(None)
        .expect("RCCL loads");
    let id = rccl.unique_id().expect("id");
    let data = |round: u64, r: usize| encode(DType::F32, &values(round * 977 + r as u64, GATHER));
    let guard = vec![0xa5u8; GUARD];
    type Round = (Vec<u8>, bool, String);
    let results: Vec<Vec<Round>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|r| {
                let (rccl, mem, guard) = (Arc::clone(&rccl), &mems[r], &guard);
                s.spawn(move || {
                    let comm = rccl
                        .open(init(r, 2, id, Duration::from_secs(60), Arc::clone(mem)))
                        .expect("open rccl");
                    let stream = mem.compute_stream();
                    let pad = if guarded { GUARD } else { 0 };
                    let recv_buf = DeviceBuffer::alloc(mem, 2 * GATHER * 4 + 2 * pad).expect("a");
                    if guarded {
                        recv_buf.slice(0, GUARD).write_bytes(guard).expect("g");
                        recv_buf
                            .slice(GUARD + 2 * GATHER * 4, GUARD)
                            .write_bytes(guard)
                            .expect("g");
                    }
                    let mut recv = recv_buf.slice(pad, 2 * GATHER * 4);
                    let mut send = DeviceBuffer::alloc(mem, GATHER * 4).expect("alloc");
                    (0..ROUNDS)
                        .map(|round| {
                            send = DeviceBuffer::alloc(mem, GATHER * 4).expect("alloc");
                            let addrs = format!(
                                "recv {:#x}+{} send {:#x}+{}",
                                recv.ptr().addr(),
                                recv.len(),
                                send.whole().ptr().addr(),
                                send.len()
                            );
                            send.whole().write_bytes(&data(round, r)).expect("write");
                            comm.step_begin();
                            comm.all_gather(&send.whole(), &mut recv, &stream)
                                .expect("all_gather");
                            mem.synchronize().expect("sync");
                            comm.step_end().expect("healthy");
                            let all = recv_buf.whole().read_bytes().expect("read");
                            let guards_ok = !guarded
                                || (all[..GUARD] == guard[..]
                                    && all[GUARD + 2 * GATHER * 4..] == guard[..]);
                            (all[pad..pad + 2 * GATHER * 4].to_vec(), guards_ok, addrs)
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
    let part = GATHER * 4;
    let mut bad = Vec::new();
    for (r, rounds) in results.iter().enumerate() {
        for (round, (got, guards_ok, addrs)) in rounds.iter().enumerate() {
            if round < 2 {
                println!("diag gather_reuse addrs rank {r} round {round}: {addrs}");
            }
            for q in 0..2 {
                let g = &got[q * part..(q + 1) * part];
                let want = data(round as u64, q);
                if g != want.as_slice() || !guards_ok {
                    bad.push(format!(
                        "rank {r} round {round} part {q} ({addrs}): guards {}; {}",
                        if *guards_ok { "intact" } else { "WRITTEN" },
                        classify(g, &want, &data(round as u64, 1 - q), &[], &[], 4)
                    ));
                }
            }
        }
    }
    println!(
        "diag gather_reuse guarded={guarded} local_register={:?} bad={}/{} {:?}",
        std::env::var("NCCL_LOCAL_REGISTER").ok(),
        bad.len(),
        2 * ROUNDS,
        bad.iter().take(4).collect::<Vec<_>>()
    );
}

/// Diagnosis child `hostmem_fresh_group`: 30 times a NEW hostmem group (as
/// `hostmem_matches_host_backend_on_two_gpus` opens one per size) whose first step is a BF16
/// all-reduce of 1,000,003 elements, then an FP32 one. Prints the bad steps, classified.
fn diag_hostmem_fresh_group(mems: &[Arc<dyn DeviceMemory>]) {
    const GROUPS: u64 = 30;
    let elems = 1_000_003usize;
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load");
    let mut bad = Vec::new();
    for g in 0..GROUPS {
        let input = |dtype: DType, r: usize| encode(dtype, &values(g * 131 + r as u64, elems));
        let id = lib.unique_id().expect("id");
        let got: Vec<[Vec<u8>; 2]> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|r| {
                    let (lib, mem) = (Arc::clone(&lib), &mems[r]);
                    s.spawn(move || {
                        let comm = lib
                            .open(init(r, 2, id, Duration::from_secs(60), Arc::clone(mem)))
                            .expect("open hostmem");
                        let stream = mem.compute_stream();
                        let b = DeviceBuffer::alloc(mem, elems * 2).expect("alloc");
                        let f = DeviceBuffer::alloc(mem, elems * 4).expect("alloc");
                        b.whole().write_bytes(&input(DType::BF16, r)).expect("w");
                        f.whole().write_bytes(&input(DType::F32, r)).expect("w");
                        mem.synchronize().expect("sync");
                        comm.step_begin();
                        comm.all_reduce(&mut b.whole(), DType::BF16, ReduceOp::Sum, &stream)
                            .expect("bf16");
                        comm.all_reduce(&mut f.whole(), DType::F32, ReduceOp::Sum, &stream)
                            .expect("f32");
                        mem.synchronize().expect("sync");
                        comm.step_end().expect("healthy");
                        [
                            b.whole().read_bytes().expect("read"),
                            f.whole().read_bytes().expect("read"),
                        ]
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank"))
                .collect()
        });
        for (k, dtype) in [DType::BF16, DType::F32].into_iter().enumerate() {
            let (a, b) = (input(dtype, 0), input(dtype, 1));
            let (fa, fb) = match dtype {
                DType::BF16 => (bf16s(&a), bf16s(&b)),
                _ => (f32s(&a), f32s(&b)),
            };
            let want = encode(
                dtype,
                &fa.iter().zip(&fb).map(|(x, y)| x + y).collect::<Vec<f32>>(),
            );
            let w = dtype.size_bytes();
            for (r, rank) in got.iter().enumerate() {
                if rank[k] != want {
                    bad.push(format!(
                        "group {g} {dtype:?} rank {r}: {}",
                        classify(&rank[k], &want, &input(dtype, r), &[], &[], w)
                    ));
                }
            }
        }
    }
    println!(
        "diag hostmem_fresh_group bad={}/{} {:?}",
        bad.len(),
        4 * GROUPS,
        bad.iter().take(6).collect::<Vec<_>>()
    );
}

/// Where the 32 bytes at word `at` of `got` occur in `candidates` (name, byte offset), if
/// anywhere: garbage copied from another buffer of the process keeps its bytes.
fn find_needle(got: &[u8], at: usize, candidates: &[(&str, &[u8])]) -> String {
    let needle = &got[at * 4..(at * 4 + 32).min(got.len())];
    for (name, hay) in candidates {
        if let Some(pos) = hay.windows(needle.len()).position(|w| w == needle) {
            return format!("{name}@{pos}");
        }
    }
    "nowhere".into()
}

/// Diagnosis child `read_paths`: the corrupting pattern — 60 rounds of an RCCL FP32 all-gather
/// of 4 MiB per rank into a reused receive buffer, each rank overwriting its send buffer with
/// junk right after its own synchronize — and after each op three reads of the receive buffer:
/// two pageable `read_bytes` and one through pinned staging (`HostStaging`, kernel ABI v2.3).
/// On a wrong or unstable round, prints which reads agree, whether the pinned read is right, and
/// where the first wrong words' bytes come from. Also prints each rank's native stream handle.
fn diag_read_paths(mems: &[Arc<dyn DeviceMemory>]) {
    const ROUNDS: u64 = 60;
    static CANARY_HIT: AtomicUsize = AtomicUsize::new(0);
    const GATHER: usize = 1 << 20;
    let rccl = collective::registry()
        .get("rccl")
        .expect("registered")
        .load(None)
        .expect("RCCL loads");
    let id = rccl.unique_id().expect("id");
    let data = |round: u64, r: usize| encode(DType::F32, &values(round * 977 + r as u64, GATHER));
    let junk = |r: usize| encode(DType::F32, &values(0xdead + r as u64, GATHER));
    type Reads = (Vec<u8>, Vec<u8>, Vec<u8>, String);
    let results: Vec<Vec<Reads>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|r| {
                let (rccl, mem) = (Arc::clone(&rccl), &mems[r]);
                s.spawn(move || {
                    let comm = rccl
                        .open(init(r, 2, id, Duration::from_secs(60), Arc::clone(mem)))
                        .expect("open rccl");
                    let stream = mem.compute_stream();
                    println!(
                        "diag read_paths rank {r} native compute stream {:#x}",
                        stream.native_handle()
                    );
                    let recv = DeviceBuffer::alloc(mem, 2 * GATHER * 4).expect("alloc");
                    let send = DeviceBuffer::alloc(mem, GATHER * 4).expect("alloc");
                    let staging =
                        turbine_tensor::HostStaging::alloc(mem, 2 * GATHER * 4).expect("staging");
                    let mut reused = vec![0x5au8; 2 * GATHER * 4];
                    // Canaries: host buffers written once and never handed to any copy.
                    let canaries: Vec<Vec<u8>> =
                        (0..8).map(|c| vec![0xc0u8 + c as u8; 2 << 20]).collect();
                    (0..ROUNDS)
                        .map(|round| {
                            send.whole().write_bytes(&data(round, r)).expect("write");
                            comm.step_begin();
                            comm.all_gather(&send.whole(), &mut recv.whole(), &stream)
                                .expect("all_gather");
                            mem.synchronize().expect("sync");
                            comm.step_end().expect("healthy");
                            send.whole().write_bytes(&junk(r)).expect("junk");
                            let first = recv.whole().read_bytes().expect("read");
                            let host = first.as_ptr() as usize;
                            let addrs = format!(
                                "len {} dev {:#x} host {host:#x} (mod 4096 {}) end mod 4096 {}",
                                first.len(),
                                recv.ptr().addr(),
                                host % 4096,
                                (host + first.len()) % 4096
                            );
                            // #2: pageable into a buffer allocated once and touched (not a
                            // fresh allocation).
                            recv.copy_to_host(0, &mut reused).expect("read");
                            let second = reused.clone();
                            staging.download(0, recv.whole()).expect("download");
                            let mut pinned = vec![0u8; 2 * GATHER * 4];
                            staging.read(0, &mut pinned).expect("staged read");
                            for (c, canary) in canaries.iter().enumerate() {
                                if canary.iter().any(|&b| b != 0xc0u8 + c as u8) {
                                    CANARY_HIT.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            (first, second, pinned, addrs)
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
    let mut lines = Vec::new();
    let (mut bad_pageable, mut bad_pinned, mut unstable) = (0, 0, 0);
    for (r, rounds) in results.iter().enumerate() {
        for (round, (first, second, pinned, addrs)) in rounds.iter().enumerate() {
            let want = [data(round as u64, 0), data(round as u64, 1)].concat();
            let (p_ok, s_ok, pin_ok) = (*first == want, *second == want, *pinned == want);
            bad_pageable += usize::from(!p_ok) + usize::from(!s_ok);
            bad_pinned += usize::from(!pin_ok);
            unstable += usize::from(first != second);
            if p_ok && s_ok && pin_ok {
                continue;
            }
            let at = first
                .chunks_exact(4)
                .zip(want.chunks_exact(4))
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let (j0, j1) = (junk(0), junk(1));
            let (d0, d1) = (data(round as u64 + 1, 0), data(round as u64 + 1, 1));
            lines.push(format!(
                "rank {r} round {round} [{addrs}]: pageable#1 {} pageable#2 {} pinned {} (#1==#2 {}); \
                 #1 {}; first wrong word {at} of #1 from {}",
                if p_ok { "ok" } else { "BAD" },
                if s_ok { "ok" } else { "BAD" },
                if pin_ok { "ok" } else { "BAD" },
                first == second,
                classify(first, &want, &[], &[], &[], 4),
                find_needle(
                    first,
                    at,
                    &[
                        ("junk rank 0", &j0),
                        ("junk rank 1", &j1),
                        ("next data rank 0", &d0),
                        ("next data rank 1", &d1),
                        ("want", &want),
                    ]
                )
            ));
        }
    }
    println!(
        "diag read_paths rounds={} bad_pageable_reads={bad_pageable} (#1 fresh Vec, #2 reused \
         touched Vec) bad_pinned_reads={bad_pinned} unstable={unstable} canary_hits={} {:?}",
        2 * ROUNDS,
        CANARY_HIT.load(Ordering::Relaxed),
        lines.iter().take(6).collect::<Vec<_>>()
    );
}

/// Lab only, two GPUs (P5 Task 32 follow-up, upstream report): builds the plain-HIP repro
/// `docs/upstream/rocm-pageable-d2h/repro.hip` with the Job's hipcc (`TURBINE_ROCM_PATH`) and
/// runs its variants (`TURBINE_REPRO_ROUNDS` rounds each, default 2,000), printing their `repro`
/// lines. Never fails on wrong bytes (it reports them).
#[test]
#[ignore = "needs two HIP devices and hipcc (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_repro_pageable_d2h() {
    if !require_backend("hip") {
        return;
    }
    let _serial = lab_serial();
    let _watchdog = watchdog("hostmem_repro_pageable_d2h", Duration::from_secs(900));
    let rocm = std::env::var("TURBINE_ROCM_PATH").unwrap_or_else(|_| "/opt/rocm/rocm".into());
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/upstream/rocm-pageable-d2h/repro.hip");
    let out = std::env::temp_dir().join(format!("turbine-repro-{}", std::process::id()));
    let build = std::process::Command::new(format!("{rocm}/bin/hipcc"))
        .args(["-O2", "--offload-arch=gfx1201", "-o"])
        .arg(&out)
        .arg(&src)
        .arg("-lpthread")
        .output()
        .expect("run hipcc");
    assert!(
        build.status.success(),
        "hipcc: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let rounds = std::env::var("TURBINE_REPRO_ROUNDS").unwrap_or_else(|_| "2000".into());
    for (mode, threads, extra) in [
        ("async-shared", "2", "busy"),
        ("sync-shared", "2", "busy"),
        ("async-shared", "2", "idle"),
        ("pinned", "2", "busy"),
        ("async-fresh", "2", "busy"),
        ("async-shared", "1", "busy"),
    ] {
        let run = std::process::Command::new(&out)
            .args([mode, threads, extra, rounds.as_str()])
            .env("NCCL_PROTO", "^LL")
            .output()
            .expect("run the repro");
        for line in String::from_utf8_lossy(&run.stdout).lines() {
            println!("{line}");
        }
        if !run.status.success() {
            println!(
                "repro {mode} {threads} failed: {}",
                String::from_utf8_lossy(&run.stderr)
            );
        }
    }
}

/// Rounds of [`hostmem_stress_collectives`] for a message of `elems` elements: the base count
/// (`TURBINE_STRESS_ROUNDS`, default 5), ×20 up to 64 Ki elements, ×4 up to 1 Mi.
fn stress_rounds(elems: usize) -> u64 {
    // The environment, else an uncommitted scripts/lab/stress-rounds.local (lab Jobs pass no
    // environment through), else 5.
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/lab/stress-rounds.local");
    let base: u64 = std::env::var("TURBINE_STRESS_ROUNDS")
        .ok()
        .or_else(|| std::fs::read_to_string(local).ok())
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(5);
    match elems {
        0..=65_537 => base * 20,
        65_538..=1_048_577 => base * 4,
        _ => base,
    }
}

/// P5 Task 32 corruption proof (lab, two GPUs): every collective op — all-reduce sum and max,
/// all-gather, reduce-scatter, broadcast, send/recv (hostmem only: RCCL point-to-point needs a
/// peer path) — on `hostmem` with its kernels at every size, `hostmem` routed at its `auto`
/// thresholds (large messages on its RCCL delegate) and plain `rccl`, BF16 and FP32, messages of
/// 2 .. 4,194,305 elements (8 B .. 16 MiB), fresh inputs every round (see [`stress_rounds`];
/// `TURBINE_STRESS_ROUNDS` scales the proof run), every output read back (through the context's
/// pinned bounce buffer) and compared bit for bit with the host reference backend. Fails on any
/// mismatch, naming it.
#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_stress_collectives() {
    if !require_backend("hip") {
        return;
    }
    let _serial = lab_serial();
    // Scaled with the rounds: about 50 s per base round, at least 30 min.
    let limit = Duration::from_secs(
        std::env::var("TURBINE_STRESS_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800.max(stress_rounds(u32::MAX as usize) * 120)),
    );
    let _watchdog = watchdog("hostmem_stress_collectives", limit);
    let mems = devices();
    assert!(mems.len() >= 2, "two AMD devices");
    let mems = &mems[..2];
    let names = [
        "all_reduce sum",
        "all_reduce max",
        "all_gather",
        "reduce_scatter",
        "broadcast",
        "send/recv",
    ];
    let mut total = 0u64;
    let mut bad = Vec::new();
    for (label, backend, route) in [
        ("hostmem kernels", "hostmem", Some(u64::MAX)),
        ("hostmem auto", "hostmem", None),
        ("rccl", "rccl", None),
    ] {
        let lib = collective::registry()
            .get(backend)
            .expect("registered")
            .load(None)
            .expect("load");
        let id = lib.unique_id().expect("id");
        let comms: Vec<Arc<dyn Collective>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|r| {
                    let (lib, mem) = (Arc::clone(&lib), &mems[r]);
                    s.spawn(move || {
                        lib.open(CollectiveInit {
                            route_max_bytes: route,
                            ..init(r, 2, id, Duration::from_secs(60), Arc::clone(mem))
                        })
                        .expect("open")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank"))
                .collect()
        });
        for dtype in [DType::BF16, DType::F32] {
            for elems in [2usize, 4099, 65_537, 524_289, 4_194_305] {
                for round in 0..stress_rounds(elems) {
                    let seed = round * 1_000_003 + elems as u64;
                    let inputs: Vec<Vec<u8>> = (0..2)
                        .map(|r| encode(dtype, &values(seed + r as u64 * 7919, elems)))
                        .collect();
                    let scatter: Vec<Vec<u8>> = (0..2)
                        .map(|r| encode(dtype, &values(seed + r as u64 * 104_729 + 3, elems * 2)))
                        .collect();
                    let want = reference(2, dtype, &inputs, &scatter);
                    let got: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
                        let handles: Vec<_> = (0..2)
                            .map(|r| {
                                let (comm, mem, i, si) =
                                    (&comms[r], &mems[r], &inputs[r], &scatter[r]);
                                s.spawn(move || ops(comm.as_ref(), mem, dtype, i, si))
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| h.join().expect("rank"))
                            .collect()
                    });
                    for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                        for (k, name) in names.iter().enumerate() {
                            if g[k].is_empty() {
                                continue;
                            }
                            total += 1;
                            if g[k] != w[k] {
                                bad.push(format!(
                                    "{label} {dtype:?} {elems} round {round} rank {r} {name}"
                                ));
                            }
                        }
                    }
                }
                println!(
                    "stress {label} {dtype:?} {elems} elements: {} rounds, mismatches so far {}",
                    stress_rounds(elems),
                    bad.len()
                );
            }
        }
    }
    println!(
        "stress total checked outputs {total}, mismatches {}",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} mismatches: {:?}",
        bad.len(),
        &bad[..bad.len().min(10)]
    );
}
