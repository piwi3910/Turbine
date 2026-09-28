//! `turbine-collbench`: collective bandwidth benchmark with the `nccl-tests` formulas (P5 S-3).
//!
//! Sizes double from `--min-bytes` to `--max-bytes`; per size and operation it checks one result
//! against the host reference backend, then runs `--warmup` untimed and `--iters` timed
//! iterations (each one operation plus a stream synchronize) and reports the median.
//! `bytes` follows nccl-tests: the all-reduce/broadcast buffer, the all-gather receive buffer and
//! the reduce-scatter send buffer. Device buffers and streams come from the phase-1 device layer
//! (`ShimContext` over the kernel shim, `HostMemory` for `--backend host`).
//!
//! Exit codes: 0 every result correct, 1 any mismatch or collective/device error, 2 usage error.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::ExitCode;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use serde::Serialize;
use turbine_core::clock::{Clock, SystemClock};
use turbine_core::config::ByteSize;
use turbine_core::types::DeviceId;
use turbine_distributed::collective::{
    self, Collective, CollectiveInit, CollectiveMetrics, CollectiveOp, HostCollective, ReduceOp,
};
use turbine_kernels::backends::BackendRequest;
use turbine_observability::MetricsRegistry;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DType, DeviceBuffer, DeviceMemory};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum OpArg {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DTypeArg {
    Bf16,
    Fp32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Output {
    Text,
    Json,
}

#[derive(Parser, Debug)]
#[command(
    name = "turbine-collbench",
    about = "Collective bandwidth benchmark (nccl-tests formulas), checked against the host backend"
)]
struct Args {
    /// A registered collective backend (`collective_backend` registry: host, rccl, nccl,
    /// hostmem).
    #[arg(long)]
    backend: String,
    /// Explicit collective library (else the backend's default search).
    #[arg(long)]
    library: Option<std::path::PathBuf>,
    /// Explicit kernel library for the device buffers (else the execution backend's search).
    #[arg(long)]
    kernel_library: Option<std::path::PathBuf>,
    /// Global device indices, comma-separated: one thread per device, or the one device of
    /// this rank process with --rank/--world/--leader.
    #[arg(long, value_delimiter = ',', required = true)]
    devices: Vec<u32>,
    #[arg(long, value_enum, default_value = "all")]
    op: OpArg,
    #[arg(long, default_value = "8")]
    min_bytes: ByteSize,
    #[arg(long, default_value = "1GiB")]
    max_bytes: ByteSize,
    #[arg(long, default_value_t = 20)]
    iters: u32,
    #[arg(long, default_value_t = 5)]
    warmup: u32,
    #[arg(long, value_enum, default_value = "bf16")]
    dtype: DTypeArg,
    /// This process's rank (multi-process mode).
    #[arg(long, requires_all = ["world", "leader"])]
    rank: Option<usize>,
    /// Ranks in the communicator (multi-process mode).
    #[arg(long, requires_all = ["rank", "leader"])]
    world: Option<usize>,
    /// Rank 0 listens here and hands the 128-byte unique id to every other rank.
    #[arg(long, requires_all = ["rank", "world"])]
    leader: Option<SocketAddr>,
    /// Enqueue the timed iterations back to back with one synchronize (mean per op) instead of
    /// timing each op plus its own synchronize (median).
    #[arg(long)]
    pipelined: bool,
    #[arg(long, value_enum, default_value = "text")]
    output: Output,
}

/// One size of one operation.
#[derive(Serialize, Clone, Debug, PartialEq)]
struct Row {
    bytes: u64,
    time_us: f64,
    algbw_gbps: f64,
    busbw_gbps: f64,
    correct: bool,
}

#[derive(Serialize, Debug)]
struct Report {
    backend: &'static str,
    world: usize,
    dtype: &'static str,
    ops: Vec<OpReport>,
}

#[derive(Serialize, Debug)]
struct OpReport {
    op: &'static str,
    rows: Vec<Row>,
}

/// nccl-tests bus-bandwidth factor for `op` over `n` ranks.
fn busbw_factor(op: CollectiveOp, n: usize) -> f64 {
    let n = n as f64;
    match op {
        CollectiveOp::AllReduce => 2.0 * (n - 1.0) / n,
        CollectiveOp::AllGather | CollectiveOp::ReduceScatter => (n - 1.0) / n,
        CollectiveOp::Broadcast | CollectiveOp::Barrier => 1.0,
    }
}

fn row(op: CollectiveOp, world: usize, bytes: u64, seconds: f64, correct: bool) -> Row {
    let algbw = if seconds > 0.0 {
        bytes as f64 / seconds / 1e9
    } else {
        0.0
    };
    Row {
        bytes,
        time_us: seconds * 1e6,
        algbw_gbps: algbw,
        busbw_gbps: algbw * busbw_factor(op, world),
        correct,
    }
}

fn ops(arg: OpArg) -> Vec<CollectiveOp> {
    match arg {
        OpArg::AllReduce => vec![CollectiveOp::AllReduce],
        OpArg::AllGather => vec![CollectiveOp::AllGather],
        OpArg::ReduceScatter => vec![CollectiveOp::ReduceScatter],
        OpArg::Broadcast => vec![CollectiveOp::Broadcast],
        OpArg::All => vec![
            CollectiveOp::AllReduce,
            CollectiveOp::AllGather,
            CollectiveOp::ReduceScatter,
            CollectiveOp::Broadcast,
        ],
    }
}

/// Byte sizes from `min` to `max`, doubling; each rounded up to whole elements and, for
/// all-gather and reduce-scatter, to a multiple of `world` elements.
fn sizes(min: u64, max: u64, elem: u64, op: CollectiveOp, world: usize) -> Vec<u64> {
    let unit = match op {
        CollectiveOp::AllGather | CollectiveOp::ReduceScatter => elem * world as u64,
        _ => elem,
    };
    let mut out = Vec::new();
    let mut s = min.max(1);
    while s <= max {
        let rounded = s.div_ceil(unit) * unit;
        if out.last() != Some(&rounded) {
            out.push(rounded);
        }
        s = s.saturating_mul(2);
    }
    out
}

/// Deterministic small-integer input of `rank` (exact in BF16 and FP32 for any summation order
/// of up to 64 ranks), encoded in `dtype`.
fn input(rank: usize, elems: usize, dtype: DType) -> Vec<u8> {
    let value = |i: usize| (((i * 7 + rank * 13) % 17) as f32) - 8.0;
    let mut out = Vec::with_capacity(elems * dtype.size_bytes());
    for i in 0..elems {
        let v = value(i);
        match dtype {
            DType::BF16 => out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
            _ => out.extend_from_slice(&v.to_le_bytes()),
        }
    }
    out
}

/// Buffer lengths `(send, recv)` for `bytes` of `op`; in-place ops use only `send`.
fn lens(op: CollectiveOp, bytes: usize, world: usize) -> (usize, usize) {
    match op {
        CollectiveOp::AllGather => (bytes / world, bytes),
        CollectiveOp::ReduceScatter => (bytes, bytes / world),
        _ => (bytes, bytes),
    }
}

/// The expected result bytes of `rank` from the host reference backend, run on every rank's
/// input (inputs are a pure function of rank, so every process computes the same reference).
fn reference(op: CollectiveOp, bytes: usize, world: usize, dtype: DType, rank: usize) -> Vec<u8> {
    let elem = dtype.size_bytes();
    let (send_len, recv_len) = lens(op, bytes, world);
    let group = HostCollective::group(world, Duration::from_secs(60));
    let mut results: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .enumerate()
            .map(|(r, comm)| {
                scope.spawn(move || {
                    let mem: Arc<dyn DeviceMemory> =
                        HostMemory::new(DeviceId(r as u32), (send_len + recv_len) as u64 * 2);
                    let stream = mem.compute_stream();
                    let send = DeviceBuffer::alloc(&mem, send_len).expect("host alloc");
                    let recv = DeviceBuffer::alloc(&mem, recv_len).expect("host alloc");
                    send.whole()
                        .write_bytes(&input(r, send_len / elem, dtype))
                        .expect("host write");
                    let mut s = send.whole();
                    let mut d = recv.whole();
                    match op {
                        CollectiveOp::AllReduce => {
                            comm.all_reduce(&mut s, dtype, ReduceOp::Sum, &stream)
                                .expect("host all-reduce");
                            s.read_bytes().expect("host read")
                        }
                        CollectiveOp::AllGather => {
                            comm.all_gather(&s, &mut d, &stream)
                                .expect("host all-gather");
                            d.read_bytes().expect("host read")
                        }
                        CollectiveOp::ReduceScatter => {
                            comm.reduce_scatter(&s, &mut d, dtype, ReduceOp::Sum, &stream)
                                .expect("host reduce-scatter");
                            d.read_bytes().expect("host read")
                        }
                        _ => {
                            comm.broadcast(&mut s, 0, &stream).expect("host broadcast");
                            s.read_bytes().expect("host read")
                        }
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("host reference rank"))
            .collect()
    });
    results.swap_remove(rank)
}

/// What one rank needs: its communicator and the device memory its buffers live in.
struct RankEnv {
    comm: Arc<dyn Collective>,
    mem: Arc<dyn DeviceMemory>,
}

/// One rank's rows per operation, or its first error.
type RankResult = Result<Vec<(CollectiveOp, Vec<Row>)>, String>;

struct Plan {
    ops: Vec<CollectiveOp>,
    min: u64,
    max: u64,
    iters: u32,
    warmup: u32,
    dtype: DType,
    /// `--pipelined`: time `iters` back-to-back ops and one synchronize (mean), not the median of
    /// op + synchronize.
    pipelined: bool,
}

/// Runs every size of every op on one rank. Every rank of the group must call this with the
/// same plan. Returns the rows (median time of this rank) or the first error.
fn run_rank(env: &RankEnv, plan: &Plan) -> RankResult {
    let world = env.comm.world_size();
    let rank = env.comm.rank();
    let elem = plan.dtype.size_bytes();
    let stream = env.mem.compute_stream();
    let mut out = Vec::new();
    for &op in &plan.ops {
        let mut rows = Vec::new();
        for bytes in sizes(plan.min, plan.max, elem as u64, op, world) {
            let bytes_us = usize::try_from(bytes).map_err(|_| "size overflows usize")?;
            let (send_len, recv_len) = lens(op, bytes_us, world);
            let send = DeviceBuffer::alloc(&env.mem, send_len).map_err(|e| e.to_string())?;
            let recv = DeviceBuffer::alloc(&env.mem, recv_len).map_err(|e| e.to_string())?;
            let enqueue = || -> Result<(), String> {
                let mut s = send.whole();
                let mut d = recv.whole();
                match op {
                    CollectiveOp::AllReduce => {
                        env.comm
                            .all_reduce(&mut s, plan.dtype, ReduceOp::Sum, &stream)
                    }
                    CollectiveOp::AllGather => env.comm.all_gather(&s, &mut d, &stream),
                    CollectiveOp::ReduceScatter => {
                        env.comm
                            .reduce_scatter(&s, &mut d, plan.dtype, ReduceOp::Sum, &stream)
                    }
                    _ => env.comm.broadcast(&mut s, 0, &stream),
                }
                .map_err(|e| format!("{} {bytes} B: {e}", op.as_str()))
            };
            let run_once = || -> Result<(), String> {
                enqueue()?;
                env.mem.synchronize().map_err(|e| e.to_string())
            };

            // Correctness: one operation on known inputs.
            send.whole()
                .write_bytes(&input(rank, send_len / elem, plan.dtype))
                .map_err(|e| e.to_string())?;
            run_once()?;
            let got = match op {
                CollectiveOp::AllGather | CollectiveOp::ReduceScatter => recv.whole().read_bytes(),
                _ => send.whole().read_bytes(),
            }
            .map_err(|e| e.to_string())?;
            let correct = got == reference(op, bytes_us, world, plan.dtype, rank);

            for _ in 0..plan.warmup {
                run_once()?;
            }
            let seconds = if plan.pipelined {
                // Every iteration enqueued back to back, one synchronize: the per-op device time
                // without the host round trip of a synchronize per op.
                let started = Instant::now();
                for _ in 0..plan.iters.max(1) {
                    enqueue()?;
                }
                env.mem.synchronize().map_err(|e| e.to_string())?;
                started.elapsed().as_secs_f64() / f64::from(plan.iters.max(1))
            } else {
                let mut times = Vec::with_capacity(plan.iters as usize);
                for _ in 0..plan.iters.max(1) {
                    let started = Instant::now();
                    run_once()?;
                    times.push(started.elapsed().as_secs_f64());
                }
                times.sort_by(f64::total_cmp);
                times[times.len() / 2]
            };
            rows.push(row(op, world, bytes, seconds, correct));
        }
        out.push((op, rows));
    }
    Ok(out)
}

fn usage(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("turbine-collbench: {msg}");
    ExitCode::from(2)
}

fn failure(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("turbine-collbench: {msg}");
    ExitCode::from(1)
}

/// Rank 0: listen on `leader` and send `id` to each of the other `world - 1` ranks.
fn serve_unique_id(leader: SocketAddr, world: usize, id: &[u8; 128]) -> Result<(), String> {
    let listener = TcpListener::bind(leader).map_err(|e| format!("bind {leader}: {e}"))?;
    for _ in 1..world {
        let (mut conn, _) = listener.accept().map_err(|e| format!("accept: {e}"))?;
        conn.write_all(id)
            .map_err(|e| format!("send unique id: {e}"))?;
    }
    Ok(())
}

/// Other ranks: connect to `leader` (retrying for 60 s) and read the 128-byte unique id.
fn fetch_unique_id(leader: SocketAddr) -> Result<[u8; 128], String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut backoff = Duration::from_millis(50);
    let mut conn = loop {
        match TcpStream::connect(leader) {
            Ok(c) => break c,
            Err(e) if Instant::now() >= deadline => {
                return Err(format!("connect {leader}: {e}"));
            }
            Err(_) => {
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(1));
            }
        }
    };
    let mut id = [0u8; 128];
    conn.read_exact(&mut id)
        .map_err(|e| format!("read unique id: {e}"))?;
    Ok(id)
}

/// The device memory of global device `index`: the registered execution backend of the device's
/// vendor opens it (its kernel library's context), as the server would.
fn device_memory(
    inventory: &turbine_device::DeviceInventory,
    index: u32,
    kernel_library: Option<&std::path::Path>,
) -> Result<Arc<dyn DeviceMemory>, String> {
    let device = inventory
        .devices
        .iter()
        .find(|d| d.index.0 == index)
        .ok_or_else(|| format!("device {index} is not in the inventory"))?;
    let exec = turbine_kernels::backends::registry()
        .iter()
        .find(|b| b.vendor() == device.vendor.as_str())
        .ok_or_else(|| format!("no execution backend is registered for {:?}", device.vendor))?;
    let opened = exec
        .open(&BackendRequest {
            device: device.index,
            kernel_library,
            inventory,
            meminfo: std::path::Path::new("/proc/meminfo"),
            card_profile: "auto",
        })
        .map_err(|e| format!("device {index}: {e}"))?;
    Ok(opened.mem)
}

fn main() -> ExitCode {
    let args = match Args::try_parse() {
        Ok(a) => a,
        Err(e) => {
            let code = if e.use_stderr() { 2 } else { 0 };
            let _ = e.print();
            return ExitCode::from(code);
        }
    };
    if args.min_bytes > args.max_bytes {
        return usage("--min-bytes must not exceed --max-bytes");
    }
    let dtype = match args.dtype {
        DTypeArg::Bf16 => DType::BF16,
        DTypeArg::Fp32 => DType::F32,
    };
    let registry = collective::registry();
    let Some(backend) = registry.get(&args.backend) else {
        return usage(registry.unknown(&args.backend));
    };
    let on_device = !backend.vendors().is_empty();
    let multi = args.rank.zip(args.world).zip(args.leader);
    let world = match multi {
        Some(((rank, world), _)) => {
            if args.devices.len() != 1 {
                return usage("--rank/--world/--leader drive exactly one --devices entry");
            }
            if world < 2 || rank >= world {
                return usage("--rank must be below --world, which must be at least 2");
            }
            if !on_device {
                return usage(
                    "a host-memory backend runs in one process; omit --rank/--world/--leader",
                );
            }
            world
        }
        None => args.devices.len(),
    };
    let plan = Plan {
        ops: ops(args.op),
        min: args.min_bytes.0,
        max: args.max_bytes.0,
        iters: args.iters,
        warmup: args.warmup,
        dtype,
        pipelined: args.pipelined,
    };
    let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
    let metrics = CollectiveMetrics::register(&MetricsRegistry::new());
    let init_timeout = Duration::from_secs(120);
    let op_timeout = Duration::from_secs(60);

    let lib = match backend.load(args.library.as_deref()) {
        Ok(lib) => lib,
        Err(e) => return failure(e),
    };
    eprintln!(
        "turbine-collbench: backend {} {}",
        lib.backend(),
        lib.version().unwrap_or_default()
    );
    let inventory = if on_device {
        match turbine_device::discover(&Default::default()) {
            Ok(inv) => Some(inv),
            Err(e) => return failure(format!("device discovery: {e}")),
        }
    } else {
        None
    };
    let (first_rank, unique_id) = match multi {
        Some(((0, world), leader)) => {
            let id = match lib.unique_id() {
                Ok(id) => id,
                Err(e) => return failure(e),
            };
            std::thread::spawn(move || {
                if let Err(e) = serve_unique_id(leader, world, &id) {
                    eprintln!("turbine-collbench: {e}");
                }
            });
            (0, id)
        }
        Some(((rank, _), leader)) => match fetch_unique_id(leader) {
            Ok(id) => (rank, id),
            Err(e) => return failure(e),
        },
        None => match lib.unique_id() {
            Ok(id) => (0, id),
            Err(e) => return failure(e),
        },
    };
    // Every local rank opens its communicator in its own thread (communicator init is
    // collective), after its device memory exists.
    let ready = Barrier::new(args.devices.len());
    let results: Vec<RankResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = args
            .devices
            .iter()
            .enumerate()
            .map(|(i, &dev)| {
                let (lib, inventory, kernel_library) =
                    (Arc::clone(&lib), &inventory, args.kernel_library.as_deref());
                let (clock, metrics, plan, ready) =
                    (Arc::clone(&clock), metrics.clone(), &plan, &ready);
                scope.spawn(move || {
                    let mem: Result<Arc<dyn DeviceMemory>, String> = match inventory {
                        Some(inv) => device_memory(inv, dev, kernel_library),
                        None => Ok(HostMemory::new(DeviceId(dev), u64::MAX)),
                    };
                    ready.wait();
                    let mem = mem?;
                    let rank = first_rank + i;
                    let comm = lib
                        .open(CollectiveInit {
                            rank,
                            world,
                            unique_id,
                            init_timeout,
                            op_timeout,
                            clock,
                            metrics: Some(metrics),
                            memory: Some(Arc::clone(&mem)),
                        })
                        .map_err(|e| format!("rank {rank}: {e}"))?;
                    run_rank(&RankEnv { comm, mem }, plan)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("rank panicked".into())))
            .collect()
    });

    let mut first = None;
    for r in results {
        match r {
            Ok(rows) if first.is_none() => first = Some(rows),
            Ok(_) => {}
            Err(e) => return failure(e),
        }
    }
    let Some(rows) = first else {
        return failure("no rank produced results");
    };
    let report = Report {
        backend: backend.name(),
        world,
        dtype: dtype.as_str(),
        ops: rows
            .into_iter()
            .map(|(op, rows)| OpReport {
                op: op.as_str(),
                rows,
            })
            .collect(),
    };
    let all_correct = report.ops.iter().all(|o| o.rows.iter().all(|r| r.correct));
    match args.output {
        Output::Json => match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{text}"),
            Err(e) => return failure(e),
        },
        Output::Text => {
            println!(
                "# backend {} world {} dtype {}",
                report.backend, report.world, report.dtype
            );
            for op in &report.ops {
                println!("# {}", op.op);
                println!(
                    "{:>14} {:>12} {:>12} {:>12} {:>8}",
                    "bytes", "time_us", "algbw_gbps", "busbw_gbps", "correct"
                );
                for r in &op.rows {
                    println!(
                        "{:>14} {:>12.2} {:>12.3} {:>12.3} {:>8}",
                        r.bytes, r.time_us, r.algbw_gbps, r.busbw_gbps, r.correct
                    );
                }
            }
        }
    }
    if all_correct {
        ExitCode::SUCCESS
    } else {
        failure("at least one result differs from the host reference")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busbw_formulas() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        for n in [2usize, 8] {
            let nf = n as f64;
            assert!(close(
                busbw_factor(CollectiveOp::AllReduce, n),
                2.0 * (nf - 1.0) / nf
            ));
            assert!(close(
                busbw_factor(CollectiveOp::AllGather, n),
                (nf - 1.0) / nf
            ));
            assert!(close(
                busbw_factor(CollectiveOp::ReduceScatter, n),
                (nf - 1.0) / nf
            ));
            assert!(close(busbw_factor(CollectiveOp::Broadcast, n), 1.0));
        }
        assert!(close(busbw_factor(CollectiveOp::AllReduce, 2), 1.0));
        assert!(close(busbw_factor(CollectiveOp::AllReduce, 8), 1.75));

        let r = row(CollectiveOp::AllReduce, 2, 1 << 20, 0.001, true);
        assert!(close(r.time_us, 1000.0));
        assert!(close(r.algbw_gbps, (1u64 << 20) as f64 / 0.001 / 1e9));
        assert!(close(r.busbw_gbps, r.algbw_gbps));
        let json = serde_json::to_value(&r).expect("serialises");
        let mut keys: Vec<&str> = json
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["algbw_gbps", "busbw_gbps", "bytes", "correct", "time_us"]
        );
    }

    #[test]
    fn sizes_double_and_round() {
        assert_eq!(
            sizes(8, 64, 2, CollectiveOp::AllReduce, 2),
            vec![8, 16, 32, 64]
        );
        // All-gather of 8 ranks of BF16: at least one element per rank.
        assert_eq!(sizes(8, 32, 2, CollectiveOp::AllGather, 8), vec![16, 32]);
    }

    #[test]
    fn host_backend_is_correct() {
        let group = HostCollective::group(3, Duration::from_secs(10));
        let plan = Plan {
            ops: ops(OpArg::All),
            min: 8,
            max: 256,
            iters: 2,
            warmup: 1,
            dtype: DType::BF16,
            pipelined: false,
        };
        let results: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .enumerate()
                .map(|(r, comm)| {
                    let plan = &plan;
                    scope.spawn(move || {
                        let env = RankEnv {
                            comm: Arc::new(comm),
                            mem: HostMemory::new(DeviceId(r as u32), 1 << 20),
                        };
                        run_rank(&env, plan).expect("rank runs")
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for rank in results {
            assert_eq!(rank.len(), 4);
            for (op, rows) in rank {
                assert!(!rows.is_empty(), "{op:?}");
                assert!(rows.iter().all(|r| r.correct), "{op:?}: {rows:?}");
            }
        }
    }
}
