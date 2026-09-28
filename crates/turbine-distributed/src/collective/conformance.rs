//! The behaviour every registered collective backend must show, run over the registry by
//! `registry_conformance` (the naming rules are `turbine_core::registry::conformance`).

use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::SystemClock;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DType, DeviceBuffer, DeviceId, DeviceMemory};

use super::{CollectiveBackend, CollectiveError, CollectiveInit, ReduceOp};

/// `Err` naming the first problem. A device backend (with vendors) must answer an explicit
/// library that does not exist with `Unavailable` naming that path — never a panic, and never
/// by loading a real vendor library on the build host; its communicators are exercised by the
/// lab (`turbine-collbench`). A host-memory backend (no vendors) must load, report its own
/// name, make distinct ids and run a two-rank all-reduce, all-gather and send/recv correctly
/// here.
pub fn check(backend: &dyn CollectiveBackend) -> Result<(), String> {
    let name = backend.name();
    if !backend.vendors().is_empty() {
        let missing = std::path::Path::new("/nonexistent/turbine-conformance/lib.so");
        return match backend.load(Some(missing)) {
            Err(CollectiveError::Unavailable { library, .. })
                if library.contains("/nonexistent/turbine-conformance") =>
            {
                Ok(())
            }
            Err(e) => Err(format!("{name}: a missing explicit library gave {e}")),
            Ok(_) => Err(format!("{name}: a missing explicit library loaded")),
        };
    }
    let lib = backend
        .load(None)
        .map_err(|e| format!("{name}: load failed with {e}"))?;
    if lib.backend() != name {
        return Err(format!("{name}: the library reports `{}`", lib.backend()));
    }
    match (lib.unique_id(), lib.unique_id()) {
        (Ok(a), Ok(b)) if a != b => {}
        (Ok(_), Ok(_)) => return Err(format!("{name}: two unique ids are equal")),
        (Err(e), _) | (_, Err(e)) => return Err(format!("{name}: unique_id failed: {e}")),
    }
    let id = lib.unique_id().map_err(|e| e.to_string())?;
    let results: Vec<Result<RankOut, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2usize)
            .map(|rank| {
                let lib = Arc::clone(&lib);
                s.spawn(move || two_rank_ops(lib, rank, id))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("rank panicked".into())))
            .collect()
    });
    for (rank, r) in results.into_iter().enumerate() {
        let (reduced, gathered, exchanged) = r.map_err(|e| format!("{name}: rank {rank}: {e}"))?;
        // Each rank receives the other's `[1 + 2·rank, 2 + 2·rank]` point to point.
        let peer = 1 - rank as u32;
        let want = [1.0 + 2.0 * peer as f32, 2.0 + 2.0 * peer as f32];
        if f32s(&exchanged) != want {
            return Err(format!("{name}: send/recv gave {:?}", f32s(&exchanged)));
        }
        if f32s(&reduced) != [4.0, 6.0] {
            return Err(format!("{name}: all_reduce gave {:?}", f32s(&reduced)));
        }
        if f32s(&gathered) != [1.0, 2.0, 3.0, 4.0] {
            return Err(format!("{name}: all_gather gave {:?}", f32s(&gathered)));
        }
    }
    Ok(())
}

/// One rank's all-reduced, all-gathered and point-to-point received bytes.
type RankOut = (Vec<u8>, Vec<u8>, Vec<u8>);

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Rank `rank` of a two-rank group over host memory: all-reduce (sum) and all-gather of
/// `[1 + 2·rank, 2 + 2·rank]`.
fn two_rank_ops(
    lib: Arc<dyn super::CollectiveLibrary>,
    rank: usize,
    id: [u8; super::UNIQUE_ID_BYTES],
) -> Result<RankOut, String> {
    let err = |e: &dyn std::fmt::Display| e.to_string();
    let comm = lib
        .open(CollectiveInit {
            rank,
            world: 2,
            unique_id: id,
            init_timeout: Duration::from_secs(10),
            op_timeout: Duration::from_secs(10),
            clock: Arc::new(SystemClock::new()),
            metrics: None,
            memory: None,
            route_max_bytes: None,
        })
        .map_err(|e| err(&e))?;
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(rank as u32), 1 << 20);
    let stream = mem.compute_stream();
    let base = rank as f32 * 2.0;
    let values: Vec<u8> = [1.0f32 + base, 2.0 + base]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let buf = DeviceBuffer::alloc(&mem, 8).map_err(|e| err(&e))?;
    let mut slice = buf.whole();
    slice.write_bytes(&values).map_err(|e| err(&e))?;
    comm.all_reduce(&mut slice, DType::F32, ReduceOp::Sum, &stream)
        .map_err(|e| err(&e))?;
    let reduced = slice.read_bytes().map_err(|e| err(&e))?;
    let send = DeviceBuffer::alloc(&mem, 8).map_err(|e| err(&e))?;
    send.whole().write_bytes(&values).map_err(|e| err(&e))?;
    let recv = DeviceBuffer::alloc(&mem, 16).map_err(|e| err(&e))?;
    let mut gathered = recv.whole();
    comm.all_gather(&send.whole(), &mut gathered, &stream)
        .map_err(|e| err(&e))?;
    let gathered = gathered.read_bytes().map_err(|e| err(&e))?;
    // Point to point: rank 0 sends then receives, rank 1 receives then sends.
    let got = DeviceBuffer::alloc(&mem, 8).map_err(|e| err(&e))?;
    let mut got_slice = got.whole();
    let peer = 1 - rank;
    if rank == 0 {
        comm.send(&send.whole(), peer, &stream)
            .map_err(|e| err(&e))?;
        comm.recv(&mut got_slice, peer, &stream)
            .map_err(|e| err(&e))?;
    } else {
        comm.recv(&mut got_slice, peer, &stream)
            .map_err(|e| err(&e))?;
        comm.send(&send.whole(), peer, &stream)
            .map_err(|e| err(&e))?;
    }
    let exchanged = got_slice.read_bytes().map_err(|e| err(&e))?;
    Ok((reduced, gathered, exchanged))
}
