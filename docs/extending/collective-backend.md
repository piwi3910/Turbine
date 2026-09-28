# Adding a collective backend

A collective backend reduces, gathers and broadcasts buffers across the ranks of one tensor-parallel group (Phase 5, TS §10). Everything around it stays in the rank runtime and the planner: which devices form a group, the unique-id exchange between ranks (`local` threads or the `static` TCP bootstrap), the per-step op timeout and the circuit breaker on failure. Point name `collective_backend`; selected by `parallel.collective_backend` (default `auto`: the first registered backend serving the plan's vendor, `host` when `parallel.tensor_parallel_size` is 1).

## The traits

`turbine_distributed::collective` (`crates/turbine-distributed/src/collective/mod.rs`):

| Item                                        | Must do                                                                                                                                                                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CollectiveBackend::name()` (from `Module`) | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry; also the `backend` label of `turbine_collective_*`.                                                                                         |
| `CollectiveBackend::vendors()`              | The device vendors whose memory its communicators reduce over. Empty for a host-memory backend, which the configuration refuses for GPU plans with tensor parallelism.                                           |
| `CollectiveBackend::configured_library()`   | The operator's explicit library path from the `parallel` section, if the backend has one (a failure to load it is fatal, exit 1). Default: none.                                                                 |
| `CollectiveBackend::load(explicit)`         | Loads the backend. A missing library is `CollectiveError::Unavailable { library, detail }` naming it, never a panic.                                                                                             |
| `CollectiveLibrary::unique_id()`            | A fresh 128-byte group id (made on one rank and handed to the others by the rank runtime).                                                                                                                       |
| `CollectiveLibrary::open(init)`             | Rank `init.rank` of `init.world` for that id, bounded by `init.init_timeout` (on expiry: abort, `Timeout { op: "comm_init" }`).                                                                                  |
| `Collective`                                | `all_reduce` / `reduce_scatter` (BF16 or FP32, `Sum` / `Max`), byte-wise `all_gather` / `broadcast`, `barrier`, `abort`. Every call is bounded by the op timeout; after an abort every call on every rank fails. |

Buffers are `DeviceSlice`s of the phase-1 device layer and ordering is its `StreamRef` (`native_handle()` is the vendor stream); a backend never allocates model memory and adds no GPU runtime binding of its own.

## Files to add

One file: `crates/turbine-distributed/src/collective/<name>.rs` (see `crates/turbine-distributed/src/collective/host.rs`):

```rust
//! `loopback`: a one-rank backend (a toy).

use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_core::types::Vendor;

use super::{CollectiveBackend, CollectiveError, CollectiveLibrary};

pub struct LoopbackBackend;

impl Module for LoopbackBackend {
    fn name(&self) -> &'static str {
        "loopback"
    }
}

impl CollectiveBackend for LoopbackBackend {
    fn vendors(&self) -> &'static [Vendor] {
        &[]
    }
    fn load(&self, _explicit: Option<&Path>) -> Result<Arc<dyn CollectiveLibrary>, CollectiveError> {
        todo!("a CollectiveLibrary whose open() returns a Collective")
    }
}
```

A backend that calls a C library through FFI keeps every `unsafe` block in the one allowlisted module (contract §1.3; `crates/turbine-kernels/tests/unsafe_isolation.rs` enforces it), each with a `// SAFETY:` comment naming the owner of every pointer, stream and communicator it touches. Run `cargo fmt --all` after adding the file.

## Registry entry

In `crates/turbine-distributed/src/collective/mod.rs`: add `mod <name>;` next to `pub mod host;`, a `static` of the new type, and append it to `COLLECTIVE_BACKENDS` (returned by `registry()`):

```rust
static COLLECTIVE_BACKENDS: Registry<dyn CollectiveBackend> =
    Registry::new("collective_backend", &[&HOST, &RCCL, &NCCL, &LOOPBACK]);
```

Then add the name to the pinned list in `registry_conformance::collective_backends` (`crates/turbine-distributed/src/lib.rs`). Registration order matters for `auto`: the first backend serving the plan's vendor wins. The server validates `parallel.collective_backend` against `registry().names()` (`crates/turbine-server/src/modules.rs`).

## Conformance suite

`check` (`crates/turbine-distributed/src/collective/conformance.rs`) runs over every registered backend. A device backend must answer an explicit library path that does not exist with `Unavailable` naming it (no panic, and no real vendor library is loaded on the build host). A host-memory backend must load, report its own name, make distinct ids and run a two-rank all-reduce and all-gather correctly. The NCCL-API binding (`crates/turbine-distributed/src/collective/ffi.rs`, registered twice by `crates/turbine-distributed/src/collective/nccl_api.rs` as `rccl` and `nccl`) is also tested against a stub library that `crates/turbine-distributed/build.rs` compiles with the host C compiler: every symbol resolved, the version floor, a missing symbol refused. The host backend's own tests compare every op for world sizes 1–8, FP32 and BF16 and odd sizes bit for bit against a naive reference, and check that a missing rank times out and aborts the group.

- `scripts/remote-cargo.sh test -p turbine-distributed registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-distributed collective::host::tests::ops_match_reference` — the reference every device backend is compared against.
- `scripts/remote-cargo.sh test -p turbine-distributed collective::ffi::tests` — the NCCL-API binding against the stubs.

## Lab checks

A device backend is proven on the GPUs it serves: `turbine-collbench --backend <name> --devices 0,1 --op all --max-bytes 1GiB --output json` must report `correct: true` for every size (results compared against the host backend) and a positive all-reduce bus bandwidth, under `scripts/bench-lock.sh` when the numbers are kept. Tensor-parallel serving with it must then meet the Phase 1 golden tolerance at `parallel.tensor_parallel_size: 2` before any throughput is compared. Both GPUs must be free (`amd-smi monitor`, the k3s pods) and the lab rules of `AGENTS.md` apply.

## Pitfalls

- A collective that can block forever hangs the whole group: every wait needs the op timeout, and a timeout must abort the communicator so the other ranks fail too instead of waiting.
- Reductions must be deterministic for a fixed world size (same rank order every call) or tensor-parallel outputs vary run to run.
- `all_gather` and `broadcast` move bytes; only `all_reduce` and `reduce_scatter` interpret the element type.
- Never `dlclose` a vendor library while a communicator or a copied function pointer is alive.
