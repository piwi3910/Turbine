# Adding a rank transport

A rank transport carries the static-mode link between the ranks of one tensor-parallel group (Phase 5, S-5): worker processes connect to the leader, each sends its `Hello`, the leader answers `Welcome { unique_id }` or `Reject`, each worker reports its memory budget once it loaded (`Loaded`), and from then on the leader streams `StepPlan`s (protocol v3: with its mirror-ledger changes) and either side may send `Shutdown`. Everything around the bytes stays in the rank runtime (`crates/turbine-distributed/src/rank.rs`): the frame format (u32 little-endian length plus a postcard body, at most 16 MiB), the `Hello` checks, the join deadline (`parallel.collective.init_timeout`), the connect backoff (50 ms doubling to 1 s), the bounded plan queue and the reaction to a lost peer. Point name `rank_transport`; selected by `parallel.ranks.transport` (default `tcp`). Addresses are `SocketAddr` (`parallel.ranks.leader`) in Phase 5; the deferred multi-node phase generalises them.

## The traits

`turbine_distributed::transport` (`crates/turbine-distributed/src/transport/mod.rs`):

| Item                                 | Must do                                                                                                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Transport::name()` (from `Module`)  | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry.                                                                                                                |
| `Transport::listen(addr)`            | Binds the leader's `RankListener` on `addr`; port 0 picks a free port.                                                                                                              |
| `Transport::connect(addr, timeout)`  | A `RankStream` to the listener at `addr`, or an error no later than `timeout` (the runtime retries with backoff until the join deadline).                                           |
| `RankListener::local_addr()`         | The address actually bound.                                                                                                                                                         |
| `RankListener::accept(timeout)`      | One incoming connection, or `Ok(None)` once `timeout` passed with none; the stream it returns is blocking, without a read timeout.                                                  |
| `RankStream` (`Read + Write + Send`) | An ordered, reliable byte stream; a closed peer reads as EOF (`Ok(0)`) or an error, never as a hang.                                                                                |
| `RankStream::try_clone()`            | Another handle on the same connection: the leader keeps a writer, a reader thread and a control handle per worker.                                                                  |
| `RankStream::set_read_timeout(t)`    | Bounds later reads (`None`: block); an expired read fails with `WouldBlock` or `TimedOut` and consumes nothing. The handshake reads under the join deadline, the step loop without. |
| `RankStream::shutdown()`             | Closes both directions: a read blocked on any handle of the connection returns, the peer sees EOF. This is how a dropped leader makes every worker abort its communicator.          |

## Files to add

One file: `crates/turbine-distributed/src/transport/<name>.rs` (see `crates/turbine-distributed/src/transport/tcp.rs`):

```rust
//! `uds`: the rank link over Unix domain sockets on one host (a sketch).

use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use turbine_core::registry::Module;

use super::{RankListener, RankStream, Transport};

pub struct UdsTransport;

impl Module for UdsTransport {
    fn name(&self) -> &'static str {
        "uds"
    }
}

impl Transport for UdsTransport {
    fn listen(&self, addr: SocketAddr) -> io::Result<Box<dyn RankListener>> {
        todo!("bind a socket path derived from {addr}")
    }
    fn connect(&self, addr: SocketAddr, timeout: Duration) -> io::Result<Box<dyn RankStream>> {
        todo!("connect to the path of {addr} within {timeout:?}")
    }
}
```

A transport over an RDMA or vendor library keeps its `unsafe` in the crate's one allowlisted FFI module (contract §1.3; `crates/turbine-kernels/tests/unsafe_isolation.rs` enforces it) and loads the library at run time, never at link time. Run `cargo fmt --all` after adding the file.

## Registry entry

In `crates/turbine-distributed/src/transport/mod.rs`: add `pub mod <name>;` next to `pub mod tcp;`, a `static` of the new type, and append it to `RANK_TRANSPORTS` (returned by `registry()`):

```rust
static RANK_TRANSPORTS: Registry<dyn Transport> = Registry::new("rank_transport", &[&TCP, &UDS]);
```

Then add the name to the pinned list in `registry_conformance::rank_transports` (`crates/turbine-distributed/src/lib.rs`). The server validates `parallel.ranks.transport` against `registry().names()` (`crates/turbine-server/src/modules.rs`) before any port is bound; `turbine_distributed::transport::select(name)` returns the module and logs `event="module_selected"`, and `RankRuntime::static_leader` / `static_worker` take it as their first argument. Keep `tcp` first; it is the configuration default.

## Conformance suite

`check` (`crates/turbine-distributed/src/transport/conformance.rs`) runs every registered transport on loopback: listen on port 0 and report the bound port; an idle `accept` answers `None` within its timeout; connect and accept, then carry a small and a 1 MiB rank frame both ways and through clones of either end; a 50 ms read timeout bounds an idle read without inventing data; `shutdown` on one handle returns a read blocked on another handle and shows the peer EOF; a connect to the closed port fails within its timeout. `conformance_rejects_broken_transport` proves the suite refuses a transport that swallows frames. The rank runtime's own tests (handshake, rejections, join timeout, leader loss) run over the `tcp` module.

- `scripts/remote-cargo.sh test -p turbine-distributed registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-distributed transport::conformance::tests` — the suite catches a broken transport.
- `scripts/remote-cargo.sh test -p turbine-distributed rank::tests` — the static bootstrap over the registered `tcp` transport.

## Lab checks

A transport is proven between real rank processes: the `static`-mode leg of `scripts/lab-cluster.sh tp2-novanas` (phase-5 plan Task 17; ranks 0 and 1, leader `127.0.0.1:18100`), run with `parallel.ranks.transport: <name>` in both rank configurations, must join both ranks, meet the Phase 1 golden tolerance at `parallel.tensor_parallel_size: 2`, and stop both processes cleanly when the leader is killed (workers report `leader_lost` and exit, no hang). Step-plan traffic is small next to the collectives, so a transport needs no throughput gate of its own, but the TP throughput must stay within 3% tok/s of the `tcp` run. Both GPUs must be free and the lab rules of `AGENTS.md` apply.

## Pitfalls

- A read that can block forever hangs the whole group: `shutdown` must wake every handle of the connection, including one blocked in another thread.
- `accept` must hand back a blocking stream; a non-blocking one makes every later frame read fail with `WouldBlock`.
- `set_read_timeout` must not consume or reorder bytes on expiry; the handshake then retries or gives up on a clean frame boundary.
- Never buffer writes past `flush`: `write_frame` flushes once per frame, and a plan left in a buffer stalls every rank.
- Addresses stay `SocketAddr` in Phase 5: a transport that needs another address form waits for the multi-node phase to generalise `parallel.ranks.leader`.
