//! The rank link of static mode (P5 S-5, contract §24): the [`Transport`] extension point
//! (`rank_transport`) the rank runtime opens its leader listener and worker connections through,
//! selected by `parallel.ranks.transport`. Phase 5 registers one module, `tcp`; the deferred
//! multi-node phase adds others (and generalises the `SocketAddr` addresses).
//!
//! A transport only moves bytes: framing (u32 LE length + postcard body), the `Hello` checks,
//! the join deadline and the backoff between connect attempts stay in [`crate::rank`].

pub mod conformance;
pub mod tcp;

use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::time::Duration;

use turbine_core::registry::{Module, Registry, UnknownModule};

pub use tcp::TcpTransport;

/// A registered rank transport: opens the leader's listener and a worker's connection.
pub trait Transport: Module {
    /// Binds the leader's listener on `addr` (port 0 picks a free one; see
    /// [`RankListener::local_addr`]).
    fn listen(&self, addr: SocketAddr) -> io::Result<Box<dyn RankListener>>;
    /// Connects to the listener at `addr`, failing no later than `timeout` (plus scheduling
    /// slack) when nothing accepts there.
    fn connect(&self, addr: SocketAddr, timeout: Duration) -> io::Result<Box<dyn RankStream>>;
}

/// The leader's side of the bootstrap: accepts worker connections.
pub trait RankListener: Send {
    /// The address actually bound.
    fn local_addr(&self) -> io::Result<SocketAddr>;
    /// Waits at most `timeout` for one connection: `Ok(None)` when none arrived in time. The
    /// returned stream is blocking, without a read timeout.
    fn accept(&self, timeout: Duration) -> io::Result<Option<Box<dyn RankStream>>>;
}

/// One bidirectional, ordered, reliable byte stream between two ranks. A closed peer reads as
/// EOF (`Ok(0)`) or an error, never as a hang.
pub trait RankStream: Read + Write + Send {
    /// Another handle on the same connection (a reader thread, a control handle for shutdown).
    fn try_clone(&self) -> io::Result<Box<dyn RankStream>>;
    /// Bounds every later read of this connection (`None`: block); an expired read fails with
    /// `WouldBlock` or `TimedOut`.
    fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()>;
    /// Closes both directions: reads blocked on any handle of this connection return, and the
    /// peer sees EOF.
    fn shutdown(&self) -> io::Result<()>;
}

static TCP: TcpTransport = TcpTransport;

static RANK_TRANSPORTS: Registry<dyn Transport> = Registry::new("rank_transport", &[&TCP]);

/// The registered rank transports, in registration order.
pub fn registry() -> &'static Registry<dyn Transport> {
    &RANK_TRANSPORTS
}

/// The transport named by `parallel.ranks.transport`, logging `event="module_selected"`.
pub fn select(name: &str) -> Result<&'static dyn Transport, UnknownModule> {
    registry().select(name, "parallel.ranks.transport")
}
