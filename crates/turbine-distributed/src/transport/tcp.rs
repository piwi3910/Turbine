//! `tcp`: the rank link over `std::net` TCP (the Phase 5 static-mode bootstrap).

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use turbine_core::registry::Module;

use super::{RankListener, RankStream, Transport};

/// Polling interval of [`TcpRankListener::accept`] while no connection is pending.
const ACCEPT_POLL: Duration = Duration::from_millis(5);

/// The `tcp` module of the `rank_transport` registry.
pub struct TcpTransport;

impl Module for TcpTransport {
    fn name(&self) -> &'static str {
        "tcp"
    }
}

impl Transport for TcpTransport {
    fn listen(&self, addr: SocketAddr) -> io::Result<Box<dyn RankListener>> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        Ok(Box::new(TcpRankListener(listener)))
    }

    fn connect(&self, addr: SocketAddr, timeout: Duration) -> io::Result<Box<dyn RankStream>> {
        let stream = TcpStream::connect_timeout(&addr, timeout.max(Duration::from_millis(1)))?;
        Ok(Box::new(TcpRankStream(stream)))
    }
}

/// A non-blocking listener polled until each `accept`'s timeout.
struct TcpRankListener(TcpListener);

impl RankListener for TcpRankListener {
    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.local_addr()
    }

    fn accept(&self, timeout: Duration) -> io::Result<Option<Box<dyn RankStream>>> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.0.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false)?;
                    return Ok(Some(Box::new(TcpRankStream(stream))));
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Ok(None);
                    }
                    std::thread::sleep(ACCEPT_POLL.min(left));
                }
                Err(e) => return Err(e),
            }
        }
    }
}

struct TcpRankStream(TcpStream);

impl Read for TcpRankStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
}

impl Write for TcpRankStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl RankStream for TcpRankStream {
    fn try_clone(&self) -> io::Result<Box<dyn RankStream>> {
        Ok(Box::new(TcpRankStream(self.0.try_clone()?)))
    }
    fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_read_timeout(timeout)
    }
    fn shutdown(&self) -> io::Result<()> {
        self.0.shutdown(Shutdown::Both)
    }
}
