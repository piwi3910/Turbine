//! The behaviour every registered rank transport must show, run over the registry by
//! `registry_conformance` (the naming rules are `turbine_core::registry::conformance`). Every
//! check runs on loopback (`127.0.0.1`), so it needs no network beyond the build host.

use std::io::{self, Read};
use std::net::SocketAddr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::{RankStream, Transport};
use crate::rank::{RankMessage, read_frame, write_frame};

/// Slack allowed over a requested bound before a wait counts as a hang.
const SLACK: Duration = Duration::from_secs(2);

fn loopback() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

/// `Err` naming the first problem. A transport must:
/// - listen on loopback port 0 and report the bound port;
/// - answer an `accept` with nothing pending by `None` within its timeout;
/// - connect, accept, and carry rank frames both ways, through clones of either end;
/// - bound a read by `set_read_timeout` (`WouldBlock` / `TimedOut`, no data invented);
/// - make `shutdown` on one handle return a read blocked on another handle of that connection,
///   and show the peer EOF or an error;
/// - fail a connect to an address nothing listens on within its timeout.
pub fn check(transport: &dyn Transport) -> Result<(), String> {
    let name = transport.name();
    let fail = |what: &str, e: &dyn std::fmt::Display| format!("{name}: {what}: {e}");

    // Listen, and nothing pending.
    let listener = transport
        .listen(loopback())
        .map_err(|e| fail("listen on 127.0.0.1:0", &e))?;
    let addr = listener.local_addr().map_err(|e| fail("local_addr", &e))?;
    if addr.port() == 0 {
        return Err(format!("{name}: local_addr reports port 0"));
    }
    let t = Instant::now();
    match listener.accept(Duration::from_millis(50)) {
        Ok(None) => {}
        Ok(Some(_)) => return Err(format!("{name}: accept returned a phantom connection")),
        Err(e) => return Err(fail("accept with nothing pending", &e)),
    }
    if t.elapsed() > Duration::from_millis(50) + SLACK {
        return Err(format!("{name}: an idle accept took {:?}", t.elapsed()));
    }

    // Connect and accept (the connect runs on its own thread: a transport may complete it only
    // once the listener accepts).
    let (tx, rx) = mpsc::channel();
    let dialer = std::thread::scope(|s| {
        s.spawn(move || {
            let _ = tx.send(transport.connect(addr, Duration::from_secs(5)));
        });
        let accepted = listener.accept(Duration::from_secs(5));
        let dialed = rx
            .recv_timeout(Duration::from_secs(5) + SLACK)
            .map_err(|_| format!("{name}: connect did not return"));
        (accepted, dialed)
    });
    let mut server = match dialer.0 {
        Ok(Some(s)) => s,
        Ok(None) => return Err(format!("{name}: accept saw no connection")),
        Err(e) => return Err(fail("accept", &e)),
    };
    let mut client = dialer.1?.map_err(|e| fail("connect", &e))?;

    // Frames both ways, also through clones.
    round_trip(name, client.as_mut(), server.as_mut())?;
    let mut server_reader = server.try_clone().map_err(|e| fail("try_clone", &e))?;
    let mut client_writer = client.try_clone().map_err(|e| fail("try_clone", &e))?;
    round_trip(name, client_writer.as_mut(), server_reader.as_mut())?;
    round_trip(name, server.as_mut(), client.as_mut())?;

    // A read timeout bounds a read on an idle connection.
    server
        .set_read_timeout(Some(Duration::from_millis(50)))
        .map_err(|e| fail("set_read_timeout", &e))?;
    let t = Instant::now();
    let mut byte = [0u8; 1];
    match server.read(&mut byte) {
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) => {}
        Ok(n) => {
            return Err(format!(
                "{name}: an idle read with a timeout returned {n} bytes"
            ));
        }
        Err(e) => return Err(fail("an idle read with a timeout", &e)),
    }
    if t.elapsed() > Duration::from_millis(50) + SLACK {
        return Err(format!(
            "{name}: a 50 ms read timeout took {:?}",
            t.elapsed()
        ));
    }
    server
        .set_read_timeout(None)
        .map_err(|e| fail("set_read_timeout(None)", &e))?;

    // Shutdown through a control handle returns a read blocked on another handle, and the peer
    // sees EOF or an error.
    let control = server.try_clone().map_err(|e| fail("try_clone", &e))?;
    let (done_tx, done_rx) = mpsc::channel();
    let blocked = std::thread::spawn(move || {
        let r = read_frame(&mut server_reader);
        let _ = done_tx.send(());
        r.is_err()
    });
    std::thread::sleep(Duration::from_millis(50));
    control.shutdown().map_err(|e| fail("shutdown", &e))?;
    if done_rx.recv_timeout(SLACK).is_err() {
        return Err(format!(
            "{name}: shutdown did not return a read blocked on another handle"
        ));
    }
    if !blocked.join().unwrap_or(false) {
        return Err(format!("{name}: a read across shutdown returned a frame"));
    }
    client
        .set_read_timeout(Some(SLACK))
        .map_err(|e| fail("set_read_timeout", &e))?;
    match client.read(&mut byte) {
        Ok(0) => {}
        Err(e)
            if !matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) => {}
        Ok(n) => return Err(format!("{name}: a closed peer read {n} bytes")),
        Err(_) => return Err(format!("{name}: a closed peer reads as a timeout, not EOF")),
    }
    drop((server, client, client_writer, control, listener));

    // Nothing listens at `addr` any more: a connect fails within its timeout.
    let t = Instant::now();
    let timeout = Duration::from_millis(200);
    if transport.connect(addr, timeout).is_ok() {
        return Err(format!("{name}: connected to {addr} with no listener"));
    }
    if t.elapsed() > timeout + SLACK {
        return Err(format!(
            "{name}: a connect bounded by {timeout:?} took {:?}",
            t.elapsed()
        ));
    }
    Ok(())
}

/// One `Hello`-sized and one 1 MiB frame from `a` to `b`, read back intact.
fn round_trip(
    name: &str,
    mut a: &mut dyn RankStream,
    mut b: &mut dyn RankStream,
) -> Result<(), String> {
    let msgs = [
        RankMessage::Reject {
            reason: "conformance".into(),
        },
        RankMessage::Shutdown {
            reason: "x".repeat(1 << 20),
        },
    ];
    std::thread::scope(|s| {
        let writer = s.spawn(|| {
            for m in &msgs {
                write_frame(&mut a, m)?;
            }
            Ok::<_, io::Error>(())
        });
        for m in &msgs {
            match read_frame(&mut b) {
                Ok(got) if &got == m => {}
                Ok(_) => return Err(format!("{name}: a frame arrived altered")),
                Err(e) => return Err(format!("{name}: reading a frame: {e}")),
            }
        }
        writer
            .join()
            .map_err(|_| format!("{name}: the writer panicked"))?
            .map_err(|e| format!("{name}: writing a frame: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::sync::Mutex;

    use turbine_core::registry::Module;

    use super::super::RankListener;
    use super::*;

    /// A transport whose streams swallow writes and read nothing: the suite must refuse it.
    struct Sink;
    impl Module for Sink {
        fn name(&self) -> &'static str {
            "sink"
        }
    }
    struct SinkListener(Mutex<bool>);
    struct SinkStream;
    impl Read for SinkStream {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Ok(0)
        }
    }
    impl Write for SinkStream {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl RankStream for SinkStream {
        fn try_clone(&self) -> io::Result<Box<dyn RankStream>> {
            Ok(Box::new(SinkStream))
        }
        fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
            Ok(())
        }
        fn shutdown(&self) -> io::Result<()> {
            Ok(())
        }
    }
    impl RankListener for SinkListener {
        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok(SocketAddr::from(([127, 0, 0, 1], 9)))
        }
        fn accept(&self, _: Duration) -> io::Result<Option<Box<dyn RankStream>>> {
            // The first (idle) accept sees nothing, every later one a connection.
            let first = std::mem::replace(&mut *self.0.lock().unwrap(), false);
            Ok((!first).then(|| Box::new(SinkStream) as Box<dyn RankStream>))
        }
    }
    impl Transport for Sink {
        fn listen(&self, _: SocketAddr) -> io::Result<Box<dyn RankListener>> {
            Ok(Box::new(SinkListener(Mutex::new(true))))
        }
        fn connect(&self, _: SocketAddr, _: Duration) -> io::Result<Box<dyn RankStream>> {
            Ok(Box::new(SinkStream))
        }
    }

    #[test]
    fn conformance_rejects_broken_transport() {
        let err = check(&Sink).expect_err("a transport that loses frames");
        assert!(err.starts_with("sink: "), "{err}");
    }
}
