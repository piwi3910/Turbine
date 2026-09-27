//! Captures the `tracing` output of a closure so tests can assert on logged events.

use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Runs `f` under a thread-local subscriber at TRACE and returns its result and everything it logged.
pub fn capture<R>(f: impl FnOnce() -> R) -> (R, String) {
    let buf = Buf::default();
    let writer = buf.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let r = tracing::subscriber::with_default(subscriber, f);
    let text = String::from_utf8_lossy(&buf.0.lock().unwrap()).into_owned();
    (r, text)
}
