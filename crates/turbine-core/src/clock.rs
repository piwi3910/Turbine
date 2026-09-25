//! Injectable time source (contract §3.7). The scheduler and every timeout read time only
//! through `Clock`, so the Phase 2 simulator can run on virtual time with no sleeps.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

/// Monotonic and wall-clock time.
pub trait Clock: Send + Sync {
    /// Monotonic time since an arbitrary fixed origin (the clock's creation for `SystemClock`).
    fn now_mono(&self) -> Duration;
    /// Wall-clock time, for timestamps shown to users (OpenAI `created`, logs).
    fn now_wall(&self) -> SystemTime;
}

/// The real clock: `Instant` for monotonic time, `SystemTime` for wall time.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        SystemClock {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_mono(&self) -> Duration {
        self.origin.elapsed()
    }
    fn now_wall(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// A manually driven clock for tests and the simulator. Clones share the same time.
/// Wall time is `UNIX_EPOCH + now_mono()`.
#[derive(Clone, Debug, Default)]
pub struct FakeClock {
    now: Arc<Mutex<Duration>>,
}

impl FakeClock {
    pub fn new(start: Duration) -> Self {
        FakeClock {
            now: Arc::new(Mutex::new(start)),
        }
    }

    /// Move time forward by `d`.
    pub fn advance(&self, d: Duration) {
        let mut now = self.lock();
        *now += d;
    }

    /// Jump to `t` (may move backwards; the simulator never does).
    pub fn set(&self, t: Duration) {
        *self.lock() = t;
    }

    // A plain `Duration` cannot be left half-written, so a poisoned lock is safe to reuse.
    fn lock(&self) -> MutexGuard<'_, Duration> {
        self.now.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Clock for FakeClock {
    fn now_mono(&self) -> Duration {
        *self.lock()
    }
    fn now_wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + *self.lock()
    }
}
