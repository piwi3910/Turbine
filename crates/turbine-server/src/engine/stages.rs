//! The engine iteration's stage breakdown (P2c S-1): [`StageClock`] partitions one turn of the
//! engine loop into the eight [`Stage`]s, so their durations sum to the iteration's duration.
//! One `Instant::now()` per mark and no allocation per iteration.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// One stage of an engine iteration; `as_str` is the `stage` label of
/// `turbine_engine_iteration_seconds` and the key in `stages_ms`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Taking commands, cancellations, disconnect and deadline checks, `Scheduler::plan`.
    Schedule,
    /// Fork copies, packing the batch and its host-to-device copy: the forward pass's time
    /// outside `Launch` and `DeviceWait`.
    Prepare,
    /// Host time inside the forward pass until its last kernel is enqueued.
    Launch,
    /// Stream synchronisation and the iteration's device-to-host copy.
    DeviceWait,
    /// Host sampling, including token masks.
    Sample,
    /// Incremental detokenisation and the stop-string search.
    Detokenize,
    /// `try_send` of events, pausing and cancelling on full or closed channels.
    Emit,
    /// `Scheduler::complete`, request accounting and publishing the diagnostics documents.
    Complete,
}

impl Stage {
    /// Every stage, in [`IterationStages`] order.
    pub const ALL: [Stage; 8] = [
        Stage::Schedule,
        Stage::Prepare,
        Stage::Launch,
        Stage::DeviceWait,
        Stage::Sample,
        Stage::Detokenize,
        Stage::Emit,
        Stage::Complete,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Schedule => "schedule",
            Stage::Prepare => "prepare",
            Stage::Launch => "launch",
            Stage::DeviceWait => "device_wait",
            Stage::Sample => "sample",
            Stage::Detokenize => "detokenize",
            Stage::Emit => "emit",
            Stage::Complete => "complete",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Times one iteration. [`StageClock::mark`] attributes the time since the previous mark (or
/// the start) to a stage; [`StageClock::add`] attributes part of that open span to another
/// stage ahead of the mark that closes it (the executor's launch and device-wait times inside
/// the forward pass, detokenisation inside a sampling step), so every instant is counted once.
#[derive(Clone, Debug)]
pub struct StageClock {
    last: Instant,
    /// Time of the open span already attributed by `add`.
    carried: Duration,
    stages: [Duration; 8],
}

impl StageClock {
    pub fn start() -> StageClock {
        StageClock {
            last: Instant::now(),
            carried: Duration::ZERO,
            stages: [Duration::ZERO; 8],
        }
    }

    /// Attributes the time since the previous mark, less what `add` took of it, to `stage`.
    pub fn mark(&mut self, stage: Stage) {
        let now = Instant::now();
        let span = now.saturating_duration_since(self.last);
        self.stages[stage.index()] += span.saturating_sub(self.carried);
        self.carried = Duration::ZERO;
        self.last = now;
    }

    /// Attributes `d` of the open span (the time since the previous mark) to `stage`; the next
    /// mark gets the rest.
    pub fn add(&mut self, stage: Stage, d: Duration) {
        self.stages[stage.index()] += d;
        self.carried += d;
    }

    /// The stages marked so far; time after the last mark is not counted.
    pub fn finish(self) -> IterationStages {
        IterationStages(self.stages)
    }
}

/// One iteration's stage durations, indexed in [`Stage::ALL`] order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IterationStages(pub [Duration; 8]);

impl IterationStages {
    pub fn get(&self, stage: Stage) -> Duration {
        self.0[stage.index()]
    }

    /// Milliseconds per stage name (the `stages_ms` document field).
    pub fn to_ms_map(&self) -> BTreeMap<&'static str, f64> {
        Stage::ALL
            .iter()
            .map(|&s| (s.as_str(), self.get(s).as_secs_f64() * 1000.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_partition_the_iteration() {
        let mut clock = StageClock::start();
        let begun = clock.last;
        std::thread::sleep(Duration::from_millis(2));
        clock.mark(Stage::Schedule);
        std::thread::sleep(Duration::from_millis(3));
        // 1 ms of the open span belongs to launch; prepare gets the rest.
        clock.add(Stage::Launch, Duration::from_millis(1));
        clock.mark(Stage::Prepare);
        clock.mark(Stage::Schedule);
        let end = clock.last;
        let stages = clock.finish();
        assert_eq!(stages.0.iter().sum::<Duration>(), end - begun);
        assert_eq!(stages.get(Stage::Launch), Duration::from_millis(1));
        assert!(stages.get(Stage::Prepare) >= Duration::from_millis(2));
        assert!(stages.get(Stage::Schedule) >= Duration::from_millis(2));
        assert_eq!(stages.get(Stage::Emit), Duration::ZERO);
        let map = stages.to_ms_map();
        let keys: Vec<&str> = map.keys().copied().collect();
        let mut want: Vec<&str> = Stage::ALL.iter().map(|s| s.as_str()).collect();
        want.sort_unstable();
        assert_eq!(keys, want);
        assert_eq!(map["launch"], 1.0);
    }
}
