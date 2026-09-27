//! The windowed decode step time behind `step_time_drift` and the circuit's latency drift
//! (P3 S-6, S-12). The engine and the overload simulator feed it the same way, so the drift the
//! simulator tests is the drift the server reports.

use std::collections::VecDeque;

/// Pure decode iterations the window keeps.
pub const STEP_WINDOW: usize = 64;

/// The step time of the last [`STEP_WINDOW`] pure decode iterations (no prefill in the batch).
#[derive(Clone, Debug, Default)]
pub struct DecodeStepWindow {
    steps: VecDeque<f64>,
}

impl DecodeStepWindow {
    pub fn new() -> Self {
        DecodeStepWindow {
            steps: VecDeque::with_capacity(STEP_WINDOW),
        }
    }

    /// One executed iteration of `secs` with `prefill_tokens` and `decode_tokens`. Iterations
    /// with a prefill are skipped, so a prefill is not mistaken for slowing down. The step time
    /// is the iteration's, the time every sequence in it waits for its token: decode is
    /// memory-bound, so dividing by the batch would read a shrinking batch as drift.
    pub fn observe(&mut self, prefill_tokens: u32, decode_tokens: u32, secs: f64) {
        if decode_tokens == 0 || prefill_tokens > 0 {
            return;
        }
        if self.steps.len() == STEP_WINDOW {
            self.steps.pop_front();
        }
        self.steps.push_back(secs);
    }

    /// The window's p95 (`None` before the first pure decode iteration).
    pub fn p95(&self) -> Option<f64> {
        if self.steps.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.steps.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        Some(sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2026-09-27 soak: a decode iteration of a 3B model on an R9700 takes about the same
    /// time at batch 1 and batch 16 (memory-bound), so the calibration run's batch shrinking
    /// from 4 to 1 read as a 4× slowdown and put the circuit in DEGRADED (`latency_drift`).
    /// Catches: the step time divided by the batch size, which turns a shrinking batch into
    /// drift and a growing one into a speed-up.
    #[test]
    fn batch_size_is_not_drift() {
        let mut w = DecodeStepWindow::new();
        assert_eq!(w.p95(), None);
        for _ in 0..STEP_WINDOW {
            w.observe(0, 16, 0.018);
        }
        assert_eq!(w.p95(), Some(0.018));
        for _ in 0..STEP_WINDOW {
            w.observe(0, 1, 0.018);
        }
        assert_eq!(w.p95(), Some(0.018), "same iteration time at batch 1");
        // A prefill in the batch is not a decode step; a slower decode step is.
        w.observe(2048, 1, 0.5);
        assert_eq!(w.p95(), Some(0.018));
        for _ in 0..STEP_WINDOW {
            w.observe(0, 4, 0.054);
        }
        assert_eq!(w.p95(), Some(0.054));
    }
}
