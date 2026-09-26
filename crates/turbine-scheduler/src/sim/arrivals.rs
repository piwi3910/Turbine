//! Arrival processes for the simulator: seeded Poisson arrivals with a mixed length
//! distribution (ChaCha8, byte-identical for equal seeds) or a fixed script.

use std::collections::VecDeque;
use std::time::Duration;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use serde::Serialize;
use turbine_core::types::Priority;

/// One simulated request.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SimArrival {
    /// Virtual submission time.
    pub at: Duration,
    pub prompt_len: u32,
    pub max_new_tokens: u32,
    /// Tokens the "model" generates before stopping (≤ `max_new_tokens`; equal → `length`).
    pub output_len: u32,
    pub priority: Priority,
    /// Choices (`n`).
    pub n: u32,
}

impl SimArrival {
    /// A single-choice, default-priority arrival.
    pub fn new(at: Duration, prompt_len: u32, output_len: u32) -> SimArrival {
        SimArrival {
            at,
            prompt_len,
            max_new_tokens: output_len,
            output_len,
            priority: Priority::default(),
            n: 1,
        }
    }
}

/// Length distribution of Poisson arrivals: a share of long prompts, the rest short; output
/// lengths uniform. Ranges are inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LengthMix {
    pub short_prompt: (u32, u32),
    pub long_prompt: (u32, u32),
    pub long_fraction: f64,
    pub output: (u32, u32),
    pub max_new_tokens: u32,
}

impl Default for LengthMix {
    fn default() -> Self {
        LengthMix {
            short_prompt: (4, 64),
            long_prompt: (65, 600),
            long_fraction: 0.2,
            output: (1, 64),
            max_new_tokens: 64,
        }
    }
}

enum Source {
    Poisson {
        rng: Box<ChaCha8Rng>,
        rate: f64,
        mix: LengthMix,
        next: Option<SimArrival>,
        remaining: Option<u64>,
        clock: f64,
    },
    Scripted(VecDeque<SimArrival>),
}

/// A deterministic stream of arrivals in time order.
pub struct ArrivalProcess {
    source: Source,
}

impl ArrivalProcess {
    /// Poisson arrivals at `rate` per virtual second with `LengthMix::default()`, unbounded.
    pub fn poisson(rate: f64, seed: u64) -> ArrivalProcess {
        ArrivalProcess {
            source: Source::Poisson {
                rng: Box::new(ChaCha8Rng::seed_from_u64(seed)),
                rate,
                mix: LengthMix::default(),
                next: None,
                remaining: None,
                clock: 0.0,
            },
        }
    }

    /// Replace the length distribution (Poisson only; set before the first arrival is drawn).
    pub fn with_mix(mut self, new_mix: LengthMix) -> ArrivalProcess {
        if let Source::Poisson { mix, .. } = &mut self.source {
            *mix = new_mix;
        }
        self
    }

    /// Stop after `count` arrivals (Poisson only; set before the first arrival is drawn).
    pub fn with_limit(mut self, count: u64) -> ArrivalProcess {
        if let Source::Poisson { remaining, .. } = &mut self.source {
            *remaining = Some(count);
        }
        self
    }

    /// A fixed script; arrivals are replayed in time order (ties keep script order).
    pub fn scripted(mut arrivals: Vec<SimArrival>) -> ArrivalProcess {
        arrivals.sort_by_key(|a| a.at);
        ArrivalProcess {
            source: Source::Scripted(arrivals.into()),
        }
    }

    /// Time of the next arrival, if any.
    pub fn peek_time(&mut self) -> Option<Duration> {
        self.refill();
        match &self.source {
            Source::Poisson { next, .. } => next.map(|a| a.at),
            Source::Scripted(q) => q.front().map(|a| a.at),
        }
    }

    /// The next arrival at or before `t`.
    pub fn next_before(&mut self, t: Duration) -> Option<SimArrival> {
        if self.peek_time()? > t {
            return None;
        }
        match &mut self.source {
            Source::Poisson { next, .. } => next.take(),
            Source::Scripted(q) => q.pop_front(),
        }
    }

    fn refill(&mut self) {
        let Source::Poisson {
            rng,
            rate,
            mix,
            next,
            remaining,
            clock,
        } = &mut self.source
        else {
            return;
        };
        if next.is_some() || *remaining == Some(0) {
            return;
        }
        if let Some(r) = remaining {
            *r -= 1;
        }
        // Exponential inter-arrival time: −ln(1 − U) / rate.
        let u = unit(rng);
        *clock += -(1.0 - u).ln() / rate.max(f64::MIN_POSITIVE);
        let long = unit(rng) < mix.long_fraction;
        let (lo, hi) = if long {
            mix.long_prompt
        } else {
            mix.short_prompt
        };
        let prompt_len = uniform(rng, lo, hi);
        let output_len = uniform(rng, mix.output.0, mix.output.1).min(mix.max_new_tokens);
        *next = Some(SimArrival {
            at: Duration::from_secs_f64(*clock),
            prompt_len,
            max_new_tokens: mix.max_new_tokens,
            output_len,
            priority: Priority::default(),
            n: 1,
        });
    }
}

/// Uniform in [0, 1) with 53 random bits.
fn unit(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform integer in `[lo, hi]`.
fn uniform(rng: &mut ChaCha8Rng, lo: u32, hi: u32) -> u32 {
    let span = u64::from(hi.saturating_sub(lo)) + 1;
    lo + (rng.next_u64() % span) as u32
}
