//! Pipelined mode of the simulator (Phase 5 S-10): the real `Scheduler` with up to
//! `micro_batches` plans in flight through `stages` pipeline stages on virtual time.
//!
//! A plan's cost-model duration is split evenly over the stages (to the nanosecond, so the
//! stages add up to the serial duration). A micro-batch enters stage 0 when stage 0 is free
//! and fewer than `micro_batches` plans are in flight, then flows stage 0 → `stages − 1`,
//! waiting for each stage to be free, so stage `s` runs micro-batch `k + 1` while stage
//! `s + 1` runs `k`. Its tokens are sampled when it leaves the last stage; only then is it
//! completed in the scheduler and may its sequences be planned again. With one micro-batch
//! the plans are the serial loop's ([`Simulation::run`]), timing included.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use serde::Serialize;
use turbine_core::clock::Clock;
use turbine_core::types::SeqId;

use super::{Hook, SimReport, Simulation};
use crate::pipeline::StageTimeline;
use crate::request::RequestState;
use crate::scheduler::{BatchKind, IterationOutcome, IterationPlan};

/// What a pipelined run produced beside the serial report.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PipelineReport {
    pub sim: SimReport,
    pub stages: u32,
    pub micro_batches: u32,
    /// Every micro-batch that left the last stage, in that order.
    pub completed_micro_batches: Vec<MicroBatchTrace>,
    /// Busy intervals `[start_s, end_s)` of every stage.
    pub stage_busy: Vec<Vec<(f64, f64)>>,
    pub max_micro_batches_in_flight: u32,
    /// Virtual time the run ended.
    pub end_s: f64,
}

/// One micro-batch's trip through the pipeline.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MicroBatchTrace {
    pub iteration: u64,
    /// Its batch items' sequences, then its forks' destinations.
    pub seqs: Vec<u64>,
    /// Entered stage 0.
    pub enter_s: f64,
    /// Left the last stage: its tokens were sampled and the plan completed.
    pub exit_s: f64,
    pub tokens: u32,
}

impl PipelineReport {
    /// Tokens sampled per second over `[from_s, to_s)`.
    pub fn tokens_per_s(&self, from_s: f64, to_s: f64) -> f64 {
        let tokens: u64 = self
            .completed_micro_batches
            .iter()
            .filter(|b| b.exit_s >= from_s && b.exit_s < to_s)
            .map(|b| u64::from(b.tokens))
            .sum();
        tokens as f64 / (to_s - from_s)
    }

    /// `turbine_pipeline_bubble_ratio` over `[from_s, to_s)`: the idle share of stage time,
    /// computed by the engine's [`StageTimeline`].
    pub fn bubble_ratio(&self, from_s: f64, to_s: f64) -> f64 {
        let mut t = StageTimeline::with_window(self.stage_busy.len(), Duration::MAX);
        for (s, busy) in self.stage_busy.iter().enumerate() {
            for &(a, b) in busy {
                t.record(s, Duration::from_secs_f64(a), Duration::from_secs_f64(b));
            }
        }
        t.bubble_ratio_between(
            Duration::from_secs_f64(from_s),
            Duration::from_secs_f64(to_s),
        )
    }
}

/// A micro-batch in the pipeline.
struct MicroBatch {
    plan: IterationPlan,
    /// Per-stage durations; they add up to the plan's cost-model duration.
    durations: Vec<Duration>,
    entered: Duration,
}

/// `d` split over `n` stages to the nanosecond.
fn split(d: Duration, n: usize) -> Vec<Duration> {
    let total = d.as_nanos();
    let at = |k: usize| total * k as u128 / n as u128;
    (0..n)
        .map(|k| Duration::from_nanos((at(k + 1) - at(k)) as u64))
        .collect()
}

fn plan_seqs(plan: &IterationPlan) -> impl Iterator<Item = SeqId> + '_ {
    plan.items
        .iter()
        .map(|i| i.seq)
        .chain(plan.forks.iter().map(|f| f.dst))
}

impl Simulation {
    /// Keep up to `m` plans in flight (`Scheduler::with_micro_batches`); set before
    /// [`Self::run_pipelined`].
    pub fn with_micro_batches(mut self, m: u32) -> Simulation {
        self.sched = self.sched.with_micro_batches(m);
        self
    }

    /// Run through `stages` pipeline stages until virtual time `until`, or until no arrival,
    /// hook or runnable work remains (module docs).
    pub fn run_pipelined(&mut self, until: Duration, stages: u32) -> PipelineReport {
        let n = stages.max(1) as usize;
        let m = self.sched.micro_batches() as usize;
        let mut report = PipelineReport {
            stages: n as u32,
            micro_batches: m as u32,
            stage_busy: vec![Vec::new(); n],
            ..PipelineReport::default()
        };
        let mut in_flight: HashMap<u64, MicroBatch> = HashMap::new();
        // Per stage: the micro-batch it runs and when it ends; micro-batches waiting for it.
        let mut running: Vec<Option<(u64, Duration)>> = vec![None; n];
        let mut waiting: Vec<VecDeque<u64>> = vec![VecDeque::new(); n];
        // Sequences in a plan in flight, and consecutive plans a decodable one was left out.
        let mut seq_in_flight: HashMap<SeqId, u64> = HashMap::new();
        let mut left_out: HashMap<SeqId, u32> = HashMap::new();
        loop {
            let now = self.clock.now_mono();
            if now >= until {
                break;
            }
            self.submit_arrivals(now);
            let mut idle_plan = false;
            if running[0].is_none() && in_flight.len() < m {
                let next = self.sched.snapshot().iterations_total + 1;
                for hook in self.hooks.remove(&next).unwrap_or_default() {
                    match hook {
                        Hook::Cancel(id, reason) => self.sched.cancel(id, reason),
                        Hook::Pause(seq) => self.sched.pause(seq),
                        Hook::Resume(seq) => self.sched.resume(seq),
                    }
                }
                let mut decodable: Vec<SeqId> = self
                    .seqs
                    .keys()
                    .copied()
                    .filter(|s| self.sched.seq_state(*s) == Some(RequestState::Decoding))
                    .collect();
                decodable.sort();
                let plan = self.sched.plan(&mut self.pool, &self.limits);
                if m == 1 {
                    self.check(&plan, &decodable);
                } else {
                    decodable.retain(|s| !seq_in_flight.contains_key(s));
                    self.check(&plan, &[]);
                    self.check_micro_batch(&plan, &decodable, &mut left_out);
                }
                for (id, _) in &plan.dropped {
                    self.forget(*id);
                }
                if plan.is_empty() {
                    self.arrivals.on_done(plan.dropped.len() as u64, now);
                    self.sched.complete(
                        &mut self.pool,
                        IterationOutcome {
                            iteration: plan.iteration,
                            ..IterationOutcome::default()
                        },
                    );
                    if !plan.dropped.is_empty() || !plan.preempted.is_empty() {
                        self.record(&plan, now, Duration::ZERO);
                    }
                    idle_plan = true;
                } else {
                    let duration = self.exec.duration(&plan);
                    self.record(&plan, now, duration);
                    for seq in plan_seqs(&plan) {
                        if let Some(other) = seq_in_flight.insert(seq, plan.iteration) {
                            self.report.violations.push(format!(
                                "iteration {}: seq {} is also in iteration {other}, in flight",
                                plan.iteration, seq.0
                            ));
                        }
                    }
                    let durations = split(duration, n);
                    running[0] = Some((plan.iteration, now + durations[0]));
                    report.stage_busy[0]
                        .push((now.as_secs_f64(), (now + durations[0]).as_secs_f64()));
                    in_flight.insert(
                        plan.iteration,
                        MicroBatch {
                            plan,
                            durations,
                            entered: now,
                        },
                    );
                    report.max_micro_batches_in_flight = report
                        .max_micro_batches_in_flight
                        .max(in_flight.len() as u32);
                    if self.sched.micro_batches_in_flight() != in_flight.len() {
                        self.report.violations.push(format!(
                            "scheduler reports {} micro-batches in flight, the pipeline holds {}",
                            self.sched.micro_batches_in_flight(),
                            in_flight.len()
                        ));
                    }
                }
            }

            let Some(t) = running.iter().flatten().map(|r| r.1).min() else {
                // Nothing executes: wait for the next arrival (the serial loop's idle step).
                match self.arrivals.peek_time() {
                    Some(t) if t < until => self.clock.set(t.max(now)),
                    Some(_) => self.clock.set(until),
                    None if !self.hooks.is_empty() && !self.sched.is_idle() => {}
                    None => break,
                }
                continue;
            };
            // An idle stage 0 wakes for an arrival before the next stage boundary.
            if idle_plan
                && let Some(a) = self.arrivals.peek_time()
                && a < t
            {
                self.clock.set(a.max(now));
                continue;
            }
            self.clock.set(t);
            // Stage boundaries at `t`, last stage first so a stage frees before its
            // predecessor hands it the next micro-batch.
            for s in (0..n).rev() {
                let Some((it, end)) = running[s] else {
                    continue;
                };
                if end != t {
                    continue;
                }
                running[s] = None;
                if s + 1 < n {
                    waiting[s + 1].push_back(it);
                    continue;
                }
                let mb = in_flight.remove(&it).expect("micro-batch in flight");
                let completed = self.report.completed;
                let outcome = self.execute(&mb.plan, t);
                let tokens: u32 = outcome.appended.iter().map(|(_, k)| *k).sum();
                report.completed_micro_batches.push(MicroBatchTrace {
                    iteration: it,
                    seqs: plan_seqs(&mb.plan).map(|s| s.0).collect(),
                    enter_s: mb.entered.as_secs_f64(),
                    exit_s: t.as_secs_f64(),
                    tokens,
                });
                self.sched.complete(&mut self.pool, outcome);
                for seq in plan_seqs(&mb.plan) {
                    seq_in_flight.remove(&seq);
                }
                let done =
                    u64::from(self.report.completed - completed) + mb.plan.dropped.len() as u64;
                self.arrivals.on_done(done, t);
            }
            for s in 1..n {
                if running[s].is_some() {
                    continue;
                }
                let Some(it) = waiting[s].pop_front() else {
                    continue;
                };
                let end = t + in_flight[&it].durations[s];
                running[s] = Some((it, end));
                report.stage_busy[s].push((t.as_secs_f64(), end.as_secs_f64()));
            }
        }
        report.end_s = self.clock.now_mono().as_secs_f64();
        report.sim = std::mem::take(&mut self.report);
        report
    }

    /// Micro-batch invariants: the plan stays within its share of the token budget, and no
    /// decodable sequence outside the plans in flight is left out of more than
    /// `micro_batches` plans in a row.
    fn check_micro_batch(
        &mut self,
        plan: &IterationPlan,
        decodable: &[SeqId],
        left_out: &mut HashMap<SeqId, u32>,
    ) {
        let p = *self.sched.params();
        let m = self.sched.micro_batches();
        if p.chunked_prefill {
            let tokens = plan.prefill_tokens() + plan.decode_tokens();
            let cap = (p.max_batch_tokens / m).max(1);
            if tokens > cap {
                self.report.violations.push(format!(
                    "iteration {}: {tokens} tokens exceed the micro-batch share {cap}",
                    plan.iteration
                ));
            }
        }
        if plan.is_empty() {
            return;
        }
        let stepped: HashSet<SeqId> = plan
            .items
            .iter()
            .filter(|i| i.kind == BatchKind::Decode)
            .map(|i| i.seq)
            .chain(plan.preempted.iter().map(|(s, _)| *s))
            .collect();
        let live: HashSet<SeqId> = decodable.iter().copied().collect();
        left_out.retain(|s, _| live.contains(s));
        for s in decodable {
            if stepped.contains(s) {
                left_out.remove(s);
                continue;
            }
            let k = left_out.entry(*s).or_default();
            *k += 1;
            if *k > m {
                self.report.violations.push(format!(
                    "iteration {}: decodable seq {} left out of {k} plans in a row",
                    plan.iteration, s.0
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stage durations add up to the plan's to the nanosecond. Catches a rounding drift that
    /// would desynchronise the one-micro-batch run from the serial loop.
    #[test]
    fn split_adds_up() {
        for d in [
            Duration::from_nanos(1),
            Duration::from_millis(20),
            Duration::from_nanos(1_000_000_007),
        ] {
            for n in 1..=7 {
                let parts = split(d, n);
                assert_eq!(parts.len(), n);
                assert_eq!(parts.iter().sum::<Duration>(), d);
            }
        }
    }
}
