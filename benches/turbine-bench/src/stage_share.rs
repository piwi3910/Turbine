//! `turbine-bench stage-share` (P5p S-1): the host share of the engine's iterations between
//! two `/metrics` scrapes. The server times every iteration in the eight
//! `turbine_engine_iteration_seconds{stage}` histograms (`crates/turbine-server/src/engine/stages.rs`);
//! the delta of each stage's `_sum` between the scrapes is that stage's seconds over the
//! interval, and `host_share` is every stage but `device_wait` (the GPU) over all eight —
//! `launch` counts as host time because its graph launch blocks while the GPU idles.
//!
//! The histograms time engine turns, which partition the engine thread's time on both loops,
//! so `host_share` is always the share of that time spent off the device wait. On the serial
//! loop the GPU idles through it, so it is the GPU's idle share too. Under overlap scheduling
//! (the default since 2026-10-07) the next iteration runs on the GPU behind most host stages:
//! `host_share` stays the engine thread's host-work share — near 1 the host sets the pace
//! (the GPU waits for launches), while the `device_wait` share is the host's slack behind the
//! GPU — and only an upper bound on the GPU's idle share. The per-iteration hidden device time
//! is `last_iteration.overlap_ms` in `/turbine/v1/scheduler`; compare host shares only between
//! runs of the same loop.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::Serialize;

/// The `stage` label values of `turbine_engine_iteration_seconds`, in engine order
/// (`Stage::ALL`); the host share excludes `device_wait`.
pub const STAGES: [&str; 8] = [
    "schedule",
    "prepare",
    "launch",
    "device_wait",
    "sample",
    "detokenize",
    "emit",
    "complete",
];

/// The stage a share is computed over; the GPU side of the iteration.
const DEVICE_STAGE: &str = "device_wait";
/// The stage whose `_count` delta is the iteration count of the interval.
const ITERATION_STAGE: &str = "schedule";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum StageShareOutput {
    Text,
    Json,
}

/// `turbine-bench stage-share --before <metrics.txt> --after <metrics.txt> [--output text|json]`
#[derive(Clone, Debug, Parser)]
#[command(
    name = "turbine-bench stage-share",
    about = "Host share of the engine iterations between two /metrics scrapes"
)]
pub struct StageShareArgs {
    /// The scrape taken before the measured interval.
    #[arg(long)]
    pub before: PathBuf,
    /// The scrape taken after it.
    #[arg(long)]
    pub after: PathBuf,
    #[arg(long, value_enum, default_value_t = StageShareOutput::Text)]
    pub output: StageShareOutput,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StageShareError {
    /// A stage's `turbine_engine_iteration_seconds_{sum,count}{stage="…"}` series is missing
    /// from a scrape; the argument names the series.
    #[error("MissingHistogram({0})")]
    MissingHistogram(String),
    /// No engine iteration lies between the two scrapes (the `schedule` count did not move,
    /// or the interval recorded no time).
    #[error("NoIterations")]
    NoIterations,
}

/// The stage split of the iterations between two scrapes (the spec's Data shape).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct StageShare {
    /// Engine iterations between the scrapes (the `schedule` count delta).
    pub iterations: u64,
    /// Seconds per stage over the interval (the `_sum` deltas).
    pub seconds: BTreeMap<String, f64>,
    /// Σ seconds of every stage except `device_wait` ÷ Σ seconds of all eight.
    pub host_share: f64,
}

/// One stage's `_sum` and `_count` from a scrape.
#[derive(Clone, Copy, Debug, Default)]
struct Series {
    sum: f64,
    count: u64,
}

/// Extracts `<metric>{stage="<s>"} <value>` fields into `stage → value`.
fn parse_series(text: &str, metric: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let Some(rest) = line.strip_prefix(metric) else {
            continue;
        };
        // Only the exact metric family with a `stage` label, not a longer name that shares
        // the prefix.
        let Some(rest) = rest.strip_prefix('{') else {
            continue;
        };
        let Some(labels_end) = rest.find('}') else {
            continue;
        };
        let stage = match rest[..labels_end].split(',').find_map(|label| {
            label
                .trim()
                .strip_prefix("stage=\"")
                .and_then(|s| s.strip_suffix('"'))
        }) {
            Some(stage) => stage.to_string(),
            None => continue,
        };
        out.insert(stage, rest[labels_end + 1..].trim().to_string());
    }
    out
}

/// The `<metric>_sum` / `<metric>_count` values per stage of one scrape.
fn scrape(text: &str) -> Result<BTreeMap<String, Series>, StageShareError> {
    let sums = parse_series(text, "turbine_engine_iteration_seconds_sum");
    let counts = parse_series(text, "turbine_engine_iteration_seconds_count");
    let mut out = BTreeMap::new();
    for stage in STAGES {
        let sum = sums
            .get(stage)
            .ok_or_else(|| missing("sum", stage))?
            .parse::<f64>()
            .map_err(|_| missing("sum", stage))?;
        let count = counts
            .get(stage)
            .ok_or_else(|| missing("count", stage))?
            .parse::<u64>()
            .map_err(|_| missing("count", stage))?;
        out.insert(stage.to_string(), Series { sum, count });
    }
    Ok(out)
}

fn missing(kind: &str, stage: &str) -> StageShareError {
    StageShareError::MissingHistogram(format!(
        "turbine_engine_iteration_seconds_{kind}{{stage=\"{stage}\"}}"
    ))
}

/// The host share of the iterations between the two scrapes.
pub fn stage_share(before: &str, after: &str) -> Result<StageShare, StageShareError> {
    let before = scrape(before)?;
    let after = scrape(after)?;
    let mut seconds = BTreeMap::new();
    let mut total = 0.0;
    let mut host = 0.0;
    for stage in STAGES {
        let delta = after[stage].sum - before[stage].sum;
        seconds.insert(stage.to_string(), delta);
        total += delta;
        if stage != DEVICE_STAGE {
            host += delta;
        }
    }
    let iterations = after[ITERATION_STAGE]
        .count
        .checked_sub(before[ITERATION_STAGE].count)
        .ok_or(StageShareError::NoIterations)?;
    if iterations == 0 || total <= 0.0 {
        return Err(StageShareError::NoIterations);
    }
    Ok(StageShare {
        iterations,
        seconds,
        host_share: host / total,
    })
}

/// The text report: the share, the iteration count and the per-stage seconds.
impl StageShare {
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "iterations: {}\nhost_share: {:.4}\n",
            self.iterations, self.host_share
        );
        for (stage, seconds) in &self.seconds {
            out.push_str(&format!("  {stage}: {seconds:.6}\n"));
        }
        out
    }
}

/// The subcommand body: read the two scrapes, report, exit 0 or 2.
pub fn run(args: &StageShareArgs) -> Result<StageShare, String> {
    let before = std::fs::read_to_string(&args.before)
        .map_err(|e| format!("cannot read {}: {e}", args.before.display()))?;
    let after = std::fs::read_to_string(&args.after)
        .map_err(|e| format!("cannot read {}: {e}", args.after.display()))?;
    stage_share(&before, &after).map_err(|e| e.to_string())
}
