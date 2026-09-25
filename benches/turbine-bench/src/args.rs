//! Command line (contract §19).

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use turbine_core::config::HumanDuration;

use crate::open_loop::RangeArg;

/// Which OpenAI route to load: `/v1/chat/completions` or `/v1/completions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EndpointArg {
    Chat,
    Completions,
}

/// Report rendering on stdout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// `turbine-bench` flags (contract §19).
#[derive(Clone, Debug, Parser)]
#[command(
    name = "turbine-bench",
    version,
    about = "Streaming load generator for OpenAI-compatible endpoints"
)]
pub struct BenchArgs {
    /// Base URL, e.g. http://127.0.0.1:8000 (http only).
    #[arg(long)]
    pub url: String,
    /// Model id; default: the first id from GET <url>/v1/models.
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, value_enum, default_value_t = EndpointArg::Chat)]
    pub endpoint: EndpointArg,
    /// Requests in flight at once.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub concurrency: u32,
    /// Total requests (ignored when --duration is given).
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..))]
    pub requests: u32,
    #[arg(long, default_value_t = 256)]
    pub prompt_words: u32,
    #[arg(long, default_value_t = 128)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Adds "ignore_eos": true (vLLM / SGLang fixed-length outputs).
    #[arg(long)]
    pub ignore_eos: bool,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub output: OutputFormat,
    /// Run for this long (`<integer><ms|s|m|h>`, e.g. `10m`) instead of --requests.
    #[arg(long, value_parser = parse_duration)]
    pub duration: Option<Duration>,
    /// Open-loop Poisson arrivals at this many requests per second (requires --duration);
    /// --concurrency then caps outstanding requests and arrivals beyond it are dropped.
    #[arg(long, value_parser = parse_rate, requires = "duration")]
    pub rate: Option<f64>,
    /// Prompt words drawn uniformly per request from `<min>..<max>` (overrides --prompt-words).
    #[arg(long)]
    pub prompt_words_range: Option<RangeArg>,
    /// max_tokens drawn uniformly per request from `<min>..<max>` (overrides --max-tokens).
    #[arg(long)]
    pub max_tokens_range: Option<RangeArg>,
    /// Poll <url>/turbine/v1/pressure once per second into this JSON-lines file.
    #[arg(long)]
    pub pressure_timeline: Option<PathBuf>,
}

/// `<integer><unit>` with unit `ms`, `s`, `m` or `h`, no space, case-sensitive, > 0 — parsed
/// by the configuration's `HumanDuration` (contract C-14).
pub fn parse_duration(s: &str) -> Result<Duration, String> {
    let d = s.parse::<HumanDuration>()?.0;
    if d.is_zero() {
        return Err(format!("duration must be > 0, got {s:?}"));
    }
    Ok(d)
}

/// A finite request rate > 0 (req/s).
pub fn parse_rate(s: &str) -> Result<f64, String> {
    let r: f64 = s.parse().map_err(|e| format!("bad rate {s:?}: {e}"))?;
    if !r.is_finite() || r <= 0.0 {
        return Err(format!("rate must be a finite number > 0, got {s:?}"));
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_parse_like_config_durations() {
        assert_eq!(parse_duration("250ms"), Ok(Duration::from_millis(250)));
        assert_eq!(parse_duration("4s"), Ok(Duration::from_secs(4)));
        assert_eq!(parse_duration("10m"), Ok(Duration::from_secs(600)));
        assert_eq!(parse_duration("4h"), Ok(Duration::from_secs(14_400)));
        for bad in ["4", "s", "4 s", "4S", "4sec", "0s", "-1s", "1.5s", ""] {
            assert!(parse_duration(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn rates_are_positive_and_finite() {
        assert_eq!(parse_rate("50"), Ok(50.0));
        assert_eq!(parse_rate("0.5"), Ok(0.5));
        for bad in ["0", "-1", "inf", "NaN", "x"] {
            assert!(parse_rate(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn rate_requires_duration() {
        let base = ["turbine-bench", "--url", "http://127.0.0.1:1"];
        assert!(BenchArgs::try_parse_from(base.iter().chain(&["--rate", "5"])).is_err());
        let a = BenchArgs::try_parse_from(base.iter().chain(&[
            "--rate",
            "5",
            "--duration",
            "2s",
            "--max-tokens-range",
            "1..4",
        ]))
        .expect("valid open-loop flags");
        assert_eq!(a.rate, Some(5.0));
        assert_eq!(a.duration, Some(Duration::from_secs(2)));
        assert_eq!(a.max_tokens_range, Some(RangeArg { min: 1, max: 4 }));
    }
}
