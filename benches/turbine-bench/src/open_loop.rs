//! Open-loop overload mode (phase 3, S-18): seeded Poisson arrivals, per-request length ranges,
//! the status / error-code breakdown and the `/turbine/v1/pressure` timeline.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write as _;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use serde::Serialize;
use serde_json::{Value, json};

/// Seed salt separating the per-request length draws from the arrival process of the same seed.
const LENGTH_SALT: u64 = 0x6c65_6e67_7468_7321;

/// An inclusive `<min>..<max>` range, e.g. `--prompt-words-range 64..6000`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeArg {
    pub min: u32,
    pub max: u32,
}

impl FromStr for RangeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<RangeArg, String> {
        let (min, max) = s
            .split_once("..")
            .ok_or_else(|| format!("expected <min>..<max>, got {s:?}"))?;
        let parse = |v: &str| {
            v.parse::<u32>()
                .map_err(|e| format!("bad bound {v:?} in {s:?}: {e}"))
        };
        let (min, max) = (parse(min)?, parse(max)?);
        if min > max {
            return Err(format!("min {min} exceeds max {max} in {s:?}"));
        }
        Ok(RangeArg { min, max })
    }
}

impl fmt::Display for RangeArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.min, self.max)
    }
}

/// The bench's seeded PRNG for arrivals and length draws (ChaCha8).
#[derive(Clone, Debug)]
pub struct OpenLoopRng(ChaCha8Rng);

impl OpenLoopRng {
    /// The arrival process of `seed`.
    pub fn new(seed: u64) -> OpenLoopRng {
        OpenLoopRng(ChaCha8Rng::seed_from_u64(seed))
    }

    /// The length draws of request `index`: an independent ChaCha stream per request, so a
    /// request's lengths do not depend on how many requests ran before it or on drops.
    pub fn for_request(seed: u64, index: u64) -> OpenLoopRng {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ LENGTH_SALT);
        rng.set_stream(index);
        OpenLoopRng(rng)
    }

    /// Uniform in (0, 1]: 53 random mantissa bits, shifted off zero so `ln` stays finite.
    fn unit_open_closed(&mut self) -> f64 {
        let bits = self.0.next_u64() >> 11;
        // 2^53 values: (bits + 1) / 2^53 ∈ (0, 1], exact in f64.
        (bits + 1) as f64 / (1u64 << 53) as f64
    }

    /// Exponential inter-arrival time `−ln(U)/rate` of a Poisson process with `rate` req/s.
    pub fn inter_arrival(&mut self, rate: f64) -> Duration {
        let secs = -self.unit_open_closed().ln() / rate;
        Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
    }

    /// Uniform draw from the inclusive range `r` (Lemire's widening multiply).
    pub fn draw(&mut self, r: RangeArg) -> u32 {
        let span = u64::from(r.max - r.min) + 1;
        let offset = (u128::from(self.0.next_u64()) * u128::from(span)) >> 64;
        // offset < span ≤ 2^32, and min + offset ≤ max, so the cast is lossless.
        r.min + offset as u32
    }
}

/// Arrival offsets from the start of the run of a Poisson process with `rate` req/s, every
/// offset strictly inside `duration`, identical for identical arguments.
pub fn arrival_schedule(rate: f64, duration: Duration, seed: u64) -> Vec<Duration> {
    let mut rng = OpenLoopRng::new(seed);
    let mut out = Vec::new();
    let mut t = Duration::ZERO;
    loop {
        t = t.saturating_add(rng.inter_arrival(rate));
        if t >= duration {
            return out;
        }
        out.push(t);
    }
}

/// Per-request outcome counts added to the report.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Breakdown {
    /// Requests per HTTP status, keyed by the decimal code (`"200"`, `"429"`, …).
    pub by_status: BTreeMap<String, u64>,
    /// Requests per OpenAI error `code` (response body or in-stream error event).
    pub by_error_code: BTreeMap<String, u64>,
    /// Arrivals not sent because `--concurrency` requests were already outstanding.
    pub client_dropped: u64,
    /// 200 responses whose stream ended without `data: [DONE]`.
    pub streams_incomplete: u64,
}

impl Breakdown {
    /// Count one answered request.
    pub fn record(&mut self, status: u16, error_code: Option<&str>, stream_done: bool) {
        *self.by_status.entry(status.to_string()).or_default() += 1;
        if let Some(code) = error_code {
            *self.by_error_code.entry(code.to_string()).or_default() += 1;
        }
        if status == 200 && !stream_done {
            self.streams_incomplete += 1;
        }
    }

    /// Count one arrival dropped at the client.
    pub fn dropped(&mut self) {
        self.client_dropped += 1;
    }

    /// Add `other`'s counts into `self`.
    pub fn merge(&mut self, other: Breakdown) {
        for (k, v) in other.by_status {
            *self.by_status.entry(k).or_default() += v;
        }
        for (k, v) in other.by_error_code {
            *self.by_error_code.entry(k).or_default() += v;
        }
        self.client_dropped += other.client_dropped;
        self.streams_incomplete += other.streams_incomplete;
    }
}

/// The OpenAI error `code` of an error body or stream event (`{"error":{"code":…}}`).
pub fn error_code(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    error_code_of(&v)
}

/// [`error_code`] on an already parsed JSON value.
pub fn error_code_of(v: &Value) -> Option<String> {
    v.get("error")?.get("code")?.as_str().map(str::to_string)
}

/// One timeline line at `t` seconds: the pressure summary of `doc`, or `{"t","error"}`.
pub fn timeline_line(t: f64, doc: Result<Value, String>) -> String {
    let t = (t * 1000.0).round() / 1000.0;
    let line = match doc {
        Ok(doc) => {
            let kv_utilization = doc
                .get("signals")
                .and_then(Value::as_array)
                .and_then(|signals| {
                    signals
                        .iter()
                        .find(|s| s.get("name").and_then(Value::as_str) == Some("kv_utilization"))
                })
                .and_then(|s| s.get("value"))
                .cloned()
                .unwrap_or(Value::Null);
            let field = |v: Option<&Value>| v.cloned().unwrap_or(Value::Null);
            json!({
                "t": t,
                "state": field(doc.get("state")),
                "circuit": field(doc.get("circuit").and_then(|c| c.get("state"))),
                "dominant_signal": field(doc.get("dominant_signal")),
                "queue": field(doc.get("admission").and_then(|a| a.get("queued"))),
                "kv_utilization": kv_utilization,
            })
        }
        Err(error) => json!({"t": t, "error": error}),
    };
    line.to_string()
}

async fn fetch_pressure(client: &reqwest::Client, url: &str) -> Result<Value, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("GET {url}: {e}"))?;
    if !status.is_success() {
        return Err(format!("GET {url}: HTTP {status}"));
    }
    serde_json::from_str(&text).map_err(|e| format!("GET {url}: {e}"))
}

/// Poll `url` (the pressure document) once per second, starting immediately, and append one
/// [`timeline_line`] per poll to `out` until `stop` turns true. Each poll is bounded by the
/// one-second period, so a hung server yields error lines instead of a stalled timeline.
pub async fn pressure_timeline(
    client: reqwest::Client,
    url: String,
    out: PathBuf,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> std::io::Result<()> {
    const PERIOD: Duration = Duration::from_secs(1);
    let mut file = std::io::LineWriter::new(std::fs::File::create(&out)?);
    let started = Instant::now();
    let mut ticks = tokio::time::interval(PERIOD);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        if *stop.borrow() {
            return Ok(());
        }
        tokio::select! {
            _ = ticks.tick() => {}
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() {
                    return Ok(());
                }
                continue;
            }
        }
        let t = started.elapsed().as_secs_f64();
        let doc = match tokio::time::timeout(PERIOD, fetch_pressure(&client, &url)).await {
            Ok(doc) => doc,
            Err(_) => Err(format!("GET {url}: no answer within {PERIOD:?}")),
        };
        writeln!(file, "{}", timeline_line(t, doc))?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_arg_parses_and_rejects() {
        assert_eq!(
            "64..6000".parse::<RangeArg>(),
            Ok(RangeArg { min: 64, max: 6000 })
        );
        assert_eq!("7..7".parse::<RangeArg>(), Ok(RangeArg { min: 7, max: 7 }));
        assert!("8..7".parse::<RangeArg>().is_err());
        assert!("8".parse::<RangeArg>().is_err());
        assert!("a..7".parse::<RangeArg>().is_err());
        assert!("1..-2".parse::<RangeArg>().is_err());
        assert_eq!(RangeArg { min: 1, max: 9 }.to_string(), "1..9");
    }

    #[test]
    fn draws_stay_in_range_and_cover_it() {
        let mut rng = OpenLoopRng::new(11);
        let r = RangeArg { min: 3, max: 6 };
        let mut seen = [false; 4];
        for _ in 0..1000 {
            let v = rng.draw(r);
            assert!((3..=6).contains(&v));
            seen[(v - 3) as usize] = true;
        }
        assert!(seen.iter().all(|s| *s), "every value drawn: {seen:?}");
        let full = RangeArg {
            min: 0,
            max: u32::MAX,
        };
        let _ = rng.draw(full);
        assert_eq!(rng.draw(RangeArg { min: 5, max: 5 }), 5);
    }

    #[test]
    fn per_request_draws_are_independent_of_order() {
        let r = RangeArg { min: 1, max: 1000 };
        let a: Vec<u32> = (0..8)
            .map(|i| OpenLoopRng::for_request(3, i).draw(r))
            .collect();
        let b: Vec<u32> = (0..8)
            .rev()
            .map(|i| OpenLoopRng::for_request(3, i).draw(r))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        assert_eq!(a, b);
        assert_ne!(a[0], a[1]);
    }

    #[test]
    fn inter_arrival_mean_matches_rate() {
        let mut rng = OpenLoopRng::new(5);
        let n = 20_000;
        let total: f64 = (0..n).map(|_| rng.inter_arrival(10.0).as_secs_f64()).sum();
        let mean = total / f64::from(n);
        assert!((mean - 0.1).abs() < 0.005, "mean {mean}");
    }

    #[test]
    fn breakdown_counts() {
        let mut b = Breakdown::default();
        b.record(200, None, true);
        b.record(200, Some("internal_error"), false);
        b.record(429, Some("queue_full"), false);
        b.record(503, Some("overloaded"), false);
        b.dropped();
        assert_eq!(b.by_status["200"], 2);
        assert_eq!(b.by_status["429"], 1);
        assert_eq!(b.by_error_code["queue_full"], 1);
        assert_eq!(b.by_error_code["internal_error"], 1);
        assert_eq!(b.streams_incomplete, 1, "non-200 never counts incomplete");
        assert_eq!(b.client_dropped, 1);
        let mut total = b.clone();
        total.merge(b);
        assert_eq!(total.by_status["200"], 4);
        assert_eq!(total.client_dropped, 2);
    }

    #[test]
    fn error_codes_from_bodies() {
        assert_eq!(
            error_code(r#"{"error":{"message":"m","code":"queue_full"}}"#).as_deref(),
            Some("queue_full")
        );
        assert_eq!(error_code(r#"{"error":{"message":"m"}}"#), None);
        assert_eq!(error_code("not json"), None);
    }

    #[test]
    fn timeline_lines() {
        let doc = json!({
            "state": "RED",
            "dominant_signal": "host_psi",
            "signals": [{"name": "kv_utilization", "value": 0.5}],
            "admission": {"queued": 3},
            "circuit": {"state": "DEGRADED"}
        });
        let v: Value = serde_json::from_str(&timeline_line(1.23456, Ok(doc))).unwrap();
        assert_eq!(
            v,
            json!({"t": 1.235, "state": "RED", "circuit": "DEGRADED",
                   "dominant_signal": "host_psi", "queue": 3, "kv_utilization": 0.5})
        );
        let v: Value =
            serde_json::from_str(&timeline_line(2.0, Err("refused".to_string()))).unwrap();
        assert_eq!(v, json!({"t": 2.0, "error": "refused"}));
    }
}
