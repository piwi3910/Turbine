//! Aggregation of per-request measurements into the bench report (text or JSON).

use std::fmt::Write as _;
use std::time::Duration;

use serde::Serialize;

use crate::open_loop::Breakdown;

/// Measurements of one successful request.
#[derive(Clone, Debug)]
pub struct RequestStats {
    /// Time from sending the request to the first content token.
    pub ttft: Duration,
    /// Gaps between consecutive content tokens.
    pub itls: Vec<Duration>,
    /// Time from sending the request to the end of the stream.
    pub e2e: Duration,
    /// Output tokens reported by the endpoint.
    pub output_tokens: u64,
    /// The streamed content text (the assistant reply of a multi-turn history).
    pub text: String,
    /// `usage.prompt_tokens`, when the endpoint reported usage.
    pub prompt_tokens: Option<u64>,
    /// `usage.prompt_tokens_details.cached_tokens`, when reported.
    pub cached_tokens: Option<u64>,
}

/// p50/p95/p99 of a latency distribution, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// The bench report; its JSON keys are the contract's P0 bench report keys.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub requests_ok: u64,
    pub requests_failed: u64,
    pub wall_seconds: f64,
    pub request_throughput: f64,
    pub output_token_throughput: f64,
    pub ttft_ms: Percentiles,
    pub itl_ms: Percentiles,
    pub e2e_ms: Percentiles,
    /// `by_status`, `by_error_code`, `client_dropped`, `streams_incomplete` (phase 3).
    #[serde(flatten)]
    pub breakdown: Breakdown,
    /// Multi-turn (P4): cached / prompt tokens over every turn that reported both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens_ratio: Option<f64>,
    /// Multi-turn: TTFT of each session's first turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms_first_turn: Option<Percentiles>,
    /// Multi-turn: TTFT of the later turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms_later_turns: Option<Percentiles>,
}

/// Nearest-rank percentiles of `values`. Empty input gives zeros.
pub fn percentiles(mut values: Vec<f64>) -> Percentiles {
    if values.is_empty() {
        return Percentiles::default();
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    // Nearest rank: the smallest value with at least p% of the values at or below it.
    let rank = |p: f64| {
        let idx = ((p / 100.0) * n as f64).ceil() as usize;
        values[idx.clamp(1, n) - 1]
    };
    Percentiles {
        p50: rank(50.0),
        p95: rank(95.0),
        p99: rank(99.0),
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

impl Report {
    /// Aggregates the successful requests `ok`, the `failed` count and the run's `wall` time.
    /// ITL percentiles pool every token gap of every successful request.
    pub fn from_results(ok: &[RequestStats], failed: u64, wall: Duration) -> Report {
        let wall_seconds = wall.as_secs_f64();
        let per_sec = |n: f64| {
            if wall_seconds > 0.0 {
                n / wall_seconds
            } else {
                0.0
            }
        };
        let tokens: u64 = ok.iter().map(|r| r.output_tokens).sum();
        Report {
            requests_ok: ok.len() as u64,
            requests_failed: failed,
            wall_seconds,
            request_throughput: per_sec(ok.len() as f64),
            output_token_throughput: per_sec(tokens as f64),
            ttft_ms: percentiles(ok.iter().map(|r| ms(r.ttft)).collect()),
            itl_ms: percentiles(
                ok.iter()
                    .flat_map(|r| r.itls.iter().map(|d| ms(*d)))
                    .collect(),
            ),
            e2e_ms: percentiles(ok.iter().map(|r| ms(r.e2e)).collect()),
            breakdown: Breakdown::default(),
            cached_tokens_ratio: None,
            ttft_ms_first_turn: None,
            ttft_ms_later_turns: None,
        }
    }

    /// The report with the run's status / error-code breakdown attached.
    pub fn with_breakdown(mut self, breakdown: Breakdown) -> Report {
        self.breakdown = breakdown;
        self
    }

    /// The multi-turn additions: the cached-token ratio of every turn and the TTFT split
    /// into first and later turns.
    pub fn add_multi_turn(&mut self, first: &[&RequestStats], later: &[&RequestStats]) {
        let (mut prompt, mut cached) = (0u64, 0u64);
        for r in first.iter().chain(later) {
            if let (Some(p), Some(c)) = (r.prompt_tokens, r.cached_tokens) {
                prompt += p;
                cached += c;
            }
        }
        self.cached_tokens_ratio = (prompt > 0).then(|| cached as f64 / prompt as f64);
        let ttft = |rs: &[&RequestStats]| percentiles(rs.iter().map(|r| ms(r.ttft)).collect());
        self.ttft_ms_first_turn = Some(ttft(first));
        self.ttft_ms_later_turns = Some(ttft(later));
    }

    /// Human-readable rendering: counts, wall time, throughputs and latency percentile rows.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        // Writing to a String cannot fail.
        let _ = writeln!(s, "requests ok:             {}", self.requests_ok);
        let _ = writeln!(s, "requests failed:         {}", self.requests_failed);
        let _ = writeln!(s, "wall time (s):           {:.3}", self.wall_seconds);
        let _ = writeln!(
            s,
            "request throughput:      {:.3} req/s",
            self.request_throughput
        );
        let _ = writeln!(
            s,
            "output token throughput: {:.3} tok/s",
            self.output_token_throughput
        );
        let counts = |m: &std::collections::BTreeMap<String, u64>| {
            if m.is_empty() {
                "-".to_string()
            } else {
                m.iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };
        let b = &self.breakdown;
        let _ = writeln!(s, "by status:               {}", counts(&b.by_status));
        let _ = writeln!(s, "by error code:           {}", counts(&b.by_error_code));
        let _ = writeln!(s, "client dropped:          {}", b.client_dropped);
        let _ = writeln!(s, "streams incomplete:      {}", b.streams_incomplete);
        s.push_str("latency (ms):\n");
        for (name, p) in [
            ("ttft", &self.ttft_ms),
            ("itl", &self.itl_ms),
            ("e2e", &self.e2e_ms),
        ] {
            let _ = writeln!(
                s,
                "{name:<8} p50 {:>10.2}  p95 {:>10.2}  p99 {:>10.2}",
                p.p50, p.p95, p.p99
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let values: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        let p = percentiles(values);
        assert_eq!(
            p,
            Percentiles {
                p50: 50.0,
                p95: 95.0,
                p99: 99.0
            }
        );
        assert_eq!(percentiles(vec![]), Percentiles::default());
        assert_eq!(percentiles(vec![3.0]).p50, 3.0);
    }

    #[test]
    fn report_aggregates_results() {
        let ms = Duration::from_millis;
        let ok = [
            RequestStats {
                ttft: ms(100),
                itls: vec![ms(20), ms(30)],
                e2e: ms(200),
                output_tokens: 3,
                text: String::new(),
                prompt_tokens: Some(100),
                cached_tokens: Some(60),
            },
            RequestStats {
                ttft: ms(120),
                itls: vec![ms(40)],
                e2e: ms(220),
                output_tokens: 2,
                text: String::new(),
                prompt_tokens: Some(300),
                cached_tokens: None,
            },
        ];
        let mut breakdown = Breakdown::default();
        breakdown.record(200, None, true);
        breakdown.record(200, None, true);
        breakdown.record(503, Some("overloaded"), false);
        breakdown.dropped();
        let r = Report::from_results(&ok, 1, Duration::from_secs(2)).with_breakdown(breakdown);
        assert_eq!(r.requests_ok, 2);
        assert_eq!(r.requests_failed, 1);
        assert_eq!(r.request_throughput, 1.0);
        assert_eq!(r.output_token_throughput, 2.5);
        assert_eq!(r.ttft_ms.p50, 100.0);
        assert_eq!(r.itl_ms.p50, 30.0);
        assert_eq!(r.itl_ms.p99, 40.0);
        assert_eq!(r.e2e_ms.p99, 220.0);

        let json = serde_json::to_value(&r).expect("report serializes");
        for key in [
            "requests_ok",
            "requests_failed",
            "wall_seconds",
            "request_throughput",
            "output_token_throughput",
            "ttft_ms",
            "itl_ms",
            "e2e_ms",
            "by_status",
            "by_error_code",
            "client_dropped",
            "streams_incomplete",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["by_status"]["200"], 2);
        assert_eq!(json["by_status"]["503"], 1);
        assert_eq!(json["by_error_code"]["overloaded"], 1);
        assert_eq!(json["client_dropped"], 1);
        assert_eq!(json["streams_incomplete"], 0);
        assert!(json["ttft_ms"].get("p95").is_some());
        assert!(
            json.get("cached_tokens_ratio").is_none(),
            "multi-turn keys only in multi-turn reports"
        );

        // Multi-turn: only turns reporting both counts enter the ratio.
        let mut m = r.clone();
        m.add_multi_turn(&[&ok[0]], &[&ok[1]]);
        assert_eq!(m.cached_tokens_ratio, Some(0.6));
        assert_eq!(m.ttft_ms_first_turn.unwrap().p50, 100.0);
        assert_eq!(m.ttft_ms_later_turns.unwrap().p50, 120.0);

        let text = r.to_text();
        assert!(text.contains("requests failed:         1"));
        assert!(text.contains("output token throughput: 2.500 tok/s"));
        assert!(text.contains("itl"));
        assert!(text.contains("by status:               200=2 503=1"));
        assert!(text.contains("by error code:           overloaded=1"));
        assert!(text.contains("client dropped:          1"));
    }

    #[test]
    fn zero_wall_time_gives_zero_throughput() {
        let r = Report::from_results(&[], 4, Duration::ZERO);
        assert_eq!(r.request_throughput, 0.0);
        assert_eq!(r.output_token_throughput, 0.0);
        assert_eq!(r.ttft_ms, Percentiles::default());
    }
}
