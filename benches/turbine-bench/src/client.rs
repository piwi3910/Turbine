//! Streaming requests against an OpenAI-compatible endpoint, the closed-loop (fixed
//! concurrency) driver and the open-loop (seeded Poisson arrivals) driver.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::{Semaphore, watch};
use tokio::task::JoinSet;

use crate::args::{BenchArgs, EndpointArg};
use crate::open_loop::{self, Breakdown, OpenLoopRng};
use crate::prompt;
use crate::report::{Report, RequestStats};

/// Setup failures before any request is sent.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    /// Usage problems: exit 2.
    #[error("{0}")]
    Usage(String),
    /// The target could not be queried before the run: exit 1.
    #[error("{0}")]
    Target(String),
}

impl BenchError {
    /// Process exit code: 2 for usage errors, 1 when the target cannot be queried.
    pub fn exit_code(&self) -> u8 {
        match self {
            BenchError::Usage(_) => 2,
            BenchError::Target(_) => 1,
        }
    }
}

/// Everything a request task needs, shared by all of them.
struct Target {
    client: reqwest::Client,
    url: String,
    model: String,
    args: BenchArgs,
}

/// Run the benchmark described by `args` and aggregate the report.
pub async fn run(args: &BenchArgs) -> Result<Report, BenchError> {
    let base = args.url.trim_end_matches('/').to_string();
    if !base.starts_with("http://") {
        return Err(BenchError::Usage(format!(
            "--url must be an http:// URL, got {}",
            args.url
        )));
    }
    if let Some(out) = &args.pressure_timeline {
        std::fs::File::create(out).map_err(|e| {
            BenchError::Usage(format!(
                "cannot create --pressure-timeline {}: {e}",
                out.display()
            ))
        })?;
    }
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| BenchError::Target(format!("cannot build HTTP client: {e}")))?;
    let model = match &args.model {
        Some(m) => m.clone(),
        None => first_model(&client, &base).await?,
    };
    let path = match args.endpoint {
        EndpointArg::Chat => "/v1/chat/completions",
        EndpointArg::Completions => "/v1/completions",
    };
    let target = Arc::new(Target {
        client: client.clone(),
        url: format!("{base}{path}"),
        model,
        args: args.clone(),
    });

    let (stop, stop_rx) = watch::channel(false);
    let timeline = args.pressure_timeline.clone().map(|out| {
        tokio::spawn(open_loop::pressure_timeline(
            client,
            format!("{base}/turbine/v1/pressure"),
            out,
            stop_rx,
        ))
    });

    let started = Instant::now();
    let mut tally = Tally::default();
    match (args.rate, args.duration) {
        (Some(rate), Some(duration)) => open_loop_run(&target, rate, duration, &mut tally).await,
        _ => closed_loop_run(&target, &mut tally).await,
    }
    let wall = started.elapsed();

    // The receiver is gone only if the timeline task already ended; its result is reported below.
    let _ = stop.send(true);
    if let Some(task) = timeline {
        match task.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("turbine-bench: pressure timeline: {e}"),
            Err(e) => eprintln!("turbine-bench: pressure timeline task failed: {e}"),
        }
    }
    Ok(Report::from_results(&tally.ok, tally.failed, wall).with_breakdown(tally.breakdown))
}

/// Accumulated outcomes of a run.
#[derive(Default)]
struct Tally {
    ok: Vec<RequestStats>,
    failed: u64,
    breakdown: Breakdown,
}

impl Tally {
    fn add(&mut self, index: u64, outcome: Outcome) {
        if let Some(status) = outcome.status {
            self.breakdown
                .record(status, outcome.error_code.as_deref(), outcome.stream_done);
        }
        match outcome.result {
            Ok(stats) => self.ok.push(stats),
            Err(e) => {
                eprintln!("turbine-bench: request {index} failed: {e}");
                self.failed += 1;
            }
        }
    }

    fn merge(&mut self, other: Tally) {
        self.ok.extend(other.ok);
        self.failed += other.failed;
        self.breakdown.merge(other.breakdown);
    }
}

/// Closed loop: `--concurrency` workers each send their next request as soon as the previous
/// one finishes, until `--requests` were taken or, with `--duration`, the duration elapsed.
async fn closed_loop_run(target: &Arc<Target>, tally: &mut Tally) {
    let args = &target.args;
    let deadline = args.duration.map(|d| Instant::now() + d);
    let workers = match deadline {
        Some(_) => args.concurrency,
        None => args.concurrency.min(args.requests),
    };
    let next = Arc::new(AtomicU64::new(0));
    let mut tasks = Vec::new();
    for _ in 0..workers {
        let (target, next) = (target.clone(), next.clone());
        tasks.push(tokio::spawn(async move {
            let mut tally = Tally::default();
            loop {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                if deadline.is_none() && index >= u64::from(target.args.requests) {
                    break;
                }
                let outcome = send(&target, index).await;
                tally.add(index, outcome);
            }
            tally
        }));
    }
    for task in tasks {
        match task.await {
            Ok(t) => tally.merge(t),
            Err(e) => eprintln!("turbine-bench: worker panicked: {e}"),
        }
    }
}

/// Open loop: requests start at the seeded Poisson arrival times however the endpoint keeps
/// up; an arrival that finds `--concurrency` requests outstanding is dropped, not queued.
async fn open_loop_run(target: &Arc<Target>, rate: f64, duration: Duration, tally: &mut Tally) {
    let schedule = open_loop::arrival_schedule(rate, duration, target.args.seed);
    let slots = Arc::new(Semaphore::new(target.args.concurrency as usize));
    let started = tokio::time::Instant::now();
    let mut running = JoinSet::new();
    for (index, at) in (0u64..).zip(schedule) {
        tokio::time::sleep_until(started + at).await;
        while let Some(done) = running.try_join_next() {
            reap(done, tally);
        }
        match slots.clone().try_acquire_owned() {
            Ok(permit) => {
                let target = target.clone();
                running.spawn(async move {
                    let outcome = send(&target, index).await;
                    drop(permit);
                    (index, outcome)
                });
            }
            Err(_) => tally.breakdown.dropped(),
        }
    }
    while let Some(done) = running.join_next().await {
        reap(done, tally);
    }
}

fn reap(done: Result<(u64, Outcome), tokio::task::JoinError>, tally: &mut Tally) {
    match done {
        Ok((index, outcome)) => tally.add(index, outcome),
        Err(e) => {
            eprintln!("turbine-bench: request task panicked: {e}");
            tally.failed += 1;
        }
    }
}

async fn send(target: &Target, index: u64) -> Outcome {
    let body = request_body(&target.args, &target.model, index);
    one_request(&target.client, &target.url, target.args.endpoint, &body).await
}

async fn first_model(client: &reqwest::Client, base: &str) -> Result<String, BenchError> {
    let url = format!("{base}/v1/models");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| BenchError::Target(format!("GET {url}: {e}")))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| BenchError::Target(format!("GET {url}: {e}")))?;
    if !status.is_success() {
        return Err(BenchError::Target(format!("GET {url}: HTTP {status}")));
    }
    let v: Value =
        serde_json::from_str(&text).map_err(|e| BenchError::Target(format!("GET {url}: {e}")))?;
    v["data"][0]["id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| BenchError::Usage(format!("GET {url} returned no models; pass --model")))
}

/// Prompt words and max_tokens of request `index`: fixed, or drawn from the ranges by the
/// request's own seeded stream.
fn request_lengths(args: &BenchArgs, index: u64) -> (u32, u32) {
    let mut rng = OpenLoopRng::for_request(args.seed, index);
    let words = args
        .prompt_words_range
        .map_or(args.prompt_words, |r| rng.draw(r));
    let max_tokens = args
        .max_tokens_range
        .map_or(args.max_tokens, |r| rng.draw(r));
    (words, max_tokens)
}

fn request_body(args: &BenchArgs, model: &str, index: u64) -> Value {
    let (words, max_tokens) = request_lengths(args, index);
    let text = prompt::prompt(args.seed, index, words);
    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    match args.endpoint {
        EndpointArg::Chat => body["messages"] = json!([{"role": "user", "content": text}]),
        EndpointArg::Completions => body["prompt"] = json!(text),
    }
    if args.ignore_eos {
        body["ignore_eos"] = json!(true);
    }
    body
}

/// Content text of one stream chunk, if any (role-only / usage-only chunks have none).
fn chunk_content(chunk: &Value, endpoint: EndpointArg) -> Option<&str> {
    let choice = chunk.get("choices")?.get(0)?;
    let text = match endpoint {
        EndpointArg::Chat => choice.get("delta")?.get("content")?.as_str()?,
        EndpointArg::Completions => choice.get("text")?.as_str()?,
    };
    (!text.is_empty()).then_some(text)
}

/// `e` and every `source()` below it, joined by `: ` (reqwest's own message names only the
/// error kind, e.g. "error decoding response body", and hides the transport cause).
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut text = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        text.push_str(": ");
        text.push_str(&s.to_string());
        source = s.source();
    }
    text
}

/// What happened to one request: the HTTP status (if a response arrived), the OpenAI error
/// `code` (response body or in-stream error event), whether the stream reached `[DONE]`, and
/// either the measurements or the failure message.
struct Outcome {
    status: Option<u16>,
    error_code: Option<String>,
    stream_done: bool,
    result: Result<RequestStats, String>,
}

impl Outcome {
    fn failed(status: Option<u16>, error_code: Option<String>, message: String) -> Outcome {
        Outcome {
            status,
            error_code,
            stream_done: false,
            result: Err(message),
        }
    }
}

async fn one_request(
    client: &reqwest::Client,
    url: &str,
    endpoint: EndpointArg,
    body: &Value,
) -> Outcome {
    let sent = Instant::now();
    let mut resp = match client
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Outcome::failed(None, None, format!("POST {url}: {}", error_chain(&e))),
    };
    let status = resp.status();
    let code = Some(status.as_u16());
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Outcome::failed(
            code,
            open_loop::error_code(&text),
            format!(
                "HTTP {status}: {}",
                text.chars().take(200).collect::<String>()
            ),
        );
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut token_times: Vec<Instant> = Vec::new();
    let mut usage_tokens: Option<u64> = None;
    // An in-stream error event fails the request; the stream still runs to `[DONE]` (C-3).
    let mut stream_error: Option<(Option<String>, String)> = None;
    loop {
        let bytes = match resp.chunk().await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                let (error_code, message) =
                    stream_error.unwrap_or((None, "stream ended without [DONE]".to_string()));
                return Outcome::failed(code, error_code, message);
            }
            Err(e) => {
                let error_code = stream_error.and_then(|(c, _)| c);
                return Outcome::failed(
                    code,
                    error_code,
                    format!(
                        "reading stream after {} content chunks: {}",
                        token_times.len(),
                        error_chain(&e)
                    ),
                );
            }
        };
        buf.extend_from_slice(&bytes);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\r', '\n']);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim_start();
            if data == "[DONE]" {
                let done = Instant::now();
                let (error_code, result) = match (stream_error, token_times.first()) {
                    (Some((error_code, message)), _) => (error_code, Err(message)),
                    (None, None) => (None, Err("stream finished without content".to_string())),
                    (None, Some(first)) => (
                        None,
                        Ok(RequestStats {
                            ttft: *first - sent,
                            itls: token_times.windows(2).map(|w| w[1] - w[0]).collect(),
                            e2e: done - sent,
                            output_tokens: usage_tokens.unwrap_or(token_times.len() as u64),
                        }),
                    ),
                };
                return Outcome {
                    status: code,
                    error_code,
                    stream_done: true,
                    result,
                };
            }
            let value: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(e) => {
                    return Outcome::failed(code, None, format!("bad SSE JSON {data:?}: {e}"));
                }
            };
            if let Some(err) = value.get("error") {
                if stream_error.is_none() {
                    stream_error = Some((
                        open_loop::error_code_of(&value),
                        format!("stream error: {err}"),
                    ));
                }
                continue;
            }
            if chunk_content(&value, endpoint).is_some() {
                token_times.push(Instant::now());
            }
            if let Some(n) = value
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(Value::as_u64)
            {
                usage_tokens = Some(n);
            }
        }
    }
}
