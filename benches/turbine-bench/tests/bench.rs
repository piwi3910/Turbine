//! `turbine-bench` against in-test SSE servers (raw HTTP/1.1 over tokio TCP, and an axum mock
//! for the open-loop overload mode).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use turbine_bench::open_loop::arrival_schedule;

#[derive(Clone, Copy)]
enum Mode {
    /// role-only chunk, 100 ms pause, 5 content chunks 20 ms apart, usage chunk, [DONE].
    Stream { usage_tokens: u64 },
    /// Every other completion request answers 500.
    FailHalf,
    /// Every completion request answers 500.
    FailAll,
    /// Chunked SSE written in small TCP writes that split events, lines and multi-byte
    /// characters (the content is "ü€𝄞" per chunk), then the terminating chunk.
    Split,
    /// Chunked SSE with two content chunks, then the connection closes before `[DONE]` and the
    /// terminating chunk.
    Truncate,
}

/// The chunked-encoding frame of `data`.
fn chunk(data: &[u8]) -> Vec<u8> {
    let mut out = format!("{:x}\r\n", data.len()).into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    out
}

/// Writes `bytes` in pieces of `step` bytes with a flush and a pause in between, so the client
/// receives them in separate reads.
async fn write_split(sock: &mut TcpStream, bytes: &[u8], step: usize) {
    for piece in bytes.chunks(step) {
        sock.write_all(piece).await.unwrap();
        sock.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

async fn respond_chunked(mut sock: TcpStream, truncate: bool) {
    sock.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
    )
    .await
    .unwrap();
    let event = |v: &str| format!("data: {v}\n\n").into_bytes();
    let mut events = vec![event(
        r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
    )];
    for _ in 0..3 {
        events.push(event(
            r#"{"choices":[{"index":0,"delta":{"content":"ü€𝄞"}}]}"#,
        ));
    }
    if truncate {
        let body: Vec<u8> = events[..3].concat();
        sock.write_all(&chunk(&body)).await.unwrap();
        sock.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        // Closed mid-body: no [DONE], no terminating chunk.
        return;
    }
    events.push(event(
        r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}"#,
    ));
    events.push(event("[DONE]"));
    // One HTTP chunk per two events, each HTTP chunk cut in 3-byte TCP writes: events, lines,
    // chunk-size lines and every multi-byte character get split between reads.
    let mut wire = Vec::new();
    for pair in events.chunks(2) {
        wire.extend(chunk(&pair.concat()));
    }
    wire.extend_from_slice(b"0\r\n\r\n");
    write_split(&mut sock, &wire, 3).await;
    sock.shutdown().await.ok();
}

/// Read one request (headers + Content-Length body) and return its request line.
async fn read_request(sock: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = sock.read(&mut tmp).await.unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).to_string();
            let len = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while buf.len() < end + 4 + len {
                let n = sock.read(&mut tmp).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            return head.lines().next().unwrap_or_default().to_string();
        }
    }
    String::new()
}

async fn respond(mut sock: TcpStream, mode: Mode, counter: Arc<AtomicUsize>) {
    let line = read_request(&mut sock).await;
    if line.starts_with("GET /v1/models") {
        let body = r#"{"object":"list","data":[{"id":"mock-model","object":"model"}]}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
        return;
    }
    let n = counter.fetch_add(1, Ordering::SeqCst);
    let fail = match mode {
        Mode::FailAll => true,
        Mode::FailHalf => n.is_multiple_of(2),
        Mode::Split => return respond_chunked(sock, false).await,
        Mode::Truncate => return respond_chunked(sock, true).await,
        Mode::Stream { .. } => false,
    };
    if fail {
        let body = r#"{"error":{"message":"boom","type":"server_error","code":"internal_error"}}"#;
        let resp = format!(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
        return;
    }
    let usage_tokens = match mode {
        Mode::Stream { usage_tokens } => usage_tokens,
        _ => 5,
    };
    sock.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
    )
    .await
    .unwrap();
    let event = |v: &str| format!("data: {v}\n\n");
    sock.write_all(
        event(r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#).as_bytes(),
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    for i in 0..5 {
        if i > 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let chunk = format!(r#"{{"choices":[{{"index":0,"delta":{{"content":"tok{i} "}}}}]}}"#);
        sock.write_all(event(&chunk).as_bytes()).await.unwrap();
    }
    let usage = format!(
        r#"{{"choices":[],"usage":{{"prompt_tokens":10,"completion_tokens":{usage_tokens},"total_tokens":{}}}}}"#,
        10 + usage_tokens
    );
    sock.write_all(event(&usage).as_bytes()).await.unwrap();
    sock.write_all(event("[DONE]").as_bytes()).await.unwrap();
    sock.shutdown().await.ok();
}

async fn mock_server(mode: Mode) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    tokio::spawn(async move {
        loop {
            let (sock, _) = listener.accept().await.unwrap();
            tokio::spawn(respond(sock, mode, counter.clone()));
        }
    });
    addr
}

async fn run_bench(args: Vec<String>) -> (Option<i32>, Value, String) {
    let out = tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_turbine-bench"))
            .args(&args)
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let report = serde_json::from_str(&stdout).unwrap_or(Value::Null);
    (out.status.code(), report, stderr)
}

fn args(addr: SocketAddr, extra: &[&str]) -> Vec<String> {
    let mut v = vec![
        "--url".to_string(),
        format!("http://{addr}"),
        "--output".into(),
        "json".into(),
    ];
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

#[tokio::test(flavor = "multi_thread")]
async fn mock_endpoint_measurements() {
    let addr = mock_server(Mode::Stream { usage_tokens: 5 }).await;
    let (code, r, stderr) = run_bench(args(
        addr,
        &[
            "--model",
            "mock-model",
            "--concurrency",
            "2",
            "--requests",
            "4",
            "--prompt-words",
            "16",
        ],
    ))
    .await;
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(r["requests_ok"], 4, "{r}");
    assert_eq!(r["requests_failed"], 0, "{r}");
    assert!(
        r["ttft_ms"]["p50"].as_f64().unwrap() >= 100.0,
        "role-only chunk counted as first token: {r}"
    );
    assert!(r["itl_ms"]["p50"].as_f64().unwrap() >= 20.0, "{r}");
    assert!(r["e2e_ms"]["p50"].as_f64().unwrap() >= 180.0, "{r}");
    let wall = r["wall_seconds"].as_f64().unwrap();
    let tokens = r["output_token_throughput"].as_f64().unwrap() * wall;
    assert!(
        (tokens - 20.0).abs() < 1e-6,
        "expected 20 output tokens, got {tokens}: {r}"
    );
    assert!(
        (r["request_throughput"].as_f64().unwrap() * wall - 4.0).abs() < 1e-6,
        "{r}"
    );
    for key in ["ttft_ms", "itl_ms", "e2e_ms"] {
        for p in ["p50", "p95", "p99"] {
            assert!(r[key][p].is_number(), "{key}.{p} missing: {r}");
        }
    }

    // usage.completion_tokens wins over the number of content chunks.
    let addr = mock_server(Mode::Stream { usage_tokens: 7 }).await;
    let (code, r, stderr) = run_bench(args(
        addr,
        &[
            "--model",
            "mock-model",
            "--requests",
            "1",
            "--endpoint",
            "chat",
        ],
    ))
    .await;
    assert_eq!(code, Some(0), "stderr: {stderr}");
    let tokens =
        r["output_token_throughput"].as_f64().unwrap() * r["wall_seconds"].as_f64().unwrap();
    assert!((tokens - 7.0).abs() < 1e-6, "usage ignored: {r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn failures_counted() {
    // No --model: the first id from GET /v1/models is used.
    let addr = mock_server(Mode::FailHalf).await;
    let (code, r, stderr) = run_bench(args(addr, &["--concurrency", "2", "--requests", "4"])).await;
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(r["requests_ok"], 2, "{r}");
    assert_eq!(r["requests_failed"], 2, "{r}");
    assert!(stderr.contains("HTTP 500"), "failures are logged: {stderr}");

    let addr = mock_server(Mode::FailAll).await;
    let (code, r, _) = run_bench(args(addr, &["--model", "m", "--requests", "3"])).await;
    assert_eq!(code, Some(1));
    assert_eq!(r["requests_ok"], 0, "{r}");
    assert_eq!(r["requests_failed"], 3, "{r}");
}

/// Events, SSE lines, chunk-size lines and multi-byte characters split across TCP reads are
/// reassembled: every request succeeds with its three content chunks.
#[tokio::test(flavor = "multi_thread")]
async fn split_events_and_multibyte_characters() {
    let addr = mock_server(Mode::Split).await;
    let (code, r, stderr) = run_bench(args(
        addr,
        &[
            "--model",
            "m",
            "--concurrency",
            "4",
            "--requests",
            "8",
            "--endpoint",
            "chat",
        ],
    ))
    .await;
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(r["requests_ok"], 8, "{r}\n{stderr}");
    assert_eq!(r["requests_failed"], 0, "{r}\n{stderr}");
    let tokens =
        r["output_token_throughput"].as_f64().unwrap() * r["wall_seconds"].as_f64().unwrap();
    assert!((tokens - 24.0).abs() < 1e-6, "{r}");
}

/// A body cut off by the peer is a failed request whose message names the transport cause
/// under reqwest's "error decoding response body", and how far the stream got.
#[tokio::test(flavor = "multi_thread")]
async fn truncated_stream_reports_the_transport_cause() {
    let addr = mock_server(Mode::Truncate).await;
    let (code, r, stderr) = run_bench(args(
        addr,
        &["--model", "m", "--requests", "1", "--endpoint", "chat"],
    ))
    .await;
    assert_eq!(code, Some(1), "stderr: {stderr}");
    assert_eq!(r["requests_failed"], 1, "{r}");
    let line = stderr
        .lines()
        .find(|l| l.contains("request 0 failed"))
        .unwrap_or_else(|| panic!("no failure line: {stderr}"));
    let prefix = "reading stream after 2 content chunks: error decoding response body: ";
    let cause = line
        .split_once(prefix)
        .unwrap_or_else(|| panic!("{prefix:?} missing: {line}"))
        .1;
    assert!(!cause.trim().is_empty(), "no transport cause: {line}");
}

/// Status and error-code counts of every chat completion the overload mock answered.
#[derive(Default)]
struct OverloadCounts {
    by_status: BTreeMap<String, u64>,
    by_error_code: BTreeMap<String, u64>,
}

#[derive(Clone, Default)]
struct OverloadMock {
    served: Arc<AtomicUsize>,
    counts: Arc<Mutex<OverloadCounts>>,
}

/// Fixed 3-cycle pattern: 200 stream with `[DONE]`, 429 `queue_full`, 503 `overloaded`.
async fn overload_chat(State(mock): State<OverloadMock>) -> Response {
    let n = mock.served.fetch_add(1, Ordering::SeqCst);
    let (status, code) = match n % 3 {
        0 => (StatusCode::OK, None),
        1 => (StatusCode::TOO_MANY_REQUESTS, Some("queue_full")),
        _ => (StatusCode::SERVICE_UNAVAILABLE, Some("overloaded")),
    };
    {
        let mut c = mock.counts.lock().unwrap();
        *c.by_status.entry(status.as_u16().to_string()).or_default() += 1;
        if let Some(code) = code {
            *c.by_error_code.entry(code.to_string()).or_default() += 1;
        }
    }
    match code {
        None => {
            let body: String = [
                r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
                r#"{"choices":[{"index":0,"delta":{"content":"a "}}]}"#,
                r#"{"choices":[{"index":0,"delta":{"content":"b"}}]}"#,
                r#"{"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#,
                "[DONE]",
            ]
            .iter()
            .map(|e| format!("data: {e}\n\n"))
            .collect();
            ([(header::CONTENT_TYPE, "text/event-stream")], body).into_response()
        }
        Some(code) => (
            status,
            Json(json!({
                "error": {"message": "busy", "type": "service_unavailable", "code": code}
            })),
        )
            .into_response(),
    }
}

async fn pressure_document() -> Json<Value> {
    Json(json!({
        "enabled": true,
        "state": "ORANGE",
        "since": "2026-09-25T18:02:11.482Z",
        "dominant_signal": "kv_utilization",
        "exhaustion_horizon_seconds": 14.2,
        "signals": [
            {"name": "host_available", "value": 3.1, "level": "YELLOW", "stale": false},
            {"name": "kv_utilization", "value": 0.84, "level": "ORANGE", "stale": false}
        ],
        "admission": {"queued": 17, "max_queue": 256},
        "circuit": {"state": "HEALTHY", "since": "2026-09-25T18:00:00.000Z", "last_reason": null}
    }))
}

async fn overload_mock() -> (SocketAddr, OverloadMock) {
    let mock = OverloadMock::default();
    let app = Router::new()
        .route(
            "/v1/models",
            get(|| async { Json(json!({"object": "list", "data": [{"id": "mock-model"}]})) }),
        )
        .route("/v1/chat/completions", post(overload_chat))
        .route("/turbine/v1/pressure", get(pressure_document))
        .with_state(mock.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (addr, mock)
}

#[tokio::test(flavor = "multi_thread")]
async fn open_loop_rate_and_breakdown() {
    let schedule = arrival_schedule(50.0, Duration::from_secs(4), 3);
    assert_eq!(
        schedule,
        arrival_schedule(50.0, Duration::from_secs(4), 3),
        "same seed must give the same arrival schedule"
    );
    assert_ne!(schedule, arrival_schedule(50.0, Duration::from_secs(4), 4));
    assert!(
        (170..=230).contains(&schedule.len()),
        "Poisson(50/s x 4 s) gave {} arrivals",
        schedule.len()
    );
    assert!(schedule.windows(2).all(|w| w[0] <= w[1]));
    assert!(schedule.iter().all(|t| *t < Duration::from_secs(4)));

    let (addr, mock) = overload_mock().await;
    let timeline = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("open-loop-{}-t.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&timeline);
    let (code, r, stderr) = run_bench(args(
        addr,
        &[
            "--rate",
            "50",
            "--duration",
            "4s",
            "--seed",
            "3",
            "--concurrency",
            "64",
            "--prompt-words-range",
            "4..32",
            "--max-tokens-range",
            "1..8",
            "--pressure-timeline",
            timeline.to_str().unwrap(),
        ],
    ))
    .await;
    assert_eq!(code, Some(0), "stderr: {stderr}");

    // Open loop: exactly the seeded schedule's arrivals, not a closed loop at concurrency 64.
    let dropped = r["client_dropped"].as_u64().expect("client_dropped");
    let sent = r["requests_ok"].as_u64().unwrap() + r["requests_failed"].as_u64().unwrap();
    assert_eq!(sent + dropped, schedule.len() as u64, "{r}");
    assert_eq!(sent, mock.served.load(Ordering::SeqCst) as u64, "{r}");

    let counts = mock.counts.lock().unwrap();
    let by_status: BTreeMap<String, u64> =
        serde_json::from_value(r["by_status"].clone()).expect("by_status");
    let by_error_code: BTreeMap<String, u64> =
        serde_json::from_value(r["by_error_code"].clone()).expect("by_error_code");
    assert_eq!(by_status, counts.by_status, "{r}");
    assert_eq!(by_error_code, counts.by_error_code, "{r}");
    assert_eq!(by_status.len(), 3, "{r}");
    assert_eq!(
        r["requests_ok"].as_u64(),
        counts.by_status.get("200").copied(),
        "{r}"
    );
    assert_eq!(r["streams_incomplete"], 0, "{r}");

    let text = std::fs::read_to_string(&timeline).expect("timeline written");
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("timeline line is JSON"))
        .collect();
    assert!(
        (3..=5).contains(&lines.len()),
        "{} lines: {text}",
        lines.len()
    );
    for line in &lines {
        for key in [
            "t",
            "state",
            "circuit",
            "dominant_signal",
            "queue",
            "kv_utilization",
        ] {
            assert!(line.get(key).is_some(), "missing {key}: {line}");
        }
        assert_eq!(line["state"], "ORANGE");
        assert_eq!(line["circuit"], "HEALTHY");
        assert_eq!(line["dominant_signal"], "kv_utilization");
        assert_eq!(line["queue"], 17);
        assert_eq!(line["kv_utilization"], 0.84);
    }
    let _ = std::fs::remove_file(&timeline);
}

/// One recorded request: lowercase header lines and the JSON body.
struct Recorded {
    headers: Vec<(String, String)>,
    body: Value,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Reads one request fully: head and body.
async fn read_full(sock: &mut TcpStream) -> (String, Recorded) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = sock.read(&mut tmp).await.unwrap();
        if n == 0 {
            return (
                String::new(),
                Recorded {
                    headers: Vec::new(),
                    body: Value::Null,
                },
            );
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).to_string();
            let headers: Vec<(String, String)> = head
                .lines()
                .skip(1)
                .filter_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    Some((k.trim().to_ascii_lowercase(), v.trim().to_string()))
                })
                .collect();
            let len = headers
                .iter()
                .find(|(k, _)| k == "content-length")
                .map_or(0, |(_, v)| v.parse::<usize>().unwrap());
            while buf.len() < end + 4 + len {
                let n = sock.read(&mut tmp).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = serde_json::from_slice(&buf[end + 4..end + 4 + len]).unwrap_or(Value::Null);
            let line = head.lines().next().unwrap_or_default().to_string();
            return (line, Recorded { headers, body });
        }
    }
}

/// Streams "reply-<n> " (two chunks) and usage with 100 prompt tokens, 40 of them cached;
/// records every completion request.
async fn recording_server() -> (SocketAddr, Arc<std::sync::Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let (line, rec) = read_full(&mut sock).await;
                if !line.starts_with("POST") {
                    return;
                }
                let n = {
                    let mut log = seen.lock().unwrap();
                    log.push(rec);
                    log.len()
                };
                sock.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
                let event = |v: &str| format!("data: {v}\n\n");
                for part in [format!("reply-{n}"), " ".to_string()] {
                    let chunk =
                        format!(r#"{{"choices":[{{"index":0,"delta":{{"content":"{part}"}}}}]}}"#);
                    sock.write_all(event(&chunk).as_bytes()).await.unwrap();
                }
                let usage = r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102,"prompt_tokens_details":{"cached_tokens":40}}}"#;
                sock.write_all(event(usage).as_bytes()).await.unwrap();
                sock.write_all(event("[DONE]").as_bytes()).await.unwrap();
                sock.shutdown().await.ok();
            });
        }
    });
    (addr, log)
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_turn_profile() {
    let (addr, log) = recording_server().await;
    let (code, r, stderr) = run_bench(args(
        addr,
        &[
            "--model",
            "mock-model",
            "--profile",
            "multi-turn",
            "--sessions",
            "3",
            "--turns",
            "4",
            "--shared-prefix-words",
            "50",
            "--prompt-words",
            "8",
            "--think-time",
            "0..0.05",
            "--concurrency",
            "3",
            "--session-hints",
        ],
    ))
    .await;
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(r["requests_ok"], 12, "{r}");
    assert_eq!(r["cached_tokens_ratio"], 0.4, "{r}");
    for key in ["ttft_ms_first_turn", "ttft_ms_later_turns"] {
        assert!(r[key]["p50"].is_number(), "{key} missing: {r}");
    }

    let log = log.lock().unwrap();
    assert_eq!(log.len(), 12);
    let mut sessions: std::collections::BTreeMap<String, Vec<&Recorded>> = Default::default();
    for rec in log.iter() {
        let key = rec.body["prompt_cache_key"]
            .as_str()
            .expect("session id sent");
        sessions.entry(key.to_string()).or_default().push(rec);
    }
    assert_eq!(sessions.len(), 3, "three sessions");
    for turns in sessions.values() {
        assert_eq!(turns.len(), 4);
        // Sequential: turn t carries the system prefix, then 2t history messages and a new
        // user message, and extends turn t-1 with its reply.
        let shared = turns[0].body["messages"][0].clone();
        assert_eq!(shared["role"], "system");
        assert_eq!(shared["content"].as_str().unwrap().split(' ').count(), 50);
        let mut by_len: Vec<&&Recorded> = turns.iter().collect();
        by_len.sort_by_key(|t| t.body["messages"].as_array().unwrap().len());
        for (t, rec) in by_len.iter().enumerate() {
            let messages = rec.body["messages"].as_array().unwrap();
            assert_eq!(messages.len(), 2 * t + 2, "turn {t}");
            assert_eq!(messages[0], shared, "the shared prefix leads every turn");
            assert_eq!(messages.last().unwrap()["role"], "user");
            if t > 0 {
                let prev = by_len[t - 1].body["messages"].as_array().unwrap();
                assert_eq!(&messages[..prev.len()], prev.as_slice(), "full history");
                let reply = &messages[prev.len()];
                assert_eq!(reply["role"], "assistant");
                assert!(reply["content"].as_str().unwrap().starts_with("reply-"));
            }
            assert_eq!(
                rec.header("x-turbine-session-resume-within"),
                Some("1"),
                "the maximum think time, at least 1 s"
            );
            let last = t == 3;
            assert_eq!(
                rec.header("x-turbine-session-end"),
                last.then_some("true"),
                "turn {t}"
            );
        }
    }
}
