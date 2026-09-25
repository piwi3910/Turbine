//! `turbine-bench` against in-test SSE servers (raw HTTP/1.1 over tokio TCP).

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone, Copy)]
enum Mode {
    /// role-only chunk, 100 ms pause, 5 content chunks 20 ms apart, usage chunk, [DONE].
    Stream { usage_tokens: u64 },
    /// Every other completion request answers 500.
    FailHalf,
    /// Every completion request answers 500.
    FailAll,
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
