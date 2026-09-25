//! Black-box tests of `turbine-server` serving the tiny synthetic checkpoint on the `cpu`
//! reference backend (P1 S-10, S-14): OpenAI shapes, the single slot and cancellation, request
//! validation, startup failures and the Phase 1 metrics.
//!
//! Every wait is bounded and polls; ports come from binding `127.0.0.1:0`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::write_tiny_llama;

const POLL: Duration = Duration::from_millis(20);
const READY_LIMIT: Duration = Duration::from_secs(60);

fn free_addr() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

fn spawn(config: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_turbine-server"))
        .arg("--config")
        .arg(config)
        .env_remove("TURBINE_AMD_SMI_LIBRARY")
        .env_remove("TURBINE_KERNEL_LIBRARY")
        .env("RUST_LOG", "info")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn turbine-server")
}

/// Config for the cpu backend on `model_dir`; `extra` is appended verbatim.
fn config_yaml(model_dir: &Path, addr: SocketAddr, extra: &str) -> String {
    format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: cpu\n\
         reliability:\n  emergency_vram_reserve: 1MiB\n{extra}",
        model_dir.display()
    )
}

/// A running server on the tiny checkpoint; killed on drop.
struct TinyServer {
    child: Child,
    addr: SocketAddr,
    model: String,
    _dir: TempDir,
}

impl TinyServer {
    fn start(extra: &str) -> TinyServer {
        let dir = TempDir::new("turbine-tiny-server");
        let model_dir = dir.path().join("tiny-llama");
        write_tiny_llama(&model_dir, 7);
        let addr = free_addr();
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, config_yaml(&model_dir, addr, extra)).unwrap();
        let mut child = spawn(&config);
        wait_until_ready(&mut child, addr);
        TinyServer {
            child,
            addr,
            model: "tiny-llama".into(),
            _dir: dir,
        }
    }

    fn get(&self, path: &str) -> Response {
        request(self.addr, "GET", path, None)
    }

    fn post(&self, path: &str, body: &Value) -> Response {
        request(self.addr, "POST", path, Some(&body.to_string()))
    }

    fn metrics(&self) -> String {
        self.get("/metrics").body
    }
}

impl Drop for TinyServer {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

fn wait_until_ready(child: &mut Child, addr: SocketAddr) {
    let started = Instant::now();
    while started.elapsed() < READY_LIMIT {
        if let Some(status) = child.try_wait().unwrap() {
            let mut stderr = String::new();
            if let Some(mut e) = child.stderr.take() {
                e.read_to_string(&mut stderr).ok();
            }
            panic!("turbine-server exited early ({status}); stderr:\n{stderr}");
        }
        if TcpStream::connect(addr).is_ok() && request(addr, "GET", "/ready", None).status == 200 {
            return;
        }
        std::thread::sleep(POLL);
    }
    child.kill().ok();
    panic!("turbine-server never became ready on {addr}");
}

fn wait_with_timeout(mut child: Child, limit: Duration) -> Output {
    let started = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if started.elapsed() > limit {
            child.kill().ok();
            let out = child.wait_with_output().unwrap();
            panic!(
                "turbine-server did not exit within {limit:?}; stderr:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        std::thread::sleep(POLL);
    }
}

struct Response {
    status: u16,
    /// Response head with lower-cased header names.
    head: String,
    body: String,
}

impl Response {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }

    fn error_code(&self) -> String {
        self.json()["error"]["code"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    /// The `data:` payloads of an SSE body, in order.
    fn sse_data(&self) -> Vec<String> {
        self.body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }
}

fn write_request(conn: &mut TcpStream, method: &str, path: &str, body: Option<&str>) {
    let body = body.unwrap_or("");
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}

/// Reads the response head (status line and headers).
fn read_head(reader: &mut BufReader<TcpStream>) -> (u16, String) {
    let mut head = String::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).expect("read response head");
        assert!(n > 0, "connection closed mid-head: {head:?}");
        if line == "\r\n" {
            break;
        }
        head.push_str(&line.to_ascii_lowercase());
    }
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("bad status line: {head}"));
    (status, head)
}

/// Reads one chunk of a chunked body; `None` at the terminating chunk.
fn read_chunk(reader: &mut BufReader<TcpStream>) -> Option<String> {
    let mut size_line = String::new();
    reader.read_line(&mut size_line).expect("read chunk size");
    let size = usize::from_str_radix(size_line.trim(), 16)
        .unwrap_or_else(|_| panic!("bad chunk size line {size_line:?}"));
    let mut data = vec![0u8; size + 2];
    reader.read_exact(&mut data).expect("read chunk");
    if size == 0 {
        return None;
    }
    data.truncate(size);
    Some(String::from_utf8(data).expect("UTF-8 chunk"))
}

fn read_body(reader: &mut BufReader<TcpStream>, head: &str) -> String {
    if head.contains("transfer-encoding: chunked") {
        let mut body = String::new();
        while let Some(chunk) = read_chunk(reader) {
            body.push_str(&chunk);
        }
        body
    } else {
        let mut body = String::new();
        reader.read_to_string(&mut body).expect("read body");
        body
    }
}

fn request(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Response {
    let mut conn = TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    write_request(&mut conn, method, path, body);
    let mut reader = BufReader::new(conn);
    let (status, head) = read_head(&mut reader);
    let body = read_body(&mut reader, &head);
    Response { status, head, body }
}

/// The value of an un-labelled or labelled sample line `<series> <value>` in `/metrics`.
fn sample(metrics: &str, series: &str) -> Option<f64> {
    metrics.lines().find_map(|l| {
        l.strip_prefix(series)
            .and_then(|rest| rest.strip_prefix(' '))
            .and_then(|v| v.trim().parse().ok())
    })
}

#[test]
fn completions_stream_and_non_stream() {
    let server = TinyServer::start("");

    let models = server.get("/v1/models");
    assert_eq!(models.status, 200, "{}", models.body);
    let models = models.json();
    assert_eq!(models["object"], "list");
    let card = &models["data"][0];
    assert_eq!(card["id"], server.model);
    assert_eq!(card["object"], "model");
    assert_eq!(card["owned_by"], "turbine");
    assert_eq!(card["max_model_len"], 512);
    assert!(card["created"].as_u64().unwrap() > 0);

    // Non-streaming completion: "Hello" is 5 byte tokens plus BOS.
    let resp = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Hello", "max_tokens": 8, "ignore_eos": true,
                "temperature": 0}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    assert_eq!(body["object"], "text_completion");
    assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(body["model"], server.model);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0]["text"].is_string());
    assert_eq!(body["usage"]["prompt_tokens"], 6);
    assert_eq!(body["usage"]["completion_tokens"], 8);
    assert_eq!(body["usage"]["total_tokens"], 14);

    // Streaming completion with usage: text chunks, a finish chunk, the usage chunk, [DONE].
    let resp = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Hello", "max_tokens": 6, "ignore_eos": true,
                "stream": true, "stream_options": {"include_usage": true}}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    assert!(
        resp.head.contains("content-type: text/event-stream"),
        "{}",
        resp.head
    );
    let data = resp.sse_data();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]), "{usage}");
    assert_eq!(usage["usage"]["prompt_tokens"], 6);
    assert_eq!(usage["usage"]["completion_tokens"], 6);
    assert_eq!(usage["usage"]["total_tokens"], 12);
    let finish = &chunks[chunks.len() - 2];
    assert_eq!(finish["choices"][0]["finish_reason"], "length", "{finish}");
    for chunk in &chunks[..chunks.len() - 2] {
        assert_eq!(chunk["object"], "text_completion", "{chunk}");
        assert!(chunk["choices"][0]["finish_reason"].is_null(), "{chunk}");
        assert!(!chunk["choices"][0]["text"].as_str().unwrap().is_empty());
    }

    // Chat, non-streaming.
    let messages = json!([{"role": "system", "content": "Be brief."},
                          {"role": "user", "content": "Hi"}]);
    let resp = server.post(
        "/v1/chat/completions",
        &json!({"model": server.model, "messages": messages, "max_tokens": 5,
                "ignore_eos": true, "chat_template_kwargs": {"date_string": "26 Jul 2024"}}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_string());
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(
        prompt_tokens > 20,
        "the template adds its headers: {prompt_tokens}"
    );
    assert_eq!(body["usage"]["completion_tokens"], 5);

    // Chat, streaming: role chunk first, content chunks, finish, usage, [DONE].
    let resp = server.post(
        "/v1/chat/completions",
        &json!({"model": server.model, "messages": messages, "max_tokens": 5,
                "ignore_eos": true, "chat_template_kwargs": {"date_string": "26 Jul 2024"},
                "stream": true, "stream_options": {"include_usage": true}}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let data = resp.sse_data();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let usage = chunks.last().unwrap();
    assert_eq!(usage["usage"]["prompt_tokens"], prompt_tokens);
    assert_eq!(usage["usage"]["completion_tokens"], 5);
    assert_eq!(
        usage["usage"]["total_tokens"].as_u64(),
        Some(prompt_tokens + 5)
    );
    assert_eq!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
        "length"
    );
    for chunk in &chunks[1..chunks.len() - 2] {
        assert!(
            chunk["choices"][0]["delta"]["content"].is_string(),
            "{chunk}"
        );
    }

    // Diagnostics carry the model.
    let status = server.get("/turbine/v1/status").json();
    assert_eq!(status["ready"], true);
    assert_eq!(status["model"]["served_name"], server.model);
    assert_eq!(status["model"]["architecture"], "LlamaForCausalLM");
    assert!(status["model"]["weight_bytes"].as_u64().unwrap() > 0);
    assert!(status["model"]["load_seconds"].as_f64().is_some());
}

#[test]
fn single_slot_and_cancel() {
    let server = TinyServer::start("");

    // A long streaming request holds the slot. Large events (20 logprobs each) and a client that
    // reads only the first chunk make the stream back-pressure the generation thread, so the
    // slot stays held until the client goes away.
    let long = json!({"model": server.model, "prompt": "Once upon a time", "max_tokens": 400,
                      "ignore_eos": true, "stream": true, "logprobs": 20});
    let mut conn = TcpStream::connect(server.addr).unwrap();
    conn.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    write_request(
        &mut conn,
        "POST",
        "/v1/completions",
        Some(&long.to_string()),
    );
    let mut reader = BufReader::new(conn);
    let (status, head) = read_head(&mut reader);
    assert_eq!(status, 200, "{head}");
    let first = read_chunk(&mut reader).expect("first SSE chunk");
    assert!(first.starts_with("data: "), "{first}");

    let busy = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Hi", "max_tokens": 1}),
    );
    assert_eq!(busy.status, 429, "{}", busy.body);
    assert_eq!(busy.error_code(), "engine_busy");
    assert!(busy.head.contains("retry-after: 1"), "{}", busy.head);

    // Disconnect; the slot must be free again within 1 s.
    drop(reader);
    let dropped = Instant::now();
    loop {
        let resp = server.post(
            "/v1/completions",
            &json!({"model": server.model, "prompt": "Hi", "max_tokens": 1}),
        );
        if resp.status == 200 {
            break;
        }
        assert_eq!(resp.status, 429, "{}", resp.body);
        assert!(
            dropped.elapsed() < Duration::from_secs(1),
            "slot still held {:?} after the client disconnected",
            dropped.elapsed()
        );
        std::thread::sleep(POLL);
    }
    assert!(dropped.elapsed() < Duration::from_secs(1));

    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="cancelled"}"#
        ),
        Some(1.0),
        "{metrics}"
    );
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"}"#
        ),
        Some(1.0),
        "{metrics}"
    );

    // A client that has read a whole response may start the next request at once: the slot
    // is free before the final event goes out.
    for i in 0..20 {
        let resp = server.post(
            "/v1/completions",
            &json!({"model": server.model, "prompt": "Hi", "max_tokens": 2, "ignore_eos": true}),
        );
        assert_eq!(resp.status, 200, "back-to-back request {i}: {}", resp.body);
    }
}

#[test]
fn request_validation() {
    let server = TinyServer::start("");
    let model = server.model.clone();
    let unsupported = [
        (
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                   "tools": [{"type": "function", "function": {"name": "f"}}]}),
            "tools",
        ),
        (
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                   "response_format": {"type": "json_object"}}),
            "response_format",
        ),
        (
            "/v1/completions",
            json!({"model": model, "prompt": "x", "n": 2}),
            "n",
        ),
        (
            "/v1/completions",
            json!({"model": model, "prompt": "x", "logit_bias": {"5": 1}}),
            "logit_bias",
        ),
        (
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]}),
            "image_url",
        ),
    ];
    for (path, body, field) in unsupported {
        let resp = server.post(path, &body);
        assert_eq!(resp.status, 400, "{field}: {}", resp.body);
        assert_eq!(
            resp.error_code(),
            "unsupported_parameter",
            "{field}: {}",
            resp.body
        );
        assert!(resp.body.contains(field), "{field}: {}", resp.body);
    }

    let resp = server.post(
        "/v1/completions",
        &json!({"model": "not-the-model", "prompt": "x"}),
    );
    assert_eq!(resp.status, 404, "{}", resp.body);
    assert_eq!(resp.error_code(), "model_not_found");

    // 600 byte tokens (plus BOS) against max_seq_len 512.
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "a".repeat(600), "max_tokens": 1}),
    );
    assert_eq!(resp.status, 400, "{}", resp.body);
    assert_eq!(resp.error_code(), "context_length_exceeded");
    // Prompt + max_tokens beyond the context, even with a short prompt.
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "abc", "max_tokens": 510}),
    );
    assert_eq!(resp.status, 400, "{}", resp.body);
    assert_eq!(resp.error_code(), "context_length_exceeded");

    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "x", "max_tokens": 2, "foo": {"bar": 1}}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);

    let metrics = server.metrics();
    let rejected = sample(
        &metrics,
        r#"turbine_requests_total{endpoint="/v1/completions",outcome="rejected"}"#,
    );
    assert_eq!(rejected, Some(5.0), "{metrics}");
}

/// Writes the config, runs the server and returns its exit code and stderr; the port must still
/// be free afterwards when `bound_after` is false.
fn run_failing(dir: &TempDir, yaml: &str) -> (Option<i32>, String) {
    let config = dir.path().join("config.yaml");
    std::fs::write(&config, yaml).unwrap();
    let out = wait_with_timeout(spawn(&config), Duration::from_secs(60));
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn startup_failures_exit_1() {
    let dir = TempDir::new("turbine-startup-failures");

    let assert_exit_1 = |yaml: &str, addr: SocketAddr, needle: &str| {
        let (code, stderr) = run_failing(&dir, yaml);
        assert_eq!(code, Some(1), "{needle}: stderr:\n{stderr}");
        assert!(stderr.contains(needle), "{needle}: stderr:\n{stderr}");
        // Failed before binding.
        TcpListener::bind(addr).expect("the configured port must still be free");
    };

    // Missing model.path directory.
    let missing: PathBuf = dir.path().join("no-such-model");
    let addr = free_addr();
    assert_exit_1(
        &config_yaml(&missing, addr, ""),
        addr,
        &missing.display().to_string(),
    );

    // Pickle-only checkpoint.
    let pickle = dir.path().join("pickle-only");
    write_tiny_llama(&pickle, 7);
    std::fs::remove_file(pickle.join("model.safetensors")).unwrap();
    std::fs::write(pickle.join("pytorch_model.bin"), b"\x80\x02}q\x00.").unwrap();
    let addr = free_addr();
    assert_exit_1(&config_yaml(&pickle, addr, ""), addr, "pickle");

    // Unsupported architecture.
    let qwen = dir.path().join("qwen");
    write_tiny_llama(&qwen, 7);
    let mut cfg: Value =
        serde_json::from_slice(&std::fs::read(qwen.join("config.json")).unwrap()).unwrap();
    cfg["architectures"] = json!(["Qwen3MoeForCausalLM"]);
    std::fs::write(qwen.join("config.json"), cfg.to_string()).unwrap();
    let addr = free_addr();
    assert_exit_1(&config_yaml(&qwen, addr, ""), addr, "Qwen3MoeForCausalLM");

    // hip with a kernel library that does not exist.
    let tiny = dir.path().join("tiny");
    write_tiny_llama(&tiny, 7);
    let addr = free_addr();
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: hip\n  \
         kernel_library: /nonexistent/libturbine_hip.so\n",
        tiny.display()
    );
    assert_exit_1(&yaml, addr, "/nonexistent/libturbine_hip.so");

    // cuda is a configuration error until phase-2b-nvidia.
    let addr = free_addr();
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: cuda\n",
        tiny.display()
    );
    let (code, stderr) = run_failing(&dir, &yaml);
    assert_eq!(code, Some(2), "stderr:\n{stderr}");
    assert!(stderr.contains("phase-2b-nvidia"), "stderr:\n{stderr}");
}

#[test]
fn phase1_metrics() {
    let server = TinyServer::start("");
    let resp = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Hello", "max_tokens": 7, "ignore_eos": true}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let completion_tokens = resp.json()["usage"]["completion_tokens"].as_f64().unwrap();
    assert_eq!(completion_tokens, 7.0);

    let metrics = server.metrics();
    let load = sample(&metrics, "turbine_model_load_seconds");
    assert!(load.is_some_and(|s| s > 0.0), "{metrics}");
    let weights = sample(&metrics, r#"turbine_model_weight_bytes{format="bf16"}"#);
    assert!(weights.is_some_and(|b| b > 0.0), "{metrics}");
    assert!(
        metrics.contains(r#"turbine_kernel_provider_selected{op="gemm",provider="cpu-reference""#),
        "{metrics}"
    );
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"}"#
        ),
        Some(1.0),
        "{metrics}"
    );
    assert_eq!(
        sample(&metrics, "turbine_request_ttft_seconds_count"),
        Some(1.0),
        "{metrics}"
    );
    assert_eq!(
        sample(&metrics, r#"turbine_tokens_total{kind="generated"}"#),
        Some(completion_tokens),
        "{metrics}"
    );
    assert_eq!(
        sample(&metrics, r#"turbine_tokens_total{kind="prompt"}"#),
        Some(6.0),
        "{metrics}"
    );
    assert!(
        metrics.contains(r#"turbine_forward_seconds_count{phase="prefill"} 1"#),
        "{metrics}"
    );
}
