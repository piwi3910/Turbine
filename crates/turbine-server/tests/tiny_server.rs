//! Black-box tests of `turbine-server` serving the tiny synthetic checkpoint on the `cpu`
//! reference backend (P1 S-10, S-14; P2 S-3, S-6, S-7, S-8, S-10, S-11, S-14, S-17, S-18):
//! OpenAI shapes, queueing and cancellation, the queue bound, KV release on disconnect, the
//! diagnostics documents, slow clients and the request and queue timeouts, request validation,
//! startup failures, the Phase 1 metrics, the Phase 2 request fields, preemption, structured
//! output, tool calls and the Phase 2 metrics and log reasons; the Phase 2c iteration stage
//! breakdown (P2c S-1).
//!
//! Streams that must stay open are made to back-pressure the engine: large events (20 logprobs
//! each), more `max_tokens` than the path to the client buffers on a checkpoint patched to 8192
//! positions, and a client that stops reading after the first chunk — the request's output
//! channel (256 events) fills and the engine pauses it, holding its KV blocks, until the client
//! reads on or goes away. The kernel socket buffers between the two are kept to a few KiB on
//! both ends ([`SEND_BUFFER_ENV`] on the server, [`held_connection`] on the client), so the
//! pause comes after a bounded few hundred tokens (see [`LONG_TOKENS`]) instead of after however
//! many megabytes the loopback autotuning allows.
//!
//! Every wait is bounded and polls; ports come from binding `127.0.0.1:0`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use turbine_core::request::{GenerationEvent, GenerationRequest, SamplingParams, StopConditions};
use turbine_core::types::{DeviceId, Priority, RequestId};
use turbine_kernels::{KernelMetrics, KernelRegistry, cpu_reference_provider};
use turbine_model::executor::{self, ExecutorOptions, SequenceKv};
use turbine_model::testing::TempDir;
use turbine_model::testing::naive;
use turbine_model::testing::tiny::{
    TINY_EOS, TinyOptions, write_tiny_family, write_tiny_llama, write_tiny_llama_with,
};
use turbine_model::{
    GenerateOptions, MAX_STAGING_BYTES, SafetensorsIndex, Tokenizer, WeightLoader, generate,
    llama_slots,
};
use turbine_observability::MetricsRegistry;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

const POLL: Duration = Duration::from_millis(20);
const READY_LIMIT: Duration = Duration::from_secs(60);
/// Launches of one test server before a lost port race fails the test.
const LAUNCH_ATTEMPTS: usize = 5;
/// The server's test-only cap on each accepted connection's kernel send buffer.
const SEND_BUFFER_ENV: &str = "TURBINE_TEST_SOCKET_SEND_BUFFER";
/// Kernel buffer bytes on each end of a held stream (Linux doubles and floors them).
const HELD_SOCKET_BUFFER: usize = 4096;

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
        .env(SEND_BUFFER_ENV, HELD_SOCKET_BUFFER.to_string())
        .env("RUST_LOG", "info")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn turbine-server")
}

/// Config for the cpu backend on `model_dir` with a 16 MiB KV pool (512 tiny-model blocks of
/// 128 tokens, 32 KiB each); `extra` is appended verbatim and must not repeat the `kv` key.
fn config_yaml(model_dir: &Path, addr: SocketAddr, extra: &str) -> String {
    config_yaml_with(model_dir, addr, "", "", "", "16MiB", extra)
}

/// [`config_yaml`] with more `model`, `server` and `execution` keys (`model_extra`,
/// `server_extra`, `execution_extra`), each verbatim with lines indented by two spaces, and a
/// `kv.gpu.max_bytes` of `kv_bytes`.
///
/// These are the Phase 1/2 serving tests: the pressure state machine is off
/// (`reliability.enabled: false`, state fixed at GREEN; worst-case KV admission, the admission
/// queue and the circuit breaker stay), and the latency-drift circuit triggers are out of reach
/// — debug builds of the tests running side by side make step times swing far beyond any real
/// drift. `tests/fault.rs` and the reliability crates test the pressure behaviour. The config
/// ends with its `reliability` section, so `extra` lines indented by two spaces continue it.
fn config_yaml_with(
    model_dir: &Path,
    addr: SocketAddr,
    model_extra: &str,
    server_extra: &str,
    execution_extra: &str,
    kv_bytes: &str,
    extra: &str,
) -> String {
    format!(
        "model:\n  path: {}\n{model_extra}server:\n  listen: {addr}\n{server_extra}execution:\n  \
         backend: cpu\n{execution_extra}kv:\n  gpu:\n    max_bytes: {kv_bytes}\nreliability:\n  \
         enabled: false\n  emergency_vram_reserve: 1MiB\n  circuit:\n    \
         latency_drift_degraded: 1000.0\n    latency_drift_open: 2000.0\n{extra}",
        model_dir.display()
    )
}

/// How [`TinyServer::launch`] sets up a server.
struct Setup<'a> {
    /// Extra `model` keys, e.g. `"  tool_call_parser: llama3_json\n"`.
    model_extra: &'a str,
    /// Extra `server` keys, e.g. `"  request_timeout: 1s\n"`.
    server_extra: &'a str,
    /// Extra `execution` keys, e.g. `"  device_sampling: false\n"`.
    execution_extra: &'a str,
    /// `kv.gpu.max_bytes`.
    kv_bytes: &'a str,
    /// Appended to the config.
    extra: &'a str,
    /// `max_position_embeddings` patched into the checkpoint.
    max_positions: Option<u32>,
    /// The checkpoint's chat template renders `tools`.
    template_with_tools: bool,
    /// Keep the server's stderr (its log) for [`TinyServer::logs`].
    capture_logs: bool,
}

impl Default for Setup<'_> {
    fn default() -> Self {
        Setup {
            model_extra: "",
            server_extra: "",
            execution_extra: "",
            kv_bytes: "16MiB",
            extra: "",
            max_positions: None,
            template_with_tools: true,
            capture_logs: false,
        }
    }
}

/// `max_position_embeddings` of the checkpoint `TinyServer::start_long` serves.
const LONG_POSITIONS: u32 = 8192;
/// Tokens a held stream asks for: well over what the output channel (256 events), hyper's write
/// queue and the capped socket buffers hold together, so a stream that stops reading pauses
/// long before it could finish — after about 300 tokens on Linux (up to 500 on a loaded host),
/// where both socket caps hold, and about 880 on macOS, whose loopback keeps a receive buffer of about 320 KiB whatever
/// `SO_RCVBUF` says. Not more: debug-build attention cost grows with the context, and some
/// tests read the streams to the end.
const LONG_TOKENS: u32 = 1500;
/// `server` keys for tests whose streams must stay paused longer than the default 30 s
/// `server.slow_client_timeout` might allow on a loaded host.
const HOLD_PAUSED: &str = "  slow_client_timeout: 10m\n";
/// KV blocks of the 16 MiB test pool (the default 128-token page, 32 KiB per tiny block).
const POOL_BLOCKS: u64 = 512;

/// A running server on the tiny checkpoint; killed on drop.
struct TinyServer {
    child: Child,
    addr: SocketAddr,
    model: String,
    /// The server's stderr so far, when captured.
    logs: Option<Arc<Mutex<String>>>,
    _dir: TempDir,
}

impl TinyServer {
    fn start(extra: &str) -> TinyServer {
        TinyServer::start_with("", extra, None)
    }

    /// The tiny checkpoint patched to `LONG_POSITIONS` positions, so streams can be held open.
    fn start_long(extra: &str) -> TinyServer {
        TinyServer::start_with("", extra, Some(LONG_POSITIONS))
    }

    /// [`TinyServer::start_long`] with `server_extra` in the `server` section.
    fn start_long_with(server_extra: &str, extra: &str) -> TinyServer {
        TinyServer::start_with(server_extra, extra, Some(LONG_POSITIONS))
    }

    fn start_with(server_extra: &str, extra: &str, max_positions: Option<u32>) -> TinyServer {
        TinyServer::launch(&Setup {
            server_extra,
            extra,
            max_positions,
            ..Setup::default()
        })
    }

    fn launch(setup: &Setup<'_>) -> TinyServer {
        let dir = TempDir::new("turbine-tiny-server");
        let model_dir = dir.path().join("tiny-llama");
        write_tiny_llama_with(
            &model_dir,
            7,
            &TinyOptions {
                template_with_tools: setup.template_with_tools,
                ..TinyOptions::default()
            },
        );
        if let Some(positions) = setup.max_positions {
            let path = model_dir.join("config.json");
            let mut cfg: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            cfg["max_position_embeddings"] = json!(positions);
            std::fs::write(&path, cfg.to_string()).unwrap();
        }
        let config = dir.path().join("config.yaml");
        // A port from `free_addr` is free only until something else binds it, and tests start
        // servers in parallel: when this server loses its port to another test's server, it
        // exits without logging `listening` and the launch retries on a fresh port. Readiness
        // is only trusted after this server logged `listening`, so another test's server
        // answering `/ready` on the same port is never mistaken for this one.
        for _ in 0..LAUNCH_ATTEMPTS {
            let addr = free_addr();
            let yaml = config_yaml_with(
                &model_dir,
                addr,
                setup.model_extra,
                setup.server_extra,
                setup.execution_extra,
                setup.kv_bytes,
                setup.extra,
            );
            std::fs::write(&config, yaml).unwrap();
            let mut child = spawn(&config);
            let logs = drain_stderr(&mut child);
            match wait_until_ready(&mut child, addr, &logs) {
                Ready::Serving => {
                    return TinyServer {
                        child,
                        addr,
                        model: "tiny-llama".into(),
                        logs: setup.capture_logs.then_some(logs),
                        _dir: dir,
                    };
                }
                Ready::PortTaken => continue,
            }
        }
        panic!("turbine-server lost its port {LAUNCH_ATTEMPTS} times in a row");
    }

    /// The JSON log records written so far (the server must log with `format: json`).
    fn log_records(&self) -> Vec<Value> {
        let logs = self.logs.as_ref().expect("logs captured");
        let text = logs.lock().unwrap().clone();
        text.lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
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

    /// Replica 0's scheduler document (the route is keyed by replica, P5).
    fn scheduler(&self) -> Value {
        let resp = self.get("/turbine/v1/scheduler");
        assert_eq!(resp.status, 200, "{}", resp.body);
        resp.json()["0"].clone()
    }

    /// The `l0` tier object of `/turbine/v1/kv` (Phase 4 lists `l0`, `l1`, `l2`).
    fn kv_tier(&self) -> Value {
        let resp = self.get("/turbine/v1/kv");
        assert_eq!(resp.status, 200, "{}", resp.body);
        let doc = resp.json();
        assert_eq!(doc["tiers"][0]["tier"], "l0", "{doc}");
        doc["tiers"][0].clone()
    }

    /// L0 blocks requests hold. From Phase 4 a finished request's full prompt blocks stay
    /// cached (allocated but unreferenced) for prefix reuse, so "the KV was released" means no
    /// block is referenced any more.
    fn blocks_used(&self) -> u64 {
        self.kv_tier()["referenced_blocks"].as_u64().unwrap()
    }

    /// A streaming completion that holds its KV (see the module comment).
    fn long_stream_body(&self) -> Value {
        json!({"model": self.model, "prompt": "Once upon a time", "max_tokens": LONG_TOKENS,
               "ignore_eos": true, "stream": true, "logprobs": 20})
    }

    /// Opens a held stream and reads its first chunk.
    fn hold_stream(&self) -> OpenStream {
        OpenStream::open(self.addr, &self.long_stream_body(), true)
    }
}

/// Running requests in a scheduler document.
fn running(doc: &Value) -> u64 {
    ["prefilling", "decoding", "paused"]
        .iter()
        .map(|k| doc[*k].as_u64().unwrap())
        .sum()
}

/// Polls `cond` every `POLL` until it holds; panics naming `what` after `limit`.
fn wait_for(limit: Duration, what: &str, mut cond: impl FnMut() -> bool) {
    let started = Instant::now();
    while !cond() {
        assert!(
            started.elapsed() < limit,
            "{what}: not reached within {limit:?}"
        );
        std::thread::sleep(POLL);
    }
}

/// A connection whose receive buffer is [`HELD_SOCKET_BUFFER`] bytes, set before connecting so
/// the advertised window stays that small: a client that stops reading stalls the server after
/// a few KiB instead of the megabytes receive autotuning would take.
fn held_connection(addr: SocketAddr) -> TcpStream {
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        None,
    )
    .unwrap();
    socket.set_recv_buffer_size(HELD_SOCKET_BUFFER).unwrap();
    socket.connect(&addr.into()).unwrap();
    socket.into()
}

/// A streaming completion whose body is read only on demand.
struct OpenStream {
    reader: BufReader<TcpStream>,
}

impl OpenStream {
    /// Sends `body` to `/v1/completions` and reads the response head; with `first_chunk` also
    /// the first SSE chunk (a queued request produces none until it runs).
    fn open(addr: SocketAddr, body: &Value, first_chunk: bool) -> OpenStream {
        let mut conn = held_connection(addr);
        conn.set_read_timeout(Some(Duration::from_secs(60)))
            .unwrap();
        write_request(
            &mut conn,
            "POST",
            "/v1/completions",
            Some(&body.to_string()),
        );
        let mut reader = BufReader::new(conn);
        let (status, head) = read_head(&mut reader);
        assert_eq!(status, 200, "{head}");
        assert!(head.contains("transfer-encoding: chunked"), "{head}");
        if first_chunk {
            let first = read_chunk(&mut reader).expect("first SSE chunk");
            assert!(first.starts_with("data: "), "{first}");
        }
        OpenStream { reader }
    }

    /// Reads the rest of the body and returns its `data:` payloads.
    fn read_rest(mut self) -> Vec<String> {
        let mut body = String::new();
        while let Some(chunk) = read_chunk(&mut self.reader) {
            body.push_str(&chunk);
        }
        body.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }
}

/// The `finish_reason` of a completed stream's finish chunk (its data ends in `[DONE]`).
fn stream_finish_reason(data: &[String]) -> String {
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    data.iter()
        .rev()
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .find_map(|c| {
            c["choices"][0]["finish_reason"]
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_else(|| panic!("no finish_reason in {data:?}"))
}

impl Drop for TinyServer {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

/// Collects `child`'s stderr on a thread, so the server never blocks on a full pipe.
fn drain_stderr(child: &mut Child) -> Arc<Mutex<String>> {
    let logs = Arc::new(Mutex::new(String::new()));
    let mut stderr = BufReader::new(child.stderr.take().expect("stderr is piped"));
    let sink = Arc::clone(&logs);
    std::thread::spawn(move || {
        let mut line = String::new();
        while stderr.read_line(&mut line).is_ok_and(|n| n > 0) {
            sink.lock().unwrap().push_str(&line);
            line.clear();
        }
    });
    logs
}

/// How [`wait_until_ready`] ended.
enum Ready {
    /// This server bound its port and answers `/ready` 200.
    Serving,
    /// Another process took the port first; this server exited.
    PortTaken,
}

fn wait_until_ready(child: &mut Child, addr: SocketAddr, logs: &Mutex<String>) -> Ready {
    let started = Instant::now();
    while started.elapsed() < READY_LIMIT {
        if let Some(status) = child.try_wait().unwrap() {
            // Give the drain thread a moment to read the last lines.
            std::thread::sleep(POLL);
            let stderr = logs.lock().unwrap().clone();
            if stderr.contains("cannot bind") {
                return Ready::PortTaken;
            }
            panic!("turbine-server exited early ({status}); stderr:\n{stderr}");
        }
        if logs.lock().unwrap().contains("listening")
            && TcpStream::connect(addr).is_ok()
            && request(addr, "GET", "/ready", None).status == 200
        {
            return Ready::Serving;
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
    // P5 S-4: the cpu backend's plan — one replica on execution.device, no communicator.
    let parallel = &status["parallel"];
    assert_eq!(parallel["tp"], 1, "{parallel}");
    assert_eq!(parallel["dp"], 1, "{parallel}");
    assert_eq!(parallel["backend"], "host", "{parallel}");
    assert_eq!(parallel["mode"], "local", "{parallel}");
    assert_eq!(parallel["groups"][0]["replica"], 0, "{parallel}");
    assert_eq!(parallel["groups"][0]["ranks"][0]["device"], 0, "{parallel}");
    assert!(
        parallel["groups"][0]["ranks"][0]["host"].is_string(),
        "{parallel}"
    );
    assert_eq!(parallel["plan_reasons"], json!(["execution_device"]));
    assert!(
        server
            .metrics()
            .contains(r#"turbine_parallel_info{tp="1",dp="1",backend="host",mode="local"} 1"#),
        "turbine_parallel_info"
    );

    // P5 S-1: the node graph captured at startup (whatever this host's sysfs holds; discovery
    // never fails), the same document with `scope=node`, and no cluster scope before Phase 6.
    let topology = server.get("/turbine/v1/topology");
    assert_eq!(topology.status, 200, "{}", topology.body);
    let graph = topology.json();
    assert!(graph["node"]["hostname"].is_string(), "{graph}");
    assert!(graph["node"]["captured_at"].is_string(), "{graph}");
    assert!(graph["vertices"].is_array(), "{graph}");
    assert!(graph["edges"].is_array(), "{graph}");
    assert_eq!(
        server.get("/turbine/v1/topology?scope=node").json(),
        graph,
        "the graph is captured once"
    );
    assert_eq!(server.get("/turbine/v1/topology?scope=cluster").status, 400);
}

/// Phase 2m: `/turbine/v1/status` names the module chosen at each extension point and, per op
/// config of the kernel registry, the provider and implementation that serve it.
#[test]
fn status_reports_modules_and_kernels() {
    let server = TinyServer::launch(&Setup {
        model_extra: "  tool_call_parser: llama3_json\n",
        ..Setup::default()
    });
    let status = server.get("/turbine/v1/status").json();
    assert_eq!(
        status["modules"],
        json!({"family": "llama", "tool_format": "llama3_json", "weight_format": "bf16",
               "backend": "cpu", "card_profile": null, "scheduling_policy": "default",
               "eviction_policy": "cost_aware"}),
        "{status}"
    );

    // The registry the server builds for the tiny checkpoint: the executor's requirements with
    // the default (fused) op sequence and the device logits reduction, on the cpu provider.
    let dir = TempDir::new("turbine-tiny-kernels");
    let spec = write_tiny_llama(dir.path(), 7);
    let provider = cpu_reference_provider();
    let order = [provider.id()];
    let providers = vec![provider];
    let mut reqs = executor::available_requirements(
        &spec.config,
        128,
        ExecutorOptions::from_fused_ops(true),
        &providers,
    );
    let reduce = executor::logits::reduce_requirement(&spec.config);
    if reduce.spec.supported_by(providers[0].as_ref()) {
        reqs.push(reduce);
    }
    let registry = KernelRegistry::build(
        providers,
        &order,
        &reqs,
        &KernelMetrics::register(&MetricsRegistry::new()),
        None,
    )
    .unwrap();
    let expected: Vec<Value> = registry
        .selections()
        .iter()
        .map(|s| {
            json!({"op": s.op.as_str(), "config": s.config, "provider": "cpu-reference",
                   "implementation": s.implementation, "impl_provider": "cpu-reference",
                   "reason": s.reason, "reason_code": "provider_internal", "tiers": []})
        })
        .collect();
    assert!(!expected.is_empty());
    assert_eq!(status["kernels"], Value::Array(expected), "{status}");
}

/// Phase 2m S-11: the support-matrix row resolved at startup is the `support` key of
/// `/turbine/v1/status`, drives `turbine_support_matrix_status`, and is logged as
/// `event="support_matrix"` (WARN for the CPU reference provider's experimental row).
#[test]
fn status_reports_support_row() {
    let server = TinyServer::launch(&Setup {
        capture_logs: true,
        extra: "logging:\n  format: json\n",
        ..Setup::default()
    });
    let status = server.get("/turbine/v1/status").json();
    assert_eq!(
        status["support"],
        json!({"vendor": "cpu", "arch": "cpu", "architecture": "LlamaForCausalLM",
               "weight_format": "bf16", "kv_format": "bf16", "speculative": "none",
               "status": "experimental", "reason": null}),
        "{status}"
    );
    let metrics = server.metrics();
    for (label, value) in [("experimental", 1), ("supported", 0), ("unsupported", 0)] {
        let line = format!("turbine_support_matrix_status{{status=\"{label}\"}} {value}");
        assert!(metrics.lines().any(|l| l == line), "{line}:\n{metrics}");
    }
    let logs = server.logs.as_ref().expect("logs captured");
    wait_for(
        Duration::from_secs(10),
        "the support-matrix log line",
        || {
            let text = logs.lock().unwrap();
            text.lines().any(|l| {
                l.contains(r#""event":"support_matrix""#)
                    && l.contains("experimental")
                    && l.contains("cpu/cpu/LlamaForCausalLM/bf16/bf16/none")
            })
        },
    );
}

/// Phase 2m S-5: on the CPU backend every `kernels` entry of `/turbine/v1/status` is served by
/// `cpu-reference`, which chooses internally (`reason_code` `provider_internal`), with the
/// implementation names main reports. Breaks if the registry's selection changes what a served
/// model runs.
#[test]
fn status_reports_kernel_choices() {
    let server = TinyServer::launch(&Setup::default());
    let status = server.get("/turbine/v1/status").json();
    let kernels = status["kernels"].as_array().expect("kernels array");
    assert!(!kernels.is_empty(), "{status}");
    let main_names: std::collections::HashMap<&str, &str> = [
        ("embedding", "cpu_embedding"),
        ("rmsnorm", "cpu_rmsnorm"),
        ("gemm", "cpu_gemm_f32acc"),
        ("rope", "cpu_rope_half_split"),
        ("attention_prefill_paged", "cpu_attention_paged_f32acc"),
        ("attention_decode_paged", "cpu_attention_paged_f32acc"),
        ("add", "cpu_add"),
        ("add_rmsnorm", "cpu_add_rmsnorm"),
        ("silu_mul", "cpu_silu_mul"),
        ("copy_blocks", "cpu_copy_blocks"),
        ("logits_reduce", "cpu_logits_reduce"),
    ]
    .into_iter()
    .collect();
    for k in kernels {
        let op = k["op"].as_str().expect("op");
        assert_eq!(k["provider"], "cpu-reference", "{k}");
        assert_eq!(k["impl_provider"], "cpu-reference", "{k}");
        assert_eq!(k["reason_code"], "provider_internal", "{k}");
        assert_eq!(k["tiers"], json!([]), "{k}");
        let want = main_names
            .get(op)
            .unwrap_or_else(|| panic!("unexpected op {op} in {status}"));
        assert_eq!(k["implementation"], *want, "{k}");
    }
}

/// Contract §20.2 (rewritten for Phase 2): with one running request allowed
/// (`continuous_batching: false`) a second concurrent request is queued — no more 429
/// `engine_busy` — and completes once the first client goes away, whose blocks are freed.
#[test]
fn single_slot_and_cancel() {
    let server = TinyServer::start_long("scheduler:\n  continuous_batching: false\n");

    let first = server.hold_stream();
    let addr = server.addr;
    let model = server.model.clone();
    let second = std::thread::spawn(move || {
        request(
            addr,
            "POST",
            "/v1/completions",
            Some(
                &json!({"model": model, "prompt": "Hi", "max_tokens": 2, "ignore_eos": true})
                    .to_string(),
            ),
        )
    });
    wait_for(Duration::from_secs(10), "second request queued", || {
        server.scheduler()["waiting"] == 1
    });
    let doc = server.scheduler();
    assert_eq!(doc["config"]["max_running_requests"], 1, "{doc}");
    assert_eq!(running(&doc), 1, "{doc}");
    assert!(server.blocks_used() > 0);
    assert!(!second.is_finished(), "the queued request must wait");

    // Disconnect: the queued request runs and completes; every block returns to the pool.
    drop(first);
    let second = second.join().unwrap();
    assert_eq!(second.status, 200, "{}", second.body);
    assert_eq!(second.json()["choices"][0]["finish_reason"], "length");
    wait_for(Duration::from_secs(1), "blocks released", || {
        server.blocks_used() == 0
    });

    let metrics = server.metrics();
    for (series, value) in [
        (
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="cancelled"}"#,
            1.0,
        ),
        (
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"}"#,
            1.0,
        ),
        (
            r#"turbine_requests_cancelled_total{reason="client_disconnect"}"#,
            1.0,
        ),
    ] {
        assert_eq!(sample(&metrics, series), Some(value), "{series}\n{metrics}");
    }

    // A client that has read a whole response may start the next request at once.
    for i in 0..20 {
        let resp = server.post(
            "/v1/completions",
            &json!({"model": server.model, "prompt": "Hi", "max_tokens": 2, "ignore_eos": true}),
        );
        assert_eq!(resp.status, 200, "back-to-back request {i}: {}", resp.body);
    }
}

/// P2 S-7, P3 S-9: the waiting queue — from Phase 3 the admission queue, bounded by
/// `reliability.admission.max_queue` (C-1) — holds requests behind a full batch; the excess gets
/// 429 `queue_full` with `retry-after`, counted by reason.
#[test]
fn queue_full_429() {
    // `extra` lines indented by two spaces continue the base config's `reliability` section.
    let server = TinyServer::start_long_with(
        HOLD_PAUSED,
        "  admission:\n    max_queue: 2\nscheduler:\n  max_running_requests: 1\n",
    );
    let held = server.hold_stream();
    wait_for(Duration::from_secs(10), "first request running", || {
        let doc = server.scheduler();
        running(&doc) == 1 && doc["waiting"] == 0
    });

    // Five more at once: two fit the queue, three are refused.
    let clients: Vec<_> = (0..5)
        .map(|_| {
            let addr = server.addr;
            let model = server.model.clone();
            std::thread::spawn(move || {
                request(
                    addr,
                    "POST",
                    "/v1/completions",
                    Some(
                        &json!({"model": model, "prompt": "Hi", "max_tokens": 4,
                                "ignore_eos": true})
                        .to_string(),
                    ),
                )
            })
        })
        .collect();
    wait_for(Duration::from_secs(10), "three rejections", || {
        clients.iter().filter(|c| c.is_finished()).count() == 3
    });
    let (done, queued): (Vec<_>, Vec<_>) = clients.into_iter().partition(|c| c.is_finished());
    for c in done {
        let resp = c.join().unwrap();
        assert_eq!(resp.status, 429, "{}", resp.body);
        assert_eq!(resp.error_code(), "queue_full");
        assert!(resp.head.contains("retry-after: 1"), "{}", resp.head);
    }
    // The document is published after each engine turn, so it may trail the admissions.
    wait_for(
        Duration::from_secs(5),
        "two queued behind the held one",
        || {
            let doc = server.scheduler();
            doc["waiting"] == 2 && running(&doc) == 1
        },
    );

    // The held stream reads on and completes; then the two queued requests run.
    assert_eq!(stream_finish_reason(&held.read_rest()), "length");
    for c in queued {
        let resp = c.join().unwrap();
        assert_eq!(resp.status, 200, "{}", resp.body);
    }
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_admission_total{outcome="rejected",reason="queue_full"}"#
        ),
        Some(3.0),
        "{metrics}"
    );
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_admission_total{outcome="queued",reason="ok"}"#
        ),
        Some(3.0),
        "{metrics}"
    );
}

/// P2 S-8: dropped clients free their KV blocks within 1 s; the others keep theirs and free
/// them when they finish.
#[test]
fn disconnect_releases_kv() {
    // The streams stay paused while the others catch up (slowly on a loaded host): no
    // slow-client cancellation may end them first.
    let server = TinyServer::start_long_with(HOLD_PAUSED, "");
    let mut streams: Vec<OpenStream> = (0..8).map(|_| server.hold_stream()).collect();
    wait_for(Duration::from_secs(60), "all 8 streams paused", || {
        let doc = server.scheduler();
        doc["paused"] == 8 && doc["waiting"] == 0
    });
    // Paused requests hold their blocks and grow no more.
    let all8 = server.blocks_used();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(server.blocks_used(), all8);

    let kept = streams.split_off(4);
    drop(streams);
    let dropped = Instant::now();
    wait_for(
        Duration::from_secs(1),
        "dropped clients' blocks freed",
        || {
            let doc = server.scheduler();
            doc["paused"] == 4 && running(&doc) == 4
        },
    );
    let remaining = server.blocks_used();
    assert!(dropped.elapsed() < Duration::from_secs(1));
    // Every dropped request held at least its prompt block; the remaining four still hold
    // exactly theirs (they stay paused, so their usage is stable).
    assert!(remaining + 4 <= all8, "{remaining} of {all8}");
    assert!(remaining >= 4, "{remaining}");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(server.blocks_used(), remaining);
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_requests_cancelled_total{reason="client_disconnect"}"#
        ),
        Some(4.0),
        "{metrics}"
    );

    for s in kept {
        assert_eq!(stream_finish_reason(&s.read_rest()), "length");
    }
    wait_for(Duration::from_secs(1), "every block free", || {
        server.blocks_used() == 0
    });
    let tier = server.kv_tier();
    assert_eq!(tier["referenced_blocks"], 0, "{tier}");
}

/// P2 S-11: `/turbine/v1/scheduler` and `/turbine/v1/kv` have the Data shapes with counts that
/// agree with the metrics; P3 S-13: `/turbine/v1/pressure` is the pressure document (its KV pool
/// agreeing with the reservations of the four requests) and `/turbine/v1/status` names the
/// pressure and circuit states.
#[test]
fn diagnostics_shapes() {
    let server = TinyServer::start_long_with(
        HOLD_PAUSED,
        "scheduler:\n  max_running_requests: 2\n  max_queued_requests: 4\n",
    );
    // Two paused streams run; two more wait behind them. The state is then stable.
    let held: Vec<OpenStream> = (0..2).map(|_| server.hold_stream()).collect();
    let waiting: Vec<OpenStream> = (0..2)
        .map(|_| OpenStream::open(server.addr, &server.long_stream_body(), false))
        .collect();
    wait_for(Duration::from_secs(60), "2 paused and 2 waiting", || {
        let doc = server.scheduler();
        doc["paused"] == 2 && doc["waiting"] == 2
    });

    let doc = server.scheduler();
    assert_eq!(
        doc["config"],
        json!({"max_running_requests": 2, "max_batch_tokens": 8192,
               "prefill_chunk_tokens": 2048, "max_queued_requests": 4,
               "chunked_prefill": true}),
        "{doc}"
    );
    let keys = [
        "config",
        "waiting",
        "prefilling",
        "decoding",
        "paused",
        "constrained",
        "iterations_total",
        "preemptions_total",
        "last_iteration",
    ];
    let mut got: Vec<&str> = doc
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    got.sort_unstable();
    let mut want = keys.to_vec();
    want.sort_unstable();
    assert_eq!(got, want, "{doc}");
    for k in ["prefill_tokens", "decode_tokens", "requests", "duration_ms"] {
        assert!(doc["last_iteration"][k].is_number(), "{k}: {doc}");
    }
    assert_eq!(doc["constrained"], 0, "{doc}");
    assert_eq!(doc["preemptions_total"], 0, "{doc}");
    assert!(doc["iterations_total"].as_u64().unwrap() > 0, "{doc}");

    // The documents and the metrics agree (read back to back until neither moved in between).
    let mut agreed = false;
    for _ in 0..50 {
        let before = server.scheduler();
        let tier = server.kv_tier();
        let metrics = server.metrics();
        let after = server.scheduler();
        let counts = |d: &Value| {
            ["waiting", "prefilling", "decoding", "paused"].map(|k| d[k].as_u64().unwrap())
        };
        if counts(&before) != counts(&after) {
            continue;
        }
        let gauge = |series: &str| sample(&metrics, series).map(|v| v as u64);
        let [waiting, prefilling, decoding, paused] = counts(&after);
        assert_eq!(gauge("turbine_requests_queued"), Some(waiting), "{metrics}");
        for (state, n) in [
            ("prefilling", prefilling),
            ("decoding", decoding),
            ("paused", paused),
        ] {
            assert_eq!(
                gauge(&format!(r#"turbine_requests_active{{state="{state}"}}"#)),
                Some(n),
                "{state}\n{metrics}"
            );
        }
        assert_eq!(
            gauge(r#"turbine_kv_blocks{tier="l0",state="used"}"#),
            tier["blocks_used"].as_u64(),
            "{metrics}"
        );
        agreed = true;
        break;
    }
    assert!(agreed, "the scheduler document never held still");

    let tier = server.kv_tier();
    assert_eq!(tier["tier"], "l0", "{tier}");
    assert_eq!(tier["dtype"], "bf16", "{tier}");
    assert_eq!(tier["block_tokens"], 128, "{tier}");
    // 2 layers × 2 × 128 tokens × 2 KV heads × 16 dims × 2 bytes.
    assert_eq!(tier["block_bytes"], 32768, "{tier}");
    assert_eq!(tier["blocks_total"], POOL_BLOCKS, "{tier}");
    let used = tier["blocks_used"].as_u64().unwrap();
    assert!(used > 0, "{tier}");
    assert_eq!(
        tier["blocks_free"].as_u64(),
        Some(POOL_BLOCKS - used),
        "{tier}"
    );

    let pressure = server.get("/turbine/v1/pressure");
    assert_eq!(pressure.status, 200, "{}", pressure.body);
    let doc = pressure.json();
    for key in [
        "enabled",
        "state",
        "since",
        "dominant_signal",
        "exhaustion_horizon_seconds",
        "signals",
        "throttle",
        "memory",
        "admission",
        "circuit",
        "transitions",
    ] {
        assert!(doc.get(key).is_some(), "{key}: {doc}");
    }
    // The serving tests run with the pressure state machine off (see `config_yaml_with`).
    assert_eq!(doc["enabled"], false, "{doc}");
    assert_eq!(doc["state"], "GREEN", "{doc}");
    assert_eq!(doc["circuit"]["state"], "HEALTHY", "{doc}");
    let memory = &doc["memory"][0];
    assert_eq!(memory["memory_kind"], "dedicated", "{doc}");
    assert_eq!(memory["emergency_reserve_held"], true, "{doc}");
    let pool = |name: &str| {
        memory["pools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == name)
            .cloned()
            .unwrap_or_else(|| panic!("no {name} pool: {doc}"))
    };
    assert_eq!(pool("kv")["capacity_bytes"], POOL_BLOCKS * 32768, "{doc}");
    assert_eq!(pool("reserve")["used_bytes"], 1 << 20, "{doc}");
    assert!(pool("weights")["used_bytes"].as_u64().unwrap() > 0, "{doc}");
    // The two running requests hold worst-case reservations; the two queued ones hold none.
    let kv = pool("kv");
    let held_kv = kv["used_bytes"].as_u64().unwrap() + kv["reserved_bytes"].as_u64().unwrap();
    assert!(held_kv >= used * 32768, "{doc}");
    let status = server.get("/turbine/v1/status").json();
    assert!(status["pressure_state"].is_string(), "{status}");
    assert_eq!(status["circuit_state"], "HEALTHY", "{status}");
    drop(held);
    drop(waiting);
}
#[test]
fn request_validation() {
    let server = TinyServer::start("");
    let model = server.model.clone();
    // Phase 2 serves n, penalties, logit_bias, response_format and tools; what stays
    // unsupported is refused naming the field.
    let tools = json!([{"type": "function", "function": {"name": "f"}}]);
    let unsupported = [
        (
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                   "echo": true}),
            "echo",
        ),
        (
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                   "response_format": {"type": "json_object"}, "tools": tools,
                   "tool_choice": "required"}),
            "response_format",
        ),
        (
            "/v1/completions",
            json!({"model": model, "prompt": "x", "best_of": 2}),
            "best_of",
        ),
        (
            "/v1/completions",
            json!({"model": model, "prompt": "x", "suffix": "y"}),
            "suffix",
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
    // The same prompt without max_tokens (whose default, the rest of the context, is 0).
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "a".repeat(600)}),
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

    // Token ids are checked against the vocabulary (263 tiny-model tokens).
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "x", "logit_bias": {"263": 1}}),
    );
    assert_eq!(resp.status, 400, "{}", resp.body);
    assert_eq!(resp.error_code(), "invalid_request");
    assert!(resp.body.contains("logit_bias"), "{}", resp.body);

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
    assert_eq!(rejected, Some(7.0), "{metrics}");
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

    // Unsupported architecture (no registered family).
    let unknown = dir.path().join("gpt-oss");
    write_tiny_llama(&unknown, 7);
    let mut cfg: Value =
        serde_json::from_slice(&std::fs::read(unknown.join("config.json")).unwrap()).unwrap();
    cfg["architectures"] = json!(["GptOssForCausalLM"]);
    std::fs::write(unknown.join("config.json"), cfg.to_string()).unwrap();
    let addr = free_addr();
    assert_exit_1(&config_yaml(&unknown, addr, ""), addr, "GptOssForCausalLM");

    // hip with a kernel library that does not exist.
    let tiny = dir.path().join("tiny");
    write_tiny_llama(&tiny, 7);

    // The GPU KV tier is required: from Phase 4 the config itself rejects `kv.gpu.enabled:
    // false` ("L0 is required"), so this is exit 2 before any port is bound.
    let addr = free_addr();
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: cpu\n\
         kv:\n  gpu:\n    enabled: false\n",
        tiny.display()
    );
    let (code, stderr) = run_failing(&dir, &yaml);
    assert_eq!(code, Some(2), "kv.gpu.enabled: stderr:\n{stderr}");
    assert!(
        stderr.contains("kv.gpu.enabled: L0 is required"),
        "stderr:\n{stderr}"
    );
    TcpListener::bind(addr).expect("the configured port must still be free");

    // A kv pool (capped by kv.gpu.max_bytes) below one full-context sequence (P3 S-2: the
    // message names every pool and its bytes).
    let addr = free_addr();
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: cpu\n\
         kv:\n  gpu:\n    max_bytes: 16KiB\n",
        tiny.display()
    );
    assert_exit_1(&yaml, addr, "capped by kv.gpu.max_bytes");
    let (_, stderr) = run_failing(&dir, &yaml);
    for pool in ["weights=", "workspace=", "runtime=", "reserve=", "kv="] {
        assert!(stderr.contains(pool), "{pool}: stderr:\n{stderr}");
    }

    let addr = free_addr();
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: hip\n  \
         kernel_library: /nonexistent/libturbine_hip.so\n",
        tiny.display()
    );
    assert_exit_1(&yaml, addr, "/nonexistent/libturbine_hip.so");

    // cuda is a configuration error until a cuda backend is registered (phase-2b-nvidia).
    let addr = free_addr();
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: cuda\n",
        tiny.display()
    );
    let (code, stderr) = run_failing(&dir, &yaml);
    assert_eq!(code, Some(2), "stderr:\n{stderr}");
    assert!(
        stderr.contains("execution.backend: `cuda` is not registered (registered: cpu, hip)"),
        "stderr:\n{stderr}"
    );
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

#[test]
fn slow_client_paused_then_cancelled() {
    let server = TinyServer::start_long_with("  slow_client_timeout: 1s\n", "");
    let slow = server.hold_stream();
    wait_for(Duration::from_secs(30), "the slow client is paused", || {
        server.scheduler()["paused"] == 1
    });
    let paused_at = Instant::now();

    // Another stream keeps progressing while the slow one is paused.
    let other = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Hi", "max_tokens": 16, "ignore_eos": true,
                "stream": true}),
    );
    assert_eq!(other.status, 200, "{}", other.body);
    assert_eq!(stream_finish_reason(&other.sse_data()), "length");
    assert_eq!(
        server.scheduler()["paused"],
        1,
        "still paused before the timeout"
    );

    // After the timeout it is cancelled with `slow_client` and its blocks are free.
    wait_for(
        Duration::from_secs(5),
        "the slow client is cancelled",
        || {
            let doc = server.scheduler();
            running(&doc) == 0 && server.blocks_used() == 0
        },
    );
    assert!(
        paused_at.elapsed() >= Duration::from_millis(900),
        "cancelled after {:?}",
        paused_at.elapsed()
    );
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            "turbine_requests_cancelled_total{reason=\"slow_client\"}"
        ),
        Some(1.0),
        "{metrics}"
    );
    // Reading on shows what was buffered, then the error event and `[DONE]` (C-3).
    let data = slow.read_rest();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
    let error: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(error["error"]["code"], "slow_client", "{error}");
}

#[test]
fn request_and_queue_timeouts() {
    // server.request_timeout: a held stream ends with the error event, then `[DONE]` (C-3).
    let server = TinyServer::start_long_with("  request_timeout: 1s\n", "");
    // Timed from before the request is sent: its deadline runs from submission, which precedes
    // the first chunk `hold_stream` waits for.
    let opened = Instant::now();
    let held = server.hold_stream();
    // The document is published after each iteration, so it may trail the first chunk by one.
    wait_for(Duration::from_secs(5), "the held stream is running", || {
        running(&server.scheduler()) == 1
    });
    wait_for(
        Duration::from_secs(5),
        "the timed-out stream is dropped",
        || running(&server.scheduler()) == 0,
    );
    assert!(opened.elapsed() >= Duration::from_millis(900));
    assert_eq!(server.blocks_used(), 0);
    let data = held.read_rest();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
    let error: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(error["error"]["code"], "request_timeout", "{error}");
    assert_eq!(error["error"]["type"], "timeout", "{error}");

    // A non-streaming request past the deadline: 504 `request_timeout`.
    let resp = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Once upon a time", "max_tokens": LONG_TOKENS,
                "ignore_eos": true, "logprobs": 20}),
    );
    assert_eq!(resp.status, 504, "{}", resp.body);
    assert_eq!(resp.error_code(), "request_timeout");
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            "turbine_requests_cancelled_total{reason=\"request_timeout\"}"
        ),
        Some(2.0),
        "{metrics}"
    );
    drop(server);

    // reliability.admission.queue_timeout (P3 replaces scheduler.queue_timeout, C-1): behind
    // the one running slot, a waiting request gets 503. (`extra` lines indented by two spaces
    // continue the base config's `reliability` section.)
    let server = TinyServer::start_long(
        "  admission:\n    queue_timeout: 1s\nscheduler:\n  max_running_requests: 1\n",
    );
    let held = server.hold_stream();
    let started = Instant::now();
    let resp = server.post(
        "/v1/completions",
        &json!({"model": server.model, "prompt": "Hi", "max_tokens": 4}),
    );
    assert_eq!(resp.status, 503, "{}", resp.body);
    assert_eq!(resp.error_code(), "queue_timeout");
    assert!(started.elapsed() >= Duration::from_millis(900));
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            "turbine_admission_total{outcome=\"rejected\",reason=\"queue_timeout\"}"
        ),
        Some(1.0),
        "{metrics}"
    );
    drop(held);
}

// ------------------------------------------------------------------ Phase 2 request features

/// BOS then the bytes of "Hello": what the tiny tokenizer makes of the prompt "Hello".
const HELLO: [u32; 6] = [256, 72, 101, 108, 108, 111];

/// Tokens the Phase 1 single-request loop (host sampler, contiguous KV, cpu provider) generates
/// for `prompt` on the same tiny checkpoint: the reference for the server's batched engine.
fn reference_tokens(prompt: &[u32], sampling: SamplingParams, max_tokens: u32) -> Vec<u32> {
    let dir = TempDir::new("turbine-tiny-reference");
    let spec = write_tiny_llama(dir.path(), 7);
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    let index = SafetensorsIndex::open(&spec.dir).unwrap();
    let weights =
        WeightLoader::load(&index, &llama_slots(&spec.config), &mem, MAX_STAGING_BYTES).unwrap();
    let provider = cpu_reference_provider();
    let order = [provider.id()];
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&spec.config, 16, ExecutorOptions::default()),
        &KernelMetrics::register(&MetricsRegistry::new()),
        None,
    )
    .unwrap();
    let mut exec = executor::build_executor(
        &spec.config,
        weights,
        Arc::new(registry),
        Arc::clone(&mem),
        16,
        512,
        1,
        ExecutorOptions::default(),
    )
    .unwrap();
    let mut kv = SequenceKv::new(&mem, *exec.kv_layout(), 512).unwrap();
    let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
    let req = GenerationRequest {
        id: RequestId::new_v4(),
        endpoint: turbine_core::request::Endpoint::Completions,
        http_request_id: "reference".into(),
        prompt_tokens: prompt.to_vec(),
        n: 1,
        sampling,
        stop: StopConditions {
            eos_token_ids: TINY_EOS.iter().copied().collect(),
            max_tokens,
            ignore_eos: true,
            ..StopConditions::default()
        },
        priority: Priority::default(),
        echo: false,
        constraint: None,
        deadline_ms: u64::MAX,
        session: None,
        cache_salt: None,
    };
    let cancel = turbine_core::request::CancelFlag::default();
    generate(
        exec.as_mut(),
        &mut kv,
        tokenizer,
        &req,
        &cancel,
        GenerateOptions {
            max_seq_len: 512,
            metrics: None,
        },
    )
    .filter_map(|e| match e {
        GenerationEvent::Token { token_id, .. } => Some(token_id),
        _ => None,
    })
    .collect()
}

/// The token ids of a completions choice (`logprobs` with `return_tokens_as_token_ids`).
fn choice_token_ids(choice: &Value) -> Vec<u32> {
    choice["logprobs"]["tokens"]
        .as_array()
        .unwrap_or_else(|| panic!("no logprobs.tokens: {choice}"))
        .iter()
        .map(|t| {
            t.as_str()
                .and_then(|s| s.strip_prefix("token_id:"))
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| panic!("bad token {t}"))
        })
        .collect()
}

/// The chunks of a streamed response (its data ends in `[DONE]`).
fn stream_chunks(resp: &Response) -> Vec<Value> {
    assert_eq!(resp.status, 200, "{}", resp.body);
    let data = resp.sse_data();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

fn prompt_tokens_total(server: &TinyServer) -> f64 {
    sample(&server.metrics(), r#"turbine_tokens_total{kind="prompt"}"#).unwrap_or(0.0)
}

/// P2 S-10: `n`, penalties, `logit_bias`, `min_tokens`, `stop_token_ids`, `echo` and
/// `priority` are served.
#[test]
fn openai_phase2_fields() {
    let server = TinyServer::start("");
    let model = server.model.clone();

    // n: 3, not streaming: three choices from one prefill (the prompt is counted once).
    let before = prompt_tokens_total(&server);
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "Hello", "max_tokens": 6, "ignore_eos": true,
                "n": 3, "temperature": 1.0, "seed": 11}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3, "{body}");
    for (i, c) in choices.iter().enumerate() {
        assert_eq!(c["index"], i, "{body}");
        assert_eq!(c["finish_reason"], "length", "{body}");
    }
    assert_eq!(body["usage"]["prompt_tokens"], 6);
    assert_eq!(body["usage"]["completion_tokens"], 18);
    assert_eq!(prompt_tokens_total(&server) - before, 6.0);

    // n: 3, streaming: every index streams and finishes once.
    let before = prompt_tokens_total(&server);
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "Hello", "max_tokens": 5, "ignore_eos": true,
                "n": 3, "stream": true}),
    );
    let mut finishes = [0; 3];
    for chunk in stream_chunks(&resp) {
        for c in chunk["choices"].as_array().unwrap() {
            let i = c["index"].as_u64().unwrap() as usize;
            if c["finish_reason"] == "length" {
                finishes[i] += 1;
            }
        }
    }
    assert_eq!(finishes, [1, 1, 1]);
    assert_eq!(prompt_tokens_total(&server) - before, 6.0);

    // Penalties and logit_bias change greedy output exactly as the host sampler predicts.
    let greedy = |extra: Value| -> Vec<u32> {
        let mut body = json!({"model": model, "prompt": HELLO, "max_tokens": 16,
                              "ignore_eos": true, "temperature": 0, "logprobs": 0,
                              "return_tokens_as_token_ids": true});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let resp = server.post("/v1/completions", &body);
        assert_eq!(resp.status, 200, "{}", resp.body);
        choice_token_ids(&resp.json()["choices"][0])
    };
    let plain = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let baseline = greedy(json!({}));
    assert_eq!(baseline, reference_tokens(&HELLO, plain.clone(), 16));
    let cases = [
        (
            json!({"presence_penalty": 1.5, "frequency_penalty": 0.5}),
            SamplingParams {
                presence_penalty: 1.5,
                frequency_penalty: 0.5,
                ..plain.clone()
            },
        ),
        (
            json!({"repetition_penalty": 1.5}),
            SamplingParams {
                repetition_penalty: 1.5,
                ..plain.clone()
            },
        ),
        (
            json!({"logit_bias": {"120": 100}}),
            SamplingParams {
                logit_bias: vec![(120, 100.0)],
                ..plain.clone()
            },
        ),
    ];
    let mut changed = 0;
    for (fields, params) in cases {
        let got = greedy(fields.clone());
        assert_eq!(got, reference_tokens(&HELLO, params, 16), "{fields}");
        changed += usize::from(got != baseline);
    }
    assert_eq!(greedy(json!({"logit_bias": {"120": 100}})), vec![120; 16]);
    assert!(changed >= 2, "penalties and bias change the greedy output");

    // min_tokens holds EOS back: with EOS favoured, the output ends at token 1, or at 6.
    let eos_favoured = |min_tokens: u32| {
        let resp = server.post(
            "/v1/completions",
            &json!({"model": model, "prompt": "Hello", "max_tokens": 20, "temperature": 0,
                    "logit_bias": {"260": 100}, "min_tokens": min_tokens}),
        );
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        assert_eq!(body["choices"][0]["finish_reason"], "stop", "{body}");
        body["usage"]["completion_tokens"].as_u64().unwrap()
    };
    assert_eq!(eos_favoured(0), 1);
    assert_eq!(eos_favoured(5), 6);

    // stop_token_ids: the first occurrence of the id ends the choice with `stop`.
    let stop_id = baseline[3];
    let first = baseline.iter().position(|&t| t == stop_id).unwrap();
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": HELLO, "max_tokens": 16, "ignore_eos": true,
                "temperature": 0, "stop_token_ids": [stop_id], "logprobs": 0,
                "return_tokens_as_token_ids": true}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    assert_eq!(body["choices"][0]["finish_reason"], "stop", "{body}");
    assert_eq!(
        choice_token_ids(&body["choices"][0]),
        baseline[..=first].to_vec()
    );

    // echo: the prompt text prefixes the completion, streaming or not.
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "Hello", "max_tokens": 3, "ignore_eos": true,
                "temperature": 0, "echo": true}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let text = resp.json()["choices"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(text.starts_with("Hello") && text.len() > 5, "{text:?}");
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "Hello", "max_tokens": 3, "ignore_eos": true,
                "temperature": 0, "echo": true, "stream": true}),
    );
    let streamed: String = stream_chunks(&resp)
        .iter()
        .filter_map(|c| c["choices"][0]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(streamed, text);
    drop(server);

    // priority: of two queued requests the one with the lower value starts (and, with one
    // running request allowed, finishes) first.
    let server = TinyServer::start_long("scheduler:\n  max_running_requests: 1\n");
    let held = server.hold_stream();
    let queued = |priority: i32| {
        let addr = server.addr;
        let body = json!({"model": server.model, "prompt": "Hi", "max_tokens": 2,
                          "ignore_eos": true, "priority": priority});
        std::thread::spawn(move || {
            let resp = request(addr, "POST", "/v1/completions", Some(&body.to_string()));
            assert_eq!(resp.status, 200, "{}", resp.body);
            Instant::now()
        })
    };
    let low = queued(5);
    wait_for(Duration::from_secs(10), "first request queued", || {
        server.scheduler()["waiting"] == 1
    });
    let high = queued(-1);
    wait_for(Duration::from_secs(10), "second request queued", || {
        server.scheduler()["waiting"] == 2
    });
    drop(held);
    let (low, high) = (low.join().unwrap(), high.join().unwrap());
    assert!(high < low, "priority -1 must be served before priority 5");
}

/// The object schema of the structured-output tests: a boolean, an integer enum and a string
/// enum, nothing else.
fn small_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "level": {"type": "integer", "enum": [1, 2, 3]},
            "color": {"type": "string", "enum": ["red", "green", "blue"]}
        },
        "required": ["flag", "level", "color"],
        "additionalProperties": false
    })
}

/// The longest run of whitespace outside JSON string literals in `text`.
fn longest_whitespace_outside_strings(text: &str) -> usize {
    let (mut in_string, mut escaped) = (false, false);
    let (mut run, mut longest) = (0, 0);
    for c in text.chars() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
        } else if c.is_whitespace() {
            run += 1;
            longest = longest.max(run);
            continue;
        } else if c == '"' {
            in_string = true;
        }
        run = 0;
    }
    longest
}

/// P2 S-6, S-9: with a pool too small for the load, preemption by recompute does not change
/// any request's tokens (seeded sampling and constrained requests included).
/// P2 S-6 under P3 S-9: a request whose worst-case KV does not fit waits in the admission queue
/// (`kv_reservation`) instead of preempting a running one below SURVIVAL, and every output is
/// still exactly what the request produces alone.
#[test]
fn preempted_output_unchanged() {
    // 8 blocks of 128 tokens: the six unconstrained requests ("Hello" + 200 tokens, 2 blocks
    // each) and the two constrained ones (1 block each) do not all fit together.
    let server = TinyServer::launch(&Setup {
        kv_bytes: "256KiB",
        // Requests waiting for KV behind the running ones may wait longer than the default 30 s
        // in a loaded debug build.
        extra: "  admission:\n    queue_timeout: 10m\nscheduler:\n  max_running_requests: 8\n",
        ..Setup::default()
    });
    let bodies: Vec<Value> = (0..8)
        .map(|i| {
            let mut body = json!({"model": server.model, "prompt": "Hello", "max_tokens": 40,
                                  "temperature": 1.0, "seed": 100 + i, "logprobs": 0,
                                  "return_tokens_as_token_ids": true});
            if i < 2 {
                body["response_format"] = json!({"type": "json_schema",
                    "json_schema": {"name": "small", "schema": small_schema()}});
                // The random-weight model favours whitespace, which the grammar allows (up to
                // JSON_MAX_WHITESPACE per gap); banning the whitespace bytes keeps the object
                // within its 40 tokens.
                body["logit_bias"] = json!({"9": -100, "10": -100, "13": -100, "32": -100});
            } else {
                body["ignore_eos"] = json!(true);
                body["max_tokens"] = json!(200);
            }
            body
        })
        .collect();
    let outcome = |resp: Response| {
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        let c = &body["choices"][0];
        (
            choice_token_ids(c),
            c["text"].as_str().unwrap().to_string(),
            c["finish_reason"].as_str().unwrap().to_string(),
        )
    };
    let concurrent: Vec<_> = bodies
        .iter()
        .map(|b| {
            let (addr, b) = (server.addr, b.to_string());
            std::thread::spawn(move || request(addr, "POST", "/v1/completions", Some(&b)))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| outcome(h.join().unwrap()))
        .collect();
    let metrics = server.metrics();
    let preemptions = sample(
        &metrics,
        r#"turbine_preemptions_total{reason="kv_exhausted"}"#,
    );
    assert!(preemptions.is_none_or(|n| n == 0.0), "{metrics}");
    let waited = sample(
        &metrics,
        r#"turbine_admission_decisions_total{decision="queue",reason="kv_reservation"}"#,
    );
    assert!(waited.is_some_and(|n| n > 0.0), "{metrics}");

    for (i, body) in bodies.iter().enumerate() {
        let alone = outcome(server.post("/v1/completions", body));
        assert_eq!(concurrent[i], alone, "request {i}");
    }
    let validator = jsonschema::validator_for(&small_schema()).unwrap();
    for (tokens, text, finish) in &concurrent[..2] {
        assert_eq!(finish, "stop", "{text}");
        let value: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert!(validator.is_valid(&value), "{text}");
        assert!(!tokens.is_empty());
    }
    assert_eq!(server.blocks_used(), 0);
}

/// P2 S-17: `json_schema` and `json_object` outputs always parse, validate and keep whitespace
/// outside strings within `JSON_MAX_WHITESPACE` per gap; a
/// schema with an unsupported keyword or over `structured_output.max_schema_bytes` is refused.
#[test]
fn response_format_json_schema() {
    let server = TinyServer::start("structured_output:\n  max_schema_bytes: 1KiB\n");
    let chat = |response_format: Value, seed: u64, extra: Value| {
        let mut body = json!({"model": server.model,
                              "messages": [{"role": "user", "content": "Describe a thing."}],
                              "max_tokens": 300, "temperature": 1.0, "seed": seed,
                              "response_format": response_format});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        server.post("/v1/chat/completions", &body)
    };
    let schema_format = json!({"type": "json_schema", "json_schema": {"name": "thing", "schema": small_schema(),
                                                       "strict": true}});
    let validator = jsonschema::validator_for(&small_schema()).unwrap();
    let mut seen = std::collections::HashSet::new();
    for seed in 0..20 {
        let resp = chat(schema_format.clone(), seed, json!({}));
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "stop", "{body}");
        let text = choice["message"]["content"].as_str().unwrap();
        let value: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert!(validator.is_valid(&value), "seed {seed}: {text}");
        assert!(
            longest_whitespace_outside_strings(text) <= turbine_model::JSON_MAX_WHITESPACE,
            "whitespace over the bound: {text:?}"
        );
        seen.insert(text.to_string());
    }
    assert!(seen.len() > 1, "sampling varies the objects: {seen:?}");

    // json_object: any object. The random-weight model would fill it with unbounded numbers
    // and strings, so `}` is favoured to close it; the grammar still decides what is allowed.
    for seed in 0..5 {
        let resp = chat(
            json!({"type": "json_object"}),
            seed,
            json!({"logit_bias": {"125": 50}}),
        );
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "stop", "{body}");
        let text = choice["message"]["content"].as_str().unwrap();
        let value: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert!(value.is_object(), "{text}");
        assert!(
            longest_whitespace_outside_strings(text) <= turbine_model::JSON_MAX_WHITESPACE,
            "whitespace over the bound: {text:?}"
        );
    }

    // Refused before queueing.
    let unsupported = json!({"type": "json_schema", "json_schema": {"name": "u",
        "schema": {"type": "array", "uniqueItems": true}}});
    let big = json!({"type": "json_schema", "json_schema": {"name": "b",
        "schema": {"type": "string", "description": "x".repeat(2048)}}});
    for (format, needle) in [(unsupported, "uniqueItems"), (big, "max_schema_bytes")] {
        let resp = chat(format, 0, json!({}));
        assert_eq!(resp.status, 400, "{}", resp.body);
        assert_eq!(resp.error_code(), "invalid_json_schema", "{}", resp.body);
        assert!(resp.body.contains(needle), "{}", resp.body);
    }
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_admission_total{outcome="rejected",reason="invalid_json_schema"}"#
        ),
        Some(2.0),
        "{metrics}"
    );
}

fn weather_tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "enum": ["Paris", "Oslo"]},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["location"],
                "additionalProperties": false
            }
        }},
        {"type": "function", "function": {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string", "enum": ["UTC", "CET"]}},
                "required": ["zone"],
                "additionalProperties": false
            }
        }}
    ])
}

/// `call_` followed by 24 ASCII alphanumerics.
fn is_call_id(id: &str) -> bool {
    id.strip_prefix("call_")
        .is_some_and(|r| r.len() == 24 && r.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// Checks one tool call object (`id`, `type`, `function.name`, `function.arguments`) against
/// the tools' schemas; returns the function name.
fn check_call(call: &Value, tools: &Value) -> String {
    assert!(is_call_id(call["id"].as_str().unwrap()), "{call}");
    assert_eq!(call["type"], "function", "{call}");
    let name = call["function"]["name"].as_str().unwrap();
    let tool = tools
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["function"]["name"] == name)
        .unwrap_or_else(|| panic!("{name} is not a listed tool"));
    let arguments = call["function"]["arguments"].as_str().unwrap();
    let args: Value = serde_json::from_str(arguments).unwrap();
    let validator = jsonschema::validator_for(&tool["function"]["parameters"]).unwrap();
    assert!(validator.is_valid(&args), "{name}: {arguments}");
    name.to_string()
}

/// P2 S-18: named and `required` tool choices return exactly one valid call; `none` returns
/// content; an unknown function and a template without tools are refused.
#[test]
fn tool_choice_modes() {
    // The Llama-3.2 template renders the tools into the prompt: about 1600 byte tokens.
    let server = TinyServer::launch(&Setup {
        model_extra: "  tool_call_parser: llama3_json\n",
        max_positions: Some(LONG_POSITIONS),
        ..Setup::default()
    });
    let tools = weather_tools();
    let chat = |extra: Value| {
        let mut body = json!({"model": server.model,
                              "messages": [{"role": "user", "content": "Weather in Paris?"}],
                              "tools": tools, "max_tokens": 120, "temperature": 1.0,
                              "seed": 3, "chat_template_kwargs": {"date_string": "26 Jul 2024"}});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        server.post("/v1/chat/completions", &body)
    };
    let named = json!({"type": "function", "function": {"name": "get_weather"}});

    // Named, not streaming.
    let resp = chat(json!({"tool_choice": named}));
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
    assert!(choice["message"]["content"].is_null(), "{body}");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(check_call(&calls[0], &tools), "get_weather");

    // Named, streaming: one delta.tool_calls entry with the complete arguments.
    let resp = chat(json!({"tool_choice": named, "stream": true}));
    let mut entries = Vec::new();
    let mut finish = None;
    for chunk in stream_chunks(&resp) {
        let c = &chunk["choices"][0];
        if let Some(calls) = c["delta"]["tool_calls"].as_array() {
            entries.extend(calls.iter().cloned());
        }
        if let Some(reason) = c["finish_reason"].as_str() {
            finish = Some(reason.to_string());
        }
    }
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0]["index"], 0);
    assert_eq!(check_call(&entries[0], &tools), "get_weather");

    // required without parallel calls: exactly one call to a listed tool.
    for seed in 0..3 {
        let resp = chat(
            json!({"tool_choice": "required", "parallel_tool_calls": false,
                               "seed": seed}),
        );
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls", "{body}");
        let calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1, "{body}");
        check_call(&calls[0], &tools);
    }

    // auto: free text that does not open with `{`, or exactly one schema-valid call. The
    // random-weight model's text is favoured to open with `{` (token 123); the grammar only
    // lets that be a call.
    for seed in 0..3 {
        let resp = chat(json!({"tool_choice": "auto", "parallel_tool_calls": false,
                               "seed": seed, "max_tokens": 40}));
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        let choice = &body["choices"][0];
        match choice["message"]["tool_calls"].as_array() {
            Some(calls) => {
                assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
                assert_eq!(calls.len(), 1, "{body}");
                check_call(&calls[0], &tools);
            }
            None => {
                let text = choice["message"]["content"].as_str().unwrap();
                assert!(!text.trim_start().starts_with('{'), "{body}");
            }
        }
    }
    for seed in 0..3 {
        let resp = chat(json!({"tool_choice": "auto", "parallel_tool_calls": false,
                               "seed": seed, "logit_bias": {"123": 100}}));
        assert_eq!(resp.status, 200, "{}", resp.body);
        let body = resp.json();
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
        let calls = choice["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1, "{body}");
        check_call(&calls[0], &tools);
    }

    // none: content only.
    let resp = chat(json!({"tool_choice": "none", "max_tokens": 8, "ignore_eos": true}));
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    let message = &body["choices"][0]["message"];
    assert!(message["content"].is_string(), "{body}");
    assert!(
        message.get("tool_calls").is_none_or(Value::is_null),
        "{body}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "length");

    // A named function that is not listed.
    let resp = chat(json!({"tool_choice": {"type": "function", "function": {"name": "nope"}}}));
    assert_eq!(resp.status, 400, "{}", resp.body);
    assert_eq!(resp.error_code(), "unknown_tool");
    drop(server);

    // A checkpoint whose template does not render tools has no tool-call parser.
    let plain = TinyServer::launch(&Setup {
        template_with_tools: false,
        ..Setup::default()
    });
    let resp = plain.post(
        "/v1/chat/completions",
        &json!({"model": plain.model, "messages": [{"role": "user", "content": "x"}],
                "tools": tools}),
    );
    assert_eq!(resp.status, 400, "{}", resp.body);
    assert_eq!(resp.error_code(), "tools_not_supported");
}

/// P2 S-14: a run that queues, rejects, preempts, cancels and serves constrained requests
/// leaves every Phase 2 metric family with the expected counts, and a JSON log record with a
/// `reason` for each reject, preemption and cancellation.
#[test]
fn phase2_metrics_and_reasons() {
    // 20 blocks of 128 tokens; the tool request's rendered prompt takes about 100 tokens. The
    // context is capped at 2,048 tokens (16 blocks) so the pool holds one full-context sequence
    // (P3 S-2).
    let server = TinyServer::launch(&Setup {
        kv_bytes: "640KiB",
        model_extra: "  max_seq_len: 2048\n",
        max_positions: Some(LONG_POSITIONS),
        extra: "  admission:\n    queue_timeout: 10m\nscheduler:\n  max_running_requests: 6\nlogging:\n  \
                format: json\n",
        capture_logs: true,
        ..Setup::default()
    });
    let model = server.model.clone();

    // Six concurrent requests of 4 blocks each (worst case): five hold the 20 blocks, the sixth
    // waits for KV in the admission queue (P3: no preemption below SURVIVAL).
    let handles: Vec<_> = (0..6)
        .map(|i| {
            let addr = server.addr;
            let body = json!({"model": model, "prompt": "Hello", "max_tokens": 450,
                              "ignore_eos": true, "temperature": 1.0, "seed": i});
            std::thread::spawn(move || {
                request(addr, "POST", "/v1/completions", Some(&body.to_string()))
            })
        })
        .collect();
    for h in handles {
        let resp = h.join().unwrap();
        assert_eq!(resp.status, 200, "{}", resp.body);
    }

    // Rejected: a KV need beyond the pool (two choices of 1,400 tokens: 22 blocks), and a bad
    // schema.
    let resp = server.post(
        "/v1/completions",
        &json!({"model": model, "prompt": "Hello", "max_tokens": 1400, "n": 2}),
    );
    assert_eq!(resp.status, 400, "{}", resp.body);
    assert_eq!(resp.error_code(), "context_exceeds_kv_capacity");
    let resp = server.post(
        "/v1/chat/completions",
        &json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "u",
                    "schema": {"type": "array", "uniqueItems": true}}}}),
    );
    assert_eq!(resp.error_code(), "invalid_json_schema", "{}", resp.body);

    // Cancelled: a client that goes away after the first chunk.
    let stream = OpenStream::open(
        server.addr,
        &json!({"model": model, "prompt": "Hello", "max_tokens": 250, "ignore_eos": true,
                "stream": true, "logprobs": 20}),
        true,
    );
    drop(stream);
    wait_for(Duration::from_secs(5), "cancellation counted", || {
        sample(
            &server.metrics(),
            r#"turbine_requests_cancelled_total{reason="client_disconnect"}"#,
        ) == Some(1.0)
    });

    // Constrained: a json_schema answer and a named tool call.
    let resp = server.post(
        "/v1/chat/completions",
        &json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 100, "response_format": {"type": "json_schema",
                "json_schema": {"name": "small", "schema": small_schema()}}}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    let resp = server.post(
        "/v1/chat/completions",
        &json!({"model": model, "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 100, "tools": weather_tools(),
                "tool_choice": {"type": "function", "function": {"name": "get_time"}}}),
    );
    assert_eq!(resp.status, 200, "{}", resp.body);
    assert_eq!(resp.json()["choices"][0]["finish_reason"], "tool_calls");
    wait_for(Duration::from_secs(1), "blocks released", || {
        server.blocks_used() == 0
    });

    let metrics = server.metrics();
    let exact = [
        (
            r#"turbine_admission_total{outcome="queued",reason="ok"}"#,
            9.0,
        ),
        (
            r#"turbine_admission_total{outcome="rejected",reason="context_exceeds_kv_capacity"}"#,
            1.0,
        ),
        (
            r#"turbine_admission_total{outcome="rejected",reason="invalid_json_schema"}"#,
            1.0,
        ),
        ("turbine_queue_wait_seconds_count", 9.0),
        (
            r#"turbine_requests_cancelled_total{reason="client_disconnect"}"#,
            1.0,
        ),
        (r#"turbine_requests_active{state="prefilling"}"#, 0.0),
        (r#"turbine_requests_active{state="decoding"}"#, 0.0),
        (r#"turbine_requests_active{state="paused"}"#, 0.0),
        ("turbine_requests_queued", 0.0),
        (
            r#"turbine_grammar_compile_seconds_count{kind="json_schema"}"#,
            2.0,
        ),
        (
            r#"turbine_grammar_compile_seconds_count{kind="tool_call"}"#,
            1.0,
        ),
        (
            r#"turbine_tool_calls_total{parser="llama3_json",outcome="parsed"}"#,
            1.0,
        ),
    ];
    for (series, value) in exact {
        assert_eq!(sample(&metrics, series), Some(value), "{series}\n{metrics}");
    }
    // No block is referenced any more; the used ones are cached for prefix reuse (Phase 4).
    let used = sample(&metrics, r#"turbine_kv_blocks{tier="l0",state="used"}"#).unwrap();
    let free = sample(&metrics, r#"turbine_kv_blocks{tier="l0",state="free"}"#).unwrap();
    assert_eq!(used + free, 20.0, "{metrics}");
    assert_eq!(server.kv_tier()["blocks_used"].as_f64(), Some(used));
    let positive = [
        r#"turbine_admission_decisions_total{decision="queue",reason="kv_reservation"}"#,
        "turbine_iteration_seconds_count",
        r#"turbine_iteration_tokens_count{phase="prefill"}"#,
        r#"turbine_iteration_tokens_count{phase="decode"}"#,
        "turbine_batch_requests_count",
        "turbine_token_mask_seconds_count",
    ];
    for series in positive {
        assert!(
            sample(&metrics, series).is_some_and(|v| v > 0.0),
            "{series}\n{metrics}"
        );
    }
    assert!(
        sample(&metrics, "turbine_stream_paused_total").is_some(),
        "{metrics}"
    );

    // Every automatic decision is logged with its reason.
    let records = server.log_records();
    let reasons: Vec<&str> = records
        .iter()
        .filter_map(|r| r["fields"]["reason"].as_str())
        .collect();
    for reason in [
        "context_exceeds_kv_capacity",
        "invalid_json_schema",
        "kv_reservation",
        "client_disconnect",
    ] {
        assert!(
            reasons.contains(&reason),
            "no log record with reason {reason}: {reasons:?}"
        );
    }
}

/// Reads one chunked response body to its terminating chunk as raw bytes, checking the framing
/// strictly (hex size line, CRLF after every chunk, the `0\r\n\r\n` terminator).
fn read_chunked_bytes(reader: &mut BufReader<TcpStream>) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    loop {
        let mut size_line = String::new();
        let n = reader
            .read_line(&mut size_line)
            .map_err(|e| format!("reading a chunk size after {} bytes: {e}", body.len()))?;
        if n == 0 {
            return Err(format!(
                "connection closed before the terminating chunk after {} bytes: {:?}",
                body.len(),
                String::from_utf8_lossy(&body[body.len().saturating_sub(300)..])
            ));
        }
        let size = usize::from_str_radix(size_line.trim_end_matches("\r\n"), 16)
            .map_err(|_| format!("bad chunk size line {size_line:?}"))?;
        let mut data = vec![0u8; size + 2];
        reader.read_exact(&mut data).map_err(|e| {
            format!(
                "reading a {size}-byte chunk after {} bytes: {e}",
                body.len()
            )
        })?;
        if &data[size..] != b"\r\n" {
            return Err(format!("chunk of {size} bytes not followed by CRLF"));
        }
        if size == 0 {
            return Ok(body);
        }
        body.extend_from_slice(&data[..size]);
    }
}

/// One keep-alive client of [`concurrent_streams_are_well_framed`]: `requests` sampled chat
/// streams in a row on one connection, each checked end to end.
fn framed_client(
    addr: SocketAddr,
    model: &str,
    client: usize,
    requests: usize,
    tokens: u64,
) -> Result<(), String> {
    let mut conn = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    conn.set_read_timeout(Some(Duration::from_secs(120)))
        .unwrap();
    let mut reader = BufReader::new(conn.try_clone().unwrap());
    for i in 0..requests {
        let tag = format!("client {client} request {i}");
        let body = json!({
            "model": model,
            "messages": [{"role": "user", "content": format!("{tag} ü € 𝄞")}],
            "max_tokens": tokens, "ignore_eos": true, "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": 1.0, "seed": client * 100 + i,
        })
        .to_string();
        write!(
            conn,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .map_err(|e| format!("{tag}: write: {e}"))?;
        let (status, head) = read_head(&mut reader);
        if status != 200 || !head.contains("transfer-encoding: chunked") {
            return Err(format!("{tag}: {status} {head}"));
        }
        let bytes = read_chunked_bytes(&mut reader).map_err(|e| format!("{tag}: {e}"))?;
        let text =
            String::from_utf8(bytes).map_err(|e| format!("{tag}: SSE body is not UTF-8: {e}"))?;
        let events: Vec<&str> = text.split("\n\n").filter(|e| !e.is_empty()).collect();
        let Some((last, rest)) = events.split_last() else {
            return Err(format!("{tag}: empty stream"));
        };
        if *last != "data: [DONE]" {
            return Err(format!("{tag}: last event {last:?}"));
        }
        let mut completion = None;
        for event in rest {
            let data = event
                .strip_prefix("data: ")
                .ok_or_else(|| format!("{tag}: bad event {event:?}"))?;
            let v: Value =
                serde_json::from_str(data).map_err(|e| format!("{tag}: bad JSON {data:?}: {e}"))?;
            if v.get("error").is_some() {
                return Err(format!("{tag}: stream error {v}"));
            }
            if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
                completion = Some(n);
            }
        }
        if completion != Some(tokens) {
            return Err(format!("{tag}: usage completion_tokens {completion:?}"));
        }
    }
    Ok(())
}

/// Sixteen concurrent keep-alive clients, as `turbine-bench --concurrency 16` drives them, each
/// streaming several sampled chat completions of byte-level tokens (random bytes: multi-byte
/// characters split across tokens, invalid sequences): every stream is well-formed chunked SSE
/// whose events are UTF-8 JSON, ends with the usage chunk, `[DONE]` and the terminating chunk,
/// and leaves its connection usable for the next request.
#[test]
fn concurrent_streams_are_well_framed() {
    let server = TinyServer::start_long("scheduler:\n  max_running_requests: 16\n");
    let workers: Vec<_> = (0..16)
        .map(|client| {
            let (addr, model) = (server.addr, server.model.clone());
            std::thread::spawn(move || framed_client(addr, &model, client, 4, 256))
        })
        .collect();
    let failures: Vec<String> = workers
        .into_iter()
        .filter_map(|w| w.join().unwrap().err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of 16 clients failed: {failures:#?}",
        failures.len()
    );
}

/// The eight engine iteration stages (P2c S-1), in `IterationStages` order.
const STAGES: [&str; 8] = [
    "schedule",
    "prepare",
    "launch",
    "device_wait",
    "sample",
    "detokenize",
    "emit",
    "complete",
];

/// P2c S-1: every iteration is timed in eight stages; `/metrics` has a histogram per stage and
/// the scheduler document's `last_iteration.stages_ms` partitions its `duration_ms`.
#[test]
fn iteration_stage_breakdown() {
    let server = TinyServer::start("");
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let (addr, model) = (server.addr, server.model.clone());
            std::thread::spawn(move || {
                let body = json!({"model": model, "prompt": format!("Hello {i}"),
                                  "max_tokens": 12, "ignore_eos": true});
                request(addr, "POST", "/v1/completions", Some(&body.to_string()))
            })
        })
        .collect();
    for h in handles {
        let resp = h.join().unwrap();
        assert_eq!(resp.status, 200, "{}", resp.body);
    }

    let metrics = server.metrics();
    for stage in STAGES {
        let series = format!(r#"turbine_engine_iteration_seconds_count{{stage="{stage}"}}"#);
        assert!(
            sample(&metrics, &series).is_some_and(|n| n > 0.0),
            "{series}\n{metrics}"
        );
    }
    // The executors measure their share of the forward pass.
    for stage in ["launch", "device_wait"] {
        let series = format!(r#"turbine_engine_iteration_seconds_sum{{stage="{stage}"}}"#);
        assert!(
            sample(&metrics, &series).is_some_and(|s| s > 0.0),
            "{series}\n{metrics}"
        );
    }

    let doc = server.scheduler();
    let last = &doc["last_iteration"];
    let stages = last["stages_ms"].as_object().expect("stages_ms object");
    let mut keys: Vec<&str> = stages.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut want = STAGES.to_vec();
    want.sort_unstable();
    assert_eq!(keys, want, "{doc}");
    let mut sum = 0.0;
    for (stage, ms) in stages {
        let ms = ms.as_f64().expect("a number");
        assert!(ms >= 0.0, "{stage}: {ms} in {doc}");
        sum += ms;
    }
    let duration = last["duration_ms"].as_f64().unwrap();
    assert!(duration > 0.0, "{doc}");
    assert!(
        (sum - duration).abs() <= (0.05 * duration).max(0.5),
        "stages sum to {sum} ms, the iteration took {duration} ms: {doc}"
    );
}

/// P2c overlap scheduling end to end: `execution.overlap_scheduling` true and false (the default)
/// give every request of a concurrent mix the same completion — greedy with and without
/// logprobs, a stop string, EOS honoured, a seeded draw, `n` = 2, a JSON schema — and the server
/// logs whether overlap scheduling is on (`overlap_scheduling`, `enabled`). Breaks if the key is
/// not wired to the engine or overlap scheduling changes an output.
#[test]
fn overlap_scheduling_matches_serial() {
    let bodies = [
        json!({"prompt": "Hello", "max_tokens": 12, "temperature": 0.0, "ignore_eos": true}),
        json!({"prompt": "Hello", "max_tokens": 12, "temperature": 0.0, "logprobs": 3,
               "ignore_eos": true}),
        json!({"prompt": "Once upon a time", "max_tokens": 16, "temperature": 0.0,
               "stop": ["e", "o"]}),
        json!({"prompt": "Hi there", "max_tokens": 20, "temperature": 0.0}),
        json!({"prompt": "Yes", "max_tokens": 10, "temperature": 0.9, "seed": 3,
               "ignore_eos": true}),
        json!({"prompt": "Bye", "max_tokens": 6, "temperature": 0.0, "n": 2,
               "ignore_eos": true}),
        json!({"prompt": "Data:", "max_tokens": 40, "temperature": 0.0,
               "response_format": {"type": "json_schema",
                   "json_schema": {"name": "small", "schema": small_schema()}}}),
        json!({"prompt": "A", "max_tokens": 1, "temperature": 0.0, "ignore_eos": true}),
    ];
    let run = |overlap: bool| -> Vec<Value> {
        let execution_extra = format!("  overlap_scheduling: {overlap}\n");
        let server = TinyServer::launch(&Setup {
            execution_extra: &execution_extra,
            extra: "logging:\n  format: json\n",
            capture_logs: true,
            ..Setup::default()
        });
        let outputs: Vec<Value> = std::thread::scope(|s| {
            let handles: Vec<_> = bodies
                .iter()
                .map(|b| {
                    let mut body = b.clone();
                    body["model"] = json!(server.model);
                    let server = &server;
                    s.spawn(move || {
                        let resp = server.post("/v1/completions", &body);
                        assert_eq!(resp.status, 200, "{body}: {}", resp.body);
                        let doc = resp.json();
                        json!({"choices": doc["choices"], "usage": doc["usage"]})
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let logged = || {
            server
                .log_records()
                .into_iter()
                .find(|r| r["fields"]["event"] == "overlap_scheduling")
        };
        wait_for(
            Duration::from_secs(10),
            "the overlap_scheduling log",
            || logged().is_some(),
        );
        let logged = logged().unwrap();
        assert_eq!(logged["fields"]["enabled"], json!(overlap), "{logged}");
        outputs
    };
    let overlapped = run(true);
    let serial = run(false);
    for (i, (o, s)) in overlapped.iter().zip(&serial).enumerate() {
        assert_eq!(o, s, "request {i}: {}", bodies[i]);
    }
    assert_eq!(serial[5]["choices"].as_array().map(Vec::len), Some(2));
}

/// One request of [`device_sampling_matches_host`]: its body and how many of its yielding rows
/// qualify for the device reduction.
struct DeviceCase {
    body: Value,
    eligible_rows: u64,
}

/// P2c S-4/S-14: seeded requests give identical tokens and logprobs whether their rows are
/// reduced on the device (`logits_reduce` of the CPU provider) or copied whole to the host
/// sampler, and `turbine_logits_rows_total{path="device_reduced"}` counts exactly the eligible
/// rows: none of a request with `logit_bias` or a JSON schema, those of a `min_tokens` 4
/// request from its fifth token on, and every row of greedy requests and of sampled ones with
/// `top_p` 1 or below (the checkpoint's 0.9 default, 0.3, and 0.5 under `top_k` 40).
#[test]
fn device_sampling_matches_host() {
    const TOKENS: u64 = 8;
    let case = |extra: Value, eligible_rows: u64| {
        let mut body = json!({"prompt": "Hello", "max_tokens": TOKENS, "ignore_eos": true,
                              "return_tokens_as_token_ids": true});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        DeviceCase {
            body,
            eligible_rows,
        }
    };
    let cases = [
        case(json!({"temperature": 0.0, "logprobs": 0}), TOKENS),
        case(json!({"temperature": 0.0, "prompt": "Hi there"}), TOKENS),
        case(json!({"temperature": 0.0, "min_tokens": 4}), TOKENS - 4),
        case(json!({"temperature": 0.0, "logprobs": 5}), TOKENS),
        case(
            json!({"temperature": 0.0, "logprobs": 5, "prompt": "Bye"}),
            TOKENS,
        ),
        case(
            json!({"temperature": 0.0, "logprobs": 5, "prompt": "Yes"}),
            TOKENS,
        ),
        case(
            json!({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 7, "logprobs": 2}),
            TOKENS,
        ),
        case(
            json!({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 8}),
            TOKENS,
        ),
        case(
            json!({"temperature": 1.0, "top_p": 1.0, "top_k": 40, "seed": 9, "logprobs": 3}),
            TOKENS,
        ),
        case(
            json!({"temperature": 1.0, "top_p": 1.0, "top_k": 40, "seed": 10}),
            TOKENS,
        ),
        case(
            json!({"temperature": 1.0, "seed": 13, "logprobs": 2}),
            TOKENS,
        ),
        case(json!({"temperature": 0.6, "seed": 14}), TOKENS),
        case(
            json!({"temperature": 1.3, "top_p": 0.3, "top_k": -1, "seed": 15, "logprobs": 1}),
            TOKENS,
        ),
        case(
            json!({"temperature": 0.8, "top_p": 0.5, "top_k": 40, "seed": 16, "logprobs": 3}),
            TOKENS,
        ),
        case(
            json!({"temperature": 1.0, "top_p": 1.0, "seed": 11, "logit_bias": {"120": 2}, "logprobs": 1}),
            0,
        ),
        case(
            json!({"temperature": 1.0, "top_p": 1.0, "seed": 12, "max_tokens": 40, "ignore_eos": false,
                   "response_format": {"type": "json_schema",
                       "json_schema": {"name": "small", "schema": small_schema()}},
                   "logit_bias": {"9": -100, "10": -100, "13": -100, "32": -100}}),
            0,
        ),
    ];
    let run = |device: bool| -> (Vec<Value>, String) {
        let execution_extra = format!("  device_sampling: {device}\n");
        let server = TinyServer::launch(&Setup {
            execution_extra: &execution_extra,
            ..Setup::default()
        });
        let outputs = cases
            .iter()
            .map(|c| {
                let mut body = c.body.clone();
                body["model"] = json!(server.model);
                let resp = server.post("/v1/completions", &body);
                assert_eq!(resp.status, 200, "{body}: {}", resp.body);
                let choice = resp.json()["choices"][0].clone();
                json!({"text": choice["text"], "logprobs": choice["logprobs"],
                       "finish_reason": choice["finish_reason"]})
            })
            .collect();
        (outputs, server.metrics())
    };
    let (device, device_metrics) = run(true);
    let (host, host_metrics) = run(false);
    for (i, (d, h)) in device.iter().zip(&host).enumerate() {
        assert_eq!(d, h, "request {i}: {}", cases[i].body);
    }
    let reduced = |m: &str| sample(m, r#"turbine_logits_rows_total{path="device_reduced"}"#);
    let expected: u64 = cases.iter().map(|c| c.eligible_rows).sum();
    assert_eq!(
        reduced(&device_metrics),
        Some(expected as f64),
        "{device_metrics}"
    );
    assert_eq!(reduced(&host_metrics), None, "{host_metrics}");
    assert!(
        sample(
            &host_metrics,
            r#"turbine_logits_rows_total{path="full_row"}"#
        )
        .is_some_and(|n| n > 0.0),
        "{host_metrics}"
    );
}

/// Phase 2m S-11: a tiny Qwen3 checkpoint starts through the family registry on the `cpu`
/// backend (its support row is `experimental`) and answers a 4-token greedy completion equal
/// to the naive decoder's argmax chain (`turbine_model::testing::naive`).
#[test]
fn qwen3_tiny_generates_on_cpu() {
    const TOKENS: usize = 4;
    let dir = TempDir::new("turbine-tiny-qwen3");
    let model_dir = dir.path().join("tiny-qwen3");
    let spec = write_tiny_family(&model_dir, "qwen3", 3);

    // The naive argmax chain from the prompt.
    let mut tokens = HELLO.to_vec();
    let mut expected = Vec::new();
    for _ in 0..TOKENS {
        let rows = naive::forward(&spec.config, &spec.dir, &tokens);
        let last = rows.last().expect("a row");
        let mut best = 0;
        for (i, &v) in last.iter().enumerate() {
            if v > last[best] {
                best = i;
            }
        }
        expected.push(best as u32);
        tokens.push(best as u32);
    }

    let config = dir.path().join("config.yaml");
    for _ in 0..LAUNCH_ATTEMPTS {
        let addr = free_addr();
        std::fs::write(&config, config_yaml(&model_dir, addr, "")).unwrap();
        let mut child = spawn(&config);
        let logs = drain_stderr(&mut child);
        match wait_until_ready(&mut child, addr, &logs) {
            Ready::PortTaken => continue,
            Ready::Serving => {}
        }
        let status = request(addr, "GET", "/turbine/v1/status", None).json();
        assert_eq!(status["modules"]["family"], "qwen3", "{status}");
        assert_eq!(status["support"]["architecture"], "Qwen3ForCausalLM");
        assert_eq!(status["support"]["status"], "experimental", "{status}");
        let body = json!({"model": "tiny-qwen3", "prompt": HELLO, "max_tokens": TOKENS,
                          "ignore_eos": true, "temperature": 0, "logprobs": 0,
                          "return_tokens_as_token_ids": true});
        let resp = request(addr, "POST", "/v1/completions", Some(&body.to_string()));
        child.kill().ok();
        child.wait().ok();
        assert_eq!(resp.status, 200, "{}", resp.body);
        assert_eq!(choice_token_ids(&resp.json()["choices"][0]), expected);
        return;
    }
    panic!("turbine-server lost its port {LAUNCH_ATTEMPTS} times in a row");
}

/// P5 S-7: two data-parallel replicas of the tiny model share the cpu "device"
/// (`allow_device_sharing`, each with half the device budget); 32 concurrent requests all
/// succeed, the router sends work to both replicas (`turbine_dp_routed_total` non-zero for
/// replica 0 and 1), the scheduler document has one entry per replica, the KV document one per
/// replica, the free-block gauge adds up over both pools, and the pressure document lists both
/// replicas' groups.
#[test]
fn dp2_routes_to_both_replicas() {
    let server = TinyServer::start(
        "parallel:\n  data_parallel_size: 2\n  allow_device_sharing: true\n\
         scheduler:\n  max_running_requests: 4\n",
    );
    let workers: Vec<_> = (0..32)
        .map(|i| {
            let (addr, model) = (server.addr, server.model.clone());
            std::thread::spawn(move || {
                let body = json!({"model": model, "prompt": format!("Hello {i}"),
                                  "max_tokens": 24, "ignore_eos": true});
                request(addr, "POST", "/v1/completions", Some(&body.to_string()))
            })
        })
        .collect();
    for (i, w) in workers.into_iter().enumerate() {
        let resp = w.join().unwrap();
        assert_eq!(resp.status, 200, "request {i}: {}", resp.body);
        assert_eq!(resp.json()["usage"]["completion_tokens"], 24, "request {i}");
    }

    let metrics = server.metrics();
    let routed = |replica: u32| -> f64 {
        metrics
            .lines()
            .filter(|l| {
                l.starts_with("turbine_dp_routed_total{")
                    && l.contains(&format!("replica=\"{replica}\""))
            })
            .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
            .sum()
    };
    assert!(routed(0) > 0.0, "{metrics}");
    assert!(routed(1) > 0.0, "{metrics}");
    assert_eq!(routed(0) + routed(1), 32.0, "{metrics}");

    let resp = server.get("/turbine/v1/scheduler");
    assert_eq!(resp.status, 200, "{}", resp.body);
    let doc = resp.json();
    let mut keys: Vec<&str> = doc
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["0", "1"], "{doc}");
    for r in ["0", "1"] {
        assert!(doc[r]["iterations_total"].as_u64().unwrap() > 0, "{doc}");
        assert_eq!(doc[r]["waiting"], 0, "{doc}");
    }
    let kv = server.get("/turbine/v1/kv").json();
    let total = |r: &str| kv[r]["tiers"][0]["blocks_total"].as_u64().unwrap();
    let (t0, t1) = (total("0"), total("1"));
    assert!(t0 > 0 && t1 > 0, "{kv}");
    // Each engine publishes after the step that finished its last request.
    wait_for(Duration::from_secs(30), "both pools free", || {
        sample(
            &server.metrics(),
            r#"turbine_kv_blocks{tier="l0",state="free"}"#,
        ) == Some((t0 + t1) as f64)
    });

    let pressure = server.get("/turbine/v1/pressure").json();
    let groups: Vec<u64> = pressure["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["replica"].as_u64().unwrap())
        .collect();
    assert_eq!(groups, [0, 1], "{pressure}");
    assert_eq!(
        pressure["replicas"].as_array().unwrap().len(),
        2,
        "{pressure}"
    );

    let parallel = &server.get("/turbine/v1/status").json()["parallel"];
    assert_eq!(parallel["dp"], 2, "{parallel}");
    assert_eq!(parallel["groups"][1]["replica"], 1, "{parallel}");
    assert_eq!(
        parallel["plan_reasons"],
        json!(["execution_device", "device_sharing_enabled"])
    );
}

/// Tensor parallelism of 2 on the cpu backend: two rank threads over the host collective
/// (the host test path, P5 S-6).
const TP2: &str = "parallel:\n  tensor_parallel_size: 2\n  collective_backend: host\n";

/// P5 S-6: the tiny Llama served at tp 2 (two cpu ranks over the host collective, each its own
/// host memory and KV pool) completes greedy prompts with exactly tp 1's tokens, alone and 8
/// at a time; `/ready` is 200 and `/turbine/v1/status` shows the group (2 ranks on host devices
/// 0 and 1, backend `host`). Breaks if the server shards, batches or steps the ranks wrongly.
#[test]
fn tp2_serves_tp1_tokens() {
    let one = TinyServer::start("");
    let two = TinyServer::start(TP2);
    let greedy = |server: &TinyServer, prompt: &str| -> Vec<u32> {
        let body = json!({"model": server.model, "prompt": prompt, "max_tokens": 16,
                          "ignore_eos": true, "temperature": 0, "logprobs": 0,
                          "return_tokens_as_token_ids": true});
        let resp = server.post("/v1/completions", &body);
        assert_eq!(resp.status, 200, "{}", resp.body);
        choice_token_ids(&resp.json()["choices"][0])
    };
    let prompts = [
        "Hello",
        "Once upon a time",
        "The quick brown fox jumps",
        "1 2 3 4",
    ];
    for prompt in prompts {
        assert_eq!(greedy(&two, prompt), greedy(&one, prompt), "{prompt:?}");
    }
    // Batched: 8 at once through the one engine, the same tokens as one at a time on tp 1.
    let want: Vec<Vec<u32>> = prompts.iter().map(|p| greedy(&one, p)).collect();
    let workers: Vec<_> = (0..8)
        .map(|i| {
            let (addr, model, prompt) = (two.addr, two.model.clone(), prompts[i % 4]);
            std::thread::spawn(move || {
                let body = json!({"model": model, "prompt": prompt, "max_tokens": 16,
                                  "ignore_eos": true, "temperature": 0, "logprobs": 0,
                                  "return_tokens_as_token_ids": true});
                request(addr, "POST", "/v1/completions", Some(&body.to_string()))
            })
        })
        .collect();
    for (i, w) in workers.into_iter().enumerate() {
        let resp = w.join().unwrap();
        assert_eq!(resp.status, 200, "request {i}: {}", resp.body);
        let got = choice_token_ids(&resp.json()["choices"][0]);
        // The tiny model's logits are far apart: batching does not move a greedy token.
        assert_eq!(got, want[i % 4], "request {i}");
    }

    assert_eq!(two.get("/ready").status, 200);
    let parallel = &two.get("/turbine/v1/status").json()["parallel"];
    assert_eq!(parallel["tp"], 2, "{parallel}");
    assert_eq!(parallel["backend"], "host", "{parallel}");
    assert_eq!(
        parallel["groups"],
        json!([{"replica": 0, "ranks": [
            {"rank": 0, "device": 0, "host": parallel["groups"][0]["ranks"][0]["host"]},
            {"rank": 1, "device": 1, "host": parallel["groups"][0]["ranks"][1]["host"]}]}]),
        "{parallel}"
    );
    wait_for(Duration::from_secs(10), "every block free", || {
        two.blocks_used() == 0
    });
}

/// P5 S-6 edge case "cancellation mid-step in a TP group": at tp 2, dropping held streams
/// frees their blocks promptly — the leader's block manager is the only one, so its pool
/// returning the blocks frees them on every rank (worker pools are indexed by the leader's
/// ids) — and the surviving streams finish normally. Breaks if a cancelled sequence keeps its
/// blocks or the group stalls after the cancellation.
#[test]
fn tp2_cancellation_frees_blocks() {
    let server = TinyServer::start_long_with(HOLD_PAUSED, TP2);
    let mut streams: Vec<OpenStream> = (0..4).map(|_| server.hold_stream()).collect();
    wait_for(Duration::from_secs(60), "all 4 streams paused", || {
        let doc = server.scheduler();
        doc["paused"] == 4 && doc["waiting"] == 0
    });
    let all4 = server.blocks_used();
    let kept = streams.split_off(2);
    drop(streams);
    wait_for(
        Duration::from_secs(5),
        "dropped clients' blocks freed",
        || {
            let doc = server.scheduler();
            doc["paused"] == 2 && running(&doc) == 2
        },
    );
    let remaining = server.blocks_used();
    assert!(remaining + 2 <= all4, "{remaining} of {all4}");
    for s in kept {
        assert_eq!(stream_finish_reason(&s.read_rest()), "length");
    }
    wait_for(Duration::from_secs(5), "every block free", || {
        server.blocks_used() == 0
    });
}

/// Each rank's `kv` ledger pool in `/turbine/v1/pressure`, in `memory` order: (device, used
/// bytes, reserved bytes).
fn kv_by_device(server: &TinyServer) -> Vec<(u64, u64, u64)> {
    let doc = server.get("/turbine/v1/pressure").json();
    doc["memory"]
        .as_array()
        .unwrap_or_else(|| panic!("no memory: {doc}"))
        .iter()
        .map(|m| {
            let kv = m["pools"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["name"] == "kv")
                .unwrap_or_else(|| panic!("no kv pool: {doc}"));
            (
                m["device"].as_u64().unwrap(),
                kv["used_bytes"].as_u64().unwrap(),
                kv["reserved_bytes"].as_u64().unwrap(),
            )
        })
        .collect()
}

/// P5 S-8, plan Task 31 (decision "P5: KV admission across tensor-parallel ranks" B): at tp 2
/// on the cpu backend over the host collective, `/turbine/v1/pressure` lists both ranks'
/// ledgers (`memory`, `devices`) and admission reserves every request's KV on both: while two
/// streams hold their KV the two ranks' `kv` pools hold the same used and reserved bytes, a
/// dropped client releases its share on both, and both return to zero at the end. Breaks if
/// admission reserves on the leader's ledger only or a release misses a rank.
#[test]
fn tp2_admission_reserves_on_both_ranks() {
    let server = TinyServer::start_long_with(HOLD_PAUSED, TP2);
    let doc = server.get("/turbine/v1/pressure").json();
    let devices: Vec<u64> = doc["devices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["device"].as_u64().unwrap())
        .collect();
    assert_eq!(devices, [0, 1], "{doc}");
    let idle = kv_by_device(&server);
    assert_eq!(
        idle.iter().map(|k| (k.0, k.1 + k.2)).collect::<Vec<_>>(),
        [(0, 0), (1, 0)],
        "{doc}"
    );

    let mut streams: Vec<OpenStream> = (0..2).map(|_| server.hold_stream()).collect();
    wait_for(Duration::from_secs(60), "both streams paused", || {
        let doc = server.scheduler();
        doc["paused"] == 2 && doc["waiting"] == 0
    });
    // The pressure document is republished on the controller's next tick.
    let mut both = Vec::new();
    wait_for(Duration::from_secs(10), "both ranks hold KV", || {
        both = kv_by_device(&server);
        both.len() == 2 && both[0].1 + both[0].2 > 0
    });
    assert_eq!(
        (both[1].1, both[1].2),
        (both[0].1, both[0].2),
        "rank 1 holds rank 0's KV: {both:?}"
    );
    let held = both[0].1 + both[0].2;

    drop(streams.pop());
    wait_for(Duration::from_secs(10), "one stream's KV released", || {
        both = kv_by_device(&server);
        let total = both[0].1 + both[0].2;
        total > 0 && total < held
    });
    assert_eq!(
        (both[1].1, both[1].2),
        (both[0].1, both[0].2),
        "released on both ranks: {both:?}"
    );

    drop(streams);
    wait_for(Duration::from_secs(10), "both ranks back to zero", || {
        kv_by_device(&server)
            .iter()
            .all(|(_, used, reserved)| used + reserved == 0)
    });
}

/// P5 S-6, `/ready` and data parallelism of tensor-parallel groups: dp 2 × tp 2 on the cpu
/// backend (two groups of two rank threads, the host devices shared) serves 8 concurrent
/// requests on both replicas with tp 1's tokens. Breaks if replicas of groups share a
/// communicator or a pool.
#[test]
fn dp2_of_tp2_groups_serve() {
    let one = TinyServer::start("");
    let server = TinyServer::start(
        "parallel:\n  tensor_parallel_size: 2\n  data_parallel_size: 2\n  \
         collective_backend: host\n  allow_device_sharing: true\n  router: least_loaded\n",
    );
    let body = |model: &str| {
        json!({"model": model, "prompt": "Once upon a time", "max_tokens": 12,
               "ignore_eos": true, "temperature": 0, "logprobs": 0,
               "return_tokens_as_token_ids": true})
    };
    let resp = one.post("/v1/completions", &body(&one.model));
    let want = choice_token_ids(&resp.json()["choices"][0]);
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let (addr, b) = (server.addr, body(&server.model).to_string());
            std::thread::spawn(move || request(addr, "POST", "/v1/completions", Some(&b)))
        })
        .collect();
    for (i, w) in workers.into_iter().enumerate() {
        let resp = w.join().unwrap();
        assert_eq!(resp.status, 200, "request {i}: {}", resp.body);
        assert_eq!(
            choice_token_ids(&resp.json()["choices"][0]),
            want,
            "request {i}"
        );
    }
    let parallel = &server.get("/turbine/v1/status").json()["parallel"];
    assert_eq!(
        (&parallel["tp"], &parallel["dp"]),
        (&json!(2), &json!(2)),
        "{parallel}"
    );
    let metrics = server.metrics();
    for replica in ["0", "1"] {
        let routed: f64 = metrics
            .lines()
            .filter(|l| {
                l.starts_with("turbine_dp_routed_total{")
                    && l.contains(&format!("replica=\"{replica}\""))
            })
            .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
            .sum();
        assert!(routed > 0.0, "replica {replica}: {metrics}");
    }
}
