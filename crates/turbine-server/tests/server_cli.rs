//! Black-box tests of the `turbine-server` binary: exit codes, bind order, graceful shutdown.
//!
//! Every wait is bounded and polls; ports come from binding `127.0.0.1:0`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::write_tiny_llama;

const POLL: Duration = Duration::from_millis(20);
/// The server's test-only cap on each accepted connection's kernel send buffer.
const SEND_BUFFER_ENV: &str = "TURBINE_TEST_SOCKET_SEND_BUFFER";
/// Kernel buffer bytes on each end of a held stream (Linux doubles and floors them), so a
/// client that stops reading pauses its request after a bounded number of tokens instead of
/// after however many megabytes the loopback autotuning allows.
const HELD_SOCKET_BUFFER: usize = 4096;

/// A config file in a per-test directory, removed on drop.
struct TempConfig {
    dir: PathBuf,
    path: PathBuf,
}

impl TempConfig {
    fn new(test: &str, yaml: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("turbine-server-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        std::fs::write(&path, yaml).unwrap();
        TempConfig { dir, path }
    }
}

impl Drop for TempConfig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A port the OS just handed out and released.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Kills the server process when a test ends early (a failed assertion panics before the test's
/// own shutdown), so a failing run never leaves an orphaned `turbine-server` behind.
struct KillOnDrop(u32);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = Command::new("kill")
            .args(["-9", &self.0.to_string()])
            .stderr(Stdio::null())
            .status();
    }
}

fn spawn_server(args: &[&str], config: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_turbine-server"))
        .args(args)
        .arg("--config")
        .arg(config)
        .env_remove("TURBINE_AMD_SMI_LIBRARY")
        .env(SEND_BUFFER_ENV, HELD_SOCKET_BUFFER.to_string())
        .env("RUST_LOG", "info")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn turbine-server")
}

/// Wait for exit; kill the child and fail if it is still running after `limit`.
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

/// Poll `GET /health` until 200; fail early if the server exits.
fn wait_until_serving(child: &mut Child, addr: SocketAddr) {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(20) {
        if let Some(status) = child.try_wait().unwrap() {
            let mut stderr = String::new();
            if let Some(mut e) = child.stderr.take() {
                e.read_to_string(&mut stderr).ok();
            }
            panic!("turbine-server exited early ({status}); stderr:\n{stderr}");
        }
        if let Ok(mut s) = TcpStream::connect(addr) {
            s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            s.write_all(b"GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut resp = String::new();
            s.read_to_string(&mut resp).ok();
            if resp.starts_with("HTTP/1.1 200") {
                return;
            }
        }
        std::thread::sleep(POLL);
    }
    child.kill().ok();
    panic!("turbine-server never served /health on {addr}");
}

/// Poll until `addr` refuses connections (the listener was dropped).
fn wait_until_refusing(addr: SocketAddr, limit: Duration) {
    let started = Instant::now();
    while started.elapsed() < limit {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_err() {
            return;
        }
        std::thread::sleep(POLL);
    }
    panic!("turbine-server kept accepting on {addr} {limit:?} after SIGTERM");
}

/// Read until the blank line that ends a response head.
fn read_head(conn: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        let n = conn.read(&mut byte).expect("read response head");
        assert!(n > 0, "connection closed mid-head: {head:?}");
        head.push(byte[0]);
    }
    String::from_utf8_lossy(&head).into_owned()
}

#[test]
fn invalid_config_exits_2_before_bind() {
    let port = free_port();
    let cfg = TempConfig::new(
        "invalid",
        &format!(
            "model:\n  path: /m\nserver:\n  listen: 127.0.0.1:{port}\nkv:\n  cpu:\n    max_bytes: 64XB\n"
        ),
    );
    let out = wait_with_timeout(spawn_server(&[], &cfg.path), Duration::from_secs(20));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("turbine-server: invalid configuration:"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("kv.cpu.max_bytes"), "stderr: {stderr}");
    TcpListener::bind(("127.0.0.1", port)).expect("the configured port must still be free");

    // --check-config reports the same error without discovering devices or binding.
    let out = wait_with_timeout(
        spawn_server(&["--check-config"], &cfg.path),
        Duration::from_secs(20),
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("kv.cpu.max_bytes"));

    let ok_cfg = TempConfig::new("check-ok", "model:\n  path: /m\n");
    let out = wait_with_timeout(
        spawn_server(&["--check-config"], &ok_cfg.path),
        Duration::from_secs(20),
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "config ok");
}

/// A tiny synthetic checkpoint served as `m` on the cpu reference backend (Phase 1 needs a
/// loadable model before the listener binds). The directory is removed on drop.
fn tiny_model_yaml(addr: SocketAddr) -> (TempDir, String) {
    let dir = TempDir::new("turbine-server-cli-model");
    write_tiny_llama(dir.path(), 7);
    let yaml = format!(
        "model:\n  path: {}\n  served_name: m\n  max_seq_len: 64\nserver:\n  listen: {addr}\n\
         execution:\n  backend: cpu\nreliability:\n  emergency_vram_reserve: 1MiB\n",
        dir.path().display()
    );
    (dir, yaml)
}

/// Poll `GET /ready` until 200; fail early if the server exits.
fn wait_until_ready(child: &mut Child, addr: SocketAddr) {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(60) {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("turbine-server exited early ({status})");
        }
        if let Ok(mut s) = TcpStream::connect(addr) {
            s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            s.write_all(b"GET /ready HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut resp = String::new();
            s.read_to_string(&mut resp).ok();
            if resp.starts_with("HTTP/1.1 200") {
                return;
            }
        }
        std::thread::sleep(POLL);
    }
    child.kill().ok();
    panic!("turbine-server never became ready on {addr}");
}

#[test]
fn port_in_use_exits_1() {
    let holder = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = holder.local_addr().unwrap();
    let (_model, yaml) = tiny_model_yaml(addr);
    let cfg = TempConfig::new("port-in-use", &yaml);
    let out = wait_with_timeout(spawn_server(&[], &cfg.path), Duration::from_secs(20));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(
        stderr.contains(&format!("turbine-server: cannot bind {addr}:")),
        "stderr: {stderr}"
    );
    drop(holder);
}

/// Sends SIGTERM to `child`.
#[cfg(unix)]
fn sigterm(child: &Child) {
    let killed = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
}

#[cfg(unix)]
#[test]
fn sigterm_graceful_shutdown() {
    let addr = SocketAddr::from(([127, 0, 0, 1], free_port()));
    let (_model, yaml) = tiny_model_yaml(addr);
    let cfg = TempConfig::new("sigterm", &yaml);
    let mut child = spawn_server(&[], &cfg.path);
    let _reaper = KillOnDrop(child.id());
    wait_until_serving(&mut child, addr);
    wait_until_ready(&mut child, addr);

    // Put a request in flight: headers plus 10 of 30 declared body bytes. `Expect: 100-continue`
    // makes the server say when the handler starts reading the body, so the request is provably
    // in flight before the signal (no timing guess).
    let body = br#"{"model":"m","prompt":"hello"}"#;
    assert_eq!(body.len(), 30);
    let mut conn = TcpStream::connect(addr).unwrap();
    conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    write!(
        conn,
        "POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nExpect: 100-continue\r\n\r\n",
        body.len()
    )
    .unwrap();
    let interim = read_head(&mut conn);
    assert!(interim.starts_with("HTTP/1.1 100"), "interim: {interim}");
    conn.write_all(&body[..10]).unwrap();

    sigterm(&child);
    let signalled = Instant::now();
    // No generation runs, so there is nothing to drain: the listener closes at once.
    wait_until_refusing(addr, Duration::from_secs(5));

    // The open connection is still answered; its request arrived after shutdown began, so it
    // is refused with 503 `shutting_down` (P2 S-13).
    conn.write_all(&body[10..]).unwrap();
    let mut resp = String::new();
    conn.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 503"), "response: {resp}");
    assert!(
        resp.contains(r#""code":"shutting_down""#),
        "response: {resp}"
    );

    let out = wait_with_timeout(child, Duration::from_secs(5));
    assert!(signalled.elapsed() < Duration::from_secs(5));
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `max_position_embeddings` of the checkpoint `long_model_yaml` serves, so streams can be held.
const LONG_POSITIONS: u32 = 8192;
/// Tokens of a stream that must outlive the shutdown grace: far more events than the output
/// channel (256), hyper's write queue and the capped socket buffers hold, so a client that stops
/// reading pauses it (after 300–500 tokens on Linux, 880 or more on macOS: see [`SHORT_TOKENS`]).
const LONG_TOKENS: u32 = 1500;
/// Tokens of a stream that is paused at the signal and completes within [`GRACE`] once read: a
/// little over the tokens a stream generates before it pauses. On Linux both socket caps hold
/// and a stream that stops reading pauses after about 300 tokens (up to 500 on a loaded host);
/// macOS loopback keeps a receive buffer of at least 320 KiB whatever `SO_RCVBUF` says (more as
/// it autotunes), so it pauses after 880 tokens or more.
const SHORT_TOKENS: u32 = if cfg!(target_os = "macos") { 1200 } else { 600 };
/// `server.shutdown_grace` of the drain test: time for the short stream's remaining tokens
/// (at most about 300 on Linux, 320 on macOS) in a debug build.
const GRACE: Duration = Duration::from_secs(if cfg!(target_os = "macos") { 5 } else { 2 });

/// The tiny checkpoint patched to `LONG_POSITIONS` positions, served as `m` on the cpu backend
/// with a 16 MiB KV pool; `server_extra` is appended to the `server` section verbatim.
fn long_model_yaml(addr: SocketAddr, server_extra: &str) -> (TempDir, String) {
    let dir = TempDir::new("turbine-server-cli-long-model");
    write_tiny_llama(dir.path(), 7);
    let path = dir.path().join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    cfg["max_position_embeddings"] = serde_json::json!(LONG_POSITIONS);
    std::fs::write(&path, cfg.to_string()).unwrap();
    let yaml = format!(
        "model:\n  path: {}\n  served_name: m\nserver:\n  listen: {addr}\n{server_extra}\
         execution:\n  backend: cpu\nreliability:\n  emergency_vram_reserve: 1MiB\n\
         kv:\n  gpu:\n    max_bytes: 16MiB\n",
        dir.path().display()
    );
    (dir, yaml)
}

/// Writes one HTTP/1.1 request with `Connection: close`.
fn write_request(conn: &mut TcpStream, method: &str, path: &str, body: &str) {
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}

/// One request; returns the status and the whole response (head and body).
fn http(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut conn = TcpStream::connect(addr).unwrap();
    conn.set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write_request(&mut conn, method, path, body);
    let mut resp = String::new();
    conn.read_to_string(&mut resp).ok();
    let status = resp
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (status, resp)
}

/// A streaming completion whose body is read only on demand.
struct HeldStream {
    reader: BufReader<TcpStream>,
}

impl HeldStream {
    /// Starts a stream of `max_tokens` large events (20 logprobs each) and reads its head and
    /// first chunk; the client then stops reading, so the engine pauses the request.
    fn open(addr: SocketAddr, max_tokens: u32) -> HeldStream {
        let body = serde_json::json!({"model": "m", "prompt": "Once upon a time",
            "max_tokens": max_tokens, "ignore_eos": true, "stream": true, "logprobs": 20});
        // The receive buffer is set before connecting so the advertised window stays small.
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::STREAM,
            None,
        )
        .unwrap();
        socket.set_recv_buffer_size(HELD_SOCKET_BUFFER).unwrap();
        socket.connect(&addr.into()).unwrap();
        let mut conn: TcpStream = socket.into();
        conn.set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write_request(&mut conn, "POST", "/v1/completions", &body.to_string());
        let mut reader = BufReader::new(conn);
        let mut head = String::new();
        loop {
            let mut line = String::new();
            assert!(reader.read_line(&mut line).unwrap() > 0, "closed: {head}");
            if line == "\r\n" {
                break;
            }
            head.push_str(&line.to_ascii_lowercase());
        }
        assert!(head.starts_with("http/1.1 200"), "{head}");
        assert!(head.contains("transfer-encoding: chunked"), "{head}");
        let mut stream = HeldStream { reader };
        let first = stream.chunk().expect("first SSE chunk");
        assert!(first.starts_with("data: "), "{first}");
        stream
    }

    /// One chunk of the chunked body; `None` at the terminating chunk.
    fn chunk(&mut self) -> Option<String> {
        let mut size_line = String::new();
        self.reader.read_line(&mut size_line).expect("chunk size");
        let size = usize::from_str_radix(size_line.trim(), 16)
            .unwrap_or_else(|_| panic!("bad chunk size line {size_line:?}"));
        let mut data = vec![0u8; size + 2];
        self.reader.read_exact(&mut data).expect("chunk data");
        if size == 0 {
            return None;
        }
        data.truncate(size);
        Some(String::from_utf8(data).expect("UTF-8 chunk"))
    }

    /// Reads the rest of the stream and returns its `data:` payloads.
    fn read_rest(mut self) -> Vec<String> {
        let mut body = String::new();
        while let Some(chunk) = self.chunk() {
            body.push_str(&chunk);
        }
        body.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }
}

/// Paused requests in `/turbine/v1/scheduler`.
fn paused_requests(addr: SocketAddr) -> u64 {
    let (status, resp) = http(addr, "GET", "/turbine/v1/scheduler", "");
    assert_eq!(status, 200, "{resp}");
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("");
    let doc: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    doc["paused"].as_u64().unwrap_or(0)
}

#[cfg(unix)]
#[test]
fn sigterm_drains_then_cancels() {
    let addr = SocketAddr::from(([127, 0, 0, 1], free_port()));
    let (_model, yaml) =
        long_model_yaml(addr, &format!("  shutdown_grace: {}s\n", GRACE.as_secs()));
    let cfg = TempConfig::new("sigterm-drain", &yaml);
    let mut child = spawn_server(&[], &cfg.path);
    let _reaper = KillOnDrop(child.id());
    wait_until_serving(&mut child, addr);
    wait_until_ready(&mut child, addr);

    // Both generations are running, paused on their unread streams, when the signal arrives.
    let short = HeldStream::open(addr, SHORT_TOKENS);
    let long = HeldStream::open(addr, LONG_TOKENS);
    let started = Instant::now();
    while paused_requests(addr) < 2 {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "both streams should be paused"
        );
        std::thread::sleep(POLL);
    }

    sigterm(&child);
    let signalled = Instant::now();

    // Draining: `/ready` and new requests answer 503 `shutting_down`; the listener stays open.
    loop {
        let (status, resp) = http(addr, "GET", "/ready", "");
        if status == 503 && resp.contains(r#""reason":"shutting_down""#) {
            break;
        }
        assert!(
            signalled.elapsed() < Duration::from_secs(1),
            "/ready: {resp}"
        );
        std::thread::sleep(POLL);
    }
    let (status, resp) = http(
        addr,
        "POST",
        "/v1/completions",
        r#"{"model":"m","prompt":"hi","max_tokens":2}"#,
    );
    assert_eq!(status, 503, "{resp}");
    assert!(resp.contains(r#""code":"shutting_down""#), "{resp}");

    // The short generation completes within the grace.
    let data = short.read_rest();
    assert!(
        signalled.elapsed() < GRACE,
        "short stream took {:?}",
        signalled.elapsed()
    );
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
    let finish = data
        .iter()
        .rev()
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .find_map(|c| {
            c["choices"][0]["finish_reason"]
                .as_str()
                .map(str::to_string)
        });
    assert_eq!(finish.as_deref(), Some("length"));

    // The long one is still unread when the grace ends: it is cancelled with `shutting_down`.
    std::thread::sleep((GRACE + Duration::from_millis(100)).saturating_sub(signalled.elapsed()));
    let data = long.read_rest();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
    let error: serde_json::Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(error["error"]["code"], "shutting_down", "{error}");

    let out = wait_with_timeout(
        child,
        (GRACE + Duration::from_secs(1)).saturating_sub(signalled.elapsed()),
    );
    assert!(
        signalled.elapsed() < GRACE + Duration::from_secs(1),
        "exit took {:?}",
        signalled.elapsed()
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
