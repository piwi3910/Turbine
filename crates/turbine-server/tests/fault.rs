//! Fault injection (P3 S-11, S-12, S-16): `turbine-server` built with `--features
//! fault-injection` serving the tiny synthetic checkpoint on the `cpu` reference backend, with
//! `reliability.fault_injection` failing pool reservations and raising device out-of-memory and
//! kernel errors at chosen engine iterations. A default build refuses the section (exit 2).
//!
//! Engine iterations are counted per forward pass after the warm-up: with a short prompt,
//! iteration 1 is the prefill and every later one a decode step.
//!
//! Every wait is bounded and polls; ports come from binding `127.0.0.1:0`.

// A default build runs only the refusal test; the server helpers serve the fault-injection build.
#![cfg_attr(not(feature = "fault-injection"), allow(dead_code, unused_imports))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
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

/// The cpu backend on the tiny checkpoint in `dir` with a 16 MiB KV pool, a 1 MiB emergency
/// reserve, a 2 s circuit cooldown, latency-drift triggers out of reach (debug builds running
/// side by side swing step times far beyond any real drift) and `reliability` keys
/// `reliability` (lines indented by two spaces).
fn write_config(dir: &TempDir, addr: SocketAddr, reliability: &str) -> std::path::PathBuf {
    let model_dir = dir.path().join("tiny-llama");
    write_tiny_llama(&model_dir, 7);
    let yaml = format!(
        "model:\n  path: {}\nserver:\n  listen: {addr}\nexecution:\n  backend: cpu\nkv:\n  gpu:\n    \
         max_bytes: 16MiB\nreliability:\n  emergency_vram_reserve: 1MiB\n  circuit:\n    \
         cooldown: 2s\n    latency_drift_degraded: 1000.0\n    latency_drift_open: 2000.0\n\
         {reliability}",
        model_dir.display()
    );
    let path = dir.path().join("config.yaml");
    std::fs::write(&path, yaml).unwrap();
    path
}

/// The server on `config`, with `env` added to its environment.
fn spawn_with_env(config: &Path, env: &[(&str, String)]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_turbine-server"))
        .arg("--config")
        .arg(config)
        .env_remove("TURBINE_AMD_SMI_LIBRARY")
        .env_remove("TURBINE_KERNEL_LIBRARY")
        .env("RUST_LOG", "info")
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn turbine-server")
}

/// A running server; killed on drop. Its stderr (the log) is drained into a buffer as it
/// arrives, so a chatty server never blocks on a full pipe.
struct Server {
    child: Child,
    addr: SocketAddr,
    /// The served model id.
    model: String,
    stderr: Arc<Mutex<String>>,
    _dir: Option<TempDir>,
}

impl Server {
    /// The tiny checkpoint on the cpu backend with `reliability` keys (see [`write_config`]).
    fn start(reliability: &str) -> Server {
        let dir = TempDir::new("turbine-fault");
        let addr = free_addr();
        let config = write_config(&dir, addr, reliability);
        Server::launch(&config, addr, "tiny-llama", Some(dir), READY_LIMIT)
    }

    fn launch(
        config: &Path,
        addr: SocketAddr,
        model: &str,
        dir: Option<TempDir>,
        ready_limit: Duration,
    ) -> Server {
        Server::launch_with_env(config, addr, model, dir, ready_limit, &[])
    }

    /// [`Server::launch`] with more environment variables for the server.
    fn launch_with_env(
        config: &Path,
        addr: SocketAddr,
        model: &str,
        dir: Option<TempDir>,
        ready_limit: Duration,
        env: &[(&str, String)],
    ) -> Server {
        let mut child = spawn_with_env(config, env);
        let stderr = Arc::new(Mutex::new(String::new()));
        let mut pipe = BufReader::new(child.stderr.take().expect("stderr is piped"));
        let sink = Arc::clone(&stderr);
        std::thread::spawn(move || {
            let mut line = String::new();
            while pipe.read_line(&mut line).is_ok_and(|n| n > 0) {
                sink.lock().unwrap().push_str(&line);
                line.clear();
            }
        });
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                std::thread::sleep(Duration::from_millis(100));
                panic!(
                    "turbine-server exited early ({status}); stderr:\n{}",
                    stderr.lock().unwrap()
                );
            }
            if TcpStream::connect(addr).is_ok()
                && request(addr, "GET", "/ready", None).status == 200
            {
                break;
            }
            assert!(started.elapsed() < ready_limit, "never became ready");
            std::thread::sleep(POLL);
        }
        Server {
            child,
            addr,
            model: model.to_string(),
            stderr,
            _dir: dir,
        }
    }

    fn get(&self, path: &str) -> Response {
        request(self.addr, "GET", path, None)
    }

    fn complete(&self, max_tokens: u32, stream: bool) -> Response {
        let body = json!({"model": self.model, "prompt": "Hello", "max_tokens": max_tokens,
                          "ignore_eos": true, "temperature": 0.0, "stream": stream});
        request(
            self.addr,
            "POST",
            "/v1/completions",
            Some(&body.to_string()),
        )
    }

    fn metrics(&self) -> String {
        self.get("/metrics").body
    }

    fn pressure(&self) -> Value {
        let resp = self.get("/turbine/v1/pressure");
        assert_eq!(resp.status, 200, "{}", resp.body);
        resp.json()
    }

    /// Waits for the process to exit; its status and stderr.
    fn wait_exit(mut self, limit: Duration) -> (ExitStatus, String) {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                // Let the drain thread read the last lines.
                std::thread::sleep(Duration::from_millis(200));
                let stderr = self.stderr.lock().unwrap().clone();
                return (status, stderr);
            }
            assert!(
                started.elapsed() < limit,
                "turbine-server did not exit within {limit:?}"
            );
            std::thread::sleep(POLL);
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
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

fn request(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Response {
    let mut conn = TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    let body = body.unwrap_or("");
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut reader = BufReader::new(conn);
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
    let mut body = String::new();
    if head.contains("transfer-encoding: chunked") {
        loop {
            let mut size_line = String::new();
            reader.read_line(&mut size_line).expect("read chunk size");
            let size = usize::from_str_radix(size_line.trim(), 16)
                .unwrap_or_else(|_| panic!("bad chunk size line {size_line:?}"));
            let mut data = vec![0u8; size + 2];
            reader.read_exact(&mut data).expect("read chunk");
            if size == 0 {
                break;
            }
            data.truncate(size);
            body.push_str(&String::from_utf8(data).expect("UTF-8 chunk"));
        }
    } else {
        reader.read_to_string(&mut body).expect("read body");
    }
    Response { status, head, body }
}

/// The value of a sample line `<series> <value>` in `/metrics`.
fn sample(metrics: &str, series: &str) -> Option<f64> {
    metrics.lines().find_map(|l| {
        l.strip_prefix(series)
            .and_then(|rest| rest.strip_prefix(' '))
            .and_then(|v| v.trim().parse().ok())
    })
}

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

/// A default build has no fault injection: the section is an unknown-key error (exit 2).
#[cfg(not(feature = "fault-injection"))]
#[test]
fn fault_injection_section_needs_the_feature() {
    let dir = TempDir::new("turbine-fault-default");
    let config = write_config(
        &dir,
        free_addr(),
        "  fault_injection:\n    alloc_fail_every: 5\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_turbine-server"))
        .arg("--config")
        .arg(&config)
        .arg("--check-config")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("reliability.fault_injection"), "{stderr}");
}

/// P3 S-16: `alloc_fail_every: 5` fails exactly every fifth pool reservation after startup (the
/// startup reservations — weights, workspace, emergency reserve — are not subject to
/// injection), each counted in `turbine_allocation_failures_total`. A request whose reservation
/// fails waits in the admission queue and takes the next one, so all 17 sequential requests
/// complete with 21 reservation attempts, of which the 5th, 10th, 15th and 20th failed.
#[cfg(feature = "fault-injection")]
#[test]
fn alloc_fail_every_injects() {
    let server = Server::start("  fault_injection:\n    alloc_fail_every: 5\n");
    for i in 0..17 {
        let resp = server.complete(2, false);
        assert_eq!(resp.status, 200, "request {i}: {}", resp.body);
        assert_eq!(resp.json()["choices"][0]["finish_reason"], "length");
    }
    let metrics = server.metrics();
    assert_eq!(
        sample(
            &metrics,
            r#"turbine_allocation_failures_total{device="0",pool="kv"}"#
        ),
        Some(4.0),
        "{metrics}"
    );
    for pool in ["weights", "workspace", "reserve", "runtime"] {
        let series = format!(r#"turbine_allocation_failures_total{{device="0",pool="{pool}"}}"#);
        assert!(
            sample(&metrics, &series).is_none_or(|v| v == 0.0),
            "{series}\n{metrics}"
        );
    }
    // Every reservation was released: the KV pool is idle again (the document is refreshed on
    // the controller's next tick).
    wait_for(Duration::from_secs(5), "kv pool idle", || {
        let kv = kv_pool(&server.pressure());
        kv["used_bytes"] == 0 && kv["reserved_bytes"] == 0
    });
}

/// The `kv` pool object of a pressure document.
fn kv_pool(doc: &Value) -> Value {
    doc["memory"][0]["pools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "kv")
        .cloned()
        .unwrap_or_else(|| panic!("no kv pool in {doc}"))
}

/// P3 S-11: a device out-of-memory error at iteration 3 enters SURVIVAL, releases the emergency
/// reserve and retries the iteration; the request completes with every token, the recovery is
/// counted, and the circuit is DEGRADED (`oom_recovered`). The state then steps down one level
/// per `deescalate_dwell` (1 s here), re-acquiring the reserve before it leaves RED, and the
/// engine serves new requests again.
#[cfg(feature = "fault-injection")]
#[test]
fn injected_oom_is_recovered() {
    let server = Server::start(
        "  pressure:\n    deescalate_dwell: 1s\n  fault_injection:\n    oom_at_iteration: 3\n",
    );
    let resp = server.complete(8, false);
    assert_eq!(resp.status, 200, "{}", resp.body);
    let body = resp.json();
    assert_eq!(body["usage"]["completion_tokens"], 8, "{body}");
    let metrics = server.metrics();
    for (series, value) in [
        (r#"turbine_recoveries_total{outcome="recovered"}"#, 1.0),
        ("turbine_recovery_retries_total", 1.0),
        (
            r#"turbine_emergency_reserve_releases_total{device="0"}"#,
            1.0,
        ),
    ] {
        assert_eq!(sample(&metrics, series), Some(value), "{series}\n{metrics}");
    }
    let doc = server.pressure();
    assert_eq!(doc["circuit"]["state"], "DEGRADED", "{doc}");
    assert_eq!(doc["circuit"]["last_reason"], "oom_recovered", "{doc}");
    assert!(
        doc["transitions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["to"] == "SURVIVAL" && t["signal"] == "allocation_failure"),
        "{doc}"
    );
    // Out of SURVIVAL, RED and ORANGE (whose frozen batch holds no request) the engine admits
    // again, with the emergency reserve held once more.
    wait_for(Duration::from_secs(30), "state YELLOW or GREEN", || {
        let state = server.pressure()["state"].clone();
        state == "YELLOW" || state == "GREEN"
    });
    assert_eq!(
        server.pressure()["memory"][0]["emergency_reserve_held"],
        true
    );
    assert_eq!(server.complete(2, false).status, 200);
}

/// P3 S-12: a non-sticky kernel error fails the iteration's request and opens the circuit:
/// `/ready` and new requests answer 503 `circuit_open` (with `Retry-After`); after the
/// cooldown, internal probes (not client requests) succeed and the circuit returns to HEALTHY.
#[cfg(feature = "fault-injection")]
#[test]
fn kernel_error_opens_the_circuit_and_probes_close_it() {
    let server = Server::start("  fault_injection:\n    kernel_error_at_iteration: 3\n");
    let resp = server.complete(8, true);
    assert_eq!(resp.status, 200, "{}", resp.body);
    let data = resp.sse_data();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    let err: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(err["error"]["code"], "internal_error", "{err}");

    let ready = server.get("/ready");
    assert_eq!(ready.status, 503, "{}", ready.body);
    assert_eq!(ready.json()["reason"], "circuit_open");
    let refused = server.complete(2, false);
    assert_eq!(refused.status, 503, "{}", refused.body);
    assert_eq!(refused.error_code(), "circuit_open");
    assert!(refused.head.contains("retry-after: "), "{}", refused.head);
    assert_eq!(server.pressure()["circuit"]["last_reason"], "device_error");

    wait_for(Duration::from_secs(30), "circuit HEALTHY again", || {
        server.pressure()["circuit"]["state"] == "HEALTHY"
    });
    assert_eq!(server.get("/ready").status, 200);
    assert_eq!(server.complete(2, false).status, 200);
    let metrics = server.metrics();
    // Two client requests completed or failed; the probes are not counted.
    let ok = sample(
        &metrics,
        r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"}"#,
    );
    assert_eq!(ok, Some(1.0), "{metrics}");
    // One device error and one return to HEALTHY; a probe slower than twice the baseline
    // step (CPU contention on a loaded test host) fails and reopens the circuit, so the
    // drain/probe cycle may run more than once.
    for (from, to, reason, exactly_once) in [
        ("HEALTHY", "CIRCUIT_OPEN", "device_error", true),
        ("CIRCUIT_OPEN", "DRAINING", "drain_started", false),
        ("DRAINING", "PROBING", "drain_complete", false),
        ("PROBING", "HEALTHY", "probes_succeeded", true),
    ] {
        let series = format!(
            r#"turbine_circuit_transitions_total{{from="{from}",to="{to}",reason="{reason}"}}"#
        );
        let n = sample(&metrics, &series);
        if exactly_once {
            assert_eq!(n, Some(1.0), "{series}\n{metrics}");
        } else {
            assert!(n.is_some_and(|n| n >= 1.0), "{series}\n{metrics}");
        }
    }
}

/// The streaming request's SSE body ends with a `resource_exhausted` error event, then
/// `[DONE]`; `/ready` answers 503 `circuit_open` while the process lives; it exits 3.
#[cfg(feature = "fault-injection")]
fn assert_sticky_exit(server: Server, body: Value, limit: Duration) {
    let addr = server.addr;
    let resp = request(addr, "POST", "/v1/completions", Some(&body.to_string()));
    assert_eq!(resp.status, 200, "{}", resp.body);
    let data = resp.sse_data();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    let err: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(err["error"]["code"], "resource_exhausted", "{err}");
    assert_eq!(err["error"]["type"], "server_error", "{err}");

    // Until the process exits, `/ready` names the open circuit.
    if let Ok(mut conn) = TcpStream::connect(addr) {
        conn.set_read_timeout(Some(Duration::from_secs(5))).ok();
        let _ = write!(
            conn,
            "GET /ready HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"
        );
        let mut text = String::new();
        if conn.read_to_string(&mut text).is_ok() && !text.is_empty() {
            assert!(text.starts_with("HTTP/1.1 503"), "{text}");
            assert!(text.contains("circuit_open"), "{text}");
        }
    }
    let (status, stderr) = server.wait_exit(limit);
    assert_eq!(status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("device_fatal"), "{stderr}");
}

/// P3 S-12 on the cpu backend (the lab test `sticky_device_error_exits_3` runs the same on a
/// GPU): a sticky device error at iteration 3 ends the stream with an error event then
/// `[DONE]`, `/ready` answers 503 `circuit_open`, and the process exits with code 3.
#[cfg(feature = "fault-injection")]
#[test]
fn sticky_device_error_exits_3_on_cpu() {
    let server = Server::start(
        "  fault_injection:\n    kernel_error_at_iteration: 3\n    kernel_error_sticky: true\n",
    );
    let body = json!({"model": "tiny-llama", "prompt": "Hello", "max_tokens": 64,
                      "ignore_eos": true, "stream": true});
    assert_sticky_exit(server, body, Duration::from_secs(120));
}

/// P3 S-12, S-16 (lab, Task 17): the real Llama-3.2-3B on the GPU backend under test
/// (`TURBINE_TEST_BACKEND`, kernel library from `TURBINE_KERNEL_LIBRARY`) with a sticky device
/// error injected at iteration 3: the stream ends with an error event then `[DONE]`, `/ready`
/// answers 503 `circuit_open`, and the process exits 3 within
/// `reliability.circuit.drain_timeout` (120 s).
#[cfg(feature = "fault-injection")]
#[test]
#[ignore = "lab: needs a GPU backend, its kernel library and the Llama-3.2-3B weights"]
fn sticky_device_error_exits_3() {
    use turbine_kernels::test_support::{require_backend, require_env_dir};
    let backend = std::env::var("TURBINE_TEST_BACKEND")
        .expect("TURBINE_TEST_BACKEND is not set; set it to the backend under test (hip or cuda)");
    if !require_backend(&backend) {
        return;
    }
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let dir = TempDir::new("turbine-fault-lab");
    let addr = free_addr();
    let config = dir.path().join("config.yaml");
    // The harness keeps TURBINE_KERNEL_LIBRARY from the server (the cpu tests stay
    // hermetic), so the lab Job's library is named in the configuration.
    let library = std::env::var("TURBINE_KERNEL_LIBRARY")
        .map(|l| format!("  kernel_library: {l}\n"))
        .unwrap_or_default();
    std::fs::write(
        &config,
        format!(
            "model:\n  path: {}\n  served_name: lab-llama\nserver:\n  listen: {addr}\n\
             execution:\n  backend: {backend}\n{library}reliability:\n  fault_injection:\n    \
             kernel_error_at_iteration: 3\n    kernel_error_sticky: true\n",
            model_dir.display()
        ),
    )
    .unwrap();
    // Weights load and warm-up of a debug build on the GPU.
    let server = Server::launch(
        &config,
        addr,
        "lab-llama",
        Some(dir),
        Duration::from_secs(900),
    );
    let body = json!({"model": "lab-llama", "prompt": "Once upon a time", "max_tokens": 64,
                      "ignore_eos": true, "stream": true});
    assert_sticky_exit(server, body, Duration::from_secs(120));
}

/// P5 Task 28 (lab, `scripts/lab-test.sh novanas --gpus 2 --features fault-injection`): the real
/// Llama-3.2-3B at tp 2 in `local` rank mode on both GPUs of the backend under test (the `auto`
/// collective backend) with one worker rank's communicator aborted during a request
/// (`TURBINE_FAULT_TP_ABORT_FILE`, fault-injection build): the stream ends with `replica_failed`,
/// `/ready` answers 503 `circuit_open` (no exit 3), and the circuit's probe re-creates the
/// communicator on both GPUs — `/ready` is 200 again and a greedy completion is identical to the
/// one before the failure. The whole test is bounded by 10 minutes.
#[cfg(feature = "fault-injection")]
#[test]
#[ignore = "lab: needs two GPUs, their kernel library and the Llama-3.2-3B weights"]
fn tp2_collective_failure_recovers_on_gpu() {
    use turbine_kernels::test_support::{require_backend, require_env_dir};
    let started = Instant::now();
    let limit = Duration::from_secs(600);
    let backend = std::env::var("TURBINE_TEST_BACKEND")
        .expect("TURBINE_TEST_BACKEND is not set; set it to the backend under test (hip or cuda)");
    if !require_backend(&backend) {
        return;
    }
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let dir = TempDir::new("turbine-fault-tp2-lab");
    let trigger = dir.path().join("abort-one-rank");
    let addr = free_addr();
    let config = dir.path().join("config.yaml");
    let library = std::env::var("TURBINE_KERNEL_LIBRARY")
        .map(|l| format!("  kernel_library: {l}\n"))
        .unwrap_or_default();
    std::fs::write(
        &config,
        format!(
            "model:\n  path: {}\n  served_name: lab-llama\nserver:\n  listen: {addr}\n\
             execution:\n  backend: {backend}\n{library}parallel:\n  tensor_parallel_size: 2\n\
             reliability:\n  circuit:\n    cooldown: 2s\n    latency_drift_degraded: 1000.0\n    \
             latency_drift_open: 2000.0\n",
            model_dir.display()
        ),
    )
    .unwrap();
    let env = [("TURBINE_FAULT_TP_ABORT_FILE", trigger.display().to_string())];
    // Weights load and warm-up of a debug build on two GPUs.
    let server = Server::launch_with_env(
        &config,
        addr,
        "lab-llama",
        Some(dir),
        Duration::from_secs(420),
        &env,
    );
    let greedy = |server: &Server| {
        let resp = server.complete(16, false);
        assert_eq!(resp.status, 200, "{}", resp.body);
        resp.json()["choices"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let before = greedy(&server);

    std::fs::write(&trigger, b"abort").unwrap();
    let resp = server.complete(64, true);
    assert_eq!(resp.status, 200, "{}", resp.body);
    let data = resp.sse_data();
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{data:?}");
    let err: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(err["error"]["code"], "replica_failed", "{err}");
    assert!(!trigger.exists(), "the abort was injected");
    let ready = server.get("/ready");
    assert_eq!(ready.status, 503, "{}", ready.body);
    assert_eq!(ready.json()["reason"], "circuit_open", "{}", ready.body);

    let left = limit.saturating_sub(started.elapsed());
    wait_for(left, "/ready 200 after the re-creation", || {
        server.get("/ready").status == 200
    });
    assert_eq!(
        greedy(&server),
        before,
        "the re-created group gives the same tokens"
    );
    let metrics = server.metrics();
    let recovered = sample(
        &metrics,
        r#"turbine_circuit_transitions_total{from="PROBING",to="HEALTHY",reason="probes_succeeded"}"#,
    );
    assert_eq!(recovered, Some(1.0), "{metrics}");
    assert!(started.elapsed() < limit, "{:?}", started.elapsed());
}
