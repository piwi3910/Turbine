//! Black-box tests of the `turbine-server` binary: exit codes, bind order, graceful shutdown.
//!
//! Every wait is bounded and polls; ports come from binding `127.0.0.1:0`.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::write_tiny_llama;

const POLL: Duration = Duration::from_millis(20);

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

fn spawn_server(args: &[&str], config: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_turbine-server"))
        .args(args)
        .arg("--config")
        .arg(config)
        .env_remove("TURBINE_AMD_SMI_LIBRARY")
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

#[cfg(unix)]
#[test]
fn sigterm_graceful_shutdown() {
    let addr = SocketAddr::from(([127, 0, 0, 1], free_port()));
    let (_model, yaml) = tiny_model_yaml(addr);
    let cfg = TempConfig::new("sigterm", &yaml);
    let mut child = spawn_server(&[], &cfg.path);
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

    let killed = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let signalled = Instant::now();
    // Shutdown has begun once new connections are refused.
    wait_until_refusing(addr, Duration::from_secs(5));

    // The in-flight request still completes.
    conn.write_all(&body[10..]).unwrap();
    let mut resp = String::new();
    conn.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 200"), "response: {resp}");
    assert!(
        resp.contains(r#""object":"text_completion""#),
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
