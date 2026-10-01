//! Phase 4 KV correctness on the GPU (P4 S-3, S-11, S-17; plan Task 16).
//!
//! The GPU tests are ignored (lab only, run by `scripts/lab-test.sh novanas`, later `dgx-spark`):
//! they start `turbine-server` on `TURBINE_TEST_MODEL_DIR` (Llama-3.2-3B-Instruct BF16) with
//! `scripts/lab/phase4-novanas.yaml` and `--set` overrides, and compare greedy completions.
//!
//! - `prefix_reuse_matches_cold`: 5 prompts of more than two 128-token blocks, 64 greedy tokens
//!   each, first with `kv.prefix_sharing: false` (cold), then on a fresh server with sharing on,
//!   each sent twice: every answer equals the cold one, and the repeat reports
//!   `cached_tokens` > 0.
//! - `prefix_reuse_suffix_lengths_match_cold`: token-id prompts of whole 128-token blocks plus an
//!   uncached suffix of 1 to 700 tokens (2 to 64 tokens make a step the size of a decode batch,
//!   more a prefill-sized one), 64 greedy tokens with their logprobs: cold (sharing off), then warm
//!   after a primer prompt that shares the whole blocks. The warm run reuses those blocks — all
//!   but the last one when the suffix is a single token: the KV planner leaves at least two
//!   tokens to prefill, since a one-token step is decode-shaped — and gives the cold answer bit
//!   for bit (text and every token's logprob): Llama's prefill GEMMs must compute a row alike at
//!   any row count (decision 2026-09-27, "Pre-Phase-5 #1 follow-up").
//! - `prefix_reuse_matches_cold_tp2`, `prefix_reuse_suffix_lengths_match_cold_tp2`: the same at
//!   tensor parallelism 2 over both R9700s (P5 Task 29: the tp 2 per-rank GEMM shapes have
//!   invariant rows too); they need `scripts/lab-test.sh novanas --gpus 2` (a one-GPU Job skips
//!   them: `TWO_GPU_TESTS` in the script); their `_overlap` variants run with the tensor-parallel
//!   prefill overlap on over `hostmem` (P5 Task 34: the split sits at a KV block boundary).
//! - `nvme_round_trip_matches_cold`: with 1 GiB of L1 (L0 demotes by capacity past 70 %)
//!   and the NVMe tier at `/home/piwi/turbine-kv`, prompt A runs cold, long filler prompts push
//!   its blocks through L1 to L2 (L0 → L2 on unified memory), then A runs again: the same greedy
//!   answer, promotions from L2, no checksum eviction (every promoted block's CRC32C matched the
//!   value written), and at most 64 GiB of slab files under the path.
//!
//! - `prefix_reuse_matches_cold_fp8_kv`, `nvme_round_trip_matches_cold_fp8_kv` (Phase 6a S-14):
//!   the same two checks at `kv.dtype: fp8_e4m3` — FP8 pages are copied through L1/L2 as their
//!   bytes, so a warm or round-tripped prefix answers exactly like the cold FP8 run — plus the
//!   FP8 L0 pool itself: F8E4M3 pages of half the bytes and at least 1.95× the BF16 run's blocks
//!   in the same budget.
//!
//! Answers are compared by their text and completion token count: greedy decoding of the same
//! token ids gives the same text, and any KV difference shows up within 64 tokens as a
//! different continuation (the Phase 1 golden run showed a single BF16 rounding flip changes
//! the text). The always-run test checks the lab config itself.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use turbine_kernels::test_support::{require_backend, require_env_dir};

const SERVED_NAME: &str = "meta-llama/Llama-3.2-3B-Instruct";
/// Weights load, calibration and warm-up on the R9700; a debug build is slower than release.
const READY_LIMIT: Duration = Duration::from_secs(900);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_millis(200);
/// The lab NVMe tier and its cap (P4 Constraints).
const KV_DIR: &str = "/home/piwi/turbine-kv";
const KV_CAP_BYTES: u64 = 64 << 30;
/// How long the NVMe round trip retries a prefetch refused for L0 pressure.
const PREFETCH_WAIT: Duration = Duration::from_secs(60);
/// Greedy tokens per answer.
const ANSWER_TOKENS: u32 = 64;

/// The lab tests share one GPU: two servers loading side by side would each budget the whole
/// card and push the device-memory signal into SURVIVAL, so they run one after the other.
fn one_server_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lab_config() -> PathBuf {
    repo_root().join("scripts/lab/phase4-novanas.yaml")
}

fn free_addr() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

/// `turbine-server` on the lab config with `--set` overrides; killed on drop.
struct LabServer {
    child: Child,
    addr: SocketAddr,
}

impl LabServer {
    fn start(model_dir: &Path, sets: &[String]) -> LabServer {
        let addr = free_addr();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_turbine-server"));
        cmd.arg("--config").arg(lab_config());
        for set in [
            format!("server.listen={addr}"),
            format!("model.path={}", model_dir.display()),
        ]
        .iter()
        .chain(sets)
        {
            cmd.arg("--set").arg(set);
        }
        // The kernel library comes from TURBINE_KERNEL_LIBRARY (set by the lab Job); the log
        // goes straight to the test output.
        let child = cmd
            .env("RUST_LOG", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn turbine-server");
        let mut server = LabServer { child, addr };
        let started = Instant::now();
        while started.elapsed() < READY_LIMIT {
            if let Some(status) = server.child.try_wait().unwrap() {
                panic!("turbine-server exited before /ready ({status}); see its log above");
            }
            if TcpStream::connect(addr).is_ok() && request(addr, "GET", "/ready", None).0 == 200 {
                return server;
            }
            std::thread::sleep(POLL);
        }
        panic!("turbine-server not ready within {READY_LIMIT:?}");
    }

    /// One greedy completion: (text, completion tokens, cached prompt tokens).
    fn complete(&self, prompt: &str, max_tokens: u32) -> (String, u64, u64) {
        self.complete_in(prompt, max_tokens, None)
    }

    /// [`LabServer::complete`] as a turn of session `session` (`prompt_cache_key`): its blocks
    /// carry the reuse evidence that makes them worth demoting to L1/L2.
    fn complete_in(
        &self,
        prompt: &str,
        max_tokens: u32,
        session: Option<&str>,
    ) -> (String, u64, u64) {
        let mut body = json!({
            "model": SERVED_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": true,
        });
        if let Some(key) = session {
            body["prompt_cache_key"] = json!(key);
        }
        let (status, text) = request(
            self.addr,
            "POST",
            "/v1/completions",
            Some(&body.to_string()),
        );
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).expect("JSON response");
        let usage = &v["usage"];
        (
            v["choices"][0]["text"].as_str().expect("text").to_string(),
            usage["completion_tokens"]
                .as_u64()
                .expect("completion_tokens"),
            usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        )
    }

    /// One greedy completion of the token-id prompt `ids` with each token's logprob.
    fn complete_ids(&self, ids: &[u32], max_tokens: u32) -> IdAnswer {
        self.complete_with_logprobs(json!(ids), max_tokens, None)
    }

    /// One greedy completion of `prompt` (text or token ids, as the request's JSON) with each
    /// token's logprob, as a turn of session `session` when given.
    fn complete_with_logprobs(
        &self,
        prompt: Value,
        max_tokens: u32,
        session: Option<&str>,
    ) -> IdAnswer {
        self.complete_with_headers(prompt, max_tokens, session, &[])
    }

    /// [`LabServer::complete_with_logprobs`] with extra request headers
    /// (`x-turbine-kv-lossy: deny`).
    fn complete_with_headers(
        &self,
        prompt: Value,
        max_tokens: u32,
        session: Option<&str>,
        headers: &[(&str, &str)],
    ) -> IdAnswer {
        let mut body = json!({
            "model": SERVED_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": true,
            "logprobs": 1,
        });
        if let Some(key) = session {
            body["prompt_cache_key"] = json!(key);
        }
        let (status, text) = request_with(
            self.addr,
            "POST",
            "/v1/completions",
            Some(&body.to_string()),
            headers,
        );
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).expect("JSON response");
        let usage = &v["usage"];
        let choice = &v["choices"][0];
        IdAnswer {
            text: choice["text"].as_str().expect("text").to_string(),
            completion_tokens: usage["completion_tokens"]
                .as_u64()
                .expect("completion_tokens"),
            logprobs: choice["logprobs"]["token_logprobs"]
                .as_array()
                .expect("token_logprobs")
                .iter()
                .map(|l| l.as_f64().expect("logprob"))
                .collect(),
            prompt_tokens: usage["prompt_tokens"].as_u64().expect("prompt_tokens"),
            cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
            lossy_cached_tokens: usage["prompt_tokens_details"]["lossy_cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        }
    }

    /// The value of one Prometheus series (0 when absent).
    fn metric(&self, series: &str) -> f64 {
        let (status, text) = request(self.addr, "GET", "/metrics", None);
        assert_eq!(status, 200);
        text.lines()
            .find_map(|l| l.strip_prefix(series)?.trim().parse().ok())
            .unwrap_or(0.0)
    }
}

/// [`LabServer::complete_ids`]'s answer.
#[derive(Debug)]
struct IdAnswer {
    text: String,
    completion_tokens: u64,
    /// Each generated token's logprob, as the server printed it (equal bits print alike).
    logprobs: Vec<f64>,
    prompt_tokens: u64,
    cached_tokens: u64,
    /// Of the prompt tokens reused, those whose blocks went through a lossy tier (P6b S-3);
    /// `cached_tokens` counts the exact ones.
    lossy_cached_tokens: u64,
}

impl IdAnswer {
    /// What must not change between a cold and a warm run.
    fn output(&self) -> (&str, u64, &[f64]) {
        (&self.text, self.completion_tokens, &self.logprobs)
    }
}

impl Drop for LabServer {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

/// Sends one request with `Connection: close`; returns the status and the body.
fn request(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    request_with(addr, method, path, body, &[])
}

/// [`request`] with extra request headers.
fn request_with(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
    headers: &[(&str, &str)],
) -> (u16, String) {
    let mut conn = TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(REQUEST_TIMEOUT)).unwrap();
    let body = body.unwrap_or("");
    let extra: String = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n{extra}\
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
    let mut out = String::new();
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
            out.push_str(&String::from_utf8(data).expect("UTF-8 chunk"));
        }
    } else {
        reader.read_to_string(&mut out).expect("read body");
    }
    (status, out)
}

/// A prompt of about `words` words, distinct per `seed` from its first word on (so two seeds
/// share no prefix block).
fn prompt(seed: u32, words: u32) -> String {
    const VOCAB: [&str; 16] = [
        "river", "copper", "lantern", "orbit", "meadow", "signal", "harbor", "ember", "quartz",
        "willow", "summit", "canyon", "velvet", "thunder", "atlas", "prism",
    ];
    let mut text = format!("Document {seed}. Summarize the following notes.\n");
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    for i in 0..words {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        text.push_str(VOCAB[(x % 16) as usize]);
        text.push_str(if i % 12 == 11 { ". " } else { " " });
    }
    text
}

/// `len` token ids of ordinary (non-special) Llama-3 vocabulary, distinct per `seed`.
fn token_ids(seed: u32, len: u32) -> Vec<u32> {
    let mut x = seed.wrapping_mul(2_654_435_761) ^ 0x9E37_79B9;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            1_000 + x % 120_000
        })
        .collect()
}

/// Total bytes of the `turbine-kv-*.slab` files under `dir`.
fn slab_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("turbine-kv-") && n.ends_with(".slab"))
                })
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn prefix_reuse_matches_cold() {
    if !require_backend("hip") {
        return;
    }
    prefix_reuse_matches_cold_with(&[]);
}

/// [`prefix_reuse_matches_cold`] at tensor parallelism 2 over both R9700s: every rank's
/// projections have half-size shapes, which need their own invariant rows in the tuned GEMM
/// table (decision 2026-09-28, "P5: bit-exact prefix reuse under tensor parallelism", B).
#[test]
#[ignore = "lab, 2 GPUs (scripts/lab-test.sh novanas --gpus 2): HIP, libturbine_hip.so, Llama-3.2-3B"]
fn prefix_reuse_matches_cold_tp2() {
    if !require_backend("hip") {
        return;
    }
    prefix_reuse_matches_cold_with(&tp2_sets());
}

/// [`prefix_reuse_matches_cold_tp2`] with the tensor-parallel prefill overlap on, over
/// `hostmem` as served (P5 Task 34): its prompts of about 400 tokens split, at a KV block
/// boundary, so the cold run's second half starts where the warm run's suffix does.
#[test]
#[ignore = "lab, 2 GPUs (scripts/lab-test.sh novanas --gpus 2): HIP, libturbine_hip.so, Llama-3.2-3B"]
fn prefix_reuse_matches_cold_tp2_overlap() {
    if !require_backend("hip") {
        return;
    }
    prefix_reuse_matches_cold_with(&tp2_overlap_sets());
}

/// [`prefix_reuse_suffix_lengths_match_cold_tp2`] with the prefill overlap on (see
/// [`prefix_reuse_matches_cold_tp2_overlap`]); the 700-token suffix splits in the warm run too.
/// Before Task 34 (the middle-row split) this failed.
#[test]
#[ignore = "lab, 2 GPUs (scripts/lab-test.sh novanas --gpus 2): HIP, libturbine_hip.so, Llama-3.2-3B"]
fn prefix_reuse_suffix_lengths_match_cold_tp2_overlap() {
    if !require_backend("hip") {
        return;
    }
    prefix_reuse_suffix_lengths_match_cold_with(&tp2_overlap_sets());
}

/// [`tp2_sets`] over `hostmem` with `parallel.tp_prefill_overlap` on.
fn tp2_overlap_sets() -> Vec<String> {
    tp2_sets()
        .into_iter()
        .map(|s| {
            if s == "parallel.collective_backend=rccl" {
                "parallel.collective_backend=hostmem".to_string()
            } else {
                s
            }
        })
        .chain(["parallel.tp_prefill_overlap=true".to_string()])
        .collect()
}

/// The `--set` overrides of the tp 2 variants: both R9700s as one tensor-parallel group over
/// RCCL (local rank mode). The L1/L2 tiers are off: these tests are about the prefill rows of a
/// warm suffix, which reuses L0 blocks only.
fn tp2_sets() -> Vec<String> {
    [
        "parallel.tensor_parallel_size=2",
        "parallel.devices=[0,1]",
        "parallel.collective_backend=rccl",
        "kv.cpu.enabled=false",
        "kv.nvme.enabled=false",
    ]
    .map(String::from)
    .to_vec()
}

/// Asserts the server runs the tensor parallelism `sets` asks for (1 when they name none).
fn assert_tp(server: &LabServer, sets: &[String]) {
    let want = if sets.iter().any(|s| s == "parallel.tensor_parallel_size=2") {
        2
    } else {
        1
    };
    let (status, text) = request(server.addr, "GET", "/turbine/v1/status", None);
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).expect("status JSON");
    let tp = v["parallel"]["tp"].as_u64().unwrap_or(1);
    assert_eq!(tp, want, "tensor parallelism: {}", v["parallel"]);
}

/// [`prefix_reuse_matches_cold`] on servers started with the extra overrides `sets`.
fn prefix_reuse_matches_cold_with(sets: &[String]) {
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    // About 400 tokens: three full 128-token blocks each.
    let prompts: Vec<String> = (1..=5).map(|s| prompt(s, 350)).collect();

    let cold_sets = [sets, &["kv.prefix_sharing=false".to_string()]].concat();
    let cold: Vec<(String, u64, u64)> = {
        let server = LabServer::start(&model_dir, &cold_sets);
        assert_tp(&server, sets);
        prompts
            .iter()
            .map(|p| server.complete(p, ANSWER_TOKENS))
            .collect()
    };
    let server = LabServer::start(&model_dir, sets);
    assert_tp(&server, sets);
    for (i, p) in prompts.iter().enumerate() {
        let first = server.complete(p, ANSWER_TOKENS);
        let warm = server.complete(p, ANSWER_TOKENS);
        println!(
            "prompt {i}: cold {:?} first cached {} warm cached {}",
            cold[i].0, first.2, warm.2
        );
        assert_eq!(cold[i].1, u64::from(ANSWER_TOKENS));
        assert_eq!(
            (&first.0, first.1),
            (&cold[i].0, cold[i].1),
            "prompt {i}: first run"
        );
        assert_eq!(
            (&warm.0, warm.1),
            (&cold[i].0, cold[i].1),
            "prompt {i}: warm run"
        );
        assert!(warm.2 > 0, "prompt {i}: the warm run reused no prefix");
    }
    println!("prefix_reuse_matches_cold ok");
}

/// The FP8 KV overrides (Phase 6a S-13).
fn fp8_kv_sets() -> Vec<String> {
    vec!["kv.dtype=fp8_e4m3".to_string()]
}

/// The L0 tier of `GET /turbine/v1/kv`: (page dtype, block bytes, total blocks).
fn l0_tier(server: &LabServer) -> (String, u64, u64) {
    let (status, text) = request(server.addr, "GET", "/turbine/v1/kv", None);
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).expect("KV document JSON");
    let l0 = v["tiers"]
        .as_array()
        .and_then(|t| t.iter().find(|t| t["tier"] == "l0"))
        .unwrap_or_else(|| panic!("no l0 tier: {text}"));
    (
        l0["dtype"].as_str().unwrap_or_default().to_string(),
        l0["block_bytes"].as_u64().unwrap_or(0),
        l0["blocks_total"].as_u64().unwrap_or(0),
    )
}

/// Phase 6a S-13 / S-14: at `kv.dtype: fp8_e4m3` the L0 pool is F8E4M3 with half the block
/// bytes and at least 1.95× the blocks of the BF16 run (same budget), and prefix reuse gives the
/// cold FP8 answers exactly (see [`prefix_reuse_matches_cold`]). Breaks if the FP8 pool is not
/// smaller per block, or a reused FP8 prefix differs from recomputing it.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn prefix_reuse_matches_cold_fp8_kv() {
    if !require_backend("hip") {
        return;
    }
    {
        let _gpu = one_server_at_a_time();
        let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
        let bf16 = l0_tier(&LabServer::start(&model_dir, &[]));
        let fp8 = l0_tier(&LabServer::start(&model_dir, &fp8_kv_sets()));
        let ratio = fp8.2 as f64 / bf16.2 as f64;
        println!("L0 bf16 {bf16:?} fp8 {fp8:?}: {ratio:.3}x the blocks");
        assert_eq!((bf16.0.as_str(), fp8.0.as_str()), ("bf16", "f8e4m3"));
        assert_eq!(2 * fp8.1, bf16.1, "FP8 halves a block");
        assert!(
            ratio >= 1.95,
            "FP8 KV holds only {ratio:.3}x the BF16 blocks"
        );
    }
    prefix_reuse_matches_cold_with(&fp8_kv_sets());
}

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn prefix_reuse_suffix_lengths_match_cold() {
    if !require_backend("hip") {
        return;
    }
    prefix_reuse_suffix_lengths_match_cold_with(&[]);
}

/// [`prefix_reuse_suffix_lengths_match_cold`] at tensor parallelism 2 over both R9700s (see
/// [`prefix_reuse_matches_cold_tp2`]).
#[test]
#[ignore = "lab, 2 GPUs (scripts/lab-test.sh novanas --gpus 2): HIP, libturbine_hip.so, Llama-3.2-3B"]
fn prefix_reuse_suffix_lengths_match_cold_tp2() {
    if !require_backend("hip") {
        return;
    }
    prefix_reuse_suffix_lengths_match_cold_with(&tp2_sets());
}

/// [`prefix_reuse_suffix_lengths_match_cold`] on servers started with the extra overrides
/// `sets`.
fn prefix_reuse_suffix_lengths_match_cold_with(sets: &[String]) {
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    const BLOCK: u32 = 128;
    // (whole blocks a primer shares with the target, uncached suffix tokens of the target)
    const CASES: [(u32, u32); 7] = [
        (3, 1),
        (3, 2),
        (3, 17),
        (3, 64),
        (2, 65),
        (2, 100),
        (1, 700),
    ];
    let prompts: Vec<(Vec<u32>, Vec<u32>)> = CASES
        .iter()
        .zip(1u32..)
        .map(|(&(blocks, suffix), i)| {
            let shared = token_ids(100 + i, blocks * BLOCK);
            let primer = [shared.as_slice(), &token_ids(200 + i, 5)].concat();
            let target = [shared.as_slice(), &token_ids(300 + i, suffix)].concat();
            (primer, target)
        })
        .collect();

    let cold_sets = [sets, &["kv.prefix_sharing=false".to_string()]].concat();
    let cold: Vec<IdAnswer> = {
        let server = LabServer::start(&model_dir, &cold_sets);
        assert_tp(&server, sets);
        prompts
            .iter()
            .map(|(_, target)| server.complete_ids(target, ANSWER_TOKENS))
            .collect()
    };
    let server = LabServer::start(&model_dir, sets);
    assert_tp(&server, sets);
    let mut failed = Vec::new();
    for (((primer, target), &(blocks, suffix)), cold) in prompts.iter().zip(&CASES).zip(&cold) {
        server.complete_ids(primer, ANSWER_TOKENS);
        let warm = server.complete_ids(target, ANSWER_TOKENS);
        println!(
            "blocks {blocks} suffix {suffix}: prompt {} cached {} cold {:?} warm {:?}",
            warm.prompt_tokens, warm.cached_tokens, cold.text, warm.text
        );
        assert_eq!(cold.completion_tokens, u64::from(ANSWER_TOKENS));
        // At least two prompt tokens are prefilled (turbine_kv::planner::MIN_RECOMPUTE_TOKENS).
        let reused = blocks.min((blocks * BLOCK + suffix - 2) / BLOCK);
        assert_eq!(
            warm.cached_tokens,
            u64::from(reused * BLOCK),
            "blocks {blocks} suffix {suffix}: the warm run reuses the shared blocks"
        );
        assert_eq!(warm.prompt_tokens, u64::from(blocks * BLOCK + suffix));
        if warm.output() != cold.output() {
            let first = cold
                .logprobs
                .iter()
                .zip(&warm.logprobs)
                .position(|(c, w)| c.to_bits() != w.to_bits());
            let at = |l: &[f64]| first.and_then(|i| l.get(i).copied());
            println!(
                "blocks {blocks} suffix {suffix}: warm differs from cold at token {first:?} \
                 (logprob cold {:?} warm {:?})",
                at(&cold.logprobs),
                at(&warm.logprobs)
            );
            failed.push(format!("blocks {blocks} suffix {suffix}"));
        }
    }
    assert!(failed.is_empty(), "warm differs from cold: {failed:?}");
    println!("prefix_reuse_suffix_lengths_match_cold ok");
}

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so, the Llama-3.2-3B weights and /home/piwi/turbine-kv"]
fn nvme_round_trip_matches_cold() {
    if !require_backend("hip") {
        return;
    }
    nvme_round_trip_matches_cold_with(&[]);
}

/// [`nvme_round_trip_matches_cold`] at `kv.dtype: fp8_e4m3` (Phase 6a S-14): the FP8 pages go to
/// L2 and back as their bytes (checksummed), so A answers exactly as its cold FP8 run.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so, the Llama-3.2-3B weights and /home/piwi/turbine-kv"]
fn nvme_round_trip_matches_cold_fp8_kv() {
    if !require_backend("hip") {
        return;
    }
    nvme_round_trip_matches_cold_with(&fp8_kv_sets());
}

/// The L2 tests' server: L2 under [`KV_DIR`] and a 1 GiB L1 (L0 stays the lab config's 8 GiB:
/// a smaller cap shrinks Phase 3's device budget below what the HIP runtime and libraries
/// already hold, and the device_memory signal would put the server in SURVIVAL; cached blocks
/// are demoted by capacity past 70 % of L0, so filler prompts still push A down through L1),
/// with the extra overrides `sets`. Returns it with the blocks 1 GiB of L1 holds (73 BF16
/// Llama blocks of 14,680,064 bytes, 146 FP8).
fn nvme_server(sets: &[String]) -> (LabServer, f64) {
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let kv_dir = Path::new(KV_DIR);
    assert!(kv_dir.is_dir(), "{KV_DIR} is not mounted");
    let tiers = [
        format!("kv.nvme.path={KV_DIR}"),
        "kv.nvme.max_bytes=64GiB".into(),
        "kv.cpu.max_bytes=1GiB".into(),
    ];
    let server = LabServer::start(&model_dir, &[&tiers[..], sets].concat());
    let (_, block_bytes, _) = l0_tier(&server);
    let l1_blocks = ((1u64 << 30) / block_bytes.max(1)) as f64;
    (server, l1_blocks)
}

/// Blocks that have gone down to L2 (from L1 or straight from L0).
fn to_l2(server: &LabServer) -> f64 {
    server.metric(r#"turbine_kv_demotions_total{from="l1",to="l2"}"#)
        + server.metric(r#"turbine_kv_demotions_total{from="l0",to="l2"}"#)
}

/// Blocks promoted out of L2.
fn from_l2(server: &LabServer) -> f64 {
    server.metric(r#"turbine_kv_promotions_total{from="l2",to="l0"}"#)
        + server.metric(r#"turbine_kv_promotions_total{from="l2",to="l1"}"#)
}

/// Filler session turns until `blocks` blocks have gone down to L2: A's (the oldest, never
/// touched again) are among them once that is twice L1's blocks plus A's own.
fn fill_until_l2(server: &LabServer, blocks: f64) {
    let mut filler = 0;
    while to_l2(server) < blocks {
        filler += 1;
        assert!(
            filler <= 400,
            "demotion to L2 never happened ({} blocks)",
            to_l2(server)
        );
        let key = format!("filler-{filler}");
        server.complete_in(&prompt(1_000 + filler, 1_800), 1, Some(&key));
    }
    println!(
        "{filler} filler prompts moved {} blocks to L2",
        to_l2(server)
    );
}

/// Waits until the demotions the fillers left in flight have drained L0: its used blocks are
/// unchanged over 1.5 s. A prefetch issued while L0 is draining (above its demotion threshold,
/// the lab runs stopped at 87 %) has its promoted blocks demoted again within a second, as the
/// lowest-value cached blocks, before any request can use them.
fn wait_l0_settled(server: &LabServer) {
    const SERIES: &str = r#"turbine_kv_blocks{tier="l0",state="used"}"#;
    let started = Instant::now();
    let mut last = server.metric(SERIES);
    let mut stable = 0;
    while stable < 5 {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "L0 never settled ({last} blocks used)"
        );
        std::thread::sleep(Duration::from_millis(300));
        let now = server.metric(SERIES);
        stable = if now == last { stable + 1 } else { 0 };
        last = now;
    }
    println!(
        "L0 settled at {last} blocks used after {:?}",
        started.elapsed()
    );
}

/// Brings A back explicitly (`POST /turbine/v1/kv/prefetch`): on the R9700 the planner may
/// rightly find recomputing a few blocks cheaper than reading them from NVMe, and these tests
/// are about the bytes surviving the round trip, not about that choice. Returns once a block
/// was promoted out of L2.
fn prefetch_a_from_l2(server: &LabServer, a: &str) {
    wait_l0_settled(server);
    let body = json!({ "prompt": a }).to_string();
    // The fillers leave L0 near its demotion threshold: a prefetch is refused (409
    // `pressure_too_high`) while L0 pressure is ORANGE or above, until the demotions in flight
    // complete, so it is retried for a bounded time.
    let asked = Instant::now();
    let (status, text) = loop {
        let (status, text) = request(server.addr, "POST", "/turbine/v1/kv/prefetch", Some(&body));
        if status != 409 || !text.contains("pressure_too_high") || asked.elapsed() > PREFETCH_WAIT {
            break (status, text);
        }
        std::thread::sleep(POLL);
    };
    assert_eq!(status, 202, "{text}");
    println!("prefetch of A: {text}");
    let started = Instant::now();
    while from_l2(server) < 1.0 {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the prefetch never promoted from L2: {text}"
        );
        std::thread::sleep(POLL);
    }
    std::thread::sleep(Duration::from_secs(2));
}

/// Prints the KV document and the KV metric series (what a failed reuse assertion needs).
fn dump_kv(server: &LabServer) {
    let (_, doc) = request(server.addr, "GET", "/turbine/v1/kv", None);
    println!("kv document: {doc}");
    let (_, metrics) = request(server.addr, "GET", "/metrics", None);
    for line in metrics.lines().filter(|l| l.starts_with("turbine_kv_")) {
        println!("  {line}");
    }
}

/// [`nvme_round_trip_matches_cold`] on a server started with the extra overrides `sets`.
fn nvme_round_trip_matches_cold_with(sets: &[String]) {
    let _gpu = one_server_at_a_time();
    let (server, l1_blocks) = nvme_server(sets);
    let a = prompt(100, 350);
    // A and the fillers are session turns: blocks of one-off requests are never copied down.
    let cold = server.complete_in(&a, ANSWER_TOKENS, Some("a"));
    fill_until_l2(&server, 2.0 * l1_blocks);
    prefetch_a_from_l2(&server, &a);

    let warm = server.complete(&a, ANSWER_TOKENS);
    if warm.2 == 0 {
        dump_kv(&server);
    }
    assert_eq!(
        (&warm.0, warm.1),
        (&cold.0, cold.1),
        "A after its L2 round trip"
    );
    assert!(warm.2 > 0, "A reused no prefix");
    let promoted = from_l2(&server);
    assert!(promoted > 0.0, "nothing was promoted from L2");
    assert_eq!(
        server.metric(r#"turbine_kv_evictions_total{tier="l2",reason="checksum"}"#),
        0.0,
        "a promoted block's bytes differed from what was written"
    );
    let slabs = slab_bytes(Path::new(KV_DIR));
    assert!(slabs <= KV_CAP_BYTES, "{slabs} bytes of slab files");
    println!("nvme_round_trip_matches_cold ok: {promoted} promotions from L2, {slabs} slab bytes");
}

/// P6b S-1 (ABI v2.11 KV transcode, FP8 on the demotion path): with `kv.nvme.format:
/// fp8_e4m3` below the BF16 pool, blocks go to L2 encoded on the device (a block of L2 holds
/// well under the BF16 block bytes: the last block of each sequence stays at `l0`,
/// `kv.lossless_tail_blocks`), are promoted out of L2 and decoded into the pages by the kernel
/// (`POST /turbine/v1/kv/prefetch`) without a copy error or a checksum failure, and A still
/// answers within the codec's bound of its cold run: the first tokens' logprobs within 0.3 and
/// at least 90 % of all positions within 0.5 (the FP8 KV golden's bounds are 0.15 / 0.55).
/// The prefetched, decoded lossy copies are then the reused prefix (`cached_tokens` and
/// `lossy_cached_tokens` > 0: the lookup takes the promoted L0 copy over the L2 one). The `l0`
/// default stays bit-exact
/// (`nvme_round_trip_matches_cold`).
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so, the Llama-3.2-3B weights and /home/piwi/turbine-kv"]
fn nvme_round_trip_fp8_tier() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let (server, l1_blocks) = nvme_server(&["kv.nvme.format=fp8_e4m3".to_string()]);
    let (_, block_bytes, _) = l0_tier(&server);
    let a = prompt(100, 350);
    let cold = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, Some("a"));
    fill_until_l2(&server, 2.0 * l1_blocks);
    let stored = to_l2(&server);
    let used = server.metric(r#"turbine_kv_bytes{tier="l2",kind="used"}"#);
    println!("L2 holds {used} bytes for {stored} demoted blocks of {block_bytes} bytes");
    assert!(
        used < 0.75 * stored * block_bytes as f64,
        "L2 holds {used} bytes for {stored} blocks of {block_bytes}: not encoded"
    );
    prefetch_a_from_l2(&server, &a);

    let warm = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, None);
    if warm.cached_tokens == 0 || warm.lossy_cached_tokens == 0 {
        dump_kv(&server);
    }
    assert_within_codec_bound(&cold, &warm, "fp8 tier");
    // The decoded lossy copies are the reused prefix, not recomputed (the lookup takes the
    // promoted L0 copy over the L2 one): the lossless tail block comes back exact.
    assert!(warm.cached_tokens > 0, "A reused no prefix");
    assert!(
        warm.lossy_cached_tokens > 0,
        "no reused block was served from a lossy copy: {warm:?}"
    );
    for series in [
        r#"turbine_kv_evictions_total{tier="l2",reason="checksum"}"#,
        r#"turbine_kv_evictions_total{tier="l2",reason="tier_degraded"}"#,
        r#"turbine_kv_evictions_total{tier="l1",reason="tier_degraded"}"#,
    ] {
        assert_eq!(server.metric(series), 0.0, "{series}");
    }
    assert!(from_l2(&server) > 0.0, "nothing was promoted from L2");
    println!(
        "nvme_round_trip_fp8_tier ok: {} promotions from L2",
        from_l2(&server)
    );
}

/// A lossy-tier answer against the cold run (the FP8 tier's bound): the same completion length,
/// the first 8 tokens' logprobs within 0.3 and at least 90 % of all positions within 0.5 (the
/// FP8 KV golden's bounds are 0.15 / 0.55).
fn assert_within_codec_bound(cold: &IdAnswer, warm: &IdAnswer, label: &str) {
    assert_within_bounds(cold, warm, label, (0.3, 0.5, 0.9));
}

/// [`assert_within_codec_bound`] with explicit bounds: (first 8 tokens' worst |Δ|, per-position
/// |Δ| limit, share of positions within it).
fn assert_within_bounds(cold: &IdAnswer, warm: &IdAnswer, label: &str, bounds: (f64, f64, f64)) {
    let (head_bound, pos_bound, share) = bounds;
    assert_eq!(warm.completion_tokens, cold.completion_tokens);
    let diff: Vec<f64> = cold
        .logprobs
        .iter()
        .zip(&warm.logprobs)
        .map(|(c, w)| (c - w).abs())
        .collect();
    let head = diff.iter().take(8).cloned().fold(0.0, f64::max);
    let within = diff.iter().filter(|d| **d <= pos_bound).count() as f64 / diff.len().max(1) as f64;
    println!(
        "{label}: worst of the first 8 |Δ logprob| {head:.3}, {within:.2} within 0.5 \
         (cached {}, lossy cached {})",
        warm.cached_tokens, warm.lossy_cached_tokens
    );
    assert!(head <= head_bound, "first tokens differ by {head}");
    assert!(
        within >= share,
        "only {within} of the positions are within {pos_bound}"
    );
}

/// P6b S-2 / S-3 / S-8 (plan Task 6): with `kv.cpu.format: fp8_e4m3` below the BF16 pool, blocks
/// that capacity demotion moved to L1 are encoded; a later request for the same prompt reuses
/// them as lossy blocks (`usage.prompt_tokens_details.lossy_cached_tokens` > 0) and answers
/// within the FP8 tier's bound of its cold run; with `x-turbine-kv-lossy: deny` the lossy
/// blocks are not reused and the answer equals the cold one bit for bit. Breaks if a lossy block
/// is reused by a request that opted out, or a lossy L1 copy is never reused.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn lossy_tier_reuse() {
    lossy_tier_reuse_with("fp8_e4m3", Some((0.3, 0.5, 0.9)));
}

/// [`lossy_tier_reuse`] with L1 at TurboQuant 4-bit (P6b Task 8: the tables are uploaded and the
/// transcode runs `turbine_hip_tq`). No quality bound: the mechanism is asserted (device transcode, lossy reuse, no checksum eviction); Task 9 sets the S-8 gate.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn lossy_tier_reuse_tq4() {
    lossy_tier_reuse_with("tq4", None);
}

/// P6b Task 8 smoke: `kv.dtype: tq4` serves on the GPU (tables uploaded to the executor before
/// the first forward, the mixed-format attention reads them): a request answers its tokens with
/// finite logprobs, and the L0 tier reports TurboQuant pages smaller than BF16's. Task 13 owns
/// the quality proof.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn tq_kv_serves_on_the_device() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let server = LabServer::start(&model_dir, &["kv.dtype=tq4".to_string()]);
    let (dtype, block_bytes, blocks) = l0_tier(&server);
    println!("L0 pages {dtype}: {block_bytes} bytes a block, {blocks} blocks");
    assert!(
        dtype.contains("tq4") && block_bytes < 14_680_064,
        "{dtype} {block_bytes}"
    );
    let a = prompt(100, 350);
    let answer = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, None);
    assert_eq!(answer.completion_tokens, ANSWER_TOKENS as u64);
    assert!(!answer.text.is_empty());
    assert!(answer.logprobs.iter().all(|l| l.is_finite()), "{answer:?}");
    println!("tq_kv_serves_on_the_device ok: {:?}", answer.text);
}

fn lossy_tier_reuse_with(format: &str, bounds: Option<(f64, f64, f64)>) {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let server = LabServer::start(
        &model_dir,
        &[
            format!("kv.cpu.format={format}"),
            "kv.cpu.max_bytes=2GiB".to_string(),
            "kv.nvme.enabled=false".to_string(),
        ],
    );
    let a = prompt(100, 350);
    let cold = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, Some("a"));
    assert_eq!(cold.lossy_cached_tokens, 0);
    // Filler session turns until capacity demotion has sent well over A's blocks to L1 (A is the
    // oldest session, never touched again).
    let to_l1 = |s: &LabServer| s.metric(r#"turbine_kv_demotions_total{from="l0",to="l1"}"#);
    let mut filler = 0;
    while to_l1(&server) < 40.0 {
        filler += 1;
        assert!(filler <= 400, "demotion to L1 never happened");
        let key = format!("filler-{filler}");
        server.complete_in(&prompt(1_000 + filler, 1_800), 1, Some(&key));
    }
    println!(
        "{filler} filler prompts moved {} blocks to L1",
        to_l1(&server)
    );
    let used = server.metric(r#"turbine_kv_bytes{tier="l1",kind="used"}"#);
    let (_, block_bytes, _) = l0_tier(&server);
    println!(
        "L1 holds {used} bytes for {} demoted blocks of {block_bytes} bytes",
        to_l1(&server)
    );
    assert!(
        used < 0.75 * to_l1(&server) * block_bytes as f64,
        "L1 holds {used} bytes: not encoded"
    );

    let warm = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, None);
    if warm.lossy_cached_tokens == 0 {
        dump_kv(&server);
    }
    assert!(
        warm.lossy_cached_tokens > 0,
        "no reused block was served from a lossy copy: {warm:?}"
    );
    match bounds {
        Some(b) => assert_within_bounds(&cold, &warm, "lossy L1 reuse", b),
        // The codec's own loss is Task 9's gate: print the spread, assert the mechanism.
        None => {
            assert_eq!(warm.completion_tokens, cold.completion_tokens);
            let worst = cold
                .logprobs
                .iter()
                .zip(&warm.logprobs)
                .map(|(c, w)| (c - w).abs())
                .fold(0.0, f64::max);
            println!(
                "{format} lossy L1 reuse: worst |Δ logprob| {worst:.3} (no bound until Task 9)"
            );
        }
    }
    assert_eq!(
        server.metric(r#"turbine_kv_evictions_total{tier="l1",reason="checksum"}"#),
        0.0
    );

    let denied = server.complete_with_headers(
        json!(a),
        ANSWER_TOKENS,
        None,
        &[("x-turbine-kv-lossy", "deny")],
    );
    println!(
        "deny: cached {} lossy cached {}",
        denied.cached_tokens, denied.lossy_cached_tokens
    );
    assert_eq!(
        denied.lossy_cached_tokens, 0,
        "a denied request reused lossy"
    );
    assert_eq!(
        denied.output(),
        cold.output(),
        "x-turbine-kv-lossy: deny must equal the cold run bit for bit"
    );
    assert!(
        server.metric("turbine_kv_lossy_denied_total") >= 1.0,
        "the denial is counted"
    );
    println!(
        "lossy_tier_reuse ok: lossy cached {} of {} prompt tokens",
        warm.lossy_cached_tokens, warm.prompt_tokens
    );
}

/// The lab config loads and spells out the tiers the lab tests rely on.
#[test]
fn phase4_lab_config_loads() {
    let c = turbine_core::config::load(&lab_config(), &[]).expect("phase4-novanas.yaml loads");
    assert_eq!(c.kv.block_tokens, 128);
    assert!(c.kv.cpu.enabled && c.kv.nvme.enabled);
    assert_eq!(c.kv.nvme.path, Path::new(KV_DIR));
    assert_eq!(c.kv.nvme.max_bytes.0, KV_CAP_BYTES);
    assert!(c.kv.prefix_sharing);
    let p = prompt(1, 350);
    assert!(p.split_whitespace().count() > 350);
    assert_ne!(prompt(1, 20)[..20], prompt(2, 20)[..20]);
}

/// The FP8 KV proof configs (plan Task 24) load and differ from their Phase 2c BF16 KV
/// comparison only in `kv.dtype`.
#[test]
fn phase6_fp8kv_lab_configs_load() {
    use turbine_core::config::KvDtypeChoice;
    for model in ["llama", "olmoe"] {
        let lab = repo_root().join("scripts/lab");
        let fp8 = turbine_core::config::load(
            &lab.join(format!("phase6-novanas-{model}-fp8kv.yaml")),
            &[],
        )
        .expect("the FP8 KV config loads");
        let bf16 =
            turbine_core::config::load(&lab.join(format!("phase2c-novanas-{model}.yaml")), &[])
                .expect("the Phase 2c config loads");
        assert_eq!(fp8.kv.dtype, KvDtypeChoice::Fp8E4m3, "{model}");
        assert_eq!(bf16.kv.dtype, KvDtypeChoice::Bf16, "{model}");
        assert_eq!(fp8.model.path, bf16.model.path, "{model}");
        assert_eq!(
            fp8.scheduler.max_batch_tokens, bf16.scheduler.max_batch_tokens,
            "{model}"
        );
        assert_eq!(fp8.kv.gpu.max_bytes, bf16.kv.gpu.max_bytes, "{model}");
    }
}

/// The Phase 6b per-tier lab config (plan Task 6) loads, spells out the tiers and differs from
/// the Phase 2c config only in its `kv` section's lower tiers.
#[test]
fn phase6b_tier_lab_config_loads() {
    let lab = repo_root().join("scripts/lab");
    let tier = turbine_core::config::load(&lab.join("phase6-novanas-llama.yaml"), &[])
        .expect("the per-tier config loads");
    let base = turbine_core::config::load(&lab.join("phase2c-novanas-llama.yaml"), &[])
        .expect("the Phase 2c config loads");
    assert!(tier.kv.cpu.enabled && !tier.kv.nvme.enabled);
    assert_eq!(tier.kv.cpu.format.as_str(), "l0");
    assert_eq!(tier.kv.gpu.max_bytes, base.kv.gpu.max_bytes);
    assert_eq!(tier.kv.dtype, base.kv.dtype);
    assert_eq!(tier.model.path, base.model.path);
}
