//! Phase 4 KV correctness on the GPU (P4 S-3, S-11, S-17; plan Task 16).
//!
//! The GPU tests are ignored (lab only, run by `scripts/lab-test.sh novanas`, later `dgx-spark`):
//! they start `turbine-server` on `TURBINE_TEST_MODEL_DIR` (Llama-3.2-3B-Instruct BF16) with
//! `scripts/lab/phase4-novanas.yaml` and `--set` overrides, and compare greedy completions.
//!
//! - `prefix_reuse_matches_cold`: 5 prompts of more than two 128-token blocks, 64 greedy tokens
//!   each, first with `kv.prefix_sharing: false` (cold), then on a fresh server with sharing on,
//!   each sent twice: both runs answer within the golden gate's batched bounds of the cold one
//!   (`assert_within_batched_bounds`: a token-granular warm prefill — even the first run, over a
//!   published partial tail of an earlier prompt — starts at an arbitrary token offset, and HIP
//!   GEMM rounding follows batch composition, user decision 2026-09-26; lead decision
//!   2026-10-05, "5p Tasks 7–8"), and the repeat reports `cached_tokens` > 0.
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
//!   bytes, so a round-tripped prefix answers exactly like the cold FP8 run and a reused one
//!   within the batched bounds (see [`prefix_reuse_matches_cold`]) — plus the FP8 L0 pool
//!   itself: F8E4M3 pages of half the bytes and at least 1.95× the BF16 run's blocks in the same
//!   budget.
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
            // The gate's top-k: the batched-bounds margin excuse at a divergence needs the
            // cold run's runner-up (see `batched_verdict`).
            "logprobs": GOLDEN_TOP_K,
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
            tokens: choice["logprobs"]["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|t| t.as_str().expect("token string").to_string())
                .collect(),
            logprobs: choice["logprobs"]["token_logprobs"]
                .as_array()
                .expect("token_logprobs")
                .iter()
                .map(|l| l.as_f64().expect("logprob"))
                .collect(),
            top: choice["logprobs"]["top_logprobs"]
                .as_array()
                .expect("top_logprobs")
                .iter()
                .map(|row| {
                    let mut row: Vec<(String, f64)> = row
                        .as_object()
                        .expect("top row object")
                        .iter()
                        .map(|(t, lp)| (t.clone(), lp.as_f64().expect("top logprob")))
                        .collect();
                    row.sort_by(|a, b| b.1.total_cmp(&a.1));
                    row
                })
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
    /// Each generated token's display string, in order (`logprobs.tokens`).
    tokens: Vec<String>,
    /// Each generated token's logprob, as the server printed it (equal bits print alike).
    logprobs: Vec<f64>,
    /// Each generated token's top-k as (display string, logprob), the chosen token first.
    top: Vec<Vec<(String, f64)>>,
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

/// [`prefix_reuse_matches_cold`] on servers started with the extra overrides `sets`. The cold
/// run is the reference; both later runs are judged by [`assert_within_batched_bounds`] — with
/// token-granular reuse even the first run of a later prompt can attach a published partial tail
/// and start its prefill mid-block, so on HIP the answers are held to the golden gate's batched
/// bounds instead of bit-equality (lead decision 2026-10-05, "5p Tasks 7–8").
fn prefix_reuse_matches_cold_with(sets: &[String]) {
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    // About 400 tokens: three full 128-token blocks each.
    let prompts: Vec<String> = (1..=5).map(|s| prompt(s, 350)).collect();

    let cold_sets = [sets, &["kv.prefix_sharing=false".to_string()]].concat();
    let cold: Vec<IdAnswer> = {
        let server = LabServer::start(&model_dir, &cold_sets);
        assert_tp(&server, sets);
        prompts
            .iter()
            .map(|p| server.complete_with_logprobs(json!(p), ANSWER_TOKENS, None))
            .collect()
    };
    let server = LabServer::start(&model_dir, sets);
    assert_tp(&server, sets);
    for (i, p) in prompts.iter().enumerate() {
        let first = server.complete_with_logprobs(json!(p), ANSWER_TOKENS, None);
        let warm = server.complete_with_logprobs(json!(p), ANSWER_TOKENS, None);
        println!(
            "prompt {i}: cold {:?} first cached {} warm cached {}",
            cold[i].text, first.cached_tokens, warm.cached_tokens
        );
        assert_eq!(cold[i].completion_tokens, u64::from(ANSWER_TOKENS));
        assert_within_batched_bounds(&cold[i], &first, &format!("prompt {i}: first run"));
        assert_within_batched_bounds(&cold[i], &warm, &format!("prompt {i}: warm run"));
        assert!(
            warm.cached_tokens > 0,
            "prompt {i}: the warm run reused no prefix"
        );
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
/// bytes and at least 1.95× the blocks of the BF16 run (same budget), and prefix reuse answers
/// within the golden gate's batched bounds of the cold FP8 run (see
/// [`prefix_reuse_matches_cold_with`]). Breaks if the FP8 pool is not smaller per block, or a
/// reused FP8 prefix answers outside the batched bounds of recomputing it.
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

/// P5p S-5 (`prefix_reuse_partial_block_matches_cold`): a finished primer publishes its partial
/// tail; a later prompt sharing it reuses up to the last written token through one tail copy,
/// and every token and logprob stays bit-equal to the cold run. Covers tail reuses of 1, 2,
/// 17, 64, 100, 126 and 127 tokens into a 128-token block and one divergence inside a shared
/// full block. Breaks if a mid-block warm prefill leaves the cold run's bits or the reported
/// `cached_tokens` misses the reused tokens.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn prefix_reuse_partial_block_matches_cold() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    const BLOCK: u32 = 128;
    // Tokens of a primer's partial tail the target reuses (the primer ends tail tokens into
    // its fourth block), and one divergence inside the third shared block.
    const TAILS: [u32; 7] = [1, 2, 17, 64, 100, 126, 127];
    const EXTRA: u32 = 8;
    let cases: Vec<(Vec<u32>, Vec<u32>, u32)> = TAILS
        .iter()
        .zip(1u32..)
        .flat_map(|(&tail, i)| {
            let shared = token_ids(100 + i, 3 * BLOCK);
            let primer = [shared.as_slice(), &token_ids(200 + i, tail)].concat();
            // The target continues the primer's tail with fresh tokens (its own suffix).
            let target = [primer.as_slice(), &token_ids(400 + i, EXTRA)].concat();
            let want = 3 * BLOCK + tail;
            vec![(primer, target, want)]
        })
        .chain(std::iter::once({
            // One divergence inside the third shared block: the target shares the primer's
            // first 3*BLOCK + DIVERGE − 1 tokens, then differs.
            const DIVERGE: u32 = 70;
            let shared = token_ids(100 + 99, 3 * BLOCK);
            let primer = [shared.as_slice(), &token_ids(200 + 99, 100)].concat();
            let mut target = primer[..(3 * BLOCK + DIVERGE - 1) as usize].to_vec();
            let last = target.last_mut().unwrap();
            *last = (*last + 7) % 100_000;
            target.extend(token_ids(500, 8));
            (primer, target, 3 * BLOCK + DIVERGE - 1)
        }))
        .collect();
    let cold_sets: [String; 1] = ["kv.partial_prefix_reuse=false".to_string()];
    let cold: Vec<IdAnswer> = {
        let server = LabServer::start(&model_dir, &cold_sets);
        let mut out = Vec::new();
        for (_, t, _) in &cases {
            out.push(server.complete_ids(t, ANSWER_TOKENS));
        }
        out
    };
    let server = LabServer::start(&model_dir, &[]);
    let mut failed = Vec::new();
    for ((primer, target, want), cold) in cases.iter().zip(&cold) {
        // The primer publishes the tail the target reuses; a target that does not continue
        // its primer is the cold-equality leg only.
        if !primer.is_empty() && target.len() > primer.len() {
            server.complete_ids(primer, ANSWER_TOKENS);
        }
        let warm = server.complete_ids(target, ANSWER_TOKENS);
        let label = format!("prompt {} want {want}", target.len());
        println!(
            "{label}: cached {} cold {:?} warm {:?}",
            warm.cached_tokens, cold.text, warm.text
        );
        if !primer.is_empty() && target.len() > primer.len() {
            // A target that continues the primer reuses its full blocks + its whole tail,
            // capped at prompt - 2 (EXTRA >= 2 keeps the cap above the tail).
            assert_eq!(
                warm.cached_tokens,
                u64::from(*want),
                "{label}: the tail reuses up to the last written token"
            );
        }
        if warm.output() != cold.output()
            || warm
                .logprobs
                .iter()
                .zip(&cold.logprobs)
                .any(|(w, c)| w.to_bits() != c.to_bits())
        {
            failed.push(label);
        }
    }
    assert!(failed.is_empty(), "warm differs from cold: {failed:?}");
    println!("prefix_reuse_partial_block_matches_cold ok");
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
        // The primer's tail diverges from the target's suffix by construction, so only the
        // shared full blocks attach — whole-block reuse, unchanged under P5p (at least two
        // prompt tokens are prefilled, `MIN_RECOMPUTE_TOKENS`).
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
    assert_within_bounds(cold, warm, label, HeadBound::Flat(0.3), 0.5, 0.9);
}

/// The first 8 answer tokens' head bound. `GoldenSplit` is exactly how `turbine-golden compare`
/// judges the slug's batched bounds (user decision 2026-10-03, "6b: stale lossless-tail tags and
/// the tq4 lab bound", 2): a position whose cold (reference) logprob is above `floor` is judged
/// against `likely`, at or below it against `tail`.
#[derive(Clone, Copy)]
enum HeadBound {
    Flat(f64),
    GoldenSplit { likely: f64, tail: f64, floor: f64 },
}

impl HeadBound {
    /// The bound for one position's cold (reference) logprob.
    fn bound_at(&self, reference: f64) -> f64 {
        match self {
            HeadBound::Flat(b) => *b,
            HeadBound::GoldenSplit {
                likely,
                tail,
                floor,
            } => {
                if reference > *floor {
                    *likely
                } else {
                    *tail
                }
            }
        }
    }
}

/// [`assert_within_codec_bound`] with explicit bounds: the first 8 tokens' head [`HeadBound`],
/// the per-position |Δ| limit, the share of positions within it.
fn assert_within_bounds(
    cold: &IdAnswer,
    warm: &IdAnswer,
    label: &str,
    head_bound: HeadBound,
    pos_bound: f64,
    share: f64,
) {
    assert_eq!(warm.completion_tokens, cold.completion_tokens);
    let diff: Vec<f64> = cold
        .logprobs
        .iter()
        .zip(&warm.logprobs)
        .map(|(c, w)| (c - w).abs())
        .collect();
    let head: Vec<(usize, f64)> = diff.iter().take(8).copied().enumerate().collect();
    let head_worst = head.iter().map(|(_, d)| *d).fold(0.0, f64::max);
    let likely = head
        .iter()
        .filter(|(i, _)| cold.logprobs[*i] > -2.0)
        .count();
    let within = diff.iter().filter(|d| **d <= pos_bound).count() as f64 / diff.len().max(1) as f64;
    let worst = diff.iter().cloned().fold(0.0, f64::max);
    println!(
        "{label}: worst of the first 8 |Δ logprob| {head_worst:.3} ({likely} of 8 likely), \
         {within:.2} within {pos_bound}, \
         worst {worst:.3} (cached {}, lossy cached {})",
        warm.cached_tokens, warm.lossy_cached_tokens
    );
    for (i, d) in &head {
        let bound = head_bound.bound_at(cold.logprobs[*i]);
        assert!(
            d <= &bound,
            "first tokens differ: position {i} |Δ| {d:.3} over {bound:.3}"
        );
    }
    assert!(
        within >= share,
        "only {within} of the positions are within {pos_bound}"
    );
}

/// The golden gate's `top_k` for this slug (`tests/golden/llama-3.2-3b-instruct/tolerance.json`),
/// also what the tests request so the margin excuse at a divergence has a runner-up to read.
const GOLDEN_TOP_K: u32 = 5;

/// The golden gate's batched bounds for this slug (`tolerance.json`
/// `max_abs_logprob_diff_{likely,tail}_batched`), as the head [`HeadBound`] the existing split
/// helper already implements: a position whose reference (cold) logprob is above the −2 floor is
/// judged against 0.25, at or below it against 0.75.
const BATCHED_BOUNDS: HeadBound = HeadBound::GoldenSplit {
    likely: 0.25,
    tail: 0.75,
    floor: -2.0,
};

/// The gate's `margin_nats`: a divergence whose reference top-1 − top-2 margin is under this is a
/// near-tie, excused.
const GOLDEN_MARGIN: f64 = 0.5;

/// The gate's `min_identical_prefix` of a 64-token answer.
const GOLDEN_MIN_IDENTICAL_PREFIX: usize = 32;

/// The reference top row's top-1 minus top-2 logprob, whatever its order (compare.rs `margin`).
fn margin_of(row: &[(String, f64)]) -> Option<f64> {
    let mut lps: Vec<f64> = row.iter().map(|e| e.1).collect();
    lps.sort_by(|a, b| b.total_cmp(a));
    (lps.len() >= 2).then(|| lps[0] - lps[1])
}

/// `turbine-golden compare`'s batched verdict for one warm answer against its cold reference
/// (lead decision 2026-10-05, "5p Tasks 7–8: HIP tolerance for partial-tail prefill tests"):
/// judge every position of the identical token prefix by [`BATCHED_BOUNDS`] — the |Δ logprob| of
/// the chosen tokens against the bound of the cold logprob's tier — and require the divergence
/// (any) to be a near-tie (the cold top-1 − top-2 margin at it under [`GOLDEN_MARGIN`]) or the
/// shared prefix to cover [`GOLDEN_MIN_IDENTICAL_PREFIX`] tokens. Unlike compare.rs this judges
/// only the chosen tokens (not every reference top-k candidate), so a re-ordered rank 3–5 is not
/// a violation on its own. Breaks if the tiers' bounds are swapped, the split keys on the warm
/// logprob, or a pre-divergence drift over the bound or a real (wide-margin) divergence passes.
fn batched_verdict(cold: &IdAnswer, warm: &IdAnswer) -> Result<(), String> {
    if warm.completion_tokens != cold.completion_tokens {
        return Err(format!(
            "completion token count {} != cold {}",
            warm.completion_tokens, cold.completion_tokens
        ));
    }
    let identical = cold
        .tokens
        .iter()
        .zip(&warm.tokens)
        .take_while(|(c, w)| c == w)
        .count();
    let mut worst = (0usize, 0.0f64);
    for (i, (c, w)) in cold.logprobs.iter().zip(&warm.logprobs).enumerate() {
        if i >= identical {
            break;
        }
        let d = (c - w).abs();
        let bound = BATCHED_BOUNDS.bound_at(*c);
        if d > bound {
            return Err(format!(
                "position {i} |Δ logprob| {d:.3} over the batched bound {bound:.3} (cold logprob {c:.3})"
            ));
        }
        if d > worst.1 {
            worst = (i, d);
        }
    }
    let margin = cold.top.get(identical).and_then(|row| margin_of(row));
    let need = GOLDEN_MIN_IDENTICAL_PREFIX.min(cold.tokens.len());
    if identical < need && !margin.is_some_and(|m| m < GOLDEN_MARGIN) {
        return Err(format!(
            "diverged at token {identical} of {} (cold margin {margin:?}, need < {GOLDEN_MARGIN}), \
             worst |Δ logprob| within the shared prefix {:.3} at position {}",
            cold.tokens.len(),
            worst.1,
            worst.0
        ));
    }
    println!(
        "batched verdict: identical prefix {identical}/{} worst |Δ logprob| {:.3} at position {} \
         (cold margin at the divergence {margin:?})",
        cold.tokens.len(),
        worst.1,
        worst.0
    );
    Ok(())
}

/// [`batched_verdict`] as an assertion.
fn assert_within_batched_bounds(cold: &IdAnswer, warm: &IdAnswer, label: &str) {
    if let Err(reason) = batched_verdict(cold, warm) {
        panic!("{label}: {reason}");
    }
}

/// The batched verdict's rules, on synthetic answers (no server): the chosen-token logprobs
/// within the identical prefix are judged by the cold logprob's tier of [`BATCHED_BOUNDS`], and a
/// divergence passes only when the shared prefix reaches [`GOLDEN_MIN_IDENTICAL_PREFIX`] (clamped
/// to the answer length, as the gate clamps it) or the cold top-2 margin at the divergence is
/// under [`GOLDEN_MARGIN`]. Breaks if the bounds are swapped, the margin excuse keys on the wrong
/// run's row, or a wide-margin divergence passes.
#[test]
fn batched_verdict_excuses_only_a_near_tie_divergence() {
    fn answer(tokens: &[&str], logprobs: &[f64], top: &[Vec<f64>]) -> IdAnswer {
        IdAnswer {
            text: String::new(),
            completion_tokens: tokens.len() as u64,
            tokens: tokens.iter().map(|t| t.to_string()).collect(),
            logprobs: logprobs.to_vec(),
            top: top
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|lp| (String::new(), *lp))
                        .collect::<Vec<(String, f64)>>()
                })
                .collect(),
            prompt_tokens: 0,
            cached_tokens: 0,
            lossy_cached_tokens: 0,
        }
    }
    // The cold answer: an all-"a" answer of 4 tokens whose divergence row (position 2) holds a
    // 0.3-margin near-tie.
    let cold = answer(
        &["a"; 4],
        &[0.0, -1.0, -0.5, -3.0],
        &[
            vec![0.0, -1.5, -3.0, -4.0],
            vec![-1.0, -2.5, -3.0, -4.0],
            vec![-0.5, -0.8, -3.0, -4.0],
            vec![-3.0, -4.5, -5.0, -6.0],
        ],
    );
    // Same tokens and logprobs: full prefix, no divergence, passes without reading a margin.
    assert_within_batched_bounds(&cold, &cold, "warm equals cold");
    // Divergence at 2 of 4, cold margin there 0.3: excused as a near-tie despite the short prefix.
    let near = answer(
        &["a", "a", "b", "a"],
        &[0.0, -1.0, -0.8, -3.0],
        &four_rows([0.0, -1.0, -0.5, -3.0]),
    );
    assert_within_batched_bounds(&cold, &near, "near-tie divergence is excused");
    // The same divergence judged against a cold answer whose top-2 at the divergence is a 1.1
    // margin: a real flip, not excused (the warm run's own top rows are never read).
    let cold_wide = answer(
        &["a"; 4],
        &[0.0, -1.0, -0.5, -3.0],
        &[
            vec![0.0, -1.5, -3.0, -4.0],
            vec![-1.0, -2.5, -3.0, -4.0],
            vec![-0.5, -1.6, -3.0, -4.0],
            vec![-3.0, -4.5, -5.0, -6.0],
        ],
    );
    let err = batched_verdict(&cold_wide, &near).unwrap_err();
    assert!(err.contains("diverged at token 2"), "{err}");
    // The completion counts must match before anything is judged.
    let short = answer(
        &["a"; 3],
        &[0.0, -1.0, -0.5],
        &four_rows([0.0, -1.0, -0.5, -3.0]),
    );
    let err = batched_verdict(&cold, &short).unwrap_err();
    assert!(err.contains("completion token count"), "{err}");
    // A pre-divergence drift over the bound fails even when the prefix rule would hold.
    let hot = answer(
        &["a"; 4],
        &[0.0, -0.5, -0.5, -3.0], // |Δ| 0.5 at position 1; cold −1.0 is likely: bound 0.25
        &four_rows([0.0, -1.0, -0.5, -3.0]),
    );
    let err = batched_verdict(&cold, &hot).unwrap_err();
    assert!(err.contains("position 1"), "{err}");
    // The tail tier: the same |Δ| 0.5 at a position whose cold logprob sits exactly at the floor
    // (−2.0 is tail, `lp > floor`): 0.5 is within the 0.75 tail bound.
    let cold_tail = answer(
        &["a"; 4],
        &[0.0, -2.0, -0.5, -3.0],
        &four_rows([0.0, -2.0, -0.5, -3.0]),
    );
    let hot_tail = answer(
        &["a"; 4],
        &[0.0, -2.5, -0.5, -3.0],
        &four_rows([0.0, -2.0, -0.5, -3.0]),
    );
    assert_within_batched_bounds(&cold_tail, &hot_tail, "tail tier");
}

/// Four top rows whose entries follow the cold chosen logprobs, margins 1.5 everywhere, so a
/// case's own divergence row decides the verdict.
fn four_rows(lps: [f64; 4]) -> Vec<Vec<f64>> {
    lps.iter()
        .map(|lp| vec![*lp, lp - 1.5, lp - 3.0, lp - 4.0])
        .collect()
}

/// The golden head bound splits by the reference logprob exactly as `turbine-golden compare`
/// does: above the floor (−2) a position is judged likely (0.25), at the floor exactly or below
/// it tail (0.75) — the same `>` comparison against `likely_logprob_floor` as the gate.
/// Breaks if the split keys on the warm logprob, if −2 counts as likely, or if the tiers'
/// bounds are swapped.
#[test]
fn golden_head_bound_splits_likely_from_tail() {
    let bound = HeadBound::GoldenSplit {
        likely: 0.25,
        tail: 0.75,
        floor: -2.0,
    };
    assert_eq!(
        bound.bound_at(0.0),
        0.25,
        "a likely position gets the likely bound"
    );
    // Exactly at the floor is tail, matching compare.rs's `lp > floor`.
    assert_eq!(bound.bound_at(-2.0), 0.75);
    assert_eq!(bound.bound_at(-2.5), 0.75);
    assert_eq!(HeadBound::Flat(0.3).bound_at(-9.0), 0.3);
}

/// P6b S-2 / S-3 / S-8 (plan Task 6): with `kv.cpu.format: fp8_e4m3` below the BF16 pool, blocks
/// that capacity demotion moved to L1 are encoded; a later request for the same prompt reuses
/// them as lossy blocks (`usage.prompt_tokens_details.lossy_cached_tokens` > 0) and answers
/// within the FP8 tier's bound of its cold run; with `x-turbine-kv-lossy: deny` the lossy
/// blocks are not reused and the answer stays within the golden gate's batched bounds of the
/// cold one (lead decision 2026-10-05, "5p Tasks 7–8": the denied run can still attach an exact
/// published partial tail and start its prefill mid-block, which shifts HIP's GEMM rounding
/// without being a lossy reuse). Breaks if a lossy block is reused by a request that opted out,
/// or a lossy L1 copy is never reused.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn lossy_tier_reuse() {
    lossy_tier_reuse_with("fp8_e4m3", (HeadBound::Flat(0.3), 0.5, 0.9));
}

/// [`lossy_tier_reuse`] with L1 at TurboQuant 4-bit (P6b Task 8: the tables are uploaded and the
/// transcode runs `turbine_hip_tq`), held to [`TQ4_TIER_BOUNDS`].
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn lossy_tier_reuse_tq4() {
    lossy_tier_reuse_with("tq4", TQ4_TIER_BOUNDS);
}

/// The `tq4` tier's bound (P6b Task 9; head bound per user decision 2026-10-03, "6b: stale
/// lossless-tail tags and the tq4 lab bound", 2), taken from the gates that judged it: S-8
/// judges lossy KV by golden with the slug's batched bounds (Llama `tolerance.json`), and
/// `scripts/lab-bench.sh --golden16` with `kv.cpu.format=tq4` passed them; the shared-prefix
/// GSM8K gate (lossy cached ratio 0.927) then showed that reuse at that level costs no accuracy
/// (median 0.775 against BF16 0.780, McNemar n.s.). The head is judged exactly as
/// `turbine-golden compare` judges: a position whose cold logprob is above the
/// `likely_logprob_floor` (−2) against 0.25, at or below it against 0.75 — the t9 flat 0.25
/// assumed an exact tail block, and the all-lossy worst case measured 0.2518 on a cold-unlikely
/// position. 90 % of the 64 positions stay within 0.75 (a greedy flip later in the answer moves
/// the positions after it; the FP8 tier's 0.3 / 0.5 / 0.9 is tighter because FP8 is). Breaks if
/// the TurboQuant tier perturbs reuse beyond what golden accepts from batch composition.
const TQ4_TIER_BOUNDS: (HeadBound, f64, f64) = (
    HeadBound::GoldenSplit {
        likely: 0.25,
        tail: 0.75,
        floor: -2.0,
    },
    0.75,
    0.9,
);

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

fn lossy_tier_reuse_with(format: &str, bounds: (HeadBound, f64, f64)) {
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
            // The lossless tail (S-2: the latest finished sequence's last full blocks demote at
            // the L0 format, tags expiring with a later finish) is orthogonal to what this test
            // judges. It stays at 0 here on purpose: with the tail at its default, A's own tail
            // block demotes raw and comes back exact, which would soften the all-lossy worst
            // case the accuracy bound is held to. The tail's own behaviour is pinned by
            // `document_lists_copies_per_codec` and `last_block_is_scored_like_its_history`.
            "kv.lossless_tail_blocks=0".to_string(),
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
    dump_kv(&server);
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
    assert_within_bounds(
        &cold,
        &warm,
        &format!("{format} lossy L1 reuse"),
        bounds.0,
        bounds.1,
        bounds.2,
    );
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
    assert_within_batched_bounds(
        &cold,
        &denied,
        "x-turbine-kv-lossy: deny reuses no lossy block",
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

/// P7 S-12 (Task 33, `hybrid_prefix_reuse_matches_cold`): a Qwen3.5 hybrid's per-sequence
/// block-boundary state snapshots travel with their KV block through L1 and L2. A session's
/// prompt is filled and published, filler turns push its blocks down to L1 and then to L2, and
/// a later turn of the same session hits the prefix from each tier. Every token and logprob of
/// the warm turns equals the cold run bit for bit — the KV comes back lossless (both tiers at
/// `l0`) and the state restores from the nearest block-boundary snapshot instead of being
/// rebuilt — and `turbine_hybrid_state_snapshots_total{outcome="restored"}` counts the
/// restores. Breaks if a snapshot is lost on demotion/promotion (a warm turn diverges), the
/// state is rebuilt rather than restored, or the snapshot bytes do not survive the tier.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so, the Qwen3.5-9B weights and /home/piwi/turbine-kv"]
fn hybrid_prefix_reuse_matches_cold() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    // This test is only meaningful on the Qwen3.5 hybrid checkpoint
    // (`TURBINE_TEST_MODEL_DIR=/models/qwen3.5-9b`); the default lab Job points at Llama, where
    // there is no state to snapshot, so it is skipped there instead of failing.
    let arch = std::fs::read(model_dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|c| {
            c["architectures"][0]
                .as_str()
                .or_else(|| c["text_config"]["architectures"][0].as_str())
                .map(str::to_string)
        });
    if arch.as_deref() != Some("Qwen3_5ForConditionalGeneration") {
        println!(
            "hybrid_prefix_reuse_matches_cold: skipping, {} is not the Qwen3.5 hybrid",
            arch.as_deref().unwrap_or("no architecture")
        );
        return;
    }
    let kv_dir = Path::new(KV_DIR);
    assert!(kv_dir.is_dir(), "{KV_DIR} is not mounted");
    // A snapshot at every full block (so a hit has one to restore), the lossless tail off (so
    // the demoted blocks carry their snapshot in the tier record), and both tiers at `l0` (the
    // KV comes back exact, so the reuse must be bit-for-bit, not just within bounds).
    let base = [
        format!("kv.nvme.path={KV_DIR}"),
        "kv.nvme.max_bytes=64GiB".into(),
        "kv.cpu.max_bytes=1GiB".into(),
        "kv.hybrid_snapshot_interval_blocks=1".into(),
        "kv.lossless_tail_blocks=0".into(),
    ];

    // A's prompt: five full 128-token blocks. The primer continues it with one decode token.
    let a = prompt(100, 700);
    // Cold reference: sharing off, so nothing is reused anywhere.
    let cold_sets = [base.as_slice(), &["kv.prefix_sharing=false".to_string()]].concat();
    let cold = {
        let server = LabServer::start(&model_dir, &cold_sets);
        server.complete_with_logprobs(json!(a), ANSWER_TOKENS, Some("a"))
    };

    let server = LabServer::start(&model_dir, &base);
    let (_, block_bytes, _) = l0_tier(&server);
    let l1_blocks = ((1u64 << 30) / block_bytes.max(1)) as f64;

    // Turn 1 fills A's block-boundary snapshots and publishes its blocks.
    let first = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, Some("a"));
    assert_within_batched_bounds(&cold, &first, "turn 1 (cold fill)");

    // Filler turns of far-apart sessions push A's (oldest, never touched) blocks down to L1.
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

    // Turn 2 hits A from L1: the KV promotes exact and the state restores from a snapshot.
    let restored =
        |s: &LabServer| s.metric(r#"turbine_hybrid_state_snapshots_total{outcome="restored"}"#);
    let before_l1 = restored(&server);
    let from_l1 = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, Some("a"));
    if from_l1.cached_tokens == 0 {
        dump_kv(&server);
    }
    assert!(
        from_l1.cached_tokens > 0,
        "the L1 turn reused no prefix: {from_l1:?}"
    );
    assert!(
        restored(&server) > before_l1,
        "the L1 hit restored no snapshot"
    );
    assert_eq!(
        cold.output(),
        from_l1.output(),
        "the L1 turn differs from the cold run"
    );

    // Push A on down to L2 and hit it again: the snapshot travels in the tier record there too.
    fill_until_l2(&server, 2.0 * l1_blocks + 40.0);
    let before_l2 = restored(&server);
    let from_l2 = server.complete_with_logprobs(json!(a), ANSWER_TOKENS, Some("a"));
    if from_l2.cached_tokens == 0 {
        dump_kv(&server);
    }
    assert!(
        from_l2.cached_tokens > 0,
        "the L2 turn reused no prefix: {from_l2:?}"
    );
    assert!(
        restored(&server) > before_l2,
        "the L2 hit restored no snapshot"
    );
    assert_eq!(
        cold.output(),
        from_l2.output(),
        "the L2 turn differs from the cold run"
    );
    assert!(
        restored(&server) > 0.0,
        "no state snapshot was ever restored"
    );
    println!("hybrid_prefix_reuse_matches_cold ok");
}

/// The lab config loads and spells out the tiers the lab tests rely on.
#[test]
fn phase4_lab_config_loads() {
    let c = turbine_core::config::load(&lab_config(), &[]).expect("phase4-novanas.yaml loads");
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

/// The L0 compression ladder serves on the GPU (P6b S-7, plan Task 18): a BF16 pool with the
/// ladder's rung page classes (`kv.ladder.l0`, `max_format: tq4`) and L1 enabled reaches
/// /ready — the v2.11 mixed-format paged attention descriptor carries the layer's TurboQuant
/// tables whenever a block table mixes formats, though `kv.dtype` is `bf16`, so the tables
/// must be uploaded under the pool's namespace seed before the warm-up forward — and a
/// completion returns text. Breaks if the mixed pool starts without the executor's tables or
/// the classed decode diverges.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn l0_ladder_serves_on_gpu() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let sets = [
        "kv.ladder.enabled=true".to_string(),
        "kv.ladder.l0=true".to_string(),
        "kv.ladder.max_format=tq4".to_string(),
        "kv.cpu.enabled=true".to_string(),
        "kv.cpu.max_bytes=1GiB".to_string(),
    ];
    let server = LabServer::start(&model_dir, &sets);
    let (text, tokens, _) = server.complete(&prompt(7, 350), ANSWER_TOKENS);
    assert_eq!(tokens, u64::from(ANSWER_TOKENS));
    assert!(!text.trim().is_empty(), "empty completion: {text:?}");
}

/// P5p (user decision 2026-10-06, "5p: partial prefix reuse fails on a classed L0 pool", A):
/// token-granular reuse on the classed L0 pool of `kv.ladder.l0` (BF16 base, the ladder's
/// rung classes). A primer publishes its partial tail; a target continuing it reuses its full
/// blocks plus the tail through one copy-on-write tail copy of a base page — which the HIP
/// `copy_blocks` used to reject (`block id N is outside the pool's id space`, HTTP 500, and the
/// circuit opened for every following request). Every warm target answers 200 with
/// `cached_tokens` up to the last written token, within the batched bounds of the same config
/// without partial reuse, and `/ready` stays 200 (no circuit opened). Breaks if the tail copy
/// on a classed pool fails again or copies the wrong page.
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn partial_reuse_on_classed_l0_pool() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    const BLOCK: u32 = 128;
    const TAILS: [u32; 4] = [1, 17, 100, 127];
    const EXTRA: u32 = 8;
    let cases: Vec<(Vec<u32>, Vec<u32>, u32)> = TAILS
        .iter()
        .zip(1u32..)
        .map(|(&tail, i)| {
            let shared = token_ids(700 + i, 3 * BLOCK);
            let primer = [shared.as_slice(), &token_ids(800 + i, tail)].concat();
            let target = [primer.as_slice(), &token_ids(900 + i, EXTRA)].concat();
            (primer, target, 3 * BLOCK + tail)
        })
        .collect();
    let ladder = |partial: bool| -> Vec<String> {
        vec![
            "kv.ladder.enabled=true".to_string(),
            "kv.ladder.l0=true".to_string(),
            "kv.ladder.max_format=tq4".to_string(),
            "kv.cpu.enabled=true".to_string(),
            "kv.cpu.max_bytes=1GiB".to_string(),
            format!("kv.partial_prefix_reuse={partial}"),
        ]
    };
    let cold: Vec<IdAnswer> = {
        let server = LabServer::start(&model_dir, &ladder(false));
        cases
            .iter()
            .map(|(_, t, _)| server.complete_ids(t, ANSWER_TOKENS))
            .collect()
    };
    let server = LabServer::start(&model_dir, &ladder(true));
    for ((primer, target, want), cold) in cases.iter().zip(&cold) {
        server.complete_ids(primer, ANSWER_TOKENS);
        let warm = server.complete_ids(target, ANSWER_TOKENS);
        let label = format!("prompt {} want {want}", target.len());
        println!(
            "{label}: cached {} cold {:?} warm {:?}",
            warm.cached_tokens, cold.text, warm.text
        );
        assert_eq!(
            warm.cached_tokens,
            u64::from(*want),
            "{label}: the tail reuses up to the last written token"
        );
        assert_within_batched_bounds(cold, &warm, &label);
    }
    assert_eq!(
        request(server.addr, "GET", "/ready", None).0,
        200,
        "the circuit stays closed"
    );
    println!("partial_reuse_on_classed_l0_pool ok");
}
