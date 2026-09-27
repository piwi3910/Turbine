//! Phase 4 KV correctness on the GPU (P4 S-3, S-11, S-17; plan Task 16).
//!
//! Both tests are ignored (lab only, run by `scripts/lab-test.sh novanas`, later `dgx-spark`):
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
//! - `nvme_round_trip_matches_cold`: with 1 GiB of L1 (L0 demotes by capacity past 70 %)
//!   and the NVMe tier at `/home/piwi/turbine-kv`, prompt A runs cold, long filler prompts push
//!   its blocks through L1 to L2 (L0 → L2 on unified memory), then A runs again: the same greedy
//!   answer, promotions from L2, no checksum eviction (every promoted block's CRC32C matched the
//!   value written), and at most 64 GiB of slab files under the path.
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
        let body = json!({
            "model": SERVED_NAME,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": true,
            "logprobs": 1,
        });
        let (status, text) = request(
            self.addr,
            "POST",
            "/v1/completions",
            Some(&body.to_string()),
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
    let mut conn = TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(REQUEST_TIMEOUT)).unwrap();
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
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    // About 400 tokens: three full 128-token blocks each.
    let prompts: Vec<String> = (1..=5).map(|s| prompt(s, 350)).collect();

    let cold: Vec<(String, u64, u64)> = {
        let server = LabServer::start(&model_dir, &["kv.prefix_sharing=false".into()]);
        prompts
            .iter()
            .map(|p| server.complete(p, ANSWER_TOKENS))
            .collect()
    };
    let server = LabServer::start(&model_dir, &[]);
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

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn prefix_reuse_suffix_lengths_match_cold() {
    if !require_backend("hip") {
        return;
    }
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

    let cold: Vec<IdAnswer> = {
        let server = LabServer::start(&model_dir, &["kv.prefix_sharing=false".into()]);
        prompts
            .iter()
            .map(|(_, target)| server.complete_ids(target, ANSWER_TOKENS))
            .collect()
    };
    let server = LabServer::start(&model_dir, &[]);
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
            println!("blocks {blocks} suffix {suffix}: warm differs from cold at token {first:?}");
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
    let _gpu = one_server_at_a_time();
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let kv_dir = Path::new(KV_DIR);
    assert!(kv_dir.is_dir(), "{KV_DIR} is not mounted");
    let server = LabServer::start(
        &model_dir,
        &[
            format!("kv.nvme.path={KV_DIR}"),
            "kv.nvme.max_bytes=64GiB".into(),
            // L0 stays the lab config's 8 GiB: a smaller cap shrinks Phase 3's device budget
            // below what the HIP runtime and libraries already hold, and the device_memory
            // signal would put the server in SURVIVAL. Cached blocks are demoted by capacity
            // past 70 % of L0, so filler prompts still push A down through a 1 GiB L1.
            "kv.cpu.max_bytes=1GiB".into(),
        ],
    );
    let a = prompt(100, 350);
    // A and the fillers are session turns: blocks of one-off requests are never copied down.
    let cold = server.complete_in(&a, ANSWER_TOKENS, Some("a"));

    // 1 GiB of L1 holds 73 blocks of 14,680,064 bytes: once twice that many have gone on to L2,
    // A's (the oldest, never touched again) are there.
    let to_l2 = || {
        server.metric(r#"turbine_kv_demotions_total{from="l1",to="l2"}"#)
            + server.metric(r#"turbine_kv_demotions_total{from="l0",to="l2"}"#)
    };
    let mut filler = 0;
    while to_l2() < 2.0 * 73.0 {
        filler += 1;
        assert!(
            filler <= 400,
            "demotion to L2 never happened ({} blocks)",
            to_l2()
        );
        let key = format!("filler-{filler}");
        server.complete_in(&prompt(1_000 + filler, 1_800), 1, Some(&key));
    }
    println!("{filler} filler prompts moved {} blocks to L2", to_l2());

    // Bring A back explicitly (`POST /turbine/v1/kv/prefetch`): on the R9700 the planner may
    // rightly find recomputing a few blocks cheaper than reading them from NVMe, and this test
    // is about the bytes surviving the round trip, not about that choice.
    let from_l2 = || {
        server.metric(r#"turbine_kv_promotions_total{from="l2",to="l0"}"#)
            + server.metric(r#"turbine_kv_promotions_total{from="l2",to="l1"}"#)
    };
    let body = json!({ "prompt": a }).to_string();
    let (status, text) = request(server.addr, "POST", "/turbine/v1/kv/prefetch", Some(&body));
    assert_eq!(status, 202, "{text}");
    println!("prefetch of A: {text}");
    let started = Instant::now();
    while from_l2() < 1.0 {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the prefetch never promoted from L2: {text}"
        );
        std::thread::sleep(POLL);
    }
    std::thread::sleep(Duration::from_secs(2));

    let warm = server.complete(&a, ANSWER_TOKENS);
    assert_eq!(
        (&warm.0, warm.1),
        (&cold.0, cold.1),
        "A after its L2 round trip"
    );
    assert!(warm.2 > 0, "A reused no prefix");
    let promoted = from_l2();
    assert!(promoted > 0.0, "nothing was promoted from L2");
    assert_eq!(
        server.metric(r#"turbine_kv_evictions_total{tier="l2",reason="checksum"}"#),
        0.0,
        "a promoted block's bytes differed from what was written"
    );
    let slabs = slab_bytes(kv_dir);
    assert!(slabs <= KV_CAP_BYTES, "{slabs} bytes of slab files");
    println!("nvme_round_trip_matches_cold ok: {promoted} promotions from L2, {slabs} slab bytes");
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
