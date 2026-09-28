//! `turbine-golden capture` / `compare` against an in-test axum mock of an OpenAI endpoint.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

const POSITIONS: usize = 10;
const TOP: usize = 20;
const PROMPT_TOKEN_IDS: [u32; 3] = [1, 2, 3];

/// What the mock generates for one prompt: the greedy tokens and the top-20 per position.
#[derive(Clone)]
struct Scripted {
    tokens: Vec<u32>,
    top: Vec<Vec<(u32, f32)>>,
}

/// Prompt text (completion) or last message content (chat) → scripted output.
type Script = HashMap<String, Scripted>;

#[derive(Clone, Copy)]
enum Variant {
    /// Exactly the reference.
    Base,
    /// p01 emits its second choice at position 5 (reference margin there: 2 nats).
    FlipP01,
    /// p02 emits its second choice at position 5 (reference margin there: 0.3 nats).
    FlipP02,
    /// p01's runner-up logprob at position 2 (reference −1.5, a likely candidate) moves by
    /// 0.2 nats.
    ShiftP01,
    /// p01's third-ranked logprob at position 2 (reference −3.0, a tail candidate) moves by
    /// 0.5 nats.
    ShiftTailP01,
    /// Every generation request answers 500.
    Fail,
}

/// Greedy token `base + i`, runner-up `alt + i` at `-0.5 - margin`, then 18 low-ranked ids.
fn scripted(base: u32, alt: u32, margin_at_5: f32) -> Scripted {
    let mut tokens = Vec::new();
    let mut top = Vec::new();
    for i in 0..POSITIONS {
        let margin = if i == 5 { margin_at_5 } else { 1.0 };
        let tok = base + i as u32;
        let mut row = vec![(tok, -0.5), (alt + i as u32, -0.5 - margin)];
        for k in 0..TOP - 2 {
            row.push((2000 + k as u32, -3.0 - k as f32 * 0.5));
        }
        tokens.push(tok);
        top.push(row);
    }
    Scripted { tokens, top }
}

fn script(variant: Variant) -> Script {
    let mut p01 = scripted(100, 500, 2.0);
    let mut p02 = scripted(200, 600, 0.3);
    match variant {
        Variant::Base | Variant::Fail => {}
        Variant::FlipP01 => p01.tokens[5] = p01.top[5][1].0,
        Variant::FlipP02 => p02.tokens[5] = p02.top[5][1].0,
        Variant::ShiftP01 => p01.top[2][1].1 -= 0.2,
        Variant::ShiftTailP01 => p01.top[2][2].1 -= 0.5,
    }
    HashMap::from([("alpha".to_string(), p01), ("beta".to_string(), p02)])
}

struct Mock {
    script: Script,
    fail: bool,
    /// Held by every generation request before it answers, so overlapping requests overlap.
    delay: Duration,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl Mock {
    fn new(script: Script, fail: bool, delay: Duration) -> Self {
        Self {
            script,
            fail,
            delay,
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        }
    }
}

/// Counts one generation request as in flight (and records the peak) until dropped.
struct InFlight<'a>(&'a Mock);

impl<'a> InFlight<'a> {
    async fn enter(mock: &'a Mock) -> Self {
        let now = mock.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        mock.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(mock.delay).await;
        Self(mock)
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

fn bad_request(msg: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": msg, "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// The fields `turbine-golden` must send on every replay; `None` when they are all present.
fn check_common(body: &Value) -> Option<String> {
    let expect = [
        ("model", json!("mock-model")),
        ("temperature", json!(0)),
        ("return_tokens_as_token_ids", json!(true)),
        ("ignore_eos", json!(true)),
        ("stream", json!(false)),
        ("max_tokens", json!(POSITIONS)),
    ];
    let matches = |got: Option<&Value>, want: &Value| match (got, want.as_f64()) {
        // 0 and 0.0 are the same temperature.
        (Some(g), Some(w)) => g.as_f64() == Some(w),
        (Some(g), None) => g == want,
        (None, _) => false,
    };
    expect
        .iter()
        .find(|(k, v)| !matches(body.get(*k), v))
        .map(|(k, v)| format!("expected {k} = {v}, body {body}"))
}

fn id_str(id: u32) -> String {
    format!("token_id:{id}")
}

async fn models() -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}))
}

async fn completions(State(mock): State<Arc<Mock>>, Json(body): Json<Value>) -> Response {
    let _in_flight = InFlight::enter(&mock).await;
    if mock.fail {
        return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
    }
    if let Some(msg) = check_common(&body) {
        return bad_request(msg);
    }
    if body["logprobs"] != json!(20) {
        return bad_request(format!("expected logprobs 20, body {body}"));
    }
    let Some(s) = body["prompt"].as_str().and_then(|p| mock.script.get(p)) else {
        return bad_request(format!("unknown prompt, body {body}"));
    };
    let tops: Vec<Value> = s
        .top
        .iter()
        .map(|row| {
            let map: serde_json::Map<String, Value> = row
                .iter()
                .map(|(id, lp)| (id_str(*id), json!(lp)))
                .collect();
            Value::Object(map)
        })
        .collect();
    Json(json!({
        "id": "cmpl-mock",
        "object": "text_completion",
        "model": "mock-model",
        "system_fingerprint": "mock-engine-1",
        "choices": [{
            "index": 0,
            "text": "irrelevant",
            "prompt_token_ids": PROMPT_TOKEN_IDS,
            "logprobs": {
                "tokens": s.tokens.iter().map(|t| id_str(*t)).collect::<Vec<_>>(),
                "token_logprobs": s
                    .tokens
                    .iter()
                    .zip(&s.top)
                    .map(|(tok, row)| row.iter().find(|(id, _)| id == tok).map_or(-9.0, |e| e.1))
                    .collect::<Vec<_>>(),
                "top_logprobs": tops,
            },
            "finish_reason": "length",
        }],
    }))
    .into_response()
}

async fn chat(State(mock): State<Arc<Mock>>, Json(body): Json<Value>) -> Response {
    let _in_flight = InFlight::enter(&mock).await;
    if mock.fail {
        return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
    }
    if let Some(msg) = check_common(&body) {
        return bad_request(msg);
    }
    if body["logprobs"] != json!(true) || body["top_logprobs"] != json!(20) {
        return bad_request(format!(
            "expected logprobs true / top_logprobs 20, body {body}"
        ));
    }
    if body["chat_template_kwargs"]["date_string"] != json!("26 Jul 2024") {
        return bad_request(format!("chat_template_kwargs not forwarded, body {body}"));
    }
    let last = body["messages"]
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m["content"].as_str());
    let Some(s) = last.and_then(|c| mock.script.get(c)) else {
        return bad_request(format!("unknown messages, body {body}"));
    };
    let content: Vec<Value> = s
        .tokens
        .iter()
        .zip(&s.top)
        .map(|(tok, row)| {
            json!({
                "token": id_str(*tok),
                "logprob": row.iter().find(|(id, _)| id == tok).map_or(-9.0, |e| e.1),
                "top_logprobs": row
                    .iter()
                    .map(|(id, lp)| json!({"token": id_str(*id), "logprob": lp}))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    Json(json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": "mock-model",
        "system_fingerprint": "mock-engine-1",
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "irrelevant"},
            "logprobs": {"content": content},
            "finish_reason": "length",
        }],
    }))
    .into_response()
}

/// The completions mock reports each generated token's own logprob, also where a flip made it
/// the runner-up (like the chat mock), not the top candidate's (Scout e6bb10da).
#[tokio::test]
async fn completions_mock_logprob_is_the_generated_tokens() {
    let mock = Arc::new(Mock::new(script(Variant::FlipP01), false, Duration::ZERO));
    let body = json!({
        "model": "mock-model", "temperature": 0, "return_tokens_as_token_ids": true,
        "ignore_eos": true, "stream": false, "max_tokens": POSITIONS, "logprobs": 20,
        "prompt": "alpha",
    });
    let resp = completions(State(mock), Json(body)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let lps = &v["choices"][0]["logprobs"]["token_logprobs"];
    assert_eq!(lps[4], json!(-0.5), "{v}");
    // Position 5 generated the runner-up, scripted at -0.5 - 2.0.
    assert_eq!(lps[5], json!(-2.5), "{v}");
}

async fn mock_server(variant: Variant) -> SocketAddr {
    let mock = Arc::new(Mock::new(
        script(variant),
        matches!(variant, Variant::Fail),
        Duration::ZERO,
    ));
    serve(mock).await
}

async fn serve(mock: Arc<Mock>) -> SocketAddr {
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .with_state(mock);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

async fn golden(args: Vec<String>) -> Run {
    let out = tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_turbine-golden"))
            .args(&args)
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn compare_args(addr: SocketAddr, reference: &Path, output: &str) -> Vec<String> {
    strings(&[
        "compare",
        "--url",
        &format!("http://{addr}"),
        "--reference",
        reference.to_str().unwrap(),
        "--output",
        output,
    ])
}

/// Compare as JSON; returns the exit code and the report.
async fn compare_json(variant: Variant, reference: &Path) -> (Option<i32>, Value, String) {
    let addr = mock_server(variant).await;
    let run = golden(compare_args(addr, reference, "json")).await;
    let report = serde_json::from_str(&run.stdout)
        .unwrap_or_else(|e| panic!("report is not JSON ({e}): {}\n{}", run.stdout, run.stderr));
    (run.code, report, run.stderr)
}

fn prompt_report<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == id)
        .unwrap_or_else(|| panic!("{id} missing from {report}"))
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbine-golden-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("mock")).unwrap();
    dir
}

#[tokio::test(flavor = "multi_thread")]
async fn capture_and_compare_roundtrip() {
    // Committed layout: prompts.jsonl one level above <slug>/{reference.jsonl,tolerance.json}.
    let dir = temp_dir("roundtrip");
    let prompts = dir.join("prompts.jsonl");
    std::fs::write(
        &prompts,
        concat!(
            r#"{"id":"p01","kind":"completion","prompt":"alpha","max_tokens":10}"#,
            "\n",
            r#"{"id":"p02","kind":"chat","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"beta"}],"max_tokens":10,"chat_template_kwargs":{"date_string":"26 Jul 2024"}}"#,
            "\n"
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("mock/tolerance.json"),
        r#"{"min_identical_prefix":8,"min_prompts_passing":2,"top_k":5,"max_abs_logprob_diff_likely":0.15,"max_abs_logprob_diff_tail":0.55,"likely_logprob_floor":-2.0,"margin_nats":0.5}"#,
    )
    .unwrap();
    let reference = dir.join("mock/reference.jsonl");

    // capture
    let addr = mock_server(Variant::Base).await;
    let run = golden(strings(&[
        "capture",
        "--url",
        &format!("http://{addr}"),
        "--prompts",
        prompts.to_str().unwrap(),
        "--out",
        reference.to_str().unwrap(),
        "--model",
        "mock-model",
    ]))
    .await;
    assert_eq!(run.code, Some(0), "capture failed: {}", run.stderr);
    assert!(!dir.join("mock/reference.jsonl.tmp").exists());
    let text = std::fs::read_to_string(&reference).unwrap();
    let records: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), 2, "{text}");
    let base = script(Variant::Base);
    for (rec, (id, key)) in records.iter().zip([("p01", "alpha"), ("p02", "beta")]) {
        assert_eq!(rec["id"], id);
        assert_eq!(rec["engine"], "mock-engine-1");
        assert_eq!(rec["model"], "mock-model");
        assert_eq!(rec["prompt_token_ids"], json!(PROMPT_TOKEN_IDS));
        assert_eq!(rec["tokens"], json!(base[key].tokens), "{id}");
        let captured = rec["captured"].as_str().unwrap();
        assert!(
            captured.len() == 20 && captured.ends_with('Z') && captured.as_bytes()[10] == b'T',
            "captured is not RFC 3339 UTC: {captured}"
        );
        let top = rec["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), POSITIONS);
        for (pos, row) in top.iter().enumerate() {
            let row = row.as_array().unwrap();
            assert_eq!(row.len(), TOP, "{id} position {pos}");
            // Highest first, [id, logprob] pairs, whichever logprob shape the endpoint used.
            assert_eq!(row[0][0], json!(base[key].tokens[pos]));
            assert_eq!(row[0][1].as_f64().unwrap(), -0.5);
            assert_eq!(row[1][0], json!(base[key].top[pos][1].0));
        }
    }

    // compare against the same endpoint: tolerance holds (defaults for --prompts, --tolerance
    // and --model).
    let (code, report, stderr) = compare_json(Variant::Base, &reference).await;
    assert_eq!(code, Some(0), "{report}\n{stderr}");
    assert_eq!(report["passed"], true, "{report}");
    assert_eq!(report["prompts_passing"], 2, "{report}");
    for id in ["p01", "p02"] {
        let p = prompt_report(&report, id);
        assert_eq!(p["identical_prefix"], 10, "{p}");
        assert_eq!(p["first_divergence"], Value::Null, "{p}");
        assert_eq!(
            p["max_abs_logprob_diff_likely"].as_f64().unwrap(),
            0.0,
            "{p}"
        );
        assert_eq!(p["max_abs_logprob_diff_tail"].as_f64().unwrap(), 0.0, "{p}");
    }

    // A flip at position 5 where the reference margin is 2 nats: violated, located, margin given.
    let (code, report, stderr) = compare_json(Variant::FlipP01, &reference).await;
    assert_eq!(code, Some(1), "{report}\n{stderr}");
    assert_eq!(report["passed"], false);
    let p = prompt_report(&report, "p01");
    assert_eq!(p["identical_prefix"], 5, "{p}");
    assert_eq!(p["first_divergence"], 5, "{p}");
    assert_eq!(p["margin_at_divergence"].as_f64().unwrap(), 2.0, "{p}");
    assert_eq!(p["passed"], false, "{p}");
    assert_eq!(prompt_report(&report, "p02")["passed"], true);
    let addr = mock_server(Variant::FlipP01).await;
    let run = golden(compare_args(addr, &reference, "text")).await;
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(
        run.stdout.contains("p01")
            && run.stdout.contains("first_divergence=5")
            && run.stdout.contains("margin=2.000"),
        "text report does not locate the divergence: {}",
        run.stdout
    );

    // The same flip where the reference margin is 0.3 nats (< 0.5): excused, counted as passing.
    let (code, report, stderr) = compare_json(Variant::FlipP02, &reference).await;
    assert_eq!(code, Some(0), "{report}\n{stderr}");
    let p = prompt_report(&report, "p02");
    assert_eq!(p["first_divergence"], 5, "{p}");
    assert!(
        (p["margin_at_divergence"].as_f64().unwrap() - 0.3).abs() < 1e-5,
        "{p}"
    );
    assert_eq!(p["passed"], true, "{p}");
    assert_eq!(report["prompts_passing"], 2, "{report}");

    // A likely top-5 logprob (reference > −2) shifted by 0.2 nats (> 0.15): violated even
    // though every token matches.
    let (code, report, stderr) = compare_json(Variant::ShiftP01, &reference).await;
    assert_eq!(code, Some(1), "{report}\n{stderr}");
    let p = prompt_report(&report, "p01");
    assert_eq!(p["identical_prefix"], 10, "{p}");
    assert!(
        (p["max_abs_logprob_diff_likely"].as_f64().unwrap() - 0.2).abs() < 1e-5,
        "{p}"
    );
    assert_eq!(p["logprob_within_bound"], false, "{p}");
    assert_eq!(p["passed"], false, "{p}");

    // A tail top-5 logprob (reference ≤ −2) shifted by 0.5 nats (≤ 0.55): holds.
    let (code, report, stderr) = compare_json(Variant::ShiftTailP01, &reference).await;
    assert_eq!(code, Some(0), "{report}\n{stderr}");
    let p = prompt_report(&report, "p01");
    assert!(
        (p["max_abs_logprob_diff_tail"].as_f64().unwrap() - 0.5).abs() < 1e-5,
        "{p}"
    );
    assert_eq!(
        p["max_abs_logprob_diff_likely"].as_f64().unwrap(),
        0.0,
        "{p}"
    );
    assert_eq!(p["passed"], true, "{p}");

    // An endpoint failing during capture: exit 1 and no fixture (not even the temp file).
    let failed_out = dir.join("mock/failed.jsonl");
    let addr = mock_server(Variant::Fail).await;
    let run = golden(strings(&[
        "capture",
        "--url",
        &format!("http://{addr}"),
        "--prompts",
        prompts.to_str().unwrap(),
        "--out",
        failed_out.to_str().unwrap(),
    ]))
    .await;
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("500"), "{}", run.stderr);
    assert!(!failed_out.exists());
    assert!(!dir.join("mock/failed.jsonl.tmp").exists());

    // Usage errors exit 2: a missing flag, and a reference file that does not exist.
    let run = golden(strings(&["compare", "--url", "http://127.0.0.1:9"])).await;
    assert_eq!(run.code, Some(2), "{}", run.stderr);
    let run = golden(compare_args(addr, &dir.join("absent.jsonl"), "json")).await;
    assert_eq!(run.code, Some(2), "{}", run.stderr);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn compare_concurrency_bounded() {
    // Eight completion prompts, each answered after 100 ms so that concurrent replays overlap.
    const PROMPTS: usize = 8;
    let dir = temp_dir("concurrency");
    let keys: Vec<String> = (0..PROMPTS).map(|i| format!("prompt-{i}")).collect();
    let script: Script = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), scripted(100 * (i as u32 + 1), 5000, 2.0)))
        .collect();
    let prompts = dir.join("prompts.jsonl");
    let lines: String = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            format!(
                "{}\n",
                json!({"id": format!("p{i:02}"), "kind": "completion", "prompt": k, "max_tokens": POSITIONS})
            )
        })
        .collect();
    std::fs::write(&prompts, lines).unwrap();
    std::fs::write(
        dir.join("mock/tolerance.json"),
        r#"{"min_identical_prefix":8,"min_prompts_passing":8,"top_k":5,"max_abs_logprob_diff_likely":0.15,"max_abs_logprob_diff_tail":0.55,"likely_logprob_floor":-2.0,"margin_nats":0.5}"#,
    )
    .unwrap();
    let reference = dir.join("mock/reference.jsonl");

    let mock = Arc::new(Mock::new(script, false, Duration::from_millis(100)));
    let addr = serve(Arc::clone(&mock)).await;
    let run = golden(strings(&[
        "capture",
        "--url",
        &format!("http://{addr}"),
        "--prompts",
        prompts.to_str().unwrap(),
        "--out",
        reference.to_str().unwrap(),
        "--model",
        "mock-model",
    ]))
    .await;
    assert_eq!(run.code, Some(0), "capture failed: {}", run.stderr);
    mock.max_in_flight.store(0, Ordering::SeqCst);

    let mut args = compare_args(addr, &reference, "json");
    args.extend(strings(&["--concurrency", "3"]));
    let run = golden(args).await;
    assert_eq!(run.code, Some(0), "{}\n{}", run.stdout, run.stderr);
    let report: Value = serde_json::from_str(&run.stdout)
        .unwrap_or_else(|e| panic!("report is not JSON ({e}): {}", run.stdout));
    assert_eq!(report["passed"], true, "{report}");
    assert_eq!(report["prompts_passing"], PROMPTS, "{report}");
    // Results in prompt order, whatever order the replies arrived in.
    let ids: Vec<&str> = report["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["id"].as_str().unwrap())
        .collect();
    let expected: Vec<String> = (0..PROMPTS).map(|i| format!("p{i:02}")).collect();
    assert_eq!(ids, expected, "{report}");

    let peak = mock.max_in_flight.load(Ordering::SeqCst);
    assert!(
        (2..=3).contains(&peak),
        "--concurrency 3 put {peak} requests in flight at once"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Two prompts captured from the `Base` mock under `<dir>/mock/reference.jsonl`, judged by
/// `tolerance` (written as `<dir>/mock/tolerance.json`).
async fn captured_fixture(name: &str, tolerance: &str) -> (PathBuf, PathBuf) {
    let dir = temp_dir(name);
    std::fs::write(
        dir.join("prompts.jsonl"),
        concat!(
            r#"{"id":"p01","kind":"completion","prompt":"alpha","max_tokens":10}"#,
            "\n",
            r#"{"id":"p02","kind":"completion","prompt":"beta","max_tokens":10}"#,
            "\n"
        ),
    )
    .unwrap();
    std::fs::write(dir.join("mock/tolerance.json"), tolerance).unwrap();
    let reference = dir.join("mock/reference.jsonl");
    let addr = mock_server(Variant::Base).await;
    let run = golden(strings(&[
        "capture",
        "--url",
        &format!("http://{addr}"),
        "--prompts",
        dir.join("prompts.jsonl").to_str().unwrap(),
        "--out",
        reference.to_str().unwrap(),
        "--model",
        "mock-model",
    ]))
    .await;
    assert_eq!(run.code, Some(0), "capture failed: {}", run.stderr);
    (dir, reference)
}

/// Compare `variant` against `reference` at `concurrency`: exit code, JSON report, text report.
async fn compare_at(
    variant: Variant,
    reference: &Path,
    concurrency: u32,
) -> (Option<i32>, Value, String) {
    let addr = mock_server(variant).await;
    let mut args = compare_args(addr, reference, "json");
    args.extend(strings(&["--concurrency", &concurrency.to_string()]));
    let run = golden(args).await;
    let report: Value = serde_json::from_str(&run.stdout)
        .unwrap_or_else(|e| panic!("report is not JSON ({e}): {}\n{}", run.stdout, run.stderr));
    let mut args = compare_args(addr, reference, "text");
    args.extend(strings(&["--concurrency", &concurrency.to_string()]));
    let text = golden(args).await;
    assert_eq!(text.code, run.code, "{}\n{}", text.stdout, text.stderr);
    (run.code, report, text.stdout)
}

fn assert_close(v: &Value, expected: f64, report: &Value) {
    let got = v
        .as_f64()
        .unwrap_or_else(|| panic!("{v} is not a number in {report}"));
    assert!(
        (got - expected).abs() < 1e-6,
        "{got} != {expected}: {report}"
    );
}

/// A likely logprob shifted by 0.2 nats (strict bound 0.15, batched bound 0.25) with every
/// token identical: violated at `--concurrency 1`, holds at `--concurrency 2`, and the report
/// says which bounds applied. Breaks if the batched bounds apply at concurrency 1, are ignored
/// above it, relax the token rule, or are not reported.
#[tokio::test(flavor = "multi_thread")]
async fn compare_batched_bounds_apply_only_above_concurrency_1() {
    let (dir, reference) = captured_fixture(
        "batched",
        r#"{"min_identical_prefix":8,"min_prompts_passing":2,"top_k":5,"max_abs_logprob_diff_likely":0.15,"max_abs_logprob_diff_tail":0.55,"likely_logprob_floor":-2.0,"margin_nats":0.5,"max_abs_logprob_diff_likely_batched":0.25,"max_abs_logprob_diff_tail_batched":0.75}"#,
    )
    .await;

    let (code, report, text) = compare_at(Variant::ShiftP01, &reference, 1).await;
    assert_eq!(code, Some(1), "{report}");
    assert_eq!(report["concurrency"], 1, "{report}");
    assert_eq!(report["logprob_bounds"]["batched"], false, "{report}");
    assert_close(
        &report["logprob_bounds"]["max_abs_logprob_diff_likely"],
        0.15,
        &report,
    );
    assert_eq!(prompt_report(&report, "p01")["passed"], false, "{report}");
    assert!(text.contains("strict bounds (concurrency 1)"), "{text}");
    // The tolerance file is echoed as read, batched keys included.
    assert_close(
        &report["tolerance"]["max_abs_logprob_diff_likely_batched"],
        0.25,
        &report,
    );

    let (code, report, text) = compare_at(Variant::ShiftP01, &reference, 2).await;
    assert_eq!(code, Some(0), "{report}");
    assert_eq!(report["passed"], true, "{report}");
    assert_eq!(report["concurrency"], 2, "{report}");
    assert_eq!(report["logprob_bounds"]["batched"], true, "{report}");
    assert_close(
        &report["logprob_bounds"]["max_abs_logprob_diff_likely"],
        0.25,
        &report,
    );
    assert_close(
        &report["logprob_bounds"]["max_abs_logprob_diff_tail"],
        0.75,
        &report,
    );
    let p = prompt_report(&report, "p01");
    assert_eq!(p["logprob_within_bound"], true, "{p}");
    assert_eq!(p["passed"], true, "{p}");
    assert!(
        text.contains("batched bounds (concurrency 2)") && text.contains("≤ 0.25"),
        "{text}"
    );

    // The token rule is unchanged under the batched bounds: a flip where the reference margin
    // is 2 nats still fails.
    let (code, report, _) = compare_at(Variant::FlipP01, &reference, 2).await;
    assert_eq!(code, Some(1), "{report}");
    assert_eq!(prompt_report(&report, "p01")["passed"], false, "{report}");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// A tolerance.json without the batched keys judges a concurrent run by the strict bounds.
/// Breaks if a missing batched key is read as 0 or as unbounded, or if the file is rejected.
#[tokio::test(flavor = "multi_thread")]
async fn compare_without_batched_keys_falls_back_to_strict_bounds() {
    let (dir, reference) = captured_fixture(
        "batched-fallback",
        r#"{"min_identical_prefix":8,"min_prompts_passing":2,"top_k":5,"max_abs_logprob_diff_likely":0.15,"max_abs_logprob_diff_tail":0.55,"likely_logprob_floor":-2.0,"margin_nats":0.5}"#,
    )
    .await;

    // Identical output passes at concurrency 2 (a missing key is not a zero bound) ...
    let (code, report, text) = compare_at(Variant::Base, &reference, 2).await;
    assert_eq!(code, Some(0), "{report}");
    assert_eq!(report["logprob_bounds"]["batched"], false, "{report}");
    assert!(text.contains("strict bounds (concurrency 2)"), "{text}");
    assert!(
        report["tolerance"]
            .get("max_abs_logprob_diff_likely_batched")
            .is_none(),
        "{report}"
    );
    // ... and the 0.2-nat shift is still judged against 0.15 (not unbounded).
    let (code, report, _) = compare_at(Variant::ShiftP01, &reference, 2).await;
    assert_eq!(code, Some(1), "{report}");
    assert_close(
        &report["logprob_bounds"]["max_abs_logprob_diff_likely"],
        0.15,
        &report,
    );
    assert_close(
        &report["logprob_bounds"]["max_abs_logprob_diff_tail"],
        0.55,
        &report,
    );
    assert_eq!(prompt_report(&report, "p01")["passed"], false, "{report}");

    std::fs::remove_dir_all(&dir).unwrap();
}
