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

/// Phase 8 (S-3, S-4): `turbine-golden eval` and `eval-compare` against an in-process mock.
mod phase8_eval {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use axum::{Json, Router, http::StatusCode, routing::get, routing::post};
    use serde_json::{Value, json};

    /// A per-test directory under the system temp dir, removed on drop.
    struct EvalTempDir(PathBuf);

    impl EvalTempDir {
        fn new() -> Self {
            let dir =
                std::env::temp_dir().join(format!("turbine-golden-eval-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            EvalTempDir(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for EvalTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Expected answer of eval mock task `i`.
    fn eval_mock_answer(i: u64) -> u64 {
        i * 1000 + 7
    }

    /// The mock's reply to a prompt `q<i>`: tasks 0..150 correct (plain, comma-grouped, trailing
    /// period), 150..200 wrong (`$` prefix, unit suffix, off by one); a prompt `fail` returns 500.
    fn eval_mock_reply(prompt: &str) -> Result<String, StatusCode> {
        if prompt == "fail" {
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        let i: u64 = prompt
            .trim_start_matches('q')
            .parse()
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        let n = eval_mock_answer(i);
        let grouped = format!("{},{:03}", n / 1000, n % 1000);
        Ok(match (i < 150, i % 3) {
            (true, 0) => n.to_string(),
            (true, 1) => grouped,
            (true, _) => format!("{n}."),
            (false, 0) => format!("${n}"),
            (false, 1) => format!("{n} apples"),
            (false, _) => (n + 1).to_string(),
        })
    }

    async fn spawn_eval_mock() -> String {
        let app = Router::new()
        .route("/v1/models", get(|| async { Json(json!({"object": "list", "data": [{"id": "mock-model"}]})) }))
        .route(
            "/v1/completions",
            post(|Json(body): Json<Value>| async move {
                assert_eq!(body["temperature"], 0.0);
                assert_eq!(body["model"], "mock-model");
                let text = eval_mock_reply(body["prompt"].as_str().unwrap_or_default())?;
                Ok::<_, StatusCode>(Json(json!({"choices": [{"index": 0, "text": text}]})))
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                let prompt = body["messages"][0]["content"].as_str().unwrap_or_default().to_string();
                let text = eval_mock_reply(&prompt)?;
                Ok::<_, StatusCode>(Json(
                    json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}),
                ))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// 200 tasks, even ids as completions, odd ids as chat.
    fn write_eval_tasks(path: &Path, fail_at: Option<u64>) {
        let mut lines = String::new();
        for i in 0..200u64 {
            let q = if Some(i) == fail_at {
                "fail".to_string()
            } else {
                format!("q{i}")
            };
            let task = if i % 2 == 0 {
                json!({"id": format!("t{i}"), "prompt": q, "answer": eval_mock_answer(i).to_string(),
                   "match": "number", "max_tokens": 16})
            } else {
                json!({"id": format!("t{i}"), "messages": [{"role": "user", "content": q}],
                   "answer": eval_mock_answer(i).to_string(), "match": "number", "max_tokens": 16})
            };
            lines.push_str(&task.to_string());
            lines.push('\n');
        }
        std::fs::write(path, lines).unwrap();
    }

    fn eval_golden_cmd() -> Command {
        Command::new(env!("CARGO_BIN_EXE_turbine-golden"))
    }

    fn eval_compare(
        dir: &Path,
        baseline: &Path,
        accuracy: f64,
        correct: u64,
        max_drop: &str,
    ) -> Option<i32> {
        let mut report: Value = serde_json::from_slice(&std::fs::read(baseline).unwrap()).unwrap();
        report["accuracy"] = json!(accuracy);
        report["correct"] = json!(correct);
        let candidate = dir.join(format!("candidate-{correct}.json"));
        std::fs::write(&candidate, report.to_string()).unwrap();
        let out = eval_golden_cmd()
            .args(["eval-compare", "--baseline"])
            .arg(baseline)
            .arg("--candidate")
            .arg(&candidate)
            .args(["--max-drop", max_drop])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("baseline accuracy 0.7500"), "{stdout}");
        out.status.code()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn eval_accuracy_report() {
        let base = spawn_eval_mock().await;
        let dir = EvalTempDir::new();
        let tasks = dir.path().join("tasks.jsonl");
        write_eval_tasks(&tasks, None);

        let out = tokio::task::spawn_blocking({
            let (base, tasks) = (base.clone(), tasks.clone());
            move || {
                eval_golden_cmd()
                    .args(["eval", "--url", &base, "--output", "json", "--tasks"])
                    .arg(&tasks)
                    .output()
                    .unwrap()
            }
        })
        .await
        .unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(report["accuracy"], json!(0.75), "{report}");
        assert_eq!(report["correct"], 150);
        assert_eq!(report["total"], 200);
        assert_eq!(report["model"], "mock-model");
        let results = report["results"].as_array().unwrap();
        assert_eq!(results.len(), 200);
        assert_eq!(results[1]["id"], "t1");
        assert_eq!(results[1]["correct"], true, "comma-grouped answer accepted");
        assert_eq!(results[2]["correct"], true, "trailing period accepted");
        assert_eq!(results[150]["correct"], false, "$ prefix rejected");
        assert_eq!(results[151]["correct"], false, "unit suffix rejected");
        let baseline = dir.path().join("baseline.json");
        std::fs::write(&baseline, &out.stdout).unwrap();

        let (d, b) = (dir.path().to_path_buf(), baseline.clone());
        let codes = tokio::task::spawn_blocking(move || {
            (
                eval_compare(&d, &b, 0.745, 149, "0.01"),
                eval_compare(&d, &b, 0.73, 146, "0.01"),
            )
        })
        .await
        .unwrap();
        assert_eq!(codes, (Some(0), Some(1)));

        // A failure midway exits 2 naming the task and prints no report.
        let failing = dir.path().join("failing.jsonl");
        write_eval_tasks(&failing, Some(120));
        let out = tokio::task::spawn_blocking(move || {
            eval_golden_cmd()
                .args(["eval", "--url", &base, "--output", "json", "--tasks"])
                .arg(&failing)
                .output()
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(out.status.code(), Some(2));
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("task t120"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.stdout.is_empty());
    }
}

/// Phase 8 (S-4): the committed GSM8K-200 task set every lossy-format gate runs, and
/// (Phase 6a) the full GSM8K test split used for the FP8-KV and MXFP4-A16 full-set
/// reruns (`.procoder/ask/decisions.md`, "Phase 6a gate misses on GSM8K-200").
mod phase8_eval_task_set {
    use std::path::{Path, PathBuf};

    fn eval_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/eval")
    }

    /// Checks the shape every eval task file must have: exactly `expected_count` lines,
    /// each a valid task with a unique id, `final_number` matching and a numeric answer.
    fn check_task_set(path: &Path, expected_count: usize) {
        use turbine_bench::golden::eval::{MatchKind, load_tasks, normalize_number};
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(
            text.lines().count(),
            expected_count,
            "{}: exactly {expected_count} lines",
            path.display()
        );
        let tasks = load_tasks(path).expect("every line parses, ids unique");
        assert_eq!(tasks.len(), expected_count);
        let mut ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), expected_count, "unique ids");
        for t in &tasks {
            assert_eq!(t.match_kind, MatchKind::FinalNumber, "{}", t.id);
            assert!(
                t.max_tokens >= 512,
                "{}: room for the chain of thought",
                t.id
            );
            assert!(
                normalize_number(&t.answer).is_some(),
                "{}: answer {:?} is not numeric",
                t.id,
                t.answer
            );
            assert!(t.max_tokens > 0, "{}", t.id);
        }
    }

    #[test]
    fn eval_task_set_valid() {
        let root = eval_root();
        check_task_set(&root.join("gsm8k-200.jsonl"), 200);
        let notice =
            std::fs::read_to_string(root.join("NOTICE")).expect("tests/eval/NOTICE exists");
        assert!(
            notice.contains("MIT License"),
            "NOTICE carries the source license"
        );
    }

    /// The full GSM8K test split: 1,319 items, same shape as gsm8k-200.jsonl, and its
    /// first 200 items byte-identical to the committed gsm8k-200.jsonl (both are the
    /// same GSM8K test split, wrapped and matched the same way).
    #[test]
    fn eval_task_set_full_valid_and_matches_200() {
        let root = eval_root();
        let full_path = root.join("gsm8k-full.jsonl");
        check_task_set(&full_path, 1319);

        let full_text = std::fs::read_to_string(&full_path).unwrap();
        let full_first_200: String = full_text
            .lines()
            .take(200)
            .map(|line| format!("{line}\n"))
            .collect();
        let text_200 = std::fs::read_to_string(root.join("gsm8k-200.jsonl")).unwrap();
        assert_eq!(
            full_first_200, text_200,
            "gsm8k-full.jsonl's first 200 items are byte-identical to gsm8k-200.jsonl"
        );
    }

    /// The completion-form counterpart of gsm8k-200.jsonl (Phase 6a: base checkpoints have no
    /// chat template): same shape, every task a `prompt` (never `messages`) with a `stop`
    /// sequence so a base model does not run on past its answer, and its 200 answers equal
    /// gsm8k-200.jsonl's in order (both wrap the same 200 GSM8K test-split questions).
    #[test]
    fn eval_task_set_completion_valid_and_matches_200_answers() {
        use turbine_bench::golden::eval::load_tasks;

        let root = eval_root();
        let completion_path = root.join("gsm8k-200-completion.jsonl");
        check_task_set(&completion_path, 200);

        let completion_tasks = load_tasks(&completion_path).unwrap();
        for t in &completion_tasks {
            assert!(t.prompt.is_some(), "{}: completion form uses prompt", t.id);
            assert!(t.messages.is_none(), "{}: never messages", t.id);
            assert!(
                t.stop.as_ref().is_some_and(|s| !s.is_empty()),
                "{}: needs a stop sequence (no chat template to end the turn)",
                t.id
            );
        }

        let chat_tasks = load_tasks(&root.join("gsm8k-200.jsonl")).unwrap();
        let chat_answers: Vec<&str> = chat_tasks.iter().map(|t| t.answer.as_str()).collect();
        let completion_answers: Vec<&str> =
            completion_tasks.iter().map(|t| t.answer.as_str()).collect();
        assert_eq!(
            completion_answers, chat_answers,
            "same 200 questions in the same order, so the same answers"
        );
    }
}

/// Phase 6a (S-11): the golden fixture of every quantized proof checkpoint (and the 8B BF16
/// baseline it is compared with) is complete: 16 reference records, the Llama tolerance keys and
/// a README naming the checkpoint and the command that produced the reference. Slugs whose
/// directory does not exist yet pass (each format task commits its own fixture).
mod quant_fixtures {
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    /// Slugs of spec phase-6a Data with the Hub repository, the pinned revision and the fixture
    /// script the README must name. `llama-3.2-3b-instruct-yarn16` is not listed: it is not a
    /// quantized checkpoint and holds a 17th (long) prompt.
    const QUANT_SLUGS: [(&str, &str, &str, &str); 8] = [
        (
            "llama-3.2-3b-instruct-fp8-dynamic",
            "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic",
            "c308a86de78778c5f904a1d82401ac85e18ca205",
            "quant_reference.py",
        ),
        (
            "llama-3.2-3b-instruct-fp8",
            "RedHatAI/Llama-3.2-3B-Instruct-FP8",
            "377571d314b30f1d58448499e4100e2deafe7d7d",
            "quant_reference.py",
        ),
        (
            "llama-3.2-3b-instruct-fp8-block",
            "unsloth/Llama-3.2-3B-Instruct-FP8-Block",
            "08cf804398b23fab4a1df02fbe8d4d5a11a800cc",
            "quant_reference.py",
        ),
        (
            "llama-3.2-3b-instruct-awq",
            "casperhansen/llama-3.2-3b-instruct-awq",
            "272b3bde867b606760447deb9a4d2719fbdfd3ae",
            "quant_reference.py",
        ),
        (
            "llama-3.2-3b-instruct-gptq",
            "shuyuej/Llama-3.2-3B-Instruct-GPTQ",
            "dd5a311f040728fbc612eb03c8dadfae0a90552f",
            "quant_reference.py",
        ),
        (
            "llama-3.1-8b-instruct-mxfp4a16",
            "FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16",
            "14c3aca849a72df8fcc8b3a30ab8d9eed86ee646",
            "quant_reference.py",
        ),
        (
            "llama-3.1-8b-instruct",
            "unsloth/Llama-3.1-8B-Instruct",
            "4699cc75b550f9c6f3173fb80f4703b62d946aa5",
            "hf_reference.py",
        ),
        (
            "llama-3.2-3b-mxfp4-a4",
            "matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq",
            "91925ffda6977d097354a99718a20e035f8af80a",
            "quant_reference.py",
        ),
    ];

    fn golden_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
    }

    fn read_json(path: &Path) -> Result<Value, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Checks one fixture directory; `Err` names the first incomplete part.
    fn check_fixture(
        dir: &Path,
        prompts_path: &Path,
        tolerance_template: &Path,
        repo: &str,
        revision: &str,
        script: &str,
    ) -> Result<(), String> {
        let name = dir.display();
        let prompts = prompts(prompts_path)?;
        if prompts.len() != 16 {
            return Err(format!(
                "{}: {} prompts, want 16",
                prompts_path.display(),
                prompts.len()
            ));
        }
        let reference = dir.join("reference.jsonl");
        let text = std::fs::read_to_string(&reference)
            .map_err(|e| format!("{}: {e}", reference.display()))?;
        let records: Vec<Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| format!("{name}: reference line: {e}")))
            .collect::<Result<_, _>>()?;
        if records.len() != prompts.len() {
            return Err(format!(
                "{name}: reference.jsonl has {} records, want 16",
                records.len()
            ));
        }
        for (rec, (id, max_tokens)) in records.iter().zip(&prompts) {
            if rec["id"].as_str() != Some(id.as_str()) {
                return Err(format!(
                    "{name}: record {} where prompt {id} is expected",
                    rec["id"]
                ));
            }
            let tokens = rec["tokens"].as_array().map_or(0, Vec::len);
            let tops = rec["top_logprobs"].as_array().map_or(0, Vec::len);
            if tokens as u64 != *max_tokens || tops != tokens {
                return Err(format!(
                    "{name}: {id} has {tokens} tokens and {tops} top_logprobs rows, want {max_tokens}"
                ));
            }
            if rec["prompt_token_ids"].as_array().is_none_or(Vec::is_empty) {
                return Err(format!("{name}: {id} has no prompt_token_ids"));
            }
        }
        let want = read_json(tolerance_template)?;
        let have = read_json(&dir.join("tolerance.json"))?;
        let keys = want
            .as_object()
            .ok_or("tolerance template is not an object")?;
        for key in keys.keys() {
            if have.get(key).is_none_or(|v| !v.is_number()) {
                return Err(format!("{name}: tolerance.json lacks numeric {key}"));
            }
        }
        let readme = dir.join("README.md");
        let readme =
            std::fs::read_to_string(&readme).map_err(|e| format!("{}: {e}", readme.display()))?;
        for needle in [repo, revision, script] {
            if !readme.contains(needle) {
                return Err(format!("{name}: README.md does not name {needle}"));
            }
        }
        Ok(())
    }

    /// The prompts of `prompts.jsonl`: `(id, max_tokens)` in file order.
    fn prompts(path: &Path) -> Result<Vec<(String, u64)>, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: Value = serde_json::from_str(l).map_err(|e| format!("prompts: {e}"))?;
                match (v["id"].as_str(), v["max_tokens"].as_u64()) {
                    (Some(id), Some(max)) => Ok((id.to_string(), max)),
                    _ => Err(format!("prompts: bad line {l}")),
                }
            })
            .collect()
    }

    #[test]
    fn quant_fixtures_valid() {
        let root = golden_root();
        let template = root.join("llama-3.2-3b-instruct/tolerance.json");
        let mut failures = Vec::new();
        for (slug, repo, revision, script) in QUANT_SLUGS {
            let dir = root.join(slug);
            if !dir.exists() {
                continue;
            }
            let prompts = root.join("prompts.jsonl");
            if let Err(e) = check_fixture(&dir, &prompts, &template, repo, revision, script) {
                failures.push(e);
            }
        }
        assert!(
            failures.is_empty(),
            "incomplete fixtures:\n{}",
            failures.join("\n")
        );
    }

    /// The checker itself: a complete fixture (built from the committed Llama one) passes, and
    /// each incomplete variant fails naming what is missing. Breaks if a missing record, a
    /// missing tolerance key or a README without the repo, revision or command slips through.
    #[test]
    fn quant_fixtures_valid_rejects_incomplete() {
        let root = golden_root();
        let prompts = root.join("prompts.jsonl");
        let template = root.join("llama-3.2-3b-instruct/tolerance.json");
        let (repo, rev, script) = (
            "org/model",
            "0123456789abcdef0123456789abcdef01234567",
            "quant_reference.py",
        );
        let base =
            std::env::temp_dir().join(format!("turbine-quant-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let write = |name: &str, drop_record: bool, drop_key: bool, readme: &str| {
            let dir = base.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            let reference =
                std::fs::read_to_string(root.join("llama-3.2-3b-instruct/reference.jsonl"))
                    .unwrap();
            let mut lines: Vec<&str> = reference.lines().collect();
            if drop_record {
                lines.pop();
            }
            std::fs::write(dir.join("reference.jsonl"), lines.join("\n") + "\n").unwrap();
            let mut tol = read_json(&template).unwrap();
            if drop_key {
                tol.as_object_mut()
                    .unwrap()
                    .remove("max_abs_logprob_diff_tail_batched");
            }
            std::fs::write(dir.join("tolerance.json"), tol.to_string()).unwrap();
            std::fs::write(dir.join("README.md"), readme).unwrap();
            dir
        };
        let full = format!("`{repo}` at `{rev}`: `uv run scripts/golden/{script} …`");
        let check = |dir: &Path| check_fixture(dir, &prompts, &template, repo, rev, script);

        assert_eq!(check(&write("ok", false, false, &full)), Ok(()));
        let e = check(&write("short", true, false, &full)).unwrap_err();
        assert!(e.contains("15 records"), "{e}");
        let e = check(&write("key", false, true, &full)).unwrap_err();
        assert!(e.contains("max_abs_logprob_diff_tail_batched"), "{e}");
        let e = check(&write("rev", false, false, &format!("`{repo}`, {script}"))).unwrap_err();
        assert!(e.contains(rev), "{e}");
        let e = check(&write("cmd", false, false, &format!("`{repo}` at `{rev}`"))).unwrap_err();
        assert!(e.contains(script), "{e}");
        let e = check(&base.join("absent")).unwrap_err();
        assert!(e.contains("reference.jsonl"), "{e}");
        let _ = std::fs::remove_dir_all(&base);
    }
}
