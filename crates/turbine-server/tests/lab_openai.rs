//! Tool calling and structured output against the real Llama-3.2-3B-Instruct (P2 S-15, S-17,
//! S-18).
//!
//! `tools_and_json_schema` (ignored; lab only, run by `scripts/lab-test.sh novanas`) starts
//! `turbine-server` on `TURBINE_TEST_MODEL_DIR` with the `hip` backend, sends every chat request
//! of `tests/golden/tools/requests.jsonl` greedily and checks each answer against what the
//! request asks for:
//!
//! - `tool_choice` `required` or a named function: `finish_reason` `tool_calls`, at least one
//!   call (exactly one when `parallel_tool_calls` is false), every name allowed (the named one
//!   only), every `arguments` string a JSON object valid against that tool's `parameters`, and no
//!   call JSON in `content`;
//! - `auto` (explicit, or implied by `tools`): either tool calls valid as above, or `content`
//!   with no raw call JSON (a special token of the `llama3_json` format or a
//!   `{"name": …, "parameters": …}` object);
//! - `response_format` `json_schema`: `finish_reason` `stop` and `content` valid against the
//!   schema; `json_object`: `content` is a JSON object.
//!
//! The schemas are checked by a validator for the JSON Schema subset the fixture uses (`type`,
//! `properties`, `required`, `additionalProperties`, `enum`, `items`, `description`); the
//! always-run tests below refuse a fixture that uses any other keyword, so a schema is never
//! checked only partly. They also check the fixture's coverage and that the response checks
//! reject malformed answers, all without a GPU.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use turbine_kernels::test_support::{require_backend, require_env_dir};
use turbine_model::formats::{self, ToolFormat};

/// The model id `turbine-server` serves the weights under (as in the lab configs).
const SERVED_NAME: &str = "meta-llama/Llama-3.2-3B-Instruct";
/// Weights load plus warm-up on the R9700; a debug build of the server is slower than release.
const READY_LIMIT: Duration = Duration::from_secs(900);
/// One greedy generation of at most a few hundred tokens.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_millis(200);

/// The lab tests start one server each; every server sizes its KV pool to the whole card, so
/// they run one at a time.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// JSON Schema keywords the validator below implements.
const SUPPORTED_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "enum",
    "items",
    "description",
];

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/tools/requests.jsonl")
}

/// One line of `tests/golden/tools/requests.jsonl`: an id and a chat request without `model`.
struct ToolRequest {
    id: String,
    request: Value,
}

fn load_fixture() -> Vec<ToolRequest> {
    let path = fixture_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, line)| {
            let v: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), i + 1));
            let id = v["id"]
                .as_str()
                .unwrap_or_else(|| panic!("{}:{}: no string id", path.display(), i + 1))
                .to_string();
            assert!(
                v["request"].is_object(),
                "{}:{}: no request object",
                path.display(),
                i + 1
            );
            ToolRequest {
                id,
                request: v["request"].clone(),
            }
        })
        .collect()
}

/// What a request's answer must satisfy, derived from the request itself.
#[derive(Debug)]
enum Expect {
    /// `required` or a named function: tool calls only.
    ToolCalls {
        tools: BTreeMap<String, Value>,
        named: Option<String>,
        single: bool,
    },
    /// `auto`: valid tool calls or content without call JSON.
    Auto {
        tools: BTreeMap<String, Value>,
        single: bool,
    },
    JsonSchema(Value),
    JsonObject,
}

/// Tool name → `parameters` schema.
fn tool_schemas(request: &Value) -> BTreeMap<String, Value> {
    request["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|t| {
                    assert_eq!(t["type"], "function", "tool type: {t}");
                    let name = t["function"]["name"]
                        .as_str()
                        .unwrap_or_else(|| panic!("tool without a name: {t}"));
                    (name.to_string(), t["function"]["parameters"].clone())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn expectation(request: &Value) -> Result<Expect, String> {
    let tools = tool_schemas(request);
    let single = request["parallel_tool_calls"] == json!(false);
    let format = &request["response_format"];
    if !format.is_null() {
        if !tools.is_empty() {
            return Err("response_format combined with tools".into());
        }
        return match format["type"].as_str() {
            Some("json_schema") => {
                let schema = &format["json_schema"]["schema"];
                if schema.is_object() {
                    Ok(Expect::JsonSchema(schema.clone()))
                } else {
                    Err("json_schema without a schema object".into())
                }
            }
            Some("json_object") => Ok(Expect::JsonObject),
            other => Err(format!("unexpected response_format type {other:?}")),
        };
    }
    if tools.is_empty() {
        return Err("neither tools nor response_format".into());
    }
    match &request["tool_choice"] {
        Value::Null => Ok(Expect::Auto { tools, single }),
        Value::String(s) if s == "auto" => Ok(Expect::Auto { tools, single }),
        Value::String(s) if s == "required" => Ok(Expect::ToolCalls {
            tools,
            named: None,
            single,
        }),
        Value::Object(_) => {
            let name = request["tool_choice"]["function"]["name"]
                .as_str()
                .ok_or("named tool_choice without a function name")?
                .to_string();
            if !tools.contains_key(&name) {
                return Err(format!("tool_choice names {name}, which is not in tools"));
            }
            Ok(Expect::ToolCalls {
                tools,
                named: Some(name),
                single,
            })
        }
        other => Err(format!("unexpected tool_choice {other}")),
    }
}

/// Every keyword of `schema` (recursively) is one the validator implements.
fn check_supported(schema: &Value, path: &str) -> Result<(), String> {
    let obj = schema
        .as_object()
        .ok_or_else(|| format!("{path}: schema is not an object"))?;
    for (key, sub) in obj {
        if !SUPPORTED_KEYWORDS.contains(&key.as_str()) {
            return Err(format!("{path}: unsupported keyword {key:?}"));
        }
        match key.as_str() {
            "properties" => {
                let props = sub
                    .as_object()
                    .ok_or_else(|| format!("{path}.properties is not an object"))?;
                for (name, s) in props {
                    check_supported(s, &format!("{path}.properties.{name}"))?;
                }
            }
            "items" => check_supported(sub, &format!("{path}.items"))?,
            "additionalProperties" if sub.is_object() => {
                check_supported(sub, &format!("{path}.additionalProperties"))?
            }
            _ => {}
        }
    }
    Ok(())
}

fn type_matches(ty: &str, v: &Value) -> Result<bool, String> {
    Ok(match ty {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => {
            v.is_i64()
                || v.is_u64()
                || v.as_f64()
                    .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }
        other => return Err(format!("unknown type {other:?}")),
    })
}

/// Validates `v` against `schema` (the subset in `SUPPORTED_KEYWORDS`).
fn validate(schema: &Value, v: &Value, path: &str) -> Result<(), String> {
    check_supported(schema, path)?;
    match &schema["type"] {
        Value::Null => {}
        Value::String(ty) => {
            if !type_matches(ty, v)? {
                return Err(format!("{path}: {v} is not of type {ty}"));
            }
        }
        Value::Array(types) => {
            let mut any = false;
            for ty in types {
                any |= type_matches(ty.as_str().ok_or("type entry is not a string")?, v)?;
            }
            if !any {
                return Err(format!("{path}: {v} matches none of {}", schema["type"]));
            }
        }
        other => return Err(format!("{path}: bad type {other}")),
    }
    if let Some(allowed) = schema["enum"].as_array()
        && !allowed.contains(v)
    {
        return Err(format!("{path}: {v} is not one of {}", schema["enum"]));
    }
    if let Some(obj) = v.as_object() {
        let props = schema["properties"].as_object();
        if let Some(required) = schema["required"].as_array() {
            for key in required {
                let key = key.as_str().ok_or("required entry is not a string")?;
                if !obj.contains_key(key) {
                    return Err(format!("{path}: missing required property {key:?}"));
                }
            }
        }
        for (key, value) in obj {
            let sub = format!("{path}.{key}");
            match props.and_then(|p| p.get(key)) {
                Some(s) => validate(s, value, &sub)?,
                None => match &schema["additionalProperties"] {
                    Value::Bool(false) => {
                        return Err(format!("{path}: additional property {key:?}"));
                    }
                    s @ Value::Object(_) => validate(s, value, &sub)?,
                    _ => {}
                },
            }
        }
    }
    if let (Some(items), Some(arr)) = (schema.get("items"), v.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            validate(items, item, &format!("{path}[{i}]"))?;
        }
    }
    Ok(())
}

/// The Llama model's tool format (the lab serves Llama-3.2-3B-Instruct).
fn llama_format() -> &'static dyn ToolFormat {
    formats::registry()
        .get("llama3_json")
        .expect("llama3_json is registered")
}

/// The first call-shaped JSON object in `content` (a special token of the served tool format,
/// or an object with a string `name` and object `parameters`/`arguments`), if any.
fn leaked_call(content: &str) -> Option<String> {
    if let Some(token) = llama_format()
        .special_tokens()
        .iter()
        .find(|t| content.contains(t.text))
    {
        return Some(token.text.into());
    }
    for (i, _) in content.match_indices('{') {
        let mut stream = serde_json::Deserializer::from_str(&content[i..]).into_iter::<Value>();
        if let Some(Ok(Value::Object(obj))) = stream.next() {
            let named = obj.get("name").is_some_and(Value::is_string);
            let args = ["parameters", "arguments"]
                .iter()
                .any(|k| obj.get(*k).is_some_and(|a| a.is_object() || a.is_string()));
            if named && args {
                return Some(Value::Object(obj).to_string());
            }
        }
    }
    None
}

/// Checks `message.tool_calls` against the tools; returns the number of calls.
fn check_tool_calls(
    message: &Value,
    tools: &BTreeMap<String, Value>,
    named: Option<&str>,
    single: bool,
) -> Result<usize, String> {
    let calls = message["tool_calls"]
        .as_array()
        .ok_or_else(|| format!("no tool_calls in {message}"))?;
    if calls.is_empty() {
        return Err("empty tool_calls".into());
    }
    if single && calls.len() != 1 {
        return Err(format!(
            "{} calls with parallel_tool_calls false",
            calls.len()
        ));
    }
    let mut ids = BTreeSet::new();
    for (i, call) in calls.iter().enumerate() {
        let id = call["id"].as_str().unwrap_or("");
        if id.is_empty() || !ids.insert(id.to_string()) {
            return Err(format!("call {i}: missing or repeated id: {call}"));
        }
        if call["type"] != "function" {
            return Err(format!("call {i}: type is not function: {call}"));
        }
        let name = call["function"]["name"]
            .as_str()
            .ok_or_else(|| format!("call {i}: no function name: {call}"))?;
        if let Some(want) = named
            && name != want
        {
            return Err(format!("call {i}: {name} called, {want} was named"));
        }
        let schema = tools
            .get(name)
            .ok_or_else(|| format!("call {i}: unknown tool {name}"))?;
        let raw = call["function"]["arguments"]
            .as_str()
            .ok_or_else(|| format!("call {i}: arguments is not a string: {call}"))?;
        let args: Value = serde_json::from_str(raw)
            .map_err(|e| format!("call {i}: arguments are not JSON ({e}): {raw}"))?;
        validate(schema, &args, &format!("{name}.arguments"))
            .map_err(|e| format!("call {i}: {e}"))?;
    }
    Ok(calls.len())
}

/// Checks one chat completion against the expectation of its request.
fn check_response(expect: &Expect, response: &Value) -> Result<(), String> {
    let choices = response["choices"]
        .as_array()
        .ok_or_else(|| format!("no choices: {response}"))?;
    if choices.len() != 1 {
        return Err(format!("{} choices, expected 1", choices.len()));
    }
    let choice = &choices[0];
    let message = &choice["message"];
    let finish = choice["finish_reason"].as_str().unwrap_or("");
    let content = message["content"].as_str().unwrap_or("");
    let has_calls = message["tool_calls"]
        .as_array()
        .is_some_and(|c| !c.is_empty());
    match expect {
        Expect::ToolCalls {
            tools,
            named,
            single,
        } => {
            check_tool_calls(message, tools, named.as_deref(), *single)?;
            if finish != "tool_calls" {
                return Err(format!("finish_reason {finish:?}, expected tool_calls"));
            }
            if let Some(leak) = leaked_call(content) {
                return Err(format!("call JSON in content: {leak}"));
            }
        }
        Expect::Auto { tools, single } => {
            if has_calls {
                check_tool_calls(message, tools, None, *single)?;
                if finish != "tool_calls" {
                    return Err(format!("finish_reason {finish:?}, expected tool_calls"));
                }
            } else if finish == "tool_calls" {
                return Err("finish_reason tool_calls without tool_calls".into());
            }
            if let Some(leak) = leaked_call(content) {
                return Err(format!("call JSON leaked into content: {leak}"));
            }
        }
        Expect::JsonSchema(schema) => {
            if finish != "stop" {
                return Err(format!("finish_reason {finish:?}, expected stop"));
            }
            let v: Value = serde_json::from_str(content)
                .map_err(|e| format!("content is not JSON ({e}): {content}"))?;
            validate(schema, &v, "content")?;
        }
        Expect::JsonObject => {
            if finish != "stop" {
                return Err(format!("finish_reason {finish:?}, expected stop"));
            }
            let v: Value = serde_json::from_str(content)
                .map_err(|e| format!("content is not JSON ({e}): {content}"))?;
            if !v.is_object() {
                return Err(format!("content is not a JSON object: {content}"));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The server and a minimal HTTP/1.1 client.

fn free_addr() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

/// `turbine-server` on the real weights; killed on drop.
struct LabServer {
    child: Child,
    addr: SocketAddr,
    _config: TempFile,
}

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl LabServer {
    fn start(model_dir: &Path) -> LabServer {
        LabServer::start_with(model_dir, "")
    }

    /// [`LabServer::start`] with `extra` appended to the config (a `structured_output`
    /// section, say).
    fn start_with(model_dir: &Path, extra: &str) -> LabServer {
        let addr = free_addr();
        let config = TempFile(
            std::env::temp_dir().join(format!("turbine-lab-openai-{}.yaml", std::process::id())),
        );
        std::fs::write(
            &config.0,
            format!(
                "server:\n  listen: {addr}\nmodel:\n  path: {}\n  served_name: {SERVED_NAME}\n\
                 execution:\n  backend: hip\n{extra}",
                model_dir.display()
            ),
        )
        .unwrap();
        // The kernel library comes from TURBINE_KERNEL_LIBRARY (set by the lab Job). The server
        // log goes straight to the test output, so a long run never blocks on a full pipe.
        let child = Command::new(env!("CARGO_BIN_EXE_turbine-server"))
            .arg("--config")
            .arg(&config.0)
            .env("RUST_LOG", "info")
            .env("RUST_BACKTRACE", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn turbine-server");
        let mut server = LabServer {
            child,
            addr,
            _config: config,
        };
        server.wait_until_ready();
        server
    }

    fn wait_until_ready(&mut self) {
        let started = Instant::now();
        while started.elapsed() < READY_LIMIT {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("turbine-server exited before /ready ({status}); see its log above");
            }
            if TcpStream::connect(self.addr).is_ok()
                && request(self.addr, "GET", "/ready", None).0 == 200
            {
                return;
            }
            std::thread::sleep(POLL);
        }
        panic!("turbine-server not ready within {READY_LIMIT:?}");
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

/// Reads one series' value from the server's `/metrics` text.
fn metric_value(addr: SocketAddr, series: &str) -> f64 {
    let (_, text) = request(addr, "GET", "/metrics", None);
    text.lines()
        .find_map(|l| l.strip_prefix(series)?.trim().parse().ok())
        .unwrap_or(0.0)
}

/// P5p S-10 (lab): the four JSON cases at temperature 0, the verified jump-forward on and off
/// — the same token ids (a divergence excused only at a near-tie under the golden token
/// rule), logprobs within the strict golden bound, schema-valid content, and at most 0.90 ×
/// the forward steps and 1.00 × the wall time summed over the four (re-bound to the measured
/// Llama workload, user decision 2026-10-06 A: short forced runs on its merged-key tokenizer).
#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn json_schema_jump_forward_matches() {
    if !require_backend("hip") {
        return;
    }
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let tolerance: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/golden/llama-3.2-3b-instruct/tolerance.json"
        ))
        .expect("the Llama tolerance"),
    )
    .expect("tolerance JSON");
    let likely = tolerance["max_abs_logprob_diff_likely"].as_f64().unwrap();
    let floor = tolerance["likely_logprob_floor"].as_f64().unwrap();
    let tail = tolerance["max_abs_logprob_diff_tail"].as_f64().unwrap();

    // One server at a time: each sizes its KV pool to the whole card.
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let cases: Vec<ToolRequest> = load_fixture()
        .into_iter()
        .filter(|c| c.id.starts_with("json_"))
        .collect();
    assert_eq!(cases.len(), 4, "the fixture's four JSON cases");
    let body_of = |case: &ToolRequest| {
        let mut body = case.request.clone();
        body["model"] = json!(SERVED_NAME);
        body["temperature"] = json!(0.0);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(5);
        body
    };
    // Every case on one server (after one warm-up request): responses, wall times, and the
    // accepted forced tokens each case added.
    let serve = |extra: &str| {
        let server = LabServer::start_with(&model_dir, extra);
        let warm = body_of(&cases[0]);
        let _ = request(
            server.addr,
            "POST",
            "/v1/chat/completions",
            Some(&warm.to_string()),
        );
        let accepted = r#"turbine_jump_forward_tokens_total{outcome="accepted"}"#;
        cases
            .iter()
            .map(|case| {
                let before = metric_value(server.addr, accepted);
                let started = Instant::now();
                let (status, text) = request(
                    server.addr,
                    "POST",
                    "/v1/chat/completions",
                    Some(&body_of(case).to_string()),
                );
                let wall = started.elapsed().as_secs_f64();
                assert_eq!(status, 200, "{}: {text}", case.id);
                let response: Value = serde_json::from_str(&text).expect("response JSON");
                (response, wall, metric_value(server.addr, accepted) - before)
            })
            .collect::<Vec<_>>()
    };
    let on = serve("");
    let off = serve("structured_output:\n  jump_forward: false\n");

    let (mut steps_on_sum, mut steps_off_sum) = (0.0f64, 0.0f64);
    let (mut time_on_sum, mut time_off_sum) = (0.0f64, 0.0f64);
    for (case, ((jumped, time_on, accepted), (masked, time_off, accepted_off))) in
        cases.iter().zip(on.iter().zip(&off))
    {
        let expect = expectation(&case.request).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(*accepted_off, 0.0, "{}: the masked server jumped", case.id);
        check_response(&expect, jumped).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        check_response(&expect, masked).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let content_on = jumped["choices"][0]["logprobs"]["content"]
            .as_array()
            .expect("logprobs.content (on)");
        let content_off = masked["choices"][0]["logprobs"]["content"]
            .as_array()
            .expect("logprobs.content (off)");
        let margin = tolerance["margin_nats"].as_f64().unwrap();
        let mut diverged = false;
        for (i, (a, b)) in content_on.iter().zip(content_off).enumerate() {
            let (lp_on, lp_off) = (
                a["logprob"].as_f64().unwrap(),
                b["logprob"].as_f64().unwrap(),
            );
            if a["token"] == b["token"] {
                let bound = if lp_off > floor { likely } else { tail };
                assert!(
                    (lp_on - lp_off).abs() <= bound,
                    "{}: logprob {i} differs by {} ({lp_on} vs {lp_off})",
                    case.id,
                    (lp_on - lp_off).abs()
                );
                continue;
            }
            // A divergence is excused only at a near-tie: the masked run's top two candidates
            // within `margin_nats` of each other (the golden token rule); the rest of the two
            // outputs then follow different prefixes and are not compared.
            let top = b["top_logprobs"].as_array().expect("top_logprobs");
            assert!(
                top.len() >= 2
                    && top[0]["logprob"].as_f64().unwrap() - top[1]["logprob"].as_f64().unwrap()
                        <= margin,
                "{}: {i}: tokens differ ({} vs {}) off a near-tie",
                case.id,
                a["token"],
                b["token"]
            );
            println!("{}: excused near-tie divergence at token {i}", case.id);
            diverged = true;
            break;
        }
        if !diverged {
            assert_eq!(
                content_on.len(),
                content_off.len(),
                "{}: token counts",
                case.id
            );
        }
        // Forward steps: the masked run decodes one token per step; a verified step yields
        // its accepted forced tokens plus one, so the jump-forward run takes
        // `completion_tokens - accepted` steps.
        let tokens_on = jumped["usage"]["completion_tokens"].as_f64().unwrap();
        let tokens_off = masked["usage"]["completion_tokens"].as_f64().unwrap();
        let steps_on = tokens_on - accepted;
        steps_on_sum += steps_on;
        steps_off_sum += tokens_off;
        time_on_sum += time_on;
        time_off_sum += time_off;
        println!(
            "{}: tokens={tokens_off} accepted={accepted} steps on={steps_on} off={tokens_off} \
             ({:.3} steps/token), wall on={time_on:.2}s off={time_off:.2}s",
            case.id,
            steps_on / tokens_on.max(1.0)
        );
    }
    println!(
        "jump_forward totals: steps on={steps_on_sum} off={steps_off_sum}, \
         wall on={time_on_sum:.2}s off={time_off_sum:.2}s"
    );
    assert!(
        steps_on_sum <= 0.90 * steps_off_sum,
        "forward steps on={steps_on_sum} must be at most 0.90 x off={steps_off_sum}"
    );
    assert!(
        time_on_sum <= 1.00 * time_off_sum,
        "wall time on={time_on_sum} must be at most 1.00 x off={time_off_sum}"
    );
}

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn tools_and_json_schema() {
    if !require_backend("hip") {
        return;
    }
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let cases = load_fixture();
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let server = LabServer::start(&model_dir);

    let mut failures = Vec::new();
    for case in &cases {
        let expect = expectation(&case.request).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let mut body = case.request.clone();
        body["model"] = json!(SERVED_NAME);
        body["temperature"] = json!(0.0);
        let (status, text) = request(
            server.addr,
            "POST",
            "/v1/chat/completions",
            Some(&body.to_string()),
        );
        let verdict = if status != 200 {
            Err(format!("HTTP {status}: {text}"))
        } else {
            serde_json::from_str::<Value>(&text)
                .map_err(|e| format!("response is not JSON ({e}): {text}"))
                .and_then(|response| {
                    let message = &response["choices"][0]["message"];
                    println!(
                        "{}: finish_reason={} content={} tool_calls={}",
                        case.id,
                        response["choices"][0]["finish_reason"],
                        message["content"],
                        message["tool_calls"]
                    );
                    check_response(&expect, &response)
                })
        };
        match verdict {
            Ok(()) => println!("{}: ok", case.id),
            Err(e) => {
                println!("{}: FAIL {e}", case.id);
                failures.push(format!("{}: {e}", case.id));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} requests failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// The `host:port` of `TURBINE_LAB_URL` (`http://<host>:<port>`, an already running lab
/// server); panics when it is unset or not of that shape (a lab test never skips).
fn lab_url_addr() -> SocketAddr {
    let url = std::env::var("TURBINE_LAB_URL").unwrap_or_else(|_| {
        panic!("TURBINE_LAB_URL is not set; point it at the served model, e.g. http://192.168.10.203:18000")
    });
    let rest = url
        .strip_prefix("http://")
        .unwrap_or_else(|| panic!("TURBINE_LAB_URL={url}: only http:// URLs"));
    rest.trim_end_matches('/')
        .parse()
        .unwrap_or_else(|e| panic!("TURBINE_LAB_URL={url}: not http://<ip>:<port> ({e})"))
}

/// One chat completion against `addr`; the response JSON (panics on a non-200 answer).
fn lab_chat(addr: SocketAddr, body: &Value) -> Value {
    let (status, text) = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&body.to_string()),
    );
    assert_eq!(status, 200, "{text}");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("response is not JSON ({e}): {text}"))
}

/// P7 S-6: the `mistral` tool round trip on a served Mistral-7B-Instruct-v0.3
/// (`scripts/lab-serve.sh novanas scripts/lab/phase7-novanas-mistral-7b.yaml`, then
/// `TURBINE_LAB_URL=http://192.168.10.203:18000 scripts/remote-cargo.sh test -p turbine-server
/// --test lab_openai mistral_tool_call_round_trip -- --ignored`). A `tools` request with
/// `tool_choice: "required"` returns exactly one call, `get_weather`, whose arguments are a JSON
/// object with a `location`; sending that call back as the assistant turn plus a `tool` result
/// (the call's id: the v0.3 template refuses any id that is not 9 characters) yields a
/// non-empty assistant `content`. Breaks if the format's grammar, parser or call ids, or the
/// rendering of `tool_calls` and `tool` messages through Mistral's template, are wrong.
#[test]
#[ignore = "lab: needs a served Mistral-7B-Instruct-v0.3 at TURBINE_LAB_URL"]
fn mistral_tool_call_round_trip() {
    let addr = lab_url_addr();
    let (status, models) = request(addr, "GET", "/v1/models", None);
    assert_eq!(status, 200, "{models}");
    let models: Value = serde_json::from_str(&models).expect("models JSON");
    let model = models["data"][0]["id"]
        .as_str()
        .expect("a served model")
        .to_string();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city, e.g. Paris"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }]);
    let user = json!({"role": "user", "content": "What is the weather like in Paris today?"});
    let first = lab_chat(
        addr,
        &json!({
            "model": model, "messages": [user], "tools": tools, "tool_choice": "required",
            "temperature": 0.0, "max_tokens": 128
        }),
    );
    println!("first: {first}");
    let choice = &first["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{first}");
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("no tool_calls: {first}"));
    assert_eq!(calls.len(), 1, "{first}");
    let call = &calls[0];
    assert_eq!(call["type"], "function", "{call}");
    assert_eq!(call["function"]["name"], "get_weather", "{call}");
    let arguments: Value = serde_json::from_str(
        call["function"]["arguments"]
            .as_str()
            .expect("arguments string"),
    )
    .unwrap_or_else(|e| panic!("arguments are not JSON ({e}): {call}"));
    assert!(
        arguments["location"]
            .as_str()
            .is_some_and(|l| !l.is_empty()),
        "arguments without a location: {arguments}"
    );
    let id = call["id"].as_str().expect("call id");

    let second = lab_chat(
        addr,
        &json!({
            "model": model,
            "messages": [
                user,
                {"role": "assistant", "content": null, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": id,
                 "content": "{\"temperature\": 18, \"unit\": \"celsius\", \"sky\": \"sunny\"}"}
            ],
            "tools": tools, "temperature": 0.0, "max_tokens": 128
        }),
    );
    println!("second: {second}");
    let message = &second["choices"][0]["message"];
    let content = message["content"].as_str().unwrap_or("");
    assert!(
        !content.trim().is_empty(),
        "no assistant content after the tool result: {second}"
    );
    assert!(
        message["tool_calls"]
            .as_array()
            .is_none_or(|c| c.is_empty()),
        "{second}"
    );
}

/// P7 S-9: the Harmony channels against a served gpt-oss (`scripts/lab-serve.sh novanas
/// scripts/lab/phase7-novanas-gpt-oss-20b.yaml`, then `TURBINE_LAB_URL=http://192.168.10.203:18000
/// scripts/remote-cargo.sh test -p turbine-server --test lab_openai harmony_channels -- --ignored`).
/// A plain question yields non-empty `reasoning_content` and `content`; a `tool_choice:
/// "required"` request yields one `get_weather` call with JSON arguments; with the same seed
/// and `temperature: 0` the streamed `reasoning_content` / `content` deltas concatenate to the
/// non-streamed text; an `n: 2` stream carries each choice's own channels; and
/// `reasoning_effort: "extreme"` is a 400 naming the field. Breaks if the marker emission
/// (P7 decision "Harmony marker visibility"), the channel parser, the tool parser or the
/// `reasoning_content` plumbing is wrong. Runs in Task 26.
#[test]
#[ignore = "lab: needs a served gpt-oss at TURBINE_LAB_URL"]
fn harmony_channels() {
    let addr = lab_url_addr();
    let (status, models) = request(addr, "GET", "/v1/models", None);
    assert_eq!(status, 200, "{models}");
    let models: Value = serde_json::from_str(&models).expect("models JSON");
    let model = models["data"][0]["id"]
        .as_str()
        .expect("a served model")
        .to_string();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city, e.g. Paris"}
                },
                "required": ["location"]
            }
        }
    }]);
    let user = json!({"role": "user", "content": "What is the capital of France? Answer briefly."});

    // A plain question: both channels non-empty.
    let base = json!({
        "model": model, "messages": [user.clone()], "temperature": 0.0, "seed": 7,
        "max_tokens": 256
    });
    let answer = lab_chat(addr, &base);
    let message = &answer["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap_or_default();
    let content = message["content"].as_str().unwrap_or_default();
    assert!(!reasoning.trim().is_empty(), "no reasoning: {answer}");
    assert!(!content.trim().is_empty(), "no content: {answer}");
    assert!(
        !content.contains("<|channel|>"),
        "channel markers leaked into content: {answer}"
    );

    // The same request streamed: the deltas concatenate to the non-streamed text.
    let mut streamed_body = base.clone();
    streamed_body["stream"] = json!(true);
    let (_, sse) = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&streamed_body.to_string()),
    );
    let mut streamed_reasoning = String::new();
    let mut streamed_content = String::new();
    for line in sse.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let delta = &chunk["choices"][0]["delta"];
        if let Some(text) = delta["reasoning_content"].as_str() {
            streamed_reasoning.push_str(text);
        }
        if let Some(text) = delta["content"].as_str() {
            streamed_content.push_str(text);
        }
    }
    assert_eq!(streamed_reasoning, reasoning, "streamed reasoning differs");
    assert_eq!(streamed_content, content, "streamed content differs");

    // `tool_choice: "required"`: one get_weather call with JSON arguments.
    let call_body = json!({
        "model": model,
        "messages": [json!({"role": "user",
                            "content": "What is the weather like in Paris today?"})],
        "tools": tools, "tool_choice": "required", "temperature": 0.0, "max_tokens": 256
    });
    let called = lab_chat(addr, &call_body);
    let choice = &called["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{called}");
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("no tool_calls: {called}"));
    assert_eq!(calls.len(), 1, "{called}");
    assert_eq!(calls[0]["function"]["name"], "get_weather", "{called}");
    serde_json::from_str::<Value>(
        calls[0]["function"]["arguments"]
            .as_str()
            .expect("arguments string"),
    )
    .unwrap_or_else(|e| panic!("arguments are not JSON ({e}): {called}"));

    // `n: 2`: both choices stream their own channels.
    let mut both = base.clone();
    both["n"] = json!(2);
    both["stream"] = json!(true);
    let (_, sse) = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&both.to_string()),
    );
    let mut channels: BTreeMap<u32, (String, String)> = BTreeMap::new();
    for line in sse.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let Some(choice) = chunk["choices"][0]["index"].as_u64() else {
            continue;
        };
        let entry = channels
            .entry(u32::try_from(choice).expect("u32 choice"))
            .or_default();
        if let Some(text) = chunk["choices"][0]["delta"]["reasoning_content"].as_str() {
            entry.0.push_str(text);
        }
        if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str() {
            entry.1.push_str(text);
        }
    }
    assert_eq!(channels.len(), 2, "expected both choices: {channels:?}");
    for (choice, (reasoning, content)) in &channels {
        assert!(
            !reasoning.trim().is_empty(),
            "choice {choice}: no reasoning"
        );
        assert!(!content.trim().is_empty(), "choice {choice}: no content");
    }

    // `reasoning_effort: "extreme"` is a 400 naming the field.
    let mut extreme = base.clone();
    extreme["reasoning_effort"] = json!("extreme");
    let (status, text) = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&extreme.to_string()),
    );
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("reasoning_effort"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Always-run checks of the fixture and of the response checks.

#[test]
fn tool_requests_fixture_is_well_formed() {
    let cases = load_fixture();
    let mut ids = BTreeSet::new();
    let (mut required, mut named, mut auto, mut schema, mut object) = (0, 0, 0, 0, 0);
    for case in &cases {
        assert!(ids.insert(case.id.clone()), "duplicate id {}", case.id);
        assert!(
            case.request.get("model").is_none(),
            "{}: the test sets model",
            case.id
        );
        let max_tokens = case.request["max_tokens"].as_u64();
        assert!(
            max_tokens.is_some_and(|n| (64..=512).contains(&n)),
            "{}: max_tokens must bound the generation (64..=512), got {max_tokens:?}",
            case.id
        );
        let messages = case.request["messages"].as_array();
        assert!(
            messages.is_some_and(|m| m.last().is_some_and(|l| l["role"] == "user")),
            "{}: messages must end with a user turn",
            case.id
        );
        match expectation(&case.request).unwrap_or_else(|e| panic!("{}: {e}", case.id)) {
            Expect::ToolCalls {
                tools, named: n, ..
            } => {
                for (name, s) in &tools {
                    check_supported(s, name).unwrap_or_else(|e| panic!("{}: {e}", case.id));
                }
                if n.is_some() {
                    named += 1
                } else {
                    required += 1
                }
            }
            Expect::Auto { tools, .. } => {
                for (name, s) in &tools {
                    check_supported(s, name).unwrap_or_else(|e| panic!("{}: {e}", case.id));
                }
                auto += 1;
            }
            Expect::JsonSchema(s) => {
                check_supported(&s, "schema").unwrap_or_else(|e| panic!("{}: {e}", case.id));
                schema += 1;
            }
            Expect::JsonObject => object += 1,
        }
    }
    assert!(required >= 2, "required cases: {required}");
    assert!(named >= 2, "named cases: {named}");
    assert!(auto >= 2, "auto cases: {auto}");
    assert!(schema >= 2, "json_schema cases: {schema}");
    assert!(object >= 1, "json_object cases: {object}");
}

fn weather_tools() -> BTreeMap<String, Value> {
    BTreeMap::from([(
        "get_weather".to_string(),
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city", "unit"],
            "additionalProperties": false
        }),
    )])
}

fn chat(content: Value, tool_calls: Value, finish: &str) -> Value {
    json!({"choices": [{"index": 0, "finish_reason": finish,
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]})
}

fn call(id: &str, name: &str, arguments: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

#[test]
fn schema_validator_accepts_and_rejects() {
    let schema = json!({
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            "maybe": {"type": ["string", "null"]}
        },
        "required": ["n"],
        "additionalProperties": false
    });
    for ok in [
        json!({"n": 3}),
        json!({"n": 3.0, "tags": ["a", "b"], "maybe": null}),
        json!({"n": -1, "maybe": "x"}),
    ] {
        validate(&schema, &ok, "v").unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in [
        json!({}),
        json!({"n": 1.5}),
        json!({"n": "3"}),
        json!({"n": 1, "tags": ["c"]}),
        json!({"n": 1, "tags": "a"}),
        json!({"n": 1, "maybe": 2}),
        json!({"n": 1, "extra": true}),
        json!([1]),
    ] {
        assert!(validate(&schema, &bad, "v").is_err(), "{bad} accepted");
    }
    let unsupported = json!({"type": "string", "pattern": "^a$"});
    let e = validate(&unsupported, &json!("a"), "v").unwrap_err();
    assert!(e.contains("pattern"), "{e}");
}

#[test]
fn response_checks_reject_malformed_answers() {
    let required = Expect::ToolCalls {
        tools: weather_tools(),
        named: None,
        single: false,
    };
    let good_call = call("c0", "get_weather", r#"{"city":"Paris","unit":"celsius"}"#);
    check_response(
        &required,
        &chat(Value::Null, json!([good_call]), "tool_calls"),
    )
    .unwrap();
    for (what, response) in [
        ("no calls", chat(json!("It is sunny."), Value::Null, "stop")),
        (
            "wrong finish",
            chat(Value::Null, json!([good_call]), "stop"),
        ),
        (
            "unknown tool",
            chat(
                Value::Null,
                json!([call("c0", "get_time", "{}")]),
                "tool_calls",
            ),
        ),
        (
            "arguments not JSON",
            chat(
                Value::Null,
                json!([call("c0", "get_weather", "{\"city\":")]),
                "tool_calls",
            ),
        ),
        (
            "arguments off schema",
            chat(
                Value::Null,
                json!([call(
                    "c0",
                    "get_weather",
                    r#"{"city":"Paris","unit":"kelvin"}"#
                )]),
                "tool_calls",
            ),
        ),
        (
            "repeated id",
            chat(Value::Null, json!([good_call, good_call]), "tool_calls"),
        ),
    ] {
        assert!(
            check_response(&required, &response).is_err(),
            "{what} accepted"
        );
    }

    let named = Expect::ToolCalls {
        tools: weather_tools(),
        named: Some("get_time".into()),
        single: true,
    };
    assert!(
        check_response(&named, &chat(Value::Null, json!([good_call]), "tool_calls")).is_err(),
        "a call to another tool than the named one was accepted"
    );

    let auto = Expect::Auto {
        tools: weather_tools(),
        single: false,
    };
    check_response(&auto, &chat(json!("The sea is calm."), Value::Null, "stop")).unwrap();
    check_response(&auto, &chat(Value::Null, json!([good_call]), "tool_calls")).unwrap();
    let tag_leak = format!(
        "{}get_weather(city='Paris')",
        llama_format().special_tokens()[0].text
    );
    for leak in [
        r#"{"name": "get_weather", "parameters": {"city": "Paris", "unit": "celsius"}}"#,
        r#"Sure: {"name":"get_weather","parameters":{"city":"Paris"}}"#,
        tag_leak.as_str(),
    ] {
        assert!(
            check_response(&auto, &chat(json!(leak), Value::Null, "stop")).is_err(),
            "leaked call accepted: {leak}"
        );
    }
    // JSON that is not call-shaped is ordinary content.
    check_response(
        &auto,
        &chat(
            json!(r#"Use {"city": "Paris"} as input."#),
            Value::Null,
            "stop",
        ),
    )
    .unwrap();

    let schema = Expect::JsonSchema(weather_tools()["get_weather"].clone());
    check_response(
        &schema,
        &chat(
            json!(r#"{"city":"Oslo","unit":"fahrenheit"}"#),
            Value::Null,
            "stop",
        ),
    )
    .unwrap();
    for (what, content, finish) in [
        ("truncated", r#"{"city":"Oslo","#, "length"),
        ("off schema", r#"{"city":"Oslo"}"#, "stop"),
        ("not JSON", "Oslo, fahrenheit", "stop"),
    ] {
        assert!(
            check_response(&schema, &chat(json!(content), Value::Null, finish)).is_err(),
            "{what} accepted"
        );
    }
    assert!(
        check_response(
            &Expect::JsonObject,
            &chat(json!("[1,2]"), Value::Null, "stop")
        )
        .is_err(),
        "a JSON array accepted as json_object"
    );
}
