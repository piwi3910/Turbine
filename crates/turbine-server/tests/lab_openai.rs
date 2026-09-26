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
        let addr = free_addr();
        let config = TempFile(
            std::env::temp_dir().join(format!("turbine-lab-openai-{}.yaml", std::process::id())),
        );
        std::fs::write(
            &config.0,
            format!(
                "server:\n  listen: {addr}\nmodel:\n  path: {}\n  served_name: {SERVED_NAME}\n\
                 execution:\n  backend: hip\n",
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

#[test]
#[ignore = "lab: needs the HIP backend, libturbine_hip.so and the Llama-3.2-3B weights"]
fn tools_and_json_schema() {
    if !require_backend("hip") {
        return;
    }
    let model_dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let cases = load_fixture();
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
