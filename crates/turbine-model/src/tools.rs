//! Tool calling for Llama-3.x (P2 S-18): the `llama3_json` tool-call parser, the llguidance
//! grammar that constrains `tool_choice` `required` / named output, and call ids.
//!
//! Llama-3.x emits a call as an optional `<|python_tag|>` followed by one or more
//! `{"name": …, "parameters": {…}}` objects separated by `;`. The parser turns that into
//! [`ToolCallOut`]s; the grammar makes the model produce exactly that shape, with each call's
//! `parameters` held to its tool's JSON schema, so a constrained call always parses.
use std::sync::Mutex;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use serde_json::{Map, Value};
pub use turbine_core::request::{ConstraintSpec, ToolCallOut};

use crate::ModelError;

/// The Llama-3 special token that may open a tool call.
pub const PYTHON_TAG: &str = "<|python_tag|>";

/// The `parser` label of `turbine_tool_calls_total` for [`Llama3JsonParser`].
pub const LLAMA3_JSON: &str = "llama3_json";

/// Length of the random part of a call id (`call_` + 24 alphanumerics).
const CALL_ID_RANDOM_LEN: usize = 24;

const ALPHANUMERIC: &[u8; 62] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// What a finished choice's text turned out to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolParse {
    /// The whole output is one or more tool calls, in order.
    Calls(Vec<ToolCallOut>),
    /// Not a tool call: the text, unchanged.
    Content(String),
}

/// Extracts tool calls from a model's output text (`model.tool_call_parser`).
pub trait ToolCallParser: Send + Sync {
    fn parse(&self, text: &str) -> ToolParse;
}

/// `tool_choice` of a request (OpenAI `none`, `auto`, `required` or a named function).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

impl ToolChoice {
    fn as_str(&self) -> &str {
        match self {
            ToolChoice::None => "none",
            ToolChoice::Auto => "auto",
            ToolChoice::Required => "required",
            ToolChoice::Named(name) => name,
        }
    }
}

/// The `llama3_json` parser: an optional `<|python_tag|>`, then — from the first `{` — one or
/// more `{"name": <string>, "parameters": <object>}` objects separated by `;` (whitespace and a
/// trailing `;` allowed), and nothing else. Anything else is [`ToolParse::Content`] unchanged.
/// Each call's `arguments` is the compact JSON of its `parameters` (key order kept).
pub struct Llama3JsonParser {
    /// Source of call ids.
    rng: Mutex<ChaCha8Rng>,
}

impl std::fmt::Debug for Llama3JsonParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Llama3JsonParser").finish_non_exhaustive()
    }
}

impl Default for Llama3JsonParser {
    fn default() -> Self {
        Llama3JsonParser::new()
    }
}

impl Llama3JsonParser {
    /// A parser whose call ids come from an OS-seeded stream.
    pub fn new() -> Llama3JsonParser {
        Llama3JsonParser {
            rng: Mutex::new(ChaCha8Rng::from_os_rng()),
        }
    }

    /// A parser with reproducible call ids (tests, simulations).
    pub fn seeded(seed: u64) -> Llama3JsonParser {
        Llama3JsonParser {
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(seed)),
        }
    }
}

impl ToolCallParser for Llama3JsonParser {
    fn parse(&self, text: &str) -> ToolParse {
        let Some(found) = parse_llama3_calls(text) else {
            return ToolParse::Content(text.to_string());
        };
        // A poisoned lock only means another parse panicked mid-draw; the stream is still valid.
        let mut rng = self.rng.lock().unwrap_or_else(|p| p.into_inner());
        let calls = found
            .into_iter()
            .zip(0u32..)
            .map(|((name, arguments), index)| ToolCallOut {
                index,
                id: new_call_id(&mut *rng),
                name,
                arguments,
            })
            .collect();
        ToolParse::Calls(calls)
    }
}

/// `(name, compact arguments)` of every call when `text` is entirely Llama-3 JSON calls.
fn parse_llama3_calls(text: &str) -> Option<Vec<(String, String)>> {
    let after_tag = match text.find(PYTHON_TAG) {
        Some(at) => &text[at + PYTHON_TAG.len()..],
        None => text,
    };
    let mut rest = &after_tag[after_tag.find('{')?..];
    let mut calls = Vec::new();
    loop {
        let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let value = values.next()?.ok()?;
        let consumed = values.byte_offset();
        calls.push(call_from_value(value)?);
        rest = rest[consumed..].trim_start();
        if let Some(after) = rest.strip_prefix(';') {
            rest = after.trim_start();
        } else if !rest.is_empty() {
            return None;
        }
        if rest.is_empty() {
            return Some(calls);
        }
        if !rest.starts_with('{') {
            return None;
        }
    }
}

/// A call object has exactly a string `name` and an object `parameters`.
fn call_from_value(value: Value) -> Option<(String, String)> {
    let Value::Object(mut object) = value else {
        return None;
    };
    if object.len() != 2 {
        return None;
    }
    let Some(Value::String(name)) = object.remove("name") else {
        return None;
    };
    let Some(parameters @ Value::Object(_)) = object.remove("parameters") else {
        return None;
    };
    Some((name, parameters.to_string()))
}

/// A fresh call id: `call_` followed by 24 characters drawn uniformly from `[A-Za-z0-9]`.
pub fn new_call_id(rng: &mut impl RngCore) -> String {
    let mut id = String::with_capacity(5 + CALL_ID_RANDOM_LEN);
    id.push_str("call_");
    let mut drawn = 0;
    let mut bytes = [0u8; 32];
    while drawn < CALL_ID_RANDOM_LEN {
        rng.fill_bytes(&mut bytes);
        // Rejection sampling: 248 = 4 × 62, so accepted bytes map uniformly onto the alphabet.
        for &b in bytes.iter().filter(|&&b| b < 248) {
            if drawn == CALL_ID_RANDOM_LEN {
                break;
            }
            id.push(char::from(
                ALPHANUMERIC[usize::from(b) % ALPHANUMERIC.len()],
            ));
            drawn += 1;
        }
    }
    id
}

/// One function tool of a request: its name and the JSON schema of its parameters.
struct FunctionTool<'a> {
    name: &'a str,
    parameters: Option<&'a Map<String, Value>>,
}

/// A tool definition or `tool_choice` the grammar cannot be built from: a
/// [`ModelError::Constraint`] (the server answers 400 `invalid_json_schema`) naming the field.
fn invalid(field: impl Into<String>, value: impl Into<String>, supported: &str) -> ModelError {
    ModelError::Constraint(format!(
        "unsupported {} = {}; supported: {supported}",
        field.into(),
        value.into()
    ))
}

/// OpenAI function names: 1–64 of `[A-Za-z0-9_-]`, so a name needs no escaping in JSON or Lark.
fn valid_function_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn function_tools(tools: &[Value]) -> Result<Vec<FunctionTool<'_>>, ModelError> {
    let mut out: Vec<FunctionTool<'_>> = Vec::with_capacity(tools.len());
    for (i, tool) in tools.iter().enumerate() {
        let kind = tool.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "function" {
            return Err(invalid(format!("tools[{i}].type"), kind, "function"));
        }
        let function = tool.get("function");
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !valid_function_name(name) {
            return Err(invalid(
                format!("tools[{i}].function.name"),
                name,
                "1 to 64 of [A-Za-z0-9_-]",
            ));
        }
        if out.iter().any(|t| t.name == name) {
            return Err(invalid(
                format!("tools[{i}].function.name"),
                name,
                "unique function names",
            ));
        }
        let parameters = match function.and_then(|f| f.get("parameters")) {
            None | Some(Value::Null) => None,
            Some(Value::Object(schema)) => Some(schema),
            Some(other) => {
                return Err(invalid(
                    format!("tools[{i}].function.parameters"),
                    other.to_string(),
                    "a JSON schema object",
                ));
            }
        };
        out.push(FunctionTool { name, parameters });
    }
    Ok(out)
}

/// The llguidance Lark grammar that constrains a `required` or named `tool_choice`: an optional
/// `<|python_tag|>`, then `{"name": "<tool>", "parameters": <that tool's JSON schema>}` for one
/// of the allowed tools (only the named one for [`ToolChoice::Named`]), repeated with `;`
/// separators when `parallel` (never for a named function: that is exactly one call). The
/// parameters use Llama's own separators (`", "`, `": "`) with no free whitespace; a tool
/// without `parameters` takes any object. `none` and `auto` are unconstrained and have no
/// grammar; an unknown named function or a malformed tool is a [`ModelError::Constraint`]
/// naming the field.
pub fn tool_call_grammar(
    tools: &[Value],
    choice: &ToolChoice,
    parallel: bool,
) -> Result<ConstraintSpec, ModelError> {
    let tools = function_tools(tools)?;
    if tools.is_empty() {
        return Err(invalid("tools", "[]", "at least one function tool"));
    }
    let allowed: Vec<&FunctionTool<'_>> = match choice {
        ToolChoice::None | ToolChoice::Auto => {
            return Err(invalid(
                "tool_choice",
                choice.as_str(),
                "required, a named function",
            ));
        }
        ToolChoice::Required => tools.iter().collect(),
        ToolChoice::Named(name) => {
            let Some(tool) = tools.iter().find(|t| t.name == name) else {
                let names: Vec<&str> = tools.iter().map(|t| t.name).collect();
                return Err(invalid(
                    "tool_choice.function.name",
                    name.as_str(),
                    &names.join(", "),
                ));
            };
            vec![tool]
        }
    };
    let repeat = parallel && matches!(choice, ToolChoice::Required);

    let mut grammar = String::from("start: <|python_tag|>? call");
    if repeat {
        grammar.push_str(" (\";\" call)*");
    }
    grammar.push('\n');
    let alternatives: Vec<String> = (0..allowed.len()).map(|i| format!("call_{i}")).collect();
    grammar.push_str(&format!("call: {}\n", alternatives.join(" | ")));
    for (i, tool) in allowed.iter().enumerate() {
        // The name is [A-Za-z0-9_-] only, so inside the Lark (JSON-syntax) string only the
        // quotes of the emitted JSON need escaping.
        grammar.push_str(&format!(
            "call_{i}: \"{{\\\"name\\\": \\\"{}\\\", \\\"parameters\\\": \" params_{i} \"}}\"\n",
            tool.name
        ));
        let mut schema = tool.parameters.cloned().unwrap_or_else(|| {
            let mut any_object = Map::new();
            any_object.insert("type".into(), "object".into());
            any_object
        });
        schema.insert(
            "x-guidance".into(),
            serde_json::json!({
                "whitespace_flexible": false,
                "item_separator": ", ",
                "key_separator": ": ",
            }),
        );
        grammar.push_str(&format!("params_{i}: %json {}\n", Value::Object(schema)));
    }
    Ok(ConstraintSpec::ToolCall {
        grammar_source: grammar,
    })
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;

    use super::*;

    fn is_call_id(id: &str) -> bool {
        id.strip_prefix("call_")
            .is_some_and(|rest| rest.len() == 24 && rest.bytes().all(|b| b.is_ascii_alphanumeric()))
    }

    fn calls(parse: ToolParse) -> Vec<ToolCallOut> {
        match parse {
            ToolParse::Calls(calls) => calls,
            ToolParse::Content(text) => panic!("expected calls, got content {text:?}"),
        }
    }

    #[test]
    fn llama3_json_parser() {
        let parser = Llama3JsonParser::seeded(7);

        // One call; `arguments` is the compact re-serialisation of `parameters`, keys in order.
        let one = calls(parser.parse(
            r#"{"name": "get_weather", "parameters": {"location": "Paris", "unit": "celsius"}}"#,
        ));
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].index, 0);
        assert_eq!(one[0].name, "get_weather");
        assert_eq!(one[0].arguments, r#"{"location":"Paris","unit":"celsius"}"#);
        assert!(is_call_id(&one[0].id), "{}", one[0].id);

        // Two `;`-separated calls, with whitespace around the separator and a trailing newline.
        let two = calls(parser.parse(concat!(
            r#"{"name": "a", "parameters": {"x": [1, 2]}} ; "#,
            "\n",
            r#"{"name": "b", "parameters": {}}"#,
            "\n"
        )));
        assert_eq!(two.len(), 2);
        assert_eq!((two[0].index, two[0].name.as_str()), (0, "a"));
        assert_eq!(two[0].arguments, r#"{"x":[1,2]}"#);
        assert_eq!((two[1].index, two[1].name.as_str()), (1, "b"));
        assert_eq!(two[1].arguments, "{}");
        assert_ne!(two[0].id, two[1].id);
        assert!(two.iter().all(|c| is_call_id(&c.id)));

        // Preceded by <|python_tag|>.
        let tagged = calls(parser.parse(
            r#"<|python_tag|>{"name": "search", "parameters": {"query": "rust ; \"llm\""}}"#,
        ));
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].name, "search");
        assert_eq!(tagged[0].arguments, r#"{"query":"rust ; \"llm\""}"#);

        // Content comes back unchanged: plain text, JSON that is not a call object, calls with
        // extra or missing keys or wrong types, trailing text, broken JSON.
        for text in [
            "The weather in Paris is sunny.",
            "  Hello!\n",
            r#"{"a": 1}"#,
            r#"{"name": "f"}"#,
            r#"{"name": "f", "parameters": {}, "extra": 1}"#,
            r#"{"name": 3, "parameters": {}}"#,
            r#"{"name": "f", "parameters": [1]}"#,
            r#"{"name": "f", "parameters": {}} and then some"#,
            r#"{"name": "f", "parameters": {}}; {"a": 1}"#,
            r#"{"name": "f", "parameters": {"#,
            "<|python_tag|>",
            "[1, 2]",
        ] {
            assert_eq!(
                parser.parse(text),
                ToolParse::Content(text.to_string()),
                "{text:?}"
            );
        }

        // Seeded parsers give reproducible ids; the trait object is usable across threads.
        let a = calls(Llama3JsonParser::seeded(1).parse(r#"{"name": "f", "parameters": {}}"#));
        let b = calls(Llama3JsonParser::seeded(1).parse(r#"{"name": "f", "parameters": {}}"#));
        assert_eq!(a[0].id, b[0].id);
        let boxed: Box<dyn ToolCallParser> = Box::new(Llama3JsonParser::new());
        assert!(matches!(
            boxed.parse(r#"{"name": "f", "parameters": {}}"#),
            ToolParse::Calls(_)
        ));
    }

    #[test]
    fn call_ids_are_24_alphanumerics() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ids: Vec<String> = (0..500).map(|_| new_call_id(&mut rng)).collect();
        assert!(ids.iter().all(|id| is_call_id(id)), "{ids:?}");
        let distinct: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(distinct.len(), ids.len());
        // Every class of the alphabet shows up.
        let all: String = ids.iter().map(|id| &id[5..]).collect();
        assert!(all.bytes().any(|b| b.is_ascii_uppercase()));
        assert!(all.bytes().any(|b| b.is_ascii_lowercase()));
        assert!(all.bytes().any(|b| b.is_ascii_digit()));
    }

    fn weather_tools() -> Vec<serde_json::Value> {
        vec![
            json!({"type": "function", "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }}),
            json!({"type": "function", "function": {"name": "get_time"}}),
        ]
    }

    fn grammar(spec: ConstraintSpec) -> String {
        match spec {
            ConstraintSpec::ToolCall { grammar_source } => grammar_source,
            other => panic!("expected a tool-call grammar, got {other:?}"),
        }
    }

    #[test]
    fn tool_grammar_shapes() {
        let tools = weather_tools();

        // `required`, parallel: every tool, `;`-separated repetition, optional python tag.
        let g = grammar(tool_call_grammar(&tools, &ToolChoice::Required, true).expect("grammar"));
        assert!(
            g.contains("start: <|python_tag|>? call (\";\" call)*"),
            "{g}"
        );
        assert!(g.contains("call: call_0 | call_1\n"), "{g}");
        assert!(
            g.contains(r#"call_0: "{\"name\": \"get_weather\", \"parameters\": " params_0 "}""#),
            "{g}"
        );
        assert!(g.contains(r#"call_1: "{\"name\": \"get_time\", \"parameters\": " params_1 "}""#));
        // Each tool's own schema, with Llama's separators and no free whitespace; a tool
        // without `parameters` takes any object.
        let params0 = schema_of(&g, "params_0");
        assert_eq!(params0["required"], json!(["location"]));
        assert_eq!(
            params0["x-guidance"],
            json!({"whitespace_flexible": false, "item_separator": ", ", "key_separator": ": "})
        );
        assert_eq!(schema_of(&g, "params_1")["type"], "object");

        // `required` without parallel calls: exactly one call.
        let g = grammar(tool_call_grammar(&tools, &ToolChoice::Required, false).expect("grammar"));
        assert!(g.contains("start: <|python_tag|>? call\n"), "{g}");

        // Named: only that tool, one call even when parallel calls are allowed.
        let g = grammar(
            tool_call_grammar(&tools, &ToolChoice::Named("get_time".into()), true)
                .expect("grammar"),
        );
        assert!(g.contains("start: <|python_tag|>? call\n"), "{g}");
        assert!(g.contains("call: call_0\n"), "{g}");
        assert!(
            g.contains(r#""{\"name\": \"get_time\", \"parameters\": ""#),
            "{g}"
        );
        assert!(!g.contains("get_weather"), "{g}");

        // Errors name the offending field.
        let err = |r: Result<ConstraintSpec, ModelError>| match r {
            Err(e @ ModelError::Constraint(_)) => e.to_string(),
            Err(e) => panic!("expected ModelError::Constraint, got {e:?}"),
            Ok(spec) => panic!("expected an error, got {spec:?}"),
        };
        let e = err(tool_call_grammar(
            &tools,
            &ToolChoice::Named("nope".into()),
            false,
        ));
        assert!(e.contains("tool_choice.function.name = nope"), "{e}");
        assert!(e.contains("get_weather, get_time"), "{e}");
        for choice in [ToolChoice::None, ToolChoice::Auto] {
            let e = err(tool_call_grammar(&tools, &choice, false));
            assert!(e.contains("tool_choice"), "{e}");
        }
        let e = err(tool_call_grammar(&[], &ToolChoice::Required, false));
        assert!(e.contains("tools"), "{e}");
        let bad_name = [json!({"type": "function", "function": {"name": "a b\"c"}})];
        let e = err(tool_call_grammar(&bad_name, &ToolChoice::Required, false));
        assert!(e.contains("tools[0].function.name"), "{e}");
        let not_function = [json!({"type": "retrieval"})];
        let e = err(tool_call_grammar(
            &not_function,
            &ToolChoice::Required,
            false,
        ));
        assert!(e.contains("tools[0].type"), "{e}");
        let bad_params = [json!({"type": "function", "function": {"name": "f", "parameters": 3}})];
        let e = err(tool_call_grammar(&bad_params, &ToolChoice::Required, false));
        assert!(e.contains("tools[0].function.parameters"), "{e}");
        let dup = [tools[1].clone(), tools[1].clone()];
        let e = err(tool_call_grammar(&dup, &ToolChoice::Required, false));
        assert!(e.contains("tools[1].function.name = get_time"), "{e}");
    }

    /// The JSON schema after `<rule>: %json ` on its own line of `grammar`.
    fn schema_of(grammar: &str, rule: &str) -> serde_json::Value {
        let prefix = format!("{rule}: %json ");
        let line = grammar
            .lines()
            .find_map(|l| l.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("no {rule} in {grammar}"));
        serde_json::from_str(line).expect("schema is JSON")
    }
}
