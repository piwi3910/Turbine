//! The `llama3_json` tool-call format (Llama-3.x, P2 S-18).
//!
//! Llama-3.x emits a call as an optional `<|python_tag|>` followed by one or more
//! `{"name": …, "parameters": {…}}` objects separated by `;`. [`Llama3JsonParser`] turns that
//! into [`ToolCallOut`]s; the grammar makes the model produce exactly that shape, with each
//! call's `parameters` held to its tool's JSON schema, so a constrained call always parses. In
//! `auto` mode an output opens like a call when its first token is `<|python_tag|>` or its first
//! non-whitespace character is `{`.
use std::sync::Mutex;

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use serde_json::{Map, Value};
use turbine_core::registry::Module;

use super::{BoundTokens, ConstraintSpec, Opening, SpecialToken, ToolFormat};
use crate::ModelError;
use crate::structured::{JSON_MAX_WHITESPACE, with_json_options};
use crate::tools::{
    FunctionTool, ToolCallOut, ToolCallParser, ToolChoice, ToolParse, function_tools, invalid,
    new_call_id,
};

/// The Llama-3 special token that may open a tool call (optional: a call may open with `{`).
pub const PYTHON_TAG: &str = "<|python_tag|>";

/// The format's registry name (and the `parser` label of `turbine_tool_calls_total`).
pub const LLAMA3_JSON: &str = "llama3_json";

/// The `llama3_json` format.
pub struct Llama3Json;

impl Module for Llama3Json {
    fn name(&self) -> &'static str {
        LLAMA3_JSON
    }
}

impl ToolFormat for Llama3Json {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        &[SpecialToken {
            text: PYTHON_TAG,
            required: false,
        }]
    }

    /// An optional `<|python_tag|>`, then `{"name": "<tool>", "parameters": <that tool's JSON
    /// schema>}` for one of the allowed tools (only the named one for [`ToolChoice::Named`]),
    /// repeated with `;` separators when `parallel` (never for a named function: that is
    /// exactly one call). The call envelope is written the way Llama writes it (`", "`,
    /// `": "`); the parameters take the bounded natural JSON whitespace of
    /// [`crate::structured::json_options`]; a tool without `parameters` takes any object.
    /// `auto` is `start: text | calls`: free text whose first non-whitespace character (after
    /// at most [`JSON_MAX_WHITESPACE`]) is not `{` and that does not open with
    /// `<|python_tag|>`, or the `required` calls, so a call the model starts always names a
    /// listed tool and satisfies its schema.
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        let tools = function_tools(tools)?;
        if tools.is_empty() {
            return Err(invalid("tools", "[]", "at least one function tool"));
        }
        let allowed: Vec<&FunctionTool<'_>> = match choice {
            ToolChoice::None => {
                return Err(invalid(
                    "tool_choice",
                    choice.as_str(),
                    "auto, required, a named function",
                ));
            }
            ToolChoice::Auto | ToolChoice::Required => tools.iter().collect(),
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
        let repeat = parallel && matches!(choice, ToolChoice::Auto | ToolChoice::Required);

        let mut grammar = match choice {
            // Free text (first non-whitespace character not `{`; the python tag is a special
            // token, never text) or the calls of `required`.
            ToolChoice::Auto => format!(
                "start: text | calls\n\
                 text: /[\\x20\\x0A\\x0D\\x09]{{0,{JSON_MAX_WHITESPACE}}}\
                 [^{{\\x20\\x0A\\x0D\\x09](?s:.)*/\n\
                 calls: <|python_tag|>? call"
            ),
            _ => String::from("start: <|python_tag|>? call"),
        };
        if repeat {
            grammar.push_str(" (CALL_SEP call)*");
        }
        grammar.push('\n');
        if repeat {
            // `;` then the same bounded whitespace as inside the JSON (Llama writes `; `).
            grammar.push_str(&format!(
                "CALL_SEP: /;[\\x20\\x0A\\x0D\\x09]{{0,{JSON_MAX_WHITESPACE}}}/\n"
            ));
        }
        let alternatives: Vec<String> = (0..allowed.len()).map(|i| format!("call_{i}")).collect();
        grammar.push_str(&format!("call: {}\n", alternatives.join(" | ")));
        for (i, tool) in allowed.iter().enumerate() {
            // The name is [A-Za-z0-9_-] only, so inside the Lark (JSON-syntax) string only the
            // quotes of the emitted JSON need escaping.
            grammar.push_str(&format!(
                "call_{i}: \"{{\\\"name\\\": \\\"{}\\\", \\\"parameters\\\": \" params_{i} \"}}\"\n",
                tool.name
            ));
            let schema = tool.parameters.cloned().unwrap_or_else(|| {
                let mut any_object = Map::new();
                any_object.insert("type".into(), "object".into());
                any_object
            });
            let schema = with_json_options(&Value::Object(schema));
            grammar.push_str(&format!("params_{i}: %json {schema}\n"));
        }
        Ok(ConstraintSpec::ToolCall {
            grammar_source: grammar,
        })
    }

    fn parser(&self) -> Box<dyn ToolCallParser> {
        Box::new(Llama3JsonParser::new())
    }

    /// `<|python_tag|>` as the first generated token (it decodes to no text) opens a call;
    /// otherwise the text waits while it is only whitespace, and opens a call when its first
    /// non-whitespace character is `{`.
    fn opens_like_call(
        &self,
        text: &str,
        first_token: Option<u32>,
        tokens: &BoundTokens,
    ) -> Opening {
        if first_token.is_some() && first_token == tokens.id(PYTHON_TAG) {
            return Opening::Call;
        }
        let start = text.trim_start();
        if start.is_empty() {
            Opening::Undecided
        } else if start.starts_with('{') {
            Opening::Call
        } else {
            Opening::Content
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

#[cfg(test)]
mod tests {
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

    fn tool_call_grammar(
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        Llama3Json.grammar(tools, choice, parallel)
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
            g.contains("start: <|python_tag|>? call (CALL_SEP call)*\n"),
            "{g}"
        );
        assert!(
            g.contains("CALL_SEP: /;[\\x20\\x0A\\x0D\\x09]{0,16}/\n"),
            "{g}"
        );
        assert!(g.contains("call: call_0 | call_1\n"), "{g}");
        assert!(
            g.contains(r#"call_0: "{\"name\": \"get_weather\", \"parameters\": " params_0 "}""#),
            "{g}"
        );
        assert!(g.contains(r#"call_1: "{\"name\": \"get_time\", \"parameters\": " params_1 "}""#));
        // Each tool's own schema, with the bounded natural JSON whitespace; a tool
        // without `parameters` takes any object.
        let params0 = schema_of(&g, "params_0");
        assert_eq!(params0["required"], json!(["location"]));
        assert_eq!(params0["x-guidance"], crate::structured::json_options());
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

        // Auto: free text or the `required` calls, parallel as asked.
        let g = grammar(tool_call_grammar(&tools, &ToolChoice::Auto, true).expect("grammar"));
        assert!(g.starts_with("start: text | calls\n"), "{g}");
        assert!(
            g.contains("text: /[\\x20\\x0A\\x0D\\x09]{0,16}[^{\\x20\\x0A\\x0D\\x09](?s:.)*/\n"),
            "{g}"
        );
        assert!(
            g.contains("calls: <|python_tag|>? call (CALL_SEP call)*\n"),
            "{g}"
        );
        assert!(g.contains("call: call_0 | call_1\n"), "{g}");
        let g = grammar(tool_call_grammar(&tools, &ToolChoice::Auto, false).expect("grammar"));
        assert!(g.contains("calls: <|python_tag|>? call\n"), "{g}");
        assert!(!g.contains("CALL_SEP"), "{g}");

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
        let e = err(tool_call_grammar(&tools, &ToolChoice::None, false));
        assert!(e.contains("tool_choice = none"), "{e}");
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

    #[test]
    fn tool_grammar_whitespace_on_llama_tokenizer() {
        use crate::structured::tests::{LLAMA_EOS, feed_text, llama_compiler};

        let (tokenizer, compiler) = llama_compiler();
        let limits = crate::structured::GrammarLimits {
            max_schema_bytes: 64 * 1024,
        };
        let spec =
            tool_call_grammar(&weather_tools(), &ToolChoice::Required, true).expect("grammar");
        let run = |text: &str| {
            let mut matcher = compiler.compile(&spec, &limits).expect("compile");
            feed_text(matcher.as_mut(), &tokenizer, &LLAMA_EOS, text)
        };
        let pad = " ".repeat(JSON_MAX_WHITESPACE);
        // Llama's own spacing, compact and pretty-printed parameters, the python tag and a
        // `;`-separated second call are complete outputs.
        for text in [
            r#"{"name": "get_weather", "parameters": {"location": "Paris", "unit": "celsius"}}"#
                .to_string(),
            r#"{"name": "get_weather", "parameters": {"location":"Paris"}}"#.to_string(),
            "{\"name\": \"get_weather\", \"parameters\": {\n  \"location\": \"Oslo\"\n}}".to_string(),
            format!(r#"<|python_tag|>{{"name": "get_time", "parameters": {{"zone":{pad}"UTC"}}}}"#),
            r#"{"name": "get_time", "parameters": {}}; {"name": "get_weather", "parameters": {"location": "Rome"}}"#
                .to_string(),
        ] {
            assert_eq!(run(&text), Ok(true), "{text:?}");
        }
        // Whitespace past the bound, an enum value the schema does not list and a call
        // missing a required parameter are refused.
        let over =
            format!(r#"{{"name": "get_weather", "parameters": {{"location": {pad}"Paris"}}}}"#);
        assert!(run(&over).is_err(), "{over:?} accepted");
        let bad_enum =
            r#"{"name": "get_weather", "parameters": {"location": "Paris", "unit": "kelvin"}}"#;
        assert!(run(bad_enum).is_err(), "{bad_enum:?} accepted");
        let missing = r#"{"name": "get_weather", "parameters": {"unit": "celsius"}}"#;
        assert!(run(missing).is_err(), "{missing:?} accepted");
    }

    #[test]
    fn auto_grammar_on_llama_tokenizer() {
        use crate::structured::tests::{LLAMA_EOS, feed_text, llama_compiler};

        let (tokenizer, compiler) = llama_compiler();
        let limits = crate::structured::GrammarLimits {
            max_schema_bytes: 64 * 1024,
        };
        let mut tools = weather_tools();
        tools.push(json!({"type": "function", "function": {
            "name": "convert_currency",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from": {"type": "string"},
                    "to": {"type": "string"}
                },
                "required": ["amount", "from", "to"]
            }
        }}));
        let spec = tool_call_grammar(&tools, &ToolChoice::Auto, true).expect("grammar");
        let run = |text: &str| {
            let mut matcher = compiler.compile(&spec, &limits).expect("compile");
            feed_text(matcher.as_mut(), &tokenizer, &LLAMA_EOS, text)
        };
        // Free text, including `{` after its first character and leading whitespace, and
        // valid calls with and without the python tag are complete outputs.
        for text in [
            "The sea is calm today.",
            "Use {curly} braces in the template, like {\"a\": 1}.",
            "\n  Sure: here is {x}",
            r#"{"name": "convert_currency", "parameters": {"amount": 100, "from": "GBP", "to": "JPY"}}"#,
            r#"<|python_tag|>{"name": "get_weather", "parameters": {"location": "London", "unit": "celsius"}}"#,
            r#"{"name": "get_time", "parameters": {}}; {"name": "get_weather", "parameters": {"location": "Rome"}}"#,
        ] {
            assert_eq!(run(text), Ok(true), "{text:?}");
        }
        // Output that opens like JSON must be a call: a non-call object, an unlisted function
        // and a string where the schema wants a number are refused.
        for text in [
            r#"{"a": 1}"#,
            r#"{"name": "get_stock", "parameters": {}}"#,
            r#"{"name": "convert_currency", "parameters": {"amount": "100", "from": "GBP", "to": "JPY"}}"#,
        ] {
            assert!(run(text).is_err(), "{text:?} accepted");
        }
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
