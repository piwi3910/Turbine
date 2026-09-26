//! Conformance of the registered tool formats (render, grammar, parse, round trip), the
//! `auto`-mode opening rule and [`bind`].

use std::sync::Arc;

use serde_json::{Value, json};
use turbine_core::registry::Module;

use super::*;
use crate::structured::GrammarLimits;
use crate::structured::tests::feed_text;
use crate::testing::TempDir;
use crate::testing::tiny::{PYTHON_TAG as TINY_PYTHON_TAG, TINY_EOS, write_tiny_llama};
use crate::tools::{ToolCallOut, ToolCallParser, ToolChoice, ToolParse, tool_call_grammar};
use crate::{GrammarCompiler, ModelError, Tokenizer};

fn tiny_tokenizer(name: &str) -> (TempDir, Arc<Tokenizer>) {
    let dir = TempDir::new(name);
    let spec = write_tiny_llama(dir.path(), 7);
    let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
    (dir, tokenizer)
}

/// The fixture tools of the `tools.rs` tests on main.
fn weather_tools() -> Vec<Value> {
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

fn grammar_source(spec: ConstraintSpec) -> String {
    match spec {
        ConstraintSpec::ToolCall { grammar_source } => grammar_source,
        other => panic!("expected a tool-call grammar, got {other:?}"),
    }
}

/// `tool_call_grammar(weather_tools(), …)` on main (0e11973), recorded before the move.
const MAIN_REQUIRED_PARALLEL: &str = "start: <|python_tag|>? call (CALL_SEP call)*\nCALL_SEP: /;[\\x20\\x0A\\x0D\\x09]{0,16}/\ncall: call_0 | call_1\ncall_0: \"{\\\"name\\\": \\\"get_weather\\\", \\\"parameters\\\": \" params_0 \"}\"\nparams_0: %json {\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\"],\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\ncall_1: \"{\\\"name\\\": \\\"get_time\\\", \\\"parameters\\\": \" params_1 \"}\"\nparams_1: %json {\"type\":\"object\",\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\n";
const MAIN_REQUIRED_SINGLE: &str = "start: <|python_tag|>? call\ncall: call_0 | call_1\ncall_0: \"{\\\"name\\\": \\\"get_weather\\\", \\\"parameters\\\": \" params_0 \"}\"\nparams_0: %json {\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\"],\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\ncall_1: \"{\\\"name\\\": \\\"get_time\\\", \\\"parameters\\\": \" params_1 \"}\"\nparams_1: %json {\"type\":\"object\",\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\n";
const MAIN_AUTO_SINGLE: &str = "start: text | calls\ntext: /[\\x20\\x0A\\x0D\\x09]{0,16}[^{\\x20\\x0A\\x0D\\x09](?s:.)*/\ncalls: <|python_tag|>? call\ncall: call_0 | call_1\ncall_0: \"{\\\"name\\\": \\\"get_weather\\\", \\\"parameters\\\": \" params_0 \"}\"\nparams_0: %json {\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\"],\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\ncall_1: \"{\\\"name\\\": \\\"get_time\\\", \\\"parameters\\\": \" params_1 \"}\"\nparams_1: %json {\"type\":\"object\",\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\n";
const MAIN_AUTO_PARALLEL: &str = "start: text | calls\ntext: /[\\x20\\x0A\\x0D\\x09]{0,16}[^{\\x20\\x0A\\x0D\\x09](?s:.)*/\ncalls: <|python_tag|>? call (CALL_SEP call)*\nCALL_SEP: /;[\\x20\\x0A\\x0D\\x09]{0,16}/\ncall: call_0 | call_1\ncall_0: \"{\\\"name\\\": \\\"get_weather\\\", \\\"parameters\\\": \" params_0 \"}\"\nparams_0: %json {\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\"],\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\ncall_1: \"{\\\"name\\\": \\\"get_time\\\", \\\"parameters\\\": \" params_1 \"}\"\nparams_1: %json {\"type\":\"object\",\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\n";
const MAIN_NAMED_GET_TIME: &str = "start: <|python_tag|>? call\ncall: call_0\ncall_0: \"{\\\"name\\\": \\\"get_time\\\", \\\"parameters\\\": \" params_0 \"}\"\nparams_0: %json {\"type\":\"object\",\"x-guidance\":{\"item_separator\":\",\",\"key_separator\":\":\",\"whitespace_flexible\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,16}\"}}\n";

/// `(name, arguments)` of each call, in order; panics on content.
fn names_and_args(parse: ToolParse) -> Vec<(String, String)> {
    match parse {
        ToolParse::Calls(calls) => calls
            .into_iter()
            .zip(0u32..)
            .map(
                |(
                    ToolCallOut {
                        index,
                        name,
                        arguments,
                        id,
                    },
                    i,
                )| {
                    assert_eq!(index, i);
                    assert!(id.starts_with("call_") && id.len() == 29, "{id}");
                    (name, arguments)
                },
            )
            .collect(),
        ToolParse::Content(text) => panic!("expected calls, got content {text:?}"),
    }
}

#[test]
fn llama3_json_conformance() {
    let format = registry()
        .get("llama3_json")
        .expect("llama3_json is registered");
    let tools = weather_tools();

    // Render: the chat template renders the tools; the format does not.
    assert_eq!(format.render_tools(&tools), None);

    // Grammar: main's `tool_call_grammar` output byte for byte, and `tool_call_grammar` still
    // equals the format's grammar.
    let named = ToolChoice::Named("get_time".into());
    for (choice, parallel, main) in [
        (&ToolChoice::Required, true, MAIN_REQUIRED_PARALLEL),
        (&ToolChoice::Required, false, MAIN_REQUIRED_SINGLE),
        (&ToolChoice::Auto, false, MAIN_AUTO_SINGLE),
        (&ToolChoice::Auto, true, MAIN_AUTO_PARALLEL),
        (&named, true, MAIN_NAMED_GET_TIME),
    ] {
        let g = grammar_source(format.grammar(&tools, choice, parallel).expect("grammar"));
        assert_eq!(g, main, "{choice:?} parallel={parallel}");
        let legacy = grammar_source(tool_call_grammar(&tools, choice, parallel).unwrap());
        assert_eq!(legacy, g);
    }
    assert!(matches!(
        format.grammar(&tools, &ToolChoice::None, false),
        Err(ModelError::Constraint(_))
    ));

    // Parse: main's parser test inputs give main's results.
    let parser = format.parser();
    assert_eq!(
        names_and_args(parser.parse(
            r#"{"name": "get_weather", "parameters": {"location": "Paris", "unit": "celsius"}}"#
        )),
        [(
            "get_weather".to_string(),
            r#"{"location":"Paris","unit":"celsius"}"#.to_string()
        )]
    );
    assert_eq!(
        names_and_args(parser.parse(concat!(
            r#"{"name": "a", "parameters": {"x": [1, 2]}} ; "#,
            "\n",
            r#"{"name": "b", "parameters": {}}"#,
            "\n"
        ))),
        [
            ("a".to_string(), r#"{"x":[1,2]}"#.to_string()),
            ("b".to_string(), "{}".to_string())
        ]
    );
    assert_eq!(
        names_and_args(parser.parse(
            r#"<|python_tag|>{"name": "search", "parameters": {"query": "rust ; \"llm\""}}"#
        )),
        [(
            "search".to_string(),
            r#"{"query":"rust ; \"llm\""}"#.to_string()
        )]
    );
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

    // Round trip: a call text the compiled grammar accepts on the tiny tokenizer parses into
    // the same names and arguments.
    let (_dir, tokenizer) = tiny_tokenizer("turbine-formats-llama3-json");
    let compiler = GrammarCompiler::new(&tokenizer, &TINY_EOS).expect("token trie");
    let limits = GrammarLimits {
        max_schema_bytes: 64 * 1024,
    };
    let spec = format
        .grammar(&tools, &ToolChoice::Required, true)
        .expect("grammar");
    let text = concat!(
        r#"<|python_tag|>{"name": "get_weather", "parameters": {"location": "Oslo", "unit": "celsius"}}"#,
        r#"; {"name": "get_time", "parameters": {}}"#
    );
    let mut matcher = compiler.compile(&spec, &limits).expect("compile");
    assert_eq!(
        feed_text(matcher.as_mut(), &tokenizer, &TINY_EOS, text),
        Ok(true)
    );
    assert_eq!(
        names_and_args(parser.parse(text)),
        [
            (
                "get_weather".to_string(),
                r#"{"location":"Oslo","unit":"celsius"}"#.to_string()
            ),
            ("get_time".to_string(), "{}".to_string())
        ]
    );
}

/// Main's `ToolText` decision (engine/requests.rs at 0e11973), copied before the move: the
/// python tag as the first generated token opens a call; else the pending text waits while it
/// is whitespace, and holds when it starts with `{`.
fn main_tool_text(text: &str, first_token: Option<u32>, python_tag: Option<u32>) -> Opening {
    if first_token.is_some() && python_tag == first_token {
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

#[test]
fn opens_like_call_matches_main() {
    let (_dir, tokenizer) = tiny_tokenizer("turbine-formats-opening");
    let bound = bind(&Llama3Json, &tokenizer).expect("bind");
    assert_eq!(bound.tokens.id("<|python_tag|>"), Some(TINY_PYTHON_TAG.1));
    let tag = Some(TINY_PYTHON_TAG.1);
    let first_a = Some(u32::from(b'a'));
    for (text, first, expected) in [
        ("", None, Opening::Undecided),
        ("  ", Some(u32::from(b' ')), Opening::Undecided),
        (" {\"name\"", Some(u32::from(b' ')), Opening::Call),
        ("Hello", Some(u32::from(b'H')), Opening::Content),
        ("<", Some(u32::from(b'<')), Opening::Content),
        ("", tag, Opening::Call),
        ("x", tag, Opening::Call),
        ("a{", first_a, Opening::Content),
    ] {
        let got = bound.opens_like_call(text, first);
        assert_eq!(got, expected, "{text:?} first={first:?}");
        assert_eq!(got, main_tool_text(text, first, tag), "{text:?} vs main");
    }
    // A tokenizer without the (optional) python tag: only the text decides.
    let unbound = BoundToolFormat {
        format: &Llama3Json,
        tokens: BoundTokens::default(),
    };
    assert_eq!(unbound.opens_like_call("", tag), Opening::Undecided);
    assert_eq!(unbound.opens_like_call("{", tag), Opening::Call);
}

/// A format needing a token no tokenizer here defines.
struct NeedsNone;

impl Module for NeedsNone {
    fn name(&self) -> &'static str {
        "needs_none"
    }
}

impl ToolFormat for NeedsNone {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        &[
            SpecialToken {
                text: "<|python_tag|>",
                required: false,
            },
            SpecialToken {
                text: "<|none|>",
                required: true,
            },
        ]
    }
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        Llama3Json.grammar(tools, choice, parallel)
    }
    fn parser(&self) -> Box<dyn ToolCallParser> {
        Llama3Json.parser()
    }
    fn opens_like_call(&self, _: &str, _: Option<u32>, _: &BoundTokens) -> Opening {
        Opening::Content
    }
}

#[test]
fn bind_refuses_missing_required_token() {
    let (_dir, tokenizer) = tiny_tokenizer("turbine-formats-bind");
    let err = bind(&NeedsNone, &tokenizer).unwrap_err();
    assert!(matches!(err, ModelError::Unsupported { .. }), "{err:?}");
    let message = err.to_string();
    assert!(message.contains("needs_none"), "{message}");
    assert!(message.contains("<|none|>"), "{message}");
    // Optional tokens never refuse: llama3_json binds on the tiny tokenizer.
    let bound = bind(&Llama3Json, &tokenizer).expect("bind");
    assert_eq!(bound.format.name(), "llama3_json");
}
