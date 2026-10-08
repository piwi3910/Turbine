//! Each registered tool format's own cases (main's grammars and parser inputs for
//! `llama3_json`, the run-ahead's for `hermes` and `mistral`), the `auto`-mode opening rules
//! and [`bind`]. The checks every format must pass (render, grammar, parse, round trip,
//! opening) are `crate::conformance::formats_suite`, run by `registry_conformance::tool_formats`.

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
                    // `call_` + 24 alphanumerics, or the 9 of the `mistral` format.
                    let alnum = |s: &str| s.bytes().all(|b| b.is_ascii_alphanumeric());
                    assert!(
                        id.strip_prefix("call_")
                            .is_some_and(|rest| rest.len() == 24 && alnum(rest))
                            || (id.len() == 9 && alnum(&id)),
                        "{id}"
                    );
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

    // Render, refusals, the sample call's parse and round trip and its opening are
    // `conformance::formats_suite`; these pin main's grammars and parser cases.

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
    fn sample_call(&self) -> &'static str {
        Llama3Json.sample_call()
    }
}

/// `llama3_json` with an empty second special token.
struct EmptySpecial;

impl Module for EmptySpecial {
    fn name(&self) -> &'static str {
        "empty_special"
    }
}

impl ToolFormat for EmptySpecial {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        &[
            SpecialToken {
                text: "<|python_tag|>",
                required: false,
            },
            SpecialToken {
                text: "",
                required: false,
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
    fn opens_like_call(&self, text: &str, first: Option<u32>, tokens: &BoundTokens) -> Opening {
        Llama3Json.opens_like_call(text, first, tokens)
    }
    fn sample_call(&self) -> &'static str {
        Llama3Json.sample_call()
    }
}

static WITH_EMPTY_SPECIAL: turbine_core::registry::Registry<dyn ToolFormat> =
    turbine_core::registry::Registry::new("tool_format", &[&EmptySpecial]);

/// The suite's empty-special-token failure names the token's index (Scout 5a957de4).
#[test]
fn empty_special_token_failure_names_its_index() {
    let failures = crate::conformance::formats_suite(&WITH_EMPTY_SPECIAL).unwrap_err();
    let special: Vec<&str> = failures
        .iter()
        .filter(|f| f.check == "special_tokens")
        .map(|f| f.detail.as_str())
        .collect();
    assert_eq!(special, ["special token 1 is empty"], "{failures:#?}");
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

/// Compiles `format`'s grammar for `choice` / `parallel` over [`weather_tools`] on `compiler`
/// and feeds `text`: `Ok(true)` when the grammar accepts the whole text and may end there.
fn accepts(
    compiler: &GrammarCompiler,
    tokenizer: &Tokenizer,
    eos: &[u32],
    format: &dyn ToolFormat,
    choice: &ToolChoice,
    parallel: bool,
    text: &str,
) -> Result<bool, String> {
    let limits = GrammarLimits {
        max_schema_bytes: 64 * 1024,
    };
    let spec = format
        .grammar(&weather_tools(), choice, parallel)
        .expect("grammar");
    let mut matcher = compiler.compile(&spec, &limits).expect("compile");
    feed_text(matcher.as_mut(), tokenizer, eos, text)
}

/// An enveloped format's sample outputs over [`weather_tools`]: `one` a single `get_weather`
/// call, `two` a `get_weather` then a `get_time` call (parallel), `time` a single `get_time`
/// call, `bad` a `get_weather` call whose arguments break the schema, `unknown` a call of an
/// unlisted tool, `texts` free text the `auto` grammar lets through and the parser leaves as
/// content.
struct Samples<'a> {
    one: &'a str,
    two: &'a str,
    time: &'a str,
    bad: &'a str,
    unknown: &'a str,
    texts: &'a [&'a str],
}

/// Render, grammar, parse and round trip of the registered format `name` over `s`.
fn enveloped_conformance(name: &str, s: &Samples<'_>) {
    // Render, the refused choices (`none`, no tools, an unknown name), the sample call's parse
    // and round trip and its opening are `conformance::formats_suite`
    // (`registry_conformance::tool_formats`); these are the format's own cases.
    let format = registry().get(name).expect("registered");

    // Parse: the sample calls give their names and compact arguments.
    let parser = format.parser();
    let weather = (
        "get_weather".to_string(),
        r#"{"location":"Oslo","unit":"celsius"}"#.to_string(),
    );
    let time = ("get_time".to_string(), "{}".to_string());
    assert_eq!(
        names_and_args(parser.parse(s.one)),
        std::slice::from_ref(&weather)
    );
    assert_eq!(names_and_args(parser.parse(s.two)), [weather, time.clone()]);
    assert_eq!(names_and_args(parser.parse(s.time)), [time]);
    for text in s.texts {
        assert_eq!(
            parser.parse(text),
            ToolParse::Content(text.to_string()),
            "{name}: {text:?}"
        );
    }

    // Round trip on the tiny tokenizer: what the grammar accepts, the parser turned into calls
    // above; a broken schema, an unlisted tool or a second call where one is allowed is
    // refused.
    let (_dir, tokenizer) = tiny_tokenizer(&format!("turbine-formats-{name}"));
    let compiler = GrammarCompiler::new(&tokenizer, &TINY_EOS).expect("token trie");
    let ok = |choice: &ToolChoice, parallel, text: &str| {
        accepts(
            &compiler, &tokenizer, &TINY_EOS, format, choice, parallel, text,
        )
    };
    let named = ToolChoice::Named("get_time".into());
    assert_eq!(ok(&ToolChoice::Required, false, s.one), Ok(true), "{name}");
    assert_eq!(ok(&ToolChoice::Required, true, s.two), Ok(true), "{name}");
    assert!(ok(&ToolChoice::Required, false, s.two).is_err(), "{name}");
    assert!(ok(&ToolChoice::Required, false, s.bad).is_err(), "{name}");
    assert!(ok(&ToolChoice::Auto, true, s.unknown).is_err(), "{name}");
    assert!(
        ok(&ToolChoice::Required, false, "No call.").is_err(),
        "{name}"
    );
    assert_eq!(ok(&named, true, s.time), Ok(true), "{name}");
    assert!(ok(&named, true, s.one).is_err(), "{name}");
    assert_eq!(ok(&ToolChoice::Auto, true, s.two), Ok(true), "{name}");
    for text in s.texts {
        assert_eq!(
            ok(&ToolChoice::Auto, true, text),
            Ok(true),
            "{name}: {text:?}"
        );
    }
}

#[test]
fn hermes_conformance() {
    let one = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Oslo\", \"unit\": \"celsius\"}}\n</tool_call>";
    let time = "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
    let two = format!("{one}\n{time}");
    enveloped_conformance(
        "hermes",
        &Samples {
            one,
            two: &two,
            time,
            bad: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"unit\": \"kelvin\"}}\n</tool_call>",
            unknown: "<tool_call>\n{\"name\": \"get_stock\", \"arguments\": {}}\n</tool_call>",
            texts: &[
                "The sea is calm.",
                "<think>\nhmm\n</think>\n\nIt is sunny.",
                "<",
                "  <b>bold</b>",
            ],
        },
    );
    // A leading reasoning block is dropped from a tool-call answer.
    let parser = registry().get("hermes").unwrap().parser();
    assert_eq!(
        names_and_args(parser.parse(&format!("<think>\nweather\n</think>\n\n{time}"))),
        [("get_time".to_string(), "{}".to_string())]
    );
    // The special tokens are optional: hermes binds on the tiny tokenizer, which lacks them.
    let (_dir, tokenizer) = tiny_tokenizer("turbine-formats-hermes-bind");
    let bound = bind(&Hermes, &tokenizer).expect("bind");
    assert_eq!(bound.tokens.id("<tool_call>"), None);
}

#[test]
fn mistral_conformance() {
    let one = r#"[{"name": "get_weather", "arguments": {"location": "Oslo", "unit": "celsius"}}]"#;
    let two = r#"[{"name": "get_weather", "arguments": {"location": "Oslo", "unit": "celsius"}}, {"name": "get_time", "arguments": {}}]"#;
    let time = r#"[{"name": "get_time", "arguments": {}}]"#;
    enveloped_conformance(
        "mistral",
        &Samples {
            one,
            two,
            time,
            bad: r#"[{"name": "get_weather", "arguments": {"unit": "kelvin"}}]"#,
            unknown: r#"[{"name": "get_stock", "arguments": {}}]"#,
            texts: &["It rains [sometimes].", "Plain answer."],
        },
    );
    // `[TOOL_CALLS]` before the array (it decodes to no text on Mistral's tokenizer) parses.
    let parser = registry().get("mistral").unwrap().parser();
    assert_eq!(
        names_and_args(parser.parse(&format!("[TOOL_CALLS] {time}"))),
        [("get_time".to_string(), "{}".to_string())]
    );
    // The `auto` grammar refuses an array that is not a call.
    let (_dir, tokenizer) = tiny_tokenizer("turbine-formats-mistral-array");
    let compiler = GrammarCompiler::new(&tokenizer, &TINY_EOS).expect("token trie");
    let array = accepts(
        &compiler,
        &tokenizer,
        &TINY_EOS,
        &Mistral,
        &ToolChoice::Auto,
        true,
        "[1, 2]",
    );
    assert!(array.is_err(), "{array:?}");
}

/// The run-ahead's grammar cases on the real Llama-3.2 tokenizer (multi-byte tokens across the
/// envelope and the JSON).
#[test]
fn hermes_and_mistral_grammars_on_llama_tokenizer() {
    use crate::structured::tests::{LLAMA_EOS, llama_compiler};

    let (tokenizer, compiler) = llama_compiler();
    let ok = |format: &dyn ToolFormat, choice: &ToolChoice, parallel, text: &str| {
        accepts(
            &compiler, &tokenizer, &LLAMA_EOS, format, choice, parallel, text,
        )
    };
    let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Oslo\"}}\n</tool_call>";
    let time = "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
    let two = format!("{call}\n{time}");
    assert_eq!(ok(&Hermes, &ToolChoice::Required, false, call), Ok(true));
    assert_eq!(ok(&Hermes, &ToolChoice::Required, true, &two), Ok(true));
    assert!(ok(&Hermes, &ToolChoice::Required, false, &two).is_err());
    for text in [
        "The sea is calm.",
        "<think>\nhmm\n</think>\n\nIt is sunny.",
        "<",
        call,
    ] {
        assert_eq!(
            ok(&Hermes, &ToolChoice::Auto, true, text),
            Ok(true),
            "{text:?}"
        );
    }
    let m_two = r#"[{"name": "get_weather", "arguments": {"location": "Rome"}}, {"name": "get_time", "arguments": {}}]"#;
    let m_one = r#"[{"name": "get_time", "arguments": {}}]"#;
    assert_eq!(ok(&Mistral, &ToolChoice::Required, true, m_two), Ok(true));
    assert_eq!(ok(&Mistral, &ToolChoice::Required, false, m_one), Ok(true));
    assert!(ok(&Mistral, &ToolChoice::Required, false, m_two).is_err());
    for text in ["It rains [sometimes].", m_one] {
        assert_eq!(
            ok(&Mistral, &ToolChoice::Auto, true, text),
            Ok(true),
            "{text:?}"
        );
    }
    assert!(ok(&Mistral, &ToolChoice::Auto, true, "[1, 2]").is_err());
}

/// The `auto` opening rules: Hermes holds an output whose text (after whitespace) starts with
/// `<tool_call>` or whose first token is `<tool_call>`, waits while the text is still a prefix
/// of it, and streams anything else — a leading `<think>` block included (calls after thinking
/// are not parsed in `auto` mode, as the run-ahead has it). Mistral holds on a leading `[` or
/// the `[TOOL_CALLS]` token and waits only on empty text.
#[test]
fn openings_of_hermes_and_mistral() {
    const OPEN: u32 = 900;
    const TOOL_CALLS: u32 = 901;
    let hermes = BoundToolFormat {
        format: &Hermes,
        tokens: BoundTokens {
            ids: vec![("<tool_call>", Some(OPEN)), ("</tool_call>", None)],
        },
    };
    let a = Some(u32::from(b'a'));
    for (text, first, expected) in [
        ("", None, Opening::Undecided),
        ("  \n", Some(u32::from(b' ')), Opening::Undecided),
        ("<", Some(u32::from(b'<')), Opening::Undecided),
        (" <tool_ca", Some(u32::from(b' ')), Opening::Undecided),
        ("<tool_call>", Some(u32::from(b'<')), Opening::Call),
        ("\n<tool_call>\n{", Some(u32::from(b'\n')), Opening::Call),
        ("<tool_call>", Some(OPEN), Opening::Call),
        ("<think>\nhmm", Some(u32::from(b'<')), Opening::Content),
        ("<b>bold", Some(u32::from(b'<')), Opening::Content),
        ("Sure", a, Opening::Content),
    ] {
        assert_eq!(
            hermes.opens_like_call(text, first),
            expected,
            "hermes {text:?} first={first:?}"
        );
    }
    let mistral = BoundToolFormat {
        format: &Mistral,
        tokens: BoundTokens {
            ids: vec![("[TOOL_CALLS]", Some(TOOL_CALLS))],
        },
    };
    for (text, first, expected) in [
        ("", None, Opening::Undecided),
        ("  ", Some(u32::from(b' ')), Opening::Undecided),
        ("", Some(TOOL_CALLS), Opening::Call),
        (" [", Some(TOOL_CALLS), Opening::Call),
        ("[", Some(u32::from(b'[')), Opening::Call),
        ("\n[{\"name\"", Some(u32::from(b'\n')), Opening::Call),
        ("It", Some(u32::from(b'I')), Opening::Content),
        ("{\"name\"", Some(u32::from(b'{')), Opening::Content),
        ("<", Some(u32::from(b'<')), Opening::Content),
        ("a[", a, Opening::Content),
    ] {
        assert_eq!(
            mistral.opens_like_call(text, first),
            expected,
            "mistral {text:?} first={first:?}"
        );
    }
}

/// The chat-template fixture (`tests/fixtures/<dir>/tokenizer_config.json`, the checkpoint's
/// own template) of a family whose default tool format is checked against its real template;
/// `None` for Llama and OLMoE (covered by Phase 2's tool tests).
fn family_template_fixture(family: &str) -> Option<&'static str> {
    match family {
        "qwen3" => Some("qwen3-0.6b"),
        // The template of the pinned Qwen3-30B-A3B-Instruct-2507-FP8 revision
        // (`scripts/lab/phase7-checkpoints.tsv`).
        "qwen3_moe" => Some("qwen3-30b-a3b"),
        "mistral" => Some("mistral-7b-instruct-v0.3"),
        "mixtral" => Some("mixtral-8x7b-instruct-v0.1"),
        _ => None,
    }
}

/// Each family's default tool format, and the special tokens of its tokenizer that decode to
/// no text (skipped by the detokenizer, so a parser never sees them).
fn family_tool_expectations(family: &str) -> (Option<&'static str>, &'static [&'static str]) {
    match family {
        "llama" => (Some(llama3_json::LLAMA3_JSON), &[]),
        "olmoe" => (None, &[]),
        "qwen3" | "qwen3_moe" => (Some(hermes::HERMES), &["<|im_end|>"]),
        "mistral" | "mixtral" => (Some(mistral::MISTRAL), &[mistral::TOOL_CALLS, "</s>"]),
        // Harmony is registered with Phase 7 Task 25, which adds the template round trip; until
        // then the name alone is asserted.
        "gpt_oss" => (Some("harmony"), &[]),
        other => panic!("family {other} has no expected default tool format"),
    }
}

/// S-18 (part 1): an assistant turn carrying one `get_weather` call, rendered by the family's
/// real chat template, parses back through the family's default format into that call. The
/// assistant text is what the template renders after the generation prompt, with the
/// tokenizer's no-text special tokens removed and the family's reasoning split off first, as
/// the server does.
#[test]
fn family_default_tool_calls_round_trip() {
    let arguments = json!({"location": "Oslo", "unit": "celsius"});
    let tools = weather_tools();
    let user = json!({"role": "user", "content": "What is the weather in Oslo?"});
    let assistant = json!({"role": "assistant", "content": "", "tool_calls": [{
        "id": "a1B2c3D4e",
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments.clone()}
    }]});
    let mut checked = 0;
    for family in crate::families::registry().iter() {
        let (default, no_text) = family_tool_expectations(family.name());
        assert_eq!(
            family.default_tool_format(),
            default,
            "family {} default tool format",
            family.name()
        );
        let Some(dir) = family_template_fixture(family.name()) else {
            continue;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(dir)
            .join("tokenizer_config.json");
        let template = crate::ChatTemplate::load(&path)
            .unwrap_or_else(|e| panic!("{}: {dir} template: {e}", family.name()));
        let kwargs = serde_json::Map::new();
        let prompt = template
            .render(std::slice::from_ref(&user), Some(&tools), true, &kwargs)
            .unwrap_or_else(|e| panic!("{}: prompt: {e}", family.name()));
        let turn = template
            .render(
                &[user.clone(), assistant.clone()],
                Some(&tools),
                false,
                &kwargs,
            )
            .unwrap_or_else(|e| panic!("{}: assistant turn: {e}", family.name()));
        let mut text = turn
            .strip_prefix(&prompt)
            .unwrap_or_else(|| {
                panic!(
                    "{}: the assistant turn does not extend the generation prompt:\n{prompt:?}\n{turn:?}",
                    family.name()
                )
            })
            .to_string();
        for token in no_text {
            text = text.replace(token, "");
        }
        if let Some(name) = family.default_reasoning_parser() {
            let parser = crate::reasoning::registry()
                .get(name)
                .unwrap_or_else(|| panic!("reasoning parser {name}"));
            let mut stream = parser.start(&prompt);
            let mut split = stream.push(&text);
            split.append(stream.finish());
            assert_eq!(split.reasoning.trim(), "", "{}: {text:?}", family.name());
            text = split.content;
        }
        let format = registry()
            .get(default.expect("a family with a template fixture has a default format"))
            .expect("registered default format");
        if family.name() == "mixtral" {
            // Mixtral-8x7B-Instruct-v0.1's template (also the pinned RedHatAI FP8 revision's)
            // renders an assistant turn's `content` only: it drops `tool_calls`, so no call can
            // round-trip through it. Kept as an explicit case so a template that starts
            // rendering calls is noticed.
            assert!(!text.contains("get_weather"), "mixtral: {text:?}");
            checked += 1;
            continue;
        }
        let calls = names_and_args(format.parser().parse(&text));
        assert_eq!(calls.len(), 1, "{}: {text:?}", family.name());
        assert_eq!(calls[0].0, "get_weather", "{}: {text:?}", family.name());
        let parsed: Value = serde_json::from_str(&calls[0].1).expect("arguments are JSON");
        assert_eq!(parsed, arguments, "{}: {text:?}", family.name());
        checked += 1;
    }
    assert_eq!(
        checked, 4,
        "qwen3, qwen3_moe, mistral and mixtral have template fixtures"
    );
}

/// P7 S-6: the `mistral` parser's call ids are 9 alphanumerics, so a parsed call sent back as
/// an assistant `tool_calls` turn plus its `tool` result renders through Mistral-7B-Instruct-
/// v0.3's real template (which raises on any other id length) with the call and the result in
/// the prompt. Breaks if the format draws `call_` + 24 ids (the template refuses the turn).
#[test]
fn mistral_call_ids_render_through_the_v03_template() {
    let parser = mistral::MistralParser::seeded(7);
    let text = r#"[{"name": "get_weather", "arguments": {"location": "Oslo"}}, {"name": "get_time", "arguments": {}}]"#;
    let ToolParse::Calls(calls) = parser.parse(text) else {
        panic!("expected calls");
    };
    assert_eq!(calls.len(), 2);
    for call in &calls {
        assert!(
            call.id.len() == crate::tools::MISTRAL_CALL_ID_LEN
                && call.id.bytes().all(|b| b.is_ascii_alphanumeric()),
            "{}",
            call.id
        );
    }
    assert_ne!(calls[0].id, calls[1].id);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mistral-7b-instruct-v0.3/tokenizer_config.json");
    let template = crate::ChatTemplate::load(&path).expect("mistral template");
    let call = &calls[0];
    let arguments: Value = serde_json::from_str(&call.arguments).expect("arguments JSON");
    let messages = [
        json!({"role": "user", "content": "What is the weather in Oslo?"}),
        json!({"role": "assistant", "content": null, "tool_calls": [{
            "id": call.id, "type": "function",
            "function": {"name": call.name, "arguments": arguments}
        }]}),
        json!({"role": "tool", "tool_call_id": call.id, "content": "{\"temperature\": 7}"}),
    ];
    let tools = weather_tools();
    let kwargs = serde_json::Map::new();
    let prompt = template
        .render(&messages, Some(&tools), true, &kwargs)
        .expect("the v0.3 template accepts the format's ids");
    assert!(
        prompt.contains(&format!("\"id\": \"{}\"", call.id)),
        "{prompt}"
    );
    assert!(
        prompt.contains(&format!("\"call_id\": \"{}\"", call.id)),
        "{prompt}"
    );

    let long =
        json!({"role": "tool", "tool_call_id": "call_0123456789abcdefghijklmn", "content": "x"});
    let refused = template.render(
        &[messages[0].clone(), messages[1].clone(), long],
        Some(&tools),
        true,
        &kwargs,
    );
    assert!(refused.is_err(), "the template refuses a call_ + 24 id");
}
