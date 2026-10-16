//! The `qwen3_xml` tool-call format (Qwen3.5 / Qwen3.6; Phase 7 S-18, user decision 2026-10-16
//! "P7 Qwen3.5 tool format `qwen3_xml`").
//!
//! The Qwen3.5 chat template writes a call as
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=k>
//! v
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! (one `<parameter=…>` block per argument, in the argument object's order; a mapping or a
//! non-string sequence is written as compact JSON and every other value as its raw text). The
//! `hermes` format cannot parse this — it expects `{"name": …, "arguments": {…}}` between the
//! tags — so Qwen3.5 needs this format. `<tool_call>` / `</tool_call>` are added tokens of the
//! Qwen3.5 tokenizer that decode to their text (optional here: on another tokenizer they are
//! plain text). The template also wraps reasoning in a `think` block, so the existing
//! `qwen_think` reasoning parser applies (the family's `default_reasoning_parser`).
//!
//! The **parser** accepts the template's output exactly: a parameter value is decoded as JSON
//! when it is an object, array, string, number, `true`, `false` or `null`, and kept as a string
//! otherwise (the template writes a scalar string unquoted), and the `\n` the template writes
//! before `</parameter>` is optional. The **grammar** is written as the template is, except that
//! every value is JSON-encoded (a raw-text terminal makes llguidance's lexer reject the grammar)
//! and the newline before `</parameter>` is dropped (a terminal directly followed by a
//! whitespace-starting literal does the same) — so a constrained generation writes a string
//! argument quoted (`"Oslo"`) and closed straight after the value; the parser dedents that raw
//! template form, and a call rendered back by the template is the raw form. Optional arguments
//! are emitted after the required ones in schema order.
//!
//! In `auto` mode an output opens like a call when its first token is `<tool_call>` or its text,
//! after leading whitespace, starts with `<tool_call>`; it waits while that text is still a
//! prefix of `<tool_call>`.
use serde_json::{Map, Value};
use turbine_core::registry::Module;

use super::envelope::{CallIds, allowed_tools, text_not_starting_with, ws};
use super::{BoundTokens, ConstraintSpec, Opening, SpecialToken, ToolFormat};
use crate::ModelError;
use crate::tools::{FunctionTool, ToolCallParser, ToolChoice, ToolParse};

/// The format's registry name (and the `parser` label of `turbine_tool_calls_total`).
pub const QWEN3_XML: &str = "qwen3_xml";
/// Opens a call.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closes a call.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// Opens the function block (its name follows up to `>`).
pub const FUNCTION_OPEN: &str = "<function=";
/// Closes the function block (also terminates the argument list).
pub const FUNCTION_CLOSE: &str = "</function>";
/// Opens a parameter block (its name follows up to `>`).
pub const PARAMETER_OPEN: &str = "<parameter=";
/// Closes a parameter block.
pub const PARAMETER_CLOSE: &str = "</parameter>";
/// The Qwen3.5 reasoning block's open tag (`qwen_think` strips it before parsing).
const REASONING_OPEN: &str = concat!("<", "think", ">");
/// The Qwen3.5 reasoning block's close tag.
const REASONING_CLOSE: &str = concat!("<", "/", "think", ">");

/// The `qwen3_xml` format.
pub struct Qwen3Xml;

impl Module for Qwen3Xml {
    fn name(&self) -> &'static str {
        QWEN3_XML
    }
}

impl ToolFormat for Qwen3Xml {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        &[
            SpecialToken {
                text: TOOL_CALL_OPEN,
                required: false,
            },
            SpecialToken {
                text: TOOL_CALL_CLOSE,
                required: false,
            },
        ]
    }

    /// The XML call for one of the allowed tools, repeated with whitespace separators when
    /// `parallel` (not for a named function). `auto` is `start: text | calls`: free text that
    /// does not start with `<tool_call>`, or the calls.
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        let allowed = allowed_tools(tools, choice)?;
        let repeat = parallel && matches!(choice, ToolChoice::Auto | ToolChoice::Required);
        let (calls, sep) = if repeat {
            ("call (CALL_SEP call)*", format!("CALL_SEP: /{}/\n", ws(1)))
        } else {
            ("call", String::new())
        };
        let rules = call_rules(&allowed);
        let start = match choice {
            ToolChoice::Auto => format!(
                "start: text | calls\ntext: /{}/\ncalls: {calls}\n",
                text_not_starting_with(TOOL_CALL_OPEN)
            ),
            _ => format!("start: {calls}\n"),
        };
        Ok(ConstraintSpec::ToolCall {
            grammar_source: format!("{start}{sep}{rules}"),
        })
    }

    fn parser(&self) -> Box<dyn ToolCallParser> {
        Box::new(Qwen3XmlParser::new())
    }

    fn opens_like_call(
        &self,
        text: &str,
        first_token: Option<u32>,
        tokens: &BoundTokens,
    ) -> Opening {
        if first_token.is_some() && first_token == tokens.id(TOOL_CALL_OPEN) {
            return Opening::Call;
        }
        let start = text.trim_start();
        if start.starts_with(TOOL_CALL_OPEN) {
            Opening::Call
        } else if TOOL_CALL_OPEN.starts_with(start) {
            // Empty, or still a prefix of the opener (`<tool`, `<tool_call`).
            Opening::Undecided
        } else {
            Opening::Content
        }
    }

    fn sample_call(&self) -> &'static str {
        "<tool_call>\n<function=get_weather>\n<parameter=location>\n\"Oslo\"</parameter>\n<parameter=unit>\n\"celsius\"</parameter>\n</function>\n</tool_call>"
    }
}

/// A Lark string literal (the format's literal pieces are ASCII; `\n` and `"` escaped).
fn lark_literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `(name, schema, required)` of a tool's parameter properties, in schema order (a tool with no
/// `properties` has none).
fn properties(tool: &FunctionTool<'_>) -> Vec<(String, Value, bool)> {
    let Some(schema) = tool.parameters else {
        return Vec::new();
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let Some(Value::Object(props)) = schema.get("properties") else {
        return Vec::new();
    };
    props
        .iter()
        .map(|(name, sub)| (name.clone(), sub.clone(), required.contains(&name.as_str())))
        .collect()
}

/// The `call` alternatives and `call_<i>` / `param_<i>_<j>` / `val_<i>_<j>` rules of `allowed`,
/// each call writing the XML body the Qwen3.5 template renders.
fn call_rules(allowed: &[FunctionTool<'_>]) -> String {
    let alternatives: Vec<String> = (0..allowed.len()).map(|i| format!("call_{i}")).collect();
    let mut rules = format!("call: {}\n", alternatives.join(" | "));
    for (i, tool) in allowed.iter().enumerate() {
        let mut items = vec![lark_literal(&format!(
            "{TOOL_CALL_OPEN}\n{FUNCTION_OPEN}{}>\n",
            tool.name
        ))];
        for (j, (name, sub, required)) in properties(tool).iter().enumerate() {
            rules.push_str(&format!(
                "param_{i}_{j}: {} val_{i}_{j} {}\n",
                lark_literal(&format!("{PARAMETER_OPEN}{name}>\n")),
                lark_literal(&format!("{PARAMETER_CLOSE}\n"))
            ));
            // Every value is the schema's JSON encoding: an unambiguous lexer (see the module
            // doc), and the parser dedents the template's raw scalars either way.
            rules.push_str(&format!("val_{i}_{j}: %json {sub}\n"));
            items.push(format!("param_{i}_{j}{}", if *required { "" } else { "?" }));
        }
        items.push(lark_literal(&format!(
            "{FUNCTION_CLOSE}\n{TOOL_CALL_CLOSE}"
        )));
        rules.push_str(&format!("call_{i}: {}\n", items.join(" ")));
    }
    rules
}

/// The `qwen3_xml` parser: after an optional leading reasoning block, one or more `<tool_call>`
/// blocks, each holding a `<function=…>` and zero or more `<parameter=…>` blocks between
/// whitespace, and nothing else. Anything else is [`ToolParse::Content`] unchanged.
pub struct Qwen3XmlParser {
    ids: CallIds,
}

impl std::fmt::Debug for Qwen3XmlParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3XmlParser").finish_non_exhaustive()
    }
}

impl Default for Qwen3XmlParser {
    fn default() -> Self {
        Qwen3XmlParser::new()
    }
}

impl Qwen3XmlParser {
    /// A parser whose call ids come from an OS-seeded stream.
    pub fn new() -> Qwen3XmlParser {
        Qwen3XmlParser {
            ids: CallIds::new(),
        }
    }

    /// A parser with reproducible call ids (tests, simulations).
    pub fn seeded(seed: u64) -> Qwen3XmlParser {
        Qwen3XmlParser {
            ids: CallIds::seeded(seed),
        }
    }
}

impl ToolCallParser for Qwen3XmlParser {
    fn parse(&self, text: &str) -> ToolParse {
        match parse_qwen3_xml_calls(text) {
            Some(found) => ToolParse::Calls(self.ids.assign(found)),
            None => ToolParse::Content(text.to_string()),
        }
    }
}

/// `(name, compact arguments)` of every call when `text` is entirely Qwen3.5 XML calls.
fn parse_qwen3_xml_calls(text: &str) -> Option<Vec<(String, String)>> {
    let mut rest = text.trim_start();
    if let Some(after) = rest.strip_prefix(REASONING_OPEN) {
        let end = after.find(REASONING_CLOSE)?;
        rest = after[end + REASONING_CLOSE.len()..].trim_start();
    }
    let mut calls = Vec::new();
    while !rest.is_empty() {
        let body = rest.strip_prefix(TOOL_CALL_OPEN)?.trim_start();
        let end = body.find(TOOL_CALL_CLOSE)?;
        calls.push(parse_function(&body[..end])?);
        rest = body[end + TOOL_CALL_CLOSE.len()..].trim_start();
    }
    (!calls.is_empty()).then_some(calls)
}

/// `(name, compact arguments)` of one `<function=…>…</function>` block.
fn parse_function(body: &str) -> Option<(String, String)> {
    let body = body.trim();
    let rest = body.strip_prefix(FUNCTION_OPEN)?;
    let gt = rest.find('>')?;
    let name = rest[..gt].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let after = rest[gt + 1..].trim_start();
    let close = after.find(FUNCTION_CLOSE)?;
    let params = parse_parameters(&after[..close])?;
    Some((name, Value::Object(params).to_string()))
}

/// The arguments of a function body: zero or more `<parameter=k>` blocks.
fn parse_parameters(text: &str) -> Option<Map<String, Value>> {
    let mut params = Map::new();
    let mut rest = text.trim_start();
    while !rest.is_empty() {
        let after = rest.strip_prefix(PARAMETER_OPEN)?;
        let gt = after.find('>')?;
        let key = after[..gt].trim().to_string();
        if key.is_empty() {
            return None;
        }
        // The template writes `\n` after the opening tag and `\n` before the closing one.
        let after = after[gt + 1..].strip_prefix('\n')?;
        let close = after.find(PARAMETER_CLOSE)?;
        let raw = &after[..close];
        let value = raw.strip_suffix('\n').unwrap_or(raw);
        params.insert(key, parse_value(value));
        rest = after[close + PARAMETER_CLOSE.len()..].trim_start();
    }
    Some(params)
}

/// A parameter's value as the template wrote it: a JSON object, array, quoted string, number,
/// `true`, `false` or `null` is decoded; every other text stays a string.
fn parse_value(text: &str) -> Value {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(_) => Value::String(text.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCallOut;

    fn calls(parse: ToolParse) -> Vec<ToolCallOut> {
        match parse {
            ToolParse::Calls(calls) => calls,
            ToolParse::Content(text) => panic!("expected calls, got content {text:?}"),
        }
    }

    #[test]
    fn qwen3_xml_parser() {
        let parser = Qwen3XmlParser::seeded(5);
        // The template's raw form: the value is unquoted text.
        let one = calls(parser.parse(&format!(
            "{TOOL_CALL_OPEN}\n{FUNCTION_OPEN}get_weather>\n{PARAMETER_OPEN}location>\nParis\n{PARAMETER_CLOSE}\n{FUNCTION_CLOSE}\n{TOOL_CALL_CLOSE}"
        )));
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].name, "get_weather");
        assert_eq!(one[0].arguments, r#"{"location":"Paris"}"#);
        assert!(one[0].id.starts_with("call_"));

        // The grammar's form (JSON-encoded values), a reasoning block, and multi-line/mapping
        // values.
        let quoted = calls(parser.parse(
            "<tool_call>\n<function=get_weather>\n<parameter=location>\n\"Oslo\"</parameter>\n<parameter=unit>\n\"celsius\"</parameter>\n</function>\n</tool_call>",
        ));
        assert_eq!(
            quoted[0].arguments,
            r#"{"location":"Oslo","unit":"celsius"}"#
        );

        let nested = calls(parser.parse(&format!(
            "{REASONING_OPEN}\nlet me check\n{REASONING_CLOSE}\n\n<tool_call>\n<function=f>\n<parameter=a>\n{{\"x\": 1}}\n</parameter>\n<parameter=b>\n[1, 2]\n</parameter>\n</function>\n</tool_call>"
        )));
        assert_eq!(nested[0].name, "f");
        assert_eq!(nested[0].arguments, r#"{"a":{"x":1},"b":[1,2]}"#);

        for text in [
            "The weather in Paris is sunny.",
            "Thinking out loud.",
            "<tool_call>\n<function=f>\n</function>",
            "<tool_call>\n<function=>\n</function>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=></parameter>\n</function>\n</tool_call>",
            "Sure! <tool_call>\n<function=f>\n</function>\n</tool_call>",
            "",
        ] {
            assert_eq!(
                parser.parse(text),
                ToolParse::Content(text.to_string()),
                "{text:?}"
            );
        }
    }
}
