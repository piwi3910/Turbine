//! The `hermes` tool-call format (Qwen3 / Qwen3-MoE; Phase 2m S-11, from the Phase 8 run-ahead).
//!
//! A call is `<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>`; several calls are
//! blocks separated by whitespace, optionally after a leading `<think>…</think>` reasoning
//! block, which a tool-call answer drops. `<tool_call>` / `</tool_call>` are added tokens of the
//! Qwen3 tokenizer that decode to their text (optional here: on another tokenizer they are
//! plain text). In `auto` mode an output opens like a call when its first token is
//! `<tool_call>` or its text, after leading whitespace, starts with `<tool_call>`; it waits
//! while that text is still a prefix of `<tool_call>`. An output that starts with a `<think>`
//! block streams as content, so a call after thinking is not parsed in `auto` mode (an open
//! Phase 8 decision, kept as the run-ahead has it: use `enable_thinking: false` or
//! `tool_choice: required`).
use serde_json::Value;
use turbine_core::registry::Module;

use super::envelope::{CallIds, allowed_tools, call_rules, named_call, text_not_starting_with, ws};
use super::{BoundTokens, ConstraintSpec, Opening, SpecialToken, ToolFormat};
use crate::ModelError;
use crate::tools::{ToolCallParser, ToolChoice, ToolParse};

/// The format's registry name (and the `parser` label of `turbine_tool_calls_total`).
pub const HERMES: &str = "hermes";
/// Opens a call.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closes a call.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// The `hermes` format.
pub struct Hermes;

impl Module for Hermes {
    fn name(&self) -> &'static str {
        HERMES
    }
}

impl ToolFormat for Hermes {
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

    /// `<tool_call>\n{"name": "<tool>", "arguments": <that tool's JSON schema>}\n</tool_call>`
    /// for one of the allowed tools, repeated with whitespace separators when `parallel` (not
    /// for a named function). `auto` is `start: text | calls`: free text that does not start
    /// with `<tool_call>` (a `<think>` block included), or the calls.
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
        let rules = call_rules(
            &allowed,
            "<tool_call>\\n{\\\"name\\\": \\\"",
            "\\\", \\\"arguments\\\": ",
            "}\\n</tool_call>",
        );
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
        Box::new(HermesParser::new())
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
            // Empty, or still a prefix of the opener (`<tool`).
            Opening::Undecided
        } else {
            Opening::Content
        }
    }
}

/// The `hermes` parser: after an optional leading `<think>…</think>` block, one or more
/// `<tool_call>` blocks, each holding exactly `{"name": <string>, "arguments": <object>}`
/// between whitespace, separated by whitespace, and nothing else. Anything else is
/// [`ToolParse::Content`] unchanged.
pub struct HermesParser {
    ids: CallIds,
}

impl std::fmt::Debug for HermesParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HermesParser").finish_non_exhaustive()
    }
}

impl Default for HermesParser {
    fn default() -> Self {
        HermesParser::new()
    }
}

impl HermesParser {
    /// A parser whose call ids come from an OS-seeded stream.
    pub fn new() -> HermesParser {
        HermesParser {
            ids: CallIds::new(),
        }
    }

    /// A parser with reproducible call ids (tests, simulations).
    pub fn seeded(seed: u64) -> HermesParser {
        HermesParser {
            ids: CallIds::seeded(seed),
        }
    }
}

impl ToolCallParser for HermesParser {
    fn parse(&self, text: &str) -> ToolParse {
        match parse_hermes_calls(text) {
            Some(found) => ToolParse::Calls(self.ids.assign(found)),
            None => ToolParse::Content(text.to_string()),
        }
    }
}

/// `(name, compact arguments)` of every call when `text` is entirely Hermes calls.
fn parse_hermes_calls(text: &str) -> Option<Vec<(String, String)>> {
    let mut rest = text.trim_start();
    if let Some(after) = rest.strip_prefix("<think>") {
        let end = after.find("</think>")?;
        rest = after[end + "</think>".len()..].trim_start();
    }
    let mut calls = Vec::new();
    while !rest.is_empty() {
        let body = rest.strip_prefix(TOOL_CALL_OPEN)?;
        let end = body.find(TOOL_CALL_CLOSE)?;
        let value: Value = serde_json::from_str(body[..end].trim()).ok()?;
        calls.push(named_call(value, "arguments", &[])?);
        rest = body[end + TOOL_CALL_CLOSE.len()..].trim_start();
    }
    (!calls.is_empty()).then_some(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCallOut;

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

    /// The run-ahead's parser cases.
    #[test]
    fn hermes_parser() {
        let parser = HermesParser::seeded(3);
        let one = calls(parser.parse(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>",
        ));
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].name, "get_weather");
        assert_eq!(one[0].arguments, r#"{"location":"Paris"}"#);
        assert!(is_call_id(&one[0].id));

        // Two calls, a reasoning block before them (dropped), surrounding whitespace.
        let two = calls(parser.parse(concat!(
            "<think>\nThe user wants two things.\n</think>\n\n",
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": [1, 2]}}\n</tool_call>\n",
            "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>\n"
        )));
        assert_eq!(
            two.iter()
                .map(|c| (c.index, c.name.as_str(), c.arguments.as_str()))
                .collect::<Vec<_>>(),
            [(0, "a", r#"{"x":[1,2]}"#), (1, "b", "{}")]
        );

        for text in [
            "The weather in Paris is sunny.",
            "<think>only thinking</think> and an answer",
            "Sure! <tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n",
            "<tool_call>\n{\"name\": \"f\", \"parameters\": {}}\n</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": \"{}\"}\n</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call> trailing",
            "<think>unterminated <tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>",
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
