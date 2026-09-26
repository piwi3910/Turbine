//! The `mistral` tool-call format (Mistral v0.3 template, Mistral / Mixtral; Phase 2m S-11,
//! from the Phase 8 run-ahead).
//!
//! A call is an optional `[TOOL_CALLS]` control token, then one JSON array of
//! `{"name": …, "arguments": {…}}` objects (an `id` string key is accepted and ignored).
//! `[TOOL_CALLS]` is a special token that decodes to no text; the grammar leaves it out, so it
//! compiles on any tokenizer, and the parser accepts calls without it (an open Phase 8 decision,
//! kept as the run-ahead has it). In `auto` mode an output opens like a call when its first
//! token is `[TOOL_CALLS]` or its first non-whitespace character is `[`.
use serde_json::Value;
use turbine_core::registry::Module;

use super::envelope::{CallIds, allowed_tools, call_rules, named_call, text_not_starting_with, ws};
use super::{BoundTokens, ConstraintSpec, Opening, SpecialToken, ToolFormat};
use crate::ModelError;
use crate::tools::{ToolCallParser, ToolChoice, ToolParse};

/// The format's registry name (and the `parser` label of `turbine_tool_calls_total`).
pub const MISTRAL: &str = "mistral";
/// The control token that opens Mistral's calls (special: it decodes to no text).
pub const TOOL_CALLS: &str = "[TOOL_CALLS]";

/// The `mistral` format.
pub struct Mistral;

impl Module for Mistral {
    fn name(&self) -> &'static str {
        MISTRAL
    }
}

impl ToolFormat for Mistral {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        &[SpecialToken {
            text: TOOL_CALLS,
            required: false,
        }]
    }

    /// `[{"name": "<tool>", "arguments": <that tool's JSON schema>}]` for one of the allowed
    /// tools, several `, `-separated objects in the array when `parallel` (not for a named
    /// function). `auto` is `start: text | calls`: free text whose first non-whitespace
    /// character is not `[`, or the calls.
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        let allowed = allowed_tools(tools, choice)?;
        let repeat = parallel && matches!(choice, ToolChoice::Auto | ToolChoice::Required);
        let (calls, sep) = if repeat {
            (
                "\"[\" call (CALL_SEP call)* \"]\"",
                format!("CALL_SEP: /,{}/\n", ws(0)),
            )
        } else {
            ("\"[\" call \"]\"", String::new())
        };
        let rules = call_rules(
            &allowed,
            "{\\\"name\\\": \\\"",
            "\\\", \\\"arguments\\\": ",
            "}",
        );
        let start = match choice {
            ToolChoice::Auto => format!(
                "start: text | calls\ntext: /{}/\ncalls: {calls}\n",
                text_not_starting_with("[")
            ),
            _ => format!("start: {calls}\n"),
        };
        Ok(ConstraintSpec::ToolCall {
            grammar_source: format!("{start}{sep}{rules}"),
        })
    }

    fn parser(&self) -> Box<dyn ToolCallParser> {
        Box::new(MistralParser::new())
    }

    fn opens_like_call(
        &self,
        text: &str,
        first_token: Option<u32>,
        tokens: &BoundTokens,
    ) -> Opening {
        if first_token.is_some() && first_token == tokens.id(TOOL_CALLS) {
            return Opening::Call;
        }
        let start = text.trim_start();
        if start.is_empty() {
            Opening::Undecided
        } else if start.starts_with('[') {
            Opening::Call
        } else {
            Opening::Content
        }
    }

    fn sample_call(&self) -> &'static str {
        r#"[{"name": "get_weather", "arguments": {"location": "Oslo", "unit": "celsius"}}]"#
    }
}

/// The `mistral` parser: an optional `[TOOL_CALLS]`, then one JSON array of one or more
/// `{"name": <string>, "arguments": <object>}` objects (an optional string `id` is ignored),
/// and nothing else. Anything else is [`ToolParse::Content`] unchanged.
pub struct MistralParser {
    ids: CallIds,
}

impl std::fmt::Debug for MistralParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralParser").finish_non_exhaustive()
    }
}

impl Default for MistralParser {
    fn default() -> Self {
        MistralParser::new()
    }
}

impl MistralParser {
    /// A parser whose call ids come from an OS-seeded stream.
    pub fn new() -> MistralParser {
        MistralParser {
            ids: CallIds::new(),
        }
    }

    /// A parser with reproducible call ids (tests, simulations).
    pub fn seeded(seed: u64) -> MistralParser {
        MistralParser {
            ids: CallIds::seeded(seed),
        }
    }
}

impl ToolCallParser for MistralParser {
    fn parse(&self, text: &str) -> ToolParse {
        match parse_mistral_calls(text) {
            Some(found) => ToolParse::Calls(self.ids.assign(found)),
            None => ToolParse::Content(text.to_string()),
        }
    }
}

/// `(name, compact arguments)` of every call when `text` is entirely one Mistral call array.
fn parse_mistral_calls(text: &str) -> Option<Vec<(String, String)>> {
    let mut rest = text.trim_start();
    if let Some(after) = rest.strip_prefix(TOOL_CALLS) {
        rest = after.trim_start();
    }
    let Value::Array(items) = serde_json::from_str::<Value>(rest).ok()? else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    items
        .into_iter()
        .map(|item| named_call(item, "arguments", &["id"]))
        .collect()
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

    /// The run-ahead's parser cases.
    #[test]
    fn mistral_parser() {
        let parser = MistralParser::seeded(4);
        let got = calls(parser.parse(concat!(
            "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}, ",
            "\"id\": \"abc123XYZ\"}, {\"name\": \"get_time\", \"arguments\": {}}]"
        )));
        assert_eq!(
            got.iter()
                .map(|c| (c.index, c.name.as_str(), c.arguments.as_str()))
                .collect::<Vec<_>>(),
            [
                (0, "get_weather", r#"{"location":"Paris"}"#),
                (1, "get_time", "{}")
            ]
        );
        // The control token decodes to no text: a bare array is a call too.
        let bare = calls(parser.parse(r#" [{"name": "f", "arguments": {"a": 1}}]"#));
        assert_eq!(
            (bare[0].name.as_str(), bare[0].arguments.as_str()),
            ("f", r#"{"a":1}"#)
        );

        for text in [
            "Plain answer.",
            "[]",
            "[1, 2]",
            r#"[{"name": "f"}]"#,
            r#"[{"name": "f", "arguments": {}, "id": 7}]"#,
            r#"[{"name": "f", "arguments": {}, "extra": 1}]"#,
            r#"[{"name": "f", "arguments": {}}] and more"#,
            r#"{"name": "f", "arguments": {}}"#,
        ] {
            assert_eq!(
                parser.parse(text),
                ToolParse::Content(text.to_string()),
                "{text:?}"
            );
        }
    }
}
