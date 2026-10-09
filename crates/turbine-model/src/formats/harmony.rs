//! The `harmony` tool-call format (gpt-oss; Phase 7 S-9).
//!
//! A generation writes one message per channel: `<|channel|>analysis<|message|>` ... the
//! reasoning, then a commentary-channel tool call, then `<|channel|>final<|message|>` ...
//! the content. The template's assistant-turn form puts the recipient before the channel
//! and parses the same.
//!
//! The seven markers are `special: true` tokens of the gpt-oss tokenizer that the
//! detokenizer would skip; the served tool format's markers are emitted by the detokenizer
//! (P7 decision "Harmony marker visibility"), so a parser sees them. They are declared
//! `required: false` (like Hermes's and Mistral's) because the conformance suite binds
//! every format to the tiny tokenizer, which defines none of them.

use serde_json::Value;
use turbine_core::registry::Module;

use super::envelope::{CallIds, allowed_tools, call_rules, text_not_starting_with};
use super::{BoundTokens, ConstraintSpec, Opening, SpecialToken, ToolFormat};
use crate::ModelError;
use crate::tools::{ToolCallParser, ToolChoice, ToolParse};

/// The format's registry name (and the `parser` label of `turbine_tool_calls_total`).
pub const HARMONY: &str = "harmony";
/// Opens a message header.
pub const START: &str = "<|start|>";
/// Opens a channel header.
pub const CHANNEL: &str = "<|channel|>";
/// Ends a message header.
pub const MESSAGE: &str = "<|message|>";
/// Announces the JSON content type.
pub const CONSTRAIN: &str = "<|constrain|>";
/// Ends a channel message.
pub const END: &str = "<|end|>";
/// Ends the generation.
pub const RETURN: &str = "<|return|>";
/// Ends a tool call.
pub const CALL: &str = "<|call|>";
/// The turn opener of a message header.
pub const ASSISTANT: &str = "assistant";
/// The channel a tool call goes to.
pub const COMMENTARY: &str = "commentary";
/// The recipient prefix of a tool call.
pub const TO_FUNCTIONS: &str = "to=functions.";

/// A generation-form call opener: `<|start|>assistant<|channel|>commentary to=`.
const CALL_OPENER: &str = "<|start|>assistant<|channel|>commentary to=";
/// The same without the turn opener: the first call follows the generation prompt, which
/// already ends with `<|start|>assistant`.
const CALL_OPENER_SHORT: &str = "<|channel|>commentary to=";

/// The `harmony` format.
pub struct Harmony;

impl Module for Harmony {
    fn name(&self) -> &'static str {
        HARMONY
    }
}

impl ToolFormat for Harmony {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        &[
            SpecialToken {
                text: START,
                required: false,
            },
            SpecialToken {
                text: CHANNEL,
                required: false,
            },
            SpecialToken {
                text: MESSAGE,
                required: false,
            },
            SpecialToken {
                text: CONSTRAIN,
                required: false,
            },
            SpecialToken {
                text: END,
                required: false,
            },
            SpecialToken {
                text: RETURN,
                required: false,
            },
            SpecialToken {
                text: CALL,
                required: false,
            },
        ]
    }

    /// `required` and named calls are constrained to the commentary-channel call form
    /// `<|start|>assistant<|channel|>commentary to=functions.<name> <|constrain|>json
    /// <|message|>` + JSON + `<|call|>`, several back to back when `parallel` (each call
    /// carries its own `<|start|>assistant`, so the separator is empty). `auto` is
    /// `start: text | calls`: free text that does not open a call, or the calls.
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        let allowed = allowed_tools(tools, choice)?;
        let repeat = parallel && matches!(choice, ToolChoice::Auto | ToolChoice::Required);
        let calls = if repeat { "call (call)*" } else { "call" };
        let rules = call_rules(
            &allowed,
            "<|start|>assistant<|channel|>commentary to=functions.",
            " <|constrain|>json<|message|>",
            "<|call|>",
        );
        let start = match choice {
            ToolChoice::Auto => format!(
                "start: text | calls\ntext: /{}/\ncalls: {calls}\n",
                text_not_starting_with(CALL_OPENER),
            ),
            _ => format!("start: {calls}\n"),
        };
        Ok(ConstraintSpec::ToolCall {
            grammar_source: format!("{start}{rules}"),
        })
    }

    fn parser(&self) -> Box<dyn ToolCallParser> {
        Box::new(HarmonyParser::new())
    }

    /// An `auto` output opens like a call when its text (what the reasoning parser released
    /// as content) opens the commentary-call form; it stays undecided while the text could
    /// still become a call opener or a channel/turn marker.
    fn opens_like_call(
        &self,
        text: &str,
        _first_token: Option<u32>,
        _tokens: &BoundTokens,
    ) -> Opening {
        if text.starts_with(CALL_OPENER) || text.starts_with(CALL_OPENER_SHORT) {
            return Opening::Call;
        }
        let waiting = [CALL_OPENER, CALL_OPENER_SHORT, CHANNEL, START]
            .iter()
            .any(|marker| marker.starts_with(text));
        if waiting {
            Opening::Undecided
        } else {
            Opening::Content
        }
    }

    /// The commentary-channel form of `get_weather`.
    fn sample_call(&self) -> &'static str {
        "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Oslo\", \"unit\": \"celsius\"}<|call|>"
    }
}

/// The `harmony` parser: zero or more commentary-channel calls, in either the generation form
/// (`<|channel|>commentary to=functions.<name> [ <|constrain|>json]<|message|>{...}<|call|>`,
/// optionally after `<|start|>assistant`) or the template's assistant-turn form
/// (`<|start|>assistant to=functions.<name><|channel|>commentary json<|message|>{...}
/// <|call|>`), back to back. Anything else is [`ToolParse::Content`] unchanged.
pub struct HarmonyParser {
    ids: CallIds,
}

impl std::fmt::Debug for HarmonyParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HarmonyParser")
    }
}

impl Default for HarmonyParser {
    fn default() -> Self {
        HarmonyParser::new()
    }
}

impl HarmonyParser {
    /// A parser whose call ids come from an OS-seeded stream.
    pub fn new() -> HarmonyParser {
        HarmonyParser {
            ids: CallIds::new(),
        }
    }

    /// A parser with reproducible call ids (tests, simulations).
    pub fn seeded(seed: u64) -> HarmonyParser {
        HarmonyParser {
            ids: CallIds::seeded(seed),
        }
    }
}

impl ToolCallParser for HarmonyParser {
    fn parse(&self, text: &str) -> ToolParse {
        match parse_harmony_calls(text) {
            Some(found) => ToolParse::Calls(self.ids.assign(found)),
            None => ToolParse::Content(text.to_string()),
        }
    }
}

/// `(name, compact arguments)` of every call when `text` is entirely Harmony calls.
fn parse_harmony_calls(text: &str) -> Option<Vec<(String, String)>> {
    let mut rest = text;
    let mut calls = Vec::new();
    while !rest.is_empty() {
        let mut after = rest.strip_prefix(START).unwrap_or(rest);
        after = after.strip_prefix(ASSISTANT).unwrap_or(after);
        let (name, body) = match after.strip_prefix(CHANNEL) {
            // The generation form: `<|channel|>commentary to=functions.<name> ...<|message|>`.
            Some(header) => {
                let at = header.find(MESSAGE)?;
                let name = generation_name(header.get(..at)?)?;
                (name, header.get(at + MESSAGE.len()..)?)
            }
            // The template's assistant-turn form: ` to=functions.<name><|channel|>commentary
            // ...<|message|>`.
            None => {
                let at = after.find(CHANNEL)?;
                let name = recipient_name(after.get(..at)?)?;
                let header = after.get(at + CHANNEL.len()..)?;
                let msg = header.find(MESSAGE)?;
                commentary_header(header.get(..msg)?)?;
                (name, header.get(msg + MESSAGE.len()..)?)
            }
        };
        let end = body.find(CALL)?;
        let arguments: Value = serde_json::from_str(body.get(..end)?).ok()?;
        if !arguments.is_object() {
            return None;
        }
        calls.push((name, arguments.to_string()));
        rest = body.get(end + CALL.len()..)?;
    }
    (!calls.is_empty()).then_some(calls)
}

/// The tool name of a generation-form header `commentary to=functions.<name>` plus an
/// optional `<|constrain|>json` / `json` content type, trimmed.
fn generation_name(header: &str) -> Option<String> {
    let rest = header.trim();
    let rest = rest.strip_prefix(COMMENTARY)?.trim_start();
    let rest = rest.strip_prefix(TO_FUNCTIONS)?;
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let name = rest[..end].to_string();
    (!name.is_empty()).then_some(name)
}

/// The tool name of a template-form recipient ` to=functions.<name>`.
fn recipient_name(recipient: &str) -> Option<String> {
    let name = recipient.trim().strip_prefix(TO_FUNCTIONS)?;
    (!name.is_empty()).then_some(name.to_string())
}

/// A template-form channel header: `commentary` plus an optional `json` content type
/// (possibly announced with `<|constrain|>`). The name itself was before the channel.
fn commentary_header(header: &str) -> Option<()> {
    let rest = header.trim();
    let rest = rest.strip_prefix(COMMENTARY)?.trim();
    let rest = rest.strip_prefix(CONSTRAIN).unwrap_or(rest);
    match rest {
        "" | "json" => Some(()),
        _ => None,
    }
}
