//! Tool-call formats (Phase 2m S-4, contract §24 `tool_format`): everything that differs
//! between the ways models write tool calls — the special tokens a format needs from the
//! tokenizer, optional tool rendering (when the chat template does not render `tools`), the
//! llguidance grammar that constrains `tool_choice` `auto` / `required` / named output, the
//! parser that turns a finished output into `tool_calls`, and the `auto`-mode test of whether an
//! output opens like a call. One file per format and one entry in [`registry`]; the server
//! resolves `model.tool_call_parser` (else the family's default) here and [`bind`]s the format
//! to the tokenizer once at startup.
//!
//! The format-neutral parts (the [`ToolCallParser`] trait, [`ToolParse`](crate::ToolParse),
//! [`ToolChoice`], call ids and the checks of a request's `tools`) stay in [`crate::tools`].

use serde_json::Value;
use turbine_core::registry::{Module, Registry};
pub use turbine_core::request::ConstraintSpec;

use crate::ModelError;
use crate::tokenizer::Tokenizer;
use crate::tools::{ToolCallParser, ToolChoice};

mod envelope;
pub mod hermes;
pub mod llama3_json;
pub mod mistral;

pub use hermes::{Hermes, HermesParser};
pub use llama3_json::{Llama3Json, Llama3JsonParser};
pub use mistral::{Mistral, MistralParser};

/// One way of writing tool calls.
pub trait ToolFormat: Module {
    /// The special tokens the format uses; a `required` one the tokenizer lacks refuses the
    /// format at startup ([`bind`]).
    fn special_tokens(&self) -> &'static [SpecialToken];
    /// The tool block to put in the prompt when the chat template renders no `tools`; `None`
    /// leaves rendering to the template (no current format needs this).
    fn render_tools(&self, tools: &[Value]) -> Option<String> {
        let _ = tools;
        None
    }
    /// The llguidance grammar for `tool_choice` `auto`, `required` or a named function over
    /// `tools` (one call, or several when `parallel`); a malformed tool, an unknown named
    /// function or `none` is a [`ModelError::Constraint`] naming the field.
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError>;
    /// A parser for finished outputs, with call ids from an OS-seeded stream.
    fn parser(&self) -> Box<dyn ToolCallParser>;
    /// `auto` mode: whether the output so far — `text`, the decoded text since the start, and
    /// `first_token`, the first generated token id — opens like a call ([`Opening::Call`]: hold
    /// the rest for the parser), is content ([`Opening::Content`]: stream it) or cannot tell
    /// yet ([`Opening::Undecided`]). `tokens` are the format's special tokens on this
    /// tokenizer.
    fn opens_like_call(
        &self,
        text: &str,
        first_token: Option<u32>,
        tokens: &BoundTokens,
    ) -> Opening;
}

impl std::fmt::Debug for dyn ToolFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A special token a format uses, by its text.
#[derive(Clone, Copy, Debug)]
pub struct SpecialToken {
    pub text: &'static str,
    /// The format cannot work without it: a tokenizer lacking it refuses the format.
    pub required: bool,
}

/// A format's special tokens resolved on one tokenizer: `None` for an optional token the
/// tokenizer does not define.
#[derive(Clone, Debug, Default)]
pub struct BoundTokens {
    ids: Vec<(&'static str, Option<u32>)>,
}

impl BoundTokens {
    /// The id of the special token `text`; `None` when the tokenizer lacks it or the format
    /// does not name it.
    pub fn id(&self, text: &str) -> Option<u32> {
        self.ids
            .iter()
            .find(|(t, _)| *t == text)
            .and_then(|&(_, id)| id)
    }
}

/// What an `auto` output is, as far as its opening shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opening {
    /// It opens like a call: hold it for the parser.
    Call,
    /// Plain content: stream it.
    Content,
    /// Not decided yet (e.g. only whitespace so far): keep it pending.
    Undecided,
}

/// A format bound to the served model's tokenizer.
#[derive(Debug)]
pub struct BoundToolFormat {
    pub format: &'static dyn ToolFormat,
    pub tokens: BoundTokens,
}

impl BoundToolFormat {
    /// [`ToolFormat::opens_like_call`] with this binding's tokens.
    pub fn opens_like_call(&self, text: &str, first_token: Option<u32>) -> Opening {
        self.format.opens_like_call(text, first_token, &self.tokens)
    }
}

/// Resolves `format`'s special tokens on `tokenizer`; a `required` token the tokenizer lacks
/// is [`ModelError::Unsupported`] naming the format and the token.
pub fn bind(
    format: &'static dyn ToolFormat,
    tokenizer: &Tokenizer,
) -> Result<BoundToolFormat, ModelError> {
    let mut ids = Vec::with_capacity(format.special_tokens().len());
    for token in format.special_tokens() {
        let id = tokenizer.token_to_id(token.text);
        if id.is_none() && token.required {
            return Err(ModelError::Unsupported {
                field: format!("tool format {} special token", format.name()),
                value: token.text.to_string(),
                supported: "a tokenizer that defines it".into(),
            });
        }
        ids.push((token.text, id));
    }
    Ok(BoundToolFormat {
        format,
        tokens: BoundTokens { ids },
    })
}

static TOOL_FORMATS: Registry<dyn ToolFormat> =
    Registry::new("tool_format", &[&Llama3Json, &Hermes, &Mistral]);

/// Every tool-call format, in registration order.
pub fn registry() -> &'static Registry<dyn ToolFormat> {
    &TOOL_FORMATS
}

#[cfg(test)]
mod tests;
