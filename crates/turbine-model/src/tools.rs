//! Tool calling (P2 S-18), the format-neutral parts: [`ToolParse`], the [`ToolCallParser`]
//! trait, [`ToolChoice`], call ids and the checks of a request's `tools`. Each way of writing
//! calls — its grammar, parser, special tokens and `auto` opening rule — is a
//! [`ToolFormat`](crate::formats::ToolFormat) in [`crate::formats`] (Phase 2m S-4);
//! [`tool_call_grammar`] is the `llama3_json` grammar, kept for its callers.
use rand_core::RngCore;
use serde_json::{Map, Value};
pub use turbine_core::request::{ConstraintSpec, ToolCallOut};

use crate::ModelError;
use crate::formats::{Llama3Json, ToolFormat};

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
    pub(crate) fn as_str(&self) -> &str {
        match self {
            ToolChoice::None => "none",
            ToolChoice::Auto => "auto",
            ToolChoice::Required => "required",
            ToolChoice::Named(name) => name,
        }
    }
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
pub(crate) struct FunctionTool<'a> {
    pub name: &'a str,
    pub parameters: Option<&'a Map<String, Value>>,
}

/// A tool definition or `tool_choice` the grammar cannot be built from: a
/// [`ModelError::Constraint`] (the server answers 400 `invalid_json_schema`) naming the field.
pub(crate) fn invalid(
    field: impl Into<String>,
    value: impl Into<String>,
    supported: &str,
) -> ModelError {
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

pub(crate) fn function_tools(tools: &[Value]) -> Result<Vec<FunctionTool<'_>>, ModelError> {
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

/// The `llama3_json` grammar for an `auto`, `required` or named `tool_choice`
/// ([`Llama3Json`]'s [`ToolFormat::grammar`]); `none` is unconstrained and has no grammar, an
/// unknown named function or a malformed tool is a [`ModelError::Constraint`] naming the field.
pub fn tool_call_grammar(
    tools: &[Value],
    choice: &ToolChoice,
    parallel: bool,
) -> Result<ConstraintSpec, ModelError> {
    Llama3Json.grammar(tools, choice, parallel)
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;

    fn is_call_id(id: &str) -> bool {
        id.strip_prefix("call_")
            .is_some_and(|rest| rest.len() == 24 && rest.bytes().all(|b| b.is_ascii_alphanumeric()))
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
}
