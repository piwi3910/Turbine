//! Building blocks the enveloped tool formats share ([`super::hermes`], [`super::mistral`];
//! from the Phase 8 run-ahead's `tools.rs`): the tools a `tool_choice` allows, the per-tool call
//! rules of a grammar, the `auto` free-text regex, the call-object check of a parser and call
//! ids. Not a format itself.
use std::sync::Mutex;

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use serde_json::{Map, Value};

use crate::ModelError;
use crate::structured::{JSON_MAX_WHITESPACE, with_json_options};
use crate::tools::{FunctionTool, ToolCallOut, ToolChoice, function_tools, invalid, new_call_id};

/// The tools of `tools` a `choice` lets the model call: all of them for `auto` / `required`,
/// the named one for a named function; `none`, an empty list or an unknown name is a
/// [`ModelError::Constraint`] naming the field (the `llama3_json` rules).
pub(crate) fn allowed_tools<'a>(
    tools: &'a [Value],
    choice: &ToolChoice,
) -> Result<Vec<FunctionTool<'a>>, ModelError> {
    let tools = function_tools(tools)?;
    if tools.is_empty() {
        return Err(invalid("tools", "[]", "at least one function tool"));
    }
    match choice {
        ToolChoice::None => Err(invalid(
            "tool_choice",
            choice.as_str(),
            "auto, required, a named function",
        )),
        ToolChoice::Auto | ToolChoice::Required => Ok(tools),
        ToolChoice::Named(name) => {
            let names: Vec<&str> = tools.iter().map(|t| t.name).collect();
            match tools.into_iter().find(|t| t.name == name) {
                Some(tool) => Ok(vec![tool]),
                None => Err(invalid(
                    "tool_choice.function.name",
                    name.as_str(),
                    &names.join(", "),
                )),
            }
        }
    }
}

/// Lark regex of the whitespace a grammar allows (bounded like the JSON inside).
pub(crate) fn ws(min: usize) -> String {
    format!("[\\x20\\x0A\\x0D\\x09]{{{min},{JSON_MAX_WHITESPACE}}}")
}

/// A Lark regex body matching text whose first non-whitespace character (after at most
/// [`JSON_MAX_WHITESPACE`]) exists and that does not start with `prefix` (ASCII): the free text
/// of an `auto` output. A proper prefix of `prefix` alone is accepted too (`<` of
/// `<tool_call>`).
pub(crate) fn text_not_starting_with(prefix: &str) -> String {
    let esc = |c: char| match c {
        '[' | ']' | '\\' | '^' | '-' | '{' | '}' | '(' | ')' | '.' | '*' | '+' | '?' | '|'
        | '$' | '/' => format!("\\{c}"),
        c => c.to_string(),
    };
    let chars: Vec<char> = prefix.chars().collect();
    let mut alternatives = Vec::new();
    for i in 0..chars.len() {
        let head: String = chars[..i].iter().map(|&c| esc(c)).collect();
        // The first character also excludes whitespace (it is the first non-whitespace one).
        let class = if i == 0 {
            format!("[^{}\\x20\\x0A\\x0D\\x09]", esc(chars[0]))
        } else {
            format!("[^{}]", esc(chars[i]))
        };
        alternatives.push(format!("{head}{class}(?s:.)*"));
        if i > 0 {
            alternatives.push(head);
        }
    }
    format!("{}({})", ws(0), alternatives.join("|"))
}

/// The `call` alternatives and `call_<i>` / `params_<i>` rules of `allowed`, each call being
/// `before` + the tool's name + `middle` + its parameters + `after` (Lark string bodies, JSON
/// quotes already escaped). The parameters take the tool's JSON schema with the bounded
/// natural JSON whitespace ([`with_json_options`]); a tool without `parameters` takes any
/// object.
pub(crate) fn call_rules(
    allowed: &[FunctionTool<'_>],
    before: &str,
    middle: &str,
    after: &str,
) -> String {
    let alternatives: Vec<String> = (0..allowed.len()).map(|i| format!("call_{i}")).collect();
    let mut rules = format!("call: {}\n", alternatives.join(" | "));
    for (i, tool) in allowed.iter().enumerate() {
        // Names are [A-Za-z0-9_-] only (`function_tools`), so they need no escaping.
        rules.push_str(&format!(
            "call_{i}: \"{before}{}{middle}\" params_{i} \"{after}\"\n",
            tool.name
        ));
        let schema = tool.parameters.cloned().unwrap_or_else(|| {
            let mut any_object = Map::new();
            any_object.insert("type".into(), "object".into());
            any_object
        });
        let schema = with_json_options(&Value::Object(schema));
        rules.push_str(&format!("params_{i}: %json {schema}\n"));
    }
    rules
}

/// `(name, compact arguments)` of a call object with exactly a string `name`, an object under
/// `args_key` and, optionally, string `optional` keys (ignored).
pub(crate) fn named_call(
    value: Value,
    args_key: &str,
    optional: &[&str],
) -> Option<(String, String)> {
    let Value::Object(mut object) = value else {
        return None;
    };
    for key in optional {
        match object.remove(*key) {
            None | Some(Value::String(_)) => {}
            Some(_) => return None,
        }
    }
    if object.len() != 2 {
        return None;
    }
    let Some(Value::String(name)) = object.remove("name") else {
        return None;
    };
    let Some(arguments @ Value::Object(_)) = object.remove(args_key) else {
        return None;
    };
    Some((name, arguments.to_string()))
}

/// A parser's source of call ids.
pub(crate) struct CallIds(Mutex<ChaCha8Rng>);

impl CallIds {
    /// OS-seeded.
    pub(crate) fn new() -> CallIds {
        CallIds(Mutex::new(ChaCha8Rng::from_os_rng()))
    }

    /// Reproducible (tests, simulations).
    pub(crate) fn seeded(seed: u64) -> CallIds {
        CallIds(Mutex::new(ChaCha8Rng::seed_from_u64(seed)))
    }

    /// `found` calls as [`ToolCallOut`]s indexed in order, with fresh ids.
    pub(crate) fn assign(&self, found: Vec<(String, String)>) -> Vec<ToolCallOut> {
        // A poisoned lock only means another parse panicked mid-draw; the stream is still valid.
        let mut rng = self.0.lock().unwrap_or_else(|p| p.into_inner());
        found
            .into_iter()
            .zip(0u32..)
            .map(|((name, arguments), index)| ToolCallOut {
                index,
                id: new_call_id(&mut *rng),
                name,
                arguments,
            })
            .collect()
    }
}
