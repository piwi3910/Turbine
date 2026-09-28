//! The `tool_format` suite: every registered format binds to the tiny tokenizer, compiles its
//! grammars, parses its own sample call ([`ToolFormat::sample_call`]) and lets that call
//! through its compiled grammars on the tiny tokenizer (the constrained sampler's matcher).

use serde_json::{Value, json};
use turbine_core::registry::Registry;

use super::{ConformanceFailure, Report, ensure};
use crate::ModelError;
use crate::formats::{BoundToolFormat, Opening, ToolFormat, bind};
use crate::structured::{GrammarCompiler, GrammarLimits, TokenMask, TokenMatcher, step_mask};
use crate::testing::TempDir;
use crate::testing::tiny::{TINY_EOS, write_tiny_llama};
use crate::tokenizer::Tokenizer;
use crate::tools::{ToolChoice, ToolParse};

/// Plain text every format must leave as content (and whose opening is content).
const CONTENT: &str = "The weather in Oslo is sunny.";

/// The fixture tools: `get_weather` (a `location` string, required, and a `unit` enum) and
/// `get_time` (no parameters). Every format's [`ToolFormat::sample_call`] calls `get_weather`
/// with `{"location": "Oslo", "unit": "celsius"}`.
pub fn fixture_tools() -> Vec<Value> {
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

/// Runs `text` (tokenized by `tokenizer`, special tokens included) through `matcher`,
/// checking each token against the step mask first. `Ok(accepting)` when every token was
/// allowed; `Err` names the first disallowed token.
pub fn feed_text(
    matcher: &mut dyn TokenMatcher,
    tokenizer: &Tokenizer,
    eos: &[u32],
    text: &str,
) -> Result<bool, String> {
    let ids = tokenizer.encode(text, false).map_err(|e| e.to_string())?;
    let mut mask = TokenMask::new_none(tokenizer.vocab_size() as usize);
    for (i, &id) in ids.iter().enumerate() {
        step_mask(matcher, eos, &mut mask).map_err(|e| format!("token {i}: {e}"))?;
        if !mask.is_allowed(id) {
            let done = tokenizer.decode(&ids[..i], false).unwrap_or_default();
            let piece = tokenizer.decode(&[id], false).unwrap_or_default();
            return Err(format!("token {i} {piece:?} disallowed after {done:?}"));
        }
        matcher.commit(id).map_err(|e| format!("token {i}: {e}"))?;
    }
    Ok(matcher.accepts_eos())
}

/// Runs every check over every format of `reg`; `Err` lists each broken check.
///
/// Per format: `special_tokens` (non-empty, distinct, and it binds to the tiny tokenizer),
/// `render` (no tool block, or a non-empty one), `grammar` (compiles for `auto` parallel,
/// `required` single and parallel and a named function; refuses `none`, no tools and an
/// unknown name), `parse` (its sample call parses into the `get_weather` call; plain text
/// stays content), `round_trip` (the compiled `required`, `auto` and named grammars accept the
/// sample call token by token; `required` refuses plain text), `opening` (the sample call
/// opens like a call, plain text like content).
pub fn formats_suite(reg: &Registry<dyn ToolFormat>) -> Result<(), Vec<ConformanceFailure>> {
    let mut report = Report::new(reg);
    let dir = TempDir::new("turbine-conformance-formats");
    let mut tiny = None;
    report.check("registry", "tokenizer", || {
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer =
            Tokenizer::from_file(&spec.dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let compiler = GrammarCompiler::new(&tokenizer, &TINY_EOS).map_err(|e| e.to_string())?;
        tiny = Some((tokenizer, compiler));
        Ok(())
    });
    let Some((tokenizer, compiler)) = tiny else {
        return report.finish();
    };
    let tools = fixture_tools();
    let weather = ToolChoice::Named("get_weather".into());
    let limits = GrammarLimits {
        max_schema_bytes: 64 * 1024,
    };

    for format in reg.iter() {
        let name = format.name();
        let mut bound = None;
        report.check(name, "special_tokens", || {
            let texts: Vec<&str> = format.special_tokens().iter().map(|t| t.text).collect();
            for (i, text) in texts.iter().enumerate() {
                ensure(!text.is_empty(), || format!("special token {i} is empty"))?;
                ensure(!texts[..i].contains(text), || format!("{text} twice"))?;
            }
            bound = Some(bind(format, &tokenizer).map_err(|e| e.to_string())?);
            Ok(())
        });
        report.check(name, "render", || match format.render_tools(&tools) {
            Some(block) if block.trim().is_empty() => Err("an empty tool block".into()),
            _ => Ok(()),
        });
        report.check(name, "grammar", || {
            for (choice, parallel) in [
                (&ToolChoice::Auto, true),
                (&ToolChoice::Required, false),
                (&ToolChoice::Required, true),
                (&weather, false),
            ] {
                let spec = format
                    .grammar(&tools, choice, parallel)
                    .map_err(|e| format!("{choice:?} parallel={parallel}: {e}"))?;
                compiler
                    .compile(&spec, &limits)
                    .map_err(|e| format!("{choice:?} parallel={parallel} does not compile: {e}"))?;
            }
            for (tools, choice) in [
                (&tools[..], ToolChoice::None),
                (&[][..], ToolChoice::Required),
                (&tools[..], ToolChoice::Named("get_stock".into())),
            ] {
                ensure(
                    matches!(
                        format.grammar(tools, &choice, false),
                        Err(ModelError::Constraint(_))
                    ),
                    || format!("{choice:?} over {} tools is not refused", tools.len()),
                )?;
            }
            Ok(())
        });
        report.check(name, "parse", || {
            let parser = format.parser();
            match parser.parse(format.sample_call()) {
                ToolParse::Calls(calls) => {
                    let got: Vec<(u32, &str, Option<Value>)> = calls
                        .iter()
                        .map(|c| {
                            (
                                c.index,
                                c.name.as_str(),
                                serde_json::from_str(&c.arguments).ok(),
                            )
                        })
                        .collect();
                    let want = json!({"location": "Oslo", "unit": "celsius"});
                    ensure(got == [(0, "get_weather", Some(want))], || {
                        format!("the sample call parses into {got:?}")
                    })?;
                }
                ToolParse::Content(_) => {
                    return Err("the sample call parses as content".into());
                }
            }
            ensure(
                parser.parse(CONTENT) == ToolParse::Content(CONTENT.into()),
                || format!("{CONTENT:?} parses as a call"),
            )
        });
        report.check(name, "round_trip", || {
            let feed = |choice: &ToolChoice, parallel: bool, text: &str| {
                let spec = format
                    .grammar(&tools, choice, parallel)
                    .map_err(|e| e.to_string())?;
                let mut matcher = compiler
                    .compile(&spec, &limits)
                    .map_err(|e| e.to_string())?;
                feed_text(matcher.as_mut(), &tokenizer, &TINY_EOS, text)
            };
            let sample = format.sample_call();
            for (choice, parallel) in [
                (&ToolChoice::Required, false),
                (&ToolChoice::Auto, true),
                (&weather, false),
            ] {
                let got = feed(choice, parallel, sample);
                ensure(got == Ok(true), || {
                    format!("{choice:?} parallel={parallel} on the sample call: {got:?}")
                })?;
            }
            let got = feed(&ToolChoice::Required, false, CONTENT);
            ensure(got != Ok(true), || {
                format!("`required` accepts plain text {CONTENT:?}")
            })
        });
        report.check(name, "opening", || {
            let bound: &BoundToolFormat = bound.as_ref().ok_or("the format does not bind")?;
            let got = bound.opens_like_call(format.sample_call(), None);
            ensure(got == Opening::Call, || {
                format!("the sample call opens as {got:?}")
            })?;
            let first = tokenizer
                .encode(CONTENT, false)
                .map_err(|e| e.to_string())?
                .first()
                .copied();
            let got = bound.opens_like_call(CONTENT, first);
            ensure(got == Opening::Content, || {
                format!("{CONTENT:?} opens as {got:?}")
            })
        });
    }
    report.finish()
}
