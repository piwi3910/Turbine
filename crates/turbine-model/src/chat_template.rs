//! The model's Jinja chat template (P1 S-4, P2 S-18), rendered with `minijinja` configured to match
//! transformers' `apply_chat_template` environment: `trim_blocks` + `lstrip_blocks`, loop controls,
//! Python string/dict methods (`minijinja-contrib` pycompat), Python-compatible `trim` and
//! `tojson`, and the host functions `raise_exception` and `strftime_now` (UTC).
use std::fmt::Write as _;
use std::path::Path;
use std::time::SystemTime;

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, State, UndefinedBehavior};

use crate::ModelError;

/// A compiled chat template plus the special-token strings it may reference.
pub struct ChatTemplate {
    env: Environment<'static>,
    name: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    renders_tools: bool,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("name", &self.name)
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .field("renders_tools", &self.renders_tools)
            .finish()
    }
}

/// Special tokens and (optionally) the template read from a `tokenizer_config.json`.
struct TokenizerConfig {
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

/// Marker source attached to the error raised by the template's `raise_exception(msg)`.
#[derive(Debug)]
struct RaisedException(String);

impl std::fmt::Display for RaisedException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RaisedException {}

impl ChatTemplate {
    /// `path` is either a `.jinja` template (special tokens then come from a sibling
    /// `tokenizer_config.json` if present) or a `tokenizer_config.json` (template from its
    /// `chat_template` key).
    pub fn load(path: &Path) -> Result<ChatTemplate, ModelError> {
        let is_json = path.extension().is_some_and(|ext| ext == "json");
        let (source, config) = if is_json {
            let config = read_tokenizer_config(path)?;
            let Some(source) = config.chat_template.clone() else {
                return Err(ModelError::Template(format!(
                    "{}: no chat_template",
                    path.display()
                )));
            };
            (source, Some(config))
        } else {
            let source = std::fs::read_to_string(path).map_err(|e| ModelError::Io {
                path: path.to_path_buf(),
                detail: e.to_string(),
            })?;
            let sibling = path
                .parent()
                .map(|dir| dir.join("tokenizer_config.json"))
                .filter(|p| p.is_file());
            let config = match sibling {
                Some(p) => Some(read_tokenizer_config(&p)?),
                None => None,
            };
            (source, config)
        };
        let (bos_token, eos_token) = match config {
            Some(c) => (c.bos_token, c.eos_token),
            None => (None, None),
        };
        Self::compile(path, source, bos_token, eos_token)
    }

    /// Default resolution for `model.chat_template = null`: `<dir>/chat_template.jinja` if it
    /// exists, else `<dir>/tokenizer_config.json`; an explicit path always wins.
    pub fn resolve(model_dir: &Path, explicit: Option<&Path>) -> Result<ChatTemplate, ModelError> {
        if let Some(path) = explicit {
            return Self::load(path);
        }
        let jinja = model_dir.join("chat_template.jinja");
        if jinja.is_file() {
            Self::load(&jinja)
        } else {
            Self::load(&model_dir.join("tokenizer_config.json"))
        }
    }

    fn compile(
        path: &Path,
        source: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<ChatTemplate, ModelError> {
        let name = path.display().to_string();
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(python_methods);
        env.add_filter("tojson", tojson_filter);
        env.add_filter("trim", trim_filter);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(name.clone(), source)
            .map_err(|e| ModelError::Template(format!("{name}: {e}")))?;
        let renders_tools = env
            .get_template(&name)
            .map_err(|e| ModelError::Template(format!("{name}: {e}")))?
            .undeclared_variables(false)
            .contains("tools");
        Ok(ChatTemplate {
            env,
            name,
            bos_token,
            eos_token,
            renders_tools,
        })
    }

    /// Renders `messages` (OpenAI-shaped JSON objects, content already a string) and optional
    /// `tools` (OpenAI tool objects; `None` leaves the `tools` variable undefined).
    /// `kwargs` (`chat_template_kwargs`) are merged into the context; `messages`, `tools` and
    /// `add_generation_prompt` cannot be overridden by them.
    pub fn render(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, ModelError> {
        let mut ctx = serde_json::Map::new();
        if let Some(bos) = &self.bos_token {
            ctx.insert("bos_token".into(), bos.clone().into());
        }
        if let Some(eos) = &self.eos_token {
            ctx.insert("eos_token".into(), eos.clone().into());
        }
        for (k, v) in kwargs {
            ctx.insert(k.clone(), v.clone());
        }
        ctx.insert("messages".into(), messages.to_vec().into());
        match tools {
            Some(tools) => {
                ctx.insert("tools".into(), tools.to_vec().into());
            }
            None => {
                ctx.remove("tools");
            }
        }
        ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
        let template = self
            .env
            .get_template(&self.name)
            .map_err(|e| ModelError::Template(e.to_string()))?;
        template
            .render(Value::from_serialize(&ctx))
            .map_err(|e| ModelError::Template(render_error_message(&e)))
    }

    /// True when the template source references the `tools` variable.
    pub fn renders_tools(&self) -> bool {
        self.renders_tools
    }
}

fn read_tokenizer_config(path: &Path) -> Result<TokenizerConfig, ModelError> {
    let text = std::fs::read_to_string(path).map_err(|e| ModelError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ModelError::Template(format!("{}: {e}", path.display())))?;
    let chat_template = match json.get("chat_template") {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(named)) => named
            .iter()
            .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
            .and_then(|t| t.get("template"))
            .and_then(|t| t.as_str())
            .map(str::to_string),
        _ => None,
    };
    Ok(TokenizerConfig {
        chat_template,
        bos_token: special_token(&json, "bos_token"),
        eos_token: special_token(&json, "eos_token"),
    })
}

/// A special token given either as a string or as an `{"content": ..}` object.
fn special_token(json: &serde_json::Value, key: &str) -> Option<String> {
    match json.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

fn render_error_message(e: &Error) -> String {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = source {
        if let Some(raised) = err.downcast_ref::<RaisedException>() {
            return raised.0.clone();
        }
        source = err.source();
    }
    e.to_string()
}

fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message.clone())
        .with_source(RaisedException(message)))
}

fn strftime_now(format: String) -> String {
    strftime(SystemTime::now(), &format)
}

/// Python `str.isspace()`: Unicode White_Space plus the ASCII separators U+001C..U+001F.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

#[derive(Clone, Copy)]
enum Side {
    Both,
    Left,
    Right,
}

fn py_strip(s: &str, chars: Option<&str>, side: Side) -> String {
    let matches = |c: char| match chars {
        Some(set) => set.contains(c),
        None => is_py_space(c),
    };
    match side {
        Side::Both => s.trim_matches(matches),
        Side::Left => s.trim_start_matches(matches),
        Side::Right => s.trim_end_matches(matches),
    }
    .to_string()
}

/// Jinja2 `trim(value, chars=None)` = `str(value).strip(chars)`.
fn trim_filter(value: Value, chars: Option<String>) -> String {
    let s = match value.as_str() {
        Some(s) => s.to_string(),
        None => value.to_string(),
    };
    py_strip(&s, chars.as_deref(), Side::Both)
}

/// Python methods on values: `strip`/`lstrip`/`rstrip` with Python's whitespace set, the rest
/// from `minijinja-contrib` pycompat (`.items()`, `.get()`, `.startswith()`, ...).
fn python_methods(
    state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if let Some(s) = value.as_str() {
        let side = match method {
            "strip" => Some(Side::Both),
            "lstrip" => Some(Side::Left),
            "rstrip" => Some(Side::Right),
            _ => None,
        };
        if let Some(side) = side {
            let chars = match args {
                [] => None,
                [c] if c.is_none() => None,
                [c] => Some(c.as_str().ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidOperation,
                        "strip argument must be a string",
                    )
                })?),
                _ => return Err(Error::from(ErrorKind::TooManyArguments)),
            };
            return Ok(Value::from(py_strip(s, chars, side)));
        }
    }
    minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
}

/// transformers' `tojson`: `json.dumps(x, ensure_ascii=False, indent=indent,
/// separators=separators, sort_keys=sort_keys)`, byte for byte.
fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    let indent: Option<Value> = kwargs.get("indent")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_string()),
            None => {
                let n = v.as_i64().ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson indent must be int or str",
                    )
                })?;
                Some(" ".repeat(usize::try_from(n.max(0)).unwrap_or(0)))
            }
        },
    };
    let (item_sep, key_sep) = match separators {
        Some(v) if !v.is_none() => {
            let a = v.get_item_by_index(0)?;
            let b = v.get_item_by_index(1)?;
            match (a.as_str(), b.as_str()) {
                (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson separators must be two strings",
                    ));
                }
            }
        }
        _ if indent.is_some() => (",".to_string(), ": ".to_string()),
        _ => (", ".to_string(), ": ".to_string()),
    };
    let opts = JsonOpts {
        indent,
        item_sep,
        key_sep,
        ensure_ascii: ensure_ascii.unwrap_or(false),
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    write_json(&mut out, &value, &opts, 0)?;
    Ok(out)
}

struct JsonOpts {
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    ensure_ascii: bool,
    sort_keys: bool,
}

fn not_serializable(value: &Value) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("Object of type {} is not JSON serializable", value.kind()),
    )
}

fn write_newline(out: &mut String, opts: &JsonOpts, level: usize) {
    if let Some(indent) = &opts.indent {
        out.push('\n');
        for _ in 0..level {
            out.push_str(indent);
        }
    }
}

fn write_json(out: &mut String, value: &Value, opts: &JsonOpts, level: usize) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if value.is_integer() {
                out.push_str(&value.to_string());
            } else {
                let f = f64::try_from(value.clone()).map_err(|_| not_serializable(value))?;
                out.push_str(&py_float_repr(f));
            }
        }
        ValueKind::String => {
            write_json_string(out, value.as_str().unwrap_or_default(), opts.ensure_ascii)
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                write_newline(out, opts, level + 1);
                write_json(out, item, opts, level + 1)?;
            }
            write_newline(out, opts, level);
            out.push(']');
        }
        ValueKind::Map => {
            let mut entries: Vec<(String, Value)> = Vec::new();
            for key in value.try_iter()? {
                let item = value.get_item(&key)?;
                entries.push((json_key(&key)?, item));
            }
            if entries.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            if opts.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                write_newline(out, opts, level + 1);
                write_json_string(out, key, opts.ensure_ascii);
                out.push_str(&opts.key_sep);
                write_json(out, item, opts, level + 1)?;
            }
            write_newline(out, opts, level);
            out.push('}');
        }
        _ => return Err(not_serializable(value)),
    }
    Ok(())
}

/// Python converts non-string dict keys: `True` → "true", `None` → "null", numbers → repr.
fn json_key(key: &Value) -> Result<String, Error> {
    match key.kind() {
        ValueKind::String => Ok(key.as_str().unwrap_or_default().to_string()),
        ValueKind::Bool => Ok(if key.is_true() { "true" } else { "false" }.to_string()),
        ValueKind::None => Ok("null".to_string()),
        ValueKind::Number if key.is_integer() => Ok(key.to_string()),
        ValueKind::Number => {
            let f = f64::try_from(key.clone()).map_err(|_| not_serializable(key))?;
            Ok(py_float_repr(f))
        }
        _ => Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "keys must be str, int, float, bool or None, not {}",
                key.kind()
            ),
        )),
    }
}

fn write_json_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)` (shortest round-trip digits; scientific below 1e-4 and from 1e16).
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if !(-4..16).contains(&exp) {
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs());
    }
    if exp < 0 {
        let zeros = "0".repeat(usize::try_from(-exp - 1).unwrap_or(0));
        return format!("{sign}0.{zeros}{digits}");
    }
    let int_len = usize::try_from(exp + 1).unwrap_or(1);
    if digits.len() <= int_len {
        let pad = "0".repeat(int_len - digits.len());
        format!("{sign}{digits}{pad}.0")
    } else {
        format!("{sign}{}.{}", &digits[..int_len], &digits[int_len..])
    }
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const WEEKDAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// Days since 1970-01-01 → (year, month 1..=12, day 1..=31) (proleptic Gregorian).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let month = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// (year, month, day) → days since 1970-01-01.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(month);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// C/Python `strftime` in UTC for `%d %e %b %B %m %Y %y %H %I %M %S %p %a %A %j %%`; other
/// directives are copied through unchanged.
pub fn strftime(time: SystemTime, format: &str) -> String {
    let secs: i64 = match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        // Before the epoch: floor to the whole second at or before `time`.
        Err(e) => {
            let before = e.duration();
            let whole = i64::try_from(before.as_secs()).unwrap_or(i64::MAX);
            -whole - i64::from(before.subsec_nanos() > 0)
        }
    };
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (sod / 3_600, (sod % 3_600) / 60, sod % 60);
    let weekday = usize::try_from((days + 4).rem_euclid(7)).unwrap_or(0);
    let yday = days - days_from_civil(year, 1, 1) + 1;
    let month_name = MONTHS[usize::try_from(month - 1).unwrap_or(0)];
    let hour12 = if hour % 12 == 0 { 12 } else { hour % 12 };

    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(d) = chars.next() else {
            out.push('%');
            break;
        };
        let _ = match d {
            'd' => write!(out, "{day:02}"),
            'e' => write!(out, "{day:>2}"),
            'b' => write!(out, "{}", &month_name[..3]),
            'B' => write!(out, "{month_name}"),
            'm' => write!(out, "{month:02}"),
            'Y' => write!(out, "{year}"),
            'y' => write!(out, "{:02}", year.rem_euclid(100)),
            'H' => write!(out, "{hour:02}"),
            'I' => write!(out, "{hour12:02}"),
            'M' => write!(out, "{minute:02}"),
            'S' => write!(out, "{second:02}"),
            'p' => write!(out, "{}", if hour < 12 { "AM" } else { "PM" }),
            'a' => write!(out, "{}", &WEEKDAYS[weekday][..3]),
            'A' => write!(out, "{}", WEEKDAYS[weekday]),
            'j' => write!(out, "{yday:03}"),
            '%' => write!(out, "%"),
            other => write!(out, "%{other}"),
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime};

    use serde_json::{Map, Value, json};

    use super::*;
    use crate::tokenizer::Tokenizer;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-3.2-3b-instruct")
    }

    // Expected renders of the Llama-3.2 template with `add_generation_prompt=True,
    // date_string="26 Jul 2024"`, produced by transformers `apply_chat_template(..., tokenize=False)`
    // and `encode(text, add_special_tokens=False)` during planning. The committed
    // `expected_renders.json` (scripts/golden/render_fixture.py) carries the same cases and is
    // cross-checked once that fixture is on this branch.
    const SYSTEM_USER_TEXT: &str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a helpful assistant. Réponds en français si on te le demande.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of Belgium? 🇧🇪<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    const SYSTEM_USER_IDS: &[u32] = &[
        128_000, 128_006, 9_125, 128_007, 271, 38_766, 1_303, 33_025, 2_696, 25, 6_790, 220, 2_366,
        18, 198, 15_724, 2_696, 25, 220, 1_627, 10_263, 220, 2_366, 19, 271, 2_675, 527, 264,
        11_190, 18_328, 13, 51_223, 3_595, 82, 665, 55_467, 4_502, 389, 1_028, 514, 62_163, 13,
        128_009, 128_006, 882, 128_007, 271, 3_923, 374, 279, 6_864, 315, 34_061, 30, 11_410, 229,
        100, 9_468, 229, 103, 128_009, 128_006, 78_191, 128_007, 271,
    ];
    const USER_ONLY_TEXT: &str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello 世界!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    const USER_ONLY_IDS: &[u32] = &[
        128_000, 128_006, 9_125, 128_007, 271, 38_766, 1_303, 33_025, 2_696, 25, 6_790, 220, 2_366,
        18, 198, 15_724, 2_696, 25, 220, 1_627, 10_263, 220, 2_366, 19, 271, 128_009, 128_006, 882,
        128_007, 271, 9_906, 127_365, 0, 128_009, 128_006, 78_191, 128_007, 271,
    ];

    fn date_kwargs() -> Map<String, Value> {
        let mut kwargs = Map::new();
        kwargs.insert("date_string".into(), json!("26 Jul 2024"));
        kwargs
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "turbine-chat-template-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn inline(dir: &Path, source: &str) -> ChatTemplate {
        let path = dir.join("chat_template.jinja");
        std::fs::write(&path, source).expect("write template");
        ChatTemplate::load(&path).expect("template loads")
    }

    #[test]
    fn renders_target_template() {
        let template = ChatTemplate::load(&fixture_dir().join("tokenizer_config.json"))
            .expect("Llama-3.2 template loads");
        let tokenizer =
            Tokenizer::from_file(&fixture_dir().join("tokenizer.json")).expect("tokenizer");
        let system_user = vec![
            json!({"role": "system", "content": "You are a helpful assistant. Réponds en français si on te le demande."}),
            json!({"role": "user", "content": "What is the capital of Belgium? 🇧🇪"}),
        ];
        let user_only = vec![json!({"role": "user", "content": "  Hello 世界!  "})];
        for (case, messages, text, want_ids) in [
            (
                "system_user",
                &system_user,
                SYSTEM_USER_TEXT,
                SYSTEM_USER_IDS,
            ),
            ("user_only", &user_only, USER_ONLY_TEXT, USER_ONLY_IDS),
        ] {
            let rendered = template
                .render(messages, None, true, &date_kwargs())
                .expect("render");
            assert_eq!(rendered, text, "case {case}");
            let ids = tokenizer.encode(&rendered, false).expect("encode");
            assert_eq!(ids, want_ids, "token ids for case {case}");
        }
        assert!(template.renders_tools());

        // Without date_string the template calls strftime_now("%d %b %Y") (UTC).
        let before = strftime(SystemTime::now(), "%d %b %Y");
        let rendered = template
            .render(&user_only, None, true, &Map::new())
            .expect("render");
        let after = strftime(SystemTime::now(), "%d %b %Y");
        assert!(
            rendered.contains(&format!("Today Date: {before}\n"))
                || rendered.contains(&format!("Today Date: {after}\n")),
            "{rendered}"
        );

        // An inline template's raise_exception surfaces as ModelError::Template with its message.
        let dir = temp_dir("bad-role");
        let inline_template = inline(&dir, "{{ raise_exception('bad role') }}");
        match inline_template.render(&user_only, None, true, &Map::new()) {
            Err(ModelError::Template(msg)) => assert!(msg.contains("bad role"), "{msg}"),
            other => panic!("expected ModelError::Template, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();

        // The Llama template's own raise_exception: more than one tool call in a message.
        let call = json!({"type": "function", "function": {"name": "a", "arguments": {}}});
        let rejected = vec![
            json!({"role": "user", "content": "x"}),
            json!({"role": "assistant", "tool_calls": [call.clone(), call]}),
        ];
        match template.render(&rejected, None, true, &date_kwargs()) {
            Err(ModelError::Template(msg)) => {
                assert_eq!(msg, "This model only supports single tool-calls at once!")
            }
            other => panic!("expected ModelError::Template, got {other:?}"),
        }
    }

    #[test]
    fn strftime_formats() {
        // 2024-07-26T13:05:09Z and 2000-02-29T00:07:00Z; expected strings from Python's strftime.
        let t1 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_721_999_109);
        let t2 = SystemTime::UNIX_EPOCH + Duration::from_secs(951_782_820);
        let fmt = "%d %b %B %Y %m %H %M %S %y %a %A %j %p %I %e %%";
        assert_eq!(
            strftime(t1, fmt),
            "26 Jul July 2024 07 13 05 09 24 Fri Friday 208 PM 01 26 %"
        );
        assert_eq!(
            strftime(t2, fmt),
            "29 Feb February 2000 02 00 07 00 00 Tue Tuesday 060 AM 12 29 %"
        );
        assert_eq!(
            strftime(SystemTime::UNIX_EPOCH, "%Y-%m-%d %e"),
            "1970-01-01  1"
        );
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let dir = temp_dir("tojson");
        let t = inline(
            &dir,
            "{{ v | tojson }}|{{ v | tojson(indent=2) }}|{{ [] | tojson(indent=2) }}",
        );
        let mut kwargs = Map::new();
        kwargs.insert(
            "v".into(),
            json!({"a": [1, 2.5, 1e20, 1.0, -0.0, 1e-7, null, true], "b": {}, "z": "é\n\t\"\\\u{1}\u{7f}<>&'"}),
        );
        let out = t.render(&[], None, false, &kwargs).expect("render");
        // Python: json.dumps(v, ensure_ascii=False) | json.dumps(v, ensure_ascii=False, indent=2)
        let want = concat!(
            r#"{"a": [1, 2.5, 1e+20, 1.0, -0.0, 1e-07, null, true], "b": {}, "z": "é\n\t\"\\\u0001"#,
            "\u{7f}",
            r#"<>&'"}|{"#,
            "\n  \"a\": [\n    1,\n    2.5,\n    1e+20,\n    1.0,\n    -0.0,\n    1e-07,\n    null,\n    true\n  ],\n  \"b\": {},\n  \"z\": \"é\\n\\t\\\"\\\\\\u0001\u{7f}<>&'\"\n}|[]"
        );
        assert_eq!(out, want);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn jinja2_semantics() {
        let dir = temp_dir("semantics");
        let t = inline(
            &dir,
            concat!(
                "{{ s | trim }}|{{ s.strip() }}|{{ 'xxaxx' | trim('x') }}|",
                "{{ 'abc' is iterable }}{{ 3 is iterable }}{{ m is mapping }}{{ 'abc' is mapping }}|",
                "{{ 'k' in m }}{{ 'q' in m }}|",
                "{% for x in l[1:] %}{% if x == 3 %}{% continue %}{% endif %}{% if x == 5 %}{% break %}{% endif %}{{ x }}{% endfor %}|",
                "{% for k, v in m.items() %}{{ k }}={{ v }}{% endfor %}|",
                "{{ bos_token }}{{ eos_token }}|{{ add_generation_prompt }}|{{ messages | length }}\n",
            ),
        );
        let mut kwargs = Map::new();
        kwargs.insert("s".into(), json!("\u{1c}\u{a0} hi \u{3000}\u{1f}"));
        kwargs.insert("m".into(), json!({"k": 1, "j": 2}));
        kwargs.insert("l".into(), json!([1, 2, 3, 4, 5, 6]));
        let messages = vec![json!({"role": "user", "content": "x"})];
        let out = t.render(&messages, None, true, &kwargs).expect("render");
        // Booleans print as Python's (`True`/`False`, as Jinja2 renders them); no sibling
        // tokenizer_config.json, so bos/eos are undefined and render empty (lenient, like Jinja2).
        assert_eq!(
            out,
            "hi|hi|a|TrueFalseTrueFalse|TrueFalse|24|k=1j=2||True|1"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inline_raise_exception_and_errors_name_file() {
        let dir = temp_dir("raise");
        let t = inline(&dir, "{{ raise_exception('bad ' + messages[0].role) }}");
        let messages = vec![json!({"role": "robot", "content": "x"})];
        match t.render(&messages, None, false, &Map::new()) {
            Err(ModelError::Template(msg)) => assert_eq!(msg, "bad robot"),
            other => panic!("expected Template error, got {other:?}"),
        }
        let bad = dir.join("broken.jinja");
        std::fs::write(&bad, "{% if %}").expect("write");
        match ChatTemplate::load(&bad) {
            Err(ModelError::Template(msg)) => {
                assert!(msg.contains("broken.jinja"), "{msg}")
            }
            other => panic!("expected Template error, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_prefers_jinja_and_reads_special_tokens() {
        let dir = temp_dir("resolve");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": {"content": "<s>", "lstrip": false}, "eos_token": "</s>", "chat_template": "config:{{ bos_token }}"}"#,
        )
        .expect("write config");
        let from_config = ChatTemplate::resolve(&dir, None).expect("resolve config");
        assert_eq!(
            from_config
                .render(&[], None, false, &Map::new())
                .expect("render"),
            "config:<s>"
        );
        std::fs::write(
            dir.join("chat_template.jinja"),
            "jinja:{{ bos_token }}{{ eos_token }}{% if tools is defined %}{{ tools | length }}{% endif %}",
        )
        .expect("write jinja");
        let from_jinja = ChatTemplate::resolve(&dir, None).expect("resolve jinja");
        assert_eq!(
            from_jinja
                .render(&[], None, false, &Map::new())
                .expect("render"),
            "jinja:<s></s>"
        );
        let tool = json!({"type": "function", "function": {"name": "f"}});
        assert_eq!(
            from_jinja
                .render(&[], Some(&[tool]), false, &Map::new())
                .expect("render"),
            "jinja:<s></s>1"
        );
        assert!(from_jinja.renders_tools());
        assert!(!from_config.renders_tools());
        let explicit = ChatTemplate::resolve(&dir, Some(&dir.join("tokenizer_config.json")))
            .expect("explicit");
        assert_eq!(
            explicit
                .render(&[], None, false, &Map::new())
                .expect("render"),
            "config:<s>"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
