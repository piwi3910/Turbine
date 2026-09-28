//! `turbine-golden eval` / `eval-compare` (phase 8 S-4): task-set accuracy against an
//! OpenAI-compatible endpoint, greedy, one request at a time, and the lossy-format gate.
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Per-request timeout (a 3B model answers a GSM8K item in seconds; 10 min is a hung server).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchKind {
    Exact,
    Number,
}

/// One line of a tasks file: `{"id","prompt"|"messages","answer","match","max_tokens"}`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalTask {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<serde_json::Value>>,
    pub answer: String,
    #[serde(rename = "match")]
    pub match_kind: MatchKind,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct TaskResult {
    pub id: String,
    pub correct: bool,
    pub output: String,
}

/// JSON report (`--output json`), committed as `tests/eval/<model-slug>/<engine>.json`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct EvalReport {
    pub model: String,
    pub tasks_file: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub results: Vec<TaskResult>,
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("{path}: {detail}")]
    Io { path: PathBuf, detail: String },
    #[error("{path}:{line}: {detail}")]
    Task {
        path: PathBuf,
        line: usize,
        detail: String,
    },
    #[error("task {id}: {detail}")]
    Request { id: String, detail: String },
    #[error("{0}")]
    Server(String),
}

/// Parses a tasks file; every line must be a task with exactly one of `prompt`/`messages`
/// and unique ids.
pub fn load_tasks(path: &Path) -> Result<Vec<EvalTask>, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|e| EvalError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    let mut tasks: Vec<EvalTask> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let bad = |detail: String| EvalError::Task {
            path: path.to_path_buf(),
            line: i + 1,
            detail,
        };
        let task: EvalTask = serde_json::from_str(line).map_err(|e| bad(e.to_string()))?;
        if task.prompt.is_some() == task.messages.is_some() {
            return Err(bad("exactly one of prompt or messages is required".into()));
        }
        if task.match_kind == MatchKind::Number && normalize_number(&task.answer).is_none() {
            return Err(bad(format!("answer {:?} is not a number", task.answer)));
        }
        if tasks.iter().any(|t| t.id == task.id) {
            return Err(bad(format!("duplicate id {}", task.id)));
        }
        tasks.push(task);
    }
    Ok(tasks)
}

/// Numeric normal form: trimmed, one trailing `.` removed, `,` separators removed, then an
/// optional `-` followed by digits with at most one `.`; anything else (units, `$`) is `None`.
pub fn normalize_number(s: &str) -> Option<String> {
    let t = s.trim();
    let t = t.strip_suffix('.').unwrap_or(t);
    let t: String = t.chars().filter(|c| *c != ',').collect();
    let digits = t.strip_prefix('-').unwrap_or(&t);
    let mut parts = digits.split('.');
    let int = parts.next()?;
    let frac = parts.next();
    if parts.next().is_some()
        || int.is_empty()
        || !int.chars().all(|c| c.is_ascii_digit())
        || frac.is_some_and(|f| f.is_empty() || !f.chars().all(|c| c.is_ascii_digit()))
    {
        return None;
    }
    Some(t)
}

pub fn is_correct(kind: MatchKind, expected: &str, output: &str) -> bool {
    match kind {
        MatchKind::Exact => output.trim() == expected.trim(),
        MatchKind::Number => match (normalize_number(expected), normalize_number(output)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
    }
}

/// A 2xx response's body parsed as JSON; any other status or an unparsable body is an error.
async fn json_body(
    sent: Result<reqwest::Response, reqwest::Error>,
) -> Result<serde_json::Value, String> {
    let resp = sent
        .and_then(|r| r.error_for_status())
        .map_err(|e| e.to_string())?;
    let text = resp.text().await.map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| format!("response is not JSON: {e}"))
}

async fn served_model(client: &reqwest::Client, base: &str) -> Result<String, EvalError> {
    let v = json_body(client.get(format!("{base}/v1/models")).send().await)
        .await
        .map_err(|e| EvalError::Server(format!("GET /v1/models: {e}")))?;
    v["data"][0]["id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| EvalError::Server("GET /v1/models: no model listed".into()))
}

async fn complete(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    task: &EvalTask,
) -> Result<String, EvalError> {
    let fail = |detail: String| EvalError::Request {
        id: task.id.clone(),
        detail,
    };
    let (path, body) = match (&task.prompt, &task.messages) {
        (Some(prompt), _) => (
            "/v1/completions",
            serde_json::json!({"model": model, "prompt": prompt, "max_tokens": task.max_tokens,
                               "temperature": 0.0, "stream": false}),
        ),
        (None, messages) => (
            "/v1/chat/completions",
            serde_json::json!({"model": model, "messages": messages, "max_tokens": task.max_tokens,
                               "temperature": 0.0, "stream": false}),
        ),
    };
    let sent = client
        .post(format!("{base}{path}"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await;
    let v = json_body(sent).await.map_err(fail)?;
    let choice = &v["choices"][0];
    choice["text"]
        .as_str()
        .or_else(|| choice["message"]["content"].as_str())
        .map(str::to_string)
        .ok_or_else(|| fail(format!("no completion text in {v}")))
}

/// Runs every task in order; the first failed request aborts the run (no partial report).
pub async fn run_eval(
    base: &str,
    model: Option<&str>,
    tasks_file: &Path,
    tasks: &[EvalTask],
) -> Result<EvalReport, EvalError> {
    let base = base.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| EvalError::Server(e.to_string()))?;
    let model = match model {
        Some(m) => m.to_string(),
        None => served_model(&client, base).await?,
    };
    let mut results = Vec::with_capacity(tasks.len());
    for task in tasks {
        let output = complete(&client, base, &model, task).await?;
        let correct = is_correct(task.match_kind, &task.answer, &output);
        results.push(TaskResult {
            id: task.id.clone(),
            correct,
            output,
        });
    }
    let correct = results.iter().filter(|r| r.correct).count();
    let total = results.len();
    Ok(EvalReport {
        model,
        tasks_file: tasks_file.display().to_string(),
        total,
        correct,
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        },
        results,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompareOutcome {
    pub baseline_accuracy: f64,
    pub candidate_accuracy: f64,
    pub max_drop: f64,
    pub pass: bool,
}

/// Gate: candidate accuracy ≥ baseline accuracy − max drop (1e-9 absorbs float rounding).
pub fn compare(baseline: &EvalReport, candidate: &EvalReport, max_drop: f64) -> CompareOutcome {
    CompareOutcome {
        baseline_accuracy: baseline.accuracy,
        candidate_accuracy: candidate.accuracy,
        max_drop,
        pass: candidate.accuracy + 1e-9 >= baseline.accuracy - max_drop,
    }
}

pub fn read_report(path: &Path) -> Result<EvalReport, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|e| EvalError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|e| EvalError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_match_rules() {
        assert!(is_correct(MatchKind::Number, "1234", "1,234"));
        assert!(is_correct(MatchKind::Number, "1234", " 1234.\n"));
        assert!(is_correct(MatchKind::Number, "-5", "-5"));
        assert!(is_correct(MatchKind::Number, "2.5", "2.5."));
        assert!(!is_correct(MatchKind::Number, "18", "$18"));
        assert!(!is_correct(MatchKind::Number, "18", "18 apples"));
        assert!(!is_correct(MatchKind::Number, "18", "18.."));
        assert!(!is_correct(MatchKind::Number, "18", "180"));
        assert!(is_correct(MatchKind::Exact, "yes", " yes "));
        assert!(!is_correct(MatchKind::Exact, "yes", "Yes"));
    }
}
