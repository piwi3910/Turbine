//! `turbine-golden eval` / `eval-compare` (phase 8 S-4): task-set accuracy against an
//! OpenAI-compatible endpoint, greedy, up to `--concurrency` requests in flight at once (default
//! 1), and the lossy-format gate.
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

/// Per-request timeout (a 3B model answers a GSM8K item in seconds; 10 min is a hung server).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    Exact,
    /// The whole output is the number.
    Number,
    /// The number after the output's last `Answer:` (else its last number), `$`, `,` and trailing
    /// units ignored: a chain-of-thought answer (GSM8K-200, Phase 6a).
    FinalNumber,
}

/// One line of a tasks file: `{"id","prompt"|"messages","answer","match","max_tokens","stop"?}`.
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
    /// Stop sequences sent as `stop` on the request. A completion (`prompt`) task on a base
    /// checkpoint with no chat template needs these to keep it from rambling past its answer
    /// into a new few-shot-looking question (GSM8K-200-completion, Phase 6a).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct TaskResult {
    pub id: String,
    pub correct: bool,
    pub output: String,
}

fn default_concurrency() -> u32 {
    1
}

/// JSON report (`--output json`), committed as `tests/eval/<model-slug>/<engine>.json`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct EvalReport {
    pub model: String,
    pub tasks_file: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// Requests kept in flight at once for this run; an older report with no field reads as 1
    /// (sequential). `eval-compare` refuses a pair measured at different concurrencies.
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
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
        if task.match_kind != MatchKind::Exact && normalize_number(&task.answer).is_none() {
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
        MatchKind::FinalNumber => match (normalize_number(expected), final_number(output)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
    }
}

/// The final numeric answer of a chain-of-thought output: the last number after the last
/// `Answer:` (case-insensitive), or the last number of the whole output when it has none, in
/// [`normalize_number`]'s form (`$` and `,` dropped, a trailing `.` removed, `18.0` kept as
/// written).
pub fn final_number(output: &str) -> Option<String> {
    let lower = output.to_ascii_lowercase();
    let tail = match lower.rfind("answer:") {
        Some(i) => &output[i + "answer:".len()..],
        None => output,
    };
    // Numbers: an optional `-`, digits with `,` separators, at most one `.` followed by digits.
    let chars: Vec<char> = tail.chars().collect();
    let mut last: Option<String> = None;
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let neg = i > 0 && chars[i - 1] == '-';
            let mut j = i;
            let mut text = String::new();
            let mut seen_dot = false;
            while j < chars.len() {
                let c = chars[j];
                if c.is_ascii_digit() {
                    text.push(c);
                } else if c == ',' && j + 1 < chars.len() && chars[j + 1].is_ascii_digit() {
                    // a thousands separator
                } else if c == '.'
                    && !seen_dot
                    && j + 1 < chars.len()
                    && chars[j + 1].is_ascii_digit()
                {
                    seen_dot = true;
                    text.push(c);
                } else {
                    break;
                }
                j += 1;
            }
            if neg {
                text.insert(0, '-');
            }
            last = Some(text);
            i = j;
        } else {
            i += 1;
        }
    }
    last.and_then(|t| normalize_number(&t))
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

/// Builds the request path and JSON body for a task: `/v1/completions` with `prompt` for a
/// completion task, `/v1/chat/completions` with `messages` otherwise. `task.stop`, when set, is
/// carried onto the request as `stop` in either case.
fn request_body(model: &str, task: &EvalTask) -> (&'static str, serde_json::Value) {
    let (path, mut body) = match (&task.prompt, &task.messages) {
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
    if let Some(stop) = &task.stop {
        body["stop"] = serde_json::json!(stop);
    }
    (path, body)
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
    let (path, body) = request_body(model, task);
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

/// Runs every task, up to `concurrency` requests in flight at once (`0` counts as 1, capped at
/// 256 by the caller); results are reported in task-file order whatever order the replies
/// arrive in. The first failed request aborts the run (no partial report).
pub async fn run_eval(
    base: &str,
    model: Option<&str>,
    tasks_file: &Path,
    tasks: &[EvalTask],
    concurrency: u32,
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
    let concurrency = (concurrency.max(1)) as usize;
    let model = &model;
    let client = &client;
    let mut replies = stream::iter(tasks.iter().enumerate())
        .map(|(index, task)| async move {
            let output = complete(client, base, model, task).await?;
            let correct = is_correct(task.match_kind, &task.answer, &output);
            Ok::<_, EvalError>((
                index,
                TaskResult {
                    id: task.id.clone(),
                    correct,
                    output,
                },
            ))
        })
        .buffer_unordered(concurrency);
    let mut results = Vec::with_capacity(tasks.len());
    while let Some(reply) = replies.next().await {
        results.push(reply?);
    }
    results.sort_by_key(|(index, _)| *index);
    let results: Vec<TaskResult> = results.into_iter().map(|(_, r)| r).collect();
    let correct = results.iter().filter(|r| r.correct).count();
    let total = results.len();
    Ok(EvalReport {
        model: model.clone(),
        tasks_file: tasks_file.display().to_string(),
        total,
        correct,
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        },
        concurrency: concurrency as u32,
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
        // Chain of thought: the number after the last `Answer:`, else the last number.
        let cot =
            "Janet has 16 eggs, eats 3 and bakes 4: 16 - 3 - 4 = 9.\n9 * 2 = 18.\nAnswer: $18";
        assert!(is_correct(MatchKind::FinalNumber, "18", cot));
        assert!(is_correct(
            MatchKind::FinalNumber,
            "1234",
            "so\nAnswer: 1,234 dollars."
        ));
        assert!(is_correct(MatchKind::FinalNumber, "-5", "Answer: -5"));
        assert!(is_correct(
            MatchKind::FinalNumber,
            "2.5",
            "the rate is 2.5."
        ));
        assert!(!is_correct(MatchKind::FinalNumber, "18", "Answer: 180"));
        assert!(!is_correct(
            MatchKind::FinalNumber,
            "18",
            "18 eggs, so\nAnswer: 9"
        ));
        assert!(!is_correct(MatchKind::FinalNumber, "18", "no number here"));
        assert!(is_correct(MatchKind::Exact, "yes", " yes "));
        assert!(!is_correct(MatchKind::Exact, "yes", "Yes"));
    }

    fn task(
        prompt: Option<&str>,
        messages: Option<Vec<serde_json::Value>>,
        stop: Option<Vec<&str>>,
    ) -> EvalTask {
        EvalTask {
            id: "t".into(),
            prompt: prompt.map(str::to_string),
            messages,
            answer: "1".into(),
            match_kind: MatchKind::FinalNumber,
            max_tokens: 8,
            stop: stop.map(|v| v.into_iter().map(str::to_string).collect()),
        }
    }

    #[test]
    fn completion_task_posts_prompt_without_stop_by_default() {
        let (path, body) = request_body("m", &task(Some("2+2="), None, None));
        assert_eq!(path, "/v1/completions");
        assert_eq!(body["prompt"], "2+2=");
        assert!(body.get("stop").is_none());
        assert!(body.get("messages").is_none());
    }

    #[test]
    fn completion_task_carries_its_stop_sequences() {
        let (path, body) = request_body(
            "m",
            &task(Some("2+2="), None, Some(vec!["\n\nQ:", "Answer:"])),
        );
        assert_eq!(path, "/v1/completions");
        assert_eq!(body["stop"], serde_json::json!(["\n\nQ:", "Answer:"]));
    }

    #[test]
    fn chat_task_posts_messages_and_can_carry_stop() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let (path, body) = request_body("m", &task(None, Some(messages), Some(vec!["\n"])));
        assert_eq!(path, "/v1/chat/completions");
        assert!(body.get("prompt").is_none());
        assert_eq!(body["stop"], serde_json::json!(["\n"]));
    }

    #[test]
    fn task_with_stop_round_trips_through_json() {
        let json = r#"{"id":"x","prompt":"p","answer":"1","match":"final_number","max_tokens":8,"stop":["\n\nQ:"]}"#;
        let t: EvalTask = serde_json::from_str(json).unwrap();
        assert_eq!(t.stop, Some(vec!["\n\nQ:".to_string()]));
    }
}
