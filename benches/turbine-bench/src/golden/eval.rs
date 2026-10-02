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
    /// The response's `usage.prompt_tokens`; absent when the server sent no usage (and in a
    /// report from before Phase 6b).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// `usage.prompt_tokens_details.cached_tokens` (0 when the details are absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// `usage.prompt_tokens_details.lossy_cached_tokens` (0 when absent): the prompt tokens
    /// served from lossy KV blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lossy_cached_tokens: Option<u64>,
}

/// Prompt-token counts of one response's `usage`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub lossy_cached_tokens: u64,
}

impl Usage {
    /// `None` when the body has no `usage.prompt_tokens`.
    fn from_response(v: &serde_json::Value) -> Option<Usage> {
        let usage = v.get("usage")?;
        Some(Usage {
            prompt_tokens: usage["prompt_tokens"].as_u64()?,
            cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
            lossy_cached_tokens: usage["prompt_tokens_details"]["lossy_cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        })
    }
}

/// Sum of the scored items' prompt-token counts (fillers excluded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenTotals {
    pub prompt: u64,
    pub cached: u64,
    pub lossy_cached: u64,
}

impl TokenTotals {
    pub fn cached_ratio(&self) -> f64 {
        ratio(self.cached, self.prompt)
    }
    pub fn lossy_cached_ratio(&self) -> f64 {
        ratio(self.lossy_cached, self.prompt)
    }
}

fn ratio(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Unrelated requests sent between the first two items and the rest (`--filler-requests`):
/// they fill L0 so the first items' shared prefix is demoted to the lower KV tier before the other
/// items arrive and a lossy tier format serves it (Phase 6b lossy-KV gates). Up to
/// `concurrency` of them are in flight at once (`--filler-concurrency`; 0 counts as 1, one after
/// another), so together they press on L0 as real load does and the pressure controller leaves
/// GREEN (the ladder gate, user decision "6b Task 16: ladder proof results — four open points",
/// 1 A).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fillers {
    pub requests: u32,
    pub words: u32,
    pub concurrency: u32,
}

/// Seed of the filler prompts (`crate::prompt::prompt(FILLER_SEED, i, words)`), apart from the
/// shared-prefix variant's own seed so no filler shares a block with it.
pub const FILLER_SEED: u64 = 6_000_001;

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
    /// `--filler-requests` / `--filler-words` of this run (0 = none). `eval-compare` refuses a
    /// pair measured with different fillers.
    #[serde(default)]
    pub filler_requests: u32,
    #[serde(default)]
    pub filler_words: u32,
    /// `--filler-concurrency` of this run; an older report with no field reads as 1 (the
    /// fillers one after another). `eval-compare` refuses a pair that differs.
    #[serde(default = "default_concurrency")]
    pub filler_concurrency: u32,
    pub results: Vec<TaskResult>,
}

impl EvalReport {
    /// Prompt-token totals over the results; `None` unless every result carries its usage.
    pub fn token_totals(&self) -> Option<TokenTotals> {
        let mut t = TokenTotals {
            prompt: 0,
            cached: 0,
            lossy_cached: 0,
        };
        for r in &self.results {
            t.prompt += r.prompt_tokens?;
            t.cached += r.cached_tokens?;
            t.lossy_cached += r.lossy_cached_tokens?;
        }
        Some(t)
    }
}

/// The reuse guards of a lossy-KV gate run: `Err` names the miss. A guard needs the server's
/// usage on every item, so a missing usage fails it too (a gate that cannot show it reused
/// lossy blocks must not pass).
pub fn check_reuse(
    report: &EvalReport,
    min_cached_ratio: Option<f64>,
    min_lossy_cached_ratio: Option<f64>,
) -> Result<(), String> {
    if min_cached_ratio.is_none() && min_lossy_cached_ratio.is_none() {
        return Ok(());
    }
    let Some(t) = report.token_totals() else {
        return Err("the server reported no usage.prompt_tokens on every item".into());
    };
    if let Some(min) = min_cached_ratio
        && t.cached_ratio() < min
    {
        return Err(format!(
            "cached prompt tokens {} of {} ({:.3}) are below --min-cached-ratio {min}",
            t.cached,
            t.prompt,
            t.cached_ratio()
        ));
    }
    if let Some(min) = min_lossy_cached_ratio
        && t.lossy_cached_ratio() < min
    {
        return Err(format!(
            "lossy-cached prompt tokens {} of {} ({:.3}) are below --min-lossy-cached-ratio {min}: no lossy KV block was reused (is the shared prefix demoted to the lossy tier? raise --filler-requests or shrink kv.gpu.max_bytes)",
            t.lossy_cached,
            t.prompt,
            t.lossy_cached_ratio()
        ));
    }
    Ok(())
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
) -> Result<(String, Option<Usage>), EvalError> {
    let (path, body) = request_body(model, task);
    post_completion(client, base, path, body, &task.id).await
}

async fn post_completion(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    body: serde_json::Value,
    id: &str,
) -> Result<(String, Option<Usage>), EvalError> {
    let fail = |detail: String| EvalError::Request {
        id: id.to_string(),
        detail,
    };
    let sent = client
        .post(format!("{base}{path}"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await;
    let v = json_body(sent).await.map_err(fail)?;
    let choice = &v["choices"][0];
    let text = choice["text"]
        .as_str()
        .or_else(|| choice["message"]["content"].as_str())
        .map(str::to_string)
        .ok_or_else(|| fail(format!("no completion text in {v}")))?;
    Ok((text, Usage::from_response(&v)))
}

/// One filler request (`--filler-requests`): an unrelated random-word chat prompt, one token.
async fn send_filler(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    index: u32,
    words: u32,
) -> Result<(), EvalError> {
    let prompt = crate::prompt::prompt(FILLER_SEED, u64::from(index), words);
    let body = serde_json::json!({"model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0.0, "stream": false});
    post_completion(
        client,
        base,
        "/v1/chat/completions",
        body,
        &format!("filler-{index}"),
    )
    .await
    .map(|_| ())
}

async fn run_task(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    task: &EvalTask,
) -> Result<TaskResult, EvalError> {
    let (output, usage) = complete(client, base, model, task).await?;
    Ok(TaskResult {
        id: task.id.clone(),
        correct: is_correct(task.match_kind, &task.answer, &output),
        output,
        prompt_tokens: usage.map(|u| u.prompt_tokens),
        cached_tokens: usage.map(|u| u.cached_tokens),
        lossy_cached_tokens: usage.map(|u| u.lossy_cached_tokens),
    })
}

/// Runs every task, up to `concurrency` requests in flight at once (`0` counts as 1, capped at
/// 256 by the caller); results are reported in task-file order whatever order the replies
/// arrive in. The first failed request aborts the run (no partial report). With
/// `fillers.requests` > 0 the first [`FILLER_HEAD`] tasks run alone, one after the other, then
/// the fillers (`fillers.concurrency` in flight at once), then the rest at `concurrency`: the
/// first task publishes the
/// shared prefix, the second hits it (the reuse evidence a server needs before it copies a
/// block down rather than dropping it), the fillers push it out of L0, and the rest read it
/// back from the lower tier.
/// Tasks run alone before the fillers (see [`run_eval`]).
pub const FILLER_HEAD: usize = 2;

pub async fn run_eval(
    base: &str,
    model: Option<&str>,
    tasks_file: &Path,
    tasks: &[EvalTask],
    concurrency: u32,
    fillers: Fillers,
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
    let (head, rest) = if fillers.requests > 0 {
        tasks.split_at(tasks.len().min(FILLER_HEAD))
    } else {
        tasks.split_at(0)
    };
    let mut results: Vec<TaskResult> = Vec::with_capacity(tasks.len());
    for task in head {
        results.push(run_task(client, base, model, task).await?);
    }
    let filler_concurrency = fillers.concurrency.max(1);
    if !head.is_empty() {
        let mut sent = stream::iter(0..fillers.requests)
            .map(|i| send_filler(client, base, model, i, fillers.words))
            .buffer_unordered(filler_concurrency as usize);
        while let Some(reply) = sent.next().await {
            reply?;
        }
    }
    let mut replies = stream::iter(rest.iter().enumerate())
        .map(|(index, task)| async move {
            Ok::<_, EvalError>((index, run_task(client, base, model, task).await?))
        })
        .buffer_unordered(concurrency);
    let mut tail = Vec::with_capacity(rest.len());
    while let Some(reply) = replies.next().await {
        tail.push(reply?);
    }
    tail.sort_by_key(|(index, _)| *index);
    results.extend(tail.into_iter().map(|(_, r)| r));
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
        filler_requests: fillers.requests,
        filler_words: fillers.words,
        filler_concurrency,
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

    fn result(id: &str, usage: Option<(u64, u64, u64)>) -> TaskResult {
        TaskResult {
            id: id.into(),
            correct: true,
            output: String::new(),
            prompt_tokens: usage.map(|u| u.0),
            cached_tokens: usage.map(|u| u.1),
            lossy_cached_tokens: usage.map(|u| u.2),
        }
    }

    fn report(results: Vec<TaskResult>) -> EvalReport {
        EvalReport {
            model: "m".into(),
            tasks_file: "t".into(),
            total: results.len(),
            correct: results.len(),
            accuracy: 1.0,
            concurrency: 1,
            filler_requests: 0,
            filler_words: 0,
            filler_concurrency: 1,
            results,
        }
    }

    #[test]
    fn usage_is_read_from_the_response_and_defaults_missing_details_to_zero() {
        let v = serde_json::json!({"usage": {"prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 64, "lossy_cached_tokens": 32}}});
        assert_eq!(
            Usage::from_response(&v),
            Some(Usage {
                prompt_tokens: 100,
                cached_tokens: 64,
                lossy_cached_tokens: 32
            })
        );
        let v = serde_json::json!({"usage": {"prompt_tokens": 7}});
        assert_eq!(
            Usage::from_response(&v),
            Some(Usage {
                prompt_tokens: 7,
                ..Usage::default()
            })
        );
        assert_eq!(Usage::from_response(&serde_json::json!({})), None);
    }

    #[test]
    fn reuse_guards_judge_the_token_shares() {
        let r = report(vec![
            result("a", Some((1000, 0, 0))),
            result("b", Some((1000, 900, 600))),
        ]);
        let t = r.token_totals().unwrap();
        assert_eq!((t.prompt, t.cached, t.lossy_cached), (2000, 900, 600));
        assert_eq!(check_reuse(&r, None, None), Ok(()));
        assert_eq!(check_reuse(&r, Some(0.45), Some(0.3)), Ok(()));
        let e = check_reuse(&r, Some(0.5), None).unwrap_err();
        assert!(e.contains("--min-cached-ratio"), "{e}");
        let e = check_reuse(&r, None, Some(0.31)).unwrap_err();
        assert!(e.contains("--min-lossy-cached-ratio"), "{e}");
        // No usage on one item: a guard cannot be shown to hold, no guard asks nothing.
        let r = report(vec![result("a", Some((10, 0, 0))), result("b", None)]);
        assert!(r.token_totals().is_none());
        assert!(check_reuse(&r, None, Some(0.0)).is_err());
        assert_eq!(check_reuse(&r, None, None), Ok(()));
    }

    #[test]
    fn an_older_report_without_usage_or_fillers_still_reads() {
        let json = r#"{"model":"m","tasks_file":"t","total":1,"correct":1,"accuracy":1.0,
            "results":[{"id":"x","correct":true,"output":"1"}]}"#;
        let r: EvalReport = serde_json::from_str(json).unwrap();
        assert_eq!(
            (
                r.concurrency,
                r.filler_requests,
                r.filler_words,
                r.filler_concurrency
            ),
            (1, 0, 0, 1)
        );
        assert!(r.token_totals().is_none());
    }
}
