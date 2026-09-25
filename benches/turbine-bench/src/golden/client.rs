//! Replaying golden prompts against an OpenAI-compatible endpoint: greedy, non-streaming,
//! token ids and top logprobs per position.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::stream::{self, StreamExt};
use serde_json::{Value, json};

use super::compare::{CompareReport, compare_prompt};
use super::fixture::{GoldenError, PromptKind, PromptRecord, ReferenceRecord, Tolerance};

/// Top logprobs requested by `compare` (the OpenAI maximum).
pub const COMPARE_TOP_LOGPROBS: u32 = 20;

/// One greedy generation as the endpoint reported it.
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub tokens: Vec<u32>,
    /// Per position, `(token_id, logprob)` highest first, at most the requested count.
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
    /// Empty when the endpoint does not return prompt token ids.
    pub prompt_token_ids: Vec<u32>,
    pub system_fingerprint: Option<String>,
    pub model: Option<String>,
}

/// An `http://` OpenAI-compatible base URL.
pub struct Endpoint {
    client: reqwest::Client,
    base: String,
}

impl Endpoint {
    pub fn new(url: &str) -> Result<Self, GoldenError> {
        let base = url.trim_end_matches('/').to_string();
        if !base.starts_with("http://") {
            return Err(GoldenError::Usage(format!(
                "--url must be an http:// URL, got {url}"
            )));
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| GoldenError::Endpoint(format!("cannot build HTTP client: {e}")))?;
        Ok(Self { client, base })
    }

    /// `explicit`, or the first id from `GET /v1/models`.
    pub async fn model(&self, explicit: Option<&str>) -> Result<String, GoldenError> {
        if let Some(m) = explicit {
            return Ok(m.to_string());
        }
        let url = format!("{}/v1/models", self.base);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| GoldenError::Endpoint(format!("GET {url}: {e}")))?;
        let v = json_body(resp, &url).await?;
        v["data"][0]["id"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                GoldenError::Endpoint(format!("GET {url} returned no models; pass --model"))
            })
    }

    /// Run one prompt greedily for its `max_tokens` positions.
    pub async fn generate(
        &self,
        model: &str,
        prompt: &PromptRecord,
        top_logprobs: u32,
    ) -> Result<Generation, GoldenError> {
        let (path, body) = request_body(model, prompt, top_logprobs);
        let url = format!("{}{path}", self.base);
        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| GoldenError::Endpoint(format!("POST {url} ({}): {e}", prompt.id)))?;
        let v = json_body(resp, &url).await?;
        let parsed = match prompt.kind {
            PromptKind::Completion => parse_completion(&v, top_logprobs as usize),
            PromptKind::Chat => parse_chat(&v, top_logprobs as usize),
        };
        parsed.map_err(|e| GoldenError::Endpoint(format!("POST {url} ({}): {e}", prompt.id)))
    }
}

async fn json_body(resp: reqwest::Response, url: &str) -> Result<Value, GoldenError> {
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| GoldenError::Endpoint(format!("{url}: {e}")))?;
    if !status.is_success() {
        return Err(GoldenError::Endpoint(format!(
            "{url}: HTTP {status}: {}",
            text.chars().take(300).collect::<String>()
        )));
    }
    serde_json::from_str(&text).map_err(|e| GoldenError::Endpoint(format!("{url}: bad JSON: {e}")))
}

/// Route and body for one greedy replay: `temperature: 0`, `top_logprobs` logprobs, token ids
/// as `token_id:<id>`, `ignore_eos` (the reference always runs `max_tokens` positions),
/// non-streaming. `return_token_ids` asks vLLM-style engines for the prompt token ids.
pub fn request_body(
    model: &str,
    prompt: &PromptRecord,
    top_logprobs: u32,
) -> (&'static str, Value) {
    let mut body = json!({
        "model": model,
        "max_tokens": prompt.max_tokens,
        "temperature": 0,
        "return_tokens_as_token_ids": true,
        "return_token_ids": true,
        "ignore_eos": true,
        "stream": false,
    });
    match prompt.kind {
        PromptKind::Completion => {
            body["prompt"] = json!(prompt.prompt);
            body["logprobs"] = json!(top_logprobs);
            ("/v1/completions", body)
        }
        PromptKind::Chat => {
            body["messages"] = json!(prompt.messages);
            body["logprobs"] = json!(true);
            body["top_logprobs"] = json!(top_logprobs);
            if let Some(kwargs) = &prompt.chat_template_kwargs {
                body["chat_template_kwargs"] = Value::Object(kwargs.clone());
            }
            ("/v1/chat/completions", body)
        }
    }
}

/// `token_id:<id>` → `<id>`.
fn token_id(s: &str) -> Result<u32, String> {
    s.strip_prefix("token_id:")
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| {
            format!(
                "token {s:?} is not \"token_id:<id>\" (is return_tokens_as_token_ids supported?)"
            )
        })
}

fn logprob(v: &Value) -> Result<f32, String> {
    v.as_f64()
        .map(|lp| lp as f32)
        .ok_or_else(|| format!("logprob {v} is not a number"))
}

/// Highest first (ties by id), at most `k` entries.
fn sort_row(mut row: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    row.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    row.truncate(k);
    row
}

fn ids(v: Option<&Value>) -> Vec<u32> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

fn envelope(
    v: &Value,
    choice: &Value,
    tokens: Vec<u32>,
    top_logprobs: Vec<Vec<(u32, f32)>>,
) -> Generation {
    Generation {
        tokens,
        top_logprobs,
        prompt_token_ids: if choice.get("prompt_token_ids").is_some() {
            ids(choice.get("prompt_token_ids"))
        } else {
            ids(v.get("prompt_token_ids"))
        },
        system_fingerprint: v["system_fingerprint"].as_str().map(str::to_string),
        model: v["model"].as_str().map(str::to_string),
    }
}

/// `/v1/completions`: `choices[0].logprobs.tokens` and `top_logprobs` maps keyed `token_id:<id>`.
pub fn parse_completion(v: &Value, k: usize) -> Result<Generation, String> {
    let choice = &v["choices"][0];
    let lp = &choice["logprobs"];
    let tokens = lp["tokens"]
        .as_array()
        .ok_or("choices[0].logprobs.tokens missing")?
        .iter()
        .map(|t| token_id(t.as_str().unwrap_or_default()))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = lp["top_logprobs"]
        .as_array()
        .ok_or("choices[0].logprobs.top_logprobs missing")?;
    if rows.len() != tokens.len() {
        return Err(format!(
            "{} tokens but {} top_logprobs rows",
            tokens.len(),
            rows.len()
        ));
    }
    let top = rows
        .iter()
        .map(|row| {
            let map = row
                .as_object()
                .ok_or_else(|| format!("top_logprobs row {row} is not an object"))?;
            let entries = map
                .iter()
                .map(|(t, lp)| Ok((token_id(t)?, logprob(lp)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(sort_row(entries, k))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(envelope(v, choice, tokens, top))
}

/// `/v1/chat/completions`: `choices[0].logprobs.content[]` with `token` and `top_logprobs[]`.
pub fn parse_chat(v: &Value, k: usize) -> Result<Generation, String> {
    let choice = &v["choices"][0];
    let content = choice["logprobs"]["content"]
        .as_array()
        .ok_or("choices[0].logprobs.content missing")?;
    let mut tokens = Vec::with_capacity(content.len());
    let mut top = Vec::with_capacity(content.len());
    for entry in content {
        tokens.push(token_id(entry["token"].as_str().unwrap_or_default())?);
        let row = entry["top_logprobs"]
            .as_array()
            .ok_or_else(|| format!("top_logprobs missing in {entry}"))?
            .iter()
            .map(|e| {
                Ok((
                    token_id(e["token"].as_str().unwrap_or_default())?,
                    logprob(&e["logprob"])?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        top.push(sort_row(row, k));
    }
    Ok(envelope(v, choice, tokens, top))
}

/// Replay every reference prompt and judge it; `prompts` supplies the text by id. At most
/// `concurrency` prompts are in flight at once (`0` counts as 1); verdicts are reported in
/// reference order whatever order the replies arrive in, and the first endpoint error ends the
/// run.
pub async fn compare(
    endpoint: &Endpoint,
    model: &str,
    references: &[ReferenceRecord],
    prompts: &[PromptRecord],
    tol: &Tolerance,
    concurrency: usize,
) -> Result<CompareReport, GoldenError> {
    let by_id: HashMap<&str, &PromptRecord> = prompts.iter().map(|p| (p.id.as_str(), p)).collect();
    // Resolve every prompt before sending anything, so a missing id is a usage error up front.
    let jobs = references
        .iter()
        .map(|reference| {
            by_id
                .get(reference.id.as_str())
                .map(|prompt| (reference, *prompt))
                .ok_or_else(|| {
                    GoldenError::Usage(format!(
                        "reference prompt {} is not in the prompts file",
                        reference.id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut replies = stream::iter(jobs.into_iter().enumerate())
        .map(|(index, (reference, prompt))| async move {
            let got = endpoint
                .generate(model, prompt, COMPARE_TOP_LOGPROBS)
                .await?;
            Ok::<_, GoldenError>((
                index,
                compare_prompt(reference, &got.tokens, &got.top_logprobs, tol),
            ))
        })
        .buffer_unordered(concurrency.max(1));
    let mut verdicts = Vec::with_capacity(references.len());
    while let Some(reply) = replies.next().await {
        verdicts.push(reply?);
    }
    verdicts.sort_by_key(|(index, _)| *index);
    Ok(CompareReport::new(
        verdicts.into_iter().map(|(_, v)| v).collect(),
        tol,
    ))
}

/// Run every prompt and build reference records (engine from `system_fingerprint`).
pub async fn capture(
    endpoint: &Endpoint,
    model: &str,
    prompts: &[PromptRecord],
    top_logprobs: u32,
) -> Result<Vec<ReferenceRecord>, GoldenError> {
    let mut out = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let got = endpoint.generate(model, prompt, top_logprobs).await?;
        out.push(ReferenceRecord {
            id: prompt.id.clone(),
            engine: got
                .system_fingerprint
                .unwrap_or_else(|| "unknown".to_string()),
            model: got.model.unwrap_or_else(|| model.to_string()),
            captured: rfc3339_utc(SystemTime::now()),
            prompt_token_ids: got.prompt_token_ids,
            tokens: got.tokens,
            top_logprobs: got.top_logprobs,
        });
    }
    Ok(out)
}

/// `YYYY-MM-DDTHH:MM:SSZ` (whole seconds, UTC).
pub fn rfc3339_utc(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_dates() {
        let at = |s| rfc3339_utc(UNIX_EPOCH + Duration::from_secs(s));
        assert_eq!(at(0), "1970-01-01T00:00:00Z");
        assert_eq!(at(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(at(1_790_338_261), "2026-09-25T12:11:01Z");
    }

    #[test]
    fn completion_rows_are_sorted_and_truncated() {
        let v = json!({
            "system_fingerprint": "fp",
            "choices": [{"logprobs": {
                "tokens": ["token_id:7", "token_id:8"],
                "top_logprobs": [
                    {"token_id:3": -2.0, "token_id:7": -0.1, "token_id:5": -1.0},
                    {"token_id:8": -0.2}
                ]
            }}]
        });
        let g = parse_completion(&v, 2).unwrap();
        assert_eq!(g.tokens, vec![7, 8]);
        assert_eq!(
            g.top_logprobs,
            vec![vec![(7, -0.1), (5, -1.0)], vec![(8, -0.2)]]
        );
        assert_eq!(g.system_fingerprint.as_deref(), Some("fp"));
        assert!(g.prompt_token_ids.is_empty());
    }

    #[test]
    fn plain_token_strings_are_refused() {
        let v = json!({"choices": [{"logprobs": {"content": [
            {"token": "Hello", "logprob": -0.1, "top_logprobs": []}
        ]}}]});
        let err = parse_chat(&v, 20).unwrap_err();
        assert!(err.contains("return_tokens_as_token_ids"), "{err}");
    }
}
