//! Streaming requests against an OpenAI-compatible endpoint and the fixed-concurrency driver.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use serde_json::{Value, json};

use crate::args::{BenchArgs, EndpointArg};
use crate::prompt;
use crate::report::{Report, RequestStats};

/// Setup failures before any request is sent.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    /// Usage problems: exit 2.
    #[error("{0}")]
    Usage(String),
    /// The target could not be queried before the run: exit 1.
    #[error("{0}")]
    Target(String),
}

impl BenchError {
    /// Process exit code: 2 for usage errors, 1 when the target cannot be queried.
    pub fn exit_code(&self) -> u8 {
        match self {
            BenchError::Usage(_) => 2,
            BenchError::Target(_) => 1,
        }
    }
}

/// Run the benchmark described by `args` and aggregate the report.
pub async fn run(args: &BenchArgs) -> Result<Report, BenchError> {
    let base = args.url.trim_end_matches('/').to_string();
    if !base.starts_with("http://") {
        return Err(BenchError::Usage(format!(
            "--url must be an http:// URL, got {}",
            args.url
        )));
    }
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| BenchError::Target(format!("cannot build HTTP client: {e}")))?;
    let model = match &args.model {
        Some(m) => m.clone(),
        None => first_model(&client, &base).await?,
    };
    let path = match args.endpoint {
        EndpointArg::Chat => "/v1/chat/completions",
        EndpointArg::Completions => "/v1/completions",
    };
    let url = Arc::new(format!("{base}{path}"));
    let model = Arc::new(model);
    let next = Arc::new(AtomicU32::new(0));

    let started = Instant::now();
    let mut workers = Vec::new();
    for _ in 0..args.concurrency.min(args.requests) {
        let (client, url, model, next, args) = (
            client.clone(),
            url.clone(),
            model.clone(),
            next.clone(),
            args.clone(),
        );
        workers.push(tokio::spawn(async move {
            let mut ok = Vec::new();
            let mut failed = 0u64;
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= args.requests {
                    break;
                }
                let body = request_body(&args, &model, index);
                match one_request(&client, &url, args.endpoint, &body).await {
                    Ok(stats) => ok.push(stats),
                    Err(e) => {
                        eprintln!("turbine-bench: request {index} failed: {e}");
                        failed += 1;
                    }
                }
            }
            (ok, failed)
        }));
    }
    let mut ok = Vec::new();
    let mut failed = 0u64;
    for w in workers {
        match w.await {
            Ok((o, f)) => {
                ok.extend(o);
                failed += f;
            }
            Err(e) => eprintln!("turbine-bench: worker panicked: {e}"),
        }
    }
    Ok(Report::from_results(&ok, failed, started.elapsed()))
}

async fn first_model(client: &reqwest::Client, base: &str) -> Result<String, BenchError> {
    let url = format!("{base}/v1/models");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| BenchError::Target(format!("GET {url}: {e}")))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| BenchError::Target(format!("GET {url}: {e}")))?;
    if !status.is_success() {
        return Err(BenchError::Target(format!("GET {url}: HTTP {status}")));
    }
    let v: Value =
        serde_json::from_str(&text).map_err(|e| BenchError::Target(format!("GET {url}: {e}")))?;
    v["data"][0]["id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| BenchError::Usage(format!("GET {url} returned no models; pass --model")))
}

fn request_body(args: &BenchArgs, model: &str, index: u32) -> Value {
    let text = prompt::prompt(args.seed, u64::from(index), args.prompt_words);
    let mut body = json!({
        "model": model,
        "max_tokens": args.max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    match args.endpoint {
        EndpointArg::Chat => body["messages"] = json!([{"role": "user", "content": text}]),
        EndpointArg::Completions => body["prompt"] = json!(text),
    }
    if args.ignore_eos {
        body["ignore_eos"] = json!(true);
    }
    body
}

/// Content text of one stream chunk, if any (role-only / usage-only chunks have none).
fn chunk_content(chunk: &Value, endpoint: EndpointArg) -> Option<&str> {
    let choice = chunk.get("choices")?.get(0)?;
    let text = match endpoint {
        EndpointArg::Chat => choice.get("delta")?.get("content")?.as_str()?,
        EndpointArg::Completions => choice.get("text")?.as_str()?,
    };
    (!text.is_empty()).then_some(text)
}

async fn one_request(
    client: &reqwest::Client,
    url: &str,
    endpoint: EndpointArg,
    body: &Value,
) -> Result<RequestStats, String> {
    let sent = Instant::now();
    let mut resp = client
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .map_err(|e| format!("POST {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!(
            "HTTP {status}: {}",
            text.chars().take(200).collect::<String>()
        ));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut token_times: Vec<Instant> = Vec::new();
    let mut usage_tokens: Option<u64> = None;
    loop {
        let chunk = resp
            .chunk()
            .await
            .map_err(|e| format!("reading stream: {e}"))?;
        let Some(bytes) = chunk else {
            return Err("stream ended without [DONE]".to_string());
        };
        buf.extend_from_slice(&bytes);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\r', '\n']);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim_start();
            if data == "[DONE]" {
                let done = Instant::now();
                let Some(first) = token_times.first() else {
                    return Err("stream finished without content".to_string());
                };
                let itls = token_times.windows(2).map(|w| w[1] - w[0]).collect();
                let output_tokens = usage_tokens.unwrap_or(token_times.len() as u64);
                return Ok(RequestStats {
                    ttft: *first - sent,
                    itls,
                    e2e: done - sent,
                    output_tokens,
                });
            }
            let value: Value =
                serde_json::from_str(data).map_err(|e| format!("bad SSE JSON {data:?}: {e}"))?;
            if let Some(err) = value.get("error") {
                return Err(format!("stream error: {err}"));
            }
            if chunk_content(&value, endpoint).is_some() {
                token_times.push(Instant::now());
            }
            if let Some(n) = value
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(Value::as_u64)
            {
                usage_tokens = Some(n);
            }
        }
    }
}
