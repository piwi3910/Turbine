//! `--profile multi-turn` (P4 S-16): sessions of `--turns` requests, each turn's prompt the
//! shared prefix + the session's full history + a new user message, sequential within a
//! session and sessions concurrent up to `--concurrency`. With `--session-hints` every turn
//! carries the session id as `prompt_cache_key` and `x-turbine-session-resume-within` (the
//! maximum think time), and the last turn `x-turbine-session-end: true`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::args::{BenchArgs, EndpointArg};
use crate::client::{BenchError, one_request, prepare};
use crate::prompt;
use crate::report::{Report, RequestStats};

/// Prompt index of the shared prefix (per-turn messages use indices below it).
const SHARED_PREFIX_INDEX: u64 = u64::MAX / 2;

/// One finished turn: its measurements and turn index.
struct Turn {
    index: u32,
    stats: RequestStats,
}

/// SplitMix64 step for seeded think times.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The think time before turn `turn` of `session`: uniform in `--think-time`, seeded.
fn think_time(args: &BenchArgs, session: u32, turn: u32) -> Duration {
    let (min, max) = args.think_time.bounds();
    let draw = mix(args.seed ^ (u64::from(session) << 32) ^ u64::from(turn));
    let unit = (draw >> 11) as f64 / (1u64 << 53) as f64;
    Duration::from_secs_f64(min + (max - min) * unit)
}

/// The session id sent as `prompt_cache_key`.
pub fn session_id(seed: u64, session: u32) -> String {
    format!("turbine-bench-{seed}-{session}")
}

/// Runs `--sessions` sessions of `--turns` turns and aggregates the report, including
/// `cached_tokens_ratio` and the TTFT split by turn.
pub async fn run_multi_turn(args: &BenchArgs) -> Result<Report, BenchError> {
    let (client, base, model) = prepare(args).await?;
    let path = match args.endpoint {
        EndpointArg::Chat => "/v1/chat/completions",
        EndpointArg::Completions => "/v1/completions",
    };
    let url = Arc::new(format!("{base}{path}"));
    let model = Arc::new(model);
    let shared = Arc::new(prompt::prompt(
        args.seed,
        SHARED_PREFIX_INDEX,
        args.shared_prefix_words,
    ));
    let next = Arc::new(AtomicU32::new(0));
    let started = Instant::now();
    let mut workers = Vec::new();
    for _ in 0..args.concurrency.min(args.sessions) {
        let (client, url, model, shared, next, args) = (
            client.clone(),
            url.clone(),
            model.clone(),
            shared.clone(),
            next.clone(),
            args.clone(),
        );
        workers.push(tokio::spawn(async move {
            let mut turns = Vec::new();
            let mut failed = 0u64;
            loop {
                let session = next.fetch_add(1, Ordering::Relaxed);
                if session >= args.sessions {
                    break;
                }
                let (t, f) = run_session(&client, &url, &model, &shared, &args, session).await;
                turns.extend(t);
                failed += f;
            }
            (turns, failed)
        }));
    }
    let mut turns = Vec::new();
    let mut failed = 0u64;
    for w in workers {
        match w.await {
            Ok((t, f)) => {
                turns.extend(t);
                failed += f;
            }
            Err(e) => eprintln!("turbine-bench: session worker panicked: {e}"),
        }
    }
    let (first, later): (Vec<&Turn>, Vec<&Turn>) = turns.iter().partition(|t| t.index == 0);
    let ok: Vec<RequestStats> = turns.iter().map(|t| t.stats.clone()).collect();
    let mut report = Report::from_results(&ok, failed, started.elapsed());
    report.add_multi_turn(
        &first.iter().map(|t| &t.stats).collect::<Vec<_>>(),
        &later.iter().map(|t| &t.stats).collect::<Vec<_>>(),
    );
    Ok(report)
}

/// One session's turns, sequential; a failed turn ends the session (its history is gone).
async fn run_session(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    shared: &str,
    args: &BenchArgs,
    session: u32,
) -> (Vec<Turn>, u64) {
    let mut history: Vec<Value> = Vec::new();
    let mut turns = Vec::new();
    for turn in 0..args.turns {
        if turn > 0 {
            tokio::time::sleep(think_time(args, session, turn)).await;
        }
        let index = u64::from(session) * u64::from(args.turns) + u64::from(turn);
        let user = prompt::prompt(args.seed, index, args.prompt_words);
        history.push(json!({"role": "user", "content": user}));
        let body = turn_body(args, model, shared, &history, session);
        let mut headers = Vec::new();
        if args.session_hints {
            headers.push((
                "x-turbine-session-resume-within",
                args.think_time.resume_within_secs().to_string(),
            ));
            if turn + 1 == args.turns {
                headers.push(("x-turbine-session-end", "true".to_string()));
            }
        }
        match one_request(client, url, args.endpoint, &body, &headers)
            .await
            .result
        {
            Ok(stats) => {
                history.push(json!({"role": "assistant", "content": stats.text}));
                turns.push(Turn { index: turn, stats });
            }
            Err(e) => {
                eprintln!("turbine-bench: session {session} turn {turn} failed: {e}");
                return (turns, u64::from(args.turns - turn));
            }
        }
    }
    (turns, 0)
}

/// The request body of one turn: chat messages (the shared prefix as the system message), or
/// for completions the same conversation as one text.
fn turn_body(
    args: &BenchArgs,
    model: &str,
    shared: &str,
    history: &[Value],
    session: u32,
) -> Value {
    let mut body = json!({
        "model": model,
        "max_tokens": args.max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    match args.endpoint {
        EndpointArg::Chat => {
            let mut messages = vec![json!({"role": "system", "content": shared})];
            messages.extend(history.iter().cloned());
            body["messages"] = Value::Array(messages);
        }
        EndpointArg::Completions => {
            let mut text = shared.to_string();
            for m in history {
                text.push('\n');
                text.push_str(m["content"].as_str().unwrap_or_default());
            }
            body["prompt"] = json!(text);
        }
    }
    if args.session_hints {
        body["prompt_cache_key"] = json!(session_id(args.seed, session));
    }
    if args.ignore_eos {
        body["ignore_eos"] = json!(true);
    }
    body
}
