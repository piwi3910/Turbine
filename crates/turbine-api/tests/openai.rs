//! Phase 1 OpenAI surface (S-10): field validation, the `InferenceBackend::submit` seam, and the
//! completion/chat response shapes with SSE streaming.

use std::sync::{Arc, Mutex};

use axum::body::{Body, to_bytes};
use axum::http::{HeaderMap, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;
use turbine_api::backend::{BoxFuture, GenerationStream, InferenceRequest};
use turbine_api::openai::request::{OpenAiRequest, PromptInput};
use turbine_api::{
    ApiError, ApiLimits, ApiState, Diagnostics, ErrorCode, ErrorType, InferenceBackend, ModelCard,
    NotReadyReason, Readiness, ReadyState, router,
};
use turbine_core::request::{Endpoint, FinishReason, GenerationEvent, Usage};
use turbine_core::types::RequestId;
use turbine_observability::MetricsRegistry;

fn parse(v: Value) -> OpenAiRequest {
    OpenAiRequest::from_slice(v.to_string().as_bytes()).expect("request body parses")
}

fn with(base: Value, extra: Value) -> OpenAiRequest {
    let mut v = base;
    let obj = v.as_object_mut().expect("base is an object");
    for (k, val) in extra.as_object().expect("extra is an object") {
        obj.insert(k.clone(), val.clone());
    }
    parse(v)
}

fn chat(extra: Value) -> OpenAiRequest {
    with(
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        extra,
    )
}

fn completion(extra: Value) -> OpenAiRequest {
    with(json!({"model": "m", "prompt": "hi"}), extra)
}

fn rejected(req: &OpenAiRequest, endpoint: Endpoint) -> ApiError {
    match req.validate(endpoint) {
        Ok(()) => panic!("accepted {req:?}"),
        Err(e) => e,
    }
}

fn accepts(req: &OpenAiRequest, endpoint: Endpoint) {
    if let Err(e) = req.validate(endpoint) {
        panic!("rejected ({:?}): {}", e.code, e.message);
    }
}

#[test]
fn request_validation_rules() {
    // Known-but-unsupported fields at a non-default value: 400 unsupported_parameter naming the field.
    let unsupported = [
        (
            json!({"tools": [{"type": "function", "function": {"name": "f"}}]}),
            "tools",
        ),
        (
            json!({"response_format": {"type": "json_object"}}),
            "response_format",
        ),
        (json!({"n": 2}), "n"),
        (json!({"logit_bias": {"5": 1}}), "logit_bias"),
        (json!({"presence_penalty": 0.5}), "presence_penalty"),
        (json!({"echo": true}), "echo"),
        (
            json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}]}),
            "image_url",
        ),
        (
            json!({"messages": [{"role": "tool", "content": "42"}]}),
            "messages.role",
        ),
    ];
    for (extra, field) in unsupported {
        let e = rejected(&chat(extra.clone()), Endpoint::ChatCompletions);
        assert_eq!(e.status.as_u16(), 400, "{extra}: {}", e.message);
        assert_eq!(e.kind, ErrorType::InvalidRequestError, "{extra}");
        assert_eq!(
            e.code,
            ErrorCode::UnsupportedParameter,
            "{extra}: {}",
            e.message
        );
        assert!(
            e.message.contains(field),
            "{extra}: {:?} lacks {field}",
            e.message
        );
    }

    // Supported fields out of range: 400 invalid_request naming the field.
    let invalid = [
        (
            chat(json!({"temperature": 3})),
            Endpoint::ChatCompletions,
            "temperature",
        ),
        (
            chat(json!({"top_p": 0})),
            Endpoint::ChatCompletions,
            "top_p",
        ),
        (
            chat(json!({"top_k": 0})),
            Endpoint::ChatCompletions,
            "top_k",
        ),
        (
            chat(json!({"stop": ["a", "b", "c", "d", "e"]})),
            Endpoint::ChatCompletions,
            "stop",
        ),
        (
            completion(json!({"logprobs": true})),
            Endpoint::Completions,
            "logprobs",
        ),
        (
            completion(json!({"logprobs": 21})),
            Endpoint::Completions,
            "logprobs",
        ),
        (
            chat(json!({"logprobs": 3})),
            Endpoint::ChatCompletions,
            "logprobs",
        ),
        (
            chat(json!({"logprobs": true, "top_logprobs": 21})),
            Endpoint::ChatCompletions,
            "top_logprobs",
        ),
        (
            chat(json!({"messages": []})),
            Endpoint::ChatCompletions,
            "messages",
        ),
        (
            completion(json!({"prompt": null})),
            Endpoint::Completions,
            "prompt",
        ),
    ];
    for (req, endpoint, field) in invalid {
        let e = rejected(&req, endpoint);
        assert_eq!(e.status.as_u16(), 400, "{field}: {}", e.message);
        assert_eq!(e.code, ErrorCode::InvalidRequest, "{field}: {}", e.message);
        assert!(e.message.contains(field), "{:?} lacks {field}", e.message);
    }

    // Default values of unsupported fields, and unknown fields, are accepted.
    accepts(
        &chat(json!({"response_format": {"type": "text"}})),
        Endpoint::ChatCompletions,
    );
    accepts(&chat(json!({"n": 1})), Endpoint::ChatCompletions);
    accepts(&chat(json!({"foo": {"bar": 1}})), Endpoint::ChatCompletions);
    accepts(
        &chat(json!({
            "tools": [], "logit_bias": {}, "presence_penalty": 0, "echo": false,
            "temperature": 2, "top_p": 1, "top_k": -1, "stop": ["a", "b", "c", "d"],
            "logprobs": true, "top_logprobs": 20
        })),
        Endpoint::ChatCompletions,
    );
    accepts(
        &completion(json!({"logprobs": 5, "temperature": 0})),
        Endpoint::Completions,
    );

    // A mistyped known field or malformed JSON is invalid_request.
    let e = OpenAiRequest::from_slice(br#"{"model":"m","temperature":"hot"}"#).unwrap_err();
    assert_eq!(e.code, ErrorCode::InvalidRequest);
    assert_eq!(e.status.as_u16(), 400, "{}", e.message);
    let e = OpenAiRequest::from_slice(b"{not json").unwrap_err();
    assert_eq!(e.code, ErrorCode::InvalidRequest);
}

#[test]
fn request_helpers() {
    let r = chat(json!({
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"}
            ]},
            {"role": "assistant", "content": "ok"}
        ],
        "max_tokens": 5, "max_completion_tokens": 9, "stop": ["x", "y"],
        "logprobs": true, "top_logprobs": 4,
        "stream": true, "stream_options": {"include_usage": true},
        "chat_template_kwargs": {"date_string": "26 Jul 2024"}
    }));
    accepts(&r, Endpoint::ChatCompletions);
    assert_eq!(
        r.messages_json(),
        vec![
            json!({"role": "system", "content": "be brief"}),
            json!({"role": "user", "content": "line one\nline two"}),
            json!({"role": "assistant", "content": "ok"}),
        ]
    );
    assert_eq!(r.max_tokens(), Some(9), "max_completion_tokens wins");
    assert_eq!(r.stop_strings(), vec!["x".to_string(), "y".to_string()]);
    assert!(r.include_usage());
    assert_eq!(r.logprobs_n(Endpoint::ChatCompletions), Some(4));
    assert_eq!(
        r.chat_template_kwargs
            .as_ref()
            .map(|k| k["date_string"].clone()),
        Some(json!("26 Jul 2024"))
    );

    let c = completion(json!({"prompt": [1, 2, 3], "stop": "END", "logprobs": 3, "max_tokens": 7}));
    accepts(&c, Endpoint::Completions);
    assert_eq!(c.prompt, Some(PromptInput::Tokens(vec![1, 2, 3])));
    assert_eq!(c.stop_strings(), vec!["END".to_string()]);
    assert_eq!(c.max_tokens(), Some(7));
    assert_eq!(c.logprobs_n(Endpoint::Completions), Some(3));
    assert!(!c.include_usage());

    let plain = completion(json!({}));
    assert_eq!(plain.max_tokens(), None);
    assert!(plain.stop_strings().is_empty());
    assert_eq!(plain.logprobs_n(Endpoint::Completions), None);
    assert_eq!(
        chat(json!({"logprobs": true})).logprobs_n(Endpoint::ChatCompletions),
        Some(0)
    );
    assert_eq!(
        chat(json!({"logprobs": false})).logprobs_n(Endpoint::ChatCompletions),
        None
    );
}

#[test]
fn error_constructors_follow_the_code_table() {
    let cases = [
        (
            ApiError::model_not_found("gpt-4"),
            404,
            ErrorType::InvalidRequestError,
            ErrorCode::ModelNotFound,
        ),
        (
            ApiError::unsupported_parameter("tools"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::UnsupportedParameter,
        ),
        (
            ApiError::context_length_exceeded("too long"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::ContextLengthExceeded,
        ),
        (
            ApiError::engine_busy(),
            429,
            ErrorType::RateLimitError,
            ErrorCode::EngineBusy,
        ),
        (
            ApiError::template_error("bad role"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::TemplateError,
        ),
        (
            ApiError::invalid_request("bad"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::InvalidRequest,
        ),
    ];
    for (e, status, kind, code) in cases {
        assert_eq!(e.status.as_u16(), status, "{e:?}");
        assert_eq!(e.kind, kind, "{e:?}");
        assert_eq!(e.code, code, "{e:?}");
        assert_eq!(
            e.retry_after,
            (code == ErrorCode::EngineBusy).then_some(1),
            "{e:?}"
        );
    }
    assert!(ApiError::model_not_found("gpt-4").message.contains("gpt-4"));
    assert!(
        ApiError::unsupported_parameter("tools")
            .message
            .contains("tools")
    );

    for (code, status) in [
        (ErrorCode::InternalError, 500),
        (ErrorCode::EngineBusy, 429),
        (ErrorCode::ContextLengthExceeded, 400),
        (ErrorCode::ModelNotLoaded, 503),
    ] {
        let e = ApiError::from_code(code, "boom");
        assert_eq!((e.code, e.status.as_u16()), (code, status));
        assert_eq!(
            e.to_json(),
            json!({"error": {
                "message": "boom",
                "type": serde_json::to_value(e.kind).expect("ErrorType serializes"),
                "code": code.as_str()
            }})
        );
    }

    for (reason, s) in [
        (NotReadyReason::LoadingModel, "loading_model"),
        (NotReadyReason::ModelLoadFailed, "model_load_failed"),
        (NotReadyReason::DeviceError, "device_error"),
    ] {
        assert_eq!(reason.as_str(), s);
    }
}

/// A backend that validates the body and answers with a fixed event script.
struct Scripted(Vec<GenerationEvent>);

impl InferenceBackend for Scripted {
    fn models(&self) -> Vec<ModelCard> {
        Vec::new()
    }
    fn submit(&self, req: InferenceRequest) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
        Box::pin(async move {
            req.body.validate(req.endpoint)?;
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            for ev in self.0.clone() {
                tx.send(ev)
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?;
            }
            Ok(rx)
        })
    }
}

/// A backend that keeps every trait default (the Phase 0 `NoModel` shape).
struct Defaults;
impl InferenceBackend for Defaults {
    fn models(&self) -> Vec<ModelCard> {
        Vec::new()
    }
}

fn inference_request(body: OpenAiRequest, endpoint: Endpoint) -> InferenceRequest {
    InferenceRequest {
        id: RequestId::new_v4(),
        endpoint,
        body,
        http_request_id: "req-1".to_string(),
    }
}

#[tokio::test]
async fn backend_submit_seam() {
    let script = vec![
        GenerationEvent::Started { choice: 0 },
        GenerationEvent::Finished {
            choice: 0,
            reason: FinishReason::Stop,
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 0,
            }),
        },
    ];
    let backend = Scripted(script.clone());
    let mut rx = backend
        .submit(inference_request(
            completion(json!({})),
            Endpoint::Completions,
        ))
        .await
        .expect("scripted submit succeeds");
    let mut got = Vec::new();
    while let Some(ev) = rx.recv().await {
        got.push(ev);
    }
    assert_eq!(got, script);

    let e = backend
        .submit(inference_request(
            chat(json!({"n": 2})),
            Endpoint::ChatCompletions,
        ))
        .await
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::UnsupportedParameter);

    // Trait defaults keep Phase 0 behaviour: no model, placeholder token text, no-op rejection count.
    let e = Defaults
        .submit(inference_request(
            completion(json!({})),
            Endpoint::Completions,
        ))
        .await
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::ModelNotLoaded);
    assert_eq!(Defaults.token_text(17), "token_id:17");
    Defaults.record_rejection(Endpoint::Completions, ErrorCode::InvalidRequest);
}

// Responses and SSE streaming: the router over a scripted backend serving model `m`.

/// A backend serving model `m` that answers every request with a fixed event script (or a fixed
/// submit error) and records API-side rejections.
struct ServedScript {
    script: Vec<GenerationEvent>,
    submit_error: Option<ApiError>,
    rejections: Mutex<Vec<(Endpoint, ErrorCode)>>,
}

impl ServedScript {
    fn new(script: Vec<GenerationEvent>) -> Self {
        ServedScript {
            script,
            submit_error: None,
            rejections: Mutex::default(),
        }
    }
}

impl InferenceBackend for ServedScript {
    fn models(&self) -> Vec<ModelCard> {
        vec![ModelCard {
            id: "m".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "turbine".to_string(),
            max_model_len: 128,
        }]
    }
    fn submit(&self, req: InferenceRequest) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
        Box::pin(async move {
            drop(req);
            if let Some(e) = &self.submit_error {
                return Err(e.clone());
            }
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            for ev in self.script.clone() {
                tx.send(ev)
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?;
            }
            Ok(rx)
        })
    }
    fn token_text(&self, token_id: u32) -> String {
        format!("<t{token_id}>")
    }
    fn record_rejection(&self, endpoint: Endpoint, code: ErrorCode) {
        self.rejections
            .lock()
            .expect("rejections lock")
            .push((endpoint, code));
    }
}

struct AlwaysReady;
impl Readiness for AlwaysReady {
    fn ready(&self) -> ReadyState {
        ReadyState::Ready
    }
}

struct NoDiagnostics;
impl Diagnostics for NoDiagnostics {
    fn status(&self) -> Value {
        json!({})
    }
    fn devices(&self) -> Value {
        json!({})
    }
    fn scheduler(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
    fn kv(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
    fn pressure(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
}

fn served_app(backend: Arc<ServedScript>) -> axum::Router {
    router(ApiState {
        inference: backend,
        diagnostics: Arc::new(NoDiagnostics),
        readiness: Arc::new(AlwaysReady),
        metrics: MetricsRegistry::new(),
        limits: ApiLimits {
            max_request_bytes: 1 << 20,
        },
    })
}

async fn post_raw(
    app: &axum::Router,
    path: &str,
    body: impl Into<Body>,
) -> (StatusCode, HeaderMap, String) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(body.into())
        .expect("request builds");
    let resp = app.clone().oneshot(req).await.expect("router answers");
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    (status, headers, text)
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, HeaderMap, String) {
    post_raw(app, path, body.to_string()).await
}

fn content_type(headers: &HeaderMap) -> &str {
    headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
}

/// The `data:` payloads of an SSE body, in order (`[DONE]` kept as a JSON string).
fn sse_data(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|frame| frame.lines().find_map(|l| l.strip_prefix("data: ")))
        .map(|d| {
            if d == "[DONE]" {
                Value::String(d.to_string())
            } else {
                serde_json::from_str(d).unwrap_or_else(|e| panic!("chunk {d:?}: {e}"))
            }
        })
        .collect()
}

fn token(text: &str, token_id: u32) -> GenerationEvent {
    GenerationEvent::Token {
        choice: 0,
        text: text.to_string(),
        token_id,
        logprob: Some(-0.5),
        top_logprobs: vec![(token_id, -0.5), (99, -1.5)],
    }
}

/// `Started`, tokens `"Hel"`, `""` (held bytes), `"lo"`, `Finished{stop, usage 3/3}`.
fn hello_script() -> Vec<GenerationEvent> {
    vec![
        GenerationEvent::Started { choice: 0 },
        token("Hel", 17),
        token("", 18),
        token("lo", 19),
        GenerationEvent::Finished {
            choice: 0,
            reason: FinishReason::Stop,
            usage: Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 3,
            }),
        },
    ]
}

fn json_body(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{body:?}: {e}"))
}

#[tokio::test]
async fn response_shapes_and_stream_order() {
    let app = served_app(Arc::new(ServedScript::new(hello_script())));

    // Non-streaming completion.
    let (s, headers, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert!(content_type(&headers).starts_with("application/json"));
    let v = json_body(&body);
    assert!(
        v["id"].as_str().is_some_and(|id| id.starts_with("cmpl-")),
        "{v}"
    );
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["model"], "m");
    assert!(
        v["created"].as_u64().is_some_and(|t| t > 1_700_000_000),
        "{v}"
    );
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["text"], "Hello");
    assert_eq!(v["choices"][0]["logprobs"], Value::Null);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        v["usage"],
        json!({"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6})
    );

    // Non-streaming chat.
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_body(&body);
    assert!(
        v["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("chatcmpl-")),
        "{v}"
    );
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(
        v["choices"][0]["message"],
        json!({"role": "assistant", "content": "Hello"})
    );
    assert_eq!(v["choices"][0]["logprobs"], Value::Null);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 6);

    // Streaming chat with include_usage: role, two content chunks, finish, usage, [DONE].
    let (s, headers, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
               "stream": true, "stream_options": {"include_usage": true}}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert!(content_type(&headers).starts_with("text/event-stream"));
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 6, "{body}");
    let id = chunks[0]["id"].clone();
    assert!(
        id.as_str().is_some_and(|id| id.starts_with("chatcmpl-")),
        "{body}"
    );
    for c in &chunks[..5] {
        assert_eq!(c["id"], id, "one id per stream: {c}");
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["model"], "m");
    }
    assert_eq!(
        chunks[0]["choices"][0]["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert_eq!(chunks[1]["choices"][0]["delta"], json!({"content": "Hel"}));
    assert_eq!(chunks[2]["choices"][0]["delta"], json!({"content": "lo"}));
    for c in &chunks[..3] {
        assert_eq!(c["choices"][0]["finish_reason"], Value::Null, "{c}");
    }
    assert_eq!(chunks[3]["choices"][0]["delta"], json!({}));
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[4]["choices"], json!([]));
    assert_eq!(
        chunks[4]["usage"],
        json!({"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6})
    );
    assert_eq!(chunks[5], "[DONE]");

    // Streaming completion without include_usage: two text chunks, finish, [DONE].
    let (s, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 4, "{body}");
    assert_eq!(chunks[0]["object"], "text_completion");
    assert!(
        chunks[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("cmpl-"))
    );
    assert_eq!(chunks[0]["choices"][0]["text"], "Hel");
    assert_eq!(chunks[1]["choices"][0]["text"], "lo");
    assert_eq!(chunks[2]["choices"][0]["text"], "");
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[3], "[DONE]");

    // Completions logprobs with return_tokens_as_token_ids: `token_id:<id>` strings and keys.
    let (s, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi", "logprobs": 2, "return_tokens_as_token_ids": true}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_body(&body);
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(
        lp["tokens"],
        json!(["token_id:17", "token_id:18", "token_id:19"])
    );
    assert_eq!(lp["token_logprobs"], json!([-0.5, -0.5, -0.5]));
    assert_eq!(
        lp["top_logprobs"][0],
        json!({"token_id:17": -0.5, "token_id:99": -1.5})
    );
    assert_eq!(lp["text_offset"], json!([0, 3, 3]));

    // Streamed completion logprobs: the held token's entry rides on the next text chunk.
    let (s, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi", "stream": true, "logprobs": 1,
               "return_tokens_as_token_ids": true}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let chunks = sse_data(&body);
    assert_eq!(
        chunks[0]["choices"][0]["logprobs"]["tokens"],
        json!(["token_id:17"])
    );
    assert_eq!(
        chunks[1]["choices"][0]["logprobs"]["tokens"],
        json!(["token_id:18", "token_id:19"])
    );
    assert_eq!(
        chunks[1]["choices"][0]["logprobs"]["text_offset"],
        json!([3, 3])
    );

    // Chat logprobs render the backend's token text.
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
               "logprobs": true, "top_logprobs": 2}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_body(&body);
    let content = &v["choices"][0]["logprobs"]["content"];
    assert_eq!(content.as_array().map(Vec::len), Some(3), "{v}");
    assert_eq!(
        content[0],
        json!({"token": "<t17>", "logprob": -0.5, "bytes": [60, 116, 49, 55, 62],
        "top_logprobs": [
            {"token": "<t17>", "logprob": -0.5, "bytes": [60, 116, 49, 55, 62]},
            {"token": "<t99>", "logprob": -1.5, "bytes": [60, 116, 57, 57, 62]}
        ]})
    );

    // A mid-stream engine error: the content so far, the error event, then [DONE] (C-3).
    let failing = vec![
        GenerationEvent::Started { choice: 0 },
        token("Hel", 17),
        GenerationEvent::Error {
            code: ErrorCode::InternalError,
            message: "kernel failed".to_string(),
        },
    ];
    let app = served_app(Arc::new(ServedScript::new(failing)));
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 4, "{body}");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "Hel");
    assert_eq!(
        chunks[2],
        json!({"error": {"message": "kernel failed", "type": "server_error", "code": "internal_error"}})
    );
    assert_eq!(chunks[3], "[DONE]");

    // The same error on a non-streaming request is a plain 500.
    let (s, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi"}),
    )
    .await;
    assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert_eq!(json_body(&body)["error"]["code"], "internal_error");

    // A stream that ends without `Finished` is an internal error, still closed by [DONE].
    let truncated = vec![GenerationEvent::Started { choice: 0 }, token("Hel", 17)];
    let app = served_app(Arc::new(ServedScript::new(truncated)));
    let (_, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi", "stream": true}),
    )
    .await;
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 3, "{body}");
    assert_eq!(chunks[1]["error"]["code"], "internal_error");
    assert_eq!(chunks[2], "[DONE]");
}

#[tokio::test]
async fn handler_errors_before_the_stream() {
    let backend = Arc::new(ServedScript::new(hello_script()));
    let app = served_app(backend.clone());

    // Malformed JSON and a validation failure: 400, each recorded as a rejection.
    let (s, _, body) = post_raw(&app, "/v1/completions", "{not json").await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_body(&body)["error"]["code"], "invalid_request");
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 2}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_body(&body)["error"]["code"], "unsupported_parameter");

    // A model that is not served: 404 model_not_found, even for a streaming request.
    let (s, headers, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "gpt-4", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{body}");
    assert!(content_type(&headers).starts_with("application/json"));
    assert_eq!(json_body(&body)["error"]["code"], "model_not_found");
    assert_eq!(
        *backend.rejections.lock().expect("rejections lock"),
        vec![
            (Endpoint::Completions, ErrorCode::InvalidRequest),
            (Endpoint::ChatCompletions, ErrorCode::UnsupportedParameter),
            (Endpoint::Completions, ErrorCode::ModelNotFound),
        ]
    );

    // A submit error on a streaming request is a plain HTTP error, not an SSE stream.
    let busy = Arc::new(ServedScript {
        submit_error: Some(ApiError::engine_busy()),
        ..ServedScript::new(Vec::new())
    });
    let app = served_app(busy);
    let (s, headers, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true}),
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{body}");
    assert_eq!(
        headers.get("retry-after").map(|v| v.as_bytes()),
        Some(&b"1"[..])
    );
    assert!(content_type(&headers).starts_with("application/json"));
    assert_eq!(json_body(&body)["error"]["code"], "engine_busy");
}
