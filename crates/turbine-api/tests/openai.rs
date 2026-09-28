//! OpenAI surface (P1 S-10; P2 S-10, S-17, S-18): field validation, the `InferenceBackend::submit` seam, and the
//! completion/chat response shapes with SSE streaming.

use std::sync::{Arc, Mutex};

use axum::body::{Body, to_bytes};
use axum::http::{HeaderMap, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;
use turbine_api::backend::{BoxFuture, GenerationStream, InferenceRequest};
use turbine_api::openai::request::{OpenAiRequest, PromptInput, ResponseFormat, ToolChoiceMode};
use turbine_api::{
    ApiError, ApiLimits, ApiState, Diagnostics, ErrorCode, ErrorType, InferenceBackend, ModelCard,
    NotReadyReason, Readiness, ReadyState, router,
};
use turbine_core::request::{Endpoint, FinishReason, GenerationEvent, ToolCallOut, Usage};
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
        (json!({"best_of": 2}), "best_of"),
        (json!({"suffix": "tail"}), "suffix"),
        (json!({"echo": true}), "echo"),
        (
            json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}]}),
            "image_url",
        ),
        (
            json!({"messages": [{"role": "developer", "content": "42"}]}),
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
        hints: turbine_api::TurbineHeaders::default(),
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
                cached_tokens: 0,
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
            chat(json!({"best_of": 2})),
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
                cached_tokens: 0,
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
        json!({"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6,
               "prompt_tokens_details": {"cached_tokens": 0}})
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
        json!({"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6,
               "prompt_tokens_details": {"cached_tokens": 0}})
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
    // The client gets a fixed message naming the request, never the engine's internal detail
    // (which is logged with the same id) (Scout feb0de60).
    let id = chunks[1]["id"].as_str().expect("chunk id");
    assert_eq!(
        chunks[2],
        json!({"error": {
            "message": format!(
                "The server had an error while processing your request (request {id})."
            ),
            "type": "server_error",
            "code": "internal_error",
        }})
    );
    assert!(!body.contains("kernel failed"), "{body}");
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
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "best_of": 2}),
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

// Phase 2 (S-10, S-13, S-17, S-18): choices by index, tool calls, new errors and fields.

fn token_at(choice: u32, text: &str, token_id: u32) -> GenerationEvent {
    GenerationEvent::Token {
        choice,
        text: text.to_string(),
        token_id,
        logprob: None,
        top_logprobs: Vec::new(),
    }
}

fn finished(choice: u32, reason: FinishReason, completion_tokens: u32) -> GenerationEvent {
    GenerationEvent::Finished {
        choice,
        reason,
        usage: Some(Usage {
            prompt_tokens: 3,
            completion_tokens,
            cached_tokens: 0,
        }),
    }
}

/// Two interleaved choices: 0 says "Hi there" and stops, 1 says "Yo" and hits the length limit.
fn two_choice_script() -> Vec<GenerationEvent> {
    vec![
        GenerationEvent::Started { choice: 0 },
        GenerationEvent::Started { choice: 1 },
        token_at(0, "Hi", 5),
        token_at(1, "Yo", 6),
        token_at(0, " there", 7),
        finished(1, FinishReason::Length, 1),
        finished(0, FinishReason::Stop, 2),
    ]
}

const PARIS: &str = "{\"location\":\"Paris\"}";
const CALL_ID: &str = "call_abcdefghijklmnopqrstuvwx";

fn weather_tools() -> Value {
    json!([{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["location"]}
    }}])
}

fn tool_call_script() -> Vec<GenerationEvent> {
    vec![
        GenerationEvent::Started { choice: 0 },
        GenerationEvent::ToolCalls {
            choice: 0,
            calls: vec![ToolCallOut {
                index: 0,
                id: CALL_ID.to_string(),
                name: "get_weather".to_string(),
                arguments: PARIS.to_string(),
            }],
        },
        finished(0, FinishReason::ToolCalls, 9),
    ]
}

#[tokio::test]
async fn phase2_shapes() {
    // n = 2, non-streaming chat: one choice per index, each with its own text and finish reason;
    // usage adds the completion tokens of every choice.
    let app = served_app(Arc::new(ServedScript::new(two_choice_script())));
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 2}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_body(&body);
    let choices = v["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 2, "{v}");
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(
        choices[0]["message"],
        json!({"role": "assistant", "content": "Hi there"})
    );
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert_eq!(choices[1]["index"], 1);
    assert_eq!(
        choices[1]["message"],
        json!({"role": "assistant", "content": "Yo"})
    );
    assert_eq!(choices[1]["finish_reason"], "length");
    assert_eq!(
        v["usage"],
        json!({"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6,
               "prompt_tokens_details": {"cached_tokens": 0}})
    );

    // n = 2, non-streaming completion.
    let (s, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi", "n": 2}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_body(&body);
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["text"], "Hi there");
    assert_eq!(v["choices"][1]["index"], 1);
    assert_eq!(v["choices"][1]["text"], "Yo");

    // n = 2, streaming chat: every chunk names its choice; one usage chunk and [DONE] after the
    // last choice finishes.
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 2,
               "stream": true, "stream_options": {"include_usage": true}}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 9, "{body}");
    let at = |i: usize| (&chunks[i]["choices"][0]["index"], &chunks[i]["choices"][0]);
    assert_eq!(at(0).0, 0);
    assert_eq!(
        at(0).1["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert_eq!(at(1).0, 1);
    assert_eq!(
        at(1).1["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert_eq!(
        (at(2).0, &at(2).1["delta"]),
        (&json!(0), &json!({"content": "Hi"}))
    );
    assert_eq!(
        (at(3).0, &at(3).1["delta"]),
        (&json!(1), &json!({"content": "Yo"}))
    );
    assert_eq!(
        (at(4).0, &at(4).1["delta"]),
        (&json!(0), &json!({"content": " there"}))
    );
    assert_eq!(
        (at(5).0, &at(5).1["finish_reason"]),
        (&json!(1), &json!("length"))
    );
    assert_eq!(
        (at(6).0, &at(6).1["finish_reason"]),
        (&json!(0), &json!("stop"))
    );
    assert_eq!(chunks[7]["choices"], json!([]));
    assert_eq!(chunks[7]["usage"]["completion_tokens"], 3);
    assert_eq!(chunks[8], "[DONE]");

    // A ToolCalls event: message.tool_calls with the complete arguments string and
    // finish_reason tool_calls (content null when the model produced no text).
    let app = served_app(Arc::new(ServedScript::new(tool_call_script())));
    let tool_request = json!({"model": "m", "messages": [{"role": "user", "content": "Paris?"}],
                              "tools": weather_tools(), "tool_choice": "required"});
    let (s, _, body) = post(&app, "/v1/chat/completions", tool_request.clone()).await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_body(&body);
    let message = &v["choices"][0]["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["content"], Value::Null, "{v}");
    assert_eq!(
        message["tool_calls"],
        json!([{"id": CALL_ID, "type": "function",
                "function": {"name": "get_weather", "arguments": PARIS}}])
    );
    assert_eq!(message["tool_calls"][0]["function"]["arguments"], PARIS);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

    // Streaming: one delta.tool_calls entry per call with index, id, type and the complete
    // arguments, then the finish chunk with tool_calls.
    let mut streaming = tool_request.clone();
    streaming["stream"] = json!(true);
    let (s, _, body) = post(&app, "/v1/chat/completions", streaming).await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 4, "{body}");
    assert_eq!(
        chunks[1]["choices"][0]["delta"],
        json!({"tool_calls": [{"index": 0, "id": CALL_ID, "type": "function",
                               "function": {"name": "get_weather", "arguments": PARIS}}]})
    );
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], Value::Null);
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(chunks[3], "[DONE]");

    // response_format other than text with tool_choice other than none: 400 unsupported_parameter.
    let mut conflicting = tool_request.clone();
    conflicting["response_format"] = json!({"type": "json_object"});
    let (s, _, body) = post(&app, "/v1/chat/completions", conflicting).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_body(&body)["error"]["code"], "unsupported_parameter");

    // A server-side timeout mid-stream: error event (504 type timeout) followed by [DONE] (C-3);
    // the same event on a non-streaming request is a plain 504.
    let timed_out = vec![
        GenerationEvent::Started { choice: 0 },
        token_at(0, "Hel", 17),
        GenerationEvent::Error {
            code: ErrorCode::RequestTimeout,
            message: "request exceeded server.request_timeout".to_string(),
        },
    ];
    let app = served_app(Arc::new(ServedScript::new(timed_out)));
    let (s, _, body) = post(
        &app,
        "/v1/chat/completions",
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let chunks = sse_data(&body);
    assert_eq!(chunks.len(), 4, "{body}");
    assert_eq!(chunks[2]["error"]["code"], "request_timeout");
    assert_eq!(chunks[2]["error"]["type"], "timeout");
    assert_eq!(chunks[3], "[DONE]");
    let (s, _, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "m", "prompt": "hi"}),
    )
    .await;
    assert_eq!(s, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_eq!(json_body(&body)["error"]["code"], "request_timeout");
}

#[test]
fn phase2_request_fields() {
    // Every Phase 2 field is accepted at a non-default value.
    accepts(
        &completion(json!({
            "n": 3, "presence_penalty": 1.5, "frequency_penalty": -2, "repetition_penalty": 1.2,
            "logit_bias": {"5": 100, "9": -100}, "min_tokens": 2, "max_tokens": 8,
            "stop_token_ids": [7, 8], "priority": -5, "echo": true, "user": "alice",
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "object"}, "strict": true}}
        })),
        Endpoint::Completions,
    );
    let tool_chat = chat(json!({
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": CALL_ID, "type": "function",
                "function": {"name": "get_weather", "arguments": PARIS}}]},
            {"role": "tool", "tool_call_id": CALL_ID, "content": "{\"temperature\": 21}"}
        ],
        "tools": weather_tools(),
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "parallel_tool_calls": false
    }));
    accepts(&tool_chat, Endpoint::ChatCompletions);
    accepts(
        &chat(
            json!({"response_format": {"type": "json_object"}, "tools": weather_tools(),
                     "tool_choice": "none"}),
        ),
        Endpoint::ChatCompletions,
    );

    // Helpers for the engine.
    assert_eq!(
        completion(json!({"logit_bias": {"9": -1.5, "5": 2}})).logit_bias(),
        vec![(5, 2.0), (9, -1.5)]
    );
    assert_eq!(
        tool_chat.tool_choice_mode(),
        ToolChoiceMode::Named("get_weather".to_string())
    );
    assert!(!tool_chat.parallel_tool_calls_enabled());
    assert_eq!(
        tool_chat.tools_json(),
        weather_tools().as_array().cloned().expect("tools array")
    );
    assert_eq!(
        tool_chat.messages_json(),
        vec![
            json!({"role": "user", "content": "Weather in Paris?"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": CALL_ID,
                "type": "function", "function": {"name": "get_weather",
                "arguments": {"location": "Paris"}}}]}),
            json!({"role": "tool", "content": "{\"temperature\": 21}", "tool_call_id": CALL_ID}),
        ]
    );
    let auto = chat(json!({"tools": weather_tools()}));
    assert_eq!(auto.tool_choice_mode(), ToolChoiceMode::Auto);
    assert!(auto.parallel_tool_calls_enabled());
    assert_eq!(chat(json!({})).tool_choice_mode(), ToolChoiceMode::None);
    assert_eq!(
        chat(json!({"tools": weather_tools(), "tool_choice": "required"})).tool_choice_mode(),
        ToolChoiceMode::Required
    );
    assert_eq!(
        completion(json!({"response_format": {"type": "json_object"}})).response_format,
        Some(ResponseFormat::JsonObject)
    );

    // Out of range or malformed: 400 invalid_request naming the field.
    let invalid = [
        (
            completion(json!({"presence_penalty": 2.5})),
            Endpoint::Completions,
            "presence_penalty",
        ),
        (
            completion(json!({"frequency_penalty": -3})),
            Endpoint::Completions,
            "frequency_penalty",
        ),
        (
            completion(json!({"repetition_penalty": 0})),
            Endpoint::Completions,
            "repetition_penalty",
        ),
        (
            completion(json!({"logit_bias": {"5": 101}})),
            Endpoint::Completions,
            "logit_bias",
        ),
        (
            completion(json!({"logit_bias": {"x": 1}})),
            Endpoint::Completions,
            "logit_bias",
        ),
        (
            completion(json!({"min_tokens": 9, "max_tokens": 8})),
            Endpoint::Completions,
            "min_tokens",
        ),
        (
            completion(json!({"response_format": {"type": "json_schema",
                                                  "json_schema": {"name": "a"}}})),
            Endpoint::Completions,
            "schema",
        ),
        (
            chat(json!({"tools": weather_tools(), "tool_choice": "sometimes"})),
            Endpoint::ChatCompletions,
            "tool_choice",
        ),
        (
            chat(json!({"tool_choice": "required"})),
            Endpoint::ChatCompletions,
            "tool_choice",
        ),
        (
            chat(json!({"tools": [{"type": "function", "function": {"name": ""}}]})),
            Endpoint::ChatCompletions,
            "tools[0].function.name",
        ),
        (
            chat(json!({"messages": [{"role": "tool", "content": "42"}]})),
            Endpoint::ChatCompletions,
            "tool_call_id",
        ),
        (
            chat(json!({"messages": [{"role": "user", "content": "x", "tool_call_id": "c"}]})),
            Endpoint::ChatCompletions,
            "tool_call_id",
        ),
        (
            chat(
                json!({"messages": [{"role": "user", "content": "x", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]}),
            ),
            Endpoint::ChatCompletions,
            "tool_calls",
        ),
    ];
    for (req, endpoint, field) in invalid {
        let e = rejected(&req, endpoint);
        assert_eq!(e.status.as_u16(), 400, "{field}: {}", e.message);
        assert_eq!(e.code, ErrorCode::InvalidRequest, "{field}: {}", e.message);
        assert!(e.message.contains(field), "{:?} lacks {field}", e.message);
    }

    // Not supported on this endpoint or in combination: 400 unsupported_parameter.
    let unsupported = [
        (
            completion(json!({"tools": weather_tools()})),
            Endpoint::Completions,
            "tools",
        ),
        (
            chat(json!({"echo": true})),
            Endpoint::ChatCompletions,
            "echo",
        ),
        (
            chat(json!({"response_format": {"type": "json_object"}, "tools": weather_tools()})),
            Endpoint::ChatCompletions,
            "response_format",
        ),
        (
            chat(json!({"tools": [{"type": "retrieval", "function": {"name": "f"}}]})),
            Endpoint::ChatCompletions,
            "tools[0].type",
        ),
        (
            chat(json!({"best_of": 2})),
            Endpoint::ChatCompletions,
            "best_of",
        ),
    ];
    for (req, endpoint, field) in unsupported {
        let e = rejected(&req, endpoint);
        assert_eq!(
            e.code,
            ErrorCode::UnsupportedParameter,
            "{field}: {}",
            e.message
        );
        assert!(e.message.contains(field), "{:?} lacks {field}", e.message);
    }

    // A named function that is not in `tools`: 400 unknown_tool naming it.
    let e = rejected(
        &chat(json!({"tools": weather_tools(),
                     "tool_choice": {"type": "function", "function": {"name": "get_time"}}})),
        Endpoint::ChatCompletions,
    );
    assert_eq!((e.status.as_u16(), e.code), (400, ErrorCode::UnknownTool));
    assert!(e.message.contains("get_time"), "{}", e.message);
}

#[test]
fn phase2_error_constructors() {
    let cases = [
        (
            ApiError::queue_full(),
            429,
            ErrorType::RateLimitError,
            ErrorCode::QueueFull,
            Some(1),
        ),
        (
            ApiError::queue_timeout(),
            503,
            ErrorType::ServiceUnavailable,
            ErrorCode::QueueTimeout,
            None,
        ),
        (
            ApiError::context_exceeds_kv_capacity("needs 90 blocks, pool has 64"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::ContextExceedsKvCapacity,
            None,
        ),
        (
            ApiError::invalid_json_schema("unsupported keyword: patternProperties"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::InvalidJsonSchema,
            None,
        ),
        (
            ApiError::tools_not_supported("tiny-llama"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::ToolsNotSupported,
            None,
        ),
        (
            ApiError::unknown_tool("get_time"),
            400,
            ErrorType::InvalidRequestError,
            ErrorCode::UnknownTool,
            None,
        ),
        (
            ApiError::shutting_down(),
            503,
            ErrorType::ServiceUnavailable,
            ErrorCode::ShuttingDown,
            None,
        ),
        (
            ApiError::request_timeout(),
            504,
            ErrorType::Timeout,
            ErrorCode::RequestTimeout,
            None,
        ),
    ];
    for (e, status, kind, code, retry_after) in cases {
        assert_eq!(e.status.as_u16(), status, "{e:?}");
        assert_eq!(e.kind, kind, "{e:?}");
        assert_eq!(e.code, code, "{e:?}");
        assert_eq!(e.retry_after, retry_after, "{e:?}");
        // Engine-reported codes map to the same status, type and retry-after.
        let from = ApiError::from_code(code, "x");
        assert_eq!(
            (from.status, from.kind, from.retry_after),
            (e.status, e.kind, e.retry_after),
            "{code:?}"
        );
    }
    assert!(
        ApiError::tools_not_supported("tiny-llama")
            .message
            .contains("tiny-llama")
    );
    let slow = ApiError::from_code(ErrorCode::SlowClient, "client too slow");
    assert_eq!(slow.kind, ErrorType::ServerError);
    assert_eq!(NotReadyReason::ShuttingDown.as_str(), "shutting_down");
}
