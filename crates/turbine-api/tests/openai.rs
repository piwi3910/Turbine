//! Phase 1 OpenAI request surface: field validation (S-10) and the `InferenceBackend::submit` seam.

use serde_json::{Value, json};
use turbine_api::backend::{BoxFuture, GenerationStream, InferenceRequest};
use turbine_api::openai::request::{OpenAiRequest, PromptInput};
use turbine_api::{ApiError, ErrorCode, ErrorType, InferenceBackend, ModelCard, NotReadyReason};
use turbine_core::request::{Endpoint, FinishReason, GenerationEvent, Usage};
use turbine_core::types::RequestId;

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
