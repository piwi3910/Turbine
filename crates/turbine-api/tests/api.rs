//! Phase 0 HTTP surface: route table, error shapes, body limit, request ids and metrics.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;
use turbine_api::{
    ApiError, ApiLimits, ApiState, Diagnostics, InferenceBackend, ModelCard, NotReadyReason,
    Readiness, ReadyState, router,
};
use turbine_observability::{MetricsRegistry, OPENMETRICS_CONTENT_TYPE};

struct NoModel;
impl InferenceBackend for NoModel {
    fn models(&self) -> Vec<ModelCard> {
        Vec::new()
    }
}

struct Phase0Diagnostics;
impl Diagnostics for Phase0Diagnostics {
    fn status(&self) -> Value {
        json!({"version": "0.1.0", "uptime_seconds": 0, "ready": false, "device_count": 0})
    }
    fn devices(&self) -> Value {
        json!({"devices": [], "backends": []})
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

struct NotReady;
impl Readiness for NotReady {
    fn ready(&self) -> ReadyState {
        ReadyState::NotReady {
            reason: NotReadyReason::NoModelLoaded,
        }
    }
}

fn app(max_request_bytes: usize) -> Router {
    router(ApiState {
        inference: Arc::new(NoModel),
        diagnostics: Arc::new(Phase0Diagnostics),
        readiness: Arc::new(NotReady),
        metrics: MetricsRegistry::new(),
        limits: ApiLimits { max_request_bytes },
    })
}

async fn send(
    app: &Router,
    method: &str,
    path: &str,
    body: Vec<u8>,
    request_id: Option<&str>,
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let mut req = Request::builder().method(method).uri(path);
    if let Some(id) = request_id {
        req = req.header("x-request-id", id);
    }
    let resp = app
        .clone()
        .oneshot(req.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap()
        .to_vec();
    (status, headers, bytes)
}

async fn get_json(app: &Router, method: &str, path: &str) -> (StatusCode, Value) {
    let (status, _, body) = send(app, method, path, Vec::new(), None).await;
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

fn assert_error(body: &Value, kind: &str, code: &str) {
    assert_eq!(body["error"]["type"], kind, "{body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty()),
        "{body}"
    );
}

#[tokio::test]
async fn route_table_phase0() {
    let app = app(8 << 20);

    let (s, b) = get_json(&app, "GET", "/health").await;
    assert_eq!((s, b), (StatusCode::OK, json!({"status": "ok"})));

    let (s, b) = get_json(&app, "GET", "/ready").await;
    assert_eq!(
        (s, b),
        (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"ready": false, "reason": "no_model_loaded"})
        )
    );

    let (s, headers, _) = send(&app, "GET", "/metrics", Vec::new(), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(headers["content-type"], OPENMETRICS_CONTENT_TYPE);

    let (s, b) = get_json(&app, "GET", "/v1/models").await;
    assert_eq!(
        (s, b),
        (StatusCode::OK, json!({"object": "list", "data": []}))
    );

    for path in ["/v1/chat/completions", "/v1/completions"] {
        let (s, _, body) = send(&app, "POST", path, br#"{"model":"x"}"#.to_vec(), None).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{path}");
        assert_error(
            &serde_json::from_slice(&body).unwrap(),
            "service_unavailable",
            "model_not_loaded",
        );
    }

    let (s, b) = get_json(&app, "GET", "/turbine/v1/status").await;
    assert_eq!(s, StatusCode::OK);
    for key in ["version", "uptime_seconds", "ready", "device_count"] {
        assert!(b.get(key).is_some(), "status lacks {key}: {b}");
    }

    let (s, b) = get_json(&app, "GET", "/turbine/v1/devices").await;
    assert_eq!(s, StatusCode::OK);
    assert!(b["devices"].is_array() && b["backends"].is_array(), "{b}");

    for path in [
        "/turbine/v1/kv",
        "/turbine/v1/pressure",
        "/turbine/v1/scheduler",
    ] {
        let (s, b) = get_json(&app, "GET", path).await;
        assert_eq!(s, StatusCode::NOT_IMPLEMENTED, "{path}");
        assert_error(&b, "not_implemented", "not_implemented");
    }

    let (s, b) = get_json(&app, "GET", "/v1/completions").await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    assert_error(&b, "invalid_request_error", "method_not_allowed");

    let (s, b) = get_json(&app, "GET", "/no/such/route").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&b, "not_found", "not_found");
}

#[tokio::test]
async fn body_limit_413() {
    let limit = 1024;
    let app = app(limit);
    let (s, _, body) = send(
        &app,
        "POST",
        "/v1/chat/completions",
        vec![b'a'; limit + 1],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    assert_error(
        &serde_json::from_slice(&body).unwrap(),
        "invalid_request_error",
        "request_too_large",
    );

    // At the limit the body is accepted (and Phase 0 answers 503).
    let (s, _, _) = send(
        &app,
        "POST",
        "/v1/chat/completions",
        vec![b'a'; limit],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
}

fn metric_value(text: &str, series: &str) -> Option<String> {
    text.lines().find_map(|l| {
        l.strip_prefix(series)
            .and_then(|rest| rest.strip_prefix(' '))
            .map(str::to_string)
    })
}

#[tokio::test]
async fn metrics_counts_requests() {
    let app = app(8 << 20);
    send(&app, "GET", "/health", Vec::new(), None).await;
    send(&app, "GET", "/health", Vec::new(), None).await;
    let (s, headers, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(headers["content-type"], OPENMETRICS_CONTENT_TYPE);
    let text = String::from_utf8(body).unwrap();
    assert_eq!(
        metric_value(
            &text,
            r#"turbine_http_requests_total{method="GET",route="/health",status="200"}"#
        )
        .as_deref(),
        Some("2"),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|l| l.starts_with("turbine_build_info{version=")),
        "{text}"
    );
    assert!(
        text.contains(
            r#"turbine_http_request_duration_seconds_count{method="GET",route="/health"} 2"#
        ),
        "{text}"
    );
}

#[tokio::test]
async fn unmatched_route_label_is_bounded() {
    let app = app(8 << 20);
    for i in 0..50 {
        let (s, _, _) = send(&app, "GET", &format!("/unknown/path-{i}"), Vec::new(), None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }
    let (_, _, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
    let text = String::from_utf8(body).unwrap();
    assert_eq!(
        metric_value(
            &text,
            r#"turbine_http_requests_total{method="GET",route="unmatched",status="404"}"#
        )
        .as_deref(),
        Some("50"),
        "{text}"
    );
    assert!(
        !text.contains("/unknown/path-"),
        "a raw path leaked into a label: {text}"
    );
}

fn is_uuid_v4(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok_and(|u| u.get_version_num() == 4)
}

#[tokio::test]
async fn request_id_echoed_or_generated() {
    let app = app(8 << 20);

    let (_, h, _) = send(&app, "GET", "/health", Vec::new(), Some("abc-123")).await;
    assert_eq!(h["x-request-id"], "abc-123");

    let (_, h, _) = send(&app, "GET", "/health", Vec::new(), None).await;
    assert!(
        is_uuid_v4(h["x-request-id"].to_str().unwrap()),
        "{:?}",
        h["x-request-id"]
    );

    let long = "a".repeat(200);
    let (_, h, _) = send(&app, "GET", "/health", Vec::new(), Some(&long)).await;
    assert!(
        is_uuid_v4(h["x-request-id"].to_str().unwrap()),
        "{:?}",
        h["x-request-id"]
    );

    // Every response carries the header, including errors and fallbacks.
    for (method, path) in [
        ("GET", "/nope"),
        ("POST", "/v1/completions"),
        ("GET", "/turbine/v1/kv"),
        ("GET", "/v1/completions"),
    ] {
        let (_, h, _) = send(&app, method, path, Vec::new(), None).await;
        assert!(h.contains_key("x-request-id"), "{method} {path}");
    }
}

/// Phase 2m S-11 (from the Phase 8 run-ahead): `turbine_support_matrix_status` pre-creates all
/// three statuses at 0 and sets the resolved one to 1.
#[tokio::test]
async fn support_matrix_status_gauge() {
    use turbine_api::support::SupportMetrics;
    use turbine_core::support::{self, SupportKey};

    let decision = support::check(SupportKey::bf16("cpu", "cpu", "LlamaForCausalLM")).unwrap();
    let metrics = MetricsRegistry::new();
    let gauge = SupportMetrics::register(&metrics);
    let app = router(ApiState {
        inference: Arc::new(NoModel),
        diagnostics: Arc::new(Phase0Diagnostics),
        readiness: Arc::new(NotReady),
        metrics: metrics.clone(),
        limits: ApiLimits {
            max_request_bytes: 1 << 20,
        },
    });
    let status_lines = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|l| l.starts_with("turbine_support_matrix_status{"))
            .map(str::to_string)
            .collect()
    };

    let (code, _, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
    assert_eq!(code, StatusCode::OK);
    let lines = status_lines(&String::from_utf8(body).unwrap());
    assert_eq!(lines.len(), 3, "{lines:?}");
    assert!(lines.iter().all(|l| l.ends_with(" 0")), "{lines:?}");

    gauge.set(&decision.status);
    let (_, _, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
    let lines = status_lines(&String::from_utf8(body).unwrap());
    let ones: Vec<&String> = lines.iter().filter(|l| l.ends_with(" 1")).collect();
    assert_eq!(
        ones,
        ["turbine_support_matrix_status{status=\"experimental\"} 1"],
        "{lines:?}"
    );
}
