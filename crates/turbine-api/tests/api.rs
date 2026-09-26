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

// ---- Phase 3: pressure document, circuit readiness, admission errors, reliability metrics ----

mod p3 {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use axum::Router;
    use axum::http::StatusCode;
    use serde_json::{Value, json};
    use turbine_api::backend::{BoxFuture, GenerationStream, InferenceRequest};
    use turbine_api::{
        ApiError, ApiLimits, ApiState, Diagnostics, ErrorCode, InferenceBackend, ModelCard,
        NotReadyReason, Readiness, ReadyState, readiness_for_circuit, router,
    };
    use turbine_core::clock::{Clock, FakeClock};
    use turbine_core::config::ReliabilityConfig;
    use turbine_core::telemetry::{
        DeviceSample, HostSample, LedgerProbe, LedgerSample, SourceStatus, TelemetrySample,
        ThrottleReasons,
    };
    use turbine_core::types::{
        CircuitState, DType, DeviceId, KvLayout, MemoryKind, PressureSignal, PressureState, Vendor,
    };
    use turbine_device::telemetry::proc::{ProcFile, ProcSource};
    use turbine_device::telemetry::{
        SamplerCore, TelemetryConfig, TelemetryMetrics, VendorTelemetry,
    };
    use turbine_device::{DeviceInfo, DeviceInventory, DeviceMemoryInfo};
    use turbine_observability::MetricsRegistry;
    use turbine_reliability::admission::{
        Admission, AdmissionParams, Calibration, PressureReason, RejectionReason,
    };
    use turbine_reliability::budget::{DeviceBudget, PoolKind};
    use turbine_reliability::circuit::{CircuitBreaker, CircuitEvent, CircuitReason};
    use turbine_reliability::controller::{
        ControllerHandle, EngineStats, NoReclaim, PressureController,
    };
    use turbine_reliability::ledger::Ledger;
    use turbine_reliability::metrics::ReliabilityMetrics;
    use turbine_reliability::recovery::RecoveryController;
    use turbine_reliability::reserve::{EmergencyReserve, ReserveAllocator};
    use turbine_reliability::signals::{SignalValue, effective_thresholds};
    use turbine_reliability::state::{Gates, MachineConfig, PressureMachine};
    use turbine_reliability::throttle::{
        KvReclaimer, SchedulerLimits, apply_reclaim, plan_for, publish_plan,
    };

    use super::{get_json, send};

    const GIB: u64 = 1 << 30;

    struct Model;
    impl InferenceBackend for Model {
        fn models(&self) -> Vec<ModelCard> {
            vec![ModelCard {
                id: "m".into(),
                object: "model".into(),
                created: 0,
                owned_by: "turbine".into(),
                max_model_len: 8192,
            }]
        }
    }

    /// Every submission fails with one fixed admission error.
    struct Rejecting(ApiError);
    impl InferenceBackend for Rejecting {
        fn models(&self) -> Vec<ModelCard> {
            Model.models()
        }
        fn submit(
            &self,
            req: InferenceRequest,
        ) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
            drop(req);
            let e = self.0.clone();
            Box::pin(async move { Err(e) })
        }
    }

    /// Readiness of a loaded engine behind a circuit breaker.
    struct CircuitReady(Arc<Mutex<CircuitState>>);
    impl Readiness for CircuitReady {
        fn ready(&self) -> ReadyState {
            readiness_for_circuit(*self.0.lock().unwrap(), ReadyState::Ready)
        }
    }

    struct Ready;
    impl Readiness for Ready {
        fn ready(&self) -> ReadyState {
            ReadyState::Ready
        }
    }

    /// Diagnostics backed by a pressure controller's handle (what turbine-server serves).
    struct Pressure(ControllerHandle);
    impl Diagnostics for Pressure {
        fn status(&self) -> Value {
            json!({
                "version": "0.1.0", "uptime_seconds": 0, "ready": true, "device_count": 1,
                "pressure_state": self.0.state(), "circuit_state": self.0.circuit(),
            })
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
            serde_json::to_value(self.0.document()).map_err(|e| ApiError::internal(e.to_string()))
        }
    }

    fn app_with(
        inference: Arc<dyn InferenceBackend>,
        diagnostics: Arc<dyn Diagnostics>,
        readiness: Arc<dyn Readiness>,
        metrics: MetricsRegistry,
    ) -> Router {
        router(ApiState {
            inference,
            diagnostics,
            readiness,
            metrics,
            limits: ApiLimits {
                max_request_bytes: 8 << 20,
            },
        })
    }

    fn no_diagnostics() -> Arc<dyn Diagnostics> {
        Arc::new(super::Phase0Diagnostics)
    }

    const CHAT: &[u8] = br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;

    struct Always;
    impl ReserveAllocator for Always {
        fn allocate(&mut self, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn free(&mut self) {}
    }

    fn budget() -> DeviceBudget {
        DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 30 * GIB,
            pools: vec![
                (PoolKind::Weights, 6 * GIB),
                (PoolKind::Kv, 20 * GIB),
                (PoolKind::Workspace, GIB),
                (PoolKind::Runtime, GIB),
                (PoolKind::Reserve, 2 * GIB),
            ],
        }
    }

    fn controller(
        enabled: bool,
        metrics: ReliabilityMetrics,
    ) -> (PressureController, ControllerHandle, FakeClock) {
        let cfg = ReliabilityConfig {
            enabled,
            ..ReliabilityConfig::default()
        };
        let budget = budget();
        let ledger = Ledger::new(&budget);
        ledger.set_metrics(metrics.clone());
        let reserve = EmergencyReserve::acquire(
            DeviceId(0),
            cfg.emergency_vram_reserve.0,
            &ledger,
            Box::new(Always),
            metrics.clone(),
        )
        .unwrap();
        let clock = FakeClock::new(Duration::ZERO);
        let (c, h) = PressureController::new(
            &cfg,
            SchedulerLimits {
                prefill_chunk_tokens: 2048,
                block_tokens: 16,
            },
            budget,
            ledger,
            reserve,
            Arc::new(NoReclaim),
            metrics,
            Arc::new(clock.clone()),
        );
        (c, h, clock)
    }

    fn sample(kv: f64) -> TelemetrySample {
        TelemetrySample {
            at_mono_ns: 0,
            host: HostSample {
                mem_available_bytes: Some(64 * GIB),
                swap_total_bytes: Some(0),
                swap_free_bytes: Some(0),
                pswpin_total: Some(0),
                psi_memory_some_avg10: Some(0.0),
                status: SourceStatus::Ok,
            },
            devices: vec![DeviceSample {
                temperature_c: Some(40.0),
                slowdown_temperature_c: Some(90.0),
                ..DeviceSample::empty(DeviceId(0), SourceStatus::Ok)
            }],
            ledger: LedgerSample {
                kv_utilization: kv,
                queue_fill: 0.1,
            },
            storage: None,
        }
    }

    fn run(c: &mut PressureController, clock: &FakeClock, secs: u64, kv: f64) {
        let stats = EngineStats {
            block_tokens: 16,
            free_kv_blocks: 1000,
            running_remaining_tokens: vec![100, 200],
            decode_tokens_per_s: 50.0,
            ..EngineStats::default()
        };
        for _ in 0..secs * 10 {
            clock.advance(Duration::from_millis(100));
            c.tick(&sample(kv), &stats);
        }
    }

    #[tokio::test]
    async fn ready_follows_circuit() {
        let circuit = Arc::new(Mutex::new(CircuitState::Healthy));
        let app = app_with(
            Arc::new(Model),
            no_diagnostics(),
            Arc::new(CircuitReady(Arc::clone(&circuit))),
            MetricsRegistry::new(),
        );
        for state in [
            CircuitState::CircuitOpen,
            CircuitState::Draining,
            CircuitState::Probing,
        ] {
            *circuit.lock().unwrap() = state;
            let (s, b) = get_json(&app, "GET", "/ready").await;
            assert_eq!(
                (s, b),
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"ready": false, "reason": "circuit_open"})
                ),
                "{state:?}"
            );
        }
        for state in [CircuitState::Healthy, CircuitState::Degraded] {
            *circuit.lock().unwrap() = state;
            let (s, b) = get_json(&app, "GET", "/ready").await;
            assert_eq!(
                (s, b),
                (StatusCode::OK, json!({"ready": true})),
                "{state:?}"
            );
        }
        // Not ready for another reason: that reason wins.
        let shutting = ReadyState::NotReady {
            reason: NotReadyReason::ShuttingDown,
        };
        assert_eq!(
            readiness_for_circuit(CircuitState::CircuitOpen, shutting),
            shutting
        );

        // While open, inference answers 503 circuit_open with the remaining cooldown.
        let app = app_with(
            Arc::new(Rejecting(ApiError::overload(
                ErrorCode::CircuitOpen,
                Some(12),
            ))),
            no_diagnostics(),
            Arc::new(Ready),
            MetricsRegistry::new(),
        );
        let (s, h, body) = send(&app, "POST", "/v1/chat/completions", CHAT.to_vec(), None).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let b: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(b["error"]["code"], "circuit_open", "{b}");
        assert_eq!(b["error"]["type"], "service_unavailable", "{b}");
        let retry: u64 = h["retry-after"].to_str().unwrap().parse().unwrap();
        assert_eq!(retry, 12);
        // Retry-After is at least 1 even when the cooldown is over.
        let e = ApiError::overload(ErrorCode::CircuitOpen, Some(0));
        assert_eq!(e.retry_after, Some(1));
    }

    fn assert_type(v: &Value, path: &str, kind: &str) {
        let ok = match kind {
            "string" => v.is_string(),
            "number" => v.is_number(),
            "bool" => v.is_boolean(),
            "array" => v.is_array(),
            "object" => v.is_object(),
            "string|null" => v.is_string() || v.is_null(),
            "number|null" => v.is_number() || v.is_null(),
            _ => unreachable!("{kind}"),
        };
        assert!(ok, "{path} should be {kind}, got {v}");
    }

    #[tokio::test]
    async fn pressure_document_shape() {
        let (mut c, handle, clock) = controller(true, ReliabilityMetrics::unregistered());
        run(&mut c, &clock, 2, 0.5);
        run(&mut c, &clock, 1, 0.85); // → ORANGE: a transition, a dominant signal
        let app = app_with(
            Arc::new(Model),
            Arc::new(Pressure(handle.clone())),
            Arc::new(Ready),
            MetricsRegistry::new(),
        );
        let (s, d) = get_json(&app, "GET", "/turbine/v1/pressure").await;
        assert_eq!(s, StatusCode::OK, "{d}");
        for (key, kind) in [
            ("enabled", "bool"),
            ("state", "string"),
            ("since", "string"),
            ("dominant_signal", "string|null"),
            ("exhaustion_horizon_seconds", "number|null"),
            ("signals", "array"),
            ("throttle", "object"),
            ("memory", "array"),
            ("admission", "object"),
            ("circuit", "object"),
            ("transitions", "array"),
        ] {
            assert_type(&d[key], key, kind);
        }
        assert_eq!(d["state"], "ORANGE");
        assert_eq!(d["dominant_signal"], "kv_utilization");
        let signal = &d["signals"][0];
        for (key, kind) in [
            ("name", "string"),
            ("value", "number"),
            ("level", "string"),
            ("stale", "bool"),
        ] {
            assert_type(&signal[key], &format!("signals[0].{key}"), kind);
        }
        for (key, kind) in [
            ("batch_growth_limit", "number|null"),
            ("prefill_budget_fraction", "number"),
            ("prefill_chunk_tokens", "number|null"),
            ("admission", "string"),
        ] {
            assert_type(&d["throttle"][key], &format!("throttle.{key}"), kind);
        }
        let mem = &d["memory"][0];
        for (key, kind) in [
            ("device", "number"),
            ("memory_kind", "string"),
            ("budget_bytes", "number"),
            ("pools", "array"),
            ("emergency_reserve_held", "bool"),
        ] {
            assert_type(&mem[key], &format!("memory[0].{key}"), kind);
        }
        let names: Vec<&str> = mem["pools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                for (key, kind) in [
                    ("name", "string"),
                    ("capacity_bytes", "number"),
                    ("used_bytes", "number"),
                    ("reserved_bytes", "number"),
                ] {
                    assert_type(&p[key], &format!("pools.{key}"), kind);
                }
                p["name"].as_str().unwrap()
            })
            .collect();
        assert_eq!(names, ["weights", "kv", "workspace", "runtime", "reserve"]);
        for (key, kind) in [
            ("queued", "number"),
            ("max_queue", "number"),
            ("decisions", "object"),
        ] {
            assert_type(&d["admission"][key], &format!("admission.{key}"), kind);
        }
        for (key, kind) in [
            ("admit", "number"),
            ("queue", "object"),
            ("reject", "object"),
        ] {
            assert_type(
                &d["admission"]["decisions"][key],
                &format!("admission.decisions.{key}"),
                kind,
            );
        }
        for (key, kind) in [
            ("state", "string"),
            ("since", "string"),
            ("last_reason", "string|null"),
        ] {
            assert_type(&d["circuit"][key], &format!("circuit.{key}"), kind);
        }
        let t = &d["transitions"][0];
        for (key, kind) in [
            ("at", "string"),
            ("from", "string"),
            ("to", "string"),
            ("signal", "string"),
            ("value", "number"),
            ("threshold", "number"),
        ] {
            assert_type(&t[key], &format!("transitions[0].{key}"), kind);
        }

        let (s, st) = get_json(&app, "GET", "/turbine/v1/status").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(st["pressure_state"], "ORANGE", "{st}");
        assert_eq!(st["circuit_state"], "HEALTHY", "{st}");

        // reliability.enabled: false → GREEN whatever the signals say.
        let (mut c, handle, clock) = controller(false, ReliabilityMetrics::unregistered());
        run(&mut c, &clock, 3, 0.99);
        let app = app_with(
            Arc::new(Model),
            Arc::new(Pressure(handle)),
            Arc::new(Ready),
            MetricsRegistry::new(),
        );
        let (s, d) = get_json(&app, "GET", "/turbine/v1/pressure").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(d["enabled"], false, "{d}");
        assert_eq!(d["state"], "GREEN", "{d}");
    }

    /// The error code the engine answers for each reject reason.
    fn code_for(reason: RejectionReason) -> ErrorCode {
        match reason {
            RejectionReason::ContextExceedsKvCapacity => ErrorCode::ContextExceedsKvCapacity,
            RejectionReason::QueueFull => ErrorCode::QueueFull,
            RejectionReason::QueueTimeout => ErrorCode::QueueTimeout,
            RejectionReason::Survival => ErrorCode::Overloaded,
            RejectionReason::CircuitOpen => ErrorCode::CircuitOpen,
            other => panic!("unmapped reason {other:?}"),
        }
    }

    /// (reason, retry hint the engine passes, status, type, code, Retry-After header)
    type MappingRow = (
        RejectionReason,
        Option<u64>,
        u16,
        &'static str,
        &'static str,
        Option<&'static str>,
    );

    #[tokio::test]
    async fn admission_error_mapping() {
        let rows: [MappingRow; 5] = [
            (
                RejectionReason::ContextExceedsKvCapacity,
                None,
                400,
                "invalid_request_error",
                "context_exceeds_kv_capacity",
                None,
            ),
            (
                RejectionReason::QueueFull,
                Some(7),
                429,
                "rate_limit_error",
                "queue_full",
                Some("7"),
            ),
            (
                RejectionReason::QueueTimeout,
                Some(7),
                503,
                "service_unavailable",
                "queue_timeout",
                Some("7"),
            ),
            (
                RejectionReason::Survival,
                Some(7),
                503,
                "service_unavailable",
                "overloaded",
                Some("7"),
            ),
            (
                RejectionReason::CircuitOpen,
                Some(1),
                503,
                "service_unavailable",
                "circuit_open",
                Some("1"),
            ),
        ];
        assert_eq!(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            RejectionReason::ALL.to_vec(),
            "one row per reject reason"
        );
        for (reason, retry, status, kind, code, header) in rows {
            let app = app_with(
                Arc::new(Rejecting(ApiError::overload(code_for(reason), retry))),
                no_diagnostics(),
                Arc::new(Ready),
                MetricsRegistry::new(),
            );
            let (s, h, body) =
                send(&app, "POST", "/v1/chat/completions", CHAT.to_vec(), None).await;
            let b: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(s.as_u16(), status, "{code}: {b}");
            assert_eq!(b["error"]["type"], kind, "{code}");
            assert_eq!(b["error"]["code"], code, "{code}");
            assert!(
                b["error"]["message"]
                    .as_str()
                    .is_some_and(|m| !m.is_empty())
            );
            assert_eq!(
                h.get("retry-after").map(|v| v.to_str().unwrap()),
                header,
                "{code}"
            );
            assert_eq!(reason.http(), (status, kind, code), "{reason:?}");
        }
        // A running request that exhausts its OOM retries: 503 resource_exhausted, server_error.
        let e = ApiError::overload(ErrorCode::ResourceExhausted, None);
        assert_eq!(e.status.as_u16(), 503);
        assert_eq!(e.to_json()["error"]["type"], "server_error");
        assert_eq!(e.to_json()["error"]["code"], "resource_exhausted");
    }

    struct Recording;
    impl KvReclaimer for Recording {
        fn demote(&self, _: f64) -> u64 {
            1 << 20
        }
        fn free_unreferenced(&self, _: f64) -> u64 {
            2 << 20
        }
        fn free_optional(&self) -> u64 {
            3 << 20
        }
    }

    struct FakeProc;
    impl ProcSource for FakeProc {
        fn read(&self, file: ProcFile) -> std::io::Result<String> {
            Ok(match file {
                ProcFile::Meminfo => {
                    "MemTotal: 131072000 kB\nMemAvailable: 44950556 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
                }
                ProcFile::Vmstat => "pswpin 1200\npswpout 0\n",
                ProcFile::PressureMemory => {
                    "some avg10=12.50 avg60=0.00 avg300=0.00 total=0\nfull avg10=4.02 avg60=0.00 avg300=0.00 total=0\n"
                }
            }
            .to_string())
        }
    }

    struct Probe;
    impl LedgerProbe for Probe {
        fn kv_utilization(&self) -> f64 {
            0.5
        }
        fn queue_fill(&self) -> f64 {
            0.1
        }
    }

    struct Vendor0;
    impl VendorTelemetry for Vendor0 {
        fn vendor(&self) -> Vendor {
            Vendor::Amd
        }
        fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String> {
            Ok(DeviceSample {
                temperature_c: Some(55.0),
                slowdown_temperature_c: Some(100.0),
                clock_mhz: Some(2350),
                power_watts: Some(180.0),
                utilization: Some(0.5),
                memory_used_bytes: Some(GIB),
                memory_free_bytes: Some(31 * GIB),
                throttle: ThrottleReasons {
                    power: true,
                    ..ThrottleReasons::default()
                },
                ..DeviceSample::empty(device.index, SourceStatus::Ok)
            })
        }
    }

    fn one_device() -> DeviceInventory {
        DeviceInventory {
            devices: vec![DeviceInfo {
                index: DeviceId(0),
                vendor: Vendor::Amd,
                vendor_index: 0,
                name: "AMD Radeon AI PRO R9700".into(),
                uuid: None,
                pci_bus_id: None,
                arch: Some("gfx1201".into()),
                driver_version: None,
                memory: DeviceMemoryInfo {
                    kind: MemoryKind::Dedicated,
                    total_bytes: 32 * GIB,
                    shared_with_host: false,
                },
            }],
            backends: Vec::new(),
        }
    }

    #[tokio::test]
    async fn reliability_metrics_bounded() {
        let registry = MetricsRegistry::new();
        let m = ReliabilityMetrics::register(&registry);
        let telemetry = TelemetryMetrics::register(&registry);
        let cfg = ReliabilityConfig::default();
        let clock = FakeClock::new(Duration::ZERO);
        let clock_dyn: Arc<dyn Clock> = Arc::new(clock.clone());

        // Every pressure state: one allocation failure → SURVIVAL, then step down to GREEN.
        let mut machine = PressureMachine::new(
            MachineConfig::from_config(&cfg),
            effective_thresholds(&cfg.pressure),
            m.clone(),
            Arc::clone(&clock_dyn),
        );
        let sig = |signal, value, level| SignalValue {
            signal,
            value,
            level,
            stale: false,
        };
        machine.evaluate(
            &[sig(
                PressureSignal::AllocationFailure,
                1.0,
                PressureState::Survival,
            )],
            Gates::default(),
        );
        for _ in 0..600 {
            clock.advance(Duration::from_millis(100));
            machine.evaluate(
                &[
                    sig(PressureSignal::AllocationFailure, 0.0, PressureState::Green),
                    sig(PressureSignal::KvUtilization, 0.1, PressureState::Green),
                ],
                Gates::default(),
            );
        }
        assert_eq!(machine.state(), PressureState::Green);

        // Circuit: DEGRADED, OPEN, DRAINING, PROBING, OPEN.
        let mut breaker = CircuitBreaker::new(&cfg.circuit, m.clone(), Duration::ZERO);
        let mut now = Duration::from_secs(1);
        breaker.on_event(CircuitEvent::Iteration, now);
        breaker.on_event(CircuitEvent::ThermalThrottle, now);
        assert_eq!(breaker.state(), CircuitState::Degraded);
        breaker.on_event(CircuitEvent::DeviceError { sticky: false }, now);
        breaker.on_event(CircuitEvent::Tick { running: 0 }, now);
        assert_eq!(breaker.state(), CircuitState::Draining);
        now += Duration::from_secs(31);
        breaker.on_event(CircuitEvent::Tick { running: 0 }, now);
        assert_eq!(breaker.state(), CircuitState::Probing);
        breaker.on_event(CircuitEvent::ProbeFailed, now);
        assert_eq!(breaker.state(), CircuitState::CircuitOpen);

        // Every queue and reject reason through the admission decision table.
        let budget = budget();
        let ledger = Ledger::new(&budget);
        ledger.set_metrics(m.clone());
        let layout = KvLayout {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        };
        let mut a = Admission::new(
            AdmissionParams {
                device: DeviceId(0),
                adaptive: true,
                max_queue: 4,
                large_prefill_tokens: 2048,
                block_bytes: layout.block_bytes(),
                prefill_chunk_tokens: 2048,
                workspace_bytes_per_token: 0,
                calibration: Calibration {
                    prefill_tokens_per_s: 10_000.0,
                    decode_step_s: 0.02,
                },
            },
            Arc::clone(&ledger),
            m.clone(),
        );
        let cheap = a.estimate(100, 0, Some(100), &layout, 8192);
        let expensive = a.estimate(4000, 0, Some(100), &layout, 8192);
        let huge = a.estimate(100, 0, Some(1 << 20), &layout, 1 << 21);
        let h = CircuitState::Healthy;
        a.decide(&cheap, PressureState::Green, h, 0);
        a.decide(&expensive, PressureState::Orange, h, 0);
        a.decide(&cheap, PressureState::Red, h, 0);
        a.decide(&cheap, PressureState::Red, h, 4);
        a.decide(&expensive, PressureState::Green, CircuitState::Degraded, 0);
        a.expensive_prefill_started();
        a.decide(&expensive, PressureState::Yellow, h, 0);
        a.decide(&cheap, PressureState::Survival, h, 0);
        a.decide(&cheap, PressureState::Green, CircuitState::CircuitOpen, 0);
        a.decide(&huge, PressureState::Green, h, 0);
        let held = ledger
            .reserve(DeviceId(0), PoolKind::Kv, 20 * GIB - GIB / 1024)
            .unwrap();
        a.decide(&cheap, PressureState::Green, h, 0);
        assert!(
            ledger.reserve(DeviceId(0), PoolKind::Kv, GIB).is_err(),
            "an allocation failure"
        );
        drop(held);
        // A queue timeout is counted by the scheduler's admission gate as a reject decision.
        m.admission_decisions
            .get_or_create(&turbine_reliability::metrics::DecisionLabels {
                decision: "reject",
                reason: RejectionReason::QueueTimeout.as_str(),
            })
            .inc();
        m.admission_queue_depth.set(3);
        m.admission_queue_wait_seconds.observe(0.25);

        // Recovery outcomes, reserve release, reclaim and the throttle plan.
        let mut rec = RecoveryController::new(&cfg.recovery, m.clone());
        rec.on_oom(8);
        rec.on_success();
        for _ in 0..=cfg.recovery.max_retries {
            rec.on_oom(8);
        }
        let mut reserve =
            EmergencyReserve::acquire(DeviceId(0), 2 * GIB, &ledger, Box::new(Always), m.clone())
                .unwrap();
        assert_eq!(
            reserve.release_for_recovery(PressureState::Survival),
            Ok(2 * GIB)
        );
        let kv = effective_thresholds(&cfg.pressure)[&PressureSignal::KvUtilization];
        let limits = SchedulerLimits {
            prefill_chunk_tokens: 2048,
            block_tokens: 16,
        };
        for state in PressureState::ALL {
            let plan = plan_for(state, &limits);
            publish_plan(&plan, &m);
            apply_reclaim(&plan, &Recording, &kv, &m);
        }
        m.reclaim_bytes
            .get_or_create(&turbine_reliability::metrics::ActionLabel {
                action: "release_reserve",
            })
            .inc_by(2 * GIB);
        for s in PressureSignal::ALL {
            let label = turbine_reliability::metrics::SignalLabel { signal: s.as_str() };
            m.pressure_signal.get_or_create(&label).set(0.5);
            m.pressure_signal_level.get_or_create(&label).set(1);
        }
        m.exhaustion_horizon_seconds.set(f64::INFINITY);

        // One device sample and the host readings through the telemetry sampler.
        let mut core = SamplerCore::new(
            TelemetryConfig::default(),
            &one_device(),
            vec![Box::new(Vendor0)],
            Box::new(FakeProc),
            Arc::new(Probe),
            Arc::new(FakeClock::new(Duration::ZERO)),
        )
        .with_metrics(telemetry.clone());
        core.poll();
        telemetry.record_device(
            &Vendor0.sample(&one_device().devices[0]).unwrap(),
            MemoryKind::Dedicated,
        );

        let app = app_with(
            Arc::new(Model),
            no_diagnostics(),
            Arc::new(Ready),
            registry.clone(),
        );
        let (s, _, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
        assert_eq!(s, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();

        let families = [
            "turbine_pressure_state",
            "turbine_pressure_transitions",
            "turbine_pressure_signal",
            "turbine_pressure_signal_level",
            "turbine_pressure_exhaustion_horizon_seconds",
            "turbine_admission_decisions",
            "turbine_admission_queue_depth",
            "turbine_admission_queue_wait_seconds",
            "turbine_throttle_plan",
            "turbine_reclaim_bytes",
            "turbine_memory_pool_bytes",
            "turbine_emergency_reserve_held",
            "turbine_emergency_reserve_releases",
            "turbine_allocation_failures",
            "turbine_recoveries",
            "turbine_recovery_retries",
            "turbine_circuit_state",
            "turbine_circuit_transitions",
            "turbine_gpu_temperature_celsius",
            "turbine_gpu_clock_mhz",
            "turbine_gpu_power_watts",
            "turbine_gpu_utilization_ratio",
            "turbine_gpu_memory_bytes",
            "turbine_gpu_throttle_active",
            "turbine_host_memory_available_bytes",
            "turbine_host_psi_memory_some_avg10",
            "turbine_host_swap_in_pages_per_second",
            "turbine_telemetry_stale",
            "turbine_telemetry_call_duration_seconds",
        ];
        for family in families {
            assert!(
                text.contains(&format!("# TYPE {family} ")),
                "no # TYPE line for {family}"
            );
        }

        // Every label value of every P3 series belongs to its closed set.
        let pressure_states: Vec<&str> = PressureState::ALL.iter().map(|s| s.as_str()).collect();
        let circuit_states: Vec<&str> = CircuitState::ALL.iter().map(|s| s.as_str()).collect();
        let states: Vec<&str> = pressure_states
            .iter()
            .chain(&circuit_states)
            .copied()
            .collect();
        let reasons: Vec<&str> = PressureReason::ALL
            .iter()
            .map(|r| r.as_str())
            .chain(RejectionReason::ALL.iter().map(|r| r.as_str()))
            .chain(CircuitReason::ALL.iter().map(|r| r.as_str()))
            .chain(["none", "thermal", "power", "other"])
            .collect();
        let mut allowed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        allowed.insert("state", states.clone());
        allowed.insert("from", states.clone());
        allowed.insert("to", states);
        allowed.insert(
            "signal",
            PressureSignal::ALL.iter().map(|s| s.as_str()).collect(),
        );
        allowed.insert("decision", vec!["admit", "queue", "reject"]);
        allowed.insert("reason", reasons);
        allowed.insert(
            "field",
            vec![
                "batch_growth_limit",
                "prefill_budget_fraction",
                "prefill_chunk_tokens",
            ],
        );
        allowed.insert(
            "action",
            vec!["demote", "free_cached", "free_optional", "release_reserve"],
        );
        allowed.insert("pool", PoolKind::ALL.iter().map(|p| p.as_str()).collect());
        allowed.insert("kind", vec!["capacity", "used", "reserved", "free"]);
        allowed.insert("outcome", vec!["recovered", "failed"]);
        let mut seen: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let Some(family) = families.iter().find(|f| line.starts_with(**f)) else {
                continue;
            };
            let Some(labels) = line
                .split_once('{')
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(l, _)| l)
            else {
                continue;
            };
            for pair in labels.split(',') {
                let (key, value) = pair.split_once('=').expect("key=\"value\"");
                let value = value.trim_matches('"');
                seen.entry(key.to_string())
                    .or_default()
                    .insert(value.to_string());
                match key {
                    "le" => {}
                    "device" => assert!(
                        value.parse::<u32>().is_ok_and(|d| d < 64),
                        "{family}: device {value}"
                    ),
                    "source" => assert!(
                        value == "host"
                            || value == "storage"
                            || value.parse::<u32>().is_ok_and(|d| d < 64),
                        "{family}: source {value}"
                    ),
                    _ => {
                        let set = allowed
                            .get(key)
                            .unwrap_or_else(|| panic!("{family}: unexpected label {key}"));
                        assert!(
                            set.contains(&value),
                            "{family}: {key}={value} is not in the closed set"
                        );
                    }
                }
            }
        }
        // The drive above reached every pressure state and each circuit state it walked.
        for s in pressure_states {
            assert!(seen["state"].contains(s), "state {s} never published");
        }
        for s in ["DEGRADED", "CIRCUIT_OPEN", "DRAINING", "PROBING"] {
            assert!(seen["to"].contains(s), "circuit never entered {s}");
        }
        for r in RejectionReason::ALL
            .iter()
            .map(|r| r.as_str())
            .chain(PressureReason::ALL.iter().map(|r| r.as_str()))
        {
            assert!(seen["reason"].contains(r), "reason {r} never counted");
        }
        assert!(seen["outcome"].contains("recovered") && seen["outcome"].contains("failed"));
    }
}
