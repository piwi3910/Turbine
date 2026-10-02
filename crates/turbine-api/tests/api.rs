//! Phase 0 HTTP surface: route table, error shapes, body limit, request ids and metrics.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;
use turbine_api::{
    ApiError, ApiLimits, ApiState, Diagnostics, InferenceBackend, ModelCard, NotReadyReason,
    Readiness, ReadyState, TopologyScope, router,
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
        assert_eq!(
            names,
            [
                "weights",
                "kv",
                "workspace",
                "collective",
                "runtime",
                "reserve"
            ]
        );
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
        // SURVIVAL (option A) returns an admitted request that had not started to the queue.
        a.record_requeue(turbine_core::types::RequestId::new_v4(), &cheap);
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

// ---- Phase 4: session hints, cache salt, cached tokens and the KV routes -------------------

mod kv_sim {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use serde_json::Value;
    use turbine_api::backend::BoxFuture;
    use turbine_api::kv::{PrefetchAccepted, PrefetchRequest, PrefetchTarget};
    use turbine_api::openai::request::{MessageContent, PromptInput};
    use turbine_api::{
        ApiError, Diagnostics, GenerationStream, InferenceBackend, InferenceRequest, ModelCard,
    };
    use turbine_core::clock::{Clock, FakeClock};
    use turbine_core::config::KvConfig;
    use turbine_core::request::{FinishReason, GenerationEvent, SessionHints, Usage};
    use turbine_core::types::{
        DType, DeviceId, KvDtype, KvLayout, MemoryKind, ModelIdentity, PressureState, Priority,
    };
    use turbine_kv::hierarchy::{
        AttachOutcome, AttachRequest, HierarchyConfig, KvHierarchy, PrefetchError,
        PrefetchTarget as KvTarget, PrefixAttach,
    };
    use turbine_kv::identity::KvFormat;
    use turbine_kv::metrics::{EvictReason, KvMetrics};
    use turbine_kv::tier::{KvTier, MemTier, TierId};
    use turbine_kv::transfer::SimTransferBackend;
    use turbine_kv::{BlockPool, BlockPoolConfig};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    pub const MODEL: &str = "sim";

    struct Inner {
        clock: FakeClock,
        h: KvHierarchy,
        pool: BlockPool,
        backend: SimTransferBackend,
    }

    /// An inference backend whose only "model" is the real KV hierarchy: prompts are split
    /// into word tokens, prefixes attach and commit, and each request samples one token.
    pub struct KvSim {
        inner: Mutex<Inner>,
    }

    /// Word tokens: FNV-1a of each whitespace-separated word.
    fn tokenize(text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| {
                w.bytes().fold(0x811c_9dc5u32, |h, b| {
                    (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
                })
            })
            .collect()
    }

    fn messages_text(messages: &[turbine_api::openai::request::ChatMessageIn]) -> String {
        messages
            .iter()
            .filter_map(|m| match &m.content {
                Some(MessageContent::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    impl KvSim {
        /// L0 of 64 blocks, L1 and L2 of 64 blocks each, 16-token blocks, prefetch queue 2.
        pub fn new(reg: &MetricsRegistry) -> Arc<KvSim> {
            let clock = FakeClock::new(Duration::ZERO);
            let arc: Arc<dyn Clock> = Arc::new(clock.clone());
            let layout = KvLayout {
                num_layers: 2,
                num_kv_heads: 2,
                head_dim: 16,
                dtype: DType::BF16,
                block_tokens: 16,
            };
            let bb = layout.block_bytes();
            let tier = |id| {
                Arc::new(MemTier::payload_free(id, 64 * bb, bb, arc.clone())) as Arc<dyn KvTier>
            };
            let (l1, l2) = (tier(TierId::L1), tier(TierId::L2));
            let mut kv = KvConfig {
                block_tokens: layout.block_tokens,
                ..KvConfig::default()
            };
            kv.prefetch.max_queue = 2;
            let h = KvHierarchy::new(
                HierarchyConfig::from_config(&kv, bb, MemoryKind::Dedicated)
                    .expect("the default eviction policy is registered"),
                ModelIdentity {
                    config_hash: [5; 32],
                    weights_index_hash: [6; 32],
                    rope_hash: [0; 32],
                },
                KvFormat {
                    dtype: KvDtype::Bf16,
                    layout,
                    shards: 1,
                    scales: None,
                },
                64,
                Some(l1.clone()),
                Some(l2.clone()),
                arc.clone(),
                KvMetrics::register(reg),
            );
            let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
            let pool = BlockPool::new(
                BlockPoolConfig {
                    layout,
                    num_blocks: 64,
                },
                mem,
            )
            .unwrap();
            let backend = SimTransferBackend::new(arc, Some(l1), Some(l2), bb as usize);
            Arc::new(KvSim {
                inner: Mutex::new(Inner {
                    clock,
                    h,
                    pool,
                    backend,
                }),
            })
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
            self.inner.lock().unwrap()
        }

        pub fn set_pressure(&self, s: PressureState) {
            self.lock().h.set_l0_state(s);
        }

        /// Copies every unreferenced L0 block down (to L1, spilling to L2) and lets the copies
        /// finish.
        pub fn demote_all(&self) {
            let g = &mut *self.lock();
            for _ in 0..8 {
                g.h.demote_to(&mut g.pool, 0.0, EvictReason::Pressure);
                g.h.poll(&mut g.pool, &mut g.backend);
                g.clock.advance(Duration::from_millis(10));
                g.h.poll(&mut g.pool, &mut g.backend);
            }
        }

        /// Pumps transfers without advancing past pending copies' due times.
        pub fn settle(&self) {
            let g = &mut *self.lock();
            for _ in 0..4 {
                g.clock.advance(Duration::from_millis(10));
                g.h.poll(&mut g.pool, &mut g.backend);
            }
        }

        fn run(
            &self,
            prompt: &[u32],
            salt: &str,
            session: Option<&SessionHints>,
            allow_lossy: Option<bool>,
        ) -> PrefixAttach {
            let g = &mut *self.lock();
            let id = turbine_core::types::RequestId::new_v4();
            let req = AttachRequest {
                request: id,
                prompt,
                cache_salt: salt,
                session,
                priority: Priority::default(),
                allow_lossy,
            };
            let attach = 'attach: loop {
                match g.h.attach_prefix(&mut g.pool, &req) {
                    AttachOutcome::Ready(a) => break a,
                    AttachOutcome::Promoting => loop {
                        g.clock.advance(Duration::from_millis(1));
                        let ready = g.h.poll(&mut g.pool, &mut g.backend);
                        if let Some((_, a)) = ready.into_iter().find(|(r, _)| *r == id) {
                            break 'attach a;
                        }
                    },
                    AttachOutcome::WaitForPrefix => {
                        g.clock.advance(Duration::from_millis(1));
                        g.h.poll(&mut g.pool, &mut g.backend);
                    }
                }
            };
            let need = (prompt.len().div_ceil(16) - attach.blocks.len()) as u32;
            let mut table = attach.blocks.to_vec();
            table.extend(g.pool.allocate(need).expect("room in L0"));
            g.h.after_plan(&mut g.pool);
            g.h.commit_progress(&mut g.pool, id, &table, prompt);
            g.pool.release(&table);
            g.h.request_done(&mut g.pool, id, false);
            g.h.poll(&mut g.pool, &mut g.backend);
            attach
        }
    }

    impl InferenceBackend for KvSim {
        fn models(&self) -> Vec<ModelCard> {
            vec![ModelCard {
                id: MODEL.into(),
                object: "model".into(),
                created: 0,
                owned_by: "turbine".into(),
                max_model_len: 4096,
            }]
        }

        fn submit(
            &self,
            req: InferenceRequest,
        ) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
            let text = match (&req.body.prompt, &req.body.messages) {
                (Some(PromptInput::Text(t)), _) => t.clone(),
                (_, Some(m)) => messages_text(m),
                _ => String::new(),
            };
            let prompt = tokenize(&text);
            let session = req.body.prompt_cache_key.clone().map(|id| SessionHints {
                session_id: id,
                resume_within_secs: req.hints.session_resume_within,
                end: req.hints.session_end,
            });
            let salt = req.hints.cache_salt.clone().unwrap_or_default();
            let allow_lossy = req.hints.kv_policy.map(|p| p.allow_lossy);
            let attach = self.run(&prompt, &salt, session.as_ref(), allow_lossy);
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::channel(8);
                let usage = Usage {
                    prompt_tokens: prompt.len() as u32,
                    completion_tokens: 1,
                    cached_tokens: attach.cached_tokens,
                    lossy_cached_tokens: attach.lossy_tokens,
                };
                for e in [
                    GenerationEvent::Started { choice: 0 },
                    GenerationEvent::Token {
                        choice: 0,
                        text: "ok".into(),
                        token_id: 1,
                        logprob: None,
                        top_logprobs: Vec::new(),
                    },
                    GenerationEvent::Finished {
                        choice: 0,
                        reason: FinishReason::Length,
                        usage: Some(usage),
                    },
                ] {
                    tx.send(e).await.expect("the receiver is alive");
                }
                Ok(rx)
            })
        }

        fn prefetch(
            &self,
            req: PrefetchRequest,
        ) -> BoxFuture<'_, Result<PrefetchAccepted, ApiError>> {
            let result = {
                let g = &mut *self.lock();
                let salt = req.cache_salt.clone().unwrap_or_default();
                let tokens = match &req.target {
                    PrefetchTarget::Session { .. } => Vec::new(),
                    PrefetchTarget::Prompt { prompt } => tokenize(prompt),
                    PrefetchTarget::Messages { messages } => tokenize(&messages_text(messages)),
                };
                let target = match &req.target {
                    PrefetchTarget::Session { session_id } => KvTarget::Session(session_id),
                    _ => KvTarget::Tokens {
                        prompt: &tokens,
                        cache_salt: &salt,
                    },
                };
                g.h.prefetch(&mut g.pool, target)
            };
            Box::pin(async move {
                match result {
                    Ok(a) => Ok(PrefetchAccepted {
                        blocks_queued: a.blocks_queued,
                        blocks_resident: a.blocks_resident,
                    }),
                    Err(PrefetchError::SessionNotFound) => Err(ApiError::session_not_found()),
                    Err(PrefetchError::QueueFull) => Err(ApiError::prefetch_queue_full()),
                    Err(PrefetchError::PressureTooHigh) => Err(ApiError::pressure_too_high()),
                }
            })
        }
    }

    /// Diagnostics serving the hierarchy's KV document.
    pub struct KvDiag(pub Arc<KvSim>);

    impl Diagnostics for KvDiag {
        fn status(&self) -> Value {
            serde_json::json!({})
        }
        fn devices(&self) -> Value {
            serde_json::json!({})
        }
        fn scheduler(&self) -> Result<Value, ApiError> {
            Err(ApiError::not_implemented())
        }
        fn kv(&self) -> Result<Value, ApiError> {
            let g = self.0.lock();
            serde_json::to_value(g.h.document(&g.pool, (0, 0)))
                .map_err(|e| ApiError::internal(e.to_string()))
        }
        fn pressure(&self) -> Result<Value, ApiError> {
            Err(ApiError::not_implemented())
        }
    }
}

struct Ready;
impl Readiness for Ready {
    fn ready(&self) -> ReadyState {
        ReadyState::Ready
    }
}

fn kv_app() -> (Router, Arc<kv_sim::KvSim>, MetricsRegistry) {
    let metrics = MetricsRegistry::new();
    let sim = kv_sim::KvSim::new(&metrics);
    let app = router(ApiState {
        inference: sim.clone(),
        diagnostics: Arc::new(kv_sim::KvDiag(sim.clone())),
        readiness: Arc::new(Ready),
        metrics: metrics.clone(),
        limits: ApiLimits {
            max_request_bytes: 1 << 20,
        },
    });
    (app, sim, metrics)
}

async fn post_json(
    app: &Router,
    path: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, Value) {
    let mut req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = app
        .clone()
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn words(n: usize, base: usize) -> String {
    (base..base + n)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn cached(app: &Router, prompt: &str, headers: &[(&str, &str)]) -> u64 {
    let (s, v) = post_json(
        app,
        "/v1/completions",
        json!({"model": kv_sim::MODEL, "prompt": prompt, "max_tokens": 1}),
        headers,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("no cached_tokens in {v}"))
}

#[tokio::test]
async fn cache_salt_isolates() {
    let (app, _sim, _) = kv_app();
    let prompt = words(40, 0);
    assert_eq!(cached(&app, &prompt, &[]).await, 0);
    assert!(
        cached(&app, &prompt, &[]).await > 0,
        "shared without a salt"
    );
    let a = [("x-turbine-cache-salt", "a")];
    assert_eq!(
        cached(&app, &prompt, &a).await,
        0,
        "a salt is its own namespace"
    );
    assert!(cached(&app, &prompt, &a).await > 0, "same salt shares");
    let b = [("x-turbine-cache-salt", "b")];
    assert_eq!(
        cached(&app, &prompt, &b).await,
        0,
        "another salt does not share"
    );
    let long = "s".repeat(129);
    let (s, v) = post_json(
        &app,
        "/v1/completions",
        json!({"model": kv_sim::MODEL, "prompt": prompt, "max_tokens": 1}),
        &[("x-turbine-cache-salt", &long)],
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "invalid_request_error", "invalid_cache_salt");
}

#[tokio::test]
async fn kv_routes() {
    let (app, sim, _) = kv_app();

    // The P4 document: every key of the §Data example.
    let (s, doc) = get_json(&app, "GET", "/turbine/v1/kv").await;
    assert_eq!(s, StatusCode::OK);
    for key in [
        "policy",
        "prefix_sharing",
        "block_tokens",
        "block_bytes",
        "tiers",
        "unified_memory",
        "hit_rate",
        "sessions",
        "prefetch",
        "transfers",
    ] {
        assert!(doc.get(key).is_some(), "{key} missing from {doc}");
    }
    for tier in doc["tiers"].as_array().unwrap() {
        for key in [
            "tier",
            "enabled",
            "capacity_bytes",
            "used_bytes",
            "blocks",
            "referenced_blocks",
            "pressure",
            "degraded",
            "est_latency_seconds",
            "est_bandwidth_bytes_per_second",
        ] {
            assert!(tier.get(key).is_some(), "{key} missing from {tier}");
        }
    }
    for (object, keys) in [
        (
            "hit_rate",
            &["window_seconds", "prompt_tokens", "cached_tokens"][..],
        ),
        ("sessions", &["active", "max"][..]),
        ("prefetch", &["queued", "used", "wasted", "cancelled"][..]),
        ("transfers", &["inflight_bytes", "max_inflight_bytes"][..]),
    ] {
        for key in keys {
            assert!(doc[object].get(*key).is_some(), "{object}.{key} missing");
        }
    }
    assert_eq!(doc["sessions"]["active"], 0);

    // A chat request with a session id opens a session.
    let chat = |key: &str| {
        json!({"model": kv_sim::MODEL, "max_tokens": 1, "prompt_cache_key": key,
               "messages": [{"role": "user", "content": words(64, 100)}]})
    };
    let (s, v) = post_json(&app, "/v1/chat/completions", chat("s1"), &[]).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (_, doc) = get_json(&app, "GET", "/turbine/v1/kv").await;
    assert_eq!(doc["sessions"]["active"], 1);

    // Its blocks leave L0; a prefetch by session id queues their promotion.
    sim.demote_all();
    let (s, v) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"session_id": "s1"}),
        &[],
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["blocks_queued"], 2, "the queue holds 2 prefetches: {v}");
    assert!(v.get("blocks_resident").is_some());
    let (s, v) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"session_id": "s1"}),
        &[],
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_error(&v, "rate_limit_error", "prefetch_queue_full");
    sim.settle();

    let (s, v) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"session_id": "nope"}),
        &[],
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "not_found", "session_not_found");

    sim.set_pressure(turbine_core::types::PressureState::Orange);
    let (s, v) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"prompt": words(32, 100)}),
        &[],
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);
    assert_error(&v, "invalid_request_error", "pressure_too_high");
    sim.set_pressure(turbine_core::types::PressureState::Green);
    let (s, v) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"messages": [{"role": "user", "content": words(32, 100)}]}),
        &[("x-turbine-cache-salt", "zz")],
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    let (s, v) = post_json(&app, "/turbine/v1/kv/prefetch", json!({"other": 1}), &[]).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "invalid_request_error", "invalid_request");

    // Session hint validation.
    let (s, v) = post_json(&app, "/v1/chat/completions", chat(&"k".repeat(129)), &[]).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "invalid_request_error", "invalid_session_id");
    let (s, v) = post_json(
        &app,
        "/v1/chat/completions",
        chat("s2"),
        &[("x-turbine-session-resume-within", "0")],
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "invalid_request_error", "invalid_session_hint");
    let (s, v) = post_json(
        &app,
        "/v1/completions",
        json!({"model": kv_sim::MODEL, "prompt": "hi", "max_tokens": 1}),
        &[("x-turbine-session-end", "true")],
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "invalid_request_error", "invalid_session_hint");
    let (s, _) = post_json(
        &app,
        "/v1/chat/completions",
        chat("s2"),
        &[
            ("x-turbine-session-resume-within", "600"),
            ("x-turbine-session-end", "false"),
        ],
    )
    .await;
    assert_eq!(s, StatusCode::OK);
}

/// Every label value of a KV metric series belongs to its documented closed set.
fn check_labels(line: &str) {
    const TIERS: &[&str] = &["l0", "l1", "l2"];
    const PATHS: &[&str] = &[
        "l0_to_l1", "l1_to_l0", "l1_to_l2", "l2_to_l1", "l0_to_l2", "l2_to_l0",
    ];
    const REASONS: &[&str] = &[
        "capacity",
        "pressure",
        "session_expired",
        "checksum",
        "tier_degraded",
        "below_min_value",
        "no_room",
        "all_l0",
        "retrieve_cheaper",
        "recompute_cheaper",
        "l0_pressure",
        "no_match",
        "compressed",
        "ladder_floor",
    ];
    // P6b S-6: the compression ladder's actions name codecs, not tiers, as from/to.
    const LADDER_REASONS: &[&str] = &[
        "fill_high_water",
        "would_drop",
        "new_demotion",
        "rung_step_up",
        "floor_evict",
        // A rewrite whose tier had no slot of the new format (counted by the server).
        "no_room",
    ];
    const CODECS: &[&str] = &["l0", "fp8_e4m3", "tq4", "tq2"];
    let ladder = line.starts_with("turbine_kv_ladder_actions");
    let Some(labels) = line.split_once('{').map(|(_, rest)| rest) else {
        return;
    };
    let labels = labels.split_once('}').map_or(labels, |(l, _)| l);
    for pair in labels.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or_else(|| panic!("{line}"));
        let v = v.trim_matches('"');
        let ok = match k {
            "from" if ladder => CODECS.contains(&v),
            "to" if ladder => CODECS.contains(&v) || v == "evict",
            "reason" if ladder => LADDER_REASONS.contains(&v),
            "tier" | "from" | "to" => TIERS.contains(&v),
            "state" => ["used", "free"].contains(&v),
            "kind" => ["capacity", "used"].contains(&v),
            "result" => TIERS.contains(&v) || v == "miss",
            "reason" => REASONS.contains(&v),
            "path" => PATHS.contains(&v),
            "outcome" => ["used", "wasted", "cancelled", "rejected"].contains(&v),
            "le" => true,
            _ => false,
        };
        assert!(ok, "label {k}={v} outside its documented set: {line}");
    }
}

#[tokio::test]
async fn kv_metrics_bounded() {
    let (app, sim, _) = kv_app();
    let prompt = words(64, 1000);
    // Miss, then an L0 hit.
    cached(&app, &prompt, &[]).await;
    assert!(cached(&app, &prompt, &[]).await > 0);
    // Demotions and L1 hits (promotions), then a recompute of an unrelated prompt.
    sim.demote_all();
    assert!(cached(&app, &prompt, &[]).await > 0);
    cached(&app, &words(48, 5000), &[]).await;
    // A used prefetch, a wasted one (evicted from L0 before use) and a rejected one.
    let other = words(64, 9000);
    cached(&app, &other, &[]).await;
    // A hit: the reuse evidence that makes its blocks worth demoting (not dropping).
    assert!(cached(&app, &other, &[]).await > 0);
    sim.demote_all();
    let (s, _) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"prompt": other}),
        &[],
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let (s, _) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"prompt": prompt}),
        &[],
    )
    .await;
    assert_eq!(
        s,
        StatusCode::TOO_MANY_REQUESTS,
        "rejected: the queue is full"
    );
    sim.settle();
    cached(&app, &other, &[]).await;
    let (s, _) = post_json(
        &app,
        "/turbine/v1/kv/prefetch",
        json!({"prompt": prompt}),
        &[],
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    sim.settle();
    sim.demote_all();

    // P6b S-3: `usage.prompt_tokens_details.lossy_cached_tokens` in completion and chat
    // responses (0: these tiers store the L0 bytes), and `x-turbine-kv-lossy` is allow or deny.
    for deny in [&[][..], &[("x-turbine-kv-lossy", "deny")][..]] {
        let (s, v) = post_json(
            &app,
            "/v1/completions",
            json!({"model": kv_sim::MODEL, "prompt": prompt, "max_tokens": 1}),
            deny,
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            v["usage"]["prompt_tokens_details"]["lossy_cached_tokens"], 0,
            "{v}"
        );
    }
    let (s, v) = post_json(
        &app,
        "/v1/chat/completions",
        json!({"model": kv_sim::MODEL, "max_tokens": 1,
               "messages": [{"role": "user", "content": prompt}]}),
        &[("x-turbine-kv-lossy", "allow")],
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v["usage"]["prompt_tokens_details"]["lossy_cached_tokens"], 0,
        "{v}"
    );
    let (s, v) = post_json(
        &app,
        "/v1/completions",
        json!({"model": kv_sim::MODEL, "prompt": prompt, "max_tokens": 1}),
        &[("x-turbine-kv-lossy", "maybe")],
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");

    let (s, _, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
    assert_eq!(s, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    for family in [
        "turbine_kv_blocks{",
        "turbine_kv_bytes{",
        "turbine_kv_lookups_total{",
        "turbine_kv_prefix_cached_tokens_total",
        "turbine_kv_lossy_cached_tokens_total ",
        "turbine_kv_lossy_denied_total ",
        "turbine_kv_prompt_tokens_total",
        "turbine_kv_promotions_total{",
        "turbine_kv_demotions_total{",
        "turbine_kv_evictions_total{",
        "turbine_kv_drops_total{",
        "turbine_kv_recompute_tokens_total{",
        "turbine_kv_plans_total{",
        "turbine_kv_transfer_seconds_bucket{",
        "turbine_kv_transfer_bytes_total{",
        "turbine_kv_transfer_bandwidth_bytes_per_second{",
        "turbine_kv_prefetch_total{",
        "turbine_kv_sessions ",
        "turbine_kv_tier_degraded{",
        "turbine_storage_queue_depth ",
        "turbine_storage_latency_seconds_bucket{",
        // P6b S-6: the compression ladder (its actions render once the ladder acts).
        "turbine_kv_ladder_rung{",
        "turbine_kv_ladder_actions_total{",
    ] {
        assert!(text.contains(family), "{family} missing");
    }
    let value = |series: &str| -> f64 {
        text.lines()
            .find_map(|l| l.strip_prefix(series)?.trim().parse().ok())
            .unwrap_or_else(|| panic!("{series} missing"))
    };
    assert!(value(r#"turbine_kv_lookups_total{result="l0"}"#) > 0.0);
    assert!(value(r#"turbine_kv_lookups_total{result="l1"}"#) > 0.0);
    assert!(value(r#"turbine_kv_lookups_total{result="miss"}"#) > 0.0);
    assert!(value(r#"turbine_kv_demotions_total{from="l0",to="l1"}"#) > 0.0);
    assert!(value(r#"turbine_kv_promotions_total{from="l1",to="l0"}"#) > 0.0);
    assert!(value(r#"turbine_kv_evictions_total{tier="l0",reason="pressure"}"#) > 0.0);
    assert!(value(r#"turbine_kv_recompute_tokens_total{reason="no_match"}"#) > 0.0);
    assert!(value(r#"turbine_kv_prefetch_total{outcome="used"}"#) > 0.0);
    assert!(value(r#"turbine_kv_prefetch_total{outcome="wasted"}"#) > 0.0);
    assert!(value(r#"turbine_kv_prefetch_total{outcome="rejected"}"#) > 0.0);
    for line in text
        .lines()
        .filter(|l| l.starts_with("turbine_kv_") || l.starts_with("turbine_storage_"))
    {
        check_labels(line);
    }
}
/// The novanas pair as `turbine_device::topology::TopologyGraph` serialises it.
fn two_gpu_graph() -> Value {
    let gpu = |i: u32, bdf: &str| {
        json!({"id": format!("gpu{i}"), "kind": "gpu", "device_index": i, "vendor": "amd",
               "arch": "gfx1201", "pci_bus_id": bdf, "numa": 0, "numa_source": "nominal"})
    };
    json!({
        "node": {"hostname": "novanas", "captured_at": "2026-09-25T12:00:00Z"},
        "vertices": [
            {"id": "numa0", "kind": "numa", "node": 0, "cpus": "0-31",
             "memory_bytes": 137_438_953_472u64, "distances": [10]},
            gpu(0, "0000:03:00.0"),
            gpu(1, "0000:07:00.0"),
        ],
        "edges": [{"a": "gpu0", "b": "gpu1", "kind": "pcie", "path": "sys", "hops": 2,
                   "link_gts": 32.0, "width": 16, "p2p": "disabled",
                   "vendor_interconnect": null, "rdma": null, "source": "vendor"}],
    })
}

struct TopologyDiagnostics(Value);
impl Diagnostics for TopologyDiagnostics {
    fn status(&self) -> Value {
        json!({})
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
    fn topology(&self, scope: TopologyScope) -> Result<Value, ApiError> {
        assert_eq!(scope, TopologyScope::Node);
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn topology_route() {
    let graph = two_gpu_graph();
    let app = router(ApiState {
        inference: Arc::new(NoModel),
        diagnostics: Arc::new(TopologyDiagnostics(graph.clone())),
        readiness: Arc::new(NotReady),
        metrics: MetricsRegistry::new(),
        limits: ApiLimits {
            max_request_bytes: 1 << 20,
        },
    });
    for path in ["/turbine/v1/topology", "/turbine/v1/topology?scope=node"] {
        let (s, b) = get_json(&app, "GET", path).await;
        assert_eq!(s, StatusCode::OK, "{path}");
        assert_eq!(b["vertices"], graph["vertices"], "{path}");
        assert_eq!(b["edges"], graph["edges"], "{path}");
        assert_eq!(b, graph, "{path}");
    }
    // Cluster scope arrives with phase 6.
    let (s, b) = get_json(&app, "GET", "/turbine/v1/topology?scope=cluster").await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&b, "invalid_request_error", "unsupported_parameter");

    // A diagnostics source without a captured graph answers 501, like the other documents.
    let (s, b) = get_json(&app_default(), "GET", "/turbine/v1/topology").await;
    assert_eq!(s, StatusCode::NOT_IMPLEMENTED);
    assert_error(&b, "not_implemented", "not_implemented");
}

fn app_default() -> Router {
    app(1 << 20)
}

/// P5 multi-GPU pressure views (Task 16, pressure part; the documents come from Task 14).
mod p5 {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use serde_json::{Value, json};
    use turbine_api::{ApiError, ApiLimits, ApiState, Diagnostics, router};
    use turbine_core::types::{CircuitState, DeviceId, MemoryKind, PressureState};
    use turbine_observability::MetricsRegistry;
    use turbine_reliability::budget::{DeviceBudget, PoolKind};
    use turbine_reliability::document::{GroupDoc, ReplicaDoc, device_doc};
    use turbine_reliability::metrics::ReliabilityMetrics;
    use turbine_reliability::multi_device::GroupState;

    use super::{NoModel, NotReady, get_json, send};

    const GIB: u64 = 1 << 30;

    /// Serves a fixed pressure document.
    struct MultiDevice(Value);
    impl Diagnostics for MultiDevice {
        fn status(&self) -> Value {
            json!({})
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
            Ok(self.0.clone())
        }
    }

    fn budget(device: u32) -> DeviceBudget {
        DeviceBudget {
            device: DeviceId(device),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 30 * GIB,
            pools: vec![
                (PoolKind::Weights, 3 * GIB),
                (PoolKind::Kv, 22 * GIB),
                (PoolKind::Workspace, GIB),
                (PoolKind::Collective, GIB),
                (PoolKind::Runtime, GIB),
                (PoolKind::Reserve, 2 * GIB),
            ],
        }
    }

    /// Two devices of one TP group (device 1 ORANGE): the pressure document carries the
    /// `devices`, `groups` (limited by device 1) and `replicas` views, and `/metrics` the
    /// per-device budget by component and the group gauges.
    #[tokio::test]
    async fn pressure_multi_device_view() {
        let (b0, b1) = (budget(0), budget(1));
        let members = [
            (DeviceId(0), PressureState::Green),
            (DeviceId(1), PressureState::Orange),
        ];
        let doc = json!({
            "devices": [device_doc(&b0, members[0].1), device_doc(&b1, members[1].1)],
            "groups": [GroupDoc::of(0, &members)],
            "replicas": [ReplicaDoc::of(0, GroupState::of(&members).state, CircuitState::Healthy)],
        });
        let metrics = MetricsRegistry::new();
        let reliability = ReliabilityMetrics::register(&metrics);
        reliability.record_device_budget(&b0);
        reliability.record_device_budget(&b1);
        reliability.record_group(0, GroupState::of(&members));
        let app = router(ApiState {
            inference: Arc::new(NoModel),
            diagnostics: Arc::new(MultiDevice(doc)),
            readiness: Arc::new(NotReady),
            metrics,
            limits: ApiLimits {
                max_request_bytes: 1 << 20,
            },
        });

        let (s, d) = get_json(&app, "GET", "/turbine/v1/pressure").await;
        assert_eq!(s, StatusCode::OK, "{d}");
        let devices = d["devices"].as_array().expect("devices[]");
        assert_eq!(devices.len(), 2, "{d}");
        for (i, dev) in devices.iter().enumerate() {
            assert_eq!(dev["device"], i, "{d}");
            for component in [
                "weights",
                "kv",
                "workspace",
                "collective",
                "runtime",
                "reserve",
            ] {
                assert!(dev["budget"][component].is_u64(), "{component}: {d}");
            }
        }
        assert_eq!(devices[1]["state"], "ORANGE");
        assert_eq!(devices[0]["budget"]["collective"], GIB);
        assert_eq!(
            d["groups"],
            json!([{"replica": 0, "state": "ORANGE", "limiting_device": 1}])
        );
        assert_eq!(
            d["replicas"],
            json!([{"replica": 0, "eligible": false, "reason": "pressure"}])
        );

        let (s, _, body) = send(&app, "GET", "/metrics", Vec::new(), None).await;
        assert_eq!(s, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();
        for line in [
            format!(r#"turbine_device_budget_bytes{{device="0",component="collective"}} {GIB}"#),
            format!(
                r#"turbine_device_budget_bytes{{device="1",component="kv"}} {}"#,
                22 * GIB
            ),
            r#"turbine_group_pressure_state{replica="0"} 2"#.to_string(),
            r#"turbine_group_limiting_device{replica="0"} 1"#.to_string(),
        ] {
            assert!(text.contains(&line), "{line} missing:\n{text}");
        }
    }
}

/// Not ready for a fixed reason.
struct NotReadyFor(NotReadyReason);
impl Readiness for NotReadyFor {
    fn ready(&self) -> ReadyState {
        ReadyState::NotReady { reason: self.0 }
    }
}

/// P5 (T16, vocabulary part): `/ready` renders the multi-GPU reasons, and the two new error
/// codes carry the contract's status and type.
#[tokio::test]
async fn phase5_ready_reasons_and_error_codes() {
    for (reason, text) in [
        (NotReadyReason::CollectiveInit, "collective_init"),
        (NotReadyReason::LoadingWeights, "loading_weights"),
        (NotReadyReason::RankMissing, "rank_missing"),
    ] {
        let app = router(ApiState {
            inference: Arc::new(NoModel),
            diagnostics: Arc::new(Phase0Diagnostics),
            readiness: Arc::new(NotReadyFor(reason)),
            metrics: MetricsRegistry::new(),
            limits: ApiLimits {
                max_request_bytes: 1 << 20,
            },
        });
        let (s, b) = get_json(&app, "GET", "/ready").await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{text}");
        assert_eq!(b, json!({"ready": false, "reason": text}));
    }
    for (e, kind, code) in [
        (ApiError::not_leader(), "service_unavailable", "not_leader"),
        (
            ApiError::replica_failed("replica 1: all_reduce timed out"),
            "server_error",
            "replica_failed",
        ),
    ] {
        assert_eq!(e.status, StatusCode::SERVICE_UNAVAILABLE, "{code}");
        assert_error(&e.to_json(), kind, code);
    }
}
