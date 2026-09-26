//! Startup order (P1 §Interfaces, contract §16.3): config (exit 2) → tracing → device discovery →
//! kernel provider → model config, tokenizer, template → kernel registry → memory budget with
//! the KV pool and the batch workspace (each exit 1, nothing bound yet) → bind (`/health` 200,
//! `/ready` 503 `loading_model`; exit 1) → weight load, KV pool allocation and one-token warm-up
//! on the engine thread → `/ready` 200 → serve until SIGINT/SIGTERM (exit 0).
//!
//! Shutdown (P2 S-13): on SIGINT/SIGTERM `/ready` and new requests answer 503 `shutting_down`
//! while the listener stays open; running requests continue until none is left or
//! `server.shutdown_grace` has passed; then `EngineCommand::Shutdown` cancels the rest with
//! reason `shutdown` (their streams end with `shutting_down`), the engine delivers what it holds
//! and stops, the listener closes, open connections finish (bounded by [`CLOSE_LIMIT`]) and the
//! process exits 0.
//!
//! A load or warm-up failure after binding keeps `/ready` at 503 `model_load_failed` for
//! [`FAILURE_GRACE`] and exits 1; so do three consecutive failed iterations or an engine panic
//! (`device_error`, C-25).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use turbine_api::{ApiLimits, ApiState};
use turbine_core::config::{self, Config};
use turbine_device::{DeviceInventory, DeviceMetrics, DiscoveryOptions};
use turbine_kv::KvMetrics;
use turbine_model::ModelMetrics;
use turbine_observability::MetricsRegistry;
use turbine_scheduler::SchedulerMetrics;

use crate::backend::ModelBackend;
use crate::cli::Cli;
use crate::engine::{self, EngineMetrics, Fatal, Timeouts};
use crate::exit::ExitCode;
use crate::metrics::ServerMetrics;
use crate::model::{self, PreparedModel};

/// How long `/ready` reports the failure before the process exits 1 (P1: at most 1 s).
const FAILURE_GRACE: Duration = Duration::from_millis(500);
/// How often the shutdown sequence looks at the engine.
const SHUTDOWN_POLL: Duration = Duration::from_millis(10);
/// After the grace: how long the engine may take to deliver the cancellations and stop, and
/// then how long open connections may take to finish, before the process exits 0 regardless.
const CLOSE_LIMIT: Duration = Duration::from_millis(500);

pub fn run(cli: Cli) -> ExitCode {
    let config = match config::load(&cli.config, &cli.set) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("turbine-server: invalid configuration: {e}");
            return ExitCode::Config;
        }
    };
    if cli.check_config {
        println!("config ok");
        return ExitCode::Clean;
    }
    if let Err(e) = turbine_observability::init_tracing(&config.logging) {
        eprintln!("turbine-server: {e}");
        return ExitCode::Startup;
    }
    tracing::info!(config = %cli.config.display(), listen = %config.server.listen, "configuration loaded");

    let inventory = match turbine_device::discover(&DiscoveryOptions::from_config(&config.devices))
    {
        Ok(inv) => inv,
        Err(e) => {
            tracing::error!(error = %e, "device discovery failed");
            eprintln!("turbine-server: device discovery failed: {e}");
            return ExitCode::Startup;
        }
    };

    let metrics = MetricsRegistry::new();
    DeviceMetrics::register(&metrics).record(&inventory);
    let prepared = match model::prepare(&config, &inventory, &metrics) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "model startup failed");
            eprintln!("turbine-server: {e}");
            return ExitCode::Startup;
        }
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("turbine-server: cannot start the async runtime: {e}");
            return ExitCode::Startup;
        }
    };
    let code = runtime.block_on(serve(config, inventory, metrics, prepared));
    // Never wait for the generation thread or in-flight blocking work on the way out.
    runtime.shutdown_background();
    code
}

async fn serve(
    config: Config,
    inventory: DeviceInventory,
    metrics: MetricsRegistry,
    prepared: PreparedModel,
) -> ExitCode {
    let addr: SocketAddr = config.server.listen;
    let engine_metrics = EngineMetrics {
        server: ServerMetrics::register(&metrics),
        model: ModelMetrics::register(&metrics),
        scheduler: SchedulerMetrics::register(&metrics),
        kv: KvMetrics::register(&metrics),
    };
    let backend = Arc::new(ModelBackend::new(
        &prepared,
        &inventory,
        engine_metrics.server.clone(),
    ));
    let state = ApiState {
        inference: backend.clone(),
        diagnostics: backend.clone(),
        readiness: backend.clone(),
        metrics,
        limits: ApiLimits {
            max_request_bytes: usize::try_from(config.server.max_request_bytes.0)
                .unwrap_or(usize::MAX),
        },
    };
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, error = %e, "cannot bind listener");
            eprintln!("turbine-server: cannot bind {addr}: {e}");
            return ExitCode::Startup;
        }
    };
    tracing::info!(%addr, devices = inventory.devices.len(), "listening; loading the model");

    let (fatal_tx, mut fatal_rx) = tokio::sync::mpsc::unbounded_channel();
    let queue_capacity = config.scheduler.max_queued_requests as usize;
    if let Err(e) = engine::spawn(
        prepared,
        Arc::clone(&backend),
        engine_metrics,
        queue_capacity,
        Timeouts::from_config(&config.server),
        fatal_tx.clone(),
    ) {
        let _ = fatal_tx.send(Fatal::LoadFailed(format!(
            "cannot start the engine thread: {e}"
        )));
    }

    let (drained_tx, drained_rx) = tokio::sync::oneshot::channel();
    let shutdown = drain_on_signal(
        Arc::clone(&backend),
        config.server.shutdown_grace.0,
        drained_tx,
    );
    let server = axum::serve(listener, turbine_api::router(state)).with_graceful_shutdown(shutdown);
    let connections_closed = async {
        match drained_rx.await {
            Ok(()) => tokio::time::sleep(CLOSE_LIMIT).await,
            Err(_) => std::future::pending().await,
        }
    };
    tokio::select! {
        served = async { server.await } => match served {
            Ok(()) => {
                // Every connection has closed; the engine cancels anything left and stops.
                backend.stop_engine();
                tracing::info!("shutdown complete");
                ExitCode::Clean
            }
            Err(e) => {
                tracing::error!(error = %e, "server error");
                eprintln!("turbine-server: server error: {e}");
                ExitCode::Startup
            }
        },
        () = connections_closed => {
            tracing::warn!(event = "shutdown_connections_open", "connections still open after shutdown; exiting");
            ExitCode::Clean
        }
        Some(fatal) = fatal_rx.recv() => {
            backend.set_failed(&fatal);
            let message = match &fatal {
                Fatal::LoadFailed(m) => format!("model load failed: {m}"),
                Fatal::DeviceError(m) => format!("device error: {m}"),
            };
            tracing::error!(error = %message, "exiting");
            eprintln!("turbine-server: {message}");
            tokio::time::sleep(FAILURE_GRACE).await;
            ExitCode::Startup
        }
    }
}

/// Resolves when the listener should close: after a signal, the drain and the engine stop
/// (module comment). `drained` fires at that point.
async fn drain_on_signal(
    backend: Arc<ModelBackend>,
    grace: Duration,
    drained: tokio::sync::oneshot::Sender<()>,
) {
    shutdown_signal().await;
    backend.begin_shutdown();
    let draining = tokio::time::Instant::now();
    while !backend.engine_idle() && draining.elapsed() < grace {
        tokio::time::sleep(SHUTDOWN_POLL).await;
    }
    let left = !backend.engine_idle();
    tracing::info!(
        event = "shutdown_drained",
        drained_seconds = draining.elapsed().as_secs_f64(),
        cancelling = left,
        "shutdown grace over; stopping the engine"
    );
    backend.stop_engine();
    let stopping = tokio::time::Instant::now();
    while !backend.engine_stopped() && stopping.elapsed() < CLOSE_LIMIT {
        tokio::time::sleep(SHUTDOWN_POLL).await;
    }
    let _ = drained.send(());
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "cannot install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => tracing::info!(signal = "SIGINT", "shutdown requested; draining running requests"),
        () = terminate => tracing::info!(signal = "SIGTERM", "shutdown requested; draining running requests"),
    }
}
