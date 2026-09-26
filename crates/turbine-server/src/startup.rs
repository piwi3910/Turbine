//! Startup order (P1 §Interfaces, contract §16.3): config and module names (exit 2) →
//! support-matrix row with the device arch unknown (exit 2 when unsupported; also under
//! `--check-config`) → tracing → device discovery → support-matrix row with the device arch
//! (exit 2 when unsupported; `event="support_matrix"`, WARN when experimental) →
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

use axum::serve::ListenerExt;
use turbine_api::support::SupportMetrics;
use turbine_api::{ApiLimits, ApiState};
use turbine_core::config::{self, Config, ConfigError};
use turbine_core::support::SupportRowView;
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
use crate::modules::known_module_names;
use crate::{support_matrix, support_startup};

/// How long `/ready` reports the failure before the process exits 1 (P1: at most 1 s).
const FAILURE_GRACE: Duration = Duration::from_millis(500);
/// How often the shutdown sequence looks at the engine.
const SHUTDOWN_POLL: Duration = Duration::from_millis(10);
/// After the grace: how long the engine may take to deliver the cancellations and stop, and
/// then how long open connections may take to finish, before the process exits 0 regardless.
const CLOSE_LIMIT: Duration = Duration::from_millis(500);
/// Test-only, not a configuration key: a positive byte count caps every accepted connection's
/// kernel send buffer (`SO_SNDBUF`; Linux doubles it and turns its autotuning off). The request
/// output channel (P2 S-7) pauses a request once the client stops reading and everything
/// between the two is full; with the kernel default that is megabytes on Linux loopback, so the
/// black-box tests set this to make the pause come after a few KiB on every OS.
const SEND_BUFFER_ENV: &str = "TURBINE_TEST_SOCKET_SEND_BUFFER";

/// Parses [`SEND_BUFFER_ENV`]: unset → `None` (kernel default), else a positive byte count.
fn send_buffer_cap(value: Option<&str>) -> Result<Option<usize>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.parse::<usize>() {
        Ok(bytes) if bytes > 0 => Ok(Some(bytes)),
        _ => Err(format!(
            "{SEND_BUFFER_ENV} must be a positive byte count, got {value:?}"
        )),
    }
}

pub fn run(cli: Cli) -> ExitCode {
    // clap requires --config unless --support-matrix, which `main` handles before this.
    let Some(config_path) = cli.config.as_deref() else {
        eprintln!("turbine-server: --config is required");
        return ExitCode::Config;
    };
    // Module names are checked against the registries with the rest of the configuration:
    // exit 2 before device discovery and before binding, also under --check-config.
    let config = match config::load(config_path, &cli.set)
        .and_then(|c| c.validate_modules(&known_module_names()).map(|()| c))
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("turbine-server: invalid configuration: {e}");
            return ExitCode::Config;
        }
    };
    // The support-matrix row before discovery (device arch unknown): an unsupported row is a
    // configuration error, exit 2, also under --check-config.
    let support = support_startup::before_discovery(&config);
    if cli.check_config {
        return match support {
            Ok(decision) => {
                println!("{}", support_matrix::check_config_line(&decision));
                println!("config ok");
                ExitCode::Clean
            }
            Err(e) => {
                eprintln!("turbine-server: invalid configuration: {e}");
                ExitCode::Config
            }
        };
    }
    if let Err(e) = turbine_observability::init_tracing(&config.logging) {
        eprintln!("turbine-server: {e}");
        return ExitCode::Startup;
    }
    tracing::info!(config = %config_path.display(), listen = %config.server.listen, "configuration loaded");
    let support = match support {
        Ok(decision) => decision,
        Err(e) => return refuse_support(&e),
    };

    let inventory = match turbine_device::discover(&DiscoveryOptions::from_config(&config.devices))
    {
        Ok(inv) => inv,
        Err(e) => {
            tracing::error!(error = %e, "device discovery failed");
            eprintln!("turbine-server: device discovery failed: {e}");
            return ExitCode::Startup;
        }
    };
    // Again with the discovered device architecture, before the kernel library and the model.
    let support = match support_startup::after_discovery(&config, &inventory, support) {
        Ok(decision) => decision,
        Err(e) => return refuse_support(&e),
    };
    support_startup::log(&support);

    let metrics = MetricsRegistry::new();
    DeviceMetrics::register(&metrics).record(&inventory);
    SupportMetrics::register(&metrics).set(&support.status);
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
    let code = runtime.block_on(serve(config, inventory, metrics, prepared, support.view()));
    // Never wait for the generation thread or in-flight blocking work on the way out.
    runtime.shutdown_background();
    code
}

/// An unsupported support-matrix row: logged as `event="support_matrix"`, exit 2 (a
/// configuration error, before any port is bound).
fn refuse_support(error: &ConfigError) -> ExitCode {
    support_startup::log_refusal(error);
    eprintln!("turbine-server: invalid configuration: {error}");
    ExitCode::Config
}

async fn serve(
    config: Config,
    inventory: DeviceInventory,
    metrics: MetricsRegistry,
    prepared: PreparedModel,
    support: SupportRowView,
) -> ExitCode {
    let addr: SocketAddr = config.server.listen;
    let engine_metrics = EngineMetrics {
        server: ServerMetrics::register(&metrics),
        model: ModelMetrics::register(&metrics),
        scheduler: SchedulerMetrics::register(&metrics),
        kv: KvMetrics::register(&metrics),
    };
    let backend =
        Arc::new(ModelBackend::new(&prepared, &inventory, &engine_metrics).with_support(support));
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
    let send_buffer = match send_buffer_cap(std::env::var(SEND_BUFFER_ENV).ok().as_deref()) {
        Ok(cap) => cap,
        Err(e) => {
            eprintln!("turbine-server: {e}");
            return ExitCode::Startup;
        }
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
    let listener = listener.tap_io(move |conn: &mut tokio::net::TcpStream| {
        if let Some(bytes) = send_buffer
            && let Err(e) = socket2::SockRef::from(&*conn).set_send_buffer_size(bytes)
        {
            tracing::warn!(error = %e, bytes, "cannot cap the connection's send buffer");
        }
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset leaves the kernel's autotuning alone; a positive byte count caps; anything else is
    /// refused rather than silently ignored (a typo would make the held-stream tests flaky).
    #[test]
    fn send_buffer_cap_parsing() {
        assert_eq!(send_buffer_cap(None), Ok(None));
        assert_eq!(send_buffer_cap(Some("4096")), Ok(Some(4096)));
        for bad in ["", "0", "-1", "4KiB", "x"] {
            let err = send_buffer_cap(Some(bad)).unwrap_err();
            assert!(err.contains(SEND_BUFFER_ENV), "{bad:?}: {err}");
        }
    }
}
