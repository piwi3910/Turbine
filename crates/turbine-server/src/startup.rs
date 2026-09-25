//! Startup order (Phase 0): config (exit 2) → tracing → device discovery (exit 1 on explicit
//! library failure) → bind (exit 1) → serve until SIGINT/SIGTERM (exit 0).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde_json::Value;
use turbine_api::{
    ApiError, ApiLimits, ApiState, Diagnostics, InferenceBackend, ModelCard, NotReadyReason,
    Readiness, ReadyState,
};
use turbine_core::config::{self, Config};
use turbine_device::{DeviceInventory, DeviceMetrics, DiscoveryOptions};
use turbine_observability::MetricsRegistry;

use crate::cli::Cli;
use crate::exit::ExitCode;

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
    runtime.block_on(serve(config, inventory))
}

async fn serve(config: Config, inventory: DeviceInventory) -> ExitCode {
    let addr: SocketAddr = config.server.listen;
    let metrics = MetricsRegistry::new();
    DeviceMetrics::register(&metrics).record(&inventory);
    let state = ApiState {
        inference: Arc::new(NoModel),
        diagnostics: Arc::new(ServerDiagnostics::new(&inventory)),
        readiness: Arc::new(NoModel),
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
    tracing::info!(%addr, devices = inventory.devices.len(), "listening");
    let served = axum::serve(listener, turbine_api::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await;
    match served {
        Ok(()) => {
            tracing::info!("shutdown complete");
            ExitCode::Clean
        }
        Err(e) => {
            tracing::error!(error = %e, "server error");
            eprintln!("turbine-server: server error: {e}");
            ExitCode::Startup
        }
    }
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
        () = ctrl_c => tracing::info!(signal = "SIGINT", "shutdown requested; finishing in-flight requests"),
        () = terminate => tracing::info!(signal = "SIGTERM", "shutdown requested; finishing in-flight requests"),
    }
}

/// Phase 0 engine stand-in: no model, never ready.
struct NoModel;

impl InferenceBackend for NoModel {
    fn models(&self) -> Vec<ModelCard> {
        Vec::new()
    }
}

impl Readiness for NoModel {
    fn ready(&self) -> ReadyState {
        ReadyState::NotReady {
            reason: NotReadyReason::NoModelLoaded,
        }
    }
}

/// `GET /turbine/v1/status` document.
#[derive(Serialize)]
struct StatusDocument {
    version: &'static str,
    uptime_seconds: u64,
    ready: bool,
    device_count: u64,
}

struct ServerDiagnostics {
    started: Instant,
    device_count: u64,
    devices: Value,
}

impl ServerDiagnostics {
    fn new(inventory: &DeviceInventory) -> Self {
        ServerDiagnostics {
            started: Instant::now(),
            device_count: inventory.devices.len() as u64,
            devices: serde_json::to_value(inventory).unwrap_or(Value::Null),
        }
    }
}

impl Diagnostics for ServerDiagnostics {
    fn status(&self) -> Value {
        serde_json::to_value(StatusDocument {
            version: env!("CARGO_PKG_VERSION"),
            uptime_seconds: self.started.elapsed().as_secs(),
            ready: false,
            device_count: self.device_count,
        })
        .unwrap_or(Value::Null)
    }
    fn devices(&self) -> Value {
        self.devices.clone()
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
