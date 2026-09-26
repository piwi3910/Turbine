//! Startup order (P1 §Interfaces, contract §16.3): config (exit 2) → tracing → device discovery →
//! kernel provider → model config, tokenizer, template → kernel registry → memory budget (each
//! exit 1, nothing bound yet) → bind (`/health` 200, `/ready` 503 `loading_model`; exit 1) →
//! weight load and one-token warm-up on the generation thread → `/ready` 200 → serve until
//! SIGINT/SIGTERM (exit 0).
//!
//! A load or warm-up failure after binding keeps `/ready` at 503 `model_load_failed` for
//! [`FAILURE_GRACE`] and exits 1; so do three consecutive failed requests (`device_error`, C-25).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;
use turbine_api::{ApiLimits, ApiState};
use turbine_core::config::{self, Config};
use turbine_device::{DeviceInventory, DeviceMetrics, DiscoveryOptions};
use turbine_model::ModelMetrics;
use turbine_model::executor::SequenceKv;
use turbine_observability::MetricsRegistry;

use crate::cli::Cli;
use crate::exit::ExitCode;
use crate::generation::{Engine, Fatal, Job, ModelBackend};
use crate::metrics::ServerMetrics;
use crate::model::{self, PreparedModel};

/// How long `/ready` reports the failure before the process exits 1 (P1: at most 1 s).
const FAILURE_GRACE: Duration = Duration::from_millis(500);

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
    let server_metrics = ServerMetrics::register(&metrics);
    let model_metrics = ModelMetrics::register(&metrics);
    let backend = Arc::new(ModelBackend::new(&prepared, &inventory, server_metrics));
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
    if let Err(e) = spawn_engine(
        prepared,
        Arc::clone(&backend),
        model_metrics,
        fatal_tx.clone(),
    ) {
        let _ = fatal_tx.send(Fatal::LoadFailed(format!(
            "cannot start the generation thread: {e}"
        )));
    }

    let server =
        axum::serve(listener, turbine_api::router(state)).with_graceful_shutdown(shutdown_signal());
    tokio::select! {
        served = async { server.await } => match served {
            Ok(()) => {
                tracing::info!("shutdown complete");
                ExitCode::Clean
            }
            Err(e) => {
                tracing::error!(error = %e, "server error");
                eprintln!("turbine-server: server error: {e}");
                ExitCode::Startup
            }
        },
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

/// Starts the generation thread: it loads the weights, warms up, marks the backend ready and
/// then serves jobs. Failures are reported on `fatal`.
fn spawn_engine(
    prepared: PreparedModel,
    backend: Arc<ModelBackend>,
    model_metrics: ModelMetrics,
    fatal: UnboundedSender<Fatal>,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("turbine-generation".into())
        .spawn(move || {
            let PreparedModel {
                provider,
                arch,
                generation,
                tokenizer,
                index,
                registry,
                max_seq_len,
                block_tokens,
                ..
            } = prepared;
            let warmup_token = generation.bos_token_id.unwrap_or(0);
            let kv = match SequenceKv::new(&provider.mem, arch.kv_layout(block_tokens), max_seq_len)
            {
                Ok(kv) => kv,
                Err(e) => {
                    let _ = fatal.send(Fatal::LoadFailed(format!("KV allocation: {e}")));
                    return;
                }
            };
            let loaded = match model::load(
                &arch,
                &index,
                registry,
                provider.mem,
                kv,
                warmup_token,
                &model_metrics,
            ) {
                Ok(l) => l,
                Err(e) => {
                    let _ = fatal.send(Fatal::LoadFailed(e.to_string()));
                    return;
                }
            };
            drop(index);
            let mut executor = loaded.executor;
            let mut kv = loaded.kv;
            // Capacity 1: the slot admits one request, so at most one job is ever queued.
            let (jobs_tx, jobs_rx) = std::sync::mpsc::sync_channel::<Job>(1);
            backend.set_ready(jobs_tx, loaded.load_seconds, loaded.weight_bytes);
            tracing::info!("ready");
            let engine = Engine {
                tokenizer,
                max_seq_len,
                slot: backend.slot(),
                metrics: backend.metrics().clone(),
                model_metrics,
            };
            drop(backend);
            if let Err(message) = engine.run(&mut executor, &mut kv, &jobs_rx) {
                let _ = fatal.send(Fatal::DeviceError(message));
            }
        })
        .map(|_| ())
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
