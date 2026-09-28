//! Startup order (P1 §Interfaces, contract §16.3): config and module names (exit 2) →
//! support-matrix row with the device arch unknown (exit 2 when unsupported; also under
//! `--check-config`) → tracing → device discovery → support-matrix row with the device arch
//! (exit 2 when unsupported; `event="support_matrix"`, WARN when experimental) → the node
//! topology graph (P5 S-1, never fails; `GET /turbine/v1/topology`) → the parallel plan (P5 S-4,
//! `crate::parallel`; exit 2, or exit 1 when the model config it needs is unreadable) → the P4 `kv`
//! host rules (`kv.cpu.max_bytes` against MemTotal minus `reliability.memory.host_reserve_bytes`
//! exit 2, `kv.nvme.max_bytes` against free disk exit 1) → kernel provider → model config,
//! tokenizer, template → kernel registry → pre-load memory budget (P3 S-2: the KV pool, the
//! batch workspace and the emergency reserve; exit 1 naming every pool, nothing bound yet) →
//! the `kv` block-size rules (exit 2) and the L2 tier (`kv.nvme.path` created, wiped of old
//! slab files and checked writable; exit 1) → bind (`/health` 200, `/ready` 503
//! `loading_model`; exit 1) → on the engine thread: weight load, the budget re-measured, the
//! reservation ledger, KV pool, the KV hierarchy (L1 tier and transfer calibration), emergency
//! reserve and one-token warm-up, then the telemetry sampler and the pressure controller →
//! `/ready` 200 → serve until SIGINT/SIGTERM (exit 0).
//!
//! Shutdown (P2 S-13): on SIGINT/SIGTERM `/ready` and new requests answer 503 `shutting_down`
//! while the listener stays open; running requests continue until none is left or
//! `server.shutdown_grace` has passed; then `EngineCommand::Shutdown` cancels the rest with
//! reason `shutdown` (their streams end with `shutting_down`), the engine delivers what it holds
//! and stops, the listener closes, open connections finish (bounded by [`CLOSE_LIMIT`]) and the
//! process exits 0.
//!
//! A load or warm-up failure after binding keeps `/ready` at 503 `model_load_failed` for
//! [`FAILURE_GRACE`] and exits 1. A fatal circuit (P3 S-12: a sticky device error, a pressure
//! controller failure or an engine panic) keeps `/ready` at 503 `circuit_open` for
//! [`FAILURE_GRACE`] and exits 3 (the Phase 2 exit after three failed iterations is retired,
//! C-25).

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::serve::ListenerExt;
use turbine_api::support::SupportMetrics;
use turbine_api::{ApiLimits, ApiState};
use turbine_core::clock::SystemClock;
use turbine_core::config::{self, ByteSize, Config, ConfigError, KvConfig, RankMode};
use turbine_core::support::SupportRowView;
use turbine_core::types::DeviceId;
use turbine_device::telemetry::TelemetryMetrics;
use turbine_device::topology::{RegisteredTopology, TopologyGraph, discover_topology};
use turbine_device::{DeviceInventory, DeviceMetrics, DiscoveryOptions};
use turbine_distributed::collective::CollectiveMetrics;
use turbine_kv::KvMetrics;
use turbine_kv::tier::L2NvmeTier;
use turbine_model::ModelMetrics;
use turbine_model::ep::{EpAttention, EpShard};
use turbine_model::tp::ShardSpec;
use turbine_observability::MetricsRegistry;
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_scheduler::SchedulerMetrics;

use crate::backend::ModelBackend;
use crate::cli::Cli;
use crate::engine::{self, EngineMetrics, Fatal, KvSetup, ReliabilityStartup, Timeouts};
use crate::exit::ExitCode;
use crate::host;
use crate::kv_orchestrator::{self, tp_kv_format};
use crate::metrics::ServerMetrics;
use crate::model::{self, EpRank, PreparedModel, RankPart};
use crate::modules::known_module_names;
use crate::parallel::{self, PlanFailure};
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
    let mut config = match config::load(config_path, &cli.set)
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
    // P5 S-1: the node-local topology graph, right after discovery (never fails).
    let mut topology =
        capture_topology(&DiscoveryOptions::from_config(&config.devices), &inventory);
    // P5 S-13: measured host links on the GPU edges (multi-GPU configurations only).
    parallel::measure_host_links(&config, &inventory, &mut topology);

    // P5 S-4: the parallel plan before the kernel provider and the listener (exit 2, or exit 1
    // when the model config it needs is unreadable).
    let plan = match parallel::plan_for(&config, &inventory, &topology).and_then(|mut p| {
        parallel::check_executable(&mut p, &config)?;
        Ok(p)
    }) {
        Ok(p) => p,
        Err(PlanFailure::Config(message)) => {
            tracing::error!(event = "parallel_plan_failed", error = %message, "invalid parallel plan");
            eprintln!("turbine-server: invalid parallel plan: {message}");
            return ExitCode::Config;
        }
        Err(PlanFailure::Startup(message)) => {
            tracing::error!(error = %message, "parallel planning failed");
            eprintln!("turbine-server: {message}");
            return ExitCode::Startup;
        }
    };
    let device = parallel::engine_device(&plan);
    if device != config.execution.device {
        tracing::info!(
            event = "execution_device_from_plan",
            configured = config.execution.device.0,
            device = device.0,
            "the parallel plan places the engine on another device than execution.device"
        );
        config.execution.device = device;
    }
    // P5 S-5 `static` rank mode: this process is one rank of the group, on its local device.
    // Tier copies need every rank's pool in one process, so L1/L2 are off (WARN, reason code).
    let static_rank =
        (plan.tp > 1 && plan.mode == RankMode::Static).then_some(config.parallel.ranks.rank);
    if static_rank.is_some() {
        if let Some(&local) = config.parallel.ranks.local_devices.first() {
            config.execution.device = local;
        }
        if config.kv.cpu.enabled || config.kv.nvme.enabled {
            tracing::warn!(
                event = "kv_tiers_unavailable",
                reason = "static_ranks",
                "kv.cpu and kv.nvme are off in static rank mode: a tier copy of a block needs \
                 every rank's shard, and the other ranks' pools live in other processes"
            );
            config.kv.cpu.enabled = false;
            config.kv.nvme.enabled = false;
        }
    }

    // P4 host rules (contract §16.3): a `kv.nvme.*` violation is a runtime failure (exit 1),
    // any other an invalid configuration (exit 2); both before anything is bound.
    if let Err(e) = config.validate_host(&host::facts(&config.kv)) {
        let nvme = e.key().is_some_and(|k| k.starts_with("kv.nvme."));
        eprintln!("turbine-server: invalid configuration: {e}");
        return if nvme {
            ExitCode::Startup
        } else {
            ExitCode::Config
        };
    }

    let metrics = MetricsRegistry::new();
    DeviceMetrics::register(&metrics).record(&inventory);
    SupportMetrics::register(&metrics).set(&support.status);
    parallel::register_info(&metrics, &plan);
    parallel::register_links(&metrics, &topology);
    // Steps 4–6 per data-parallel replica (P5 S-7): replica 0 on the plan's first device, the
    // others on theirs, sharing the grammar compiler and the kernel metrics. Replicas that share
    // a device (`parallel.allow_device_sharing`) split its budget. With tensor parallelism
    // (P5 S-6) each replica is a group: every rank is prepared on its device for its shard, and
    // the collective backend loads once (exit 1 naming the library when it cannot). An
    // expert-parallel group (P5 S-11) runs through the same group runtime: every rank holds its
    // experts, fed the same batches as a tensor-parallel group's ranks.
    let tp = plan.tp;
    let group_size = plan.group_size();
    let collective = if plan.replica_size() > 1 {
        match parallel::load_collective(&config, &plan) {
            Ok(library) => Some((library, CollectiveMetrics::register(&metrics))),
            Err(e) => {
                tracing::error!(event = "collective_unavailable", backend = plan.backend, error = %e, "collective backend unavailable");
                eprintln!("turbine-server: {e}");
                return ExitCode::Startup;
            }
        }
    } else {
        None
    };
    // Rank slots of every group on `device`: replicas sharing it split its budget.
    let slots_on = |device: DeviceId| {
        plan.groups
            .iter()
            .flat_map(|g| &g.ranks)
            .filter(|s| s.device == device)
            .count() as u32
    };
    // Expert parallelism: every replica's rank-0 token counts, and the metrics they feed.
    let expert_metrics = (plan.ep > 1).then(|| parallel::ExpertMetrics::register(&metrics));
    let part = |rank: u32, stats: Option<&Arc<parallel::ExpertStats>>| -> Option<RankPart> {
        match (&plan.experts, stats) {
            (Some(placement), Some(stats)) => Some(RankPart::Expert(EpRank {
                shard: EpShard {
                    rank,
                    world: plan.ep,
                    attention: if tp == plan.ep {
                        EpAttention::TensorParallel
                    } else {
                        EpAttention::Replicated
                    },
                },
                placement: Arc::clone(placement),
                // Rank 0's counts are the server's; the other ranks count the same.
                counts: if rank == 0 {
                    Arc::clone(stats.counts())
                } else {
                    Arc::default()
                },
            })),
            _ => (tp > 1).then_some(RankPart::Tensor(ShardSpec { rank, world: tp })),
        }
    };
    let startup_failed = |r: usize, e: &dyn std::fmt::Display| {
        tracing::error!(replica = r, error = %e, "model startup failed");
        eprintln!("turbine-server: {e}");
        ExitCode::Startup
    };
    let kv_metrics = KvMetrics::register(&metrics);
    let pipeline_metrics =
        (plan.pp > 1).then(|| turbine_scheduler::PipelineMetrics::register(&metrics));
    let mut replicas: Vec<ReplicaStart> = Vec::with_capacity(plan.groups.len());
    for (r, group) in plan.groups.iter().enumerate() {
        // Static mode (dp 1): this process prepares only its own rank, on its local device.
        let (device, first_rank) = match static_rank {
            Some(rank) => (config.execution.device, rank),
            None => (parallel::leader_device(&plan, group), 0),
        };
        let cfg = replica_config(&config, r, plan.groups.len(), device);
        let experts = expert_metrics.as_ref().map(|m| {
            Arc::new(parallel::ExpertStats::new(
                plan.ep,
                &config.parallel.expert.placement,
                m.clone(),
            ))
        });
        let stats = experts.as_ref();
        // P5 S-10: a pipeline's stages are prepared on their devices; the engine runs the last.
        let mut pipeline = None;
        let prepared = match (replicas.first(), &collective, &pipeline_metrics) {
            (base, Some(c), Some(pm)) => engine::pp::prepare_stages(
                &cfg,
                &inventory,
                &metrics,
                base.map(|b| &b.prepared),
                &plan,
                group,
                c,
                pm,
            )
            .map(|(last, start)| {
                pipeline = Some(start);
                last
            }),
            (None, _, _) => {
                model::prepare_rank(&cfg, &inventory, &metrics, part(first_rank, stats))
            }
            (Some(base), _, _) => {
                model::prepare_replica(&cfg, &inventory, &base.prepared, part(first_rank, stats))
            }
        };
        let mut prepared = match prepared {
            Ok(p) => p,
            Err(e) => return startup_failed(r, &e),
        };
        if pipeline.is_none()
            && let Err(e) = prepared.share_device(slots_on(device))
        {
            return startup_failed(r, &e);
        }
        let mut workers = Vec::with_capacity(group.ranks.len().saturating_sub(1));
        let local_workers = if static_rank.is_some() || pipeline.is_some() {
            0
        } else {
            group.ranks.len()
        };
        for slot in group.ranks.iter().take(local_workers).skip(1) {
            let mut rank_cfg = cfg.clone();
            rank_cfg.execution.device = slot.device;
            let mut worker = match model::prepare_replica(
                &rank_cfg,
                &inventory,
                &prepared,
                part(slot.rank, stats),
            ) {
                Ok(w) => w,
                Err(e) => return startup_failed(r, &e),
            };
            if let Err(e) = worker.share_device(slots_on(slot.device)) {
                return startup_failed(r, &e);
            }
            workers.push(worker);
        }
        // P4: the model's block size is known now (exit 2), and L2 opens before the listener
        // binds (exit 1 naming the path). A tier copy of a tensor-parallel block holds every
        // rank's shard (decision "P5 T17" B); an expert-parallel block at tp 1 every rank's
        // replica of it (each rank runs the whole attention).
        let format = pipeline.as_ref().map_or_else(
            || tp_kv_format(prepared.pool.layout, group_size),
            |p| p.format,
        );
        if let Err(e) = cfg.kv.validate_block_bytes(format.block_bytes()) {
            eprintln!("turbine-server: invalid configuration: {e}");
            return ExitCode::Config;
        }
        let replica_kv_metrics = if r == 0 {
            kv_metrics.clone()
        } else {
            kv_metrics.another_replica()
        };
        let l2 = match kv_orchestrator::open_l2(
            &cfg.kv,
            &format,
            &prepared.identity,
            Arc::new(SystemClock::new()),
            replica_kv_metrics.clone(),
        ) {
            Ok(l2) => l2,
            Err(e) => {
                tracing::error!(replica = r, error = %e, "KV tier startup failed");
                eprintln!("turbine-server: {e}");
                return ExitCode::Startup;
            }
        };
        let statics = match (static_rank, &collective) {
            (Some(rank), Some((library, cmetrics))) => {
                match static_parts(&config, &inventory, &plan, &prepared, rank) {
                    Ok((transport, expect, hello)) => Some(if rank == 0 {
                        StaticRole::Leader(engine::tp::StaticLeader {
                            transport,
                            listen: config.parallel.ranks.leader.unwrap_or(config.server.listen),
                            expect,
                            world: tp,
                        })
                    } else {
                        StaticRole::Worker(engine::tp::StaticWorker {
                            transport,
                            leader: config.parallel.ranks.leader.unwrap_or(config.server.listen),
                            hello,
                            library: Arc::clone(library),
                            init_timeout: config.parallel.collective.init_timeout.0,
                            op_timeout: config.parallel.collective.op_timeout.0,
                            route_max_bytes: config.parallel.collective.hostmem_max_bytes.fixed(),
                            metrics: cmetrics.clone(),
                            clock: Arc::new(SystemClock::new()),
                        })
                    }),
                    Err(e) => {
                        eprintln!("turbine-server: {e}");
                        return ExitCode::Config;
                    }
                }
            }
            _ => None,
        };
        let (remote, worker) = match statics {
            Some(StaticRole::Leader(l)) => (Some(l), None),
            Some(StaticRole::Worker(w)) => (None, Some(w)),
            None => (None, None),
        };
        let group = collective
            .as_ref()
            .filter(|_| worker.is_none() && pipeline.is_none())
            .map(|(library, metrics)| engine::tp::TpGroupStart {
                workers,
                library: Arc::clone(library),
                init_timeout: config.parallel.collective.init_timeout.0,
                op_timeout: config.parallel.collective.op_timeout.0,
                route_max_bytes: config.parallel.collective.hostmem_max_bytes.fixed(),
                depth: config.parallel.plan_queue_depth as usize,
                metrics: metrics.clone(),
                clock: Arc::new(SystemClock::new()),
                remote,
                experts: experts.clone(),
            });
        replicas.push(ReplicaStart {
            prepared,
            group,
            pipeline,
            experts,
            worker,
            kv_cfg: cfg.kv.clone(),
            kv_metrics: replica_kv_metrics,
            l2,
        });
    }

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
    let code = runtime.block_on(serve(
        config,
        inventory,
        ServeCluster {
            topology,
            parallel: parallel::status(&plan),
        },
        metrics,
        replicas,
        support.view(),
    ));
    // Never wait for the generation thread or in-flight blocking work on the way out.
    runtime.shutdown_background();
    code
}

/// This process's part of a `static` group.
enum StaticRole {
    Leader(engine::tp::StaticLeader),
    Worker(engine::tp::StaticWorker),
}

/// The rank transport (`parallel.ranks.transport`), what the leader expects of every rank and
/// this rank's `Hello` (P5 S-5): the model and configuration fingerprints and the local device's
/// vendor and architecture.
fn static_parts(
    config: &Config,
    inventory: &DeviceInventory,
    plan: &turbine_distributed::plan::ParallelPlan,
    prepared: &PreparedModel,
    rank: u32,
) -> Result<
    (
        &'static dyn turbine_distributed::transport::Transport,
        turbine_distributed::rank::HelloExpect,
        turbine_distributed::rank::RankMessage,
    ),
    String,
> {
    let transport =
        turbine_distributed::transport::select(config.parallel.ranks.transport.as_str())
            .map_err(|e| format!("parallel.ranks.transport: {e}"))?;
    let device = inventory
        .devices
        .iter()
        .find(|d| d.index == config.execution.device);
    let vendor = device
        .map(|d| d.vendor)
        .or(plan.vendor)
        .ok_or("parallel.ranks.mode: static ranks need a GPU device")?;
    let arch = device.and_then(|d| d.arch.clone()).unwrap_or_default();
    let expect = turbine_distributed::rank::HelloExpect {
        model_fingerprint: prepared.identity.fingerprint(),
        config_fingerprint: parallel::config_fingerprint(config),
        device_vendor: vendor,
        device_arch: arch.clone(),
    };
    let hello = turbine_distributed::rank::RankMessage::Hello {
        protocol: turbine_distributed::rank::PROTOCOL_VERSION,
        rank,
        world_size: plan.tp,
        model_fingerprint: expect.model_fingerprint,
        config_fingerprint: expect.config_fingerprint,
        device_vendor: vendor,
        device_arch: arch,
    };
    Ok((transport, expect, hello))
}

/// An unsupported support-matrix row: logged as `event="support_matrix"`, exit 2 (a
/// configuration error, before any port is bound).
fn refuse_support(error: &ConfigError) -> ExitCode {
    support_startup::log_refusal(error);
    eprintln!("turbine-server: invalid configuration: {error}");
    ExitCode::Config
}

/// The multi-GPU pieces computed before the listener binds (Phase 5): the node topology graph
/// and the `parallel` status object of the plan.
struct ServeCluster {
    topology: TopologyGraph,
    parallel: serde_json::Value,
}

/// One data-parallel replica prepared before the listener binds (P5 S-7): its model on its
/// device, its `kv` section (per-replica L1 share and L2 directory), its share of the KV
/// metrics and its L2 tier (Phase 4).
struct ReplicaStart {
    prepared: PreparedModel,
    /// Tensor parallelism: the group's worker ranks and its collective backend (P5 S-6).
    group: Option<engine::tp::TpGroupStart>,
    /// Pipeline parallelism: the stages before the last and the collective backend (P5 S-10).
    pipeline: Option<engine::pp::PipelineStart>,
    /// Expert parallelism (P5 S-11): the group's token counts (`/turbine/v1/scheduler`
    /// `expert`, the expert metrics).
    experts: Option<Arc<parallel::ExpertStats>>,
    /// `static` rank mode, ranks 1..: this process is a worker rank, not an engine.
    worker: Option<engine::tp::StaticWorker>,
    kv_cfg: KvConfig,
    kv_metrics: KvMetrics,
    l2: Option<Arc<L2NvmeTier>>,
}

/// Replica `r`'s configuration (of `replicas`) on `device`: with data parallelism every replica
/// gets its share of the host-wide L1 cap (`kv.cpu.max_bytes / replicas`) and of the L2 cap, and
/// its own L2 directory (`kv.nvme.path/replica-<r>`), since each replica wipes and owns its slab
/// files.
fn replica_config(config: &Config, r: usize, replicas: usize, device: DeviceId) -> Config {
    let mut cfg = config.clone();
    cfg.execution.device = device;
    if replicas > 1 {
        let n = replicas as u64;
        cfg.kv.cpu.max_bytes = ByteSize(cfg.kv.cpu.max_bytes.0 / n);
        cfg.kv.nvme.max_bytes = ByteSize(cfg.kv.nvme.max_bytes.0 / n);
        cfg.kv.nvme.path = cfg.kv.nvme.path.join(format!("replica-{r}"));
    }
    cfg
}

async fn serve(
    config: Config,
    inventory: DeviceInventory,
    cluster: ServeCluster,
    metrics: MetricsRegistry,
    replicas: Vec<ReplicaStart>,
    support: SupportRowView,
) -> ExitCode {
    let addr: SocketAddr = config.server.listen;
    let engine_metrics = EngineMetrics {
        server: ServerMetrics::register(&metrics),
        model: ModelMetrics::register(&metrics),
        scheduler: SchedulerMetrics::register(&metrics),
        kv: replicas[0].kv_metrics.clone(),
    };
    // `parallel.router` was checked against the registry before any port was bound
    // (`Config::validate_modules`); `select` logs `module_selected`.
    let router_policy = match turbine_distributed::router::select(config.parallel.router.as_str()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("turbine-server: parallel.router: {e}");
            return ExitCode::Config;
        }
    };
    let backend = Arc::new(
        ModelBackend::new(&replicas[0].prepared, &inventory, &engine_metrics)
            .with_replicas(replicas.len(), router_policy, &metrics)
            .with_support(support)
            .with_topology(&cluster.topology)
            .with_parallel(cluster.parallel)
            .with_experts(replicas.iter().map(|r| r.experts.clone()).collect())
            .with_rank_worker(replicas[0].worker.is_some()),
    );
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
    let reliability_metrics = ReliabilityMetrics::register(&state.metrics);
    let telemetry_metrics = TelemetryMetrics::register(&state.metrics);
    for (r, replica) in replicas.into_iter().enumerate() {
        let startup = ReliabilityStartup {
            inventory: inventory.clone(),
            devices: config.devices.clone(),
            metrics: reliability_metrics.clone(),
            telemetry: telemetry_metrics.clone(),
            kv: KvSetup {
                cfg: replica.kv_cfg,
                l2: replica.l2,
            },
            replica: r as u32,
        };
        // Replica 0 uses the metric shares registered above; the others add their own shares
        // of the scheduler and KV gauges (the gauges report the sum over replicas).
        let metrics = EngineMetrics {
            server: engine_metrics.server.clone(),
            model: engine_metrics.model.clone(),
            scheduler: if r == 0 {
                engine_metrics.scheduler.clone()
            } else {
                engine_metrics.scheduler.another_replica()
            },
            kv: replica.kv_metrics,
        };
        if let Some(worker) = replica.worker {
            // P5 S-5: a static worker rank runs its shard, not an engine.
            let (backend, fatal, reliability) = (
                Arc::clone(&backend),
                fatal_tx.clone(),
                reliability_metrics.clone(),
            );
            let prepared = replica.prepared;
            let spawned = std::thread::Builder::new()
                .name("turbine-rank-worker".into())
                .spawn(move || {
                    let phase = |reason| backend.set_loading(reason);
                    let result = engine::tp::run_static_worker(
                        &prepared,
                        worker,
                        &reliability,
                        &phase,
                        || backend.set_rank_ready(),
                    );
                    let _ = fatal.send(Fatal::RankStopped(result.err()));
                });
            if let Err(e) = spawned {
                let _ = fatal_tx.send(Fatal::LoadFailed(format!(
                    "cannot start the worker rank thread: {e}"
                )));
            }
            continue;
        }
        if let Err(e) = engine::spawn(
            replica.prepared,
            replica.group,
            replica.pipeline,
            Arc::clone(&backend),
            metrics,
            startup,
            queue_capacity,
            Timeouts::from_config(&config.server),
            fatal_tx.clone(),
        ) {
            let _ = fatal_tx.send(Fatal::LoadFailed(format!(
                "cannot start engine thread {r}: {e}"
            )));
        }
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
            let (message, code) = match &fatal {
                Fatal::LoadFailed(m) => (format!("model load failed: {m}"), ExitCode::Startup),
                Fatal::DeviceFatal(m) => (format!("device error: {m}"), ExitCode::DeviceFatal),
                Fatal::RankStopped(None) => {
                    tracing::info!(event = "rank_shutdown", "the leader shut this rank down");
                    return ExitCode::Clean;
                }
                Fatal::RankStopped(Some(m)) => (format!("rank stopped: {m}"), ExitCode::Startup),
            };
            tracing::error!(error = %message, "exiting");
            eprintln!("turbine-server: {message}");
            tokio::time::sleep(FAILURE_GRACE).await;
            code
        }
    }
}

/// Startup step after discovery (P5 S-1): the node-local topology graph from `/sys` and the
/// vendor source of every registered discovery kind (never fails; missing sources are `unknown`
/// with one WARN each), logged as `event="topology_captured"`.
fn capture_topology(opts: &DiscoveryOptions, inventory: &DeviceInventory) -> TopologyGraph {
    let started = std::time::Instant::now();
    let mut vendor = RegisteredTopology::new(opts.clone());
    let graph = discover_topology(Path::new("/sys"), inventory, &mut vendor);
    tracing::info!(
        event = "topology_captured",
        hostname = %graph.node.hostname,
        vertices = graph.vertices.len(),
        edges = graph.edges.len(),
        seconds = started.elapsed().as_secs_f64(),
        "node topology captured"
    );
    graph
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
