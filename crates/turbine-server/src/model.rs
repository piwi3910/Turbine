//! Model startup (P1 §Interfaces, contract §16.3 steps 4–6 and 8–10): the kernel provider for
//! `execution.backend`, the model config, tokenizer and chat template, `model.max_seq_len`, the
//! kernel registry and the pre-load memory budget (P3 S-2: `compute_budget` on the memory
//! measured free before any weight byte is read) — all before the listener binds; then the
//! weight load, the memory budget re-measured after it, the reservation ledger (weights and
//! workspace committed), the KV pool sized from the budget's `kv` pool, the emergency reserve
//! and the one-token warm-up (after it binds).

use std::fmt;
use std::io::Read;
use std::path::{Component, Path};
use std::sync::Arc;
use std::time::Instant;

use turbine_core::config::{ByteSize, Config, ReliabilityConfig, StructuredOutputConfig};
use turbine_core::types::{DeviceId, KvLayout, MemoryKind, ModelIdentity, SeqId};
use turbine_device::DeviceInventory;
use turbine_device::telemetry::proc::FsProc;
use turbine_device::telemetry::read_host;
use turbine_kernels::backends::{
    self, BackendNote, BackendRequest, ExecutionBackend, OpenedBackend,
};
use turbine_kernels::{
    KernelError, KernelMetrics, KernelProvider, KernelRegistry, ShimContext,
    TURBINE_OPTION_GEMM_AUTOTUNE,
};
use turbine_kv::metrics::log_pool_startup;
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_model::executor::{
    self, BatchInput, DecodeGraphs, ExecutorOptions, GraphBackend, ModelExecutor, SeqSlice, graphs,
};
use turbine_model::formats::{self, BoundToolFormat, ToolFormat};
use turbine_model::loader::LoadedWeights;
use turbine_model::{
    ChatTemplate, GenerationConfig, GrammarCompiler, MAX_STAGING_BYTES, ModelArchConfig,
    ModelError, ModelFamily, ModelMetrics, SafetensorsIndex, Tokenizer, WeightLoader,
    load_generation_config, load_model_config,
};
use turbine_observability::MetricsRegistry;
use turbine_reliability::budget::{BudgetInputs, DeviceBudget, PoolKind, compute_budget};
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::reserve::EmergencyReserve;
use turbine_scheduler::SchedulerParams;

use crate::modules::ModuleChoices;
use turbine_tensor::DeviceMemory;

/// Linux host memory figures (`MemAvailable`); absent elsewhere.
const MEMINFO: &str = "/proc/meminfo";
/// Default `model.max_seq_len` cap (P1 §Configuration).
const DEFAULT_MAX_SEQ_LEN: u32 = 32_768;

/// A startup failure after configuration validation: exit 1 with this message.
#[derive(Debug)]
pub struct StartupError(String);

impl StartupError {
    pub(crate) fn new(message: impl Into<String>) -> StartupError {
        StartupError(message.into())
    }
}

impl fmt::Display for StartupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn model_error(context: &str, e: ModelError) -> StartupError {
    StartupError::new(format!("{context}: {e}"))
}

fn kernel_error(context: &str, e: KernelError) -> StartupError {
    StartupError::new(format!("{context}: {e}"))
}

/// `model.served_name` default: `<org>/<name>` for a Hugging Face cache snapshot
/// (`…/models--<org>--<name>/snapshots/<rev>`), else the last path component.
pub fn default_served_name(path: &Path) -> String {
    let parts: Vec<&str> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    if let [.., repo, "snapshots", _rev] = parts.as_slice()
        && let Some(rest) = repo.strip_prefix("models--")
        && let Some((org, name)) = rest.split_once("--")
        && !org.is_empty()
        && !name.is_empty()
    {
        return format!("{org}/{name}");
    }
    parts
        .last()
        .map_or_else(|| path.display().to_string(), |s| (*s).to_string())
}

/// `model.max_seq_len`: the configured value, at most the model's `max_position_embeddings`;
/// by default min(32768, `max_position_embeddings`).
pub fn resolve_max_seq_len(configured: Option<u32>, max_positions: u32) -> Result<u32, String> {
    match configured {
        None => Ok(DEFAULT_MAX_SEQ_LEN.min(max_positions)),
        Some(n) if n >= 1 && n <= max_positions => Ok(n),
        Some(n) => Err(format!(
            "model.max_seq_len {n} is outside 1..={max_positions} (the model's \
             max_position_embeddings)"
        )),
    }
}

/// `model.tool_call_parser`, as a format of the tool-format registry
/// ([`turbine_model::formats::registry`]) or `None` (tool calling off): when null, the family's
/// default format ([`ModelFamily::default_tool_format`], e.g. `llama3_json` for `llama`) if the
/// chat template renders `tools`.
/// A format is only usable with a template that renders `tools`: a configured `llama3_json` on
/// any other template resolves to none (logged), so `tools` requests get 400
/// `tools_not_supported` instead of a prompt that silently lacks them. `none` and a name
/// `Config::validate_modules` would have refused resolve to none.
pub fn resolve_tool_call_parser(
    configured: Option<&str>,
    family: &dyn ModelFamily,
    renders_tools: bool,
) -> Option<&'static dyn ToolFormat> {
    match configured {
        Some("none") => None,
        Some(name) if renders_tools => formats::registry().get(name),
        Some(name) => {
            tracing::warn!(
                parser = name,
                "model.tool_call_parser is set but the chat template does not render tools; \
                 tool calling is disabled"
            );
            None
        }
        None if renders_tools => family
            .default_tool_format()
            .and_then(|d| formats::registry().get(d)),
        None => None,
    }
}

/// The ids that end a generation: `config.json` `eos_token_id`, else
/// `generation_config.json`'s.
pub fn eos_token_ids(arch: &ModelArchConfig, generation: &GenerationConfig) -> Vec<u32> {
    if arch.eos_token_ids.is_empty() {
        generation.eos_token_ids.to_vec()
    } else {
        arch.eos_token_ids.to_vec()
    }
}

/// The opened `execution.backend`: device memory, kernel providers and the card profile
/// (`opened.card`).
pub struct Provider {
    /// The registered backend (`execution_backend`), for its notes on the kernel selections.
    pub backend: &'static dyn ExecutionBackend,
    pub opened: OpenedBackend,
}

/// Step 4: open the registered backend `execution.backend` names (logged as `module_selected`)
/// on `execution.device` with `execution.card_profile`. An unregistered name was refused with
/// exit 2 before this point; a backend that cannot open (including a device no card profile
/// describes) is exit 1.
pub fn load_provider(
    config: &Config,
    inventory: &DeviceInventory,
) -> Result<Provider, StartupError> {
    let exec = &config.execution;
    let backend = backends::registry()
        .select(exec.backend.as_str(), "execution.backend")
        .map_err(|e| StartupError::new(e.to_string()))?;
    let opened = backend
        .open(&BackendRequest {
            device: exec.device,
            kernel_library: exec.kernel_library.as_deref(),
            inventory,
            meminfo: Path::new(MEMINFO),
            card_profile: exec.card_profile.as_str(),
        })
        .map_err(|e| StartupError::new(e.to_string()))?;
    if let Some(ctx) = &opened.context {
        apply_gemm_autotune(ctx, exec.gemm_autotune);
    }
    Ok(Provider { backend, opened })
}

/// `execution.gemm_autotune` → the kernel library's `TURBINE_OPTION_GEMM_AUTOTUNE` (the tuned
/// GEMM table on or off), before any GEMM runs. A library without context options keeps its own
/// choice: logged, not fatal.
fn apply_gemm_autotune(ctx: &ShimContext, on: bool) {
    match ctx.set_option(TURBINE_OPTION_GEMM_AUTOTUNE, i64::from(on)) {
        Ok(()) => tracing::info!(
            event = "gemm_tuning",
            tuned_table = on,
            "execution.gemm_autotune: {}",
            if on {
                "GEMM shapes the kernel library's tuned table covers run its pinned algorithms"
            } else {
                "every GEMM runs the first heuristic answer"
            }
        ),
        Err(e) => tracing::info!(
            event = "gemm_tuning_unavailable",
            reason = %e,
            "execution.gemm_autotune ignored: the kernel library has no context options"
        ),
    }
}

/// Everything resolved before the listener binds: the model is known to be loadable and to fit.
pub struct PreparedModel {
    pub provider: Provider,
    pub arch: ModelArchConfig,
    pub generation: GenerationConfig,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<ChatTemplate>,
    pub index: SafetensorsIndex,
    pub registry: Arc<KernelRegistry>,
    pub max_seq_len: u32,
    /// `kv.block_tokens`: the block size of the KV pool.
    pub block_tokens: u32,
    /// The L0 block pool of the pre-load budget's `kv` pool (capped by `kv.gpu.max_bytes`);
    /// the pool allocated after the weights load is at most this large.
    pub pool: BlockPoolConfig,
    /// Scheduler bounds; `max_batch_tokens` and `max_running_requests` also size the executor.
    pub scheduler: SchedulerParams,
    /// The executor's op sequence (`execution.fused_ops`).
    pub executor_options: ExecutorOptions,
    /// `execution.decode_graphs`, and the provider can capture graphs.
    pub decode_graphs: bool,
    /// `execution.overlap_scheduling` (P2c): the engine launches each iteration before the host
    /// work of the previous one, when the executor can.
    pub overlap_scheduling: bool,
    /// Compiles `response_format` and tool grammars; its token trie is built once, here.
    pub grammar: Arc<GrammarCompiler>,
    /// `structured_output` bounds on those grammars.
    pub structured_output: StructuredOutputConfig,
    /// The resolved `model.tool_call_parser`: a registered tool format bound to the tokenizer,
    /// or `None` (off).
    pub tool_format: Option<Arc<BoundToolFormat>>,
    /// The module picked at each extension point (`/turbine/v1/status` `modules`).
    pub modules: ModuleChoices,
    pub served_name: String,
    /// The pre-load memory budget (P3 S-2); the engine re-measures it after the weights load.
    pub budget: DeviceBudget,
    /// `execution.device`: the device the budget and the ledger describe.
    pub device: DeviceId,
    /// The `reliability` section with `memory.workspace_bytes` raised to the executor's
    /// workspace when that is larger (the workspace pool must hold what the executor allocates).
    pub reliability: ReliabilityConfig,
    /// `kv.gpu.max_bytes`: caps the budget's `kv` pool (CONFLICT C-8).
    pub kv_cap: Option<ByteSize>,
    /// Bytes of the executor workspace for `scheduler.max_batch_tokens`.
    pub workspace_bytes: u64,
    /// What the KV cache depends on: the namespace of every cached block (P4 S-1).
    pub identity: ModelIdentity,
}

/// The model's identity (P4 §Data namespace key): BLAKE3 of `config.json` and of the
/// safetensors index, or of the single file's header when there is no index.
pub fn model_identity(dir: &Path) -> Result<ModelIdentity, StartupError> {
    let read =
        |p: &Path| std::fs::read(p).map_err(|e| StartupError::new(format!("{}: {e}", p.display())));
    let config = read(&dir.join("config.json"))?;
    let index_path = dir.join("model.safetensors.index.json");
    let index = if index_path.is_file() {
        read(&index_path)?
    } else {
        let path = dir.join("model.safetensors");
        let io = |e: std::io::Error| StartupError::new(format!("{}: {e}", path.display()));
        let mut f = std::fs::File::open(&path).map_err(io)?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(io)?;
        let n = u64::from_le_bytes(len);
        let mut header = Vec::new();
        f.take(n).read_to_end(&mut header).map_err(io)?;
        header
    };
    Ok(ModelIdentity::from_bytes(&config, &index))
}

/// Steps 4–6: provider, model config + tokenizer + template, registry, memory budget. No weight
/// byte is read here (safetensors headers only).
pub fn prepare(
    config: &Config,
    inventory: &DeviceInventory,
    metrics: &MetricsRegistry,
) -> Result<PreparedModel, StartupError> {
    let provider = load_provider(config, inventory)?;

    let dir = config.model.path.as_path();
    if !dir.is_dir() {
        return Err(StartupError::new(format!(
            "model.path {}: not an existing directory",
            dir.display()
        )));
    }
    let arch = load_model_config(dir).map_err(|e| model_error("model config", e))?;
    let generation = if dir.join("generation_config.json").is_file() {
        load_generation_config(dir).map_err(|e| model_error("generation config", e))?
    } else {
        GenerationConfig::default()
    };
    let tokenizer_path = config
        .model
        .tokenizer
        .clone()
        .unwrap_or_else(|| dir.join("tokenizer.json"));
    let tokenizer =
        Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(|e| model_error("tokenizer", e))?);
    let template = ChatTemplate::resolve(dir, config.model.chat_template.as_deref())
        .map(Arc::new)
        .map_err(|e| model_error("chat template", e))?;
    let max_seq_len = resolve_max_seq_len(config.model.max_seq_len, arch.max_position_embeddings)
        .map_err(StartupError::new)?;
    let index = SafetensorsIndex::open(dir).map_err(|e| model_error("weights", e))?;
    arch.check_supported_weights(&index)
        .map_err(|e| model_error("weights", e))?;

    if !config.kv.gpu.enabled {
        return Err(StartupError::new(
            "kv.gpu.enabled is false: the GPU KV tier (the L0 block pool) is required",
        ));
    }
    let block_tokens = config.kv.block_tokens;
    let executor_options = ExecutorOptions::from_fused_ops(config.execution.fused_ops);
    // Optional ops (kernel ABI v2.1) join the requirements only when a provider in the
    // selection order has them; otherwise the executor runs their ABI v2 equivalent.
    let opened = &provider.opened;
    let ordered: Vec<Arc<dyn KernelProvider>> = opened
        .providers
        .iter()
        .filter(|p| opened.order.contains(&p.id()))
        .cloned()
        .collect();
    let mut requirements =
        executor::available_requirements(&arch, block_tokens, executor_options, &ordered);
    // Device-side logits reduction (P2c S-4) is optional: only when enabled and a provider
    // implements it; otherwise the executor copies every row whole.
    let reduce = executor::logits::reduce_requirement(&arch);
    if config.execution.device_sampling
        && ordered.iter().any(|p| reduce.spec.supported_by(p.as_ref()))
    {
        requirements.push(reduce);
    }
    let registry = KernelRegistry::build(
        opened.providers.clone(),
        &opened.order,
        &requirements,
        &KernelMetrics::register(metrics),
        opened.card,
    )
    .map_err(|e| kernel_error("kernel selection", e))?;
    for note in provider
        .backend
        .selection_notes(opened.card, registry.selections())
    {
        log_backend_note(&note, block_tokens);
    }

    let decode_graphs = config.execution.decode_graphs && opened.graphs.is_some();
    if config.execution.decode_graphs && !decode_graphs {
        tracing::warn!(
            event = "decode_graphs_unavailable",
            backend = %config.execution.backend,
            "execution.decode_graphs is on but the kernel provider cannot capture graphs \
             (kernel ABI v2.1 graph functions); decode iterations run eagerly"
        );
    }

    let scheduler = SchedulerParams::from_config(config, max_seq_len);
    let device_free = opened
        .mem
        .mem_info()
        .map_err(|e| StartupError::new(format!("device memory info: {e}")))?
        .free_bytes;
    let weights = arch.shape().weight_bytes;
    let workspace = executor::workspace_bytes(
        &arch,
        block_tokens,
        scheduler.max_batch_tokens,
        scheduler.max_running_requests,
    );
    let reliability = reliability_for_workspace(&config.reliability, workspace);
    let device = config.execution.device;
    let layout = arch.kv_layout(block_tokens);
    // P3 S-2 pre-check: the budget of the memory free now, before any weight byte is read,
    // must hold the model; the engine re-measures it after the weights load.
    let budget = measure_budget(
        device,
        provider.opened.memory_kind,
        Some(device_free),
        0,
        weights,
        &layout,
        max_seq_len,
        &reliability,
        config.kv.gpu.max_bytes,
    )?;
    let pool = kv_pool_config(
        &arch,
        block_tokens,
        budget.pool(PoolKind::Kv),
        scheduler.max_running_requests,
    )?;
    tracing::info!(
        event = "memory_budget",
        device = device.0,
        memory_kind = turbine_reliability::budget::memory_kind_str(budget.memory_kind),
        budget_bytes = budget.budget_bytes,
        weights,
        kv = budget.pool(PoolKind::Kv),
        workspace = budget.pool(PoolKind::Workspace),
        runtime = budget.pool(PoolKind::Runtime),
        reserve = budget.pool(PoolKind::Reserve),
        kv_blocks = pool.num_blocks,
        "pre-load memory budget"
    );

    let started = Instant::now();
    let grammar = GrammarCompiler::new(&tokenizer, &eos_token_ids(&arch, &generation))
        .map(Arc::new)
        .map_err(|e| model_error("structured output", e))?;
    // A format whose required special tokens the tokenizer lacks is refused here (exit 1).
    let tool_format = resolve_tool_call_parser(
        config.model.tool_call_parser.as_ref().map(|n| n.as_str()),
        arch.family.0,
        template.renders_tools(),
    )
    .map(|format| formats::bind(format, &tokenizer).map(Arc::new))
    .transpose()
    .map_err(|e| model_error("model.tool_call_parser", e))?;
    let tool_format_name = tool_format.as_ref().map(|f| f.format.name());
    tracing::info!(
        token_trie_seconds = started.elapsed().as_secs_f64(),
        tool_call_parser = tool_format_name.unwrap_or("none"),
        "structured output ready"
    );
    let modules = ModuleChoices {
        family: arch.family.0.name().to_string(),
        tool_format: tool_format_name.map(str::to_string),
        weight_format: arch.weight_format.0.name().to_string(),
        backend: config.execution.backend.to_string(),
        card_profile: provider.opened.card.map(|card| card.name.to_string()),
        scheduling_policy: config.scheduler.policy.to_string(),
        eviction_policy: config.kv.policy.to_string(),
    };
    modules.log();

    let served_name = config
        .model
        .served_name
        .clone()
        .unwrap_or_else(|| default_served_name(dir));
    tracing::info!(
        served_name = %served_name,
        architecture = %arch.hf_architecture,
        max_seq_len,
        weight_bytes = weights,
        "model prepared"
    );
    Ok(PreparedModel {
        provider,
        arch,
        generation,
        tokenizer,
        template,
        index,
        registry: Arc::new(registry),
        max_seq_len,
        block_tokens,
        pool,
        scheduler,
        executor_options,
        decode_graphs,
        overlap_scheduling: config.execution.overlap_scheduling,
        grammar,
        structured_output: config.structured_output.clone(),
        tool_format,
        modules,
        served_name,
        budget,
        device,
        reliability,
        kv_cap: config.kv.gpu.max_bytes,
        workspace_bytes: workspace,
        identity: model_identity(dir)?,
    })
}

/// `reliability` with `memory.workspace_bytes` at least the executor's `workspace` (logged when
/// raised): the workspace pool holds the executor workspace allocated at startup.
fn reliability_for_workspace(cfg: &ReliabilityConfig, workspace: u64) -> ReliabilityConfig {
    let mut cfg = cfg.clone();
    if workspace > cfg.memory.workspace_bytes.0 {
        tracing::info!(
            event = "memory_budget",
            configured = cfg.memory.workspace_bytes.0,
            executor_workspace = workspace,
            "reliability.memory.workspace_bytes raised to the executor workspace"
        );
        cfg.memory.workspace_bytes = ByteSize(workspace);
    }
    cfg
}

/// `compute_budget` for `device`: dedicated memory from `measured_free` plus what Turbine
/// already holds there, unified memory from the host `MemAvailable` read now. The error names
/// every pool and its bytes.
#[allow(clippy::too_many_arguments)]
fn measure_budget(
    device: DeviceId,
    memory_kind: MemoryKind,
    measured_free: Option<u64>,
    already_held: u64,
    weights: u64,
    layout: &KvLayout,
    max_seq_len: u32,
    reliability: &ReliabilityConfig,
    kv_cap: Option<ByteSize>,
) -> Result<DeviceBudget, StartupError> {
    let host = read_host(&FsProc::default()).sample.mem_available_bytes;
    compute_budget(
        &BudgetInputs {
            device,
            memory_kind,
            measured_free_bytes: measured_free,
            already_held_bytes: already_held,
            host_mem_available_bytes: host,
            weights_bytes: weights,
            kv_bytes_per_token: layout.bytes_per_token(),
            max_seq_len,
            block_bytes: layout.block_bytes(),
            // P5: communicator buffers (0 on a single device; TP measures them at init).
            collective_bytes: 0,
        },
        reliability,
        kv_cap,
    )
    .map_err(|e| StartupError::new(format!("startup: {e}")))
}

/// Logs a backend's note on the kernel selections at INFO: `event=<note.event>`, `block_tokens`
/// (`kv.block_tokens`) and the note's fields (e.g. `impl` of `paged_attention_fallback`).
fn log_backend_note(note: &BackendNote, block_tokens: u32) {
    let fields: Vec<String> = note
        .fields
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    tracing::info!(
        event = note.event,
        block_tokens,
        fields = %fields.join(" "),
        "{}",
        note.message
    );
}

/// The L0 pool: as many blocks as the budget's `kv` pool (`bytes`: the budget remainder after
/// weights, workspace, runtime overhead and the emergency reserve, capped by
/// `kv.gpu.max_bytes`) holds. At least one block per running request.
fn kv_pool_config(
    arch: &ModelArchConfig,
    block_tokens: u32,
    bytes: u64,
    max_running: u32,
) -> Result<BlockPoolConfig, StartupError> {
    let pool = BlockPoolConfig::for_bytes(arch.kv_layout(block_tokens), bytes);
    if pool.num_blocks < max_running {
        return Err(StartupError::new(format!(
            "the kv pool of {bytes} B holds {} KV blocks of {} B; at least one block per \
             running request (scheduler.max_running_requests = {max_running}) is required",
            pool.num_blocks,
            pool.layout.block_bytes()
        )));
    }
    Ok(pool)
}

/// The one place the server builds a model executor: the family's executor for ragged
/// batches of up to `max_batch_tokens` tokens and `max_seqs` sequences over KV blocks of
/// `block_tokens` tokens, running the op sequence `options` selects.
#[allow(clippy::too_many_arguments)]
pub fn build_executor(
    arch: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
    options: ExecutorOptions,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    executor::build_executor(
        arch,
        weights,
        registry,
        mem,
        block_tokens,
        max_batch_tokens,
        max_seqs,
        options,
    )
}

/// The loaded, warmed-up executor, the L0 block pool its requests run on, and the memory
/// budget, reservation ledger and emergency reserve measured and taken after the weights load.
pub struct LoadedModel {
    pub executor: Box<dyn ModelExecutor>,
    pub pool: BlockPool,
    pub weight_bytes: u64,
    pub load_seconds: f64,
    pub budget: DeviceBudget,
    pub ledger: Arc<Ledger>,
    pub reserve: EmergencyReserve,
    /// The committed `weights` and `workspace` reservations: held while the engine runs.
    pub held: Vec<Reservation>,
}

/// Steps 8–10: upload the weights; re-measure the memory budget (P3 S-2: dedicated memory
/// free now plus the weights Turbine holds, or unified `MemAvailable`); open the reservation
/// ledger with the weights and the executor workspace committed; build the executor for
/// `prepared`'s scheduler bounds; allocate the KV block pool (the budget's `kv` pool, at most
/// the pre-load size) linked to the ledger; acquire the emergency reserve on the device; run
/// one one-token forward on a block of the pool. Records `turbine_model_load_seconds` and
/// `turbine_model_weight_bytes{format}` (the weight format).
pub fn load(
    prepared: &PreparedModel,
    warmup_token: u32,
    metrics: &ModelMetrics,
    reliability: &ReliabilityMetrics,
) -> Result<LoadedModel, StartupError> {
    let started = Instant::now();
    let arch = &prepared.arch;
    let mem = &prepared.provider.opened.mem;
    let weights = WeightLoader::load_format(
        arch.weight_format.0,
        &prepared.index,
        &arch.family.0.weight_slots(arch),
        mem,
        MAX_STAGING_BYTES,
    )
    .map_err(|e| model_error("weight load", e))?;
    let weight_bytes = weights.weight_bytes;

    let device = prepared.device;
    let layout = arch.kv_layout(prepared.block_tokens);
    let free = mem
        .mem_info()
        .map_err(|e| StartupError::new(format!("device memory info: {e}")))?
        .free_bytes;
    let budget = measure_budget(
        device,
        prepared.provider.opened.memory_kind,
        Some(free),
        weight_bytes,
        weight_bytes,
        &layout,
        prepared.max_seq_len,
        &prepared.reliability,
        prepared.kv_cap,
    )?;
    let ledger = Ledger::new(&budget);
    ledger.set_metrics(reliability.clone());
    let mut held = Vec::with_capacity(2);
    for (pool, bytes) in [
        (PoolKind::Weights, weight_bytes),
        (PoolKind::Workspace, prepared.workspace_bytes),
    ] {
        let mut r = ledger
            .reserve(device, pool, bytes)
            .map_err(|e| StartupError::new(format!("memory budget: {e}")))?;
        r.commit();
        held.push(r);
    }

    let mut executor = build_executor(
        arch,
        weights,
        Arc::clone(&prepared.registry),
        Arc::clone(mem),
        prepared.block_tokens,
        prepared.scheduler.max_batch_tokens,
        prepared.scheduler.max_running_requests,
        prepared.executor_options,
    )
    .map_err(|e| model_error("executor", e))?;
    if let Some(ctx) = prepared
        .provider
        .opened
        .graphs
        .as_ref()
        .filter(|_| prepared.decode_graphs)
    {
        let backend: Arc<dyn GraphBackend<Graph = _>> = Arc::<ShimContext>::clone(ctx);
        let capacity = graphs::capacity_for(prepared.scheduler.max_running_requests);
        executor.set_decode_graphs(Some(DecodeGraphs::new(backend, capacity)));
        tracing::info!(event = "decode_graphs", capacity, "decode graphs on");
    }
    let measured = kv_pool_config(
        arch,
        prepared.block_tokens,
        budget.pool(PoolKind::Kv),
        prepared.scheduler.max_running_requests,
    )?;
    let pool_config = BlockPoolConfig {
        num_blocks: measured.num_blocks.min(prepared.pool.num_blocks),
        ..measured
    };
    let mut pool = BlockPool::new(pool_config, Arc::clone(mem))
        .map_err(|e| StartupError::new(format!("KV block pool: {e}")))?
        .with_ledger(Arc::clone(&ledger), device);
    log_pool_startup(&pool);
    let reserve_bytes = prepared.reliability.emergency_vram_reserve.0;
    if reserve_bytes == 0 {
        tracing::warn!(
            event = "emergency_reserve",
            reason = "disabled",
            device = device.0,
            "reliability.emergency_vram_reserve is 0: no emergency reserve"
        );
    }
    let reserve = EmergencyReserve::acquire(
        device,
        reserve_bytes,
        &ledger,
        Box::new(crate::reliability::DeviceReserve::new(Arc::clone(mem))),
        reliability.clone(),
    )
    .map_err(|e| StartupError::new(format!("emergency reserve: {e}")))?;
    warm_up(executor.as_mut(), &mut pool, warmup_token)?;
    let load_seconds = started.elapsed().as_secs_f64();
    metrics.record_load(load_seconds, arch.weight_format.0.name(), weight_bytes);
    tracing::info!(
        load_seconds,
        weight_bytes,
        budget_bytes = budget.budget_bytes,
        kv_blocks = pool.total_blocks(),
        emergency_reserve = reserve_bytes,
        "model loaded and warmed up"
    );
    Ok(LoadedModel {
        executor,
        pool,
        weight_bytes,
        load_seconds,
        budget,
        ledger,
        reserve,
        held,
    })
}

/// One one-token forward on a block borrowed from the pool, which gets it back.
fn warm_up(
    exec: &mut dyn ModelExecutor,
    pool: &mut BlockPool,
    token: u32,
) -> Result<(), StartupError> {
    let blocks = pool
        .allocate(1)
        .map_err(|e| StartupError::new(format!("warm-up: {e}")))?;
    let result = {
        let view = pool.view();
        let seqs = [SeqSlice {
            seq: SeqId(0),
            q_start: 0,
            q_len: 1,
            kv_len: 1,
            block_table: &blocks,
            reduce: None,
        }];
        exec.forward(&BatchInput {
            tokens: &[token],
            positions: &[0],
            seqs: &seqs,
            kv: &view,
        })
    };
    pool.release(&blocks);
    let logits = result.map_err(|e| model_error("warm-up forward", e))?;
    if logits.rows != 1 || logits.data.iter().any(|v| !v.is_finite()) {
        return Err(StartupError::new(format!(
            "warm-up forward: expected one finite logits row, got {} rows",
            logits.rows
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbine_model::families::{Llama, Mistral, Mixtral, Olmoe, Qwen3, Qwen3Moe};
    use turbine_model::formats::llama3_json::LLAMA3_JSON;

    #[test]
    fn served_name_from_hf_snapshot_or_last_component() {
        assert_eq!(
            default_served_name(Path::new(
                "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f"
            )),
            "meta-llama/Llama-3.2-3B-Instruct"
        );
        assert_eq!(
            default_served_name(Path::new("/models/llama-3.2-3b-instruct")),
            "llama-3.2-3b-instruct"
        );
        assert_eq!(
            default_served_name(Path::new("/models/llama-3.2-3b-instruct/")),
            "llama-3.2-3b-instruct"
        );
        // Not a cache layout: `snapshots` without the `models--<org>--<name>` parent.
        assert_eq!(default_served_name(Path::new("/x/snapshots/abc")), "abc");
    }

    #[test]
    fn max_seq_len_defaults_and_bounds() {
        assert_eq!(resolve_max_seq_len(None, 131_072), Ok(32_768));
        assert_eq!(resolve_max_seq_len(None, 512), Ok(512));
        assert_eq!(resolve_max_seq_len(Some(256), 512), Ok(256));
        assert_eq!(resolve_max_seq_len(Some(512), 512), Ok(512));
        let err = resolve_max_seq_len(Some(513), 512).unwrap_err();
        assert!(err.contains("model.max_seq_len 513"), "{err}");
        assert!(resolve_max_seq_len(Some(0), 512).is_err());
    }

    #[test]
    fn tool_call_parser_resolution() {
        const LLAMA: Option<&str> = Some(LLAMA3_JSON);
        const OFF: Option<&str> = None;
        let resolve_tool_call_parser = |configured, family: &dyn ModelFamily, renders| {
            resolve_tool_call_parser(configured, family, renders).map(|f| f.name())
        };
        // Null: the family default (llama3_json for llama, none for olmoe) when the template
        // renders tools.
        assert_eq!(resolve_tool_call_parser(None, &Llama, true), LLAMA);
        assert_eq!(resolve_tool_call_parser(None, &Llama, false), OFF);
        assert_eq!(resolve_tool_call_parser(None, &Olmoe, true), OFF);
        // Explicit values; a parser needs a template that renders tools.
        assert_eq!(resolve_tool_call_parser(LLAMA, &Olmoe, true), LLAMA);
        assert_eq!(resolve_tool_call_parser(LLAMA, &Llama, false), OFF);
        assert_eq!(resolve_tool_call_parser(Some("none"), &Llama, true), OFF);
        // The Phase 8 families' defaults (Phase 2m S-11), and explicit Phase 8 formats.
        let hermes = Some("hermes");
        let mistral = Some("mistral");
        assert_eq!(resolve_tool_call_parser(None, &Qwen3, true), hermes);
        assert_eq!(resolve_tool_call_parser(None, &Qwen3Moe, true), hermes);
        assert_eq!(resolve_tool_call_parser(None, &Mistral, true), mistral);
        assert_eq!(resolve_tool_call_parser(None, &Mixtral, true), mistral);
        assert_eq!(resolve_tool_call_parser(None, &Qwen3, false), OFF);
        assert_eq!(resolve_tool_call_parser(hermes, &Llama, true), hermes);
        assert_eq!(resolve_tool_call_parser(mistral, &Qwen3, true), mistral);
    }
}
