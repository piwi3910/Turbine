//! Model startup (P1 §Interfaces, contract §16.3 steps 4–6 and 8–10): the kernel provider for
//! `execution.backend`, the model config, tokenizer and chat template, `model.max_seq_len`, the
//! kernel registry, the memory budget — weights + the `kv.gpu.max_bytes` block pool + the
//! executor workspace for `scheduler.max_batch_tokens` + the emergency reserve (P2
//! Constraints) — all before the listener binds; then the weight load, the KV pool allocation
//! and the one-token warm-up (after it binds).

use std::fmt;
use std::path::{Component, Path};
use std::sync::Arc;
use std::time::Instant;

use turbine_core::config::{Config, StructuredOutputConfig};
use turbine_core::types::SeqId;
use turbine_device::DeviceInventory;
use turbine_kernels::backends::{
    self, BackendNote, BackendRequest, ExecutionBackend, OpenedBackend,
};
use turbine_kernels::{KernelError, KernelMetrics, KernelProvider, KernelRegistry, ShimContext};
use turbine_kv::metrics::log_pool_startup;
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_model::executor::{
    self, BatchInput, DecodeGraphs, ExecutorOptions, GraphBackend, ModelExecutor, SeqSlice, graphs,
};
use turbine_model::loader::LoadedWeights;
use turbine_model::{
    BudgetTerms, ChatTemplate, GenerationConfig, GrammarCompiler, MAX_STAGING_BYTES,
    ModelArchConfig, ModelError, ModelFamily, ModelMetrics, SafetensorsIndex, Tokenizer,
    WeightLoader, available_bytes, check_budget, host_mem_available, load_generation_config,
    load_model_config,
};
use turbine_observability::MetricsRegistry;
use turbine_scheduler::SchedulerParams;

use crate::modules::{self, ModuleChoices};
use turbine_tensor::DeviceMemory;

/// Linux host memory figures (`MemAvailable`); absent elsewhere.
const MEMINFO: &str = "/proc/meminfo";
/// Default `model.max_seq_len` cap (P1 §Configuration).
const DEFAULT_MAX_SEQ_LEN: u32 = 32_768;

/// A startup failure after configuration validation: exit 1 with this message.
#[derive(Debug)]
pub struct StartupError(String);

impl StartupError {
    fn new(message: impl Into<String>) -> StartupError {
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

/// `model.tool_call_parser`, as the name of a registered tool format or `None` (tool calling
/// off): when null, the family's default format ([`ModelFamily::default_tool_format`], e.g.
/// `llama3_json` for `llama`) if the chat template renders `tools`.
/// A format is only usable with a template that renders `tools`: a configured `llama3_json` on
/// any other template resolves to none (logged), so `tools` requests get 400
/// `tools_not_supported` instead of a prompt that silently lacks them. `none` and a name
/// `Config::validate_modules` would have refused resolve to none.
pub fn resolve_tool_call_parser(
    configured: Option<&str>,
    family: &dyn ModelFamily,
    renders_tools: bool,
) -> Option<&'static str> {
    match configured {
        Some("none") => None,
        Some(name) if renders_tools => modules::TOOL_FORMATS.iter().copied().find(|f| *f == name),
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
            .and_then(|d| modules::TOOL_FORMATS.iter().copied().find(|f| *f == d)),
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
    Ok(Provider { backend, opened })
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
    /// The L0 block pool `kv.gpu.max_bytes` holds.
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
    /// The resolved `model.tool_call_parser`: a registered tool format, or `None` (off).
    pub tool_call_parser: Option<&'static str>,
    /// The module picked at each extension point (`/turbine/v1/status` `modules`).
    pub modules: ModuleChoices,
    pub served_name: String,
    pub budget: BudgetTerms,
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
    )
    .map_err(|e| kernel_error("kernel selection", e))?;
    for note in provider.backend.selection_notes(registry.selections()) {
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
    let available = available_bytes(
        opened.memory_kind,
        device_free,
        host_mem_available(Path::new(MEMINFO)),
    );
    let weights = arch.shape().weight_bytes;
    let workspace = executor::workspace_bytes(
        &arch,
        block_tokens,
        scheduler.max_batch_tokens,
        scheduler.max_running_requests,
    );
    let emergency_reserve = config.reliability.emergency_vram_reserve.0;
    let pool = kv_pool_config(
        &arch,
        block_tokens,
        config.kv.gpu.max_bytes.map(|b| b.0),
        available.saturating_sub(
            weights
                .saturating_add(workspace)
                .saturating_add(emergency_reserve),
        ),
        scheduler.max_running_requests,
    )?;
    let budget = BudgetTerms {
        weights,
        kv_reservation: u64::from(pool.num_blocks) * pool.layout.block_bytes(),
        workspace,
        emergency_reserve,
        available,
    };
    check_budget(&budget).map_err(|e| model_error("startup", e))?;

    let started = Instant::now();
    let grammar = GrammarCompiler::new(&tokenizer, &eos_token_ids(&arch, &generation))
        .map(Arc::new)
        .map_err(|e| model_error("structured output", e))?;
    let tool_call_parser = resolve_tool_call_parser(
        config.model.tool_call_parser.as_ref().map(|n| n.as_str()),
        arch.family.0,
        template.renders_tools(),
    );
    tracing::info!(
        token_trie_seconds = started.elapsed().as_secs_f64(),
        tool_call_parser = tool_call_parser.unwrap_or("none"),
        "structured output ready"
    );
    let modules = ModuleChoices {
        family: arch.family.0.name().to_string(),
        tool_format: tool_call_parser.map(str::to_string),
        weight_format: arch.weight_format.0.name().to_string(),
        backend: config.execution.backend.to_string(),
        card_profile: provider.opened.card.map(|card| card.name.to_string()),
        scheduling_policy: config.scheduler.policy.to_string(),
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
        weight_bytes = budget.weights,
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
        tool_call_parser,
        modules,
        served_name,
        budget,
    })
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

/// The L0 pool: as many blocks as `max_bytes` (`kv.gpu.max_bytes`) holds, or, when it is null,
/// as the budget `remainder` after weights, workspace and the emergency reserve holds. At least
/// one block per running request.
fn kv_pool_config(
    arch: &ModelArchConfig,
    block_tokens: u32,
    max_bytes: Option<u64>,
    remainder: u64,
    max_running: u32,
) -> Result<BlockPoolConfig, StartupError> {
    let bytes = max_bytes.unwrap_or(remainder);
    let pool = BlockPoolConfig::for_bytes(arch.kv_layout(block_tokens), bytes);
    if pool.num_blocks < max_running {
        return Err(StartupError::new(format!(
            "kv.gpu.max_bytes: {bytes} B hold {} KV blocks of {} B; at least one block per \
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

/// The loaded, warmed-up executor and the L0 block pool its requests run on.
pub struct LoadedModel {
    pub executor: Box<dyn ModelExecutor>,
    pub pool: BlockPool,
    pub weight_bytes: u64,
    pub load_seconds: f64,
}

/// Steps 8–10: upload the weights, build the executor for `prepared`'s scheduler bounds,
/// allocate the KV block pool and run one one-token forward on a block of it. Records
/// `turbine_model_load_seconds` and `turbine_model_weight_bytes{format}` (the weight format).
pub fn load(
    prepared: &PreparedModel,
    warmup_token: u32,
    metrics: &ModelMetrics,
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
    let mut pool = BlockPool::new(prepared.pool, Arc::clone(mem))
        .map_err(|e| StartupError::new(format!("KV block pool: {e}")))?;
    log_pool_startup(&pool);
    warm_up(executor.as_mut(), &mut pool, warmup_token)?;
    let load_seconds = started.elapsed().as_secs_f64();
    metrics.record_load(load_seconds, arch.weight_format.0.name(), weight_bytes);
    tracing::info!(load_seconds, weight_bytes, "model loaded and warmed up");
    Ok(LoadedModel {
        executor,
        pool,
        weight_bytes,
        load_seconds,
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
    use turbine_model::families::{Llama, Olmoe};
    use turbine_model::tools::LLAMA3_JSON;

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
        // Null: the family default (llama3_json for llama, none for olmoe) when the template
        // renders tools.
        assert_eq!(resolve_tool_call_parser(None, &Llama, true), LLAMA);
        assert_eq!(resolve_tool_call_parser(None, &Llama, false), OFF);
        assert_eq!(resolve_tool_call_parser(None, &Olmoe, true), OFF);
        // Explicit values; a parser needs a template that renders tools.
        assert_eq!(resolve_tool_call_parser(LLAMA, &Olmoe, true), LLAMA);
        assert_eq!(resolve_tool_call_parser(LLAMA, &Llama, false), OFF);
        assert_eq!(resolve_tool_call_parser(Some("none"), &Llama, true), OFF);
    }
}
