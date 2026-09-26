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

use turbine_core::config::{Config, StructuredOutputConfig, ToolCallParserKind};
use turbine_core::types::SeqId;
use turbine_core::types::{ExecutionBackend, MemoryKind, Vendor};
use turbine_device::{DeviceInfo, DeviceInventory};
use turbine_kernels::{
    KernelError, KernelMetrics, KernelProvider, KernelRegistry, ProviderId, ShimContext,
    ShimLibrary, cpu_reference_provider, shim_provider,
};
use turbine_kv::metrics::log_pool_startup;
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_model::executor::{self, BatchInput, ModelExecutor, SeqSlice};
use turbine_model::loader::LoadedWeights;
use turbine_model::{
    Architecture, BudgetTerms, ChatTemplate, GenerationConfig, GrammarCompiler, MAX_STAGING_BYTES,
    ModelArchConfig, ModelError, ModelMetrics, SafetensorsIndex, Tokenizer, WeightLoader,
    WeightSlot, available_bytes, check_budget, host_mem_available, llama_slots,
    load_generation_config, load_model_config, olmoe_slots,
};
use turbine_observability::MetricsRegistry;
use turbine_scheduler::SchedulerParams;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

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

/// `model.tool_call_parser`: `llama3_json` for a `LlamaForCausalLM` whose chat template renders
/// `tools` when null. A parser is only usable with a template that renders `tools`: a
/// configured `llama3_json` on any other template resolves to `none` (logged), so `tools`
/// requests get 400 `tools_not_supported` instead of a prompt that silently lacks them.
pub fn resolve_tool_call_parser(
    configured: Option<ToolCallParserKind>,
    architecture: Architecture,
    renders_tools: bool,
) -> ToolCallParserKind {
    match configured {
        Some(ToolCallParserKind::None) => ToolCallParserKind::None,
        Some(kind) if renders_tools => kind,
        Some(kind) => {
            tracing::warn!(
                parser = ?kind,
                "model.tool_call_parser is set but the chat template does not render tools; \
                 tool calling is disabled"
            );
            ToolCallParserKind::None
        }
        None if renders_tools && architecture == Architecture::Llama => {
            ToolCallParserKind::Llama3Json
        }
        None => ToolCallParserKind::None,
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

/// The device-memory backend and kernel providers of `execution.backend`.
pub struct Provider {
    pub mem: Arc<dyn DeviceMemory>,
    pub providers: Vec<Arc<dyn KernelProvider>>,
    pub order: Vec<ProviderId>,
    pub memory_kind: MemoryKind,
}

/// Step 4: load the kernel provider. `hip`: the first loadable library of
/// `ShimLibrary::search_paths` (an ABI, backend or architecture mismatch is fatal, a missing file
/// moves on), then a context on the configured AMD device. `cpu`: the reference provider on host
/// memory sized to host `MemAvailable`, else the configured device's total memory.
pub fn load_provider(
    config: &Config,
    inventory: &DeviceInventory,
) -> Result<Provider, StartupError> {
    let exec = &config.execution;
    match exec.backend {
        ExecutionBackend::Cpu => {
            let device_total = inventory
                .devices
                .iter()
                .find(|d| d.index == exec.device)
                .map(|d| d.memory.total_bytes);
            let host = host_mem_available(Path::new(MEMINFO));
            let capacity = host.or(device_total).unwrap_or_else(|| {
                tracing::warn!(
                    event = "memory_budget",
                    "host MemAvailable and device memory unknown; the cpu backend is unbounded"
                );
                u64::MAX
            });
            let mem: Arc<dyn DeviceMemory> = HostMemory::new(exec.device, capacity);
            tracing::info!(backend = "cpu", capacity, "kernel provider: cpu-reference");
            Ok(Provider {
                mem,
                providers: vec![cpu_reference_provider()],
                order: vec![ProviderId("cpu-reference")],
                memory_kind: MemoryKind::Dedicated,
            })
        }
        ExecutionBackend::Hip => {
            let lib = load_shim(ExecutionBackend::Hip, exec.kernel_library.as_deref())?;
            let device = amd_device(inventory, exec.device.0)?;
            let ctx: Arc<ShimContext> = lib
                .create_context(device)
                .map_err(|e| kernel_error("kernel library context", e))?;
            tracing::info!(
                event = "kernel_library_loaded",
                path = %lib.path().display(),
                backend = lib.backend_name(),
                abi_version = lib.abi_version(),
                build_archs = %lib.build_archs().join(","),
                device_arch = device.arch.as_deref().unwrap_or("unknown"),
                driver_version = device.driver_version.as_deref().unwrap_or("unknown"),
                "kernel library loaded"
            );
            let provider = shim_provider(Arc::clone(&ctx));
            let id = provider.id();
            Ok(Provider {
                mem: ctx,
                providers: vec![provider],
                order: vec![id],
                memory_kind: device.memory.kind,
            })
        }
        other => Err(StartupError::new(format!(
            "execution.backend {} is not available in this build",
            other.as_str()
        ))),
    }
}

fn load_shim(
    backend: ExecutionBackend,
    explicit: Option<&Path>,
) -> Result<Arc<ShimLibrary>, StartupError> {
    let paths = ShimLibrary::search_paths(backend, explicit);
    if explicit.is_none() {
        let order: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        tracing::info!(search_order = %order.join(", "), "kernel library search order");
    }
    let mut misses = Vec::new();
    for path in &paths {
        match ShimLibrary::load(path, backend) {
            Ok(lib) => return Ok(lib),
            Err(e @ KernelError::Load { .. }) => misses.push(e.to_string()),
            Err(e) => return Err(kernel_error("kernel library", e)),
        }
    }
    Err(StartupError::new(format!(
        "kernel library: no loadable libturbine_{}.so: {}",
        backend.as_str(),
        misses.join("; ")
    )))
}

fn amd_device(inventory: &DeviceInventory, index: u32) -> Result<&DeviceInfo, StartupError> {
    let device = inventory
        .devices
        .iter()
        .find(|d| d.index.0 == index)
        .ok_or_else(|| {
            StartupError::new(format!(
                "execution.device {index} is not in the device inventory ({} devices)",
                inventory.devices.len()
            ))
        })?;
    if device.vendor != Vendor::Amd {
        return Err(StartupError::new(format!(
            "execution.device {index} is a {} device; backend hip needs an AMD device",
            device.vendor.as_str()
        )));
    }
    Ok(device)
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
    /// Compiles `response_format` and tool grammars; its token trie is built once, here.
    pub grammar: Arc<GrammarCompiler>,
    /// `structured_output` bounds on those grammars.
    pub structured_output: StructuredOutputConfig,
    /// The resolved `model.tool_call_parser`.
    pub tool_call_parser: ToolCallParserKind,
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
    let registry = KernelRegistry::build(
        provider.providers.clone(),
        &provider.order,
        &executor::requirements(&arch, block_tokens),
        &KernelMetrics::register(metrics),
    )
    .map_err(|e| kernel_error("kernel selection", e))?;

    let scheduler = SchedulerParams::from_config(config, max_seq_len);
    let device_free = provider
        .mem
        .mem_info()
        .map_err(|e| StartupError::new(format!("device memory info: {e}")))?
        .free_bytes;
    let available = available_bytes(
        provider.memory_kind,
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
        config.model.tool_call_parser,
        arch.architecture,
        template.renders_tools(),
    );
    tracing::info!(
        token_trie_seconds = started.elapsed().as_secs_f64(),
        tool_call_parser = ?tool_call_parser,
        "structured output ready"
    );

    let served_name = config
        .model
        .served_name
        .clone()
        .unwrap_or_else(|| default_served_name(dir));
    tracing::info!(
        served_name = %served_name,
        architecture = arch.architecture.as_str(),
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
        grammar,
        structured_output: config.structured_output.clone(),
        tool_call_parser,
        served_name,
        budget,
    })
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

/// The weights `arch`'s executor takes.
fn weight_slots(arch: &ModelArchConfig) -> Result<Vec<WeightSlot>, StartupError> {
    match arch.architecture {
        Architecture::Llama => Ok(llama_slots(arch)),
        Architecture::Olmoe => Ok(olmoe_slots(arch)),
        other => Err(StartupError::new(format!(
            "no weight layout for architecture {}",
            other.as_str()
        ))),
    }
}

/// The one place the server builds a model executor: the architecture's executor for ragged
/// batches of up to `max_batch_tokens` tokens and `max_seqs` sequences over KV blocks of
/// `block_tokens` tokens.
pub fn build_executor(
    arch: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    block_tokens: u32,
    max_batch_tokens: u32,
    max_seqs: u32,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    executor::build_executor(
        arch,
        weights,
        registry,
        mem,
        block_tokens,
        max_batch_tokens,
        max_seqs,
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
/// `turbine_model_load_seconds` and `turbine_model_weight_bytes{format="bf16"}`.
pub fn load(
    prepared: &PreparedModel,
    warmup_token: u32,
    metrics: &ModelMetrics,
) -> Result<LoadedModel, StartupError> {
    let started = Instant::now();
    let arch = &prepared.arch;
    let mem = &prepared.provider.mem;
    let weights = WeightLoader::load(
        &prepared.index,
        &weight_slots(arch)?,
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
    )
    .map_err(|e| model_error("executor", e))?;
    let mut pool = BlockPool::new(prepared.pool, Arc::clone(mem))
        .map_err(|e| StartupError::new(format!("KV block pool: {e}")))?;
    log_pool_startup(&pool);
    warm_up(executor.as_mut(), &mut pool, warmup_token)?;
    let load_seconds = started.elapsed().as_secs_f64();
    // Weights and KV are BF16 (`model.dtype`).
    metrics.record_load(load_seconds, "bf16", weight_bytes);
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
        use ToolCallParserKind::{Llama3Json, None as NoParser};
        // Null: llama3_json only for a Llama whose template renders tools.
        assert_eq!(
            resolve_tool_call_parser(None, Architecture::Llama, true),
            Llama3Json
        );
        assert_eq!(
            resolve_tool_call_parser(None, Architecture::Llama, false),
            NoParser
        );
        assert_eq!(
            resolve_tool_call_parser(None, Architecture::Olmoe, true),
            NoParser
        );
        // Explicit values; a parser needs a template that renders tools.
        assert_eq!(
            resolve_tool_call_parser(Some(Llama3Json), Architecture::Olmoe, true),
            Llama3Json
        );
        assert_eq!(
            resolve_tool_call_parser(Some(Llama3Json), Architecture::Llama, false),
            NoParser
        );
        assert_eq!(
            resolve_tool_call_parser(Some(NoParser), Architecture::Llama, true),
            NoParser
        );
    }
}
