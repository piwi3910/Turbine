//! Model startup (P1 §Interfaces, contract §16.3 steps 4–6, 8 and 10): the kernel provider for
//! `execution.backend`, the model config, tokenizer and chat template, `model.max_seq_len`, the
//! kernel registry, the memory budget (all before the listener binds), then the weight load and
//! the one-token warm-up (after it binds).

use std::fmt;
use std::path::{Component, Path};
use std::sync::Arc;
use std::time::Instant;

use turbine_core::config::Config;
use turbine_core::types::{ExecutionBackend, MemoryKind, Vendor};
use turbine_device::{DeviceInfo, DeviceInventory};
use turbine_kernels::{
    KernelError, KernelMetrics, KernelProvider, KernelRegistry, ProviderId, ShimContext,
    ShimLibrary, cpu_reference_provider, shim_provider,
};
use turbine_model::executor::{LlamaExecutor, SequenceKv};

use turbine_model::{
    BudgetTerms, ChatTemplate, GenerationConfig, MAX_STAGING_BYTES, ModelArchConfig, ModelError,
    ModelMetrics, SafetensorsIndex, Tokenizer, WeightLoader, available_bytes, check_budget,
    host_mem_available, llama_slots, load_generation_config, load_model_config,
};
use turbine_observability::MetricsRegistry;
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
    /// `kv.block_tokens`: the block size of the request's KV pool.
    pub block_tokens: u32,
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

    let block_tokens = config.kv.block_tokens;
    let registry = KernelRegistry::build(
        provider.providers.clone(),
        &provider.order,
        &LlamaExecutor::requirements(&arch, block_tokens),
        &KernelMetrics::register(metrics),
    )
    .map_err(|e| kernel_error("kernel selection", e))?;

    let device_free = provider
        .mem
        .mem_info()
        .map_err(|e| StartupError::new(format!("device memory info: {e}")))?
        .free_bytes;
    let budget = BudgetTerms {
        weights: arch.shape().weight_bytes,
        kv_reservation: SequenceKv::bytes(&arch.kv_layout(block_tokens), max_seq_len),
        workspace: LlamaExecutor::workspace_bytes(&arch, block_tokens, max_seq_len, 1),
        emergency_reserve: config.reliability.emergency_vram_reserve.0,
        available: available_bytes(
            provider.memory_kind,
            device_free,
            host_mem_available(Path::new(MEMINFO)),
        ),
    };
    check_budget(&budget).map_err(|e| model_error("startup", e))?;

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
        served_name,
        budget,
    })
}

/// The loaded, warmed-up executor and the single-sequence KV its requests run on.
pub struct LoadedModel {
    pub executor: LlamaExecutor,
    pub kv: SequenceKv,
    pub weight_bytes: u64,
    pub load_seconds: f64,
}

/// Steps 8 and 10: upload the weights, build the executor for the single-sequence KV `kv` (its
/// layout and `max_seq_len` size the executor) and run one one-token forward on it. Records
/// `turbine_model_load_seconds` and `turbine_model_weight_bytes{format="bf16"}`.
pub fn load(
    arch: &ModelArchConfig,
    index: &SafetensorsIndex,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    mut kv: SequenceKv,
    warmup_token: u32,
    metrics: &ModelMetrics,
) -> Result<LoadedModel, StartupError> {
    let started = Instant::now();
    let weights = WeightLoader::load(index, &llama_slots(arch), &mem, MAX_STAGING_BYTES)
        .map_err(|e| model_error("weight load", e))?;
    let weight_bytes = weights.weight_bytes;
    let block_tokens = kv.view().layout.block_tokens;
    let max_seq_len = kv.max_seq_len();
    let mut executor =
        LlamaExecutor::new(arch, weights, registry, mem, block_tokens, max_seq_len, 1)
            .map_err(|e| model_error("executor", e))?;
    let logits = kv
        .forward(&mut executor, &[warmup_token], &[0])
        .map_err(|e| model_error("warm-up forward", e))?;
    if logits.rows != 1 || logits.data.iter().any(|v| !v.is_finite()) {
        return Err(StartupError::new(format!(
            "warm-up forward: expected one finite logits row, got {} rows",
            logits.rows
        )));
    }
    let load_seconds = started.elapsed().as_secs_f64();
    // Weights and KV are BF16 in Phase 1 (`model.dtype`).
    metrics.record_load(load_seconds, "bf16", weight_bytes);
    tracing::info!(load_seconds, weight_bytes, "model loaded and warmed up");
    Ok(LoadedModel {
        executor,
        kv,
        weight_bytes,
        load_seconds,
    })
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
}
