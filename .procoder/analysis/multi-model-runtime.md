# multi-model-runtime

Status: open
Created: 2026-09-26

## Question

Should Turbine become a GPU-owning runtime that starts with devices and no model, loads, stops and places models through a control API, runs several models per GPU with a shared KV arena, per-model quotas and fractional compute, and if so, where in the phase order (after Phase 3, alongside Phases 4–6, after Phase 8, or not at all in-process)?

Source of the idea: the user, 2026-09-26 (last entry of `.procoder/ask/decisions.md`, "Multi-model runtime … write an analysis brief first; decide placement before any spec"). The user also asked why vLLM, llama.cpp and SGLang all run one model per process and whether the idea is feasible.

## What we know

### How the others do it, and why (general knowledge, not checked against current releases)

Every claim in this subsection comes from general knowledge of those projects as of mid-2026. None was checked against their current source or docs for this brief.

- **vLLM** runs one engine and one model per process. At startup it profiles a forward pass, then preallocates `gpu_memory_utilization` × VRAM minus weights and activations as one KV tensor. It captures CUDA graphs per batch-size bucket with fixed device pointers and relies on PyTorch's per-process caching allocator. Multiple models means multiple processes, each with a static VRAM fraction. Within one process the closest features are multi-LoRA (many adapters on one base model, adapters swapped between GPU and CPU) and "sleep mode" (offload weights and drop KV, mainly for RL training loops).
- **SGLang** has the same shape: `--mem-fraction-static` preallocation, one model per server, multi-LoRA, a weight-update API for RL. Many models means many servers behind a router.
- **llama.cpp**: `llama-server` loads one model per process. Multi-model setups use external proxies (llama-swap) that start and stop server processes. Recent versions reportedly add a router mode that also spawns child processes (unverified).
- **Ollama** is the most "multi-model" of the popular tools. It loads models on demand, keeps each for `keep_alive` (5 min by default), caps residency with `OLLAMA_MAX_LOADED_MODELS`, and evicts idle models when VRAM runs short. Each loaded model runs in **its own runner subprocess**. So Ollama is option C below, not a shared-memory runtime.
- **Triton Inference Server**: explicit model-control mode (`POST v2/repository/models/<m>/load|unload`), a model repository, several models and instances per GPU, and a rate limiter. Each backend owns its own memory, and there is no cross-model KV management.
- **KServe ModelMesh** packs many small models into shared serving pods with LRU load/unload across the fleet. It works at the orchestration layer, not inside the GPU.
- **Research systems**: AlpaServe (OSDI '23, statistical multiplexing of models across GPUs), MuxServe (2024, spatial-temporal multiplexing of several LLMs on shared GPUs with a unified KV cache and MPS-style partitioning), ServerlessLLM (OSDI '24, fast multi-tier checkpoint loading), and 2025 work on elastic cross-model KV through CUDA virtual memory (kvcached / Prism). This work shows the idea is real and yields gains for many lightly loaded models. It also shows that nobody mainstream ships it as the default.

Why one model per process is the norm:

1. **Static memory ownership.** Preallocating a KV fraction and capturing graphs against fixed pointers is simple and fast, but it assumes one tenant. Growing or shrinking at runtime means re-capturing graphs and fighting the framework allocator.
2. **Python and PyTorch.** One interpreter, the GIL, one CUDA context, a caching allocator that does not cooperate across models, and `torch.compile` and autotune caches per process.
3. **Failure isolation.** A sticky device error (illegal address, ECC, or a memory fault on ROCm) poisons the context. The only safe recovery is to kill the process. Co-located models would share that fate.
4. **Kubernetes already does placement.** The device plugin hands out whole GPUs (fractions only through MIG, time-slicing configs or MPS). Orchestrators and gateways (KServe, llm-d, Dynamo, LiteLLM-style routers) handle many-model setups by running many processes.
5. **The big deployments do not need it.** A hot model saturates its GPUs, and packing only pays for many models that are each lightly loaded.

### What Turbine already has that the others lack (this repo)

- **No framework allocator.** Every device byte goes through `DeviceMemory` / `DeviceBuffer` (`crates/turbine-tensor`). The KV pool is one allocation (`BlockPool::new` in `crates/turbine-kv/src/pool.rs`) with a free list and refcounts. Phase 3 adds a byte ledger with RAII `Reservation`s per `(device, PoolKind)` (contract §8.2 `Ledger::reserve`).
- **Paged attention takes explicit pointers per call.** `turbine_attention_paged_desc` in `kernels/include/turbine_kernels.h` receives `kv_layer`, a `block_table [num_seqs, max_blocks_per_seq]`, `num_blocks` and `block_tokens` per layer call. Nothing in the ABI assumes one pool or one model per context. `turbine_copy_blocks_desc` addresses blocks as `pool + l * layer_stride_bytes + b * block_bytes`.
- **The fit check is already a pure function.** `crates/turbine-model/src/budget.rs` has `BudgetTerms { weights, kv_reservation, workspace, emergency_reserve, available }`, `check_budget` and `available_bytes(MemoryKind, …)`, computed from safetensors headers before any weight byte is read (`model::prepare` in `crates/turbine-server/src/model.rs`). A `POST /models/check` endpoint is this function behind HTTP.
- **KV identity is already per model.** Phase 4 S-1 roots every `KvKey` in a namespace built from model identity, KV format and cache salt. Two models can share an arena but can never alias each other's KV contents.
- **Co-tenancy hooks exist.** Phase 3 `reliability.memory.device_budget_bytes` ("optional hard cap on everything Turbine claims on a device (co-tenancy)"). Phase 5 S-7 `parallel.allow_device_sharing` and S-8 `SharedDeviceBudget` (two replicas on one device, test `multi_device::tests::shared_device_budget`) already model two workloads splitting one device's budget.
- **The API already speaks lists.** `InferenceBackend::models()` returns `Vec<ModelCard>` (`crates/turbine-api/src/backend.rs`), and `ApiError::model_not_found` exists (404 when `model` is not the served one).
- **Hardware capability, checked read-only on novanas 2026-09-26.** HIP 7.14.60850 at `/opt/rocm/rocm` declares `hipExtStreamCreateWithCUMask` (`include/hip/hip_runtime_api.h:3207`) plus `hipExtStreamGetCUMask`, `hipIpcGetMemHandle` / `hipIpcOpenMemHandle` (line 2708 / 2744), and the VMM calls `hipMemCreate` / `hipMemExportToShareableHandle` / `hipMemImportFromShareableHandle` (lines 9837–9904). Spatial CU partitioning and cross-process device memory sharing therefore exist in the installed ROCm. They were **declared**, not exercised on gfx1201.

### What the current code assumes (the cost of changing it)

- `crates/turbine-server/src/startup.rs` is a linear pipeline for exactly one model: `config::load` → `discover` → `model::prepare(&config, …)` (one `PreparedModel`) → bind → `engine::spawn(prepared, …)` (one engine thread, one executor, one `BlockPool`). `model.path` is required (contract §3.2, P0).
- `model::prepare` sizes the KV pool from **measured device free memory** at startup (`provider.mem.mem_info().free_bytes` → `kv_pool_config` → the remainder after weights, workspace and reserve). With models arriving later, memory must be owned by a ledger, not re-measured. Phase 3 S-2 already moves that way ("plus what Turbine itself already holds for weights").
- `ModelBackend` (`crates/turbine-server/src/backend.rs`) holds one `served_name`, tokenizer, template and grammar compiler. `/ready` is one boolean with one `loading_model` reason. Metrics carry no `model` label.
- `BlockPool` has one `KvLayout` (per model: Llama-3.2-3B 28 layers × 8 KV heads × 128 = 112 KiB/token, a 14 MiB block at 128 tokens; OLMoE-1B-7B 16 × 16 × 128 = 128 KiB/token, a 16 MiB block; Phase 2 spec Data). Its storage is layer-major, and the kernel assumes block stride = per-layer block bytes. **Two models cannot share one `BlockPool` today** without either per-layer page tables or a page size common to both models (see Options A2).
- `execution.decode_graphs` and `execution.gemm_autotune` (on by default, `crates/turbine-core/src/config/mod.rs`) mean each loaded model carries captured graphs (holding device pointers, per the header's graph rules) and autotune results. Both are per model and cost load time and memory.
- The phase specs explicitly exclude this idea today:
  - Phase 3 Out of scope: per-tenant quotas.
  - Phase 5 Out of scope: "Sharing one set of weight buffers between DP replicas", "Changing TP or DP size at runtime".
  - Phase 6 Out of scope: "Dynamic re-planning of model placement at runtime", per-tenant quotas.
  - Phase 8 Out of scope: "Hot-swapping models or serving several models in one process".
- Phase 3 S-12: a sticky device error opens the circuit, drains and **exits the process with code 3** for an external supervisor to restart. In one process, that takes every co-located model down.

### Sizes on the lab hardware (derived from the configs above; nothing was measured)

- One R9700 has 32 GB. Llama-3.2-3B BF16 needs about 6.4 GB of weights and OLMoE-1B-7B about 14 GB (Phase 2 spec). Both fit together on one card with about 8 GB left for KV after the 2 GiB emergency reserve and workspace.
- Loading Llama's weights from host RAM over PCIe takes about 0.3–0.5 s at 15–25 GB/s effective (assumed bandwidth, not measured). From NVMe it takes about 1–2 s. Warm-up, graph capture and GEMM autotune add an unknown amount (see below).
- GB10 (the Sparks) has unified memory, so "weights in pinned RAM" costs nothing there. The Sparks also share their memory pool with production vLLM, which limits how much can be packed on them.

## What we do not know

1. **Real demand.** How many models, of what sizes, with what traffic shape (many cold models, or two warm ones?). This decides whether packing pays at all. _Resolve:_ ask the user for the target workloads (which models, expected QPS per model, acceptable cold-start latency).
2. **Two engines on one R9700.** What throughput and ITL do two concurrent engines get versus driver time-slicing between processes? _Resolve:_ one lab Job with two `turbine-server` processes on one R9700 (each capped with `kv.gpu.max_bytes`), each benchmarked with `turbine-bench --concurrency 8` at the same time, compared with each alone. This is option C's experiment. It is cheap, needs no code, and needs the user's approval (it is not covered by the Phase 2c standing approval).
3. **CU masks on gfx1201.** Do hipBLASLt and Composable Kernel launches on a `hipExtStreamCreateWithCUMask` stream respect the mask, and does throughput scale roughly linearly with CUs? Autotuned GEMM picks assume the full CU count. _Resolve:_ a microbenchmark in the HIP shim test (`hip_ops`) sweeping masks of 16/32/48/64 CUs over the Llama GEMM shapes and paged attention.
4. **Blast radius of a sticky fault on ROCm.** Does a memory fault in one process kill only that process's queues, or does it disturb other processes on the same GPU? And within one process, is the whole HIP context lost? _Resolve:_ a fault-injection kernel (out-of-bounds write) in one of two processes sharing the R9700. The answer decides between one worker per GPU and one worker per model (risks below).
5. **Page-table cost of a shared arena.** Per-layer block tables cost `layers × num_seqs × max_blocks` int32 uploads per iteration, and `layers ×` more refcount bookkeeping, versus one table for a uniform page size. _Resolve:_ measure host-side iteration time in the executor profile (`crates/turbine-model/src/executor/profile.rs`) with per-layer tables on Llama at batch 16.
6. **Cost of a cold load.** What does one load cost end to end: weight read, upload, autotune, graph capture, warm-up? `turbine_model_load_seconds` exists but has no breakdown. _Resolve:_ add timing spans to `model::load` and read them from a `lab-serve.sh` run. This decides whether on-demand swap-in (model tiering) is interactive (under 2 s) or only a batch feature.
7. **Cross-process KV arena.** Does `hipIpcOpenMemHandle` (or VMM shareable handles) work between two containers of one k3s pod on novanas (shared `/dev/kfd`, `hostIPC`)? And does CK paged attention accept a pool that another process maps? _Resolve:_ a two-process test in one Job. This only matters for the per-model-process variant.
8. **NVIDIA partitioning on GB10.** MIG is a datacenter feature and is likely unavailable on GB10. MPS and CUDA "green contexts" (SM partitioning) are the likely equivalents (general knowledge). _Resolve:_ check on a Spark read-only (`nvidia-smi mig -lgip`, the CUDA version) when Phase 2b runs there.

## Options

Common vocabulary for all options:

- **Control plane**: `/turbine/v1/models` routes. `POST` with a model config (path, served name, `max_seq_len`, KV min/max, compute share, priority, device hint) loads a model. `POST …/check` returns fits / does not fit plus the proposed placement and the budget terms, without loading. Stop, start, `DELETE` and list round out the routes. The control plane starts from `devices:` only, with no `model:` section required.
- **Placement**: a bin-packing planner over the Phase 5 topology graph and the Phase 3 per-device ledgers. It uses best fit by bytes, respects TP groups for models that need more than one device, and returns reason codes.

### A — staged, woven into the existing phases (the user's proposed placement)

- **A1: after Phase 3, as a new phase `phase-3b-multi-model`.**
  - Control plane and placement as above, plus dynamic load and unload.
  - Several models per GPU, **each with its own `BlockPool`** carved from a per-model KV quota in the Phase 3 ledger: pools keyed `(device, model, PoolKind)`, a guaranteed minimum plus a cap.
  - A GPU-level fair-share scheduler: one engine per model, each on its own compute stream (one `turbine_ctx` per model). A per-GPU arbiter hands out iteration slots by weighted deficit round robin on measured step time, with priorities and SLO classes feeding the weights.
  - Per-model pressure views, a `model` label on metrics, and per-model `/ready` in `/turbine/v1/models`.
  - Process shape: a control-plane process without a GPU context plus one worker process per GPU (details under Risks). The control plane is the supervisor Phase 3 S-12 assumes, so it restarts a worker that exits with code 3 and reloads that worker's models.
- **A2: with Phase 4, the shared KV arena and model tiering.**
  - L0 becomes one arena per device of fixed-size pages. Each model keeps its own page table and `KvLayout`. Two ways to address pages:
    - (i) Per-layer page tables. `kv_layer` = arena base and one `block_table` per layer. The current ABI already allows this, because every attention call takes its own table.
    - (ii) A common page size. Pick each model's `block_tokens` so its per-layer block is a power-of-two multiple of the page, and allocate aligned runs buddy-style. Llama at 128 tokens is 512 KiB per layer; OLMoE at 128 tokens is 1 MiB, which is 2 pages.
  - Guaranteed minimums and caps per model, both ledger reservations.
  - Cross-model cost-aware eviction: the Phase 4 `EvictionPolicy` scores blocks from all models in one arena, with model priority entering the priority term. Demotion to L1/L2 uses the Phase 4 transfer engine. KV contents stay namespaced per model (Phase 4 S-1), so they are never shared across models.
  - Model tiering: idle models' weights demote to pinned host RAM (ABI v3 pinned memory, R9700 only) or stay in the page cache and NVMe, and swap in on demand behind admission (`Queue { reason: model_loading }`).
- **A3: with Phase 5, fractional compute and multi-GPU placement.**
  - CU-mask streams, as an additive ABI minor revision: a context or stream created with a CU mask. The HIP shim calls `hipExtStreamCreateWithCUMask`; the CUDA shim uses green contexts or MPS where available, or reports unsupported. Both shims bump together (Phase 8 constraint).
  - Placement across the devices of a node, including TP models next to small models on the same devices.
- **A4: with Phase 6, cluster-wide placement.**
  - Membership heartbeats carry per-node model residency and free budget, so the control plane can place and move models across nodes. This lifts Phase 6's "no dynamic re-planning".
- **Cost of A:**
  - About one phase of new work (A1) plus about 20–30 % added scope to Phases 4, 5 and 6.
  - Every later phase must carry a model id from the start: scheduler, pressure document, KV directory, metrics, and the Phase 5 router becomes model then replica.
  - A zero-regression gate is needed for the single-model case.
- **Benefit of A:** the model dimension enters the types while they are still being written, which is far cheaper than retrofitting later.

### B — one new phase after Phase 8

- Build Phases 3–8 single-model as specified, then add a `phase-9-multi-model` that retrofits everything above at once.
- **Cost:** by then scheduler, KV directory, tiers, pressure controller, DP router, cluster protocol and support matrix are all single-model. The retrofit touches every crate and every acceptance test, and the Phase 6 wire protocol needs a version bump.
- **Benefit:** nothing changes in the current plan, and the Phase 2c target (at least 75 % of vLLM-ROCm) and the golden gates stay undisturbed until then.
- **Risk:** the retrofit is large enough that it is likely to be deferred indefinitely.

### C — keep one model per process; the control plane orchestrates processes

- A `turbine-controller` process owns the device inventory, fit check and placement, and exposes the same `/turbine/v1/models` API.
- It spawns one `turbine-server` per model with `--set model.path=… --set reliability.memory.device_budget_bytes=… --set kv.gpu.max_bytes=…`, and pins devices with `HIP_VISIBLE_DEVICES`.
- It proxies OpenAI routes by `model` and implements load, stop and list by starting and stopping processes. This is Ollama's and Triton's shape with Turbine's reliability underneath.
- **Cost:** small. The fit check is `budget.rs` reused, the processes already exist, and the proxy is thin. It can start after Phase 3 without touching Phases 4–8.
- **Loses:**
  - The shared KV arena and cross-model eviction (each process's KV is a static slice).
  - Coordinated fair share (the driver time-slices between processes blindly).
  - Fine-grained quotas below a static byte cap.
  - Model tiering beyond "kill and restart" (a cold start pays the full load).
  - Per-process runtime overhead: context, workspace and emergency reserve are duplicated per model, about 3–4 GB each at the Phase 3 defaults (1 GiB workspace + 1 GiB runtime + 2 GiB reserve).
- **Gains:** the best failure isolation, and it is the experiment that answers unknown 2.

## Recommendation

**Option A (staged), with one change: its first stage starts from C's process shape.** Placement:

1. **Now (before any spec):** run the cheap experiments behind unknowns 2–4 and 6, and ask the user unknown 1 (target models and traffic). They decide whether A1 needs fair-share scheduling in-process or can lean on driver time-slicing, and whether CU masks are worth an ABI revision.
2. **`phase-3b-multi-model`, after Phase 3 closes and before Phase 4.** Scope:
   - Control-plane process with `/turbine/v1/models` (load, check, stop, start, list, delete) and bin-packing placement.
   - One worker process per GPU, supervised by the control plane, hosting several models, each with its own `BlockPool` sized from a per-model quota in the Phase 3 ledger.
   - A per-GPU weighted fair-share arbiter across model engines.
   - Per-model readiness and metrics.
   - A single-model regression gate: Phase 2c throughput within noise and golden unchanged.
3. **Phase 4:** the shared L0 arena with per-model page tables, cross-model cost-aware eviction, and model weight tiering.
4. **Phase 5:** CU-mask fractional compute, only if the unknown-3 microbenchmark shows masks work and scale, and multi-device packing.
5. **Phase 6:** cluster-wide model placement.

Why A over B and C:

- Turbine's distinguishing pieces (own allocator, byte ledger, pointer-explicit paged ABI, model-namespaced KV identity, pressure controller) are exactly what the Python engines lack. That is why they stop at one model per process.
- The model dimension is cheapest to add while the Phase 3–6 types are still unwritten. B pays for it everywhere at once, later.
- C is worth keeping as the first measurement and the fallback, but it gives up the shared arena and coordinated quotas, which are the parts only an in-process runtime can deliver.
- Starting A1 with a separate control-plane process and per-GPU workers keeps Phase 3's "exit 3 and restart" rule intact. It also bounds a fault to one GPU's models, and leaves room to split further (one worker per model) if unknown 4 says a fault on one GPU should not take down the other models there.

Spec amendments the recommended placement requires. Each is to be written when that phase opens, per AGENTS.md "update the spec first":

- **`phase-3-reliability.md`:**
  - S-2 and S-3: the budget and ledger key pools by `(device, model, PoolKind)`, with per-model guaranteed minimum and cap. "Refuse to start when the budget cannot hold weights…" becomes "refuse to load" when a model arrives at runtime.
  - S-12: exit code 3 is read by the control plane as its supervisor.
  - Out of scope: narrow "per-tenant quotas" to exclude only tenant accounting, not per-model quotas.
  - The Phase 3 spec itself stays single-model if 3b follows it; 3b amends it.
- **New `phase-3b-multi-model.md`** (spec and plan), plus contract changes:
  - §3.2: `model.path` becomes optional, and a `models:` list is added.
  - §14 route table: `/turbine/v1/models*`, and `/v1/models` lists every resident model.
  - §16: startup order without a model.
  - §17: a `model` label on per-model metrics.
  - §8: ledger keys.
- **`phase-4-kv-intelligence.md`:**
  - S-4 and S-8: L0 is a shared per-device arena with per-model page tables.
  - S-7: eviction scores across models, with a model priority term and minimum guarantees.
  - Out of scope: remove the per-tenant cache quota exclusion only for per-model quotas.
  - New item: weight tiering for idle models.
  - The "no kernel-level change to attention" constraint holds under A2 (i), per-layer tables. A2 (ii) needs a page-stride field in the ABI.
- **`phase-5-multi-gpu.md`:**
  - S-4: the planner packs several models (bin-packing with reason codes).
  - Out of scope: lift "Changing TP or DP size at runtime" for load and unload of whole models (not live re-sharding).
  - New item: the CU-mask ABI minor revision (contract §9.1), gated on the microbenchmark.
- **`phase-9-multi-node.md`** (written as `phase-6-multi-node.md`; deferred 2026-09-28):
  - S-4 heartbeats carry model residency.
  - S-6 model placement becomes dynamic through the control plane.
  - Out of scope: remove "Dynamic re-planning of model placement at runtime".
- **`phase-6-8-expansion.md`** (written as `phase-8-expansion.md`): remove "serving several models in one process" from Out of scope. Keep speculative decoding's draft model modelled as part of the target deployment, or as a co-resident model under the same quota.

Where it does not pay, stated so the gate stays honest: one hot model that saturates a GPU gains nothing from any of this. The single-model regression gate in 3b must prove the multi-model machinery costs that case nothing measurable.

Main risks A carries into those specs:

- **Failure isolation.** One fault drops every model on that GPU's worker. Mitigate with per-GPU workers now; add per-model workers sharing a KV arena through `hipIpc` / VMM handles only if unknowns 4 and 7 justify it.
- **Noisy neighbours and latency SLOs.** Handle them with the arbiter's weights, per-model admission, and later CU masks.
- **Arena fragmentation across layouts.** Handle it with one common page size or per-layer tables. Graphs stay valid because block tables are data and the arena base never moves.
- **Kernel shape assumptions.** Graphs and GEMM autotune are per model and per batch bucket. They multiply capture memory and load time, and they are tuned for full-GPU occupancy.
- **Security.** Arbitrary `model.path` via HTTP is a trust boundary (TS §16). Control routes need a separate listener or token, an allowlist of model roots, and safetensors only.
- **Operational complexity.** More states (loading, resident, demoted, swapping), more metrics, and harder capacity planning.
