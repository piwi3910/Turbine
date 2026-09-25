# Turbine — AI Coder Implementation Specification

**Codename:** Turbine  
**Language:** Rust  
**Mission:** Build a reliability-first, distributed LLM inference engine with intelligent hierarchical KV caching and no Python/PyTorch dependency in the production serving path.

## 1. Product principles

Turbine should compete with vLLM/SGLang at the runtime layer, but differentiate around three pillars:

1. **Intelligent KV:** KV is a managed, tiered, distributed resource rather than disposable GPU memory.
2. **Reliability first:** predict pressure and throttle/rebalance before OOM, thermal collapse, allocator failure, or OS instability.
3. **Distributed by design:** topology-aware DP, TP, PP, EP and prefill/decode disaggregation across GPUs and nodes.

Use best-of-breed external GPU kernels. Turbine owns scheduling, memory, KV, placement, reliability, execution planning, distributed coordination, APIs and observability—not reinventing every GPU kernel.

## 2. Non-goals

Do not build a training framework, autograd, a PyTorch replacement, generic tensor framework, or Python-first runtime. Write custom GPU kernels only when profiling proves an existing provider cannot meet requirements.

## 3. V1 scope

- Linux + NVIDIA CUDA; single GPU first while APIs remain multi-device aware.
- Qwen-family decoder model first; prioritize current Qwen MoE architecture.
- BF16 first; Safetensors + Hugging Face `tokenizer.json`.
- OpenAI-compatible API with SSE streaming.
- Continuous batching, separate prefill/decode scheduling, chunked prefill.
- Paged/block KV with GPU + pinned CPU tiers.
- Predictive admission/resource controller.
- Prometheus metrics and structured tracing.

## 4. Architecture

```text
OpenAI API
    │
Request Manager
    │
Admission + Pressure Controller
    │
Scheduler ─────────────── KV Orchestrator
    │                         │
Batch Builder            KV Directory
    │                         │
Execution Planner        GPU/RAM/NVMe/Remote
    │
Model Executor
    │
Kernel Registry
    │
FlashInfer / CUTLASS / cuBLASLt / other providers
    │
GPU
```

## 5. Cargo workspace

```text
turbine/
├── crates/
│   ├── turbine-api/
│   ├── turbine-core/
│   ├── turbine-model/
│   ├── turbine-tensor/
│   ├── turbine-device/
│   ├── turbine-kernels/
│   ├── turbine-kv/
│   ├── turbine-scheduler/
│   ├── turbine-reliability/
│   ├── turbine-distributed/
│   ├── turbine-transport/
│   ├── turbine-observability/
│   └── turbine-server/
├── kernels/{cuda,rocm,include}/
├── benches/
├── tests/
└── docs/
```

Avoid micro-crates without a useful ownership/API boundary.

## 6. Runtime, tensors and kernels

Use Tokio, Axum, Serde, tracing, Prometheus-compatible metrics, `safetensors`, `tokenizers`, `thiserror`, and `smallvec` where useful. Hot paths minimize allocations, copies, locks and synchronization. No Python subprocesses in normal inference.

```rust
pub struct Tensor {
    pub storage: DeviceBuffer,
    pub shape: SmallVec<[usize; 4]>,
    pub strides: SmallVec<[usize; 4]>,
    pub dtype: DType,
    pub device: DeviceId,
}
```

Initial dtypes: BF16, FP16, FP32 where required, INT32/INT64 indices. Later add FP8/FP4 and quantized weights. No autograd/training semantics.

Use capability-based kernel providers for attention, GEMM, MoE, sampling, normalization and quantization. Candidates: FlashInfer, FlashAttention, CUTLASS, cuBLAS/cuBLASLt, DeepGEMM, later ROCm/Intel equivalents.

```rust
pub trait AttentionKernel: Send + Sync {
    fn supports(&self, cfg: &AttentionConfig) -> bool;
    fn execute(&self, ctx: &mut AttentionContext) -> Result<()>;
}
```

Use thin C ABI shims where necessary: `Rust → Turbine CUDA shim → kernel library → CUDA`. Prefer build-time/precompiled kernels. Production serving must not require Python/PyTorch. Review licenses and retain attribution.

## 7. Model, request and scheduler

Parse model config, load safetensors, map/transfer weights, expose layer execution, describe KV layout and parallelization capabilities. Implement one Qwen architecture well before broad model coverage.

```text
HTTP → tokenize → estimate resources → admission → prefix/KV lookup
→ prefill → decode → sample → stream → retain/demote/release KV
```

Every request has a stable ID, lifecycle state, cancellation path and resource estimate.

Scheduler requirements: continuous batching, separate prefill/decode queues, chunked prefill, priorities/fairness, cancellation, backpressure, KV locality and pressure-aware scheduling.

```text
iter 1: A-prefill-chunk + C-decode + D-decode
iter 2: A-prefill-chunk + B-prefill + C-decode + D-decode
iter 3: A-prefill-chunk + B-decode + C-decode + D-decode
```

Eventually adapt chunk size dynamically.

## 8. Hierarchical KV subsystem

KV is a first-class subsystem:

```text
L0 GPU VRAM → L1 pinned CPU RAM → L2 local NVMe → L3 cluster RAM/NVMe → L4 optional external/object tier
```

A deployment may enable only selected tiers.

### KV blocks

```rust
pub struct KvBlock {
    pub key: KvKey,
    pub model: ModelId,
    pub token_range: TokenRange,
    pub format: KvFormat,
    pub size_bytes: u64,
    pub locations: SmallVec<[KvLocation; 3]>,
    pub access_count: u64,
    pub last_access: Timestamp,
    pub ref_count: u32,
    pub priority: KvPriority,
    pub recompute_cost: CostEstimate,
}
```

Blocks may have copies in multiple tiers. Stable identity includes model/version, relevant runtime config, token/block lineage and KV format so identical prefixes can be shared safely.

Each tier exposes capacity, pressure, estimated latency, get/put/evict/contains semantics. Do not force async abstractions into GPU hot paths merely for uniformity.

### Cost-aware eviction

Do not use LRU alone. Consider recency, frequency, prefix popularity, active-session state, size, reload/recompute/transfer cost, request priority and current pressure.

```text
value ≈ reuse_probability × recompute_cost × priority
        ------------------------------------------------
              memory_cost × retrieval_cost
```

The policy must be pluggable and benchmarkable.

### Recompute as a virtual tier

For required KV choose the cheapest safe path: GPU hit, CPU promotion, NVMe promotion, remote retrieval, or recompute.

### Session-aware KV and prefetch

Active session KV should remain hot; idle KV can demote GPU→RAM→NVMe. On likely resume, prefetch asynchronously and overlap transfer with useful compute.

Architecture may later permit BF16 on GPU, FP8 on CPU and compressed/quantized KV on slower tiers, but lossy transforms require quality validation.

## 9. Reliability and pressure controller

Continuously observe and predict resource pressure.

Inputs include GPU VRAM/free/reserved/reclaimable memory, fragmentation, largest usable allocation, KV growth, workspace needs, utilization/bandwidth/temperature/clocks, allocation failures and kernel latency drift; host RAM/pinned memory/swap/CPU; storage queue depth/latency/bandwidth; active/queued requests, prefill/decode backlog, TTFT and ITL.

### Pressure states

```text
GREEN → YELLOW → ORANGE → RED → SURVIVAL
```

Use hysteresis.

- **GREEN:** optimize throughput.
- **YELLOW:** constrain batch growth; proactive KV demotion.
- **ORANGE:** throttle expensive prefill, shrink chunks, slow admission, aggressive KV demotion.
- **RED:** protect active generations; queue expensive new work; reclaim resources.
- **SURVIVAL:** stop admission; release optional buffers; use recovery reserve; drain/recover; resume only after safe pressure returns.

### Predictive admission

Estimate prompt length, cached prefix, new prefill, max output, projected KV growth, workspace cost, execution duration and transfer/recompute cost before admission.

```rust
enum AdmissionDecision {
    Admit,
    Queue { reason: PressureReason },
    Reject { reason: RejectionReason },
}
```

Protect healthy existing generations from oversized new requests.

### Resource reservations and graceful degradation

Explicitly budget VRAM for weights, KV target, execution workspace, runtime overhead and an emergency reserve unavailable to normal scheduling.

```text
reduce batch growth → throttle prefill → demote KV → shrink chunks
→ queue requests → stop admission → drain/recover
```

Allocation failure should enter recovery/survival mode, reclaim resources and retry safely where possible rather than immediately killing the worker. Never retry indefinitely.

### Circuit breaker

```text
HEALTHY → DEGRADED → CIRCUIT_OPEN → DRAINING → PROBING → HEALTHY
```

Trigger from repeated OOM recovery, kernel/device failures, severe latency drift or thermal degradation.

### Self-tuning

Long term, accept objectives such as reliability target and p99 TTFT instead of dozens of manual knobs. Automatic decisions remain observable and overridable.

## 10. Multi-GPU and multi-node

Design for composable DP (replicas/data parallel), TP (tensor parallel), PP (pipeline parallel), EP (expert parallel), and PD (prefill/decode disaggregation).

### Topology graph

Represent node→NUMA→CPU/RAM→PCIe root→GPU/VRAM/NVMe and NICs. Edges expose bandwidth, latency, NUMA distance, P2P, RDMA and vendor-interconnect capabilities. Never model the cluster as a flat GPU list.

### TP / PP / EP

Abstract collectives behind NCCL/RCCL/oneCCL/future backends. Prefer TP over fast local links and DP across slower nodes when the model fits. PP may span devices/nodes when required for capacity or favorable communication cost.

MoE is first-class: expert sharding, placement, routing-aware communication, utilization metrics, future hot-expert replication and dynamic placement.

### Prefill/decode disaggregation

```text
Request → Prefill pool → KV transfer/fabric → Decode pool → tokens
```

### Heterogeneous hardware

Understand mixed NVIDIA/AMD/Intel pools. Do not pretend incompatible devices can participate in every parallel strategy; capability/topology discovery drives placement.

## 11. Cluster-wide KV directory

KV belongs logically to the cluster. The directory answers where a block exists, in what format, cheapest source, whether transfer is in flight, and whether recompute is cheaper.

Placement/routing considers:

```text
compute availability + KV locality + transfer cost + device pressure + SLA/priority
```

Plan pluggable transport for GPU P2P, high-speed vendor links, RDMA, host/GPU paths and TCP fallback.

## 12. Failure model

A node failure is a scheduling event, not a cluster failure. Remove it from admission, identify affected requests/KV, recover KV from replicas/storage or recompute, reroute/restart affected work where safe, and keep unaffected traffic running. Do not claim transparent generation continuation until state preservation truly supports it.

## 13. API

V1:

```text
GET  /health
GET  /ready
GET  /metrics
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

Support SSE streaming. Keep Turbine diagnostics separate:

```text
GET /turbine/v1/status
GET /turbine/v1/devices
GET /turbine/v1/kv
GET /turbine/v1/pressure
GET /turbine/v1/scheduler
```

## 14. Observability

Expose:

- Requests: active/queued/completed/failed/cancelled, admission outcomes, queue time, TTFT, ITL, E2E latency, input/output TPS.
- KV: bytes/blocks and hit rate by tier, prefix reuse, promotions/demotions/evictions/recomputes, transfer latency/bandwidth, prefetch effectiveness.
- Reliability: pressure state/transitions, predicted exhaustion horizon, throttling, allocation failures/recoveries, circuit state, emergency reserve use.
- GPU/runtime: VRAM pools/fragmentation, batch sizes, prefill/decode tokens per iteration, kernel/scheduler timings, utilization/temperature/clocks where available.
- Distributed: node/device health, transport/collective latency, KV locality, reroutes and recovery.

Every automatic control decision should be explainable through structured logs/traces.

## 15. Configuration

Use a validated configuration model with CLI overrides:

```yaml
server:
  listen: 0.0.0.0:8000
model:
  path: /models/qwen
  dtype: bf16
kv:
  block_tokens: 16
  gpu: { enabled: true }
  cpu:
    enabled: true
    max_bytes: 64GiB
  nvme:
    enabled: false
    path: /var/lib/turbine/kv
reliability:
  enabled: true
  emergency_vram_reserve: 2GiB
  adaptive_admission: true
scheduler:
  continuous_batching: true
  chunked_prefill: true
distributed:
  enabled: false
```

Fail early on impossible configurations.

## 16. Security and isolation

Treat model paths, remote KV endpoints and cluster peers as trust boundaries. Add bounded request sizes, timeouts, cancellation, memory quotas and per-tenant accounting hooks. Never deserialize arbitrary executable model formats. Cluster transport should eventually support authentication and encryption.

## 17. Testing strategy

1. Unit tests for allocators, KV metadata/hashing/scoring, admission and state machines.
2. Deterministic scheduler simulation without GPUs.
3. KV tier fault/latency simulation.
4. CUDA integration tests.
5. Golden-output correctness against a reference implementation.
6. Long soak tests.
7. Chaos tests: node loss, transport loss, NVMe slowdown, allocator failure and pressure simulation.
8. Performance regression benchmarks.

Never accept throughput improvements that silently break correctness or reliability.

## 18. Benchmarking

Compare against current vLLM/SGLang using identical model, hardware, dtype, prompt/output distributions and concurrency. Measure TTFT, ITL, request throughput, token throughput, VRAM, cache hit rate, p95/p99 latency, overload stability, recovery from pressure and multi-node scaling efficiency.

A central Turbine benchmark must deliberately overload the engine. Success is graceful queuing/throttling and recovery **without worker death/OOM**.

## 19. Implementation phases

### Phase 0 — skeleton

Cargo workspace, config, tracing/metrics, device discovery, API shell, benchmark harness.

### Phase 1 — single-request correctness

Safetensors/tokenizer, Qwen BF16, CUDA/kernel bridge, single request and streaming. Validate logits/tokens against a reference.

### Phase 2 — serving runtime

Continuous batching, prefill/decode queues, chunked prefill, cancellation, GPU paged KV, OpenAI compatibility.

### Phase 3 — reliability

Memory pools/reservations, pressure telemetry/state machine, predictive admission, throttling, recovery and overload soak tests.

### Phase 4 — KV intelligence

Pinned CPU tier, prefix sharing, cost-aware eviction, recompute-vs-retrieve, session hints and prefetch; then NVMe tier.

### Phase 5 — multi-GPU

Topology discovery, NCCL abstraction, TP, DP and multi-GPU pressure accounting.

### Phase 6 — multi-node

Worker discovery, transport layer, topology graph, distributed placement, cluster KV directory and node-failure handling.

### Phase 7 — advanced distribution

PP, EP, prefill/decode disaggregation, RDMA/direct KV transfer and heterogeneous pools.

### Phase 8 — expansion

AMD/ROCm, Intel where viable, additional model families, quantization, speculative decoding and multimodal as separately scoped work.

## 20. Definition of done — first useful release

The first useful Turbine release must:

- launch as a Rust service without Python/PyTorch;
- load the target Qwen model from safetensors;
- expose OpenAI-compatible streaming inference;
- serve concurrent requests with continuous batching/chunked prefill;
- use paged GPU KV plus a working pinned-CPU tier;
- share reusable prefixes safely;
- expose pressure state and predictive admission;
- survive deliberate overload by throttling/queuing instead of OOMing;
- expose useful Prometheus metrics and diagnostics;
- have repeatable correctness/performance benchmarks against a reference engine.

## 21. Engineering rules for the AI coder

1. **Correctness before optimization.** Every optimized path needs a reference/correctness test.
2. **Reliability before benchmark vanity.** Never trade process stability for a small TPS gain by default.
3. **Measure before optimizing.** Add benchmarks/profiling before specialized hot-path code.
4. **No hidden fallback to Python/PyTorch.** If a feature requires it, mark it unsupported until a native path exists.
5. **No fake distributed abstractions.** Multi-node APIs must model topology, failure and transport explicitly.
6. **No monolithic KV module.** KV directory, placement, tiers, policy, transport and metrics need clear boundaries.
7. **No magic pressure behavior.** Decisions must expose reason codes and metrics.
8. **Bound all queues and caches.** Backpressure is mandatory.
9. **Cancellation must release resources promptly.** Test this.
10. **Unsafe Rust and FFI stay isolated.** Document ownership/lifetime rules around GPU pointers and streams.
11. **Vendor/kernel dependencies remain replaceable.** Core scheduler/KV/reliability logic must not depend on one kernel provider.
12. **Do not over-engineer V1.** Preserve interfaces for the future, but implement the smallest correct vertical slice first.

## 22. Core product statement

> **Turbine is a reliability-first distributed LLM inference engine that treats KV cache, compute, memory and topology as managed resources. It uses the best available hardware kernels while intelligently scheduling, tiering, throttling and distributing inference before resource pressure becomes failure.**
