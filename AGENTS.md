# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project state

Turbine is a Rust LLM inference engine (reliability-first, hierarchical KV cache, no Python/PyTorch in the serving path). `turbine-spec.md` is the original product vision; the work is driven by procoder's chain in `.procoder/`:

- `.procoder/specs/phase-*.md` — one complete spec per phase (0, 1, 2, 2b, 3–8). Where a phase spec amends `turbine-spec.md`, the phase spec wins.
- `.procoder/plans/phase-*.md` — one implementation plan per phase; build task by task, test first, gate-clean, one commit per task.
- `.procoder/contract/interfaces.md` — cross-phase names (crates, types, config keys, metrics, C ABI). Its §23 conflict picks are binding.
- `.procoder/ask/decisions.md` — every user decision with its options. Read it before re-opening any design question.

If reality contradicts a spec or plan mid-build, update the spec/plan first and re-run `spec check` / `plan check`.

## Commands

None exist yet. The spec prescribes a Cargo workspace, so once Phase 0 lands the standard forms apply:

- `cargo build --workspace`
- `cargo test --workspace` / single crate: `cargo test -p turbine-kv` / single test: `cargo test -p turbine-kv <test_name>`
- `cargo clippy --workspace --all-targets` and `cargo fmt --all`

Update this section with real commands (CUDA build requirements, benches, integration-test gating) as they are added.

## Architecture (planned — spec §4–§11)

Request path: `OpenAI API → Request Manager → Admission/Pressure Controller → Scheduler ↔ KV Orchestrator → Batch Builder → Execution Planner → Model Executor → Kernel Registry → external kernels (hipBLASLt/Composable Kernel on AMD; FlashInfer/cuBLASLt on NVIDIA) → GPU`.

Workspace layout: `crates/turbine-*` (each crate is created in the phase that gives it content — see the contract §1) plus `kernels/{rocm,cuda}` for the vendor-neutral kernel C ABI shims (`libturbine_hip.so`, `libturbine_cuda.so`, loaded at runtime). Key cross-cutting ideas that span several crates:

- **Turbine owns orchestration, not kernels.** Kernels come from external providers behind capability-based traits (`supports(cfg)` + `execute(ctx)`); FFI goes `Rust → Turbine HIP/CUDA shim → kernel library`. Scheduler/KV/reliability logic must never depend on a specific kernel provider.
- **KV is a tiered, managed resource** (L0 GPU → L1 pinned CPU → L2 NVMe → L3 cluster → L4 external), with recompute treated as a virtual tier. Eviction is cost-aware and pluggable, never LRU alone. Keep directory, placement, tiers, policy, transport and metrics as separate boundaries — no monolithic KV module.
- **Pressure controller** drives behavior via states `GREEN → YELLOW → ORANGE → RED → SURVIVAL` with hysteresis, plus a circuit breaker (`HEALTHY → DEGRADED → CIRCUIT_OPEN → DRAINING → PROBING`). Admission is predictive (`Admit | Queue{reason} | Reject{reason}`) and an emergency VRAM reserve is excluded from normal scheduling.
- **Distributed-aware from day one**: V1 is single-GPU, but APIs stay multi-device aware; topology is a graph (node/NUMA/PCIe/GPU/NIC with edge bandwidth/latency), never a flat GPU list.

Target order (user decisions): AMD first — Phases 1–2 run Llama-3.2-3B-Instruct (BF16) and then OLMoE-1B-7B on the Radeon R9700s in `novanas` (ROCm/HIP); NVIDIA GB10 (`dgx-spark`, `dgx-spark2`) follows in Phase 2b; Qwen families arrive in Phase 8. OpenAI-compatible API with SSE, tools and JSON-schema output (llguidance), continuous batching with chunked prefill, paged KV, Prometheus metrics. Diagnostics live under `/turbine/v1/*`, separate from the OpenAI routes.

Lab hosts are shared: ask the user before any run that needs workloads moved or memory freed; never touch the production vLLM containers on the Sparks.

## Engineering rules (spec §21 — binding)

- Every optimized path needs a correctness/reference test; never trade stability for throughput by default; benchmark before specializing hot paths.
- No Python/PyTorch fallback — mark a feature unsupported until a native path exists.
- Every automatic control decision exposes reason codes, metrics, and structured traces.
- Bound all queues and caches; backpressure is mandatory. Cancellation must release resources promptly, with a test.
- Unsafe Rust and FFI stay isolated, with documented ownership/lifetime rules for GPU pointers and streams.
- Scheduler logic must be testable as a deterministic simulation without GPUs (spec §17).
- Build the smallest correct vertical slice first; avoid micro-crates without a real ownership/API boundary.
