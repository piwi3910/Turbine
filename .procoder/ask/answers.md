# What a human decided

Written 2026-09-25 17:52 UTC. procoder reads this
file to avoid asking a question twice; edit an answer here to change what
it believes. Reword the question and it will be asked again.

## [decision] decisions.md

Key: 082b2799daec
Question: Phase 0: how to report GB10 unified memory

- Device record carries memory kind `unified`, total = host MemTotal, and a flag that VRAM budgeting must share it with the host (recommended)
- Report memory as unknown in Phase 0

Answer: Mark as unified (user chose the recommended option).## Q9: [decision] decisions.md

## (no longer asked)

Key: 106da4cd1cac
Question: Approve the Phase 0 spec (with Claude-chosen defaults) before planning?

Answer: Approve as written and write the plan (user chose the recommended option).

## [decision] decisions.md

Key: 160da71c7911
Question: P2b: head_dim for tiny test checkpoints

- 128 only (recommended)
- Also compile FlashInfer for smaller head_dims

Answer: 128 only (recommended).

## [decision] decisions.md

Key: 1b420900b6f9
Question: Phase 0: discover AMD GPUs too?

- NVIDIA (NVML) and AMD (amd-smi), both loaded at runtime; missing library → that vendor contributes no devices (recommended)
- NVIDIA only in Phase 0; AMD with the ROCm work in Phase 8

Answer: User: "we will do nvidia and amd" — both vendors discovered in Phase 0.## Q7: [decision] decisions.md

## [decision] decisions.md

Key: 1be2327dcc8e
Question: Accept the defaults the spec-writing agents chose on their own?

- Accept all (recommended)
- Change some (named in chat)

Answer: Accept all (user chose the recommended option).## Q2: [decision] decisions.md

## [decision] decisions.md

Key: 1c5007b7a4a3
Question: P4: KV eviction value — multiply or divide by retrieval cost (TS §8 divides; the spec's own criterion requires multiply)?

- Multiply: cheap-to-retrieve blocks are evicted first (recommended)
- Divide as TS §8 writes, and change the criterion

Answer: Multiply (user chose the recommended option).

## [decision] decisions.md

Key: 30a24a9ef52b
Question: P1: source of Llama config/tokenizer fixtures before gated weights exist

- Ungated mirror unsloth/Llama-3.2-3B-Instruct pinned by revision, sha256-checked against meta-llama later (recommended)
- Wait for the HF token and copy from novanas

Answer: Ungated mirror pinned by revision, sha256-checked later (recommended).

## [decision] decisions.md

Key: 319bbbccbd87
Question: P1: allow a second fixture-only Python script scripts/golden/render_fixture.py?

- Yes, fixture generation only (recommended)
- No, add --render-only to hf_reference.py

Answer: Allow it (recommended).

## [decision] decisions.md

Key: 3246cbf7e6cb
Question: P2b: pinned staging for weight upload on GB10

- Two 32 MiB pinned halves inside turbine_memcpy_h2d for copies ≥1 MiB, no ABI change (recommended)
- Pageable cudaMemcpyAsync only
- Bump the ABI now for pinned allocation

Answer: Staging inside turbine_memcpy_h2d, no ABI change (recommended).

## [decision] decisions.md

Key: 3613ae07e77d
Question: gpt-oss ships MXFP4 weights, outside the chosen quantization scope — how is it served?

- Add MXFP4 to the Phase 8 quantization track (recommended)
- Serve gpt-oss only from a BF16/in-scope conversion
- Drop gpt-oss from the families track

Answer: Drop gpt-oss (user's choice).

## [decision] decisions.md

Key: 3cc915714ef7
Question: Phase 0: behaviour of unbuilt routes

- Inference routes → 503 OpenAI-style error "no model loaded"; `/ready` → 503; unbuilt diagnostics → 501 (recommended)
- Every unbuilt route → 501 Not Implemented

Answer: 503/501 split: inference routes and /ready → 503 model_not_loaded; unbuilt diagnostics → 501 (user chose the recommended option).## Q3: [decision] decisions.md

## [decision] decisions.md

Key: 523a5a82834c
Question: CLAUDE.md duplicates AGENTS.md — how to keep them in sync?

- Replace CLAUDE.md with the required header plus an `@AGENTS.md` import (recommended: one source of truth)
- Keep both full copies and edit them by hand together

**Answer (2026-09-25):** Import AGENTS.md — CLAUDE.md is the header plus `@AGENTS.md`; AGENTS.md is the single source of truth.

Answer: Import AGENTS.md — CLAUDE.md is the required header plus `@AGENTS.md` (user chose the recommended option).## Q2: [decision] decisions.md

## [decision] decisions.md

Key: 575b4c1b35ef
Question: Phase 0: device discovery backend

- NVML loaded at runtime (`nvml-wrapper`); no driver → empty inventory + warning (recommended)
- CUDA driver API (`cudarc`) loaded at runtime
- Stub only in Phase 0; real discovery in Phase 1

Answer: User: develop and test on novanas and the two DGX boxes, which have AMD and NVIDIA GPUs — NVML for NVIDIA, runtime-loaded; see Q6 for AMD.## Q6: [decision] decisions.md

## [decision] decisions.md

Key: 57b7c752f952
Question: Approve the Phase 0 spec (with Claude-chosen defaults) before planning?

- Approve as written and write the plan (recommended)
- Change specific defaults first (listed in the chat summary)

**Answer (2026-09-25):** Approved as written; write the plan.

**Answers log for phase 1–8 spec questions (2026-09-25)**

- Target: BF16 default; small BF16 model, not necessarily Qwen → Llama-3.2-3B dense. User will free the hosts.
- AMD first: Phases 1–2 on R9700 (novanas, to be emptied); NVIDIA/GB10 follows right after Phase 2 behind the same vendor-neutral kernel traits.
- Sparks/production vLLM: always ask the user before any run that needs vLLM moved or memory freed; the user moves workloads.
- Multi-GPU (Phase 5): novanas AMD pair (RCCL).
- Kernel build: prebuilt shared libs (libturbine_hip.so, later libturbine_cuda.so) via CMake, loaded at runtime.
- Golden reference: HF transformers BF16 dumps as committed fixtures; tolerance as proposed (≥14/16 prompts first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats).
- AMD attention: Composable Kernel (ck_tile FMHA); own HIP kernels only after profiling shows gaps.
- Weights: Claude downloads over SSH with the user's HF token (token supplied by the user at that time, never stored in the repo).
- OpenAI breadth (Phase 2): core + tools/tool_choice + response_format JSON schema. Constrained decoding: llguidance.
- KV dtype (Phase 2): BF16 only.
- Phase 2 perf bar: baseline only (record Turbine and, if it runs, vLLM-ROCm).
- GB10 budget (Phase 3): MemAvailable at startup − host reserve, capped by optional device_budget_bytes.
- Pressure thresholds: fixed documented defaults now; derivation deferred to self-tuning.
- Telemetry: /proc + allocator every 100 ms; vendor calls every 1 s.
- KV reservation at admission: worst case.
- Soak: 10 min runs + one 4 h run before Phase 3 exit; ask the user before any run needing hosts freed.
- Sticky GPU fault: open circuit, drain, exit code 3; external supervisor restarts.
- Phase 4 hybrid-model prefix snapshot question: obsolete (Llama-3.2-3B has no linear-attention layers).
- L1 on GB10 (Phase 4): disabled on unified-memory devices (WARN), L0 demotes straight to L2; L1 is real on discrete-VRAM devices (R9700).
- Session hints: OpenAI prompt_cache_key as session id + x-turbine-session-resume-within / x-turbine-session-end headers.
- NVMe tier: in Phase 4, /home/piwi/turbine-kv, cap 64 GiB per host.
- Prefix sharing: global namespace + opt-in per-request cache salt.
- Collectives (Phase 5): one runtime-loaded NCCL-API binding serving RCCL and NCCL.
- Mixed vendor: no mixed-vendor serving at all until Phase 7.
- Phase 6 transport: TCP first; RDMA in Phase 7. Discovery: static seed list + peer exchange + heartbeats.
- Control protocol: Turbine's own versioned postcard messages over turbine-transport.
- Cluster security: pre-shared-key HMAC challenge-response, no encryption.
- Ingress: every node serves the API and forwards/relays.
- KV directory: owner-authoritative, sequenced deltas pushed, lookups are hints.
- Node death mid-request: rerun on another replica only if no token streamed; otherwise end stream with worker_lost.
- KV storage-only nodes: not in Phase 6.
- PD topology (Phase 7): Spark→Spark over RoCE first, then heterogeneous AMD prefill → NVIDIA decode (10 GbE) as a second milestone.
- Cross-vendor KV wire format: sender-native + receiver exact upcast only; lossy conversion refused unless opted in.
- MoE: OLMoE-1B-7B (BF16) arrives in Phase 2; EP built in Phase 7 on it.
- RDMA stack (Phase 7): rdma-core libibverbs loaded at runtime, own RC queue-pair code.
- Phase 8: umbrella spec + one spec per track later. Track order: quantization → speculative decoding → model families → (multimodal out). Intel: nothing until hardware exists.
- Phase 8 families: Qwen3 dense + MoE, Qwen3.5/3.6 hybrid (Gated DeltaNet), gpt-oss, Mistral/Mixtral.
- Quantization: exactly what cached checkpoints use (NVFP4/FP8 mixed precision via modelopt + compressed-tensors, FP8 KV).
- Speculative decoding: separate small draft model of the same family (e.g. Llama-3.2-1B for 3B).
- Multimodal: out.
- Merged ask (P2 bench / P3 soak location / P7 memory): Claude asks the user before any run that needs production workloads moved or memory freed on any host.

Answer: Approve as written and write the plan (user chose the recommended option, 2026-09-25).## Q4: [decision] decisions.md

## [decision] decisions.md

Key: 601360bbe76a
Question: Phase 0: benchmark harness scope

- Load-generator CLI against any OpenAI-compatible endpoint, streaming, reporting TTFT/ITL/throughput/p50/p95/p99 (recommended)
- Criterion micro-benchmark scaffolding only

**Answers (2026-09-25):** crates → only what's used; routes → 503/501 split; benchmark → load generator. Devices → user: develop and test on novanas (AMD R9700 ×2) and dgx-spark / dgx-spark2 (NVIDIA GB10); follow-ups below.

Answer: Load-generator CLI against any OpenAI-compatible endpoint (user chose the recommended option).## Q4: [decision] decisions.md

## [decision] decisions.md

Key: 8a1f120e0105
Question: Phase 8 support matrix: which row covers the CPU reference provider?

- Add vendor `cpu` with one `experimental` row (cpu, *, *, bf16, bf16, none) (recommended)
- Skip support resolution on the cpu backend (fails the status criterion)
- Make the cpu row `supported`

Answer: Add vendor cpu with one experimental row (user chose the recommended option).

## [decision] decisions.md

Key: 8ae79fddc2f3
Question: Plan depth: keep compile-verified literal code in every plan, or write lighter plans?

- Lighter plans: per task — files, interfaces, test names + exact commands, acceptance; code is written only during implementation (recommended: faster, no scratch code)
- Keep full literal, compile-verified code per step (procoder's default; slow, heavy scratch work)

**Answer (2026-09-25):** Lighter plans — per task: files, interfaces, test names + exact commands, done criteria; no code in plans, no scratch code.

Answer: Lighter plans — no code in plans, no scratch code (user chose the recommended option).

## [decision] decisions.md

Key: 99b4c1fff320
Question: Phase 0: how builds and GPU tests run on the lab machines

- Install rustup for user piwi on each box; `scripts/remote-test.sh <host>` rsyncs the tree and runs `cargo test` there including GPU-gated tests (recommended)
- CI runners on the internal-lab org
- Docker build container on each box, no toolchain on the host

Answer: Docker build container (user's choice); on novanas via k3s Job per Q4.## Q8: [decision] decisions.md

## [decision] decisions.md

Key: a5dde3ce819c
Question: P1/P2: kv.block_tokens on R9700 when CK paged kernels need multiples of 128

- Keep 16; Turbine HIP paged kernel for 16, CK when block_tokens % 128 == 0 (recommended)
- Default 128 everywhere

Answer: Keep 16; Turbine HIP paged kernel, CK when block_tokens % 128 == 0 (recommended).

## [decision] decisions.md

Key: b682bcf4c704
Question: Project license

- Apache-2.0
- MIT OR Apache-2.0
- Proprietary / all rights reserved

**Answers (2026-09-25):** AMD → NVIDIA and AMD both in Phase 0; remote test → Docker build container; GB10 memory → mark unified; license → Apache-2.0. novanas SSH works with the existing key (earlier timeout was transient).

Answer: Apache-2.0 (user's choice).## Q11: [spec] phase-0-skeleton

## [decision] decisions.md

Key: ba759f87c2e2
Question: Phase 0: which crates to create

- Only crates with Phase 0 content: turbine-core, turbine-observability, turbine-device, turbine-api, turbine-server, plus a bench crate (recommended)
- All 13 crates from TS §5 as empty stubs

Answer: Only crates with Phase 0 content (user chose the recommended option).## Q10: [decision] decisions.md

## [decision] decisions.md

Key: bc27af249674
Question: P2: OLMoE expert GEMM on gfx1201 (no hipBLASLt grouped kernels in ROCm 7.14.1)

- Per-expert hipBLASLt calls; switch to grouped automatically if a kernel appears (recommended)
- Custom grouped HIP kernel now

Answer: Per-expert hipBLASLt calls (recommended).

## [decision] decisions.md

Key: bcf82bd4363f
Question: Amend turbine-spec.md to match the decisions (AMD first, Llama/OLMoE first, Qwen in Phase 8)?

- Yes, edit TS §3/§6/§19 and note the date (recommended)
- No, leave TS as the original vision; phase specs carry the amendments

Answer: No — keep turbine-spec.md as the original vision; phase specs carry the amendments (user's choice).## Q3: [decision] decisions.md

## [decision] decisions.md

Key: c45a8821bf7a
Question: Phase 0 lab run: free novanas GPUs (held by a kuvryn-ai-workloads pod)?

- User moves the pod before `scripts/lab-test.sh novanas` runs in Phase 0 (recommended)
- Defer the novanas lab check until the host is emptied for Phase 1

Answer: User frees the novanas GPUs for Phase 0; Claude asks right before the lab step (user chose the recommended option).

## [decision] decisions.md

Key: d022a8cdb95f
Question: Phase 0: container runtime on novanas (no Docker; k3s containerd only, root-owned)

- Install Docker Engine on novanas so the same `docker run` path works on all three hosts (recommended)
- Run the build/test container as a k3s Job/Pod with /dev/kfd and /dev/dri mounted
- Rootless Podman on novanas

Answer: Run the build/test container as a k3s Job/Pod on novanas (user's choice).## Q5: [decision] decisions.md

## [decision] decisions.md

Key: ddb5d506dbd0
Question: P3: lock-free latest-value cell and atomic plan snapshot vs "no new runtime dependencies"

- Add arc-swap 1.x (safe, lock-free, tiny) (recommended)
- std RwLock<Arc<T>> (not lock-free, no new dependency)
- Hand-written AtomicPtr cell with unsafe in turbine-device

Answer: Add arc-swap 1.x (user chose the recommended option).

## [decision] decisions.md

Key: f22fefebddde
Question: phase-2b-nvidia: may correctness runs on the Sparks proceed without asking when a MemAvailable pre-check passes?

- Yes for correctness runs that fit current free memory; always ask for benchmarks/soak/overload (recommended)
- No — ask before every Turbine run on the Sparks

**phase-2b-nvidia answers (2026-09-25):** attention FlashInfer; CUDA deps via CMake FetchContent pinned; dense GEMM cuBLASLt; MoE grouped GEMM cublasGemmGroupedBatchedEx (per-expert cuBLASLt fallback); CUDA 13.0 devel arm64 image pinned by digest; GB10 allocation cudaMalloc + pinned staging; Spark correctness runs proceed after MemAvailable pre-check, benchmarks/soak/overload always ask first. Lab hosts: no `docker run` of any image outside the defined lab scripts.

Answer: Yes for correctness runs that fit current free memory; always ask for benchmarks/soak/overload (user chose the recommended option).

## [decision] decisions.md

Key: f3e9836a3a8e
Question: P7: FP8 KV with non-power-of-two scale → BF16: exact or lossy?

- Exact only when every scale is a power of two; otherwise lossy (refused unless allow_lossy) (recommended)
- Always exact, round to nearest-even

Answer: Exact only when every scale is a power of two; otherwise lossy (recommended).

## [decision] decisions.md

Key: f7c532727df4
Question: P2b: model revisions on the Sparks

- Same pinned revisions the Phase 1/2 plans record for novanas (recommended)
- Current main at download time

Answer: Same pinned revisions as novanas (recommended).
