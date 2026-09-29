## CLAUDE.md duplicates AGENTS.md — how to keep them in sync?

- Replace CLAUDE.md with the required header plus an `@AGENTS.md` import (recommended: one source of truth)
- Keep both full copies and edit them by hand together

**Answer (2026-09-25):** Import AGENTS.md — CLAUDE.md is the header plus `@AGENTS.md`; AGENTS.md is the single source of truth.

## Phase 0: device discovery backend

- NVML loaded at runtime (`nvml-wrapper`); no driver → empty inventory + warning (recommended)
- CUDA driver API (`cudarc`) loaded at runtime
- Stub only in Phase 0; real discovery in Phase 1

## Phase 0: which crates to create

- Only crates with Phase 0 content: turbine-core, turbine-observability, turbine-device, turbine-api, turbine-server, plus a bench crate (recommended)
- All 13 crates from TS §5 as empty stubs

## Phase 0: behaviour of unbuilt routes

- Inference routes → 503 OpenAI-style error "no model loaded"; `/ready` → 503; unbuilt diagnostics → 501 (recommended)
- Every unbuilt route → 501 Not Implemented

## Phase 0: benchmark harness scope

- Load-generator CLI against any OpenAI-compatible endpoint, streaming, reporting TTFT/ITL/throughput/p50/p95/p99 (recommended)
- Criterion micro-benchmark scaffolding only

**Answers (2026-09-25):** crates → only what's used; routes → 503/501 split; benchmark → load generator. Devices → user: develop and test on novanas (AMD R9700 ×2) and dgx-spark / dgx-spark2 (NVIDIA GB10); follow-ups below.

## Phase 0: discover AMD GPUs too?

- NVIDIA (NVML) and AMD (amd-smi), both loaded at runtime; missing library → that vendor contributes no devices (recommended)
- NVIDIA only in Phase 0; AMD with the ROCm work in Phase 8

## Phase 0: how builds and GPU tests run on the lab machines

- Install rustup for user piwi on each box; `scripts/remote-test.sh <host>` rsyncs the tree and runs `cargo test` there including GPU-gated tests (recommended)
- CI runners on the internal-lab org
- Docker build container on each box, no toolchain on the host

## Phase 0: how to report GB10 unified memory

- Device record carries memory kind `unified`, total = host MemTotal, and a flag that VRAM budgeting must share it with the host (recommended)
- Report memory as unknown in Phase 0

## Project license

- Apache-2.0
- MIT OR Apache-2.0
- Proprietary / all rights reserved

**Answers (2026-09-25):** AMD → NVIDIA and AMD both in Phase 0; remote test → Docker build container; GB10 memory → mark unified; license → Apache-2.0. novanas SSH works with the existing key (earlier timeout was transient).

## Phase 0: container runtime on novanas (no Docker; k3s containerd only, root-owned)

- Install Docker Engine on novanas so the same `docker run` path works on all three hosts (recommended)
- Run the build/test container as a k3s Job/Pod with /dev/kfd and /dev/dri mounted
- Rootless Podman on novanas

## Approve the Phase 0 spec (with Claude-chosen defaults) before planning?

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

## gpt-oss ships MXFP4 weights, outside the chosen quantization scope — how is it served?

- Add MXFP4 to the Phase 8 quantization track (recommended)
- Serve gpt-oss only from a BF16/in-scope conversion
- Drop gpt-oss from the families track

## Accept the defaults the spec-writing agents chose on their own?

- Accept all (recommended)
- Change some (named in chat)

## Amend turbine-spec.md to match the decisions (AMD first, Llama/OLMoE first, Qwen in Phase 8)?

- Yes, edit TS §3/§6/§19 and note the date (recommended)
- No, leave TS as the original vision; phase specs carry the amendments

## phase-2b-nvidia: may correctness runs on the Sparks proceed without asking when a MemAvailable pre-check passes?

- Yes for correctness runs that fit current free memory; always ask for benchmarks/soak/overload (recommended)
- No — ask before every Turbine run on the Sparks

**phase-2b-nvidia answers (2026-09-25):** attention FlashInfer; CUDA deps via CMake FetchContent pinned; dense GEMM cuBLASLt; MoE grouped GEMM cublasGemmGroupedBatchedEx (per-expert cuBLASLt fallback); CUDA 13.0 devel arm64 image pinned by digest; GB10 allocation cudaMalloc + pinned staging; Spark correctness runs proceed after MemAvailable pre-check, benchmarks/soak/overload always ask first. Lab hosts: no `docker run` of any image outside the defined lab scripts.

## Plan depth: keep compile-verified literal code in every plan, or write lighter plans?

- Lighter plans: per task — files, interfaces, test names + exact commands, acceptance; code is written only during implementation (recommended: faster, no scratch code)
- Keep full literal, compile-verified code per step (procoder's default; slow, heavy scratch work)

**Answer (2026-09-25):** Lighter plans — per task: files, interfaces, test names + exact commands, done criteria; no code in plans, no scratch code.

## Phase 8 support matrix: which row covers the CPU reference provider?

- Add vendor `cpu` with one `experimental` row (cpu, *, *, bf16, bf16, none) (recommended)
- Skip support resolution on the cpu backend (fails the status criterion)
- Make the cpu row `supported`

## Phase 0 lab run: free novanas GPUs (held by a kuvryn-ai-workloads pod)?

- User moves the pod before `scripts/lab-test.sh novanas` runs in Phase 0 (recommended)
- Defer the novanas lab check until the host is emptied for Phase 1

## P4: KV eviction value — multiply or divide by retrieval cost (TS §8 divides; the spec's own criterion requires multiply)?

- Multiply: cheap-to-retrieve blocks are evicted first (recommended)
- Divide as TS §8 writes, and change the criterion

## P1/P2: kv.block_tokens on R9700 when CK paged kernels need multiples of 128

- Keep 16; Turbine HIP paged kernel for 16, CK when block_tokens % 128 == 0 (recommended)
- Default 128 everywhere

## P1: source of Llama config/tokenizer fixtures before gated weights exist

- Ungated mirror unsloth/Llama-3.2-3B-Instruct pinned by revision, sha256-checked against meta-llama later (recommended)
- Wait for the HF token and copy from novanas

## P2: OLMoE expert GEMM on gfx1201 (no hipBLASLt grouped kernels in ROCm 7.14.1)

- Per-expert hipBLASLt calls; switch to grouped automatically if a kernel appears (recommended)
- Custom grouped HIP kernel now

## P1: allow a second fixture-only Python script scripts/golden/render_fixture.py?

- Yes, fixture generation only (recommended)
- No, add --render-only to hf_reference.py

## P2b: pinned staging for weight upload on GB10

- Two 32 MiB pinned halves inside turbine_memcpy_h2d for copies ≥1 MiB, no ABI change (recommended)
- Pageable cudaMemcpyAsync only
- Bump the ABI now for pinned allocation

## P2b: head_dim for tiny test checkpoints

- 128 only (recommended)
- Also compile FlashInfer for smaller head_dims

## P2b: model revisions on the Sparks

- Same pinned revisions the Phase 1/2 plans record for novanas (recommended)
- Current main at download time

## P7: FP8 KV with non-power-of-two scale → BF16: exact or lossy?

- Exact only when every scale is a power of two; otherwise lossy (refused unless allow_lossy) (recommended)
- Always exact, round to nearest-even

## P3: lock-free latest-value cell and atomic plan snapshot vs "no new runtime dependencies"

- Add arc-swap 1.x (safe, lock-free, tiny) (recommended)
- std RwLock<Arc<T>> (not lock-free, no new dependency)
- Hand-written AtomicPtr cell with unsafe in turbine-device

## P0-T7: run the manual turbine-bench check against production vLLM on dgx-spark (10 requests, concurrency 2)?

- Yes, run it now (small read-only load on production vLLM) (recommended)
- Later, when you have moved workloads

**Answer (2026-09-25):** "don't run on the spark, novanas only" — no turbine-bench run against the Sparks.

## Scope of "no Sparks": Phase 0 only, or all phases until further notice?

- Until you say otherwise: no Turbine runs on the Sparks; Spark lab steps (P0 Spark discovery check, P2b, P3 Spark soak, P6, P7 M1) wait (recommended)
- Phase 0 only
- Permanently: re-plan the Spark-based phases onto novanas

## P0-T7 real-stream check with no OpenAI endpoint on novanas

- Defer the check to Phase 1, run against Turbine itself on novanas (recommended)
- Start a vLLM-ROCm k3s Job on novanas with a small model now

**Answers (2026-09-25):** "no Sparks" applies to Phase 0 only — Phase 0's Spark lab checks move to Phase 2b (first phase that runs on the Sparks); later phases use the Sparks as planned, asking before heavy runs. P0-T7 real-stream check deferred to Phase 1, run against Turbine itself on novanas.

## P0-T8: run the Phase 0 lab suite on novanas now?

- Yes: create /home/piwi/turbine-ci and k3s namespace turbine-ci, run one Job with amd.com/gpu: 2 (rust:1.97-trixie, ~10–20 min incl. first build) (recommended)
- Not yet

## Phase 1: standing approval for novanas lab Jobs (HIP library build, GPU op tests, golden runs, serve Job)?

- Yes for all Phase 1 lab Jobs on novanas while its GPUs are free; ask again if anything else holds them (recommended)
- Ask before each Job

## Phase 1: how the Hugging Face token reaches novanas for the gated Llama download

- User writes the token to /home/piwi/.cache/huggingface/token on novanas (chmod 600) after accepting Meta's license; Claude uses it over SSH without seeing it (recommended)
- User pastes the token in chat for this one download

**Answers (2026-09-25):** standing approval for Phase 1 lab Jobs on novanas while its GPUs are free. HF token: the user logs in on novanas themselves (`hf auth login`) after Claude installed uv 0.12.19 + hf CLI (huggingface_hub 2.0.0) for piwi in ~/.local/bin; Claude never sees the token.

## Phase 1 model source while Meta's gate approval is pending

- unsloth/Llama-3.2-3B-Instruct (ungated re-upload, identical weights/architecture/tokenizer, same Llama 3.2 license) — no plan or spec change beyond the repo id; switch back to meta-llama later only if wanted (recommended)
- A different ungated model family (e.g. Qwen2.5-3B-Instruct, Apache-2.0) — re-plan Phase 1 model code (bias terms, template), redo fixtures
- Wait for Meta's approval

**Answer (2026-09-25):** use unsloth/Llama-3.2-3B-Instruct at the plan-pinned revision 006f5dcd1393c3add266de40994ba96225e9689d (ungated; identical weights/architecture/tokenizer) for Phase 1 weights; meta-llama gate request is pending and may replace it later.

## Start pure-Rust parts of later phases now, ahead of phase order?

- Yes: run independent, GPU-free pieces of Phases 3/4/6 (state machines, policies, simulators, transport/protocol) on their own branches in parallel with Phase 1; merged when their phase opens, adjusted to any Phase 1–2 type changes (recommended for throughput)
- No: keep strict phase order

**Answer (2026-09-25):** Yes — run GPU-free parts of later phases ahead on their own branches (runahead/*), merged when their phase opens.

## P1: 16 MB Llama tokenizer.json fixture vs the gate's 5 MB file limit

- Commit it gzip-compressed (2.5 MB, `tokenizer.json.gz`), decompress in tests via a `flate2` dev-dependency; sha256 of the decompressed file checked against the pinned revision (recommended)
- Git LFS for large fixtures
- Raise the gate limit to 20 MB in .procoder/config.toml

**Answer (2026-09-25):** raise the gate limit — `.procoder/config.toml` `max_file_mb = 20`; raw tokenizer.json committed. `.prettierignore` keeps downloaded fixtures and golden files byte-identical.

## Focus: Phase 1 first, or keep running later phases ahead in parallel?

- Phase 1 first: start T14 (sampler) and T17 (server wiring) now against T13's interfaces; let the running run-ahead agents finish but start no new later-phase work until Phase 1 is merged and pushed (recommended)
- Phase 1 only: also stop the running later-phase agents now
- Keep going as now: Phase 1 plus run-ahead in parallel

**Answer (2026-09-26):** Phase 1 first — T14 and T17 start now in parallel with T13; running run-ahead agents finish, no new later-phase work until Phase 1 is merged to main and pushed.

**Applied (2026-09-26):** the user's P2b decision "tiny test checkpoints head_dim 128 only" is applied to the AMD/HIP GPU executor test too (`hip_matches_cpu` failed with NoProvider for head_dim 16; CK FMHA supports head_dim 128 per spec S-7). CPU-only tiny tests keep head_dim 16.

## Golden reference: regenerate the committed Llama reference with FP32 final logits?

- Yes — commit reference.fp32-logits.jsonl as the golden reference (removes BF16 tie-induced token splits; 3/16 → 9/16 pass at 0.15) (recommended)
- No — keep the BF16-logit reference

## Golden logprob bound (0.15 on all top-5 candidates is below the BF16 noise floor: HF-vs-HF BF16 variants differ up to 0.38)

- Keep 0.15 but only for candidates with logprob > −2; tail candidates (< −2) get a looser bound of 0.55 (Turbine max: 0.087 likely / 0.513 tail) (recommended)
- Raise the bound to 0.55 for all top-5 candidates
- Full-FP32 HF reference with a 0.3 bound

## Tiny-model HIP-vs-CPU bound (2e-2) vs CK's BF16 softmax probabilities (measured 0.097)

- Make the CPU reference attention round softmax probabilities to BF16 like CK and HF sdpa, then keep a tight bound (recommended)
- Raise the bound to 0.2

**Answers (2026-09-26):** golden reference = FP32-final-logit HF reference; bound = 0.15 for candidates with reference logprob > −2 and 0.55 for tail candidates (< −2); CPU reference attention rounds softmax probabilities to BF16 like CK/HF sdpa, tiny HIP-vs-CPU test keeps a tight bound.

## Merge Phase 1 into main and push main to github.com/piwi3910/Turbine (public)?

- Yes: fast-forward main to phase-1-single-request and push main (recommended)
- Merge locally only, no push yet

**Answer (2026-09-26):** merge and push — fast-forward main to phase-1-single-request and push main to origin.

## Phase 2 lab work on novanas (standing approval), OLMoE weights, vLLM-ROCm baseline

- Standing approval for Phase 2 novanas lab Jobs (HIP v2 build, GPU tests, serve Jobs, golden --concurrency 16, baseline, overload), same rules as Phase 1 (recommended)
- Ask before each run
- Download OLMoE-1B-7B-0125-Instruct to novanas with the user's logged-in hf CLI and build the OLMoE golden reference there on CPU (recommended)
- The user downloads it
- Record a vLLM-ROCm baseline Job on one R9700 alongside Turbine (recommended)
- Turbine baseline only

**Answers (2026-09-26):** standing approval for Phase 2 novanas lab Jobs while its R9700s are free (stop and ask if another workload holds `amd.com/gpu`; always `--stop` serve Jobs); Claude downloads `allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` (ungated) into `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct` with the hf CLI the user logged in on novanas (the token is never read or passed); the vLLM-ROCm baseline Job runs.

## Performance phase between Phase 2 and Phase 3

**Answer (2026-09-26, user):** after Phase 2 is complete and before Phase 3, a performance-optimization phase runs. Turbine does not need to beat vLLM, but must reach at least 75% of vLLM-ROCm's performance on the same hardware. Recorded vLLM-ROCm reference (rocm/vllm rocm7.14.1 RDNA image, vLLM 0.23.0, one R9700, `turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos`): Llama-3.2-3B-Instruct 738 output tok/s (ITL p50 17 ms, TTFT p50 338 ms); OLMoE-1B-7B-0125-Instruct 535 output tok/s (ITL p50 27 ms, TTFT p50 201 ms). Targets: ≥ 553 and ≥ 401 output tok/s, golden correctness unchanged. Turbine Phase 2 engine at first measurement: Llama 92 tok/s (decode forward ~47 ms at batch 16, ~115 ms host overhead per iteration).

## Phase 2c: lab approval, GEMM autotune default, GPU sampling

- Standing approval for Phase 2c novanas lab Jobs, same rules as Phases 1–2 (recommended) / ask each time
- `execution.gemm_autotune` on by default (recommended) / off by default
- GPU sampling (`execution.device_sampling`) on by default (recommended) / host sampling only

**Answers (2026-09-26):** standing approval for Phase 2c novanas lab Jobs (builds, GPU tests, serve Jobs, Turbine and vLLM benchmark runs; stop and ask if another workload holds `amd.com/gpu`; always stop serve Jobs); GEMM autotune on by default (outputs may differ across restarts at BF16 noise level, within golden tolerance; `false` restores restart-stable output); GPU sampling on by default (seeded sampling reproducible run to run, may differ from host sampling at rare probability boundaries; greedy/golden unaffected).

## Default KV page size (kv.block_tokens): 16 or 128?

- 128 tokens: both paged-attention paths run on Composable Kernel (`fmha_fwd_pagedkv`); measured decode forward at batch 16 ~33 ms → ~17 ms on the R9700; golden 16/16, tiny hip_matches_cpu 3.8e-6 (recommended)
- Keep 16 and tune the Turbine paged kernel (no vendor kernel accepts pages < 128 on gfx1201: CK `fmha_batch_prefill` builds for gfx9 only; pagedkv/splitkv/appendkv need 128-aligned pages)

**Answer (2026-09-26):** 128 tokens — reuse CK. The Turbine 16-token kernel stays only as the fallback for other page sizes. Applied in Phase 2c (after Phase 2 closes). Trade-offs accepted: ~64 tokens of KV wasted per sequence on average, prefix sharing (Phase 4) in 128-token units, larger `copy_blocks` forks.

## Phase 3: RED pressure admission under sustained overload

- Refill finished slots: RED blocks growth but queued requests may replace finished ones (running count never rises; KV bounded by reservation) (recommended)
- Keep RED admit-nothing (as specified; simulator: 10× overload for 600 s served 16 requests, queue drained only by timeouts)
- Refill at a reduced rate

**Answer (2026-09-26):** refill finished slots — in RED, admission may replace completed requests from the queue (no net growth of running requests, KV within the worst-case reservation); Phase 3 spec/plan to be amended when Phase 3 opens (run-ahead branch `runahead/p3-reliability` implements admit-nothing today).

## tool_choice "auto": constrained or free?

- Constrain with an llguidance grammar `start: text | calls` — free text allowed, but once a call starts its name and arguments are schema-enforced (recommended)
- Keep auto unconstrained (vLLM default); invalid arguments possible

**Answer (2026-09-26):** constrain `auto` with the `text | calls` grammar (S-18 amended). Also accepted: constrained JSON allows natural whitespace bounded to 16 characters between tokens (S-17 amended; the compact-only rule degraded Llama's output, e.g. `{"name":": "}`).

## OLMoE golden gate (5/16 with the Llama tolerance)

- Match the router (BF16 router logits, torch top-k tie-breaking), then calibrate OLMoE's tolerance to transformers' own variant spread (sdpa/eager, BF16/FP32) over the 16 prompts (recommended)
- Calibrate only
- Looser fixed gate for MoE

**Answer (2026-09-26):** match the router, then calibrate. Evidence: Turbine's OLMoE semantics match transformers 4.57.1 `modeling_olmoe.py`; the cpu-reference path drifts as far as HIP (p14 likely 0.368 / tail 0.906); transformers against its own reference exceeds 0.15/0.55 when only attention (eager) or precision (FP32) changes, because top-8-of-64 routing flips on BF16 rounding. OLMoE's `tolerance.json` becomes the measured transformers self-spread; the Llama tolerance is unchanged.

## Golden at concurrency 16: strict or batched bound?

- Concurrency 1 strict (full rule, 16/16 within bounds) + concurrency 16 token rule (≥ 14/16 identical prefixes) with a looser batched logprob bound, reported every run (recommended)
- Concurrency 16 strict, best of 3
- Keep concurrency 16 strict

**Answer (2026-09-26):** concurrency 1 is the strict gate; concurrency 16 must pass the token rule with a looser batched logprob bound (batch composition changes GEMM rounding: p14 likely Δ ranged 0.066–0.178 across runs with identical tokens). Applies to Phase 2 acceptance and every later golden run.

## Multi-model runtime (GPU-owning engine, models as workloads): write an analysis brief?

- Yes: a procoder analysis brief with options, risks (failure isolation, noisy neighbours, fragmentation) and a proposed phase placement (control plane + multi-model after Phase 3; shared KV arena, fractional compute via CU masks, model tiering alongside Phases 4–6) (recommended)
- Not now: finish Phase 2 / 2c integration first, revisit later
- Go straight to a spec for a new phase

**Answer (2026-09-26):** write a procoder analysis brief first (options, risks, proposed phase placement); decide placement before any spec.

## Multi-model runtime: placement and first experiment

- Option A, staged, with a control-plane process supervising one worker process per GPU: new Phase 3b (after Phase 3) for the /turbine/v1/models API, fit check, placement, several models per GPU with per-model KV pools and quotas, and a fair-share GPU scheduler; shared KV arena and weight tiering in Phase 4; CU-mask fractional compute and multi-model packing in Phase 5; cluster placement in Phase 6 (recommended)
- Option B: one new phase after Phase 8
- Option C: keep one model per process, orchestrate processes via a control plane only
- First experiment: two turbine-server processes sharing one R9700 (needs lab approval beyond Phase 2c)

**Answer (2026-09-26):** not decided yet — more research needed; continue the original phase plan for now. No experiment on novanas.

## Closing Phase 2: macOS acceptance build and merge

- Run the Phase 2 S-1 acceptance once on the Mac (cargo build/test/clippy/fmt on macOS arm64, ~12 GB target, deleted afterwards) (recommended)
- Accept the novanas (Linux) workspace run instead and amend S-1
- Merge `phase-2c-performance` (Phase 2 + the measured Phase 2c work) into main and push once Phase 2 criteria pass (recommended)
- Merge only Phase 2 (`phase-2-serving-runtime` + its later fixes) and keep Phase 2c on its branch

**Answers (2026-09-26):** S-1 is satisfied by the novanas (Linux) workspace run via scripts/remote-cargo.sh (spec amended; the Mac no longer builds). When Phase 2 passes, merge phase-2c-performance (Phase 2 + the measured Phase 2c work) into main and push; Phase 2c stays open for its remaining tasks.

## NVIDIA (Phase 2b and every Spark / CUDA item) on hold

**Decision (2026-09-26, user):** no NVIDIA work in the plan for now. Phase 2b and every CUDA / DGX Spark item in later phases (Spark lab runs, CUDA kernels, cross-host runs with the Sparks) are on hold, to be revisited after everything works well on novanas. Order after Phase 2 / 2c: Phase 3 next. The run-ahead branch `runahead/p2b-nvidia` is kept as is, not integrated.

## Pluggability beyond models, tool formats and backends

- File issues for quantization formats as modules (one file per weight format; KV formats as a separate registry), speculative decoding as a proposer trait with shared verification, and logits processors + scheduling policy behind traits; and add a "pluggability" engineering rule to AGENTS.md (recommended)
- File the issues only, no AGENTS.md rule
- Neither for now; revisit when Phase 8 track specs are written

**Answer (2026-09-26):** issues + AGENTS.md rule. Filed piwi3910/Turbine #4 (quantization formats), #5 (speculative proposers), #6 (logits processors and scheduling policies), alongside #1 (model families), #2 (prompt/tool-call formats), #3 (backends, card families, kernel providers); AGENTS.md engineering rules gain the pluggability rule.

## Modularity refactor before Phase 3 (issues #1–#6)

Scope:

- Restructure what exists today: model families (#1), prompt/tool-call formats (#2), backends + card profiles + kernel providers (#3), logits processors + scheduling policy (#6); for #4/#5 only BF16 as the first weight-format module, no speculative seam yet (recommended)
- All six issues, including empty seams for quantization and speculative decoding
- Only #1 and #2 now

Run-ahead branches (p3–p8):

- Refactor main first; run-ahead branches are rebased/ported onto the new layout afterwards, each ported when its phase starts (recommended)
- Merge the Phase 3 run-ahead first, then refactor

Where the modules live:

- As modules inside the existing crates (e.g. turbine-model/src/families/, turbine-model/src/formats/, turbine-kernels/src/providers/), no new crates unless a real ownership boundary appears (recommended)
- New crates per extension area (turbine-families, turbine-formats, …)

Process:

- A procoder spec + plan ("phase-2m-modularity"), landed one slice at a time with host tests, lab golden and lab-bench after each slice proving no behaviour or throughput change (recommended)
- Lighter: one agent per issue in parallel worktrees, verified at the end

**Answers (2026-09-26):** scope = existing areas (#1 model families, #2 prompt/tool formats, #3 backends + card profiles + kernel providers, #6 logits processors + scheduling policy; BF16 as the first weight-format module; no speculative seam yet). Refactor main first; run-ahead branches are ported onto the new layout when their phase starts. Modules live inside the existing crates (no new crates without a real ownership boundary). Run as a procoder spec + plan (`phase-2m-modularity`), landed slice by slice with host tests, lab golden and lab-bench after each slice proving no behaviour or throughput change; parallel agents per independent slice.

## Phase 2m modularity: design choices

Kernel providers (today the HIP library chooses hipBLASLt / CK / Turbine kernels internally in C++):

- Selection moves to Rust: the ABI lists each op's implementations (name, supported configs), the Rust registry picks one per op with reason codes, card profiles supply thresholds; one library per backend (recommended)
- Split the HIP library into one shared library per provider (hipBLASLt, CK, Turbine kernels)
- Keep selection in C++; only move thresholds into card profiles passed through the ABI

Model family executors (llama.rs / olmoe.rs are ~55–60% duplicated):

- One shared decoder skeleton with per-family hooks (attention variant such as Q/K norm, FFN variant: dense SwiGLU or MoE); a family file holds config parsing, weight slots and its hooks (recommended)
- Family trait over two separate executors (no deduplication)

Phase 8 run-ahead registry work (runahead/p8-umbrella: architecture registry, Hermes/Mistral tool parsers):

- Use its registry design as input but ship only Llama, OLMoE and llama3_json now; the other families and formats stay on the branch for Phase 8 (recommended)
- Merge its registry and parsers as part of this refactor

Card profiles:

- Declarative Rust profile per card family (gfx1201 first) in turbine-kernels, passed to the library at context creation; CMake builds the architectures the profiles list (recommended)
- Profiles as data files (YAML/TOML) loaded at startup

**Answers (2026-09-26):** kernel selection moves to Rust (the ABI lists each op's implementations and what they support, the Rust registry picks with reason codes, card profiles supply thresholds, one library per backend); families share one decoder skeleton with per-family attention/FFN hooks; the Phase 8 run-ahead registry, families (Qwen3, Qwen3-MoE, Mistral, Mixtral on CPU) and Hermes/Mistral tool parsers are merged in as part of this refactor; card profiles are declarative Rust profiles in turbine-kernels passed to the library at context creation, CMake builds the architectures the profiles list.

## Phase 2m modularity: start implementation?

- Yes: build the plan (17 tasks), parallel agents per lane (A model, B sampler, C scheduler, D kernels) in worktrees, landing one task at a time on `phase-2m-modularity` with gate, GPU suites, golden and lab-bench after each (recommended)
- Yes, but serially (one task at a time, no parallel lanes)
- Not yet — review the spec and plan first

**Answer (2026-09-26):** yes, parallel lanes — agents build lanes A (model), B (sampler), C (scheduler), D (kernels) in worktrees; each task lands one at a time on `phase-2m-modularity` with gate, GPU suites, golden and lab-bench after it.

## Continue through the phases unattended (2026-09-27)

**Decision (user, 2026-09-27):** "continue through the phases, keeping nvidia out of it" while the user is away. Coordinator rules for the unattended run: Phase 2m merges into main and is pushed (approved); Phases 3 onward proceed in order on novanas only (lab Jobs, serve and bench runs under the same rules as the Phase 2 standing approval: novanas only, stop and wait if `amd.com/gpu` is held by another workload); NVIDIA / DGX Spark / CUDA work stays on hold; open design decisions take the recommended option, recorded here as "provisional (coordinator default, pending user review)" and implemented so they can be switched; later phases merge into main locally but are not pushed to the public repository until the user reviews them.

## Phase 3: SURVIVAL liveness fix

- A) On entering SURVIVAL, requeue admitted requests that have not started (no KV written) and drop their reservations, so in-flight work can finish and the pool drains (recommended)
- B) Let in-flight prefills continue in SURVIVAL (changes the spec's SURVIVAL row)

**Answer (2026-09-27): provisional (coordinator default, pending user review) — A.** Implemented behind the recovery controller so B can be switched in; the overload simulation seeds that exposed the gap (seed 6 stuck in SURVIVAL, seed 1 recovering in 64 s against the 60 s criterion) become regression tests.

**Implemented (2026-09-27, branch `phase-3-reliability`, Task 12a):** `reliability.recovery.survival_liveness: requeue_unstarted` (A, default) | `continue_prefills` (B). A requeues at the request's original turn and queue-wait start (`survival_requeue` queue decision; `503 overloaded` if the queue is full). Regression tests `overload_sim survival_liveness_seed_6`, `survival_liveness_seed_1`, `survival_liveness_option_b`, `survival_requeues_unstarted_admitted`. Seed 6 recovers in 42.9 s under A (46.9 s under B); disabling the requeue brings back the stuck state (0.934 of the pool held, never GREEN). Seed 1 never entered SURVIVAL: its 64 s came from the backlog re-escalating YELLOW → RED, fixed by the KV headroom rule below.

## Phase 3: KV headroom in admission (seed 1 recovery)

The overload simulation's seed 1 recovered 64 s after the load stopped (criterion 60 s) without ever reaching SURVIVAL: de-escalating through YELLOW, its growth limit (+1 admitted request per iteration) admitted the queued backlog within seconds, the worst-case reservations lifted `kv_utilization` to 0.92 and the state went back to RED.

- A) KV headroom: with adaptive admission, in YELLOW, ORANGE and RED an admission or refill waits (`kv_reservation`) when its reservation would lift `kv_utilization` past the next state's threshold; GREEN unchanged (recommended)
- B) Slow YELLOW's batch growth (e.g. +1 per second instead of per iteration)
- C) Relax the 60 s recovery criterion

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Seeds 1–12 all recover in 42–49 s; completions change from 137–548 to 247–581 per seed (seed 8: 408 → 346, seeds 10–12 up to 2×); GREEN admission, and so the Phase 2c throughput path below pressure, is unchanged. Spec S-9 amended; test `admission::tests::kv_headroom`.

## Phase 3: device_memory counts idle pre-allocated pools as free

The first 10-minute soak on novanas (`scripts/overload-soak.sh novanas`, 2026-09-27) failed its calibration: every request answered `503 queue_timeout`. The soak config leaves `kv.gpu.max_bytes` at its default (null), so the `kv` pool is the rest of the budget and is allocated at startup. 200 ms after `/ready`, with no load, the log showed `pressure_transition GREEN → RED signal=device_memory value=0.972 threshold=0.95`, and RED admitted nothing. `device_memory` was device used / budget, and the device reports the pre-allocated KV pool and emergency reserve as used whether they hold data or not. The Phase 2c lab configs cap the pool at 8 GiB (about 0.54 at idle), which is why lab-bench never hit this, and the overload simulation never reported device memory.

- A) `device_memory` = (device used − idle pre-allocated bytes) / budget, where idle pre-allocated = free `kv` pool bytes + the emergency reserve while held. Same denominator and thresholds; a full pool reads as before; releasing or re-acquiring the reserve does not move it (recommended)
- B) Cap `kv.gpu.max_bytes` in the soak config (leaves the default config RED at idle)
- C) Measure only the memory outside the pre-allocated pools (weights, workspace, runtime, co-tenants) against its own budget (about 2 GiB of slack on the R9700, so a few hundred MB of transient workspace would move it tens of percent)

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Idle novanas soak startup: 0.972 → about 0.22 (GREEN). Spec signal table amended, plan Task 12b, contract §8.3 row. Tests `signals::tests::device_memory_counts_idle_preallocated_bytes_as_free`, `controller::tests::idle_full_budget_kv_pool_stays_green`; the overload simulation now reports device memory (without A, `ten_x_overload` and the SURVIVAL liveness seeds never return to GREEN; with A, seeds 1–12 still recover in 42–49 s).

## Phase 3: soak stall — drift per iteration, idle floor

With `device_memory` fixed, the second 10-minute soak on novanas (2026-09-27) calibrated (4R = 7.64 req/s) and then stalled. During the overload `/turbine/v1/scheduler` showed 222 waiting, 0 prefilling and 0 decoding. The circuit was DEGRADED (`latency_drift`) from the calibration, the state ORANGE on `queue_fill` 0.87, and the batch growth limit 0. Of the overload requests, 9 completed and 4,534 ended `queue_timeout`. In the cool-down the state stayed ORANGE for all 5 minutes. Three defects:

1. `step_time_drift` divided the iteration time by the batch size. Decode is memory-bound (ITL p50 17.6 ms at concurrency 16, 20 ms at 4), so a batch shrinking from 4 to 1 read as a 4× slowdown and put the circuit in DEGRADED.
2. DEGRADED and ORANGE queue prefills above `large_prefill_tokens` (three quarters of the soak's arrivals). A slot the blocked queue head could not refill was lost, because the next growth limit is measured from the lower admitted count. The count drained to 0, and ORANGE's freeze then held an idle engine behind a full queue whose `queue_fill` kept the state ORANGE.
3. With nothing decoding, the drift window kept its last p95 (1.95 × baseline), above ORANGE's exit threshold (1.9), so the state never de-escalated.

Options:

- A) Drift is the p95 time of a pure decode iteration, not computed while nothing runs. Add a work-conserving floor: while nothing is admitted, below SURVIVAL, the gate admits the first queued request that passes the KV checks whatever its pressure reason (`idle_floor`) (recommended)
- B) A, plus keep a frozen admitted target in ORANGE/RED so deferred refills are not lost (larger change; RED's "never rises between plans" rule, user decision 2026-09-26, would need rewording)
- C) Drop `queue_fill` as a pressure signal (spec table change; does not fix the drain to 0)

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Spec signal table (`step_time_drift`) and batch-growth paragraph amended, plan Task 12c, contract §8.3 note. The simulator now sends the server's `Iteration` circuit events and uses the engine's drift window. With the old formula the three SURVIVAL liveness cases end DEGRADED; without the floor, `degraded_circuit_keeps_serving` idles 4.2 s with requests queued. With A: `ten_x_overload` is unchanged (493 completions, GREEN 42.7 s after the stop), the soak workload completes 842–844 requests (capacity bound 688) and is back to GREEN + HEALTHY in 40–42 s, and seeds 1–12 still recover in 42–49 s.

## Phase 3: soak config max_batch_tokens

With the `device_memory`, drift and idle-floor fixes, the third 10-minute soak on novanas (2026-09-27) served through the overload (1,933 completions; states YELLOW → SURVIVAL → back; GREEN + HEALTHY 59 s into the cool-down) but failed `itl_p99_within_2x`: overload ITL p50 163 ms and p99 545 ms, against a calibration p99 of 185 ms. `scripts/lab/phase3-novanas-soak.yaml` set `scheduler.max_batch_tokens: 8192` while its header says "the Phase 2c lab values", and Phase 2c uses 2,048 (spec S-12 of Phase 2c: 2,048 batch tokens put TTFT at a fifth for the same throughput). With 8,192, four 2,048-token prefill chunks join every overload iteration, and each running sequence waits for them.

- A) Use the Phase 2c value, 2,048, in the soak config (recommended)
- B) Keep 8,192 and relax the ITL criterion
- C) Shrink the prefill budget further under pressure (a throttle-table change)

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Fourth soak: calibration 4R = 8.46 req/s; overload ITL p99 189 ms against 170 ms in calibration (passes); 4,826 × 200, 3,734 `queue_timeout`, 171 `queue_full`, 74 `overloaded`, no incomplete stream; KV idle and the reserve held after the cool-down; reached RED and SURVIVAL. It still failed `green_within_60s` (61 s). A `step_time_drift` spike (≥ 2.0 for two samples) 20 s after the queue emptied, while the last long-context sequences drained, put the circuit in DEGRADED (`latency_drift`). DEGRADED holds the floor at YELLOW until `reliability.circuit.window` (60 s) passes without a trigger, so any drift trigger in the last seconds of the load makes the 60 s criterion unreachable. Open question for the user, not decided here: `step_time_drift` compares raw decode-iteration time with a baseline learned under light load, so a full batch of long contexts (more KV read per step) reads as device degradation. Either normalise drift by the work of the step, or keep drift out of the circuit while pressure is above GREEN, or relax the criterion to `window` + dwell.

**Answer to the open question (2026-09-27): provisional (coordinator, pending user review) — option 1, tried and reverted.** Implemented as b6a398f: drift = each pure decode step's time over a work-cost model's prediction, `a + b · rows + c · context tokens`, fitted by weighted least squares on GREEN + HEALTHY steps. The fifth soak (4R = 8.67 req/s, ITL p99 319 ms against 167 ms) never returned to GREEN in the 5-minute cool-down. During the overload the model predicted the heavy steps at a fifth to a seventh of their time, so the circuit was DEGRADED for 471 of 692 timeline samples (12 in the fourth soak) and opened on `latency_drift` (≥ 4.0). In the cool-down every 16-token circuit probe re-opened it: the 64-step window still held about 48 overload steps at 5–7× the prediction. The model is fitted on calibration steps (at most 4 rows, context growing with rows, heavy-tailed noise), which do not identify the rows and context terms well enough to extrapolate to 20+ long-context rows. Reverted (bf8afed), so the branch is back at the fourth soak's state (all checks but `green_within_60s`, 61 s). Still open for the user: whether to pursue option 1 with per-step telemetry to fit the model (and a window cleared when the circuit starts probing), keep drift out of the circuit while pressure is above GREEN (option 2), or relax the criterion to `window` + dwell (option 3).

**Answer (2026-09-27): provisional (coordinator, under "Continue through the phases unattended", pending user review) — option 2.** Option 1 was tried and reverted (above). `latency_drift` feeds the circuit breaker only while the pressure state is GREEN: the pressure controller owns load and the circuit owns device health. Drift still raises the `step_time_drift` pressure signal in every state and still degrades or opens the circuit in GREEN; the probe and drain logic are unchanged. Spec (circuit transitions), plan Task 12d, contract §8.3 note; test `controller::tests::drift_under_pressure_leaves_the_circuit`. Note from the coordinator: GPU 1's PCIe root port 00:01.1 had fallen back to Gen1 (2.5 GT/s) during soaks 1–4; it was retrained to Gen5 x8 during soak 5.

## Phase 3: per-shape drift baselines (OLMoE landing regression)

Landing the soak fixes (branch `phase-3-reliability-land`, d6d5e1d), the coordinator's lab-bench on GPU 0 measured Llama at 771.9 tok/s (unchanged) but OLMoE at 545.6 tok/s against 617.4 (−12 %). `turbine_pressure_transitions_total` showed GREEN → ORANGE and YELLOW → ORANGE on `step_time_drift`, the circuit went HEALTHY → DEGRADED (`latency_drift`) and back, and admission queued. 915d30b's drift took the raw decode-iteration time against a single light-load baseline. That holds roughly for dense Llama but not for MoE, whose expert GEMMs grow with the rows, so a full batch read as drift.

- A) One calm baseline per shape bucket (exact decoding rows × total context tokens in half-powers of two), judged only against its own bucket once it has enough calm samples; no extrapolation across buckets (recommended; coordinator default)
- B) A work-cost model (option 1 of the soak question: tried and reverted)
- C) Divide by the batch again (the calibration's shrinking batches then read as drift)

**Answer (2026-09-27): provisional (coordinator default, pending user review) — A.** The idle rule, the admission floor and option 2 stay. Spec signal table, plan Task 12e, contract §8.3 note; tests `step_window::tests::moe_full_batch_is_not_drift`, `same_bucket_slowdown_is_drift`, `unseen_shapes_and_prefills_are_not_judged`, `context_buckets`.

## Shorter test cycles

- Tiered: affected-crate gate with nextest (full workspace only at phase end / before merge); GPU suite split into `quick` (ops, tiny_model, golden) and `perf` (serving_mix, forward_profile, decode_forward_timing); lab-bench without its duplicate host-test run, golden c16 opt-in, `--quick` 64-request bench; soaks only at phase exit (recommended)
- The above plus both models benched in parallel (Llama GPU 0, OLMoE GPU 1) — needs GPU 1's PCIe/ASPM fix first
- Only drop the duplicate host-test run from lab-bench

**Answer (2026-09-27):** tiered tests — affected-crate gate with nextest (full workspace at phase end / before merge); GPU suite split into `quick` and `perf` tiers; lab-bench without its duplicate host tests, golden c16 opt-in, `--quick` 64-request mode; soaks only at phase exit.

## Before Phase 5: performance and the OLMoE c16 golden flip

Asked 2026-09-27 (after Phase 4, before Phase 5).

Flake (OLMoE golden `--concurrency 16`, prompts p10/p14 flip with batch composition):

- A) Root-cause, then fix: trace p10/p14 alone vs batched, find the op whose result depends on batch composition, fix it (e.g. fixed reduction order); gate stays strict (recommended)
- B) Batch-invariant mode: every op independent of batch composition, gate runs with it on (costs throughput)
- C) Tolerate near-ties: excuse a flip when the top-2 margin is under a measured bound

**Answer (2026-09-27): A — root-cause, then fix.** The one-retry allowance in the landing chain goes once the fix lands.

Performance work before Phase 5 (multi-select): OLMoE decode host round trip per layer; TTFT / prefill; decode ITL; profile first.

**Answer (2026-09-27): all four — profile first**, then land the top items one at a time (one change, then measure) across decode ITL, TTFT/prefill and the OLMoE decode round trip. Phase 5 starts only when the user says so.

## Phase 4: eviction policy as a registered extension point

The Phase 4 plan (written before Phase 2m) made `kv.policy` a closed enum `cost_aware | lru` matched in `make_policy`; the pluggability rule (decision 2026-09-26) asks for a trait, one file per implementation and a static registry.

- A) `EvictionPolicy: Module` with `cost_aware.rs` and `lru.rs` under `turbine_kv::policy`, a `static` registry (point `eviction_policy`), a conformance suite run by `registry_conformance`, `kv.policy` a `ModuleName` validated by `Config::validate_modules` (exit 2 naming the registered ones) and a `docs/extending/eviction-policy.md` page; weights (`kv.policy_weights`) are passed to `score` so policies stay stateless statics (recommended)
- B) Keep the closed enum and add the registry later

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Contract §11/§24, spec S-7 and the configuration table, and plan Task 6 amended. KV tiers stay a fixed set (L0/L1/L2 are the spec's physical tiers, each with its own sizing keys and transfer paths), so they are not a registry point in Phase 4.

## Phase 4: kernel ABI v2.5 instead of v3

The Phase 4 spec and contract (CONFLICT C-6) planned a major bump, ABI v3, making pinned memory, copy streams, asynchronous copies and events required. Since then Phase 2c added the compute-stream subset of those functions as the optional v2.3 group and Phase 2m the optional v2.4 group, so what Phase 4 still needs is additive.

- A) An optional minor group v2.5 (`turbine_copy_stream_create/destroy`, `turbine_memcpy_async`, `turbine_event_query`, `turbine_stream_wait_event`; `turbine_event_record` accepts a copy stream), resolved only with minor ≥ 5 and the v2.3 group; a library without it runs the KV cache on L0 only with a WARN; `TURBINE_ABI_VERSION` stays 2 (recommended)
- B) The planned v3: every function required, a v2 library refused

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** No breaking change for existing libraries (`libturbine_hip_v23.so` still loads and serves), the names and signatures are the v3 ones so a later major bump only makes them required. Contract §9.1, spec S-5/S-6 and plan Task 13 amended; the CUDA shim's copy (pinned.cu) stays out while NVIDIA is on hold.

## Phase 4: prefix attach before the admission gate, and the KV reservation of shared blocks

Phase 3 puts an admission gate with worst-case KV reservations in front of the scheduler; the Phase 4 plan attaches a request's cached prefix at admission. Where the attach happens decides what the reservation covers and when the client hears the admission decision.

- A) Attach before the gate: the reservation excludes the attached (already resident, shared) blocks, the attached blocks are released on every path that drops an unadmitted request (gate cancel, timeout, circuit rejection, SURVIVAL requeue — which also clears the prefix), and a request whose prefix is still being promoted or computed by another request is held on the engine thread with its admission answer pending (at most the directory's 2 s pending wait plus the copies), so the P3 rule "the decision arrives before any event" holds (recommended)
- B) Attach after the gate admits: the reservation covers the whole prompt (sharers each reserve the shared prefix), no hold, but admission under-uses the pool when many requests share a prefix

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Known limit: blocks attached by requests still waiting in the gate's queue are referenced without a reservation of their own (they are shared prefixes, bounded by the distinct prefixes queued); the `kv_utilization` signal counts reservations, not these. Under overlap scheduling the blocks an in-flight iteration writes are held from its ahead completion until it is collected and committed only on success. Plan Tasks 11 and 15 amended.

## Phase 4: L0 capacity demotion

Phase 3's pressure controller measures `kv_utilization` over worst-case reservations, so a pool full of cached (finished, unreferenced) prefix blocks stays GREEN and the controller never asks for demotion; allocations then reclaim cached blocks by dropping them, and L1/L2 stay empty (seen on novanas: 400 filler requests, 0 demotions).

- A) The orchestrator keeps headroom by capacity: while referenced plus cached blocks exceed 0.70 of the pool (the `kv_utilization` YELLOW threshold) and some are cached, it demotes the lowest-valued cached blocks down to 0.70 (reason `capacity`), before each plan; the controller's reclaim stays as is (recommended)
- B) Count cached blocks in `kv_utilization` (the controller would throttle admissions for reusable cache)
- C) Demote synchronously inside allocation (blocks the engine on copies)

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** `turbine_server::kv_orchestrator::CAPACITY_DEMOTE_AT`; spec S-8's "an allocation needing blocks" path.

Free accounting with Phase 3's `device_memory` fix ("device_memory counts idle pre-allocated KV and reserve as free"): cached-but-unreferenced L0 blocks are not covered by any reservation, so they fall in the `kv` pool's available bytes that the controller subtracts from used device memory — they count as free, like `kv_utilization` counts them and like the next allocation treats them (it reclaims them). The engine's `free_kv_blocks` for the exhaustion horizon is `BlockPool::available_blocks()` (free plus cached unreferenced) for the same reason. Known skew: prefix blocks attached by running requests are referenced but not reserved (decision "Phase 4: prefix attach before the admission gate"), so they too read as free to `device_memory`; bounded by the distinct shared prefixes in use.

## Phase 4: capacity demotion only for blocks with reuse evidence

The coordinator's bench of the Phase 4 tip (fbc9e1e) on GPU 0 showed an 8 % Llama throughput regression on a no-reuse workload (200 random 512-word prompts, c16): 708.1 tok/s and TTFT p50 229 ms against Phase 3's 770.5 / 195; 1,214 L0→L1 demotions (129 s of copy time) for 1,920 cached prompt tokens in total, and the engine's `schedule` stage at 4.03 s against 0.03 s. With `kv.cpu.enabled=false` it was 768.0 / 196. The orchestrator's capacity demotion (decision "Phase 4: L0 capacity demotion") copied every finished one-off request's blocks: the cost-aware value of a never-hit block is still positive, since the prefix-popularity term counts the one child every chain block has. Each turn also scanned the whole directory, about 400 µs with L1 full, plus the reclaim-order refresh.

- A) Reuse-evidence gate on capacity demotion: a block is copied down by capacity only if it was hit at least once since it was written, belongs to a session (`prompt_cache_key`), or is a shared prefix (≥ 2 cached children); others stay cached in L0 for allocation to reclaim at no cost. Capacity demotion and the reclaim-order refresh run at most every 50 ms, scan the L0 index rather than the whole directory, and move at most 32 blocks per run. Pressure reclaim (the Phase 3 controller) stays value-ordered and ungated. Policy-independent and deterministic (recommended)
- A') The same gate on pressure reclaim too, freeing evidence-free blocks (`no_reuse`): tried, and `turbine-bench kv-sim` `mixed` went from 0.76 to 0.93 × LRU (fails the ≤ 0.9 acceptance), because first-use blocks (a system prompt's first computation, reused only after it was demoted) are lost
- B) A `kv.demote_min_value` default > 0: the cost-aware value spans orders of magnitude with block size, depth, prefill rate and tier capacity, so no single threshold separates one-off blocks from reusable ones, and it would not apply to `lru`
- C) Change the cost-aware reuse term (e.g. count only branching prefixes): it only helps `cost_aware`, and it changes every policy test and the kv-sim margins

**Answer (2026-09-27): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** `turbine_kv::hierarchy::{has_reuse_evidence, CAPACITY_BATCH}` and `turbine_server::kv_orchestrator::HOUSEKEEPING_INTERVAL`. Host profile (release, 372-block L0 holding 360 one-off cached blocks): the old full L0 victims scan cost 103 µs per call and the reclaim-order refresh 115 µs, both every turn and more with L1 in the directory. The gated capacity scan costs 22 µs per call and runs at most every 50 ms. Test `hierarchy::tests::capacity_demotion_needs_reuse_evidence` covers a one-off block (not copied by capacity), a session block (copied), a re-used block (copied) and pressure (ungated). Tests that force demotion and expect copies give their blocks evidence: a second run in the simulator, `hierarchy`, `api` and engine-loop tests, and session keys for A and the fillers in `kv_gpu`.

## Phase 4: pressure reclaim copies only blocks with reuse evidence, bounded in flight

The coordinator's 10-minute overload soak on main 186f439 (GPU 0, `scripts/lab/phase3-novanas-soak.yaml`, L1 on by default at 64 GiB) failed `itl_p99_within_2x`: calibration ITL p99 176 ms, overload 393 ms, 97.7 output tok/s, 1,010 requests ok. With `kv.cpu.enabled: false` it passed at 173 / 205 ms, 185.2 tok/s, 1,339 ok. The soak's prompts are all distinct (`--prompt-words-range 64..6000`, seeded per request), so no block is ever reused. Its timeline is driven by `queue_fill` in both runs (`kv_utilization` averages 0.02 at RED, since cached blocks count as free), but with L1 on the run spends 359 samples at RED against 196 without.

Mechanism. On every 100 ms tick at ORANGE, the Phase 3 controller calls `free_unreferenced(0.82)` and `demote(0.82)`; at RED it calls `free_unreferenced(0.0)`. `KvHierarchy::apply_reclaim` mapped both calls to `demote_to(.., Pressure)`, which copies every unreferenced L0 block to L1 in value order, whether or not it has reuse evidence. A block is only freed once its copy completes. So at RED every finished request's blocks were copied down, each pinning its L0 block until the copy was done. Without L1 the same call drops them immediately. Evidence:

- Host reproduction: `kv_sim one_off_overload_does_not_demote`, before the fix. Its workload is 360 one-off requests into a 64-block L0, with L1 copies costing about 3.9 ms against a 5 ms step, as on the R9700.
  - RED: 2,186 L0→L1 copies (every full block written), up to 39 started in one iteration, and up to 57 of the 64 L0 blocks pinned by in-flight copies at once.
  - ORANGE: 969 copies, up to 12 blocks pinned.
  - Drain time: 6.39 s at RED and 6.30 s at ORANGE with L1, against 6.17 s without.
- Lab measurement (`turbine-kernels --test lab demotion_host_cost`, novanas, 32 Llama blocks of 128 tokens = 14.7 MB). The orchestrator copies a block with one `copy_async` per layer; each copy carries a compute-stream fence, a wait and a completion event.
  - Enqueuing one block costs 115–420 µs of host time on the engine thread; each `poll` costs 0.17 µs.
  - The per-layer copies reach 5.3–6.2 GB/s. The same bytes as one copy per block enqueue in 5 µs and reach 12.0 GB/s.
  - With `kv.transfer.max_inflight_bytes` at 1 GiB, up to 73 blocks start in one pump, which is 8–31 ms of host time in a single turn on an idle GPU. That turn's cost lands in the next turn's `schedule` stage (`turbine_engine_iteration_seconds{stage="schedule"}`). The copies also compete with the forward pass for PCIe and the device.

Options:

- A) Pressure reclaim follows the evidence gate, bounded in flight. `free_unreferenced` (ORANGE and above) drops blocks without reuse evidence immediately, exactly as without a lower tier. `demote` (YELLOW, and ORANGE's second call) leaves them cached in L0, where the next allocation takes them at no cost. Blocks with evidence are still copied down in value order. Any L0 copy, whether from capacity or pressure, starts only while fewer than 32 are in flight (`DEMOTION_INFLIGHT`), so the copy stream paces the work. The offline simulator's direct `demote_to(.., Pressure)` stays ungated, which keeps the kv-sim bounds (recommended)
- B) Only pace the pressure path (a per-call or per-second cap) and keep copying one-off blocks: less engine time per turn, but every one-off block still costs a copy and pins its L0 block, and the PCIe traffic stays
- C) Move copy submission and polling to a helper thread: removes the host time from the engine thread, but the copies, the pinned L0 blocks and the PCIe contention remain; a larger change to the copy-stream ownership (`ShimContext` is driven from the engine thread)
- D) Gate all `Pressure` reclaim inside `demote_to` (A' of "capacity demotion only for blocks with reuse evidence"): fails kv-sim `mixed` (0.93 × LRU)

**Answer (2026-09-28): provisional (coordinator default under "Continue through the phases unattended", pending user review) — A.** Code: `turbine_kv::hierarchy::{KvHierarchy::pressure_reclaim, DEMOTION_INFLIGHT}`. `apply_reclaim` now runs the free request before the demote request.

What it gives up: at ORANGE and above, a block first computed without a session or a second sharer (a system prompt before its second use, turn 1 of a conversation without `prompt_cache_key`) is dropped rather than kept in L1. That is exactly what happens with L1 off. At YELLOW such blocks stay cached in L0. Session blocks, hit blocks and shared prefixes still move to L1.

Tests:

- `kv_sim one_off_overload_does_not_demote`: at ORANGE and RED with L1 on, 0 copies, 0 pinned blocks, the same drain time as with L1 off (6.17 s), and 2,229 blocks dropped for `pressure` at RED.
- `hierarchy::tests::pressure_reclaim_drops_one_off_blocks_and_bounds_copies`:
  - `demote` copies 36 re-used blocks at most 32 at a time and leaves the one-off blocks cached;
  - `free_unreferenced` then drops the one-off blocks at once, with no copy.
- `kv_sim demotion_under_pressure` and `cancellation_releases_kv` still pass. Their `reuse` helper now sends two extra tokens instead of one: since 3ff23c2, prefix reuse leaves two prompt tokens to prefill, so with one extra token the last full block was never hit and had no evidence.
- The kv-sim bench tests (`cost_aware_beats_lru`, `mixed` ≤ 0.9 × LRU) are unchanged and pass.

Follow-up, not in this change: copy a block with one `copy_async` per contiguous run instead of one per layer. That would need a block-contiguous L0 layout or a batched copy call in the ABI. The lab measurement shows about 2× the bandwidth and 20–80× less enqueue time per block.

## Pre-Phase-5 perf items (from the 2026-09-27 profile)

Profile: branch `perf-profile` 5db197c, `.procoder/perf-profile-2026-09-27.md`. Options (multi-select):

- #1 Per-card GEMM algorithm table (pin the best hipBLASLt algorithm per shape; `execution.gemm_autotune` is read by nothing today): ~+7 % Llama, ~+2 % OLMoE (recommended first: changes every GEMM's rounding)
- #4 + #5: parallel `logits_reduce` (+1.4 % Llama); OLMoE small-m down-projection retune (~+2 % OLMoE, sweep first)
- #3 GQA split-context decode attention as a registered kernel implementation: ~+4 % Llama, more at long context
- #2 Grouped MoE prefill kernel as a registered implementation: OLMoE TTFT −25–30 %

**Answer (2026-09-27): all four**, landed one at a time in the order #1, #4, #5, #3, #2, each with `lab-bench --golden16` on both models before the next (bounds as in the Phase 2m chain). Phase 5 waits for the user.

## Kernel reuse policy (2026-09-27)

The user asked why we are writing our own kernels when the spec says Turbine reuses existing kernels (vLLM, SGLang and others). State then: GEMM on hipBLASLt, attention and RMSNorm on Composable Kernel. Own HIP kernels: the MoE grouped and small-m tiers (plus the WMMA decode kernels of the OLMoE flip fix), `logits_reduce`, elementwise ops and the fallback paged attention. Reason: vLLM's and SGLang's fast AMD kernels (AITER, CK fused MoE) target CDNA (gfx942/gfx950); on RDNA4 (gfx1201) vLLM falls back to Triton, which JIT-compiles through Python.

- A) Reuse first, own last: before writing any kernel, evaluate CK, vLLM/SGLang kernels and llama.cpp's HIP kernels on gfx1201 as registered implementations; write our own only if none works or all are clearly slower, and record the evaluation (recommended)
- B) As A, plus vLLM Triton kernels compiled ahead of time to HSACO (Triton/Python in the build only), loaded through the shim
- C) Own kernels are fine where the profile points

**Answer (2026-09-27): A — reuse first, own last.** No Python in the build (B not taken). Applies to perf items #2 (MoE prefill: CK `ck_tile` fused MoE, vLLM/SGLang kernels, llama.cpp `mul_mat_id`) and #3 (decode attention: CK split-KV / paged decode FMHA, llama.cpp flash attention) before any own kernel, and to every new kernel gap. The existing own kernels (MoE tiers, `logits_reduce`, elementwise, fallback paged attention) get the same evaluation when their area is next touched.

## OLMoE c16 flip: root cause and fix (follows "Before Phase 5", option A)

Found 2026-09-27 on branch `fix/olmoe-c16-flip` with two lab probes: `turbine-kernels --test hip_batch_invariance` (one row alone vs inside batches, per op) and `turbine-model --test batch_invariance` (p10/p14 teacher-forced alone vs in prefill, decode and mixed batches, traced op by op).

Root cause 1 (fixed): the gfx1201 card profile runs `moe_experts` on the scalar small-m kernels up to 512 routed rows (64 tokens) and on the grouped WMMA kernels above. The two sum every gate/up/down element in a different F32 order, so a row's MoE output depended on its batch's row count: a decode alone or among decodes took small-m, the same decode next to another request's prefill chunk took WMMA. `moe_out` changed by one BF16 ulp on 1–75 of 2,048 elements (≤ 3e-5), which moved later layers' routing and the logits by up to 2.3 (32 of 32 p14 decode steps next to a prefill chunk; small-m vs WMMA on a heavy-tailed 65-token batch: 426 of 133,120 outputs differ). Decode-only batches of 1–17 sequences, CK paged attention (decode/prefill kind, neighbours, 2,004-token neighbour), `moe_route`, the fused Q/K/V GEMM below ~2,000 rows and the LM head were bit-invariant.

Root cause 2 (open, handed to the per-card GEMM table, perf item #1): hipBLASLt's dense GEMMs are not row-invariant across m. The router GEMM (n=64, F32 out) gives other F32 bits for rows past the first 16 of a batch of more than 16 rows (|Δ| ≤ 3.3e-6, absorbed by the BF16 rounding of the router logits in every probe run so far); the O projection gives one-ulp BF16 flips on tail rows of batches ≥ 129 rows; the fused Q/K/V GEMM on the tail rows of a 2,048-row batch. A prompt prefilled behind p09's 2,004 tokens gets logits up to 0.54 apart. Pinning the heuristic's m=2,048 algorithm for every m (an experiment) made Q/K/V invariant but not the O projection or the router, so the table must pick, per shape, an algorithm verified row-invariant for every m (no runtime split-k), not only one fixed algorithm; `hip_batch_invariance gemm_rows_are_batch_invariant` is its acceptance check (`PENDING_GEMM_TABLE` lists the shapes it reports without asserting until then), and `batch_invariance` marks the long-prefill scenario pending likewise.

- A) The small-m tier runs the grouped WMMA kernels with 16-row tiles for shapes the WMMA path serves (hidden and inter multiples of 64): every output is the same WMMA chain in both tiers, bitwise equal row by row; other shapes keep the scalar kernels (chosen)
- B) Drop the small-m tier (WMMA 64-row tiles for every batch): same numerics, 4× the wasted WMMA work on decode
- C) Scalar small-m for every batch: prefill several times slower

**Chosen (2026-09-27): provisional (agent decision, pending user review) — A.** Consequences: OLMoE decode numerics change once (small-m → WMMA order); served golden c1 goes from 16/16 to 15/16 (p14 diverges at position 14, reference margin 0.553, just above the 0.5 excuse; the same divergence the old kernels showed in 2 of 8 c16 runs), still passing the gate (≥ 14). Decode throughput of the new small tier is not measured (no perf runs on this branch): the coordinator benches OLMoE on GPU 0 against main. Served golden on GPU 0 (phase2c config, 8 × c16): every run PASS both before and after; p10/p14 FAILs 2 + 2 before, p14 6 after (the c1 result), but 14 prompts still vary run to run (their prefills land at the tail of 2,048-row batches behind p09, the shape the probe shows root cause 2 in; not yet proven end to end), so the landing chain keeps OLMoE c16's one retry until the GEMM table makes the dense GEMMs row-invariant.

## OLMoE c16 flip fix: recovering the decode speed

The first fix (small-m tier on 16-row LDS WMMA tiles, a776ef3) cost OLMoE −14.1 % at concurrency 16 in the coordinator's lab-bench on GPU 0 (528.2 vs 614.6 tok/s, ITL p50 28.4 vs 24.0 ms); the landing bound is −3 %. The numerics property stays: a row's MoE output must not depend on its neighbours.

- A) Tune the small tier's WMMA kernels, keeping the grouped tier's per-element chain: dedicated decode kernels, one wave per 16 columns of an expert, operands straight from global memory in the WMMA lane layout, 8 k slices per load batch (chosen)
- B) Choose the tier by row kind (decode rows always small-m scalar, prefill rows always WMMA), splitting mixed batches into two routing + expert calls: keeps the scalar decode speed, but needs a decode/prefill flag the executor does not have (a 1-token prefill chunk looks like a decode, and chunking depends on the batch), two launches per mixed layer, and the scalar kernel is slower than WMMA at 48–64 decodes
- C) Accept the regression

**Chosen (2026-09-27): provisional (agent decision, pending user review) — A.** Kernel microbenchmark (`hip_batch_invariance moe_decode_tier_timings`, GPU 0, OLMoE, uniform routing), µs per `moe_experts` call, pre-fix scalar / 16-row tiles / decode kernels: 1 token 236 / 328 / 267; 4: 633 / 852 / 714; 8: 940 / 1,196 / 1,004; 16: 1,152 / 1,441 / 1,189 (+3.2 % vs scalar); 32: 1,473 / 1,753 / 1,440; 64: 1,948 / 1,778 / 1,460. Load-batch sweep: 4 slices ~480 GB/s, 8 ~550, 16 on the down projection ~520, non-temporal loads ~360; 1–4 waves per block equal. Served tok/s is the coordinator's lab-bench to confirm; expected within the bound at c16 (MoE ≈ 71 % of a decode step, +3 % on it). The tier bound stays at 512 routed rows (the small tier is faster than grouped WMMA up to at least 96 tokens). B stays the fallback if the served bench still misses the bound.

## Pre-Phase-5 #1: what `execution.gemm_autotune` means with a tuned GEMM table

The 2026-09-27 profile found `execution.gemm_autotune` read by nothing: the Phase 2c plan's first-use autotuning (S-8a) was never built, so every GEMM ran hipBLASLt's first heuristic answer. Item #1 adds a per-card table of pinned hipBLASLt solutions, measured offline (`kernels/rocm/tuning/<arch>/gemm.tsv`, `turbine_gemm_tune`) and compiled into the HIP library.

- A) Keep the key; it switches the table: `true` (default) runs each shape the table covers on its pinned solution, `false` runs the first heuristic answer for every shape (the pre-table path, for A/B and bisecting). The server passes it through the existing kernel ABI v2.1 option `TURBINE_OPTION_GEMM_AUTOTUNE`, which the library now implements; no ABI change, no config migration (recommended)
- B) Deprecate `gemm_autotune` (accepted with a WARN, then removed) and add `execution.gemm_table: bool`: a clearer name, but a config migration for one boolean with the same meaning
- C) Remove the switch: the table is always on; A/B only by building a library without the table

**Answer (2026-09-27): provisional (agent default under the coordinator's brief, pending user review) — A.** The spec S-8a meaning ("time candidates at first use") is replaced by the offline table: choices are restart-stable (no timing in the server), so the S-8 determinism note's "may differ between server restarts" no longer applies. A pinned solution missing from the installed hipBLASLt or rejecting a call falls back to the heuristic with `event=gemm_table_fallback reason=gemm_table_unavailable|gemm_table_unsupported`. Solutions are pinned by hipBLASLt solution name (the index is only a checked fast path). Every row that changes a solution changes that GEMM's rounding: golden references stay as they are and the c1/c16 tolerance is the gate.

Batch invariance (coordinator requirement, 2026-09-27, after the OLMoE c16 flip root cause): every pinned shape must give a row the same bits at every m and position. What the tuner found on the R9700 (hipBLASLt 1.4.1):

- hipBLASLt splits K for small problems by default, so one solution sums a row in another order at m = 1 than in a large batch; pinned rows therefore run through the ext API (`hipblaslt_ext::Gemm`) with split-K off (`GemmTuning::setSplitK(1)`), about 2–4 µs more host time per eager call (none under decode graphs).
- With split-K off, about 50 of the ~256 offered solutions per shape are row-invariant (a heavy-tailed target row checked alone and at 7 positions of batches up to 8,192 rows). Invariant solutions that give the row the same bits form a class; a shape's buckets may use different members of one class.
- The fastest invariant classes are fast either at decode or at prefill, not both, for the Llama down (3072 × 8192) and fused Q/K/V GEMMs. Pinning the decode-fast class costs Llama +25 % on a mixed (prefill) step, i.e. TTFT (down +54 %, gate/up +15 % at m = 2,048) for −7 % on a decode step. The tuner therefore refuses any class more than 5 % slower than the heuristic at m 1,024–2,048 (`--max-prefill-loss`); Llama down and k/v stay on the heuristic (not invariant), the other Llama shapes pin classes that are about neutral.
- OLMoE: all four GEMMs (fused Q/K/V, O, router, LM head) pin invariant classes; `hip_batch_invariance gemm_rows_are_batch_invariant` asserts all of them (`PENDING_GEMM_TABLE` empty) and `batch_invariance olmoe_rows_are_batch_invariant` shows bit-identical logits in every scenario, the long-prefill one included (now asserted).
- Cost against the same library without the table (interleaved A/B, GPU 0): OLMoE decode forward b16 −0.9 % (O −18 %, router −6 % per call), prefill 2,048 ±0 %; Llama eager decode forward b16 +2 % (the ext API's host time; decode graphs hide it), prefill 2,048 +0.2 %. The #1 gain the profile estimated for Llama (~+7 %) is not available with batch invariance under the TTFT guard.

Options put to the user: (a) keep the invariant table with the TTFT guard; (b) pin Llama's decode-fast invariant classes and accept ~+25 % Llama TTFT for ~−7 % decode step; (c) require invariance only for the OLMoE shapes and pin Llama for speed (the pre-invariance table measured −5 % Llama decode forward).

**Answer (2026-09-27, user): (c).** OLMoE shapes stay pinned batch-invariant (the c16 flip fix); Llama shapes are pinned purely for speed (fastest solution per bucket, split-K allowed, no invariance; Llama down and k/v included). Invariance is per shape, set per model in `kernels/rocm/tuning/gemm_shapes.txt` (`invariant` | `speed`) and recorded per table row (mode column), not a global switch; the library runs `invariant` rows with split-K off and `speed` rows through `hipblasLtMatmul`. The `gemm_table_fallback` reason codes are unchanged. The 5 % prefill guard applies to `invariant` shapes only (a `speed` bucket never pins anything slower than the heuristic).

## Pre-Phase-5 #4: parallel `logits_reduce` — library evaluation (kernel reuse rule)

Measured on the R9700 (novanas GPU 0, 16 rows × 128,256 F32 logits, 50 calls each, scratch program; `logits_reduce` today: 342 µs id-order draw at T 1, 318 µs nucleus at T 0.6 / top_p 0.9 over peaked rows, 620 µs over broad rows):

- rocPRIM 4.5 `segmented_reduce` max: 13.6 µs; f64 sum of an `expf` transform: 39.3 µs; hipCUB `DeviceSegmentedReduce::ArgMax`: 17.3 µs. Fit for the max, argmax and lse sum, not for top-n, the id-order draw or the nucleus.
- rocPRIM `topk` / `topk_pairs` (AIR radix top-k): no segmented form (one call per row: 1,020 µs for 16 rows at k = 64), and the radix algorithm refuses `Ordered` and `Deterministic` (static assertions), so ties at the threshold are not the lowest ids and repeated calls may differ: fails the exact top-n contract.
- rocPRIM `segmented_radix_sort_pairs_desc` (the nucleus order by sorting): 847 µs.
- Composable Kernel: `topk_softmax` is MoE routing (few experts), `ops/topk` a block-level streaming argmax; neither covers a 128k-wide row, a draw or a nucleus.

- A) Keep the Turbine kernel and make it faster with the same outputs: argmax shortcut for one candidate, one gather sweep plus a shared-memory sort instead of radix rounds 1–3 and the collection, integer nucleus weights without f64, the id-order mass shared with the lse sum at T = 1 and skipped for nucleus rows (chosen)
- B) Compose rocPRIM/hipCUB passes (max, argmax, lse sum) with the Turbine kernel for the rest: saves one sweep at most, adds launches and temporary storage, and top-n/draw/nucleus stay ours
- C) rocPRIM top-k and sort for top-n and the nucleus: slower (≥ 850 µs) and not exact

**Chosen (2026-09-27): provisional (agent, under the coordinator's brief) — A.** Outputs are bit-identical to the previous kernel (same top ids, lse and draws on the timing harness and in `hip_ops logits_reduce_matches_cpu`, which gains a top_n = 1 case and flat rows that exercise the radix-round path). The remaining time is dominated by the f64 lse sum (one f64 add per element; RDNA4's f64 rate); replacing it with exact integer masses changes lse and draws at the ~1e-9 level and is a separate step.

Second step (2026-09-27, same provisional answer): the lse and id-order draw masses become integers (each f32 `expf` weight times 2^S truncated, the scheme the nucleus already used), so no per-element f64 arithmetic remains and no sum depends on an order; the id-order draw's final scan is split over all 32 waves; round 0 of the radix select counts the 8 top-byte bins below the maximum in registers instead of a per-wave shared histogram (a full histogram sweep only when the threshold lies below them). The lse moves by at most vocab · 2^-S (< 2e-9) of the row's mass; draws still equal the host's except within that of a CDF boundary (`logits_reduce_matches_cpu`: lse |Δ| 0, every draw identical). 16 × 128,256 rows: id-order draw 342 → 68 µs, nucleus over peaked rows 318 → 110 µs, over broad rows (mass searches) 620 → 333 µs.

## Pre-Phase-5 #5: retune OLMoE's small-m down projection — measured, no change

The profile's #5 (down projection at ~540 GB/s against gate/up's ~600 GB/s) measured the scalar small-m kernels. The small-m tier now runs the WMMA decode kernels (0652a00), so the sweep ran on those. The down projection's waves per block and k slices per load batch were made independent of gate/up; both only change how the work is spread, not any output's ascending-k WMMA chain, so batch invariance and the grouped tier's numerics are untouched. Results from `hip_batch_invariance moe_decode_tier_timings` on GPU 0 (OLMoE, uniform routing), µs per `moe_experts` call at 1 / 4 / 8 / 16 / 32 / 64 tokens:

- k slices 4, waves 1 / 2 / 4 / 8: 278–281 / 735–740 / 1,034–1,035 / 1,218–1,221 / 1,474–1,480 / 1,496–1,505
- k slices 8 (today's value), waves 1 / 2 / 4 / 8: 267–270 / 709–713 / 1,002–1,005 / 1,185–1,189 / 1,436–1,440 / 1,456–1,461
- k slices 16, waves 1 / 2 / 4 / 8: 273–280 / 734–740 / 1,044–1,046 / 1,239–1,255 / 1,475–1,487 / 1,491–1,502

A kernel trace (rocprofv3) of the same test at 64 expert slots gives down 457 µs against gate/up 949 µs for twice the bytes: the down projection already streams its weights as fast as gate/up.

- A) No change: today's shared setting (8 k slices, 4 waves) is already the best for down, within 0.5 % of every waves value (chosen)
- B) Land per-projection constants anyway (no measured gain)

**Chosen (2026-09-27): provisional (agent) — A.** The remaining OLMoE decode headroom is the weight-bandwidth ceiling both projections share (~550 of ~640 GB/s), not the down projection's tuning.

## Pre-Phase-5 #1 follow-up: prefix reuse must stay bit-exact for Llama

Found 2026-09-27: `kv_gpu prefix_reuse_matches_cold` passes at 92da19b and fails from 39916b3 (#1, option c). Llama's speed-tuned GEMMs round differently by m, so a warm request (prefill of the uncached suffix only) diverges from the cold one (whole-prompt prefill) at a near-tie. That breaks Phase 4's "outputs identical to a cold run".

- A) Invariant Llama prefill: keep Llama's decode-sized buckets speed-tuned, and pin Llama's prefill-sized buckets to one batch-invariant algorithm group, so a suffix prefill gives the same rows as the cold prefill; also cover short suffixes that fall into decode-sized buckets (recommended)
- B) Relax the check to the golden tolerance (identical tokens with the near-tie excuse), as for batch composition; amends the Phase 4 spec
- C) Full Llama invariance (option b of #1): ~+25 % Llama TTFT, most of the +7.7 % gone

**Answer (2026-09-27): A — invariant Llama prefill.** The prefix-reuse check stays bit-exact.

## Pre-Phase-5 #3: decode attention for grouped query heads — provider evaluation (kernel reuse rule)

Llama-3.2-3B decodes 24 query heads over 8 KV heads (GQA 3), head_dim 128, 128-token pages. The profile put paged decode attention (CK `fmha_fwd_pagedkv`) at 22.5 % of a c16 decode step, ≈ 390 GB/s of KV against OLMoE's ≈ 600 GB/s on the same kernel: pagedkv runs one workgroup per query head (a 64-row tile holding one query), so each K/V page is read three times. Measured on novanas GPU 0 with `hip_ops decode_attention_timings` (every enumerated implementation bound as the registry binds it, KV pools cycled past the 64 MB MALL, wall time per call incl. the ~3 µs append), µs per call:

| Candidate                                                                                                                                       | Builds on gfx1201?                                                                                                             | Correct vs CPU reference?                                                                                                               | b16 @768                                                                                                                                                                                  | b16 @2k             | b16 @8k               | b1 @768 / @8k                 | b64 @768            |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------- | ----------------------------- | ------------------- |
| CK `fmha_fwd_pagedkv` (today; the only gfx12 pagedkv tile, b64x64)                                                                              | yes                                                                                                                            | yes (reference model)                                                                                                                   | 133.5                                                                                                                                                                                     | 348.6               | 1,378 (389 GB/s)      | 47.4 / 366                    | 527                 |
| CK `fmha_fwd_splitkv`, group mode, paged, no mask — merges the query heads of a KV head into one tile (`kMergeNumHeadGroupsSeqLenQ`), one split | yes (b16x128 nwarp-sshuffle tile + combine)                                                                                    | yes: max \|Δ\| ≤ 1 BF16 ulp on `paged_decode_splitkv_matches_cpu`, tiny Llama 3.8e-6 (= pagedkv)                                        | **107.8**                                                                                                                                                                                 | **243.9**           | **880 (610 GB/s)**    | 46.9 / 297                    | **388**             |
| same, context split by batch size (CK's heuristic: splits until CUs × 4 workgroups)                                                             | yes                                                                                                                            | outputs within 1 ulp, but leaves the CPU numerics model: tiny Llama 0.10 max \|logit Δ\| (bound 1e-4); golden c1/c16-free replay passes | 101.9 (1 split)                                                                                                                                                                           | 235 (1 split)       | 875 (1 split)         | 24 / 72                       | 379 (1 split)       |
| same on OLMoE's MHA 16/16 (experiment build)                                                                                                    | yes                                                                                                                            | –                                                                                                                                       | 198 vs pagedkv 183 (−8 %)                                                                                                                                                                 | 492 vs 470          | 1,859 vs 1,778        | 28 / 126 vs 45 / 390          | 706 vs 690          |
| CK `fmha_fwd_appendkv`                                                                                                                          | yes (paged gfx12 instances)                                                                                                    | –                                                                                                                                       | replaces only the ~3 µs append (can fuse RoPE: item #10), not attention                                                                                                                   |                     |                       |                               |                     |
| llama.cpp `ggml-cuda` FA (HIP, a97cce8, `test-backend-ops perf`, 16 streams × per-stream KV, GQA 3)                                             | yes                                                                                                                            | not run (not wired)                                                                                                                     | BF16 K/V 437; F16 K/V 74*                                                                                                                                                                 | BF16 1,170; F16 244 | BF16 4,289; F16 1,686 | BF16 543 / 284; F16 156 / 48* | BF16 1,712; F16 393 |
| aiter paged attention (`pa_v1`/`pa_ragged`; ASM `pa_decode_bf16`)                                                                               | no: MFMA (gfx9) only; ASM asserts gfx1250; the Gluon path is Triton (excluded); its `mha_fwd_split` wraps the same CK split-KV | –                                                                                                                                       | –                                                                                                                                                                                         | –                   | –                     | –                             | –                   |
| vLLM ROCm custom paged attention (`csrc/rocm/attention.cu`, Apache-2.0)                                                                         | has a gfx12 WMMA path, packs GQA, partitions the context                                                                       | –                                                                                                                                       | not measured: pages of 16/32 tokens only, K `[blocks, kv_heads, head/x, block, x]` and V transposed — needs a KV-format change (pool, append, prefill, tiers) or a fork of its addressing |                     |                       |                               |                     |

\* llama.cpp's harness reuses one KV set per case; below ~64 MB it stays in the MALL, so these F16 numbers are warm-cache (not comparable). At 2k/8k (134/537 MB, cold) F16 ties split-KV and is 1.9× slower; with BF16 K/V (our KV format) its WMMA/tile kernels convert K and V to F16 on every call (3–5× slower). It has no page table (contiguous per-stream KV plus a KQ mask), so a registered implementation would be a fork of its load path, i.e. our own kernel.

Split sweep (1–32 splits, `TURBINE_CK_SPLITKV_SPLITS` experiment build, not committed): 1 split is fastest at every batch ≥ 16 and every context (the head merge already gives 128 workgroups at b16); splitting pays only at small batches (b1: 40 → 24 µs @768, 297 → 72 µs @8k; b4 @2k: 89 → 71). With more than one split each split rounds P to BF16 against its own running maximum, so the result leaves the cpu-reference model of CK attention (tiny Llama HIP-vs-CPU 3.8e-6 → 0.10; `tiny_model hip_matches_cpu` / `hip_v23_library_matches_cpu` bound 1e-4). The Llama golden (`logits_match_reference`, batch 1) passes with and without splitting (likely ≤ 0.098, tail ≤ 0.29).

- A) CK split-KV with the head merge, one split, as the registered `ck_tile_fmha_splitkv` for grouped heads (library and gfx1201 order first for `attention_decode_paged`; equal head counts refused, so OLMoE keeps pagedkv): c16 decode attention −19 % per call (≈ −0.6 ms/step net of the extra combine launch, ≈ +3.5 % Llama tok/s), −30–36 % at 2k–8k and b64; CPU numerics model kept, a Llama decode row independent of its batch size (recommended)
- B) As A plus context splitting for small batches (CK's heuristic): the same c16 numbers, plus single-stream latency (b1 @8k 4×) — but Llama decode numerics then depend on batch size and leave the CPU model; the tiny-Llama HIP-vs-CPU bound must be loosened (~0.3) or those tests pinned to one split
- C) llama.cpp FA vendored: needs F16 KV (a KV-format change) and a page-table fork; no faster where measured fairly
- D) Write our own GQA split-context kernel: not needed — CK's existing kernel already merges the heads and reaches the ~610 GB/s bandwidth floor at 8k

**Chosen (2026-09-27): provisional (agent, under the coordinator's brief) — A.** Per-op before/after (µs per call, GPU 0): b16 @768 133.5 → 107.8, @2k 348.6 → 243.9, @8k 1,378 → 880; b64 527 → 388; b1 47.4 → 46.9, b1 @8k 366 → 297. Kernel trace: combine 6.4 µs and append 3.6 µs at b16. OLMoE (MHA) is unchanged: golden byte-identical to the previous library, `batch_invariance olmoe_rows_are_batch_invariant` and `hip_batch_invariance paged_attention_rows_are_batch_invariant` pass. `paged_attention_fallback` now fires only when a paged selection runs the profile order's last (any-page-size) entry. B is one constant (`kSplits` in `kernels/rocm/src/paged_attention_splitkv.cpp`, plus the heuristic in this record) away once the user decides how the tiny-model numerics bound should treat it; the scratch already never shrinks under captured graphs.

## Pre-Phase-5 #1 follow-up (A): how Llama's prefill GEMMs become batch-invariant, and short suffixes

Implemented on `perf-gemm-prefix`. The GEMM library cannot tell a prefill step from a decode step by m alone: a warm suffix of 2–64 tokens has a decode-sized m. Options measured on novanas GPU 0 (tuner cost lines, µs summed over one Llama forward: 28 × fused Q/K/V, O, fused gate/up, down, plus the LM head; against 39916b3's table):

| m (step)                | 39916b3                  | A1: one invariant class for every m (decode-weighted) | A2: step kind flag (chosen)                         |
| ----------------------- | ------------------------ | ----------------------------------------------------- | --------------------------------------------------- |
| 1 / 16 / 64 (decode)    | 10.59 / 10.79 / 11.34 ms | 10.76 / 10.84 / 11.30 ms (≈ 0)                        | unchanged (speed rows kept)                         |
| 128 / 512 (prefill)     | 12.11 / 28.68 ms         | 12.10 / 30.74 ms (+7 %)                               | 13.04 / 27.19 ms                                    |
| 1,024 / 2,048 (prefill) | 50.71 / 100.75 ms        | 60.69 / 116.26 ms (+20 % / +15 %)                     | 53.15 / 105.07 ms (+5 % / +4 %)                     |
| 1–64 as a prefill step  | –                        | as decode                                             | +8–10 % (tiny absolute: short prompts and suffixes) |

- A1) One row-invariant class per shape for every m, decode buckets taking only class members (`invariant` mode, decode weight 0.85, no guard): decode costs nothing (the speed-tuned decode solutions of every shape turned out to be members of a decode-fast class), but no class is fast at both ends — down's decode-fast class is +40–60 % at m ≥ 1,024 — so prefill (TTFT) pays +15–20 %.
- A2) The step kind reaches the library: `GemmContext::prefill` (true in every step that is not decode-only, set by the decoder executor), passed as the context option `TURBINE_OPTION_GEMM_PREFILL` (additive in ABI v2.5; the shim sets it when it changes; an older library answers unsupported once). Tuning mode `prefix` (the Llama shapes) writes `speed` rows for the decode buckets and one invariant class over every bucket scored for prefill steps only; decode steps run the speed rows, prefill steps the invariant class at any m. Decode unchanged; prefill pays only for down, where no row-invariant solution matches the heuristic's (split-K-free but still batch-variant) large-m kernel: down +33–48 % at m ≥ 1,024, other shapes −12 % to +5 %.
- A3) A threshold (invariant rows from m = 65 up, speed below): not correct — a warm suffix of 2–64 tokens runs other rows than the cold prefill — unless prefix reuse were cut back to leave ≥ 65 tokens (a lost block on every warm request).
- A 1-token suffix (prompt one token past a block boundary: the planner reused up to `prompt − 1` tokens) makes a decode-shaped step (`max_q_len` 1: decode attention on CK split-KV and the decode GEMM rows), which cannot reproduce the cold prefill's bits with any GEMM table: `kv_gpu prefix_reuse_suffix_lengths_match_cold` showed it (warm ≠ cold from token 0). The KV planner now always leaves `MIN_RECOMPUTE_TOKENS` = 2 prompt tokens (such a prompt recomputes its last block, 129 tokens); Phase 4 spec edge case amended.

**Chosen (2026-09-27): provisional (agent, under the coordinator's brief) — A2 with the planner's two-token minimum.** Measured on GPU 0 natively (`kv.prefix_sharing: false`, two interleaved A/B rounds against main's library, same server build):

- c1 prefill forward: ~700-token prompt 49.9–50.2 → 51.8–52.2 ms (+4 %), TTFT p50 51.7–52.0 → 53.5–53.7 ms; ~2,000-token prompt 69.0–69.5 → 73.9–74.4 ms (+7 %), TTFT p50 139.6–140.6 → 149.5–150.5 ms.
- c16 (512 words, 256 tokens, 64 requests): decode forward 15.30 → 15.31–15.32 ms, ITL p50 15.37 → 15.38 ms, tok/s 890.1–890.6 → 877.3–877.9 (−1.4 %), TTFT p50 229–230 → 249–250 ms.
- `kv_gpu` (lab-test, all four): `prefix_reuse_matches_cold` passes again; `prefix_reuse_suffix_lengths_match_cold` (suffixes 1, 2, 17, 64, 65, 100, 700; text and every logprob bit-equal) passes; `nvme_round_trip_matches_cold` passes. `hip_batch_invariance`: `llama_prefill_gemm_rows_are_batch_invariant` (new; fails on main's library: 316 row shapes change) and the OLMoE `gemm_rows_are_batch_invariant` pass; `batch_invariance olmoe_rows_are_batch_invariant` passes; golden `logits_match_reference` 16/16 and `olmoe_logits_match_reference` 15/16 pass; `hip_ops gemm_table_matches_cpu` passes.
- Llama decode rows stay 39916b3's speed rows (clipped to m ≤ 64); a re-tune of them moved gate/up's m = 16 bucket to the heuristic by noise (+0.2 ms per decode step), so they were kept, not regenerated.

Follow-up worth a look: down's fast large-m kernel is split-K-free yet not row-invariant (likely a stream-K style partition of K across workgroups that depends on the tile count); a hipBLASLt tuning knob that pins it would remove most of the +4–7 % prefill cost.

## Pre-Phase-5 #2: MoE prefill for OLMoE — provider evaluation (kernel reuse rule)

OLMoE-1B-7B's grouped MoE (64 experts, top-8, hidden 2048, expert inter 1024, BF16) was 68.7 % of a served mixed step (profile: `moe_wmma_kernel` ≈ 59 TFLOPS). Measured with `turbine-model --test perf moe_prefill_timings` (1e78d60): one 2,048-token OLMoE prefill over the golden prompts is traced; layers 0/5/10/15's real MoE inputs, router logits and expert weights are routed again and every enumerated `moe_experts` implementation runs alone at 512 / 2,048 / 16,384 routed rows (real routing is skewed: at 16,384 rows the largest expert holds 964–1,205 rows against a mean of 256), with a bitwise comparison against `turbine_hip_moe_wmma`. Everything below ran natively on novanas GPU 0 under `scripts/bench-lock.sh`; µs per whole `moe_experts` call (positions, gate-up, down, scatter) unless noted.

| Candidate                                                                                                                                                                              | Builds on gfx1201?                                                                                                                                                                                   | Correct vs CPU?                                                                                    | Invariance-compatible (same WMMA chain as the small-m decode tier)?                                                                           | 512 rows                             | 2,048 rows      | 16,384 rows                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `turbine_hip_moe_wmma` (before)                                                                                                                                                        | yes                                                                                                                                                                                                  | yes                                                                                                | yes (reference)                                                                                                                               | 1,550–1,660                          | 1,680–1,740     | 2,990–3,100 (≈ 68 TFLOPS)                                                                                                                                                                      |
| CK `ck_tile` fused MoE (`fused_moegemm`, `moe_sorting`)                                                                                                                                | no: the fused-MoE pipeline is hard-wired to MFMA 32×32×8 (gfx9)                                                                                                                                      | –                                                                                                  | –                                                                                                                                             | –                                    | –               | –                                                                                                                                                                                              |
| CK `ck_tile` grouped GEMM (`GroupedGemmKernel`, persistent tile loop, WMMA, `CK_TILE_USE_WMMA`; scratch program, kargs on the device as a device-filled table would be)                | yes (the 8-wave 256×128 / 128×256 configs return wrong results; 128×128 and 256×256 are correct)                                                                                                     | tolerance not needed: outputs bitwise equal to the Turbine chain                                   | **yes: 0 of 16.8M / 33.6M outputs differ** from a reference kernel with our chain                                                             | –                                    | –               | per projection at the best of 19 configs (256×256×32, 4×4 warps): gate 1,390, up 1,390, down 1,540 → ≈ 4,300 + gather + SiLU·up (≈ 49 TFLOPS; 50–60 even on one dense 16,384×1,024×2,048 GEMM) |
| hipBLASLt `GroupedGemm` (existing `hipblaslt_grouped`)                                                                                                                                 | no grouped solution in hipBLASLt 1.4.1 / ROCm 7.14.1 for gfx1201 (the heuristic returns none; the implementation refuses)                                                                            | –                                                                                                  | –                                                                                                                                             | unsupported                          | unsupported     | unsupported                                                                                                                                                                                    |
| hipBLASLt one GEMM per expert (existing `hipblaslt_per_expert`; needs host offsets, i.e. a device-to-host read per layer)                                                              | yes                                                                                                                                                                                                  | yes (`every_implementation_matches_cpu`)                                                           | no: 5–1,197 of the outputs differ from the chain (13–16k of 4.2M at 16k rows)                                                                 | 3,250–3,480                          | 3,630–3,740     | 5,200–5,390                                                                                                                                                                                    |
| hipBLASLt batched GEMM over padded expert segments                                                                                                                                     | not built: needs the largest segment on the host (a sync) and pads to it (up to 4.7× the mean with real routing); numerics as per-expert                                                             | –                                                                                                  | no                                                                                                                                            | –                                    | –               | –                                                                                                                                                                                              |
| llama.cpp `mul_mat_id` (HIP, a97cce8, BF16 weights, `test-backend-ops perf` at OLMoE shapes, uniform ids)                                                                              | yes                                                                                                                                                                                                  | its own eval passes                                                                                | no (F32 activations; hipBLAS/MMF per expert)                                                                                                  | gate 485 + up 485 + down 528 ≈ 1,500 | ≈ 4,020         | ≈ 8,790                                                                                                                                                                                        |
| vLLM C++ MoE (`moe_align_sum_kernels`, `moe_permute_unpermute`, `moe_wna16`, `marlin_moe_wna16`, CUTLASS grouped, `moe_q_gemm_rdna3`) / SGLang (aiter on AMD)                          | the GEMMs are CUDA-only, GPTQ W4A16 for gfx1100, or gfx9/gfx1250 (aiter CK/ASM; its gfx1250 grouped MoE is FlyDSL, Python); the align/permute helpers duplicate our `moe_route` and are not the cost | –                                                                                                  | –                                                                                                                                             | –                                    | –               | –                                                                                                                                                                                              |
| **Own: `turbine_hip_moe_wmma_prefill`** (weights streamed from memory straight into the WMMA registers, only the gathered activations in LDS, double-buffered, one barrier per k step) | yes                                                                                                                                                                                                  | yes (`every_implementation_matches_cpu`, `moe_experts_grouped_matches_cpu` incl. a wide-tile case) | **yes: 0 differing on real data at every size**; `hip_batch_invariance` A/B of all three device-offset implementations at 65 and 1,536 tokens | **1,415–1,500**                      | **1,552–1,579** | **2,500–2,560** (≈ 81 TFLOPS)                                                                                                                                                                  |

Why no provider wins: CK is the only one that keeps the WMMA chain (its WMMA GEMM is bitwise equal to ours), but on RDNA4 it reaches only about half of what our fused gate-up kernel already did, and it would add a gather and a separate SiLU·up pass. hipBLASLt reaches 128–139 TFLOPS on dense GEMMs here but has no grouped kernel for gfx1201, and per-expert calls cost 192 launches plus a host sync per layer. llama.cpp is 2.9× slower at 16k rows. So step 3 (own kernel) applied.

Own-kernel findings (standalone sweep, not committed): the old kernel was bound by staging both operands through LDS (global+LDS traffic ≈ as long as the whole kernel; bigger LDS tiles and deeper register prefetch did not help, and a register-only WMMA loop peaks at ~180 TFLOPS on this card). Loading B (weights, K-contiguous rows = the WMMA B lane layout) directly into registers removed 2/3 of the LDS traffic: gate-up 1.82 → 1.53 ms, down 1.06 → 0.78 ms at 16k rows. Down picks 64×128 tiles with k steps of 128 below, 64×256 with k steps of 64 from 12,288 routed rows (the average expert filling three 64-row tiles; 0.61 vs 0.68 ms at 8,192, 0.78 vs 0.89 at 16,384); tile shape never changes an output's chain. Row-tile grouping for L2 reuse of weights: ≤ 2.5 %, not taken.

- A) Own prefill kernels as the new registered `turbine_hip_moe_wmma_prefill`, first in the gfx1201 tier above 512 rows and index 1 in library order; `turbine_hip_moe_wmma` stays enumerated (A/B, fallback). Bitwise equal to the previous tier, so batch invariance, golden and prefix-reuse exactness are unchanged (chosen)
- B) CK grouped GEMM as a registered implementation: invariance-compatible but ≈ 45 % slower than the old kernel at 16k rows
- C) hipBLASLt per expert / llama.cpp: slower and not invariance-compatible (the decode tier would have to change too)

**Chosen (2026-09-27): provisional (agent, under the coordinator's brief) — A.** Per call (µs, GPU 0, real routing): prefill 512 rows 1,550–1,660 → 1,415–1,500 (−9 %; the small-m tier still serves ≤ 512 rows at 1,405–1,473), 2,048 rows 1,680–1,740 → 1,552–1,579 (−8 %), 16,384 rows 2,990–3,100 → 2,500–2,560 (−16 %). Decode unchanged (the small-m tier, `moe_decode_tier_timings`: 1 / 16 / 64 tokens 270 / 1,189 / 1,461 µs, as before). Expected effect: ≈ −0.45 ms per layer on a served ~1,680-token mixed step, ≈ −7 ms of 68.7 (−10 %), so OLMoE TTFT p50 ≈ 135 → ≈ 122 ms and ITL p99 −10 %; tok/s ≈ +1 % (mixed steps are 8.4 % of engine time). This falls short of the profile's −25–30 % (it assumed ~110 TFLOPS; the kernel reaches ~81 end to end, 90 on gate-up). The rest of the headroom is in the kernel's compute side (LDS reads of A + WMMA issue ≈ 1.15 ms of gate-up's 1.53) and `moe_scatter` (#8). The served numbers are the coordinator's `lab-bench.sh --golden16` to confirm.

## Pre-Phase-5 #1 recovery: winning back the prefix-exact prefill cost (findings, no change)

After the prefix fix (6f6e18f) the served Llama bench on e7474ef reads 855.6 tok/s, ITL 15.4 ms, TTFT 207 ms against 863.1 / 15.4 / 194 before it (−0.9 % tok/s, +7 % TTFT; a first reading of −4.3 % overlapped native GPU 0 runs of this agent whose server start-up was outside the bench lock — every native run now holds `scripts/bench-lock.sh` for its whole duration). Two ways back were examined on novanas GPU 0, natively, under the lock:

- (1) Split mixed steps' GEMMs by row kind (decode rows on the speed rows, prefill rows on the invariant rows). Not worth building: the scheduler does put decode rows first (`decode_first_then_chunked_prefill`), but a second launch per projection re-reads the whole weight. At m = 16 that is 54.8 + 34.1 + 165.9 + 83.9 µs per layer ≈ 9.5 ms per mixed step (tuner cost lines), while moving 16 rows off the invariant rows saves nothing measurable; the mixed-step cost is the invariant rows' speed at m ≈ 2,000, not the decode rows riding on them. It would also leave a one-token prefill chunk classified as a decode row.
- (2) A row-invariant fast kernel for the down projection (3072×8192). Root cause found: every fast large-m hipBLASLt solution for down staggers the start of its K loop by the workgroup's M (token) tile (Tensile StaggerU mapping, `SUM1` in the solution name), so a row's summation order changes with the row count; with split-K off the heuristic's m = 2,048 kernel (MT128x128x32) still changes 3 of 6,144 output bytes of a row at position 1,024 of a 2,048-row batch. Fixed split-K values (2, 3, 4, 8) and the whole solution list (845 solutions, 589 supported, the 64 fastest at m = 2,048) give no invariant fast one. `TENSILE_FIXED_STAGGERU_MAPPING=0` (read by hipBLASLt at its first solution selection) makes every configurable solution stagger by its N tile instead: all 256 heuristic candidates of every Llama shape then pass the row-invariance check (the down class at m = 1,024 / 2,048: 578 / 1,077 µs against 705 / 1,397 µs for today's pinned class). With the Llama invariant rows re-tuned under it and the decode speed rows kept: prefill GEMM time per forward at m = 2,048 105.1 → 96.1 ms; served-shape A/B against e7474ef (3 interleaved rounds each, 128 requests): c16 TTFT p50 208–210 → 195–196 ms (−6 %), c1 ~2,000-token TTFT 150 → 140 ms (−7 %), c1 ~700-token 53.8 → 52.0 ms, c16 879 → 890 tok/s (+1.2 %), decode forward 15.30 ms unchanged, OLMoE unchanged (the env alone moves nothing on e7474ef's library). `hip_batch_invariance` (all five, including `llama_prefill_gemm_rows_are_batch_invariant`; mapping 1 breaks it: 55 row shapes), `batch_invariance olmoe_rows_are_batch_invariant`, `kv_gpu` prefix tests and OLMoE golden pass. **But the Llama golden fails**: p16's tail logprob moves 0.74 from the reference (bound 0.55; 0.23–0.46 with today's rows under either mapping), and it is the re-tuned down class alone (the table with only down's rows re-tuned gives the same 0.7403). So the ≥ 5 % TTFT win does not come with the tests green, and the track is closing: no change lands.

Parked on the local branch `perf-gemm-stagger-wip` (f8339e3: `fix_stagger_mapping` in `gemm_problem.hpp`, set by the library and the tuner before their first hipBLASLt handle, the re-tuned table, docs). Worth picking up later: pick the down class by golden numerics as well as speed (the map0 tune's second down class is still 1.25–1.35× the heuristic at m ≥ 1,024 against today's 1.33–1.52×), or re-derive the Llama tail tolerance if the 0.74 is judged rounding noise of an equally valid summation order.

## After the pre-Phase-5 close-out: push, and Phase 5

Local main e2f8178 holds Phase 3, Phase 4 and the pre-Phase-5 track, unpushed; `.procoder/review-2026-09-28.md` lists 19 provisional decisions.

- A) Review the list first; push and start Phase 5 only after the review (recommended)
- B) Push now, then review the list; hold Phase 5
- C) Push now and start Phase 5 (port the `p5-distributed` run-ahead onto main)
- D) Hold everything: no push, no Phase 5

**Answer (2026-09-28): A — review first.** Push and Phase 5 wait until the user has reviewed `.procoder/review-2026-09-28.md`.

## Review of the 19 provisional decisions (`.procoder/review-2026-09-28.md`)

- A) Accept all 19 as recorded
- B) Change some of them

**Answer (2026-09-28): A — the user accepted all 19 ("all good").** Every item listed in `.procoder/review-2026-09-28.md` is now a confirmed decision; #18's parked `perf-gemm-stagger-wip` stays parked.

**Answer (2026-09-28, after the review): push and start Phase 5.** main pushed to origin at 6545638; Phase 5 starts by porting the `p5-distributed` run-ahead onto main (novanas only, NVIDIA still on hold).

## P5 T6: kernel ABI for tensor parallelism — major v4 or an optional minor group v2.6, and which new ops?

Asked 2026-09-28 (Phase 5 port, branch `phase-5-multi-gpu`). The Phase 5 plan (Task 6) and contract §9.1 bump the kernel ABI to a major v4 that adds `turbine_stream_native_handle` and three ops (`row_sumsq`, `rmsnorm_sharded`, `fill`), and a major bump would also make the v2.3 / v2.5 groups required. Phase 4 took the other road (decision "Phase 4: kernel ABI v2.5 instead of v3", accepted): additive optional minor groups, `TURBINE_ABI_VERSION` stays 2.

What tensor parallelism actually needs from the library, checked against main:

- The vendor stream is required: the HIP shim's compute stream is created `hipStreamNonBlocking`, and `StreamRef::native_handle()` is 0 today, so RCCL on the null stream would not be ordered with the forward pass. `turbine_stream_native_handle(ctx, s, void**)` is the only way to get it.
- The vocab-parallel embedding needs nothing new: the v1 `turbine_embedding_desc` already has `vocab_offset` / `vocab_rows` and writes zeros for ids outside the shard (then all-reduce).
- `fill` (−∞ into padded vocab rows) is only needed when the vocabulary is not divisible by tp; Llama-3.2 (128,256) and OLMoE (50,304) divide by 2, 4 and 8. Where it is needed, an uploaded −∞ row copied with the existing `turbine_memcpy_*` does the same.
- OLMoE's full-projection QK-norm has two correct forms: (a) the spec's sharded norm — `row_sumsq` of the rank's slice, FP32 all-reduce of the partial sums, `rmsnorm_sharded` (two new own kernels; the reduction order of the sum of squares then differs from tp = 1); or (b) all-gather Q and K (BF16, tokens × 2,048 each per layer), run the existing CK `rmsnorm2d` over the full width, keep the rank's heads (no new kernel; the norm itself is the tp = 1 computation; costs one extra all-gather per layer: ~16 KB per decode token, ~8 MB for a 2,048-token prefill chunk, host-staged on novanas).

Options:

- A) Optional minor group v2.6 with only `turbine_stream_native_handle` (resolved when minor ≥ 6; without it tp > 1 is refused at startup with a reason code, tp = 1 unaffected); OLMoE QK-norm by all-gather + the existing rmsnorm (b); vocab padding with the existing copies; contract §9.1 and plan Task 6 amended (recommended: follows the accepted v2.5 precedent and the kernel-reuse rule, no new kernels, no break for existing libraries)
- B) As A, but add `row_sumsq` / `rmsnorm_sharded` in the v2.6 group for the sharded QK-norm (a): less PCIe traffic on OLMoE prefill, two own kernels (reuse evaluation first: CK has no sharded RMSNorm)
- C) The planned major v4 (native handle + the three ops, v2.3 / v2.5 groups required, older libraries refused)

Until answered: the Rust side of the stream handle (`StreamRef::native_handle()` filled when the library exports it) and the NCCL-API communicator (Task 7) proceed, since every option has the same `turbine_stream_native_handle(ctx, s, void**)` signature; no kernel or header change lands.

**Answer (2026-09-28, user): B.** An optional minor group v2.6 with `turbine_stream_native_handle` plus own `row_sumsq` and `rmsnorm_sharded` kernels for the sharded QK-norm (a); `TURBINE_ABI_VERSION` stays 2. Under the reuse-first rule the evaluation of existing providers for the sharded case is recorded with the kernels (decision "P5 T6: sharded RMSNorm — provider evaluation"), and each new kernel gets a correctness test against the CPU reference.

## P5 T17: Phase 4 KV tiers (L1 pinned host, L2 NVMe) under tensor parallelism

Asked 2026-09-28. Spec §Data: "Phase-4 tier copies (CPU, NVMe) of a TP block are stored per rank shard, keyed by (block key, tp size, rank)". Today the KV hierarchy and its orchestrator (`turbine_kv::hierarchy`, `turbine_server::kv_orchestrator`) drive one pool on one device: demotion, promotion, the copy streams, the L2 slab files and the transfer calibration are all per pool. With tp = 2 every block has one shard per rank, so every tier copy becomes one copy per rank that must all complete before the block counts as demoted or promoted.

- A) First TP landing serves with L0 only when tp > 1: prefix sharing inside L0 works (block ids are logical and `copy_blocks` runs on every rank), `kv.cpu` / `kv.nvme` are refused with a reason code (`kv_tiers_unavailable_under_tp`, WARN, L0 only) and per-rank tier copies follow as their own task after the TP golden gate passes; spec §Data amended to say so (recommended: correctness of TP first, as the spec's own rule demands, and the tier fan-out is a separate change with its own tests)
- B) Build per-rank tier copies in the same task: one copy per rank per block, the hierarchy tracks completion across ranks, L2 slab files keyed by (block key, tp, rank)

DP replicas (tp = 1 each) are unaffected: each replica has its own pool and hierarchy.

**Answer (2026-09-28, user): B.** Per-rank L1/L2 tier copies are built inside T17 (one copy per rank per block, completion tracked across ranks, L2 slab files keyed by (block key, tp, rank)); no L0-only first landing.

## P5 T6: sharded RMSNorm — provider evaluation (kernel reuse rule)

Recorded 2026-09-28 with the v2.6 kernels (decision "P5 T6", answer B). The op: a tensor-parallel rank holds a contiguous slice of a row normalised across ranks (OLMoE's QK-norm over 16 heads × 128 = 2,048, split by heads); it needs (1) the FP32 sum of squares of its slice per row, then, after an FP32 all-reduce of those sums, (2) `x · rsqrt(sumsq / full_dim + eps) · weight_slice` with the Hugging Face BF16 rounding of `x · inv_rms`. Checked read-only against the ROCm install on novanas (`/opt/rocm/rocm/include`, CK `therock-7.14.1`, the commit `kernels/rocm` pins); nothing measured, since no provider covers the op:

- Composable Kernel `ck_tile` rmsnorm2d (`ops/rmsnorm2d`, the one-pass, two-pass and model-sensitive T5 pipelines): `Rmsnorm2dFwdHostArgs` has no sum-of-squares input; each pipeline reduces `SquareAdd` over the `n` columns it is given and scales with `rsqrtf(square_sum / row_size + epsilon)` in the same kernel, `row_size` being that `n`. It can emit only `p_invRms` (the inverse RMS of the slice, stored in the instance's `InvRmsDataType`), not the partial sum, and never takes an all-reduced one: run on a slice it normalises the slice by its own RMS, which is wrong for tp > 1. Recovering the partial sum from the stored inverse RMS (`n · (1 / inv² − eps)`) would be lossy and still leave the scaling step without a kernel.
- Composable Kernel `ck_tile` reduce2d (`ops/reduce`, `ReduceKernel` with `ReduceOp::SquareAdd`): a generic 2-D reduction that could produce `row_sumsq` with a new instance (its per-row order follows the instance's tile shape), but not the scaling step, so (2) would still be ours.
- rocPRIM / hipCUB `segmented_reduce` (`rocprim/device/device_segmented_reduce.hpp`): per-row sums with a squaring transform iterator over BF16 are possible, at the cost of a device array of segment offsets for strided rows, a temporary-storage size query plus the run (two calls, workspace per call), and again no scaling step.
- hipBLASLt, vLLM / SGLang and llama.cpp HIP kernels: fused RMSNorms over a whole row (the same shape as CK's), no external sum of squares.

Outcome: own kernels, as the answer chose — `kernels/rocm/src/sharded_norm.hip` (`row_sumsq`, `rmsnorm_sharded`): one 256-thread block per row, the reduction order of the Turbine rmsnorm fallback (so `row_sumsq` then `rmsnorm_sharded` over a whole row is bitwise the HIP `rmsnorm` at 2,048), independent of the number of rows (batch invariance); correctness against the cpu-reference in `hip_ops sharded_norm_ops`. A CK reduce2d `row_sumsq` could be registered as a second implementation later if a measurement favours it.

## P5: the static-mode rank link and the DP router policy as registries?

Asked 2026-09-28 (Phase 5 port): the static-mode rank bootstrap is plain TCP (spec S-5) and `parallel.router` a closed `prefix_affinity | least_loaded` enum (spec S-7), while the pluggability rule (2026-09-26) lists "collectives and transports" and policies as registered extension points.

- A) Keep TCP and the enum until the multi-node phase adds `turbine-transport`
- B) Registries now: a small `Transport` trait for the rank link with a static registry holding `tcp`, and a DP router policy registry holding `prefix_affinity` and `least_loaded`, each with a conformance suite and a `docs/extending/` page (the configuration keeps the same names)

**Answer (2026-09-28, user): B.** The rank link goes behind a `Transport` trait with a static registry (one entry, `tcp`; the deferred multi-node phase adds more), and the DP router policy becomes a registry (`prefix_affinity`, `least_loaded`); both with conformance tests and a `docs/extending/` page, same pattern as `collective_backend`. `parallel.router` keeps its names.

## Parallelism modes: pipeline and expert parallelism in Phase 5; sharded data parallelism

Asked 2026-09-28. The user listed five parallelism modes — data parallel (DP), sharded data parallel (ZeRO / FSDP), pipeline parallel (PP), tensor parallel (TP) and MoE expert parallel (EP) — and said "we are doing all 5".

Q1, where PP and EP go:

- A) A new Phase 5b after Phase 5 (recommended)
- B) Fold into Phase 5
- C) After quantization

**Answer (2026-09-28, user): B.** PP and EP join Phase 5 as new tasks after T19 (T17 unchanged), single node (novanas, both R9700s); their single-node designs come from the deferred `phase-10-advanced-distribution` spec (formerly Phase 7), whose multi-node, RDMA and disaggregation parts stay deferred.

Q2, what sharded data parallelism means for inference:

- A) ZeRO-Inference-style weight streaming from the L1/L2 tiers (recommended)
- B) FSDP-style per-layer all-gather of the weights
- C) Both

**Answer (2026-09-28, user): skip it.** No ZeRO/FSDP or weight-streaming mode.

## P5: small-message all-reduce latency on novanas — host-memory all-reduce?

Found 2026-09-28: RCCL 2.30.4 all-reduce took ~1.0–1.4 ms for 8 B – 64 KiB and 2.8–11 ms for 128–512 KiB on the two R9700s. Host facts (read-only check, user-confirmed): i5-14400T, the x16 bifurcated x8/x8 over root ports 00:01.0 / 00:01.1; GPU0 Gen5 x8, GPU1 Gen4 x8 (a board limit); VT-d on, one IOMMU group per GPU; large BAR on (32 GiB); the Debian kernel has no `CONFIG_HSA_AMD_P2P` and KFD lists no GPU↔GPU p2p link — **no peer-to-peer is possible on this board**, so vLLM custom all-reduce / quick reduce (IPC) are out.

Measured cause (`scripts/lab-cluster.sh collbench-sweep-novanas`, bench-lock held): RCCL's LL protocol for small messages. BF16 all-reduce, op + synchronize / back-to-back per op: default 8 B 1,034 / 1,078 µs, 64 KiB 1,409 / 1,388, 1 MiB 257 / 259; `NCCL_PROTO=LL128` 47 / 22, 50 / 27, 258 / 251; `Simple` 49 / 23, 50 / 29, 256 / 252. `NCCL_ALGO`, MSCCL / MSCCL++ off, one channel, SHM memcpy and SDMA off change nothing. Fix landed: the `rccl` backend sets `NCCL_PROTO=^LL` unless the operator set it (commit "RCCL excludes the LL protocol by default").

**User decision (2026-09-28): build a host-memory all-reduce ("can we do hostmem like vllm does?").** A `hostmem` collective backend in the `collective_backend` registry: pinned host memory mapped into both devices, one-shot all-reduce (each rank writes its partial to its slot, publishes a sequence-numbered flag with system-scope release, polls the peer's flag with acquire, reads the peer's partial and sums in a fixed rank order in the same kernel; double-buffered slots; no CPU on the path), also all-gather / reduce-scatter / all-to-all on the same buffer; broadcast and large messages may stay on RCCL through a size threshold chosen per call with a reason code; a bounded spin that errors instead of hanging, tied into `step_begin` / `step_end`; bit-exact against the host backend with the same summation order on both ranks. Reuse first: RCCL's no-p2p path is the SHM transport measured above; no existing host-mapped one-shot all-reduce kernel was found for gfx1201 without p2p (vLLM custom all-reduce, quick reduce and mscclpp need p2p/IPC). 2-GPU numbers are labelled "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)"; a 2-rank collective runs at GPU1's link speed.

Attribution (corrected 2026-09-28): the user's decision is only "do hostmem like vLLM does". The condition "build `hostmem` if RCCL stays above ~50 µs for 8 B – ~1 MB messages" and the design details above came from the Phase 5 coordinator, not the user. After the `^LL` fix RCCL measures ~21 µs (8 B – 32 KiB), 27 µs (64 KiB), 38 µs (128 KiB), 61 µs (256 KiB), 123 µs (512 KiB) and 251 µs (1 MiB) back to back; `hostmem` is built as the user decided either way.

## P5: tensor-parallel accuracy bound

Asked 2026-09-28. Tensor parallelism is not bit-exact with one device: each all-reduce sums BF16-rounded partial results, where one device rounds once after an F32 accumulation over the whole reduction. On the tiny checkpoints (cpu-reference provider, tp 2 and 4) every greedy token is identical to tp 1 and the logits move by 0.95–1.2 % relative to the row's largest |logit|, so the plan's 1e-4 raw-logit bound for `tp2_matches_tp1_on_host` cannot hold.

- A) (recommended) golden tolerance: the served tp 2 model must pass `turbine-golden compare` at concurrency 1 (strict bounds) and 16 (batched bounds), as for batching changes; the tiny-model test checks exact greedy tokens and a logprob comparison (top-k |Δ logprob| within the golden bounds) instead of raw logits
- B) a raw-logit bound around 2 % relative plus the golden check
- C) bit-exact TP against tp 1 (a fixed-order FP32 reduction of full partial sums; costs bandwidth and still differs in GEMM shapes)

**Decision (user, 2026-09-28): A.** The spec's and plan's 1e-4 logit bound is replaced by the golden check (c1 strict, c16 batched) for tp 2 serving; `tiny_model tp2_matches_tp1_on_host` and the lab `hip_tp2_matches_tp1` require exactly identical greedy tokens, bitwise-identical logits on every rank, and each row's top-5 logprobs (of tp 1) within the strict bounds of `tests/golden/llama-3.2-3b-instruct/tolerance.json` (0.15 above logprob −2, 0.55 below); a group of one stays bitwise equal to one device.

## P5: PP activation hand-off, EP combine and the supported parallel combinations (implementation choices, 2026-09-28)

Within the fold-in decision above (PP and EP in Phase 5, sharded DP skipped), the Phase 5 lead proposed the following (spec S-10 to S-14, plan Tasks 20–27). Alternatives considered: EP all-to-all dispatch now; allowing TP × PP combinations.

- PP hand-off: `Collective::send` / `recv` (point-to-point on the group's backend: `hostmem` slot, `ncclSend` / `ncclRecv` on RCCL, the host backend in tests) rather than the rank transport, so the hidden state never leaves the device path through a socket and `static` mode needs nothing new; PP runs in `local` mode only in Phase 5.
- EP combine: tokens stay replicated on every EP rank (attention replicated at tp = 1, TP-sharded at tp = ep), so dispatch is a local selection and combine is one FP32 all-reduce in fixed rank order — the "all-gather + reduce-scatter" option; all-to-all dispatch between data-parallel attention ranks is deferred (it needs lockstep DP engines). The router runs identically on every rank, so its choices are exact.
- Supported combinations in Phase 5: tp × dp; pp × dp with tp = ep = 1; ep × dp with tp ∈ {1, ep}. Anything else is refused at startup with `combination_unsupported:<modes>` (two cards cannot validate pp × tp).
- PP stage placement on novanas's asymmetric slots: the last stage (logits or their device reduction, plus its KV tier copies) goes on the GPU with the fastest measured host link (GPU0, Gen5 x8), reason `pp_stage_host_traffic:<device>`.

**Decision (user, 2026-09-28): all four accepted.** PP send / recv on the `hostmem` slots, with `ncclSend` / `ncclRecv` as the fallback, `local` mode only; EP with replicated tokens and an FP32 all-reduce combine, all-to-all deferred; the supported-combination set, everything else exits 2 at startup; the last PP stage on GPU0. Not taken: EP all-to-all now, TP × PP combinations.

## P5: collective failure recovery

Asked 2026-09-28 (after the first tp2 serving run). A collective timeout or async error mid-generation aborts the group's communicator; the spec's failure mode says the phase-3 breaker's `PROBING` re-creates it, which the tp2 wiring does not do yet.

- A) leave the replica `CIRCUIT_OPEN` until a restart
- B) re-create the communicator while probing: swap every rank executor's collective for a new one (fresh unique id, bounded init) and run a re-init step plan, then the ordinary probes
- C) exit 3 and let the supervisor restart the process

**Decision (user, 2026-09-28): B, with C as the interim** ("Re-create, exit 3 meanwhile"). A collective failure exits 3 now; re-creation is a Phase 5 task (plan Task 28), after which the interim exit is dropped.

## P5: bit-exact prefix reuse under tensor parallelism

Asked 2026-09-28. At tp 1 Llama's prefix reuse is bit-exact (the invariant GEMM rows of `kernels/rocm/tuning/gfx1201/gemm.tsv`, decision "Pre-Phase-5 #1 follow-up (A)"); at tp 2 every rank's projections have half-size shapes the table has no rows for, so the library's heuristic choice may change with m and a reused prefix may differ from a cold one.

- A) accept the golden bound only (no bit-exactness claim under TP)
- B) tune invariant GEMM rows for the TP shapes

**Decision (user, 2026-09-28): B.** Add batch-invariant rows for the tp 2 shapes of Llama and OLMoE (prefill and decode) to `kernels/rocm/tuning/gfx1201/gemm.tsv`; the target is `kv_gpu` prefix reuse bit-exact at tp 2 as at tp 1, with throughput re-measured (plan Task 29).

## P5: KV tiers in static rank mode

Asked 2026-09-28. In `local` mode the leader drives every rank's L1/L2 copies (T17b); in `static` mode the ranks are separate processes, and the first wiring turned the tiers off there.

- A) keep tiers off in `static` mode for now
- B) build tiers in `static` mode too

**Decision (user, 2026-09-28): B.** Worker processes manage their own per-rank L1/L2 copies, driven by the leader over the rank link (protocol messages for promote, demote and tier copies), symmetric with `local` mode; the tp2 static lab scenario gains the Phase 4 multi-turn check (plan Task 30).

## P5: KV admission across tensor-parallel ranks

Asked 2026-09-28. The tp2 wiring reserves each admission's worst-case KV on the leader's ledger only (the pools agree on the minimum block count at startup).

- A) keep the leader's ledger as the group's
- B) reserve on every rank's ledger per admission

**Decision (user, 2026-09-28): B.** Admission calls `reserve_group` across every rank's ledger, atomically with rollback on a partial failure, covered by a simulator test with unequal pools (plan Task 31).

## P5 Task 25: EP expert-subset GEMMs for OLMoE — provider evaluation (kernel reuse rule)

Measured 2026-09-28. Kernel ABI v2 already gives every `moe_experts` provider (cpu-reference and all five HIP implementations: small-m, WMMA prefill, WMMA, `hipblaslt_per_expert`, `hipblaslt_grouped`) a local expert range `[expert_begin, expert_end)`; the positions kernel marks rows routed outside it as −1, so they are never gathered, multiplied or scattered. No ABI change is needed.

`turbine-model --test perf moe_ep_local_timings` (commit f2b2812), OLMoE-1B-7B real routing from the traced 2,048-token golden prefill (its first 16 tokens stand for a 16-sequence decode step), real expert weights of layers 0/5/10/15, the library's default choice, natively on novanas GPU 0 under `scripts/bench-lock.sh`; µs per `moe_experts` call over those layers:

| Case                                                               | 16 tokens (128 routed rows) | 2,048 tokens (16,384 routed rows) |
| ------------------------------------------------------------------ | --------------------------- | --------------------------------- |
| All 64 experts (one device, or remote experts at weight 0)         | 966–1,207                   | 2,493–2,559                       |
| EP 2 rank 0, experts 0–31                                          | 504–639                     | 1,279–1,331                       |
| EP 2 rank 1, experts 32–63                                         | 576–690                     | 1,284–1,415                       |
| Interleaved placement, rank 0's even experts as 32 one-expert runs | 1,097–1,388                 | 2,992–3,096                       |

Bitwise at every layer and size: range `[0,32)` then `[32,64)` into one accumulator equals the 64-expert call, and so do the 64 one-expert runs in ascending order (the scatter adds each token's experts in ascending order either way).

- A) the existing local expert range with the rank's own weight stacks, one `moe_experts` call per run of consecutive expert ids: halves the MoE cost per rank, bitwise-compatible with one device, no ABI change (recommended)
- B) a global→local id map with remote experts at weight 0: the full 64-expert cost, no saving
- C) CK `moe_sorting` with a local-expert mask: replaces only the `moe_route` sort, and CK's fused MoE does not build on gfx1201 (MFMA 32×32×8; "Pre-Phase-5 #2")
- D) vLLM `fused_moe` `expert_map`: the same idea as A for arbitrary maps, but its GEMMs are CUDA / gfx9 / Triton and its align/permute helpers duplicate `moe_route`
- E) llama.cpp `mul_mat_id`: no expert parallelism (ids index the full weight tensor), 2.9× slower at 16k rows and not invariance-compatible ("Pre-Phase-5 #2")

**Choice (Phase 5 lead, per the kernel reuse rule, 2026-09-28): A**, with contiguous placements (the default). Placement files are supported and exact, but fragmented runs cost as much as the full call (last row), so `docs` and the `parallel.expert.placement` description steer files towards contiguous blocks.

## P5 Task 21: `hostmem` collective — provider evaluation and measured result (kernel reuse rule)

Measured 2026-09-28 (branch p5-hostmem 55f8def, merged at 72c0995). Reuse first: no existing provider runs a one-shot collective on this board — vLLM custom all-reduce, quick reduce and mscclpp all need peer access / IPC, which the novanas board does not allow; RCCL's no-p2p path is its SHM transport, the baseline. So Turbine's own kernels (kernel ABI v2.7, `kernels/rocm/src/hostmem.hip`, the evaluation also recorded there): host-mapped pinned slots, sequence-numbered flags with system-scope release/acquire, fixed rank-order FP32 reduction — bitwise equal to the `host` backend on both R9700s for every op, BF16/F32, 1–20,000,001 elements; a missing peer ends in 536–658 ms at a 500 ms op timeout; `abort` releases a spinning peer.

`scripts/bench-lock.sh scripts/lab-cluster.sh collbench-hostmem-novanas`, 2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8), BF16 all-reduce µs (op + synchronize median / back to back mean): 8 B rccl 48.3 / 27.5, hostmem 28.9 / 10.1; 16 KiB 44.8 / 22.9 vs 35.4 / 12.0; 64 KiB 51.0 / 30.1 vs 53.0 / 38.0; 128 KiB 64.5 / 40.6 vs 67.8 / 43.4; 1 MiB 261 / 252 vs 221 / 215; 16 MiB 4,299 / 4,270 vs 4,621 / 4,599; 1 GiB 249 ms vs 280 ms (labbook runs 97384e81, d2649602, 850e87a7; full table in `docs/extending/collective-backend.md`). Kernel access to mapped host memory tops out near 6.3 GB/s (all-reduce bus bandwidth ≈ 3.8 GB/s), about half the 12.5–13 GB/s the DMA copy engine reaches (host-link probe); wider grids, non-coherent memory and more loads in flight gained ≤ 5 %.

Result: `hostmem` wins 1.5–2.7× below ~64 KiB (back to back), ties or loses at 64 KiB–16 MiB (the TP decode all-reduce, ~96 KiB, gains 0–10 %), and RCCL is 5–25 % faster above 2 MiB. Chosen (Phase 5 lead): `hostmem` routes per call — `parallel.collective.hostmem_max_bytes: auto` keeps all-reduce ≤ 128 KiB, all-gather / reduce-scatter ≤ 256 KiB and broadcast / send / recv ≤ 32 KiB on its kernels and hands larger messages to an RCCL delegate (`turbine_collective_route_total{op,backend,reason}`); `auto` backend = `hostmem` for AMD plans in `local` rank mode (registry order host, hostmem, rccl, nccl), `rccl` in `static` mode (`hostmem` there exits 2). Deviation from S-14: on a kernel library without ABI v2.7 `hostmem` hands the whole communicator to RCCL (reason `op_unsupported`, logged) instead of the planner refusing it. Every NCCL-API communicator init (plain `rccl` too) now runs on a helper thread abandoned after the init timeout + 5 s, because RCCL's init with a missing peer never returns (a lab run hung 25 min before this bound). Open for later: a DMA-engine (copy-stream) path for the 64 KiB–16 MiB range, where the copy engine is ~2× the kernels' mapped-memory rate.

## P5: group KV admission in static rank mode

Asked 2026-09-28 (Task 31 as built, branch p5-group-reservation 3d069c2). In `local` mode admission reserves each request's worst-case KV on every rank's ledger atomically (user decision "P5: KV admission across tensor-parallel ranks", B). In `static` mode the worker ranks' ledgers live in the worker processes, so the leader cannot reserve on them; admission there still uses the leader's ledger only.

- A) keep leader-only admission in `static` mode
- B) the leader keeps a mirror ledger per worker, built from the budget each worker sends in its join message (one more `Hello` field); admission stays atomic inside the leader process (recommended)
- C) add reservation messages to the rank protocol (a round trip per admission)

Recommendation: B — atomic in one process, no per-admission round trip; the worker's own ledger still guards its allocations. Pending the user's answer; `static` mode keeps A meanwhile.

**Decision (user, 2026-09-28): B, mirror ledgers.** Each worker sends its budget in its join message and the leader keeps an exact mirror ledger per worker, reserving through `reserve_group` as in `local` mode. A host or simulator test shows that after a mixed workload with cancels and preemption, each mirror equals the worker's real ledger. Workers check that equality periodically: a ledger digest travels in the step acknowledgement, and a mismatch is logged with a reason code (plan Task 33).

## P5: OLMoE golden tolerance under expert parallelism

Found 2026-09-28 (ep2-novanas, run 0928064441-1e2319e6, branch p5-ep-server 3d1b8d8). OLMoE at ep 2 (tp 1) fails the committed golden reference on one prompt's logprob bound only: p14's likely |Δ logprob| is 1.0246 against the 1.01 bound, identically at c1 and c16, over RCCL and hostmem. One device (ep 1) passes 15/16 with p14 at 1.0003 — 0.007 under the bound — because it diverges from the reference at token 14 and its later positions are never scored, while ep 2 follows the reference to token 27 (an excused near-tie, margin 0.059) and is scored on positions 14–26. Against a capture of ep 1's own output, ep 2 passes 16/16 at the strict bounds (p14 |Δ| 0.107). OLMoE's 1.01 bound is transformers' own run-to-run spread (tests/golden/olmoe-1b-7b-0125-instruct/README.md). ep 2 with tp 2 fails p10 the same way plain tp 2 already does (Tasks 29 / 32).

- A) first confirm per position that the violation lies in positions 14–26 of p14 (which only ep 2 reaches), then decide (recommended)
- B) re-calibrate OLMoE's tolerance for the parallel modes (a wider likely bound, measured with the self-spread method over the parallel outputs)
- C) gate EP on its one-device capture: ep N must match an ep 1 capture within the strict bounds, plus the committed-reference check with the margin excuse extended to positions one device never reaches
- D) accept the failure as a known exception for p14

Recommendation: A, then C if A confirms (EP changes no arithmetic beyond the fixed-order combine, so one device is the right reference; the committed reference stays the gate for one device). Pending the user's answer; the ep2-novanas scenario reports the committed-reference verdict and the ep 1 comparison side by side.

**Decision (user, 2026-09-28): A, then C** ("Inspect, then gate vs 1 GPU"). Options shown: inspect, then gate vs 1 GPU (recommended); re-calibrate the OLMoE tolerance; known exception for p14; inspect only. First the per-position analysis of ep2 p14: which position exceeds 1.01, which candidate (top-5 rank, reference logprob, near-tie or tail), and how far ep1 is from the reference at the same positions. If it is benign (ep2 no further from the reference than ep1 at the same positions), then multi-GPU modes (EP, and TP / PP under the TP accuracy decision) are gated strictly against a committed one-GPU capture of the same model and commit, at c1 strict and c16 batched, captured in the same lab run; one GPU stays gated against the transformers reference, and the multi-GPU verdict against the transformers reference is reported for information only. The ep2 × tp2 p10 failure is tracked with plan Tasks 29 / 32.

**Follow-up (user, 2026-09-28): accept p14 as benign and keep the one-GPU-capture gate** ("Accept, gate vs 1 GPU"; the other option was to investigate the 0.024 first). Evidence (ep2-novanas run 0928071634-3f9f6c29, commit 1c16934, `turbine-golden positions` teacher-forced on the reference): the violation is at position 11 of p14 — inside both runs' shared prefix, not at positions 14–26 as first guessed — reference token 187 (margin 1.468, no near-tie), candidate #2, id 346, reference logprob −1.8486 (likely tier); ep1 −2.8489 (|Δ| 1.0003, passes), ep2 −2.8732 (|Δ| 1.0246, fails), ep2 − ep1 0.024; ep2 against the ep1 capture passes 16/16 at strict bounds (c1) and batched (c16), over rccl and hostmem. The test "ep2 no further from the reference than ep1" was the coordinator's over-strict reading; the working rule is **a multi-GPU run lies within the strict bounds of the one-GPU capture** (c1 strict, c16 batched). OLMoE tp 2 p10 (divergence at token 21, margin 1.3 — not a near-tie; ep2 × tp2 against the ep1 capture likely |Δ| 1.36 / 1.33) is the main open accuracy item for the Phase 5 exit, owned by plan Task 29: fix or explain.

## P5: tensor-parallel accuracy gate against the one-GPU capture

Asked 2026-09-28 (Task 29 results). Against a one-GPU capture of the same commit taken in the same lab run, tp 2 misses the strict c1 bound for both models, with and without the Task 29 GEMM rows: Llama p11 likely |Δ| 0.174 (bound 0.15) and p16 tail 0.5526 (bound 0.55); OLMoE p10 likely 1.198 (bound 1.01) and p08 diverging at token 25 (margin 0.52). RCCL and hostmem give bit-identical results, so the collective is not the cause; tensor parallelism rounds partial sums differently (~1 %), which shows at low-margin positions. EP and PP change almost no arithmetic (PP is bitwise one GPU; EP's ep2 − ep1 was 0.024 at the worst position).

- A) gate TP against the one-GPU capture with the batched bounds at c1 and c16, plus the transformers golden (c1 strict, c16 batched) as in the original TP decision; EP and PP stay strict against the capture (recommended)
- B) keep the strict c1 bound against the capture and investigate the TP drift further
- C) gate TP against the transformers reference only (the original TP decision)

**Decision (user, 2026-09-28): A.** A multi-GPU mode that includes tensor parallelism (tp > 1, including ep × tp) is gated against the one-GPU capture with the batched bounds at c1 and c16 and against the committed transformers reference with the golden rule (c1 strict, c16 batched); expert and pipeline parallelism without TP stay strict against the capture (c1 strict, c16 batched).

**Follow-up (user, 2026-09-28): (a) batched limits for TP against the transformers reference too.** Options shown: batched limits for TP (recommended); keep strict and investigate Llama p16; HF informational only. Evidence: under rule A as first written, Llama tp 2 against the transformers reference at c1 strict fails on p16's tail candidate, |Δ| 0.584 against the strict bound 0.55 (within the batched 0.75); one bound violation fails a golden run whatever `min_prompts_passing` says. The rule now: every mode that includes tensor parallelism is judged with the batched bounds at c1 and c16 both against the one-GPU capture and against the committed transformers reference (`turbine-golden compare --batched-bounds`); one GPU keeps the strict transformers gate at c1; expert and pipeline parallelism without TP stay strict against the capture.

**Follow-up (user, 2026-09-28): OLMoE under TP is gated against the transformers reference only; its one-GPU-capture leg is informational; the drift is investigated later.** Options shown: HF only for OLMoE TP, investigate later (chosen); calibrate TP bounds from the measured spread; investigate now. Evidence (tp2-novanas on 8c8ccce, 2-GPU GPU0 Gen5 x8 + GPU1 Gen4 x8): OLMoE's batched bounds equal its strict ones (likely 1.01, tail 1.66 — calibrated to transformers' own spread), so rule (a) gives it no room; against the one-GPU capture p10 has likely |Δ| 1.198 with 32/32 identical tokens (not a near-tie) and p08 diverges at token 25 (margin 0.52, just over the 0.5 excuse); against the transformers reference OLMoE tp 2 passes 16/16 at c1 and c16. The rule now: for OLMoE in any mode that includes TP, the transformers golden (batched bounds per (a)) is the gate and the one-GPU-capture leg is reported as `info olmoe_tp_capture`; Llama TP keeps both legs as gates; EP and PP stay strict against the capture. Tracked for a later phase: investigate the OLMoE TP drift (hypothesis: expert-routing near-ties amplify the ~1 % partial-sum rounding of tensor parallelism; a routing trace per position of p10 / p08 would show it).

## P5 Task 32 (b): hostmem copy-engine all-reduce — measured, kept off

Built 2026-09-28 (p5-tp-perf-clean 21dfec0): kernel ABI v2.8 `turbine_mapped_all_reduce_dma` (+ `turbine_host_alloc_dma`): each rank's chunks leave the device on a copy stream into host slots (a flag per chunk after its copy), the reduction waits for every rank's flag and combines in rank order 0..world−1 — reading the peers' chunks from the mapped slots (default) or after copying them in (A/B `--hostmem-dma-copy-in`) — bitwise the one-shot step and the host backend (lab `hostmem_dma_matches_host_backend_on_two_gpus`, BF16/F32 up to 20,000,001 elements). Routed by `parallel.collective.hostmem_dma_min_bytes` (reason `copy_engine`; never while a graph is captured).

Measured (`scripts/lab-cluster.sh --bench-lock collbench-hostmem-novanas`, 2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8), BF16 all-reduce, op + synchronize median; labbook 7aa1231b, fba16d16, 06217c91, 347e3caf): 16 MiB — rccl 4.24–4.28 ms, hostmem one-shot 4.64 ms, copy engine 4.53–4.67 ms (coherent or portable slots, copy-in or peer-read); 1 MiB — rccl 263 µs, one-shot 224 µs, copy engine 376–472 µs; 256 MiB busbw 3.7–4.0 GB/s for every all-reduce transport, broadcast 8.3 GB/s. One rank's copies out alone: 16 MiB in 1.79 ms (9.4 GB/s).

Finding: a 2-rank all-reduce through host memory makes four transfers of the message (each GPU out and in); every transport lands near 4 × 16 MiB / 4.3 ms ≈ 15 GB/s of aggregate host traffic (broadcast, two transfers, the same aggregate). The cap is in the host path (root complex / IOMMU / memory), not in RCCL or our kernels: no transport change can beat it. The tp2 prefill all-reduce floor for a 2,048-token Llama chunk is therefore ≈ 57 × 3.45 ms ≈ 198 ms (the Task 32 profile's ~200 ms), not the ~150 ms estimated for (b); the remaining lever is hiding it behind compute ((c), at most the ~50–58 ms of compute per chunk) or moving less.

Decision (Phase 5 lead, 2026-09-28): no end-to-end tp2 bench of (b); `hostmem_dma_min_bytes` stays off (null) by default; the code stays behind its option.

(c) measured in the same series (`scripts/lab-cluster.sh --bench-lock tp2-novanas`, run 0928110628-273fa1ed, 2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8), Llama c16; labbook f8fa3766, e98a9877, ee74de42): tp1 858.0 tok/s / TTFT p50 208.6 ms / ITL 15.42 ms; tp2 914.4 / 445.2 / 11.84; tp2 with `parallel.tp_prefill_overlap` 1007.7 / 274.0 / 11.82 (+10.2 % tok/s, −38 % TTFT); bitwise equal to the unsplit run; golden c1/c16 16/16 against the capture and HF (batched bounds). (a) decode graphs: ITL 11.84 → 11.52 ms, tok/s within noise; off by default.

## P5 Task 32 (c): tensor-parallel prefill overlap on by default?

- Default on, `parallel.tp_prefill_overlap: false` turns it off (recommended: bitwise equal to the unsplit run, golden-clean, +10.2 % tok/s and −38 % TTFT at tp 2 on Llama; dense-FFN families only)
- Keep it opt-in (default off)

**Decision (user, 2026-09-28): default on.** `parallel.tp_prefill_overlap` defaults to `true`; the switch stays to turn it off; the Phase 5 exit bench runs with it on.

**Superseded (user, 2026-09-28): A now, then B.** Options shown: off now, fix, then on (recommended, chosen); keep on and fix before the exit; keep on and relax exactness. Evidence: the Phase 5 exit gate 3 (`lab-test novanas --gpus 2 --features fault-injection --tier full`, run 0928125603-04f6f801, 599e746) failed kv_gpu `prefix_reuse_suffix_lengths_match_cold_tp2` in 6 of 7 cases (warm vs cold differ at token 0, |Δ logprob| 0.005–0.06) with the overlap on; the same test with `parallel.tp_prefill_overlap=false` passes — the default conflicted with "P5: bit-exact prefix reuse under tensor parallelism" (B). `parallel.tp_prefill_overlap` is off by default again; plan Task 34 (tp-perf) splits the overlapped prefill at a `kv.block_tokens` boundary with batch-invariant GEMM rows for the half shapes, and re-enables the default only once that kv_gpu test and golden pass at tp 2, with a re-bench (inside Phase 5 if quick, else a Phase 5p item; the exit does not wait for it).

## P5 Task 32 (d): per-shard logits reduction before the gather

- A) exact-only: merge per shard only for rows with no categorical draw and no log-sum-exp consumer (a new row flag the server sets); the standard bench (sampled) and golden (logprobs) gain nothing
- B) per-shard log-sum-exp and draw merge, not bitwise (ulp-level logprob differences, a different draw rounding), judged by golden
- C) defer (d) (recommended)

**Decision (user, 2026-09-28): C, defer.** Reason: the full-row log-sum-exp and the categorical draw are one full-row reduction whose summation order changes per shard, so they cannot be reproduced exactly per shard; the saving is ~1.2 ms per decode step (the tp 2 logits all-gather).

## Lab disk: pruning stale remote build caches on novanas

Context: novanas reached 95 % disk and k3s evicted a lab Job (2026-09-28). The Phase 5 exit-prep agent then deleted 24 stale `remote-cargo` target directories (~135 GB) **without asking first**; that was outside the lab rules at the time and was raised with the user.

- Auto-prune before each lab run: the `target/` of any `/home/piwi/turbine-ci/remote/agent-*` workspace with no matching local worktree under `.claude/worktrees/` and no writes for 12 h, logged with the bytes freed (recommended)
- Prune only by hand, asking the user each time

**Decision (user, 2026-09-28): auto-prune approved**, with these limits: only `remote/agent-*/target`; never the per-slot caches under `turbine-ci/cache/`, anything in use, or anything else; every removal logged with the bytes freed. This rule now covers the earlier manual prune after the fact; any prune beyond it still needs the user.

## Closing Phase 5: recheck, merge and push

- Close: quick recheck then merge (chosen); full reruns then merge; merge now
- Push: push after the merge (chosen); do not push yet

**Decision (user, 2026-09-28): quick recheck, then merge; push after the merge.** The recheck is `scripts/lab-bench.sh --golden16` for Llama and OLMoE on GPU 0 on the final tip (golden c1 16/16 for Llama, OLMoE ≥ 14/16, tok/s ≥ 0.97× the exit numbers 855.3 / 613.8), then `scripts/gate.sh --full`, the merge of `phase-5-multi-gpu` (with the Phase 5p docs of `phase-5p-docs`) into main, a gate on main, a gitleaks scan, and `git push origin main` only. The one-GPU full tier (stopped at the user's request with 33 binaries done and no failure) is re-run in release mode as Phase 5p task 1; the vLLM two-GPU matrix was stopped early at the user's request (partial results in labbook set `phase-5-vs-vllm`).

## Lab-test release profile

- A `--release` option for `scripts/lab-test.sh`, the default for `--tier full`, built in the same cached slot target dirs (first item of Phase 5p)
- Keep the debug profile for every tier

**Decision (user, 2026-09-28): add it right after the Phase 5 exit, as the first item of Phase 5p.** The full tier takes 40+ min in the debug profile. `--release` becomes the default for `--tier full`; the build uses the same cached slot target dirs (the release artifacts live beside the debug ones). The Phase 5 exit runs are not changed.

## P5 exit: `parallel.collective.verify` as a diagnosis mode

Built at the coordinator's request during the collective-corruption diagnosis (b38ec2b): each collective's output is checksummed across ranks and a mismatch fails the step with `collective_corrupt` (metric `turbine_collective_errors_total{kind="corrupt"}`, WARN `collective_corrupt`; WARN `collective_verify_on` at startup). **Status (coordinator, 2026-09-28): a diagnosis/canary mode, default off.** Cost measured at tp 2: ~237 vs 912 tok/s. It ran the proof's tp 2 verify leg (~657,000 collectives cross-checked, no mismatch).

## P5 exit: vLLM-ROCm baselines for the two-GPU modes

- Run vLLM-ROCm two-GPU baselines before the merge-to-main decision
- Skip them (Phase 5 compares multi-GPU modes against Turbine's own one-GPU numbers only)

**Decision (user, 2026-09-28): run them**, strictly after the Phase 5 proofs (the verify-on tp2 run and the gate 3 rerun) and before the merge-to-main decision, in queue order and under the benchmark lock for the whole run: a two-GPU variant of the pinned vLLM-ROCm Job (`amd.com/gpu: 2`, same `rocm/vllm` tag) and a `lab-serve.sh --vllm` option for the GPU count and extra vLLM arguments; the standard workload (512-word prompts, 256 tokens, 200 requests, `--ignore-eos`) at c16 and c32; Llama at `--tensor-parallel-size 2`, `--pipeline-parallel-size 2` and data parallel (`--data-parallel-size 2` if the pinned version supports it on ROCm, else two one-GPU instances behind a round-robin, said which); OLMoE at `--tensor-parallel-size 2`, `--enable-expert-parallel` with TP 2, and DP 2; the one-GPU vLLM baselines (Llama, OLMoE) re-run in the same session. Recorded in labbook (set `phase-5-multi-gpu`, vLLM version and flags) with a Turbine vs vLLM table per mode (tok/s, TTFT p50, ITL p50, ratio) and vLLM's fallbacks noted (e.g. no custom all-reduce without peer-to-peer). A mode vLLM cannot run on this board is recorded with its error and skipped, not debugged at length. Plan Task 35.

## P5 exit: OLMoE with expert × tensor parallelism

- (1) Mark OLMoE ep × tp `experimental` (recommended)
- (2) Refuse it for OLMoE (`unsupported`) until phase 7 fixes the OLMoE tensor-parallel drift
- (3) Fix the drift before the Phase 5 exit

**Decision (user, 2026-09-28): (2) refuse.** Evidence (tp-perf ep2 proof on 31fa6b0, and the same p10 failure on 3d1b8d8 and 1c16934, before the pinned bounce buffer): the ep 2 × tp 2 OLMoE leg fails the transformers golden on p10 (likely |Δ| 1.031 > 1.01) and p08, while plain EP and plain TP pass. `OlmoeForCausalLM` with ep > 1 and tp > 1 is `unsupported` with reason `olmoe_ep_tp_drift` (`turbine_core::support::PARALLEL_REFUSALS`, shown by `--support-matrix`), refused at startup with exit 2 naming `parallel.expert_parallel_size` before any port is bound; plain EP and plain TP for OLMoE stay supported; dense models are unaffected (EP does not apply to them). The ep2-novanas scenario checks the refusal instead of serving ep 2 × tp 2. Re-opened by the phase 7 investigation (expansion umbrella question (d)).

## P5 exit: "collective corruption" was wrong bytes in pageable device-to-host copies (unexplained; avoided by pinned staging)

Context (2026-09-28): the Phase 5 exit gate 3 (`lab-test novanas --gpus 2 --features fault-injection --tier full`, run 0928125603-04f6f801) failed `hostmem_lab` `hostmem_matches_host_backend_on_two_gpus` and `hostmem_graph_replays_with_rccl_delegate_on_two_gpus` with wrong bytes; the exit was blocked as a possible silent collective corruption in every collective mode (TP, EP, PP).

Evidence (tp-perf diagnosis runs diag1–diag6, 2 GPUs):

- RCCL ran RING / SIMPLE under every `NCCL_PROTO`; send-buffer reuse, buffer registration (`NCCL_LOCAL_REGISTER=0`, `NCCL_GRAPH_REGISTER=0`), our allocator (plain `hipMalloc`), the binding's sizing and the stream handles (each rank's non-blocking compute stream, non-zero and distinct) were all ruled out; hostmem one-shot steps were clean in steady state and on fresh groups (0/240, 0/120).
- diag5: every pageable write read back correctly (write_bad 0), but two consecutive pageable `read_bytes` of the same buffer — after the stream synchronize, nothing enqueued between them — disagreed 9–14 times.
- diag6 (the decider): 60 rounds × 2 ranks × 2 processes; pinned-staging reads 0/240 bad, pageable reads 15 and 11 bad; the garbage is always at an edge of the transfer (the first or last 56–1,016 FP32 words of an 8 MiB read), matches no data in the process, and hits either of two back-to-back reads.

diag7 (120 rounds × 2 processes, one device buffer read three ways per round): pageable into a fresh, never-touched `vec![0u8; n]` 20/240 bad; pageable into a Vec allocated once and already touched 0/240; pinned staging 0/240; host canary buffers never hit. The garbage sits in the tail of the fresh Vec's last partial page (24·round − 436 words, moving with the heap). Our side checked: `DeviceSlice::read_bytes` allocates `vec![0u8; self.len]` (zeroed, no `set_len`) and passes exactly `dst.len()` = the slice length to `turbine_memcpy_d2h` (buffer.rs `read_bytes`, shim.rs `copy_d2h`); an untouched destination would read back zeros, not garbage, if bytes were merely skipped.

**Status (coordinator, 2026-09-28): unexplained, avoided by pinned staging.** It is called a ROCm bug — and the upstream draft below is filed at all — only if a plain-HIP repro mirroring the Rust pattern exactly (same sizes, zeroed untouched allocation, same threads and devices, kernels busy) reproduces it; if a reasonable effort cannot, the user decides whether to file anything.

**Reproduced in plain HIP (2026-09-28, tp-perf `eae6c29`, `docs/upstream/rocm-pageable-d2h/`).** diag8: the wrong range is exactly the destination Vec's last partial page ((host end address mod 4096)/4 words in every bad round); two rank threads' heap allocations share that page. `repro.hip` (plain HIP, two threads, two GPUs, destinations back to back in one `calloc`'d region so they share a page, 2,000 rounds): `hipMemcpyAsync` shared busy 1, `hipMemcpy` (sync) shared busy 2 (earlier 4), async shared idle 1, pinned 0, separate allocations 0, single thread 0 — synchronous `hipMemcpy` into pageable memory is affected too. The fix is `2a26784` (every host copy through a per-context pinned bounce buffer). The upstream draft awaits the user's review; recommended repository ROCm/clr (the HIP runtime's pageable staging path).

**Draft reviewed (user, 2026-09-28): wait for the GPU proof.** Options shown: file now; wait for the GPU proof (chosen); do not file. Repository: ROCm/clr. The coordinator files it only after the pinned bounce-buffer fix has passed the two-GPU proof runs (dp2/tp2/ep2/pp2 golden and the read-twice verify); the one change to the draft is a sentence in its workaround section saying the workaround held in those runs, with the round count.

**Filed (2026-09-28): https://github.com/ROCm/clr/issues/291**, by the coordinator from the user's account, after the two-GPU proof (stress 221,000 outputs, tp2 verify ~657,000 cross-checked collectives, tp2/ep2/pp2/dp2 golden). The repository copy `docs/upstream/rocm-pageable-d2h/ISSUE.md` matches the filed text (the repro source appended).

Observation: `hipMemcpyAsync` device-to-host into pageable host memory, followed by `hipStreamSynchronize` (`kernels/rocm/src/memory.cpp` `turbine_memcpy_d2h`; `crates/turbine-kernels/src/shim.rs` `copy_d2h`), sometimes returns wrong host bytes at the transfer's edges while another thread copies on another device in the same process. The device data was always correct: the collectives were never wrong. Exposure: every mode with two devices in one process — TP, EP, PP and DP (two replicas on two threads) — wherever a pageable copy is on the path; single-GPU serving has one copying thread.

Plan (coordinator, 2026-09-28): audit every host↔device copy in the serving path (pageable vs pinned); move every hot or concurrent copy to pinned staging (or a synchronous copy only where the repro proves it safe); a shim guard refuses pageable async copies on GPU contexts (debug assert, counter, reason code); the lab harness reads back through pinned memory; proof: dp2/tp2/ep2/pp2 golden c1/c16, a long verify-mode read-twice run and the guarded serves, then the gate 3 rerun. A minimal standalone plain-HIP repro (two threads, two devices; pageable async, pageable sync, pinned and a single-thread control) is kept for an upstream ROCm report — path and results recorded here when it lands.

Upstream report — options shown: report upstream (a draft first), or keep it internal. **Decision (user, 2026-09-28): report it upstream, filed by the coordinator with `gh` under the user's account after the user has seen the draft.** No agent files anything. The draft (a plain-HIP `repro.hip`, the build line and `ISSUE.md`: environment, expected vs actual, variant results, frequency, the pinned-staging workaround; nothing Turbine-internal, no hostnames, IPs or credentials) goes under `docs/upstream/rocm-pageable-d2h/` once the repro reproduces on novanas.

Consequences for earlier work: `bffabda` (refusing RCCL inside a captured graph) was motivated by a pageable-read artefact; it is held until a pinned re-test of RCCL graph replays. The Task 34 attention diagnosis (split at a non-64-aligned q-start) is re-run with pinned reads to confirm.

## Roadmap reorganisation after Phase 5 (2026-09-28)

Asked 2026-09-28, while Phase 5 (multi-GPU, novanas) is being built on `phase-5-multi-gpu`. The user does not want multi-node or anything NVIDIA yet, and wants quantization and more model families brought forward. Before this, the order after Phase 5 was: Phase 6 multi-node (the two Sparks), Phase 7 advanced distribution (PP, EP, PD, RDMA, mixed vendor), Phase 8 expansion umbrella with tracks 8a quantization → 8b speculative decoding → 8c model families; Phase 2b NVIDIA on hold since 2026-09-26.

**1. Order after Phase 5**

- A) Phase 6 = quantization (AMD), Phase 7 = model families, Phase 8 = speculative decoding; multi-node (old 6), advanced distribution (old 7) and NVIDIA (2b) move to a deferred block, re-specced when un-held (recommended)
- B) Paired by model: each new family arrives together with the quantized format its checkpoints need
- C) Families first, then quantization, then speculative decoding

**Answer (2026-09-28, user): A.**

**2. Quantization formats on the R9700** (`gfx1201`, RDNA4: native FP8 WMMA, no FP4 matrix path) — multi-select

- FP8 weights (per-channel compressed-tensors and block-scaled Qwen3-FP8 style) + FP8 e4m3 KV cache
- MXFP4 (weight-only on RDNA4, dequantized to FP8/BF16 in the kernel)
- INT4 AWQ / GPTQ (weight-only, group scales)
- NVFP4 weight-only (the old Phase 8 scope)

**Answer (2026-09-28, user): FP8 weights + FP8 KV, MXFP4, INT4 AWQ/GPTQ.** NVFP4 is not selected. **Clarification (2026-09-28, user): NVFP4 support is added together with NVIDIA support** — modelopt NVFP4/FP8 mixed precision (`hf_quant_config.json`, `MIXED_PRECISION`) and compressed-tensors `nvfp4-pack-quantized` belong to the deferred NVIDIA block (phase-2b-nvidia), not to phase 6.

**3. Model families** — multi-select

- Qwen3 dense + Qwen3 MoE
- gpt-oss-20b (MXFP4 experts, attention sinks, alternating sliding-window / full attention)
- Qwen3.5 / 3.6 hybrids (Gated DeltaNet + full attention, recurrent-state KV)
- Mistral / Mixtral (Mixtral needs FP8 + TP to fit 2 × 32 GB)

**Answer (2026-09-28, user): all four.**

**4. Speculative decoding**

- A) After quantization and families, as the last active phase (recommended)
- B) Defer it with the multi-node block
- C) Before families (right after quantization)

**Answer (2026-09-28, user): A.**

Consequences (docs amended the same day):

- Active order: Phase 5 multi-GPU (in progress), Phase 6 quantization, Phase 7 model families, Phase 8 speculative decoding — all AMD only, on `novanas` (`gfx1201`). Track specs `phase-6-quantization`, `phase-7-model-families`, `phase-8-speculative-decoding` are each written when their phase starts, under the umbrella `.procoder/specs/phase-6-8-expansion.md` (renamed from `phase-8-expansion.md`, amended: AMD only, new order, new S-6 / S-8 scope).
- Deferred block, re-specced when the user lifts the multi-node / NVIDIA hold: phase-2b-nvidia (now also carrying NVFP4), `phase-9-multi-node` (was `phase-6-multi-node`), `phase-10-advanced-distribution` (was `phase-7-advanced-distribution`). The two renamed specs and plans keep their text as written and carry `Status: deferred` plus a numbering note; NVIDIA support-matrix rows stay `unsupported`.
- Phase 6 scope (umbrella S-6): FP8 e4m3 weights with per-tensor / per-channel scales (`fp8`) and block-scaled (`fp8_block`), FP8 e4m3 KV cache (`kv.dtype: fp8_e4m3`), MXFP4 (OCP, weight-only on RDNA4; `mxfp4`), INT4 AWQ (`awq_int4`) and GPTQ (`gptq_int4`), weight-only and group-wise. Support-matrix `weight_format` gains these five values; `modelopt_nvfp4`, `modelopt_fp8`, `modelopt_mixed` and `ct_nvfp4` stay reserved for the deferred NVIDIA block.
- Phase 7 families (umbrella S-8): Qwen3 dense and MoE, gpt-oss-20b, the Qwen3.5 / 3.6 hybrids, Mistral and Mixtral. This lifts the 2026-09-25 gpt-oss exclusion (it was out only because MXFP4 was out of scope).
- Phase 8 speculative decoding (umbrella S-7) now comes after the families, so recurrent-state rollback for the hybrids belongs to phase 8, not to phase 7.
- Open for the track specs (not decided here): the hybrids' cached checkpoints are NVFP4, so phase 7 needs FP8 or BF16 checkpoints of them (Qwen3.6-35B-A3B needs FP8 + TP 2 or similar to fit 2 × 32 GB); each phase-6 format is proven on a registered architecture (Llama-3.2-3B or OLMoE) where such a checkpoint exists — which ones is the phase-6 spec's choice (e.g. RedHatAI / neuralmagic FP8, AWQ and GPTQ Llama-3.2-3B checkpoints); MXFP4 checkpoints may exist only for gpt-oss, which needs the phase-7 family — the phase-6 spec decides between an MXFP4 fixture checkpoint quantized offline (Python quantizer at fixture-generation time only) and proving MXFP4 in phase 7 with gpt-oss.
- Code still names the old track names: the support-matrix refusal reasons in `crates/turbine-core/src/support.rs` (`phase-8a-quantization`, `phase-8b-speculative-decoding`, `phase-8c-model-families`) and the tests that assert them, and tests that use `GptOssForCausalLM` as the example of an unregistered architecture; and the four `nvidia`/`sm_121` × Llama / OLMoE BF16 baseline rows are still `supported` although phase-2b never ran (unreachable today: `execution.backend: cuda` exits 2). They change with the first code task of phase 6 (reasons, NVIDIA rows, the five new `weight_format` values) and phase 7 (the gpt-oss test name), not in this docs change.

## Phase 6 MXFP4: packaging formats and proof checkpoints (2026-09-28)

Found on Hugging Face (2026-09-28, real safetensors checkpoints, `config.json` inspected): OpenAI native `quant_method: mxfp4` (`openai/gpt-oss-20b`, Apache-2.0); compressed-tensors `mxfp4-pack-quantized` (float4 group 32, E8M0 `uint8` scales; W4A16 in `nm-testing/Qwen3-30B-A3B-MXFP4A16` and `FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16`); AMD Quark MXFP4 (`amd/Qwen3.5-35B-A3B-MXFP4`, `amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8`, `matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq`; mostly W4A4). Rejected as proof: `ISTA-DASLab/*-FPQuant-*` (Hadamard rotations, not plain MXFP4).

**1. Which MXFP4 packaging formats does phase 6 support?** — multi-select

- compressed-tensors `mxfp4-pack-quantized` (recommended)
- OpenAI native `mxfp4` (gpt-oss; loader in phase 6, proven with gpt-oss in phase 7)
- AMD Quark MXFP4 (mostly W4A4; activation FP4 emulated on RDNA4)

**Answer (2026-09-28, user): all three.**

**2. Which real checkpoint proves MXFP4 on a registered architecture?**

- A) `FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16` (compressed-tensors, `LlamaForCausalLM`, 5.8 GB) (recommended)
- B) `matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq` (Quark W4A4, `LlamaForCausalLM`)
- C) defer the first real proof to `openai/gpt-oss-20b` in phase 7

**Answer (2026-09-28, user): all three** — A and B are phase-6 proofs, C is the phase-7 proof of the OpenAI native packaging.

Consequences: the `phase-6-quantization` spec covers the three MXFP4 packagings under one `mxfp4` weight format (or one value per packaging, the spec's choice) and proves compressed-tensors on A and Quark on B; B's activation FP4 is emulated on RDNA4 (activations quantize-dequantized to FP4 before an FP8/BF16 matmul), and its row is `supported` only if it passes the S-3 gate against the checkpoint's reference output, else `experimental`. The OpenAI native loader may land in phase 6 under unit tests; its row turns `supported` when phase 7 closes gpt-oss-20b. Reference outputs for A and B are captured once with transformers (compressed-tensors / Quark dequantization, fixture generation only) and committed; quality is compared with BF16 `meta-llama/Llama-3.1-8B-Instruct` and `Llama-3.2-3B-Instruct`. The weights are downloaded on `novanas` with `hf download` into `/home/piwi/turbine-models/<slug>` like the Phase 1/2 weights.

## Phase 5p: serving efficiency interlude (SGLang-inspired items) (2026-09-28)

Context: comparing vLLM and SGLang, four SGLang ideas are missing from Turbine: (1) token-granular prefix matching (RadixAttention-style: reuse the partial last block of a cached prefix, copy-on-write; today only full 128-token blocks are shared, so up to 127 reusable tokens are lost per request); (2) a cache-aware scheduling policy (admit waiting requests with the longest cached prefix first, with a starvation bound; a new entry in the scheduling-policy registry); (3) jump-forward structured output (append llguidance's forced tokens in one forward pass through the chunked-prefill path instead of one decode step each); (4) CPU/GPU overlap scheduling (prepare step N+1 while the GPU runs step N; at c16 host overhead is ~0.2 ms of a 15.4 ms step, so measure at c1 first).

**Where do they go?**

- A) Interlude "Phase 5p: serving efficiency" after Phase 5 merges, before Phase 6, run like the pre-Phase-5 fixes (one change, then measure) (recommended)
- B) Split: 1 + 2 in the interlude, 3 in Phase 8 with speculative decoding, 4 measured in the interlude
- C) After Phase 7, across every KV format at once

**Answer (2026-09-28, user): A.**

Consequences: `phase-5p-serving-efficiency` is specced when Phase 5 has merged (spec and plan written then, under the usual chain) and runs before `phase-6-quantization`. Order inside it: (4) measurement first (host share of a step at c1 and c16 on Llama and OLMoE; build overlap scheduling only if the host share exceeds 5 % at c1), then (1), (2), (3), each landed alone with `scripts/lab-bench.sh` (golden c1 + throughput) and, for (1) and (2), the Phase 4 multi-turn profile (`cached_tokens_ratio`, later-turn TTFT) before and after. Constraints: (1) must keep the Phase 4 tiers (L1/L2 hold full blocks; a partial block is L0-only or copied whole) and the prefix-exact prefill invariance (the #1 follow-up), and Phase 7's hybrid recurrent state can be cached only at block boundaries, which the phase-7 spec handles; (2) is a registered scheduling policy with its conformance suite and a deterministic simulator test for starvation; (3) must keep outputs identical to token-by-token decoding under greedy (golden JSON-schema cases) and handle retokenization at the forced-span boundary.

## Startup time in the Turbine vs vLLM comparison package (2026-09-28)

Measured on novanas today (Llama-3.2-3B, TP 2): Turbine is ready 6–8 s after process start; vLLM-ROCm needs ~98 s (≈124 s from pod start), mostly spawning workers, torch.compile (11 s) and graph capture (49 s); weight loading is ~2.5 s for both.

- Add `startup_s` (process start → /ready 200) as a data point to `engine-comparison-multi-gpu` for every run, both engines (recommended)
- Leave it out; mention it only in the set conclusion

**Answer (2026-09-28, user): neither — it was a question, not a request; nothing is added to the tests or the package.**

## Phase 5p spec: provisional design choices (2026-09-28)

Written with `.procoder/specs/phase-5p-serving-efficiency.md` and its plan, ahead of Phase 5's merge. The decision "Phase 5p: serving efficiency interlude" fixes scope, order and constraints; the choices below are the ones it leaves open. The spec first wrote each recommended option marked "(provisional)"; the user answered all of them on 2026-09-28 (below) and the spec and plan now mark them "(user decision 2026-09-28)". None is implemented yet.

**1. Jump-forward and "identical to token-by-token decoding".** llguidance's forced tokens are the canonical tokenization of the forced bytes; greedy token-by-token decoding may pick another tokenization of the same bytes, and a prefill-shaped step rounds differently from a decode step.

- A) Verified jump-forward: feed the sampled token plus the forced tokens in one chunked-prefill step with logits for every position; step each row through the normal mask, processors and sampler (its own uniform), accept while the sampled token equals the forced one, take the model's token at the first disagreement and roll the rest back. Same tokens as token-by-token for greedy and seeded sampling up to prefill-vs-decode rounding; handles the retokenization boundary by construction; costs full logits rows for the forced positions; its multi-token append and roll-back are reusable by Phase 8 (recommended)
- B) SGLang-style: append the canonical forced tokens unverified (llguidance already re-tokenizes the last committed token with the forced bytes and chops tokens that could merge with what follows); cheapest, but the output can differ where the model would tokenize differently
- C) Jump only where the mask allows exactly one token: bit-identical, but multi-byte forced strings almost always allow several tokenizations, so it rarely fires

**Answer (2026-09-28, user): A** — verify the forced tokens as a draft, roll back at the first disagreement.

**2. The `cache_aware` admission key and its starvation bound.** A scheduling policy is stateless and computes a request's key once, at push.

- A) Windowed longest-prefix-first: requests are grouped by arrival window (`scheduler.cache_aware_window`, default 1 s); within a window and a priority the longest cached prefix goes first; windows keep arrival order. Bound: a request is never admitted after an equal-priority request that arrived one window or more later (recommended)
- B) Virtual deadline: key = arrival + window − credit, credit growing with the cached tokens (capped at the window); smoother, but needs a token scale (a second knob)
- C) Order by the cached fraction of the prompt instead of the cached length (closer to shortest-remaining-prefill-first)

**Answer (2026-09-28, user): A** — 1 s arrival windows, longest cached prefix first within a window.

**3. Default scheduling policy.** **Answer (2026-09-28, user): accepted as written** — `default` stays the default in this phase; the saturated multi-turn A/B (plan Task 9) is recorded and the user decides whether `cache_aware` becomes the default.

**4. Partial blocks and the KV tiers** ("a partial block is L0-only or copied whole"). **Answer (2026-09-28, user): accepted as written** — L0-only — partial entries are never demoted, prefetched or promoted, a reclaim drops them (`partial_l0_only`), and a full block that is only in L1/L2 is not used for a token-granular match. Alternative not taken: promote such a block whole, then copy (more reuse after demotion, more copy traffic). The user confirmed L0 only.

**5. Which sequences publish a partial tail.** **Answer (2026-09-28, user): accepted as written** — choice 0 of a request that finished normally (stop, EOS, length), holding the tokens whose KV is written; cancelled and failed requests publish nothing partial; at most 8 partial entries per parent, 64 children compared per lookup.

**6. Host share and the overlap rule.** **Answer (2026-09-28, user): accepted as written** — host share = all engine iteration stages except `device_wait` over all stages, from `turbine_engine_iteration_seconds{stage}` (so `launch`, which includes the blocking graph launch, counts as host); overlap work is built if either model's c1 share exceeds 0.05; if built, its default flips only when throughput rises and TTFT p50 stays ≤ 1.10 × serial for both models.

**7. Performance targets** (spec Interfaces, "Performance targets"). **Answer (2026-09-28, user): accepted as written** —

- Standard bench, every landing step: tok/s ≥ 0.98 × the Task 3 baseline and TTFT p50 ≤ 1.10 ×.
- Token-granular reuse: multi-turn `cached_tokens_ratio` +0.01 or more (reference 0.907) and later-turn TTFT p50 ≤ 0.95 × (reference 76 ms).
- `cache_aware` against `default`, saturated profile: later-turn TTFT p50 ≤ 0.85 ×, cached ratio ≥, profile tok/s ≥ 0.98 ×, first-turn TTFT p99 ≤ 1.5 ×.
- Jump-forward on the four golden JSON cases: forward steps per completion token ≤ 0.8 × and wall time ≤ 0.9 ×.
- `structured_output.jump_forward_max_tokens` default 32.

Known limit, unchanged from Phase 4 and stated in the spec: KV written by decode steps (a previous turn's generated tokens) is reused within the golden tolerance, not bit-exactly, because decode steps run Llama's speed-tuned GEMM rows; the bit-exact `kv_gpu` checks cover prefill-written prefixes.

## Phase 5p moves after Phase 6 (2026-09-28)

The user asked to move all Phase 5p items to after Phase 6. Asked whether that includes the two items carried over from the Phase 5 exit (5p plan Task 0: `lab-test.sh --release` plus the release-mode one-GPU full-tier rerun; Task 0b: the `parallel.tp_prefill_overlap` golden and bench confirmation, then turning it on):

- Keep the release-mode lab tests now, move the rest (recommended)
- Move everything
- Keep both carry-overs now

**Answer (2026-09-28, user): "skip the tests".** Tasks 0 and 0b are skipped: the stopped one-GPU full tier is not rerun, `tp_prefill_overlap` stays off by default, and `lab-test.sh` keeps the debug profile. Phase 5p (the measurement, token-granular prefix reuse, `cache_aware`, jump-forward) now runs after Phase 6 quantization; next up is `phase-6-quantization`. Tasks 0/0b stay in the 5p plan marked skipped, so they can be picked up again on request.

## YaRN RoPE scaling moves into Phase 6 (2026-09-28)

The user asked for YaRN (RoPE context extension: per-dimension frequency interpolation plus attention temperature; used by Qwen3 at 32K→128K and gpt-oss at 4K→128K) to move from the Phase 7 families track into Phase 6.

**Answer (2026-09-28, user): move it to Phase 6.** Consequences: umbrella S-6 now includes static YaRN `rope_scaling` for every registered family; dynamic scaling is refused (cached keys are stored after RoPE, so the factor must not change within a sequence); the RoPE configuration is part of the prefix-cache identity. The `phase-6-quantization` spec picks the proof. Candidates: Llama-3.2-3B-Instruct with a YaRN `rope_scaling` override, compared against transformers' native YaRN at fixture-generation time; `NousResearch/Yarn-Llama-2-7b-64k` is `LlamaForCausalLM` but needs remote code for its reference, so it is a weaker candidate. Qwen3 and gpt-oss then use it in Phase 7.

## KV-cache quantization beyond FP8 (2026-09-28)

After a survey of KV quantization methods (FP8/INT8, KIVI 2-bit, KVQuant, QJL, TurboQuant, rotation + 4-bit, MXFP4/NVFP4 KV; token-dropping methods excluded because they change outputs), the recommendation was: FP8 e4m3 KV stays the Phase 6 baseline; the first sub-8-bit KV format is TurboQuant (random rotation + optimal per-coordinate scalar quantizer + 1-bit QJL residual, calibration-free, ~3.5 bits reported quality-neutral), with KIVI-style 2-bit later if TurboQuant's kernel works out.

**Answer (2026-09-28, user): agree.** Sub-8-bit KV formats are lossy, so they're gated like lossy weights (eval accuracy within `quality.max_accuracy_drop`, not only golden). Each is a registered `kv_format` entry, following the reuse-first rule: check CK, llama.cpp's HIP flash-attention with q8_0/q4_0 KV, and vLLM's FP8 KV before writing our own attention kernel. The phase-6 spec decides whether TurboQuant sits at the end of Phase 6 or in a small phase right after it.

## Sub-8-bit KV and per-tier KV formats in Phase 6 (2026-09-28)

After the per-tier discussion (different KV formats per tier: L0 BF16/FP8 for attention speed, L1 ~4-bit for capacity and the ~12 GB/s host link, L2 2–4-bit plus lossless compression; quantize on the GPU during demotion; keep the recent window precise; retrieve vs recompute weighs quality):

**Answer (2026-09-28, user): add it to Phase 6.** This supersedes the "end of Phase 6 or a small phase after" placement in the previous entry. TurboQuant and per-tier formats are both in `phase-6-quantization` scope (umbrella S-6 amended). Open for the phase-6 spec, with recommendations:

- promotion: unpack to the L0 format first; mixed-format attention later;
- K and V bit widths per format;
- per-layer precision;
- a per-request opt-out of lossy reuse (recommended: yes);
- kernel providers under the reuse-first rule.

## Pressure-driven KV compression ladder in Phase 6 (2026-09-28)

User proposal: instead of fixed per-tier formats, start lossless everywhere. When all tiers fill, lower the precision of the lowest tier first, so only the oldest data loses precision. Under more pressure, lower the next faster tier, and so on until all tiers are at the same level, then repeat with the next, stronger quantization.

Refined in discussion: a per-block ladder (lossless → FP8 → ~4-bit → ~2-bit → evict) driven by the pressure controller. "Compress" becomes a third eviction-policy action next to demote and evict, applied to the blocks least likely to be reused, which in practice means oldest-first from the lowest tier. It works mostly on new demotions, rewrites existing blocks only oldest-first and bounded per step, uses hysteresis, and never upgrades a block that has lost precision. L1/L2 come first; L0 joins after mixed-format attention.

**Answer (2026-09-28, user): do it.** Added to `phase-6-quantization` scope as its last step (umbrella S-6 amended), after the weight formats, FP8 KV, TurboQuant and the per-tier formats. Observability rules: reason codes, per-tier × format metrics, lossy-token counts per response, an opt-out that recomputes instead, and tests that pin the pressure state.

## Phase 6 spec: provisional design choices (2026-09-28)

Written with `.procoder/specs/phase-6-quantization.md` and its plan. The scope, the formats, the MXFP4 packagings and proof checkpoints, YaRN, TurboQuant, the per-tier formats and the compression ladder are fixed by the entries "Roadmap reorganisation after Phase 5", "Phase 6 MXFP4: packaging formats and proof checkpoints", "YaRN RoPE scaling moves into Phase 6", "KV-cache quantization beyond FP8", "Sub-8-bit KV and per-tier KV formats in Phase 6" and "Pressure-driven KV compression ladder in Phase 6" (all 2026-09-28). The choices below are the ones those entries leave open. The spec first wrote each recommended option marked "(provisional)"; the user answered all twenty on 2026-09-28 (below; Q11 differs from the recommendation). Facts behind them, gathered 2026-09-28: hipBLASLt on `novanas` (ROCm 7.14.1, `gfx1201`) ships FP8 × FP8 → BF16/F32 Tensile kernels with scalar (`SAB`) and per-row/column vector (`SABV`) scales, but no BF16 × FP8 mixed kernel and no block-scaled or microscaled variant; the pinned CK (`therock-7.14.1`) has `ck_tile` `gemm_quant` (`TensorQuant`, `RowColQuant`, `AQuantGrouped`, `BQuantGrouped`, `ABQuantGrouped`, a microscale pipeline, some with WMMA policies) and FP8 hooks in the paged / split-KV FMHA kernels; whether any of them builds and is correct on `gfx1201` is what the reuse evaluations of the plan establish.

**1. Order of the sub-steps.** The kickoff listed weights → FP8 KV → TurboQuant → per-tier formats → YaRN → ladder.

- A) As listed
- B) Foundations → weights (fp8, fp8_block, INT4, MXFP4) → FP8 KV in L0 → YaRN → per-tier formats (first with `fp8_e4m3` as the lower-tier format) → TurboQuant (a registered codec plugged into the per-tier path) → ladder. Reason: TurboQuant in Phase 6 is a lower-tier storage format (question 11), so it needs the per-tier transcoding path first; proving that path with FP8 (a cheap, well-understood codec) separates plumbing bugs from codec bugs. YaRN is small and independent and changes the prefix namespace, so it lands before the per-tier identity rework (recommended)
- C) YaRN first (smallest, and Phase 7 depends on it), then B's order

**2. FP8 weight arithmetic (`fp8`).** No BF16 × FP8 kernel exists in hipBLASLt on `gfx1201`; FP8 × FP8 does.

- A) W8A8 following the checkpoint: activations quantized to FP8 e4m3 per the checkpoint's `input_activations` (static per-tensor scale, or dynamic per-token), then hipBLASLt FP8 × FP8 → BF16 with the weight's per-tensor or per-channel scale; a checkpoint without an activation scheme runs W8A16 through a dequantize path. Same arithmetic the checkpoint was calibrated for, and what vLLM runs (recommended)
- B) W8A16 always: dequantize FP8 weights to BF16 inside the GEMM (needs a dequant kernel — CK or own); decode gains the bandwidth, prefill runs at BF16 speed
- C) Both, selectable with a configuration key

**3. `fp8_block` kernel fallback.** Block-scaled FP8 (128 × 128 weight blocks, per-token groups of 128 activations) has no hipBLASLt kernel; the plan evaluates CK `ABQuantGrouped` first.

- A) If CK does not work on `gfx1201`: W8A16 for `fp8_block` through a dequantize-to-BF16 path (CK, llama.cpp-style, or own — each own kernel recorded by the reuse rule) (recommended)
- B) If CK does not work: `fp8_block` stays `experimental` and the phase continues
- C) Write an own block-scaled FP8 WMMA kernel

**4. Proof checkpoints for `fp8`, `fp8_block`, `awq_int4`, `gptq_int4`** (all `LlamaForCausalLM` Llama-3.2-3B-Instruct, ungated, revisions read 2026-09-28; no quantized OLMoE checkpoint exists).

- A) `fp8`: `RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic` @ `c308a86de78778c5f904a1d82401ac85e18ca205` (compressed-tensors, per-channel weights, dynamic per-token activations, 4.4 GB) as the gate, plus `RedHatAI/Llama-3.2-3B-Instruct-FP8` @ `377571d314b30f1d58448499e4100e2deafe7d7d` (per-tensor weights, static per-tensor activations) for the per-tensor path; `fp8_block`: `unsloth/Llama-3.2-3B-Instruct-FP8-Block` @ `08cf804398b23fab4a1df02fbe8d4d5a11a800cc` (compressed-tensors 128 × 128 blocks, 3.6 GB); `awq_int4`: `casperhansen/llama-3.2-3b-instruct-awq` @ `272b3bde867b606760447deb9a4d2719fbdfd3ae` (AutoAWQ GEMM, zero points, group 128, 2.3 GB); `gptq_int4`: `shuyuej/Llama-3.2-3B-Instruct-GPTQ` @ `dd5a311f040728fbc612eb03c8dadfae0a90552f` (AutoGPTQ, symmetric, `desc_act: false`, group 128, Apache-2.0, 2.3 GB) (recommended)
- B) As A, `fp8` with the dynamic checkpoint only
- C) As A, `awq_int4` from `AMead10/Llama-3.2-3B-Instruct-AWQ` @ `df494d4903f031dadaeb40434529f1c01efcd130` (3.1 GB) instead

The MXFP4 proofs are fixed: `FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16` @ `14c3aca849a72df8fcc8b3a30ab8d9eed86ee646` (5.8 GB) and `matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq` @ `91925ffda6977d097354a99718a20e035f8af80a` (2.3 GB); the BF16 quality baseline for the 8B proof is `unsloth/Llama-3.1-8B-Instruct` @ `4699cc75b550f9c6f3173fb80f4703b62d946aa5` (ungated mirror of `meta-llama/Llama-3.1-8B-Instruct`, 16.1 GB) unless the user prefers the gated original.

**5. Checkpoint containers per `weight_format`.**

- A) `fp8` ← compressed-tensors `float-quantized` 8-bit with weight strategy `tensor` or `channel`, and HF/Quark `quant_method: fp8` without `weight_block_size`; `fp8_block` ← compressed-tensors weight strategy `block` [128, 128], and `quant_method: fp8` with `weight_block_size` [128, 128]; `awq_int4` ← `quant_method: awq`, `version: gemm`, 4 bits, with zero points; `gptq_int4` ← `quant_method: gptq` 4 bits with `desc_act: false`, and compressed-tensors `pack-quantized` 4-bit int group strategy (W4A16) without a non-trivial `g_idx`; MXFP4 as question 6. Act-order (`desc_act: true`, or a non-identity `g_idx`) is refused with reason `gptq_act_order` (recommended)
- B) As A, plus act-order GPTQ (permute the weight's input dimension at load and the activation columns at run time)
- C) As A, without compressed-tensors `pack-quantized` (AutoGPTQ containers only)

**6. Support-matrix values for the three MXFP4 packagings.** The Quark proof is W4A4 (activation FP4 emulated) and may end `experimental` while compressed-tensors W4A16 ends `supported`, on the same architecture; one column value cannot hold both.

- A) Two values: `mxfp4` (weight-only W4A16: compressed-tensors, OpenAI native, and Quark weight-only checkpoints) and `mxfp4_a4` (W4A4 checkpoints, activations quantize-dequantized to MXFP4 before a BF16 GEMM); this widens the umbrella's bounded `weight_format` set by one value, which amends `phase-6-8-expansion` Interfaces (recommended)
- B) One `mxfp4` value; the Quark W4A4 checkpoint is refused by a separate refusal list (like `PARALLEL_REFUSALS`) until it passes
- C) One value per packaging: `mxfp4_ct`, `mxfp4_openai`, `mxfp4_quark`

**7. Quantized MoE experts and tensor parallelism.** No quantized OLMoE checkpoint exists; Phase 7 needs FP8 experts (Qwen3-MoE, Mixtral with TP 2) and MXFP4 experts (gpt-oss).

- A) Phase 6 quantizes dense linear layers (attention and dense MLP projections) at tp 1 and under Phase 5 TP (scales sharded with their rows or columns; a block or group boundary that does not divide the shard is refused with `quant_shard_misaligned`), proven at tp 2 by the lab golden for `fp8` and `awq_int4`; quantized MoE expert GEMMs move to Phase 7 with the families that need them (recommended)
- B) As A, plus quantized MoE experts proven on an OLMoE FP8 checkpoint quantized offline at fixture time (llm-compressor FP8-dynamic; test data, not a served format)
- C) Phase 6 at tp 1 only; everything under TP moves to Phase 7

**8. Golden references for quantized checkpoints.** The umbrella asks for the checkpoint's own reference output; transformers' AWQ/GPTQ/Quark integrations need extra packages that are mostly GPU-only.

- A) A fixture script `scripts/golden/quant_reference.py` decodes the checkpoint's weights exactly (FP8 × scale, INT4 (q − zero) × scale, FP4 × 2^E8M0) into a temporary BF16 checkpoint and runs the existing transformers reference on it, with activation fake-quantization hooks for W8A8 (FP8 e4m3, the checkpoint's scheme) and W4A4 (MXFP4); the tolerance per slug is calibrated from transformers' own spread (`self_spread.py`, the OLMoE method); where the checkpoint's own integration runs on CPU (compressed-tensors FP8), one cross-check is recorded (recommended)
- B) Capture the reference from vLLM-ROCm on the same checkpoint where it loads on `gfx1201`, transformers-dequant elsewhere
- C) Each checkpoint's own transformers integration only; formats whose integration does not run are `experimental`

**9. Accuracy gate for lossy formats.** The umbrella gate is GSM8K-200 accuracy within `quality.max_accuracy_drop` (default 0.01) of BF16. With 200 items one answer is 0.005, the paired noise is about ±0.02, and published 4-bit Llama-3.2-3B results lose 1–3 points on GSM8K, so a literal 0.01 against BF16 would likely fail every 4-bit weight format without a defect.

- A) Weight formats: compared with the reference engine on the same checkpoint (vLLM-ROCm, where it loads the checkpoint on `gfx1201`) at 0.01; where it does not, compared with BF16 at a per-format drop recorded in `tests/eval/<slug>/gate.json` (provisional: FP8 0.02, INT4 and MXFP4 0.04), confirmed by the user at the track close. KV formats (FP8 KV, TurboQuant, the ladder): compared with the same weights at BF16 KV at 0.01 (recommended)
- B) The umbrella literally: BF16 at 0.01 for everything; a format that fails stays `experimental`
- C) As A, but the gate runs on the full GSM8K test split (1,319 items) to cut the noise

**10. FP8 KV scales and attention.**

- A) Per-layer K and V scales from the checkpoint (`kv_cache_scheme`, `k_scale` / `v_scale` tensors) when present, else 1.0 (vLLM's default); `kv.dtype: fp8_e4m3` must be set explicitly (nothing lossy by default); attention reads FP8 pages through the provider the evaluation picks (CK FMHA FP8 instances first, then an FP8-reading variant of the Turbine paged kernel) (recommended)
- B) As A, but calibrate missing scales at warm-up on a fixed prompt set (deterministic, logged)
- C) Per-block dynamic scales (`KvDtype::Fp8E4m3PerBlockScale`), computed at page write

**11. Where TurboQuant runs in Phase 6.**

- A) As a lower-tier storage format only (L1/L2, and ladder rungs there): encoded on the GPU at demotion, decoded to the L0 format on promotion; attention never reads it. L0 formats stay `bf16` / `fp8_e4m3`; TurboQuant attention and mixed-format attention are deferred to a later decision (recommended)
- B) Also as an L0 format, decoded into a BF16 scratch before attention
- C) Also as an L0 format with a native TurboQuant attention kernel

**12. TurboQuant formats and bit widths.**

- A) Two registered formats: `tq4` (K: 3-bit Lloyd–Max codebook for the rotated coordinates + 1-bit QJL residual sign, the paper's inner-product variant; V: 4-bit MSE codebook; one BF16 norm per token-head vector for K, V and the K residual; ≈ 4.4 bits per element) and `tq2` (K: 1-bit codebook + 1-bit QJL; V: 2-bit codebook; ≈ 2.4 bits); the rotation is a randomized Hadamard transform (random signs from a seed fixed per namespace, then the fast Walsh–Hadamard transform over `head_dim` = 128); the same widths for every layer (recommended)
- B) As A, but keep the first and last two layers at FP8 in `tq2`
- C) Three formats: `tq4`, `tq3` (3.5 bits, the paper's quality-neutral point) and `tq2`

**13. Per-tier configuration and the recent window.**

- A) `kv.cpu.format` and `kv.nvme.format` ∈ {`l0` (the L0 format, default), `fp8_e4m3`, `tq4`, `tq2`}; a tier may not hold a format more precise than the tier above it; the last `kv.lossless_tail_blocks` full blocks of a sequence (default 1) are demoted at the L0 format whatever the tier's format; a partial last block never leaves L0 (recommended)
- B) As A, with the window in tokens (`kv.lossless_tail_tokens`, default 256)
- C) No window: every demoted block takes the tier's format

**14. Prefix identity of lossy blocks.** A block that passed through a lossy tier no longer holds the exact KV, and a block prefilled on top of it inherits the difference.

- A) Lineage keys: a block keeps its exact key while any copy of it is exact; a lossy copy is filed under `lossy_key(key, format)`; a block computed on top of a lossy prefix is chained from the lossy parent key, so exact and lossy lineages never alias; a lookup takes the exact chain first and, unless the request opts out, continues on lossy keys; the format and the TurboQuant seed enter the key (recommended)
- B) One key per block; locations carry a format tag; blocks computed on top of a lossy prefix are never published (no aliasing, less reuse)
- C) One key per block with a format tag and no lineage (a child computed over lossy KV may be served to an exact request)

**15. Per-request opt-out and reporting.**

- A) Header `x-turbine-kv-lossy: deny` and config default `kv.lossy_reuse: allow|deny` (default `allow`); an opted-out request matches exact lineage only and recomputes the rest; every response reports `usage.prompt_tokens_details.lossy_cached_tokens` (Turbine extension, 0 when none) (recommended)
- B) A body field instead of the header
- C) Default `deny`: lossy reuse only for requests that ask for it

**16. Lossy retrieve against recompute in the planner.**

- A) A lossy block's retrieval cost is multiplied by `1 + penalty(format)` (provisional `fp8_e4m3` 0.1, `tq4` 0.5, `tq2` 1.0, config `kv.lossy_penalty`), so a lossy block is used only when clearly cheaper than recompute; every plan logs the penalty (recommended)
- B) No penalty: transfer cost only
- C) Lossy blocks used only at pressure ORANGE or worse

**17. The ladder's trigger and bounds.**

- A) Compress replaces a drop: when the policy would drop (not demote) a block from the lowest enabled tier, or that tier is above its high-water mark (0.95 used), the oldest, least-reusable blocks of that tier move one rung down (`l0` → `fp8_e4m3` → `tq4` → `tq2` → evict), then the next tier up once the lowest tier sits on one rung; new demotions into a tier take its current rung; at most 32 rewrites per reclaim tick, ticks ≥ 50 ms apart (the Phase 4 bounds); a tier's rung for new demotions steps back up only after it has stayed below 0.85 used for `reliability.pressure.deescalate_dwell`; compressed blocks are never upgraded; no compression while the pressure controller is GREEN (recommended)
- B) Driven by the global pressure state only: YELLOW → new demotions at `fp8_e4m3`, ORANGE → `tq4` plus rewrites, RED → `tq2`
- C) As A, and L0 joins now (needs mixed BF16/FP8 attention)

**18. The "lossless" rung.**

- A) The L0 format's bytes unchanged; no general-purpose compression in Phase 6 (recommended)
- B) LZ4 on L2 slots at the lossless rung

**19. YaRN proof and override.**

- A) A `model.rope_scaling` configuration key (a mapping with HF's field names; replaces `config.json`'s `rope_scaling` and enters the model identity) serves Llama-3.2-3B-Instruct with `{rope_type: yarn, factor: 16.0, original_max_position_embeddings: 8192, beta_fast: 32, beta_slow: 1}`; the reference is transformers' native YaRN on a copy of the checkpoint with that `config.json`, over the 16 golden prompts plus one ≈ 12,000-token prompt; the attention factor (`mscale`) folds into the attention scale (`scale × mscale²`), so the RoPE ABI does not change; `truncate: false` (gpt-oss) is parsed and unit-tested (recommended)
- B) As A with a Qwen3-style override (`factor: 4.0`, `original_max_position_embeddings: 32768`)
- C) As A, but mscale scales the cos/sin tables through a new RoPE descriptor field (a minor ABI group), as transformers does

**20. Performance targets** (novanas GPU 0, `lab-bench.sh`, Llama-3.2-3B, against the BF16 baseline of 855 tok/s at c16).

- A) `fp8`: c16 tok/s ≥ 1.10 × BF16 and c1 ITL p50 ≤ 0.75 × BF16; `fp8_block`: c16 ≥ 1.0 ×; `awq_int4` / `gptq_int4` / `mxfp4` (on 3B-sized checkpoints): c1 ITL p50 ≤ 0.6 × BF16 and c16 tok/s ≥ 0.9 × BF16; where vLLM-ROCm serves the same checkpoint, Turbine c16 tok/s ≥ 0.9 × vLLM; FP8 KV: pool blocks ≥ 1.95 × BF16 KV and c16 tok/s ≥ 0.95 × BF16 KV; `tq4` in L1: L1 blocks per GiB ≥ 3.5 × `l0`, and multi-turn `mt_cached` not below the `l0` run; the ladder: `scripts/overload-soak.sh novanas --duration 10m` passes with a small L1/L2 (recommended)
- B) No hard targets; record the numbers and decide at the track close

**Answers (2026-09-28, user, relayed by the coordinator):**

- **Q1: B** — foundations → weights → FP8 KV → YaRN → per-tier formats (proven with FP8) → TurboQuant → ladder.
- **Q2: A** — W8A8 FP8 on hipBLASLt following the checkpoint's activation scheme.
- **Q3: A** — block-scaled FP8 falls back to a dequantize-to-BF16 path if CK does not work on `gfx1201`.
- **Q4: A** — the proof checkpoints as listed, with `unsloth/Llama-3.1-8B-Instruct` as the 8B BF16 baseline. The ≈ 41 GB of downloads are approved; keep ≥ 60 GB free on `novanas`.
- **Q5: A** — container mapping as listed; act-order GPTQ refused (`gptq_act_order`).
- **Q6: A** — a separate `mxfp4_a4` column value. **Signed off as an amendment of the umbrella** `phase-6-8-expansion` (bounded `weight_format` set).
- **Q7: A** — dense linear layers only, including under TP; quantized MoE experts move to Phase 7 (added to the umbrella's S-8 list).
- **Q8: A** — exact-BF16 dequantized references with activation fake-quantization, tolerances calibrated from transformers' spread.
- **Q9: A** — weights: vLLM-ROCm on the same checkpoint where it runs (0.01), otherwise BF16 with max drop 0.02 for FP8 and 0.04 for 4-bit formats; KV formats: 0.01 against BF16 KV. **Signed off as an amendment of the umbrella** S-3 item 4.
- **Q10: A** — FP8 KV scales from the checkpoint, else 1.0; explicit setting only.
- **Q11: changed from the recommendation — TurboQuant also lives in L0 in Phase 6.** Build a TurboQuant-aware paged attention with a per-block format tag: attention reads BF16/FP8 blocks and `tq4`/`tq2` blocks, rotates q once per step and dots against the compressed K, and decodes V. Reuse first: evaluate existing quantized-KV attention (llama.cpp HIP flash attention with q4/q8 KV, CK FP8 KV, vLLM/aiter) before writing our own, and record it. Gates: a bitwise or tolerance test against a CPU reference TurboQuant attention; golden and eval-compare with `tq4`/`tq2` KV in L0; the decode ITL impact measured and reported. Consequence: mixed-format attention exists in Phase 6, so the compression ladder includes L0 once that attention has passed its correctness and performance gates — L1/L2 first, L0 as the ladder's final sub-step.
- **Q12: A** — `tq4` and `tq2` with the randomized Hadamard rotation.
- **Q13: A** — `kv.cpu.format` / `kv.nvme.format`, the last full block kept at the L0 format.
- **Q14: A** — lineage keys for lossy blocks and their descendants.
- **Q15: A** — header `x-turbine-kv-lossy: deny` plus `usage.prompt_tokens_details.lossy_cached_tokens`.
- **Q16: A** — per-format lossy retrieval penalty.
- **Q17: A** — ladder driven by tier fill, never at GREEN, ≤ 32 rewrites per tick with hysteresis (L0 joins as the final sub-step, per Q11).
- **Q18: A** — the lossless rung stored plain.
- **Q19: A** — `model.rope_scaling`, factor 16 proof plus a ≈ 12,000-token prompt, attention factor folded into the scale.
- **Q20: A** — the performance targets as listed; for `tq4`/`tq2` in L0 the decode ITL impact is measured and reported (no fixed bound).

The spec and plan now mark these "(user decision 2026-09-28)".

## Phase 6 split: 6a quantization, 6b KV compression (2026-09-28)

After the answers to "Phase 6 spec: provisional design choices" (with Q11 widening TurboQuant into L0), the user split Phase 6 in two. Options shown by the coordinator:

**1. Where does YaRN go?**

- A) In 6a, with the weight formats and FP8 KV
- B) In 6b, with the KV compression work

**Answer (2026-09-28, user): A — YaRN in 6a.**

**2. When does 6b run?**

- A) Right after 6a
- B) After Phase 5p
- C) After Phase 7

**Answer (2026-09-28, user): A — right after 6a.**

The split, as decided:

- `phase-6a-quantization`: foundations (the quality-gate tooling port from `runahead/p8-umbrella` and the support-matrix cleanup), every weight format (`fp8`, `fp8_block`, `mxfp4` in three packagings, `mxfp4_a4`, `awq_int4`, `gptq_int4`), FP8 KV in L0, YaRN.
- `phase-6b-kv-compression`: per-tier KV formats (proven with FP8 first), TurboQuant `tq4` / `tq2` including the TurboQuant-aware L0 attention, the pressure-driven compression ladder (L1/L2, then L0). It starts only after 6a closes and gets its own full test and gate cycle.
- Order: 6a → 6b → 5p → 7 → 8.

Consequences (docs changed the same day): the joint `phase-6-quantization` spec and plan are replaced by `.procoder/specs/phase-6a-quantization.md` / `.procoder/plans/phase-6a-quantization.md` and `.procoder/specs/phase-6b-kv-compression.md` / `.procoder/plans/phase-6b-kv-compression.md`; the answers of "Phase 6 spec: provisional design choices" carry over unchanged. 6a keeps the joint spec's item numbers S-1 … S-16 (its gate items S-22 … S-25 become S-17 … S-20); 6b renumbers its items S-1 … S-11 (joint S-17 … S-21, S-26, S-27, gates S-22 … S-25). The work branch `phase-6-quantization` was renamed `phase-6a-quantization` before any code landed. The umbrella `phase-6-8-expansion` S-1 lists 6a and 6b as the two phases of track 1; AGENTS.md's order line follows. `scripts/track-gate.sh` (ported in 6a Task 1) knows both track names.

## Continue through Phases 6a and 6b unattended (2026-09-29)

**Decision (user, 2026-09-29, relayed by the coordinator):** "continue through all the phases a and b" — the user has gone to bed; 6a runs to its close, then 6b from the merged main to its close. Rules (the same as the 2026-09-27 overnight run, "Continue through the phases unattended"):

1. This entry records the decision.
2. Design decisions: the user is not available, so questions are not waited on — the recommended option is taken, marked here "provisional, pending user review", and listed one line each (options and pick) in `.procoder/review-2026-09-29.md`.
3. Lab runs: the instruction covers every 6a/6b lab run on novanas (`lab-test.sh`, `lab-bench.sh`, `lab-serve.sh`, `lab-cluster.sh`, the vLLM baselines of the new formats, the 10-minute soak) under the usual rules: novanas only, nothing on the Sparks and nothing NVIDIA; perf numbers on GPU 0 only; a busy GPU or bench lock is identified first, our own hung runs stopped, nothing that is not ours evicted; serve Jobs always `--stop`ped. The HF token stays on novanas (`hf download --revision <pinned>`, never read or passed); only the models the 6a/6b specs name are downloaded, after a free-disk check.
4. Perf: one change, then measure (`lab-bench.sh --quick` plus golden); results in labbook, with vLLM-ROCm as the baseline of each new format.
5. Phase close: 6a closes when every exit gate passes (tier full, the two-GPU leg, golden16, soak) and each format's accuracy gate holds; it merges into local main, not pushed. 6b starts from that main, closes the same way and merges locally; pushing waits for the user's review. A failing gate that needs a real design change takes the provisional option; a truly blocked item (host down, model unavailable) is left unfinished, written in the review file, and the rest continues.
6. novanas crashes (flaky PSU until the new one arrives): wait with a bounded check (every 10 min, up to 2 h), then continue; do not debug the crash.
7. `.procoder/review-2026-09-29.md` is kept current: what landed, perf against vLLM, provisional decisions, anything unfinished — the file the user reads in the morning.

Also relayed the same night (coordinator, at the user's request "more speed"): build in parallel. The Phase 6a lead stays the integrator; builder agents run in their own worktrees on branches cut from `phase-6a-quantization`, one owner per file (a shared registry or config file is owned by one track and the other sends its edit to the owner), each task test-first, one commit, `scripts/gate.sh` clean, merged or rebased back by the lead after it passes the gate; GPU work staggered (`lab-test.sh --tier quick` on either card, every perf number on GPU 0 under the bench lock). Branch `scout-fixes` is merged into main by the coordinator; main may move under the phase branches.

## Phase 6a proof checkpoints downloaded (2026-09-29)

Downloaded on `novanas` by the fixtures builder with `hf download <repo> --revision <sha> --local-dir /home/piwi/turbine-models/<slug>` (approved: user answer Q4 and "Continue through Phases 6a and 6b unattended"), every one rc 0, `config.json` and the listed safetensors present:

| Slug                                | Repo @ revision                                                                                       | Bytes         |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------- |
| `llama-3.2-3b-instruct-fp8-dynamic` | `RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic` @ `c308a86de78778c5f904a1d82401ac85e18ca205`             | 4,413,814,259 |
| `llama-3.2-3b-instruct-fp8`         | `RedHatAI/Llama-3.2-3B-Instruct-FP8` @ `377571d314b30f1d58448499e4100e2deafe7d7d`                     | 4,404,163,249 |
| `llama-3.2-3b-instruct-fp8-block`   | `unsloth/Llama-3.2-3B-Instruct-FP8-Block` @ `08cf804398b23fab4a1df02fbe8d4d5a11a800cc`                | 3,624,601,065 |
| `llama-3.2-3b-instruct-awq`         | `casperhansen/llama-3.2-3b-instruct-awq` @ `272b3bde867b606760447deb9a4d2719fbdfd3ae`                 | 2,270,034,219 |
| `llama-3.2-3b-instruct-gptq`        | `shuyuej/Llama-3.2-3B-Instruct-GPTQ` @ `dd5a311f040728fbc612eb03c8dadfae0a90552f`                     | 2,264,970,345 |
| `llama-3.2-3b-mxfp4-a4`             | `matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq` @ `91925ffda6977d097354a99718a20e035f8af80a`             | 2,303,041,140 |
| `llama-3.1-8b-instruct-mxfp4a16`    | `FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16` @ `14c3aca849a72df8fcc8b3a30ab8d9eed86ee646` | 5,827,024,421 |

Free disk went from 99.0 GB before to 72.3 GB after the last download (00:17). The 8B BF16 baseline (`unsloth/Llama-3.1-8B-Instruct` @ `4699cc75b550f9c6f3173fb80f4703b62d946aa5`, 16 GB) is on hold: novanas fell to 49 GB free at 00:25 (other workspaces' build trees) and the kubelet evicts lab pods below ~47.7 GB; it is downloaded once stale build trees are cleared.

## Phase 6a/6b builder decisions (2026-09-29; answered by the user 2026-09-29)

Taken under "Continue through Phases 6a and 6b unattended (2026-09-29)" (rule 2: recommended option, user decision 2026-09-29: accepted as recommended); each is also a line of `.procoder/review-2026-09-29.md`.

**YaRN (6a Task 26, landed 42ee226):**

1. Tolerance of `yarn_parameters_match_transformers`: transformers computes the YaRN table in FP32 with a `powf` 1 ulp off, so the spec's 1e-7 relative is below FP32 resolution. A) FP64 as specified, each value within 2 FP32 ulps (max seen 1.46e-7) — chosen; B) mimic transformers' FP32 operation order (252/256 bitwise, the rest 1 ulp, max 1.12e-7). Spec S-15 criterion amended to "within 2 FP32 ulps". **User decision 2026-09-29: accepted as recommended — A.**
2. Context extension: YaRN extends `max_positions` only when `original_max_position_embeddings` < `max_position_embeddings` (the spec's edge case literally; a model with both equal, e.g. Qwen2.5 at 32768, is not extended). Alternative: always allow up to `max(factor × original, max_position_embeddings)`. **User decision 2026-09-29: accepted as recommended — as the spec.**
3. Refusals where transformers only warns: factor < 1, attention factor ≤ 0, `beta_slow` > `beta_fast` or ≤ 0. **User decision 2026-09-29: accepted as recommended.**
4. Defaults follow Python's `or` (0 means default for the original length and the betas); unknown keys ignored; `dynamic: false` accepted. **User decision 2026-09-29: accepted as recommended.**
5. `rope_identity()` includes `rotary_dim` next to theta and the scaling fields. **User decision 2026-09-29: accepted as recommended.**

**FP8 KV (6a Task 22, landed 877b0ab):** a paged `AttentionConfig.dtype` of F8E4M3 means FP8 pages while Q, the new rows and the output stay BF16 (the C descriptor's `dtype` = 16 means the same); the CPU row `cpu/*/*/bf16/fp8_e4m3/none` is `experimental`, `amd` stays `unsupported` until Task 24's gate. **User decision 2026-09-29: accepted as recommended.**

**6b groundwork (branch `p6b-groundwork`: 0a0b20a, 3d07054, 66dcf2d; lands with 6b):**

1. TurboQuant QJL projection: the plan's second randomized Hadamard (`H·(s'⊙r)`) failed the unbiasedness test (overestimates ⟨q, r⟩ by ~2.7 %: tq4 mean error 6.5e-4 > 3 SE 3.7e-4). A) the paper's Gaussian projection S (128 × 128, seeded per layer and head) — chosen: unbiased (tq4 5.4e-5 vs limit 6.0e-4), but O(d²) encode/decode and 64 KB of S per (layer, head) passed to the GPU codec (~15 MB Llama, ~17 MB OLMoE); B) one S per namespace (64 KB); C) keep Hadamard and loosen unbiasedness. **User decision 2026-09-29: accepted as recommended — A** (final); 6b plan Task 7 / spec S-4 updated when 6b starts.
2. `turbine-kv` keeps its own bit-exact copy of the e4m3fn rounding (tested against the kernels' table) instead of moving it to `turbine-core`. **User decision 2026-09-29: accepted as recommended.**
3. `KvCodec`: takes `&KvLayout`; `CodecParams` carries per-layer `k_scales` / `v_scales`; `lossy(layout)` (FP8 is lossless from an FP8 L0); extra `abi_code`, `nmse_bound`, `supports`; registration order is the ladder order. **User decision 2026-09-29: accepted as recommended.**
4. FP8 codec from BF16 L0: one K and one V scale per block per layer, `max(absmax / 448, 1 / (448·512))`, in a slot header (alternative: per token or per head). **User decision 2026-09-29: accepted as recommended.**
5. TurboQuant record layout: array of structs per token-head in the spec's field order, padded to 16 bytes (tq4 144 B = 3.56× BF16, tq2 80 B = 6.4×; unpadded would be 3.82× / 7.3×); codes LSB-first. Alternatives: struct of arrays per (layer, head), or no padding. **User decision 2026-09-29: accepted as recommended.**
6. Rounding: codes from the exact F32 norm, decode and residual from the BF16-rounded norm (QJL corrects the norm rounding); ties between centroids take the lower code; a zero QJL projection signs +1. **User decision 2026-09-29: accepted as recommended.**
7. Seeds: SplitMix64 from the namespace seed mixed with (layer, head, kind: K 0, V 1, QJL 2); Box–Muller in F64 rounded to F32; block-local head index (global head index under TP is a 6b Task 8/15 decision). **User decision 2026-09-29: accepted as recommended.**
8. Codebooks: trapezoid integration on 2^18 points, Lloyd iteration to 1e-13, committed as symmetrised F32 constants regenerated within 1e-6 by the test (distortion 0.3609 / 0.1160 / 0.0340 / 0.00931 for 1–4 bits vs the paper's 0.36 / 0.117 / 0.03 / 0.009). **User decision 2026-09-29: accepted as recommended.**
9. Conformance NMSE bounds: tq4 0.07, tq2 0.75 over a block (K error ≈ (π/2)·D_mse from the Gaussian QJL). **User decision 2026-09-29: accepted as recommended.**
10. Ladder policy: no action at GREEN; a tier acts only when it is the lowest and about to drop, or above high water; one rung down from the copy's own format, capped at `max_format`; an upper tier acts only once every lower tier reached the target rung; at the floor the lowest tier evicts and an upper tier keeps or demotes; `LadderContext` gains `format`, `must_leave`, `demote_to`, `lower_rung`, `ladder` (the spec's `PressureLevel` is `PressureState`). **User decision 2026-09-29: changed — "Start at YELLOW earlier": compression begins at YELLOW pressure, before the tiers are full, still the lowest tier first and one rung at a time.**
11. `EvictReason::{Compressed, LadderFloor}` exist but join the pre-registered metric label sets only with 6b Task 15 (so `api.rs check_labels` stays green). **User decision 2026-09-29: accepted as recommended.**

**FP8 packagings (6a Task 8, lead):**

1. `quant_method: fp8` without `weight_block_size` and `activation_scheme: dynamic`: activations quantized per token (vLLM quantizes them per tensor, dynamically, a mode the v2.9 ABI does not have). A) per token — chosen (at least as accurate, one fewer kernel mode); B) add a dynamic per-tensor mode to the ABI to match vLLM exactly. **User decision 2026-09-29: accepted as recommended — A.**
2. A layer's quantization is decided from `config.json` (the packaging's ignore list, plus `lm_head` always BF16), not from each tensor's dtype; a checkpoint that leaves a layer in BF16 without listing it is refused by the tensor check. Alternative: decide per tensor from the checkpoint dtype (needs the index before the memory budget). **User decision 2026-09-29: accepted as recommended.**
3. Weight formats are configured per checkpoint: `WeightFormat::configure(config.json)` returns an `Arc<dyn WeightFormat>` and `ModelArchConfig::weight_format` holds it (registry entries stay `&'static`, their default layout feeding the conformance fixtures). Alternative: pass the parsed `quantization_config` to every trait method. **User decision 2026-09-29: accepted as recommended.**
4. A decoder with some BF16 and some quantized linear layers requires both the GEMM and `qgemm` configs for each linear shape (the registry resolves a BF16 GEMM that a fully quantized model never calls). Alternative: per-layer requirements. **User decision 2026-09-29: accepted as recommended.**
5. Tiny fixtures are written by each format (`WeightFormat::write_tiny`, test support in the format's file, as families' `write_tiny`), not by one writer in `testing/tiny.rs`; power-of-two scales so the dequantized twin is exact in BF16; static input scales differ per projection so the fused-stack max rule is tested. **User decision 2026-09-29: accepted as recommended.**

## Phase 6a proof checkpoints downloaded on novanas (2026-09-29)

Task 11, fixtures builder, under the approved downloads (user decision 2026-09-28, ≈ 41 GB, ≥ 60 GB free kept; floor raised to 70 GB by the coordinator on 2026-09-29). `hf download <repo> --revision <sha> --local-dir /home/piwi/turbine-models/<slug>`; the token stayed on the host. Every run exited 0; each directory holds `config.json` and every safetensors file the Hub lists. Free disk in bytes (`df -B1 /home/piwi`); two lanes ran at once, so an "after" can include the other lane's progress.

| slug                              | repo                                                   | revision                                 | bytes (du)     | free before     | free after      |
| --------------------------------- | ------------------------------------------------------ | ---------------------------------------- | -------------- | --------------- | --------------- |
| llama-3.2-3b-instruct-fp8-dynamic | RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic             | c308a86de78778c5f904a1d82401ac85e18ca205 | 4,413,814,259  | 99,008,782,336  | 94,534,942,720  |
| llama-3.2-3b-instruct-fp8         | RedHatAI/Llama-3.2-3B-Instruct-FP8                     | 377571d314b30f1d58448499e4100e2deafe7d7d | 4,404,163,249  | 92,897,808,384  | 87,486,226,432  |
| llama-3.2-3b-instruct-fp8-block   | unsloth/Llama-3.2-3B-Instruct-FP8-Block                | 08cf804398b23fab4a1df02fbe8d4d5a11a800cc | 3,624,601,065  | 94,534,942,720  | 90,773,061,632  |
| llama-3.2-3b-instruct-awq         | casperhansen/llama-3.2-3b-instruct-awq                 | 272b3bde867b606760447deb9a4d2719fbdfd3ae | 2,270,034,219  | 90,773,061,632  | 86,498,451,456  |
| llama-3.2-3b-instruct-gptq        | shuyuej/Llama-3.2-3B-Instruct-GPTQ                     | dd5a311f040728fbc612eb03c8dadfae0a90552f | 2,264,970,345  | 87,486,226,432  | 82,073,235,456  |
| llama-3.2-3b-mxfp4-a4             | matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq             | 91925ffda6977d097354a99718a20e035f8af80a | 2,303,041,140  | 86,498,451,456  | 80,744,726,528  |
| llama-3.1-8b-instruct-mxfp4a16    | FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16 | 14c3aca849a72df8fcc8b3a30ab8d9eed86ee646 | 5,827,024,421  | 82,076,160,000  | 72,268,472,320  |
| llama-3.1-8b-instruct             | unsloth/Llama-3.1-8B-Instruct                          | 4699cc75b550f9c6f3173fb80f4703b62d946aa5 | 16,077,901,654 | 132,625,727,488 | 116,133,445,632 |

Total 41,185,550,352 bytes. The 8B BF16 download waited while free disk was ≈ 49 GB and ran after the cleanup (70 GB floor).

**Fixture scripts (Task 11, fixtures builder) (user decision 2026-09-29: accepted as recommended):**

1. The dequantized reference copy rounds each decoded weight (computed in F32) to nearest-even BF16: exact for MXFP4, up to 2^-9 relative for FP8 and INT4 × scale. Alternative: an F32 copy with F32 quantized layers in the reference.
2. Activation fake-quantization hooks run in F32 and hand BF16 back to the layer, which runs in BF16. Alternative: those linears in F32.
3. F16 embeddings and norms of AWQ / GPTQ checkpoints are rounded to BF16 in the copy (as a BF16 transformers load).
4. Unsupported variants (compressed-tensors asymmetric INT4, act-order `g_idx`) exit with an error; `hf_fp8` activations map dynamic → per token (per group 128 with blocks), static → per tensor.
5. `dequantize_checkpoint.py` uses the CPU torch build; `quant_reference.py` keeps `hf_reference.py`'s exact pins.
6. `--act-quant` defaults to `auto` (the checkpoint's scheme); an explicit different mode only warns.
7. `quant_fixtures_valid` lists the 8B BF16 baseline slug; `yarn16` is left out (not quantized, 17 prompts).
8. The reference's `model` field is `--model-name`, else the directory name (the checkpoints' `_name_or_path` is a local path); committed fixtures pass `--model-name <hub-id>`.

**YaRN namespace (6a Task 27, YaRN builder, landed c848810 + 835d37d) (user decision 2026-09-29: accepted as recommended):**

1. `ModelIdentity` carries `rope_hash: [u8; 32]` (BLAKE3 of `ModelArchConfig::rope_identity()`) instead of the plan's `rope: String`, keeping `ModelIdentity` `Copy`; the namespace key is equivalent (its JSON is hashed).
2. The model fingerprint does not include RoPE (lookups go through the namespace); alternative: fold it in, changing the fingerprint golden.
3. `rope_hash` is always in the namespace JSON, so every cached key changes once (L2 is wiped at startup anyway).

**INT4 packagings (6a Task 9, lead) (user decision 2026-09-29: accepted as recommended):**

1. Group sizes 32, 64 and 128 are served (the proof checkpoints use 128; 32 lets every registered family's tiny checkpoint be written in the format for the conformance suite); others are refused `quant_scheme_unsupported`. Alternative: 128 only.
2. GPTQ `sym: false` loads with stored zero points (`Int4GroupZp`, `checkpoint_format: gptq` zeros + 1); symmetric GPTQ must store zero 8 everywhere, else refused. Alternative: refuse asymmetric GPTQ.
3. AWQ and GPTQ tensors left unquantized may be F16 and are converted to BF16 at load, each read whole (an embedding is up to ≈ 1 GB of host memory for an 8B model, above the 256 MB staging bound). Alternative: a chunked conversion in the loader.
4. AWQ `modules_to_not_convert` entries match as substrings of the module name (AutoAWQ / vLLM), compressed-tensors `ignore` entries as compressed-tensors defines them.
5. compressed-tensors `pack-quantized` is served symmetric only and without `actorder` (refused `gptq_act_order`), matching the fixture scripts.

**MXFP4 packagings (6a Task 10, lead) (user decision 2026-09-29: accepted as recommended):**

1. compressed-tensors `actorder: static` (the FabioTrindade 8B checkpoint) and Quark GPTQ `desc_act: true` with `static_groups: true` (the matmelis 3B checkpoint) are served: static groups keep the weights in order, so nothing is permuted at run time; other act orders are refused `gptq_act_order`. The same `static` acceptance now applies to compressed-tensors INT4.
2. Quark checkpoints are served only with an empty `layer_quant_config` / `layer_type_quant_config` / `kv_cache_quant_config`, `export.weight_format: real_quantized`, `pack_method` `reorder` or `order`, weights and inputs `fp4 per_group 32 e8m0 half_even even`; anything else is refused `quant_scheme_unsupported`.
3. Ignore entries written as globs (Quark `exclude`, OpenAI `modules_to_not_convert`) match with `*` as any text from the start of the module name.
4. `quant_method: modelopt` and compressed-tensors `nvfp4-pack-quantized` are refused naming `phase-2b-nvidia` before any format is tried.

**Quantized layers under tensor parallelism (6a Task 21, lead) (user decision 2026-09-29: accepted as recommended):**

1. With quantized activations (static per-tensor, per-token or per-group FP8, MXFP4 emulation) the host TP test holds greedy tokens and the likely candidates (logprob > −2) to the golden bounds but not the far tail: each all-reduce's BF16 rounding can flip an FP8 code of the next layer's input (seen: a −25 logprob candidate moved 0.79 at tp 2 with static FP8). Weight-only formats keep the full golden bounds (worst 0.34–0.41 of them). The lab gate against each slug's reference (`--batched-bounds`) stays the real bound. Alternative: a looser tail bound, or quantizing activations on the full rows before the split.
2. Per-token dynamic activation scales are computed per rank on its row-parallel input slice (as vLLM does), not over the full row.
3. Pipeline-parallel stages with quantized weights are refused (`quant_pipeline_unsupported`) until tested; the spec's S-12 names tensor parallelism only.
4. The test is model-level (`tiny_model tp2_quantized_matches_tp1_on_host`, host collective) instead of the plan's server-level `tiny_server`; the lab two-GPU leg stays for when the HIP kernels land (Tasks 14, 18, 20).

**Quantization status and metrics (6a Task 25, lead) (user decision 2026-09-29: accepted as recommended):**

1. `turbine_qgemm_calls_total{scheme,impl}` is dropped: a decode-graph replay runs every quantized GEMM without the host call a counter would count; the per-config implementation is already `turbine_kernel_provider_selected{op="qgemm",…}`. Alternative: count per graph key at capture and add on replay. Spec S-19 and Interfaces amended.
2. The support key's weight column is now the detected packaging's (Task 2 had left it BF16). Every Phase 6a format gets an `experimental` row on `amd/gfx1201/LlamaForCausalLM` with BF16 KV while its proof runs (turned `supported` by each proof task after its gate), and on the `cpu` backend with BF16 or FP8 KV (tests and tiny checkpoints). Without these rows the proofs could not serve.
3. The status test lives in `tiny_server` (the server harness) instead of `turbine-api tests/api.rs`.

## P6: FP8 paged attention — provider evaluation (kernel reuse rule)

Plan Task 23 (spec S-13): `kv.dtype: fp8_e4m3` stores L0 pages as OCP e4m3fn bytes with one K and one V scale per layer (decision Q10). Contract kept from the CPU reference: a page element is written `e4m3(x / scale)` and read `bf16(e4m3 · scale)`, and attention is then exactly the BF16 attention of the CPU model (Q, P and the output stay BF16/F32 as today). Candidates, on `gfx1201` (R9700, ROCm 7.14.1, CK `therock-7.14.1` as pinned by `kernels/rocm/CMakeLists.txt`):

| Candidate                                                                                                                                                                                                                    | Builds on gfx1201?                                                      | Fits the contract?                                                                                                                                                                                                                                                                                                                              | Outcome                                                                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CK `fmha_fwd_pagedkv` / `fmha_fwd_splitkv` FP8 instances (`01_fmha/codegen/ops/fmha_pagedkv_prefill.py`, `fmha_fwd_splitkv.py`: gfx12 factory has `fp8`/`bf8` d64/128/256 tiles)                                             | the generator lists them (`FmhaFwdFp8`)                                 | no: `FmhaFwdTypeConfig<FmhaFwdFp8>` makes Q, K, V **and P and O** e4m3 with `do_fp8_static_quant` (static `scale_s`/`scale_p`/`scale_o`); the BF16-output variant `fp8bf16` is `pass # TODO` for pagedkv and split-KV. Using it would quantize Q, the probabilities and the output with static scales we do not have (Q10 excludes calibration) | rejected on numerics (not measured)                                                                                                                            |
| CK BF16 `fmha_fwd_pagedkv` over FP8 pages **dequantized into a BF16 staging pool** (the batch's pages converted to BF16 blocks in the context's attention scratch with a staged block table, then the unchanged CK instance) | yes (existing instances)                                                | yes: CK sees exactly the BF16 values the contract defines                                                                                                                                                                                                                                                                                       | **chosen for prefill** (`ck_tile_fmha_pagedkv_fp8_staged`, pages of a multiple of 128) — see timings                                                           |
| Same staging for decode (CK split-KV over staged pages)                                                                                                                                                                      | yes                                                                     | yes                                                                                                                                                                                                                                                                                                                                             | rejected: decode is bandwidth-bound; staging reads 1 B + writes 2 B + CK reads 2 B per element (2.5× the BF16 KV traffic)                                      |
| FP8-reading variant of the Turbine paged kernel (dequantize on load into the LDS tiles)                                                                                                                                      | yes                                                                     | yes                                                                                                                                                                                                                                                                                                                                             | kept as the any-page-size fallback (`turbine_hip_fp8`); scalar two-pass kernel, too slow as the main prefill path — see timings                                |
| llama.cpp HIP flash attention with quantized KV (`fattn-vec`/`fattn-tile`)                                                                                                                                                   | (built for the Pre-Phase-5 #3 evaluation at a97cce8)                    | no: its quantized KV types are q8_0/q4_0/… blocks with F16 scales, not e4m3 pages; contiguous per-stream KV with a mask, no page table — adapting means a new KV type plus a page-table load path, i.e. our own kernel inside its templates                                                                                                     | rejected                                                                                                                                                       |
| vLLM ROCm paged attention (`csrc/rocm/attention.cu`, FP8 KV via `Fp8KVCacheDataType`, gfx12 WMMA path)                                                                                                                       | not built                                                               | partly: FP8 K/V with per-layer scales and BF16 Q is exactly this contract, but pages of 16/32 tokens only, K laid out `[blocks, kv_heads, head/x, block, x]` and V transposed (a KV-format change for pool, append, prefill and tiers), decode only, and the file includes torch headers                                                        | rejected (layout/page size, as in "Pre-Phase-5 #3")                                                                                                            |
| aiter paged attention FP8                                                                                                                                                                                                    | no: MFMA (gfx9) / gfx1250 ASM only; the Gluon path is Triton (excluded) | –                                                                                                                                                                                                                                                                                                                                               | rejected                                                                                                                                                       |
| Own FP8 decode kernel (`turbine_hip_fp8_decode`: one workgroup per sequence and KV head, every query head of the group at once, K/V rows read once as bytes, exact two-pass maximum)                                         | yes                                                                     | yes (CPU model up to f32 summation order)                                                                                                                                                                                                                                                                                                       | **chosen for decode** (P rounded against a running maximum as in CK's FMHA, so within the BF16 tolerance rather than bit-exact to the CPU model) — see timings |

Timings (GPU 0, bench lock, `hip_ops decode_attention_timings_fp8` / `prefill_op_timings_fp8`, µs per call, append included):

Decode, `attention_decode_paged` at 128-token pages (µs; BF16 rows are the served CK choice, FP8 rows the new implementations; final kernel = online-softmax version):

| shape               | BF16 CK split-KV (Llama) / pagedkv (OLMoE) | FP8 `turbine_hip_fp8_decode` | FP8 `turbine_hip_fp8` (Turbine kernel) | BF16 `turbine_hip` |
| ------------------- | ------------------------------------------ | ---------------------------- | -------------------------------------- | ------------------ |
| Llama 24/8 b1 @768  | 49.6–54.6                                  | 55.4–56.4                    | 484                                    | 396                |
| Llama b16 @768      | 117.5–131.9                                | 133.8–138.5                  | 1,048                                  | 855                |
| Llama b64 @768      | 394–403                                    | 481–496                      | 3,027                                  | 2,497              |
| Llama b16 @2k       | 250–260                                    | 289–294                      | 2,726                                  | 2,190              |
| Llama b16 @8k       | 882–890                                    | 928–966                      | 11,497                                 | 9,221              |
| OLMoE 16/16 b1 @768 | 48.8                                       | 50.2                         | 491                                    | 386                |
| OLMoE b16 @768      | 200.4                                      | 157.2                        | 1,466                                  | 1,198              |
| OLMoE b64 @768      | 698.8                                      | 721.1                        | 5,269                                  | 4,438              |
| OLMoE b16 @2k       | 479.0                                      | 349.6                        | 3,877                                  | 3,264              |
| OLMoE b16 @8k       | 1,765                                      | 1,143                        | 16,377                                 | 12,947             |

(A first two-pass version of the FP8 decode kernel with the exact CPU maximum ran 226 µs at Llama b16 @768 and 1,648 at @8k; the single-pass online-softmax version above is within ~5–10 % of CK split-KV for Llama and faster than CK pagedkv for OLMoE, whose KV bytes it halves.)

Prefill, `attention_prefill_paged` at 128-token pages (µs):

| shape                                 | BF16 CK pagedkv | FP8 `ck_tile_fmha_pagedkv_fp8_staged` | FP8 `turbine_hip_fp8` |
| ------------------------------------- | --------------- | ------------------------------------- | --------------------- |
| Llama 16 × 512 new                    | 500             | 701                                   | 20,514                |
| Llama 16 × 512 new after 1,024 cached | 1,696           | 1,926                                 | 86,324                |
| Llama 1 × 2,048                       | 336             | 480                                   | 19,987                |
| OLMoE 16 × 512                        | 577             | 857                                   | 14,372                |
| OLMoE 16 × 512 after 1,024            | 2,482           | 3,050                                 | 57,915                |
| OLMoE 1 × 2,048                       | 259             | 534                                   | 14,527                |

The staging adds 140–560 µs per call (convert + a second CK read of BF16), small against the prefill GEMMs; the Turbine FP8 kernel is 30–45× slower than CK and stays the any-page-size fallback only.

Correctness (lab, R9700): `hip_ops paged_fp8_matches_cpu` passes (worst |Δ| vs CPU 3.9e-3 for the staged CK and Turbine FP8 kernels, 9.8e-4 for the FP8 decode kernel in its two-pass form, 3.9e-3 in the final online form, all within the paged BF16 tolerance) (prefill and decode, 128- and 16-token pages, Llama 24/8, OLMoE 16/16, GQA 32/8, unit and non-unit scales, every FP8 implementation bound alone, page bytes exact after the append, a 64-sequence batch staged in two CK groups, a 66,000-token sequence that exceeds the 256 MiB staging bound and runs the Turbine FP8 kernel) and `every_implementation_matches_cpu` (FP8 cases added).

- A) Prefill `ck_tile_fmha_pagedkv_fp8_staged` (BF16 CK over staged pages, groups of ≤ 256 MiB of staged pages, the Turbine FP8 kernel for a sequence that alone exceeds it), decode own `turbine_hip_fp8_decode`, `turbine_hip_fp8` for other page sizes (chosen, provisional pending user review)
- B) The Turbine FP8 kernel for everything (smallest code, slower prefill and decode)
- C) Fork CK's pagedkv pipeline to convert FP8 K/V tiles to BF16 after the DRAM load (no staging copy; a CK pipeline fork to maintain)

## P6: INT4 group GEMM — provider evaluation (kernel reuse rule)

Phase 6a Task 16 (INT4 builder, 2026-09-29). W4A16 with group-128 scales, BF16 activations, for AWQ zero points (`INT4_GROUP_ZP`) and symmetric GPTQ (`INT4_GROUP_SYM`, implicit 8), at the Llama-3.2-3B linear shapes qkv 5120 × 3072, o 3072 × 3072, gate_up 16384 × 3072 and down 3072 × 8192. Everything ran natively on novanas GPU 0 under `scripts/bench-lock.sh` (the harness also checks correctness on GPU 1). Third-party sources were fetched shallow and sparse into `/home/piwi/turbine-ci/scratch/p6a-int4/` (coordinator's approval, 2026-09-29) and deleted after this entry: ggml-org/llama.cpp @ `680a036285273a3ff56032ec5d7f3352609eba4f` (MIT) and vllm-project/vllm @ `32cc3f1ea886c38cbacde35550eb74defaa7bca1` (Apache-2.0). Nothing from them is in the repository. CK is the pinned `cd9574023093742434e8c992d13b89ab9a6c1cf8` (MIT). The harness is `kernels/rocm/tools/qgemm_int4_eval.cpp` (`-DTURBINE_BUILD_QGEMM_INT4_EVAL=ON`). It runs every `turbine_qgemm` implementation of the library through the ABI, plus the BF16 baseline (`turbine_gemm` on the BF16-rounded dequantized weight: the BF16 model's cost). It rotates over ≥ 256 MiB of weight copies (past the 64 MiB Infinity Cache), reports the median µs of 5 × 20 calls, and checks the sampled rows against the CPU provider's semantics.

| Candidate                                                                                          | Builds on gfx1201?                                                                                                                                                                                                                                                                              | Correct?                                                                                                                                         | m=1 (µs, qkv/o/gate_up/down = sum)                               | m=16                                 | m=128              | m=2048                                                       |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------------ | ------------------ | ------------------------------------------------------------ |
| BF16 baseline (hipBLASLt, tuned table; not an INT4 candidate)                                      | yes                                                                                                                                                                                                                                                                                             | reference                                                                                                                                        | 57.0/36.1/164.8/85.6 = 344                                       | 358                                  | 439                | 5,663                                                        |
| CK `ck_tile` `gemm_quant` BQuantGrouped, BF16 A × `pk_int4` B, F32 group scales                    | the plain BQuant pipeline does not (its block GEMM static-asserts FP8/BF8 compute: W4A8 only); the preshuffled-B pipeline (`WPQuantBPipelineAgBgCrV2`) builds with a 16×16×16 WMMA warp tile and K tile 128 (CK example 38's own configs fail on gfx12: K tile < group, warp-GEMM distribution) | CK's CPU check: correct at m 1/4/16 (sym)                                                                                                        | 29.8/27.6/40.8/58.6 = 157                                        | 156                                  | 519                | 7,516 (≈ 55 TFLOPS; the 128×128 prefill config is 3× slower) |
| llama.cpp HIP `mul_mat_vec_q` / `mmq`, q4_0 (sym) and q4_1 (scale + min), `test-backend-ops perf`  | yes (FA off, gfx1201)                                                                                                                                                                                                                                                                           | its own test mode passes; W4A8 (activations quantized to q8_1), not the W4A16 reference; q4_1's FP16 d and m cannot hold AWQ's (q − z)·s exactly | q4_0 24.5/112.8†/42.3/18.8 (weights not rotated: cache-resident) | 147                                  | q4_0 397, q4_1 431 | q4_0 4,803, q4_1 5,225                                       |
| vLLM `wvSplitK_int4_g` (`csrc/rocm/skinny_gemms_int4.cu`, gfx11/gfx12)                             | yes (kernel templates, no torch; scratch driver)                                                                                                                                                                                                                                                | yes after a repack: its own nibble interleave, BF16 scales, zero points [n/8, groups] u32                                                        | ZP 18.1/13.0/45.2/25.7 = 102                                     | not supported (≤ 4 tokens; m=4: 145) | –                  | –                                                            |
| vLLM `gptq_gemm_rdna3` / `_wmma` (`csrc/rocm/q_gemm_rdna3*.cu`)                                    | no: compiled for `__gfx1100__` only, gfx11 WMMA builtins (duplicated-lane operands)                                                                                                                                                                                                             | –; its K-split accumulates with BF16 atomics (order-dependent results)                                                                           | –                                                                | –                                    | –                  | –                                                            |
| vLLM AWQ on ROCm / aiter                                                                           | excluded: Triton (Python) / gfx9 only                                                                                                                                                                                                                                                           | –                                                                                                                                                | –                                                                | –                                    | –                  | –                                                            |
| **Own `turbine_hip_int4_wmma`** (codes streamed into the WMMA registers, `qgemm_int4_kernels.hpp`) | yes                                                                                                                                                                                                                                                                                             | yes: exact F32 weights, max \|Δ\| ≤ 2.6e-5 in F32 (`hip_qgemm_int4`); rows batch-invariant                                                       | ZP 21.3/15.2/52.9/31.9 = **121**                                 | **134**                              | **637**            | 9,868                                                        |
| **Own `turbine_hip_int4_dequant`** (dequantize 32 MiB chunks to BF16, then hipBLASLt)              | yes                                                                                                                                                                                                                                                                                             | yes: bitwise the BF16 baseline's arithmetic (the golden reference's BF16-rounded weight)                                                         | 551                                                              | 551                                  | 705                | **5,136**                                                    |

† an outlier of that run (q4_1 241.8). All µs are GPU 0, ZP; SYM within 2 % (e.g. fused m=1 120, m=128 632). The external candidates are from the run of 03:28, the BF16 baseline and the own implementations from the harness run of 04:08 (0 failures, every INT4 implementation within tolerance at every m). The own kernels at m=64 take 347 µs (fused) and 591 (dequant), at 256 1,239 and 886, at 512 2,414 and 1,400 (BF16: 370, 751, 1,484). Launch-shape sweep (`--sweep 1`, m 1 and 16): one 16-column tile per wave is best; 2 waves splitting k are best for k 3072 (gate_up m=1 53.2 µs vs 60.4 with 8 waves) and 8 for k 8192 (down 31.3 vs 44.0 with 2).

Why the own kernels. No external kernel serves the ABI v2.9 INT4 layout (`[n, k/2]`, low nibble first, F32 scales, U8 zero points) with BF16 activations and zero points:

- CK has no zero points and needs its preshuffled B. A preshuffle would be a load-time repack per implementation, which the v2.9 ABI does not have.
- vLLM's kernel stops at 4 tokens and needs its own packing.
- llama.cpp changes the arithmetic (8-bit activations) and would bring in ggml's CUDA/HIP tree.

What each external candidate did well:

- vLLM's skinny kernel is 16 % faster at one token (102 vs 121 µs per layer).
- CK is faster from about 64 to 128 rows (519 vs 637 at 128).
- llama.cpp's mmq is faster at 128 rows (397), but on cache-resident weights and W4A8.

The fused kernel decodes codes without arithmetic: 0x4300 | q is the BF16 of 128 + q. A ones-WMMA gives each group's Σa, and the group adds s·(acc − (128 + z)·Σa) in F32. It streams 16 bytes of codes per lane per 64-deep step, and the waves of a block split k by a rule that depends on k only (2 waves for k 3072, 8 for k 8192). The sweep of 1/2/4 column tiles × 1–8 waves is in `--sweep`. Rows are therefore batch invariant.

**Chosen (2026-09-29): user decision 2026-09-29: accepted as recommended:**

1. Own kernels, registered as `turbine_hip_int4_wmma` and `turbine_hip_int4_dequant`, after `hipblaslt_fp8` in library order. Alternative: vendor vLLM's `wvSplitK_int4_g` (Apache-2.0, adapted to the ABI layout and F32 scales) as a third implementation for m ≤ 2, for about 0.5 ms per token at c1.
2. The gfx1201 card profile gives `qgemm` row tiers: ≤ 128 rows the fused kernel, above that the dequant path (layer sums 637 vs 705 µs at 128, 1,239 vs 886 at 256). The FP8 implementation leads both tiers' orders. Alternative: 64, the earlier GPU 1 crossover.
3. The dequant path's staging buffer: the largest layer's BF16 weight (96 MiB for the 3B gate_up, 224 MiB for the 8B; above 512 MiB the weight is staged in column chunks), per context, grown at its first call (refused while a graph is captured) and freed with the context. It is not in the memory budget, like the MoE scratch. Decode graphs are decode steps (`scheduler.max_running_requests` 64 rows in the lab configs), so they run the fused kernel and never grow it. Alternative: register the buffer with the memory budget (a qgemm workspace the loader sizes).
4. Supported groups:
   - fused: 64, 128 and 256 (the step is 64 deep);
   - dequant: any multiple of 32, so group-32 checkpoints (the tiny conformance fixtures) run the dequant path on HIP;
   - others are refused by `supports`.
5. Numerics. The fused kernel uses exact F32 weights; the dequant path uses the BF16-rounded weight, as the golden references do. Decode calls run the implementation their row tier binds. Prefill calls (`turbine_qgemm_desc::prefill`) run the dequant path under either binding, as one GEMM of the layer's own shape with `TURBINE_OPTION_GEMM_PREFILL` raised, so hipBLASLt runs the tuned table's invariant class and every row is computed the same way whatever m. A prefix-reused INT4 prefill therefore reproduces the cold one bit for bit (Phase 4), as BF16 does, for shapes with table rows (Llama-3.2-3B at tp 1 and 2); `hip_qgemm_int4 int4_rows_are_batch_invariant` checks prefill rows at 1 to 513 rows across both bindings (coordinator's correction, 2026-09-29: this item was first recorded as not bit-exact, which Phase 4 does not allow). Cost: a small prefill chunk pays the dequant pass (about 0.55 ms per Llama-3.2-3B layer set at any m) instead of the fused kernel.
6. Follow-up, not in this task: a mid-m kernel (weights reused across row tiles, 64–512 rows), where CK and llama.cpp show 20–40 % headroom over both tiers.

**Golden tolerance for activation-quantized checkpoints (6a Tasks 14, 20; lead, 2026-09-29) (user decision 2026-09-29: accepted as recommended):** FP8 W8A8 golden c1 fails the BF16 Llama bounds (likely |Δ| 0.17–0.57, tail 0.5–2.8; greedy tokens mostly identical, divergences at margins 0.01–0.17) while `hipblaslt_fp8` matches the CPU reference within one BF16 ulp. A) calibrate each activation-quantized slug from transformers' own spread with the activation fake-quantization hooks (`self_spread.py --act-quant`, the OLMoE method) and make `quant_reference.py` quantize a fused projection with its parts' largest `input_scale` as the loader and vLLM do — chosen; B) adopt a fixed looser bound (e.g. OLMoE's 1.01 / 1.66) without calibration; C) keep the BF16 bounds and leave the rows unsupported. The FP8 builder owns `scripts/golden/{self_spread,quant_reference}.py` from now on.

## P6: MXFP4 GEMM — provider evaluation (kernel reuse rule)

Date: 2026-09-29 (Phase 6a Task 19, MXFP4 builder). Card: R9700 (gfx1201), GPU 0 under `bench.lock` (and `port18000.lock`, queued like lab-bench) with no swap-in, ROCm 7.14.1, hipBLASLt 1.4.1, CK at the pinned `cd9574023093742434e8c992d13b89ab9a6c1cf8`, llama.cpp at `680a036285273a3ff56032ec5d7f3352609eba4f` (the INT4 builder's checkout, copied; nothing downloaded). Harness: `kernels/rocm/tools/qgemm_mxfp4_eval.cpp` (`-DTURBINE_BUILD_QGEMM_MXFP4_EVAL=ON`): per shape and m, hipBLASLt BF16 on the dequantized weight (first heuristic answer and best of 8), a dequantize-to-BF16 kernel + that GEMM, and each `turbine_hip_mxfp4` tile; median of 5 rounds × 20 calls, rotating over enough weight copies (≥ 512 MiB) to defeat the 64 MiB infinity cache; correctness on 6 sampled rows against a host reference with `cpu::qgemm`'s semantics (E2M1 × 2^(E8M0 − 127) weights, BF16 activations, exact products, one rounding to BF16; bound one BF16 rounding plus 1e-6). Shapes: Llama-3.2-3B qkv 5120×3072, o 3072×3072, gate_up 16384×3072, down 3072×8192; Llama-3.1-8B qkv 6144×4096, o 4096×4096, gate_up 28672×4096, down 4096×14336.

### Candidates

| Candidate                                                                                                                                                                            | Builds on gfx1201 | Correct                                                                                                                                                                                                                                                                                                                | Notes                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CK `ck_tile` MX GEMM (`example/ck_tile/42_mx_gemm`, `gemm_mx` ops)                                                                                                                   | no                | —                                                                                                                                                                                                                                                                                                                      | gfx950 only (scaled MFMA); its CMake lists `SUPPORTED_GPUS gfx950`                                                                                                                                                                                                   |
| CK `ck_tile` `gemm_quant` BQuantGrouped BF16 × `pk_fp4_t` with E8M0 group-32 scales (`38_block_scale_gemm/gemm_bquant_quantgrouped_mx_bf16fp4.cpp`, the W4A16 MXFP4 microscale path) | yes               | **no** with the example's `GemmConfigQuantPrefill` (all outputs 0, 98 % wrong, "353 TFLOPS"); **yes** with the WMMA warp tile `GemmConfigBQuantPrefill_Wmma` (16×16×16), CK's own CPU check passes at m = 128 and 2048                                                                                                 | m must be a multiple of 128 (`kPadM` false: m = 1, 16, 513 refused `Arguments not supported`); 11–13 TFLOPS at m = 2048 on every shape (3b.o 3.02 ms vs 0.27 ms hipBLASLt BF16; 8b.gate_up 43.5 ms vs 4.4 ms): ~10× slower than BF16                                 |
| llama.cpp HIP `GGML_TYPE_MXFP4` `mul_mat_vec_q` / `mmq` (`test-backend-ops`, patched perf list with our shapes)                                                                      | yes               | its own check vs ggml's CPU backend passes (46/46 MUL_MAT mxfp4 cases); **not the checkpoint's arithmetic**: activations are quantized to 8 bits per 32 (`q8_1`) before an int8 dot product, i.e. W4A8, not W4A16 (and not the BF16 GEMM after MXFP4 emulation that Quark W4A4 needs); F16 activations are unsupported | decode (m 4–16) about equal to `turbine_hip_mxfp4` (its perf mode does not rotate weights, so shapes under 64 MiB run from the infinity cache: 3b.gate_up m = 1 at an apparent 900 GB/s); prefill equal to ours (3b.gate_up m = 2048 2,213 µs vs 2,223)              |
| hipBLASLt                                                                                                                                                                            | not probed        | —                                                                                                                                                                                                                                                                                                                      | the FP8 evaluation found no mixed BF16 × low-precision kernel on gfx1201; MXFP4 needs BF16 × FP4 with E8M0 vector-32 scales, which hipBLASLt offers only for gfx950 FP4 × FP4 (inference, not measured)                                                              |
| The INT4 path generalised to E2M1 × E8M0: dequantize to BF16 + hipBLASLt (`dequant+bf16` in the harness; the INT4 tier above 128 rows)                                               | yes               | yes                                                                                                                                                                                                                                                                                                                    | needs an n × k BF16 scratch (up to 235 MB for 8b.gate_up); 1.5–9× slower than the fused kernel up to m = 128; at m = 2048 faster on 3b.qkv (20 %), 3b.gate_up (14 %) and 3b.o (9 %), equal on 3b.down, 8b.qkv and 8b.gate_up, slower on 8b.o (4 %) and 8b.down (9 %) |
| The INT4 path generalised: a fused WMMA kernel decoding the codes in registers (own) — `turbine_hip_mxfp4`                                                                           | yes               | yes (all rows within the bound, max \|Δ\| ≤ 2e-2 at outputs ≈ 4–8; lab `hip_qgemm_mxfp4` below)                                                                                                                                                                                                                        | picked, see below                                                                                                                                                                                                                                                    |

### µs per call, GPU 0 (the recorded run, `turbine_hip_mxfp4` with its final tiles)

| shape      | m    | BF16 first | BF16 best of 8 | dequant+BF16 | `turbine_hip_mxfp4` | llama.cpp MXFP4 (cached) | llama.cpp BF16 |
| ---------- | ---- | ---------- | -------------- | ------------ | ------------------- | ------------------------ | -------------- |
| 3b.qkv     | 1    | 68.0       | 54.6           | 122.1        | **16.1**            | 25.0                     | 38.4           |
| 3b.qkv     | 16   | 68.0       | 55.6           | 123.9        | **17.9**            | 23.3                     | 23.2           |
| 3b.qkv     | 128  | 70.8       | 59.2           | 139.2        | **78.0**            | 68.7                     | 44.3           |
| 3b.qkv     | 2048 | 453.3      | 447.0          | 576.1        | **715.9**           | 731.5                    | 545.0          |
| 3b.o       | 1    | 49.1       | 39.5           | 80.4         | **12.1**            | 124.3                    | 47.8           |
| 3b.o       | 16   | 52.3       | 40.6           | 79.7         | **13.6**            | 19.6                     | 14.8           |
| 3b.o       | 128  | 65.3       | 47.0           | 92.1         | **52.3**            | 45.0                     | 38.3           |
| 3b.o       | 2048 | 308.9      | 269.1          | 384.5        | **421.1**           | 432.1                    | 385.8          |
| 3b.gate_up | 1    | 174.2      | 165.6          | 510.4        | **69.8**            | 29.9                     | 169.8          |
| 3b.gate_up | 16   | 178.0      | 170.4          | 514.4        | **66.5**            | 51.6                     | 174.8          |
| 3b.gate_up | 128  | 241.1      | 220.0          | 534.0        | **153.4**           | 147.5                    | 257.7          |
| 3b.gate_up | 2048 | 1497.3     | 1511.4         | 1901.1       | **2223.0**          | 2213.4                   | 1807.7         |
| 3b.down    | 1    | 116.9      | 98.8           | 235.5        | **47.3**            | 21.9                     | 48.5           |
| 3b.down    | 16   | 118.2      | 98.5           | 234.8        | **49.8**            | 40.2                     | 53.8           |
| 3b.down    | 128  | 169.8      | 103.2          | 339.7        | **132.7**           | 114.0                    | 134.8          |
| 3b.down    | 2048 | 1019.2     | 1005.4         | 1181.0       | **1185.6**          | 1213.2                   | 1206.3         |
| 8b.qkv     | 1    | 95.5       | 83.5           | 198.9        | **38.2**            | 33.6                     | 51.5           |
| 8b.qkv     | 16   | 84.4       | 84.3           | 199.4        | **43.4**            | 30.8                     | 39.3           |
| 8b.qkv     | 128  | 100.1      | 89.8           | 237.4        | **113.4**           | 94.7                     | 76.7           |
| 8b.qkv     | 2048 | 961.3      | 885.5          | 1148.2       | **1157.1**          | 1142.5                   | 1116.2         |
| 8b.o       | 1    | 71.9       | 57.2           | 151.6        | **20.9**            | 37.6                     | 46.9           |
| 8b.o       | 16   | 83.2       | 57.5           | 154.2        | **24.8**            | 24.0                     | 26.0           |
| 8b.o       | 128  | 76.6       | 59.7           | 172.6        | **71.9**            | 65.3                     | 80.0           |
| 8b.o       | 2048 | 714.1      | 581.0          | 814.7        | **779.7**           | 788.7                    | 675.6          |
| 8b.gate_up | 1    | 372.8      | 371.7          | 1136.3       | **110.4**           | 56.5                     | 380.4          |
| 8b.gate_up | 16   | 380.4      | 379.9          | 1142.4       | **116.7**           | 109.1                    | 391.2          |
| 8b.gate_up | 128  | 471.0      | 470.4          | 1259.7       | **354.9**           | 344.6                    | 513.7          |
| 8b.gate_up | 2048 | 4359.5     | 4385.4         | 5183.8       | **5174.2**          | 5252.6                   | 4566.4         |
| 8b.down    | 1    | 221.0      | 189.5          | 606.0        | **61.5**            | 33.4                     | 197.2          |
| 8b.down    | 16   | 224.9      | 197.8          | 607.2        | **75.0**            | 70.9                     | 192.3          |
| 8b.down    | 128  | 246.2      | 212.0          | 646.2        | **226.8**           | 205.3                    | 310.2          |
| 8b.down    | 2048 | 2342.7     | 2089.2         | 2790.4       | **2561.7**          | 2763.4                   | 2356.0         |

(m = 4 rows are in the log; they equal m = 1 within noise.) The tile sweep before it (same lock rules, less host load: no lab-test build running) measured the decode tile 25–40 % faster than this run on some shapes (3b.down m = 1–16 29–34 µs, 3b.gate_up 51–58, 8b.down 60–67, 8b.gate_up 108–117; BF16 unchanged), so the decode ratios above are conservative: `turbine_hip_mxfp4` streams the 4.25-bit weights at 400–520 GB/s for m ≤ 16, 3–4× the BF16 GEMM. Tile sweep (µs, m = 32 / 64, the medium tiles chosen): 3b.down 52 / 73, 3b.gate_up 65 / 97, 3b.o 15 / 23, 8b.down 75 / 137, 8b.gate_up 129 / 225 (BF16 best 98–400).

MXFP4_EMULATED activation quantize-dequantize (`turbine_hip_mxfp4` `quantize_act`, own): bit-exact with `cpu::quant` (values and scales) at every size; 3.4–4.1 µs for 1–16 rows, 115 / 152 / 303 / 561 µs for 2,048 rows of 3072 / 4096 / 8192 / 14336 columns (~210 GB/s: one wave per 32-column group, not tuned).

### Pick (user decision 2026-09-29: accepted as recommended)

- `turbine_hip_mxfp4` (provider turbine_hip, own; `src/qgemm_mxfp4.hip`) for every MXFP4 GEMM, scheme MXFP4 with act NONE or MXFP4_EMULATED, BF16 activations, BF16 or F32 out, k a multiple of 64: m ≤ 16 a decode tile (one 16-row fragment, one column fragment per wave, k split over 8 waves, the group sums scaled after the WMMAs), m ≤ 64 2 or 4 row fragments with k split over 4, above an LDS-staged 128 × 128 tile (scaled weights decoded once for 8 row fragments). Codes become BF16 through a 256-entry LDS byte table; the half-waves exchange codes so each WMMA slice holds one group's k values in both operands. Decode-step calls pick the tile by m; every prefill-step call (`prefill` set) runs the LDS tile, whose per-row summation order does not depend on m, so prefill rows are bitwise the same in any batch (Phase 4 prefix reuse; lab `hip_qgemm_mxfp4 qgemm_mxfp4_prefill_rows_are_batch_invariant`).
- Not taken: CK BQuant MX (10× slower, m multiple of 128 only); llama.cpp (W4A8 arithmetic, not the checkpoint's; no faster where both run from memory); dequantize + hipBLASLt (a 235 MB scratch outside the memory budget for 9–20 % at m = 2048 on 3 of 8 shapes, 1.5–9× slower up to m = 128).
- Activation emulation: own kernel, no provider takes the v2.9 `quantize_act` op (CK's MX quantization lives inside its gfx950 MX GEMM pipelines).
- Findings to act on: (1) prefill (m = 2048) takes 1.18–1.60× the hipBLASLt BF16 time (3b.qkv 1.60×, 3b.o 1.56×, 3b.gate_up 1.47×, 3b.down 1.18×, 8B 1.18–1.34×): TTFT of MXFP4 checkpoints is above BF16's; an MXFP4 prefill kernel at hipBLASLt speed is a follow-up (the S-20 targets are c1 ITL and c16 tok/s). (2) Decode-step rows depend on the tile (k-split order), like hipBLASLt's `speed` rows for Llama; prefill-step rows do not (above). (3) The quantize-dequantize is ~210 GB/s at 2,048 rows (~5 % of a W4A4 prefill step), untuned.

**FP8 KV golden (6a Task 24; lead, 2026-09-29) (user decision 2026-09-29: accepted as recommended):** Llama with FP8 KV (scales 1.0) fails the BF16 golden bounds (c1 4/16, c16 9/16; likely |Δ| up to 0.44, tail 1.58; greedy tokens mostly identical) while the HIP kernels match the CPU reference. A) judge FP8 KV by the token rule and the GSM8K eval only; B′) a transformers reference with FP8 KV emulated (K after RoPE and V quantize-dequantized with the served scales, read back as BF16) and a tolerance calibrated with `self_spread.py --kv-quant fp8_e4m3` (the method used for activation-quantized weights) — chosen; C) warm-up scale calibration (declined at Q10). Spec S-13 AC amended; lab-bench models `llama-fp8kv` / `olmoe-fp8kv`.

**Quark W4A4 accuracy baseline (6a Task 20; lead with the coordinator's approval for the sleeping user, 2026-09-29) (user decision 2026-09-29: accepted as recommended):** the proof checkpoint `matmelis/Llama_3.2_3B_w_mxfp4_a_mxfp4_gptq` quantizes the Llama-3.2-3B _base_ model, so the S-17 gate against BF16 3B _Instruct_ would fail on instruction following regardless of quantization. A) keep the Instruct baseline as written; B) compare with BF16 Llama-3.2-3B base — chosen, with A's result reported alongside; C) drop the W4A4 proof. Download (not in the approved list; approved as provisional): `unsloth/Llama-3.2-3B` @ `d4446454d87d51aa42e1fb174f25acc5f8762331` into `/home/piwi/turbine-models/llama-3.2-3b` (the `meta-llama` repo is gated for the host token; unsloth's is the same weights, ungated, the source the Instruct model already came from). Golden prompts: the overnight pick (the Instruct chat template on the base checkpoint) is replaced by the user's answer **"Find another checkpoint"** — prove W4A4 golden on an Instruct W4A4 MXFP4 checkpoint loadable by our packagings (pinned, downloaded on novanas); if none exists, completion prompts only for the base checkpoint; no self-made quantization without asking. The base-3B accuracy baseline stays as chosen. Downloaded 2026-09-29: 6,442,822,178 bytes, exit 0, revision recorded in the local metadata (free on `/` 220.3 GB before, 149.2 GB after — other agents' builds ran meanwhile).

**GSM8K-200 task set (umbrella S-4 tool, used by the Phase 6a S-17 gates; lead, 2026-09-29) (user decision 2026-09-29: accepted as recommended):** the committed set asked for a bare number in 32 tokens; Turbine BF16 scored 5/200 (Llama-3.2-3B-Instruct) and 16/200 (Llama-3.1-8B-Instruct) — it measured format compliance (`$18`, a sentence), and a 0.02 / 0.04 drop bound on 5–16 correct answers has no power. A) chain of thought: "Solve the problem step by step. On the last line, write "Answer: " followed by the final answer as a number.", `max_tokens` 512, matcher `final_number` (the number after the last `Answer:`, else the last number; `$`, `,` and units ignored) — chosen; B) keep the bare-number prompt, loosen only the matcher (still no reasoning room); C) few-shot prompting. Same 200 items (commit 3101c7d); every baseline and candidate is re-run on the new set; the old eval JSONs are void.

**`fp8_block` served decoded to BF16 (6a Task 15; FP8 builder's evaluation, lead decision 2026-09-29) (user decision 2026-09-29: changed):** no provider computes block-scaled FP8 on gfx1201 — CK `ck_tile` ABQuantGrouped builds but returns ~1e38 on every shape (its own example fails its CPU check at this CK pin), hipBLASLt has no BLK128x128 kernels, and a per-call dequantize costs 2–4 × the BF16 GEMM. A) decode each block-scaled weight to BF16 once at load (`cpu::quant::dequantize` with its scales as a loader companion) and serve it through the BF16 GEMM with BF16 activations — the overnight pick: BF16 speed and memory, no weight-memory saving, the checkpoint's activation scheme not applied (its golden reference is made with `--act-quant none`); B) an own fused W8A16 WMMA kernel (FP8 in memory, dequantized in registers) — **chosen by the user: "Write own kernel in 6a"** (the reuse evaluation above stands: no working block-scaled FP8 provider on gfx1201), with real memory savings, tests against the CPU reference and the fp8_block proof; A stays only as the logged fallback for shapes the kernel does not support; C) refuse `fp8_block` on amd. The column stays `fp8_block`.

## P6: FP8 GEMM — provider evaluation (kernel reuse rule)

Date: 2026-09-29 (Phase 6a Task 12). Card: R9700 (gfx1201), GPU 0, ROCm 7.14.1, hipBLASLt 1.4.1
(100401), CK at the pinned `cd9574023093742434e8c992d13b89ab9a6c1cf8` (therock-7.14.1).
Harness: `kernels/rocm/tools/qgemm_eval.cpp` (+ `qgemm_eval_ck.cpp` for the CK candidates),
built with `-DTURBINE_BUILD_QGEMM_EVAL=ON`, run under `scripts/bench-lock.sh` with
`ROCR_VISIBLE_DEVICES=0`. Per candidate, shape and m: hipBLASLt's first heuristic answer and the
best of its first 8 answers, median of 5 rounds × 20 calls, rotating over 2–8 copies of the weight (as many as 512 MiB allows, at most 8); correctness on 6 sampled rows against a host reference with the CPU provider's semantics
(`cpu::qgemm`: e4m3 values × scales, exact products, one rounding to BF16); tolerance one BF16
rounding (2^-8 relative) plus summation order; "row-invariant" = row 0 of the first answer is
bitwise the same at every m.

Shapes (Llama-3.2-3B, fused as the executor runs them): qkv 5120×3072, o 3072×3072,
gate_up 16384×3072, down 3072×8192.

### Candidates

| Candidate                                                                                                                   | Builds                  | Correct                                                                                                                                                                                                                                                     | Notes                                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hipBLASLt BF16×BF16→BF16 on the dequantized weight (today's path; W8A16 via dequantize)                                     | yes                     | — (baseline)                                                                                                                                                                                                                                                | timing baseline                                                                                                                                                                                                           |
| hipBLASLt e4m3×e4m3→BF16, scalar A/B scales (`SAB`: FP8_TENSOR × FP8_TENSOR)                                                | yes                     | yes, max \|Δ\| ≤ 1.9e-2 at outputs ≈ 5–9 (one BF16 ulp)                                                                                                                                                                                                     | 8 solutions per shape                                                                                                                                                                                                     |
| hipBLASLt e4m3×e4m3→BF16, vector scales `OUTER_VEC_32F` on both (`SABV`: FP8_CHANNEL × FP8_TOKEN)                           | yes                     | yes, max \|Δ\| ≤ 2.6e-2 at outputs ≈ 9                                                                                                                                                                                                                      | 8 solutions per shape; row-invariant on every shape                                                                                                                                                                       |
| hipBLASLt mixed scalar × vector (FP8_CHANNEL × FP8_TENSOR, FP8_TENSOR × FP8_TOKEN)                                          | —                       | —                                                                                                                                                                                                                                                           | no solution on gfx1201 (heuristic returns 0 algorithms); served by broadcasting the scalar to a vector and running `SABV`                                                                                                 |
| hipBLASLt FP8 with vector scales and F32 D                                                                                  | —                       | —                                                                                                                                                                                                                                                           | no solution on gfx1201 (found by `hip_qgemm`): `hipblaslt_fp8` supports BF16 out only                                                                                                                                     |
| CK `ck_tile` `gemm_quant` RowColQuant (per-token × per-channel), decode tile 16×64×256 and prefill tile 128×128×128, BF16 C | yes (gfx1201, OCP e4m3) | **no**: outputs of magnitude 1e35–1e38 on every shape; m = 1 (and m = 16 with the prefill tile) refused by `IsSupportedArgument` (kPadM = false); CK's own example (`tile_example_gemm_quant -quant_mode=rowcol`) fails its CPU verification on gfx1201 too | reported "times" (e.g. o m=2048 90 µs = 430 TFLOP/s, above the card's FP8 peak) are not real work                                                                                                                         |
| CK `ck_tile` `gemm_quant` TensorQuant (scalar × scalar), same tiles                                                         | yes                     | **no** (same garbage outputs)                                                                                                                                                                                                                               |                                                                                                                                                                                                                           |
| vLLM / aiter ROCm scaled-mm                                                                                                 | not built               | —                                                                                                                                                                                                                                                           | vLLM's FP8 linear on ROCm dispatches to `torch._scaled_mm` (hipBLASLt, i.e. the candidates above) or to aiter, whose kernels are gfx942/gfx950 assembly; no separate gfx12 provider exists                                |
| Activation quantization: CK `add_rmsnorm2d_rdquant` (fused norm + per-row quant)                                            | not built               | —                                                                                                                                                                                                                                                           | the v2.9 `quantize_act` descriptor carries no norm inputs, so fusing needs an ABI addition (lead-owned); CK's rdquant also rounds by its own conversion and scale reciprocal, which the bit-exact reference rule excludes |
| Activation quantization: Turbine elementwise kernel (`src/qgemm_quantize.hpp`)                                              | yes                     | **bit-exact** (codes and scales) for FP8_TOKEN, FP8_GROUP128, FP8_TENSOR at every size, including zero rows (floor), saturation, ties                                                                                                                       | own kernel: no provider fits the ABI op                                                                                                                                                                                   |

### µs per call, GPU 0 (first heuristic answer / best of 8)

| shape   | m    | BF16            | FP8 SAB       | FP8 SABV        |
| ------- | ---- | --------------- | ------------- | --------------- |
| qkv     | 1    | 64.8 / 54.6     | 48.4 / 26.7   | 66.8 / 35.4     |
| qkv     | 16   | 56.5 / 55.2     | 36.7 / 27.6   | 57.9 / 36.2     |
| qkv     | 128  | 58.5 / 58.5     | 49.0 / 32.4   | 60.4 / 44.1     |
| qkv     | 2048 | 454.7 / 454.7   | 270.1 / 264.5 | 406.1 / 385.3   |
| o       | 1    | 51.3 / 39.5     | 27.0 / 18.6   | 56.5 / 23.1     |
| o       | 16   | 40.0 / 40.0     | 25.2 / 19.3   | 41.0 / 23.4     |
| o       | 128  | 48.4 / 46.8     | 42.6 / 22.3   | 42.3 / 26.1     |
| o       | 2048 | 302.1 / 301.3   | 164.3 / 164.3 | 256.1 / 248.5   |
| gate_up | 1    | 168.3 / 162.4   | 97.1 / 90.2   | 124.3 / 117.9   |
| gate_up | 16   | 172.8 / 166.5   | 105.7 / 92.9  | 130.8 / 127.1   |
| gate_up | 128  | 233.8 / 221.4   | 167.6 / 107.1 | 182.1 / 160.3   |
| gate_up | 2048 | 1644.6 / 1596.2 | 851.3 / 851.3 | 1238.5 / 1220.7 |
| down    | 1    | 111.4 / 96.5    | 61.7 / 44.1   | 79.5 / 58.2     |
| down    | 16   | 112.9 / 97.2    | 66.1 / 44.6   | 83.9 / 59.5     |
| down    | 128  | 152.1 / 100.0   | 74.3 / 46.6   | 90.5 / 70.0     |
| down    | 2048 | 1005.3 / 1005.3 | 581.9 / 474.6 | 1145.5 / 968.4  |

Activation quantization (Turbine kernel, µs): FP8_TOKEN k=3072: 9.3 / 8.1 / 8.4 / 61.3 at
m = 1 / 16 / 128 / 2048; k=8192: 15.8 / 15.6 / 16.8 / 125.1. FP8_GROUP128 k=3072: 11.9 / 11.4 /
12.1 / 89.0; k=8192: 26.4 / 26.0 / 27.3 / 212.9. FP8_TENSOR k=3072: 6.3 / 6.4 / 6.7 / 53.9;
k=8192: 11.9 / 11.8 / 12.5 / 92.7.

Per decode layer at m = 16 (four GEMMs; the BF16 path runs its tuned table, ≈ best): BF16 358.9 µs;
FP8 SABV 313.6 µs with the first heuristic answers, 246.2 µs with the best, plus ≈ 40 µs of
activation quantization (three k=3072 and one k=8192 token quantizations).

### Pick

- `hipblaslt_fp8` (provider hipblaslt): FP8_TENSOR / FP8_CHANNEL weights × FP8_TENSOR / FP8_TOKEN
  activations through hipBLASLt's FP8 kernels with `SCALAR_32F` (tensor × tensor) or
  `OUTER_VEC_32F` (vector × vector) scale modes; a mixed pairing broadcasts its scalar scale to a
  vector (a device buffer of the context, ≥ 65,536 floats, grown only outside graph capture);
  BF16 out only; k and lda multiples of 16. The only correct provider on gfx1201.
- Activation quantization: own kernel `turbine_hip` (`src/quantize_act.hip`,
  `src/qgemm_quantize.hpp`) — no provider takes the v2.9 `quantize_act` op; CK's fused
  norm+quant would need an ABI change and is not bit-exact with the reference rule.
- Findings to act on: (1) the first heuristic answer is 1.2–2.4× slower than the best of hipBLASLt's
  top 8 at decode sizes (qkv, o, down), so FP8 shapes need pinned solutions (a tuned FP8 table) to
  reach the S-20 targets — without it FP8 decode GEMM time is ≈ BF16's; (2) SABV at m = 2048 is no
  faster than BF16 on down (968 vs 1005 µs) and only 1.2–1.3× on the others (per-token × per-channel
  checkpoints gain in decode, little in prefill); SAB (per-tensor) is 1.7–2.1× faster than BF16
  at m = 2048. (3) CK `gemm_quant` RowCol/Tensor is not usable on gfx1201 at this CK pin.

## P6: FP8 GEMM — pinned solutions and prefill row invariance (addendum to the FP8 GEMM evaluation)

Date: 2026-09-29 (Phase 6a Task 13 follow-up, commit "perf(rocm): pinned FP8 decode solutions and
row-invariant FP8 prefill"). GPU 0 under `scripts/bench-lock.sh`, hipBLASLt 1.4.1 (791 FP8
BF16-out and 780 F32-out solutions listed).

**Decode.** `turbine_qgemm_eval --tune 1` timed every FP8 solution that takes each Llama-3.2-3B
shape at m ∈ {1, 2, 4, 8, 16, 32, 64}; the best beats hipBLASLt's first heuristic answer by
1.3–2.9× (e.g. SABV qkv m=16 35.4 vs 103.7 µs, o 22.0 vs 62.4, down 53.5 vs 121.2). The winners are
pinned in `kernels/rocm/src/qgemm_tuned.hpp` (decode steps only, `TURBINE_OPTION_GEMM_AUTOTUNE`).
Per decode layer at m = 16 (qkv + o + gate_up + down): BF16 tuned 358.9 µs, FP8 SABV pinned
240.0 µs, FP8 SAB pinned 167.0 µs, plus the activation quantization.

**Prefill.** Prefix reuse (Phase 4) needs a prefill row's result to be bitwise independent of the
call's size and of the row's position in it. `--tune-prefill 1` checked every solution with split-K
off, fastest first, against rows of a 513-row call recomputed as calls of 1, 7, 128 rows (from
row 0), 213 rows (from row 300) and 64 rows (from row 1), each in fresh buffers:

- scalar scales (SAB, BF16 out): row-invariant solutions exist for every shape (1–25 faster ones
  rejected); pinned as the shapes' prefill rows.
- vector scales (OUTER_VEC, SABV): **none** of the 11 solutions that take the problem is
  row-invariant on gfx1201. Evaluated alternative: a scalar-scale solution with unit scales and F32
  out (row-invariant ones exist, 0–5 faster rejected) plus an **own epilogue kernel**
  (`src/qgemm_epilogue.hpp`) applying the row and column scales into BF16. At m = 2048 (GEMM +
  epilogue) vs SABV's first answer: qkv 320 vs 406 µs, o 200 vs 256, gate_up 1332 vs 1238, down
  528 vs 1145. Picked for FP8_CHANNEL / FP8_TOKEN prefills (chunked to 64 MiB of F32 sums).

User decision 2026-09-29: accepted as recommended: the epilogue is an own kernel (no provider has a row-invariant
vector-scale FP8 solution on gfx1201); the 64 MiB of prefill sums per context are not in the model's
memory accounting; the 8B shapes have no pinned rows yet (they run the heuristic and log
`event=qgemm_prefill_unpinned`).

## P6: block-scaled FP8 GEMM — provider evaluation (kernel reuse rule)

Date: 2026-09-29 (Phase 6a Task 15). Card: R9700 (gfx1201), GPU 0 under `scripts/bench-lock.sh`,
ROCm 7.14.1, hipBLASLt 1.4.1, CK at the pinned `cd9574023093742434e8c992d13b89ab9a6c1cf8`.
Harness: `kernels/rocm/tools/qgemm_eval.cpp --block 1` (CK instances in `qgemm_eval_ck.cpp`),
Llama-3.2-3B shapes, 128 × 128 weight blocks (F32 block scales, `[n/128, k/128]`), activations
per token and 128-column group (FP8_GROUP128) for the W8A8 candidates; correctness on 6 sampled
rows against a host reference of `cpu::qgemm` semantics (one BF16 rounding of the exact sum).

| Candidate                                                                                | Builds | Correct                                                                      | µs (m = 1 / 16 / 128 / 2048)                                                                                                |
| ---------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| CK `ck_tile` `gemm_quant` ABQuantGrouped (A 1×1×128, B 1×128×128), decode tile 16×64×256 | yes    | **no**: outputs of magnitude 1e37–1e38 on every shape; m = 1 refused (kPadM) | qkv —/48/82/1070 (not real work)                                                                                            |
| same, prefill tile 128×128×128 (`GemmConfigABQuantPrefill`)                              | yes    | **no** (same); m < 128 refused                                               | qkv —/—/54/203 (not real work)                                                                                              |
| hipBLASLt block scales (`BLK128x128_32F`, `VEC128_32F`)                                  | —      | —                                                                            | "not supported yet" in hipBLASLt 1.4.1; no gfx1201 kernel                                                                   |
| W8A16 per call: own dequantize-to-BF16 kernel + hipBLASLt BF16 GEMM                      | yes    | dequantize **bit-exact** vs the host decode                                  | dequantize 175 (qkv) / 111 (o) / 573 (gate_up) / 282 (down) per call, **plus** the BF16 GEMM: 2–4× the BF16 layer at decode |
| W8A16 dequantized once at load (BF16 weights on the device, hipBLASLt BF16 GEMM)         | yes    | exact decode (host `cpu::quant::dequantize`)                                 | = BF16: qkv 56/57/59/454, o 40/40/52/312, gate_up 169/173/235/1499, down 114/113/143/1015                                   |

CK's `ck_tile` FP8 `gemm_quant` path gives the same wrong results for RowColQuant and TensorQuant
(decision "P6: FP8 GEMM — provider evaluation"), and CK's own `tile_example_gemm_quant` fails
its CPU verification on gfx1201: not usable at this CK pin (a vendor defect, not debugged further).

### Pick (per Q3: W8A16 through a dequantize-to-BF16 path)

`fp8_block` is served W8A16 with its weights **dequantized to BF16 once at load** (the loader decodes
each block-scaled tensor with `cpu::quant::dequantize` through the staging buffer and runs the layer
on the BF16 GEMM; no activation quantization). It is the only candidate that is correct and meets
the S-20 target (c16 ≥ 1.0 × BF16): the per-call dequantize path costs 2–4× the BF16 GEMM, and no
provider has a correct block-scaled FP8 GEMM on gfx1201. `hipblaslt_fp8` keeps refusing
`FP8_BLOCK`. The golden reference is made with `--act-quant none` (what is served).

User decision 2026-09-29: accepted as recommended: the memory saving of FP8 is given up for `fp8_block` (the device
holds BF16 weights). An own fused W8A16 kernel (FP8 read, dequantized in registers, WMMA) would keep
it and could beat BF16 at decode; it is allowed by the reuse rule (no provider works) but deferred.

## User review of the 2026-09-29 overnight decisions (2026-09-29)

**Decision (user, 2026-09-29, relayed by the coordinator):** every provisional decision of the unattended run is answered. Accepted as recommended: the GSM8K chain of thought; the W4A4 base-3B accuracy baseline with the Instruct result reported too; FP8 per-token activation scales; the FP8 KV golden against the emulated FP8 reference; the calibrated fake-quant tolerance for activation-quantized formats; the TP quantized gate on greedy tokens and likely candidates; TurboQuant's Gaussian `S` per (layer, head) and its 16-byte-padded layout; one page class per format in L0; the 1 h prune for finished agents; and every remaining minor item (YaRN details, FP8 KV page semantics, the e4m3 copy in `turbine-kv`, `KvCodec` details, TurboQuant rounding, seeds and NMSE bounds, FP8 layer selection and configured formats, the Task 25 status items, the MXFP4 act-order / Quark / NVFP4 items, the INT4 details, the fixture scripts). Changed:

1. `fp8_block`: **"Write own kernel in 6a"** — a fused block-scaled FP8 GEMM for gfx1201 as a registered implementation, with real memory savings, correctness tests against the CPU reference and the fp8_block proof; BF16 decode only as a logged fallback for shapes it does not support (spec and plan Task 15 amended).
2. W4A4 golden prompts: **"Find another checkpoint"** — an Instruct W4A4 MXFP4 checkpoint loadable by our packagings; else completion prompts only for the base checkpoint; do not quantize one ourselves without asking.
3. 6b ladder: **"Start at YELLOW earlier"** — compression begins at YELLOW, before the tiers are full, lowest tier first, one rung at a time (6b spec, plan and the `p6b-groundwork` policy and tests follow).

Housekeeping by the coordinator: `origin/main` pushed (ecbd043..38703b9); on novanas the kubelet eviction thresholds are 5 % (imagefs and nodefs), so the disk floor is ≈ 45 GB free; the lead's cleanup trigger is ≈ 100 GB free.

## W4A4 proof checkpoint: AMD Llama-3.1-8B-Instruct Quark W4A4 (2026-09-29)

After the user's "Find another checkpoint" answer the lead searched Hugging Face: the only Instruct W4A4 MXFP4 checkpoint of a family Turbine serves on amd is `amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ` @ `00b0d018950a5466fa1fc8bc0ccf174bd38b15da` (AMD Quark fp4 weights and activations, per_group 32, e8m0, half_even, `even`; GPTQ with desc_act + static_groups; SmoothQuant folded; `exclude: [lm_head]`; ungated, 5,826,947,776 bytes, downloaded to `/home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4-a4`, free disk 232.5 GB → 226.9 GB); the others were Qwen3 / MoE / gpt-oss (Phase 7), GGUF or MLX. Its recipe also quantizes the KV cache (`kv_cache_quant_config`: fp4 K/V projection outputs), which the Quark parser refused. Options: (a) accept it and ignore the KV quantization with a WARN `kv_cache_quant_ignored`, the KV at the configured `kv.dtype`, the golden reference built the same way (activation fake-quant, BF16 KV), accuracy baseline BF16 Llama-3.1-8B-Instruct; (b) refuse it and fall back to completion prompts on the base 3B checkpoint; (c) keep looking or wait.

**Decision (user, 2026-09-29, relayed by the coordinator): (a).** The base-3B checkpoint's accuracy is still reported against BF16 3B base as a side result.

Correction (6a lead, 2026-09-29, facts only; the decision stands): the checkpoint's KV recipe is `fp8_e4m3` per-tensor static K/V projection outputs (with `k_proj` / `v_proj` `output_scale` tensors), not fp4. It appears both in `kv_cache_quant_config` and as identical `layer_quant_config` entries for `*k_proj` / `*v_proj`; d1de280 ignores both, and the scale tensors load as `unexpected_tensor` WARNs.


## Phase 6a gate misses on GSM8K-200: FP8 KV (Llama, OLMoE) and MXFP4-A16 (8B) (2026-09-29)

Asked 2026-09-29 by the 6a lead after the lost agents' results were collected. GSM8K-200 (chain of thought, `final_number`):

- FP8 KV on Llama-3.2-3B-Instruct (Task 24): BF16 KV 0.805 (161/200), FP8 KV 0.790 (158/200); drop 0.015 > the plan's 0.01 bound. Throughput and the `kv_gpu` round trips pass.
- FP8 KV on OLMoE-1B-7B-0125-Instruct (Task 24): BF16 KV 0.655 (131/200), FP8 KV 0.615 (123/200); drop 0.040 > the plan's 0.01 bound — larger than Llama's.
- MXFP4-A16 on Llama-3.1-8B-Instruct (Task 20, `FabioTrindade/…-MXFP4A16`): BF16 8B 0.89 (178/200), MXFP4 0.835 (167/200); drop 0.055 > 0.04 (BF16 baseline because vLLM-ROCm refuses the checkpoint on gfx1201).

**1. FP8 KV, Llama and OLMoE.** Options: A) run the full GSM8K test split (1,319 items) at BF16 KV and FP8 KV with the same prompt wrapper and matcher, pass if the drop ≤ 0.01 (each model judged separately) — chosen; B) accept the 200-item result as noise; C) keep the FP8 KV row `experimental`. **Decision (user, 2026-09-29, relayed by the coordinator): A**, for Llama initially; **extended by the coordinator to OLMoE, 2026-09-29** (the OLMoE drop, found by the Task 24 wrap-up, is larger than Llama's and was not in the original ask). If OLMoE still misses the ≤ 0.01 bound on the full set, look for a numerics cause before reporting to the coordinator: compare Turbine against the emulated-FP8-KV reference and check the per-layer KV scales (OLMoE's are per layer, unlike Llama's) — do not just accept the drop.

**2. MXFP4-A16, 8B.** Options: A) run the full GSM8K on MXFP4-A16 and BF16 8B; if the drop is still > 0.04, do not blame the format yet — first look for a Turbine numerics error by comparing Turbine with the dequantized-checkpoint reference (logits / golden positions) and report to the coordinator before any support-status change — chosen; B) accept and document the drop; C) keep the row `experimental`. **Decision (user, 2026-09-29, relayed by the coordinator): A.**

Dataset: `openai/gsm8k`, config `main`, split `test`, downloaded on novanas with `hf download --repo-type dataset` at a pinned revision (token stays on the host), converted by a committed generator script into `tests/eval/gsm8k-full.jsonl` (MIT, 1,319 items, < 1 MB) with the GSM8K-200 wrapper and matcher. Full runs (~6× GSM8K-200 each) queue under the one-GPU-job rule, never in parallel; Llama and OLMoE runs (BF16 KV, FP8 KV each) queue one after another, not in parallel.

## Golden tolerance floor for quantized checkpoints (lead decision, test policy, 2026-09-29)

Asked by the Task 18 INT4 builder: a quantized checkpoint's golden tolerance is calibrated from transformers' own spread on the dequantized checkpoint (the OLMoE method); when that spread is tighter than the BF16 model's bounds, which applies? Options: A) the calibrated spread alone; B) max(calibrated spread, the BF16 model's bounds) — chosen. **Decision (6a lead, confirmed by the coordinator, 2026-09-29): B.** The golden check compares Turbine's kernels with transformers, so the kernel noise the BF16 bounds already accept applies to a quantized checkpoint as well; a bound tighter than BF16's would fail on noise already accepted. Applies to every Phase 6a weight format (AWQ, GPTQ, FP8, fp8_block, MXFP4). Test-policy detail, not a user decision.

Related (coordinator, same day): a proof may measure natively on novanas with a detached script instead of `lab-bench.sh` (ssh rules), provided its collector prints a `BENCH`-equivalent line and uploads to labbook as usual; the 10-minute soak runs on AWQ once its proof passes, under the one-GPU-job rule.

## 6b Task 4: planner copy bytes and the lossy chain rule (2026-09-30)

Asked by the 6b Task 4 builder (branch `p6b-t4`, ef69456), relayed by the lead.

1. `PlanInputs.copy_bytes`: transfer rates are measured per encoded byte, so pricing a tq4 copy at the full L0 block size made the planner never choose a lossy block. Options: A) the planner prices each transfer at the block's encoded (compressed) bytes, a new `PlanInputs.copy_bytes`, within Task 4; B) keep pricing at L0 bytes and observe estimates at the logical size instead. **User decision 2026-09-30: A — accepted.**
2. Lossy chain rule. Spec S-3 said both "a block keeps its key while an exact copy exists" and "a lossy copy is filed under `lossy_key`". The builder's design: while an exact copy exists, a lossy copy is one more location on the exact entry; only an L0 copy promoted from a lossy copy gets its own entry under `lossy_key(key, format, seed)`, and a request's later blocks chain from that key. Alternative: every lossy copy gets its own `lossy_key` entry. Either way exact and lossy lineages never alias. **User decision 2026-09-30: the builder's design — accepted; spec S-3 amended to match.**

Consequence (lead): 6b Task 15 (`p6b-t15`) observes transfer estimates at max(bytes, codec from/to bytes) to fix the same mispricing from the other side. Applied together the two corrections would count the saving twice. The t4 + t15 merge reconciles them to one consistent model under decision 1 (the planner prices at encoded bytes, and the estimate it multiplies must be a rate per encoded byte, however the simulator's copy times scale), and re-checks `ladder_under_pinned_pressure` and `cost_aware_beats_lru`.

## 6b ladder: when a tier's rung steps back up (2026-09-30)

Asked by the 6b Task 15 builder (`p6b-t15`, 940b871), relayed by the lead. Spec S-6 made step-up depend on fill only (below `kv.ladder.low_water` 0.85 for `reliability.pressure.deescalate_dwell`), while compression runs whenever the controller is not GREEN ("Start at YELLOW earlier"). So under sustained ORANGE a floor tier below low water stepped up after the dwell and was compressed again on the next sweep, once per dwell.

- A) Step-up also requires GREEN: the rung relaxes only once the controller is GREEN and the tier has stayed below 0.85 for the dwell. No churn under sustained pressure; blocks stay compressed a little longer after pressure eases. (Lead recommendation.)
- B) Keep fill-only (the spec as it was): the rung may relax under YELLOW / ORANGE; churn about once per dwell under sustained pressure.
- C) Step-up requires YELLOW or better: a middle ground; still churns at YELLOW, where compression is active.

**User decision 2026-09-30: A.** Spec S-6 and its AC amended; the t4 + t15 merge updates the ladder code and the pinned expected rung fixture of `ladder_under_pinned_pressure`, and adds an assertion that no step-up happens while not GREEN.

