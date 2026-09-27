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
