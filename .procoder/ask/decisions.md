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
