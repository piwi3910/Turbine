# phase-8-expansion

Status: complete

Source: `turbine-spec.md` §19 Phase 8 (expansion: "AMD/ROCm, Intel where viable, additional model families, quantization, speculative decoding and multimodal as separately scoped work"), with TS §2 (non-goals), TS §3 (V1 scope), TS §6 (runtime, tensors and kernels), TS §8 (lossy KV transforms need quality validation), TS §16 (security), TS §17 (testing), TS §18 (benchmarking), TS §20 (definition of done) and TS §21 (engineering rules). Sections of that document are cited as "TS §N". Decisions are recorded in `.procoder/ask/decisions.md` ("Answers log for phase 1–8 spec questions (2026-09-25)"). Earlier phase specs are cited by name (phase-1-single-request … phase-7-advanced-distribution).

This is the **umbrella spec** for Phase 8. It fixes the track list, the track order and the rules every track shares (support matrix, exit gate, vendor neutrality, quality evaluation). Each track gets its own spec, written when the track starts; this spec does not contain them:

1. `phase-8a-quantization`
2. `phase-8b-speculative-decoding`
3. `phase-8c-model-families`

Amendments to TS §19 Phase 8: **AMD/ROCm is not a Phase 8 track** — AMD execution moved to Phase 1 (Phases 1–2 run on the R9700 cards first, NVIDIA follows behind the same vendor-neutral kernel traits). **Multimodal is out** of the roadmap. **Intel** has no track and no code until an Intel GPU exists in the lab.

## Problem

By the end of Phase 7 Turbine serves two BF16 models — `meta-llama/Llama-3.2-3B-Instruct` (dense) and `allenai/OLMoE-1B-7B-0125-Instruct` (MoE) — on AMD R9700 (`gfx1201`) and NVIDIA GB10 (`sm_121`), text only, one decoded token per forward step per sequence, with BF16 weights and BF16 KV only. Three gaps remain. Every larger checkpoint the lab actually has on disk is quantized (modelopt NVFP4/FP8 mixed precision with FP8 KV, and compressed-tensors NVFP4), so quantization is the entry ticket to the models people run, and each format must be proven against a reference before it is trusted (TS §8, §21 rule 1). Decode is memory-bandwidth bound on both vendors, so without speculative decoding Turbine leaves a large ITL gap compared with engines that draft. And two architectures are not enough for real use: the families people deploy — Qwen3 dense and MoE, the Qwen3.5/3.6 hybrids with Gated DeltaNet linear attention, Mistral and Mixtral — each need their own config parsing, weight mapping and layer execution. TS §19 lists these as "separately scoped work": the tracks share little code and have different dependencies, so they are delivered as separate specs in a fixed order, and they must share one honest mechanism for saying which device × model × format × method combinations are supported (TS §21 rule 4).

## Users

- **Operators:** need to know, before deploying, whether a given device × model architecture × weight format × KV format × speculative method combination is supported, and to have Turbine refuse an unsupported one at startup with a reason instead of failing mid-request.
- **Clients of the OpenAI API:** need lower ITL from speculative decoding with no change in output distribution, and access to more model families and quantized checkpoints behind the same API.
- **Turbine developers (humans and AI agents):** need each track to be buildable and testable on the macOS workstation where no GPU is involved, a lab procedure per track on the host with the right hardware, and a fixed contract (this spec) each track spec must satisfy so no track re-decides the shared rules.
- **Benchmark runners:** need per-track comparisons against a reference engine on the identical checkpoint, hardware and request profile (TS §18), including quality metrics for lossy formats.

## In scope

- [S-1] **Track list and order:** three tracks, delivered strictly in this order, each by its own spec: (1) `phase-8a-quantization`, (2) `phase-8b-speculative-decoding`, (3) `phase-8c-model-families`. A track spec is written when its track starts, cites this spec for S-2 … S-5 instead of restating them, and must reach `Status: complete` before implementation of that track begins. A later track starts only when the earlier track's exit gate (S-3) has passed for at least one support-matrix row.
- [S-2] **Support matrix (shared):** a single declarative table in `turbine-core` of `(vendor, arch, architecture, weight_format, kv_format, speculative_method) → supported | experimental | unsupported{reason}`; `turbine-server --support-matrix` prints it; startup (before binding) resolves the configured combination against it and exits 1 on `unsupported` naming the combination and reason; `experimental` starts with a WARN. `GET /turbine/v1/status` reports the resolved row. The table starts with the Phase 1–7 combinations (both vendors, Llama-3.2-3B and OLMoE, BF16 weights, BF16 KV, no speculation) as `supported`; each track adds its rows only when its acceptance criteria pass.
- [S-3] **Per-track exit gate (shared):** a track's rows enter the support matrix as `supported` only when (1) its golden comparison passes the phase-1 tolerance (≥ 14 of 16 prompts with the first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats) against committed HF transformers reference fixtures for BF16 models, or — for quantized checkpoints, where a BF16 transformers run is not the same arithmetic — against the same checkpoint's reference output captured once with the checkpoint's reference runtime and committed; (2) `turbine-bench` on the same hardware and profile is recorded against a reference engine where one runs on that device, else as a Turbine baseline (TS §18); (3) `scripts/overload-soak.sh` (Phase 3) still passes on the track's lab host with the track's feature enabled; and (4) for a lossy format, `turbine-golden eval` accuracy on the committed task set is within _quality.max_accuracy_drop_ of the same model in BF16 (or of the reference engine on the same checkpoint). The TS §20 first-useful-release items must not regress on either vendor. Rows that were not validated on a vendor stay `unsupported{reason}` for that vendor.
- [S-4] **Quality evaluation tool (shared):** `turbine-golden eval` (Interfaces) and the committed task set `tests/eval/gsm8k-200.jsonl`, used by every lossy-format gate.
- [S-5] **Vendor neutrality and Intel re-entry (shared):** no public type or function signature in `turbine-kernels`, `turbine-tensor`, `turbine-scheduler`, `turbine-kv` or `turbine-reliability` names a CUDA-, HIP- or other vendor-specific type, so every track's kernels go behind the Phase 1 vendor-neutral kernel traits and a future Intel provider can be added without touching core logic (TS §21 rule 11). Intel work re-enters the roadmap only when an Intel GPU is installed in the lab; until then no Intel code is written and Intel devices are discovered nowhere (logged, no crash).
- [S-6] **Track 1 — quantization (`phase-8a-quantization`), fixed scope:** exactly the formats the cached checkpoints use: modelopt NVFP4 + FP8 mixed precision (per-layer resolution from _hf_quant_config.json_, including `MIXED_PRECISION` with FP8 projections, W4A16 NVFP4 group-16 experts and BF16 excluded modules), compressed-tensors NVFP4 (`nvfp4-pack-quantized`, from _quantization_config_ in _config.json_, divisor global-scale convention) and FP8 e4m3 KV cache as an explicit _kv.dtype_ choice. Because every cached checkpoint in these formats is a Qwen3.5/3.6-style architecture that arrives only in track 3, this track proves each format on a checkpoint in that same format of an architecture already registered (Llama-3.2 or OLMoE); the cached checkpoints are served once track 3 registers their architectures. The track spec decides which such checkpoints, the per-vendor kernel providers, and how NVFP4 runs on RDNA4 (hardware FP8, no FP4 matrix path, so NVFP4 there can only be weight-only).
- [S-7] **Track 2 — speculative decoding (`phase-8b-speculative-decoding`), fixed scope:** a separate small draft model of the same family as the target, sharing its tokenizer (first pairing: `meta-llama/Llama-3.2-1B-Instruct` drafting for `meta-llama/Llama-3.2-3B-Instruct`), behind a `Proposer` trait; a verifier in `turbine-scheduler` that scores k proposals in one target forward and accepts with standard speculative rejection sampling (greedy: exact prefix match); rollback by truncating KV blocks past the last accepted position; per-request disable below _speculative.min_acceptance_ and global disable at pressure ORANGE or worse (Phase 3). MTP heads, EAGLE heads and DFlash-style drafters are not in this track. Rollback of linear-attention recurrent state (hybrid targets) is added by track 3 together with the hybrid families.
- [S-8] **Track 3 — model families (`phase-8c-model-families`), fixed scope:** an architecture registry in `turbine-model` keyed on the checkpoint's `architectures[0]` and `model_type` (top level and nested _text_config_), each entry providing config parsing, weight-name mapping, layer execution and KV/state layout; an unregistered architecture is exit 1 naming it. Families: Qwen3 dense and Qwen3 MoE; the Qwen3.5/3.6 hybrids (Gated DeltaNet linear attention + full attention, including the cached `Qwen3.6-35B-A3B` checkpoint, text only with vision weights skipped), which also bring TKV1 recurrent/conv-state segments (phase-7) and speculative state rollback (S-7); Mistral and Mixtral. gpt-oss is out (decision 2026-09-25: its weights ship only in MXFP4, which is out of the quantization scope). The linear-attention kernel provider for each vendor is decided in the track spec. Each family gets its own golden fixtures in the phase-1 format.

## Out of scope

- AMD/ROCm execution as a Phase 8 track (Phase 1 delivers it; amends TS §19).
- Multimodal input of any kind (image, video, audio); checkpoints with a vision tower are served text-only with vision weights skipped, and image content parts keep the phase-1 `400 unsupported_parameter` behaviour (amends TS §19).
- Intel execution, discovery or build support until an Intel GPU exists in the lab (S-5); Apple/Metal; CPU serving beyond the phase-1 reference provider.
- Quantization formats beyond S-6: FP8 block-scaled weights, INT4 GPTQ/AWQ, MXFP4, NVFP4 KV cache, GGUF. A family whose only checkpoints use such a format runs from a checkpoint in an in-scope format, or the format is added by a new recorded decision first.
- gpt-oss (dropped by decision 2026-09-25: MXFP4-only weights).
- Speculative methods other than a separate draft model (S-7).
- Training, fine-tuning, quantization-aware training, calibration or producing quantized checkpoints; Turbine only loads checkpoints quantized elsewhere (TS §2).
- Custom GPU kernels without a profile showing no provider meets the need (TS §2); each such kernel is its own decision recorded in an ADR.
- Formats that are not safetensors (GGUF, PyTorch pickle `.bin`/`.pt`): never deserialized (TS §16).
- Speculative decoding combined with Phase 7 PD or PP (validated colocated only).
- Hot-swapping models or serving several models in one process (a draft model is part of the target's deployment, not a second served model).
- Writing the three track specs; this spec only names them and fixes their shared contract.

## Constraints

- Rust only in the serving path; no Python at runtime (TS §21 rule 4). Python is allowed only at fixture-generation time (HF transformers dumps, reference-runtime captures), never in a build or test that `cargo test` runs. Vendor libraries are loaded at runtime through the prebuilt kernel libraries (_libturbine_hip.so_, _libturbine_cuda.so_, CMake-built), never linked by Cargo, so the workspace still builds and every non-ignored test passes on macOS arm64 with no GPU libraries.
- `unsafe` stays in four locations: `turbine-device`, `turbine-kernels`, `turbine-distributed/src/collective/ffi` (Phase 5) and `turbine-transport/src/rdma` (Phase 7) (CONFLICT C-19).
- The kernel C ABI is shared by both vendors' libraries; any additive change bumps `turbine_abi_version` for both together.
- Every lossy format or lossy KV transform ships with its quality gate (TS §8, S-3); nothing lossy is enabled by default unless the checkpoint itself is stored in that format.
- Bounded inputs (TS §16, §21 rule 8): speculative k ≤ 8; draft-model memory is reserved through the Phase 3 budget before speculation is enabled.
- Model weights live in `/home/piwi/turbine-models/<slug>` on each host, downloaded by Claude with the user's HF token at that time (never stored in the repo); tests read `TURBINE_TEST_MODEL_DIR` and never download.
- **Host workloads:** any lab run that needs production workloads (production vLLM on the Sparks, anything using the R9700 cards on `novanas`) moved or memory freed on any host is started only after the implementer has asked the user and the user has moved the workloads; the implementer never stops, moves or reconfigures production workloads itself. Reference-engine captures for a checkpoint other than the one production serves use a temporary container named `turbine-ref-*` within the free memory measured at run start, removed afterwards.
- Lab hosts: `novanas` (192.168.10.203), 2× Radeon AI PRO R9700 (`gfx1201`, 32 GB each), ROCm 7.14.1 at `/opt/rocm/rocm`, k3s Jobs requesting `amd.com/gpu`, 10 GbE; `dgx-spark` (192.168.10.246) and `dgx-spark2` (192.168.10.245), 1× GB10 (`sm_121`) each, ~121 GB unified memory, Docker, RoCE between them. Families that do not fit one device (e.g. Mixtral in BF16) rely on Phase 5 TP or Phase 7 PP/EP, or on an in-scope quantized checkpoint; the track 3 spec states which per family.
- Cached checkpoints that define the quantization scope (from their _config.json_, verified 2026-09-25): `nvidia/Qwen3.6-35B-A3B-NVFP4` (`qwen3_5_moe`, modelopt mixed precision, FP8 KV); `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (`qwen3_5` dense hybrid, modelopt); `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`qwen4_exp`, 126 GB — does not fit one Spark); `YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4` (compressed-tensors `nvfp4-pack-quantized`).

## Interfaces

### `turbine-server`

```
turbine-server --support-matrix [--output text|json]
turbine-server --config <path> [--set <dotted.key>=<yaml value>]... [--check-config]
```

- `--support-matrix` prints the S-2 table and exits 0 without reading a config. JSON: `{"rows":[{"vendor":"amd","arch":"gfx1201","architecture":"LlamaForCausalLM","weight_format":"bf16","kv_format":"bf16","speculative":"none","status":"supported","reason":null}]}`; `*` is allowed as a wildcard in any column, most specific row wins.
- `--check-config` additionally resolves the support-matrix row; `unsupported` prints the reason and exits 2 (configuration error), before model weights are read.
- Bounded column values: `vendor` ∈ {`amd`, `nvidia`}; `weight_format` ∈ {`bf16`, `modelopt_nvfp4`, `modelopt_fp8`, `modelopt_mixed`, `ct_nvfp4`}; `kv_format` ∈ {`bf16`, `fp8_e4m3`}; `speculative` ∈ {`none`, `draft`}. `arch` and `architecture` are free strings from the device inventory and _config.json_.

### Configuration additions (umbrella-owned)

| Key                         | Type  | Default | Validation                                                      |
| --------------------------- | ----- | ------- | --------------------------------------------------------------- |
| _quality.max_accuracy_drop_ | float | `0.01`  | 0 ≤ value ≤ 0.1; used only by `turbine-golden eval` comparisons |

Track-owned keys are defined in the track specs; their names are reserved here so tracks do not collide: _kv.dtype_ (track 1; adds `fp8_e4m3`), _speculative.method_, _speculative.num_tokens_, _speculative.draft_model_path_, _speculative.min_acceptance_ (track 2).

### `turbine-golden` addition

```
turbine-golden eval --url <base> --tasks <tasks.jsonl> [--model <name>] [--output text|json]
turbine-golden eval-compare --baseline <report.json> --candidate <report.json> [--max-drop <float>]
```

- The tasks file (JSONL): one `{"id","prompt"|"messages","answer","match":"exact"|"number"}` per line; greedy, `max_tokens` from the file. `eval` reports accuracy and per-task correctness; exit 0 on a completed run, 2 on usage/I/O errors.
- `eval-compare` exits 0 when candidate accuracy ≥ baseline accuracy − max drop (default _quality.max_accuracy_drop_), 1 otherwise, printing both accuracies.
- A committed task set `tests/eval/gsm8k-200.jsonl` (200 GSM8K test items, numeric match) is used for every lossy-format gate.

### Metrics (added)

- `turbine_support_matrix_status{status}` gauge — 1 for the resolved status of the running configuration.
- Track metrics (e.g. speculative acceptance) are defined in the track specs with bounded label sets.

### Lab

- Each track spec names its lab host(s) and runs its GPU tests through `scripts/lab-test.sh <host>` (`novanas` for AMD, `dgx-spark` for NVIDIA) and its server configs under `scripts/lab/phase8-<track>-<host>.yaml` on port 18000.

## Data

- **Support matrix:** compiled into `turbine-core` as a static table (source of truth in code, printed by `--support-matrix`); no runtime file.
- **Golden fixtures** per architecture and format under `tests/golden/<model-slug>/` in the phase-1 format (`reference.jsonl`, `tolerance.json`).
- **Eval set:** `tests/eval/gsm8k-200.jsonl` (MIT-licensed source, attribution in `tests/eval/NOTICE`), and one `tests/eval/<model-slug>/<engine>.json` report per gated combination.
- **Track specs:** `.procoder/specs/phase-8a-quantization.md`, `.procoder/specs/phase-8b-speculative-decoding.md`, `.procoder/specs/phase-8c-model-families.md`, created when each track starts.

## Edge cases

- A configuration that matches two wildcard rows with equal specificity (must be impossible: the table is validated at build time by a unit test).
- A configured combination with no matching row at all: treated as `unsupported{reason: "no support-matrix row"}`.
- A track rolled back after release (its row moves from `supported` to `experimental` or `unsupported`): existing configs fail at next startup with the new reason, never mid-request.
- A checkpoint with a vision tower (e.g. Qwen3.6-35B-A3B): served text-only; vision weights skipped, not loaded; image parts rejected as in phase-1.
- An Intel GPU present on some host: discovered nowhere, logged once, no crash.
- Eval answers with thousands separators, trailing periods or units (numeric match tolerates commas and a trailing period only).
- A track whose gate passes on one vendor only: the other vendor's rows stay `unsupported{reason}`; the track may close with that recorded.

## Failure modes

- **Unsupported combination (support matrix):** exit 2 with `--check-config`, exit 1 at startup, before weights are read, naming the combination and the reason.
- **Experimental combination:** starts with a WARN naming the row; `turbine_support_matrix_status{status="experimental"}` is 1.
- **Quality gate fails for a format:** the row stays `experimental` (or `unsupported` if outputs are wrong rather than merely degraded); the track does not close.
- **Eval run fails midway (server error, timeout):** `turbine-golden eval` exits 2 with the failing task id; no partial report is written as if complete.
- **A track spec cannot satisfy this contract** (e.g. needs a format or method outside S-6 … S-8): the track stops and the question goes to the user as a new decision; the umbrella is amended, never silently widened by a track.

## Acceptance criteria

- [ ] [S-2] `cargo test -p turbine-core support::tests::resolution_and_refusal` exits 0; it asserts the most specific row wins over wildcards, an `unsupported` row makes config validation fail with the reason, an `experimental` row passes with a WARN recorded, a combination with no row is unsupported, no two rows tie in specificity, and every row's enum values are from the documented bounded sets; fails if a wildcard shadows a specific row or an unsupported combination validates.
- [ ] [S-2] `cargo test -p turbine-core support::tests::baseline_rows_present` exits 0; it asserts the table contains `supported` rows for `amd`/`gfx1201` and `nvidia`/`sm_121` × `LlamaForCausalLM` and `OlmoeForCausalLM` × `bf16` weights × `bf16` KV × `none`, and no `supported` row for any quantized format, `fp8_e4m3` KV or `draft` before the tracks add them; fails if a row is marked supported without its track.
- [ ] [S-2] `cargo test -p turbine-server --test server_cli support_matrix_output` exits 0; it runs `turbine-server --support-matrix --output json` and asserts exit 0 and a JSON body with a non-empty `rows` array, and runs `--check-config` with `speculative.method: draft` on a build where `draft` is unsupported and asserts exit 2 naming _speculative.method_; fails if an unsupported combination starts.
- [ ] [S-2] `cargo test -p turbine-api --test api status_reports_support_row` exits 0; it starts the server on the CPU reference provider and asserts `GET /turbine/v1/status` includes the resolved row and `/metrics` shows `turbine_support_matrix_status` at 1 for exactly one status; fails if the row or gauge is missing.
- [ ] [S-3] [S-4] `cargo test -p turbine-bench --test golden eval_accuracy_report` exits 0; it runs `turbine-golden eval` against an in-test mock that answers 150 of 200 items correctly (numeric match tolerating commas and trailing periods) and asserts accuracy 0.75 in the JSON report, then asserts `turbine-golden eval-compare` exits 0 for a candidate at 0.745 against a 0.75 baseline with max drop 0.01 and exits 1 for 0.73; fails if matching is wrong, the report shape changes, or the gate passes a drop larger than allowed.
- [ ] [S-4] `cargo test -p turbine-bench --test golden eval_task_set_valid` exits 0; it asserts `tests/eval/gsm8k-200.jsonl` has exactly 200 lines, unique ids, a numeric `answer` for every `match: "number"` item, and that `tests/eval/NOTICE` exists; fails if the committed task set is malformed.
- [ ] [S-5] `cargo test -p turbine-kernels --test vendor_neutral_api` exits 0; it parses each listed crate's `src` with `syn` and walks the signatures of every `pub` item reachable from the crate root in `turbine-kernels`, `turbine-tensor`, `turbine-scheduler`, `turbine-kv` and `turbine-reliability`, and asserts no path contains `cuda`, `hip`, `rocm`, `nccl`, `rccl`, `cublas`, `sycl` or `level_zero` outside the backend-enum variant names; fails if a vendor type leaks into a core public signature.
- [ ] [S-1] [S-6] Before track 1 implementation starts: `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8a-quantization` exits 0 reporting COMPLETE, the file's Status line reads `complete`, and `grep -c -E "modelopt|compressed-tensors|fp8_e4m3" .procoder/specs/phase-8a-quantization.md` is non-zero while the spec lists no format outside S-6; fails if track 1 starts on an incomplete spec or its scope differs from S-6.
- [ ] [S-1] [S-7] Before track 2 implementation starts (and only after track 1 closed): `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8b-speculative-decoding` exits 0 reporting COMPLETE with Status `complete`, and the spec names `Llama-3.2-1B-Instruct` as the first draft model and no MTP, EAGLE or DFlash proposer; fails if track 2 starts early, on an incomplete spec, or with a method outside S-7.
- [ ] [S-1] [S-8] Before track 3 implementation starts (and only after track 2 closed): `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8c-model-families` exits 0 reporting COMPLETE with Status `complete`, and the spec covers Qwen3 dense, Qwen3 MoE, the Qwen3.5/3.6 hybrids with its linear-attention kernel provider decided per vendor, Mistral and Mixtral; fails if track 3 starts early, on an incomplete spec, or a family from S-8 is missing.
- [ ] [S-3] [S-6] [S-7] [S-8] Manual track close, per track in S-1 order: `scripts/overload-soak.sh <host>` exits 0 with the track's feature enabled, `turbine-golden compare --url <turbine> --reference tests/golden/<model-slug>/reference.jsonl` exits 0 under that slug's tolerance file, `turbine-bench --url <turbine> --output json` exits 0 with its report committed, for lossy formats `turbine-golden eval-compare --baseline <bf16.json> --candidate <quantized.json>` exits 0, and then `turbine-server --support-matrix --output json` exits 0 showing the track's new rows as `supported` only for the vendors where all four passed; every output pasted into the task evidence; fails if a row turns `supported` without all four gate items or a track closes out of order.

## Open questions

<!-- None: decisions recorded in .procoder/ask/decisions.md -->
