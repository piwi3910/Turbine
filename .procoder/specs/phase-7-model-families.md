# phase-7-model-families

Status: complete

Source: the umbrella `.procoder/specs/phase-6-8-expansion.md` S-1 … S-5 (cited, not restated) and S-8 (this track's fixed
scope), line 66 inputs (a)–(d); `turbine-spec.md` §5 (model support), §9 (kernels), §21 (engineering rules). Decisions:
`.procoder/ask/decisions.md` "Phase 7 spec: order, exit bar, checkpoints, disk, GDN kernel, carry-over (2026-10-07)" and
its follow-up rounds (disk, Harmony, hybrid prefix state, no-vLLM fallback, reasoning output, MoE over two cards,
`mxfp4_a4`). Research inputs: the lead's briefing of 2026-10-07 (checkpoint sizes, kernel providers, code state).

Amendments to earlier specs made by this phase:

- Amends Phase 2m (model-family registry): a family declares its per-layer kinds (full attention, sliding-window
  attention, linear attention) and its per-sequence state layout next to the paged KV (S-1).
- Amends Phase 4 / Phase 5p (prefix reuse): a full block of a hybrid family carries the recurrent and conv state at its
  end boundary; reuse restores KV and state together (S-12). Partial-block reuse stays L0-only (5p rule).
- Amends Phase 6a S-12 (Q7, "track 1 quantizes dense linear layers only"): quantized MoE expert weights land here and
  the `quant_moe_phase7` refusal is removed for every combination this phase supports (S-4).
- Amends Phase 5 exit (`olmoe_ep_tp_drift`): the refusal is removed once S-3 fixes or explains the drift and the
  ep2×tp2 golden passes.

## Problem

Turbine serves only Llama (dense) and OLMoE (MoE) on AMD. Qwen3, Qwen3-MoE, Mistral and Mixtral run on the CPU
backend only (`FAMILY_REASON`, `unsupported` on `amd`), and gpt-oss and the Qwen3.5/3.6 hybrids are not registered at
all (exit 1). Users of the engine want the current open-weight families — the ones vLLM and llama.cpp already serve —
on the same reliability-first runtime, including quantized MoE checkpoints that only fit over the two R9700s. Phase 5p
brought batch-1 decode to ~0.90× of the 640 GB/s roofline, so the base the families build on is now efficient; Phase 8
(speculative decoding) needs these families (Qwen3.5/3.6 MTP heads, draft models) as its targets.

## Users

- **API clients:** need each family served through the OpenAI routes with its native chat template, tool calls in
  `tool_calls`, reasoning text in `reasoning_content` (streaming too), and outputs that match transformers within the
  golden tolerance.
- **Operators:** need a support-matrix row per family/format/placement that says `supported` only when its gates
  passed, a documented lab config per family (one GPU, TP2 or EP2), refusals with reason codes for everything else, and
  enough disk on novanas to hold the checkpoints.
- **Turbine developers:** need one module per family behind the registry and its conformance suite, the
  `docs/extending/` model-family page extended with layer kinds and state layout, kernel-provider evaluations recorded
  in decisions.md, and per-family golden/eval fixtures in the Phase 1 format.
- **Phase 8:** needs the hybrid state layout (snapshots at block boundaries) that speculative rollback will extend, and
  the families as speculation targets.

## In scope

Order of delivery (decision 1 A): S-1 and S-17 first; S-2 Qwen3 dense; S-3 the OLMoE drift investigation and S-4
quantized experts; S-5 Qwen3-MoE; S-6 Mistral; S-7 Mixtral; S-8/S-9 gpt-oss-20b; S-11/S-12 the hybrids. The carry-over
items S-14 … S-16 fit where their prerequisites land (S-14 with gpt-oss's MXFP4 work). Each family's row closes on its
own gate (S-13).

- [S-1] **Family layer kinds and state layout.** `ModelFamily` (`crates/turbine-model/src/families/mod.rs`) gains a
  per-layer kind list (`FullAttention`, `SlidingWindow { window }`, `LinearAttention`) and a `StateLayout` (bytes per
  sequence of recurrent and conv state per linear-attention layer, dtype); config parsing reads nested `text_config`
  (`architectures[0]` / `model_type` as today). The executor builder, KV sizing (`kv_layout`), memory budget and the
  scheduler's worst-case reservation account for both. The existing sliding-window refusal stays for every checkpoint
  whose window pattern this phase does not implement (only gpt-oss's alternating 128/full pattern is implemented).
- [S-2] **Qwen3 dense on AMD**: `Qwen/Qwen3-4B-Instruct-2507` BF16 (revision pinned at download), one GPU. HIP path for
  per-head QK-norm (`QK_NORM_PER_HEAD`), `tp_decoder_spec` for Qwen3 so TP2 is allowed, golden fixtures
  `tests/golden/qwen3-4b-instruct-2507/`, GSM8K pair against vLLM-ROCm.
- [S-3] **OLMoE expert × tensor parallel drift** (carry-over C1): a per-position routing trace (top-k ids, gate
  values, margin to the k+1-th) for one-GPU vs ep2×tp2 on p08 and p10 tests the near-tie hypothesis; fix the cause
  (e.g. route on a rank-invariant reduction) or explain it with evidence; remove the `olmoe_ep_tp_drift` refusal from
  `PARALLEL_REFUSALS` once ep2×tp2 passes golden c1 and c16 against the transformers reference. Done before any MoE
  family relies on TP2 (S-5, S-7, S-11).
- [S-4] **Quantized MoE experts** (C2): an evaluated, registered provider per expert format — FP8 block-128
  (`hf_fp8`, Qwen FP8), FP8 per-tensor static W8A8 (`ct_fp8`, RedHatAI Mixtral) and MXFP4 weight-only
  (`openai_mxfp4`, gpt-oss) — under the reuse-first rule (candidates: llama.cpp `mul_mat_id` MXFP4 mmq/mmvq, CK /
  hipBLASLt grouped GEMM, our own `qgemm_fp8_*` extended to grouped form only if those fail or are slower; evaluation in
  decisions.md). The expert ABI gains bias and activation variants (gpt-oss's clamped `x·σ(1.702x)` with `(up+1)` and
  interleaved gate/up) and a shared expert with a sigmoid gate (Qwen3.5/3.6 MoE). `quant_moe_phase7` is removed for
  every combination with a `supported` or `experimental` row.
- [S-5] **Qwen3 MoE** (`Qwen3MoeForCausalLM`): `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` over both cards, TP2 and EP2 both gated (decision: MoE over
  two cards A); `norm_topk_prob` routing; the faster placement is the documented lab config.
- [S-6] **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3` BF16 (HF shards only, not the consolidated single-file copy), one
  GPU, `mistral` tool format; TP2 allowed through a `tp_decoder_spec`.
- [S-7] **Mixtral**: `RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8` (per-tensor static W8A8 experts) over both cards, TP2 and
  EP2 both gated; the KV headroom per card (~6–7 GB) is reported by `/turbine/v1/kv` and its lab config sizes the scheduler's running-request cap to fit.
- [S-8] **gpt-oss-20b**: `openai/gpt-oss-20b` (root `model-0000*-of-00002` shards only), one GPU. New: learned per-head
  attention sinks, alternating sliding-window (128) / full attention layers, head_dim 64, attention biases, router
  top-k on logits then softmax over the k with router bias, MXFP4 experts with biases and the clamped activation (S-4),
  YaRN (landed in 6a). CK `fmha_fwd` / `pagedkv` / `splitkv` instances for d64 with `sink_ptr` and the local mask (CK
  carries both at the pinned commit; the instances are new); the `turbine_attention_paged_desc` gains window and sink
  fields (kernel ABI minor bump). The `openai_mxfp4` packaging row turns `supported` with this family.
- [S-9] **Harmony** (decision: Harmony A): a registered Harmony chat template and tool-call format; the analysis channel
  goes to `reasoning_content`, the commentary channel's tool calls to `tool_calls`, the final channel to `content`;
  `reasoning_effort` (low / medium / high) from the request reaches the system message; channel markers split across
  tokens stream correctly.
- [S-10] **Reasoning parsers** (decision: reasoning A): a `reasoning_parser` registry (one file per parser, static
  registry, conformance suite) with `qwen_think` (`<think>…</think>`) and `harmony`; each family names its default;
  responses carry the new reasoning_content field in the message and in streaming deltas; `chat_template_kwargs`
  (`enable_thinking`) passes through to the template.
- [S-11] **Qwen3.5/3.6 hybrids**: `Qwen/Qwen3.5-9B` BF16 one GPU and `Qwen/Qwen3.6-35B-A3B-FP8` over both cards (TP2 and
  EP2 gated). Text only: the vision tower and the `mtp.*` layer are skipped at load (logged). Layers: Gated DeltaNet
  linear attention (causal conv1d k=4, recurrent state per head in F32) interleaved 3:1 with full attention (head_dim
  256, attention output gate, partial rotary 0.25, 1-D RoPE for text), the (1+w) RMSNorm form, MoE with a shared expert
  (S-4). The AMD linear-attention kernel provider (HIP, decision GDN A): evaluate llama.cpp `ggml_gated_delta_net` + `ssm_conv` (decode, short
  prefill) and AMD AITER's gfx12 `chunk_gated_delta_rule_fwd_h` (long prefill); write our own only where both fail or
  are measurably slower; record it in decisions.md. CK d256 attention instances.
- [S-12] **Hybrid state and prefix reuse** (decision: hybrid cache A): per-sequence state slots sized by `StateLayout`,
  allocated with the sequence, released on finish/cancel/preemption; the state is snapshotted at every `kv.hybrid_snapshot_interval_blocks`-th full-block boundary (default 8, i.e. 1024 tokens at 128-token blocks) and at the last full block of every finished sequence (decision 2026-10-08 1 B); a snapshot is stored with its block in L0 (in a snapshot pool charged to the budget) and travels with it through L1/L2 (demotion, promotion, eviction); a prefix hit restores KV up to the match and the nearest snapshot at or before it, and recomputes the tokens after that snapshot; partial-block reuse stays L0-only and does not
  apply to hybrids' state (state is restored only at block boundaries, the remainder is recomputed); `n` > 1 forks copy
  the state; preemption by recompute rebuilds it.
- [S-13] **Per-family gate and support rows** (decisions: bar A, no-vLLM A): a row turns `supported` when golden c1 and
  c16 pass against a transformers reference (`tests/golden/<slug>/`, Phase 1 format, tolerance per the Phase 1 method;
  TP2/EP2 rows with `--batched-bounds`), GSM8K against vLLM-ROCm on the same checkpoint holds ≤ 0.01 (quantized rows the
  6a per-format drop of `tests/eval/<slug>/gate.json`) at concurrency 16, and c16 throughput is ≥ 0.75× vLLM-ROCm on the
  same placement. When the pinned rocm/vllm tag cannot serve a family on gfx1201, a newer tag is tried; if none can, the
  row needs golden and GSM8K only (against the BF16/dequantized transformers run or the 6a per-format rule) and the
  throughput is logged with the reason. Rows that do not pass stay `experimental` with the evidence in perf-log.
- [S-14] **MXFP4 carry-over** (C3, C4): trace Turbine's CPU MXFP4 path against transformers op by op on p05 and fix or
  explain the position-5 miss; the `mxfp4` Llama row turns `supported` if golden then passes. `mxfp4_a4` (decision B):
  `supported` when GSM8K holds the 4-bit drop (≤ 0.04) and golden passes a tolerance derived from transformers' own
  self-spread (`scripts/golden/self_spread.py`, method as OLMoE's).
- [S-15] **OLMoE FP8 KV** (C5): calibrated V scales for scale-less checkpoints (`crates/turbine-model/src/kv_scales.rs`), then golden c1/c16;
  the row returns to `supported` on a pass.
- [S-16] **L0 `tq4`** (C6): re-run the Phase 6b Task 13 gates for `kv.dtype: tq4` on Llama and OLMoE on the current
  tree; flip a row to `supported` only when all of its gates pass, else record the result and keep `experimental`.
- [S-17] **Checkpoint provisioning and disk** (decision: disk, three rounds): each checkpoint downloaded once on
  novanas under `/home/piwi/turbine-models/<slug>` at a pinned revision (`hf download … --revision <rev>`, token never
  read or passed), only when its family starts; the build caches (`turbine-ci/remote`, `turbine-ci/cache/slots`) live
  on the 10 TB iSCSI volume `/mnt/turbine-iscsi` bind-mounted back; dequantized BF16 copies and transformers references
  above ~60 GB RAM are built on cpuhost (CPU fixture rules).
- [S-18] **Tool-call formats per family**: `hermes` for Qwen3 / Qwen3-MoE; Qwen3.5/3.6's template format verified
  against its chat template and a new registered parser added if it differs (e.g. the XML `<function=…>` form);
  `mistral` for Mistral / Mixtral; Harmony for gpt-oss (S-9). Each family's default is set in its module.
- [S-19] **Housekeeping** (C8): the stale "Phase 8 track" wording in `crates/` is corrected, and tests that used
  `GptOssForCausalLM` as the example of an unregistered architecture use another name.
- [S-20] **Docs and contract**: `docs/extending/model-family.md` (layer kinds, state layout, reasoning parser),
  `docs/extending/` index, `.procoder/contract/interfaces.md` (new ABI entry points, config keys, metrics, reason
  codes), AGENTS.md's support-matrix line; `cargo test -p turbine-model --test docs_extending` stays green.

## Out of scope

- Vision / multimodal inputs of the Qwen3.5/3.6 checkpoints (weights skipped).
- MTP heads of Qwen3.5/3.6 and any speculative use of them; linear-attention state rollback — Phase 8.
- Sliding-window patterns other than gpt-oss's (e.g. Mistral v0.1's 4096 window, Qwen3 `use_sliding_window: true`):
  stay refused.
- NVFP4 checkpoints, NVIDIA, multi-node (deferred as before); Qwen3-Next-80B and any checkpoint that does not fit 2×32 GB.
- Other sizes of each family beyond the named checkpoints (they may load through the same module, but have no row of
  their own unless gated).
- BF16 Qwen3-30B-A3B / Qwen3.6-35B / Mixtral (do not fit two cards with KV).
- Lifting the 0.90× roofline work to the new families (perf beyond the 0.75× vLLM bar is a later phase).

## Constraints

- AMD only, novanas (`gfx1201`, 2× R9700 32 GB, ~640 GB/s each); lab rules of AGENTS.md (locks, ask first for new lab
  classes, never evict others' workloads, never dgx-spark*).
- Reuse kernels first, write our own last; no Python/Triton in build or runtime; every evaluation recorded in
  decisions.md.
- Pluggability: one file (or directory) per family, tool format, reasoning parser and kernel provider, behind static
  registries with conformance suites; unsupported combinations refused at startup with reason codes (exit 2).
- Correctness first: every new kernel has a CPU reference test and a batch-invariance test (`hip_batch_invariance`
  conventions); overlap scheduling (default on) stays bit-exact against serial for every family.
- Memory: a checkpoint plus its KV and state must fit its placement; TP2/EP2 rows leave the documented KV headroom.
- Disk: local disk only for active checkpoints; build caches on iSCSI.

## Interfaces

- Config: `kv.hybrid_snapshot_interval_blocks` (integer ≥ 1, default 8; only read for families with state); the new key model.reasoning_parser (default `auto` = the family's default; a registered name forces it; unregistered →
  exit 2); the existing tool-call parser key gets a per-family default.
- OpenAI API: request `reasoning_effort` (Harmony; `low|medium|high`, other → 400), `chat_template_kwargs` object passed
  to the template; response `choices[].message.reasoning_content`, streaming `choices[].delta.reasoning_content`.
- Kernel ABI (minor bumps from v2.15): paged-attention descriptor window and sink fields; d64 and d256 attention
  instances; `turbine_gdn_*` (gated delta net prefill/decode) and `turbine_conv1d_*` (causal conv, state update); MoE
  experts with bias, activation variant and quantized dtypes (FP8 block, FP8 per-tensor, MXFP4); shared-expert gate.
  A library without an op a configured family needs → exit 1 with a reason code naming the op.
- Support matrix: rows for `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `MistralForCausalLM`, `MixtralForCausalLM`,
  `GptOssForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration` on `amd` with their
  formats and placements; `PARALLEL_REFUSALS` loses `olmoe_ep_tp_drift` (S-3).
- Metrics: `turbine_hybrid_state_bytes` (gauge), `turbine_hybrid_state_snapshots_total{outcome="stored|restored|recomputed"}`,
  `turbine_reasoning_tokens_total`; `/turbine/v1/kv` reports state bytes per tier; `/turbine/v1/status` lists the
  reasoning parser under `modules`.
- Lab configs: `scripts/lab/phase7-novanas-<slug>[-tp2|-ep2].yaml`; `lab-bench.sh --model` gains the family slugs.

## Data

- Checkpoints: `/home/piwi/turbine-models/<slug>` (novanas local disk, pinned revision recorded in the family's
  `tests/golden/<slug>/README.md`).
- Fixtures: `tests/golden/<slug>/{reference.jsonl,tolerance.json,README.md}`; `tests/eval/<slug>/{vllm.json,turbine.json,gate.json}`.
- Hybrid state: per-sequence slots in the GPU pool next to KV pages; snapshots stored in the block's tier slab next to
  its KV bytes (same lifetime, same format version); never shared across lineages (lossy KV formats do not apply to
  state: state is stored exactly in F32/BF16 as the checkpoint dictates).
- Build caches on `/mnt/turbine-iscsi/turbine-ci/{remote,cache/slots}`.

## Edge cases

- A prefix hit that ends mid-block on a hybrid: restore the state at the last full block and recompute the rest; the
  output equals a cold run bit for bit.
- Chunked prefill splitting a GDN chunk boundary, and a decode batch mixing hybrid and full-attention-only sequences.
- Preemption by recompute, cancellation and `n` > 1 forks for hybrids (state copied, released promptly, tested).
- Jump-forward (5p S-10) multi-token steps on a hybrid: the state advances by every accepted token and rolled-back
  tokens do not corrupt it (the rollback re-runs from the last kept state or is refused for hybrids with a reason code).
- Sliding-window layers under prefix reuse and long contexts beyond the window; sinks with splitkv decode.
- Harmony: a channel marker split across tokens, tool calls emitted in the analysis channel, a request without
  `reasoning_effort`, streaming with `n` > 1.
- `<think>` never closed before `max_tokens`; `enable_thinking: false` with templates that ignore it.
- TP2/EP2 shapes: kv heads 2 (Qwen3.6) or 4 (Qwen3.5-9B) over two ranks, vocab 248320 sharding, 256 experts over two
  ranks, router near-ties across ranks (S-3).
- Mixtral's thin KV headroom under c16 and long prompts (admission must refuse, never OOM).
- A checkpoint with an unexpected tensor (e.g. `consolidated.safetensors` present) or missing shard.

## Failure modes

- Checkpoint directory missing or incomplete → exit 1 naming the file (never download at runtime).
- Kernel library without an op the family needs (attention d64/d256, sinks, window, GDN, conv, quantized experts) →
  exit 1 with a reason code naming the ABI entry point.
- Unsupported combination (format × family × placement × KV format, sliding pattern, tq2 L0, etc.) → exit 2 with a reason
  code before any port is bound.
- vLLM-ROCm cannot serve a family → the no-vLLM rule of S-13 (logged, not a failure of the row).
- A GDN provider is slower than a reference or wrong on gfx1201 → the evaluation records it and the next candidate (or
  our own kernel) is used; never a Python fallback.
- iSCSI volume unavailable at boot → the bind mounts are `nofail`; build caches start cold on local disk; models are
  unaffected (local).
- transformers cannot build a reference on novanas within RAM → build it on cpuhost.
- Router or reduction drift under TP2/EP2 → the row stays `experimental` or the placement is refused with a reason code
  until S-3-style evidence closes it.

## Acceptance criteria

- [ ] [S-1] `cargo test -p turbine-model families::tests::layer_kinds_and_state_layout` exits 0; for the tiny fixture of every registered family it asserts the layer-kind list and `StateLayout` bytes derived from `config.json` (gpt-oss alternates sliding 128 / full; Qwen3.5 has 3 linear : 1 full with conv k=4 and F32 recurrent state; dense families have no state); fails if a family's kinds or state bytes differ from its config.
- [ ] [S-1] `cargo test -p turbine-scheduler --test kv_sim hybrid_reservation_includes_state` exits 0; the worst-case reservation of a hybrid request equals its KV pages plus its state slot bytes, and admission refuses a request whose KV + state exceeds the budget; fails if the reservation omits the state.
- [ ] [S-1] `cargo test -p turbine-server --bin turbine-server support_startup::tests::sliding_window_refused` exits 0; a Qwen3 config with `use_sliding_window: true` and a Mistral config with `sliding_window: 4096` exit 2 with reason code `sliding_window_unsupported` before any port is bound; fails if either starts.
- [ ] [S-2] [S-13] Manual lab gate (GPU 0): `scripts/lab-bench.sh --model qwen3-4b --golden16` prints golden1=PASS and golden16=PASS against `tests/golden/qwen3-4b-instruct-2507/reference.jsonl`, `turbine-golden eval-compare` exits 0 for the c16 GSM8K pair against vLLM-ROCm (max drop 0.01), and the BENCH tok/s is ≥ 0.75× the vLLM-ROCm c16 run of `scripts/lab-perf.sh`; perf-log records all three; fails if any of them misses.
- [ ] [S-3] `cargo test -p turbine-server --test lab_cluster olmoe_ep2_tp2_golden -- --ignored` exits 0 under `scripts/lab-cluster.sh ep2-novanas` (both R9700s): golden c1 and c16 PASS against `tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl`; `grep olmoe_ep_tp_drift crates/turbine-core/src/support.rs` finds nothing; perf-log section "P7 OLMoE EP×TP drift" holds the per-position routing trace for p08 and p10 and names the cause; fails if the refusal remains, the cause is not stated or ep2×tp2 golden fails.
- [ ] [S-4] `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_moe_quant` exits 0: `fp8_block_experts_match_reference`, `fp8_tensor_experts_match_reference`, `mxfp4_experts_match_reference` (CPU reference within the 6a per-format bounds) and `quant_experts_batch_invariant` pass; `grep -rn quant_moe_phase7 crates/` finds only the refusal of combinations with no row; decisions.md has the entry "P7 quantized MoE expert provider"; fails if a format lacks a test or the evaluation entry is missing.
- [ ] [S-5] [S-13] Manual lab gate (both R9700s): `scripts/lab-cluster.sh --bench-lock qwen3moe-tp2-novanas` and `qwen3moe-ep2-novanas` each print `lab-cluster: … PASS` with golden c1/c16 (`--batched-bounds`) and the GSM8K pair (6a FP8 drop) passing; perf-log records both placements' c16 tok/s against vLLM-ROCm on the same placement and `scripts/lab/phase7-novanas-qwen3moe.yaml` uses the faster; fails if a placement neither passes nor is refused with a reason code.
- [ ] [S-6] [S-13] Manual lab gate (GPU 0): `scripts/lab-bench.sh --model mistral-7b --golden16` golden1/golden16 PASS, the GSM8K pair passes, BENCH ≥ 0.75× vLLM-ROCm; `cargo test -p turbine-server --test lab_openai mistral_tool_call_round_trip -- --ignored` exits 0; fails if any of them misses.
- [ ] [S-7] [S-13] Manual lab gate (both R9700s): `scripts/lab-cluster.sh --bench-lock mixtral-tp2-novanas` and `mixtral-ep2-novanas` each PASS (golden c1/c16 batched bounds, GSM8K 6a FP8 drop, c16 throughput vs vLLM-ROCm), or the failing placement exits 2 with a reason code in `support_startup::tests::mixtral_placement_refusal`; a c16 run with 512-word prompts ends with `failed=0` and no `device_oom` in the log, and `/turbine/v1/kv` reports `headroom_bytes` per rank; fails if a placement neither passes nor is refused, or the run OOMs.
- [ ] [S-8] `cargo test -p turbine-model --test tiny_model gpt_oss_matches_reference` exits 0 (CPU executor vs the committed transformers trace of the tiny gpt-oss fixture, per op), `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops attention_sinks_window_match_reference attention_sinks_batch_invariant` exits 0, and `scripts/lab-test.sh novanas -- -p turbine-model --test golden hip_trace_vs_cpu_gpt_oss` exits 0; fails if any of them fails.
- [ ] [S-8] [S-13] Manual lab gate (GPU 0): `scripts/lab-bench.sh --model gpt-oss-20b --golden16` exits 0 with golden1/golden16 PASS against `tests/golden/gpt-oss-20b/reference.jsonl` and the GSM8K pair passes the S-13 rule (vLLM-ROCm or the no-vLLM fallback); `cargo run -p turbine-server -- --support-matrix` shows `GptOssForCausalLM` mxfp4 and the `openai_mxfp4` packaging `supported`; fails if a gate misses or either row is not `supported`.
- [ ] [S-9] `cargo test -p turbine-model formats::tests::harmony_round_trip` and `cargo test -p turbine-server --test lab_openai harmony_channels -- --ignored` exit 0: analysis text in `reasoning_content`, final text in `content`, a commentary tool call in `tool_calls` with name and arguments, streaming equal to non-streaming, a marker split across tokens handled, and `reasoning_effort: "extreme"` answered 400; fails if any assertion fails.
- [ ] [S-10] `cargo test -p turbine-model reasoning::registry_conformance` exits 0 over `qwen_think` and `harmony`; `cargo run -p turbine-server -- --config examples/turbine.yaml --check-config --set model.reasoning_parser=nope` exits 2 naming the key; `cargo test -p turbine-server --test tiny_server reasoning_content_follows_enable_thinking` exits 0 (thinking on → `reasoning_content` present; `chat_template_kwargs: {"enable_thinking": false}` → absent); fails if any of them fails.
- [ ] [S-11] `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_gdn` exits 0: `gdn_prefill_matches_reference`, `gdn_decode_matches_reference`, `conv1d_state_update_matches_reference` and `gdn_batch_invariant` pass; decisions.md has the entry "P7 Gated DeltaNet provider" with timings for llama.cpp, AITER and (only where both fail or are slower) our own kernel; the server log of a Qwen3.5 load has `event="weights_skipped"` for the vision tower and `mtp.*`; fails if a test fails or the entry is missing.
- [ ] [S-11] [S-13] Manual lab gates: `scripts/lab-bench.sh --model qwen3.5-9b --golden16` (GPU 0) and `scripts/lab-cluster.sh --bench-lock qwen36-tp2-novanas` / `qwen36-ep2-novanas` (both R9700s) exit 0 and pass golden c1/c16 and the GSM8K pair per S-13; perf-log records their throughput against vLLM-ROCm or the no-vLLM fallback; fails if a checkpoint misses a gate.
- [ ] [S-12] `cargo test -p turbine-server --bin turbine-server engine::r#loop::tests::hybrid_prefix_reuse_matches_cold` exits 0 on the tiny Qwen3.5 fixture (CPU): multi-turn prefix hits at 1, 2 and 3 full blocks and a mid-block hit equal a cold run bit for bit, `n` = 2 forks equal two separate requests, and cancellation returns the state pool to its start size, and with `kv.hybrid_snapshot_interval_blocks` 2 a hit at 3 blocks restores the snapshot of block 2 and recomputes block 3 (`turbine_hybrid_state_snapshots_total{outcome="restored"}` and the recomputed token count asserted); fails if any stream differs or a slot leaks.
- [ ] [S-12] `cargo test -p turbine-kv hierarchy::tests::state_snapshot_follows_block` exits 0 (a block demoted to L1 and L2 and promoted back keeps its snapshot bytes) and `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu hybrid_prefix_reuse_matches_cold` exits 0 with Qwen3.5-9B; `turbine_hybrid_state_snapshots_total{outcome="restored"}` > 0 in that test; fails if a snapshot is lost or the GPU run differs from cold.
- [ ] [S-13] `cargo test -p turbine-core support::tests::phase7_rows` exits 0: every Phase 7 family/format/placement row exists with its status and reason, and every row the perf-log marks closed is `supported`; fails if a row is missing or its status disagrees with perf-log.
- [ ] [S-14] perf-log section "P7 MXFP4 p05 trace" holds the CPU-vs-transformers per-op table with the cause; `cargo test -p turbine-core support::tests::phase7_rows` asserts the `mxfp4` Llama row matches its golden result and `mxfp4_a4` is `supported` only with GSM8K drop ≤ 0.04 and golden PASS against `tests/golden/llama-3.2-3b-instruct-mxfp4-a4/tolerance.json` (self-spread derivation in its README); fails if the trace is missing or a row disagrees with its gate.
- [ ] [S-15] Manual lab gate (GPU 0): `scripts/lab-bench.sh --model olmoe --golden16 -- --set kv.dtype=fp8_e4m3` exits 0 with golden1/golden16 PASS with calibrated V scales and the row is `supported`, else perf-log lists the failing prompts and `support::tests::phase7_rows` keeps it `experimental`; fails if the row's status disagrees with the gate.
- [ ] [S-16] perf-log section "P7 L0 tq4 re-gate" records the Phase 6b Task 13 gates re-run on the Phase 7 tree for Llama and OLMoE, and `cargo test -p turbine-core support::tests::phase7_rows` exits 0 matching each row's status to them; fails if the re-run is unrecorded or a status disagrees.
- [ ] [S-17] `ssh piwi@192.168.10.203 'findmnt -n -o SOURCE --target /home/piwi/turbine-ci/remote; findmnt -n -o SOURCE --target /home/piwi/turbine-ci/cache/slots'` exits 0 printing `/dev/sda[/turbine-ci/remote]` and `/dev/sda[/turbine-ci/cache/slots]`, `scripts/gate.sh` exits 0 on that layout, and every `tests/golden/<slug>/README.md` of this phase records the checkpoint revision; fails if a path is local or a revision is missing.
- [ ] [S-18] `cargo test -p turbine-model formats::tests::family_default_tool_calls_round_trip` exits 0: each family's default parser round-trips a tool call rendered by its real chat-template fixture, including Qwen3.5's format (a new parser registered if it is not `hermes`); fails if any family's call does not round-trip.
- [ ] [S-19] `grep -rn "Phase 8 track" crates/` exits 1 (no match) and `grep -rln GptOssForCausalLM crates/*/tests crates/turbine-server/src/support_startup.rs` names no test using it as an unregistered architecture; fails if either finds a match.
- [ ] [S-20] `cargo test -p turbine-model --test docs_extending` exits 0 with `docs/extending/model-family.md` describing layer kinds, `StateLayout` and reasoning parsers, and `.procoder/contract/interfaces.md` lists every new ABI entry point, config key, metric and reason code of this phase; fails if the test fails or an entry is missing.
- [ ] [S-1] [S-13] Phase exit: `scripts/gate.sh --full`, `scripts/lab-test.sh novanas --tier full`, `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full`, `scripts/lab-bench.sh --golden16` for Llama and OLMoE and `scripts/overload-soak.sh novanas --duration 10m` each exit 0 on the final tree, with Llama/OLMoE BENCH tok/s within 3 % of the 5p exit lines (864.0 / 636.1); fails if any tier fails or a BENCH line regresses by more than 3 %.

## Open questions

<!-- None open for the checker: scope is fixed by the umbrella S-8; every design choice was decided by the user in the decisions.md entries "Phase 7 spec: order, exit bar, checkpoints, disk, GDN kernel, carry-over (2026-10-07)" and its follow-up rounds (2026-10-07/08). -->
