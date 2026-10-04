# Support matrix

The human-readable view of `turbine_core::support::SUPPORT_MATRIX` (`crates/turbine-core/src/support.rs`): the declarative table of `(vendor, arch, architecture, weight_format, kv_format, speculative) → status` that decides at startup whether a configured combination may serve. This page is a snapshot for readers; **the command is the always-current source**:

```
scripts/remote-cargo.sh run -q -p turbine-server -- --support-matrix [--output json]
```

`cargo test -p turbine-model --test docs_extending` renders every row of the table from code and fails when this page misses a row, contradicts a status, or drops a reason anchor (the `docs_support_matrix_*` tests), so the page cannot drift quietly.

## How a key reads

A key is `vendor/arch/architecture/weight_format/kv_format/speculative`:

- `vendor` — `amd` (the lab R9700s), `cpu` (the host reference provider), `nvidia` (deferred); `*` means any, or not known yet before device discovery.
- `arch` — the device architecture (`gfx1201` on the R9700); `*` before discovery.
- `architecture` — the Hugging Face class of the model's `config.json` (`LlamaForCausalLM`, `OlmoeForCausalLM`, …).
- `weight_format` — the checkpoint packaging `weights::detect` picks (see `docs/extending/weight-format.md`).
- `kv_format` — the L0 KV dtype (`kv.dtype`).
- `speculative` — `none`, or `draft` for draft-model speculative decoding (`speculative.method`).

Statuses:

- `supported` — serves.
- `experimental` — serves, with a WARN at startup (`event="support_matrix"`, `support matrix: <key> is experimental`).
- `unsupported` — refused as a configuration error before any port is bound (exit 2, naming the blamed config key — `execution.backend`, `model.path`, `kv.dtype` or `speculative.method` — and the reason). Where the tables below say "refused", this is meant.

Resolution: a key of a deferred vendor ([`DEFERRED_VENDORS`], § Deferred vendors) is refused before the rows are consulted; otherwise the most specific matching row wins. Lower-tier KV formats (`kv.cpu.format`, `kv.nvme.format`, the ladder's rungs) resolve through `TIER_FORMAT_REFUSALS`, not through this table (§ Lower-tier KV formats). Parallel-mode refusals are a third table (`PARALLEL_REFUSALS`, § Parallel-mode refusals).

## amd / gfx1201 (the Radeon R9700s)

The resolved status of every `amd/gfx1201/<architecture>/<weight>/<kv>/none` combination. The cells are generated from `support::resolve` by the drift test; the notes column says why.

### LlamaForCausalLM

| Weight format    | KV `bf16`    | KV `fp8_e4m3` | KV `tq4`     | KV `tq2`    | Notes                                                                                                                                                                                                                   |
| ---------------- | ------------ | ------------- | ------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bf16`           | supported    | supported     | experimental | unsupported | FP8 KV since the Task 24 proof (full GSM8K c16 0.7885 vs BF16 0.7801, golden c1/c16 16/16 against `tests/golden/llama-3.2-3b-instruct-fp8kv`); L0 `tq4` failed the Task 13 golden gate; L0 `tq2` is `kv_tq2_l0_refused` |
| `fp8`            | supported    | unsupported   | unsupported  | unsupported | weights since Task 14 (golden 16/16 at c1 and c16; the GSM8K drop accepted as noise, McNemar p 0.152); FP8 KV is proven for BF16 weights only                                                                           |
| `fp8_block`      | supported    | unsupported   | unsupported  | unsupported | Task 15: proof 2026-09-29, golden 16/16 against `tests/golden/llama-3.2-3b-instruct-fp8-block`, 10-min soak PASS                                                                                                        |
| `mxfp4`          | experimental | unsupported   | unsupported  | unsupported | Phase 7 item (user decision 2026-09-30 A); the MXFP4 prefill kernel is a recorded follow-up                                                                                                                             |
| `mxfp4_a4`       | experimental | unsupported   | unsupported  | unsupported | MXFP4 weights with emulated MXFP4 activations (W4A4, decision 2026-09-28 Q6); same Phase 7 item                                                                                                                         |
| `awq_int4`       | supported    | unsupported   | unsupported  | unsupported | Task 18: c16 1.50× BF16 tok/s, c1 ITL 0.47×, golden c1 strict and c16 batched 16/16, GSM8K drop 0.030 ≤ 0.04, soak 8/8                                                                                                  |
| `gptq_int4`      | supported    | unsupported   | unsupported  | unsupported | Task 18 on the AutoRound GPTQ checkpoint (decision 2026-09-30 B): full GSM8K drop 0.0235 ≤ 0.04, golden 16/16 both, soak 8/8                                                                                            |
| `modelopt_nvfp4` | unsupported  | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia` (NVFP4 arrives with NVIDIA support)                                                                                                                                                      |
| `modelopt_fp8`   | unsupported  | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                          |
| `modelopt_mixed` | unsupported  | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                          |
| `ct_nvfp4`       | unsupported  | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                          |

### OlmoeForCausalLM

| Weight format                                                                                                                                                                                                       | KV `bf16`   | KV `fp8_e4m3` | KV `tq4`     | KV `tq2`    | Notes                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | ------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bf16`                                                                                                                                                                                                              | supported   | experimental  | experimental | unsupported | FP8 KV: full GSM8K 0.6603 vs 0.6459 but golden c1/c16 13/16 (need 14; p03 tail 5.38) — user decision 2026-09-30 B, calibrated V scales and the golden miss are Phase 7 items; L0 `tq4` failed the Task 13 gates (golden 8/16, shared-prefix GSM8K 0.615 vs 0.635) |
| `fp8`                                                                                                                                                                                                               | unsupported | unsupported   | unsupported  | unsupported | `phase-6a-quantization`: the proven weight rows are Llama-only so far                                                                                                                                                                                             |
| `fp8_block`                                                                                                                                                                                                         | unsupported | unsupported   | unsupported  | unsupported | `phase-6a-quantization`                                                                                                                                                                                                                                           |
| `mxfp4`                                                                                                                                                                                                             | unsupported | unsupported   | unsupported  | unsupported | `phase-6a-quantization`                                                                                                                                                                                                                                           |
| `mxfp4_a4`                                                                                                                                                                                                          | unsupported | unsupported   | unsupported  | unsupported | `phase-6a-quantization`                                                                                                                                                                                                                                           |
| `awq_int4`                                                                                                                                                                                                          | unsupported | unsupported   | unsupported  | unsupported | `phase-6a-quantization`                                                                                                                                                                                                                                           |
| `gptq_int4`                                                                                                                                                                                                         | unsupported | unsupported   | unsupported  | unsupported | `phase-6a-quantization`                                                                                                                                                                                                                                           |
| `modelopt_nvfp4`                                                                                                                                                                                                    | unsupported | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                                                                    |
| `modelopt_fp8`                                                                                                                                                                                                      | unsupported | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                                                                    |
| `modelopt_mixed`                                                                                                                                                                                                    | unsupported | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                                                                    |
| `ct_nvfp4`                                                                                                                                                                                                          | unsupported | unsupported   | unsupported  | unsupported | reserved for `phase-2b-nvidia`                                                                                                                                                                                                                                    |
| Quantized weight formats on any _other_ architecture or vendor are refused with `phase-6a-quantization` until their track proves them; FP8 / TurboQuant KV on any other arch is refused with the KV track's reason. |

## Every row of the table

The rows of `SUPPORT_MATRIX`, in code order; the drift test requires exactly these lines (key, status and, for a refusal, the reason anchor — its `phase-*` track or reason code — in the last column). `*` is the wildcard of `support.rs`.

| Key                                                | Status       | Why                                                                                                                                                                                 |
| -------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `amd/gfx1201/LlamaForCausalLM/bf16/bf16/none`      | supported    | The Phase 1–5 baseline; the golden gate and every perf row of `.procoder/perf-log.md` run on it                                                                                     |
| `amd/gfx1201/OlmoeForCausalLM/bf16/bf16/none`      | supported    | The Phase 2 baseline (MoE)                                                                                                                                                          |
| `nvidia/sm_121/LlamaForCausalLM/bf16/bf16/none`    | unsupported  | Deferred vendor: `phase-2b-nvidia` (§ Deferred vendors)                                                                                                                             |
| `nvidia/sm_121/OlmoeForCausalLM/bf16/bf16/none`    | unsupported  | Deferred vendor: `phase-2b-nvidia`                                                                                                                                                  |
| `cpu/*/*/bf16/bf16/none`                           | experimental | The CPU reference provider: tests and tiny checkpoints only (user decision 2026-09-25)                                                                                              |
| `amd/gfx1201/LlamaForCausalLM/bf16/fp8_e4m3/none`  | supported    | Phase 6a Task 24: full GSM8K at c16 0.7885 vs BF16 0.7801, golden c1/c16 16/16                                                                                                      |
| `amd/gfx1201/OlmoeForCausalLM/bf16/fp8_e4m3/none`  | experimental | Golden 13/16 at c1/c16 (need 14; p03 tail 5.38), user decision 2026-09-30 B; a Phase 7 item                                                                                         |
| `amd/gfx1201/LlamaForCausalLM/fp8/bf16/none`       | supported    | Task 14: dynamic and static FP8 checkpoints, golden 16/16 at c1 and c16; the c1 ITL target stays a perf follow-up                                                                   |
| `amd/gfx1201/LlamaForCausalLM/fp8_block/bf16/none` | supported    | Task 15: proof 2026-09-29, golden 16/16 both, 10-min soak PASS, closed 2026-09-30                                                                                                   |
| `amd/gfx1201/LlamaForCausalLM/mxfp4/bf16/none`     | experimental | Phase 7 item (user decision 2026-09-30 A)                                                                                                                                           |
| `amd/gfx1201/LlamaForCausalLM/mxfp4_a4/bf16/none`  | experimental | Phase 7 item (decision 2026-09-28 Q6, W4A4)                                                                                                                                         |
| `amd/gfx1201/LlamaForCausalLM/awq_int4/bf16/none`  | supported    | Task 18 (t18-run): perf, golden, eval and the rotation-9 soak all passed                                                                                                            |
| `amd/gfx1201/LlamaForCausalLM/gptq_int4/bf16/none` | supported    | Task 18 on the AutoRound checkpoint (decision 2026-09-30 B)                                                                                                                         |
| `cpu/*/*/fp8/bf16/none`                            | experimental | Reference provider: tests and tiny checkpoints (Phase 6a S-3)                                                                                                                       |
| `cpu/*/*/fp8/fp8_e4m3/none`                        | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/fp8_block/bf16/none`                      | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/fp8_block/fp8_e4m3/none`                  | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/mxfp4/bf16/none`                          | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/mxfp4/fp8_e4m3/none`                      | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/mxfp4_a4/bf16/none`                       | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/mxfp4_a4/fp8_e4m3/none`                   | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/awq_int4/bf16/none`                       | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/awq_int4/fp8_e4m3/none`                   | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/gptq_int4/bf16/none`                      | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/gptq_int4/fp8_e4m3/none`                  | experimental | Reference provider                                                                                                                                                                  |
| `cpu/*/*/bf16/fp8_e4m3/none`                       | experimental | FP8 KV on the reference provider (Phase 6a S-13)                                                                                                                                    |
| `cpu/*/*/bf16/tq4/none`                            | experimental | TurboQuant L0 pages through `cpu::tq_attention` (P6b S-5); a GPU provider needs the v2.11 mixed-format attention                                                                    |
| `amd/gfx1201/LlamaForCausalLM/bf16/tq4/none`       | experimental | Task 13 S-8 gate failed: golden 0/16 under the batched bounds, but the shared-prefix GSM8K eval passed (0.785 vs 0.780) — `.procoder/perf-log.md`, "TurboQuant in L0"               |
| `amd/gfx1201/OlmoeForCausalLM/bf16/tq4/none`       | experimental | Task 13 failed both: golden 8/16, shared-prefix GSM8K 0.615 vs 0.635                                                                                                                |
| `amd/*/Qwen3ForCausalLM/bf16/bf16/none`            | unsupported  | `phase-7-model-families`; the CPU reference provider serves it `experimental`                                                                                                       |
| `amd/*/Qwen3MoeForCausalLM/bf16/bf16/none`         | unsupported  | `phase-7-model-families`                                                                                                                                                            |
| `amd/*/MistralForCausalLM/bf16/bf16/none`          | unsupported  | `phase-7-model-families`                                                                                                                                                            |
| `amd/*/MixtralForCausalLM/bf16/bf16/none`          | unsupported  | `phase-7-model-families`                                                                                                                                                            |
| `*/*/*/fp8/bf16/none`                              | unsupported  | `phase-6a-quantization` — the proven gfx1201 Llama row above overrides                                                                                                              |
| `*/*/*/fp8_block/bf16/none`                        | unsupported  | `phase-6a-quantization`                                                                                                                                                             |
| `*/*/*/mxfp4/bf16/none`                            | unsupported  | `phase-6a-quantization`                                                                                                                                                             |
| `*/*/*/mxfp4_a4/bf16/none`                         | unsupported  | `phase-6a-quantization`                                                                                                                                                             |
| `*/*/*/awq_int4/bf16/none`                         | unsupported  | `phase-6a-quantization`                                                                                                                                                             |
| `*/*/*/gptq_int4/bf16/none`                        | unsupported  | `phase-6a-quantization`                                                                                                                                                             |
| `*/*/*/modelopt_nvfp4/bf16/none`                   | unsupported  | `phase-2b-nvidia`                                                                                                                                                                   |
| `*/*/*/modelopt_fp8/bf16/none`                     | unsupported  | `phase-2b-nvidia`                                                                                                                                                                   |
| `*/*/*/modelopt_mixed/bf16/none`                   | unsupported  | `phase-2b-nvidia`                                                                                                                                                                   |
| `*/*/*/ct_nvfp4/bf16/none`                         | unsupported  | `phase-2b-nvidia`                                                                                                                                                                   |
| `*/*/*/*/fp8_e4m3/none`                            | unsupported  | `phase-6a-quantization` — the proven gfx1201 rows and the reference-provider rows above override                                                                                    |
| `*/*/*/*/tq4/none`                                 | unsupported  | `phase-6b-kv-compression` — the gfx1201 and reference-provider rows above override                                                                                                  |
| `*/*/*/*/tq2/none`                                 | unsupported  | `kv_tq2_l0_refused`: shared-prefix GSM8K collapses to 0.15–0.20 on Llama and OLMoE (user decision 2026-10-02, "6b Task 13", 2 B); `tq2` stays a lower-tier format and a ladder rung |
| `*/*/*/*/*/draft`                                  | unsupported  | `phase-8-speculative-decoding`                                                                                                                                                      |

## Lower-tier KV formats

`kv.cpu.format`, `kv.nvme.format` and the ladder's `kv.ladder.max_format` resolve through `TIER_FORMAT_REFUSALS` (`crates/turbine-core/src/support.rs`), one row per format, not per model. A format the table does not list is `supported` (its device availability is decided by the kernel library's ABI v2.11 KV transcode, `docs/extending/kv-format.md`).

| Format     | Status       | Evidence                                                                                                                                                                                                                 |
| ---------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `l0`       | supported    | The tier stores the L0 bytes unchanged                                                                                                                                                                                   |
| `fp8_e4m3` | supported    | P6b Task 6 (2026-10-01): shared-prefix GSM8K-200 median drop 0.005 (max 0.01), McNemar p ≥ 0.55 on all nine pairs; `kv_gpu` green on gfx1201                                                                             |
| `tq4`      | supported    | Task 9 golden c1/c16 PASS both models; multi-turn A/B (2026-10-02): cached-token ratio at least the `l0` run's on both models, measured after the copy-drift fix (`.procoder/perf-log.md`, "KV copies and decode steps") |
| `tq2`      | experimental | Task 9 shared-prefix GSM8K FAIL (Llama 0.720, OLMoE 0.610 vs BF16 0.780 / 0.635); usable as a lower-tier format and a ladder rung, never as `kv.dtype`                                                                   |

The compression ladder (`kv.ladder.enabled`, with its L0 step `kv.ladder.l0`) is opt-in with a documented throughput/tail trade (user decision 2026-10-04 A): the Task 18 multi-turn A/B measured recomputed tokens −23 % and cached ratio 0.860 vs 0.837 with the L0 step, but tok/s 233 vs 304 and later-turn TTFT p99 35.1 s vs 12.4 s — the server spends longer at ORANGE while L0 rewrites occupy the rewrite lanes. The serving-cost investigation is a recorded follow-up, not a gate.

## The cpu reference provider

Every `cpu` row above is `experimental` on purpose (user decision 2026-09-25): the host provider exists for tests, tiny checkpoints and fixtures, not for serving. It serves every Phase 6a weight format with BF16 or FP8 KV, TurboQuant L0 pages through `cpu::tq_attention`, and the Phase 7 families before their GPU rows exist.

## Deferred vendors

| Vendor   | Status      | Why                                                                                                                                                                                                                                                                                                                                 |
| -------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia` | unsupported | `phase-2b-nvidia`: NVIDIA execution is on hold until the user lifts it (decision 2026-09-26, "AMD first"); the NVFP4 / ModelOpt weight columns (`modelopt_nvfp4`, `modelopt_fp8`, `modelopt_mixed`, `ct_nvfp4`) are reserved for it, and `DEFERRED_VENDORS` refuses every `nvidia` key with this reason before any row is consulted |

## Phase 7 families

`Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `MistralForCausalLM` and `MixtralForCausalLM` are refused on `amd` with `phase-7-model-families` until the track validates them (the umbrella's Task 10 flips each row). The CPU reference provider serves them `experimental` today.

## Parallel-mode refusals

| Architecture     | Modes   | Reason                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OlmoeForCausalLM | `ep+tp` | `olmoe_ep_tp_drift`: OLMoE with expert parallelism > 1 and tensor parallelism > 1 together drifts past its golden tolerance (p10 likely Δ 1.031 > 1.01, and p08); user decision 2026-09-28, refused at startup naming `parallel.expert_parallel_size` — `phase 7` investigates the drift (expansion umbrella question (d)). Each mode alone stays supported |

## Experimental today, and what would flip it

- **OLMoE FP8 KV (`experimental`).** Golden c1/c16 13/16 against `tests/golden/olmoe-1b-7b-0125-instruct-fp8kv` (need 14; p03 tail 5.38), user decision 2026-09-30 B. Flips when a calibrated-scales checkpoint passes the golden gate; the checkpoint ships no K/V scales, so this is a Phase 7 item.
- **L0 `tq4` on gfx1201 (`experimental`).** The Task 13 S-8 gate failed the golden batched bounds on both models (Llama 0/16, OLMoE 8/16) even though Llama passed the shared-prefix GSM8K eval; `.procoder/perf-log.md`, "TurboQuant in L0". Flips when the golden logprob bounds hold for whole-lossy L0 pages (a better TurboQuant tail, or a partial-precision L0 layout).
- **Lower-tier `tq2` (`experimental`).** Task 9 shared-prefix GSM8K collapsed on both models; it stays a lower-tier format and ladder rung. Flips only if a future eval passes; L0 `tq2` is refused outright (`kv_tq2_l0_refused`, decision 2026-10-02).
- **`mxfp4` / `mxfp4_a4` on gfx1201 Llama (`experimental`).** Phase 7 items (decision 2026-09-30 A); the MXFP4 prefill kernel is the recorded perf follow-up.
- **The Phase 7 families on `amd` (refused).** Flip per family when `scripts/track-gate.sh phase-7-model-families` passes and the track's rows land.
- **Draft speculation (refused everywhere).** `phase-8-speculative-decoding`.
- **`cpu` serving (`experimental`).** By design; the reference provider never becomes a serving target.
- **The L0 ladder (opt-in).** Off by default; accepted with the documented trade (decision 2026-10-04 A). The serving-cost investigation (rewrite-lane occupancy, ORANGE dwell) is the follow-up that could change the default.

### Recorded follow-ups (6b lead handoff, `.procoder/handoff/phase-6b-lead.md`)

Ladder serving-cost investigation; FP8 c1 ITL; MXFP4 prefill kernel; OLMoE BF16 0.982×; slab mixing (B-style reformat — 128 MiB slabs landed but mixing never triggered in the workloads); upstream ROCm items (rocm-systems#12677, rocm-libraries#12895 + PR #12900); `kv_sim` attach-order nondeterminism (HashMap iteration, noted by t17).

## Checking at runtime

- `scripts/remote-cargo.sh run -q -p turbine-server -- --support-matrix` prints every row (`vendor/arch/architecture/weight/kv/spec → status`, with reasons); `--output json` prints the rows plus `parallel_refusals` and `deferred_vendors`. Exits 0 without reading a config.
- `--check-config` resolves only the configured row (before device discovery, `arch` is `*`) and prints it as `support: <status> (<vendor>/<arch>/<architecture>/<weight>/<kv>/<spec>)`, then `config ok`; an `unsupported` resolution exits 2 before any port is bound.
- The running server reports the resolved row as `support` on `/turbine/v1/status`, the gauge `turbine_support_matrix_status{status}`, the chosen module per extension point under `modules` and every kernel choice under `kernels`; the log carries `event="support_matrix"` (WARN when `experimental`).
