# phase-6b-kv-compression — implementation plan

Status: draft
Spec: .procoder/specs/phase-6b-kv-compression.md

The user answered questions 1–20 of `.procoder/ask/decisions.md`, entry "Phase 6 spec: provisional design choices (2026-09-28)" (Q11 changed: TurboQuant also lives in L0), and split Phase 6 in two (entry "Phase 6 split: 6a quantization, 6b KV compression (2026-09-28)"). This plan starts only after `phase-6a-quantization` has closed. Tasks: per-tier formats (1–6), TurboQuant as a lower-tier codec (7–9) and in L0 (10–13), compression ladder in L1/L2 (14–16) and in L0 (17–18), phase exit (19). In the joint Phase 6 plan these were Tasks 29–40b.

## Goal

Add per-tier KV formats with GPU-side transcoding (proven with FP8 first), TurboQuant `tq4` / `tq2` as a lower-tier codec and as the L0 format through a mixed-format paged attention, lossy lineage with a per-request opt-out, and a pressure-driven KV compression ladder in L1/L2 and then L0 — each lossy format gated by golden and accuracy against BF16 KV, measured on GPU 0, and entered into the support matrix only when its gate passes.

## Architecture

`turbine_kv::codec` is a new `kv_format` registry of codecs (`l0`, `fp8_e4m3`, `tq4`, `tq2`) whose CPU side lives in `turbine-kv` and whose GPU side is the v2.11 `turbine_kv_transcode` op called by the server's orchestrator around the existing demotion/promotion copies. Directory locations carry a format; lossy copies and blocks computed over them get lineage keys; the planner weighs lossy retrieval with a penalty; requests may opt out with `x-turbine-kv-lossy: deny`. In L0, block-table entries carry a format tag, pages come from one page class per format, and a mixed-format paged attention (v2.11) reads BF16 / FP8 blocks directly and TurboQuant blocks in the rotated domain (q rotated once per step, V accumulated rotated and rotated back once). The ladder is a third `EvictAction` (`Compress`) decided by the eviction policy from tier fill and the pressure state and applied by the hierarchy within the Phase 4 per-tick bounds, first in L1/L2 and then in L0.

## Constraints

Copied verbatim from the spec (Constraints):

- Test tiers (AGENTS.md "Test tiers", decision 2026-09-27): per landing step `scripts/gate.sh`, plus `scripts/lab-test.sh novanas --tier quick` when GPU-facing code changes (`turbine-kernels`, `turbine-model`, `turbine-device`, the ABI, `kernels/rocm`), plus `scripts/lab-bench.sh --quick` when throughput or numerics can change; phase exit: `scripts/gate.sh --full`, `scripts/lab-test.sh novanas --tier full` and its two-GPU leg, `scripts/lab-bench.sh --golden16` for Llama and OLMoE BF16 with every KV format on, and `scripts/overload-soak.sh novanas --duration 10m` (asked first).
- Reuse first (AGENTS.md rule, decision "Kernel reuse policy"): every kernel of S-1 (transcode), S-4 and S-5 is preceded by a recorded evaluation in `.procoder/ask/decisions.md` of CK (`ck_tile` `gemm_quant`, microscale, FMHA FP8), hipBLASLt (FP8, scaled), llama.cpp HIP (q4/q8 and MXFP4 mat-vec and mmq, flash attention with quantized KV), vLLM / SGLang / aiter ROCm kernels and the compressed-tensors / Quark unpack paths; an own kernel only when none builds, none is correct on `gfx1201` or all are measurably slower. No Python in the build or runtime path; third-party kernel sources are pinned (CK by its existing `FetchContent` commit; llama.cpp by commit if used) and their licenses land in `kernels/rocm/third_party/LICENSES/`.
- Correctness bar: every GPU implementation has a lab test against the CPU reference (codecs and `cpu::tq_attention`); the BF16 and FP8 golden tolerances, the OLMoE calibration and the Phase 4 `kv_gpu` bit-exact checks at the `l0` tier format do not change; lossy blocks are never served to an opted-out request.
- Nothing lossy by default: `kv.cpu.format`, `kv.nvme.format`, `kv.dtype` and `kv.ladder.enabled` default to the exact behaviour.
- Pluggability: every KV codec and kernel implementation is one file (or directory) plus a registry entry; a new `docs/extending/kv-format.md` describes codecs and `docs/extending/eviction-policy.md` the compress action; `cargo test -p turbine-model --test docs_extending` keeps them true.
- Vendor neutrality (umbrella S-5): no HIP type in a public signature of `turbine-kernels`, `turbine-tensor`, `turbine-scheduler`, `turbine-kv` or `turbine-reliability`; every ABI addition is an optional minor group specified so a CUDA library can implement it when `phase-2b-nvidia` is re-specced.
- Unsafe Rust and FFI stay in `turbine-kernels` (and the existing allowlist); `turbine-kv` stays GPU-free (codecs' GPU side is a kernel op the server's orchestrator calls).
- Host copies go through pinned memory (L1 slots, the per-shard pinned bounce buffer); no pageable async copy is added (ROCm pageable-copy bug).
- Bounded everything (TS §21 rule 8): transcode batches (≤ 32 blocks, the Phase 4 `DEMOTION_INFLIGHT`), ladder rewrites per tick (32), lineage keys (one per lossy copy, reclaimed with it), the `lossy_penalty` map (one entry per registered codec).
- Lab: `novanas` only; perf numbers on GPU 0 only, through `scripts/lab-bench.sh` holding `scripts/bench-lock.sh`; GPU 1 for functional tests; two-GPU runs through `scripts/lab-cluster.sh --bench-lock`; every GPU test has a hard timeout; a busy GPU or lock is identified (`kubectl -n turbine-ci get pods`, `pgrep -fa`) before waiting; a card held by another workload stops the run and goes to the coordinator; no new weights are downloaded (the phase runs on the Phase 1/2 BF16 models and 6a's checkpoints).
- Git: one commit per plan task, gate-clean; no push; nothing filed outside the repository.

From the interface contract and the work in flight (binding):

- Names in `.procoder/contract/interfaces.md` §9 (kernel C ABI), §11 (`turbine-kv`), §12 (`turbine-scheduler`), §17 (metrics), §24 (registries) and the §26 "Phase 6 additions" written by 6a are used verbatim; this phase appends its additions to §26 and adds the optional minor v2.11 (`TURBINE_ABI_VERSION` stays `2u`).
- Toolchain edition 2024, `rust-version = "1.97"`; `#[non_exhaustive]` on `EvictAction`, `EvictReason`, `KvDtypeChoice`, `KvFormatColumn`; config structs `#[serde(deny_unknown_fields, default)]`; metric labels from closed enums rendered with `as_str()`; time through `Arc<dyn Clock>`.
- Starting point: `phase-6a-quantization` closed (its exit task done, its rows in the support matrix); this phase branches from that tip as `phase-6b-kv-compression`. Task references written "6a Task N" point into `.procoder/plans/phase-6a-quantization.md`.
- Builds and tests run on novanas through `scripts/remote-cargo.sh`; every task ends with `scripts/gate.sh` printing `gate: ok`; GPU-facing tasks add `scripts/lab-test.sh novanas --tier quick`; every task that changes serving code ends with `scripts/lab-bench.sh --quick --model llama` and `--model olmoe` (GPU 0, under `scripts/bench-lock.sh`, `LABBOOK_SET=phase-6b-kv-compression`) and a row in `.procoder/perf-log.md` — one change, then measure.
- Lab runs fall under the standing novanas approvals (2026-09-25, 2026-09-26); no downloads; every soak is asked first. Sub-agents used for file-disjoint work run in worktrees and receive these lab rules verbatim.

## Task 1: `KvCodec` registry with `l0` and `fp8_e4m3` (CPU)

Files: `crates/turbine-core/src/support.rs` (`KvFormatColumn::{Tq4, Tq2}` with unsupported rows naming `phase-6b-kv-compression`; `TierFormatRefusal`, `TIER_FORMAT_REFUSALS` with `tq4`/`tq2` unsupported; `check_tier_format`), `crates/turbine-kv/src/codec/mod.rs` (trait, registry, conformance), `crates/turbine-kv/src/codec/l0.rs`, `crates/turbine-kv/src/codec/fp8_e4m3.rs` (new), `crates/turbine-kv/src/lib.rs`, `crates/turbine-kv/tests/registry_conformance.rs` or the crate's existing conformance module (`kv_codecs`), `docs/extending/kv-format.md` (new), `docs/extending/README.md` (index), `crates/turbine-model/tests/docs_extending.rs` (page list)
Interfaces:

- `trait KvCodec` and `CodecParams` as in the spec; `fn registry() -> &'static Registry<dyn KvCodec>` named `"kv_format"`; `fp8_e4m3` from a BF16 source uses per-layer scales = the L0 scales if L0 is FP8, else per-block-per-layer absmax/448 stored in the slot header (so a BF16 L0 demoting to FP8 needs no calibration)
  Covers: spec S-1 (CPU); AC `codec::tests`, `docs_extending`
  Depends on: 6a Task 24

- [ ] Write failing tests `codec::tests::{registry_lists_codecs, l0_is_identity, fp8_from_bf16_within_bound, fp8_from_fp8_is_identity}` and the conformance suite. Run: `scripts/remote-cargo.sh test -p turbine-kv codec` — expect FAIL
- [ ] Write failing test `support::tests::baseline_rows_present` (extend): `tq4`/`tq2` KV columns and tier formats resolve unsupported naming `phase-6b-kv-compression`. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect FAIL
- [ ] Implement; write `docs/extending/kv-format.md` (files, registry entry, CPU codec, GPU transcode op, suite command, lab checks, pitfalls: tier ordering, lineage keys, bit-exact decode).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv` and `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): KV codec registry with l0 and fp8_e4m3`

## Task 2: Per-tier format configuration keys

Files: `crates/turbine-core/src/config/kv.rs` (`cpu.format`, `nvme.format`, `lossless_tail_blocks`, `lossy_reuse`, `lossy_penalty`, `ladder.*`), `crates/turbine-core/src/config/mod.rs` (tests `phase6_keys`), `crates/turbine-server/src/startup.rs` (codec names validated against the `kv_format` registry and `check_tier_format`), `examples/turbine.yaml` (commented defaults)
Interfaces:

- keys, defaults and validation exactly as the spec's configuration table, including `kv.dtype: tq4|tq2` (accepted by the parser now; refused at startup with `kv_tq_unavailable` until Task 12's attention exists) and `kv.ladder.l0`
  Covers: spec S-2, S-9; AC `phase6_keys`, `--check-config` criterion
  Depends on: Task 1

- [ ] Write failing test `config::tests::phase6_keys`. Run: `scripts/remote-cargo.sh test -p turbine-core config::tests::phase6_keys` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core`; `scripts/remote-cargo.sh run -p turbine-server -- --config examples/turbine.yaml --check-config --set kv.dtype=fp8_e4m3 --set kv.cpu.format=fp8_e4m3` — expect `config ok`; `--set kv.dtype=int8` — expect exit 2 naming `kv.dtype`
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- Note (lead, 2026-09-29): tier format names are registry-driven strings validated at startup against `kv_format` (ordering from codec metadata), not a core enum. Progressive gating: the S-9 `--check-config` criterion with a `tq4` / `tq2` tier exits 2 until those rows turn `experimental` (Task 9) — it is a phase-end criterion; a lossy tier or `kv.ladder.enabled` exits 1 (`kv_transcode_unavailable` / `kv_tq_unavailable`) until Tasks 5 and 12 land.
- [ ] Commit: `feat(core): per-tier KV format and lossy-reuse configuration`

## Task 3: Per-tier demotion and promotion through codecs (host path)

Files: `crates/turbine-kv/src/tier/mod.rs` (`KvLocation.format`), `crates/turbine-kv/src/tier/l1.rs`, `crates/turbine-kv/src/tier/l2.rs` (slot size from the codec; slab header v2), `crates/turbine-kv/src/hierarchy.rs` (demotion picks the tier format except the lossless tail; promotion decodes), `crates/turbine-kv/src/transfer.rs` (`TransferRequest.codec`), `crates/turbine-server/src/kv_orchestrator.rs` (`CopyDevice::Sync` path calls `encode_cpu` / `decode_cpu`), `crates/turbine-scheduler/tests/kv_sim.rs` (`per_tier_formats`)
Interfaces:

- `KvLocation { tier, slot, format: &'static str }`; `HierarchyConfig.{l1_format, l2_format, lossless_tail_blocks}`; the lossless tail is computed from the sequence's block table at `request_done` / session demotion (blocks within the last N full blocks are flagged `tail`)
  Covers: spec S-1, S-2 (host); AC `kv_sim per_tier_formats`
  Depends on: Task 2

- [ ] Write failing test `kv_sim per_tier_formats`. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim per_tier_formats` — expect FAIL
- [ ] Implement; every existing `turbine-kv` and `kv_sim` test passes unchanged with `l0` formats.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv -p turbine-scheduler -p turbine-server` — expect PASS (including `engine::r#loop` `l2_round_trip_matches_cold`)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): lower tiers store blocks in their configured format`

## Task 4: Lossy lineage, opt-out, lossy token counts and the planner penalty

Files: `crates/turbine-kv/src/identity.rs` (`lossy_key`), `crates/turbine-kv/src/directory.rs` (`Lineage`, lookup order, opt-out cut), `crates/turbine-kv/src/planner.rs` (`lossy_penalty`, `allow_lossy`), `crates/turbine-kv/src/hierarchy.rs` (`PrefixAttach.lossy_tokens`, publishing over lossy prefixes), `crates/turbine-kv/src/metrics.rs` (lossy counters), `crates/turbine-api/src/openai/…` (header `x-turbine-kv-lossy`, `usage.prompt_tokens_details.lossy_cached_tokens`), `crates/turbine-core/src/request.rs` (`Usage.lossy_cached_tokens`, `RequestKvPolicy`), `crates/turbine-server/src/engine/requests.rs`, `crates/turbine-scheduler/tests/kv_sim.rs` (`lossy_lineage_never_reaches_opted_out`), `crates/turbine-api/tests/api.rs` (`kv_metrics_bounded`)
Interfaces:

- `lossy_key(key, format, seed) = BLAKE3("lossy" ‖ key ‖ format ‖ seed)[..16]`; a block computed with any lossy ancestor gets `Lineage::Lossy` and its key chains from the lossy parent key; `KvDirectory::lookup(.., allow_lossy: bool)`
  Covers: spec S-3; AC `lossy_lineage_never_reaches_opted_out`, `kv_metrics_bounded`
  Depends on: Task 3

- [ ] Write failing tests. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim lossy_lineage` and `scripts/remote-cargo.sh test -p turbine-api --test api kv_metrics_bounded` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv -p turbine-scheduler -p turbine-api -p turbine-server` — expect PASS
- [ ] Mutation check (do not commit): make `lookup` ignore `allow_lossy` — expect `lossy_lineage_never_reaches_opted_out` to FAIL; revert.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): lossy lineage keys, per-request opt-out and lossy token counts`

## Task 5: Kernel ABI v2.11 `kv_transcode` and the FP8 transcode on HIP

Files: `kernels/include/turbine_kernels.h` (v2.11 group), `crates/turbine-kernels/src/ffi.rs` (`V211Symbols`, depends on v2.9 and v2.5), `crates/turbine-kernels/src/ops/mod.rs` (`OpKind::KvTranscode`, config, trait), `crates/turbine-kernels/src/cpu/kv_transcode.rs` (CPU provider calls a codec function table passed in from the server, keeping `turbine-kernels` free of `turbine-kv`), `kernels/rocm/src/kv_transcode.hip` (FP8 encode/decode; the TurboQuant slots added by Task 8), `kernels/rocm/src/impl_table.cpp`, `kernels/rocm/src/abi_minor.cpp` (10), `crates/turbine-server/src/kv_orchestrator.rs` (`CopyStreamBackend`: encode into a device staging buffer, then the existing pinned copies; promotion: copy small bytes into a device staging buffer, decode into the L0 page), `.procoder/ask/decisions.md` (entry "P6b: KV transcode — provider evaluation": CK elementwise/transform, own), `crates/turbine-kernels/tests/hip_ops.rs` (`kv_transcode_matches_cpu`), `crates/turbine-server/tests/kv_gpu.rs` (`nvme_round_trip_fp8_tier`)
Interfaces:

- C and Rust shapes as the spec's Interfaces; one staging buffer of `DEMOTION_INFLIGHT` × the largest encoded block per shard, allocated at startup when a lower-tier format is not `l0`
  Covers: spec S-1 (GPU); AC `optional_groups_v211`, `kv_transcode_matches_cpu` (FP8); when it lands, `fp8_e4m3` as a lower-tier format gets an `experimental` entry in `TIER_FORMAT_REFUSALS` (startup stops refusing it with `kv_transcode_unavailable`), `supported` after Task 6
  Depends on: Task 4

- [ ] Evaluate transcode providers (short: CK `elementwise` / `batched_transpose` building blocks vs own) and write the entry.
- [ ] Write failing tests: `ffi::tests::optional_groups_v211`, lab `hip_ops::kv_transcode_matches_cpu` (FP8), lab `kv_gpu::nvme_round_trip_fp8_tier` (L2 `fp8_e4m3` from BF16 L0 within the codec bound; `l0` still bit-exact). Run: `scripts/remote-cargo.sh test -p turbine-kernels ffi::tests` — expect FAIL; `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops kv_transcode` — expect FAIL
- [ ] Implement.
- [ ] Run the three — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS; `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): ABI v2.11 KV transcode with FP8 on the demotion path`

## Task 6: Per-tier FP8 lab proof

Files: `scripts/lab/phase6-novanas-llama.yaml` (commented per-tier example), `tests/eval/llama-3.2-3b-instruct/turbine-l1-fp8.json`, `.procoder/perf-log.md`, `crates/turbine-server/tests/kv_gpu.rs` (`lossy_tier_reuse` with FP8 L1)
Interfaces:

- no new interface; measures S-1 … S-3 with the FP8 codec
  Covers: spec S-1 … S-3, S-8 (FP8 tier)
  Depends on: Task 5

- [ ] Write failing lab test `kv_gpu::lossy_tier_reuse` (FP8 L1 from BF16 L0: within the golden token rule, `lossy_cached_tokens` > 0; with `x-turbine-kv-lossy: deny` bit-equal to cold). Run: `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu lossy_tier_reuse` — expect FAIL, then implement any gap — expect PASS
- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama --golden16 -- --set kv.cpu.format=fp8_e4m3 --set kv.cpu.max_bytes=4GiB` — expect PASS; the Phase 4 multi-turn profile with `kv.cpu.format` `fp8_e4m3` vs `l0` on the same L1 bytes (commands of the spec's TurboQuant lab criterion) — record `cached_tokens_ratio` and L1 blocks per GiB; eval-compare at 0.01 — expect exit 0.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(kv): per-tier FP8 lab proof`

## Task 7: TurboQuant CPU codec (`tq4`, `tq2`)

Files: `crates/turbine-kv/src/codec/turboquant/mod.rs` (codec, layout), `crates/turbine-kv/src/codec/turboquant/hadamard.rs` (randomized fast Walsh–Hadamard), `crates/turbine-kv/src/codec/turboquant/codebook.rs` (Lloyd–Max codebooks as constants plus the generator used by the test), `crates/turbine-kv/src/codec/turboquant/qjl.rs` (1-bit residual projection through a seeded Gaussian `S`), `crates/turbine-kv/src/codec/mod.rs` (registry adds `tq4`, `tq2`), `crates/turbine-core/src/support.rs` (`TIER_FORMAT_REFUSALS`: `tq4`/`tq2` `experimental`)
Interfaces:

- per token-head vector of 128: signs `s = rademacher(seed, layer, head, kind)`, `y = H·(s ⊙ x) / √128`, `norm = ‖x‖` (BF16), codes = nearest codebook entry of `y_i·√128 / norm`; K residual `r = y − ŷ`, QJL signs of `S·r` with `S` a 128 × 128 Gaussian matrix seeded per (layer, head, kind) (SplitMix64 + Box–Muller in F64 rounded to F32), residual norm (BF16); decode adds the paper's estimate of the residual from the signs and its norm (`p6b-groundwork` `qjl.rs`); decode inverts; `seed` = first 8 bytes of the namespace key
  Covers: spec S-4 (CPU); AC `codec::turboquant::tests`
  Depends on: Task 1

- [ ] Write failing tests `codec::turboquant::tests::{hadamard_orthonormal, codebooks_reproduce, k_inner_product_unbiased, v_mse_bound_4bit, v_mse_bound_2bit, layout_round_trip}` and the registry conformance for both codecs. Run: `scripts/remote-cargo.sh test -p turbine-kv codec::turboquant` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv` — expect PASS
- [ ] Mutation check (do not commit): drop the QJL residual term in K decode — expect `k_inner_product_unbiased` to FAIL; revert.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): TurboQuant tq4 and tq2 codecs (CPU reference)`

## Task 8: TurboQuant GPU transcode — evaluation and implementation

Files: `.procoder/ask/decisions.md` (entry "P6b: TurboQuant transcode — provider evaluation": llama.cpp / vLLM / SGLang TurboQuant or QJL HIP kernels at their current commits, CK building blocks, own), `kernels/rocm/src/kv_transcode_tq.hip` (encode/decode per the pick), `kernels/rocm/src/impl_table.cpp`, `crates/turbine-kernels/tests/hip_ops.rs` (`kv_transcode_matches_cpu` TurboQuant cases)
Interfaces:

- codebooks and seeds passed through `turbine_kv_transcode_desc`; decode bit-exact to the CPU codec; encode ties documented and counted
  Covers: spec S-4 (GPU); AC `kv_transcode_matches_cpu`
  Depends on: Tasks 5, 7

- [ ] Evaluate; write the entry.
- [ ] Write failing lab cases; run `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops kv_transcode` — expect FAIL
- [ ] Implement; run — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): TurboQuant KV transcode`

## Task 9: TurboQuant lab proof

Files: `tests/eval/llama-3.2-3b-instruct/{turbine-l1-tq4.json,turbine-l1-tq2.json}`, `tests/eval/olmoe-1b-7b-0125-instruct/{…}`, `crates/turbine-core/src/support.rs` (`TIER_FORMAT_REFUSALS` → `supported` for the passing codecs), `.procoder/perf-log.md`, `crates/turbine-server/tests/kv_gpu.rs` (`lossy_tier_reuse` with `tq4`)
Interfaces:

- no new interface; decides `tq4` / `tq2` `supported` or `experimental`
  Covers: spec S-4, S-8; AC TurboQuant lab criterion, `lossy_tier_reuse`
  Depends on: Task 8

- [ ] Run the spec's TurboQuant lab criterion for Llama and OLMoE, `tq4` then `tq2` (GPU 0, bench lock): lab-bench golden16 with `kv.cpu.format`, the multi-turn profile against `lab-serve.sh`, `/turbine/v1/kv` capacity, eval-compare — record every number in the perf log and labbook.
- [ ] `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu lossy_tier_reuse` with the `tq4` case — expect PASS
- [ ] Flip the passing codecs; `support::tests` updated. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): TurboQuant lower-tier formats gated on gfx1201`

## Task 10: Mixed-format / TurboQuant paged attention — provider evaluation

Files: `.procoder/ask/decisions.md` (entry "P6b: mixed-format / TurboQuant paged attention — provider evaluation (kernel reuse rule)"), `kernels/rocm/tools/attn_eval.cpp` (new tool target, off by default: builds each candidate for gfx1201, checks it against a host reference, times Llama and OLMoE decode and prefill shapes), `kernels/rocm/CMakeLists.txt` (tool target)
Interfaces:

- candidates: llama.cpp HIP flash attention with q4_0 / q8_0 KV (its dequant-on-load inner loop as a template for TurboQuant decode), CK FMHA pagedkv / split-KV with FP8 KV (6a Task 23's instances), vLLM / aiter ROCm paged attention with quantized KV (gfx12 paths), the Turbine paged kernel extended with per-block tags; recorded per candidate: builds, correctness against `cpu::tq_attention` once Task 11 lands (until then against the dequantize-then-attend host reference), decode µs at c1/c16 shapes, and the pick (adapt or own)
  Covers: spec S-5 (evaluation); AC "P6b: mixed-format / TurboQuant paged attention" entry
  Depends on: Task 9

- [ ] Run the harness on GPU 0 under `scripts/bench-lock.sh` with a 30-minute timeout (identify any holder first); write the decisions entry with the table and the pick.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs(decisions): mixed-format and TurboQuant paged attention provider evaluation`

## Task 11: L0 page classes, block format tags and the CPU TurboQuant attention

Files: `crates/turbine-kv/src/pool.rs` (`PageClass`, per-format classes grown in slabs, `allocate_in`, `free` returning empty slabs, byte accounting), `crates/turbine-kv/src/block_table.rs` or the file holding block-table entries (`(BlockId, format)`), `crates/turbine-kernels/src/cpu/tq_attention.rs` (new: reference prefill and decode over mixed block tables, rotated-domain and decode-then-attend formulations), `crates/turbine-kernels/src/cpu/paged.rs` (paged append encodes by the block's tag through a codec function table passed in by the caller), `crates/turbine-kernels/src/ops/mod.rs` (`AttentionPagedDesc.block_formats`, `tq_params`), `crates/turbine-model/src/executor/decoder/mod.rs` (passes tags and params), `crates/turbine-server/src/kv_orchestrator.rs` (`kv.dtype: tq4|tq2` on the CPU backend), `crates/turbine-model/tests/tiny_model.rs` (`tq_kv_matches_reference`)
Interfaces:

- `pub struct PageClass { pub format: &'static str, pub page_bytes: u64 }`; `BlockPool::allocate_in(&mut self, format: &str, n: usize) -> Result<Vec<BlockId>, PoolError>`; `BlockPool::format_of(BlockId) -> &'static str`
- `turbine_kernels::cpu::tq_attention::{prefill, decode}` with the same shapes as the CPU paged attention plus `block_formats: &[u8]` and `&TqParams`
  Covers: spec S-5 (host); AC `cpu::tq_attention::tests`, `tq_kv_matches_reference`, `pool::tests::page_classes`
  Depends on: Task 10
  Amended (lead, 2026-09-30): the block-table entries `(BlockId, format)` fed from the scheduler and the decoder's addressing of pages in non-base classes move to Task 17 — until the ladder's L0 step every L0 block is in the pool's base format, so Task 11 passes `block_formats` empty (uniform) from the decoder and proves mixed tables at the kernel level (`cpu::tq_attention::tests::mixed_block_table`, `cpu::paged::tests::mixed_formats_append_and_read_by_tag`).

- [ ] Write failing tests `cpu::tq_attention::tests::{matches_decoded_attention, rotated_equals_decoded, mixed_block_table}`, `pool::tests::page_classes` and `tiny_model tq_kv_matches_reference`. Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::tq_attention` and `scripts/remote-cargo.sh test -p turbine-kv pool` and `scripts/remote-cargo.sh test -p turbine-model --test tiny_model tq_kv` — expect FAIL
- [ ] Implement; every existing pool, `kv_sim` and `tiny_model` test passes unchanged with one class.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv -p turbine-kernels -p turbine-model -p turbine-scheduler -p turbine-server` — expect PASS
- [ ] Mutation check (do not commit): read every block as `bf16` regardless of its tag — expect `mixed_block_table` to FAIL; revert.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): per-format L0 page classes, block format tags and the CPU TurboQuant attention`

## Task 12: Mixed-format paged attention on HIP (ABI v2.11)

Files: `kernels/include/turbine_kernels.h` (v2.11: `block_formats`, `turbine_tq_params`, `TURBINE_DTYPE_TQ4` 18, `TURBINE_DTYPE_TQ2` 19), `crates/turbine-kernels/src/ffi.rs` (desc fields read at minor ≥ 10), `kernels/rocm/src/paged_attention_mixed.*` (the Task 10 pick: an adapted provider kernel or own; TurboQuant append encoding through Task 8's codec), `kernels/rocm/src/impl_table.cpp`, `crates/turbine-kernels/tests/hip_ops.rs` (`paged_mixed_matches_cpu`), `scripts/lab-test.sh` (timing variant into `SLOW_TESTS`)
Interfaces:

- implementation name from the pick (e.g. `turbine_hip_mixed`); `supports` requires head_dim 128 and block_tokens a multiple of 16; BF16-only block tables keep choosing the existing CK implementations (no change to the BF16 path)
  Covers: spec S-5 (GPU); AC `paged_mixed_matches_cpu`
  Depends on: Task 11

- [ ] Write failing lab test `hip_ops::paged_mixed_matches_cpu` (prefill and decode, tables mixing all four formats, Llama and OLMoE head layouts). Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops paged_mixed` — expect FAIL
- [ ] Implement; run — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Lab: `scripts/lab-bench.sh --quick --model llama` and `--model olmoe` — BF16 KV unchanged within the no-regression bound; perf-log row.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): mixed-format paged attention reading TurboQuant blocks in the rotated domain`

## Task 13: TurboQuant in L0 — lab proof and decode ITL

Files: `tests/eval/llama-3.2-3b-instruct/{turbine-l0-tq4.json,turbine-l0-tq2.json}`, `tests/eval/olmoe-1b-7b-0125-instruct/{turbine-l0-tq4.json,turbine-l0-tq2.json}`, `crates/turbine-core/src/support.rs` (`KvFormatColumn` `tq4`/`tq2` rows `supported` or `experimental`), `.procoder/perf-log.md`
Interfaces:

- no new interface; decides the L0 TurboQuant rows
  Covers: spec S-5 (gates), S-8; AC S-5 lab criterion
  Depends on: Task 12

- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama --golden16 -- --set kv.dtype=tq4`, then `tq2`, then the same for `--model olmoe` — expect golden1/golden16 PASS under the batched bounds; record L0 blocks from `/turbine/v1/kv` (targets ≥ 3.5 × / ≥ 6 ×) and the c1 (`scripts/lab-bench.sh --quick` with the bench at concurrency 1 through `turbine-bench --concurrency 1 --requests 32 --max-tokens 256 --ignore-eos`) and c16 decode ITL p50 against the BF16 KV runs; eval-compare at 0.01 — expect exit 0.
- [ ] Flip the passing rows; update `baseline_rows_present`. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect PASS
- [ ] Report the ITL numbers to the coordinator (milestone).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): TurboQuant L0 KV gated on gfx1201 (golden, eval, ITL recorded)`

## Task 14: `EvictAction` and the ladder decision in the eviction policy

Files: `crates/turbine-kv/src/policy/mod.rs` (`EvictAction`, `LadderContext`, default `action`), `crates/turbine-kv/src/policy/cost_aware.rs` (ladder rule), `crates/turbine-kv/src/policy/lru.rs` (default), `crates/turbine-kv/src/metrics.rs` (`EvictReason::{Compressed, LadderFloor}`), `docs/extending/eviction-policy.md` (action, ladder, pitfalls)
Interfaces:

- as the spec's Interfaces; rung order from `kv.ladder.max_format` and the codec registry's lossiness order (`l0` < `fp8_e4m3` < `tq4` < `tq2`); compression starts at YELLOW even when no tier is full (user decision 2026-09-29, "Start at YELLOW earlier"): the `p6b-groundwork` rule "a tier acts only when it is the lowest and about to drop, or above high water" becomes "at YELLOW or above the lowest enabled tier acts; at any non-GREEN state a tier about to drop or above high water acts", with `ladder_actions` updated (a YELLOW case with free room compresses one rung); YELLOW depth (user decision 2026-09-29, "Compress only until GREEN"): at YELLOW, with `must_leave` false and the tier not above high water, the lowest tier compresses only while `ctx.fill + ctx.demand > low_water` (`LadderLimits.low_water`, `LadderContext.demand`), so `ladder_actions` also has: YELLOW with `fill + demand ≤ low_water` keeps; YELLOW with demand pushing it over low water compresses; GREEN after YELLOW keeps; ORANGE with room still compresses (as built); a repeated sweep at steady YELLOW with room never reaches `tq2`
  Covers: spec S-6 (policy); AC `policy::tests::ladder_actions`
  Depends on: Task 9

- [ ] Write failing test `policy::tests::ladder_actions` and extend `registry_conformance::eviction_policies` (every policy's `action` never upgrades and never compresses at GREEN). Run: `scripts/remote-cargo.sh test -p turbine-kv policy` — expect FAIL
- [ ] Implement; update the docs page.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv` and `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): compress as a third eviction action with the ladder rule`

## Task 15: The ladder in the hierarchy, driven by the pressure controller

Files: `crates/turbine-kv/src/hierarchy.rs` (rung state per tier, hysteresis, bounded rewrites per tick through `apply_reclaim` / `tick`, new demotions at the current rung), `crates/turbine-kv/src/metrics.rs` (`turbine_kv_ladder_rung`, `turbine_kv_ladder_actions_total`), `crates/turbine-server/src/kv_orchestrator.rs` (rewrites: L1/L2 slot → device staging → transcode → back, through the pinned paths), `crates/turbine-server/src/status.rs` (ladder fields), `crates/turbine-scheduler/tests/kv_sim.rs` (`ladder_under_pinned_pressure`), `crates/turbine-scheduler/tests/fixtures/ladder_pressure_trace.json`, `crates/turbine-scheduler/tests/fixtures/ladder_expected_rungs.json`
Interfaces:

- `HierarchyConfig.ladder: Option<LadderConfig { max_format, high_water, low_water, dwell }>`; `KvHierarchy::ladder_tick(&mut self, pool, pressure: PressureLevel, now)`; log event `kv_ladder`
- YELLOW depth: `ladder_tick` fills `LadderContext.demand` from the bytes the YELLOW reclaim (`apply_reclaim` `DemoteIdle`, target the `kv_utilization` YELLOW threshold) would demote into the tier, refreshes `fill` after every rewrite and stops the tick once the policy keeps; the ≤ 32 rewrites per tick, ≥ 50 ms spacing and the `deescalate_dwell` hysteresis are unchanged. The committed trace holds a long steady-YELLOW stretch with room, and the expected sequence pins no rung change in it and none in the tick after GREEN returns
  Covers: spec S-6; AC `ladder_under_pinned_pressure`, `kv_metrics_bounded` (ladder families)
  Depends on: Task 14

- [ ] Write failing test `kv_sim ladder_under_pinned_pressure` with the committed trace and expected rung sequence. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim ladder` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv -p turbine-scheduler -p turbine-server -p turbine-api` — expect PASS; `overload_sim` unchanged
- [ ] Mutation check (do not commit): remove the dwell check on stepping up — expect `ladder_under_pinned_pressure` to FAIL on the rung sequence; revert. Then drop the `low_water` stop at YELLOW — expect it to FAIL on the steady-YELLOW stretch; revert.
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): pressure-driven compression ladder in L1 and L2`

Notes (lead, 2026-09-30, from the Task 15 handoff): the static-rank tier driver (`TierDriver`) does not handle `TransferPurpose::Compress`; the ladder stays refused at startup in that mode until a task teaches the driver `Compress` (add that task when the ladder is allowed there). Left for after the Task 4 + Task 15 merge: `/turbine/v1/status` `quantization.ladder` and the `/turbine/v1/kv` tier `rung` (accessor `KvHierarchy::ladder_rung`), and the AC's opted-out identical-output check in `ladder_under_pinned_pressure`. `floor_evict` actions carry `to="evict"`.
Step-up (user decision 2026-09-30, option A): a rung relaxes only at GREEN after `dwell` below low water; the t4 + t15 merge updates `ladder_tick`, regenerates `ladder_expected_rungs.json` and adds the assertion that no `rung_step_up` happens while the pinned state is not GREEN (mutation: drop the GREEN check → the test FAILS).

## Task 16: Ladder lab proof and soak

Files: `scripts/lab/phase6-novanas-ladder.yaml` (small L1/L2, `kv.ladder.enabled: true`), `.procoder/perf-log.md`, `tests/eval/llama-3.2-3b-instruct/turbine-ladder.json`
Interfaces:

- no new interface
  Covers: spec S-6, S-8, S-11; AC ladder soak criterion
  Depends on: Task 15

- [ ] Lab (GPU 0, bench lock): the multi-turn profile with the ladder config and with `kv.ladder.enabled: false` on the same bytes — record recomputed tokens, `cached_tokens_ratio`, rung metrics; eval-compare at 0.01 against BF16 KV — expect exit 0.
- [ ] Soak (ask the coordinator first): `scripts/overload-soak.sh novanas --duration 10m` with the ladder config — expect verdict pass and `turbine_kv_ladder_actions_total` > 0.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(kv): compression ladder lab proof and soak`

## Task 17: The ladder's L0 step

Files: `crates/turbine-kv/src/hierarchy.rs` (L0 as the ladder's top tier: rung state, candidates = unreferenced cached L0 blocks outside each sequence's lossless tail, not promoting or demoting), `crates/turbine-kv/src/policy/cost_aware.rs` (L0 in `LadderContext`), `crates/turbine-server/src/kv_orchestrator.rs` (in-place GPU recompression: base-class page → smaller-class page through `turbine_kv_transcode`, free the base page, lineage key), `crates/turbine-core/src/config/kv.rs` (`ladder.l0`), `crates/turbine-scheduler/tests/kv_sim.rs` (`ladder_l0_under_pinned_pressure`), `crates/turbine-scheduler/tests/fixtures/ladder_l0_expected_rungs.json`
Interfaces:

- Block-table entries carry the block's format (`(BlockId, format)`, moved here from Task 11) and the decoder passes them as `block_formats` and addresses each block in its page class's slabs (`BlockPool::format_of`)
- `LadderConfig.l0: bool`; `EvictAction::Compress { to }` now also for `TierId::L0`; `turbine_kv_ladder_rung{tier="l0"}` and `turbine_kv_ladder_actions_total{tier="l0",…}`; skip reason `no_room` when a page class cannot grow
  Covers: spec S-7; AC `ladder_l0_under_pinned_pressure`
  Depends on: Tasks 13, 16

- [ ] Write failing test `kv_sim ladder_l0_under_pinned_pressure`. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim ladder_l0` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv -p turbine-scheduler -p turbine-server -p turbine-core` — expect PASS; `ladder_under_pinned_pressure` (L1/L2) unchanged with `kv.ladder.l0: false`
- [ ] Mutation check (do not commit): allow referenced blocks as L0 candidates — expect `ladder_l0_under_pinned_pressure` to FAIL; revert.
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): the compression ladder reaches L0`

## Task 18: L0 ladder lab proof and soak

Files: `scripts/lab/phase6-novanas-ladder.yaml` (`kv.ladder.l0: true` variant), `.procoder/perf-log.md`, `tests/eval/llama-3.2-3b-instruct/turbine-ladder-l0.json`
Interfaces:

- no new interface
  Covers: spec S-7, S-8, S-11; AC S-7 lab criterion
  Depends on: Task 17

- [ ] Lab (GPU 0, bench lock): the multi-turn profile with `kv.ladder.l0: true` against `false` on the same budget — record recomputed tokens, `cached_tokens_ratio`, rung metrics, golden c1; eval-compare at 0.01 — expect exit 0.
- [ ] Soak (ask the coordinator first): `scripts/overload-soak.sh novanas --duration 10m` with the L0 ladder config — expect pass and `turbine_kv_ladder_actions_total{tier="l0"}` above 0.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(kv): L0 compression ladder lab proof and soak`

## Task 19: Phase exit (6b)

Files: `.procoder/perf-log.md` (6b summary), `AGENTS.md` (6b commands: per-tier formats, `kv.dtype: tq4|tq2`, lossy opt-out, the ladder), `.procoder/contract/interfaces.md` (§26, 6b part), `.procoder/specs/phase-6b-kv-compression.md` (criteria ticked with evidence), `.procoder/plans/phase-6-8-expansion.md` (track 1 closed)
Interfaces:

- no new interface
  Covers: spec S-11 phase-exit criterion; umbrella Task 10 (track close runbook) for 6b
  Depends on: Tasks 1–18

- [ ] `scripts/gate.sh --full` — expect `gate: ok`
- [ ] `scripts/lab-test.sh novanas --tier full` and `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` — expect exit 0
- [ ] `scripts/lab-bench.sh --golden16` for `llama` and `olmoe` with each KV format that turned `supported` — expect PASS; every performance target met or its miss recorded with the user's decision.
- [ ] `scripts/overload-soak.sh novanas --duration 10m` (asked first) with the ladder on (L1/L2 and L0) — expect pass.
- [ ] `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix --output json` — paste into the evidence; `scripts/track-gate.sh phase-7-model-families` — expect the order check to pass.
- [ ] Report to the coordinator: every KV format's status and every finding.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs: phase 6b exit — KV compression closed`
