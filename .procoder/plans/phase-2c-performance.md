# phase-2c-performance — implementation plan

Status: draft
Spec: .procoder/specs/phase-2c-performance.md

## Goal

Raise Turbine's output token throughput on one novanas R9700 for the Phase 2 baseline workload to at least 75% of same-day vLLM-ROCm (Llama-3.2-3B ≥ 553, OLMoE-1B-7B ≥ 401 tok/s) by measuring first, moving the default KV page size to 128 tokens so paged attention runs on Composable Kernel, then removing host overhead, reducing logits on the device, fusing and tuning kernels, capturing decode graphs, removing the MoE per-layer host sync and tuning batch shape — with golden correctness unchanged.

## Architecture

Measurement lands first: `turbine-server`'s engine loop times eight iteration stages (`turbine_engine_iteration_seconds{stage}`, `stages_ms` in `/turbine/v1/scheduler`), the executors gain `ForwardTimings` and an op profile mode, and `scripts/lab-perf.sh` measures Turbine and vLLM-ROCm the same day. The host path is then made allocation-free and parallel in `turbine-model`'s sampler and the engine's packing; the kernel ABI gains an additive, optional v2.1 (`add_rmsnorm`, `logits_reduce`, graph capture, `turbine_abi_minor`) resolved optionally by `turbine-kernels`, with CPU reference ops, so a v2.0 library (and the Phase 2b CUDA shim) keeps working. The default KV page size becomes 128 tokens so both paged-attention ops run on CK `fmha_fwd_pagedkv` (the Turbine paged kernel stays as the fallback for other page sizes). The HIP shim gets the new ops, hipBLASLt autotuning and a device-offset MoE path; the executors fuse projections, use `add_rmsnorm` and `logits_reduce`, and replay decode-only iterations from a bounded graph cache; the scheduler is tuned by a lab sweep into new Phase 2c lab configs.

## Constraints

From the spec (verbatim):

- TS §21 is binding: every optimised path in this phase has a correctness/reference test (CPU reference or the existing path) that runs before the path is enabled by default; a path is specialised only after the S-1/S-2 measurements show it matters; no Python in the serving path.
- Correctness bar unchanged: the golden tolerance of `tests/golden/<slug>/tolerance.json` (|Δ logprob| ≤ 0.15 for reference candidates with logprob > −2, ≤ 0.55 below) for both models under `--concurrency 16`; the Phase 1/2 HIP-vs-CPU op tolerances of `hip_ops`; the tiny-model HIP-vs-CPU bound of `hip_matches_cpu`. No tolerance file or bound is loosened in this phase.
- Determinism: within one process, identical inputs (same batch composition and seeds) give bitwise-identical outputs, with or without decode graphs. GEMM tuning picks algorithms by timing, so the algorithm choice — and outputs at BF16 noise level — may differ between server restarts; _execution.gemm_autotune_ false restores restart-stable choices. Seeded sampling with device reduction is reproducible run to run; it may differ from the host path only where the drawn uniform falls within float rounding of a CDF boundary (the device sums in a fixed parallel order).
- ABI rules (contract §9.2) hold for every v2.1 addition: no vendor identifiers in the header, all work on the context's compute stream, `turbine_stream_sync` the only blocking call besides the graph functions' capture boundaries, and no caller pointer retained beyond the call — except that a captured graph records the device pointers it was captured with; the executor owns those buffers for the graph's lifetime and destroys the graph before freeing them.
- Unsafe Rust and FFI stay in `crates/turbine-kernels/src` (contract §1.3); `turbine-scheduler` and `turbine-kv` gain no GPU or model dependency; no vendor GPU crate enters the dependency tree; no new external Rust dependency.
- Bounded resources (TS §21 rule 8): the graph cache (≤ 64 graphs), the GEMM tuning cache (one entry per distinct shape seen; ≤ 4,096 entries, then the heuristic answer without caching), and the sampler thread count (≤ 64) are all bounded; the startup budget of Phase 2 includes the decode-graph device buffers.
- Lab: novanas (192.168.10.203), one R9700 (`gfx1201`, 32 GB) per k3s Job, ROCm 7.14.1, weights under `/home/piwi/turbine-models/<slug>`; lab Jobs of this phase run under the standing novanas approval (decisions 2026-09-25 and 2026-09-26: while its R9700s are free; stop and ask the user if another workload holds `amd.com/gpu`; never evict another workload; always stop serve Jobs when done). The vLLM-ROCm comparison uses the Phase 2 Job `scripts/lab/novanas-vllm-job.yaml` with its pinned image digest, unchanged.
- Measurement discipline: throughput numbers compare the median of 3 runs per engine, measured the same day on the same card with the same `turbine-bench` binary and flags, each engine warmed up first with 16 requests of the same shape.

From the interface contract and the work in flight (binding):

- Names in `.procoder/contract/interfaces.md` §3, §7, §9, §10, §12, §14, §17, §20, §21 are used verbatim; the additions listed in the spec's Interfaces (ABI v2.1, `turbine_engine_iteration_seconds`, `turbine_decode_graph_total`, `turbine_logits_rows_total`, the five `execution.*` keys, `ForwardTimings`, `OpProfile`, `RowReduce`, `ReducedRow`, `lab-perf.sh`) are contract additions recorded with this plan. `TURBINE_ABI_VERSION` stays `2u` and `TURBINE_KERNELS_ABI_VERSION` stays `2`; the v3/v4/v5 majors of §9.1 are untouched.
- Work in flight: an agent is working on host overhead in the sampler and engine; its branch is merged into Tasks 6–7, whose tests are the acceptance for that work (write the missing tests, do not redo finished work). Branch `perf/p2c-kernels` (commit 687b0dd: the `hip_ops decode_op_timings` and `decode_forward_timing` lab benchmarks, faster `rope` and `silu_mul`) is merged in Task 3, and the op-level profiling builds on those benchmarks instead of adding new ones. The planned Turbine split-KV decode kernel and gathered-KV CK prefill for 16-token pages are dropped: the default page size becomes 128 (decision 2026-09-26, Task 5), where both paged ops run on CK `fmha_fwd_pagedkv`.
- Measured starting point (R9700, Llama-3.2-3B decode shapes, batch 16): synthetic 28-layer decode forward 17.0 ms at 128-token pages, of which GEMMs ≈ 13.8 ms (Q/O 397 GB/s, K/V 268, gate/up 544, down 422, LM head 600; peak ≈ 640) — fused QKV and gate/up GEMMs (Task 10) are the next GPU lever; a per-shape hipBLASLt autotune prototype gained only ≈ 0.4 ms, within noise (Task 12 keeps it, on by default, with low expected value). The engine sees ≈ 47 ms per forward against ≈ 34 ms synthetic at 16-token pages: ≈ 13 ms executor overhead (8.2 MB F32 full-vocabulary logits into pageable memory, per-op host work — Tasks 7, 9, 14), plus ≈ 115 ms host overhead per iteration (Tasks 6–7).
- Builds on `.procoder/plans/phase-2-serving-runtime.md` (engine thread, `BatchInput`/`SeqSlice`, ABI v2, `hip_ops paged_and_moe_ops`, the vLLM Job, `turbine-golden compare --concurrency`).
- Every task ends gate-clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`. Lab steps run under the standing novanas approval; if `amd.com/gpu` is held by another workload, stop and ask the user.

## Task 1: Iteration stage breakdown and forward timings

Files: `crates/turbine-server/src/engine/stages.rs` (new: `StageClock`, `IterationStages`), `crates/turbine-server/src/engine/loop.rs` (stage marks around each step of a turn), `crates/turbine-server/src/engine/requests.rs` (detokenize timing split out of `step`), `crates/turbine-server/src/metrics.rs` (`turbine_engine_iteration_seconds{stage}`), `crates/turbine-scheduler/src/scheduler.rs` (document field `stages_ms`), `crates/turbine-model/src/executor/mod.rs` (`ForwardTimings`, `last_timings`), `crates/turbine-model/src/executor/llama.rs` and `crates/turbine-model/src/executor/olmoe.rs` (record launch and device-wait), `crates/turbine-server/tests/tiny_server.rs` (test)
Interfaces:

- `pub enum Stage { Schedule, Prepare, Launch, DeviceWait, Sample, Detokenize, Emit, Complete }` with `fn as_str(self) -> &'static str` (the eight label values)
- `pub struct StageClock` with `fn start() -> StageClock`, `fn mark(&mut self, stage: Stage)` (attributes the time since the previous mark), `fn add(&mut self, stage: Stage, d: Duration)`, `fn finish(self) -> IterationStages`
- `pub struct IterationStages(pub [Duration; 8])` with `fn to_ms_map(&self) -> BTreeMap<&'static str, f64>`
- `#[derive(Clone, Copy, Default)] pub struct ForwardTimings { pub launch: Duration, pub device_wait: Duration }`; `ModelExecutor::last_timings(&self) -> ForwardTimings` (default method returning zeros)
- `LastIteration` (scheduler document) gains `stages_ms: BTreeMap<String, f64>`; `EngineLoop` publishes it with `turbine_engine_iteration_seconds{stage}` (buckets 50 µs × 2^k up to 1 s)
  Covers: S-1 AC `tiny_server iteration_stage_breakdown`
  Depends on: Phase 2 plan Tasks 15–17

- [ ] Write failing test `turbine-server --test tiny_server iteration_stage_breakdown`: after 8 concurrent tiny-model requests of 12 tokens, `/metrics` shows `turbine_engine_iteration_seconds_count{stage="<s>"}` > 0 for each of `schedule`, `prepare`, `launch`, `device_wait`, `sample`, `detokenize`, `emit`, `complete`; `/turbine/v1/scheduler` `last_iteration.stages_ms` has exactly those 8 keys, all ≥ 0, summing to `duration_ms` within 5% or 0.5 ms. Run: `cargo test -p turbine-server --test tiny_server iteration_stage_breakdown` — expect FAIL
- [ ] Implement the stage clock (one `Instant::now()` per mark, no allocation per iteration), the executor timings (launch = until the last op returns, device_wait = the synchronising logits copy) and the document field; `launch` and `device_wait` come from `last_timings()`, the rest of `forward` time is `prepare`.
- [ ] Run: `cargo test -p turbine-server --test tiny_server && cargo test -p turbine-scheduler && cargo test -p turbine-model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): per-stage engine iteration breakdown`

## Task 2: Phase 2c execution configuration keys

Files: `crates/turbine-core/src/config/mod.rs` (`ExecutionConfig` fields, validation), `crates/turbine-core/src/config/tests.rs` (test), `examples/turbine.yaml` (the five keys with comments)
Interfaces:

- `ExecutionConfig` gains `gemm_autotune: bool` (true), `decode_graphs: bool` (true), `device_sampling: bool` (true), `fused_ops: bool` (true), `sampler_threads: u32` (4, 1..=64)
- `ExecutionConfig::effective_sampler_threads(&self, available: usize) -> usize` = max(1, min(sampler_threads, available − 1))
  Covers: S-14 AC `config::tests::phase2c_execution_keys`
  Depends on: Phase 2 plan Task 2

- [ ] Write failing test `turbine-core config::tests::phase2c_execution_keys`: defaults are true/true/true/true/4; `execution.sampler_threads: 0` and `65` are rejected with messages naming `execution.sampler_threads`; `--set execution.decode_graphs=false` yields false; `effective_sampler_threads(2)` is 1 and `effective_sampler_threads(16)` is 4; `examples/turbine.yaml` loads. Run: `cargo test -p turbine-core config::tests::phase2c_execution_keys` — expect FAIL
- [ ] Implement the fields with serde defaults and the range check through the existing `invalid(key, reason)` helper.
- [ ] Run: `cargo test -p turbine-core` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-core): phase 2c execution switches`

## Task 3: Op-level forward profile and the lab profile test

Files: branch `perf/p2c-kernels` merged first (commit 687b0dd: `kernels/rocm/src/elementwise.hip` faster `rope` and `silu_mul`; `crates/turbine-kernels/tests/hip_ops.rs` `decode_op_timings` and `decode_forward_timing`), `crates/turbine-kernels/tests/hip_ops.rs` (extend those two benchmarks: OLMoE-1B-7B decode shapes, batches 1, 16 and 64, one `op_timings: <json>` line per case), `crates/turbine-model/src/executor/profile.rs` (new: `OpProfile`, `OpProfileEntry`), `crates/turbine-model/src/executor/llama.rs` and `crates/turbine-model/src/executor/olmoe.rs` (profile hooks around every registry op), `crates/turbine-model/src/executor/mod.rs` (re-export), `crates/turbine-model/tests/tiny_model.rs` (`op_profile_accounts_forward`), `crates/turbine-model/tests/perf.rs` (new, ignored `forward_profile`)
Interfaces:

- `#[derive(Serialize)] pub struct OpProfileEntry { pub op: String, pub r#impl: String, pub calls: u32, pub total_ms: f64 }`; `#[derive(Serialize, Default)] pub struct OpProfile { pub entries: Vec<OpProfileEntry> }` with `fn total_ms(&self) -> f64`
- `LlamaExecutor::set_profile(&mut self, on: bool)`, `LlamaExecutor::take_profile(&mut self) -> OpProfile`; the same two on `OlmoeExecutor`; with profile on every op is followed by `DeviceMemory::synchronize` and timed; with it off no timer or sync is added
- `hip_ops decode_op_timings` / `decode_forward_timing` (ignored, from 687b0dd, print only, no speed assertion): gain a `model` (`llama` | `olmoe`) and `batch` (1, 16, 64) dimension and print `op_timings: {"model","batch","block_tokens","ops":[{"op","impl","us_per_call"}],"forward_ms"}`
- `perf forward_profile` (ignored): the real-executor counterpart of `decode_forward_timing`, so their difference is the executor overhead (≈ 13 ms at the start); `require_backend("hip")`; reads `TURBINE_TEST_MODEL_DIR` and `TURBINE_TEST_MOE_MODEL_DIR`; cases `decode_b1_ctx768`, `decode_b16_ctx768`, `decode_b64_ctx768`, `prefill_2048`; prints `forward_profile: <json>` per model (spec §Data shape)
  Covers: S-2 AC `tiny_model op_profile_accounts_forward`; S-2/S-15 AC `perf forward_profile` (lab); S-2 AC `hip_ops decode_op_timings decode_forward_timing` (lab)
  Depends on: Task 1

- [ ] Merge `perf/p2c-kernels` (687b0dd) and extend `decode_op_timings` / `decode_forward_timing` to both models and batches 1, 16, 64 without adding a new benchmark. Run: `cargo test -p turbine-kernels --test hip_ops` — expect PASS with the benchmarks listed as ignored
- [ ] Write failing test `turbine-model --test tiny_model op_profile_accounts_forward`: on the tiny Llama and tiny OLMoE checkpoints (CPU provider) a decode forward of 3 sequences with profile on yields one entry per op kind used, `attention_decode_paged` calls = layers, `moe_route` calls = layers for OLMoE, `total_ms` ≤ forward wall time, logits bitwise equal to a run with profile off, and `take_profile()` after a profile-off run is empty. Run: `cargo test -p turbine-model --test tiny_model op_profile_accounts_forward` — expect FAIL
- [ ] Write the ignored test `turbine-model --test perf forward_profile` asserting every case lists every op kind the executor's requirements name and `profiled_ms` ≥ 0.8 × `forward_ms` (the unprofiled median of 5 runs). Run: `cargo test -p turbine-model --test perf` — expect the test listed as ignored
- [ ] Implement the profile hooks as one wrapper method per executor that every op call goes through.
- [ ] Run: `cargo test -p turbine-model --test tiny_model` — expect PASS
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- decode_op_timings decode_forward_timing --nocapture` — expect exit 0 and six `op_timings: ` lines (2 models × 3 batches); paste them into task evidence
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-model --test perf -- forward_profile --nocapture` — expect exit 0, `test forward_profile ... ok` and two `forward_profile: ` lines; paste both into task evidence as the Phase 2 baseline profile, next to the synthetic `forward_ms` of the same batch
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): op-level forward profile and lab profile test`

## Task 4: `lab-perf.sh`, serve `--set` passthrough and Phase 2c lab configs

Files: `scripts/lab-perf.sh` (new), `scripts/lab-serve.sh` (`--set` passthrough), `scripts/lab/novanas-serve-job.yaml` (extra-args placeholder rendered into the `turbine-server` command), `scripts/lab/phase2c-novanas-llama.yaml`, `scripts/lab/phase2c-novanas-olmoe.yaml` (new: Phase 2 values plus the five execution keys; S-12 tunes them in Task 15), `AGENTS.md` (Commands: `lab-perf.sh`, `--set` on `lab-serve.sh`)
Interfaces:

- `scripts/lab-perf.sh [--dry-run] novanas <llama|olmoe> [--runs <n>] [--skip-vllm] [--config <yaml>] [--set <dotted.key>=<value>]…` — exit 0 PASS, 1 FAIL/serve/bench failure, 2 usage; writes `target/lab-perf/<run-id>/{turbine-<i>.json,vllm-<i>.json,summary.json}`; last line `lab-perf: <model> turbine=<tok/s> vllm=<tok/s> ratio=<r> target=<553|401> verdict=<PASS|FAIL>` (with `--skip-vllm`: `vllm=<738|535>(recorded)`)
- `scripts/lab-serve.sh [--dry-run] novanas <config.yaml> [--set <k>=<v>]…` — each pair appended as `--set <k>=<v>` to the serve Job's `turbine-server` command line
- Medians and ratio computed with `jq` on the workstation (`output_token_throughput`, `itl_ms.p50`, `ttft_ms.p50`)
  Covers: S-13 AC (`bash -n` / `shellcheck` / `--dry-run`)
  Depends on: Phase 2 plan Tasks 20–21

- [ ] Write the failing check: `scripts/lab-perf.sh --dry-run novanas llama --set scheduler.max_batch_tokens=4096` must print, in order, `scripts/lab-serve.sh novanas scripts/lab/phase2c-novanas-llama.yaml --set scheduler.max_batch_tokens=4096`, one warm-up `turbine-bench … --requests 16 …`, three `turbine-bench --url http://192.168.10.203:18000 --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos --output json`, `scripts/lab-serve.sh novanas --stop`, then the same for `--vllm llama-3.2-3b-instruct` on port 18100. Run: `bash -n scripts/lab-perf.sh` — expect FAIL (file missing)
- [ ] Implement the script (strict mode, usage on bad arguments, `trap` that stops the serve Job it started) and the `--set` passthrough in `lab-serve.sh` and the serve Job template.
- [ ] Run: `bash -n scripts/lab-perf.sh && shellcheck scripts/lab-perf.sh scripts/lab-serve.sh && scripts/lab-perf.sh --dry-run novanas llama --set scheduler.max_batch_tokens=4096` — expect PASS with the commands above
- [ ] Run the Phase 2 baseline under the new tool: `scripts/lab-perf.sh novanas llama --runs 1` and `scripts/lab-perf.sh novanas olmoe --runs 1` — expect `verdict=FAIL` lines with Turbine near 92 tok/s for Llama; paste both summaries into evidence as the Phase 2c starting point
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(lab): lab-perf.sh same-day Turbine vs vLLM comparison`

## Task 5: Default KV page size 128

Files: `crates/turbine-core/src/config/mod.rs` (`KvConfig` default `block_tokens: 128`), `crates/turbine-core/src/config/tests.rs` (`example_config_loads` expects 128; new `default_block_tokens_is_128`), `crates/turbine-core/src/types.rs` (doc comment and `dtype_codes_and_kv_layout_sizes`: add the 128-token block size 14,680,064), `examples/turbine.yaml` (`kv.block_tokens: 128` with a comment naming the CK page-size rule), `scripts/lab/phase2-novanas-llama.yaml` and `scripts/lab/phase2-novanas-olmoe.yaml` (`block_tokens: 128`), `scripts/lab/phase2c-novanas-llama.yaml` and `scripts/lab/phase2c-novanas-olmoe.yaml` (128), `crates/turbine-model/tests/tiny_model.rs` (`BLOCK_TOKENS` 128; `requirements_and_workspace` expects `block_tokens=128`; `forward_rejects_invalid_batches` keeps a mismatching page size, 64; the pool sizes of the multi-sequence tests re-derived from `BLOCK_TOKENS`), `crates/turbine-model/tests/golden.rs` (`BLOCK_TOKENS` 128), `crates/turbine-server/tests/tiny_server.rs` (`diagnostics_shapes` expects 128; tests whose preemption or exhaustion depends on the block count set `kv.gpu.max_bytes` from the 128-token tiny block size), `crates/turbine-scheduler/src/scheduler.rs` and `crates/turbine-scheduler/src/sim/mod.rs` (tests keep the explicit 16 and add a 128-token params case), `crates/turbine-kv/src/pool.rs` and `crates/turbine-kv/src/document.rs` (tests keep 16 and add a 128-token case), `crates/turbine-server/src/model.rs` (startup INFO `event="paged_attention_fallback"` when the selected paged `_impl` is not `ck_tile_fmha_pagedkv` on HIP), `crates/turbine-kernels/tests/hip_ops.rs` (`paged_prefill_ck_128_matches_cpu`; `decode_op_timings` / `decode_forward_timing` already cover both page sizes), `AGENTS.md` (page-size note in Commands)
Interfaces:

- `KvConfig::default().block_tokens == 128`; validation unchanged (`1..=1024`, `scheduler.max_batch_tokens ≥ kv.block_tokens`)
- Consumes the Phase 2 HIP dispatch: `block_tokens % 128 == 0` → `ck_tile_fmha_pagedkv` for `attention_prefill_paged` and `attention_decode_paged`, else `turbine_hip` (no kernel change in this task)
- Startup log: `event="paged_attention_fallback" block_tokens=<n> impl=turbine_hip` (INFO, once) when a non-multiple of 128 is configured on HIP
- Test helper `paged_case(block_tokens: usize, q_lens: &[usize], cached: &[usize])` reused from the Phase 2 `hip_ops` paged helper
  Covers: S-6 AC `config::tests::default_block_tokens_is_128` / `example_config_loads`; S-6 AC `cargo test --workspace` default-page tests; S-6/S-15 AC lab `hip_ops`, `tiny_model hip_matches_cpu`, `golden logits_match_reference`; S-7 AC `hip_ops paged_prefill_ck_128_matches_cpu`
  Depends on: Task 3 (the `decode_op_timings` / `decode_forward_timing` benchmarks merged there), Phase 2 plan Task 18

- [ ] Write failing test `turbine-core config::tests::default_block_tokens_is_128`: a config without a `kv` section has `kv.block_tokens` 128; `--set kv.block_tokens=16` and `=1024` load, `=0` and `=1025` are rejected naming `kv.block_tokens`; `examples/turbine.yaml` and both `scripts/lab/phase2-novanas-*.yaml` load with 128. Run: `cargo test -p turbine-core config::tests::default_block_tokens_is_128` — expect FAIL
- [ ] Write the ignored test `turbine-kernels --test hip_ops paged_prefill_ck_128_matches_cpu`: ragged batches of prefill chunks 1, 17, 512 and 2,048 after 0, 100 and 1,000 cached tokens plus three decodes, both head shapes (24/8, 16/16), shuffled block tables; at `block_tokens` 128 the `_impl` is `ck_tile_fmha_pagedkv`, at 16 it is `turbine_hip`; outputs within the Phase 2 paged tolerance of the CPU reference and appended pages equal to the CPU pages. Run: `cargo test -p turbine-kernels --test hip_ops` — expect the test listed as ignored
- [ ] Implement: change the default and every shipped config to 128; move the default-page test constants and expectations to 128; add the 128-token cases to the scheduler, simulator and KV pool unit tests (block counts, partial-tail accounting, watermark); add the fallback INFO line.
- [ ] Run: `cargo test --workspace` — expect PASS
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- --nocapture` — expect exit 0 with `test paged_prefill_ck_128_matches_cpu ... ok`, `test paged_and_moe_ops ... ok` and the `decode_op_timings` lines showing `attention_decode_paged` ≈ 620 µs at 16-token and ≈ 115 µs at 128-token pages and `decode_forward_timing` ≈ 17 ms at 128; paste them into evidence
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model --test golden` — expect exit 0 with `test hip_matches_cpu ... ok` (max error ≈ 3.8e-6) and `test logits_match_reference ... ok` (16/16 prompts) for both models
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(turbine-core): default kv.block_tokens 128 so paged attention runs on CK`

## Task 6: Sampler fast paths and the reference sampler

Files: `crates/turbine-model/src/sampler.rs` (fast paths, `SamplerScratch`), `crates/turbine-model/src/testing/reference_sampler.rs` (new: the Phase 2 sampler verbatim as `ReferenceSampler`), `crates/turbine-model/src/testing/mod.rs` (declare it), `crates/turbine-model/tests/sampler_alloc.rs` (new: counting global allocator)
Interfaces:

- `pub struct SamplerScratch` (reused index and value buffers sized to the vocabulary on first use); `Sampler::sample_with(&mut self, logits: &mut [f32], mask: Option<&TokenMask>, scratch: &mut SamplerScratch) -> SampledToken`; `Sampler::sample` keeps its signature and uses a scratch owned by the sampler
- Fast paths: greedy without logprobs = `argmax` only; `log_sum_exp` only when `logprobs` is requested; top-k / top-p / `top_logprobs` candidates by `select_nth_unstable_by(by_value_desc)` on the smallest prefix that holds them, then sorting only that prefix; the uniform is drawn at the same point and count as before
- `pub struct ReferenceSampler` with the Phase 2 `new`, `sample`, `observe`, `state` signatures
  Covers: S-3 AC `sampler::tests::fast_paths_match_reference`; S-3 AC `sampler_alloc steady_state_sampling_does_not_allocate`
  Depends on: Phase 2 plan Task 12

- [ ] Write failing test `turbine-model sampler::tests::fast_paths_match_reference`: a `proptest` with 2,000 cases over vocabulary 1..=5000 (values with ties, NaN, −∞), temperature 0..2, `top_k` −1..=70, `top_p` 0.05..=1, `top_logprobs` 0..=20, penalties, `logit_bias`, a random token mask and a seed runs 8 steps through `Sampler` and `ReferenceSampler` and asserts equal token ids and logprobs within 1e-6. Run: `cargo test -p turbine-model sampler::tests::fast_paths_match_reference` — expect FAIL (no `ReferenceSampler` yet)
- [ ] Write failing test `turbine-model --test sampler_alloc steady_state_sampling_does_not_allocate`: with a `#[global_allocator]` that counts allocations, after one warm-up step on a 128,256-entry row, a greedy step without logprobs and a `temperature` 1.0 `top_p` 0.9 step allocate 0 times and a `top_logprobs` 5 step allocates exactly once. Run: `cargo test -p turbine-model --test sampler_alloc` — expect FAIL
- [ ] Copy the Phase 2 sampler into `testing/reference_sampler.rs` unchanged, then implement the fast paths and the scratch.
- [ ] Run: `cargo test -p turbine-model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(turbine-model): allocation-free sampler fast paths with reference equivalence`

## Task 7: Engine packing, single metadata upload, logits rows and parallel sampling

Files: `crates/turbine-model/src/executor/batch.rs` (one staging buffer, one H2D copy, rows only for `wants_logits`), `crates/turbine-model/src/executor/mod.rs` (`SeqSlice::wants_logits`), `crates/turbine-model/src/executor/llama.rs` and `crates/turbine-model/src/executor/olmoe.rs` (final norm and LM head over the yielding rows), `crates/turbine-model/src/testing/counting_memory.rs` (new: `CountingMemory` wrapper), `crates/turbine-server/src/engine/loop.rs` (reused packing vectors, `wants_logits`, parallel `sample`), `crates/turbine-server/src/engine/requests.rs` (split of sampling from event building so rows can be sampled on worker threads), `crates/turbine-model/tests/tiny_model.rs` (`one_metadata_upload_per_forward`), `crates/turbine-server/tests/tiny_server.rs` (`parallel_sampling_matches_serial`)
Interfaces:

- `SeqSlice` gains `pub wants_logits: bool`; `Logits.rows` = number of slices with `wants_logits`, in slice order
- `pub struct CountingMemory { inner: Arc<dyn DeviceMemory> }` implementing `DeviceMemory`, with `fn h2d_copies(&self) -> u64`, `fn d2h_copies(&self) -> u64`, `fn reset(&self)`
- Engine: rows are sampled with `std::thread::scope` over `ExecutionConfig::effective_sampler_threads(available_parallelism)` chunks of consecutive rows; each worker owns disjoint `&mut` request samplers; events are emitted afterwards on the engine thread in row order
  Covers: S-3 AC `tiny_model one_metadata_upload_per_forward`; S-3/S-14 AC `tiny_server parallel_sampling_matches_serial`
  Depends on: Tasks 1, 2, 6

- [ ] Write failing test `turbine-model --test tiny_model one_metadata_upload_per_forward`: on both tiny checkpoints with `CountingMemory` over the host backend, a forward with 2 prefill chunks (one non-final, `wants_logits` false) and 5 decodes makes exactly 1 metadata H2D copy and 1 D2H copy and returns 6 logits rows. Run: `cargo test -p turbine-model --test tiny_model one_metadata_upload_per_forward` — expect FAIL
- [ ] Write failing test `turbine-server --test tiny_server parallel_sampling_matches_serial`: 16 concurrent seeded requests (8 greedy with `top_logprobs` 3, 8 at `temperature` 0.8 and `top_p` 0.9, 24 tokens each) give identical tokens and logprobs with `execution.sampler_threads` 4 and 1. Run: `cargo test -p turbine-server --test tiny_server parallel_sampling_matches_serial` — expect FAIL
- [ ] Implement: one `Vec<u8>` staging buffer grown to the largest batch and reused, one `copy_h2d` into one device metadata buffer with fixed sub-offsets; the scheduler's `prefill_target` decides `wants_logits`; the parallel sampler split.
- [ ] Run: `cargo test -p turbine-model && cargo test -p turbine-server` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(turbine-server): single metadata upload, logits for yielding rows only, parallel sampling`

## Task 8: Kernel ABI v2.1 on the Rust side and CPU reference ops

Files: `kernels/include/turbine_kernels.h` (v2.1 block), `crates/turbine-kernels/src/ffi.rs` (`AddRmsnormDesc`, `LogitsReduceDesc`, optional symbol table), `crates/turbine-kernels/src/shim.rs` (`abi_minor`, optional resolution, `GraphHandle`, v2.1 provider methods), `crates/turbine-kernels/src/ops/mod.rs` (`OpKind::AddRmsnorm`, `OpKind::LogitsReduce`, configs, contexts, traits, `KernelProvider` defaults), `crates/turbine-kernels/src/registry.rs` (selection for the two ops), `crates/turbine-kernels/src/cpu/mod.rs` and `crates/turbine-kernels/src/cpu/math.rs` (reference `add_rmsnorm`, `logits_reduce`), `crates/turbine-kernels/stub/stub_shim.c` (a `-DTURBINE_STUB_V21` build exporting the v2.1 symbols), `crates/turbine-kernels/build.rs` or the test helper that builds the stubs (both variants), `crates/turbine-kernels/tests/abi_header_neutral.rs` (v2.1 names)
Interfaces:

- Header: `#define TURBINE_ABI_MINOR 1u`, `uint32_t turbine_abi_minor(void)`, `TURBINE_OPTION_GEMM_AUTOTUNE 1`, `TURBINE_OPTION_GEMM_TUNED_SHAPES 2`, `int32_t turbine_ctx_set_option(turbine_ctx *ctx, int32_t option, int64_t value)`, `int32_t turbine_ctx_get_option(turbine_ctx *ctx, int32_t option, int64_t *out)`, `turbine_add_rmsnorm_desc`, `turbine_logits_reduce_desc` and their trios, `typedef struct turbine_graph turbine_graph`, `turbine_graph_begin/end/launch/destroy` — exactly as spec §Kernel ABI v2.1
- `ShimLibrary::abi_minor(&self) -> u32` (0 when the symbol is absent); v2.1 symbols held as `Option<Symbol<…>>`; `ShimContext::set_option(&self, option: i32, value: i64) -> Result<(), KernelError>`, `ShimContext::get_option(&self, option: i32) -> Result<i64, KernelError>` (`Unsupported` without the symbols)
- `pub struct AddRmsnormConfig { pub dtype: DType, pub dim: u32 }`; `pub struct AddRmsnormContext<'a> { pub residual: TensorView<'a>, pub x: TensorView<'a>, pub weight: TensorView<'a>, pub out: TensorView<'a>, pub eps: f32 }`; `pub trait AddRmsnormKernel { fn supports(&self, cfg: &AddRmsnormConfig) -> bool; fn implementation(&self, cfg: &AddRmsnormConfig) -> String; fn execute(&self, ctx: &mut AddRmsnormContext<'_>) -> Result<(), KernelError>; }`
- `pub struct LogitsReduceConfig { pub vocab: u32, pub top_n: u32 }`; `pub struct LogitsReduceContext<'a> { pub logits, temperature, uniform, mode, top_ids, top_values, lse, sampled, sampled_logit: TensorView<'a>, pub rows: u32 }`; `pub trait LogitsReduceKernel` with the same three methods
- `KernelProvider::add_rmsnorm(&self) -> Option<&dyn AddRmsnormKernel>` and `KernelProvider::logits_reduce(&self) -> Option<&dyn LogitsReduceKernel>` (default `None`)
- `ShimContext::graph_begin(&self) -> Result<(), KernelError>`, `ShimContext::graph_end(&self) -> Result<GraphHandle, KernelError>`, `ShimContext::graph_launch(&self, g: &GraphHandle) -> Result<(), KernelError>`, `impl Drop for GraphHandle` (calls `turbine_graph_destroy`); all return `KernelError::Unsupported` when the library lacks the symbols
  Covers: S-5 AC (`abi_header_neutral`, `shim::tests::v21_symbols_optional`); S-4 AC `cpu::tests::logits_reduce_matches_sampler`; writes `cpu::tests::add_rmsnorm_equals_add_then_rmsnorm` (part of the S-9 AC closed in Task 11)
  Depends on: Phase 2 plan Task 8, Task 6

- [ ] Write failing test `turbine-kernels shim::tests::v21_symbols_optional`: the plain stub loads with `abi_minor() == 0`, `add_rmsnorm()`/`logits_reduce()` `None` and `graph_begin` → `Unsupported`; the `TURBINE_STUB_V21` stub reports minor 1, both ops `Some`, `graph_begin` Ok and `set_option(1, 1)` then `get_option(1)` = 1; the plain stub's `set_option` is `Unsupported`. Run: `cargo test -p turbine-kernels shim::tests::v21_symbols_optional` — expect FAIL
- [ ] Write failing test `turbine-kernels cpu::tests::logits_reduce_matches_sampler`: seeded rows of 50,304 and 128,256 values (with ties and NaN) give the `ReferenceSampler`'s top-20 ids and order, lse within 1e-6, and in categorical mode (temperature 0.7, uniform from a seeded ChaCha8) the same token as the reference inverse-CDF draw (the test re-implements the sequential id-order draw). Run: `cargo test -p turbine-kernels cpu::tests::logits_reduce_matches_sampler` — expect FAIL
- [ ] Write failing test `turbine-kernels cpu::tests::add_rmsnorm_equals_add_then_rmsnorm`: for rows 1, 7, 64 and dims 64, 3,072 the CPU `add_rmsnorm` leaves `residual` and `out` bitwise equal to `add` then `rmsnorm`. Run: `cargo test -p turbine-kernels cpu::tests::add_rmsnorm_equals_add_then_rmsnorm` — expect FAIL
- [ ] Extend `abi_header_neutral` to require `TURBINE_ABI_MINOR 1u` and the v2.1 names. Run: `cargo test -p turbine-kernels --test abi_header_neutral` — expect FAIL
- [ ] Implement the header block, the optional symbol resolution (`Library::get` failures for v2.1 names are not errors), the traits, registry selection and the CPU reference ops (sequential F32 sums in index order, like the sampler).
- [ ] Run: `cargo test -p turbine-kernels` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): kernel ABI v2.1 with optional add_rmsnorm, logits_reduce and graphs`

## Task 9: Device-side logits reduction in the executor, sampler and engine

Files: `crates/turbine-model/src/executor/mod.rs` (`RowReduce`, `ReducedRow`, `SeqSlice::reduce`, `Logits::reduced`), `crates/turbine-model/src/executor/logits.rs` (new: runs `logits_reduce` over the reduced rows and packs both outputs into the one D2H copy), `crates/turbine-model/src/executor/llama.rs` and `crates/turbine-model/src/executor/olmoe.rs` (call it after the LM head), `crates/turbine-model/src/sampler.rs` (`device_request`, `finish_reduced`, eligibility), `crates/turbine-server/src/engine/loop.rs` (ask each yielding sampler, route rows, `turbine_logits_rows_total{path}`), `crates/turbine-server/src/metrics.rs`, `crates/turbine-server/tests/tiny_server.rs` (`device_sampling_matches_host`)
Interfaces:

- `#[derive(Clone, Copy)] pub struct RowReduce { pub top_n: u8, pub temperature: f32, pub uniform: Option<f32>, pub top_p: f32 }` (`top_p`: the device draw's nucleus mass, 1 = none); `pub struct ReducedRow { pub lse: f32, pub top: Vec<(u32, f32)>, pub sampled: Option<(u32, f32)> }`
- `SeqSlice` gains `pub reduce: Option<RowReduce>`; `Logits` gains `pub reduced: Vec<ReducedRow>` and `pub fn slot(&self, i: usize) -> LogitsSlot<'_>` with `pub enum LogitsSlot<'a> { Full(&'a [f32]), Reduced(&'a ReducedRow) }`
- `Sampler::device_request(&mut self) -> Option<RowReduce>` — `None` unless eligible (no penalties, no `logit_bias`, no mask, `min_tokens` reached, `top_k` −1 or 1..=64, `top_logprobs` ≤ 20; any `top_p`); draws the step's uniform from the ChaCha stream when the step samples
- `Sampler::finish_reduced(&mut self, r: &ReducedRow) -> SampledToken` — greedy: top[0]; `top_k` ≤ 64: the reference sorted-candidate draw over `top` (then `top_p` over them); categorical: `sampled` (id order, or the device's `top_p` nucleus draw); logprobs = raw logit − lse
- `turbine_logits_rows_total{path="device_reduced"|"full_row"}` on `ServerMetrics`; with `execution.device_sampling` false or no provider for `logits_reduce`, every row is `full_row`
  Covers: S-4/S-14 AC `tiny_server device_sampling_matches_host`
  Depends on: Tasks 7, 8

- [ ] Write failing test `turbine-server --test tiny_server device_sampling_matches_host`: with the CPU provider, 12 seeded requests (3 greedy, 3 greedy with `top_logprobs` 5, 2 at `temperature` 1.0 with `top_k` −1, 2 with `top_k` 40, and ineligible ones with `logit_bias`, a `json_schema` response format and `min_tokens` 4) give identical tokens and logprobs with `execution.device_sampling` true and false, and `turbine_logits_rows_total{path="device_reduced"}` equals the eligible yielding rows counted from the requests. Run: `cargo test -p turbine-server --test tiny_server device_sampling_matches_host` — expect FAIL
- [ ] Implement: the executor writes `mode`/`temperature`/`uniform` into the metadata staging buffer (Task 7), runs `logits_reduce` into a device result block adjacent to the full rows, and copies full rows plus the result block in the single D2H copy.
- [ ] Run: `cargo test -p turbine-model && cargo test -p turbine-server` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(turbine-model): device-side logits reduction for eligible rows`

## Task 10: Fused projections and `add_rmsnorm` in both executors

Files: `crates/turbine-model/src/loader.rs` (load-time concatenation into `qkv_proj` and `gate_up_proj`; OLMoE stacked `[experts, 2·inter, hidden]`), `crates/turbine-model/src/executor/llama.rs` and `crates/turbine-model/src/executor/olmoe.rs` (one QKV GEMM, one gate-up GEMM, strided `silu_mul`, `add_rmsnorm` between blocks; unfused path kept behind `fused_ops`), `crates/turbine-model/src/executor/mod.rs` (`ExecutorOptions`), `crates/turbine-model/tests/tiny_model.rs` (`fused_ops_match_unfused`)
Interfaces:

- `#[derive(Clone, Copy)] pub struct ExecutorOptions { pub fused_ops: bool, pub decode_graphs: bool, pub device_sampling: bool }` passed to `LlamaExecutor::new`, `OlmoeExecutor::new` and `build_executor` as a new last argument (contract addition; `Default` = all true)
- Fused weights: `layers.<i>.qkv_proj` `[q_dim + 2·kv_dim, hidden]` (rows Q, then K, then V) and `layers.<i>.gate_up_proj` `[2·inter, hidden]` (gate rows then up rows); the separate tensors are not kept when fused, so weight memory is unchanged
- Amendment (2026-09-26): projection fusion is its own field, `ExecutorOptions::fused_projections`, default `executor::FUSED_PROJECTIONS_DEFAULT` (off, spec S-8 amendment); `fused_ops` selects `add_rmsnorm`; the server maps _execution.fused_ops_ through `ExecutorOptions::from_fused_ops` (true: the defaults, false: every fusion off)
- `requirements()` lists `add_rmsnorm` only when `fused_ops` is true and falls back to `add` + `rmsnorm` when the registry has no provider for it
  Covers: S-8/S-9/S-14 AC `tiny_model fused_ops_match_unfused`
  Depends on: Tasks 2, 8

- [ ] Write failing test `turbine-model --test tiny_model fused_ops_match_unfused`: on both tiny checkpoints (CPU provider) a 40-token prefill and 10 decode steps give bitwise-equal logits with `fused_ops` true and false. Run: `cargo test -p turbine-model --test tiny_model fused_ops_match_unfused` — expect FAIL
- [ ] Implement the concatenation at upload (host-side row copies before the H2D upload) and the fused layer bodies; Q/K/V and gate/up views are row slices of the fused output via strides. This is the phase's main GEMM lever: the separate K/V (268 GB/s) and Q/O (397 GB/s) GEMMs are the furthest from the ≈ 640 GB/s peak.
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-model --test perf -- forward_profile --nocapture` with `fused_ops` on — expect exit 0; paste the `gemm` entries of `decode_b16_ctx768` next to the Task 3 baseline into evidence
- [ ] Run: `cargo test -p turbine-model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(turbine-model): fused QKV and gate-up projections and add_rmsnorm`

## Task 11: HIP `add_rmsnorm`, `logits_reduce` and `turbine_abi_minor`

Files: `kernels/rocm/src/fused.hip` (new: `add_rmsnorm` kernel, `logits_reduce` kernels — per-row block reduction for max/lse, per-row top-n by a two-level partial selection, categorical inverse CDF by per-block partial sums in fixed order), `kernels/rocm/src/fused.cpp` (trios, `_impl` `turbine_hip`), `kernels/rocm/src/context.cpp` (`turbine_abi_minor` returns 1), `kernels/rocm/CMakeLists.txt` (sources), `crates/turbine-kernels/tests/hip_ops.rs` (`fused_ops_match_cpu`)
Interfaces:

- Exports `turbine_abi_minor`, `turbine_add_rmsnorm{,_supported,_impl}`, `turbine_logits_reduce{,_supported,_impl}` (BF16 `add_rmsnorm`; F32 logits; `top_n` ≤ 64)
- `hip_ops fused_ops_match_cpu` (ignored): `add_rmsnorm` rows 1/16/2048 × dims 2048/3072 — residual exact, out within the Phase 1 RMSNorm tolerance; `logits_reduce` over 64 rows of 50,304 and 128,256 — top-n ids identical, lse within 1e-5 relative, categorical ids identical except where the uniform lies within 1e-6 of a CDF boundary (the test computes the boundary distance on the CPU)
  Covers: S-9 AC (`cpu::tests::add_rmsnorm_equals_add_then_rmsnorm` from Task 8 and `hip_ops fused_ops_match_cpu`)
  Depends on: Task 8

- [ ] Write the ignored test `turbine-kernels --test hip_ops fused_ops_match_cpu` as above, printing each `_impl`. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- fused_ops_match_cpu` — expect FAIL (symbols missing)
- [ ] Implement the kernels and trios; all reductions use a fixed tree order so results are run-to-run identical.
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- fused_ops_match_cpu` — expect exit 0 and `test fused_ops_match_cpu ... ok`
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels): hip add_rmsnorm, logits_reduce and ABI minor 1`

## Task 12: hipBLASLt autotuning per GEMM shape

Files: `kernels/rocm/src/gemm.cpp` (tuning on first use, tuned-choice cache ≤ 4,096 entries, one `gemm_tuned` stderr line per shape), `kernels/rocm/src/context.cpp` (`turbine_ctx_set_option` / `turbine_ctx_get_option` for `TURBINE_OPTION_GEMM_AUTOTUNE` and `TURBINE_OPTION_GEMM_TUNED_SHAPES`), `crates/turbine-server/src/startup.rs` (after context creation: `ShimContext::set_option(TURBINE_OPTION_GEMM_AUTOTUNE, execution.gemm_autotune as i64)`, an `Unsupported` answer logged once at INFO), `crates/turbine-kernels/tests/hip_ops.rs` (`gemm_autotune_matches_cpu`)
Interfaces:

- Tuning: `hipblasLtMatmulAlgoGetHeuristic` with `requestedAlgoCount = 8`; each candidate run once to warm up, then timed over 3 runs with `hipEventRecord`/`hipEventElapsedTime` on the compute stream; the fastest cached by (m, n, k, a/b/c dtypes, trans_b); failed candidates skipped; with the flag off the first heuristic result is used as in Phase 2
- Log line (stderr, one per shape): `turbine_hip: gemm_tuned m=<m> n=<n> k=<k> algo_index=<i> candidates=<c> best_us=<t>`
- Expected value is low: a lab prototype gained ≈ 0.4 ms of a 17 ms decode forward, within run-to-run noise; the task is kept because the decision of 2026-09-26 turns it on by default and it is bisectable through `execution.gemm_autotune`. Record the `decode_forward_timing` difference with the option on and off as evidence; do not chase further GEMM-algorithm gains here (fusion in Task 10 is the larger lever).
  Covers: S-8 AC `hip_ops gemm_autotune_matches_cpu`
  Depends on: Tasks 2, 8, 10

- [ ] Write the ignored test `turbine-kernels --test hip_ops gemm_autotune_matches_cpu`: after `set_option(TURBINE_OPTION_GEMM_AUTOTUNE, 1)`, m ∈ {1, 2, 3, 8, 16, 17, 64, 256, 2048} for the Llama shapes (n,k) ∈ {(5120,3072), (3072,3072), (16384,3072), (3072,8192), (128256,3072)} and the OLMoE shapes (6144,2048), (2048,2048), (64,2048), (50304,2048), BF16 and F32 outputs, within the Phase 1 GEMM tolerance; `get_option(TURBINE_OPTION_GEMM_TUNED_SHAPES)` grows by exactly 1 per new shape and by 0 on a repeated call. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- gemm_autotune_matches_cpu` — expect FAIL
- [ ] Implement the tuner and the flag plumbing.
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` — expect exit 0 with `test gemm_autotune_matches_cpu ... ok` and `test gemm_matches_cpu ... ok`
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- decode_forward_timing --nocapture` with autotune on and off — paste both `forward_ms` values into evidence
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(kernels): hipBLASLt autotuning per GEMM shape`

## Task 13: MoE small-m expert path without host synchronisation

Files: `kernels/rocm/src/moe_small_m.hip` (new: fused gate-up per selected expert over `sorted_rows`, SiLU·up, down projection, weighted scatter-add in fixed order, all offsets read on the device), `kernels/rocm/src/moe.cpp` (dispatch: `num_tokens × top_k ≤ 512` → `turbine_hip_moe_small_m`, `host_expert_offsets` may be NULL then), `kernels/rocm/CMakeLists.txt`, `crates/turbine-kernels/src/ops/mod.rs` (`MoeExpertsConfig::routed_rows`, `MoeKernel::needs_host_offsets(&self, cfg: &MoeExpertsConfig) -> bool`), `crates/turbine-kernels/src/cpu/moe.rs` (`needs_host_offsets` false), `crates/turbine-model/src/executor/olmoe.rs` (skip the offsets D2H when the provider does not need them; stacked gate-up weights from Task 10), `crates/turbine-model/tests/tiny_model.rs` (`olmoe_decode_single_device_copy`), `crates/turbine-kernels/tests/hip_ops.rs` (`moe_experts_small_m_matches_cpu`)
Interfaces:

- `MoeKernel::needs_host_offsets(&self, cfg: &MoeExpertsConfig) -> bool` (default true; contract addition); HIP returns false when `routed_rows ≤ 512`
- `_impl` `turbine_hip_moe_small_m` for ≤ 512 routed rows, `hipblaslt_per_expert` above
  Covers: S-11 AC `hip_ops moe_experts_small_m_matches_cpu`; S-11 AC `tiny_model olmoe_decode_single_device_copy`
  Depends on: Tasks 7, 10

- [ ] Write failing test `turbine-model --test tiny_model olmoe_decode_single_device_copy`: with `CountingMemory` and a test provider wrapping the CPU provider whose `needs_host_offsets` is false for ≤ 512 rows, an OLMoE decode-only forward of 8 sequences makes exactly 1 D2H copy, and a mixed forward with 600 routed rows makes layers + 1. Run: `cargo test -p turbine-model --test tiny_model olmoe_decode_single_device_copy` — expect FAIL
- [ ] Write the ignored test `turbine-kernels --test hip_ops moe_experts_small_m_matches_cpu`: 1, 16 and 64 tokens × top-8 over 64 experts (width 1024, hidden 2048), including all rows on one expert and experts with no rows, `host_expert_offsets` NULL; within the Phase 2 MoE tolerance; two runs bitwise identical. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- moe_experts_small_m_matches_cpu` — expect FAIL
- [ ] Implement the trait method, the executor change and the HIP kernels.
- [ ] Run: `cargo test -p turbine-model --test tiny_model` — expect PASS
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` — expect exit 0 with `test moe_experts_small_m_matches_cpu ... ok` and `test paged_and_moe_ops ... ok`
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(kernels): MoE small-m expert path without host offsets`

## Task 14: Decode graphs

Files: `kernels/rocm/src/graph.cpp` (new: `turbine_graph_begin` = `hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal)`, `turbine_graph_end` = `hipStreamEndCapture` + `hipGraphInstantiate`, `turbine_graph_launch` = `hipGraphLaunch`, `turbine_graph_destroy`; a capture flag on the context that makes memcpy/sync/malloc/free return `TURBINE_E_ARGUMENT`), `kernels/rocm/src/memory.cpp` (capture flag checks), `kernels/rocm/CMakeLists.txt`, `crates/turbine-model/src/executor/graphs.rs` (new: `GraphCache`, `GraphBackend` trait, tests), `crates/turbine-model/src/executor/llama.rs` and `crates/turbine-model/src/executor/olmoe.rs` (decode-only iterations through the cache; capture passes `max_kv_len = max_seq_len`), `crates/turbine-server/src/metrics.rs` and `crates/turbine-server/src/engine/loop.rs` (`turbine_decode_graph_total{outcome}` from executor counters), `crates/turbine-model/tests/tiny_model.rs` (`hip_decode_graph_matches_eager`)
Interfaces:

- `pub trait GraphBackend { type Graph; fn begin(&self) -> Result<(), KernelError>; fn end(&self) -> Result<Self::Graph, KernelError>; fn launch(&self, g: &Self::Graph) -> Result<(), KernelError>; }` implemented for `ShimContext` (`Graph = GraphHandle`)
- `pub struct GraphCache<G> { capacity: usize, … }` with `fn new(capacity: usize) -> GraphCache<G>`, `fn plan(&mut self, batch: u32, decode_only: bool) -> GraphStep` (`Eager`, `Capture`, `Replay`), `fn insert(&mut self, batch: u32, g: G)`, `fn disable(&mut self)`, `fn counters(&self) -> GraphCounters { captured, replayed, evicted, capture_failed: u64 }`; capacity = min(`max_running_requests`, 64); LRU eviction
- Per-step values live in device buffers at fixed addresses reserved at executor construction for `max_seqs` sequences (tokens, positions, `q_indptr`, `kv_lens`, block tables of `max_seq_len / block_tokens` entries); the metadata upload (Task 7) targets them before `launch`
  Covers: S-10 AC `executor::graphs::tests::cache_bounded_and_fallback`; S-10 AC `tiny_model hip_decode_graph_matches_eager`
  Depends on: Tasks 5, 8, 9, 10, 13

- [ ] Write failing test `turbine-model executor::graphs::tests::cache_bounded_and_fallback`: with a mock backend, batch 4 plans `Eager` then `Capture` then `Replay`; mixed iterations always `Eager`; capacity 3 evicts the least recently used of 4 sizes (`evicted` 1); a failed `end` counts `capture_failed` 1 and every later plan is `Eager`. Run: `cargo test -p turbine-model executor::graphs::tests::cache_bounded_and_fallback` — expect FAIL
- [ ] Write the ignored test `turbine-model --test tiny_model hip_decode_graph_matches_eager`: `require_backend("hip")`; tiny head_dim-128 Llama and OLMoE; 40 decode steps for batches 1, 5 and 16 crossing block boundaries; logits bitwise equal with `decode_graphs` true and false; `replayed` > 0. Run: `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model -- hip_decode_graph_matches_eager` — expect FAIL
- [ ] Implement the shim functions, `GraphCache`, the executor integration (capture on the second occurrence, never while a GEMM shape is untuned) and the metric.
- [ ] Run: `cargo test -p turbine-model` — expect PASS
- [ ] Run: `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model` — expect exit 0 with `test hip_decode_graph_matches_eager ... ok` and `test hip_matches_cpu ... ok`
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(turbine-model): decode graph capture and replay`

## Task 15: Scheduler batch-shape sweep and tuned lab configs

Files: `scripts/lab/phase2c-novanas-llama.yaml`, `scripts/lab/phase2c-novanas-olmoe.yaml` (tuned `scheduler.max_batch_tokens` / `scheduler.prefill_chunk_tokens`), `crates/turbine-scheduler/src/sim/tests.rs` (`phase2c_lab_configs_hold_invariants`), `crates/turbine-core/src/config/mod.rs` (defaults changed only if one pair wins for both models; then `config::tests` and the P2 table note updated)
Interfaces:

- Consumes `scripts/lab-perf.sh … --set` (Task 4), `Config::load` (Phase 0), the Phase 2 simulator invariants helpers
  Covers: S-12 AC `sim::tests::phase2c_lab_configs_hold_invariants`; S-12/S-13 AC manual lab sweep
  Depends on: Tasks 4, 14

- [ ] Run the sweep: for model in `llama olmoe`, `b` in `2048 4096 8192`, `c` in `512 1024 2048`: `scripts/lab-perf.sh novanas <model> --skip-vllm --runs 1 --set scheduler.max_batch_tokens=<b> --set scheduler.prefill_chunk_tokens=<c>` — expect 18 `lab-perf:` lines (skip pairs with `c` > `b`: none occur); paste them into evidence and pick per model the highest `turbine=` value (ties: lower ITL p95 from the summary)
- [ ] Write failing test `turbine-scheduler sim::tests::phase2c_lab_configs_hold_invariants`: loads both Phase 2c lab configs, asserts their scheduler values equal the chosen pairs, and runs `decode_never_starved`, `chunk_budget_respected` and `bounded_under_overload` checks with 1,000 seeded arrivals each. Run: `cargo test -p turbine-scheduler sim::tests::phase2c_lab_configs_hold_invariants` — expect FAIL
- [ ] Implement: write the chosen pairs into the configs; change the code defaults only if both models chose the same pair.
- [ ] Run: `cargo test -p turbine-scheduler && cargo test -p turbine-core` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `perf(lab): tuned scheduler batch shape for phase 2c`

## Task 16: Acceptance — workspace, GPU suites, golden and same-day throughput

Files: `AGENTS.md` (Commands: Phase 2c configs, `lab-perf.sh` acceptance, the five execution switches for bisecting), `examples/turbine.yaml` (final defaults)
Interfaces:

- Consumes `scripts/lab-test.sh`, `scripts/lab-serve.sh`, `scripts/lab-perf.sh`, `turbine-golden compare --concurrency`
  Covers: S-15 AC (macOS workspace gate); S-15 AC `scripts/lab-test.sh novanas` full suite; S-15 AC manual golden run; S-15/S-13 AC manual acceptance run
  Depends on: Tasks 1–15

- [ ] Run on the macOS workstation: `cargo build --workspace && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check && ! cargo tree --workspace | grep -Ei 'hip|rocm|cuda'` — expect PASS
- [ ] Run: `scripts/lab-test.sh novanas` — expect exit 0 with `test paged_and_moe_ops ... ok`, `test paged_prefill_ck_128_matches_cpu ... ok`, `test hip_matches_cpu ... ok`, `test hip_decode_graph_matches_eager ... ok` and `test logits_match_reference ... ok` for both models
- [ ] For each of `scripts/lab/phase2c-novanas-llama.yaml` (`llama-3.2-3b-instruct`) and `scripts/lab/phase2c-novanas-olmoe.yaml` (`olmoe-1b-7b-0125-instruct`): `scripts/lab-serve.sh novanas <config>` — expect `/ready` 200; `cargo run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/<slug>/reference.jsonl --concurrency 16` — expect exit 0 and 16/16 within tolerance; `scripts/lab-serve.sh novanas --stop`; paste both outputs into evidence
- [ ] Run: `scripts/lab-perf.sh novanas llama` then `scripts/lab-perf.sh novanas olmoe` — expect exit 0 and `verdict=PASS` on each (ratio ≥ 0.75 and turbine ≥ 553 / ≥ 401), `"requests_failed": 0` in every Turbine report; paste both `summary.json` files and the `perf forward_profile` output of the same build (`scripts/lab-test.sh novanas -- -p turbine-model --test perf -- forward_profile --nocapture`) into evidence
- [ ] Update `AGENTS.md` and `examples/turbine.yaml`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `docs: phase 2c performance commands and acceptance evidence`
