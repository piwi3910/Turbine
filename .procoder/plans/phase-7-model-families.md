# phase-7-model-families — implementation plan

Status: complete
Spec: .procoder/specs/phase-7-model-families.md

## Goal

Serve Qwen3 dense, Qwen3-MoE, Mistral, Mixtral, gpt-oss-20b and the Qwen3.5/3.6 Gated DeltaNet hybrids on the novanas R9700s — each behind its own registered module, each support-matrix row `supported` only after its golden, GSM8K and throughput gates — and close the four Phase 6a/6b carry-overs (OLMoE EP×TP drift, MXFP4 p05 / `mxfp4_a4`, OLMoE FP8 KV, L0 `tq4`).

## Architecture

The family registry (`turbine_model::families`) learns two things every later task builds on: a per-layer `LayerKind` list (full, sliding-window, linear attention) and a per-sequence `turbine_core::types::StateLayout`, which KV sizing, the memory budget, the scheduler's worst-case reservation and (for hybrids) the KV hierarchy's block snapshots all account for. Each new family is one file under `crates/turbine-model/src/families/` with its CPU executor first (validated against a committed transformers trace of its tiny fixture), then its HIP path over kernel ABI v2.16+ additions — quantized/biased/activation-variant MoE experts with a shared-expert gate, attention window and sink fields with d64/d256 CK instances, and `turbine_gdn_*` / `turbine_conv1d_*` — each provider chosen by a recorded reuse-first evaluation the user decides. Reasoning output gets its own registry (`turbine_model::reasoning`: `qwen_think`, `harmony`) beside the tool-call formats (`harmony` added), wired through `model.reasoning_parser`, `reasoning_content` and `reasoning_effort`; every family closes on its own lab gate task that flips its row and records the evidence in `.procoder/perf-log.md`.

## Constraints

Copied verbatim from the spec (Constraints):

- AMD only, novanas (`gfx1201`, 2× R9700 32 GB, ~640 GB/s each); lab rules of AGENTS.md (locks, ask first for new lab classes, never evict others' workloads, never dgx-spark*).
- Reuse kernels first, write our own last; no Python/Triton in build or runtime; every evaluation recorded in decisions.md.
- Pluggability: one file (or directory) per family, tool format, reasoning parser and kernel provider, behind static registries with conformance suites; unsupported combinations refused at startup with reason codes (exit 2).
- Correctness first: every new kernel has a CPU reference test and a batch-invariance test (`hip_batch_invariance` conventions); overlap scheduling (default on) stays bit-exact against serial for every family.
- Memory: a checkpoint plus its KV and state must fit its placement; TP2/EP2 rows leave the documented KV headroom.
- Disk: local disk only for active checkpoints; build caches on iSCSI.

From AGENTS.md and the contract (binding for every task):

- Order of delivery (decision 1 A): S-1 and S-17 first; Qwen3 dense; the OLMoE drift and quantized experts; Qwen3-MoE; Mistral; Mixtral; gpt-oss-20b with Harmony and the reasoning parsers; the hybrids with state and prefix reuse; the carry-overs S-14 … S-16 where their prerequisites land (S-14 after gpt-oss's MXFP4 work); housekeeping and docs; phase exit. Each family's row closes on its own gate (S-13).
- Precondition: `scripts/track-gate.sh phase-7-model-families` printed `GATE PASS phase-7-model-families` before Task 1 starts.
- Builds and tests run on novanas through `scripts/remote-cargo.sh` (no local target directories). Every task ends with `scripts/gate.sh` printing `gate: ok`; tasks touching `turbine-kernels`, `turbine-model`, `turbine-device`, the kernel ABI or `kernels/rocm` add `scripts/lab-test.sh novanas --tier quick`; tasks that can change Llama or OLMoE throughput or numerics add `scripts/lab-bench.sh --quick --model llama` and `--model olmoe` (GPU 0) and a row in `.procoder/perf-log.md` — one change, then measure.
- Lab runs start with `run_in_background` (never a foreground wait on a lock or a long remote run); throughput numbers on GPU 0 under `scripts/bench-lock.sh`; two-GPU numbers through `scripts/lab-cluster.sh --bench-lock` (Turbine legs) and two-GPU Turbine-vs-vLLM ratios through `scripts/lab-perf.sh --placement <tp2|ep2>` (decision 2026-10-08 B). Ask the user first for every checkpoint download, every soak and anything that needs workloads moved; if `amd.com/gpu` is held by another workload, stop and ask; never evict. The Hugging Face token lives on novanas only: never read it, ask for it or pass it.
- Kernel ABI: `TURBINE_ABI_VERSION` stays `2u`; each new group bumps `TURBINE_ABI_MINOR` from 15 (v2.16 experts, v2.17 attention window/sinks and d64/d256, v2.18 GDN/conv1d); a caller never passes a new field to a library below its minor; a configured family whose op the library lacks exits 1 with reason code `kernel_op_unavailable` naming the ABI entry point (Task 15).
- Unsafe Rust and FFI stay in `crates/turbine-kernels/src`; `turbine-kv` and `turbine-scheduler` gain no GPU or model dependency (they see `turbine_core::types::StateLayout` only); no vendor GPU crate (`! cargo tree --workspace | grep -Ei 'hip|rocm|cuda'`).
- Toolchain edition 2024, `rust-version = "1.97"`; `#[non_exhaustive]` on enums later phases extend (`LayerKind`, `ExpertActivation`, `ExpertDtype`); config structs `#[serde(deny_unknown_fields, default)]`; metric labels from closed enums rendered with `as_str()`.
- Names in `.procoder/contract/interfaces.md` are used verbatim; this phase's additions go to a new §27 "Phase 7 additions" (Task 38).
- Fixture generation (transformers references, self-spread, dequantized copies) is fixture time only: `nice -n 19` under `fixture.lock` on novanas, or on cpuhost (192.168.10.253 via the dgx-spark jump, `~/turbine-cpu`, CPU only) when it needs more than ~60 GB RAM; give each job a command line `/home/piwi/turbine-ci/fixture.queue` can match (`p7-ref-<slug>`).
- A golden reference is the Phase 1 format: `uv run scripts/golden/hf_reference.py --model-dir <dir> --prompts tests/golden/prompts.jsonl --out tests/golden/<slug>/reference.jsonl` (FP32 LM head on the BF16 final-norm output); `tolerance.json` starts from the Llama bounds (0.15 / 0.55, `min_prompts_passing` 14) and is widened only by the `scripts/golden/self_spread.py` method with its derivation in the slug's README.
- Each row's gate (S-13, decisions bar A and no-vLLM A): golden c1 and c16 (`--batched-bounds` for TP2/EP2), GSM8K at concurrency 16 against vLLM-ROCm on the same checkpoint (`eval-compare --max-drop 0.01`, quantized rows the 6a per-format drop of `tests/eval/<slug>/gate.json`), c16 tok/s ≥ 0.75× vLLM-ROCm on the same placement; if the pinned rocm/vllm tag cannot serve the family, try a newer tag, and if none can, golden + GSM8K against the transformers/dequantized run only, with the throughput and the reason logged in perf-log. A row that misses stays `experimental` with its evidence in perf-log.
- Every gate result is a perf-log line `- P7 closed: <architecture>/<weight>/<kv>/<placement> <evidence run ids>` (or `- P7 open: … <reason>`), which `support::tests::phase7_rows` reads (Task 6).

## Task 1: Checkpoint fetch script and the iSCSI build-cache layout (S-17)

Files: `scripts/fetch-checkpoint.sh` (new: one pinned download on novanas), `scripts/lab/phase7-checkpoints.tsv` (new: slug, repo, revision, include globs, GB), `benches/turbine-bench/tests/lab_scripts.rs` (dry-run test), `AGENTS.md` (Commands: the fetch line; Lab hosts: the iSCSI cache layout)
Interfaces:

- `scripts/fetch-checkpoint.sh [--dry-run] novanas <slug>` — reads the slug's row of `scripts/lab/phase7-checkpoints.tsv`, refuses (exit 1, `fetch: not enough disk`) when `df --output=avail /home/piwi/turbine-models` is below the row's GB + 20, else runs on novanas `hf download <repo> --revision <rev> --local-dir /home/piwi/turbine-models/<slug> [--include <glob>]…` (the user's cached `hf auth login`; the script never reads, prints or passes a token) and prints `fetch: <slug> <rev> ok`
- `scripts/lab/phase7-checkpoints.tsv` rows (tab-separated `slug repo revision include GB`):
  - `qwen3-4b-instruct-2507	Qwen/Qwen3-4B-Instruct-2507	cdbee75f17c01a7cc42f958dc650907174af0554	*	9`
  - `qwen3-30b-a3b-instruct-2507-fp8	Qwen/Qwen3-30B-A3B-Instruct-2507-FP8	5a5a776300a41aaa681dd7ff0106608ef2bc90db	*	32`
  - `mistral-7b-instruct-v0.3	mistralai/Mistral-7B-Instruct-v0.3	c170c708c41dac9275d15a8fff4eca08d52bab71	model-*.safetensors,*.json,tokenizer*	15`
  - `mixtral-8x7b-instruct-v0.1-fp8	RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8	9a978d5cc7dabc37b3d13d6501426081c99c4cbd	*	48`
  - `gpt-oss-20b	openai/gpt-oss-20b	6cee5e81ee83917806bbde320786a8fb61efebee	model-0000*-of-00002.safetensors,*.json,tokenizer*,chat_template.jinja	14`
  - `qwen3.5-9b	Qwen/Qwen3.5-9B	c202236235762e1c871ad0ccb60c8ee5ba337b9a	*	20`
  - `qwen3.6-35b-a3b-fp8	Qwen/Qwen3.6-35B-A3B-FP8	95a723d08a9490559dae23d0cff1d9466213d989	*	38`
    Covers: S-17
    Depends on: nothing in this phase

- [ ] Write failing test `lab_scripts fetch_checkpoint_dry_run_pins_revision`: `scripts/fetch-checkpoint.sh --dry-run novanas gpt-oss-20b` exits 0 and prints the `df` check and `hf download openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee --local-dir /home/piwi/turbine-models/gpt-oss-20b --include model-0000*-of-00002.safetensors …` with no `original/` path and no `HF_TOKEN`/`--token`; an unknown slug exits 2; `PATH` holds a failing `ssh` stub so nothing is contacted. Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts fetch_checkpoint` — expect FAIL with "No such file or directory"
- [ ] Implement the script and the table; `bash -n scripts/fetch-checkpoint.sh && shellcheck scripts/fetch-checkpoint.sh` — expect no output.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts` — expect PASS
- [ ] Verify the cache layout the user's disk decision put in place: `ssh piwi@192.168.10.203 'findmnt -n -o SOURCE --target /home/piwi/turbine-ci/remote; findmnt -n -o SOURCE --target /home/piwi/turbine-ci/cache/slots'` — expect exit 0 and both SOURCE columns under `/mnt/turbine-iscsi`; if a path is local, stop and ask the user (the lead's iSCSI setup is outside this plan).
- [ ] Document both in `AGENTS.md` (fetch command; "build caches live on `/mnt/turbine-iscsi`, bind-mounted, `nofail`; models stay local; a family's checkpoint is fetched when its family starts").
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok` (this run is the "gate on that layout" evidence of the S-17 criterion)
- [ ] Commit: `feat(scripts): fetch-checkpoint — pinned Phase 7 checkpoint downloads on novanas`

## Task 2: Layer kinds and `StateLayout` in the family registry (S-1)

Files: `crates/turbine-core/src/types.rs` (`StateLayout`, `StateDtype`), `crates/turbine-model/src/families/mod.rs` (`LayerKind`, trait methods `layer_kinds` / `state_layout`, test `layer_kinds_and_state_layout`), `crates/turbine-model/src/config.rs` (`ModelArchConfig.layer_kinds`, `.state`; nested `text_config`; `kv_layout` counts attention layers), `crates/turbine-model/src/budget.rs` (state bytes per running sequence), `crates/turbine-model/src/tp.rs`, `crates/turbine-model/src/ep.rs`, `crates/turbine-model/src/pp.rs` (their `kv_layout` use the attention-layer count)
Interfaces:

- `turbine_model::families::LayerKind` — `#[non_exhaustive] #[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum LayerKind { FullAttention, SlidingWindow { window: u32 }, LinearAttention }`
- `turbine_core::types::StateDtype` — `#[non_exhaustive] pub enum StateDtype { F32, Bf16 }` with `fn bytes(self) -> u64`
- `turbine_core::types::StateLayout` — `#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)] pub struct StateLayout { pub linear_layers: u32, pub recurrent_bytes_per_layer: u64, pub conv_bytes_per_layer: u64, pub dtype: Option<StateDtype> }`; `pub fn bytes_per_sequence(&self) -> u64` (= `linear_layers × (recurrent + conv)`, saturating); `pub fn is_empty(&self) -> bool`
- `ModelFamily::layer_kinds(&self, cfg: &ModelArchConfig) -> Vec<LayerKind>` (default: `num_layers × FullAttention`); `ModelFamily::state_layout(&self, cfg: &ModelArchConfig) -> StateLayout` (default `StateLayout::default()`)
- `ModelArchConfig { …, pub layer_kinds: Vec<LayerKind>, pub state: StateLayout }`; `ModelArchConfig::attention_layers(&self) -> u32` (kinds other than `LinearAttention`); `kv_layout(block_tokens).num_layers == attention_layers()`; `ModelArchConfig::kv_layer_index(&self, layer: u32) -> Option<u32>` (the KV slot of an attention layer, `None` for a linear one)
- Config parsing: when the top level has `text_config`, the family's `parse_config` and the shared keys read it; `architectures[0]` stays top-level
- `ModelBudget` gains `state_bytes_per_sequence: u64` and charges `max_running_requests × state_bytes_per_sequence` beside KV
  Covers: S-1 AC `families::tests::layer_kinds_and_state_layout` (dense part; gpt-oss and Qwen3.5 rows are added by Tasks 24 and 31 through the same test, which iterates the registry)
  Depends on: nothing in this phase

- [ ] Write failing test `families::tests::layer_kinds_and_state_layout` in `crates/turbine-model/src/families/mod.rs`:

  ```rust
  #[test]
  fn layer_kinds_and_state_layout() {
      for family in registry().iter() {
          let dir = tempfile::tempdir().unwrap();
          let spec = family.write_tiny(dir.path(), 7);
          let cfg = crate::config::load_model_config(dir.path()).unwrap();
          let expect = expected_kinds_and_state(family.name(), &spec.config_json);
          assert_eq!(cfg.layer_kinds, expect.0, "{}", family.name());
          assert_eq!(cfg.state, expect.1, "{}", family.name());
          assert_eq!(cfg.kv_layout(128).num_layers, cfg.attention_layers(), "{}", family.name());
      }
  }
  ```

  with `expected_kinds_and_state` deriving the kinds from the raw `config.json` independently (`layer_types` / `sliding_window` / `full_attention_interval`; dense and MoE families: all `FullAttention`, `StateLayout::default()`), and a second assertion that a `text_config`-wrapped copy of the tiny Qwen3 config parses to the same `ModelArchConfig`. Run: `scripts/remote-cargo.sh test -p turbine-model families::tests::layer_kinds_and_state_layout` — expect FAIL with "no field `layer_kinds`"

- [ ] Add the types, the trait defaults, the `ModelArchConfig` fields and `attention_layers` / `kv_layer_index`; switch `kv_layout` (config, tp, ep, pp) to the attention-layer count; read `text_config` in `config::load_model_config`.
- [ ] Charge the state in `budget.rs`; extend the budget unit test `budget::tests` with a `StateLayout { linear_layers: 3, recurrent_bytes_per_layer: 1 << 20, conv_bytes_per_layer: 4096, dtype: Some(StateDtype::F32) }` model whose KV + state for 4 running sequences over the budget is refused.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model` — expect PASS
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): family layer kinds and per-sequence state layout`

## Task 3: State in the scheduler's reservation and overlap bit-exactness for every family (S-1)

Files: `crates/turbine-scheduler/src/policy/mod.rs` (`AdmissionInfo.state_bytes`), `crates/turbine-scheduler/src/lib.rs` or the module holding the worst-case reservation (`worst_case_bytes` adds the state), `crates/turbine-reliability/src/ledger.rs` (`Reservation` carries state bytes), `crates/turbine-scheduler/tests/kv_sim.rs` (test), `crates/turbine-server/src/engine/loop.rs` (passes `cfg.state` to the scheduler; test `overlap_matches_serial_every_family`)
Interfaces:

- `AdmissionInfo { …, pub state_bytes: u64 }` (0 for families without state)
- `Scheduler::worst_case_reservation(&self, req: &AdmissionInfo) -> u64` = KV pages × page bytes + `state_bytes`; `Reservation::state_bytes() -> u64`; admission returns `Reject { reason: "kv_budget" }` (the existing reason) when KV + state exceeds the budget
- `engine::r#loop::tests::overlap_matches_serial_every_family`: for every registered family's tiny checkpoint on the CPU backend, 4 concurrent greedy requests with `execution.overlap_scheduling: true` (the default) produce token and logprob streams bit-equal to the serial run
  Covers: S-1 AC `kv_sim hybrid_reservation_includes_state`; the spec constraint "overlap … bit-exact against serial for every family"
  Depends on: Task 2

- [ ] Write failing test `kv_sim hybrid_reservation_includes_state`: a simulated pool of 64 pages × 1 MiB and a request with 4 worst-case pages and `state_bytes = 3 << 20` reserves exactly 7 MiB (`assert_eq!(res.bytes(), 7 << 20)`); with a 6 MiB budget the same request is rejected `kv_budget`; finishing it returns the ledger to 0. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim hybrid_reservation_includes_state` — expect FAIL with "no field `state_bytes`"
- [ ] Add the field, the reservation term and the ledger accounting (released on finish, cancel and preemption like the KV part).
- [ ] Write failing test `engine::r#loop::tests::overlap_matches_serial_every_family` (iterates `turbine_model::families::registry()`, so later families are covered when they register). Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::overlap_matches_serial_every_family` — expect FAIL with "cannot find function"; then implement the test helper and pass `cfg.state` into `AdmissionInfo` — expect PASS.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-scheduler && scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(scheduler): worst-case reservation includes per-sequence state`

## Task 4: Sliding-window refusal with a reason code before bind (S-1)

Files: `crates/turbine-model/src/config.rs` (refusal carries the code), `crates/turbine-server/src/support_startup.rs` (maps it to exit 2; test `sliding_window_refused`), `crates/turbine-server/tests/server_cli.rs` (exit-code check)
Interfaces:

- `ModelError::Unsupported` text for a sliding window starts with `sliding_window_unsupported:` and names the key (`use_sliding_window` / `sliding_window`) and value; `support_startup::model_feature_refusal(model_dir: &Path) -> Result<(), ConfigError>` runs before discovery and maps it to exit 2 with `code = "sliding_window_unsupported"`
- Kept accepted: `sliding_window: null`, a window ≥ `max_position_embeddings`, and (Task 24) gpt-oss's `layer_types` alternation of `sliding_attention` 128 / `full_attention`
  Covers: S-1 AC `support_startup::tests::sliding_window_refused`
  Depends on: Task 2

- [ ] Write failing test `support_startup::tests::sliding_window_refused`: a temp copy of `crates/turbine-model/tests/fixtures/qwen3-0.6b/config.json` with `"use_sliding_window": true` and one of `mistral-7b-instruct-v0.3/config.json` with `"sliding_window": 4096`, each passed to `model_feature_refusal`, return `Err` with `code == "sliding_window_unsupported"` and exit code 2; the unedited Mistral v0.3 config returns `Ok`. Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server support_startup::tests::sliding_window_refused` — expect FAIL with "cannot find function `model_feature_refusal`"
- [ ] Implement the code prefix and the pre-discovery check; add `server_cli` test `sliding_window_config_exits_2_before_bind` (the binary with `--check-config --set model.path=<edited dir>` exits 2 and prints `sliding_window_unsupported`).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(server): sliding_window_unsupported refused before bind`

## Task 5: Housekeeping — stale "Phase 8 track" wording and the unregistered-architecture example (S-19)

Files: `crates/turbine-model/src/families/qwen3.rs`, `families/qwen3_moe.rs`, `families/mistral.rs`, `families/mixtral.rs`, `crates/turbine-model/src/config.rs`, `crates/turbine-model/src/families/mod.rs` (tests), `crates/turbine-server/src/support_startup.rs` (tests), `crates/turbine-server/tests/tiny_server.rs`, and every other file `grep -rln "Phase 8 track\|open Phase 8 decision\|Phase 8 run-ahead" crates/` names
Interfaces:

- The example unregistered architecture in tests becomes `ExampleUnregisteredForCausalLM` (one `const UNREGISTERED_ARCH: &str` per test module)
  Covers: S-19 AC (both greps)
  Depends on: nothing in this phase (must land before Task 24 registers gpt-oss)

- [ ] Red: `grep -rn "Phase 8 track" crates/` — expect matches (exit 0); `grep -rln GptOssForCausalLM crates/*/tests crates/turbine-server/src/support_startup.rs` — expect file names.
- [ ] Reword each comment to the decided state ("Phase 7 model families (track phase-7-model-families)" or the concrete reason); replace `GptOssForCausalLM` in those tests with `UNREGISTERED_ARCH`.
- [ ] Green: `grep -rn "Phase 8 track" crates/` — expect exit 1; the `GptOssForCausalLM` grep — expect no test file.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `chore: drop stale Phase 8 wording; unregistered-architecture tests use a neutral name`

## Task 6: Phase 7 support rows and `support::tests::phase7_rows` (S-13)

Files: `crates/turbine-core/src/support.rs` (rows for the seven architectures on `amd`, all `experimental` with reason `P7_PENDING_REASON`; `PHASE7_ROWS` table; test `phase7_rows`), `crates/turbine-core/src/support.rs` `baseline_rows_present` (updated expectations), `.procoder/perf-log.md` (new section "Phase 7" with the `P7 closed` / `P7 open` line format)
Interfaces:

- `pub static PHASE7_ROWS: &[Phase7Row]` with `pub struct Phase7Row { pub architecture: &'static str, pub weight: &'static str, pub kv: &'static str, pub placement: Placement }` and `pub enum Placement { Tp1, Tp2, Ep2 }` (`as_str()` → `tp1|tp2|ep2`); the rows: `Qwen3ForCausalLM/bf16/bf16/tp1`, `Qwen3MoeForCausalLM/fp8_block/bf16/{tp2,ep2}`, `MistralForCausalLM/bf16/bf16/tp1`, `MixtralForCausalLM/fp8/bf16/{tp2,ep2}`, `GptOssForCausalLM/mxfp4/bf16/tp1`, `Qwen3_5ForConditionalGeneration/bf16/bf16/tp1`, `Qwen3_5MoeForConditionalGeneration/fp8_block/bf16/{tp2,ep2}`, plus the carry-over rows `LlamaForCausalLM/mxfp4/bf16/tp1`, `LlamaForCausalLM/mxfp4_a4/bf16/tp1`, `OlmoeForCausalLM/bf16/fp8_e4m3/tp1`, `LlamaForCausalLM/bf16/tq4/tp1`, `OlmoeForCausalLM/bf16/tq4/tp1`
- `pub fn placement_status(row: &Phase7Row) -> SupportStatus` — the row's matrix status, or `Unsupported` when `parallel_refusal` refuses its placement
- `support::tests::phase7_rows` reads `.procoder/perf-log.md` (path from `CARGO_MANIFEST_DIR`), collects `- P7 closed: <arch>/<weight>/<kv>/<placement>` lines, and asserts: every `PHASE7_ROWS` entry resolves to a status; every closed entry is `Supported`; every `Supported` entry has a closed line; `mxfp4_a4` is `Supported` only with a closed line (Task 28)
  Covers: S-13 AC `support::tests::phase7_rows`; the mechanism S-14/S-15/S-16 assert through
  Depends on: nothing in this phase

- [ ] Write failing test `support::tests::phase7_rows` as above. Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect FAIL with "cannot find value `PHASE7_ROWS`"
- [ ] Add the rows (the four already-registered families' `FAMILY_REASON` rows on `amd` become `experimental` with `P7_PENDING_REASON` = "gate pending (track phase-7-model-families)"; the three new architectures get the same status), the table and `placement_status`; add the perf-log section with no `closed` lines yet.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::` and `cargo run -p turbine-server -- --support-matrix` (via `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix`) — expect PASS and the rows listed `experimental`
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): Phase 7 support rows (experimental until gated) and phase7_rows`

## Task 7: Reasoning-parser registry, `model.reasoning_parser` and `reasoning_content` (S-10)

Files: `crates/turbine-model/src/reasoning/mod.rs` (new: trait, registry, `registry_conformance` tests), `crates/turbine-model/src/reasoning/qwen_think.rs` (new), `crates/turbine-model/src/conformance/reasoning.rs` (new: the shared suite), `crates/turbine-model/src/lib.rs` (`pub mod reasoning`), `crates/turbine-model/src/families/mod.rs` (`default_reasoning_parser`), `crates/turbine-core/src/config/mod.rs` (`model.reasoning_parser`, `ModuleNames.reasoning_parsers`), `crates/turbine-core/src/config/tests.rs`, `crates/turbine-server/src/modules.rs` (resolve `auto`, `/turbine/v1/status` `modules.reasoning_parser`), `crates/turbine-server/src/metrics.rs` (`turbine_reasoning_tokens_total`), `crates/turbine-api/src/openai/request.rs` / response types (`reasoning_content` in `message` and `delta`), the engine's output path in `crates/turbine-server/src/engine/requests.rs`, `crates/turbine-server/tests/tiny_server.rs` (test)
Interfaces:

- `pub trait ReasoningParser: Module { fn start(&self, prompt_tail: &str) -> Box<dyn ReasoningStream>; fn sample(&self) -> ReasoningSample; }`
- `pub trait ReasoningStream: Send { fn push(&mut self, text: &str) -> ReasoningDelta; fn finish(&mut self) -> ReasoningDelta; }`; `#[derive(Default, PartialEq, Debug)] pub struct ReasoningDelta { pub reasoning: String, pub content: String }`
- `pub struct ReasoningSample { pub prompt_tail: &'static str, pub output: &'static str, pub reasoning: &'static str, pub content: &'static str }` — the conformance suite feeds `output` whole and split at every char boundary and expects the same `reasoning` / `content`
- `turbine_model::reasoning::registry() -> &'static Registry<dyn ReasoningParser>` (`"reasoning_parser"`, `[&QwenThink]`; Task 25 adds `&Harmony`)
- `QwenThink` (`qwen_think`): starts in reasoning when `prompt_tail` ends with an unclosed `<think>`; text up to `</think>` is reasoning; a `<think>` never closed before `max_tokens` is all reasoning; `<think>`/`</think>` split across pushes are held back until decided
- `ModelFamily::default_reasoning_parser(&self) -> Option<&'static str>` (default `None`; Qwen3 and Qwen3-MoE `Some("qwen_think")`)
- Config `model.reasoning_parser: ModuleName` (default `auto` = the family's default, `none` always accepted, unregistered → exit 2 naming `model.reasoning_parser`)
- API: `choices[].message.reasoning_content: Option<String>`, `choices[].delta.reasoning_content: Option<String>` (omitted when `None`); `chat_template_kwargs` already reaches `ChatTemplate::render`
- Metric `turbine_reasoning_tokens_total` (counter, tokens attributed to reasoning)
  Covers: S-10 (all three ACs)
  Depends on: nothing in this phase

- [ ] Write failing test `reasoning::registry_conformance::every_parser_passes_the_suite` (runs `crate::conformance::reasoning::suite` over `registry()`) and `reasoning::registry_conformance::names` (`["qwen_think"]`). Run: `scripts/remote-cargo.sh test -p turbine-model reasoning::registry_conformance` — expect FAIL with "could not find `reasoning`"
- [ ] Implement the trait, registry, suite and `QwenThink` (sample: tail `"<|im_start|>assistant\n<think>\n"`, output `"plan</think>\n\nanswer"`, reasoning `"plan"`, content `"\n\nanswer"`) — run the same command, expect PASS.
- [ ] Write failing test `config::tests::reasoning_parser_key`: default `auto`; `validate_modules` with `reasoning_parsers: &["qwen_think"]` refuses `nope` naming `model.reasoning_parser`. Run: `scripts/remote-cargo.sh test -p turbine-core config::tests::reasoning_parser_key` — expect FAIL; implement the key — expect PASS; then `scripts/remote-cargo.sh run -p turbine-server -- --config examples/turbine.yaml --check-config --set model.reasoning_parser=nope` — expect exit 2 and `model.reasoning_parser` in stderr.
- [ ] Write failing test `tiny_server reasoning_content_follows_enable_thinking`: the tiny Qwen3 checkpoint with a test chat template whose generation prompt ends `<think>\n` unless `enable_thinking` is false (then `<think>\n\n</think>\n\n`); a greedy request with `max_tokens: 8` returns non-empty `message.reasoning_content` and the streaming deltas concatenate to the same `reasoning_content`; with `"chat_template_kwargs": {"enable_thinking": false}` the field is absent and `content` holds the text; `/turbine/v1/status` lists `modules.reasoning_parser == "qwen_think"`; `/metrics` has `turbine_reasoning_tokens_total` = 8 after the first request. Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server reasoning_content_follows_enable_thinking` — expect FAIL
- [ ] Wire the parser into the engine's text path (the last 64 bytes of the rendered prompt as `prompt_tail`; parser state per choice so `n` > 1 is independent), the response types, the status `modules` entry and the metric.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server && scripts/remote-cargo.sh test -p turbine-api` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model,api): reasoning-parser registry (qwen_think) and reasoning_content`

## Task 8: Per-family default tool formats round-trip through their real templates (S-18, part 1)

Files: `crates/turbine-model/src/formats/tests.rs` (test `family_default_tool_calls_round_trip`), `crates/turbine-model/src/families/{qwen3,qwen3_moe,mistral,mixtral}.rs` (`default_tool_format`), `crates/turbine-model/tests/fixtures/qwen3-0.6b/tokenizer_config.json`, `qwen3-30b-a3b/…`, `mistral-7b-instruct-v0.3/…`, `mixtral-8x7b-instruct-v0.1/…` (the real chat templates, already committed)
Interfaces:

- `default_tool_format`: Qwen3 / Qwen3-MoE `Some("hermes")`; Mistral / Mixtral `Some("mistral")`; gpt-oss `Some("harmony")` (Task 25); Qwen3.5/3.6 per Task 31
- Test helper `fn family_template_fixture(family: &str) -> Option<&'static str>` (fixture dir per family; `None` for Llama/OLMoE, covered by Phase 2)
  Covers: S-18 AC `formats::tests::family_default_tool_calls_round_trip` (dense/MoE part; Tasks 25 and 31 add their families' fixtures to the same table)
  Depends on: nothing in this phase
  As built (2026-10-08): the four defaults were already set by Phase 2m, so the test's red step was the missing `qwen3-30b-a3b/tokenizer_config.json` (added from the pinned `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` revision of `scripts/lab/phase7-checkpoints.tsv`). Mixtral-8x7B-Instruct-v0.1's real template (and the pinned RedHatAI FP8 revision's, identical) renders an assistant turn's `content` only and drops `tool_calls`, so no call can round-trip through it: the test asserts that explicitly for `mixtral` (open point for Task 22: how tools reach a Mixtral prompt).

- [ ] Write failing test `formats::tests::family_default_tool_calls_round_trip`: for every registered family with a template fixture, render an assistant turn carrying one `get_weather` tool call (`{"location":"Oslo","unit":"celsius"}`) through the fixture's chat template, take the rendered assistant text, parse it with the family's default format's `parser()` and assert one call named `get_weather` with those arguments; assert each family's `default_tool_format()` is the name above. Run: `scripts/remote-cargo.sh test -p turbine-model formats::tests::family_default_tool_calls_round_trip` — expect FAIL with the first family whose default is unset
- [ ] Set the defaults in the family modules.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model formats::` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): default tool formats for Qwen3, Qwen3-MoE, Mistral, Mixtral`

## Task 9: Lab scripts learn the Phase 7 slugs and two-card family scenarios (S-13 tooling)

Files: `scripts/lab-bench.sh` (`--model qwen3-4b|mistral-7b|gpt-oss-20b|qwen3.5-9b`), `scripts/lab-serve.sh` (`--vllm` accepts the seven Phase 7 slugs, model id from the lab config), `scripts/lab-perf.sh` (`<model>` accepts the nine short names below, `--placement tp1|tp2|ep2`, `--eval`), `scripts/lab-cluster.sh` (scenarios `qwen3moe-tp2-novanas`, `qwen3moe-ep2-novanas`, `mixtral-tp2-novanas`, `mixtral-ep2-novanas`, `qwen36-tp2-novanas`, `qwen36-ep2-novanas` through one `scenario_family <short> <tp2|ep2>`; Turbine legs only, user decision 2026-10-08 B), `scripts/lab/phase7-novanas-{qwen3-4b,mistral-7b,gpt-oss-20b,qwen3.5-9b}.yaml`, `scripts/lab/phase7-novanas-{qwen3moe,mixtral,qwen36}-{tp2,ep2}.yaml` (new lab configs; `model.path` `/models/<slug>`, the Jobs' read-only mount of `/home/piwi/turbine-models`, as every existing lab config; `lab-bench.sh` overrides it with the host path), `benches/turbine-bench/tests/lab_scripts.rs` (dry-run tests), `AGENTS.md` (lab-bench / lab-perf / lab-cluster lines)
Interfaces:

- Short name → slug: `qwen3-4b` → `qwen3-4b-instruct-2507`, `qwen3moe` → `qwen3-30b-a3b-instruct-2507-fp8`, `mistral-7b` → `mistral-7b-instruct-v0.3`, `mixtral` → `mixtral-8x7b-instruct-v0.1-fp8`, `gpt-oss-20b` → `gpt-oss-20b`, `qwen3.5-9b` → `qwen3.5-9b`, `qwen36` → `qwen3.6-35b-a3b-fp8`; golden reference `tests/golden/<slug>/reference.jsonl`; served model id = the checkpoint's Hugging Face repo id (`scripts/lab/phase7-checkpoints.tsv`), in the lab configs and in `lab-serve.sh --vllm` alike; an unknown `--vllm` slug exits 2 naming the list (as today)
- `scenario_family <short> <placement>` (both R9700s, the cluster pod; Turbine only — the pod has no vLLM and GPUs are per container, decision 2026-10-08 B): serve `scripts/lab/phase7-novanas-<short>-<placement>.yaml` on the pod loopback; `turbine-golden compare --batched-bounds` at `--concurrency 1` and `16`; `turbine-golden eval --tasks tests/eval/gsm8k-200.jsonl --concurrency 16 --output json` into `$RUN_DIR/results/turbine-<placement>.json`, copied back by the workstation to `target/lab-cluster/<run id>/` before cleanup; `turbine-golden eval-compare --baseline tests/eval/<slug>/vllm-<placement>.json --candidate … --max-drop <gate.json max_drop>` when that baseline is committed (else `gsm8k=nobaseline`, no GSM8K verdict); the standard c16 bench (`--concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos`, after a 16-request warm-up); last line in the Job `lab-cluster: <scenario> PASS|FAIL golden1=… golden16=… gsm8k=<PASS|FAIL|nobaseline> tok_s=<t>`. `--dry-run` of a family scenario also prints these in-Job steps (`+ …` lines, ending `lab-cluster: dry run: in-Job steps of <scenario>`); the other scenarios' dry-run output is unchanged
- `scripts/lab-perf.sh [--dry-run] novanas <model> [--placement tp1|tp2|ep2] [--eval] …`: `<model>` ∈ `llama|olmoe` (unchanged) or the Phase 7 short names; one-card models (`qwen3-4b`, `mistral-7b`, `gpt-oss-20b`, `qwen3.5-9b`) take `tp1` (default) and `scripts/lab/phase7-novanas-<short>.yaml`; two-card models (`qwen3moe`, `mixtral`, `qwen36`) take `tp2` or `ep2` (required) and `scripts/lab/phase7-novanas-<short>-<placement>.yaml` served with `lab-serve.sh --gpus 2 … --set server.listen=0.0.0.0:18000`, then `lab-serve.sh --vllm <slug> --gpus 2 --vllm-arg --tensor-parallel-size --vllm-arg 2` (+ `--vllm-arg --enable-expert-parallel` for `ep2`) on the same cards, each stopped by its run id; `--eval` adds, after vLLM's benches, `turbine-golden eval --url http://192.168.10.203:18100 --tasks tests/eval/gsm8k-200.jsonl --concurrency 16 --output json` into `target/lab-perf/<run id>/vllm-gsm8k.json` (the baseline committed as `tests/eval/<slug>/vllm[-<placement>].json`); a Phase 7 model has no recorded vLLM number, so `--skip-vllm` is a usage error for it; last line `lab-perf: <model> placement=<p> turbine=<t> vllm=<v> ratio=<r> target=0.75x verdict=<PASS|FAIL>` (PASS: ratio ≥ 0.75 and no Turbine request failed), or `vllm=unavailable ratio=none verdict=NO_VLLM` (exit 1) when vLLM's serve fails — the S-13 no-vLLM rule; the `llama`/`olmoe` line is unchanged
- `ep2-novanas` gains a leg `ep2tp2` (Task 13) — not changed here
  Covers: tooling for the S-2, S-5, S-6, S-7, S-8, S-11 lab gates
  Depends on: nothing in this phase

- [ ] Write failing tests in `lab_scripts`: `lab_bench_dry_run_phase7_models` (`scripts/lab-bench.sh --dry-run --model qwen3-4b --golden16` prints the serve of `scripts/lab/phase7-novanas-qwen3-4b.yaml` and `turbine-golden compare … --reference tests/golden/qwen3-4b-instruct-2507/reference.jsonl --concurrency 16`; `--print-model` of the four one-card names), `lab_serve_dry_run_vllm_phase7_slugs` (each of the seven slugs renders a Job serving its repo id; an unknown slug still exits 2 naming the list), `lab_perf_dry_run_phase7_placements` (`qwen3moe --placement ep2 --eval`: Turbine served with `--gpus 2` and the ep2 config, vLLM with `--gpus 2` and the tp/ep args, the vLLM eval, both stops; `mistral-7b` one card; usage errors: a two-card model without a placement, `tp2` on a one-card model, `--skip-vllm` on a Phase 7 model — exit 2), `lab_cluster_dry_run_family_scenarios` (each of the six scenarios renders the two-GPU Job and prints the in-Job steps: the server on its config, the batched-bounds golden at c1 and c16, the eval, the eval-compare against `vllm-<placement>.json`, the bench; no vLLM step), `phase7_novanas_configs_load` (the ten configs load with their slug path, served id and parallel sizes). Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts phase7` — expect FAIL
- [ ] Implement the slug tables, the configs, `lab-perf.sh` placements and `scenario_family`; keep every existing scenario's rendered commands unchanged (the existing `lab_scripts` tests stay green).
- [ ] Run: `bash -n scripts/lab-{bench,serve,perf,cluster}.sh && shellcheck scripts/lab-{bench,serve,perf,cluster}.sh && scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(lab): Phase 7 model slugs, lab configs and two-card family scenarios`

## Task 10: Qwen3 dense on HIP — per-head QK-norm and tensor parallelism (S-2)

Files: `crates/turbine-model/src/executor/decoder/hooks/` (the `QK_NORM_PER_HEAD` hook's HIP path: one `turbine_rmsnorm` over `[tokens × heads, head_dim]` rows of Q and of K before RoPE), `crates/turbine-model/src/families/qwen3.rs` (`tp_decoder_spec`, `requirements` include the per-head RMSNorm config), `crates/turbine-model/tests/tiny_model.rs` (tests), `crates/turbine-model/tests/golden.rs` (lab test `hip_trace_vs_cpu_qwen3`)
Interfaces:

- `Qwen3::tp_decoder_spec() -> Some(DecoderSpec { hooks: [QK_NORM_PER_HEAD, SWIGLU], .. })` — heads and KV heads split by `turbine_distributed::tp` (32/8 over 2 ranks), per-head norm weights replicated
- Lab test `golden hip_trace_vs_cpu_qwen3` (needs `TURBINE_TEST_QWEN3_MODEL_DIR`, which `scripts/lab/novanas-test-job.yaml` sets to `/models/qwen3-4b-instruct-2507` beside `TURBINE_TEST_MOE_MODEL_DIR` — `lab-test.sh` passes no environment through; a no-op in a debug build, run with `--release`): HIP vs CPU per-op trace on prompt p01 within the Phase 1 trace bounds (per op: HIP `q_norm`/`k_norm` within 2 BF16 ulps (two roundings) of the host recompute from their own inputs; top-5 logits order equal each step)
  Covers: S-2 (code)
  Depends on: Task 2

- [ ] Write failing test `tiny_model tp2_matches_tp1_on_host_every_tp_family` (iterates the registry, skips families whose `tp_decoder_spec()` is `None`, asserts tp 2 equals tp 1 on the CPU backend within the existing TP host bound, and asserts Qwen3 is among the families run). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model tp2_matches_tp1_on_host_every_tp_family` — expect FAIL with "Qwen3 has no tp_decoder_spec"
- [ ] Implement `Qwen3::tp_decoder_spec` — run, expect PASS.
- [ ] Write the lab test `tiny_model hip_trace_vs_cpu_qwen3_tiny` (the tiny Qwen3 checkpoint, HIP vs CPU per op, `#[ignore]`) and `golden hip_trace_vs_cpu_qwen3`; implement the HIP path of the hook.
- [ ] Lab (ask the user first; the download is a new lab class): `scripts/fetch-checkpoint.sh novanas qwen3-4b-instruct-2507` — expect `fetch: qwen3-4b-instruct-2507 cdbee75f… ok`
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` and `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model -- hip_trace_vs_cpu_qwen3_tiny --include-ignored` — expect exit 0; then `scripts/lab-test.sh novanas -- --release -p turbine-model --test golden -- hip_trace_vs_cpu_qwen3` — expect exit 0
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): Qwen3 on HIP (per-head QK-norm) and tensor parallel`

## Task 11: Qwen3-4B fixtures and gate (S-2, S-13)

Files: `tests/golden/qwen3-4b-instruct-2507/{reference.jsonl,tolerance.json,README.md}` (new), `tests/eval/qwen3-4b-instruct-2507/{vllm.json,turbine.json,gate.json}` (new), `crates/turbine-core/src/support.rs` (row flip if the gate passes), `.procoder/perf-log.md` (section "P7 Qwen3-4B gate")
Interfaces:

- README records repo `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, transformers version, command lines and the tolerance derivation
- `gate.json`: `{"max_drop": 0.01}`
  Covers: S-2 AC (lab gate), S-13, S-17 (README revision)
  Depends on: Tasks 9, 10, 6

- [ ] Reference (fixture time, under `fixture.lock`, command line `p7-ref-qwen3-4b`): `nice -n 19 flock /home/piwi/turbine-ci/fixture.lock uv run scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/qwen3-4b-instruct-2507 --prompts tests/golden/prompts.jsonl --out tests/golden/qwen3-4b-instruct-2507/reference.jsonl` on novanas — expect 16 lines; write `tolerance.json` with the Llama bounds and the README.
- [ ] Lab gate: `scripts/lab-bench.sh --model qwen3-4b --golden16` (background) — expect `BENCH … golden1=PASS golden16=PASS tok/s=<t>`
- [ ] GSM8K: `scripts/lab-serve.sh novanas scripts/lab/phase7-novanas-qwen3-4b.yaml`, `cargo run -q -p turbine-bench --bin turbine-golden -- eval --url http://192.168.10.203:18000 --tasks tests/eval/gsm8k-200.jsonl --concurrency 16 --output json > tests/eval/qwen3-4b-instruct-2507/turbine.json`, `scripts/lab-serve.sh novanas --stop`; `scripts/lab-serve.sh novanas --vllm qwen3-4b-instruct-2507` and the same eval against `:18100` into `vllm.json`; `turbine-golden eval-compare --baseline tests/eval/qwen3-4b-instruct-2507/vllm.json --candidate tests/eval/qwen3-4b-instruct-2507/turbine.json --max-drop 0.01` — expect exit 0
- [ ] Throughput: `scripts/lab-perf.sh novanas qwen3-4b --config scripts/lab/phase7-novanas-qwen3-4b.yaml` — expect `ratio=` ≥ 0.75 (on vLLM failure: try the next rocm/vllm tag in `scripts/lab/novanas-vllm-job.yaml` once, else log `vllm=unavailable` with the reason)
- [ ] Record in perf-log "P7 Qwen3-4B gate" the three results and `- P7 closed: Qwen3ForCausalLM/bf16/bf16/tp1 <run ids>` if all pass (else `- P7 open: … <what missed>`); flip the row to `supported` only on `closed`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(qwen3): Qwen3-4B golden, GSM8K and throughput gate`

## Task 12: OLMoE EP×TP drift — the per-position routing trace (S-3, investigate)

Files: `crates/turbine-model/src/executor/decoder/moe.rs` or the MoE hook (a routing trace sink: per layer and position the top-k ids, gate values and the margin to the k+1-th logit), `crates/turbine-model/src/testing/trace.rs` (`RoutingTrace` rows in the existing trace), `crates/turbine-server/tests/lab_cluster.rs` (new; lab test `olmoe_routing_trace_ep2_tp2`, `#[ignore]`), `scripts/lab-cluster.sh` (`ep2-novanas` leg `trace` that runs it), `.procoder/perf-log.md` (section "P7 OLMoE EP×TP drift")
Interfaces:

- `turbine_model::testing::trace::RoutingRow { layer: u32, position: u32, topk_ids: Vec<u32>, topk_gates: Vec<f32>, margin: f32 }`; `DecoderExecutor::set_routing_trace(&mut self, on: bool)`; `take_routing_trace() -> Vec<RoutingRow>`
- `lab_cluster olmoe_routing_trace_ep2_tp2`: one-GPU vs ep2×tp2 (the refusal bypassed with the test-only `parallel.allow_refused_for_diagnosis: true`, accepted only in a `cfg(test)`/`fault-injection` build) on prompts p08 and p10; prints, per position where the greedy token differs or any top-k set differs, the layer, both id sets and the one-GPU margin
  Covers: S-3 (trace and cause)
  Depends on: Task 9

- [ ] Write failing host test `tiny_model routing_trace_records_topk_and_margin` (tiny OLMoE, CPU: trace rows count = layers × positions, margins ≥ 0, ids match `turbine_kernels::cpu::topk` on the logged logits). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model routing_trace_records_topk_and_margin` — expect FAIL with "no method named `set_routing_trace`"
- [ ] Implement the trace sink (off by default; zero cost when off) — run, expect PASS.
- [ ] Write the lab test and the `ep2-novanas` `trace` leg; lab: `scripts/lab-cluster.sh ep2-novanas` (background, both R9700s) — expect the trace table in the log.
- [ ] Record in perf-log "P7 OLMoE EP×TP drift": the p08 and p10 tables, whether every first divergence sits on a margin below the TP partial-sum rounding (|Δ| measured), and the stated cause. If the cause is not the near-tie hypothesis, record the evidence and the next hypothesis before Task 13.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(olmoe): per-position routing trace for one GPU vs ep2×tp2`

## Task 13: OLMoE EP×TP — fix, golden and removing `olmoe_ep_tp_drift` (S-3)

Files: the cause's site from Task 12 (expected: `crates/turbine-model/src/tp.rs` / the MoE hook — the router input reduced rank-invariantly: every rank routes on the same all-reduced hidden row, so ties break identically), `crates/turbine-core/src/support.rs` (`PARALLEL_REFUSALS` loses the OLMoE row), `crates/turbine-server/tests/lab_cluster.rs` (lab test `olmoe_ep2_tp2_golden`), `scripts/lab-cluster.sh` (`ep2-novanas` runs `olmoe_ep2_tp2_golden` instead of checking the refusal), `crates/turbine-model/tests/tiny_model.rs` (host test)
Interfaces:

- `lab_cluster olmoe_ep2_tp2_golden` (`#[ignore]`): reads `TURBINE_LAB_CLUSTER_URL` (the ep2×tp2 server on the pod loopback) and runs `turbine-golden compare --reference tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl` at `--concurrency 1` and `--concurrency 16` (`--batched-bounds`) through `TURBINE_GOLDEN_BIN`; asserts both exit 0
- `PARALLEL_REFUSALS` becomes `&[]` (the type stays for later refusals)
  Covers: S-3 AC (lab test, grep, perf-log cause)
  Depends on: Task 12

- [ ] Write failing host test `tiny_model olmoe_ep2_tp2_routing_matches_tp1_on_host`: tiny OLMoE with router logits engineered to a near-tie (two experts' logits within 1e-6 at one position), ep 2 × tp 2 on the CPU backend picks the same experts as tp 1 at every position. Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model olmoe_ep2_tp2_routing_matches_tp1_on_host` — expect FAIL at the engineered position
- [ ] Implement the fix named by Task 12's cause — run, expect PASS.
- [ ] Remove the refusal row; `grep olmoe_ep_tp_drift crates/turbine-core/src/support.rs` — expect exit 1; update `support::tests` and the `ep2-novanas` refusal check.
- [ ] Lab: `scripts/lab-cluster.sh --bench-lock ep2-novanas` (background) — expect `lab-cluster: ep2-novanas PASS` with `olmoe_ep2_tp2_golden` ok; then `scripts/lab-bench.sh --quick --model olmoe` — expect golden1=PASS and tok/s within 1 % of the last perf-log OLMoE row (perf-log row).
- [ ] Record the fix and the run ids in perf-log "P7 OLMoE EP×TP drift".
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `fix(moe): rank-invariant routing under ep×tp; drop olmoe_ep_tp_drift`

## Task 14: Quantized MoE expert provider — evaluation and decision (S-4)

Files: `kernels/rocm/tools/moe_quant_eval/` (new: a standalone HIP harness timing each candidate on gfx1201 on the three expert shapes), `.procoder/ask/decisions.md` (entry "P7 quantized MoE expert provider")
Interfaces:

- Shapes: Qwen3-30B-A3B (128 experts, top-8, hidden 2048, inter 768, FP8 block 128×128), Mixtral (8 experts, top-2, hidden 4096, inter 14336, FP8 per-tensor static W8A8), gpt-oss-20b (32 experts, top-4, hidden 2880, inter 2880, MXFP4 weight-only with biases); token counts 1, 16, 256, 4096
- Candidates per format: llama.cpp `mul_mat_id` (mmq / mmvq; MXFP4 and its FP8 path if present), CK grouped GEMM (`ck_tile` grouped / `moe_gemm` instances for gfx12) and hipBLASLt grouped GEMM, and our `qgemm_fp8_wmma` / `qgemm_fp8_block` / `qgemm_mxfp4` looped over the expert offsets (the "own kernel extended to grouped form" baseline)
  Covers: S-4 AC (decisions entry)
  Depends on: nothing in this phase (Task 15 waits for the decision)

- [ ] Build the harness on novanas (`scripts/remote-cargo.sh` is not involved: `cmake -S kernels/rocm/tools/moe_quant_eval -B <build> -DGPU_TARGETS=gfx1201 && cmake --build <build>` inside `scripts/bench-lock.sh`) and run each candidate: correctness against the dequantized BF16 grouped reference (max |Δ| within the 6a per-format bound) and median µs over 50 runs per shape and token count.
- [ ] Write the decisions.md entry "P7 quantized MoE expert provider" with the table (candidate × format × tokens → µs, pass/fail, reason a candidate cannot build on gfx1201) and options per format (A the fastest correct reused provider, B our own grouped kernel, C a split: reused for decode token counts, own for prefill).
- [ ] ASK THE USER for the decision per format; record it under the entry (`**Decision (user, <date>):** …`).
- [ ] Commit: `docs(decisions): P7 quantized MoE expert provider evaluation`

## Task 15: Expert ABI v2.16 — quantized dtypes, biases, activation variants, shared expert; CPU reference (S-4)

Files: `kernels/include/turbine_kernels.h` (`turbine_moe_experts_desc` v2.16 fields, `TURBINE_ABI_MINOR 16u`), `crates/turbine-kernels/src/ffi.rs`, `crates/turbine-kernels/src/ops/mod.rs` (`MoeExpertsConfig` fields), `crates/turbine-kernels/src/cpu/moe.rs` (reference for every variant), `crates/turbine-kernels/src/cpu/topk.rs` (router mode `TopkThenSoftmax` with bias), `kernels/include/turbine_kernels.h` `turbine_moe_route_desc` (`ROUTE_TOPK_THEN_SOFTMAX`, `router_bias`), `crates/turbine-server/src/startup.rs` (`kernel_op_unavailable`), `crates/turbine-kernels/tests/abi_header_neutral.rs`
Interfaces:

- `turbine_moe_experts_desc` v2.16 (read only by a library reporting minor ≥ 16): `int32_t weight_dtype` (`TURBINE_EXPERT_BF16 = 0`, `TURBINE_EXPERT_FP8_BLOCK = 1`, `TURBINE_EXPERT_FP8_TENSOR = 2`, `TURBINE_EXPERT_MXFP4 = 3`), `const void *w_gate_scale, *w_up_scale, *w_down_scale`, `const float *act_static_scale` (FP8 per-tensor W8A8 input scales per expert), `const void *b_gate, *b_up, *b_down` (nullable, `[experts, n]` BF16), `int32_t activation` (`TURBINE_EXPERT_SWIGLU = 0`, `TURBINE_EXPERT_GPTOSS_CLAMP = 1`: gate clamped ≤ 7.0, up clamped ±7.0, `out = (up + 1) · gate · σ(1.702 · gate)`), `int32_t gate_up_interleaved` (gpt-oss), `float clamp_limit`, `const void *shared_w_gate, *shared_w_up, *shared_w_down, *shared_gate_w` (nullable; shared expert output × `σ(x · shared_gate_w)` added), `int32_t shared_inter`
- `turbine_moe_route_desc` v2.16: flag `TURBINE_ROUTE_TOPK_THEN_SOFTMAX` and `const void *router_bias`
- Rust: `#[non_exhaustive] pub enum ExpertDtype { Bf16, Fp8Block, Fp8Tensor, Mxfp4 }`, `#[non_exhaustive] pub enum ExpertActivation { SwiGlu, GptOssClamp { limit: f32 } }`, `MoeExpertsConfig { …, dtype: ExpertDtype, activation: ExpertActivation, biases: bool, gate_up_interleaved: bool, shared_inter: Option<u32> }`
- Startup: a registry construction failure for an op config the library lacks → exit 1, `code = "kernel_op_unavailable"`, message naming the ABI entry point (e.g. `turbine_moe_experts (weight_dtype=mxfp4)`)
  Covers: S-4 (ABI + CPU reference), the spec failure mode "Kernel library without an op the family needs"
  Depends on: Task 14 (decision)

- [ ] Write failing tests in `crates/turbine-kernels/src/cpu/moe.rs`: `quant_experts_match_dequantized` (each `ExpertDtype` on a seeded 8-expert problem equals the BF16 reference over the dequantized weights within 1e-2 relative), `gptoss_clamp_activation_matches_formula` (hand-computed values at gate 8.0, −8.0, 0.5 and interleaved columns), `shared_expert_gate_adds_sigmoid_scaled_output`, and in `cpu/topk.rs` `topk_then_softmax_with_bias` (top-k on biased logits, softmax over the k only). Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::` — expect FAIL with "no variant `Fp8Block`"
- [ ] Implement the header fields, the FFI mirror (`abi_header_neutral` updated), the Rust configs and the CPU reference — run, expect PASS.
- [ ] Write failing test `server_cli kernel_op_unavailable_exits_1` (the CPU backend with a test registry that refuses `MoeExpertsConfig { dtype: Mxfp4, .. }`, loading a tiny MXFP4-expert checkpoint, exits 1 printing `kernel_op_unavailable` and `turbine_moe_experts`); implement the mapping — expect PASS.
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0 (the existing library reports minor 15 and keeps working)
- [ ] Commit: `feat(kernels): expert ABI v2.16 — quantized dtypes, biases, gpt-oss activation, shared expert`

## Task 16: HIP quantized experts and the quantized-MoE loader (S-4)

Files: `kernels/rocm/src/moe_quant.hip` (new, or the vendored provider under `kernels/rocm/third_party/<provider>/` with its licence in `kernels/rocm/third_party/LICENSES/`, per Task 14's decision), `kernels/rocm/src/moe.cpp` (dispatch on `weight_dtype`, activation, biases, shared expert), `kernels/rocm/src/abi_minor.cpp` (16), `kernels/rocm/CMakeLists.txt`, `crates/turbine-kernels/tests/hip_moe_quant.rs` (new), `crates/turbine-kernels/tests/hip_batch_invariance.rs` (`quant_experts_batch_invariant` lives in `hip_moe_quant.rs` with the same conventions), `crates/turbine-model/src/weights/mod.rs` and the `hf_fp8` / `ct_fp8` / `openai_mxfp4` modules (expert tensors repacked to the expert layout), `crates/turbine-model/src/executor/decoder/mod.rs` (`quant_moe_phase7` refusal narrowed), `crates/turbine-model/tests/tiny_model.rs`
Interfaces:

- Loader: expert slots `mlp.experts.N.{gate,up,down}_proj.weight` (+ `weight_scale_inv` / `weight_scale` / `input_scale`) stacked into `[experts, n, k]` per dtype; gpt-oss's fused `mlp.experts.gate_up_proj_{blocks,scales,bias}` and `down_proj_{blocks,scales,bias}` mapped to `gate_up_interleaved`
- `quant_moe_phase7` remains only for a format × family pair without a row (e.g. AWQ/GPTQ experts, `mxfp4_a4` experts), message `quant_moe_phase7: <format> experts for <architecture> have no supported row`
- Lab tests (`hip_moe_quant`): `fp8_block_experts_match_reference`, `fp8_tensor_experts_match_reference`, `mxfp4_experts_match_reference` (HIP vs the Task 15 CPU reference within the 6a per-format bounds on the three Task 14 shapes at 1, 16 and 256 tokens, biases and the clamp variant on the MXFP4 case, a shared expert on one FP8 block case), `quant_experts_batch_invariant` (a row's output identical whether run alone or in a batch of 1…64 rows)
  Covers: S-4 AC (lab tests, grep)
  Depends on: Task 15

- [ ] Write the four lab tests. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_moe_quant` — expect FAIL with "TURBINE_E_UNSUPPORTED" (library minor 15)
- [ ] Implement the HIP dispatch over the decided provider — run, expect PASS.
- [ ] Write failing host test `tiny_model quantized_moe_matches_dequantized_bf16` (tiny Qwen3-MoE FP8-block, tiny Mixtral FP8-tensor and tiny MXFP4-expert checkpoints written by the fixture writer; CPU executor output equals the dequantized BF16 checkpoint within the 6a host bound) and update `tiny_model`'s existing `quant_moe_phase7` assertion to an AWQ-expert case. Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model quantized_moe` — expect FAIL; implement the loader repack and narrow the refusal — expect PASS.
- [ ] `grep -rn quant_moe_phase7 crates/` — expect only the narrowed refusal, its doc line and its test.
- [ ] Gate: `scripts/gate.sh`, `scripts/lab-test.sh novanas --tier quick` and `scripts/lab-bench.sh --quick --model olmoe` — expect `gate: ok`, exit 0 and OLMoE BF16 tok/s within 1 % (perf-log row)
- [ ] Commit: `feat(kernels,model): quantized MoE experts on HIP (fp8 block, fp8 tensor, mxfp4)`

## Task 17: Qwen3-MoE on HIP — FP8-block experts, tensor and expert parallel (S-5)

Files: `crates/turbine-model/src/families/qwen3_moe.rs` (`tp_decoder_spec`, `requirements` with `ExpertDtype::Fp8Block`, `norm_topk_prob` → `RENORMALIZE`), `crates/turbine-model/src/ep.rs` (Qwen3-MoE in the expert-parallel families), `crates/turbine-model/tests/tiny_model.rs`, `crates/turbine-model/tests/golden.rs` (lab test `hip_trace_vs_cpu_qwen3moe`)
Interfaces:

- `Qwen3Moe::tp_decoder_spec() -> Some(DecoderSpec { hooks: [QK_NORM_PER_HEAD, MOE], .. })` (32/4 heads over 2 ranks; experts column-split under TP, expert-range split under EP); routing on the all-reduced hidden row (Task 13)
- Lab config placements: `scripts/lab/phase7-novanas-qwen3moe-tp2.yaml` (`parallel.tensor_parallel_size: 2`), `-ep2.yaml` (`parallel.expert_parallel_size: 2`, `tensor_parallel_size: 1`)
  Covers: S-5 (code)
  Depends on: Tasks 13, 16, 10

- [ ] Write failing host tests `tiny_model qwen3moe_tp2_matches_tp1_on_host` and `qwen3moe_ep2_matches_tp1_on_host` (tiny FP8-block Qwen3-MoE with `norm_topk_prob: true`). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model qwen3moe_` — expect FAIL with "tensor parallel size 2 is not supported by Qwen3MoeForCausalLM"
- [ ] Implement the spec and the EP registration — run, expect PASS; `tp2_matches_tp1_on_host_every_tp_family` now runs Qwen3-MoE too.
- [ ] Lab (ask the user first): `scripts/fetch-checkpoint.sh novanas qwen3-30b-a3b-instruct-2507-fp8` — expect `ok`
- [ ] Lab: `TURBINE_TEST_MODEL_DIR=/models/qwen3-30b-a3b-instruct-2507-fp8 scripts/lab-test.sh novanas --gpus 2 -- -p turbine-model --test golden -- hip_trace_vs_cpu_qwen3moe --include-ignored` (HIP rank traces vs the CPU executor on p01, first 4 layers) — expect exit 0
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): Qwen3-MoE on HIP with FP8 experts, TP2 and EP2`

## Task 18: Qwen3-MoE fixtures and two-placement gate (S-5, S-13)

Files: `tests/golden/qwen3-30b-a3b-instruct-2507-fp8/{reference.jsonl,tolerance.json,README.md}`, `tests/eval/qwen3-30b-a3b-instruct-2507-fp8/{vllm-tp2.json,vllm-ep2.json,turbine-tp2.json,turbine-ep2.json,gate.json}`, `scripts/lab/phase7-novanas-qwen3moe.yaml` (copy of the faster placement), `crates/turbine-core/src/support.rs` (row flips / placement refusal), `.procoder/perf-log.md` ("P7 Qwen3-MoE gate")
Interfaces:

- Reference: transformers on the dequantized BF16 copy (the FP8 checkpoint dequantized by `scripts/golden/dequantize_checkpoint.py`, ~61 GB): built on novanas under `fixture.lock` if `MemAvailable` > 80 GB at start, else on cpuhost; `gate.json` `{"max_drop": 0.02}` (6a FP8 drop)
- A placement that fails its gate and cannot be fixed in this task is refused with a new `PARALLEL_REFUSALS` row `qwen3moe_<tp2|ep2>_gate: …` (exit 2) and recorded `P7 open`
  Covers: S-5 AC, S-13
  Depends on: Tasks 9, 17

- [ ] Fixture (command line `p7-ref-qwen3moe`): `uv run scripts/golden/dequantize_checkpoint.py --model-dir /home/piwi/turbine-models/qwen3-30b-a3b-instruct-2507-fp8 --out <host>/qwen3moe-bf16` then `hf_reference.py --model-dir <host>/qwen3moe-bf16 … --out tests/golden/qwen3-30b-a3b-instruct-2507-fp8/reference.jsonl`; delete the BF16 copy afterwards; README with revision `5a5a776300a41aaa681dd7ff0106608ef2bc90db`.
- [ ] vLLM baseline and ratio (decision 2026-10-08 B): `scripts/lab-perf.sh novanas qwen3moe --placement tp2 --eval` then `--placement ep2 --eval` (background) — expect each `lab-perf: qwen3moe placement=<p> … ratio=` ≥ 0.75 `verdict=PASS` (or `vllm=unavailable … verdict=NO_VLLM` under the no-vLLM rule: try the next rocm/vllm tag once, then golden + GSM8K against the transformers run only); copy each `target/lab-perf/<run id>/vllm-gsm8k.json` to `tests/eval/qwen3-30b-a3b-instruct-2507-fp8/vllm-<placement>.json`
- [ ] Lab: `scripts/lab-cluster.sh --bench-lock qwen3moe-tp2-novanas` then `--bench-lock qwen3moe-ep2-novanas` (background) — expect each `lab-cluster: … PASS golden1=PASS golden16=PASS gsm8k=PASS tok_s=<t>`; copy `target/lab-cluster/<run id>/turbine-<placement>.json` to `tests/eval/qwen3-30b-a3b-instruct-2507-fp8/`
- [ ] Copy the faster passing placement's config to `scripts/lab/phase7-novanas-qwen3moe.yaml`; record both placements' tok/s vs vLLM in perf-log with `P7 closed`/`P7 open` lines per placement; flip or refuse per the outcome.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(qwen3moe): Qwen3-30B-A3B FP8 golden, GSM8K and TP2/EP2 throughput gate`

## Task 19: Mistral on HIP, tensor parallel and the `mistral` tool round trip (S-6)

Files: `crates/turbine-model/src/families/mistral.rs` (`tp_decoder_spec`, refuse `consolidated.safetensors`-only directories), `crates/turbine-model/src/loader.rs` (a directory with both `consolidated.safetensors` and HF shards loads the shards and logs `event="weights_skipped" file=consolidated.safetensors`), `crates/turbine-server/tests/lab_openai.rs` (lab test `mistral_tool_call_round_trip`), `crates/turbine-model/tests/tiny_model.rs`
Interfaces:

- `Mistral::tp_decoder_spec() -> Some(DecoderSpec { hooks: [PLAIN_ATTENTION, SWIGLU], .. })`
- `lab_openai mistral_tool_call_round_trip` (`#[ignore]`, server URL from `TURBINE_LAB_URL`): a `tools` request with `tool_choice: "required"` returns one `tool_calls[0]` with `function.name == "get_weather"` and JSON arguments with `location`; sending the tool result back yields a non-empty assistant `content`
  Covers: S-6 (code), the edge case "a checkpoint with an unexpected tensor (`consolidated.safetensors` present)"
  Depends on: Task 2

- [ ] Write failing host test `tiny_model mistral_shards_preferred_over_consolidated` (the tiny Mistral checkpoint plus a garbage `consolidated.safetensors` loads and matches the plain tiny run; a directory with only `consolidated.safetensors` fails `ModelError::Io` naming `model.safetensors.index.json`). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model mistral_shards` — expect FAIL
- [ ] Implement the loader rule and `tp_decoder_spec` — run `scripts/remote-cargo.sh test -p turbine-model --test tiny_model` — expect PASS.
- [ ] Lab (ask the user first): `scripts/fetch-checkpoint.sh novanas mistral-7b-instruct-v0.3` — expect `ok`
- [ ] Write the lab test; lab: `scripts/lab-serve.sh novanas scripts/lab/phase7-novanas-mistral-7b.yaml` then `TURBINE_LAB_URL=http://192.168.10.203:18000 scripts/remote-cargo.sh test -p turbine-server --test lab_openai mistral_tool_call_round_trip -- --ignored` and `scripts/lab-serve.sh novanas --stop` — expect PASS
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): Mistral on HIP and tensor parallel; HF shards over consolidated`

## Task 20: Mistral-7B fixtures and gate (S-6, S-13)

Files: `tests/golden/mistral-7b-instruct-v0.3/{reference.jsonl,tolerance.json,README.md}`, `tests/eval/mistral-7b-instruct-v0.3/{vllm.json,turbine.json,gate.json}`, `crates/turbine-core/src/support.rs`, `.procoder/perf-log.md` ("P7 Mistral gate")
Interfaces:

- README revision `c170c708c41dac9275d15a8fff4eca08d52bab71`; `gate.json` `{"max_drop": 0.01}`
  Covers: S-6 AC, S-13
  Depends on: Tasks 9, 19, 6

- [ ] Reference (command line `p7-ref-mistral`): `hf_reference.py --model-dir /home/piwi/turbine-models/mistral-7b-instruct-v0.3 --prompts tests/golden/prompts.jsonl --out tests/golden/mistral-7b-instruct-v0.3/reference.jsonl`; tolerance and README.
- [ ] Lab gate: `scripts/lab-bench.sh --model mistral-7b --golden16` — expect golden1=PASS golden16=PASS
- [ ] GSM8K pair at c16 against `scripts/lab-serve.sh novanas --vllm mistral-7b-instruct-v0.3` into `tests/eval/mistral-7b-instruct-v0.3/{turbine,vllm}.json`; `turbine-golden eval-compare --baseline …/vllm.json --candidate …/turbine.json --max-drop 0.01` — expect exit 0
- [ ] Throughput: `scripts/lab-perf.sh novanas mistral-7b --config scripts/lab/phase7-novanas-mistral-7b.yaml` — expect ratio ≥ 0.75
- [ ] Record in perf-log (`P7 closed: MistralForCausalLM/bf16/bf16/tp1 …` on a full pass), flip the row; `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(mistral): Mistral-7B golden, GSM8K and throughput gate`

## Task 21: Mixtral FP8 over two cards — static W8A8 experts, headroom report, placement refusal (S-7)

Files: `crates/turbine-model/src/families/mixtral.rs` (`tp_decoder_spec`, `requirements` with `ExpertDtype::Fp8Tensor`, the `ct_fp8` router `gate` kept BF16 as the checkpoint's `ignore` list says), `crates/turbine-model/src/ep.rs` (Mixtral in the EP families), `crates/turbine-api/src/kv.rs` and the server's KV document (`ranks[].headroom_bytes`), `crates/turbine-server/src/support_startup.rs` (test `mixtral_placement_refusal`), `crates/turbine-model/tests/tiny_model.rs`, `scripts/lab/phase7-novanas-mixtral-{tp2,ep2}.yaml` (`scheduler.max_running_requests` sized from the measured headroom)
Interfaces:

- `/turbine/v1/kv` gains `ranks: [{ rank: u32, device: u32, headroom_bytes: u64 }]` (L0 pool bytes minus model, workspace and the emergency reserve, per rank)
- `support_startup::tests::mixtral_placement_refusal`: a placement whose gate failed is refused with exit 2 and its `PARALLEL_REFUSALS` code before bind; this test asserts the refusal mechanism for a Mixtral row (it stays green whether Task 22 adds a refusal or not, by asserting against `PARALLEL_REFUSALS` contents)
  Covers: S-7 (code), the edge case "Mixtral's thin KV headroom under c16 and long prompts (admission must refuse, never OOM)"
  Depends on: Tasks 13, 16

- [ ] Mixtral tool calls (user decision 2026-10-08 C): commit `crates/turbine-model/src/families/templates/mixtral_tools.jinja` (a tool-aware Mistral template rendering `[AVAILABLE_TOOLS]` from `tools` and `[TOOL_CALLS]` from an assistant turn's `tool_calls`, modelled on vLLM's `examples/tool_chat_template_mistral.jinja`); the Mixtral family selects it when the checkpoint's template renders neither `tools` nor `tool_calls`; flip the `mixtral` case of `formats::tests::family_default_tool_calls_round_trip` from "template drops the call" to a full round trip — expect FAIL first with the old assertion, then PASS.
- [ ] Write failing host tests `tiny_model mixtral_fp8_tp2_matches_tp1_on_host` and `mixtral_fp8_ep2_matches_tp1_on_host`. Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model mixtral_fp8` — expect FAIL; implement — expect PASS.
- [ ] Write failing test `tiny_server kv_document_reports_rank_headroom` (cpu backend, tp 2: `ranks.len() == 2`, each `headroom_bytes` > 0 and ≤ the pool size). Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server kv_document_reports_rank_headroom` — expect FAIL; implement — expect PASS.
- [ ] Write `support_startup::tests::mixtral_placement_refusal` (a synthetic refusal row inserted through the test-only `with_parallel_refusals` seam exits 2 with its code for `MixtralForCausalLM` tp 2, and the real table's Mixtral entries, if any, carry a `mixtral_` code) — run, expect PASS after implementing the seam.
- [ ] Lab (ask the user first; 48 GB): `scripts/fetch-checkpoint.sh novanas mixtral-8x7b-instruct-v0.1-fp8` — expect `ok`
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): Mixtral FP8 over two cards; per-rank KV headroom`

## Task 22: Mixtral fixtures and two-placement gate (S-7, S-13)

Files: `tests/golden/mixtral-8x7b-instruct-v0.1-fp8/{reference.jsonl,tolerance.json,README.md}`, `tests/eval/mixtral-8x7b-instruct-v0.1-fp8/{vllm-tp2.json,vllm-ep2.json,turbine-tp2.json,turbine-ep2.json,gate.json}`, `scripts/lab/phase7-novanas-mixtral.yaml`, `crates/turbine-core/src/support.rs`, `.procoder/perf-log.md` ("P7 Mixtral gate")
Interfaces:

- Reference on cpuhost (dequantized BF16 ~93 GB exceeds novanas RAM headroom): `~/turbine-cpu` checkout, `nice -n 19`, the checkpoint copied from novanas with `rsync`; README revision `9a978d5cc7dabc37b3d13d6501426081c99c4cbd`; `gate.json` `{"max_drop": 0.02}`
  Covers: S-7 AC, S-13
  Depends on: Tasks 9, 21

- [ ] Fixture on cpuhost (command line `p7-ref-mixtral`): dequantize, `hf_reference.py`, copy `reference.jsonl` back; tolerance and README; remove the BF16 copy and the checkpoint copy from cpuhost.
- [ ] vLLM baseline and ratio (decision 2026-10-08 B): `scripts/lab-perf.sh novanas mixtral --placement tp2 --eval` and `--placement ep2 --eval` — expect `ratio=` ≥ 0.75 `verdict=PASS` (or `verdict=NO_VLLM` under the no-vLLM rule); copy each `vllm-gsm8k.json` to `tests/eval/mixtral-8x7b-instruct-v0.1-fp8/vllm-<placement>.json`
- [ ] Lab: `scripts/lab-cluster.sh --bench-lock mixtral-tp2-novanas` and `--bench-lock mixtral-ep2-novanas` — expect `PASS` lines (batched-bounds golden c1/c16, `gsm8k=PASS` within the FP8 drop, `tok_s=`); copy each `turbine-<placement>.json` into `tests/eval/mixtral-8x7b-instruct-v0.1-fp8/`
- [ ] Headroom: in the same lab-cluster runs, `curl -fsS http://127.0.0.1:18000/turbine/v1/kv | jq .ranks` logged with `headroom_bytes` per rank, and the c16 bench with `--prompt-words 512` ends `failed=0` with no `device_oom` in the server log (`grep -c device_oom` = 0)
- [ ] Record per placement in perf-log; flip passing placements, add a `mixtral_<placement>_gate` refusal for a failing one; copy the faster passing config to `scripts/lab/phase7-novanas-mixtral.yaml`; `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows && scripts/remote-cargo.sh test -p turbine-server --bin turbine-server support_startup::tests::mixtral_placement_refusal` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(mixtral): Mixtral FP8 golden, GSM8K, headroom and TP2/EP2 gate`

## Task 23: Attention ABI v2.17 — window and sink fields, d64 and d256 CK instances (S-8, S-11 attention)

Files: `kernels/include/turbine_kernels.h` (`turbine_attention_paged_desc` v2.17: `int32_t window_left` (−1 = none), `const float *sinks` (nullable, `[num_q_heads]` F32 logits); `TURBINE_ABI_MINOR 17u`), `kernels/rocm/CMakeLists.txt` (CK codegen filters: `_sink` and masked variants of `fmha_fwd_pagedkv` at d128 — the pinned commit has no gfx12 pagedkv d64/d256 instance — and of `fmha_fwd_splitkv` at d64, d128 and d256; as built 2026-10-08: d64/d256 prefill and decode both run `fmha_fwd_splitkv` with one split, decision "P7 attention sinks/window instances"), `kernels/rocm/src/paged_attention_splitkv.cpp` and `paged_attention.cpp` (dispatch on head_dim, window, sinks; the head dim a runtime value of CK's own runtime dispatch), `kernels/rocm/src/impl_table.cpp` (`ck_tile_fmha_splitkv` appended to the prefill implementations), `kernels/rocm/src/abi_minor.cpp`, `crates/turbine-kernels/src/ffi.rs`, `crates/turbine-kernels/src/ops/mod.rs` (as built: `AttentionConfig { window: Option<u32>, sinks: bool }` — the existing config type, no separate `PagedAttentionConfig` — and `PagedAttentionContext::sinks: Option<TensorView>`), `crates/turbine-kernels/src/cpu/paged.rs` (reference: sink logit concatenated into the softmax, window mask), `crates/turbine-kernels/tests/hip_ops.rs` (tests), `.procoder/ask/decisions.md` (entry "P7 attention sinks/window instances")
Interfaces:

- Semantics: for query position `p`, keys with `p − window_left ≤ j ≤ p` (all `j ≤ p` when −1); softmax over `{s_j} ∪ {sinks[h]}` where the sink contributes to the denominator only (gpt-oss), read by CK as `sink_ptr[i_nhead]`
- Lab tests (`hip_ops`): `attention_sinks_window_match_reference` (d64 window 128 with sinks, d64 full with sinks, d256 full without sinks; prefill and decode; sequence lengths 1, 127, 128, 129, 1000) and `attention_sinks_batch_invariant` (a sequence's output identical alone and in a batch of 16 mixed lengths, splitkv decode included)
  Covers: S-8 (attention part), S-11 ("CK d256 attention instances"), edge case "sinks with splitkv decode", "long contexts beyond the window"
  Depends on: nothing in this phase

- [ ] Evaluate: build CK at the pinned commit `cd957402` with the d64 / d256 / sink / local-mask filters for `gfx1201` on novanas; record in decisions.md "P7 attention sinks/window instances" which instances codegen and compile, their build time and binary size, and (only if a needed instance does not build) the options A newer CK pin, B Turbine's own paged kernel for that case; ASK THE USER only in that case and record the decision.
- [ ] Write failing CPU tests `cpu::paged::tests::sinks_and_window_match_naive` (against a naive softmax over the masked scores plus the sink term). Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::paged::tests::sinks_and_window` — expect FAIL; implement the reference — expect PASS.
- [ ] Write the two lab tests. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops attention_sinks` — expect FAIL with "TURBINE_E_UNSUPPORTED"
- [ ] Implement the header fields, FFI and the HIP dispatch — run the lab command, expect PASS.
- [ ] Lab: `scripts/lab-bench.sh --quick --model llama` and `--model olmoe` — expect golden1=PASS and tok/s within 1 % of the previous perf-log rows (the d128 path unchanged), perf-log rows.
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(kernels): paged attention window and sinks, d64 and d256 instances (ABI v2.17)`

## Task 24: gpt-oss family module on the CPU executor (S-8)

Files: `crates/turbine-model/src/families/gpt_oss.rs` (new), `crates/turbine-model/src/families/mod.rs` (registered after `Mixtral`), `crates/turbine-model/src/executor/decoder/hooks/` (`ATTENTION_SINKS_WINDOW` hook: biases on q/k/v/o, sinks, per-layer window from `LayerKind`), `crates/turbine-model/src/testing/tiny.rs` (`gpt_oss` tiny writer: 4 layers alternating `sliding_attention` 128 / `full_attention`, head_dim 64, 4 experts top-2, MXFP4 experts with biases, YaRN), `crates/turbine-model/tests/fixtures/gpt-oss-20b/` (real `config.json`, `tokenizer_config.json`, `chat_template.jinja`), `crates/turbine-model/tests/fixtures/tiny-gpt-oss-trace.json` (committed transformers per-op trace of the tiny checkpoint), `scripts/golden/tiny_trace.py` (new fixture tool: transformers per-op trace of a tiny checkpoint), `crates/turbine-model/examples/write_tiny.rs` (new: writes a registered family's tiny checkpoint, `write_tiny <family> <seed> <dir>`), `crates/turbine-model/tests/tiny_model.rs` (test `gpt_oss_matches_reference`)
Interfaces:

- `pub struct GptOss;` `hf_architectures() == ["GptOssForCausalLM"]`; `layer_kinds` from `layer_types` (`sliding_attention` → `SlidingWindow { window: sliding_window }`); `state_layout` empty; `default_tool_format() == Some("harmony")` and `default_reasoning_parser() == Some("harmony")` (both registered in Task 25; until then the family test asserts the names only)
- Router: `TopkThenSoftmax` with `router.bias`; experts `ExpertDtype::Mxfp4`, `ExpertActivation::GptOssClamp { limit: 7.0 }`, `gate_up_interleaved: true`, biases; attention `AttentionConfig { window: Some(128) | None, sinks: true }` (Task 23 as built), head_dim 64
- `scripts/golden/tiny_trace.py --model-dir <tiny> --prompt-ids 1,2,3,4,5 --out <json>` — per-layer hidden state, attention output, router top-k and final logits in F32
  Covers: S-8 AC `tiny_model gpt_oss_matches_reference`; S-1 (gpt-oss row of `layer_kinds_and_state_layout`)
  Depends on: Tasks 2, 4, 5, 15, 23

- [ ] Write failing test `tiny_model gpt_oss_matches_reference`: write the tiny gpt-oss checkpoint with seed 7, run the CPU executor with `set_trace`, compare each op of `tests/fixtures/tiny-gpt-oss-trace.json` within the Phase 1 trace bounds (max |Δ| 2e-2 on hidden states, logits top-5 order equal). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model gpt_oss_matches_reference` — expect FAIL with "unregistered architecture GptOssForCausalLM"
- [ ] Generate the trace fixture (fixture time, novanas `fixture.lock`): `cargo run -p turbine-model --example write_tiny -- gpt_oss 7 <dir>` then `uv run scripts/golden/tiny_trace.py --model-dir <dir> --prompt-ids 1,2,3,4,5 --out crates/turbine-model/tests/fixtures/tiny-gpt-oss-trace.json`.
- [ ] Implement the family, the hook and the tiny writer — run the test, expect PASS; `scripts/remote-cargo.sh test -p turbine-model families::tests::layer_kinds_and_state_layout` — expect PASS with the gpt-oss alternation.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model && scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::overlap_matches_serial_every_family` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): gpt-oss family (sinks, alternating window, MXFP4 experts) on the CPU executor`

## Task 25: Harmony — chat template, tool-call format, reasoning parser, `reasoning_effort` (S-9, S-10)

Files: `crates/turbine-model/src/formats/harmony.rs` (new `ToolFormat`), `crates/turbine-model/src/formats/mod.rs` (registered: `[&Llama3Json, &Hermes, &Mistral, &Harmony]`), `crates/turbine-model/src/formats/tests.rs` (test `harmony_round_trip`; gpt-oss added to `family_default_tool_calls_round_trip`), `crates/turbine-model/src/reasoning/harmony.rs` (new `ReasoningParser`), `crates/turbine-model/src/reasoning/mod.rs` (registered), `crates/turbine-model/src/chat_template.rs` (the request's `reasoning_effort` reaches the template context as `reasoning_effort`, default `medium`), `crates/turbine-api/src/openai/request.rs` (`reasoning_effort: Option<ReasoningEffort>`, other values → 400 `invalid_request_error` naming `reasoning_effort`), `crates/turbine-server/tests/lab_openai.rs` (lab test `harmony_channels`), `crates/turbine-server/tests/tiny_server.rs` (host test `reasoning_effort_rejects_unknown`)
Interfaces:

- `pub enum ReasoningEffort { Low, Medium, High }` (serde lowercase)
- Harmony output grammar: `<|channel|>analysis<|message|>…<|end|>` → reasoning; `<|start|>assistant<|channel|>commentary to=functions.<name> <|constrain|>json<|message|>{…}<|call|>` → one tool call; `<|channel|>final<|message|>…<|return|>` → content; a tool call in the analysis channel is a tool call too; markers arrive split across tokens and pushes
- `Harmony` (`ToolFormat`, name `harmony`): `special_tokens` `<|channel|>`, `<|message|>`, `<|call|>`, `<|constrain|>`, `<|end|>`, `<|return|>`, `<|start|>` (required); `grammar` constrains `required`/named calls to the commentary-channel call form; `sample_call` = the commentary form of `get_weather`
- `Harmony` (`ReasoningParser`, name `harmony`): channel state machine; `ReasoningDelta.content` carries only the final channel; tool-call text is withheld for the tool parser
  Covers: S-9 (both ACs), S-10 (`harmony` in `registry_conformance`), S-18 (gpt-oss default)
  Depends on: Tasks 7, 8, 24

- [ ] Write failing test `formats::tests::harmony_round_trip`: (1) a full output `"<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"Oslo\",\"unit\":\"celsius\"}<|call|>"` parses to reasoning `think` and one call; (2) `"<|channel|>analysis<|message|>a<|end|><|start|>assistant<|channel|>final<|message|>b<|return|>"` gives reasoning `a`, content `b`; (3) the same fed one char at a time and split inside `<|chan` / `nel|>` equals (2); (4) the rendered gpt-oss template fixture with `reasoning_effort: "high"` contains `Reasoning: high`, and without it `Reasoning: medium`. Run: `scripts/remote-cargo.sh test -p turbine-model formats::tests::harmony_round_trip` — expect FAIL with "cannot find `harmony`"
- [ ] Implement both modules and the template context — run `scripts/remote-cargo.sh test -p turbine-model formats:: reasoning::` — expect PASS (conformance suites included).
- [ ] Write failing test `tiny_server reasoning_effort_rejects_unknown` (`"reasoning_effort": "extreme"` → 400 naming `reasoning_effort`). Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server reasoning_effort` — expect FAIL; implement — expect PASS.
- [ ] Write the lab test `lab_openai harmony_channels` (`#[ignore]`, gpt-oss served, `TURBINE_LAB_URL`): a plain question yields non-empty `reasoning_content` and `content`; a tools request with `tool_choice: "required"` yields `tool_calls[0].function.name == "get_weather"` with valid JSON arguments; the streamed `reasoning_content` / `content` deltas concatenate to the non-streamed ones for the same seed with `temperature: 0`; `n: 2` streams carry independent channels; `reasoning_effort: "extreme"` → 400. It runs in Task 26.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model,api): Harmony template, tool format and reasoning parser; reasoning_effort`

## Task 26: gpt-oss on HIP (S-8)

Files: `crates/turbine-model/src/families/gpt_oss.rs` (`requirements` over the v2.16 experts and v2.17 attention), `crates/turbine-model/src/executor/decoder/hooks/` (HIP path of `ATTENTION_SINKS_WINDOW`; attention biases fused as GEMM bias where the gemm desc supports it, else `turbine_add`), `crates/turbine-model/tests/golden.rs` (lab test `hip_trace_vs_cpu_gpt_oss`), `crates/turbine-model/tests/tiny_model.rs` (lab test `hip_trace_vs_cpu_gpt_oss_tiny`)
Interfaces:

- `golden hip_trace_vs_cpu_gpt_oss` (`TURBINE_TEST_MODEL_DIR=/models/gpt-oss-20b`, release build): HIP vs CPU per op on p01 within the Phase 1 trace bounds, including a decode step past position 128 (window active) through splitkv
  Covers: S-8 AC (`hip_ops` and `golden hip_trace_vs_cpu_gpt_oss` lab commands)
  Depends on: Tasks 16, 23, 24, 25

- [ ] Lab (ask the user first): `scripts/fetch-checkpoint.sh novanas gpt-oss-20b` — expect `ok` and no `original/` directory
- [ ] Write both lab tests. Run: `TURBINE_TEST_MODEL_DIR=/models/gpt-oss-20b scripts/lab-test.sh novanas -- -p turbine-model --test golden hip_trace_vs_cpu_gpt_oss` — expect FAIL with "kernel_op_unavailable" or a trace mismatch
- [ ] Implement the HIP path — run the lab command and `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops attention_sinks_window_match_reference attention_sinks_batch_invariant` — expect exit 0 each.
- [ ] Lab: `scripts/lab-serve.sh novanas scripts/lab/phase7-novanas-gpt-oss-20b.yaml`, `TURBINE_LAB_URL=http://192.168.10.203:18000 scripts/remote-cargo.sh test -p turbine-server --test lab_openai harmony_channels -- --ignored` — expect PASS; `scripts/lab-serve.sh novanas --stop`
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): gpt-oss-20b on HIP`

## Task 27: gpt-oss-20b fixtures and gate; `openai_mxfp4` packaging row (S-8, S-13)

Files: `tests/golden/gpt-oss-20b/{reference.jsonl,tolerance.json,README.md}`, `tests/eval/gpt-oss-20b/{vllm.json,turbine.json,gate.json}`, `crates/turbine-core/src/support.rs` (`GptOssForCausalLM` mxfp4 row and the `openai_mxfp4` packaging row), `.procoder/perf-log.md` ("P7 gpt-oss gate")
Interfaces:

- Golden and GSM8K are judged on the final channel (decision Harmony A): `turbine-golden compare` compares token logprobs of the whole output as today; GSM8K extracts the answer from `content`; the reference is transformers on the dequantized MXFP4 → BF16 weights (~40 GB, novanas `fixture.lock`), `reasoning_effort` `low` in the system message of both runs
- `gate.json` `{"max_drop": 0.04}` (6a 4-bit drop) when the baseline is the BF16 transformers run; `0.01` against vLLM-ROCm on the same checkpoint
- One logged GSM8K run with `reasoning_effort: high` (not gated), per decision reasoning A
  Covers: S-8 AC (lab gate, `--support-matrix` rows), S-13
  Depends on: Tasks 9, 26

- [ ] Fixture (command line `p7-ref-gpt-oss`): `hf_reference.py --model-dir /home/piwi/turbine-models/gpt-oss-20b … --out tests/golden/gpt-oss-20b/reference.jsonl`; tolerance and README (revision `6cee5e81ee83917806bbde320786a8fb61efebee`).
- [ ] Lab: `scripts/lab-bench.sh --model gpt-oss-20b --golden16` — expect golden1=PASS golden16=PASS
- [ ] GSM8K pair at c16 against `scripts/lab-serve.sh novanas --vllm gpt-oss-20b` (if the pinned tag fails on gfx1201: one newer tag, else the no-vLLM rule against the transformers run with `max_drop` 0.04); `eval-compare` — expect exit 0; throughput `scripts/lab-perf.sh novanas gpt-oss-20b --config scripts/lab/phase7-novanas-gpt-oss-20b.yaml` — expect ratio ≥ 0.75 or `vllm=unavailable` logged with the reason.
- [ ] Record in perf-log; on a pass flip `GptOssForCausalLM/mxfp4` and the `openai_mxfp4` packaging row to `supported`; verify `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix | grep -E 'GptOssForCausalLM|openai_mxfp4'` — expect `supported`; `scripts/remote-cargo.sh test -p turbine-core support::` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(gpt-oss): gpt-oss-20b golden, GSM8K and throughput gate; openai_mxfp4 supported`

## Task 28: MXFP4 p05 trace and the `mxfp4_a4` gate (S-14)

Files: `crates/turbine-model/tests/golden.rs` (`mxfp4_decode_vs_prefill_trace` extended to a CPU-vs-transformers per-op table on p05), `scripts/golden/tiny_trace.py` (`--prompt-file` / `--prompt-id` mode for a real checkpoint), the cause's site if a fix is found, `tests/golden/llama-3.2-3b-instruct-mxfp4-a4/{tolerance.json,README.md}` (self-spread tolerance), `tests/eval/llama-3.2-3b-instruct-mxfp4-a4/{turbine.json,gate.json}`, `crates/turbine-core/src/support.rs` (`mxfp4` and `mxfp4_a4` Llama rows), `.procoder/perf-log.md` (sections "P7 MXFP4 p05 trace" and "P7 mxfp4_a4 gate")
Interfaces:

- `golden mxfp4_cpu_vs_transformers_p05` (`#[ignore]`, `TURBINE_TEST_MODEL_DIR=/models/llama-3.1-8b-instruct-mxfp4a16`, `TURBINE_P05_TRACE=tests/golden/llama-3.1-8b-instruct-mxfp4a16/p05-trace.json`): prints per layer and op the max |Δ| of Turbine's CPU path against the transformers trace and the first op above 1e-2
- `mxfp4_a4` tolerance: `uv run scripts/golden/self_spread.py` over the checkpoint (method of `tests/golden/olmoe-1b-7b-0125-instruct/README.md`); `gate.json` `{"max_drop": 0.04}`
  Covers: S-14 AC
  Depends on: Task 27 (the MXFP4 work), Task 6

- [ ] Fixture (command line `p7-ref-mxfp4-p05`): `uv run scripts/golden/tiny_trace.py --model-dir /home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4a16 --prompt-file tests/golden/prompts.jsonl --prompt-id p05 --out tests/golden/llama-3.1-8b-instruct-mxfp4a16/p05-trace.json` (the 8B checkpoint sits on iSCSI since the disk decision: copy it back to local disk first, ask the user if local space is short).
- [ ] Lab: `TURBINE_TEST_MODEL_DIR=/models/llama-3.1-8b-instruct-mxfp4a16 scripts/lab-test.sh novanas -- -p turbine-model --test golden mxfp4_cpu_vs_transformers_p05` — expect the table; write perf-log "P7 MXFP4 p05 trace" with the table and the cause (or the evidence that it is the reference's own BF16 rounding).
- [ ] If a fix follows from the cause, land it with a host test naming the rounding site, then `scripts/lab-bench.sh --model llama --golden16 -- --set model.path=/home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4a16` (golden against `tests/golden/llama-3.1-8b-instruct-mxfp4a16/reference.jsonl`); on golden PASS add `- P7 closed: LlamaForCausalLM/mxfp4/bf16/tp1` and flip the row.
- [ ] `mxfp4_a4`: self-spread tolerance (command line `p7-spread-mxfp4a4`) into `tests/golden/llama-3.2-3b-instruct-mxfp4-a4/tolerance.json` with the derivation in its README; golden c1/c16 against it and GSM8K c16 vs the BF16 Llama-3.2-3B baseline (`eval-compare --max-drop 0.04`); on both PASS add `- P7 closed: LlamaForCausalLM/mxfp4_a4/bf16/tp1` and flip the row, else `P7 open` with the numbers.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(mxfp4): p05 CPU-vs-transformers trace; mxfp4_a4 self-spread gate`

## Task 29: Gated DeltaNet provider — evaluation and decision (S-11)

Files: `kernels/rocm/tools/gdn_eval/` (new HIP harness), `.procoder/ask/decisions.md` (entry "P7 Gated DeltaNet provider")
Interfaces:

- Shapes: Qwen3.5-9B / Qwen3.6-35B GDN (16 key heads × 128, 32 value heads × 128, conv k=4, F32 state); decode batch 1, 16, 64; prefill 128, 2048, 16384 tokens; chunk boundaries at 64
- Candidates: llama.cpp `ggml_gated_delta_net` + `ssm_conv` (ggml-cuda sources built through ggml-hip for gfx1201; MIT, vendored under `kernels/rocm/third_party/llama.cpp/` with its licence) for decode and short prefill; AMD AITER `chunk_gated_delta_rule_fwd_h.cu` (gfx12 WMMA path) plus `causal_conv1d_update.cu` / `causal_conv1d_fwd_split_qkv.cu` for long prefill, with the remaining chunk stages (kkt, solve, w/u, o) as Turbine HIP kernels only if AITER's are Triton/FlyDSL; own kernel only where both fail or are slower
- Reference: a CPU F32 recurrence (the transformers torch fallback math) for correctness in the harness
  Covers: S-11 AC (decisions entry with timings)
  Depends on: nothing in this phase

- [ ] Build and run the harness on novanas under `scripts/bench-lock.sh`: correctness against the F32 recurrence (max |Δ| on outputs and final state) and median µs per shape for each candidate.
- [ ] Write the decisions.md entry "P7 Gated DeltaNet provider" with the timing table and options: A llama.cpp decode + AITER chunk-h prefill (+ own glue stages), B llama.cpp everywhere, C own kernels where measurably faster.
- [ ] ASK THE USER for the decision; record it under the entry.
- [ ] Commit: `docs(decisions): P7 Gated DeltaNet provider evaluation`

## Task 30: GDN and causal-conv ABI v2.18 and the HIP kernels (S-11)

Files: `kernels/include/turbine_kernels.h` (`turbine_gdn_desc`, `turbine_gdn_prefill`, `turbine_gdn_decode`, `turbine_conv1d_desc`, `turbine_conv1d_prefill`, `turbine_conv1d_update`, each with `_supported`/`_impl`; `TURBINE_ABI_MINOR 18u`), `kernels/rocm/src/gdn.hip` / vendored providers per Task 29, `kernels/rocm/src/conv1d.hip`, `kernels/rocm/CMakeLists.txt`, `kernels/rocm/third_party/LICENSES/`, `crates/turbine-kernels/src/ffi.rs`, `crates/turbine-kernels/src/ops/mod.rs` (`GdnConfig`, `Conv1dConfig`), `crates/turbine-kernels/src/cpu/gdn.rs` (new reference), `crates/turbine-kernels/src/cpu/mod.rs`, `crates/turbine-kernels/tests/hip_gdn.rs` (new)
Interfaces:

- `turbine_gdn_desc { const void *q, *k, *v; const float *g, *beta; float *state /* [seqs, v_heads, k_dim, v_dim] F32, updated in place */; const int32_t *state_slots; const int32_t *cu_seqlens; void *out; float *boundary_states /* nullable: the state after every boundary_tokens tokens of each sequence (Task 32 snapshots) */; int32_t boundary_tokens; int32_t num_tokens, num_seqs, k_heads, v_heads, k_dim, v_dim; int32_t dtype; }`; `turbine_conv1d_desc { const void *x, *w, *bias; void *conv_state /* [seqs, channels, k-1] */; const int32_t *state_slots, *cu_seqlens; void *out; void *boundary_conv_states /* nullable, as boundary_states */; int32_t boundary_tokens; int32_t num_tokens, num_seqs, channels, kernel; int32_t activation /* 1 = SiLU */; int32_t dtype; }`
- Rust `GdnConfig { k_heads, v_heads, k_dim, v_dim, mode: GdnMode::{Prefill, Decode} }`, `Conv1dConfig { channels, kernel: 4, mode }`
- Lab tests (`hip_gdn`): `gdn_prefill_matches_reference` (lengths 1, 63, 64, 65, 2048, a chunked-prefill split at 1000 equal to the unsplit run, and `boundary_states` at every 128 tokens equal to the state of a prefill stopped there), `gdn_decode_matches_reference`, `conv1d_state_update_matches_reference`, `gdn_batch_invariant`
  Covers: S-11 AC (`hip_gdn`), edge case "chunked prefill splitting a GDN chunk boundary"
  Depends on: Task 29

- [ ] Write failing CPU tests `cpu::gdn::tests::recurrence_matches_naive` and `cpu::gdn::tests::split_prefill_equals_whole` (state carried across a split). Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::gdn` — expect FAIL with "could not find `gdn`"; implement the reference — expect PASS.
- [ ] Write the four lab tests. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_gdn` — expect FAIL with "TURBINE_E_UNSUPPORTED"
- [ ] Implement header, FFI and HIP per the decision — run, expect PASS.
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(kernels): gated delta net and causal conv1d (ABI v2.18)`

## Task 31: Qwen3.5 hybrid family (dense) on CPU and HIP (S-11, S-18 part 2)

Files: `crates/turbine-model/src/families/qwen3_5.rs` (new: `Qwen3_5ForConditionalGeneration`), `crates/turbine-model/src/families/mod.rs` (registered), `crates/turbine-model/src/executor/decoder/hooks/` (`GATED_DELTA_NET` hook: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, conv1d, `A_log`, `dt_bias`, gated RMSNorm `norm`, `out_proj`; `ATTENTION_OUTPUT_GATE`; partial rotary through `turbine_rope_desc.rotary_dim = head_dim / 4`; the `(1 + w)` RMSNorm form), `crates/turbine-model/src/loader.rs` (skips `model.visual.*` and `mtp.*`, logs `event="weights_skipped" prefix=<p> tensors=<n> bytes=<b>`), `crates/turbine-model/src/testing/tiny.rs` (`qwen3_5` writer: 8 layers, `full_attention_interval` 4, head_dim 256 → 64 in tiny, GDN heads 2/4), `crates/turbine-model/tests/fixtures/qwen3.5-9b/` (real `config.json`, `tokenizer_config.json` with the chat template), `crates/turbine-model/tests/fixtures/tiny-qwen3_5-trace.json`, `crates/turbine-model/src/formats/` (a new `qwen3_xml` format file only if the template's tool form is the XML `<function=…>` form), `crates/turbine-model/src/formats/tests.rs`, `crates/turbine-model/tests/tiny_model.rs` (tests), `crates/turbine-model/tests/golden.rs` (lab test `hip_trace_vs_cpu_qwen3_5`)
Interfaces:

- `pub struct Qwen3_5;` `hf_architectures() == ["Qwen3_5ForConditionalGeneration"]`; `parse_config` reads `text_config` (`model_type` `qwen3_5_text`); `layer_kinds` from `layer_types` (`linear_attention` → `LinearAttention`); `state_layout` = `{ linear_layers, recurrent_bytes_per_layer: v_heads × k_dim × v_dim × 4, conv_bytes_per_layer: (2·k_heads·k_dim + v_heads·v_dim) × (k − 1) × 2, dtype: Some(F32) }` (conv state in the activation dtype, recurrent in `mamba_ssm_dtype` F32); `default_reasoning_parser() == Some("qwen_think")`; `default_tool_format()` = `hermes` or the new `qwen3_xml`, decided by the round-trip test on the real template
- The executor reads/writes state through `StateSlots` (Task 32); until then it holds a private per-sequence slot map (CPU only) so this task's tests run
  Covers: S-11 (family, `weights_skipped`), S-1 (Qwen3.5 row of `layer_kinds_and_state_layout`), S-18 (Qwen3.5 format)
  Depends on: Tasks 2, 7, 8, 23, 30

- [ ] Write failing test `tiny_model qwen3_5_matches_reference` (CPU vs the committed transformers per-op trace of the tiny checkpoint, `scripts/golden/tiny_trace.py` as in Task 24, run with transformers' torch GDN fallback). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model qwen3_5_matches_reference` — expect FAIL with "unregistered architecture Qwen3_5ForConditionalGeneration"
- [ ] Generate the trace fixture (fixture time); implement the family, the hooks, the loader skip and the tiny writer — run, expect PASS; `families::tests::layer_kinds_and_state_layout` — expect PASS with 3 linear : 1 full and the F32 state bytes.
- [ ] Write failing test `tiny_model weights_skipped_logged_for_vision_and_mtp` (a tiny checkpoint with extra `model.visual.blocks.0.attn.qkv.weight` and `mtp.layers.0.mlp.gate_proj.weight` loads, both prefixes logged with `event="weights_skipped"` through a tracing capture). Run it — expect FAIL; implement — expect PASS.
- [ ] Add the Qwen3.5 template to `family_default_tool_calls_round_trip`; if `hermes` does not parse its rendered call, add `formats/qwen3_xml.rs` (registered, its conformance suite run by `registry_conformance`) and make it the default — run `scripts/remote-cargo.sh test -p turbine-model formats::` — expect PASS.
- [ ] Lab (ask the user first): `scripts/fetch-checkpoint.sh novanas qwen3.5-9b` — expect `ok`
- [ ] Lab: `TURBINE_TEST_MODEL_DIR=/models/qwen3.5-9b scripts/lab-test.sh novanas -- -p turbine-model --test golden hip_trace_vs_cpu_qwen3_5` — expect exit 0 and, in the test log, `event="weights_skipped"` for `model.visual` and `mtp`
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): Qwen3.5 Gated DeltaNet hybrid (text only) on CPU and HIP`

## Task 32: Hybrid state slots, block-boundary snapshots in L0 and prefix reuse (S-12)

Files: `crates/turbine-kv/src/state.rs` (new: slot pool, snapshot references, copies), `crates/turbine-kv/src/lib.rs`, `crates/turbine-kv/src/directory.rs` (a hybrid full block carries its `SnapshotRef`; partial entries carry none), `crates/turbine-kv/src/planner.rs` (a hybrid prefix hit ends at the last full block with a snapshot; the mid-block remainder is recomputed, no partial-block reuse for hybrids), `crates/turbine-scheduler/src/` (`IterationPlan.state_copies`, slot allocation with the sequence, release on finish/cancel/preemption, `n` > 1 fork copies), `crates/turbine-model/src/executor/mod.rs` (`ModelExecutor::copy_state`, `StepSeq.state_slot`, snapshot targets), `crates/turbine-model/src/families/qwen3_5.rs` (uses `StateSlots` instead of its private map), `crates/turbine-model/src/budget.rs` (a hybrid L0 block costs KV + snapshot bytes), `crates/turbine-server/src/engine/loop.rs` (executes `state_copies` with the block copies; test `hybrid_prefix_reuse_matches_cold`), `crates/turbine-server/src/engine/grammar.rs` (jump-forward refusal for hybrids), `crates/turbine-server/src/metrics.rs`
Interfaces:

- `turbine_kv::state::StateSlotPool::new(layout: StateLayout, slots: u32)`; `alloc(&mut self) -> Result<StateSlot, KvError>`; `release(&mut self, slot: StateSlot)`; `free(&self) -> u32`; `StateSlot(u32)`
- `turbine_kv::state::SnapshotPool::new(layout: StateLayout, entries: u32)` (snapshots live in their own pool, not in KV pages; the budget charges `entries × layout.bytes_per_sequence()`); `turbine_kv::state::SnapshotRef { block: BlockId, entry: u32 }` — taken at every `kv.hybrid_snapshot_interval_blocks`-th full block (config key `turbine_core::config::KvConfig::hybrid_snapshot_interval_blocks: u32`, default 8, ≥ 1) and at the last full block of a finished sequence; a block without a snapshot carries `None`; a hit restores the nearest snapshot at or before the match and recomputes the rest (user decision 2026-10-08 1 B); `StateLoc::{Slot(StateSlot), Snapshot(SnapshotRef)}`; `StateCopy { src: StateLoc, dst: StateLoc }`
- `IterationPlan { …, state_copies: Vec<StateCopy> }` (executed before the step, after `tail_copies`); `ModelExecutor::copy_state(&mut self, copies: &[StateCopy]) -> Result<(), ModelError>` (TP ranks copy their own head shard)
- `StepSeq { …, state_slot: Option<u32>, snapshot_blocks: Vec<(u32 /* token offset within the step */, BlockId)> }` — the executor writes the boundary state (Task 30 `boundary_states`) into each block's snapshot area
- Jump-forward on a family with state: off; an explicit `structured_output.jump_forward: true` exits 2 with `jump_forward_hybrid`; otherwise WARN `event="jump_forward_disabled" reason="jump_forward_hybrid"` once at startup
- Metrics: `turbine_hybrid_state_bytes` (gauge: slot bytes in use + snapshot bytes held in L0), `turbine_hybrid_state_snapshots_total{outcome="stored|restored|recomputed"}` (`recomputed` = a hit whose state had to be rebuilt by prefill because the snapshot was missing)
  Covers: S-12 AC `engine::r#loop::tests::hybrid_prefix_reuse_matches_cold`; edge cases (mid-block hit, preemption by recompute, cancellation, `n` > 1, jump-forward on a hybrid, a decode batch mixing hybrid and full-attention-only sequences)
  Depends on: Tasks 3, 30, 31

- [ ] Write failing unit tests `state::tests::slots_alloc_release_and_exhaust` (pool of 4: 4 allocs succeed, the 5th errors `KvError::StateSlotsExhausted`, release restores) and `planner::tests::hybrid_hit_stops_at_last_snapshot` (a 3.5-block match plans reuse of 3 blocks plus a restore from block 3's snapshot and recompute of the half block). Run: `scripts/remote-cargo.sh test -p turbine-kv state:: planner::tests::hybrid` — expect FAIL with "could not find `state`"
- [ ] Write failing test `planner::tests::hybrid_hit_uses_nearest_snapshot` (interval 2, a 5-block cached chain with snapshots at blocks 2 and 4 and at the finished sequence's last block 5: a prompt matching 3 blocks plans KV reuse of 3 blocks, state restore from block 2 and recompute of block 3's tokens; a prompt matching all 5 restores block 5's snapshot) — expect FAIL with "no function or associated item named `nearest_snapshot`"; and `config::tests::hybrid_snapshot_interval` in turbine-core (default 8; `0` rejected naming the key).
- [ ] Implement `state.rs` (`StateSlotPool`, `SnapshotPool`), the config key, the directory/planner rules — run, expect PASS.
- [ ] Write failing test `engine::r#loop::tests::hybrid_prefix_reuse_matches_cold` (tiny Qwen3.5, CPU, `kv.block_tokens` 16): a three-turn conversation hits at 1, 2 and 3 full blocks and once mid-block, each turn's greedy tokens and logprobs bit-equal a cold server's; an `n: 2` request equals two separate requests; a request cancelled mid-decode and one preempted by recompute (forced with `scheduler.max_running_requests: 1` and a higher-priority arrival) leave `StateSlotPool::free()` at its start value and the preempted request's output equals its cold run; `turbine_hybrid_state_snapshots_total{outcome="restored"}` > 0. Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::hybrid_prefix_reuse_matches_cold` — expect FAIL
- [ ] Implement the scheduler slot lifecycle, `state_copies`, the executor `copy_state` and snapshot writes, the budget term and the metrics — run, expect PASS; `engine::r#loop::tests::overlap_matches_serial_every_family` — expect PASS with Qwen3.5.
- [ ] Write failing test `server_cli jump_forward_on_hybrid_exits_2` (tiny Qwen3.5, `--set structured_output.jump_forward=true --check-config` exits 2 naming `jump_forward_hybrid`); implement — expect PASS.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv && scripts/remote-cargo.sh test -p turbine-scheduler && scripts/remote-cargo.sh test -p turbine-server` — expect PASS
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(kv,scheduler): hybrid state slots and block-boundary snapshots with prefix reuse`

## Task 33: Snapshots through the L1/L2 tiers and the hybrid GPU reuse test (S-12)

Files: `crates/turbine-kv/src/hierarchy.rs` (demotion/promotion/eviction move a block's snapshot bytes with its KV; lossy tier formats transcode the KV part only, the snapshot stays exact), `crates/turbine-kv/src/tier/` (slab layout: KV bytes then snapshot bytes, the slab format version bumped), `crates/turbine-api/src/kv.rs` (`/turbine/v1/kv` `tiers[].state_bytes`), `crates/turbine-server/tests/kv_gpu.rs` (lab test `hybrid_prefix_reuse_matches_cold`)
Interfaces:

- Slab record for a hybrid block: `[kv bytes in the tier format][snapshot bytes exact]`; `SLAB_FORMAT_VERSION` + 1; a lineage key never shares a snapshot across lineages
- `/turbine/v1/kv` `tiers[] { …, state_bytes: u64 }`
- `kv_gpu hybrid_prefix_reuse_matches_cold` (`TURBINE_TEST_MODEL_DIR=/models/qwen3.5-9b`, `scripts/lab/phase4-novanas.yaml` tier sizes): multi-turn prefix hits after forced demotion to L1 and to L2 equal the cold run bit for bit; `turbine_hybrid_state_snapshots_total{outcome="restored"}` > 0
  Covers: S-12 ACs (`hierarchy::tests::state_snapshot_follows_block`, `kv_gpu hybrid_prefix_reuse_matches_cold`)
  Depends on: Task 32

- [ ] Write failing test `hierarchy::tests::state_snapshot_follows_block`: a hybrid block with a known snapshot pattern demoted to L1 (`l0` format), to L2 (`tq4` format) and promoted back to L0 keeps its snapshot bytes identical at each step, and an eviction drops both. Run: `scripts/remote-cargo.sh test -p turbine-kv hierarchy::tests::state_snapshot_follows_block` — expect FAIL
- [ ] Implement the slab layout and the hierarchy moves — run, expect PASS; `scripts/remote-cargo.sh test -p turbine-kv && scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim` — expect PASS.
- [ ] Add `state_bytes` to the KV document with a `tiny_server` assertion `kv_document_reports_state_bytes` (written failing first, then passing).
- [ ] Write the lab test. Run: `TURBINE_TEST_MODEL_DIR=/models/qwen3.5-9b scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu hybrid_prefix_reuse_matches_cold` — expect exit 0
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(kv): hybrid state snapshots travel with their blocks through L1 and L2`

## Task 34: Qwen3.6 MoE hybrid — shared expert, FP8 experts, TP2 and EP2 (S-11)

Files: `crates/turbine-model/src/families/qwen3_5_moe.rs` (new: `Qwen3_5MoeForConditionalGeneration`, reuses the Qwen3.5 layer hooks), `crates/turbine-model/src/families/mod.rs` (registered), `crates/turbine-model/src/families/qwen3_5.rs` (`tp_decoder_spec`: GDN heads and state sharded by value head; kv heads 4 → 2 per rank), `crates/turbine-model/src/ep.rs`, `crates/turbine-model/src/testing/tiny.rs` (`qwen3_5_moe` writer: 8 experts top-2, shared expert with `shared_expert_gate`), `crates/turbine-model/tests/fixtures/qwen3.6-35b-a3b-fp8/` (real configs), `crates/turbine-model/tests/fixtures/tiny-qwen3_5_moe-trace.json`, `crates/turbine-model/tests/tiny_model.rs`, `crates/turbine-model/src/tp.rs` (vocab 248320 sharding)
Interfaces:

- `pub struct Qwen3_5Moe;` routing: softmax → top-k with renormalisation (always on: `norm_topk_prob` absent from the config class), experts `ExpertDtype::Fp8Block`, shared expert via `shared_inter` and `shared_gate_w` (Task 15)
- Under TP2 kv heads 2 → 1 per rank (Qwen3.6-35B) and the state's `v_heads` 32 → 16 per rank; under EP2 256 experts → 128 per rank, attention and GDN replicated
  Covers: S-11 (Qwen3.6 code), edge cases "kv heads 2 or 4 over two ranks, vocab 248320 sharding, 256 experts over two ranks"
  Depends on: Tasks 16, 31, 32, 13

- [ ] Write failing tests `tiny_model qwen3_5_moe_matches_reference` (CPU vs committed tiny transformers trace), `qwen3_5_moe_tp2_matches_tp1_on_host`, `qwen3_5_moe_ep2_matches_tp1_on_host`, `qwen3_5_tp2_matches_tp1_on_host` (dense hybrid with kv heads 4). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model qwen3_5` — expect FAIL with "unregistered architecture Qwen3_5MoeForConditionalGeneration"
- [ ] Generate the tiny trace fixture; implement the family, the TP spec and EP registration — run, expect PASS; `families::tests::layer_kinds_and_state_layout` — expect PASS.
- [ ] Lab (ask the user first; 38 GB): `scripts/fetch-checkpoint.sh novanas qwen3.6-35b-a3b-fp8` — expect `ok`
- [ ] Lab: `TURBINE_TEST_MODEL_DIR=/models/qwen3.6-35b-a3b-fp8 scripts/lab-test.sh novanas --gpus 2 -- -p turbine-model --test golden hip_trace_vs_cpu_qwen3_5` (the test runs tp 2 when two devices are visible) — expect exit 0
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(model): Qwen3.6 MoE hybrid with shared expert, FP8 experts, TP2 and EP2`

## Task 35: Hybrid fixtures and gates — Qwen3.5-9B and Qwen3.6-35B-A3B-FP8 (S-11, S-13)

Files: `tests/golden/qwen3.5-9b/{reference.jsonl,tolerance.json,README.md}`, `tests/golden/qwen3.6-35b-a3b-fp8/{reference.jsonl,tolerance.json,README.md}`, `tests/eval/qwen3.5-9b/{vllm.json,turbine.json,gate.json}`, `tests/eval/qwen3.6-35b-a3b-fp8/{vllm-tp2.json,vllm-ep2.json,turbine-tp2.json,turbine-ep2.json,gate.json}`, `scripts/lab/phase7-novanas-qwen36.yaml`, `crates/turbine-core/src/support.rs`, `.procoder/perf-log.md` ("P7 hybrid gates")
Interfaces:

- Golden and GSM8K with thinking off (`chat_template_kwargs: {"enable_thinking": false}` in the reference run and in `turbine-golden`'s requests through its `--chat-template-kwargs '{"enable_thinking":false}'` flag, added here to `benches/turbine-bench/src/golden.rs` with a `bench` test `golden_passes_chat_template_kwargs`), plus one logged GSM8K run with thinking on (decision reasoning A)
- Qwen3.6 reference on the dequantized BF16 copy (~72 GB): novanas `fixture.lock` if `MemAvailable` > 90 GB at start, else cpuhost; READMEs carry revisions `c202236235762e1c871ad0ccb60c8ee5ba337b9a` / `95a723d08a9490559dae23d0cff1d9466213d989`; `gate.json` 0.01 (Qwen3.5 BF16) / 0.02 (Qwen3.6 FP8)
  Covers: S-11 AC (lab gates), S-13
  Depends on: Tasks 9, 33, 34

- [ ] Write failing test `bench golden_passes_chat_template_kwargs` (a mock endpoint records the request body; the flag's object appears verbatim). Run: `scripts/remote-cargo.sh test -p turbine-bench --test golden golden_passes_chat_template_kwargs` — expect FAIL; implement — expect PASS.
- [ ] Fixtures (command lines `p7-ref-qwen3.5`, `p7-ref-qwen36`): `hf_reference.py` with thinking off for both (Qwen3.6 via the dequantized copy, deleted afterwards).
- [ ] Lab: `scripts/lab-bench.sh --model qwen3.5-9b --golden16` — expect golden1=PASS golden16=PASS; GSM8K pair against `scripts/lab-serve.sh novanas --vllm qwen3.5-9b` (newer tag once, else the no-vLLM rule) and `scripts/lab-perf.sh novanas qwen3.5-9b --config scripts/lab/phase7-novanas-qwen3.5-9b.yaml` — expect eval-compare exit 0 and ratio ≥ 0.75 or `vllm=unavailable`.
- [ ] vLLM baseline and ratio (decision 2026-10-08 B): `scripts/lab-perf.sh novanas qwen36 --placement tp2 --eval` and `--placement ep2 --eval` — expect `ratio=` ≥ 0.75 `verdict=PASS` (newer tag once, else `verdict=NO_VLLM` under the no-vLLM rule); copy each `vllm-gsm8k.json` to `tests/eval/qwen3.6-35b-a3b-fp8/vllm-<placement>.json`
- [ ] Lab: `scripts/lab-cluster.sh --bench-lock qwen36-tp2-novanas` and `--bench-lock qwen36-ep2-novanas` — expect `PASS … gsm8k=PASS` lines; copy each `turbine-<placement>.json` into `tests/eval/qwen3.6-35b-a3b-fp8/`; copy the faster passing config to `scripts/lab/phase7-novanas-qwen36.yaml`.
- [ ] Record per row/placement in perf-log (`P7 closed` / `P7 open`), flip or refuse (`qwen36_<placement>_gate`); `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(qwen3.5): hybrid golden, GSM8K and throughput gates (Qwen3.5-9B, Qwen3.6-35B-A3B FP8)`

## Task 36: OLMoE FP8 KV — calibrated V scales (S-15)

Files: `crates/turbine-model/src/kv_scales.rs` (calibration for scale-less checkpoints: per-layer K and V absmax over a fixed calibration prompt set at load, scale = absmax / 448, cached beside the checkpoint's id in the server's cache dir), `crates/turbine-model/tests/fixtures/kv_calibration_prompts.jsonl` (new: 32 prompts, none from `tests/golden/prompts.jsonl` or GSM8K), `crates/turbine-model/tests/tiny_model.rs`, `crates/turbine-core/src/support.rs` (OLMoE `fp8_e4m3` row), `.procoder/perf-log.md` ("P7 OLMoE FP8 KV")
Interfaces:

- `kv_scales::calibrate(executor: &mut dyn ModelExecutor, prompts: &[Vec<u32>]) -> KvScales` (deterministic: same checkpoint and prompts → same scales); `KvScales { k: Vec<f32>, v: Vec<f32> }` per layer; used only when the checkpoint has no K/V scales; logged `event="kv_scales_calibrated" layers=<n>`
  Covers: S-15 AC
  Depends on: Task 6

- [ ] Write failing test `tiny_model kv_scales_calibrated_for_scaleless_checkpoint` (tiny OLMoE without scales: calibration returns per-layer scales equal to a hand computation over the CPU executor's K/V, deterministic over two runs; FP8 KV decode error vs BF16 KV lower than with the unit-scale default). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model kv_scales_calibrated` — expect FAIL; implement — expect PASS.
- [ ] Lab: `scripts/lab-bench.sh --model olmoe --golden16 -- --set kv.dtype=fp8_e4m3` — expect golden1/golden16 against `tests/golden/olmoe-1b-7b-0125-instruct-fp8kv/` (the existing emulated-KV reference) PASS; on PASS add `- P7 closed: OlmoeForCausalLM/bf16/fp8_e4m3/tp1` and flip the row; on a miss list the failing prompts with per-position |Δ| in perf-log and keep `experimental`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` and `scripts/lab-test.sh novanas --tier quick` — expect `gate: ok` and exit 0
- [ ] Commit: `feat(kv): calibrated FP8 KV scales for scale-less checkpoints (OLMoE)`

## Task 37: L0 `tq4` re-gate on the Phase 7 tree (S-16)

Files: `.procoder/perf-log.md` ("P7 L0 tq4 re-gate"), `crates/turbine-core/src/support.rs` (the two L0 `tq4` rows only if every gate passes)
Interfaces:

- The Phase 6b Task 13 gates, unchanged: golden c1/c16 with `--batched-bounds` (`scripts/lab-bench.sh --golden16 --batched-bounds -- --set kv.dtype=tq4`), the shared-prefix GSM8K pair (`tests/eval/gsm8k-200-shared-prefix.jsonl`, 3 + 3 fresh-server runs, `scripts/eval/paired_compare.py --max-drop 0.01`, recipe `.procoder/handoff/p6b-eval-prefix.md`) and the multi-turn cached-ratio criterion (`--multi-turn`)
  Covers: S-16 AC
  Depends on: Task 6 (runs after the families so the tree is the Phase 7 tree; may run at any point after Task 35)

- [ ] Lab: `scripts/lab-bench.sh --model llama --golden16 --batched-bounds --multi-turn -- --set kv.dtype=tq4` and the same for `--model olmoe` — record each BENCH line.
- [ ] Lab: the shared-prefix GSM8K pair per the recipe for both models — record the paired drops.
- [ ] Write perf-log "P7 L0 tq4 re-gate" with every result; add `- P7 closed: <Llama|Olmoe>ForCausalLM/bf16/tq4/tp1` only for a model whose gates all passed and flip its row; else `P7 open` with the failing gate.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(kv): L0 tq4 re-gate on the Phase 7 tree`

## Task 38: Docs, contract and AGENTS.md (S-20)

Files: `docs/extending/model-family.md` (layer kinds, `StateLayout`, `default_reasoning_parser`, the hybrid snapshot rule, the tiny-trace fixture tool), `docs/extending/reasoning-parser.md` (new page: files, registry entry, suite and command, pitfalls), `docs/extending/README.md` (index row), `docs/extending/tool-format.md` (Harmony, `qwen3_xml` if added), `.procoder/contract/interfaces.md` (new §27 "Phase 7 additions"), `AGENTS.md` (support-matrix line, Commands: Phase 7 lab configs, `--model` slugs, cluster scenarios, `model.reasoning_parser`), `crates/turbine-model/tests/docs_extending.rs` (the reasoning-parser page added to its checked pages)
Interfaces:

- §27 lists: ABI v2.16 (`turbine_moe_experts_desc` fields, `TURBINE_EXPERT_*`, `TURBINE_ROUTE_TOPK_THEN_SOFTMAX`), v2.17 (`window_left`, `sinks`, d64/d256), v2.18 (`turbine_gdn_*`, `turbine_conv1d_*`, `boundary_states`); config keys `model.reasoning_parser`; API `reasoning_effort`, `reasoning_content`; metrics `turbine_hybrid_state_bytes`, `turbine_hybrid_state_snapshots_total{outcome}`, `turbine_reasoning_tokens_total`; KV document `ranks[].headroom_bytes`, `tiers[].state_bytes`; reason codes `sliding_window_unsupported`, `kernel_op_unavailable`, `jump_forward_hybrid`, the narrowed `quant_moe_phase7`, any `<family>_<placement>_gate` refusals; registries `reasoning_parser`; types `LayerKind`, `StateLayout`, `StateDtype`, `StateSlotPool`, `StateCopy`, `ExpertDtype`, `ExpertActivation`, `ReasoningParser`, `ReasoningDelta`; events `weights_skipped`, `kv_scales_calibrated`, `jump_forward_disabled`
  Covers: S-20 AC
  Depends on: Tasks 1–37

- [ ] Write failing check: extend `docs_extending` with the reasoning-parser page (the files and command it names must exist). Run: `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — expect FAIL with "docs/extending/reasoning-parser.md"
- [ ] Write the pages, §27 and the AGENTS.md lines — run, expect PASS.
- [ ] Verify the contract lists each name: `for n in turbine_gdn_prefill turbine_conv1d_update reasoning_parser turbine_hybrid_state_bytes sliding_window_unsupported kernel_op_unavailable jump_forward_hybrid headroom_bytes; do grep -q "$n" .procoder/contract/interfaces.md || echo "missing $n"; done` — expect no output.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs: Phase 7 extending pages, contract §27 and AGENTS.md`

## Task 39: Phase exit (S-1, S-13)

Files: `.procoder/perf-log.md` ("P7 exit"), `tests/bench/phase-7-model-families/` (track-close bench reports), `.procoder/specs/phase-7-model-families.md` (acceptance boxes ticked with evidence)
Interfaces:

- The 5p exit lines this phase may not regress by more than 3 %: Llama 864.0 tok/s, OLMoE 636.1 tok/s (BF16, GPU 0)
- The umbrella `phase-6-8-expansion` Task 10 runbook, run once for `phase-7-model-families`
  Covers: the spec's phase-exit AC; S-13 (every row's status matches its gate)
  Depends on: Tasks 1–38

- [ ] `scripts/gate.sh --full` — expect `gate: ok`
- [ ] `scripts/lab-test.sh novanas --tier full` — expect exit 0
- [ ] `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` — expect exit 0
- [ ] `scripts/lab-bench.sh --golden16 --model llama` and `--model olmoe` — expect golden1/golden16 PASS and tok/s ≥ 838.1 (Llama) and ≥ 617.0 (OLMoE)
- [ ] ASK THE USER FIRST, then `scripts/overload-soak.sh novanas --duration 10m` — expect exit 0 and a passing `verdict.json`
- [ ] Run the umbrella Task 10 runbook for `phase-7-model-families` (serve each closed row's lab config, golden, eval, bench reports into `tests/bench/phase-7-model-families/`, soak per the runbook, no regression of the Phase 2 Llama/OLMoE golden).
- [ ] `scripts/remote-cargo.sh test -p turbine-core support::tests::phase7_rows` and `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix --output json | jq -c '.rows[] | select(.status == "supported")'` — expect PASS and exactly the baseline rows plus the `P7 closed` rows.
- [ ] Closed Phase 7 checkpoints stay on local disk under `/home/piwi/turbine-models/<slug>` (user decision 2026-10-08 2 A); record the per-slug disk use (`du -sh`) in perf-log "P7 exit".
- [ ] Commit: `test(phase-7-model-families): exit-gate reports`
