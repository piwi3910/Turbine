# phase-8-speculative-decoding — implementation plan

Status: complete
Spec: .procoder/specs/phase-8-speculative-decoding.md

## Goal

Serve a BF16 Llama-3.2-1B draft beside Llama-3.2-3B on one R9700: the draft proposes k greedy tokens, one target forward verifies them with exact greedy acceptance or exact rejection sampling, KV and hybrid-state rollback keep the invariants, and the Llama draft row turns `supported` after the golden, A/B, throughput, TTFT, vLLM and soak gates.

## Architecture

A `Proposer` registry in `turbine-model::speculative` owns the draft executor, its private one-sequence KV slot and the greedy proposal; `turbine_scheduler::speculative` owns the pure acceptance math over per-row target distributions. The engine reuses the Phase 5p multi-token append (`Scheduler::extend` + `BatchItem.draft` + `IterationOutcome::rolled_back`) to feed the sampled token plus k proposals as one target step, emits only the accepted tokens through the ordinary host pipeline, and rolls the KV table and a hybrid scratch state back to the kept length. Startup prepares the draft as a second model, validates the family/tokenizer/format/context and charges its weights, workspace and slot to the Phase 3 budget; every outside condition (batch, constraint, seed, pressure, slot, rows) falls back to a plain decode with a reason code.

## Constraints

Copied verbatim from the spec (Constraints):

- Rust only in the serving path, no Python runtime. No new crate; the proposer registry lives in `turbine-model` and the verifier in `turbine-scheduler`; `unsafe` stays in the existing allowlisted locations and no vendor type enters a core signature.
- Bounded inputs: k ≤ 8; one draft KV slot; the verification span is capped by the plan's token and logits-row budgets; no unbounded queue, cache or retry. A span the plan cannot cover is dropped, never forced.
- Exactness: with `method: draft` the greedy token stream is the target's greedy stream for the accepted prefix, sampling is distribution-preserving by rejection, and seeded requests bypass so their output stays bit-identical. `method: none` changes nothing.
- Memory: the draft's weights, workspace and slot are reserved through the Phase 3 budget before speculation is enabled; a combination that does not fit exits 1 before `/ready`; per-request draft KV failure falls back, never fails the request.
- Latency and throughput: the S-11 bar (≥ 1.3× single-stream decode, TTFT ≤ 1.15×, c16 ≥ 0.97× the Phase 7 exit) is the gate; a miss leaves the row `experimental` and is recorded, not papered over.
- The draft checkpoint is provisioned once under `/home/piwi/turbine-models/<slug>` with the user's approval, pinned by revision, never committed and never downloaded by a test or by the server.
- Lab rules: `novanas` only, the bench lock and fixture lock as in AGENTS.md, the user asked before any lab run outside the standing approvals; no workload on the host is moved, stopped or evicted.

From AGENTS.md and the contract (binding for every task):

- Precondition: `scripts/track-gate.sh phase-8-speculative-decoding` printed `GATE PASS phase-8-speculative-decoding` before Task 1 (it did on 2026-10-25, `.procoder/ask/decisions.md`).
- Builds and tests run on novanas through `scripts/remote-cargo.sh`; no local target directories. Every task ends with `scripts/gate.sh` printing `gate: ok`. Tasks touching `turbine-model`, `turbine-kernels`, `turbine-device`, the kernel ABI or `kernels/rocm` add `scripts/lab-test.sh novanas --tier quick`; tasks that can change Llama or OLMoE throughput or numerics add `scripts/lab-bench.sh --quick --model llama` (GPU 0) and a `.procoder/perf-log.md` row.
- Lab runs start with `run_in_background` (never a foreground wait on a lock or a long remote run); throughput on GPU 0 under `scripts/bench-lock.sh`; the user is asked first for every soak and anything that needs workloads moved; if `amd.com/gpu` is held by another workload, stop and ask; never evict.
- The draft checkpoint download is approved (decision 2026-10-25) and runs on novanas with the user's cached `hf auth login`; the token is never read, printed or passed.
- Toolchain edition 2024, `rust-version = "1.97"`; config structs `#[serde(deny_unknown_fields, default)]`; metric labels from closed enums rendered with `as_str()`; `#[non_exhaustive]` on enums later phases extend.
- Names in `.procoder/contract/interfaces.md` are used verbatim; this phase's additions go to a new §28 "Phase 8 additions" (Task 13).
- One change, then measure; a golden reference is never widened to make a gate pass; a row that misses stays `experimental` with its evidence in `.procoder/perf-log.md`.
- Every task commits one change with a Conventional Commit message and no attribution footer.

## Task 1: The four speculative config keys and validation (S-1, S-8)

Files: `crates/turbine-core/src/config/speculative.rs` (extend), `crates/turbine-core/src/config/mod.rs` (validation hook), `examples/turbine.yaml` (documented commented keys)

Interfaces:

- `pub const MAX_SPECULATIVE_TOKENS: u32 = 8;`
- `#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)] #[serde(deny_unknown_fields, default)] pub struct SpeculativeConfig { pub method: SpeculativeMethod, pub num_tokens: u32, pub draft_model_path: Option<PathBuf>, pub min_acceptance: f32 }` with `Default` = `none`, 4, `None`, 0.4
- `impl SpeculativeConfig { pub fn validate(&self, effective_max_running: u32) -> Result<(), ConfigError> }`: `num_tokens` in 1..=MAX_SPECULATIVE_TOKENS, `min_acceptance` in 0..=1, `draft` requires `draft_model_path`, and `num_tokens + 1 <= effective_max_running` (the message starts `speculative_rows:`); `none` leaves the other keys inert (bounds still validated, the path is not read)
- `Config::validate` calls `self.speculative.validate(self.effective_max_running())` after `validate_phase2`, and, when `method == Draft`, refuses an explicit `parallel.tensor_parallel_size` or `parallel.pipeline_parallel_size` above 1 under key `speculative.method` with a message starting `speculative_parallel_unsupported:`

Covers: S-1, S-8; AC 1
Depends on: nothing in this phase

- [ ] Write failing test `config::speculative::tests::speculative_config_defaults_and_bounds` in `crates/turbine-core/src/config/speculative.rs`:

  ```rust
  #[test]
  fn speculative_config_defaults_and_bounds() {
      let cfg: SpeculativeConfig = serde_norway::from_str("{}").unwrap();
      assert_eq!(cfg.method, SpeculativeMethod::None);
      assert_eq!(cfg.num_tokens, 4);
      assert_eq!(cfg.draft_model_path, None);
      assert_eq!(cfg.min_acceptance, 0.4);
      cfg.validate(64).unwrap();
      for (yaml, key) in [
          ("num_tokens: 0", "speculative.num_tokens"),
          ("num_tokens: 9", "speculative.num_tokens"),
          ("min_acceptance: -0.1", "speculative.min_acceptance"),
          ("min_acceptance: 1.1", "speculative.min_acceptance"),
          ("method: draft", "speculative.draft_model_path"),
      ] {
          let cfg: SpeculativeConfig = serde_norway::from_str(yaml).unwrap();
          assert_eq!(cfg.validate(64).unwrap_err().key(), Some(key), "{yaml}");
      }
      // The row capacity is a draft rule; `none` leaves num_tokens inert even at one row.
      let cfg = SpeculativeConfig { num_tokens: 8, ..Default::default() };
      cfg.validate(1).unwrap();
      let cfg = SpeculativeConfig {
          method: SpeculativeMethod::Draft,
          num_tokens: 8,
          draft_model_path: Some("/nonexistent".into()),
          ..Default::default()
      };
      let err = cfg.validate(8).unwrap_err();
      assert!(err.to_string().contains("speculative_rows:"), "{err}");
      cfg.validate(16).unwrap(); // the path is read (and refused) later, by the draft startup
  }
  ```

- [ ] Write failing test `config::tests::speculative_parallel_refused` in `crates/turbine-core/src/config/tests.rs`: parse `TS15_EXAMPLE` with `--set speculative.method=draft --set speculative.draft_model_path=/x --set parallel.tensor_parallel_size=2` (and the same with `parallel.pipeline_parallel_size=2`) → an error whose key is `speculative.method` and whose message contains `speculative_parallel_unsupported:`; `tensor_parallel_size=1` and `=auto` pass.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core config::speculative::tests::speculative_config_defaults_and_bounds` and `... config::tests::speculative_parallel_refused` — expect FAIL with "unexpected end of input" / "missing field `num_tokens`" / "no method named `validate`".
- [ ] Implement the struct, bounds, the rows check and the parallel refusal; add the four keys to `examples/turbine.yaml` at their defaults.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core config::speculative::tests::` and `... config::tests::speculative_parallel_refused` — expect PASS.
- [ ] Run: `scripts/remote-cargo.sh run -p turbine-server -- --config examples/turbine.yaml --check-config` — expect `config ok` (the `method: none` defaults do not change the resolved row).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(config): speculative method, num_tokens, draft_model_path and min_acceptance`

## Task 2: Support-matrix rows, the speculative column and `phase8_rows` (S-9)

Files: `crates/turbine-core/src/support.rs` (rows and tests), `crates/turbine-server/src/support_startup.rs` (key carries the column), `crates/turbine-server/src/support_matrix.rs` (output unchanged), `crates/turbine-core/src/support.rs` tests

Interfaces:

- `pub fn SupportKey::before_discovery_with(vendor: &str, arch: Option<&str>, architecture: Option<&str>, weight: WeightFormatColumn, kv: KvFormatColumn, speculative: SpeculativeColumn) -> SupportKey`; `SupportKey::before_discovery(...)` stays as `before_discovery_with(..., SpeculativeColumn::None)` for existing callers
- `crates/turbine-server/src/support_startup.rs`: `pub fn speculative_column(cfg: &Config) -> SpeculativeColumn` (`Draft` for `method: draft`, `None` otherwise); `key_before_discovery` passes it to `before_discovery_with`
- `SUPPORT_MATRIX` gains `row(Some("cpu"), None, None, BF16, KV_BF16, Some(SpeculativeColumn::Draft), SupportStatus::Experimental)` and, for the gate run, `row(Some("amd"), Some("gfx1201"), Some("LlamaForCausalLM"), BF16, KV_BF16, Some(SpeculativeColumn::Draft), SupportStatus::Experimental)` with a `P8_PENDING_REASON` constant (`"gate pending (track phase-8-speculative-decoding)"`). The wildcard draft row's reason becomes `"draft-model speculative decoding is supported only for the validated Llama-3.2-1B -> Llama-3.2-3B pairing (track phase-8-speculative-decoding)"` and covers every other combination. Task 14 flips the amd row to `Supported` and removes the gate-pending reason from it.
- `support::tests::phase8_rows`: the cpu draft key resolves `experimental`; the amd Llama draft key resolves `experimental` with the gate-pending reason before Task 14; every other architecture × draft key resolves `unsupported` with the wildcard reason
- `support_startup::tests::key_before_discovery_carries_draft`: a `Config` with `method: draft` yields `<vendor>/*/<arch>/bf16/bf16/draft`

Covers: S-9; AC 12 (the pre-gate half; Task 14 completes it)
Depends on: Task 1

- [ ] Write failing test `support::tests::phase8_rows` in `crates/turbine-core/src/support.rs`:

  ```rust
  #[test]
  fn phase8_rows() {
      use SpeculativeColumn as S;
      let cpu = SupportKey::for_model("cpu", "*", "LlamaForCausalLM", W::Bf16, K::Bf16, S::Draft);
      assert_eq!(resolve_partial_in(SUPPORT_MATRIX, &cpu).as_str(), "experimental");
      let amd = key("amd", "gfx1201", "LlamaForCausalLM", W::Bf16, K::Bf16, S::Draft);
      let d = check(amd.clone()).expect("an experimental row is not refused");
      assert_eq!(d.status.as_str(), "experimental", "{amd}");
      assert_eq!(d.view().reason.as_deref(), Some(P8_PENDING_REASON), "{amd}");
      for architecture in ["OlmoeForCausalLM", "Qwen3ForCausalLM", "MistralForCausalLM"] {
          let k = key("amd", "gfx1201", architecture, W::Bf16, K::Bf16, S::Draft);
          let status = resolve(&k);
          assert_eq!(status.as_str(), "unsupported", "{k}");
          assert!(status.reason().unwrap().contains("validated Llama"), "{k}: {status:?}");
      }
  }
  ```

- [ ] Write failing test `support_startup::tests::key_before_discovery_carries_draft`: `Config { execution: ExecutionConfig { backend: BackendChoice::Cpu, ..Default::default() }, speculative: SpeculativeConfig { method: Draft, draft_model_path: Some("/x".into()), ..Default::default() }, ..Default::default() }` → `key_before_discovery(...).to_string()` ends `/draft`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::phase8_rows` — expect FAIL "no variant `draft`"/"unknown row"; `scripts/remote-cargo.sh test -p turbine-server support_startup::tests::key_before_discovery_carries_draft` — expect FAIL "left .../none".
- [ ] Implement `before_discovery_with`, the cpu and amd draft rows, the wildcard reason and the column wiring.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core support::tests::` and `scripts/remote-cargo.sh test -p turbine-server support_startup::tests::` — expect PASS; `scripts/remote-cargo.sh run -q -p turbine-server -- --support-matrix --output text` prints the cpu and amd draft rows and the wildcard reason.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(support): draft speculative row on cpu and the speculative column in the matrix key`

## Task 3: Prepare the draft model, validate the pair and charge the budget (S-1, S-7, S-8, S-12)

Files: `crates/turbine-server/src/model.rs` (draft preparation and tokenizer fingerprint), `crates/turbine-server/src/startup.rs` (prepare both models; resolved tp/pp refusal), `crates/turbine-reliability/src/budget.rs` (`DraftBudget` and the combined budget), `crates/turbine-server/src/engine/mod.rs` (hold `Option<DraftModel>`), `crates/turbine-server/tests/server_cli.rs` (refusals), `crates/turbine-reliability/src/budget.rs` tests

Interfaces:

- `crates/turbine-reliability/src/budget.rs`:
  - `pub struct DraftBudget { pub weights_bytes: u64, pub workspace_bytes: u64, pub slot_bytes: u64 }`
  - `pub fn compute_budget_with_draft(inp: &BudgetInputs, draft: &DraftBudget, cfg: &ReliabilityConfig, kv_cap: Option<ByteSize>) -> Result<DeviceBudget, BudgetError>`: adds `draft.weights_bytes + draft.workspace_bytes + draft.slot_bytes` to the fixed cost and runs the same `kv >= min_kv` check; the error breakdown appends `draft_weights=…; draft_workspace=…; draft_slot=…`
  - `pub fn draft_slot_bytes(kv_bytes_per_token: u64, max_seq_len: u32, block_bytes: u64) -> u64` (one sequence, rounded to blocks)
- `crates/turbine-server/src/model.rs`:
  - `#[derive(Clone, Debug, PartialEq, Eq)] pub struct TokenizerFingerprint { pub vocab_size: usize, pub bos: Option<u32>, pub eos: Option<u32>, pub pad: Option<u32>, pub unk: Option<u32>, pub hash: u64 }`
  - `pub fn tokenizer_fingerprint(tokenizer: &Tokenizer) -> TokenizerFingerprint` (DefaultHasher over sorted `(id, token)` pairs plus the special ids)
  - `pub fn draft_budget(cfg: &Config) -> Result<Option<DraftBudget>, ConfigError>`: `None` for `method: none`; otherwise read the target config/tokenizer and the draft `config.json`/tokenizer, run the S-1 checks in this order — path exists (`speculative_draft_path`), architecture equals the target's (`speculative_draft_family`), tokenizer fingerprint equals the target's (`speculative_draft_tokenizer`), weight format BF16 (`speculative_draft_format`), draft `max_position_embeddings >= cfg.model.max_seq_len` (`speculative_draft_context`) — and return `DraftBudget { weights_bytes: draft index total, workspace_bytes: reliability.memory.workspace_bytes.0, slot_bytes: draft_slot_bytes(draft.kv_layout(block_tokens).bytes_per_token(), max_seq_len, draft block bytes) }`
  - `pub fn prepare(cfg, inventory, metrics, draft: Option<&DraftBudget>) -> Result<PreparedModel, StartupError>` (the existing target preparation; its budget call is `compute_budget_with_draft` when `draft` is `Some`, so the target's `budget` and `pool.num_blocks` are already net of the draft; the call sites pass `None` elsewhere)
  - `pub struct DraftModel { pub executor: Box<dyn ModelExecutor>, pub pool: BlockPool, pub tokenizer: Arc<Tokenizer>, pub arch: ModelArchConfig, pub max_seq_len: u32, pub slot_blocks: u32 }`
  - `pub fn load_draft(cfg: &Config, budget: &DraftBudget, inventory: &DeviceInventory, metrics: &Metrics) -> Result<Option<DraftModel>, StartupError>`: loads the draft weights/executor (the existing loader path with `model.path = draft_model_path`) and allocates the draft `BlockPool` of `ceil(max_seq_len / block_tokens) + ceil(num_tokens / block_tokens) + 1` blocks; a load failure is an exit-1 `StartupError` whose message names `speculative.draft_model_path` and the draft dir
- `crates/turbine-server/src/startup.rs`: the order is `draft_budget` (exit 2 on a pair/config error) → target `prepare` with the budget → `load_draft` (exit 1 on a load/budget error) → the engine; after device discovery, a resolved tp > 1 or pp > 1 with `method: draft` exits 2 with `speculative_parallel_unsupported:`

Covers: S-1, S-7, S-8, S-12; AC 2, 10
Depends on: Tasks 1, 2

- [ ] Write failing test `budget::tests::speculative_draft_footprint`:

  ```rust
  #[test]
  fn speculative_draft_footprint() {
      let base = inputs(MemoryKind::Dedicated, Some(40 * GIB), None);
      let cfg = ReliabilityConfig::default();
      let without = compute_budget(&base, &cfg, None).unwrap();
      let slot = draft_slot_bytes(32_768, 32_768, 4 * MIB);
      assert_eq!(slot, GIB, "32768 tokens × 32 KiB, in 4 MiB blocks");
      assert_eq!(draft_slot_bytes(32_768, 1, 4 * MIB), 4 * MIB, "one token rounds to a block");
      let draft = DraftBudget { weights_bytes: 2 * GIB, workspace_bytes: 256 * MIB, slot_bytes: slot };
      let with = compute_budget_with_draft(&base, &draft, &cfg, None).unwrap();
      assert_eq!(with.pool(PoolKind::Kv), without.pool(PoolKind::Kv) - 3 * GIB - 256 * MIB);
      assert_eq!(with.pool(PoolKind::Weights), base.weights_bytes);
      let tight = BudgetInputs { measured_free_bytes: Some(8 * GIB), ..base.clone() };
      let err = compute_budget_with_draft(&tight, &draft, &cfg, None).unwrap_err();
      assert!(err.breakdown.contains("draft_weights="), "{err}");
      assert!(err.breakdown.contains("draft_slot="), "{err}");
  }
  ```

- [ ] Write failing test `crates/turbine-server/tests/server_cli.rs::speculative_startup_refusals`: build tiny target/draft directories (`write_tiny_llama_with`) and run `--check-config` with `--set execution.backend=cpu --set speculative.method=draft --set speculative.draft_model_path=<dir>`; cases: a missing dir (`speculative_draft_path`, exit 2), a tiny Qwen3 draft (`speculative_draft_family`), a tokenizer with one special id changed (`speculative_draft_tokenizer`), a tiny quantized Llama draft (`speculative_draft_format`), a draft whose `max_position_embeddings` is 8 with `--set model.max_seq_len=64` (`speculative_draft_context`), `--set speculative.num_tokens=8 --set scheduler.max_running_requests=8` (`speculative_rows`), and an explicit `--set parallel.tensor_parallel_size=2` (`speculative_parallel_unsupported`); the valid tiny pair prints `config ok` and the support line `cpu/*/LlamaForCausalLM/bf16/bf16/draft`; a valid dir with a corrupt `model.safetensors` (the last 8 bytes overwritten) exits 1 before `/ready` with the draft path in the message.
- [ ] Write failing test `model::tests::draft_tokenizer_fingerprint_detects_a_change`: two tiny tokenizer dirs; one with a modified `special_tokens_map.json`/added vocab entry → fingerprints differ; the same dir → equal.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-reliability budget::tests::speculative_draft_footprint` and `scripts/remote-cargo.sh test -p turbine-server --test server_cli speculative_startup_refusals` — expect FAIL "no function `compute_budget_with_draft`"/"unknown key `speculative.draft_model_path`".
- [ ] Implement `DraftBudget`, the combined budget, the fingerprint, `draft_budget`, the `prepare` parameter and `load_draft`; the budget error wrapper `fn budget_error_with_draft(error: BudgetError, path: &Path) -> StartupError` (message starts `speculative_budget:`, names `speculative.draft_model_path` and `path`).
- [ ] Run the three tests — expect PASS; `scripts/remote-cargo.sh test -p turbine-server --test server_cli` — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(server): prepare and validate the speculative draft model and charge its budget`

## Task 4: The `Proposer` registry, the draft KV slot and the conformance suite (S-2, S-7)

Files: `crates/turbine-model/src/speculative/mod.rs` (new: trait, registry, errors), `crates/turbine-model/src/speculative/draft.rs` (new: `draft_model`), `crates/turbine-model/src/lib.rs` (`pub mod speculative`), `crates/turbine-model/tests/conformance.rs`, `crates/turbine-model/tests/docs_extending.rs`, `docs/extending/speculative-proposer.md` (new), `docs/extending/README.md` (index)

Interfaces:

- `pub trait Proposer: Send { fn name(&self) -> &'static str; fn validate(&self, target: &ModelArchConfig, draft: &ModelArchConfig, tokenizer: &Tokenizer) -> Result<(), ProposeError>; fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<(), ProposeError>; fn propose(&mut self, request: ProposeRequest<'_>) -> Result<Proposals, ProposeError>; fn truncate(&mut self, seq: SeqId, len: u32) -> Result<(), ProposeError>; fn release(&mut self, seq: SeqId); }`
- `pub struct ProposeRequest<'a> { pub seq: SeqId, pub last_token: u32, pub accepted_len: u32, pub num_tokens: u32 }`
- `pub struct Proposals { pub tokens: Vec<u32> }`
- `pub enum ProposeError { SlotBusy, NotBegun, Kv(turbine_kv::KvError), Model(turbine_model::ModelError), Unsupported(&'static str) }` with `pub fn reason(&self) -> &'static str` (`draft_slot_busy`, `draft_not_begun`, `draft_kv`, `draft_model`, `draft_unsupported`)
- `pub struct DraftProposerConfig { pub executor: Box<dyn ModelExecutor>, pub pool: BlockPool, pub tokenizer: Arc<Tokenizer>, pub block_tokens: u32, pub max_seq_len: u32, pub target: ModelArchConfig, pub draft: ModelArchConfig }`
- `pub trait ProposerFactory: Send + Sync { fn name(&self) -> &'static str; fn build(&self, cfg: DraftProposerConfig) -> Result<Box<dyn Proposer>, ProposeError>; }`
- `pub static PROPOSERS: &[&dyn ProposerFactory] = &[&DraftModelFactory]; pub fn proposer(name: &str) -> Option<&'static dyn ProposerFactory>; pub fn conformance(factory: &dyn ProposerFactory, cfg: DraftProposerConfig) -> Result<(), String>;`
- `draft_model` semantics: `begin` takes the single full-context slot (`SlotBusy` for a second sequence) and prefills `prompt` in chunks of `scheduler.max_batch_tokens`; `propose` truncates the draft table to `accepted_len`, feeds `last_token`, runs `num_tokens` greedy steps and returns their argmax ids; `truncate` keeps `len`; `release` frees the slot and clears the table; every call returns `NotBegun` for an unknown sequence.

Covers: S-2; AC 3
Depends on: Tasks 1, 3

- [ ] Write failing test `speculative::tests::draft_model_conformance` in `crates/turbine-model/src/speculative/mod.rs`: build a tiny draft executor and pool (the existing `testing::tiny::write_tiny_llama_with` plus `DecoderExecutor` and `BlockPool`), `conformance(&DraftModelFactory, cfg)` returns `Ok(())`, and `conformance` on a factory without a `draft_model` name returns an error naming the entry.
- [ ] Write failing test `crates/turbine-model/tests/conformance.rs::proposer_registry`: iterate `PROPOSERS`, assert `proposer(f.name()).is_some()`, the names are distinct, and `DraftModelFactory.name() == "draft_model"`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model speculative::tests::draft_model_conformance` — expect FAIL "cannot find module `speculative`".
- [ ] Implement `speculative/mod.rs` and `speculative/draft.rs`: the pool is `BlockPool::new(BlockPoolConfig { layout: draft.kv_layout(block_tokens), num_blocks: one slot + ceil(k/block_tokens) + 1, ..Default::default() }, Arc::clone(&mem))`; the slot is one `BlockTable` (the pool's `table` helper or `BlockTable::new`); the draft executor is called through `ModelExecutor::prefill`/`decode`/`forward` (use the existing executor API); the greedy token is the argmax of its logits row (host reduction for the tiny/cpu path).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model speculative::tests::` and `scripts/remote-cargo.sh test -p turbine-model --test conformance` — expect PASS.
- [ ] Write `docs/extending/speculative-proposer.md`: the trait, the registry entry, the files, the conformance command, `verify_*` hand-off, the one-slot rule, the lab check and the pitfalls; add it to `docs/extending/README.md`; run `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — expect FAIL until the page is listed and current, then PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Lab quick (touching `turbine-model`): `scripts/lab-test.sh novanas --tier quick` — expect PASS.
- [ ] Commit: `feat(model): Proposer trait, draft_model registry and the draft KV slot`

## Task 5: The sampler's row API for verification (S-3)

Files: `crates/turbine-model/src/sampling/sampler.rs` (refactor), `crates/turbine-model/src/sampling/mod.rs` (exports), its tests

Interfaces:

- `pub struct ProcessedRow { pub greedy: u32, pub probs: Vec<f32>, /* post-processing, sums to 1 */ pub logprob: Vec<f32>, /* raw log-softmax, the request's reported logprobs */ }`
- `impl Sampler { pub fn row(&mut self, logits: &mut [f32], mask: Option<&TokenMask>) -> ProcessedRow; pub fn draw_row(&mut self, row: &ProcessedRow, exclude: Option<u32>) -> u32; pub fn logprob_of(&self, row: &ProcessedRow, token: u32) -> f32; }`
- `row` applies the existing processor chain (bias → penalties → EOS/stop suppression → mask → temperature → top-k → top-p) without observing a token; `draw_row` draws one uniform from the sampler RNG over `probs` with `exclude` masked; `Sampler::sample` is reimplemented as `row` + `draw_row` + `observe` so the non-speculative stream stays bit-identical.

Covers: S-3 (the exactness substrate)
Depends on: nothing in this phase

- [ ] Write failing test `sampling::tests::row_matches_sample`: for 10 000 seeded samplers (`SamplingParams { temperature: 0.8, top_p: 0.9, top_k: 40, seed: Some(i), ..Default::default() }`), take a fixed logits vector with a mask, clone the sampler, call `sample` on one clone and `row`+`draw_row` on the other, assert the tokens equality; for `temperature: 0.0` assert `row.greedy == sample.token`; assert `probs` sums to 1 ± 1e-5, every `exclude`-drawn token differs from `exclude`, and the logprob of the drawn token equals `row.logprob[token]`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model sampling::tests::row_matches_sample` — expect FAIL "no method named `row`".
- [ ] Refactor `sample` into `row`/`draw_row`/`observe` without changing the arithmetic or the RNG calls; keep `sample_rows` unchanged.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model sampling::` — expect PASS (including the existing `sample_matches...` and `sample_rows_matches_one_by_one`).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Lab quick: `scripts/lab-test.sh novanas --tier quick` — expect PASS (the sampler is GPU-facing through the device logits path).
- [ ] Commit: `refactor(model): sampler row and draw_row for verification without observing`

## Task 6: The verifier in `turbine-scheduler` (S-3)

Files: `crates/turbine-scheduler/src/speculative.rs` (new), `crates/turbine-scheduler/src/lib.rs` (`pub mod speculative`), its tests

Interfaces:

- `pub trait Rows { fn greedy(&self, row: usize) -> u32; fn probabilities(&self, row: usize) -> &[f32]; }`
- `pub trait Draw { fn uniform(&mut self) -> f64; fn draw(&mut self, weights: &[f32], exclude: Option<u32>) -> u32; }`
- `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct Verdict { pub accepted: u32, pub next: u32 }`
- `pub fn verify_greedy(rows: &impl Rows, proposals: &[u32]) -> Result<Verdict, VerifyError>`: for j in 0..k, if `rows.greedy(j) == proposals[j]` accept and continue; otherwise `next = rows.greedy(j)` and stop; if all k accepted, `next = rows.greedy(k)`; `accepted = j`
- `pub fn verify_sampling(rows: &impl Rows, proposals: &[u32], draw: &mut impl Draw) -> Result<Verdict, VerifyError>`: for j, `p = rows.probabilities(j)`, `t = proposals[j]`, accept with probability `p[t]` (`draw.uniform() < p[t]`), else `next = draw.draw(p, Some(t))` and stop; if all accepted, `next = draw.draw(rows.probabilities(k), None)`; `accepted = j`
- `VerifyError::MissingRow(usize)` when `proposals.len() >= rows.len()`.

Covers: S-3; AC 4
Depends on: nothing in this phase

- [ ] Write failing test `speculative::tests::greedy_prefix` and `speculative::tests::sampling_is_exact` in `crates/turbine-scheduler/src/speculative.rs`:

  ```rust
  struct FixedRows(Vec<u32>, Vec<Vec<f32>>);
  impl Rows for FixedRows { /* greedy = self.0[row], probabilities = &self.1[row] */ }
  struct ChaCha(f64);
  impl Draw for ChaCha { fn uniform(&mut self) -> f64 { /* cycle 0.1, 0.9 */ } fn draw(&mut self, w: &[f32], exclude: Option<u32>) -> u32 { /* weighted, exclude masked */ } }

  #[test]
  fn greedy_prefix() {
      let rows = FixedRows(vec![7, 8, 9, 10], vec![vec![]; 4]);
      assert_eq!(verify_greedy(&rows, &[7, 8, 9]).unwrap(), Verdict { accepted: 3, next: 10 });
      assert_eq!(verify_greedy(&rows, &[7, 1, 9]).unwrap(), Verdict { accepted: 1, next: 8 });
      assert_eq!(verify_greedy(&rows, &[1]).unwrap(), Verdict { accepted: 0, next: 7 });
  }

  #[test]
  fn sampling_is_exact() {
      let p = |t: u64| vec![0.6, 0.3, 0.1];
      let mut tv = [0.0f64; 3];
      for i in 0..1_000_000 {
          let rows = FixedRows(vec![0; 2], vec![p(i), p(i)]);
          let mut d = ChaCha((i as f64) / 1_000_000.0);
          let v = verify_sampling(&rows, &[0], &mut d).unwrap();
          tv[v.next as usize] += 1.0 / 1_000_000.0;
      }
      for (got, want) in tv.iter().zip([0.6, 0.3, 0.1]) {
          assert!((got - want).abs() < 0.01, "{tv:?}");
      }
      // p(t) = 0 is always rejected and never drawn on the fallback.
      let rows = FixedRows(vec![0; 2], vec![vec![0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0]]);
      let mut d = ChaCha(0.5);
      assert_eq!(verify_sampling(&rows, &[0], &mut d).unwrap(), Verdict { accepted: 0, next: 1 });
  }
  ```

- [ ] Run: `scripts/remote-cargo.sh test -p turbine-scheduler speculative::tests::` — expect FAIL "cannot find module".
- [ ] Implement the traits and both functions; no scheduler/model state, no allocation beyond the caller's rows.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-scheduler speculative::tests::` — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(scheduler): speculative verifier with exact greedy and rejection acceptance`

## Task 7: The engine's greedy speculative round, serial turns and metrics (S-3, S-4, S-10)

Files: `crates/turbine-server/src/engine/loop.rs` (round, verification, rollback, metrics), `crates/turbine-server/src/engine/requests.rs` (`step_verified`), `crates/turbine-server/src/engine/mod.rs` (own the proposer), `crates/turbine-server/src/metrics.rs` (the speculative series), `crates/turbine-server/src/engine/loop.rs` tests

Interfaces:

- `EngineParts` gains `pub draft: Option<crate::model::DraftModel>`; the engine builds the `Proposer` via `turbine_model::speculative::proposer("draft_model")` and its `DraftProposerConfig` (`executor`, `pool`, `tokenizer`, `block_tokens`, `max_seq_len`, `target`/`draft` configs) and keeps `Option<Box<dyn Proposer>>` beside the executor
- `ActiveRequest::step_verified(&mut self, choice: usize, rows: Vec<ProcessedRow>, proposals: &[u32], mode: VerifyMode, max_seq_len: u32, metrics: Option<&ModelMetrics>) -> Result<VerifiedOutcome, ModelError>`: `VerifyMode::{Greedy, Sampling}`, `VerifiedOutcome { accepted: u32, emitted: u32, finished: Option<FinishReason>, steps: Vec<Step> }`; each accepted proposal (or the target's fallback token) goes through `step_sampled` with a `SampledToken` built from the row (`token`, `logprob_of`, `top_logprobs`); a finish inside the span drops the rest
- `ServerMetrics` (in `crates/turbine-server/src/metrics.rs`) gains `speculative_rounds: Family<SpecOutcomeLabels, Counter>`, `speculative_proposed_tokens: Counter`, `speculative_accepted_tokens: Counter`, `speculative_disabled: Family<SpecReasonLabels, Gauge>`, `speculative_accepted_per_round: Histogram` with buckets `[0,1,2,3,4,5,6,7,8]`; label enums `SpecOutcomeLabels` (`accepted_all|partial|none|disabled`), `SpecReasonLabels` (the S-6 set); the render names are `turbine_speculative_rounds_total`, `turbine_speculative_proposed_tokens_total`, `turbine_speculative_accepted_tokens_total`, `turbine_speculative_disabled`, `turbine_speculative_accepted_per_round`
- `LoopState` gains `speculative_drafts: HashMap<SeqId, Vec<u32>>` and `speculative_mode: HashMap<SeqId, VerifyMode>`; `Scheduler::extend(seq, proposals)` is called after a successful decode (single-item plan and S-6 eligibility), a second extension while pending is a no-op, and a sequence that cannot get the extension decodes normally
- The round is a serial turn: `TurnOutcome::SerialTurn { reason: SerialTurnReason::SpeculativeStep }` when the eligible sequence's extension is in flight (the existing `overlap_serial_turns` counter gains the value `speculative_step`).

Covers: S-3, S-4, S-10; AC 5, 11, 13 (the metrics part), 16
Depends on: Tasks 3, 4, 5, 6

- [ ] Extend the engine test harness with `run_submissions_with_draft(spec: TinySpec, draft: TinySpec, tokenizer, requests, grammar, draft_on: bool, overlap: bool)` (a `Config` with `speculative.method: draft` and the draft dir; the target and draft tiny models from two seeds) in `crates/turbine-server/src/engine/loop.rs`.
- [ ] Write failing test `engine::r#loop::tests::speculative_greedy_matches_serial`: a tiny target + tiny draft, a request with 64 max tokens and the greedy sampler; for k in 1..=8, with the draft off and on and overlap false and true, the token streams are equal; `turbine_speculative_rounds_total{outcome="accepted_all"}` or `partial` is > 0, `turbine_speculative_accepted_tokens_total > 0`, and `turbine_overlap_serial_turns_total{reason="speculative_step"}` is > 0; with prefix reuse on (a second identical request) the stream still equals the serial one and the cached tokens are unchanged; after a preemption the stream still equals.
- [ ] Write failing test `engine::r#loop::tests::draft_kv_lifecycle`: after a request completes, a second request's `begin` succeeds (the slot was released); while a request holds the slot, a second `begin` on the same proposer returns `ProposeError::SlotBusy` (no sharing); a partial accept truncates the draft table to the target's accepted length (`propose` from that length equals the serial draft run); a finish, cancel and preempt each release the slot; the draft pool's free-block count returns to its start between requests.
- [ ] Write failing test `engine::r#loop::tests::speculative_errors`: a tiny draft executor wrapped to return `ModelError` on its first forward makes the request end with an error event and no partial token, and a following request is served normally; a shutdown in the middle of a speculative round logs `event="engine_stopped"` once and drops both executors. (The draft load failure and the config exits are Task 3's `speculative_startup_refusals`.)
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::speculative_greedy_matches_serial` — expect FAIL "unknown field `draft`"/"no method named `step_verified`".
- [ ] Implement the proposer/engine wiring, the extension, `step_verified` for greedy, the row adapter (`Rows` over `ProcessedRow`), the metrics and the serial turn; the accepted proposals' events are the `step_sampled` outputs (so detokenization, stop strings, tool/reasoning parsing stay on the ordinary path); a proposal/step error follows the ordinary iteration error path and never emits a partial token.
- [ ] Run the three tests — expect PASS. Then `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::jump_forward` — expect PASS (the jump-forward path is unchanged).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Lab quick: `scripts/lab-test.sh novanas --tier quick` — expect PASS (GPU-facing engine code).
- [ ] Commit: `feat(engine): greedy speculative round over the multi-token append`

## Task 8: Sampling rounds, seeded bypass and the distribution test (S-3, S-6)

Files: `crates/turbine-server/src/engine/loop.rs` (sampling mode), `crates/turbine-server/src/engine/requests.rs` (`step_verified` sampling), `crates/turbine-model/tests/tiny_model.rs` (the distribution test), `crates/turbine-server/src/engine/loop.rs` tests

Interfaces:

- `step_verified` in `VerifyMode::Sampling`: builds `ProcessedRow`s with `Sampler::row`, calls `turbine_scheduler::speculative::verify_sampling` with a `Draw` adapter over the choice's sampler (`uniform()` = the sampler's 24-bit draw scaled; `draw(weights, exclude)` = `Sampler::draw_row` with a temporary row), emits the accepted proposals with their target logprobs, and the fallback token from the row's distribution without the proposal; only the emitted tokens are observed.
- Seeded choices (`Choice::is_seeded`) and constrained choices bypass under the S-6 reasons; a seeded request's stream is identical with the draft on and off.
- `crates/turbine-model/tests/tiny_model.rs::speculative_sampling_exactness`: `Sampler::row`/`draw_row` plus the verifier's acceptance/fallback over 10^6 draws yields a TV distance below 0.01 from the sampler's own distribution on the tiny model's logits.

Covers: S-3, S-6; AC 15
Depends on: Tasks 5, 6, 7

- [ ] Write failing test `engine::r#loop::tests::speculative_seeded_identical`: a seeded request (`seed: Some(7)`, temperature 0.8, top_p 0.9) with the draft off and on produces the identical token stream; a seeded request with `method: draft` never runs a speculative round (`turbine_speculative_rounds_total` unchanged; the disabled gauge/reason is `seeded`).
- [ ] Write failing test `engine::r#loop::tests::speculative_sampling_runs` (unseeded, temperature 0.8): the stream differs from the serial stream (not asserted token-equal), every emitted token is in the vocabulary, `turbine_speculative_rounds_total` includes `partial`/`accepted_all`, and no rejected proposal appears in the output (a draft whose proposals are the serial stream's next tokens cannot be fully rejected while the round counts accepted > 0).
- [ ] Write failing test `crates/turbine-model/tests/tiny_model.rs::speculative_sampling_exactness`: run 10^6 seeded rounds of the acceptance/fallback (a synthetic proposal = a fixed token) and histogram the emitted tokens; assert TV < 0.01 against the tiny model's `Sampler` distribution and that a `p(t)=0` token is never emitted.
- [ ] Run the three tests — expect FAIL "no method named `draw_row`" / "VerifyMode::Sampling".
- [ ] Implement the sampling `step_verified` with the `Draw` adapter and the bypasses; make the acceptance and fallback draws use the choice's sampler RNG.
- [ ] Run the tests — expect PASS; `scripts/remote-cargo.sh test -p turbine-model sampling::` — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Lab quick: `scripts/lab-test.sh novanas --tier quick` — expect PASS.
- [ ] Commit: `feat(engine): exact rejection sampling for speculative rounds, seeded bypass`

## Task 9: Eligibility, pressure and the per-request acceptance window (S-6, S-10)

Files: `crates/turbine-server/src/engine/loop.rs` (eligibility gate, reason counters), `crates/turbine-server/src/engine/requests.rs` (the trailing window), `crates/turbine-server/src/status.rs` or `crates/turbine-server/src/backend.rs` (the `speculative` status object), `crates/turbine-server/src/metrics.rs` (the disabled gauge), tests

Interfaces:

- `enum SpecDisabled { MethodNone, Pressure, Batch, Constrained, Seeded, Paused, LowAcceptance, NoDraftKv, NoRoom }` with `as_str()` (`method_none`, `pressure`, `batch`, `constrained`, `seeded`, `paused`, `low_acceptance`, `no_draft_kv`, `no_room`)
- `Choice` gains `spec_accept_window: [u32; 8]`/`spec_accept_index: u8`/`spec_disabled: Option<SpecDisabled>`; `record_spec_round(accepted, k)` after 8 rounds disables on `accepted_sum < min_acceptance * k * 8`
- The round gate: `plan.items.len() == 1 && plan.items[0].kind == Decode && running == 1 && !choice.is_constrained() && !choice.is_seeded() && choice not paused && !choice.spec_disabled && matches!(pressure, Green | Yellow) && proposal/slot/rows available`; `pressure` is the turn's `snap.state`
- `/turbine/v1/status` gains `speculative: { method, draft_model_path, num_tokens, min_acceptance, active, acceptance }` (backend's `status()` JSON); the disabled gauge is set per reason.

Covers: S-6, S-10; AC 9, 13
Depends on: Tasks 7, 8

- [ ] Write failing test `engine::r#loop::tests::speculative_disabled_reasons`: two concurrent requests (a two-item plan) → `batch`; a constrained choice → `constrained`; `seed: Some(1)` → `seeded`; a paused choice (output channel full) → `paused`; `pressure.state = Orange` (tick the controller) → `pressure`, then Green → a round runs; a draft that always proposes wrong tokens with `min_acceptance=0.4` for 9 rounds → the request's speculation stops (`low_acceptance`) and the stream equals the serial one; a second request with the slot held does not fail (`no_draft_kv`/`no_room` fall back). In every case the response completes and equals the serial stream.
- [ ] Write failing test `engine::r#loop::tests::speculative_metrics_and_status`: a successful speculative run renders the five series (only the documented labels), the status JSON's `speculative` object matches `method: draft`, `num_tokens: 4`, `min_acceptance: 0.4`, `active: true`, and `acceptance` in 0..=1; `method: none` reports `active: false`; ORANGE reports `active: false` and the `pressure` gauge > 0.
- [ ] Run both tests — expect FAIL "no variant `batch`"/"missing field `speculative`".
- [ ] Implement the gate, the window, the status object and the gauges.
- [ ] Run the tests — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(engine): speculative eligibility, pressure and low-acceptance disable`

## Task 10: Stop, usage, logprobs and the prefix-publish rollback invariants (S-3, S-4)

Files: `crates/turbine-server/src/engine/loop.rs` (span finish handling, rollback), `crates/turbine-server/src/engine/requests.rs` (usage/logprobs), `crates/turbine-server/src/engine/loop.rs` tests

Interfaces:

- A finish inside a span (EOS, stop id, stop string, `max_tokens`, tool finish) ends the choice at the first emitted token that triggers it; the remaining proposals are rolled back (`IterationOutcome.rolled_back`), the draft table is truncated, and the events are exactly the serial path's.
- `usage.completion_tokens` counts only emitted tokens (accepted proposals + the target's next token); `logprobs`/`top_logprobs` for an emitted proposal are the target row's values (`logprob_of`, `top_logprobs` from the same post-processing); `echo`, `cached_tokens` and `lossy_cached_tokens` are unchanged.
- A full block whose tokens are all kept publishes its prefix; a block with a rejected position is not published; the next non-speculative step publishes at the next boundary.

Covers: S-3, S-4; AC 6, 7
Depends on: Tasks 7, 8

- [ ] Write failing test `engine::r#loop::tests::speculative_stop_and_usage`: with the tiny draft, `max_tokens: 3` and `ignore_eos: false`, the finish reason and the emitted events equal the serial run; `usage.completion_tokens` equals the emitted tokens; a `stop` string and a stop token id inside a span both end at the serial position; with `logprobs: Some(2)` every emitted token's `logprob`/`top_logprobs` equals the serial run's for the same token.
- [ ] Write failing test `engine::r#loop::tests::speculative_rollback_publishes_kept_prefix`: with a draft forced to propose a known-wrong token (a test proposer or a draft whose next token is fixed), a later request with the kept prefix hits the cache only for the kept tokens; a full block with a rejected token is not served; the KV document's block count after finish equals the serial run's; a tail copy still restores the shared prefix.
- [ ] Run both tests — expect FAIL "usage.completion_tokens == 4"/"the rejected block was published".
- [ ] Implement the finish/usage/logprobs handling and the publish rule (the existing rollback path already shortens the table; add the publish-hash check and the snapshot drop).
- [ ] Run the tests — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(engine): speculative finish, usage and kept-prefix publish invariants`

## Task 11: Hybrid recurrent-state rollback with a scratch slot (S-5)

Files: `crates/turbine-server/src/engine/loop.rs` (scratch slot, copy/restore, replay), `crates/turbine-server/src/kv_orchestrator.rs` (state copy helper), `crates/turbine-kv/src/state.rs` (a copy entry point if needed), `crates/turbine-server/src/engine/loop.rs` tests

Interfaces:

- The target's hybrid state sets gain a per-sequence scratch state entry — a state-sized entry reserved from the hybrid snapshot pool (its one-entry-per-block allocation is already charged to the budget; a second live `StateSlot` would double every per-sequence executor buffer, because live slots are indexed within the batch's `max_seqs`) — released with the live slot; before a speculative span the live state is copied to the scratch (the existing `StateCopy` path, `StateLoc::Slot → StateLoc::Snapshot`); full acceptance keeps the live state; partial acceptance copies the scratch back, then runs one replay forward of the kept tokens (≤ k) to re-advance the live state, then truncates the draft table to the kept length.
- The block-boundary snapshot for a block inside a rejected span is dropped (the snapshot entry is released) and retaken at the next non-speculative boundary; a kept block's snapshot uses the state after the kept tokens only.
- Snapshot/state release on finish, cancel and preempt releases the scratch with the live slot; `n > 1` state copies are unchanged.

Covers: S-5; AC 8
Depends on: Tasks 7, 10

- [ ] Write failing test `engine::r#loop::tests::hybrid_speculative_rollback_matches_serial`: a tiny Qwen3.5 target and a scripted proposer (k=4, greedy) that repeats the serial stream and shifts its second proposal on round 2 → the next and later rounds' stream equals the serial stream (a wrong restored state would change it); a block-boundary snapshot inside the rejected span is not stored (the speculative run's `turbine_hybrid_state_snapshots_total{outcome="stored"}` is lower than the serial run's by the skipped boundary, and still non-zero); the scratch entry is released on finish and cancel; `speculative_scratch_released_on_preempt` covers release on preemption. (Deviation from the first draft: a test-only proposer, no second draft model, and state equality is observed through the stream rather than raw bytes — the engine's executor buffer is not readable from the test.)
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::hybrid_speculative_rollback_matches_serial` — expect FAIL "no scratch state slot".
- [ ] Implement the scratch allocation, copy/restore, replay and snapshot drop; the replay forward is a single small step over the kept tokens (`Step` with `rows = kept`), measured by the existing step metric.
- [ ] Run the test — expect PASS; the existing hybrid tests (`hybrid_state`, `prefix_reuse`, `snapshot`) — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Lab quick: `scripts/lab-test.sh novanas --tier quick` — expect PASS.
- [ ] Commit: `feat(engine): hybrid recurrent-state rollback for speculative spans`

## Task 12: Lab tooling, the pinned draft checkpoint and the lab config (S-11)

Files: `scripts/lab/phase8-checkpoints.tsv` (new), `scripts/fetch-checkpoint.sh` (`--table`), `scripts/lab-serve.sh` (`--vllm-speculative`), `scripts/lab-bench.sh` (`--save-metrics`), `benches/turbine-bench/tests/lab_scripts.rs` (dry-run tests), `scripts/lab/phase8-novanas-llama-draft.yaml` (new), `AGENTS.md` (Commands/Lab hosts), `.procoder/perf-log.md` (the fetch row)

Interfaces:

- `scripts/fetch-checkpoint.sh [--dry-run] [--table scripts/lab/phase8-checkpoints.tsv] novanas <slug>`; the default table stays `phase7-checkpoints.tsv`
- `scripts/lab/phase8-checkpoints.tsv` row: `llama-3.2-1b-instruct	unsloth/Llama-3.2-1B-Instruct	5a8abab4a5d6f164389b1079fb721cfab8d7126c	*	2.5`
- `scripts/lab-serve.sh … --vllm <slug> --vllm-speculative <json>`: appends `--speculative-config` and the JSON to `vllm serve`; the JSON is emitted as a YAML single-quoted line in `scripts/lab/novanas-vllm-job.yaml`'s `__EXTRA_ARGS__`, and a JSON containing a single quote is refused (exit 2, `lab-serve: --vllm-speculative expects a JSON object without a single quote`); `--dry-run` prints both arguments
- `scripts/lab-bench.sh … --save-metrics`: after the throughput bench, writes the `/metrics` scrape to `<run dir>/metrics.prom` (the same scrape the stage-share already takes); the dry-run prints the `curl` line; the `BENCH` line is unchanged
- `scripts/lab/phase8-novanas-llama-draft.yaml`: the Phase 7 Llama config keys plus `speculative.method: draft`, `speculative.num_tokens: 4`, `speculative.min_acceptance: 0.4`, `speculative.draft_model_path: /home/piwi/turbine-models/llama-3.2-1b-instruct`, `model.path` the 3B, `model.max_seq_len` and `kv.block_tokens` from Phase 7.

Covers: S-11 (tooling); AC 14 (the recipe)
Depends on: Tasks 1, 3

- [ ] Write failing test `lab_scripts::fetch_checkpoint_phase8_table`: `scripts/fetch-checkpoint.sh --dry-run --table scripts/lab/phase8-checkpoints.tsv novanas llama-3.2-1b-instruct` exits 0 and prints `hf download unsloth/Llama-3.2-1B-Instruct --revision 5a8abab4a5d6f164389b1079fb721cfab8d7126c --local-dir /home/piwi/turbine-models/llama-3.2-1b-instruct`; an unknown slug exits 2; `--table` is honoured; a failing `ssh` stub keeps the test off the host.
- [ ] Write failing test `lab_scripts::vllm_speculative_dry_run`: `scripts/lab-serve.sh --dry-run novanas --vllm llama-3.2-3b-instruct --vllm-speculative '{"method":"draft_model","model":"/models/llama-3.2-1b-instruct","num_speculative_tokens":4}'` exits 0 and prints `--speculative-config` followed by the JSON as one argument; a JSON with `'` exits 2.
- [ ] Write failing test `lab_scripts::lab_bench_save_metrics_dry_run`: `scripts/lab-bench.sh --dry-run --model llama --save-metrics` prints the `/metrics` `curl` into `target/lab-bench/<label>-llama/metrics.prom`.
- [ ] Run the three tests — expect FAIL "unknown option --table"/"unknown flag --vllm-speculative"/"unknown option --save-metrics".
- [ ] Implement the three flags, add the TSV row and the lab config; `bash -n scripts/fetch-checkpoint.sh scripts/lab-serve.sh scripts/lab-bench.sh` and `shellcheck scripts/fetch-checkpoint.sh scripts/lab-serve.sh scripts/lab-bench.sh` — expect no output.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts` — expect PASS.
- [ ] Fetch the checkpoint (user approved 2026-10-25; run in the background): `scripts/fetch-checkpoint.sh --table scripts/lab/phase8-checkpoints.tsv novanas llama-3.2-1b-instruct` — expect `fetch: llama-3.2-1b-instruct 5a8abab4a5d6f164389b1079fb721cfab8d7126c ok` and `ssh novanas 'ls /home/piwi/turbine-models/llama-3.2-1b-instruct/config.json'`.
- [ ] Document the config and the fetch in `AGENTS.md`; record the fetch in `.procoder/perf-log.md`.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `feat(lab): phase-8 checkpoint table, draft lab config and the vLLM/bench flags`

## Task 13: Contract, docs and AGENTS.md (S-2, S-9, S-10)

Files: `.procoder/contract/interfaces.md` (§28), `docs/extending/speculative-proposer.md` (final pass), `docs/extending/README.md`, `AGENTS.md` (the Phase 8 config/commands/support rows), `crates/turbine-model/tests/docs_extending.rs`

Interfaces:

- §28 "Phase 8 additions" names `SpeculativeConfig`/`SpeculativeMethod`, the four keys, `Proposer`/`ProposeRequest`/`Proposals`/`ProposeError`/`ProposerFactory`/`PROPOSERS`/`proposer`/`DraftProposerConfig`, `speculative::{Rows, Draw, Verdict, verify_greedy, verify_sampling}`, `SpeculativeColumn::Draft`, the five metric names and their label enums, the status object, the reason codes and the two support rows.
- `AGENTS.md` gains the `speculative.*` keys in the Commands section, the phase-8 lab recipe, the draft checkpoint row and the support-matrix sentence.

Covers: S-2, S-9, S-10; AC 3 (docs), 12
Depends on: Tasks 1–12

- [ ] Write failing test `docs_extending::speculative_proposer_page_is_current`: the page's commands/names appear in the tree (`PROPOSERS`, `DraftProposerConfig`, `draft_model`) and the index links it; run `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — expect FAIL until the page and the index entry are added.
- [ ] Write the §28 additions and the AGENTS.md paragraphs; run the test — expect PASS.
- [ ] Run `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` and `scripts/remote-cargo.sh test -p turbine-core support::tests::` — expect PASS.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`.
- [ ] Commit: `docs: Phase 8 contract, proposer page and AGENTS.md`

## Task 14: Phase exit and the Llama draft row (S-9, S-11, S-12)

Files: `crates/turbine-core/src/support.rs` (the supported amd row and `phase8_rows`), `.procoder/perf-log.md`, `tests/bench/phase-8-speculative-decoding/README.md` (new) and its JSON reports, `AGENTS.md` (the row), `.procoder/specs/phase-8-speculative-decoding.md` (tick the gate box only after the evidence), `.procoder/ask/decisions.md` (the gate result)

Interfaces:

- The `amd gfx1201 LlamaForCausalLM bf16 bf16 draft` row is `supported` only when every gate below passes; a performance-only miss makes it `experimental` with the measured number in the reason and the row stays there.
- `support::tests::phase8_rows` asserts the supported state after the flip; `baseline_rows_present` continues to pass (the supported row is `amd`, bf16, gfx1201, Llama).

Covers: S-9, S-11, S-12; AC 12, 14, 17, 18
Depends on: all earlier tasks

- [ ] Gate 1 — full host/GPU tier: `scripts/gate.sh --full` (`gate: ok`), `scripts/lab-test.sh novanas --tier full` (PASS), `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` (PASS). Start each in the background; paste the one-line result.
- [ ] Gate 2 — serve the draft: `scripts/lab-serve.sh novanas scripts/lab/phase8-novanas-llama-draft.yaml` (background; `/ready` 200; the log has `event="speculative"` with `draft_model_path`); `curl -s http://192.168.10.203:18000/turbine/v1/status | jq .speculative` shows `method: draft`, `active: true`.
- [ ] Gate 3 — golden and the serial A/B: with the draft server running, `scripts/remote-cargo.sh run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --concurrency 1 --output json > target/bench-p8/golden-c1.json` and `--concurrency 16 --output json > target/bench-p8/golden-c16.json` (both exit 0 under the slug's tolerance); then `... capture --url http://192.168.10.203:18000 --prompts tests/golden/prompts.jsonl --out target/bench-p8/spec-capture.jsonl`; `scripts/lab-serve.sh novanas --stop`; `scripts/lab-serve.sh novanas scripts/lab/phase8-novanas-llama-draft.yaml --set speculative.method=none` (the no-draft server); `... compare --url http://192.168.10.203:18000 --reference target/bench-p8/spec-capture.jsonl --concurrency 1 --output json > target/bench-p8/ab-serial.json` (exit 0); `scripts/lab-serve.sh novanas --stop`.
- [ ] Gate 4 — throughput/TTFT/acceptance with `scripts/lab-bench.sh` (it serves, benches and stops): `scripts/lab-bench.sh --model llama --concurrency 1 --label p8-c1-draft --save-metrics -- --set speculative.method=draft --set speculative.draft_model_path=/home/piwi/turbine-models/llama-3.2-1b-instruct --set speculative.num_tokens=4 --set speculative.min_acceptance=0.4`; `scripts/lab-bench.sh --model llama --concurrency 1 --label p8-c1-none --save-metrics -- --set speculative.method=none`; `scripts/lab-bench.sh --model llama --label p8-c16-draft --save-metrics -- --set speculative.method=draft --set speculative.draft_model_path=/home/piwi/turbine-models/llama-3.2-1b-instruct --set speculative.num_tokens=4 --set speculative.min_acceptance=0.4`. From the three `BENCH` lines: `tok/s`(draft c1) / `tok/s`(none c1) ≥ 1.3 and `ttft_p50`(draft c1) / `ttft_p50`(none c1) ≤ 1.15; `tok/s`(draft c16) ≥ 0.97 × 863.9; `target/lab-bench/p8-c1-draft-llama/metrics.prom` contains `turbine_speculative_rounds_total`, `turbine_speculative_accepted_tokens_total`, `turbine_speculative_proposed_tokens_total`, the `turbine_speculative_accepted_per_round` histogram and a non-zero `turbine_speculative_accepted_tokens_total`; record the accepted/proposed ratio and the histogram mean.
- [ ] Gate 5 — vLLM reference: `scripts/lab-serve.sh novanas --vllm llama-3.2-3b-instruct --vllm-speculative '{"method":"draft_model","model":"/models/llama-3.2-1b-instruct","num_speculative_tokens":4}'`; then `scripts/remote-cargo.sh run --release -p turbine-bench --bin turbine-bench -- --url http://192.168.10.203:18100 --model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --requests 50 --prompt-words 512 --max-tokens 256 --ignore-eos --output json > target/bench-p8/vllm-c1.json`; record `vllm_tok_s` and the Turbine ratio; if vLLM-ROCm cannot serve speculation on this card (the Job log names the refusal), record it and apply the no-vLLM rule; `scripts/lab-serve.sh novanas --stop`.
- [ ] Gate 6 — soak: `scripts/overload-soak.sh novanas --duration 10m --set speculative.method=draft --set speculative.draft_model_path=/models/llama-3.2-1b-instruct --set speculative.num_tokens=4 --set speculative.min_acceptance=0.4` (the serve Job sees the mounted `/models`, not the host path); the verdict is `pass: true` (the draft-configured server stays stable; the c1 exercise is Gate 4).
- [ ] Flip the row only if the S-11 bar passes; the 2026-10-27 outcome is below, so this was **not** applied: change the existing `amd gfx1201 LlamaForCausalLM bf16 bf16 draft` row's status from `Experimental` (and its `P8_PENDING_REASON`) to `Supported` with no reason, and update `support::tests::phase8_rows` to expect `supported`; run `scripts/remote-cargo.sh run -q -p turbine-server -- --support-matrix --output text` and `scripts/remote-cargo.sh test -p turbine-core support::tests::phase8_rows` — expect the row supported and the test PASS.
- [ ] Write `tests/bench/phase-8-speculative-decoding/README.md` with every command, the raw numbers, the acceptance length, the vLLM result/no-vLLM note and the row decision; copy the four JSON reports and the `metrics.prom` beside it; add the `.procoder/perf-log.md` `- P8 closed:` line (or `- P8 open:` with the miss) and the `.procoder/ask/decisions.md` gate entry; tick the spec's `[S-11]` box only with the evidence.
- [ ] Run `scripts/gate.sh --full` again after the docs/row edit — expect `gate: ok`.
- [ ] `scripts/lab-serve.sh novanas --stop`; confirm no `turbine-lab-serve-*` or `turbine-lab-vllm-*` Job remains and the device memory is free.
- [ ] Commit: `feat(support): flip the Llama draft row after the Phase 8 gate` (the row plus the evidence files; do not commit the transient serve logs).


## Task 14 outcome (2026-10-27)

Executed on `novanas`; evidence in `tests/bench/phase-8-speculative-decoding/`.

| Gate | Result |
| --- | --- |
| 1a `gate.sh --full` | PASS, 1154 tests |
| 1b `lab-test --tier full` | PASS (run `1026151053-12725e3c`) |
| 1c `lab-test --gpus 2 --features fault-injection --tier full` | PASS (70 targets, run `1026172029-2d850040`) |
| 1d `lab-bench --golden16` | PASS: Llama 16/16, 866.3 tok/s; OLMoE 16/16, 637.6 tok/s |
| 2 serve/status | PASS: `active: true`, `event="speculative"` |
| 3a golden c1/c16 | PASS: 16/16 and 16/16 |
| 3b serial A/B | PASS: 16/16 strict |
| 4a c1 TTFT | PASS: 1.006× |
| 4b c1 decode | **MISS: 0.937×** (82.99 vs 88.54 tok/s, 50-request S-11 profile; 0.938× on the 200-request lab-bench profile; acceptance 17.1 %, 0.68 tokens/round) |
| 4c c16 | PASS: 845.4 tok/s = 0.979× of 863.9 |
| 5 vLLM speculative | recorded: 64.47 tok/s c1 (Turbine draft 1.29×) |
| 6 10-minute soak | PASS: ITL p99 210 ms vs 170.5 ms, GREEN 5 s, 0 drops |

The row stays `experimental` with `P8_PENDING_REASON` carrying the measured miss
(S-9 forbids a supported row without the gate). The phase-exit requirement of a
`supported` row is therefore not met; the performance miss is the recorded outcome.
