# P1-T7 Model config parsing and the architecture allowlist

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 7 (`.procoder/plans/phase-1-single-request.md`, "## Task 7"): Model config parsing and the architecture allowlist. Covers S-3 AC `config::tests::parses_target_config`, `config::tests::rejects_unsupported`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model config::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message
- [x] `config::tests::rejects_unsupported` F8_E4M3 case: `check_supported_weights` names field `tensor dtype`, the tensor and supported `BF16` (landed with Task 9)

## Evidence

- Red: `cargo test -p turbine-model config::tests::parses_target_config` before the implementation → `error[E0425]: cannot find function load_model_config in this scope` and `error[E0433]: cannot find type RopeScaling in this scope` (compile failure: config API absent).
- Green: `cargo test -p turbine-model config::tests` → `test result: ok. 3 passed; 0 failed` (`parses_target_config`: 28 layers, hidden 3072, 24/8 heads, head_dim 128, intermediate 8192, vocab 128256, tied, theta 500000, `Llama3 { 32, 1, 4, 8192 }`, EOS `[128001, 128008, 128009]`, `kv_layout(16)` 114_688 / 1_835_008 B, shape weight_bytes 6_425_499_648; `rejects_unsupported`: `unsupported architectures = Qwen3MoeForCausalLM; supported: LlamaForCausalLM`, `quantization_config` → supported `none`, `torch_dtype float16` → `bfloat16`, rope_type `yarn` → `default, llama3`; `head_dim_and_eos_fallbacks`: hidden/heads head_dim, config.json EOS fallback, missing dir → `Io` naming `config.json`).
- Deferred to Task 9 (coordinator adjustment): the third `rejects_unsupported` case (tiny checkpoint holding an `F8_E4M3` tensor → `ModelArchConfig::check_supported_weights` naming field `tensor dtype`, the tensor and supported `BF16`) needs `SafetensorsIndex` (Task 8) and the tiny checkpoint writer (Task 9); `check_supported_weights` and that case land with Task 9, which takes over the criterion.
- Fixtures: `config.json` and `generation_config.json` fetched from `unsloth/Llama-3.2-3B-Instruct` at `006f5dcd1393c3add266de40994ba96225e9689d`, then reformatted by the gate's JSON formatter (whitespace and `1e-05` → `1e-5` only; same values). `tokenizer.json` (16 MB, over the gate's 5 MB blocking limit) and `tokenizer_config.json` are not committed here — they are outside this task's `Files:` line and are left to Tasks 11/12.
- Workspace: `cargo test --workspace` → 72 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → 0 blocking.
- Commit: `feat(turbine-model): HF config parsing and architecture allowlist` on the Task 7 worktree branch (branched from phase-1-single-request; lands there on merge).
- F8_E4M3 case (Task 9 commit `feat(turbine-model): positioned-read weight loader and tiny synthetic checkpoint`): `ModelArchConfig::check_supported_weights` implemented; `cargo test -p turbine-model config::tests::rejects_unsupported` → `test result: ok. 1 passed` with the tiny checkpoint re-serialized so `model.layers.0.mlp.down_proj.weight` is F8_E4M3 → `unsupported tensor dtype = F8_E4M3 (model.layers.0.mlp.down_proj.weight); supported: BF16`. Mutation check: forcing the BF16 test to pass → the case fails with `called Result::unwrap_err() on an Ok value`.
