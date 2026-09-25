# P0-T9 Workspace acceptance, examples/turbine.yaml and AGENTS.md commands

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 9 (`.procoder/plans/phase-0-skeleton.md`, "## Task 9"): Workspace acceptance, examples/turbine.yaml and AGENTS.md commands. Covers S-1 (workspace-wide build/test/lint/format on macOS; `unsafe` isolation; `forbid` elsewhere), S-8 (`AGENTS.md` Commands).. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo build --workspace && cargo test --workspace && cargo test -p turbine-core && cargo test -p turbine-core config::tests::byte_size_parsing && cargo test -p turbine-api --test api route_table_phase0 && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check && cargo run -q -p turbine-server -- --config examples/turbine.yaml --check-config && cargo run -q -p turbine-bench -- --help >/dev/null` passes (expect PASS (exit 0; 23 passed, 0 failed, 1 ignored across the workspace).) — measured 26 passed, 0 failed, 0 ignored: Tasks 1–7 added more tests than the plan counted, and the 1 ignored test is Task 8's lab test, not yet merged here.
- [x] `cargo run -q -p turbine-server -- --config examples/turbine.yaml --set server.listen=127.0.0.1:8000 & sleep 3; curl -s -i http://127.0.0.1:8000/ready | head -1; curl -s http://127.0.0.1:8000/metrics | grep turbine_devices; kill -TERM %1; wait %1; echo exit=$?` passes (expect PASS: `HTTP/1.1 503 Service Unavailable`, `turbine_devices{vendor="nvidia"} 0`, `turbine_devices{vendor="amd"} 0`, log `shutdown requested; finishing in-flight requests signal="SIGTERM"`, `exit=0` (use SIGTERM: a non-interactive shell starts background jobs with SIGINT ignored).)
- [x] `grep -rn "unsafe" crates/*/src benches/*/src | grep -v '^crates/turbine-device/src/'` passes (expect no output; `grep -L 'unsafe_code = "forbid"' crates/*/Cargo.toml benches/*/Cargo.toml` — expect exactly `crates/turbine-device/Cargo.toml`; every `unsafe {` in `crates/turbine-device/src` has `// SAFETY:` within the four lines above it.)
- [x] `printf '\npub fn unsafe_probe() {\n    unsafe {}\n}\n' >> crates/turbine-core/src/lib.rs && cargo build -p turbine-core` passes (expect FAIL (`error: usage of an `unsafe` block`); restore with `git checkout -- crates/turbine-core/src/lib.rs` and `cargo build --workspace` — expect PASS.)
- [x] `cargo tree --workspace -e normal | grep -iE 'cudarc|cuda-sys|rocm|hip-sys'` passes (expect no output (`nvml-wrapper-sys` is present and resolves NVML through `libloading` at runtime).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message — committed on worktree branch `worktree-agent-a6c881d96cf0106c6`; the coordinator ticks this after merging into `phase-0-skeleton`.
- [x] Lab-dependent (coordinator ticks after Task 8 lands): `scripts/lab-test.sh` with no argument prints the usage line and exits 2, and `scripts/lab-test.sh novanas` (ask the user first) prints `lab-test: novanas: PASS` — verifies the AGENTS.md lab command as written; the workspace test count then includes Task 8's ignored lab test.

## Evidence

- Red: `cargo run -q -p turbine-server -- --config examples/turbine.yaml --check-config` before the file existed → `turbine-server: invalid configuration: cannot read examples/turbine.yaml: No such file or directory (os error 2)`, `exit=2`. Green after adding `examples/turbine.yaml` → `config ok`, `exit=0`.
- Acceptance chain (build, test, one crate, unit test, integration test, clippy `-D warnings`, `fmt --check`, check-config, bench `--help`) → `exit=0`. `cargo test --workspace` totals: 26 passed, 0 failed, 0 ignored. `test config::tests::byte_size_parsing ... ok`; `test route_table_phase0 ... ok`.
- Server run (via `bash -c`, since zsh's non-interactive job control does not resolve `%1`) → `HTTP/1.1 503 Service Unavailable`, `turbine_devices{vendor="nvidia"} 0`, `turbine_devices{vendor="amd"} 0`, `INFO turbine_server::startup: shutdown requested; finishing in-flight requests signal="SIGTERM"`, `shutdown complete`, `exit=0`.
- `grep -rn "unsafe" crates/*/src benches/*/src | grep -v '^crates/turbine-device/src/'` → no output. `grep -L 'unsafe_code = "forbid"' crates/*/Cargo.toml benches/*/Cargo.toml` → `crates/turbine-device/Cargo.toml` only. SAFETY scan (`grep -rn -B4 "unsafe {" crates/turbine-device/src` + awk) → `17 unsafe blocks, 0 missing SAFETY`.
- Unsafe probe in `crates/turbine-core/src/lib.rs` → `error: usage of an `unsafe` block`; restored with `git checkout -- crates/turbine-core/src/lib.rs`; `cargo build --workspace` → `Finished`, exit 0.
- `cargo tree --workspace -e normal | grep -iE 'cudarc|cuda-sys|rocm|hip-sys'` → no output (grep exit 1); `nvml-wrapper-sys v0.10.0`, `libloading v0.8.9`/`v0.9.0` present.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0. `launcher.sh check` → `0 blocking`. `launcher.sh agents --host claude` → `every agent rule file matches AGENTS.md`.
- Remaining AGENTS.md commands run as written: `cargo fmt --all` exit 0; `cargo run -p turbine-bench -- --help` exit 0; `cargo run -p turbine-server -- --config examples/turbine.yaml --check-config` → `config ok`.
- Bench run command: against the Phase 0 server it exits 2 (`GET http://127.0.0.1:8000/v1/models returned no models; pass --model`), with `--model` it exits 1 (the skeleton serves no completions). The real-stream bench run is Phase 1's (decision 2026-09-25; spec §Out of scope); AGENTS.md states this next to the command.
- Lab (coordinator, after Task 8 landed as 4255644): `scripts/lab-test.sh` with no argument exits 2 (checked on phase-0-skeleton); `scripts/lab-test.sh novanas` printed `lab-test: novanas: PASS` (P0-T8 evidence, Job complete 1/1 in 55 s).
