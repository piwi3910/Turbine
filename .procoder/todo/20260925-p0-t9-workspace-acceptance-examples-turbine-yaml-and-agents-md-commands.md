# P0-T9 Workspace acceptance, examples/turbine.yaml and AGENTS.md commands

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 9 (`.procoder/plans/phase-0-skeleton.md`, "## Task 9"): Workspace acceptance, examples/turbine.yaml and AGENTS.md commands. Covers S-1 (workspace-wide build/test/lint/format on macOS; `unsafe` isolation; `forbid` elsewhere), S-8 (`AGENTS.md` Commands).. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo build --workspace && cargo test --workspace && cargo test -p turbine-core && cargo test -p turbine-core config::tests::byte_size_parsing && cargo test -p turbine-api --test api route_table_phase0 && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check && cargo run -q -p turbine-server -- --config examples/turbine.yaml --check-config && cargo run -q -p turbine-bench -- --help >/dev/null` passes (expect PASS (exit 0; 23 passed, 0 failed, 1 ignored across the workspace).)
- [ ] `cargo run -q -p turbine-server -- --config examples/turbine.yaml --set server.listen=127.0.0.1:8000 & sleep 3; curl -s -i http://127.0.0.1:8000/ready | head -1; curl -s http://127.0.0.1:8000/metrics | grep turbine_devices; kill -TERM %1; wait %1; echo exit=$?` passes (expect PASS: `HTTP/1.1 503 Service Unavailable`, `turbine_devices{vendor="nvidia"} 0`, `turbine_devices{vendor="amd"} 0`, log `shutdown requested; finishing in-flight requests signal="SIGTERM"`, `exit=0` (use SIGTERM: a non-interactive shell starts background jobs with SIGINT ignored).)
- [ ] `grep -rn "unsafe" crates/*/src benches/*/src | grep -v '^crates/turbine-device/src/'` passes (expect no output; `grep -L 'unsafe_code = "forbid"' crates/*/Cargo.toml benches/*/Cargo.toml` — expect exactly `crates/turbine-device/Cargo.toml`; every `unsafe {` in `crates/turbine-device/src` has `// SAFETY:` within the four lines above it.)
- [ ] `printf '\npub fn unsafe_probe() {\n    unsafe {}\n}\n' >> crates/turbine-core/src/lib.rs && cargo build -p turbine-core` passes (expect FAIL (`error: usage of an `unsafe` block`); restore with `git checkout -- crates/turbine-core/src/lib.rs` and `cargo build --workspace` — expect PASS.)
- [ ] `cargo tree --workspace -e normal | grep -iE 'cudarc|cuda-sys|rocm|hip-sys'` passes (expect no output (`nvml-wrapper-sys` is present and resolves NVML through `libloading` at runtime).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
