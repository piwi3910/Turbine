# phase-0-skeleton

Status: complete

Source: `turbine-spec.md` §19 Phase 0 (skeleton). Sections of that document are cited as "TS §N". Decisions and their options are recorded in `.procoder/ask/decisions.md`.

## Problem

Turbine exists only as a written specification. Every later phase (single-request correctness, serving runtime, reliability, KV tiers, distribution) needs the same foundation: a Cargo workspace with real crate boundaries, a validated configuration model, tracing and Prometheus metrics, an inventory of the GPUs actually present, an HTTP server exposing the V1 route surface, and a benchmark harness that can measure an OpenAI-compatible endpoint. The development lab is heterogeneous — two NVIDIA GB10 (DGX Spark, arm64, unified memory) machines and one amd64 machine with two AMD Radeon AI PRO R9700 cards — and the primary workstation (macOS arm64) has no GPU at all. Building the foundation first, and proving it on novanas (the NVIDIA proof on the Sparks follows in `phase-2b-nvidia`, the first phase that runs there), means Phase 1 starts on a runnable, observable, testable service instead of inventing infrastructure while chasing numerical correctness.

## Users

- **Turbine developers (humans and AI agents):** need a workspace that builds, tests and lints with single commands on macOS, clear crate boundaries for Phase 1+ code, and one command that runs the suite against real GPUs on each lab host.
- **Operators evaluating Turbine:** need to start `turbine-server` from a config file with CLI overrides, see it refuse an impossible config before binding a port, and probe `/health`, `/ready`, `/metrics` and `/turbine/v1/*`.
- **Benchmark runners:** need a harness that drives any OpenAI-compatible endpoint (Turbine, vLLM, SGLang) with a fixed, seeded request profile and reports TTFT, ITL, throughput and latency percentiles (TS §18).

## In scope

- [S-1] Cargo workspace at the repository root with six crates: `crates/turbine-core` (configuration model), `crates/turbine-observability` (tracing + metrics), `crates/turbine-device` (GPU discovery), `crates/turbine-api` (Axum router and handlers), `crates/turbine-server` (the `turbine-server` binary), `benches/turbine-bench` (the `turbine-bench` binary). Edition 2024, `rust-version = "1.97"`, `license = "Apache-2.0"`, a root `LICENSE` file with the Apache-2.0 text. `unsafe_code = "forbid"` in every crate except `turbine-device`.
- [S-2] Configuration model in `turbine-core`: the TS §15 YAML shape plus _server.max_request_bytes_, a `logging` section and a `devices` section (see Interfaces). Loaded from a file, overridden by `--set <dotted.key>=<yaml value>` flags, rejected before any port is bound when invalid, with the error naming the dotted key path.
- [S-3] Observability in `turbine-observability`: `tracing` subscriber (text or JSON), a `prometheus-client` registry rendered at `GET /metrics`, the metrics listed under Interfaces, and an `x-request-id` on every HTTP request, echoed in the response and recorded on the request's tracing span.
- [S-4] Device discovery in `turbine-device`: NVIDIA via NVML (`nvml-wrapper`) and AMD via a minimal `libloading` FFI to _libamd_smi.so_, both loaded at runtime. Produces a static inventory (identity, architecture, memory) exposed at `GET /turbine/v1/devices`. GB10-style unified memory is reported as kind `unified`.
- [S-5] HTTP API shell in `turbine-api` with every V1 route from TS §13 registered, the Phase 0 behaviour defined under Interfaces, a request body limit, and graceful shutdown on SIGINT/SIGTERM.
- [S-6] Benchmark harness `turbine-bench`: streaming load generator against any OpenAI-compatible endpoint, seeded synthetic prompts, fixed concurrency, reporting TTFT, ITL, E2E latency (p50/p95/p99), request throughput and output-token throughput as text or JSON.
- [S-7] Lab test runner `scripts/lab-test.sh <host>`: runs the full test suite including GPU tests inside a `rust:1.97-trixie` container on `dgx-spark`, `dgx-spark2` (Docker, `--gpus all`) and `novanas` (a k3s Job requesting `amd.com/gpu: 2`). The script supports all three hosts as an interface; the Phase 0 acceptance run is on novanas only (decision 2026-09-25) — the Spark runs are first exercised in `phase-2b-nvidia`.
- [S-8] Developer commands (build, test, single test, lint, format, lab test, run server, run bench) documented in `AGENTS.md`.

## Out of scope

- Model loading, tokenization, safetensors, any inference or token generation (Phase 1).
- CUDA/ROCm kernels, the kernel registry, FFI kernel shims, the `kernels/` tree (Phase 1).
- Scheduler, batching, KV cache, pressure controller logic (Phases 2–4). Their diagnostic routes exist and return 501.
- Live GPU telemetry (utilisation, temperature, clocks, free memory) — Phase 3 reliability inputs. Phase 0 reports a static inventory captured at startup.
- Crates `turbine-model`, `turbine-tensor`, `turbine-kernels`, `turbine-kv`, `turbine-scheduler`, `turbine-reliability`, `turbine-distributed`, `turbine-transport` — each is created in the phase that gives it content.
- Request timeouts (they interact with SSE streaming and arrive with Phase 2).
- Parsing or validating inference request bodies (Phase 2).
- Multi-GPU, multi-node, topology graph, transport (Phases 5–6).
- Authentication, TLS, multi-tenant accounting.
- GitHub/CI pipelines, container images for Turbine itself, deployment manifests.
- Any run on `dgx-spark` or `dgx-spark2` (decision 2026-09-25): the Spark discovery check (`scripts/lab-test.sh dgx-spark|dgx-spark2`, 1 NVIDIA GB10 of kind `unified`) moves to `phase-2b-nvidia`, the first phase that runs on the Sparks.
- A `turbine-bench` run against a real OpenAI-compatible stream (decision 2026-09-25): moved to `phase-1-single-request`, run against Turbine itself on novanas; Phase 0 verifies the harness against in-test mock SSE servers only.

## Constraints

- Rust only; no Python in the build or runtime path (TS §21 rule 4).
- Libraries: Tokio, Axum 0.8, Serde, `serde_norway` (YAML; `serde_yaml` is archived), `tracing` + `tracing-subscriber`, `prometheus-client`, `thiserror`, `clap` 4, `nvml-wrapper`, `libloading`, `reqwest` (bench), `tower-http`. No other runtime dependency without a recorded reason.
- The workspace must build and every non-ignored test must pass on macOS arm64 with no GPU and no GPU libraries installed.
- GPU libraries are loaded at runtime only; nothing links against CUDA, NVML or ROCm at build time, so one source tree builds on macOS, arm64 Linux and amd64 Linux.
- `unsafe` code and FFI exist only in `turbine-device`, each `unsafe` block carrying a `// SAFETY:` comment stating the ownership/lifetime rule it relies on (TS §21 rule 10).
- Every queue and buffer is bounded (TS §21 rule 8): request bodies are limited by _server.max_request_bytes_; the bench's in-flight requests are limited by `--concurrency`.
- Metric label values are drawn from bounded sets (route templates, not raw paths) so cardinality cannot grow with traffic.
- Every automatic decision is explained in structured logs (TS §14): each discovery backend logs its outcome and reason.
- Lab hosts: `dgx-spark` (192.168.10.246) and `dgx-spark2` (192.168.10.245) are Ubuntu 24.04 arm64 with Docker 29 and the `nvidia` runtime; `novanas` (192.168.10.203) is Debian 13 amd64, k3s v1.35 with `amdgpu-device-plugin`, ROCm 7.14.1 at `/opt/rocm/rocm`. All reached as user `piwi` over SSH with the existing key. The Sparks also serve production vLLM on :8000/:8890/:8891; lab runs must not stop or restart those containers.

## Interfaces

### `turbine-server`

```
turbine-server --config <path> [--set <dotted.key>=<yaml value>]... [--check-config]
```

- `--config` is required. `--set` may repeat; values are parsed as YAML scalars (`--set kv.cpu.enabled=false`, `--set server.listen=127.0.0.1:9000`). Overrides apply after the file is read and before validation.
- `--check-config` validates, prints `config ok` to stdout and exits 0, or prints the error to stderr and exits 2, without discovering devices or binding.
- Exit codes: 0 clean shutdown; 2 invalid configuration or CLI usage; 1 any other startup failure (bind failure, explicitly configured GPU library failed to load).
- Environment: `RUST_LOG` overrides _logging.level_ when set.

### Configuration file (YAML)

Every key is optional except _model.path_; defaults are the TS §15 example values. Unknown keys are errors.

| Key                                  | Type           | Default               | Validation                                                            |
| ------------------------------------ | -------------- | --------------------- | --------------------------------------------------------------------- |
| _server.listen_                      | socket address | `0.0.0.0:8000`        | parses as `SocketAddr`                                                |
| _server.max_request_bytes_           | byte size      | `8MiB`                | 1 KiB ≤ value ≤ 256 MiB                                               |
| _model.path_                         | string         | — (required)          | non-empty; existence is not checked in Phase 0                        |
| _model.dtype_                        | enum           | `bf16`                | only `bf16` is accepted                                               |
| _kv.block_tokens_                    | integer        | `16`                  | 1 ≤ value ≤ 1024                                                      |
| `kv.gpu.enabled`                     | bool           | `true`                | —                                                                     |
| `kv.cpu.enabled`                     | bool           | `true`                | —                                                                     |
| `kv.cpu.max_bytes`                   | byte size      | `64GiB`               | > 0 when `kv.cpu.enabled`                                             |
| `kv.nvme.enabled`                    | bool           | `false`               | —                                                                     |
| `kv.nvme.path`                       | string         | `/var/lib/turbine/kv` | absolute path when `kv.nvme.enabled`                                  |
| _reliability.enabled_                | bool           | `true`                | —                                                                     |
| _reliability.emergency_vram_reserve_ | byte size      | `2GiB`                | —                                                                     |
| _reliability.adaptive_admission_     | bool           | `true`                | —                                                                     |
| _scheduler.continuous_batching_      | bool           | `true`                | —                                                                     |
| _scheduler.chunked_prefill_          | bool           | `true`                | —                                                                     |
| _distributed.enabled_                | bool           | `false`               | `true` is rejected: "distributed mode is not supported in this build" |
| _logging.format_                     | enum           | `text`                | `text` or `json`                                                      |
| _logging.level_                      | string         | `info`                | valid `tracing` `EnvFilter` directive                                 |
| _devices.nvml_library_               | path or null   | null                  | when set, failure to load is fatal (exit 1)                           |
| _devices.amd_smi_library_            | path or null   | null                  | when set, failure to load is fatal (exit 1)                           |

Byte sizes accept a non-negative integer (bytes) or a string `<integer><unit>` with unit one of `B`, `KB`, `MB`, `GB`, `TB` (powers of 1000) or `KiB`, `MiB`, `GiB`, `TiB` (powers of 1024), no space, case-sensitive. Anything else is an error naming the key.

### HTTP routes

All error bodies use the OpenAI shape `{"error":{"message":"<text>","type":"<type>","code":"<code>"}}`.

| Route                                           | Phase 0 response                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `GET /health`                                   | `200` `{"status":"ok"}`                                                                         |
| `GET /ready`                                    | `503` `{"ready":false,"reason":"no_model_loaded"}`                                              |
| `GET /metrics`                                  | `200`, `content-type: application/openmetrics-text; version=1.0.0; charset=utf-8`               |
| `GET /v1/models`                                | `200` `{"object":"list","data":[]}`                                                             |
| `POST /v1/chat/completions`                     | `503`, type `service_unavailable`, code `model_not_loaded`                                      |
| `POST /v1/completions`                          | `503`, type `service_unavailable`, code `model_not_loaded`                                      |
| `GET /turbine/v1/status`                        | `200` `{"version":"<crate version>","uptime_seconds":<u64>,"ready":false,"device_count":<u64>}` |
| `GET /turbine/v1/devices`                       | `200` device inventory (see Data)                                                               |
| `GET /turbine/v1/kv`, `/pressure`, `/scheduler` | `501`, type `not_implemented`, code `not_implemented`                                           |
| any other path                                  | `404`, type `not_found`, code `not_found`                                                       |
| body over _server.max_request_bytes_            | `413`, type `invalid_request_error`, code `request_too_large`                                   |

Every response carries `x-request-id`: the request's own value if it sent one (≤ 128 visible ASCII characters), otherwise a generated UUIDv4.

### Metrics

- `turbine_http_requests_total{method,route,status}` counter — `route` is the matched route template, or `unmatched`.
- `turbine_http_request_duration_seconds{method,route}` histogram.
- `turbine_build_info{version}` gauge, always 1.
- `turbine_devices{vendor}` gauge — devices discovered per vendor (`nvidia`, `amd`).

### `turbine-bench`

```
turbine-bench --url <base-url> [--model <name>] [--endpoint chat|completions]
              [--concurrency <n>] [--requests <n>] [--prompt-words <n>]
              [--max-tokens <n>] [--seed <u64>] [--ignore-eos] [--output text|json]
```

- Defaults: `--endpoint chat`, `--concurrency 1`, `--requests 10`, `--prompt-words 256`, `--max-tokens 128`, `--seed 0`, `--output text`. Without `--model`, the first id from `GET <url>/v1/models` is used; an empty list is an error (exit 2).
- Every request sets `"stream": true` and `"stream_options": {"include_usage": true}`; `--ignore-eos` adds `"ignore_eos": true` (honoured by vLLM/SGLang for fixed-length outputs).
- Prompts: `--prompt-words` words drawn from a built-in 1,000-word list by a PRNG seeded with `--seed` + request index, so the same seed yields byte-identical prompts on every run.
- Measurements per request: TTFT = send → first SSE chunk carrying non-empty content; ITL = gap between consecutive content-bearing chunks; E2E = send → `[DONE]`; output tokens = _usage.completion_tokens_ from the final chunk, else the number of content-bearing chunks.
- Report: requests ok/failed, wall time, request throughput (ok requests / wall s), output-token throughput (tokens / wall s), and p50/p95/p99 for TTFT, ITL and E2E in milliseconds. JSON report keys: `requests_ok`, `requests_failed`, `wall_seconds`, `request_throughput`, `output_token_throughput`, `ttft_ms`, `itl_ms`, `e2e_ms` (each `{"p50":..,"p95":..,"p99":..}`).
- Exit codes: 0 when at least one request succeeded; 1 when every request failed; 2 usage errors.

### `scripts/lab-test.sh`

```
scripts/lab-test.sh <dgx-spark|dgx-spark2|novanas>
```

- Syncs the working tree (excluding `target/` and `.git/`) with `rsync` to `/home/piwi/turbine-ci/src` on the host.
- DGX Spark: `docker run --rm --gpus all` of `rust:1.97-trixie` with the tree at `/src`, a named volume `turbine-cargo` for the Cargo registry and `/home/piwi/turbine-ci/target` for build output, running `cargo test --workspace -- --include-ignored` with `TURBINE_EXPECT_NVIDIA=1`.
- novanas: `kubectl apply` of `scripts/lab/novanas-test-job.yaml` into namespace `turbine-ci` (created if absent) — image `rust:1.97-trixie`, `resources.limits: {amd.com/gpu: 2}`, hostPath mounts for `/home/piwi/turbine-ci` and `/opt/rocm/rocm` (read-only), `LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib`, `TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0`, `TURBINE_EXPECT_AMD=2`, `backoffLimit: 0`, `ttlSecondsAfterFinished: 600`. Waits for completion (timeout 30 min), streams the pod log, exits with the Job's result.
- Exit code: the test command's exit code; non-zero with a message when SSH, rsync, Docker or kubectl fails.

## Data

- Nothing is persisted by `turbine-server` in Phase 0. Configuration is read once at startup; the device inventory is captured once at startup and held in memory; metrics are in-process.
- Device inventory JSON (`GET /turbine/v1/devices`):

```json
{
  "devices": [
    {
      "index": 0,
      "vendor": "nvidia",
      "vendor_index": 0,
      "name": "NVIDIA GB10",
      "uuid": "GPU-…",
      "pci_bus_id": "0000:0f:01.0",
      "arch": "sm_121",
      "driver_version": "580.173.02",
      "memory": {
        "kind": "unified",
        "total_bytes": 129922002944,
        "shared_with_host": true
      }
    }
  ],
  "backends": [
    { "vendor": "nvidia", "status": "ok", "detail": "1 device(s)" },
    {
      "vendor": "amd",
      "status": "unavailable",
      "detail": "libamd_smi.so: cannot open shared object file"
    }
  ]
}
```

- `index` is global and stable for the process lifetime: NVIDIA devices first in NVML order, then AMD in amd-smi order. `arch` is `sm_<major><minor>` for NVIDIA and `gfx<target>` for AMD; `pci_bus_id`, `uuid`, `arch` and `driver_version` are `null` when the backend cannot supply them. `memory.kind` is `dedicated` (total from the vendor library, `shared_with_host: false`) or `unified` (NVML reports memory info as not supported; total is host `MemTotal` from `/proc/meminfo`, `shared_with_host: true`). `backends[].status` is `ok`, `unavailable` (library not found or init failed) or `timeout`.
- Bench results are written only to stdout.

## Edge cases

- Config file missing, unreadable, not YAML, empty, or containing an unknown key at any depth.
- `model.path` absent.
- Byte sizes: `64GiB`, `64 GiB` (rejected: space), `64gib` (rejected: case), `-1`, `1.5GiB` (rejected: non-integer), a bare integer, a value overflowing `u64`.
- `--set` with no `=`, with an unknown key, or with a value of the wrong type.
- `kv.cpu.enabled: true` with `kv.cpu.max_bytes: 0`; `kv.nvme.enabled: true` with a relative path.
- `server.listen` port already in use.
- macOS or any machine without GPU libraries: both backends `unavailable`, zero devices, server starts.
- NVML present but reporting memory info as not supported (GB10 unified memory).
- NVML loads but a per-device query fails: that field is `null`; the device is still listed.
- amd-smi installed at a non-standard prefix (`/opt/rocm/rocm/lib` on novanas).
- GPU library call hangs (bad driver state).
- Request body larger than `server.max_request_bytes`; client-supplied `x-request-id` longer than 128 characters or containing non-visible characters (replaced by a generated id).
- SIGTERM while a request is in flight.
- Bench: endpoint returns non-2xx; stream ends without `[DONE]`; chunks with empty content (role-only first chunk); no `usage` in the stream; `/v1/models` empty.

## Failure modes

- **Config invalid:** exit 2 before device discovery or binding; stderr names the dotted key path and the reason.
- **Port bind fails:** exit 1; stderr names the address and the OS error.
- **GPU library absent (default search):** that backend is `unavailable`, logged at WARN with the loader error; the server starts.
- **GPU library absent at an explicitly configured path (`devices.*_library` or `TURBINE_AMD_SMI_LIBRARY`):** exit 1 naming the path and the loader error — the operator asked for it explicitly.
- **GPU library call hangs:** each backend's discovery runs on a blocking thread with a 10-second deadline; on expiry the backend is `timeout`, logged at WARN, the server starts without that vendor's devices.
- **Metrics rendering fails:** `/metrics` returns 500 and logs the error; other routes are unaffected.
- **Bench target unreachable or failing:** each failed request is counted in `requests_failed` with its error logged; exit 1 only when every request failed.
- **Lab host unreachable, Docker/kubectl missing or failing:** `lab-test.sh` exits non-zero naming the step that failed; it never modifies other containers or workloads on the host.

## Acceptance criteria

- [ ] [S-1] [S-8] `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` all exit 0 on macOS arm64 with no GPU libraries installed; fails if any crate links a GPU library at build time or a test requires a GPU without `#[ignore]`.
- [ ] [S-1] `grep -rn "unsafe" crates/*/src benches/*/src` finds matches only under `crates/turbine-device/src`, each `unsafe` block preceded by a `// SAFETY:` comment, and `cargo build --workspace` exits 0 while `cargo build` fails if `unsafe` is added to another crate; fails if the `unsafe_code = "forbid"` lint is removed from any other crate's manifest.
- [ ] [S-2] `cargo test -p turbine-core config::tests::example_config_loads` exits 0; it parses the TS §15 example verbatim and asserts every field equals the documented value; fails if a key is renamed or a default changes.
- [ ] [S-2] `cargo test -p turbine-core config::tests::byte_size_parsing` exits 0; it asserts `64GiB` = 68719476736, `8MiB` = 8388608, `1000` = 1000, `2GB` = 2000000000, and that `64 GiB`, `64gib`, `-1`, `1.5GiB` and `99999999999TiB` are rejected with the key path in the message; fails if any unit or rejection rule changes.
- [ ] [S-2] `cargo test -p turbine-core config::tests::impossible_configs_rejected` exits 0; it asserts an error naming the key for each of: unknown key `kv.cpu.max_byte`, missing `model.path`, `model.dtype: fp16`, `kv.block_tokens: 0`, `kv.block_tokens: 1025`, `kv.cpu.enabled: true` with `max_bytes: 0`, `kv.nvme.enabled: true` with `path: relative/kv`, `distributed.enabled: true`, `server.max_request_bytes: 512`; fails if any of these is accepted.
- [ ] [S-2] `cargo test -p turbine-core config::tests::set_overrides_apply` exits 0; it asserts `--set kv.cpu.enabled=false` and `--set server.listen=127.0.0.1:9000` change the loaded config, and that `--set nope=1` and `--set kv.block_tokens=abc` are rejected naming the key; fails if overrides are applied after validation or ignored.
- [ ] [S-2] `cargo test -p turbine-server --test server_cli invalid_config_exits_2_before_bind` exits 0; it runs the `turbine-server` binary with `kv.cpu.max_bytes: 64XB` and asserts exit code 2, stderr containing `kv.cpu.max_bytes`, and that the configured port is still free afterwards; fails if validation happens after binding.
- [ ] [S-3] `cargo test -p turbine-api --test api metrics_counts_requests` exits 0; it issues `GET /health` twice, then `GET /metrics`, and asserts status 200, the OpenMetrics content type, `turbine_http_requests_total{method="GET",route="/health",status="200"} 2`, and a `turbine_build_info` line; fails if the counter is not incremented or labels use raw paths.
- [ ] [S-3] `cargo test -p turbine-api --test api unmatched_route_label_is_bounded` exits 0; it requests 50 distinct unknown paths and asserts every one is counted under `route="unmatched"` with no raw path appearing in `/metrics`; fails if a raw path becomes a label value.
- [ ] [S-3] `cargo test -p turbine-api --test api request_id_echoed_or_generated` exits 0; it asserts a request with `x-request-id: abc-123` gets `abc-123` back, one without gets a valid UUIDv4, and one with a 200-character id gets a generated UUIDv4; fails if the header is missing from any response.
- [ ] [S-4] `cargo test -p turbine-device discovery::tests::no_libraries_means_empty_inventory` exits 0; it asserts discovery with default search returns zero devices and both backends `unavailable` with a non-empty `detail`; fails if a missing library panics or aborts startup.
- [ ] [S-4] `cargo test -p turbine-device discovery::tests::explicit_missing_library_is_fatal` exits 0; it asserts `devices.amd_smi_library: /nonexistent/libamd_smi.so` returns an error naming that path; fails if it silently degrades to `unavailable`.
- [ ] [S-4] `cargo test -p turbine-device discovery::tests::backend_timeout` exits 0; it injects a backend that sleeps 30 s and asserts discovery returns within 11 s with that backend `timeout`; fails if a hung library blocks startup.
- [ ] [S-4] `cargo test -p turbine-device discovery::tests::unified_memory_uses_host_total` exits 0; it feeds the NVML adapter a "memory info not supported" result and a `/proc/meminfo` fixture with `MemTotal: 126877932 kB`, and asserts `kind: unified`, `total_bytes: 129923002368`, `shared_with_host: true`; fails if unified memory is reported as 0 or as dedicated.
- [ ] [S-4] [S-7] `cargo test -p turbine-device --test lab inventory_matches_expectation -- --ignored` exits 0; it asserts the discovered NVIDIA count equals `TURBINE_EXPECT_NVIDIA` and the AMD count equals `TURBINE_EXPECT_AMD` (each skipped when unset); `scripts/lab-test.sh novanas` exits 0 with the log showing 2 AMD devices with `arch: gfx1201` and kind `dedicated`; fails if discovery finds the wrong count, vendor or memory kind on real hardware. (The Spark runs of this check — 1 NVIDIA GB10 of kind `unified` on each — moved to `phase-2b-nvidia`, decision 2026-09-25.)
- [ ] [S-5] `cargo test -p turbine-api --test api route_table_phase0` exits 0; it asserts the status code and body `code`/shape of every row in the Interfaces route table; fails if any route is unregistered or returns a different status.
- [ ] [S-5] `cargo test -p turbine-api --test api body_limit_413` exits 0; it posts `server.max_request_bytes + 1` bytes to `/v1/chat/completions` with the limit set to 1 KiB and asserts `413` with code `request_too_large`; fails if the body limit is not enforced.
- [ ] [S-5] `cargo test -p turbine-server --test server_cli sigterm_graceful_shutdown` exits 0; it starts the binary, opens a request, sends SIGTERM and asserts the in-flight request completes and the process exits 0 within 5 s; fails if shutdown drops the in-flight request or hangs.
- [ ] [S-5] `cargo test -p turbine-server --test server_cli port_in_use_exits_1` exits 0; it binds a port, starts the server on it, and asserts exit code 1 with the address in stderr; fails if the bind error is swallowed.
- [ ] [S-6] `cargo test -p turbine-bench --test bench mock_endpoint_measurements` exits 0; it runs the harness against an in-test SSE server that sends a role-only chunk, then 5 content chunks 20 ms apart after a 100 ms delay, then usage `completion_tokens: 5` and `[DONE]`, with `--concurrency 2 --requests 4`; asserts `requests_ok: 4`, TTFT p50 ≥ 100 ms, ITL p50 ≥ 20 ms, and output-token throughput computed from 20 tokens; fails if the role-only chunk counts as the first token or usage is ignored.
- [ ] [S-6] `cargo test -p turbine-bench prompt::tests::prompts_are_deterministic` exits 0; it asserts two prompt sets generated with `--seed 7` are byte-identical and one generated with `--seed 8` differs; fails if prompt generation stops being seeded.
- [ ] [S-6] `cargo test -p turbine-bench --test bench failures_counted` exits 0; it points the harness at an endpoint returning 500 for half the requests and asserts `requests_failed` equals that half and exit code 0; and at one returning 500 for all, asserting exit code 1; fails if failures are dropped from the report.
- [ ] [S-8] `AGENTS.md` "Commands" section lists the exact build, test, single-test, lint, format, `scripts/lab-test.sh`, `turbine-server` and `turbine-bench` commands, and each listed command exits 0 when run as written; fails if a documented command is wrong.

## Open questions

<!-- None: every decision is recorded in .procoder/ask/decisions.md and folded into the sections above. -->
