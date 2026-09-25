# phase-0-skeleton — implementation plan

Status: draft
Spec: .procoder/specs/phase-0-skeleton.md

## Goal

Stand up the Turbine Cargo workspace — configuration model, observability, GPU discovery, the V1 HTTP route shell, the `turbine-server` binary, the `turbine-bench` load generator and the lab test runner — so it builds, tests and lints on macOS without a GPU and proves device discovery on dgx-spark, dgx-spark2 and novanas.

## Architecture

Six crates follow the contract DAG: `turbine-core` (config + shared vocabulary) ← `turbine-observability` (tracing, Prometheus registry, request-id and HTTP-metrics tower layers) ← `turbine-device` (runtime-loaded NVML and amd-smi discovery; the only crate allowed `unsafe`) ← `turbine-api` (Axum router reaching the engine only through the `InferenceBackend` / `Diagnostics` / `Readiness` traits) ← `turbine-server` (CLI, startup order, exit codes, Phase 0 trait impls); `benches/turbine-bench` is an independent lib + binary. Startup is config (exit 2) → tracing → discovery with a 10 s deadline per backend (exit 1 only for explicitly configured libraries) → bind (exit 1) → serve until SIGINT/SIGTERM with graceful shutdown (exit 0). `scripts/lab-test.sh` rsyncs the tree and runs the whole suite including `#[ignore]` GPU tests in `rust:1.97-trixie` via Docker on the Sparks and a k3s Job on novanas.

## Constraints

Copied verbatim from the spec (Constraints):

- Rust only; no Python in the build or runtime path (TS §21 rule 4).
- Libraries: Tokio, Axum 0.8, Serde, `serde_norway` (YAML; `serde_yaml` is archived), `tracing` + `tracing-subscriber`, `prometheus-client`, `thiserror`, `clap` 4, `nvml-wrapper`, `libloading`, `reqwest` (bench), `tower-http`. No other runtime dependency without a recorded reason.
- The workspace must build and every non-ignored test must pass on macOS arm64 with no GPU and no GPU libraries installed.
- GPU libraries are loaded at runtime only; nothing links against CUDA, NVML or ROCm at build time, so one source tree builds on macOS, arm64 Linux and amd64 Linux.
- `unsafe` code and FFI exist only in `turbine-device`, each `unsafe` block carrying a `// SAFETY:` comment stating the ownership/lifetime rule it relies on (TS §21 rule 10).
- Every queue and buffer is bounded (TS §21 rule 8): request bodies are limited by _server.max_request_bytes_; the bench's in-flight requests are limited by `--concurrency`.
- Metric label values are drawn from bounded sets (route templates, not raw paths) so cardinality cannot grow with traffic.
- Every automatic decision is explained in structured logs (TS §14): each discovery backend logs its outcome and reason.
- Lab hosts: `dgx-spark` (192.168.10.246) and `dgx-spark2` (192.168.10.245) are Ubuntu 24.04 arm64 with Docker 29 and the `nvidia` runtime; `novanas` (192.168.10.203) is Debian 13 amd64, k3s v1.35 with `amdgpu-device-plugin`, ROCm 7.14.1 at `/opt/rocm/rocm`. All reached as user `piwi` over SSH with the existing key. The Sparks also serve production vLLM on :8000/:8890/:8891; lab runs must not stop or restart those containers.

Contract rules every task inherits (`.procoder/contract/interfaces.md` §0, §1.3, §1.4, §20.1, §21):

- Edition 2024, `rust-version = "1.97"`, `license = "Apache-2.0"` in `[workspace.package]`, inherited by every crate.
- Every public enum a later phase extends is `#[non_exhaustive]`; every config struct is `#[serde(deny_unknown_fields, default)]`; every crate has one top-level error enum deriving `thiserror::Error`.
- Every crate except `turbine-device` sets `[lints.rust] unsafe_code = "forbid"`.
- Every metric label value comes from a closed set; no request id, raw path or address is ever a label.
- Unit tests live in `#[cfg(test)] mod tests` (addressed `cargo test -p <crate> <module>::tests::<name>`); integration tests are `crates/<crate>/tests/<binary>.rs` (addressed `cargo test -p <crate> --test <binary> <name>`); anything needing a GPU or lab host is `#[ignore]` and runs only via `scripts/lab-test.sh <host>`.
- Lab hosts: never stop, restart or reconfigure non-`turbine-lab-*` workloads; any run needing production workloads moved or GPUs/memory freed is asked of the user first; no `docker run` outside `scripts/lab-test.sh`.
- Pinned versions (checked against crates.io on 2026-09-25): axum 0.8.9, tokio 1.53.1 (features io-util, macros, net, rt-multi-thread, signal, sync, time), serde 1.0.229, serde_json 1.0.151, serde_norway 0.9.42, tracing 0.1.44, tracing-subscriber 0.3.23 (env-filter, fmt, json), prometheus-client 0.25.1, thiserror 2.0.21, clap 4.6.7 (derive), nvml-wrapper 0.13.0, libloading 0.9.0, reqwest 0.13.5 (default-features off, no TLS), tower 0.5.3 (util), uuid 1.26.1 (v4).

Decisions taken where spec/contract are silent (reported as contract additions): `turbine_core::request` is created in P0 holding only `Endpoint` and `ErrorCode`; `InferenceBackend` carries only `models()` in P0 (P1 adds `submit`); a wrong method on a known route answers 405 `method_not_allowed`; `ObservabilityError::Init`; `examples/turbine.yaml`; lab-only env vars `TURBINE_EXPECT_NVIDIA_MEMORY` and `TURBINE_EXPECT_AMD_ARCH`; `--show-output` on the lab test command; the novanas Job also mounts `/etc/alternatives/rocm-lib` (on novanas `/opt/rocm/rocm/lib` is a symlink into `/etc/alternatives`). `tower-http` is not needed in P0 (Axum `DefaultBodyLimit` enforces the body limit). `turbine-bench` accepts `http://` URLs only (reqwest built without TLS) and rejects others with exit 2.

## Task 1: Workspace root and `turbine-core` configuration model

Files:

- `Cargo.toml` — workspace root: `resolver = "3"`, `members = ["crates/*"]`, `[workspace.package]`, every P0 dependency version in `[workspace.dependencies]`.
- `Cargo.lock` — generated, committed (the workspace ships binaries).
- `LICENSE` — canonical Apache-2.0 text (sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
- `.gitignore` — add `/target/`.
- `crates/turbine-core/Cargo.toml` — manifest (serde, serde_norway, thiserror, tracing-subscriber), `unsafe_code = "forbid"`.
- `crates/turbine-core/src/lib.rs` — `pub mod config; pub mod request; pub mod types;`.
- `crates/turbine-core/src/types.rs` — `DeviceId`, `Vendor`, `MemoryKind`.
- `crates/turbine-core/src/request.rs` — `Endpoint`, `ErrorCode`.
- `crates/turbine-core/src/config/byte_size.rs` — `ByteSize` parsing and serde.
- `crates/turbine-core/src/config/overrides.rs` — `Override` and YAML value-tree helpers (set dotted path, first unknown key, leaves).
- `crates/turbine-core/src/config/mod.rs` — section structs, `load`, key-naming deserialization, `Config::validate`.
- `crates/turbine-core/src/config/tests.rs` — the four acceptance tests.

Interfaces:

- `pub fn load(path: &Path, overrides: &[Override]) -> Result<Config, ConfigError>`
- `fn load_from_str(text: &str, origin: &Path, overrides: &[Override]) -> Result<Config, ConfigError>` (private; used by tests)
- `pub struct Override { pub key: String, pub value: serde_norway::Value }` + `impl FromStr for Override { type Err = ConfigError; }`
- `pub enum ConfigError { Io { path: PathBuf, source: std::io::Error }, Syntax { path: PathBuf, detail: String }, UnknownKey { key: String }, Invalid { key: String, reason: String }, BadOverride { arg: String, reason: String } }` + `pub fn key(&self) -> Option<&str>`
- `pub struct Config { pub server: ServerConfig, pub model: ModelConfig, pub kv: KvConfig, pub reliability: ReliabilityConfig, pub scheduler: SchedulerConfig, pub distributed: DistributedConfig, pub logging: LoggingConfig, pub devices: DevicesConfig }` + `pub fn validate(&self) -> Result<(), ConfigError>`
- `KvConfig { block_tokens: u32, gpu: KvGpuConfig, cpu: KvCpuConfig, nvme: KvNvmeConfig }`; `ModelConfig { path: PathBuf, dtype: ModelDtype }`; `ModelDtype { Bf16 }`; `LogFormat { Text, Json }`; `DevicesConfig { nvml_library: Option<PathBuf>, amd_smi_library: Option<PathBuf> }`
- `pub struct ByteSize(pub u64)` + `impl FromStr<Err = String>`, `Display`, `Serialize` (as integer), `Deserialize` (int or string), `pub const fn kib/mib/gib(n: u64) -> ByteSize`
- `pub struct DeviceId(pub u32)`; `pub enum Vendor { Nvidia, Amd }` + `as_str()`; `pub enum MemoryKind { Dedicated, Unified }`
- `pub enum Endpoint { Completions, ChatCompletions }`; `pub enum ErrorCode { ModelNotLoaded, NotImplemented, NotFound, RequestTooLarge, MethodNotAllowed, InternalError }` + `as_str()`

Covers: S-1 (workspace, edition, rust-version, license, `LICENSE`), S-2; `config::tests::example_config_loads`, `config::tests::byte_size_parsing`, `config::tests::impossible_configs_rejected`, `config::tests::set_overrides_apply`.
Depends on: nothing.

- [ ] Create the workspace root, `LICENSE` (`curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE`, then check the sha256 above), `.gitignore`, and the turbine-core manifest, `lib.rs`, `types.rs`, `request.rs`; `config/mod.rs` holds only `#[cfg(test)] mod tests;`.
- [ ] Write failing test `config::tests::example_config_loads`: parses the TS §15 YAML verbatim and asserts every field (listen `0.0.0.0:8000`, max_request_bytes 8388608, path `/models/qwen`, dtype bf16, block_tokens 16, cpu.max_bytes 68719476736, nvme.path `/var/lib/turbine/kv`, emergency_vram_reserve 2147483648, all booleans, logging text/info, devices null), and that a file with only `model.path` yields the same defaults.
- [ ] Write failing test `config::tests::byte_size_parsing`: `64GiB`=68719476736, `8MiB`=8388608, `1000`=1000, `2GB`=2000000000 both via `ByteSize::from_str` and via `kv.cpu.max_bytes`; `64 GiB`, `64gib`, `-1`, `1.5GiB`, `99999999999TiB` are rejected with `kv.cpu.max_bytes` in the message.
- [ ] Write failing test `config::tests::impossible_configs_rejected`: an error naming the key for unknown `kv.cpu.max_byte`, missing `model.path`, `model.dtype: fp16`, `kv.block_tokens` 0 and 1025, cpu enabled with `max_bytes: 0`, nvme enabled with `path: relative/kv`, `distributed.enabled: true` (message contains "distributed mode is not supported in this build"), `server.max_request_bytes: 512`.
- [ ] Write failing test `config::tests::set_overrides_apply`: `kv.cpu.enabled=false` and `server.listen=127.0.0.1:9000` change the config; a file with `block_tokens: 0` plus `--set kv.block_tokens=32` loads (overrides before validation); `nope=1` and `kv.block_tokens=abc` are rejected naming the key; `no-equals-sign` is `BadOverride`.
- [ ] Run: `cargo test -p turbine-core config::tests` — expect FAIL (`error[E0425]: cannot find type `Config` in this scope`).
- [ ] Implement `ByteSize`: split leading ASCII digits from the unit; unit empty or one of `B KB MB GB TB KiB MiB GiB TiB` (case-sensitive, no space); `u64::checked_mul` for overflow; the serde visitor accepts `u64`, rejects negative `i64` and any `f64`.
- [ ] Implement loading: parse YAML to `serde_norway::Value` (null → empty mapping, non-mapping root → `Syntax`), apply each `Override` with `set_path`, detect unknown keys by walking the user tree against `serde_norway::to_value(Config::default())`, then `from_value::<Config>`; on failure name the key by re-deserializing the defaults with one user leaf at a time and reporting the first leaf that fails (`serde_norway` puts the path only in `from_str` Display, never on `from_value`).
- [ ] Implement `Config::validate` rules from the spec key table: 1 KiB ≤ max_request_bytes ≤ 256 MiB, non-empty `model.path`, 1 ≤ block_tokens ≤ 1024, cpu max_bytes > 0 when enabled, absolute nvme path when enabled, `distributed.enabled` rejected, `logging.level` parsed with `tracing_subscriber::EnvFilter::try_new`.
- [ ] Run: `cargo test -p turbine-core config::tests` — expect PASS (4 passed).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(core): workspace root and validated configuration model`

## Task 2: `turbine-observability` — tracing, metrics registry, HTTP layers

Files:

- `crates/turbine-observability/Cargo.toml` — manifest (axum, prometheus-client, thiserror, tower, tracing, tracing-subscriber, turbine-core, uuid), `unsafe_code = "forbid"`.
- `crates/turbine-observability/src/lib.rs` — modules, re-exports, `ObservabilityError`.
- `crates/turbine-observability/src/tracing.rs` — `init_tracing`.
- `crates/turbine-observability/src/metrics.rs` — `MetricsRegistry`, `OPENMETRICS_CONTENT_TYPE`, `turbine_build_info`.
- `crates/turbine-observability/src/http.rs` — request-id layer, HTTP metrics layer, `HttpMetrics`, `RequestIdExt`.

Interfaces:

- `pub fn init_tracing(cfg: &LoggingConfig) -> Result<(), ObservabilityError>`
- `#[derive(Clone)] pub struct MetricsRegistry(Arc<Mutex<prometheus_client::registry::Registry>>)` + `pub fn new() -> Self`, `pub fn register<M: prometheus_client::registry::Metric + Clone>(&self, name: &str, help: &str, metric: M) -> M`, `pub fn render(&self) -> Result<String, ObservabilityError>`, `impl Default`
- `pub const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8"`
- `pub enum ObservabilityError { Filter(String), Render(String), Init(String) }`
- `pub mod http`: `pub fn request_id_layer() -> RequestIdLayer`, `pub fn http_metrics_layer(m: HttpMetrics) -> HttpMetricsLayer`, `impl HttpMetrics { pub fn register(reg: &MetricsRegistry) -> Self }`, `pub struct RequestIdExt(pub String)`, `pub const REQUEST_ID_HEADER: HeaderName`, `pub fn valid_request_id(value: &str) -> bool`
- Metrics: `turbine_http_requests_total{method,route,status}` counter (registered as `turbine_http_requests`; prometheus-client appends `_total`), `turbine_http_request_duration_seconds{method,route}` histogram (`exponential_buckets(0.001, 2.0, 16)`), `turbine_build_info{version} 1`.

Covers: S-3 (subscriber, registry, request-id, bounded labels; the S-3 acceptance tests exercise these through the router in Task 4).
Depends on: Task 1.

- [ ] Write failing test `metrics::tests::build_info_and_registered_metrics_render`: a new registry renders `turbine_build_info{version="0.1.0"} 1`, a registered counter incremented once renders `turbine_test_events_total 1`, output ends with `# EOF\n`.
- [ ] Write failing test `http::tests::request_id_validation`: `abc-123` and a 128-char id are valid; 129 chars, empty, containing a space, a tab or `é` are invalid; method `BREW` maps to label `OTHER`.
- [ ] Run: `cargo test -p turbine-observability` — expect FAIL (unresolved imports `metrics::MetricsRegistry`).
- [ ] Implement `init_tracing`: `RUST_LOG` (when set) else `cfg.level` into `EnvFilter::try_new`, `tracing_subscriber::fmt()` to stderr, `.json()` for `LogFormat::Json`, `try_init` errors → `Init`.
- [ ] Implement `MetricsRegistry`: `new()` registers a `Family<BuildInfoLabels{version}, Gauge>` set to 1 with `env!("CARGO_PKG_VERSION")`; `render` uses `prometheus_client::encoding::text::encode`.
- [ ] Implement the layers as generic tower `Layer`/`Service` over `axum::http::Request<B>` with boxed futures: the request-id service keeps a client `x-request-id` when 1..=128 bytes all in 0x21..=0x7E, else `uuid::Uuid::new_v4()`, inserts `RequestIdExt`, instruments the call with `info_span!("http_request", request_id, method, path)` and sets the response header; the metrics service reads `axum::extract::MatchedPath` from request extensions (absent → `unmatched`), maps the method to the closed set GET/POST/PUT/DELETE/HEAD/OPTIONS/PATCH/OTHER and records status and duration after the response.
- [ ] Run: `cargo test -p turbine-observability` — expect PASS (2 passed).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(observability): tracing init, metrics registry and HTTP request-id/metrics layers`

## Task 3: `turbine-device` — NVML and amd-smi discovery with deadlines

Files:

- `crates/turbine-device/Cargo.toml` — manifest (libloading, nvml-wrapper, prometheus-client, serde, thiserror, tracing, turbine-core, turbine-observability; dev serde_json); `[lints.rust] unsafe_code = "allow"`.
- `crates/turbine-device/src/lib.rs` — re-exports.
- `crates/turbine-device/src/inventory.rs` — inventory types, `DiscoveryError`, `DeviceMetrics`.
- `crates/turbine-device/src/discovery/mod.rs` — options, backend trait, `discover`, `run_backends`, explicit-path check, outcome logging.
- `crates/turbine-device/src/discovery/nvml.rs` — NVML backend, unified-memory mapping, bus-id normalisation.
- `crates/turbine-device/src/discovery/amd_smi.rs` — `libloading` FFI to amd-smi, layout and format unit tests.
- `crates/turbine-device/src/discovery/tests.rs` — the four acceptance tests.
- `crates/turbine-device/tests/fixtures/proc/meminfo-gb10` — `/proc/meminfo` fixture whose first line is `MemTotal:       126877932 kB`.

Interfaces:

- `pub struct DiscoveryOptions { pub nvml_library: Option<PathBuf>, pub amd_smi_library: Option<PathBuf>, pub deadline: Duration, pub meminfo_path: PathBuf }` + `Default` (None, None, 10 s, `/proc/meminfo`) + `pub fn from_config(cfg: &DevicesConfig) -> Self` (null amd path falls back to env `TURBINE_AMD_SMI_LIBRARY`, `pub const AMD_SMI_LIBRARY_ENV`)
- `pub fn discover(opts: &DiscoveryOptions) -> Result<DeviceInventory, DiscoveryError>`
- `fn discover_with_defaults(opts: &DiscoveryOptions, nvml_default: &str, amd_smi_default: &str) -> Result<DeviceInventory, DiscoveryError>` (defaults `libnvidia-ml.so.1`, `libamd_smi.so`)
- `pub trait DiscoveryBackend: Send { fn vendor(&self) -> Vendor; fn discover(&mut self) -> Result<Vec<DeviceInfo>, String>; }`
- `pub fn run_backends(backends: Vec<Box<dyn DiscoveryBackend>>, deadline: Duration) -> DeviceInventory`
- `#[derive(Serialize, Clone, Debug)] pub struct DeviceInventory { pub devices: Vec<DeviceInfo>, pub backends: Vec<BackendReport> }` + `pub fn count(&self, vendor: Vendor) -> usize`
- `pub struct DeviceInfo { pub index: DeviceId, pub vendor: Vendor, pub vendor_index: u32, pub name: String, pub uuid: Option<String>, pub pci_bus_id: Option<String>, pub arch: Option<String>, pub driver_version: Option<String>, pub memory: DeviceMemoryInfo }`
- `pub struct DeviceMemoryInfo { pub kind: MemoryKind, pub total_bytes: u64, pub shared_with_host: bool }`; `pub struct BackendReport { pub vendor: Vendor, pub status: BackendStatus, pub detail: String }`; `pub enum BackendStatus { Ok, Unavailable, Timeout }` (serde lowercase)
- `pub enum DiscoveryError { ExplicitLibrary { path: PathBuf, detail: String } }` — Display `cannot load <path>: <detail>`
- `pub struct DeviceMetrics` + `pub fn register(reg: &MetricsRegistry) -> Self`, `pub fn record(&self, inventory: &DeviceInventory)` (`turbine_devices{vendor}` for both vendors)
- `pub(crate) fn nvml::memory_info(total: Result<u64, nvml_wrapper::error::NvmlError>, meminfo_path: &Path) -> DeviceMemoryInfo`
- nvml-wrapper 0.13: `Nvml::builder().lib_path(&OsStr).init()`, `sys_driver_version()`, `device_count()`, `device_by_index(u32)`, `Device::{name, uuid, pci_info().bus_id, cuda_compute_capability() -> {major, minor}, memory_info().total}`, `NvmlError::NotSupported`
- amd-smi (ROCm 7.14.1 `amdsmi.h`, lib 26.5.0): `amdsmi_init(uint64_t)` with `AMDSMI_INIT_AMD_GPUS = 1 << 1`; `amdsmi_shut_down(void)`; `amdsmi_get_socket_handles(uint32_t*, amdsmi_socket_handle*)`; `amdsmi_get_processor_handles(amdsmi_socket_handle, uint32_t*, amdsmi_processor_handle*)` (NULL array → count); `amdsmi_get_processor_type(handle, amdsmi_processor_type_t*)` with `AMDSMI_PROCESSOR_TYPE_AMD_GPU = 1`; `amdsmi_get_gpu_asic_info(handle, amdsmi_asic_info_t*)` (896 bytes: `market_name[256]` @0, `vendor_id` @256, `device_id` u64 @520, `target_graphics_version` u64 @800, `reserved[18]` u32); `amdsmi_get_gpu_device_uuid(handle, unsigned int*, char*)` with `AMDSMI_GPU_UUID_SIZE 38`; `amdsmi_get_gpu_device_bdf(handle, amdsmi_bdf_t*)` (u64 bitfield: function 0–2, device 3–7, bus 8–15, domain 16–63); `amdsmi_get_gpu_memory_total(handle, AMDSMI_MEM_TYPE_VRAM = 0, uint64_t*)`; `amdsmi_get_gpu_driver_info(handle, amdsmi_driver_info_t*)` (3 × `char[256]`, 768 bytes); `AMDSMI_STATUS_SUCCESS = 0`; handles are `void*`.

Covers: S-4; `discovery::tests::no_libraries_means_empty_inventory`, `discovery::tests::explicit_missing_library_is_fatal`, `discovery::tests::backend_timeout`, `discovery::tests::unified_memory_uses_host_total`.
Depends on: Tasks 1, 2.

- [ ] Write failing test `discovery::tests::no_libraries_means_empty_inventory`: `discover_with_defaults(&DiscoveryOptions::default(), "libturbine-test-missing-nvml.so", "libturbine-test-missing-amd-smi.so")` is `Ok`, zero devices, backends `[nvidia, amd]` both `Unavailable` with non-empty detail (unfindable names keep the test green on lab hosts that have the libraries).
- [ ] Write failing test `discovery::tests::explicit_missing_library_is_fatal`: `amd_smi_library: /nonexistent/libamd_smi.so` and, separately, `nvml_library: /nonexistent/libnvidia-ml.so.1` return `Err` whose message contains that path.
- [ ] Write failing test `discovery::tests::backend_timeout`: `run_backends` with an instant NVIDIA fake returning one device (index 99) and an AMD fake sleeping 30 s, deadline `DiscoveryOptions::default().deadline` (asserted = 10 s), returns in < 11 s with backends `[Ok, Timeout]` and the device re-indexed to `DeviceId(0)`.
- [ ] Write failing test `discovery::tests::unified_memory_uses_host_total`: `nvml::memory_info(Err(NvmlError::NotSupported), fixture)` gives kind `Unified`, `total_bytes` 129923002368, `shared_with_host` true (JSON `"kind":"unified"`); `Ok(34208743424)` gives `Dedicated`; `normalize_bus_id("00000000:0F:01.0")` = `0000:0f:01.0`.
- [ ] Run: `cargo test -p turbine-device --lib` — expect FAIL (unresolved imports `discovery::DiscoveryBackend`, …).
- [ ] Implement `run_backends`: one named `std::thread` per backend sending `(slot, result)` over `mpsc`; `recv_timeout(deadline − elapsed)` until all report or time runs out; missing slots are `Timeout` (thread left detached); global indices follow backend order; log `event="device_discovery"` with `vendor`, `status`, `detail` at INFO (ok) or WARN.
- [ ] Implement `discover`: for each explicit path, `libloading::Library::new` then drop (fatal `ExplicitLibrary` with libloading's message plus its `source()` dlerror text); then run the NVML backend then the amd-smi backend with explicit-or-default library names.
- [ ] Implement the NVML backend: init via `lib_path`, per device fill each field from its own query (`None` on failure, device still listed), `arch = sm_{major}{minor}`, bus id normalised to 4-hex-digit lowercase domain; `memory_info`: `Ok(total)` → dedicated; `NotSupported` → unified with `MemTotal` kB × 1024 from `meminfo_path`; other errors → dedicated 0 with WARN.
- [ ] Implement the amd-smi backend: `#[repr(C)]` structs matching the layouts above, fn-pointer types resolved with `Library::get` and owned with the `Library` in one struct; enumerate sockets → processors (two-call count/fill), keep `AMD_GPU` handles, query asic info (name = `market_name`, arch = `format!("gfx{:x}", target_graphics_version)` unless 0 or `u64::MAX`, matching `amd-smi static` output `gfx1201`), uuid, bdf (`{domain:04x}:{bus:02x}:{device:02x}.{function:x}`), VRAM total, driver version (`""`/`N/A` → null); kind always `Dedicated`; `amdsmi_shut_down` after, then `std::mem::forget` the library (no dlclose of ROCm). Every `unsafe` block gets its own `// SAFETY:` comment; add unit tests `amd_smi::tests::struct_layouts_match_amdsmi_h` (896/256/520/800/768 via `size_of`/`offset_of!`) and `amd_smi::tests::bdf_and_arch_format` (`0x0300` → `0000:03:00.0`, `0x1201` → `gfx1201`).
- [ ] Run: `cargo test -p turbine-device --lib` — expect PASS (6 passed; `backend_timeout` takes ~10 s).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(device): runtime-loaded NVML and amd-smi discovery with per-backend deadline`

## Task 4: `turbine-api` — V1 route shell, OpenAI errors, body limit

Files:

- `crates/turbine-api/Cargo.toml` — manifest (axum, serde, serde_json, tracing, turbine-core, turbine-observability; dev tokio, tower, uuid), `unsafe_code = "forbid"`.
- `crates/turbine-api/src/lib.rs` — re-exports (including `turbine_core::request::ErrorCode`).
- `crates/turbine-api/src/error.rs` — `ApiError`, `ErrorType`, OpenAI error body.
- `crates/turbine-api/src/backend.rs` — `ApiState`, `ApiLimits`, traits, `ReadyState`, `NotReadyReason`, `ModelCard`.
- `crates/turbine-api/src/routes/mod.rs` — `router()`: routes, 404/405 fallbacks, body limit, layers.
- `crates/turbine-api/src/routes/openai.rs` — health, ready, metrics, models, completion handlers.
- `crates/turbine-api/src/routes/diagnostics.rs` — `/turbine/v1/*` handlers.
- `crates/turbine-api/tests/api.rs` — the five acceptance tests.

Interfaces:

- `pub fn router(state: ApiState) -> axum::Router`
- `#[derive(Clone)] pub struct ApiState { pub inference: Arc<dyn InferenceBackend>, pub diagnostics: Arc<dyn Diagnostics>, pub readiness: Arc<dyn Readiness>, pub metrics: MetricsRegistry, pub limits: ApiLimits }`; `pub struct ApiLimits { pub max_request_bytes: usize }`
- `pub trait InferenceBackend: Send + Sync { fn models(&self) -> Vec<ModelCard>; }`
- `pub trait Diagnostics: Send + Sync { fn status(&self) -> serde_json::Value; fn devices(&self) -> serde_json::Value; fn scheduler(&self) -> Result<serde_json::Value, ApiError>; fn kv(&self) -> Result<serde_json::Value, ApiError>; fn pressure(&self) -> Result<serde_json::Value, ApiError>; }`
- `pub trait Readiness: Send + Sync { fn ready(&self) -> ReadyState; }`; `pub enum ReadyState { Ready, NotReady { reason: NotReadyReason } }`; `#[non_exhaustive] pub enum NotReadyReason { NoModelLoaded }` + `as_str()`
- `pub struct ModelCard { pub id: String, pub object: String, pub created: u64, pub owned_by: String, pub max_model_len: u32 }`
- `pub struct ApiError { pub status: StatusCode, pub kind: ErrorType, pub code: ErrorCode, pub message: String, pub retry_after: Option<u64> }` + `new`, `model_not_loaded()`, `not_implemented()`, `not_found(path: &str)`, `method_not_allowed(method: &str, path: &str)`, `request_too_large(limit: usize)`, `internal(message)`; `impl IntoResponse`
- `#[non_exhaustive] pub enum ErrorType { InvalidRequestError, RateLimitError, ServiceUnavailable, NotImplemented, NotFound, ServerError, Timeout }` (serde snake_case)

Covers: S-3, S-5; `api metrics_counts_requests`, `api unmatched_route_label_is_bounded`, `api request_id_echoed_or_generated`, `api route_table_phase0`, `api body_limit_413`.
Depends on: Tasks 1, 2.

- [ ] Write failing test `api route_table_phase0` (fakes: no models, NotReady `NoModelLoaded`, P0 diagnostics; requests via `tower::ServiceExt::oneshot`): `/health` 200 `{"status":"ok"}`; `/ready` 503 `{"ready":false,"reason":"no_model_loaded"}`; `/metrics` 200 with `OPENMETRICS_CONTENT_TYPE`; `/v1/models` 200 `{"object":"list","data":[]}`; both POST completion routes 503 `service_unavailable`/`model_not_loaded`; status has `version`, `uptime_seconds`, `ready`, `device_count`; devices has arrays `devices`, `backends`; kv/pressure/scheduler 501 `not_implemented`; `GET /v1/completions` 405 `method_not_allowed`; `/no/such/route` 404 `not_found`.
- [ ] Write failing test `api body_limit_413`: limit 1024, a 1025-byte POST to `/v1/chat/completions` → 413 `invalid_request_error`/`request_too_large`; exactly 1024 bytes → 503.
- [ ] Write failing test `api metrics_counts_requests`: two `GET /health` then `GET /metrics` → 200, OpenMetrics content type, line `turbine_http_requests_total{method="GET",route="/health",status="200"} 2`, a `turbine_build_info{version=` line and `turbine_http_request_duration_seconds_count{method="GET",route="/health"} 2`.
- [ ] Write failing test `api unmatched_route_label_is_bounded`: 50 GETs to `/unknown/path-<i>` → `turbine_http_requests_total{method="GET",route="unmatched",status="404"} 50` and no `/unknown/path-` text in `/metrics`.
- [ ] Write failing test `api request_id_echoed_or_generated`: `x-request-id: abc-123` echoed; none → UUID v4; 200 chars → UUID v4; 404, 503, 501 and 405 responses all carry the header.
- [ ] Run: `cargo test -p turbine-api --test api` — expect FAIL (unresolved imports `turbine_api::ApiError`, …).
- [ ] Implement the router: the eleven routes, `.fallback` (404) and `.method_not_allowed_fallback` (405), then `.layer(DefaultBodyLimit::max(limit))`, `.layer(http_metrics_layer(HttpMetrics::register(&state.metrics)))`, `.layer(request_id_layer())` — `Router::layer` runs after routing, so `MatchedPath` is visible to the metrics layer and the fallback has none.
- [ ] Implement handlers: completion handlers take `Result<Bytes, BytesRejection>` and return 413 when `rejection.status() == PAYLOAD_TOO_LARGE`, otherwise 503 `model_not_loaded` (P0 never parses bodies); `/metrics` returns `render()` with the OpenMetrics content type, or 500 `internal_error` with an ERROR log; diagnostics map `Result<Value, ApiError>` to JSON or the error.
- [ ] Run: `cargo test -p turbine-api --test api` — expect PASS (5 passed).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(api): V1 route shell with OpenAI errors, body limit, request ids and HTTP metrics`

## Task 5: `turbine-server` binary — CLI, startup order, exit codes, graceful shutdown

Files:

- `crates/turbine-server/Cargo.toml` — manifest (axum, clap, serde, serde_json, tokio, tracing, turbine-api, turbine-core, turbine-device, turbine-observability); binary auto-discovered from `src/main.rs`; `unsafe_code = "forbid"`.
- `crates/turbine-server/src/main.rs` — parse CLI, call `startup::run`.
- `crates/turbine-server/src/cli.rs` — `Cli` (clap derive; usage errors exit 2).
- `crates/turbine-server/src/exit.rs` — `ExitCode`.
- `crates/turbine-server/src/startup.rs` — startup order, bind, graceful shutdown, `NoModel`, `ServerDiagnostics`, `StatusDocument`.
- `crates/turbine-server/tests/server_cli.rs` — the three acceptance tests.

Interfaces:

- CLI: `turbine-server --config <path> [--set <dotted.key>=<yaml value>]... [--check-config]`
- `pub struct Cli { pub config: PathBuf, pub set: Vec<Override>, pub check_config: bool }`
- `pub enum ExitCode { Clean = 0, Startup = 1, Config = 2 }` + `impl From<ExitCode> for std::process::ExitCode` (P3 adds `DeviceFatal = 3`)
- `pub fn run(cli: Cli) -> ExitCode`; `async fn serve(config: Config, inventory: DeviceInventory) -> ExitCode`; `async fn shutdown_signal()`
- `struct StatusDocument { version: &'static str, uptime_seconds: u64, ready: bool, device_count: u64 }`
- Stderr: `turbine-server: invalid configuration: <ConfigError>` (exit 2); `turbine-server: device discovery failed: <DiscoveryError>` (exit 1); `turbine-server: cannot bind <addr>: <os error>` (exit 1); stdout `config ok` for `--check-config`.
- Consumes `config::load`, `init_tracing`, `MetricsRegistry`, `discover`, `DiscoveryOptions::from_config`, `DeviceMetrics`, `router`, `ApiState` (Tasks 1–4).

Covers: S-2 (reject before bind, `--set`, `--check-config`, `RUST_LOG`), S-5 (graceful shutdown); `server_cli invalid_config_exits_2_before_bind`, `server_cli sigterm_graceful_shutdown`, `server_cli port_in_use_exits_1`.
Depends on: Tasks 1–4.

- [ ] Write failing test `server_cli invalid_config_exits_2_before_bind`: config with `server.listen: 127.0.0.1:<free port>` and `kv.cpu.max_bytes: 64XB` → exit 2, stderr contains `kv.cpu.max_bytes`, the port binds afterwards; `--check-config` on it → exit 2; `--check-config` on `model.path: /m` → exit 0 and stdout `config ok`.
- [ ] Write failing test `server_cli port_in_use_exits_1`: hold a `127.0.0.1:0` listener, start the server on that address → exit 1 and the address in stderr (process killed after 20 s if it never exits).
- [ ] Write failing test `server_cli sigterm_graceful_shutdown`: start on a free port, poll `GET /health` until 200, send POST `/v1/completions` headers with `Content-Length` 30 and only 10 body bytes, `kill -TERM <pid>`, send the remaining 20 bytes → the response is `HTTP/1.1 503` containing `model_not_loaded`, and the process exits 0 within 5 s.
- [ ] Run: `cargo test -p turbine-server --test server_cli` with `src/main.rs` holding only an empty `main` — expect FAIL (`left: Some(0)`).
- [ ] Implement `run`: `config::load` (error → stderr, `Config`), `--check-config` → `config ok`/`Clean`, `init_tracing`, `turbine_device::discover(&DiscoveryOptions::from_config(&config.devices))` on the main thread (error → `Startup`), then a multi-thread tokio runtime.
- [ ] Implement `serve`: `DeviceMetrics::register(&metrics).record(&inventory)`, build `ApiState` (limit = `server.max_request_bytes`), `tokio::net::TcpListener::bind` (error → `Startup`), `axum::serve(listener, router).with_graceful_shutdown(shutdown_signal())`; `shutdown_signal` selects `tokio::signal::ctrl_c()` and `tokio::signal::unix::signal(SignalKind::terminate())`, logging which signal arrived.
- [ ] Run: `cargo test -p turbine-server --test server_cli` — expect PASS (3 passed).
- [ ] Run: `printf 'model:\n  path: /m\n' > /tmp/turbine-p0.yaml && cargo run -q -p turbine-server -- --config /tmp/turbine-p0.yaml --set devices.amd_smi_library=/nonexistent/libamd_smi.so; echo exit=$?` — expect stderr `turbine-server: device discovery failed: cannot load /nonexistent/libamd_smi.so: dlopen failed: …` and `exit=1`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(server): turbine-server CLI with validated startup order, exit codes and graceful shutdown`

## Task 6: `turbine-bench` library — seeded prompts and report aggregation

Files:

- `Cargo.toml` — `members = ["crates/*", "benches/turbine-bench"]`.
- `benches/turbine-bench/Cargo.toml` — manifest (clap, reqwest, serde, serde_json, thiserror, tokio), `unsafe_code = "forbid"`.
- `benches/turbine-bench/src/lib.rs` — `pub mod prompt; pub mod report;`.
- `benches/turbine-bench/src/prompt.rs` — built-in 1,000-word list and seeded prompts.
- `benches/turbine-bench/src/report.rs` — per-request stats, percentiles, report and text rendering.

Interfaces:

- `pub const WORD_COUNT: usize = 1000`; `pub fn word(i: usize) -> String`; `pub fn prompt(seed: u64, index: u64, words: u32) -> String`; `pub fn prompts(seed: u64, count: u32, words: u32) -> Vec<String>`
- `pub struct RequestStats { pub ttft: Duration, pub itls: Vec<Duration>, pub e2e: Duration, pub output_tokens: u64 }`
- `pub struct Percentiles { pub p50: f64, pub p95: f64, pub p99: f64 }`; `pub fn percentiles(values: Vec<f64>) -> Percentiles`
- `#[derive(Serialize)] pub struct Report { pub requests_ok: u64, pub requests_failed: u64, pub wall_seconds: f64, pub request_throughput: f64, pub output_token_throughput: f64, pub ttft_ms: Percentiles, pub itl_ms: Percentiles, pub e2e_ms: Percentiles }` + `pub fn from_results(ok: &[RequestStats], failed: u64, wall: Duration) -> Report`, `pub fn to_text(&self) -> String`

Covers: S-6 (seeded prompts, report keys); `prompt::tests::prompts_are_deterministic`.
Depends on: Task 1 (workspace).

- [ ] Write failing test `prompt::tests::prompts_are_deterministic`: `prompts(7, 10, 256)` twice is byte-identical, `prompts(8, 10, 256)` differs, the first prompt has 256 words, and the 1,000 words are distinct.
- [ ] Run: `cargo test -p turbine-bench --lib prompt::tests` — expect FAIL (`cannot find value `WORD_COUNT` in this scope`).
- [ ] Implement `prompt`: word `i` = onset[i/100] + nucleus[(i/10)%10] + coda[i%10] from onsets `b d f g k l m n p t`, nuclei `a e i o u ai ea io ou ue`, codas `"" n r s t l m x nd st`; SplitMix64 seeded with `seed.wrapping_add(index)`, word = `next() % 1000`, joined by single spaces (no `rand` dependency before P1).
- [ ] Implement `report`: nearest-rank percentiles on sorted values (empty → zeros); throughputs divide by wall seconds; ITL pools every gap of every successful request; text output lists counts, wall time, both throughputs and p50/p95/p99 rows for ttft/itl/e2e in ms.
- [ ] Run: `cargo test -p turbine-bench --lib prompt::tests::prompts_are_deterministic` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(bench): seeded prompt generator and latency/throughput report`

## Task 7: `turbine-bench` binary — streaming client, fixed concurrency, exit codes

Files:

- `benches/turbine-bench/src/lib.rs` — add `pub mod args; pub mod client;` and re-exports.
- `benches/turbine-bench/src/args.rs` — `BenchArgs`, `EndpointArg`, `OutputFormat`.
- `benches/turbine-bench/src/client.rs` — `run`, model resolution, workers, SSE parsing and timing.
- `benches/turbine-bench/src/main.rs` — runtime, report printing, exit codes.
- `benches/turbine-bench/tests/bench.rs` — raw-TCP mock SSE servers and the two acceptance tests.

Interfaces:

- CLI: `turbine-bench --url <base-url> [--model <name>] [--endpoint chat|completions] [--concurrency <n>] [--requests <n>] [--prompt-words <n>] [--max-tokens <n>] [--seed <u64>] [--ignore-eos] [--output text|json]` (defaults chat, 1, 10, 256, 128, 0, text; concurrency/requests ≥ 1 via `clap::value_parser!(u32).range(1..)`)
- `pub struct BenchArgs { pub url: String, pub model: Option<String>, pub endpoint: EndpointArg, pub concurrency: u32, pub requests: u32, pub prompt_words: u32, pub max_tokens: u32, pub seed: u64, pub ignore_eos: bool, pub output: OutputFormat }`
- `pub async fn run(args: &BenchArgs) -> Result<Report, BenchError>`; `pub enum BenchError { Usage(String), Target(String) }` + `pub fn exit_code(&self) -> u8` (2 / 1)
- reqwest 0.13 without default features: `Client::post(url).header(..).body(String).send()`, `Response::chunk()` for streaming (no `stream` feature needed).

Covers: S-6; `bench mock_endpoint_measurements`, `bench failures_counted`, manual run against dgx-spark vLLM.
Depends on: Task 6.

- [ ] Write failing test `bench mock_endpoint_measurements`: mock server sends a role-only chunk (`delta.content: ""`), waits 100 ms, sends 5 content chunks 20 ms apart, a usage chunk `completion_tokens: 5`, then `data: [DONE]`; `--model mock-model --concurrency 2 --requests 4 --output json` → exit 0, `requests_ok` 4, `ttft_ms.p50` ≥ 100, `itl_ms.p50` ≥ 20, `e2e_ms.p50` ≥ 180, `output_token_throughput × wall_seconds` = 20; a second server with `completion_tokens: 7` and 1 request gives 7 tokens (usage wins over chunk count).
- [ ] Write failing test `bench failures_counted`: server failing every other completion with HTTP 500, no `--model` (resolved from `/v1/models`), `--concurrency 2 --requests 4` → exit 0, `requests_ok` 2, `requests_failed` 2, stderr contains `HTTP 500`; server failing all, `--requests 3` → exit 1, `requests_failed` 3.
- [ ] Run: `cargo test -p turbine-bench --test bench` with `src/main.rs` holding only an empty `main` — expect FAIL (`assertion `left == right` failed`, report `Null`).
- [ ] Implement `run`: reject non-`http://` URLs (Usage); resolve the model from `GET <url>/v1/models` `data[0].id` (empty → Usage, unreachable → Target); spawn `min(concurrency, requests)` tokio workers pulling indices from an `AtomicU32`; body has `model`, `max_tokens`, `stream: true`, `stream_options.include_usage: true`, `messages` (chat) or `prompt` (completions), plus `ignore_eos: true` when set.
- [ ] Implement per-request streaming: non-2xx → failure with status and ≤ 200 chars of body; buffer `chunk()` bytes, split on `\n`, handle `data:` lines; content = `choices[0].delta.content` (chat) or `choices[0].text` (completions), non-empty only; record `Instant` per content chunk; keep the last `usage.completion_tokens`; an `error` object, stream end without `[DONE]`, or `[DONE]` before any content → failure logged as `turbine-bench: request <i> failed: <reason>`.
- [ ] Implement `main`: `BenchArgs::parse()`, multi-thread runtime, print JSON (`serde_json::to_string_pretty`) or `to_text()`, exit 0 if `requests_ok > 0` else 1; `BenchError::exit_code()` on setup errors.
- [ ] Run: `cargo test -p turbine-bench` — expect PASS (unit + 2 integration tests).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(bench): streaming OpenAI load generator with TTFT/ITL/E2E percentiles`
- [ ] Lab (manual acceptance): ASK THE USER FIRST — this sends 10 requests to the production vLLM on dgx-spark:8000 (benchmark runs are always asked). After approval run `cargo run --release -p turbine-bench -- --url http://192.168.10.246:8000 --concurrency 2 --requests 10 --output json` — expect exit 0, `"requests_ok": 10`, `ttft_ms.p50` > 0 and `output_token_throughput` > 0; paste the JSON into the task evidence.

## Task 8: Lab runner — `scripts/lab-test.sh`, novanas Job, lab inventory test

Files:

- `crates/turbine-device/tests/lab.rs` — `#[ignore]` test `inventory_matches_expectation`.
- `scripts/lab-test.sh` — rsync + Docker (Sparks) / k3s Job (novanas); exit = test exit code; names the failing step.
- `scripts/lab/novanas-test-job.yaml` — Job `turbine-lab-test` in namespace `turbine-ci`.

Interfaces:

- `scripts/lab-test.sh <dgx-spark|dgx-spark2|novanas>`; anything else → usage line and exit 2; hosts map to `piwi@192.168.10.246`, `piwi@192.168.10.245`, `piwi@192.168.10.203`; SSH with `-o BatchMode=yes -o ConnectTimeout=10`.
- Env read by the lab test: `TURBINE_EXPECT_NVIDIA`, `TURBINE_EXPECT_AMD` (counts, skipped when unset), `TURBINE_EXPECT_NVIDIA_MEMORY` (`unified`|`dedicated`), `TURBINE_EXPECT_AMD_ARCH`; `TURBINE_AMD_SMI_LIBRARY` via `DiscoveryOptions::from_config`.
- Log lines: the pretty inventory JSON and per device `lab-inventory: index=<i> vendor=<v> name=<debug-quoted> arch=<a> memory.kind=<k> total_bytes=<n>`.
- Consumes `discover`, `DiscoveryOptions::from_config`, `DeviceInventory::count` (Task 3).

Covers: S-7, S-4 on real hardware; `lab inventory_matches_expectation` and the three `scripts/lab-test.sh` runs.
Depends on: Tasks 1–7 (the lab run executes the whole suite).

- [ ] Write failing test `lab inventory_matches_expectation` (`#[ignore = "needs lab GPUs; run via scripts/lab-test.sh"]`): discovers with `DiscoveryOptions::from_config(&DevicesConfig::default())`, prints the log lines above, asserts NVIDIA count = `TURBINE_EXPECT_NVIDIA`, AMD count = `TURBINE_EXPECT_AMD`, every NVIDIA device's kind = `TURBINE_EXPECT_NVIDIA_MEMORY` with total > 0 and `shared_with_host` matching, every AMD device's arch = `TURBINE_EXPECT_AMD_ARCH`, kind dedicated, total > 0.
- [ ] Run: `cargo test -p turbine-device --test lab` before the file exists — expect FAIL (`no test target named `lab``); after creating it, `cargo test -p turbine-device --test lab -- --include-ignored` — expect PASS on macOS (no expectations set).
- [ ] Implement `scripts/lab-test.sh` (`set -euo pipefail`): `mkdir -p /home/piwi/turbine-ci/{src,target,cargo-registry}` over SSH; `rsync -az --delete --exclude 'target/' --exclude '.git/'` to `/home/piwi/turbine-ci/src/`; Sparks: check `docker`, `docker rm -f turbine-lab-test` (only our own name), then `docker run --rm --gpus all --name turbine-lab-test --memory 32g -v /home/piwi/turbine-ci/src:/src -v /home/piwi/turbine-ci/target:/target -v turbine-cargo:/usr/local/cargo/registry -e CARGO_TARGET_DIR=/target -e TURBINE_EXPECT_NVIDIA=1 -e TURBINE_EXPECT_NVIDIA_MEMORY=unified -w /src rust:1.97-trixie cargo test --workspace -- --include-ignored --show-output`, mapping exit 125/126/127 to "docker run failed" and 255 to "ssh connection lost"; every failure prints `lab-test: <host>: <step>` and exits non-zero; success prints `lab-test: <host>: PASS`.
- [ ] Implement the novanas branch: check `kubectl`; `kubectl create namespace turbine-ci --dry-run=client -o yaml | kubectl apply -f -`; delete a previous `turbine-lab-test` Job; `kubectl apply -f /home/piwi/turbine-ci/src/scripts/lab/novanas-test-job.yaml`; poll the pod phase every 5 s, failing (and deleting our Job) after 120 s `Unschedulable` with "amd.com/gpu is held by another workload — ask the user to free the GPUs" or after 30 min; `kubectl -n turbine-ci logs -f job/turbine-lab-test`; wait for `.status.succeeded`/`.status.failed`; exit with the container's `terminated.exitCode`. Drop only the kubectl stderr lines containing `permission denied` (k3s config warnings seen for user piwi).
- [ ] Implement `scripts/lab/novanas-test-job.yaml`: `batch/v1` Job `turbine-lab-test`, namespace `turbine-ci`, label `turbine-lab: "true"`, `backoffLimit: 0`, `ttlSecondsAfterFinished: 600`, `activeDeadlineSeconds: 1800`, `restartPolicy: Never`, image `rust:1.97-trixie`, `workingDir: /home/piwi/turbine-ci/src`, command `cargo test --workspace -- --include-ignored --show-output`, env `CARGO_TARGET_DIR=/home/piwi/turbine-ci/target`, `LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib`, `TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0`, `TURBINE_EXPECT_AMD=2`, `TURBINE_EXPECT_AMD_ARCH=gfx1201`, `resources.limits: {amd.com/gpu: 2}`; hostPath mounts `/home/piwi/turbine-ci` (Directory, same path), `/home/piwi/turbine-ci/cargo-registry` → `/usr/local/cargo/registry` (DirectoryOrCreate), `/opt/rocm/rocm` read-only, and `/etc/alternatives/rocm-lib` → `/etc/alternatives/rocm-lib` read-only (on novanas `/opt/rocm/rocm/lib` → `/etc/alternatives/rocm-lib` → `/opt/rocm/rocm/core-7.14/lib`).
- [ ] Run: `shellcheck scripts/lab-test.sh && bash -n scripts/lab-test.sh; scripts/lab-test.sh; echo exit=$?` — expect PASS (no findings) and `exit=2` with the usage line.
- [ ] Run: `ssh -o BatchMode=yes piwi@192.168.10.203 'kubectl apply --dry-run=client -o name -f -' < scripts/lab/novanas-test-job.yaml` — expect PASS (`job.batch/turbine-lab-test`, nothing created).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(lab): lab-test runner for dgx-spark, dgx-spark2 and novanas with inventory check`
- [ ] Lab dgx-spark: check `ssh -o BatchMode=yes piwi@192.168.10.246 "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"` ≥ 40 (GiB); if lower, ASK THE USER FIRST (production vLLM holds the memory; never stop it yourself). Run `scripts/lab-test.sh dgx-spark` — expect exit 0, `lab-inventory: index=0 vendor=nvidia name="NVIDIA GB10" arch=sm_121 memory.kind=unified total_bytes=<non-zero>`, `test inventory_matches_expectation ... ok`, final line `lab-test: dgx-spark: PASS`.
- [ ] Lab dgx-spark2: same MemAvailable check against `piwi@192.168.10.245` (ASK THE USER FIRST if < 40 GiB), then `scripts/lab-test.sh dgx-spark2` — expect the same `vendor=nvidia name="NVIDIA GB10" arch=sm_121 memory.kind=unified` line and `lab-test: dgx-spark2: PASS`.
- [ ] Lab novanas: ASK THE USER FIRST — on 2026-09-25 both `amd.com/gpu` were allocated to a pod in namespace `kuvryn-ai-workloads`; the user frees them. Confirm with `ssh -o BatchMode=yes piwi@192.168.10.203 'kubectl describe node 2>/dev/null | grep -A10 "Allocated resources" | grep amd.com/gpu'` showing `0  0`, then run `scripts/lab-test.sh novanas` — expect `lab-inventory: index=0 vendor=amd name="AMD Radeon AI PRO R9700" arch=gfx1201 memory.kind=dedicated`, the same for `index=1`, `test inventory_matches_expectation ... ok`, and `lab-test: novanas: PASS`. If the script reports the pod unschedulable, stop and ask.

## Task 9: Workspace acceptance, `examples/turbine.yaml` and `AGENTS.md` commands

Files:

- `examples/turbine.yaml` — the TS §15 example with `server.listen: 127.0.0.1:8000`, `server.max_request_bytes: 8MiB`, `model.path: /models/llama-3.2-3b-instruct`, `logging` (text/info) and `devices` (both null); used by the documented commands.
- `AGENTS.md` — "Project state" paragraph and "Commands" section.

Interfaces:

- Consumes the `turbine-server` CLI (Task 5), the `turbine-bench` CLI (Task 7), `scripts/lab-test.sh` (Task 8).

Covers: S-1 (workspace-wide build/test/lint/format on macOS; `unsafe` isolation; `forbid` elsewhere), S-8 (`AGENTS.md` Commands).
Depends on: Tasks 1–8.

- [ ] Write failing test: `cargo run -q -p turbine-server -- --config examples/turbine.yaml --check-config` — expect FAIL (`cannot read examples/turbine.yaml`, exit 2) before the file exists.
- [ ] Implement `examples/turbine.yaml` as described; rerun — expect PASS (`config ok`, exit 0).
- [ ] Implement `AGENTS.md`: replace the "Project state" paragraph with one stating the repo holds the spec plus the Phase 0 skeleton (the six crates serving the V1 route surface without a model; phase specs in `.procoder/specs/`, cross-phase names in `.procoder/contract/interfaces.md`); replace the whole "Commands" section with: toolchain Rust 1.97 / edition 2024 and "no GPU needed to build or test"; build `cargo build --workspace`; test `cargo test --workspace`; one crate `cargo test -p turbine-core`; one unit test `cargo test -p turbine-core config::tests::byte_size_parsing`; one integration test `cargo test -p turbine-api --test api route_table_phase0`; lint `cargo clippy --workspace --all-targets -- -D warnings`; format `cargo fmt --all` / `cargo fmt --all --check`; lab `scripts/lab-test.sh dgx-spark|dgx-spark2|novanas` with the ask-first rule; `cargo run -p turbine-server -- --config examples/turbine.yaml --check-config`; `cargo run -p turbine-server -- --config examples/turbine.yaml --set server.listen=127.0.0.1:8000`; `cargo run -p turbine-bench -- --help`; `cargo run --release -p turbine-bench -- --url http://127.0.0.1:8000 --concurrency 2 --requests 10 --output json` (http only; exit 0/1/2; production vLLM benchmarks asked first); and the server exit codes 0/1/2.
- [ ] Run: `cargo build --workspace && cargo test --workspace && cargo test -p turbine-core && cargo test -p turbine-core config::tests::byte_size_parsing && cargo test -p turbine-api --test api route_table_phase0 && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check && cargo run -q -p turbine-server -- --config examples/turbine.yaml --check-config && cargo run -q -p turbine-bench -- --help >/dev/null` — expect PASS (exit 0; 23 passed, 0 failed, 1 ignored across the workspace).
- [ ] Run: `cargo run -q -p turbine-server -- --config examples/turbine.yaml --set server.listen=127.0.0.1:8000 & sleep 3; curl -s -i http://127.0.0.1:8000/ready | head -1; curl -s http://127.0.0.1:8000/metrics | grep turbine_devices; kill -TERM %1; wait %1; echo exit=$?` — expect PASS: `HTTP/1.1 503 Service Unavailable`, `turbine_devices{vendor="nvidia"} 0`, `turbine_devices{vendor="amd"} 0`, log `shutdown requested; finishing in-flight requests signal="SIGTERM"`, `exit=0` (use SIGTERM: a non-interactive shell starts background jobs with SIGINT ignored).
- [ ] Run: `grep -rn "unsafe" crates/*/src benches/*/src | grep -v '^crates/turbine-device/src/'` — expect no output; `grep -L 'unsafe_code = "forbid"' crates/*/Cargo.toml benches/*/Cargo.toml` — expect exactly `crates/turbine-device/Cargo.toml`; every `unsafe {` in `crates/turbine-device/src` has `// SAFETY:` within the four lines above it.
- [ ] Run: `printf '\npub fn unsafe_probe() {\n    unsafe {}\n}\n' >> crates/turbine-core/src/lib.rs && cargo build -p turbine-core` — expect FAIL (`error: usage of an `unsafe` block`); restore with `git checkout -- crates/turbine-core/src/lib.rs` and `cargo build --workspace` — expect PASS.
- [ ] Run: `cargo tree --workspace -e normal | grep -iE 'cudarc|cuda-sys|rocm|hip-sys'` — expect no output (`nvml-wrapper-sys` is present and resolves NVML through `libloading` at runtime).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `docs(agents): Phase 0 commands and example configuration`
