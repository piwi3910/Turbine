# phase-2b-nvidia — implementation plan

Status: deferred
Spec: .procoder/specs/phase-2b-nvidia.md

Deferred with its spec (on hold since 2026-09-26; decision "Roadmap reorganisation after Phase 5 (2026-09-28)" in `.procoder/ask/decisions.md`). The plan is rewritten from the re-specced spec when the hold is lifted; that spec also adds NVFP4 (modelopt NVFP4 / FP8 mixed precision and compressed-tensors `nvfp4-pack-quantized`), so the task list below is incomplete for it.

## Goal

Run the complete Phase 2 serving runtime — both models, every op, the scheduler, paged KV, llguidance and tool calling — unchanged on the DGX Sparks (GB10, `sm_121`, unified memory) through a second implementation of the ABI v2 kernel C ABI, _libturbine_cuda.so_, built from FlashInfer, cuBLASLt and cuBLAS, proven against the committed golden fixtures and benchmarked next to the production vLLM without disturbing it.

## Architecture

`kernels/cuda/` is a CMake/`nvcc` project that implements `kernels/include/turbine_kernels.h` (ABI v2, unchanged) for `121a`: a context layer (device selection, one non-blocking stream, cuBLAS/cuBLASLt handles, fixed workspace, CUDA-error-name messages), FlashInfer header-only templates fetched at a pinned commit for attention (single and batch-paged, FA2 path) and RMSNorm, cuBLASLt for dense GEMM, `cublasGemmGroupedBatchedEx` (with a per-expert cuBLASLt fallback) for MoE experts, and minimal Turbine CUDA kernels for everything else, including a device-side paged-attention plan so no op blocks. On the Rust side nothing CUDA-specific is added: the Phase 1 `ShimLibrary` gains backend-name checks and CUDA stub shims for tests, `turbine-server` accepts `execution.backend: cuda` with a vendor check and logs `kernel_library_loaded`, the startup budget applies the unified rule, and GPU tests pick their backend from `TURBINE_TEST_BACKEND`. Spark lab scripts run everything in a pinned CUDA 13.0.3 arm64 image behind a `MemAvailable` precondition so production vLLM is never squeezed.

## Constraints

From the spec (verbatim):

- Same C ABI: _libturbine_cuda.so_ implements `kernels/include/turbine_kernels.h` exactly as _libturbine_hip.so_ does; the header stays free of vendor identifiers (phase-1 `abi_header_neutral` test). If an NVIDIA provider needs information the descriptors do not carry, the ABI version is bumped and both shims implement the change in the same commit — never a CUDA-only side channel.
- Rust only in the workspace build and runtime path; the shim is C++/CUDA compiled by `nvcc` through CMake and exposes only the C ABI (TS §6). No Python in the build or serving path (TS §21 rule 4): no FlashInfer JIT, no Python code generation during `cmake --build`; Python appears only in the unchanged `scripts/golden/hf_reference.py`.
- The Rust workspace builds and every non-ignored test passes on macOS arm64 with no GPU, no CUDA and no ROCm (Phase 0/1 constraint, unchanged). Nothing links CUDA at build time.
- `unsafe` and FFI only in `turbine-device` and `turbine-kernels`, every block with a `// SAFETY:` comment; the Phase 1 ownership rules (TS §21 rule 10) apply unchanged to CUDA device pointers, streams and handles.
- External kernel libraries: each fetched provider pinned by exact commit or release in `kernels/cuda/CMakeLists.txt`, only the template instantiations Turbine uses compiled (BF16, head_dim 128, the GQA ratios of the two models, the Phase 2 page layout); licenses retained under `kernels/cuda/third_party/LICENSES/` (TS §6). Toolkit libraries (cuBLASLt, cuBLAS, the static CUDA runtime) come from the CUDA 13.0 toolkit of the lab image, which matches the host driver 580.173.02's native CUDA version; the toolkit is never taken from the host's `/usr/local/cuda`.
- Target hardware: `dgx-spark` (192.168.10.246) and `dgx-spark2` (192.168.10.245) — NVIDIA GB10, compute capability 12.1 (`sm_121`), arm64 (aarch64) Ubuntu 24.04, driver 580.173.02 (CUDA 13.0 driver API), host CUDA toolkit 13.0 (`nvcc` 13.0.88, cuBLAS/cuBLASLt 13.1.1.3, no NCCL, no cuDNN), CMake 3.28.3 on the host, ~121 GB unified memory, Docker 29 with the `nvidia` runtime. Lab execution is always `docker run --gpus all` (Phase 0), user `piwi`, containers named `turbine-lab-*`, HTTP ports 18000–18099 (Turbine 18000, vLLM baseline 18100).
- Production co-tenancy: both Sparks serve production vLLM (image `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5`, host networking, 55–75 GB of the unified pool; `MemAvailable` was 42 GiB on dgx-spark and 29 GiB on dgx-spark2 on 2026-09-25). Turbine scripts never stop, restart, move or reconfigure those containers. Any run that needs production workloads moved or memory freed on any host is preceded by asking the user, and the user moves workloads. Correctness runs (`lab-test.sh`, golden compare) proceed without asking only when the memory precondition passes. Benchmark, soak and overload runs always ask the user first, because GB10 compute is shared with production vLLM: an unannounced benchmark degrades production and measures co-tenant noise.
- Bounded resources (TS §21 rule 8): the Phase 1 staging buffer (≤ 256 MiB) and the Phase 2 bounds apply unchanged; the shim's GEMM workspace is fixed at context creation; the lab containers get `--memory` caps (Interfaces) so host-side runaway cannot squeeze production.

From the interface contract (`.procoder/contract/interfaces.md`, binding):

- Toolchain: edition 2024, `rust-version = "1.97"`, `license = "Apache-2.0"` from `[workspace.package]`; `unsafe_code = "forbid"` in every crate except `turbine-device` and `turbine-kernels` (§1.3); no `cudarc` or other CUDA binding crate anywhere (§1.2).
- ABI: this phase implements exactly `TURBINE_ABI_VERSION` 2 (§9.1) — the v1 functions, `turbine_ctx_get_info` and the five Phase 2 op trios — with no header change; only `turbine_*` symbols exported (§9.2); every entry point returns the §9.3 codes −1…−5 and `turbine_last_error` messages start with the CUDA error name (§9.2).
- Names: `ShimLibrary`, `ShimContext`, `ContextInfo`, `KernelError::{Device, OutOfMemory, BackendMismatch, ArchMismatch, Load, AbiMismatch}`, `test_support::require_backend`, `ExecutionBackend::Cuda`, `turbine_model::budget::{BudgetTerms, available_bytes, check_budget}`, log events `kernel_library_loaded` and `memory_budget` (§7.1, §10, §18); provider order for backend `cuda` is `["cuda"]` and the CPU reference is never used for a GPU model (§7.1); CUDA impl strings exactly `cublaslt`, `flashinfer_prefill`, `flashinfer_decode`, `flashinfer_batch_prefill_paged`, `flashinfer_batch_decode_paged`, `flashinfer_rmsnorm`, `cublas_grouped_batched`, `cublaslt_per_expert`, `turbine_cuda` (§9.3).
- Config (§3.2, §23): `execution.backend` accepts `cuda`; `kv.block_tokens` other than 16 is refused at startup by the CUDA `_supported` functions; `server.listen: 0.0.0.0:18000` (CONFLICT C-13, no `server.port`); `kv.gpu.max_bytes` is `Option<ByteSize>` and the Spark configs keep `4GiB` (C-8); golden slugs `llama-3.2-3b-instruct`, `olmoe-1b-7b-0125-instruct` (C-15); pool layout identical on both backends (C-23); Phase 2 device-error rule (3 failed iterations → `/ready` 503 `device_error`, exit 1) applies unchanged (C-25).
- Tests (§20.1): unit tests in `#[cfg(test)] mod tests`, integration tests under `crates/<crate>/tests/`; every ignored GPU test starts with `if !turbine_kernels::test_support::require_backend("hip"|"cuda") { return; }`; non-ignored tests pass on macOS arm64.
- Lab (§21.1, DEC): no `docker run` outside `scripts/lab-*.sh`; only `turbine-lab-*` containers are ever touched; Spark correctness runs proceed after the `MemAvailable` precondition; benchmark, soak and overload runs always ASK THE USER FIRST; weights downloads need the user's Hugging Face token, supplied at that time and never stored.
- Gate after every task: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`.

Pinned values used across tasks (verified 2026-09-25, read-only):

- Lab image base (Docker Hub registry API, arm64 entry of the multi-arch index `sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266`; `13.0.4`/`13.0.5` tags do not exist; `/usr/local/cuda-13.0/version.json` on dgx-spark reports CUDA SDK 13.0.3, nvcc 13.0.88, cudart 13.0.96):

```dockerfile
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:2396393dc8a031a6faf810894ec8e40c8d80e2a1177cfda0c2bf81acac9aa240
```

- rustup-init 1.29.1 for `aarch64-unknown-linux-gnu`, SHA-256 `15f6e4ce9f583b929c996c91562bad6d4454f3281de858b02cdfdef615fac433` (static.rust-lang.org `.sha256`); toolchain `1.97.1` with `rustfmt` and `clippy`; uv `ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424`; apt packages `binutils build-essential ca-certificates cmake curl git pkg-config rsync` (Ubuntu 24.04 `cmake` is 3.28.3).
- FlashInfer `v0.6.18.post1`, commit `8bc3b578027791336c6ae87db5c9d76f82cef8bc` (header-only use of `include/flashinfer`, submodules not fetched). CCCL `v3.3.2`, commit `876867684f7fac130e0f5911236e0a92a970d4fd` — the commit FlashInfer pins as `3rdparty/cccl`; required because `include/flashinfer/fastdiv.cuh:39` uses `cuda::fast_mod_div<uint32_t>` from `<cuda/cmath>`, which the CCCL 3.0.1 bundled with CUDA 13.0.3 (`cccl/cuda/std/__cccl/version.h`: `CCCL_VERSION 3000001`) does not provide.
- Production vLLM image `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5@sha256:b94ac8ac47603f9c3d14f88c12b90eacbc37c7a42f922631cb8c9ac80f7b37c3` (entrypoint `/opt/nvidia/nvidia_entrypoint.sh`; production containers run command `serve /model …`).

Open decisions flagged to the user (planned with the recommended option):

- OPEN-1: pinned `cudaHostAlloc` staging (S-6) versus "`turbine_stream_sync` is the only blocking call" (S-5) under ABI v2, which has no pinned-memory function (contract C-6 adds it in v3). Planned: `turbine_memcpy_h2d` stages copies of 1 MiB or more through two context-owned 32 MiB pinned halves and waits only on the event of that half's previous chunk; smaller copies go straight to `cudaMemcpyAsync`.
- OPEN-2: the Phase 1/2 tiny checkpoints must use head_dim 128, because only head_dim 128 FlashInfer instantiations are compiled (spec Constraints). Planned: tiny checkpoints use head_dim 128.

## Task 1: Backend-selected GPU tests (`require_backend`)

Files: `crates/turbine-kernels/src/test_support.rs` (P1 file; make the unset/empty rule a panic, add `backend_under_test`, the pure decision helper and its unit test)
Interfaces:

- produces `pub fn require_backend(backend: &str) -> bool` (contract §7.1: equal → `true`; other value → prints `SKIP backend=<value>`, `false`; unset or empty → panic naming `TURBINE_TEST_BACKEND`)
- produces `pub fn backend_under_test() -> turbine_core::types::ExecutionBackend` (reads `TURBINE_TEST_BACKEND`; `hip`/`cuda` only, else panic naming the variable) — used by `golden` and `lab_openai` (Tasks 17, 19)
- produces `pub const BACKEND_ENV: &str = "TURBINE_TEST_BACKEND"` and `pub(crate) fn backend_decision(backend: &str, value: Option<&str>) -> BackendDecision` with `pub(crate) enum BackendDecision { Run, Skip(String) }`
- consumes P1 `require_env_dir(var: &str) -> PathBuf` unchanged

Covers: S-8; `cargo test -p turbine-kernels test_support::tests::require_backend_semantics`
Depends on: phase-1 plan (`turbine_kernels::test_support` exists)

- [ ] Write failing test `test_support::tests::require_backend_semantics`: `backend_decision("cuda", Some("cuda"))` is `Run`, `backend_decision("cuda", Some("hip"))` is `Skip("SKIP backend=hip")`, `backend_decision("hip", Some("cuda"))` is `Skip("SKIP backend=cuda")`, and both `None` and `Some("")` panic (caught with `std::panic::catch_unwind`) with a message containing `TURBINE_TEST_BACKEND`.
- [ ] Run: `cargo test -p turbine-kernels test_support::tests::require_backend_semantics` — expect FAIL (`backend_decision` not found).
- [ ] Implement `backend_decision` as a pure match on the value and route `require_backend` through it with `std::env::var(BACKEND_ENV).ok()`; the `Skip` line is printed with `println!` so `cargo test` shows it in captured output; add `backend_under_test` on the same variable. The empty string counts as unset so a lab container with `-e TURBINE_TEST_BACKEND=` fails loudly instead of skipping.
- [ ] Run: `cargo test -p turbine-kernels test_support::tests::require_backend_semantics` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): fail GPU tests loudly when TURBINE_TEST_BACKEND is unset`

## Task 2: CUDA stub shims and backend-name/arch checks in `ShimLibrary`

Files: `crates/turbine-kernels/build.rs` (P1 file; data-driven stub list adding `stub_hip_gfx1201`, `stub_cuda_sm121`, `stub_cuda_sm90`, `stub_cuda_errors`), `crates/turbine-kernels/stub/stub_shim.c` (P1 file; `STUB_BACKEND`/`STUB_ARCHS`/`STUB_ERRORS` defines, `turbine_ctx_get_info` reporting 32 MiB workspace and compute capability 12.1, every v2 trio returning impl `stub_impl`), `crates/turbine-kernels/src/shim.rs` (backend-name check in `load`, arch check in `create_context`, pure search-order helper and unit tests), `crates/turbine-kernels/src/test_support.rs` (stub path helper)
Interfaces:

- consumes `ShimLibrary::load(path: &Path, expected_backend: ExecutionBackend) -> Result<Arc<ShimLibrary>, KernelError>`, `ShimLibrary::create_context(self: &Arc<Self>, device: &DeviceInfo) -> Result<Arc<ShimContext>, KernelError>`, `ShimContext::info(&self) -> ContextInfo` (P1/P2)
- produces `pub fn expected_backend_name(backend: ExecutionBackend) -> Option<&'static str>` (`hip`→`"hip"`, `cuda`→`"cuda"`, `cpu`→`None`) and `pub fn library_file_name(backend: ExecutionBackend) -> Option<&'static str>` (`libturbine_hip.so` / `libturbine_cuda.so`)
- produces `pub(crate) fn search_paths_from(backend: ExecutionBackend, explicit: Option<&Path>, env: Option<PathBuf>, exe_dir: Option<&Path>) -> Vec<PathBuf>`; `ShimLibrary::search_paths(backend, explicit) -> Vec<PathBuf>` becomes a thin wrapper over it
- produces `pub fn test_support::stub_library(variant: &str) -> PathBuf` (`$TURBINE_STUB_DIR/libstub_<variant>.so`); `build.rs` exports `TURBINE_STUB_DIR` and one `TURBINE_STUB_<NAME>` per stub, keeping P1's `TURBINE_STUB_ABI999` and `TURBINE_STUB_GFX942`
- errors: `KernelError::BackendMismatch { expected, found }` ("kernel library backend {found}, configured {expected}"), `KernelError::ArchMismatch { device_arch, build_archs }` (contract §7.1 messages)

Covers: S-2, S-5 (arch refusal); `cargo test -p turbine-kernels shim::tests::backend_name_and_arch_checked`
Depends on: Task 1; phase-1 plan (`ShimLibrary`, stub build), phase-2 plan (ABI v2 header, `turbine_ctx_get_info`, `ContextInfo`)

- [ ] Write failing test `shim::tests::backend_name_and_arch_checked`: for a mocked GB10 `DeviceInfo` (vendor `nvidia`, arch `sm_121`, kind `unified`) (a) `stub_cuda_sm121` loads under `ExecutionBackend::Cuda`, reports backend `cuda` and archs `["sm_121"]`, and its context reports compute capability `(12, 1)` and 33,554,432 workspace bytes; (b) `stub_cuda_sm90` loads but `create_context` returns `ArchMismatch` whose message contains `sm_90` and `sm_121`; (c) `stub_hip_gfx1201` under `Cuda` returns `BackendMismatch` whose message contains `hip` and `cuda`. Add `shim::tests::cuda_search_order`: with env `/target/kernels-cuda/libturbine_cuda.so` and exe dir `/opt/turbine/bin`, the order is env path, `/opt/turbine/bin/libturbine_cuda.so`, `libturbine_cuda.so`; an explicit path alone; `cpu` yields an empty list.
- [ ] Run: `cargo test -p turbine-kernels shim::tests::backend_name_and_arch_checked` — expect FAIL (`stub_cuda_sm121` not built / `stub_library` not found).
- [ ] Implement the stub variants in `build.rs` (host C compiler from `cc::Build::new().get_compiler()`, `-shared -fPIC`, `-I ../../kernels/include`, defines per variant, ABI from the header unless the variant overrides it) and the checks: `load` compares `turbine_backend_name()` with `expected_backend_name(expected_backend)` right after the ABI check; `create_context` checks the inventory `arch` is in the comma-split `turbine_build_archs()` before calling `turbine_ctx_create(vendor_index)`. Every `unsafe` block keeps its `// SAFETY:` comment (symbols resolved once, strings are static per contract §9.3).
- [ ] Run: `cargo test -p turbine-kernels shim::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): check shim backend name and build archs for cuda`

## Task 3: CUDA error names survive the FFI boundary

Files: `crates/turbine-kernels/src/ffi.rs` (P1 `check` keeps the shim message verbatim; `last_error` grows its buffer to the reported length), `crates/turbine-kernels/src/shim.rs` (unit test)
Interfaces:

- consumes `pub(crate) fn check(code: i32, syms: &ShimSymbols, ctx: *mut RawCtx) -> Result<(), KernelError>` and `pub(crate) fn last_error(syms: &ShimSymbols, ctx: *mut RawCtx) -> String` (contract §9.4)
- mapping: −1 `InvalidArgument`, −2 `Unsupported`, −3 `OutOfMemory`, −4 `Device { message }` with `Display` = the message itself, −5 `Library`, any other negative → `Library` with `unknown status <n>: <message>`
- `KernelError::is_oom(&self) -> bool` true for `OutOfMemory`

Covers: S-5; `cargo test -p turbine-kernels shim::tests::cuda_error_names_preserved`
Depends on: Task 2

- [ ] Write failing test `shim::tests::cuda_error_names_preserved`: with `stub_cuda_errors` (its `turbine_stream_sync` returns −4 with `cudaErrorIllegalAddress: an illegal memory access`, its `turbine_malloc` returns −3 with `cudaErrorMemoryAllocation: out of memory`), `ShimContext::stream_sync()` yields `KernelError::Device` whose message equals `cudaErrorIllegalAddress: an illegal memory access` and whose `to_string()` starts with `cudaErrorIllegalAddress`; `ShimContext::malloc(1 << 20)` yields `OutOfMemory`, `is_oom()` is true and the text contains `cudaErrorMemoryAllocation`.
- [ ] Run: `cargo test -p turbine-kernels shim::tests::cuda_error_names_preserved` — expect FAIL (stub variant missing or message prefixed).
- [ ] Implement: `check` reads `turbine_last_error` into a 512-byte buffer, re-reads with `full + 1` bytes when the returned length does not fit (contract §9.2 return value), truncates to `full` and never prefixes or rewrites the text, so Phase 3's `is_sticky` can classify by the leading CUDA error name.
- [ ] Run: `cargo test -p turbine-kernels shim::tests::cuda_error_names_preserved` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): keep CUDA error names in KernelError messages`

## Task 4: Unified-memory startup budget

Files: `crates/turbine-model/src/budget.rs` (P1 file; `AvailableSources`, unified min rule in `available_bytes`, both sources in the refusal text and in the `memory_budget` record, unit tests)
Interfaces:

- consumes `pub fn available_bytes(kind: MemoryKind, device_free: u64, host_mem_available: Option<u64>) -> u64` and `pub fn check_budget(terms: &BudgetTerms) -> Result<(), ModelError>` (contract §10; `ModelError::Budget(String)`)
- produces `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct AvailableSources { pub kind: MemoryKind, pub device_free: u64, pub host_mem_available: Option<u64> }`
- extends `pub struct BudgetTerms { weights, kv_reservation, workspace, emergency_reserve, available, pub sources: AvailableSources }`, with `pub fn BudgetTerms::new(weights: u64, kv_reservation: u64, workspace: u64, emergency_reserve: u64, sources: AvailableSources) -> BudgetTerms` and `pub fn claim(&self) -> u64`
- produces `pub fn log_memory_budget(terms: &BudgetTerms)` emitting INFO `event = "memory_budget"` with `weights`, `kv_reservation`, `workspace`, `emergency_reserve`, `available_bytes`, `device_kind` (`unified`|`dedicated`), `device_free_bytes`, `host_mem_available_bytes`

Covers: S-6; `cargo test -p turbine-model budget::tests::unified_available_is_minimum`
Depends on: phase-1 plan (`budget` module, `MemoryKind` from the Phase 0 inventory)

- [ ] Write failing test `budget::tests::unified_available_is_minimum`: `available_bytes(Unified, 60 GiB, Some(29 GiB))` = 29 GiB (never 89 GiB), `available_bytes(Unified, 20 GiB, Some(42 GiB))` = 20 GiB; `BudgetTerms::new(24 GiB, 4 GiB, 512 MiB, 1.5 GiB, unified(60 GiB, 29 GiB))` claims 30 GiB and `check_budget` refuses it with a message containing `available_bytes=31138512896`, `device_free_bytes=64424509440` and `host_mem_available_bytes=31138512896`, while the same claim against `unified(60 GiB, 42 GiB)` passes. Add `budget::tests::memory_budget_record_carries_unified_sources`: a capturing `tracing_subscriber::Layer` sees one `memory_budget` record with `device_kind` `unified` and `available_bytes` equal to the smaller source.
- [ ] Run: `cargo test -p turbine-model budget::tests::unified_available_is_minimum` — expect FAIL (`AvailableSources` not found).
- [ ] Implement: `available_bytes` returns `device_free.min(host)` for `Unified` with a host value and `device_free` otherwise (a unified device without `/proc/meminfo` falls back to the device view); `check_budget` text lists every term, the sum and `available_bytes=<n> (unified: min of device_free_bytes=<a>, host_mem_available_bytes=<b>)`; the startup code (Task 5) builds the sources from `turbine_mem_info` free bytes and the Phase 0 `/proc/meminfo` reader, before any weight byte is read.
- [ ] Run: `cargo test -p turbine-model budget::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): apply the unified-memory budget rule with both sources logged`

## Task 5: `execution.backend: cuda` with a vendor check and the CUDA library search

Files: `crates/turbine-core/src/config/execution.rs` (P1 file; drop the `cuda` → exit-2 rejection), `crates/turbine-server/src/startup.rs` (vendor match, library search loop, provider order, budget sources), `crates/turbine-server/src/main.rs` (debug-build inventory hook wiring), `crates/turbine-server/tests/tiny_server.rs` (P1 file; new test)
Interfaces:

- produces `pub fn backend_vendor(backend: ExecutionBackend) -> Option<Vendor>` (`Hip`→`Amd`, `Cuda`→`Nvidia`, `Cpu`→`None`)
- produces `pub fn select_device(backend: ExecutionBackend, device: DeviceId, inventory: &DeviceInventory) -> Result<&DeviceInfo, String>` (message form `execution.device 0 is an amd device (<name>); execution.backend cuda needs an nvidia device`; missing index → `execution.device 0: no such device in the inventory (0 devices)…`)
- produces `pub fn load_kernel_library(backend: ExecutionBackend, explicit: Option<&Path>) -> Result<Arc<ShimLibrary>, String>` (logs `search_order`; a `KernelError::Load` moves to the next path, any other refusal is fatal; the final error repeats the order and every loader error)
- produces `pub const TEST_INVENTORY_ENV: &str = "TURBINE_TEST_INVENTORY"` and `#[cfg(debug_assertions)] pub fn parse_test_inventory(spec: &str) -> Result<DeviceInventory, String>` (`vendor:arch:kind[,…]`, e.g. `amd:gfx1201:dedicated`) — debug builds only, replaces discovery in `cargo test` binaries
- provider order: `ExecutionBackend::Cuda` → `[ProviderId("cuda")]` passed to `KernelRegistry::build` (contract §7.1)

Covers: S-7; `cargo test -p turbine-server --test tiny_server backend_device_mismatch`
Depends on: Tasks 2, 4; phase-0 plan (startup order, exit codes), phase-1 plan (`Config`, `ExecutionConfig`, registry)

- [ ] Write failing test `tiny_server backend_device_mismatch`: a config with `execution.backend: cuda`, `execution.device: 0` and `TURBINE_TEST_INVENTORY=amd:gfx1201:dedicated` exits 1 with stderr containing `execution.device 0`, `amd` and `cuda` and not `phase-2b-nvidia`; with `TURBINE_TEST_INVENTORY=nvidia:sm_121:unified`, `TURBINE_KERNEL_LIBRARY` removed and `execution.kernel_library` null it exits 1 with stderr containing `search order` and `libturbine_cuda.so` and not `libturbine_hip.so`. Add unit test `startup::tests::vendor_mismatch_names_index_vendor_and_backend` over `select_device` for the same cases plus an empty inventory.
- [ ] Run: `cargo test -p turbine-server --test tiny_server backend_device_mismatch` — expect FAIL (exit 2 naming `phase-2b-nvidia`).
- [ ] Implement: remove the Phase 1 `cuda` rejection from config validation; at startup step 4 call `select_device`, then `load_kernel_library` over `ShimLibrary::search_paths(backend, cfg.execution.kernel_library)`, then `create_context`; map every refusal to `ExitCode::Startup` (1) after config validation; build `AvailableSources` from `ShimContext` `mem_info` and `/proc/meminfo` `MemAvailable` for `MemoryKind::Unified` devices (Task 4).
- [ ] Run: `cargo test -p turbine-server --test tiny_server backend_device_mismatch` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): accept execution.backend cuda with vendor and library checks`

## Task 6: `kernel_library_loaded` record and CUDA provider labels

Files: `crates/turbine-server/src/startup.rs` (`log_kernel_library_loaded`), `crates/turbine-server/tests/tiny_server.rs` (new test)
Interfaces:

- produces `pub fn log_kernel_library_loaded(lib: &ShimLibrary, device: &DeviceInfo, info: &ContextInfo)` — INFO `event = "kernel_library_loaded"` with `path`, `backend`, `abi_version`, `build_archs` (comma-joined), `device_arch`, `driver_version` (Phase 0 inventory), `workspace_bytes` (from `ContextInfo`) and `compute_capability` (`"12.1"`, from `ContextInfo.compute_capability`)
- consumes `ShimLibrary::{path, backend_name, abi_version, build_archs}`, `turbine_kernel_provider_selected{op,provider,impl}` set by `KernelRegistry::build` (P1)
- consumes `turbine_model::testing::tiny::write_tiny_llama(dir: &Path, seed: u64) -> TinySpec` (contract §10)

Covers: S-14; `cargo test -p turbine-server --test tiny_server kernel_library_loaded_record`
Depends on: Task 5

- [ ] Write failing test `tiny_server kernel_library_loaded_record`: on a GPU-less host, start `turbine-server` with `execution.backend: cuda`, `execution.kernel_library` = `test_support::stub_library("cuda_sm121")`, `TURBINE_TEST_INVENTORY=nvidia:sm_121:unified`, `logging.format: json` and the tiny Llama checkpoint; parse stderr as JSON lines and assert exactly one record with `event` `kernel_library_loaded` carrying `path` (the stub path), `backend` `cuda`, `abi_version` 2, `build_archs` `sm_121`, `device_arch` `sm_121`, `driver_version` `test` and `workspace_bytes` 33554432; after `/ready` 200, `GET /metrics` contains `turbine_kernel_provider_selected{op="gemm",provider="cuda",impl="stub_impl"} 1`.
- [ ] Run: `cargo test -p turbine-server --test tiny_server kernel_library_loaded_record` — expect FAIL (no such record).
- [ ] Implement the record right after `create_context` in startup step 4 (contract §16.3) and pass `shim_provider(ctx)` (id = backend name) to the registry so the gauge carries `provider="cuda"` and the stub's `_impl` string.
- [ ] Run: `cargo test -p turbine-server --test tiny_server kernel_library_loaded_record` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): log kernel_library_loaded with device arch and workspace`

## Task 7: Spark lab image, `lab-test.sh` Spark branch and the memory precondition

Files: `scripts/lab/spark.Dockerfile` (new: CUDA 13.0.3 arm64 lab image), `scripts/lab/spark-common.sh` (new, sourced: host IPs, SSH, `MemAvailable` reader, precondition arithmetic, image tag, tree sync, image build-if-absent, `turbine-lab-*`-only container removal, dry-run printing), `scripts/lab-test.sh` (P0 file; Spark branch), `benches/turbine-bench/tests/lab_scripts.rs` (new test binary)
Interfaces:

- shell functions: `spark_ip <host>`, `spark_ssh <host> <cmd>`, `run_remote <host> <cmd>` (prints `+ ssh piwi@<ip> <cmd>` when `DRY_RUN=1`), `spark_meminfo_bytes <host> <MemAvailable|MemTotal>` (test override `TURBINE_LAB_MEMINFO=<file>`), `spark_weight_bytes <host> <slug>` (`du -sb`; test override `TURBINE_LAB_WEIGHT_BYTES`), `spark_require_mem <available> <required> <arithmetic>`, `spark_test_precondition <host>`, `spark_image_tag` (`turbine-lab-cuda:<first 12 hex of sha256(scripts/lab/spark.Dockerfile)>`), `spark_sync_tree <host>`, `spark_ensure_image <host> <tag>`, `lab_rm_containers <host> <name>…` (refuses any name not starting `turbine-lab-`)
- constants: host reserve 8 GiB, test claim 16 GiB (required ≥ 24 GiB), Turbine port 18000, vLLM port 18100
- refusal message (stderr, exit 1): `precondition: MemAvailable <n> GiB < <m> GiB; ask the user to free memory`

Covers: S-9 (lab-test Spark path, precondition, image); `cargo test -p turbine-bench --test lab_scripts spark_precondition_arithmetic`
Depends on: phase-0 plan (`scripts/lab-test.sh`, rsync layout `/home/piwi/turbine-ci/{src,target}`, `turbine-cargo` volume)

- [ ] Write failing test `lab_scripts spark_precondition_arithmetic`: run `bash -c 'source scripts/lab-serve.sh && spark_serve_precondition dgx-spark scripts/lab/phase2b-spark-olmoe.yaml'` with `TURBINE_LAB_MEMINFO` holding `MemAvailable` 26 GiB and `TURBINE_LAB_WEIGHT_BYTES=13851000000`: exit 1, stderr contains `precondition: MemAvailable 26.0 GiB < 28.9 GiB`, stdout contains `weights 12.9 GiB + kv 4.0 GiB + runtime 4.0 GiB = needed 20.9 GiB`; with 42 GiB exit 0 and `precondition: ok`; `spark_test_precondition dgx-spark2` refuses 23 GiB and passes 29 GiB.
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts spark_precondition_arithmetic` — expect FAIL (script missing).
- [ ] Implement `scripts/lab/spark.Dockerfile` from the pinned values in Constraints: the exact `FROM` line, `apt-get install --no-install-recommends binutils build-essential ca-certificates cmake curl git pkg-config rsync`, rustup-init 1.29.1 verified with `sha256sum -c` then `--default-toolchain 1.97.1 --profile minimal --component rustfmt --component clippy`, `RUSTUP_HOME=/usr/local/rustup`, `CARGO_HOME=/usr/local/cargo`, `PATH` with `/usr/local/cuda/bin`, uv copied from the pinned `ghcr.io/astral-sh/uv:0.12.19@sha256:…` image, and a final `RUN rustc --version && cargo --version && cmake --version && nvcc --version && uv --version`. Implement `spark-common.sh` (Bash 3.2 compatible; arithmetic in bytes, `awk` only for printing GiB with one decimal) and the Spark branch of `lab-test.sh`: ssh check → precondition → rsync → image build if absent → `docker run --rm --gpus all --name turbine-lab-test --memory 32g` with `/src`, `/target`, `turbine-cargo:/usr/local/cargo/registry`, `/home/piwi/turbine-models:/models:ro`, `CARGO_TARGET_DIR=/target` and the §Interfaces test variables, running CMake configure, CMake build into `/target/kernels-cuda`, then `cargo test --workspace -- --include-ignored`; each failed step prints `lab-test: step <name> failed`; the novanas branch stays as Phase 0–2 left it.
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts spark_precondition_arithmetic` — expect PASS.
- [ ] Run: `bash -n scripts/lab-test.sh && shellcheck -x scripts/lab-test.sh scripts/lab/spark-common.sh` — expect exit 0.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(scripts): Spark lab image and lab-test.sh CUDA path behind a MemAvailable precondition`

## Task 8: `lab-serve.sh` Spark modes, Spark configs and weights manifest

Files: `scripts/lab-serve.sh` (P1/P2 file; Spark Turbine, vLLM and stop modes, `--dry-run`), `scripts/lab/phase2b-spark-llama.yaml` and `scripts/lab/phase2b-spark-olmoe.yaml` (new lab configs), `scripts/lab/weights-manifest.sh` (new), `benches/turbine-bench/tests/lab_scripts.rs` (dry-run test)
Interfaces:

- shell functions: `config_model_slug <config>` (from `model.path: /models/<slug>`), `config_kv_bytes <config>` (`kv.gpu.max_bytes` `<n>GiB|<n>MiB`), `spark_serve_precondition <host> <config>` (needed = weights + KV + 4 GiB; required = needed + 8 GiB), `spark_vllm_precondition <host>` (required = MemTotal/5 + 8 GiB), `spark_build_command <tag> <extra>`, `spark_wait_http <host> <url> <seconds> <container>`, `hf_id_for_slug <slug>`
- variables at the top of `lab-serve.sh`: `VLLM_IMAGE=ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5`, `VLLM_IMAGE_DIGEST=sha256:b94ac8ac47603f9c3d14f88c12b90eacbc37c7a42f922631cb8c9ac80f7b37c3`
- `scripts/lab/weights-manifest.sh <dgx-spark|dgx-spark2|novanas> <slug>` prints `sha256  ./<file>` sorted with `LC_ALL=C` for every file under `/home/piwi/turbine-models/<slug>` except `./.cache/*`; read-only

Covers: S-9 (serve modes, dry-run, cleanup scope), S-10 (manifest script); `bash -n scripts/lab-test.sh scripts/lab-serve.sh` + `scripts/lab-serve.sh dgx-spark --dry-run scripts/lab/phase2b-spark-llama.yaml`
Depends on: Task 7; phase-1/phase-2 plans (novanas modes of `lab-serve.sh`)

- [ ] Write failing test `lab_scripts spark_dry_run_prints_commands_only`: `scripts/lab-serve.sh dgx-spark --dry-run scripts/lab/phase2b-spark-llama.yaml` with `TURBINE_LAB_MEMINFO` at 42 GiB and `TURBINE_LAB_WEIGHT_BYTES=6425000000` exits 0, prints `precondition: weights 6.0 GiB + kv 4.0 GiB`, one `docker run -d` line containing `--gpus all`, `--name turbine-lab-serve`, `--memory 16g`, `-p 18000:18000` and `/home/piwi/turbine-models/llama-3.2-3b-instruct:/models/llama-3.2-3b-instruct:ro`, and every `docker rm -f` line names only `turbine-lab-*` containers.
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts spark_dry_run_prints_commands_only` — expect FAIL.
- [ ] Implement: `--dry-run` may appear anywhere, sets `DRY_RUN=1`, prints the precondition arithmetic (a refusing precondition prints `dry-run: the precondition would refuse this run (exit 1)`) and every remote command, ends with `dry-run: nothing was run` and exit 0. Turbine mode: precondition → rsync → image → build container `turbine-lab-build` running `cmake -S kernels/cuda -B /target/kernels-cuda -DCMAKE_CUDA_ARCHITECTURES=121a && cmake --build /target/kernels-cuda --parallel 4 && cargo build --release -p turbine-server` → `docker run -d --gpus all --name turbine-lab-serve --memory 16g -p 18000:18000` with the model directory read-only at `/models/<slug>`, the config at `/config.yaml:ro`, `/target:ro` and `TURBINE_KERNEL_LIBRARY=/target/kernels-cuda/libturbine_cuda.so` → wait up to 600 s for `/ready` 200 while streaming `docker logs -f`, on timeout print `docker logs --tail 100` and remove the container. vLLM mode: `docker run -d --gpus all --name turbine-lab-vllm -p 18100:8000 --ipc host` of `$VLLM_IMAGE@$VLLM_IMAGE_DIGEST` with `serve /models/<slug> --served-model-name <HF id> --dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.20 --max-model-len 8192`, waiting for `/v1/models`. `--stop`: `docker rm -f turbine-lab-serve turbine-lab-vllm` only. The two configs contain exactly the values below (plus `logging.format: json`):

```yaml
server:
  listen: 0.0.0.0:18000
model:
  path: /models/llama-3.2-3b-instruct # olmoe: /models/olmoe-1b-7b-0125-instruct
  served_name: meta-llama/Llama-3.2-3B-Instruct # olmoe: allenai/OLMoE-1B-7B-0125-Instruct
execution:
  backend: cuda
  device: 0
kv:
  gpu:
    max_bytes: 4GiB
```

- [ ] Run: `bash -n scripts/lab-test.sh scripts/lab-serve.sh scripts/lab/weights-manifest.sh && scripts/lab-serve.sh dgx-spark --dry-run scripts/lab/phase2b-spark-llama.yaml && cargo test -p turbine-bench --test lab_scripts` — expect PASS (exit 0; the dry-run reads `MemAvailable` and `du -sb` over SSH, which is read-only).
- [ ] Run: `for c in scripts/lab/phase2b-spark-*.yaml; do cargo run -q -p turbine-server -- --config "$c" --check-config; done` — expect `config ok` twice.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(scripts): lab-serve.sh Spark modes, phase2b Spark configs and weights manifest`

## Task 9: `kernels/cuda` CMake project, context, memory and error layer

Files: `kernels/cuda/CMakeLists.txt` (project, toolkit lookup, flags, export list, static runtime, build-log versions), `kernels/cuda/cmake/providers.cmake` (FetchContent pins for FlashInfer and CCCL, header presence checks, license comparison), `kernels/cuda/cmake/exports.cmake` (version script generated from the header), `kernels/cuda/third_party/LICENSES/{flashinfer-LICENSE,flashinfer-NOTICE,cccl-LICENSE}` (verbatim copies from the pinned sources), `kernels/cuda/src/shim.h` (internal: `struct turbine_ctx`, status helpers, `TURBINE_API` visibility, exception guard), `kernels/cuda/src/context.cu` (identity, context, stream, allocation, copies, memory info, errors, `turbine_ctx_get_info`), `crates/turbine-kernels/tests/abi_header_neutral.rs` and `crates/turbine-kernels/tests/unsafe_isolation.rs` (P1 files, unchanged; rerun with `kernels/cuda/` present)
Interfaces (C ABI v2, `kernels/include/turbine_kernels.h`, unchanged):

- `uint32_t turbine_abi_version(void)` → `TURBINE_ABI_VERSION` (2); `const char *turbine_backend_name(void)` → `"cuda"`; `const char *turbine_build_archs(void)` → `"sm_121"`
- `int32_t turbine_ctx_create(int32_t device_ordinal, turbine_ctx **out)`, `void turbine_ctx_destroy(turbine_ctx *ctx)`, `int32_t turbine_ctx_get_info(turbine_ctx *ctx, turbine_ctx_info *out)`
- `int32_t turbine_malloc(turbine_ctx *ctx, size_t bytes, void **out)`, `int32_t turbine_free(turbine_ctx *ctx, void *ptr)`, `int32_t turbine_memcpy_h2d(turbine_ctx *ctx, void *dst_device, const void *src_host, size_t bytes)`, `int32_t turbine_memcpy_d2h(turbine_ctx *ctx, void *dst_host, const void *src_device, size_t bytes)`, `int32_t turbine_stream_sync(turbine_ctx *ctx)`, `int32_t turbine_mem_info(turbine_ctx *ctx, size_t *free_bytes, size_t *total_bytes)`, `size_t turbine_last_error(turbine_ctx *ctx, char *buf, size_t len)`
- internal (`shim.h`): `struct turbine_ctx { int device; cudaStream_t stream; cublasLtHandle_t lt; cublasHandle_t blas; void *gemm_workspace; size_t gemm_workspace_bytes /* 32 MiB */; void *scratch; size_t scratch_bytes /* 8 MiB */; void *staging[2]; cudaEvent_t staging_done[2] /* 2 × 32 MiB cudaHostAlloc */; int cc_major, cc_minor; char last_error[512]; }`; `int32_t turbine_cuda_status(turbine_ctx *, cudaError_t, const char *what)`; `int32_t turbine_blas_status(turbine_ctx *, cublasStatus_t, const char *what)`; `int32_t turbine_fail(turbine_ctx *, int32_t code, const char *fmt, ...)`; `TURBINE_GUARD(ctx, body)` catching `std::exception` (FlashInfer's `FLASHINFER_ERROR` throws `flashinfer::Error`, `include/flashinfer/exception.h:23`) → −5

Covers: S-1 (build, identity, static runtime, exports), S-5 (context/stream/thread/error rules), S-6 (allocation and staging), S-9 (Phase 0 Spark discovery check, moved from phase-0 by decision 2026-09-25 — this is the first task that runs on the Sparks); `cargo test -p turbine-kernels --test abi_header_neutral` and `cargo test -p turbine-kernels --test unsafe_isolation` with `kernels/cuda/` present; `scripts/lab-test.sh dgx-spark` and `scripts/lab-test.sh dgx-spark2` inventory check
Depends on: Task 7 (lab image); phase-1 plan (header, `abi_header_neutral`, `unsafe_isolation`), phase-2 plan (ABI v2 header)

- [ ] Write failing test: run the Phase 1 guards against the new tree — `abi_header_neutral` asserts `kernels/include/turbine_kernels.h` has no identifier starting with `hip`, `cuda`, `rocm` or `nv` (case-insensitive) and `unsafe_isolation` asserts no `unsafe` outside `crates/turbine-device/src` and `crates/turbine-kernels/src`; they pass before this task, so the failing check here is the lab configure: `scripts/lab-test.sh dgx-spark` — expect FAIL with `lab-test: step cmake configure failed` (no `kernels/cuda/CMakeLists.txt`).
- [ ] Implement `CMakeLists.txt`: `cmake_minimum_required(VERSION 3.28)`; cache `TURBINE_CUDA_ROOT` (default `/usr/local/cuda`) and set `CMAKE_CUDA_COMPILER` to `${TURBINE_CUDA_ROOT}/bin/nvcc` when present; `project(turbine_cuda LANGUAGES C CXX)`, `include(CheckLanguage)`, `check_language(CUDA)` and `message(FATAL_ERROR "CUDA compiler not found …")` when absent, then `enable_language(CUDA)`; default `CMAKE_CUDA_ARCHITECTURES` `121a` (CMake 3.28's `CMakeCUDAArchitecturesValidate.cmake` accepts `[0-9]+a?`; `nvcc --list-gpu-arch` of 13.0.88 lists `compute_121`); C++17 and CUDA 17; CUDA flags `--expt-relaxed-constexpr -static-global-template-stub=false -DFLASHINFER_ENABLE_BF16 -DNDEBUG -O3` (the flags FlashInfer's own `flashinfer/jit/core.py`/`cpp_ext.py` use; no `FP16_QK_REDUCTION_SUPPORTED`, so `fp16.h` and its Boost include are never pulled in); `add_library(turbine_cuda SHARED src/*.cu)` with `OUTPUT_NAME turbine_cuda`, SONAME `libturbine_cuda.so`, `CUDA_RUNTIME_LIBRARY Static`, `CXX_VISIBILITY_PRESET hidden`, `CUDA_VISIBILITY_PRESET hidden`, link `CUDA::cublasLt CUDA::cublas` from `find_package(CUDAToolkit REQUIRED)`, and `-Wl,--version-script=<generated> -Wl,--exclude-libs,ALL`; `message(STATUS)` lines print `nvcc` version (`CMAKE_CUDA_COMPILER_VERSION`), cuBLAS version (`CUBLAS_VER_MAJOR.MINOR.PATCH.BUILD` parsed from `cublas_api.h:82-85`, 13.1.1.3) and each provider's pinned commit.
- [ ] Implement `providers.cmake`: `FetchContent_Declare(flashinfer GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git GIT_TAG 8bc3b578027791336c6ae87db5c9d76f82cef8bc GIT_SUBMODULES "" SOURCE_SUBDIR turbine-headers-only)` and the same for CCCL at `876867684f7fac130e0f5911236e0a92a970d4fd`, so `FetchContent_MakeAvailable` only populates sources; include order CCCL (`libcudacxx/include`, `cub`, `thrust`) before the toolkit so `<cuda/cmath>` resolves to CCCL 3.3.2; `FATAL_ERROR` if `attention/prefill.cuh`, `attention/decode.cuh`, `norm.cuh`, `page.cuh` or CCCL's `cuda/__cmath/fast_modulo_division.h` is missing; compare `file(SHA256)` of each committed license with the fetched `LICENSE`/`NOTICE` and fail on a difference. `exports.cmake` regex-extracts every `turbine_[a-z0-9_]+` followed by `(` from the header into `{ global: …; local: *; };`.
- [ ] Implement `context.cu`: `turbine_ctx_create` → `cudaSetDevice(ordinal)` (`cuda_runtime_api.h:1659`), `cudaDeviceGetAttribute` for `cudaDevAttrComputeCapabilityMajor/Minor` (`driver_types.h:1993-1994`; refuse anything but 12.1 with −2 and `device <n> has compute capability <M>.<m> (sm_<Mm>); libturbine_cuda.so is built for sm_121`), `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` (`:1907`), `cublasLtCreate`, `cublasCreate` + `cublasSetStream_v2` + `cublasSetWorkspace_v2` (`cublas_api.h:256-258`), `cudaMalloc` of the 32 MiB GEMM workspace and 8 MiB scratch (`:4581`), two 32 MiB `cudaHostAlloc(…, cudaHostAllocDefault)` staging halves (`:4887`) with events; every entry point starts with `cudaSetDevice(ctx->device)`; `turbine_malloc` = `cudaMalloc` only (no `cudaMallocManaged`, no stream-ordered pools); `turbine_memcpy_h2d` issues `cudaMemcpyAsync` (`:6332`) on the context stream directly for copies under 1 MiB and, for larger ones, copies 32 MiB chunks into the alternating pinned halves after `cudaEventSynchronize` on that half's previous copy (OPEN-1); `turbine_memcpy_d2h` = `cudaMemcpyAsync` on the stream; `turbine_stream_sync` = `cudaStreamSynchronize` (`:2320`); `turbine_mem_info` = `cudaMemGetInfo` (`:5752`); `turbine_ctx_get_info` reports `workspace_bytes` 41,943,040 (GEMM 32 MiB + scratch 8 MiB), the compute capability and `device_arch` `sm_121`. Status mapping: `cudaErrorMemoryAllocation` (2) → −3, any other `cudaError_t` → −4, cuBLAS/cuBLASLt status → −5, message `"<cudaGetErrorName(e)>: <cudaGetErrorString(e)> (<call>)"` (`:1232`, `:1248`) or `"<cublasLtGetStatusName(s)>: <call>"` (`cublasLt.h:73`); a failed `turbine_ctx_create` stores its message in a `thread_local` buffer read by `turbine_last_error(NULL, …)`; an old driver surfaces as `cudaErrorInsufficientDriver: …` (−4) from the first runtime call.
- [ ] Run: `cargo test -p turbine-kernels --test abi_header_neutral && cargo test -p turbine-kernels --test unsafe_isolation` — expect PASS; then `scripts/lab-test.sh dgx-spark` — expect the configure/build steps to pass with log lines `-- turbine: nvcc 13.0.88`, `-- turbine: cuBLAS 13.1.1.3`, `-- turbine: FlashInfer commit 8bc3b578027791336c6ae87db5c9d76f82cef8bc (v0.6.18.post1)` and `-- turbine: CCCL commit 876867684f7fac130e0f5911236e0a92a970d4fd (v3.3.2)` (op symbols arrive in Tasks 10–13; the link succeeds because nothing references them yet). Not compile-verifiable off the Spark: CUDA sources are built only inside the lab image.
- [ ] Lab (Phase 0 Spark discovery check, moved from phase-0 by decision 2026-09-25): for each of `dgx-spark` and `dgx-spark2`, `scripts/lab-test.sh <host>` first prints `precondition: ok` (the `MemAvailable` ≥ 24 GiB pre-check of Task 7; on `precondition: MemAvailable … ask the user to free memory`, ASK THE USER FIRST and rerun only after they confirm — never stop production vLLM yourself), then expect exit 0 with `lab-inventory: index=0 vendor=nvidia name="NVIDIA GB10" arch=sm_121 memory.kind=unified total_bytes=<non-zero>` and `test inventory_matches_expectation ... ok` (`TURBINE_EXPECT_NVIDIA=1`); paste both inventory lines into the task evidence.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels-cuda): CMake project, pinned FlashInfer/CCCL and CUDA context layer`

## Task 10: Dense Phase 1 ops on CUDA (GEMM, RMSNorm, elementwise)

Files: `kernels/cuda/src/gemm.cu` (cuBLASLt GEMM trio with heuristic cache), `kernels/cuda/src/norm.cu` (FlashInfer RMSNorm trio), `kernels/cuda/src/elementwise.cu` (Turbine kernels: rope, silu_mul, embedding, add trios), `crates/turbine-kernels/tests/ops_common/mod.rs` (new: backend-parametric helpers moved out of `hip_ops.rs`), `crates/turbine-kernels/tests/hip_ops.rs` (P1/P2 file; use `ops_common`), `crates/turbine-kernels/tests/cuda_ops.rs` (new: first tests)
Interfaces:

- C ABI trios for `gemm`, `rmsnorm`, `rope`, `silu_mul`, `embedding`, `add`: `int32_t turbine_<op>(turbine_ctx *ctx, const turbine_<op>_desc *d)`, `int32_t turbine_<op>_supported(const turbine_<op>_desc *d)`, `const char *turbine_<op>_impl(const turbine_<op>_desc *d)`; impl strings `cublaslt` (gemm), `flashinfer_rmsnorm` (rmsnorm), `turbine_cuda` (rope, silu_mul, embedding, add)
- Rust test helpers (`ops_common`): `pub struct Pair { pub gpu: Arc<dyn KernelProvider>, pub cpu: Arc<dyn KernelProvider>, pub gpu_mem: Arc<dyn DeviceMemory>, pub cpu_mem: Arc<dyn DeviceMemory> }`, `pub fn setup(backend: ExecutionBackend) -> Pair` (loads `TURBINE_KERNEL_LIBRARY`, discovers the first device of the backend's vendor, `shim_provider(ctx)` vs `cpu_reference_provider()`), and the moved P1/P2 case functions with unchanged bodies: `gemm_case`, `attention_case` (gains `heads: (usize, usize)`), `norm_rope_silu_embedding_add_cases`, `paged_attention_case`, `copy_blocks_case`, `moe_route_case`, `moe_experts_case`, `assert_close`
- tolerance (spec AC): BF16 outputs max |Δ| ≤ 1e-2, FP32 outputs ≤ 1e-4

Covers: S-3 (dense ops), S-14 (impl names); no acceptance criterion on its own (Task 11 runs the lab criterion)
Depends on: Task 9; phase-1/phase-2 plans (`hip_ops.rs` cases, `cpu_reference_provider`, `shim_provider`)

- [ ] Write failing tests in `cuda_ops.rs`: `gemm_matches_cpu` (m ∈ {1, 17}; (n,k) ∈ {(3072,3072), (1024,3072), (8192,3072), (3072,8192)} BF16 out, plus the LM head 128256×3072 with F32 out) and `norm_rope_silu_embedding_add_match_cpu` (Llama-3 scaled and standard rope, OLMoE widths 2048/1024), each `#[ignore]` and starting with `if !require_backend("cuda") { return; }`; `hip_ops.rs` keeps its tests calling the same `ops_common` cases with `require_backend("hip")`. Run: `scripts/lab-test.sh dgx-spark` — expect FAIL (`gemm_matches_cpu` panics: `cuda must support gemm`).
- [ ] Implement `gemm.cu` with cuBLASLt (`cublasLt.h` lines checked on dgx-spark): row-major `c[m,n] = a[m,k]·op(b)` is computed column-major as `Cᵀ(n×m) = op(Bᵀ)·Aᵀ` — `cublasLtMatrixLayoutCreate` (`:1212`) A-operand = b (`CUDA_R_16BF`, rows k, cols n, ld ldb, `CUBLASLT_MATMUL_DESC_TRANSA = CUBLAS_OP_T` when `trans_b = 1`, else rows n, cols k, `CUBLAS_OP_N`), B-operand = a (rows k, cols m, ld lda, `CUBLAS_OP_N`), C = D = c (`CUDA_R_16BF` or `CUDA_R_32F`, rows n, cols m, ld ldc); `cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)` (`:1623`); preference `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` = 32 MiB (`:2088`); `cublasLtMatmulAlgoGetHeuristic(…, 1, &result, &count)` (`:2270`); `cublasLtMatmul(…, &result.algo, ctx->gemm_workspace, 32 MiB, ctx->stream)` (`:1011`). Heuristic results are cached per context in a map keyed by (m, n, k, lda, ldb, ldc, trans_b, dtypes) bounded to 4096 entries (oldest evicted). `turbine_gemm_supported` runs the heuristic on a process-wide `cublasLtHandle_t` created once with `std::call_once` and returns 0 when `count == 0` or the status is not success, so the registry falls through (spec edge case); dtypes other than BF16 in / BF16 or F32 out return 0.
- [ ] Implement `norm.cu`: `flashinfer::norm::RMSNorm<nv_bfloat16>(x, weight, out, rows, dim, x_stride_row, out_stride_row, eps, false, ctx->stream)` (`include/flashinfer/norm.cuh:140`; pointer arguments are non-const, so `const_cast` the inputs; the kernel multiplies by `weight`, the Llama/OLMoE form); `_supported` requires BF16, `dim % 8 == 0`, both strides `% 8 == 0` (its vector width is `gcd(16 / sizeof(T), d)`) and `dim ≤ 8192`.
- [ ] Implement `elementwise.cu` Turbine kernels (one thread block per token row, FP32 math, BF16 I/O): `rope` in place on q and k with the host-computed `inv_freq` table (`style 0` = HF `rotate_half`, angle `positions[t] * inv_freq[i]` in FP32, `rotary_dim ≤ head_dim`), `silu_mul` (`out = silu(gate) * up`), `embedding` (row `ids[t] - vocab_offset`, zeros outside `[0, vocab_rows)`), `add` (elementwise, `n` elements); each launch is followed by `cudaGetLastError` mapped through `turbine_cuda_status`.
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect `test gemm_matches_cpu ... ok` and `test norm_rope_silu_embedding_add_match_cpu ... ok` for `cuda_ops`; locally `cargo test --workspace` — expect PASS (the ignored tests do not run on macOS).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels-cuda): cuBLASLt GEMM, FlashInfer RMSNorm and Turbine elementwise kernels`

## Task 11: Single-request attention through FlashInfer and the Phase 1 lab gate

Files: `kernels/cuda/src/attention.cu` (FlashInfer single prefill/decode trios, explicit template instantiations), `crates/turbine-kernels/tests/cuda_ops.rs` (attention test)
Interfaces:

- `int32_t turbine_attention_prefill(turbine_ctx *, const turbine_attention_prefill_desc *)` + `_supported` + `_impl` (→ `flashinfer_prefill`); `int32_t turbine_attention_decode(turbine_ctx *, const turbine_attention_decode_desc *)` + `_supported` + `_impl` (→ `flashinfer_decode`)
- FlashInfer entry points (pinned commit): `flashinfer::SinglePrefillWithKVCacheDispatched<128, 128, PosEncodingMode::kNone, false, MaskMode::kCausal, DefaultAttention<false, false, false, false>, SinglePrefillParams<nv_bfloat16, nv_bfloat16, nv_bfloat16>>(params, nullptr, stream)` (`attention/prefill.cuh:2517`); `flashinfer::SingleDecodeWithKVCacheDispatched<128, PosEncodingMode::kNone, DefaultAttention<false, false, false, false>, SingleDecodeParams<nv_bfloat16, nv_bfloat16, nv_bfloat16>>(params, nullptr, stream)` (`attention/decode.cuh:660`)

Covers: S-1, S-3, S-9; `scripts/lab-test.sh dgx-spark` exits 0 with the CMake build log and `cargo test -p turbine-kernels --test cuda_ops -- --ignored` passing (every Phase 1 op; attention GQA 24/8 and 16/16 at 1, 17, 512, 4096)
Depends on: Task 10

- [ ] Write failing test `cuda_ops attention_matches_cpu`: for heads (24, 8) and (16, 16) and s ∈ {1, 17, 512, 4096}, `attention_case(Prefill, q_len = s, q_start = 0)` and `attention_case(Decode, q_len = 1, q_start = s − 1)`, plus a prefill of 17 rows at `q_start` 100, each compared with the CPU reference on row blocks (|Δ| ≤ 1e-2). Run: `scripts/lab-test.sh dgx-spark` — expect FAIL (`cuda must support attention_prefill head_dim=128 kv_heads=8 dtype=bf16`).
- [ ] Implement prefill: `SinglePrefillParams(q, k_cache, v_cache, nullptr, out, nullptr, nullptr, num_q_heads, num_kv_heads, q_len, q_start + q_len, q_stride_token, head_dim, kv_stride_token, head_dim, 128, -1, 0.f, scale, 1.f, 1e4f)` (`default_prefill_params.cuh:88`); FlashInfer's causal mask aligns the query block to the end of the KV rows, which is exactly Turbine's `q_start` offset, and `kv_len < qo_len` is refused by FlashInfer (`prefill.cuh:2526`); `tmp = nullptr` disables KV partitioning (`prefill.cuh:2679`), so no extra workspace is needed. Decode: `SingleDecodeParams(q, k, v, out, nullptr, q_start + 1, num_q_heads, num_kv_heads, QKVLayout::kNHD, 128, -1, 0.f, scale, 1.f, 1e4f)` (`default_decode_params.cuh:75`), then overwrite `q_stride_n` and `kv_stride_n` with the descriptor strides; `tmp = nullptr` (`decode.cuh:691` no-partition branch). Both calls run inside `TURBINE_GUARD` and map the returned `cudaError_t`.
- [ ] Implement `_supported` for both: BF16, `head_dim == 128`, `causal == 1`, `num_q_heads % num_kv_heads == 0` (the GQA ratio is a runtime `uint_fastdiv group_size`, not a template parameter, so 3 and 1 both run the same instantiation), `out_stride_token == num_q_heads * head_dim` (FlashInfer writes O contiguously), `q_len == 1` for decode. Only the two instantiations above are compiled.
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect exit 0 with `-- turbine: FlashInfer commit 8bc3b578…` in the CMake log, `test attention_matches_cpu ... ok` in `cuda_ops`, and no `SKIP backend=` line from `cuda_ops`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels-cuda): FlashInfer single prefill and decode attention`

## Task 12: Paged attention and `copy_blocks` on CUDA

Files: `kernels/cuda/src/paged.cu` (device-side plan kernel, KV append, FlashInfer batch prefill/decode trios), `kernels/cuda/src/copy_blocks.cu` (Turbine kernel trio), `crates/turbine-kernels/tests/cuda_ops.rs` (paged cases, first half of `paged_and_moe_ops`)
Interfaces:

- `turbine_attention_prefill_paged` / `turbine_attention_decode_paged` trios (impl `flashinfer_batch_prefill_paged` / `flashinfer_batch_decode_paged`), `turbine_copy_blocks` trio (impl `turbine_cuda`)
- internal: `__global__ void turbine_paged_plan_kernel(const int32_t *block_table, const int32_t *q_indptr, const int32_t *kv_lens, int32_t num_seqs, int32_t max_blocks_per_seq, int32_t block_tokens, int32_t group, int32_t cta_tile_q, PagedPlan plan)` writing into `ctx->scratch`: `kv_indptr[num_seqs+1]`, `kv_indices[num_seqs*max_blocks_per_seq]`, `last_page_len[num_seqs]`, `append_batch[total_q]`, `append_pos[total_q]`, `request_indices`, `qo_tile_indices`, `kv_tile_indices`, `block_valid_mask[padded]`, `kv_chunk_size[1]`
- FlashInfer (pinned): `paged_kv_t<nv_bfloat16, int32_t>(num_kv_heads, block_tokens, 128, num_seqs, QKVLayout::kNHD, pool, pool + block_tokens*num_kv_heads*128, kv_strides, kv_indices, kv_indptr, last_page_len)` (`page.cuh:136`, the `kv_strides` constructor), `AppendPagedKVCache(paged_kv, k_new, v_new, append_batch, append_pos, total_q, new_stride_token, 128, new_stride_token, 128, stream)` (`page.cuh:465`), `BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q ∈ {16, 128}, 128, 128, PosEncodingMode::kNone, false, MaskMode::kCausal, DefaultAttention<false, false, false, false>, BatchPrefillPagedParams<nv_bfloat16, nv_bfloat16, nv_bfloat16, int32_t>>(params, nullptr, nullptr, false, stream)` (`prefill.cuh:4335`), `BatchDecodeWithPagedKVCacheDispatched<128, PosEncodingMode::kNone, DefaultAttention<false, false, false, false>, BatchDecodeParams<nv_bfloat16, nv_bfloat16, nv_bfloat16, int32_t>>(params, nullptr, nullptr, false, stream)` (`decode.cuh:743`)

Covers: S-4 (paged attention, copy_blocks); no acceptance criterion on its own (Task 13 runs `paged_and_moe_ops`)
Depends on: Task 11; phase-2 plan (paged descriptor, pool layout `[num_blocks, 2, block_tokens, kv_heads, head_dim]`)

- [ ] Write failing test cases in `cuda_ops paged_and_moe_ops` (paged half): ragged batches over the Phase 2 pool with `block_tokens` 16 for heads (24, 8) and (16, 16): (a) three 1-token decodes plus one 2,048-token prefill chunk whose sequences end in partially filled last blocks; (b) four prefill chunks of 17, 1, 33 and 512 tokens after existing context; plus `copy_blocks_case` forking 5 blocks across 4 layers; all against the CPU reference (|Δ| ≤ 1e-2, pool bytes after append identical to the reference's). Run: `scripts/lab-test.sh dgx-spark` — expect FAIL (`cuda must support attention_prefill_paged`).
- [ ] Implement the plan: FlashInfer's host planners (`scheduler.cuh` `PrefillPlan`/`DecodePlan`) need host copies of the indptr arrays, which the ABI v2 paged descriptor does not carry (device pointers only), and a device-to-host copy plus sync inside an op would break S-5 — so one Turbine kernel builds FlashInfer's CSR page table (`kv_indptr[i] = Σ ceil(kv_lens[j]/block_tokens)`, `kv_indices` gathered from `block_table`, `last_page_len = kv_len − (pages − 1)·block_tokens`), the per-token append coordinates (`append_batch[t] = seq`, `append_pos[t] = kv_lens[seq] − q_len(seq) + (t − q_indptr[seq])`) and the tile arrays without KV partitioning (`kv_tile_indices = 0`, `kv_chunk_size = max_kv_len`); the grid uses the host-known bound `padded = ceil(total_q · group / CTA_TILE_Q) + num_seqs` and `block_valid_mask` marks the unused tail (FlashInfer skips those CTAs, `prefill.cuh:3594`). The pool is passed as-is: `k_data = pool`, `v_data = pool + block_tokens·kv_heads·128`, `kv_strides = {2·block_tokens·kv_heads·128, kv_heads·128, 128}` in NHD, which is FlashInfer's paged layout with a doubled page stride (C-23: no layout change).
- [ ] Implement the attends: prefill picks `CTA_TILE_Q = 128` when `total_q · group / num_seqs > 64`, else 16 (FlashInfer's `FA2DetermineCtaTileQ` rule for head_dim 128, `utils.cuh:408`); decode sets `padded_batch_size = num_seqs`, `request_indices[i] = i`; both use `BatchPrefillPagedParams`/`BatchDecodeParams` constructors with `q_stride_n = q_stride_token`, `q_stride_h = 128`, `window_left = −1`, `sm_scale = scale`, then attach the plan arrays; `tmp_v = nullptr` keeps `partition_kv = false`. Order on the stream: plan kernel → `AppendPagedKVCache` → attend. `_supported`: BF16, head_dim 128, `block_tokens == 16` (other values refused naming the value, per spec edge case), causal, contiguous output, plan size ≤ `ctx->scratch_bytes` computed from `num_seqs`, `max_blocks_per_seq` and `total_q` (else 0 naming the sizes).
- [ ] Implement `copy_blocks.cu`: the host `src_blocks`/`dst_blocks` arrays are packed into `int2` pairs and copied with `cudaMemcpyAsync` from pageable memory into `ctx->scratch` (the runtime stages pageable host-to-device copies immediately, so the host arrays may be freed on return), then one kernel copies `block_bytes` per (layer, pair) with 16-byte vectors; `_supported` requires `count · 8 ≤ scratch_bytes` and `block_bytes % 16 == 0`.
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect the paged and `copy_blocks` cases of `paged_and_moe_ops` to pass (the MoE half fails until Task 13 with `cuda must support moe_route`).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels-cuda): FlashInfer batch paged attention with a device-side plan and copy_blocks`

## Task 13: MoE routing and experts on CUDA and the paged/MoE lab gate

Files: `kernels/cuda/src/moe.cu` (Turbine `moe_route` kernels; `moe_experts` with grouped-GEMM and per-expert paths; gather, SiLU-mul and weighted combine kernels), `kernels/cuda/src/context.cu` (grouped-GEMM probe at context creation), `crates/turbine-kernels/tests/cuda_ops.rs` (MoE half of `paged_and_moe_ops`)
Interfaces:

- `turbine_moe_route` trio (impl `turbine_cuda`); `turbine_moe_experts` trio (impl `cublas_grouped_batched`, or `cublaslt_per_expert` when the probe or the shape rules out the grouped call)
- cuBLAS (`cublas_api.h:5032`, dgx-spark): `cublasGemmGroupedBatchedEx(cublasHandle_t, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const void *alpha_array, const void *const Aarray[], cudaDataType_t Atype, const int lda_array[], const void *const Barray[], cudaDataType_t Btype, const int ldb_array[], const void *beta_array, void *const Carray[], cudaDataType_t Ctype, const int ldc_array[], int group_count, const int group_size[], cublasComputeType_t computeType)` with `CUDA_R_16BF` (`library_types.h:61`) and `CUBLAS_COMPUTE_32F` (`cublas_api.h:208`), float `alpha`/`beta` arrays on the host
- internal: `static std::atomic<int> g_grouped_ok` (−1 unknown, 0 no, 1 yes) set by `turbine_probe_grouped(turbine_ctx *)` during `turbine_ctx_create`
- workspace rule for `turbine_moe_experts_desc.workspace` (Contract additions): `rows · (2·hidden + 3·inter) · 2` bytes plus 5 × 256 alignment bytes plus `8 · rows` bytes of inverse permutation, where `rows = host_expert_offsets[expert_end] − host_expert_offsets[expert_begin]`; smaller → −1 naming both sizes

Covers: S-4 (MoE), S-9; `cargo test -p turbine-kernels --test cuda_ops paged_and_moe_ops -- --ignored` inside `scripts/lab-test.sh dgx-spark` (exit 0)
Depends on: Task 12; phase-2 plan (`moe_route`/`moe_experts` descriptors, CPU reference, tie rule)

- [ ] Write failing test cases in `cuda_ops paged_and_moe_ops` (MoE half), OLMoE shapes (hidden 2048, inter 1024, 64 experts, top-8, `renormalize = 0`): 37 tokens with seeded logits (selections identical to the CPU reference, weights |Δ| ≤ 1e-4); 16 tokens whose logits make every token pick experts 0–7 (ties broken by lower id); 9 tokens where expert 5 receives no token (an empty group); each followed by `moe_experts` compared with the CPU reference (|Δ| ≤ 1e-2 on the accumulated output). Run: `scripts/lab-test.sh dgx-spark` — expect FAIL (`cuda must support moe_route`).
- [ ] Implement `moe_route`: kernel 1 (one warp per token) computes an FP32 softmax over `num_experts` and a top-k by repeated arg-max where equal values keep the lower expert id, writing `topk_ids`/`topk_weights` (renormalised only when `renormalize = 1`); kernel 2 (one block, one thread per expert) writes `expert_offsets` by counting and an exclusive scan, then each expert thread appends its rows `token·top_k + slot` in ascending order into `sorted_rows` — stable and deterministic, matching the HIP kernel and the CPU reference.
- [ ] Implement `moe_experts`: host group sizes come from `host_expert_offsets` (no device read); a gather kernel copies `x[row / top_k]` for rows in `[offsets[expert_begin], offsets[expert_end])` into the workspace and records the inverse permutation; gate and up projections run as one `cublasGemmGroupedBatchedEx` call with one group (group size 1) per non-empty expert and per projection, using the row-major trick of Task 10 (`transa = CUBLAS_OP_T` on the `[inter, hidden]` weight, `transb = CUBLAS_OP_N` on the gathered `[rows_e, hidden]` activations, `m = inter`, `n = rows_e`, `k = hidden`); the `Aarray`/`Barray`/`Carray` pointer arrays are device-resident, so they are built on the host and copied with `cudaMemcpyAsync` into `ctx->scratch` before the call; SiLU-mul kernel; the down projection as a second grouped call; a combine kernel adds `Σ_slot topk_weights · down[inverse[row]]` into `out` per token in slot order (deterministic, no atomics). The per-expert path issues the same GEMMs one by one through the Task 10 cuBLASLt routine. `turbine_probe_grouped` runs a 2-group 16×16×16 BF16 grouped call on the scratch at context creation and stores the result; `_impl` returns `cublas_grouped_batched` when the probe passed and every size fits `int`, else `cublaslt_per_expert`; a grouped call that fails at run time returns −5 with `cublasGetStatusName` (`cublas_api.h:313`).
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect exit 0 with `test paged_and_moe_ops ... ok` in `cuda_ops`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels-cuda): MoE routing and cublasGemmGroupedBatchedEx experts with per-expert fallback`

## Task 14: Export check and the second Spark

Files: `crates/turbine-kernels/src/test_support.rs` (header parser and symbol resolver helpers with unit tests), `crates/turbine-kernels/tests/cuda_ops.rs` (`exports_match_header`)
Interfaces:

- produces `pub fn header_functions(header: &str) -> Vec<String>` (every `turbine_*` identifier directly followed by `(`, declaration order, no duplicates, ignores identifiers glued to a preceding identifier)
- produces `pub fn unresolved_symbols(library: &Path, names: &[String]) -> Result<Vec<String>, String>` (loads with `libloading`, resolves each name, never calls one)
- consumes `nm -D --defined-only` from `binutils` in the lab image (Task 7)

Covers: S-1, S-9; `cargo test -p turbine-kernels --test cuda_ops exports_match_header -- --ignored` inside `scripts/lab-test.sh dgx-spark`; `scripts/lab-test.sh dgx-spark2` exits 0 with the same `cuda_ops` results
Depends on: Task 13

- [ ] Write failing tests: `test_support::tests::header_functions_lists_declarations_only` (a header snippet with a typedef, a comment mentioning `turbine_last_error` and two declarations yields exactly `turbine_abi_version`, `turbine_gemm`, `turbine_gemm_impl`) and `test_support::tests::stub_exports_every_header_function` (the real header yields ≥ 52 names and `stub_cuda_sm121` resolves all of them); `cuda_ops exports_match_header` (ignored, `require_backend("cuda")`): every `header_functions` name resolves from `TURBINE_KERNEL_LIBRARY` and every symbol printed by `nm -D --defined-only` (excluding the `_init`/`_fini` and version-node entries) starts with `turbine_`. Run: `cargo test -p turbine-kernels test_support::tests` — expect FAIL (`header_functions` not found).
- [ ] Implement both helpers (`unsafe` only around `libloading::Library::new` and `Library::get`, each with `// SAFETY:`); the export side is already enforced by the Task 9 version script and `--exclude-libs,ALL`, which keeps the static CUDA runtime and CCCL/FlashInfer template symbols local.
- [ ] Run: `cargo test -p turbine-kernels test_support::tests` — expect PASS; then `scripts/lab-test.sh dgx-spark` — expect exit 0 with `test exports_match_header ... ok`.
- [ ] Run: `scripts/lab-test.sh dgx-spark2` — expect `precondition: ok` (its `MemAvailable` was 29 GiB on 2026-09-25; if it prints `precondition: MemAvailable … < 24.0 GiB; ask the user to free memory`, ASK THE USER FIRST and rerun only after they confirm), the image tag equal to dgx-spark's (`spark_image_tag` hashes the same Dockerfile), and exit 0 with every `cuda_ops` test `ok`; paste both hosts' `cuda_ops` result lines into the task evidence.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(turbine-kernels): check libturbine_cuda.so exports against the header`

## Task 15: Tiny models on CUDA (`cuda_matches_cpu`)

Files: `crates/turbine-model/tests/tiny_model.rs` (P1/P2 file; new ignored test)
Interfaces:

- consumes `turbine_model::testing::tiny::{write_tiny_llama, write_tiny_olmoe}(dir: &Path, seed: u64) -> TinySpec` (contract §10), the P1/P2 executor constructor over a `KernelRegistry`, and `ShimLibrary::load(path, ExecutionBackend::Cuda)` with `TURBINE_KERNEL_LIBRARY`
- tolerance: logits max |Δ| ≤ 2e-2 vs the CPU provider; 32 greedy tokens identical

Covers: S-3, S-8; `cargo test -p turbine-model --test tiny_model cuda_matches_cpu -- --ignored` inside `scripts/lab-test.sh dgx-spark`
Depends on: Task 14; phase-1/phase-2 plans (`hip_matches_cpu`, chunked prefill, paged decode)

- [ ] Write failing test `tiny_model cuda_matches_cpu` (`#[ignore]`, `require_backend("cuda")`): for the tiny Llama and tiny OLMoE checkpoints (seed 7), run 4 sequences through prefill, chunked prefill (chunk 5) and paged decode on the CUDA provider and on the CPU provider; assert max |Δ| logits ≤ 2e-2 at every step and identical 32 greedy tokens per sequence. The body reuses `hip_matches_cpu`'s driver with the backend as a parameter. Run: `cargo test -p turbine-model --test tiny_model cuda_matches_cpu -- --ignored` on the workstation with `TURBINE_TEST_BACKEND=cuda` — expect FAIL (no `TURBINE_KERNEL_LIBRARY` / NVIDIA device on macOS: the setup panics naming the variable).
- [ ] Implement: make the shared driver take `ExecutionBackend` and the vendor to discover; nothing else changes — the tiny checkpoints use head_dim 128 like the real models (OPEN-2), so the FlashInfer instantiations of Tasks 11–12 cover them.
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect exit 0 with `test cuda_matches_cpu ... ok`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(turbine-model): tiny Llama and OLMoE match the CPU provider on CUDA`

## Task 16: Weights on both Sparks and cross-host manifests

Files: no repository files (weights live in `/home/piwi/turbine-models/<slug>` on the hosts; manifests and download log go into the task evidence)
Interfaces:

- consumes `scripts/lab/weights-manifest.sh <host> <slug>` (Task 8), the novanas copies and revisions recorded by the phase-1/phase-2 plans
- slugs: `llama-3.2-3b-instruct` (`meta-llama/Llama-3.2-3B-Instruct`), `olmoe-1b-7b-0125-instruct` (`allenai/OLMoE-1B-7B-0125-Instruct`)

Covers: S-10; weights-manifest diff across dgx-spark, dgx-spark2 and novanas for both slugs, and `git grep -n "hf_[A-Za-z0-9]\{30,\}"` exits 1
Depends on: Task 8; phase-1/phase-2 plans (novanas weights and revisions)

- [ ] Write failing check: `scripts/lab/weights-manifest.sh dgx-spark llama-3.2-3b-instruct` — expect FAIL (`cd: /home/piwi/turbine-models/llama-3.2-3b-instruct: No such file or directory`; verified absent on 2026-09-25).
- [ ] ASK THE USER for a Hugging Face token (read access to `meta-llama/Llama-3.2-3B-Instruct`); it is passed to the remote shell on stdin for this one download only — never on a command line, never written to the repository, a script, a container environment or a file on the host.
- [ ] Implement the download on each Spark over SSH with the host's `curl` only (no package, Python or `hf` CLI installed on the host): the file list and revision of each model come from the novanas manifest and the Phase 1/2 records; the remote command reads the token with `read -r HF_TOKEN`, creates `/home/piwi/turbine-models/<slug>` and fetches every file with `curl -fL --retry 3 -H "Authorization: Bearer $HF_TOKEN" -o <file> https://huggingface.co/<HF id>/resolve/<revision>/<file>`; the token variable dies with that shell.
- [ ] Run: for each slug, `scripts/lab/weights-manifest.sh dgx-spark <slug> > /tmp/spark.txt`, `scripts/lab/weights-manifest.sh dgx-spark2 <slug> > /tmp/spark2.txt`, `scripts/lab/weights-manifest.sh novanas <slug> > /tmp/novanas.txt`, then `diff /tmp/spark.txt /tmp/spark2.txt && diff /tmp/spark.txt /tmp/novanas.txt` — expect PASS (exit 0), pasting the three outputs into the evidence; a difference means re-downloading on that host (asking for the token again) before any golden run there.
- [ ] Run: `git grep -n "hf_[A-Za-z0-9]\{30,\}"` — expect exit 1 (no token in the repository).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: none (no repository change; evidence only)

## Task 17: Golden parity in-process on GB10

Files: `crates/turbine-model/tests/golden.rs` (P1/P2 file; backend from `TURBINE_TEST_BACKEND`, both fixture directories)
Interfaces:

- consumes `turbine_kernels::test_support::backend_under_test() -> ExecutionBackend` (Task 1), `require_env_dir("TURBINE_TEST_MODEL_DIR")`, `require_env_dir("TURBINE_TEST_MOE_MODEL_DIR")`, the P1 tolerance loader for `tests/golden/<slug>/tolerance.json`
- consumes the Task 4 `memory_budget` record (captured by the test's tracing layer)

Covers: S-6, S-11; `cargo test -p turbine-model --test golden logits_match_reference -- --ignored` inside `scripts/lab-test.sh dgx-spark` with `TURBINE_TEST_BACKEND=cuda`
Depends on: Tasks 15, 16

- [ ] Write failing test change `golden logits_match_reference`: replace the hard-coded `hip` provider with `backend_under_test()` (library from `TURBINE_KERNEL_LIBRARY`, device = first inventory device of the backend's vendor), run both `tests/golden/llama-3.2-3b-instruct/` and `tests/golden/olmoe-1b-7b-0125-instruct/` under their committed `tolerance.json` (≥ 14/16 prompts with the first 32 greedy tokens identical, top-5 |Δ logprob| ≤ 0.15 nats), and assert a captured `memory_budget` record has `device_kind` `unified` and `available_bytes` equal to `min(device_free_bytes, host_mem_available_bytes)` when the device is unified. Run: `scripts/lab-test.sh dgx-spark` — expect FAIL before the change (`hip` library requested on a CUDA host).
- [ ] Implement: the test selects `ExecutionBackend::Cuda` → `Vendor::Nvidia` and `Hip` → `Amd`; the budget assertion is skipped for `dedicated` devices so novanas keeps passing; nothing is re-captured.
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect exit 0 with `test logits_match_reference ... ok` and a log record `"event":"memory_budget"` with `"device_kind":"unified"`; paste the per-prompt summary lines into the evidence.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(turbine-model): golden parity on the backend named by TURBINE_TEST_BACKEND`

## Task 18: HIP path still green on novanas

Files: `crates/turbine-kernels/tests/hip_ops.rs` (P1/P2 file, already on `ops_common` since Task 10; no further change), `scripts/lab/novanas-test-job.yaml` (P0 file; confirm `TURBINE_TEST_BACKEND=hip` is set)
Interfaces:

- consumes `require_backend("hip")` / `require_backend("cuda")` in every ignored GPU test (Tasks 10–15)
- `scripts/lab-test.sh novanas` (P0/P1 k3s Job path, CONFLICT C-24)

Covers: S-8; `scripts/lab-test.sh novanas` still exits 0 with every `cuda_ops`/`cuda_matches_cpu` test printing `SKIP backend=hip` and every `hip_ops` test running
Depends on: Task 17

- [ ] Write failing check: `grep -n TURBINE_TEST_BACKEND scripts/lab/novanas-test-job.yaml` — expect FAIL (exit 1) if the Phase 0–2 Job manifest does not set it yet; in that case every GPU test there would panic naming the variable.
- [ ] Implement: add `- name: TURBINE_TEST_BACKEND` / `value: hip` to the Job's container `env` next to `TURBINE_KERNEL_LIBRARY`.
- [ ] Run: `scripts/lab-test.sh novanas` — expect exit 0, `SKIP backend=hip` printed by `gemm_matches_cpu`, `norm_rope_silu_embedding_add_match_cpu`, `attention_matches_cpu`, `paged_and_moe_ops` and `exports_match_header` of `cuda_ops` and by `cuda_matches_cpu`, and every `hip_ops` test `ok` without `SKIP`. The R9700s must be free of the `kuvryn-ai-workloads` pod per the Phase 0 decision; if they are not, ASK THE USER FIRST.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(scripts): run novanas GPU tests with TURBINE_TEST_BACKEND=hip`

## Task 19: Above-kernel features on CUDA (`lab_openai`)

Files: `crates/turbine-server/tests/lab_openai.rs` (P2 file; backend and library from the environment)
Interfaces:

- consumes `backend_under_test()` (Task 1), `require_env_dir("TURBINE_TEST_MODEL_DIR")`, `TURBINE_KERNEL_LIBRARY`, the committed `tests/golden/tools/*.jsonl` (P2)
- server config written by the test: `execution.backend` = `backend_under_test().as_str()`, `execution.kernel_library` = `$TURBINE_KERNEL_LIBRARY`, `server.listen: 127.0.0.1:0`

Covers: S-12; `cargo test -p turbine-server --test lab_openai tools_and_json_schema -- --ignored` inside `scripts/lab-test.sh dgx-spark`
Depends on: Task 17; phase-2 plan (`lab_openai tools_and_json_schema`, preemption, llguidance, `llama3_json` parser, graceful shutdown)

- [ ] Write failing test change `lab_openai tools_and_json_schema`: start the server with the backend from `backend_under_test()` instead of `hip`, keeping the P2 assertions — every `tests/golden/tools/` request run greedily, `required`/named tool calls and `json_schema` outputs parse and validate, `auto` requests return valid tool calls or content with no raw call JSON leaking, and an `n: 3` request with a tiny `kv.gpu.max_bytes` that forces preemption returns the same tokens as the same request run alone. Run: `scripts/lab-test.sh dgx-spark` — expect FAIL before the change (`execution.backend hip` finds no AMD device: exit 1 naming `execution.device 0`, `nvidia`, `hip`).
- [ ] Implement the backend switch; the scheduler, KV accounting, sampling, llguidance and tool parsing are untouched (S-12 is a parity claim, not new code); the tiny `kv.gpu.max_bytes` must still be ≥ one 16-token block per running request (1,835,008 bytes per block for Llama-3.2-3B).
- [ ] Run: `scripts/lab-test.sh dgx-spark` — expect exit 0 with `test tools_and_json_schema ... ok`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(turbine-server): run lab_openai on the backend named by TURBINE_TEST_BACKEND`

## Task 20: Serving on GB10 — golden compare over HTTP and provider metrics

Files: no repository files (outputs pasted into the task evidence)
Interfaces:

- consumes `scripts/lab-serve.sh dgx-spark <config>` and `--stop` (Task 8), `turbine-golden compare --url <url> --reference <jsonl> --concurrency 16` (P1/P2), `GET /metrics`, the `kernel_library_loaded` record (Task 6)

Covers: S-11, S-13, S-14; manual golden compare over HTTP for both Spark configs, and `curl -s http://192.168.10.246:18000/metrics | grep 'turbine_kernel_provider_selected{.*provider="cuda"'` listing one line per op
Depends on: Tasks 16, 19

- [ ] Write failing check: `curl -fsS http://192.168.10.246:18000/ready` — expect FAIL (connection refused: nothing serves on 18000 yet).
- [ ] Implement the run for `scripts/lab/phase2b-spark-llama.yaml`: `scripts/lab-serve.sh dgx-spark scripts/lab/phase2b-spark-llama.yaml` — expect `precondition: ok` (a correctness run proceeds without asking only then; on `precondition: MemAvailable … ask the user to free memory`, ASK THE USER FIRST and wait for their confirmation), the streamed container log to contain `"event":"kernel_library_loaded"` with `"backend":"cuda"`, `"device_arch":"sm_121"` and `"event":"memory_budget"` with `"device_kind":"unified"`, and the final line `lab-serve: turbine-server ready at http://192.168.10.246:18000`.
- [ ] Run: `cargo run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.246:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --concurrency 16` — expect PASS (exit 0 under the committed tolerance); then `curl -s http://192.168.10.246:18000/metrics | grep 'turbine_kernel_provider_selected{.*provider="cuda"'` — expect one line per op in the startup selection log (for Llama at least `gemm`, `attention_prefill_paged`, `attention_decode_paged`, `rmsnorm`, `rope`, `silu_mul`, `embedding`, `add`, `copy_blocks`), each with a named CUDA `impl` from the contract list and none with `cpu-reference`; paste both outputs and the `kernel_library_loaded` record.
- [ ] Run: `scripts/lab-serve.sh dgx-spark --stop`, then repeat the three steps above with `scripts/lab/phase2b-spark-olmoe.yaml` and `tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl` (the metrics grep additionally shows `moe_route` and `moe_experts`); finish with `scripts/lab-serve.sh dgx-spark --stop` — expect `docker ps` on dgx-spark to list only the production containers.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: none (evidence only)

## Task 21: GB10 baseline benchmark against production vLLM (ASK THE USER FIRST)

Files: no repository files (JSON reports and environment facts pasted into the task evidence)
Interfaces:

- consumes `turbine-bench --url <url> --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos --output json` (P0/P2), `scripts/lab-serve.sh dgx-spark <config>`, `--vllm <slug>`, `--stop` (Task 8)
- `VLLM_IMAGE_DIGEST` at the top of `scripts/lab-serve.sh` (Task 8)

Covers: S-13; manual baseline run on dgx-spark with `requests_failed: 0` for Turbine and the vLLM reports (or the "vLLM did not run" record)
Depends on: Task 20

- [ ] Write failing check: the evidence has no GB10 baseline yet — `ls target/bench/gb10-baseline-*.json` — expect FAIL (no such file).
- [ ] ASK THE USER FIRST: benchmarks share GB10 compute with production vLLM; record their go-ahead (and any workload they moved) in the evidence before continuing.
- [ ] Implement the environment record: `ssh piwi@192.168.10.246 "docker image inspect --format '{{json .RepoDigests}}' ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5"` (update `VLLM_IMAGE_DIGEST` in `scripts/lab-serve.sh` if production moved to another digest, as its own commit), the lab image tag from `spark_image_tag`, `nvcc`/cuBLAS versions from the CMake log, the driver version from `nvidia-smi`, and `ssh piwi@192.168.10.246 nvidia-smi --query-compute-apps=pid,used_memory --format=csv` taken immediately before each run.
- [ ] Run for each model: `scripts/lab-serve.sh dgx-spark scripts/lab/phase2b-spark-<llama|olmoe>.yaml`, then `cargo run --release -p turbine-bench -- --url http://192.168.10.246:18000 --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos --output json > target/bench/gb10-baseline-turbine-<slug>.json` — expect exit 0 with `"requests_failed": 0`; then `scripts/lab-serve.sh dgx-spark --stop`.
- [ ] Run for each slug: `scripts/lab-serve.sh dgx-spark --vllm <slug>` (expect `lab-serve: vLLM baseline ready at http://192.168.10.246:18100 (…@sha256:…)`), the same `turbine-bench` command against `http://192.168.10.246:18100` into `target/bench/gb10-baseline-vllm-<slug>.json`, then `scripts/lab-serve.sh dgx-spark --stop`; if vLLM does not become ready, record "vLLM did not run" with the printed log tail (this does not block the phase). There is no numeric bar.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: none (reports are evidence, not committed)

## Task 22: Overload run against Turbine on GB10 (ASK THE USER FIRST)

Files: no repository files (report and observations pasted into the task evidence)
Interfaces:

- consumes `turbine-bench --url <url> --concurrency 512 --requests 2000 --max-tokens 256 --ignore-eos --output json`, `GET /ready`, `GET /turbine/v1/kv` (P2), `scheduler.max_queued_requests` default 256 (contract §3.2)

Covers: S-13; manual overload run with 429 rejections, the worker still ready, 0 used KV blocks once idle and production containers undisturbed
Depends on: Task 21

- [ ] Write failing check: `ls target/bench/gb10-overload-turbine-llama.json` — expect FAIL (no such file).
- [ ] ASK THE USER FIRST (overload runs always ask); record the go-ahead, then capture `ssh piwi@192.168.10.246 "docker ps --format '{{.Names}} {{.Status}}'"` as the before-snapshot.
- [ ] Implement the run: `scripts/lab-serve.sh dgx-spark scripts/lab/phase2b-spark-llama.yaml` (Phase 2 scheduler defaults, so `max_queued_requests` is 256), then `cargo run --release -p turbine-bench -- --url http://192.168.10.246:18000 --concurrency 512 --requests 2000 --max-tokens 256 --ignore-eos --output json > target/bench/gb10-overload-turbine-llama.json`.
- [ ] Run: expect exit 0 with a non-zero 429 count in the report; `curl -fsS http://192.168.10.246:18000/ready` — expect 200; after 30 s idle `curl -s http://192.168.10.246:18000/turbine/v1/kv` — expect `"used_blocks":0`; the after-snapshot of `docker ps` shows the production containers with `Up` durations grown by the elapsed time only (no restart); then `scripts/lab-serve.sh dgx-spark --stop`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: none (evidence only)

## Task 23: macOS workspace gate and commands documentation

Files: `AGENTS.md` (Commands section: CUDA shim build, Spark lab commands, `TURBINE_TEST_BACKEND`)
Interfaces:

- consumes every earlier task; documents `cmake -S kernels/cuda -B <build> -DCMAKE_CUDA_ARCHITECTURES=121a [-DTURBINE_CUDA_ROOT=<toolkit>]`, `cmake --build <build> --parallel`, `scripts/lab-test.sh dgx-spark|dgx-spark2|novanas`, `scripts/lab-serve.sh [--dry-run] <host> <config>|--vllm <slug>|--stop`, `scripts/lab/weights-manifest.sh <host> <slug>`

Covers: S-2, S-7; `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` exit 0 on macOS arm64 with no CUDA, no ROCm and no weights, and `cargo tree --workspace` lists no `cudarc` or other CUDA binding crate
Depends on: Tasks 1–22

- [ ] Write failing check: `grep -n "kernels/cuda" AGENTS.md` — expect FAIL (exit 1: the Commands section does not mention the CUDA shim yet).
- [ ] Implement the AGENTS.md Commands additions (CUDA build only inside the Spark lab image; `TURBINE_TEST_BACKEND` is `hip` on novanas and `cuda` on the Sparks; Spark correctness runs need the `MemAvailable` precondition, benchmarks/soak/overload ask the user first); `CLAUDE.md` already imports `AGENTS.md` and needs no change.
- [ ] Run: `cargo build --workspace && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check` on the macOS workstation — expect PASS; `cargo tree --workspace | grep -Ei 'cudarc|cuda-sys|cust|nvrtc'` — expect exit 1 (no match); `otool -L target/debug/turbine-server | grep -i cuda` — expect exit 1.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `docs(agents): CUDA shim build and Spark lab commands`
