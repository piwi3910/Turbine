# phase-1-single-request — implementation plan

Status: draft
Spec: .procoder/specs/phase-1-single-request.md

## Goal

Serve `meta-llama/Llama-3.2-3B-Instruct` (BF16) for one request at a time on one R9700 through a runtime-loaded `libturbine_hip.so` behind vendor-neutral kernel traits and a C ABI, with OpenAI completions/chat (SSE), and prove tokens and logprobs match the committed Hugging Face BF16 reference within `tolerance.json`.

## Architecture

Three new crates carry the model path: `turbine-tensor` (dtypes, `Tensor`, `DeviceBuffer`, the safe `DeviceMemory` trait with a host-memory backend), `turbine-kernels` (capability traits per op family, `KernelRegistry`, the pure-Rust `cpu-reference` provider, and the `unsafe` FFI to the shim via `libloading`), and `turbine-model` (HF config + allowlist, safetensors index and positioned-read loader, tokenizer + minijinja chat template, startup budget, `LlamaExecutor`, host sampler, single-request `generate` iterator, tiny synthetic checkpoint). `kernels/include/turbine_kernels.h` (ABI v1) is implemented by `kernels/rocm/` (CMake + hipcc; hipBLASLt GEMM, CK ck_tile FMHA attention and rmsnorm2d, minimal Turbine HIP kernels for embedding/rope/silu_mul/add). `turbine-server` follows the P1 startup order, owns one generation thread with a single slot, and implements the Phase 0 `turbine-api` traits extended with `submit`; `turbine-bench` gains `turbine-golden`, and `scripts/golden/hf_reference.py` produces the committed fixtures.

## Constraints

From the spec (verbatim):

- Rust only in the build and runtime path of the workspace; the HIP shim is C++/HIP compiled by `hipcc` through CMake and exposes only a C ABI (TS §6). No Python in the serving path (TS §21 rule 4): Python appears only in `scripts/golden/hf_reference.py` (fixture generation) and, if Composable Kernel's instance generator requires it, inside the CMake build of _libturbine_hip.so_ — never in `cargo build` or `turbine-server`.
- The Rust workspace must build and every non-ignored test must pass on macOS arm64 with no GPU, no ROCm and no model weights (Phase 0 constraint, unchanged). Nothing links HIP at build time; _libturbine_hip.so_ (which itself links the ROCm runtime, hipBLASLt and CK) is loaded at run time.
- `unsafe` and FFI only in `turbine-device` and `turbine-kernels`, every block with a `// SAFETY:` comment. Ownership rules (TS §21 rule 10): every device pointer is allocated by the shim and owned by exactly one `DeviceBuffer`; the shim never retains a caller pointer beyond the call; streams and workspaces are owned by the shim context and destroyed with it; a `DeviceBuffer` must not outlive its context (enforced by holding an `Arc` of the context).
- New runtime dependencies: `safetensors` (header parsing), `tokenizers` (default features off, `onig` off, `fancy-regex` on), `minijinja` and `minijinja-contrib` (pycompat, for the Llama template), `smallvec`, `half` (bf16/f16 host conversion), `rand_chacha` (seeded sampling). Each is recorded in the crate manifest comment with its reason.
- External kernel libraries: hipBLASLt is taken from the ROCm 7.14.1 install at `/opt/rocm/rocm` (version logged at context creation); Composable Kernel is pinned by exact commit in `kernels/rocm/CMakeLists.txt`; licenses retained under `kernels/rocm/third_party/LICENSES/` (TS §6).
- Target hardware: novanas (192.168.10.203, Debian 13 amd64), one of its two AMD Radeon AI PRO R9700 (RDNA4, `gfx1201`, 32 GB dedicated VRAM each), ROCm 7.14.1 at `/opt/rocm/rocm` (off AMD's support matrix for Debian + RDNA4, so library coverage gaps are expected and are handled per S-7), k3s with `amdgpu-device-plugin`; runs are k3s Jobs (Phase 0). The user empties the R9700s for Turbine work.
- Any lab run that needs production workloads moved or memory freed on any host (novanas GPUs or the DGX Sparks' production vLLM) is preceded by asking the user; the user moves workloads. Turbine scripts never stop, restart or reconfigure other workloads.
- Weights: `meta-llama/Llama-3.2-3B-Instruct` in `/home/piwi/turbine-models/llama-3.2-3b-instruct` on novanas (and later on each host that runs it), downloaded once over SSH by Claude with a Hugging Face token the user supplies at that time; the token is never written to the repository, the scripts, the Job manifests or a file on the host. Tests and scripts never download.
- Bounded resources (TS §21 rule 8): one generation slot; the weight staging buffer is at most 256 MiB; safetensors headers larger than 100 MiB are rejected; request bodies keep the Phase 0 limit; the SSE channel to a client holds at most 64 events.

From the interface contract (binding):

- Names in `.procoder/contract/interfaces.md` §3–§21 are used verbatim; §23 picks override the spec: C-3 (a mid-stream error event is followed by `data: [DONE]`), C-13 (`server.listen: 0.0.0.0:18000`, no `server.port`), C-15 (fixture slug `llama-3.2-3b-instruct`), C-25 (P1 rule: 3 consecutive failed requests → `/ready` 503 `device_error`, exit 1).
- Edition 2024, `rust-version = "1.97"`; every public enum a later phase extends is `#[non_exhaustive]`; every config struct `#[serde(deny_unknown_fields, default)]`; one top-level `thiserror` enum per crate; metric label values from closed sets rendered by `as_str()`.
- Tests: unit tests `cargo test -p <crate> <module>::tests::<name>`, integration tests `cargo test -p <crate> --test <binary> <name>`; GPU/weights tests are `#[ignore]`, start with `if !turbine_kernels::test_support::require_backend("hip") { return; }`, and fail (never skip) when `TURBINE_TEST_MODEL_DIR` is unset.
- Builds on the Phase 0 plan (`.procoder/plans/phase-0-skeleton.md`): `turbine_core::config::{Config, ByteSize, Override, load, ConfigError}`, `turbine_observability::{MetricsRegistry, init_tracing, http::RequestIdExt}`, `turbine_device::{discover, DiscoveryOptions, DeviceInventory, DeviceInfo}`, `turbine_api::{router, ApiState, ApiLimits, InferenceBackend, Diagnostics, Readiness, ReadyState, NotReadyReason, ModelCard, ApiError, ErrorType}`, `turbine-server` `startup.rs`/`exit.rs`, `turbine-bench` lib (`args`, `client`, `prompt`, `report`), `scripts/lab-test.sh`, `scripts/lab/novanas-test-job.yaml`.
- Every task ends gate-clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`.

## Task 1: Phase 1 core vocabulary and `execution` / model configuration keys

Files: `crates/turbine-core/src/types.rs` (add `ExecutionBackend`, `DType`, `RequestId`, `KvLayout`, `ModelShape`), `crates/turbine-core/src/request.rs` (add sampling/stop/event types and P1 `ErrorCode` variants), `crates/turbine-core/src/config/mod.rs` (add `ModelConfig` fields, `ExecutionConfig`, validation), `crates/turbine-core/src/config/tests.rs` (new test), `crates/turbine-core/Cargo.toml` (add `uuid`, `smallvec` with reason comments), `Cargo.toml` (workspace deps `smallvec`, `uuid` feature `serde`)
Interfaces:

- `pub enum ExecutionBackend { Hip, Cuda, Cpu }` (serde snake_case, `#[non_exhaustive]`), `fn as_str(self) -> &'static str`
- `pub enum DType { BF16, F16, F32, I32, I64 }` with `fn size_bytes(self) -> usize`, `fn abi_code(self) -> i32` (0..4 = `TURBINE_DTYPE_*`), `fn as_str(self) -> &'static str`
- `pub struct RequestId(pub uuid::Uuid)`, `fn new_v4() -> RequestId`
- `pub struct KvLayout { num_layers, num_kv_heads, head_dim: u32, dtype: DType, block_tokens: u32 }`, `fn bytes_per_token(&self) -> u64`, `fn block_bytes(&self) -> u64`
- `pub struct ModelShape { architecture: String, num_layers, hidden, num_attention_heads, num_kv_heads, head_dim, intermediate, vocab, num_experts, experts_per_token: u32, tied_embeddings: bool, weight_bytes: u64, max_position_embeddings: u32 }`
- `pub struct SamplingParams { temperature: f32, top_p: f32, top_k: i32, seed: Option<u64>, logprobs: Option<u32> }` (Default 1.0 / 1.0 / -1 / None / None)
- `pub struct StopConditions { eos_token_ids: SmallVec<[u32; 4]>, stop_strings: Vec<String>, max_tokens: u32, ignore_eos: bool }`
- `pub enum FinishReason { Stop, Length }`, `pub struct Usage { prompt_tokens: u32, completion_tokens: u32 }`
- `pub struct GenerationRequest { id: RequestId, endpoint: Endpoint, http_request_id: String, prompt_tokens: Vec<u32>, sampling: SamplingParams, stop: StopConditions }`
- `pub struct CancelFlag(Arc<AtomicBool>)`, `fn cancel(&self)`, `fn is_cancelled(&self) -> bool`
- `pub enum GenerationEvent { Started { choice: u32 }, Token { choice: u32, text: String, token_id: u32, logprob: Option<f32>, top_logprobs: Vec<(u32, f32)> }, Finished { choice: u32, reason: FinishReason, usage: Option<Usage> }, Error { code: ErrorCode, message: String } }`
- `ErrorCode` gains `ModelNotFound`, `UnsupportedParameter`, `ContextLengthExceeded`, `EngineBusy`, `TemplateError`, `InvalidRequest` (`"invalid_request"`, contract addition)
- `ModelConfig` gains `served_name: Option<String>`, `tokenizer: Option<PathBuf>`, `chat_template: Option<PathBuf>`, `max_seq_len: Option<u32>`; `pub struct ExecutionConfig { backend: ExecutionBackend, device: DeviceId, kernel_library: Option<PathBuf> }` as `Config::execution`
  Covers: S-1 (shared vocabulary), config keys of §Configuration additions; the exit-2 half of `tiny_server startup_failures_exit_1` is asserted end to end in Task 17
  Depends on: Phase 0 plan Task 1

- [ ] Write failing test `turbine-core config::tests::execution_and_model_keys`: loading `model: {path: /m, served_name: meta-llama/Llama-3.2-3B-Instruct, max_seq_len: 4096}` plus `execution: {backend: cpu, device: 1, kernel_library: /opt/k/libturbine_hip.so}` yields those values; defaults give backend `hip`, device 0; `execution.backend: cuda` fails with key `execution.backend` and a message containing `phase-2b-nvidia`; `served_name: ""`, `max_seq_len: 0` and `backend: rocm` are rejected naming their keys. Run: `cargo test -p turbine-core config::tests::execution_and_model_keys` — expect FAIL
- [ ] Implement the types, request vocabulary and config additions: `ExecutionConfig` defaults `hip`/`DeviceId(0)`/null; `Config::validate` rejects `served_name` outside 1..=256 chars, `max_seq_len: 0` and backend `cuda` (reason "cuda is not available in this build; NVIDIA execution arrives with phase-2b-nvidia"); the upper bound of `max_seq_len` against the model is a startup check (exit 1, Task 17).
- [ ] Run: `cargo test -p turbine-core` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-core): phase 1 vocabulary and execution configuration`

## Task 2: `turbine-tensor` — device memory handles, host backend, tensors

Files: `crates/turbine-tensor/Cargo.toml` (new crate, `unsafe_code = "forbid"`, deps core/thiserror/smallvec), `crates/turbine-tensor/src/lib.rs` (re-exports), `crates/turbine-tensor/src/buffer.rs` (`DevicePtr`, `DeviceMemory`, `DeviceBuffer`, `DeviceSlice`, `StreamRef`, `MemInfo`, `MemoryError`), `crates/turbine-tensor/src/host.rs` (`HostMemory` + test), `crates/turbine-tensor/src/tensor.rs` (`Tensor`, `TensorView`), `crates/turbine-tensor/src/dtype.rs` (re-export `DType`), `Cargo.toml` (member + workspace dep)
Interfaces:

- `pub struct DevicePtr(u64)`: `fn from_addr(u64) -> DevicePtr`, `fn addr(self) -> u64`, `fn offset(self, bytes: u64) -> DevicePtr`, `const NULL`
- `pub trait DeviceMemory: Send + Sync { fn device(&self) -> DeviceId; fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError>; fn free(&self, ptr: DevicePtr); fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError>; fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError>; fn copy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<(), MemoryError>; fn synchronize(&self) -> Result<(), MemoryError>; fn mem_info(&self) -> Result<MemInfo, MemoryError>; fn compute_stream(&self) -> StreamRef; fn as_host(&self) -> Option<&host::HostMemory> { None } }`
- `pub struct DeviceBuffer`: `fn alloc(mem: &Arc<dyn DeviceMemory>, bytes: usize) -> Result<DeviceBuffer, MemoryError>`, `len`, `device`, `ptr`, `memory(&self) -> &Arc<dyn DeviceMemory>` (contract addition), `slice(&self, offset: usize, len: usize) -> DeviceSlice<'_>`, `whole(&self) -> DeviceSlice<'_>` (addition), `copy_from_host(&mut self, offset: usize, src: &[u8])`, `copy_to_host(&self, offset: usize, dst: &mut [u8])`; frees on `Drop`; not `Clone`
- `pub struct DeviceSlice<'a>`: `ptr`, `len`, `device`, `memory(&self) -> &'a Arc<dyn DeviceMemory>`, `sub(offset, len)`, `read_bytes(&self) -> Result<Vec<u8>, MemoryError>`, `write_bytes(&self, src: &[u8]) -> Result<(), MemoryError>` (additions; blocking, synchronize first)
- `pub struct MemInfo { free_bytes: u64, total_bytes: u64 }`; `pub enum MemoryError { OutOfMemory { requested: u64 }, Device { message: String, sticky: bool }, InvalidArgument(String), Unsupported(String) }`
- `pub struct StreamRef`: `fn new(native: u64, device: DeviceId, owner: Arc<dyn DeviceMemory>) -> StreamRef`, `native_handle(&self) -> u64` (0 before ABI v4), `device(&self) -> DeviceId`
- `host::HostMemory::new(device: DeviceId, capacity: u64) -> Arc<HostMemory>`, `with_slice<R>(&self, p: DevicePtr, len: usize, f: impl FnOnce(&[u8]) -> R) -> R`, `with_slice_mut<R>(…)`; synthetic addresses from `0x1000_0000`, 4 KiB-aligned, `mem_info` = capacity − used
- `pub struct Tensor { storage: DeviceBuffer, shape: SmallVec<[usize; 4]>, strides: SmallVec<[usize; 4]>, dtype: DType, device: DeviceId }`, `fn empty(mem: &Arc<dyn DeviceMemory>, shape: &[usize], dtype: DType) -> Result<Tensor, MemoryError>`, `fn view(&self) -> TensorView<'_>`
- `pub struct TensorView<'a> { slice: DeviceSlice<'a>, shape, strides: SmallVec<[usize; 4]>, dtype: DType }`, `fn contiguous(slice, offset_elems: usize, shape: &[usize], dtype) -> TensorView<'a>`, `fn rows(&self, start: usize, count: usize) -> TensorView<'a>`, `fn numel(&self) -> usize`
  Covers: S-5 (`Tensor` per TS §6, `DeviceBuffer` owning one allocation freed on `Drop`)
  Depends on: Task 1

- [ ] Write failing test `turbine-tensor host::tests::alloc_copy_free_round_trip`: on `HostMemory::new(DeviceId(0), 1 MiB)`, a 16-byte buffer written with `[1,2,3]` at offset 4 reads back `[1,2,3]`, `mem_info().free_bytes` is 1 MiB − 16 while allocated and 1 MiB after drop, and a 2 MiB allocation fails with `OutOfMemory`. Run: `cargo test -p turbine-tensor host::tests::alloc_copy_free_round_trip` — expect FAIL
- [ ] Implement the crate: `HostMemory` keeps a `RwLock<BTreeMap<base, Arc<RwLock<Box<[u8]>>>>>` looked up by `range(..=addr).next_back()` with bounds checks; `DeviceBuffer` holds `Arc<dyn DeviceMemory>` so it cannot outlive its context; `TensorView::rows` covers `(count−1)·row_stride + row_width` bytes.
- [ ] Run: `cargo test -p turbine-tensor` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-tensor): device buffers, host memory backend and tensors`

## Task 3: Kernel C ABI v1 header and `turbine-kernels` op traits

Files: `kernels/include/turbine_kernels.h` (ABI v1 exactly as contract §9.3 v1 block, `#define TURBINE_ABI_VERSION 1u`, comments free of vendor words), `crates/turbine-kernels/Cargo.toml` (new crate, `unsafe_code = "allow"`, deps core/observability/tensor/device/thiserror/tracing/prometheus-client/libloading/half; build-dep `cc`; dev-deps tracing-subscriber, rand_chacha, rand_core), `crates/turbine-kernels/src/lib.rs` (`TURBINE_KERNELS_ABI_VERSION`, `KernelError`), `crates/turbine-kernels/src/ops/mod.rs` (configs, contexts, traits, `OpKind`, `ProviderId`, `KernelProvider`), `crates/turbine-kernels/tests/unsafe_isolation.rs`, `crates/turbine-kernels/tests/abi_header_neutral.rs`, `Cargo.toml` (member, `[profile.dev.package.turbine-kernels] opt-level = 3` so the CPU oracle is fast in tests)
Interfaces:

- `pub const TURBINE_KERNELS_ABI_VERSION: u32 = 1;`
- `pub enum KernelError { InvalidArgument { message }, Unsupported { message }, OutOfMemory { message }, Device { message }, Library { message }, Load { path: PathBuf, detail: String }, AbiMismatch { expected: u32, found: u32 }, BackendMismatch { expected: String, found: String }, ArchMismatch { device_arch: String, build_archs: String }, NoProvider { op: OpKind, config: String } }` with the contract §7.1 messages; `fn is_sticky(&self) -> bool` (Device message starting with `hipErrorIllegalAddress`, `hipErrorLaunchFailure`, `hipErrorAssert`), `fn is_oom(&self) -> bool`; `From<KernelError> for MemoryError` and `From<MemoryError> for KernelError`
- `pub enum OpKind { Gemm, AttentionPrefill, AttentionDecode, Rmsnorm, Rope, SiluMul, Embedding, Add }` (`#[non_exhaustive]`), `fn as_str(&self) -> &'static str` (= C ABI suffix), `pub const ALL: &'static [OpKind]` (contract addition)
- Configs (shapes only, `Display` = the P1 failure-message form): `GemmConfig { n: u64, k: u64, trans_b: bool, a_dtype, b_dtype, c_dtype: DType }`; `AttentionKind { Prefill, Decode }`; `AttentionConfig { kind, num_q_heads, num_kv_heads, head_dim: u32, dtype: DType, block_tokens: Option<u32>, causal: bool }` rendered `head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1`; `NormConfig { dim: u64, dtype }`; `RopeConfig { num_q_heads, num_kv_heads, head_dim, rotary_dim: u32, dtype }`; `ActivationConfig { cols: u64, dtype }`; `EmbeddingConfig { hidden: u64, vocab_rows: u64, dtype }`; `ElementwiseConfig { dtype }`
- Contexts (tensor views; the compute stream is implicit in the provider's context — contract addition): `GemmContext<'a> { a, b, c: TensorView<'a>, trans_b: bool, alpha: f32, beta: f32 }`; `AttentionContext<'a> { cfg, q, k_cache, v_cache, out, q_start: u32, scale: f32 }`; `NormContext<'a> { x, weight, out, eps: f32 }`; `RopeContext<'a> { cfg, q, k, positions, inv_freq }`; `ActivationContext<'a> { gate, up, out }`; `EmbeddingContext<'a> { ids, table, out, vocab_offset: i64 }`; `ElementwiseContext<'a> { a, b, out }`
- Traits `GemmKernel`, `AttentionKernel`, `NormKernel`, `RopeKernel`, `ActivationKernel`, `EmbeddingKernel`, `ElementwiseKernel`, each `fn supports(&self, cfg: &XConfig) -> bool; fn implementation(&self, cfg: &XConfig) -> String; fn execute(&self, ctx: &mut XContext<'_>) -> Result<(), KernelError>`
- `pub trait KernelProvider: Send + Sync { fn id(&self) -> ProviderId; fn gemm/attention/norm/rope/activation/embedding/elementwise(&self) -> Option<&dyn …Kernel>; }`, `pub struct ProviderId(pub &'static str)`
  Covers: S-1 AC `cargo test -p turbine-kernels --test unsafe_isolation`; S-1/S-7 AC `cargo test -p turbine-kernels --test abi_header_neutral`; S-6 (traits carry no vendor types)
  Depends on: Task 2

- [ ] Write failing test `turbine-kernels --test unsafe_isolation`: scans `crates/*/src` and `benches/*/src`, asserts every word-bounded `unsafe` lies under `crates/turbine-device/src` or `crates/turbine-kernels/src`, every `unsafe {`/`unsafe impl` line is preceded (over comment and attribute lines) by `// SAFETY:`, and every other crate/bench manifest contains `unsafe_code = "forbid"`. Run: `cargo test -p turbine-kernels --test unsafe_isolation` — expect FAIL
- [ ] Write failing test `turbine-kernels --test abi_header_neutral`: with comments stripped, no identifier in `turbine_kernels.h` starts with `hip`, `cuda`, `rocm` or `nv` (case-insensitive), the header declares `turbine_<op>(`, `turbine_<op>_supported(` and `turbine_<op>_impl(` for every `OpKind::ALL`, and `#define TURBINE_ABI_VERSION 1u` equals `TURBINE_KERNELS_ABI_VERSION`. Run: `cargo test -p turbine-kernels --test abi_header_neutral` — expect FAIL
- [ ] Implement the header, `lib.rs` and `ops/mod.rs` exactly as the Interfaces list; a temporary injection of `unsafe { … }` into `crates/turbine-tensor/src` must make `unsafe_isolation` fail naming the file and line.
- [ ] Run: `cargo test -p turbine-kernels --test unsafe_isolation --test abi_header_neutral` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): vendor-neutral kernel ABI v1 and op capability traits`

## Task 4: Kernel registry with logged, metered selection

Files: `crates/turbine-kernels/src/registry.rs` (`KernelRegistry`, `KernelMetrics`, `OpConfig`, `OpRequirement`, `Selection`, tests)
Interfaces:

- `pub enum OpConfig { Gemm(GemmConfig), Attention(AttentionConfig), Rmsnorm(NormConfig), Rope(RopeConfig), SiluMul(ActivationConfig), Embedding(EmbeddingConfig), Add(ElementwiseConfig) }`, `fn op(&self) -> OpKind`, `fn render(&self) -> String`
- `pub struct OpRequirement { pub op: OpKind, pub config: String, pub spec: OpConfig }`, `impl From<OpConfig> for OpRequirement`
- `pub struct Selection { op: OpKind, config: String, provider: ProviderId, implementation: String, reason: String }` (contract addition)
- `pub struct KernelMetrics { pub provider_selected: Family<SelectedLabels, Gauge> }`, `fn register(reg: &MetricsRegistry) -> KernelMetrics` — family `turbine_kernel_provider_selected{op,provider,impl}`
- `KernelRegistry::build(providers: Vec<Arc<dyn KernelProvider>>, order: &[ProviderId], reqs: &[OpRequirement], metrics: &KernelMetrics) -> Result<KernelRegistry, KernelError>`, `fn selections(&self) -> &[Selection]`, accessors `gemm(&self, cfg: &GemmConfig) -> &dyn GemmKernel`, `attention(&AttentionConfig)`, `norm(&NormConfig)`, `rope(&RopeConfig)`, `activation(&ActivationConfig)`, `embedding(&EmbeddingConfig)`, `elementwise(&ElementwiseConfig)` (panic naming op and config if not selected at startup — the executor derives its requirement list from the same code path)
  Covers: S-6 AC `registry::tests::selection_order_and_reason`, `registry::tests::no_provider_is_startup_error`; S-14 (kernel selection log + gauge)
  Depends on: Task 3

- [ ] Write failing test `turbine-kernels registry::tests::selection_order_and_reason`: two fake providers `first` (supports head_dim 64 only) and `second` (supports head_dim 128, kv_heads 8) with order `[first, second]` and one `attention_prefill head_dim=128 kv_heads=8` requirement select `second`/`second_fmha`; the JSON log captured through a `tracing_subscriber::fmt().json()` writer contains `"op":"attention_prefill"`, the rendered config, `"provider":"second"`, `"impl":"second_fmha"` and `"reason":"first provider in order supporting config; unsupported by: first"`; `/metrics` text contains `turbine_kernel_provider_selected{op="attention_prefill",provider="second",impl="second_fmha"} 1`. Run: `cargo test -p turbine-kernels registry::tests::selection_order_and_reason` — expect FAIL
- [ ] Write failing test `turbine-kernels registry::tests::no_provider_is_startup_error`: only `first` registered → `build` returns `Err` whose text is `no kernel provider supports attention_prefill head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1`. Run: `cargo test -p turbine-kernels registry::tests::no_provider_is_startup_error` — expect FAIL
- [ ] Implement `build`: for each distinct `OpConfig` walk `order`, probe the provider's family trait `supports`, take its `implementation`, log `event="kernel_selected"` at INFO with `op, config, provider, impl, reason`, set the gauge, return `NoProvider` on the first unsatisfied requirement.
- [ ] Run: `cargo test -p turbine-kernels registry::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): kernel registry with selection log and gauge`

## Task 5: `cpu-reference` provider

Files: `crates/turbine-kernels/src/cpu/mod.rs` (`CpuReference`, `cpu_reference_provider`, load/store helpers, tests), `crates/turbine-kernels/src/cpu/math.rs` (scalar gemm, attention, rmsnorm, rope, silu)
Interfaces:

- `pub fn cpu_reference_provider() -> Arc<dyn KernelProvider>` (id `cpu-reference`); `pub fn round_to(dtype: DType, v: f32) -> f32`
- Implementation names: `cpu_gemm_f32acc`, `cpu_attention_f32acc`, `cpu_rmsnorm`, `cpu_rope_half_split`, `cpu_silu_mul`, `cpu_embedding`, `cpu_add`
- Numerics (the reference every provider is tested against): f32 accumulation sequential over `k`; output rounded to the view dtype at each op boundary; RMSNorm `round(round(x·rsqrt(mean(x²)+eps))·w)` (HF LlamaRMSNorm); RoPE half-split with `f = pos·inv_freq[i]` in f32, cos/sin rounded to the activation dtype, `x1' = round(round(x1·c) + round(−x2·s))`, `x2' = round(round(x2·c) + round(x1·s))`; attention query `i` at absolute position `q_start+i` attends keys `0..=q_start+i`, head `h` reads KV head `h / (Hq/Hkv)`; SiLU·up `round(round(silu(g))·u)`; embedding rows outside `[0, vocab_rows)` are zero
  Covers: S-6 (`cpu-reference`, pure Rust, f32 accumulation, always available)
  Depends on: Task 4

- [ ] Write failing test `turbine-kernels cpu::tests::gemm_and_causal_gqa_attention_reference`: `[[1,2,3],[−1,0.5,2]]·[[1,0,1],[2,1,0]]ᵀ` gives F32 `[4,4,1,−1.5]`; attention with 2 query heads sharing 1 KV head, q=k one-hot, v=`[[10,0],[0,20]]`, scale 1: token 0 outputs `[10,0]` for both heads, token 1 head 0 outputs `[10·σ(1), 20·(1−σ(1))]` within 1e-5. Run: `cargo test -p turbine-kernels cpu::tests` — expect FAIL
- [ ] Implement the provider: every op reads its views with strides through `DeviceSlice::read_bytes`, computes in f32, rounds, and writes back with a read-modify-write so bytes between strided rows (KV cache rows) are preserved.
- [ ] Run: `cargo test -p turbine-kernels cpu::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): cpu-reference provider`

## Task 6: Runtime-loaded shim bindings, stub libraries and `test_support`

Files: `crates/turbine-kernels/src/ffi.rs` (`#[repr(C)]` descriptors, `ShimSymbols`, `check`), `crates/turbine-kernels/src/shim.rs` (`ShimLibrary`, `ShimContext`, `ShimProvider`, `shim_provider`, test), `crates/turbine-kernels/src/test_support.rs` (`require_backend`, `require_env_dir`), `crates/turbine-kernels/build.rs` (compiles stub shims with the host C compiler), `crates/turbine-kernels/stub/stub_shim.c` (every ABI v1 symbol; `STUB_ABI`, `STUB_BACKEND`, `STUB_ARCHS` macros)
Interfaces:

- `pub(crate) struct GemmDesc, AttentionDesc, RmsnormDesc, RopeDesc, SiluMulDesc, EmbeddingDesc, AddDesc` field-for-field with the header; `ShimSymbols` resolved once (missing symbol → `KernelError::Load`); `check(code: i32, syms, ctx) -> Result<(), KernelError>` maps −1…−5 with the `turbine_last_error` message
- `ShimLibrary::load(path: &Path, expected_backend: ExecutionBackend) -> Result<Arc<ShimLibrary>, KernelError>` (ABI version checked before any other symbol, then backend name), `ShimLibrary::search_paths(backend: ExecutionBackend, explicit: Option<&Path>) -> Vec<PathBuf>` (explicit alone; else `TURBINE_KERNEL_LIBRARY`, `libturbine_hip.so` beside the executable, bare name for the loader path), `abi_version(&self) -> u32`, `backend_name(&self) -> &str`, `build_archs(&self) -> &[String]`, `path(&self) -> &Path`, `create_context(self: &Arc<Self>, device: &DeviceInfo) -> Result<Arc<ShimContext>, KernelError>` (arch ∉ build_archs → `ArchMismatch` before `turbine_ctx_create(vendor_index)`)
- `ShimContext: DeviceMemory + Send + Sync` (`copy_h2d`/`copy_d2h` synchronize before returning so host slices outlive the async copy; `copy_d2d` → `Unsupported` until ABI v3), `fn library(&self) -> &Arc<ShimLibrary>`
- `pub fn shim_provider(ctx: Arc<ShimContext>) -> Arc<dyn KernelProvider>` (id = backend name; `supports` → `turbine_<op>_supported` with null pointers; `implementation` → `turbine_<op>_impl`)
- `pub fn require_backend(backend: &str) -> bool` (contract §7; delivered in P1 so every GPU test uses it from the start), `pub fn require_env_dir(var: &str) -> PathBuf`
- Build-time env for tests: `TURBINE_STUB_ABI999`, `TURBINE_STUB_GFX942`
  Covers: S-7 AC `shim::tests::abi_and_arch_mismatch_are_fatal`; S-1 (nothing links HIP at build time)
  Depends on: Task 5

- [ ] Write failing test `turbine-kernels shim::tests::abi_and_arch_mismatch_are_fatal`: loading the ABI-999 stub fails with `kernel ABI version mismatch: library 999, expected 1`; the `gfx942` stub loads, reports `build_archs() == ["gfx942"]`, and `create_context` for a mocked R9700 `DeviceInfo` (arch `gfx1201`, vendor_index 0) fails with `device arch gfx1201 not in library build archs gfx942`; `/nonexistent/libturbine_hip.so` fails with a message starting `cannot load /nonexistent/libturbine_hip.so`. Run: `cargo test -p turbine-kernels shim::tests::abi_and_arch_mismatch_are_fatal` — expect FAIL
- [ ] Implement `build.rs` (invoke `cc::Build::new().get_compiler()` with `-shared -fPIC -I../../kernels/include -DSTUB_ABI=… -DSTUB_BACKEND="hip" -DSTUB_ARCHS="…"`, emit `cargo:rustc-env`), the FFI, `ShimLibrary`, `ShimContext` and `ShimProvider`; every `unsafe` block carries `// SAFETY:` naming the ownership rule it relies on; `ShimContext::drop` calls `turbine_ctx_destroy` exactly once and every `DeviceBuffer` holds an `Arc` of the context.
- [ ] Run: `cargo test -p turbine-kernels` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): runtime-loaded shim bindings with ABI, backend and arch checks`

## Task 7: Model config parsing and the architecture allowlist

Files: `crates/turbine-model/Cargo.toml` (new crate, `unsafe_code = "forbid"`, deps with reason comments: `safetensors`, `tokenizers` pinned `=0.21.*` with default features off and `fancy-regex` — the version `toktrie_hf_tokenizers` 1.8 builds against, so Phase 2 needs no tokenizer upgrade — `minijinja`, `minijinja-contrib` pycompat, `smallvec`, `half`, `rand_chacha`, `rand_core` `os_rng`), `crates/turbine-model/src/lib.rs` (modules, `ModelError`), `crates/turbine-model/src/config.rs`, `crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct/{config.json,generation_config.json}`
Interfaces:

- `pub enum ModelError { Io { path, detail }, Pickle { path }, Unsupported { field, value, supported }, Safetensors { file, tensor, rule }, MissingTensor(String), Budget(String), Template(String), Kernel(#[from] KernelError) }` + `From<MemoryError>`
- `pub enum Architecture { Llama }` (`#[non_exhaustive]`, `as_str() -> "LlamaForCausalLM"`); `pub enum RopeScaling { Llama3 { factor: f64, low_freq_factor: f64, high_freq_factor: f64, original_max_position_embeddings: u32 } }`
- `pub struct ModelArchConfig { architecture, num_layers, hidden, num_attention_heads, num_kv_heads, head_dim, intermediate: u32, rms_norm_eps: f32, rope_theta: f64, rope_scaling: Option<RopeScaling>, tie_word_embeddings: bool, vocab_size, max_position_embeddings: u32, eos_token_ids: SmallVec<[u32; 4]> }`
- `pub struct GenerationConfig { eos_token_ids: SmallVec<[u32; 4]>, bos_token_id: Option<u32>, temperature: Option<f32>, top_p: Option<f32>, top_k: Option<i32> }` (contract addition, with `pub fn load_generation_config(dir: &Path) -> Result<GenerationConfig, ModelError>`)
- `pub fn load_model_config(dir: &Path) -> Result<ModelArchConfig, ModelError>`; `ModelArchConfig::shape(&self) -> ModelShape`; `ModelArchConfig::kv_layout(&self, block_tokens: u32) -> KvLayout`; `ModelArchConfig::check_supported_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelError>` (contract addition)
  Covers: S-3 AC `config::tests::parses_target_config`, `config::tests::rejects_unsupported`
  Depends on: Task 6

- [ ] Fetch fixtures (the gated meta-llama repo is not needed for config/tokenizer files; the ungated mirror carries byte-identical tokenizer files — confirm with `sha256sum` against the novanas copy in Task 21):
  ```
  REV=006f5dcd1393c3add266de40994ba96225e9689d
  D=crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do curl -fsSL -o $D/$f https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/resolve/$REV/$f; done
  ```
- [ ] Write failing test `turbine-model config::tests::parses_target_config`: the fixture gives 28 layers, hidden 3072, 24 heads, 8 KV heads, head_dim 128, intermediate 8192, vocab 128256, tied, rope theta 500000 with `Llama3 { factor: 32, low: 1, high: 4, original: 8192 }`, EOS `[128001, 128008, 128009]`, and `kv_layout(16)` bytes per token 114 688 / block 1 835 008. Run: `cargo test -p turbine-model config::tests::parses_target_config` — expect FAIL
- [ ] Write failing test `turbine-model config::tests::rejects_unsupported`: `architectures: ["Qwen3MoeForCausalLM"]` → `unsupported architectures = Qwen3MoeForCausalLM; supported: LlamaForCausalLM`; a config with `quantization_config` → field `quantization_config`, supported `none`; a tiny checkpoint (Task 9 writer) whose safetensors holds an `F8_E4M3` tensor → `check_supported_weights` error naming field `tensor dtype`, the tensor, and supported `BF16`. Run: `cargo test -p turbine-model config::tests::rejects_unsupported` — expect FAIL
- [ ] Implement `load_model_config`: parse `config.json` with serde (head_dim explicit else hidden/heads; `torch_dtype` must be `bfloat16` if present), EOS from `generation_config.json` `eos_token_id` (int or list) falling back to `config.json`; a missing directory or `config.json` → `ModelError::Io` naming the path.
- [ ] Run: `cargo test -p turbine-model config::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): HF config parsing and architecture allowlist`

## Task 8: Safetensors index with header validation

Files: `crates/turbine-model/src/safetensors.rs` (`SafetensorsIndex`, `TensorEntry`, tests)
Interfaces:

- `pub struct TensorEntry { pub name: String, pub dtype: safetensors::Dtype, pub shape: Vec<usize>, pub file: PathBuf, pub range: std::ops::Range<u64> }` (absolute byte range)
- `SafetensorsIndex::open(dir: &Path) -> Result<SafetensorsIndex, ModelError>`, `get(&self, name: &str) -> Option<&TensorEntry>`, `entries(&self) -> impl Iterator<Item = &TensorEntry>`, `total_bytes(&self) -> u64`
- Error rules (text in `ModelError::Safetensors::rule`): `range outside file`, `overlapping ranges with <other>`, `shape × dtype size != byte range`, `unknown dtype <s>`, `header of <n> bytes exceeds 100 MiB`, `shard listed in index is missing`, `tensor listed twice`; pickle-only directory (`*.bin`, `*.pt`, `*.pth`, `*.ckpt`) → `ModelError::Pickle { path }` without opening the file
  Covers: S-2 AC `safetensors::tests::rejects_malformed_headers`
  Depends on: Task 7

- [ ] Write failing test `turbine-model safetensors::tests::rejects_malformed_headers`: handcrafted files (8-byte LE length + JSON header + data) with an out-of-file range, overlapping ranges, a `[2,2]` BF16 tensor spanning 6 bytes, dtype `"Q4"`, and a length prefix claiming 101 MiB (no 101 MiB allocation), plus an index naming a missing shard and a tensor listed in two shards — each rejected with the file path, tensor name and rule in the message. Run: `cargo test -p turbine-model safetensors::tests::rejects_malformed_headers` — expect FAIL
- [ ] Implement `open`: prefer `model.safetensors.index.json` (`weight_map`), else `model.safetensors`; read each header with `std::os::unix::fs::FileExt::read_exact_at` after checking the declared length ≤ 100 MiB; deserialize with `safetensors::tensor::Metadata`-compatible serde types, then validate against the file length from `metadata()`; symlinks are followed only to regular files.
- [ ] Run: `cargo test -p turbine-model safetensors::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): validated safetensors index`

## Task 9: Weight loader and the tiny synthetic checkpoint

Files: `crates/turbine-model/src/loader.rs` (`WeightSlot`, `llama_slots`, `WeightLoader`, `LoadedWeights`, tests), `crates/turbine-model/src/testing/mod.rs` (`TempDir`), `crates/turbine-model/src/testing/tiny.rs` (`write_tiny_llama`, `write_tiny_llama_with`, `TinyOptions`, `TinySpec`, tiny tokenizer and template)
Interfaces:

- `pub struct WeightSlot { pub name: String, pub shape: Vec<usize> }`, `pub fn llama_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot>` (`model.embed_tokens.weight`, per layer `input_layernorm`, `self_attn.{q,k,v,o}_proj`, `post_attention_layernorm`, `mlp.{gate,up,down}_proj`, `model.norm.weight`, `lm_head.weight` only when untied)
- `pub const MAX_STAGING_BYTES: usize = 256 << 20;`
- `WeightLoader::load(index: &SafetensorsIndex, slots: &[WeightSlot], mem: &Arc<dyn DeviceMemory>, staging_bytes: usize) -> Result<LoadedWeights, ModelError>`
- `pub struct LoadedWeights { pub tensors: HashMap<String, Tensor>, pub weight_bytes: u64, pub unexpected: Vec<String>, pub ignored: Vec<String> }`, `fn take(&mut self, name: &str) -> Result<Tensor, ModelError>`
- `pub struct TempDir`: `fn new(prefix: &str) -> TempDir`, `fn path(&self) -> &Path` (removed on drop)
- `pub struct TinyOptions { tied: bool, ship_lm_head: bool, omit: Vec<String>, extra: Vec<String>, template_with_tools: bool }` (Default: tied, no lm_head, Llama-3.2 template), `pub struct TinySpec { dir: PathBuf, config: ModelArchConfig, vocab: u32 }`, `pub fn write_tiny_llama(dir: &Path, seed: u64) -> TinySpec`, `pub fn write_tiny_llama_with(dir: &Path, seed: u64, opts: &TinyOptions) -> TinySpec`
- Tiny checkpoint: `LlamaForCausalLM`, 2 layers, hidden 64, 4 query / 2 KV heads, head_dim 16, intermediate 128, eps 1e-5, rope theta 10000 with llama3 scaling factor 8 / low 1 / high 4 / original 32, max positions 512, tied, BF16 weights from ChaCha8 seeded by `seed`; byte-level BPE `tokenizer.json` (256 byte symbols) with specials `<|begin_of_text|>`=256, `<|end_of_text|>`=257, `<|start_header_id|>`=258, `<|end_header_id|>`=259, `<|eot_id|>`=260, `<|eom_id|>`=261, `<|python_tag|>`=262 and a BOS post-processor; `tokenizer_config.json` carrying the exact Llama-3.2 `chat_template`; `generation_config.json` EOS `[257, 260]`. Loads in transformers (`AutoTokenizer`, `AutoModelForCausalLM`) so `hf_reference.py` can use it.
  Covers: S-2 AC `loader::tests::tensor_mapping`, `loader::tests::never_opens_pickle`; S-12 (tiny synthetic checkpoint)
  Depends on: Task 8

- [ ] Write failing test `turbine-model loader::tests::tensor_mapping`: on the tiny checkpoint with `HostMemory` every slot is filled with the expected shape and bytes equal to the file; a checkpoint written with `omit: ["model.layers.1.mlp.down_proj.weight"]` fails with `missing tensor model.layers.1.mlp.down_proj.weight` before any tensor is allocated; `extra: ["model.extra.weight"]` lands in `unexpected`; tied + `ship_lm_head` lands in `ignored`; an untied checkpoint loads `lm_head.weight`; a 64-byte staging buffer loads identical bytes. Run: `cargo test -p turbine-model loader::tests::tensor_mapping` — expect FAIL
- [ ] Write failing test `turbine-model loader::tests::never_opens_pickle`: a directory holding only `pytorch_model.bin` with mode 0o000 → error containing the file path and `pickle formats are not supported`. Run: `cargo test -p turbine-model loader::tests::never_opens_pickle` — expect FAIL
- [ ] Implement the loader: validate every slot (presence, BF16, shape) first, then allocate each `Tensor` on `mem` and upload through one reusable staging `Vec` of `min(staging_bytes, MAX_STAGING_BYTES)` bytes with `read_exact_at` in chunks (no `mmap`); log `event="unexpected_tensor"` / `event="ignored_tensor"` at WARN; write the tiny checkpoint with `safetensors::serialize_to_file`.
- [ ] Run: `cargo test -p turbine-model loader::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): positioned-read weight loader and tiny synthetic checkpoint`

## Task 10: Startup memory budget

Files: `crates/turbine-model/src/budget.rs` (`BudgetTerms`, `available_bytes`, `check_budget`, `host_mem_available`, tests)
Interfaces:

- `pub struct BudgetTerms { pub weights: u64, pub kv_reservation: u64, pub workspace: u64, pub emergency_reserve: u64, pub available: u64 }`
- `pub fn available_bytes(kind: MemoryKind, device_free: u64, host_mem_available: Option<u64>) -> u64` (dedicated → device free; unified → min of both, device free when host unknown)
- `pub fn check_budget(terms: &BudgetTerms) -> Result<(), ModelError>` — message `weights <n> B + kv_reservation <n> B + workspace <n> B + emergency_reserve <n> B = <sum> B > available <n> B`
- `pub fn host_mem_available(meminfo_path: &Path) -> Option<u64>` (contract addition; `MemAvailable` kB × 1024)
- Log event `memory_budget` (INFO) with every term and `available_bytes`
  Covers: S-5 AC `budget::tests::refuses_before_loading`, `budget::tests::available_memory_by_kind`
  Depends on: Task 9

- [ ] Write failing test `turbine-model budget::tests::refuses_before_loading`: open the tiny checkpoint's index, truncate `model.safetensors` to its header length (any weight read would now fail with `Io`), compute terms (weights = `index.total_bytes()`, kv = `kv_layout(1).bytes_per_token() × 512`, workspace 1 MiB, reserve 1 MiB) against 1 KiB available → `ModelError::Budget` whose text lists weights, kv_reservation, workspace, emergency_reserve and available numbers. Run: `cargo test -p turbine-model budget::tests::refuses_before_loading` — expect FAIL
- [ ] Write failing test `turbine-model budget::tests::available_memory_by_kind`: dedicated (10 GiB free, host 5 GiB) → 10 GiB; unified with host 5 GiB → 5 GiB and with host 20 GiB → 10 GiB; unified with host unknown → 10 GiB; `host_mem_available` on a written `/proc/meminfo` fixture with `MemAvailable: 126877932 kB` → 129923002368. Run: `cargo test -p turbine-model budget::tests::available_memory_by_kind` — expect FAIL
- [ ] Implement the module (pure functions; nothing here touches weight files).
- [ ] Run: `cargo test -p turbine-model budget::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): startup memory budget`

## Task 11: Tokenizer and incremental detokenization

Files: `crates/turbine-model/src/tokenizer.rs` (`Tokenizer`, `IncrementalDetokenizer`, test)
Interfaces:

- `Tokenizer::from_file(path: &Path) -> Result<Tokenizer, ModelError>`, `encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, ModelError>`, `decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, ModelError>`, `vocab_size(&self) -> u32`, `token_to_id(&self, token: &str) -> Option<u32>`, `id_to_token(&self, id: u32) -> Option<String>`, `inner(&self) -> &tokenizers::Tokenizer`
- `IncrementalDetokenizer::new(tokenizer: Arc<Tokenizer>) -> IncrementalDetokenizer`, `push(&mut self, token: u32) -> Option<String>`, `flush(&mut self) -> Option<String>` (contract addition)
  Covers: S-4 AC `tokenizer::tests::incremental_detokenize_utf8`
  Depends on: Task 7 (fixtures)

- [ ] Write failing test `turbine-model tokenizer::tests::incremental_detokenize_utf8`: encoding `Hello 世界 👩‍👩‍👧‍👦 🇧🇪 naïve café 日本語テキスト` with the committed fixture tokenizer and pushing the ids one by one yields chunks whose concatenation plus `flush()` equals `decode(all)`, no chunk contains U+FFFD, and at least one `push` returns `None`. Run: `cargo test -p turbine-model tokenizer::tests::incremental_detokenize_utf8` — expect FAIL
- [ ] Implement with the prefix/read-offset method: decode `ids[prefix..]` and `ids[prefix..read]` (skip special tokens), emit the difference only when the new text does not end in U+FFFD, then advance both offsets.
- [ ] Run: `cargo test -p turbine-model tokenizer::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): tokenizer and UTF-8-safe incremental detokenizer`

## Task 12: Chat template rendering (minijinja + pycompat)

Files: `crates/turbine-model/src/chat_template.rs` (`ChatTemplate`, strftime, Python-compatible `tojson`, tests), `scripts/golden/render_fixture.py` (PEP 723 fixture generator, run once), `crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct/expected_renders.json` (generated once with transformers), `Cargo.toml` (enable `minijinja` features `loader`, `json`, `loop_controls`, `preserve_order` and `serde_json` feature `preserve_order` — required for byte-identical `tojson` output)
Interfaces:

- `ChatTemplate::load(path: &Path) -> Result<ChatTemplate, ModelError>` (`.jinja` with specials from a sibling `tokenizer_config.json`, or `tokenizer_config.json` `chat_template`; `bos_token`/`eos_token` as string or `{"content"}`)
- `ChatTemplate::resolve(model_dir: &Path, explicit: Option<&Path>) -> Result<ChatTemplate, ModelError>` (contract addition; default `chat_template.jinja` else `tokenizer_config.json`)
- `ChatTemplate::render(&self, messages: &[serde_json::Value], tools: Option<&[serde_json::Value]>, add_generation_prompt: bool, kwargs: &serde_json::Map<String, serde_json::Value>) -> Result<String, ModelError>` (Phase 1 always passes `tools: None`)
- `ChatTemplate::renders_tools(&self) -> bool` (contract addition, used by Phase 2's `tool_call_parser` default)
- Host functions `raise_exception(msg)` → `ModelError::Template(msg)`, `strftime_now(fmt)` in UTC (own formatter for `%d %b %B %Y %y %m %H %M %S %a %A %j %p %I %e %%`); `tojson` byte-identical to Python `json.dumps(x, ensure_ascii=False, indent=…)` with insertion order preserved (minijinja `preserve_order`)
  Covers: S-4 AC `chat_template::tests::renders_target_template`
  Depends on: Task 11

- [ ] Generate `expected_renders.json` once (fixture-generation time only) with:
  ```
  uv run --with 'transformers==4.57.1' --with jinja2 python3 scripts/golden/render_fixture.py crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct
  ```
  where `scripts/golden/render_fixture.py` (PEP 723, pins as above) renders system+user and user-only conversations with `add_generation_prompt=True, date_string="26 Jul 2024"` via `AutoTokenizer.apply_chat_template(..., tokenize=False)` and records text, token ids (`encode(text, add_special_tokens=False)`) and `transformers_version`.
- [ ] Write failing test `turbine-model chat_template::tests::renders_target_template`: both conversations render to the fixture strings and token ids exactly; without `date_string` the output contains `Today Date: ` followed by today's UTC date in `%d %b %Y`; an inline template calling `raise_exception('bad role')` returns `ModelError::Template` with `bad role`. Run: `cargo test -p turbine-model chat_template::tests::renders_target_template` — expect FAIL
- [ ] Implement with a `minijinja::Environment` (`set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback)`, custom `tojson` filter, `trim` matching Jinja2, `bos_token`/`eos_token`/`add_generation_prompt` globals, kwargs merged into the context); template syntax errors at `load` name the file.
- [ ] Run: `cargo test -p turbine-model chat_template::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): Llama-3.2 chat template rendering with pycompat`

## Task 13: Llama executor on the kernel registry

Files: `crates/turbine-model/src/executor/mod.rs` (`ModelExecutor`, `BatchInput`, `Logits`), `crates/turbine-model/src/executor/rope.rs` (`inv_freq`, test), `crates/turbine-model/src/executor/llama.rs` (`LlamaExecutor`), `crates/turbine-model/tests/tiny_model.rs` (`cpu_forward_matches_naive`, `hip_matches_cpu`)
Interfaces:

- `pub struct BatchInput<'a> { pub tokens: &'a [u32], pub positions: &'a [u32] }` (Phase 1: one sequence with consecutive positions starting at or before the cached length; Phase 2 adds `seqs` and `kv`)
- `pub struct Logits { pub rows: usize, pub vocab: usize, pub data: Vec<f32> }`, `fn row(&self, r: usize) -> &[f32]`
- `pub trait ModelExecutor: Send { fn shape(&self) -> &ModelShape; fn kv_layout(&self) -> &KvLayout; fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError>; }`
- `pub fn rope::inv_freq(theta: f64, rotary_dim: u32, scaling: Option<&RopeScaling>) -> Vec<f32>` (FP64 host computation of the transformers llama3 formula, stored F32)
- `LlamaExecutor::requirements(cfg: &ModelArchConfig) -> Vec<OpRequirement>`, `LlamaExecutor::workspace_bytes(cfg: &ModelArchConfig, max_tokens: u32) -> u64`, `LlamaExecutor::new(cfg: &ModelArchConfig, weights: LoadedWeights, registry: Arc<KernelRegistry>, mem: Arc<dyn DeviceMemory>, max_seq_len: u32, max_forward_tokens: u32) -> Result<LlamaExecutor, ModelError>`
- Forward: embedding → per layer RMSNorm → Q/K/V GEMMs (K and V written straight into the contiguous cache `[layers, 2, max_seq_len, kv_heads, head_dim]` BF16) → RoPE on Q and the new K rows → attention (`Decode` config when one token, else `Prefill`) → O GEMM → residual add → RMSNorm → gate/up → SiLU·up → down → residual add; final RMSNorm on the last row; LM head (`embed_tokens` when tied) with F32 output; one device-to-host copy of logits
  Covers: S-8, S-12 AC `tiny_model cpu_forward_matches_naive` (the ignored `tiny_model hip_matches_cpu` is written here; its acceptance run belongs to Task 21)
  Depends on: Tasks 5, 9

- [ ] Write failing test `turbine-model rope::tests::llama3_bands_follow_transformers`: for theta 500000, dim 128, factor 32/1/4/8192 the highest frequency is unchanged, the lowest is divided by 32, and a medium-band frequency lies strictly between. Run: `cargo test -p turbine-model rope::tests` — expect FAIL
- [ ] Write failing test `turbine-model --test tiny_model cpu_forward_matches_naive`: tiny checkpoint (seed 7), CPU registry; prefill of 20 tokens then 30 decode steps (positions past the tiny original length 32 so the scaled bands are exercised) give logits within 1e-4 of an independent naive f32 implementation in the test that reads the safetensors directly and rounds to BF16 at the same op boundaries. Run: `cargo test -p turbine-model --test tiny_model cpu_forward_matches_naive` — expect FAIL
- [ ] Write the ignored test `turbine-model --test tiny_model hip_matches_cpu`: starts with `require_backend("hip")`, loads `TURBINE_KERNEL_LIBRARY`, runs the same prompt on the HIP and CPU registries and asserts logits within 2e-2 and 32 identical greedy tokens.
- [ ] Implement the executor: activation buffers sized once for `max_forward_tokens`; the registry lookups use the same configs `requirements` lists; `forward` rejects empty batches, non-consecutive positions and positions ≥ `max_seq_len` with `KernelError::InvalidArgument`.
- [ ] Run: `cargo test -p turbine-model --test tiny_model cpu_forward_matches_naive && cargo test -p turbine-model rope::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): Llama executor over the kernel registry`

## Task 14: Sampler, generation loop and model metrics

Files: `crates/turbine-model/src/sampler.rs` (`Sampler`, `SampledToken`, `argmax`, `log_softmax`, unit tests), `crates/turbine-model/src/generate.rs` (`generate`, `Generation`, `GenerateOptions`, tests), `crates/turbine-model/src/metrics.rs` (`ModelMetrics`, `ForwardPhase`)
Interfaces:

- `Sampler::new(params: &SamplingParams) -> Sampler` (ChaCha8 from `seed`, else OS entropy), `sample(&mut self, logits: &mut [f32]) -> SampledToken` (Phase 2 adds the `mask` parameter of contract §10), `pub struct SampledToken { token: u32, logprob: f32, top_logprobs: Vec<(u32, f32)> }` — logprobs are the raw log-softmax; greedy = argmax with ties to the lower id
- `pub struct GenerateOptions<'a> { pub max_seq_len: u32, pub metrics: Option<&'a ModelMetrics> }`
- `pub fn generate<'a>(exec: &'a mut dyn ModelExecutor, tokenizer: Arc<Tokenizer>, req: &'a GenerationRequest, cancel: &'a CancelFlag, opts: GenerateOptions<'a>) -> Generation<'a>` with `impl Iterator<Item = GenerationEvent>`: `Started`, one `Token` per generated token (EOS included; text empty while held), `Finished { reason, usage }`, or `Error { code: InternalError }`; ends without `Finished` when `cancel` fires
- `ModelMetrics::register(reg: &MetricsRegistry) -> ModelMetrics` with `load_seconds` (`turbine_model_load_seconds`), `weight_bytes` (`turbine_model_weight_bytes{format}`), `forward_seconds` (`turbine_forward_seconds{phase}`), `observe_forward(&self, phase: ForwardPhase, seconds: f64)`
  Covers: S-9 AC `generate::tests::stop_conditions`, `generate::tests::seeded_sampling_is_deterministic`; S-14 (forward histogram, model gauges)
  Depends on: Tasks 11, 13

- [ ] Write failing test `turbine-model generate::tests::stop_conditions`: with the tiny tokenizer and a scripted executor emitting one-hot logits, generation ends with `stop` on each of EOS 257 and 260, on the stop string `"ab"` spanning two tokens (output excludes `"ab"`, the held `"a"` never streams), with `length` at `max_tokens: 5`, and with `length` when prompt + generated reaches `max_seq_len`; with `ignore_eos: true` it continues past 257. Run: `cargo test -p turbine-model generate::tests::stop_conditions` — expect FAIL
- [ ] Write failing test `turbine-model generate::tests::seeded_sampling_is_deterministic`: on the real tiny executor, two requests with `seed: 7, temperature: 0.8, top_p: 0.9, top_k: 50` produce identical 16-token outputs, `seed: 8` differs, and `temperature: 0` with `top_p: 0.1, top_k: 3` equals the argmax sequence. Run: `cargo test -p turbine-model generate::tests::seeded_sampling_is_deterministic` — expect FAIL
- [ ] Implement: temperature → top-k → top-p (sorted descending, ties by id, 24-bit uniform draw); stop strings search the held text, emit only the part that cannot begin a stop string; at finish the detokenizer is flushed; `max_tokens: 0` finishes with `length` and no forward; forward durations go to `turbine_forward_seconds{phase="prefill"|"decode"}`.
- [ ] Run: `cargo test -p turbine-model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): host sampler and single-request generation loop`

## Task 15: OpenAI request types and validation in `turbine-api`

Files: `crates/turbine-api/src/openai/mod.rs` (module list), `crates/turbine-api/src/openai/request.rs` (`OpenAiRequest` + validation + helpers), `crates/turbine-api/src/error.rs` (new constructors), `crates/turbine-api/src/backend.rs` (`submit`, `InferenceRequest`, `GenerationStream`, readiness reasons), `crates/turbine-api/Cargo.toml` (add `tokio` sync, `futures-util`, `uuid`), `crates/turbine-api/tests/openai.rs` (request-validation tests with a scripted backend)
Interfaces:

- `pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>`
- `InferenceBackend` gains `fn submit(&self, req: InferenceRequest) -> BoxFuture<'_, Result<GenerationStream, ApiError>>` (default returns `ApiError::model_not_loaded()` so the Phase 0 `NoModel` still compiles), `fn token_text(&self, token_id: u32) -> String` (default `token_id:<id>`; contract addition), `fn record_rejection(&self, endpoint: Endpoint, code: ErrorCode)` (default no-op; contract addition — lets the server-owned `turbine_requests_total{outcome="rejected"}` count API-side rejections)
- `pub struct InferenceRequest { pub id: RequestId, pub endpoint: Endpoint, pub body: OpenAiRequest, pub http_request_id: String }` (`id` is a contract addition: the API creates it and renders `cmpl-<uuid>` / `chatcmpl-<uuid>`)
- `pub type GenerationStream = tokio::sync::mpsc::Receiver<GenerationEvent>`
- `NotReadyReason` gains `LoadingModel`, `ModelLoadFailed`, `DeviceError`
- `ApiError::{model_not_found(name), unsupported_parameter(field), context_length_exceeded(msg), engine_busy(), template_error(msg), invalid_request(msg)}` per contract §14.3 (engine_busy: 429 `rate_limit_error`, `retry_after: Some(1)`)
- `pub struct OpenAiRequest { model, prompt: Option<PromptInput>, messages: Option<Vec<ChatMessageIn>>, max_tokens, max_completion_tokens: Option<u32>, temperature, top_p: Option<f32>, top_k: Option<i32>, seed: Option<u64>, stop: Option<StopInput>, stream: Option<bool>, stream_options: Option<StreamOptions>, logprobs: Option<LogprobsField>, top_logprobs: Option<u32>, echo: Option<bool>, n: Option<u32>, ignore_eos, return_tokens_as_token_ids: Option<bool>, chat_template_kwargs: Option<Map<String, Value>>, tools, tool_choice, response_format, logit_bias, presence_penalty, frequency_penalty, repetition_penalty, best_of, suffix, parallel_tool_calls: Option<Value> }` (unknown fields ignored), `fn validate(&self, endpoint: Endpoint) -> Result<(), ApiError>`, `fn messages_json(&self) -> Vec<Value>`, `fn stop_strings(&self) -> Vec<String>`, `fn max_tokens(&self) -> Option<u32>`, `fn include_usage(&self) -> bool`, `fn logprobs_n(&self, endpoint: Endpoint) -> Option<u32>`
  Covers: S-10 (supported request fields, unsupported → 400); the end-to-end assertions are `tiny_server request_validation` in Task 17
  Depends on: Task 1, Phase 0 plan Task 4

- [ ] Write failing test `turbine-api --test openai request_validation_rules`: `tools: [{…}]`, `response_format: {"type":"json_object"}`, `n: 2`, `logit_bias: {"5": 1}`, `presence_penalty: 0.5`, `echo: true` and an `image_url` content part each yield 400 `unsupported_parameter` naming the field; `temperature: 3`, `top_p: 0`, `top_k: 0`, five stop strings and completions `logprobs: true` yield 400 `invalid_request`; `response_format: {"type":"text"}`, `n: 1` and an unknown field `foo` are accepted. Run: `cargo test -p turbine-api --test openai request_validation_rules` — expect FAIL
- [ ] Implement the types with serde untagged enums `PromptInput { Text(String), Tokens(Vec<u32>) }`, `MessageContent { Text(String), Parts(Vec<ContentPart>) }`, `StopInput { One(String), Many(Vec<String>) }`, `LogprobsField { Bool(bool), Int(u32) }`; roles other than `system`/`user`/`assistant` → `unsupported_parameter` (`messages.role`).
- [ ] Run: `cargo test -p turbine-api` — expect PASS (Phase 0 `api` tests unchanged)
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-api): OpenAI request types and phase 1 validation`

## Task 16: OpenAI responses and SSE streaming

Files: `crates/turbine-api/src/openai/response.rs` (non-streaming aggregation), `crates/turbine-api/src/openai/stream.rs` (SSE chunk mapping), `crates/turbine-api/src/routes/openai.rs` (completions/chat handlers replace the Phase 0 503 bodies), `crates/turbine-api/tests/openai.rs` (response-shape tests)
Interfaces:

- Handlers `completions`, `chat_completions`: parse JSON (malformed → 400 `invalid_request`), `validate`, `model` must be an id of `backend.models()` (else 404 `model_not_found` + `record_rejection`), create `RequestId::new_v4()`, `backend.submit`; an `Err` before the first event is a plain HTTP error even when `stream: true`
- `pub(crate) async fn collect(stream: GenerationStream, …) -> Result<Value, ApiError>` builds `{"id":"cmpl-…","object":"text_completion","created","model","choices":[{"index":0,"text","logprobs","finish_reason"}],"usage":{prompt_tokens,completion_tokens,total_tokens}}` or `{"id":"chatcmpl-…","object":"chat.completion",…,"message":{"role":"assistant","content"},…}`; an `Error` event maps to its status (500 `internal_error`)
- `pub(crate) fn sse(stream: GenerationStream, …) -> Sse<impl Stream<Item = Result<Event, Infallible>>>`: chat first chunk `delta: {"role":"assistant","content":""}` from `Started`; content chunks from non-empty `Token.text`; a finish chunk with `finish_reason`; a `{"choices":[],"usage":{…}}` chunk when `include_usage`; `data: [DONE]`; a mid-stream `Error` event → `data: {"error":{"message","type","code"}}` then `data: [DONE]` (C-3)
- Logprobs: completions `{tokens, token_logprobs, top_logprobs, text_offset}`; chat `{"content":[{token, logprob, bytes, top_logprobs:[{token, logprob, bytes}]}]}`; token strings `token_id:<id>` when `return_tokens_as_token_ids`, else `backend.token_text(id)`
- The receiver lives inside the response body stream, so a client disconnect drops it (the server's cancellation signal)
  Covers: S-10 (streaming and non-streaming shapes); end-to-end in Task 17 `tiny_server completions_stream_and_non_stream`
  Depends on: Task 15

- [ ] Write failing test `turbine-api --test openai response_shapes_and_stream_order`: a scripted backend emitting `Started`, tokens `"Hel"`, `""`, `"lo"`, `Finished{stop, usage 3/3}` gives a non-streaming completion with text `Hello`, `finish_reason: "stop"`, usage total 6; the chat stream yields role chunk, two content chunks, finish chunk, usage chunk, `[DONE]` in that order; a backend emitting `Error{internal_error}` after one token yields an error event followed by `[DONE]`; `logprobs: 2` with `return_tokens_as_token_ids` renders `"token_id:17"` keys. Run: `cargo test -p turbine-api --test openai response_shapes_and_stream_order` — expect FAIL
- [ ] Implement with `axum::response::sse::{Sse, Event}` over an `async_stream`-free `futures_util::stream::unfold` on the receiver; `created` is Unix seconds at request start.
- [ ] Run: `cargo test -p turbine-api` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-api): OpenAI completion and chat responses with SSE`

## Task 17: `turbine-server` model startup, single generation slot and Phase 1 metrics

Files: `crates/turbine-server/src/startup.rs` (P1 startup order, readiness state, exit handling), `crates/turbine-server/src/model.rs` (new: provider loading, config/tokenizer/template resolution, budget, weight load, warm-up), `crates/turbine-server/src/generation.rs` (new: `ModelBackend` implementing `InferenceBackend`, single-slot generation thread), `crates/turbine-server/src/metrics.rs` (new: `ServerMetrics`), `crates/turbine-server/src/exit.rs` (unchanged codes), `crates/turbine-server/Cargo.toml` (add turbine-tensor, turbine-kernels, turbine-model, prometheus-client), `crates/turbine-server/tests/tiny_server.rs` (new)
Interfaces:

- Startup (contract §16.3): config (exit 2) → discovery → provider for `execution.backend` (`hip`: `ShimLibrary::search_paths` + `load` + `create_context(inventory device)`, vendor must be AMD; `cpu`: `HostMemory` with capacity = host MemAvailable or device total) → `load_model_config`, `Tokenizer::from_file`, `ChatTemplate::resolve`, `max_seq_len` ≤ `max_position_embeddings` (exit 1 otherwise) → `KernelRegistry::build(…, LlamaExecutor::requirements)` → `check_budget` (weights + KV for `max_seq_len` + `workspace_bytes(max_seq_len)` + `reliability.emergency_vram_reserve` vs `available_bytes`) → bind (`/ready` 503 `loading_model`) → `WeightLoader::load` → one-token warm-up → `/ready` 200; any failure after bind: `/ready` 503 `model_load_failed` for ≤ 1 s, exit 1
- `served_name` default: `<org>/<name>` from `…/models--<org>--<name>/snapshots/<rev>`, else the last path component
- `ModelBackend: InferenceBackend + Readiness + Diagnostics` — `submit` renders the template (chat, `add_generation_prompt`, `chat_template_kwargs`) or tokenizes the prompt (`add_special_tokens: true`) / takes token ids; 0 tokens → 400; prompt + `max_tokens` > `max_seq_len` → 400 `context_length_exceeded` (default `max_tokens` = remaining context); template error → 400 `template_error`; defaults for temperature/top_p/top_k from `generation_config.json`; slot busy → 429 `engine_busy` + `retry-after: 1`; returns a 64-event channel
- Generation thread: owns the executor; runs `generate`, `blocking_send`s events, treats a closed receiver as cancellation (`outcome="cancelled"`, slot freed); 3 consecutive failed requests → `/ready` 503 `device_error`, exit 1 (C-25)
- `ServerMetrics::register(reg) -> ServerMetrics`: `turbine_requests_total{endpoint,outcome}`, `turbine_request_ttft_seconds`, `turbine_request_itl_seconds`, `turbine_request_e2e_seconds`, `turbine_tokens_total{kind}`
- `/turbine/v1/status` adds `"model":{"served_name","architecture","weight_bytes","load_seconds"}`; `/v1/models` lists one `ModelCard { id: served_name, object: "model", owned_by: "turbine", max_model_len }`
  Covers: S-10 AC `tiny_server completions_stream_and_non_stream`, `tiny_server single_slot_and_cancel`, `tiny_server request_validation`, `tiny_server startup_failures_exit_1`; S-14 AC `tiny_server phase1_metrics`; S-5 (budget before weights)
  Depends on: Tasks 10, 12, 14, 16

- [ ] Write failing test `turbine-server --test tiny_server completions_stream_and_non_stream`: start the binary with `execution.backend: cpu` on a tiny checkpoint and a free port, wait for `/ready` 200, assert `/v1/models`, a non-streaming completion, a streaming completion with `include_usage` (order and usage equal to token counts) and the same for chat. Run: `cargo test -p turbine-server --test tiny_server completions_stream_and_non_stream` — expect FAIL
- [ ] Write failing test `tiny_server single_slot_and_cancel`: a long streaming request (`max_tokens: 400, ignore_eos: true`) holds the slot; a concurrent request gets 429 `engine_busy` with `retry-after`; dropping the first client lets a new request start within 1 s and `turbine_requests_total{endpoint="/v1/completions",outcome="cancelled"} 1`. Run: `cargo test -p turbine-server --test tiny_server single_slot_and_cancel` — expect FAIL
- [ ] Write failing test `tiny_server request_validation`: 400 `unsupported_parameter` for `tools`, `response_format`, `n: 2`, `logit_bias`, an image part; 404 `model_not_found` for a wrong `model`; 400 `context_length_exceeded` for a 600-token prompt with `max_seq_len: 512`; an unknown field is ignored. Run: `cargo test -p turbine-server --test tiny_server request_validation` — expect FAIL
- [ ] Write failing test `tiny_server startup_failures_exit_1`: exit 1 with the cause on stderr for a missing `model.path`, a pickle-only directory, `architectures: ["Qwen3MoeForCausalLM"]`, and backend `hip` with `kernel_library: /nonexistent/libturbine_hip.so`; exit 2 with `phase-2b-nvidia` for backend `cuda`. Run: `cargo test -p turbine-server --test tiny_server startup_failures_exit_1` — expect FAIL
- [ ] Write failing test `tiny_server phase1_metrics`: after one completion `/metrics` contains `turbine_model_load_seconds`, `turbine_model_weight_bytes{format="bf16"}`, `turbine_kernel_provider_selected`, `turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 1`, a `turbine_request_ttft_seconds_count 1` and `turbine_tokens_total{kind="generated"}` equal to the completion tokens. Run: `cargo test -p turbine-server --test tiny_server phase1_metrics` — expect FAIL
- [ ] Implement the startup path, `ModelBackend` and the generation thread as the Interfaces list; the slot is an `AtomicBool` claimed in `submit` and released by the thread when the request ends or is cancelled.
- [ ] Run: `cargo test -p turbine-server` — expect PASS (Phase 0 `server_cli` tests unchanged)
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): model startup, single-slot generation and phase 1 metrics`

## Task 18: `turbine-golden` compare and capture

Files: `benches/turbine-bench/src/golden/mod.rs` (module list), `benches/turbine-bench/src/golden/fixture.rs` (`PromptRecord`, `ReferenceRecord`, `Tolerance`, JSONL read, temp-file-then-rename write), `benches/turbine-bench/src/golden/compare.rs` (`compare_prompt`, `judge`, report), `benches/turbine-bench/src/golden/client.rs` (replay against an OpenAI endpoint), `benches/turbine-bench/src/bin/turbine-golden.rs` (CLI), `benches/turbine-bench/src/lib.rs` (`pub mod golden;`), `benches/turbine-bench/Cargo.toml` (`[[bin]] turbine-golden`, dev-dep axum), `benches/turbine-bench/tests/golden.rs`
Interfaces:

- `pub struct PromptRecord { id, kind: PromptKind { Completion, Chat }, prompt: Option<String>, messages: Option<Vec<Value>>, max_tokens: u32, chat_template_kwargs: Option<Map<String, Value>> }`
- `pub struct ReferenceRecord { id, engine, model, captured: String, prompt_token_ids: Vec<u32>, tokens: Vec<u32>, top_logprobs: Vec<Vec<(u32, f32)>> }` (serialised `[[[id, lp], …], …]`)
- `pub struct Tolerance { min_identical_prefix: usize, min_prompts_passing: usize, top_k: usize, max_abs_logprob_diff: f32, margin_nats: f32 }`
- `pub fn compare_prompt(reference: &ReferenceRecord, got_tokens: &[u32], got_top: &[Vec<(u32, f32)>], tol: &Tolerance) -> PromptVerdict` (`identical_prefix`, `first_divergence: Option<usize>`, `margin_at_divergence: Option<f32>`, `max_abs_logprob_diff`, `passed`); a candidate that stops early diverges at its length; a reference top-k id missing from the candidate's top-20 violates the logprob bound
- `pub fn judge(verdicts: &[PromptVerdict], tol: &Tolerance) -> bool`
- CLI: `turbine-golden compare --url <base> --reference <reference.jsonl> [--prompts <prompts.jsonl>] [--model <name>] [--tolerance <tolerance.json>] [--output text|json]` — `--prompts` (contract addition: the reference records carry no prompt text) defaults to `prompts.jsonl` in the reference's parent directory; `--tolerance` defaults beside the reference; requests use `temperature: 0`, top-20 logprobs, `return_tokens_as_token_ids: true`, `ignore_eos: true` (the reference always runs `max_tokens` positions), non-streaming; exit 0/1/2. `turbine-golden capture --url <base> --prompts <prompts.jsonl> --out <reference.jsonl> [--model <name>] [--top-logprobs 20]` — engine from `system_fingerprint`, writes `<out>.tmp` then renames; exit 1 leaves no file
  Covers: S-11 AC `golden capture_and_compare_roundtrip`
  Depends on: Phase 0 plan Tasks 6–7

- [ ] Write failing test `turbine-bench --test golden capture_and_compare_roundtrip`: an in-test axum mock serving `/v1/models`, `/v1/completions` and `/v1/chat/completions` from a scripted table; `capture` into a temp file; `compare` against the same mock exits 0; a mock flipping the token at position 5 where the reference margin is 2 nats exits 1 and reports position 5 and margin 2.0; a flip where the margin is 0.3 exits 0; shifting one top-5 logprob by 0.2 exits 1 (test tolerance `{"min_identical_prefix":8,"min_prompts_passing":2,"top_k":5,"max_abs_logprob_diff":0.15,"margin_nats":0.5}`). Run: `cargo test -p turbine-bench --test golden capture_and_compare_roundtrip` — expect FAIL
- [ ] Implement the library and binary with `clap` (usage errors exit 2), `reqwest` JSON calls, and parsing of both logprob shapes (completions `top_logprobs` maps keyed `token_id:<id>`; chat `logprobs.content[].top_logprobs[]`).
- [ ] Run: `cargo test -p turbine-bench` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-bench): turbine-golden compare and capture`

## Task 19: HF reference script, committed prompts, tolerance and in-process golden tests

Files: `scripts/golden/hf_reference.py` (PEP 723, exact pins `torch==2.9.0`, `transformers==4.57.1`, `safetensors==0.6.2`), `tests/golden/prompts.jsonl` (16 prompts), `tests/golden/llama-3.2-3b-instruct/tolerance.json`, `crates/turbine-model/tests/golden.rs` (`hf_reference_matches_cpu`, `logits_match_reference`)
Interfaces:

- `uv run scripts/golden/hf_reference.py --model-dir <dir> --prompts tests/golden/prompts.jsonl --out tests/golden/<model-slug>/reference.jsonl [--top-logprobs 20] [--device cpu|cuda]` — BF16 `AutoModelForCausalLM`, same template and `chat_template_kwargs`, greedy for exactly `max_tokens` positions, top-N log-softmax of BF16 logits upcast to FP32, `engine = transformers-<ver>-bf16-<device>`, temp file + `os.replace`, exit 1 on failure
- `tests/golden/prompts.jsonl`: 16 lines `{"id":"p01".."p16","kind","prompt"|"messages","max_tokens":32,"chat_template_kwargs":{"date_string":"26 Jul 2024"}}` covering English prose, code, CJK, emoji, one ≈2,000-token prompt (p09), chat with and without a system message
- `tolerance.json`:
  ```
  {"min_identical_prefix":32,"min_prompts_passing":14,"top_k":5,"max_abs_logprob_diff":0.15,"margin_nats":0.5}
  ```
- `golden.rs` does not depend on `turbine-bench` (contract §1.2 DAG); it applies the same tolerance rule to `reference.jsonl` records parsed with `serde_json` in one helper function inside the test file
  Covers: S-11 (reference generator, committed prompts and tolerance; the ignored tests `golden hf_reference_matches_cpu` and `golden logits_match_reference` are written here and accepted by their lab run in Task 21)
  Depends on: Tasks 13, 14, 18

- [ ] Write the ignored test `turbine-model --test golden hf_reference_matches_cpu`: writes the tiny checkpoint, runs `uv run scripts/golden/hf_reference.py --model-dir <tiny> --prompts tests/golden/prompts.jsonl --out <tmp>` (asserting exit 0), then runs every prompt on the CPU executor and asserts the tolerance and that Turbine's prompt token ids equal `prompt_token_ids`. Run: `cargo test -p turbine-model --test golden hf_reference_matches_cpu -- --ignored` with `TURBINE_TEST_BACKEND=hip` on a host with `uv` — expect FAIL before the script exists
- [ ] Write the ignored test `turbine-model --test golden logits_match_reference`: `require_backend("hip")`, `require_env_dir("TURBINE_TEST_MODEL_DIR")`, loads Llama-3.2-3B on the HIP provider, replays every committed prompt in-process and checks `tests/golden/llama-3.2-3b-instruct/tolerance.json` against `reference.jsonl`.
- [ ] Implement the script, commit `prompts.jsonl` and `tolerance.json`; `reference.jsonl` is produced in Task 21.
- [ ] Run: `uv run scripts/golden/hf_reference.py --model-dir /tmp/tiny --prompts tests/golden/prompts.jsonl --out /tmp/ref.jsonl --top-logprobs 5` after `cargo test -p turbine-model --test tiny_model` wrote `/tmp/tiny` — expect PASS (exit 0, 16 lines)
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(golden): HF reference generator, prompts, tolerance and golden tests`

## Task 20: `libturbine_hip.so` — CMake build and ABI v1 ops

Files: `kernels/rocm/CMakeLists.txt` (HIP+CXX project, `TURBINE_ROCM_PATH`, `GPU_TARGETS=gfx1201`, `find_package(hip)`, `find_package(hipblaslt)`, CK via `FetchContent` pinned, generator run, shared library with `-fvisibility=hidden`), `kernels/rocm/src/context.cpp` (context, stream, hipBLASLt handle, 32 MiB workspace, last error, identity functions), `kernels/rocm/src/memory.cpp` (malloc/free/copies/sync/mem_info), `kernels/rocm/src/gemm.cpp` (hipBLASLt), `kernels/rocm/src/attention.cpp` (CK `fmha_fwd`), `kernels/rocm/src/rmsnorm.cpp` (CK `rmsnorm2d_fwd` hand-instantiated BF16 bucket + fallback), `kernels/rocm/src/elementwise.hip` (Turbine kernels: embedding, rope, silu_mul, add, rmsnorm fallback), `kernels/rocm/third_party/LICENSES/composable_kernel-LICENSE`, `kernels/rocm/src/turbine_hip.hpp` (the `turbine_ctx` struct and error helpers shared by the sources), `kernels/rocm/cmake/fetch_ck.cmake` (the FetchContent download step)
Interfaces:

- Implements every ABI v1 symbol of `kernels/include/turbine_kernels.h`; `turbine_backend_name()` = `"hip"`, `turbine_build_archs()` = `"gfx1201"`; `turbine_ctx_create` checks `hipGetDeviceProperties(...).gcnArchName` starts with `gfx1201` (else −2 naming it), creates a `hipStreamNonBlocking` stream, `hipblasLtCreate`, logs `ROCM_VERSION_MAJOR/MINOR/PATCH`, `HIPBLASLT_VERSION_MAJOR/MINOR/PATCH` and `TURBINE_CK_COMMIT` to stderr once; error messages start with `hipGetErrorName` (e.g. `hipErrorOutOfMemory: …`); `turbine_last_error(NULL, …)` returns the last failed `ctx_create` on the calling thread
- GEMM: `hipblasLtMatmul` with `HIPBLAS_COMPUTE_32F`, scale `HIP_R_32F`, `opA = HIPBLAS_OP_T` on the weight (layout `k×n`, ld `ldb`), `opB = HIPBLAS_OP_N` on the activation (`k×m`, ld `lda`), C/D layout `n×m` ld `ldc`, types `HIP_R_16BF` in and `HIP_R_16BF`/`HIP_R_32F` out, algorithm from `hipblasLtMatmulAlgoGetHeuristic` with `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`; `_impl` = `hipblaslt`
- Attention: `fmha_fwd(fmha_fwd_traits{hdim 128, "bf16", is_group_mode true, is_v_rowmajor true, mask_bottom_right, no bias, no lse, no dropout}, fmha_fwd_args{batch 1, seqstart_q {0,q_len}, seqstart_k {0,q_start+q_len}, stride = heads·128, nhead_stride = 128, window_size_left −1, window_size_right 0, scale_s}, ck_tile::stream_config{stream})`; a negative return → −2; `_supported` = head_dim 128, BF16, `Hq % Hkv == 0`; `_impl` = `ck_tile_fmha_fwd`
- RMSNorm: `rmsnorm2d_fwd` BF16→BF16 for the instantiated `n` bucket (`_impl` = `ck_tile_rmsnorm2d`), other dims the Turbine kernel (`_impl` = `turbine_hip`); embedding/rope/silu_mul/add follow the `cpu-reference` numerics of Task 5 (`_impl` = `turbine_hip`)
- CK pin: repository `https://github.com/ROCm/rocm-libraries.git`, commit `cd9574023093742434e8c992d13b89ab9a6c1cf8` (tag `therock-7.14.1`, identical to the installed `/opt/rocm/rocm/include/ck_tile`), fetched by `FetchContent` with a `DOWNLOAD_COMMAND` running `cmake/fetch_ck.cmake` (shallow, blob-less, sparse checkout of `projects/composablekernel` at exactly that commit — the monorepo is ~7.8 GB, so a plain `GIT_REPOSITORY`/`GIT_TAG` clone is not used), `SOURCE_SUBDIR` pointing at a directory without a `CMakeLists.txt` so CK is fetched, not built; `-DCMAKE_HIP_COMPILER=…/hipcc` is accepted and mapped to the ROCm clang it wraps (CMake's HIP language rejects the wrapper), with `--rocm-path=${TURBINE_ROCM_PATH}`; inside a container whose `/opt/rocm/rocm/*` entries are `/etc/alternatives` symlinks (novanas), set `TURBINE_ROCM_PATH=/opt/rocm/rocm/core-7.14`; generator command:

  ```
  python3 ${CK}/projects/composablekernel/example/ck_tile/01_fmha/generate.py --targets gfx1201 --api fwd --filter "*d128_bf16*nlogits_nbias*nlse_ndropout_nskip_nqscale_ntrload_nsink*" --output_dir ${CMAKE_BINARY_DIR}/fmha
  ```

  with CK's flags `-DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5 -Wno-undefined-func-template -Wno-float-equal`
  Covers: S-7 (HIP shim, hipBLASLt GEMM, CK FMHA, CK rmsnorm2d where available, Turbine kernels, CMake + runtime load); build verified in Task 21
  Depends on: Task 6

- [ ] Write failing lab check: `scripts/lab-test.sh novanas` — expect FAIL with `cannot load /home/piwi/turbine-ci/target/kernels/libturbine_hip.so` from `hip_ops` (library not built yet)
- [ ] Implement the CMake project and sources as listed; `turbine_<op>_supported` never dereferences descriptor pointers; the context owns stream, handle, workspace and the small `seqstart` scratch, all destroyed in `turbine_ctx_destroy`.
- [ ] Run: `cmake -S kernels/rocm -B /home/piwi/turbine-ci/target/kernels -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 && cmake --build /home/piwi/turbine-ci/target/kernels` inside the lab Job (Task 21) — expect PASS with `libturbine_hip.so` listed by `nm -D --defined-only | grep ' T turbine_'` and no non-`turbine_` exports
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels): libturbine_hip.so with hipBLASLt, CK FMHA and Turbine HIP kernels`

## Task 21: Lab scripts, weights, GPU op tests and golden reference on novanas

Files: `scripts/lab/novanas-test-job.yaml` (build step, env, model mount), `scripts/lab-serve.sh` (new), `scripts/lab/novanas-serve-job.yaml` (new), `scripts/lab/phase1-novanas.yaml` (new), `crates/turbine-kernels/tests/hip_ops.rs` (ignored GPU op comparisons), `tests/golden/llama-3.2-3b-instruct/reference.jsonl` (generated, committed)
Interfaces:

- `novanas-test-job.yaml` setup: `apt-get install -y cmake`, build `kernels/rocm` into `/home/piwi/turbine-ci/target/kernels`, install `uv` with its cache at `/home/piwi/turbine-ci/uv-cache`; env `TURBINE_KERNEL_LIBRARY=/home/piwi/turbine-ci/target/kernels/libturbine_hip.so`, `TURBINE_TEST_BACKEND=hip`, `TURBINE_TEST_MODEL_DIR=/models/llama-3.2-3b-instruct`; read-only hostPath `/home/piwi/turbine-models` → `/models`
- `scripts/lab-serve.sh novanas <config.yaml>` | `novanas --stop`: builds release `turbine-server` and the kernel library, applies `novanas-serve-job.yaml` (namespace `turbine-ci`, `amd.com/gpu: 1`, `hostNetwork: true`, same mounts), waits for `/ready` 200 at `http://192.168.10.203:18000` streaming the log, exits non-zero naming the failed step; `--stop` deletes only that Job
- `scripts/lab/phase1-novanas.yaml`:
  ```
  server:
    listen: 0.0.0.0:18000
  model:
    path: /models/llama-3.2-3b-instruct
    served_name: meta-llama/Llama-3.2-3B-Instruct
  execution:
    backend: hip
  ```
- `hip_ops.rs` tests `gemm_matches_cpu` (m ∈ {1,17}; (n,k) ∈ {(3072,3072),(1024,3072),(8192,3072),(3072,8192)} BF16 out, (128256,3072) F32 out), `attention_matches_cpu` (GQA 24/8, sequence lengths 1, 17, 512, 4096 for prefill and decode, CPU checked on row blocks), `norm_rope_silu_embedding_add_match_cpu`; tolerance |Δ| ≤ 1e-2 for BF16 (one BF16 ulp where the reference magnitude exceeds 2) and ≤ 1e-4 for F32; each prints the `_impl` name
  Covers: S-7/S-13 AC `scripts/lab-test.sh novanas` with `hip_ops`; S-8/S-12/S-13 `tiny_model hip_matches_cpu`; S-11 `golden hf_reference_matches_cpu`; S-11/S-13 `golden logits_match_reference`; S-13 (lab execution, weights)
  Depends on: Tasks 13, 19, 20

- [ ] ASK THE USER FIRST for a Hugging Face token and confirmation that `/home/piwi/turbine-models` may be created; then download once over SSH without writing the token anywhere: `ssh piwi@192.168.10.203 'HF_TOKEN=<pasted> uvx --from huggingface_hub hf download meta-llama/Llama-3.2-3B-Instruct --local-dir /home/piwi/turbine-models/llama-3.2-3b-instruct'` and compare `sha256sum` of `tokenizer.json`, `tokenizer_config.json` with the committed fixtures.
- [ ] Write failing test `turbine-kernels --test hip_ops` (three ignored tests as listed). Run: `scripts/lab-test.sh novanas` — expect FAIL until the Job builds the library
- [ ] Implement the Job, `lab-serve.sh`, serve Job and lab config; verify with `bash -n scripts/lab-serve.sh` and `shellcheck scripts/lab-serve.sh`.
- [ ] ASK THE USER FIRST that one R9700 is free, then generate the reference on novanas CPU: `ssh piwi@192.168.10.203 'cd /home/piwi/turbine-ci/src && uv run scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct --prompts tests/golden/prompts.jsonl --out tests/golden/llama-3.2-3b-instruct/reference.jsonl --top-logprobs 20 --device cpu'` and copy the file back.
- [ ] Run: `scripts/lab-test.sh novanas` — expect PASS with log lines `libturbine_hip.so` built for `gfx1201`, `hipBLASLt 1.4.1`, `CK cd9574023093742434e8c992d13b89ab9a6c1cf8`, `test gemm_matches_cpu ... ok`, `test attention_matches_cpu ... ok`, `test hip_matches_cpu ... ok`, `test hf_reference_matches_cpu ... ok`, `test logits_match_reference ... ok`, Job exit 0
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(lab): novanas GPU op tests, serve script and Llama-3.2 golden reference`

## Task 22: Manual golden and baseline runs, AGENTS.md commands

Files: `AGENTS.md` (Commands section: `scripts/lab-serve.sh`, `turbine-golden`, `uv run scripts/golden/hf_reference.py`, kernel build command), `examples/turbine.yaml` (add `execution` section with defaults)
Interfaces:

- Consumes `scripts/lab-serve.sh novanas scripts/lab/phase1-novanas.yaml`, `turbine-golden compare`, `turbine-bench` (Phase 0)
  Covers: S-1 AC (`cargo build --workspace`, `cargo test --workspace`, clippy and fmt on macOS arm64 with no ROCm and no weights); S-11/S-13 AC manual `turbine-golden compare --url http://192.168.10.203:18000 …`; S-14 AC manual `turbine-bench … --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json`; S-14 AC manual real-stream check `turbine-bench … --concurrency 1 --requests 10 --output json` (moved from phase-0 S-6, decision 2026-09-25)
  Depends on: Tasks 17, 18, 21

- [ ] Run on the macOS workstation: `cargo build --workspace && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check && ! cargo tree --workspace | grep -Ei 'hip|rocm|cuda'` — expect PASS with no ROCm installed and no `TURBINE_TEST_MODEL_DIR`.
- [ ] ASK THE USER FIRST that an R9700 is free; run `scripts/lab-serve.sh novanas scripts/lab/phase1-novanas.yaml` — expect log line `listening` then `/ready` 200 reported by the script.
- [ ] Run: `cargo run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` — expect exit 0 (≥ 14/16 prompts passing, max |Δ logprob| ≤ 0.15); paste the output into task evidence.
- [ ] Run: `cargo run --release -p turbine-bench --bin turbine-bench -- --url http://192.168.10.203:18000 --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json` — expect exit 0 with `"requests_ok": 10`; paste the JSON as the Phase 1 single-request baseline.
- [ ] Run (real-stream check moved from phase-0, decision 2026-09-25): `cargo run --release -p turbine-bench --bin turbine-bench -- --url http://192.168.10.203:18000 --concurrency 1 --requests 10 --output json` — expect exit 0 with `"requests_ok": 10`, `ttft_ms.p50` > 0 and `output_token_throughput` > 0 (natural EOS; concurrency 1 because a second concurrent request gets 429); paste the JSON into task evidence.
- [ ] Run: `scripts/lab-serve.sh novanas --stop` — expect the Job deleted and nothing else touched.
- [ ] Implement the AGENTS.md Commands update and `examples/turbine.yaml` `execution` block; every documented command runs as written.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `docs: phase 1 commands and lab evidence`
