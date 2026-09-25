# phase-7-advanced-distribution — implementation plan

Status: draft
Spec: .procoder/specs/phase-7-advanced-distribution.md

## Goal

Add pipeline parallelism, expert parallelism and prefill/decode disaggregation on top of the Phase 6 cluster, move KV between workers in the self-describing TKV1 format over TCP or a runtime-loaded libibverbs RC queue-pair transport, and open mixed-vendor serving for PD (and PP) behind a capability validator — each proven by deterministic simulations on macOS and by lab runs on the Sparks and novanas.

## Architecture

`turbine-kv::tkv1` owns the TKV1 wire format (128-byte header, 32-byte segment entries, crc32c, receiver-side conversion table, canonical `[K|V]` pack/unpack); `turbine-transport` gains `transfer` (segmented pulls with per-segment checksums, region registry with strict deregistration rules, TCP implementation) and `rdma` (the only new `unsafe`: libibverbs.so.1 opened with `libloading`, own RC QP code, one-sided RDMA READ). `turbine-distributed` gains `pipeline` (cost-balanced partition), `expert` (placement map, dispatch/combine with a rank-count-independent order), `capability` (strategy validator over `turbine_device::capability` records) and protocol 2 messages (`Pd*`, `Rdma*`, `StageActivation`); `turbine-scheduler` gains `pd` (request flow state machine), `router` (TS §11 scoring with every term logged) and `pipeline` (micro-batch scheduler) with GPU-free simulations; `turbine-model` gains the PP stage executor and the EP OLMoE executor over the Phase 5 collective (`all_to_all_v` from `ncclSend`/`ncclRecv`); `turbine-server` wires the PD protocol, PP stages, diagnostics and metrics; `turbine-bench` and `scripts/lab-pd.sh` carry the benchmarks and lab runs.

## Constraints

From the spec (verbatim):

- Rust only in the serving path; no Python (TS §21 rule 4). RDMA verbs (_libibverbs.so.1_) and collective libraries (RCCL/NCCL) are loaded at runtime with `libloading`, exactly as Phase 0 loads NVML and amd-smi and Phase 1 loads the kernel libraries, so the workspace still builds and every non-ignored test passes on macOS arm64 with no GPU and no RDMA libraries.
- `unsafe` and FFI added by this phase live only in the `rdma` module of `turbine-transport` (verbs) and in the existing collective FFI module from Phase 5; the phase-1 `unsafe_isolation` test's allowlist gains exactly that module; every `unsafe` block carries a `// SAFETY:` comment stating the lifetime rule for registered memory regions: a region is deregistered only after every posted work request referencing it has completed or the queue pair is in the error state (TS §21 rule 10).
- Every queue and buffer is bounded (TS §21 rule 8): in-flight micro-batches per pipeline (_distributed.pipeline.micro_batches_), in-flight transfers per worker (_distributed.pd.max_inflight_transfers_), EP dispatch buffers (sized at load from _distributed.expert.max_tokens_per_rank_), RDMA send/receive queue depth (_distributed.transport.rdma.queue_depth_).
- Correctness before optimization (TS §21 rule 1): PP must be token-identical to the colocated single-device run under greedy decoding with the same batch composition (it changes no arithmetic); PD in simulation must be token-identical, and on hardware within the phase-1 golden tolerance (≥ 14 of 16 prompts with the first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats; prefill and decode run in different batch compositions, and in M2 on different vendors' kernels); EP must match within the same tolerance with the combine order fixed by S-2.
- Memory registered for RDMA is exposed only with `REMOTE_READ` access, only to peers listed in the Phase 6 worker registry, only for the lifetime of one transfer, and is deregistered (or its rkey invalidated) on commit, release or timeout (TS §16).
- Models: `meta-llama/Llama-3.2-3B-Instruct` for PP and PD (28 layers, hidden size 3072, 24 query heads, 8 KV heads × head_dim 128, tied embeddings, ~6.4 GB in BF16; attention KV = 28 × 2 × 8 × 128 × 2 B = 114,688 B/token in BF16) and `allenai/OLMoE-1B-7B-0125-Instruct` for EP (16 layers, hidden size 2048, 16 query and 16 KV heads × head_dim 128, 64 routed experts with top-8 routing, no shared expert, 4,096-token context, ~13.8 GB in BF16). Figures come from the published _config.json_ files and are re-checked against the downloaded copies before the lab runs. Weights live in `/home/piwi/turbine-models/<slug>` on each host, downloaded by Claude with the user's HF token at that time (never stored in the repo); tests read `TURBINE_TEST_MODEL_DIR` and never download.
- Lab: `dgx-spark` (192.168.10.246) and `dgx-spark2` (192.168.10.245), 1× GB10 each, ~121 GB unified memory each, ConnectX-7 RoCE at 200 Gb/s on 192.168.47.0/24 (_rocep1s0f0_) and 192.168.48.0/24 (_roceP2p1s0f0_), rdma-core user tools (`ibv_devinfo`, `ib_write_bw`) installed, `nvidia_peermem` not loaded; Turbine runs there in `docker run` containers. `novanas` (192.168.10.203): 2× Radeon AI PRO R9700 (`gfx1201`, 32 GB each), ROCm 7.14.1 at `/opt/rocm/rocm`, 10 GbE only, no RDMA, no Docker — Turbine runs as k3s Jobs requesting `amd.com/gpu` with host networking. Every lab run uses a memory budget (_server.memory_budget_) within the host's free memory measured at the start of that run and a port that is not 8000/8890/8891.
- **Host workloads:** any lab run that needs production workloads (e.g. production vLLM on the Sparks, anything using the R9700 cards) moved or memory freed on any host is started only after the implementer has asked the user and the user has moved the workloads; the implementer never stops, moves or reconfigures production workloads itself. The runs below are sized to fit beside production (largest: PD `prefill-heavy`, ≈ 6.4 GB weights + 8 × ~10k tokens × 112 KiB ≈ 9 GiB KV per Spark).

From the interface contract (`.procoder/contract/interfaces.md`, binding; its §23 picks override the spec text above):

- C-12: the memory budget key is `reliability.memory.device_budget_bytes` (there is no `server.memory_budget`). C-15: golden fixtures are `tests/golden/llama-3.2-3b-instruct/` and `tests/golden/olmoe-1b-7b-0125-instruct/` (not the `meta-llama--…`/`allenai--…` slugs the spec's commands name). C-20: Turbine lab port 18100; `scripts/lab-pd.sh` exits 1 `precondition` when anything already holds 18100. C-21: the RDMA/TCP data port in lab commands is 18111 (`--peer 192.168.47.245:18111`). C-9: the network transfer counter is `turbine_kv_network_transfer_bytes_total{transport,direction}`. C-18: `GET /turbine/v1/kv` `transfers` carries P4's `inflight_bytes`, `max_inflight_bytes` plus P7's `inflight[]`, `completed_total`, `failed_total_by_reason`. C-19/§1.3: unsafe allowlist adds `crates/turbine-transport/src/rdma`; `turbine-transport` switches its manifest lint to `unsafe_code = "deny"` with `#[allow(unsafe_code)]` on `mod rdma` only. C-23: the pool layout is identical on HIP and CUDA; TKV1 pack/unpack stays generic and its round-trip test uses two synthetic layouts.
- §15.3: this phase raises `PROTOCOL_MAX` to 2 and changes the P6 `message_roundtrip_and_versioning` "unsupported" protocol from 2 to 3. §15.1: `Collective::all_to_all_v` built from `ncclSend`/`ncclRecv` inside `ncclGroupStart`/`ncclGroupEnd`; the P7 NCCL-API stub test asserts 15 symbols.
- Toolchain: edition 2024, `rust-version = "1.97"`; public enums later phases extend are `#[non_exhaustive]`; config structs `#[serde(deny_unknown_fields, default)]`; one `thiserror` error enum per crate; every time-dependent component takes `Arc<dyn Clock>`; metric labels only from closed enums via `as_str()`.
- Tests: unit tests in `#[cfg(test)] mod tests`; integration tests `crates/<crate>/tests/<binary>.rs`; GPU/RDMA/lab tests `#[ignore]`, run through `scripts/lab-test.sh <host>` or the lab script named in the step, and every ignored GPU test starts with `turbine_kernels::test_support::require_backend(..)`.
- Gate after every task: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`.
- Lab rules (§21.1): no `docker run` outside `scripts/lab-pd.sh` / the defined lab scripts; scripts never stop/restart/reconfigure non-`turbine-lab-*` workloads; Spark correctness runs proceed after the MemAvailable precondition, benchmark/soak/overload runs always ask the user first.

## Task 1: advanced-distribution configuration keys

Files: `crates/turbine-core/src/config/distributed.rs` (P7 sub-structs and fields, static validation), `crates/turbine-core/src/config/mod.rs` (unit test in `mod tests`; `allow_lossy` WARN text helper).
Interfaces:

- produces `pub enum WorkerRole { Prefill, Decode, Both }`; `DistributedConfig.role: WorkerRole`
- produces `pub struct PipelineConfig { pub stages: u32, pub layer_split: Option<Vec<u32>>, pub micro_batches: Option<u32> }`
- produces `pub enum ExpertPlacementSource { Contiguous, File(PathBuf) }` (YAML: `contiguous` or a path), `pub struct ExpertConfig { pub parallel_size: u32, pub placement: ExpertPlacementSource, pub max_tokens_per_rank: u32 }`
- produces `pub enum PdFallback { Colocate, Reject }`, `pub struct PdConfig { pub enabled: bool, pub transfer_timeout: HumanDuration, pub max_inflight_transfers: u32, pub fallback: PdFallback, pub min_prompt_tokens: u32 }`
- produces `pub enum GidIndex { Auto, Index(u32) }`, `pub struct RdmaConfig { pub enabled: bool, pub library: Option<PathBuf>, pub device: Option<String>, pub gid_index: GidIndex, pub queue_depth: u32, pub bounce_bytes: ByteSize }`; `TransportConfig.rdma: RdmaConfig`
- produces `pub struct KvConversionConfig { pub allow_lossy: bool }`; fields `DistributedConfig.{pipeline, expert, pd, kv_conversion}`
- produces `impl DistributedConfig { pub fn lossy_warnings(&self) -> Vec<String> }`
  Covers: S-1/S-2/S-3/S-5 config keys (static rules); unit test `config::tests::advanced_distribution_rejections`.
  Depends on: phase-6 Task 1.

- [ ] Write failing test `config::tests::advanced_distribution_rejections`: starting from a valid phase-6 distributed YAML, assert `validate()` errors whose `key()` is `distributed.pd.enabled` (`pd.enabled: true` with `distributed.enabled: false`), `distributed.pd.transfer_timeout` (`50ms`), `distributed.pd.max_inflight_transfers` (0 and 257), `distributed.pipeline.stages` (0), `distributed.pipeline.micro_batches` (0; 9 with `stages: 2`), `distributed.pipeline.layer_split` (length 3 with `stages: 2`; an entry 0), `distributed.expert.parallel_size` (0), `distributed.transport.rdma.device` (`rdma.enabled: true` without device), `distributed.transport.rdma.queue_depth` (8), `distributed.transport.rdma.bounce_bytes` (`8MiB`); and that `kv_conversion.allow_lossy: true` yields one `lossy_warnings()` line per lossy pair. Run: `cargo test -p turbine-core config::tests::advanced_distribution_rejections` — expect FAIL.
- [ ] Implement the keys with the spec defaults (`both`, 1 stage, null split, micro-batches = stages, EP 1 / contiguous / 8192, PD off / 10s / 8 / colocate / 0, RDMA off / null / auto / 256 / 256MiB, lossy off); checks that need the model (split sum = layers, stages ≤ layers, parallel size divides the expert count, `max_tokens_per_rank ≥ max_batch_tokens / parallel_size`, placement file contents) run at load in Tasks 6–7 and exit 2 naming the same keys.
- [ ] Run: `cargo test -p turbine-core config::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(core): pipeline, expert, PD, RDMA and KV conversion configuration keys`

## Task 2: TKV1 encode/decode and rejection rules

Files: `crates/turbine-kv/src/tkv1.rs` (new: header, segment table, encode/decode, verification, KV identity fingerprint, tests), `crates/turbine-kv/src/lib.rs` (`pub mod tkv1`), `crates/turbine-kv/Cargo.toml` (`sha2` for the TKV1 `model_fp`).
Interfaces:

- produces `pub const TKV1_MAGIC: [u8; 4] = *b"TKV1"; pub const TKV1_VERSION: u16 = 1; pub const HEADER_BYTES: usize = 128; pub const SEGMENT_BYTES: usize = 32`
- produces `pub struct Tkv1Header { flags, request_id: [u8; 16], model_fp: [u8; 32], kv_format: KvDtype, state_format: u16, block_tokens: u32, token_start: u64, token_count: u64, first_token: u32, attn_layers: u16, state_layers: u16, segment_count: u32 }` (contract), `pub const FLAG_HAS_SAMPLER_STATE: u16 = 1 << 1`
- produces `pub enum SegmentKind { AttnKvBlock = 1, RecurrentState = 2, ConvState = 3, SamplerState = 4 }`, `pub struct SegmentEntry { pub kind: SegmentKind, pub layer: u16, pub block_index: u32, pub offset: u64, pub length: u64, pub crc32c: u32 }`
- produces `pub enum Tkv1Error { BadMagic, UnsupportedVersion(u16), HeaderCrc, SegmentCrc { index: u32 }, ModelFingerprintMismatch, SegmentOutOfRange { index: u32 }, ReservedSegmentKind(u16), FormatMismatch { from: KvDtype, to: KvDtype } }` + `pub fn failure_reason(&self) -> &'static str` (`checksum` for the two CRC variants, else `format_mismatch`)
- produces `pub fn encode(h: &Tkv1Header, segs: &[SegmentEntry]) -> Vec<u8>` (header + table; header CRC over the 128-byte fixed part with bytes 92..96 zeroed)
- produces `pub struct Tkv1Expect { pub model_fp: [u8; 32], pub region_len: u64 }`, `pub fn decode(bytes: &[u8], expect: &Tkv1Expect) -> Result<(Tkv1Header, Vec<SegmentEntry>), Tkv1Error>`, `pub fn verify_segment(index: u32, entry: &SegmentEntry, data: &[u8]) -> Result<(), Tkv1Error>`
- produces `pub fn kv_identity_fp(id: &ModelIdentity, fmt: &KvFormat) -> [u8; 32]` (SHA-256 of config hash ‖ weights-index hash ‖ kv_format code ‖ block_tokens ‖ layers/kv_heads/head_dim, LE)
- fixed layout (LE): magic 0..4, version 4..6, flags 6..8, request_id 8..24, model_fp 24..56, kv_format 56..58, state_format 58..60, block_tokens 60..64, token_start 64..72, token_count 72..80, first_token 80..84, attn_layers 84..86, state_layers 86..88, segment_count 88..92, header_crc32c 92..96, zero 96..128; segment entry: kind 0..2, layer 2..4, block_index 4..8, offset 8..16, length 16..24, crc32c 24..28, zero 28..32
  Covers: S-4 (`tkv1::tests::roundtrip_and_reject`).
  Depends on: phase-4 (`ModelIdentity`, `KvFormat`), phase-6 (`crc32c`).

- [ ] Write failing test `tkv1::tests::roundtrip_and_reject`: a Llama-shaped transfer (28 attention layers, `block_tokens` 16, `token_count` 1,000 so the final block holds 8 valid rows plus zero padding, sampler-state segment, random payload bytes in one region) encodes and decodes to an identical header and table and every segment verifies; then distinct errors for wrong magic (`BadMagic`), version 2 (`UnsupportedVersion(2)`), a flipped header byte (`HeaderCrc`), a flipped payload byte (`SegmentCrc { index }`), another `model_fp` (`ModelFingerprintMismatch`), a segment ending past `region_len` (`SegmentOutOfRange`), segment kind 2 (`ReservedSegmentKind(2)`) and `state_layers: 1` (`ReservedSegmentKind(2)`, reason `format_mismatch`). Run: `cargo test -p turbine-kv tkv1::tests::roundtrip_and_reject` — expect FAIL.
- [ ] Implement with explicit `to_le_bytes`/`from_le_bytes` at the fixed offsets (no serde), checks in order magic → version → header CRC → `model_fp` → `state_layers`/reserved kinds → segment bounds (`offset + length ≤ region_len`, checked with `checked_add`) → per-segment CRC on arrival (`verify_segment`); decode never allocates more than `segment_count × 32` bytes after bounding `segment_count` by `(bytes.len() − 128) / 32`.
- [ ] Run: `cargo test -p turbine-kv tkv1::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(kv): TKV1 wire format with header and per-segment crc32c`

## Task 3: receiver conversion table and canonical block layout

Files: `crates/turbine-kv/src/tkv1.rs` (conversion, pack/unpack, FP8 upcast, tests).
Interfaces:

- produces `pub enum ConversionKind { Copy, ExactUpcast, Lossy }` + `as_str` (`copy`, `exact_upcast`, `lossy`)
- produces `pub fn conversion(from: KvDtype, to: KvDtype, allow_lossy: bool) -> Result<ConversionKind, Tkv1Error>` (receiver only; identical → `Copy`; `fp8_*` → `bf16` → `ExactUpcast` only when every scale of the transfer is a power of two — user decision 2026-09-25; `bf16 → fp8_*`, `fp16 ↔ bf16` → `Lossy`, `Err(FormatMismatch)` unless `allow_lossy`)
- produces `pub enum PoolOrder { KvTokenHeadDim, KvHeadTokenDim }`, `pub struct PoolLayoutDesc { pub block_tokens: u32, pub kv_heads: u32, pub head_dim: u32, pub elem_bytes: u32, pub order: PoolOrder }`
- produces `pub fn pack_block(src: &[u8], layout: &PoolLayoutDesc, dst: &mut [u8])` (pool → canonical `[K | V]`, each `[block_tokens, kv_heads, head_dim]` row-major), `pub fn unpack_block(src: &[u8], layout: &PoolLayoutDesc, dst: &mut [u8])` (inverse; a byte permutation, never a dtype change)
- produces `pub fn upcast_fp8_e4m3_to_bf16(src: &[u8], scales: &[f32], values_per_scale: usize, dst: &mut [u8])`
  Covers: S-4/S-6 (`tkv1::tests::conversion_table`, `tkv1::tests::canonical_layout_roundtrip`).
  Depends on: Task 2.

- [ ] Write failing test `tkv1::tests::conversion_table`: `bf16 → bf16` is `Copy`; `fp8_e4m3_per_tensor_scale → bf16` is `ExactUpcast` and every one of the 256 e4m3 codes × scale 0.5 decodes to the bf16 bit pattern of its exact value; `bf16 → fp8_e4m3_per_tensor_scale`, `bf16 → fp8_e4m3_per_block_scale`, `fp16 → bf16` and `bf16 → fp16` are `Err(FormatMismatch)` with `allow_lossy: false` and `Lossy` with `true`; the sender-side API (`pack_block`) has no dtype parameter, so a packed bf16 block is byte-equal to its source values. Run: `cargo test -p turbine-kv tkv1::tests::conversion_table` — expect FAIL.
- [ ] Write failing test `tkv1::tests::canonical_layout_roundtrip`: 1,000 random BF16 blocks (`block_tokens` 16, 8 KV heads, head_dim 128, seeded `rand_chacha`) packed from a `KvTokenHeadDim` pool layout and unpacked into a `KvHeadTokenDim` layout give, for every (K/V, token, head, dim), the identical 16-bit value, and packed length = 2 × 16 × 8 × 128 × 2 bytes. Run: `cargo test -p turbine-kv tkv1::tests::canonical_layout_roundtrip` — expect FAIL.
- [ ] Implement the table as a `match` on `(from, to)`, the e4m3 decode (bias 7, subnormals, NaN = S.1111.111) into f32 then `half::bf16::from_f32` (exact for power-of-two scales), and pack/unpack as index-mapped `copy_from_slice` of `elem_bytes` chunks.
- [ ] Run: `cargo test -p turbine-kv tkv1::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(kv): TKV1 conversion table and canonical block pack/unpack`

## Task 4: directory entries with KV format and in-flight transfers

Files: `crates/turbine-kv/src/directory/cluster.rs` (`kv_format`, `transfer` fields; transfer tracking), `crates/turbine-kv/src/directory/tests.rs` (test).
Interfaces:

- produces `DirectoryEntry.kv_format: Option<KvDtype>`, `DirectoryEntry.transfer: Option<InflightTransfer>`, `pub struct InflightTransfer { pub source: NodeId, pub destination: NodeId, pub bytes: u64, pub deadline_ms: u64 }` (contract)
- produces `impl ClusterIndex { pub fn begin_transfer(&mut self, key: &KvKey, t: InflightTransfer) -> bool; pub fn commit_transfer(&mut self, key: &KvKey, destination: &NodeId, tier: TierId); pub fn abort_transfer(&mut self, key: &KvKey, destination: &NodeId); pub fn transfer_sources(&self, key: &KvKey, want: KvDtype, allow_lossy: bool) -> Vec<&DirectoryEntry> }` (in-flight copies are never sources; a different dtype is a miss unless `conversion` says `Copy`/`ExactUpcast`)
  Covers: S-7 (`directory::tests::inflight_transfer_visible`).
  Depends on: Task 3, phase-6 Task 4.

- [ ] Write failing test `directory::tests::inflight_transfer_visible`: key K held by `dgx-spark` (bf16); `begin_transfer(K, {source: dgx-spark, destination: dgx-spark2, bytes: 1_835_008, deadline_ms: 5_000})` makes `lookup(K)` report an entry for `dgx-spark2` carrying that `transfer`; `transfer_sources(K, Bf16, false)` returns only `dgx-spark`; a second `begin_transfer` to `dgx-spark2` returns false; `abort_transfer` leaves only the source entry; after `begin` + `commit_transfer` both are sources with `transfer: None`; an fp8 copy on a third node is a source for `Bf16` only via `ExactUpcast` and never for `Fp16`. Run: `cargo test -p turbine-kv directory::tests::inflight_transfer_visible` — expect FAIL.
- [ ] Implement transfers as a per-(key, destination) marker entry excluded from `transfer_sources` and from phase-6 `L3ClusterTier::plan`, removed on abort or deadline, turned into a normal entry on commit.
- [ ] Run: `cargo test -p turbine-kv directory::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(kv): directory tracks KV format and in-flight transfers`

## Task 5: capability records and heterogeneous placement validator

Files: `crates/turbine-device/src/capability.rs` (new: `DeviceCapabilities`, `RuntimeProbes`, `probe_capabilities`), `crates/turbine-device/src/inventory.rs` (`DeviceInfo.capabilities`), `crates/turbine-distributed/src/capability.rs` (new: `Strategy`, `validate`, `CapabilityRefusal`, `PdPairing`), `crates/turbine-distributed/src/placement.rs` (test `heterogeneous_refusals`; P7 model placement calls `capability::validate` before any group is formed).
Interfaces:

- produces `pub struct DeviceCapabilities { pub vendor: Vendor, pub arch: Option<String>, pub dtypes: Vec<DType>, pub kv_formats: Vec<KvDtype>, pub execution_backend: Option<ExecutionBackend>, pub collective_backend: Option<CollectiveBackendKind>, pub rdma_devices: Vec<String> }` (contract)
- produces `pub struct RuntimeProbes { pub kernel_loaded: BTreeMap<Vendor, ExecutionBackend>, pub collective_loaded: BTreeMap<Vendor, CollectiveBackendKind>, pub verbs_devices: Vec<String> }`, `pub fn probe_capabilities(inv: &DeviceInventory, probes: &RuntimeProbes) -> Vec<DeviceCapabilities>`
- produces `#[non_exhaustive] pub enum Strategy { Tp, Ep, Pp, Pd }`, `pub struct PdPairing { pub prefill_fp: [u8; 32], pub decode_fp: [u8; 32], pub prefill_block_tokens: u32, pub decode_block_tokens: u32, pub prefill_kv: KvDtype, pub decode_kv: KvDtype, pub allow_lossy: bool }`
- produces `pub struct CapabilityRefusal { pub strategy: Strategy, pub devices: Vec<DeviceId>, pub missing: String }` (`Display`: `<strategy> refused on devices <i,j>: missing <capability>`), `pub fn validate(strategy: Strategy, devices: &[(DeviceId, DeviceCapabilities)], pd: Option<&PdPairing>) -> Result<(), CapabilityRefusal>`, `pub fn placeable(dev: DeviceId, caps: &DeviceCapabilities) -> Result<(), CapabilityRefusal>`
  Covers: S-6 (`placement::tests::heterogeneous_refusals`).
  Depends on: Task 3, phase-0 inventory, phase-6 Task 7.

- [ ] Write failing test `crates/turbine-distributed/src/placement.rs` `placement::tests::heterogeneous_refusals`: synthetic inventory devices 0–1 NVIDIA `sm_121` (CUDA, NCCL, BF16) and 2–3 AMD `gfx1201` (HIP, RCCL, BF16); TP over {1, 2} and EP over {0, 3} are refused with `devices` containing both indices and `missing` naming `collective_backend`; TP over {0, 1} passes; PP over {1, 2} passes when both list BF16 and is refused `missing: dtype bf16` when device 2 lacks it; PD over {2 → 0} passes with equal fingerprints, equal `block_tokens` and bf16 → bf16, and is refused for a different fingerprint, `block_tokens` 16 vs 32, and bf16 → fp8 without `allow_lossy`; a device with `execution_backend: None` fails `placeable` with `missing: kernel library`. Run: `cargo test -p turbine-distributed placement::tests::heterogeneous_refusals` — expect FAIL.
- [ ] Implement the rules: TP/EP require one vendor and one `collective_backend` present on every device; PP requires BF16 (the boundary activation dtype) on both sides of each vendor boundary; PD across vendors requires equal fingerprints, equal `block_tokens` and `tkv1::conversion(prefill_kv, decode_kv, allow_lossy)` not `Lossy` unless allowed; refusals at startup exit 2 before binding; `probe_capabilities` sets `execution_backend` only when that vendor's kernel library loaded and logs each unplaceable device with its reason.
- [ ] Run: `cargo test -p turbine-distributed placement:: && cargo test -p turbine-device capability` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(distributed): device capability records and per-strategy heterogeneous validator`

## Task 6: pipeline partitioner

Files: `crates/turbine-distributed/src/pipeline.rs` (new: layer costs, partition, split validation, tests).
Interfaces:

- produces `pub struct PipelinePlan { pub stages: Vec<StagePlan> }`, `pub struct StagePlan { pub stage: u32, pub device_group: u32, pub first_layer: u32, pub last_layer: u32, pub weight_bytes: u64, pub est_cost: f64 }` (contract), `impl PipelinePlan { pub fn has_embedding(&self, stage: u32) -> bool; pub fn has_lm_head(&self, stage: u32) -> bool; pub fn max_min_ratio(&self) -> f64 }`
- produces `pub fn layer_costs(shape: &ModelShape) -> Vec<f64>` (normalized weight bytes + per-token FLOPs; MoE layers costed by `experts_per_token` active experts)
- produces `pub fn partition(layer_costs: &[f64], stages: u32) -> PipelinePlan` (contract), `pub fn partition_with_ends(layer_costs: &[f64], embed_cost: f64, head_cost: f64, stages: u32) -> PipelinePlan`
- produces `pub fn validate_split(split: &[u32], layers: u32, stages: u32) -> Result<(), PlanError>` (contract; key `distributed.pipeline.layer_split`)
  Covers: S-1 (`pipeline::tests::partition_balances_cost`, `pipeline::tests::explicit_split_validated`).
  Depends on: phase-5 (`PlanError`, `ModelShape`).

- [ ] Write failing test `pipeline::tests::partition_balances_cost`: for the Llama-3.2-3B `ModelShape` (28 uniform layers, tied embeddings) with 2, 4 and 7 stages and the OLMoE shape (16 MoE layers) with 2 and 4 stages, ranges are contiguous, cover every layer exactly once in order, `has_embedding(0)` and `has_lm_head(last)` hold (for tied embeddings both stage 0 and the last stage load the embedding tensor), and `max_min_ratio() ≤ 1.25`. Run: `cargo test -p turbine-distributed pipeline::tests::partition_balances_cost` — expect FAIL.
- [ ] Write failing test `pipeline::tests::explicit_split_validated`: `[14, 14]` for 28 layers and 2 stages is `Ok`; `[14, 13]`, `[28, 0]` and `[10, 9, 9]` with 2 stages are `Err(PlanError { key: "distributed.pipeline.layer_split", .. })`. Run: `cargo test -p turbine-distributed pipeline::tests::explicit_split_validated` — expect FAIL.
- [ ] Implement `partition` as the classic linear-partition dynamic programme minimizing the maximum stage cost (O(stages × layers²), ties broken toward earlier cuts), with embedding cost charged to stage 0 and LM-head cost to the last stage; log the plan once (`pipeline_plan` INFO with every stage) and expose it for `/turbine/v1/scheduler`.
- [ ] Run: `cargo test -p turbine-distributed pipeline::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(distributed): cost-balanced pipeline partitioner and explicit split validation`

## Task 7: expert placement, dispatch and combine

Files: `crates/turbine-distributed/src/expert.rs` (new: placement map, file loader, dispatch plan, combine, tests).
Interfaces:

- produces `pub struct ExpertPlacement { pub parallel_size: u32, pub map: Vec<Vec<u32>> }` (layer → rank per expert; contract), `pub fn contiguous(num_layers: u32, num_experts: u32, parallel_size: u32) -> Result<ExpertPlacement, PlanError>`, `pub fn from_file(path: &Path, num_layers: u32, num_experts: u32, parallel_size: u32) -> Result<ExpertPlacement, PlanError>` (key `distributed.expert.placement`; divisibility errors key `distributed.expert.parallel_size`)
- produces `pub struct TokenRoute { pub token: u32, pub experts: Vec<u32>, pub weights: Vec<f32> }`, `pub struct DispatchPlan { pub send_counts: Vec<usize>, pub per_rank: Vec<Vec<(u32, u32, u32)>> /* (token, expert, top-k slot) */ }`
- produces `pub fn dispatch(routes: &[TokenRoute], layer_map: &[u32], ranks: u32, max_tokens_per_rank: u32) -> Result<DispatchPlan, ExpertError>` (`ExpertError::DispatchOverflow { rank, tokens }` when a rank exceeds its buffer)
- produces `pub fn combine(routes: &[TokenRoute], plan: &DispatchPlan, rank_outputs: &[Vec<f32>], hidden: usize) -> Vec<f32>` (per token, accumulate expert outputs in top-k slot order in f32, independent of rank count)
  Covers: S-2 (`expert::tests::placement_map_valid`, `expert::tests::dispatch_combine_matches_local`).
  Depends on: phase-5 (`PlanError`).

- [ ] Write failing test `expert::tests::placement_map_valid`: contiguous placement of 64 experts × 16 layers over 2 ranks gives experts 0–31 → rank 0 and 32–63 → rank 1 in every layer; a YAML placement file missing expert 17 of layer 3 and one naming rank 2 with `parallel_size: 2` are rejected with key `distributed.expert.placement`; `parallel_size: 3` with 64 experts is rejected with key `distributed.expert.parallel_size`. Run: `cargo test -p turbine-distributed expert::tests::placement_map_valid` — expect FAIL.
- [ ] Write failing test `expert::tests::dispatch_combine_matches_local`: with CPU reference experts (expert e maps x to `tanh(x × (e + 1) / 64)`), hidden 32, 64 experts, top-8 routing and 1,000 seeded random routings (including a batch where every token routes only to rank 0's experts and one where rank 1 receives zero tokens), dispatch → per-rank expert compute → combine is bitwise identical for 1, 2 and 4 ranks, and a rank with zero tokens still gets a `send_counts` entry of 0 (it enters the collective). Run: `cargo test -p turbine-distributed expert::tests::dispatch_combine_matches_local` — expect FAIL.
- [ ] Implement placement validation (every expert of every layer exactly once, every rank ≥ 1 expert), dispatch grouping by owning rank in (token, slot) order, and combine by scattering rank outputs back to `(token, slot)` then summing slots 0..k in order with the router weight.
- [ ] Run: `cargo test -p turbine-distributed expert::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(distributed): expert placement map with rank-count-independent dispatch and combine`

## Task 8: control protocol 2 — PD, RDMA and pipeline messages

Files: `crates/turbine-distributed/src/proto.rs` (`PROTOCOL_MAX = 2`, new variants, versioning test update, new test).
Interfaces:

- produces `pub const PROTOCOL_MAX: u16 = 2` (a node speaking only 1 still interoperates for P6 messages; P7 messages are sent only when the negotiated protocol is 2)
- produces `Message::{PdReserve { request_id, projected_blocks: u32, prompt_tokens: u32, max_tokens: u32 }, PdReserved { request_id, held_blocks: Vec<u32> }, PdRefused { request_id, reason: String, pressure: PressureState }, PdReady { request_id, header: Vec<u8>, regions: Vec<RemoteRegionMsg> }, PdCommitted { request_id }, PdAborted { request_id, reason: TransferFailureMsg }, RdmaConnect { gid: [u8; 16], qpn: u32, psn: u32 }, RdmaConnectAck { gid: [u8; 16], qpn: u32, psn: u32 }, StageActivation { pipeline: u32, micro_batch: u32, step: u64, from_stage: u32, bytes: Vec<u8> }}` (contract §15.3)
- produces `pub struct RemoteRegionMsg { pub addr: u64, pub rkey: u32, pub len: u64 }`, `pub enum TransferFailureMsg { Timeout, Checksum, TransportError, PeerLost, FormatMismatch, ReservationRefused }`
  Covers: S-3/S-5 protocol; tests `proto::tests::message_roundtrip_and_versioning` (updated: unsupported protocol is now 3) and `proto::tests::protocol_two_messages_roundtrip`.
  Depends on: phase-6 Task 5.

- [ ] Write failing test `proto::tests::protocol_two_messages_roundtrip`: every new variant round-trips; `PdReady.regions` longer than 4,096 and `StageActivation.bytes` longer than `max_frame_bytes` are rejected; two nodes with `PROTOCOL_MAX` 2 negotiate protocol 2 and a node pinned to `protocol_max: 1` negotiates 1. Change `message_roundtrip_and_versioning` to send `protocol_min = protocol_max = 3`. Run: `cargo test -p turbine-distributed proto::` — expect FAIL.
- [ ] Implement the variants with the bounded-list deserializer from phase 6 and record the negotiated protocol per link so P7 messages are never sent over a protocol-1 link.
- [ ] Run: `cargo test -p turbine-distributed proto:: auth::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(distributed): protocol 2 with PD, RDMA connect and stage activation messages`

## Task 9: segmented transfer engine over TCP with fault outcomes

Files: `crates/turbine-transport/src/transfer.rs` (new: segment descriptors, failures, region registry, `TransferConn`, pull/serve over TCP frames), `crates/turbine-transport/src/lib.rs` (`pub mod transfer`), `crates/turbine-transport/tests/transfer.rs` (new integration test).
Interfaces:

- produces `pub struct SegmentDesc { pub offset: u64, pub length: u64, pub crc32c: u32 }`, `pub enum TransferFailure { Timeout, Checksum, TransportError, PeerLost, FormatMismatch, ReservationRefused }` + `as_str` (contract label values)
- produces `pub struct RemoteRegion { pub addr: u64, pub rkey: u32, pub len: u64 }` (shared by TCP and RDMA; for TCP `addr` is the registry handle and `rkey` a random token)
- produces `pub struct RegionRegistry` with `pub fn register(&self, bytes: Arc<[u8]>, peer: &str, deadline: Duration) -> RemoteRegion`, `pub fn release(&self, r: &RemoteRegion)`, `pub fn live(&self) -> usize`, `pub fn expire(&self, now: Duration) -> usize` (a region is readable only by the named peer, only until release or deadline)
- produces `pub trait SegmentSink: Send { fn write(&mut self, index: u32, offset: u64, data: &[u8]) -> Result<(), TransferFailure>; }`
- produces `pub enum TransferConn { Tcp(Arc<Connection>), Rdma(rdma::RdmaPeer) }`, `pub async fn pull_segments(conn: &TransferConn, src: &[RemoteRegion], segs: &[SegmentDesc], dst: &mut dyn SegmentSink, deadline: Duration) -> Result<u64, TransferFailure>` (contract), `pub async fn serve_segments(conn: &Connection, registry: &RegionRegistry, peer: &str) -> Result<(), TransportError>`
- TCP payload framing (inside this module, independent of `turbine-distributed::proto`): request frame `[0x01, rkey u32, addr u64, offset u64, length u64]`, data frame `[0x02, index u32, bytes…]`, error frame `[0x03, code u8]`
  Covers: S-5/S-9 (`--test transfer tcp_transfer_protocol_faults`).
  Depends on: phase-6 Tasks 2–3.

- [ ] Write failing test `crates/turbine-transport/tests/transfer.rs` `tcp_transfer_protocol_faults` (mem transport, paused time): a 64-segment region pulled cleanly verifies every crc32c; then (a) `MemFault::Close` injected by the sink after 50 % of the bytes → `Err(TransportError)`, (b) the source region's byte flipped after its CRCs were computed → `Err(Checksum)`, (c) a server that stops answering → `Err(Timeout)` at the deadline; after each fault the source `RegionRegistry::live()` is 0 once the pulling side's abort handler releases it and the test's reservation counter is back to 0. Run: `cargo test -p turbine-transport --test transfer tcp_transfer_protocol_faults` — expect FAIL.
- [ ] Implement pull as pipelined segment requests (at most `queue_depth` outstanding), CRC check per segment before `SegmentSink::write`, one deadline for the whole transfer, `Closed`/`Io` mapped to `TransportError`; `serve_segments` answers only registered, unexpired regions for the named peer and never reads outside `[0, len)`.
- [ ] Run: `cargo test -p turbine-transport --test transfer` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(transport): segmented KV transfer engine over TCP with timeout, checksum and loss outcomes`

## Task 10: runtime loading of libibverbs

Files: `crates/turbine-transport/Cargo.toml` (`libloading`; lint `unsafe_code = "deny"`), `crates/turbine-transport/src/lib.rs` (`#[allow(unsafe_code)] pub mod rdma;`, `TransportKind::Rdma`), `crates/turbine-transport/src/rdma/mod.rs` (new: `open_verbs`, `RdmaError`, selection), `crates/turbine-transport/src/rdma/sys.rs` (new: symbol table), `crates/turbine-kernels/tests/unsafe_isolation.rs` (allowlist gains `crates/turbine-transport/src/rdma`).
Interfaces:

- produces `pub enum RdmaError { Unavailable(String), Library { path: PathBuf, detail: String }, Device(String), WorkCompletion(String), QpError }` (contract)
- produces `pub fn open_verbs(explicit: Option<&Path>) -> Result<VerbsLibrary, RdmaError>` (default search: `libibverbs.so.1` through the loader path; miss → `Unavailable`; explicit miss → `Library { path }`)
- produces `pub struct VerbsLibrary` holding `libloading::Library` plus resolved `Symbol`s for exactly: `ibv_get_device_list`, `ibv_free_device_list`, `ibv_get_device_name`, `ibv_open_device`, `ibv_close_device`, `ibv_query_port`, `ibv_query_gid`, `ibv_alloc_pd`, `ibv_dealloc_pd`, `ibv_reg_mr`, `ibv_dereg_mr`, `ibv_create_cq`, `ibv_destroy_cq`, `ibv_create_qp`, `ibv_destroy_qp`, `ibv_modify_qp`, `ibv_wc_status_str` (all exported by rdma-core `libibverbs.so.1`, checked on dgx-spark; `ibv_post_send`/`ibv_poll_cq` are `static inline` in `verbs.h` and are called through `ibv_context.ops` in Task 11)
- produces `pub enum TransportChoice { Rdma, Tcp { reason: String } }`, `pub fn select_transport(cfg: &RdmaConfig) -> Result<TransportChoice, RdmaError>` (enabled + load/device failure → error, exit 1; disabled → `Tcp` with an INFO reason)
  Covers: S-5 (`rdma::tests::missing_library_behaviour`).
  Depends on: Task 1, phase-6 Task 2, phase-1 (`unsafe_isolation`).

- [ ] Write failing test `rdma::tests::missing_library_behaviour`: on a host without libibverbs (macOS) `open_verbs(None)` is `Err(Unavailable(_))` and `select_transport` with rdma disabled is `Tcp` with a reason; `open_verbs(Some("/nonexistent/libibverbs.so.1"))` is `Err(Library { path })` whose `Display` contains that path; `select_transport` with `enabled: true` and no library is an error, never a panic and never a silent `Tcp`. Run: `cargo test -p turbine-transport rdma::tests::missing_library_behaviour` — expect FAIL.
- [ ] Implement the loader with `unsafe { libloading::Library::new(..) }` and `lib.get::<unsafe extern "C" fn(..)>(b"ibv_…\0")` for the 17 symbols (each `unsafe` with `// SAFETY:` stating the C signature from rdma-core `verbs.h` and that the `Library` outlives every `Symbol`), and update `unsafe_isolation` so the only unsafe outside the prior allowlist is under `crates/turbine-transport/src/rdma`.
- [ ] Run: `cargo test -p turbine-transport rdma:: && cargo test -p turbine-kernels --test unsafe_isolation` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(transport): runtime-loaded libibverbs with explicit-path and fallback rules`

## Task 11: RDMA RC queue pairs and one-sided READ transfers

Files: `crates/turbine-transport/src/rdma/sys.rs` (`#[repr(C)]` mirrors of `ibv_gid`, `ibv_port_attr`, `ibv_context`/`ibv_context_ops` (for `poll_cq`, the 12th function pointer of `ibv_context_ops` (index 11), and `post_send`, the 26th (index 25)), `ibv_qp_init_attr`, `ibv_qp_cap`, `ibv_qp_attr`, `ibv_ah_attr`, `ibv_global_route`, `ibv_sge`, `ibv_send_wr`, `ibv_wc`, `ibv_mr`, `ibv_qp`, `ibv_cq` and the enum constants below), `crates/turbine-transport/src/rdma/device.rs` (device open, port/GID selection), `crates/turbine-transport/src/rdma/qp.rs` (RC QP lifecycle, READ, completion polling), `crates/turbine-transport/src/rdma/mr.rs` (registration with the deregistration rule), `crates/turbine-transport/src/rdma/transport.rs` (`RdmaTransport`, `RdmaPeer`, bounce buffer), `crates/turbine-transport/tests/lab.rs` (ignored lab test), `crates/turbine-transport/src/rdma/tests.rs` (layout test).
Interfaces:

- constants (verified in dgx-spark `/usr/include/infiniband/verbs.h`): `IBV_QPT_RC = 2`, `IBV_QPS_INIT = 1`, `IBV_QPS_RTR = 2`, `IBV_QPS_RTS = 3`, `IBV_QPS_ERR = 6`, `IBV_WR_RDMA_READ = 4`, `IBV_SEND_SIGNALED = 2`, `IBV_ACCESS_LOCAL_WRITE = 1`, `IBV_ACCESS_REMOTE_READ = 4`, `IBV_MTU_1024 = 3`, `IBV_MTU_4096 = 5`, `IBV_PORT_ACTIVE = 4`, `IBV_LINK_LAYER_ETHERNET = 2`, QP attr masks `IBV_QP_STATE 1<<0`, `ACCESS_FLAGS 1<<3`, `PKEY_INDEX 1<<4`, `PORT 1<<5`, `AV 1<<7`, `PATH_MTU 1<<8`, `TIMEOUT 1<<9`, `RETRY_CNT 1<<10`, `RNR_RETRY 1<<11`, `RQ_PSN 1<<12`, `MAX_QP_RD_ATOMIC 1<<13`, `MIN_RNR_TIMER 1<<15`, `SQ_PSN 1<<16`, `MAX_DEST_RD_ATOMIC 1<<17`, `DEST_QPN 1<<20`
- produces `pub struct RdmaDevice` with `pub fn open(lib: Arc<VerbsLibrary>, name: &str, gid: GidIndex) -> Result<RdmaDevice, RdmaError>` (port 1 must be `ACTIVE`; `auto` = first GID whose sysfs `/sys/class/infiniband/<dev>/ports/1/gid_attrs/types/<i>` reads `RoCE v2` and whose GID is IPv4-mapped `::ffff:a.b.c.d`; none → `Device("no RoCE v2 IPv4 GID")`)
- produces `pub struct MemoryRegion` with `pub fn register(dev: &RdmaDevice, ptr: *mut u8, len: usize, access: Access) -> Result<MemoryRegion, RdmaError>` (unsafe caller contract documented), `pub fn remote(&self) -> RemoteRegion`, `pub enum Access { RemoteRead, LocalWrite }` (never remote write)
- produces `pub struct QueuePair` with `pub fn create(dev: &RdmaDevice, depth: u32) -> Result<QueuePair, RdmaError>`, `pub fn endpoint(&self) -> QpEndpoint` (contract `{ gid, qpn, psn }`), `pub fn connect(&mut self, remote: &QpEndpoint) -> Result<(), RdmaError>` (RESET→INIT→RTR→RTS: INIT with `PKEY_INDEX|PORT|ACCESS_FLAGS` (`REMOTE_READ` on the source side); RTR with `AV` (`is_global = 1`, `grh.dgid = remote.gid`, `sgid_index`, `hop_limit = 64`), `PATH_MTU` = port active MTU, `DEST_QPN`, `RQ_PSN`, `MAX_DEST_RD_ATOMIC = 16`, `MIN_RNR_TIMER = 12`; RTS with `TIMEOUT = 14`, `RETRY_CNT = 7`, `RNR_RETRY = 7`, `SQ_PSN`, `MAX_QP_RD_ATOMIC = 16`), `pub fn post_read(&self, local: &MemoryRegion, local_off: u64, remote: &RemoteRegion, remote_off: u64, len: u32, wr_id: u64) -> Result<(), RdmaError>`, `pub fn poll(&self, out: &mut Vec<WorkCompletion>) -> Result<usize, RdmaError>`, `pub struct WorkCompletion { pub wr_id: u64, pub ok: bool, pub status: String, pub byte_len: u32 }`
- produces `pub struct RdmaTransport` (`impl Transport`, `TransportKind::Rdma`, `caps().rdma = true`, frames delegated to TCP), `pub struct RdmaPeer` (QP + completion task; used by `TransferConn::Rdma`), `pub struct BounceBuffer` (`bounce_bytes`, registered once, used when pool registration fails)
  Covers: S-5 (lab test `--test lab rdma_read_roundtrip`, run in Task 21); unit test `rdma::tests::abi_layouts_match_verbs_h`.
  Depends on: Tasks 9–10, Task 8 (`RdmaConnect`/`RdmaConnectAck`).

- [ ] Write failing test `rdma::tests::abi_layouts_match_verbs_h`: `size_of`/`offset_of!` of the mirrors equal the aarch64 values of rdma-core `verbs.h` (`ibv_sge` 16; `ibv_wc` 48 with `status` at 8, `byte_len` at 20; `ibv_send_wr.wr.rdma.remote_addr` at 40 and `rkey` at 48; `ibv_qp.qp_num` at 52; `ibv_context.ops` at 8; `ibv_mr.lkey`/`rkey` at 36/40), and `ibv_context_ops.poll_cq`/`post_send` sit at byte offsets 88/200 of the ops table (index 11/25). Write failing ignored test `crates/turbine-transport/tests/lab.rs` `rdma_read_roundtrip`: with `TURBINE_RDMA_DEVICE=rocep1s0f0`, `TURBINE_RDMA_ROLE=source|reader` and `TURBINE_RDMA_PEER=192.168.47.245`, the reader pulls 1,000 TKV1-framed payloads of random size 4 KiB–512 MiB from the source over RDMA READ and every segment crc32c matches; missing env → the test fails naming the variable. Run: `cargo test -p turbine-transport rdma::tests::abi_layouts_match_verbs_h` — expect FAIL.
- [ ] Implement the QP code: `ibv_create_cq(ctx, depth, null, null, 0)`, `ibv_create_qp` (RC, `max_send_wr = queue_depth`, `max_recv_wr = 1`, `max_send_sge = max_recv_sge = 1`), transitions via `ibv_modify_qp`, `post_send` through `(*(*qp).context).ops.post_send` with an `IBV_WR_RDMA_READ` + `IBV_SEND_SIGNALED` work request, completions through `ops.poll_cq`, `ibv_wc_status_str` for errors; a failed completion moves the QP to ERR, the peer pair is marked `rdma_degraded` and transfers fall back to TCP (reconnect with backoff capped at 30 s, each attempt logged); every `unsafe` block has a `// SAFETY:` naming the region-lifetime rule (deregister only after all WRs referencing it completed or the QP is in ERR); QP endpoints are exchanged once per peer pair with `RdmaConnect`/`RdmaConnectAck` and a peer's new incarnation forces a new QP (old QPN never reused); the reader registers destination memory `LOCAL_WRITE`, the source `REMOTE_READ` only, only for one transfer.
- [ ] Run: `cargo test -p turbine-transport rdma::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(transport): RoCE v2 RC queue pairs with one-sided RDMA READ over runtime-loaded verbs`

## Task 12: PD request flow and worker router

Files: `crates/turbine-scheduler/src/pd.rs` (new: `PdPath`, `PdRequestFlow`, path choice, tests), `crates/turbine-scheduler/src/router.rs` (new: `WorkerCandidate`, `ScoreTerms`, `PdRouteDecision`, `route_pd`, tests), `crates/turbine-scheduler/src/lib.rs` (modules), `crates/turbine-scheduler/src/metrics.rs` (`turbine_pd_requests_total{path}`).
Interfaces:

- produces `pub enum PdPath { Disaggregated, ShortPromptLocal, FallbackColocate, FallbackRecompute, Rejected }` + `as_str` (contract)
- produces `pub struct PdParams { pub enabled: bool, pub min_prompt_tokens: u32, pub fallback: PdFallback, pub transfer_timeout: Duration, pub max_inflight_transfers: u32 }`
- produces `pub enum PdStep { Reserve, Prefill, Transfer, Decode, Done, Fallback(PdPath), Failed(TransferFailureReason) }`, `pub enum TransferFailureReason { Timeout, Checksum, TransportError, PeerLost, FormatMismatch, ReservationRefused }`
- produces `impl PdRequestFlow { pub fn new(id: RequestId, prompt_tokens: u32, max_tokens: u32, p: &PdParams) -> Self; pub fn path(&self) -> PdPath; pub fn on_reserved(&mut self, held_blocks: &[u32]) -> PdStep; pub fn on_refused(&mut self, second_choice: bool) -> PdStep; pub fn on_first_token(&mut self, is_final: bool) -> PdStep; pub fn on_committed(&mut self) -> PdStep; pub fn on_failure(&mut self, r: TransferFailureReason, decode_can_admit: bool) -> PdStep; pub fn emitted(&self) -> &[u32] }`
- produces `pub enum LinkTransport { Rdma, Tcp, Local }`, `pub struct WorkerCandidate { pub worker: NodeId, pub pressure: PressureState, pub free_kv_blocks: u32, pub queued_prefill_tokens: u64, pub prefill_tps: f64, pub held_prefix_tokens: u32, pub link: LinkTransport, pub link_bytes_per_s: f64, pub rtt_ms: f64 }`, `pub struct ScoreTerms { pub compute: f64, pub locality: f64, pub transfer: f64, pub pressure: f64, pub sla: f64 }`, `pub struct PdRouteQuery { pub prompt_tokens: u32, pub kv_bytes_per_token: u64, pub priority: Priority }`, `pub struct PdRouteDecision { pub prefill: NodeId, pub decode: NodeId, pub reason: &'static str, pub winner: ScoreTerms, pub runner_up: Option<ScoreTerms> }`, `pub fn route_pd(q: &PdRouteQuery, prefill: &[WorkerCandidate], decode: &[WorkerCandidate]) -> Option<PdRouteDecision>`
  Covers: S-3 (`pd::tests::short_and_single_token_requests_skip_transfer`), S-7 (`router::tests::scores_locality_and_cost`).
  Depends on: Task 1, phase-3 (pressure), phase-6 (node ids).

- [ ] Write failing test `pd::tests::short_and_single_token_requests_skip_transfer`: with `min_prompt_tokens: 512`, a 100-token prompt has path `short_prompt_local` and never yields `PdStep::Transfer`; a 2,000-token prompt with `max_tokens: 1` goes Reserve → Prefill, `on_first_token(true)` returns `Done` with path `disaggregated`, zero transfers and the reservation marked released. Run: `cargo test -p turbine-scheduler pd::tests::short_and_single_token_requests_skip_transfer` — expect FAIL.
- [ ] Write failing test `router::tests::scores_locality_and_cost`: equal load, decode worker A holding 90 % of the prefix beats B holding none; an ORANGE decode worker with locality loses to a GREEN one without; for an 8,192-token Llama prompt (896 MiB) an RDMA peer (25 GB/s) beats a 10 GbE TCP peer (1.25 GB/s) with equal compute; the decision's `winner` and `runner_up` carry all five terms and the emitted `pd_route` log event (captured with a `tracing` test subscriber) lists every term of both. Run: `cargo test -p turbine-scheduler router::tests::scores_locality_and_cost` — expect FAIL.
- [ ] Implement the flow per P7 §RDMA transfer protocol and §Failure modes (reserve on D before P starts; one retry on the next-best decode worker after a refusal; `fallback_recompute` when D can admit, else `fallback_colocate` or `rejected` per `fallback`) and `route_pd` scoring TS §11 terms in milliseconds (compute = queue/tps, locality = uncached tokens/tps, transfer = rtt + missing bytes / link bandwidth, pressure 0/50/500/5000, SLA = priority weight), logged as `pd_route` INFO with `reason` and both term sets.
- [ ] Run: `cargo test -p turbine-scheduler pd:: router::` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(scheduler): PD request flow state machine and TS §11 worker router`

## Task 13: PD simulation — flow, admission on both sides, chaos

Files: `crates/turbine-scheduler/src/sim/pd.rs` (new: `PdSim` with ingress, prefill and decode workers on `FakeClock`, injectable transfer latency/loss/corruption, deterministic fake model), `crates/turbine-scheduler/src/sim/mod.rs` (module), `crates/turbine-scheduler/tests/sim.rs` (new: three tests).
Interfaces:

- produces `pub struct PdSimConfig { pub seed: u64, pub prefill_workers: u32, pub decode_workers: u32, pub max_inflight_transfers: u32, pub transfer_ms: u64, pub fallback: PdFallback }`, `pub enum SimFault { KillDecode { at_step: u32 }, KillPrefillBeforeReady, CorruptSegment, DropTransfer, DecodePressure(PressureState) }`
- produces `impl PdSim { pub fn new(cfg: PdSimConfig) -> Self; pub fn submit(&mut self, prompt: Vec<u32>, max_tokens: u32) -> RequestId; pub fn inject(&mut self, f: SimFault); pub fn run_until_idle(&mut self) -> PdSimReport }`, `pub struct PdSimReport { pub streams: BTreeMap<RequestId, SimStream>, pub max_inflight_transfers: u32, pub leaked_reservations: u32, pub leaked_registrations: u32, pub paths: BTreeMap<&'static str, u32>, pub reserved_bytes_peak: (u64, u64) }`, `pub struct SimStream { pub tokens: Vec<u32>, pub first_token_from: &'static str, pub finish: FinishReason, pub error: Option<ErrorCode> }`
- produces `pub fn colocated_tokens(prompt: &[u32], max_tokens: u32) -> Vec<u32>` (reference)
  Covers: S-3/S-11 (`--test sim pd_request_flow`), S-8 (`--test sim pd_admission_both_sides`), S-9/S-11 (`--test sim pd_chaos`).
  Depends on: Task 12, phase-2 simulator, phase-3 admission.

- [ ] Write failing test `pd_request_flow`: 200 requests with seeded random prompt lengths 1–4,000 and `max_tokens` 1–64 through one prefill and one decode worker: each stream equals `colocated_tokens`, `first_token_from == "prefill"`, and `leaked_reservations == leaked_registrations == 0` at the end. Run: `cargo test -p turbine-scheduler --test sim pd_request_flow` — expect FAIL.
- [ ] Write failing test `pd_admission_both_sides`: with the decode worker driven to RED while prefill is GREEN, new disaggregated requests are queued or take the fallback path with reason `reservation_refused`, no request is prefilled for a worker that refused it, `max_inflight_transfers` never exceeds the configured 8, and reserved bytes are counted on both workers from reservation to commit (`reserved_bytes_peak` both non-zero). Run: `cargo test -p turbine-scheduler --test sim pd_admission_both_sides` — expect FAIL.
- [ ] Write failing test `pd_chaos`: 100 generations with the decode worker killed at a seeded random step and 100 with the prefill worker killed before ready: every stream either equals `colocated_tokens` (after `fallback_recompute` or re-routing) or ends with `finish_reason: error` + `worker_lost` and its emitted tokens are a prefix of the colocated sequence with no duplicate or skipped token. Run: `cargo test -p turbine-scheduler --test sim pd_chaos` — expect FAIL.
- [ ] Implement `PdSim` as an event-queue simulation on `FakeClock` (no sleeps; `BTreeMap` ordering) whose workers use the real `PdRequestFlow`, phase-3 admission with worst-case KV reservation per side, and a fake model where token t = hash(prompt ‖ emitted prefix) so recompute from prompt + emitted tokens reproduces the stream.
- [ ] Run: `cargo test -p turbine-scheduler --test sim pd_` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `test(scheduler): PD flow, two-sided admission and chaos simulations`

## Task 14: pipeline micro-batch scheduler and stage loss

Files: `crates/turbine-scheduler/src/pipeline.rs` (new: `MicroBatchScheduler`), `crates/turbine-scheduler/src/sim/pipeline.rs` (new: `PipelineSim`), `crates/turbine-scheduler/tests/sim.rs` (two tests), `crates/turbine-scheduler/src/metrics.rs` (`turbine_pipeline_bubble_ratio`).
Interfaces:

- produces `pub struct MicroBatchScheduler { pub stages: u32, pub micro_batches: u32 }` (contract) with `pub fn new(stages: u32, micro_batches: u32) -> Self`, `pub fn ready_to_launch(&self) -> bool`, `pub fn launch(&mut self, mb: u32, step: u64)`, `pub fn stage_done(&mut self, stage: u32, mb: u32) -> Option<(u32, u32)>` (next stage to run), `pub fn in_flight(&self) -> u32`, `pub fn bubble_ratio(&self, window: Duration, now: Duration) -> f64`, `pub fn stage_lost(&mut self, stage: u32) -> Vec<u32>` (micro-batches to fail)
- produces `pub struct PipelineSim` with `pub fn new(stages: u32, micro_batches: u32, stage_step: Duration, colocated_pool: bool) -> Self`, `pub fn run_decode(&mut self, steps: u64) -> PipelineReport`, `pub fn drop_stage(&mut self, stage: u32, at: Duration)`, `pub fn restore_stage(&mut self, stage: u32, at: Duration)`; `pub struct PipelineReport { pub tokens_per_s: f64, pub bubble_ratio: f64, pub failed: Vec<(RequestId, ErrorCode)>, pub ready_timeline: Vec<(Duration, bool)>, pub colocated_completed: u32 }`
  Covers: S-1/S-11 (`--test sim pipeline_micro_batches_overlap`), S-9 (`--test sim pipeline_stage_loss`).
  Depends on: Task 6, phase-2 simulator.

- [ ] Write failing test `pipeline_micro_batches_overlap`: 2 stages, 10 ms per stage-step: steady-state decode throughput with 2 micro-batches ≥ 1.8 × the 1-micro-batch run and `bubble_ratio ≤ 0.1`. Run: `cargo test -p turbine-scheduler --test sim pipeline_micro_batches_overlap` — expect FAIL.
- [ ] Write failing test `pipeline_stage_loss`: 2 stages, 4 micro-batches in flight, stage 1 dropped at 1 s: exactly those micro-batches' requests fail with `pipeline_stage_lost`, `ready_timeline` is false from the drop until `restore_stage`, the colocated pool in the same simulation keeps completing requests, and no request is left without an outcome. Run: `cargo test -p turbine-scheduler --test sim pipeline_stage_loss` — expect FAIL.
- [ ] Implement the scheduler as a bounded ring of `micro_batches` slots advancing stage by stage (stage s of micro-batch m runs when stage s is free and stage s−1 of m finished), bubble = idle stage-time / total stage-time over a 10 s window; stage loss fails every in-flight micro-batch, sets readiness `pipeline_stage_lost` and stops admission until warm restart.
- [ ] Run: `cargo test -p turbine-scheduler --test sim pipeline_` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(scheduler): pipeline micro-batch scheduler with bubble metric and stage-loss handling`

## Task 15: PP stage and EP executors, all-to-all collective, EP lab test

Files: `crates/turbine-distributed/src/collective/ffi.rs` (bind `ncclSend`, `ncclRecv`; stub test asserts 15 symbols), `crates/turbine-distributed/src/collective/mod.rs` (`all_to_all_v` on the trait), `crates/turbine-distributed/src/collective/host.rs` (host `all_to_all_v` + test), `crates/turbine-model/src/executor/stage.rs` (new: PP stage executor over a layer range, embedding/head flags, activation in/out as BF16 `hidden × tokens`), `crates/turbine-model/src/executor/olmoe.rs` (EP mode: local expert shard, dispatch/combine from Task 7, per-rank and per-expert token counts), `crates/turbine-model/src/loader.rs` (load only the stage's layers / the rank's experts), `crates/turbine-model/tests/tiny_model.rs` (`pp_stages_match_single`, `ep_ranks_match_single` on the CPU provider), `crates/turbine-distributed/tests/ep_lab.rs` (new ignored test), `crates/turbine-distributed/src/metrics.rs` (`turbine_expert_rank_tokens_total{rank}`, `turbine_expert_imbalance_ratio`, `turbine_ep_all_to_all_duration_seconds{phase}`, `turbine_pipeline_stage_duration_seconds{stage}`).
Interfaces:

- produces `fn all_to_all_v(&self, send: &DeviceSlice, send_counts: &[usize], recv: &mut DeviceSlice, recv_counts: &[usize], stream: &StreamRef) -> Result<(), CollectiveError>` (contract; NCCL/RCCL: `ncclGroupStart` + one `ncclSend`/`ncclRecv` per peer + `ncclGroupEnd`; zero counts still enter the group)
- produces `pub struct StageExecutor` with `pub fn new(model: &LoadedModel, stage: &StagePlan, plan: &PipelinePlan) -> Result<Self, ModelError>`, `pub fn forward(&mut self, input: StageInput, kv: &mut KvPoolView) -> Result<StageOutput, ModelError>`; `pub enum StageInput { Tokens(Vec<u32>), Activations(Vec<u8>) }`, `pub enum StageOutput { Activations(Vec<u8>), Logits(Vec<f32>) }`
- produces `pub struct ExpertShard { pub rank: u32, pub placement: Arc<ExpertPlacement> }` and `OlmoeExecutor::with_expert_parallel(shard: ExpertShard, collective: Arc<dyn Collective>)`
  Covers: S-1 (stage executor), S-2/S-12 (`scripts/lab-test.sh novanas` running `cargo test -p turbine-distributed --test ep_lab ep2_olmoe_matches_single_device -- --ignored`); tests `collective::host::tests::all_to_all_v_matches_reference`, `tiny_model pp_stages_match_single`, `tiny_model ep_ranks_match_single`.
  Depends on: Tasks 6–7, phase-5 (collective, TP executor), phase-2 (OLMoE executor).

- [ ] Write failing tests: `collective::host::tests::all_to_all_v_matches_reference` (4 host ranks with uneven counts, one rank sending and receiving zero, equals a reference permutation); `tiny_model pp_stages_match_single` (tiny Llama checkpoint split into 2 and 3 stages on the CPU provider: greedy tokens and logits bitwise equal to the single-stage run); `tiny_model ep_ranks_match_single` (tiny OLMoE, EP 2 over host collective ranks: logits bitwise equal to single-rank); ignored `ep_lab ep2_olmoe_matches_single_device` (starts with `require_backend("hip")`, reads `TURBINE_TEST_MOE_MODEL_DIR`, runs OLMoE EP 2 across both R9700 cards over RCCL vs the single-card run on the 16 golden prompts under `tests/golden/olmoe-1b-7b-0125-instruct/tolerance.json`, and asserts both ranks' token counts > 0). Run: `cargo test -p turbine-distributed collective::host && cargo test -p turbine-model --test tiny_model pp_stages_match_single ep_ranks_match_single` — expect FAIL.
- [ ] Implement the stage executor (runs layers `first..=last`, embedding on stage 0, final norm + LM head on the last, tied embedding loaded on both), activations crossing stages as `Message::StageActivation` over the phase-6 control link (same-host stages use a local channel), and the EP OLMoE layer: route locally, `dispatch`, `all_to_all_v` of hidden rows (buffers sized from `max_tokens_per_rank`), local expert GEMMs, reverse `all_to_all_v`, `combine`; record per-rank tokens (metric) and per-expert tokens (only in `/turbine/v1/scheduler`); an EP rank or stage loss aborts the collective and marks the group unusable.
- [ ] Run: `cargo test --workspace` — expect PASS.
- [ ] Lab (correctness; novanas GPUs must be free — ASK THE USER FIRST if anything holds the R9700 cards): `scripts/lab-test.sh novanas` — expect exit 0 and the log line `test ep2_olmoe_matches_single_device ... ok`.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(model): pipeline stage executor and expert-parallel OLMoE over all-to-all`

## Task 16: server wiring — PD transfer protocol, pipelines, diagnostics and metrics

Files: `crates/turbine-server/src/pd.rs` (new: ingress PD orchestration, decode reservation, prefill-side `PdReady` with TKV1 header + regions, decode-side pull via `TransferConn` (RDMA when both ends have it, else TCP), commit/abort/timeout release, fallbacks, re-routing on worker loss), `crates/turbine-server/src/pipeline.rs` (new: stage placement, `MicroBatchScheduler` driving `StageExecutor`s), `crates/turbine-server/src/startup.rs` (capability probe + validator before bind, exit 2 with the refusal; RDMA selection, exit 1 on explicit failure; `/ready` reasons `pd_roles_incomplete`, `pipeline_stage_lost`), `crates/turbine-server/src/diagnostics.rs` (`pipeline`, `expert`, `pd` scheduler sections; `transfers` KV section (C-18, ≤ 64 in-flight entries); device `capabilities`), `crates/turbine-kv/src/metrics.rs` (`turbine_kv_network_transfer_bytes_total{transport,direction}`, `turbine_kv_transfer_duration_seconds{transport}`, `turbine_kv_transfers_inflight`, `turbine_kv_transfer_failures_total{reason}`, `turbine_kv_conversions_total{kind}`), `crates/turbine-api/src/error.rs` (`kv_transfer_failed`, `pipeline_stage_lost` 503), `crates/turbine-api/tests/api.rs` (test in `mod phase7`), `crates/turbine-server/tests/pd.rs` (new test), `AGENTS.md` (Commands).
Interfaces:

- produces `pub struct PdCoordinator` with `pub async fn serve(&self, req: ClientRequest) -> Result<Submitted, ErrorCode>` (used by the backend when `distributed.pd.enabled`), `pub fn transfers_document(&self) -> serde_json::Value`
- consumes `route_pd`, `PdRequestFlow`, `tkv1::{encode, decode, pack_block, unpack_block, conversion}`, `transfer::{pull_segments, serve_segments, RegionRegistry}`, `ClusterIndex::{begin_transfer, commit_transfer, abort_transfer}`, `Message::Pd*`, phase-3 admission, phase-6 `ClusterNode`
  Covers: S-3, S-8, S-9 (server paths), S-10 (`api distribution_diagnostics_shape`); integration test `turbine-server --test pd pd_over_tcp_tiny_model`.
  Depends on: Tasks 2–15, phase-6 Task 14.

- [ ] Write failing test `crates/turbine-api/tests/api.rs` `distribution_diagnostics_shape`: with fake diagnostics fed by the server's document builders, `/turbine/v1/scheduler` has `pipeline {stages[{index, device_group, layers:[first,last], busy_ratio}], micro_batches_in_flight}`, `expert {parallel_size, placement, tokens_per_rank, top_experts[≤10]{layer, expert, tokens}}`, `pd {role, peers[{worker, role, vendor, state}], inflight_transfers, fallbacks_total}`; `/turbine/v1/kv` `transfers` has the five C-18 keys and `inflight` is capped at 64 when 100 transfers are live; `/turbine/v1/devices` entries carry `capabilities`; `/metrics` contains every P7 family with only the listed label values. Run: `cargo test -p turbine-api --test api distribution_diagnostics_shape` — expect FAIL.
- [ ] Write failing test `crates/turbine-server/tests/pd.rs` `pd_over_tcp_tiny_model`: three processes on 127.0.0.1 with the tiny checkpoint on `cpu` — ingress (`role: both`, no local replica), prefill (`role: prefill`), decode (`role: decode`) — PD over TCP gives greedy tokens identical to a colocated run for 8 prompts, `turbine_pd_requests_total{path="disaggregated"}` = 8, `turbine_kv_conversions_total{kind="copy"}` > 0, and both workers' reservation/registration counts return to 0. Run: `cargo test -p turbine-server --test pd pd_over_tcp_tiny_model` — expect FAIL.
- [ ] Implement the P7 protocol steps 1–4 (reserve → prefill + first token → `PdReady` (TKV1 header + segment table + regions registered `REMOTE_READ` for the decode peer only) → pull, verify each segment, unpack into reserved blocks (or bounce buffer then copy) → `PdCommitted`; release on commit, abort or `transfer_timeout`), in-flight bound `max_inflight_transfers` per worker counted as reserved on both sides, fallbacks and decode-loss recompute from prompt + emitted tokens with no duplicate token, `turbine_pd_requests_total{path}` and the transfer metrics, and the diagnostics sections.
- [ ] Run: `cargo test --workspace` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(server): prefill/decode disaggregation over TKV1, pipelines and distribution diagnostics`

## Task 17: `turbine-bench` profiles, compare and kv-transfer

Files: `benches/turbine-bench/src/main.rs` (`--profile prefill-heavy|decode-heavy`, `compare` and `kv-transfer` subcommands), `benches/turbine-bench/src/report.rs` (compare deltas), `benches/turbine-bench/src/kv_transfer.rs` (new: talks to a `turbine-server` with `distributed.pd.enabled`, drives repeated TKV1 pulls of `--bytes` over `--transport rdma|tcp`, reports p50/p95 GB/s), `benches/turbine-bench/tests/bench.rs` (three tests).
Interfaces:

- produces `pub enum Profile { PrefillHeavy, DecodeHeavy }` (prefill-heavy: 8,192 prompt words, 128 max tokens, concurrency 8; decode-heavy: 256 words, 1,024 max tokens, concurrency 32; seeded like Phase 0 prompts)
- produces `pub fn compare(baseline: &Report, candidate: &Report) -> CompareReport` (ttft, itl, e2e p50/p95/p99 and throughputs, absolute and %)
- produces `pub struct KvTransferReport { pub transport: String, pub bytes: u64, pub iterations: u32, pub p50_gbps: f64, pub p95_gbps: f64 }`
  Covers: S-12 (`turbine-bench` additions); tests `bench profiles_are_seeded`, `bench compare_prints_deltas`, `bench kv_transfer_reports_percentiles`.
  Depends on: Task 16, phase-0 bench.

- [ ] Write failing tests: `profiles_are_seeded` (`--profile prefill-heavy` against the mock endpoint sends 8 concurrent requests of 8,192 words with `max_tokens: 128`, identical prompts for equal seeds); `compare_prints_deltas` (two fixed report JSON files → stdout lists every metric with both values and the delta, exit 0); `kv_transfer_reports_percentiles` (against an in-process tiny PD server over TCP, `kv-transfer --transport tcp --bytes 16MiB --iterations 10 --output json` prints `p50_gbps` and `p95_gbps` > 0). Run: `cargo test -p turbine-bench --test bench` — expect FAIL.
- [ ] Implement the profiles, `compare`, and `kv-transfer` (control connection with the PSK from `--psk-file`, repeated `PdReserve`/`PdReady`-style pulls of a synthetic region, GB/s per iteration).
- [ ] Run: `cargo test -p turbine-bench --test bench` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(bench): prefill/decode-heavy profiles, report compare and kv-transfer bandwidth`

## Task 18: lab configs, k3s manifests and `scripts/lab-pd.sh`

Files: `scripts/lab/pd-prefill-spark.yaml`, `scripts/lab/pd-decode-spark2.yaml`, `scripts/lab/pp2-spark.yaml` (port 18100, `reliability.memory.device_budget_bytes` sized by the script from measured headroom via `--set`, RDMA device `rocep1s0f0`, data 18111), `scripts/lab/ep2-novanas.yaml`, `scripts/lab/pd-prefill-novanas.yaml` (HIP, TCP), `scripts/lab/k3s/ep2-novanas-job.yaml`, `scripts/lab/k3s/pd-prefill-novanas-job.yaml` (host networking, `amd.com/gpu`, port 18100, name `turbine-lab-*`), `scripts/lab-pd.sh` (`<pd|pp|ep|colocated|rdma-test> [--hetero] [--bench <profile>] [--dry-run]`: prints each host's free memory and budget, exits 1 `precondition` when it does not fit or when 18100 is taken (C-20), starts `docker run --rm --gpus all --network host --device /dev/infiniband --ulimit memlock=-1` containers `turbine-lab-*` on the Sparks and k3s Jobs on novanas, waits for `/ready`, optionally benches, removes only `turbine-lab-*`), `benches/turbine-bench/tests/lab_scripts.rs` (`pd_dry_run_commands`).
Interfaces:

- consumes `turbine-server --config <yaml> --set reliability.memory.device_budget_bytes=<bytes>`, `turbine-bench --profile`, `turbine-golden capture|compare`
  Covers: S-12 (lab files and runbook); test `lab_scripts pd_dry_run_commands`.
  Depends on: Tasks 16–17, phase-6 Task 15 (Spark image with NCCL).

- [ ] Write failing test `lab_scripts pd_dry_run_commands`: `scripts/lab-pd.sh --dry-run pd --bench prefill-heavy` prints a prefill container on dgx-spark and a decode container on dgx-spark2, both `turbine-lab-*` with `--device /dev/infiniband`, port 18100 and a `device_budget_bytes` override; `--dry-run pd --hetero` prints a `kubectl apply` of `scripts/lab/k3s/pd-prefill-novanas-job.yaml` and a decode container on dgx-spark; no line stops, kills or removes anything not named `turbine-lab-*`. Run: `cargo test -p turbine-bench --test lab_scripts pd_dry_run_commands` — expect FAIL.
- [ ] Implement the files and script (budget = MemAvailable − `reliability.memory.host_reserve_bytes` − 4 GiB margin, capped at the run's size from the spec; refusal text tells the operator to ask the user to free the host).
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(lab): PD, PP and EP lab configs, k3s manifests and lab-pd.sh`

## Task 19: lab — PP token identity across two Sparks

Files: `.procoder/evidence/phase-7-pp.md` (task evidence: pasted command output).
Interfaces:

- consumes `scripts/lab-pd.sh colocated`, `scripts/lab-pd.sh pp --bench decode-heavy`, `turbine-golden capture|compare`
  Covers: S-1/S-12 (manual PP lab run).
  Depends on: Task 18.

- [ ] Write failing check: before Task 16's pipeline wiring is deployed, `scripts/lab-pd.sh --dry-run pp` must list two stage containers (`turbine-lab-pp-stage0` on dgx-spark, `turbine-lab-pp-stage1` on dgx-spark2). Run: `scripts/lab-pd.sh --dry-run pp` — expect FAIL until Task 18 is merged.
- [ ] Implement nothing new; this task is the lab run.
- [ ] Run (correctness; after the precondition passes, otherwise ASK THE USER FIRST): `scripts/lab-pd.sh colocated` then `turbine-golden capture --url http://192.168.10.246:18100 --prompts tests/golden/prompts.jsonl --out colocated.jsonl` — expect PASS with `captured 16 prompts`.
- [ ] Run (includes the `decode-heavy` benchmark: ASK THE USER FIRST): `scripts/lab-pd.sh pp --bench decode-heavy` then `turbine-golden compare --url http://192.168.10.246:18100 --reference colocated.jsonl` — expect PASS with `identical prefix = max_tokens on 16/16 prompts`; paste both outputs into the evidence file.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `docs(evidence): phase 7 pipeline token-identity lab run`

## Task 20: lab — EP on novanas against the HF reference

Files: `.procoder/evidence/phase-7-ep.md` (task evidence).
Interfaces:

- consumes `scripts/lab-pd.sh ep`, `turbine-golden compare`, `/metrics` `turbine_expert_rank_tokens_total{rank}`
  Covers: S-2/S-12 (manual EP lab run).
  Depends on: Tasks 15, 18.

- [ ] Write failing check: `scripts/lab-pd.sh --dry-run ep` must print the `kubectl apply` of `scripts/lab/k3s/ep2-novanas-job.yaml` with `amd.com/gpu: 2`. Run: `scripts/lab-pd.sh --dry-run ep` — expect FAIL until Task 18 is merged.
- [ ] Implement nothing new; this task is the lab run.
- [ ] Run (the R9700 cards must be free — ASK THE USER FIRST to move any workload using them): `scripts/lab-pd.sh ep` then `turbine-golden compare --url http://192.168.10.203:18100 --reference tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl --tolerance tests/golden/olmoe-1b-7b-0125-instruct/tolerance.json` — expect PASS with `passed ≥ 14/16`.
- [ ] Run: `curl -s http://192.168.10.203:18100/metrics | grep turbine_expert_rank_tokens_total` — expect both `rank="0"` and `rank="1"` non-zero; paste all output into the evidence file.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `docs(evidence): phase 7 expert-parallel OLMoE lab run`

## Task 21: lab — RDMA READ round trip and bandwidth on the Sparks

Files: `.procoder/evidence/phase-7-rdma.md` (task evidence).
Interfaces:

- consumes `scripts/lab-pd.sh rdma-test` (runs `cargo test -p turbine-transport --test lab rdma_read_roundtrip -- --ignored` as `TURBINE_RDMA_ROLE=source` on dgx-spark2 and `reader` on dgx-spark with `TURBINE_RDMA_DEVICE=rocep1s0f0 TURBINE_RDMA_PEER=192.168.47.245`), `turbine-bench kv-transfer`, `ib_write_bw`
  Covers: S-5/S-12 (manual RDMA lab run).
  Depends on: Tasks 11, 17, 18.

- [ ] Write failing check: `scripts/lab-pd.sh --dry-run rdma-test` must print both test containers with `--device /dev/infiniband --ulimit memlock=-1` and the three `TURBINE_RDMA_*` variables. Run: `scripts/lab-pd.sh --dry-run rdma-test` — expect FAIL until Task 18 is merged.
- [ ] Implement nothing new; this task is the lab run.
- [ ] Run (correctness; after the precondition): `scripts/lab-pd.sh rdma-test` — expect PASS with `test rdma_read_roundtrip ... ok` and `1000 payloads verified`.
- [ ] Run (bandwidth benchmark: ASK THE USER FIRST): `ib_write_bw -d rocep1s0f0 --report_gbits -s 8388608` (server on dgx-spark2, client on dgx-spark) then `turbine-bench kv-transfer --peer 192.168.47.245:18111 --transport rdma --bytes 256MiB --iterations 50 --output json` (C-21 port) — expect PASS with `p50_gbps` ≥ 50 % of the `ib_write_bw` figure; paste both outputs into the evidence file.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `docs(evidence): phase 7 RDMA round trip and bandwidth lab run`

## Task 22: lab — PD milestone M1 (Spark → Spark over RoCE)

Files: `.procoder/evidence/phase-7-pd-m1.md` (task evidence).
Interfaces:

- consumes `scripts/lab-pd.sh colocated --bench prefill-heavy`, `scripts/lab-pd.sh pd --bench prefill-heavy`, `turbine-golden compare`, `turbine-bench compare`
  Covers: S-3/S-12 (manual PD M1 run).
  Depends on: Tasks 16–18, 21.

- [ ] Write failing check: `scripts/lab-pd.sh --dry-run pd --bench prefill-heavy` must print `distributed.transport.rdma.enabled: true` for both workers. Run: `scripts/lab-pd.sh --dry-run pd --bench prefill-heavy` — expect FAIL until Task 18 is merged.
- [ ] Implement nothing new; this task is the lab run.
- [ ] Run (benchmarks on production hosts: ASK THE USER FIRST; record `docker ps` on both Sparks before): `scripts/lab-pd.sh colocated --bench prefill-heavy` (saves `colocated.json`) then `scripts/lab-pd.sh pd --bench prefill-heavy` (saves `pd.json`) — expect both exit 0 with `requests_ok` equal to the requests sent.
- [ ] Run: `turbine-golden compare --url http://192.168.10.246:18100 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` against the PD deployment and `turbine-bench compare --baseline colocated.json --candidate pd.json` — expect PASS within tolerance and PD `itl_ms.p99` ≤ colocated `itl_ms.p99`, with `turbine_kv_network_transfer_bytes_total{transport="rdma"}` > 0; `docker ps` after the run shows production containers unchanged; paste everything into the evidence file.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `docs(evidence): phase 7 PD milestone M1 over RoCE`

## Task 23: lab — PD milestone M2 (AMD prefill → NVIDIA decode over 10 GbE)

Files: `.procoder/evidence/phase-7-pd-m2.md` (task evidence).
Interfaces:

- consumes `scripts/lab-pd.sh pd --hetero`, `turbine-golden compare`, `/metrics` `turbine_kv_conversions_total{kind}`, `turbine_pd_requests_total{path}`
  Covers: S-3/S-6/S-12 (manual PD M2 run).
  Depends on: Task 22 (M2 starts only after M1 passes).

- [ ] Write failing check: `scripts/lab-pd.sh --dry-run pd --hetero` must show the novanas prefill Job with `execution.backend: hip` and TCP transport and the decode container on dgx-spark with `execution.backend: cuda`. Run: `scripts/lab-pd.sh --dry-run pd --hetero` — expect FAIL until Task 18 is merged.
- [ ] Implement nothing new; this task is the lab run.
- [ ] Run (uses the R9700 cards and a Spark: ASK THE USER FIRST): `scripts/lab-pd.sh pd --hetero` then `turbine-golden compare --url http://192.168.10.246:18100 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` — expect PASS within the tolerance file.
- [ ] Run: `curl -s http://192.168.10.246:18100/metrics | grep -E 'turbine_kv_conversions_total|turbine_pd_requests_total'` — expect `kind="lossy"` 0, `kind="copy"` > 0 and `path="disaggregated"` > 0; paste all output into the evidence file.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `docs(evidence): phase 7 heterogeneous PD milestone M2`
