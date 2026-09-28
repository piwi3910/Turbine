# phase-5-multi-gpu — implementation plan

Status: draft
Spec: .procoder/specs/phase-5-multi-gpu.md

## Goal

Let one Turbine process drive several GPUs: discover the node-local topology graph, abstract collectives behind a backend-neutral `Collective` trait with a host reference backend and one runtime-loaded NCCL-API binding (RCCL now, NCCL later), plan single-vendor TP groups and DP replicas from link classes, run TP (local and static rank modes) and DP behind one API, and account pressure per device, group and replica — proven on the novanas R9700 pair. Amended 2026-09-28: single-node pipeline and expert parallelism (Tasks 22–27), a `hostmem` collective backend for the peer-less board (Task 21) and measured host-link costs for its asymmetric slots (Task 20).

## Architecture

`turbine-device::topology` builds a typed `TopologyGraph` from a sysfs root plus a `TopologyVendor` (amd-smi / NVML) and never fails. The new crate `turbine-distributed` holds `collective` (trait, `host` backend over `turbine_tensor::host::HostMemory`, `ffi` — the only `unsafe` module — with `NcclApi` and the watchdog-guarded `NcclCollective`), `plan` (config + inventory + graph → `ParallelPlan`, before bind), `rank` (leader/worker lockstep over bounded channels in `local` mode or postcard frames over a registered rank transport in `static` mode), `transport` (the `rank_transport` registry: `tcp`), `tp` (pure sharding rules), `router` (DP replica choice through the `dp_router_policy` registry: `prefix_affinity`, `least_loaded`) and the `turbine-collbench` binary. `turbine-model` shards weights and inserts collectives through `TpContext`, the optional kernel C ABI minor group v2.6 supplies the stream handle and the sharded-norm ops (decision "P5 T6", answer B), `turbine-reliability::multi_device` adds group state, atomic group reservations and shared-device budgets, and `turbine-server` builds one engine per DP replica and one `RankRuntime` per TP group.

## Constraints

Copied verbatim from the spec (Constraints):

- Rust only; no Python in the build or runtime path (TS §21 rule 4). RCCL and NCCL are loaded at runtime; nothing links against them at build time, so the workspace still builds and every non-ignored test passes on macOS arm64 with no GPU libraries.
- `unsafe` and FFI stay isolated (TS §21 rule 10): in `turbine-distributed` only the module `collective::ffi` may contain `unsafe`; the crate root has `#![deny(unsafe_code)]` and that module carries `#[allow(unsafe_code)]`, each block with a `// SAFETY:` comment naming the owner of every device pointer, stream and communicator it touches. Device buffers and streams come from the phase-1 device layer; `turbine-distributed` never allocates model memory. The phase-1 `unsafe_isolation` allowlist gains exactly that one module.
- New dependency, with its reason: `postcard` (compact serde framing for the static-mode rank protocol; no schema compiler). Local mode uses bounded `std::sync::mpsc::sync_channel`.
- Every queue is bounded (TS §21 rule 8): the leader→worker plan channel holds at most _parallel.plan_queue_depth_ plans (default 2); frames on the static-mode socket are capped at 16 MiB.
- No collective call may block forever: communicator init is bounded by `parallel.collective.init_timeout` and each step's collectives by `parallel.collective.op_timeout`; on expiry the communicator is aborted, the group enters the phase-3 circuit breaker as `CIRCUIT_OPEN`, and the reason is logged.
- Every automatic decision (device grouping, vendor choice, backend choice, replica routing, pressure state of a group) emits a structured log line with a reason code and a metric (TS §14, §21 rule 7).
- Correctness before speed (TS §21 rule 1): TP=2 must meet the phase-1 golden tolerance (≥ 14/16 prompts with the first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats against the committed HF transformers BF16 fixtures) at concurrency 1 and the batched bounds at concurrency 16, as for any batching change, before any TP performance work is accepted; the same golden gate (c1 strict, c16 batched) holds for PP and EP. Amended 2026-09-28 (user decision "P5: OLMoE golden tolerance under expert parallelism", A then C): a multi-GPU run (EP, TP, PP) is gated against a one-GPU capture of the same model and commit taken in the same lab run (`turbine-golden capture`, then `compare` with the model's `tolerance.json`: strict at c1, batched at c16); one GPU stays gated against the committed transformers reference, and the multi-GPU verdict against that reference is reported for information only; `turbine-golden positions` prints a prompt's teacher-forced per-position |Δ| for such checks. Amended 2026-09-28 (user decision "P5: tensor-parallel accuracy gate against the one-GPU capture", A): a mode that includes tensor parallelism (tp > 1, including ep × tp) is gated against the one-GPU capture with the batched bounds at c1 and c16 (`turbine-golden compare --batched-bounds`) and against the committed transformers reference with the batched bounds at c1 and c16 as well (follow-up (a), 2026-09-28: Llama tp 2 p16 tail 0.584 against strict 0.55, batched 0.75); one GPU keeps the strict transformers gate at c1; for OLMoE under TP the transformers reference is the only gate and the capture leg is informational (follow-up 2026-09-28: OLMoE's batched bounds equal its strict ones; its TP drift is investigated in a later phase); expert and pipeline parallelism without TP stay strict against the capture (c1 strict, c16 batched). That golden check is the TP accuracy bound (user decision 2026-09-28, "P5: tensor-parallel accuracy bound"): TP output is not bit-exact with tp = 1 and raw logits are not bounded, since each all-reduce rounds BF16 partial sums (~1 % relative logit drift, identical greedy tokens).
- Lab: Phase 5 hardware runs are on novanas only (k3s Job, `amd.com/gpu: 2`, `rust:1.97-trixie` image as in phase-0/phase-1 (CONFLICT C-7), hostPath `/opt/rocm/rocm` read-only; RCCL is `/opt/rocm/rocm/lib/librccl.so.1`). Ports are loopback inside the Job (HTTP 18000, static-mode leader 18100); no Service or host port is created. Read-only topology checks also run under `scripts/lab-test.sh dgx-spark`. Memory is capped by the phase-3 device budget.
- Before any lab run that needs production workloads moved, GPUs emptied or memory freed on any host, the implementer asks the user first and waits; scripts never evict, scale or stop other workloads, and a Job that cannot be scheduled (GPUs in use) makes the script exit non-zero naming that reason.
- With peer access disabled on novanas, RCCL uses host-staged transfers; the topology graph must report `p2p: disabled` rather than assume.

From the interface contract (`.procoder/contract/interfaces.md`, binding):

- Toolchain edition 2024, `rust-version = "1.97"`; `#[non_exhaustive]` on enums later phases extend; config structs `#[serde(deny_unknown_fields, default)]`; one `thiserror` error enum per crate; time through `Arc<dyn Clock>`.
- `turbine-distributed` sets `unsafe_code = "deny"` in its manifest with `#[allow(unsafe_code)]` only on `collective::ffi` (file `crates/turbine-distributed/src/collective/ffi.rs`); the `unsafe_isolation` allowlist adds exactly `crates/turbine-distributed/src/collective/ffi` (§1.3, CONFLICT C-19).
- `turbine-distributed` depends on core, observability, tensor, device, reliability, kv; `turbine-model` may depend on `turbine-distributed` (`collective`, `tp`); `turbine-reliability` never depends on `turbine-device`.
- Durations accept `ms`, `s`, `m`, `h` through the one parser (CONFLICT C-14); an incomplete group reservation queues with `PressureReason::KvReservation` (CONFLICT C-11); the novanas lab image is `rust:1.97-trixie` with hostPath ROCm (CONFLICT C-7); golden fixtures live under `tests/golden/llama-3.2-3b-instruct/` and `tests/golden/olmoe-1b-7b-0125-instruct/` (CONFLICT C-15).
- Kernel C ABI v2.6 (§9.1; decision "P5 T6", answer B, supersedes the planned v4): optional minor group with `turbine_stream_native_handle` and the ops `row_sumsq`, `rmsnorm_sharded` in the ROCm shim (CUDA on hold); `TURBINE_KERNELS_ABI_VERSION` stays 2, `TURBINE_ABI_MINOR` = 6; no `fill`.
- Metric labels from closed enums; `device` and `replica` rendered as decimal indices; pressure states upper-case (CONFLICT C-5).
- `/turbine/v1/scheduler` becomes an object keyed by replica index (`{"0": {…}}`); `/ready` adds `collective_init`, `loading_weights`, `rank_missing`; static-mode workers answer inference routes with 503 `not_leader`.
- Lab: `scripts/lab-cluster.sh [--dry-run] <collbench-novanas|collbench-sweep-novanas|collbench-hostmem-novanas|tp2-novanas|dp2-novanas|pp2-novanas|ep2-novanas>`, one k3s Job labelled `turbine-lab=true` in namespace `turbine-ci`; no `docker run` outside lab scripts.

## Task 1: `parallel` configuration section

Files: `crates/turbine-core/src/config/parallel.rs` (section structs, value types, static validation, unit test), `crates/turbine-core/src/config/mod.rs` (add `Config.parallel`, call `self.parallel.validate()?`), `examples/turbine.yaml` (document the section)
Interfaces:

- `pub struct ParallelConfig { pub tensor_parallel_size: SizeOrAuto, pub data_parallel_size: SizeOrAuto, pub devices: DeviceSelection, pub collective_backend: CollectiveBackendChoice, pub nccl_library: Option<PathBuf>, pub rccl_library: Option<PathBuf>, pub allow_device_sharing: bool, pub plan_queue_depth: u32, pub router: DpRouterPolicy, pub collective: CollectiveTimeouts, pub ranks: RanksConfig }`
- `pub enum SizeOrAuto { Auto, Size(u32) }` (YAML integer or `auto`), `pub enum DeviceSelection { Auto, List(Vec<DeviceId>) }`, `pub enum CollectiveBackendChoice { Auto, Nccl, Rccl, Host }`, `pub enum DpRouterPolicy { PrefixAffinity, LeastLoaded }`, `pub enum RankMode { Local, Static }`
- `pub struct CollectiveTimeouts { pub init_timeout: HumanDuration, pub op_timeout: HumanDuration }` (contract addition: struct name for `parallel.collective`), `pub struct RanksConfig { pub mode: RankMode, pub rank: u32, pub leader: Option<SocketAddr>, pub local_devices: Vec<DeviceId> }` (contract addition: struct name for `parallel.ranks`)
- `impl ParallelConfig { pub fn validate(&self) -> Result<(), ConfigError>; pub fn validate_devices(&self, inv: &[(DeviceId, Vendor, Option<String>)]) -> Result<(), ConfigError> }`
  Covers: S-4 (configuration); `config::tests::parallel_rejections`
  Depends on: phase-0 plan (loader), phase-2 plan (`HumanDuration`), phase-3 plan (config sections)

- [ ] Write failing test `config::tests::parallel_rejections`: asserts the defaults (tp 1, dp 1, `devices: auto`, backend `auto`, depth 2, router `prefix_affinity`, 120s/30s, mode `local`, `local_devices: [0]`) and that each of these fails with an error naming its key: `tensor_parallel_size: 3`, `tensor_parallel_size: 16`, `data_parallel_size: 0`, `devices: [0, 0]` with tp 2, `collective_backend: host` with tp 2 and inventory device 0 a GPU (`validate_devices`), `ranks.mode: static` with tp 1, `ranks.mode: static` without `ranks.leader`, `collective.op_timeout: 50ms`, `collective.op_timeout: 30 s`. Run: `cargo test -p turbine-core config::tests::parallel_rejections` — expect FAIL.
- [ ] Implement: static rules from the spec's table (tp 1..8 power of two; dp 1..64; depth 1..16; init 1 s..30 m; op 100 ms..10 m; `static` requires tp > 1 and dp = 1 and a leader; `rank < tp`; exactly one `local_devices` entry in `static`); inventory rules in `validate_devices` (indices exist, one vendor, no index twice in a TP group, list length = tp × dp unless sharing, `host` backend with GPUs and tp > 1 rejected, `rccl` with NVIDIA / `nccl` with AMD rejected). Model-dependent head divisibility lives in the planner (Task 10). `"30 s"` fails in the shared duration parser because of the space.
- [ ] Run: `cargo test -p turbine-core config::tests::parallel_rejections` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(core): parallel configuration section with static validation`

## Task 2: Node-local topology discovery

Files: `crates/turbine-device/src/topology/mod.rs` (graph types, `TopologyVendor`, assembly, unit tests), `crates/turbine-device/src/topology/sysfs.rs` (NUMA, PCIe tree, NICs, InfiniBand, NVMe parsers over a sysfs root), `crates/turbine-device/src/topology/vendor.rs` (`AmdSmiTopology`, `NvmlTopology`, `NoVendorTopology`), `crates/turbine-device/src/discovery/amd_smi.rs` (resolve `amdsmi_topo_get_link_type`, `amdsmi_is_P2P_accessible`), `crates/turbine-device/tests/fixtures/topology/novanas/capture.txt`, `crates/turbine-device/tests/fixtures/topology/dgx-spark/capture.txt` (captured sysfs files as `path<TAB>content` lines plus `vendor.json`), `crates/turbine-device/src/lib.rs`
Interfaces:

- `pub fn discover_topology(sysfs_root: &Path, inventory: &DeviceInventory, vendor: &mut dyn TopologyVendor) -> TopologyGraph` (contract §5.3; never fails)
- `pub trait TopologyVendor { fn link(&mut self, a: &DeviceInfo, b: &DeviceInfo) -> Result<VendorLink, String>; fn gpu_nic_p2p(&mut self, gpu: &DeviceInfo) -> Result<Option<P2pStatus>, String>; fn coherent_host_link(&mut self, gpu: &DeviceInfo) -> Result<Option<String>, String>; }` and `pub struct VendorLink { pub kind: EdgeKind, pub path: PathClass, pub hops: Option<u32>, pub p2p: P2pStatus }` (contract additions)
- `pub struct TopologyGraph { pub node: TopologyNode, pub vertices: Vec<Vertex>, pub edges: Vec<Edge> }`, `Vertex`, `VertexKind`, `VertexAttrs`, `Edge`, `EdgeKind`, `PathClass` (with `pub fn rank(self) -> u8`, NVLink/xGMI best … SYS worst), `P2pStatus`, `AttrSource` per contract §5.3
- test helper `pub fn materialize_capture(capture: &Path, into: &Path) -> std::io::Result<()>` (`#[doc(hidden)]`, writes the capture lines as files)
  Covers: S-1; `topology::tests::novanas_fixture`, `topology::tests::spark_fixture`, `topology::tests::missing_sources_degrade`
  Depends on: phase-0 plan (inventory, amd-smi and NVML loaders)

- [ ] Write failing test `topology::tests::novanas_fixture`: materialises the novanas capture (two gfx1201 GPUs at 32 GT/s x16, `numa_node` = -1, amd-smi link type PCIE, 2 hops, P2P accessible false) and asserts exactly one GPU↔GPU edge with `kind: pcie`, `path: sys`, `p2p: disabled`, `hops: 2`, `source: vendor`, and both GPU vertices with `numa: 0`, `numa_source: nominal`, plus one WARN per `numa_node = -1`. Run: `cargo test -p turbine-device topology::tests::novanas_fixture` — expect FAIL.
- [ ] Write failing test `topology::tests::spark_fixture`: the dgx-spark capture yields one `gpu` vertex, four `nic` vertices with RDMA devices `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1` (two at 200 Gb/s), a `coherent` GPU↔`numa0` edge with `vendor_interconnect: nvlink_c2c`, and `p2p: unknown` on every GPU↔NIC edge. Run: `cargo test -p turbine-device topology::tests::spark_fixture` — expect FAIL.
- [ ] Write failing test `topology::tests::missing_sources_degrade`: an empty sysfs root with a vendor whose calls all return `Err` produces a graph (no panic) with the GPU vertices from the inventory, `unknown`/`null` attributes without `source`, the GPU↔GPU path `sys`, and exactly one WARN per missing source (`numa`, `pci`, `net`, `infiniband`, `nvme`, `vendor_link`). Run: `cargo test -p turbine-device topology::tests::missing_sources_degrade` — expect FAIL.
- [ ] Implement: NUMA from `devices/system/node/node*/{cpulist,meminfo,distance}`; PCIe tree by resolving `bus/pci/devices/<bdf>` symlinks to their parent chain (root ports → `pcie_root`, bridges with class `0x0604` → `pcie_switch`), `current_link_speed` (e.g. `32.0 GT/s PCIe`) and `current_link_width`; NICs from `class/net/<if>/{device,speed,address}` joined to `class/infiniband/<dev>/{device,ports/1/link_layer,ports/1/rate}`; IPv4 addresses from `/proc/net/fib_trie` under the same root; NVMe from `class/nvme`. GPU↔GPU edges come from the vendor (`amdsmi_topo_get_link_type(src, dst, &hops, &type)`, `amdsmi_is_P2P_accessible(src, dst, &bool)`; NVML `Device::topology_common_ancestor` and `Device::p2p_status(&other, P2pCapabilitiesIndex::Read)` in nvml-wrapper 0.13), else from the PCIe tree with `source: nominal`; failures leave `unknown` and log WARN once per source.
- [ ] Run: `cargo test -p turbine-device topology::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(device): node-local topology graph from sysfs and vendor links`

## Task 3: `GET /turbine/v1/topology`

Files: `crates/turbine-api/src/backend.rs` (`Diagnostics::topology`, `TopologyScope`), `crates/turbine-api/src/routes/diagnostics.rs` (handler), `crates/turbine-api/src/routes/mod.rs` (route), `crates/turbine-server/src/startup.rs` (capture the graph after discovery with `discover_topology(Path::new("/sys"), …)` and log `topology_captured`), `crates/turbine-api/tests/api.rs` (test)
Interfaces:

- `fn topology(&self, scope: TopologyScope) -> Result<serde_json::Value, ApiError>` (contract §14.1), `pub enum TopologyScope { Node }` (`#[non_exhaustive]`; P6 adds `Cluster`)
  Covers: S-1; `api topology_route`
  Depends on: Task 2; phase-0 plan (router, `Diagnostics`)

- [ ] Write failing test `api topology_route`: with a diagnostics fake returning a two-GPU graph, `GET /turbine/v1/topology` returns 200 whose `vertices` and `edges` arrays equal the injected graph's serialisation. Run: `cargo test -p turbine-api --test api topology_route` — expect FAIL.
- [ ] Implement the route (query `scope=node` optional; other values 400 `unsupported_parameter` until P6) and the server's capture at startup.
- [ ] Run: `cargo test -p turbine-api --test api topology_route` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(api): node topology diagnostics route`

## Task 4: `turbine-distributed` crate, `Collective` trait and host backend

Files: `crates/turbine-distributed/Cargo.toml` (new crate; `unsafe_code = "deny"`; deps core, observability, tensor, device, reliability, kv, `postcard`, `libloading`, `half`), `crates/turbine-distributed/src/lib.rs` (`#![deny(unsafe_code)]`, modules), `crates/turbine-distributed/src/collective/mod.rs` (trait, `ReduceOp`, `CollectiveError`, `CollectiveMetrics`), `crates/turbine-distributed/src/collective/host.rs` (thread-rank reference backend + unit tests), `Cargo.toml` (member, workspace deps `postcard = { version = "1", features = ["use-std"] }`)
Interfaces:

- `pub trait Collective: Send + Sync` exactly as contract §15.1 (P5 methods; `all_to_all_v` arrives with P7), `pub enum ReduceOp { Sum, Max }`, `pub use turbine_core::types::CollectiveBackendKind`
- `pub enum CollectiveError { Timeout { op: &'static str, after: Duration }, RemoteAbort { rank: usize }, Backend { code: i32, message: String }, ShapeMismatch, Unavailable { library: String, detail: String } }`
- `impl HostCollective { pub fn group(world: usize, op_timeout: Duration) -> Vec<HostCollective>; pub fn with_dtype(self, dtype: DType) -> Self }` (DType BF16 or F32 decides element interpretation; contract addition)
- `pub struct CollectiveMetrics` with `register(reg: &MetricsRegistry) -> Self`, `observe(op: CollectiveOp, backend: CollectiveBackendKind, bytes: u64, seconds: f64)`, `error(backend, kind: CollectiveErrorKind)`; `pub enum CollectiveOp { AllReduce, AllGather, ReduceScatter, Broadcast, Barrier }`
  Covers: S-2; `collective::host::tests::ops_match_reference`, `collective::host::tests::op_timeout_aborts`
  Depends on: phase-1 plan (`DeviceSlice::{read_bytes, write_bytes}`, `StreamRef`, `host::HostMemory`)

- [ ] Write failing test `collective::host::tests::ops_match_reference`: for world sizes 1, 2, 3, 4, 8 and FP32 and BF16 buffers of 1, 7 and 4099 elements (seeded values), runs all-reduce (Sum and Max), all-gather, reduce-scatter and broadcast on one thread per rank and asserts each rank's result equals a naive single-threaded computation bit for bit (BF16 sums accumulate in FP32 in rank order 0..n then round once). Run: `cargo test -p turbine-distributed collective::host::tests::ops_match_reference` — expect FAIL.
- [ ] Write failing test `collective::host::tests::op_timeout_aborts`: with `op_timeout` 500 ms and world 2, rank 1 never enters; rank 0's all-reduce returns `Timeout { op: "all_reduce", .. }` within 1.5 s and every later call on either rank returns `RemoteAbort`. Run: `cargo test -p turbine-distributed collective::host::tests::op_timeout_aborts` — expect FAIL.
- [ ] Implement the host backend: a shared rendezvous (`Mutex` + `Condvar`) per operation generation; each rank deposits its bytes, the last arrival computes the result in rank order and releases the others; waits use `wait_timeout` and on expiry set a shared aborted flag (then every call returns `RemoteAbort`); lengths that disagree return `ShapeMismatch`; buffers are read/written through `DeviceSlice::read_bytes`/`write_bytes` over `HostMemory`.
- [ ] Run: `cargo test -p turbine-distributed collective::host::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): collective trait and deterministic host backend`

## Task 5: Runtime-loaded NCCL-API binding and the unsafe allowlist

Files: `crates/turbine-distributed/src/collective/ffi.rs` (`#[allow(unsafe_code)]`; `NcclApi` binding table, version check, unit tests), `crates/turbine-distributed/build.rs` (compile `tests/stub/nccl_stub.c` with `cc` into `librccl.so.1`, `libnccl.so.2` and a low-version variant under `OUT_DIR`), `crates/turbine-distributed/tests/stub/nccl_stub.c` (the 13 symbols; version from `-DSTUB_VERSION`), `crates/turbine-kernels/tests/unsafe_isolation.rs` (allowlist + one path)
Interfaces:

- `impl NcclApi { pub fn load(kind: CollectiveBackendKind, explicit: Option<&Path>) -> Result<Arc<NcclApi>, CollectiveError>; pub fn load_from(path: &Path, kind: CollectiveBackendKind) -> Result<Arc<NcclApi>, CollectiveError>; pub fn backend(&self) -> CollectiveBackendKind; pub fn version(&self) -> i32; pub fn path(&self) -> &Path }` (`load_from` is a contract addition used by tests)
- `pub const RCCL_MIN_VERSION: i32` (the `NCCL_VERSION_CODE` of the RCCL shipped with ROCm 7.14.1, read from `/opt/rocm/rocm/include/rccl/rccl.h` on novanas), `pub const NCCL_MIN_VERSION: i32 = 22_700`
- resolved symbols (exactly 13): `ncclGetVersion`, `ncclGetUniqueId`, `ncclCommInitRankConfig`, `ncclCommGetAsyncError`, `ncclCommAbort`, `ncclCommDestroy`, `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast`, `ncclGroupStart`, `ncclGroupEnd`, `ncclGetErrorString`
  Covers: S-2; `collective::ffi::tests::missing_library`, `collective::ffi::tests::one_binding_both_libraries`, `turbine-kernels --test unsafe_isolation`
  Depends on: Task 4; phase-1 plan (`unsafe_isolation` test)

- [ ] Write failing test `collective::ffi::tests::missing_library`: on macOS `NcclApi::load(Rccl, None)` and `load(Nccl, None)` each return `Unavailable` whose detail carries the loader message, and `load(Rccl, Some("/nonexistent/librccl.so.1"))` returns an error whose text names that path; nothing panics. Run: `cargo test -p turbine-distributed collective::ffi::tests::missing_library` — expect FAIL.
- [ ] Write failing test `collective::ffi::tests::one_binding_both_libraries`: `load_from` of the stub named `librccl.so.1` reports backend `rccl` and of the stub named `libnccl.so.2` reports `nccl`, both through the same `NcclApi` type with all 13 symbols resolved and `version()` equal to the stub's; the low-version stub is rejected with `Unavailable` naming the version; a stub missing `ncclGroupEnd` (built with `-DSTUB_OMIT_GROUP_END`) is rejected naming the symbol. Run: `cargo test -p turbine-distributed collective::ffi::tests::one_binding_both_libraries` — expect FAIL.
- [ ] Extend `crates/turbine-kernels/tests/unsafe_isolation.rs` with `crates/turbine-distributed/src/collective/ffi` and run `cargo test -p turbine-kernels --test unsafe_isolation` — expect FAIL until the module exists, then it must also fail if `unsafe` appears anywhere else in `turbine-distributed`.
- [ ] Implement with `libloading::Library::new` (0.9); default search: RCCL `/opt/rocm/lib/librccl.so.1`, then `librccl.so.1` on the loader path; NCCL `libnccl.so.2` on the loader path; an explicit path that fails is fatal naming it. Kind is taken from the file name the caller asked for (`librccl*` → `rccl`, `libnccl*` → `nccl`). Symbols are copied out as `extern "C"` function pointers while the `Library` stays owned by `NcclApi`; each `unsafe` block states that the library outlives every pointer copied from it. The stub C file implements the 13 functions with the NCCL signatures (`ncclResult_t` = int, `ncclUniqueId` = 128 bytes) and simple host semantics for world size 1.
- [ ] Run: `cargo test -p turbine-distributed collective::ffi::tests && cargo test -p turbine-kernels --test unsafe_isolation` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): one runtime-loaded NCCL-API binding for RCCL and NCCL`

## Task 6: Kernel C ABI v2.6 — stream handle, sharded norm

Decision "P5 T6", answer B (2026-09-28): an optional minor group v2.6 instead of the planned major v4 (`TURBINE_ABI_VERSION` stays 2, `TURBINE_ABI_MINOR` 6), following the v2.5 precedent; `fill` is dropped (vocab padding uses the existing copies); the CUDA shim is on hold (NVIDIA on hold) and gains the group with its v2.5 group. Provider evaluation: decision "P5 T6: sharded RMSNorm — provider evaluation".

Files: `kernels/include/turbine_kernels.h` (v2.6 block: `turbine_stream_native_handle`, `turbine_row_sumsq_desc`, `turbine_rmsnorm_sharded_desc`, their trios, `TURBINE_OP_ROW_SUMSQ 15`, `TURBINE_OP_RMSNORM_SHARDED 16`), `kernels/rocm/src/sharded_norm.hip` (`row_sumsq`, `rmsnorm_sharded` trios), `kernels/rocm/src/copy_stream.cpp` (`turbine_stream_native_handle`), `kernels/rocm/src/impl_table.cpp` (one `turbine_hip` implementation each), `kernels/rocm/src/abi_minor.cpp`, `kernels/rocm/CMakeLists.txt`, `crates/turbine-kernels/src/ffi.rs` (`RowSumsqDesc`, `RmsnormShardedDesc`, `TensorParallelFns` resolved when minor ≥ 6), `crates/turbine-kernels/src/ops/mod.rs` (configs, contexts, `ShardedNormKernel`, `KernelProvider::sharded_norm`, `OpKind::abi_minor`), `crates/turbine-kernels/src/registry.rs` (`OpConfig::{RowSumsq, RmsnormSharded}`, accessors), `crates/turbine-kernels/src/cpu/norm.rs` and `cpu/math.rs` (CPU reference, exact FP32), `crates/turbine-kernels/src/shim.rs` (`StreamRef` native handle filled from v2.6, the shim provider), `crates/turbine-kernels/src/cards/gfx1201.rs` (preference `turbine_hip`), `crates/turbine-kernels/build.rs` and `stub/stub_shim.c` (`TURBINE_STUB_GFX942_V26`), `crates/turbine-kernels/tests/abi_header_neutral.rs`, `crates/turbine-kernels/tests/hip_ops.rs` (`sharded_norm_ops`, ignored)
Interfaces:

- C: `int32_t turbine_stream_native_handle(turbine_ctx*, turbine_stream*, void**)`, `turbine_row_sumsq[_supported|_impl]`, `turbine_rmsnorm_sharded[…]` with the descriptor structs of contract §9.3
- `OpKind::{RowSumsq, RmsnormSharded}` (codes 15, 16; `OpKind::abi_minor()` = 6); `ShardedNormKernel` (`supports_row_sumsq`, `supports_rmsnorm_sharded`, `implementation_row_sumsq`, `implementation_rmsnorm_sharded`, `row_sumsq(&mut RowSumsqContext)`, `rmsnorm_sharded(&mut RmsnormShardedContext)`), `KernelProvider::sharded_norm()`, `KernelRegistry::{row_sumsq, rmsnorm_sharded}` (contract §7.1)
- `ShimLibrary::exports_native_streams() -> bool`, `ShimContext::has_native_streams() -> bool`; `DeviceMemory::compute_stream()` of a `ShimContext` carries the native handle (0 without v2.6)
  Covers: S-6 (kernel support); `hip_ops sharded_norm_ops` (contract addition)
  Depends on: phase-1 plan (shim, registry, `cpu-reference`), phase-4 plan (v2.5 minor-group precedent)

- [ ] Write failing tests: `cpu::norm::tests::sharded_norm_matches_full` asserts that splitting a 4×256 row (F32 and BF16) into two halves, summing `row_sumsq` of the halves and applying `rmsnorm_sharded` per half equals the full `rmsnorm` within 1e-6 (F32; BF16 within one rounding step) and that one full-width slice is bitwise `rmsnorm`; `shim::tests::v26_native_stream_and_sharded_norm_group` (a v2.5 stub has handle 0 and no family, the v2.6 stub resolves the group); ignored `hip_ops sharded_norm_ops` asserts the HIP ops match the CPU reference for BF16 inputs (sums within 1e-5 relative, outputs bitwise except ≤ 0.1 % one-rounding-step flips), row invariance (one row alone vs in a 37-row call, bitwise) and a non-null compute-stream handle. Run: `cargo test -p turbine-kernels sharded_norm_matches_full` — expect FAIL.
- [ ] Implement: `row_sumsq` accumulates FP32 per row (one 256-thread block per row, the rmsnorm fallback's order); `rmsnorm_sharded` computes `round(round(x · 1/sqrt(sumsq / full_dim + eps)) · weight_shard)`; `turbine_stream_native_handle` returns the `hipStream_t` of the context (NULL = compute stream) as an opaque pointer; the library reports minor 6.
- [ ] Run: `cargo test -p turbine-kernels sharded_norm_matches_full` — expect PASS; lab `scripts/lab-test.sh novanas --tier quick -- -p turbine-kernels --test hip_ops` — expect `test sharded_norm_ops ... ok`; `scripts/lab-test.sh novanas --tier quick -- -p turbine-model --test tiny_model` — no regression.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(kernels): ABI v2.6 stream handle, sharded RMSNorm ops`

## Task 7: `NcclCollective` with init and op watchdogs

Files: `crates/turbine-distributed/src/collective/ffi.rs` (`NcclCollective` implementing `Collective`; communicator init/abort/destroy; watchdog thread), `crates/turbine-distributed/src/collective/mod.rs` (`pub fn open_collective(kind, api: Option<Arc<NcclApi>>, rank, world, unique_id: [u8; 128], init_timeout, op_timeout, clock, metrics) -> Result<Arc<dyn Collective>, CollectiveError>`)
Interfaces:

- `impl NcclApi { pub fn unique_id(&self) -> Result<[u8; 128], CollectiveError> }`
- `impl NcclCollective { pub fn init(api: Arc<NcclApi>, rank: usize, world: usize, unique_id: [u8; 128], init_timeout: Duration, op_timeout: Duration, clock: Arc<dyn Clock>, metrics: CollectiveMetrics) -> Result<Self, CollectiveError> }`
- `pub fn open_collective(…) -> Result<Arc<dyn Collective>, CollectiveError>` (contract addition)
  Covers: S-2 (watchdog, abort); exercised by Task 9's lab run
  Depends on: Tasks 5, 6

- [ ] Write failing test `collective::ffi::tests::stub_init_times_out`: against a stub built with `-DSTUB_INIT_NEVER_COMPLETES` (`ncclCommGetAsyncError` keeps returning `ncclInProgress` = 7), `NcclCollective::init` with a 200 ms init timeout returns `Timeout { op: "comm_init", .. }` within 1 s and the stub records one `ncclCommAbort` call. Run: `cargo test -p turbine-distributed collective::ffi::tests::stub_init_times_out` — expect FAIL.
- [ ] Implement: build `ncclConfig_t` with `blocking = 0` exactly as the installed header's `NCCL_CONFIG_INITIALIZER` (size, magic `0xcafebeef`, version, then fields in header order — verify against `/opt/rocm/rocm/include/rccl/rccl.h` on novanas and NCCL 2.27 `nccl.h` read-only before coding), call `ncclCommInitRankConfig` then poll `ncclCommGetAsyncError` every 1 ms until `ncclSuccess` or the deadline (then `ncclCommAbort`); every op passes `StreamRef::native_handle()`, maps dtype BF16 → `ncclBfloat16` (9), FP32 → `ncclFloat32` (7) and `ReduceOp` Sum → 0, Max → 2, and records a deadline that the watchdog thread checks every 10 ms (`ncclCommGetAsyncError` non-success or deadline passed → `ncclCommAbort`, `Timeout`/`Backend`/`RemoteAbort` returned to the caller and counted in `turbine_collective_errors_total`); `barrier` is an all-reduce of one FP32 on the stream followed by a stream sync.
- [ ] Run: `cargo test -p turbine-distributed collective::ffi::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): NCCL-API communicator with init and op watchdogs`

## Task 8: `turbine-collbench` binary

Files: `crates/turbine-distributed/src/bin/turbine-collbench.rs` (CLI, size sweep, busbw, host comparison, report, `tests` module), `crates/turbine-distributed/Cargo.toml` (`[[bin]]`, `clap`)
Interfaces:

- CLI per contract §19 (`--backend rccl|nccl|host --devices <i,j,..> [--op …] [--min-bytes 8] [--max-bytes 1GiB] [--iters 20] [--warmup 5] [--dtype bf16|fp32] [--rank --world --leader] [--output text|json]`)
- `fn busbw_factor(op: CollectiveOp, n: usize) -> f64`; JSON row `{bytes, time_us, algbw_gbps, busbw_gbps, correct}`
  Covers: S-3; `--bin turbine-collbench tests::busbw_formulas`
  Depends on: Tasks 4, 7; phase-1 plan (kernel context for device buffers)

- [ ] Write failing test `tests::busbw_formulas`: asserts factors 2(n−1)/n for all-reduce, (n−1)/n for all-gather and reduce-scatter and 1 for broadcast at n = 2 and n = 8, and that a serialised row has exactly the keys `bytes`, `time_us`, `algbw_gbps`, `busbw_gbps`, `correct`. Run: `cargo test -p turbine-distributed --bin turbine-collbench tests::busbw_formulas` — expect FAIL.
- [ ] Implement: sizes double from `--min-bytes` to `--max-bytes`; per size run `--warmup` then `--iters` timed iterations (median `time_us`), algbw = bytes / time, busbw = algbw × factor; one thread per local device in one process, or one process per rank with `--rank/--world/--leader` (rank 0 listens on the leader address and writes the 128-byte `ncclGetUniqueId` value to each connecting rank; no other protocol); results copied back and compared with `HostCollective` on the same inputs (`correct`); exit 0 all correct, 1 any mismatch or collective error, 2 usage errors. Device buffers and streams come from `ShimContext` (`DeviceBuffer::alloc`, `compute_stream`); the binary adds no HIP/CUDA binding.
- [ ] Run: `cargo test -p turbine-distributed --bin turbine-collbench tests::busbw_formulas` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): turbine-collbench with nccl-tests bandwidth formulas`

## Task 9: `scripts/lab-cluster.sh` and the RCCL collective run

Files: `scripts/lab-cluster.sh` (scenario runner with `--dry-run`), `scripts/lab/novanas-cluster-job.yaml` (Job template beside the other novanas templates: a GPU-less build Job then the scenario Job with `amd.com/gpu: 2`, image `rust:1.97-trixie`, label `turbine-lab=true` plus `turbine-lab-run`/`turbine-scenario`, hostPaths `/opt/rocm/rocm` and `/home/piwi/turbine-models` read-only, `TURBINE_TEST_MODEL_DIR`, cached release slot `cluster-0`), `scripts/lab/phase5-novanas-{llama,olmoe,dp2}.yaml` (the scenario configs, created here so the runner is complete), `benches/turbine-bench/tests/lab_scripts.rs` (dry-run test)
Interfaces:

- `scripts/lab-cluster.sh [--dry-run] <collbench-novanas|tp2-novanas|dp2-novanas>`; final line `lab-cluster: <scenario> PASS` or `lab-cluster: <scenario> FAIL <reason>`; unschedulable Job → exit 1 with `lab-cluster: amd.com/gpu unavailable on novanas`
  Covers: S-3, S-9; `bash -n scripts/lab-cluster.sh` + `lab-cluster.sh --dry-run tp2-novanas`, lab `lab-cluster.sh collbench-novanas`
  Depends on: Task 8; phase-0 plan (`scripts/lab-test.sh`, `turbine-ci` namespace)

- [ ] Write failing test `lab_scripts cluster_dry_run_manifest`: runs `bash -n scripts/lab-cluster.sh` and `scripts/lab-cluster.sh --dry-run tp2-novanas`, both exit 0, and the printed manifest requests `amd.com/gpu: 2`, mounts `/home/piwi/turbine-models` with `readOnly: true`, contains no `kind: Service` and no `hostPort`, and the script's cleanup command selects only `-l turbine-lab=true,turbine-lab-run=<run id>` (its own run). Run: `cargo test -p turbine-bench --test lab_scripts cluster_dry_run_manifest` — expect FAIL.
- [ ] Implement the script: rsync the tree to `/home/piwi/turbine-ci/src`, render the Job for the scenario, `kubectl -n turbine-ci apply`, wait for scheduling (Pending with `Insufficient amd.com/gpu` for 60 s → delete the Job, exit 1), stream logs, exit with the Job's status, `trap` and the end of the run delete only the Jobs it created (`-l turbine-lab=true,turbine-lab-run=<run id>`; `--stop <run-id>` does the same from elsewhere); the scenario steps run inside the Job as `scripts/lab-cluster.sh --in-job <scenario>`. `collbench-novanas` runs `turbine-collbench --backend rccl --devices 0,1 --op all --max-bytes 1GiB --output json` inside the Job and fails unless every row is `correct: true` and all-reduce `busbw_gbps` > 0 at 268,435,456 bytes.
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts cluster_dry_run_manifest` — expect PASS.
- [ ] Lab (ASK THE USER FIRST: both R9700 must be free): `scripts/lab-cluster.sh collbench-novanas` — expect exit 0 and final log line `lab-cluster: collbench-novanas PASS`; paste the JSON into the task evidence.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(scripts): lab-cluster runner with RCCL collective benchmark scenario`

## Task 10: Parallel planner

Files: `crates/turbine-distributed/src/plan.rs` (planner, reason codes, unit tests), `crates/turbine-distributed/src/lib.rs`
Interfaces:

- `pub struct ParallelPlan { pub tp: u32, pub dp: u32, pub backend: CollectiveBackendKind, pub mode: RankMode, pub vendor: Vendor, pub excluded_devices: Vec<DeviceId>, pub groups: Vec<ReplicaGroup>, pub reasons: Vec<PlanReason> }`, `pub struct ReplicaGroup { pub replica: ReplicaId, pub ranks: Vec<RankSlot> }`, `pub struct RankSlot { pub rank: u32, pub device: DeviceId, pub host: String }`
- `pub enum PlanReason { FitsSingleDevice, TpRequiredForCapacity, GroupedByLink(PathClass), VendorHomogeneous, VendorExcluded(Vendor), ExplicitDevices, DeviceSharingEnabled }` (Display per §Data)
- `pub fn plan(inv: &DeviceInventory, topo: &TopologyGraph, cfg: &ParallelConfig, model: &ModelShape, device_budget: &dyn Fn(DeviceId) -> u64) -> Result<ParallelPlan, PlanError>`; `pub struct PlanError { pub key: String, pub reason: String }`
  Covers: S-4; `plan::tests::planner_cases`
  Depends on: Tasks 1, 2

- [ ] Write failing test `plan::tests::planner_cases`: on synthetic graphs asserts the novanas graph with tp 2 → one group {0,1}, reason `grouped_by_link:sys`, backend `rccl`; four NVIDIA GPUs as two NVLink pairs with tp 2 → groups {0,1} and {2,3} with `grouped_by_link:nvlink`; `devices: [0,1]` with one NVIDIA and one AMD → error `vendor-mixed plan` for tp 2 and for tp 1/dp 2; three AMD + one NVIDIA with `devices: auto`, tp 1 → dp 3 on AMD with the NVIDIA index in `excluded_devices` and `vendor_excluded:nvidia`; Llama-3.2-3B shape with `tp: auto` on two 32 GiB R9700 budgets → tp 1, dp 2, `fits_single_device`; tp 4 with 2 KV heads accepted, tp 8 with 3 KV heads rejected naming `parallel.tensor_parallel_size`. Run: `cargo test -p turbine-distributed plan::tests::planner_cases` — expect FAIL.
- [ ] Implement: vendor choice (most usable devices, tie → vendor of the lowest index; explicit lists must be single-vendor and single-arch per TP group); head rules (tp ≤ heads, tp divides heads, tp divides KV heads or is a multiple of them); `tp: auto` = smallest power of two whose per-rank weight shard + one max-length sequence's KV fits `device_budget`; TP groups built greedily from the best `PathClass::rank` edges between unassigned devices, never by index order; DP replicas take the remaining devices; each decision logs INFO with its reason and is returned in `reasons`.
- [ ] Run: `cargo test -p turbine-distributed plan::tests::planner_cases` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): topology-driven single-vendor parallel planner`

## Task 11: Plan before bind in `turbine-server`

Files: `crates/turbine-server/src/startup.rs` (startup step 3: `ParallelConfig::validate_devices` + `plan`, exit 2 before bind; `turbine_parallel_info` gauge; `/turbine/v1/status` `parallel` object), `crates/turbine-server/src/parallel.rs` (the step itself: cpu backend and single-GPU default via `plan_execution_device`, otherwise `validate_devices` + `plan` with the model shape read from `config.json`; a plan with tp > 1 or dp > 1 is exit 2 until Tasks 17/18 wire the execution; the engine device is rank 0 of replica 0), `crates/turbine-distributed/src/plan.rs` (`plan_execution_device`, `PlanReason::ExecutionDevice`, `ParallelPlan.vendor: Option<Vendor>`), `crates/turbine-server/src/backend.rs` (status `parallel`), `crates/turbine-server/tests/server_cli.rs` (test); `exit.rs` needs no change (`ExitCode::Config` is 2)
Interfaces:

- consumes `turbine_distributed::plan::{plan, ParallelPlan, PlanError}`; status `"parallel": {"tp","dp","backend","mode","groups":[{"replica","ranks":[{"rank","device","host"}]}],"plan_reasons"}`
  Covers: S-4; `server_cli impossible_plan_exits_2_before_bind`
  Depends on: Task 10; phase-0/phase-1 plans (startup order, exit codes)

- [ ] Write failing test `server_cli impossible_plan_exits_2_before_bind`: starts the binary with `parallel.tensor_parallel_size: 2`, no GPU libraries (empty inventory) and a free port; asserts exit code 2, stderr containing `parallel.tensor_parallel_size`, and that the port can still be bound afterwards. Run: `cargo test -p turbine-server --test server_cli impossible_plan_exits_2_before_bind` — expect FAIL.
- [ ] Implement the startup step between discovery and kernel loading (contract §16.3 step 3), registering `turbine_parallel_info{tp,dp,backend,mode} = 1` and storing the plan for the status document.
- [ ] Run: `cargo test -p turbine-server --test server_cli impossible_plan_exits_2_before_bind` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(server): validate the parallel plan before binding`

## Task 12: Rank runtime — local and static modes

Files: `crates/turbine-distributed/src/rank.rs` (messages, framing, leader/worker runtime, unit tests), `crates/turbine-distributed/src/transport/{mod.rs,tcp.rs,conformance.rs}` (the `rank_transport` registry, added 2026-09-28), `crates/turbine-distributed/src/lib.rs`
Interfaces:

- Rank transport registry (user decision 2026-09-28, "P5: the static-mode rank link and the DP router policy as registries?", answer B): `pub trait Transport: Module { fn listen(&self, addr: SocketAddr) -> io::Result<Box<dyn RankListener>>; fn connect(&self, addr: SocketAddr, timeout: Duration) -> io::Result<Box<dyn RankStream>>; }`, `pub trait RankListener: Send { fn local_addr(&self) -> io::Result<SocketAddr>; fn accept(&self, timeout: Duration) -> io::Result<Option<Box<dyn RankStream>>>; }`, `pub trait RankStream: Read + Write + Send { fn try_clone(&self) -> io::Result<Box<dyn RankStream>>; fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()>; fn shutdown(&self) -> io::Result<()>; }`, `pub fn registry() -> &'static Registry<dyn Transport>` (`tcp`), `pub fn select(name: &str) -> Result<&'static dyn Transport, UnknownModule>`; configuration key `parallel.ranks.transport` (default `tcp`, checked by `Config::validate_modules`); suite `transport::conformance::check` run by `registry_conformance::rank_transports`; page `docs/extending/rank-transport.md`

- `pub struct StepPlan { pub step: u64, pub sequences: Vec<StepSeq> }`, `pub struct StepSeq { pub seq_id: SeqId, pub tokens: Vec<u32>, pub positions: Vec<u32>, pub block_table: Vec<BlockId>, pub is_prefill: bool }`, `pub trait StepExecutor: Send { fn execute(&mut self, plan: &StepPlan) -> Result<StepOutput, ExecError>; }`, `pub struct StepOutput { pub logits: Option<Vec<f32>>, pub rows: usize, pub vocab: usize }`, `pub enum ExecError { Collective(CollectiveError), Executor(String), DeviceFatal(String) }` (contract §15.2)
- `pub enum RankMessage { Hello { protocol: u16, rank: u32, world_size: u32, model_fingerprint: ModelFingerprint, config_fingerprint: [u8; 32], device_vendor: Vendor, device_arch: String }, Welcome { unique_id: [u8; 128] }, Reject { reason: String }, StepPlan(StepPlan), Shutdown { reason: String } }`
- `pub fn write_frame(w: &mut impl Write, m: &RankMessage) -> std::io::Result<()>; pub fn read_frame(r: &mut impl Read) -> std::io::Result<RankMessage>` (u32 LE length, ≤ 16 MiB, postcard)
- `impl RankRuntime { pub fn local(executors: Vec<Box<dyn StepExecutor>>, depth: usize) -> Self; pub fn static_leader(transport: &dyn Transport, listen: SocketAddr, expect: HelloExpect, world: usize, init_timeout: Duration, unique_id: [u8; 128], depth: usize) -> Result<Self, RankError>; pub fn static_worker(transport: &dyn Transport, leader: SocketAddr, hello: RankMessage, init_timeout: Duration) -> Result<WorkerLink, RankError>; pub fn step(&mut self, plan: StepPlan) -> Result<(), RankError>; pub fn shutdown(&mut self, reason: &str) }`; `pub struct HelloExpect { pub model_fingerprint: ModelFingerprint, pub config_fingerprint: [u8; 32], pub device_vendor: Vendor, pub device_arch: String }`; `pub enum RankError { Timeout { missing: Vec<u32> }, Rejected(String), Closed { rank: u32 }, Io(String) }` (contract additions)
  Covers: S-5; `rank::tests::static_protocol_handshake`, `rank::tests::leader_loss_aborts_workers`, `rank::tests::plan_queue_bounded`
  Depends on: Task 4

- [ ] Write failing test `rank::tests::static_protocol_handshake`: over loopback TCP a leader and 3 workers exchange `Hello`/`Welcome` (all workers receive the same 128-byte id); a worker with a different `model_fingerprint` and one with `device_vendor: nvidia` each receive `Reject` whose reason names the field; a duplicate rank is rejected; with rank 2 absent the leader fails within the 1 s init timeout with `Timeout { missing: [2] }`. Run: `cargo test -p turbine-distributed rank::tests::static_protocol_handshake` — expect FAIL.
- [ ] Write failing test `rank::tests::leader_loss_aborts_workers`: after the handshake the leader socket is dropped; every worker aborts its host-backend communicator (later calls return `RemoteAbort`) and its loop returns `RankError::Closed { rank: 0 }` within 2 s. Run: `cargo test -p turbine-distributed rank::tests::leader_loss_aborts_workers` — expect FAIL.
- [ ] Write failing test `rank::tests::plan_queue_bounded`: in local mode with depth 2 and a worker executor blocked on a barrier, the leader's third `step` blocks (observed with a 200 ms timeout thread) instead of buffering, and resumes when the worker is released. Run: `cargo test -p turbine-distributed rank::tests::plan_queue_bounded` — expect FAIL.
- [ ] Implement: local mode = one thread per rank fed by `std::sync::mpsc::sync_channel(depth)`; static mode = leader accepts until all ranks joined or the init timeout, checks fingerprints/vendor/arch/unique ranks, answers `Welcome` or `Reject`, then writes each `StepPlan` to every worker; workers retry connect with backoff (50 ms doubling to 1 s) until the init timeout; a closed socket is `RemoteAbort` (workers abort the communicator, release device memory and exit 1 from the server); `Shutdown` travels both ways.
- [ ] Run: `cargo test -p turbine-distributed rank::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): leader/worker rank runtime with local and static modes`

## Task 13: Tensor-parallel sharding rules

Files: `crates/turbine-distributed/src/tp.rs` (range functions, unit test with the in-test FP32 reference decoder), `crates/turbine-distributed/src/lib.rs`
Interfaces:

- `pub struct ShardSpec { pub rank: u32, pub world: u32 }`
- `pub fn head_range(num_heads: u32, s: ShardSpec) -> Range<u32>`, `pub fn kv_head_range(num_kv_heads: u32, s: ShardSpec) -> Range<u32>`, `pub fn column_range(dim: u32, s: ShardSpec) -> Range<u32>`, `pub fn row_range(dim: u32, s: ShardSpec) -> Range<u32>`, `pub fn vocab_shard(vocab: u32, s: ShardSpec) -> (u32, u32, u32)` (contract §15.2)
  Covers: S-6; `tp::tests::sharded_layers_match_unsharded`
  Depends on: Task 4

- [ ] Write failing test `tp::tests::sharded_layers_match_unsharded`: an in-test naive FP32 decoder block (seeded weights) — dense: 8 heads / 2 KV heads, tied embeddings, vocab 1003; MoE: 4 heads / 4 KV heads with full-projection QK-norm, 8 experts top-2, vocab 1003 — executed at tp 1, 2 and 4 on `HostCollective` using only the range functions (column-parallel q/k/v, gate/up and every expert along the intermediate dimension, row-parallel o/down + all-reduce, replicated router, QK-norm from all-reduced partial sums of squares, vocab-parallel embedding + all-reduce, LM head + all-gather with padded rows filled −∞, KV heads replicated at tp 4 with 2 KV heads) gives logits within 1e-5 absolute of tp 1. Run: `cargo test -p turbine-distributed tp::tests::sharded_layers_match_unsharded` — expect FAIL.
- [ ] Implement: contiguous equal splits (`dim % world == 0` asserted for heads/intermediate); `kv_head_range` returns `rank / (world / kv_heads)` as a one-head range when `world > kv_heads` and `world % kv_heads == 0`; `vocab_shard` = (rank × ceil(vocab/world), rows in range, ceil(vocab/world)).
- [ ] Run: `cargo test -p turbine-distributed tp::tests::sharded_layers_match_unsharded` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): tensor-parallel sharding rules verified against an unsharded reference`

## Task 14: Multi-device pressure accounting

Files: `crates/turbine-reliability/src/multi_device.rs` (group state, atomic reservation, shared budgets, unit tests), `crates/turbine-reliability/src/budget.rs` (`PoolKind::Collective`, `BudgetInputs.collective_bytes`), `crates/turbine-reliability/src/metrics.rs` (`turbine_device_budget_bytes`, `turbine_group_pressure_state`, `turbine_group_limiting_device`), `crates/turbine-reliability/src/document.rs` (`devices[]`, `groups[]`, `replicas[]`), `crates/turbine-reliability/src/lib.rs`
Interfaces:

- `pub struct GroupState { pub state: PressureState, pub limiting_device: DeviceId }`, `pub fn group_state(members: &[(DeviceId, PressureState)]) -> (PressureState, DeviceId)` (contract §8.2)
- `pub fn reserve_group(ledgers: &[(DeviceId, Arc<Ledger>)], blocks: u32, block_bytes_per_rank: u64) -> Result<Vec<Reservation>, AdmissionDecision>` (all-or-nothing → `Queue { reason: PressureReason::KvReservation }`), `pub struct GroupReservation { pub reservations: Vec<Reservation> }` (RAII wrapper, contract addition of fields)
- `pub struct SharedDeviceBudget` with `pub fn split(budget: &DeviceBudget, replicas: u32) -> Vec<DeviceBudget>` and `pub fn available(&self, device: DeviceId) -> u64`
  Covers: S-8; `multi_device::tests::group_state_is_worst_member`, `multi_device::tests::atomic_group_reservation`, `multi_device::tests::shared_device_budget`
  Depends on: phase-3 plan (`Ledger`, `PressureMachine`, `compute_budget`)

- [ ] Write failing test `multi_device::tests::group_state_is_worst_member`: two `PressureMachine`s fed so device 0 is GREEN and device 1 ORANGE → group ORANGE with `limiting_device` 1; after device 1's samples drop below ORANGE the group stays ORANGE until device 1 has cleared the phase-3 hysteresis (`deescalate_dwell` on the fake clock), then returns stepwise to GREEN. Run: `cargo test -p turbine-reliability multi_device::tests::group_state_is_worst_member` — expect FAIL.
- [ ] Write failing test `multi_device::tests::atomic_group_reservation`: with rank 1's `kv` pool one block short of N, `reserve_group` for N blocks returns `Queue { reason: KvReservation }` and both ledgers' `usage(kv)` are unchanged (nothing held on rank 0). Run: `cargo test -p turbine-reliability multi_device::tests::atomic_group_reservation` — expect FAIL.
- [ ] Write failing test `multi_device::tests::shared_device_budget`: two replicas on one 32 GiB dedicated device each see half of the phase-3 device budget, and a KV reservation by replica 0 reduces replica 1's available headroom by the same bytes (one ledger per physical device, never two). Run: `cargo test -p turbine-reliability multi_device::tests::shared_device_budget` — expect FAIL.
- [ ] Implement: group state = max of member states, limiting = the first device at that state; `reserve_group` reserves rank by rank and drops the already taken guards on the first failure; the collective component is measured by `turbine-server` as the drop of `mem_info().free_bytes` across communicator init and passed as `BudgetInputs.collective_bytes`; the pressure document adds the three arrays.
- [ ] Run: `cargo test -p turbine-reliability multi_device::tests` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): group pressure state, atomic group reservations, shared device budgets`

## Task 15: DP router

Files: `crates/turbine-distributed/src/router/mod.rs` (routing decision, metric, unit test; was `router.rs`), `crates/turbine-distributed/src/router/{prefix_affinity.rs,least_loaded.rs,conformance.rs}` (the `dp_router_policy` registry, added 2026-09-28), `crates/turbine-distributed/src/lib.rs`
Interfaces:

- `pub enum DpRouteReason { PrefixAffinity, LeastLoaded, PressureAvoidance, OnlyCandidate }` (label values verbatim), `pub struct ReplicaView { pub replica: ReplicaId, pub state: PressureState, pub circuit: CircuitState, pub outstanding_tokens: u64, pub has_prefix: bool }`
- DP router policy registry (user decision 2026-09-28, "P5: the static-mode rank link and the DP router policy as registries?", answer B): `pub trait RouterPolicy: Module { fn choose(&self, views: &[ReplicaView]) -> (ReplicaId, DpRouteReason); }` (called with at least two views), helpers `ReplicaView::eligible`, `candidates(views)`, `least_loaded(iter)`, `pub fn registry() -> &'static Registry<dyn RouterPolicy>` (`prefix_affinity`, `least_loaded`), `pub fn select(name: &str) -> Result<&'static dyn RouterPolicy, UnknownModule>`; `parallel.router` is a `ModuleName` (same names, default `prefix_affinity`) checked by `Config::validate_modules`, and the `DpRouterPolicy` enum is gone; suite `router::conformance::check` run by `registry_conformance::dp_router_policies`; page `docs/extending/dp-router-policy.md`; the server's `ReplicaRouter::new` takes the selected `&'static dyn RouterPolicy`
- `pub fn route(views: &[ReplicaView], policy: &dyn RouterPolicy) -> (ReplicaId, DpRouteReason)` (contract §15.2); `pub struct RouterMetrics { pub fn register(reg: &MetricsRegistry) -> Self; pub fn routed(&self, r: ReplicaId, reason: DpRouteReason) }`
  Covers: S-7; `router::tests::routing_policy`
  Depends on: Task 4

- [ ] Write failing test `router::tests::routing_policy`: prefix cached on replica 1 → (1, `prefix_affinity`); no prefix → the fewest outstanding tokens with `least_loaded`; prefix on an ORANGE replica while another is GREEN → the GREEN one with `pressure_avoidance`; all replicas RED → the least-loaded RED one; one replica → `only_candidate`; each decision passed to `RouterMetrics::routed` increments `turbine_dp_routed_total{replica,reason}` by one. Run: `cargo test -p turbine-distributed router::tests::routing_policy` — expect FAIL.
- [ ] Implement: eligible = state < ORANGE and circuit not `CIRCUIT_OPEN`/`DRAINING`, falling back to all replicas when none is eligible; with `prefix_affinity` a prefix holder among the eligible wins; otherwise the least outstanding tokens (ties → lowest index); `pressure_avoidance` when a prefix holder was skipped for pressure; the server fills `has_prefix` from each replica's `KvDirectory::lookup` of the prompt's first block keys.
- [ ] Run: `cargo test -p turbine-distributed router::tests::routing_policy` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(distributed): pressure-aware prefix-affinity DP router`

## Task 16: Diagnostics for replicas, groups and devices

Files: `crates/turbine-server/src/diagnostics.rs` (pressure document with `devices`, `groups`, `replicas`; scheduler keyed by replica; `/ready` P5 reasons; static workers `not_leader`), `crates/turbine-api/src/backend.rs` (`NotReadyReason::{CollectiveInit, LoadingWeights, RankMissing}`), `crates/turbine-core/src/request.rs` (`ErrorCode::{NotLeader, ReplicaFailed}`), `crates/turbine-api/tests/api.rs` (test)
Interfaces:

- `/turbine/v1/pressure` adds `devices[{device, state, budget{weights,kv,workspace,collective,runtime,reserve}}]`, `groups[{replica, state, limiting_device}]`, `replicas[{replica, eligible, reason}]`; `/turbine/v1/scheduler` = `{"<replica>": SchedulerSnapshot}`
  Covers: S-8; `api pressure_multi_device_view`
  Depends on: Tasks 14, 15

- [ ] Write failing test `api pressure_multi_device_view`: with a diagnostics fake built from two `DeviceBudget`s and one group, asserts `/turbine/v1/pressure` has `devices`, `groups` (each with `limiting_device`) and `replicas` arrays, and `/metrics` contains `turbine_device_budget_bytes{device="0",component="collective"}`. Run: `cargo test -p turbine-api --test api pressure_multi_device_view` — expect FAIL.
- [ ] Implement the document assembly and metric export (`component` label from `PoolKind`), the ready reasons and the `not_leader` 503 for inference routes on static workers.
- [ ] Run: `cargo test -p turbine-api --test api pressure_multi_device_view` — expect PASS.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(server): per-device, group and replica pressure diagnostics`

## Task 17: Tensor-parallel model execution and the TP lab run

Files: `crates/turbine-model/src/loader.rs` (`ShardSpec` argument: row/column/vocab slices read by positioned reads of the rank's byte ranges only), `crates/turbine-model/src/executor/llama.rs` and `executor/olmoe.rs` (`Option<TpContext>`: all-reduce after o_proj and down_proj, vocab-parallel embedding + all-reduce, LM head + all-gather to the leader, sharded QK-norm via `row_sumsq` → all-reduce → `rmsnorm_sharded`), `crates/turbine-model/src/lib.rs` (`TpContext`), `crates/turbine-server/src/engine/tp.rs` (`StepExecutor` over the executors; one `RankRuntime` per group; circuit `CollectiveFailed` on collective errors, in-flight requests end with `replica_failed`), `crates/turbine-model/tests/tiny_model.rs` (TP test), `scripts/lab/phase5-novanas-{llama,olmoe}.yaml` (tp 2 configs, `server.listen: 127.0.0.1:18000`)
Interfaces:

- `pub struct TpContext { pub rank: u32, pub world: u32, pub collective: Arc<dyn Collective>, pub stream: StreamRef }` (contract §10)
- `WeightLoader::load(index, slots, mem, staging_bytes, shard: Option<ShardSpec>) -> Result<LoadedWeights, ModelError>`; per-rank KV pools of the same block count indexed by the leader's logical `BlockId`s
  Covers: S-6, S-9; `tiny_model tp2_matches_tp1_on_host` (contract addition), lab `lab-cluster.sh tp2-novanas`
  Depends on: Tasks 6, 7, 11, 12, 13, 14; phase-1/phase-2 plans (executors, golden fixtures, `turbine-golden`)

- [ ] Write failing test `tiny_model tp2_matches_tp1_on_host`: the tiny Llama (4 heads / 2 KV heads) and tiny OLMoE checkpoints run on the `cpu-reference` provider with two ranks over `HostCollective` produce greedy tokens exactly identical to tp 1 for 16 steps, logits bitwise equal on every rank, and each row's top-5 logprobs (of tp 1) within the strict golden bounds of `tests/golden/llama-3.2-3b-instruct/tolerance.json` (user decision 2026-09-28: the golden check replaces the former 1e-4 raw-logit bound, since each all-reduce rounds BF16 partial sums and moves logits ~1 % relative; `tp_layer0_matches_one_device_slices` pins the sharding bitwise). Run: `cargo test -p turbine-model --test tiny_model tp2_matches_tp1_on_host` — expect FAIL.
- [ ] Implement sharded loading and the collective insertion points listed above; the leader samples, workers only execute; cancellation drops a sequence from the next `StepPlan` and the leader's block manager frees its blocks on every rank; out-of-memory while loading a shard exits 1 naming rank, device, tensor and the budget breakdown.
- [ ] Run: `cargo test -p turbine-model --test tiny_model tp2_matches_tp1_on_host` — expect PASS.
- [ ] Lab (ASK THE USER FIRST: both R9700 must be free): `scripts/lab-cluster.sh tp2-novanas` — runs Llama-3.2-3B-Instruct at tp 2 in `local` mode, waits for `/ready` 200, `turbine-golden compare --url http://127.0.0.1:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` at `--concurrency 1` (strict bounds) and `--concurrency 16` (batched bounds); repeats in `static` mode (ranks 0 and 1, leader 127.0.0.1:18100); repeats `local` for OLMoE against `tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl`; then `turbine-bench --url http://127.0.0.1:18000 --concurrency 4 --requests 64 --output json` requiring `requests_ok: 64` — expect final log line `lab-cluster: tp2-novanas PASS`; paste compare outputs and TTFT/ITL beside the tp 1 run into the task evidence.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(model): tensor-parallel Llama and OLMoE execution over the collective trait`

As built (2026-09-28, branch p5-t17-server a1f19e2): the static-mode rank protocol is v2 (`StepPlan.copies` carries `copy_blocks` forks); `RankRuntime::wait_idle` lets the leader touch worker contexts (KV calibration, tier copies) only while every worker is idle; the cpu backend's TP path needs an explicit `parallel.collective_backend: host`; collective errors open the circuit with reason `collective_failed` and, until Task 28, exit 3 (user decision "P5: collective failure recovery"); a stopped worker is the server fatal `Fatal::RankStopped` (server fatal); rank pools agree on the minimum block count; `collective_bytes` is budgeted per device; decode graphs and overlapped launches are off under TP with a WARN and reason code; an L1 share below one 1 GiB slab per rank logs `kv_l1_share_below_slab`; `/ready` reports `collective_init` / `loading_weights` / `rank_missing`; dp × tp works. First tp2-novanas (2-GPU, GPU0 Gen5 x8 + GPU1 Gen4 x8): Llama local and static golden 16/16 at c1 and c16, OLMoE 15/16 at c1 and 14/16 at c16 (p10, p14 diverge; margin pursued in Tasks 29/32); standard workload tp1 856.4 tok/s (TTFT 208 ms, ITL 15.42 ms) vs tp2 854.0 tok/s (TTFT 453 ms, ITL 12.88 ms).

## Task 18: Data-parallel replicas and the DP lab run

Files: `crates/turbine-server/src/engine/replicas.rs` (one engine thread, scheduler, KV pool, KV orchestrator and pressure controller per replica; router in front; shared-device budgets when `allow_device_sharing`), `crates/turbine-server/src/startup.rs` (build replicas from `ParallelPlan.groups`), `crates/turbine-server/tests/tiny_server.rs` (DP test on the CPU backend), `scripts/lab/phase5-novanas-dp2.yaml`
Interfaces:

- `pub struct Replicas` with `pub fn submit(&self, req: GenerationRequest) -> Result<(ReplicaId, mpsc::Receiver<GenerationEvent>), ApiError>`; consumes `router::route`, `RouterMetrics`, `SharedDeviceBudget`
  Covers: S-7, S-9; `tiny_server dp2_routes_to_both_replicas` (contract addition), lab `lab-cluster.sh dp2-novanas`
  Depends on: Tasks 15, 16, 17

- [ ] Write failing test `tiny_server dp2_routes_to_both_replicas`: the tiny model with `execution.backend: cpu`, `data_parallel_size: 2`, `allow_device_sharing: true` serves 32 concurrent requests, all succeed, `/metrics` shows non-zero `turbine_dp_routed_total` for replica 0 and replica 1, and `/turbine/v1/scheduler` has keys `"0"` and `"1"`. Run: `cargo test -p turbine-server --test tiny_server dp2_routes_to_both_replicas` — expect FAIL.
- [ ] Implement the replica set: each replica is one TP group with its own engine; the router reads each replica's `ControllerHandle::state()`, circuit and outstanding tokens; a replica whose communicator fails goes `CIRCUIT_OPEN` while the others serve.
- [ ] Run: `cargo test -p turbine-server --test tiny_server dp2_routes_to_both_replicas` — expect PASS.
- [ ] Lab (ASK THE USER FIRST: both R9700 must be free): `scripts/lab-cluster.sh dp2-novanas` — serves Llama-3.2-3B-Instruct with tp 1, dp 2, runs `turbine-bench --url http://127.0.0.1:18000 --concurrency 8 --requests 128 --output json` requiring `requests_ok: 128` and non-zero `turbine_dp_routed_total` for both replicas — expect final log line `lab-cluster: dp2-novanas PASS`; record throughput beside a dp 1 run in the task evidence.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(server): data-parallel replicas behind one API`

## Task 19: Topology on the real hosts

Files: `crates/turbine-device/tests/lab.rs` (`topology_matches_host`, ignored)
Interfaces:

- consumes `discover_topology(Path::new("/sys"), &inventory, &mut vendor)`, `TURBINE_EXPECT_AMD`, `TURBINE_EXPECT_NVIDIA`
  Covers: S-1, S-9; lab `cargo test -p turbine-device --test lab topology_matches_host -- --ignored` under `scripts/lab-test.sh novanas` and `scripts/lab-test.sh dgx-spark`
  Depends on: Task 2; phase-0 plan (`scripts/lab-test.sh`, `lab.rs`)

- [ ] Write failing test `lab topology_matches_host` (`#[ignore]`): read-only discovery (no device memory allocated); with `TURBINE_EXPECT_AMD=2` asserts two `gfx1201` GPU vertices and a PCIe GPU↔GPU edge with `p2p: disabled`; with `TURBINE_EXPECT_NVIDIA=1` asserts one GPU and a NIC with an RDMA device at 200 Gb/s; prints `topology_matches_host vertices=<n> edges=<m>`. Run: `cargo test -p turbine-device --test lab topology_matches_host -- --ignored` on macOS — expect FAIL (no devices, expectation variables unset → assertion on the empty inventory).
- [ ] Implement only what the real captures reveal as misparsed (the fixture parsers from Task 2 are the implementation); add any new real-host line to the Task 2 capture files so the unit tests keep covering it.
- [ ] Run: `scripts/lab-test.sh novanas` (ASK THE USER FIRST: the Job requests one R9700 even though discovery is read-only) and `scripts/lab-test.sh dgx-spark` (correctness run: proceeds after the script's MemAvailable precondition) — expect PASS with log line `test topology_matches_host ... ok` on both.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(device): topology discovery matches novanas and dgx-spark`

## Task 20: Host-link probe and measured edge costs

Files: `crates/turbine-device/src/topology/{mod.rs,probe.rs}` (probe trait and edge attributes), `crates/turbine-server/src/startup.rs` (run the probe through the kernel library's v2.5 copy streams after discovery), `crates/turbine-distributed/src/plan.rs` (device ordering by measured cost), `crates/turbine-core/src/config/parallel.rs` (`parallel.topology.measure_links`), `crates/turbine-kernels/src/link_probe.rs` (the probe over a context's pinned memory and copy stream), `crates/turbine-kernels/tests/lab.rs`
Interfaces:

- `pub trait LinkProbe { fn host_link(&mut self, device: DeviceId, bytes: u64) -> Result<LinkBandwidth, String>; }`, `pub struct LinkBandwidth { pub h2d_gbps: f64, pub d2h_gbps: f64 }`, `pub fn apply_link_probe(graph: &mut TopologyGraph, results: &BTreeMap<DeviceId, Result<LinkBandwidth, String>>)`; edge fields `measured_h2d_gbps`, `measured_d2h_gbps`, `cost_gbps: Option<f64>`, `TopologyGraph::host_link_gbps(device)`, `probe_links(graph, probe)`, `turbine_kernels::link_probe::{CopyLinkProbe, ProbeTarget, measure_host_link}`; metric `turbine_topology_link_gbps{device,direction}`
  Covers: S-13; `topology::tests::measured_links_on_edges`, `link_probe::tests::host_probe_measures_both_directions`, lab `turbine-kernels --test lab host_link_probe`
  Depends on: Tasks 2, 10, 19

- [ ] Write failing test `topology::tests::measured_links_on_edges` (the S-13 criterion). Run: `cargo test -p turbine-device topology::tests::measured_links_on_edges` — expect FAIL.
- [ ] Implement the probe (≤ 64 MiB per direction per GPU, ≤ 2 s total, pinned host memory, median of 3), the edge attributes and the planner's use of them (TP/EP groups costed by the slowest member; PP placement input for Task 22).
- [ ] Lab `host_link_probe` in `crates/turbine-kernels/tests/lab.rs` (the probe `turbine_kernels::link_probe::CopyLinkProbe` over each device's context): both GPUs > 1 GB/s each way, GPU0 h2d ≥ 0.9 × GPU1 h2d with `TURBINE_EXPECT_AMD=2`. Run: `scripts/bench-lock.sh scripts/lab-test.sh novanas --gpus 2 -- -p turbine-kernels --test lab host_link_probe` — expect PASS.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(device,distributed): measured host-link bandwidth on topology edges and in the planner`

## Task 21: `hostmem` collective backend

Files: `kernels/include/turbine_kernels.h` and `kernels/rocm/src/hostmem.hip` (ABI minor group v2.7: mapped host allocation, one-shot all-reduce / all-gather / reduce-scatter / send / recv kernels), `crates/turbine-kernels/src/{ffi.rs,shim.rs}` (bindings, host stub), `crates/turbine-distributed/src/collective/{hostmem.rs,mod.rs,conformance.rs,nccl_api.rs,ffi.rs}` (registry entry, `send` / `recv` on the trait, `ncclSend` / `ncclRecv`), `crates/turbine-distributed/src/bin/turbine-collbench.rs`, `scripts/lab-cluster.sh` (`collbench-hostmem-novanas`), `docs/extending/collective-backend.md`, `docs/extending/kernel-implementation.md`
Interfaces:

- `Collective::send(&self, buf, peer, stream)`, `Collective::recv(&self, buf, peer, stream)`; registry `COLLECTIVE_BACKENDS` = `host`, `rccl`, `nccl`, `hostmem`; config `parallel.collective.hostmem_max_bytes`; metric `turbine_collective_route_total{op,backend,reason}`; kernel ABI `TURBINE_ABI_MINOR` = 7 (optional group)
  Covers: S-14, S-2 (send / recv); `collective::hostmem::tests::matches_host_backend`, lab `collbench-hostmem-novanas`
  Depends on: Tasks 4, 5, 6, 7, 8; user decision 2026-09-28 "P5: small-message all-reduce latency on novanas — host-memory all-reduce?"

- [ ] Write failing test `collective::hostmem::tests::matches_host_backend` (the S-14 criterion) over the host stub of the v2.7 group. Run: `cargo test -p turbine-distributed collective::hostmem::tests::matches_host_backend` — expect FAIL.
- [ ] Implement the v2.7 group in the ROCm shim, the bindings, the backend (fixed rank-order FP32 reduction, sequence-numbered flags with system-scope release/acquire, double-buffered slots, bounded spin tied to the op timeout and `step_begin` / `step_end`, abort releases a spinning peer, RCCL above the threshold with a reason code), `send` / `recv` on every backend, and the conformance cases.
- [ ] Lab: `scripts/lab-test.sh novanas --gpus 2 --tier quick -- -p turbine-distributed` (bit-exact on both ranks, missing peer times out), then `scripts/bench-lock.sh scripts/lab-cluster.sh collbench-hostmem-novanas`; upload the rccl vs hostmem latency table to labbook (set `phase-5-multi-gpu`, type `turbine-collbench`), labelled "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)".
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(kernels,distributed): hostmem collective backend over mapped pinned host memory (ABI v2.7)`

## Task 22: Pipeline partitioner, stage placement and the plan's parallel modes

Files: `crates/turbine-distributed/src/{pipeline.rs,plan.rs}`, `crates/turbine-core/src/config/parallel.rs` (`pipeline_parallel_size`, `pipeline.layer_split`, `pipeline.micro_batches`, `expert_parallel_size`, `expert.placement`), `crates/turbine-server/src/parallel.rs` (`check_executable`), `crates/turbine-server/tests/server_cli.rs`
Interfaces:

- `pub struct StageSpec { pub stage: u32, pub device: DeviceId, pub layers: Range<u32>, pub embedding: bool, pub lm_head: bool }`, `pub struct LayerCost { pub weight_bytes: u64, pub flops_per_token: u64 }`, `pub struct PipelineCosts { pub layers: Vec<LayerCost>, pub first: LayerCost, pub last: LayerCost }`, `pub fn partition(costs: &PipelineCosts, stages: u32, split: Option<&[u32]>) -> Result<Vec<Range<u32>>, PlanError>`, `pub fn validate_split(split: &[u32], layers: u32, stages: u32) -> Result<(), PlanError>`, `pub fn place_stages(ranges: &[Range<u32>], devices: &[DeviceId], host_link_gbps: &dyn Fn(DeviceId) -> Option<f64>) -> Result<(Vec<StageSpec>, StageReason), PlanError>` (`StageReason` ∈ `pp_stage_host_traffic:<device>`, `pp_stage_device_order`); `ParallelPlan` gains `pp`, `ep`, `stages`, `experts`; reason codes of the spec's Data section
  Covers: S-10, S-12; `pipeline::tests::{partition_balances_cost, explicit_split_validated, last_stage_on_fastest_link}`, `plan::tests::parallel_modes`, `server_cli unsupported_parallel_combination_exits_2`
  Depends on: Tasks 1, 10, 11, 20

- [ ] Write failing tests `pipeline::tests::partition_balances_cost`, `explicit_split_validated`, `last_stage_on_fastest_link`, `plan::tests::parallel_modes`, `server_cli unsupported_parallel_combination_exits_2` (the spec criteria). Run each — expect FAIL.
- [ ] Implement the config keys and their validation, the cost-balanced partition, stage placement by measured host-link cost (the last stage on the fastest link), and the combination rules of S-12 refused before bind.
- [ ] Run the five tests — expect PASS.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(distributed,core): pipeline partitioner, stage placement and the parallel-mode rules`

## Task 23: Pipeline stages in `turbine-model`

Files: `crates/turbine-model/src/pp.rs` (stage config, weight slots of a layer range, stage executor with `send` / `recv` of the hidden state), `crates/turbine-model/src/executor/decoder.rs` (layer range, optional embedding / LM head), `crates/turbine-model/tests/tiny_model.rs`
Interfaces:

- `pub struct PpContext { pub stage: u32, pub stages: u32, pub layers: Range<u32>, pub collective: Arc<dyn Collective>, pub stream: StreamRef }`; `pp::weight_slots(cfg, &StageSpec)`, `pp::kv_layout(cfg, layers, block_tokens)`, `pp::build_executor(...)`; a non-last stage's `forward` returns no logits (its output is sent to the next stage)
  Covers: S-10; `tiny_model pp2_matches_pp1_on_host`
  Depends on: Tasks 17, 21 (`send` / `recv`), 22

- [ ] Write failing test `tiny_model pp2_matches_pp1_on_host` (the S-10 criterion; bitwise equal logits). Run: `cargo test -p turbine-model --test tiny_model pp2_matches_pp1_on_host` — expect FAIL.
- [ ] Implement stage loading (only the stage's layers; tied embeddings on the first and last stage), the stage executor and the hidden-state hand-off.
- [ ] Run the test — expect PASS; `scripts/lab-test.sh novanas --tier quick -- -p turbine-model --test tiny_model` for the HIP variant `hip_pp2_matches_pp1`.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(model): pipeline stages over the collective's point-to-point pair`

As built (2026-09-28, branch p5-pp-model 122253a): `turbine_model::pp` — `PpContext { stage, stages, layers, collective, stream }`, `check_stage`, `check`, `stage_config`, `kv_layout(cfg, layers, block_tokens)`, `weight_slots(cfg, &StageSpec)`, `requirements`, `available_requirements`, `workspace_bytes`, `build_executor`; the hidden state crosses stages through the merged `Collective::send` / `recv`; a stage's `ModelShape` and KV layout cover only its layers; only the last stage returns logits (a non-last stage's forward returns empty `Logits`); every stage receives the same `BatchInput` and runs `copy_blocks` on its own pool; decode graphs and overlapped launches are off under PP (WARN, reason `pipeline_parallel`). Tests `tiny_model pp2_matches_pp1_on_host` (last stage bitwise one device; the tiny checkpoints have 2 layers, so only 2 stages) and lab `hip_pp2_matches_pp1` (tiny_model quick tier 31/31). Loader: stages and EP ranks load through `WeightLoader::load_part(format, index, slots, whole, …)`, which skips tensors of other stages / ranks quietly (`LoadedWeights.elsewhere`, one debug line) instead of warning `unexpected_tensor` per tensor. Engine half (this task): one thread per stage, stages run concurrently (`send` may block until the peer receives).

## Task 24: Micro-batched pipeline in the engine and the PP lab run

Files: `crates/turbine-scheduler/src/pipeline.rs` and `crates/turbine-scheduler/tests/sim.rs` (micro-batch assignment in the deterministic simulator), `crates/turbine-server/src/engine/{pp.rs,loop.rs}` (stage threads, bounded in-flight micro-batches, per-stage KV pools), `crates/turbine-server/src/{startup.rs,kv_orchestrator.rs}` (per-stage KV shards of unequal size), `scripts/lab/phase5-novanas-pp2.yaml`, `scripts/lab-cluster.sh` (`pp2-novanas`)
Interfaces:

- `pub struct MicroBatchPlan { pub micro_batch: u32, pub seqs: Vec<SeqId> }`; metrics `turbine_pipeline_stage_duration_seconds{stage}`, `turbine_pipeline_bubble_ratio`; `/turbine/v1/scheduler` `pipeline` section
  Covers: S-10, S-9; `sim pipeline_micro_batches_overlap`, lab `pp2-novanas`
  Depends on: Tasks 17, 22, 23; T17b (per-rank KV tier copies)

- [ ] Write failing test `sim pipeline_micro_batches_overlap` (the S-10 criterion). Run: `cargo test -p turbine-scheduler --test sim pipeline_micro_batches_overlap` — expect FAIL.
- [ ] Implement micro-batch assignment compatible with continuous batching and chunked prefill (a sequence's next step enters stage 0 only after its token was sampled), stage threads with bounded queues, cancellation freeing blocks on every stage, stage failure → `replica_failed` and `CIRCUIT_OPEN`, per-stage KV tier shards.
- [ ] Run the test and `cargo test -p turbine-server` — expect PASS.
- [ ] Lab: `scripts/bench-lock.sh scripts/lab-cluster.sh pp2-novanas` — golden c1 / c16, then the standard workload at c16 and c32 beside tp 2 and dp 2; upload to labbook (`turbine-parallel-serving`), labelled "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)".
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(scheduler,server): micro-batched pipeline parallelism and the pp2 lab run`

As built (2026-09-28, branch p5-pp-scheduler 68b1492, the scheduler half): `Scheduler::with_micro_batches(m)` keeps up to m plans in flight (`crates/turbine-scheduler/src/pipeline.rs`: `MicroBatchPlan`, `StageTimeline`, `PipelineMetrics`; `SchedulerSnapshot.pipeline`; `Simulation::run_pipelined` in `sim/pipeline.rs`; `tests/sim.rs` `pipeline_micro_batches_overlap`, `pipeline_cancel_in_flight`, `pipeline_m1_matches_serial`). A plan takes at most ceil(live / m) sequences, and max_batch_tokens / m tokens only with chunked prefill on; a cancellation lands after the in-flight plan completes, a preemption victim in flight is waited for, and a finish for a sequence of another in-flight plan takes effect when that plan completes; with one sequence and m = 2 the second plan is empty and completes at once; the sim reaches 2.0× with 2 micro-batches and bubble 0.0 when stage cost scales with the batch (a flat cost gives 1.0×). Pipelining is not combined with the Phase 2c `turn_overlap` path. The engine half (stage threads, per-stage KV, pp2-novanas) remains.

## Task 25: Expert placement and the EP provider evaluation

Files: `crates/turbine-distributed/src/expert.rs` (placement map, validation, per-rank token counts), `.procoder/ask/decisions.md` (provider evaluation: CK `moe_sorting` local-expert mask, vLLM `fused_moe` `expert_map`, llama.cpp MoE, the Phase 2 providers with a global→local map)
Interfaces:

- `pub struct ExpertPlacement { pub ranks: u32, pub layers: Vec<(u32, Vec<u32>)> }` (per MoE layer: its index and the rank of each expert), `ExpertPlacement::contiguous(num_experts: u32, moe_layers: &[u32], ep: u32) -> Result<Self, PlanError>`, `ExpertPlacement::parse(text, num_experts, moe_layers, ep)` / `from_file(path, num_experts, moe_layers, ep)`, `fn layer(&self, layer) -> Option<&[u32]>`, `fn local_experts(&self, layer, rank) -> Vec<u32>`
  Covers: S-11; `expert::tests::placement_map_valid`
  Depends on: Task 22

- [ ] Write failing test `expert::tests::placement_map_valid` (the S-11 criterion). Run: `cargo test -p turbine-distributed expert::tests::placement_map_valid` — expect FAIL.
- [ ] Implement the placement map; measure the candidate providers for the expert-subset GEMMs on one R9700 (`scripts/lab-test.sh novanas --tier perf` under bench-lock) and record the evaluation with a recommendation.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(distributed): expert placement map; docs(decisions): EP provider evaluation`

## Task 26: Expert-parallel OLMoE in `turbine-model`

Files: `crates/turbine-model/src/ep.rs` (weight slots of the rank's experts, `EpContext`), `crates/turbine-model/src/executor/decoder.rs` (MoE layer: local expert selection, combine all-reduce in fixed rank order), `crates/turbine-model/tests/tiny_model.rs`
Interfaces:

- `pub struct EpContext { pub rank: u32, pub world: u32, pub placement: Arc<ExpertPlacement>, pub collective: Arc<dyn Collective>, pub stream: StreamRef }`; composes with `TpContext` when tp = ep
  Covers: S-11; `tiny_model ep2_matches_ep1_on_host`
  Depends on: Tasks 17, 25

- [ ] Write failing test `tiny_model ep2_matches_ep1_on_host` (the S-11 criterion). Run: `cargo test -p turbine-model --test tiny_model ep2_matches_ep1_on_host` — expect FAIL.
- [ ] Implement expert-sharded loading, the local-expert MoE path over the provider chosen in Task 25, the FP32 combine all-reduce, and per-rank / per-expert token counts.
- [ ] Run the test — expect PASS; `scripts/lab-test.sh novanas --tier quick -- -p turbine-model --test tiny_model` for `hip_ep2_matches_ep1`.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(model): expert-parallel OLMoE over the collective trait`

As built (2026-09-28, branch p5-ep-model 95f4e7f): `turbine_model::ep` (`EpContext`, `EpShard`, `EpAttention::{Replicated, TensorParallel}`, `check`, `moe_layers`, `rank_config`, `kv_layout`, `weight_slots`, `available_requirements`, `workspace_bytes`, `build_executor`, `ExpertTokenCounts`) runs each rank's experts through the `moe_experts` local expert range the kernel ABI v2 already has (no ABI change; decision "P5 Task 25: EP expert-subset GEMMs"); the combine is one all-reduce at tp 1 and folds into the TP FFN all-reduce at tp = ep; `hip_ep2_matches_ep1` runs tiny-model seed 1 (seed 7 has a 0.0006 near-tie TP attention flips — the golden margin excuse); `moe_ep_local_timings` is a slow perf test in `scripts/lab-test.sh`.

## Task 27: EP serving and the EP lab run

Files: `crates/turbine-server/src/{startup.rs,parallel.rs,engine/tp.rs}` (EP groups reuse the TP group runtime), `crates/turbine-api` (the `expert` section of `/turbine/v1/scheduler`), `scripts/lab/phase5-novanas-ep2.yaml`, `scripts/lab-cluster.sh` (`ep2-novanas`)
Interfaces:

- metrics `turbine_expert_rank_tokens_total{rank}`, `turbine_expert_imbalance_ratio`; `/turbine/v1/status` `parallel.ep` and `experts`
  Covers: S-11, S-9; lab `ep2-novanas`
  Depends on: Tasks 17, 26

- [ ] Write failing test `tiny_server ep2_serves_like_ep1` on the cpu backend (greedy completion tokens identical to ep 1, the `expert` section present). Run: `cargo test -p turbine-server --test tiny_server ep2_serves_like_ep1` — expect FAIL.
- [ ] Implement EP group start-up (per-rank expert shards, replicated or TP-sharded attention), diagnostics and metrics.
- [ ] Lab: `scripts/bench-lock.sh scripts/lab-cluster.sh ep2-novanas` — OLMoE golden c1 / c16, both ranks' token counts non-zero, the standard workload beside tp 2 and dp 2; upload to labbook, labelled "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)".
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(server): expert-parallel serving and the ep2 lab run`

As built (2026-09-28, branch p5-ep-server 3d1b8d8, Task 27): `ParallelPlan` gains `ep`, `experts` and `group_size()` = max(tp, ep); reasons `configured`, `ep_moe_only`, `backend:<name>`; `plan::expert_size` / `plan::expert_placement`; ep × dp with tp ∈ {1, ep}; refused before bind: ep on a dense model, ep not dividing the experts, other tp (`combination_unsupported:ep+tp`), too few devices, a bad placement file; `ep: auto` resolves to 1 (capacity is left to `tp: auto`). `check_executable(&mut ParallelPlan)` rebuilds the placement from the model's MoE layers and checks every rank; EP groups start on the TP group runtime (`RankPart` / `EpRank`, `PreparedModel.expert`), load through `WeightLoader::load_part`, with decode graphs and overlap off (reason `expert_parallel`); KV tier blocks hold every rank's copy. `/turbine/v1/status` `parallel` shows `pp`, `ep` and each rank's experts; the scheduler document's `expert` section is injected in `backend.rs` (not `turbine-api`); metrics `turbine_expert_rank_tokens_total{rank}`, `turbine_expert_imbalance_ratio`. Tests `plan::tests::expert_parallel_plans`, `parallel::tests::expert_parallel_checked_against_the_model`, `parallel::tests::expert_stats_feed_metrics_and_document`, `tiny_server ep2_serves_like_ep1` (tp 1 and tp 2), `server_cli ep_on_dense_model_exits_2`. ep2-novanas (2-GPU, GPU0 Gen5 x8 + GPU1 Gen4 x8): standard workload c16 ep1 611.8 tok/s (TTFT 118 ms, ITL 24.4 ms), ep2 rccl 802.3 (141 ms, 17.8 ms), ep2 hostmem 812.9 (138 ms, 17.6 ms); both ranks' expert counts non-zero; golden against the committed reference fails p14's likely bound (1.0246 > 1.01) while ep2 matches an ep1 capture 16/16 strict — open question "P5: OLMoE golden tolerance under expert parallelism". The scenario runs to the end and reports both comparisons.

## Task 28: Re-create a failed communicator while probing

Files: `crates/turbine-server/src/engine/tp.rs` (swap every rank executor's collective, re-init step plan), `crates/turbine-server/src/reliability.rs` (probing hook), `crates/turbine-distributed/src/rank.rs` (a `Reinit` step message, protocol v3), `crates/turbine-model/src/tp.rs` / `ep.rs` (`set_collective` on the executor), `crates/turbine-server/tests/tiny_server.rs`
Interfaces:

- `ModelExecutor`-side `fn set_collective(&mut self, c: Arc<dyn Collective>)` for TP/EP executors; `RankMessage::Reinit { unique_id: [u8; 128] }`; circuit reason `collective_failed` recovers through `PROBING` instead of exiting 3
  Covers: S-6 failure mode "Collective op times out"; user decision "P5: collective failure recovery" (B; C until this task lands)
  Depends on: Tasks 12, 17

- [ ] Write failing test `tiny_server tp2_collective_failure_recovers`: a tp 2 server on the cpu backend whose host collective fails one all-reduce answers the in-flight request with `replica_failed`, `/ready` 503 `circuit_open`, then, after the probe re-creates the communicator, 200 and a greedy completion identical to before. Run: `cargo test -p turbine-server --test tiny_server tp2_collective_failure_recovers` — expect FAIL.
- [ ] Implement re-creation (fresh unique id, bounded init on every rank concurrently, the re-init step plan in `local` and `static` mode) and drop the interim exit 3 for collective failures (a sticky device error still exits 3).
- [ ] Lab: `scripts/lab-test.sh novanas --gpus 2 --features fault-injection -- -p turbine-server --test fault` gains `tp2_collective_failure_recovers_on_gpu` (hard timeout 10 min).
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(server,distributed): re-create a failed communicator while probing`

As built (2026-09-28, Task 28 69b9011 + 074c5df, merged bb8a876 / fece955): a failed collective no longer ends a worker — local worker threads keep running, static workers send `StepFailed` and wait; the leader marks the group broken and its next step (the circuit's probe) re-creates the communicator on every rank with a fresh unique id (`Reinit` / `ReinitDone`, bounded by the init timeout); a failed re-creation fails the probe and the next probe retries; a sticky device error still exits 3. There was no interim exit 3 to remove (a static worker used to exit 1). `set_collective` is a default `ModelExecutor` method implemented in `executor/decoder/mod.rs` (TP and EP contexts swapped; PP not); no probing hook in `reliability.rs`. Test hooks `TURBINE_TEST_HOST_COLLECTIVE_FAIL_FILE` (host collective) and `TURBINE_FAULT_TP_ABORT_FILE` (fault-injection). Tests `tiny_server tp2_collective_failure_recovers`, `engine::tp::tests::static_group_recovers_from_a_failed_step`, `rank::tests::failed_group_is_recreated`; lab `fault tp2_collective_failure_recovers_on_gpu` passed on 2-GPU (run 0928093026-12a37124, labbook cb9e5e2b). Tests poll `/ready` for `circuit_open` (the error event reaches the client just before the circuit opens).

As built (Task 33 f1aa910, merged bb61aaf): the worker's budget travels in a new `RankMessage::Loaded` after it loads (not in `Hello`, sent before loading); every step plan carries the mirror's ledger changes, which the worker replays before the step; every 64 steps and in the last plan before shutdown the plan carries the leader's digest, compared on the worker. The test is `engine::tp::tests::static_mirror_matches_worker_ledger` (a two-process tiny_server test is not possible: the host collective joins threads of one process and static mode is refused on the cpu backend); worst-case admission queued 7 admissions for KV instead of preempting, so the test shows 0 preemptions. Protocol v3 also carries Task 30's `TierCopy` / `TierAck` / `TierReady` after `ReinitDone`.

## Task 29: Batch-invariant GEMM rows for the tensor-parallel shapes

Main open accuracy item for the Phase 5 exit (coordinator, 2026-09-28): this task must fix or explain OLMoE tp 2 p10 — greedy divergence at token 21 with margin 1.3 (not a near-tie), likely |Δ| 1.36 / 1.33 against the one-GPU capture (ep 2 × tp 2) and 0.98–1.17 against the transformers reference. Until it lands, `ep2-novanas` reports its ep 2 × tp 2 leg as `known_fail task29` (not deciding the verdict).

Files: `kernels/rocm/tuning/gfx1201/gemm.tsv`, `crates/turbine-kernels/tests/hip_ops.rs` (the invariance check over the new rows), `crates/turbine-server/tests/kv_gpu.rs` (prefix reuse at tp 2)
Interfaces:

- new table rows for the tp 2 per-rank shapes (Llama qkv 2560×3072, o 3072×1536, gate/up and down halves, lm_head 64128×3072; OLMoE qkv 3072×2048, o 2048×1024, lm_head 25152×2048), each with one algorithm for every m (batch-invariant)
  Covers: user decision "P5: bit-exact prefix reuse under tensor parallelism" (B)
  Depends on: Task 17

- [ ] Write failing lab test `kv_gpu prefix_reuse_matches_cold_tp2` (the Phase 4 prefix-reuse check on a tp 2 group, bit-exact against cold). Run: `scripts/lab-test.sh novanas --gpus 2 -- -p turbine-server --test kv_gpu` — expect FAIL.
- [ ] Tune the rows with the existing tuning procedure (decision "Pre-Phase-5 #1 follow-up (A)") on GPU 0 under `scripts/bench-lock.sh`; add them; extend the table invariance test.
- [ ] Run the lab test — expect PASS; then `scripts/bench-lock.sh scripts/lab-cluster.sh tp2-novanas` and compare tok/s, TTFT and ITL with the previous tp2 run (labbook).
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `perf(kernels): batch-invariant GEMM rows for the tensor-parallel shapes`

## Task 30: KV tiers in static rank mode

Files: `crates/turbine-distributed/src/rank.rs` (tier messages), `crates/turbine-server/src/{kv_orchestrator.rs,engine/tp.rs,startup.rs}` (worker-side per-rank L1/L2, leader-driven), `scripts/lab-cluster.sh` (the tp2 static leg runs the Phase 4 multi-turn check)
Interfaces:

- `RankMessage::{TierCopy { copies: Vec<TierCopy> }, TierAck { … }}` carrying promote / demote / tier copies of the leader's block ids; a worker's `KvShard` is its own process's; tier state symmetric with `local` mode
  Covers: user decision "P5: KV tiers in static rank mode" (B); S-6 KV
  Depends on: Tasks 12, 17; T17b

- [ ] Write failing test `kv_orchestrator::static_workers_round_trip_through_l1_and_l2`: two ranks in separate threads linked only by the rank transport (loopback `tcp`), leader-driven demote to L1 and L2 and promote back, each rank's bytes in its own pool. Run: `cargo test -p turbine-server --bin turbine-server kv_orchestrator::static_workers_round_trip_through_l1_and_l2` — expect FAIL.
- [ ] Implement the messages, the worker side and the leader's wait for acks (bounded; a lost worker is `RankStopped`).
- [ ] Lab: the static leg of `scripts/lab-cluster.sh tp2-novanas` adds `turbine-bench --profile multi-turn` with `cached_tokens_ratio` > 0 and golden c1.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(server,distributed): per-rank KV tiers in static rank mode`

## Task 31: Group KV reservation on every rank's ledger

Files: `crates/turbine-reliability/src/multi_device.rs` (`reserve_group` across ledgers), `crates/turbine-server/src/{engine/mod.rs,reliability.rs}` (admission through the group), `crates/turbine-scheduler/tests/overload_sim.rs`
Interfaces:

- admission reserves each request's worst-case KV on every rank's ledger through `reserve_group`, atomically: a partial failure rolls back and queues with `PressureReason::KvReservation`
  Covers: S-8; user decision "P5: KV admission across tensor-parallel ranks" (B)
  Depends on: Tasks 14, 17

- [ ] Write failing test `overload_sim group_reservation_unequal_pools`: a 2-rank group whose rank 1 pool is smaller; under overload no request is admitted beyond rank 1's capacity, a partial reservation never leaks, and every admitted request completes. Run: `cargo test -p turbine-scheduler --test overload_sim group_reservation_unequal_pools` — expect FAIL.
- [ ] Implement.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(reliability,server): KV admission reserves on every rank's ledger`

As built (2026-09-28, branch p5-group-reservation 3d069c2): the group's reservation is rank 0's `Reservation` carrying the other ranks' as members (`Reservation::with_members` / `members`, `GroupReservation::into_reservation`), so commit, preemption, cancellation, SURVIVAL requeue, forks and pool payment reach every rank without scheduler changes; `multi_device::try_reserve_group` reserves rank by rank and rolls back on the first refusal (`reserve_group` wraps it); `Admission::with_group_ledgers` decides against the tightest rank (a request larger than the smallest pool is `context_exceeds_kv_capacity`); one pressure machine per group, fed by the worst rank's KV utilisation (`PressureController::with_group_ranks`), every rank reporting the group state; `/turbine/v1/pressure` lists every rank; sim `OverloadConfig.group_kv_blocks`. Tests: `overload_sim group_reservation_unequal_pools`, `admission::tests::group_admission_reserves_on_every_rank`, `multi_device::tests::group_reservation_commits_and_releases_every_rank`, `tiny_server tp2_admission_reserves_on_both_ranks`. `static` mode still admits on the leader's ledger (open question "P5: group KV admission in static rank mode").

## Task 32: Tensor-parallel performance

Also tracked here and in Task 29 (coordinator, 2026-09-28): OLMoE p10 at tp 2 and at ep 2 × tp 2 (identical prefix 21/32, margin 1.24 at the divergence, likely |Δ| 0.98–1.17 against the transformers reference) — under the multi-GPU gate it is judged against the one-GPU capture; check whether the Task 29 rows remove it.

Files: as the measurements direct (`crates/turbine-model/src/tp.rs`, `crates/turbine-distributed/src/collective/hostmem.rs` thresholds, decode graphs under TP), `crates/turbine-model/tests/perf.rs` (`tp_step_profile`)
Interfaces:

- a lab profile of one tp 2 prefill chunk and one decode step: collective time vs GEMM vs other, per rank
  Covers: S-6 performance after correctness (spec Constraints)
  Depends on: Tasks 17, 21, 29

- [ ] Profile (GPU 0 + GPU 1 under `scripts/bench-lock.sh`, hard timeout); record the split.
- [ ] Land fixes one at a time — hostmem route thresholds for prefill-size messages, the Task 29 rows, decode graphs under TP if feasible — each followed by `scripts/bench-lock.sh scripts/lab-cluster.sh tp2-novanas` (golden c1/c16 and the bench) and a labbook upload.
- [ ] Gate: `scripts/gate.sh` per commit.
- [ ] Commit: one `perf(...)` commit per fix.

## Task 33: Mirror ledgers for group admission in static rank mode

Files: `crates/turbine-distributed/src/rank.rs` (the budget in `Hello`, a ledger digest on the rank link, protocol v3), `crates/turbine-reliability/src/{ledger.rs,multi_device.rs}` (a mirror ledger built from a worker's budget; a ledger digest), `crates/turbine-server/src/{engine/tp.rs,startup.rs,model.rs}` (the leader builds one mirror per joined worker and admits through `reserve_group` over its own ledger and the mirrors; the worker applies the leader's reservations to its real ledger in step order and compares digests), `crates/turbine-server/tests/tiny_server.rs`
Interfaces:

- `Hello` gains the worker's KV budget (blocks, bytes per block, reserve); `RankMessage::StepPlan` carries, every _n_ steps, the leader's mirror digest for that rank (the ledger state after the plan's reservations are applied); the worker compares it with its real ledger's digest after the same step and logs `event="ledger_mirror_divergence"` (WARN, reason code `mirror_digest_mismatch`, metric `turbine_ledger_mirror_divergence_total{rank}`) on a mismatch
  Covers: user decisions "P5: KV admission across tensor-parallel ranks" (B) and "P5: group KV admission in static rank mode" (B); S-8
  Depends on: Tasks 12, 17, 31

- [ ] Write failing test `tiny_server tp2_static_mirror_matches_worker_ledger`: a tp 2 `static` group on the cpu backend over loopback `tcp` and the host collective, a mixed workload (concurrent streams, cancels mid-stream, preemption forced by a small pool) — after it, each mirror's reservations equal the worker's real ledger exactly, and no divergence was logged; a deliberately skewed worker ledger logs `mirror_digest_mismatch` once per check. Run: `cargo test -p turbine-server --test tiny_server tp2_static_mirror_matches_worker_ledger` — expect FAIL.
- [ ] Implement.
- [ ] Gate: `scripts/gate.sh`
- [ ] Commit: `feat(server,distributed,reliability): mirror ledgers for group admission in static rank mode`
