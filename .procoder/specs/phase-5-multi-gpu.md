# phase-5-multi-gpu

Status: complete

Source: `turbine-spec.md` §19 Phase 5 (multi-GPU), with §10 (multi-GPU and multi-node), §14 (observability), §17 (testing) and §21 (engineering rules). Sections of that document are cited as "TS §N". Decisions are recorded in `.procoder/ask/decisions.md` ("Answers log for phase 1–8 spec questions (2026-09-25)"); where they depart from TS this spec says "amends TS §N". This spec builds on phase-0-skeleton through phase-4-kv-intelligence.

## Problem

After Phase 4 Turbine serves one model on one GPU: one scheduler, one paged KV pool, one pressure controller. That caps model size and throughput at a single device and leaves TS §10's promise — composable DP and TP driven by a topology graph, never a flat GPU list — untouched. Phase 5 makes a Turbine process drive several GPUs: it discovers how devices, NUMA nodes, NICs and storage are connected, abstracts collectives behind a backend-neutral trait, splits one model across ranks with tensor parallelism, runs several replicas behind one API with data parallelism, and extends pressure accounting so a TP group is only as healthy as its worst device.

The lab's only multi-GPU host is novanas: 2× AMD Radeon AI PRO R9700 (gfx1201, 32 GB each, PCIe, peer-to-peer access reported **disabled** by `amd-smi topology`, measured 2026-09-25), ROCm 7.14.1 at `/opt/rocm/rocm`, jobs run as k3s Jobs. Because AMD execution already exists from Phase 1 (amends TS §19 Phase 8: AMD/ROCm moved to Phase 1), Phase 5 is built and proven on that pair with RCCL. The collective layer is one runtime-loaded binding to the NCCL C API, which RCCL implements; it serves RCCL now and NCCL on NVIDIA later without a second binding. NCCL hardware validation happens in phase-6-multi-node on the two DGX Sparks (amends TS §19 Phase 5 "NCCL abstraction": the abstraction is NCCL-API-shaped and proven first on RCCL). No plan may mix vendors until Phase 7.

## Users

- **Turbine developers (humans and AI agents):** need TP/DP logic that is testable on macOS without a GPU (a host-memory reference collective backend and deterministic fixtures), and one command per lab scenario that proves it on the R9700 pair.
- **Operators:** need to say `tensor_parallel_size: 2` or `devices: auto`, have Turbine refuse impossible plans (vendor-mixed plans, TP not dividing the head count, too few devices) before binding a port, and see in `/turbine/v1/topology`, `/turbine/v1/status` and logs why each device was grouped the way it was.
- **Benchmark runners:** need a collective bandwidth benchmark (all-reduce, all-gather, reduce-scatter, broadcast) comparable to `nccl-tests`/`rccl-tests`, and `turbine-bench` results for TP=1 vs TP=2 and DP=1 vs DP=2 on the same model (TS §18).
- **Phase 6 (multi-node) implementers:** need the node-local topology subgraph, the collective trait, the NCCL-API binding and the `static` rank bootstrap as the pieces they compose into a cluster.

## In scope

- [S-1] Node-local topology discovery in `turbine-device` (module `topology`): NUMA nodes with CPU lists, memory and distance matrix from sysfs; the PCIe tree (root ports, switches, endpoints, current link speed and width) from `/sys/bus/pci/devices`; GPU↔GPU link type, hop count and peer-to-peer access from amd-smi (`amdsmi_topo_get_link_type`, `amdsmi_is_P2P_accessible`) and NVML (`nvmlDeviceGetTopologyCommonAncestor`, `nvmlDeviceGetP2PStatus`, NVLink state); NICs from `/sys/class/net` and `/sys/class/infiniband` (netdev, RDMA device, link layer, rate, IPv4 addresses); NVMe controllers from `/sys/class/nvme`. The result is a typed graph (vertices + edges with bandwidth, latency, NUMA distance, P2P, RDMA and vendor-interconnect attributes, each value tagged with its source) captured at startup and exposed at `GET /turbine/v1/topology`.
- [S-2] New crate `crates/turbine-distributed` containing the `Collective` trait (all-reduce, all-gather, reduce-scatter, broadcast, barrier, abort), a `host` reference backend (ranks are threads, buffers are host memory, deterministic reduction order), and a `nccl_api` backend: one runtime-loaded (`libloading`) binding table for the NCCL C API that loads _librccl.so.1_ (AMD, reported as backend `rccl`) or _libnccl.so.2_ (NVIDIA, reported as backend `nccl`). The binding covers exactly `ncclGetVersion`, `ncclGetUniqueId`, `ncclCommInitRankConfig` (non-blocking), `ncclCommGetAsyncError`, `ncclCommAbort`, `ncclCommDestroy`, `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast`, `ncclGroupStart`, `ncclGroupEnd`, `ncclGetErrorString`. A watchdog polls `ncclCommGetAsyncError` and calls `ncclCommAbort` on timeout.
- [S-3] Collective benchmark binary `turbine-collbench` (in `turbine-distributed`, `src/bin`): sizes 8 B–1 GiB, reports algorithm and bus bandwidth with the `nccl-tests` formulas and checks results against the host backend. Device buffers and streams come from the phase-1 device layer (`DeviceBuffer`, `StreamRef` over _libturbine_hip.so_ on AMD, _libturbine_cuda.so_ on NVIDIA); `turbine-distributed` adds no HIP or CUDA runtime binding of its own.
- [S-4] Parallel plan: a `parallel` configuration section (see Interfaces), validation that fails before binding, and a planner that turns _parallel.devices_ `auto` into TP groups and DP replicas using the S-1 graph: every plan is single-vendor and TP groups are also single-architecture; `auto` picks the vendor with the most usable devices (tie → the vendor of the lowest device index) and lists the excluded devices; TP groups prefer the fastest link class (NVLink/xGMI > PIX > PXB > PHB > NODE > SYS); DP replicas take whatever remains. Every choice is logged with a reason code and returned in `GET /turbine/v1/status`.
- [S-5] Rank runtime in `turbine-distributed`: one leader rank (owns scheduler, admission, KV block manager and sampling) and worker ranks that execute `StepPlan`s in lockstep. Two modes: `local` (one thread per local device in one process — the default on novanas) and `static` (one process per rank, ranks listed in config, joined to the leader for unique-id exchange and per-step plan broadcast; the bootstrap Phase 6 reuses for cross-node groups). The static link goes through a registered rank transport (extension point `rank_transport`, trait `turbine_distributed::transport::Transport` with a static registry and a conformance suite, `docs/extending/rank-transport.md`; user decision 2026-09-28): Phase 5 registers `tcp`, selected by `parallel.ranks.transport`; addresses stay `SocketAddr` until the multi-node phase generalises them. Clean shutdown and abort propagate to every rank.
- [S-6] Tensor parallelism for the two phase-2 models, meta-llama/Llama-3.2-3B-Instruct (dense, 24 attention heads, 8 KV heads, tied embeddings) and OLMoE-1B-7B (BF16 MoE, 64 experts top-8, 16 heads = 16 KV heads, QK-norm over the full projection): each rank loads only its weight shard from safetensors; attention is split by heads (KV heads replicated when `tp > num_kv_heads` and `tp % num_kv_heads == 0`); a QK-norm whose weight spans all heads is computed from per-rank partial sums of squares combined by all-reduce; the o-projection and down-projections are row-parallel followed by all-reduce; gate/up projections and every routed (and, when present, shared) MoE expert are column-parallel along the intermediate dimension with the router replicated; the embedding is vocab-parallel with all-reduce and the LM head vocab-parallel with all-gather to the leader, tied embeddings sharing one vocab shard. Each rank holds the KV for its heads in a rank-local paged pool indexed by the leader's logical block ids. All weights and KV are BF16.
- [S-7] Data parallelism: _parallel.data_parallel_size_ replicas, each one TP group with its own scheduler, KV pools and pressure controller, behind one API. An in-process router picks the replica per request by prefix affinity (phase-4 prefix index), then least outstanding tokens, excluding replicas in ORANGE or worse unless all are. The routing policy is a registered module (extension point `dp_router_policy`, trait `turbine_distributed::router::RouterPolicy` with a static registry and a conformance suite, `docs/extending/dp-router-policy.md`; user decision 2026-09-28): `prefix_affinity` (the default) and `least_loaded`, selected by _parallel.router_. DP replicas may share one device only when _parallel.allow_device_sharing_ is true (testing; TP ranks never share a device).
- [S-8] Multi-GPU pressure accounting: the phase-3 per-device budget gains a `collective` component (communicator buffers measured as the drop in free device memory across communicator init) and is kept per device even when several replicas share it; a TP group's pressure state is the worst of its members' states; a DP replica's admission reserves KV blocks (phase-3 worst-case reservation) on every rank of its group atomically or not at all; `GET /turbine/v1/pressure` reports device, group and replica views; new metrics listed under Interfaces.
- [S-9] Lab scenarios in `scripts/lab-cluster.sh <scenario>` (this phase: `collbench-novanas`, `tp2-novanas`, `dp2-novanas`): each runs as one k3s Job on novanas requesting `amd.com/gpu: 2`, with weights from `/home/piwi/turbine-models/<slug>` mounted read-only and `TURBINE_TEST_MODEL_DIR` set, starts Turbine and its clients inside the Job on loopback, runs the scenario's checks and benchmarks, deletes only the Job it created, and exits with the scenario's result.

## Out of scope

- Worker discovery, membership, the general transport layer, the cluster-wide topology graph, request forwarding between nodes, the cluster KV directory, node-failure handling and NCCL hardware validation on the Sparks (Phase 6). The `static` rank mode is a fixed rank table for one TP group, not discovery.
- Pipeline parallelism, expert parallelism, prefill/decode disaggregation, RDMA/direct KV transfer and heterogeneous pools (Phase 7).
- Any plan mixing vendors (NVIDIA + AMD), in TP or DP — rejected at plan time until Phase 7.
- Tensor parallelism of quantized weights (Phase 8 quantization track); Phase 5 serves BF16 only.
- oneCCL / Intel devices.
- Changing TP or DP size at runtime, elastic scaling, live re-sharding. A plan is fixed for the process lifetime.
- Measured link bandwidth at startup: topology edges carry nominal and vendor-reported values; measured numbers come from `turbine-collbench` and are not fed back automatically.
- Sharing one set of weight buffers between DP replicas on the same device.
- Custom all-reduce kernels (one-shot/two-shot IPC all-reduce). Only after profiling proves RCCL/NCCL insufficient (TS §2, §21 rule 3).

## Constraints

- Rust only; no Python in the build or runtime path (TS §21 rule 4). RCCL and NCCL are loaded at runtime; nothing links against them at build time, so the workspace still builds and every non-ignored test passes on macOS arm64 with no GPU libraries.
- `unsafe` and FFI stay isolated (TS §21 rule 10): in `turbine-distributed` only the module `collective::ffi` may contain `unsafe`; the crate root has `#![deny(unsafe_code)]` and that module carries `#[allow(unsafe_code)]`, each block with a `// SAFETY:` comment naming the owner of every device pointer, stream and communicator it touches. Device buffers and streams come from the phase-1 device layer; `turbine-distributed` never allocates model memory. The phase-1 `unsafe_isolation` allowlist gains exactly that one module.
- New dependency, with its reason: `postcard` (compact serde framing for the static-mode rank protocol; no schema compiler). Local mode uses bounded `std::sync::mpsc::sync_channel`.
- Every queue is bounded (TS §21 rule 8): the leader→worker plan channel holds at most _parallel.plan_queue_depth_ plans (default 2); frames on the static-mode socket are capped at 16 MiB.
- No collective call may block forever: communicator init is bounded by `parallel.collective.init_timeout` and each step's collectives by `parallel.collective.op_timeout`; on expiry the communicator is aborted, the group enters the phase-3 circuit breaker as `CIRCUIT_OPEN`, and the reason is logged.
- Every automatic decision (device grouping, vendor choice, backend choice, replica routing, pressure state of a group) emits a structured log line with a reason code and a metric (TS §14, §21 rule 7).
- Correctness before speed (TS §21 rule 1): TP=2 must meet the phase-1 golden tolerance (≥ 14/16 prompts with the first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats against the committed HF transformers BF16 fixtures) before any TP performance work is accepted.
- Lab: Phase 5 hardware runs are on novanas only (k3s Job, `amd.com/gpu: 2`, `rust:1.97-trixie` image as in phase-0/phase-1 (CONFLICT C-7), hostPath `/opt/rocm/rocm` read-only; RCCL is `/opt/rocm/rocm/lib/librccl.so.1`). Ports are loopback inside the Job (HTTP 18000, static-mode leader 18100); no Service or host port is created. Read-only topology checks also run under `scripts/lab-test.sh dgx-spark`. Memory is capped by the phase-3 device budget.
- Before any lab run that needs production workloads moved, GPUs emptied or memory freed on any host, the implementer asks the user first and waits; scripts never evict, scale or stop other workloads, and a Job that cannot be scheduled (GPUs in use) makes the script exit non-zero naming that reason.
- With peer access disabled on novanas, RCCL uses host-staged transfers; the topology graph must report `p2p: disabled` rather than assume.

## Interfaces

### Configuration (`parallel` section; unknown keys are errors)

| Key                                | Type                                                                                  | Default           | Validation                                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _parallel.tensor_parallel_size_    | integer or `auto`                                                                     | `1`               | 1 ≤ n ≤ 8 and a power of two; must divide the model's attention heads, and either divide the KV heads or be a multiple of them; `auto` = smallest size whose shard fits a device budget |
| _parallel.data_parallel_size_      | integer or `auto`                                                                     | `1`               | 1 ≤ n ≤ 64; `auto` = ⌊usable devices of the chosen vendor / tp⌋                                                                                                                         |
| _parallel.devices_                 | `auto` or list of global device indices                                               | `auto`            | indices exist in the inventory and share one vendor; list length = tp × dp unless _parallel.allow_device_sharing_; no index twice within one TP group                                   |
| _parallel.collective_backend_      | `auto`, `nccl`, `rccl`, `host`                                                        | `auto`            | `auto` = `rccl` for AMD plans, `nccl` for NVIDIA, `host` only when tp = 1; `host` with real devices and tp > 1 is rejected; `rccl` with NVIDIA devices (or `nccl` with AMD) is rejected |
| _parallel.nccl_library_            | path or null                                                                          | null              | when set, failure to load is fatal (exit 1)                                                                                                                                             |
| _parallel.rccl_library_            | path or null                                                                          | null              | when set, failure to load is fatal (exit 1); default search tries `/opt/rocm/lib` then the loader path                                                                                  |
| _parallel.allow_device_sharing_    | bool                                                                                  | `false`           | only DP replicas may share; each sharing replica's budget is the device budget divided by the replicas on it                                                                            |
| _parallel.plan_queue_depth_        | integer                                                                               | `2`               | 1 ≤ n ≤ 16                                                                                                                                                                              |
| _parallel.router_                  | a registered DP router policy (`dp_router_policy`): `prefix_affinity`, `least_loaded` | `prefix_affinity` | a name of the registry (`Config::validate_modules`, exit 2 before binding)                                                                                                              |
| `parallel.collective.init_timeout` | duration                                                                              | `120s`            | 1 s ≤ value ≤ 30 m                                                                                                                                                                      |
| `parallel.collective.op_timeout`   | duration                                                                              | `30s`             | 100 ms ≤ value ≤ 10 m                                                                                                                                                                   |
| `parallel.ranks.mode`              | `local`, `static`                                                                     | `local`           | `static` requires tp > 1 and dp = 1                                                                                                                                                     |
| `parallel.ranks.rank`              | integer                                                                               | `0`               | `static` only; 0 ≤ rank < world size                                                                                                                                                    |
| `parallel.ranks.leader`            | socket address                                                                        | null              | required in `static` mode; rank 0 listens here, others connect                                                                                                                          |
| `parallel.ranks.local_devices`     | list of global device indices                                                         | `[0]`             | `static` only; exactly one device per rank process in Phase 5                                                                                                                           |
| `parallel.ranks.transport`         | a registered rank transport (`rank_transport`): `tcp`                                 | `tcp`             | a name of the registry (`Config::validate_modules`, exit 2 before binding); carries the `static` bootstrap and step plans                                                               |

Durations are `<integer><unit>` with unit `ms`, `s`, `m` or `h` (one parser for all keys, CONFLICT C-14), no space, case-sensitive. In `static` mode world size equals _parallel.tensor_parallel_size_. _distributed.enabled_ stays rejected (Phase 6).

### HTTP routes (changes)

| Route                       | Phase 5 response                                                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GET /turbine/v1/topology`  | `200` node-local topology graph (see Data)                                                                                                                                                 |
| `GET /turbine/v1/status`    | adds `"parallel": {"tp":<n>,"dp":<n>,"backend":"rccl","mode":"local","groups":[{"replica":0,"ranks":[{"rank":0,"device":0,"host":"novanas"}]}],"plan_reasons":["<code>"]}`                 |
| `GET /turbine/v1/pressure`  | adds `devices[]` (per-device budget by component and state), `groups[]` (state = worst member, `limiting_device`) and `replicas[]` (router eligibility)                                    |
| `GET /turbine/v1/scheduler` | one entry per replica, keyed by replica index                                                                                                                                              |
| `GET /ready`                | `200` only when every replica's communicator is initialised and its weights are loaded on every rank; otherwise `503` with `reason` `collective_init`, `loading_weights` or `rank_missing` |

Worker ranks in `static` mode (rank ≠ 0) serve only `/health`, `/ready` and `/metrics`; inference routes return `503` with code `not_leader`.

### Collective trait (`turbine-distributed::collective`)

```rust
pub enum ReduceOp { Sum, Max }
pub enum CollectiveBackendKind { Nccl, Rccl, Host }

pub trait Collective: Send + Sync {
    fn backend(&self) -> CollectiveBackendKind;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn all_reduce(&self, buf: &mut DeviceSlice, op: ReduceOp, stream: &StreamRef) -> Result<(), CollectiveError>;
    fn all_gather(&self, send: &DeviceSlice, recv: &mut DeviceSlice, stream: &StreamRef) -> Result<(), CollectiveError>;
    fn reduce_scatter(&self, send: &DeviceSlice, recv: &mut DeviceSlice, op: ReduceOp, stream: &StreamRef) -> Result<(), CollectiveError>;
    fn broadcast(&self, buf: &mut DeviceSlice, root: usize, stream: &StreamRef) -> Result<(), CollectiveError>;
    fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError>;
    fn abort(&self);
}
```

`DeviceSlice` and `StreamRef` are the phase-1 buffer and stream handles; the `host` backend implements them over host memory. `CollectiveError` variants: `Timeout { op, after }`, `RemoteAbort { rank }`, `Backend { code, message }`, `ShapeMismatch`, `Unavailable { library, detail }`. Supported dtypes: BF16 and FP32 (QK-norm partial sums and logits use FP32).

### Static-mode rank protocol (over the registered rank transport, `tcp` in Phase 5; `postcard` frames with a `u32` little-endian length prefix, max 16 MiB)

1. Worker → leader `Hello { protocol: 1, rank, world_size, model_fingerprint, config_fingerprint, device_vendor, device_arch }`.
2. Leader waits for all ranks (bounded by `parallel.collective.init_timeout`), checks fingerprints, vendor and arch are identical and ranks unique, then sends `Welcome { unique_id: [u8; 128] }` or `Reject { reason }`.
3. Per engine step: leader → workers `StepPlan { step: u64, sequences: [{ seq_id, tokens, positions, block_table, is_prefill }] }`; workers execute; collectives synchronise the step, so there is no per-step ack.
4. `Shutdown { reason }` either direction; a closed socket is treated as `RemoteAbort`.

### CLI

```
turbine-collbench --backend rccl|nccl|host --devices <i,j,..> [--op all_reduce|all_gather|reduce_scatter|broadcast|all]
                  [--min-bytes 8] [--max-bytes 1GiB] [--iters 20] [--warmup 5] [--dtype bf16|fp32]
                  [--rank <r> --world <n> --leader <addr>] [--output text|json]
scripts/lab-cluster.sh <collbench-novanas|tp2-novanas|dp2-novanas>
```

`turbine-collbench` reports per size: `bytes`, `time_us` (median), `algbw_gbps` = bytes / time, `busbw_gbps` = algbw × factor (all-reduce 2(n−1)/n, all-gather and reduce-scatter (n−1)/n, broadcast 1), and `correct` (result compared against the host backend). Exit 0 when every size is correct, 1 on any mismatch or collective error, 2 on usage errors.

### Metrics

- `turbine_collective_duration_seconds{op,backend}` histogram; `turbine_collective_bytes_total{op,backend}` counter; `turbine_collective_errors_total{backend,kind}` counter (`kind` ∈ `timeout`, `remote_abort`, `backend`).
- `turbine_tp_step_skew_seconds{replica}` histogram — slowest minus fastest rank per step.
- `turbine_device_budget_bytes{device,component}` gauge — `component` ∈ `weights`, `kv`, `workspace`, `collective`, `runtime`, `reserve`.
- `turbine_group_pressure_state{replica}` gauge (0 GREEN … 4 SURVIVAL); `turbine_group_limiting_device{replica}` gauge (device index).
- `turbine_dp_routed_total{replica,reason}` counter — `reason` ∈ `prefix_affinity`, `least_loaded`, `pressure_avoidance`, `only_candidate`.
- `turbine_parallel_info{tp,dp,backend,mode}` gauge, always 1.

`device` and `replica` labels are bounded by the inventory and _parallel.data_parallel_size_.

## Data

- Nothing new is persisted. Topology and plan are computed once at startup and held in memory.
- Topology JSON (`GET /turbine/v1/topology`), abbreviated for novanas:

```json
{
  "node": { "hostname": "novanas", "captured_at": "2026-…Z" },
  "vertices": [
    {
      "id": "numa0",
      "kind": "numa",
      "cpus": "…",
      "memory_bytes": 137438953472
    },
    {
      "id": "gpu0",
      "kind": "gpu",
      "device_index": 0,
      "vendor": "amd",
      "arch": "gfx1201",
      "pci_bus_id": "…",
      "numa": 0,
      "numa_source": "nominal"
    },
    {
      "id": "gpu1",
      "kind": "gpu",
      "device_index": 1,
      "vendor": "amd",
      "arch": "gfx1201",
      "pci_bus_id": "…",
      "numa": 0,
      "numa_source": "nominal"
    },
    {
      "id": "nic:…",
      "kind": "nic",
      "netdev": "…",
      "rdma_device": null,
      "link_layer": "ethernet",
      "rate_gbps": 10,
      "ipv4": ["192.168.10.203"],
      "numa": 0
    }
  ],
  "edges": [
    {
      "a": "gpu0",
      "b": "gpu1",
      "kind": "pcie",
      "path": "sys",
      "hops": 2,
      "link_gts": 32,
      "width": 16,
      "p2p": "disabled",
      "source": "vendor"
    }
  ]
}
```

- Vertex kinds: `numa`, `pcie_root`, `pcie_switch`, `gpu`, `nic`, `nvme`. Edge kinds: `pcie`, `nvlink`, `xgmi`, `coherent`, `numa`. `path` uses the NVML topology levels mapped to `self`, `pix`, `pxb`, `phb`, `node`, `sys`, plus `nvlink`/`xgmi`. `p2p` ∈ `enabled`, `disabled`, `unknown`. `source` ∈ `nominal` (derived from link type/speed/width), `vendor` (amd-smi/NVML), `sysfs`. Vertex ids are stable for the process lifetime. On a DGX Spark the GPU has a `coherent` edge to `numa0` with `vendor_interconnect: nvlink_c2c` and GPU↔NIC edges carry `p2p: unknown`.
- Parallel plan (held by the leader, echoed in `/turbine/v1/status`): `{ tp, dp, backend, mode, vendor, excluded_devices, groups: [{ replica, ranks: [{ rank, device, host }] }], reasons: [code] }`. Reason codes: `fits_single_device`, `tp_required_for_capacity`, `grouped_by_link:<path>`, `vendor_homogeneous`, `vendor_excluded:<vendor>`, `explicit_devices`, `device_sharing_enabled`.
- KV: BF16; block ids are logical and allocated by the leader; each rank's physical pool has the same block count and stores its heads' slice, so one block id means the same token range on every rank. Phase-4 tier copies (CPU, NVMe) of a TP block are stored per rank shard, keyed by (block key, tp size, rank). For Llama-3.2-3B at tp = 2 each rank stores 4 KV heads × 128 dims × 28 layers × 2 (K,V) × 2 bytes = 57,344 bytes per token.
- Budget per device: `weights` (this rank's shard) + `kv` + `workspace` + `collective` + `runtime` + `reserve` (phase-3 emergency reserve), each reported separately.

## Edge cases

- One device in the inventory with `tensor_parallel_size: 2` and no `static` mode (rejected naming the key and the device count).
- `devices: [0, 1]` where 0 is NVIDIA and 1 is AMD, with tp = 2 or with tp = 1, dp = 2 (both rejected: `vendor-mixed plan`, until Phase 7); `devices: auto` on such a host picks one vendor and reports the other's devices under `excluded_devices` with `vendor_excluded:<vendor>`.
- Two GPUs of the same vendor but different architecture in one TP group (rejected).
- tp = 4 on a model with 2 KV heads (KV heads replicated ×2), tp = 8 on a model with 3 KV heads (rejected: neither divides nor is a multiple), tp larger than the attention head count (rejected). Llama-3.2-3B (24 heads, 8 KV heads) accepts tp = 2, 4 and 8; OLMoE (16/16) accepts tp = 2, 4, 8.
- OLMoE QK-norm at tp = 2: each rank holds half of the normalised dimension; the norm must use the full-dimension mean of squares (all-reduce of partial sums), never the rank-local one.
- Vocabulary size not divisible by tp (last shard padded; padded logits masked to −∞ before the all-gather). Tied embeddings (Llama-3.2-3B) load the embedding shard once and use it for the LM head.
- PCIe peer access disabled (novanas): topology reports `p2p: disabled`, planner still allows TP but logs `grouped_by_link:sys` and RCCL falls back to host-staged copies.
- GB10: NVML returns no PCIe P2P data for an integrated GPU; the edge is `p2p: unknown`, never guessed.
- `numa_node = -1` in sysfs (novanas): vertex attached to `numa0` with `source: nominal` and a WARN.
- RCCL/NCCL present but older than the minimum version recorded in the code (`ncclGetVersion`; RCCL minimum = the version shipped with ROCm 7.14.1): backend `unavailable` with the version in the detail.
- `static` mode: a worker starts before the leader (retries connect with backoff until init timeout); two workers claim the same rank; a worker with a different model fingerprint or device arch; the leader restarts while workers are connected.
- DP with one replica in SURVIVAL and the other GREEN; every replica in RED (router still routes to the least-bad one, admission decides); a request whose prefix is cached on a replica in ORANGE.
- Cancellation of a request mid-step in a TP group: the leader drops it from the next `StepPlan`; worker KV for its blocks is released by the leader's block manager, never by workers.
- Two DP replicas sharing one device (_parallel.allow_device_sharing_): the device budget is split, never double-counted.

## Failure modes

- **Collective library missing:** with `auto` and tp > 1 → exit 1 naming the library and loader error (tp > 1 cannot run without it); with an explicit _parallel.rccl_library_ / _parallel.nccl_library_ path → exit 1 naming the path.
- **Communicator init hangs** (peer never joins, fabric misconfigured): after `parallel.collective.init_timeout` the communicator is aborted, `/ready` stays `503 collective_init`, the error is logged with the rank list; the process exits 1 in `static` mode, and in `local` mode the affected replica is marked `CIRCUIT_OPEN` while other replicas serve.
- **Collective op times out or a rank reports an async error mid-generation:** `ncclCommAbort` on every rank of the group, all in-flight requests of that replica end with an OpenAI error event `{"error":{"code":"replica_failed"}}` (no silent continuation), the replica goes `CIRCUIT_OPEN`, other replicas keep serving, and the phase-3 breaker's `PROBING` re-creates the communicator.
- **Sticky GPU fault on a rank:** handled by the phase-3 rule (open circuit, drain, exit code 3); in `local` mode the whole process exits 3 because the faulted device's context cannot be isolated from its group.
- **Worker process dies in `static` mode:** the leader sees the socket close or the async error, aborts, fails in-flight requests as above and waits for all ranks to rejoin before `/ready` returns 200 again.
- **Leader dies in `static` mode:** workers see the socket close, abort their communicator, release device memory and exit 1 (their supervisor restarts them).
- **Out of device memory while loading a shard:** exit 1 naming the rank, device, tensor and the budget breakdown; no partial replica is left serving.
- **Pressure on one rank:** the group's state follows the worst rank; admission and the router act on the group state; the limiting device is reported.
- **Topology source unavailable** (sysfs unreadable in a container, vendor topology call unsupported): the affected attributes are `unknown`/`null` with `source` absent, a WARN is logged, and planning uses the most conservative link class (`sys`).
- **Lab Job unschedulable** (novanas GPUs held by another workload): `lab-cluster.sh` deletes its pending Job, exits 1 naming `amd.com/gpu` as unavailable, and the implementer asks the user to free the GPUs.

## Acceptance criteria

- [ ] [S-1] `cargo test -p turbine-device topology::tests::novanas_fixture` exits 0; it builds the graph from a captured novanas fixture (two gfx1201 GPUs at 32 GT/s x16, `numa_node = -1`, amd-smi access table DISABLED, link type PCIE, 2 hops) and asserts one GPU↔GPU edge with `kind: pcie`, `p2p: disabled`, `hops: 2`, both GPUs attached to `numa0` with `source: nominal`; fails if disabled peer access is reported as enabled or unknown.
- [ ] [S-1] `cargo test -p turbine-device topology::tests::spark_fixture` exits 0; it builds the graph from a captured dgx-spark sysfs/NVML fixture and asserts one `gpu` vertex, four `nic` vertices with RDMA devices `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1` (two at 200 Gb/s), a `coherent` GPU↔NUMA edge and `p2p: unknown` on GPU↔NIC; fails if a field is guessed instead of `unknown`.
- [ ] [S-1] `cargo test -p turbine-device topology::tests::missing_sources_degrade` exits 0; it runs discovery against an empty sysfs root and failing vendor calls and asserts a graph with no panics, `unknown` attributes and one WARN per missing source; fails if missing sysfs aborts startup.
- [ ] [S-1] `cargo test -p turbine-api --test api topology_route` exits 0; it asserts `GET /turbine/v1/topology` returns 200 with `vertices` and `edges` arrays matching the injected graph; fails if the route is missing.
- [ ] [S-1] [S-9] `scripts/lab-test.sh novanas` and `scripts/lab-test.sh dgx-spark` exit 0 including `cargo test -p turbine-device --test lab topology_matches_host -- --ignored`, which asserts 2 gfx1201 GPUs with a `p2p: disabled` PCIe edge on novanas and 1 GPU plus a 200 Gb/s RDMA NIC on dgx-spark (read-only discovery, no device memory allocated); fails if real sysfs/vendor output is misparsed.
- [ ] [S-2] `cargo test -p turbine-distributed collective::host::tests::ops_match_reference` exits 0; for world sizes 1, 2, 3, 4 and 8 and FP32/BF16 buffers of 1, 7 and 4099 elements it asserts all-reduce (sum, max), all-gather, reduce-scatter and broadcast equal a naive single-threaded computation bit-for-bit; fails if any op or odd size is wrong.
- [ ] [S-2] `cargo test -p turbine-distributed collective::host::tests::op_timeout_aborts` exits 0; one of two ranks never enters the all-reduce, and the test asserts the other returns `CollectiveError::Timeout` within `op_timeout` + 1 s and later calls return `RemoteAbort`; fails if a missing rank blocks forever.
- [ ] [S-2] `cargo test -p turbine-distributed collective::ffi::tests::missing_library` exits 0; it asserts loading `rccl` and `nccl` with default search on macOS each yield `Unavailable` with a loader message, and an explicit nonexistent _parallel.rccl_library_ path yields an error naming the path; fails if a missing library panics.
- [ ] [S-2] `cargo test -p turbine-distributed collective::ffi::tests::one_binding_both_libraries` exits 0; it loads a stub shared library (built by the test's `build.rs` with the host C compiler) exporting the 13 NCCL-API symbols under each file name _librccl.so.1_ and _libnccl.so.2_, and asserts the same binding table resolves all symbols for both, reports backend `rccl` and `nccl` respectively, and rejects a stub whose `ncclGetVersion` is below the recorded minimum; fails if a second binding or a missing symbol is tolerated.
- [ ] [S-2] `cargo test -p turbine-kernels --test unsafe_isolation` exits 0 with its allowlist extended by exactly `crates/turbine-distributed/src/collective/ffi`; it asserts `unsafe` appears nowhere else in `turbine-distributed` and every block is preceded by `// SAFETY:`; fails if `unsafe` leaks outside that module.
- [ ] [S-3] `cargo test -p turbine-distributed --bin turbine-collbench tests::busbw_formulas` exits 0; it asserts busbw factors 2(n−1)/n, (n−1)/n, (n−1)/n and 1 for n = 2 and 8 and the JSON keys `bytes`, `time_us`, `algbw_gbps`, `busbw_gbps`, `correct`; fails if a formula or key changes.
- [ ] [S-3] [S-9] `scripts/lab-cluster.sh collbench-novanas` exits 0: it runs `turbine-collbench --backend rccl --devices 0,1 --op all --max-bytes 1GiB --output json` in a k3s Job with `amd.com/gpu: 2`, every size `correct: true`, all-reduce `busbw_gbps` > 0 at 256 MiB, JSON pasted into the task evidence; fails if RCCL cannot form a two-GPU communicator without peer access or any result is wrong.
- [ ] [S-4] `cargo test -p turbine-core config::tests::parallel_rejections` exits 0; it asserts an error naming the key for: tp = 3, tp = 16, dp = 0, `devices: [0, 0]` with tp = 2, `collective_backend: host` with a GPU device and tp = 2, `ranks.mode: static` with tp = 1, `ranks.mode: static` without `ranks.leader`, `collective.op_timeout: 50ms`, `collective.op_timeout: 30 s`; fails if any is accepted.
- [ ] [S-4] `cargo test -p turbine-distributed plan::tests::planner_cases` exits 0; on synthetic graphs it asserts: the novanas fixture with tp = 2 → one group {0,1} with `grouped_by_link:sys` and backend `rccl`; 4 NVIDIA GPUs in two NVLink pairs with tp = 2 → groups {0,1},{2,3} with `grouped_by_link:nvlink`; `devices: [0,1]` with 1 NVIDIA + 1 AMD → error `vendor-mixed plan` for both tp = 2 and tp = 1/dp = 2; 3 AMD + 1 NVIDIA with `devices: auto`, tp = 1 → dp = 3 on AMD with the NVIDIA device in `excluded_devices`; Llama-3.2-3B with `tp: auto` on 2 R9700s → tp = 1, dp = 2, reason `fits_single_device`; tp = 4 with 2 KV heads → accepted with KV replication, 3 KV heads → rejected; fails if a flat device order is used instead of link classes or any plan mixes vendors.
- [ ] [S-4] `cargo test -p turbine-server --test server_cli impossible_plan_exits_2_before_bind` exits 0; it starts the binary with tp = 2 on an inventory of zero devices and asserts exit 2, stderr naming _parallel.tensor_parallel_size_, and the port still free; fails if the plan is validated after binding.
- [ ] [S-5] `cargo test -p turbine-distributed rank::tests::static_protocol_handshake` exits 0; with in-process TCP it asserts a leader and 3 workers exchange `Hello`/`Welcome`, a worker with a different `model_fingerprint` or `device_vendor` receives `Reject` naming the field, a duplicate rank is rejected, and a missing worker makes the leader fail with a timeout naming the rank; fails if mismatched ranks are admitted.
- [ ] [S-5] `cargo test -p turbine-distributed rank::tests::leader_loss_aborts_workers` exits 0; closing the leader socket makes every worker abort its (host-backend) communicator and return within 2 s; fails if a worker hangs.
- [ ] [S-5] `cargo test -p turbine-distributed rank::tests::plan_queue_bounded` exits 0; a stalled worker makes the leader block after _parallel.plan_queue_depth_ outstanding plans rather than buffering more; fails if the queue grows unbounded.
- [ ] [S-6] `cargo test -p turbine-distributed tp::tests::sharded_layers_match_unsharded` exits 0; on the host backend with a naive FP32 reference implemented in the test, two synthetic decoder blocks — dense (8 heads / 2 KV heads, tied embeddings, vocab 1003) and MoE (4 heads / 4 KV heads with full-projection QK-norm, 8 experts top-2, vocab 1003) — run at tp = 1, 2 and 4 produce logits within 1e-5 absolute of tp = 1; fails if any sharding rule (row/column split, KV replication, QK-norm all-reduce, vocab padding, tied embedding, router replication) is wrong.
- [ ] [S-6] [S-9] `scripts/lab-cluster.sh tp2-novanas` exits 0: in one k3s Job with `amd.com/gpu: 2` it serves Llama-3.2-3B-Instruct at tp = 2 in `local` mode on 127.0.0.1:18000, waits for `/ready` 200 and runs `turbine-golden compare --url http://127.0.0.1:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl`; repeats in `static` mode (two processes, ranks 0 and 1, leader 127.0.0.1:18100); repeats the `local` run for OLMoE-1B-7B against its phase-2 fixtures under `tests/golden/`; every compare meets the phase-1 tolerance (≥ 14/16 prompts first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats); then `turbine-bench --url http://127.0.0.1:18000 --concurrency 4 --requests 64 --output json` returns `requests_ok: 64` with TTFT/ITL recorded beside a tp = 1 run in the task evidence; fails if any compare exceeds the tolerance or any request fails.
- [ ] [S-7] `cargo test -p turbine-distributed router::tests::routing_policy` exits 0; it asserts: prefix cached on replica 1 → replica 1 with `prefix_affinity`; no prefix → fewest outstanding tokens with `least_loaded`; prefix on a replica in ORANGE while another is GREEN → the GREEN one with `pressure_avoidance`; all replicas RED → least-loaded RED one; one replica → `only_candidate`; and each decision increments `turbine_dp_routed_total` with that reason; fails if the pressure state is ignored.
- [ ] [S-7] [S-9] `scripts/lab-cluster.sh dp2-novanas` exits 0: in one k3s Job with `amd.com/gpu: 2` it serves Llama-3.2-3B-Instruct with tp = 1, dp = 2 (one replica per R9700), runs `turbine-bench --url http://127.0.0.1:18000 --concurrency 8 --requests 128 --output json`, requires `requests_ok: 128` and `/metrics` showing non-zero `turbine_dp_routed_total` for both replicas, and records throughput beside a dp = 1 run in the task evidence; fails if one replica never receives traffic or any request fails.
- [ ] [S-8] `cargo test -p turbine-reliability multi_device::tests::group_state_is_worst_member` exits 0; a simulated TP group of 2 devices at GREEN/ORANGE reports ORANGE with `limiting_device` = the ORANGE device, and recovers to GREEN only after both devices clear the phase-3 hysteresis; fails if group state averages or ignores a member.
- [ ] [S-8] `cargo test -p turbine-reliability multi_device::tests::atomic_group_reservation` exits 0; with rank 1's pool one block short, admitting a request needing N blocks reserves nothing on rank 0, returns `Queue { reason: KvReservation }` (phase-3 `kv_reservation`, CONFLICT C-11), and leaves both pools unchanged; fails if a partial reservation leaks.
- [ ] [S-8] `cargo test -p turbine-reliability multi_device::tests::shared_device_budget` exits 0; two replicas on one 32 GiB discrete device with _parallel.allow_device_sharing_ each see half of the phase-3 device budget, and a KV reservation by one reduces the other's available headroom; fails if replicas sharing a device double-count memory.
- [ ] [S-8] `cargo test -p turbine-api --test api pressure_multi_device_view` exits 0; it asserts `/turbine/v1/pressure` contains `devices`, `groups` (with `limiting_device`) and `replicas` arrays and that `/metrics` exposes `turbine_device_budget_bytes{device="0",component="collective"}`; fails if the collective buffer budget is missing.
- [ ] [S-9] `bash -n scripts/lab-cluster.sh` and `scripts/lab-cluster.sh --dry-run tp2-novanas` exit 0; the dry run prints the Job manifest and asserts it requests `amd.com/gpu: 2`, mounts `/home/piwi/turbine-models` read-only, creates no Service, and that the script's cleanup deletes only Jobs labelled `turbine-lab=true`; fails if the manifest touches other workloads or exposes a host port.

## Open questions

<!-- None: decisions recorded in .procoder/ask/decisions.md -->
