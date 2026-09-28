# phase-9-multi-node

Status: deferred

Numbering: written as Phase 6 (`phase-6-multi-node`) and renumbered on 2026-09-28; only spec file names in the text were updated, the phase numbers in the prose are as written. Within this document "Phase 6" means this spec (phase-9-multi-node), "Phase 7" means `phase-10-advanced-distribution`, and the "Phase 8" tracks (P8a/P8b/P8c) mean active phases 6–8 under `phase-6-8-expansion` (6 quantization, 7 model families, 8 speculative decoding). Deferred until the user lifts the multi-node / NVIDIA hold (decision "Roadmap reorganisation after Phase 5 (2026-09-28)" in `.procoder/ask/decisions.md`); re-specced against the tree of that day before implementation starts.

Source: `turbine-spec.md` §19 Phase 6 (multi-node), with §10 (multi-GPU and multi-node), §11 (cluster-wide KV directory), §12 (failure model), §14 (observability), §16 (security and isolation), §17 (testing) and §21 (engineering rules). Sections of that document are cited as "TS §N". Decisions are recorded in `.procoder/ask/decisions.md` ("Answers log for phase 1–8 spec questions (2026-09-25)"); where they depart from TS this spec says "amends TS §N". This spec builds on phase-0-skeleton through phase-5-multi-gpu; phase-10-advanced-distribution (written as phase-7) builds on the worker registry, authenticated control channel, TCP transport and KV directory defined here.

## Problem

After Phase 5 one Turbine process drives the GPUs of one node (or, in the `static` rank mode, one fixed TP group whose ranks are listed by address). There is still no cluster: a node does not know which other nodes exist, whether they are alive, what they hold or how busy they are; requests cannot be sent to the node best placed to serve them; KV computed on one node cannot be reused by another; and when a node dies nothing notices except the clients whose streams break. TS §10–§12 require the opposite: a topology graph spanning nodes and their NICs, placement that weighs compute, KV locality, transfer cost, pressure and priority, a cluster-wide KV directory that says where a block lives and whether fetching it beats recomputing it, and node failure treated as a scheduling event that removes the node from admission while unaffected traffic keeps flowing. TS §21 rule 5 forbids faking any of it: topology, failure and transport must be explicit. Phase 6 is proven on the two DGX Sparks (one GB10 each, NVIDIA execution available since the NVIDIA track that follows Phase 2), joined by 200 Gb/s RoCE on 192.168.47.0/24 and 192.168.48.0/24 plus the 192.168.10.0/24 LAN: a real two-node cluster with a fast (RoCE subnet) and a slow (LAN) path, serving meta-llama/Llama-3.2-3B-Instruct in BF16. novanas (2× AMD R9700, 10 GbE, no RDMA) is not a lab member in Phase 6: joining it to the Sparks would be mixed-vendor serving, which is excluded until Phase 7; its slow-path behaviour is covered by delay faults in the simulation. Phase 6 is also where the NCCL side of the Phase 5 NCCL-API binding is first proven on hardware (NCCL over RoCE between the Sparks). Transport is TCP; RDMA follows in Phase 7 as TS §19 schedules.

## Users

- **Operators running several nodes:** need to start the same binary on each node with a seed list, see every node's state, devices, replicas, pressure and links at `/turbine/v1/cluster` and `/turbine/v1/topology?scope=cluster`, point clients (or a load balancer) at any node, and lose a node without losing the cluster.
- **API clients:** need the same OpenAI API from every node, requests served by the best replica in the cluster, and a clear, typed error — never a hang or a silently truncated stream — when the node serving them dies.
- **Turbine developers (humans and AI agents):** need membership, placement, directory and failure handling testable as a deterministic multi-node simulation on macOS with injected faults (TS §17 items 2, 3, 7), and lab scenarios that kill a real node.
- **Phase 7 implementers:** need the `Transport` trait (to add RDMA), the worker registry (to restrict RDMA peers), the authenticated control channel, the cluster topology graph and the KV directory as stable foundations.

## In scope

- [S-1] New crate `crates/turbine-transport`: a `Transport` trait with capability flags, a TCP implementation (Tokio; control connections with `TCP_NODELAY`, bulk data over _distributed.transport.tcp.data_streams_ parallel connections per peer), a length-prefixed frame codec with a hard size cap, connect/read/write timeouts, bounded send queues with backpressure, and an in-memory `mem` transport with fault injection (drop, delay, partition, corrupt, close) for tests.
- [S-2] Control-plane protocol in `turbine-distributed::proto`: a versioned set of messages (handshake, peer list, heartbeat, topology, directory delta and snapshot, request forward, stream events, cancel, KV fetch, drain, goodbye) encoded with `postcard` in frames of at most `distributed.transport.max_frame_bytes`, with protocol-version negotiation and rejection of unknown message tags.
- [S-3] Cluster transport security: every control and data connection starts with a mutual challenge-response over a pre-shared key (HMAC-SHA256, key from `distributed.auth.psk_file`, file mode 0600 or stricter); traffic after the handshake is authenticated at connection level only and not encrypted (TS §16 "eventually" is met by authentication now; TLS can wrap the same frames later without a protocol change); peers are identified by node id and cluster name; startup refuses a non-loopback listen address without a key unless `distributed.auth.mode` is explicitly `none`; every decoded message is size- and count-bounded before allocation (TS §16).
- [S-4] Worker discovery and membership: static seed list (_distributed.seeds_), peer-list exchange so every node learns every other (full mesh), heartbeats carrying load and pressure summaries, a single serving vendor per cluster (a node whose replicas run on a different vendor than the cluster's serving members is rejected `vendor_mismatch`; nodes with zero devices join any cluster), node states `JOINING → ALIVE → SUSPECT → DEAD` plus `LEAVING` (graceful drain) and the terminal `LEFT` after `Goodbye` (six states, serialised lowercase — CONFLICT C-16), incarnation numbers so a restarted node is a new member, a node cap (_distributed.max_nodes_), and `GET /turbine/v1/cluster`.
- [S-5] Cluster topology graph: each node sends its Phase 5 node-local subgraph at join; the cluster graph adds `network` edges between NICs that share an IPv4 subnet (nominal rate, RDMA capability, link layer) and between control addresses, annotated with RTT from heartbeats and throughput from a bounded probe at join (64 MiB over the data plane, at most 2 s); `GET /turbine/v1/topology?scope=cluster`. Vertex ids are prefixed with the node id.
- [S-6] Distributed placement, two levels. Model placement: by default each node runs its own Phase 5 plan and every replica is node-local (TP inside a node, DP across nodes — TS §10); cross-node TP groups (the Phase 5 `static` mode, now bootstrapped through membership by node id) are formed only when _distributed.groups_ declares them or `distributed.placement.cross_node_tp` is `when_required` and no single node can hold one replica. Request placement: every node serves the API (symmetric ingress, so losing any node cannot remove ingress — TS §12); the ingress node scores every eligible replica in the cluster (formula under Interfaces) and either serves locally or forwards the request over the control plane and relays the token stream back to the client, with cancellation propagated. Candidates are restricted to replicas of the cluster's serving vendor, so no request is ever served by a mixed-vendor set.
- [S-7] Cluster KV directory in `turbine-kv::directory`: each node is authoritative for its own prefix-cache holdings and pushes additions/removals to every peer as sequenced deltas (no consensus, no sharding, no coordinator); every node keeps a bounded hint index (block key → holders, tier, bytes, compatibility id); lookups are hints — a stale hit becomes a miss. An L3 "cluster" tier adapter implements the phase-4 tier trait: it decides retrieve-vs-recompute from the transfer estimate and the phase-4 recompute cost, fetches blocks over the data plane with per-block crc32c, lands them in the local pinned-CPU tier and promotes them through the phase-4 path; any fetch failure or timeout falls back to recompute.
- [S-8] Node-failure handling (TS §12): detection (heartbeat timeout, connection loss, collective async error), removal from admission and placement, identification of affected requests, each forwarded request restarted on another replica only if no token has reached the client (at most once) and otherwise ended with `worker_lost` (no continuation after streamed tokens — TS §12), directory purge of the node's entries, in-flight KV fetches aborted to recompute, cross-node TP groups containing the node aborted on survivors, and rejoin as a new incarnation. Every step logs a reason code and increments a metric.
- [S-9] Deterministic cluster simulation (`turbine-distributed::sim`): N simulated nodes in one process on the `mem` transport with paused Tokio time, a fake engine (deterministic token generator with configurable step time, KV holdings and pressure), and scripted faults (kill, restart, partition, heal, delay, drop, corrupt), run by `cargo test` on macOS.
- [S-10] Observability: the metrics listed under Interfaces; `/turbine/v1/status` gains `cluster` (node id, state, member count); `/turbine/v1/kv` gains a `directory` section; a structured log event with reason codes for every membership transition, placement decision, forward, retrieve-vs-recompute choice and failure-handling step.
- [S-11] Lab scenarios added to `scripts/lab-cluster.sh`, all on dgx-spark and dgx-spark2: `collbench-sparks` (NCCL collectives across the two GB10s over RoCE — the NCCL hardware proof of the Phase 5 binding), `tp2-sparks` (one declared cross-node TP group spanning both Sparks), `dp2-sparks` (one replica per Spark, cluster over the RoCE subnet), `kv-remote-hit` (a prefix computed on one Spark served from the other), `chaos-kill-node` (kill the Turbine container on dgx-spark2 under load) and `chaos-partition` (fault-injection build drops the control plane between the Sparks), each tearing down only `turbine-lab-*` containers.

## Out of scope

- RDMA/verbs transport, direct GPU-to-GPU KV transfer, the TKV1 transfer format, PP, EP, prefill/decode disaggregation and heterogeneous placement (Phase 7). Phase 6 KV transfer is TCP, host-staged.
- Encryption of cluster traffic; mTLS; certificate management. Phase 6 authenticates connections with the PSK only.
- KV storage-only nodes that hold KV without executing a model (e.g. novanas RAM as an L3 tier); only executing nodes hold cluster KV in Phase 6.
- Continuing a generation on another node after tokens have been streamed.
- Mixed-vendor clusters (NVIDIA and AMD replicas serving together) and cross-vendor KV exchange (Phase 7). novanas is not a Phase 6 lab member.
- Other discovery mechanisms (SWIM gossip, Kubernetes API, mDNS) and other control protocols (gRPC, HTTP/JSON, Cap'n Proto).
- Consensus (Raft/Paxos), leader election and any cluster-wide strongly consistent state: nothing in Phase 6 needs it; the directory is a hint and membership is per-node view.
- Autoscaling, node provisioning, Kubernetes operators, service meshes, multi-cluster federation.
- Dynamic re-planning of model placement at runtime (a node's replicas are fixed for its process lifetime; cross-node groups are re-formed only by restart/rejoin).
- Per-tenant accounting and quotas across nodes.

## Constraints

- Rust only; no Python (TS §21 rule 4). `turbine-transport` and all of Phase 6 are safe Rust (`unsafe_code = "forbid"` in `turbine-transport`; nothing is added to the phase-1 `unsafe_isolation` allowlist).
- New dependencies, each with its reason: `hmac` + `sha2` (PSK challenge-response), `crc32c` (block checksums; same algorithm Phase 7 uses), `rand` (nonces from the OS random source). `postcard` arrives in Phase 5.
- No fake distributed abstractions (TS §21 rule 5): every remote call has an explicit timeout, every message type an explicit failure outcome, and placement consumes the topology graph, never a flat node list.
- Every queue and cache is bounded (TS §21 rule 8): per-peer send queue (`distributed.transport.send_queue_frames`), forwarded requests per peer (`distributed.forwarding.max_inflight_per_peer`), in-flight KV bytes per node (`distributed.kv_fetch.max_inflight_bytes`), directory entries (`distributed.kv_directory.max_entries`), delta entries per frame (4,096), peers (_distributed.max_nodes_).
- Failure detection uses each node's local monotonic clock only; no cross-node clock agreement is assumed. Heartbeat `sent_at` is used only for RTT on the sender.
- KV blocks cross nodes only between compatible replicas: identical compatibility id = (phase-4 block identity fields: model fingerprint and version, KV format, block_tokens) + TP layout (tp size) — mismatches are never fetched (TS §8, §11).
- Every automatic decision (membership transition, routing target, forward, retrieve vs recompute, failure-handling step) carries a reason code and a metric (TS §14, §21 rule 7).
- Lab: dgx-spark (192.168.10.246 / 192.168.47.246) and dgx-spark2 (192.168.10.245 / 192.168.47.245) only. The Sparks keep serving production vLLM, which lab runs never stop, restart or move; lab HTTP ports 18000–18099, control 18110, data 18111; containers named `turbine-lab-*`; cluster traffic on 192.168.47.0/24; model Llama-3.2-3B-Instruct from `/home/piwi/turbine-models/llama-3.2-3b-instruct` mounted read-only; Turbine's memory capped with `reliability.memory.device_budget_bytes` = 16 GiB per node. Before starting, every scenario checks on each Spark that `MemAvailable` exceeds the cap plus `reliability.memory.host_reserve_bytes`; if not, it exits 1 with `precondition` naming the host, starts nothing, and the implementer asks the user to free memory — any run that needs production workloads moved or memory freed is asked of the user first, never done by a script.
- NCCL on the Sparks: the CUDA 13.0 host installs have no NCCL, so the lab image gains `libnccl2` from NVIDIA's CUDA apt repository (arm64 sbsa); `turbine-lab-*` containers for cross-Spark collectives get `--network host`, `--device /dev/infiniband`, `--ulimit memlock=-1` and `NCCL_IB_HCA=rocep1s0f0`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`. Chaos scenarios kill only `turbine-lab-*` containers and never touch host networking (no `iptables`); partitions are injected by a `fault-injection` Cargo feature of `turbine-server`, never compiled into release builds.
- Cross-node TP (collectives over RoCE) uses NCCL's own IB path through the Phase 5 NCCL-API binding; Phase 6's transport carries control and KV traffic only. Transport is TCP only in Phase 6 (RDMA verbs in Phase 7); TCP runs everywhere including macOS tests.

## Interfaces

### Configuration (`distributed` section; unknown keys are errors)

| Key                                            | Type                                 | Default         | Validation                                                                                               |
| ---------------------------------------------- | ------------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------- |
| _distributed.enabled_                          | bool                                 | `false`         | `true` now accepted                                                                                      |
| _distributed.cluster_name_                     | string                               | `turbine`       | `[a-z0-9-]{1,63}`                                                                                        |
| _distributed.node_id_                          | string                               | hostname        | `[a-z0-9-]{1,63}`, unique in the cluster (duplicate → join rejected)                                     |
| _distributed.control_listen_                   | socket address                       | `0.0.0.0:8100`  | —                                                                                                        |
| _distributed.control_advertise_                | socket address or null               | null            | required when the listen IP is unspecified (`0.0.0.0`)                                                   |
| _distributed.data_listen_                      | socket address                       | `0.0.0.0:8101`  | port differs from control and server ports                                                               |
| _distributed.data_advertise_                   | socket address or null               | null            | as for control; may be on a different subnet (e.g. the RoCE subnet)                                      |
| _distributed.seeds_                            | list of socket addresses             | `[]`            | may include the node's own address (ignored); empty = single-node cluster                                |
| _distributed.max_nodes_                        | integer                              | `16`            | 1 ≤ n ≤ 256                                                                                              |
| _distributed.groups_                           | list of cross-node TP groups or null | null            | each `{replica, ranks: [{node, device}]}`; nodes must be distinct or devices distinct; tp = ranks length |
| `distributed.heartbeat.interval`               | duration                             | `1s`            | 100 ms ≤ value ≤ 10 s                                                                                    |
| `distributed.heartbeat.suspect_after`          | duration                             | `3s`            | ≥ 2 × interval                                                                                           |
| `distributed.heartbeat.dead_after`             | duration                             | `10s`           | > suspect_after                                                                                          |
| `distributed.auth.mode`                        | `psk`, `none`                        | `psk`           | `none` with a non-loopback listen address logs a WARN at every startup                                   |
| `distributed.auth.psk_file`                    | path                                 | null            | required for `psk`; file ≥ 32 bytes, mode 0600 or stricter, else exit 2                                  |
| `distributed.transport.max_frame_bytes`        | byte size                            | `16MiB`         | 64 KiB ≤ value ≤ 256 MiB                                                                                 |
| `distributed.transport.send_queue_frames`      | integer                              | `1024`          | 16 ≤ n ≤ 65536                                                                                           |
| `distributed.transport.tcp.data_streams`       | integer                              | `4`             | 1 ≤ n ≤ 16                                                                                               |
| `distributed.transport.tcp.connect_timeout`    | duration                             | `5s`            | —                                                                                                        |
| `distributed.forwarding.max_inflight_per_peer` | integer                              | `256`           | 1 ≤ n ≤ 65536                                                                                            |
| `distributed.kv_directory.enabled`             | bool                                 | `true`          | requires the phase-4 prefix cache                                                                        |
| `distributed.kv_directory.max_entries`         | integer                              | `1000000`       | ≥ 1024                                                                                                   |
| `distributed.kv_directory.announce_interval`   | duration                             | `250ms`         | 10 ms ≤ value ≤ 10 s                                                                                     |
| `distributed.kv_fetch.timeout`                 | duration                             | `2s`            | 10 ms ≤ value ≤ 60 s                                                                                     |
| `distributed.kv_fetch.max_inflight_bytes`      | byte size                            | `1GiB`          | ≥ one block                                                                                              |
| `distributed.placement.cross_node_tp`          | `never`, `when_required`, `declared` | `when_required` | `declared` = only _distributed.groups_                                                                   |
| `distributed.placement.weights.queue`          | float                                | `1.0`           | ≥ 0 (same for `prefill`, `transfer`, `forward`, `pressure`)                                              |

Durations and byte sizes use the Phase 0 / Phase 5 formats. _distributed.enabled_ `false` keeps every Phase 5 behaviour and opens no port other than _server.listen_.

### HTTP routes (changes)

| Route                                          | Phase 6 response                                                                                                                                           |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /turbine/v1/cluster`                      | `200` membership document (see Data); `404` code `distributed_disabled` when _distributed.enabled_ is false                                                |
| `GET /turbine/v1/topology?scope=cluster`       | `200` cluster graph; `scope=node` (default) is the Phase 5 node graph                                                                                      |
| `GET /turbine/v1/kv`                           | adds `directory: {entries, holders_by_node, last_delta_seq_by_node, fetches_inflight, fetch_bytes_inflight}`                                               |
| `GET /turbine/v1/status`                       | adds `cluster: {node_id, incarnation, state, members_alive, members_total}`                                                                                |
| `GET /ready`                                   | `200` when this node can serve or forward (≥ 1 eligible replica cluster-wide); a node with zero local replicas is still ready if peers have one            |
| `POST /v1/chat/completions`, `/v1/completions` | may be served by a remote replica; response header `x-turbine-served-by: <node_id>/<replica>`; errors below                                                |
| `POST /turbine/v1/debug/faults`                | only with the `fault-injection` feature: `{"peer":"<node_id>","action":"<action>","ms":<u64>}` with action `partition`, `heal` or `delay`; otherwise `404` |

Errors added (OpenAI error shape; mid-stream as a final SSE `data: {"error":…}` event followed by `[DONE]`):

| Code             | Status | When                                                                                                    |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------- |
| `worker_lost`    | 502    | the node serving the request died after at least one token was streamed                                 |
| `forward_failed` | 502    | forwarding failed and the restart budget (1 restart) is exhausted or disabled                           |
| `no_capacity`    | 503    | no eligible replica anywhere in the cluster (every candidate DEAD, SUSPECT, SURVIVAL or `CIRCUIT_OPEN`) |

### Control-plane messages (protocol version 1)

Turbine's own versioned message enum over `turbine-transport` (no gRPC, no schema compiler). Frames: `u32` little-endian length, then a `postcard`-encoded `Message` enum. After TCP connect: `Hello { protocol_min, protocol_max, cluster_name, node_id, incarnation, serving_vendor, nonce }` (`serving_vendor` is `nvidia`, `amd` or null for a node with no replicas) → `Challenge { nonce, hmac(psk, their_nonce ‖ node_id) }` → `Proof { hmac(psk, our_nonce ‖ node_id) }` → `Accept { protocol, cluster_vendor }` or `Reject { reason }` (reasons: `bad_auth`, `cluster_mismatch`, `duplicate_node_id`, `protocol_unsupported`, `cluster_full`, `vendor_mismatch`). `cluster_vendor` is the vendor of the ALIVE serving member with the lowest incarnation that the acceptor knows (null if none); a joiner whose non-null `serving_vendor` differs is rejected `vendor_mismatch`.

| Message                                                                                                                                                             | Direction            | Purpose                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `PeerList { peers: [{node_id, incarnation, control, data}] }`                                                                                                       | both, on join/change | full-mesh discovery                                                                                        |
| `Heartbeat { seq, sent_at_mono_ns, replicas: [{replica, pressure, circuit, queued_prefill_tokens, active_seqs, free_kv_blocks, prefill_tps_ewma}], directory_seq }` | both                 | liveness, load and pressure                                                                                |
| `Topology { graph }`                                                                                                                                                | on join              | node-local subgraph (Phase 5 shape)                                                                        |
| `DirectoryDelta { seq, added: [Entry], removed: [BlockKey] }` / `DirectorySnapshotRequest` / `DirectorySnapshot { seq, chunk, last, entries }`                      | both                 | KV directory sync                                                                                          |
| `Forward { request_id, deadline_ms, replica, body }` → `StreamEvent { request_id, payload }` … `StreamEnd { request_id, usage, finish }`                            | ingress ↔ server     | remote serving with relayed stream                                                                         |
| `Cancel { request_id }`                                                                                                                                             | ingress → server     | client disconnect or timeout                                                                               |
| `KvFetch { transfer_id, compat, keys }` → `KvBlock { transfer_id, key, rank, crc32c, bytes }`… `KvFetchEnd { transfer_id, missing }`                                | data plane           | L3 retrieval (wire message `Message::KvBlock`, distinct from phase-4 `turbine_kv::KvBlock`, CONFLICT C-17) |
| `Drain { reason }` / `Goodbye`                                                                                                                                      | leaving node → peers | graceful leave                                                                                             |

### Transport trait (`turbine-transport`)

```rust
pub struct TransportCaps { pub kind: TransportKind, pub rdma: bool, pub gpu_direct: bool, pub max_frame: u64 }
pub enum TransportKind { Tcp, Mem }            // Phase 7 adds Rdma

pub trait Transport: Send + Sync + 'static {
    fn caps(&self) -> TransportCaps;
    fn connect(&self, peer: &PeerAddr, class: ConnClass) -> BoxFuture<'_, Result<Connection, TransportError>>;
    fn listen(&self, addr: &SocketAddr, class: ConnClass) -> BoxFuture<'_, Result<Listener, TransportError>>;
}
pub enum ConnClass { Control, Data }
```

`Connection` exposes `send(Frame)` (bounded, awaits capacity), `recv() -> Frame`, `close()`; `TransportError` variants `Timeout`, `Closed`, `FrameTooLarge`, `Auth`, `Io`.

### Request placement score

For each eligible replica _r_ (node ALIVE, replica serving the requested model on the cluster vendor, replica circuit not `CIRCUIT_OPEN`, pressure not SURVIVAL; high-priority requests — Phase 2 `priority` < 0 (CONFLICT C-10) — also exclude ORANGE and RED):

```
cost_ms(r) = w_queue    · queued_prefill_tokens(r) / prefill_tps(r)
           + w_prefill  · uncached_tokens(r)      / prefill_tps(r)
           + w_transfer · transfer_ms(r)                   # 0 unless the best prefix is on another node and retrieval beats recompute
           + w_forward  · rtt_ms(ingress → node(r))        # 0 when local
           + w_pressure · {GREEN 0, YELLOW 50, ORANGE 500, RED 5000}
```

The lowest cost wins; a local replica within 5 % of the best wins ties. The reason code is the term that most separates the winner from the runner-up: `local_prefix_hit`, `remote_prefix_hit`, `least_loaded`, `pressure_avoidance`, `forward_cheaper`, `only_candidate`. Inputs come from the latest heartbeat plus the ingress node's own not-yet-acknowledged forwards, so bursts do not all land on one replica.

### Retrieve vs recompute

`retrieve_ms = rtt_ms + bytes / bandwidth_ewma` (bandwidth seeded by the join probe, updated by every fetch); `recompute_ms` from the phase-4 recompute cost estimate. Retrieve when `1.2 × retrieve_ms < recompute_ms` and the holder is ALIVE; reason codes `remote_retrieve`, `recompute_cheaper`, `incompatible_layout`, `holder_not_alive`, `fetch_budget_full`.

### Metrics

- `turbine_cluster_members{state}` gauge; `turbine_cluster_membership_transitions_total{from,to}` counter.
- `turbine_transport_bytes_total{peer,class,direction}` counter; `turbine_transport_errors_total{peer,kind}` counter; `turbine_transport_rtt_seconds{peer}` histogram. `peer` is a node id, bounded by _distributed.max_nodes_.
- `turbine_placement_decisions_total{target,reason}` counter — `target` ∈ `local`, `remote`.
- `turbine_forwarded_requests_total{peer,outcome}` counter — `outcome` ∈ `ok`, `restarted`, `worker_lost`, `forward_failed`, `cancelled`.
- `turbine_kv_directory_entries{holder}` gauge; `turbine_kv_remote_decisions_total{reason}` counter; `turbine_kv_fetch_bytes_total{peer}` counter; `turbine_kv_fetch_duration_seconds` histogram; `turbine_kv_fetch_failures_total{kind}` counter (`timeout`, `checksum`, `closed`, `missing`).
- `turbine_node_failures_total{peer,detector}` counter — `detector` ∈ `heartbeat`, `connection`, `collective`; `turbine_recovery_actions_total{action}` counter — `action` ∈ `restart_request`, `fail_request`, `purge_directory`, `abort_fetch`, `abort_group`.

## Data

- Nothing new is persisted. Membership, directory, topology and load views live in memory and are rebuilt from peers after a restart.
- Membership document (`GET /turbine/v1/cluster`); pressure and circuit values use the phase-3 uppercase strings, in JSON and metric labels alike (CONFLICT C-5):

```json
{
  "cluster_name": "turbine",
  "self": "dgx-spark",
  "members": [
    {
      "node_id": "dgx-spark2",
      "incarnation": 1790000000123456789,
      "state": "alive",
      "control": "192.168.47.245:18110",
      "data": "192.168.47.245:18111",
      "last_heartbeat_ms_ago": 412,
      "rtt_ms": 0.09,
      "replicas": [
        {
          "replica": 0,
          "tp": 1,
          "pressure": "GREEN",
          "circuit": "HEALTHY",
          "devices": ["dgx-spark2/gpu0"]
        }
      ]
    }
  ],
  "groups": [
    {
      "replica": 1,
      "ranks": [
        { "node": "dgx-spark", "device": 0 },
        { "node": "dgx-spark2", "device": 0 }
      ],
      "state": "HEALTHY"
    }
  ]
}
```

- Incarnation = wall-clock nanoseconds at process start (u64); a higher incarnation for a known node id replaces the old member only if the old one is `SUSPECT` or `DEAD`, otherwise the join is rejected `duplicate_node_id`.
- Directory entry: `{ key: BlockKey (phase-4 block hash, 128-bit; BlockKey is an alias of phase-4 KvKey, CONFLICT C-17), compat: u64, holder: node_id, tier: l0|l1|l2 (phase-4 tier names, CONFLICT C-4), bytes: u64, prefix_depth: u32 }`. Each node stores at most `distributed.kv_directory.max_entries` remote entries; on overflow it drops the entries with the smallest `prefix_depth` first and logs the count. Deltas carry a per-holder `seq`; a gap triggers `DirectorySnapshotRequest` for that holder.
- What a fetch moves: the phase-4 prefix-cache entry for each key — attention KV blocks per TP rank shard only; linear-attention state snapshots arrive with the Phase 8 model-families track (P8c; TKV1 segment kinds 2/3 are reserved for them in Phase 7) (CONFLICT C-22) — framed per block with its crc32c. Received blocks land in the local pinned-CPU tier and enter the local directory as local holdings.
- Cluster topology graph: the union of node subgraphs with ids `<node_id>/<vertex id>` plus `network` edges `{a, b, subnet, rate_gbps, rdma, link_layer, rtt_ms, measured_gbps, source}`.

## Edge cases

- A node lists itself in _distributed.seeds_; two nodes seed only each other; a seed that is down at startup (retried with capped exponential backoff, node still starts as a one-node cluster).
- Two processes with the same node id (second rejected `duplicate_node_id` while the first is alive); a node restarting within `dead_after` (new incarnation replaces the SUSPECT one; its old forwards are failed per the in-flight rule).
- Asymmetric reachability: A reaches B but B cannot reach A (B's connections time out; A sees missing heartbeats and marks B SUSPECT; neither blocks on the other).
- Symmetric partition of a two-node cluster: each marks the other DEAD and keeps serving its own replicas; directory hints to the other side are purged; on heal they re-handshake and exchange snapshots. No split-brain hazard because no state requires agreement.
- A cross-node TP group where one member node is DEAD: the group aborts on the survivor, its local devices return to idle and the node serves no replica until the group re-forms (config-declared) — surfaced as `groups[].state: "broken"`.
- Forwarded request whose client disconnects: `Cancel` sent; the server releases KV within the phase-2 cancellation bound.
- Directory says node B holds a prefix, B evicted it a moment ago: `KvFetchEnd { missing: [key] }` → recompute; counted as `missing`.
- KV fetch corrupted in flight (crc mismatch): that block discarded, whole fetch falls back to recompute for the uncovered range.
- Heartbeat storms: 256 nodes × 1 s heartbeat is bounded by _distributed.max_nodes_; frames larger than `distributed.transport.max_frame_bytes` close the connection with `FrameTooLarge`.
- Peer speaks protocol version 2 only (rejected `protocol_unsupported` with both ranges logged); once Phase 7 raises the maximum protocol version to 2, the unsupported example becomes version 3.
- PSK file missing, world-readable, or shorter than 32 bytes (exit 2 naming the key).
- Replicas on different nodes serving different models or different TP layouts: routing only considers replicas serving the requested model; KV exchange only between equal compatibility ids.
- A node with zero devices (e.g. a macOS workstation): joins any cluster, serves the API, forwards every request.
- An AMD node (e.g. novanas with its R9700 replicas) trying to join a cluster whose serving members are NVIDIA: rejected `vendor_mismatch`, logged on both sides; it keeps serving alone as a one-node cluster. If two single-vendor clusters race (both sides seeded while no serving member was known), placement still restricts candidates to the cluster vendor, so no request is served by the other vendor.

## Failure modes

- **Seed unreachable at startup:** WARN, retry with backoff (1 s doubling to 30 s); the node serves as a one-node cluster meanwhile.
- **Peer stops heartbeating:** `SUSPECT` after `suspect_after` (removed from new placement, in-flight work continues), `DEAD` after `dead_after` (recovery runs). Recovery for a DEAD node, in order: stop forwarding to it; for each request this node forwarded to it — restart on another replica if no token has reached the client (at most one restart), else end the stream with `worker_lost`; purge its directory entries; abort KV fetches from it (requests continue by recompute); abort any cross-node TP group containing it (Phase 5 abort path); emit `turbine_node_failures_total` and one structured event listing the affected request ids and counts.
- **Connection closed while heartbeats were fine:** immediate `SUSPECT`, reconnect attempts every heartbeat interval; `DEAD` only at `dead_after` without a successful reconnect.
- **Transport slow but alive:** forwarding and fetch timeouts fire per request; the RTT/throughput EWMAs rise so the placement score shifts traffic away; no membership change.
- **Authentication failure:** connection closed with `Reject { bad_auth }`, logged at WARN with the remote address (never the key), counted; repeated failures from one address are rate-limited to one log line per minute.
- **Send queue full to a peer:** the sender awaits capacity (backpressure) up to the message's timeout; heartbeats use a separate priority slot so a saturated data path cannot cause a false `DEAD`.
- **Directory memory cap reached:** lowest-value entries dropped (smallest `prefix_depth`), counted; correctness unaffected (hints only).
- **Every replica in the cluster ineligible:** new requests get `503 no_capacity` with `retry_after`; `/ready` returns 503 `no_eligible_replica`.
- **This node's own engine fails (phase-3 `CIRCUIT_OPEN`):** its replicas are advertised `circuit_open` in the next heartbeat and peers stop routing to them; it keeps forwarding its ingress traffic to healthy peers.

## Acceptance criteria

- [ ] [S-1] `cargo test -p turbine-transport tcp::tests::frame_roundtrip_and_cap` exits 0; over loopback it sends frames of 0 B, 1 B, 1 MiB and exactly `max_frame_bytes`, receives them intact, and asserts a frame of `max_frame_bytes + 1` closes the connection with `FrameTooLarge` before the payload is allocated; fails if the cap is checked after allocation or frames are corrupted.
- [ ] [S-1] `cargo test -p turbine-transport tcp::tests::send_queue_backpressure` exits 0; with a receiver that never reads, `send` blocks after `send_queue_frames` frames and returns `Timeout` at the deadline; fails if the queue grows past its bound.
- [ ] [S-1] `cargo test -p turbine-transport mem::tests::fault_injection` exits 0; it asserts `partition` makes sends time out, `heal` restores delivery, `delay 50ms` delays delivery by ≥ 50 ms in paused time, `corrupt` flips a byte that the receiver's codec rejects, and `close` yields `Closed`; fails if any injected fault is not observable.
- [ ] [S-2] `cargo test -p turbine-distributed proto::tests::message_roundtrip_and_versioning` exits 0; every `Message` variant round-trips through the codec, an unknown tag is rejected, a `Hello` with `protocol_min = protocol_max = 2` gets `Reject { protocol_unsupported }` (the Phase 7 plan changes this to 3 when it raises `PROTOCOL_MAX` to 2), and a `DirectoryDelta` with 4,097 entries is rejected; fails if any variant fails to decode or oversized lists are accepted.
- [ ] [S-3] `cargo test -p turbine-distributed auth::tests::psk_handshake` exits 0; peers with the same key connect; a wrong key gets `Reject { bad_auth }` on both sides; a replayed `Proof` from an earlier session is rejected; a mismatched cluster name gets `cluster_mismatch`; fails if a peer without the key can send any message past the handshake.
- [ ] [S-3] `cargo test -p turbine-core config::tests::distributed_rejections` exits 0; it asserts errors naming the key for: `auth.mode: psk` without `auth.psk_file`, a PSK file with mode 0644, a 16-byte PSK file, `node_id: Bad_Name`, `control_listen: 0.0.0.0:8100` without `control_advertise`, `data_listen` equal to `control_listen`, `heartbeat.suspect_after` < 2 × interval, `heartbeat.dead_after` ≤ `suspect_after`, `max_nodes: 0`; fails if any is accepted.
- [ ] [S-4] `cargo test -p turbine-distributed sim::tests::three_nodes_converge` exits 0; three simulated NVIDIA-serving nodes where only node a is a seed all list three `alive` members within 3 heartbeat intervals of paused time; a fourth node with a duplicate node id is rejected `duplicate_node_id`; a node advertising `serving_vendor: amd` is rejected `vendor_mismatch`; a node with no replicas is admitted; fails if peer-list exchange does not reach a full mesh or a mixed-vendor member is admitted.
- [ ] [S-4] `cargo test -p turbine-distributed sim::tests::suspect_dead_rejoin` exits 0; after node b is killed, a sees b `suspect` at `suspect_after` ± one interval and `dead` at `dead_after` ± one interval; b restarts with a higher incarnation and is `alive` again within 3 intervals; fails if a transition happens early, late, or the old incarnation is resurrected.
- [ ] [S-4] `cargo test -p turbine-distributed sim::tests::graceful_leave` exits 0; node b sends `Drain`, receives no new placements, finishes its in-flight requests, sends `Goodbye`, and a lists it `left` with zero `worker_lost` errors; fails if a draining node receives new work.
- [ ] [S-4] [S-10] `cargo test -p turbine-api --test api cluster_route` exits 0; it asserts `GET /turbine/v1/cluster` returns the membership document shape from Data with an injected view and `404 distributed_disabled` when disabled, and that `/turbine/v1/status` contains `cluster.members_alive`; fails if the route or fields are missing.
- [ ] [S-5] `cargo test -p turbine-distributed topology::tests::cluster_graph_merge` exits 0; merging the dgx-spark and dgx-spark2 Phase 5 fixtures yields vertex ids prefixed by node id and exactly two `network` edges with `rdma: true`, `rate_gbps: 200` for the shared 192.168.47.0/24 and 192.168.48.0/24 subnets plus one LAN edge, and no edge between NICs on unrelated subnets; fails if the cluster graph is a flat device list or invents links.
- [ ] [S-6] `cargo test -p turbine-distributed placement::tests::score_cases` exits 0; with fixed heartbeats it asserts: equal load, prefix cached locally → local with `local_prefix_hit`; local replica ORANGE and remote GREEN → remote with `pressure_avoidance`; high-priority request with all local replicas ORANGE → remote; prefix on remote and remote idle → remote with `remote_prefix_hit`; costs within 5 % → local; SUSPECT and `circuit_open` replicas never chosen; 100 simultaneous requests with equal inputs split within ±10 % across two replicas thanks to pending-forward accounting; fails if any term is ignored.
- [ ] [S-6] `cargo test -p turbine-distributed placement::tests::model_placement` exits 0; a model that fits one node with two one-GPU nodes → two node-local replicas (`fits_single_device`); a model that fits neither node alone with `cross_node_tp: when_required` → one cross-node TP group of 2; the same with `never` → startup error naming `distributed.placement.cross_node_tp`; fails if TP is placed across nodes when DP suffices.
- [ ] [S-6] `cargo test -p turbine-distributed sim::tests::forward_and_relay` exits 0; a streaming request entering node a and placed on b yields the same chunk sequence the fake engine produced on b, `x-turbine-served-by` is `b/0`, a client disconnect sends `Cancel` and b frees the sequence within one step; fails if chunks are reordered, duplicated or cancellation is not propagated.
- [ ] [S-7] `cargo test -p turbine-kv directory::tests::delta_sequencing_and_cap` exits 0; deltas applied in order produce the expected holder sets, a gap triggers a snapshot request, a snapshot replaces that holder's entries, and exceeding `max_entries` evicts smallest `prefix_depth` first; fails if a gap is silently skipped or the index exceeds its cap.
- [ ] [S-7] `cargo test -p turbine-kv directory::tests::retrieve_vs_recompute` exits 0; with rtt 0.1 ms and bandwidth 2 GB/s a 200 MB prefix whose recompute estimate is 1 s is retrieved (`remote_retrieve`), with bandwidth 0.1 GB/s it is recomputed (`recompute_cheaper`), a different compat id gives `incompatible_layout`, a SUSPECT holder gives `holder_not_alive`; fails if the 1.2 margin or any rule is not applied.
- [ ] [S-7] `cargo test -p turbine-distributed sim::tests::fetch_failures_fall_back` exits 0; for a fetch that times out, returns a crc mismatch, loses its connection mid-stream, or reports `missing`, the request completes by recompute with output identical to a no-directory run, and the matching `turbine_kv_fetch_failures_total{kind}` increments; fails if a failed fetch fails the request or corrupt data is used.
- [ ] [S-8] `cargo test -p turbine-distributed sim::tests::node_loss_is_scheduling_event` exits 0; with 40 streaming requests spread over nodes a and b, killing b mid-run yields: every request placed on a completes normally; each request on b with no token delivered is restarted on a and completes (`restarted`); each request on b with tokens delivered ends with a `worker_lost` error event and `[DONE]`; b's directory entries are gone from a; zero requests hang past the test deadline; fails if any unaffected request fails or any affected request hangs.
- [ ] [S-8] `cargo test -p turbine-distributed sim::tests::partition_and_heal` exits 0; a symmetric partition makes each side serve only its local replicas with no errors for requests entering and served on the same side, and after heal both sides list each other `alive` and exchange directory snapshots; fails if either side stops serving its own traffic.
- [ ] [S-8] `cargo test -p turbine-distributed sim::tests::cross_node_group_abort` exits 0; killing one member node of a declared cross-node TP group aborts the group's (host-backend) communicator on the survivor within `dead_after` + 1 s, fails its in-flight requests with `worker_lost` or restarts unstarted ones, and marks the group `broken`; fails if the survivor blocks in a collective.
- [ ] [S-9] `cargo test -p turbine-distributed sim::tests::deterministic_replay` exits 0; the same seed and fault script run twice produce byte-identical event logs; fails if the simulation depends on wall-clock time or hash-map iteration order.
- [ ] [S-10] `cargo test -p turbine-api --test api distributed_metrics_bounded` exits 0; after simulated traffic from 3 peers it asserts the metrics under Interfaces are present, every `peer` label is one of the 3 node ids, and no request id or address appears as a label value; fails if a label is unbounded.
- [ ] [S-11] `scripts/lab-cluster.sh collbench-sparks` exits 0: it runs `turbine-collbench --backend nccl --op all --max-bytes 1GiB --output json` as rank 0 on dgx-spark and rank 1 on dgx-spark2 over `rocep1s0f0`, every size `correct: true` and all-reduce `busbw_gbps` ≥ 10 at 256 MiB, JSON in the task evidence; fails if NCCL falls back to TCP sockets (the `NCCL_DEBUG=INFO` log lacks `NET/IB`), bandwidth is below 10 GB/s, or a non-`turbine-lab-*` container changes.
- [ ] [S-6] [S-11] `scripts/lab-cluster.sh tp2-sparks` exits 0: with `distributed.placement.cross_node_tp: declared` and _distributed.groups_ declaring replica 0 as {dgx-spark gpu0, dgx-spark2 gpu0}, both nodes join, the group forms through membership and NCCL over RoCE, `/ready` is 200 on both, `turbine-golden compare --url http://192.168.10.246:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` meets the phase-1 tolerance (≥ 14/16 prompts first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats), and `turbine-bench --url http://192.168.10.246:18000 --concurrency 4 --requests 64 --output json` returns `requests_ok: 64` with TTFT/ITL recorded beside a single-node run; fails if the group does not form, outputs exceed the tolerance, any request fails, or a non-`turbine-lab-*` container changes.
- [ ] [S-11] `scripts/lab-cluster.sh dp2-sparks` exits 0: it starts one node per Spark (HTTP :18000, control :18110 and data :18111 on 192.168.47.x, PSK from a file created in the run directory), waits for both `/turbine/v1/cluster` to list 2 alive members, runs `turbine-bench --url http://192.168.10.246:18000 --concurrency 16 --requests 256 --output json`, requires `requests_ok: 256` and `turbine_placement_decisions_total{target="remote"}` > 0 on dgx-spark, records throughput beside the single-node result in the task evidence, and leaves non-`turbine-lab-*` containers untouched; fails if the second node never receives traffic or any request fails.
- [ ] [S-7] [S-11] `scripts/lab-cluster.sh kv-remote-hit` exits 0: it sends a 6,000-token prompt to dgx-spark2 directly, then the same prefix plus a new suffix to dgx-spark with dgx-spark2's replica pinned as target by a lab-only request header allowed under the `fault-injection` feature, and asserts `turbine_kv_remote_decisions_total{reason="remote_retrieve"}` ≥ 1, `turbine_kv_fetch_bytes_total{peer="dgx-spark2"}` > 0, and the greedy output equal to a run with the directory disabled; fails if the remote prefix is never fetched or output changes.
- [ ] [S-8] [S-11] `scripts/lab-cluster.sh chaos-kill-node` exits 0: under `turbine-bench --url http://192.168.10.246:18000 --concurrency 16 --requests 400` it runs `docker kill turbine-lab-node-b` on dgx-spark2 after 100 completions, then restarts it; asserts dgx-spark's `/ready` stays 200 throughout, failed requests ≤ the number in flight on dgx-spark2 at the kill (from `turbine_forwarded_requests_total{outcome="worker_lost"}`), every failure is `worker_lost`, dgx-spark2 is `alive` with a new incarnation within 15 s of restart and receives traffic again; fails if an unaffected request fails, the survivor becomes unready, or the restarted node is not readmitted.
- [ ] [S-8] [S-11] `scripts/lab-cluster.sh chaos-partition` exits 0: with `fault-injection` builds it posts `{"peer":"dgx-spark2","action":"partition"}` to dgx-spark under load, asserts both nodes keep serving requests entering them with zero errors for locally served requests, then heals and asserts both list each other `alive` within `dead_after` + 3 s; fails if either node stops serving its own traffic.

## Open questions

<!-- None: decisions recorded in .procoder/ask/decisions.md -->
