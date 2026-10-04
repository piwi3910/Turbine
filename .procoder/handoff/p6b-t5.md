# Handoff: p6b-t5 (Phase 6b Task 5: ABI v2.11 `kv_transcode`, FP8 on HIP and on the demotion path)

Branch `p6b-t5` from `p6b-stack` 254a0c7, merged with `p6b-stack` twice (t4t15 at 403ed0b, then 71574a6).
Steps 1 to 5 are done: kernel level (first half of this file), then the server side (second half).

## Done

- Provider evaluation, decisions entry "P6b: KV transcode — provider evaluation (kernel reuse rule)": CK `ck_tile`
  `elementwise`, `batched_transpose` and `reduce` judged on their headers (one strided tensor per call, no scattered
  page gather, no absmax feeding a scale, hardware e4m3 conversion not pinned to the codec); pick: an own kernel,
  reusing the bit-exact `fp8_e4m3_round` / `fp8_e4m3_value` of `qgemm_quantize.hpp`. Timing recorded (below).
- ABI v2.11 group (`TURBINE_ABI_MINOR` 11, op 19): `turbine_kv_transcode_desc` + trio in `kernels/include/turbine_kernels.h`.
  **Descriptor as built** (the spec's interface line was amended to match): one descriptor for both directions,
  `pages` (host array `[num_blocks × layers]` of device page addresses, block-major, read during the call),
  `k_scales` / `v_scales` (device F32 `[layers]` of FP8 pages, NULL = 1.0), `coded` (num_blocks slots),
  `coded_block_bytes`, `seed`, `num_blocks`, `layers`, `block_tokens`, `num_kv_heads`, `head_dim`, `page_dtype`,
  `format` (`TURBINE_KVFMT_*`), `direction`. The spec's `src_pages` / `dst` / `dst_format` / `src_dtype` are
  `pages` / `coded` / `format` / `page_dtype`. Task 8 (TurboQuant) adds codebook pointers and the tq slots.
- Rust (`turbine-kernels`): `OpKind::KvTranscode`, `KvTranscodeFormat`, `KvTranscodeDirection`, `KvTranscodeConfig`
  (`direction()`, `codec()`, `page_bytes()`), `KvTranscodeContext { cfg, pages: &[DeviceSlice] (block-major), coded,
coded_block_bytes, seed, k_scales, v_scales, codecs: &dyn KvCodecFns }`, `trait KvCodecFns { encode, decode }`,
  `KvTranscodeKernel`, `KernelProvider::kv_transcode`, `KernelRegistry::kv_transcode`, `OpConfig::KvTranscode`,
  `ffi::KvTranscodeDesc`, `V21Symbols.kv_transcode` (minor ≥ 11 and the v2.9 and v2.5 groups resolved, whole trio or
  nothing), shim provider (validates page count, page size, slot size, device ownership of every slice before the
  library is called), CPU provider `cpu/kv_transcode.rs` (`cpu_kv_transcode_ref`, runs the codec table the caller
  passes in, so `turbine-kernels` stays free of `turbine-kv`).
- HIP: `kernels/rocm/src/kv_transcode.hip` (+ `.hpp`), implementation `turbine_hip_fp8` in `impl_table.cpp`,
  `abi_minor.cpp` reports 11, CMake source. One workgroup per (block, layer, K or V): absmax, scale, encode in one
  launch; the page table goes up through a 4-slot pinned upload ring (`KvTranscodeScratch`, waits only for the call
  four calls earlier). FP8 pages are refused by `supports` (the copy path: a slot is the page). The TurboQuant slots
  (Task 8) go in `kv_transcode_tq.hip` behind their own `ImplEntry`; `kv_transcode.hpp` and the table upload are shared.
- Stubs `TURBINE_STUB_GFX942_V211`, `_V211_PARTIAL` (no `_impl` symbol), `_V211_NOQUANT` (minor 11, no v2.9 group).
- Contract: §9.1 v2.11 paragraph and a §26 entry. Plan Task 5's `abi_minor.cpp (10)` corrected to 11; spec S-1
  Interfaces line corrected (minor ≥ 11, descriptor as built).

## Evidence

- Host: `scripts/gate.sh` → `gate: ok crates=all passed=845 failed=0`. New tests: `ffi::tests::optional_groups_v211`,
  `cpu::kv_transcode::tests::encodes_and_decodes_block_by_block`, `abi_header_neutral
header_declares_the_v211_kv_transcode_group`, `ops::tests::op_minor_revisions` (extended).
- Lab `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- kv_transcode`: job
  `turbine-lab-test-0930171713-1c3059d1`, `kv_transcode_matches_cpu` PASS (Llama, OLMoE and an odd shape; encode
  byte-exact and decode bit-exact against `turbine_kv::codec::fp8_e4m3`; the cpu-reference provider over the same
  table agrees). Timing, 32 blocks at the Llama shape (448 MiB of pages): encode 2382 µs (197 GB/s), decode 1554 µs
  (302 GB/s).
- Mutation checks (snapshot / restore, not committed): dropping the scale floor in the kernel → lab
  `kv_transcode_matches_cpu` FAILS (`encoded block 1 differs from the codec at byte Some(0)`, job
  `turbine-lab-test-0930172925-30ac0776`); making `V21Symbols::resolve` ignore the minor and the v2.9 / v2.5 groups →
  `optional_groups_v211` FAILS.
- Test-first note: the Rust types the tests name did not exist, so the tests and the plumbing were written together
  (a compile failure is the red state); the two mutation checks above are the evidence that the tests catch a break.
- `scripts/lab-test.sh novanas --tier quick`: PASS, job `turbine-lab-test-0930173016-0ad31d4b` (whole workspace, slow
  perf tests skipped).

## Server side (step 4 and 5)

- `crates/turbine-server/src/kv_orchestrator.rs`:
  - `DeviceTranscode` (kernel provider + copy engine + device staging): `DEMOTION_INFLIGHT` slots of the largest
    encoded block among the served tier formats, allocated only when a configured tier format is one the library
    serves in both directions (`fp8_e4m3` over BF16 pages today), else nothing is allocated. Switched on by
    `KvOrchestrator::enable_device_transcode` (engine thread, after `start`; the cpu backend passes the
    cpu-reference provider). `CodecTable` is the `KvCodecFns` over `turbine_kv::codec`.
  - Demotion (L0 to L1 or L2, non-identity codec): encode kernel into a slot, D2H of the small bytes into the pinned
    staging buffer (ordered after the kernel by `copy_async`'s compute wait), slot freed, then the I/O pool stores
    `to_bytes` of them (`HostTranscode.pre_encoded`). Promotion: the pool reads `from_bytes` into the staging
    buffer, H2D into a slot, decode kernel into the L0 pages, a compute fence (`CopyEngine::fence_compute`, new;
    `ShimContext` implements it with an event on the compute stream) frees the slot once the kernel has read it.
    L1 non-identity copies now take the same staging/I-O path as L2 (the old "copy stream moves l0 blocks only"
    refusal is gone); a copy the device cannot serve (another codec, no free slot) uses the host codec as before.
    A transcode failure abandons the copy, frees slot and buffers and counts an L1 copy error.
  - Copies are timed by the backends (decision "6b: production KV copy backends time copies to the polling
    boundary", A): `IoPoolBackend` and `CopyStreamBackend` implement `TransferBackend::took` from a start time and
    the I/O thread's own finish time (`IoDone.finished`). A copy that ends on the copy stream is timed to the poll
    that sees its event complete (the ABI has no event timestamps), and the stage hops of a staged copy still happen
    at polls: the bias is gone for I/O, reduced for staged copies, unchanged for pure stream copies.
  - `KvTranscodeContext.pages` is `&[DevicePtr]` now (the pool exposes addresses, not `DeviceSlice`s), documented
    like `copy_async`'s addresses; the `coded` slice is still owner-checked.
- Startup: `support_startup::kv_format_availability(cfg, library: Option<bool>)` runs before the library is loaded
  (`None`) and again in `startup.rs` once the provider is prepared (`Some(ShimLibrary::kv_transcode())`; cpu
  reference = true); a lower tier needing the transcode on a library without it exits 1
  `kv_transcode_unavailable`. `kv.ladder.enabled` stays refused (its rewrites still run on the host codec).
  `TIER_FORMAT_REFUSALS`: `fp8_e4m3` `experimental` (WARN `support_matrix` at startup).
- Tests: `kv_orchestrator::tests::device_transcode_matches_the_host_codec_through_l1_and_l2` (host/device paths store
  and promote the same bytes through L1 and L2, all slots back; mutation: not returning the slot on promotion fails
  it), `copies_are_timed_to_their_completion_not_to_the_poll` (mutations: stamping the I/O finish at the poll, and
  stamping the pool's own timing at the poll, each fail it), `support_startup::tests::tier_formats_and_availability`,
  `core support::tests::baseline_rows_present`, lab `kv_gpu::nvme_round_trip_fp8_tier`.
- The FakeCtx test stand-in now has a lock per pinned buffer (it held one lock across the closure, which deadlocked
  an I/O write into an L1 slot; the kernel library already locks per buffer).

## Evidence (server side)

- `scripts/gate.sh`: `gate: ok crates=all passed=859 failed=0`. `scripts/lab-test.sh novanas --tier quick`: PASS, job
  `turbine-lab-test-0930204040-223f1dd3`.
- Lab `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu`: job `turbine-lab-test-0930203108-3bd31291`
  PASS, 8 tests including `nvme_round_trip_fp8_tier` (L2 holds 1.23 GB for 156 demoted blocks of 14.7 MB: the
  blocks are encoded; A answers within the bound after its prefetch, worst first-8 logprob difference 0.000) and
  both `nvme_round_trip_matches_cold` variants (bit-exact, `l0`).
- Two earlier full runs hung one test each on a 300 s HTTP read timeout (`WouldBlock` in `request`): job
  `turbine-lab-test-0930191933-2d68e07e` (`nvme_round_trip_matches_cold_fp8_kv`, a test that uses no lower-tier
  format) and `turbine-lab-test-0930202024-378e678b` (`nvme_round_trip_fp8_tier`, at the prefetch); the same
  tests pass alone and in the two other full runs (`...0930193806-34574682`, `...0930203108-3bd31291`). Other
  Turbine jobs were running on the node at those times. Not reproduced and not explained; if it recurs, take a
  thread dump of the server before the read timeout.

## Open

1. A fp8 lower tier does not get reused by a lookup after a prefetch: `POST /turbine/v1/kv/prefetch` promotes the
   lossy blocks into L0 under their own `lossy_key` entries, but the exact chain still finds the entries' lossy L2
   locations first, and for three blocks the planner then recomputes (`recompute_cheaper`); the decoded L0 copies
   are not consulted. That is the lossy lineage of Task 4 meeting the planner (Task 6 writes the lab test
   `lossy_tier_reuse` for it), so `nvme_round_trip_fp8_tier` checks encoded size, promotion without a copy or
   checksum error, and the answer's logprobs, not a reused prefix. Options for Task 6: prefer the L0 lossy copy in
   the lookup, or make the prefetch also add the L0 location to the exact entry.
2. The staging slots are allocated after the memory budget is fixed (`DEMOTION_INFLIGHT` x 7 MiB for Llama FP8,
   235 MiB): they are not in the reliability ledger. A failed allocation falls back to the host codec with a WARN.
3. The ladder (`kv.ladder.enabled`) still refuses at startup; its `Compress` rewrites are host-codec I/O jobs.
4. Tensor or pipeline parallelism still refuses a lower-tier format other than `l0` (one shard per block).
