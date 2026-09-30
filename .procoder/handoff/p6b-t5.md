# Handoff: p6b-t5 (Phase 6b Task 5, kernel level: ABI v2.11 `kv_transcode`, FP8 on HIP)

Branch `p6b-t5` from `p6b-stack` 254a0c7. Steps 1 to 3 of the task are done; the server step (4) and the
support-matrix flip (5) are NOT started: the t4+t15 merge had not landed on `p6b-stack` when steps 1 to 3
finished (`git log p6b-stack` shows no commit mentioning t4t15), and `crates/turbine-server/src/kv_orchestrator.rs`
belongs to that builder until then.

**Checkpoint: ready for the server step.**

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

## Left (step 4 and 5, blocked on the t4+t15 merge)

1. `git merge p6b-stack` once a commit mentioning t4t15 is on it.
2. `CopyStreamBackend` (`crates/turbine-server/src/kv_orchestrator.rs`): demotion = build the `pages` table from the
   pool (`KvPoolView.storage.slice(offset, len)` from each `block_segments` address, one `DeviceSlice` per layer),
   run `kv_transcode` (encode) into a device staging buffer, then the existing pinned copies of the coded bytes;
   promotion = pinned copy of the small bytes into the staging buffer, decode into the L0 pages. Staging =
   `DEMOTION_INFLIGHT` × the largest encoded block per shard, allocated at startup only when a lower-tier format
   is not `l0`. A `KvCodecFns` over `turbine_kv::codec` (see `KvCodecs` in `tests/hip_ops.rs`) is the CPU table. Task 3
   left the L1 copy stream refusing a non-identity codec and the host `HostCodec` path (L2) in place; the GPU path
   replaces the refusal. Transcode failure: the demotion is abandoned, the tier marked degraded (spec S-12).
3. `support_startup::kv_format_availability` stops refusing `fp8_e4m3` with `kv_transcode_unavailable` when the
   library resolves v2.11 (`ShimLibrary` → `kv_transcode` family present); `TIER_FORMAT_REFUSALS` (`turbine-core`
   `support.rs`) gets `fp8_e4m3` as `experimental`; update `docs/extending/kv-format.md` "Progressive gating".
4. Lab `kv_gpu::nvme_round_trip_fp8_tier` (written with the server step, since it cannot pass before it: L2
   `fp8_e4m3` from BF16 L0 within the codec bound, `l0` still bit-exact), then `scripts/lab-test.sh novanas -- -p
turbine-server --test kv_gpu`, `scripts/lab-test.sh novanas --tier quick`, `scripts/gate.sh`.
