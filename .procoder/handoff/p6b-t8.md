# Handoff: p6b-t8 (Phase 6b Task 8, TurboQuant GPU transcode)

Branch `p6b-t8` from 0bae895 (the Task 5 kernel commit merged with `p6b-stack`). Task 8 is done at the kernel level;
the server upload of the tables is the remaining item (not part of Task 8, lead's scope note).

**Checkpoint: done — ready to merge into `p6b-stack`.**

## Commits

- c3839c9 `docs(decisions)`: provider evaluation "P6b: TurboQuant transcode — provider evaluation (kernel reuse
  rule)" (vLLM Triton/FlyDSL and a different codec, SGLang generic CUDA FWHT, CK nothing; pick: own `turbine_hip_tq`
  with llama.cpp's wave-shuffle FWHT pattern) and the first version of this handoff (stopped on the ABI question).
- 1abac18 (lead): user decision A — `const turbine_tq_params *tq_params` at the end of `turbine_kv_transcode_desc`.
- 2b1f46f `test(kernels)`: the ABI field and the Rust plumbing, plus the failing lab cases.
- feb5d83 `feat(rocm): TurboQuant KV transcode`: `kernels/rocm/src/kv_transcode_tq.hip`, the `impl_table.cpp` entry,
  CMake, the result paragraph of the decisions entry.

## What was built

- Header (still minor 11): `turbine_tq_params { seed, codebooks[4], tables }` (device F32; `tables` =
  `[layers][num_kv_heads][2·128 + 128²]`: K signs, V signs, QJL `S` row-major), trailing `tq_params` in the transcode
  descriptor (required by TQ4/TQ2, ignored by FP8, NULL in probes). Contract §9.1 / §26 updated.
- Rust (`turbine-kernels`), source-compatible for FP8 callers: `KvTranscodeContext` is unchanged; new
  `KvTranscodeTables { codebooks: [TensorView; 4], tables: TensorView }` (`head_elems`) and
  `KvTranscodeKernel::execute_with_tables(ctx, Option<&KvTranscodeTables>)` (default method: `execute` when `None`).
  The shim refuses a TurboQuant call without tables and checks their shapes; `ffi::TqParamsDesc`; stub
  `stub_desc_size(20)`. The cpu-reference provider ignores the tables.
- HIP `turbine_hip_tq`: BF16 pages, head_dim 128, one workgroup per (KV head, layer, block), 32-token chunks in LDS,
  `#pragma clang fp contract(off)`; refuses tables of another seed, unaligned tables, NULL codebooks. FP8 pages are
  refused (the codec supports them on the CPU; not built here).

## Evidence

- Red: `turbine-lab-test-0930193434-121fdc52` (tq4 unsupported). Green: `turbine-lab-test-0930194110-3ffc00fb` —
  tq4 / tq2 at Llama, OLMoE and an odd shape: encode byte-exact (**0 tie bytes**, counted per field), decode bit-exact;
  cpu provider agrees.
- Timings (32 blocks, Llama): tq4 encode 20,473 µs, decode 8,116 µs; tq2 17,328 / 7,889 µs (FP8 2,390 / 1,555).
- Mutation (snapshot / restore, not committed): the encode's K rotation reading the V signs → lab `kv_transcode_matches_cpu` FAILS (`tq4 llama-3.2-3b: encode differs from the codec in 14646388 bytes`, job `turbine-lab-test-0930201928-1dd65272`).
- `scripts/gate.sh`: `gate: ok crates=all passed=857 failed=0` on both commits.
- `scripts/lab-test.sh novanas --tier quick` (`turbine-lab-test-0930194331-114c0002`): every target passes except
  `turbine-model --test tiny_model`, which crashes (SIGSEGV) in `hip_decode_graph_matches_eager` with hipBLASLt
  "operation would make the legacy stream depend on a capturing blocking stream". Reproduced on this branch
  (`turbine-lab-test-0930201409-3203b425`) **and on the base 0bae895 without any Task 8 change**
  (`turbine-lab-test-0930201801-10529cff`), so it predates Task 8 (known intermittent in
  `.procoder/review-2026-09-29.md`, now reproducible on the 6b stack). Not investigated further here.

## Left

1. Server: build the tables once at startup from `turbine_kv::codec::turboquant` (`hadamard::rademacher`,
   `qjl::projection`, `codebook::codebook`, as `hip_ops::TqTables` and `kv_tq::layer_params` do) for the namespace
   seed, upload them (≈ 15 MiB Llama, ≈ 17 MiB OLMoE per rank), and call `execute_with_tables` in `CopyStreamBackend`
   for `tq4` / `tq2` tiers; then the support-matrix / `TIER_FORMAT_REFUSALS` step for `tq4` / `tq2` on `amd`.
2. The `tiny_model` hipBLASLt capture SIGSEGV on the 6b stack (pre-existing; needs an owner).
3. Encode speed: 0.64 ms a block bounds demotion (≈ 4× the link time of the coded bytes); optimise the F64 norm loops
   and the `S · r` products if Task 9's lab proof shows demotion lagging.
4. FP8 L0 pages under a TurboQuant lower tier are refused by `turbine_hip_tq` (`supports` false).
