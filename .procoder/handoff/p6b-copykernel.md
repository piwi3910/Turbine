# Handoff: p6b-copykernel (decision "6b: KV promotions slow decode — which fix", A)

Branch `p6b-copykernel` from `p6b-stack` 6652755. Numbers are in `.procoder/perf-log.md`, Phase 6b, "Promotion copy kernel".
Not pushed.

## Done

- Microbenchmark `kernels/rocm/tools/copy_eval.cpp` (CMake target `turbine_copy_eval`, a lab tool). The stall comes from host-link
  reads in flight, not from the SDMA engine. A copy kernel stalls compute in proportion to its workgroups: 16 workgroups stall as
  much as SDMA, 2 workgroups give 22 GB/s (SDMA 12.4) with the slope 17× lower. Smaller SDMA segments, the blit kernels
  (`HSA_ENABLE_SDMA=0`, `DEBUG_CLR_LIMIT_BLIT_WG`) and `HSA_ENABLE_SDMA_HDP_FLUSH=0` do not help.
- 3562cfb `feat(kv)`: an optional copy path. Kernel ABI v2.11 `turbine_copy_seg` + `turbine_memcpy_h2d_kernel` (optional symbol;
  v2.11 has not shipped, so no minor bump), HIP `kernels/rocm/src/copy_kernel.hip` (2 workgroups by default, bit-exact for any
  length and alignment). `CopyEngine::{has_copy_kernel, copy_async_batch_kernel}` (defaults: false / `Unsupported`) implemented
  by `ShimContext` (pinned → device batches only, one ticket, no compute fence). Config `kv.transfer.promotion_copy: sdma|kernel`
  (default `sdma`). Startup refusal `promotion_copy_kernel_unavailable`, log `event="kv_promotion_copy"`. Contract entries are
  added (config table, v2.11 paragraph).
- Hunks in `kv_orchestrator.rs` (promotion copy path only, for the ladder builder): the import `PromotionCopy`, the field
  `kernel_promotions` on `CopyStreamBackend` (and its `false` in `new`), `set_promotion_copy` (after `set_metrics`), the
  `to_device && self.kernel_promotions` branch in `stream_copies`, the `set_promotion_copy` call in `KvOrchestrator` start (after
  `set_metrics`), the test `promotions_use_the_copy_kernel_when_configured` and the `FakeCtx` fields `kernel` / `kernel_batches`
  and its two trait methods. In `engine/tp_tiers.rs`, the worker backend calls `set_promotion_copy` (3 lines).
- Tests: `turbine-kernels shim::tests::a_kernel_copy_batch_is_one_kernel_call` (stub V211), `abi_header_neutral
header_declares_the_v211_copy_kernel`, `turbine-core config::tests::kv_config_validation`, `turbine-server
kv_orchestrator::tests::promotions_use_the_copy_kernel_when_configured`, lab `turbine-kernels --test lab
pinned_copy_kernel_is_bit_exact` (passes; GPU 0: 23.79 GB/s kernel against 11.67 GB/s SDMA). Mutations: `stream_copies`
  ignores the switch, the batch goes to the copy engine, the H2D-only check is dropped, the startup refusal is dropped (each fails
  its test); a shim tail off by one byte fails the lab test on GPU 0. Gate `scripts/gate.sh --base 6652755`: `gate: ok
passed=924`.
- Lab A/B (OLMoE `l0` multi-turn, 5 runs per arm): the slope goes from 0.064 to 0.000 ms/MiB, the extra time per overlapped step
  from 6.9 to 1.0 ms, and the steps ≥ 1.5× from 17 to 3. Promotion median goes from 30 to 20 ms. Reuse and tok/s are unchanged
  (median 246.7 / 246.4), and later-turn TTFT p99 goes from 360 to 288 ms. Two kernel runs show p99 copy bounds of 276–293 ms.
  They are poll bounds across idle engine time or host-side step gaps, and are not decided by the traces (perf log).

## Open (user's call)

1. Flip the default `kv.transfer.promotion_copy` to `kernel`? The numbers support it. Before flipping: a Llama A/B and a
   `lab-bench --golden16` with the switch on.
2. The p99 outliers: event timestamps on the copy stream would give exact copy times (an ABI addition such as
   `hipEventElapsedTime` between two events), or a rocprofv3 capture of the kernel arm.
3. Demotions (D2H, SDMA) were not measured for the same stall. A device → pinned copy kernel would be the same shape.

Results and scripts on novanas: `scratch/copykernel-ab/` (`ab.sh`, `runall.sh`, `runmore.sh`, `corr3.py`, `summ.py`,
`slow.py`, `stall.py`, `gaps.py`; dirs `sdma`, `kernel`). The microbenchmark scratch dir was removed. The remote workspace
`remote/agent-p6b-copykernel/` holds `kbuild/` and the release binaries of 314c848.
