# Handoff: p6b-graph-segv (decode-graph capture crash, user decision A 2026-10-01)

Branch `p6b-graph-segv` from `p6b-stack` 7e32f4f.

**Checkpoint: done — ready to merge into `p6b-stack`.**

## Root cause

`tiny_model hip_decode_graph_matches_eager` did not fail on its own: it failed when **another test thread created a
HIP context while it was capturing a decode graph**.

- HIP clr at ROCm 7.14.1 (`hipamd/src/hip_internal.hpp`, `CHECK_STREAM_CAPTURING`): the synchronous APIs (`hipMemset`,
  `hipMemcpy`, …) fail with `hipErrorStreamCaptureImplicit` (906, "operation would make the legacy stream depend on a
  capturing blocking stream") while **any** stream of the process captures, "for any capture mode (Global,
  ThreadLocal, or Relaxed)", and they **invalidate every capture in progress**. Our capture is thread-local on a
  non-blocking stream, which by CUDA semantics should not be affected; HIP is stricter.
- `hipblasLtCreate` (hipBLASLt 1.4.1, `therock-7.14.1` `hipblaslt.cpp:164-165`) does `hipMalloc` + a synchronous
  `hipMemset` of its synchronizer inside `CHECK_HIP_ERROR`, which prints `Hip error: … at hipblaslt.cpp:165` and calls
  `exit(EXIT_FAILURE)`. That is the exact line in every failing log.
- `turbine_ctx_create` calls `hipblasLtCreate`, so a context opened on thread B during a capture on thread A:
  invalidates A's capture (A sees "hipExtModuleLaunchKernel failed … previous error during capture", the other
  message seen in 6a), and exits the process with status 1 from B; other threads still in HIP while `exit` runs the
  static destructors give the SIGSEGV. The test binary never prints a summary.
- In `tiny_model`, `hip_ep2_matches_ep1`, `hip_pp2_matches_pp1` and `hip_tp2_matches_tp1` open contexts inside their
  loops, i.e. late, while `hip_decode_graph_matches_eager` / `hip_matches_cpu` capture. Whether the two overlap is
  pure timing: the 6b stack only changed timing (new CPU-heavy tests and more code in the same binary), no GPU code in
  the capture path. No bisect was needed once the mechanism was reproduced deterministically.

## Evidence

- Red: `turbine-lab-test-0930203603-2f8cf1f0` (detached checkout of bb61b31, the test without the fix):
  `turbine-kernels --test lab context_created_during_another_threads_capture_waits` dies with the same `Hip
error … (906) at … hipblaslt.cpp:165`, exit status 1; in the same job `tiny_model hip_decode_graph_matches_eager`
  alone passes (12 cases, `capture_failed: 0`).
- Green: `turbine-lab-test-0930210837-2aac1d40` (67fc7ef): all of `tiny_model` (38 passed, the two hostmem two-GPU
  tests filtered as always) and `turbine-kernels --test lab` (4 passed).
- Quick tier: `turbine-lab-test-0930210944-2b27df3b` (`--tier quick`, 67fc7ef): PASS, 913 passed, 0 failed, no `Hip error`.
- Gate `scripts/gate.sh`: ok, crates=all, 857 passed.

## Commits

- bb61b31 `test(kernels)`: `crates/turbine-kernels/tests/lab.rs`
  `context_created_during_another_threads_capture_waits` — thread A begins a capture and records a D2D copy, thread B
  creates a second context on the library, A ends the capture after 500 ms; asserts the capture survives, B's context
  was created only after the capture ended, and the replay copies the data. Without the fix the binary exits 1.
- 67fc7ef `fix(rocm)`: process-wide capture gate in the shim (`graph.cpp`, `turbine_hip.hpp`, `context.cpp`):
  - `turbine_graph_begin` registers the capture (its thread) before `hipStreamBeginCapture`, waiting while any
    context creation is in progress; the entry leaves when the capture ends (`turbine_graph_end`, after
    `hipStreamEndCapture`, or `abandon_capture`), or when begin fails.
  - `turbine_ctx_create` holds a `CreationGuard` from before the stream / hipBLASLt / workspace creation to the end:
    it waits until no capture is open and blocks new ones. A thread with its own capture open gets
    `TURBINE_E_ARGUMENT` instead of deadlocking.
  - The gate object is heap-allocated once and never freed (no destroyed mutex at process exit).
  - `turbine_kernels.h` (graph section) states the rule vendor-neutrally; no ABI change, no minor bump.

## Left / open

- Other synchronous HIP calls from another thread during a capture would still break it (HIP invalidates every
  capture). The shim itself makes none outside `turbine_ctx_create` (all copies are `hipMemcpyAsync` on our streams),
  but RCCL init (`ncclCommInitRank`) or another library in the process could; today TP ranks create contexts and
  communicators before any capture. Worth a line in the ROCm upstream report.
- Upstream (not filed; draft for the user): (1) clr: `CHECK_STREAM_CAPTURING` rejects sync APIs and invalidates
  captures of other threads even in `hipStreamCaptureModeThreadLocal` on a non-blocking stream, unlike CUDA; the error
  text mentions a blocking stream although none is involved. (2) hipBLASLt: `hipblasLtCreate` uses a synchronous
  `hipMemset` and `exit(1)`s on any HIP error instead of returning a status.
- `turbine-device --test lab` SIGSEGV at process exit (6a, once): not investigated; that binary creates no graph, so
  it is probably not this bug.
