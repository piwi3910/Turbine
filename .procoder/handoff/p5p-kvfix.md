# Handoff: p5p-kvfix (classed-pool partial reuse fix, S-14 close, lazy L0 page classes)

Branch `p5p-kvfix` from `p6b-stack` 59b62754. Not pushed. Nothing of this branch is left running on novanas.

## Commits

- 0fad9bb4 `fix(kv)`: stopgap — `kv_partial_reuse_classed_l0` (partial reuse + `kv.ladder.l0` on a GPU backend, exit 2
  before discovery). Gate ok 1000/0. Retired by the next commit.
- 418b742d `fix(kernels)`: the root cause. `kernels/rocm/src/copy_blocks.cpp` `page_offset` resolved every id of a classed
  pool through `page_classes`, which never lists the base class; a base page id (< `base_blocks`) got page size 0 and
  was rejected (`block id N is outside the pool's id space`). The copy-on-write tail copy of token-granular reuse is a
  base→base pair, so every partial-tail hit failed. Now ids below `base_blocks` sit at `b * block_bytes`, as in the cpu
  reference's `KvPageClasses::page_offset` (the cpu provider was always right; HIP only). No ABI change. The same bug
  hit `kv.dtype: tq4` (recent-window classed pool) partial reuse. Stopgap removed, contract notes the code retired.
  Tests (lab; both red before the fix with the exact served error): `hip_ops copy_blocks_classed_matches_cpu`,
  `kv_gpu partial_reuse_on_classed_l0_pool`. Gate ok 998/0.
- d26c458a `docs(5p)`: S-14 closed (decision A): spec S-14 text + target (recompute re-bound to > 0, measured −6.2 %),
  ACs rewritten (budget test dropped, lazy-classes test + measurement ACs added), plan Task 13 ticked, Task 13b added;
  `kv.ladder.l0=false` documented as the cheaper opt-in shape in AGENTS.md and `docs/extending/kv-format.md`.
  spec/plan check COMPLETE; gate ok (crates=none).
- cc85653a `perf(kv)`: lazy L0 page classes (decision B). `BlockPool::page_classes_active()`; `view()` flat until the
  first `allocate_in` of a class, then classed for good; decoder passes `block_formats` only on a classed view; engine
  fills per-block codes only once active. Test `turbine-kv pool::tests::page_classes_are_built_lazily` (red before).
  Gate ok 999/0.
- docs commit after cc85653a: perf log 5p "Task 13b", plan Task 13b + spec S-14 follow-up ACs ticked, AGENTS.md and
  `docs/extending/kv-format.md` updated for the lazy classes, this handoff.

## Lab (novanas GPU 0; measured from fixed scratch clones `scratchpad/kvfix-t2` = 418b742d, `kvfix-t4` = cc85653a)

- Quick tier on 418b742d and again on cc85653a: all pass, incl. `copy_blocks_classed_matches_cpu`,
  `l0_ladder_serves_on_gpu`, `partial_reuse_on_classed_l0_pool`, except `tiny_model hip_matches_cpu` /
  `hip_v23_library_matches_cpu` (step 1 max |Δ| 0.1028) — the known b1-attention mismatch (decisions "5p b1 attention
  breaks tiny_model hip_matches_cpu", user A1, not implemented on p6b-stack); not this branch's.
- Task 13b (perf log 5p "Task 13b"): standard bench 854.0 tok/s (1.002× S-2), golden c1 PASS. Exact Task 13 recipe
  (`kv.partial_prefix_reuse=false`), 3 + 3 interleaved: ITL p50 medians **24.4 (l0=true) vs 24.5 ms (l0=false)** —
  the +9 % is gone; tok/s 309.3 vs 326.5 and ORANGE 37 vs 36 samples, inside the workload's spread. The L0 step now
  fires in 2 of 3 on-runs (never in Task 13). Side pair with partial reuse live: 192/192, golden PASS on the classed
  pool (was HTTP 500).
- A first t13b on/off pair ran with partial reuse on (side data above); the script was stopped there and rerun as the
  exact recipe, so the comparison stays with Task 13. labbook set `phase-5p-serving-efficiency` has every run.

## Circuit breaker on one request's copy failure

By design today: `IterationError::model` maps every non-OOM, non-collective `ModelError` to
`CircuitEvent::DeviceError { sticky: false }`, which opens the circuit (Phase 3). A deterministic pre-launch argument
rejection (`TURBINE_E_ARGUMENT` from the library's own validation, no device work enqueued) is treated like a device
fault, so a bug that hits every warm request keeps re-opening the circuit after each probe. Options for the user (not
decided): A keep (a bug is loud); B classify `KernelError::InvalidArgument` raised before launch as a request failure
(fail the iteration's requests, no circuit event) with a metric/reason code; C B plus a per-reason rate limit that
still opens the circuit when argument errors repeat. Builder recommendation: C.
