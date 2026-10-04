# Handoff: p6b-copyahead (decision "6b: shared-prefix eval never demotes the prefix to L1", A; amended plan Task 6)

Branch `p6b-copyahead` from `p6b-stack` e02e8c2. `a8c3b7c` passed `scripts/gate.sh --base e02e8c2` → `gate: ok` (769 passed). The later
commits change only eval JSONs and docs. Numbers are in `.procoder/perf-log.md`, Phase 6b, "Copy ahead and the shared-prefix FP8 gate".

## Commits

- `a8c3b7c` `feat(kv)`: copy ahead in `turbine-kv` (`hierarchy.rs`; `directory.rs` `has_child_in`; `metrics.rs`). It applies at GREEN and
  YELLOW, when capacity demotion (`demote_to(.., Capacity)`) or `pressure_reclaim` would demote. The blocks it takes belong to a shared
  prefix (≥ 2 children, or an ancestor of such a block) and are unreferenced, with reuse evidence, a child resident in L0 and the L0 copy
  as their only one. Each is copied into the L0 demotion target while its L0 copy stays: lowest score first, within `DEMOTION_INFLIGHT`,
  only into free room up to `COPY_AHEAD_MAX_FILL` (0.5, a constant with a debt note), with nothing evicted for it. The L0 copy then leaves
  by the Phase 4 rules through the existing "already copied down" paths, with no further copy. A copy into a lossy tier is one more location
  on the exact entry (S-3). `Demoting.keep_source` marks a copy ahead, and `l0_leaving` keeps copies ahead out of the reclaim targets.
  Metric `turbine_kv_copy_ahead_total{to}`, `KvStats.copy_aheads`, log event `kv_copy_ahead` (reason `copy_ahead`). The P4 S-7 leaf-first
  rule is unchanged, and there is no config key or `turbine-server` edit. Spec 6b S-8 sentence, Interfaces and AC, and the contract were
  updated.
- `565723b` `test(eval)`: the first gate pair, superseded by `7ebf9b8`; its BF16 server served foreign soak traffic.
- `40cad6e` docs (first perf log entry and recipe), rewritten by the last docs commit.
- `7ebf9b8` `test(eval)`: the clean pair. `turbine-bf16-sp.json` 0.780 and `turbine-l1-fp8-sp.json` 0.770 (lossy ratio 0.927).
  `eval-compare --max-drop 0.01` PASSes at the bound.
- Last commit: perf log, recipe (`p6b-eval-prefix.md`: 32 fillers, prompt-token check), this handoff.

## Tests and mutations

`hierarchy::tests::{copy_ahead_keeps_a_shared_prefix_in_l0, copy_ahead_into_a_lossy_tier_keeps_the_exact_entry,
copy_ahead_needs_a_lower_tier, copy_ahead_skips_unshared_chains, copied_ahead_prefix_leaves_l0_once_its_children_are_reclaimed}`. Each of
these was red before the change. Each of these mutations made a test FAIL:

- the landed copy frees L0;
- copy ahead is disabled;
- pressure reclaim's free path copies again;
- capacity demotion's free path copies again;
- already-copied blocks are copied ahead again;
- no ORANGE guard;
- no shared-prefix filter.

The kv_sim tests and the pinned ladder fixture are unchanged and pass. A first cut (copy every evidence parent at any pressure) broke
`demotion_under_pressure` (4 extra copies ahead into L2 at ORANGE) and moved the ladder fixture: 137,376 recomputed tokens with the ladder
on, against 131,808. Linear session chains' copies were held in L1 by leaf-first. That cut was not committed. The ORANGE guard and the
shared-prefix filter are the fix.

## Eval (Task 6 gate), runs outside the 11:00–11:18 UTC soak window

| Arm             | Serve run           | Accuracy | Lossy ratio | Prompt tokens served |
| --------------- | ------------------- | -------- | ----------- | -------------------- |
| BF16            | 1001114618-27e26a5b | 0.780    | —           | 726,723              |
| FP8 L1          | 1001114752-39f57ca5 | 0.770    | 0.927       | 726,723              |
| FP8 L1          | 1001111446-1e7e5c52 | 0.775    | 0.927       | 726,723              |
| FP8 L1, e02e8c2 | 1001114920-1b5634b9 | 0.775    | 0.927       | 726,723              |

The gate (the first two rows) is a PASS at drop 0.010. Paired, 8 items flip one way and 6 the other.

The base commit also demotes the prefix with 32 fillers, so planner2's 0 lossy tokens came from too few fillers. Copy ahead makes that L0
drop free and starts the copy at GREEN; the gate itself does not depend on it.

Discarded as soak-contaminated: 1001105753-3afb91e2 (2.94M prompt tokens), 1001110311-3d1b4cef and 1001110705-21e8b128 (503 / RED, 4.9M
and 4.5M), 1001111055-19904604 (1.77M).

## Lab checks

- `scripts/lab-bench.sh --quick --model llama -- --set kv.cpu.enabled=true --set kv.cpu.max_bytes=4GiB` (40cad6e, GPU 0): golden c1 PASS,
  877.1 tok/s, ITL p50 15.4 ms, TTFT p50 246 ms. de6f948 measured 877.3 / 15.4 / 247: no regression.
- `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu` (job 1001114417-1a1a26b3): 4 passed, 6 failed. Every failure is the
  same host precondition: `kv.nvme.max_bytes: 64GiB exceeds the free space of /home/piwi/turbine-kv (≈63 GiB) minus 10 %`, so the server
  exits before `/ready`. The four that passed include `lossy_tier_reuse` (L1 fp8). Not rerun: freeing disk needs the user.

## Open

- novanas `/home/piwi/turbine-kv` has about 63 GiB free, below the 64 GiB plus 10 % the Phase 4 lab config needs. That blocks the kv_gpu
  NVMe and prefix tests (user: free disk, or a smaller `kv.nvme.max_bytes` in `scripts/lab/phase4-novanas.yaml`).
- The FP8 gate passes exactly at 0.01. A second clean fp8 run scored 0.775. For more margin: run more pairs, or use the full GSM8K
  variant.
- `COPY_AHEAD_MAX_FILL` is a constant, to be re-measured on a real shared-system-prompt workload.
- The TQ and ladder `-sp` gates should use the same 32 fillers against `turbine-bf16-sp.json`.

The eval client (`bin/turbine-golden`, copied from `agent-p6b-planner2`) and the report copies are under
`/home/piwi/turbine-ci/remote/agent-p6b-copyahead/bin/` on novanas. Every serve Job of this branch was stopped by its run id.
