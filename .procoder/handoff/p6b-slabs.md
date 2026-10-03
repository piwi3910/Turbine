# Handoff: p6b-t16-slabs — L1 slabs at 128 MiB landed and measured; mixing needs an empty slab

Branch `p6b-t16-slabs` at 4c62e82 (+ the lead's merge of `p6b-t17` as d6b526b on top), gate `ok crates=all passed=942 failed=0` at the
commit; remote `target/debug` deleted. Worktree `.claude/worktrees/agent-p6b-t2`. All lab work ran from the committed tree (binaries built
at commit 4c62e82, before the t17 merge) on novanas GPU 0 under the port18000 and bench locks; no server or lock of this branch is left
(run-one.sh stops its server and releases its locks on exit; last state: all six runs done, no leftover process, `/tmp/slabs-server.pid`
removed).

## What landed

- 4c62e82 `feat(kv)`: `kv.cpu.slab_bytes` (default 128 MiB, was the hard-coded `L1_SLAB_BYTES` 1 GiB). Validation: 0 refused, and a size
  below one L0-format block refused (exit 2, both with reason codes). TP warning uses the key (`max_bytes / world < slab_bytes`).
- Host AC (red first, mutation-checked both ways): `tier::tests::l1_default_slab_size_stores_a_new_format_block_beside_occupied_old_format_slots`
  — two slabs full of l0 blocks, evict one 128 MiB slab's worth, a tq4 block then stores beside the still-occupied l0 slots. At the old
  1 GiB default the eviction budget never frees a slab and the test fails with `Full`.
- Size arithmetic (Llama, 128-token blocks): a 128 MiB slab holds 9 l0 / 18 fp8 / 32 tq4 / 58 tq2 slots, worst slack 2.0 MiB (1.6 %);
  OLMoE l0/fp8 fit exactly (8/16). Docs: AGENTS.md, spec S-5 + key table, contract §, lab config comment.
- Perf log 6b section "L1 slab size 128 MiB": the mt3 A/B table, medians and the reading below.
- Labbook set `phase-6b-kv-compression`, runs `p6b-slabs:mt3:llama:slab{1g,128}:r{1,2,3}` (type `turbine-multi-turn`).
- 6493c16 `docs(perf)`: the perf-log entry and this handoff; gate `ok crates=all passed=944 failed=0` on the merged tree.

## The A/B (medians of 3, mt3 recipe, ladder on, L2 untouched at 1 GiB slabs)

tok/s 307.9 (128 MiB) vs 291.2 (1 GiB), +5.7 %; later-turn TTFT p99 11.0 vs 17.3 s; TTFT p50 147 vs 131 ms; cached ratio 0.791 vs 0.840
(−4.9 pp); recomputed tokens 208,785 vs 160,262 (+30 %). All runs 192/192 ok.

Reading:

- Mixing never triggered in the lab, either arm: L1 ends 100 % l0 (146 vs 144 blocks — the 2-block difference is slack) because the tier
  never frees a whole slab under load and `take_slot` reformats only an empty slab. `no_room` for fp8 into L1 is similar in both arms
  (3–30), `no_room_backoff` exactly once per run in both. The mixing gain is host-test-proven but this workload never reaches the state.
- Run-to-run ladder dynamics dominate at this workload (the historical 1 GiB spread spans recompute 148k–193k, cached 0.809–0.852,
  tok/s 311–329); 3 runs per arm cannot separate a −4.9 pp cached / +30 % recompute shift from that noise. Numbers do not argue for a
  second size change — 128 MiB stays the default (better tail, no throughput loss, finer reformat granularity).
- L2: nothing here argues for following; the decision named L1 only.

## For the lead

- Transient: the first 1 GiB run crashed mid-bench with `engine thread panicked: release of unreferenced KV block BlockId(146)` (fatal
  exit 3, bench 153 ok/39 failed). Not reproduced in 6 subsequent full runs on either arm (0 panics in every server log). One-off on the
  pre-merge p6b-stack ladder path; handing it to whoever owns the ladder release path (t17 territory).
- Uncommitted in this worktree, not mine: `.procoder/ask/decisions.md` (+14 lines, the 6b Task 17 window decision entry) — left as found.
- Harness kept on novanas for re-runs: `/home/piwi/slabs/{run-one.sh,driver.sh,summarize.py,summary.json}`, artifacts
  `/home/piwi/slabs/runs/<arm>-r<n>.{bench,kv,metrics,samples}.*`, server logs `/tmp/slabs-server-<arm>-<n>.log`.
