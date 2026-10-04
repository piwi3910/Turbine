# Handoff: p6b-ladderperf — fix landed, A/B measured, B/C open for the user

Branch `p6b-ladderperf` at 81dfd76 (merge of `p6b-stack` 2c027ef + fix 81dfd76), gate `ok crates=all passed=938 failed=0`; remote
`target/debug` deleted after the commit. Worktree `.claude/worktrees/agent-p6b-ladderperf`. All lab work ran from the committed tree on
novanas GPU 0 under the port18000 and bench locks; no Job, server or lock of this branch is left (serve stopped by run id after every run;
last `kubectl -n turbine-ci get jobs`: none).

## What landed

- 81dfd76 `fix(kv)`: a demotion into a tier with no slot of its rung's size stores at the tier's format (`rung_no_slot`). `copy_codec`
  takes the rung only while `rung_has_slot` (new `KvTier::free_slots(format, bytes)` > copies in flight into the tier at that size);
  `make_room`, `demotion_bytes` and eviction are byte-accurate (`make_room` returns bytes; the old units arithmetic undercounted — 3 l0
  slots are 5 fp8 units but 6 were needed). Tests: the regression test, the in-flight test, the free_slots/`take_slot` consistency test,
  and the re-pinned kv_sim fixture `ladder_expected_rungs.json` (identical rung sequence, GREEN step-ups earlier because demotions
  complete; re-pin reason recorded with the pin). Mutation checks kill their tests: fallback bypass, in-flight subtraction, L1 empty-slab
  credit.
- Docs: spec S-6 line, metrics line and edge-case bullet (`.procoder/specs/phase-6b-kv-compression.md`), contract ladder entry
  (`.procoder/contract/interfaces.md`), `LadderReason` doc, `turbine-api` reason list. The drafted `.procoder/handoff/p6b-ladderperf.patch`
  is deleted with this commit — the fix it carried is 81dfd76, and a re-apply would clash.
- Perf log 6b section "Ladder rung-slot fallback": the mt3 A/B table, medians and the B/C assessment.

## The mt3 A/B (medians of 3, novanas GPU 0, mt2 recipe; `kv.transfer.promotion_copy: sdma` — `p6b-copykernel` is not merged)

| Arm | Run | ok  | recomputed tokens | cached ratio | later TTFT p50 / p99 (ms) | tok/s | ORANGE (s) | rung_no_slot | dem L0→L1 / L1→L2 |
| --- | --- | --- | ----------------- | ------------ | ------------------------- | ----- | ---------- | ------------ | ----------------- |
| on  | r1  | 192 | 192,637           | 0.8087       | 155.1 / 18,148            | 311.7 | 25.0       | 1,134        | 1,473 / 727       |
| on  | r2  | 192 | 172,543           | 0.8276       | 149.1 / 9,101             | 329.2 | 28.0       | 1,376        | 1,709 / 1,052     |
| on  | r3  | 192 | 148,242           | 0.8519       | 128.0 / 16,573            | 328.3 | 8.5        | 878          | 1,340 / 769       |
| off | r1  | 192 | 200,087           | 0.8003       | 143.5 / 13,256            | 324.2 | 22.0       | —            | 1,683 / 1,000     |
| off | r2  | 192 | 197,976           | 0.8047       | 141.8 / 12,259            | 306.5 | 22.0       | —            | 1,698 / 970       |
| off | r3  | 192 | 189,945           | 0.8109       | 145.6 / 10,247            | 319.2 | 22.0       | —            | 1,688 / 936       |

The mt2 regression is gone: on 243 → 328 tok/s, later-turn TTFT p99 40.5 → 16.6 s (off 12.3 s), ORANGE 118 → 25 samples, no
`queue_timeout`. Medians on against off: recomputed −13 %, cached ratio +2.3 pp, tok/s +3 %, ORANGE 25 against 22 s; all runs 192/192 ok.
The fallback does all the work: 878–1,376 `rung_no_slot` per run and almost every stored copy still `l0` at the end — a rung change alone
converts nothing on slot-sized slabs. Labbook set `phase-6b-kv-compression`, runs `p6b-ladderperf:mt3:llama:ladder-{on,off}:r{1,2,3}`,
all pass.

## Design question for the user — B or C, or stop?

With the fallback alone the lower tiers compress nothing: L1's 2 GiB and L2's 4 GiB hold 438 l0 blocks (146 + 292 at 14,680,064 B per
block); at the rung formats the same bytes hold 876 fp8 / 1,748 tq4 / 3,496 tq2 blocks, so B or C would multiply retained-block capacity
2–8× exactly where recompute comes from — the on-arm still recomputes 148k–193k tokens per run, nearly all of it blocks that overflowed
L1/L2 and were dropped. The ordering-only gains already measured (cached +2.3 pp, recompute −13 %, demotions −13 %) are what B/C would
amplify.

- B) Slab re-sizing: when a tier's rung changes, reformat slabs, spilling or evicting their copies. Buys conversion now, but does
  slab-by-slab eviction work per rung change and re-stores the displaced copies.
- C) Smaller slabs: slabs at compressed slot sizes so `take_slot` reformats naturally; no eviction, at the cost of more, smaller pinned
  allocations. Lower risk; the 10-minute soak already stored 41 tq4 copies into L2 under mixed formats.
- Recommendation recorded here and in the perf log: A landed; C over B unless compression must be realized on this workload soon. The user
  decides B vs C vs stop.
