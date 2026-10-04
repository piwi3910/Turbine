# Handoff: p6b-greenhead (GREEN admission headroom), paused

Branch `p6b-greenhead` from `p6b-stack` 202bd4a; decision "6b: which cap the GREEN admission headroom uses" (A): RED threshold 0.90. Tip f58208d, clean.

## Done (commit f58208d, gate-clean: 857 pass / 0 fail)

- `Admission::within_headroom`: `PressureState::Green => Red` arm; test `admission::tests::kv_headroom_green_burst`
  (14 x 70 blocks of 1,024: 13 admitted, 14th queues `kv_reservation`, refill obeys it). Spec S-9 text and AC, contract line amended.
- Red-before-fix + mutation check done in one operation on the stashed tree (Green arm removed → the test fails with
  14 admitted); snapshot `target/greenhead/admission.rs.mutated.bak`.
- Overload_sim regression fixed honestly: the cap makes admissions-only SURVIVAL impossible (cap 0.90 < SURVIVAL exit
  0.9215), sweep seeds 1-12 both options peak at RED, recover 43.4-50.2 s. `OverloadConfig::oom_once_at` injects S-11's
  device OOM once; seed 6 (requeue_unstarted, OOM at 60 s) passes through SURVIVAL, GREEN+HEALTHY 42 s after stop;
  option_b gained the same OOM leg (47.4 s); seed_1 stays OOM-free. Spec line 143 narrative + S-11 AC amended.
- Lab (GPU 0, clean tree): 16-filler burst peaks kv_utilization 0.8271, GREEN->ORANGE->YELLOW->GREEN only, 15 queued
  (6 kv_reservation, 11 pressure_orange), client 2/2 (`target/greenhead/burst-pressure.log`). Quick bench tok/s 877.3
  (baseline ~850), golden c1 PASS (`target/greenhead/soak-bench.log`).
- Perf log 6b entry appended.

## Open: nothing

- The 10-minute soak PASSes 8/8 (`target/soak/novanas-20261003T114226Z`, serve 1003114226-29519f6d): ITL p99 206.0 ms
  vs calibration 173.2, GREEN 5 s into the cool-down, KV idle, reserve held, 4,632 x 200 / 21 x 429 `queue_full`,
  3,321 `queue_timeout`, `streams_complete` true. First attempt with a Mac-local client failed only `streams_complete`
  (48 of 4,603 streams truncated over the VPN; the server emits `[DONE]` after every mid-stream error, see
  `turbine-api/src/openai/stream.rs` `StreamState::fail`) — transport loss, not server behavior. SOAK_BENCH wrapper
  kept at `target/greenhead/soak-bench-remote.sh` (forwards `"$@"`, relocates `--pressure-timeline` to a remote path).

## Next steps

1. Nothing for this task; the branch is ready for the lead's review/merge.
2. Remote `target/debug` was deleted after f58208d; delete again only if more Rust commits land.
