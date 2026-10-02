# Handoff: p6b-greenhead (GREEN admission headroom), paused

Branch `p6b-greenhead` from `p6b-stack` 202bd4a; decision "6b: which cap the GREEN admission headroom uses" (A): RED threshold 0.90.

## Done (one commit, NOT gate-clean)
- `Admission::within_headroom`: `PressureState::Green => Red` arm; test `admission::tests::kv_headroom_green_burst`
  (14 x 70 blocks of 1,024: 13 admitted, 14th queues `kv_reservation`, refill obeys it). Spec S-9 text and AC, contract line amended.
- Gate (`scripts/gate.sh --base 4b8c0ea`): 856 pass, 1 FAIL: `turbine-scheduler::overload_sim survival_liveness_seed_6`
  asserts seed 6 passes through SURVIVAL (line ~525); with the fix it peaks at RED (47.1 s recovery, 531 completions, fine).

## Next steps
1. Red/mutation check not yet done: remove the Green arm, confirm `kv_headroom_green_burst` fails, restore (snapshot first).
2. Fix the overload_sim regression: the liveness regression needs a seed (sweep `survival_liveness_sweep -- --ignored --nocapture`) or an
   OOM-triggered scenario that still reaches SURVIVAL; update seed_6/seed_1/option_b and the spec S-11 AC text; rerun gate until clean.
3. Lab (GPU 0, from a clean committed tree, run_in_background): 16-filler burst per `p6b-t16.md` (no GREEN->SURVIVAL),
   `scripts/overload-soak.sh novanas --duration 10m`, `scripts/lab-bench.sh --quick --model llama`; record in perf log 6b.
4. Delete remote `target/debug` after the Rust commit.
No lab runs were started; no Jobs of this branch exist.
