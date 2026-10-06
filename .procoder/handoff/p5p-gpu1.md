# Handoff: p5p-gpu1 — one-GPU-job rule lifted, GPU 1 benchmarked (2026-10-06)

Branch `p5p-gpu1` (from `p6b-stack` a004fd93), worktree `.claude/worktrees/agent-p5p-gpu1`. Not pushed.

## Commits

- fa70043d `feat(lab)`: lifts the one-GPU-job rule. `lab-test.sh --gpus 2`, `lab-serve.sh --gpus 2` and `lab-cluster.sh` are no longer refused. lab-test takes bench.lock shared again. lab-serve's holder (`runs/serve-locks/<run id>.sh`) now takes the lock shared, passing `bench.gate` first, for its Job's whole life, and takes nothing under a caller's bench-lock hold. One test or serve Job per card can run side by side, and a benchmark (exclusive, with writer preference) still waits for them. `TURBINE_LAB_ONE_GPU_JOB=1` restores the old rule. `lab-bench.sh --gpu 1` is accepted, and any value other than 0 or 1 is a usage error.
- 15a16034 `docs(perf)`: GPU 1 vs GPU 0 on the new PSU. GPU 1 lands within 0.6 % of GPU 0, so it joins perf work: the WARN is gone and AGENTS.md, `docs/extending/card-family.md`, the perf log and decisions.md are updated.

## Tests

`benches/turbine-bench/tests/lab_scripts.rs` gets `lab_script_env`, which also clears `TURBINE_LAB_ONE_GPU_JOB` and `TURBINE_BENCH_LOCK_HELD` for every scripted run. New or changed tests:

- `lab_serve_holds_bench_lock_shared_unless_one_gpu_job`
- `two_gpu_lab_runs_are_not_refused_by_default` (a real run with stubs: the lab-test shared lock, the lab-cluster run, and the refusals with `=1`)
- `lab_bench_gpu_1_serves_on_gpu_1`
- the usage test (`--gpu 2` is refused with "--gpu is 0 or 1")
- the failed-upload holder test (it now looks for `setsid nohup sh -c`)

Red against the old scripts: 5 failed. Mutation: setting the lab-test default back to 1 fails `two_gpu_lab_runs_are_not_refused_by_default`. The shared holder command was also checked live on novanas with throwaway lock files: exclusive blocked, shared ok, released on exit.

## Benchmark (perf log "GPU 1 vs GPU 0 on the new PSU")

| Model | conc | GPU 0 tok/s (g0a/g0b) | GPU 1 | ratio |
| ----- | ---- | --------------------- | ----- | ----- |
| Llama | 16   | 852.8 / 848.5         | 855.9 | 1.006 |
| OLMoE | 16   | 616.7 / 617.7         | 614.9 | 0.996 |
| Llama | 1    | 78.0 / 78.0           | 78.1  | 1.001 |
| OLMoE | 1    | 150.0 / 150.1         | 150.3 | 1.002 |

All 12 runs are in labbook (ids in the perf log). Raw logs and the amd-smi samples are in `target/gpu1cmp/` in this worktree (not committed). In sysfs both cards link Gen5 x8 at the root port, then Gen5 x16 inside the card. amd-smi's PCIe field is unreliable here.

## Follow-ups for the coordinator

- The auto-memory note "GPU 1 not for perf" (gpu1-no-perf.md) is now wrong and should be updated: GPU 1 is a perf peer, Gen5 x8 like GPU 0.
- `scripts/lab-decode-profile.sh` and some `docs/extending/*` pages still name `--gpu 0` / GPU 0 as the example card. Those are still valid invocations, so they were left alone.
- Other builders' trees still default to `TURBINE_LAB_ONE_GPU_JOB=1` (exclusive) until they merge this branch.
