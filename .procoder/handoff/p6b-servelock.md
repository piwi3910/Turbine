# p6b-servelock handoff

Bug: `scripts/lab-serve.sh` took bench.lock (holder on novanas) before the upload; `fail` exits
(rsync or apply failure) never released it, so the holder waited up to 1 h for a Job that never came.

Fix (`scripts/lab-serve.sh`):

- `on_exit` EXIT trap: until `JOB_APPLIED=1` (set right after a successful `kubectl apply`, Turbine and vLLM paths) it deletes this run's Job if any and calls `gpu_unlock` (kills the holder, removes its files).
- The holder's wait for the Job is now `HOLDER_GRACE` (`TURBINE_LAB_HOLDER_GRACE`, default 900 s from getting the lock) instead of 3600 s, so it also self-terminates when the script was SIGKILLed.
- Success path and `--stop` unchanged.

Tests (`benches/turbine-bench/tests/lab_scripts.rs`): `lab_serve_failed_upload_releases_the_bench_lock_holder` (stubbed ssh/rsync, real run, asserts the pkill follows the holder start) and `lab_serve_holder_self_terminates_without_a_job`. Mutation checked by hand: with the trap reverted to `stop_log_stream`, no pkill is issued.

No real-host check was done.
