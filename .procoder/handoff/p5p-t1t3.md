# Handoff: Phase 5p Tasks 1–3 (2026-10-04)

Worktree `agent-p6b-t18`, branch `p5p-t1t3` (branched from `p6b-stack` = `main` tip f5550087,
per the tasking; the plan's "branch `phase-5p`" step is satisfied by starting from the same
commit — rename or fast-forward when the lead wants the canonical branch name).

## Landed

1. **Task 1 — `turbine-bench stage-share`** (9461c4e3): `benches/turbine-bench/src/stage_share.rs`
   parses the `turbine_engine_iteration_seconds_{sum,count}{stage="…"}` series of two `/metrics`
   scrapes and reports `StageShare { iterations, seconds: BTreeMap<String, f64>, host_share }`
   (`host_share` = all eight stages except `device_wait` over all eight; `launch` is host time).
   CLI: `stage-share --before <f> --after <f> [--output text|json]`, exit 0 / 2 on
   `MissingHistogram` / `NoIterations`. Test `bench stage_share_from_metrics` over two committed
   fixture scrapes (hand-computed deltas, the moved-second check, CLI exit codes);
   mutation (device_wait folded into the host sum) turns it red.
2. **Task 2 — `lab-bench.sh` flags** (694340d9): `--dry-run` (an `ssh()` shadow prints every host
   command; local steps guarded; prints build → serve → golden → bench → stage-share →
   multi-turn → stop and contacts nothing), `--concurrency <n>` (default 16),
   `--multi-turn` (16 sessions at c8, the Phase 4 command) and `--multi-turn-sessions <n>`
   (saturated, sessions = concurrency; implies the multi-turn step). Every run now computes the
   host share by running the stage-share subcommand on novanas over the before/after scrapes
   (`conc=` + `host_share=` on the BENCH line; `concurrency`/`host_share`/`mt_*` in
   `labbook-values.json`). Default command line and BENCH fields otherwise byte-compatible.
   Tests: `lab_scripts lab_bench_dry_run_passes_flags` (step order, no host contact) and
   `lab_bench_dry_run_defaults_are_unchanged`; the older flag tests now run as dry runs.
   Mutation (stage-share step dropped) turns both new tests red. AGENTS.md's lab-bench line
   documents the flags. 5 labbook data points added (`concurrency`, `host_share`, `mt_cached`,
   `mt_ttft_first_p50/p99`, `mt_ttft_later_p50`, `mt_tok_s` — type `turbine-lab-bench` v5) and
   the uploader's `config` param now carries the workload flags so a c1 run is never the
   "previous comparable run" of a c16 one.
3. **Task 3 — the S-2 baselines** (perf log "Phase 5p", decisions "Phase 5p: host share
   measured — overlap scheduling go/no-go"): nine `lab-bench.sh` runs, novanas GPU 0, ladder
   off, commit 694340d9, all `golden1=PASS`, `requests_failed` 0, labbook set
   `phase-5p-serving-efficiency`. **Host share: Llama c1 0.0037, OLMoE c1 0.0077 (c16: 0.0192 /
   0.0119) — both c1 values far below the 0.05 rule, so Task 4 (overlap scheduling) is CLOSED
   with no code change**, recorded in decisions.md. Baselines for later tasks: Llama c16
   852.2 tok/s (ITL p50 15.5 ms, TTFT p50 207 ms), OLMoE c16 615.4 (24.4 ms, 118 ms); Llama c1
   77.8, OLMoE c1 126.0. Multi-turn standard (median of 3): cached 0.9065, first-turn TTFT p50
   129 ms (p99 501), later-turn TTFT p50 80 ms, mt tok/s 342.4. Saturated (64 sessions, c64,
   `max_running_requests=16`, median of 3): cached 0.9097, first-turn TTFT p50 3 731 ms
   (p99 7 931), later-turn TTFT p50 10 736 ms, mt tok/s 488.4 (mt64-2 is the low outlier:
   cached 0.8038, mt tok/s 346.3 — deepest queue, most eviction).

## Notes for the next builder

- The stage-share subcommand runs on novanas (fresh binary) over scrapes pushed to
  `$remote/stage-share/`; keep that path when touching the steps array.
- Plan Task 4's checklist says "if every c1 host_share ≤ 0.05: mark this task done with a
  pointer to the Task 3 decisions entry; no code, no commit" — the pointer is the decisions
  entry "Phase 5p: host share measured — overlap scheduling go/no-go (2026-10-04)".
- S-4 (token-granular prefix reuse) is next per the phase order; its before-numbers are the
  standard multi-turn medians above (`mt_cached` ≥ +0.01, later-turn TTFT ≤ 0.95 ×).
- Scratch (not committed): `target/p5p/` (run logs, values, the run/watch scripts).
- gate ok on every commit; lab runs under the standing novanas approvals; no serve Jobs left
  running (each lab-bench run stopped its own server; `--stop` not needed).
