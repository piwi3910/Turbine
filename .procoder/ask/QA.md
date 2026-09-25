# Questions procoder cannot answer for you

Written 2026-09-25 18:47 UTC.

Answer each one by writing a line beginning `Answer: ` under it, then
hand the file back with `procoder ask --file .procoder/ask/QA.md`.
Leave the `Key:` lines alone — they are what ties an answer to its question.

## Q1: [decision] decisions.md

Key: c824e5b033ae
Question: P0-T7 real-stream check with no OpenAI endpoint on novanas

- Defer the check to Phase 1, run against Turbine itself on novanas (recommended)
- Start a vLLM-ROCm k3s Job on novanas with a small model now

**Answers (2026-09-25):** "no Sparks" applies to Phase 0 only — Phase 0's Spark lab checks move to Phase 2b (first phase that runs on the Sparks); later phases use the Sparks as planned, asking before heavy runs. P0-T7 real-stream check deferred to Phase 1, run against Turbine itself on novanas.

Answer: Defer to Phase 1, against Turbine on novanas (user chose the recommended option).

## Q2: [decision] decisions.md

Key: 3a1e7d3d983a
Question: P0-T7: run the manual turbine-bench check against production vLLM on dgx-spark (10 requests, concurrency 2)?

- Yes, run it now (small read-only load on production vLLM) (recommended)
- Later, when you have moved workloads

**Answer (2026-09-25):** "don't run on the spark, novanas only" — no turbine-bench run against the Sparks.

Answer: No — "don't run on the spark, novanas only" (user).

## Q3: [decision] decisions.md

Key: 76e205b1e389
Question: P0-T8: run the Phase 0 lab suite on novanas now?

- Yes: create /home/piwi/turbine-ci and k3s namespace turbine-ci, run one Job with amd.com/gpu: 2 (rust:1.97-trixie, ~10–20 min incl. first build) (recommended)
- Not yet

Answer: Yes, run it (user approved).

## Q4: [decision] decisions.md

Key: e4d08548e81d
Question: Scope of "no Sparks": Phase 0 only, or all phases until further notice?

- Until you say otherwise: no Turbine runs on the Sparks; Spark lab steps (P0 Spark discovery check, P2b, P3 Spark soak, P6, P7 M1) wait (recommended)
- Phase 0 only
- Permanently: re-plan the Spark-based phases onto novanas

Answer: Phase 0 only (user).
