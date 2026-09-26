# Turbine performance log

One row per landed change (user rule 2026-09-26: land one improvement, then benchmark and compare; fix or revert regressions before the next landing).

Workload (fixed): one Radeon AI PRO R9700 on novanas, `turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos --output json`; correctness: `turbine-golden compare --concurrency 1` (16/16 required; deterministic) and `--concurrency 16` (reported; batch composition changes GEMM rounding, so c16 logprobs vary run to run — p14 likely Δ 0.066–0.178). Target (Phase 2c): ≥ 75% of vLLM-ROCm — Llama-3.2-3B ≥ 553 tok/s, OLMoE-1B-7B ≥ 401 tok/s.

## Llama-3.2-3B-Instruct (BF16)

| Date | Commit | Change | tok/s | ITL p50 ms | TTFT p50 ms | decode fwd ms | Golden | vs previous |
|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | vLLM-ROCm 0.23.0 | reference | 738.0 | 17.0 | 338 | – | – | – |
| 2026-09-26 | 7850162 | Phase 2 engine baseline (16-token pages) | 92.2 | 164 | 1528 | 47 | 16/16 | – |
| 2026-09-26 | da01324 | Phase 2c start: all Phase 2 fixes, clean run under bench-lock (host tests 267/0) | 91.9 | 164 | 1533 | 47.4 | 16/16 | ±0 |
| 2026-09-26 | 462f068 | host: logprobs only when requested (269/0) | 99.9 | 150 | 1519 | 47.3 | 16/16 | +8.7% |
| 2026-09-26 | 156681b | host: sampler reuses vocab-sized buffers — REGRESSION (re-measured 91.6); perf profile: full candidate sort hot | 91.9 | 154 | 1528 | 47.4 | 16/16 | −8.0% |
| 2026-09-26 | 8bcf398 | host: integer sort keys — fixes 156681b's regression (270/0) | 146.1 | 99 | 1428 | 51.1 | c1 16/16 (repeatable); c16 15/16 once, 16/16 on two reruns | +46% vs 462f068 |

## OLMoE-1B-7B-0125-Instruct (BF16)

| Date | Commit | Change | tok/s | ITL p50 ms | TTFT p50 ms | decode fwd ms | Golden | vs previous |
|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | vLLM-ROCm 0.23.0 | reference | 534.9 | 26.5 | 201 | – | – | – |
