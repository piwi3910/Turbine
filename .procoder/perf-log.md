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
| 2026-09-26 | 744fe8c | host: vectorised draw for unseeded requests (276/0) | 250.7 | 54.5 | 1543 | 44.8 | c1 16/16; c16 16/16 | +72% |
| 2026-09-26 | 85ed316 | host: decode rows sampled in parallel (277/0) | 268.3 | 50.2 | 1536 | 43.9 | c1 16/16; c16 16/16 | +7.0% |
| 2026-09-26 | 6fc9854 | engine stage breakdown metric (279/0) | 261.6 | 47.6 | 1553 | 42.1 | c1 16/16; c16 16/16 | −2.5% (within noise) |
| 2026-09-26 | 551c3db | execution.* config keys, no runtime change (280/0) — shows run-to-run noise ≈ ±5% | 286.0 | 46.5 | 1558 | 40.7 | c1 16/16; c16 16/16 | +9.3% (noise) |
| 2026-09-26 | 006cec7 | rope/silu_mul spread over the device (280/0) | 285.5 | 45.9 | 1512 | 40.3 | c1 16/16; c16 16/16 | −0.2% |
| 2026-09-26 | 288dbc2 | kernel ABI v2.1 (optional symbols; no runtime change) (285/0) | 292.1 | 45.6 | 1509 | 39.9 | c1 16/16; c16 16/16 | +2.3% |
| 2026-09-26 | 750872c | **KV pages 128 → CK fmha_fwd_pagedkv for prefill and decode (290/0) — passes the 553 target (84% of vLLM)** | **623.6** | 21.9 | 575 | 19.6 | c1 16/16; c16 16/16 | +113% |
| 2026-09-26 | (06f87c1, reverted) | fused QKV + gate/up GEMMs — REGRESSION; same build with execution.fused_ops=false: 624.6 tok/s, fwd 19.6 ms → the fused path itself is slower; reverted pending diagnosis | 556.7 | 25.1 | 572 | 22.2 | c1 16/16; c16 16/16 | −10.7% |
| 2026-09-26 | 8c7d92a | **GPU 0 vs GPU 1, same build**: GPU 0 618.0 / GPU 1 533.3 tok/s. GPU 1's PCIe link was trained at Gen 1 (fixed by retrain → Gen 5 x8, but still 533.6); clocks/power/IRQs equal; cause open. From here all rows are **pinned to GPU 0** (scripts/lab-bench.sh). The 06f87c1 "regression" and the 536 tok/s after the MoE commit were GPU 1 runs. | 618.0 | 22.1 | – | – | c1 16/16 | new basis |
| 2026-09-26 | d7dd545 | fused QKV/gate-up projections, opt-in (default off) (293/0) | 618.4 | 22.1 | 577 | 19.7 | c1 16/16; c16 16/16 | +0.1% |
| 2026-09-26 | eaad734 | fused residual add + RMSNorm (CK rmsnorm2d add variant) (293/0) | 625.5 | 21.9 | 574 | 19.5 | c1 16/16; c16 15/16 flake | +1.1% |
| 2026-09-26 | 3f21ad7 | GPU sampling 1/3: device logits reduce, Rust side (no kernel yet) (298/0) | 594.3 | 22.9 | 575 | 20.6 | c1 16/16; c16 16/16 | −5.0% (judged as a unit with the next two) |
| 2026-09-26 | 0f4bb8f | GPU sampling 2/3: HIP logits_reduce kernel — only greedy rows eligible (1,024 of 52,225) (298/0) | 603.8 | 22.8 | 574 | 20.3 | c1 16/16; c16 16/16 | −3.5% vs eaad734 |
| 2026-09-26 | 09a41da | **GPU sampling 3/3: device top_p — 52,224 of 52,225 rows reduced on the GPU (299/0) — 96% of vLLM's 738** | **708.2** | 18.8 | 573 | 18.1 | c1 16/16; c16 16/16 | **+13.2% vs eaad734** |
| 2026-09-26 | 5b653f6 | op-level forward profile hooks (off by default) (301/0) | 706.6 | 18.9 | 572 | 18.1 | c1 16/16; c16 16/16 | −0.2% |
| 2026-09-26 | 936d1ce | lab diagnostics tests only (301/0) | 697.3 | 19.1 | 575 | 18.3 | c1 16/16; c16 15/16 flake | −1.3% (noise) |
| 2026-09-26 | bd99d87 | **fused QKV + gate/up projections on by default (301/0) — parity with vLLM-ROCm (738.0)** | **739.4** | 18.0 | 572 | 17.2 | c1 16/16; c16 16/16 | +6.0% |

## OLMoE-1B-7B-0125-Instruct (BF16)

| Date | Commit | Change | tok/s | ITL p50 ms | TTFT p50 ms | decode fwd ms | Golden | vs previous |
|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | vLLM-ROCm 0.23.0 | reference | 534.9 | 26.5 | 201 | – | – | – |
| 2026-09-26 | e088bd7 | first OLMoE run on the perf branch: host sampler + 128-token pages (290/0); 200/200 streams ok (no SSE failures) | 290.3 | 52.3 | 407 | 45.5 | 5/16 (known: tolerance to be calibrated, decision 2026-09-26) | – |
| 2026-09-26 | 870f842 | **OLMoE on GPU 0 at the Llama-parity tip: MoE small-m path, 128-token pages, GPU sampling, fused add+norm/QKV — 108% of vLLM (534.9)**; 200/200 streams ok | **575.4** | 25.1 | 401 | 22.3 | 5/16 (known; calibration pending) | +98% vs e088bd7 |
