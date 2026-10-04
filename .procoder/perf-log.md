# Turbine performance log

One row per landed change (user rule 2026-09-26: land one improvement, then benchmark and compare; fix or revert regressions before the next landing).

Workload (fixed): one Radeon AI PRO R9700 on novanas, `turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos --output json`; correctness: `turbine-golden compare --concurrency 1` (16/16 required; deterministic) and `--concurrency 16` (reported; batch composition changes GEMM rounding, so c16 logprobs vary run to run — p14 likely Δ 0.066–0.178). Target (Phase 2c): ≥ 75% of vLLM-ROCm — Llama-3.2-3B ≥ 553 tok/s, OLMoE-1B-7B ≥ 401 tok/s.

## Llama-3.2-3B-Instruct (BF16)

| Date       | Commit              | Change                                                                                                                                                                                                                                                                                                                                           | tok/s     | ITL p50 ms | TTFT p50 ms | decode fwd ms | Golden                                                     | vs previous                                |
| ---------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ---------- | ----------- | ------------- | ---------------------------------------------------------- | ------------------------------------------ |
| 2026-09-26 | vLLM-ROCm 0.23.0    | reference (k8s-scheduled card, unknown)                                                                                                                                                                                                                                                                                                          | 738.0     | 17.0       | 338         | –             | –                                                          | –                                          |
| 2026-09-26 | vLLM-ROCm 0.23.0    | **reference pinned to GPU 0** (placeholder pod on GPU 1); 199/200 ok                                                                                                                                                                                                                                                                             | 715.1     | 17.0       | 339         | –             | –                                                          | Turbine bd99d87 = **103%**                 |
| 2026-09-26 | 7850162             | Phase 2 engine baseline (16-token pages)                                                                                                                                                                                                                                                                                                         | 92.2      | 164        | 1528        | 47            | 16/16                                                      | –                                          |
| 2026-09-26 | da01324             | Phase 2c start: all Phase 2 fixes, clean run under bench-lock (host tests 267/0)                                                                                                                                                                                                                                                                 | 91.9      | 164        | 1533        | 47.4          | 16/16                                                      | ±0                                         |
| 2026-09-26 | 462f068             | host: logprobs only when requested (269/0)                                                                                                                                                                                                                                                                                                       | 99.9      | 150        | 1519        | 47.3          | 16/16                                                      | +8.7%                                      |
| 2026-09-26 | 156681b             | host: sampler reuses vocab-sized buffers — REGRESSION (re-measured 91.6); perf profile: full candidate sort hot                                                                                                                                                                                                                                  | 91.9      | 154        | 1528        | 47.4          | 16/16                                                      | −8.0%                                      |
| 2026-09-26 | 8bcf398             | host: integer sort keys — fixes 156681b's regression (270/0)                                                                                                                                                                                                                                                                                     | 146.1     | 99         | 1428        | 51.1          | c1 16/16 (repeatable); c16 15/16 once, 16/16 on two reruns | +46% vs 462f068                            |
| 2026-09-26 | 744fe8c             | host: vectorised draw for unseeded requests (276/0)                                                                                                                                                                                                                                                                                              | 250.7     | 54.5       | 1543        | 44.8          | c1 16/16; c16 16/16                                        | +72%                                       |
| 2026-09-26 | 85ed316             | host: decode rows sampled in parallel (277/0)                                                                                                                                                                                                                                                                                                    | 268.3     | 50.2       | 1536        | 43.9          | c1 16/16; c16 16/16                                        | +7.0%                                      |
| 2026-09-26 | 6fc9854             | engine stage breakdown metric (279/0)                                                                                                                                                                                                                                                                                                            | 261.6     | 47.6       | 1553        | 42.1          | c1 16/16; c16 16/16                                        | −2.5% (within noise)                       |
| 2026-09-26 | 551c3db             | execution.* config keys, no runtime change (280/0) — shows run-to-run noise ≈ ±5%                                                                                                                                                                                                                                                                | 286.0     | 46.5       | 1558        | 40.7          | c1 16/16; c16 16/16                                        | +9.3% (noise)                              |
| 2026-09-26 | 006cec7             | rope/silu_mul spread over the device (280/0)                                                                                                                                                                                                                                                                                                     | 285.5     | 45.9       | 1512        | 40.3          | c1 16/16; c16 16/16                                        | −0.2%                                      |
| 2026-09-26 | 288dbc2             | kernel ABI v2.1 (optional symbols; no runtime change) (285/0)                                                                                                                                                                                                                                                                                    | 292.1     | 45.6       | 1509        | 39.9          | c1 16/16; c16 16/16                                        | +2.3%                                      |
| 2026-09-26 | 750872c             | **KV pages 128 → CK fmha_fwd_pagedkv for prefill and decode (290/0) — passes the 553 target (84% of vLLM)**                                                                                                                                                                                                                                      | **623.6** | 21.9       | 575         | 19.6          | c1 16/16; c16 16/16                                        | +113%                                      |
| 2026-09-26 | (06f87c1, reverted) | fused QKV + gate/up GEMMs — REGRESSION; same build with execution.fused_ops=false: 624.6 tok/s, fwd 19.6 ms → the fused path itself is slower; reverted pending diagnosis                                                                                                                                                                        | 556.7     | 25.1       | 572         | 22.2          | c1 16/16; c16 16/16                                        | −10.7%                                     |
| 2026-09-26 | 8c7d92a             | **GPU 0 vs GPU 1, same build**: GPU 0 618.0 / GPU 1 533.3 tok/s. GPU 1's PCIe link was trained at Gen 1 (fixed by retrain → Gen 5 x8, but still 533.6); clocks/power/IRQs equal; cause open. From here all rows are **pinned to GPU 0** (scripts/lab-bench.sh). The 06f87c1 "regression" and the 536 tok/s after the MoE commit were GPU 1 runs. | 618.0     | 22.1       | –           | –             | c1 16/16                                                   | new basis                                  |
| 2026-09-26 | d7dd545             | fused QKV/gate-up projections, opt-in (default off) (293/0)                                                                                                                                                                                                                                                                                      | 618.4     | 22.1       | 577         | 19.7          | c1 16/16; c16 16/16                                        | +0.1%                                      |
| 2026-09-26 | eaad734             | fused residual add + RMSNorm (CK rmsnorm2d add variant) (293/0)                                                                                                                                                                                                                                                                                  | 625.5     | 21.9       | 574         | 19.5          | c1 16/16; c16 15/16 flake                                  | +1.1%                                      |
| 2026-09-26 | 3f21ad7             | GPU sampling 1/3: device logits reduce, Rust side (no kernel yet) (298/0)                                                                                                                                                                                                                                                                        | 594.3     | 22.9       | 575         | 20.6          | c1 16/16; c16 16/16                                        | −5.0% (judged as a unit with the next two) |
| 2026-09-26 | 0f4bb8f             | GPU sampling 2/3: HIP logits_reduce kernel — only greedy rows eligible (1,024 of 52,225) (298/0)                                                                                                                                                                                                                                                 | 603.8     | 22.8       | 574         | 20.3          | c1 16/16; c16 16/16                                        | −3.5% vs eaad734                           |
| 2026-09-26 | 09a41da             | **GPU sampling 3/3: device top_p — 52,224 of 52,225 rows reduced on the GPU (299/0) — 96% of vLLM's 738**                                                                                                                                                                                                                                        | **708.2** | 18.8       | 573         | 18.1          | c1 16/16; c16 16/16                                        | **+13.2% vs eaad734**                      |
| 2026-09-26 | 5b653f6             | op-level forward profile hooks (off by default) (301/0)                                                                                                                                                                                                                                                                                          | 706.6     | 18.9       | 572         | 18.1          | c1 16/16; c16 16/16                                        | −0.2%                                      |
| 2026-09-26 | 936d1ce             | lab diagnostics tests only (301/0)                                                                                                                                                                                                                                                                                                               | 697.3     | 19.1       | 575         | 18.3          | c1 16/16; c16 15/16 flake                                  | −1.3% (noise)                              |
| 2026-09-26 | bd99d87             | **fused QKV + gate/up projections on by default (301/0) — parity with vLLM-ROCm (738.0)**                                                                                                                                                                                                                                                        | **739.4** | 18.0       | 572         | 17.2          | c1 16/16; c16 16/16                                        | +6.0%                                      |

| 2026-09-26 | 8acce3e | Phase 2 final (merged to main 72504ad): decode graphs, GPU 0, `scripts/lab-bench.sh` (phase2-novanas config, 8,192 batch tokens) | 745.0 | 17.8 | 573 | 17.0 | c1 16/16; c16 16/16 | – |
| 2026-09-26 | 958ea68 | Scout pass-2 fixes (7 commits) (315/0) | 748.4 | 17.7 | 566 | 17.0 | c1 16/16; c16 16/16 | +0.5% |
| 2026-09-26 | 6121a86 | kernel ABI v2.3 pinned host staging + events (317/0) | 746.4 | 17.8 | 591 | 17.0 | c1 16/16; c16 16/16 | −0.3% |
| 2026-09-26 | f5d0c41 | launch-ahead executors with device token feeds (319/1: tiny_server port-race flake, fixed in 728649b) | 752.3 | 17.6 | 571 | 16.9 | c1 16/16; c16 16/16 | +0.8% |
| 2026-09-26 | 1b9730c | overlap scheduling, on (323/0) | 753.3 | 17.5 | 580 | 16.8 | c1 16/16; c16 16/16 | +0.1% (no gain → default off, 19f67b1) |
| 2026-09-26 | 62a8fc2 | overlap off by default + prefill lab tests (323/0) | 753.5 | 17.6 | 566 | 16.8 | c1 16/16; c16 16/16 | 0.0% |
| 2026-09-26 | a39f859 | whole-token vectorised RoPE (323/0) | 765.4 | 17.6 | 510 | 16.8 | c1 16/16; c16 16/16 | +1.6% |
| 2026-09-26 | f105923 | 16-byte SiLU·up (323/0) | 765.7 | 17.7 | 501 | 16.9 | c1 16/16; c16 16/16 | 0.0% |
| 2026-09-26 | a25a6d2 | tip after MoE port + sim (326/0), phase2 config | 768.4 | 17.6 | 497 | 16.8 | c1 16/16; c16 16/16 | +0.4% |
| 2026-09-26 | a25a6d2 | **`scheduler.max_batch_tokens` 2,048 (phase2c config) — TTFT −61%, below vLLM's 339** | **770.4** | 17.7 | **195** | 16.9 | c1 16/16; c16 16/16 | +0.3% tok/s |

## OLMoE-1B-7B-0125-Instruct (BF16)

| Date       | Commit           | Change                                                                                                                                                     | tok/s     | ITL p50 ms | TTFT p50 ms | decode fwd ms | Golden                                                                                                                  | vs previous                    |
| ---------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- | ----------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 2026-09-26 | vLLM-ROCm 0.23.0 | reference (k8s-scheduled card, unknown)                                                                                                                    | 534.9     | 26.5       | 201         | –             | –                                                                                                                       | –                              |
| 2026-09-26 | vLLM-ROCm 0.23.0 | **reference pinned to GPU 0**; 200/200 ok                                                                                                                  | 535.5     | 26.6       | 200         | –             | –                                                                                                                       | Turbine 870f842 = **107%**     |
| 2026-09-26 | e088bd7          | first OLMoE run on the perf branch: host sampler + 128-token pages (290/0); 200/200 streams ok (no SSE failures)                                           | 290.3     | 52.3       | 407         | 45.5          | 5/16 (known: tolerance to be calibrated, decision 2026-09-26)                                                           | –                              |
| 2026-09-26 | 870f842          | **OLMoE on GPU 0 at the Llama-parity tip: MoE small-m path, 128-token pages, GPU sampling, fused add+norm/QKV — 108% of vLLM (534.9)**; 200/200 streams ok | **575.4** | 25.1       | 401         | 22.3          | 5/16 (known; calibration pending)                                                                                       | +98% vs e088bd7                |
| 2026-09-26 | c5c0d3b          | Phase 2 final: BF16 router + torch.topk ties, calibrated tolerance (310/0)                                                                                 | 567.1     | 25.5       | 405         | 22.6          | c1 16/16; c16 16/16                                                                                                     | –                              |
| 2026-09-26 | 728649b          | launch-ahead + device token feeds (320/0)                                                                                                                  | 569.1     | 25.4       | 403         | 22.5          | c1 PASS; c16 14/16 + p14 likely Δ 1.041 (pre-existing batch-composition flip: 3/3 fail at 6121a86); reruns 16/16, 14/16 | +0.4%                          |
| 2026-09-26 | 1b9730c          | overlap scheduling, on                                                                                                                                     | 570.5     | 25.3       | 635         | 22.4          | c1 PASS; c16 PASS                                                                                                       | +0.0%, TTFT +57% → default off |
| 2026-09-26 | 62a8fc2          | overlap off by default                                                                                                                                     | 571.4     | 25.3       | 402         | 22.5          | c1 PASS; c16 PASS                                                                                                       | +0.2%                          |
| 2026-09-26 | 5505299          | whole-token vectorised RoPE                                                                                                                                | 575.5     | 25.3       | 371         | 22.4          | c1 PASS; c16 PASS                                                                                                       | +0.7%                          |
| 2026-09-26 | f105923          | 16-byte SiLU·up                                                                                                                                            | 574.7     | 25.3       | 370         | 22.5          | c1 PASS; c16 PASS                                                                                                       | −0.1%                          |
| 2026-09-26 | 2454bb3          | **parallel MoE routing (ported onto the BF16/torch.topk router)** (323/0)                                                                                  | **613.1** | 24.0       | 314         | 21.2          | c1 PASS; c16 retry PASS (first: known p10/p14 flip)                                                                     | **+6.7%**                      |
| 2026-09-26 | a42c961          | grouped WMMA MoE experts above 512 rows (323/0)                                                                                                            | 619.5     | 23.9       | 273         | 21.2          | c1 PASS; c16 PASS                                                                                                       | +1.0%                          |
| 2026-09-26 | a25a6d2          | tip after sim (326/0), phase2 config                                                                                                                       | 618.4     | 24.0       | 272         | 21.2          | c1 PASS; c16 retry PASS                                                                                                 | −0.2%                          |
| 2026-09-26 | a25a6d2          | **`scheduler.max_batch_tokens` 2,048 (phase2c config) — 116% of vLLM, TTFT −51%, below vLLM's 200**                                                        | **619.8** | 23.9       | **134**     | 21.1          | c1 PASS; c16 PASS                                                                                                       | +0.2% tok/s                    |

## Phase 2m modularity chain

Novanas GPU 0, `scripts/lab-bench.sh` with the phase2c configs (2,048 batch tokens); bounds per slice: tok/s ≥ 0.97 × and TTFT p50 ≤ 1.10 × the previous row; OLMoE c16 gets one retry for the known p10/p14 batch-composition flip (superseded 2026-09-27: the flip is fixed and c16 gets no retry; see the Pre-Phase-5 section below).

| Date       | Commit  | Task                                                                                                                                                | Host tests | GPU suites                                                                                       | Llama tok/s | Llama TTFT p50 ms | OLMoE tok/s | OLMoE TTFT p50 ms | Golden (Llama c1/c16; OLMoE c1/c16)                     | vs previous                              |
| ---------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------ | ----------- | ----------------- | ----------- | ----------------- | ------------------------------------------------------- | ---------------------------------------- |
| 2026-09-26 | a705b7e | baseline (= a25a6d2 code, measured there)                                                                                                           | 326/0      | full suite 344/0                                                                                 | 770.4       | 195               | 619.8       | 134               | PASS/PASS; PASS/PASS                                    | –                                        |
| 2026-09-26 | 8b014bf | Task 1: registry convention, open module names, status modules + kernels                                                                            | 336/0      | tiny_model 21/21, golden 8/8                                                                     | 774.4       | 194               | 619.0       | 134               | PASS/PASS; PASS/retry PASS (first 13/16: p05, p10, p14) | +0.5% / −0.1%                            |
| 2026-09-26 | a947112 | Task 2: scheduling policy registry                                                                                                                  | 339/0      | – (host-only; simulator plan digests = main)                                                     | 771.2       | 195               | 617.0       | 135               | PASS/PASS; PASS/PASS                                    | −0.4% / −0.3%                            |
| 2026-09-26 | 176c607 | Task 3: logits-processor chain                                                                                                                      | 344/0      | tiny_model reduce/launch-ahead 2/2                                                               | 770.5       | 196               | 618.4       | 134               | PASS/PASS; PASS/PASS                                    | −0.1% / +0.2%                            |
| 2026-09-26 | 57b53bb | Tasks 5+6: BF16 weight format, model family registry (measured together)                                                                            | 350/0      | tiny_model 21/21, golden 8/8 (both tasks)                                                        | 771.9       | 195               | 617.0       | 134               | PASS/PASS; PASS/PASS                                    | +0.2% / −0.2%                            |
| 2026-09-26 | e48d021 | Tasks 4+7: execution backends + discovery registry, gfx1201 card profile (measured together)                                                        | 360/0      | hip_ops 15/15, tiny_model 21/21, golden 8/8, turbine-device 1-GPU                                | 771.7       | 196               | 616.9       | 135               | PASS/PASS; PASS/retry PASS                              | 0.0% / 0.0%                              |
| 2026-09-26 | a2efeb1 | Task 9: tool-format registry (llama3_json)                                                                                                          | 364/0      | lab_openai 4/4                                                                                   | 770.7       | 195               | 616.9       | 135               | PASS/PASS; PASS/PASS                                    | −0.1% / 0.0%                             |
| 2026-09-27 | 17f97af | Task 8: shared decoder skeleton (llama.rs/olmoe.rs executors removed)                                                                               | 366/0      | tiny_model 22/22, golden 8/8 (Llama/OLMoE 16/16); CPU bitwise = base                             | 772.1       | 195               | 618.1       | 134               | PASS/PASS; PASS/PASS                                    | +0.2% / +0.2%                            |
| 2026-09-27 | a757fd6 | Task 13: support matrix, --support-matrix, startup refusal (log: supported amd/gfx1201/…)                                                           | 380/0      | – (host-only)                                                                                    | 771.7       | 195               | 618.1       | 135               | PASS/PASS; PASS/retry PASS                              | −0.1% / 0.0%                             |
| 2026-09-27 | 08dd00b | Task 14: Qwen3, Qwen3-MoE, Mistral, Mixtral (CPU) + hermes/mistral formats                                                                          | 399/0      | tiny_model + golden 30/0 (Llama/OLMoE 16/16)                                                     | 773.1       | 194               | 617.8       | 135               | PASS/PASS; PASS/PASS                                    | +0.2% / −0.0%                            |
| 2026-09-27 | 1bf3363 | Task 10: kernel ABI v2.4 implementation enumeration + card profile (library side), libturbine_hip_v23.so                                            | 402/0      | hip_ops 16/16 (implementations_enumerated)                                                       | 774.0       | 194               | 618.5       | 134               | PASS/PASS; PASS/PASS                                    | +0.1% / +0.1%                            |
| 2026-09-27 | 3594eb6 | Task 11: implementation selection in Rust (card profile order, bound provider); served kernel choices = tests/lab/kernel-choices-{llama,olmoe}.json | 407/0      | hip_ops incl. every_implementation_matches_cpu, hip_v23_library_matches_cpu; tiny_model + golden | 771.5       | 196               | 618.4       | 135               | PASS/PASS; PASS/PASS                                    | −0.3% / −0.0%                            |
| 2026-09-27 | 4f8ab0f | Task 12: CPU reference split per op family                                                                                                          | 408/0      | hip_ops 17/17                                                                                    | 771.3       | 195               | 617.9       | 135               | PASS/PASS; PASS/retry PASS                              | −0.0% / −0.1%                            |
| 2026-09-27 | c123c37 | Final (Tasks 15–16: conformance suites, docs/extending)                                                                                             | 414/0      | full lab suite 433/0 (9985a91), 2-GPU device 13/0                                                | 773.8       | 194               | 617.2       | 135               | PASS/PASS; PASS/PASS                                    | +0.3% / −0.1%; vs baseline +0.4% / −0.4% |

## Phase 3 reliability chain

Novanas GPU 0, `scripts/lab-bench.sh` (phase2c configs); Tasks 1–11, 13, 15 (new crate, config, API, bench tooling; not on the serving path until Task 14) landed without a bench; bounds as in the Phase 2m chain.

| Date       | Commit  | Task                                                                                                                                                                                                                     | Host tests | GPU suites                                                | Llama tok/s | Llama TTFT p50 ms | OLMoE tok/s | OLMoE TTFT p50 ms | Golden (Llama c1/c16; OLMoE c1/c16) | vs previous                                                                                                                                                                                             |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------- | ----------- | ----------------- | ----------- | ----------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-09-27 | 96c2b8e | baseline: main after Phase 2m (= c123c37 code)                                                                                                                                                                           | 414/0      | full 433/0                                                | 773.8       | 194               | 617.2       | 135               | PASS/PASS; PASS/PASS                | –                                                                                                                                                                                                       |
| 2026-09-27 | 59cad8d | Task 12: scheduler admission gate, throttle overlay, ledger-backed KV, overload sim                                                                                                                                      | 469/0      | –                                                         | 772.9       | 195               | 618.2       | 134               | PASS/PASS; PASS/PASS                | −0.1% / +0.2%                                                                                                                                                                                           |
| 2026-09-27 | 161021e | Task 14: server wiring (budget, pressure controller, telemetry, recovery, circuit breaker)                                                                                                                               | 473/0      | –                                                         | 770.8       | 195               | 620.3       | 135               | PASS/PASS; PASS/PASS                | −0.3% / +0.3%                                                                                                                                                                                           |
| 2026-09-27 | a3ee22f | Task 12a: SURVIVAL requeue (A) + KV headroom (provisional)                                                                                                                                                               | 482/0      | full with fault-injection 509/0, golden Llama/OLMoE 16/16 | 772.6       | 195               | 617.4       | 134               | PASS/PASS; PASS/PASS                | +0.2% / −0.5%                                                                                                                                                                                           |
| 2026-09-27 | d6d5e1d | soak fixes 12b–12d: device_memory counts idle pre-allocated KV + reserve as free; drift per decode step, ignored while idle; admission idle floor; soak config batch 2048; latency drift feeds the circuit only in GREEN | 489/0      | –                                                         | 771.9       | 195               | 545.6       | 97                | PASS/PASS; PASS/PASS                | **OLMoE −11.6%: step_time_drift pushed ORANGE under normal load (MoE decode cost scales with rows)** → fixed next                                                                                       |
| 2026-09-27 | de416d1 | 12e: drift baselines per step shape (rows × context bucket), no cross-bucket extrapolation                                                                                                                               | 492/0      | –                                                         | 770.5       | 195               | 618.0       | 135               | PASS/PASS; PASS/retry PASS          | vs a3ee22f −0.3% / +0.1%; 10-min overload soak (GPU 0) PASS: ITL p99 192 vs 166 ms calibration, GREEN 2 s after cool-down, 4,728 × 200 / 3,673 queue_timeout / 117 queue_full / 72 overloaded, no drops |

## Phase 4 KV tiers

Novanas GPU 0, `scripts/lab-bench.sh --golden16` (phase2c configs, so `kv` keys at their Phase 4 defaults: L1 on, prefix sharing on, NVMe off); bounds as in the Phase 2m chain. Phase 4 was ported as a branch (run-ahead Tasks 1–14, 17 plus Tasks 15–16 on novanas) and measured at its tip and at the merge candidate.

| Date       | Commit  | Task                                                                                                                 | Host tests          | GPU suites                       | Llama tok/s | Llama TTFT p50 ms | OLMoE tok/s | OLMoE TTFT p50 ms | Golden (Llama c1/c16; OLMoE c1/c16) | vs previous                                                                                          |
| ---------- | ------- | -------------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------- | ----------- | ----------------- | ----------- | ----------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 2026-09-27 | 9301996 | baseline: main after Phase 3 (= de416d1 code)                                                                        | 492/0               | full lab (fault-injection) 509/0 | 770.5       | 195               | 618.0       | 135               | pass/pass; pass/pass                | –                                                                                                    |
| 2026-09-27 | fbc9e1e | Phase 4 tip (Tasks 1–17)                                                                                             | 542/0               | kv_gpu 3/3 (reliability on)      | **708.1**   | 229               | –           | –                 | pass/pass; –                        | **−8.1 % Llama: stop**                                                                               |
| 2026-09-27 | fbc9e1e | A/B: `kv.cpu.enabled=false`                                                                                          | –                   | –                                | 768.0       | 196               | –           | –                 | pass/pass; –                        | L1 capacity demotion is the whole regression (1,214 L0→L1 copies, `schedule` stage 4.03 s vs 0.03 s) |
| 2026-09-27 | fbc9e1e | A/B: `kv.prefix_sharing=false` (rerun; first run lost 16 streams to a client-side TCP timeout, server 232/232 ok)    | –                   | –                                | 767.0       | 197               | –           | –                 | pass/pass; –                        | same                                                                                                 |
| 2026-09-27 | 1c4f92e | merge candidate: main + ba5ca6c (capacity demotion only for blocks with reuse evidence, at most 32 blocks per 50 ms) | 544/0 (gate --full) | kv_gpu 3/3, pinned_round_trip    | 761.4       | 197               | 614.6       | 135               | pass/pass; pass/pass                | −1.2 % / −0.6 % (within 3 %)                                                                         |

Multi-turn (S-16), 1c4f92e code, Llama, native server on GPU 0 with `scripts/lab/phase4-novanas.yaml`, `turbine-bench --profile multi-turn --sessions 16 --turns 8 --shared-prefix-words 2000 --concurrency 8 --session-hints` (client on novanas):

| Prefix sharing | Requests ok | Cached-token ratio | TTFT p50 turns ≥ 2 ms | TTFT p99 turns ≥ 2 ms | First-turn TTFT p50 ms | Output tok/s | Wall s |
| -------------- | ----------- | ------------------ | --------------------- | --------------------- | ---------------------- | ------------ | ------ |
| on             | 128/128     | 0.906              | 89                    | 232                   | 212                    | 275          | 55     |
| off            | 128/128     | 0                  | 468                   | 865                   | 309                    | 156          | 92     |

Criterion: ratio ≥ 0.6 and later-turn TTFT ≤ 0.5× off — met (0.906; 0.19×). Pinned L1 round trip (`turbine-kernels --test lab pinned_round_trip`, GPU 0, PCIe Gen5 x8 after the runtime-PM fix): d2h 8.34 GB/s, h2d 9.73 GB/s (Gen1: 1.65 / 1.57).

Full GPU suite on the merge candidate (GPU 1): 395 passed / 0 failed before the old 90-minute Job deadline cut `serving_mix` short (run 0927065245), then the unfinished packages (`turbine-model` onward) 322 passed / 0 failed (run 0927153552, 3-hour deadline, log kept under `target/lab-test/`).

## Pre-Phase-5: OLMoE c16 flip and perf items

Novanas GPU 0, `scripts/lab-bench.sh --golden16` (decode_fwd_ms from 6bfb0c8 on counts the throughput run only); every row is also in labbook (set `olmoe-c16-flip`, later `pre-phase5-perf`).

| Date       | Commit  | Change                                                                       | Host tests | GPU suites                        | OLMoE tok/s | OLMoE ITL p50 ms | OLMoE TTFT p50 ms | Golden OLMoE c1/c16 | vs previous                            |
| ---------- | ------- | ---------------------------------------------------------------------------- | ---------- | --------------------------------- | ----------- | ---------------- | ----------------- | ------------------- | -------------------------------------- |
| 2026-09-27 | 1c4f92e | baseline: Phase 4 merge candidate                                            | 544/0      | –                                 | 614.6       | 24.0             | 135               | pass/pass           | –                                      |
| 2026-09-27 | 59a4e3c | flip root cause 1: small-m MoE tier on the grouped WMMA chain (16-row tiles) | 493/0      | batch_invariance, hip_ops, golden | 528.2       | 28.4             | 135               | pass (15/16)/pass   | **−14.1 %: rejected**                  |
| 2026-09-27 | db26940 | + WMMA decode kernels for the small-m tier (same accumulation order)         | 544/0      | batch_invariance, hip_ops, golden | 600.9       | 24.7             | 136               | pass/pass           | −2.2 % (within 3 %): landed as e22c375 |

| Date       | Commit  | Change                                                                                                                                | Llama tok/s | Llama ITL p50 | Llama TTFT p50 | OLMoE tok/s | OLMoE ITL p50 | OLMoE TTFT p50 | Golden (Llama c1/c16; OLMoE c1/c16) | vs previous         |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | -------------- | ----------- | ------------- | -------------- | ----------------------------------- | ------------------- |
| 2026-09-27 | 92da19b | baseline: main after Phase 4 + flip fix                                                                                               | 761.4       | 17.8          | 197            | 600.9       | 24.7          | 136            | pass/pass; pass/pass                | –                   |
| 2026-09-27 | 39916b3 | #1 per-card GEMM table (option c: OLMoE shapes batch-invariant, Llama shapes pinned for speed; `execution.gemm_autotune` switches it) | **819.9**   | 16.4          | 195            | 603.9       | 24.6          | 135            | pass/pass; pass/pass                | **+7.7 %** / +0.5 % |

OLMoE c16 flip closed: 10 back-to-back `turbine-golden compare --concurrency 16` runs on 39916b3 (one server, GPU 0) gave identical verdicts, 15/16 PASS every time with the same single miss (p14 at its c1 near-tie, margin 0.553). Before the fixes the failing prompts varied run to run. From here on the landing chain gives OLMoE c16 no retry.

| Date       | Commit                   | Change                                                                                                                                | Llama tok/s | Llama ITL p50 | Llama TTFT p50 | OLMoE tok/s | OLMoE ITL p50 | OLMoE TTFT p50 | Golden (Llama c1/c16; OLMoE c1/c16) | vs previous     |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | -------------- | ----------- | ------------- | -------------- | ----------------------------------- | --------------- |
| 2026-09-27 | 28f7901 (landed f485bd1) | #4 `logits_reduce`: one-sweep top keys, integer masses, parallel draw scan (rocPRIM/hipCUB evaluated, none covers top-n/draw/nucleus) | 827.0       | 16.3          | 194            | 611.0       | 24.3          | 134            | pass/pass; pass/pass                | +0.9 % / +1.2 % |
| 2026-09-27 | 117e999                  | #5 OLMoE small-m down projection sweep: current WMMA decode setting already best, no change                                           | –           | –             | –              | –           | –             | –              | –                                   | –               |

| Date       | Commit                   | Change                                                                                                                                                                  | Llama tok/s | Llama ITL p50 | Llama TTFT p50 | OLMoE tok/s           | OLMoE ITL p50 | OLMoE TTFT p50 | Golden (Llama c1/c16; OLMoE c1/c16) | vs previous                                 |
| ---------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | -------------- | --------------------- | ------------- | -------------- | ----------------------------------- | ------------------------------------------- |
| 2026-09-27 | 0cb6cb5 (landed daef7ba) | #3 Llama decode attention on CK `fmha_fwd_splitkv` (heads grouped per KV head, 1 split; llama.cpp FA, aiter and vLLM paged attention evaluated; OLMoE stays on pagedkv) | **863.1**   | 15.4          | 194            | 597.2 / 603.1 (rerun) | 24.8 / 24.6   | 135            | pass/pass; pass/pass                | **+4.4 %** / −1.3 % (noise: unchanged path) |

Known failing on main from 39916b3: `kv_gpu prefix_reuse_matches_cold` (Llama's speed-tuned GEMMs round by m, so a warm suffix prefill diverges from the cold one at a near-tie). The fix (decision "Pre-Phase-5 #1 follow-up": batch-invariant Llama prefill buckets) is in progress.

| Date       | Commit  | Change                                                                                                                                                                                                                                     | Llama tok/s | Llama ITL p50 | Llama TTFT p50 | OLMoE tok/s | OLMoE ITL p50 | OLMoE TTFT p50 | Golden (Llama c1/c16; OLMoE c1/c16) | vs previous                                                                                                                                                                                                      |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- | ------------- | -------------- | ----------- | ------------- | -------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-09-27 | 6f6e18f | #1 follow-up: prefix reuse bit-exact again (prefill steps use batch-invariant Llama GEMM rows, decode steps keep the speed rows; planner leaves ≥ 2 prompt tokens to prefill); `kv_gpu` 4/4 incl. `prefix_reuse_suffix_lengths_match_cold` | 826.4       | 15.9          | 210            | 596.0       | 24.6          | 135            | pass/pass; pass/pass                | **−4.3 % Llama, over the 3 % bound: landed anyway** (it fixes a correctness regression on main; mixed decode+prefill steps take the slower invariant rows for their decode rows too); recovery follow-up started |

#2 OLMoE MoE prefill (branch `perf-moe-prefill`; decision "Pre-Phase-5 #2"): new `turbine_hip_moe_wmma_prefill` above the 512-row tier, bitwise equal to `turbine_hip_moe_wmma`. Kernel level only (`perf moe_prefill_timings`, real routing, GPU 0), µs per `moe_experts` call at 512 / 2,048 / 16,384 routed rows: 1,550–1,660 / 1,680–1,740 / 2,990–3,100 → 1,415–1,500 / 1,552–1,579 / 2,500–2,560; decode tier unchanged. Served lab-bench (both models, `--golden16`) pending with the coordinator; expected OLMoE TTFT p50 ≈ −10 % (135 → ≈ 122 ms), tok/s ≈ +1 %, Llama unchanged.
| 2026-09-28 | f49149f | #2 OLMoE MoE prefill: own `turbine_hip_moe_wmma_prefill` (weights straight to WMMA registers, same accumulation order; CK grouped GEMM, hipBLASLt grouped/per-expert, llama.cpp `mul_mat_id`, vLLM/SGLang evaluated) | 854.9 | 15.4 | 208 | **613.5** | 24.3 | **119** | pass/pass; pass/pass | OLMoE +2.9 %, TTFT −12 %; Llama path unchanged (+3.4 % vs the 826.4 row: that row looks low) |

Close-out (2026-09-28, main 46b1957): `scripts/gate.sh --full` and the full GPU suite green after 2d1e9e5 (596/0 apart from the slow-test list fixed there); lab-bench Llama 855.2 tok/s (ITL 15.4, TTFT 208), OLMoE 613.9 (ITL 24.3, TTFT 118), golden c1/c16 pass both; multi-turn cached ratio 0.907, later-turn TTFT 76 vs 460 ms (0.17×); 10-minute overload soak with L1 on: pass (ITL p99 205 vs calibration 174 ms). The first close-out soak failed (393 vs 176 ms): Phase 4's pressure reclaim copied every one-off block to L1 (A/B with `kv.cpu.enabled=false` passed at 205 vs 173); fixed by 46b1957. All runs are in labbook, set `pre-phase5-perf`. Against the Phase 4 baseline: Llama +12.3 % (1.20× vLLM 715), OLMoE −0.1 % tok/s with TTFT −13 % (1.15× vLLM 535).

## Phase 6a: quantization (branch `phase-6a-quantization`, labbook set `phase-6a-quantization`)

Workload unchanged (`scripts/lab-bench.sh`, novanas GPU 0, 16 concurrent, 512-word prompts, 256 tokens, 200 requests; BF16 models on the phase2c configs). No-regression bound for the BF16 paths at every landing: c16 tok/s ≥ 0.98 × and TTFT p50 ≤ 1.10 × the phase-start baseline.

| Date       | Commit  | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Model                    | tok/s  | ITL p50 (ms) | TTFT p50 (ms) | golden c1 / c16                    |
| ---------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------ | ------------ | ------------- | ---------------------------------- |
| 2026-09-29 | 39d54d1 | Phase-start baseline (Tasks 1–5 landed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Llama                    | 854.7  | 15.4         | 207           | PASS / PASS                        |
| 2026-09-29 | 39d54d1 | Phase-start baseline (Tasks 1–5 landed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | OLMoE                    | 613.4  | 24.5         | 118           | PASS / PASS                        |
| 2026-09-29 | 3635f68 | Task 14 proof run, bf16 pass (Llama BF16 reference for the two FP8 passes below; GSM8K-200 c16 0.795, 159/200)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Llama BF16               | 849.7  | 15.5         | 207           | – (not run this pass)              |
| 2026-09-29 | 3635f68 | **Task 14 proof: FP8-dynamic (RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic) — golden PASS/PASS; tok/s 1.149× bf16 (PASS ≥1.10×); c1 ITL 0.846× bf16 (MISS, target ≤0.75×); GSM8K 0.79 (158/200) vs BF16 alongside 0.795 drop 0.005 (PASS, bound 0.02) but vs vLLM-ROCm same checkpoint (0.82, 164/200) drop 0.03 (MISS, binding gate.json bound 0.01)**                                                                                                                                                                                                                                                                                                                                                                                                                        | Llama FP8-dynamic        | 976.4  | 13.9         | 155           | PASS / PASS                        |
| 2026-09-29 | 3635f68 | Task 14 proof: FP8 per-tensor (RedHatAI/Llama-3.2-3B-Instruct-FP8) — golden judged only against the tree's provisional BF16 tolerance (FAIL 1/16 c1, 3/16 c16; real tolerance waits on `t14_spread.sh`, queued behind fixture.lock); tok/s 1.149× bf16 (PASS); c1 ITL 0.855× bf16 (MISS, target ≤0.75×); GSM8K not run this pass                                                                                                                                                                                                                                                                                                                                                                                                                                             | Llama FP8 (tensor)       | 977.0  | 14.0         | 150           | FAIL / FAIL (provisional, pending) |
| 2026-09-29 | 3635f68 | Task 14 proof: vLLM-ROCm 0.23.0 on the same FP8-dynamic checkpoint (status SERVED) — baseline for the gate above; GSM8K 0.82 (164/200)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Llama FP8-dynamic (vLLM) | 651.1  | 19.8         | 258           | n/a                                |
| 2026-09-30 | ee6bfb1 | Task 15 proof: FP8 block-scaled own kernel `turbine_hip_fp8_block` (W8A16; RedHatAI-style unsloth/Llama-3.2-3B-Instruct-FP8-Block) — golden PASS/PASS (16/16 both, tolerance 0.19/0.55 strict, 0.25/0.75 batched from the self-spread, floor-bound since spread max 0.1829/0.3107 is below the BF16 floor); tok/s 1062.4 (1.24× BF16 floor 854.7, PASS ≥1.0×); c1 ITL 8.43 ms / tok/s_c1 112.7; no `fp8_block_decoded` or `CIRCUIT_OPEN` in the server log; GSM8K 0.800 vs BF16 0.805 drop 0.005 (PASS, bound 0.02); vLLM-ROCm same checkpoint 677.8 tok/s (Turbine 1.57×, PASS ≥0.9×); 10-min overload soak PASS (ITL p99 216.4 vs calibration 186.6 ms, GREEN 30 s after cool-down); support row flipped to `supported` (amd/gfx1201/LlamaForCausalLM/fp8_block/bf16/none) | Llama FP8-block          | 1062.4 | 11.5         | 226           | PASS / PASS                        |
| 2026-09-30 | 2a53dbb | Task 18 proof: `gptq_int4` on kaitchup/Llama-3.2-3B-Instruct-AutoRoundGPTQ-4bit (decision 2026-09-30 B) — golden PASS/PASS (16/16, BF16 3B bounds; spread max 0.1166/0.4000); 10-min soak PASS 8/8 incl. `reached_orange`; full GSM8K 0.7566 (drop 0.0235 ≤ 0.04); labbook 1659f427                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Llama GPTQ (AutoRound)   | 1267.3 | 9.1          | 228           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit (review fixes in): BF16 Llama, 0.999× phase start                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Llama                    | 854.2  | 15.4         | 207           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit: BF16 OLMoE, 0.982× phase start (bound ≥ 0.98×, holds by 0.002; watch item)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | OLMoE                    | 602.5  | 24.8         | 119           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Llama FP8-dynamic        | 960.3  | 14.1         | 155           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Llama FP8 (tensor)       | 957.6  | 14.2         | 151           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Llama FP8-block          | 1049.9 | 11.6         | 228           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Llama AWQ                | 1261.3 | 9.2          | 224           | PASS / PASS                        |
| 2026-09-30 | fbddca9 | Task 29 exit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Llama GPTQ (AutoRound)   | 1268.9 | 9.1          | 222           | PASS / PASS                        |
| 2026-09-30 | 7ad4b03 | Task 29 exit: FP8 KV against the new emulated-KV fixtures (tolerance 0.40 / 2.44)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Llama FP8 KV             | 829.7  | 15.7         | 213           | PASS / PASS                        |
| 2026-09-30 | 7ad4b03 | Task 29 exit: FP8 KV against the new emulated-KV fixtures (tolerance 1.10 / 2.01) — 13/16 (need 14); row demoted to `experimental` (user decision 2026-09-30 B)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | OLMoE FP8 KV             | 631.1  | 23.5         | 127           | FAIL / FAIL                        |

6a summary (2026-09-30, Task 29): every `supported` quantized row beats the BF16 Llama baseline at c16 — FP8-dynamic 1.12×, FP8 per-tensor 1.12×, FP8-block 1.23×, AWQ 1.48×, GPTQ (AutoRound) 1.49×; Llama FP8 KV 0.97× (the KV format saves memory, not time); BF16 Llama 0.999× and OLMoE 0.982× the phase start. Open perf items: FP8 c1 ITL (Task 14b), an MXFP4 prefill kernel at hipBLASLt speed, OLMoE BF16 at the edge of the 0.98× bound.

## Phase 6b: KV compression (branch `p6b-stack`, labbook set `phase-6b-kv-compression`)

Task 6, per-tier FP8 (`kv.cpu.format=fp8_e4m3`, `kv.cpu.max_bytes=4GiB`), Llama-3.2-3B BF16 weights and BF16 KV pages, novanas GPU 0, phase2c config. Phase-6a exit baseline for the BF16 path: 854.2 tok/s, ITL p50 15.4 ms, TTFT p50 207 ms.

| Date       | Commit  | Change                                                                                                                                                                | tok/s | ITL p50 (ms) | TTFT p50 (ms) | golden c1 / c16 |
| ---------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------ | ------------- | --------------- |
| 2026-10-01 | d6e564e | Task 6: `scripts/lab-bench.sh --model llama --golden16 -- --set kv.cpu.format=fp8_e4m3 --set kv.cpu.max_bytes=4GiB` (0.995x of the 6a baseline; labbook run 7683e112) | 849.6 | 15.4         | 208           | PASS / PASS     |

Multi-turn (`turbine-bench --profile multi-turn --turns 8 --shared-prefix-words 2000 --session-hints`, client on novanas, `lab-serve.sh` with `phase2c-novanas-llama.yaml --set kv.cpu.format=<f> --set kv.cpu.max_bytes=4GiB`, one run per cell):

| Workload                                        | L1 format  | Requests ok | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | L1 lookups | lossy_cached_tokens |
| ----------------------------------------------- | ---------- | ----------- | ------------------- | ------------------------------ | ---------- | ------------------- |
| spec command: 16 sessions, c8                   | `l0`       | 128/128     | 0.9064              | 75.7 / 109                     | 0          | 0                   |
| spec command: 16 sessions, c8                   | `fp8_e4m3` | 128/128     | 0.9056              | 76.4 / 137                     | 0          | 0                   |
| stressed: 32 sessions, c32, `--think-time 1..4` | `l0`       | 256/256     | 0.6180              | 2102 / 23370                   | 2062       | 0                   |
| stressed: 32 sessions, c32, `--think-time 1..4` | `fp8_e4m3` | 256/256     | 0.6324              | 522 / 8502                     | 2016       | 384                 |

The spec command never reaches L1 for reuse (L0 holds the 16 sessions), so both ratios equal; the stressed run (L0 of 585 blocks is smaller than the live histories) is the discriminating one. 64 sessions at c64 overloaded the server (SURVIVAL, 503 `overloaded`, 350 of 512 requests failed in both arms) and is not used. An earlier fp8 run of the stressed workload lost 65 of 256 requests to 503 `overloaded` (ratio 0.735, not comparable); the recorded cell is its rerun.

L1 capacity in the same 4 GiB (`/turbine/v1/kv` `formats`, added in this task): `l0` 292 blocks; `fp8_e4m3` 438 fp8 blocks (7.34 MB each) plus 60 blocks at the lossless-tail `l0` format = 498 blocks in 4.10 GB of 4.29 GB, i.e. 1.71x at 95.5 % fill, about 1.78x when full with the observed 12 % tail share, 2.0x for pure fp8 (584). The kv_sim AC (>= 1.9x) assumes no tail blocks.

The planner chose `recompute_cheaper` for 155 of the stressed fp8 run's requests even with L1 lookups at 2016 (lossy penalty 0.1 on the fp8 copies plus the measured L1 to L0 rate), so few lossy blocks were reused (384 tokens = 3 blocks) and the ratio gain is 0.014.

GSM8K-200 at concurrency 16 with `kv.cpu.format=fp8_e4m3` (server after the stressed run): 162/200 = 0.810 (`tests/eval/llama-3.2-3b-instruct/turbine-l1-fp8.json`) against `turbine-bf16-c16.json` 0.795: `eval-compare --max-drop 0.01` PASS. The eval's prompts share no full block, so no lossy block was reused during it (`lossy_cached_tokens` unchanged across the eval); it shows the tier configuration does not disturb serving, not the quality of lossy reuse. That is `kv_gpu::lossy_tier_reuse` (worst first-8 |delta logprob| 0.067, `x-turbine-kv-lossy: deny` bit-equal to cold).

### Planner recompute investigation (decision "6b Task 6", point 3, A; branch `p6b-planner`)

Same stressed workload as Task 6 (`turbine-bench --profile multi-turn --sessions 32 --turns 8 --concurrency 32 --shared-prefix-words 2000 --think-time 1..4 --session-hints`, client on novanas), served by `lab-serve.sh` with `phase2c-novanas-llama.yaml` plus `logging.level: info,turbine_kv=debug,turbine_server::kv_orchestrator=debug` and `--set kv.cpu.format=<f> --set kv.cpu.max_bytes=4GiB`, GPU 0 unless noted, one serve Job per row. The DEBUG events `kv_plan` (planner inputs and costs), `kv_copy_timed` and `kv_prefill_rate` (c420c09) give the estimator's inputs at decision time.

Root cause: copies that end on the GPU copy stream (L0 → L1, L1 → L0, the device transcode) were timed to the poll that saw their event done, once per engine iteration (25–200 ms here). Startup calibration measures L1 → L0 at 10.45 GB/s (1.4 ms per 14.7 MB block); the observed L1 → L0 "durations" were 23–149 ms, so after a handful of promotions the latency estimate sat at 63 ms (`l0`) / 97 ms (`fp8_e4m3`) and every L1 block priced above recomputing its 128 tokens (prefill EWMA 7–12k tok/s, 11–18 ms per block). With no further promotions the estimate never saw another sample: in the pre-fix fp8 run only 8 L1 → L0 copies were ever observed and 154 of 156 L1-hit plans recomputed. Not the cause: the lossy penalty (0.1, fp8 copies are half-size), in-flight caps (no truncated plans), L0 pressure (no `l0_pressure` plans), the lossless-tail rule, session hints.

Fix: 69769e2 (a copy seen done only at a poll is `CopyTime::Within { at_least, at_most }`, folded in clamped to its bounds) and d9b5f92 (the clamped value is the path's unloaded, calibrated cost, so a slow burst no longer pins the estimate at the poll interval).

| Run                               | Commit  | Serve run id        | ok      | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | L1 lookups | recompute / retrieve plans | lossy_cached_tokens | SURVIVAL |
| --------------------------------- | ------- | ------------------- | ------- | ------------------- | ------------------------------ | ---------- | -------------------------- | ------------------- | -------- |
| `l0` before                       | c420c09 | 1001033606-2395994b | 174/256 | 0.7638              | 191 / 25145                    | 830        | 74 / 6                     | 0                   | 1        |
| `fp8_e4m3` before                 | c420c09 | 1001034410-0eb35462 | 256/256 | 0.6240              | 653 / 8347                     | 2090       | 154 / 2                    | 768                 | 0        |
| `l0` bound, current prior (GPU 1) | 69769e2 | 1001041537-02a98dff | 256/256 | 0.7762              | 381 / 34739                    | 1273       | 114 / 41                   | 0                   | 0        |
| `fp8_e4m3` bound, current prior   | 69769e2 | 1001041816-09badab7 | 193/256 | 0.8012              | 162 / 38578                    | 705        | 73 / 36                    | 26624               | 1        |
| `l0` after                        | d9b5f92 | 1001043044-2fcc68d3 | 184/256 | 0.7911              | 143 / 22517                    | 813        | 49 / 30                    | 0                   | 1        |
| `l0` after                        | d9b5f92 | 1001043549-023f6b3f | 256/256 | 0.6359              | 597 / 8467                     | 1959       | 111 / 29                   | 0                   | 0        |
| `l0` after                        | d9b5f92 | 1001044056-042c265a | 256/256 | 0.6470              | 445 / 8220                     | 1896       | 98 / 38                    | 0                   | 0        |
| `fp8_e4m3` after                  | d9b5f92 | 1001043218-2938a6a8 | 143/256 | 0.8569              | 128 / 1164                     | 480        | 36 / 32                    | 44160               | 1        |
| `fp8_e4m3` after                  | d9b5f92 | 1001043815-041607a7 | 214/256 | 0.7873              | 227 / 43094                    | 740        | 86 / 45                    | 74240               | 1        |
| `fp8_e4m3` after                  | d9b5f92 | 1001044322-1dfee52b | 175/256 | 0.8202              | 182 / 22043                    | 549        | 35 / 48                    | 87040               | 1        |

Retrievals rose from 2–6 to 29–48 plans and lossy reuse from 768 to 44k–87k tokens, but the comparison is confounded: the workload sits at the overload edge and a run that trips SURVIVAL (`exhaustion_horizon`, 10 s of 503 `overloaded`) loses its later turns, which raises the ratio over the requests that succeed. The fp8 arm tripped it in 3 of 3 runs after the fix (once right after a 64-block promotion burst), against 0 of 1 before (Task 6 also lost one earlier fp8 run to it); `l0` 1 of 3 after, 1 of 1 before. `cached_tokens_ratio` across arms is not comparable until the runs are clean.

Remaining `recompute_cheaper` plans (35–111 per run): every one ran while the L1 → L0 latency estimate was above 1 ms after a genuinely slow burst. Each slow burst (promotions seen running 33–255 ms) coincided with 250–800 MB of pressure demotions (L0 → L1, calibrated at only 1.6 GB/s device-to-host) on the same FIFO copy stream: the promotions really waited, but the planner sums that shared wait once per block (`k × latency`).

### Pinned D2H (decision "6b Task 6" follow-up 4, A; branch `p6b-d2h`)

Question: startup calibration priced L0 → L1 (pinned device-to-host) at 4.16 GB/s (6a log, 2026-09-30) or 1.6 GB/s (6b planner runs) against ~10.6 GB/s host-to-device on GPU 0. Both cards sit behind a Gen5 x8 root port (`00:01.0` / `00:01.1`, 32 GT/s x8; the x16 links are the card's own switch).

Cause (Turbine usage): `ShimContext::copy_async` fenced the compute stream (event create + record + `hipStreamWaitEvent`) and recorded a completion event for every device-source copy, and the KV orchestrator issued one copy per 512 KiB layer segment (28 per Llama block). Host-to-device copies have no fence, hence the asymmetry. The calibration also timed the first copies on a fresh copy stream, D2H first.

Plain-HIP microbenchmark (`hipHostMalloc` default, non-blocking streams, GPU 0 under `bench.lock`, 64 MiB in 256 KiB segments scattered like the pool's layers; cold = first pass on a new stream):

| Variant                                       | D2H cold | D2H warm  | H2D warm  |
| --------------------------------------------- | -------- | --------- | --------- |
| fence + event per segment (Turbine before)    | 3.1–3.5  | 4.4       | 9.5       |
| no fence, event per segment                   | 6.1–6.5  | 10.3      | 9.5       |
| one fence + one event per block (56 segments) | 6.6–6.9  | 11.1–11.5 | 10.2–10.4 |
| one 64 MiB copy (SDMA ceiling)                | —        | 12.5      | 12.7      |

GB/s. Not the cause: pinned vs registered memory (`hipHostMalloc` throughout), slab size (64 MiB vs 1 GiB), device stride, host first touch, NUMA (one node). `HSA_ENABLE_SDMA=0` (blit kernels) reaches 18.5 GB/s H2D on one large copy but is slower at segment sizes, so SDMA's ~12.5 GB/s is the practical ceiling of this path.

Fix (6be446b): `CopyEngine::copy_async_batch` — the shim resolves every end, fences once, enqueues the segments and records one event (one ticket per block per shard); `stream_copies` uses it; `calibrate_l1` runs one untimed pass each way before timing.

| Measure (GPU 0 unless noted)                                    | Before                   | After                                                   |
| --------------------------------------------------------------- | ------------------------ | ------------------------------------------------------- |
| `kv_calibration` l0_to_l1 / l1_to_l0 (GB/s)                     | 4.16 / 10.63             | 11.57 / 11.52                                           |
| lab `pinned_block_batches_d2h_keeps_up_with_h2d` (k3s Job card) | per segment 5.21 / 8.80  | batched 9.42 / 9.22 (ratio 1.02)                        |
| `lab-bench --quick` (golden c1, tok/s, ITL p50, TTFT p50)       | 854.2 (6a exit, 200 req) | PASS, 863.7, 15.4 ms, 225 ms (64 req; labbook 71350afa) |

### Mixed-format paged attention (Task 12; branch `p6b-t12`, ee7b950)

Kernel timings (`hip_ops paged_mixed_timings`, GPU 0 under `bench.lock`, pools rotated over ≥ 256 MiB, append included) are in the decisions entry "P6b: mixed-format / TurboQuant paged attention — provider evaluation", result paragraph. Decode tq4 / BF16 CK: Llama 1.26× (b1 @768), 1.46× (b16 @768), 1.15× (b16 @2k); OLMoE 1.08×, 0.90×, 0.58×. Staged CK prefill tq4 / BF16 3.95–7.31× (TurboQuant append encode included). Served ITL with `kv.dtype: tq4` / `tq2` is Task 13.

BF16 KV after the attention changes (`scripts/lab-bench.sh --quick`, 64 requests, client on novanas, golden c1 PASS both), against the 6a-exit baseline (200 requests):

| Date       | Commit  | Model | tok/s | ITL p50 (ms) | TTFT p50 (ms) | vs baseline (tok/s, TTFT)   |
| ---------- | ------- | ----- | ----- | ------------ | ------------- | --------------------------- |
| 2026-10-01 | de6f948 | Llama | 877.3 | 15.4         | 247           | 1.027× (854.2), 1.19× (207) |
| 2026-10-01 | de6f948 | OLMoE | 623.0 | 24.7         | 126           | 1.034× (602.5), 1.06× (119) |

de6f948 is the pre-squash tip; its tree is ee7b950's. The Llama quick TTFT (247 ms) sits above the 1.10× bound of the 200-request baseline, as the earlier quick run did (225 ms, 64 requests, `p6b-d2h` above); the quick run's 64 requests weigh the start-up burst more. The bound is judged on the 200-request phase-exit run, not here.

### Planner follow-ups re-measured (decision "6b planner follow-ups", 1 A, 2 A, 3 A; branch `p6b-planner2`)

Commits under test: 70ea0ce (path latency once per plan), 2df40ad (promotions ahead of demotions on the copy stream), c206b19 (slow estimates decay toward the calibration, half-life 5 s), ef244fa (static-mode timing, not on this path). Server tree dbcd614. Same workload as above (`turbine-bench --profile multi-turn --turns 8 --shared-prefix-words 2000 --think-time 1..4 --session-hints`, client on novanas) with `--sessions N --concurrency N`, `lab-serve.sh` with `phase2c-novanas-llama.yaml --set kv.cpu.format=<f> --set kv.cpu.max_bytes=4GiB`, every serve Job on GPU 0, one fresh server per row. Plans, L1 lookups, lossy tokens and SURVIVAL from the server's `/metrics` after the run (`turbine_kv_plans_total`, `turbine_kv_lookups_total`, `turbine_kv_lossy_cached_tokens_total`, `turbine_pressure_transitions_total{to="SURVIVAL"}`).

Finding the load below SURVIVAL (step down from the Task 6 stressed settings until no run trips it):

| Sessions / c | L1 format  | Serve run id        | ok      | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | recompute / retrieve plans | lossy_cached_tokens | SURVIVAL |
| ------------ | ---------- | ------------------- | ------- | ------------------- | ------------------------------ | -------------------------- | ------------------- | -------- |
| 32 / 32      | `fp8_e4m3` | 1001090845-268475a9 | 186/256 | 0.8388              | 167 / 24928                    | 19 / 53                    | 97408               | 1        |
| 24 / 24      | `fp8_e4m3` | 1001093332-12db5255 | 155/192 | 0.8839              | 145 / 14370                    | 0 / 51                     | 137984              | 1        |
| 20 / 20      | `fp8_e4m3` | 1001094018-3cbef529 | 160/160 | 0.8810              | 88 / 600                       | 7 / 41                     | 79616               | 0        |
| 20 / 20      | `fp8_e4m3` | 1001094533-1aa82c1e | 160/160 | 0.8999              | 99 / 383                       | 0 / 53                     | 115328              | 0        |
| 20 / 20      | `l0`       | 1001094346-268a78ce | 150/160 | 0.8432              | 90 / 959                       | 0 / 46                     | 0                   | 1        |

Every SURVIVAL entry here is `GREEN → SURVIVAL` on `exhaustion_horizon` (the reliability forecast), never a KV-utilisation climb. 20 sessions still tripped it once (`l0`), so the A/B runs at 16 sessions, c16, where none of six runs did:

| L1 format  | Serve run id        | ok      | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | L1 lookups | recompute / retrieve plans | lossy_cached_tokens | SURVIVAL |
| ---------- | ------------------- | ------- | ------------------- | ------------------------------ | ---------- | -------------------------- | ------------------- | -------- |
| `l0`       | 1001094722-33d84df4 | 128/128 | 0.9029              | 83.9 / 257                     | 290        | 0 / 29                     | 0                   | 0        |
| `l0`       | 1001094943-0d7025d6 | 128/128 | 0.9065              | 80.4 / 392                     | 191        | 0 / 25                     | 0                   | 0        |
| `l0`       | 1001095216-35931fb7 | 128/128 | 0.9059              | 74.6 / 329                     | 220        | 1 / 27                     | 0                   | 0        |
| `fp8_e4m3` | 1001093713-05302d5a | 128/128 | 0.9027              | 77.8 / 420                     | 291        | 2 / 21                     | 48512               | 0        |
| `fp8_e4m3` | 1001094830-3695e953 | 128/128 | 0.9042              | 78.4 / 290                     | 209        | 0 / 26                     | 42624               | 0        |
| `fp8_e4m3` | 1001095100-0fcb95f7 | 128/128 | 0.8785              | 81.7 / 561                     | 328        | 7 / 19                     | 30336               | 0        |

Medians of 3 (16 sessions, c16):

| L1 format  | cached_tokens_ratio | later-turn TTFT p50 (ms) | recompute / retrieve plans | lossy_cached_tokens |
| ---------- | ------------------- | ------------------------ | -------------------------- | ------------------- |
| `l0`       | 0.9059              | 80.4                     | 0 / 27                     | 0                   |
| `fp8_e4m3` | 0.9027              | 78.4                     | 2 / 21                     | 42624               |

At this load both arms are equal within run-to-run spread (ratio −0.003, TTFT −2 ms): L0 holds most of the 16 live histories, L1 serves 191–328 lookups per run and the fp8 tier's extra capacity is not needed. Recomputes of L1 hits are now rare (0–7 plans, against 35–111 per run before these commits at 32 sessions); at 32 sessions the fp8 arm still recomputed 19 plans, all in the minutes around its SURVIVAL episode. The remaining SURVIVAL trips at 20–32 sessions come from `exhaustion_horizon` (decision point 3 B, reliability code, not this branch).

Shared-prefix eval (`tests/eval/gsm8k-200-shared-prefix.jsonl`, c16, `phase6-novanas-llama.yaml --set kv.gpu.max_bytes=4GiB --set kv.cpu.max_bytes=4GiB`, `--filler-words 2000`): BF16 KV (`kv.cpu.format=l0`) 8 fillers 0.755 (1001095429-021df70e), 16 fillers 0.775 (1001095706-39bc19c8); FP8 L1 16 fillers 0.790 (1001095956-0c96bdb6) with 0 lossy-cached tokens, so its `--min-lossy-cached-ratio 0.5` guard failed (exit 1) and no gate pair exists. In all three the prefix (2,944 tokens, 23 blocks) stayed in L0 (0 demotions, 0 L1 lookups, 585,856 of 629,014 prompt tokens cached from L0): capacity demotion cannot take a prefix whose L0 children lack reuse evidence (`p6b-planner2` handoff, options A–C).

### SURVIVAL at 20–32 multi-turn sessions (decision "6b: SURVIVAL at 20–32 multi-turn sessions", A; branch `p6b-survival`)

Same workload as above (`turbine-bench --profile multi-turn --turns 8 --shared-prefix-words 2000 --think-time 1..4 --session-hints`, `--sessions N --concurrency N`, client on novanas), `lab-serve.sh` with `phase2c-novanas-llama.yaml --set kv.cpu.format=l0 --set kv.cpu.max_bytes=4GiB` (585 L0 blocks), GPU 0, one fresh server per row. `/turbine/v1/pressure` and `/turbine/v1/kv` were polled every 100 ms.

Before, e02e8c2, 32 sessions (serve 1001102511-200e23db): 184/256 ok. GREEN → SURVIVAL on `exhaustion_horizon` 31 s in. In the 100 ms before the transition L0 went from 563 to 585 of 585 blocks referenced (0 free, 0 cached), while `kv_utilization` read 0.70 (104 blocks reserved, about 308 committed). The horizon went from +∞ (growth still fit the 22 free blocks) to 0 in one tick. The server logged `decode_deferred reason="kv_exhausted"` for admitted decodes just before. The forecast was right: L0 really was full. The ledger was wrong: a request's reservation leaves out the cached prefix blocks it attaches, so blocks taken from the cache were in no reservation, and ~173 referenced blocks were invisible to `kv_utilization` and to admission. That is why YELLOW, ORANGE and RED never fired.

After, 083b1c7 (the engine reports L0's referenced blocks to the ledger as `held`):

| Sessions / c | Serve run id        | ok      | cached_tokens_ratio | later-turn TTFT p50 / p95 / p99 (ms) | max kv_utilization | deepest state | SURVIVAL | decode_deferred |
| ------------ | ------------------- | ------- | ------------------- | ------------------------------------ | ------------------ | ------------- | -------- | --------------- |
| 32 / 32      | 1001104326-1aa26258 | 256/256 | 0.7487              | 3868 / 35676 / 43617                 | 0.959              | RED           | 0        | 0               |
| 24 / 24      | 1001105324-0a5182ca | 192/192 | 0.8191              | 162 / 36284 / 39656                  | 0.962              | RED           | 0        | 0               |
| 20 / 20      | 1001105601-324c03b7 | 160/160 | 0.8827              | 114 / 6994 / 7863                    | 0.894              | ORANGE        | 0        | 0               |
| 16 / 16      | 1001105744-298258fb | 128/128 | 0.9056              | 78 / 139 / 180                       | 0.714              | GREEN         | 0        | 0               |

All four counts now stay out of SURVIVAL, and none fails a request (before: 32 → 184–186/256, 24 → 155/192, 20 → 150/160 in 1 of 3). The live histories at 20+ sessions exceed L0. Pressure now climbs one state at a time (kv_utilization, then YELLOW demotions and ORANGE/RED admission queueing: 32 sessions queued 55 requests in ORANGE/RED, 14 s mean wait). So the overload shows up as tail TTFT instead of 503 `overloaded`. Before, TTFT counted only the requests that succeeded. 16 sessions is unchanged (78 vs 80 ms p50). One run per row.

10-minute overload soak on 59e36ff (`scripts/overload-soak.sh novanas --duration 10m`, serve 1001112421-15fb3885): PASS, all 8 checks true; ITL p99 211 ms (calibration 176 ms), GREEN 24 s into the cool-down, 4385 × 200, 72 × 503 `overloaded`, 2421 `queue_timeout`.

### Copy ahead and the shared-prefix FP8 gate (decision "6b: shared-prefix eval never demotes the prefix to L1", A; branch `p6b-copyahead`)

Copy ahead (a8c3b7c): at GREEN and YELLOW, when capacity demotion or the controller's reclaim would demote, the blocks of shared prefixes held in L0 by their children are copied into L1 while the L0 copy stays. Once the children leave, the L0 copy is freed with no further copy. Shared-prefix eval, c16, `phase6-novanas-llama.yaml --set kv.gpu.max_bytes=4GiB --set kv.cpu.max_bytes=4GiB`, `--filler-requests 32 --filler-words 2000` (32 fillers, not 8 or 16: the first two items' own blocks must be reclaimed from L0 before the other 198 items arrive). Every kept run's server counted exactly the eval's own 726,723 prompt tokens:

| Run                      | Commit  | L1 format  | Serve run id        | Accuracy | cached ratio | lossy cached (ratio) | copies ahead | result                                                  |
| ------------------------ | ------- | ---------- | ------------------- | -------- | ------------ | -------------------- | ------------ | ------------------------------------------------------- |
| `turbine-bf16-sp.json`   | a8c3b7c | `l0`       | 1001114618-27e26a5b | 0.780    | 0.931        | 0 (0)                | 23           | baseline                                                |
| `turbine-l1-fp8-sp.json` | a8c3b7c | `fp8_e4m3` | 1001114752-39f57ca5 | 0.770    | 0.931        | 582,912 (0.927)      | 23           | `eval-compare` PASS at the bound (drop 0.010, max 0.01) |
| (not kept)               | a8c3b7c | `fp8_e4m3` | 1001111446-1e7e5c52 | 0.775    | 0.931        | 582,912 (0.927)      | 23           | second clean fp8 run; drop 0.005 against the baseline   |
| (not kept)               | e02e8c2 | `fp8_e4m3` | 1001114920-1b5634b9 | 0.775    | 0.931        | 582,912 (0.927)      | —            | base commit, same recipe                                |

Paired, the gate pair differs on 14 items: 8 are right only at BF16, 6 only at FP8. Planner2 saw 0.755–0.790 across exact-KV runs at c16. In every run the 23 prefix blocks left L0 once the head items' blocks were reclaimed and came back from L1 once (23 L1 lookups); every later item reused the promoted copy. The base commit demotes the prefix too with 32 fillers (23 demotions after the children left), so planner2's 0 lossy tokens came from too few fillers. Copy ahead makes that drop free (the L1 copy already exists) and starts the copy at GREEN.

Discarded, because another builder's overload soak sent traffic to whatever server held port 18000 between about 11:00 and 11:18 UTC: 1001105753-3afb91e2 (BF16, 0.775, 2,943,726 prompt tokens), 1001110311-3d1b4cef and 1001110705-21e8b128 (fp8, 503 at filler 2–3 with L0 RED; 4.9M and 4.5M prompt tokens) and 1001111055-19904604 (base, 0.770, 1.77M). The first committed pair (565723b) was replaced by the rerun (7ebf9b8).

`scripts/lab-bench.sh --quick --model llama -- --set kv.cpu.enabled=true --set kv.cpu.max_bytes=4GiB` on 40cad6e: golden c1 PASS, 877.1 tok/s, ITL p50 15.4 ms, TTFT p50 246 ms (64 requests, client novanas). The de6f948 quick run without L1 measured 877.3 / 15.4 / 247, so there is no BF16 regression (the bench shares no prefix, so copy ahead stays idle).

### FP8 lower-tier gate on 3 + 3 runs (decision "6b: FP8 lower-tier shared-prefix gate passes exactly at the bound", A; branch `p6b-fp8gate`)

Same recipe as above on the f9a4c91 tree (a fresh server per run, GPU 0 under the bench lock, c16, 32 fillers, 4 GiB L0 and L1, `phase6-novanas-llama.yaml`); the fp8 runs add `--set kv.cpu.format=fp8_e4m3` and `--min-lossy-cached-ratio 0.5` (all passed). Every server counted exactly the eval's own 726,723 prompt tokens (`turbine_kv_prompt_tokens_total`); cached ratio 0.931 and lossy cached ratio 0.927 (582,912 of 629,014 tokens) in all three fp8 runs.

| Arm    | Report                      | Serve run id        | Accuracy    |
| ------ | --------------------------- | ------------------- | ----------- |
| BF16 1 | `turbine-bf16-sp-r1.json`   | 1001131143-01812869 | 0.780 (156) |
| BF16 2 | `turbine-bf16-sp-r2.json`   | 1001131324-3eaf9425 | 0.785 (157) |
| BF16 3 | `turbine-bf16-sp-r3.json`   | 1001131447-036a8926 | 0.770 (154) |
| FP8 1  | `turbine-l1-fp8-sp-r1.json` | 1001135450-3cc7df5a | 0.775 (155) |
| FP8 2  | `turbine-l1-fp8-sp-r2.json` | 1001135611-001242f8 | 0.775 (155) |
| FP8 3  | `turbine-l1-fp8-sp-r3.json` | 1001135732-13b1af29 | 0.770 (154) |

Medians: BF16 0.780, FP8 0.775: drop 0.005 (max 0.01), PASS. Paired exact two-sided McNemar (binomial on the discordant items, alpha 0.05, as for 6a's FP8 KV) on all nine BF16 x FP8 pairs: lost/gained 5/4, 7/6, 7/5, 5/3, 7/5, 7/4, 4/5, 4/5, 6/6; p between 0.55 and 1.0, none significant (median pair BF16 1 vs FP8 1: 5/4, p = 1.0; pooled 52 lost, 43 gained). `kv_gpu` on gfx1201 (job 1001135903-1ec4bf89): 12 passed, 0 failed (the 10 tests plus the config-load tests). Lower-tier `fp8_e4m3` flipped to `supported`.

### Queued-prefix demotion (decisions "6b: after the held-prefix ledger fix", 1 B, and "6b: queued-prefix demotion — granularity and scope", 1 A, 2 A; branch `p6b-queued-demote`)

Same workload and server as the SURVIVAL entry above (`turbine-bench --profile multi-turn --turns 8 --shared-prefix-words 2000 --think-time 1..4 --session-hints`, `--sessions N --concurrency N`, client on novanas; `phase2c-novanas-llama.yaml --set kv.cpu.format=l0 --set kv.cpu.max_bytes=4GiB`, 585 L0 blocks; one fresh server per row; `/turbine/v1/pressure` and `/turbine/v1/kv` polled every 100 ms). Before is 76af3cb (= `p6b-stack` f9a4c91, with copy ahead). After is 9d4cf27: at YELLOW/ORANGE the reclaim's shortfall makes requests behind the admission queue's head release their whole prefixes (c96a07a), plus the three fixes the first lab runs forced (below). Control is 9d4cf27 with the release switched off (the headroom fix alone; a throwaway commit, not on the branch). Medians of 3 (control: 1 and 2 runs). Card: the device plugin picks it; root port 00:01.0 (03:00.0, GPU 0) unless marked ¹ (00:01.1, GPU 1); ² not recorded (an ssh failure).

| Sessions | Arm     | ok (per run)  | later-turn TTFT p50 / p99, median (ms) | p99 per run (s)     | deepest state per run | max kv_utilization | blocks released | cached_tokens_ratio |
| -------- | ------- | ------------- | -------------------------------------- | ------------------- | --------------------- | ------------------ | --------------- | ------------------- |
| 24       | before  | 192, 192, 192 | 149 / 28,335                           | 17.5, 38.3, 28.3¹   | RED, RED, RED         | 0.904–0.921        | 0               | 0.820–0.826         |
| 24       | after   | 192, 192, 192 | 140 / 9,593                            | 10.0, 9.6, 9.4      | ORANGE ×3             | 0.844–0.894        | 212–381         | 0.795–0.823         |
| 24       | control | 192           | 143 / 10,317                           | 10.3                | ORANGE                | 0.878              | 0               | 0.841               |
| 32       | before  | 245, 249, 256 | 4,676 / 58,930                         | 58.9¹, 59.9¹, 42.2¹ | RED, RED, RED         | 0.919–0.938        | 0               | 0.751–0.780         |
| 32       | after   | 256, 256, 256 | 540 / 19,320                           | 18.3, 27.4², 19.3   | ORANGE ×3             | 0.885–0.896        | 607–701         | 0.758–0.761         |
| 32       | control | 256, 256      | 1,430–2,017 / 27,238–52,283            | 27.2, 52.3¹         | ORANGE, RED           | 0.914–0.959        | 0               | 0.799–0.805         |

Serve runs: before 1001141442-049a53ef, 1001155401-3ca25ff3, 1001141632-377fd80b (24), 1001144706-2c618051, 1001155900-001efbd1, 1001145036-04655602 (32); after 1001160525-10d3639e, 1001160934-0dbc4787, 1001161119-0b337166 (24), 1001161254-05f78725, 1001161502-31f8c2c5, 1001164854-089ab7df (32); control 1001165347-3ce64578 (24), 1001170243-27dd7125, 1001172320-0bbfe0e9 (32). The before failures are `queue_timeout` (11 and 7). Every before row at 32 sessions ran on GPU 1 (the device plugin's choice; the workload is bound by L0 capacity and admission queueing, not by the PCIe link, but the rows are not GPU-0 numbers).

Card placement (lead, 2026-10-01): the three 32-session "before" runs landed on GPU 1 (`lab-serve.sh` cannot pin a card), so under the GPU-0-only perf rule the 32-session before/after comparison is indicative, not a recorded perf result; the builder reported no other run off GPU 0, but card placement was not recorded per run. A GPU-0 rerun of the 32-session "before" arm is needed before quoting those numbers as a gain.

What the first lab runs found (each fixed with a test, commits on the branch):

- c96a07a alone, 24 sessions (1001145510-34f70592): 163/192 ok, SURVIVAL, 18 `overloaded`. A released request keeps a whole-request reservation (it may recompute), and its re-attached blocks were referenced, so the ledger's `held` counted them while the reservation still did: `kv_utilization` 0.58 → 0.99. 8c5de54 commits the attached blocks against the reservation.
- 8c5de54, 24 sessions (1001151729-3f44490c): 160/192, SURVIVAL again. A re-attach from L1 allocates its promotion targets at attach time; they were held and reserved until they landed. 82a47b2 commits `pending_blocks` on `Promoting`.
- With both, 3 × 24 and 3 × 32 sessions: no double count, but a RED refill of 2–3 released requests (whole-prompt reservations, ~26 blocks each) took `kv_utilization` 0.93 → 0.98 past SURVIVAL's 0.97 in 1 of 3 runs at 32 sessions (227/256), and the 24-session p99 was 28–33 s. The S-9 headroom rule summed only committed and reserved bytes, while `kv_utilization` counts `held` since the held-prefix ledger fix. 9d4cf27 makes the headroom count `held` (`PoolUsage::in_use`).

Reading: the headroom fix alone (control) keeps 24 sessions at ORANGE and halves its p99 (28 → 10 s); queued-prefix release adds little there. At 32 sessions the release matters: p50 1.4–2.0 s → 0.4–0.8 s and p99 27–52 s → 18–27 s against the control, every run ORANGE, no failure (before: 7–11 `queue_timeout` in 2 of 3, p99 42–60 s). The released blocks (600–700 per run) mostly come back from L1 (cached ratio 0.76, against 0.80 for the control and 0.75–0.78 before).

10-minute overload soak on 9d4cf27 (`SOAK_BENCH` an ssh wrapper running the bench on novanas; serve 1001172949-2621fd90, `target/soak/novanas-20261001T172949Z`): PASS, all 8 checks true; ITL p99 210.7 ms (calibration 174.0 ms), GREEN 30 s into the cool-down, 4259 × 200, 71 × 503 `overloaded`, 2282 `queue_timeout` (the p6b-survival soak: 4385 / 72 / 2421).

### TurboQuant lower-tier gates (Task 9, `tq4` / `tq2` MSE-only K; branch `p6b-t9`)

Trees f7bcebb (Llama) and e88c333 / 4a8469c (OLMoE; test and eval-data commits only, same server). Every serve Job of ours ran alone under the bench lock. `lab-bench` ran on GPU 0, the k3s serve Jobs on whichever card was free.

`scripts/lab-bench.sh --model <m> --golden16 -- --set kv.cpu.format=<f> --set kv.cpu.max_bytes=4GiB` (200 requests, client novanas; labbook 0949307a, d7c0d5a0, defe0be9, 87aba952):

| Model | L1 format | tok/s | ITL p50 (ms) | TTFT p50 (ms) | golden c1 / c16 (batched) |
| ----- | --------- | ----- | ------------ | ------------- | ------------------------- |
| Llama | `tq4`     | 850.3 | 15.4         | 207           | PASS / PASS (16/16)       |
| Llama | `tq2`     | 844.4 | 15.6         | 208           | PASS / PASS (16/16)       |
| OLMoE | `tq4`     | 607.4 | 24.6         | 119           | PASS / PASS (15/16)       |
| OLMoE | `tq2`     | 606.7 | 24.6         | 119           | PASS / PASS (15/16)       |

The golden and bench prompts share no full block, so no lossy block is read here. These runs show that the tier configuration does not disturb serving. They do not show the codec's quality.

Shared-prefix GSM8K-200 (`p6b-eval-prefix.md` recipe: c16, 32 fillers, `kv.gpu.max_bytes=4GiB`, `kv.cpu.max_bytes=4GiB`; Llama `phase6-novanas-llama.yaml`, OLMoE `phase2c-novanas-olmoe.yaml`):

- Candidates ran with `--min-lossy-cached-ratio 0.5` and baselines with `--min-cached-ratio 0.5`. Every guard passed.
- Every server counted only its eval's prompt tokens: 726,723 for Llama, 762,875 for OLMoE (OLMoE's tokenizer; its prefix is 25 blocks).
- Lossy cached ratio: 0.927 (Llama), 0.960 (OLMoE).
- The Llama baselines are `turbine-bf16-sp-r{1,2,3}.json` from f9a4c91. Since that commit only TurboQuant code changed (codec, transcode, mixed attention, tables), and the recipe and eval runner are unchanged.
- The OLMoE runs are bit-identical across repeats (each arm's three reports are equal).
- Pairs are judged with `scripts/eval/paired_compare.py --max-drop 0.01` and committed as `turbine-l1-<f>-sp-r<i>-paired.json`.

| Model | Arm  | r1    | r2    | r3    | Median | Drop   | McNemar exact p (9 pairs)      | Gate |
| ----- | ---- | ----- | ----- | ----- | ------ | ------ | ------------------------------ | ---- |
| Llama | BF16 | 0.780 | 0.785 | 0.770 | 0.780  |        |                                |      |
| Llama | tq4  | 0.775 | 0.760 | 0.790 | 0.775  | 0.005  | 0.23–1.0                       | PASS |
| Llama | tq2  | 0.730 | 0.720 | 0.705 | 0.720  | 0.060  | 0.011–0.24 (3 of 9 below 0.05) | FAIL |
| OLMoE | BF16 | 0.635 | 0.635 | 0.635 | 0.635  |        |                                |      |
| OLMoE | tq4  | 0.665 | 0.665 | 0.665 | 0.665  | −0.030 | 0.24 (lost 6, gained 12)       | PASS |
| OLMoE | tq2  | 0.610 | 0.610 | 0.610 | 0.610  | 0.025  | 0.53 (lost 23, gained 18)      | FAIL |

Multi-turn A/B against `l0`, 16 sessions, c16, `--think-time 1..4 --session-hints`, a fresh server per run, medians of 3:

- Llama: `phase2c-novanas-llama.yaml --set kv.cpu.format=<f> --set kv.cpu.max_bytes=4GiB`, `--shared-prefix-words 2000`.
- OLMoE: its 4,096-token context cannot hold the spec workload (2,000 words is already ~4,100 tokens at turn 1; the first attempt answered 112 of 128 requests with 400 `context_length_exceeded` and was discarded). It ran `--shared-prefix-words 600 --prompt-words 128 --max-tokens 64` with `--set kv.gpu.max_bytes=4GiB` (256 L0 blocks, below the ~430 the live histories need), so L1 serves 190–260 lookups per run.
- Capacity is from `/turbine/v1/kv` L1 `formats`. Blocks per GiB of filled L1 include the lossless-tail `l0` blocks. Pure-codec block bytes are 4,128,768 for `tq4` (Llama), 3.56× fewer than BF16's 14,680,064.

| Model | L1 format | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | L1 blocks per GiB (vs `l0`) | lossless-tail share | recompute / retrieve plans | mean L1→L0 promotion (ms) |
| ----- | --------- | ------------------- | ------------------------------ | --------------------------- | ------------------- | -------------------------- | ------------------------- |
| Llama | `l0`      | 0.9052              | 78.5 / 356                     | 73.1                        | none                | 0 / 29                     | 42 / 41 / 61              |
| Llama | `tq4`     | 0.8854              | 75.5 / 1754                    | 216.3 (2.96×)               | 0.079               | 11 / 9                     | 245 / 289 / 85            |
| Llama | `tq2`     | 0.9048              | 73.8 / 316                     | 380.6 (5.21×)               | 0.043               | 0 / 15                     | 97 / 108 / 96             |
| OLMoE | `l0`      | 0.8607              | 53.2 / 243                     | 64.0                        | none                | 0 / 45                     | 29–45                     |
| OLMoE | `tq4`     | 0.8581              | 48.8 / 180                     | 185.8 (2.90×)               | 0.088               | 0 / 25                     | 63–70                     |
| OLMoE | `tq2`     | 0.8583              | 49.2 / 195                     | 302.9 (4.73×)               | 0.065               | 0 / 27                     | 57–74                     |

The spec criterion "`cached_tokens_ratio` ≥ the `l0` run" fails on medians for every TurboQuant cell:

- Llama `tq4`: −0.020. Runs 1 and 2 recomputed 17 and 11 plans (`recompute_cheaper`); run 3 (0.905) did not. Their promotions averaged 245–289 ms against 41–61 ms at `l0`, which also shows in the later-turn p99 (1.8–2.1 s). The planner priced the slow `tq4` retrieves above recomputing.
- OLMoE `tq4` and `tq2`: −0.003. There were no recomputes, but all six runs are at or below the lowest `l0` run.
- Llama `tq2`: −0.0004, within the `l0` spread.

Later-turn TTFT p50 is equal or better with TurboQuant (fewer bytes over the link).

`kv_gpu` (job 1001170002-222ba975, e88c333): 12 passed, 0 failed. `lossy_tier_reuse_tq4` is now held to (0.25, 0.75, 0.9): the first 8 tokens within 0.25 of cold, 90 % of 64 positions within 0.75. These bounds are golden's Llama batched bounds, which the tq4 golden16 run passed, and the GSM8K gate showed that reuse at that level costs no accuracy. Measured: first-8 worst 0.242, 0.98 of positions within 0.75, worst position 0.979.

Result: `tq2` fails the GSM8K gate on both models. `tq4` passes golden and GSM8K on both, but misses the multi-turn ratio criterion (Llama by 0.020, from slow promotions; OLMoE by 0.003). Both stay `experimental` in `TIER_FORMAT_REFUSALS`. Options are in `.procoder/handoff/p6b-t9.md`. Labbook set `phase-6b-kv-compression`: the 4 lab-bench runs, 18 multi-turn runs and the kv_gpu run.

### TurboQuant promotion path (decision "6b Task 9: TurboQuant lower-tier gate results" 1 A; branch `p6b-tqspeed`)

Workload and arms as in Task 9: `turbine-bench --profile multi-turn --sessions 16 --turns 8 --concurrency 16 --think-time 1..4 --session-hints` (Llama `--shared-prefix-words 2000`; OLMoE the spec's variant `--shared-prefix-words 600 --prompt-words 128 --max-tokens 64` with `--set kv.gpu.max_bytes=4GiB`), `phase2c-novanas-<m>.yaml --set kv.cpu.format=<f> --set kv.cpu.max_bytes=4GiB`, `logging.level: info,turbine_kv=debug,turbine_server::kv_orchestrator=debug`. Every server ran natively on GPU 0 (`ROCR_VISIBLE_DEVICES=0`, cores 0-11) under `bench.lock` and `port18000.lock`, a fresh server per run, the client on novanas.

Profile (ae67d24): the DEBUG event `kv_copy_stages` lists each finished copy's stages as `name:ran..seen` ms, and `kv_copy_timed` gained `queued_s`. A lossy L1 → L0 promotion ran three stages: the I/O pool read the encoded L1 slot into a pinned staging buffer (`io_read_coded`), the copy stream moved it into a device staging slot (`h2d_slot`), and the decode kernel ran with a compute fence (`decode`). Each stage is seen done at the next engine poll. One run per format, Llama:

| L1 format | promotions | mean / p50 / p90 (ms) | `io_read_coded` seen mean / p90 | `h2d_slot` mean | `decode` mean | host-codec demotions (I/O pool) | demotion I/O mean / p90 | recompute / retrieve | ratio  |
| --------- | ---------- | --------------------- | ------------------------------- | --------------- | ------------- | ------------------------------- | ----------------------- | -------------------- | ------ |
| `l0`      | 267        | 40 / 42 / 58          | — (one copy-stream stage)       | —               | —             | —                               | —                       | 0 / 30               | 0.9055 |
| `tq2`     | 49         | 123 / 76 / 181        | 2.1 / 3.8                       | 42              | 32            | 5 of 202                        | 12 / 3                  | 0 / 3                | 0.9050 |
| `tq4`     | 131        | 272 / 147 / 926       | 162 / 858                       | 46              | 38            | 59 of 398                       | 254 / 852               | 11 / 9               | 0.8795 |

Cause: the tq4 promotions queued on the I/O pool behind demotion writes. When all 32 device staging slots (`DEMOTION_INFLIGHT`) were taken, a demotion fell back to the host codec. Each tq4 host encode of a 14.7 MB block holds an I/O thread for hundreds of ms. A promotion took its slot first and then held it through that wait, so fewer slots were free and more demotions fell back: a feedback loop. `tq2` had 5 fallbacks and did not enter it. The decode kernel was not the cost: its stage is one poll, like the H2D. The kernel itself takes about 0.25 ms per block (Task 8: 8.1 ms per 32 blocks).

Fix (b6c8e11): a lossy L1 → L0 promotion that the device transcode serves copies the encoded L1 slot (pinned) straight into the device staging slot on the copy stream (`L1PinnedTier::locate_len`), then decodes. It never touches the I/O pool. Otherwise (several shards, no free slot, a slot of another size) it takes the old path. Test: `kv_orchestrator::tests::device_transcode_matches_the_host_codec_through_l1_and_l2` (the promotion starts as copy-stream copies; mutation "path disabled": FAIL). No kernel changed.

After, one profile run (Llama tq4): 281 promotions, mean / p50 / p90 66.6 / 56.7 / 126 ms, all `h2d_slot/decode`. Recompute / retrieve 0 / 14, ratio 0.9052, later-turn p99 325 ms. 46 demotions still took the host codec (demotion I/O p90 529 ms); promotions no longer wait for them.

Multi-turn A/B after the fix (b6c8e11), medians of 3, arms interleaved:

| Model | L1 format | cached_tokens_ratio (runs) | median     | later-turn TTFT p50 / p99 (ms, median) | mean L1→L0 promotion (ms) | recompute / retrieve plans | L1 blocks per GiB (tail share) |
| ----- | --------- | -------------------------- | ---------- | -------------------------------------- | ------------------------- | -------------------------- | ------------------------------ |
| Llama | `l0`      | 0.9016 / 0.9033 / 0.8904   | 0.9016     | 80.6 / 411                             | 44 / 47 / 43              | 0 / 28–31                  | 73.1                           |
| Llama | `tq4`     | 0.9047 / 0.9061 / 0.9020   | **0.9047** | 78.2 / 255                             | 61 / 50 / 63              | 0 / 11–15                  | 224.5–227.8 (0.055–0.062)      |
| OLMoE | `l0`      | 0.8626 / 0.8633 / 0.8617   | 0.8626     | 51.7 / 280                             | 40 / 27 / 38              | 0 / 45–48                  |                                |
| OLMoE | `tq4`     | 0.8605 / 0.8584 / 0.8591   | 0.8591     | 48.2 / 150                             | 45 / 43 / 44              | 0 / 24–31                  |                                |

- Llama: the criterion holds (0.9047 ≥ 0.9016). tq4 promotions dropped from 245–289 ms to 50–63 ms, no plan recomputes, and later-turn p99 fell from 1.8–2.1 s to 255 ms (the `l0` arm's is 411 ms).
- OLMoE: it still misses by 0.0035, and every tq4 run is below every `l0` run. This is not promotion speed: promotions take 43–45 vs 27–40 ms and there are no recompute plans. With prompt tokens in the same range (274.9–275.8k vs 274.0–276.2k), tq4 caches about 1k fewer tokens per run (236.4–236.8k vs 236.5–238.0k). It also has more lookup misses (247 vs 231, r3 vs r2) and fewer retrieve plans. The shortfall equals Task 9's (−0.003), so it predates this fix. Not investigated further here; options are in `.procoder/handoff/p6b-tqspeed.md`.

Transcode kernels, remeasured with MSE-only K (no kernel changed; lab job `turbine-lab-test-1001211526-3189c413`, `kv_transcode_matches_cpu` and `paged_mixed_matches_cpu` PASS), 32 Llama blocks: `tq4` encode 22,387 µs and decode 1,299 µs (Task 8 with the QJL K: 20,473 / 8,116), `tq2` 12,586 / 1,164, FP8 2,449 / 1,576.

`kv_gpu lossy_tier_reuse_tq4` / `lossy_tier_reuse` through the new path (job `turbine-lab-test-1001211921-28c11f9d`): PASS, the same numbers as Task 9 (tq4 first-8 worst 0.242, 0.98 within 0.75; fp8 0.067), so the copy-stream promotion decodes the same bytes.

Result: `tq4` stays `experimental`. It passes on Llama and misses on OLMoE.

### OLMoE tq4 multi-turn block loss (decision "6b: lower-tier tq4 after the promotion fix", A; branch `p6b-olmoe-tq4`)

Two mechanisms, both reproduced in `kv_sim` with the OLMoE shape scaled to 16-token blocks (16 sessions behind one 96-token prefix, seeded short turns with think time, `submit_session` hints, a 64–128-block L0, an L1 that holds every history):

1. Lossy-chain misses (fixed, 018016c). A block computed over a lossy prefix is keyed by the chain of the request that computed it, which starts at that request's first lossy block. Histories leave L0 leaf-first, so a session's first lossy block moves earlier from turn to turn, and the later lookup's chain never named the earlier turn's computed blocks: they were recomputed. The lookup now also takes the lossy-lineage children of the previous block's matching entries. kv_sim at YELLOW, L0 64, seeds 1 / 2 / 4: `tq4` cached 1,184 / 640 / 1,328 tokens fewer than `l0` before, equal after. Tests `hierarchy::tests::lossy_lineage_blocks_survive_an_earlier_switch`, `kv_sim lossy_multi_turn_matches_l0_reuse` (mutation "child walk off": both FAIL).
2. Lossless-tail scoring (not changed; needs a decision). A lossless-tail block is scored at the L0-format bytes it would be demoted at (decision B, encoded-size scores), about 3.5× its chain's `tq4` retrieval. Leaf-first, the tail is its chain's leaf: once it goes, its cheap parents follow, so whole histories drain where `l0` trims across sessions. kv_sim at GREEN (no lossy block reused at all), seeds 1 / 2 / 4 / 6 × L0 64 / 96 / 128: `tq4` −944 … +208 tokens against `l0`; with `kv.lossless_tail_blocks: 0`, or with the tail scored at the tier's rung bytes, `tq4` equals `l0` in all twelve, and every other kv_sim test still passes.

Lab A/B after (1) (tree 018016c, harness `scratch/p6b-olmoe-tq4/ab.sh` on novanas: a fresh native server per run on GPU 0, cores 0-11, under `port18000` and `bench.lock`, client on novanas; workloads as in "TurboQuant promotion path"; arms interleaved):

| Model | L1 format | cached_tokens_ratio (runs)  | median | prompt − cached tokens (runs) | retrieve plans | later-turn TTFT p50 / p99 (ms) |
| ----- | --------- | --------------------------- | ------ | ----------------------------- | -------------- | ------------------------------ |
| OLMoE | `l0`      | 0.8613 / 0.8619 / 0.8622    | 0.8619 | 38,170 / 38,035 / 38,020      | 48 / 50 / 47   | 51–54 / 123–334                |
| OLMoE | `tq4`     | 0.8611 / 0.8609 / 0.8611    | 0.8611 | 38,292 / 38,318 / 38,327      | 24 / 24 / 28   | 47–51 / 130–136                |
| Llama | `l0`      | 0.9063 / 0.9050 / 0.9052    | 0.9052 | 62,520 / 63,494 / 63,566      | 31 / 28 / 31   | 78–80 / 264–386                |
| Llama | `tq4`     | 0.9063 / 0.9067 / (invalid) | —      | 62,857 / 62,488 / —           | 17 / 11 / —    | 79 / 198–220                   |

- OLMoE: the shortfall fell from 0.0035 to 0.0008 (about 1–2 blocks of 128 per run instead of 8), but every `tq4` run is still below every `l0` run: the gate (tq4 median ≥ `l0`) still fails. The residue matches mechanism 2: the `tq4` servers stayed GREEN throughout (no reclaim event), where capacity demotion and allocation reclaim take victims in the tail-distorted order.
- Llama `tq4` r3 is invalid: 3 s after `/ready` the controller went RED on `step_time_drift` (3.37 > 3.0) and stayed there (admission queued, `free_cached` reclaims, TTFT p50 32 s, ratio 0.558, no lossy reuse). Not caused by this change (no KV event preceded it); rerun that arm.
- Not flipped: `tq4` stays `experimental`.

### Lossless last block scored like its history (decision "6b: OLMoE tq4 — lossless last block in eviction order", 1 A; branch `p6b-lastblock`)

Change 028f465: `score_one` prices a lossless last block's retrieval at the bytes its history would be stored at in the tier below; its demotion, memory term and planner pricing keep its L0-format size. kv_sim `lossy_multi_turn_green_matches_l0_reuse` (GREEN, L0 64 / 96 / 128 / 192 × seeds 1 / 2 / 4): `tq4` ≥ `l0` in all twelve after, L0 64 seed 1 16,496 vs 16,976 before. No fixture moved.

Lab A/B, tree 028f465, harness `scratch/p6b-lastblock/ab.sh` on novanas (the `p6b-olmoe-tq4` harness with an absolute output path and the pressure transitions counted per run): same workloads, servers and locks as "OLMoE tq4 multi-turn block loss", arms interleaved, results in `scratch/p6b-lastblock/ab1/`:

| Model | L1 format | cached_tokens_ratio (runs) | median | prompt − cached tokens (runs) | retrieve plans | later-turn TTFT p50 / p99 (ms) | pressure transitions                 |
| ----- | --------- | -------------------------- | ------ | ----------------------------- | -------------- | ------------------------------ | ------------------------------------ |
| OLMoE | `l0`      | 0.8627 / 0.8628 / 0.8616   | 0.8627 | 37,648 / 37,778 / 38,029      | 45 / 45 / 48   | 51–56 / 117–289                | YELLOW on `step_time_drift`, 24–25 s |
| OLMoE | `tq4`     | 0.8619 / 0.8609 / 0.8604   | 0.8609 | 38,062 / 38,364 / 38,306      | 43 / 40 / 42   | 48–52 / 131–148                | none                                 |
| Llama | `l0`      | 0.9052 / 0.9056 / 0.9022   | 0.9052 | 63,083 / 63,143 / 65,421      | 30 / 32 / 29   | 78–80 / 172–401                | YELLOW (r2 on drift at 33 s)         |
| Llama | `tq4`     | 0.9059 / 0.9074 / 0.9061   | 0.9061 | 62,594 / 62,094 / 62,949      | 26 / 30 / 31   | 77–83 / 205–402                | YELLOW on `kv_utilization` (r1, r3)  |

- Llama passes: `tq4` 0.9061 ≥ `l0` 0.9052, every `tq4` run at or above every `l0` run.
- OLMoE fails: `tq4` 0.8609 vs `l0` 0.8627, every `tq4` run below every `l0` run. `tq4`'s uncached tokens are unchanged from the previous A/B (38.1–38.4k against 38.3k); the gap moved only because the `l0` runs came out higher (37.6–38.0k against 38.0–38.2k). In the lab the change does not measurably move OLMoE `tq4`.
- No run left GREEN in its first seconds, so none was discarded. Every OLMoE `l0` run, here and in the previous A/B, goes YELLOW on `step_time_drift` (1.55–1.63 against 1.5) 24–25 s after `/ready`, late in a ~35 s run, and stays there: 15 `reclaim` events and throttle changes follow. No OLMoE `tq4` run leaves GREEN or reclaims. The arms therefore run under different pressure states: `l0` gets YELLOW's proactive demotion, `tq4` only GREEN's on-demand reclaim.
- Not flipped: `tq4` stays `experimental`.

### TurboQuant in L0 (Task 13, `kv.dtype=tq4` / `tq2`; branch `p6b-t13`)

Tree 427f6f2 (`p6b-stack` 3349b4a plus `lab-bench --batched-bounds`). `scripts/lab-bench.sh --model <m> --golden16 --batched-bounds --c1 -- --set kv.dtype=<f>`: GPU 0, 200 requests at c16, client on novanas. The c1 leg is lab-bench's (10 requests, 128 tokens, `--ignore-eos`), not the plan's 32 × 256. Golden is judged by the batched bounds at c1 and c16; the strict c1 verdict is noted too. Labbook set `phase-6b-kv-compression`: cad82a75, 72719594, f5e74f28, 1f2126fd, 4ab782a0, 3aadbc6b.

| Model | KV     | tok/s (vs BF16) | ITL p50 c16 (ms) | ITL p50 c1 (ms) | TTFT p50 (ms) | golden c1 / c16 (batched) | L0 blocks (vs BF16) |
| ----- | ------ | --------------- | ---------------- | --------------- | ------------- | ------------------------- | ------------------- |
| Llama | `bf16` | 849.5           | 15.5             | 12.32           | 207           | PASS / PASS               | 585                 |
| Llama | `tq4`  | 806.8 (0.950×)  | 16.0 (1.03×)     | 12.69 (1.03×)   | 240 (1.16×)   | FAIL 0/16 / FAIL 0/16     | 2080 (3.56×)        |
| Llama | `tq2`  | 828.5 (0.975×)  | 15.6 (1.01×)     | 12.61 (1.02×)   | 231 (1.12×)   | FAIL 1/16 / FAIL 1/16     | 3744 (6.40×)        |
| OLMoE | `bf16` | 610.1           | 24.5             | 7.48            | 119           | PASS / PASS               | 512                 |
| OLMoE | `tq4`  | 590.1 (0.967×)  | 23.0 (0.94×)     | 7.44 (0.99×)    | 184 (1.55×)   | FAIL 8/16 / FAIL 8/16     | 1820 (3.55×)        |
| OLMoE | `tq2`  | 646.2 (1.059×)  | 22.6 (0.92×)     | 7.52 (1.01×)    | 138 (1.16×)   | FAIL 1/16 / FAIL 1/16     | 3276 (6.40×)        |

- L0 blocks come from `/turbine/v1/kv` in the same 8 GiB L0. Block bytes: Llama 14,680,064 / 4,128,768 / 2,293,760; OLMoE 16,777,216 / 4,718,592 / 2,621,440. The measured ratios match the codec's (3.56× / 6.40×), so the ≥ 3.5× / ≥ 6× targets hold.
- Golden: the token rule mostly holds (Llama tq4 has 10 of 16 prompts with an identical 32-token prefix). What fails is the logprob bound: Llama tq4 likely |Δ| 0.26–1.65 against 0.25, tail up to 2.36 against 0.75. Every block, including the newest tokens, is lossy, so the error is larger than the lossy-prefix reuse of Task 9 (first-8 worst 0.242).

Shared-prefix GSM8K-200 (`p6b-eval-prefix.md` recipe: c16, 32 fillers, `kv.gpu.max_bytes=4GiB`, `kv.cpu.max_bytes=4GiB`, `kv.cpu.format=l0`, `--min-cached-ratio 0.5`, plus `kv.dtype=<f>` for the candidate). Each run had a fresh `lab-serve` Job, and each server counted only its eval (`turbine_kv_prompt_tokens_total` 726,723 Llama, 762,875 OLMoE). Llama got three fresh BF16 baselines on this tree (`turbine-bf16-sp-r{4,5,6}.json`), because queued-prefix demotion landed after r1–r3. OLMoE's fresh BF16 run was bit-identical to `turbine-bf16-sp-r1.json`, so its r1–r3 stand. Pairs: `scripts/eval/paired_compare.py --max-drop 0.01`, committed as `turbine-l0-<f>-sp-r<i>-paired.json` (Llama r_i against BF16 r_(i+3)). `lossy_cached_tokens` is 0 throughout: L0 TurboQuant reuse is not counted as lossy reuse.

| Model | Arm  | r1    | r2    | r3    | Median | Drop   | McNemar exact p (9 pairs) | Gate |
| ----- | ---- | ----- | ----- | ----- | ------ | ------ | ------------------------- | ---- |
| Llama | BF16 | 0.770 | 0.780 | 0.780 | 0.780  |        |                           |      |
| Llama | tq4  | 0.785 | 0.765 | 0.790 | 0.785  | −0.005 | 0.59–1.0                  | PASS |
| Llama | tq2  | 0.195 | 0.195 | 0.150 | 0.195  | 0.585  | ≤ 6e-33                   | FAIL |
| OLMoE | BF16 | 0.635 | 0.635 | 0.635 | 0.635  |        |                           |      |
| OLMoE | tq4  | 0.615 | 0.615 | 0.615 | 0.615  | 0.020  | 0.61 (lost 19, gained 15) | FAIL |
| OLMoE | tq2  | 0.170 | 0.170 | 0.170 | 0.170  | 0.465  | 5e-25                     | FAIL |

Staged-prefill profile (decision "6b Task 12: follow-ups", 1 A; measure only). Method: `rocprofv3 --kernel-trace` around a native server on GPU 0 under `bench.lock`, `phase2c` config, 20 requests at c1, `--max-tokens 1`, kernels summed inside the bench window. Llama prompts were 2000 words (≈ 2,950 tokens, two prefill chunks); OLMoE prompts were 1500 words. Per request:

| Model, KV  | TTFT p50 (ms) | + vs BF16 | encode `mixed_append_tq` | post-CK `mixed_attn` (+ prep, combine) | staging `mixed_stage` | CK fmha |
| ---------- | ------------- | --------- | ------------------------ | -------------------------------------- | --------------------- | ------- |
| Llama bf16 | 198.7         |           | (BF16 append 0.9)        |                                        |                       | 20.9    |
| Llama tq4  | 232.4         | +33.7     | 19.6                     | 12.3                                   | 3.0                   | 20.4    |
| Llama tq2  | 225.4         | +26.7     | 11.9                     | 12.2                                   | 3.0                   | 20.5    |
| OLMoE bf16 | 95.3          |           | (BF16 append 0.7)        |                                        |                       | 5.5     |
| OLMoE tq4  | 116.7         | +21.4     | 14.7                     | 4.5                                    | 2.2                   | 5.4     |

- Encode dominates: 58 % of the Llama tq4 increase and 69 % of OLMoE's. That is 0.35 ms per 2048-row layer chunk on Llama, about 30 GB/s effective and 21× the BF16 append.
- Second is the single-query `turbine_hip_mixed` pass that `run_mixed_staged` launches after CK on every layer and chunk: 7 + 4 full-grid launches per layer per Llama request. It runs even when the batch has no `q_len` 1 row, and this workload has none.
- Staging decode is small, and CK is unchanged.

### TurboQuant prefill follow-ups (decision "6b Task 13", 3 C; branch `p6b-tqfollow`)

Two changes, measured one after the other: the single-query `turbine_hip_mixed` pass after the staged prefill now runs only when the descriptor allows a single-query row (497ddd1), then the TurboQuant encode shared by the transcode and the mixed append got faster (ad28f9a, 0a26bd4, 8e39d7d; bit-exact, `kv_transcode_matches_cpu` and `paged_mixed_matches_cpu` green, three mutations red). `f2faa53` (a double-F32 norm) was measured slower and replaced by `0a26bd4`.

Kernel timings (`hip_ops`, k3s card, mean of 10 calls; the 32-block Llama-shape transcode batch, 448 MiB of BF16 pages):

| Step                                                                       | Commit  | tq4 encode (µs) | tq2 encode (µs) | Llama staged prefill tq4, 1 × 2,048 / 16 × 512 (µs) | OLMoE, same (µs) |
| -------------------------------------------------------------------------- | ------- | --------------- | --------------- | --------------------------------------------------- | ---------------- |
| earlier entries (encode: `p6b-tqspeed` remeasure; prefill: Task 12, GPU 0) | ee7b950 | 22,387          | 12,586          | 1,517 / 7,088                                       | 1,952 / 11,391   |
| skip the single-query pass                                                 | 497ddd1 | 22,278          | 12,662          | 880 / 3,653                                         | 1,201 / 6,414    |
| midpoint binary search, byte packing                                       | ad28f9a | 12,555          | 11,509          |                                                     |                  |
| F64 FMA norm, 16-byte loads, words                                         | 8e39d7d | 7,162           | 5,985           | 530 / 2,496                                         | 577 / 4,091      |

BF16 for comparison: the FP8 transcode encodes the same batch in 2,430 µs; the BF16 CK prefill is 350 / 1,757 µs (Llama) and 271 / 2,538 µs (OLMoE). A one-run variant sweep on top of ad28f9a (`p6b-tqfollow-exp`, not merged; tq4 µs) split the encode: scalar loads with the F64 norm 12,477, scalar loads without any norm 9,883, 16-byte loads with the F64 norm 8,168, 16-byte loads without a norm 5,042, 16-byte loads with the F64 FMA norm 7,216; the double-F32 norm 15,716 (scalar) / 11,111 (16-byte). The F64 norm is still about 2 ms of the 7.2.

Served prefill TTFT, c1, 20 requests, `--max-tokens 1` (Llama 2,000-word prompts ≈ 2,950 tokens in two chunks, OLMoE 1,500 words; native server on GPU 0 under `bench.lock`, `phase2c` config, `--set kv.dtype=<f>`), the Task 13 profile's workload:

| Model | KV     | before (b4e9f46) | pass skipped (497ddd1) | faster encode (8e39d7d) | overhead vs BF16 (ms) |
| ----- | ------ | ---------------- | ---------------------- | ----------------------- | --------------------- |
| Llama | `bf16` |                  | 196.8                  |                         |                       |
| Llama | `tq4`  | 227.2            | 215.7                  | 204.3                   | 30.4 → 18.9 → 7.5     |
| OLMoE | `bf16` |                  |                        | 94.1                    |                       |
| OLMoE | `tq4`  | 114.3            |                        | 101.0                   | 20.2 → 6.9            |

`scripts/lab-bench.sh --quick --c1 --batched-bounds --model llama -- --set kv.dtype=tq4` (64 requests at c16, 512-word prompts; the c1 leg uses short prompts):

| Commit  | tok/s | ITL p50 (ms) | TTFT p50 c16 (ms) | TTFT p50 c1 (ms) | ITL p50 c1 (ms) | golden c1 (batched) |
| ------- | ----- | ------------ | ----------------- | ---------------- | --------------- | ------------------- |
| b4e9f46 | 834.9 | 15.9         | 281               | 42.9             | 12.57           | FAIL 0/16           |
| 497ddd1 | 829.8 | 16.0         | 285               | 43.5             | 12.57           | FAIL 0/16           |
| 8e39d7d | 866.6 | 15.3         | 248               | 43.5             | 12.51           | FAIL 0/16           |

- The pass skip shows only on prefill-only batches: at c16 almost every batch carries decode rows, so the pass still runs there (now over every query row of the batch, prefill rows included, as before); the c1 leg's prompts are short. The 2,000-word c1 run shows it: −11.5 ms, the 12.3 ms the profile attributed to the pass.
- The encode speed-up shows at c16: TTFT p50 −12 %, tok/s +3.8 %.
- Golden output (`golden1.txt`) is byte-identical across the three runs: the encode is bit-exact and the skipped pass never wrote a prefill row. Golden still fails (decision 1: tq4 stays `experimental`).
- Not done: a pass over only the single-query rows (a grid over sequences instead of query rows) would remove the pass's cost from mixed c16 batches too; the remaining F64 norm loop (≈ 2 ms of 7.2) and the LDS footprint (≈ 41 KB a workgroup) are the next encode items.

### Step-time drift at startup (decision "6b: OLMoE tq4 — lossless last block in eviction order; step-time drift at startup", 2 A; branch `p6b-drift`)

Observation: in the p6b-olmoe-tq4 A/B (novanas `scratch/p6b-olmoe-tq4/ab1/llama-tq4-r3.server.log`) the server went RED on `step_time_drift` 2.4 s after `/ready` (circuit DEGRADED on `latency_drift` at +2.40 s, RED at +2.50 s, value 3.37). It stayed RED for the whole 4.5 min run with no further pressure transition. RED's admission then let one request in at a time (`idle_floor`).

Harness: `scratch/p6b-drift/drift.sh` on novanas, a fresh native server per run on GPU 0 (cores 0-11), under `port18000` and `bench.lock`, client on novanas. The workload is the A/B's Llama multi-turn (16 sessions × 8 turns, c16, `--shared-prefix-words 2000`, think time 1..4 s, session hints, `phase2c-novanas-llama.yaml --set kv.cpu.max_bytes=4GiB`) or its OLMoE variant. `/turbine/v1/pressure` is polled every 100 ms, and the new `decode_step` debug trace (25fc430) logs each pure decode step's rows, context, seconds and judged ratio.

Root cause: two defects in `DecodeStepWindow`, both in the reliability crate.

1. A short window's p95 is its maximum. The p95 index is `round((n − 1) × 0.95)`, so up to 10 judged steps it is the largest one. In the first second after a bucket became judged, one slow step was the drift. The once-off step does happen: 1 of about 11,500 Llama decode steps across six starts took 138 ms, 7.31× its bucket (`burst1/llama-tq4-r1`, +24.3 s, rows 7). That step landed in a full window and did no harm.
2. The window never forgot. Above GREEN no shape learns a baseline (spec edge case), and shapes never seen while calm are not judged. Once RED let one request in at a time, at contexts the 16-row start never decoded at, no step was judged again. The window kept the spike, and its p95 held RED for as long as anything ran. `controller::tests::a_drift_spike_does_not_latch_once_its_shape_stops_running` replays that trace through the real window and controller.

Ruled out:

- Warm-up, graph capture and autotune baselines. The first step of a new decode shape runs eager and the second captures: 17.5 / 19.4 ms against 17.2 ms replays, so a bucket's first judged steps read 0.98–1.00.
- A decode-only baseline judging prefills. Prefill iterations are skipped.
- A unit mismatch.
- Host CPU contention. 24 spinning processes on cores 0-11 for 2 s, from 1.8 s after ready, left decode at 17.3 ms (`burst1`).

The startup spike did not reproduce in 6 starts on the old window (`r1/` ×5, `burst1/`), and the single 3.37 step of the original run cannot be attributed after the fact.

Fix:

- 5924e3f: judged steps older than `STEP_MAX_AGE` (10 s, the default `deescalate_dwell`) leave the window. `StepSample.at` and `p95(now)`; tests `judged_steps_age_out` and the controller test above.
- 66994e1: the p95 needs `MIN_JUDGED_STEPS` (20) judged steps inside that age, which makes it at least the second-largest; test `one_slow_step_in_a_short_window_is_not_drift`.
- Mutations: the age filter off makes both age tests FAIL; the minimum off makes the short-window test FAIL. `scripts/gate.sh --base aa42093`: `gate: ok passed=834`.
- The spec signal table and the contract §8.3 note are amended.

Re-measured on 66994e1 (`fix/`, Llama `tq4` ×2 and `l0` ×2): none left GREEN in the first 25 s. Later escalations were `kv_utilization` YELLOW at +30 s, plus one `step_time_drift` YELLOW at +34 s (`l0-r2`), which is the late pattern below. The cached_tokens_ratio values were 0.907 / 0.898 / 0.903 / 0.902.

Late-run drift is the signal being right (OLMoE `l0`, lead's report from `scratch/p6b-lastblock/ab1/olmoe-l0-r{1,2,3}`: YELLOW on drift 23–25 s after ready, held to the end of the run):

- From about 17 s on, OLMoE `l0` decode steps run 1.5–2.4× their bucket. With `kv_copy_stages` debug logging (`olmoe-old/`, `olmoe-fix/`), every judged step at ≥ 1.5× and 89–100 % of those at 1.2–1.5× overlapped a KV tier copy in flight, against 2–3 % of normal steps (6–9 % for `tq4`). Mostly they overlapped an L1 → L0 promotion: 16 MiB raw blocks at about 30 ms each in `l0` format (11 of 13 slow steps in `olmoe-fix/olmoe-l0-r2`); `tq4` moves 4.5 MiB blocks and has 1–3 slow steps per run.
- The window p95 reached 1.50–1.81 in all four `l0` runs here. One went YELLOW on drift; in two, `kv_utilization` had already made the state YELLOW.
- "Stays YELLOW" is the de-escalation dwell: in `olmoe-fix/olmoe-l0-r2` drift fell under the exit threshold (1.425) at +25.5 s, but the run ended at +33.6 s, before the 10 s dwell ran out. It is not a latch.
- Llama shows the same late steps (1.5–2.3×, +29–38 s, more under `l0` than `tq4`). Its `fix/` runs have no copy trace.
- The controller reads the KV hierarchy's own promote traffic as device slowdown, and YELLOW's reclaim then adds copies. The thresholds are unchanged; this needs a decision (handoff `p6b-drift`).

Earlier soaks: the 2026-09-30 10-minute soak timeline (worktree `agent-a4b513efedb95892f-lab`) is RED on `step_time_drift` for its first 13 s of overload and then on `queue_fill`. The 2026-09-27 soak shows 5 such seconds. Neither latched.

Soak: `scripts/overload-soak.sh novanas --duration 10m` on 66994e1 (serve run 1002071230-3f047df9, client on novanas through an ssh `SOAK_BENCH` wrapper) passed 8/8. Calibration ITL p99 was 174 ms and overload 209 ms. Status counts were 4,484 × 200 and 66 × 503 `overloaded`, with 2,835 `queue_timeout` and no client drop. GREEN came 30 s after the cool-down. The timeline has no second dominated by `step_time_drift` (the 2026-09-30 soak had 14).

### TurboQuant prefill follow-ups (decision "6b Task 13", 3 C; branch `p6b-tqfollow`)

Two changes, measured one after the other: the single-query `turbine_hip_mixed` pass after the staged prefill now runs only when the descriptor allows a single-query row (497ddd1), then the TurboQuant encode shared by the transcode and the mixed append got faster (ad28f9a, 0a26bd4, 8e39d7d; bit-exact, `kv_transcode_matches_cpu` and `paged_mixed_matches_cpu` green, three mutations red). `f2faa53` (a double-F32 norm) was measured slower and replaced by `0a26bd4`.

Kernel timings (`hip_ops`, k3s card, mean of 10 calls; the 32-block Llama-shape transcode batch, 448 MiB of BF16 pages):

| Step                                                                       | Commit  | tq4 encode (µs) | tq2 encode (µs) | Llama staged prefill tq4, 1 × 2,048 / 16 × 512 (µs) | OLMoE, same (µs) |
| -------------------------------------------------------------------------- | ------- | --------------- | --------------- | --------------------------------------------------- | ---------------- |
| earlier entries (encode: `p6b-tqspeed` remeasure; prefill: Task 12, GPU 0) | ee7b950 | 22,387          | 12,586          | 1,517 / 7,088                                       | 1,952 / 11,391   |
| skip the single-query pass                                                 | 497ddd1 | 22,278          | 12,662          | 880 / 3,653                                         | 1,201 / 6,414    |
| midpoint binary search, byte packing                                       | ad28f9a | 12,555          | 11,509          |                                                     |                  |
| F64 FMA norm, 16-byte loads, words                                         | 8e39d7d | 7,162           | 5,985           | 530 / 2,496                                         | 577 / 4,091      |

BF16 for comparison: the FP8 transcode encodes the same batch in 2,430 µs; the BF16 CK prefill is 350 / 1,757 µs (Llama) and 271 / 2,538 µs (OLMoE). A one-run variant sweep on top of ad28f9a (`p6b-tqfollow-exp`, not merged; tq4 µs) split the encode: scalar loads with the F64 norm 12,477, scalar loads without any norm 9,883, 16-byte loads with the F64 norm 8,168, 16-byte loads without a norm 5,042, 16-byte loads with the F64 FMA norm 7,216; the double-F32 norm 15,716 (scalar) / 11,111 (16-byte). The F64 norm is still about 2 ms of the 7.2.

Served prefill TTFT, c1, 20 requests, `--max-tokens 1` (Llama 2,000-word prompts ≈ 2,950 tokens in two chunks, OLMoE 1,500 words; native server on GPU 0 under `bench.lock`, `phase2c` config, `--set kv.dtype=<f>`), the Task 13 profile's workload:

| Model | KV     | before (b4e9f46) | pass skipped (497ddd1) | faster encode (8e39d7d) | overhead vs BF16 (ms) |
| ----- | ------ | ---------------- | ---------------------- | ----------------------- | --------------------- |
| Llama | `bf16` |                  | 196.8                  |                         |                       |
| Llama | `tq4`  | 227.2            | 215.7                  | 204.3                   | 30.4 → 18.9 → 7.5     |
| OLMoE | `bf16` |                  |                        | 94.1                    |                       |
| OLMoE | `tq4`  | 114.3            |                        | 101.0                   | 20.2 → 6.9            |

`scripts/lab-bench.sh --quick --c1 --batched-bounds --model llama -- --set kv.dtype=tq4` (64 requests at c16, 512-word prompts; the c1 leg uses short prompts):

| Commit  | tok/s | ITL p50 (ms) | TTFT p50 c16 (ms) | TTFT p50 c1 (ms) | ITL p50 c1 (ms) | golden c1 (batched) |
| ------- | ----- | ------------ | ----------------- | ---------------- | --------------- | ------------------- |
| b4e9f46 | 834.9 | 15.9         | 281               | 42.9             | 12.57           | FAIL 0/16           |
| 497ddd1 | 829.8 | 16.0         | 285               | 43.5             | 12.57           | FAIL 0/16           |
| 8e39d7d | 866.6 | 15.3         | 248               | 43.5             | 12.51           | FAIL 0/16           |

- The pass skip shows only on prefill-only batches: at c16 almost every batch carries decode rows, so the pass still runs there (now over every query row of the batch, prefill rows included, as before); the c1 leg's prompts are short. The 2,000-word c1 run shows it: −11.5 ms, the 12.3 ms the profile attributed to the pass.
- The encode speed-up shows at c16: TTFT p50 −12 %, tok/s +3.8 %.
- Golden output (`golden1.txt`) is byte-identical across the three runs: the encode is bit-exact and the skipped pass never wrote a prefill row. Golden still fails (decision 1: tq4 stays `experimental`).
- Not done: a pass over only the single-query rows (a grid over sequences instead of query rows) would remove the pass's cost from mixed c16 batches too; the remaining F64 norm loop (≈ 2 ms of 7.2) and the LDS footprint (≈ 41 KB a workgroup) are the next encode items.

### Compression ladder in L1/L2 on the server (Task 16; branch `p6b-t16`, 4074b5d)

Multi-turn A/B, Llama-3.2-3B, `scripts/lab/phase6-novanas-ladder.yaml` (L0 8 GiB, L1 2 GiB `l0`, L2 4 GiB `l0`, `max_format: tq4`) against the same file with `--set kv.ladder.enabled=false`. Tree 4074b5d (`p6b-stack` ce42218 and `p6b-drift` merged; no run left GREEN on `step_time_drift`). A fresh native server per run on GPU 0 (port 18010, under `bench.lock`, fixtures paused), arms interleaved. Workload: `turbine-bench --profile multi-turn --sessions 24 --turns 8 --concurrency 24 --shared-prefix-words 2000 --think-time 1..4 --session-hints`, client on novanas. Recomputed tokens are `turbine_kv_recompute_tokens_total`. Every run went GREEN → YELLOW → ORANGE and served 192 of 192 requests.

| Arm | Run | recomputed tokens | cached_tokens_ratio | lossy cached tokens | later-turn TTFT p50 / p99 (ms) | tok/s | L1 rewrites (no_room) | L2 rewrites (no_room) | L2 new demotions |
| --- | --- | ----------------- | ------------------- | ------------------- | ------------------------------ | ----- | --------------------- | --------------------- | ---------------- |
| on  | r1  | 163,166           | 0.8365              | 26,624              | 142.7 / 12,288                 | 307.8 | 369 (359)             | 229 (95)              | 73               |
| on  | r2  | 172,126           | 0.8280              | 0                   | 162.6 / 17,421                 | 327.2 | 562 (550)             | 349 (302)             | 12               |
| on  | r3  | 149,683           | 0.8507              | 0                   | 132.0 / 49,487                 | 202.2 | 2,290 (2,281)         | 387 (206)             | 20               |
| off | r1  | 202,388           | 0.7997              | 0                   | 170.1 / 14,640                 | 316.8 |                       |                       |                  |
| off | r2  | 206,206           | 0.7952              | 0                   | 162.8 / 10,557                 | 321.0 |                       |                       |                  |
| off | r3  | 194,571           | 0.8058              | 0                   | 150.3 / 10,956                 | 330.6 |                       |                       |                  |

Medians, on against off:

- Recomputed tokens: 163,166 against 202,388 (−19 %).
- `cached_tokens_ratio`: 0.8365 against 0.7997.
- Later-turn TTFT p50: 143 against 163 ms.
- Later-turn TTFT p99: 17.4 against 11.0 s.
- Output tok/s: 308 against 321.

Rungs at the end: L1 `fp8_e4m3`, L2 `tq4`. Labbook set `phase-6b-kv-compression`, runs `p6b-t16:mt:llama:ladder-{on,off}:r{1,2,3}`.

An earlier pass on 4e49409, before the merge, gave the same direction (recomputed 149,895 against 206,270; cached ratio 0.852 against 0.794; p99 27.8 against 14.0 s; one run lost 8 requests to `queue_timeout` at RED). It was superseded because the drift fixes were missing.

- L1 rewrites almost never store: 97–100 % end `no_room`. L1's two 1 GiB slabs are both `l0`-sized and never empty, so no slot of the `fp8_e4m3` size exists (slab fragmentation). The sweep picks the same blocks again every tick, up to 32 device transcodes per tick for nothing (2,290 submits in on-r3, the run with the 49 s p99 and 202 tok/s). That load is the likely cause of the worse tail.
- L2 rewrites fail `no_room` in 41–87 % of cases. In L2 that loses the copy: `L2NvmeTier::store` frees the replaced slot before it looks for one of the new size, and `on_copy_failed` then evicts the block. `a_rewrite_without_room_counts_no_room` pins this behaviour.
- The gain in recomputed tokens therefore comes from the L2 rewrites that do store, plus the `tq4` rung for new demotions into L2.

Shared-prefix GSM8K with the ladder (`p6b-eval-prefix.md` recipe on the ladder config plus `--set kv.gpu.max_bytes=4GiB`, candidate guard `--min-lossy-cached-ratio 0.5`): no usable report, nothing committed.

- Default thresholds: the whole eval stayed GREEN, so the ladder never ran; lossy ratio 0.000 (cached ratio 0.931, 726,723 prompt tokens, accuracy 0.745). The guard exited 1.
- `reliability.pressure.thresholds.kv_utilization=[0.03,0.06,0.9,0.97]` (decision "6b Task 16 … four points", 4 A): the eval ran at ORANGE from the first item. Still no ladder action: the sweep starts at the lowest tier (L2, which the eval leaves empty), and L1 holds the prefix at `l0` while L2 sits on its base rung. Admission also queued the items on `kv_reservation`, and item 6 got 503 `queue_timeout`, which aborted the run (exit 2). Lowering the thresholds cannot make this eval gate the ladder. Options are in `.procoder/handoff/p6b-t16.md`.

10-minute overload soak with the ladder on 4074b5d. Command: `scripts/overload-soak.sh novanas --duration 10m --set kv.cpu.enabled=true --set kv.cpu.max_bytes=4GiB --set kv.nvme.enabled=true --set kv.nvme.path=/home/piwi/turbine-kv-ladder --set kv.nvme.max_bytes=16GiB --set kv.ladder.enabled=true --set kv.ladder.l0=false --set kv.ladder.max_format=tq4`, wrapped in the `port18000` lock (serve run 1002075609-26334eb5).

- Verdict PASS: all 8 checks true. ITL p99 156 ms against a calibration of 176 ms; GREEN 0 s into the cool-down.
- Responses: 4709 × 200 and 98 × 429 `queue_full`; 4034 `queue_timeout`.
- `turbine_kv_ladder_actions_total` is 0, so the S-6 soak criterion ("> 0") is not met. The soak's random prompts share nothing: `turbine_kv_prefix_cached_tokens_total` was 0 and there were no demotions, so L1 and L2 stayed empty and the ladder had nothing to act on.
- An earlier attempt at 07:13 never started: another builder's soak server held port 18000. A second one, on 4e49409, was stopped during overload because the drift fixes were missing.

#### Ladder gate, soak and SURVIVAL burst after the fixes (2026-10-02; `p6b-t16` 25e8dc2)

Tree: the L2 copy-loss fix and the no-room back-off (56a2cb3, decision "6b Task 16: ladder proof results — four open points" 2 A, 3 A), the
eval runner's `--filler-concurrency`, filler resends and `--fillers-settle`, and 25e8dc2 `fix(server)`: an attach plans with the controller's
current pressure state (before, an idle engine planned a new request with the state of its last busy turn, so the first item after RED →
GREEN planned `l0_pressure` and recomputed the prefix into an exact L0 copy).

Shared-prefix GSM8K, ladder variant of spec S-8 (both sides `scripts/lab/phase6-novanas-ladder.yaml --set kv.gpu.max_bytes=4GiB --set
kv.cpu.max_bytes=4GiB --set kv.nvme.enabled=false`, baseline `--set kv.ladder.enabled=false`; c16, 32 fillers of 2000 words 16 at a time,
`--fillers-settle`). A fresh native server per run on GPU 0 (port 18010, `bench.lock`, fixtures paused), arms interleaved. "Served" is the sum of
`turbine_requests_total`, which matched 232 + `filler_retries` in every run.

| Arm        | Run | accuracy | lossy cached ratio | cached ratio | filler resends | served |
| ---------- | --- | -------- | ------------------ | ------------ | -------------- | ------ |
| ladder     | r1  | 0.750    | 0.927              | 0.931        | 79             | 311    |
| ladder     | r2  | 0.765    | 0.927              | 0.931        | 80             | 312    |
| ladder     | r3  | 0.765    | 0.927              | 0.931        | 78             | 310    |
| ladder off | r1  | 0.770    | 0.000              | 0.931        | 40             | 272    |
| ladder off | r2  | 0.775    | 0.000              | 0.931        | 40             | 272    |
| ladder off | r3  | 0.765    | 0.000              | 0.931        | 78             | 310    |

- Nine candidate × baseline pairs (`scripts/eval/paired_compare.py --max-drop 0.01`): drops 0.000–0.025, median 0.010, lowest McNemar p
  0.302. Gate PASS at the bound (`tests/eval/llama-3.2-3b-instruct/turbine-ladder-sp-paired.json`; per-run pairs r1 0.020 FAIL, r2 0.010, r3
  0.000).
- Every candidate rewrote the prefix's 23 L1 blocks `l0 → fp8_e4m3 → tq4` (`fill_high_water` 23 + 23) and served 198 items from them;
  plans `l0_pressure` 0, `retrieve_cheaper` 1, `all_l0` 198.

SURVIVAL at 12 or more concurrent fillers (diagnosis, nothing changed). `/turbine/v1/pressure` every 0.5 s (`lad-cand-r1`): the only transition
out of GREEN is `GREEN → SURVIVAL`, signal `kv_utilization` 0.9946 against the SURVIVAL threshold 0.97; `exhaustion_horizon` is null
throughout. The sample at SURVIVAL has KV `used` 350 MiB and `reserved` 3,724 MiB of 4,096 MiB, with 14 admitted and 4 queued on
`kv_reservation`: twelve fillers' worst-case reservations (about 310 MiB, 22 blocks each) were admitted within one controller sample. The
ledger is right (the reservations are real), and the forecast plays no part. The cause is a policy gap: the P3 S-9 headroom rule (an
admission waits when its reservation would lift `kv_utilization` past the next state's threshold) does not apply at GREEN, so admission at
GREEN reserves up to the whole pool, past the 0.97 SURVIVAL threshold. SURVIVAL then requeued 10 unstarted requests (`survival_requeue`),
rejected 79 fillers (`reject.survival`, the client's resends), and stepped down one level per 10 s dwell to GREEN 40 s later.

10-minute overload soak with the ladder and `--shared-prefix-share 0.5` (`scripts/overload-soak.sh novanas --duration 10m --shared-prefix-share
0.5 --set kv.cpu.enabled=true --set kv.cpu.max_bytes=4GiB --set kv.nvme.enabled=true --set kv.nvme.path=/home/piwi/turbine-kv-ladder --set
kv.nvme.max_bytes=16GiB --set kv.ladder.enabled=true --set kv.ladder.l0=false --set kv.ladder.max_format=tq4`, client on novanas, serve run
1002104721-2c9f5e72, `target/soak/novanas-20261002T104721Z`):

- Verdict PASS, all 8 checks true; ITL p99 210 ms against a calibration of 174 ms; GREEN 27 s into the cool-down.
- Responses: 4,629 × 200, 2,570 `queue_timeout`.
- `turbine_kv_ladder_actions_total` 169 (S-6 soak criterion met): L2 `l0 → fp8_e4m3` 42 and `fp8_e4m3 → tq4` 42 (`fill_high_water`),
  `new_demotion` into L2 at `tq4` 41, L1 `l0 → fp8_e4m3` 21 of which 21 `no_room` with one back-off, one L2 `rung_step_up`. Prefix cached
  tokens 2,167,168; demotions L0 → L1 1,968, L1 → L2 83.

Multi-turn A/B rerun on 25e8dc2 (after the L2 copy-loss fix and the no-room back-off), same workload and harness as the table above
(`scripts/lab/phase6-novanas-ladder.yaml` against `--set kv.ladder.enabled=false`; 24 sessions × 8 turns, 2000-word shared prefix, think time
1..4 s, session hints; arms interleaved, runs `target/t16/runs/mt2-{on,off}-r{1,2,3}`). States are the share of 0.5 s `/turbine/v1/pressure`
samples.

| Arm | Run | ok  | recomputed tokens | cached_tokens_ratio | lossy cached tokens | later-turn TTFT p50 / p99 (ms) | tok/s | L1 rewrites (no_room) | L2 rewrites (no_room) | L2 new demotions | L0→L1 / L1→L2 demotions | GREEN / YELLOW / ORANGE samples |
| --- | --- | --- | ----------------- | ------------------- | ------------------- | ------------------------------ | ----- | --------------------- | --------------------- | ---------------- | ----------------------- | ------------------------------- |
| on  | r1  | 192 | 126,856           | 0.8745              | 0                   | 126.4 / 40,453                 | 243.3 | 16 (16)               | 32 (32)               | 0                | 726 / 325               | 50 / 19 / 118                   |
| on  | r2  | 187 | 132,906           | 0.8643              | 20,096              | 169.5 / 59,668                 | 164.0 | 9 (9)                 | 330 (0)               | 21               | 523 / 186               | 40 / 17 / 219                   |
| on  | r3  | 192 | 180,486           | 0.8207              | 0                   | 158.5 / 11,615                 | 319.6 | 15 (15)               | 32 (32)               | 0                | 704 / 296               | 52 / 30 / 61                    |
| off | r1  | 192 | 176,657           | 0.8246              | 0                   | 141.6 / 15,328                 | 336.0 |                       |                       |                  | 1,567 / 820             | 41 / 44 / 52                    |
| off | r2  | 192 | 218,835           | 0.7834              | 0                   | 135.6 / 14,151                 | 326.6 |                       |                       |                  | 1,770 / 983             | 46 / 76 / 20                    |
| off | r3  | 192 | 211,884           | 0.7904              | 0                   | 146.5 / 9,233                  | 350.8 |                       |                       |                  | 1,724 / 986             | 41 / 70 / 23                    |

Medians, on against off:

- Recomputed tokens: 132,906 against 211,884 (−37 %; −19 % before the fixes).
- `cached_tokens_ratio`: 0.8643 against 0.7904.
- Later-turn TTFT p50: 159 against 142 ms.
- Later-turn TTFT p99: 40.5 against 14.2 s.
- Output tok/s: 243 against 336 (−28 %).

- The L1 spin is gone: each L1 no-room is followed by a back-off (`no_room_backoff` 1–2 per run), against 369–2,290 futile L1 submits
  before. L2 rewrites no longer lose copies; in r1 and r3 all 32 end `no_room` with the copy kept, in r2 all 330 stored.
- New: the ladder arm demotes less than half as much (L0 → L1 523–726 against 1,567–1,770) and spends longer at ORANGE (median 118 against 23 samples), where
  admission waits and prefill is throttled; that is where the worse tail, the lower tok/s and on-r2's 5 `queue_timeout` failures come
  from. Not diagnosed here (open point in `.procoder/handoff/p6b-t16.md`).
- Labbook set `phase-6b-kv-compression`, runs `p6b-t16:mt2:llama:ladder-{on,off}:r{1,2,3}`.

### KV copies and decode steps (decision "6b: step-time drift during KV promotions", C; branch `p6b-copydrift`)

Part 1 (signal), commit 05471e6 `fix(reliability)`: the `step_time_drift` window neither judges nor learns a pure decode step that overlapped an in-flight KV tier copy (`StepSample.copy_overlap`, `kv_orchestrator::CopyMark`: a copy in flight at the step's launch or collection, or found or left in flight by a transfer poll in between). They are counted in `turbine_decode_steps_unjudged_total{reason="kv_copy"}` and the `decode_step` trace (`kv_copy`, `kv_copy_excluded`). They still count for throughput.

Part 2 (why promotions slow decode), from the `p6b-drift` traces (`scratch/p6b-drift/olmoe-fix/*.server.log`, `scratch/p6b-copydrift/corr2.py`; judged decode steps against the `kv_copy_stages` of L1 → L0 copies that overlap them):

| run          | steps ≥ 1.5× | step ms (slow / normal) | extra ms | L1→L0 MiB in flight | extra vs MiB (slope, r) |
| ------------ | ------------ | ----------------------- | -------- | ------------------- | ----------------------- |
| OLMoE l0 r1  | 9            | 26.9 / 15.2             | 11.6     | 176                 | 0.065 ms/MiB, r 0.99    |
| OLMoE l0 r2  | 13           | 30.1 / 14.3             | 11.9     | 192                 | 0.063 ms/MiB, r 0.98    |
| OLMoE tq4 r1 | 3            | 22.1 / 14.8             | 8.5      | 90                  | 0.084 ms/MiB, r 0.67    |

- A step's extra time is linear in the promotion bytes in flight with it (r 0.98–0.99 for `l0`). The slope is 0.063–0.065 ms/MiB, about 16 GB/s, the order of the copy stream's SDMA rate (≈ 12 GB/s, `p6b-d2h`). So the step waits for a share of the queued copy bytes to drain, and does not slow by a constant factor. HBM contention cannot do that: 12 GB/s is under 2 % of the card's memory bandwidth.
- `l0` promotions are pure H2D on the copy stream. The copy does no compute-stream fence (`ShimContext::enqueue` fences only device-source copies) and runs no transcode kernel, yet `l0` shows the largest effect. The compute-stream fence and the staging decode kernel are therefore not the mechanism for `l0`. For `tq4`, the decode kernel runs on the compute stream (0.04 ms a block) after the host sees the H2D finish, which is too small to matter.
- Remaining candidate: the decode step's own transfers on the compute stream (the batch metadata upload through staging, the logits inputs and the reads, `turbine_model::executor::batch` / `logits`) queue on the SDMA engine behind the copy stream's batches, up to `kv.transfer.max_inflight_bytes` (default 1 GiB) ahead of them. 13 × 16 MiB in flight take about 17 ms at 12 GB/s, and the measured extra is 14–16 ms. To be confirmed with rocprofv3 (`--kernel-trace --memory-copy-trace`, `scratch/p6b-copydrift/prof.py`: compute-stream copy durations and gaps inside and outside promotion intervals, against kernel durations).

Part 3 (rocprofv3 and the in-flight cap), tree 108c734 (`p6b-stack` with the drift exclusion), harness `scratch/p6b-copydrift/ab.sh` on novanas, GPU 0, OLMoE `l0` L1, the multi-turn workload of "Lossless last block scored like its history". Results in `scratch/p6b-copydrift/{prof,cap-def,cap-64,kernarg,ab}/`; scripts `corr3.py` (step extra time against L1 → L0 bytes in flight, baseline from the steps that overlap no copy, since overlapped steps are no longer judged), `prof2.py`, `blit.py`, `queue.py` and `align.py` (rocprofv3 `--kernel-trace --memory-copy-trace`).

- The copy stream's promotions run on SDMA as one 1 MiB copy per layer (16 per 16 MiB block), 83 µs each (12.6 GB/s), never two at once. During serving, decode does no SDMA copy at all: the compute stream's 3,320 H2D copies are the weight upload, all in the first 13 s. Decode's batch upload is a `__amd_rocclr_copyBuffer` blit kernel on the compute stream (6,514 of them).
- While an SDMA copy runs, every compute-stream kernel stretches to about one copy's length. The batch-upload blit takes 83.8 µs (2.5 µs otherwise) and ends a median 18 µs after the SDMA copy that was running when it started. `rmsnorm` takes 79.7 µs (6.4 otherwise), `moe_topk` 80.6 µs (11.0), and the hipBLASLt GEMMs 2.5× longer. Gaps between compute kernels grow to 85 µs (3.6 µs otherwise), and to 397 µs before a blit (7.6 µs otherwise). Kernel occupancy is 0.45 inside promotion intervals against 0.83 in the windows just before them. So decode's small transfers do not queue behind the copy batch: they wait for the one copy in flight. Compute largely stalls for as long as SDMA streams, which gives 61–69 µs lost per MiB promoted against 83 µs per MiB copied.
- `HIP_FORCE_DEV_KERNARG=1` (diagnostic env only, 2 runs) leaves the slope unchanged at 0.060–0.064 ms/MiB, so kernel arguments in host memory are not the mechanism.

| OLMoE `l0`, 2 runs each              | default (1 GiB cap)        | `kv.transfer.max_inflight_bytes=64MiB` |
| ------------------------------------ | -------------------------- | -------------------------------------- |
| decode steps overlapping a promotion | 38 / 36                    | 84 / 83                                |
| L1 → L0 MiB in flight (mean / max)   | 84 / 304, 83 / 208         | 43 / 64, 45 / 64                       |
| extra ms per overlapped step (mean)  | 6.2 / 6.5                  | 3.6 / 4.1                              |
| extra ms at the most bytes in flight | 16.8 (≥ 256 MiB)           | 4.5–4.6 (64 MiB)                       |
| slope (ms/MiB, r)                    | 0.063 r 0.98, 0.069 r 0.95 | 0.047 r 0.67, 0.042 r 0.53             |
| total extra over the run (ms)        | 236 / 234                  | 302 / 336                              |
| decode steps ≥ 1.5× their baseline   | 17 / 14                    | 0 / 0                                  |
| L1 → L0 latency median / p99 (ms)    | 29.4 / 46.5, 30.1 / 46.7   | 19.1 / 39.5, 20.8 / 41.9               |
| cached_tokens_ratio                  | 0.8623 / 0.8631            | 0.8613 / 0.8629                        |
| output tok/s                         | 252.3 / 249.2              | 248.0 / 245.4                          |
| later-turn TTFT p50 / p99 (ms)       | 52.7 / 127, 54.9 / 204     | 52.8 / 271, 52.7 / 346                 |

- The cap trades a few large stalls for twice as many small ones. The total time lost stays about the same, a little higher (the cost is per byte, not per batch). Promotions finish sooner because they queue less. Reuse is unchanged. tok/s is 1.6 % lower and later-turn TTFT p99 is worse. It is not better on every measure, so the default stays.

Tq4 multi-turn A/B rerun with the drift exclusion (decision "6b: OLMoE tq4 after the last-block change" A), same tree and harness, no debug logging, arms interleaved, results in `scratch/p6b-copydrift/ab/`:

| Model | L1 format | cached_tokens_ratio (runs) | median | output tok/s (runs)   | later-turn TTFT p99 (ms) | pressure                              |
| ----- | --------- | -------------------------- | ------ | --------------------- | ------------------------ | ------------------------------------- |
| OLMoE | `l0`      | 0.8625 / 0.8624 / 0.8608   | 0.8624 | 248.6 / 241.8 / 250.0 | 322 / 170 / 328          | GREEN throughout                      |
| OLMoE | `tq4`     | 0.8625 / 0.8595 / 0.8629   | 0.8625 | 247.8 / 249.1 / 244.0 | 217 / 140 / 275          | GREEN throughout                      |
| Llama | `l0`      | 0.9007 / 0.9040 / 0.8980   | 0.9007 | 336.5 / 339.8 / 331.6 | 302 / 476 / 376          | YELLOW on `kv_utilization` at 29–33 s |
| Llama | `tq4`     | 0.9061 / 0.9076 / 0.9075   | 0.9075 | 344.2 / 344.8 / 342.9 | 290 / 312 / 447          | YELLOW on `kv_utilization` at 29–33 s |

- Gate holds on both models: OLMoE `tq4` 0.8625 ≥ `l0` 0.8624 (a 0.0001 margin, inside run-to-run spread), Llama `tq4` 0.9075 ≥ `l0` 0.9007. No run went YELLOW on `step_time_drift`; the drift window left 66–87 (`l0`) and 157–254 (`tq4`) copy-overlapped steps unjudged per run. Both Llama arms leave GREEN the same way (L0 fill reaches 0.71–0.78 against 0.70, a real signal), so the arms run under the same pressure. Llama `l0` drops 90–110 blocks on L1 capacity per run, `tq4` none.
- Flipped: lower-tier `tq4` is `supported` (`TIER_FORMAT_REFUSALS` keeps only `tq2`).

### Promotion copy kernel (decision "6b: KV promotions slow decode — which fix", A; branch `p6b-copykernel`)

Microbenchmark `kernels/rocm/tools/copy_eval.cpp` (`turbine_copy_eval`), novanas GPU 0 under the bench lock, ROCm 7.14. A decode-like loop runs on one non-blocking stream: per step one 16 KiB pinned upload and 300 pairs of an rmsnorm-like kernel (16 rows × 2048) and a 256-workgroup weight-stream kernel (8 MiB from a 512 MiB ring), 2.3 ms a step (11.3 ms with 1500 pairs). 256 MiB of pinned (`hipHostMallocDefault`, like L1) → device copies start at step 2 on a second non-blocking stream. The slope is the extra compute time over the no-copy median step, divided by the MiB copied. Each cell is the median of 3 runs; the copied bytes are checked.

| H2D path (256 MiB)                                  | segment | GB/s under compute (alone) | slope ms/MiB (3 runs)  | worst step ms (base 2.3) |
| --------------------------------------------------- | ------- | -------------------------- | ---------------------- | ------------------------ |
| SDMA `hipMemcpyAsync` (today)                       | 1 MiB   | 12.42 (12.27)              | 0.0495 (0.0495–0.0500) | 5.9                      |
| SDMA                                                | 256 KiB | 10.50 (10.51)              | 0.0532                 | 5.1                      |
| SDMA                                                | 64 KiB  | 6.65 (6.55)                | 0.0370                 | 3.1                      |
| blit kernels, `HSA_ENABLE_SDMA=0`                   | 1 MiB   | 16.03 (15.85)              | 0.0585                 | 17.6                     |
| blit, `HSA_ENABLE_SDMA=0 DEBUG_CLR_LIMIT_BLIT_WG=2` | 1 MiB   | 16.01 (15.80)              | 0.0587                 | 17.5                     |
| SDMA, `HSA_ENABLE_SDMA_HDP_FLUSH=0`                 | 1 MiB   | 12.53 (12.37)              | 0.0502                 | 6.1                      |
| copy kernel, 1 workgroup                            | 1 MiB   | 15.65 (15.50)              | 0.0006                 | 2.5                      |
| copy kernel, 2 workgroups                           | 1 MiB   | 22.31 (23.40)              | 0.0029 (0.0020–0.0034) | 2.8                      |
| copy kernel, 2 workgroups                           | 256 KiB | 20.43 (22.72)              | 0.0034                 | 2.7                      |
| copy kernel, 3 workgroups                           | 1 MiB   | 22.44 (23.06)              | 0.0079                 | 3.1                      |
| copy kernel, 4 workgroups                           | 1 MiB   | 21.94 (22.83)              | 0.0135                 | 3.6                      |
| copy kernel, 6 workgroups                           | 1 MiB   | 21.75 (21.54)              | 0.0181                 | 4.1                      |
| copy kernel, 16 / 32 / 64 workgroups (1 run)        | 1 MiB   | 12.1 / 10.0 / 9.1          | 0.065 / 0.095 / 0.112  | 10.6 / 26.9 / 31.2       |
| SDMA, 11.3 ms steps                                 | 1 MiB   | 12.11                      | 0.0495                 | 25.0 (base 11.3)         |
| copy kernel, 2 workgroups, 11.3 ms steps            | 1 MiB   | 21.50                      | −0.0003                | 13.4 (base 11.3)         |

- The loop reproduces the serving slope: SDMA promotions cost 0.050 ms per MiB (serving: 0.061–0.069). Single runs isolate it: only the small kernels (`CB_ONLY=small`) or only the stream kernel give 0.034 / 0.062, so every kernel kind stalls, and the per-step upload is not the cause (`CB_NOUP`: 0.051).
- The stall comes from host-link reads in flight, not from the copy engine. Device-to-device copies of the same size (`CB_SRC=dev`, SDMA or kernel, no host link, 1 run each) cost nothing (−0.002 / −0.005). A copy kernel stalls compute in proportion to its workgroups, that is to its reads in flight: 0.0006 at 1, 0.003 at 2, 0.065 at 16 (SDMA's level) and 0.11 at 64. The runtime's blit kernels use many workgroups, and `DEBUG_CLR_LIMIT_BLIT_WG` does not change them. Smaller SDMA segments cut the slope per MiB only by moving fewer MiB per second (64 KiB: 0.037 at 6.7 GB/s), so the cost per second of copying stays the same.
- Chosen: a copy kernel with 2 workgroups on the copy stream (the knee: rate at its peak, 22 GB/s, 1.8× SDMA, and the slope 17× lower). Reuse first: ROCm offers no host-to-device copy with a bounded number of reads in flight (SDMA, or blit kernels whose workgroup count cannot be limited). SGLang's `sgl-kernel` kvcacheio (`transfer_kv_*`) and LMCache's `multi_layer_kv_transfer` are CUDA kernels of this shape that do not ship for ROCm. The copy is a vector load/store loop, so it is ours: `kernels/rocm/src/copy_kernel.hip`, ABI v2.11 `turbine_memcpy_h2d_kernel` (v2.11 has not shipped; optional symbol), behind `kv.transfer.promotion_copy: kernel` (default `sdma`).

Lab A/B, tree 314c848 (`p6b-stack` 6652755 + the switch), harness `scratch/copykernel-ab/ab.sh` (the `p6b-copydrift` one) on novanas GPU 0: OLMoE `l0` L1, the multi-turn workload of "Lossless last block scored like its history", debug traces on, a fresh server per arm, arms interleaved, 5 runs each. `corr3.py` gives the slope from the `decode_step` and `kv_copy_stages` traces.

| OLMoE `l0`, 5 runs each                       | `promotion_copy: sdma` (default) | `promotion_copy: kernel`           |
| --------------------------------------------- | -------------------------------- | ---------------------------------- |
| slope, extra ms per MiB promoted in flight    | 0.064 (0.058–0.066, r 0.87–0.98) | 0.000 (−0.004–0.003, r −0.11–0.11) |
| extra ms per promotion-overlapped step (mean) | 6.89 (5.33–7.41)                 | 0.97 (0.88–1.63)                   |
| decode steps ≥ 1.5× their baseline            | 17 (8–19)                        | 3 (3–4)                            |
| L1 → L0 latency median / p90 (ms)             | 30.2 / 45.9                      | 19.5 / 31.3                        |
| L1 → L0 latency p99 (ms, runs)                | 52.7 / 57.1 / 65.5 / 43.1 / 48.5 | 41.0 / 293 / 276 / 54.1 / 48.5     |
| cached_tokens_ratio                           | 0.8616 (0.8607–0.8620)           | 0.8611 (0.8604–0.8631)             |
| output tok/s                                  | 246.7 (242.4–251.7)              | 246.4 (230.3–248.8)                |
| later-turn TTFT p50 / p99 (ms)                | 50.1 / 360 (325–412)             | 50.3 / 288 (107–336)               |

- The copy kernel removes the per-byte decode cost of promotions: the slope drops from 0.064 ms/MiB to zero, and the steps that overlap a promotion lose 1.0 ms instead of 6.9. Promotions finish a third sooner (median 30 → 20 ms). Reuse is unchanged, later-turn TTFT p99 is lower in the median (360 → 288 ms), and median tok/s is the same (two kernel runs at 230 and 235 widen the spread; the decode time saved, ~0.2 s per run, is under 1 % of the wall time).
- The two p99 outliers of the kernel arm (276 and 293 ms) are copy bounds, not measured copy times: the copy stream has no timestamps, so a copy ends somewhere between the last poll that saw it running and the poll that saw it done. In r2 all ten long copies spanned an idle engine (no step in between). In r3, three copies spanned a 257 ms host-side gap between two decode steps, and gaps of that length occur in both arms (sdma 100–502 ms, kernel 104–1371 ms). Whether the kernel copies were really still running then is not decided by these traces.
- The default stays `sdma` (flipping it is the user's call). A shim mutation (tail loop off by one byte) fails `pinned_copy_kernel_is_bit_exact` on the GPU. Over a Llama block's 28 × 512 KiB batches the lab test measured 23.79 GB/s for the kernel against 11.67 GB/s for SDMA on GPU 0 (18.46 / 9.57 in a k3s Job on an unpinned card).

Llama A/B (decision "6b: promotion copy kernel — default" A, step 1), tree a43c4f7 (`p6b-stack` a4e0941 plus the `copy_eval` D2H variant; server unchanged), binaries frozen in `scratch/copykernel-llama/frozen/` on novanas, harness `ab.sh` there: GPU 0, Llama `l0` L1 (`kv.cpu.max_bytes=4GiB`), 16 sessions × 8 turns at c16, think time 1..4 s, `--shared-prefix-words 2000`, session hints, debug traces on, a fresh server per arm, arms interleaved, both arms set explicitly, 5 runs each. No `step_time_drift` transition in any run.

| Llama `l0`, 5 runs each                       | `promotion_copy: sdma`           | `promotion_copy: kernel`         |
| --------------------------------------------- | -------------------------------- | -------------------------------- |
| slope, extra ms per MiB promoted in flight    | 0.056 (0.046–0.057, r 0.81–0.99) | 0.006 (0.001–0.008, r 0.05–0.58) |
| extra ms per promotion-overlapped step (mean) | 9.83 (7.96–12.35)                | 1.41 (1.12–1.64)                 |
| decode steps ≥ 1.5× their baseline            | 14 (13–17)                       | 4 (2–4)                          |
| L1 → L0 latency median / p90 (ms)             | 41.6 / 46.8                      | 24.2 / 29.4                      |
| L1 → L0 latency p99 (ms, runs)                | 86.4 / 71.3 / 79.8 / 47.2 / 282  | 70.4 / 260 / 25.8 / 64.0 / 40.6  |
| cached_tokens_ratio                           | 0.9054 (0.9015–0.9059)           | 0.9031 (0.8955–0.9061)           |
| output tok/s                                  | 331.4 (324.6–347.3)              | 342.5 (331.8–347.6)              |
| later-turn TTFT p50 / p99 (ms)                | 80.0 / 389 (185–1564)            | 78.4 / 325 (170–378)             |

- Llama confirms OLMoE: the slope drops 9× (0.056 → 0.006 ms/MiB), overlapped steps lose 1.4 ms instead of 9.8, promotions are 42 % faster, tok/s +3.3 % in the median and TTFT p99 lower. The cached ratio is 0.002 lower in the median (one kernel run at 0.8955); within run-to-run spread of the sdma arm's low run (0.9015), noted as a watch item.
  Validation of the default flip (decision "6b: promotion copy kernel — default" A, steps 2–3), tree 5076c04 (server code unchanged from 4ba941c, `p6b-stack` 2c027ef merged in).

Step 2, `lab-bench --golden16` with `kv.transfer.promotion_copy=kernel` explicit, novanas GPU 0, the phase2c configs:

| Model | golden c1 / c16 | tok/s (6a exit)       | ITL p50 (ms) | TTFT p50 / p99 (ms) |
| ----- | --------------- | --------------------- | ------------ | ------------------- |
| Llama | PASS / PASS     | 853.0 (854.2, 0.999×) | 15.4         | 207 / 774           |
| OLMoE | PASS / PASS     | 612.7 (602.5, 1.017×) | 24.5         | 119 / 391           |

Both above the ≥ 0.98× bound. Llama matches the recent 6b rows (844–850); OLMoE is the best OLMoE c16 row in the set (previous best 610.1).

Step 3, demotions, `copy_eval CB_DIR=d2h` (a43c4f7), novanas GPU 0, 256 MiB total, medians of 3 runs (`scratch/copykernel-d2h/run.sh` under the bench locks; raw output attached to the labbook runs):

| Cell                                   | copy GB/s               | slope (ms/MiB)                | extra ms (overlapped)     |
| -------------------------------------- | ----------------------- | ----------------------------- | ------------------------- |
| H2D SDMA, 1 MiB segments (reference)   | 11.9                    | 0.053                         | 13.6                      |
| H2D copy kernel, 2 workgroups (ref.)   | 21.7                    | 0.004                         | 0.9                       |
| D2H SDMA, 1 MiB / 256 KiB / 64 KiB     | 11.8 / 10.9 / 7.2       | 0.053 / 0.049 / 0.045         | 13.7 / 12.5 / 11.5        |
| D2H copy kernel, 1 / 2 / 4 / 16 wg     | 13.5 / 11.6 / 9.5 / 8.1 | 0.056 / 0.073 / 0.095 / 0.124 | 14.2 / 18.7 / 24.4 / 31.7 |
| D2H SDMA vs kernel 2 wg, 11.3 ms steps | 11.7 / 12.5             | 0.035 / 0.060                 | 9.0 / 15.4                |

- Demotions stall decode the same way as promotions: D2H SDMA's slope matches H2D's (0.053 ms/MiB), and smaller segments do not help (the same finding as H2D). The serving traces agree: in the Llama A/B runs, demotions overlap 217–243 decode steps per run at a slope of 0.046–0.047 ms/MiB (mean 69–81 MiB in flight, 4–5 ms per overlapped step, ~17 GiB demoted per run) — identical in both arms, since demotions run on SDMA either way and the switch only touches H2D promotions. That is ~1 s of added decode time per multi-turn run, under 1 %.
- The copy kernel does not transfer to D2H: a device → pinned copy is as bad as SDMA at 1 workgroup and worse from 2 up (0.073 at 2, 0.124 at 16 — the H2D pattern inverted). Its host stores hit the same host-link path without the SDMA engine's efficiency. A demotion copy kernel is not worth implementing on this evidence.
- Verdict (user decision A): both legs hold — golden c1 and c16 pass on both models, tok/s 0.999× / 1.017× of the 6a-exit baselines, demotions unchanged — so the `kernel` default (4ba941c) stays. The cached-ratio watch item is unchanged: Llama A/B medians 0.9054 (sdma) vs 0.9031 (kernel), one kernel run at 0.8955 against an sdma low run of 0.9015; the golden16 bench and OLMoE show no drop beyond that spread.

### Ladder rung-slot fallback (decision "6b Task 16: ladder-on throughput regression", 2 A; branch `p6b-ladderperf`)

Tree 81dfd76 (`p6b-stack` 2c027ef + the fix; `kv.transfer.promotion_copy: sdma` — the copy-kernel branch `p6b-copykernel` is not merged, so these runs still carry the SDMA promotion cost). The fix: when a slab tier's rung format has no free slot (`take_slot` reformats a slab only when it empties, which never happens under load), a new demotion stores at the tier's own format instead of ending `Full` and dropping. `copy_codec` takes the rung only while `rung_has_slot` (`KvTier::free_slots(format, bytes)` > copies in flight into the tier at that size); the fallback counts `rung_no_slot`. Root-caused during the test resolution: `make_room`'s rung-unit arithmetic also undercounted (3 l0 slots = 5 rung units but 6 needed), so all room accounting is now byte-accurate (`make_room` returns bytes; `demotion_bytes`, eviction frees the victim's stored bytes). Mutation checks kill their tests for the fallback, the in-flight subtraction and the L1 empty-slab credit.

Multi-turn A/B, same workload and harness as the mt2 table above (`scripts/lab/phase6-novanas-ladder.yaml` against `--set kv.ladder.enabled=false`; 24 sessions × 8 turns, 2000-word shared prefix, think 1..4 s, session hints; a fresh native server per run on GPU 0 under the port and bench locks, arms interleaved per rep, runs `target/ladderperf/mt3/`). States are seconds at the state over 0.5 s `/turbine/v1/pressure` samples.

| Arm | Run | ok  | recomputed tokens | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | tok/s | ORANGE (s) | rung_no_slot | L0→L1 / L1→L2 demotions | rung L1 / L2 at end | formats at end     |
| --- | --- | --- | ----------------- | ------------------- | ------------------------------ | ----- | ---------- | ------------ | ----------------------- | ------------------- | ------------------ |
| on  | r1  | 192 | 192,637           | 0.8087              | 155.1 / 18,148                 | 311.7 | 25.0       | 1,134        | 1,473 / 727             | fp8 / fp8           | l0                 |
| on  | r2  | 192 | 172,543           | 0.8276              | 149.1 / 9,101                  | 329.2 | 28.0       | 1,376        | 1,709 / 1,052           | fp8 / fp8           | l0 + some fp8      |
| on  | r3  | 192 | 148,242           | 0.8519              | 128.0 / 16,573                 | 328.3 | 8.5        | 878          | 1,340 / 769             | fp8 / tq4           | l0 + some fp8, tq4 |
| off | r1  | 192 | 200,087           | 0.8003              | 143.5 / 13,256                 | 324.2 | 22.0       | —            | 1,683 / 1,000           | l0 / l0             | l0                 |
| off | r2  | 192 | 197,976           | 0.8047              | 141.8 / 12,259                 | 306.5 | 22.0       | —            | 1,698 / 970             | l0 / l0             | l0                 |
| off | r3  | 192 | 189,945           | 0.8109              | 145.6 / 10,247                 | 319.2 | 22.0       | —            | 1,688 / 936             | l0 / l0             | l0                 |

Medians, on against off: recomputed tokens 172,543 against 197,976 (−13 %); `cached_tokens_ratio` 0.8276 against 0.8047 (+2.3 pp); later-turn TTFT p50 149 against 144 ms; later-turn TTFT p99 16.6 against 12.3 s; tok/s 328 against 319 (+3 %); ORANGE 25 against 22 s. All runs 192/192 ok, RED/SURVIVAL 0 s.

- The mt2 regression is gone: on 243 → 328 tok/s (off 336 → 319 shows ±5 % run-to-run spread across the two sessions), later-turn TTFT p99 40.5 → 16.6 s against off 12.3 s, ORANGE 118 → 25 samples, no `queue_timeout` (5 in mt2-on-r2). The ladder arm is at parity with off on throughput and retains its recompute win (−13 %) with zero failed requests in all six runs.
- The fallback does all the work: 878–1,376 `rung_no_slot` per run, and at the end almost every stored copy is still `l0` — a slab tier whose slabs hold `l0` slots never holds rung-format copies while it stays non-empty. The rungs step down (L1 reaches `fp8_e4m3` in every on-run, L2 reaches `tq4` in r3) but a rung change alone converts nothing.
- Labbook set `phase-6b-kv-compression`, runs `p6b-ladderperf:mt3:llama:ladder-{on,off}:r{1,2,3}`.

Slab re-sizing (B) against smaller slabs (C), on these numbers — the user decides; nothing implemented:

- With the fallback alone the lower tiers compress nothing: L1's 2 GiB and L2's 4 GiB hold 438 l0 blocks (146 + 292 at 14,680,064 B/block). At the rung formats the same bytes hold 876 fp8, 1,748 tq4 or 3,496 tq2 blocks, so B or C would more than double (fp8) to 4–8× the retained-block capacity exactly where recompute comes from — the ladder arm still recomputes 148k–193k tokens per run, nearly all of it blocks that overflowed L1/L2 and were dropped.
- The retention gain already visible without any compression (cached ratio +2.3 pp, recompute −13 %, from ladder ordering alone — the off-arm churns 1,683 L0 → L1 demotions per run against the on-arm's 1,473) is the shape of what B/C would amplify; the mt2 run that stored 330 L2 copies (r2) still lost throughput to the old defect, so there is no clean measurement of a compressed-copy serving win yet.
- B (reformat a slab when its rung changes, spilling or evicting its copies) gets full-size slabs of the rung format but does slab-by-slab eviction work per rung change and throws away or re-stores the displaced copies; C (slabs at compressed slot sizes, e.g. 1/2–1/8 the 14.7 MB l0 slot) lets `take_slot` reformat naturally as slabs empty and needs no eviction, at the cost of more, smaller pinned allocations. C is the lower-risk path and matches how the soak run already stored 41 tq4 copies into L2; B buys conversion now, at a copy/evict cost per rung change.

### GREEN admission headroom and the SURVIVAL liveness scenario (decision "6b: which cap the GREEN admission headroom uses", A; branch `p6b-greenhead`, f58208d)

Decision A caps GREEN's admission headroom at RED's 0.90: `Admission::within_headroom` queues `kv_reservation` from GREEN too, so a burst can never lift held + reserved past RED's threshold. That closes the hole the pre-fix t16 record showed — a 16-filler burst to kv_utilization 0.9946 that jumped GREEN → SURVIVAL on admissions alone. Host check: 14 × 70-block requests of 1,024 tokens admit 13 and queue the 14th with `kv_reservation` (`admission::tests::kv_headroom_green_burst`). Red-before-fix and the mutation check ran as one operation on the stashed tree: with the `Green` arm removed the test fails (14 admitted), snapshot kept at `target/greenhead/admission.rs.mutated.bak`.

The cap also removes the liveness scenario's SURVIVAL entry: held + reserved is bounded at 0.90, below SURVIVAL's 0.9215 exit threshold, so SURVIVAL cannot latch from reservations, and the exhaustion horizon cannot fire inside the cap's headroom (≈ 2 s > the 1 s threshold). Sweep after the fix (seeds 1–12, both `survival_liveness` options, `target/greenhead/sweep-after-fix.log`): every seed peaks at RED and recovers in 43.4–50.2 s, `survival false` everywhere (seed 6: 47.1 s, 531 completions). The regression therefore enters SURVIVAL the way the engine does under a real fault, via S-11's device OOM: `OverloadConfig::oom_once_at` injects one pool-reservation failure the first time `run_load` passes the given virtual time. Seed 6 with the OOM at 60 s passes through SURVIVAL and is back GREEN + HEALTHY 42 s after the load stops (47.4 s under `continue_prefills`), with the full `assert_survival_case` invariants. Under the sustained 10× load behind the fault the engine rejects instead of failing (seed 6: 120 completed against 4,823 `queue_timeout`, 1,741 `queue_full`, 247 `overloaded`) — the spec'd overload response, not a regression. Gate: 857 pass / 0 fail.

Lab on novanas GPU 0 (clean tree f58208d):

- Burst (t16 recipe): 32 × 2,000-word fillers at concurrency 16 against a 4 GiB L0 (the `phase6-novanas-ladder` config, ladder on), pressure polled at 0.5 s (`target/greenhead/burst-pressure.log`): peak kv_utilization 0.8271, transitions GREEN → ORANGE → YELLOW → GREEN only — no RED, no SURVIVAL; 15 queued at the peak (6 on `kv_reservation`, 11 on `pressure_orange`); the eval client answers 2/2.
- Quick bench: `scripts/lab-bench.sh --quick --model llama` — tok/s 877.3 against the ~850 baseline, golden at concurrency 1 PASS, ITL p50 15.4 ms, 64/64 ok (`target/greenhead/soak-bench.log`).
- The 10-minute overload soak PASSes 8/8 on f58208d (serve 1003114226-29519f6d, `target/soak/novanas-20261003T114226Z`, client on novanas through the `SOAK_BENCH` ssh wrapper — the configuration of every earlier PASS soak): ITL p99 206.0 ms against calibration 173.2 (within 2×), GREEN 5 s into the cool-down, KV idle and the reserve held at the end, 4,632 × 200 / 21 × 429 `queue_full`, 3,321 `queue_timeout`, no client drop, `streams_complete` true. The soak shifts rejections from SURVIVAL's 503 `overloaded` to gate `queue_timeout`s on open streams, as designed (pre-cap soaks: ~70 × 503, 2,282–2,835 `queue_timeout`). An earlier attempt with the client on the workstation (`target/soak/novanas-20261003T082234Z`) failed only `streams_complete` — 48 of 4,603 streams ended without `[DONE]` over the ProxyJump dgx-spark VPN; the server emits `[DONE]` after every mid-stream error (`turbine-api/src/openai/stream.rs` `StreamState::fail`), so that is transport loss (~1%), not server behavior. SOAK_BENCH caveat for the next runner: the wrapper must forward `"$@"` itself and relocate `--pressure-timeline` to a remote path (the soak passes a workstation-local path).

### L1 slab size 128 MiB (decision "6b: after the demotion fallback — C: smaller L1 slabs"; branch `p6b-t16-slabs`, 4c62e82)

`kv.cpu.slab_bytes` (default 128 MiB, was the hard-coded 1 GiB) sizes each pinned L1 slab; validation refuses 0 and a size below one L0-format block (exit 2). Arithmetic at Llama-3.2-3B (`block_tokens` 128; l0 14,680,064 B, fp8 7,340,032, tq4 4,128,768, tq2 2,293,760): a 128 MiB slab holds 9 l0 / 18 fp8 / 32 tq4 / 58 tq2 slots (worst slack 2.0 MiB, 1.6 %); OLMoE l0/fp8 fit exactly (8 / 16). 2 GiB of L1 becomes 16 slabs instead of 2, so a slab drains (and `take_slot` reformats it) after 9 instead of 73 evictions. The host AC: with two slabs full of l0 blocks, evicting one 128 MiB slab's worth makes room for a tq4 block beside the still-occupied l0 slots (`tier::tests::l1_default_slab_size_stores_a_new_format_block_beside_occupied_old_format_slots`; red at the old 1 GiB default with a fixed 128 MiB eviction budget, mutation-checked both ways).

Multi-turn A/B on novanas GPU 0, same mt3 workload and harness as the ladderperf table (24 sessions × 8 turns, 2000-word shared prefix, think 1..4 s, session hints; ladder on, `scripts/lab/phase6-novanas-ladder.yaml`; before arm `--set kv.cpu.slab_bytes=1GiB`, after arm the 128 MiB default; L2 untouched at its 1 GiB slab size; a fresh native server per run under the port and bench locks, runs `novanas:/home/piwi/slabs/runs/`, summary `summary.json`):

| Arm     | Run | ok  | recomputed tokens | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | tok/s | ORANGE (s) | L1 rung_no_slot | L0→L1 fill/no_room | formats at end L1 / L2 |
| ------- | --- | --- | ----------------- | ------------------- | ------------------------------ | ----- | ---------- | --------------- | ------------------ | ---------------------- |
| 1 GiB   | r1  | 192 | 160,262           | 0.8398              | 131.4 / 17,252                 | 278.6 | 20.5       | 188             | 17 / 17            | l0 / fp8 + l0          |
| 1 GiB   | r2  | 192 | 150,925           | 0.8521              | 136.3 / 18,225                 | 291.2 | 25.0       | 424             | 28 / 3             | l0 / fp8 + l0 + tq4    |
| 1 GiB   | r3  | 192 | 162,494           | 0.8374              | 130.9 / 16,806                 | 330.4 | 25.0       | 753             | 28 / 28            | l0 / l0                |
| 128 MiB | r1  | 192 | 212,583           | 0.7878              | 147.1 / 17,361                 | 305.2 | 27.0       | 757             | 20 / 20            | l0 / l0                |
| 128 MiB | r2  | 192 | 208,785           | 0.7912              | 152.6 / 11,019                 | 312.0 | 27.0       | 953             | 15 / 15            | l0 / l0                |
| 128 MiB | r3  | 192 | 171,779           | 0.8273              | 134.7 / 10,275                 | 307.9 | 24.0       | 664             | 30 / 30            | l0 / l0                |

Medians, 128 MiB against 1 GiB: tok/s 307.9 against 291.2 (+5.7 %); later-turn TTFT p99 11.0 against 17.3 s; later-turn TTFT p50 147 against 131 ms; `cached_tokens_ratio` 0.791 against 0.840 (−4.9 pp); recomputed tokens 208,785 against 160,262 (+30 %). All runs 192/192 ok.

- No mixing in the lab either way: in every run L1 ends 100 % l0 (146 blocks at 1 GiB, 144 at 128 MiB — the 2-block difference is the slack), because the tier never frees a whole slab under load and `take_slot` only reformats an empty slab. `no_room` for fp8 demotions into L1 is similar in both arms (3–30), and `no_room_backoff` fires exactly once per run in both. The mixing gain of C is real per the unit test but this workload never reaches the state that triggers it.
- Throughput and tail: 128 MiB is not worse — median tok/s +5.7 %, p99 TTFT 11.0 against 17.3 s (one overlapping run) — but retention medians moved the other way (cached −4.9 pp, recompute +30 %), with arms inside the historical 1 GiB spread (ladderperf on-arm: recompute 148k–193k, cached 0.809–0.852, tok/s 311–329). Run-to-run ladder dynamics, not slab arithmetic, dominate at this workload; 3 runs per arm cannot separate them. No second size adjustment is warranted by these numbers; 128 MiB stays the default (finer reformat granularity at equal slack, better tail, host AC green).
- L2 untouched (the decision named L1 only): nothing here argues for following — L2's larger slabs hold more blocks per slab and showed no no_room increase.
- Transient (flagged for the lead): the first 1 GiB run crashed mid-bench with `engine thread panicked: release of unreferenced KV block BlockId(146)` (fatal exit 3, bench 153/39) — not reproduced in 6 subsequent full runs on either arm (0 panics in all server logs); one-off on the pre-merge p6b-stack ladder path.
- Labbook set `phase-6b-kv-compression`, runs `p6b-slabs:mt3:llama:slab{1g,128}:r{1,2,3}`.

### The lossless tail in the lossy-tier capacity (branch `p6b-tq4enc`, 4848c98)

The t17 quick tier failed `turbine-server --test kv_gpu lossy_tier_reuse{,_tq4}` ("L1 holds
0.83 x full size: not encoded"). The lab dump (job turbine-lab-test-1003223847-3666bf94, log
`target/tq4enc/diag-lossy.log`) shows the tq4 demotion encoding is exact and unregressed: of the
42 blocks 29 one-shot fillers demoted, 28 were **stale lossless-tail blocks** — every finished
sequence tags its last full block, and a finished sequence cannot grow out of the tag — stored
raw at the L0 format per spec S-2 (user decision 2026-09-28, Q13). What changed is the eviction
order: 028f465 (user decision 2026-10-02, "6b: OLMoE tq4 — lossless last block in eviction
order", 1 A) prices a tail block's retrieval like its history's, so under `cost_aware` the tails
(memory term still 3.56× a tq4 slot) now demote **first**: 28 stale tails + 14 encoded blocks.
The tq4 lower-tier `supported` evidence is untouched (all 14 non-tail copies encoded at exactly
4,128,768 B; the codec is unchanged since 2c3c1f0).

Capacity consequence on one-shot-heavy workloads: up to one raw 14 MiB L1 slot per finished
sequence (3.56× a tq4 slot), and with the ladder the L0 sweep skips tail-tagged blocks forever —
consistent with the mt3 ladder-on runs ending "almost every stored copy still l0". Fix options
A (tail tags expire with the latest finished sequence), B (keep, document) and C (per-session
tails) are in `.procoder/handoff/p6b-tq4enc.md`; recommendation A.

The two lab tests now run with `kv.lossless_tail_blocks=0` — they predate the tail rule and
never meant to exercise it; the tail's behaviour stays pinned by
`document_lists_copies_per_codec` and `last_block_is_scored_like_its_history`. Verified on the
lab after the change (labbook `p6b-tq4enc:kv_gpu:*`): the fp8 arm passes end to end (L1 = 42 x
7,340,032 exactly, worst first-8 |delta logprob| 0.050) and the tq4 arm's byte assertion passes
(L1 = 42 x 4,128,768 exactly) — the encoding is exact — while its accuracy head bound misses by
0.002 (worst first-8 0.2518 against the t9-calibrated 0.25; with the tail exempted, all of A's
cached blocks are tq4, where at t9 its lossless tail block was still exact). The tq4 tier's
bound is a gate decision: options and the recommendation are in
`.procoder/handoff/p6b-tq4enc.md`.

### Release-path audit after the slab-run crash transient (branch `p6b-crash`)

Investigation of the "release of unreferenced KV block BlockId(146)" transient above (fatal exit 3, one-off, pre-t17-merge binaries; the crashed run's server log was overwritten by the reruns, and it had no DEBUG events, so no stack or event trail exists). No repro: an engine-level stress on the cpu backend driving the crashed run's shape — 12 seeds × 6 multi-turn sessions over one shared prefix, YELLOW from 1 % utilisation so queued prefixes release and attach again every turn (queued-prefix demotion, `reattach`), ladder on over L2, promotions, copy-ahead, one stream dropped mid-run — runs clean (`engine::r#loop::tests::multi_turn_release_paths_stress_without_an_unreferenced_release`, 9 s; asserts the engine never panics and L0 ends empty). The audit of every `pool.release` site found and fixed two reference-discipline defects; neither is proven to be the transient, and the pool assertion stays as the safety net:

- A promotion completing after a sibling's failed copy kept its target block referenced by nobody (the failure resolves the attach, so the completion found no pending entry and released nothing): a permanent per-page leak under copy failures (`on_copy_done` now releases the block when the owner's pending entry is gone). Red first: `hierarchy::tests::a_promotion_completing_after_a_failed_sibling_releases_its_block` (3 blocks leaked, mutation-checked).
- An L0 rewrite whose block's copy gains a holder (an attach) or is replaced (a recompute re-publishes a new L0 page) while it runs took over anyway: it filed the rewritten page and silently dropped the referenced source's location — the source page stranded outside the directory (`evict_cached` no-ops on a referenced page), or a live page released. Dormant today (the recent window defaults off and `kv.ladder.l0` is refused at startup), live the moment either switches on. `on_copy_done` now completes a rewrite only when the directory's L0 location is still the page it rewrote and that page is unreferenced; the rewritten page goes back otherwise. The old page's `l0_keys` mapping now goes with the page (a stale entry let the mirror attribute a new holder's refcount to the recompressed copy — masked in tests by HashMap order, so its dedicated mutation check is inconclusive; noted in the handoff). Red first: `hierarchy::tests::an_l0_rewrite_over_an_attached_copy_keeps_the_old_page_serving` (attach served the rewrite's page), `a_rewrite_completions_old_page_forgets_only_itself`.

No lab run: the fixes are host-side refcount logic covered on the cpu backend (engine stress, kv_sim, hierarchy units); the transient itself did not reproduce in 6 lab runs before this branch. A ladder-on mt3 ×3 watch is worthwhile at the next lab visit, and before `kv.recent_window_blocks` defaults on.

### The BF16 recent window on the GPU (Task 17 follow-up; branch `p6b-window`)

Per-class page addressing through the ABI v2.11 descriptors (`page_classes` + the pool's slab
constants on the paged-attention and copy-blocks descriptors; NULL keeps the flat addressing
byte-identical), so the recent window's BF16 pages — larger than a TurboQuant base page — are
addressable. `kv.recent_window_blocks` defaults to 1.

- Lab kernels (`hip_ops::paged_mixed_classed_matches_cpu`, job
  turbine-lab-test-1004004436-387b5582): a tq4 base pool with BF16 class pages (3.5× a base
  page) appends byte for byte identically on HIP and CPU, and the staged prefill rows match
  the CPU provider within the BF16 tolerance (max |Δ| 1.56e-2). A decode-only probe over the
  same pool state diverged from the CPU provider (hip 0.0 vs cpu NaN on one q head, synthetic
  constant q row re-appended into a written slot) and is left OPEN for Task 18; the served
  decode below is the proof that matters.
- Served (`lab-bench --quick --model llama -- --set kv.dtype=tq4`, GPU 0, window default 1,
  run `target/lab-bench/9e1180e-llama/`): golden c1 **15/16 prompts with full 32-token
  identical prefixes** (Task 13's tq4 gate was 0/16); p09 diverges at token 6 (margin 0.113)
  and p10 at token 27. The c1 verdict is still FAIL by the tolerance's letter — the strict
  every-prompt logprob bound does not hold on all 16 — which is Task 18's gate call
  (golden c16 + eval), not this branch's. Quick throughput 853.1 tok/s, ITL p50 15.7 ms,
  TTFT p50 248 ms, decode_fwd 15.4 ms, 64/64 ok.

### The L0 ladder on the server (Task 18; branches `p6b-t18`, stack tip 8be6dccd)

Task 16's startup refusal of `kv.ladder.l0` is lifted (6542ff14): the lossy rung classes are
read through the same ABI v2.11 per-class addressing as TurboQuant L0 pages, so
`kv_format_availability` refuses it only on a GPU library without the v2.11 group. Two serve
bugs were found and fixed on the way to the first run: the v2.11 mixed-format attention
descriptor always carries the layer's TurboQuant tables, which a classed pool without
TurboQuant pages had nowhere to take from (0338bbb8, then 6152d4fb: a classed pool — the
recent window or the ladder's L0 rungs — now carries its tables in the model `kv_cache` under
the pool's own namespace seed, and `reserved_bytes` counts the upload); and the OPEN decode
probe of `p6b-window` root-caused as a harness artifact (94bd16aa): the classed test geometry
referenced base blocks whose bytes a carved slab also holds — the real pool retires those
base ids before carving (`convertible_slabs`) — and the two kernels only disagree on that
garbage (CPU NaN against HIP 0). With a valid geometry the classed decode matches the CPU
provider on the R9700 (`paged_mixed_classed_matches_cpu` decode leg, job
turbine-lab-test-1004034318) and a host test pins both legs
(`classed_pool_prefill_and_decode_stay_finite`).

Multi-turn A/B, Llama-3.2-3B, `scripts/lab/phase6-novanas-ladder.yaml` (L0 8 GiB, L1 2 GiB
`l0`, L2 4 GiB `l0`, `max_format: tq4`) with `--set kv.ladder.l0=true` against
`--set kv.ladder.l0=false` on the same budget (the ladder on in both arms). Tree 94bd16aa. A
fresh native server per run on GPU 0 (port 18000, `bench.lock`, fixtures paused), arms
interleaved. Workload: `turbine-bench --profile multi-turn --sessions 24 --turns 8
--concurrency 24 --shared-prefix-words 2000 --think-time 1..4 --session-hints`, client on
novanas; golden c1 ran on each run's server before the bench. Every run served 192/192 and
golden c1 was 16/16 PASS on both arms — the L0 ladder's rewritten pages cost no golden prompt.

| Arm | Run | recomputed tokens | cached_tokens_ratio | later-turn TTFT p50 / p99 (ms) | tok/s | L0 rewrites (no_room backoffs) | ORANGE samples |
| --- | --- | ----------------- | ------------------- | ------------------------------ | ----- | ------------------------------ | -------------- |
| on  | r1  | 121,476           | 0.8812              | 202.8 / 44,333                 | 172.7 | 56 (37)                        | 192            |
| on  | r2  | 147,444           | 0.8602              | 193.1 / 35,092                 | 233.0 | 61 (34)                        | 131            |
| on  | r3  | 180,692           | 0.8210              | 154.9 / 11,791                 | 302.2 | 0                              | 47             |
| off | r1  | 192,115           | 0.8119              | 178.6 / 14,443                 | 315.5 |                                | 47             |
| off | r2  | 164,978           | 0.8370              | 171.8 / 10,454                 | 300.6 |                                | 50             |
| off | r3  | 224,002           | 0.7793              | 157.4 / 12,411                 | 304.2 |                                | 51             |

Medians, L0 on against off: recomputed tokens 147,444 against 192,115 (−23 %);
`cached_tokens_ratio` 0.8602 against 0.8370; later-turn TTFT p50 193.1 against 171.8 ms,
p99 35.1 against 12.4 s; output tok/s 233.0 against 304.2 (−23 %); ORANGE 131 against 50
samples of 0.5 s. The L0 rung compressed in two of three on-runs (56 and 61 rewrites at
`fill_high_water`, each with 34–37 `no_room` back-offs where a class could not grow); in r3
the L1/L2 rungs absorbed the pressure and the L0 rung stayed at the base format. The
ladder-on tail and throughput regression of Task 16 (open point in
`.procoder/handoff/p6b-t16.md`) is unchanged in shape: the on-arm sits at ORANGE far longer
and loses tail and throughput against the off-arm.

Golden c16 with the L0 ladder on (`lab-bench --golden16 --model llama -- --set
kv.ladder.enabled=true --set kv.ladder.l0=true`, run `target/lab-bench/p6b-t18-l0ladder-llama/`):
**golden c1 and c16 both PASS** — 16/16 prompts at the full 32-token identical prefix (c1
strict bounds, max likely diff 0.0401; c16 batched), 200/200 bench requests, 781.1 tok/s,
ITL p50 16.9 ms, TTFT p50 226 ms, decode_fwd 16.5 ms.

Shared-prefix GSM8K, spec S-8's ladder variant with the L0 step (the Task 16 recipe: both
sides `--set kv.gpu.max_bytes=4GiB --set kv.cpu.max_bytes=4GiB --set kv.nvme.enabled=false`,
the candidate `kv.ladder.l0=true`, the baseline `kv.ladder.enabled=false`; c16, 32 fillers 16
at a time, `--fillers-settle`; fresh server per run, arms interleaved). Every run served
exactly its eval: `turbine_requests_total` 232, `filler_retries` 0; every candidate's lossy
cached ratio 0.927 (`--min-lossy-cached-ratio 0.5`), cached ratio 0.931 on both sides.

| Arm          | Run | accuracy | lossy cached ratio |
| ------------ | --- | -------- | ------------------ |
| ladder l0 on | r1  | 0.795    | 0.927              |
| ladder l0 on | r2  | 0.785    | 0.927              |
| ladder l0 on | r3  | 0.765    | 0.927              |
| ladder off   | r1  | 0.755    | 0.000              |
| ladder off   | r2  | 0.765    | 0.000              |
| ladder off   | r3  | 0.765    | 0.000              |

Nine candidate × baseline pairs (`scripts/eval/paired_compare.py --max-drop 0.01`): median
drop **−0.02**, lowest exact McNemar p 0.0768 — the gate passes with the L0 ladder on, the
candidates never below their baselines
(`tests/eval/llama-3.2-3b-instruct/turbine-ladder-l0-sp-r{1,2,3}.json` + `-paired.json`, nine
pairs in `turbine-ladder-l0-sp-paired.json`). Labbook set `phase-6b-kv-compression`, runs
`p6b-t18:mt:llama:l0-{on,off}:r{1,2,3}` and `p6b-t18:l0ladder:golden:{c1,c16}`.

Soak (S-7 AC), 10-minute overload with the L0 ladder on:
`scripts/overload-soak.sh novanas --duration 10m --shared-prefix-share 0.5 --set
kv.cpu.max_bytes=4GiB --set kv.nvme.max_bytes=16GiB --set kv.ladder.enabled=true --set
kv.ladder.l0=true --set kv.ladder.max_format=tq4` (serve run 1004035400-0fa35b94,
`target/soak/novanas-20261004T035400Z`). **Verdict PASS, all 8 checks true**: ITL p99 220.4 ms
against a calibration of 187.7 ms, GREEN 0 s into the cool-down, 4,665 × 200, 2,886
`queue_timeout`, 0 client-dropped. `turbine_kv_ladder_actions_total{tier="l0"}` 1,148 —
712 `l0 → fp8_e4m3` rewrites at `fill_high_water`, 435 `no_room_backoff`, 1 `rung_step_up` —
so the S-7 soak criterion (above 0) is met; every tier's ladder actions sum to 6,790.

### Stale tail tags expire, the tq4 lab bound splits like golden (decisions "6b: stale lossless-tail tags and the tq4 lab bound" 1 A, 2; branch `p6b-smallfix`)

Both follow-ups from the tq4enc entry above, host-only (no lab run of this branch):

- **Tail-tag expiry (1 A).** `KvHierarchy.tail` now holds only the latest finished sequence's
  last `kv.lossless_tail_blocks` full blocks: a later finish replaces the set (`tail.clear()`
  before re-tagging). The one-raw-L1-slot-per-finished-sequence cost on one-shot-heavy
  workloads and the ladder's permanent L0-sweep skip on those blocks are gone; an expired tail
  demotes, evicts and ladders like any other block, while a growing sequence keeps the S-2
  guarantee (its newest demoted block stays exact until the turn that grows it finishes). The
  set itself is bounded at N. Pinned host-side:
  `a_later_finish_expires_the_previous_sequence_tail` (expired tail demotes encoded, latest
  tail raw; mutation: dropping the `clear()` fails it) and
  `a_growing_sequence_keeps_its_tail_exact_until_the_turn_finishes`; kv_sim
  `per_tier_formats` re-pinned (at most one raw lower-tier copy, at least one expired tail
  encoded; the fp8/tq4 capacity factors unchanged). Both ladder ACs
  (`ladder_under_pinned_pressure`, `ladder_l0_under_pinned_pressure`) pass without a fixture
  re-bless.
- **The tq4 lab bound (2).** `lossy_tier_reuse_tq4`'s head bound was the flat 0.25 — golden's
  _likely_ bound, calibrated at t9 against a run whose lossless-tail block was still exact. It
  now applies the gate's actual likely/tail split per position (0.25 above the
  `likely_logprob_floor` −2, else the tail bound 0.75), the same rule
  `turbine-golden compare` uses; nothing else relaxed (the 90 %-within-0.75 share and the FP8
  arm's tighter 0.3 / 0.5 / 0.9 stay). The all-lossy worst case that measured 0.2518 sits
  within the split rule as it does within golden at c16. The split logic is unit-pinned
  host-side (`golden_head_bound_splits_likely_from_tail`); the lab test itself is verified at
  its next lab pass (the phase-exit pass takes it).

### Phase 6b summary (exit, 2026-10-04; stack tip `af9b82af` + `p6b-smallfix` (tail-tag expiry, tq4 bound split))

What the phase shipped:

- **Per-tier formats.** L1/L2 store blocks in their own codec. Lower-tier `fp8_e4m3` (Task 6: shared-prefix eval drop 0.005, kv_gpu green) and `tq4` (Task 9 + the promotion-path and lossless-last-block fixes: OLMoE 0.8625 vs `l0` 0.8624, Llama 0.9075 vs 0.9007) are `supported`; `tq2` failed the eval on both models (drop 0.060 / 0.025) and stays `experimental` as a capacity rung. The fixed c16 bench is unchanged by any tier format (Llama 849.6–850.3 tok/s against the 854.2 6a-exit baseline; the prompts share no full block, so no lossy block is read).
- **TurboQuant.** `tq4` in L1 holds 2.96× the blocks of `l0` per GiB measured; as the L0 format it holds 3.56× (targets ≥ 3.5× / 6× met on `kv_sim` and in `/turbine/v1/kv`). L0 `tq4` is `experimental` (Task 13 failed its golden and OLMoE eval gates), reached in production only through the ladder's L0 rung; L0 `tq2` is refused (`kv_tq2_l0_refused`, GSM8K 0.15–0.20). The BF16 recent window (Task 17, per-class page addressing) restored the newest-block exactness that made the ladder's L0 rung servable.
- **The compression ladder** (Tasks 14–16, L0 step Task 18): opt-in (`kv.ladder.enabled`, default false), rung order `l0` → `fp8_e4m3` → `tq4` (`max_format`, `tq2` failed its gate) → evict, from YELLOW on and under capacity pressure, with back-off on a tier without room, rung-slot fallback so demotions keep flowing, and GREEN-only step-up after the dwell. Multi-turn A/B with the L0 step: recomputed tokens −23 %, cached ratio 0.860 vs 0.837, at tok/s 233 vs 304 and later-turn TTFT p99 35.1 vs 12.4 s — the documented trade accepted by user decision 2026-10-04 A. Both shared-prefix eval gates pass with the ladder on (Task 16 median drop 0.010 at the bound; Task 18 median −0.02), and both 10-minute soaks pass with ladder actions well above 0 (Task 18 L0: 1,148 actions, 712 rewrites).
- **Serving fixes the phase forced:** planner copy timing (poll-bounded `CopyTime::Within`, clamped to calibrated costs), pinned D2H batching, the promotion copy kernel (ABI v2.11 H2D kernel, default; ~2× the copy engine under decode compute), queued-prefix demotion, copy ahead, the GREEN admission headroom cap at RED's 0.90 (no GREEN → SURVIVAL burst; the soak's rejections move to `queue_timeout`), the step-time drift window fixes (age-out + minimum judged steps; steps overlapping a KV copy are not judged), the lossless-last-block eviction score, expiring tail tags (`p6b-smallfix`), and 128 MiB L1 slabs. Kernel ABI v2.11 (unshipped before 6b): KV transcode + TurboQuant transcode, the mixed-format paged attention (own `turbine_hip_mixed` decode, staged CK prefill), per-class page addressing, and the copy kernel — each with a recorded provider evaluation.

Open at exit: the ladder-on tail/throughput regression keeps its shape with the L0 step (the on-arm sits at ORANGE far longer; follow-up, not a blocker — decision 2026-10-04 A); OLMoE `l0`'s L1 `rung_no_slot` churn on the ladder workload is covered by the 128 MiB slab decision but not re-measured end to end; `/turbine/v1/status` does not yet carry `quantization.tier_formats` / `quantization.ladder` (S-10, carried from Task 15 — see the plan's Task 19 report). Labbook set `phase-6b-kv-compression` holds every run cited above.

Exit runs (Task 19, 2026-10-04, merge `29f4de10` = `af9b82af` + `p6b-smallfix`; logs under `target/t19/` on the workstation):

- `scripts/gate.sh --full`: **ok** on `af9b82af` (955 tests) and again on the merge (958) — `gate: ok crates=all failed=0`.
- `scripts/lab-test.sh novanas --tier full` (job turbine-lab-test-1004090558): 1,051 passed, **2 failed** — `hip_ops::implementations_enumerated` and `hip_ops::every_implementation_matches_cpu`, both full-tier-only exhaustive checks whose expected implementation tables predate Task 12's mixed paged attention (the library enumerates 7 `attention_prefill_paged` implementations, the test expects 4; the exhaustive-match test has no mixed scenario, so the mixed implementations "never ran"). The quick tier skips both, so the full tier is the first run to see it. Everything else green, including `kv_gpu lossy_tier_reuse{,_tq4}` (the smallfix bound split holds end to end on the R9700), both ladder kv_sim traces, and `paged_mixed_matches_cpu`.
- `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` (`TURBINE_LAB_ONE_GPU_JOB=0`; job turbine-lab-test-1004103638, after the host reboot the lead ran this morning — GPU 1 had dropped after the previous boot): **PASS, 27 passed / 0 failed** — the Phase 0 inventory, the Phase 5 two-GPU lists (hostmem, tp2, rccl) and the fault-injection suite.
- `scripts/lab-bench.sh --golden16 --label t19-exit` for the 9 models: **PASS** for `llama` (853.7 tok/s, 1.00× the 6a-exit baseline), `olmoe` (610.3), `llama-fp8` (977.3), `llama-fp8-block` (1059.7), `llama-fp8-tensor` (976.5), `llama-awq` (1258.1), `llama-gptq-autoround` (1265.9). **FAIL for `llama-fp8kv`**, deterministically (two runs, bit-identical verdicts): golden c1 15/16 (need 14 — every prompt's token rule holds; p10 exceeds the strict likely bound 0.4274 vs 0.40, tail 0.8230 vs 2.44, identical 32-token prefix), c16 the same. The 6a exit passed this gate at 829.7 tok/s (2026-09-30), so a 6b commit moved the fp8-KV paged path's numerics just past the slug's likely bound — not bisected here. **`olmoe-fp8kv` FAIL 13/16**, exactly the 6a-exit result behind its `experimental` demotion (user decision 2026-09-30 B) — unchanged, not a new miss.
- 10-minute ladder soak × 2 (`--shared-prefix-share 0.5`, L1 4 GiB, L2 16 GiB, `kv.ladder.enabled` + `l0` + `max_format=tq4`): both runs **fail exactly one check, `kv_idle`** — at the final scrape L0 still holds 1,556 / 1,291 blocks and L1 134 / 108, where the Task 18 soak on tree 94bd16aa (same flags) drained to idle and passed 8/8. Every other check is true in both runs (ITL p99 217 ms against calibrations 186 / 183, GREEN 0 s into the cool-down, `streams_complete`, 0 client-dropped, ladder actions 984 / 1,666 with the L0 share above 0). The only code change between the passing and failing trees is `p6b-smallfix`'s tail-tag expiry (`ecf29422`) — suspect recorded for the lead, not diagnosed here.
- Support matrix and track gates: the full `--support-matrix --output json` is on the workstation (`target/t19/support-matrix.json`); `--check-config` with `kv.dtype=tq4 --kv.cpu.format=tq4 --kv.ladder.enabled=true` prints `support: experimental (amd/*/*/bf16/tq4/none)`, `config ok`, and `kv.cpu.format: zstd` exits 2 naming the key. `scripts/track-gate.sh phase-6b-kv-compression` → **GATE PASS**; `phase-5p-serving-efficiency` is not an argument the script accepts (it gates the phase 6–8 tracks; 5p's order evidence is 6b's close); `phase-7-model-families` → GATE FAIL "no supported amd row with tq4 or tq2 KV" — the umbrella's phase-7 start rule cannot pass after 6b's accepted end state (the L0 `tq4` rows stay `experimental`, user decisions 2026-10-02 / 2026-10-04 A) and needs amending before track 2.

The llama-fp8kv golden miss root-caused: the recent window on the fp8 base (2026-10-04, branch
`p6b-fp8kvbisect`, commit `70f22e92`, from the exit merge `5bec8eea`; logs under `target/fp8kv/`
on the workstation). The window's eligibility predicate was `KvDtypeChoice::is_lossy()` — true
for `fp8_e4m3` — so with the default `kv.recent_window_blocks: 1` an fp8_e4m3 L0 pool went
classed and every sequence's newest full block (for a short prompt, its only, partial block) was
created in the BF16 page class and served unquantized, shifting the served numerics away from
the append-time quantize the FP8-KV golden reference emulates. p10 is a ~50-token chat prompt, so
its whole sequence sat on a BF16 window page — the deterministic likely-bound miss (0.4274 vs
0.40) with identical 32-token prefixes, no bisect needed. Fix: the window applies to TurboQuant
bases only (`bf16` and `fp8_e4m3` are lossless-with-matched-scales reference representations,
user decision 2026-10-04 A); `fp8_e4m3` pools are flat again and serve the 6a bytes. Evidence:
new cpu test `tiny_server::recent_window_skips_lossless_fp8kv_base` red pre-fix, green post-fix
(mutation-checked); `gate.sh` ok (959 tests); `scripts/lab-bench.sh --model llama-fp8kv --label
fp8kvfix --golden16` on novanas GPU 0 → **golden c1 PASS 16/16** (strict bounds) **and c16 PASS
16/16** (batched bounds), bench 838.8 tok/s (6a exit: 829.7), ITL p50 15.6 ms, 200/200 ok — the
6a Task 24 tolerance (0.40 / 2.44) unchanged, no recalibration. Recorded in labbook
(`67de6c5f-7579-4c47-a1f1-f3dd6da70e98`).

### Exit fixes (`p6b-exitfix`, 2026-10-04; branch from `p6b-stack` `5bec8eea`)

The four exit findings of `.procoder/handoff/p6b-exit.md`, host-only plus two lab-test runs (logs `target/exitfix/` on the workstation; no serve, bench or lock of this branch beyond the lab-test Jobs' shared bench-lock hold):

- **Soak `kv_idle` root-caused — a reporting defect, not a leak (commit `fix(kv): a cached classed page is not a referenced block`).** The residual L0 blocks of both failing soaks are the shared prefixes' cached copies, which the cost-aware policy legitimately retains and `kv_idle` (the pressure document's kv pool) does not judge: the pool metrics' 1,556 / 1,291 used slots are cache. The check actually reads `EngineLoop::sync_kv_held` — `referenced_blocks() × block_bytes` — and `BlockPool::referenced_blocks` subtracted only the base class's cached count, so every cached unreferenced page of a non-base class counted as a holder. The evidence is exact: the pressure document's kv `used_bytes` was 1,512,046,592 / 1,130,364,928 = 103.0 / 77.0 whole l0 blocks (14,680,064 B) of ladder-rewritten (fp8-class) copies that survived as cache — 288 / 1,038 `fill_high_water` L0 rewrites whose rest the capacity churn had evicted — constant through the whole 5-minute cool-down, while `reserved` was 0 and no live request existed. Run-to-run dynamics decide how many rewritten copies survive to the end, which is why Task 18's soak drained and these did not; the tail-tag expiry changed the mix (expired tails no longer demote first as raw copies), not the accounting. Fix: `referenced_blocks() = used_blocks() - cached_unreferenced()` (the base class and the page classes alike). Pinned by `pool::tests::classed_cached_pages_are_not_referenced` and by `kv_sim recent_window_holds_newest_blocks_at_bf16`, which now asserts an idle end state with a cached BF16 window page reports 0 referenced — both red at the old subtraction, the kv_sim one reproducing the soak state on the cpu backend. The criterion stays as written; a GPU soak rerun is the lead's re-verification.
- **S-10 status keys (commit `feat(server)`):** `quantization.tier_formats` — every local tier's copies per codec (blocks, encoded bytes) of the kv document — and `quantization.ladder` — `KvHierarchy::ladder_document` (the resolved `kv.ladder` config, each tier's current rung) — under `/turbine/v1/status`, replica 0's published engine documents; `kernels` names the ABI v2.11 `kv_transcode` a lossy tier format selects (`reason_code: tier_format`, the provider's own implementation for the encode direction over the model's layout — appended beside the registry's selections, not a selection). Pinned by `tiny_server status_reports_tier_formats_and_ladder` (cpu backend, FP8 L0, L1 `tq4`, ladder on; mutation: dropping the keys fails it). Found on the way: a BF16-base ladder builds an fp8 class page whose per-layer size carries the codec's scales header (`header_bytes`), which the cpu provider's class sanity check (`cpu/paged.rs`) refuses at warm-up — the gpu path never checks it; recorded for the lead, the AC test uses an FP8 L0 base whose classes are exact.
- **Stale hip_ops full-tier tests (commit `test(kernels)`):** `implementations_enumerated` now pins the library's actual enumeration — 7 `attention_prefill_paged` (the BF16 and FP8 paths plus Task 12's `ck_tile_fmha_pagedkv_mixed_staged`, `turbine_hip_mixed_staged`, `turbine_hip_mixed`), 6 `attention_decode_paged` (`turbine_hip_mixed`), 2 `kv_transcode` (`turbine_hip_fp8`, `turbine_hip_tq`) — read from `kernels/rocm/src/impl_table.cpp`; `every_implementation_matches_cpu` gained a TurboQuant-page mixed scenario per paged kind (only the mixed implementations support it; its case bodies take the bound implementation's name to pick the staged-vs-rotated judgement `paged_mixed_matches_cpu` uses). Lab: both green (job turbine-lab-test-1004135524-16c0c1b2); mutation (dropping `turbine_hip_mixed_staged` from the expected table) red (job turbine-lab-test-1004135817-1627eab1).
- **The phase-7 track-gate rule (commit `fix(scripts)`):** the order check accepts 6b's accepted end state — any amd row with a `tq4`/`tq2` KV column in `supported` or `experimental` state — and its failure message names exactly that (`tq2` is an `experimental` lower-tier rung with no amd matrix row; L0 `tq2` refused; a refused `tq4` row still fails; the phase-7 spec owns any flip of the L0 rows to `supported`). `lab_scripts track_gate` feeds it the experimental row (passes), a refused one (fails) and a supported one (a later flip passes); mutation (back to supported-only) red. The umbrella plan's state paragraph records the amendment; `track-gate.sh phase-7-model-families` passes on the real tree shape the test mirrors.

Gate after each commit: `scripts/gate.sh --base 5bec8eea` ok. No throughput or numerics change is claimed by this branch: the pool-accounting fix only corrects what the pressure document reports at idle (`kv_utilization` no longer holds the phantom classed-cache share), the status keys are reporting only, and the test and script changes run nothing in the serving path.

### kv_idle verification after the used_bytes fix (2026-10-04)

`overload-soak.sh novanas --duration 10m --shared-prefix-share 0.5 --set kv.cpu.max_bytes=4GiB --set kv.nvme.max_bytes=16GiB --set kv.ladder.enabled=true --set kv.ladder.l0=true --set kv.ladder.max_format=tq4` on the merged stack (84c22878 + the fix): **rc=0, `kv_idle: true`, `streams_complete: true`** — the exit's only failing check is cleared. Ladder actions > 0 in the run log.
