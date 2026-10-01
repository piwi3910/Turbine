# Handoff: p6b-fp8gate (FP8 lower-tier gate, user decision A: 3 + 3 runs, medians plus McNemar)

Paused by the lead after the BF16 half. Branch `p6b-fp8gate` from `p6b-stack` f9a4c91. No Rust change, nothing flipped.

## Done

Recipe of `p6b-eval-prefix.md` (32 fillers, c16, `phase6-novanas-llama.yaml --set kv.gpu.max_bytes=4GiB --set kv.cpu.max_bytes=4GiB
--set kv.cpu.format=l0`, `--min-cached-ratio 0.5`), a fresh server per run, GPU 0 under the bench lock, on the f9a4c91 tree. Each
server's `turbine_kv_prompt_tokens_total` was 726,723, so it served only its eval.

| Run     | Report                                                     | Serve run id        | Accuracy    |
| ------- | ---------------------------------------------------------- | ------------------- | ----------- |
| BF16 r1 | `tests/eval/llama-3.2-3b-instruct/turbine-bf16-sp-r1.json` | 1001131143-01812869 | 0.780 (156) |
| BF16 r2 | `turbine-bf16-sp-r2.json`                                  | 1001131324-3eaf9425 | 0.785 (157) |
| BF16 r3 | `turbine-bf16-sp-r3.json`                                  | 1001131447-036a8926 | 0.770 (154) |

BF16 median 0.780. The earlier `turbine-bf16-sp.json` (0.780, a8c3b7c) stays as it was.

## Finished (2026-10-01)

Three FP8-L1 runs on the same tree and recipe (`turbine-l1-fp8-sp-r1..r3.json`, serve runs 1001135450-3cc7df5a, 1001135611-001242f8,
1001135732-13b1af29): 0.775 / 0.775 / 0.770, each server at 726,723 prompt tokens, lossy cached ratio 0.927, guard passed. Median drop
0.005 (BF16 median 0.780); exact McNemar on all nine pairs p 0.55 to 1.0. `kv_gpu` job 1001135903-1ec4bf89: 12 passed, 0 failed. Lower-tier
`fp8_e4m3` flipped to `supported` (`TIER_FORMAT_REFUSALS` entry removed, `support::tests`, `support_startup` test, AGENTS.md line, perf
log 6b section). Nothing left; no lab Job of this branch remains.
