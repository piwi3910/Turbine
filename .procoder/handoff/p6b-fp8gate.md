# Handoff: p6b-fp8gate (FP8 lower-tier gate, user decision A: 3 + 3 runs, medians plus McNemar)

Paused by the lead after the BF16 half. Branch `p6b-fp8gate` from `p6b-stack` f9a4c91. No Rust change, nothing flipped.

## Done

Recipe of `p6b-eval-prefix.md` (32 fillers, c16, `phase6-novanas-llama.yaml --set kv.gpu.max_bytes=4GiB --set kv.cpu.max_bytes=4GiB
--set kv.cpu.format=l0`, `--min-cached-ratio 0.5`), a fresh server per run, GPU 0 under the bench lock, on the f9a4c91 tree. Each
server's `turbine_kv_prompt_tokens_total` was 726,723, so it served only its eval.

| Run | Report | Serve run id | Accuracy |
| --- | ------ | ------------ | -------- |
| BF16 r1 | `tests/eval/llama-3.2-3b-instruct/turbine-bf16-sp-r1.json` | 1001131143-01812869 | 0.780 (156) |
| BF16 r2 | `turbine-bf16-sp-r2.json` | 1001131324-3eaf9425 | 0.785 (157) |
| BF16 r3 | `turbine-bf16-sp-r3.json` | 1001131447-036a8926 | 0.770 (154) |

BF16 median 0.780. The earlier `turbine-bf16-sp.json` (0.780, a8c3b7c) stays as it was.

## Left

1. Three FP8-L1 runs (`--set kv.cpu.format=fp8_e4m3`, `--min-lossy-cached-ratio 0.5`, the same eval command), a fresh server each. The
   three fp8 attempts of the first batch never started: `lab-serve.sh --stop <id>` returns before the pod is gone, so the next start
   saw port 18000 answering and refused. Wait until `/ready` stops answering (plus a few seconds) between runs.
2. Judge: median drop at most 0.01 and the exact McNemar test on the median runs (and the nine cross pairs) at the 6a alpha. No tooling
   exists in the repo; the calculation is an exact two-sided binomial on the discordant items (as in the 6a decisions).
3. `kv_gpu` (all 10 must pass; the disk now has room), then, if both are green, remove the `fp8_e4m3` entry from
   `TIER_FORMAT_REFUSALS` in `crates/turbine-core/src/support.rs` (an unlisted format is `supported`) with the evidence in its comment,
   update `support::tests` and `support_startup.rs` (the `tier_formats` test expects `Experimental` for `fp8_e4m3`), then the perf log
   6b section and the AGENTS.md support line.

The release `turbine-golden` is built at `/home/piwi/turbine-ci/remote/agent-p6b-fp8gate/target/release/turbine-golden`.
No serve Job or lab Job of this branch remains.
