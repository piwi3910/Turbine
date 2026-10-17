# Phase 7 Task 10 exit-runbook reports

Track-close reports for `phase-7-model-families`, produced by Task 39 on the integration tip
`212ae15e` (branch `p7-t39`) under the umbrella runbook
`.procoder/plans/phase-6-8-expansion.md` Task 10. Every closed `turbine_core::support::PHASE7_ROWS`
entry was served from its lab config and re-run here; the per-run narrative is in
`.procoder/perf-log.md` ("P7 exit").

Per slug: `novanas-<slug>-turbine.json` is the raw `turbine-bench --output json` throughput report
(the runbook's bench artifact), `...-golden1.txt` / `...-golden16.txt` are the `turbine-golden
compare` verdicts, `...-status.json` is `/turbine/v1/status` and `...-kv.json` `/turbine/v1/kv`
after the run. The two-card rows' reports (`...-tp2-*`, `...-ep2-*`) come from
`scripts/lab-cluster.sh --bench-lock (qwen3moe-tp2|qwen3moe-ep2)-novanas`; their `...-gsm8k.json`
is the served GSM8K-200 eval.

| row (`PHASE7_ROWS` id) | artifacts |
| --- | --- |
| `Qwen3ForCausalLM/bf16/bf16/tp1` | `novanas-qwen3-4b-instruct-2507-*` |
| `Qwen3MoeForCausalLM/fp8_block/bf16/tp2` | `novanas-qwen3-30b-a3b-instruct-2507-fp8-tp2-*` |
| `Qwen3MoeForCausalLM/fp8_block/bf16/ep2` | `novanas-qwen3-30b-a3b-instruct-2507-fp8-ep2-*` |
| `MistralForCausalLM/bf16/bf16/tp1` | `novanas-mistral-7b-instruct-v0.3-*` |
| `GptOssForCausalLM/mxfp4/bf16/tp1` | `novanas-gpt-oss-20b-*` (golden c1 15/16, see perf-log) |
| `OlmoeForCausalLM/bf16/fp8_e4m3/tp1` | `novanas-olmoe-1b-7b-0125-instruct-fp8kv-*` |
| `LlamaForCausalLM/bf16/tq4/tp1` | `novanas-llama-3.2-3b-instruct-tq4-*` |
| Phase 2 no-regression (`llama`, `olmoe` BF16) | `novanas-llama-3.2-3b-instruct-*`, `novanas-olmoe-1b-7b-0125-instruct-*` |

Goldens for every row c1/c16 PASS except gpt-oss c1, whose p15 `max_abs_logprob_diff_likely`
(0.3546) just exceeds the strict bound 0.35 — reproducible on a clean host; recorded for the lead.
