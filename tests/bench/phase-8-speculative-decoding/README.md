# Phase 8 — speculative decoding: gate evidence

Status: **implemented and gated; the S-11 throughput bar missed, so the
`amd/gfx1201 LlamaForCausalLM bf16 bf16 draft` support row stays `experimental`**
with the measured miss in its reason. Everything else (golden c1/c16, the serial
A/B, the exit gates, vLLM, the soak) passed.

## Outcome

| Gate | Result | Evidence |
| --- | --- | --- |
| 1a `scripts/gate.sh --full` | **PASS** (1154 tests) | gate log |
| 1b `scripts/lab-test.sh novanas --tier full` | **PASS** (run `1026151053-12725e3c`; one served-test skip fix and one test-side KV-publish race fix) | `target/lab-test/` |
| 1c `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` | **PASS** (70 test targets; run `1026172029-2d850040`) | `target/lab-test/` |
| 1d `scripts/lab-bench.sh --golden16` | **PASS**: Llama 16/16 at 866.3 tok/s; OLMoE 16/16 at 637.6 tok/s | `target/lab-bench/p8-gate1d-*` |
| 2 serve the draft | **PASS**: `/ready` 200, `status.speculative.active: true`, startup `event="speculative"` (draft 1B BF16, 2.47 GB weights, 1 GiB KV slot) | server log |
| 3a golden with the draft | **PASS**: c1 16/16 (strict bounds), c16 16/16 (batched bounds) | `golden-c1.json`, `golden-c16.json` |
| 3b serial A/B | **PASS**: 16/16 strict against the draft server's capture | `spec-capture.jsonl`, `ab-serial.json` |
| 4a c1 TTFT | **PASS**: 59.63 ms vs 59.30 ms = 1.006× (≤ 1.15) | `c1-draft-50.json`, `c1-none-50.json` |
| 4b c1 decode throughput | **MISS**: 82.99 vs 88.54 tok/s = **0.937×** (needs ≥ 1.3); acceptance 278/1624 = 17.1 %, 0.68 tokens/round | same |
| 4c c16 no-regression | **PASS**: 845.4 tok/s = 0.979× of the 863.9 Phase 7 exit (≥ 0.97); speculation inactive under the single-item rule | `bench-c16-draft-200.json` |
| 5 vLLM-ROCm speculative | **recorded**: served `draft_model` on 18100, c1 64.47 tok/s (Turbine draft 1.29× vLLM) | `vllm-c1.json` |
| 6 10-minute soak with the draft | **PASS**: 4276×200, 11×429 (`queue_full`), 2557 `queue_timeout`, 0 client drops, ITL p99 210 ms vs 170.5 ms calibrated (1.23×), GREEN 5 s after cooldown, KV idle, reserve held | `soak-verdict.json` |

## Why the row did not flip

The S-11 bar (−1.3× single-stream decode over `method: none`) is not reachable with
the validated pairing on the mandated workload: the 1B draft agrees with the 3B
target on only 17 % of greedy proposals for `--prompt-words 512`, so each
verification round returns ~0.68 tokens while the target pays a 5-row forward plus
four 1B draft forwards. The 200-request `lab-bench` profile reproduces the miss
(79.04 vs 84.28 tok/s = 0.938×), and vLLM's own `draft_model` on the same pairing is
slower still (64.47 tok/s). The row therefore keeps
`SupportStatus::Experimental` with `P8_PENDING_REASON` carrying the measured numbers;
the spec's S-9 forbids marking it supported without the gate.

## Reproduce

```sh
scripts/gate.sh --full
scripts/lab-test.sh novanas --tier full
scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full
scripts/lab-bench.sh --model llama --golden16
scripts/lab-bench.sh --model olmoe --golden16
# Gate 2/3/4 (50-request S-11 profile):
scripts/lab-serve.sh novanas scripts/lab/phase8-novanas-llama-draft.yaml
target/release/turbine-golden compare --url http://192.168.10.203:18000 \
  --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --concurrency 1 --output json
target/release/turbine-bench --url http://192.168.10.203:18000 \
  --model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --requests 50 \
  --prompt-words 512 --max-tokens 256 --ignore-eos --output json
scripts/lab-serve.sh novanas --stop
# Gate 5:
scripts/lab-serve.sh novanas --vllm llama-3.2-3b-instruct --vllm-speculative \
  '{"method":"draft_model","model":"/models/llama-3.2-1b-instruct","num_speculative_tokens":4}'
# Gate 6:
scripts/overload-soak.sh novanas --duration 10m --set speculative.method=draft \
  --set speculative.draft_model_path=/models/llama-3.2-1b-instruct
```
