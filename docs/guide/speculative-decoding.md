# Speculative decoding

Speculative decoding lets a small draft model propose several tokens, and the target model verifies all of them in one forward pass. Accepted tokens are the same tokens the target would have produced greedily. Turbine supports one draft method, `draft`, on a single device. Its measured speed on the R9700 does not beat plain decoding, so the feature is experimental and off by default.

## Status

| Pairing                                                                   | Device              | Status                                                   |
| ------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------- |
| `unsloth/Llama-3.2-1B-Instruct` drafting for Llama-3.2-3B-Instruct (BF16) | AMD gfx1201 (R9700) | Experimental. Served, with a startup warning.            |
| The same pairing                                                          | CPU backend         | Experimental (host tests).                               |
| Any other draft pairing                                                   | any                 | Unsupported: refused at startup with exit code 2.        |
| Draft with tensor or pipeline parallelism above 1                         | any                 | Unsupported: exit 2, `speculative_parallel_unsupported`. |
| Multi-token-prediction (MTP) heads                                        | any                 | Planned. Not served.                                     |

The row stays experimental because the 2026-10-27 gate missed its speed bar. See [Measured results](#measured-results).

## Configuration

```yaml
speculative:
  method: draft # none (default) or draft
  num_tokens: 4 # draft tokens per round, 1 to 8
  draft_model_path: /models/llama-3.2-1b-instruct # required with draft
  min_acceptance: 0.4 # 0 to 1; below this a request stops speculating
```

The same keys can be set on the command line:

```sh
cargo run --release -p turbine-server -- --config examples/turbine.yaml \
  --set model.path=/models/llama-3.2-3b-instruct \
  --set execution.kernel_library=/path/to/libturbine_hip.so \
  --set speculative.method=draft \
  --set speculative.draft_model_path=/models/llama-3.2-1b-instruct \
  --set speculative.num_tokens=4
```

How it works, in brief: the draft model proposes `num_tokens` greedy tokens; the target model checks them in one step; the longest matching prefix is kept, and the target's own token is used at the first mismatch. Speculation is per request: when a request's acceptance falls below `min_acceptance`, that request stops speculating.

## Requirements and refusals

The draft must be a checkpoint of the same family, with the same tokenizer, in BF16, and with a context length compatible with the target. A mismatch is refused before the port is bound, with exit code 2.

| Condition                                                       | Result                                      |
| --------------------------------------------------------------- | ------------------------------------------- |
| `method: draft` without `draft_model_path`                      | Exit 2, configuration error.                |
| Family, tokenizer, dtype or context mismatch                    | Exit 2, before the port is bound.           |
| Tensor or pipeline parallelism above 1                          | Exit 2, `speculative_parallel_unsupported`. |
| The device memory footprint cannot hold the target's minimum KV | Exit 1, `speculative_budget`.               |

Turbine only speculates on one device. Use a single-device placement when you enable it.

## Observability

- `/turbine/v1/status` reports `speculative` with `method`, `draft_model_path`, `num_tokens`, `min_acceptance`, `active` and `acceptance`.
- Metrics: `turbine_speculative_rounds_total`, `turbine_speculative_proposed_tokens_total`, `turbine_speculative_accepted_tokens_total`, `turbine_speculative_accepted_per_round` and `turbine_speculative_disabled`.

Compare the acceptance rate with `min_acceptance`. Requests whose acceptance falls below it stop speculating, so a low rate shows up as requests that have stopped, not as errors.

## Measured results

These were measured on one R9700 with Llama-3.2-3B-Instruct (BF16) and the 1B draft. They are the project's gate evidence, not general claims.

| Comparison                                                                                | Result                                                                                    |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Turbine with draft against Turbine without speculation, concurrency 1, decode speed       | 0.937 times (82.99 against 88.54 tokens per second on 50 requests), at 17.1 % acceptance. |
| Time to first token, same comparison                                                      | 1.006 times.                                                                              |
| Concurrency 16 throughput, draft against no speculation                                   | 0.979 times.                                                                              |
| Turbine with draft against vLLM-ROCm 0.23.0 running its `draft_model` path, concurrency 1 | 1.29 times (82.99 against 64.47 tokens per second).                                       |

Read these together. The draft mode is slower than Turbine's own plain decoding, so the 1.29 times figure reflects vLLM's draft path being slower still on this card. It does not mean speculation is a win for Turbine. The gate required at least 1.3 times Turbine's own c1 decode speed, and the row stayed experimental.

Golden compare passed at concurrency 1 and 16 with the draft enabled. The recorded gate used `num_tokens` 4 and `min_acceptance` 0.4.

Speculation's value depends on the acceptance rate and on the cost of the draft pass. The recorded gate covers one draft and one `num_tokens` setting; other combinations are not measured here.

## Testing it

The proposer lives in `turbine-model` and the verifier in `turbine-scheduler`. Their host tests run without a GPU:

```sh
cargo test -p turbine-model
cargo test -p turbine-scheduler
```

The GPU run of the draft path is a lab procedure that the project runs on its own hardware. See [Testing and quality gates](../testing.md) for what is run and where the records are kept.

## Adding a proposer

Proposers are pluggable: a new method is one file and one registry entry. See [Speculative proposer](../extending/speculative-proposer.md).
