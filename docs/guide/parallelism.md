# Multi-GPU parallelism

Turbine can split one model across several GPUs with tensor, pipeline and expert parallelism, and run several replicas with data parallelism. All of it is configured under the `parallel` section and validated before the port is bound. Multi-node operation is deferred.

## Modes

| Mode                   | Key                               | What it does                                                                                                                                                     |
| ---------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tensor parallel (TP)   | `parallel.tensor_parallel_size`   | Shards each layer's weights and attention heads across devices. Needs an all-reduce per layer. Must be a power of two that divides the attention heads (1 to 8). |
| Data parallel (DP)     | `parallel.data_parallel_size`     | Runs independent replicas, one engine each, behind a router (1 to 64).                                                                                           |
| Pipeline parallel (PP) | `parallel.pipeline_parallel_size` | Splits the layers into stages, one thread per stage, for ranks in the same process.                                                                              |
| Expert parallel (EP)   | `parallel.expert_parallel_size`   | Places MoE experts on different devices. Applies to MoE families.                                                                                                |

`parallel.devices` lists the device indices to use, or `auto`. All devices in one group must be the same vendor.

## Accepted and refused combinations

Accepted:

- tensor × data parallel,
- pipeline × data parallel, with TP and EP at 1,
- expert × data parallel, with TP equal to 1 or to the EP degree.

Refused at startup with exit code 2:

| Reason                              | Combination                          |
| ----------------------------------- | ------------------------------------ |
| `combination_unsupported:pp+tp`     | Pipeline with tensor parallelism.    |
| `combination_unsupported:pp+ep`     | Pipeline with expert parallelism.    |
| `combination_unsupported:pp+static` | Pipeline with static rank processes. |

A draft-model speculation with TP or PP above 1 is refused with `speculative_parallel_unsupported` (see [Speculative decoding](speculative-decoding.md)).

## Rank runtime

Each device runs a rank. How the ranks are hosted is set by `parallel.ranks`:

- `mode: local` (the default): one thread per device in one process. The leader sends each step's plan to the workers through a bounded queue, `parallel.plan_queue_depth` (2 by default, 1 to 16).
- `mode: static`: one process per rank, joined to the leader over the rank transport. `transport: tcp` is the registered transport. Use this when each rank must run in its own process, for example one per host, with `ranks.leader` and `ranks.rank` set on each.

Collectives run on the backend selected by `parallel.collective_backend`:

| Backend   | Use                                                                                                                    |
| --------- | ---------------------------------------------------------------------------------------------------------------------- |
| `host`    | Collectives through host memory. Used for a single device.                                                             |
| `hostmem` | Host-memory collectives with a DMA path. Its threshold is `parallel.collective.hostmem_dma_min_bytes`, off by default. |
| `rccl`    | RCCL on AMD. The default for TP above 1 on AMD (`auto`).                                                               |

NCCL is reserved for the deferred NVIDIA work. A collective that times out is bounded by `parallel.collective.op_timeout` (30 s by default), and communicator setup by `parallel.collective.init_timeout` (120 s). A failed collective opens the circuit breaker and ends the affected requests with `replica_failed`.

## Data-parallel routing

With `parallel.data_parallel_size` above 1, a router in front of the replicas picks one for each request. `parallel.router` is either `prefix_affinity` (the default: requests with the same prefix go to the same replica, so the prefix cache is reused) or `least_loaded`.

Replicas each have their own KV pool. A prefix cached on one replica is not visible to another.

## Tensor-parallel prefill overlap

`parallel.tp_prefill_overlap` (off by default) splits a prefill of at least 256 rows into two halves. The all-reduce of one half overlaps the compute of the other. The split falls on a sequence start or a `kv.block_tokens` boundary, so the output is bitwise equal to the unsplit run and prefix reuse stays exact.

On Llama-3.2-3B at tensor parallel 2 on the R9700, the measured effect was +10 % tokens per second and −38 % TTFT p50 at concurrency 16 (see [Performance](../performance.md)). It is supported for dense-FFN families.

Two further switches are off by default: `parallel.tp_decode_graphs` (graph capture for decode under TP) and `parallel.collective.verify`. The verify mode checksums each collective's output across ranks and fails the step with `collective_corrupt` on a mismatch. It is a diagnostic: it cut throughput from about 912 to about 237 tokens per second in the measurement.

## Expert parallelism and MoE numerics

Expert parallelism with tensor parallelism is supported for OLMoE. Any change in summation order can flip a near-tie router decision, so multi-GPU MoE runs are judged with calibrated multi-GPU bounds (`turbine-golden compare --multi-gpu-bounds`) rather than the single-GPU ones. A near-tie flip of a top-k expert can be excused by the judging rule `missing_top_k_margin_nats`, which the golden report records.

## Measured multi-GPU results

These are on two R9700 cards on one host, against vLLM-ROCm 0.23.0 on the same cards. Read the caveats in [Performance](../performance.md) before quoting any ratio.

| Model                     | Placement | Turbine tokens/s (c16) | Note                                                                                                         |
| ------------------------- | --------- | ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| Qwen3-30B-A3B (FP8 block) | TP 2      | 400.8 (vLLM 91.6)      | Golden compare 16 of 16 at concurrency 1 and 16.                                                             |
| Qwen3-30B-A3B (FP8 block) | EP 2      | 208.3 (vLLM 89.7)      | Same golden result.                                                                                          |
| Mixtral-8x7B (FP8)        | TP 2      | 204.9 (vLLM 64.6)      | GSM8K 0.60 against vLLM's 0.59.                                                                              |
| Mixtral-8x7B (FP8)        | EP 2      | 162.2 (vLLM 63.9)      | Golden compare passes under the multi-GPU rule; one near-tie flip is excused by `missing_top_k_margin_nats`. |
| Llama-3.2-3B (BF16)       | TP 2      | 912 (vLLM 97.5)        | A partial comparison: one point at concurrency 16, stopped early. Not a result to quote.                     |

Pipeline parallelism and data parallelism for Llama, and expert parallelism for OLMoE, have no recorded vLLM comparison.

## Status

| Feature                                                   | Status                        |
| --------------------------------------------------------- | ----------------------------- |
| Tensor, data, pipeline and expert parallelism on one host | Supported, as listed above.   |
| Multi-node                                                | Deferred.                     |
| NCCL                                                      | Deferred with NVIDIA support. |

The [feature matrix](../features.md#parallelism) lists each accepted combination with its status.
