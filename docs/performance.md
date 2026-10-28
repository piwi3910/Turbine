# Performance against vLLM-ROCm

This page reports Turbine's measured throughput against vLLM-ROCm on the same AMD Radeon AI PRO R9700 cards, with the method, the dates and every caveat that applies. The numbers were gathered on 2026-10-29 from the committed records in the repository. Read the caveats before quoting any figure.

## Headline

On one R9700 (gfx1201), Llama-3.2-3B-Instruct in BF16 served 863.9 output tokens per second at 16 concurrent requests in Turbine's latest recorded run. vLLM-ROCm 0.23.0 served 738.0 tokens per second on the same workload in its recorded run. The ratio is 1.17. Against the vLLM figure taken on GPU 0 alone (715.1), the ratio is 1.21.

The two sides come from different runs a month apart (see [caveat 1](#notes-and-caveats)). The defensible public claim is "about 1.17 to 1.21 times vLLM-ROCm 0.23.0 for Llama-3.2-3B BF16 on one R9700, at 16 concurrent requests, on one workload", not a general claim about either engine.

## Hardware and software

- **Hardware.** One lab host with two AMD Radeon AI PRO R9700 cards (RDNA4, gfx1201, 32 GB). Each card has a 300 W power cap and a PCIe Gen5 x8 link at the root port. Since 2026-10-06 GPU 1 has benchmarked within 0.6 % of GPU 0. Before that date GPU 1 had link and power problems, so the earliest Phase 2 rows are pinned to GPU 0.
- **Turbine.** The latest throughput figures for Llama and OLMoE come from commit `212ae15e`, measured at the Phase 7 exit on 2026-10-18.
- **vLLM.** The upstream `rocm/vllm` image at the pinned tag `rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`, that is vLLM 0.23.0. The gpt-oss-20b run used vLLM 0.23.1.
- **Serving.** Each engine serves the same checkpoint on the same card or cards, on the same host, measured by the same client.

## Workloads

Throughput workload, the standard for each comparison:

```sh
turbine-bench --url <base> --model <id> --concurrency 16 --requests 200 \
  --prompt-words 512 --max-tokens 256 --ignore-eos
```

The metric is output-token throughput in tokens per second. Each run starts with 16 warm-up requests. Each comparison runs the workload three times per engine and reports the median and the ratio Turbine ÷ vLLM.

Single-stream (concurrency 1) numbers use the same client with `--concurrency 1`, and are reported separately below.

Accuracy uses GSM8K, 200 items (`tests/eval/gsm8k-200.jsonl`), at concurrency 16 and greedy decoding. Both engines are scored by the same `turbine-golden eval` client. A GSM8K pass means the drop from the reference is within the budget in the checkpoint's `gate.json`.

Correctness gates each Turbine throughput figure. A throughput number from a run that failed golden compare is not counted. Golden compare is strict at concurrency 1 and uses batched bounds at concurrency 16, against a Hugging Face transformers reference. See [Testing and quality gates](testing.md).

## Results

Concurrency 16, the standard workload unless stated. Ratio is Turbine ÷ vLLM-ROCm 0.23.x on the same card or cards.

| Model (weights)                                   | Placement | Turbine tok/s                | vLLM tok/s                                | Ratio                              | Turbine TTFT p50 / ITL p50               | GSM8K Turbine vs vLLM                                              | Turbine run / vLLM run              | Status                          |
| ------------------------------------------------- | --------- | ---------------------------- | ----------------------------------------- | ---------------------------------- | ---------------------------------------- | ------------------------------------------------------------------ | ----------------------------------- | ------------------------------- |
| Llama-3.2-3B-Instruct (BF16)                      | 1 GPU     | 863.9 (866.3 at the P8 gate) | 738.0 (715.1 on GPU 0)                    | **1.17×** (1.21× against 715.1)    | 208 ms / 15.3 ms                         | 0.795 Turbine; no vLLM GSM8K for BF16                              | Turbine 2026-10-18; vLLM 2026-09-26 | Supported                       |
| OLMoE-1B-7B-0125-Instruct (BF16)                  | 1 GPU     | 638.8 (637.6 at the P8 gate) | 534.9 (535.5 on GPU 0)                    | **1.19×**                          | 128 ms / 23.2 ms (vLLM 200 ms / 26.6 ms) | none for vLLM                                                      | Turbine 2026-10-18; vLLM 2026-09-26 | Supported                       |
| Qwen3-4B-Instruct-2507 (BF16)                     | 1 GPU     | 598.8                        | 630.5                                     | **0.95×**                          | not recorded                             | 0.915 vs 0.920                                                     | 2026-10-08                          | Supported                       |
| Mistral-7B-Instruct-v0.3 (BF16)                   | 1 GPU     | 406.7                        | 410.3                                     | **0.99×**                          | 508 ms / 30.4 ms                         | 0.550 vs 0.530                                                     | 2026-10-08                          | Supported                       |
| gpt-oss-20b (MXFP4)                               | 1 GPU     | 277.2                        | 133.6 (only 37 of 200 requests succeeded) | 2.07×, **indicative only**         | about 4.9 s / 37.6 ms                    | 0.925 vs 0.930                                                     | 2026-10-16                          | Supported                       |
| Qwen3-30B-A3B-Instruct-2507 (FP8 block)           | TP 2      | 400.8                        | 91.6                                      | **4.37×**                          | 583 ms / 31.7 ms                         | 0.94 vs 0.945                                                      | 2026-10-14                          | Supported                       |
| Qwen3-30B-A3B-Instruct-2507 (FP8 block)           | EP 2      | 208.3                        | 89.7                                      | **2.32×**                          | 445 ms / 71.3 ms                         | 0.94 vs 0.94                                                       | 2026-10-14                          | Supported                       |
| Mixtral-8x7B-Instruct-v0.1 (FP8)                  | TP 2      | 204.9                        | 64.6                                      | **3.17×**                          | not recorded                             | 0.60 vs 0.59                                                       | 2026-10-09 run; gate 2026-10-14     | Supported                       |
| Mixtral-8x7B-Instruct-v0.1 (FP8)                  | EP 2      | 162.2                        | 63.9                                      | **2.54×**                          | not recorded                             | 0.64 vs 0.56                                                       | 2026-10-18 re-gate                  | Supported                       |
| Llama-3.2-3B FP8-dynamic (third-party checkpoint) | 1 GPU     | 976.4 (960.3 at the 6a exit) | 651.1                                     | **1.50×** (1.47× at exit)          | 155 ms / 13.9 ms (vLLM 258 / 19.8)       | 0.79 vs 0.82, a drop of 0.03 against vLLM, **over the 0.01 bound** | 2026-09-29; exit 2026-09-30         | Supported (see caveat 7)        |
| Llama-3.2-3B FP8-block                            | 1 GPU     | 1062.4 (1049.9 at exit)      | 677.8                                     | **1.57×**                          | 226 ms / 11.5 ms                         | 0.800 vs Turbine BF16 0.805; no vLLM GSM8K                         | 2026-09-30                          | Supported                       |
| Llama-3.2-3B AWQ                                  | 1 GPU     | 1261.3                       | not recorded                              | not compared                       | 224 ms / 9.2 ms                          | 0.775 vs 0.755                                                     | 2026-09-30                          | Supported                       |
| Llama-3.2-3B GPTQ (AutoRound)                     | 1 GPU     | 1268.9                       | not recorded                              | not compared                       | 222 ms / 9.1 ms                          | Turbine only; GPTQ checkpoint full GSM8K 0.749 vs 0.749            | 2026-09-30                          | Supported                       |
| Llama-3.2-3B BF16                                 | TP 2      | 912                          | 97.5                                      | about 9.4×, **partial run**        | not recorded                             | not run                                                            | 2026-09-28, stopped early           | Not a result (see caveat 6)     |
| Llama-3.2-3B BF16 with 1B draft (speculative)     | 1 GPU, c1 | 82.99 (50 requests)          | 64.47 (vLLM `draft_model`, 4 tokens)      | 1.29×, **c1 only**                 | 59.6 ms / 11.1 ms                        | not run                                                            | 2026-10-27                          | Experimental (see caveat 8)     |
| Qwen3.6-35B-A3B (FP8 block, hybrid)               | TP 2, c1  | 41.65 / 42.30 (c16: 428.75)  | "40+" at c1, no committed measurement     | about 1.0×, **unsourced baseline** | not recorded                             | not run                                                            | 2026-10-28                          | Experimental (see caveat 9)     |
| Qwen3.5-9B (BF16, hybrid)                         | 1 GPU     | no comparison                | not measured                              | not measured                       | not recorded                             | not run                                                            | not recorded                        | Experimental (golden gate open) |

The rows Turbine's records mark as not recorded have no committed vLLM figure. Pipeline and data parallelism for Llama, and expert parallelism for OLMoE, have no vLLM comparison at all.

### Single-stream reference points

These are Turbine's own concurrency 1 numbers. No plain vLLM concurrency 1 figure is committed for the same workload, so there is no ratio.

- Llama-3.2-3B BF16: 88.54 tokens per second, TTFT 59.3 ms, ITL 11.05 ms (2026-10-27, 50 requests). With 200 requests, 84.28 tokens per second.
- Llama-3.2-3B BF16: 78.0 tokens per second, and OLMoE: 150 tokens per second (2026-10-06, quick run).
- Llama-3.2-3B FP8-block: 112.7 tokens per second, ITL 8.43 ms.

### Startup

Turbine reports ready 6 to 8 seconds after process start for Llama-3.2-3B on tensor parallel 2 (2026-09-28). vLLM-ROCm took about 98 seconds to become ready in the same comparison, or about 124 seconds counted from pod start. vLLM spent most of that time spawning workers, running `torch.compile` (about 11 s) and capturing graphs (about 49 s). Loading the weights took about 2.5 seconds on both engines.

## Notes and caveats

1. **The two sides come from different runs.** The Llama and OLMoE BF16 vLLM baselines are from 2026-09-26, at the start of Phase 2c, and were not measured again. The Turbine numbers come from the Phase 7 exit on 2026-10-18, a single run of the bench script. Both use the same workload and the same card class, but the runs are a month apart. Each model has two recorded vLLM figures: one on a card chosen by the scheduler (738.0 for Llama, 534.9 for OLMoE) and one pinned to GPU 0 (715.1 with 199 of 200 requests succeeding, and 535.5). The published claim, "about 1.17 to 1.21 times on Llama and about 1.19 times on OLMoE", should be re-measured with the same harness before it is repeated in a comparison.

2. **Superseded Turbine numbers.** The 2026-09-26 parity point (739.4 tokens per second for Llama, commit `bd99d87`) and the Phase 2c figures (770.4 and 619.8) are superseded. Later kernel work raised them to about 864 for Llama and 637 to 639 for OLMoE, at the 5p exit and at the Phase 7 exit.

3. **The gpt-oss-20b 2.07 times ratio is not a fair comparison.** The fixed-length workload with `--ignore-eos` breaks on a reasoning model. vLLM 0.23.1's Harmony parser returned HTTP 500 for about 80 % of requests, so the median was taken over only 37 successful requests of 200. About 5 % of Turbine's requests produced no `content`. The project's own record calls it "an indicative direction (Turbine is not slower than vLLM), not a measured-equivalent ratio". Do not quote 2.07 times as a speed-up. A date-dependent chat template also made golden concurrency 1 read 15 of 16 until the template was pinned.

4. **The multi-GPU MoE ratios partly reflect vLLM-ROCm 0.23.0 being weak on two RDNA4 cards.** In the Phase 5 tensor-parallel comparison, vLLM fell back to an RCCL-only all-reduce without its custom all-reduce. It produced about 90 tokens per second on Qwen3-30B-A3B, which suggests it is not well tuned for this hardware. Present these ratios as "against vLLM-ROCm 0.23.0 on this hardware", not as general claims. The Phase 7 exit re-measured Turbine alone: Qwen3-MoE 398.0 at TP 2 and 206.7 at EP 2, and Mixtral 181 at TP 2 and 192.9 at EP 2. These agree with the comparison runs within run-to-run noise or differences in bench shape.

5. **Mixtral status.** The TP 2 and EP 2 rows are supported after the re-gate of 2026-10-18. An earlier decision had kept them experimental.

6. **The Llama TP 2 figure (912 against 97.5) is a partial run, not a result.** Only the concurrency 16 point was recorded, and the comparison was stopped early. The details are in an external record, not in the repository. At that time GPU 1 was on a Gen4 x8 link. Pipeline and data parallelism for Llama, and OLMoE expert parallelism, have no vLLM comparison recorded.

7. **FP8-dynamic accuracy miss.** On the same third-party checkpoint, Turbine scored 0.79 on GSM8K against vLLM's 0.82 (a drop of 0.03, over the 0.01 bound in that checkpoint's `gate.json`). The row is supported because the drop against Turbine's own BF16 result stayed within the 0.02 budget. Disclose this difference whenever the FP8 throughput is shown.

8. **Speculative decoding.** The 1.29 times figure compares Turbine's draft path with vLLM's `draft_model` path at concurrency 1. Turbine's own draft mode is slower than Turbine without speculation: 0.937 times, at 17.1 % acceptance. The row stays experimental. The vLLM run record is cited in the project's speculative-decoding README but is not committed alongside it.

9. **Qwen3.6 vLLM figure.** The "40+ tokens per second at concurrency 1" for vLLM appears only in prose in the project's record, with no run identifier or file. The Qwen3.5 and Qwen3.6 rows are experimental because their golden gates are open, so do not publish them as comparisons.

10. **Rows with no vLLM throughput.** AWQ and GPTQ have no vLLM throughput (vLLM GSM8K exists for AWQ only). The Llama 3.1-8B MXFP4 variants are experimental and failed their gate (vLLM GSM8K 0.735 against Turbine's 0.54 for the W4A4 variant). The KV formats (FP8 KV and `tq4`) have no vLLM comparison, and neither does any Phase 5 mode except the partial TP 2 point.

11. **Correctness.** Golden concurrency 1 passed on every Turbine throughput row above, except where noted. The gpt-oss row at the Phase 7 exit read 15 of 16 because its fixture depended on the date; the fixture was then pinned.

## Reproducing

The project's benchmark scripts drive lab hosts with two cards, so a reader on other hardware should use the client directly:

```sh
cargo run --release -p turbine-bench --bin turbine-bench -- \
  --url http://127.0.0.1:8000 --concurrency 16 --requests 200 \
  --prompt-words 512 --max-tokens 256 --ignore-eos --output json
```

Run the same command against a vLLM-ROCm server serving the same checkpoint on the same card, with three runs each, and compare medians. Report the vLLM version, the card, the driver and the date with any number you publish.

The golden check that must pass before a throughput figure is counted:

```sh
cargo run --release -p turbine-bench --bin turbine-golden -- compare \
  --url http://127.0.0.1:8000 \
  --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --concurrency 1
```

## What is not measured

- Multi-node throughput (deferred).
- NVIDIA throughput (deferred).
- Speculative decoding beyond the one draft pairing and the one setting in the table.
- Prefill-heavy and long-context workloads: the standard workload uses 512-word prompts and 256 output tokens.
