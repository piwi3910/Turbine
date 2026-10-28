# Quantization

Turbine serves quantized checkpoints for weights and KV cache. This page lists the formats, how a checkpoint is recognised, what is refused and why, and how accuracy is judged. The support status of each combination is in the [feature matrix](../features.md).

## Weight formats

You do not choose a weight format. `weights::detect` reads the checkpoint's `config.json`, selects the packaging, and the loader repacks each quantized layer into the kernels' layout. Quantized layers run through the kernel library's `quantize_act` and `qgemm` (ABI v2.9).

| Packaging (`config.json`)                                      | Weight format                  | Status on the R9700 (gfx1201)                                                                                  |
| -------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Unquantized BF16                                               | `bf16`                         | Supported for Llama, OLMoE, Qwen3 and Mistral on one GPU. Qwen3-MoE, Mixtral and the hybrids are experimental. |
| `ct_fp8`, `hf_fp8`                                             | `fp8` (per-tensor FP8)         | Supported on Llama. Mixtral `fp8` is supported at tp2 and ep2.                                                 |
| `fp8_block`                                                    | `fp8_block` (block-scaled FP8) | Supported on Llama and on Qwen3-MoE (tp2 and ep2). Qwen3.6 is experimental.                                    |
| `awq`                                                          | `awq_int4`                     | Supported on Llama.                                                                                            |
| `gptq`                                                         | `gptq_int4`                    | Supported on Llama. Checkpoints with `act_order` are refused (`gptq_act_order`).                               |
| `ct_pack_int4`                                                 | `awq_int4`                     | Supported on Llama, through the same kernels as AWQ.                                                           |
| `ct_mxfp4`, `quark_mxfp4`                                      | `mxfp4`                        | Experimental on Llama. Supported for gpt-oss-20b (below).                                                      |
| `openai_mxfp4`                                                 | `mxfp4`                        | Supported for gpt-oss-20b.                                                                                     |
| MXFP4 weights with emulated MXFP4 activations                  | `mxfp4_a4`                     | Experimental on Llama (golden compare 10 of 16).                                                               |
| `modelopt_nvfp4`, `modelopt_fp8`, `modelopt_mixed`, `ct_nvfp4` | NVFP4 and ModelOpt             | Deferred with NVIDIA support.                                                                                  |

Packagings and formats that are not in the matrix for a given architecture and device are refused, as described below. The CPU backend serves every format except the deferred ones, marked experimental.

### Notes on individual formats

- **Llama FP8**: dynamic and static per-tensor checkpoints both pass golden compare at concurrency 1 and 16 (16 of 16). The GSM8K difference from BF16 is within noise by McNemar's test.
- **Llama FP8 block**: passes golden compare and a 10-minute overload soak.
- **Llama AWQ and GPTQ**: both pass the accuracy budget (a GSM8K drop of 0.030 for AWQ and 0.0235 for the AutoRound GPTQ checkpoint, both against a 0.04 budget for 4-bit formats). At concurrency 1 the AWQ inter-token latency was 0.47 times the BF16 value. The FP8 single-stream latency is a follow-up.
- **gpt-oss-20b (MXFP4)**: supported. The checkpoint's experts are stored fused in MXFP4, so there is no BF16 variant of this architecture, and a BF16 request for it is refused with the reason stated in the matrix.
- **Mixtral FP8 and Qwen3-MoE FP8 block**: supported at tensor parallel 2 and expert parallel 2 with the multi-GPU judging bounds. See [Multi-GPU parallelism](parallelism.md).
- **Qwen3.5 and Qwen3.6 hybrids**: experimental. The gated-delta-net recurrence amplifies BF16 rounding, and the golden gate is open.

## Load-time refusals

A checkpoint whose quantization Turbine cannot serve is refused while loading. The server logs `event="quant_refused"` with a reason code and does not serve the checkpoint.

| Reason code                  | Meaning                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| `quant_scheme_unsupported`   | The packaging is detected, but the scheme is not implemented for this architecture or device. |
| `gptq_act_order`             | A GPTQ checkpoint uses activation ordering, which is not implemented.                         |
| `quant_moe_phase7`           | A quantized MoE layout whose family row is not yet gated.                                     |
| `quant_shard_misaligned`     | Tensor shard boundaries do not match the repacking layout.                                    |
| `quant_pipeline_unsupported` | The quantized format is not supported with pipeline parallelism.                              |

NVFP4 and ModelOpt checkpoints are refused with a note that they belong to the deferred NVIDIA work.

`/turbine/v1/status` reports the detected format under `quantization`.

## KV cache formats

Quantization of the KV cache is a separate setting, `kv.dtype` for the GPU pool and `kv.cpu.format` or `kv.nvme.format` for lower tiers.

- `kv.dtype: fp8_e4m3` stores FP8 KV pages using the checkpoint's per-layer scales. Support is per row: Llama with BF16 weights and OLMoE with calibrated scales are supported. Other rows are refused.
- `kv.dtype: tq4` stores TurboQuant 4-bit pages. Support is per row: supported for Llama on gfx1201, experimental for OLMoE.
- `kv.dtype: tq2` is refused at startup with exit code 2 and the reason `kv_tq2_l0_refused`. The 2-bit format fails the shared-prefix accuracy check on both models, so it is used only as a lower-tier format and a ladder rung.

The full table, the lower-tier results and the ladder are in [KV cache tiers and compression](kv-cache.md#formats).

## RoPE scaling

`model.rope_scaling` overrides the checkpoint's RoPE settings. YaRN is the scaling Turbine implements for extending the context window. The override lives in the configuration; the checkpoint is not modified.

## How accuracy is judged

Each quantized row is checked in three ways before it is supported:

1. **Golden compare.** Greedy generations are replayed on fixed prompts and compared, token by token, with a reference made by Hugging Face transformers. Concurrency 1 is the strict gate. Concurrency 16 uses looser batched bounds, because batch composition changes GEMM rounding. The Llama tolerance requires each reference top-5 candidate to be within 0.15 nats when the reference logprob is above -2, and within 0.55 nats otherwise.
2. **GSM8K accuracy.** The 200-item GSM8K set (`tests/eval/gsm8k-200.jsonl`) is run at concurrency 16. Where no reference engine runs the same checkpoint, the drop from the BF16 result must stay within the per-format budget: 0.02 for FP8 and 0.04 for 4-bit formats (`tests/eval/<model>/gate.json`). Where a reference engine runs the checkpoint, the comparison is against that engine, with the limit set in the model's `gate.json`.
3. **Shared-prefix eval.** For KV formats, a GSM8K variant whose questions share a long prefix exercises the reuse path, since a lossy format is only as good as the blocks it reuses.

Results are compared with `turbine-golden eval-compare`, which exits 1 when the drop is above `--max-drop`. The procedure is described in [Testing and quality gates](../testing.md).

Two results should be read with their caveats:

- **Llama FP8-dynamic** (a third-party checkpoint) measured 0.79 GSM8K against vLLM's 0.82 on the same checkpoint. That is a drop of 0.03, above the 0.01 bound against vLLM. The row is supported because the drop against Turbine's own BF16 result is within the 0.02 budget. Disclose the difference if FP8 throughput is shown alongside vLLM.
- **MXFP4 on Llama** is experimental, and its MXFP4 prefill kernel is a recorded follow-up.

## Quick reference

```yaml
model:
  path: /models/some-checkpoint # the format is detected, not configured
kv:
  dtype: fp8_e4m3 # only where the matrix row is supported
```

Check the resolved row before serving:

```sh
cargo run -p turbine-server -- --config my.yaml --check-config
```
