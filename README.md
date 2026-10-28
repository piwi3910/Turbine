# Turbine

Turbine is an LLM inference engine written in Rust. It serves models through an OpenAI-compatible API with streaming, tool calls, reasoning content and JSON-schema output. Its serving path has no Python and no PyTorch: requests go from HTTP through Turbine's scheduler and KV cache to a kernel library that Turbine loads at run time. The first production target is the AMD Radeon AI PRO R9700 (gfx1201) with ROCm and HIP. Turbine keeps the KV cache in managed tiers (GPU, pinned host memory, NVMe), degrades under memory pressure with documented states rather than running out of memory, and states the support status of every model, format and placement it can serve.

Documentation: **https://piwi3910.github.io/Turbine/**

## Key differences

- **KV cache as a managed resource.** Blocks move between GPU memory, pinned host memory and NVMe. Prefix reuse is token-granular, sessions have tiers and timers, and eviction is cost-aware. Lower tiers can store blocks in FP8 or TurboQuant formats, and lossy reuse is reported per request.
- **Overload is designed for.** A pressure controller with hysteresis moves between five states. Admission reserves each request's worst-case KV. A circuit breaker drains and probes the device after errors, and a sticky device error exits with a code a supervisor can act on.
- **Support is declared.** Every combination of vendor, device, model, weight format, KV format and speculation maps to `supported`, `experimental` or `unsupported`. Unsupported combinations are refused at startup with a reason code, before any port is bound.
- **Fast start.** Llama-3.2-3B is ready 6 to 8 seconds after launch in the measured run.
- **Pluggable.** Model families, tool and reasoning formats, weight and KV formats, scheduling and eviction policies, backends and kernels are each a trait with a static registry and a conformance suite.
- **Measured, with the caveats stated.** Golden logprob comparison gates every change that can affect numerics, and GSM8K accuracy gates every quantized format.

## Status

| Area                                                                                  | Status                                                |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Llama-3.2-3B-Instruct, BF16, one R9700                                                | Supported                                             |
| OLMoE-1B-7B, BF16, one R9700                                                          | Supported                                             |
| Qwen3 dense (BF16), Mistral (BF16), gpt-oss-20b (MXFP4) on one R9700                  | Supported                                             |
| Qwen3-MoE (FP8 block) and Mixtral (FP8) at tensor and expert parallel 2               | Supported                                             |
| Llama FP8, FP8 block, AWQ INT4 and GPTQ INT4 weights                                  | Supported                                             |
| KV cache: host and NVMe tiers, prefix reuse, sessions, prefetch                       | Supported                                             |
| KV cache: FP8 and TurboQuant 4-bit in lower tiers; TurboQuant 4-bit in L0 for Llama   | Supported (per row)                                   |
| Compression ladder                                                                    | Supported, opt-in                                     |
| Data, pipeline and tensor parallelism on one host                                     | Supported (see the matrix)                            |
| Draft-model speculative decoding (Llama 1B for 3B)                                    | Experimental: slower than plain decoding on this card |
| Qwen3.5 and Qwen3.6 hybrids, MXFP4 weights on Llama, TurboQuant 4-bit as L0 for OLMoE | Experimental                                          |
| TurboQuant 2-bit as the L0 format                                                     | Refused at startup (exit 2)                           |
| NVIDIA support, NVFP4 weights, multi-node                                             | Deferred                                              |

The full matrix, with the evidence for each row and the level definitions, is in [docs/features.md](docs/features.md). The support matrix in code is authoritative: `turbine-server --support-matrix` prints it, and the server reports the row it resolved for its configuration.

## Performance

On one R9700, Llama-3.2-3B-Instruct in BF16 served **863.9 output tokens per second** at 16 concurrent requests, against **738.0** for vLLM-ROCm 0.23.0 on the same workload, a ratio of **1.17**. Against the vLLM figure measured on GPU 0 alone (715.1), the ratio is 1.21.

These come from different runs a month apart, on one workload, with one card class. The claim is specific to that setup. [docs/performance.md](docs/performance.md) gives the method, the other models, and the comparisons that should not be quoted, such as the gpt-oss ratio and a partial tensor-parallel run.

## Quickstart

Requirements: Rust 1.97 or newer. Serving needs a Linux host with ROCm and an R9700, a built HIP kernel library and a Llama-3.2-3B-Instruct checkpoint in Hugging Face format.

Build and check on any machine, with no GPU:

```sh
git clone https://github.com/piwi3910/Turbine.git
cd Turbine
cargo build --workspace
cargo run -p turbine-server -- --config examples/turbine.yaml --check-config
cargo run -p turbine-server -- --support-matrix
```

Build the HIP kernel library on the ROCm host:

```sh
cmake -S kernels/rocm -B build/rocm -DCMAKE_HIP_COMPILER="$(command -v hipcc)" -DGPU_TARGETS=gfx1201
cmake --build build/rocm
```

Serve the model:

```sh
hf download unsloth/Llama-3.2-3B-Instruct --revision 006f5dcd1393c3add266de40994ba96225e9689d \
  --local-dir ./models/llama-3.2-3b-instruct

cargo run --release -p turbine-server -- --config examples/turbine.yaml \
  --set model.path=$PWD/models/llama-3.2-3b-instruct \
  --set execution.kernel_library=$PWD/build/rocm/libturbine_hip.so
```

When `/ready` answers 200, send a request:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"Hello."}],"max_tokens":64}'
```

The full walkthrough is in the [quickstart](https://piwi3910.github.io/Turbine/getting-started/quickstart.html), and the configuration keys are in the [configuration reference](https://piwi3910.github.io/Turbine/getting-started/configuration.html).

## Documentation

- [Introduction](https://piwi3910.github.io/Turbine/introduction.html) and [why Turbine](https://piwi3910.github.io/Turbine/why-turbine.html)
- [Install and build](https://piwi3910.github.io/Turbine/getting-started/install.html), [quickstart](https://piwi3910.github.io/Turbine/getting-started/quickstart.html), [configuration](https://piwi3910.github.io/Turbine/getting-started/configuration.html)
- User guide: [API and diagnostics](https://piwi3910.github.io/Turbine/guide/openai-api.html), [KV cache](https://piwi3910.github.io/Turbine/guide/kv-cache.html), [reliability](https://piwi3910.github.io/Turbine/guide/reliability.html), [quantization](https://piwi3910.github.io/Turbine/guide/quantization.html), [parallelism](https://piwi3910.github.io/Turbine/guide/parallelism.html), [speculative decoding](https://piwi3910.github.io/Turbine/guide/speculative-decoding.html), [tools, reasoning and structured output](https://piwi3910.github.io/Turbine/guide/tools-reasoning-structured-output.html), [observability](https://piwi3910.github.io/Turbine/guide/observability.html)
- [Architecture](https://piwi3910.github.io/Turbine/architecture.html), [features](https://piwi3910.github.io/Turbine/features.html), [performance](https://piwi3910.github.io/Turbine/performance.html), [testing](https://piwi3910.github.io/Turbine/testing.html)
- [Extending Turbine](https://piwi3910.github.io/Turbine/extending/README.html) and [engineering standards](https://piwi3910.github.io/Turbine/standards.html)

The same pages are readable as Markdown under [`docs/`](docs/introduction.md).

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the build and test steps, what a change has to pass, and the commit and pull-request conventions. The binding engineering rules are in [AGENTS.md](AGENTS.md) and [docs/standards.md](docs/standards.md). Report issues on GitHub with the Turbine commit, the `--check-config` output, the device status and the log lines around the failure.

## License

Turbine is licensed under the [Apache License 2.0](LICENSE). Third-party code in the kernel library is attributed in [NOTICE](NOTICE), with its licence texts under `kernels/rocm/third_party/LICENSES/`.
