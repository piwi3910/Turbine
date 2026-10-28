# Quickstart

This walkthrough serves Llama-3.2-3B-Instruct in BF16 on one AMD Radeon AI PRO R9700 (gfx1201) and sends it a chat request. It assumes that the [install steps](install.md) are done and that `build/rocm/libturbine_hip.so` exists.

Without a GPU you can still run steps 1 and 2 and the whole test suite.

## 1. Validate the configuration

```sh
cargo run -p turbine-server -- --config examples/turbine.yaml --check-config
```

The output has a line of the form `support: <status> (<vendor>/<arch>/<architecture>/<weight>/<kv>/<speculation>)` followed by `config ok`. Before device discovery the device architecture is `*`. An `unsupported` row exits with code 2.

## 2. Read the support matrix

```sh
cargo run -p turbine-server -- --support-matrix
cargo run -p turbine-server -- --support-matrix --output json
```

The first form prints every row of the table. The second prints the same rows as JSON, plus the parallel refusals and the deferred vendors. The command reads no configuration and exits with code 0.

## 3. Download the weights

Turbine never downloads weights. Fetch the checkpoint yourself, for example with the Hugging Face CLI. This is the revision the project's measurements used:

```sh
hf download unsloth/Llama-3.2-3B-Instruct \
  --revision 006f5dcd1393c3add266de40994ba96225e9689d \
  --local-dir ./models/llama-3.2-3b-instruct
```

The directory must contain `config.json`, the safetensors shards and the tokenizer files. If the repository you use is gated on Hugging Face, accept its license on the website and log in with `hf auth login` first.

## 4. Start the server

Point the configuration at the model directory and at the kernel library:

```sh
cargo run --release -p turbine-server -- \
  --config examples/turbine.yaml \
  --set model.path=$PWD/models/llama-3.2-3b-instruct \
  --set execution.kernel_library=$PWD/build/rocm/libturbine_hip.so
```

The example configuration listens on `127.0.0.1:8000`. While the model loads, `/ready` answers 503 with `loading_model`. When warm-up is done it answers 200.

```sh
until curl -fs http://127.0.0.1:8000/ready >/dev/null; do sleep 1; done
curl -s http://127.0.0.1:8000/v1/models
```

With `model.served_name` left at `null`, the model id is the last component of the model path: `llama-3.2-3b-instruct`.

## 5. Send a chat request

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "llama-3.2-3b-instruct",
        "messages": [{"role": "user", "content": "Name three KV cache tiers."}],
        "max_tokens": 128,
        "temperature": 0
      }'
```

The response is an OpenAI-style chat completion. `usage.prompt_tokens_details.cached_tokens` shows how many prompt tokens came from the prefix cache. Full-block prefix sharing works in units of `kv.block_tokens` (128 by default), so a repeated prompt only reports a non-zero value once its shared prefix covers at least one full block.

Add `"stream": true` to receive server-sent events:

```sh
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"Count to five."}],"max_tokens":64,"stream":true}'
```

## 6. Inspect the server

```sh
curl -s http://127.0.0.1:8000/turbine/v1/status     # the configured support row, modules, kernels, tiers
curl -s http://127.0.0.1:8000/turbine/v1/pressure   # the pressure state and admission document
curl -s http://127.0.0.1:8000/turbine/v1/kv         # KV tiers, blocks per format, cache counters
curl -s http://127.0.0.1:8000/metrics | head        # Prometheus text
```

The full list of endpoints is in the [OpenAI API and diagnostics](../guide/openai-api.md) guide.

## 7. Stop the server

Press Ctrl-C, or send SIGTERM. The server answers 503 `shutting_down` on `/ready` and on new requests, gives running requests `server.shutdown_grace` (30 s by default) to finish, cancels the rest and exits with code 0.

## Where to go next

- [Configuration reference](configuration.md): the keys behind this example.
- [KV cache tiers and compression](../guide/kv-cache.md): enable the host and NVMe tiers.
- [Reliability](../guide/reliability.md): what the server does under memory pressure.
- [Performance](../performance.md): the measured throughput for this model and the vLLM-ROCm comparison, with its caveats.
