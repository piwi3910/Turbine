# OpenAI API and diagnostics

Turbine serves the OpenAI chat and completions routes with server-sent event streaming. Diagnostics live under `/turbine/v1/*`, separate from the OpenAI routes, and the operational endpoints `/health`, `/ready` and `/metrics` sit at the root.

## Endpoints

| Method and path                | Purpose                                                                                                                                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/chat/completions`    | Chat completions, with or without streaming, tools and structured output.                                                                                                                    |
| `POST /v1/completions`         | Text completions.                                                                                                                                                                            |
| `GET /v1/models`               | The served model id.                                                                                                                                                                         |
| `GET /health`                  | Liveness: the process answers.                                                                                                                                                               |
| `GET /ready`                   | Readiness: 200 when the server accepts requests, 503 with a reason otherwise.                                                                                                                |
| `GET /metrics`                 | Prometheus text exposition. See [Observability](observability.md).                                                                                                                           |
| `GET /turbine/v1/status`       | The status document: configured support row, resolved modules and kernels, quantization, KV tier formats, pressure, speculation.                                                             |
| `GET /turbine/v1/devices`      | The device inventory and the telemetry Turbine reads from it.                                                                                                                                |
| `GET /turbine/v1/kv`           | The KV document: tiers, blocks per format, cache and transfer counters.                                                                                                                      |
| `POST /turbine/v1/kv/prefetch` | Queues promotions of a prefix to a faster tier. The request body is defined in [`crates/turbine-api/src/kv.rs`](https://github.com/piwi3910/Turbine/blob/main/crates/turbine-api/src/kv.rs). |
| `GET /turbine/v1/pressure`     | The pressure document: the current pressure state and the admission state.                                                                                                                   |
| `GET /turbine/v1/scheduler`    | The scheduler state.                                                                                                                                                                         |
| `GET /turbine/v1/topology`     | The device and link topology used for placement.                                                                                                                                             |

The `/turbine/v1` routes are Turbine's own. They are not part of the OpenAI API and may grow new fields.

## Readiness

`GET /ready` answers:

- `200 {"ready": true}` when the server accepts requests.
- `503` with `loading_model` while the model loads and warms up.
- `503` with `circuit_open` after a device error, until the circuit breaker's probes pass.
- `503` with `shutting_down` after SIGINT or SIGTERM.

## Chat and completion requests

Requests use the OpenAI field names. The features listed below are supported on `/v1/chat/completions` and `/v1/completions` unless noted.

- **Sampling**: `temperature`, `top_p`, `top_k`, `seed`, `n`, `stop`, `stop_token_ids`, `min_tokens`, `ignore_eos`, `max_tokens`. `ignore_eos` is for benchmarks.
- **Logprobs**: `logprobs`, `top_logprobs`, `echo` and `return_tokens_as_token_ids`.
- **Penalties and bias**: presence, frequency and repetition penalties, and `logit_bias`. These run as logits processors.
- **Streaming**: `stream` and `stream_options`. Streams are server-sent events.
- **Tools**: `tools`, `tool_choice` and `parallel_tool_calls`. Formats are described in [Tool calls, reasoning and structured output](tools-reasoning-structured-output.md).
- **Structured output**: `response_format` with `json_object` or `json_schema`, enforced token by token.
- **Reasoning**: `reasoning_effort` for the Harmony format (gpt-oss). Reasoning text appears in `reasoning_content`.
- **Templates**: `chat_template_kwargs` passes variables to the chat template.
- **Scheduling**: `priority` orders a request in the admission queue.
- **Sessions and cache**: `prompt_cache_key`, with the headers below.

Requests larger than `server.max_request_bytes` are refused with 413.

## Turbine headers

| Header                            | Values                      | Effect                                                                                           |
| --------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------ |
| `x-turbine-cache-salt`            | any string                  | Requests share prefixes only with requests that carry the same salt. Use it to isolate tenants.  |
| `x-turbine-session-resume-within` | integer seconds, 1 to 86400 | Keeps the session's cache boosted for this long after the response. Requires `prompt_cache_key`. |
| `x-turbine-session-end`           | `true` or `false`           | `true` drops the session after this response. Requires `prompt_cache_key`.                       |
| `x-turbine-kv-lossy`              | `allow` or `deny`           | `deny` stops this request from reusing lossy cached blocks. Any other value is refused with 400. |

A session header without `prompt_cache_key`, or a malformed session header, is refused with 400 `invalid_session_hint`.

## Usage

`usage` reports the usual token counts, plus two cache fields:

- `usage.prompt_tokens_details.cached_tokens`: prompt tokens served from the prefix cache, on any tier.
- `usage.prompt_tokens_details.lossy_cached_tokens`: the part of `cached_tokens` that came from lossy blocks (FP8 or TurboQuant formats), including L0 TurboQuant pages.

## Errors

Errors carry an OpenAI-style error object whose `code` is one of the snake_case codes below. The HTTP status is fixed per code.

| Status | Codes                                                                                                                                                                                                                                                                              |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 400    | `invalid_request`, `invalid_json_schema`, `invalid_session_hint`, `invalid_session_id`, `invalid_cache_salt`, `unsupported_parameter`, `context_length_exceeded`, `context_exceeds_kv_capacity`, `state_exceeds_capacity`, `template_error`, `tools_not_supported`, `unknown_tool` |
| 404    | `model_not_found`, `session_not_found`, `not_found`                                                                                                                                                                                                                                |
| 409    | `pressure_too_high` (a prefetch refused at ORANGE or above)                                                                                                                                                                                                                        |
| 413    | `request_too_large`                                                                                                                                                                                                                                                                |
| 429    | `queue_full` (admission queue full), `engine_busy`, `prefetch_queue_full`                                                                                                                                                                                                          |
| 500    | `internal_error`                                                                                                                                                                                                                                                                   |
| 501    | `not_implemented`                                                                                                                                                                                                                                                                  |
| 503    | `queue_timeout` (waited longer than `reliability.admission.queue_timeout`), `overloaded` (SURVIVAL), `circuit_open`, `shutting_down`, `model_not_loaded`, `resource_exhausted`, `replica_failed`, `not_leader`                                                                     |
| 504    | `request_timeout` (longer than `server.request_timeout`)                                                                                                                                                                                                                           |

A 429 response for `queue_full` or `engine_busy` carries a `retry-after` header. Clients should back off on 429 and on 503.

`invalid_json_schema` is returned when a schema is larger than `structured_output.max_schema_bytes` or does not compile within `structured_output.compile_timeout`.

Within a stream, a failure after the first token is reported as an event with an error type. The HTTP status of a stream is already 200 at that point.

## Admission and queueing

A request is admitted when its worst-case KV footprint fits the budget. A request that does not fit waits in a bounded queue. The queue has two limits:

- `reliability.admission.max_queue` (256 by default): when the queue is full, new requests get 429 `queue_full`.
- `reliability.admission.queue_timeout` (30 s by default): a request that waits longer gets 503 `queue_timeout`.

`scheduler.max_running_requests` bounds how many requests run at once, and `scheduler.max_queued_requests` bounds the channel between HTTP and the engine. The full model is in [Reliability](reliability.md).

## Models endpoint

`GET /v1/models` lists the one model the server serves. Its id is `model.served_name`, or the last component of `model.path` when that is `null`.

## Example

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-turbine-cache-salt: tenant-7' \
  -d '{
        "model": "llama-3.2-3b-instruct",
        "messages": [{"role": "user", "content": "Summarise the KV tiers."}],
        "max_tokens": 200,
        "stream": true,
        "response_format": {"type": "json_object"}
      }'
```
