# Tool calls, reasoning and structured output

Three features shape what a model returns: tool calls, which the server parses from the model's text into OpenAI `tool_calls`; reasoning content, which the server separates from the final answer; and structured output, which constrains the model's tokens to a JSON schema. Each has a parser or enforcement layer chosen by a registry name.

## Tool calls

Tools are passed as in the OpenAI API, with `tools`, `tool_choice` and `parallel_tool_calls`. The model's text is parsed into `tool_calls` by a tool-call format.

### Formats

| Format        | Used by default for                                               |
| ------------- | ----------------------------------------------------------------- |
| `llama3_json` | Llama (`LlamaForCausalLM`), when its chat template renders tools. |
| `hermes`      | Qwen3 and Qwen3-MoE.                                              |
| `qwen3_xml`   | Qwen3.5 and Qwen3.6 hybrids.                                      |
| `mistral`     | Mistral and Mixtral.                                              |
| `harmony`     | gpt-oss.                                                          |
| none          | OLMoE, which has no tool format.                                  |

Set `model.tool_call_parser` to a format name, or to `none` to disable parsing. `null` (the default) takes the family default above.

```yaml
model:
  tool_call_parser: null # or llama3_json, hermes, mistral, harmony, qwen3_xml, none
```

An unknown name exits with code 2 before the port is bound, and the message lists the registered formats.

### Requests

- A request with `tools` on a model whose template cannot render them is refused with 400 `tools_not_supported`.
- `tool_choice` controls whether the model must, may, or must not call a tool. `parallel_tool_calls` controls whether several calls can come in one turn.
- Tool definitions are JSON. They are bounded by `structured_output.max_schema_bytes` (64 KiB by default, between 1 KiB and 1 MiB).

Tool calls are counted in `turbine_tool_calls`. Format-level refusals and failures are logged with their reason codes.

## Reasoning content

Reasoning models write their chain of thought before the answer. A reasoning parser separates it, so that the chain of thought goes to `reasoning_content` and the answer goes to `content`.

| Parser       | Used by default for                | Notes                                                    |
| ------------ | ---------------------------------- | -------------------------------------------------------- |
| `qwen_think` | Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.6 | Splits the `<think>` section.                            |
| `harmony`    | gpt-oss                            | Reads the Harmony channels. Supports `reasoning_effort`. |
| `none`       | everything else                    | Reasoning content is not separated.                      |

```yaml
model:
  reasoning_parser: auto # auto (the family default), qwen_think, harmony or none
```

`auto` is the default and takes the family's parser. `none` turns `reasoning_content` off for every family.

`reasoning_effort` is a request field for the Harmony format. Reasoning tokens are counted in `turbine_reasoning_tokens`.

## Structured output

`response_format` constrains the output to a JSON document:

- `{"type": "json_object"}` forces any valid JSON object.
- `{"type": "json_schema", "json_schema": {...}}` forces the document to match the schema.

Enforcement uses llguidance, linked into the server. At each step the grammar produces a token mask, and the sampler only chooses among allowed tokens. Grammar compilation time is in `turbine_grammar_compile_seconds`, and the cost of each mask in `turbine_token_mask_seconds`.

### Limits

| Key                                  | Default | Effect                                                                        |
| ------------------------------------ | ------- | ----------------------------------------------------------------------------- |
| `structured_output.max_schema_bytes` | `64KiB` | Largest accepted schema or tool definition. A larger one is refused with 400. |
| `structured_output.compile_timeout`  | `5s`    | Longest grammar compilation. Expiry returns 400 `invalid_json_schema`.        |

A schema that does not compile, or that is too large, is refused before generation starts.

### Jump-forward decoding

When the grammar allows only one next token, the model would produce it anyway. Jump-forward decoding emits such forced tokens as a single verified step, without a forward pass for each.

| Key                                         | Default | Effect                                                                                            |
| ------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------- |
| `structured_output.jump_forward`            | `true`  | Enables the step.                                                                                 |
| `structured_output.jump_forward_max_tokens` | `32`    | Longest forced run in one step, from 1 to 256 and at most `scheduler.max_batch_tokens` minus one. |

Jump-forward steps and tokens are counted in `turbine_jump_forward_steps_total` and `turbine_jump_forward_tokens`. The forced tokens are verified as one multi-token step, so the grammar still decides every token.

## Chat templates

The chat template renders messages, tools and `chat_template_kwargs` into the prompt. Two settings control it:

- `model.chat_template` replaces the checkpoint's template with a file. Use it to pin a template for reproducible output.
- `chat_template_kwargs` in a request passes variables to the template.

A template that fails to render returns 400 `template_error`.

## Example

A request that asks for a JSON answer with a schema:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "llama-3.2-3b-instruct",
        "messages": [{"role": "user", "content": "Give the capital and population of France."}],
        "max_tokens": 200,
        "response_format": {
          "type": "json_schema",
          "json_schema": {
            "name": "country",
            "schema": {
              "type": "object",
              "properties": {
                "capital": {"type": "string"},
                "population": {"type": "integer"}
              },
              "required": ["capital", "population"]
            }
          }
        }
      }'
```

The message content is a JSON object that matches the schema.
