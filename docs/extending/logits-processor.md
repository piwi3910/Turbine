# Adding a logits processor

A logits processor adjusts one request's FP32 logits row in place before the sampling steps (greedy, or temperature → top-k → top-p and one draw). Processors run as a chain in registry order: `logit_bias` → `repetition_penalty` → `presence_frequency_penalty` → `min_tokens` → `grammar_mask`. Point name `logits_processor`; every registered processor is always in the chain and decides per step whether it `applies`.

## The trait

`turbine_model::sampling::LogitsProcessor: Module` (`crates/turbine-model/src/sampling/mod.rs`):

| Method                                                               | Must do                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)                                             | Unique name, `^[a-z0-9_]{1,64}$`.                                                                                                                                                                                                                    |
| `applies(&ProcessorParams, &ProcessorState)`                         | Whether this step's row changes. Must be `false` for a request that does not ask for the processor (the suite's `neutral` check) and `true` when its request field is set (`applies`). Cheap: it runs every step, also to decide device eligibility. |
| `device_capable()`                                                   | Whether a step it applies to may still be reduced on the device (the P2c fast path). Say `true` only if the device's `logits_reduce` of the raw row picks the same candidates as your host path (the suite's `device` check); otherwise `false`.     |
| `needs_full_row()`                                                   | Whether it reads or writes the whole vocabulary row (informational today).                                                                                                                                                                           |
| `apply(&mut [f32], &mut Touched, &ProcessorParams, &ProcessorState)` | Changes the row. Call `touched.touch(logits, id)` before writing any id (it records the raw value for the reported logprobs and returns the index, `None` past the row). A pure mask that only sets −∞ on ids that are never sampled need not.       |

`ProcessorParams` holds the request fields processors read (built once per request from `SamplingParams` by `ProcessorParams::new`); `ProcessorState` is the step's view (prompt ids, generated counts, step index, EOS/stop ids, the grammar mask).

## Files to add

1. `crates/turbine-model/src/sampling/processors/<name>.rs` — the processor (see `crates/turbine-model/src/sampling/processors/min_tokens.rs`):

```rust
//! `ban_token_zero`: token 0 is set to −∞ when the request asks for it.
use turbine_core::registry::Module;

use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};

pub struct BanTokenZero;

impl Module for BanTokenZero {
    fn name(&self) -> &'static str {
        "ban_token_zero"
    }
}

impl LogitsProcessor for BanTokenZero {
    fn applies(&self, p: &ProcessorParams, _: &ProcessorState<'_>) -> bool {
        p.ban_token_zero
    }
    fn device_capable(&self) -> bool {
        false
    }
    fn needs_full_row(&self) -> bool {
        false
    }
    fn apply(&self, logits: &mut [f32], touched: &mut Touched, _: &ProcessorParams, _: &ProcessorState<'_>) {
        if let Some(i) = touched.touch(logits, 0) {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}
```

2. The request field it reads, when no existing one fits: a field of `SamplingParams` in `crates/turbine-core/src/request.rs` (with its neutral value in `Default`), the same field on `ProcessorParams` and in `ProcessorParams::new` (`crates/turbine-model/src/sampling/mod.rs`), and — to expose it over HTTP — the request body field and its validation in `crates/turbine-api/src/openai/request.rs` plus the mapping into `SamplingParams` in `crates/turbine-server/src/backend.rs`. Struct literals that list every `SamplingParams` field fail to compile until you add yours (e.g. `device_eligibility_matches_main` in `crates/turbine-model/src/sampling/tests.rs`); give them the neutral value.

Run `cargo fmt --all` after adding the file: the skeleton above is not rustfmt-formatted.

## Registry entry

In `crates/turbine-model/src/sampling/mod.rs`: `mod <name>;` and `pub use <name>::<Type>;` inside `pub mod processors`, and `&processors::<Type>` in the `PROCESSORS` list **at its place in the chain** — the list order is the application order. Put a processor that removes candidates before `grammar_mask` (the mask must stay last so it wins over every bias).

Then update two assertions in `crates/turbine-model/src/registries.rs` (`registry_conformance::logits_processors`):

1. the pinned list of names, in chain order, with yours at its position;
2. the assertion that every processor is host-only and needs the full row — relax it to what is true once yours is registered (e.g. a processor with `needs_full_row() == false`).

## Conformance suite

`processors_suite` (`crates/turbine-model/src/conformance/processors.rs`) runs over the registry: `neutral` (does not apply to a default request), `applies` (applies when every processor field is set and a mask is present — **add your field to the suite's `busy_params`**, or `applies` fails), and `device` for a device-capable processor (over four seeded rows the host path's top-5 ids equal the `cpu-reference` `logits_reduce` result of the raw row).

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-model sampling::` — chain order against main (`chain_order_is_mains`), device eligibility, seeded streams.
- `scripts/remote-cargo.sh test -p turbine-model --test conformance` — the suite still rejects broken modules.

## Lab checks

A processor that never applies to existing requests changes no served output, but the sampler is on every step's path: run `scripts/lab-bench.sh --gpu 0 --model llama` and `scripts/lab-bench.sh --gpu 0 --model olmoe` (golden c1 strict and c16 batched plus the fixed bench; within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md`) and append a row there. If you exposed a request field, also run `scripts/lab-test.sh novanas -- -p turbine-server --test tiny_server` (seeded requests and `device_sampling_matches_host`).

## Pitfalls

- **Seeded draw order**: processors draw no random numbers. Every draw stays in `Sampler` (`crates/turbine-model/src/sampling/sampler.rs`), one uniform per step in step order, so an identical `seed` gives an identical stream. A processor that consumed the RNG, or changed whether a step draws, breaks the seeded-stream tests and the `tiny_server` seeded requests.
- **Device eligibility is derived**: a step leaves the device fast path when any processor that applies to it is not device-capable (`ProcessorChain::device_eligible`). An `applies` that returns `true` for requests that did not ask for it silently moves every step to the host and costs throughput, even if the row is unchanged.
- **Touch before write**: logprobs are reported from the raw row. Writing an id without `touched.touch` first reports the adjusted value; touching after an earlier processor keeps the first (raw) value, which is correct.
- Reading `s.mask` for anything but presence is the grammar mask's job; keep processors independent of each other.
