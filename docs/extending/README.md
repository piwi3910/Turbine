# Extending Turbine

Every pluggable part of Turbine is one file (or directory) plus one entry in a static registry, checked by a conformance suite that runs over the registry itself — adding a module never means adding conditionals across crates. One page per extension point:

| Extension point       | Page                                                 | Registry (`point`)                                                 | Selected by                                         |
| --------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------- |
| Model family          | [model-family.md](model-family.md)                   | `turbine_model::families` (`model_family`)                         | `config.json` `architectures[0]`                    |
| Tool-call format      | [tool-format.md](tool-format.md)                     | `turbine_model::formats` (`tool_format`)                           | `model.tool_call_parser`, else the family default   |
| Weight format         | [weight-format.md](weight-format.md)                 | `turbine_model::weights` (`weight_format`)                         | `config.json` (first format that accepts it)        |
| Kernel implementation | [kernel-implementation.md](kernel-implementation.md) | the kernel library's table (ABI v2.4), ordered by the card profile | the card profile's preference order per op          |
| Card family           | [card-family.md](card-family.md)                     | `turbine_kernels::cards` (`card_profile`)                          | `execution.card_profile` (`auto` = the device arch) |
| Execution backend     | [backend.md](backend.md)                             | `turbine_kernels::backends` (`execution_backend`)                  | `execution.backend`                                 |
| Logits processor      | [logits-processor.md](logits-processor.md)           | `turbine_model::sampling` (`logits_processor`)                     | always; each decides per step whether it applies    |
| Scheduling policy     | [scheduling-policy.md](scheduling-policy.md)         | `turbine_scheduler::policy` (`scheduling_policy`)                  | `scheduler.policy`                                  |
| Eviction policy       | [eviction-policy.md](eviction-policy.md)             | `turbine_kv::policy` (`eviction_policy`)                           | `kv.policy`                                         |

## The registry convention

`turbine_core::registry` (`crates/turbine-core/src/registry.rs`):

- A module implements `Module` (`fn name(&self) -> &'static str`) and its extension point's trait. Names match `^[a-z0-9_]{1,64}$` and are unique per registry.
- A registry is a compiled-in `static Registry<dyn Trait>` (`Registry::new(point, &[&A, &B])`) returned by the owning module's `registry()`. Lookup is by name (`get`, first registration wins), `iter` is registration order (which is meaningful for the logits-processor chain and weight-format detection), and `select(name, reason)` logs `event="module_selected"` with `point`, `name` and `reason`. Nothing is loaded at run time except kernel libraries.
- Configuration names (`model.tool_call_parser`, `execution.backend`, `execution.card_profile`, `scheduler.policy`, `kv.policy`) are validated against the registries before any port is bound; an unknown name exits 2 naming the registered ones. `/turbine/v1/status` reports the chosen module per point under `modules`.
- Each crate's `registry_conformance` tests run the point's suite over the real registry and pin the registered names, so a module registered without passing its suite fails `cargo test --workspace` (`conformance_rejects_broken_module` in `crates/turbine-model/tests/conformance.rs` proves the suites catch broken modules).

## Every page has

- **Files to add** — the new file(s) and where they go, with a skeleton against the real trait.
- **Registry entry** — the exact list to append to and the pinned name lists to update.
- **Conformance suite** — what it checks and the commands to run it (host builds and tests go through `scripts/remote-cargo.sh`, never the workstation).
- **Lab checks** — the GPU suites (`scripts/lab-test.sh novanas`), the per-slice measurement (`scripts/lab-bench.sh --gpu 0 --model llama|olmoe`: golden c1 / c16 and throughput within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md`) and the golden gate a module needs before landing, under the lab rules of `AGENTS.md`.
- **Pitfalls** — what breaks quietly at that point.

`cargo test -p turbine-model --test docs_extending` keeps these pages honest: every page exists with its four sections, and every path and test it names exists in the tree.
