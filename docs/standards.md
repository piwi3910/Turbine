# Engineering standards

These rules say how code in this repository is shaped, so that two modules of the same kind look the same. They add to the binding engineering rules in `AGENTS.md` (spec §21) and to the extension-point pages under `docs/extending/`. New code follows them; existing code is brought in line by the issues labelled `standards`.

When a rule here and a page under `docs/extending/` disagree, fix whichever is wrong in the same change.

## 1. Extension points

Every pluggable kind of module (model family, tool format, weight format, KV format, kernel implementation, card profile, backend, logits processor, scheduling policy, eviction policy, speculative proposer, reasoning parser, collective, transport, DP router policy, device discovery) has the same shape.

- **Trait.** `pub trait <Point>: Module + Send + Sync`. `Module::name()` returns the registry name (snake_case, stable, used in config and in `/turbine/v1/status`). There is no `id()`, `kind()` or other spelling for the name.
- **Capability check.** A module that can refuse a configuration implements `fn supports(&self, ctx: &<Point>Ctx) -> Result<(), Refusal>`. It is always called `supports`, never `check_config`, `applies` or `validate`. `Refusal { code: ReasonCode, detail: String }`.
- **Capabilities are data, not names.** Callers never compare `module.name()` with a string literal to decide behaviour. If a caller needs to know something about a module (it is a MoE FFN, it uses plain attention, it supports tensor parallelism), the trait exposes it as a method or a capabilities struct.
- **Registry.** One `pub static REGISTRY: Registry<&'static dyn <Point>>` per point, plus `pub fn registry() -> &'static Registry<…>` and `pub fn lookup(name) -> Result<…, Refusal>`. A registry is never a bare slice.
- **Selection.** Each point says in its docs page whether it is selected by name (config key) or by probe (first module whose `supports` accepts). Probe order is never load-bearing: two modules whose `supports` both accept the same input is a conformance failure.
- **No closed enums over implementations in turbine-core.** Config holds the registry name as a validated string (or a newtype around it). The server never matches on an enum of implementations; it calls through the trait. Adding a module never edits turbine-core or turbine-server.
- **Conformance.** `<point>/conformance.rs` in the owning crate runs the shared suite over every registered module; the crate's `registry_conformance` test calls it.
- **Docs.** `docs/extending/<point>.md` has the fixed headings _Trait_, _Files to add_, _Registry entry_, _Conformance suite_, _Lab checks_, _Pitfalls_, and a `Registered:` line listing the names. `tests/docs_extending` checks the paths and that the `Registered:` line equals the registry.

The support matrix (`turbine_core::support`) stays central on purpose: it is the one table reviewers read to know what is supported. A new family adds rows there and nothing else in turbine-core.

## 2. Reason codes and errors

- **One type.** Every machine-readable refusal or failure code is a `turbine_core::reason::ReasonCode`, defined once in a central table with its exit code and the contract section that names it. Codes are `snake_case`, prefixed by the point or subsystem (`kv_tq2_l0_refused`, `speculative_budget`, `quant_scheme_unsupported`).
- **Carried as a field.** A code travels as a field on the error (`Refusal`, a thiserror variant), never as the first word of a message string. Tests assert on the field, never with `contains`.
- **Contract parity.** A test checks that every code in `.procoder/contract/interfaces.md` exists in the table and every table entry is in the contract. A code the contract or `AGENTS.md` promises must be emitted.
- **Typed errors.** Each library crate has a `thiserror` error enum (`<Crate>Error`). `Result<_, String>` does not cross a module boundary. turbine-server has a `ServerError` that maps to the exit codes.
- **Startup refusals** are logged as `event="startup_refused" reason=<code>` before the process exits, in addition to the human message on stderr.

## 3. Model families

- **Required:** `hf_architectures`, `parse_config`, `weight_slots`, `decoder_spec`, `write_tiny`. Everything else (`build_executor`, `requirements`, `workspace_bytes`, `tp_decoder_spec`, default tool and reasoning formats, default RoPE theta) has a default on the trait and is overridden only when the family differs.
- **Refusals are explicit.** A family that cannot run under TP, EP or PP says so with a `Refusal` from its own method; the parallel planners do not keep hand-typed lists of family names.
- **Shared helpers** (dense and MoE slot builders, common tensor-name tables) live in `families/common.rs`, not in another family's file. A family file holds only what is specific to that family.
- **Tests per family** come from one harness: every registered family gets the tiny-model CPU test, the TP-equals-TP1 host test, a trace fixture and a golden directory. A family without one of them is listed in the harness with the reason.

## 4. Weight formats, KV formats, tool and reasoning formats

- Shared parsing helpers live in `<point>/common.rs` (`weights/common.rs`, `tools/common.rs`). No helper is copied into a second implementation file.
- Packaging names (`"openai_mxfp4"` etc.) are constants on the implementation, never string literals in callers.
- Quantization-scheme decisions belong to the weight format (`supports`, and a capabilities struct the decoder reads), not to `match QuantScheme` in the executor.
- The L0 KV dtype is selected through the same KV-format registry as the lower tiers; a new L0 format adds one file.
- A tool format documents whether it parses whole-text or streams; the trait states the contract both kinds follow.

## 5. Kernel library (C ABI and HIP)

- **Layout.** `kernels/rocm/src/<family>/` per kernel family (gemm, attention, moe, norm, kv, copy, quant, collective). A file stays under about 800 lines and does not mix host dispatch with device code.
- **Registration.** One `ImplEntry` idiom in `impl_table.cpp`: a stable string id, a provider enum value (`ck`, `hipblaslt`, `vllm`, `llamacpp`, `turbine`), and a `supports` function. An implementation never switches to another algorithm inside its `run`; each variant is its own entry. `supports` predicates of entries for the same op are disjoint, so table order never decides correctness.
- **Launch.** Every launch goes through one `TURBINE_LAUNCH(...)` macro that checks the error and maps it to a `TURBINE_E_*` status. No raw `hipError_t` return, no uncommented `(void)hipGetLastError()`.
- **Status codes.** Add `TURBINE_E_VERSION` and `TURBINE_E_WORKSPACE` beside the existing ones; a new failure class gets a code, not a reused one.
- **Shared device code** (bf16 load/store, warp and wave reductions, WMMA constants, arch traits) lives in `kernels/rocm/include/device_common.hpp` and `arch_traits.hpp`. Tile sizes are named constants per arch.
- **ABI.** Functions are `turbine_<family>_<verb>`; argument order is (handle, descriptor, inputs, outputs, workspace, stream). Every new descriptor starts with `uint32_t struct_size`. Each entry point carries `@since <major>.<minor>` in `turbine_kernels.h`; the Rust version gate is generated from (or tested against) those tags. `TURBINE_ABI_MINOR` sits beside `TURBINE_ABI_VERSION`.
- **Provenance.** A ported or adapted file starts with a fixed header: upstream project, path, commit, licence, what changed. A CI check matches the headers against `third_party/LICENSES`.
- **Rust side.** `crates/turbine-kernels/src/ffi/<family>.rs` and `shim/<family>.rs`; the crate sets `#![deny(clippy::undocumented_unsafe_blocks)]`.
- **Tools and tuning.** Each `kernels/tools/<name>/` has a README with how to build and run it; tuning tables are `kernels/tuning/<arch>/<family>.tsv`.

## 6. Size, structure and bounds

- A production source file stays under about 1,500 lines and a function under about 150. Over that, split along a real seam (build / run / graphs; per tier; per request phase). Tests can live in a sibling `tests.rs` or under `tests/`.
- GPU- or weights-dependent tests live under `tests/`, never in `src/`.
- Every queue, channel and buffer documents its bound where it is declared. An unbounded channel carries a comment naming what bounds it in practice; a per-client buffer has an explicit cap.
- Library code does not `println!`; it logs through `tracing` with an `event=` field.

## 7. Contract parity checks

The contract (`.procoder/contract/interfaces.md`) is the source of truth for cross-phase names. CI checks:

- every registered metric is in the contract and every contract metric is registered;
- the crate dependency graph matches contract §1.2;
- every `TURBINE_TEST_*` environment variable read by a test is documented in `AGENTS.md`, and tests read them through the shared test-env helper, not `std::env::var`;
- every reason code (§2 above).
