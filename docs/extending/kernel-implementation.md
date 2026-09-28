# Adding a kernel implementation

A kernel implementation is one way to run one op (e.g. `moe_experts` on the small-m WMMA kernel) inside a kernel library. The library enumerates its implementations per op through the kernel ABI v2.4 group; at startup the Rust `KernelRegistry` picks one per op config from the card profile's preference order and binds the provider to it, so every call runs exactly that implementation. This page covers the HIP library (`kernels/rocm/`, _libturbine_hip.so_); implementations are not a Rust registry but a C++ table the library exports, and the card profile decides between them.

## The contract

The C ABI `kernels/include/turbine_kernels.h`, v2.4 group (optional; resolved only when `turbine_abi_minor() >= 4` and all five functions exist):

| Function                                 | Meaning                                                                                                                           |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `turbine_impl_count(op)`                 | Implementations of `op` (a `TURBINE_OP_*` code, the order of `OpKind::ALL`).                                                      |
| `turbine_impl_info(op, index, &entry)`   | `{ name, provider, flags }` — static strings; `TURBINE_IMPL_NEEDS_HOST_OFFSETS` when it reads `host_expert_offsets`.              |
| `turbine_impl_supports(op, index, desc)` | 1 / 0 for a descriptor, pointer fields ignored, no context.                                                                       |
| `turbine_impl_run(ctx, op, index, desc)` | Runs exactly that implementation on the compute stream.                                                                           |
| `turbine_ctx_set_profile(ctx, &profile)` | The card profile's thresholds, read by the library's own default when a caller passes no index (the `turbine_<op>` entry points). |

Inside the library each implementation is an `ImplEntry` (`kernels/rocm/src/turbine_hip.hpp`): `name`, `provider` (`hipblaslt`, `ck` or `turbine_hip`), `flags`, `profile_allows` (the thresholds of the default path; `nullptr` = always), `supports`, `run`.

## Files to add

1. The kernel: a new `kernels/rocm/src/<name>.hip` (device code) or `.cpp` (a vendor library call), exposing `bool <op>_supports(const turbine_<op>_desc *)` and `int32_t <op>_run(turbine_ctx *, const turbine_<op>_desc *)` through `kernels/rocm/src/turbine_hip.hpp`; add the file to the source list in `kernels/rocm/CMakeLists.txt` (`CK_SOURCES` as well when it includes Composable Kernel).
2. The table entry in `kernels/rocm/src/impl_table.cpp`: an `entry<...>("<impl_name>", kTurbine, flags, profile_allows)` in the op's `ImplEntry` array **at its library-order position**, and the header comment of that file updated. Index order is library order: it is what a library without a card-profile preference, and the v2 `turbine_<op>` default, runs first.
3. The preference: the name in the op's `order` (or row tier) of every card profile that should prefer it — `crates/turbine-kernels/src/cards/gfx1201.rs`.
4. The pins: the enumerated table of `implementations_enumerated` in `crates/turbine-kernels/tests/hip_ops.rs`, `gfx1201_carries_every_threshold` in `crates/turbine-kernels/src/cards/mod.rs`, and the table of contract §9.1 (`.procoder/contract/interfaces.md`).

```cpp
// impl_table.cpp — a second GEMM behind hipBLASLt, preferred only where the profile says so
template <bool Tiled> struct MyGemm {
  static bool supports(const void *d) { return my_gemm_supports(static_cast<const turbine_gemm_desc *>(d), Tiled); }
  static int32_t run(turbine_ctx *ctx, const void *d) { return my_gemm_run(ctx, static_cast<const turbine_gemm_desc *>(d), Tiled); }
};
const ImplEntry kGemm[] = {
    whole<turbine_gemm_desc, turbine_gemm_supported, turbine_gemm>("hipblaslt", kHipblaslt),
    entry<MyGemm<true>>("turbine_hip_gemm_tiled", kTurbine),
};
```

## Registry entry

No Rust registry lists implementations: the library's table is the source, and `KernelRegistry::build` (`crates/turbine-kernels/src/registry.rs`) orders them by the card profile. The entry that makes an implementation run is its name in the profile's `OpPreference` in `crates/turbine-kernels/src/cards/gfx1201.rs`:

```rust
OpPreference { op: OpKind::Gemm, order: &["turbine_hip_gemm_tiled", "hipblaslt"], row_tiers: &[] },
```

The selection's `reason_code` says why: `profile_preferred` (the first listed name that the library has supports the config), `profile_fallback` (a later one), `library_order` (the profile lists no order for the op), `provider_internal` (no enumeration: the CPU reference, or a library of minor ≤ 3). It is logged as `event="kernel_selected"`, exported as `turbine_kernel_provider_selected{op,provider,impl}` and listed under `kernels` in `GET /turbine/v1/status`.

## Conformance suite

Implementations need the device, so their suite is a lab test: `every_implementation_matches_cpu` in `crates/turbine-kernels/tests/hip_ops.rs` enumerates the library and runs every implementation of every op alone (`turbine_impl_run`) on the op's `hip_ops` shapes it supports, against the `cpu-reference` provider (`crates/turbine-kernels/src/cpu/`) under the op's tolerance. It enumerates, so a new implementation is covered without a test change — as long as some case shape is one it supports (add a shape if none is).

- `cargo test -p turbine-kernels --test hip_ops every_implementation_matches_cpu -- --include-ignored` — run on the lab as `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- every_implementation_matches_cpu`.
- `scripts/remote-cargo.sh test -p turbine-kernels registry_conformance` — the card profiles still name only `OpKind`s with distinct, non-empty orders.
- `scripts/remote-cargo.sh test -p turbine-kernels --test abi_header_neutral` — the header stays vendor-neutral and its minor groups stay declared.

## Lab checks

- `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` (all op tests, `implementations_enumerated`, `every_implementation_matches_cpu`).
- `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model` (`hip_matches_cpu`, `hip_decode_graph_matches_eager`, `hip_v23_library_matches_cpu`).
- If a served choice changes: serve each model and diff `curl -s http://192.168.10.203:18000/turbine/v1/status | jq '[.kernels[] | {op, config, implementation}]'` against `tests/lab/kernel-choices-llama.json` / `tests/lab/kernel-choices-olmoe.json`; update the recording in the same change, with the reason.
- `scripts/lab-bench.sh --gpu 0 --model llama` and `--model olmoe`: golden c1 / c16 pass, tok/s ≥ 0.97× and TTFT p50 ≤ 1.10× the last row of `.procoder/perf-log.md`; append the row.

## Pitfalls

- **ABI minor groups**: a new implementation of an existing op needs no header change — enumeration carries it. A new op, a new descriptor field or a new flag bit is an additive **minor group**: bump `TURBINE_ABI_MINOR` in `kernels/include/turbine_kernels.h`, declare the symbols as optional, resolve them only when `turbine_abi_minor()` is high enough and the whole group exists (`V21Symbols::resolve` in `crates/turbine-kernels/src/ffi.rs`), extend the stub in `crates/turbine-kernels/stub/stub_shim.c` (a new `-DTURBINE_STUB_V<nn>` variant in `crates/turbine-kernels/build.rs`; the previous variant keeps reporting its own minor) and `crates/turbine-kernels/tests/abi_header_neutral.rs`. `TURBINE_ABI_VERSION` (major) never changes for this.
- **A new op** (e.g. the v2.6 `row_sumsq` / `rmsnorm_sharded`) is appended to `OpKind::ALL` — its position is its `TURBINE_OP_*` code, so existing codes never move — with `OpKind::abi_minor` naming its group (a library of an earlier minor is never asked for the code), an `OpConfig` variant, a family trait with a `KernelProvider` accessor that defaults to `None` (`crates/turbine-kernels/src/ops/mod.rs`), the cpu-reference implementation (`crates/turbine-kernels/src/cpu/`), the shim provider (`crates/turbine-kernels/src/shim.rs`), a list in `kOps` of `kernels/rocm/src/impl_table.cpp` and a case in `every_implementation_matches_cpu`.
- **Fallback**: a library without the group must still load and run — the Rust side falls back (to the library's own choice, `provider_internal`, for v2.4). `libturbine_hip_v23.so`, built from the same objects with `kernels/rocm/src/abi_minor.cpp` reporting minor 3, and `hip_v23_library_matches_cpu` prove it; keep both passing.
- **Default path**: callers that pass no index run `default_entry` — the first entry in library order whose `supports` and `profile_allows` accept the descriptor. Putting a new entry first changes the default for every v2.3 caller; thresholds belong in `profile_allows` reading `Profile`, never as literals (no architecture or row-count literal may appear in `kernels/rocm/src`).
- **GEMM algorithms are not implementations**: which hipBLASLt solution a `hipblaslt` GEMM runs per shape is card data, the tuned GEMM table (`docs/extending/card-family.md`, Tuned GEMM table), not an entry of `impl_table.cpp`.
- A name the profile lists but the library lacks is skipped silently (`profile_fallback`); check `/turbine/v1/status` after changing either side.
- Vendor identifiers (`hip*`, `rocm`) stay out of the header and out of Rust public signatures (`crates/turbine-kernels/tests/vendor_neutral_api.rs`); unsafe FFI stays in `crates/turbine-kernels`.
