# Adding an execution backend

An execution backend opens the device memory and the kernel providers a model runs on: which kernel library to load and from where, how the configured device is matched, which device errors are sticky, and what to say about the kernels it selected. Everything vendor-specific lives behind it; the server only looks the name up and opens it. Point name `execution_backend`; selected by `execution.backend` (default `hip`). Registered: `cpu` (the `cpu-reference` provider on host memory), `hip` (_libturbine_hip.so_ on an AMD device).

## The trait

`turbine_kernels::backends::ExecutionBackend: Module` (`crates/turbine-kernels/src/backends/mod.rs`):

| Method                                | Must do                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)              | The configuration name; a shim backend also loads `libturbine_<name>.so` (`ShimLibrary::search_paths(name, explicit)`).                                                                                                                                                                                                                                                     |
| `vendor()`                            | The support-matrix vendor column: `cpu` for the host, else the device vendor (`amd`, `nvidia`) — one of `turbine_core::support::VENDORS`.                                                                                                                                                                                                                                   |
| `open(&BackendRequest)`               | Loads the library, finds `req.device` in `req.inventory` (refusing a device of another vendor), picks the card profile (`cards::select(req.card_profile, device)`), creates the context and returns `OpenedBackend { mem, providers, order, memory_kind, context, graphs, device, card }`. Failures are `BackendError::Startup` (exit 1) with a message naming what to fix. |
| `sticky_error_prefixes()`             | Device error names after which the context is corrupted (every later call fails); default none.                                                                                                                                                                                                                                                                             |
| `selection_notes(card, &[Selection])` | Notes the server logs at INFO after kernel selection (HIP: `paged_attention_fallback` when paged attention is not on the profile's first choice); default none.                                                                                                                                                                                                             |

## Files to add

1. `crates/turbine-kernels/src/backends/<name>.rs` (see `crates/turbine-kernels/src/backends/hip.rs`; the host backend `crates/turbine-kernels/src/backends/cpu.rs` is the minimal one):

```rust
pub struct CudaBackend;

impl Module for CudaBackend {
    fn name(&self) -> &'static str { "cuda" }
}

impl ExecutionBackend for CudaBackend {
    fn vendor(&self) -> &'static str { Vendor::Nvidia.as_str() }
    fn open(&self, req: &BackendRequest<'_>) -> Result<OpenedBackend, BackendError> {
        let lib = load_shim(self.name(), req.kernel_library)?;          // libturbine_cuda.so
        let device = nvidia_device(req.inventory, req.device.0)?;       // refuse another vendor
        let card = cards::select(req.card_profile, device)
            .map_err(|e| BackendError::Startup(format!("execution.card_profile: {e}")))?;
        let ctx = lib.create_context(device).map_err(|e| BackendError::Startup(e.to_string()))?;
        ctx.set_profile(card).map_err(|e| BackendError::Startup(e.to_string()))?;
        let provider = shim_provider(Arc::clone(&ctx));
        Ok(OpenedBackend { mem: ctx.clone(), order: vec![provider.id()], providers: vec![provider],
            memory_kind: device.memory.kind, graphs: lib.supports_graphs().then(|| Arc::clone(&ctx)),
            context: Some(ctx), device: Some(device.clone()), card: Some(card) })
    }
    fn sticky_error_prefixes(&self) -> &'static [&'static str] { &["cudaErrorIllegalAddress"] }
}
```

2. For a new GPU vendor, the pieces the backend stands on: a kernel library implementing `kernels/include/turbine_kernels.h` (a `kernels/<vendor>/` directory beside `kernels/rocm/`, built by its own CMake), device discovery for the vendor (a `DiscoveryKind` in `crates/turbine-device/src/discovery/mod.rs`, registry point `device_discovery`, whose `telemetry` method opens the vendor's live telemetry backend in `crates/turbine-device/src/telemetry/<vendor>.rs` for the Phase 3 pressure controller, and whose `topology` method opens its GPU link / peer-access source in `crates/turbine-device/src/topology/vendor.rs` for the Phase 5 topology graph), card profiles with `vendor` set to it (see [card-family.md](card-family.md)), and support-matrix rows.

## Registry entry

In `crates/turbine-kernels/src/backends/mod.rs`: `pub mod <name>;`, `pub use <name>::<Type>;`, and `&<Type>` in `BACKENDS`. Update the pinned names in `crates/turbine-kernels/src/registries.rs` (`registry_conformance::backends`) and `backends::tests::cpu_and_hip_registered`, and the fixed lists in `crates/turbine-core/src/config/tests.rs` if a test names it. `execution.backend` is validated against the registry (exit 2 before bind, `crates/turbine-server/src/modules.rs`); the server opens it through `registry().select` — no server change.

## Conformance suite

`backends_suite` (`crates/turbine-kernels/src/backends/conformance.rs`): the registry has at least one host backend; per backend `vendor` (a support-matrix vendor), `sticky` (prefixes non-empty and distinct), `notes` (no note without selections), and for a host backend `host` (opens on an empty inventory with at least one provider, `order` naming exactly them, no device, card or context).

- `scripts/remote-cargo.sh test -p turbine-kernels registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-kernels --test vendor_neutral_api --test unsafe_isolation` — no vendor type in public signatures, unsafe only in its modules.
- `scripts/remote-cargo.sh test -p turbine-server --test server_cli` — an unregistered `execution.backend` exits 2 before bind; the support-matrix decision per vendor.

## Lab checks

A device backend is validated on its hardware with its own library: the op suite (`hip_ops` or its counterpart, including `every_implementation_matches_cpu`), `-p turbine-model --test tiny_model` (`hip_matches_cpu`-style executor checks and decode graphs), `-p turbine-server --test lab_openai`, and the golden gate for both models at `--concurrency 1` and `--concurrency 16`. NVIDIA work is on hold until everything works well on novanas (decision 2026-09-26): ask the user before any run on `dgx-spark` / `dgx-spark2`. Existing backends must not move: `scripts/lab-bench.sh --gpu 0 --model llama` and `--model olmoe` within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md`.

## Pitfalls

- **Support-matrix rows**: the key's vendor column is `vendor()`. A vendor with no rows resolves to `unsupported` for every model (exit 2 before bind); a new vendor word must also be added to `VENDORS` in `crates/turbine-core/src/support.rs`. Add `supported` rows per `(vendor, arch, architecture)` only after the lab checks.
- **Vendor isolation**: no `Vendor::Amd`, `hipError…` or backend names in `crates/turbine-server`, `crates/turbine-model`, `crates/turbine-scheduler`, `crates/turbine-kv` or `crates/turbine-core` (the Phase 2m acceptance greps for it); vendor error names live in `sticky_error_prefixes`, vendor notes in `selection_notes`.
- **No vendor crate**: GPU runtimes are loaded at run time (`libloading`), never linked; `cargo tree --workspace | grep -Ei 'hip|rocm|cuda'` must stay empty. Unsafe FFI stays in `crates/turbine-kernels` (and discovery in `crates/turbine-device`) with documented pointer and stream ownership.
- The library is ABI-checked on load (major version exact, backend name, build archs against the device): a mismatch is fatal, a missing file moves to the next search path. Keep `execution.kernel_library` → `TURBINE_KERNEL_LIBRARY` → beside the executable → loader path.
- `graphs` must be `None` unless the library exports the v2.1 graph functions; the executor then runs eagerly.
