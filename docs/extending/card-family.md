# Adding a card family

A card profile describes one GPU family declaratively: the device architectures it covers, its capabilities, tuned thresholds and the preferred kernel-implementation order per op. It is data, not code: the HIP backend picks the profile of the opened device and the kernel registry reads it; nothing else branches on the architecture. Point name `card_profile`; selected by `execution.card_profile` (`auto` = the profile listing the discovered device arch; a name forces that profile, for experiments). Registered: `gfx1201` (Radeon AI PRO R9700).

## The type

`turbine_kernels::cards::CardProfile` (`crates/turbine-kernels/src/cards/mod.rs`) — a `static`, not a trait implementation:

| Field          | Meaning                                                                                                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`         | The registry and configuration name.                                                                                                                                                                              |
| `vendor`       | The support-matrix vendor column (`amd`).                                                                                                                                                                         |
| `archs`        | `DeviceInfo::arch` values it describes (`gfx1201`); each arch belongs to exactly one profile.                                                                                                                     |
| `capabilities` | `CardCapabilities { matrix_instructions, bf16, wave_size, lds_bytes }`; `turbine_ctx_set_profile` refuses a wave size the library was not compiled for, or too little LDS.                                        |
| `thresholds`   | `CardThresholds { moe_small_max_rows, paged_page_multiple }` — the tuned numbers the library's default path and the backend's notes read; never literals in C++.                                                  |
| `preferences`  | `&[OpPreference { op, order, row_tiers }]`: implementation names in preference order per `OpKind`; `row_tiers` (ascending `max_rows`, last `None`) for `moe_experts`. An op without an entry keeps library order. |

## Files to add

1. `crates/turbine-kernels/src/cards/<name>.rs` (see `crates/turbine-kernels/src/cards/gfx1201.rs`):

```rust
//! `gfx942`: AMD CDNA3 (MI300X). Wave64, MFMA, 64 KiB LDS.
use super::{CardCapabilities, CardProfile, CardThresholds, OpPreference};
use crate::OpKind;

pub static GFX942: CardProfile = CardProfile {
    name: "gfx942",
    vendor: "amd",
    archs: &["gfx942"],
    capabilities: CardCapabilities { matrix_instructions: &["mfma"], bf16: true, wave_size: 64, lds_bytes: 65536 },
    thresholds: CardThresholds { moe_small_max_rows: 256, paged_page_multiple: 128 },
    preferences: &[
        OpPreference { op: OpKind::Gemm, order: &["hipblaslt"], row_tiers: &[] },
        // measured orders only; leave an op out to keep library order
    ],
};
```

2. The architecture in `kernels/rocm/cmake/card_profiles.cmake` (`set(TURBINE_PROFILE_ARCHS gfx1201 gfx942)`): _libturbine_hip.so_ is built for exactly these by default and `kernels/rocm/CMakeLists.txt` refuses a `GPU_TARGETS` entry without a profile.
3. Support-matrix rows for the arch in `crates/turbine-core/src/support.rs` (see Pitfalls).

## Registry entry

In `crates/turbine-kernels/src/cards/mod.rs`: `mod <name>;`, `pub use <name>::<STATIC>;`, and `&<STATIC>` in `CARD_PROFILES`. Update the pinned names in `crates/turbine-kernels/src/registries.rs` (`registry_conformance::card_profiles`) and the refusal message of `profile_for_device_and_refusal` in `cards::tests` (it lists the registered profiles). The server validates `execution.card_profile` against the registry — no server change.

## Conformance suite

`cards_suite` (`crates/turbine-kernels/src/cards/conformance.rs`): the vendor is a support-matrix vendor; every arch belongs to one profile and is in the vendor's build list (for `amd`, `kernels/rocm/cmake/card_profiles.cmake`, parsed by `cmake_profile_archs`); thresholds and capabilities non-zero; each op listed once with a non-empty, duplicate-free order; row tiers ascending to one final open tier; the `moe_experts` first tier equals `moe_small_max_rows`.

- `scripts/remote-cargo.sh test -p turbine-kernels registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-kernels cards::` — selection, refusal and `cmake_lists_every_profile_arch`.

## Lab checks

A profile is validated on its own hardware: novanas has only gfx1201, so a new family needs a lab host with that card (ask before using one). There: `scripts/lab-test.sh <host> -- -p turbine-kernels --test hip_ops` (`every_implementation_matches_cpu` against the profile's orders), `-p turbine-model --test tiny_model`, the golden gate for both models at `--concurrency 1` and `--concurrency 16`, and a throughput run to justify every threshold and order. On novanas, `scripts/lab-bench.sh --gpu 0 --model llama` and `--model olmoe` must be unchanged (the gfx1201 profile and the library default must not move).

## Pitfalls

- **CMake list**: the profile's archs and `TURBINE_PROFILE_ARCHS` must be equal sets; `cards::tests::cmake_lists_every_profile_arch` and the suite fail otherwise. Adding the arch also makes every lab build compile for it (longer builds, more Composable Kernel instances).
- **Support matrix**: rows are per `(vendor, arch, architecture)`. With no row for the new arch the server refuses it (exit 2, "no support-matrix row"); add fully specific `supported` rows only after the lab checks passed on that card, and `unsupported` rows with a reason until then.
- **Library default**: `kDefaultProfile` in `kernels/rocm/src/turbine_hip.hpp` is the first profile's values (used by callers that never set a profile). A new profile does not change it; do not reorder `CARD_PROFILES` to put yours first.
- A wave size or LDS size the library was not compiled for makes `turbine_ctx_set_profile` fail at startup; that is the intended refusal — do not relax it.
- Names in `order` must be implementation names the library enumerates (`impl_table.cpp`); a typo silently falls back (`profile_fallback`). Check `kernels` in `/turbine/v1/status` on the card.
