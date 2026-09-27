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
4. Optional, AMD: the card's tuned GEMM table `kernels/rocm/tuning/<arch>/gemm.tsv`, generated on the card by the GEMM tuner (see [Tuned GEMM table](#tuned-gemm-table)). Without one every GEMM runs hipBLASLt's first heuristic answer.

## Registry entry

In `crates/turbine-kernels/src/cards/mod.rs`: `mod <name>;`, `pub use <name>::<STATIC>;`, and `&<STATIC>` in `CARD_PROFILES`. Update the pinned names in `crates/turbine-kernels/src/registries.rs` (`registry_conformance::card_profiles`) and the refusal message of `profile_for_device_and_refusal` in `cards::tests` (it lists the registered profiles). The server validates `execution.card_profile` against the registry — no server change.

## Conformance suite

`cards_suite` (`crates/turbine-kernels/src/cards/conformance.rs`): the vendor is a support-matrix vendor; every arch belongs to one profile and is in the vendor's build list (for `amd`, `kernels/rocm/cmake/card_profiles.cmake`, parsed by `cmake_profile_archs`); thresholds and capabilities non-zero; each op listed once with a non-empty, duplicate-free order; row tiers ascending to one final open tier; the `moe_experts` first tier equals `moe_small_max_rows`.

- `scripts/remote-cargo.sh test -p turbine-kernels registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-kernels cards::` — selection, refusal and `cmake_lists_every_profile_arch`.
- `scripts/remote-cargo.sh test -p turbine-kernels cards::tests::tuned_gemm_tables_are_card_data` — every `kernels/rocm/tuning/<arch>/` directory is a profile's architecture and its table's rows are well formed.

## Lab checks

A profile is validated on its own hardware: novanas has only gfx1201, so a new family needs a lab host with that card (ask before using one). There: `scripts/lab-test.sh <host> -- -p turbine-kernels --test hip_ops` (`every_implementation_matches_cpu` against the profile's orders), `-p turbine-model --test tiny_model`, the golden gate for both models at `--concurrency 1` and `--concurrency 16`, and a throughput run to justify every threshold and order. On novanas, `scripts/lab-bench.sh --gpu 0 --model llama` and `--model olmoe` must be unchanged (the gfx1201 profile and the library default must not move).

## Tuned GEMM table

Card data for the HIP library, next to the profile: per GEMM shape `(n, k, trans_b, c_dtype)` and m bucket, the hipBLASLt solution the card runs, and the shape's **mode** (set per shape in `kernels/rocm/tuning/gemm_shapes.txt`, so per model): `invariant` shapes are verified **row-invariant** and of one numerics class (a row's output bits do not depend on how many rows share its batch or where it sits, at any m, so greedy decoding does not change with concurrency; the OLMoE shapes); `speed` shapes pin the fastest solution per bucket with no invariance requirement (the Llama shapes; user decision 2026-09-27, option (c)). `kernels/rocm/cmake/gemm_table.py` compiles every build architecture's `kernels/rocm/tuning/<arch>/gemm.tsv` into the library (`kernels/rocm/src/gemm_table.cpp`); `kernels/rocm/src/gemm.cpp` looks the table up per new shape (by the profile's arch, else the device's) and caches the choice.

- A pinned `invariant` row runs through the hipBLASLt ext API (`hipblaslt_ext::Gemm`) with **split-K off** (`GemmTuning::setSplitK(1)`): by default hipBLASLt splits K for small problems, so the same solution sums a row in another order at m = 1 than inside a large batch. A pinned `speed` row runs `hipblasLtMatmul` with its solution (split-K as hipBLASLt chooses); rows without a pin run `hipblasLtMatmul` with the heuristic's first answer, as before.
- A row pins its solution by the stable hipBLASLt **solution name**, plus the index it had when tuned: the index is used only when it still carries that name, else the name is looked up in the installed hipBLASLt's solution list. A name that is gone logs `event=gemm_table_fallback reason=gemm_table_unavailable` (once per row) and the shape runs the heuristic's first answer; a solution that rejects the call (a leading dimension, the workspace) logs `reason=gemm_table_unsupported`.
- A row serves m in (the previous row's `m_max`, its `m_max`] of its shape; an m above every row takes the largest (rows of one `invariant` shape name members of one numerics class: solutions that give every row the same bits, e.g. a small tile for decode and a large one for prefill). A row with index -1 and the name `heuristic` pins nothing (no solution passed; the tuner reports it).
- `execution.gemm_autotune` (default `true`) is the switch: the server sets the kernel ABI v2.1 option `TURBINE_OPTION_GEMM_AUTOTUNE`, and `false` runs the heuristic's first answer for every shape (the pre-table behaviour, for A/B). `TURBINE_OPTION_GEMM_TUNED_SHAPES` counts the shapes on pinned solutions.
- Regenerate on an idle card of the family (novanas: GPU 0, under the bench lock): build the library (`turbine_gemm_tune` is built beside it), then `turbine_gemm_tune --shapes kernels/rocm/tuning/gemm_shapes.txt --out kernels/rocm/tuning/<arch>/gemm.tsv`. Per `speed` shape and bucket it times the heuristic's solutions at the bucket's bounds and middle (correct and deterministic ones only) and pins the fastest when it beats the heuristic's per-m answers by more than 2 % (else a `heuristic` row). Per `invariant` shape it takes the heuristic's solutions at every bucket of the shape's m list, keeps those that support every m, agree with the heuristic's answer, are bitwise deterministic over three runs and are row-invariant (a target row alone equals the same row at positions 0, 1, 15, 16, 17, m/2 and m−1 of batches at every bucket's bounds and middle), groups them into classes by the target row's bits, times them with weights cycled out of the caches, and pins the class whose fastest member per bucket has the least weighted mean time relative to the heuristic's own per-m answer (`--decode-weight`, default 0.85, weighs the buckets up to m = 64), among the classes at most `--max-prefill-loss` (default 0.05) slower than the heuristic at the served mixed-step sizes (m 1,024 to 2,048), so a pin never buys decode time with TTFT; the file's `# cost` lines record the cost per bucket. A shape with no eligible row-invariant class gets a `heuristic` row and stays batch-variant (the tuner reports it; `hip_batch_invariance` lists such shapes in `PENDING_GEMM_TABLE`). The heavy-tailed target rows matter: on uniform data a BF16 output hides almost every change of summation order, and the check would pass solutions that are not invariant. Add a model's GEMM shapes to `kernels/rocm/tuning/gemm_shapes.txt` first. Regenerate after a ROCm/hipBLASLt upgrade; every changed row changes that GEMM's rounding, so the golden gate (c1 and c16) and a throughput run follow.
- Lab checks: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- gemm_table_matches_cpu` runs every row at the smallest and largest m it serves against the CPU reference and asserts no row fell back; `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_batch_invariance -- gemm_rows_are_batch_invariant` asserts row invariance of the `invariant` (OLMoE) shapes at the served batch sizes.

## Pitfalls

- **CMake list**: the profile's archs and `TURBINE_PROFILE_ARCHS` must be equal sets; `cards::tests::cmake_lists_every_profile_arch` and the suite fail otherwise. Adding the arch also makes every lab build compile for it (longer builds, more Composable Kernel instances).
- **Support matrix**: rows are per `(vendor, arch, architecture)`. With no row for the new arch the server refuses it (exit 2, "no support-matrix row"); add fully specific `supported` rows only after the lab checks passed on that card, and `unsupported` rows with a reason until then.
- **Library default**: `kDefaultProfile` in `kernels/rocm/src/turbine_hip.hpp` is the first profile's values (used by callers that never set a profile). A new profile does not change it; do not reorder `CARD_PROFILES` to put yours first.
- A wave size or LDS size the library was not compiled for makes `turbine_ctx_set_profile` fail at startup; that is the intended refusal — do not relax it.
- **Tuned GEMM table**: timings from a busy or wrong card (GPU 1 on novanas, another tenant's work) produce a table that is worse than the heuristic. Tune only on the idle card and compare `heuristic_us` / `tuned_us` in the file with the served trace.
- Names in `order` must be implementation names the library enumerates (`impl_table.cpp`); a typo silently falls back (`profile_fallback`). Check `kernels` in `/turbine/v1/status` on the card.
