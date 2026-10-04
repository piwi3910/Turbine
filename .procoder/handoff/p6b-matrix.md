# Handoff: p6b-matrix — docs/support-matrix.md, kept true by a test

Branch `p6b-stack` (worktree `agent-p6b-t2`), on top of c4b9a302 (the 6b closing state). One commit:
`docs: the support matrix page, kept true by a test`.

## What landed

- `docs/support-matrix.md` — the human view of `turbine_core::support::SUPPORT_MATRIX`. Covers: how a
  key reads (columns, statuses, refusal = unsupported → exit 2 blaming a config key, most-specific-row
  wins); the two generated gfx1201 tables (Llama, OLMoE — every weight format × KV `bf16` / `fp8_e4m3` /
  `tq4` / `tq2`, cells generated from `support::resolve`); "Every row of the table" (all 47
  `SUPPORT_MATRIX` rows in code order with status and a why); the lower-tier format table
  (`TIER_FORMAT_REFUSALS` + the supported-by-default formats) and the ladder opt-in trade (2026-10-04 A);
  the cpu reference-provider rows; deferred vendors (`nvidia`, the reserved NVFP4/ModelOpt columns);
  the Phase 7 families; parallel refusals (`olmoe_ep_tp_drift`); "Experimental today, and what would
  flip it" with the decision/perf-log pointers; the 6b recorded follow-ups; runtime checks
  (`--support-matrix`, `--check-config`, `/turbine/v1/status`).
- Drift tests in `crates/turbine-model/tests/docs_extending.rs` (the existing docs test target):
  - `docs_support_matrix_rows_match_code` — renders all 47 rows from `SupportRow::view()` and requires
    the page's "Every row" table to carry them in order with the status; an unsupported row's line must
    also name its reason anchor (the `phase-*` track, else the `reason_code:` prefix).
  - `docs_support_matrix_gfx1201_tables_match_resolution` — regenerates the two gfx1201 tables from
    `support::resolve` over every `WeightFormatColumn::ALL` × `KvFormatColumn::ALL` combination.
  - `docs_support_matrix_tier_formats_match_code` — regenerates the lower-tier table from
    `check_tier_format` (`l0`, `fp8_e4m3`, `tq4`) plus `TIER_FORMAT_REFUSALS` (`tq2` experimental).
  - `docs_support_matrix_deferred_and_parallel_match_code` — `DEFERRED_VENDORS` / `PARALLEL_REFUSALS`
    rows with status and reason code.
  - `docs_support_matrix_reason_anchors` — the checker itself (anchor extraction, table comparison).
  Comparison is whitespace-squashed so the prettier table padding does not matter; the page may append
  prose notes after the generated prefix of a row.
- Link from `docs/extending/README.md` (no `docs/README.md` exists; the extending index is the
  established docs hub the test already reads).

## Verification

- `scripts/remote-cargo.sh run -q -p turbine-server -- --support-matrix --output json` (novanas,
  host-only) captured before writing: 47 rows + parallel_refusals + deferred_vendors — the page was
  written from that output, and the tests regenerate the same from code.
- `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` green (all 7 tests in the
  target).
- Mutation checks (each red, then restored):
  1. status edit in the page (`supported` → `experimental` on a full-table row) →
     `docs_support_matrix_rows_match_code` red.
  2. dropped a gfx1201 table row → `docs_support_matrix_gfx1201_tables_match_resolution` red.
  3. dropped a tier-format row → `docs_support_matrix_tier_formats_match_code` red.
- `scripts/gate.sh` ok. `procoder format` clean on the page and the test file.
