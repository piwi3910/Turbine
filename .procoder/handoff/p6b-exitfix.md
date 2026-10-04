# Handoff: p6b-exitfix — the four 6b exit findings

Branch `p6b-exitfix`, from `p6b-stack` `5bec8eea`. Four findings of `.procoder/handoff/p6b-exit.md`,
all landed; gate-clean per commit (`scripts/gate.sh --base 5bec8eea` / `--base 8e4d64af`). Host-only
plus two lab-test Jobs (logs `target/exitfix/` on the workstation); no serve, bench or exclusive
lock of this branch; nothing pushed.

## Commits

- `fix(kv)` `b3e04a98` — **soak `kv_idle`: a cached classed page is not a referenced block.**
  Root cause first: the residual L0 blocks of both failing soaks are the shared prefixes'
  cached copies, legitimately retained and not judged by `kv_idle` (which reads the pressure
  document's kv pool). That document reads `EngineLoop::sync_kv_held` =
  `pool.referenced_blocks() × block_bytes`, and `referenced_blocks` subtracted only the base
  class's cached count — every cached unreferenced page of a non-base class (the ladder's L0
  rung rewrites) counted as a holder. Evidence is exact: kv `used_bytes` was
  1,512,046,592 / 1,130,364,928 = 103.0 / 77.0 whole l0 blocks (14,680,064 B) in runs
  `novanas-20261004T{113806,120705}Z`, constant through the cool-down, reserved 0; run 1 had
  288 L0 `fill_high_water` rewrites, run 2 had 1,038, whose evicted rest explains the
  difference. Run-to-run dynamics decide how many rewritten copies survive, which is why the
  Task 18 soak drained and these did not; tail-tag expiry changed the mix (expired tails no
  longer demote first as raw copies), not the accounting. Fix: `used_blocks() -
cached_unreferenced()`. Pinned by `pool::tests::classed_cached_pages_are_not_referenced`
  and by `kv_sim recent_window_holds_newest_blocks_at_bf16` (idle with a cached BF16 window
  page → referenced 0), both red at the old subtraction — the kv_sim one reproduces the soak
  state on the cpu backend. Criterion unchanged; GPU soak rerun left to the lead.
- `fix(scripts)` `ff855c42` — **phase-7 track gate.** Real-tree check (novanas, the live
  `--support-matrix` text, `target/exitfix/support-matrix.txt`): the order check no longer
  fails — the gate's only remaining line is "phase-7-model-families.md does not exist; write
  it with /procoder:spec", correct before track 2 starts (the spec is written at start);
  `lab_scripts track_gate` proves the full GATE PASS with a stub spec. `closed_rows
kv_compression` accepts any amd tq4/tq2 row in `supported` or `experimental` state; the failure message names exactly
  what is accepted (L0 tq4 rows `experimental`, tq2 an `experimental` lower-tier rung with no
  amd matrix row, L0 tq2 refused; a refused tq4 row still fails; the phase-7 spec owns any
  flip). `lab_scripts track_gate` covers experimental (pass), refused (fail) and supported
  (a later flip passes); mutation back to supported-only fails it. The umbrella plan's state
  paragraph records the amendment.
- `test(kernels)` `8e4d64af` — **the stale hip_ops full-tier tests.** `implementations_enumerated`
  pins today's enumeration read from `impl_table.cpp`: 7 `attention_prefill_paged`, 6
  `attention_decode_paged`, 2 `kv_transcode`. `every_implementation_matches_cpu` gained a
  TurboQuant-page mixed scenario per paged kind (case bodies take the bound implementation's
  name to pick the staged-vs-rotated judgement). Lab: both green (job
  turbine-lab-test-1004135524-16c0c1b2); mutation (dropping `turbine_hip_mixed_staged` from
  the table) red (job turbine-lab-test-1004135817-1627eab1).
- `feat(server)` — **S-10 status keys.** `quantization.tier_formats` (per tier × format
  blocks/bytes, from the kv document) and `quantization.ladder`
  (`KvHierarchy::ladder_document` → `EngineDocs.ladder`) in `/turbine/v1/status`, replica 0's
  documents; `kernels` names the ABI v2.11 `kv_transcode` a lossy tier format selects
  (`reason_code: tier_format`, appended beside the registry's selections, not a selection).
  Pinned by `tiny_server status_reports_tier_formats_and_ladder`; mutation (dropping the
  keys) fails it. Spec S-10 AC ticked; contract and `docs/extending/kernel-implementation.md`
  updated.
- docs — perf log 6b "Exit fixes", plan Task 19 exitfix addendum, spec S-6/S-11 exit lines
  updated (soak diagnosed; hip_ops fixed), this handoff.

## For the lead / user

1. **The cpu provider refuses a BF16-base ladder at warm-up**: with `kv.ladder.enabled` +
   `l0` on a BF16 L0, the fp8 rung class page carries the codec's per-layer scales header
   (`header_bytes` = 8 B/layer) and `cpu/paged.rs`'s class sanity check refuses the mismatch
   ("the pool's 1 class holds 65544 bytes per layer, the codec gives Some(65536)"). The GPU
   path has no such check (Task 18's A/B ran there), so this is cpu-provider-only, but it
   means the ladder's L0 step is not host-servable with a BF16 base until the class page
   model agrees on the header. The S-10 AC test uses an FP8 L0 base (whose classes are
   exact). Small fix options: teach the check the header, or size the class page without it
   and store the scales elsewhere — owner: whoever takes the fp8-KV/window paths.
2. **GPU re-verification still open** (not budgeted here): one ladder soak rerun for
   `kv_idle` (the fix's host evidence is exact-arithmetic + a cpu-backend reproduction), and
   a full `lab-test --tier full` so the exit's "1,051 passed / 2 failed" line can be retired.
3. Unchanged from the exit handoff: `llama-fp8kv` golden (bisect/recalibrate/row decision),
   the ladder-on tail/throughput trade, AGENTS.md's one-GPU PSU rule.

## Tree state

Clean at the docs commit; `procoder spec check phase-6b-kv-compression` and `plan check`
COMPLETE; remote `target/debug` deleted after the last Rust commit. Logs: `target/exitfix/`
(workstation), lab Jobs under `target/lab-test/1004135524-*` and `1004135817-*`.
