fmt + whole-workspace clippy, 982 tests passed for the first one):

- db51c9b6 `test(kv_gpu): partial-tail reuse judged by the batched-bounds tolerance on HIP
(lead decision 2026-10-05)`
- docs commit: this handoff.

Host-only, one lab Job per run, all runs on novanas GPU 0 (logs `target/tolerance/*.log` in the
worktree; `red-baseline.log`, `green-run.log`, `mutation-run.log`, `host-tests.log`, `gate.log`).
No serve Jobs were started, so nothing to `--stop`; the tp 2 kv_gpu tests did not run (they need
`--gpus 2`, refused under the one-GPU PSU rule).

## The red baseline (unmodified p6b-stack HEAD, one GPU Job, 242 s)

`kv_gpu` on novanas: 11 passed, **4 failed** — `prefix_reuse_matches_cold`,
`prefix_reuse_matches_cold_fp8_kv` (both at the **first run** leg: "prompt 1: first cached 3" —
the first run of a later prompt attaches the _published partial tail_ of the earlier prompt and
starts its prefill mid-block), and `lossy_tier_reuse{,_tq4}` at the **deny leg** ("deny: cached 3"
— the denied run attaches an _exact_ partial tail, no lossy reuse). All four failed on
logprob drift of ~0.01–0.05 (one text flip at a 0.045-margin near-tie), the documented mid-block
warm-prefill drift. `prefix_reuse_suffix_lengths_match_cold` and
`prefix_reuse_partial_block_matches_cold` passed bit-exact on this run and were left
bit-exactness (they are the whole-block-suffix and tail-reuse shape; the lead may want them
tolerance-judged too for flake-proofing — noted, not done, out of my scope).

## The conversion

`crates/turbine-server/tests/kv_gpu.rs` (the file's HIP gating — `require_backend("hip")`
early-returns — is unchanged; every converted assertion is a HIP-only path, and the cpu-backend
bit-exactness tests elsewhere are untouched):

- `prefix_reuse_matches_cold_with` (both `prefix_reuse_matches_cold` and `_fp8_kv`): all three
  runs collect logprobs (`complete_with_logprobs`, `"logprobs": 5` = the gate's `top_k` so the
  margin excuse has a runner-up); first and warm are judged by `assert_within_batched_bounds`
  against the cold reference instead of `(text, completion_tokens)` equality. Kept: completion
  count == 64 and warm `cached_tokens` > 0.
- `lossy_tier_reuse_with` deny leg: `assert_within_batched_bounds(cold, denied)` instead of
  `output()` bit-equality. Kept: `lossy_cached_tokens == 0`, the `turbine_kv_lossy_denied_total`
  metric count, and the whole lossy-reuse leg (codec bounds unchanged).
- The judge is `batched_verdict` + `assert_within_batched_bounds`, mirroring
  `turbine-golden compare`'s batched rule (compare.rs) with the run's own cold logprobs as the
  reference: per-position |Δ logprob| over the identical token prefix against the cold
  logprob's tier of `BATCHED_BOUNDS` (likely 0.25 / tail 0.75 at the −2 floor =
  `tolerance.json`'s `max_abs_logprob_diff_{likely,tail}_batched`), then the divergence must be a
  near-tie (cold top-1 − top-2 margin < `margin_nats` 0.5, read from the **cold** run's
  `top_logprobs` row) or the shared prefix must reach `min_identical_prefix` 32 (clamped to the
  answer length as compare.rs clamps). Unlike compare.rs it judges only the chosen tokens, so a
  re-ordered rank 3–5 is not a violation by itself. Pinned by the new always-run unit test
  `batched_verdict_excuses_only_a_near_tie_divergence` (bounds, margin source, clamp,
  count check) plus the existing `golden_head_bound_splits_likely_from_tail`.
- `IdAnswer` grew `tokens` and `top` (both from the response's `logprobs` object); every other
  request/metric/cached-ratio assertion in the file is unchanged.

## Lab evidence

- Green run (the 4 tests by name, one GPU Job, 100 s): **4 passed, 0 failed**. The verdict lines
  show the tolerance doing exactly its job: first run of prompt 1 identical prefix 14/64 with a
  0.045 cold margin at the divergence (near-tie, excused); prompt 2 first 27/64 (margin 0.10);
  prompt 3 first 59/64 (margin 0.013); warm runs 64/64 with worst |Δ| 0.043–0.15 — all inside
  the batched bounds.
- Mutation check (both lossy tests, one GPU Job, 35 s): with `BATCHED_BOUNDS` tightened to
  1e-6 both go red at "position 0 |Δ logprob| 0.009 over the batched bound 0.000" — the same
  drift — then restored (verified: `git diff` back to the gated 293-line state, fmt clean).

## Plan drift for the lead (not mine to edit)

- `phase-6b-kv-compression.md` line 22 ("the Phase 4 `kv_gpu` bit-exact checks at the `l0` tier
  format do not change") and Task 6's "with `x-turbine-kv-lossy: deny` bit-equal to cold"
  now read too strong for these two tests; the recorded lead decision
  ("5p Tasks 7–8: HIP tolerance for partial-tail prefill tests", test policy) covers the change.
- `phase-5p-serving-efficiency.md` Task 8's lab bullet ("text and every logprob bit-equal to
  cold" for `prefix_reuse_partial_block_matches_cold`) is unaffected — that test still passes
  bit-exact.

## State

Nothing pushed. Tree clean, fmt clean, `cargo fmt --all --check` ok. The always-run kv_gpu host
tests (5) and the whole-workspace gate passed. The tp 2 variants (`*_tp2`, `*_overlap`) and the
NVMe round trips were not run — unchanged assertions, and two-GPU Jobs are refused under the PSU
rule.
