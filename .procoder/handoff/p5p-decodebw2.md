# Handoff: Phase 5p decode bandwidth II (`p5p-decodebw2`)

Branch `p5p-decodebw2`, from `p6b-stack` b8894df7. Tip `16af9991` + work in progress. Everything
committed is gate-clean. Traces under `target/decodebw/` (base4/ksplit2 = OLMoE A/B,
ksplit = Llama; op bench dump `target/decodebw/op-timings.txt`). Profile command:
`scripts/lab-decode-profile.sh --model <m> --label <L>` (now copies rocprofv3's host-named CSV
dir), then `python3 scripts/decode-profile-report.py target/decodebw/<L>-<m> --weights <B>`
(OLMoE 2 357 836 608, Llama 6 431 025 664).

## Landed (2026-10-05)

**K-split small-m MoE GEMV** (e501123a; decision entry "5p decode bandwidth II", 16af9991).
Each wave now owns 16 columns AND a 1/8 k slice; the block sums the 8 partials through LDS in
ascending slice order; 8x the waves. Numerics class of its own: within the small-m tier rows are
bitwise batch-invariant (verified 8-512 routed rows); across `moe_small_max_rows` the spread is
1 BF16 ulp (max_abs 0.125, 377/133,120 outputs at 520 rows); the invariance test compares
within-tier bitwise and cross-tier against 0.25 (`Outcome::bounded`, mutation-checked at bound 0).

| OLMoE b1 (novanas GPU 0)      | base    | landed                      |
| ----------------------------- | ------- | --------------------------- |
| traced step (median)          | 7.95 ms | 6.96 ms                     |
| effective bandwidth (2.36 GB) | 297     | **339 GB/s (0.53x of 640)** |
| served c1 tok/s               | 136.3   | **156.8**                   |
| c16 tok/s                     | 630.0   | 624.4 (bound 0.97x: ok)     |
| golden c1                     | PASS    | PASS                        |

Expert GEMVs now ~610 GB/s served (gate_up 109 us, down 55 us per layer). Op-level
(cache-cycled) moe_experts: 264.5 -> 201.7 us/layer at 8 rows; 575-607 GB/s at 128-512 rows.

## The rest of the task, measured (profiles at 16af9991 + base4)

**Llama c1** (228 steps): 13.09 ms span, busy 11.78, gaps 1.32; 491 GB/s (0.77x). Kernel map
(the trace's Cijk names truncate; disambiguated by duration):
gate_up 159.5 us (629 GB/s), down 84.9 (592), qkv-fused 55.3 (566), o_proj ~33.8 (551) --
all four AT their tuned gemm-table state (the tsv comment's per-m costs are real; no
gemm_table_fallback in the server log; the m<=16 o_proj row MT16x16x128 = the 34 us kernel).
LM head 1 244 us (633). CK attention 21.7 us/layer incl. merge (KV ~3 MB at ctx 768 -> ~140 GB/s,
launch-bound). rope 2.9 + append 2.9 + silu 2.9 us/layer. Residual add is already fused into the
CK rmsnorm2d (one add_kernel per step).
**Llama GEMV bandwidth is 604 GB/s (94%) already; the whole remaining deficit is gaps (1.32 ms,
~300 kernels x ~3.6 us device-side dispatch/drain inside the single graph replay) + attention
(0.6) + small kernels (0.25). 90% (576 GB/s = 11.16 ms) needs gaps AND attention AND fusion --
essentially everything.**

**OLMoE c1** (424 steps, post-K-split): 6.96 ms span, busy 5.75, gaps 1.21; qkv 54.6 us
(462 GB/s), o_proj 32.6 (258), router GEMM 10.5 (launch-bound), moe_topk 11.2 us/layer
(the libstdc++ introselect port runs on one thread after a parallel softmax), group 2.2,
positions 1.6, scatter 5.2, rmsnorm (Turbine, n=2048 below the CK bucket) 6.7 us x34,
attention ~27.5 us/layer. Headroom qkv+o+router ~0.55 ms/step, but:
**the OLMoE gemm shapes have speed rows LOCKED OUT by design -- only `invariant` rows exist
(kernels/rocm/tuning/gemm_shapes.txt) because OLMoE requires every step's GEMM rows
batch-invariant (hip_batch_invariance `ANY_STEP`, bitwise at m = 1..2048). An own GEMV
replacing them at small m would break row invariance across the m boundary (a full-GEMM project
to cover every m); the invariant hipBLASLt rows are already the tuner's best per-class pick.**
So the gemm-side headroom is not actionable without reopening the OLMoE invariance design.

## Remaining items, sized (my assessment)

1. **Routing-chain fusion (OLMoE, ~0.2-0.3 ms, bit-exact)**: fuse `moe_small_positions` into
   `moe_group` (or into the scatter; pos feeds the scatter only -- verify) and drop the
   scatter's zero-add if it exists. Integer/fixed-order ops -> bit-exact, no golden class
   change. Saves ~2 launches/layer + ~4 us kernels = ~11 us/layer x16.
2. **rope+append fusion (Llama + OLMoE, ~0.18 ms, bit-exact)**: rope (2.9 us) writes q/k,
   paged_append (2.9) then copies k/v into the pool: one kernel, same math, saves a launch +
   boundary + intermediate traffic per layer.
3. **b1 attention (both models, ~0.4-0.6 ms, numerics class)**: CK splitkv is launch-bound at
   ~140 GB/s effective. Reuse-first evaluation owed (llama.cpp RDNA4 decode kernel, CK group
   tile variants); an own b1 kernel would be a new numerics class (golden tolerance +
   invariance amend, as with the K-split).
4. **The gaps are the floor**: ~3.6 us per dependent-kernel boundary inside ONE graph replay
   (device dispatch/drain, not host overhead -- graphs already elide host work). Only fusion
   shrinks the boundary count. At ~300 kernels/step this is 1.1-1.3 ms on both models.
5. **CK rmsnorm n=2048**: blocked by `rmsnorm_sharded` bitwise TP constraint (user; untouched).
6. moe_topk's serial selection (11.2 us/layer): the port IS the semantics (torch.topk
   ties); parallelizing faithfully is high-risk -- left alone.

**Floor estimate if 1-3 land**: Llama 13.09 - 0.18 - 0.45 - (gap savings ~0.1) ~ 12.3 ms ->
522 GB/s (82%). OLMoE 6.96 - 0.25 - 0.4 (attention) ~ 6.3 ms -> 374 GB/s (58%). The ~90 %
target (576 GB/s) is out of reach on both models without eliminating the per-kernel gaps
(large-scale megakernel fusion) or the blocked rmsnorm/attention-projection items; the binding
constraints are the gap floor, OLMoE's all-steps GEMM invariance, and the `rmsnorm_sharded`
bitwise TP constraint.

## Next

- Items 1-2 are small bit-exact landings (tests: moe route/group/positions tests already cover
  the semantics; invariance suite must stay bitwise-green).
- Item 3 needs the reuse evaluation recorded in decisions.md before any new kernel.
- c16 + golden c1 after every landing (lab-bench.sh --quick; lab-decode-profile.sh for the
  trace). Tests first, gate-clean commits, delete target/debug after each Rust commit.
- Scratch scripts target/measure*.sh (uncommitted). Bench/port locks via the scripts; labbook
  rows recorded for the two BENCH runs.
