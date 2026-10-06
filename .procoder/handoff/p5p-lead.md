# Phase 5p lead handoff (2026-10-06)

Integration branch `p6b-stack` (worktree `.claude/worktrees/agent-p6b-t2`), gate ok 1003 at 2ba95514. Local main
is behind; the user pushes. Merge each builder branch here, resolve the append-only decisions.md / perf-log.md
conflicts by keeping both sides (`<<<<<<< x … ======= … >>>>>>> y` → ours + blank + theirs; resolve in its own Bash
call, then `procoder format F > F.formatted && mv`), gate, report, ask every design decision (record it in
decisions.md first, then AskUserQuestion).

## Closed in 5p

Tasks 1–12, 14 (FP8 c1 0.60–0.62× BF16), 15 (MXFP4 re-bound ≤1.45×), 16 (OLMoE 609.5 ≥ 608), 17 (slab drain, default
1); b1 attention own kernel (8.96 µs/layer, RoPE fused, ABI v2.12, CK edge at 10 seqs); MoE tiers bitwise again;
tiny_model A1 (CK strict leg + b1 leg); GPU 1 perf-equal, one-GPU rule lifted; FP8 act-quant fused (ABI v2.13).
Llama b1 decode unprofiled 11.81 ms/token = 0.85× of 640 GB/s (traced numbers read low).

## Running builders (branches)

- `p5p-kvfix`: stopgap refusal → L0 ladder + partial reuse bug fix → S-14 close (Task 13) → lazy L0 classes
- `p5p-o2`: tiny_model b1 bound 0.12 → OLMoE prefill gate-up k step 128 without spill (c16 TTFT back to ~128 ms)
- `p5p-gemv-eval`: fused batch-1 GEMV provider evaluation + proposal (decision B, no kernel yet)
- `p5p-overlap`: overlap scheduling, plan Task 4 (decision C)
- `p5p-goldtol`: golden tolerances re-derived with engine summation-order noise (decision A + B)

## Next

After those: implement the chosen fused GEMV; then Task 19 (exit tiers: gate --full, lab-test full + 2-GPU leg,
lab-bench --golden16, 10-min soak), merge p6b-stack into main, exit summary to the user, then the Phase 8 spec
(Strata's controller/prompt-lookup design and Paiton's inputs are recorded as candidates).

Rules: one small job per builder, fresh agent per follow-up, replace at ~150k tokens (user 2026-10-06).
MXFP4 3b.qkv sits ~1 % under its 1.45 bound (flaky on a busy host — come back to the user if it flips).
