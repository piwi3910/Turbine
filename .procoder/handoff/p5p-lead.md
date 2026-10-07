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

## PAUSED (user, 2026-10-07) — resume points

`p6b-stack` gate ok 1012 at a7723db4 (+ 66d3c930 docs). Merged since the last handoff: KV fixes (classed-pool copy,
S-14, lazy L0), circuit C, golden bounds applied (A), O2 (OLMoE c16 TTFT 128.7 ms), MoE tile (no win, recorded),
sigterm drain race fix, implementations_enumerated, GEMV stage A (wvSplitK, Llama c1 11.3 ms) and stage B (silu·mul
fused, ABI v2.14, 11.23 ms/token, 87.6 tok/s = ~0.89x). Nothing of ours runs on novanas.

- `p5p-gemv-c` (worktree agent-a97948d81b4f7037a): WIP ad92721e, ABI v2.15 residual-add epilogue + rmsnorm prologue,
  never measured, decoder does not compile (two TP call sites, handoff p5p-gemv-c.md). Next: fix, kernel-level
  timing go/stop, then served A/B.
- `p5p-overlap2` (worktree agent-af6d74d093fcacc3f): a9f7779a (admit during the serial device wait, not the cause)
  - WIP a6457c4e (after a finish the next turn runs serially — met the 1.10x TTFT rule in short runs, +1.3/+4.8 %
    tok/s; hangs `overlap_failed_steps_fail_their_requests_only`). Needs the hang fixed, a clean A/B, and a decisions
    entry for the new S-3 rule (user approval) before any default flip. Handoff p5p-overlap2.md.

## Next

After those: implement the chosen fused GEMV; then Task 19 (exit tiers: gate --full, lab-test full + 2-GPU leg,
lab-bench --golden16, 10-min soak), merge p6b-stack into main, exit summary to the user, then the Phase 8 spec
(Strata's controller/prompt-lookup design and Paiton's inputs are recorded as candidates).

Rules: one small job per builder, fresh agent per follow-up, replace at ~150k tokens (user 2026-10-06).
MXFP4 3b.qkv sits ~1 % under its 1.45 bound (flaky on a busy host — come back to the user if it flips).
