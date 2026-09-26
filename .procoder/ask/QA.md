# Questions procoder cannot answer for you

Written 2026-09-26 04:03 UTC.

Answer each one by writing a line beginning `Answer: ` under it, then
hand the file back with `procoder ask --file .procoder/ask/QA.md`.
Leave the `Key:` lines alone — they are what ties an answer to its question.

## Q1: [decision] decisions.md

Key: 46deae046b91
Question: Focus: Phase 1 first, or keep running later phases ahead in parallel?

- Phase 1 first: start T14 (sampler) and T17 (server wiring) now against T13's interfaces; let the running run-ahead agents finish but start no new later-phase work until Phase 1 is merged and pushed (recommended)
- Phase 1 only: also stop the running later-phase agents now
- Keep going as now: Phase 1 plus run-ahead in parallel

**Answer (2026-09-26):** Phase 1 first — T14 and T17 start now in parallel with T13; running run-ahead agents finish, no new later-phase work until Phase 1 is merged to main and pushed.

**Applied (2026-09-26):** the user's P2b decision "tiny test checkpoints head_dim 128 only" is applied to the AMD/HIP GPU executor test too (`hip_matches_cpu` failed with NoProvider for head_dim 16; CK FMHA supports head_dim 128 per spec S-7). CPU-only tiny tests keep head_dim 16.

Answer: Phase 1 first (user chose the recommended option).

## Q2: [decision] decisions.md

Key: 47684a7c4dcf
Question: Golden logprob bound (0.15 on all top-5 candidates is below the BF16 noise floor: HF-vs-HF BF16 variants differ up to 0.38)

- Keep 0.15 but only for candidates with logprob > −2; tail candidates (< −2) get a looser bound of 0.55 (Turbine max: 0.087 likely / 0.513 tail) (recommended)
- Raise the bound to 0.55 for all top-5 candidates
- Full-FP32 HF reference with a 0.3 bound

Answer: 0.15 on candidates with reference logprob > −2, 0.55 on tail (user chose the recommended option).

## Q3: [decision] decisions.md

Key: 3aa51b8da042
Question: Golden reference: regenerate the committed Llama reference with FP32 final logits?

- Yes — commit reference.fp32-logits.jsonl as the golden reference (removes BF16 tie-induced token splits; 3/16 → 9/16 pass at 0.15) (recommended)
- No — keep the BF16-logit reference

Answer: Use the FP32-logit reference (user chose the recommended option).

## Q4: [decision] decisions.md

Key: 4b0c4027dbba
Question: Tiny-model HIP-vs-CPU bound (2e-2) vs CK's BF16 softmax probabilities (measured 0.097)

- Make the CPU reference attention round softmax probabilities to BF16 like CK and HF sdpa, then keep a tight bound (recommended)
- Raise the bound to 0.2

**Answers (2026-09-26):** golden reference = FP32-final-logit HF reference; bound = 0.15 for candidates with reference logprob > −2 and 0.55 for tail candidates (< −2); CPU reference attention rounds softmax probabilities to BF16 like CK/HF sdpa, tiny HIP-vs-CPU test keeps a tight bound.

Answer: Make the CPU reference attention round probabilities to BF16 like CK (user chose the recommended option).
