# Handoff: p6a-gptq-numerics (T18 GPTQ GSM8K drop investigation)

## Rotation 12 close-out, part 2 (2026-09-30 ~12:20Z, builder r12, brief r12-gptq-proof)

Mid-task user instruction (via the lead, Claude usage low): finish only the running
fixture/tolerance step, do not start lab-bench, vLLM, the soak, or the support.rs flip. Handing
off now with everything needed for the next rotation (lead + Codex) to pick up without re-deriving
any of this.

## What is DONE and committed on `p6a-gptq-numerics`

1. Merged `phase-6a-quantization` **twice** (its tip moved mid-rotation): first 96d3b76 (clean),
   then re-merged at **5c5f69b** (also clean, no conflicts) once it turned out 96d3b76 predated
   the `p6a-mxfp4-golden` merge that carries `d64f0cc` (`dequantize_checkpoint.py`'s
   transformers-5 `TokenizersBackend` → `PreTrainedTokenizerFast` fix) — the own checkpoint's
   `tokenizer_config.json` has `tokenizer_class: TokenizersBackend` and needs this fix to be
   dequantized by `quant_reference.py` (confirmed: without it, `AutoTokenizer.from_pretrained`
   raises `ValueError` after decode+dequant succeed). **If you re-merge `phase-6a-quantization`
   again, re-verify `grep _fix_tokenizer_class scripts/golden/dequantize_checkpoint.py` still
   finds it** (do not assume a stale local merge has it).
2. `7127762` test(eval): full-GSM8K results for the own GPTQ checkpoint and AutoRound —
   `tests/eval/llama-3.2-3b-instruct/{turbine-gptq-own-full,vllm-gptq-own-full,
   turbine-gptq-autoround-full}.json` + `*-paired.json`, README updated. Own checkpoint: Turbine
   988/1319 = 0.7491 vs BF16 0.7801, drop 0.0311 ≤ 0.04 **PASS** (McNemar p 0.00222, exact p
   0.00213, 95% CI [+0.0117, +0.0504]); vLLM-ROCm same checkpoint also 988/1319 = 0.7491 (Turbine
   vs vLLM drop 0.0000). AutoRound side result 998/1319 = 0.7566, drop 0.0235 PASS (does not
   decide the row). No circuit transitions in either run. **This decides `gptq_int4`'s
   quality gate** per decision "GPTQ INT4: which better checkpoint" option A.
3. `8ea67bc` feat(lab): the own checkpoint as a proof model —
   `scripts/lab-bench.sh --model llama-gptq-own` (slug `llama-3.2-3b-instruct-gptq-own`),
   `scripts/lab/phase6-novanas-llama-gptq-own.yaml`, `scripts/lab-serve.sh --vllm
   llama-3.2-3b-instruct-gptq-own`, `.procoder/handoff/gptq_own_fixture.sh`.
4. Second merge commit (message: "merge: integration tip phase-6a-quantization (5c5f69b) — brings
   in the TokenizersBackend fix") — clean, no code of mine touched by the resolution.
5. Gate green **after the second merge**: `scripts/gate.sh --base 2c3fbb7` → `gate: ok crates=all
   passed=801 failed=0` (log was `/tmp/gate-r12-gptq-2.log` on the Mac side, not preserved in the
   repo — rerun if you need to see it again; nothing to fix, just re-confirm after any further
   merge).
6. `target/debug` deleted from this worktree per the mid-task instruction.

## What is IN PROGRESS on novanas (unattended; do not restart, just wait for the marker)

`gptq-own-fixture` CPU fixture job, started the **second** time (after re-syncing the fixed
`scripts/golden/` and `tests/golden/` to the remote workspace) at 11:56:30Z. As of this handoff
(12:20Z) the reference pass **succeeded** (`reference rc=0`) and it is running the all-8-variant
`self_spread.py` pass, which was next in the fixture queue after `r12:mxfp4-inc-spread` and
`w4a4-mxfp4-a4-spread` finish holding the CPU (it got the lock at 11:56:30Z, so it's already
running, not still queued).

- Driver: `.procoder/handoff/gptq_own_fixture.sh` (committed at 8ea67bc; also on the host at
  `/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/gptq_own_fixture.sh`).
- Log: `/home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture/gptq-own-fixture.log` on novanas.
  **Wait for the line `gptq-own-fixture: done rc=`** (0 = success). Do not poll tightly; it is a
  single CPU job (3B, weight-only act-quant none) that should finish well under the multi-hour
  budgets of the 8B jobs ahead of it in the queue — check back periodically instead.
- On success it writes `tests/golden/llama-3.2-3b-instruct-gptq-own/{reference.jsonl,spread.json}`
  **on the remote workspace's checkout**
  (`/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/src/tests/golden/llama-3.2-3b-instruct-gptq-own/`),
  not in this Mac worktree — `scp`/`rsync` them down before committing.
- If it fails (`rc=` nonzero), check `/home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture/reference.log`
  and `spread.log` in the same directory first.

## Next step once the fixture lands (rc=0): derive `tolerance.json` and the README

Follow the AWQ precedent exactly (`tests/golden/llama-3.2-3b-instruct-awq/README.md`, "Tolerance
calibration (the OLMoE method)" section — copy its structure):

1. `scp novanas:/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/src/tests/golden/llama-3.2-3b-instruct-gptq-own/{reference.jsonl,spread.json} tests/golden/llama-3.2-3b-instruct-gptq-own/` (mkdir the dir first).
2. Check the reference log's rotary line says `llama3` / theta `500000` (own checkpoint is
   Llama-3.2-3B-Instruct, same rope config as BF16/AWQ/published-GPTQ — should be automatic; the
   brief calls it out explicitly, so verify rather than assume:
   `grep -i "rotary\|rope" /home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture/reference.log` on
   novanas, or re-derive from `quant_reference.py`'s stderr if the grep finds nothing there —
   the rope config print may be in the dequantize step's log instead).
3. `spread.json` has all eight variants (bf16/fp32 × sdpa/eager × incremental/full). For each:
   "prefix ok" count (identical greedy prefix or divergence at a reference margin < 0.5 nats),
   max |Δ logprob| over the reference top-5 before first divergence ("max likely"), and the same
   restricted to divergent positions only isn't separately tracked — follow the AWQ README's table
   format and interpretation exactly; the script's own field names should match.
4. Round the *worst* variant's max likely / max tail up to two decimals; the committed
   `tolerance.json` **must never be tighter than the BF16 Llama-3.2-3B-Instruct bounds**: likely
   0.15, tail 0.55, batched 0.25 / 0.75, `min_prompts_passing` 14 (copy
   `tests/golden/llama-3.2-3b-instruct/tolerance.json`'s other keys unchanged) — if the measured
   spread is inside those bounds (expected: GPTQ int4 weight-only, same shape as AWQ, whose spread
   was well inside), just copy the BF16 file; only widen a bound if a variant's spread genuinely
   exceeds it.
5. Write `tests/golden/llama-3.2-3b-instruct-gptq-own/README.md`: checkpoint provenance (already
   in `.procoder/handoff/p6a-gptq-numerics.md`'s "Do 3" history and in
   `tests/eval/llama-3.2-3b-instruct/README.md`), the exact `quant_reference.py` /
   `self_spread.py` commands run (from `gptq_own_fixture.sh`), the 8-variant table, and the
   tolerance derivation, mirroring the AWQ README's structure and wording.
6. Commit `reference.jsonl`, `tolerance.json`, `README.md` (not `spread.json` — matching the AWQ
   precedent, where it stays a working artifact under the fixture's scratch dir, not the repo).

## Do NOT start yet (per the mid-task instruction) — left fully specified for the next rotation

### Golden c1/c16 + throughput bench

Once the golden fixture above is committed:
```
scripts/lab-bench.sh --model llama-gptq-own --label r12-gptq-own --golden16
```
(background; takes `bench.lock`; builds release `turbine-server`/`turbine-bench`/kernels on
novanas via `scripts/remote-cargo.sh`, serves GPU 0 with
`scripts/lab/phase6-novanas-llama-gptq-own.yaml`, runs golden c1 always + c16 via `--golden16`,
then the fixed throughput bench — 16 concurrent, 512-word prompts, 256 tokens, 200 requests).
Judges: exits 1 if tests/golden c1/bench fail; results and the `BENCH` line land in
`target/lab-bench/r12-gptq-own-llama-gptq-own/`.

### vLLM-ROCm throughput, same checkpoint, "for the record"

```
scripts/lab-serve.sh novanas --vllm llama-3.2-3b-instruct-gptq-own
```
(port 18100; served name `turbine/Llama-3.2-3B-Instruct-GPTQ-own`, added to `lab-serve.sh`'s vLLM
slug table at `8ea67bc`). Then a comparable throughput run:
```
turbine-bench --url http://192.168.10.203:18100 --concurrency 16 --requests 200 \
  --prompt-words 512 --max-tokens 256 --ignore-eos --output json
```
`scripts/lab-serve.sh novanas --stop` after. No separate accuracy run needed here — the
full-GSM8K vLLM accuracy number is already committed (`vllm-gptq-own-full.json`, Do 2 above); this
is throughput only.

### Soak (gated on golden c1/c16 passing)

```
scripts/overload-soak.sh novanas --duration 10m --model /home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own
```
detached; judge `verdict.json` under `target/soak/`.

### Support-matrix flip (gated on golden c1/c16 AND the soak both passing) — LAST commit

Flip `gptq_int4` to `supported` on gfx1201 Llama in `crates/turbine-core/src/support.rs`, comment
citing the two 2026-09-30 "GPTQ INT4" decisions (`.procoder/ask/decisions.md`: "full-GSM8K drop
just over the 4-bit bound" and "which better checkpoint") and the evidence above (own checkpoint
0.0311 drop PASS, matches vLLM exactly, golden and soak green), following the `awq_int4` row's
comment style. Update the support tests
(`crates/turbine-server::support_startup::tests::*`, `crates/turbine-core` support tests — grep
`awq_int4` for every place its `supported` flip touched when that row closed, as the pattern to
replicate). **Any of golden c1/c16 or the soak failing → do not flip; report the numbers instead**
and leave `gptq_int4` `experimental`.

### Spec

Do **not** edit `.procoder/specs/phase-6a-quantization.md` — the lead amends S-11 at merge (proof
checkpoint for `gptq_int4` becomes the own checkpoint, per the "which better checkpoint" decision).

## Commits this rotation (branch `p6a-gptq-numerics`, on top of 2c3fbb7)

- merge `phase-6a-quantization` @ 96d3b76 (superseded by the next merge; harmless, just an earlier
  integration point).
- `7127762` test(eval): full-GSM8K results for the own GPTQ checkpoint and AutoRound.
- `8ea67bc` feat(lab): add the own GPTQ checkpoint as a proof model.
- `37badaf` docs(handoff): rotation 12 close-out (superseded by this file's current content).
- merge `phase-6a-quantization` @ 5c5f69b ("brings in the TokenizersBackend fix").
- this handoff commit (docs(handoff): rotation 12 close-out, part 2).

Gate green at the tip of all of the above (`gate: ok crates=all passed=801 failed=0`).
