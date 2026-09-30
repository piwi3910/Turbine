# Handoff: p6a-gptq-numerics (T18 GPTQ GSM8K drop investigation)

## Rotation 12 close-out (2026-09-30 ~11:20Z, builder r12, brief r12-gptq-proof)

Picked up rotation 11/12's cascade at "everything past `gptq-calib3: done rc=0`, no further action
needed" — confirmed all three go-files had already landed on novanas:

- `gptq-calib3: done rc=0` (05:30Z) — checkpoint written to
  `/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own` (2.2G), `turbine_calibration.json`
  intact (torch 2.13.0+rocm7.2, llmcompressor 0.14.0, compressed-tensors 0.19.0, transformers
  5.17.0, `gptq_backend: torch_eager`, `ct_triton: disabled`, 1099.6 s).
- `gptq-own: done rc=0` / `gptq-own-vllm: done rc=0` (09:39Z / 09:59Z) — full GSM8K c16, both
  paired judgements PASS (see below).
- `gptq-autoround2: done rc=0` (10:03Z) — AutoRound early data point, PASS, does not decide the row.

### Do 1-2 (merge, eval results) — DONE

1. Merged `phase-6a-quantization` (96d3b76) into `p6a-gptq-numerics` — clean, no conflicts
   (7127762's parent, commit `<merge sha, see git log>`). Brings in the YaRN rope-parameters work
   and, importantly, **d64f0cc** (`dequantize_checkpoint.py` transformers-5 `TokenizersBackend`
   fix) that this checkpoint's `tokenizer_config.json` needs.
2. Copied and committed (7127762) into `tests/eval/llama-3.2-3b-instruct/` (the existing
   full-GSM8K comparison directory, per its own README's naming plan):
   - `turbine-gptq-own-full.json` / `vllm-gptq-own-full.json` + `*-paired.json` — **decides the
     row**: Turbine 988/1319 = 0.7491 vs BF16 0.7801, drop 0.0311 ≤ 0.04 **PASS** (McNemar p
     0.00222, exact p 0.00213, 95% CI [+0.0117, +0.0504]); vLLM-ROCm on the same checkpoint also
     988/1319 = 0.7491 (Turbine vs vLLM drop 0.0000). No circuit transitions in either run.
   - `turbine-gptq-autoround-full.json` / `*-paired.json` — AutoRound side result (kaitchup
     checkpoint), 998/1319 = 0.7566, drop 0.0235 ≤ 0.04 PASS. Does not decide `gptq_int4`.
   - `tests/eval/llama-3.2-3b-instruct/README.md` updated with the final numbers and the
     calibration package versions.

### Do 3 (golden fixture) — IN PROGRESS on novanas, queued

`.procoder/handoff/gptq_own_fixture.sh` (committed, 8ea67bc) — one CPU fixture job, `FIXTURE_JOB=
gptq-own-fixture`, `nice -n 19 taskset -c 12-15`, 4 threads, under `fixture.lock`:
1. `quant_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own
   --act-quant none` (weight-only, ct_pack_int4 decode path already in `dequantize_checkpoint.py`)
   → reference + kept dequantized BF16 copy in `/dev/shm`.
2. `self_spread.py` over **all eight** variants in one call (the MXFP4 p05 lesson: never split
   into partial subsets that could each individually miss a knife-edge position) against that
   reference.
3. Copies both into `tests/golden/llama-3.2-3b-instruct-gptq-own/{reference.jsonl,spread.json}`.

Added `gptq-own-fixture` to `/home/piwi/turbine-ci/fixture.queue` right before the
`w4a4-mxfp4-a4` line (after `mxfp4-inc-spread`), per the brief. Started detached on novanas
(`setsid nohup bash gptq_own_fixture.sh </dev/null &`, log
`/home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture/gptq-own-fixture.log`), currently **waiting on
fixture.lock** behind two other jobs already ahead of it when it joined (`w4a4-mxfp4-a4-reference`,
the current holder — an 8B reference, already running before the queue insert took effect, so it
was not pre-empted — then `r12:mxfp4-inc-spread`); `gptq-own-fixture` is next after those two, and
should be short (3B, weight-only, no act-quant emulation). Marker to watch for:
`gptq-own-fixture: done rc=`.

**Still to do once the fixture finishes (rc=0):**
- Check `$OUT/reference.log`'s rotary line says `llama3` / theta `500000` (own checkpoint is
  Llama-3.2-3B, same rope config as the BF16/AWQ/GPTQ-published fixtures — should be automatic,
  but the brief calls it out explicitly, so verify it).
- Read `tests/golden/llama-3.2-3b-instruct-gptq-own/spread.json`, derive `tolerance.json` by the
  calibration rule used for AWQ (`tests/golden/llama-3.2-3b-instruct-awq/README.md`): **never
  tighter than the BF16 Llama-3.2-3B-Instruct bounds (likely 0.15, tail 0.55, batched 0.25/0.75,
  `min_prompts_passing` 14)** — round the measured spread up to two decimals, floor at those BF16
  values.
- Write `tests/golden/llama-3.2-3b-instruct-gptq-own/README.md` (method + the 8-variant table),
  commit `reference.jsonl` / `tolerance.json` / `README.md` (not `spread.json`, matching the AWQ
  precedent — spread.json is a working artifact, not committed there either).

### Do 4 (golden c1/c16 + bench) — NOT STARTED, needs the fixture above first

Added to the tree and committed (8ea67bc):
- `scripts/lab-bench.sh --model llama-gptq-own` → slug `llama-3.2-3b-instruct-gptq-own`, config
  `scripts/lab/phase6-novanas-llama-gptq-own.yaml` (copy of the published-checkpoint one, served
  name `turbine/Llama-3.2-3B-Instruct-GPTQ-own`).
- `scripts/lab-serve.sh --vllm llama-3.2-3b-instruct-gptq-own` (served name
  `turbine/Llama-3.2-3B-Instruct-GPTQ-own`) — for a vLLM-ROCm throughput number on the same
  checkpoint "for the record" (no new script needed; lab-serve.sh already generalizes over any
  slug in `turbine-models/`).

Once the golden fixture is committed, run (background; holds `bench.lock`, refuses if GPU busy):
```
scripts/lab-bench.sh --model llama-gptq-own --label r12-gptq-own --golden16
```
For the vLLM throughput number, after lab-bench.sh's Turbine run has `--stop`ped its server:
`scripts/lab-serve.sh novanas --vllm llama-3.2-3b-instruct-gptq-own` (port 18100) then a
`turbine-bench --url http://192.168.10.203:18100 --concurrency 16 --requests 200 --prompt-words
512 --max-tokens 256 --ignore-eos` for the comparable throughput line, `--stop` after.

### Do 5-6 (soak, flip) — NOT STARTED, gated on Do 4 passing

`scripts/overload-soak.sh novanas --duration 10m --model
/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own` detached, after golden passes. If golden
c1/c16 AND the soak both pass, flip `gptq_int4` to `supported` on gfx1201 Llama in
`crates/turbine-core/src/support.rs` as the **last** commit (comment citing the two 2026-09-30
"GPTQ INT4" decisions and the evidence above, like the `awq_int4` row) and update the support
tests. Any failure → no flip, report the numbers instead (do not touch `support.rs`).

### Do 7 — not touched: spec S-11 stays as-is; the lead amends it at merge (own checkpoint becomes
the `gptq_int4` proof checkpoint).

### Gate

`scripts/gate.sh --base 2c3fbb7` launched detached in the background after the
`phase-6a-quantization` merge (7127762 is downstream of it); check
`/tmp/gate-r12-gptq.log` on the Mac side, or rerun if that shell is gone — the merge's Rust diff
(YaRN rope-parameters work) needs the full-workspace clippy/test gate it triggers (a
workspace-wide file changed). Not yet confirmed green as of this handoff.

## Commits this rotation (branch `p6a-gptq-numerics`, on top of 2c3fbb7)

- merge: `phase-6a-quantization` (96d3b76) — clean.
- `7127762` test(eval): full-GSM8K results for the own GPTQ checkpoint and AutoRound.
- `8ea67bc` feat(lab): add the own GPTQ checkpoint as a proof model (lab-bench.sh, lab-serve.sh,
  the phase6 lab config, the fixture driver).

## Handing off

Everything left is either a long CPU/GPU queue wait (golden fixture → golden run → soak) or a
follow-on step gated on those results (tolerance.json / README, the support.rs flip). Per the
"never block hours in the foreground" rule, ending here rather than polling; the lead (or the next
rotation) picks up at "Do 3 still to do" above once `gptq-own-fixture: done rc=` appears in
`/home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture/gptq-own-fixture.log` on novanas.
