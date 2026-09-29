# Handoff: p6a-kv-t24 (plan Task 24, FP8 KV lab proof — mechanical wrap-up)

Written 2026-09-29 by the Task 24 wrap-up builder. Branch `p6a-kv-t24` (rebuilt from `p6a-kv`
8f256dc, merged with `phase-6a-quantization`; the `wip` commit was reworded via cherry-pick, not
`rebase -i`, so no interactive flags were used).

## Done

- ca45bd9 `test(server): FP8 KV prefix reuse and NVMe round trips on the GPU` — the reworded
  `kv_gpu.rs` tests (already lab-PASS 2/2, `scratchpad/lab-kvgpu2.log` on the old worktree).
- 8fb9e90 `test(eval): GSM8K-200 for Llama and OLMoE at BF16 and FP8 KV` — copied in from the old
  worktree (`agent-a0aaad9d44989312b`), which finished `scratchpad/evals.sh` (`done-all`).
- `scripts/gate.sh`: started, see result below (log `scratchpad/gate_t24.log` on the Mac, not committed).
- Labbook (`phase-6a-quantization` set, `turbine-lab-bench` type): 3 of 4 Task 24 BENCH runs were
  already recorded by the earlier builder (t24c-bf16-llama `8301b0ce`, t24c-bf16-olmoe `092139c1`,
  t24c-fp8-olmoe `2f840f87`); created the missing one, t24c-fp8-llama, run `0db7f128`.

## Eval verdicts (GSM8K-200, `eval-compare --max-drop 0.01`) — recorded, not the gate any more

- Llama: 0.805 (BF16) → 0.790 (FP8 KV), drop 0.015 → **FAIL**.
- OLMoE: 0.655 (BF16) → 0.615 (FP8 KV), drop 0.040 → **FAIL** (new result; the old handoff only had OLMoE BF16).
- **Superseded**: user decision 2026-09-29 (`.procoder/ask/decisions.md` commit 4f45232 on
  `phase-6a-quantization`) makes the full 1,319-item GSM8K the actual gate for Llama FP8 KV (and
  MXFP4-A16). Keep these 200-item numbers as a record only.

## Not done — full GSM8K gate

`tests/eval/gsm8k-full.jsonl` is being written by another builder on branch `p6a-gsm8k-full`; as of
now that branch's tip is only the decisions-doc commit (4f45232, same as this branch) — the dataset
commit hasn't landed. Next agent: `git log p6a-gsm8k-full`, merge/cherry-pick once it's there, then
serve+eval BF16 (`scripts/lab/phase2c-novanas-llama.yaml`) and FP8 KV
(`scripts/lab/phase6-novanas-llama-fp8kv.yaml`) one after the other (never in parallel — one GPU job
at a time on novanas), each `turbine-golden eval --tasks tests/eval/gsm8k-full.jsonl` with a ~6h
timeout, following the `scratchpad/evals.sh` pattern (port18000 lock + fixture-pause). Output to
`tests/eval/llama-3.2-3b-instruct/turbine-{bf16,fp8_e4m3}-full.json`, then `eval-compare --max-drop
0.01`, then commit.

## Fixture job re-queued (golden fixtures lost to the crash)

Launched detached on novanas from my own remote slot (not the dead worktree's):
`ssh piwi@192.168.10.203 setsid nohup bash fp8kv_fixtures_t24.sh` in
`/home/piwi/turbine-ci/remote/agent-ad2d9c8c7cc380c42/` (pid 115130 at launch), log
`.../fp8kv_fixtures_t24.log`, output `/home/piwi/turbine-ci/golden-work/fp8kv/<slug>/`. Serializes on
`fixture.lock` behind other agents' fixture jobs — as of 08:12 it had only reached "llama reference"
start; not finished. `scripts/lab-bench.sh --print-model llama-fp8kv` / `olmoe-fp8kv` confirm the
golden slugs: `tests/golden/llama-3.2-3b-instruct-fp8kv/` and
`tests/golden/olmoe-1b-7b-0125-instruct-fp8kv/`.

## Exact next step

1. Poll the fixture log; once both slugs have `reference.jsonl` + `self_spread.json`, build
   `tolerance.json` from the spread (as other `-fp8kv`-style fixtures do) and commit
   `tests/golden/<slug>-fp8kv/{reference.jsonl,tolerance.json,README.md}`.
2. `scripts/lab-bench.sh --model llama-fp8kv --golden16` and `--model olmoe-fp8kv --golden16`.
3. Full-GSM8K gate (above), then eval-compare at 0.01.
4. Only then flip the support rows and commit `feat(core): FP8 KV supported on gfx1201 (golden,
capacity, eval recorded)` (plan Task 24's own final commit — not done by this handoff).
