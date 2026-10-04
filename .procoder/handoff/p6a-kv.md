# Handoff: p6a-kv (plan Task 24, FP8 KV lab proof)

Written 2026-09-29 ~08:00 by the new 6a lead (builder killed without a handoff).
Worktree `.claude/worktrees/agent-a0aaad9d44989312b`, tip a92259c `wip: task 24 kv_gpu` on 2116221.

## Done

- On integration: Task 22 (a43cc7b, 877b0ab, fa23fe4, 6232346), Task 23 (00614a6, 3d1b3f5, 58fe8c0, 984e05b),
  ea20c64 (FP8 KV experimental rows for the proof), 6ad977c (lab-bench models `llama-fp8kv` / `olmoe-fp8kv`),
  02cab3d (`--kv-quant fp8_e4m3` in the fixture scripts), dd3e839 (golden decision B′, accepted by the user).
- a92259c (wip, this branch only): `kv_gpu.rs` `prefix_reuse_matches_cold_fp8_kv`, `nvme_round_trip_matches_cold_fp8_kv`,
  `phase6_fp8kv_lab_configs_load`; configs `scripts/lab/phase6-novanas-{llama,olmoe}-fp8kv.yaml`.
  **Lab PASS** (`scratchpad/lab-kvgpu2.log`, job 0929010737-01b74d32: 2/2). The commit only needs a proper
  message (`test(server): FP8 KV prefix reuse and NVMe round trips on the GPU`) — squash/reword, then integrate.
- Bench (GPU 0, 200 req, `scratchpad/benches3.out`): Llama FP8 KV 835.8 tok/s = 0.980× BF16 KV 852.7 (target ≥ 0.95 ✓);
  OLMoE 629.2 = 1.027× 612.5 ✓. Golden vs the BF16 reference FAIL (expected; the gate is B′, the emulated-FP8 reference).
  L0 blocks ≥ 1.95× — not yet recorded; read it from `/turbine/v1/kv` or the startup log.

## Uncommitted

- `tests/eval/llama-3.2-3b-instruct/{turbine-bf16,turbine-fp8_e4m3}.json`: 0.805 vs 0.790 — **drop 0.015 > the plan's
  0.01 bound (FAILS as written)**. 3 items of 200; likely noise, but the lead must take it to the coordinator.
- `tests/eval/olmoe-1b-7b-0125-instruct/turbine-bf16.json` 0.655; `turbine-fp8_e4m3.json` being written now.
- `handoff/decisions-task23.md` — already integrated as 984e05b; delete.

## Running

- `scratchpad/evals.sh` (local, under port18000 + fixture-pause): the OLMoE FP8 KV GSM8K eval is the last step;
  it stops its own serve Job.

## Not done — FP8 KV golden fixtures (decision B′)

`scratchpad/fp8kv_fixtures.sh` wrote nothing: `/home/piwi/turbine-ci/golden-work/fp8kv/` is empty (killed by a crash).
Re-queue it (fixture.lock, one at a time): `quant_reference.py --kv-quant fp8_e4m3` then
`self_spread.py --kv-quant fp8_e4m3` for Llama and OLMoE → `tests/golden/<slug>-fp8kv/{reference.jsonl,tolerance.json,README.md}`
(check `lab-bench.sh --print-model llama-fp8kv` for the slug it expects).

## Exact next steps (fresh builder, default model)

1. Reword a92259c; re-queue the fixtures; after them `lab-bench --model llama-fp8kv --golden16` and `olmoe-fp8kv`,
   plus `--model llama-fp8 --golden16 -- --set kv.dtype=fp8_e4m3` once Task 14's FP8 fixtures exist.
2. Eval-compare at 0.01 both models (the lead decides the Llama 0.015 case with the coordinator).
3. Soak with FP8 KV — ask the coordinator first. Rows via `handoff(support.rs)`.

## Lab traps

- ONE GPU job at a time; the NVMe test writes under `/home/piwi/turbine-kv` (≤ 64 GiB) — watch disk (≥ 100 GB free).
