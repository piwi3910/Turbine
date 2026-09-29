# Handoff: p6a-kv-t24-full (plan Task 24, full-GSM8K FP8 KV gate, Llama + OLMoE)

Written 2026-09-29 ~09:10 +04 by builder a67eec8abc8f117eb, stopped on the coordinator's order while
its run waits in the novanas GPU queue. Branch `p6a-kv-t24-full` = `p6a-kv-t24` + `phase-6a-quantization`
(d466c0b, which carries `tests/eval/gsm8k-full.jsonl`, 1,319 items).

## The detached run on novanas

- Script: `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.sh` (bash pid 260700,
  started 08:40 +04 with `setsid nohup`, session leader, survives any ssh disconnect).
- Log: `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.log`
  (server log of the current pass: `/tmp/gsm8kfull-server.log`, pid file `/tmp/gsm8kfull-server.pid`).
- Binaries: release `turbine-server` / `turbine-golden` built from this branch in
  `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/target/release/`, kernel library
  `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/kbuild/libturbine_hip.so`.
- Passes, strictly one after another: Llama BF16 KV (`scripts/lab/phase2c-novanas-llama.yaml`),
  Llama FP8 KV (`scripts/lab/phase6-novanas-llama-fp8kv.yaml`), OLMoE BF16 KV
  (`scripts/lab/phase2c-novanas-olmoe.yaml`), OLMoE FP8 KV (`scripts/lab/phase6-novanas-olmoe-fp8kv.yaml`).
  Each pass takes `port18000.lock`, then `bench.gate` + `bench.lock` exclusively (fd-based `flock`,
  released when the pass returns), pauses the `scripts/golden/` fixture jobs (SIGSTOP, SIGCONT after),
  serves natively on GPU 0 (cores 0-11), runs
  `turbine-golden eval --url http://127.0.0.1:18000 --tasks tests/eval/gsm8k-full.jsonl --output json`
  under `timeout 21600`, then stops only its own server.
- Outputs (in the remote source tree `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/src/`):
  - `tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json`
  - `tests/eval/llama-3.2-3b-instruct/turbine-fp8_e4m3-full.json`
  - `tests/eval/olmoe-1b-7b-0125-instruct/turbine-bf16-full.json`
  - `tests/eval/olmoe-1b-7b-0125-instruct/turbine-fp8_e4m3-full.json`
  - each with a `.err` beside it (the eval's stderr).
- State at handoff: queued (`pass llama-bf16: waiting for bench.gate`) behind another agent's exclusive
  `bench.lock` session (pid 251034, since 08:39, a lab-test Job on GPU 1) and two earlier exclusive
  waiters (pids 194608, 243975). Nothing of ours is on the GPU yet.

**Do not** run `scripts/remote-cargo.sh`, `lab-bench.sh` or `gate.sh` from worktree
`agent-a67eec8abc8f117eb` until the run is done: its `rsync --delete` would wipe the four output files
(not in the local tree) and rebuild the binaries mid-run. Do not remove that worktree either:
`lab-prune.sh` may then delete its remote `target/` (the binaries later passes exec).

## How to tell it is done

`tail -3 /home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.log` ends with
`ALLDONE rc: llama-bf16=<rc> llama-fp8kv=<rc> olmoe-bf16=<rc> olmoe-fp8kv=<rc>` (and
`ps -p 260700` finds nothing). rc 0 = report written; 2 = eval failed (see the `.err`); 124 = the 6 h
timeout hit (report missing or partial); 1 = `SERVER NOT READY` (the log holds the server tail). Each
pass logs `pass <name>: locks held, starting`, `server ready`, and `eval rc=` with UTC times.

## Collect and judge

```sh
d=/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/src/tests/eval
for s in llama-3.2-3b-instruct olmoe-1b-7b-0125-instruct; do
  scp "piwi@192.168.10.203:$d/$s/turbine-{bf16,fp8_e4m3}-full.json" tests/eval/$s/
done
# eval-compare runs on novanas (nothing is built on the Mac):
ssh piwi@192.168.10.203 'cd /home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/src && \
  for s in llama-3.2-3b-instruct olmoe-1b-7b-0125-instruct; do \
    ../target/release/turbine-golden eval-compare --baseline tests/eval/$s/turbine-bf16-full.json \
      --candidate tests/eval/$s/turbine-fp8_e4m3-full.json --max-drop 0.01; echo "$s exit=$?"; done'
```

Exit 0 = PASS, 1 = drop > 0.01, 2 = I/O. Commit the four JSONs as
`test(eval): full GSM8K (1,319 items) for the FP8 KV gate — Llama and OLMoE`. Reference points
(GSM8K-200): Llama 0.805 → 0.790, OLMoE 0.655 → 0.615. The first 200 lines of `gsm8k-full.jsonl`
are the GSM8K-200 set, so those items can be compared per item with the 200-run JSONs as a sanity check.
Do not flip `support.rs` rows or loosen any bound: report the verdicts to the coordinator first.

## If OLMoE misses 0.01 (decision 185ccee): look for a numerics cause first

What this builder already read (no run yet):

- `crates/turbine-model/src/kv_scales.rs`: `KvCache::fp8_from_checkpoint` reads
  `model.layers.{l}.self_attn.{k,v}_scale` (or `{k,v}_proj.output_scale`), all-or-nothing, else **all
  1.0**. Both lab configs say the checkpoints store none, so Llama and OLMoE both run with every scale
  1.0 — the "wrong layer's scale" hypothesis cannot change any value unless a checkpoint does carry
  scales. Confirm first: list the tensor names of `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct`
  for `_scale` (and check `/turbine/v1/status` `quantization` / the startup log of an FP8 KV serve).
- `crates/turbine-model/src/executor/decoder/mod.rs` ~1800: `model_layer = pp.layers.start + i`
  selects the scales; `batch::kv_layer(kv, i)` selects the pages. Without pipeline parallelism both are
  `i`, consistent.

Steps, one hypothesis at a time (procoder:debug):

1. Scale values: as above. If scales are all 1.0, the per-layer-scale path is ruled out for this
   checkpoint; say so in the report.
2. Saturation / range: with scale 1.0, e4m3 clips at ±448 and has 3 mantissa bits. Check OLMoE's
   post-RoPE K and V magnitudes (OLMoE has QK-norm, so K should be small; V is not normalised): trace
   one golden prompt with `DecoderExecutor::set_trace` (`golden hip_trace_vs_cpu_3b`-style, release build,
   `TURBINE_GOLDEN_TRACE=<ids>`) and record per-layer max |k_rope| / |v|. Any value > 448, or many
   subnormals (< 2^-6), points at the missing calibration rather than a bug.
3. Turbine vs the emulated-FP8-KV transformers reference: the FP8 KV golden fixture job
   (`/home/piwi/turbine-ci/remote/agent-ad2d9c8c7cc380c42/fp8kv_fixtures_t24.log`, outputs
   `/home/piwi/turbine-ci/golden-work/fp8kv/<slug>/{reference.jsonl,self_spread.json}`, made with
   `scripts/golden/quant_reference.py --kv-quant fp8_e4m3` and `self_spread.py --kv-quant fp8_e4m3`)
   gives the reference. Serve `phase6-novanas-olmoe-fp8kv.yaml` and run
   `turbine-golden compare --reference <that reference.jsonl> --concurrency 1` plus
   `turbine-golden positions --prompt-id <worst id>`: if Turbine tracks the emulated reference within
   its self-spread, Turbine implements FP8 KV correctly and the drop is the format's (no bug); if it
   diverges where BF16 KV does not, bisect by layer with the trace (K write vs FP8 read in the paged
   attention kernel, prefill vs decode path, CK FP8 instance vs Turbine fallback).
4. Report the finding (bug found + fixing commit with a test that catches it, or "format drop, Turbine
   matches the emulated reference") to the coordinator before any `support.rs` or tolerance change;
   `support.rs` changes go as a `handoff(crates/turbine-core/src/support.rs)` commit.

## Verdict (collector, lead rotations 7 and 8, 2026-09-29 ~17:40 +04)

All four passes finished (`ALLDONE`), and the Llama BF16 re-run at c16 finished too
(`gsm8k_bf16_c16_r7.log`: `r7-bf16-c16: done rc=0`). The result JSONs are committed under
`tests/eval/<slug>/turbine-{bf16,fp8_e4m3}-full.json`. The Llama BF16 c1 pass is kept as
`turbine-bf16-full.c1.json`. All `.err` files are empty. `eval-compare --max-drop 0.01` ran on novanas.

| model | BF16 KV (c16) | FP8 KV (c16) | drop | eval-compare | BF16→FP8 lost / gained | McNemar p | 95 % CI of drop |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | 1029/1319 = 0.7801 | 1040/1319 = 0.7885 | −0.0083 | PASS (exit 0) | 41 / 52 | 0.30 | [−0.023, +0.006] |
| OLMoE-1B-7B | 871/1319 = 0.6603 | 852/1319 = 0.6459 | +0.0144 | FAIL (exit 1) | 122 / 103 | 0.23 | [−0.008, +0.037] |

Noise floor and determinism:
- The first 200 items of every full run give the same outputs, bit for bit, as the GSM8K-200 runs
  (OLMoE BF16 and FP8, 200/200). So run-to-run noise is zero and every flip comes from the KV format.
- Llama BF16 at c1 vs c16: 14 lost / 9 gained (the drop is 0.0038, from batch composition alone).
- OLMoE outputs diverge somewhere under FP8 KV in 1205/1319 items (91 %), against 1004/1319 (76 %) for Llama.
  The median divergence point is 15 % into the BF16 answer.
- No item flips when its output is unchanged.

Numerics check (decision "Phase 6a gate misses on GSM8K-200", item 1):
1. **Scales.** `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct/model.safetensors.index.json` has
   0 `_scale` tensors, so every per-layer K/V scale is 1.0 (`kv_scales.rs` all-or-nothing default). The
   "wrong layer's scale" path cannot matter for this checkpoint.
2. **Range vs e4m3 at scale 1.0.**
   - How it was measured: CPU transformers 4.57.1, BF16, on 12 GSM8K prompts plus Turbine's BF16 answers.
     Script `/home/piwi/turbine-ci/scratch/kvrange-r7/kv_range.py`, outputs `olmoe.txt` and `llama.txt` beside it.
     K is taken after k_norm and RoPE. The hooks are on `apply_rotary_pos_emb` and `v_proj`, because OLMoE in
     4.57.1 has no `eager_attention_forward`.
   - **No saturation.** max |K| ≤ 22 (OLMoE) and ≤ 24 (Llama), and max |V| ≤ 1.5 / 6.3, all far below 448.
   - **Underflow.** OLMoE V is tiny in the early layers: RMS 0.008–0.014 in layers 0–4, against 0.06–0.25 for Llama.
     - Share of V values in the e4m3 subnormal range (< 2^-6): 95 → 75 %, and 35 → 7 % flush to 0.
     - V quantization relative error: 7.6 / 6.8 / 5.7 / 5.0 / 4.2 % in layers 0–4, against the 2.7 % e4m3
       floor that later layers and all of Llama sit at.
     - OLMoE K has many near-zero channels after k_norm (layer 0: 74 % flush to 0), but K's relative error
       stays at 2.6–2.8 %, the same as Llama's.
   - The attention-output relative error from K/V quantize-dequantize is 4.0–8.2 % per layer for OLMoE and
     3.4–8.8 % for Llama, so the two are comparable. OLMoE's larger answer divergence is consistent with its
     discrete top-8 expert routing amplifying the same size of perturbation.
3. **Turbine vs the emulated FP8 KV reference.** Not run: the new OLMoE reference (fp8kv-eager regen) is not
   ready. The old one is void.
   - Reason: in transformers 4.57.1, `OlmoeDecoderLayer` builds its attention from
     `OLMOE_ATTENTION_CLASSES[config._attn_implementation]` (modeling_olmoe.py:629). It never calls a
     registered AttentionInterface function, so the pre-eager `install_kv_quant` swap never reached OLMoE.
   - On the integration tip (d877f55), `scripts/golden/quant_reference.py:347-386` hooks the KV cache instead:
     a forward pre-hook on every `self_attn` hands it a `_Fp8KvCache` view, and OLMoE calls
     `past_key_values.update` after RoPE. That covers OLMoE.
   - Transformers is pinned to 4.57.1 in the uv header (line 5).
   - When the regen lands, run `turbine-golden compare --reference <new olmoe fp8kv reference> --concurrency 1`
     against a `phase6-novanas-olmoe-fp8kv.yaml` serve. That is the remaining check for a Turbine bug.

**Cause:** (b) + (c).
- Nothing points at a Turbine bug. There is no saturation, the K error sits at the e4m3 floor, and the
  attention-output perturbation is the same size as Llama's, which passes.
- The measurable format effect is V underflow in OLMoE layers 0–4, from the missing scales at 1.0.
  Calibrated per-layer V scales (≈ amax/448, e.g. 0.26/448 for layer 0) would put V in the normal range and
  cut its error from 5–8 % to ≈ 2.7 %. K would not change.
- The 0.0144 drop itself is not significant (122 vs 103 flips, McNemar p = 0.23, CI includes 0).

**Recommendation for the rows:**
- Llama FP8 KV: PASS, eligible for `supported`.
- OLMoE FP8 KV: the literal 0.01 bound fails on a non-significant drop. Keep it `experimental` until the
  emulated-reference compare (item 3) is done.
- Then either accept it as format-limited noise (user decision) or add a calibrated / dynamic per-layer V scale
  for checkpoints without scales, and re-run the OLMoE FP8 pass.
- No `support.rs` change has been made.
