# Handoff: p6a-gptq-numerics (T18 GPTQ GSM8K drop investigation)

## State 2026-09-29 ~18:40Z (rotation 9 builder)

Context: T18 GPTQ (`shuyuej/Llama-3.2-3B-Instruct-GPTQ`) GSM8K-200 0.715 vs BF16 0.805 (drop 0.090 > 0.04)
FAIL; perf and golden pass; AWQ passes on the same INT4 kernels (0.775). GPTQ stays experimental.

Checkpoint facts (`config.json` `quantization_config`, optimum export, transformers 4.43.4): bits 4,
group_size 128, sym true, desc_act false, no `checkpoint_format` (so v1: qzeros hold zero − 1),
damp_percent 0.1 (AutoGPTQ default 0.01), `use_exllama` true, torch_dtype float16, tied embeddings.
Quantized by a third party (medpodgpt repo); calibration data unknown.

### (b) Independent CPU dequant check — DONE: Turbine's GPTQ load is bit-exact

`scripts/golden/gptq_dequant_check.py` (numpy only; a literal transcription of AutoGPTQ
`qlinear_cuda_old`'s dequant: shift-unpack `qweight` along k and `qzeros` along n, `+ 1` then `& 0xF`
for v1, `scales[g_idx]` / `zeros[g_idx]`, so any g_idx / act order is honoured) vs Turbine's CPU
dequant after the real loader repack (`crates/turbine-model/examples/int4_layer_dump.rs`: config →
registered `gptq` format → `slots` / `check_tensor` / `repack` → `turbine_kernels::cpu::quant::dequantize`).
Driver `.procoder/handoff/gptq_dequant_run.sh` (novanas, nice 19, cores 12-15, no lock; ~5 min).

Layers 0, 13, 27 × q/k/v/o/gate/up/down (21 layers): **0 bit mismatches in every layer** (max |Δ| 0.0).
Per layer: g_idx is the identity `i / 128`; every stored zero nibble is 7 → zero 8 (sym v1, offset +1
correct); scales F16; packing order confirmed (a wrong nibble order, axis or offset could not be bit-exact).
Turbine vs AutoGPTQ's F16-product dequant differs by ≤ 2.4e-4 (F16 rounding of the product only;
Turbine keeps F32).

Weight error against the BF16 original (relative Frobenius), GPTQ vs a round-to-nearest of the same
scheme (AutoGPTQ symmetric quantizer, group 128) on the same weights:

| layer | q | k | v | o | gate | up | down |
|---|---|---|---|---|---|---|---|
| 0 GPTQ | 0.160 | 0.163 | 0.144 | 0.140 | 0.130 | 0.129 | 0.137 |
| 0 RTN | 0.131 | 0.135 | 0.122 | 0.119 | 0.114 | 0.113 | 0.117 |
| 13 GPTQ | 0.137 | 0.142 | 0.136 | 0.128 | 0.134 | 0.130 | 0.130 |
| 13 RTN | 0.119 | 0.123 | 0.118 | 0.112 | 0.119 | 0.115 | 0.117 |
| 27 GPTQ | 0.135 | 0.153 | 0.142 | 0.131 | 0.128 | 0.128 | 0.132 |
| 27 RTN | 0.120 | 0.135 | 0.126 | 0.113 | 0.116 | 0.115 | 0.120 |

GPTQ / RTN = 1.10–1.22 (layer 0 attention highest). GPTQ minimizes the layer *output* error on its
calibration data, not the weight error, so a weight error somewhat above RTN is expected; this does not by
itself show a bad checkpoint.

Results on novanas: `/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/gptq-dequant/results.jsonl`
(first run, with a void raw-AWQ column, in `gptq-dequant-r1/`). AWQ is not compared that way: its
weights carry the activation-aware input scales folded in, so they differ from the original by design.

Conclusion: no Turbine load / repack / CPU-dequant bug for this checkpoint (g_idx, zero offset, packing,
group, sym, scale dtype all verified). The GPU kernels are the ones AWQ passes on, and golden c1/c16 pass
against the transformers reference of the dequantized GPTQ weights. So the GSM8K-200 drop is a property of
this checkpoint (damp 0.1, unknown calibration set) or n=200 noise, not Turbine numerics — part (a) sizes
the drop on the full set. No code fix, no loader commit.

### (a) Full GSM8K at c16 on Turbine GPTQ — started detached, waiting for its go-file

`.procoder/handoff/gptq_full_run.sh` (copy on the host: `/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/gptq_full_run.sh`,
started 18:12Z, pid 1031924). Binaries of this branch (built by `scripts/remote-cargo.sh`, release
`turbine-server` + `turbine-golden` in `agent-p6a-gptq-numerics/target/release`), kernel library built
by the script into `…/kbuild`. It waits (bounded 24 h, 60 s sleeps, log `waiting for go-file`) for
`/home/piwi/turbine-ci/gpu-queue/gptq-full.go`, then port18000.lock → bench.gate → bench.lock, fixtures
paused, GPU 0, cores 0-11, config `scripts/lab/phase6-novanas-llama-gptq.yaml`, full GSM8K (1319) at
concurrency 16 (6 h timeout), kills its server.

- Log: `/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/gptq-full/run.log`, last line
  `gptq-full: done rc=<rc>`.
- Result: `…/gptq-full/turbine-full.json` (+ `.err`, `server.log`, `status.json`).
- Judge: copy to `tests/eval/llama-3.2-3b-instruct-gptq/turbine-full.json`, then
  `turbine-golden eval-compare --baseline tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json
  --candidate tests/eval/llama-3.2-3b-instruct-gptq/turbine-full.json --max-drop 0.04`
  (BF16 full 0.7801 at c16 → pass ≥ 0.7401). GSM8K-200 had 0.715; the full set tells whether the
  200-item drop is sampling noise (±0.03 at n=200) or real.

## Open questions for the lead

- With the load proven exact, a remaining independent check of the *checkpoint* quality would be an
  HF-transformers GSM8K run on the dequantized BF16 copy (CPU, many hours) or vLLM with another GPTQ
  kernel (vLLM-ROCm refuses this checkpoint). Alternative: a better GPTQ checkpoint (e.g. damp 0.01,
  a known-good publisher) as the gate's candidate.

Gate at 1f7290b: 785 passed, 1 failed — `turbine-server::tiny_server queue_full_429` (503 `queue_timeout`
instead of 200, a timing flake while the GPTQ kernel build loaded the host); rerun alone it passes. No
Rust change of this branch touches turbine-server (only the `int4_layer_dump` example).

## Rotation 11: C then A (2026-09-30, builder r11)

User decision 06a880f: C (AutoRound early data point, does NOT decide the row), then A (our own
llm-compressor GPTQ, decides the row). `gptq_int4` stays `experimental`; the lead re-judges it.
Branch merged `phase-6a-quantization` (c144abf) cleanly.

First run (gptq-full, shuyuej): 972/1319 = 0.7369, drop 0.0432; `scripts/eval/paired_compare.py`
reproduces it: lost 122 / gained 65, McNemar cc p 4.2e-5, exact p 3.7e-5, CI [+0.0230, +0.0634].

Tools (committed):
- `.procoder/handoff/gptq_full_run.sh --name <label> --model-dir <dir> --served-name <id>
  --expect-packaging gptq|ct_pack_int4 [--vllm]` — the gptq-full driver, parameterised (no fork).
  Waits for `gpu-queue/<label>.go`, then port18000 → bench.gate → bench.lock (held through the
  vLLM pass, plus port18100.lock), GPU 0, cores 0-11. Checks `/turbine/v1/status` packaging and the
  `gptq_int4` row before evaluating. Output `$REMOTE/<label>/`; markers `<label>: done rc=` and
  `<label>-vllm: done rc=`. Each eval is judged into `turbine-paired.json` / `vllm-paired.json`.
- `scripts/eval/paired_compare.py <bf16-full.json> <candidate.json> --max-drop 0.04 [--json]`.
- `scripts/eval/gptq_calibrate.py` (uv inline script: torch 2.13.0+rocm7.2, triton-rocm 3.7.1,
  llmcompressor 0.14.0, compressed-tensors 0.19.0, transformers 5.17.0, accelerate 1.15.0,
  datasets 5.0.1; recipe W4A16 sym g128 damp 0.01 actorder None, lm_head ignored; 512 × 2048
  ultrachat_200k @ 8049631c train_sft shard 0 seed 42). Writes `turbine_calibration.json` into the
  checkpoint with all versions and the device.
- `.procoder/handoff/gptq_calib_run.sh` — setup (dataset shard download, uv env `--check`, CPU,
  GPU hidden, no lock) then waits for `gpu-queue/gptq-calib.go`, same three locks, one GPU
  (ROCR/HIP_VISIBLE_DEVICES=0), 4 h timeout. Log `/home/piwi/turbine-ci/scratch/gptq-own/calib.log`
  (first attempt's resolver failure kept as `calib-r1.log`: triton-rocm had to come from the
  pytorch index too). uv cache / python under `/home/piwi/turbine-ci/scratch/gptq-own/` (delete
  `uv-cache` after the run if disk is short).

Detached on novanas (REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics):

| go-file | what | log | marker |
|---|---|---|---|
| `gptq-autoround.go` | AutoRound, first try: FAILED rc=2 (circuit `latency_drift`, see below) | `$REMOTE/gptq-autoround/run.log` | `gptq-autoround: done rc=2` |
| `gptq-autoround2.go` | AutoRound rerun, full GSM8K c16 (Turbine) | `$REMOTE/gptq-autoround2/run.log` | `gptq-autoround2: done rc=` |
| `gptq-calib.go` | llm-compressor calibration on GPU 0: FAILED rc=1, ROCm Triton compile error (see below) | `/home/piwi/turbine-ci/scratch/gptq-own/calib.log` | `gptq-calib: done rc=1` |
| `gptq-calib2.go` | llm-compressor calibration retry, torch (non-Triton) GPTQ block-update path | `/home/piwi/turbine-ci/scratch/gptq-own/calib2.log` | `gptq-calib2: done rc=` |
| `gptq-own.go` | own checkpoint: Turbine, then vLLM-ROCm | `$REMOTE/gptq-own/run.log` | `gptq-own: done rc=`, `gptq-own-vllm: done rc=` |

Lead queue (r11): w4a4-rerun → `gptq-calib2.go` → `gptq-own.go` (after calib2 rc=0) →
`gptq-autoround2.go` (after gptq-own-vllm, or after gptq-own fails).

`gptq-own.go` must come after `gptq-calib2: done rc=0` (the driver refuses with rc=1 if the
checkpoint directory has no config.json when its pass starts).

### gptq-calib rc=1 fix (rotation 11, 2026-09-30): ROCm Triton GPTQ kernel does not compile

`gptq-calib.go` (22:28Z) died with `RuntimeError: Implicit conversion of CUDA __nv_fdiv_rn device
function has been dropped; ...` from Triton's AMD backend
(`triton/backends/amd/compiler.py:make_llir` → `need_extern_lib`), raised while llm-compressor
0.14.0's fused Triton GPTQ block update JIT-compiles
(`llmcompressor/modifiers/gptq/gptq_quantize.py` `fused_gptq_block_update` →
`_gptq_block_update_kernel`, called from `_gptq_block_update_triton` via the `compressed_tensors`
`ImplBackend` dispatch in `quantize_weight`). The kernel's extern-lib lowering assumes CUDA libdevice
and is not portable to ROCm Triton 3.7.1's AMD backend.

Fix (no monkeypatch — a supported switch already exists, checked in this order per the brief):
1. A `GPTQModifier` constructor argument: none — grepped `llmcompressor/modifiers/gptq/base.py` for
   `triton`/`backend`, no hits.
2. An environment variable: **yes** — `gptq_quantize.py`'s own dispatch predicate,
   `_gptq_block_update_triton_req` (registered against `compressed_tensors.utils.impl_backend
   .ImplBackend` at priority 0, ahead of the eager-torch `gptq_block_update` base implementation),
   reads `os.environ.get("LLMCOMPRESSOR_DISABLE_GPTQ_TRITON", "0") != "1"` as one of its conditions;
   setting it to `"1"` makes the predicate return `False` and the dispatcher falls through to the
   plain-torch `gptq_block_update` (same file, ~line 152: per-column `fake_quantize` + Hessian-inverse
   error propagation — the reference GPTQ algorithm, no CUDA/Triton). Algorithm is unchanged: same
   recipe (GPTQ, damp 0.01, g128, sym, no act-order), same math, just the eager instead of the fused
   kernel for the inner per-block update.
3. `scripts/eval/gptq_calibrate.py`: added `--gptq-backend {auto,torch,triton}` (default `auto`).
   `auto` sets `LLMCOMPRESSOR_DISABLE_GPTQ_TRITON=1` when `torch.version.hip is not None` (a ROCm
   build) and leaves Triton alone on CUDA; `torch` / `triton` force it either way. The device print
   line and `--check`'s JSON now include `gptq_backend`; `turbine_calibration.json` gets a
   `"gptq_backend": "torch_eager"|"triton"` field recording which path ran.
4. CPU-only smoke (cheap, ~1 s, on novanas under the existing uv env, GPU hidden): a tiny synthetic
   layer (`weights` `[1,8,16]`, a random SPD Hessian, `QuantizationArgs` W4A16 group 8, blocksize 8)
   through `quantize_weight` with `LLMCOMPRESSOR_DISABLE_GPTQ_TRITON=1`; confirmed
   `_gptq_block_update_triton_req(...)` is `False` and the eager path returns a finite,
   correctly-shaped, non-NaN quantized weight (`used_rtn_fallback=[False]`). Not committed (a
   throwaway script under the session scratchpad, not the repo) — the real signal is the full
   calibration run under `gptq-calib2.go`.
5. Requeued: `.procoder/handoff/gptq_calib_run2.sh` (copy of `gptq_calib_run.sh` with the marker/log
   renamed `gptq-calib2`/`calib2.log`, go-file `gpu-queue/gptq-calib2.go`, and `--gptq-backend auto`
   passed explicitly to both the `--check` setup step and the real run for a self-documenting log
   line) uploaded to `$REMOTE/gptq_calib_run2.sh` and started detached
   (`setsid nohup bash … </dev/null &`, pid 1441715) on novanas; it is past setup and waiting on
   `gpu-queue/gptq-calib2.go` now. Dataset shard and uv environment are reused from the first attempt
   (no re-download). `gptq_calib_run.sh` / `gptq-calib.go` / `calib.log` are left as-is, the historical
   record of the failure.

AutoRound first try (21:46Z): the circuit opened on `latency_drift` (ratio >= 4, HEALTHY →
CIRCUIT_OPEN directly, GREEN pressure) 2 min into the c16 eval; turbine-golden aborted on the 503
(gsm8k-test-1185), rc=2. Host load was 12-13 (MXFP4 CPU fixture + my uv/torch env install, which
was not under the fixture pause pattern). gptq-full ran the same config for 2 h with 0 circuit
transitions. Fix in the driver: `--set reliability.circuit.latency_drift_open=100` (drift ≥ 2
still degrades and is logged; the driver prints the run's circuit transitions after each eval;
device errors / OOM / other triggers unchanged), and the kernel build is serialized on
`kbuild.lock` (two drivers started together corrupted the shared build.ninja once). The CPU setup
is finished; nothing CPU-heavy of mine runs during the GPU evals.

AutoRound checkpoint: `/home/piwi/turbine-models/llama-3.2-3b-instruct-autoround-gptq`
(kaitchup @ e11f15d, 2.27 GB; quant_method gptq, bits 4, group 128, sym, desc_act false, no
checkpoint_format → v1, damp 0.01, auto-round 0.4.5, 500 iters, 512 × 2048). Turbine loads it on
the cpu backend: packaging `gptq`, row `gptq_int4` (experimental), 196 `int4_group_sym` layers.

When the markers land:
1. copy `$REMOTE/gptq-autoround2/turbine-full.json` → `tests/eval/llama-3.2-3b-instruct/turbine-gptq-autoround-full.json`,
   `$REMOTE/gptq-own/turbine-full.json` → `turbine-gptq-own-full.json`, `vllm-full.json` → `vllm-gptq-own-full.json`;
2. `python3 scripts/eval/paired_compare.py tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json <each> --max-drop 0.04`
   (already in `*-paired.json` on the host);
3. check the own checkpoint's `config.json` quantization_config (one group, targets Linear, sym,
   group 128, actorder null, ignore lm_head) and copy the versions of `turbine_calibration.json`
   into `tests/eval/llama-3.2-3b-instruct/README.md`; if Turbine refused it, the loader change is a
   lead-owned `handoff(<file>)` commit with a test first;
4. delete `/home/piwi/turbine-ci/scratch/gptq-own/uv-cache` and any `*-gptq-own.tmp`.
