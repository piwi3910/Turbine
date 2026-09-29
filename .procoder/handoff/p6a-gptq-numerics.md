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
