# Handoff: p6a-w4a4-numerics (Task 20 W4A4 accuracy investigation)

Written 2026-09-29 by the W4A4 numerics investigator. Worktree
`.claude/worktrees/agent-p6a-w4a4-numerics`, branch `p6a-w4a4-numerics` (from 7277c8b).

## Root cause: RoPE config not read (not an MXFP4 numerics bug)

`amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ` @ 00b0d018 was exported with transformers 5.9.
Its `config.json` has **no top-level `rope_theta` and no `rope_scaling`**; both are folded into one
`rope_parameters` object:

    rope_parameters: {factor 8.0, high_freq_factor 4.0, low_freq_factor 1.0,
                      original_max_position_embeddings 8192, rope_theta 500000.0, rope_type llama3}

(the BF16 `llama-3.1-8b-instruct` config has the old `rope_theta` + `rope_scaling` keys).
`crates/turbine-model/src/config.rs` reads only the old keys (`RawConfig`, lines ~298–322) and
**silently** falls back to `DEFAULT_ROPE_THETA = 10_000.0` (line 108, used at line 435) with no
llama3 scaling. Turbine therefore served the W4A4 checkpoint with theta 10 000 instead of 500 000
and no llama3 frequency scaling. The rotary table is wrong at every position. Nothing is logged
about it (the server log has no rope line besides the kernel selection).

Evidence:
- The config key listing on novanas: top-level keys include `rope_parameters` and no `rope_theta`
  or `rope_scaling`. It is the only model under `/home/piwi/turbine-models` with `rope_parameters`.
- Turbine and vLLM outputs on GSM8K-200 differ from the very first sentence of every item: 0 of 200
  share the first 200 characters. They are still coherent, which fits a wrong RoPE base rather than
  a broken GEMM.
- vLLM 0.23 (transformers 5 in the image) reads `rope_parameters`, so its 0.735 has the right RoPE.
- **The pending fixq reference would have the same bug.** transformers 4.57.1, the pinned version in
  `quant_reference.py` / `hf_reference.py`, gives `LlamaConfig(**config)` →
  `rope_theta 10000.0, rope_scaling None` for this checkpoint. The BF16 8B config gives 500000 +
  llama3. This was checked in the `self-spread` uv env on novanas. A reference built from the raw
  directory would agree with Turbine's bug and hide it.
- Confirmation: GSM8K-200 with the RoPE keys patched in scores 0.740 against vLLM 0.735 (Results).

Pinning test: `crates/turbine-model/tests/config_rope_parameters.rs`. It takes the 3B fixture config,
rewrites it the transformers-5 way and expects theta 500 000 + Llama3 scaling. It is `#[ignore]`d
so the gate stays green. `cargo test -p turbine-model --test config_rope_parameters -- --ignored`
fails today (see Results).

## Fix (lead-owned `config.rs`; being done by a builder on `p6a-rope-parameters`)

Proposed: `RawConfig` reads `rope_parameters`. Take `rope_theta` from it when the top-level key is
absent. When `rope_scaling` is absent, pass `rope_parameters` minus `rope_theta` to
`parse_rope_scaling`. Refuse a mix of old and new keys that disagree, and never default
`rope_theta` silently. When the fix lands, un-ignore `tests/config_rope_parameters.rs`.

## Golden reference: this branch's fixture fix + the exact command (lead queues it)

The raw checkpoint directory cannot give a valid reference, for two reasons:
1. transformers 4.57.1 reads theta 10000 and no scaling (above).
2. `dequantize_checkpoint.py` refused it: its Quark detect refused any `layer_quant_config`
   (this checkpoint has the `*k_proj`/`*v_proj` KV mirrors), and its copy loop raised
   `unexpected tensor of a quantized layer` on `k_proj.output_scale` / `v_proj.output_scale`.

This branch fixes (2) in `scripts/golden/dequantize_checkpoint.py`, the same way Turbine's parser
does. It accepts only the K/V overrides that equal `kv_cache_quant_config[pattern]` and share the
global weight/input spec; any other override is still refused (checked with `detect()` on this
config and on a mutated one). It also skips the K/V `output_scale` scalars.
For (1), use the patched directory `/home/piwi/turbine-ci/scratch/w4a4-numerics/model/`. It holds
symlinks to the checkpoint's weights, tokenizer, template and `.cache` revision metadata, plus a
`config.json` that adds top-level `rope_theta: 500000.0` and `rope_scaling` = `rope_parameters`
minus theta; the quantization_config is unchanged. Command, run from a tree that has this branch's
`scripts/golden/` (fixture-queue style, CPU only):

    cd <tree with this branch's scripts/golden> && W=$(mktemp -d /dev/shm/turbine-mxfp4-a4-XXXXXX) && \
    flock /home/piwi/turbine-ci/fixture.lock nice -n 19 taskset -c 12-15 \
      env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
      timeout 24h /home/piwi/.local/bin/uv run scripts/golden/quant_reference.py \
        --model-dir /home/piwi/turbine-ci/scratch/w4a4-numerics/model \
        --prompts tests/golden/prompts.jsonl \
        --out <out>/llama-3.1-8b-instruct-mxfp4-a4.reference.jsonl \
        --act-quant mxfp4 --model-name amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ \
        --work-dir "$W" --keep-dequantized

Leave `--kv-quant` at its default `none`, i.e. BF16 KV, as Turbine and vLLM (`--kv-cache-dtype
auto`) ran. Check the reference log's `rotary` line: it must say llama3, not default.
The spread then runs on `$W/*/bf16` as `fixtures_r5.sh` section 3 does (`self_spread.py … --act-quant mxfp4`).

## Lines checked and cleared

- **Activation quant-dequant** (`kernels/rocm/src/quantize_act_mxfp4.hip`, `cpu::quant`) matches
  Quark's `even` rule: amax + 2^21 on the F32 bits, masked to the exponent, then floor(log2) − 2,
  clamped to [−127, 127]. The block is 32 along the last axis (k); E2M1 rounding is nearest with
  ties to the even code, saturating at 6; NaN amax gives e = 255; a zero block gives 2^−127 and zeros.
- **GPTQ** `desc_act` + `static_groups`: groups are static, so the export is in original column
  order. The checkpoint has no `g_idx` tensor, only `weight` U8 `[n, k/2]` + `weight_scale` U8
  `[n, k/32]`.
- **SmoothQuant** is folded into the norms and the v/up weights: there are no smooth-scale tensors.
- **KV recipe** (line 4): the vLLM Job ran `--kv-cache-dtype auto` (q4n/vllm-job.yaml), so vLLM's
  0.735 used BF16 KV and did not apply the `k_proj`/`v_proj` `output_scale`. Ignoring them in
  Turbine (d1de280) does not explain the gap.
- The chat template matches the BF16 8B one (only a trailing newline differs), and the EOS ids are
  the same.
- Not needed given the root cause: `turbine-golden positions` / trace. The fixq W4A4 reference
  never ran: it is queued, and it is invalid as-is (see above).

## Confirmation run (done)

`/home/piwi/turbine-ci/scratch/w4a4-numerics/w4a4rope.sh` runs GSM8K-200 at c1 with the **same frozen
binaries** as the 0.54 run (q4n/bin `turbine-server`, `libturbine_hip.so`, `turbine-golden.c1`).
Only `config.json` differs (the patched model directory above). It takes the locks port18000 →
bench.gate → bench.lock and runs on GPU 0, cores 0–11.

- Log: `/home/piwi/turbine-ci/scratch/w4a4-numerics/w4a4rope.log`, which ends with `w4a4rope: done rc=<rc>`.
- Result: `/home/piwi/turbine-ci/scratch/w4a4-numerics/llama8b-a4-rope.json`, server log `server.log`.
- Judge: accuracy near vLLM's 0.735 (|Δ| ≤ ~0.04) confirms that the RoPE config is the whole gap.
  A result still well below would mean a second cause: then run the fixq reference on the patched
  directory, then `turbine-golden compare` / `positions`.

## Results

- **GSM8K-200 at c1 with the RoPE-patched config: 148/200 = 0.740** (`w4a4rope: done rc=0`, 13:52–14:01Z).
  vLLM on the same checkpoint scores 0.735 (147/200); before the patch Turbine scored 0.54 (108/200).
  Per item against vLLM: both correct 128, Turbine only 20, vLLM only 19, so the two agree within
  noise. The whole 0.195 gap was the RoPE config. The MXFP4 act quant-dequant, the E8M0 rounding,
  act-order and the ignored FP8 KV are not implicated. Result JSON:
  `/home/piwi/turbine-ci/scratch/w4a4-numerics/llama8b-a4-rope.json`. It is not committed as
  `tests/eval/.../turbine.json`, because it ran on a patched config; rerun the eval on the raw
  directory once the `config.rs` fix lands, and commit that.
- Residual to confirm with the golden reference: only 4/200 answers share their first 200 characters
  with vLLM's. That is plausible for W4A4 with different GEMM accumulation orders (vLLM computes BF16
  after dequantization, Turbine uses `turbine_hip_mxfp4`), but it is unmeasured. Judge it with
  `turbine-golden compare` against the rebuilt reference, using the calibrated spread with the
  BF16-floor rule.
- **Segfault at process exit (a finding; the eval is valid).** Timeline in `server.log`: last request
  finished 14:01:16.512Z; the eval had written all 200 results. The script's SIGTERM came at 16.548,
  then `shutdown_drained`, `engine_shutdown`, and `shutdown complete` at 16.560. After that the
  process died with SIGSEGV, which bash reports as `Segmentation fault` for the `setsid bash -c
  "exec turbine-server"` job. So it is past `main`'s last log line, in process teardown (static
  destructors / HIP runtime or kernel-library unload are the likely suspects). There is no core dump
  (no coredumpctl on novanas) and no backtrace. The same frozen q4n binaries (13:19 build) show no
  `Segmentation` line in `q4n/queue4n.log` or `t20/t20-bench.log`, but those scripts launch the
  server the same way, so the crash is intermittent or specific to this run (the only difference is
  a model directory of symlinks). Next step if it matters: rerun with `ulimit -c unlimited` and
  `catchsegv`/gdb batch on the current binaries.
- Pinning test: `cargo test -p turbine-model --test config_rope_parameters -- --ignored` fails on
  7277c8b with `left: 10000.0 right: 500000.0` (run on novanas via remote-cargo).
