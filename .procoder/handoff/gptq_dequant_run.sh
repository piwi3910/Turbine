#!/usr/bin/env bash
# CPU-only GPTQ dequant cross-check (p6a-gptq-numerics part b). Runs on novanas, detached, at
# nice 19 on cores 12-15 (no GPU, no GPU lock). For layers 0, 13 and 27, every q/k/v/o and
# gate/up/down projection: Turbine's CPU dequant (int4_layer_dump example, built by
# scripts/remote-cargo.sh) vs an independent transcription of AutoGPTQ's dequant
# (scripts/golden/gptq_dequant_check.py), bit for bit, plus the relative error of the GPTQ weight
# and of a round-to-nearest of the same scheme against the BF16 original. One JSON line per layer
# in $OUT/results.jsonl; last log line
# "gptq-dequant: done rc=<rc>" (rc 0: every layer bit-exact).
set -uo pipefail
REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics
SRC=$REMOTE/src
DUMP=$REMOTE/target/release/examples/int4_layer_dump
M=/home/piwi/turbine-models
OUT=$REMOTE/gptq-dequant
mkdir -p "$OUT"
cd "$SRC" || exit 1
{
	echo "== $(date -u +%FT%TZ) gptq_dequant_run.sh pid $$"
	rc=0
	: >"$OUT/results.jsonl"
	for l in 0 13 27; do
		for spec in self_attn.q_proj:3072:3072 self_attn.k_proj:1024:3072 self_attn.v_proj:1024:3072 \
			self_attn.o_proj:3072:3072 mlp.gate_proj:8192:3072 mlp.up_proj:8192:3072 mlp.down_proj:3072:8192; do
			IFS=: read -r proj n k <<<"$spec"
			layer=model.layers.$l.$proj
			f=$OUT/dump.f32
			if ! nice -n 19 taskset -c 12-15 "$DUMP" "$M/llama-3.2-3b-instruct-gptq" gptq "$layer" "$n" "$k" "$f"; then
				echo "dump failed: $layer"
				rc=1
				continue
			fi
			nice -n 19 taskset -c 12-15 /home/piwi/.local/bin/uv run --quiet scripts/golden/gptq_dequant_check.py \
				--gptq-dir "$M/llama-3.2-3b-instruct-gptq" --layer "$layer" --turbine "$f" \
				--bf16-dir "$M/llama-3.2-3b-instruct" --json \
				>>"$OUT/results.jsonl"
			r=$?
			echo "$layer check rc=$r"
			[ "$r" = 0 ] || rc=1
			rm -f "$f"
		done
	done
	echo "gptq-dequant: done rc=$rc"
} >>"$OUT/run.log" 2>&1
