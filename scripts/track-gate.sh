#!/usr/bin/env bash
# Track start gate (umbrella phase-6-8-expansion S-1, amended by the Phase 6 split of 2026-09-28):
# a track's implementation starts only when
#   1. the previous track closed (a `supported` amd support-matrix row carries its feature),
#   2. its spec passes `procoder spec check` (COMPLETE) and reads `Status: complete`,
#   3. its spec's "## In scope" section stays inside the umbrella scope (S-6, S-7, S-8).
# Order: phase-6a-quantization → phase-6b-kv-compression → (phase-5p) → phase-7-model-families →
#        phase-8-speculative-decoding. NVIDIA rows never count (phase-2b-nvidia is deferred).
# Usage: scripts/track-gate.sh <phase-6a-quantization|phase-6b-kv-compression|phase-7-model-families|phase-8-speculative-decoding>
# Env:   TURBINE_PROCODER_LAUNCHER  procoder launcher (default: the 3.7.0 plugin launcher)
#        TURBINE_SPEC_DIR           spec directory (default: .procoder/specs)
#        TURBINE_SUPPORT_MATRIX     file holding `turbine-server --support-matrix --output text`
#                                   (default: produced with `cargo run -q -p turbine-server`)
# Exit:  0 gate passed, 1 gate failed (every failure printed), 2 usage.
set -euo pipefail

tracks="phase-6a-quantization|phase-6b-kv-compression|phase-7-model-families|phase-8-speculative-decoding"
track="${1:-}"
case "$track" in
phase-6a-quantization | phase-6b-kv-compression | phase-7-model-families | phase-8-speculative-decoding) ;;
*)
	echo "usage: $0 <$tracks>" >&2
	exit 2
	;;
esac

launcher="${TURBINE_PROCODER_LAUNCHER:-$HOME/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh}"
spec_dir="${TURBINE_SPEC_DIR:-.procoder/specs}"
spec="$spec_dir/$track.md"
failures=0

fail() {
	echo "GATE FAIL $track: $*"
	failures=$((failures + 1))
}

matrix_file="${TURBINE_SUPPORT_MATRIX:-}"
if [ -z "$matrix_file" ]; then
	matrix_file="$(mktemp)"
	trap 'rm -f "$matrix_file"' EXIT
	cargo run -q -p turbine-server -- --support-matrix --output text >"$matrix_file"
fi

# supported_rows <kind>: amd rows with status `supported` ($1..$7 = vendor arch architecture
# weight kv speculative status) that carry the feature of a closed track.
supported_rows() {
	awk -v kind="$1" 'NR > 1 && $1 == "amd" && $7 == "supported" {
    if (kind == "quantized" && ($4 != "bf16" || $5 == "fp8_e4m3")) n++
    if (kind == "kv_compression" && ($5 == "tq4" || $5 == "tq2")) n++
    if (kind == "family" && $3 != "LlamaForCausalLM" && $3 != "OlmoeForCausalLM") n++
  } END { print n + 0 }' "$matrix_file"
}

case "$track" in
phase-6b-kv-compression)
	[ "$(supported_rows quantized)" -gt 0 ] ||
		fail "phase-6a-quantization has not closed: no supported amd row with a quantized weight format or fp8_e4m3 KV"
	;;
phase-7-model-families)
	[ "$(supported_rows kv_compression)" -gt 0 ] ||
		fail "phase-6b-kv-compression has not closed: no supported amd row with tq4 or tq2 KV"
	;;
phase-8-speculative-decoding)
	[ "$(supported_rows family)" -gt 0 ] ||
		fail "phase-7-model-families has not closed: no supported amd row for a Phase 7 family"
	;;
esac

if [ ! -f "$spec" ]; then
	fail "$spec does not exist; write it with /procoder:spec $track"
else
	if check_out="$("$launcher" spec check "$track" 2>&1)" && printf '%s\n' "$check_out" | grep -q 'COMPLETE'; then
		:
	else
		fail "procoder spec check is not COMPLETE: $(printf '%s' "$check_out" | head -n 1)"
	fi
	grep -qx 'Status: complete' "$spec" || fail "Status line is not 'Status: complete'"

	in_scope="$(awk '/^## In scope/ { f = 1; next } /^## / { f = 0 } f' "$spec")"
	[ -n "$in_scope" ] || fail "no '## In scope' section"

	# need <description> <extended regex>: the In scope section must match (case-insensitive).
	need() { printf '%s\n' "$in_scope" | grep -Eqi "$2" || fail "In scope does not cover $1 (/$2/)"; }
	# refuse <description> <extended regex>: the In scope section must not match.
	refuse() { if printf '%s\n' "$in_scope" | grep -Eqi "$2"; then fail "In scope names $1, outside the umbrella scope"; fi; }

	case "$track" in
	phase-6a-quantization)
		[ "$(grep -c -E 'fp8_block|mxfp4|awq_int4|gptq_int4|fp8_e4m3' "$spec" || true)" -gt 0 ] ||
			fail "spec names none of fp8_block, mxfp4, awq_int4, gptq_int4, fp8_e4m3"
		need "block-scaled FP8" 'fp8_block'
		need "MXFP4" 'mxfp4'
		need "AWQ" 'awq_int4'
		need "GPTQ" 'gptq_int4'
		need "FP8 KV" 'fp8_e4m3'
		need "YaRN" 'yarn'
		refuse "NVFP4" 'nvfp4'
		refuse "GGUF" 'gguf'
		;;
	phase-6b-kv-compression)
		need "TurboQuant" 'turboquant'
		need "tq4" 'tq4'
		need "per-tier KV formats" 'kv\.cpu\.format'
		need "the compression ladder" 'ladder'
		refuse "NVFP4" 'nvfp4'
		refuse "GGUF" 'gguf'
		;;
	phase-7-model-families)
		need "Qwen3 dense" 'qwen3 dense|Qwen3ForCausalLM'
		need "Qwen3 MoE" 'qwen3 moe|Qwen3MoeForCausalLM'
		need "gpt-oss-20b" 'gpt-oss'
		need "the Qwen3.5/3.6 hybrids" 'qwen3\.5|qwen3\.6|qwen3_5'
		need "Gated DeltaNet" 'gated deltanet'
		need "Mistral" 'mistral'
		need "Mixtral" 'mixtral'
		need "the AMD linear-attention kernel provider" 'linear-attention.*(amd|gfx1201|hip|rocm)|(amd|gfx1201|hip|rocm).*linear-attention'
		;;
	phase-8-speculative-decoding)
		grep -q 'Llama-3.2-1B-Instruct' "$spec" || fail "spec does not name Llama-3.2-1B-Instruct as the first draft model"
		refuse "MTP heads" '(^|[^a-z])mtp([^a-z]|$)'
		refuse "EAGLE" 'eagle'
		refuse "DFlash" 'dflash'
		;;
	esac
fi

if [ "$failures" -gt 0 ]; then
	echo "GATE FAIL $track: $failures check(s) failed"
	exit 1
fi
echo "GATE PASS $track"
