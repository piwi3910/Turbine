#!/usr/bin/env bash
# Build turbine gdn_eval (Phase 7 Task 29) on a ROCm host from checkouts of llama.cpp and AITER
# (both MIT; nothing of theirs is committed here). Evaluated at:
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp  # bd4eeaa047006cb1fe71999fbd11134b5836e167
#   git clone --depth 1 https://github.com/ROCm/aiter          # 4bb520baca1d39ad6cee99e0261fe19d1778ec14
# GDN_EVAL_SRC names the directory holding both clones. Objects already built are kept
# (delete build/ to rebuild everything). Run under scripts/bench-lock.sh:
#   LD_LIBRARY_PATH=/opt/rocm/rocm/lib ROCR_VISIBLE_DEVICES=0 ./build/gdn_eval [all|decode|prefill|conv]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="${GDN_EVAL_SRC:-/home/piwi/turbine-ci/scratch/gdn-eval}"
hipcc="${TURBINE_ROCM_PATH:-/opt/rocm/rocm}/bin/hipcc"
out="$here/build"
mkdir -p "$out/aiter_src"
cp "$here"/shim/aiter_*.h "$out/aiter_src/"
cp -p "$src/aiter/csrc/kernels/causal_conv1d_update.cu" \
	"$src/aiter/csrc/kernels/causal_conv1d_fwd_split_qkv.cu" \
	"$src/aiter/csrc/include/chunk_gated_delta_rule_fwd_h.h" \
	"$src/aiter/csrc/include/causal_conv1d_update.h" \
	"$src/aiter/csrc/include/causal_conv1d_fwd_split_qkv.h" "$out/aiter_src/"
{
	sed -n '1,/^#define LAUNCH_HIP_KERNEL/p' "$src/aiter/csrc/kernels/chunk_gated_delta_rule_fwd_h.cu" | sed '$d'
	cat "$here/chunk_h_launcher.inc"
} >"$out/aiter_src/chunk_h_eval.cu"
g="$src/llama.cpp/ggml/src/ggml-cuda"
{
	echo '// extracted from llama.cpp ggml-cuda/gated_delta_net.cu and ssm-conv.cu (MIT)'
	sed -n '/^template <int S_v, bool KDA, bool keep_rs_t>/,/^static void ggml_cuda_op_gated_delta_net_impl/p' "$g/gated_delta_net.cu" | sed '$d'
	sed -n '/^template <bool apply_silu, size_t split_d_inner, size_t d_conv>/,/^void ggml_cuda_op_ssm_conv/p' "$g/ssm-conv.cu" | sed '$d'
} >"$out/llama_kernels.inc"
flags=(-O3 -std=c++20 --offload-arch=gfx1201 -fopenmp -I"$out/aiter_src" -I"$here/shim" -I"$out")
pids=()
for f in chunk_h_eval causal_conv1d_update causal_conv1d_fwd_split_qkv; do
	[[ -f "$out/$f.o" && "$out/$f.o" -nt "$out/aiter_src/$f.cu" && $f != chunk_h_eval ]] && continue
	extra=()
	[[ $f == chunk_h_eval ]] && extra=(-mwavefrontsize64)
	"$hipcc" "${flags[@]}" "${extra[@]}" -x hip -c "$out/aiter_src/$f.cu" -o "$out/$f.o" &
	pids+=($!)
done
"$hipcc" "${flags[@]}" -x hip -c "$here/gdn_eval.hip" -o "$out/gdn_eval.o" &
pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
"$hipcc" "${flags[@]}" "$out"/*.o -o "$out/gdn_eval"
echo "built $out/gdn_eval"
