# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==4.57.1",
#     "safetensors==0.6.2",
#     "jinja2==3.1.6",
#     "numpy==2.3.4",
# ]
# ///
"""Golden reference for a quantized checkpoint (Phase 6a S-11, fixture time only).

Decodes the checkpoint's quantized linear layers exactly as Turbine's CPU reference does into a
temporary BF16 checkpoint (`dequantize_checkpoint.py`), then runs the transformers reference of
`hf_reference.py` on it (same prompts, greedy decode for exactly `max_tokens`, FP32 LM head on the
BF16 final-norm output, same JSONL records) with activation fake-quantization: a forward
pre-hook on every decoded Linear quantize-dequantizes its input exactly as
`cpu::quant::quantize_dequantize_activations` does, in F32, and hands the result back in BF16.

`--act-quant` (default `auto`: the checkpoint's own activation scheme, from
`turbine_dequant.json`):
- `none`          weight-only (W4A16, W8A16);
- `fp8_token`     FP8 e4m3 per row (token), scale `max(amax / 448, 1 / (448 × 512))`;
- `fp8_tensor`    FP8 e4m3 with the layer's static `input_scale` from the checkpoint; a layer
                  Turbine runs fused (q/k/v, gate/up) takes the largest `input_scale` of its
                  parts, as Turbine's loader and vLLM do (`--input-scale per-part` keeps each
                  layer's own);
- `fp8_group128`  FP8 e4m3 per row and group of 128 columns, dynamic like `fp8_token`;
- `mxfp4`         MXFP4 per 32 columns: E8M0 scale from Quark's `even` rule, E2M1 round to
                  nearest even, saturated to ±6 (Quark W4A4).
FP8 values round to nearest even and saturate to ±448 before the cast.

The record's `engine` is `transformers-<v>-bf16-<device>-dequant-<packaging>-act-<mode>` plus
`-fp32-logits`; `model` is `--model-name`, else the checkpoint's `_name_or_path` when it is a
Hub id (not a local path), else its directory name (pass `--model-name <hub-id>` for the
committed fixtures); `revision` is the Hub commit `hf download --local-dir` recorded.

Output goes to `<out>.tmp`, renamed over `<out>` only when every prompt succeeded; any failure
exits 1 and leaves no file. The dequantized copy lives in `--work-dir` (default: a temporary
directory; on `novanas` use `/dev/shm`) and is removed at the end unless `--keep-dequantized`;
`--dequantized <dir>` reuses a copy written earlier.

`--kv-quant fp8_e4m3` (default `none`) emulates Turbine's FP8 e4m3 L0 KV cache (`kv.dtype:
fp8_e4m3`, Phase 6a S-13): every attention call sees K (after RoPE) and V quantize-dequantized
per layer as a page write and read, `bf16(e4m3(x / scale) · scale)` with FP8 rounding to nearest
even saturated to ±448, the layer's `self_attn.k_scale` / `v_scale` (or `k_proj` /
`v_proj.output_scale`) from the checkpoint when it stores them, else 1.0. The hook is a view of
the KV cache handed to every attention layer (`install_kv_quant`), so it holds for every family
and attention implementation (eager and sdpa alike, and OLMoE's per-class attention in 4.57.1).
The cache keeps the unquantized values and every call quantizes the whole K/V again, which gives
the same values as quantizing once (each element's result depends on that element only). A checkpoint without a
`quantization_config` (BF16) is then run as it is, without the dequantized copy (`--act-quant`
must be `auto` or `none`). The engine gains `-kv-fp8_e4m3`.

`--self-test` checks the vectorized torch quantizers used by the hooks against scalar ports of
the Rust functions (every BF16 input, random F32 inputs, the Rust unit-test table), then the FP8
KV emulation on tiny random Llama and OLMoE models (eager equals sdpa, the KV quantization is
applied, per-layer scales reach their layer), and exits.

Usage:
    uv run scripts/golden/quant_reference.py --model-dir <quantized-dir> \
        --prompts tests/golden/prompts.jsonl --out tests/golden/<slug>/reference.jsonl \
        [--act-quant auto|none|fp8_token|fp8_tensor|fp8_group128|mxfp4] [--top-logprobs 20] \
        [--work-dir /dev/shm] [--dequantized <dir>] [--keep-dequantized] [--model-name <hub-id>]
        [--input-scale fused-max|per-part] [--chat-template <file.jinja>]
        [--kv-quant none|fp8_e4m3]
    uv run scripts/golden/quant_reference.py --self-test
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dequantize_checkpoint as dq
import hf_reference

ACT_MODES = ["auto", "none", "fp8_token", "fp8_tensor", "fp8_group128", "mxfp4"]
KV_MODES = ["none", "fp8_e4m3"]
FP8_MAX = 448.0
E2M1_MAX = 6.0
MX_BLOCK = 32


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors are failures too: exit 1
        self.print_usage(sys.stderr)
        print(f"quant_reference.py: error: {message}", file=sys.stderr)
        sys.exit(1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = _Parser(
        description="Golden reference of a quantized checkpoint (exact BF16 decode)."
    )
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--prompts", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--act-quant", choices=ACT_MODES, default="auto")
    p.add_argument("--top-logprobs", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--model-name")
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--dequantized", type=Path, help="reuse this dequantized copy")
    p.add_argument("--keep-dequantized", action="store_true")
    p.add_argument(
        "--input-scale",
        choices=["fused-max", "per-part"],
        default="fused-max",
        help="fp8_tensor: a fused projection's parts share their largest input_scale",
    )
    p.add_argument(
        "--chat-template",
        type=Path,
        help="render chat prompts with this Jinja template instead of the checkpoint's "
        "(a base checkpoint without one; the served config names the same file)",
    )
    p.add_argument("--kv-quant", choices=KV_MODES, default="none")
    args = p.parse_args(argv)
    if not args.self_test:
        missing = [
            f for f in ("model_dir", "prompts", "out") if getattr(args, f) is None
        ]
        if missing:
            p.error(
                "required: " + ", ".join("--" + m.replace("_", "-") for m in missing)
            )
    if not 1 <= args.top_logprobs <= 20:
        p.error("--top-logprobs must be between 1 and 20")
    return args


# ---------------------------------------------------------------------------------------------
# Vectorized quantize-dequantize (torch, F32), as cpu::quant::quantize_dequantize_activations


def fp8_min_scale(torch):
    return torch.tensor(1.0, dtype=torch.float32) / torch.tensor(
        FP8_MAX * 512.0, dtype=torch.float32
    )


def fp8_qdq(torch, x, scale):
    """fp8_e4m3_value(fp8_e4m3_round(x / scale)) × scale, all F32."""
    q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return q * scale


def dynamic_fp8_scale(torch, x):
    """max(amax / 448, 1 / (448 × 512)) over the last axis (kept)."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    return torch.maximum(amax / FP8_MAX, fp8_min_scale(torch))


def mxfp4_scale_even(torch, amax):
    """E8M0 byte (as int32) of Quark's `even` rule for a non-negative F32 `amax` tensor."""
    bits = amax.contiguous().view(torch.int32)
    exp = (
        (bits + (1 << 21)) & 0x7F800000
    ) >> 23  # amax ≥ 0: no sign bit, no int32 overflow
    log2 = torch.where(exp == 255, torch.full_like(exp, 32767), exp - 127)
    log2 = torch.where(exp == 0, torch.full_like(exp, -127), log2)
    e = (log2 - 2).clamp(-127, 127) + 127
    return torch.where(torch.isnan(amax), torch.full_like(e, 255), e)


def e2m1_qdq_unit(torch, y):
    """e2m1_value(e2m1_round(y)): nearest E2M1 value, ties to the even code, ±6 saturation,
    NaN → 0, a result of 0 carries no sign."""
    a = torch.nan_to_num(y.abs(), nan=0.0).clamp(max=E2M1_MAX)
    m = torch.zeros_like(a)
    m = torch.where(a > 0.25, 0.5, m)
    m = torch.where(a >= 0.75, 1.0, m)
    m = torch.where(a > 1.25, 1.5, m)
    m = torch.where(a >= 1.75, 2.0, m)
    m = torch.where(a > 2.5, 3.0, m)
    m = torch.where(a >= 3.5, 4.0, m)
    m = torch.where(a > 5.0, 6.0, m)
    m = torch.where(torch.isnan(y), 0.0, m)
    return torch.where(m == 0, torch.zeros_like(m), torch.copysign(m, y))


def qdq_activations(torch, x, mode: str, input_scale: float | None = None):
    """Quantize-dequantizes F32 `x` ([..., cols]) along its last axis."""
    if mode == "none":
        return x
    if mode == "fp8_tensor":
        return fp8_qdq(torch, x, torch.tensor(input_scale, dtype=torch.float32))
    if mode == "fp8_token":
        return fp8_qdq(torch, x, dynamic_fp8_scale(torch, x))
    cols = x.shape[-1]
    group = 128 if mode == "fp8_group128" else MX_BLOCK
    if cols % group:
        raise ValueError(f"{mode}: {cols} columns are not a multiple of {group}")
    g = x.reshape(*x.shape[:-1], cols // group, group)
    if mode == "fp8_group128":
        out = fp8_qdq(torch, g, dynamic_fp8_scale(torch, g))
    elif mode == "mxfp4":
        s = dq.e8m0_values(
            torch, mxfp4_scale_even(torch, g.abs().amax(dim=-1, keepdim=True))
        )
        out = e2m1_qdq_unit(torch, g / s) * s
    else:
        raise ValueError(f"activation mode {mode!r}")
    return out.reshape(x.shape)


# The projections Turbine's decoder runs as one fused GEMM on one quantized input: a static
# per-tensor activation scale is shared by the parts (the largest of theirs).
FUSED_PARTS = re.compile(
    r"^(?P<prefix>.*)\.(?:(?P<qkv>[qkv])_proj|(?P<gu>gate|up)_proj)$"
)


def fused_group(name: str) -> str:
    m = FUSED_PARTS.match(name)
    if m is None:
        return name
    return m.group("prefix") + (".qkv_proj" if m.group("qkv") else ".gate_up_proj")


def input_scales(layers: dict, fused_max: bool) -> dict:
    """Each layer's static input_scale, a fused group's parts sharing their largest."""
    own = {name: entry.get("input_scale") for name, entry in layers.items()}
    if not fused_max:
        return own
    groups: dict = {}
    for name, scale in own.items():
        if scale is not None:
            key = fused_group(name)
            groups[key] = max(groups.get(key, scale), scale)
    return {
        name: (groups[fused_group(name)] if scale is not None else None)
        for name, scale in own.items()
    }


def install_hooks(torch, model, layers: dict, mode: str, fused_max: bool = True) -> int:
    """A forward pre-hook per decoded Linear: BF16 input → F32 → quantize-dequantize → BF16."""
    if mode == "none":
        return 0
    scales = input_scales(layers, fused_max)
    count = 0
    for name in layers:
        if name == "lm_head":
            raise ValueError(
                "lm_head is quantized: the FP32 LM head of hf_reference bypasses it"
            )
        module = model.get_submodule(name)
        scale = scales[name]
        if mode == "fp8_tensor" and scale is None:
            raise ValueError(f"{name}: fp8_tensor needs the checkpoint's input_scale")

        def hook(_module, args, scale=scale):
            x = args[0]
            y = qdq_activations(torch, x.to(torch.float32), mode, scale).to(x.dtype)
            return (y, *args[1:])

        module.register_forward_pre_hook(hook)
        count += 1
    return count


# ---------------------------------------------------------------------------------------------
# FP8 KV cache emulation (Phase 6a S-13), as the kv.dtype fp8_e4m3 pages of the CPU reference


def kv_scales(model_dir: Path, num_layers: int) -> list[tuple[float, float]]:
    """Per-layer (k_scale, v_scale): the checkpoint's `self_attn.{k,v}_scale` (or
    `{k,v}_proj.output_scale`) scalars when it stores them for every layer, 1.0 when it stores
    none, an error when only some (turbine_model::kv_scales::KvCache::fp8_from_checkpoint)."""
    from safetensors import safe_open

    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(
            set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        )
    else:
        files = ["model.safetensors"]
    values: dict[str, float] = {}
    for name in files:
        with safe_open(str(model_dir / name), framework="pt") as f:
            for key in f.keys():
                if key.endswith(("_scale", ".output_scale")) and ".self_attn." in key:
                    values[key] = float(f.get_tensor(key).float().reshape(-1)[0])
    out: list[tuple[float, float]] = []
    found = missing = 0
    for layer in range(num_layers):
        pair = []
        for half in ("k", "v"):
            p = f"model.layers.{layer}.self_attn"
            v = values.get(
                f"{p}.{half}_scale", values.get(f"{p}.{half}_proj.output_scale")
            )
            if v is None:
                missing += 1
                v = 1.0
            else:
                found += 1
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"{p}.{half}_scale = {v}: not finite and positive")
            pair.append(v)
        out.append((pair[0], pair[1]))
    if found and missing:
        raise ValueError(
            f"the checkpoint stores {found} of the {2 * num_layers} per-layer KV scales"
        )
    return out


class _Fp8KvCache:
    """A view of a transformers KV cache whose `update` hands the attention its layer's whole K
    and V quantize-dequantized, while the cache itself keeps the unquantized values."""

    def __init__(self, torch, cache, scales: list[tuple[float, float]]):
        self._torch = torch
        self._cache = cache
        self._scales = scales

    def _qdq(self, x, scale: float):
        torch = self._torch
        s = torch.tensor(scale, dtype=torch.float32)
        return fp8_qdq(torch, x.to(torch.float32), s).to(x.dtype)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = self._cache.update(key_states, value_states, layer_idx, cache_kwargs)
        ks, vs = self._scales[layer_idx]
        return self._qdq(k, ks), self._qdq(v, vs)

    def __getattr__(self, name):
        return getattr(self._cache, name)


def install_kv_quant(torch, model, scales: list[tuple[float, float]]) -> str:
    """Makes every attention layer see its K (after RoPE) and V quantize-dequantized with its
    layer's scales (in F32, back in the tensors' dtype), whatever the attention implementation.

    The hook sits on the KV cache, not on the attention function: a forward pre-hook on each
    layer's `self_attn` hands it a `_Fp8KvCache` view of `past_key_values`, whose `update` returns
    the whole K/V quantize-dequantized. Every family calls `past_key_values.update` right after
    RoPE and before its attention kernel, both the families on transformers' AttentionInterface
    (Llama) and those that still pick an attention class at construction (OLMoE in 4.57.1, which
    never calls a registered attention function), and the model's attention implementation and
    its mask stay untouched (a custom AttentionInterface name gets no mask at all, which left the
    eager path non-causal). An attention call without a cache raises: run with `use_cache=True`.
    Returns a description for the log."""
    layers = {}
    for name, module in model.named_modules():
        if name.endswith(".self_attn") and hasattr(module, "layer_idx"):
            if module.layer_idx in layers:
                raise ValueError(f"{name}: layer {module.layer_idx} seen twice")
            layers[module.layer_idx] = module
    if sorted(layers) != list(range(len(scales))):
        raise ValueError(
            f"FP8 KV: {len(scales)} layer scale pairs, attention layers {sorted(layers)}"
        )

    def hook(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        if cache is None:
            raise ValueError(
                f"FP8 KV emulation: layer {module.layer_idx} ran without a KV cache "
                "(call the model with use_cache=True)"
            )
        if not isinstance(cache, _Fp8KvCache):
            kwargs = dict(kwargs, past_key_values=_Fp8KvCache(torch, cache, scales))
        return args, kwargs

    for module in layers.values():
        module.register_forward_pre_hook(hook, with_kwargs=True)
    return (
        f"fp8_kv_cache_view on {len(layers)} attention layers "
        f"({model.config._attn_implementation} attention)"
    )


def kv_self_test() -> None:
    """install_kv_quant on tiny random Llama and OLMoE models: eager and sdpa agree (F32, with
    and without FP8 KV, prefill + incremental decode and one full forward), the FP8 KV changes
    the output, each layer uses its own scales, and a call without a cache is refused."""
    import copy

    import torch
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        OlmoeConfig,
        OlmoeForCausalLM,
    )

    torch.manual_seed(24)
    common = {
        "vocab_size": 97,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "initializer_range": 0.2,
        "tie_word_embeddings": False,
    }
    families = {
        "llama": (LlamaForCausalLM, LlamaConfig(**common)),
        "olmoe": (
            OlmoeForCausalLM,
            OlmoeConfig(**common, num_experts=4, num_experts_per_tok=2),
        ),
    }
    ids = torch.randint(0, 97, (1, 12))
    prompt, rest = ids[:, :7], ids[0, 7:]
    scales = [(0.011, 0.007), (0.023, 0.005)]

    def run(model, mode):
        with torch.inference_mode():
            if mode == "full":
                return model.model(input_ids=ids, use_cache=True).last_hidden_state[0]
            out = model.model(input_ids=prompt, use_cache=True)
            rows = [out.last_hidden_state[0]]
            for tok in rest:
                out = model.model(
                    input_ids=tok.view(1, 1),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
                rows.append(out.last_hidden_state[0])
            return torch.cat(rows)

    for fam, (cls, cfg) in families.items():
        state = cls(cfg).state_dict()
        outs = {}
        for attn in ("sdpa", "eager"):
            for kv in (None, scales, [scales[0], (0.5, 0.5)]):
                c = copy.deepcopy(cfg)
                c._attn_implementation = attn
                model = cls(c)
                model.load_state_dict(state)
                model.eval()
                assert model.config._attn_implementation == attn, (fam, attn)
                tag = "none" if kv is None else ("fp8" if kv is scales else "fp8-l1")
                if kv is not None:
                    install_kv_quant(torch, model, kv)
                for mode in ("incremental", "full"):
                    outs[(attn, tag, mode)] = run(model, mode)
                if kv is scales:
                    try:
                        with torch.inference_mode():
                            model.model(input_ids=prompt, use_cache=False)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(f"{fam}: a call without a cache ran")
        for tag in ("none", "fp8", "fp8-l1"):
            for mode in ("incremental", "full"):
                d = (outs[("sdpa", tag, mode)] - outs[("eager", tag, mode)]).abs().max()
                assert d < 1e-4, f"{fam} {tag} {mode}: eager vs sdpa differ by {d}"
            d = (outs[("sdpa", tag, "incremental")] - outs[("sdpa", tag, "full")]).abs()
            assert d.max() < 1e-4, (
                f"{fam} {tag}: incremental vs full differ by {d.max()}"
            )
        for mode in ("incremental", "full"):
            d = (outs[("sdpa", "fp8", mode)] - outs[("sdpa", "none", mode)]).abs().max()
            assert d > 1e-3, f"{fam} {mode}: FP8 KV changed the output by only {d}"
            # Each layer takes its own pair: changing only layer 1's changes the output.
            d = (
                (outs[("sdpa", "fp8-l1", mode)] - outs[("sdpa", "fp8", mode)])
                .abs()
                .max()
            )
            assert d > 1e-3, f"{fam} {mode}: layer 1's scales changed nothing ({d})"
        print(f"quant_reference.py: kv self-test ok ({fam})", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# Self-test: scalar ports of cpu::quant against the vectorized code above


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def ref_fp8_round(x: float) -> int:
    if math.isnan(x):
        return 0x7F
    sign = 0x80 if _bits(x) >> 31 else 0
    a = abs(x)
    if a >= FP8_MAX:
        return sign | 0x7E
    if a < 2.0**-6:
        return sign | round(a * 512.0)  # exact scaling; Python round() is ties-to-even
    bits = _bits(a)
    exp = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    q = mant >> 20
    rem = mant & 0xFFFFF
    if rem > 0x80000 or (rem == 0x80000 and q & 1):
        q += 1
    if q == 8:
        q = 0
        exp += 1
    return sign | ((exp + 7) << 3) | q


def ref_fp8_value(b: int) -> float:
    sign = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 15 and m == 7:
        return float("nan")
    return sign * (m / 8.0 * 2.0**-6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))


def ref_e2m1_round(x: float) -> int:
    if math.isnan(x):
        return 0
    sign = 8 if _bits(x) >> 31 else 0
    a = min(abs(x), E2M1_MAX)
    best = 0
    for c in range(1, 8):
        d = abs(dq.E2M1_VALUES[c] - a)
        bd = abs(dq.E2M1_VALUES[best] - a)
        if d < bd or (d == bd and c % 2 == 0):
            best = c
    return 0 if best == 0 else sign | best


def ref_mxfp4_scale_even(amax: float) -> int:
    if math.isnan(amax):
        return 255
    rounded_bits = (_bits(amax) + (1 << 21)) & 0xFF800000
    exp = (rounded_bits >> 23) & 0xFF
    if exp == 255:
        log2 = 32767
    elif exp == 0:
        log2 = -127
    else:
        log2 = exp - 127
    return max(-127, min(127, log2 - 2)) + 127


def self_test() -> None:
    import numpy as np
    import torch

    rng = np.random.default_rng(6)
    bf16 = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    rand = (
        rng.integers(0, 2**32, size=200_000, dtype=np.uint64)
        .astype(np.uint32)
        .view(np.float32)
    )
    edge = np.array(
        [
            0.0,
            -0.0,
            1.0,
            448.0,
            500.0,
            np.inf,
            -1e9,
            2.0**-9,
            2.0**-10,
            3 * 2.0**-10,
            1.0625,
            1.1875,
        ]
        + [
            2.0**-6,
            15.5 * 2.0**-10,
            447.0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            7.0,
            np.nan,
        ],
        dtype=np.float32,
    )
    xs = np.concatenate([bf16, rand, edge, -edge])

    # FP8 e4m3: torch's cast (after the ±448 clamp) against the scalar port, scale 1 and 0.37.
    for scale in (1.0, _f32(0.37)):
        got = fp8_qdq(
            torch, torch.from_numpy(xs.copy()), torch.tensor(scale, dtype=torch.float32)
        ).numpy()
        want = np.array(
            [
                ref_fp8_value(ref_fp8_round(float(np.float32(x) / np.float32(scale))))
                for x in xs
            ],
            dtype=np.float32,
        ) * np.float32(scale)
        bad = ~(
            (got.view(np.uint32) == want.view(np.uint32))
            | (np.isnan(got) & np.isnan(want))
        )
        assert not bad.any(), (
            f"fp8 qdq (scale {scale}) differs at {xs[bad][:5]}: {got[bad][:5]} vs {want[bad][:5]}"
        )
    table = [
        (1.0, 0x38),
        (1.0625, 0x38),
        (1.1875, 0x3A),
        (2.0**-10, 0x00),
        (447.0, 0x7E),
        (-1e9, 0xFE),
    ]
    for x, code in table:
        assert ref_fp8_round(x) == code, (x, code)

    # E2M1 round-to-nearest-even over the unit grid.
    fin = xs[np.isfinite(xs) & (np.abs(xs) < 16)]
    got = e2m1_qdq_unit(torch, torch.from_numpy(fin.copy())).numpy()
    want = np.array(
        [dq.E2M1_VALUES[ref_e2m1_round(float(x)) & 7] for x in fin], dtype=np.float32
    )
    want = np.where(
        np.array([ref_e2m1_round(float(x)) & 8 for x in fin]) != 0, -want, want
    )
    bad = got.view(np.uint32) != want.view(np.uint32)
    assert not bad.any(), (
        f"e2m1 differs at {fin[bad][:5]}: {got[bad][:5]} vs {want[bad][:5]}"
    )

    # E8M0 `even` scale over every non-negative BF16 and random F32 amax.
    amax = np.abs(xs)
    got = mxfp4_scale_even(torch, torch.from_numpy(amax.copy())).numpy()
    want = np.array([ref_mxfp4_scale_even(float(a)) for a in amax])
    bad = got != want
    assert not bad.any(), (
        f"mxfp4 scale differs at {amax[bad][:5]}: {got[bad][:5]} vs {want[bad][:5]}"
    )

    # Per-row / per-group dynamic scales: vectorized rows against a scalar loop.
    rows = rng.standard_normal((8, 256)).astype(np.float32) * np.float32(3.0)
    rows[3] = 0.0
    for mode in ("fp8_token", "fp8_group128", "mxfp4"):
        got = qdq_activations(torch, torch.from_numpy(rows.copy()), mode).numpy()
        group = {"fp8_token": 256, "fp8_group128": 128, "mxfp4": 32}[mode]
        want = np.empty_like(rows)
        for r in range(rows.shape[0]):
            for c0 in range(0, rows.shape[1], group):
                grp = rows[r, c0 : c0 + group]
                am = float(np.max(np.abs(grp)))
                if mode == "mxfp4":
                    e = ref_mxfp4_scale_even(am)
                    s = np.float32(2.0 ** (e - 127))
                    for i, v in enumerate(grp):
                        code = ref_e2m1_round(float(np.float32(v) / s))
                        val = dq.E2M1_VALUES[code & 7] * (-1 if code & 8 else 1)
                        want[r, c0 + i] = np.float32(val) * s
                else:
                    s = max(
                        np.float32(am) / np.float32(FP8_MAX),
                        np.float32(1.0) / np.float32(FP8_MAX * 512.0),
                    )
                    for i, v in enumerate(grp):
                        want[r, c0 + i] = (
                            np.float32(
                                ref_fp8_value(ref_fp8_round(float(np.float32(v) / s)))
                            )
                            * s
                        )
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (
            f"{mode} rows differ"
        )
    print(f"quant_reference.py: self-test ok ({len(xs)} inputs)", file=sys.stderr)
    kv_self_test()


# ---------------------------------------------------------------------------------------------
# Reference generation


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is false"
        )
    prompts = hf_reference.load_prompts(args.prompts)
    work = None
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    unquantized = args.kv_quant != "none" and "quantization_config" not in config
    if unquantized and args.act_quant not in ("auto", "none"):
        raise ValueError("a BF16 checkpoint has no activation quantization")
    if unquantized:
        bf16_dir = args.model_dir
    elif args.dequantized is not None:
        bf16_dir = args.dequantized
    else:
        work = Path(tempfile.mkdtemp(prefix="turbine-dequant-", dir=args.work_dir))
        bf16_dir = work / "bf16"
        dq.dequantize(args.model_dir, bf16_dir, log=lambda m: print(m, file=sys.stderr))
    try:
        record = (
            {"activation": "none", "packaging": "bf16", "layers": {}}
            if unquantized
            else json.loads(
                (bf16_dir / "turbine_dequant.json").read_text(encoding="utf-8")
            )
        )
        mode = record["activation"] if args.act_quant == "auto" else args.act_quant
        if mode != record["activation"]:
            print(
                f"warning: --act-quant {mode} differs from the checkpoint's {record['activation']}",
                file=sys.stderr,
            )
        tokenizer = AutoTokenizer.from_pretrained(bf16_dir)
        if args.chat_template is not None:
            tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
        model = AutoModelForCausalLM.from_pretrained(bf16_dir, dtype=torch.bfloat16)
        model.to(args.device)
        model.eval()
        hooks = install_hooks(
            torch, model, record["layers"], mode, args.input_scale == "fused-max"
        )
        if args.kv_quant != "none":
            scales = kv_scales(args.model_dir, model.config.num_hidden_layers)
            attn = install_kv_quant(torch, model, scales)
            print(
                f"{attn}: FP8 KV with {len(scales)} layer scale pairs "
                f"({'checkpoint' if any(p != (1.0, 1.0) for p in scales) else 'all 1.0'})",
                file=sys.stderr,
            )
        source = "" if unquantized else f"-dequant-{record['packaging']}"
        engine = (
            f"transformers-{transformers.__version__}-bf16-{args.device}"
            f"{source}-act-{mode}"
            f"{'-per-part-scales' if args.input_scale == 'per-part' else ''}"
            f"{'-kv-' + args.kv_quant if args.kv_quant != 'none' else ''}-fp32-logits"
        )
        name = args.model_name or hf_reference.model_name(args.model_dir)
        revision = hf_reference.model_revision(args.model_dir) or record.get("revision")
        print(
            f"engine {engine}, model {name}, revision {revision or 'unknown'}, {hooks} activation hooks",
            file=sys.stderr,
        )
        write_reference(torch, args, prompts, tokenizer, model, engine, name, revision)
    finally:
        if work is not None:
            if args.keep_dequantized:
                print(f"dequantized copy kept in {bf16_dir}", file=sys.stderr)
            else:
                shutil.rmtree(work, ignore_errors=True)


def write_reference(
    torch, args, prompts, tokenizer, model, engine, name, revision
) -> None:
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for rec in prompts:
                ids = hf_reference.prompt_ids(tokenizer, rec)
                tokens, tops = hf_reference.generate(
                    torch,
                    model,
                    ids,
                    rec["max_tokens"],
                    args.top_logprobs,
                    args.device,
                    True,
                )
                captured = (
                    datetime.datetime.now(datetime.UTC)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
                line = {"id": rec["id"], "engine": engine, "model": name}
                if revision is not None:
                    line["revision"] = revision
                line |= {
                    "captured": captured,
                    "prompt_token_ids": ids,
                    "tokens": tokens,
                    "top_logprobs": tops,
                }
                f.write(
                    json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                print(
                    f"{rec['id']}: {len(ids)} prompt tokens, {len(tokens)} generated",
                    file=sys.stderr,
                )
        os.replace(tmp, args.out)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
        else:
            run(args)
    except Exception as e:  # noqa: BLE001 -- any failure: exit 1, no partial file
        print(f"quant_reference.py: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
