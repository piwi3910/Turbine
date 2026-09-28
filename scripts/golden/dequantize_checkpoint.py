# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch==2.9.0",
#     "safetensors==0.6.2",
#     "numpy==2.3.4",
#     "packaging==25.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Dequantize a quantized checkpoint into a BF16 copy (Phase 6a S-11, fixture time only).

Every quantized linear layer is decoded exactly as Turbine's CPU reference does
(`crates/turbine-kernels/src/cpu/quant.rs`, layouts in `crates/turbine-kernels/src/quant.rs`):
the value is computed in F32 and then rounded to BF16 (round to nearest, ties to even), the
dtype the transformers reference runs in. MXFP4 values are exact in BF16; FP8 and INT4 values
times a scale may lose low mantissa bits in that last rounding.

Packagings (detected from `config.json` `quantization_config` and the tensors present):
- compressed-tensors `float-quantized` FP8 (`ct_fp8`): `weight` F8_E4M3 × `weight_scale`
  (`[1]` / `[]` tensor, `[n, 1]` channel, `[ceil(n/bn), ceil(k/bk)]` block);
- `quant_method: fp8` (`hf_fp8`): `weight` F8_E4M3 × `weight_scale_inv` (block) or `weight_scale`;
- AutoAWQ GEMM (`awq`): `qweight` I32 `[k, n/8]`, `qzeros` I32 `[k/g, n/8]` (nibble order
  `[0, 2, 4, 6, 1, 3, 5, 7]`), `scales` `[k/g, n]`; value `(q − z) × s`;
- AutoGPTQ (`gptq`): `qweight` I32 `[k/8, n]` packed along k, `qzeros` `[k/g, n/8]` holding
  `zero − 1` (the v1 checkpoint format; `checkpoint_format: gptq_v2` stores the zero itself),
  `scales` `[k/g, n]`, `g_idx` that must be `i // g`; symmetric checkpoints must decode to zero 8
  (Turbine's `Int4GroupSym`);
- compressed-tensors `pack-quantized` INT4 (`ct_pack_int4`): `weight_packed` I32 `[n, k/8]`
  (nibble i = column 8c + i, offset binary: value `(u − 8) × s`), `weight_scale` `[n, k/g]`,
  `weight_shape`; symmetric only;
- compressed-tensors `mxfp4-pack-quantized` (`ct_mxfp4`) and AMD Quark `fp4` (`quark_mxfp4`,
  `pack_method: reorder` does not reorder fp4 in Quark's `Pack_fp4`): `weight_packed` / `weight`
  U8 `[n, k/2]` E2M1 codes, low nibble = even column, `weight_scale` U8 `[n, k/32]` E8M0;
  value `e2m1(q) × 2^(e − 127)`;
- OpenAI native `quant_method: mxfp4` (`openai_mxfp4`): `weight_blocks` U8 `[n, k/32, 16]` (the
  same E2M1 bytes, low nibble first) and `weight_scales` U8 `[n, k/32]` E8M0.

Every other floating tensor is copied as BF16 (F16/F32 checkpoints are rounded to BF16, as the
BF16 transformers load does). `config.json` loses `quantization_config` (and a legacy
`compression_config`) and gets `torch_dtype: bfloat16`; tokenizer, generation and template files
are copied. `turbine_dequant.json` beside the weights records the source, the packaging, the
checkpoint's activation scheme and every decoded layer with its scheme and static `input_scale`
(read by `quant_reference.py` for the activation fake-quantization hooks).

Output is written to `<out>.tmp` and renamed over `<out>` only when everything succeeded; any
failure exits 1 and leaves no output directory.

Usage:
    uv run scripts/golden/dequantize_checkpoint.py --model-dir <quantized-dir> --out <bf16-dir>
        [--shard-bytes 4000000000] [--compare-with <bf16-original-dir>]
    uv run scripts/golden/dequantize_checkpoint.py --check-tiny <dir>

`--compare-with` prints, per decoded layer, the relative RMS error of the dequantized weight
against the same tensor of an unquantized checkpoint (a wrong packing order gives ~1 or more;
real quantization error is a few percent) and exits 1 if any layer exceeds `--max-rel-rms`.
Not meaningful for AWQ: it folds its activation scales into the preceding norm or projection, so
single layers (and norms) legitimately differ from the unquantized checkpoint.

`--check-tiny <dir>` checks this decode against Turbine's Rust one: every `<dir>/<case>/` (or
`<dir>` itself) holding `quantized/` and `twin/`, as written by
`cargo run -p turbine-model --example dump_dequant -- <dir>` (`write_tiny_quantized`: the twin is
the exact BF16 dequantization in the same tensor names), is dequantized into a temporary
directory and must equal `twin/` bit for bit, tensor by tensor (names, dtypes, shapes, bytes).
Exits 1 naming the first differing tensor of every failing case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ACT_NONE = "none"
ACT_FP8_TOKEN = "fp8_token"
ACT_FP8_TENSOR = "fp8_tensor"
ACT_FP8_GROUP128 = "fp8_group128"
ACT_MXFP4 = "mxfp4"

AWQ_REVERSE_ORDER = [
    0,
    4,
    1,
    5,
    2,
    6,
    3,
    7,
]  # column j of 8 sits in nibble AWQ_REVERSE_ORDER[j]
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
AUX_SUFFIXES = (
    "weight_scale",
    "weight_scale_inv",
    "input_scale",
    "qzeros",
    "scales",
    "g_idx",
    "weight_shape",
    "weight_zero_point",
    "weight_g_idx",
)
COPY_SKIP = {"config.json", "model.safetensors.index.json", "turbine_dequant.json"}


class DequantError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors are failures too: exit 1
        self.print_usage(sys.stderr)
        print(f"dequantize_checkpoint.py: error: {message}", file=sys.stderr)
        sys.exit(1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = _Parser(
        description="Write a BF16 copy of a quantized checkpoint (Turbine CPU decode)."
    )
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--check-tiny", type=Path, help="compare with Rust tiny fixtures (see above)"
    )
    p.add_argument("--shard-bytes", type=int, default=4_000_000_000)
    p.add_argument(
        "--compare-with", type=Path, help="unquantized checkpoint to measure against"
    )
    p.add_argument("--max-rel-rms", type=float, default=0.5)
    args = p.parse_args(argv)
    if args.check_tiny is None and (args.model_dir is None or args.out is None):
        p.error("--model-dir and --out are required (or --check-tiny <dir>)")
    return args


# ---------------------------------------------------------------------------------------------
# Packaging detection


@dataclass
class Packaging:
    name: str  # ct_fp8 | hf_fp8 | awq | gptq | ct_pack_int4 | ct_mxfp4 | quark_mxfp4
    activation: str  # ACT_*
    group: int | None = None  # INT4 group size
    block: tuple[int, int] | None = None  # FP8 block (n, k)
    gptq_zero_offset: int = 1
    symmetric: bool = True
    notes: list[str] = field(default_factory=list)


def _ct_activation(inp: dict | None) -> str:
    if inp is None:
        return ACT_NONE
    if inp.get("type") != "float" or inp.get("num_bits") != 8:
        raise DequantError(f"compressed-tensors input_activations not FP8: {inp}")
    strategy, dynamic = inp.get("strategy"), inp.get("dynamic")
    if strategy == "token" and dynamic:
        return ACT_FP8_TOKEN
    if strategy == "tensor" and not dynamic:
        return ACT_FP8_TENSOR
    if strategy == "group" and dynamic and inp.get("group_size") == 128:
        return ACT_FP8_GROUP128
    raise DequantError(f"compressed-tensors input_activations unsupported: {inp}")


def detect(config: dict) -> Packaging:
    qc = config.get("quantization_config") or config.get("compression_config")
    if not qc:
        raise DequantError(
            "config.json has no quantization_config: not a quantized checkpoint"
        )
    method = qc.get("quant_method")
    if method == "compressed-tensors":
        groups = list((qc.get("config_groups") or {}).values())
        if len(groups) != 1:
            raise DequantError(
                f"compressed-tensors: {len(groups)} config groups, want 1"
            )
        g = groups[0]
        fmt = g.get("format") or qc.get("format")
        w = g.get("weights") or {}
        act = _ct_activation(g.get("input_activations"))
        if (
            fmt == "float-quantized"
            and w.get("num_bits") == 8
            and w.get("type") == "float"
        ):
            strategy = w.get("strategy")
            if strategy == "block":
                bs = w.get("block_structure")
                if not bs or len(bs) != 2:
                    raise DequantError(
                        f"compressed-tensors block without block_structure: {w}"
                    )
                return Packaging("ct_fp8", act, block=(int(bs[0]), int(bs[1])))
            if strategy in ("tensor", "channel"):
                return Packaging("ct_fp8", act)
            raise DequantError(f"compressed-tensors FP8 weight strategy {strategy!r}")
        if (
            fmt == "pack-quantized"
            and w.get("num_bits") == 4
            and w.get("type") == "int"
        ):
            if act != ACT_NONE:
                raise DequantError("compressed-tensors INT4 with input activations")
            if not w.get("symmetric", True):
                raise DequantError(
                    "compressed-tensors INT4 asymmetric (zero points) unsupported"
                )
            if w.get("strategy") != "group":
                raise DequantError(
                    f"compressed-tensors INT4 strategy {w.get('strategy')!r}"
                )
            return Packaging("ct_pack_int4", act, group=int(w["group_size"]))
        if fmt == "mxfp4-pack-quantized":
            if (
                w.get("num_bits") != 4
                or w.get("type") != "float"
                or w.get("group_size") != 32
            ):
                raise DequantError(f"compressed-tensors MXFP4 weights unsupported: {w}")
            if act != ACT_NONE:
                raise DequantError("compressed-tensors MXFP4 with input activations")
            return Packaging("ct_mxfp4", act)
        raise DequantError(
            f"compressed-tensors format {fmt!r} with weights {w} unsupported"
        )
    if method == "awq":
        if qc.get("bits") != 4 or str(qc.get("version", "gemm")).lower() != "gemm":
            raise DequantError(
                f"awq: bits {qc.get('bits')} version {qc.get('version')}"
            )
        if not qc.get("zero_point", True):
            raise DequantError("awq without zero points unsupported")
        return Packaging("awq", ACT_NONE, group=int(qc["group_size"]))
    if method == "gptq":
        if qc.get("bits") != 4:
            raise DequantError(f"gptq bits {qc.get('bits')}")
        if qc.get("desc_act"):
            raise DequantError(
                "gptq desc_act: true (act order) is refused (gptq_act_order)"
            )
        offset = 0 if qc.get("checkpoint_format") == "gptq_v2" else 1
        return Packaging(
            "gptq",
            ACT_NONE,
            group=int(qc["group_size"]),
            gptq_zero_offset=offset,
            symmetric=bool(qc.get("sym", True)),
        )
    if method == "fp8":
        act = {"dynamic": ACT_FP8_TOKEN, "static": ACT_FP8_TENSOR}.get(
            qc.get("activation_scheme", "dynamic")
        )
        if act is None:
            raise DequantError(f"fp8 activation_scheme {qc.get('activation_scheme')!r}")
        bs = qc.get("weight_block_size")
        if bs:
            return Packaging(
                "hf_fp8",
                ACT_FP8_GROUP128 if act == ACT_FP8_TOKEN else act,
                block=(int(bs[0]), int(bs[1])),
            )
        return Packaging("hf_fp8", act)
    if method == "quark":
        gq = qc.get("global_quant_config") or {}
        w = gq.get("weight") or {}
        if (qc.get("layer_quant_config") or {}) or (
            qc.get("layer_type_quant_config") or {}
        ):
            raise DequantError("quark per-layer quant configs unsupported")
        if (
            w.get("dtype") != "fp4"
            or w.get("group_size") != 32
            or w.get("scale_format") != "e8m0"
        ):
            raise DequantError(f"quark weight spec unsupported (want fp4/32/e8m0): {w}")
        inp = gq.get("input_tensors")
        if inp is None:
            act = ACT_NONE
        elif (
            inp.get("dtype") == "fp4"
            and inp.get("group_size") == 32
            and inp.get("is_dynamic")
            and inp.get("scale_calculation_mode", "even") == "even"
            and inp.get("round_method", "half_even") == "half_even"
        ):
            act = ACT_MXFP4
        else:
            raise DequantError(f"quark input_tensors spec unsupported: {inp}")
        return Packaging("quark_mxfp4", act)
    if method == "mxfp4":
        return Packaging("openai_mxfp4", ACT_NONE)
    raise DequantError(f"quant_method {method!r} unsupported")


# ---------------------------------------------------------------------------------------------
# Decoding (every function returns the F32 value of an [n, k] weight)


def _f32(t):
    import torch

    return t.to(torch.float32)


def fp8_values(torch, w):
    if w.dtype != torch.float8_e4m3fn:
        raise DequantError(f"FP8 weight has dtype {w.dtype}")
    return w.to(torch.float32)  # exact; 0x7f / 0xff are NaN as in fp8_e4m3_value


def expand_block(torch, s, n: int, k: int, bn: int, bk: int):
    rows = (n + bn - 1) // bn
    cols = (k + bk - 1) // bk
    if tuple(s.shape) != (rows, cols):
        raise DequantError(f"block scale shape {tuple(s.shape)}, want {(rows, cols)}")
    return s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]


def decode_fp8(torch, pk: Packaging, w, scale):
    n, k = w.shape
    v = fp8_values(torch, w)
    s = _f32(scale)
    if pk.block is not None:
        return v * expand_block(torch, s, n, k, pk.block[0], pk.block[1])
    if s.numel() == 1:
        return v * s.reshape(())
    if tuple(s.shape) in ((n, 1), (n,)):
        return v * s.reshape(n, 1)
    if pk.block is None:
        raise DequantError(
            f"FP8 scale shape {tuple(s.shape)} for weight {(n, k)} without a block"
        )
    return v * expand_block(torch, s, n, k, pk.block[0], pk.block[1])


def nibbles(torch, packed, per_word: int = 8):
    """int32 words → their 8 nibbles, lowest bits first, as a new last axis."""
    x = packed.to(torch.int64)
    shifts = torch.arange(per_word, dtype=torch.int64) * 4
    return (x.unsqueeze(-1) >> shifts) & 0xF


def int4_value(torch, q, z, s, group: int):
    """(q − z) × s with q [n, k], z and s [n, k/group] (F32 throughout, like cpu::quant)."""
    zf = z.to(torch.float32).repeat_interleave(group, 1)
    sf = s.to(torch.float32).repeat_interleave(group, 1)
    return (q.to(torch.float32) - zf) * sf


def decode_awq(torch, pk: Packaging, qweight, qzeros, scales):
    k, n8 = qweight.shape
    n = n8 * 8
    g = pk.group
    q = nibbles(torch, qweight)[:, :, AWQ_REVERSE_ORDER].reshape(k, n)
    z = nibbles(torch, qzeros)[:, :, AWQ_REVERSE_ORDER].reshape(-1, n)
    if tuple(scales.shape) != (k // g, n) or tuple(z.shape) != (k // g, n):
        raise DequantError(
            f"awq scales {tuple(scales.shape)} / zeros {tuple(z.shape)} for k={k} n={n}"
        )
    return int4_value(torch, q.t(), z.t(), _f32(scales).t(), g)


def decode_gptq(torch, pk: Packaging, qweight, qzeros, scales, g_idx):
    k8, n = qweight.shape
    k = k8 * 8
    g = pk.group
    if g_idx is not None:
        want = torch.arange(k, dtype=torch.int64) // g
        if not torch.equal(g_idx.to(torch.int64), want):
            raise DequantError(
                "gptq g_idx is not i // group_size (act order): refused (gptq_act_order)"
            )
    q = (
        nibbles(torch, qweight).permute(0, 2, 1).reshape(k, n)
    )  # row r8, nibble i → k = 8 r8 + i
    z = nibbles(torch, qzeros).reshape(-1, n) + pk.gptq_zero_offset
    if tuple(scales.shape) != (k // g, n) or tuple(z.shape) != (k // g, n):
        raise DequantError(
            f"gptq scales {tuple(scales.shape)} / zeros {tuple(z.shape)} for k={k} n={n}"
        )
    if pk.symmetric and not bool((z == 8).all()):
        raise DequantError("gptq sym: true but a zero point is not 8")
    return int4_value(torch, q.t(), z.t(), _f32(scales).t(), g)


def decode_ct_int4(torch, pk: Packaging, packed, scale, shape):
    n, k8 = packed.shape
    k = k8 * 8
    if shape is not None and [int(v) for v in shape.tolist()] != [n, k]:
        raise DequantError(f"weight_shape {shape.tolist()} != unpacked {[n, k]}")
    q = nibbles(torch, packed).reshape(n, k)
    g = pk.group
    if tuple(scale.shape) != (n, k // g):
        raise DequantError(f"ct int4 scale {tuple(scale.shape)} for {(n, k)} group {g}")
    return int4_value(torch, q, torch.full((n, k // g), 8), _f32(scale), g)


def e8m0_values(torch, e):
    e32 = e.to(torch.int32)
    bits = torch.where(e32 == 0, torch.full_like(e32, 1 << 22), e32 << 23)
    v = bits.view(torch.float32).clone()  # 2^(e − 127), 2^-127 as the F32 subnormal
    v[e32 == 255] = float("nan")
    return v


def decode_mxfp4(torch, packed, scale):
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise DequantError(
            f"MXFP4 tensors must be U8, got {packed.dtype} / {scale.dtype}"
        )
    n, k2 = packed.shape
    k = k2 * 2
    if tuple(scale.shape) != (n, (k + 31) // 32):
        raise DequantError(f"MXFP4 scale {tuple(scale.shape)} for {(n, k)}")
    p = packed.to(torch.int64)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(n, k)
    table = torch.tensor(E2M1_VALUES + [-v for v in E2M1_VALUES], dtype=torch.float32)
    vals = table[codes]
    return vals * e8m0_values(torch, scale).repeat_interleave(32, 1)[:, :k]


# ---------------------------------------------------------------------------------------------
# Checkpoint I/O


class Tensors:
    """Lazy access to every tensor of a (possibly sharded) safetensors checkpoint."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        files = sorted(model_dir.glob("*.safetensors"))
        if not files:
            raise DequantError(f"{model_dir}: no *.safetensors")
        self.handles = {}
        self.where: dict[str, str] = {}
        for f in files:
            h = safe_open(str(f), framework="pt")
            self.handles[f.name] = h
            for name in h.keys():  # noqa: SIM118 -- safe_open is not a dict
                if name in self.where:
                    raise DequantError(
                        f"tensor {name} in {self.where[name]} and {f.name}"
                    )
                self.where[name] = f.name

    def names(self) -> list[str]:
        return list(self.where)

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str):
        return self.handles[self.where[name]].get_tensor(name)

    def opt(self, name: str):
        return self.get(name) if self.has(name) else None


def model_revision(model_dir: Path) -> str | None:
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        first = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return first if re.fullmatch(r"[0-9a-f]{40}", first) else None


def quantized_prefixes(tensors: Tensors, pk: Packaging) -> list[str]:
    """Layer prefixes holding a quantized weight, in checkpoint order."""
    out = []
    for name in tensors.names():
        if pk.name in ("awq", "gptq"):
            if name.endswith(".qweight"):
                out.append(name[: -len(".qweight")])
        elif pk.name in ("ct_pack_int4", "ct_mxfp4"):
            if name.endswith(".weight_packed"):
                out.append(name[: -len(".weight_packed")])
        elif pk.name == "openai_mxfp4":
            if name.endswith(".weight_blocks"):
                out.append(name[: -len(".weight_blocks")])
        elif name.endswith(".weight"):
            prefix = name[: -len(".weight")]
            has_scale = tensors.has(prefix + ".weight_scale") or tensors.has(
                prefix + ".weight_scale_inv"
            )
            if not has_scale:
                continue
            dtype = tensors.handles[tensors.where[name]].get_slice(name).get_dtype()
            if pk.name in ("ct_fp8", "hf_fp8") and dtype != "F8_E4M3":
                raise DequantError(
                    f"{name}: {dtype} with a weight scale in an FP8 checkpoint"
                )
            if pk.name == "quark_mxfp4" and dtype != "U8":
                raise DequantError(
                    f"{name}: {dtype} with a weight scale in a Quark fp4 checkpoint"
                )
            out.append(prefix)
    return out


def decode_layer(torch, tensors: Tensors, pk: Packaging, prefix: str):
    """→ (F32 [n, k] weight, scheme name, consumed tensor names)."""
    t = tensors.get
    used = [n for n in (f"{prefix}.{s}" for s in AUX_SUFFIXES) if tensors.has(n)]
    if pk.name in ("ct_fp8", "hf_fp8"):
        w = t(prefix + ".weight")
        scale = tensors.opt(prefix + ".weight_scale")
        if scale is None:
            scale = t(prefix + ".weight_scale_inv")
        value = decode_fp8(torch, pk, w, scale)
        if pk.block is not None:
            scheme = "fp8_block"
        elif scale.numel() == 1:
            scheme = "fp8_tensor"
        elif tuple(scale.shape) in ((w.shape[0], 1), (w.shape[0],)):
            scheme = "fp8_channel"
        else:
            scheme = "fp8_block"
        used.append(prefix + ".weight")
    elif pk.name == "awq":
        value = decode_awq(
            torch,
            pk,
            t(prefix + ".qweight"),
            t(prefix + ".qzeros"),
            t(prefix + ".scales"),
        )
        scheme = "int4_group_zp"
        used.append(prefix + ".qweight")
    elif pk.name == "gptq":
        value = decode_gptq(
            torch,
            pk,
            t(prefix + ".qweight"),
            t(prefix + ".qzeros"),
            t(prefix + ".scales"),
            tensors.opt(prefix + ".g_idx"),
        )
        scheme = "int4_group_sym" if pk.symmetric else "int4_group_zp"
        used.append(prefix + ".qweight")
    elif pk.name == "ct_pack_int4":
        if tensors.has(prefix + ".weight_zero_point"):
            raise DequantError(f"{prefix}: weight_zero_point (asymmetric) unsupported")
        gi = tensors.opt(prefix + ".weight_g_idx")
        if gi is not None:
            k = t(prefix + ".weight_packed").shape[1] * 8
            if not torch.equal(
                gi.to(torch.int64), torch.arange(k, dtype=torch.int64) // pk.group
            ):
                raise DequantError(
                    f"{prefix}: non-trivial weight_g_idx (act order) refused"
                )
        value = decode_ct_int4(
            torch,
            pk,
            t(prefix + ".weight_packed"),
            t(prefix + ".weight_scale"),
            tensors.opt(prefix + ".weight_shape"),
        )
        scheme = "int4_group_sym"
        used.append(prefix + ".weight_packed")
    elif pk.name == "ct_mxfp4":
        value = decode_mxfp4(
            torch, t(prefix + ".weight_packed"), t(prefix + ".weight_scale")
        )
        scheme = "mxfp4"
        used.append(prefix + ".weight_packed")
    elif pk.name == "quark_mxfp4":
        value = decode_mxfp4(torch, t(prefix + ".weight"), t(prefix + ".weight_scale"))
        scheme = "mxfp4"
        used.append(prefix + ".weight")
    elif pk.name == "openai_mxfp4":
        blocks = t(prefix + ".weight_blocks")
        if blocks.dim() != 3 or blocks.shape[2] != 16:
            raise DequantError(f"{prefix}.weight_blocks shape {tuple(blocks.shape)}")
        packed = blocks.reshape(blocks.shape[0], -1)
        value = decode_mxfp4(torch, packed, t(prefix + ".weight_scales"))
        scheme = "mxfp4"
        used += [prefix + ".weight_blocks", prefix + ".weight_scales"]
    else:
        raise DequantError(f"packaging {pk.name}")
    return value, scheme, used


class ShardWriter:
    def __init__(self, out: Path, shard_bytes: int):
        self.out = out
        self.shard_bytes = shard_bytes
        self.pending: dict = {}
        self.pending_bytes = 0
        self.shards: list[tuple[str, list[str]]] = []
        self.total = 0

    def add(self, name: str, tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + nbytes > self.shard_bytes:
            self.flush()
        self.pending[name] = tensor.contiguous()
        self.pending_bytes += nbytes
        self.total += nbytes

    def flush(self) -> None:
        from safetensors.torch import save_file

        if not self.pending:
            return
        tmp = f"shard-{len(self.shards):05d}.safetensors"
        save_file(self.pending, str(self.out / tmp), metadata={"format": "pt"})
        self.shards.append((tmp, list(self.pending)))
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> None:
        self.flush()
        if len(self.shards) == 1:
            os.replace(self.out / self.shards[0][0], self.out / "model.safetensors")
            return
        count = len(self.shards)
        weight_map = {}
        for i, (tmp, names) in enumerate(self.shards):
            final = f"model-{i + 1:05d}-of-{count:05d}.safetensors"
            os.replace(self.out / tmp, self.out / final)
            for n in names:
                weight_map[n] = final
        index = {
            "metadata": {"total_size": self.total},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )


def dequantize(
    model_dir: Path, out: Path, shard_bytes: int = 4_000_000_000, log=print
) -> dict:
    """Writes the BF16 copy of `model_dir` into `out` (created; must not exist) and returns the
    `turbine_dequant.json` record."""
    import torch

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    pk = detect(config)
    tensors = Tensors(model_dir)
    prefixes = quantized_prefixes(tensors, pk)
    if not prefixes:
        raise DequantError(
            f"{model_dir}: no quantized linear layer found for {pk.name}"
        )
    out.mkdir(parents=True)
    writer = ShardWriter(out, shard_bytes)
    consumed: set[str] = set()
    layers = {}
    with torch.inference_mode():
        prefix_set = set(prefixes)
        for name in tensors.names():
            prefix = name.rsplit(".", 1)[0]
            if prefix in prefix_set and prefix not in layers:
                value, scheme, used = decode_layer(torch, tensors, pk, prefix)
                if not torch.isfinite(value).all():
                    raise DequantError(f"{prefix}: non-finite dequantized value")
                writer.add(prefix + ".weight", value.to(torch.bfloat16))
                consumed.update(used)
                entry = {"scheme": scheme, "shape": list(value.shape)}
                if tensors.has(prefix + ".input_scale"):
                    entry["input_scale"] = float(
                        _f32(tensors.get(prefix + ".input_scale")).reshape(-1)[0]
                    )
                layers[prefix] = entry
        for name in tensors.names():
            if name in consumed:
                continue
            if name.rsplit(".", 1)[0] in prefix_set:
                raise DequantError(f"{name}: unexpected tensor of a quantized layer")
            if name.endswith((".k_scale", ".v_scale")):
                continue  # KV-cache scales: not weights (FP8 KV reads them from the checkpoint)
            t = tensors.get(name)
            if not t.is_floating_point() or t.dtype == torch.float8_e4m3fn:
                raise DequantError(
                    f"{name}: {t.dtype} tensor outside a quantized layer"
                )
            writer.add(name, t.to(torch.bfloat16))
    writer.finish()

    config.pop("quantization_config", None)
    config.pop("compression_config", None)
    config["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for f in sorted(model_dir.iterdir()):
        if (
            f.is_file()
            and f.name not in COPY_SKIP
            and not f.name.endswith(".safetensors")
        ):
            shutil.copy2(f, out / f.name)
    acts = [e.get("input_scale") for e in layers.values()]
    if pk.activation == ACT_FP8_TENSOR and any(a is None for a in acts):
        raise DequantError("static FP8 activations but a layer has no input_scale")
    record = {
        "source": str(model_dir.resolve()),
        "revision": model_revision(model_dir),
        "name_or_path": config.get("_name_or_path"),
        "packaging": pk.name,
        "activation": pk.activation,
        "layers": layers,
    }
    (out / "turbine_dequant.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    log(
        f"{pk.name}: {len(layers)} layers decoded, activation {pk.activation}, {writer.total} bytes BF16"
    )
    return record


def compare(out: Path, reference: Path, max_rel_rms: float, log=print) -> bool:
    import torch

    record = json.loads((out / "turbine_dequant.json").read_text(encoding="utf-8"))
    ours, ref = Tensors(out), Tensors(reference)
    worst = 0.0
    ok = True
    for prefix in record["layers"]:
        name = prefix + ".weight"
        if not ref.has(name):
            log(f"{name}: not in {reference}")
            ok = False
            continue
        a = ours.get(name).to(torch.float32)
        b = ref.get(name).to(torch.float32)
        if a.shape != b.shape:
            log(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        rel = float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())
        worst = max(worst, rel)
        if rel > max_rel_rms:
            ok = False
            log(f"{name}: relative RMS error {rel:.4f} > {max_rel_rms}")
    log(
        f"compare: {len(record['layers'])} layers, worst relative RMS error {worst:.4f}"
    )
    return ok


def tensors_equal(ours: Path, twin: Path) -> str | None:
    """None when both checkpoints hold the same tensors bit for bit, else the first difference."""
    import torch

    a, b = Tensors(ours), Tensors(twin)
    if set(a.names()) != set(b.names()):
        only_a = sorted(set(a.names()) - set(b.names()))[:3]
        only_b = sorted(set(b.names()) - set(a.names()))[:3]
        return f"tensor names differ: only decoded {only_a}, only twin {only_b}"
    for name in sorted(b.names()):
        x, y = a.get(name), b.get(name)
        if x.dtype != y.dtype or x.shape != y.shape:
            return (
                f"{name}: {x.dtype}{tuple(x.shape)} vs twin {y.dtype}{tuple(y.shape)}"
            )
        xb = x.contiguous().reshape(-1).view(torch.uint8).reshape(x.numel(), -1)
        yb = y.contiguous().reshape(-1).view(torch.uint8).reshape(y.numel(), -1)
        if not torch.equal(xb, yb):
            diff = (x.to(torch.float32) - y.to(torch.float32)).abs()
            count = int((xb != yb).any(-1).sum())
            return (
                f"{name}: {count} of {x.numel()} values differ,"
                f" max |diff| {float(diff.max()):.3g}"
            )
    return None


def check_tiny(root: Path, log=print) -> bool:
    import tempfile

    if (root / "quantized").is_dir():
        cases = [root]
    else:
        cases = sorted(d for d in root.iterdir() if (d / "quantized").is_dir())
    if not cases:
        raise DequantError(f"{root}: no <case>/quantized directory")
    ok = True
    for case in cases:
        if not (case / "twin").is_dir():
            raise DequantError(f"{case}: quantized/ without twin/")
        with tempfile.TemporaryDirectory(prefix="turbine-check-tiny-") as tmp:
            out = Path(tmp) / "bf16"
            record = dequantize(case / "quantized", out, log=lambda _m: None)
            problem = tensors_equal(out, case / "twin")
        schemes = "/".join(sorted({e["scheme"] for e in record["layers"].values()}))
        label = (
            f"{case.name}: {record['packaging']} {schemes} act {record['activation']}"
        )
        if problem is None:
            log(f"{label}: {len(record['layers'])} layers bit-exact")
        else:
            ok = False
            log(f"{label}: MISMATCH {problem}")
    log(f"check-tiny: {len(cases)} case(s) {'ok' if ok else 'FAIL'}")
    return ok


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.check_tiny is not None:
        try:
            ok = check_tiny(args.check_tiny, log=lambda m: print(m, file=sys.stderr))
        except Exception as e:  # noqa: BLE001 -- any failure: exit 1
            print(
                f"dequantize_checkpoint.py: error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 1
        return 0 if ok else 1
    tmp = args.out.with_name(args.out.name + ".tmp")
    try:
        if args.out.exists():
            raise DequantError(f"{args.out} exists")
        if tmp.exists():
            shutil.rmtree(tmp)
        dequantize(
            args.model_dir,
            tmp,
            args.shard_bytes,
            log=lambda m: print(m, file=sys.stderr),
        )
        os.replace(tmp, args.out)
        if args.compare_with is not None and not compare(
            args.out,
            args.compare_with,
            args.max_rel_rms,
            log=lambda m: print(m, file=sys.stderr),
        ):
            return 1
    except Exception as e:  # noqa: BLE001 -- any failure: exit 1, no partial output
        print(
            f"dequantize_checkpoint.py: error: {type(e).__name__}: {e}", file=sys.stderr
        )
        if tmp.exists():
            shutil.rmtree(tmp)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
