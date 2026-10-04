# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Independent GPTQ dequantization check (Phase 6a GPTQ numerics investigation).

Dequantizes INT4 linear layers of an AutoGPTQ checkpoint with a transcription of AutoGPTQ's own
`qlinear_cuda_old` dequant (shift-unpack of `qweight` along k and `qzeros` along n, `zeros + 1`
then `& 0xF` for `checkpoint_format: gptq` v1, `scales[g_idx]` / `zeros[g_idx]` so act order and
any `g_idx` are honoured) and compares the result bit for bit against Turbine's CPU dequant of the
same layer (`cargo run --release -p turbine-model --example int4_layer_dump`, an `[n, k]` F32
little-endian file). It also reports, per layer, the relative Frobenius error of the GPTQ weight
and of a round-to-nearest quantization of the same scheme against the BF16 original, to tell a
faithful load of a noisier checkpoint from a load bug. (AWQ checkpoints are not compared: their
weights carry the activation-aware input scales, so they differ from the original by design.)

Fixture/diagnostic tooling only, never the serving path. numpy only: safetensors headers are
parsed here, BF16 is widened by hand.

usage: uv run scripts/golden/gptq_dequant_check.py --gptq-dir <dir> --layer <module>
         [--turbine <dump.f32>] [--bf16-dir <dir>] [--json]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


class Checkpoint:
    def __init__(self, directory):
        self.dir = directory
        index = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.tensors = {}
        for name in files:
            path = os.path.join(directory, name)
            with open(path, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                self.tensors[key] = (path, 8 + hlen, meta)
        with open(os.path.join(directory, "config.json")) as f:
            self.config = json.load(f)

    def has(self, key):
        return key in self.tensors

    def get(self, key):
        path, base, meta = self.tensors[key]
        start, end = meta["data_offsets"]
        dtype = meta["dtype"]
        with open(path, "rb") as f:
            f.seek(base + start)
            raw = f.read(end - start)
        shape = meta["shape"]
        if dtype == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        np_dtype = {"F16": "<f2", "F32": "<f4", "I32": "<i4", "I64": "<i8"}[dtype]
        return np.frombuffer(raw, dtype=np_dtype).reshape(shape)


def gptq_dequant(ck, layer, bits=4):
    """AutoGPTQ qlinear_cuda_old.forward's dequant, transcribed to numpy. Returns ([n, k] f32
    exact product, [n, k] f16 product as AutoGPTQ computes it, facts)."""
    q = ck.config["quantization_config"]
    qweight = ck.get(f"{layer}.qweight").astype(np.int64) & 0xFFFFFFFF  # [k/8, n]
    qzeros = ck.get(f"{layer}.qzeros").astype(np.int64) & 0xFFFFFFFF  # [groups, n/8]
    scales = ck.get(f"{layer}.scales")  # [groups, n] F16
    g_idx = ck.get(f"{layer}.g_idx").astype(np.int64)  # [k]
    wf = np.arange(0, 32, bits, dtype=np.int64)
    # zeros: unsqueeze(2).expand(..., per) >> wf, to int8, + 1, & 0xF (v1); v2 skips the + 1.
    zeros = (qzeros[:, :, None] >> wf[None, None, :]).astype(np.int8).astype(np.int64)
    fmt = q.get("checkpoint_format", "gptq")
    raw_zeros = zeros & ((1 << bits) - 1)
    if fmt != "gptq_v2":
        zeros = zeros + 1
    zeros = (zeros & ((1 << bits) - 1)).reshape(zeros.shape[0], -1)  # [groups, n]
    # weight: unsqueeze(1).expand(-1, per, -1) >> wf[:, None], to int8, & 0xF -> [k, n]
    weight = (qweight[:, None, :] >> wf[None, :, None]).astype(np.int8).astype(np.int64)
    weight = (weight & ((1 << bits) - 1)).reshape(-1, weight.shape[2])
    k, n = weight.shape
    s = scales[g_idx]  # [k, n]
    z = zeros[g_idx]
    exact = (s.astype(np.float32) * (weight - z).astype(np.float32)).T.copy()
    half = (s * (weight - z).astype(np.float16)).astype(np.float32).T.copy()
    group = int(q.get("group_size", -1))
    facts = {
        "n": n,
        "k": k,
        "group_size": group,
        "groups": int(scales.shape[0]),
        "checkpoint_format": fmt,
        "sym": q.get("sym"),
        "desc_act": q.get("desc_act"),
        "g_idx_identity": bool(np.array_equal(g_idx, np.arange(k) // group)),
        "stored_zero_values": sorted(int(v) for v in np.unique(raw_zeros)),
        "zero_values": sorted(int(v) for v in np.unique(zeros)),
        "scales_dtype": str(scales.dtype),
        "code_histogram": np.bincount(weight.ravel(), minlength=16).tolist(),
    }
    return exact, half, facts


def rtn_sym(w, group, bits=4):
    """Round-to-nearest of `w` [n, k] with AutoGPTQ's symmetric quantizer (per row and group of
    `group` inputs: scale = 2 max|x| / maxq, zero (maxq + 1) / 2), dequantized: the error floor a
    GPTQ checkpoint of the same scheme is compared with."""
    maxq = (1 << bits) - 1
    n, k = w.shape
    g = w.reshape(n, k // group, group).astype(np.float64)
    xmax = np.maximum(np.abs(np.minimum(g.min(-1), 0)), np.maximum(g.max(-1), 0))
    xmax = np.where(xmax == 0, 1.0, xmax)
    scale = (2 * xmax / maxq)[..., None]
    zero = (maxq + 1) / 2
    q = np.clip(np.round(g / scale) + zero, 0, maxq)
    return (scale * (q - zero)).reshape(n, k)


def rel_err(a, ref):
    d = a.astype(np.float64) - ref.astype(np.float64)
    return float(np.linalg.norm(d) / np.linalg.norm(ref.astype(np.float64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gptq-dir", required=True)
    ap.add_argument(
        "--layer", required=True, help="e.g. model.layers.0.self_attn.q_proj"
    )
    ap.add_argument("--turbine", help="int4_layer_dump output ([n, k] F32 LE)")
    ap.add_argument("--bf16-dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    gptq = Checkpoint(a.gptq_dir)
    exact, half, facts = gptq_dequant(gptq, a.layer)
    n, k = facts["n"], facts["k"]
    out = {"layer": a.layer, **facts}
    ok = True
    if a.turbine:
        t = np.fromfile(a.turbine, dtype="<f4")
        if t.size != n * k:
            print(f"turbine dump has {t.size} values, want {n}x{k}", file=sys.stderr)
            return 2
        t = t.reshape(n, k)
        mism = int(np.count_nonzero(t.view(np.uint32) != exact.view(np.uint32)))
        out["turbine_bit_mismatches"] = mism
        out["turbine_max_abs_diff"] = float(np.max(np.abs(t - exact)))
        out["turbine_vs_f16_product_max_abs_diff"] = float(np.max(np.abs(t - half)))
        if mism:
            idx = np.argwhere(t.view(np.uint32) != exact.view(np.uint32))[:5]
            out["first_mismatches"] = [
                {
                    "n": int(r),
                    "k": int(c),
                    "turbine": float(t[r, c]),
                    "ref": float(exact[r, c]),
                }
                for r, c in idx
            ]
            ok = False
    if a.bf16_dir:
        ref = Checkpoint(a.bf16_dir).get(f"{a.layer}.weight").astype(np.float32)
        out["gptq_rel_err_vs_bf16"] = rel_err(exact, ref)
        out["rtn_sym_rel_err_vs_bf16"] = rel_err(rtn_sym(ref, facts["group_size"]), ref)
    if a.json:
        print(json.dumps(out))
    else:
        for key, v in out.items():
            print(f"{key}: {v}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
