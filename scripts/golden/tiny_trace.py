# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==4.57.1",
#     "safetensors==0.6.2",
# ]
# ///
"""Per-op transformers trace of a tiny Turbine checkpoint (Phase 7 Task 24; fixture time only).

Runs one prefill of `--prompt-ids` through Hugging Face transformers on the CPU and writes the
op outputs Turbine's `DecoderExecutor::set_trace` records, in F32, as JSON:

    {"family", "model_dir", "prompt_ids", "ops": [{"layer", "name", "shape", "data"}, ...]}

`layer` is null for `embed`, `final_norm` and `logits`. Per layer: `attn_norm`, `q`, `k`, `v`
(the biased projections), `attn` (the attention output before the O projection), `o_proj`
(biased), `resid_attn`, `mlp_norm`, `router_logits` (the router's matrix product, the bias not
included: Turbine's router adds it inside `moe_route`), `topk_ids` (the selected experts, in the
router's order), `moe_out`, `resid_mlp`. `final_norm` and `logits` are the last row only; the
logits are F32 from the BF16 final-norm output and the LM head weight (the form of
`hf_reference.py`).

Supported architecture: `GptOssForCausalLM`. The checkpoint's MXFP4 experts are dequantized here
(E2M1 codes times 2^(e - 127), exact in BF16) into transformers' BF16 layout (`gate_up_proj`
`[experts, hidden, 2 * inter]`, the transpose of the stored `[experts, 2 * inter, hidden]`), and
the model is built from the config with its `quantization_config` removed, so transformers'
quantizer (which needs a GPU) is not involved. Attention runs eager (the only implementation
with the sinks).

Usage:
    uv run scripts/golden/tiny_trace.py --model-dir <tiny> --prompt-ids 1,2,3,4,5 --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]  # fmt: skip


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[..., n, k/32, 16]` U8 codes and `[..., n, k/32]` E8M0 exponents -> `[..., n, k]` BF16.

    Low nibble first; value = e2m1(code) * 2^(e - 127), exact in BF16.
    """
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    lo = lut[(blocks & 0x0F).long()]
    hi = lut[(blocks >> 4).long()]
    out = torch.stack([lo, hi], dim=-1).flatten(-2)  # [..., n, k/32, 32]
    out = out * torch.pow(2.0, scales.to(torch.int32) - 127).unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    raw: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for name in st.keys():
                raw[name] = st.get_tensor(name)
    for name, t in raw.items():
        if name.endswith("_blocks"):
            base = name[: -len("_blocks")]
            deq = dequantize_mxfp4(t, raw[base + "_scales"])  # [E, n, k]
            # The stored rows interleave gate and up (row 2j = gate column j, row 2j + 1 =
            # up column j); transformers' gate_up_proj keeps them and slices its output
            # even/odd, so the dequantized tensor loads as is.
            state[base] = deq.transpose(1, 2).contiguous()  # [E, k, n]
        elif name.endswith("_scales"):
            continue
        else:
            state[name] = t
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--prompt-ids", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from transformers import GptOssConfig, GptOssForCausalLM

    cfg_json = json.loads((args.model_dir / "config.json").read_text())
    if cfg_json.get("architectures") != ["GptOssForCausalLM"]:
        print(f"tiny_trace: unsupported architectures {cfg_json.get('architectures')}", file=sys.stderr)
        return 2
    ids = [int(x) for x in args.prompt_ids.split(",")]

    config = GptOssConfig.from_pretrained(args.model_dir)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = GptOssForCausalLM(config).to(torch.bfloat16).eval()
    missing = model.load_state_dict(load_state_dict(args.model_dir), strict=True)
    print(f"tiny_trace: loaded {missing}", file=sys.stderr)

    ops: list[dict] = []

    def rec(layer, name, t: torch.Tensor) -> None:
        t = t.detach().to(torch.float32).reshape(-1, t.shape[-1])
        ops.append({"layer": layer, "name": name, "shape": list(t.shape), "data": t.flatten().tolist()})

    handles = []

    def out_hook(layer, name):
        def hook(_m, _i, out):
            rec(layer, name, out[0] if isinstance(out, tuple) else out)

        return hook

    def in_hook(layer, name):
        def hook(_m, inp):
            rec(layer, name, inp[0])

        return hook

    m = model.model
    handles.append(m.embed_tokens.register_forward_hook(out_hook(None, "embed")))
    for i, layer in enumerate(m.layers):
        a = layer.self_attn
        handles += [
            layer.input_layernorm.register_forward_hook(out_hook(i, "attn_norm")),
            a.q_proj.register_forward_hook(out_hook(i, "q")),
            a.k_proj.register_forward_hook(out_hook(i, "k")),
            a.v_proj.register_forward_hook(out_hook(i, "v")),
            a.o_proj.register_forward_pre_hook(in_hook(i, "attn")),
            a.o_proj.register_forward_hook(out_hook(i, "o_proj")),
            layer.post_attention_layernorm.register_forward_pre_hook(in_hook(i, "resid_attn")),
            layer.post_attention_layernorm.register_forward_hook(out_hook(i, "mlp_norm")),
            layer.register_forward_hook(out_hook(i, "resid_mlp")),
        ]

        def router_pre(i):
            def hook(mod, inp):
                x = inp[0].reshape(-1, mod.weight.shape[1])
                rec(i, "router_logits", x.float() @ mod.weight.float().T)

            return hook

        def router_post(i):
            def hook(_mod, _inp, out):
                rec(i, "topk_ids", out[1].float())

            return hook

        handles += [
            layer.mlp.router.register_forward_pre_hook(router_pre(i)),
            layer.mlp.router.register_forward_hook(router_post(i)),
            layer.mlp.register_forward_hook(out_hook(i, "moe_out")),
        ]
    final = {}

    def norm_hook(_m, _i, out):
        final["norm"] = out[0, -1:, :]

    handles.append(m.norm.register_forward_hook(norm_hook))

    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), use_cache=False)
    for h in handles:
        h.remove()

    # Forward hooks fire in module-call order; Turbine's trace order is per layer in op order.
    order = [
        "attn_norm", "q", "k", "v", "attn", "o_proj", "resid_attn", "mlp_norm",
        "router_logits", "topk_ids", "moe_out", "resid_mlp",
    ]  # fmt: skip
    embed = [o for o in ops if o["name"] == "embed"]
    layers = [
        next(o for o in ops if o["layer"] == i and o["name"] == n)
        for i in range(len(m.layers))
        for n in order
    ]
    last_norm = final["norm"]
    rec(None, "final_norm", last_norm)
    logits = last_norm.float() @ model.lm_head.weight.float().T
    rec(None, "logits", logits)
    tail = ops[-2:]
    out = {
        "family": "gpt_oss",
        "model_dir": args.model_dir.name,
        "prompt_ids": ids,
        "ops": embed + layers + tail,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"tiny_trace: wrote {len(out['ops'])} ops to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
