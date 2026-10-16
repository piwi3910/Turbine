# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==4.57.1",
#     "safetensors==0.6.2",
# ]
# ///
"""Per-op transformers trace of a Turbine checkpoint (Phase 7 Task 24; fixture time only).

Runs one prefill through Hugging Face transformers on the CPU and writes the op outputs Turbine's
`DecoderExecutor::set_trace` records, as JSON:

    {"family", "model_dir", "prompt_ids", "ops": [{"layer", "name", "shape", "data"}, ...]}

`layer` is null for `embed`, `final_norm` and `logits`. Per layer: `attn_norm`, `q`, `k`, `v`
(the biased projections), `attn` (the attention output before the O projection), `o_proj`
(biased), `resid_attn`, `mlp_norm`, `router_logits` (the router's matrix product, the bias not
included: Turbine's router adds it inside `moe_route`), `topk_ids` (the selected experts, in the
router's order), `moe_out`, `resid_mlp`. `final_norm` and `logits` are the last row only; the
logits are F32 from the BF16 final-norm output and the LM head weight (the form of
`hf_reference.py`).

Two modes:

- `--prompt-ids 1,2,3`: a tiny synthetic checkpoint. Supported architecture: `GptOssForCausalLM`.
  The checkpoint's MXFP4 experts are dequantized here (E2M1 codes times 2^(e - 127), exact in
  BF16) into transformers' BF16 layout (`gate_up_proj` `[experts, hidden, 2 * inter]`, the
  transpose of the stored `[experts, 2 * inter, hidden]`), and the model is built from the config
  with its `quantization_config` removed, so transformers' quantizer (which needs a GPU) is not
  involved. Attention runs eager (the only implementation with the sinks).
- `--prompt-file tests/golden/prompts.jsonl --prompt-id p05`: a real checkpoint. Supported
  architecture: `LlamaForCausalLM`. The prompt is tokenized as `hf_reference.py` does (a
  `completion` prompt with `add_special_tokens=True`; a `chat` prompt through the chat template
  with `add_generation_prompt=True`). `--context-tokens N` appends the committed reference's
  first `N` greedy tokens (`tests/golden/<model-dir-name>/reference.jsonl`) so the trace reaches
  the position under study. A quantized checkpoint is dequantized into transformers' BF16 layout
  by `dequantize_checkpoint.py` (its exact CPU decode). Per layer the trace records `attn_norm`,
  `q`, `k`, `v`, `q_rope`, `k_rope` (transformers' own rotary, applied to the recorded q/k),
  `attn`, `o_proj`, `resid_attn`, `mlp_norm`, `gate`, `up`, `act`, `down`, `resid_mlp`. With
  `--row last` (default in this mode) only the prefill's last row of every op is kept, as a
  compact little-endian hex blob (`"dtype"` `bf16` for the BF16 activations, `f32` for the
  logits), so the fixture of a big checkpoint stays small:

      {"family", "model_dir", "prompt_ids", "row", "ops": [{"layer", "name", "shape", "dtype",
      "data_hex"}, ...]}

Usage:
    uv run scripts/golden/tiny_trace.py --model-dir <tiny> --prompt-ids 1,2,3,4,5 --out <json>
    uv run scripts/golden/tiny_trace.py --model-dir <real> --prompt-file <prompts.jsonl>
        --prompt-id <id> [--context-tokens N] [--row last|all] --out <json>
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

LLAMA_OP_ORDER = [
    "attn_norm", "q", "k", "v", "q_rope", "k_rope", "attn", "o_proj", "resid_attn",
    "mlp_norm", "gate", "up", "act", "down", "resid_mlp",
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


def load_state_dict_quantized(model_dir: Path) -> dict[str, torch.Tensor]:
    """The checkpoint's BF16 state dict, every quantized linear decoded by
    `dequantize_checkpoint.py` (the exact CPU decode: value in F32, rounded to BF16)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dequantize_checkpoint as dq

    cfg = json.loads((model_dir / "config.json").read_text())
    tensors = dq.Tensors(model_dir)
    pk = dq.detect(cfg)
    prefixes = set(dq.quantized_prefixes(tensors, pk))
    state: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    with torch.inference_mode():
        for prefix in sorted(prefixes):
            value, _scheme, used = dq.decode_layer(torch, tensors, pk, prefix)
            state[prefix + ".weight"] = value.to(torch.bfloat16)
            consumed.update(used)
        for name in tensors.names():
            if name in consumed:
                continue
            if name.endswith(
                (
                    ".weight_scale",
                    ".weight_scale_inv",
                    "input_scale",
                    ".k_scale",
                    ".v_scale",
                )
            ):
                continue
            if name.rsplit(".", 1)[0] in prefixes:
                raise SystemExit(
                    f"tiny_trace: unexpected tensor of a quantized layer: {name}"
                )
            t = tensors.get(name)
            state[name] = t.to(torch.bfloat16) if t.is_floating_point() else t
    return state


def prompt_ids(model_dir: Path, prompt_file: Path, prompt_id: str, context_tokens: int):
    """The prompt's token ids (as `hf_reference.py` renders them), plus the committed
    reference's first `context_tokens` greedy tokens when asked."""
    from transformers import AutoTokenizer

    rec = None
    for line in prompt_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("id") == prompt_id:
            rec = r
            break
    if rec is None:
        raise SystemExit(f"tiny_trace: no prompt {prompt_id!r} in {prompt_file}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if rec.get("kind") == "chat":
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=True,
            **(rec.get("chat_template_kwargs") or {}),
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tokenizer(rec["prompt"], add_special_tokens=True)["input_ids"]
    if context_tokens:
        root = Path(__file__).resolve().parents[2]
        ref = root / "tests" / "golden" / model_dir.name / "reference.jsonl"
        entry = None
        for line in ref.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("id") == prompt_id:
                entry = r
                break
        if entry is None:
            raise SystemExit(f"tiny_trace: no prompt {prompt_id!r} in {ref}")
        ids = ids + entry["tokens"][:context_tokens]
    return ids


def trace_llama_real(model_dir: Path, ids, out_path: Path, keep_last: bool) -> int:
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    cfg_json = json.loads((model_dir / "config.json").read_text())
    if cfg_json.get("architectures") != ["LlamaForCausalLM"]:
        print(
            f"tiny_trace: unsupported architectures {cfg_json.get('architectures')}",
            file=sys.stderr,
        )
        return 2

    config = LlamaConfig.from_pretrained(model_dir)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    missing = model.load_state_dict(load_state_dict_quantized(model_dir), strict=True)
    print(f"tiny_trace: loaded {missing}", file=sys.stderr)

    recorded: list[dict] = []

    def emit(layer, name, t: torch.Tensor) -> None:
        t = t.detach().reshape(-1, t.shape[-1])
        if keep_last:
            t = t[-1:]
        if name == "logits":
            raw = t.to(torch.float32).contiguous().view(torch.uint8).numpy().tobytes()
            dtype = "f32"
        else:
            raw = t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
            dtype = "bf16"
        recorded.append(
            {
                "layer": layer,
                "name": name,
                "shape": list(t.shape),
                "dtype": dtype,
                "data_hex": raw.hex(),
            }
        )

    def out_hook(layer, name):
        def hook(_m, _i, out):
            emit(layer, name, out[0] if isinstance(out, tuple) else out)

        return hook

    def in_hook(layer, name):
        def hook(_m, inp):
            emit(layer, name, inp[0])

        return hook

    handles = []
    raw: dict[int, dict[str, torch.Tensor]] = {}
    m = model.model

    def qk_hook(layer, name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            raw.setdefault(layer, {})[name] = t.detach()
            emit(layer, name, t)

        return hook

    handles.append(m.embed_tokens.register_forward_hook(out_hook(None, "embed")))
    for i, layer in enumerate(m.layers):
        a = layer.self_attn
        handles += [
            layer.input_layernorm.register_forward_hook(out_hook(i, "attn_norm")),
            a.q_proj.register_forward_hook(qk_hook(i, "q")),
            a.k_proj.register_forward_hook(qk_hook(i, "k")),
            a.v_proj.register_forward_hook(out_hook(i, "v")),
            a.o_proj.register_forward_pre_hook(in_hook(i, "attn")),
            a.o_proj.register_forward_hook(out_hook(i, "o_proj")),
            layer.post_attention_layernorm.register_forward_pre_hook(
                in_hook(i, "resid_attn")
            ),
            layer.post_attention_layernorm.register_forward_hook(
                out_hook(i, "mlp_norm")
            ),
            layer.mlp.gate_proj.register_forward_hook(out_hook(i, "gate")),
            layer.mlp.up_proj.register_forward_hook(out_hook(i, "up")),
            layer.mlp.down_proj.register_forward_pre_hook(in_hook(i, "act")),
            layer.mlp.down_proj.register_forward_hook(out_hook(i, "down")),
            layer.register_forward_hook(out_hook(i, "resid_mlp")),
        ]
    final: dict[str, torch.Tensor] = {}
    handles.append(
        m.norm.register_forward_hook(
            lambda _m, _i, out: final.update(norm=out[0, -1:, :])
        )
    )

    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), use_cache=False)
    for h in handles:
        h.remove()

    # q_rope / k_rope: transformers' own rotation (its rotary embedding and
    # `apply_rotary_pos_emb`) applied to the recorded q/k, so the trace also carries the
    # post-RoPE tensors Turbine names.
    hidden = config.hidden_size
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden // heads)
    position_ids = torch.arange(len(ids)).unsqueeze(0)
    cos, sin = m.rotary_emb(raw[0]["q"], position_ids)
    for i in range(len(m.layers)):
        q = raw[i]["q"].view(1, len(ids), heads, head_dim).transpose(1, 2)
        k = raw[i]["k"].view(1, len(ids), kv_heads, head_dim).transpose(1, 2)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        emit(i, "q_rope", q_rope.transpose(1, 2).reshape(1, len(ids), heads * head_dim))
        emit(
            i,
            "k_rope",
            k_rope.transpose(1, 2).reshape(1, len(ids), kv_heads * head_dim),
        )

    emit(None, "final_norm", final["norm"])
    logits = final["norm"].float() @ model.lm_head.weight.float().T
    emit(None, "logits", logits)

    embed = [o for o in recorded if o["name"] == "embed"]
    layers = [
        next(o for o in recorded if o["layer"] == i and o["name"] == n)
        for i in range(len(m.layers))
        for n in LLAMA_OP_ORDER
    ]
    tail = [o for o in recorded if o["name"] in ("final_norm", "logits")]
    out = {
        "family": "llama",
        "model_dir": model_dir.name,
        "prompt_ids": ids,
        "row": "last" if keep_last else "all",
        "ops": embed + layers + tail,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out) + "\n")
    print(f"tiny_trace: wrote {len(out['ops'])} ops to {out_path}", file=sys.stderr)
    return 0


def trace_gpt_oss(model_dir: Path, ids, out_path: Path) -> int:
    from transformers import GptOssConfig, GptOssForCausalLM

    cfg_json = json.loads((model_dir / "config.json").read_text())
    if cfg_json.get("architectures") != ["GptOssForCausalLM"]:
        print(
            f"tiny_trace: unsupported architectures {cfg_json.get('architectures')}",
            file=sys.stderr,
        )
        return 2

    config = GptOssConfig.from_pretrained(model_dir)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = GptOssForCausalLM(config).to(torch.bfloat16).eval()
    missing = model.load_state_dict(load_state_dict(model_dir), strict=True)
    print(f"tiny_trace: loaded {missing}", file=sys.stderr)

    ops: list[dict] = []

    def rec(layer, name, t: torch.Tensor) -> None:
        t = t.detach().to(torch.float32).reshape(-1, t.shape[-1])
        ops.append(
            {
                "layer": layer,
                "name": name,
                "shape": list(t.shape),
                "data": t.flatten().tolist(),
            }
        )

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
            layer.post_attention_layernorm.register_forward_pre_hook(
                in_hook(i, "resid_attn")
            ),
            layer.post_attention_layernorm.register_forward_hook(
                out_hook(i, "mlp_norm")
            ),
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
        "model_dir": model_dir.name,
        "prompt_ids": ids,
        "ops": embed + layers + tail,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out) + "\n")
    print(f"tiny_trace: wrote {len(out['ops'])} ops to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--prompt-ids", help="comma-separated token ids (a tiny checkpoint)"
    )
    ap.add_argument(
        "--prompt-file", type=Path, help="prompts.jsonl (a real checkpoint)"
    )
    ap.add_argument("--prompt-id", help="the prompt id in --prompt-file")
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="append the committed reference's first N greedy tokens (needs --prompt-file)",
    )
    ap.add_argument("--row", choices=("all", "last"), default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    real = args.prompt_file is not None or args.prompt_id is not None
    if real and (args.prompt_file is None or args.prompt_id is None):
        print("tiny_trace: --prompt-file and --prompt-id go together", file=sys.stderr)
        return 2
    if not real and args.prompt_ids is None:
        print(
            "tiny_trace: one of --prompt-ids or (--prompt-file, --prompt-id) is required",
            file=sys.stderr,
        )
        return 2
    keep_last = (args.row or ("last" if real else "all")) == "last"

    cfg_json = json.loads((args.model_dir / "config.json").read_text())
    arch = (cfg_json.get("architectures") or [None])[0]
    if real:
        ids = prompt_ids(
            args.model_dir, args.prompt_file, args.prompt_id, args.context_tokens
        )
        if arch == "LlamaForCausalLM":
            return trace_llama_real(args.model_dir, ids, args.out, keep_last)
        if arch == "GptOssForCausalLM":
            return trace_gpt_oss(args.model_dir, ids, args.out)
        print(
            f"tiny_trace: unsupported architectures {cfg_json.get('architectures')}",
            file=sys.stderr,
        )
        return 2
    if arch != "GptOssForCausalLM":
        print(
            f"tiny_trace: unsupported architectures {cfg_json.get('architectures')}",
            file=sys.stderr,
        )
        return 2
    return trace_gpt_oss(
        args.model_dir, [int(x) for x in args.prompt_ids.split(",")], args.out
    )


if __name__ == "__main__":
    sys.exit(main())
