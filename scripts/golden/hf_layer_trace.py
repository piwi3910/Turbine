#!/usr/bin/env python3
"""Print transformers' per-layer hidden-state summaries for a prompt (fixture diagnostics).

Used to diff the engine's CPU-reference trace (TURBINE_TRACE_DUMP=1) against transformers on the
same token ids — the first divergent layer names the wrong op for the hybrid families.

    uv run scripts/golden/hf_layer_trace.py --model-dir /models/qwen3.5-9b \
        --ids '[760,3712,314,279,17933,3379,11690,303,279,10805,43872,9018,11,948]'

Prints, per decoder layer, the last position's first four values, RMS and mean of the layer
output (`hidden_states[layer + 1]`), in the same shape as the engine's `resid_mlp` dump.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ids", required=True, help="JSON list of prompt token ids")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    ids = json.loads(args.ids)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="cpu",
        trust_remote_code=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids], dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )
    for layer, h in enumerate(out.hidden_states):
        row = h[0, -1].float()
        print(
            f"hf layer={layer} shape={list(h.shape)} rms={row.pow(2).mean().sqrt().item():.6f} "
            f"mean={row.mean().item():.6f} first={row[:4].tolist()}"
        )


if __name__ == "__main__":
    main()
