# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch==2.9.0", "transformers==4.57.1", "safetensors==0.6.2", "jinja2==3.1.6"]
# ///
"""transformers' own spread on the YaRN golden (Phase 6a plan Task 28): `self_spread.py` with
`config.json`'s `rope_scaling` replaced by the factor-16 override, exactly as
`hf_reference.py --config-override` made `tests/golden/llama-3.2-3b-instruct-yarn16/`.

Runs `scripts/golden/self_spread.py` unchanged (its variants, teacher forcing and the
`turbine-golden compare` rule) after wrapping `AutoModelForCausalLM.from_pretrained` so every
variant loads the override. Fixture time only; reads the weights, downloads nothing.

usage: uv run scripts/golden/yarn_self_spread.py [--fold] <model-dir> <reference.jsonl> <out.json> [variant,...]

`--fold` applies the attention factor as Turbine does (softmax scale × factor², cos and sin
unscaled; user decision 2026-09-28, Q19), to measure what that placement alone costs.
"""

import json
import runpy
import sys
from pathlib import Path

import transformers
from transformers import AutoConfig

OVERRIDE = {
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 16.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32,
        "beta_slow": 1,
    }
}

_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained


def _with_override(path, *args, **kwargs):
    config = AutoConfig.from_pretrained(path)
    for key, value in OVERRIDE.items():
        setattr(config, key, value)
    model = _from_pretrained(path, *args, config=config, **kwargs)
    rotary = model.model.rotary_emb
    print(
        f"override {json.dumps(OVERRIDE)}: rotary {rotary.rope_type}, "
        f"attention scaling {rotary.attention_scaling}",
        flush=True,
    )
    if FOLD:
        # Turbine's placement of the attention factor (user decision 2026-09-28, Q19): cos and
        # sin unscaled, the softmax scale multiplied by factor² instead.
        factor = rotary.attention_scaling
        rotary.attention_scaling = 1.0
        for layer in model.model.layers:
            layer.self_attn.scaling *= factor * factor
        print(f"folded: softmax scale x {factor * factor}", flush=True)
    return model


# `--fold` (first argument): the attention factor folded into the softmax scale, as Turbine
# applies it, instead of transformers' scaling of cos and sin.
FOLD = len(sys.argv) > 1 and sys.argv[1] == "--fold"
if FOLD:
    del sys.argv[1]


transformers.AutoModelForCausalLM.from_pretrained = _with_override
sys.argv = [str(Path(__file__).with_name("self_spread.py")), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
