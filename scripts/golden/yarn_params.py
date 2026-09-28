# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==4.57.1",
# ]
# ///
"""Records transformers' YaRN RoPE parameters for the Phase 6a fixture (spec S-15).

For each of four YaRN configurations the script builds a `LlamaConfig` carrying only the keys
`_compute_yarn_parameters` reads (rope_theta, head_dim, max_position_embeddings, rope_scaling),
instantiates `LlamaRotaryEmbedding` from it (the rotary module every decoder family of
transformers 4.57.1 uses) and records its `inv_freq` (FP32, written as exact decimal floats)
and `attention_scaling` (the factor transformers multiplies cos and sin by). No weights are
read and nothing is downloaded.

The configurations:
- `llama-yarn16`: the Llama-3.2-3B-Instruct override of the Phase 6a proof (factor 16 over
  8192 positions, beta_fast 32, beta_slow 1).
- `qwen3-yarn4`: Qwen3's documented YaRN (factor 4 over 32768 positions, theta 1e6).
- `gpt-oss-yarn32`: gpt-oss-20b's shape (factor 32 over 4096 positions, theta 150000,
  head_dim 64, `truncate: false`).
- `mscale-pair`: a DeepSeek-style entry with `mscale` / `mscale_all_dim`, whose ratio sets the
  attention factor.

Output: JSON `{"transformers", "torch", "configs": [{"name", "config", "inv_freq",
"attention_factor"}]}`, `config` holding the four keys above exactly as passed.

usage: uv run scripts/golden/yarn_params.py --out crates/turbine-model/tests/fixtures/yarn_params.json
"""

import argparse
import json

import torch
import transformers
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

CONFIGS = [
    {
        "name": "llama-yarn16",
        "config": {
            "rope_theta": 500000.0,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        },
    },
    {
        "name": "qwen3-yarn4",
        "config": {
            "rope_theta": 1000000.0,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        },
    },
    {
        "name": "gpt-oss-yarn32",
        "config": {
            "rope_theta": 150000.0,
            "head_dim": 64,
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
            },
        },
    },
    {
        "name": "mscale-pair",
        "config": {
            "rope_theta": 10000.0,
            "head_dim": 64,
            "max_position_embeddings": 163840,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 40.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 0.707,
                "mscale_all_dim": 1.0,
            },
        },
    },
]


def record(entry: dict) -> dict:
    """The rotary module's inverse frequencies and attention scaling for one configuration."""
    c = entry["config"]
    heads = 8
    config = LlamaConfig(
        hidden_size=c["head_dim"] * heads,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=c["head_dim"],
        rope_theta=c["rope_theta"],
        max_position_embeddings=c["max_position_embeddings"],
        rope_scaling=dict(c["rope_scaling"]),
    )
    emb = LlamaRotaryEmbedding(config, device=torch.device("cpu"))
    assert emb.rope_type == "yarn", emb.rope_type
    inv_freq = emb.inv_freq
    assert inv_freq.dtype == torch.float32, inv_freq.dtype
    return {
        "name": entry["name"],
        "config": c,
        "inv_freq": [float(x) for x in inv_freq.tolist()],
        "attention_factor": float(emb.attention_scaling),
    }


def main() -> None:
    """Writes the fixture file named by --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "configs": [record(e) for e in CONFIGS],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
