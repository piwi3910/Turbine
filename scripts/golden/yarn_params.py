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

With `--rope-out` (plan Task 28a, the YaRN attention factor on cos/sin) it also writes the
rotation fixture of `llama-yarn16`: for a few positions, transformers' cos and sin (FP32, already
multiplied by the attention factor, and as the BF16 values `LlamaRotaryEmbedding` hands a BF16
model), seeded random BF16 q (2 heads) and k (1 head) and their rotation by
`apply_rotary_pos_emb` in BF16. BF16 values are written as their 16-bit patterns. JSON
`{"transformers", "torch", "name", "config", "attention_factor", "inv_freq", "positions",
"q_heads", "k_heads", "cos_f32", "sin_f32", "cos_bf16", "sin_bf16", "q", "k", "q_rot", "k_rot"}`,
per-position rows of `head_dim / 2` (cos, sin) or `heads × head_dim` (q, k) values.

usage: uv run scripts/golden/yarn_params.py --out crates/turbine-model/tests/fixtures/yarn_params.json \
  [--rope-out crates/turbine-model/tests/fixtures/yarn_rope.json]
"""

import argparse
import json

import torch
import transformers
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

# Positions of the rotation fixture: the start, small offsets, the original context's end and
# positions past it (the ≈ 12,000-token golden prompt and beyond).
ROPE_POSITIONS = [0, 1, 5, 127, 1000, 8191, 12000, 60000]
ROPE_Q_HEADS = 2
ROPE_K_HEADS = 1
ROPE_SEED = 20260929

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


def rotary(c: dict) -> LlamaRotaryEmbedding:
    """transformers' rotary module for one fixture configuration."""
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
    return emb


def record(entry: dict) -> dict:
    """The rotary module's inverse frequencies and attention scaling for one configuration."""
    c = entry["config"]
    emb = rotary(c)
    inv_freq = emb.inv_freq
    assert inv_freq.dtype == torch.float32, inv_freq.dtype
    return {
        "name": entry["name"],
        "config": c,
        "inv_freq": [float(x) for x in inv_freq.tolist()],
        "attention_factor": float(emb.attention_scaling),
    }


def bf16_bits(t: torch.Tensor) -> list:
    """The 16-bit patterns of a BF16 tensor, flattened."""
    assert t.dtype == torch.bfloat16, t.dtype
    return [int(v) & 0xFFFF for v in t.contiguous().view(torch.int16).flatten().tolist()]


def record_rotation(entry: dict) -> dict:
    """transformers' YaRN cos/sin (× the attention factor) and a BF16 rotation of q and k."""
    c = entry["config"]
    emb = rotary(c)
    head_dim = c["head_dim"]
    half = head_dim // 2
    positions = torch.tensor([ROPE_POSITIONS], dtype=torch.long)
    probe32 = torch.zeros(1, dtype=torch.float32)
    probe16 = torch.zeros(1, dtype=torch.bfloat16)
    cos32, sin32 = emb(probe32, positions)
    cos16, sin16 = emb(probe16, positions)
    assert cos32.dtype == torch.float32 and cos16.dtype == torch.bfloat16
    gen = torch.Generator().manual_seed(ROPE_SEED)
    n = len(ROPE_POSITIONS)
    q = (4.0 * torch.randn(1, ROPE_Q_HEADS, n, head_dim, generator=gen)).to(torch.bfloat16)
    k = (4.0 * torch.randn(1, ROPE_K_HEADS, n, head_dim, generator=gen)).to(torch.bfloat16)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos16, sin16)
    assert q_rot.dtype == torch.bfloat16 and k_rot.dtype == torch.bfloat16

    def per_token(t: torch.Tensor) -> list:
        # [1, heads, n, d] -> per position, heads × head_dim values
        return [bf16_bits(t[0, :, i, :]) for i in range(n)]

    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "name": entry["name"],
        "config": c,
        "attention_factor": float(emb.attention_scaling),
        "inv_freq": [float(x) for x in emb.inv_freq.tolist()],
        "positions": ROPE_POSITIONS,
        "q_heads": ROPE_Q_HEADS,
        "k_heads": ROPE_K_HEADS,
        "cos_f32": [[float(x) for x in cos32[0, i, :half].tolist()] for i in range(n)],
        "sin_f32": [[float(x) for x in sin32[0, i, :half].tolist()] for i in range(n)],
        "cos_bf16": [bf16_bits(cos16[0, i, :half]) for i in range(n)],
        "sin_bf16": [bf16_bits(sin16[0, i, :half]) for i in range(n)],
        "q": per_token(q),
        "k": per_token(k),
        "q_rot": per_token(q_rot),
        "k_rot": per_token(k_rot),
    }


def main() -> None:
    """Writes the fixture file named by --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rope-out")
    args = p.parse_args()
    out = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "configs": [record(e) for e in CONFIGS],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    if args.rope_out:
        rotation = record_rotation(CONFIGS[0])
        with open(args.rope_out, "w") as f:
            json.dump(rotation, f)
            f.write("\n")


if __name__ == "__main__":
    main()
