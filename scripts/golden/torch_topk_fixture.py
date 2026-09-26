# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch==2.9.0"]
# ///
"""Records which indices torch.topk (CPU, float32, largest, sorted) selects on rows full of ties.

transformers' MoE routers (OLMoE: softmax over BF16 router logits, then torch.topk) resolve
ties the way torch's CPU top-k does: libstdc++ std::nth_element (introselect), or
std::partial_sort when k * 64 <= n. That order is not "lower id first", so the Turbine router
emulates it; this fixture pins the emulation to torch.

Output (the file crates/turbine-kernels/src/cpu/torch_topk_fixture.txt): one case per line,
`k;v0 v1 ...;i0 i1 ...` with the selected indices ascending (the set is what routing uses).

usage: uv run scripts/golden/torch_topk_fixture.py --out <file> [--cases N] [--seed S]
"""

import argparse
import random

import torch


def main() -> None:
    """Writes the fixture file named by --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    rng = random.Random(args.seed)
    # (n, k): OLMoE's 64/8 dominates; small rows; wide rows whose k takes the partial_sort path.
    shapes = [(64, 8)] * 6 + [(64, 1), (64, 2), (64, 16), (8, 2), (4, 2), (5, 3)]
    shapes += [(128, 2), (256, 4), (256, 8), (256, 1)]
    lines = []
    for _ in range(args.cases):
        n, k = rng.choice(shapes)
        if rng.random() < 0.7:
            # A few distinct levels: many ties, often straddling the k-th place.
            levels = rng.choice([1, 2, 3, 4, 6, 8, 16])
            vals = [rng.randint(0, levels) * 0.25 for _ in range(n)]
        else:
            # Router-like logits rounded to BF16, then quantised further to force ties.
            vals = [round(rng.gauss(0.0, 1.5) * 8) / 8 for _ in range(n)]
        t = torch.tensor(vals, dtype=torch.float32)
        idx = sorted(torch.topk(t, k).indices.tolist())
        lines.append(
            f"{k};{' '.join(repr(v) for v in vals)};{' '.join(str(i) for i in idx)}"
        )
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
