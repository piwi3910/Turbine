# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch==2.9.0", "transformers==4.57.1", "safetensors==0.6.2", "jinja2==3.1.6"]
# ///
"""OLMoE golden self-spread: transformers 4.57.1 variants scored against the committed reference
with the exact `turbine-golden compare` rule.

Teacher-forced on the reference tokens, which is exactly what greedy generation compares: up to
the first divergence the generated context equals the reference's, and the rule compares
nothing after it. Variants: attention (sdpa, eager) x compute dtype (bf16, fp32) x decode shape
(incremental = prompt prefill then one token at a time with the KV cache, as the reference was
made; full = one forward over prompt + reference tokens). The LM head runs in FP32 on the
final-norm output, as in scripts/golden/hf_reference.py.

usage: spread.py <model-dir> <reference.jsonl> <out.json> [variant,...]
"""

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

LIKELY_FLOOR = -2.0
TOP_K = 5
TOP_N = 20
MIN_PREFIX = 32
MARGIN_NATS = 0.5

model_dir, ref_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
ALL = [
    f"{dt}-{attn}-{mode}"
    for dt in ("bf16", "fp32")
    for attn in ("sdpa", "eager")
    for mode in ("incremental", "full")
]
variants = sys.argv[4].split(",") if len(sys.argv) > 4 else ALL
with open(ref_path) as f:
    refs = [json.loads(line) for line in f]


def hidden_states(model, ref, mode):
    prompt, toks = ref["prompt_token_ids"], ref["tokens"]
    with torch.inference_mode():
        if mode == "full":
            out = model.model(input_ids=torch.tensor([prompt + toks[:-1]]))
            return out.last_hidden_state[0, len(prompt) - 1 :].to(torch.float32)
        out = model.model(input_ids=torch.tensor([prompt]), use_cache=True)
        rows = [out.last_hidden_state[0, -1]]
        for tok in toks[:-1]:
            out = model.model(
                input_ids=torch.tensor([[tok]]),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
            rows.append(out.last_hidden_state[0, -1])
        return torch.stack(rows).to(torch.float32)


def judge(ref, lps):
    """`benches/turbine-bench/src/golden/compare.rs` compare_prompt on teacher-forced rows."""
    n = len(ref["tokens"])
    argmax = [int(torch.argmax(row)) for row in lps]
    prefix = 0
    while prefix < n and argmax[prefix] == ref["tokens"][prefix]:
        prefix += 1
    div = prefix if prefix < n else None
    margin = None
    if div is not None:
        vals = sorted((e[1] for e in ref["top_logprobs"][div]), reverse=True)
        margin = vals[0] - vals[1]
    likely = tail = 0.0
    missing = None
    worst_likely_pos = None
    for pos in range(div if div is not None else n):
        top = torch.topk(lps[pos], TOP_N)
        got = {int(i): float(v) for i, v in zip(top.indices, top.values, strict=True)}
        row = sorted(ref["top_logprobs"][pos], key=lambda e: (-e[1], e[0]))[:TOP_K]
        for tid, rlp in row:
            if tid not in got:
                missing = missing or [pos, tid]
                continue
            d = abs(got[tid] - rlp)
            if rlp > LIKELY_FLOOR:
                if d > likely:
                    likely, worst_likely_pos = d, pos
            else:
                tail = max(tail, d)
    prefix_ok = prefix >= min(MIN_PREFIX, n)
    excused = margin is not None and margin < MARGIN_NATS
    return {
        "id": ref["id"],
        "len": n,
        "prefix": prefix,
        "margin": margin,
        "likely": likely,
        "worst_likely_pos": worst_likely_pos,
        "tail": tail,
        "missing": missing,
        "prefix_or_excused": prefix_ok or excused,
    }


results = {}
loaded = None
model = head = None
for variant in variants:
    dt, attn, mode = variant.split("-")
    if loaded != (dt, attn):
        del model, head
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.bfloat16 if dt == "bf16" else torch.float32,
            attn_implementation=attn,
        )
        model.eval()
        head = model.lm_head.weight.to(torch.float32)
        loaded = (dt, attn)
        print(f"loaded {dt} {attn} in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    rows = []
    for ref in refs:
        lps = torch.log_softmax(hidden_states(model, ref, mode) @ head.T, dim=-1)
        v = judge(ref, lps)
        rows.append(v)
        print(variant, json.dumps(v), flush=True)
    results[variant] = rows
    print(f"{variant}: {time.time() - t0:.0f}s", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)

print("variant                 prefix_ok  max_likely  max_tail  missing")
for variant, rows in results.items():
    print(
        f"{variant:<24} {sum(r['prefix_or_excused'] for r in rows):>5}/16  "
        f"{max(r['likely'] for r in rows):>10.4f} {max(r['tail'] for r in rows):>9.4f}  "
        f"{sum(r['missing'] is not None for r in rows)}"
    )
