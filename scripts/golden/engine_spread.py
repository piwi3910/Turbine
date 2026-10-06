"""Golden engine spread: how far Turbine moves its own logprobs when only the summation order
changes, and what that does to each slug's tolerance (decision "5p golden tolerance vs engine
summation order", 2026-10-06 B).

Input: the captures of scripts/lab/golden-engine-spread.sh, one directory per golden slug, each
holding `turbine-golden capture` files (free-running greedy, top-20 logprobs):
c1-cold, c1-warm (prefix reuse), c1-heuristic (execution.gemm_autotune=false), c1-unfused
(execution.fused_ops=false) and par-00..15 (16 captures at once: batched GEMMs, CK split-KV
decode attention instead of b1).

Every capture is judged with the exact `turbine-golden compare` rule (the one
scripts/golden/self_spread.py implements; compare.rs compare_prompt): compare the reference
top-5 candidates at every position before the first token divergence, a candidate missing from
the capture's top-20 is a violation, "likely" when the reference logprob is above -2, else
"tail"; prefix ok = first 32 tokens identical or the divergence at a reference margin < 0.5.

Engine noise (what the bounds use): for every prompt, position and reference top-5 candidate,
the range (max - min) of the candidate's logprob over every capture whose tokens still equal the
reference's before that position (every capture the gate would judge there). Ranges over all
captures, not distances to one baseline capture: behind a reference near-tie only some captures
follow the reference's branch, and a single baseline that took the other branch would hide those
positions. Also printed: every variant judged against c1-cold with the compare rule (noise by
variant), and every capture judged against the committed reference (vs ref).

Bounds per tier (likely, tail), strict and batched:
- old: tolerance.json today (transformers' self-spread, or the BF16 floor);
- triangle: old + the slug's largest engine noise, rounded up to two decimals — a correct engine
  is within the reference's spread of the reference and within the engine noise of any reordered
  run of itself, so this is the most the measured noise can justify;
- observed: the largest engine-vs-reference |Δ| over all variants, rounded up;
- proposed: max(old, observed), capped at triangle (never looser than the noise justifies; a
  variant above the cap is flagged, since then something beyond reduction order moved it).
The token-identity rule (min_identical_prefix, min_prompts_passing, margin_nats) is unchanged.

usage: engine_spread.py <captures-dir> [--slug <slug>]... [--json <out.json>] [--slug-json <label>]

--slug-json writes each slug's report to tests/golden/<slug>/engine_spread.json, tagged with
<label> (the capture run, e.g. "ges3 2ba95514").
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"


def load_jsonl(path):
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def ceil2(x):
    return math.ceil(x * 100 - 1e-9) / 100


def judge(ref, cand, tol):
    """compare.rs compare_prompt for a free-running capture `cand` against `ref`."""
    n = len(ref["tokens"])
    toks = cand["tokens"]
    prefix = 0
    while prefix < n and prefix < len(toks) and toks[prefix] == ref["tokens"][prefix]:
        prefix += 1
    div = prefix if prefix < n else None
    margin = None
    if div is not None:
        vals = sorted((e[1] for e in ref["top_logprobs"][div]), reverse=True)
        margin = vals[0] - vals[1]
    likely, tail, missing = [], [], []
    for pos in range(div if div is not None else n):
        got = {int(t): lp for t, lp in cand["top_logprobs"][pos]}
        row = sorted(ref["top_logprobs"][pos], key=lambda e: (-e[1], e[0]))[
            : tol["top_k"]
        ]
        for tid, rlp in row:
            if tid not in got:
                missing.append((pos, tid))
                continue
            d = abs(got[tid] - rlp)
            (likely if rlp > tol["likely_logprob_floor"] else tail).append((d, pos))
    prefix_ok = prefix >= min(tol["min_identical_prefix"], n)
    excused = margin is not None and margin < tol["margin_nats"]
    return {
        "prefix": prefix,
        "margin": margin,
        "likely": max((d for d, _ in likely), default=0.0),
        "tail": max((d for d, _ in tail), default=0.0),
        "likely_all": [d for d, _ in likely],
        "tail_all": [d for d, _ in tail],
        "missing": missing[:1],
        "ok_prefix": prefix_ok or excused,
    }


def judge_all(refs, cands, tol):
    rows = {pid: judge(refs[pid], c, tol) for pid, c in cands.items() if pid in refs}
    worst = lambda k: max(((r[k], pid) for pid, r in rows.items()), default=(0.0, "-"))
    return {
        "prefix_ok": sum(r["ok_prefix"] for r in rows.values()),
        "prompts": len(rows),
        "likely": worst("likely"),
        "tail": worst("tail"),
        "missing": sum(bool(r["missing"]) for r in rows.values()),
        "likely_all": [d for r in rows.values() for d in r["likely_all"]],
        "tail_all": [d for r in rows.values() for d in r["tail_all"]],
        "per_prompt": {
            pid: (r["likely"], r["tail"], r["ok_prefix"]) for pid, r in rows.items()
        },
    }


def quantile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def range_noise(refs, caps, tol):
    """Engine noise on the reference's own path: for every prompt, position and reference top-5
    candidate, the range (max - min) of the candidate's logprob over every capture whose tokens
    still equal the reference's before that position, i.e. every capture the gate would judge
    there. Pairwise by construction, so positions behind a reference near-tie that only some
    captures follow are covered too (judging against one baseline capture loses them)."""
    likely, tail, worst = [], [], {"likely": (0.0, "-", -1), "tail": (0.0, "-", -1)}
    for pid, ref in refs.items():
        n = len(ref["tokens"])
        runs = []
        for cap in caps.values():
            c = cap.get(pid)
            if c is None:
                continue
            f = 0
            while f < n and f < len(c["tokens"]) and c["tokens"][f] == ref["tokens"][f]:
                f += 1
            runs.append((f, c))
        for pos in range(n):
            live = [c for f, c in runs if f > pos]
            if len(live) < 2:
                continue
            row = sorted(ref["top_logprobs"][pos], key=lambda e: (-e[1], e[0]))[
                : tol["top_k"]
            ]
            got = [{int(t): lp for t, lp in c["top_logprobs"][pos]} for c in live]
            for tid, rlp in row:
                vals = [g[tid] for g in got if tid in g]
                if len(vals) < 2:
                    continue
                d = max(vals) - min(vals)
                tier = "likely" if rlp > tol["likely_logprob_floor"] else "tail"
                (likely if tier == "likely" else tail).append(d)
                if d > worst[tier][0]:
                    worst[tier] = (d, pid, pos)
    return likely, tail, worst


def slug_report(slug, cap_dir):
    ref_path = GOLDEN / slug / "reference.jsonl"
    tol = json.loads((GOLDEN / slug / "tolerance.json").read_text())
    refs = load_jsonl(ref_path)
    caps = {p.stem: load_jsonl(p) for p in sorted(cap_dir.glob("*.jsonl"))}
    caps = {k: v for k, v in caps.items() if v}
    if "c1-cold" not in caps:
        return None
    base = caps["c1-cold"]
    noise = {k: judge_all(base, v, tol) for k, v in caps.items() if k != "c1-cold"}
    vs_ref = {k: judge_all(refs, v, tol) for k, v in caps.items()}
    old = {
        "likely": tol["max_abs_logprob_diff_likely"],
        "tail": tol["max_abs_logprob_diff_tail"],
        "likely_batched": tol.get(
            "max_abs_logprob_diff_likely_batched", tol["max_abs_logprob_diff_likely"]
        ),
        "tail_batched": tol.get(
            "max_abs_logprob_diff_tail_batched", tol["max_abs_logprob_diff_tail"]
        ),
    }
    all_l, all_t, worst_noise = range_noise(refs, caps, tol)
    n_l = max(all_l, default=0.0)
    n_t = max(all_t, default=0.0)
    c1 = [k for k in vs_ref if not k.startswith("par-")]
    par = [k for k in vs_ref if k.startswith("par-")]
    obs = lambda keys, tier: max((vs_ref[k][tier][0] for k in keys), default=0.0)
    out = {
        "slug": slug,
        "variants": sorted(caps),
        "old": old,
        "noise": {"likely": n_l, "tail": n_t},
        "noise_worst": worst_noise,
    }
    prop = {}
    for key, tier, keys in (
        ("likely", "likely", c1),
        ("tail", "tail", c1),
        ("likely_batched", "likely", c1 + par),
        ("tail_batched", "tail", c1 + par),
    ):
        tri = ceil2(old[key] + (n_l if tier == "likely" else n_t))
        o = ceil2(obs(keys, tier))
        prop[key] = {
            "observed": o,
            "triangle": tri,
            "proposed": min(tri, max(old[key], o)),
            "over_cap": o > tri,
        }
    out["bounds"] = prop
    out["noise_distribution"] = {
        tier: {q: quantile(xs, float(q)) for q in ("0.5", "0.9", "0.99", "1.0")}
        | {"n": len(xs)}
        for tier, xs in (("likely", all_l), ("tail", all_t))
    }
    strip = lambda d: {
        k: v for k, v in d.items() if not k.endswith("_all") and k != "per_prompt"
    }
    out["noise_by_variant"] = {k: strip(v) for k, v in noise.items()}
    out["vs_reference"] = {k: strip(v) for k, v in vs_ref.items()}
    return out


def print_report(r):
    print(f"== {r['slug']}  ({len(r['variants'])} captures)")
    nd = r["noise_distribution"]
    print(
        "  engine noise (range over captures on the reference path): likely max {:.4f} at {} (p50 {:.4f} p99 {:.4f}, n {}), tail max {:.4f} at {} (p50 {:.4f} p99 {:.4f}, n {})".format(
            r["noise"]["likely"],
            "{}@{}".format(*r["noise_worst"]["likely"][1:]),
            nd["likely"]["0.5"],
            nd["likely"]["0.99"],
            nd["likely"]["n"],
            r["noise"]["tail"],
            "{}@{}".format(*r["noise_worst"]["tail"][1:]),
            nd["tail"]["0.5"],
            nd["tail"]["0.99"],
            nd["tail"]["n"],
        )
    )
    for k, v in sorted(r["noise_by_variant"].items()):
        if k.startswith("par-") and k not in ("par-00",):
            continue
        print(
            f"    noise {k:<13} prefix {v['prefix_ok']}/{v['prompts']} likely {v['likely'][0]:.4f} ({v['likely'][1]}) tail {v['tail'][0]:.4f} ({v['tail'][1]}) missing {v['missing']}"
        )
    pars = [v for k, v in r["noise_by_variant"].items() if k.startswith("par-")]
    if pars:
        print(
            f"    noise par-*         prefix min {min(v['prefix_ok'] for v in pars)}/{pars[0]['prompts']} likely max {max(v['likely'][0] for v in pars):.4f} tail max {max(v['tail'][0] for v in pars):.4f}"
        )
    for k, v in sorted(r["vs_reference"].items()):
        if k.startswith("par-") and k != "par-00":
            continue
        print(
            f"    vs ref {k:<12} prefix {v['prefix_ok']}/{v['prompts']} likely {v['likely'][0]:.4f} ({v['likely'][1]}) tail {v['tail'][0]:.4f} ({v['tail'][1]}) missing {v['missing']}"
        )
    pars = [v for k, v in r["vs_reference"].items() if k.startswith("par-")]
    if pars:
        print(
            f"    vs ref par-*        prefix min {min(v['prefix_ok'] for v in pars)} likely max {max(v['likely'][0] for v in pars):.4f} tail max {max(v['tail'][0] for v in pars):.4f}"
        )
    for key, b in r["bounds"].items():
        flag = "  OVER CAP" if b["over_cap"] else ""
        print(
            f"  {key:<15} old {r['old'][key]:.2f}  observed {b['observed']:.2f}  triangle {b['triangle']:.2f}  proposed {b['proposed']:.2f}{flag}"
        )


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cap_root = Path(argv[0])
    slugs, out, label = [], None, None
    i = 1
    while i < len(argv):
        if argv[i] == "--slug":
            slugs.append(argv[i + 1])
        elif argv[i] == "--json":
            out = argv[i + 1]
        elif argv[i] == "--slug-json":
            label = argv[i + 1]
        else:
            sys.exit(f"unknown argument {argv[i]}")
        i += 2
    reports = []
    for d in sorted(p for p in cap_root.iterdir() if p.is_dir()):
        if slugs and d.name not in slugs:
            continue
        if not (GOLDEN / d.name / "tolerance.json").exists():
            continue
        r = slug_report(d.name, d)
        if r is None:
            print(f"== {d.name}: no c1-cold capture, skipped")
            continue
        print_report(r)
        reports.append(r)
        if label:
            (GOLDEN / d.name / "engine_spread.json").write_text(
                json.dumps({"captures": label} | r, indent=1) + "\n"
            )
    if out:
        Path(out).write_text(json.dumps(reports, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
