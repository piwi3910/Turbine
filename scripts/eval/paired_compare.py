#!/usr/bin/env python3
"""Paired significance read of two `turbine-golden eval` reports over the same task file.

Standard library only (analysis time; never in the build or serving path). Pairs the per-item
`results` of a baseline and a candidate report by `id` (both must cover the same ids), then prints:

- the accuracies and the drop (baseline minus candidate);
- lost / gained: items the baseline got right and the candidate wrong, and the reverse (the
  discordant pairs);
- McNemar's test with continuity correction: p = P(chi2_1 > (|lost - gained| - 1)^2 / (lost + gained)),
  and its exact (two-sided binomial) form, the one the GPTQ decisions quote;
- the 95 % confidence interval of the paired drop (normal approximation on the per-item difference);
- the literal verdict against `--max-drop` (drop <= max-drop passes), as `turbine-golden eval-compare`
  judges it, plus whether the whole CI lies above the bound.

Usage: paired_compare.py <baseline.json> <candidate.json> [--max-drop 0.04] [--json]
Exit code: 0 when the literal bound holds, 1 when it does not, 2 on a usage or pairing error.
"""

import argparse
import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return report, {r["id"]: bool(r["correct"]) for r in report["results"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--max-drop", type=float, default=0.04)
    ap.add_argument(
        "--json", action="store_true", help="print one JSON object instead of text"
    )
    a = ap.parse_args()

    try:
        base_report, base = load(a.baseline)
        cand_report, cand = load(a.candidate)
    except (OSError, ValueError, KeyError) as e:
        print(f"paired_compare: cannot read reports: {e}", file=sys.stderr)
        return 2
    if set(base) != set(cand):
        print(
            f"paired_compare: id sets differ (baseline {len(base)}, candidate {len(cand)}, "
            f"common {len(set(base) & set(cand))})",
            file=sys.stderr,
        )
        return 2

    n = len(base)
    lost = sum(1 for i in base if base[i] and not cand[i])
    gained = sum(1 for i in base if cand[i] and not base[i])
    acc_b = sum(base.values()) / n
    acc_c = sum(cand.values()) / n
    drop = acc_b - acc_c  # == (lost - gained) / n
    disc = lost + gained
    if disc:
        chi2 = (abs(lost - gained) - 1) ** 2 / disc if abs(lost - gained) >= 1 else 0.0
        p = math.erfc(math.sqrt(chi2 / 2))  # survival function of chi2 with 1 dof
    else:
        chi2, p = 0.0, 1.0
    k = min(lost, gained)
    p_exact = (
        min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / 2**disc)
        if disc
        else 1.0
    )
    # Per-item difference d_i in {-1, 0, 1}; mean = drop, variance of the mean below.
    var = (disc / n - drop**2) / n
    half = 1.959964 * math.sqrt(max(var, 0.0))
    ci = (drop - half, drop + half)
    passes = drop <= a.max_drop + 1e-12
    out = {
        "baseline": a.baseline,
        "candidate": a.candidate,
        "baseline_model": base_report.get("model"),
        "candidate_model": cand_report.get("model"),
        "n": n,
        "baseline_correct": sum(base.values()),
        "candidate_correct": sum(cand.values()),
        "baseline_accuracy": round(acc_b, 4),
        "candidate_accuracy": round(acc_c, 4),
        "drop": round(drop, 4),
        "lost": lost,
        "gained": gained,
        "mcnemar_chi2": round(chi2, 3),
        "mcnemar_p": float(f"{p:.3g}"),
        "mcnemar_exact_p": float(f"{p_exact:.3g}"),
        "drop_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "max_drop": a.max_drop,
        "verdict": "PASS" if passes else "FAIL",
        "ci_entirely_above_bound": ci[0] > a.max_drop,
    }
    if a.json:
        print(json.dumps(out))
    else:
        print(
            f"n={n} baseline={out['baseline_correct']}/{n}={acc_b:.4f} "
            f"candidate={out['candidate_correct']}/{n}={acc_c:.4f} drop={drop:.4f}"
        )
        print(
            f"lost={lost} gained={gained} mcnemar_chi2={chi2:.3f} p={p:.3g} exact_p={p_exact:.3g} "
            f"drop_ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
        print(
            f"verdict: {out['verdict']} (max-drop {a.max_drop}; CI entirely above the bound: "
            f"{out['ci_entirely_above_bound']})"
        )
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
