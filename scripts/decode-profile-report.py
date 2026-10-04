#!/usr/bin/env python3
"""decode-profile-report.py -- per-kernel account of a batch-1 decode step (5p Task 18).

Reads the rocprofv3 kernel-trace CSVs under target/decodebw/<label>-<model>/ and prints
per model: traced steps, per-kernel ms/step and per-call us, share of busy, the top
kernel-pair gaps, step span / busy / gaps, and effective weight-read bandwidth
(weight bytes / step span). usage: decode-profile-report.py <csv-dir>... [--weights <bytes>]
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from itertools import pairwise

DEFAULT_WEIGHT_BYTES = 6_431_025_664  # Llama-3.2-3B-Instruct BF16 served weights


def read_rows(csv_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        with open(path, newline="") as f:
            first = f.readline()
        if "Kernel_Name" not in first or "Start_Timestamp" not in first:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    begin = float(row["Start_Timestamp"])
                    end = float(row["End_Timestamp"])
                    name = row["Kernel_Name"]
                except (KeyError, TypeError, ValueError):
                    continue
                dur = float(row.get("Duration_Ns") or (end - begin))
                rows.append((begin, end, dur, name))
    rows.sort(key=lambda r: r[0])
    return rows


def split_steps(rows):
    # A decode step ends with logits_reduce (device sampling).
    steps = []
    cur = []
    for r in rows:
        if re.search(r"logits_reduce", r[3]):
            cur.append(r)
            steps.append(cur)
            cur = []
        else:
            cur.append(r)
    return [s for s in steps if s]


def summarize(steps):
    per_kernel = defaultdict(lambda: [0, 0.0])
    per_pair_gaps = defaultdict(float)
    spans, busy_totals = [], []
    for step in steps:
        begin = step[0][0]
        end = step[-1][1]
        spans.append(end - begin)
        busy = sum(d for _, _, d, _ in step)
        busy_totals.append(busy)
        for b, e, d, name in step:
            per_kernel[name][0] += 1
            per_kernel[name][1] += d
        for (_, e1, _, n1), (b2, _, _, n2) in pairwise(step):
            gap = b2 - e1
            if gap > 0:
                per_pair_gaps[(n1, n2)] += gap
    return per_kernel, per_pair_gaps, spans, busy_totals


def top_name(name):
    # rocprofv3 kernel names can carry long template arguments; keep the head.
    return name.split("<")[0][:72]


def main():
    args = sys.argv[1:]
    weight_bytes = DEFAULT_WEIGHT_BYTES
    if "--weights" in args:
        i = args.index("--weights")
        weight_bytes = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if not args:
        print(__doc__)
        return 2
    for csv_dir in args:
        rows = read_rows(csv_dir)
        steps = split_steps(rows)
        if not steps:
            print(f"{csv_dir}: no steps found (rows={len(rows)})")
            continue
        per_kernel, per_pair_gaps, spans, busy_totals = summarize(steps)
        n = len(steps)
        median_span = sorted(spans)[n // 2]
        median_busy = sorted(busy_totals)[n //2]
        print(f"== {csv_dir}: {n} steps, span median {median_span/1e6:.2f} ms "
              f"(p90 {sorted(spans)[int(n*0.9)]/1e6:.2f}), busy median {median_busy/1e6:.2f} ms, "
              f"gaps median {(median_span - median_busy)/1e6:.2f} ms")
        print(f"   effective weight-read bandwidth: {weight_bytes/ (median_span/1e9) / 1e9:.0f} GB/s "
              f"({weight_bytes/1e9:.2f} GB / {median_span/1e6:.2f} ms)")
        print("-- per-kernel (ms/step, us/call, share of busy)")
        busy_sum = sum(v[1] for v in per_kernel.values())
        for name, (calls, total_ns) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1]):
            print(f"   {top_name(name):<64} x{calls/n:>6.1f}  {total_ns/n/1e6:>7.3f} ms  "
                  f"{total_ns/calls/1e3:>7.1f} us  {100*total_ns/busy_sum:>5.1f}%")
        print("-- top gaps (ms/step)")
        for (a, b), gap in sorted(per_pair_gaps.items(), key=lambda kv: -kv[1])[:12]:
            print(f"   {top_name(a):<40} -> {top_name(b):<40} {gap/n/1e6:.3f} ms")


if __name__ == "__main__":
    sys.exit(main())
