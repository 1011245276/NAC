#!/usr/bin/env python3
"""
NAC: Aggregate multi-seed results from log files.

Usage:
    python aggregate_results.py --root ./results/main

Parses log files matching `<method>_<dataset>_seed<seed>.log` and produces
`aggregated_results.json` plus a Markdown summary table.
"""
import argparse
import json
import os
import re
from collections import defaultdict


def parse_log_file(path):
    """Extract (clean, adv, defended, gain) from a log file."""
    pattern = re.compile(
        r"INFO - (\w+):\s*clean=([\d.]+)\s*\|\s*adv=([\d.]+)\s*\|\s*adv\+(?:nac_m[\d.]+|ttc(?:_original)?|momentum|pure_la|doc|adam)=([\d.]+)"
    )
    matches = pattern.findall(open(path, encoding='utf-8', errors='ignore').read())
    return [(d, float(c), float(a), float(df)) for d, c, a, df in matches]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, 'aggregated_results.json')

    # Walk: <root>/<method>_<dataset>_seed<seed>.log
    agg = defaultdict(lambda: defaultdict(list))  # method -> dataset -> [(clean,adv,def)]

    for fname in sorted(os.listdir(args.root)):
        if not fname.endswith('.log'):
            continue
        m = re.match(r'^(ttc|nac|momentum|pure_la|adam|doc)_(.+)_(seed\d+)\.log$', fname)
        if not m:
            continue
        method, dataset, seed_tag = m.groups()
        fpath = os.path.join(args.root, fname)
        for dataset_in_log, clean, adv, defended in parse_log_file(fpath):
            agg[method][dataset_in_log].append({
                'seed': seed_tag,
                'clean': clean,
                'adv': adv,
                'defended': defended,
                'gain': defended - adv,
            })

    # Aggregate
    out_data = {}
    for method, datasets in agg.items():
        out_data[method] = {}
        for dataset, runs in datasets.items():
            cleans = [r['clean'] for r in runs]
            advs = [r['adv'] for r in runs]
            defed = [r['defended'] for r in runs]
            gains = [r['gain'] for r in runs]
            out_data[method][dataset] = {
                'n_seeds': len(runs),
                'clean_mean': sum(cleans) / len(cleans),
                'adv_mean': sum(advs) / len(advs),
                'defended_mean': sum(defed) / len(defed),
                'defended_std': (sum((d - sum(defed)/len(defed))**2 for d in defed) / len(defed))**0.5,
                'gain_mean': sum(gains) / len(gains),
            }

    with open(out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"[OK] Aggregated results written to {out}")

    # Markdown summary
    md_path = out.replace('.json', '.md')
    with open(md_path, 'w') as f:
        f.write(f"# Aggregated Results ({len(next(iter(agg.values())).values().__iter__().__next__())} seeds)\n\n")
        f.write("| Method | Dataset | Clean | Adv | Defended (mean ± std) | Gain |\n")
        f.write("|--------|---------|-------|-----|---------------------|------|\n")
        for method, datasets in agg.items():
            for dataset, runs in datasets.items():
                d = out_data[method][dataset]
                f.write(f"| {method} | {dataset} | {d['clean_mean']:.2f} | {d['adv_mean']:.2f} | {d['defended_mean']:.2f} ± {d['defended_std']:.2f} | {d['gain_mean']:.2f} |\n")
    print(f"[OK] Markdown summary: {md_path}")


if __name__ == '__main__':
    main()
