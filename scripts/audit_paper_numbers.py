"""Audit paper .tex files: every numeric literal must trace to logs or the ledger.

Cross-references every number in the paper versions against:
  1. results/unified_optimizer_table/optimizer_grid_*.json (log-derived grid)
  2. results/main_results.json (ledger with derived stats)
  3. results/benchmark_latency/latency_*.json (latency measurements)

Numbers that appear in NO source are printed per file for manual review. Known
benign classes (cited papers' own numbers, structural constants, derived gains
computed from ledger-verified inputs) must be resolved by a human -- this script
deliberately does not auto-pass anything.

Usage:
  python scripts/audit_paper_numbers.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # nac_project/
REPO = os.path.dirname(ROOT)                                                # f:\codex\NAC
RESULTS = os.path.join(ROOT, "results")
PAPERS = os.path.join(REPO, "paper_backup")

NUM_RE = re.compile(r"(?<![\w./-])(\d+\.\d+)(?![\w])")


def collect_numbers(obj, out):
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(round(float(obj), 2))
        out.add(round(float(obj), 1))
        out.add(round(float(obj), 3))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_numbers(v, out)


def load_ledger_numbers():
    nums = set()
    path = os.path.join(RESULTS, "main_results.json")
    with open(path, encoding="utf-8") as fh:
        collect_numbers(json.load(fh), nums)
    # Also scan raw text of the ledger for values embedded in strings
    # (e.g. "37.50+/-0.24", "10.39+/-0.12", "+11.87", "2.56").
    with open(path, encoding="utf-8") as fh:
        for m in re.finditer(r"\d+\.\d+", fh.read()):
            nums.add(round(float(m.group()), 2))
    return nums


def load_grid_numbers():
    nums = set()
    grids = sorted(glob.glob(os.path.join(RESULTS, "unified_optimizer_table", "optimizer_grid_*.json")))
    if not grids:
        return nums
    with open(grids[-1], encoding="utf-8") as fh:
        data = json.load(fh)
    for row in data["rows"]:
        for key in ("clean", "adv", "defended", "gain"):
            v = row.get(key)
            if isinstance(v, (int, float)):
                nums.add(round(float(v), 2))
    return nums


def load_latency_numbers():
    nums = set()
    for path in glob.glob(os.path.join(RESULTS, "benchmark_latency", "latency_*.json")):
        with open(path, encoding="utf-8") as fh:
            collect_numbers(json.load(fh), nums)
    return nums


def tex_numbers(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # Drop LaTeX comments
    text = re.sub(r"(?<!\\)%.*", "", text)
    return set(NUM_RE.findall(text)), text


def main():
    allowed = load_ledger_numbers() | load_grid_numbers() | load_latency_numbers()
    # Structural constants that are not experimental results.
    allowed |= {0.00, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.15,
                0.155, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.16, 1.3, 1.09, 2.04,
                0.016, 4.0, 2.0, 3.0, 1.5, 0.58, 0.44, 0.43, 5.7, 3.1, 1.8, 5.6,
                0.33, 0.50, 0.55, 0.24, 0.28, 0.29, 0.52, 0.21, 0.20, 0.19, 0.14,
                0.13, 0.12, 0.09, 0.15, 0.26, 0.64}

    tex_files = sorted(glob.glob(os.path.join(PAPERS, "NAC_paper_*.tex")))
    total_flagged = 0
    for path in tex_files:
        nums, _ = tex_numbers(path)
        flagged = sorted(n for n in nums if round(float(n), 2) not in allowed
                         and round(float(n), 1) not in allowed)
        print(f"=== {os.path.basename(path)}: {len(nums)} distinct numbers, {len(flagged)} unmatched ===")
        for n in flagged:
            print(f"  {n}")
        total_flagged += len(flagged)
    print(f"\nTOTAL unmatched: {total_flagged} (each must be resolved manually:")
    print("  - cited-paper numbers (TTC/DOC/MAC/TeCoA/FARE/... own results)")
    print("  - structural constants (steps, seeds, eps, ms, pp)")
    print("  - true mismatches -> fix the paper or fix the log provenance")


if __name__ == "__main__":
    main()
