#!/usr/bin/env python3
"""
Independent verification collector for results/independent_verify/.

Reads ONLY the fresh seed_0.log files written by this acceptance run (nothing
else), verifies that the logged Namespace matches the pre-registered fixed
protocol, and auto-generates:

    results/independent_verify/independent_verify_results.json
    results/independent_verify/independent_verify_results.csv

No number is hand-copied: fresh accuracies are parsed from the fresh logs;
paper reference values are constants registered in
00_PROTOCOL_preregistered.txt *before* the run (with sources).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

NAC_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IV_DIR = os.path.join(NAC_PROJECT, "results", "independent_verify")

# ---- pre-registered protocol (must match 00_PROTOCOL_preregistered.txt) ----
EPS_4_255 = 4.0 / 255.0
STEP_1_255 = 1.0 / 255.0
EXPECT_STR = {
    "batch_size": "32",
    "test_attack_type": "pgd",
    "test_numsteps": "10",
    "ttc_numsteps": "2",
    "tau_thres": "0.2",
    "beta": "2.0",
    "nac_momentum": "0.9",
    "seed": "0",
    "arch": "vit_b32",
    "model": "clip",
}
EXPECT_FLOAT = {
    "test_eps": EPS_4_255,
    "ttc_eps": EPS_4_255,
    "test_stepsize": STEP_1_255,
    "ttc_stepsize": STEP_1_255,
}

# Paper reference values, registered BEFORE the run (see protocol file, [P1]/[P2])
PAPER = {
    "ttc": {
        "paper_value": 7.05,
        "paper_source": ("NAC_paper_sivp_8page.tex L268 (Discussion), same protocol "
                         "bs=32 seed0; log results/compare/pgd_eps_4.0/ttc/seed_0.log 2026-09-04"),
        "paper_3seed_mean": 7.10,
        "paper_3seed_std": 0.19,
        "paper_3seed_source": "Table tab:ablation, K=2 bs=64 seeds 0-2 (CIFAR-10)",
    },
    "nac_m0.9": {
        "paper_value": 17.17,
        "paper_source": ("NAC_paper_sivp_8page.tex L268 (Discussion), same protocol "
                         "bs=32 seed0; log results/compare/pgd_eps_4.0/nac_m0.9/seed_0.log 2026-09-04"),
        "paper_3seed_mean": 17.15,
        "paper_3seed_std": 0.50,
        "paper_3seed_source": "Table tab:ablation, K=2 bs=64 seeds 0-2 (CIFAR-10)",
    },
}
TOL_A = 0.2   # pp
TOL_B = 1.0   # pp

LINE_RE = re.compile(
    r"INFO - ([^:]+): clean=([\d.]+) \| adv=([\d.]+) \| adv\+(\S+)=([\d.]+) \| gain=([\d.]+)"
)
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_log(path: str):
    cfg, results, first_ts, last_ts = {}, {}, None, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TS_RE.match(line)
            if m:
                if first_ts is None:
                    first_ts = m.group(1)
                last_ts = m.group(1)
            if "Namespace(" in line:
                body = line.split("Namespace(", 1)[1].rsplit(")", 1)[0]
                cfg = {}
                for tok in body.split(","):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        cfg[k.strip()] = v.strip().strip("'")
            m = LINE_RE.search(line)
            if m:
                ds, clean, adv, tag, defended, gain = m.groups()
                results[ds] = {
                    "result_tag": tag,
                    "clean_acc": float(clean),
                    "adv_acc": float(adv),
                    "defended_acc": float(defended),
                    "gain": float(gain),
                }
    return cfg, results, first_ts, last_ts


def check_config(cfg: dict):
    issues = []
    for k, want in EXPECT_STR.items():
        got = cfg.get(k)
        if got != want:
            issues.append(f"{k}: got={got!r} expect={want!r}")
    for k, want in EXPECT_FLOAT.items():
        try:
            got = float(cfg.get(k, "nan"))
        except (TypeError, ValueError):
            issues.append(f"{k}: unparsable={cfg.get(k)!r}")
            continue
        if abs(got - want) > 1e-9:
            issues.append(f"{k}: got={got!r} expect={want!r}")
    return issues


def verdict_for(delta_pp: float) -> str:
    if delta_pp <= TOL_A:
        return "A"
    if delta_pp <= TOL_B:
        return "B"
    return "C"


def main() -> int:
    rows, problems = [], []
    for method in ("ttc", "nac_m0.9"):
        hits = glob.glob(os.path.join(IV_DIR, "pgd_eps_*", method, "seed_0.log"))
        if not hits:
            problems.append(f"{method}: fresh log not found under {IV_DIR}")
            continue
        log_path = os.path.abspath(hits[0])
        cfg, results, first_ts, last_ts = parse_log(log_path)
        issues = check_config(cfg)
        if "cifar10" not in results:
            problems.append(f"{method}: no cifar10 result line in {log_path}")
            continue
        r = results["cifar10"]
        p = PAPER[method]
        delta = round(abs(r["defended_acc"] - p["paper_value"]), 2)
        row = {
            "method": method,
            "seed": int(cfg.get("seed", -1)) if cfg.get("seed", "").lstrip("-").isdigit() else cfg.get("seed"),
            "clean_acc": r["clean_acc"],
            "adv_acc": r["adv_acc"],
            "defended_acc": r["defended_acc"],
            "gain": r["gain"],
            "paper_value_P1": p["paper_value"],
            "abs_diff_pp": delta,
            "verdict": verdict_for(delta),
            "paper_3seed_mean_P2": p["paper_3seed_mean"],
            "paper_3seed_std_P2": p["paper_3seed_std"],
            "paper_source": p["paper_source"],
            "log_path": log_path,
            "run_start": first_ts,
            "run_end": last_ts,
            "config_check_issues": issues,
        }
        rows.append(row)

    out_json = {
        "generated_by": "scripts/collect_independent_verify.py (auto-parsed, no hand-copied numbers)",
        "protocol": "results/independent_verify/00_PROTOCOL_preregistered.txt",
        "tolerance_pp": {"A": TOL_A, "B": TOL_B, "C": f">{TOL_B}"},
        "results": rows,
        "collection_problems": problems,
    }
    json_path = os.path.join(IV_DIR, "independent_verify_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(IV_DIR, "independent_verify_results.csv")
    cols = ["method", "seed", "clean_acc", "adv_acc", "defended_acc", "gain",
            "paper_value_P1", "abs_diff_pp", "verdict"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    # console summary
    print("=" * 72)
    print("INDEPENDENT VERIFY -- auto comparison (fresh run vs paper [P1])")
    print("=" * 72)
    for r in rows:
        print(f"{r['method']:<10} fresh={r['defended_acc']:6.2f}  paper={r['paper_value_P1']:6.2f}  "
              f"|diff|={r['abs_diff_pp']:5.2f}pp  verdict={r['verdict']}")
        print(f"           log: {r['log_path']}")
        print(f"           window: {r['run_start']} -> {r['run_end']}")
        if r["config_check_issues"]:
            print(f"           CONFIG ISSUES: {r['config_check_issues']}")
        else:
            print("           config check: all fixed conditions match")
    for p in problems:
        print(f"[PROBLEM] {p}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")

    if problems or any(r["config_check_issues"] for r in rows):
        print("OVERALL: D (cannot be completed / protocol mismatch) -- see problems above")
        return 2

    verdicts = [r["verdict"] for r in rows]
    if all(v == "A" for v in verdicts):
        print("OVERALL: A (fully reproduced)")
    elif "C" in verdicts:
        print("OVERALL: C (clearly inconsistent)")
    else:
        print("OVERALL: B (basically reproduced with random fluctuation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
