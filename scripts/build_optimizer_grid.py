"""Build the unified K x alpha x optimizer accuracy grid from result logs.

Scans results/ directories for seed_*.log files, parses every Namespace line
plus the per-dataset metric lines that follow it, and emits a
provenance-complete CSV/JSON grid. One output row per (log, namespace,
metric) triple, so logs that contain several appended runs are split
correctly instead of being mixed.

Per project rules, every value is read from a log file -- nothing is
hand-entered. Each row carries its source log path so paper numbers stay
traceable.

Usage:
  python scripts/build_optimizer_grid.py                     # all datasets
  python scripts/build_optimizer_grid.py --dataset cifar10   # one dataset
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys

METRIC_RE = re.compile(
    r"INFO - ([\w\d]+): clean=([\d.]+) \| adv=([\d.]+) \| adv\+([\w.]+)=([\d.]+) \| gain=(-?[\d.]+)\s*$"
)

KV_RE = re.compile(r"(\w+)=(\"[^\"]*\"|'[^']*'|\[[^\]]*\]|[^,)]+)")

# Keys kept from the Namespace line (small subset; full dict stays in JSON meta).
KEEP_KEYS = [
    "batch_size", "seed", "test_attack_type", "test_eps", "test_numsteps",
    "test_stepsize", "arch", "counterattack", "tau_thres", "beta",
    "ttc_eps", "ttc_numsteps", "ttc_stepsize", "nac_momentum",
    "mac_eps", "mac_numsteps", "mac_num_views", "mac_tau_thres", "mac_tau_temp",
    "learnable_tau", "temperature", "prompt_template", "prompt_texts_file",
    "nac_clamp_lookahead", "victim_resume",
]


def parse_namespace(line):
    body = line.split("Namespace(", 1)[1].rsplit(")", 1)[0]
    meta = {}
    for m in KV_RE.finditer(body):
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith(("'", '"')):
            val = raw[1:-1]
        elif raw.startswith("["):
            val = raw
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        meta[key] = val
    return meta


def derive_row(src, meta, m):
    dataset, clean, adv, tag, defended, gain = m.groups()
    ca = meta.get("counterattack", "")
    try:
        if ca == "mac":
            k = meta.get("mac_numsteps")
            eps_def255 = round(float(meta.get("mac_eps", 0.0)) * 255.0, 2)
            alpha255 = round(eps_def255 / 4.0, 3)  # official MAC rule: step = eps/4
        else:
            k = meta.get("ttc_numsteps")
            eps_def255 = round(float(meta.get("ttc_eps", 0.0)) * 255.0, 2)
            alpha255 = round(float(meta.get("ttc_stepsize", 0.0)) * 255.0, 3)
        atk_eps255 = round(float(meta.get("test_eps", 0.0)) * 255.0, 2)
    except (TypeError, ValueError):
        k, eps_def255, alpha255, atk_eps255 = None, None, None, None
    row = {
        "dataset": dataset, "method_tag": tag, "counterattack": ca,
        "K": k, "alpha_255": alpha255, "eps_def_255": eps_def255,
        "attack_steps": meta.get("test_numsteps"), "attack_eps_255": atk_eps255,
        "attack_step_255": round(float(meta.get("test_stepsize", 0.0)) * 255.0, 3)
        if meta.get("test_stepsize") not in (None, "") else None,
        "bs": meta.get("batch_size"), "seed": meta.get("seed"),
        "arch": meta.get("arch"), "clean": float(clean), "adv": float(adv),
        "defended": float(defended), "gain": float(gain),
        "src": os.path.relpath(src).replace("\\", "/"),
    }
    row["_meta"] = {key: meta.get(key) for key in KEEP_KEYS if key in meta}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", help="results root to scan")
    ap.add_argument("--dataset", default=None, help="filter by dataset name")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.root, "unified_optimizer_table")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    n_files = 0
    for path in sorted(glob.glob(os.path.join(args.root, "**", "seed_*.log"), recursive=True)):
        if "_archive" in path:
            continue  # archived logs are superseded; ledger documents each quarantine
        n_files += 1
        meta = None
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if " - INFO - Namespace(" in line:
                    meta = parse_namespace(line)
                    continue
                m = METRIC_RE.search(line)
                if m and meta is not None:
                    rows.append(derive_row(path, meta, m))

    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(outdir, f"optimizer_grid_{stamp}.csv")
    json_path = os.path.join(outdir, f"optimizer_grid_{stamp}.json")
    fields = [k for k in rows[0].keys() if k != "_meta"] if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": stamp,
            "root": args.root,
            "n_logs": n_files,
            "n_rows": len(rows),
            "rows": rows,
        }, fh, indent=1, default=str)

    # Compact pivot for CIFAR-10-style grids: (K, alpha, method) -> defended values
    pivot = {}
    for r in rows:
        key = (r["dataset"], r["attack_eps_255"], r["K"], r["alpha_255"], r["method_tag"], r["bs"])
        pivot.setdefault(key, []).append(r["defended"])
    print(f"logs scanned: {n_files} | rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("\npivot (dataset, atk_eps, K, alpha, method, bs) -> defended runs:")
    for key in sorted(pivot, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]))):
        vals = ", ".join(f"{v:.2f}" for v in sorted(pivot[key]))
        print(f"  {key}: [{vals}]")


if __name__ == "__main__":
    main()
