# -*- coding: utf-8 -*-
"""E1 fixed-budget caliber (pre-registered 2026-10-28, see
thesis_proposal/研究二实验计划.md E1 双口径规划; commit a479aff predates
this run).  Budget constant at 10 counterattack evaluations per sample:
(M_att, steps) in {(1,10),(2,5),(5,2),(10,1)} x {ttc,nac}, M_def=1
deployment, 1024 samples, seed 0, bs=32.  Skip-completed keyed on JSON.
"""
import subprocess, sys, time, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

GRID = [(1, 10), (2, 5), (5, 2), (10, 1)]
METHODS = ["ttc", "nac"]

for m_att, steps in GRID:
    for method in METHODS:
        outdir = os.path.join("results", "e1_fixed_budget",
                              f"{method}_M{m_att}_S{steps}_seed0")
        marker = os.path.join(outdir, "e4_results.json")
        done = False
        if os.path.exists(marker):
            try:
                d = json.load(open(marker))
                done = f"{method}_seed0_Mdef1" in d.get("results", {})
            except Exception:
                done = False
        if done:
            print(f"[skip] {method} M{m_att}xS{steps} already done", flush=True)
            continue
        t0 = time.time()
        print(f"[start] {method} M{m_att} x S{steps} (budget=10, 1024 samples)",
              flush=True)
        r = subprocess.run(
            [sys.executable, os.path.join("scripts", "e4_aggregate_pipeline.py"),
             "--methods", method, "--m_def_list", "1", "--seeds", "0",
             "--m_att", str(m_att), "--steps", str(steps),
             "--max_batches", "32", "--batch_size", "32",
             "--outdir", outdir],
            capture_output=True, text=True)
        print(f"[done ] {method} M{m_att}xS{steps} rc={r.returncode} "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        if r.returncode != 0:
            print(r.stderr[-1200:], flush=True)
print("[E1 fixed-budget complete]", flush=True)
