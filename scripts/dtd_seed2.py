#!/usr/bin/env python3
"""Multi-seed: DTD seed 2 only."""
import subprocess, sys, os

os.chdir(r"F:\codex\NAC\nac_project")
ROOT = r"F:\codex\NAC\data"
BASE = f"python nac_fair_experiment.py --root {ROOT} --batch_size 64 --test_attack_type pgd --test_eps 1 --test_numsteps 10 --test_stepsize 1 --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 --num_seeds 1 --seed 2 --outdir ./results/eps1 --test_set dtd"

for method in ["ttc", "nac"]:
    nac_flag = " --nac_momentum 0.9" if method == "nac" else ""
    cmd = f"{BASE} --counterattack {method}{nac_flag}"
    print(f"\n=== {method.upper()} DTD seed 2 ===")
    sys.stdout.flush()
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"FAILED: {method}")
    else:
        print(f"OK: {method}")

print("\n=== DONE ===")
