#!/usr/bin/env python3
"""Multi-seed: ImageNet-100 only (seeds 1,2, TTC+NAC)."""
import subprocess, sys, os

os.chdir(r"F:\codex\NAC\nac_project")
ROOT = r"F:\codex\NAC\data"
BASE = f"python nac_fair_experiment.py --root {ROOT} --batch_size 64 --test_attack_type pgd --test_eps 1 --test_numsteps 10 --test_stepsize 1 --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 --num_seeds 2 --seed 1 --outdir ./results/eps1"

for method in ["ttc", "nac"]:
    nac_flag = " --nac_momentum 0.9" if method == "nac" else ""
    cmd = f"{BASE} --test_set ImageNet --counterattack {method}{nac_flag}"
    print(f"\n=== {method.upper()} ImageNet seeds 1-2 ===")
    sys.stdout.flush()
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"FAILED: {method} (exit {r.returncode})")
    else:
        print(f"OK: {method}")

print("\n=== DONE ===")
