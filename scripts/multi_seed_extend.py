#!/usr/bin/env python3
"""Multi-seed extension: seeds 1,2 for 5 datasets at eps=1 (TTC + NAC)."""
import subprocess, sys, os

os.chdir(r"F:\codex\NAC\nac_project")
BASE = "python nac_fair_experiment.py --root ./data --batch_size 64 --test_attack_type pgd --test_eps 1 --test_numsteps 10 --test_stepsize 1 --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 --num_seeds 2 --seed 1 --outdir ./results/eps1"

DATASETS = "cifar100 STL10 flowers102 DTD ImageNet"

configs = [
    (f"{BASE} --test_set {DATASETS} --counterattack ttc", "TTC seeds 1-2"),
    (f"{BASE} --test_set {DATASETS} --counterattack nac --nac_momentum 0.9", "NAC seeds 1-2"),
]

for cmd, desc in configs:
    print(f"\n=== {desc} ===")
    print(f"CMD: {cmd}")
    sys.stdout.flush()
    r = subprocess.run(cmd, shell=True, capture_output=False)
    if r.returncode != 0:
        print(f"FAILED: {desc} (exit {r.returncode})")

print("\n=== ALL DONE ===")
