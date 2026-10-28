# -*- coding: utf-8 -*-
"""Post-E5 queue item 1 (external ruling 2026-10-27): rerun the four
reference arms {nac-no-clamp, ttc} x K in {2,4} (bs=32, seed 0, CIFAR-10,
eps_att=4/255) under the same conditions as the 2026-10-12 clamped arms
(15.23% / 36.44%), completing the same-batch six-element tuple for the
clamp paragraph.  ~25 min total.  Skip-completed keyed on SUMMARY in log.
"""
import subprocess, sys, time, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CONFIGS = [
    ("nac_K2", "nac", 2),
    ("nac_K4", "nac", 4),
    ("ttc_K2", "ttc", 2),
    ("ttc_K4", "ttc", 4),
]

for name, ca, k in CONFIGS:
    outdir = os.path.join("results", "clamp_rerun_20261028", f"{name}")
    os.makedirs(outdir, exist_ok=True)
    log_candidates = []
    for sub in ("pgd_eps_4.0", ):
        base = os.path.join(outdir, sub)
        if os.path.isdir(base):
            for d in os.listdir(base):
                p = os.path.join(base, d, "seed_0.log")
                if os.path.exists(p):
                    log_candidates.append(p)
    done = any("SUMMARY" in open(p, encoding="utf-8", errors="ignore").read()
               for p in log_candidates)
    if done:
        print(f"[skip] {name} already done", flush=True)
        continue
    t0 = time.time()
    print(f"[start] {name} (K={k}, counterattack={ca}, bs=32, seed 0)",
          flush=True)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "nac_fair_experiment.py"),
         "--evaluate", "True", "--batch_size", "32",
         "--test_set", "cifar10", "--test_attack_type", "pgd",
         "--test_eps", "4.0", "--test_numsteps", "10", "--test_stepsize", "1.0",
         "--method", "null_patch", "--root", "./data", "--dataset", "cifar10",
         "--seed", "0", "--num_seeds", "1",
         "--tau_thres", "0.2", "--beta", "2.0", "--ttc_eps", "4.0",
         "--ttc_numsteps", str(k), "--ttc_stepsize", "1.0",
         "--nac_momentum", "0.9", "--learnable_tau", "0.155",
         "--temperature", "75.0", "--counterattack", ca,
         "--outdir", outdir],
        capture_output=True, text=True)
    print(f"[done ] {name} rc={r.returncode} {(time.time()-t0)/60:.1f} min",
          flush=True)
    if r.returncode != 0:
        print(r.stderr[-1200:], flush=True)
print("[clamp rerun complete]", flush=True)
