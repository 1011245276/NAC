# -*- coding: utf-8 -*-
"""Research-2 E1: EOT M-scan launcher. Sequential, one subdir per M (no overwrite).
M=5 already exists in results/adaptive_attack_eot/20261014_202836 (E-E) and is reused."""
import subprocess, sys, time, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # nac_project root
MS = [1, 2, 10, 20]
os.chdir(ROOT)

for m in MS:
    outdir = os.path.join("results", "research2_eot_mscan", f"m{m}")
    if os.path.exists(os.path.join(outdir, "eot_results.json")):
        print(f"[skip] m{m} already has eot_results.json", flush=True)
        continue
    t0 = time.time()
    print(f"[start] M={m} -> {outdir}", flush=True)
    r = subprocess.run([sys.executable, os.path.join("scripts", "adaptive_attack_eot.py"),
                        "--eot_inits", str(m), "--outdir", outdir],
                       capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    print(f"[done ] M={m} rc={r.returncode} {dt:.1f} min", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
        sys.exit(1)
print("[all done]", flush=True)
