# -*- coding: utf-8 -*-
"""Research-2 E1 v2 (trend-first): M in {1,2,10} x {ttc,nac} x seed 0.
M=5 seed 0 already covered by E-E. DOC/extra seeds deferred until trend confirmed."""
import subprocess, sys, time, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

JOBS = []
for m in [1, 2, 10]:
    outdir = os.path.join("results", "research2_eot_mscan", f"m{m}", "trend_seed0")
    if os.path.exists(os.path.join(outdir, "eot_results.json")):
        print(f"[skip] M={m} trend already done", flush=True)
        continue
    JOBS.append((m, outdir))

for m, outdir in JOBS:
    t0 = time.time()
    print(f"[start] M={m} seed0 ttc+nac -> {outdir}", flush=True)
    r = subprocess.run([sys.executable, os.path.join("scripts", "adaptive_attack_eot.py"),
                        "--eot_inits", str(m), "--seeds", "0", "--methods", "ttc", "nac",
                        "--outdir", outdir],
                       capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    print(f"[done ] M={m} rc={r.returncode} {dt:.1f} min", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1200:], flush=True)
        sys.exit(1)
print("[trend scan done]", flush=True)
