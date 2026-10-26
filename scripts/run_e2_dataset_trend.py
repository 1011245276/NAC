# -*- coding: utf-8 -*-
"""E2 dataset-extension trend: {STL10, dtd} x M=5 x {ttc,nac} x 256 samples.
Sequential (single-GPU OOM discipline); --max_batches 8 = 256 samples."""
import subprocess, sys, time, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATASETS = [("STL10", "stl10"), ("dtd", "dtd"),
            ("flowers102", "flowers102"), ("EuroSAT", "eurosat")]
for dskey, tag in DATASETS:
    outdir = os.path.join("results", "e2_dataset_trend", tag, "m5_trend_seed0")
    if os.path.exists(os.path.join(outdir, "eot_results.json")):
        print(f"[skip] {tag} already done", flush=True)
        continue
    t0 = time.time()
    print(f"[start] {tag} M=5 ttc+nac 256 samples -> {outdir}", flush=True)
    r = subprocess.run([sys.executable, os.path.join("scripts", "adaptive_attack_eot.py"),
                        "--eot_inits", "5", "--seeds", "0", "--methods", "ttc", "nac",
                        "--dataset", dskey, "--max_batches", "8", "--outdir", outdir],
                       capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    print(f"[done ] {tag} rc={r.returncode} {dt:.1f} min", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1200:], flush=True)
print("[E2 trend done]", flush=True)
