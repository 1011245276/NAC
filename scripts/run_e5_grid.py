# -*- coding: utf-8 -*-
"""E5 grid driver (crash-isolated): one subprocess per (method, M_def)
attack config, then one clean-eval pass via the verified S3 paired script.

Crash-avoidance design (per 2026-10-27 user approval):
  * process isolation -- a failure in one config cannot corrupt or block the
    others; completed configs are preserved on disk and skipped on rerun;
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True against fragmentation;
  * attack-step peak memory is bounded at ONE repetition (M_def branches) by
    gradient-linearity accumulation inside e4_aggregate_pipeline.py;
  * skip-completed discipline keyed on the per-config e4_results.json.
"""
import subprocess, sys, time, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

METHODS = ["ttc", "nac"]
M_DEFS = [1, 2, 5, 10]
SEED = 0
MAX_BATCHES = 32          # 32 x 32 = 1024 samples
M_ATT = 5
STEPS = 10

env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

t_all = time.time()
# ---- phase 1: aggregate-aware attack + robustness, one process per config ----
for m in METHODS:
    for md in M_DEFS:
        outdir = os.path.join("results", "e5_grid", f"{m}_Mdef{md}_seed0")
        marker = os.path.join(outdir, "e4_results.json")
        done = False
        if os.path.exists(marker):
            try:
                d = json.load(open(marker))
                done = f"{m}_seed0_Mdef{md}" in d.get("results", {})
            except Exception:
                done = False
        if done:
            print(f"[skip] {m} Mdef{md} already done", flush=True)
            continue
        t0 = time.time()
        print(f"[start] {m} Mdef{md} attack+robustness (1024 samples)",
              flush=True)
        r = subprocess.run(
            [sys.executable, os.path.join("scripts", "e4_aggregate_pipeline.py"),
             "--methods", m, "--m_def_list", str(md), "--seeds", str(SEED),
             "--m_att", str(M_ATT), "--steps", str(STEPS),
             "--max_batches", str(MAX_BATCHES), "--outdir", outdir],
            env=env, capture_output=True, text=True)
        dt = (time.time() - t0) / 60
        print(f"[done ] {m} Mdef{md} rc={r.returncode} {dt:.1f} min", flush=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True)

# ---- phase 2: clean column (1024 x M_eval=10, all M_def) ----
clean_out = os.path.join("results", "e5_grid", "clean_1024")
clean_marker = os.path.join(clean_out, "s3_paired_results.json")
if not os.path.exists(clean_marker):
    t0 = time.time()
    print("[start] clean column (1024 x M_eval=10 x M_def {1,2,5,10})",
          flush=True)
    r = subprocess.run(
        [sys.executable, os.path.join("scripts", "e4_s3_clean_paired.py"),
         "--methods", "ttc", "nac", "--seed", str(SEED),
         "--m_def_agg_list", "2", "5", "10",
         "--max_batches", str(MAX_BATCHES), "--outdir", clean_out],
        env=env, capture_output=True, text=True)
    print(f"[done ] clean rc={r.returncode} "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
else:
    print("[skip] clean column already done", flush=True)

print(f"[E5 grid complete] total {(time.time() - t_all) / 60:.1f} min",
      flush=True)
