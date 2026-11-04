# -*- coding: utf-8 -*-
"""E1 收尾（预登记 2026-11-05，thesis_proposal/研究二实验计划.md）:
(a) DOC x M in {1,2} x seed 0; (b) {ttc,nac,doc} x M in {1,2} x seeds 1-2.
Skip-completed discipline: a job is skipped iff its eot_results.json exists
and every expected per-(method,seed) key is present with per_sample.n == 10000."""
import subprocess, sys, time, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

M_METHODS = ["ttc", "nac", "doc"]


def job_complete(outdir, methods, seeds, n_expected=10000):
    jp = os.path.join(outdir, "eot_results.json")
    if not os.path.exists(jp):
        return False
    try:
        with open(jp, "r", encoding="utf-8") as f:
            res = json.load(f)["results"]
    except Exception:
        return False
    for s in seeds:
        for m in methods:
            key = f"{m}_seed{s}"
            if key not in res:
                return False
            if res[key].get("per_sample", {}).get("n") != n_expected:
                return False
    return True


JOBS = [
    # (tag, eot_inits, seeds, methods, outdir)
    ("m1_doc_seed0", 1, [0], ["doc"], "results/research2_eot_mscan/m1/trend_seed0_doc"),
    ("m2_doc_seed0", 2, [0], ["doc"], "results/research2_eot_mscan/m2/trend_seed0_doc"),
    ("m1_seed12", 1, [1, 2], M_METHODS, "results/research2_eot_mscan/m1/trend_seed12"),
    ("m2_seed12", 2, [1, 2], M_METHODS, "results/research2_eot_mscan/m2/trend_seed12"),
]

for tag, m, seeds, methods, outdir in JOBS:
    if job_complete(outdir, methods, seeds):
        print(f"[skip] {tag}: complete", flush=True)
        continue
    t0 = time.time()
    print(f"[start] {tag}: M={m} seeds={seeds} methods={methods} -> {outdir}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run(
        [sys.executable, os.path.join("scripts", "adaptive_attack_eot.py"),
         "--eot_inits", str(m), "--seeds", *[str(s) for s in seeds],
         "--methods", *methods, "--outdir", outdir],
        capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    print(f"[done ] {tag}: rc={r.returncode} {dt:.1f} min", flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        print("[S1] rc!=0: fix and rerun this unit; aborting queue to avoid silent gaps", flush=True)
        sys.exit(1)
    if not job_complete(outdir, methods, seeds):
        print(f"[S2] {tag}: per-sample records incomplete after run", flush=True)
        sys.exit(1)
print("[E1 fill-in done]", flush=True)
