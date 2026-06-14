import subprocess, sys, os, time
from datetime import datetime

os.chdir(r"F:\codex\NAC\nac_project")

BASE = [
    "python", "nac_fair_experiment.py",
    "--test_numsteps", "10",
    "--ttc_eps", "4", "--ttc_numsteps", "2",
    "--root", "F:/codex/NAC/data",
    "--batch_size", "64",
]

experiments = [
    # === DTD multiseed: NAC seeds 0,1,2 (eps=1) ===
    {"test_set": ["dtd"], "test_eps": "1", "counterattack": "nac", "seed": "0", "outdir": "./results/multiseed_dtd", "desc": "DTD eps=1 NAC seed 0"},
    {"test_set": ["dtd"], "test_eps": "1", "counterattack": "nac", "seed": "1", "outdir": "./results/multiseed_dtd", "desc": "DTD eps=1 NAC seed 1"},
    {"test_set": ["dtd"], "test_eps": "1", "counterattack": "nac", "seed": "2", "outdir": "./results/multiseed_dtd", "desc": "DTD eps=1 NAC seed 2"},
    {"test_set": ["dtd"], "test_eps": "1", "counterattack": "ttc", "seed": "0", "outdir": "./results/multiseed_dtd", "desc": "DTD eps=1 TTC seed 0"},
    {"test_set": ["dtd"], "test_eps": "1", "counterattack": "ttc", "seed": "1", "outdir": "./results/multiseed_dtd", "desc": "DTD eps=1 TTC seed 1"},
    {"test_set": ["dtd"], "test_eps": "1", "counterattack": "ttc", "seed": "2", "outdir": "./results/multiseed_dtd", "desc": "DTD eps=1 TTC seed 2"},
    # === ImageNet-100 multiseed: NAC seeds 0,1,2 (eps=1) ===
    {"test_set": ["ImageNet"], "test_eps": "1", "counterattack": "nac", "seed": "0", "outdir": "./results/multiseed_imagenet", "desc": "ImageNet eps=1 NAC seed 0"},
    {"test_set": ["ImageNet"], "test_eps": "1", "counterattack": "nac", "seed": "1", "outdir": "./results/multiseed_imagenet", "desc": "ImageNet eps=1 NAC seed 1"},
    {"test_set": ["ImageNet"], "test_eps": "1", "counterattack": "nac", "seed": "2", "outdir": "./results/multiseed_imagenet", "desc": "ImageNet eps=1 NAC seed 2"},
    {"test_set": ["ImageNet"], "test_eps": "1", "counterattack": "ttc", "seed": "0", "outdir": "./results/multiseed_imagenet", "desc": "ImageNet eps=1 TTC seed 0"},
    {"test_set": ["ImageNet"], "test_eps": "1", "counterattack": "ttc", "seed": "1", "outdir": "./results/multiseed_imagenet", "desc": "ImageNet eps=1 TTC seed 1"},
    {"test_set": ["ImageNet"], "test_eps": "1", "counterattack": "ttc", "seed": "2", "outdir": "./results/multiseed_imagenet", "desc": "ImageNet eps=1 TTC seed 2"},
    # === Standard momentum at eps=1 (CIFAR-10 + STL-10) ===
    {"test_set": ["cifar10", "STL10"], "test_eps": "1", "counterattack": "momentum", "seed": "0", "outdir": "./results/momentum_eps1", "desc": "Momentum eps=1 C10+STL10 seed 0"},
]

total = len(experiments)
logfile = "./results/batch_run2.log"
with open(logfile, "w", encoding="utf-8") as lf:
    lf.write(f"BATCH START: {datetime.now()}\n{total} experiments\n\n")

failures = []
for idx, exp in enumerate(experiments):
    cmd = BASE + ["--test_set"] + exp["test_set"] + [
        "--test_eps", exp["test_eps"],
        "--counterattack", exp["counterattack"],
        "--seed", exp["seed"],
        "--outdir", exp["outdir"],
    ]
    desc = exp["desc"]
    t0 = time.time()
    line = f"[{idx+1}/{total}] {desc}  {datetime.now().strftime('%H:%M:%S')}"
    print(line, flush=True)
    
    with open(logfile, "a", encoding="utf-8") as lf:
        lf.write(f"\n{line}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        result = None
        status = "TIMEOUT"
    else:
        status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    
    elapsed = time.time() - t0
    print(f"  -> {status}  {elapsed/60:.1f}min", flush=True)
    
    with open(logfile, "a", encoding="utf-8") as lf:
        lf.write(f"  -> {status}  {elapsed/60:.1f}min\n")
    
    # Save individual log
    log_dir = os.path.join(exp["outdir"], f"pgd_eps_{exp['test_eps']}", exp["counterattack"])
    os.makedirs(log_dir, exist_ok=True)
    ind_log = os.path.join(log_dir, f"seed_{exp['seed']}.log")
    with open(ind_log, "w", encoding="utf-8") as f:
        if result:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n" + result.stderr)
        else:
            f.write("TIMEOUT after 2 hours\n")
    
    if status != "OK":
        failures.append(desc)

print(f"\nDONE: {datetime.now().strftime('%H:%M:%S')}", flush=True)
print(f"Failures: {len(failures)}/{total}", flush=True)
for f in failures:
    print(f"  FAIL: {f}", flush=True)
with open(logfile, "a", encoding="utf-8") as lf:
    lf.write(f"\nDONE: {datetime.now()}\nFailures: {len(failures)}/{total}\n")
