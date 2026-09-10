#!/usr/bin/env python3
"""
Aggregate all experiment results from the nested log layout:

    results/<exp>/pgd_eps_<eps>/<method>/seed_<N>.log

Each log line looks like:
    2026-... - INFO - ImageNet: clean=67.82 | adv=0.00 | adv+nac_m0.9=51.40 | gain=51.40

A single ``seed_*.log`` file may contain MULTIPLE runs appended over time with
DIFFERENT configurations (e.g. different ``ttc_numsteps``/``test_numsteps``).
This script splits each log into runs delimited by their ``Namespace(...)``
header and groups results by the actual experiment configuration, so runs with
different hyperparameters are never averaged together as if they were seeds.

Usage:
    python scripts/aggregate_all.py [--out aggregated_results]

Outputs:
    results/aggregated_all.json   (per method/dataset/config, mean ± std across seeds)
    results/aggregated_all.md     (Markdown table)
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

# Matches the per-dataset result line (skip the "SUMMARY:" lines).
LINE_RE = re.compile(
    r'INFO - ([^:]+):\s*clean=([\d.]+)\s*\|\s*adv=([\d.]+)\s*\|\s*adv\+([^=]+?)=([\d.]+)'
)

# Fields that identify an experiment configuration.  Kept small so that noise
# fields (root/outdir paths, seed, num_workers) do not spuriously split configs.
CONFIG_FIELDS = [
    'test_eps', 'test_numsteps', 'test_stepsize',
    'ttc_eps', 'ttc_numsteps', 'ttc_stepsize',
    'counterattack', 'nac_momentum',
    'DOC_eps', 'DOC_numsteps', 'DOC_stepsize', 'learnable_tau', 'temperature',
    'tau_thres', 'beta', 'batch_size',
]


def parse_namespace(line):
    """Parse the key=value pairs inside a Namespace(...) header line."""
    body = line.split('Namespace(', 1)[1].rsplit(')', 1)[0]
    cfg = {}
    for tok in body.split(','):
        if '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        cfg[k.strip()] = v.strip().strip("'")
    return cfg


def config_signature(cfg):
    """A stable string identifying one configuration, for grouping."""
    parts = [f'{f}={cfg[f]}' for f in CONFIG_FIELDS if f in cfg]
    return '; '.join(parts)


def short_label(cfg):
    """A compact human-readable config label for the Markdown table."""
    k = cfg.get('ttc_numsteps', '?')
    atk = cfg.get('test_numsteps', '?')
    bs = cfg.get('batch_size', '?')
    if cfg.get('counterattack') == 'nac':
        meth = "nac m={}".format(cfg.get('nac_momentum', '?'))
    else:
        meth = cfg.get('counterattack', '?')
    return 'K={} atk={} bs={} {}'.format(k, atk, bs, meth)


def parse_log(path):
    """Return list of (config_dict, [(dataset, clean, adv, defended), ...]).

    Each run begins with a ``Namespace(...)`` line; result lines are attributed
    to the most recent config until the next Namespace header.
    """
    runs = []
    cur_cfg = None
    cur_results = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Namespace(' in line:
                if cur_cfg is not None and cur_results:
                    runs.append((cur_cfg, cur_results))
                cur_cfg = parse_namespace(line)
                cur_results = []
                continue
            m = LINE_RE.search(line)
            if m and 'SUMMARY' not in line:
                dataset, clean, adv, tag, defended = m.groups()
                cur_results.append(
                    (dataset.strip(), float(clean), float(adv), float(defended))
                )
        if cur_cfg is not None and cur_results:
            runs.append((cur_cfg, cur_results))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = args.root.rstrip('/\\')
    pattern = os.path.join(root, '*', 'pgd_eps_*', '*', 'seed_*.log')

    # (exp, method, cfgkey) -> dataset -> {seed: [defended values]}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # (exp, method, cfgkey) -> (cfg_dict, label)
    configs = {}
    # (exp, method, cfgkey) -> dataset -> (clean, adv)   (first seen wins)
    clean_adv = defaultdict(lambda: defaultdict(lambda: (0.0, 0.0)))

    for path in sorted(glob.glob(pattern, recursive=True)):
        parts = path.replace('\\', '/').split('/')
        exp = parts[-4]
        method = parts[-2]
        seed = os.path.basename(path).replace('seed_', '').replace('.log', '')
        for cfg, results in parse_log(path):
            cfgkey = config_signature(cfg)
            configs[(exp, method, cfgkey)] = (cfg, short_label(cfg))
            for dataset, clean, adv, defended in results:
                agg[(exp, method, cfgkey)][dataset].setdefault(seed, []).append(defended)
                if (exp, method, cfgkey) not in clean_adv or \
                        dataset not in clean_adv[(exp, method, cfgkey)]:
                    clean_adv[(exp, method, cfgkey)][dataset] = (clean, adv)

    out_json = {}
    rows = []
    for (exp, method, cfgkey) in sorted(agg):
        cfg, label = configs[(exp, method, cfgkey)]
        out_json.setdefault(exp, {})[method] = out_json.setdefault(exp, {}).setdefault(method, {})
        for dataset in sorted(agg[(exp, method, cfgkey)]):
            by_seed = agg[(exp, method, cfgkey)][dataset]
            # collapse re-runs within a single seed first, then across seeds
            seed_means = [statistics.mean(v) for v in by_seed.values()]
            clean, adv = clean_adv[(exp, method, cfgkey)][dataset]
            def_mean = statistics.mean(seed_means)
            def_std = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
            entry = {
                'n_seeds': len(seed_means),
                'clean': round(clean, 2),
                'adv': round(adv, 2),
                'defended_mean': round(def_mean, 2),
                'defended_std': round(def_std, 2),
                'gain_mean': round(def_mean - adv, 2),
                'config': cfg,
            }
            out_json[exp][method][dataset] = entry
            rows.append((exp, method, label, dataset, len(seed_means),
                         clean, adv, def_mean, round(def_std, 2), round(def_mean - adv, 2)))

    # Write JSON
    out = args.out or os.path.join(root, 'aggregated_all.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f'[OK] JSON: {out}')

    # Write Markdown
    md_path = out.replace('.json', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Aggregated Results\n\n')
        cur_exp = None
        for exp, method, label, dataset, n, clean, adv, dm, ds, gain in rows:
            if exp != cur_exp:
                cur_exp = exp
                f.write(f'\n## {exp}\n\n')
                f.write('| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |\n')
                f.write('|--------|--------|---------|-------|-------|-----|----------|------|\n')
            std_s = f' ± {ds:.2f}' if ds else ''
            f.write(f'| {method} | {label} | {dataset} | {n} | {clean:.2f} | {adv:.2f} | {dm:.2f}{std_s} | {gain:+.2f} |\n')
    print(f'[OK] Markdown: {md_path}')


if __name__ == '__main__':
    main()
