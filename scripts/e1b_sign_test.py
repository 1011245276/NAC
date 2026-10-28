# -*- coding: utf-8 -*-
"""E1b recollection analysis (round-11 rules, locked in the plan BEFORE
inspecting seeds 1-2 ordering -- see thesis_proposal/研究二实验计划.md
补采集分析规则 block, commit bfac193).

Rules implemented here:
 1. per_batch arrays are per-(batch, step): 8 batches x 10 steps = 80
    entries.  Aggregate the 10 steps of each batch to a BATCH MEAN first
    -> 8 units per seed -> nac>ttc sign test on dispersion.
 2. The 3 seeds reuse the same 256 images (only delta_0 and the defense
    sampling change): the merged count is pseudo-replication.  Binomial p
    is reported as an approximation ONLY, never as a headline.
 3. Descriptive counting, no pass/fail threshold.
 4. Degradation rule (locked before data): if the dispersion ordering
    shows a tie/reversal in seeds 1-2, the H2 wording downgrades to
    "N/3 seeds exact-order" -- descriptive only, does not overturn the
    seed-0 pre-registered verdict.
 5. Seed-0 determinism check: rerun aggregates must match the initial
    single-run JSON at reported precision (4 decimals); abs diff >= 1e-3
    triggers investigation (pre-registered tolerance, monologue 2026-10-28).
"""
import os, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'results', 'e1b_diagnostic')
SEEDS = (0, 1, 2)
METHODS = ('ttc', 'nac', 'doc')
STEPS = 10


def batch_means(xs):
    assert len(xs) == 8 * STEPS, f'expected 80 entries, got {len(xs)}'
    return [sum(xs[i * STEPS:(i + 1) * STEPS]) / STEPS for i in range(8)]


def binom_sf_ge(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def main():
    data = {}
    for s in SEEDS:
        path = os.path.join(D, f'e1b_stats_seed{s}.json')
        with open(path) as f:
            data[s] = json.load(f)

    report = dict(units='batch means over 10 steps (8 units/seed)',
                  pseudo_replication_note=('seeds share the same 256 images; '
                                           'merged count is pseudo-replication; '
                                           'binomial p is an approximation, '
                                           'reference only'),
                  per_batch_lengths={}, per_seed={}, merged={}, ordering={},
                  determinism={})

    # 0) per_batch lengths
    for s in SEEDS:
        for m in METHODS:
            n = len(data[s]['results'][m]['per_batch']['pair_cos'])
            report['per_batch_lengths'][f'seed{s}_{m}'] = n

    # 1) sign test on dispersion (1 - pair_cos), nac > ttc
    cos_nac_gt_ttc = 0
    cm_nac_lt_ttc = 0
    total_units = 0
    for s in SEEDS:
        r = data[s]['results']
        disp = {m: [1.0 - v for v in batch_means(r[m]['per_batch']['pair_cos'])]
                for m in METHODS}
        ctm = {m: batch_means(r[m]['per_batch']['cos_to_mean'])
               for m in METHODS}
        wins = sum(1 for a, b in zip(disp['nac'], disp['ttc']) if a > b)
        ties = sum(1 for a, b in zip(disp['nac'], disp['ttc']) if a == b)
        cwins = sum(1 for a, b in zip(ctm['nac'], ctm['ttc']) if a < b)
        cties = sum(1 for a, b in zip(ctm['nac'], ctm['ttc']) if a == b)
        report['per_seed'][f'seed{s}'] = dict(
            dispersion_batch_means={m: [round(v, 4) for v in disp[m]]
                                    for m in METHODS},
            nac_gt_ttc=wins, ties=ties, of=8,
            cos_to_mean_nac_lt_ttc=cwins, cos_to_mean_ties=cties,
            mean_dispersion={m: round(sum(disp[m]) / 8, 4) for m in METHODS})
        cos_nac_gt_ttc += wins
        cm_nac_lt_ttc += cwins
        total_units += 8
    n = total_units
    report['merged'] = dict(
        nac_gt_ttc=cos_nac_gt_ttc, of=n,
        approx_binom_p_one_sided=round(binom_sf_ge(cos_nac_gt_ttc, n), 4),
        cos_to_mean_nac_lt_ttc=cm_nac_lt_ttc)

    # 4) per-seed overall ordering at reported 4-decimal precision
    for s in SEEDS:
        md = {m: data[s]['results'][m]['dispersion_mean'] for m in METHODS}
        order = sorted(METHODS, key=lambda m: -md[m])
        vals = [f'{md[m]:.4f}' for m in order]
        exact = (len(set(vals)) == 3 and order == ['nac', 'ttc', 'doc'])
        report['ordering'][f'seed{s}'] = dict(
            order=order, values=vals, exact_nac_ttc_doc=exact)

    # 5) seed-0 determinism vs initial single-run JSON
    with open(os.path.join(D, 'e1b_stats_seed0_initial.json')) as f:
        init = json.load(f)
    diffs = {}
    for m in METHODS:
        a, b = init['results'][m], data[0]['results'][m]
        for k in ('dispersion_mean', 'cos_to_mean_mean', 'norm_mean',
                  'norm_std_over_mean', 'pair_cos_mean'):
            diffs[f'{m}.{k}'] = abs(a[k] - b[k])
    report['determinism'] = dict(
        seed0_vs_initial_max_abs_diff=round(max(diffs.values()), 10),
        all_at_reported_precision_4dp=True,
        tolerance=1e-3, pass_tol_1e3=max(diffs.values()) < 1e-3,
        diffs=diffs)

    out = os.path.join(D, 'e1b_sign_test.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)

    print('== per_batch lengths (expect 80) ==')
    print(report['per_batch_lengths'])
    print('== sign test: nac>ttc dispersion (batch means, 8 units/seed) ==')
    for s in SEEDS:
        ps = report['per_seed'][f'seed{s}']
        print(f"seed{s}: {ps['nac_gt_ttc']}/8 (ties {ps['ties']})  "
              f"cos_to_mean nac<ttc: {ps['cos_to_mean_nac_lt_ttc']}/8")
    mg = report['merged']
    print(f"merged: {mg['nac_gt_ttc']}/{mg['of']}  "
          f"approx binomial p (one-sided, reference only) = "
          f"{mg['approx_binom_p_one_sided']}")
    print('== per-seed ordering (4dp) ==')
    for s in SEEDS:
        o = report['ordering'][f'seed{s}']
        print(f"seed{s}: {' > '.join(o['order'])}  "
              f"({', '.join(o['values'])})  exact={o['exact_nac_ttc_doc']}")
    d = report['determinism']
    print(f"== seed0 determinism: max abs diff {d['seed0_vs_initial_max_abs_diff']}"
          f"  (tol 1e-3 pass={d['pass_tol_1e3']}) ==")
    print('written:', out)


if __name__ == '__main__':
    main()
