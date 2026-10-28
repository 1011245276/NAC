# -*- coding: utf-8 -*-
"""E6 cost-benefit figure: two panels, x-axes NOT merged (round-10 ruling).

Panel A (defense side, main): x = deployment M_def in {1,2,5,10}, y = robust
  rate (%), attack fixed at M_att=5 draws x 10 steps (the E5 grid).  Right
  axis = measured deployment evaluation latency per 1024 samples (from
  deployment_eval_seconds recorded in the run JSONs).  Wilson 95% error
  bars; gray floor band < 1%.
Panel B (attack side): x = attack M_att in {1,2,5,10}, y = robust rate (%),
  deployment fixed at M_def=1 (the E1 fixed-budget grid); every point is
  annotated with its step count; total budget M_att x steps = 10 (constant)
  -- a DIFFERENT caliber from Panel A, stated in the caption.

Data discipline: every number is read from results/e5_grid/*/e4_results.json
and results/e1_fixed_budget/*/e4_results.json at run time; nothing is
hardcoded.  Robust counts come from the per-sample vectors (k of n=1024),
Wilson intervals computed here.  Aggregation = mean logits (majority is
identical in all 16 configs -- asserted).  Fonts Times/STIX, Okabe-Ito
palette, no in-figure titles.
"""
import os, glob, json, math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUTDIR = os.path.normpath(os.path.join(ROOT, '..', 'paper_backup',
                                       'figures_pending'))
Z95 = 1.959963984540054
TTC_C, NAC_C = '#0072B2', '#D55E00'          # Okabe-Ito
GRAY = '#B0B0B0'


def wilson(k, n):
    p = k / n
    denom = 1 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denom
    half = (Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n))
            / denom)
    lo = max(0.0, center - half)
    return 100 * p, 100 * lo, 100 * (center + half)


def load(path, res_key, agg_key):
    with open(path) as f:
        j = json.load(f)
    res = j['results'][res_key]
    agg = res['aggregations'][agg_key]
    maj = res['aggregations'][agg_key.replace('_mean', '_majority')]
    assert abs(agg['robust'] - maj['robust']) < 1e-9, \
        f'mean/majority differ in {path}: {agg["robust"]} vs {maj["robust"]}'
    vec = agg['per_sample_defended_correct']
    k = sum(1 for v in vec if v)
    n = len(vec)
    assert n == agg['n'], f'n mismatch in {path}'
    pct, lo, hi = wilson(k, n)
    assert abs(pct - agg['robust']) < 0.02, \
        f'robust mismatch in {path}: {pct} vs {agg["robust"]}'
    return dict(pct=pct, lo=lo, hi=hi, k=k, n=n,
                latency=res.get('deployment_eval_seconds'))


def panelA():
    out = {}
    for m in ('ttc', 'nac'):
        pts = []
        for md in (1, 2, 5, 10):
            d = os.path.join(ROOT, 'results', 'e5_grid',
                             f'{m}_Mdef{md}_seed0', 'e4_results.json')
            r = load(d, f'{m}_seed0_Mdef{md}', f'M{md}_mean')
            ac = json.load(open(d))['results'][f'{m}_seed0_Mdef{md}'][
                'attack_config']
            assert ac['m_att'] == 5 and ac['steps'] == 10, \
                f'panel A caliber broken in {d}: {ac}'
            pts.append(r)
        out[m] = pts
    return out


def panelB():
    out = {}
    for m in ('ttc', 'nac'):
        pts = []
        for ma, st in ((1, 10), (2, 5), (5, 2), (10, 1)):
            d = os.path.join(ROOT, 'results', 'e1_fixed_budget',
                             f'{m}_M{ma}_S{st}_seed0', 'e4_results.json')
            r = load(d, f'{m}_seed0_Mdef1', 'M1_mean')
            ac = json.load(open(d))['results'][f'{m}_seed0_Mdef1'][
                'attack_config']
            assert ac['m_att'] == ma and ac['steps'] == st, \
                f'panel B caliber broken in {d}: {ac}'
            assert ac.get('m_def_target', 1) == 1
            pts.append(r)
        out[m] = pts
    return out


def main():
    A, B = panelA(), panelB()
    xs = [1, 2, 5, 10]
    xpos = [0, 1, 2, 3]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8.5,
        'axes.linewidth': 0.6,
    })
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 2.9), sharey=True,
        gridspec_kw=dict(wspace=0.26))

    # ---------- Panel A ----------
    lat = {m: [p['latency'] for p in A[m]] for m in ('ttc', 'nac')}
    lo_band = [min(lat['ttc'][i], lat['nac'][i]) for i in range(4)]
    hi_band = [max(lat['ttc'][i], lat['nac'][i]) for i in range(4)]
    ax1b = ax1.twinx()
    ax1b.fill_between(xpos, lo_band, hi_band, color=GRAY, alpha=0.28,
                      linewidth=0)
    ax1b.plot(xpos, [(a + b) / 2 for a, b in zip(lo_band, hi_band)],
              color='#707070', linewidth=0.9, linestyle='--', marker='s',
              markersize=2.5, label='eval latency (right)')
    ax1b.set_ylabel('eval latency per 1024 samples (s)',
                    fontsize=8, labelpad=4)
    ax1b.set_ylim(0, 210)
    ax1b.tick_params(labelsize=7, width=0.6, pad=2)
    ax1b.annotate(f"{lat['ttc'][0]:.1f} s", (xpos[0], lo_band[0]),
                  textcoords='offset points', xytext=(5, 4),
                  fontsize=7, color='#505050')
    ax1b.annotate(f"{lo_band[3]:.1f}\u2013{hi_band[3]:.1f} s",
                  (xpos[3], hi_band[3]), textcoords='offset points',
                  xytext=(-6, 6), fontsize=7, color='#505050', ha='right')

    ax1.axhspan(0, 1.0, color=GRAY, alpha=0.22, linewidth=0)
    ax1.text(xpos[1], 1.35, 'floor < 1%', fontsize=7, color='#606060',
             ha='center')
    for m, c, mk in (('ttc', TTC_C, 'o'), ('nac', NAC_C, '^')):
        ys = [p['pct'] for p in A[m]]
        ylo = [p['pct'] - p['lo'] for p in A[m]]
        yhi = [p['hi'] - p['pct'] for p in A[m]]
        ax1.errorbar(xpos, ys, yerr=[ylo, yhi], color=c, marker=mk,
                     markersize=3.5, linewidth=1.2, capsize=2.5,
                     elinewidth=0.8, label=m.upper())
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(['1', '2', '5', '10'])
    ax1.set_xlabel('deployment $M_{def}$  (attack $M_{att}$=5, steps=10)')
    ax1.set_ylabel('robust accuracy (%)')
    ax1.set_ylim(0, 36)
    ax1.text(0.02, 0.96, 'budget $M_{att}\\times$steps = 50',
             transform=ax1.transAxes, fontsize=7, va='top', color='#404040')
    ax1.legend(fontsize=7.5, loc='center left', frameon=False)

    # ---------- Panel B ----------
    ax2.axhspan(0, 1.0, color=GRAY, alpha=0.22, linewidth=0)
    for m, c, mk in (('ttc', TTC_C, 'o'), ('nac', NAC_C, '^')):
        ys = [p['pct'] for p in B[m]]
        ylo = [p['pct'] - p['lo'] for p in B[m]]
        yhi = [p['hi'] - p['pct'] for p in B[m]]
        ax2.errorbar(xpos, ys, yerr=[ylo, yhi], color=c, marker=mk,
                     markersize=3.5, linewidth=1.2, capsize=2.5,
                     elinewidth=0.8)
        for i, p in enumerate(B[m]):
            if i == 3:
                ax2.annotate(f"S{[10, 5, 2, 1][i]}", (xpos[i], p['hi']),
                             textcoords='offset points', xytext=(-4, 3),
                             fontsize=6.5, color=c, ha='right')
            else:
                ax2.annotate(f"S{[10, 5, 2, 1][i]}", (xpos[i], p['hi']),
                             textcoords='offset points', xytext=(4, 3),
                             fontsize=6.5, color=c)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(['1', '2', '5', '10'])
    ax2.set_xlabel('attack $M_{att}$  (deployment $M_{def}$=1; '
                   'steps = 10/5/2/1)')
    ax2.text(0.02, 0.96, 'budget $M_{att}\\times$steps = 10 (constant)',
             transform=ax2.transAxes, fontsize=7, va='top', color='#404040')

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, width=0.6)
        ax.grid(axis='y', linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)

    os.makedirs(OUTDIR, exist_ok=True)
    pdf = os.path.join(OUTDIR, 'fig_e6_cost_benefit.pdf')
    png = os.path.join(OUTDIR, 'fig_e6_cost_benefit_preview.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=220, bbox_inches='tight')

    print('== Panel A (E5 grid, attack M_att=5 x steps=10) ==')
    for m in ('ttc', 'nac'):
        print(m, [(f"M{xs[i]}", f"{A[m][i]['pct']:.2f}% "
                   f"(k={A[m][i]['k']}/{A[m][i]['n']}, "
                   f"CI {A[m][i]['lo']:.2f}-{A[m][i]['hi']:.2f})", 
                   f"{A[m][i]['latency']}s") for i in range(4)])
    print('== Panel B (E1 fixed budget, deployment M_def=1) ==')
    for m in ('ttc', 'nac'):
        print(m, [(f"M{[1,2,5,10][i]}S{[10,5,2,1][i]}",
                   f"{B[m][i]['pct']:.2f}% (k={B[m][i]['k']}/{B[m][i]['n']},"
                   f" CI {B[m][i]['lo']:.2f}-{B[m][i]['hi']:.2f})")
                  for i in range(4)])
    print('written:', pdf)
    print('written:', png)


if __name__ == '__main__':
    main()
