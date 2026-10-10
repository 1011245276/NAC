"""
Phase-2 pilot analysis: which signals predict per-sample benefit of
deeper counterattack (K=2 -> K=4) at eps=4/255?

Reads results/adaptive_pilot/pilot_data.npz (produced by adaptive_signal_pilot.py,
equivalence-verified instrumented NAC, 10000 CIFAR-10 test samples).

Outputs (all auto-generated, no manual copying):
  results/adaptive_pilot/pilot_analysis.json   -- full statistics
  results/adaptive_pilot/pilot_analysis.md     -- human-readable tables
"""
from __future__ import print_function

import os
import sys
import json

import numpy as np

_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot')

EPS_TAGS = {0: '1/255', 1: '2/255', 2: '4/255'}


def pearson(x, y):
    x = x.astype(np.float64); y = y.astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else np.nan


def spearman(x, y):
    def rank(v):
        """Midrank (average ranks for ties); NaN for non-finite entries."""
        r = np.full(v.shape, np.nan)
        m = np.isfinite(v)
        v_f = v[m]
        order = np.argsort(v_f, kind='mergesort')
        srt = v_f[order]
        ranks = np.empty(len(v_f))
        i = 0
        while i < len(srt):
            j = i
            while j + 1 < len(srt) and srt[j + 1] == srt[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        r[m] = ranks
        return r
    return pearson(rank(x), rank(y))


def auroc(score, pos, neg):
    """AUROC of `score` ranking pos (label 1) above neg (label 0)."""
    s = np.concatenate([score[pos], score[neg]])
    m = np.isfinite(s)
    if m.sum() < 2 or (~m).any():
        # drop non-finite samples from both groups consistently
        keep = np.isfinite(score)
        pos, neg = pos & keep, neg & keep
        s = np.concatenate([score[pos], score[neg]])
        if len(s) < 2 or pos.sum() == 0 or neg.sum() == 0:
            return np.nan
    lab = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    d = np.load(os.path.join(OUT, 'pilot_data.npz'))
    n = len(d['clean_correct'])
    rep = {'n_samples': int(n),
           'clean_acc': float(d['clean_correct'].mean() * 100),
           'adv_acc': float(d['adv_correct'].mean() * 100)}

    # ---------------- 1. grid accuracy table ----------------
    grid = {}
    for e in range(3):
        for K in range(1, 5):
            k = 'nac_K%d_eps%d__correct' % (K, e)
            grid['nac_K%d_eps%d' % (K, e)] = float(d[k].mean() * 100)
    grid['ttc_K2_eps2'] = float(d['ttc_K2_eps2__correct'].mean() * 100)
    rep['grid_accuracy'] = grid

    # ---------------- 2. benefit labels at eps=4/255 ----------------
    c2 = d['nac_K2_eps2__correct'].astype(np.int8)
    c3 = d['nac_K3_eps2__correct'].astype(np.int8)
    c4 = d['nac_K4_eps2__correct'].astype(np.int8)
    ben24 = (c4 - c2)          # {-1,0,+1}: benefit of extending K2 -> K4
    ben23 = (c3 - c2)
    m24 = d['nac_K4_eps2__margin'].astype(np.float64)
    m22 = d['nac_K2_eps2__margin'].astype(np.float64)
    mgn24 = m24 - m22          # continuous margin gain K2 -> K4
    pos = ben24 == 1           # incorrect@K2 -> correct@K4  (recoverable)
    neg = ben24 == -1          # correct@K2 -> wrong@K4      (regression risk)
    rep['benefit_counts_eps4'] = {
        'K2_acc': float(c2.mean() * 100), 'K3_acc': float(c3.mean() * 100),
        'K4_acc': float(c4.mean() * 100),
        'recoverable_K2toK4': int(pos.sum()), 'regression_K2toK4': int(neg.sum()),
        'recoverable_K2toK3': int((ben23 == 1).sum()),
        'regression_K2toK3': int((ben23 == -1).sum())}

    # ---------------- 3. signal ranking (signals from cheap K=2 prefix) -------
    SIGNALS = ['diff_ratio', 'grad_norm0', 'grad_norm_mean', 'cos_atkdelta_g0',
               'grad_cos_succ', 'step_align', 'velocity_cos', 'emb_jump',
               'gain_rate', 'loss_final', 'lookahead_pred', 'disp_growth']
    sig_rank = {}
    for src in ('nac_K2_eps2', 'nac_K1_eps2'):   # K2 = realistic, K1 = cheapest
        for s in SIGNALS:
            key = '%s__sig_%s' % (src, s)
            if key not in d.files:
                continue
            x = d[key].astype(np.float64)
            if np.isfinite(x).sum() < n * 0.5:
                continue
            r = {
                'pearson_vs_ben24': pearson(x, ben24),
                'spearman_vs_ben24': spearman(x, ben24),
                'spearman_vs_margin_gain': spearman(x, mgn24),
                'auroc_recover_vs_rest': auroc(x, pos, (ben24 == 0) | neg),
                'auroc_regression': auroc(x, neg, (ben24 == 0) | pos),
                'nan_frac': float((~np.isfinite(x)).mean()),
            }
            sig_rank['%s:%s' % (src.replace('nac_', ''), s)] = r
    # rank by |spearman|
    rep['signal_ranking'] = dict(
        sorted(sig_rank.items(),
               key=lambda kv: -(abs(kv[1]['spearman_vs_ben24'])
                                if np.isfinite(kv[1]['spearman_vs_ben24']) else 0)))

    # ---------------- 4. quartile groups for top signals ----------------
    def quartile_table(x, name):
        q = np.quantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
        g = np.digitize(x, q)
        tab = []
        for i in range(4):
            m = g == i
            if m.sum() == 0:
                continue
            tab.append({'quartile': i + 1, 'n': int(m.sum()),
                        'K2_acc': float(c2[m].mean() * 100),
                        'K4_acc': float(c4[m].mean() * 100),
                        'gain_K4_K2': float((c4[m].mean() - c2[m].mean()) * 100),
                        'recover_rate': float((ben24[m] == 1).mean()),
                        'regress_rate': float((ben24[m] == -1).mean())})
        return {'name': name, 'cuts': q.tolist(), 'groups': tab}

    top_keys = list(rep['signal_ranking'].keys())[:5]
    rep['quartile_tables'] = {}
    for tk in top_keys:
        src, s = tk.split(':')
        key = 'nac_%s_eps2__sig_%s' % (src, s)
        if key in d.files:
            rep['quartile_tables'][tk] = quartile_table(d[key].astype(np.float64), tk)

    # ---------------- 5. adaptive-rule simulation (route K2 -> K4) ------------
    # rule: run K=2; if signal > tau, continue to K=4 (cost model: fwd-bwd passes)
    # cost: K=2 -> 2 units(+1 emb), K=4 continuation -> 4 units total.
    sim = []
    for src in ('K2', 'K1'):
        for s in SIGNALS:
            key = 'nac_%s_eps2__sig_%s' % (src, s)
            if key not in d.files:
                continue
            x = d[key].astype(np.float64)
            if np.isfinite(x).sum() < n * 0.5:
                continue
            qs = np.quantile(x[np.isfinite(x)], np.linspace(0.05, 0.95, 19))
            for q in qs:
                route = x > q
                # predicted correct: route ? c4 : c2  (continuation replaces)
                acc = np.where(route, c4, c2).mean() * 100
                avg_k = 2 + 2 * route.mean()
                sim.append({'signal': s, 'src': src, 'thr': float(q),
                            'route_frac': float(route.mean()),
                            'adaptive_acc': float(acc), 'avg_K': float(avg_k)})
    # fixed anchors
    anchors = {'fixed_K2_acc': float(c2.mean() * 100), 'fixed_K4_acc': float(c4.mean() * 100),
               'fixed_K2_cost': 2.0, 'fixed_K4_cost': 4.0}
    rep['anchors'] = anchors
    # Pareto: for each avg_K budget, best adaptive acc
    best = {}
    for r in sim:
        kbucket = round(r['avg_K'], 2)
        if kbucket not in best or r['adaptive_acc'] > best[kbucket]['adaptive_acc']:
            best[kbucket] = r
    rep['adaptive_pareto'] = dict(sorted(best.items()))
    # overall best by acc and best acc with avg_K <= 3.0 (budget-constrained)
    rep['best_overall'] = max(sim, key=lambda r: r['adaptive_acc'])
    bud = [r for r in sim if r['avg_K'] <= 3.0]
    rep['best_budget3'] = max(bud, key=lambda r: r['adaptive_acc']) if bud else None

    # ---------------- save ----------------
    with open(os.path.join(OUT, 'pilot_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=float)

    # markdown
    md = ['# Adaptive-signal pilot analysis (auto-generated)',
          '',
          '- samples: %d  clean: %.2f%%  adv(PGD-10,4/255): %.2f%%' %
          (rep['n_samples'], rep['clean_acc'], rep['adv_acc']),
          '', '## Grid accuracy (NAC, %%)', '',
          '| eps \\ K | 1 | 2 | 3 | 4 |', '|---|---|---|---|---|']
    for e in range(3):
        md.append('| %s | %s |' % (EPS_TAGS[e], ' | '.join(
            '%.2f' % grid['nac_K%d_eps%d' % (K, e)] for K in range(1, 5))))
    md += ['', 'TTC(K=2,4/255): %.2f%%   NAC(K=2,4/255): %.2f%%' %
           (grid['ttc_K2_eps2'], grid['nac_K2_eps2']),
           '', '## Benefit structure (eps=4/255)', '']
    bc = rep['benefit_counts_eps4']
    md += ['- K2 %.2f%% -> K3 %.2f%% -> K4 %.2f%%' % (bc['K2_acc'], bc['K3_acc'], bc['K4_acc']),
           '- recoverable K2->K4: %d   regression K2->K4: %d' %
           (bc['recoverable_K2toK4'], bc['regression_K2toK4']),
           '', '## Signal ranking (top 15 by |Spearman| vs benefit)', '',
           '| signal | spearman | spearman_mg | pearson | AUROC_recover | AUROC_regress |', '|---|---|---|---|---|---|']
    for k, v in list(rep['signal_ranking'].items())[:15]:
        md.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
            k, v['spearman_vs_ben24'], v['spearman_vs_margin_gain'],
            v['pearson_vs_ben24'],
            v['auroc_recover_vs_rest'], v['auroc_regression']))
    md += ['', '## Best adaptive rules', '']
    md.append('- best overall: %s (from %s prefix) thr=%.4f route=%.0f%% -> acc %.2f%% @ avgK %.2f' % (
        rep['best_overall']['signal'], rep['best_overall']['src'], rep['best_overall']['thr'],
        rep['best_overall']['route_frac'] * 100, rep['best_overall']['adaptive_acc'],
        rep['best_overall']['avg_K']))
    if rep['best_budget3']:
        b = rep['best_budget3']
        md.append('- best budget(avgK<=3): %s thr=%.4f route=%.0f%% -> acc %.2f%% @ avgK %.2f' % (
            b['signal'], b['thr'], b['route_frac'] * 100, b['adaptive_acc'], b['avg_K']))
    md.append('- anchors: fixed K2 %.2f%% @2fwd, fixed K4 %.2f%% @4fwd' %
              (anchors['fixed_K2_acc'], anchors['fixed_K4_acc']))
    with open(os.path.join(OUT, 'pilot_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

    print('[analyze] done. grid K4(4/255)=%.2f  recoverable=%d  best_budget3=%s' % (
        bc['K4_acc'], bc['recoverable_K2toK4'],
        rep['best_budget3']['signal'] if rep['best_budget3'] else 'n/a'))
    print('[analyze] wrote pilot_analysis.json / pilot_analysis.md')


if __name__ == '__main__':
    main()
