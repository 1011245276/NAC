# -*- coding: utf-8 -*-
"""E1 fixed-budget verdict script (auditable; round-9 ruling).

Reads the 8 e4_results.json files from results/e1_fixed_budget/, computes
Wilson 95% CIs, and applies the PRE-REGISTERED decision rules exactly as
locked in thesis_proposal/研究二实验计划.md (commit a479aff, mtime
2026-10-28 14:03:28, predating the run):

  R1: robustness(M_att=10, steps=1) within the Wilson 95% CI of
      robustness(M_att=1, steps=10)              -> attribution stays retracted
  R2: robustness(10,1) below that CI lower bound
      by more than 0.5pp                          -> attribution restorable
  R3: otherwise                                   -> inconclusive, neutral

NOTE: an earlier inline verdict script had an R1 condition bug
(``r10 >= lo1`` instead of the two-sided within-CI test); it was caught
before reporting and the pre-registered text governed.  This script is the
authoritative implementation.
"""
import json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "e1_fixed_budget")
GRID = [(1, 10), (2, 5), (5, 2), (10, 1)]
METHODS = ["ttc", "nac"]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # bounds in PERCENT units; cap at 100, not 1 (a previous version capped
    # at 1.0 in percent units, clamping every upper bound above 1.00%)
    return (max(0.0, (c - hw) * 100), min(100.0, (c + hw) * 100))


def load(method, m_att, steps):
    p = os.path.join(BASE, f"{method}_M{m_att}_S{steps}_seed0", "e4_results.json")
    d = json.load(open(p))
    agg = d["results"][f"{method}_seed0_Mdef1"]["aggregations"]["M1_mean"]
    k, n = agg["k"], agg["n"]
    return k, n, agg["robust"], wilson(k, n)


def main():
    out = {}
    for method in METHODS:
        for m_att, steps in GRID:
            k, n, r, ci = load(method, m_att, steps)
            out[(method, m_att, steps)] = (k, n, r, ci)
            print(f"{method} M{m_att}xS{steps}: {k}/{n} = {r:.2f}% "
                  f"Wilson95 ({ci[0]:.2f}, {ci[1]:.2f})")
    print()
    for method in METHODS:
        _, _, r1, ci1 = out[(method, 1, 10)]
        _, _, r10, ci10 = out[(method, 10, 1)]
        if ci1[0] <= r10 <= ci1[1]:
            verdict = ("R1: (10,1) within the (1,10) Wilson CI -> "
                       "attribution stays retracted")
        elif r10 < ci1[0] - 0.5:
            verdict = ("R2: (10,1) below the (1,10) CI lower bound by >0.5pp -> "
                       "attribution restorable")
        else:
            verdict = ("R3: otherwise (observed: (10,1) ABOVE the CI upper "
                       "bound) -> inconclusive, neutral wording retained")
        print(f"{method}: M1xS10={r1:.2f}% CI({ci1[0]:.2f},{ci1[1]:.2f}) | "
              f"M10xS1={r10:.2f}% CI({ci10[0]:.2f},{ci10[1]:.2f}) -> {verdict}")


if __name__ == "__main__":
    main()
