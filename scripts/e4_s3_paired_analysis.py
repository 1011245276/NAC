# -*- coding: utf-8 -*-
"""
Paired analysis for the E4-S3 full clean validation (external ruling
2026-10-27).  Reads s3_paired_results.json (per-image, per-repetition
correctness vectors) and computes, per method and per aggregation:

  * pooled rates with Wilson 95% CI;
  * image-level paired drop: d_i = p_base,i - p_agg,i (i = image,
    p = fraction of the M_eval repetitions correct), net drop in pp;
  * image-level PAIRED standard error SE(d) = std(d_i)/sqrt(N) -- the
    corrected counterpart of the unpaired binomial SE quoted earlier;
  * repetition-pair discordant counts: a = #{base correct, agg wrong},
    b = #{base wrong, agg correct}, with McNemar exact test;
  * paired bootstrap over IMAGES (B=10000, fixed bootstrap seed):
    one-sided 95% upper bound of the drop; verdict
      PASS      if upper <= 0.5pp (pre-registered constraint 1);
      UNCERTAIN if the CI crosses 0.5pp;
      FAIL      if lower > 0.5pp.
The 0.5pp threshold is pre-registered and NOT adjusted here.
"""
import json, math, os, sys
import numpy as np

THRESH_PP = 0.5
BOOT_B = 10000
BOOT_SEED = 2027


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def analyze(method_key, entry):
    reps = entry["per_image_reps"]
    m_eval = entry["m_eval"]
    base = np.array(reps["base_correct_reps"], dtype=np.int8)
    agg_mean = np.array(reps["agg_mean_correct_reps"], dtype=np.int8)
    agg_maj = np.array(reps["agg_maj_correct_reps"], dtype=np.int8)
    n_images = entry["n_images"]
    n_pairs = len(base)
    assert n_pairs == n_images * m_eval, (n_pairs, n_images, m_eval)

    out = {}
    # --- regroup pairs from storage order [batch, rep, image] to [image, rep] ---
    # base/agg vectors are extended per batch as [rep0: img0..img_s-1,
    # rep1: img0..img_s-1, ...]; image ids per batch come from idx_vec
    # (image_index, sorted ascending within a batch).  Recover batch
    # boundaries from id wrap-around and rebuild the aligned id vector.
    idx = reps["image_index"]
    batches = []
    start = 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or idx[i] < idx[i - 1]:
            batches.append(idx[start:i])
            start = i
    aligned_ids = np.concatenate(
        [np.tile(np.array(b, dtype=np.int64), m_eval) for b in batches])
    assert len(aligned_ids) == n_pairs, (len(aligned_ids), n_pairs)
    order = np.argsort(aligned_ids, kind="stable")
    # sanity: every image exactly m_eval times, ids grouped
    sorted_ids = aligned_ids[order]
    assert all(sorted_ids[i * m_eval] == sorted_ids[(i + 1) * m_eval - 1]
               for i in range(n_images)), "grouping sanity failed"
    for agg_name, agg in (("mean", agg_mean), ("majority", agg_maj)):
        # correct per-image grouping: each row = one image's m_eval reps
        B = base[order].reshape(n_images, m_eval).astype(np.float64)
        A = agg[order].reshape(n_images, m_eval).astype(np.float64)
        p_base = B.mean(axis=1)          # per-image correct rate
        p_agg = A.mean(axis=1)
        d = p_base - p_agg               # positive = aggregate worse
        net_drop_pp = 100.0 * d.mean()
        sepaired_pp = 100.0 * d.std(ddof=1) / math.sqrt(n_images)
        # worst-case clustering bound: if all m_eval repetitions of an image
        # were perfectly correlated, SE would use the single-rep std
        d_single = (base.astype(np.float64) - agg.astype(np.float64))
        se_worst_pp = 100.0 * d_single.std(ddof=1) / math.sqrt(n_images)

        # repetition-pair discordant counts (paired by repetition index)
        a = int(np.sum((base == 1) & (agg == 0)))   # base right, agg wrong
        b = int(np.sum((base == 0) & (agg == 1)))   # base wrong, agg right

        # image-level paired bootstrap (true image clusters)
        rng = np.random.default_rng(BOOT_SEED)
        drops = np.empty(BOOT_B)
        for i in range(BOOT_B):
            idxb = rng.integers(0, n_images, n_images)
            drops[i] = 100.0 * d[idxb].mean()
        lo, hi = np.percentile(drops, [2.5, 97.5])

        if hi <= THRESH_PP:
            verdict = "PASS"
        elif lo > THRESH_PP:
            verdict = "FAIL"
        else:
            verdict = "UNCERTAIN"
        out[f"agg_{agg_name}"] = dict(
            pooled_base_pct=round(100.0 * base.mean(), 3),
            pooled_agg_pct=round(100.0 * agg.mean(), 3),
            pooled_base_wilson=[round(x, 3) for x in wilson(int(base.sum()), n_pairs)],
            pooled_agg_wilson=[round(x, 3) for x in wilson(int(agg.sum()), n_pairs)],
            net_drop_pp=round(net_drop_pp, 3),
            paired_se_pp=round(sepaired_pp, 3),
            se_worst_cluster_pp=round(se_worst_pp, 3),
            worst_case_upper_pp=round(net_drop_pp + 1.645 * se_worst_pp, 3),
            discordant_base_right_agg_wrong=a,
            discordant_base_wrong_agg_right=b,
            mcnemar_exact_p=round(mcnemar_exact(b, a), 5),
            boot_ci_pp=[round(lo, 3), round(hi, 3)],
            boot_B=BOOT_B, boot_seed=BOOT_SEED,
            threshold_pp=THRESH_PP, rule="one-sided: PASS iff boot upper <= 0.5pp",
            verdict=verdict)
    out["note"] = ("paired design: d_i computed from M_eval repetition rates "
                   "per image; bootstrap resamples images (respects within-"
                   "image repetition correlation)")
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join("results", "e4_s3_clean_paired", "latest",
                     "s3_paired_results.json")
    d = json.load(open(src, encoding="utf-8"))
    report = dict(source=os.path.abspath(src), methods={})
    for key, entry in d["results"].items():
        report["methods"][key] = analyze(key, entry)
        print(f"[{key}]")
        for agg, st in report["methods"][key].items():
            if agg == "note":
                continue
            print(f"  {agg}: drop={st['net_drop_pp']:+.3f}pp "
                  f"pairedSE={st['paired_se_pp']:.3f}pp "
                  f"disc(a={st['discordant_base_right_agg_wrong']}, "
                  f"b={st['discordant_base_wrong_agg_right']}) "
                  f"McNemar p={st['mcnemar_exact_p']} "
                  f"bootCI=[{st['boot_ci_pp'][0]:+.3f},{st['boot_ci_pp'][1]:+.3f}] "
                  f"-> {st['verdict']}")
    dst = os.path.join(os.path.dirname(src), "s3_paired_analysis.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("analysis written:", dst)


if __name__ == "__main__":
    main()
