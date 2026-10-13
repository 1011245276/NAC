#!/usr/bin/env python3
"""Compare batch-1 diagnostic dumps from official MAC vs the ported pipeline."""
import sys
import numpy as np


def main():
    f_o, f_p = sys.argv[1], sys.argv[2]
    o = np.load(f_o, allow_pickle=True)
    p = np.load(f_p, allow_pickle=True)

    print("=== batch-1 stage comparison (official vs port) ===")

    for k in ('images', 'target', 'tokens', 'clean_logits'):
        so, sp = o[k].shape, p[k].shape
        print(f"{k:14s} shapes: official {so}  port {sp}  {'OK' if so == sp else 'MISMATCH'}")

    to, tp = o['target'].astype(np.int64), p['target'].astype(np.int64)
    if to.shape == tp.shape:
        same_t = int((to == tp).sum())
        print(f"target         identical: {same_t}/{len(to)}  {'OK' if same_t == len(to) else 'MISMATCH'}")
        if same_t != len(to):
            idx = np.where(to != tp)[0][:10]
            print(f"  first differing idx {idx.tolist()}  official {to[idx].tolist()}  port {tp[idx].tolist()}")

    io, ip = o['images'].astype(np.float64), p['images'].astype(np.float64)
    if io.shape == ip.shape:
        d = np.abs(io - ip)
        print(f"images         max|diff| {d.max():.3e}  mean {d.mean():.3e}  "
              f"identical={bool(d.max() == 0)}")

    ko, kp = o['tokens'].astype(np.int64), p['tokens'].astype(np.int64)
    if ko.shape == kp.shape:
        print(f"token ids      identical: {bool((ko == kp).all())}")

    so_t = [str(x) for x in o['texts']]
    sp_t = [str(x) for x in p['texts']]
    print(f"texts          identical: {so_t == sp_t}  (len {len(so_t)} vs {len(sp_t)})")
    if so_t != sp_t:
        for i, (x, y) in enumerate(zip(so_t, sp_t)):
            if x != y:
                print(f"  first differing text idx {i}: official {x!r}  port {y!r}")
                break

    lo, lp = o['clean_logits'].astype(np.float64), p['clean_logits'].astype(np.float64)
    if lo.shape == lp.shape:
        dl = np.abs(lo - lp)
        print(f"clean_logits   max|diff| {dl.max():.3e}  mean {dl.mean():.3e}")
        ao, ap_ = lo.argmax(1), lp.argmax(1)
        agree = int((ao == ap_).sum())
        print(f"clean argmax   agreement: {agree}/{len(ao)}  {'OK' if agree == len(ao) else 'MISMATCH'}")
        to_ = o['target'].astype(np.int64)
        print(f"clean top1     official {100.0*(ao==to_).mean():.2f}  port {100.0*(ap_==to_).mean():.2f}")


if __name__ == '__main__':
    main()