# -*- coding: utf-8 -*-
"""V2 acceptance (verification/audit_eot_attack.md): finite-difference direction
check for the detached-gradient EOT attacker, WITHOUT touching production code.

What it checks
--------------
The attacker's analytic gradient g = dL/d(delta) (with Delta = counter_fn(...)
detached, i.e. only the direct term dCE/dx_attack) against central finite
differences of the SAME forward pipeline along R random directions:
    d_r = [L(delta + e*v_r) - L(delta - e*v_r)] / (2e)   ~=   <g, v_r>
If the analytic gradient is the true gradient of the executed loss, the two
must be strongly correlated across directions.

Randomness control (pre-registered)
-----------------------------------
- Defense-side randomness (counterattack uniform init xi) is reset before
  EVERY loss evaluation: torch.manual_seed / cuda.manual_seed_all(XI_SEED).
  Hence L(delta) is a deterministic function of delta, and both sides of the
  central difference see the SAME xi - no noise term in d_r.
- Direction set and start delta are drawn once with separate fixed seeds
  (DIR_SEED / DELTA_SEED) BEFORE any xi reset, so they do not depend on the
  global RNG stream afterwards.
- M=1 (single defense re-run) isolates the gradient path; the M>1 path is the
  same autograd mechanism applied to stack().mean().

Pass rule (pre-registered)
--------------------------
Pearson(grad_proj, finite_diff) > 0.9  AND  sign agreement > 85%  per method.
Failure => the detached-gradient path is NOT the gradient of the executed
loss; locate the discrepancy source before any production code change.

Run (cwd = nac_project):
    python scripts/verify_v2_finite_diff.py --methods ttc nac
"""
import argparse, json, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (
    load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, clip_img_preprocessing,
    doc_counterattack, nac_counterattack, ttc_counterattack,
)
from utils import convert_models_to_fp32

clip_load = clip.load
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- pre-registered constants ----
XI_SEED = 1234        # defense-side RNG reset before every loss evaluation
DIR_SEED = 2026       # random direction set
DELTA_SEED = 7        # attack start delta
EPS_FD = 1e-3         # finite-difference step in [0,1] pixel space
PEARSON_PASS = 0.90
SIGN_PASS = 0.85
N_BATCHES = 2         # 2 x 32 = 64 samples (audit V2 requirement: >=64)
DIRECTIONS = 12


def loss_eval(fn, mk, model, prompter, text_tokens, dataset, images, target,
              delta):
    """Deterministic loss L(delta): mean CE over the batch, xi fixed."""
    torch.manual_seed(XI_SEED)
    torch.cuda.manual_seed_all(XI_SEED)
    x_attack = torch.clamp(images + delta, 0, 1)
    Delta = fn(**mk(x_attack.detach())).detach()
    defended = torch.clamp(x_attack + Delta, 0, 1)
    output, _, _, _ = multiGPU_CLIP(
        None, None, None, model,
        prompter(clip_img_preprocessing(defended)),
        text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
    return F.cross_entropy(output, target)


def loss_eval_frozen(images, target, delta, Delta_frozen, model, prompter,
                     text_tokens, dataset):
    """Direct-term-only loss with the counterattack output FROZEN at its
    delta0 value: smooth in delta, no inner re-optimization, no RNG use."""
    x_attack = torch.clamp(images + delta, 0, 1)
    defended = torch.clamp(x_attack + Delta_frozen, 0, 1)
    output, _, _, _ = multiGPU_CLIP(
        None, None, None, model,
        prompter(clip_img_preprocessing(defended)),
        text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
    return F.cross_entropy(output, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--n_batches', type=int, default=N_BATCHES)
    ap.add_argument('--directions', type=int, default=DIRECTIONS)
    ap.add_argument('--eps_fd', type=float, default=EPS_FD)
    ap.add_argument('--frozen_delta', action='store_true',
                    help='freeze Delta at its delta0 value: FD then verifies '
                         'only the smooth direct term (discriminating test '
                         'against inner-trajectory flip noise)')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float64'],
                    help='float64 kills fp32 forward-error noise in the FD '
                         'measurement (frozen mode only needs plain forwards)')
    ap.add_argument('--eps_ladder', nargs='+', type=float, default=None,
                    help='frozen mode only: FD steps tried in one pass '
                         '(no eps-ball clamp - the check targets the gradient '
                         'function, not attack feasibility). Overrides --eps_fd')
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'verify_v2_finite_diff', stamp)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outdir, 'v2.log'),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("V2 finite-diff check | methods=%s n_batches=%d R=%d eps_fd=%g",
                 cli.methods, cli.n_batches, cli.directions, cli.eps_fd)
    print(f"V2 finite-diff check; outdir: {outdir}", flush=True)

    # ---- identical init path to adaptive_attack_eot.py ----
    sys.argv = [
        'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd', '--test_eps', '4.0',
        '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
        '--root', './data', '--dataset', 'cifar10', '--seed', '0', '--num_seeds', '1',
        '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
        '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
        '--learnable_tau', '0.155', '--temperature', '75.0',
        '--counterattack', 'ttc', '--outdir', outdir,
    ]
    from nac_fair_experiment import parse_options
    args = parse_options()
    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.

    model, _ = clip_load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    if cli.dtype == 'float64':
        model = model.double()
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
    if cli.dtype == 'float64':
        prompter = prompter.double()
        add_prompter = add_prompter.double()

    val_dataset, val_loader = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)

    class _A: pass
    docargs = _A()
    docargs.learnable_tau = args.learnable_tau
    docargs.temperature = args.temperature

    def mk(name, x):
        if name == 'ttc':
            return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None)
        if name == 'nac':
            return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None,
                        nac_momentum=args.nac_momentum, clamp_lookahead=False)
        return dict(args=docargs, model=model, X=x, prompter=prompter,
                    add_prompter=add_prompter, alpha=args.ttc_stepsize,
                    attack_iters=args.ttc_numsteps, norm='l_inf',
                    epsilon=args.ttc_eps, beta=args.beta, clip_visual=None)

    fns = {'ttc': ttc_counterattack, 'nac': nac_counterattack,
           'doc': doc_counterattack}

    # ---- draw directions & start deltas once (independent of xi stream) ----
    # Shape is taken from the first batch (loader yields CLIP-preprocessed
    # 3x224x224 tensors, the same space the attack perturbation lives in).
    v = None
    e = cli.eps_fd
    results = {}
    for name in cli.methods:
        fn, method_mk = fns[name], (lambda x, _n=name: mk(_n, x))
        per_batch = []
        for bi, (images, target) in enumerate(val_loader):
            if bi >= cli.n_batches:
                break
            images, target = images.to(device), target.to(device)
            if cli.dtype == 'float64':
                images = images.double()
            if v is None:
                torch.manual_seed(DIR_SEED)
                torch.cuda.manual_seed_all(DIR_SEED)
                v = torch.randn(cli.directions, *images.shape[1:],
                                device=device, dtype=images.dtype)
                v = v / v.view(cli.directions, -1).norm(dim=1) \
                        .view(-1, 1, 1, 1)
            eps = args.ttc_eps
            torch.manual_seed(DELTA_SEED + bi)
            torch.cuda.manual_seed_all(DELTA_SEED + bi)
            delta0 = torch.clamp(torch.zeros_like(images).uniform_(-eps, eps),
                                 -eps, eps)
            sat = ((images + delta0 <= 0) | (images + delta0 >= 1)) \
                       .float().mean().item()

            if cli.frozen_delta:
                # Delta computed ONCE at delta0 (fixed xi), then frozen.
                torch.manual_seed(XI_SEED)
                torch.cuda.manual_seed_all(XI_SEED)
                x_att0 = torch.clamp(images + delta0, 0, 1)
                Delta_frozen = fn(**method_mk(x_att0.detach())).detach()

                # analytic gradient of the frozen-Delta loss (direct term)
                d = delta0.clone().requires_grad_(True)
                loss = loss_eval_frozen(images, target, d, Delta_frozen,
                                        model, prompter, text_tokens,
                                        'cifar10')
                g = torch.autograd.grad(loss, d)[0]

                # central differences - no RNG consumers left, no_grad safe;
                # NO eps-ball clamp: the check targets the gradient function,
                # feasibility is irrelevant and clamping would project v.
                eps_list = cli.eps_ladder or [cli.eps_fd]
                proj, fd_by_e = [], {e: [] for e in eps_list}
                with torch.no_grad():
                    for r in range(cli.directions):
                        proj.append((g * v[r]).sum().item())
                        for e in eps_list:
                            lp = loss_eval_frozen(images, target,
                                                  delta0 + e * v[r],
                                                  Delta_frozen, model, prompter,
                                                  text_tokens, 'cifar10')
                            lm = loss_eval_frozen(images, target,
                                                  delta0 - e * v[r],
                                                  Delta_frozen, model, prompter,
                                                  text_tokens, 'cifar10')
                            fd_by_e[e].append(((lp - lm) / (2 * e)).item())
            else:
                # analytic gradient (detached-Delta path, exactly as the attacker)
                d = delta0.clone().requires_grad_(True)
                loss = loss_eval(fn, method_mk, model, prompter, text_tokens,
                                 'cifar10', images, target, d)
                g = torch.autograd.grad(loss, d)[0]

                # central differences along R directions
                # NOTE: no torch.no_grad() here - the counterattack functions
                # build their own internal autograd graph; the outer loss is only
                # read as a scalar, its graph is freed without a backward call.
                proj, fd = [], []
                for r in range(cli.directions):
                    lp = loss_eval(fn, method_mk, model, prompter, text_tokens,
                                   'cifar10', images, target,
                                   torch.clamp(delta0 + e * v[r], -eps, eps))
                    lm = loss_eval(fn, method_mk, model, prompter, text_tokens,
                                   'cifar10', images, target,
                                   torch.clamp(delta0 - e * v[r], -eps, eps))
                    fd.append(((lp - lm) / (2 * e)).item())
                    proj.append((g * v[r]).sum().item())
            rec = dict(batch=bi, sat_frac=sat, loss=float(loss),
                       grad_norm=float(g.view(-1).norm()), proj=proj)
            if cli.frozen_delta:
                rec['fd_by_eps'] = {str(e): f_ for e, f_ in fd_by_e.items()}
            else:
                rec['fd'] = fd
            per_batch.append(rec)
            logging.info("%s batch %d: loss=%.4f |g|=%.4f sat=%.3f",
                         name, bi, loss, g.view(-1).norm(), sat)
            print(f"  {name} batch {bi} done (loss={float(loss):.4f}, "
                  f"|g|={float(g.view(-1).norm()):.4f})", flush=True)

        # ---- aggregate across batches ----
        def _stats(p, f):
            pc = float(np.corrcoef(p, f)[0, 1])
            rs = np.argsort(np.argsort(p)), np.argsort(np.argsort(f))
            sc = float(np.corrcoef(rs[0], rs[1])[0, 1])
            sa = float((np.sign(p) == np.sign(f)).mean())
            scale = float(np.linalg.norm(f) / (np.linalg.norm(p) + 1e-12))
            return dict(pearson=pc, spearman=sc, sign_agree=sa,
                        fd_over_proj_norm=scale)

        p = np.array([x for b in per_batch for x in b['proj']])
        if cli.frozen_delta:
            eps_keys = sorted(per_batch[0]['fd_by_eps'], key=float)
            per_eps = {k: _stats(p, np.array([x for b in per_batch
                                              for x in b['fd_by_eps'][k]]))
                       for k in eps_keys}
            best_key = max(per_eps, key=lambda k: per_eps[k]['pearson'])
            st = per_eps[best_key]
        else:
            per_eps = None
            st = _stats(p, np.array([x for b in per_batch for x in b['fd']]))
        pc, sc, sa, scale = (st['pearson'], st['spearman'],
                             st['sign_agree'], st['fd_over_proj_norm'])
        verdict = "PASS" if (pc > PEARSON_PASS and sa > SIGN_PASS) else "FAIL"
        results[name] = dict(per_batch=per_batch, pearson=pc, spearman=sc,
                             sign_agree=sa, fd_over_proj_norm=scale,
                             mode='frozen_delta' if cli.frozen_delta else 'reopt',
                             dtype=cli.dtype, per_eps=per_eps,
                             eps_fd=e, xi_seed=XI_SEED, dir_seed=DIR_SEED,
                             delta_seed=DELTA_SEED, verdict=verdict)
        logging.info("%s: pearson=%.4f spearman=%.4f sign=%.3f scale=%.3f -> %s",
                     name, pc, sc, sa, scale, verdict)
        if per_eps:
            for k in per_eps:
                s = per_eps[k]
                print(f"[{name}] eps={k}: pearson={s['pearson']:.4f} "
                      f"spearman={s['spearman']:.4f} "
                      f"sign={s['sign_agree']:.3f} scale={s['fd_over_proj_norm']:.3f}",
                      flush=True)
        print(f"[{name}] pearson={pc:.4f} spearman={sc:.4f} "
              f"sign_agree={sa:.3f} -> {verdict}", flush=True)

    with open(os.path.join(outdir, 'v2_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    overall = all(r['verdict'] == 'PASS' for r in results.values())
    print(f"V2 overall: {'PASS' if overall else 'FAIL'} -> {outdir}", flush=True)


if __name__ == '__main__':
    main()
