# -*- coding: utf-8 -*-
"""V4 acceptance (verification/audit_eot_attack.md section 三·D): attack
saturation check for the production (detach) EOT attacker.

Axes (pre-registered; detach attacker only, deployed defense for eval):
  steps   : PGD-20 x R=1 x M=1        (ref: PGD-10 x 1 x 1 from V3)
  restarts: PGD-10 x R=3 x M=1        (any-restart success = attack success)
  M       : PGD-10 x R=1 x M=10       (ref: M=1 from V3, M=5 from E-E)
128 samples x seed 0 per config, per-method; per-sample success vectors
saved (any-restart and per-restart); Wilson 95% CI on robust rates.

Pass rule (pre-registered): an axis is saturated if the robust rate drops
<0.5pp versus its reference point or stops decreasing; >=2 of 3 axes
saturated -> attack-convergence verdict. Unsaturated axes are disclosed.

Run (cwd = nac_project):
    python scripts/verify_v4_saturation.py --methods ttc nac
"""
import argparse, json, logging, math, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (
    load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, clip_img_preprocessing,
    nac_counterattack, ttc_counterattack,
)
from utils import convert_models_to_fp32

clip_load = clip.load
device = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_SEED = 999
N_BATCHES = 8
BATCH_SIZE = 16
CONFIGS = [          # (steps, restarts, eot_m) - reference (10,1,1) in V3
    dict(tag='steps20', steps=20, restarts=1, eot_m=1),
    dict(tag='restarts3', steps=10, restarts=3, eot_m=1),
    dict(tag='M10', steps=10, restarts=1, eot_m=10),
]


def wilson(p_succ, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = p_succ / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--n_batches', type=int, default=N_BATCHES)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'verify_v4_saturation', stamp)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outdir, 'v4.log'),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    print(f"V4 saturation check; outdir: {outdir}", flush=True)

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
    args.batch_size = cli.batch_size

    model, _ = clip_load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    val_dataset, val_loader = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)

    def mk(name, x):
        if name == 'ttc':
            return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None)
        return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                    alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                    norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None,
                    nac_momentum=args.nac_momentum, clamp_lookahead=False)

    deployed = {'ttc': ttc_counterattack, 'nac': nac_counterattack}
    eps, alpha = args.ttc_eps, args.ttc_stepsize

    def eval_success(images, target, delta, name):
        """Deployed-defense success eval (defense fails -> attack success)."""
        torch.manual_seed(EVAL_SEED)
        torch.cuda.manual_seed_all(EVAL_SEED)
        x_adv = torch.clamp(images + delta, 0, 1)
        Delta_ev = deployed[name](**mk(name, x_adv.detach())).detach()
        defended = torch.clamp(x_adv + Delta_ev, 0, 1)
        with torch.no_grad():
            output, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None,
                dataset_name='cifar10')
            return (output.argmax(dim=1) != target).cpu().tolist()

    def attack(images, target, name, steps, restarts, eot_m):
        """Detach EOT attacker, R restarts; returns any-restart success,
        per-restart success vectors, per-sample L2 norms (best restart)."""
        per_restart = []
        final_norms = None
        for r in range(restarts):
            torch.manual_seed(5000 + 100 * r)
            torch.cuda.manual_seed_all(5000 + 100 * r)
            delta = torch.clamp(
                torch.zeros_like(images).uniform_(-eps, eps), -eps, eps)
            delta.requires_grad_(True)
            for _s in range(steps):
                x_attack = torch.clamp(images + delta, 0, 1)
                losses = []
                for _m in range(eot_m):
                    Delta = deployed[name](
                        **mk(name, x_attack.detach())).detach()
                    defended = torch.clamp(x_attack + Delta, 0, 1)
                    output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(defended)),
                        text_tokens=text_tokens, prompt_token=None,
                        dataset_name='cifar10')
                    losses.append(F.cross_entropy(output, target))
                loss = torch.stack(losses).mean()
                grad = torch.autograd.grad(loss, delta)[0]
                delta = torch.clamp(delta + alpha * torch.sign(grad),
                                    -eps, eps).detach()
                delta.requires_grad_(True)
            delta = delta.detach()
            per_restart.append(eval_success(images, target, delta, name))
            n2 = delta.view(delta.size(0), -1).norm(dim=1).cpu()
            final_norms = n2 if final_norms is None else torch.maximum(final_norms, n2)
        any_succ = [bool(any(pr[i] for pr in per_restart))
                    for i in range(len(per_restart[0]))] if restarts > 1 \
            else list(per_restart[0])
        return any_succ, per_restart, final_norms.tolist()

    results = {}
    for name in cli.methods:
        cfg_out = {}
        for cfg in CONFIGS:
            any_s, per_r, nrm = [], [], []
            for bi, (images, target) in enumerate(val_loader):
                if bi >= cli.n_batches:
                    break
                images, target = images.to(device), target.to(device)
                a, pr, n2 = attack(images, target, name, cfg['steps'],
                                   cfg['restarts'], cfg['eot_m'])
                any_s.extend(a)
                if len(per_r) == 0:
                    per_r = [[] for _ in pr]
                for i, v in enumerate(pr):
                    per_r[i].extend(v)
                nrm.extend(n2)
                print(f"  {name} {cfg['tag']} batch {bi}: "
                      f"any-success={float(np.mean(a)):.3f}", flush=True)
            n = len(any_s)
            succ = int(np.sum(any_s))
            lo, hi = wilson(n - succ, n)   # CI on the robust side
            cfg_out[cfg['tag']] = dict(
                config=cfg, n_samples=n, attack_success=succ,
                robust_rate=100.0 * (n - succ) / n,
                robust_ci95=[round(lo, 3), round(hi, 3)],
                per_restart_success=per_r,
                any_restart_success=any_s,
                l2_norm_mean=float(np.mean(nrm)))
            logging.info("%s %s: robust=%.2f%% CI=[%.2f, %.2f]",
                         name, cfg['tag'], cfg_out[cfg['tag']]['robust_rate'],
                         lo, hi)
            print(f"[{name}] {cfg['tag']}: robust="
                  f"{cfg_out[cfg['tag']]['robust_rate']:.2f}% "
                  f"CI95={cfg_out[cfg['tag']]['robust_ci95']}", flush=True)
            # incremental save after EVERY config: a crash preserves finished ones
            results.setdefault(name, {}).update({cfg['tag']: cfg_out[cfg['tag']]})
            with open(os.path.join(outdir, 'v4_results.json'), 'w') as fh:
                json.dump(results, fh, indent=2)
        results[name] = cfg_out

    with open(os.path.join(outdir, 'v4_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"V4 done -> {outdir}", flush=True)


if __name__ == '__main__':
    main()
