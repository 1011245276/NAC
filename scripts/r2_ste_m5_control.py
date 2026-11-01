# -*- coding: utf-8 -*-
"""R2 audit addendum: extend the V3 STE control to the production M_att=5.

Motivation (Independent_Audit_R2/05 D-3 + 06 §2): V3 compared detach vs
graph-connected STE at M_att=1 only, while every production robustness
number uses M_att=5. This script re-runs the paired strength comparison at
M_att=5 (EOT over the defense's random init, CE-space mean, PGD-10,
eps=4/255, 128 samples) so that "attacker sufficiency" rests on the
production M, and so the CLIP-MAD reconciliation (their 'Full' row reports
TTC@CIFAR-10 = 12.8% under nominally PGD-10 + EOT-5) can be tested: if the
full-graph STE surrogate at M=5 still collapses within OUR TTC config, the
12.8pp gap cannot be explained by detach-vs-full-gradient semantics and
must come from configuration/protocol differences.

Variants (identical to verify_v3 definitions, reused by import):
  detach : deployed functions as-is (structural stop-gradient).
  STE    : verify_v3 graph-connected reimplementation (3 disclosed changes).

Protocol: CIFAR-10, ViT-B/32, bs=8 (STE retains M=5 inner graphs), 16
batches = 128 samples, seed 0, eval by the DEPLOYED defense at fixed
EVAL_SEED (single draw, same as V3/V4). Pre-registered interpretation
bands carry over from V3: |robust diff| <= 1pp -> detach sufficient.

Run (cwd = nac_project):
    python scripts/r2_ste_m5_control.py --methods ttc nac
"""
import argparse, json, logging, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, r'F:\codex\NAC\nac_research2\scripts')

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
from verify_v3_gradient_variants import (
    ttc_counterattack_ste, nac_counterattack_ste, XI_SEED, EVAL_SEED)

clip_load = clip.load
device = "cuda" if torch.cuda.is_available() else "cpu"

N_BATCHES = 16        # 16 x 8 = 128 samples (V3 caliber)
BATCH_SIZE = 8        # STE retains M=5 inner graphs -> smaller batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--n_batches', type=int, default=N_BATCHES)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--eot_m', type=int, default=5)
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'verify_v3_ste_m5', stamp)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outdir, 'v3m5.log'),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("V3@M=5 | methods=%s n_batches=%d bs=%d steps=%d M=%d",
                 cli.methods, cli.n_batches, cli.batch_size, cli.steps, cli.eot_m)
    print(f"STE M={cli.eot_m} control; outdir: {outdir}", flush=True)

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

    deployed = {'ttc': ttc_counterattack, 'nac': nac_counterattack}

    def ste_call(name, x):
        kw = dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                  alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                  norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                  beta=args.beta, create_graph=True)
        if name == 'nac':
            kw['nac_momentum'] = args.nac_momentum
        return (nac_counterattack_ste if name == 'nac'
                else ttc_counterattack_ste)(**kw)

    eps, alpha, steps, M = args.ttc_eps, args.ttc_stepsize, cli.steps, cli.eot_m

    def eot_losses(name, x_attack, variant):
        """Mean CE over M defense draws (fresh inits from the global RNG
        stream, no reseeding between draws -- same as production)."""
        losses = []
        for _m in range(M):
            if variant == 'detach':
                Delta = deployed[name](**dict(
                    model=model, X=x_attack.detach(), prompter=prompter,
                    add_prompter=add_prompter, alpha=args.ttc_stepsize,
                    attack_iters=args.ttc_numsteps, norm='l_inf',
                    epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None,
                    **({'nac_momentum': args.nac_momentum,
                        'clamp_lookahead': False} if name == 'nac' else {})
                )).detach()
            else:
                Delta = ste_call(name, x_attack)
            defended = torch.clamp(x_attack + Delta, 0, 1)
            output, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None,
                dataset_name='cifar10')
            losses.append(F.cross_entropy(output, target))
        return torch.stack(losses).mean()

    results = {}
    for name in cli.methods:
        per_sample = {'detach': [], 'ste': []}
        norms = {'detach': [], 'ste': []}
        dir_cos = []

        for bi, (images, target) in enumerate(val_loader):
            if bi >= cli.n_batches:
                break
            images, target = images.to(device), target.to(device)
            target = target  # used inside eot_losses closure

            # ---- direction check at fixed delta0 / fixed xi stream ----
            torch.manual_seed(XI_SEED)
            torch.cuda.manual_seed_all(XI_SEED)
            delta0 = torch.clamp(
                torch.zeros_like(images).uniform_(-eps, eps), -eps, eps)
            delta0.requires_grad_(True)
            torch.manual_seed(2000 + bi)
            torch.cuda.manual_seed_all(2000 + bi)
            cos_pair = []
            for variant in ('detach', 'ste'):
                torch.manual_seed(3000 + bi)   # same xi stream for both
                torch.cuda.manual_seed_all(3000 + bi)
                x0 = torch.clamp(images + delta0, 0, 1)
                loss = eot_losses(name, x0, variant)
                g = torch.autograd.grad(loss, delta0)[0]
                cos_pair.append(g.flatten().detach())
            cos = float(torch.nn.functional.cosine_similarity(
                cos_pair[0], cos_pair[1], dim=0))
            dir_cos.append(cos)
            logging.info("%s batch %d direction cos(M=%d)=%.4f",
                         name, bi, M, cos)
            print(f"  {name} batch {bi} direction cos={cos:.4f}", flush=True)
            del cos_pair
            torch.cuda.empty_cache()

            # ---- paired attacks: detach (production) vs STE, both at M ----
            for variant in ('detach', 'ste'):
                torch.manual_seed(1000 + bi)
                torch.cuda.manual_seed_all(1000 + bi)
                delta = torch.clamp(
                    torch.zeros_like(images).uniform_(-eps, eps), -eps, eps)
                delta.requires_grad_(True)
                for _s in range(steps):
                    x_attack = torch.clamp(images + delta, 0, 1)
                    loss = eot_losses(name, x_attack, variant)
                    grad = torch.autograd.grad(loss, delta)[0]
                    delta = torch.clamp(delta + alpha * torch.sign(grad),
                                        -eps, eps).detach()
                    delta.requires_grad_(True)
                delta = delta.detach()
                # ---- success eval with the DEPLOYED defense (fixed seed) ----
                torch.manual_seed(EVAL_SEED)
                torch.cuda.manual_seed_all(EVAL_SEED)
                x_adv = torch.clamp(images + delta, 0, 1)
                Delta_ev = deployed[name](**dict(
                    model=model, X=x_adv.detach(), prompter=prompter,
                    add_prompter=add_prompter, alpha=args.ttc_stepsize,
                    attack_iters=args.ttc_numsteps, norm='l_inf',
                    epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None,
                    **({'nac_momentum': args.nac_momentum,
                        'clamp_lookahead': False} if name == 'nac' else {})
                )).detach()
                defended = torch.clamp(x_adv + Delta_ev, 0, 1)
                with torch.no_grad():
                    output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(defended)),
                        text_tokens=text_tokens, prompt_token=None,
                        dataset_name='cifar10')
                    success = (output.argmax(dim=1) != target).cpu().tolist()
                per_sample[variant].extend(success)
                norms[variant].extend(
                    delta.view(delta.size(0), -1).norm(dim=1).cpu().tolist())
                logging.info("%s batch %d variant=%s success=%.3f",
                             name, bi, variant, float(np.mean(success)))
                print(f"  {name} batch {bi} {variant} attack-success="
                      f"{float(np.mean(success)):.3f}", flush=True)
                torch.cuda.empty_cache()

        sd = np.array(per_sample['detach'], dtype=bool)
        ss = np.array(per_sample['ste'], dtype=bool)
        n = len(sd)
        rate = {v: 100.0 * float(np.mean(per_sample[v])) for v in per_sample}

        def wilson(k, n, z=1.96):
            if n == 0:
                return (0.0, 0.0)
            p = k / n
            d = 1 + z * z / n
            c = (p + z * z / (2 * n)) / d
            h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
            return (100 * (c - h), 100 * (c + h))

        paired = dict(
            n=int(n),
            both_fail=int(np.sum(sd & ss)),
            only_detach_fail=int(np.sum(sd & ~ss)),
            only_ste_fail=int(np.sum(~sd & ss)),
            neither=int(np.sum(~sd & ~ss)),
        )
        results[name] = dict(
            n_samples=n, eot_m=M, steps=steps,
            robust_rate_detach=100.0 - rate['detach'],
            robust_rate_ste=100.0 - rate['ste'],
            robust_rate_diff_ste_minus_detach=
                (100.0 - rate['ste']) - (100.0 - rate['detach']),
            wilson_ste_robust=[round(x, 3)
                               for x in wilson(int(np.sum(~ss)), n)],
            direction_cos_mean=float(np.mean(dir_cos)),
            direction_cos_per_batch=dir_cos,
            paired=paired,
            per_sample_attack_success=per_sample,
            l2_perturb_norm_mean={v: float(np.mean(norms[v]))
                                  for v in norms},
            verdict='OK')
        logging.info("%s robust detach=%.2f%% ste=%.2f%% diff=%.2fpp cos=%.4f",
                     name, results[name]['robust_rate_detach'],
                     results[name]['robust_rate_ste'],
                     results[name]['robust_rate_diff_ste_minus_detach'],
                     results[name]['direction_cos_mean'])
        print(f"[{name}] robust_detach={results[name]['robust_rate_detach']:.2f}% "
              f"robust_ste={results[name]['robust_rate_ste']:.2f}% "
              f"diff={results[name]['robust_rate_diff_ste_minus_detach']:+.2f}pp "
              f"cos={results[name]['direction_cos_mean']:.4f}", flush=True)

    with open(os.path.join(outdir, 'v3m5_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"STE-M5 control done -> {outdir}", flush=True)


if __name__ == '__main__':
    main()
