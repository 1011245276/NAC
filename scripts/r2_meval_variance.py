# -*- coding: utf-8 -*-
"""R2 audit addendum: deployment-side M_eval variance at the floor.

Motivation (Independent_Audit_R2/05 D-1): the protocol pre-registered the
M_att / M_def / M_eval separation, but robustness-side evaluation was
implemented with M_eval=1 (a single fresh defense draw per batch). This
experiment quantifies what M_eval=1 actually estiminates at the floor:
for each image, attack once (production EOT M=5 attacker), then deploy the
defense M_eval=10 times with fresh random inits and record every draw's
correctness. Outputs the per-draw-index robust rates, the across-draw
spread (the defense-init variance component that a single draw fails to
average out), and the any-draw robust rate.

Protocol: CIFAR-10, ViT-B/32, eps_att=eps_def=4/255, PGD-10 alpha=1/255,
EOT M_att=5 (production detach attacker), K=2, 1024 samples (32 batches),
seed 0. Eval draws reseeded per batch (EVAL_SEED+batch) so the 10 draws are
reproducible; draws within a batch are sequential (independent) samples of
the defense's init randomness.

Run (cwd = nac_project):
    python scripts/r2_meval_variance.py --methods ttc nac
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
    nac_counterattack, ttc_counterattack,
)
from utils import convert_models_to_fp32

clip_load = clip.load
device = "cuda" if torch.cuda.is_available() else "cpu"

N_BATCHES = 32     # 32 x 32 = 1024 samples (formal caliber)
BATCH_SIZE = 32
M_EVAL = 10
EVAL_SEED = 999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--n_batches', type=int, default=N_BATCHES)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--eot_m', type=int, default=5)
    ap.add_argument('--m_eval', type=int, default=M_EVAL)
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'r2_meval_variance', stamp)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outdir, 'meval.log'),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("M_eval variance | methods=%s n_batches=%d bs=%d M_att=%d "
                 "M_eval=%d", cli.methods, cli.n_batches, cli.batch_size,
                 cli.eot_m, cli.m_eval)
    print(f"M_eval variance experiment; outdir: {outdir}", flush=True)

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
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    val_dataset, val_loader = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)

    fns = {'ttc': ttc_counterattack, 'nac': nac_counterattack}

    def mk(name, x):
        if name == 'ttc':
            return dict(model=model, X=x, prompter=prompter,
                        add_prompter=add_prompter, alpha=args.ttc_stepsize,
                        attack_iters=args.ttc_numsteps, norm='l_inf',
                        epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None)
        return dict(model=model, X=x, prompter=prompter,
                    add_prompter=add_prompter, alpha=args.ttc_stepsize,
                    attack_iters=args.ttc_numsteps, norm='l_inf',
                    epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None,
                    nac_momentum=args.nac_momentum, clamp_lookahead=False)

    eps, alpha, steps, M = args.ttc_eps, args.ttc_stepsize, cli.steps, cli.eot_m
    ME = cli.m_eval

    results = {}
    for name in cli.methods:
        fn = fns[name]
        # per-sample storage: [n_samples, ME] booleans (draw j correctness)
        draw_matrix = []
        index_list = []
        t0 = time.time()
        for bi, (images, target) in enumerate(val_loader):
            if bi >= cli.n_batches:
                break
            images, target = images.to(device), target.to(device)

            # ---- production EOT attack (M_att=5, detach, CE-mean) ----
            torch.manual_seed(1000 + bi)
            torch.cuda.manual_seed_all(1000 + bi)
            delta = torch.clamp(
                torch.zeros_like(images).uniform_(-eps, eps), -eps, eps)
            delta.requires_grad_(True)
            for _s in range(steps):
                x_attack = torch.clamp(images + delta, 0, 1)
                losses = []
                for _m in range(M):
                    Delta = fn(**mk(name, x_attack.detach())).detach()
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
            attacked = torch.clamp(images + delta, 0, 1).detach()

            # ---- M_eval deployment draws (fresh inits, sequential stream) ----
            torch.manual_seed(EVAL_SEED + bi)
            torch.cuda.manual_seed_all(EVAL_SEED + bi)
            draws = []
            for _j in range(ME):
                with torch.enable_grad():
                    Delta = fn(**mk(name, attacked.data))
                with torch.no_grad():
                    defended = torch.clamp(attacked + Delta, 0, 1)
                    output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(defended)),
                        text_tokens=text_tokens, prompt_token=None,
                        dataset_name='cifar10')
                    draws.append((output.argmax(dim=1) == target).cpu())
            draw_matrix.extend(torch.stack(draws, dim=1).tolist())  # [bs, ME]
            index_list.extend(
                (bi * args.batch_size + torch.arange(images.size(0))).tolist())
            dm = np.array(draw_matrix, dtype=bool)
            logging.info("%s batch %d: per-draw robust=%s any=%.3f",
                         name, bi,
                         [round(100 * float(dm[:, j].mean()), 2)
                          for j in range(dm.shape[1])],
                         float(dm.any(axis=1).mean()))
            print(f"  {name} batch {bi}: "
                  f"draw-robust={[round(100 * float(dm[:, j].mean()), 2) for j in range(min(3, dm.shape[1]))]}...",
                  flush=True)
            torch.cuda.empty_cache()

        dm = np.array(draw_matrix, dtype=bool)  # [n, ME]
        n = dm.shape[0]
        per_draw_rates = [round(100 * float(dm[:, j].mean()), 4)
                          for j in range(dm.shape[1])]
        mean_rate = 100 * float(dm.mean())
        per_draw_std = float(np.std(per_draw_rates, ddof=1))
        binom_only = 100 * float(np.std(
            [np.random.default_rng(7).binomial(n, dm.mean() / 100) / n
             for _ in range(2000)], ddof=1))
        hist = {int(k): int(v) for k, v in
                zip(*np.unique(dm.sum(axis=1), return_counts=True))}
        results[name] = dict(
            n_samples=n, m_eval=ME, eot_m=M, steps=steps,
            per_draw_robust_rate=per_draw_rates,
            mean_robust_rate=round(mean_rate, 4),
            per_draw_rate_std=round(per_draw_std, 4),
            binomial_only_std_approx=round(binom_only, 4),
            any_draw_robust_rate=round(100 * float(dm.any(axis=1).mean()), 4),
            all_draws_robust_rate=round(100 * float(dm.all(axis=1).mean()), 4),
            draws_correct_histogram=hist,
            elapsed_min=round((time.time() - t0) / 60, 1),
            # per-sample per-draw vectors (V5 practice)
            per_sample=dict(
                n=n,
                draws_correct=dm.astype(int).tolist(),
                image_index=index_list),
        )
        logging.info("%s mean=%.4f per-draw-std=%.4f any=%.4f",
                     name, mean_rate, per_draw_std,
                     100 * float(dm.any(axis=1).mean()))
        print(f"[{name}] mean_robust={mean_rate:.4f}% "
              f"per-draw-std={per_draw_std:.4f}pp "
              f"any-draw={100 * float(dm.any(axis=1).mean()):.4f}%", flush=True)

    with open(os.path.join(outdir, 'meval_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"M_eval variance done -> {outdir}", flush=True)


if __name__ == '__main__':
    main()
