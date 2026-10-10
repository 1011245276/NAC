# -*- coding: utf-8 -*-
"""
Mechanism-evidence diagnostic for NAC (why look-ahead + momentum helps).

Measures, on real PGD-attacked CIFAR-10 images and real CLIP trajectories:

  1. Gradient predictiveness (NAC): is the look-ahead gradient a better
     predictor of the *next* position's gradient than the current-position
     gradient?   cos(g_look, g_next)  vs  cos(g_cur, g_next)
  2. Gradient autocorrelation along the counterattack trajectory:
     cos(g_t, g_{t+1}) for TTC and for NAC's current-position gradients.
     (Momentum benefit is known to scale with gradient autocorrelation.)
  3. Look-ahead distance: E[||mu * v_t||] in units of the step size alpha.
     Pure look-ahead (no momentum) probes at distance ~alpha; NAC probes
     at distance mu*||v|| >> alpha if velocity accumulates.
  4. Control: the same autocorrelation measurement along a *random-anchor*
     counterattack trajectory (anchor = random vector with the same norm as
     f(x)). Tests whether high autocorrelation is special to the adversarial
     anchor or trivially true of any fixed-anchor escape.

The trajectory update rules replicate nac_counterattack / the official TTC
exactly (same init, same clamps, same sign updates); gating/weighting does
not affect the trajectory and therefore is not applied.

Run:
  python scripts/gradient_alignment_diagnostic.py --max_batches 16
Output:
  results/mechanism_diagnostic/gradient_alignment.json (+ .log)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replace import clip                                    # noqa: E402
from models.prompters import TokenPrompter, NullPrompter    # noqa: E402
from func import clip_img_preprocessing                     # noqa: E402
from utils import (convert_models_to_fp32, load_val_dataset,    # noqa: E402
                   get_text_prompts_val)
from attacks import attack_pgd                              # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', type=str, default='vit_b32')
    p.add_argument('--test_set', default=['cifar10'], type=str, nargs='*')
    p.add_argument('--root', type=str, default='./data')
    p.add_argument('--dataset', type=str, default='tinyImageNet')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test_attack_type', type=str, default='pgd')
    p.add_argument('--test_eps', type=float, default=4)       # /255 applied later
    p.add_argument('--test_numsteps', type=int, default=10)
    p.add_argument('--test_stepsize', type=float, default=1)  # /255 applied later
    p.add_argument('--ttc_eps', type=float, default=4)        # /255 applied later
    p.add_argument('--ttc_stepsize', type=float, default=1)   # /255 applied later
    p.add_argument('--ttc_numsteps', type=int, default=4)
    p.add_argument('--nac_momentum', type=float, default=0.9)
    p.add_argument('--max_batches', type=int, default=16)
    p.add_argument('--victim_resume', type=str, default=None)
    p.add_argument('--method', type=str, default='null_patch')
    p.add_argument('--prompt_size', type=int, default=30)
    p.add_argument('--add_prompt_size', type=int, default=0)
    p.add_argument('--name', type=str, default='')
    p.add_argument('--cache', type=str, default='./cache')
    p.add_argument('--imagenet_root', type=str, default='./data/imagenet-100/imagenet_folder')
    p.add_argument('--tinyimagenet_root', type=str,
                   default='./data/tiny-imagenet-200/tiny-imagenet-200')
    p.add_argument('--outdir', type=str, default='./results/mechanism_diagnostic')
    return p.parse_args()


def per_sample_cos(a, b):
    a = a.detach().float().reshape(a.size(0), -1)
    b = b.detach().float().reshape(b.size(0), -1)
    return F.cosine_similarity(a, b, dim=1)  # [B]


def grad_at(model, prompter, prompt_token, X, X_anchor, pos, leaf):
    """Per-sample gradient of sum_i ||f(pos_i) - anchor_i||^2 wrt leaf."""
    reps = model.module.encode_image(prompter(clip_img_preprocessing(pos)), prompt_token)
    loss = (((reps - X_anchor) ** 2).sum(1)).sum()
    return torch.autograd.grad(loss, leaf)[0]


def run_trajectory(model, prompter, prompt_token, X, X_anchor, epsilon, alpha, K,
                   momentum=None):
    """Replicates the TTC/NAC trajectory with gradient probes.

    momentum=None  -> TTC trajectory (natural gradient at current position).
    momentum=float -> NAC trajectory (natural gradient at look-ahead position).

    Probes: gradient evaluated at each post-update position delta_{t+1}.
      TTC: cos(g_t, g_{t+1})                 -> trajectory autocorrelation
      NAC: cos(g_look_t, g_{t+1})            -> look-ahead predictiveness
           cos(g_cur_t, g_{t+1})             -> current-grad predictiveness
           (g_cur probe computed at every step position delta_t)
    """
    lower, upper = 0., 1.

    def clamp(X_, lo, hi):
        return torch.max(torch.min(X_, hi), lo)

    delta = torch.zeros_like(X)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower - X, upper - X)
    delta.requires_grad_(True)

    velocity = torch.zeros_like(X)
    rec = {'cos_cur_next': [], 'cos_look_next': [], 'cos_step_next': [],
           'v_pixel_ratio': [], 'v_pixel_ratio_p90': [], 'final_shift': None}

    with torch.amp.autocast('cuda'):
        for t in range(K):
            if momentum is None:
                # TTC: natural gradient at the current position
                g_nat = grad_at(model, prompter, prompt_token, X, X_anchor,
                                X + delta, delta)
                step = alpha * torch.sign(g_nat)
            else:
                # NAC: natural gradient at the look-ahead position
                look_ahead = X + delta + momentum * velocity
                g_nat = grad_at(model, prompter, prompt_token, X, X_anchor,
                                look_ahead, delta)
                velocity = momentum * velocity + alpha * torch.sign(g_nat)
                step = velocity

            # update position (same clamps as the authoritative implementations)
            d = delta.detach() + step
            d = torch.clamp(d, min=-epsilon, max=epsilon)
            d = clamp(d, lower - X, upper - X)

            # probe: gradient at the next position delta_{t+1}
            delta_next = d.clone().requires_grad_(True)
            g_next = grad_at(model, prompter, prompt_token, X, X_anchor,
                             X + delta_next, delta_next)

            if momentum is None:
                rec['cos_cur_next'].append(per_sample_cos(g_nat, g_next).cpu())
                # step-direction alignment: the taken step is alpha*sign(g_t)
                rec['cos_step_next'].append(
                    per_sample_cos(torch.sign(g_nat.detach().float()), g_next).cpu())
            else:
                rec['cos_look_next'].append(per_sample_cos(g_nat, g_next).cpu())
                # step-direction alignment: the taken step is v_{t+1}
                rec['cos_step_next'].append(
                    per_sample_cos(velocity.detach().float(), g_next).cpu())
                # current-position gradient at delta_t (probe)
                delta_cur = delta.detach().clone().requires_grad_(True)
                g_cur = grad_at(model, prompter, prompt_token, X, X_anchor,
                                X + delta_cur, delta_cur)
                rec['cos_cur_next'].append(per_sample_cos(g_cur, g_next).cpu())
                # per-pixel look-ahead distance of the NEXT step: mu*|v_{t+1}| / alpha
                vpix = (velocity.detach().float().abs() * momentum / alpha)
                rec['v_pixel_ratio'].append(vpix.reshape(vpix.size(0), -1).mean(dim=1).cpu())
                rec['v_pixel_ratio_p90'].append(
                    vpix.reshape(vpix.size(0), -1).quantile(0.9, dim=1).cpu())

            delta = delta_next  # continue from the probed position

        with torch.no_grad():
            final_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X + delta)), prompt_token)
            rec['final_shift'] = ((final_reps - X_anchor).norm(dim=-1)
                                  / X_anchor.norm(dim=-1)).cpu()

    return rec


def stack_mean(list_of_tensors):
    if not list_of_tensors:
        return float('nan')
    return torch.cat(list_of_tensors).mean().item()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.

    arch_map = {'vit_b32': 'ViT-B/32', 'vit_b16': 'ViT-B/16', 'RN50': 'RN50'}
    arch = arch_map.get(args.arch, args.arch)

    model, _ = clip.load(arch, device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    log_path = os.path.join(args.outdir, 'gradient_alignment.log')
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line)
        log_f.write(line + '\n')
        log_f.flush()

    log(f'arch={arch} K={args.ttc_numsteps} mu={args.nac_momentum} '
        f'eps_def={args.ttc_eps:.5f} alpha={args.ttc_stepsize:.5f} '
        f'max_batches={args.max_batches}')

    results = {'config': vars(args), 'datasets': {}}

    for dataset_name in args.test_set:
        val_dataset, val_loader = load_val_dataset(args, dataset_name)
        texts = get_text_prompts_val([val_dataset], [dataset_name])[0]
        text_tokens = clip.tokenize(texts).to(device)

        agg = {'ttc_autocorr': [], 'nac_autocorr': [], 'nac_look_next': [],
               'ttc_step_align': [], 'nac_step_align': [],
               'v_pixel': [], 'v_pixel_p90': [], 'rand_autocorr': [], 'n_samples': 0}

        n_done = 0
        for images, target in tqdm(val_loader, desc=f'[{dataset_name}]'):
            if n_done >= args.max_batches:
                break
            images = images.to(device)
            target = target.to(device)
            B = images.size(0)

            with torch.amp.autocast('cuda'):
                delta_atk = attack_pgd(
                    args, prompter, model, None, None, add_prompter, criterion,
                    images, target, args.test_stepsize, args.test_numsteps, 'l_inf',
                    text_tokens=text_tokens, epsilon=args.test_eps,
                    dataset_name=dataset_name
                )
                attacked = (images + delta_atk).detach()

            prompt_token = add_prompter()
            with torch.no_grad():
                X_ori_reps = model.module.encode_image(
                    prompter(clip_img_preprocessing(attacked)), prompt_token)
                rand_dir = torch.randn_like(X_ori_reps)
                rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
                rand_anchor = rand_dir * X_ori_reps.norm(dim=-1, keepdim=True)

            K = args.ttc_numsteps
            eps_def, alpha = args.ttc_eps, args.ttc_stepsize

            rec_ttc = run_trajectory(model, prompter, prompt_token, attacked,
                                     X_ori_reps, eps_def, alpha, K, momentum=None)
            rec_nac = run_trajectory(model, prompter, prompt_token, attacked,
                                     X_ori_reps, eps_def, alpha, K,
                                     momentum=args.nac_momentum)
            rec_rand = run_trajectory(model, prompter, prompt_token, attacked,
                                      rand_anchor, eps_def, alpha, K, momentum=None)

            agg['ttc_autocorr'] += rec_ttc['cos_cur_next']
            agg['nac_autocorr'] += rec_nac['cos_cur_next']
            agg['nac_look_next'] += rec_nac['cos_look_next']
            agg['ttc_step_align'] += rec_ttc['cos_step_next']
            agg['nac_step_align'] += rec_nac['cos_step_next']
            agg['v_pixel'] += rec_nac['v_pixel_ratio']
            agg['v_pixel_p90'] += rec_nac['v_pixel_ratio_p90']
            agg['rand_autocorr'] += rec_rand['cos_cur_next']
            agg['n_samples'] += B
            n_done += 1

            if n_done % 4 == 0:
                log(f'  batch {n_done}/{args.max_batches} | '
                    f'ttc_autocorr={stack_mean(agg["ttc_autocorr"]):.4f} | '
                    f'predictiveness_gain='
                    f'{stack_mean(agg["nac_look_next"]) - stack_mean(agg["nac_autocorr"]):+.4f}')

        ttc_autocorr = stack_mean(agg['ttc_autocorr'])
        nac_autocorr = stack_mean(agg['nac_autocorr'])
        nac_look_next = stack_mean(agg['nac_look_next'])
        ttc_step_align = stack_mean(agg['ttc_step_align'])
        nac_step_align = stack_mean(agg['nac_step_align'])
        v_pixel_mean = stack_mean(agg['v_pixel'])
        v_pixel_p90 = stack_mean(agg['v_pixel_p90'])
        rand_autocorr = stack_mean(agg['rand_autocorr'])

        summary = {
            'n_samples': agg['n_samples'],
            'ttc_gradient_autocorrelation': ttc_autocorr,
            'nac_current_gradient_autocorrelation': nac_autocorr,
            'nac_lookahead_gradient_predictiveness': nac_look_next,
            'predictiveness_gain_look_minus_cur': nac_look_next - nac_autocorr,
            'ttc_step_alignment_with_landing_gradient': ttc_step_align,
            'nac_step_alignment_with_landing_gradient': nac_step_align,
            'step_alignment_gain_nac_minus_ttc': nac_step_align - ttc_step_align,
            'lookahead_pixel_distance_mean_over_alpha': v_pixel_mean,
            'lookahead_pixel_distance_p90_over_alpha': v_pixel_p90,
            'random_anchor_autocorrelation_control': rand_autocorr,
        }

        log(f'--- {dataset_name} summary (n={agg["n_samples"]}) ---')
        log(f'  TTC gradient autocorrelation cos(g_t, g_t+1)      : {ttc_autocorr:.4f}')
        log(f'  NAC current-grad autocorrelation                  : {nac_autocorr:.4f}')
        log(f'  NAC look-ahead grad -> next-grad cosine           : {nac_look_next:.4f}')
        log(f'  PREDICTIVENESS GAIN (look-ahead - current)        : {nac_look_next - nac_autocorr:+.4f}')
        log(f'  TTC step-alignment with landing gradient          : {ttc_step_align:.4f}')
        log(f'  NAC step-alignment with landing gradient          : {nac_step_align:.4f}')
        log(f'  STEP ALIGNMENT GAIN (NAC - TTC)                   : {nac_step_align - ttc_step_align:+.4f}')
        log(f'  Look-ahead pixel distance mean |mu*v|/alpha       : {v_pixel_mean:.2f} (p90 {v_pixel_p90:.2f})')
        log(f'  Random-anchor control autocorrelation             : {rand_autocorr:.4f}')

        results['datasets'][dataset_name] = summary

    json_path = os.path.join(args.outdir, 'gradient_alignment.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    log(f'saved: {json_path}')
    log_f.close()


if __name__ == '__main__':
    main()
