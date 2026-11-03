# -*- coding: utf-8 -*-
"""Diagnostic: how does the gating statistic s(x) evolve along a PGD trajectory?

Question this answers (5-minute check, no new attack):
  As a standard CE-maximizing PGD moves a CLEAN image toward an adversarial one,
  does s(x_t) ever rise (rebound) toward tau, or does it fall monotonically?
  Also: among misclassified samples, what fraction already have s > tau?

Read-only; writes only a JSON under results/_diag_s_curve/.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

import torch
import torch.nn.functional as F
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (load_val_dataset, get_text_prompts_val,
                                 clip_img_preprocessing, multiGPU_CLIP)
from utils import convert_models_to_fp32

device = "cuda"
sys.argv = [
    'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
    '--test_set', 'cifar10', '--test_attack_type', 'pgd', '--test_eps', '4.0',
    '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
    '--root', './data', '--dataset', 'cifar10', '--seed', '0', '--num_seeds', '1',
    '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
    '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
    '--learnable_tau', '0.155', '--temperature', '75.0',
    '--counterattack', 'ttc', '--outdir', os.path.join('results', '_diag_gate'),
]
from nac_fair_experiment import parse_options
args = parse_options()
args.ttc_eps = args.ttc_eps / 255.
args.ttc_stepsize = args.ttc_stepsize / 255.
args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
for p in model.parameters():
    p.requires_grad = False
convert_models_to_fp32(model)
model = torch.nn.DataParallel(model)
model.eval()
prompter = torch.nn.DataParallel(NullPrompter()).cuda()
add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

val_dataset, val_loader = load_val_dataset(args, 'cifar10')

N_BATCHES = 4          # 128 images: preliminary proportion only
TAU = args.tau_thres   # 0.2
SEED_D0 = 1234


def gate_s(x, d0):
    """s(x) = ||f(x+d0)-f(x)|| / ||f(x)||, d0 fixed per sample."""
    with torch.no_grad():
        f = model.module.encode_image
        pt = add_prompter()
        ori = f(prompter(clip_img_preprocessing(x)), pt)
        att = f(prompter(clip_img_preprocessing(torch.clamp(x + d0, 0, 1))), pt)
        return (att - ori).norm(dim=-1) / ori.norm(dim=-1)


def accuracy(x, text_tokens, target):
    with torch.no_grad():
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(x)),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
        return (out.argmax(1) == target)


def run_epsilon(images, target, text_tokens, d0, eps, alpha, n_steps=10):
    delta = torch.zeros_like(images)
    rows = []
    for t in range(n_steps + 1):
        x_t = torch.clamp(images + delta, 0, 1)
        s = gate_s(x_t, d0)
        corr = accuracy(x_t, text_tokens, target)
        mis = ~corr
        gt_tau = s > TAU
        rows.append(dict(
            step=t,
            eps=float(eps),
            alpha=float(alpha),
            s_mean=float(s.mean()),
            s_max=float(s.max()),
            frac_gt_tau=float(gt_tau.float().mean()),
            acc=float(corr.float().mean() * 100),
            n_mis=int(mis.sum()),
            frac_mis_and_gt_tau=float((mis & gt_tau).float().sum() / max(int(mis.sum()), 1)),
            delta_linf=float(delta.abs().max()),
            delta_absmean=float(delta.abs().mean() * 255),
        ))
        if t == n_steps:
            break
        delta_g = delta.detach().requires_grad_(True)
        x_g = torch.clamp(images + delta_g, 0, 1)
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(x_g)),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
        loss = F.cross_entropy(out, target)
        g = torch.autograd.grad(loss, delta_g)[0]
        delta = torch.clamp(delta + alpha * torch.sign(g), -eps, eps).detach()
    return rows, delta.detach()


def eot_gate_check(images, target, text_tokens, delta, n_draws=5, seed_base=7000):
    """Does the gate decision survive re-sampling the defense's random delta0?"""
    out = dict(s_mean=[], frac_gt_tau=[], acc=[])
    x = torch.clamp(images + delta, 0, 1)
    for j in range(n_draws):
        torch.manual_seed(seed_base + j)
        torch.cuda.manual_seed_all(seed_base + j)
        d0 = torch.zeros_like(images).uniform_(-args.ttc_eps, args.ttc_eps)
        s = gate_s(x, d0)
        out['s_mean'].append(float(s.mean()))
        out['frac_gt_tau'].append(float((s > TAU).float().mean()))
    corr = accuracy(x, text_tokens, target)
    out['acc'] = float(corr.float().mean() * 100)
    out['frac_gt_tau_min'] = min(out['frac_gt_tau'])
    out['frac_gt_tau_max'] = max(out['frac_gt_tau'])
    out['s_mean_min'] = min(out['s_mean'])
    out['s_mean_max'] = max(out['s_mean'])
    return out


def main():
    images_all, target_all = [], []
    for i, (im, tg) in enumerate(val_loader):
        if i >= N_BATCHES:
            break
        images_all.append(im)
        target_all.append(tg)
    images = torch.cat(images_all).to(device)
    target = torch.cat(target_all).to(device)
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)

    torch.manual_seed(SEED_D0)
    torch.cuda.manual_seed_all(SEED_D0)
    d0 = torch.zeros_like(images).uniform_(-args.ttc_eps, args.ttc_eps)

    all_rows = []
    protocols = [
        ('fixed_1_255', lambda eps: 1 / 255.),      # matches the real experiment
        ('convergent_eps_10', lambda eps: eps / 10.),  # never saturates
    ]
    for tag, alpha_fn in protocols:
        for eps in [1 / 255., 2 / 255., 4 / 255.]:
            alpha = alpha_fn(eps)
            rows, delta = run_epsilon(images, target, text_tokens, d0, eps, alpha)
            for r in rows:
                r['protocol'] = tag
            all_rows.extend(rows)
            print(f"\n=== [{tag}] eps_att = {eps*255:.0f}/255 "
                  f"(alpha={alpha*255:.3f}/255, tau={TAU}, N={images.shape[0]}) ===")
            print("step  s_mean  s_max  frac>tau  acc%   mis&>tau  |d|inf |d|mean")
            for r in rows:
                print(f"{r['step']:>4}  {r['s_mean']:.4f}  {r['s_max']:.4f}  "
                      f"{r['frac_gt_tau']:.3f}     {r['acc']:.2f}   "
                      f"{r['frac_mis_and_gt_tau']:.3f} (n_mis={r['n_mis']}) "
                      f"{r['delta_linf']*255:.2f} {r['delta_absmean']:.2f}")
            eot = eot_gate_check(images, target, text_tokens, delta)
            print(f"  [final-step, 5 fresh delta0 draws] s_mean "
                  f"{eot['s_mean_min']:.3f}-{eot['s_mean_max']:.3f} | frac>tau "
                  f"{eot['frac_gt_tau_min']:.3f}-{eot['frac_gt_tau_max']:.3f} | "
                  f"acc={eot['acc']:.2f}%")
            all_rows.append(dict(protocol=tag, eps=float(eps), eot=eot, step='final_eot'))

    os.makedirs(os.path.join('results', '_diag_s_curve'), exist_ok=True)
    with open(os.path.join('results', '_diag_s_curve', 's_curve_cifar10.json'), 'w') as fh:
        json.dump(dict(n_images=int(images.shape[0]), tau=TAU, seed_d0=SEED_D0,
                       protocols=[t for t, _ in protocols], rows=all_rows), fh, indent=2)
    print("\nsaved -> results/_diag_s_curve/s_curve_cifar10.json")


if __name__ == '__main__':
    main()
