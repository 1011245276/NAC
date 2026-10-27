# -*- coding: utf-8 -*-
"""Diagnostic for the E4-S3 identical-clean-vector phenomenon (2026-10-27).

The full clean paired validation recorded byte-identical correctness vectors
for TTC and NAC.  Hypothesis: on CLEAN images the counterattack trajectory
contributes nothing beyond the random initialization (gating collapses the
output to the random-init delta0, which depends only on the RNG stream), so
two defenses with the same seed produce identical Delta and hence identical
predictions.  This script verifies the hypothesis empirically:

  for a batch of clean CIFAR-10 images and for the same images with a strong
  random perturbation (high feature deviation):
    - run ttc_counterattack and nac_counterattack under identical RNG;
    - compare Delta_ttc vs Delta_nac;
    - compare each Delta against the random init delta0 drawn from the same
      seed (identical  =>  trajectory suppressed, output == delta0).

Read-only w.r.t. results; prints findings to stdout.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

import torch
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (load_val_dataset, get_text_prompts_val,
                                 ttc_counterattack, nac_counterattack,
                                 clip_img_preprocessing, multiGPU_CLIP)
from utils import convert_models_to_fp32

device = "cuda"
clip_load = clip.load

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

model, _ = clip_load('ViT-B/32', device, jit=False, prompt_len=0)
for p in model.parameters():
    p.requires_grad = False
convert_models_to_fp32(model)
model = torch.nn.DataParallel(model)
model.eval()
prompter = torch.nn.DataParallel(NullPrompter()).cuda()
add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

val_dataset, val_loader = load_val_dataset(args, 'cifar10')


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


def run(name, x, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kw = mk(name, x)
    fn = ttc_counterattack if name == 'ttc' else nac_counterattack
    with torch.enable_grad():
        return fn(**kw).detach()


def delta0_like(x, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.clamp(torch.zeros_like(x).uniform_(-args.ttc_eps, args.ttc_eps),
                       -args.ttc_eps, args.ttc_eps)


def probe(tag, x):
    seed = 1234
    dt = run('ttc', x, seed)
    dn = run('nac', x, seed)
    d0 = delta0_like(x, seed)
    print(f"[{tag}]")
    print(f"  max|Delta_ttc - Delta_nac| = {(dt - dn).abs().max().item():.6f}")
    print(f"  max|Delta_ttc - d0|        = {(dt - d0).abs().max().item():.6f}")
    print(f"  max|Delta_nac - d0|        = {(dn - d0).abs().max().item():.6f}")
    print(f"  mean|Delta_ttc|={dt.abs().mean().item():.6f}  "
          f"mean|Delta_nac|={dn.abs().mean().item():.6f}  "
          f"mean|d0|={d0.abs().mean().item():.6f}")
    # feature-deviation ratio at the random init (the gating signal)
    with torch.no_grad():
        f = model.module.encode_image
        pt = add_prompter()
        import torch.nn.functional as F  # noqa
        ori = f(prompter(clip_img_preprocessing(x)), pt)
        att = f(prompter(clip_img_preprocessing(torch.clamp(x + d0, 0, 1))), pt)
        ratio = ((att - ori).norm(dim=-1) / ori.norm(dim=-1))
    print(f"  diff_ratio(delta0): mean={ratio.mean().item():.4f} "
          f"max={ratio.max().item():.4f}  (tau_thres={args.tau_thres})")


def main():
    images, target = next(iter(val_loader))
    images, target = images.to(device), target.to(device)
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)
    probe("clean", images)
    strong = torch.clamp(images + 8 / 255. * torch.randn_like(images).sign(), 0, 1)
    probe("x + 8/255 sign-random (high-deviation proxy)", strong)

    # adversarial probe: quick PGD-10 on UNDEFENDED CLIP (CE loss), then the
    # same ttc/nac comparison on x_adv -- decides whether the gating also
    # suppresses the method-specific trajectory on attacked inputs (which
    # would make the deployed defense a pure random-noise defense there).
    x_adv = images.clone().detach().requires_grad_(True)
    for _s in range(10):
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(torch.clamp(images + x_adv, 0, 1))),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
        loss = torch.nn.functional.cross_entropy(out, target)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.clamp(x_adv + args.ttc_stepsize * torch.sign(g),
                            -args.ttc_eps, args.ttc_eps).detach().requires_grad_(True)
    x_adv = torch.clamp(images + x_adv.detach(), 0, 1)
    with torch.no_grad():
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(x_adv)),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
        print(f"[pgd-10 probe] undefended adv acc = "
              f"{(out.argmax(1) == target).float().mean().item() * 100:.2f}%")
    probe("PGD-10 adversarial images", x_adv)


if __name__ == '__main__':
    main()
