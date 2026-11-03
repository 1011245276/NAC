# -*- coding: utf-8 -*-
"""Phase 0.5 diagnostics (2026-11-03).

mode=grid      : alpha x eps grid  -> adv acc, gate s/frac>tau, TTC/NAC defended acc
mode=structure : fixed budget, controlled perturbation STRUCTURE (spatial clustering
                 / smoothness) at matched L_inf and matched L1 -> does s track density
                 or structure?

All numbers land in results/_diag_grid/<mode>.json. Nothing existing is overwritten.
"""
import sys, os, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

import torch
import torch.nn.functional as F
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (load_val_dataset, get_text_prompts_val,
                                 clip_img_preprocessing, multiGPU_CLIP,
                                 ttc_counterattack, nac_counterattack)
from utils import convert_models_to_fp32

device = "cuda"

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='grid')
ap.add_argument('--eps', default='1,2,4')
ap.add_argument('--nb', type=int, default=8)          # number of 32-image batches
ap.add_argument('--steps', type=int, default=10)
ap.add_argument('--methods', default='ttc,nac')
ap.add_argument('--tag', default='main')
ap.add_argument('--init', default='zero')             # zero|random attack init
my, _ = ap.parse_known_args()

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
TAU = args.tau_thres
METHODS = [m.strip() for m in my.methods.split(',') if m.strip()]


def load_images(nb):
    ims, tgs = [], []
    for i, (im, tg) in enumerate(val_loader):
        if i >= nb:
            break
        ims.append(im); tgs.append(tg)
    return torch.cat(ims).to(device), torch.cat(tgs).to(device)


def make_text():
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    return clip.tokenize(texts).to(device)


def acc_of(x, tt, tg):
    with torch.no_grad():
        out = multiGPU_CLIP(None, None, None, model, prompter(clip_img_preprocessing(x)),
                            text_tokens=tt, prompt_token=None, dataset_name='cifar10')[0]
    return float((out.argmax(1) == tg).float().mean() * 100)


def gate(x, draws=2, seed_base=9100):
    """s = ||f(x+d0)-f(x)||/||f(x)||, runtime convention (clip_img_preprocessing)."""
    ss = []
    with torch.no_grad():
        f = model.module.encode_image
        pt = add_prompter()
        ori = f(prompter(clip_img_preprocessing(x)), pt)
        on = ori.norm(dim=-1)
        for j in range(draws):
            torch.manual_seed(seed_base + j)
            d0 = torch.zeros_like(x).uniform_(-args.ttc_eps, args.ttc_eps)
            at = f(prompter(clip_img_preprocessing(torch.clamp(x + d0, 0, 1))), pt)
            ss.append(((at - ori).norm(dim=-1) / on))
    s = torch.stack(ss).mean(0)
    return float(s.mean()), float((s > TAU).float().mean()), float(s.max())


def defend(x, tt, tg, method, seed=1234):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    kw = dict(alpha=args.ttc_stepsize, attack_iters=2, norm='l_inf',
              epsilon=args.ttc_eps, tau_thres=TAU, beta=args.beta, clip_visual=None)
    with torch.enable_grad():
        if method == 'ttc':
            d = ttc_counterattack(model, x, prompter, add_prompter, **kw)
        else:
            d = nac_counterattack(model, x, prompter, add_prompter,
                                  nac_momentum=args.nac_momentum, **kw)
    d = d.detach()
    return acc_of(torch.clamp(x + d, 0, 1), tt, tg), float(d.abs().mean() * 255)


def pgd(images, tg, tt, eps, alpha, steps, latent=None, init='zero'):
    """latent=None -> full-res PGD; else PGD on a latent grid upsampled to 224."""
    if latent is None:
        delta = torch.zeros_like(images)
    else:
        delta = torch.zeros((images.shape[0], 3, latent, latent), device=images.device)
    if init == 'random':
        delta = torch.zeros_like(delta).uniform_(-eps, eps)
    for _ in range(steps):
        dg = delta.detach().requires_grad_(True)
        d = dg if latent is None else torch.clamp(
            F.interpolate(dg, size=images.shape[-1], mode='bilinear', align_corners=False),
            -eps, eps)
        xg = torch.clamp(images + d, 0, 1)
        out = multiGPU_CLIP(None, None, None, model, prompter(clip_img_preprocessing(xg)),
                            text_tokens=tt, prompt_token=None, dataset_name='cifar10')[0]
        loss = F.cross_entropy(out, tg)
        g = torch.autograd.grad(loss, dg)[0]
        delta = torch.clamp(delta + alpha * torch.sign(g), -eps, eps).detach()
    d = delta if latent is None else torch.clamp(
        F.interpolate(delta, size=images.shape[-1], mode='bilinear', align_corners=False),
        -eps, eps)
    return d


def scatter_mask(x, frac, blocky=False, block=8, seed=0):
    """exactly `frac` of the domain active (blocky: 8x8 blocks at 224 res)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    B, _, H, W = x.shape
    if blocky:
        h, w = H // block, W // block
        n = h * w
        k = int(round(frac * n))
        v = torch.rand(B, n, generator=g)
        idx = v.argsort(dim=1)[:, :k]
        m = torch.zeros(B, n)
        m.scatter_(1, idx, 1.0)
        m = m.view(B, 1, h, w).repeat_interleave(block, 2).repeat_interleave(block, 3)
    else:
        n = H * W
        k = int(round(frac * n))
        v = torch.rand(B, n, generator=g)
        idx = v.argsort(dim=1)[:, :k]
        m = torch.zeros(B, n)
        m.scatter_(1, idx, 1.0)
        m = m.view(B, 1, H, W)
    sgn = torch.sign(torch.randn(B, 1, H, W, generator=g))
    return (m * sgn).to(x.device)


def report(tag, x_adv, x_base, tt, tg, row):
    row.update(adv_acc=acc_of(x_adv, tt, tg),
               delta_linf=float((x_adv - x_base).abs().max() * 255),
               delta_absmean=float((x_adv - x_base).abs().mean() * 255))
    s_mu, f_tau, s_max = gate(x_adv)
    row.update(s_mean=s_mu, frac_gt_tau=f_tau, s_max=s_max)
    for m in METHODS:
        a, dm = defend(x_adv, tt, tg, m)
        row[f'def_{m}'] = a
        row[f'ddef_{m}'] = dm
    print(f"{tag:<34} adv={row['adv_acc']:6.2f}%  s={s_mu:.3f}  >tau={f_tau:.3f}  "
          f"|d|inf={row['delta_linf']:4.2f} |d|1={row['delta_absmean']:5.2f}  "
          + "  ".join(f"{m}={row[f'def_{m}']:.2f}% (D={row[f'ddef_{m}']:.2f})"
                      for m in METHODS))
    return row


def main():
    images, target = load_images(my.nb)
    tt = make_text()
    out = dict(mode=my.mode, tag=my.tag, n=int(images.shape[0]), tau=TAU,
               steps=my.steps, init=my.init, methods=METHODS, rows=[])
    print(f"[{my.mode}] N={images.shape[0]} tau={TAU} K={my.steps} "
          f"def budget={args.ttc_eps*255:.0f}/255 step={args.ttc_stepsize*255:.0f}/255")

    if my.mode == 'grid':
        for eps255 in [float(v) for v in my.eps.split(',')]:
            eps = eps255 / 255.
            for lbl, alpha in [('a=e', eps), ('a=e/2', eps / 2), ('a=e/4', eps / 4),
                               ('a=e/10', eps / 10.)]:
                d = pgd(images, target, tt, eps, alpha, my.steps, init=my.init)
                xa = torch.clamp(images + d, 0, 1)
                row = report(f"eps={eps255:g} {lbl}", xa, images, tt, target,
                             dict(eps=eps255, alpha_255=alpha * 255, family=lbl, extra={}))
                out['rows'].append(row)
    else:
        eps = float(my.eps.split(',')[0]) / 255.
        eps255 = eps * 255
        # adversarial, sharp vs smooth
        d = pgd(images, target, tt, eps, eps / 10., my.steps)
        xa = torch.clamp(images + d, 0, 1)
        out['rows'].append(report(f"eps={eps255:g} pgd_sharp", xa, images, tt, target,
                                  dict(eps=eps255, family='pgd_sharp', extra={})))
        d = pgd(images, target, tt, eps, eps / 10., my.steps, latent=28)
        xa = torch.clamp(images + d, 0, 1)
        out['rows'].append(report(f"eps={eps255:g} pgd_smooth28", xa, images, tt, target,
                                  dict(eps=eps255, family='pgd_smooth28', extra={})))
        # non-adversarial, matched L1 and L_inf: structure only
        for frac in [0.25, 0.5, 0.75]:
            for blocky in [False, True]:
                m = scatter_mask(images, frac, blocky=blocky, seed=int(frac * 100))
                xa = torch.clamp(images + eps * m, 0, 1)
                out['rows'].append(report(
                    f"eps={eps255:g} scat{frac:g}{'b' if blocky else 'r'}", xa, images, tt, target,
                    dict(eps=eps255, family='scatter',
                         extra=dict(frac=frac, blocky=blocky, l1=frac * eps255))))

    os.makedirs(os.path.join('results', '_diag_grid'), exist_ok=True)
    p = os.path.join('results', '_diag_grid', f"{my.mode}_{my.tag}.json")
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=2)
    print("saved ->", p)


if __name__ == '__main__':
    main()
