# -*- coding: utf-8 -*-
"""Audit control (2026-11-03): the step-size grid TRAE's plan missed.

TRAE's verify_dense_vs_defense.py compared alpha=1/255 (paper protocol) vs
alpha=eps/10 (convergent PGD) and concluded "standard PGD collapses the
gate". The actually-standard Madry convention is alpha = eps/4. This script
fills the missing cells: alpha in {eps, eps/2, eps/4, eps/10} at eps in
{1/255, 2/255, 4/255}, 128 CIFAR-10 images, real deployed defenses.

Read-only w.r.t. results; writes results/_diag_s_curve/stepsize_grid.json.
"""
import sys, os, json
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
N_BATCHES = 4
TAU = args.tau_thres
N_STEPS = 10


def classify(x, text_tokens, target):
    with torch.no_grad():
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model, prompter(clip_img_preprocessing(x)),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
    return out.argmax(1) == target


def gate_probe(x, n_draws=3, seed_base=8100):
    per_draw = []
    with torch.no_grad():
        f = model.module.encode_image
        pt = add_prompter()
        ori = f(prompter(clip_img_preprocessing(x)), pt)
        ori_n = ori.norm(dim=-1)
        for j in range(n_draws):
            torch.manual_seed(seed_base + j)
            torch.cuda.manual_seed_all(seed_base + j)
            d0 = torch.zeros_like(x).uniform_(-args.ttc_eps, args.ttc_eps)
            att = f(prompter(clip_img_preprocessing(torch.clamp(x + d0, 0, 1))), pt)
            s = (att - ori).norm(dim=-1) / ori_n
            per_draw.append(s)
    s_mu = torch.stack(per_draw).mean(0)
    return dict(s_mean=float(s_mu.mean()),
                frac_gt_tau=float((s_mu > TAU).float().mean()))


def build_attack(images, target, text_tokens, eps, alpha):
    delta = torch.zeros_like(images)
    for _ in range(N_STEPS):
        delta_g = delta.detach().requires_grad_(True)
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(torch.clamp(images + delta_g, 0, 1))),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
        loss = F.cross_entropy(out, target)
        g = torch.autograd.grad(loss, delta_g)[0]
        delta = torch.clamp(delta + alpha * torch.sign(g), -eps, eps).detach()
    return delta


def defend(x, text_tokens, target, method, seed, iters=2):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kw = dict(alpha=args.ttc_stepsize, attack_iters=iters, norm='l_inf',
              epsilon=args.ttc_eps, tau_thres=TAU, beta=args.beta,
              clip_visual=None)
    with torch.enable_grad():
        if method == 'ttc':
            d = ttc_counterattack(model, x, prompter, add_prompter, **kw)
        else:
            d = nac_counterattack(model, x, prompter, add_prompter,
                                  nac_momentum=args.nac_momentum,
                                  clamp_lookahead=False, **kw)
    return float(classify(torch.clamp(x + d.detach(), 0, 1), text_tokens, target)
                 .float().mean() * 100), float(d.detach().abs().mean() * 255)


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

    print(f"N={images.shape[0]} tau={TAU} defense eps={args.ttc_eps*255:.0f}/255 "
          f"step={args.ttc_stepsize*255:.0f}/255 iters=2")

    results = dict(clean=dict(
        acc=float(classify(images, text_tokens, target).float().mean() * 100),
        gate=gate_probe(images),
        defended={m: defend(images, text_tokens, target, m, 1234)[0]
                  for m in ['ttc', 'nac']}), runs=[])

    for eps in [1 / 255., 2 / 255., 4 / 255.]:
        for ratio_name, ratio in [('eps', 1.0), ('eps/2', 0.5),
                                  ('eps/4', 0.25), ('eps/10', 0.1)]:
            alpha = eps * ratio
            accs, gates = [], []
            defs, dms = {}, {}
            for c0 in range(0, images.shape[0], 32):
                im = images[c0:c0 + 32]
                tg = target[c0:c0 + 32]
                delta = build_attack(im, tg, text_tokens, eps, alpha)
                x_adv = torch.clamp(im + delta, 0, 1)
                accs.append(float(classify(x_adv, text_tokens, tg).float().mean() * 100))
                gates.append(gate_probe(x_adv))
                for m in ['ttc', 'nac']:
                    a, dm = defend(x_adv, text_tokens, tg, m, 1234)
                    defs.setdefault(m, []).append(a)
                    dms.setdefault(m, []).append(dm)
            gate = dict(s_mean=sum(g['s_mean'] for g in gates) / len(gates),
                        frac_gt_tau=sum(g['frac_gt_tau'] for g in gates) / len(gates))
            row = dict(eps=float(eps), alpha_ratio=ratio_name, alpha=float(alpha),
                       delta_absmean=float(delta.abs().mean() * 255),
                       adv_acc=sum(accs) / len(accs), gate=gate,
                       defended={m: dict(acc=sum(defs[m]) / len(defs[m]),
                                         delta_absmean=sum(dms[m]) / len(dms[m]))
                                 for m in ['ttc', 'nac']})
            results['runs'].append(row)
            print(f"eps={eps*255:.0f}/255 alpha={ratio_name} "
                  f"(={alpha*255:.3f}/255) |d|mean={row['delta_absmean']:.2f} "
                  f"adv={row['adv_acc']:.2f}% s={gate['s_mean']:.3f} "
                  f"frac>tau={gate['frac_gt_tau']:.3f} | "
                  f"ttc={row['defended']['ttc']['acc']:.2f} "
                  f"nac={row['defended']['nac']['acc']:.2f} "
                  f"(|D|={row['defended']['ttc']['delta_absmean']:.2f})")

    os.makedirs(os.path.join('results', '_diag_s_curve'), exist_ok=True)
    with open(os.path.join('results', '_diag_s_curve', 'stepsize_grid.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved -> results/_diag_s_curve/stepsize_grid.json")


if __name__ == '__main__':
    main()