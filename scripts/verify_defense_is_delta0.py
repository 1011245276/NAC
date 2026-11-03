# -*- coding: utf-8 -*-
"""Definitive confirmation: is TTC/NAC's Delta output EXACTLY equal to delta0
when every sample has s > tau (i.e. gate fully suppressed)?

Verifies that the defense really degenerates to random noise, not just that
accuracy drops.  Compares Delta_def against an independent fresh delta0.
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
    'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '16',
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
for p in model.parameters(): p.requires_grad = False
convert_models_to_fp32(model)
model = torch.nn.DataParallel(model).eval()
prompter = torch.nn.DataParallel(NullPrompter()).cuda()
add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
_, val_loader = load_val_dataset(args, 'cifar10')
TAU = args.tau_thres

# Grab ONE batch
images, target = next(iter(val_loader))
images, target = images.to(device), target.to(device)
texts = get_text_prompts_val([], ['cifar10'])[0]
text_tokens = clip.tokenize(texts).to(device)

# Build dense attack, eps=1/255 alpha=0.1/255, 10 steps
def build(images, target, text_tokens, eps, alpha, n_steps=10):
    delta = torch.zeros_like(images)
    for _ in range(n_steps):
        dg = delta.detach().requires_grad_(True)
        out, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(torch.clamp(images + dg, 0, 1))),
            text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
        loss = F.cross_entropy(out, target)
        g = torch.autograd.grad(loss, dg)[0]
        delta = torch.clamp(delta + alpha * torch.sign(g), -eps, eps).detach()
    return torch.clamp(images + delta, 0, 1)

x_adv = build(images, target, text_tokens, 1/255., 0.1/255., 10)
with torch.no_grad():
    out = multiGPU_CLIP(None, None, None, model,
                        prompter(clip_img_preprocessing(x_adv)),
                        text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')[0]
print(f"dense attack: adv acc = {(out.argmax(1) == target).float().mean().item()*100:.2f}%")

# Compute gate s for this x_adv under 5 fresh draws
with torch.no_grad():
    f = model.module.encode_image
    pt = add_prompter()
    ori = f(prompter(clip_img_preprocessing(x_adv)), pt)
    ori_n = ori.norm(dim=-1)
    for j in range(5):
        torch.manual_seed(777 + j)
        d0j = torch.zeros_like(x_adv).uniform_(-args.ttc_eps, args.ttc_eps)
        att = f(prompter(clip_img_preprocessing(torch.clamp(x_adv + d0j, 0, 1))), pt)
        s = (att - ori).norm(dim=-1) / ori_n
        print(f"  draw {j}: s_mean={s.mean():.4f}  frac>tau={float((s > TAU).float().mean()):.3f}")

# Now run TTC defense and extract Delta_output
# Use THE SAME seed (1234) and verify Delta_output == d0_seed_1234 exactly
torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
d0_ref = torch.zeros_like(x_adv).uniform_(-args.ttc_eps, args.ttc_eps)

kw = dict(alpha=args.ttc_stepsize, attack_iters=2, norm='l_inf',
          epsilon=args.ttc_eps, tau_thres=TAU, beta=args.beta, clip_visual=None)
with torch.enable_grad():
    Delta_ttc = ttc_counterattack(model, x_adv, prompter, add_prompter, **kw)
    Delta_nac = nac_counterattack(model, x_adv, prompter, add_prompter,
                                  nac_momentum=args.nac_momentum,
                                  clamp_lookahead=False, **kw)

print(f"\nTTC Delta vs delta0_ref (seed 1234):")
print(f"  max|Delta - delta0| = {(Delta_ttc - d0_ref).abs().max().item():.8f}")
print(f"  mean|Delta - delta0| = {(Delta_ttc - d0_ref).abs().mean().item():.8f}")

print(f"\nNAC Delta vs delta0_ref (seed 1234):")
print(f"  max|Delta - delta0| = {(Delta_nac - d0_ref).abs().max().item():.8f}")
print(f"  mean|Delta - delta0| = {(Delta_nac - d0_ref).abs().mean().item():.8f}")

# And the output image is exactly x_adv + delta0
with torch.no_grad():
    out_adv0, _, _, _ = multiGPU_CLIP(
        None, None, None, model,
        prompter(clip_img_preprocessing(torch.clamp(x_adv + d0_ref, 0, 1))),
        text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
    out_ttc, _, _, _ = multiGPU_CLIP(
        None, None, None, model,
        prompter(clip_img_preprocessing(torch.clamp(x_adv + Delta_ttc, 0, 1))),
        text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')

acc0 = float((out_adv0.argmax(1) == target).float().mean() * 100)
acc_ttc = float((out_ttc.argmax(1) == target).float().mean() * 100)
print(f"\nx_adv + delta0_ref accuracy: {acc0:.2f}%")
print(f"x_adv + Delta_ttc    accuracy: {acc_ttc:.2f}%")
print(f"logits identical? {(out_adv0 - out_ttc).abs().max().item():.6f}")
