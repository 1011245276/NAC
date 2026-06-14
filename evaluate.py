#!/usr/bin/env python3
"""
NAC: Lightweight single-dataset evaluation.
Uses the same counterattack implementations as nac_fair_experiment.py
(with proper tau-threshold gating) for quick testing.

Usage:
  python evaluate.py --dataset cifar10 --method nac --attack_eps 4
  python evaluate.py --dataset STL10 --method ttc --attack_eps 1
"""
import argparse, os, sys, random
import numpy as np
from tqdm import tqdm
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import load_val_dataset, get_text_prompts_val, AverageMeter, accuracy, convert_models_to_fp32
from attacks import attack_pgd
from func import clip_img_preprocessing, multiGPU_CLIP

# Import proper counterattack functions (WITH tau-threshold gating)
from nac_fair_experiment import (
    nac_counterattack,
    momentum_counterattack,
    pure_la_counterattack,
    adam_counterattack,
    doc_counterattack,
    lbfgs_counterattack,
)

# Standalone TTC (original implementation, with tau gating)
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleArgs:
    """Minimal args object for functions that need it (DOC)."""
    def __init__(self, learnable_tau=0.155, temperature=70.0):
        self.learnable_tau = learnable_tau
        self.temperature = temperature


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--method', default='nac',
                   choices=['ttc', 'nac', 'momentum', 'pure_la', 'adam', 'doc', 'lbfgs'])
    p.add_argument('--attack_eps', type=int, default=4, help='Attack budget in /255')
    p.add_argument('--root', default='./data')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--nac_momentum', type=float, default=0.9)
    p.add_argument('--defense_steps', type=int, default=2, help='Counterattack steps (K)')
    p.add_argument('--defense_eps', type=int, default=4, help='Defense budget in /255')
    args = p.parse_args()

    # Ensure required paths exist (needed by load_val_dataset)
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')
    args.cache = './cache'
    args.evaluate = True
    args.num_workers = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load model
    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).eval()

    # Load dataset
    ds, loader = load_val_dataset(args, args.dataset)
    texts = get_text_prompts_val([ds], [args.dataset],
                                 template='a photo of a {}.', force_template=True)[0]
    text_tokens = clip.tokenize(texts).to(device)

    prompter = NullPrompter()
    add_prompter = TokenPrompter(0)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    # Convert from /255 to [0,1]
    eps_att = args.attack_eps / 255.
    alpha_att = 1 / 255.
    eps_def = args.defense_eps / 255.
    alpha_def = 1 / 255.

    top1_clean = AverageMeter('Clean', ':.2f')
    top1_adv = AverageMeter('Adv', ':.2f')
    top1_def = AverageMeter('Def', ':.2f')

    doc_args = SimpleArgs()

    for images, targets in tqdm(loader, desc=args.dataset):
        images, targets = images.to(device), targets.to(device)

        # Clean accuracy
        with torch.no_grad():
            out = multiGPU_CLIP(None, None, None, model,
                                clip_img_preprocessing(images),
                                text_tokens, prompt_token=None,
                                dataset_name=args.dataset)
            top1_clean.update(accuracy(out[0], targets, topk=(1,))[0].item(), images.size(0))

        # Generate adversarial attack (PGD-10)
        delta_atk = attack_pgd(None, prompter, model, None, None, add_prompter,
                               criterion, images, targets, alpha_att, 10, 'l_inf',
                               text_tokens=text_tokens, epsilon=eps_att,
                               dataset_name=args.dataset)
        adv_imgs = images + delta_atk

        # Adv accuracy (no defense)
        with torch.no_grad():
            out = multiGPU_CLIP(None, None, None, model,
                                clip_img_preprocessing(adv_imgs),
                                text_tokens, prompt_token=None,
                                dataset_name=args.dataset)
            top1_adv.update(accuracy(out[0], targets, topk=(1,))[0].item(), images.size(0))

        # Counterattack (with proper tau-threshold gating)
        if args.method == 'nac':
            delta_def = nac_counterattack(
                model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                tau_thres=0.2, beta=2.0,
                clip_visual=None,
                nac_momentum=args.nac_momentum,
            )
        elif args.method == 'momentum':
            delta_def = momentum_counterattack(
                model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                tau_thres=0.2, beta=2.0,
                clip_visual=None,
                preprocess_fn=clip_img_preprocessing,
                momentum_coef=args.nac_momentum,
            )
        elif args.method == 'pure_la':
            delta_def = pure_la_counterattack(
                model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                tau_thres=0.2, beta=2.0,
                clip_visual=None,
                preprocess_fn=clip_img_preprocessing,
            )
        elif args.method == 'adam':
            delta_def = adam_counterattack(
                model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                tau_thres=0.2, beta=2.0,
                clip_visual=None,
                preprocess_fn=clip_img_preprocessing,
            )
        elif args.method == 'lbfgs':
            delta_def = lbfgs_counterattack(
                model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                tau_thres=0.2, beta=2.0,
                clip_visual=None,
                preprocess_fn=clip_img_preprocessing,
            )
        elif args.method == 'doc':
            delta_def = doc_counterattack(
                doc_args, model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                beta=2.0, clip_visual=None,
            )
        else:  # ttc
            delta_def = ttc_counterattack(
                model, adv_imgs.data, prompter, add_prompter,
                alpha=alpha_def, attack_iters=args.defense_steps,
                norm='l_inf', epsilon=eps_def,
                tau_thres=0.2, beta=2.0,
                clip_visual=None,
            )

        # Defended accuracy
        with torch.no_grad():
            out = multiGPU_CLIP(None, None, None, model,
                                clip_img_preprocessing(adv_imgs + delta_def),
                                text_tokens, prompt_token=None,
                                dataset_name=args.dataset)
            top1_def.update(accuracy(out[0], targets, topk=(1,))[0].item(), images.size(0))

    print(f"\n{args.dataset}: clean={top1_clean.avg:.2f} | "
          f"adv={top1_adv.avg:.2f} | "
          f"{args.method}={top1_def.avg:.2f} | "
          f"gain={top1_def.avg - top1_adv.avg:.2f}")


if __name__ == '__main__':
    main()
