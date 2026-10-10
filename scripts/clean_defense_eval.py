"""Effect of the counterattack on CLEAN images (no attack).

Reports zero-shot accuracy before and after applying TTC / NAC to clean
images, i.e. the "clean-side cost" of the defense. Complements the
adversarial tables in the paper.

Usage:
  python scripts/clean_defense_eval.py --test_set cifar10 STL10 --batch_size 64
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import load_val_dataset, get_text_prompts_val, AverageMeter, accuracy, convert_models_to_fp32
from func import clip_img_preprocessing, multiGPU_CLIP
from nac_fair_experiment import nac_counterattack
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_method(method_fn, loader, model, prompter, add_prompter, text_tokens,
                    eps, alpha, K, tag):
    top1_clean = AverageMeter('Clean', ':.2f')
    top1_def = AverageMeter(tag, ':.2f')
    for images, target in tqdm(loader, desc=tag):
        images, target = images.to(device), target.to(device)
        with torch.no_grad():
            out, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                         prompter(clip_img_preprocessing(images)),
                                         text_tokens, prompt_token=None)
            top1_clean.update(accuracy(out, target, topk=(1,))[0].item(), images.size(0))

        delta = method_fn(model, images, prompter, add_prompter, alpha, K,
                          'l_inf', eps, tau_thres=0.2, beta=2.0)
        with torch.no_grad():
            out, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                         prompter(clip_img_preprocessing(images + delta)),
                                         text_tokens, prompt_token=None)
            top1_def.update(accuracy(out, target, topk=(1,))[0].item(), images.size(0))
    return top1_clean.avg, top1_def.avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_set', default=['cifar10'], type=str, nargs='*')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--K', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.root = './data'
    args.num_workers = 0
    args.evaluate = True
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p_ in model.parameters():
        p_.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).eval()

    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    eps = 4 / 255.
    alpha = 1 / 255.

    for name in args.test_set:
        ds, loader = load_val_dataset(args, name)
        texts = get_text_prompts_val([ds], [name])[0]
        text_tokens = clip.tokenize(texts).to(device)

        nac_fn = lambda m, X, pr, ap, al, K, nrm, ep, tau_thres, beta: nac_counterattack(
            m, X, pr, ap, al, K, nrm, ep, tau_thres=tau_thres, beta=beta, nac_momentum=0.9)
        ttc_fn = lambda m, X, pr, ap, al, K, nrm, ep, tau_thres, beta: ttc_counterattack(
            m, X, pr, ap, al, K, nrm, ep, tau_thres=tau_thres, beta=beta)

        print(f"\n=== {name} (K={args.K}, eps_def=4/255) ===")
        for method, fn in [('TTC', ttc_fn), ('NAC', nac_fn)]:
            clean_acc, def_acc = evaluate_method(fn, loader, model, prompter, add_prompter,
                                                 text_tokens, eps, alpha, args.K, f'{name}/{method}')
            print(f"{name}: clean={clean_acc:.2f} | clean+{method}={def_acc:.2f} "
                  f"| delta={def_acc - clean_acc:+.2f}")


if __name__ == '__main__':
    main()
