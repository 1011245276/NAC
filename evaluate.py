#!/usr/bin/env python3
"""NAC evaluation script. Run from nac_project/ root directory."""
import argparse, os, sys, random, json
import numpy as np
from tqdm import tqdm
import torch, torch.nn.functional as F

# Use root-level modules (same as TTC project convention)
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import load_val_dataset, get_text_prompts_val, AverageMeter, accuracy
from attacks import attack_pgd
from func import clip_img_preprocessing, multiGPU_CLIP
from nac import nac_counterattack
from ttc import ttc_counterattack

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--method', default='nac', choices=['ttc','nac'])
    p.add_argument('--attack_eps', type=int, default=4)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters(): p.requires_grad = False
    model = torch.nn.DataParallel(model).eval()

    ds, loader = load_val_dataset(args, args.dataset)
    texts = get_text_prompts_val([ds], [args.dataset], template='a photo of a {}.', force_template=True)[0]
    text_tokens = clip.tokenize(texts).to(device)

    prompter = NullPrompter(); add_prompter = TokenPrompter(0)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    eps_a, alpha_a = args.attack_eps/255., 1/255.
    eps_d, alpha_d = 4/255., 1/255.

    top1_clean, top1_adv, top1_def = AverageMeter(), AverageMeter(), AverageMeter()

    for images, targets in tqdm(loader, desc=args.dataset):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            out = multiGPU_CLIP(None,None,None,model,clip_img_preprocessing(images),text_tokens,prompt_token=None,dataset_name=args.dataset)
            top1_clean.update(accuracy(out[0],targets,topk=(1,))[0].item(), images.size(0))

        delta = attack_pgd(None,prompter,model,None,None,add_prompter,criterion,images,targets,alpha_a,10,'l_inf',text_tokens=text_tokens,epsilon=eps_a,dataset_name=args.dataset)
        adv_imgs = images + delta

        with torch.no_grad():
            out = multiGPU_CLIP(None,None,None,model,clip_img_preprocessing(adv_imgs),text_tokens,prompt_token=None,dataset_name=args.dataset)
            top1_adv.update(accuracy(out[0],targets,topk=(1,))[0].item(), images.size(0))

        if args.method == 'nac':
            d = nac_counterattack(model,adv_imgs.data,prompter,add_prompter,alpha_d,2,'l_inf',eps_d,None,0.2,2.0,None,clip_img_preprocessing,0.9)
        else:
            d = ttc_counterattack(model,adv_imgs.data,prompter,add_prompter,alpha_d,2,'l_inf',eps_d,None,0.2,2.0,None,clip_img_preprocessing)

        with torch.no_grad():
            out = multiGPU_CLIP(None,None,None,model,clip_img_preprocessing(adv_imgs+d),text_tokens,prompt_token=None,dataset_name=args.dataset)
            top1_def.update(accuracy(out[0],targets,topk=(1,))[0].item(), images.size(0))

    print(f"{args.dataset}: clean={top1_clean.avg:.2f} adv={top1_adv.avg:.2f} {args.method}={top1_def.avg:.2f}")

if __name__ == '__main__': main()
