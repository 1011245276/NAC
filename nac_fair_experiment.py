"""
NAC vs TTC — fair comparison using original TTC implementation.

Only difference: NAC uses Nesterov look-ahead gradient.
Everything else (tau_threshold, step weighting, etc.) is identical.

Run this script directly — it loads the original TTC counterattack and its NAC variant.
"""
from __future__ import print_function

import argparse
import os
import time
import random
import logging
from tqdm import tqdm
from copy import deepcopy as dcopy

import torch
from torch.cuda.amp import GradScaler, autocast

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import *
from attacks import *
from func import clip_img_preprocessing, multiGPU_CLIP, multiGPU_CLIP_image_logits


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cache', type=str, default='./cache')
    parser.add_argument('--test_set', default=[], type=str, nargs='*')
    parser.add_argument('--test_attack_type', type=str, default="pgd",
                        choices=['pgd', 'CW', 'autoattack'])
    parser.add_argument('--test_eps', type=float, default=4)
    parser.add_argument('--test_numsteps', type=int, default=5)
    parser.add_argument('--test_stepsize', type=float, default=1)
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30)
    parser.add_argument('--add_prompt_size', type=int, default=0)
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='tinyImageNet')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--victim_resume', type=str, default=None)
    parser.add_argument('--outdir', type=str,
                        default='./results')
    parser.add_argument('--tau_thres', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--ttc_eps', type=float, default=4)
    parser.add_argument('--ttc_numsteps', type=int, default=2)
    parser.add_argument('--ttc_stepsize', type=float, default=1.)
    # NAC config
    parser.add_argument('--counterattack', type=str, default='nac',
                        choices=['ttc', 'nac', 'doc', 'momentum'],
                        help='ttc=original TTC, nac=nesterov TTC, doc=DOC, momentum=standard momentum')
    parser.add_argument('--nac_momentum', type=float, default=0.9)
    # DOC config
    parser.add_argument('--DOC_eps', type=float, default=4.0)
    parser.add_argument('--DOC_numsteps', type=int, default=2)
    parser.add_argument('--DOC_stepsize', type=float, default=1.0)
    parser.add_argument('--learnable_tau', type=float, default=0.155)
    parser.add_argument('--temperature', type=float, default=75.0)
    return parser.parse_args()


def compute_tau(clip_visual, images, n):
    orig_feat = clip_visual(clip_img_preprocessing(images), None)
    noisy_feat = clip_visual(clip_img_preprocessing(images + n), None)
    diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)
    return diff_ratio


def nac_counterattack(model, X, prompter, add_prompter, alpha, attack_iters,
                       norm="l_inf", epsilon=0, visual_model_orig=None,
                       tau_thres=None, beta=None, clip_visual=None,
                       nac_momentum=0.9):
    """
    NAC: Nesterov Accelerated version of TTC counterattack.
    Same framework as original TTC, only gradient computation differs.
    NOTE: This is a self-contained implementation optimized for the experiment
    framework. The standalone nac.py module provides an equivalent implementation
    with a slightly different interface for independent use.
    """
    lower_limit, upper_limit = 0, 1

    def clamp(X, lower, upper):
        return torch.max(torch.min(X, upper), lower)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters == 0:
        return delta.data

    diff_ratio = compute_tau(clip_visual, X, delta.data) if clip_visual is not None else None

    tunable_param_names = []
    for n, p in model.module.named_parameters():
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X)), prompt_token
        )
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = []
    deltas_per_step.append(delta.data.clone())

    # Nesterov velocity
    velocity = torch.zeros_like(delta)

    for _step_id in range(attack_iters):

        # === NAC CHANGE: compute gradient at look-ahead position ===
        look_ahead = X + delta + nac_momentum * velocity
        prompted_images = prompter(clip_img_preprocessing(look_ahead))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if _step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        scheme_sign = (tau_thres - diff_ratio).sign()

        l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]

        # === NAC CHANGE: Nesterov momentum update ===
        velocity = nac_momentum * velocity + alpha * torch.sign(grad)
        d = delta[:, :, :, :] + velocity
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        elif norm == "l_2":
            d = d.view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    # === Same step weighting as original TTC ===
    Delta = torch.stack(deltas_per_step, dim=1)

    weights = torch.arange(attack_iters + 1).unsqueeze(0).expand(X.size(0), -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)

    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.

    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)

    Delta = (weights * Delta).sum(dim=1)

    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return Delta


# Import original TTC counterattack (with tau_threshold)
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack


def momentum_counterattack(model, X, prompter, add_prompter, alpha, attack_iters,
                            norm="l_inf", epsilon=0, visual_model_orig=None,
                            tau_thres=None, beta=None, clip_visual=None,
                            preprocess_fn=None, momentum_coef=0.9):
    """Standard momentum counterattack (NO look-ahead). Control for ablation."""
    lower_limit, upper_limit = 0, 1

    def clamp(X, lo, hi):
        return torch.max(torch.min(X, hi), lo)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    if attack_iters == 0:
        return delta.data

    tunable_param_names = []
    for n, p in model.module.named_parameters():
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_prep = preprocess_fn(X) if preprocess_fn else X
        X_ori_reps = model.module.encode_image(prompter(X_prep), prompt_token)
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]
    momentum = torch.zeros_like(delta)

    for _step_id in range(attack_iters):
        # Standard momentum: gradient at CURRENT position (NO look-ahead)
        X_prep = preprocess_fn(X + delta) if preprocess_fn else (X + delta)
        prompted_images = prompter(X_prep)
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if _step_id == 0:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        scheme_sign = (tau_thres - diff_ratio).sign()
        l2_loss = (((X_att_reps - X_ori_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]

        # Standard momentum update (gradient at current position)
        momentum = momentum_coef * momentum + alpha * torch.sign(grad)
        d = delta[:, :, :, :] + momentum
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    Delta = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1).unsqueeze(0).expand(X.size(0), -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)
    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    Delta = (weights * Delta).sum(dim=1)

    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True
    return Delta


# ========== DOC Counterattack ==========
def compute_tau_directional(clip_visual, images, eps=0.05, trials=5):
    with torch.no_grad():
        orig_feat = clip_visual(clip_img_preprocessing(images), None)
        directions = []
        for _ in range(trials):
            noise = torch.randn_like(images).sign() * eps
            noisy_feat = clip_visual(clip_img_preprocessing(images + noise), None)
            cos_sim = F.cosine_similarity(noisy_feat, orig_feat, dim=-1)
            directions.append(cos_sim)
        return 1 - torch.stack(directions, dim=1).mean(dim=1)


def compute_scheme_weight(diff_ratio, learnable_tau=0.3, temperature=20):
    return torch.sigmoid((learnable_tau - diff_ratio) * temperature)


def doc_counterattack(args, model, X, prompter, add_prompter, alpha, attack_iters,
                       norm="l_inf", epsilon=0, beta=None, clip_visual=None):
    """DOC: Directional Orthogonal Counterattack (AAAI 2026)."""
    lower_limit, upper_limit = 0, 1

    def clamp(X, lo, hi):
        return torch.max(torch.min(X, hi), lo)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta

    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters == 0:
        return delta.data

    diff_ratio = compute_tau_directional(clip_visual, X, eps=epsilon, trials=5) if clip_visual is not None else None

    tunable_param_names = []
    for n, p in model.module.named_parameters():
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(prompter(clip_img_preprocessing(X)), prompt_token)
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]
    momentum = torch.zeros_like(X)

    learnable_tau = args.learnable_tau
    temperature = args.temperature

    for _step_id in range(attack_iters):
        prompted_images = prompter(clip_img_preprocessing(X + delta))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if _step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        scheme_weight = compute_scheme_weight(diff_ratio, learnable_tau=learnable_tau, temperature=temperature)
        scheme_weight = scheme_weight.clamp(min=0., max=1.)

        l2_loss = (((X_att_reps - X_ori_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta, retain_graph=False, create_graph=False)[0]

        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
        grad = grad / (grad_norm + 1e-8)

        orth_noise = torch.randn_like(grad)
        inner = (grad * orth_noise).view(grad.size(0), -1).sum(dim=1).view(-1, 1, 1, 1)
        orth_noise = orth_noise - inner * grad
        orth_noise = orth_noise / (torch.norm(orth_noise.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
        grad = grad + 0.05 * orth_noise

        if _step_id == 0:
            momentum = grad.clone()
        else:
            momentum = 0.75 * momentum + 0.25 * grad

        delta.data = delta.data + alpha * torch.sign(momentum)
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = clamp(delta.data, lower_limit - X, upper_limit - X)
        deltas_per_step.append(delta.data.clone())

    Delta = torch.stack(deltas_per_step, dim=1)
    raw_weights = torch.arange(attack_iters + 1).float().unsqueeze(0).expand(X.size(0), -1).to(X.device)
    soft_weights = torch.exp(raw_weights * beta)
    soft_weights = soft_weights / soft_weights.sum(dim=1, keepdim=True)
    soft_weights = soft_weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    Delta = (soft_weights * Delta).sum(dim=1)

    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return Delta


def validate(args, val_dataset_name, model, model_text, model_image,
             prompter, add_prompter, criterion, visual_model_orig=None, clip_visual=None):
    tag = f"{args.counterattack}" + (f"_m{args.nac_momentum}" if args.counterattack == 'nac' else "")

    logging.info(f"Counterattack: {tag} | Attack: {args.test_attack_type}")
    logging.info(f"tau_thres={args.tau_thres} beta={args.beta}")

    dataset_num = len(val_dataset_name)
    all_clean_org, all_adv_org, all_adv_def = {}, {}, {}

    for cnt in range(dataset_num):
        val_dataset, val_loader = load_val_dataset(args, val_dataset_name[cnt])
        dataset_name = val_dataset_name[cnt]
        texts = get_text_prompts_val([val_dataset], [dataset_name],
                                     template='a photo of a {}.', force_template=True)[0]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run = ['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run = ['apgd-ce']

        top1_org = AverageMeter('Clean', ':.2f')
        top1_adv = AverageMeter('Adv', ':.2f')
        top1_def = AverageMeter(f'Adv+{tag}', ':.2f')

        prompter.eval()
        add_prompter.eval()
        model.eval()

        text_tokens = clip.tokenize(texts).to(device)

        for images, target in tqdm(val_loader, desc=f"  [{dataset_name}/{tag}]"):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # Clean accuracy
                with torch.no_grad():
                    clean_output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(images)),
                        text_tokens=text_tokens, prompt_token=None, dataset_name=dataset_name
                    )
                    top1_org.update(accuracy(clean_output, target, topk=(1,))[0].item(), images.size(0))

                # Generate adversarial attack
                torch.cuda.empty_cache()
                if args.test_attack_type == "pgd":
                    delta_atk = attack_pgd(
                        args, prompter, model, model_text, model_image, add_prompter, criterion,
                        images, target, args.test_stepsize, args.test_numsteps, 'l_inf',
                        text_tokens=text_tokens, epsilon=args.test_eps, dataset_name=dataset_name
                    )
                    attacked = images + delta_atk
                elif args.test_attack_type == "autoattack":
                    binary = ['PCAM', 'hateful_memes']
                    attacks_to_run = ['apgd-ce', 'apgd-dlr']
                    if dataset_name in binary:
                        attacks_to_run = ['apgd-ce']
                    attacked = attack_auto(
                        model, images, target, text_tokens,
                        None, None, epsilon=args.test_eps, attacks_to_run=attacks_to_run
                    )

                # Adv accuracy (no defense)
                with torch.no_grad():
                    adv_output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(attacked)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name
                    )
                    top1_adv.update(accuracy(adv_output, target, topk=(1,))[0].item(), images.size(0))

                # Counterattack — TTC or NAC
                ttc_eps_val = args.ttc_eps
                ttc_stepsize_val = args.ttc_stepsize

                if args.counterattack == 'momentum':
                    delta_def = momentum_counterattack(
                        model, attacked.data, prompter, add_prompter,
                        alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=ttc_eps_val,
                        tau_thres=args.tau_thres, beta=args.beta,
                        clip_visual=clip_visual,
                        preprocess_fn=clip_img_preprocessing,
                        momentum_coef=getattr(args, 'nac_momentum', 0.9)
                    )
                elif args.counterattack == 'doc':
                    delta_def = doc_counterattack(
                        args, model, attacked.data, prompter, add_prompter,
                        alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=ttc_eps_val,
                        beta=args.beta, clip_visual=clip_visual
                    )
                elif args.counterattack == 'nac':
                    delta_def = nac_counterattack(
                        model, attacked.data, prompter, add_prompter,
                        alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=ttc_eps_val,
                        visual_model_orig=visual_model_orig,
                        tau_thres=args.tau_thres, beta=args.beta,
                        clip_visual=clip_visual,
                        nac_momentum=args.nac_momentum
                    )
                else:
                    delta_def = ttc_counterattack(
                        model, attacked.data, prompter, add_prompter,
                        alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=ttc_eps_val,
                        visual_model_orig=visual_model_orig,
                        tau_thres=args.tau_thres, beta=args.beta,
                        clip_visual=clip_visual
                    )

                with torch.no_grad():
                    def_output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(attacked + delta_def)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name
                    )
                    top1_def.update(accuracy(def_output, target, topk=(1,))[0].item(), images.size(0))

        torch.cuda.empty_cache()

        all_clean_org[dataset_name] = top1_org.avg
        all_adv_org[dataset_name] = top1_adv.avg
        all_adv_def[dataset_name] = top1_def.avg

        show_text = f"{dataset_name}: clean={top1_org.avg:.2f} | "
        show_text += f"adv={top1_adv.avg:.2f} | adv+{tag}={top1_def.avg:.2f} | "
        show_text += f"gain={top1_def.avg - top1_adv.avg:.2f}"
        logging.info(show_text)
        print(f"  {show_text}")

    valid = [n for n in val_dataset_name if n in all_adv_def]
    avg_adv = np.mean([all_adv_org[n] for n in valid]) if valid else 0
    avg_def = np.mean([all_adv_def[n] for n in valid]) if valid else 0
    summary = f"SUMMARY: avg adv={avg_adv:.2f} | avg def={avg_def:.2f} | avg gain={avg_def - avg_adv:.2f}"
    logging.info(summary)
    print(summary)


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_options()

    tag = f"{args.counterattack}" + (f"_m{args.nac_momentum}" if args.counterattack == 'nac' else "")
    outdir = os.path.join(args.outdir, f"{args.test_attack_type}_eps_{args.test_eps}", tag)
    os.makedirs(outdir, exist_ok=True)

    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_filename = f"seed_{seed}.log"
    log_filename = os.path.join(outdir, log_filename)
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"NAC Fair: method={args.counterattack} momentum={args.nac_momentum}")
    logging.info(args)

    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.

    # Use user-provided root; set defaults for ImageNet/tinyImageNet if not specified
    if not hasattr(args, 'imagenet_root') or args.imagenet_root is None:
        args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    if not hasattr(args, 'tinyimagenet_root') or args.tinyimagenet_root is None:
        args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    arch_map = {'vit_b32': 'ViT-B/32', 'vit_b16': 'ViT-B/16', 'RN50': 'RN50'}
    arch = getattr(args, 'arch', 'ViT-B/32')
    arch = arch_map.get(arch, arch)
    model, _ = clip.load(arch, device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)

    clip_visual = None
    if args.victim_resume:
        clip_visual = dcopy(model.visual)
        model = load_checkpoints2(args, args.victim_resume, model, None)

    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = NullPrompter()
    add_prompter = TokenPrompter(0)
    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    if len(args.test_set) == 0:
        test_set = DATASETS
    else:
        test_set = args.test_set

    criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    validate(args, test_set, model, None, None, prompter,
             add_prompter, criterion_attack, None, clip_visual)


if __name__ == "__main__":
    main()
