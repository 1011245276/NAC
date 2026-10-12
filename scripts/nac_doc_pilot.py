"""
NAC+DOC 组合 pilot: 前瞻梯度 (NAC) + 正交噪声 + 梯度归一化 + 可学习 tau 门控 (DOC)

组合逻辑：
  - NAC 改的是"梯度在哪算"（前瞻位置 delta + mu * velocity）
  - DOC 改的是"梯度怎么处理"（归一化 + 正交噪声）+ "步怎么加权"（软门控）
  - 两者改的是内层不同维度，理论上可叠加

对照：TTC / NAC-only / DOC-only / NAC+DOC
"""
import sys, os, json, logging, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

# Reuse the fair experiment infrastructure
from nac_fair_experiment import (
    parse_options, load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, AverageMeter, accuracy,
    clip_img_preprocessing, compute_tau_directional, compute_scheme_weight
)

def nac_doc_counterattack(args, model, X, prompter, add_prompter,
                          alpha, attack_iters, norm="l_inf", epsilon=0,
                          beta=None, clip_visual=None, nac_momentum=0.9):
    """
    NAC+DOC: Nesterov look-ahead + DOC gradient processing + DOC soft gating.

    Combines:
      NAC: gradient at look-ahead position (X + delta + mu * velocity)
           Nesterov velocity update: v = mu*v + alpha*sign(modified_grad)
      DOC: gradient normalization
           orthogonal noise injection (0.05 weight)
           learnable tau soft gating (scheme_weight)
           soft step weighting (not hard scheme_sign)
    """
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
    velocity = torch.zeros_like(delta)  # NAC velocity
    learnable_tau = args.learnable_tau
    temperature = args.temperature

    for step_id in range(attack_iters):
        # === NAC: look-ahead position ===
        look_ahead = X + delta + nac_momentum * velocity
        prompted_images = prompter(clip_img_preprocessing(look_ahead))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        # === DOC: learnable tau soft gating ===
        scheme_weight = compute_scheme_weight(diff_ratio, learnable_tau, temperature)
        scheme_weight = scheme_weight.clamp(min=0., max=1.)

        # === DOC: gradient normalization + orthogonal noise ===
        l2_loss = (((X_att_reps - X_ori_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta, retain_graph=False, create_graph=False)[0]

        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
        grad = grad / (grad_norm + 1e-8)

        orth_noise = torch.randn_like(grad)
        inner = (grad * orth_noise).view(grad.size(0), -1).sum(dim=1).view(-1, 1, 1, 1)
        orth_noise = orth_noise - inner * grad
        orth_noise = orth_noise / (torch.norm(orth_noise.view(orth_noise.size(0), -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
        grad = grad + 0.05 * orth_noise

        # === NAC: Nesterov velocity update (NOT DOC's Polyak momentum) ===
        velocity = nac_momentum * velocity + alpha * torch.sign(grad)
        d = delta.data + velocity

        if norm == "l_inf":
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - X, upper_limit - X)
        delta.data = d
        deltas_per_step.append(delta.data.clone())

    # === DOC: soft weighting (not TTC's hard scheme_sign) ===
    Delta = torch.stack(deltas_per_step, dim=1)
    raw_weights = torch.arange(attack_iters + 1).float().unsqueeze(0).expand(X.size(0), -1).to(X.device)
    soft_weights = torch.exp(raw_weights * beta)
    soft_weights = soft_weights / soft_weights.sum(dim=1, keepdim=True)

    one_hot_first = torch.zeros_like(soft_weights)
    one_hot_first[:, 0] = 1.
    weights = scheme_weight.unsqueeze(1) * soft_weights + (1 - scheme_weight).unsqueeze(1) * one_hot_first
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    Delta = (weights * Delta).sum(dim=1)

    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True
    return Delta


def run_pilot():
    """Run 4 methods on CIFAR-10, K=2, eps_att=4/255."""
    # Minimal argparse: reuse parse_options but override
    sys.argv = [
        'nac_fair_experiment.py',
        '--evaluate', 'True',
        '--batch_size', '32',
        '--test_set', 'cifar10',
        '--test_attack_type', 'pgd',
        '--test_eps', '4.0',
        '--test_numsteps', '10',
        '--test_stepsize', '1.0',
        '--method', 'null_patch',
        '--root', './data',
        '--dataset', 'cifar10',
        '--seed', '0',
        '--num_seeds', '1',
        '--tau_thres', '0.2',
        '--beta', '2.0',
        '--ttc_eps', '4.0',
        '--ttc_numsteps', '2',
        '--ttc_stepsize', '1.0',
        '--nac_momentum', '0.9',
        '--learnable_tau', '0.155',
        '--temperature', '75.0',
        '--counterattack', 'nac_doc',  # our new method
        '--outdir', './results/nac_doc_pilot',
    ]
    args = parse_options()

    # Import after args parsed (model loading needs args)
    from nac_fair_experiment import setup_model_and_data

    model, model_text, model_image, prompter, add_prompter, \
        visual_model_orig, clip_visual, val_dataset_name = setup_model_and_data(args)

    # Attack params
    ttc_eps_val = args.ttc_eps
    ttc_stepsize_val = args.ttc_stepsize

    from nac_fair_experiment import doc_counterattack, nac_counterattack, ttc_counterattack

    methods = {
        'ttc': ttc_counterattack,
        'nac': nac_counterattack,
        'doc': doc_counterattack,
        'nac_doc': nac_doc_counterattack,
    }

    results = {}
    for method_name, func in methods.items():
        print(f"\n{'='*60}")
        print(f"  Running {method_name.upper()} on CIFAR-10, K=2, eps=4/255")
        print(f"{'='*60}")

        val_dataset, val_loader = load_val_dataset(args, 'cifar10')
        texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
        text_tokens = model_text([texts]).to(args.device if hasattr(args, 'device') else 'cuda')

        top1_org = AverageMeter('Clean', ':.2f')
        top1_adv = AverageMeter('Adv', ':.2f')
        top1_def = AverageMeter(f'Adv+{method_name}', ':.2f')

        for batch_idx, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # Clean
            with torch.no_grad():
                def_output, _, _, _ = multiGPU_CLIP(
                    None, None, None, model,
                    prompter(clip_img_preprocessing(images)),
                    text_tokens, prompt_token=None, dataset_name='cifar10'
                )
                top1_org.update(accuracy(def_output, target, topk=(1,))[0].item(), images.size(0))

            # Attack
            delta_adv = torch.zeros_like(images).uniform_(-args.test_eps, args.test_eps)
            delta_adv.requires_grad = True
            # Simple PGD attack
            for _ in range(args.test_numsteps):
                prompted = prompter(clip_img_preprocessing(images + delta_adv))
                output = model.module.encode_image(prompted, add_prompter())
                loss = F.cross_entropy(output @ text_tokens.t(), target)
                grad = torch.autograd.grad(loss, delta_adv)[0]
                delta_adv.data = torch.clamp(
                    delta_adv.data + args.test_stepsize * torch.sign(grad),
                    -args.test_eps, args.test_eps
                )

            attacked = torch.clamp(images + delta_adv, 0, 1)

            with torch.no_grad():
                def_output_adv, _, _, _ = multiGPU_CLIP(
                    None, None, None, model,
                    prompter(clip_img_preprocessing(attacked)),
                    text_tokens, prompt_token=None, dataset_name='cifar10'
                )
                top1_adv.update(accuracy(def_output_adv, target, topk=(1,))[0].item(), images.size(0))

            # Defense
            if method_name == 'nac_doc':
                delta_def = nac_doc_counterattack(
                    args, model, attacked.data, prompter, add_prompter,
                    alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                    norm='l_inf', epsilon=ttc_eps_val,
                    beta=args.beta, clip_visual=clip_visual,
                    nac_momentum=args.nac_momentum
                )
            elif method_name == 'nac':
                delta_def = nac_counterattack(
                    model, attacked.data, prompter, add_prompter,
                    alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                    norm='l_inf', epsilon=ttc_eps_val,
                    visual_model_orig=visual_model_orig,
                    tau_thres=args.tau_thres, beta=args.beta,
                    clip_visual=clip_visual,
                    nac_momentum=args.nac_momentum,
                    clamp_lookahead=False
                )
            elif method_name == 'doc':
                delta_def = doc_counterattack(
                    args, model, attacked.data, prompter, add_prompter,
                    alpha=ttc_stepsize_val, attack_iters=args.ttc_numsteps,
                    norm='l_inf', epsilon=ttc_eps_val,
                    beta=args.beta, clip_visual=clip_visual
                )
            else:  # ttc
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
                    text_tokens, prompt_token=None, dataset_name='cifar10'
                )
                top1_def.update(accuracy(def_output, target, topk=(1,))[0].item(), images.size(0))

            if batch_idx % 50 == 0:
                print(f"  batch {batch_idx}: clean={top1_org.avg:.2f} adv={top1_adv.avg:.2f} def={top1_def.avg:.2f}")

        results[method_name] = {
            'clean': round(top1_org.avg, 2),
            'adv': round(top1_adv.avg, 2),
            'defended': round(top1_def.avg, 2),
            'gain': round(top1_def.avg - top1_adv.avg, 2)
        }
        print(f"\n  {method_name}: clean={top1_org.avg:.2f} | adv={top1_adv.avg:.2f} | def={top1_def.avg:.2f} | gain={top1_def.avg - top1_adv.avg:.2f}")

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for m, r in results.items():
        print(f"  {m:10s}: clean={r['clean']:.2f} | adv={r['adv']:.2f} | def={r['defended']:.2f} | gain={r['gain']:.2f}")

    # Save
    out_path = './results/nac_doc_pilot/pilot_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Orthogonality check
    ttc_gain = results['ttc']['gain']
    nac_gain = results['nac']['gain']
    doc_gain = results['doc']['gain']
    combo_gain = results['nac_doc']['gain']
    additive = nac_gain + doc_gain
    print(f"\n  Orthogonality check:")
    print(f"    NAC gain: {nac_gain:.2f} | DOC gain: {doc_gain:.2f} | Additive: {additive:.2f} | Combo: {combo_gain:.2f}")
    print(f"    Combo > max(NAC,DOC)? {'YES' if combo_gain > max(nac_gain, doc_gain) else 'NO'}")
    print(f"    Combo >= additive? {'YES' if combo_gain >= additive * 0.8 else 'NO (partial overlap)'}")


if __name__ == '__main__':
    run_pilot()
