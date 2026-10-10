# -*- coding: utf-8 -*-
"""
Defense-strength routing (cascade) pilot — the advisor's "MoE routing" idea,
implemented as conditional computation over defense depth.

Idea: on a mixed stream (clean + attacked images in unknown proportion),
route each sample to a defense depth based on an oracle-free signal:
    tier K=0 : no counterattack (save compute)
    tier K=2 : NAC with 2 steps
    tier K=4 : NAC with 4 steps
Signal: shift0 = ||f(x+delta_init) - f(x)|| / ||f(x)||, the embedding shift
induced by the random init perturbation. This signal is computed for free
inside the NAC trajectory (it IS the step-0 gating signal). Hypothesis:
attacked images sit in flat adversarial regions, so a random eps-perturbation
moves their embedding LESS than for clean images. Also records shift1
(shift after 1 counterattack step) as an alternative signal.

Protocol:
  Phase 1 (collection): per-sample records {shift0, shift1, correct under
      K=0 / K=2 / K=4} for both the clean stream and the attacked stream
      (same images), on the first max_batches batches.
  Phase 2 (analysis, offline numpy): even batches -> calibration (AUROC +
      threshold grid), odd batches -> held-out evaluation. Cost-accuracy
      frontier for attack ratios rho in {0, .25, .5, 1.0} vs baselines
      {no-defense, always-K2, always-K4, oracle}.

"always-K2" / "always-K4" labels come from the SAME trajectory pass, so they
reproduce the standard NAC runs exactly (same init, gate, and weighting).

Run:
  python scripts/defense_routing_pilot.py --max_batches 60
Output:
  results/defense_routing/routing_pilot.json / .log
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replace import clip                                    # noqa: E402
from models.prompters import TokenPrompter, NullPrompter    # noqa: E402
from func import clip_img_preprocessing, multiGPU_CLIP      # noqa: E402
from utils import (convert_models_to_fp32, load_val_dataset,    # noqa: E402
                   get_text_prompts_val)
from attacks import attack_pgd                              # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', type=str, default='vit_b32')
    p.add_argument('--test_set', default=['cifar10'], type=str, nargs='*')
    p.add_argument('--root', type=str, default='./data')
    p.add_argument('--dataset', type=str, default='tinyImageNet')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test_attack_type', type=str, default='pgd')
    p.add_argument('--test_eps', type=float, default=4)       # /255 applied later
    p.add_argument('--test_numsteps', type=int, default=10)
    p.add_argument('--test_stepsize', type=float, default=1)  # /255 applied later
    p.add_argument('--ttc_eps', type=float, default=4)        # /255 applied later
    p.add_argument('--ttc_stepsize', type=float, default=1)   # /255 applied later
    p.add_argument('--ttc_numsteps', type=int, default=4)     # deepest tier
    p.add_argument('--mid_tier', type=int, default=2)
    p.add_argument('--nac_momentum', type=float, default=0.9)
    p.add_argument('--tau_thres', type=float, default=0.2)
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--max_batches', type=int, default=60)
    p.add_argument('--victim_resume', type=str, default=None)
    p.add_argument('--method', type=str, default='null_patch')
    p.add_argument('--prompt_size', type=int, default=30)
    p.add_argument('--add_prompt_size', type=int, default=0)
    p.add_argument('--name', type=str, default='')
    p.add_argument('--cache', type=str, default='./cache')
    p.add_argument('--imagenet_root', type=str, default='./data/imagenet-100/imagenet_folder')
    p.add_argument('--tinyimagenet_root', type=str,
                   default='./data/tiny-imagenet-200/tiny-imagenet-200')
    p.add_argument('--outdir', type=str, default='./results/defense_routing')
    return p.parse_args()


def gated_delta(deltas_per_step, scheme_sign, beta, T):
    """TTC/NAC tau-threshold weighted step averaging over the first T+1 entries
    (steps 0..T), identical to the authoritative implementations."""
    Delta = torch.stack(deltas_per_step[:T + 1], dim=1)  # [B, T+1, C, H, W]
    B = Delta.size(0)
    weights = torch.arange(T + 1).unsqueeze(0).expand(B, -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)
    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(B, T + 1, 1, 1, 1)
    return (weights * Delta).sum(dim=1)


def eval_logits(model, prompter, add_prompter, X, text_tokens, dataset_name):
    output, _, _, _ = multiGPU_CLIP(
        None, None, None, model,
        prompter(clip_img_preprocessing(X)),
        text_tokens=text_tokens, prompt_token=None, dataset_name=dataset_name
    )
    return output


def per_sample_correct(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).detach().cpu().numpy().astype(np.int8)


def collect_records(args, model, prompter, add_prompter, criterion):
    """One pass over the loader; per-sample records for clean & adv streams."""
    dataset_name = args.test_set[0]
    val_dataset, val_loader = load_val_dataset(args, dataset_name)
    texts = get_text_prompts_val([val_dataset], [dataset_name])[0]
    text_tokens = clip.tokenize(texts).to(device)

    K_deep, K_mid = args.ttc_numsteps, args.mid_tier
    rec = {s: {'shift0': [], 'shift1': [], 'c0': [], f'c{K_mid}': [],
               f'c{K_deep}': []} for s in ('clean', 'adv')}

    lower_limit, upper_limit = 0, 1

    def clamp(X_, lo, hi):
        return torch.max(torch.min(X_, hi), lo)

    n_done = 0
    for images, target in tqdm(val_loader, desc=f'[{dataset_name}]'):
        if n_done >= args.max_batches:
            break
        images = images.to(device)
        target = target.to(device)

        with torch.amp.autocast('cuda'):
            delta_atk = attack_pgd(
                args, prompter, model, None, None, add_prompter, criterion,
                images, target, args.test_stepsize, args.test_numsteps, 'l_inf',
                text_tokens=text_tokens, epsilon=args.test_eps,
                dataset_name=dataset_name
            )
            attacked = (images + delta_atk).detach()

            for stream, X_in in (('clean', images), ('adv', attacked)):
                prompt_token = add_prompter()
                epsilon, alpha = args.ttc_eps, args.ttc_stepsize

                delta = torch.zeros_like(X_in)
                delta.uniform_(-epsilon, epsilon)
                delta = clamp(delta, lower_limit - X_in, upper_limit - X_in)
                delta.requires_grad_(True)
                velocity = torch.zeros_like(X_in)
                deltas_per_step = [delta.data.clone()]

                with torch.no_grad():
                    X_ori_reps = model.module.encode_image(
                        prompter(clip_img_preprocessing(X_in)), prompt_token)
                    X_ori_norm = torch.norm(X_ori_reps, dim=-1)

                scheme_sign, shift1 = None, None
                for _step_id in range(K_deep):
                    look_ahead = X_in + delta + args.nac_momentum * velocity
                    X_att_reps = model.module.encode_image(
                        prompter(clip_img_preprocessing(look_ahead)), prompt_token)
                    if _step_id == 0:
                        feature_diff = X_att_reps - X_ori_reps
                        diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm
                        scheme_sign = (args.tau_thres - diff_ratio).sign()
                    l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
                    grad = torch.autograd.grad(l2_loss, delta)[0]
                    velocity = args.nac_momentum * velocity + alpha * torch.sign(grad)
                    d = delta + velocity
                    d = torch.clamp(d, min=-epsilon, max=epsilon)
                    d = clamp(d, lower_limit - X_in, upper_limit - X_in)
                    delta.data = d
                    deltas_per_step.append(delta.data.clone())
                    if _step_id == 0:
                        reps1 = model.module.encode_image(
                            prompter(clip_img_preprocessing(X_in + delta.data.detach())),
                            prompt_token)
                        shift1 = ((reps1 - X_ori_reps).norm(dim=-1)
                                  / X_ori_norm).cpu().numpy()

                with torch.no_grad():
                    shift0 = diff_ratio.detach().cpu().numpy()
                    Delta_mid = gated_delta(deltas_per_step, scheme_sign,
                                            args.beta, K_mid)
                    Delta_deep = gated_delta(deltas_per_step, scheme_sign,
                                             args.beta, K_deep)
                    outs = {
                        'c0': eval_logits(model, prompter, add_prompter, X_in,
                                          text_tokens, dataset_name),
                        f'c{K_mid}': eval_logits(model, prompter, add_prompter,
                                                 X_in + Delta_mid, text_tokens,
                                                 dataset_name),
                        f'c{K_deep}': eval_logits(model, prompter, add_prompter,
                                                  X_in + Delta_deep, text_tokens,
                                                  dataset_name),
                    }

                r = rec[stream]
                r['shift0'].append(shift0)
                r['shift1'].append(shift1)
                for k, out in outs.items():
                    r[k].append(per_sample_correct(out, target))

        n_done += 1
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            print(f'  batch {n_done}/{args.max_batches}')

    out = {}
    for stream in rec:
        out[stream] = {k: np.concatenate(v) for k, v in rec[stream].items()}
    return out


def auroc(pos, neg):
    """P(pos > neg) + 0.5*P(pos == neg), via rank statistics."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    all_v = np.concatenate([pos, neg])
    order = all_v.argsort(kind='stable')
    ranks = np.empty(len(all_v), dtype=np.float64)
    ranks[order] = np.arange(1, len(all_v) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.

    arch_map = {'vit_b32': 'ViT-B/32', 'vit_b16': 'ViT-B/16', 'RN50': 'RN50'}
    arch = arch_map.get(args.arch, args.arch)

    model, _ = clip.load(arch, device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    log_path = os.path.join(args.outdir, 'routing_pilot.log')
    # Never overwrite a previous log/json: archive with a timestamp first
    # (same policy as nac_fair_experiment.py -- overwriting destroyed raw
    # logs backing earlier drafts of the paper).
    for prev in (log_path, os.path.join(args.outdir, 'routing_pilot.json'),
                 os.path.join(args.outdir, 'records.npz')):
        if os.path.exists(prev):
            archive_dir = os.path.join(args.outdir, '_archive')
            os.makedirs(archive_dir, exist_ok=True)
            stem, ext = os.path.splitext(os.path.basename(prev))
            os.replace(prev, os.path.join(
                archive_dir, f'{stem}_{time.strftime("%Y%m%d_%H%M%S")}{ext}'))
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line)
        log_f.write(line + '\n')
        log_f.flush()

    log(f'routing pilot: arch={arch} tiers K=0/{args.mid_tier}/{args.ttc_numsteps} '
        f'mu={args.nac_momentum} tau={args.tau_thres} beta={args.beta} '
        f'max_batches={args.max_batches}')

    rec = collect_records(args, model, prompter, add_prompter, criterion)

    # Persist per-sample records so threshold/signal changes can be re-analyzed
    # offline without another GPU collection pass.
    np.savez(os.path.join(args.outdir, 'records.npz'),
             **{f'{stream}_{k}': v for stream in rec for k, v in rec[stream].items()})

    K_mid, K_deep = args.mid_tier, args.ttc_numsteps
    c_mid, c_deep = f'c{K_mid}', f'c{K_deep}'

    n = rec['clean']['c0'].shape[0]
    idx = np.arange(n)
    cal_mask = (idx % 2 == 0)
    ev_mask = ~cal_mask
    log(f'records: n={n} (calibration={int(cal_mask.sum())}, eval={int(ev_mask.sum())})')

    # ---------------- signal quality ----------------
    sig_stats = {}
    for sig in ('shift0', 'shift1'):
        a = auroc(rec['adv'][sig][cal_mask], rec['clean'][sig][cal_mask])
        sig_stats[sig] = a
        log(f'AUROC(attacked vs clean | {sig}, calibration) = {a:.4f}')
    # Select by DISCRIMINABILITY regardless of direction: a signal with
    # AUROC ~ 0 separates the two streams perfectly in the inverted direction
    # (attacked -> LOW values), which is just as useful as AUROC ~ 1.
    # (The earlier max-AUROC rule wrongly discarded shift0, whose AUROC was
    # ~4e-6 -- i.e. a perfect inverted separator -- in favor of shift1 at 0.39.)
    signal = max(sig_stats, key=lambda k: abs(sig_stats[k] - 0.5))
    s_clean, s_adv = rec['clean'][signal], rec['adv'][signal]
    # Direction: if AUROC(adv as positive) < 0.5, attacked samples have SMALLER
    # shift values -> route LOW values to deeper defense.
    low_is_attack = sig_stats[signal] < 0.5
    log(f'router signal = {signal} (AUROC={sig_stats[signal]:.4f}, '
        f'effective_discriminability={max(sig_stats[signal], 1 - sig_stats[signal]):.4f}, '
        f'low_is_attack={low_is_attack})')

    # ---------------- baselines (held-out eval split) ----------------
    base = {
        'no_defense': {'clean': float(rec['clean']['c0'][ev_mask].mean()),
                       'adv': float(rec['adv']['c0'][ev_mask].mean()),
                       'cost_clean': 0.0, 'cost_adv': 0.0},
        f'always_K{K_mid}': {'clean': float(rec['clean'][c_mid][ev_mask].mean()),
                             'adv': float(rec['adv'][c_mid][ev_mask].mean()),
                             'cost_clean': float(K_mid), 'cost_adv': float(K_mid)},
        f'always_K{K_deep}': {'clean': float(rec['clean'][c_deep][ev_mask].mean()),
                              'adv': float(rec['adv'][c_deep][ev_mask].mean()),
                              'cost_clean': float(K_deep), 'cost_adv': float(K_deep)},
    }
    for k, v in base.items():
        log(f'baseline {k}: clean={v["clean"]*100:.2f} adv={v["adv"]*100:.2f}')

    oracle = {'clean': float(rec['clean']['c0'][ev_mask].mean()),
              'adv': float(rec['adv'][c_deep][ev_mask].mean()),
              'cost_clean': 0.0, 'cost_adv': float(K_deep)}
    log(f'oracle router: clean={oracle["clean"]*100:.2f} adv={oracle["adv"]*100:.2f}')

    # ---------------- threshold grid on calibration ----------------
    # Grids span high quantiles of the attacked stream so that a (near-)
    # perfectly separating signal can route almost all attacked samples to the
    # deep tier -- the original (10..50) cap made that unreachable.
    if low_is_attack:
        t1_grid = [np.percentile(s_adv[cal_mask], q) for q in (10, 25, 50, 75, 90, 95)]
        t2_grid = [np.percentile(s_clean[cal_mask], q) for q in (10, 25, 50, 75, 90)]
    else:
        t1_grid = [np.percentile(s_adv[cal_mask], q) for q in (50, 75, 90, 95)]
        t2_grid = [np.percentile(s_clean[cal_mask], q) for q in (10, 25, 50, 75, 90)]

    def tier_of(s, t1, t2):
        if low_is_attack:
            return np.where(s < t1, K_deep, np.where(s < t2, K_mid, 0))
        return np.where(s > t1, K_deep, np.where(s > t2, K_mid, 0))

    def routed_acc_cost(s, correct_by_tier, t1, t2):
        tiers = tier_of(s, t1, t2)
        acc = np.zeros(s.shape[0], dtype=np.float64)
        cost = np.zeros(s.shape[0], dtype=np.float64)
        for tier in (0, K_mid, K_deep):
            m = (tiers == tier)
            acc[m] = correct_by_tier[tier][m]
            cost[m] = tier
        return acc.mean(), cost.mean()

    best = None
    grid_log = []
    for t1 in t1_grid:
        for t2 in t2_grid:
            if low_is_attack and t1 >= t2:
                continue
            if (not low_is_attack) and t1 <= t2:
                continue
            c_acc, c_cost = routed_acc_cost(
                s_clean[cal_mask],
                {0: rec['clean']['c0'][cal_mask], K_mid: rec['clean'][c_mid][cal_mask],
                 K_deep: rec['clean'][c_deep][cal_mask]}, t1, t2)
            a_acc, a_cost = routed_acc_cost(
                s_adv[cal_mask],
                {0: rec['adv']['c0'][cal_mask], K_mid: rec['adv'][c_mid][cal_mask],
                 K_deep: rec['adv'][c_deep][cal_mask]}, t1, t2)
            entry = {'t1': float(t1), 't2': float(t2),
                     'clean_acc': float(c_acc), 'adv_acc': float(a_acc),
                     'avg_cost': float(0.5 * (c_cost + a_cost)),
                     'objective': float(0.5 * (c_acc + a_acc)
                                        - 0.02 * (0.5 * (c_cost + a_cost)) / K_deep)}
            grid_log.append(entry)
            if best is None or entry['objective'] > best['objective']:
                best = entry

    t1, t2 = best['t1'], best['t2']
    log(f'best thresholds (calibration): t1={t1:.5f} t2={t2:.5f} '
        f'clean={best["clean_acc"]*100:.2f} adv={best["adv_acc"]*100:.2f} '
        f'avg_cost={best["avg_cost"]:.2f}')

    # ---------------- routed evaluation (held-out split) ----------------
    clean_acc, clean_cost = routed_acc_cost(
        s_clean[ev_mask], {0: rec['clean']['c0'][ev_mask],
                           K_mid: rec['clean'][c_mid][ev_mask],
                           K_deep: rec['clean'][c_deep][ev_mask]}, t1, t2)
    adv_acc, adv_cost = routed_acc_cost(
        s_adv[ev_mask], {0: rec['adv']['c0'][ev_mask],
                         K_mid: rec['adv'][c_mid][ev_mask],
                         K_deep: rec['adv'][c_deep][ev_mask]}, t1, t2)

    tiers_clean = tier_of(s_clean[ev_mask], t1, t2)
    tiers_adv = tier_of(s_adv[ev_mask], t1, t2)
    tier_fracs = lambda t: {str(k): float((t == k).mean()) for k in (0, K_mid, K_deep)}
    log(f'ROUTED (eval): clean acc={clean_acc*100:.2f} (avg steps={clean_cost:.2f}) | '
        f'adv acc={adv_acc*100:.2f} (avg steps={adv_cost:.2f})')
    log(f'  tier fractions clean: {tier_fracs(tiers_clean)}')
    log(f'  tier fractions adv  : {tier_fracs(tiers_adv)}')

    # ---------------- cost-accuracy frontier ----------------
    router = {'clean': float(clean_acc), 'adv': float(adv_acc),
              'cost_clean': float(clean_cost), 'cost_adv': float(adv_cost)}
    all_systems = dict(base)
    all_systems['router'] = router
    all_systems['oracle'] = oracle

    frontier = {}
    for rho in (0.0, 0.25, 0.5, 1.0):
        row = {}
        for name, st in all_systems.items():
            row[name] = {
                'overall_acc': (1 - rho) * st['clean'] + rho * st['adv'],
                'avg_steps': (1 - rho) * st['cost_clean'] + rho * st['cost_adv'],
            }
        frontier[f'rho_{rho}'] = row
        pretty = ' | '.join(f'{k}: acc={v["overall_acc"]*100:.2f} steps={v["avg_steps"]:.2f}'
                            for k, v in row.items())
        log(f'rho={rho}: {pretty}')

    results = {
        'config': vars(args),
        'signal_auroc_calibration': sig_stats,
        'signal_used': signal, 'low_is_attack': bool(low_is_attack),
        'thresholds': {'t1': t1, 't2': t2},
        'baselines_eval': base, 'oracle_eval': oracle,
        'routed_eval': {'clean_acc': float(clean_acc), 'adv_acc': float(adv_acc),
                        'clean_avg_steps': float(clean_cost),
                        'adv_avg_steps': float(adv_cost),
                        'clean_tier_fractions': tier_fracs(tiers_clean),
                        'adv_tier_fractions': tier_fracs(tiers_adv)},
        'frontier': frontier, 'grid': grid_log,
        'note': ('Routing signal (shift0) is computed for free inside the NAC '
                 'trajectory (step-0 gating forward); tier-0 samples pay one '
                 'extra forward only if the cascade must decide before running '
                 'any step. Costs here are counted in NAC gradient steps.'),
    }
    with open(os.path.join(args.outdir, 'routing_pilot.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    log(f'saved: {os.path.join(args.outdir, "routing_pilot.json")}')
    log_f.close()


if __name__ == '__main__':
    main()
