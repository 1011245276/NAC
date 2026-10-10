"""
Adaptive-signal pilot study for NAC (Phase-2 mechanism search).

Self-contained harness. Does NOT modify any core file. The instrumented NAC
below is a line-faithful copy of nac_fair_experiment.nac_counterattack with
per-sample signal capture added; an equivalence check against the original
(--equiv_check) verifies bitwise-identical outputs before any grid data is
trusted.

Protocol (pre-registered): results/adaptive_pilot/00_PROTOCOL.md

Grid: K in {1,2,3,4} x eps_ca in {1,2,4}/255 (NAC) + TTC(K=2, eps=4) protocol cell.
Controls:
  - adversarial batch generated ONCE and reused by all configs
  - torch.manual_seed(10000+batch_idx) before EVERY counterattack call so all
    configs share the same random init delta
  - attack = PGD-10 eps 4/255 step 1/255, seed 0, CIFAR-10, ViT-B/32

Outputs: results/adaptive_pilot/pilot_data.npz + pilot_manifest.json
"""
from __future__ import print_function

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)

from nac_fair_experiment import parse_options, nac_counterattack  # noqa: E402
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack  # noqa: E402
from func import device, clip_img_preprocessing, multiGPU_CLIP  # noqa: E402
from models.prompters import TokenPrompter, NullPrompter  # noqa: E402
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32  # noqa: E402
from attacks import attack_pgd  # noqa: E402
from replace import clip  # noqa: E402


# ---------------------------------------------------------------- utilities
def per_cos(a, b):
    """Per-sample cosine between flattened tensors [B, ...] -> [B] float32."""
    a = a.detach().reshape(a.size(0), -1).float()
    b = b.detach().reshape(b.size(0), -1).float()
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def per_norm(a):
    return a.detach().reshape(a.size(0), -1).float().norm(dim=1)


@torch.no_grad()
def embed(model, prompter, images, text_tokens, dataset_name):
    """Return (logits, image_features) exactly as the production eval does."""
    logits, _, img_feat, _ = multiGPU_CLIP(
        None, None, None, model, prompter(clip_img_preprocessing(images)),
        text_tokens, prompt_token=None, dataset_name=dataset_name
    )
    return logits, img_feat


@torch.no_grad()
def embed_ca(model, prompter, add_prompter, images):
    """Embedding via the EXACT call path used inside nac_counterattack
    (encode_image with add_prompter() token), so signal anchors are
    bitwise identical to the original function's internals."""
    return model.module.encode_image(
        prompter(clip_img_preprocessing(images)), add_prompter())


def per_sample_stats(logits, target):
    probs = torch.softmax(logits.float(), dim=1)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    pred = logits.argmax(dim=1)
    correct = (pred == target).cpu().numpy().astype(np.int8)
    return correct, margin, probs.cpu().numpy()


# ------------------------------------------------- instrumented NAC (pilot)
def instrumented_nac(model, X, prompter, add_prompter, alpha, attack_iters,
                     norm="l_inf", epsilon=0, tau_thres=0.2, beta=2.0,
                     nac_momentum=0.9, x_adv_emb=None, atk_delta=None):
    """Line-faithful copy of nac_fair_experiment.nac_counterattack
    (zero-shot path, clip_visual=None) + per-sample signal capture.
    Returns (Delta, signals_dict_of_per_sample_numpy)."""
    lower_limit, upper_limit = 0, 1

    def clamp(X_, lower, upper):
        return torch.max(torch.min(X_, upper), lower)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta, {}
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters == 0:
        return delta.data, {}

    # diff_ratio is None initially in the zero-shot production path
    # (clip_visual=None); computed at step 0 below. Identical to original.

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
    if x_adv_emb is not None:
        X_ori_reps = x_adv_emb.to(X_ori_reps.dtype)  # identical value, saved forward
    X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = []
    deltas_per_step.append(delta.data.clone())

    velocity = torch.zeros_like(delta)

    B = X.size(0)
    sig = {
        "grad_norm0": torch.zeros(B), "grad_norm_mean": torch.zeros(B),
        "cos_atkdelta_g0": torch.zeros(B),
        "grad_cos_succ": [], "step_align": [], "velocity_cos": [],
        "emb_jump": [], "loss_t": [], "lookahead_pred": [],
    }

    prev_grad, prev_vel_dir, prev_la_emb = None, None, None

    for _step_id in range(attack_iters):
        look_ahead = X + delta + nac_momentum * velocity
        prompted_images = prompter(clip_img_preprocessing(look_ahead))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if _step_id == 0:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        scheme_sign = (tau_thres - diff_ratio).sign()

        l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]

        # ---- signals (all from tensors already computed; no extra forwards)
        g_flat = grad.reshape(B, -1).float()
        loss_ps = (((X_att_reps.float() - X_ori_reps.float()) ** 2).sum(1)).detach().cpu()
        sig["loss_t"].append(loss_ps)
        if _step_id == 0:
            sig["grad_norm0"] = per_norm(grad).cpu()
            if atk_delta is not None:
                sig["cos_atkdelta_g0"] = per_cos(atk_delta, grad).cpu()
        else:
            sig["grad_cos_succ"].append(per_cos(grad, prev_grad).cpu())
            if prev_la_emb is not None:
                sig["emb_jump"].append(
                    (X_att_reps.detach().reshape(B, -1).float()
                     - prev_la_emb.reshape(B, -1).float()).norm(dim=1).cpu())
        sig["grad_norm_mean"] += per_norm(grad).cpu() / attack_iters
        prev_grad = grad.detach()

        velocity = nac_momentum * velocity + alpha * torch.sign(grad)
        if prev_vel_dir is not None:
            sig["velocity_cos"].append(per_cos(velocity, prev_vel_dir).cpu())
        prev_vel_dir = velocity.detach()

        d = delta[:, :, :, :] + velocity
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        else:
            raise ValueError
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())
        prev_la_emb = X_att_reps.detach()

        # step alignment + look-ahead predictiveness need next-step gradient;
        # accumulate (g_t, v_t) pairs and resolve after the loop.
        sig.setdefault("_g_hist", []).append(grad.detach())
        sig.setdefault("_v_hist", []).append(velocity.detach())
        sig.setdefault("_la_pred", []).append(None)  # placeholder

    # resolve step_align: cos(v_t, g_{t+1}); lookahead_pred: cos(g_t, g_{t+1})
    g_hist, v_hist = sig.pop("_g_hist"), sig.pop("_v_hist")
    la_pairs, sa_pairs = [], []
    for t in range(len(g_hist) - 1):
        la_pairs.append(per_cos(g_hist[t], g_hist[t + 1]).cpu())
        sa_pairs.append(per_cos(v_hist[t], g_hist[t + 1]).cpu())
    sig["lookahead_pred"] = la_pairs
    sig["step_align"] = sa_pairs

    # --- same step weighting as original TTC/NAC
    Delta = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1).unsqueeze(0).expand(B, -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)
    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(B, attack_iters + 1, 1, 1, 1)
    Delta = (weights * Delta).sum(dim=1)

    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    # aggregate per-step lists into per-sample means
    out = {}
    out["grad_norm0"] = sig["grad_norm0"].numpy()
    out["grad_norm_mean"] = sig["grad_norm_mean"].numpy()
    out["cos_atkdelta_g0"] = sig["cos_atkdelta_g0"].numpy()
    loss_ts = torch.stack(sig["loss_t"], 1).numpy()          # [B, K]
    out["gain_rate"] = (loss_ts[:, -1] - loss_ts[:, 0]) / attack_iters
    out["loss_final"] = loss_ts[:, -1]
    for k in ("grad_cos_succ", "step_align", "velocity_cos", "emb_jump", "lookahead_pred"):
        if len(sig[k]) > 0:
            out[k] = torch.stack(sig[k], 1).mean(1).numpy()
        else:
            out[k] = np.full(B, np.nan, dtype=np.float64)
    if len(sig["emb_jump"]) >= 2:
        jmp = torch.stack(sig["emb_jump"], 1).numpy()        # [B, K-1]
        out["disp_growth"] = (jmp[:, -1] - jmp[:, 0]) / np.maximum(jmp[:, 0], 1e-6)
    else:
        out["disp_growth"] = np.full(B, np.nan, dtype=np.float64)
    out["diff_ratio"] = diff_ratio.detach().float().cpu().numpy()  # = signal #1
    return Delta, out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_batches', type=int, default=0, help='0 = full test set')
    ap.add_argument('--equiv_check', action='store_true',
                    help='verify instrumented NAC == original nac_counterattack')
    ap.add_argument('--checkpoint_every', type=int, default=25)
    pilot_args = ap.parse_args()

    # Reuse the production parser for a protocol-identical Namespace.
    sys.argv = [
        'pilot', '--batch_size', '32', '--num_workers', '2',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd',
        '--test_eps', '4', '--test_numsteps', '10', '--test_stepsize', '1',
        '--ttc_eps', '4', '--ttc_numsteps', '2', '--ttc_stepsize', '1',
        '--tau_thres', '0.2', '--beta', '2.0', '--nac_momentum', '0.9',
        '--seed', '0', '--outdir', os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot'),
    ]
    args = parse_options()
    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.

    outdir = os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot')
    os.makedirs(outdir, exist_ok=True)

    seed = args.seed
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    val_dataset, val_loader = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)
    criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    # grid: (method, K, eps_idx) ; eps in 1/2/4 over 255
    eps_list = [1. / 255., 2. / 255., 4. / 255.]
    K_list = [1, 2, 3, 4]
    grid = [('nac', K, e) for e in range(3) for K in K_list]
    grid.append(('ttc', 2, 2))  # protocol-check cell

    store = {cfg: {'correct': [], 'margin': [], 'def_shift': [], 'def_cos': [],
                   'signals': {}} for cfg in grid}
    clean_c, adv_c, clean_m = [], [], []
    batch_meta = []

    n_batches = len(val_loader)
    if pilot_args.max_batches > 0:
        n_batches = min(n_batches, pilot_args.max_batches)
    total = n_batches * args.batch_size
    print(f'[pilot] batches={n_batches} configs={len(grid)} samples~={total}', flush=True)

    t0 = time.time()
    for bi, (images, target) in enumerate(val_loader):
        if bi >= n_batches:
            break
        images = images.to(device)
        target = target.to(device)

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                clean_logits, emb_clean = embed(model, prompter, images, text_tokens, 'cifar10')
            torch.cuda.empty_cache()
            delta_atk = attack_pgd(
                args, prompter, model, None, None, add_prompter, criterion_attack,
                images, target, args.test_stepsize, args.test_numsteps, 'l_inf',
                text_tokens=text_tokens, epsilon=args.test_eps, dataset_name='cifar10'
            )
            attacked = images + delta_atk
            with torch.no_grad():
                adv_logits, emb_adv_logits = embed(model, prompter, attacked, text_tokens, 'cifar10')
                adv_logits = adv_logits.float()
                # counterattack-internal path for signal anchors (bitwise identical
                # to nac_counterattack's X_ori_reps computation)
                emb_adv = embed_ca(model, prompter, add_prompter, attacked)

        cc, cm, _ = per_sample_stats(clean_logits.float(), target)
        ac, _, _ = per_sample_stats(adv_logits, target)
        clean_c.append(cc); adv_c.append(ac); clean_m.append(cm)

        for cfg in grid:
            method, K, e = cfg
            torch.manual_seed(10000 + bi)  # same init delta across configs
            with torch.amp.autocast('cuda'):
                if method == 'nac':
                    if pilot_args.equiv_check:
                        delta_def = nac_counterattack(
                            model, attacked.data, prompter, add_prompter,
                            alpha=args.ttc_stepsize, attack_iters=K,
                            norm='l_inf', epsilon=eps_list[e],
                            visual_model_orig=None, tau_thres=args.tau_thres,
                            beta=args.beta, clip_visual=None,
                            nac_momentum=args.nac_momentum)
                        torch.manual_seed(10000 + bi)
                    delta_def_inst, sig = instrumented_nac(
                        model, attacked.data, prompter, add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=K,
                        norm='l_inf', epsilon=eps_list[e],
                        tau_thres=args.tau_thres, beta=args.beta,
                        nac_momentum=args.nac_momentum,
                        x_adv_emb=emb_adv, atk_delta=delta_atk)
                    if pilot_args.equiv_check:
                        dmax = (delta_def - delta_def_inst).abs().max().item()
                        print(f'  [equiv] batch{bi} K={K} eps={e}: max|Δorig−Δpilot|={dmax:.3e}',
                              flush=True)
                    delta_use, signals = delta_def_inst, sig
                else:
                    delta_use = ttc_counterattack(
                        model, attacked.data, prompter, add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=K,
                        norm='l_inf', epsilon=eps_list[e],
                        visual_model_orig=None, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None)
                    signals = {}
                with torch.no_grad():
                    def_logits, def_emb = embed(model, prompter, attacked + delta_use,
                                                text_tokens, 'cifar10')
                    def_logits = def_logits.float()
            dc, dm, _ = per_sample_stats(def_logits, target)
            store[cfg]['correct'].append(dc)
            store[cfg]['margin'].append(dm)
            store[cfg]['def_shift'].append(
                ((def_emb.float() - emb_adv.float()).norm(dim=1)
                 / (emb_adv.float().norm(dim=1) + 1e-12)).cpu().numpy())
            store[cfg]['def_cos'].append(per_cos(def_emb, emb_adv).cpu().numpy())
            if signals:
                for k, v in signals.items():
                    store[cfg]['signals'].setdefault(k, []).append(v)

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            el = time.time() - t0
            print(f'[pilot] batch {bi+1}/{n_batches} elapsed={el/60:.1f}min '
                  f'eta={(el/(bi+1)*(n_batches-bi-1))/60:.1f}min', flush=True)
        if (bi + 1) % pilot_args.checkpoint_every == 0:
            _save(outdir, store, clean_c, adv_c, clean_m, grid, K_list, eps_list,
                  done=False, n_done=bi + 1)

    _save(outdir, store, clean_c, adv_c, clean_m, grid, K_list, eps_list,
          done=True, n_done=n_batches)
    print('[pilot] ALL DONE', flush=True)


def _save(outdir, store, clean_c, adv_c, clean_m, grid, K_list, eps_list,
          done, n_done):
    data, meta = {}, {'done': done, 'n_batches_done': n_done,
                      'K_list': K_list, 'eps_list': eps_list,
                      'grid': [f'{m}_K{K}_eps{e}' for (m, K, e) in grid]}
    for cfg in grid:
        key = f'{cfg[0]}_K{cfg[1]}_eps{cfg[2]}'
        data[f'{key}__correct'] = np.concatenate(store[cfg]['correct']) \
            if store[cfg]['correct'] else np.array([])
        data[f'{key}__margin'] = np.concatenate(store[cfg]['margin']) \
            if store[cfg]['margin'] else np.array([])
        data[f'{key}__def_shift'] = np.concatenate(store[cfg]['def_shift']) \
            if store[cfg]['def_shift'] else np.array([])
        data[f'{key}__def_cos'] = np.concatenate(store[cfg]['def_cos']) \
            if store[cfg]['def_cos'] else np.array([])
        for sk, chunks in store[cfg]['signals'].items():
            data[f'{key}__sig_{sk}'] = np.concatenate(chunks) if chunks else np.array([])
    data['clean_correct'] = np.concatenate(clean_c) if clean_c else np.array([])
    data['adv_correct'] = np.concatenate(adv_c) if adv_c else np.array([])
    data['clean_margin'] = np.concatenate(clean_m) if clean_m else np.array([])
    np.savez_compressed(os.path.join(outdir, 'pilot_data.npz'), **data)
    with open(os.path.join(outdir, 'pilot_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
