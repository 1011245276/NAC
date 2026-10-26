# -*- coding: utf-8 -*-
"""
E4-S3 full clean paired validation (external ruling 2026-10-27, option a).

Purpose: adjudicate the pre-registered clean constraint 1 (drop of the
M_def=5 aggregate deployment relative to the M_def=1 single-run deployment
on CLEAN images, one-sided <= 0.5pp) on the FULL CIFAR-10 test set with the
pre-registered M_eval=10 stochastic repetitions, replacing the underpowered
256-sample smoke check.  NO attack is run; NO robustness matrix; the result
does not by itself unlock E5 (external ruling required).

Design
------
* Full CIFAR-10 test set (10,000 images), bs=32, same deployment namespace
  as every other run (ViT-B/32, K=2, alpha=1/255, eps_def=4/255, tau=0.2,
  beta=2.0, momentum=0.9, learnable_tau=0.155, T=75.0).
* Per (method in {ttc, nac}):
  - baseline : M_eval=10 independent deployments with M_def=1;
  - aggregate: M_eval=10 independent deployments with M_def=5; both
    mean-logits and majority-vote predictions are derived from the same
    M_def=5 member logits per repetition.
* Per-image, per-repetition binary correctness vectors are saved for every
  config, together with the seed registration (single global stream,
  seed=0, cudnn deterministic; reproducible by re-running with the same
  command).  The paired analysis (discordant counts, net drop, image-level
  bootstrap CI, one-sided verdict) is computed by
  scripts/e4_s3_paired_analysis.py from the saved vectors.
"""
import sys, os, json, logging, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import random

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (
    load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, clip_img_preprocessing,
    nac_counterattack, ttc_counterattack,
)

clip_load = clip.load
from utils import convert_models_to_fp32

device = "cuda" if torch.cuda.is_available() else "cpu"

M_EVAL = 10              # pre-registered repetition count (E0a protocol)
M_DEF_LIST = [1, 5]      # baseline vs aggregate


def deploy_aggregate(counter_fn, mk, model, images, text_tokens, prompter,
                     dataset, m_def):
    """Deployed aggregate defense: M_def independent counterattacks (fresh
    random inits), returns per-member logits [M_def, B, C].  Counter fns
    need autograd internally -> enable_grad wrapper (audited pattern)."""
    member_logits = []
    for _j in range(m_def):
        with torch.enable_grad():
            Delta = counter_fn(**mk(images))
        with torch.no_grad():
            defended = torch.clamp(images + Delta, 0, 1)
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
        member_logits.append(out)
    return torch.stack(member_logits, dim=0)


def aggregate_correct(logits_stack, target):
    """Correctness vectors for both aggregations from member logits
    [M_def, B, C]: (mean_correct, majority_correct)."""
    pred_mean = logits_stack.mean(dim=0).argmax(dim=1)
    member_preds = logits_stack.argmax(dim=2)
    modes = torch.mode(member_preds, dim=0).values   # smallest-label ties
    return ((pred_mean == target).cpu().tolist(),
            (modes == target).cpu().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--m_eval', type=int, default=M_EVAL)
    ap.add_argument('--max_batches', type=int, default=0,
                    help='smoke hook only (0 = full test set)')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = cli.outdir or os.path.join('results', 'e4_s3_clean_paired', stamp)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 's3_paired.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    sys.argv = [
        'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd', '--test_eps', '4.0',
        '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
        '--root', './data', '--dataset', 'cifar10', '--seed', '0', '--num_seeds', '1',
        '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
        '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
        '--learnable_tau', '0.155', '--temperature', '75.0',
        '--counterattack', 'ttc', '--outdir', outdir,
    ]
    from nac_fair_experiment import parse_options
    args = parse_options()
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    logging.info("E4-S3 full clean paired | methods=%s seed=%d M_eval=%d "
                 "M_def=%s", cli.methods, cli.seed, cli.m_eval, M_DEF_LIST)
    print(f"E4-S3 clean paired M_eval={cli.m_eval} M_def={M_DEF_LIST}; log:",
          log_path, flush=True)

    model, _ = clip_load('ViT-B/32', device, jit=False, prompt_len=0)
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

    def mk(name, x):
        if name == 'ttc':
            return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None)
        return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                    alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                    norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None,
                    nac_momentum=args.nac_momentum, clamp_lookahead=False)

    fns = {'ttc': ttc_counterattack, 'nac': nac_counterattack}
    out_json = os.path.join(outdir, 's3_paired_results.json')
    results = {}
    for name in cli.methods:
        random.seed(cli.seed); np.random.seed(cli.seed)
        torch.manual_seed(cli.seed); torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        fn = fns[name]
        mk_fn = lambda x, _n=name: mk(_n, x)
        base_vec, agg_mean_vec, agg_maj_vec, idx_vec = [], [], [], []
        t0 = time.time()
        logging.info("=== %s (seed %d) ===", name, cli.seed)
        for bi, (images, target) in enumerate(val_loader):
            if cli.max_batches and bi >= cli.max_batches:
                break
            images, target = images.to(device), target.to(device)
            for _m in range(cli.m_eval):
                # baseline deployment: M_def=1 (mean == majority at M_def=1)
                ls1 = deploy_aggregate(fn, mk_fn, model, images, text_tokens,
                                       prompter, 'cifar10', 1)
                c1, _ = aggregate_correct(ls1, target)
                base_vec.extend(c1)
                # aggregate deployment: M_def=5, both aggregations
                ls5 = deploy_aggregate(fn, mk_fn, model, images, text_tokens,
                                       prompter, 'cifar10', 5)
                cm, cj = aggregate_correct(ls5, target)
                agg_mean_vec.extend(cm)
                agg_maj_vec.extend(cj)
            idx_vec.extend((bi * args.batch_size
                            + torch.arange(images.size(0))).tolist())
            if bi % 20 == 0:
                nb = len(base_vec)
                logging.info("  %s batch %d: n=%d base=%.2f agg_mean=%.2f "
                             "agg_maj=%.2f", name, bi, nb,
                             100 * sum(base_vec) / max(nb, 1),
                             100 * sum(agg_mean_vec) / max(len(agg_mean_vec), 1),
                             100 * sum(agg_maj_vec) / max(len(agg_maj_vec), 1))
                print(f"  {name} batch {bi}: n={nb} base="
                      f"{100 * sum(base_vec) / max(nb, 1):.2f}", flush=True)
                torch.cuda.empty_cache()
        results[f"{name}_seed{cli.seed}"] = dict(
            n_images=len(idx_vec),
            m_eval=cli.m_eval,
            m_def_list=M_DEF_LIST,
            seed=cli.seed,
            elapsed_min=round((time.time() - t0) / 60, 1),
            per_image_reps=dict(
                image_index=idx_vec,
                base_correct_reps=base_vec,
                agg_mean_correct_reps=agg_mean_vec,
                agg_maj_correct_reps=agg_maj_vec),
            config=dict(method=name, dataset='cifar10', batch_size=32,
                        k=args.ttc_numsteps, eps_def=round(args.ttc_eps, 6),
                        tau_thres=args.tau_thres, beta=args.beta,
                        nac_momentum=args.nac_momentum,
                        learnable_tau=args.learnable_tau,
                        temperature=args.temperature,
                        attack='NONE (clean images only)',
                        note='majority tie-break = smallest label (disclosed); '
                             'cudnn deterministic; single global RNG stream'))
        with open(out_json, 'w') as f:
            json.dump(dict(config=(f"CIFAR-10 full clean paired validation: "
                                   f"M_eval={cli.m_eval} repetitions, "
                                   f"M_def in {M_DEF_LIST}, seed={cli.seed}, "
                                   f"no attack; per-image per-repetition "
                                   f"correctness vectors"),
                           results=results), f, indent=2)
        logging.info("%s done: n=%d elapsed=%.1fmin", name, len(idx_vec),
                     (time.time() - t0) / 60)
        print(f"{name} done: n={len(idx_vec)} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.cuda.empty_cache()
    logging.info("DONE")
    print("DONE:", flush=True)


if __name__ == '__main__':
    main()
