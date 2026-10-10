"""
Supplementary cells: TTC depth scaling at eps=4/255, K in {1,2,3,4}.

Mirrors adaptive_signal_pilot.py protocol EXACTLY (same loader, same PGD-10
attack at seed 0, same per-batch torch.manual_seed(10000+bi) before each
counterattack, batch_size 32) so cells are directly comparable with the NAC
grid in pilot_data.npz. Purpose: test whether plain-PGD TTC also improves
with depth K, or saturates (NAC-specific depth scalability claim).

Output: results/adaptive_pilot/ttc_depth.npz + ttc_depth_manifest.json
"""
from __future__ import print_function

import os
import sys
import json
import time

import numpy as np
import torch

_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)

from nac_fair_experiment import parse_options  # noqa: E402
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack  # noqa: E402
from func import device, clip_img_preprocessing, multiGPU_CLIP  # noqa: E402
from models.prompters import TokenPrompter, NullPrompter  # noqa: E402
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32  # noqa: E402
from attacks import attack_pgd  # noqa: E402
from replace import clip  # noqa: E402


@torch.no_grad()
def embed(model, prompter, images, text_tokens, dataset_name):
    logits, _, _, _ = multiGPU_CLIP(
        None, None, None, model, prompter(clip_img_preprocessing(images)),
        text_tokens, prompt_token=None, dataset_name=dataset_name
    )
    return logits


def _save_partial(outdir, store, K_list, n_done):
    """Incremental checkpoint (non-final) so a crash never loses the run."""
    data = {}
    for K in K_list:
        if store[K]['correct']:
            data['ttc_K%d_eps2__correct' % K] = np.concatenate(store[K]['correct'])
            data['ttc_K%d_eps2__margin' % K] = np.concatenate(store[K]['margin'])
    np.savez_compressed(os.path.join(outdir, 'ttc_depth_partial.npz'), **data)
    with open(os.path.join(outdir, 'ttc_depth_partial_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'done': False, 'n_batches_done': n_done, 'K_list': K_list,
                   'eps': '4/255'}, f, ensure_ascii=False, indent=2)


def main():
    sys.argv = [
        'ttc_depth', '--batch_size', '32', '--num_workers', '2',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd',
        '--test_eps', '4', '--test_numsteps', '10', '--test_stepsize', '1',
        '--ttc_eps', '4', '--ttc_numsteps', '2', '--ttc_stepsize', '1',
        '--tau_thres', '0.2', '--beta', '2.0',
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

    eps = 4. / 255.
    K_list = [1, 2, 3, 4]
    store = {K: {'correct': [], 'margin': []} for K in K_list}

    n_batches = len(val_loader)
    print(f'[ttc_depth] batches={n_batches} K_list={K_list} eps=4/255', flush=True)

    t0 = time.time()
    for bi, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        with torch.amp.autocast('cuda'):
            torch.cuda.empty_cache()
            delta_atk = attack_pgd(
                args, prompter, model, None, None, add_prompter, criterion_attack,
                images, target, args.test_stepsize, args.test_numsteps, 'l_inf',
                text_tokens=text_tokens, epsilon=args.test_eps, dataset_name='cifar10'
            )
            attacked = images + delta_atk

        for K in K_list:
            torch.manual_seed(10000 + bi)  # same init delta across K (as pilot)
            with torch.amp.autocast('cuda'):
                delta_def = ttc_counterattack(
                    model, attacked.data, prompter, add_prompter,
                    alpha=args.ttc_stepsize, attack_iters=K,
                    norm='l_inf', epsilon=eps,
                    visual_model_orig=None, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None)
                with torch.no_grad():
                    def_logits = embed(model, prompter, attacked + delta_def,
                                       text_tokens, 'cifar10').float()
            probs = torch.softmax(def_logits, dim=1)
            pred = def_logits.argmax(dim=1)
            store[K]['correct'].append((pred == target).cpu().numpy())
            top2 = probs.topk(2, dim=1).values
            store[K]['margin'].append((top2[:, 0] - top2[:, 1]).cpu().numpy())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            el = time.time() - t0
            print(f'[ttc_depth] batch {bi+1}/{n_batches} elapsed={el/60:.1f}min '
                  f'eta={(el/(bi+1)*(n_batches-bi-1))/60:.1f}min', flush=True)

        if (bi + 1) % 25 == 0 or bi == n_batches - 1:
            _save_partial(outdir, store, K_list, bi + 1)

    data = {}
    for K in K_list:
        data['ttc_K%d_eps2__correct' % K] = np.concatenate(store[K]['correct'])
        data['ttc_K%d_eps2__margin' % K] = np.concatenate(store[K]['margin'])
    data['ttc_K2_eps2__correct_ref_pilot'] = np.load(
        os.path.join(outdir, 'pilot_data.npz'))['ttc_K2_eps2__correct']
    np.savez_compressed(os.path.join(outdir, 'ttc_depth.npz'), **data)
    with open(os.path.join(outdir, 'ttc_depth_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'done': True, 'n_batches': n_batches, 'K_list': K_list,
                   'eps': '4/255'}, f, ensure_ascii=False, indent=2)
    for K in K_list:
        print('[ttc_depth] TTC K=%d eps=4/255 acc: %.2f%%' %
              (K, data['ttc_K%d_eps2__correct' % K].mean() * 100), flush=True)
    print('[ttc_depth] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
