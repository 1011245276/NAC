"""
Mechanism attribution: TTC vs Momentum vs Pure Look-ahead vs NAC.

Goal: why can NAC breach TTC's defended-accuracy ceiling? Compares the 4
counter-attack optimizers (identical tau-threshold framework, identical init
seed per batch, identical PGD adversary) on a fixed sample, K in {1,2,4,6,8}.
Calls the EXACT production functions (no reimplementation -> behaviourally
faithful), then records per-(method,K):
  - defended accuracy, margin
  - defended-feature distance to CLEAN anchor, to ADV anchor
  - perturbation usage: mean |delta|, fraction of pixels at +-eps boundary,
    L2 norm of delta
  - NEW recoveries at K (samples wrong at previous K, right at this K) -> plateau
These give empiric evidence for the ceiling/plateau question.

Small scale: 12 batches (384 samples). Output files under results/adaptive_pilot/.
"""
from __future__ import print_function
import os, sys, json
import numpy as np, torch
_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)
from nac_fair_experiment import (parse_options, ttc_counterattack,
                                 momentum_counterattack, pure_la_counterattack,
                                 nac_counterattack)
from func import device, clip_img_preprocessing, multiGPU_CLIP
from replace import clip

@torch.no_grad()
def _logits(model, prompter, imgs, text_tokens):
    lg, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                 prompter(clip_img_preprocessing(imgs)),
                                 text_tokens, prompt_token=None,
                                 dataset_name='cifar10')
    return lg.float()
from utils import (load_val_dataset, get_text_prompts_val, convert_models_to_fp32)
from attacks import attack_pgd
from models.prompters import TokenPrompter, NullPrompter

def run():
    sys.argv = ['attr','--batch_size','32','--num_workers','1','--test_set','cifar10',
                '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
                '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
                '--tau_thres','0.2','--beta','2.0','--nac_momentum','0.9',
                '--seed','0','--outdir', os.path.join(_NAC_ROOT,'results','adaptive_pilot')]
    args = parse_options()
    args.test_eps/=255.; args.test_stepsize/=255.; args.ttc_stepsize/=255.; args.ttc_eps/=255.
    import random
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    outdir = os.path.join(_NAC_ROOT,'results','adaptive_pilot')
    model,_ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model = torch.nn.DataParallel(model); model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd, vl = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([vd], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)
    crit = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    methods = {'ttc': ttc_counterattack, 'momentum': momentum_counterattack,
               'purela': pure_la_counterattack, 'nac': nac_counterattack}
    Ks = sorted(set([1,2,4,6,8]))
    # per-(method,K) accumulators
    acc = {m: {k: [] for k in Ks} for m in methods}
    margin = {m: {k: [] for k in Ks} for m in methods}
    dist_clean = {m: {k: [] for k in Ks} for m in methods}
    dist_adv = {m: {k: [] for k in Ks} for m in methods}
    perturb = {m: {k: {} for k in Ks} for m in methods}  # {key: np array}

    n_batches = 12
    for bi, (images, target) in enumerate(vl):
        if bi >= n_batches: break
        images, target = images.to(device), target.to(device)
        with torch.amp.autocast('cuda'):
            delta_atk = attack_pgd(args, prompter, model, None, None, add_prompter, crit,
                                   images, target, args.test_stepsize, args.test_numsteps,
                                   'l_inf', text_tokens=text_tokens, epsilon=args.test_eps,
                                   dataset_name='cifar10')
            attacked = images + delta_atk
        # anchors (clean, adv) features (ind_prompt=None -> standard visual)
        with torch.no_grad():
            fc = model.module.encode_image(
                prompter(clip_img_preprocessing(images)), None).float()
            fa = model.module.encode_image(
                prompter(clip_img_preprocessing(attacked)), None).float()
        for m, fun in methods.items():
            for k in Ks:
                torch.manual_seed(10000 + bi)  # identical init delta across all (m, K)
                kwargs = dict(alpha=args.ttc_stepsize, attack_iters=k, norm='l_inf',
                              epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                              beta=args.beta, clip_visual=None)
                with torch.amp.autocast('cuda'):
                    comp = attacked.data
                    if m == 'ttc':
                        dd = fun(model, comp, prompter, add_prompter, **kwargs)
                    elif m == 'momentum':
                        dd = fun(model, comp, prompter, add_prompter,
                                 preprocess_fn=clip_img_preprocessing,
                                 momentum_coef=args.nac_momentum, **kwargs)
                    elif m == 'purela':
                        dd = fun(model, comp, prompter, add_prompter,
                                 preprocess_fn=clip_img_preprocessing, **kwargs)
                    else:
                        dd = fun(model, comp, prompter, add_prompter,
                                 nac_momentum=args.nac_momentum, **kwargs)
                    with torch.no_grad():
                        fd = model.module.encode_image(
                            prompter(clip_img_preprocessing(attacked + dd)), None).float()
                        logits = _logits(model, prompter, attacked + dd, text_tokens)
                acc[m][k].append((logits.argmax(1) == target).cpu().numpy())
                pr = torch.softmax(logits.float(),1)
                top2 = pr.topk(2,1).values
                margin[m][k].append((top2[:,0]-top2[:,1]).cpu().numpy())
                dist_clean[m][k].append((fd - fc).norm(dim=1).cpu().numpy())
                dist_adv[m][k].append((fd - fa).norm(dim=1).cpu().numpy())
                de = dd.detach().float()
                flat = de.reshape(de.size(0), -1)
                perturb[m][k].setdefault('mean_abs',[]).append(flat.abs().mean(1).cpu().numpy())
                perturb[m][k].setdefault('frac_bound',[]).append(
                    (flat.abs() >= args.ttc_eps - 1e-4).float().mean(1).cpu().numpy())
                perturb[m][k].setdefault('l2',[]).append(flat.flatten(1).norm(2,1).cpu().numpy())

    R = {}
    for m in methods:
        for k in Ks:
            R[f'{m}_k{k}_acc'] = np.concatenate(acc[m][k])
            R[f'{m}_k{k}_margin'] = np.concatenate(margin[m][k])
            R[f'{m}_k{k}_dclean'] = np.concatenate(dist_clean[m][k])
            R[f'{m}_k{k}_dadv'] = np.concatenate(dist_adv[m][k])
            for key in ('mean_abs','frac_bound','l2'):
                R[f'{m}_k{k}_p_{key}'] = np.concatenate(perturb[m][k][key])
    np.savez_compressed(os.path.join(outdir, 'mechanism_attr.npz'), **R)
    json.dump({'methods': list(methods), 'Ks': Ks, 'n_batches': n_batches},
              open(os.path.join(outdir,'mechanism_attr_manifest.json'),'w'), indent=2)

    # Report table
    print('=== Mechanism attribution (eps=4/255, %d samples) ===' % (n_batches*32))
    print('%-9s' % 'method' + ''.join('K%-2d%7s  '%(k, '') for k in Ks))
    for m in ('ttc','momentum','purela','nac'):
        print('%-9s' % m + ''.join('%5.2f    '%(R[f'{m}_k{k}_acc'].mean()*100) for k in Ks))
    print()
    for k in Ks:
        base = R[f'ttc_k{k}_acc']
        print('K=%d acc  TTC %.2f | Mom %.2f | P_LA %.2f | NAC %.2f' % (
            k, base.mean()*100, R[f'momentum_k{k}_acc'].mean()*100,
            R[f'purela_k{k}_acc'].mean()*100, R[f'nac_k{k}_acc'].mean()*100))
    print('\n--- perturbation & feature-distance at K=8 ---')
    for m in ('ttc','momentum','purela','nac'):
        print('%s: mean|d|=%s frac@bound=%s d2clean=%.1f d2adv=%.1f' % (
            m, round(float(R[f'{m}_k8_p_mean_abs'].mean()),4),
            round(float(R[f'{m}_k8_p_frac_bound'].mean()),3),
            R[f'{m}_k8_dclean'].mean(), R[f'{m}_k8_dadv'].mean()))
    print('\n--- new recoveries (wrong at prev K -> right now), per method ---')
    prev = {m: None for m in methods}
    for k in [1,2,4,6,8]:
        for m in methods:
            cur = R[f'{m}_k{k}_acc'].astype(bool)
            if prev[m] is not None:
                dr = (~prev[m] & cur).sum()
                print('  K=%2d %-9s new_recover=%d' % (k, m, dr))
            prev[m] = cur
    print('ALL DONE')

if __name__ == '__main__':
    run()