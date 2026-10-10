"""Equivalence check: instrumented fns (final_mechanism_verify) vs production
counter-attacks (nac_fair_experiment) must be bitwise identical on Delta.
Runs 1 batch. Failure -> the instrumented fns must NOT be trusted.
"""
from __future__ import print_function
import os, sys, numpy as np, torch
_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT); sys.path.insert(0, os.path.join(_NAC_ROOT, 'scripts'))
from nac_fair_experiment import (parse_options, ttc_counterattack,
                                 momentum_counterattack, pure_la_counterattack,
                                 nac_counterattack)
from final_mechanism_verify import FUNS, EPS, OUTDIR
from func import device, clip_img_preprocessing
from replace import clip
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32
from attacks import attack_pgd
from models.prompters import TokenPrompter, NullPrompter

def main():
    seed = 0
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    sys.argv = ['eq', '--batch_size', '32', '--num_workers', '1', '--test_set', 'cifar10',
                '--test_attack_type', 'pgd', '--test_eps', '4', '--test_numsteps', '10',
                '--test_stepsize', '1', '--ttc_eps', '4', '--ttc_numsteps', '2',
                '--ttc_stepsize', '1', '--tau_thres', '0.2', '--beta', '2.0',
                '--seed', '0', '--outdir', OUTDIR]
    args = parse_options()
    args.test_eps /= 255.; args.test_stepsize /= 255.; args.ttc_stepsize /= 255.; args.ttc_eps /= 255.
    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters(): p.requires_grad = False
    convert_models_to_fp32(model); model = torch.nn.DataParallel(model); model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd, vl = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([vd], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)
    crit = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    images, target = next(iter(vl)); images, target = images.to(device), target.to(device)
    with torch.amp.autocast('cuda'):
        da = attack_pgd(args, prompter, model, None, None, add_prompter, crit, images, target,
                        args.test_stepsize, args.test_numsteps, 'l_inf', text_tokens=text_tokens,
                        epsilon=args.test_eps, dataset_name='cifar10')
        att = images + da
    prod = {'ttc': ttc_counterattack, 'momentum': momentum_counterattack,
            'purela': pure_la_counterattack, 'nac': nac_counterattack}
    PKW = {'ttc': {'clip_visual': None},
           'momentum': {'clip_visual': None, 'preprocess_fn': clip_img_preprocessing, 'momentum_coef': 0.9},
           'purela': {'clip_visual': None, 'preprocess_fn': clip_img_preprocessing},
           'nac': {'clip_visual': None, 'nac_momentum': 0.9}}
    print('=== equiv check (K=4, eps=4/255, seed %d) ===' % seed, flush=True)
    ok = True
    for m in ['ttc', 'momentum', 'purela', 'nac']:
        torch.manual_seed(10000 + 0)
        with torch.amp.autocast('cuda'):
            dd_prod = prod[m](model, att.data, prompter, add_prompter,
                              alpha=args.ttc_stepsize, attack_iters=4, norm='l_inf',
                              epsilon=EPS, tau_thres=args.tau_thres, beta=args.beta,
                              **PKW[m])
        torch.manual_seed(10000 + 0)
        with torch.amp.autocast('cuda'):
            if m in ('nac', 'momentum'):
                dd_inst, *_ = FUNS[m](model, att.data, prompter, add_prompter,
                                      alpha=args.ttc_stepsize, K=4, eps=EPS,
                                      tau=args.tau_thres, beta=args.beta, mu=0.9)
            else:
                dd_inst, *_ = FUNS[m](model, att.data, prompter, add_prompter,
                                      alpha=args.ttc_stepsize, K=4, eps=EPS,
                                      tau=args.tau_thres, beta=args.beta)
        dmax = (dd_prod - dd_inst).abs().max().item()
        print('  %-10s max|prod-inst| = %.3e %s' % (m, dmax, 'OK' if dmax == 0 else 'FAIL'), flush=True)
        ok = ok and (dmax == 0)
    print('EQUIV %s' % ('PASS' if ok else 'FAIL'), flush=True)

if __name__ == '__main__':
    main()