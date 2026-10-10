"""
Saturation probe (SMALL). Decisive question: does NAC reach a HIGHER defended
accuracy ceiling than TTC at large K, or do both asymptote to the SAME value?
If same ceiling -> NAC is just a faster re-parametrisation of TTC (analysis,
NOT a new method). If NAC ceiling strictly above -> qualitative difference.

Runs K in {1,2,3,4,6,8} on a fixed small sample (6 batches), eps=4/255,
identical protocol (same PGD adversary, same init seed, same code path with
nac_momentum 0.0=TTC / 0.9=NAC).
"""
from __future__ import print_function
import os, sys, json
import numpy as np, torch
_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)
from nac_fair_experiment import parse_options
from mechanism_trajectory import (embed, instrumented_ca, convert_models_to_fp32,
                                  clip, device, TokenPrompter, NullPrompter,
                                  load_val_dataset, get_text_prompts_val,
                                  attack_pgd)

def main():
    # Reset argv like the harness does
    sys.argv = ['sat','--batch_size','32','--num_workers','1','--test_set','cifar10',
                '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
                '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2',
                '--ttc_stepsize','1','--tau_thres','0.2','--beta','2.0',
                '--seed','0','--outdir', os.path.join(_NAC_ROOT,'results','adaptive_pilot')]
    args = parse_options()
    for a in ('test_eps','test_stepsize','ttc_stepsize','ttc_eps'):
        setattr(args, a, getattr(args,a)/255.)
    outdir = os.path.join(_NAC_ROOT,'results','adaptive_pilot')
    import random
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model,_=clip.load('ViT-B/32',device,jit=False,prompt_len=0)
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
    prompter=torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd,vl=load_val_dataset(args,'cifar10'); texts=get_text_prompts_val([vd],['cifar10'])[0]
    text_tokens=clip.tokenize(texts).to(device); crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    Ks=[1,2,3,4,6,8]; eps=4./255.
    acc={'ttc':{k:[] for k in Ks},'nac':{k:[] for k in Ks}}
    for bi,(images,target) in enumerate(vl):
        if bi>=6: break
        images,target=images.to(device),target.to(device)
        with torch.amp.autocast('cuda'):
            da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                          args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                          epsilon=args.test_eps,dataset_name='cifar10')
            att=images+da
        for name,mom in (('ttc',0.0),('nac',0.9)):
            for k in Ks:
                torch.manual_seed(10000+bi)
                with torch.amp.autocast('cuda'):
                    dd,_=instrumented_ca(model,att.data,prompter,add_prompter,
                              alpha=args.ttc_stepsize,attack_iters=k,epsilon=eps,
                              tau_thres=args.tau_thres,beta=args.beta,nac_momentum=mom)
                    with torch.no_grad():
                        lg=embed(model,prompter,att+dd,text_tokens,'cifar10').float()
                acc[name][k].append((lg.argmax(1)==target).cpu().numpy())
    res={name:{k:float(np.mean(acc[name][k])*100) for k in Ks} for name in ('ttc','nac')}
    print('=== Saturation probe (6 batches, 192 samples, eps=4/255) ===')
    print('K  :', '  '.join('K%d'%k for k in Ks))
    print('TTC:', '  '.join('%5.1f'%res['ttc'][k] for k in Ks))
    print('NAC:', '  '.join('%5.1f'%res['nac'][k] for k in Ks))
    print('gap:', '  '.join('%+4.1f'%(res['nac'][k]-res['ttc'][k]) for k in Ks))
    # ceiling deltas
    print('TTC K8-K4 = %+.1f   NAC K8-K4 = %+.1f'%(res['ttc'][8]-res['ttc'][4], res['nac'][8]-res['nac'][4]))
    print('TTC K8 = %.1f, NAC K8 = %.1f (gap %+.1f)'%(res['ttc'][8],res['nac'][8],res['nac'][8]-res['ttc'][8]))
    json.dump(res,open(os.path.join(outdir,'saturation_probe.json'),'w'),indent=2)
    np.savez_compressed(os.path.join(outdir,'saturation_probe.npz'),
        **{'ttc_k%d'%k:np.concatenate(acc['ttc'][k]) for k in Ks},
        **{'nac_k%d'%k:np.concatenate(acc['nac'][k]) for k in Ks})
    print('ALL DONE')
if __name__=='__main__':
    main()