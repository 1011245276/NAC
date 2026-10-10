"""
Attack-type axis: does NAC equal-backward dominance hold under CW attack
(margin-50 C&W loss) instead of PGD-CE?

Mirrors robustness_frontier.py exactly (ViT-B/32 x cifar10, eps {2,4}/255,
K {1,2,4,6,8}, seeds {0,1,2}, 12 batches/seed, shared default hyperparams,
same per-batch attack / per-CA reseed(10000+bi) RNG policy) with ONE change:
the attack is attack_CW instead of attack_pgd. PGD refs are loaded from
robust_frontier.npy for side-by-side comparison.

Crash-safe: accumulated store saved to cw_frontier.npy after every (eps,seed).
"""
from __future__ import print_function
import os, sys, time
import numpy as np, torch
_NAC_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,_NAC_ROOT)
from nac_fair_experiment import parse_options, nac_counterattack as _nac, doc_counterattack as _doc
from test_time_counterattack import tau_thres_weighted_counterattacks as _ttc
from func import device, clip_img_preprocessing, multiGPU_CLIP
from replace import clip
from utils import load_val_dataset,get_text_prompts_val,convert_models_to_fp32
from attacks import attack_CW
from models.prompters import TokenPrompter,NullPrompter
KS=[1,2,4,6,8]; SEEDS=[0,1,2]; ATK_EPS=[2.,4.]; NBT_CAP=12; PROG_EVERY=6

def main():
    sys.argv=['cw','--batch_size','32','--num_workers','2','--test_set','cifar10',
              '--test_attack_type','CW','--test_eps','4','--test_numsteps','10',
              '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
              '--tau_thres','0.2','--beta','2.0','--seed','0','--outdir',_NAC_ROOT+'/results/adaptive_pilot']
    args=parse_options(); args.test_stepsize/=255.; args.ttc_stepsize/=255.; args.ttc_eps/=255.
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model,_=clip.load('ViT-B/32',device,jit=False,prompt_len=0)
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
    prompter=torch.nn.DataParallel(NullPrompter()).cuda(); add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd,vl=load_val_dataset(args,'cifar10'); texts=get_text_prompts_val([vd],['cifar10'])[0]
    text_tokens=clip.tokenize(texts).to(device); crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    def logits(x):
        lg,_,_,_=multiGPU_CLIP(None,None,None,model,prompter(clip_img_preprocessing(x)),text_tokens,
                               prompt_token=None,dataset_name='cifar10'); return lg.float()
    NBT=min(NBT_CAP,len(vl))
    store={}
    t0=time.time()
    for aep in ATK_EPS:
        eps=aep/255.
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            res={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}
            for bi,(images,target) in enumerate(vl):
                if bi>=NBT: break
                images,target=images.to(device),target.to(device)
                with torch.amp.autocast('cuda'):
                    da=attack_CW(args,prompter,model,None,None,add_prompter,crit,images,target,
                                 text_tokens,args.test_stepsize,args.test_numsteps,'l_inf',epsilon=eps)
                    att=images+da
                for m,fn in (('ttc',_ttc),('nac',_nac),('doc',_doc)):
                    for K in KS:
                        torch.manual_seed(10000+bi)
                        with torch.amp.autocast('cuda'):
                            if m=='ttc':
                                dd=fn(model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                                      norm='l_inf',epsilon=args.ttc_eps,visual_model_orig=None,tau_thres=args.tau_thres,
                                      beta=args.beta,clip_visual=None)
                            elif m=='nac':
                                dd=fn(model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                                      norm='l_inf',epsilon=args.ttc_eps,visual_model_orig=None,tau_thres=args.tau_thres,
                                      beta=args.beta,clip_visual=None,nac_momentum=0.9)
                            else:
                                dd=fn(args,model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                                      norm='l_inf',epsilon=args.ttc_eps,beta=args.beta,clip_visual=None)
                            lg=logits(att+dd)
                        res[m][K].append((lg.argmax(1)==target).cpu().numpy())
                if (bi+1)%PROG_EVERY==0 or bi==NBT-1:
                    print('[cw] eps=%g seed=%d %d/%d el=%.1fmin'%(aep,seed,bi+1,NBT,(time.time()-t0)/60),flush=True)
            for m in res:
                store[(seed,aep,m)]={K:float(np.concatenate(res[m][K]).mean()*100) for K in KS}
            np.save(_NAC_ROOT+'/results/adaptive_pilot/cw_frontier.npy',
                    {str(k):v for k,v in store.items()})   # crash-safe: grows after each (eps,seed)
    # ---- report ----
    ref=np.load(_NAC_ROOT+'/results/adaptive_pilot/robust_frontier.npy',allow_pickle=True).item()
    print('\n=== CW ATTACK AXIS: NAC vs TTC vs DOC (B/32 x cifar10, CW-10) ===',flush=True)
    nac_tops=doc_tops=oth=0
    for aep in ATK_EPS:
        print('--- CW eps=%d/255 (mean over seeds %s)   [PGD ref in brackets] ---'%(aep,SEEDS),flush=True)
        for K in KS:
            vals={m:np.mean([store[(s,aep,m)][K] for s in SEEDS]) for m in ('ttc','nac','doc')}
            pgd={m:np.mean([ref['(%d, %s, \'%s\')'%(s,aep,m)][K] for s in SEEDS]) for m in ('ttc','nac','doc')}
            best=max(vals,key=vals.get)
            if best=='nac': nac_tops+=1
            elif best=='doc': doc_tops+=1
            else: oth+=1
            print('K%-2d NAC %5.2f[%5.2f]  TTC %5.2f[%5.2f]  DOC %5.2f[%5.2f]  best=%s'%(
                K,vals['nac'],pgd['nac'],vals['ttc'],pgd['ttc'],vals['doc'],pgd['doc'],best),flush=True)
    print('\nCW dominance over %d (eps,K) cells: NAC-best=%d DOC-best=%d other=%d'%(len(ATK_EPS)*len(KS),nac_tops,doc_tops,oth),flush=True)
    print('[cw] ALL DONE',flush=True)

if __name__=='__main__': main()
