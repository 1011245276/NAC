"""
Robustness frontier: confirm NAC Pareto-dominance over TTC/DOC on the
accuracy-per-backward curve is NOT one-sided. Axes:
  - seeds {0,1,2}
  - attack eps {2,4}/255
  - K {1,2,4,6,8}
  - shared default hyperparams
Records defended accuracy; a dominance summary over all (seed,eps,K) says
whether NAC is argmax-everywhere or only sometimes.
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
from attacks import attack_pgd
from models.prompters import TokenPrompter,NullPrompter
KS=[1,2,4,6,8]; SEEDS=[0,1,2]; ATK_EPS=[2.,4.]

def main():
    sys.argv=['rb','--batch_size','32','--num_workers','1','--test_set','cifar10',
              '--test_attack_type','pgd','--test_numsteps','10','--test_stepsize','1',
              '--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
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
    NBT=12
    # store mean acc per (seed,eps,method,K)
    store={}  # (seed,eps,m): {K: acc}
    t0=time.time()
    for aep in ATK_EPS:
        args.test_eps=aep/255.
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            res={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}
            for bi,(images,target) in enumerate(vl):
                if bi>=NBT: break
                images,target=images.to(device),target.to(device)
                with torch.amp.autocast('cuda'):
                    da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                                  args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                                  epsilon=args.test_eps,dataset_name='cifar10')
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
                if (bi+1)%6==0 or bi==NBT-1:
                    print('[rb] eps=%g seed=%d %d/%d el=%.1fmin'%(aep,seed,bi+1,NBT,(time.time()-t0)/60),flush=True)
            for m in res:
                store[(seed,aep,m)]={K:float(np.concatenate(res[m][K]).mean()*100) for K in KS}
    # save data BEFORE reporting (idempotent / crash-safe)
    np.save(_NAC_ROOT+'/results/adaptive_pilot/robust_frontier.npy',
            {str(k):v for k,v in store.items()})
    # dominance summary
    print('\n=== ROBUSTNESS: NAC dominance over TTC/DOC across all (seed,eps,K) ===')
    nac_tops=ttc_tops=doc_tops=0; tot=0
    rows=[]
    for (seed,aep,m),v in store.items():
        rows.append((aep,seed,m,v))
    for aep in ATK_EPS:
        print('--- attack eps=%d/255 (mean over seeds %s) ---'%(aep,SEEDS))
        for K in KS:
            vals={m:np.mean([store[(s,aep,m)][K] for s in SEEDS]) for m in ('ttc','nac','doc')}
            best=max(vals,key=vals.get)
            if best=='nac': nac_tops+=1
            elif best=='ttc': ttc_tops+=1
            else: doc_tops+=1
            tot+=1
            print('K%-2d NAC %6.2f  TTC %6.2f  DOC %6.2f   best=%s'%(K,vals['nac'],vals['ttc'],vals['doc'],best))
    print('\nDominance count over %d (eps,K) cells (mean-over-seeds):  NAC-best=%d  TTC-best=%d  DOC-best=%d'%(
        tot,nac_tops,ttc_tops,doc_tops))
    print('NAC is best in %.0f%% of cells'%(100.0*nac_tops/tot))
    print('[rb] ALL DONE')

if __name__=='__main__': main()