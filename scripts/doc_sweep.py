"""
DOC hyperparameter sweep probe: does DOC close the gap to NAC with dedicated
tuning, and does the tuned config transfer across datasets?

Background: all frontier/robustness runs used DOC's README default
(temperature=75, learnable_tau=0.155) under the shared-default protocol.
The earlier minimal experiment suggested (t=20, tau=0.3) can approach NAC on
one cell. This sweep quantifies that:

  Cell A (ViT-B/32 x cifar10, eps=4/255, 12 batches x 3 seeds, K{2,4,6,8}):
      DOC variants: (t75,tau.155)=README default [also serves as protocol-
      equivalence sanity check vs robust_frontier.npy],
                    (t20,tau.3)=paper-canonical "tuned",
                    (t20,tau.155) / (t75,tau.3) = single-factor decomposition
      Refs: NAC/TTC from robust_frontier.npy (identical protocol & RNG streams:
      same seed init, same per-batch attack, same per-CA reseed 10000+bi).

  Cell B (ViT-B/32 x tinyImageNet, 40 batches x 3 seeds, K{2,4,6,8}):
      DOC variant: (t20,tau.3) only -> does the CIFAR-tuned config transfer?
      Refs: NAC/TTC from frontier_ViT-B32_tinyImageNet_seed{0,1,2}.npz
      (identical protocol, same RNG policy).

K=1 excluded everywhere (established dead zone, disclosed separately).
Incremental per-(cell,variant,seed) npz saves with meta.
"""
from __future__ import print_function
import os, sys, time, datetime, ast
import numpy as np, torch
_NAC_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,_NAC_ROOT)
from nac_fair_experiment import parse_options, doc_counterattack as _doc
from func import device, clip_img_preprocessing, multiGPU_CLIP
from replace import clip
from utils import load_val_dataset,get_text_prompts_val,convert_models_to_fp32
from attacks import attack_pgd
from models.prompters import TokenPrompter,NullPrompter

SEEDS=[0,1,2]; KS=[2,4,6,8]
NBT_CAP={'cifar10':12,'tinyImageNet':40}   # matches robustness / cross_model protocols
PROG_EVERY=6
VARIANTS_A=[('doc_t75_tau155',75.,0.155),   # README default (as used in all frontier runs)
            ('doc_t20_tau30',20.,0.3),      # paper-canonical "tuned"
            ('doc_t20_tau155',20.,0.155),   # factor decomp: temp only
            ('doc_t75_tau30',75.,0.3)]      # factor decomp: tau only
VARIANTS_B=[('doc_t20_tau30',20.,0.3)]      # transfer test only

def _save(cell,var,seed,acc,NBT):
    out=_NAC_ROOT+'/results/adaptive_pilot/docsweep_%s_%s_seed%d.npz'%(cell,var,seed)
    kw={f'K{k}__acc':np.concatenate(acc[k]) for k in KS}
    kw.update(cell=cell,variant=var,seed=seed,n_batches=NBT,
              ts=datetime.datetime.now().isoformat(timespec='seconds'))
    np.savez_compressed(out,**kw); return out

def main():
    sys.argv=['ds','--batch_size','32','--num_workers','2','--test_set','cifar10',
              '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
              '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
              '--tau_thres','0.2','--beta','2.0','--seed','0','--outdir',_NAC_ROOT+'/results/adaptive_pilot']
    args=parse_options(); args.test_eps/=255.; args.test_stepsize/=255.; args.ttc_stepsize/=255.; args.ttc_eps/=255.
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model,_=clip.load('ViT-B/32',device,jit=False,prompt_len=0)   # both cells use B/32
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
    prompter=torch.nn.DataParallel(NullPrompter()).cuda(); add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
    crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    t00=time.time()
    for cell,variants in (('cifar10',VARIANTS_A),('tinyImageNet',VARIANTS_B)):
        t0=time.time()
        args.test_set=[cell]
        if cell=='tinyImageNet':
            _tiny='F:/codex/NAC/data/tiny-imagenet-200/tiny-imagenet-200'
            if not os.path.isdir(os.path.join(_tiny,'val_')):
                raise FileNotFoundError('val_ missing under %s'%_tiny)
            args.tinyimagenet_root=_tiny
        vd,vl=load_val_dataset(args,cell)
        texts=get_text_prompts_val([vd],[cell])[0]; text_tokens=clip.tokenize(texts).to(device)
        def logits(x):
            lg,_,_,_=multiGPU_CLIP(None,None,None,model,prompter(clip_img_preprocessing(x)),text_tokens,
                                   prompt_token=None,dataset_name=cell); return lg.float()
        NBT=min(NBT_CAP[cell],len(vl))
        results={}  # var -> {K: [per-seed mean acc]}
        for var,temp,tau in variants:
            args.temperature=temp; args.learnable_tau=tau
            for seed in SEEDS:
                np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                acc={k:[] for k in KS}
                for bi,(images,target) in enumerate(vl):
                    if bi>=NBT: break
                    images,target=images.to(device),target.to(device)
                    with torch.amp.autocast('cuda'):
                        da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                                      args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                                      epsilon=args.test_eps,dataset_name=cell)
                        att=images+da
                    for K in KS:
                        torch.manual_seed(10000+bi)
                        with torch.amp.autocast('cuda'):
                            dd=_doc(args,model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,
                                    attack_iters=K,norm='l_inf',epsilon=args.ttc_eps,beta=args.beta,
                                    clip_visual=None)
                            lg=logits(att+dd)
                        acc[K].append((lg.argmax(1)==target).cpu().numpy())
                    if (bi+1)%PROG_EVERY==0 or bi==NBT-1:
                        print('[ds] %s %s seed=%d %d/%d el=%.1fmin'%(cell,var,seed,bi+1,NBT,(time.time()-t0)/60),flush=True)
                _save(cell,var,seed,acc,NBT)
                v4=np.concatenate(acc[4]).mean()*100
                print('[ds] %s %s seed=%d SAVED (K4=%.2f) el=%.1fmin'%(cell,var,seed,v4,(time.time()-t0)/60),flush=True)
                results.setdefault(var,{k:[] for k in KS})
                for k in KS: results[var][k].append(np.concatenate(acc[k]).mean()*100)
        # ---- refs ----
        if cell=='cifar10':
            ref=np.load(_NAC_ROOT+'/results/adaptive_pilot/robust_frontier.npy',allow_pickle=True).item()
            ref={ast.literal_eval(k):v for k,v in ref.items()}
            nac={K:float(np.mean([ref[(s,4.0,'nac')][K] for s in SEEDS])) for K in KS}
            ttc={K:float(np.mean([ref[(s,4.0,'ttc')][K] for s in SEEDS])) for K in KS}
            docd={K:float(np.mean([ref[(s,4.0,'doc')][K] for s in SEEDS])) for K in KS}
            print('\n[ds] sanity: DOC-default rerun vs robust_frontier DOC (eps=4):')
            for K in KS:
                r=np.mean(results['doc_t75_tau155'][K])
                print('  K%-2d rerun %.2f  ref %.2f  diff %+.2f'%(K,r,docd[K],r-docd[K]),flush=True)
        else:
            sdata={s:np.load(_NAC_ROOT+'/results/adaptive_pilot/frontier_ViT-B32_tinyImageNet_seed%d.npz'%s) for s in SEEDS}
            nac={K:float(np.mean([sdata[s]['nac_K%d__acc'%K].mean() for s in SEEDS])*100) for K in KS}
            ttc={K:float(np.mean([sdata[s]['ttc_K%d__acc'%K].mean() for s in SEEDS])*100) for K in KS}
        print('\n=== DOC SWEEP  cell=%s  (B/32, eps=4/255, %d bt x %d seeds; refs: NAC/TTC same protocol) ==='%(
            cell,NBT,len(SEEDS)),flush=True)
        hdr='K   NAC*  TTC*  '+''.join('%-14s'%v for v in results)
        print(hdr,flush=True)
        nac_beats=0
        for K in KS:
            row='K%-2d %5.2f %5.2f '%(K,nac[K],ttc[K])
            best_doc=max(np.mean(results[v][K]) for v in results)
            for v in results: row+='%12.2f'%np.mean(results[v][K])
            row+='   best=doc' if best_doc>nac[K] else '   best=nac'
            if best_doc<=nac[K]: nac_beats+=1
            print(row,flush=True)
        print('[ds] cell %s: NAC >= all DOC variants in %d/%d K cells'%(cell,nac_beats,len(KS)),flush=True)
    print('[ds] ALL DONE  total %.1fmin'%((time.time()-t00)/60),flush=True)

if __name__=='__main__': main()
