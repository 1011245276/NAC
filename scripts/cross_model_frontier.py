"""
Cross-model / cross-dataset frontier: verify NAC equal-backward dominance is
not CIFAR-10/ViT-B/32-specific.
Cells:
  (model=ViT-B/16, dataset=cifar10)
  (model=ViT-B/32, dataset=tinyimagenet)
seeds {0,1,2} x K {1,2,4,6,8}, same attack (PGD-10 eps4/255) / same counter
eps / same per-batch init seed policy. Reports NAC-best cell coverage.
"""
from __future__ import print_function
import os, sys, time, datetime
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
KS=[1,2,4,6,8]; SEEDS=[0,1,2]; EPS=4./255.
NBT_CAP=40      # module-level so smoke tests can patch: M.NBT_CAP=2
PROG_EVERY=10   # print progress every N batches

def _save_seed(arch,dset,seed,acc,NBT):
    """Crash-safe incremental save: one npz per (cell,seed), self-documenting."""
    out=_NAC_ROOT+'/results/adaptive_pilot/frontier_%s_%s_seed%d.npz'%(arch.replace('/',''),dset,seed)
    kw={f'{m}_K{k}__acc':np.concatenate(acc[m][k]) for m in acc for k in KS}
    kw.update(arch=arch,dset=dset,seed=seed,n_batches=NBT,
              ts=datetime.datetime.now().isoformat(timespec='seconds'))
    np.savez_compressed(out,**kw)
    return out

def main():
    sys.argv=['cm','--batch_size','32','--num_workers','2','--test_set','cifar10',
              '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
              '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
              '--tau_thres','0.2','--beta','2.0','--seed','0','--outdir',_NAC_ROOT+'/results/adaptive_pilot']
    args=parse_options(); args.test_eps/=255.; args.test_stepsize/=255.; args.ttc_stepsize/=255.; args.ttc_eps/=255.
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    # ---- two cells: (arch,dataset) ----
    CELLS=[('ViT-B/16','cifar10'),('ViT-B/32','tinyImageNet')]
    for arch,dset in CELLS:
        t0=time.time()
        model,_=clip.load(arch,device,jit=False,prompt_len=0)
        for p in model.parameters(): p.requires_grad=False
        convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
        prompter=torch.nn.DataParallel(NullPrompter()).cuda(); add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
        # rewrite args for dataset
        args.test_set=[dset]
        if dset=='tinyImageNet':
            # tiny-imagenet-200 lives outside nac_project/data (see results/aggregated_all.json provenance)
            _tiny='F:/codex/NAC/data/tiny-imagenet-200/tiny-imagenet-200'
            if not os.path.isdir(os.path.join(_tiny,'val_')):
                raise FileNotFoundError('val_ (wnid-organized val set) missing under %s -- build it from val/val_annotations.txt first'%_tiny)
            args.tinyimagenet_root=_tiny
        vd,vl=load_val_dataset(args,dset)
        texts=get_text_prompts_val([vd],[dset])[0]; text_tokens=clip.tokenize(texts).to(device)
        crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
        def logits(x):
            lg,_,_,_=multiGPU_CLIP(None,None,None,model,prompter(clip_img_preprocessing(x)),text_tokens,
                                   prompt_token=None,dataset_name=dset); return lg.float()
        allacc={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}   # cumulative across seeds
        nb=len(vl); NBT=min(NBT_CAP,nb)   # ~1280 samples cifar / up to tiny (200*?) keep 40 batches for time
        for seed in SEEDS:
            acc={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}  # per-seed only
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            for bi,(images,target) in enumerate(vl):
                if bi>=NBT: break
                images,target=images.to(device),target.to(device)
                with torch.amp.autocast('cuda'):
                    da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                                  args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                                  epsilon=args.test_eps,dataset_name=dset)
                    att=images+da
                for m,fn in (('ttc',_ttc),('nac',_nac),('doc',_doc)):
                    for K in KS:
                        torch.manual_seed(10000+bi)
                        with torch.amp.autocast('cuda'):
                            if m=='ttc':
                                dd=fn(model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                                      norm='l_inf',epsilon=EPS,visual_model_orig=None,tau_thres=args.tau_thres,
                                      beta=args.beta,clip_visual=None)
                            elif m=='nac':
                                dd=fn(model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                                      norm='l_inf',epsilon=EPS,visual_model_orig=None,tau_thres=args.tau_thres,
                                      beta=args.beta,clip_visual=None,nac_momentum=0.9)
                            else:
                                dd=fn(args,model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                                      norm='l_inf',epsilon=EPS,beta=args.beta,clip_visual=None)
                            lg=logits(att+dd)
                        acc[m][K].append((lg.argmax(1)==target).cpu().numpy())
                if (bi+1)%PROG_EVERY==0 or bi==NBT-1:
                    print('[cm] %s/%s seed=%d %d/%d el=%.1fmin'%(arch,dset,seed,bi+1,NBT,(time.time()-t0)/60),flush=True)
            _save_seed(arch,dset,seed,acc,NBT)
            for m in acc:
                for k in KS: allacc[m][k]+=acc[m][k]
            v1={m:np.mean(np.concatenate(acc[m][4]))*100 for m in ('ttc','nac','doc')}
            print('[cm] %s/%s seed=%d SAVED (K4: NAC %.2f TTC %.2f DOC %.2f) el=%.1fmin'%(
                arch,dset,seed,v1['nac'],v1['ttc'],v1['doc'],(time.time()-t0)/60),flush=True)
        # report
        print('\n=== %s / %s  (%d batches/seed x %d seeds) ==='%(arch,dset,NBT,len(SEEDS)),flush=True)
        for K in KS:
            v={m:np.mean(np.concatenate(allacc[m][K]))*100 for m in ('ttc','nac','doc')}
            # mean over all seeds concatenated
            best=max(v,key=v.get)
            print('K%-2d NAC %6.2f  TTC %6.2f  DOC %6.2f   best=%s'%(K,v['nac'],v['ttc'],v['doc'],best),flush=True)
        np.savez_compressed(_NAC_ROOT+'/results/adaptive_pilot/frontier_%s_%s.npz'%(arch.replace('/',''),dset),
                            **{f'{m}_K{k}__acc':np.concatenate(allacc[m][k]) for m in allacc for k in KS})
        del model, prompter, add_prompter; torch.cuda.empty_cache()
    print('[cm] ALL DONE',flush=True)

if __name__=='__main__': main()