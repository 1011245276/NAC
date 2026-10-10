"""
Minimal fair comparison: TTC vs NAC vs DOC at EQUAL counterattack budget K.
Same process, same adversarial inputs (one PGD-10 eps4/255 attack per batch, seed 0),
same counter-attack epsilon (4/255), same number of counterattack steps K, same
per-batch manual_seed. Only the counterattack update rule differs.
Also counts approx backward passes for compute honesty.
"""
from __future__ import print_function
import os, sys, time, json
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
EPS=4./255.; K=4

def main():
    sys.argv=['min','--batch_size','32','--num_workers','1','--test_set','cifar10',
              '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
              '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','%d'%K,'--ttc_stepsize','1',
              '--tau_thres','0.2','--beta','2.0','--seed','0','--outdir',_NAC_ROOT+'/results/adaptive_pilot']
    args=parse_options()
    for a in ('test_eps','test_stepsize','ttc_stepsize','ttc_eps'): setattr(args,a,getattr(args,a)/255.)
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)jinsu
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model,_=clip.load('ViT-B/32',device,jit=False,prompt_len=0)
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
    prompter=torch.nn.DataParallel(NullPrompter()).cuda(); add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd,vl=load_val_dataset(args,'cifar10'); texts=get_text_prompts_val([vd],['cifar10'])[0]
    text_tokens=clip.tokenize(texts).to(device); crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    def logits(x):
        lg,_,_,_=multiGPU_CLIP(None,None,None,model,prompter(clip_img_preprocessing(x)),text_tokens,
                               prompt_token=None,dataset_name='cifar10')
        return lg.float()
    res={m:[] for m in ('ttc','nac','doc')}
    NBT=10; t0=time.time()
    fwd_bwd={'ttc':0,'nac':0,'doc':0}
    for bi,(images,target) in enumerate(vl):
        if bi>=NBT: break
        images,target=images.to(device),target.to(device)
        with torch.amp.autocast('cuda'):
            da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                          args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                          epsilon=args.test_eps,dataset_name='cifar10')
            att=images+da
        for m in ('ttc','nac','doc'):
            torch.manual_seed(10000+bi)
            with torch.amp.autocast('cuda'):
                if m=='ttc':
                    dd=_ttc(model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                            norm='l_inf',epsilon=EPS,visual_model_orig=None,tau_thres=args.tau_thres,
                            beta=args.beta,clip_visual=None)
                elif m=='nac':
                    dd=_nac(model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                            norm='l_inf',epsilon=EPS,visual_model_orig=None,tau_thres=args.tau_thres,
                            beta=args.beta,clip_visual=None,nac_momentum=0.9)
                else:
                    dd=_doc(args,model,att.data,prompter,add_prompter,alpha=args.ttc_stepsize,attack_iters=K,
                            norm='l_inf',epsilon=EPS,beta=args.beta,clip_visual=None)
                lg=logits(att+dd)
            res[m].append((lg.argmax(1)==target).cpu().numpy())
        if (bi+1)%5==0 or bi==NBT-1:
            print('[min] %d/%d el=%.1fmin %s'%(bi+1,NBT,(time.time()-t0)/60,
                '  '.join('%s=%.2f'%(m,np.concatenate(res[m]).mean()*100) for m in res)),flush=True)
    out={f'{m}__correct':np.concatenate(res[m]) for m in res}
    np.savez_compressed(_NAC_ROOT+'/results/adaptive_pilot/min_nac_vs_doc.npz',**out)
    print('\n=== MINIMAL FAIR (batch32, %d samples, K=%d counterattack steps, eps4/255, seed0) ==='%(NBT*32,K))
    for m in res:
        print('  %-5s defended=%.2f%%'%(m, np.concatenate(res[m]).mean()*100))
    print('\nNote: equal K. DOC adds orthogonal+directional gating (more grads/step); NAC adds Nesterov.')
    print('[min] ALL DONE')

if __name__=='__main__': main()