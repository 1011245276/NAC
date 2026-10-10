"""
Budget-frontier: TTC vs NAC vs DOC at K in {1,2,4,6,8}, EQUAL-per-backward.
Instruments actual backward-pass count (autograd.grad / Tensor.backward) per
counterattack call + wall latency. Same attack/seed/eps as prior runs.
Output: results/adaptive_pilot/frontier.npz + printed table.
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
EPS=4./255.; KS=[1,2,4,6,8]

# ---- backward-pass counter (scope per counterattack call) ----
_COUNTER=[0]
_orig_grad=torch.autograd.grad
def _wrap_grad(*a,**k):
    _COUNTER[0]+=1; return _orig_grad(*a,**k)
_orig_bw=torch.Tensor.backward
def _wrap_bw(self,*a,**k):
    _COUNTER[0]+=1; return _orig_bw(self,*a,**k)

def main():
    sys.argv=['fr','--batch_size','32','--num_workers','1','--test_set','cifar10',
              '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
              '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
              '--tau_thres','0.2','--beta','2.0','--seed','0','--outdir',_NAC_ROOT+'/results/adaptive_pilot']
    args=parse_options()
    for a in ('test_eps','test_stepsize','ttc_stepsize','ttc_eps'): setattr(args,a,getattr(args,a)/255.)
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
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
    acc={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}
    bw ={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}
    lat={m:{k:[] for k in KS} for m in ('ttc','nac','doc')}
    NBT=16; t0=time.time()
    torch.autograd.grad=_wrap_grad; torch.Tensor.backward=_wrap_bw
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
                _COUNTER[0]=0; st=time.time()
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
                el=time.time()-st
                acc[m][K].append((lg.argmax(1)==target).cpu().numpy())
                bw[m][K].append(_COUNTER[0]); lat[m][K].append(el)
        if (bi+1)%8==0 or bi==NBT-1:
            print('[fr] %d/%d el=%.1fmin'%(bi+1,NBT,(time.time()-t0)/60),flush=True)
    torch.autograd.grad=_orig_grad; torch.Tensor.backward=_orig_bw
    data={}
    for m in acc: 
        for K in KS: data[f'{m}_K{K}__acc']=np.concatenate(acc[m][K])
    np.savez_compressed(_NAC_ROOT+'/results/adaptive_pilot/frontier.npz',**data)
    print('\n=== budget-frontier (eps4/255, %d samples, same attack) ==='%(NBT*32))
    print('%-5s'+''.join('%7s'%('K=%d'%K) for K in KS))
    for m in acc:
        a=' '.join('%6.2f%%'%(np.concatenate(acc[m][K]).mean()*100) for K in KS)
        bb=' '.join('%3.0f'%(np.mean(bw[m][K])) for K in KS)
        print('%-5s acc:'%m, a)
        print('     bwd  :', bb)
    print('\navg wall-ms/call (per K):')
    for m in acc: print('  ',m, ' '.join('%5.1f'%(np.mean(lat[m][K])*1000) for K in KS))
    print('[fr] ALL DONE')

if __name__=='__main__': main()