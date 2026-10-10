"""Isolate PyTorch SDPA repeatability at model dimensions; not a model audit."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers.integrations.sdpa_attention import repeat_kv
from native_gradient_compare import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--tokens',type=int,default=16249)
    p.add_argument('--heads',type=int,default=24);p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--head-dim',type=int,default=256)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(20261009)
    n=args.tokens
    values=[torch.randn(1,h,n,args.head_dim,device='cuda',dtype=torch.bfloat16)
            for h in (args.heads,args.kv_heads,args.kv_heads)]
    upstream=torch.randn_like(values[0])
    mask=torch.empty(n,n,device='cuda',dtype=torch.bool)
    keys=torch.arange(n,device='cuda')[None,:]
    for start in range(0,n,256):
        distance=torch.arange(start,min(start+256,n),device='cuda')[:,None]-keys
        mask[start:start+256]=(distance>=0)&((distance%127<12)|(distance<64))
    results=[]
    for deterministic in (False,True):
        torch.use_deterministic_algorithms(deterministic)
        runs=[];operators=None
        for repetition in range(2):
            q,k,v=[x.detach().clone().requires_grad_(True) for x in values]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                result=F.scaled_dot_product_attention(q,repeat_kv(k,args.heads//args.kv_heads),
                    repeat_kv(v,args.heads//args.kv_heads),attn_mask=mask[None,None],
                    dropout_p=0.,scale=args.head_dim**-.5)
                grads=torch.autograd.grad(result,(q,k,v),upstream)
            operators=sorted({x.key for x in prof.key_averages() if 'attention' in x.key})
            runs.append(dict(output=result.detach().cpu(),q=grads[0].cpu(),k=grads[1].cpu(),v=grads[2].cpu()))
            del q,k,v,result,grads
        comparison=compare(runs[1],runs[0])
        row=dict(deterministic=deterministic,operators=operators,comparison=comparison)
        results.append(row)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(dict(deterministic=deterministic,operators=operators,
            **{k:v for k,v in comparison.items() if k!='per_parameter'})),flush=True)
        del runs
    (out/'scope.json').write_text(json.dumps(dict(arguments=vars(args),
        scope='Synthetic native-SDPA operator diagnostic only; not adapter-gradient qualification'),indent=2)+'\n')


if __name__=='__main__':main()
