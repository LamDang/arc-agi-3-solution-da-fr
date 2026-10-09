"""Custom head candidate against saved synthetic native head gradients."""
import sys
sys.path=[p for p in sys.path if p.rstrip('/')!='/usr/local/lib/python3.13/dist-packages'];sys.path[:0]=['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao','/kaggle/working/training-gradient-audit/exp/sft-flash-next/train']
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch,transformers
from safetensors import safe_open
from native_head_loss import selected_native_backward_loss
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--sample',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=Path(a.source)
native=json.loads((source/'report.json').read_text());torch.use_deterministic_algorithms(True);torch.set_num_threads(8);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
settings={'deterministic':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'transformers':transformers.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'autocast':'bfloat16'};assert settings==native['settings']
index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())
with safe_open(str(Path(a.model)/index['weight_map']['lm_head.weight']),framework='pt',device='cpu') as f:weight_cpu=f.get_tensor('lm_head.weight')
head_digest=hashlib.sha256(memoryview(weight_cpu.view(torch.uint8).reshape(-1).numpy())).hexdigest();assert head_digest==native['weight']['tensor_sha256'];weight=weight_cpu.cuda();del weight_cpu
hidden_cpu=torch.load(source/'hidden.pt',map_location='cpu',weights_only=True);reference=torch.load(source/'full_dhidden.pt',map_location='cpu',weights_only=True)
labels=torch.load(a.sample,map_location='cpu',weights_only=True)['input_ids'].cuda();prompt=native['dimensions']['prompt'];labels[:,:prompt]=-100
report={'operator_only':True,'synthetic_hidden':True,'settings':settings,'head_tensor_sha256':head_digest,'source_sha256':{n:hashlib.sha256((Path('/kaggle/working/training-gradient-audit/exp/sft-flash-next/train')/n).read_bytes()).hexdigest() for n in ['native_head_loss.py','native_optimization_flags.py','native_gradient_compare.py','native_gradient_audit.py']},'cases':[]}
for rows in [None,16384]:
 hidden=hidden_cpu.cuda().requires_grad_();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 with torch.autograd.graph.save_on_cpu(pin_memory=False):
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=selected_native_backward_loss(hidden,weight,labels,prompt,rows)
  loss.backward()
 result=hidden.grad.cpu();d=result.float()-reference.float();rr=reference.float();row={'backward_rows':rows,'loss':loss.item(),'loss_bitwise_equal':loss.item()==native['native']['loss'],'gradient_bitwise_equal':bool(torch.equal(result,reference)),'mismatched_values':int((result!=reference).sum()),'failed_values':int((d.abs()>1e-8+1e-5*rr.abs()).sum()),'relative_l2':(d.double().square().sum()/rr.double().square().sum()).sqrt().item(),'max_abs':d.abs().max().item(),'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-start}
 torch.save(result,out/f'dhidden-{rows}.pt');report['cases'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
 del hidden,loss,result,d,rr;gc.collect();torch.cuda.empty_cache()
report['passed']=all(x['loss_bitwise_equal'] and x['gradient_bitwise_equal'] for x in report['cases']);report['finished']=True;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed'],'Custom head candidate failed'
