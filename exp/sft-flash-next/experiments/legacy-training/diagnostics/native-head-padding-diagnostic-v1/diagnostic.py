"""Probe LM-head BF16 VJP padding; operator diagnostic, not training qualification."""
import sys
sys.path=[p for p in sys.path if p.rstrip('/')!='/usr/local/lib/python3.13/dist-packages']
sys.path[:0]=['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao','/kaggle/working/training-gradient-audit/exp/sft-flash-next/train']
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
import transformers
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=Path(a.source);native=json.loads((source/'report.json').read_text());assert 'finished' in native
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8';torch.use_deterministic_algorithms(True);torch.set_num_threads(8)
index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())
with safe_open(str(Path(a.model)/index['weight_map']['lm_head.weight']),framework='pt',device='cpu') as f:weight_cpu=f.get_tensor('lm_head.weight')
head_digest=hashlib.sha256(weight_cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();assert head_digest==native['weight']['tensor_sha256']
weight=weight_cpu.cuda();del weight_cpu
settings={'deterministic':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'transformers':transformers.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'autocast':'bfloat16'}
assert settings==native['settings'], 'Diagnostic math settings changed'
dl=torch.load(source/'full_target_dlogits.pt',map_location='cpu',weights_only=True)[0];q,v=dl.shape
reference=torch.load(source/'full_dhidden.pt',map_location='cpu',weights_only=True)[0];original_start=native['dimensions']['prompt']-1;reference=reference[original_start:original_start+q].float()
report={'operator_only':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),'settings':settings,'weight_tensor_sha256':head_digest,'shapes':native['dimensions'],'cases':[]}
for rows in [651,1024,2048,4096,8192,16249]:
 offsets=[0]
 aligned=original_start%128
 while aligned+128+q<=rows:aligned+=128
 if aligned+q<=rows and aligned not in offsets:offsets.append(aligned)
 for offset in offsets:
  started=time.monotonic();torch.cuda.reset_peak_memory_stats();values=torch.zeros((rows,v),dtype=torch.bfloat16,device='cuda');values[offset:offset+q]=dl.cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):result=torch.mm(values,weight)[offset:offset+q].cpu()
  rr=result.float();diff=rr-reference
  item={'rows':rows,'offset':offset,'same_start_mod128':offset%128==original_start%128,'bitwise_equal':bool(torch.equal(rr,reference)),'mismatched_values':int((rr!=reference).sum()),'failed_values':int((diff.abs()>1e-8+1e-5*reference.abs()).sum()),'relative_l2':(diff.double().square().sum()/reference.double().square().sum()).sqrt().item(),'max_abs':diff.abs().max().item(),'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started}
  torch.save(result,out/f'dhidden-{rows}-{offset}.pt');report['cases'].append(item);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
  del values,result,rr,diff;gc.collect();torch.cuda.empty_cache()
report['finished']=True;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
