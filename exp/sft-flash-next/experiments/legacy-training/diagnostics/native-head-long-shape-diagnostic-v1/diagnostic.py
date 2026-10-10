"""Synthetic common-dLogits LM-head VJP at capacity shapes; no full-model claim."""
import sys
sys.path=[p for p in sys.path if p.rstrip('/')!='/usr/local/lib/python3.13/dist-packages']
sys.path[:0]=['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao','/kaggle/working/training-gradient-audit/exp/sft-flash-next/train']
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch,transformers
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=Path(a.source)
native=json.loads((source/'report.json').read_text());assert 'finished' in native
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8';torch.use_deterministic_algorithms(True);torch.set_num_threads(8)
settings={'deterministic':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'transformers':transformers.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'autocast':'bfloat16'}
assert settings==native['settings']
index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())
with safe_open(str(Path(a.model)/index['weight_map']['lm_head.weight']),framework='pt',device='cpu') as f:weight_cpu=f.get_tensor('lm_head.weight')
head_digest=hashlib.sha256(memoryview(weight_cpu.contiguous().view(torch.uint8).reshape(-1).numpy())).hexdigest();assert head_digest==native['weight']['tensor_sha256'];weight=weight_cpu.cuda();del weight_cpu
small=torch.load(source/'full_target_dlogits.pt',map_location='cpu',weights_only=True)[0];original_q,v=small.shape;q=10000
dl=small.repeat((q+original_q-1)//original_q,1)[:q].contiguous();dl.mul_(original_q/q)
dl_digest=hashlib.sha256(memoryview(dl.view(torch.uint8).reshape(-1).numpy())).hexdigest();del small
report={'operator_only':True,'synthetic_target_dlogits':True,'generation':{'source':'full_target_dlogits.pt','source_tensor_sha256':native['native']['target_dlogits']['tensor_sha256'],'repeat_rows_then_slice':q,'bf16_scale':original_q/q,'generated_tensor_sha256':dl_digest},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),'settings':settings,'weight_tensor_sha256':head_digest,'cases':[]}

def emit(row):report['cases'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
def vjp(rows,offset):
 assert offset+q<=rows
 started=time.monotonic();torch.cuda.reset_peak_memory_stats();values=torch.zeros((rows,v),dtype=torch.bfloat16,device='cuda');values[offset:offset+q]=dl.cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):result=torch.mm(values,weight)[offset:offset+q].cpu()
 peak=torch.cuda.max_memory_allocated()/2**30;del values;gc.collect();torch.cuda.empty_cache();return result,peak,time.monotonic()-started
references={}
for rows,prompt in [(90000,80000),(130000,120000)]:
 result,peak,seconds=vjp(rows,prompt-1);references[rows]=result;torch.save(result,out/f'native-target-dhidden-{rows}.pt');emit({'kind':'native_shape','rows':rows,'offset':prompt-1,'targets':q,'H':weight.shape[1],'V':v,'peak_gib':peak,'seconds':seconds,'tensor_sha256':hashlib.sha256(memoryview(result.view(torch.uint8).reshape(-1).numpy())).hexdigest()})
for rows in [10000,10240,12288,16384,32768,65536]:
 result,peak,seconds=vjp(rows,0);torch.save(result,out/f'padded-target-dhidden-{rows}.pt');comparisons={}
 for n,ref in references.items():
  c=result.float();r=ref.float();d=c-r
  comparisons[str(n)]={'bitwise_equal':bool(torch.equal(result,ref)),'mismatched_values':int((result!=ref).sum()),'failed_values':int((d.abs()>1e-8+1e-5*r.abs()).sum()),'relative_l2':(d.double().square().sum()/r.double().square().sum()).sqrt().item(),'max_abs':d.abs().max().item()}
 emit({'kind':'candidate_shape','rows':rows,'offset':0,'peak_gib':peak,'seconds':seconds,'vs_native':comparisons})
 del result,c,r,d
report['finished']=True;report['full_context_model_execution_proven']=False;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
