"""Compare mixed-dtype native and candidate LM-head loss/input gradients on GPU.

Run this script in the exact Python environment used for the native reference;
a site-specific wrapper may supply dependency paths. No operator is replaced in
the direct native reference: it calls F.linear and ForCausalLMLoss under BF16
CUDA autocast and stock save_on_cpu, separately for every hidden input and loss
scale. This synthetic operator diagnostic does not qualify a complete model.

The source directory contains report.json and hidden.pt from the saved native
head diagnostic. Its environment and actual safetensors head digest must match.
FP32 random inputs contain values beyond BF16; a second input uses the saved
BF16 hidden values cast to FP32. Both test upstream scales 1 and 0.125, with
native-sized and explicit 16384-row candidate backward matrices.

Example (use the native-reference environment wrapper where required):
    python diagnose_native_head_mixed_dtype.py --model /path/to/model \
        --sample /path/to/sample.pt --source /path/to/native-head-diagnostic \
        --out /path/to/new-mixed-dtype-diagnostic
"""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch,transformers
from safetensors import safe_open
from transformers.loss.loss_utils import ForCausalLMLoss
from native_head_loss import selected_native_backward_loss
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--sample',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=Path(a.source)
prior=json.loads((source/'report.json').read_text());torch.manual_seed(20261009);torch.use_deterministic_algorithms(True);torch.set_num_threads(8);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
settings={'deterministic':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'transformers':transformers.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'autocast':'bfloat16'};assert settings==prior['settings']
index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())
with safe_open(str(Path(a.model)/index['weight_map']['lm_head.weight']),framework='pt',device='cpu') as f:weight_cpu=f.get_tensor('lm_head.weight')
def digest(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).reshape(-1).numpy())).hexdigest()
head_digest=digest(weight_cpu);assert head_digest==prior['weight']['tensor_sha256'];assert weight_cpu.dtype==torch.bfloat16;weight=weight_cpu.cuda();del weight_cpu
batch=torch.load(a.sample,map_location='cpu',weights_only=True);labels=batch['input_ids'].cuda();prompt=prior['dimensions']['prompt'];labels[:,:prompt]=-100;t=labels.shape[1];h=weight.shape[1]
rounded=torch.load(source/'hidden.pt',map_location='cpu',weights_only=True).float();raw=torch.randn((1,t,h),dtype=torch.float32);assert torch.count_nonzero(raw!=raw.to(torch.bfloat16).float())
train=Path(__file__).resolve().parent
report={'operator_only':True,'synthetic_hidden':True,'reference':'Fresh direct native full-head F.linear and ForCausalLMLoss for each input/scale, with stock save_on_cpu','sample_sha256':hashlib.sha256(Path(a.sample).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'settings':settings,'head_tensor_sha256':head_digest,'dimensions':{'T':t,'H':h,'V':weight.shape[0],'prompt':prompt,'targets':t-prompt},'source_sha256':{n:hashlib.sha256((train/n).read_bytes()).hexdigest() for n in ['native_head_loss.py','native_optimization_flags.py','native_gradient_compare.py','native_gradient_audit.py']},'inputs':{},'cases':[]}
def emit(row):report['cases'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
def metric(candidate,reference):
 assert candidate.dtype==reference.dtype==torch.float32 and candidate.shape==reference.shape
 mismatched=failed=0;error=ref2=maxabs=0.
 for c,r in zip(candidate.reshape(-1).split(1<<20),reference.reshape(-1).split(1<<20)):
  dd=c.double()-r.double();error+=dd.square().sum().item();ref2+=r.double().square().sum().item();maxabs=max(maxabs,dd.abs().max().item());mismatched+=(c!=r).sum().item();failed+=(dd.abs()>1e-8+1e-5*r.double().abs()).sum().item()
 return {'gradient_bitwise_equal':bool(torch.equal(candidate,reference)),'mismatched_values':mismatched,'failed_values':failed,'relative_l2':(error/ref2)**.5,'max_abs':maxabs,'gradient_dtype':str(candidate.dtype)}
for mode,original in [('bf16_rounded_to_fp32',rounded),('raw_fp32',raw)]:
 report['inputs'][mode]={'dtype':str(original.dtype),'tensor_sha256':digest(original),'values_differ_from_bf16_rounding':int(torch.count_nonzero(original!=original.to(torch.bfloat16).float()))};torch.save(original,out/(mode+'-hidden.pt'))
 for scale in [1.,.125]:
  native=original.cuda().requires_grad_();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
  with torch.autograd.graph.save_on_cpu(pin_memory=False):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=torch.nn.functional.linear(native,weight);loss=ForCausalLMLoss(logits,labels,vocab_size=weight.shape[0])
   del logits;(loss*scale).backward()
  expected_loss=loss.item();expected_gradient=native.grad.cpu();torch.save(expected_gradient,out/f'{mode}-{scale}-native-dhidden.pt')
  emit({'kind':'direct_native','input':mode,'scale':scale,'loss':expected_loss,'hidden_dtype':str(native.dtype),'head_dtype':str(weight.dtype),'gradient_dtype':str(expected_gradient.dtype),'gradient_tensor_sha256':digest(expected_gradient),'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started})
  del native,loss;gc.collect();torch.cuda.empty_cache()
  for rows in [None,16384]:
   candidate=original.cuda().requires_grad_();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
   with torch.autograd.graph.save_on_cpu(pin_memory=False):
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=selected_native_backward_loss(candidate,weight,labels,prompt,rows)
    (loss*scale).backward()
   measured=candidate.grad.cpu();item={'kind':'candidate','input':mode,'scale':scale,'backward_rows':rows,'loss':loss.item(),'loss_bitwise_equal':loss.item()==expected_loss,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started,**metric(measured,expected_gradient)}
   torch.save(measured,out/f'{mode}-{scale}-{rows}-dhidden.pt');emit(item)
   del candidate,loss,measured;gc.collect();torch.cuda.empty_cache()
  del expected_gradient
report['finished']=True;report['passed']=all(row['loss_bitwise_equal'] and row['gradient_bitwise_equal'] for row in report['cases'] if row['kind']=='candidate');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed'],'Mixed-dtype direct-native gate failed'
