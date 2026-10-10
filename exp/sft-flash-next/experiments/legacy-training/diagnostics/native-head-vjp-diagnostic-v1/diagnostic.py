"""Synthetic-hidden native LM-head diagnostic; not whole-model qualification."""
import argparse,gc,hashlib,inspect,json,os,time
from pathlib import Path
import sys
sys.path=[value for value in sys.path if value.rstrip('/')!='/usr/local/lib/python3.13/dist-packages']
sys.path[:0]=['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao','/kaggle/working/training-gradient-audit/exp/sft-flash-next/train']
import torch
import transformers
from safetensors import safe_open
from transformers.loss.loss_utils import ForCausalLMLoss

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--sample',required=True);p.add_argument('--prompt',type=int,default=15598);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();torch.manual_seed(20261009);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'

def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def metric(candidate,reference):
 assert candidate.shape==reference.shape and candidate.dtype==reference.dtype
 # CPU chunking avoids a large FP64 scratch buffer and compares every value.
 c=candidate.reshape(-1);r=reference.reshape(-1);error=ref2=0.;failed=mismatched=0;maxabs=0.
 for cc,rr in zip(c.split(1<<20),r.split(1<<20)):
  cd=cc.double();rd=rr.double();diff=cd-rd
  error+=diff.square().sum().item();ref2+=rd.square().sum().item()
  maxabs=max(maxabs,diff.abs().max().item());mismatched+=(cc!=rr).sum().item()
  failed+=(diff.abs()>1e-8+1e-5*rd.abs()).sum().item()
 return dict(shape=list(candidate.shape),dtype=str(candidate.dtype),bitwise_equal=bool(torch.equal(candidate,reference)),mismatched_values=mismatched,failed_values=failed,max_abs=maxabs,relative_l2=(error/ref2)**.5 if ref2 else (0. if error==0 else None),finite=bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all()))

def emit(name,result):
 report[name]=result;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps(result),flush=True)

def save(name,t):
 torch.save(t,out/(name+'.pt'));return {'file':name+'.pt','shape':list(t.shape),'dtype':str(t.dtype),'tensor_sha256':digest(t)}

index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text());key='lm_head.weight';shard=Path(a.model)/index['weight_map'][key]
with safe_open(str(shard),framework='pt',device='cpu') as f:weight_cpu=f.get_tensor(key)
assert weight_cpu.dtype==torch.bfloat16
v,h=weight_cpu.shape
batch=torch.load(a.sample,map_location='cpu',weights_only=True);ids=batch['input_ids'];t=ids.shape[1];q=t-a.prompt
labels=ids.clone();labels[:,:a.prompt]=-100
positions=torch.arange(a.prompt-1,t-1)
shift_labels=torch.nn.functional.pad(labels,(0,1),value=-100)[:,1:].contiguous()
assert (shift_labels!=-100).sum().item()==q
hidden_cpu=torch.randn((1,t,h),dtype=torch.float32).to(torch.bfloat16)
report={'sample_sha256':hashlib.sha256(Path(a.sample).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python_path':sys.path,'diagnostic_only':True,'synthetic_hidden':True,'arguments':vars(a),'settings':{'deterministic':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'transformers':transformers.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'autocast':'bfloat16'},'dimensions':{'T':t,'prompt':a.prompt,'targets':q,'V':v,'H':h},'weight':{'shard':shard.name,'key':key,'tensor_sha256':digest(weight_cpu)},'hidden':save('hidden',hidden_cpu),'labels':save('shift_labels',shift_labels),'native_loss_source_sha256':hashlib.sha256(inspect.getsource(ForCausalLMLoss).encode()).hexdigest()}
weight=weight_cpu.cuda();del weight_cpu
labels_cuda=labels.cuda();shift_cuda=shift_labels.cuda();targets=labels_cuda[:,a.prompt:];positions_cuda=positions.cuda()
torch.cuda.reset_peak_memory_stats()
# Separate graphs reproduce the unchanged native causal CE and selected-logit CE.
hidden=hidden_cpu.cuda().requires_grad_()
with torch.autocast('cuda',dtype=torch.bfloat16):
 logits=torch.nn.functional.linear(hidden,weight)
 loss=ForCausalLMLoss(logits,labels_cuda,vocab_size=v)
grad_logits,grad_hidden=torch.autograd.grad(loss,(logits,hidden))
full_logits=logits[:,positions_cuda].detach().cpu();full_dl=grad_logits[:,positions_cuda].detach().cpu();full_dh=grad_hidden.detach().cpu()
ignored=torch.ones(t,dtype=torch.bool,device='cuda');ignored[positions_cuda]=False
emit('native',{'loss':loss.item(),'ignored_dlogits_exact_zero':not bool(torch.count_nonzero(grad_logits[:,ignored])),'ignored_dhidden_exact_zero':not bool(torch.count_nonzero(grad_hidden[:,ignored])),'target_logits':save('full_target_logits',full_logits),'target_dlogits':save('full_target_dlogits',full_dl),'dhidden':save('full_dhidden',full_dh),'peak_gib':torch.cuda.max_memory_allocated()/2**30})
del hidden,logits,loss,grad_logits,grad_hidden;gc.collect();torch.cuda.empty_cache()
hidden=hidden_cpu.cuda().requires_grad_()
with torch.autocast('cuda',dtype=torch.bfloat16):
 logits=torch.nn.functional.linear(hidden[:,positions_cuda],weight)
 loss=ForCausalLMLoss(logits,labels_cuda,vocab_size=v,shift_labels=targets)
grad_logits,grad_hidden=torch.autograd.grad(loss,(logits,hidden))
selected_logits=logits.detach().cpu();selected_dl=grad_logits.detach().cpu();selected_dh=grad_hidden.detach().cpu()
emit('selected',{'loss':loss.item(),'target_logits':save('selected_target_logits',selected_logits),'target_dlogits':save('selected_target_dlogits',selected_dl),'dhidden':save('selected_dhidden',selected_dh),'logits_vs_native':metric(selected_logits,full_logits),'dlogits_vs_native':metric(selected_dl,full_dl),'dhidden_vs_native':metric(selected_dh,full_dh)})
del hidden,logits,loss,grad_logits,grad_hidden;gc.collect();torch.cuda.empty_cache()
# Isolate CE shape using identical target logits, independent of forward GEMM.
common_logits=torch.zeros((1,t,v),dtype=torch.bfloat16,device='cuda');common_logits[:,positions_cuda]=full_logits.cuda();common_logits.requires_grad_()
loss=ForCausalLMLoss(common_logits,labels_cuda,vocab_size=v)
common_full_dl=torch.autograd.grad(loss,common_logits)[0][:,positions_cuda].cpu()
del common_logits,loss;gc.collect();torch.cuda.empty_cache()
common_logits=full_logits.cuda().requires_grad_();loss=ForCausalLMLoss(common_logits,labels_cuda,vocab_size=v,shift_labels=targets)
common_selected_dl=torch.autograd.grad(loss,common_logits)[0].cpu()
emit('ce_common_forward_logits',{'dlogits_vs_native_full':metric(common_selected_dl,common_full_dl),'full_vs_captured':metric(common_full_dl,full_dl)})
del common_logits,loss;gc.collect();torch.cuda.empty_cache()
# Isolate backward shape using one fixed dLogits and native frozen BF16 weight.
# mm is the unchanged native linear input-VJP arithmetic; padding preserves all T rows.
def head_vjp(dl,padded):
 if padded:
  values=torch.zeros((t,v),dtype=torch.bfloat16,device='cuda');values[positions_cuda]=dl[0].cuda()
 else:values=dl[0].cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):dh=torch.mm(values,weight)
 result=dh.cpu();del values,dh;gc.collect();torch.cuda.empty_cache();return result
fixed_full=head_vjp(full_dl,True);fixed_selected=head_vjp(full_dl,False)
emit('backward_common_dlogits',{'native_mm_vs_autograd':metric(fixed_full,full_dh[0]),'selected_vs_full_rows':metric(fixed_selected,fixed_full[positions]),'full_dhidden':save('fixed_dlogits_full_dhidden',fixed_full),'selected_dhidden':save('fixed_dlogits_selected_dhidden',fixed_selected)})
selected_padded=head_vjp(selected_dl,True)
emit('selected_padded_full_backward',{'dhidden_vs_native':metric(selected_padded,full_dh[0]),'dhidden':save('selected_padded_full_dhidden',selected_padded)})
emit('finished',{'elapsed_seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'whole_model_gradient_parity_proven':False})
