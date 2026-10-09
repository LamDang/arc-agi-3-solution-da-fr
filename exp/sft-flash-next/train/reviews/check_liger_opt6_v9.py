"""Historical overfit-only reviewer; superseded by check_opt6_exact_v9.py.

The Opt6 attempt was interrupted and rejected. This success-only checker does
not qualify Opt6 under the current bitwise equality requirement.

Requires NumPy; does not launch a GPU job, download artifacts or push anything.
"""
import collections,hashlib,io,json,math,pickle,struct,zipfile,tarfile
from pathlib import Path
import numpy as np
root=Path('exp/sft-flash-next/train');folder=root/'gradient-results/liger-opt6-overfit-v9'
class Reader(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  if module=='torch' and name in ('FloatStorage','LongStorage','IntStorage','BFloat16Storage','BoolStorage'):
   return {'FloatStorage':'<f4','LongStorage':'<i8','IntStorage':'<i4','BFloat16Storage':'<u2','BoolStorage':'?'}[name]
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return lambda storage,offset,size,stride,*rest:(storage,offset,size,stride)
  raise ValueError((module,name))
 def persistent_load(self,item):
  kind,dtype,key,device,numel=item;assert kind=='storage' and device=='cpu';return key,numel,dtype
class Tensors:
 def __init__(self,path):
  self.zip=zipfile.ZipFile(path);self.prefix=next(n[:-len('data.pkl')] for n in self.zip.namelist() if n.endswith('/data.pkl'))
  assert self.zip.read(self.prefix+'byteorder')==b'little'
  self.state=Reader(io.BytesIO(self.zip.read(self.prefix+'data.pkl'))).load()
 def tensor(self,meta):
  (key,stored,dtype),offset,size,stride=meta;raw=self.zip.read(self.prefix+'data/'+key);dt=np.dtype(dtype)
  return np.ndarray(shape=size,dtype=dt,buffer=raw,offset=offset*dt.itemsize,strides=tuple(x*dt.itemsize for x in stride))
 def values(self,meta):
  x=self.tensor(meta)
  return (x.astype(np.uint32)<<16).view(np.float32) if x.dtype==np.dtype('<u2') else x

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(name):return json.loads((folder/name).read_text())
config=read('resolved-config.json');result=read('result.json');audit=read('learning-audit.json');policy=read('bf16-policy.json');head=read('target-head.json');ple=read('ple-preparation.json');monitor=read('launch-monitor.json');launch=read('launch-launch.json');record=read('runner-result.json');timing=read('launch-timing-summary.json')
assert record['passed'] and record['structurally_valid'] and not record['gradient_equality_required']
assert config['learning']['target_loss_ratio']==.05 and config['learning']['max_updates']==20
assert monitor['returncode']==0 and not monitor['timed_out']
assert '--first-pass-only' not in launch['command'] and '--adapter-state' not in launch['command']
assert '--max-steps' in launch['command'] and '--learning-rate' in launch['command']
assert launch['no_optimizer_update']==False and launch['no_clipping']==False
updates=result['optimizer_updates'];assert 0<updates<=20 and audit['completed_updates']==updates
rows=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
losses=[x for x in rows if x['event']=='loss'];steps=[x for x in rows if x['event']=='update']
assert len(steps)==updates and len(losses)==updates+1 and [x['step'] for x in losses]==list(range(updates+1))
assert [x['step'] for x in steps]==list(range(1,updates+1))
assert all(math.isfinite(x['loss']) and x['loss']>=0 for x in losses)
assert losses[0]['loss']==result['baseline_loss'] and losses[-1]['loss']==result['measured_loss']
assert math.isclose(losses[-1]['loss']/losses[0]['loss'],result['ratio'],rel_tol=1e-14)
assert result['ratio']<=.05 and result['passed'] and result['best_loss']==min(x['loss'] for x in losses)
assert all(x['ratio']>.05 for x in losses[:-1])
assert len(timing['steps'])==updates+1
for i,row in enumerate(timing['steps']):
 assert row['step']==i and row['forward_seconds']>0
 assert row['phase_resource_samples']['forward']['exact_cuda_peak_allocated_bytes']>0
 assert row['phase_resource_samples']['forward']['peak_process_rss_bytes']>0
 if i<updates:
  assert row['pure_backward_seconds']>0 and row['optimizer_seconds']>0
  assert row['phase_resource_samples']['pure_backward']['exact_cuda_peak_allocated_bytes']>0
  assert row['phase_resource_samples']['pure_backward']['peak_process_rss_bytes']>0
 else:assert row['pure_backward_seconds'] is None
files=read('file-hashes.json')
for name,row in files.items():
 assert (folder/name).stat().st_size==row['bytes'] and sha(folder/name)==row['sha256'],name
for name,checksum in config['candidate_sources_sha256'].items():
 assert sha(folder/('frozen-candidate-'+name))==sha(folder/('launch-candidate-'+name))==checksum
assert sha(folder/'reference.py')==sha(folder/'frozen-reference.py')==config['expected_sha256']['reference_script']
assert sha(folder/'launch-sample.pt')==config['expected_sha256']['sample']
# Native CCE source pin and original statistics contract.
dep=config['cce_dependency'];archive=folder/('launch-'+Path(dep['remote_archive']).name)
assert sha(archive)==dep['sha256']
imports=read('launch-imported-source-hashes.json');verified_cce=0
with tarfile.open(archive) as tar:
 for filename,digest_value in imports.items():
  marker='/cce-runtime/'+dep['archive_prefix']+'/'
  if marker in filename:
   relative=filename.split(marker,1)[1]
   assert hashlib.sha256(tar.extractfile(dep['archive_prefix']+'/'+relative).read()).hexdigest()==digest_value
   verified_cce+=1
assert verified_cce==15
# The training extension keeps v7 activation ports and CPU-pack observer identical.
import ast
old=ast.parse((root/'gradient-results/cce-opt5-activations-v7/frozen-candidate-bf16_model_activations.py').read_text())
new=ast.parse((folder/'launch-candidate-bf16_model_activations.py').read_text())
for name in ('cast_boundary','cast_hidden_input','cast_hidden_output','activation_role','observe_saved_hooks'):
 left=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
 right=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
assert head['frozen_head'] and head['implementation']=='cce_exact' 
options=head['effective_options'];assert options['filter_e_grad']==False and options['filter_c_grad']==False
assert options['accum_e_fp32'] and options['accum_c_fp32']
assert not head['vocabulary_saved_tensor_calls'] and len(head['head_calls'])==updates+1
assert all(x['hidden_dtype']=='torch.bfloat16' and x['target_tokens']==651 for x in head['head_calls'])
assert policy['finalized'] and policy['training'] and policy['optimizer_updates']==updates
assert policy['adapter_parameter_dtype_counts']=={'torch.bfloat16':744}
assert not policy['saved_tensor_conversion'] and not policy['saved_fp32_statistics_compressed']
assert policy['saved_original_dtype_counts']==policy['saved_storage_dtype_counts'] and policy['saved_original_bytes']==policy['saved_storage_bytes']
assert len(policy['cce_fp32_lse'])==updates+1 and all(x['dtype']=='torch.float32' and x['shape']==[651] and x['roundtrip_values_exact'] for x in policy['cce_fp32_lse'])
assert all(x['dtype']=='torch.bfloat16' for x in policy['layer_calls'])
assert len({x['module'] for x in policy['layer_calls'] if x['event']=='exit'})==48
# Exact fresh random-A/zero-B initialization, not a trained diagnostic adapter.
source=Tensors(folder/'launch-bf16-policy-clean-fp32-source-adapter.pt');rounded=Tensors(folder/'launch-bf16-policy-rounded-adapter.pt');initial=Tensors(folder/'initial-adapter.pt');final=Tensors(folder/'final-adapter.pt');best=Tensors(folder/'best-adapter.pt')
assert set(source.state)==set(rounded.state)==set(initial.state)==set(final.state)==set(best.state) and len(initial.state)==744
digest=hashlib.sha256();changed=0
for name in sorted(initial.state):
 x=source.tensor(source.state[name]);y=rounded.tensor(rounded.state[name]);z=initial.tensor(initial.state[name]);last=final.tensor(final.state[name]);opt=best.tensor(best.state[name])
 assert x.dtype==np.dtype('<f4') and y.dtype==z.dtype==last.dtype==opt.dtype==np.dtype('<u2')
 bits=x.view(np.uint32);expected=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
 assert np.array_equal(y,expected) and y.tobytes()==z.tobytes()
 assert np.isfinite(final.values(final.state[name])).all()
 assert last.tobytes()==opt.tobytes()
 changed+=int(last.tobytes()!=z.tobytes())
 if 'lora_B' in name:assert not np.count_nonzero(z) and np.count_nonzero(last)>0
 else:assert np.count_nonzero(z)>0
 digest.update(name.encode());digest.update(str(tuple(y.shape)).encode());digest.update(b'torch.bfloat16');digest.update(y.tobytes())
assert digest.hexdigest()==policy['adapter_rounding']['bf16_state_digest']
assert changed==audit['parameter_tensors_changed'] and changed>0
assert audit['initialization']=='seeded-random-A-zero-B' and audit['all_parameters_finite'] and audit['all_gradients_finite'] and audit['optimizer_states_finite']
# Every raw pre-clipping gradient at each actual optimizer update.
assert len(audit['gradient_exports'])==updates
for step,row in enumerate(audit['gradient_exports']):
 assert row['step']==step and row['tensors']==744 and row['all_finite'] and row['raw_before_clipping']
 archive=Tensors(folder/row['file']);summary=read(f'gradient-summary-step-{step:03d}.json')
 assert sha(folder/row['file'])==row['sha256'] and set(archive.state)==set(summary) and len(summary)==744
 counts={'A':0,'B':0}
 for name,meta in archive.state.items():
  raw=archive.tensor(meta);value=archive.values(meta);item=summary[name]
  assert raw.dtype==np.dtype('<u2') and item['dtype']=='torch.bfloat16'
  assert value.shape==tuple(item['shape']) and np.isfinite(value).all() and item['finite']
  assert np.count_nonzero(value)==item['nonzero'] and float(np.max(np.abs(value)))==item['max_absolute']
  assert math.isclose(float(np.linalg.norm(value.astype(np.float64))),item['norm'],rel_tol=1e-10,abs_tol=1e-15)
  counts['A' if 'lora_A' in name else 'B']+=int(np.count_nonzero(value)>0)
 assert counts['A']==row['a_nonzero'] and counts['B']==row['b_nonzero']
 if step==0:assert counts=={'A':0,'B':372}
# Actual AdamW state and completed step counters.
optimizer=Tensors(folder/'optimizer-state.pt');optstate=optimizer.state['state']
assert len(optstate)==744 and len(optimizer.state['param_groups'])==1
pg=optimizer.state['param_groups'][0];assert pg['lr']==.0002 and pg['weight_decay']==0
for state in optstate.values():
 assert set(state)=={'step','exp_avg','exp_avg_sq'}
 assert float(optimizer.values(state['step']))==updates
 for key in ('exp_avg','exp_avg_sq'):
  assert optimizer.tensor(state[key]).dtype==np.dtype('<u2') and np.isfinite(optimizer.values(state[key])).all()
assert len(audit['optimizer_records'])==updates and all(x['finite'] for x in audit['optimizer_records'])
# Prepared PLE reused through every pass, without disk reads during GPU training.
assert ple['optimizer_updates']==updates and ple['current_retained_through_backward'] and ple['workers_shutdown']
assert len(ple['calls'])==2*updates+1 and len({x['prepared_cpu_storage_pointer'] for x in ple['calls']})==1
assert all(x['disk_reads']==0 for x in ple['calls'])
private_file=Path('/tmp/kaggle_probe_url')
private=private_file.read_text().strip() if private_file.exists() else None
for path in folder.iterdir():
 if path.suffix in ('.json','.jsonl','.py','.mjs','.log','.txt'):
  content=path.read_text(errors='replace')
  assert 'https://kkb-production.jupyter-proxy.kaggle.net/k/' not in content
  if private:assert private not in content
# Independently verify the Opt6 kernels and exact scope of generated expert change.
assert config['opt6']=='liger-rmsnorm-swiglu' and record['opt6_valid']
opt6=read('opt6-kernels.json');ops=read('launch-opt6-operator-check.json')
assert opt6['finalized'] and ops['passed'] and len(ops['cases'])==11
assert opt6['gated_activation_counts']=={'sigmoid':36} and ops['configured_gate']=='sigmoid'
assert any(x['label']=='swiglu640' for x in ops['cases'])
assert opt6['module_counts']=={'Qwen4ExpTextRMSNorm':148,'Qwen4ExpTextRMSNormGated':36,'Qwen4ExpTextMLP':48,'Qwen4ExpTextExperts':48}
assert opt6['rms_offset']==1 and opt6['rms_casting_mode']=='gemma' and not opt6['rms_in_place_backward']
assert opt6['routing_and_projections_unchanged'] and not opt6['statistics_blanket_cast']
assert all(opt6['calls'][name]>0 for name in ('rms','grouped_rms','gated_rms','shared_mlp','expert_swiglu'))
assert opt6['calls']['swiglu']==opt6['calls']['shared_mlp']+opt6['calls']['expert_swiglu']
liger_archive=folder/('launch-'+Path(config['liger_dependency']['remote_wheel']).name)
assert sha(liger_archive)==config['liger_dependency']['sha256']
verified_liger=0
with zipfile.ZipFile(liger_archive) as wheel:
 for name in ('rms_norm','swiglu'):
  expected=hashlib.sha256(wheel.read('liger_kernel/ops/'+name+'.py')).hexdigest()
  assert sha(folder/('launch-opt6-liger-'+name+'.py'))==opt6['kernel_sources'][name]==expected
 for filename,value in imports.items():
  if '/liger-runtime/liger_kernel/' in filename:
   relative='liger_kernel/'+filename.split('/liger-runtime/liger_kernel/',1)[1]
   assert hashlib.sha256(wheel.read(relative)).hexdigest()==value
   verified_liger+=1
assert verified_liger>0
native_moe=folder/'launch-opt6-native-moe.py';native_model=folder/'launch-opt6-native-model.py'
assert sha(native_moe)==opt6['native_sources']['moe']=='2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c'
assert sha(native_model)==opt6['native_sources']['model']=='0154ba57593c79330a97aefa2a909390f5f1f949aaacc6cbc8586174cd301206'
original=next(n for n in ast.parse(native_moe.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='linear_loop_experts_forward')
generated=folder/'launch-opt6-generated-experts.py';assert sha(generated)==opt6['generated_experts_sha256']
generated_ast=ast.parse(generated.read_text()).body[0]
wanted=ast.parse('gated_out = _opt6_swiglu(gate_out, up_out)').body[0]
original_if=next(n for n in ast.walk(original) if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Name) and n.test.func.id=='hasattr' and len(n.test.args)==2 and isinstance(n.test.args[1],ast.Constant) and n.test.args[1].value=='_apply_gate')
reversed_nodes=[]
class Reverse(ast.NodeTransformer):
 def visit_Assign(self,node):
  if ast.dump(node,include_attributes=False)==ast.dump(wanted,include_attributes=False):
   reversed_nodes.append(node);return original_if
  return self.generic_visit(node)
restored=Reverse().visit(generated_ast)
assert len(reversed_nodes)==1 and ast.dump(original,include_attributes=False)==ast.dump(restored,include_attributes=False)
assert all(case['passed'] for case in ops['cases'])
for case in ops['cases']:
 assert case['output_relative_l2']<=case['tolerance_relative_l2']
 assert max(case['gradient_relative_l2'])<=case['tolerance_relative_l2']
# Same clean initialization as v8 isolates this optimization from seed changes.
v8=root/'gradient-results/bf16-overfit-v8'
assert policy['adapter_rounding']['bf16_state_digest']==json.loads((v8/'bf16-policy.json').read_text())['adapter_rounding']['bf16_state_digest']
report=dict(run_id=config['run_id'],attempt_id=config['attempt_id'],passed=True,
 acceptance='one-sample learning; exact gradient equality not required',updates=updates,
 baseline_loss=result['baseline_loss'],final_loss=result['measured_loss'],final_ratio=result['ratio'],
 loss_reduction_percent=100*(1-result['ratio']),loss_curve=record['loss_curve'],
 fresh_random_a_zero_b_independently_verified=True,bf16_initialization_digest=digest.hexdigest(),
 bf16_rounding_and_native_initial_export_exact=True,trained_parameter_tensors_changed=changed,
 raw_gradient_archives_verified=updates,raw_tensors_per_archive=744,all_raw_gradients_finite=True,
 first_step_a_gradients_zero_b_gradients_nonzero=True,gradients_saved_before_clipping=True,
 bf16_parameters_and_optimizer_moments_verified=True,optimizer_completed_updates_verified=True,
 all_48_decoder_outputs_bf16=True,cce_lse_original_fp32_roundtrip_exact=True,
 saved_original_dtype_and_bytes_preserved=True,all_file_hashes_verified=True,
 per_step_gpu_ram_timing_verified=True,ple_payload_reused_all_steps_zero_gpu_disk_reads=True,
 source_hashes_verified=True,imported_cce_sources_verified=verified_cce,
 v7_activation_arithmetic_ast_unchanged=True,private_url_absent=True,pushed_to_remote=False,
 diagnostic_only=True,capacity_130k_measured=False,opt6_kernels_verified=True,
 imported_liger_sources_verified=verified_liger,expert_routing_ast_unchanged_outside_gate=True,
 exact_same_initialization_as_v8=True,opt6_call_counts=opt6['calls'])
(root/'reviews/liger-opt6-overfit-v9.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
