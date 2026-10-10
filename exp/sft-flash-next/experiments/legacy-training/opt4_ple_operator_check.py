"""Opt4 native hashing, complete anchor lookup and native PLE derivative checks."""
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from ple_preparation import PREFIX, hash_computer, DiskPLERows, tensor_sha256


def reference_lookup(model_dir,manifest,ids):
    """Independent safetensors slicing path, every distinct anchor row verified."""
    flat=ids.numpy().reshape(-1);unique,inverse=np.unique(flat,return_inverse=True)
    offsets=np.cumsum([0]+[s['rows'] for s in manifest['shards']]);shards=np.searchsorted(offsets,unique,side='right')-1
    values=torch.empty((len(unique),160),dtype=torch.bfloat16)
    for shard in np.unique(shards):
        info=manifest['shards'][int(shard)];positions=np.flatnonzero(shards==shard)
        with safe_open(str(Path(model_dir)/info['filename']),framework='pt',device='cpu') as archive:
            sliced=archive.get_slice(info['key'])
            for pos in positions:
                row=int(unique[pos]-offsets[shard]);values[int(pos)]=sliced[row:row+1][0]
    return values[torch.from_numpy(inverse)].reshape(*ids.shape,160).flatten(-2)


def qualify(model_dir,sample_path,out):
    out=Path(out);started=time.monotonic()
    batch=torch.load(sample_path,map_location='cpu',weights_only=True);tokens=batch['input_ids']
    cpu=hash_computer(model_dir);gpu=hash_computer(model_dir,'cuda')
    rows=DiskPLERows(model_dir);ids=cpu(tokens,None);gpu_ids=gpu(tokens.cuda(),None).cpu()
    assert torch.equal(ids,gpu_ids)
    payload,stats=rows.lookup(ids);reference=reference_lookup(model_dir,rows.manifest,ids)
    assert tensor_sha256(payload)==tensor_sha256(reference)
    torch.save(reference,out/'ple-reference-anchor-embeddings.pt')
    torch.save(ids,out/'ple-reference-anchor-row-ids.pt')
    # Deliberately include unfamiliar legal tokens, EOS boundaries, repeats,
    # singleton context and sequence beginnings; do not prune the hash space.
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(model_dir,local_files_only=True).text_config
    edge=torch.tensor([[0,1,cfg.vocab_size-1,cpu.eos_token_id,17,17,0,cpu.eos_token_id,42,42]])
    edge_ids=cpu(edge,None);assert torch.equal(edge_ids,gpu(edge.cuda(),None).cpu())
    edge_payload,_=rows.lookup(edge_ids);assert tensor_sha256(edge_payload)==tensor_sha256(reference_lookup(model_dir,rows.manifest,edge_ids))
    report=dict(passed=False,anchor_tokens=tokens.shape[1],anchor_hash_cpu_cuda_exact=True,
        full_anchor_lookup_bytes_exact=True,full_table_bytes=rows.manifest['table_bytes'],
        lookup=stats,anchor_payload_sha256=tensor_sha256(payload),anchor_row_ids_sha256=tensor_sha256(ids),
        edge_hash_cpu_cuda_exact=True,edge_lookup_bytes_exact=True,optimizer_updates=0)
    # Use native PLE layer/projection/gate/convolution with real checkpoint weights.
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
    source=inspect.getsource(Qwen4ExpTextPLELayer.forward)
    assert hashlib.sha256(source.encode()).hexdigest()=='6371822e8ff6187649e2b0ffe2bad4991ea3ee5f1b55e6dd97af8e52b10cee91'
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261013)
        with torch.device('meta'):ple=Qwen4ExpTextPLELayer(cfg,layer_idx=1,ple_layer_index=0)
        ple.ple_embedding=nn.Identity();ple.to_empty(device='cuda')
        index=json.loads((Path(model_dir)/'model.safetensors.index.json').read_text())['weight_map']
        for name,value in list(ple.named_parameters()):
            key='model.language_model.layers.1.ple.'+name
            with safe_open(str(Path(model_dir)/index[key]),framework='pt',device='cpu') as archive:
                weight=archive.get_tensor(key).cuda()
            parent,_,attr=name.rpartition('.');target=ple.get_submodule(parent) if parent else ple
            target._parameters[attr]=nn.Parameter(weight,requires_grad=False)
        original=torch.randn((1,edge.shape[1],cfg.hidden_size*cfg.hc_count),device='cuda',dtype=torch.float32)
        native_hash=gpu;disk_rows=rows
        class NativeLookup(nn.Module):
            def forward(self,input_ids,past_key_values):
                selected=native_hash(input_ids,past_key_values).cpu()
                return reference_lookup(model_dir,disk_rows.manifest,selected).to(input_ids.device)
        class PreparedLookup(nn.Module):
            def forward(self,input_ids,past_key_values):return edge_payload.to(input_ids.device)
        outputs=[];gradients=[]
        for lookup in (NativeLookup(),PreparedLookup()):
            ple.ple_embedding=lookup;hidden=original.detach().clone().requires_grad_()
            with torch.autocast('cuda',dtype=torch.bfloat16):result=ple(hidden,edge.cuda(),None)
            gradient=torch.autograd.grad(result.float().square().mean(),hidden)[0]
            outputs.append(result.detach().cpu());gradients.append(gradient.cpu())
        assert tensor_sha256(outputs[0])==tensor_sha256(outputs[1])
        assert tensor_sha256(gradients[0])==tensor_sha256(gradients[1])
        torch.save(dict(tokens=edge,row_ids=edge_ids,embeddings=edge_payload,hidden=original.cpu(),
            native_output=outputs[0],prepared_output=outputs[1],native_dhidden=gradients[0],prepared_dhidden=gradients[1]),out/'ple-operator-edge-fixture.pt')
        report.update(native_ple_output_bytes_exact=True,native_ple_hidden_gradient_bytes_exact=True)
    report.update(passed=True,seconds=time.monotonic()-started)
    (out/'ple-operator-check-report.json').write_text(json.dumps(report,indent=2)+'\n')
    del payload,reference,cpu,gpu,ple;gc.collect();torch.cuda.empty_cache()
    return report
