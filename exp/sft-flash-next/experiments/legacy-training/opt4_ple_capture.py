"""Diagnostic Opt4 loader/storage change; native PLE computation stays unchanged."""
import inspect
import json
from pathlib import Path
import textwrap
import time
from types import MethodType

import torch
from torch import nn
from ple_preparation import PREFIX, native_class, table_manifest, prepared_loader, EncodedPaths, tensor_sha256


class DiskTablePlaceholder(nn.Module):
    def __init__(self, rows, width):
        super().__init__();self.num_embeddings=rows;self.embedding_dim=width
        self.register_buffer('weight',torch.empty(0,device='cpu'),persistent=False)
    def forward(self,ids):raise RuntimeError('Opt4 requires an activated prepared sample')


def install_for_capture(model_dir,sample_path,launch_dir,report_path,lookahead=2):
    """Install before native reference loads its batch/model. All patches are local."""
    launch=Path(launch_dir);manifest=table_manifest(model_dir)
    cls=native_class();original_init=cls.__init__
    source=textwrap.dedent(inspect.getsource(original_init))
    old='nn.Embedding(padded_vocab_size, head_dim_per_ngram)'
    assert source.count(old)==1
    namespace=dict(original_init.__globals__,DiskTablePlaceholder=DiskTablePlaceholder)
    exec(compile(source.replace(old,'DiskTablePlaceholder(padded_vocab_size, head_dim_per_ngram)').replace('super().__init__()', 'nn.Module.__init__(self)'),'<opt4-ple-init>','exec'),namespace)
    cls.__init__=namespace['__init__']
    # The index view omits only frozen table tensors; their complete original
    # shard manifest stays accessible to the CPU preparer. All other weights
    # still load through the original HF/AutoRound path.
    root=Path(model_dir);view=launch/'ple-disk-model-view';view.mkdir()
    index=json.loads((root/'model.safetensors.index.json').read_text())
    omitted=[key for key in index['weight_map'] if key.startswith(PREFIX+'.ngram_embedding.shard_')]
    assert len(omitted)==128
    index['weight_map']={key:value for key,value in index['weight_map'].items() if key not in omitted}
    index['metadata']['total_size']-=manifest['table_bytes']
    for path in root.iterdir():
        if path.name!='model.safetensors.index.json':(view/path.name).symlink_to(path.resolve())
    index_bytes=json.dumps(index,indent=2)+'\n';(view/'model.safetensors.index.json').write_text(index_bytes)
    (launch/'ple-filtered-model-index.json').write_text(index_bytes)
    (launch/'ple-full-table-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import transformers
    original_load=transformers.AutoModelForImageTextToText.from_pretrained
    def load(inner_cls,path,*args,**kwargs):
        if Path(path)!=root:raise ValueError('Unexpected Opt4 model path')
        return original_load(str(view),*args,**kwargs)
    transformers.AutoModelForImageTextToText.from_pretrained=classmethod(load)
    report=dict(implementation='opt4_disk_prepared_ple',table_bytes=manifest['table_bytes'],
        full_table_available=True,fully_resident_table_bytes=0,table_parameters=0,
        native_ple_layer_forward_unchanged=True,native_hash_forward_unchanged=True,
        num_workers=1,prefetch_factor=lookahead,maximum_prepared_samples=1+lookahead,
        diagnostic_source='three identical pre-encoded anchors; only first receives GPU forward/backward',
        preparation_events='launch-ple-preparation-events.jsonl',calls=[],optimizer_updates=0)
    path=Path(report_path)
    def save():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    loader=prepared_loader(EncodedPaths([str(sample_path)]*(1+lookahead)),model_dir,lookahead,
                           str(launch/'ple-preparation-events.jsonl'))
    iterator=iter(loader);state={'current':None}
    original_torch_load=torch.load
    def torch_load(file,*args,**kwargs):
        if isinstance(file,(str,Path)) and Path(file)==Path(sample_path):
            if state['current'] is not None:raise RuntimeError('Unexpected second anchor load')
            started=time.monotonic_ns();current=next(iterator);state['current']=current
            report['first_sample_wait_seconds']=(time.monotonic_ns()-started)/1e9
            report['current_sample']=current['preparation'];report['current_payload_bytes']=current['ple_embeddings'].numel()*2
            torch.save(current['ple_embeddings'],launch/'ple-prepared-embeddings.pt')
            torch.save(current['ple_row_ids'],launch/'ple-prepared-row-ids.pt')
            save();return current['batch']
        return original_torch_load(file,*args,**kwargs)
    torch.load=torch_load
    import peft
    original_factory=peft.get_peft_model
    def factory(base,*args,**kwargs):
        model=original_factory(base,*args,**kwargs)
        module=base.model.language_model.layers[1].ple.ple_embedding
        if not isinstance(module.ngram_embedding,DiskTablePlaceholder):raise RuntimeError('Resident table was not bypassed')
        if module.ngram_embedding.num_embeddings!=manifest['table_rows']:raise ValueError('PLE row count differs')
        if any('ngram_embedding' in name for name,_ in base.named_parameters()):raise RuntimeError('Unexpected resident PLE parameter')
        def prepared_forward(self,input_ids,past_key_values):
            if past_key_values is not None:raise ValueError('Opt4 requires cache-free complete samples')
            current=state['current']
            if current is None or not torch.equal(input_ids.detach().cpu(),current['batch']['input_ids']):
                raise ValueError('Prepared PLE input identity differs')
            started=time.monotonic_ns()
            output=current['ple_embeddings'].to(input_ids.device)
            report['calls'].append(dict(sequence=len(report['calls']),payload_sha256=current['preparation']['payload_sha256'],
                prepared_cpu_storage_pointer=current['ple_embeddings'].untyped_storage().data_ptr(),
                output_shape=list(output.shape),output_dtype=str(output.dtype),
                transfer_enqueue_seconds=(time.monotonic_ns()-started)/1e9,disk_reads=0))
            save();return output
        module.forward=MethodType(prepared_forward,module)
        save();return model
    peft.get_peft_model=factory
    def finalize():
        events=Path(launch/'ple-preparation-events.jsonl')
        report['worker_preparations']=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        report['current_retained_through_backward']=len(report['calls'])==2 and len({row['prepared_cpu_storage_pointer'] for row in report['calls']})==1
        report['maximum_current_plus_queued_payload_bytes']=(1+lookahead)*report.get('current_payload_bytes',0)
        if getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()
        report['workers_shutdown']=True;save();state['current']=None
    return finalize
