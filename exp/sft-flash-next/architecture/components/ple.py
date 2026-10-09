"""Frozen PLE preparation on CPU with native hashes and bounded DataLoader lookahead.

A source Dataset must return one complete, already encoded CPU batch (batch 1).
It may tokenize in __getitem__; this wrapper neither tokenizes nor truncates it.
Only the table is disk-backed. Prepared BF16 values are ordinary CPU tensors,
kept by the consumer through backward and moved to CUDA only at the PLE layer.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import struct
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

PREFIX = 'model.language_model.layers.1.ple.ple_embedding'
SOURCE_HASHES = {
    '__init__': 'dbd5dab9c6e34a95af63bdce35aaf9fbf84abd91fb8db361de9047e81da4741b',
    'forward': '07c0e8b547fee72ccced88c1f825cfbb07cf179f44cae4f1959138755d704968',
    '_shift_right_ignore_eos': '7865dc6b7742813157d3e57ca93ed8ca91d4148140443f15764b2575fe8d3969',
}


def native_class():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256(inspect.getsource(getattr(Qwen4ExpTextNGramEmbedding, name)).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError('Unsupported native PLE source: ' + name)
    return Qwen4ExpTextNGramEmbedding


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def table_manifest(model_dir):
    root = Path(model_dir)
    index_bytes = (root / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)['weight_map']
    records = {}
    headers = {}
    for key, filename in index.items():
        match = re.fullmatch(re.escape(PREFIX) + r'\.ngram_embedding\.shard_(\d+)\.weight', key)
        if not match:
            continue
        if Path(filename).name != filename:
            raise ValueError('Unsafe PLE shard filename')
        if filename not in headers:
            with (root / filename).open('rb') as stream:
                count = struct.unpack('<Q', stream.read(8))[0]
                raw = stream.read(count)
                headers[filename] = (json.loads(raw), count + 8, hashlib.sha256(raw).hexdigest())
        header, start, checksum = headers[filename]
        info = header[key];rows, width = info['shape'];begin, end = info['data_offsets']
        if info['dtype'] != 'BF16' or width != 160 or end-begin != rows*width*2:
            raise ValueError('Unsupported PLE table shape/dtype')
        if (root / filename).stat().st_size < start+end:
            raise ValueError('Truncated PLE checkpoint')
        records[int(match[1])] = dict(key=key, filename=filename, rows=rows, width=width,
            offset=start+begin, bytes=end-begin, header_sha256=checksum)
    if sorted(records) != list(range(128)):
        raise ValueError('Require all 128 original PLE shards')
    shards = [records[i] for i in range(128)]
    return dict(checkpoint_index_sha256=hashlib.sha256(index_bytes).hexdigest(), shards=shards,
                table_bytes=sum(row['bytes'] for row in shards),
                table_rows=sum(row['rows'] for row in shards), row_width=160, dtype='BF16')


class RowIds(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__();self.register_buffer('weight', torch.empty(0, device=device), persistent=False)
    def forward(self, ids):
        return ids.unsqueeze(-1)


def hash_computer(model_dir, device='cpu'):
    """Run the installed native forward unchanged, replacing only its final lookup."""
    cls = native_class()
    from transformers import AutoConfig
    from safetensors import safe_open
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).text_config
    instance = cls.__new__(cls);nn.Module.__init__(instance)
    instance.layer_idx = 1;instance.ngram_size=config.ngram_size
    instance.context_len=config.ngram_size-1;instance.heads_per_ngram=config.heads_per_ngram
    instance.eos_token_id=config.eos_token_id[0] if isinstance(config.eos_token_id,list) else config.eos_token_id
    index=json.loads((Path(model_dir)/'model.safetensors.index.json').read_text())['weight_map']
    for name in ('layer_multipliers','ngram_heads_vocab_sizes','ngram_heads_offsets'):
        key=PREFIX+'.'+name
        with safe_open(str(Path(model_dir)/index[key]), framework='pt',device='cpu') as archive:
            instance.register_buffer(name,archive.get_tensor(key).to(device))
    instance.ngram_embedding=RowIds(device)
    return instance


class DiskPLERows:
    """All original frozen rows remain available; no pruning or persistent row cache."""
    def __init__(self, model_dir):
        self.root=Path(model_dir);self.manifest=table_manifest(model_dir)
        self.offsets=np.cumsum([0]+[s['rows'] for s in self.manifest['shards']])
    def lookup(self, ids):
        if ids.device.type!='cpu' or ids.dtype!=torch.int64:
            raise ValueError('PLE preparation requires CPU int64 row IDs')
        flat=ids.contiguous().numpy().reshape(-1)
        unique,inverse=np.unique(flat,return_inverse=True)
        if len(unique) and (unique[0]<0 or unique[-1]>=self.offsets[-1]):
            raise IndexError('PLE row outside full table')
        values=np.empty((len(unique),160),dtype=np.int16)
        shard_ids=np.searchsorted(self.offsets,unique,side='right')-1
        for shard_id in np.unique(shard_ids):
            info=self.manifest['shards'][int(shard_id)];positions=np.flatnonzero(shard_ids==shard_id)
            mapped=np.memmap(self.root/info['filename'],dtype=np.int16,mode='r',offset=info['offset'],shape=(info['rows'],160))
            try:values[positions]=mapped[unique[positions]-self.offsets[shard_id]]
            finally:mapped._mmap.close()  # Do not accumulate process-resident mapped pages.
        result=torch.from_numpy(values[inverse]).view(torch.bfloat16).reshape(*ids.shape,160).flatten(-2)
        return result,dict(total_row_ids=len(flat),unique_rows=len(unique),raw_unique_bytes=values.nbytes,
                           lookup_bytes=result.numel()*result.element_size(),disk_read_passes=1)


class EncodedPaths(Dataset):
    """Diagnostic source; production may supply any complete-sequence CPU Dataset."""
    def __init__(self, paths):self.paths=list(paths)
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):return torch.load(self.paths[index],map_location='cpu',weights_only=True)


class PreparedPLEDataset(Dataset):
    def __init__(self, source, model_dir, event_path=None):
        self.source=source;self.model_dir=str(model_dir);self.event_path=event_path
        self._hash=None;self._rows=None
    def __len__(self):return len(self.source)
    def __getitem__(self,index):
        started=time.monotonic_ns()
        batch=self.source[index]
        if not isinstance(batch,dict) or 'input_ids' not in batch:
            raise ValueError('Source must return a complete encoded batch')
        tokens=batch['input_ids']
        if tokens.device.type!='cpu' or tokens.dtype!=torch.int64 or tokens.ndim!=2 or tokens.shape[0]!=1:
            raise ValueError('PLE preparation requires one complete CPU int64 token sequence')
        mask=batch.get('attention_mask')
        if mask is not None and not bool(mask.bool().all()):raise ValueError('Opt4 capture requires unpadded input')
        if self._hash is None:self._hash=hash_computer(self.model_dir);self._rows=DiskPLERows(self.model_dir)
        ids=self._hash(tokens,None)
        payload,stats=self._rows.lookup(ids)
        stats.update(index=int(index),worker_pid=os.getpid(),tokens=tokens.shape[1],
                     started_monotonic_ns=started,finished_monotonic_ns=time.monotonic_ns(),
                     input_ids_sha256=tensor_sha256(tokens),row_ids_sha256=tensor_sha256(ids),
                     payload_sha256=tensor_sha256(payload))
        if self.event_path:
            with open(self.event_path,'a') as stream:stream.write(json.dumps(stats)+'\n')
        return dict(batch=batch,ple_embeddings=payload,ple_row_ids=ids,preparation=stats)


def prepared_loader(source, model_dir, lookahead=2, event_path=None):
    """One CPU worker, at most lookahead outstanding sample preparations.

    With the consumer retaining one current sample, at most 1+lookahead sample
    results exist. Dataset, ids, dedup temporaries and IPC add bounded overhead.
    Spawn is required; callers must use a guarded __main__ entry point.
    """
    if not isinstance(lookahead,int) or not 1<=lookahead<=2:
        raise ValueError('PLE lookahead must be one or two samples')
    return DataLoader(PreparedPLEDataset(source,model_dir,event_path),batch_size=None,
                      num_workers=1,prefetch_factor=lookahead,persistent_workers=True,
                      multiprocessing_context='spawn',pin_memory=False)


class DiskTablePlaceholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.empty(0, device='cpu'), persistent=False)
    def forward(self, ids):
        raise RuntimeError('Activate a prepared sample before forward')


def checkpoint_view(model_dir, destination):
    """Exclude only table tensors from HF loading; full table stays accessible."""
    root = Path(model_dir)
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    manifest = table_manifest(root)
    index = json.loads((root/'model.safetensors.index.json').read_text())
    omitted = [key for key in index['weight_map'] if key.startswith(PREFIX+'.ngram_embedding.shard_')]
    if len(omitted) != 128:
        raise ValueError('Expected 128 PLE shard tensors')
    index['weight_map'] = {key:value for key,value in index['weight_map'].items() if key not in omitted}
    index['metadata']['total_size'] -= manifest['table_bytes']
    for path in root.iterdir():
        if path.name != 'model.safetensors.index.json':
            (destination/path.name).symlink_to(path.resolve())
    (destination/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
    return destination, manifest


from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding


class PreparedNGramEmbedding(Qwen4ExpTextNGramEmbedding):
    def forward(self, input_ids, past_key_values):
        if past_key_values is not None or not torch.equal(input_ids.detach().cpu(), self.prepared_input_ids):
            raise ValueError('Prepared PLE identity/cache mismatch')
        return self.prepared_payload.to(input_ids.device)
