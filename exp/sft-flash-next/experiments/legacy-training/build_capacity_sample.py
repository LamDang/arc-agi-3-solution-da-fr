"""Build a disposable 130K capacity composite from the verified real panel.

Whole requests provide context, whole final replies provide 10K target tokens.
This is not a coherent new teacher trajectory or a training/evaluation dataset.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from dataset import Requests
from data import hf_messages, render


def subset(rows, field, limit):
    dp = {0: []}
    for i, row in enumerate(rows):
        for total, ids in list(dp.items()):
            value = total + row[field]
            if value <= limit and (value not in dp or len(ids)+1 < len(dp[value])):
                dp[value] = ids+[i]
    total = max(dp)
    return total, dp[total]


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--out',required=True)
    p.add_argument('--context-tokens',type=int,default=120000)
    a=p.parse_args();bundle=Requests(a.panel);rows=bundle.rows
    context_limit=a.context_tokens
    n_context, context_indices=subset(rows,'total_tokens',context_limit)
    n_target, target_indices=subset(rows,'target_tokens',10000)
    assert n_target==10000 and 0<=context_limit-n_context<=32
    encs=[];context_sources=[]
    for i in context_indices:
        enc,ann=bundle.get(rows[i]);encs.append(enc)
        context_sources.append({k:rows[i][k] for k in ('sample_id','total_tokens','images','line_sha256')})
        print('context',context_sources[-1],flush=True)
    target_parts=[];target_sources=[];prefix=None
    for i in target_indices:
        row=rows[i]
        with (bundle.root/'requests.jsonl').open('rb') as f:
            f.seek(row['offset']);sample=json.loads(f.read(row['length']))
        text=render(bundle.processor,hf_messages(sample['messages']),sample)
        ids=bundle.processor.tokenizer(text,add_special_tokens=False)['input_ids']
        count=row['target_tokens']
        target_parts.append(torch.tensor(ids[-count:]).view(1,-1))
        if prefix is None:prefix=torch.tensor(ids[-count-(context_limit-n_context):-count]).view(1,-1)
        target_sources.append({k:row[k] for k in ('sample_id','target_tokens','line_sha256')})
    target=torch.cat(target_parts,dim=1)
    image_id=bundle.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    # Small filler comes from real text, never an orphan image placeholder.
    filler_source='pre-target suffix'
    if bool((prefix==image_id).any()):
        prefix=target[:, :context_limit-n_context]
        filler_source='generated-target text prefix'

    ids=torch.cat([*(e['input_ids'] for e in encs),prefix,target],dim=1)
    image_id=bundle.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    result=dict(input_ids=ids,
                image_grid_thw=torch.cat([e['image_grid_thw'] for e in encs]),
                pixel_values=torch.cat([e['pixel_values'] for e in encs]),
                mm_token_type_ids=(ids==image_id).int())
    assert ids.shape==(1,context_limit+10000)
    assert int((ids==image_id).sum())==int(result['image_grid_thw'].prod(-1).sum())//4
    assert not bool((target==image_id).any())
    assert result['pixel_values'].shape[0]==int(result['image_grid_thw'].prod(-1).sum())
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wb',compresslevel=3) as f:torch.save(result,f)
    meta=dict(panel_manifest_sha256=bundle.manifest['sha256'],context_tokens=context_limit,target_tokens=10000,
              whole_context_tokens=n_context,real_filler_tokens=prefix.numel(),filler_source=filler_source,total_tokens=context_limit+10000,
              images=len(result['image_grid_thw']),pixel_values_shape=list(result['pixel_values'].shape),
              context_sources=context_sources,target_sources=target_sources,
              purpose='Disposable capacity test only; all updates discarded. Composite is not a coherent teacher trajectory.',
              composition='Whole real requests concatenated causally; real text suffix before a target; whole final replies concatenated. No image blocks or original requests truncated.',
              input_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
              archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),archive_bytes=out.stat().st_size)
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
