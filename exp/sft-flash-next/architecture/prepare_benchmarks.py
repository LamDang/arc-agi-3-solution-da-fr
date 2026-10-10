"""Make diagnostic capacity prefixes from one real interleaved trajectory.

This does not alter production trajectory preparation or its no-truncation rule.
Assistant labels use the existing verified trajectory encoder. Prefix boundaries
inside a vision block are rejected, never padded or silently moved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--processor',required=True)
    parser.add_argument('--encoder-root',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    sys.path.insert(0,args.encoder_root)
    from dataset import load_processor,file_hash
    from trajectory_encode import encode
    import torch
    torch.set_num_threads(4)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    sample=json.loads(Path(args.source).read_text())
    processor=load_processor(args.processor)
    batch,annotation=encode(processor,sample)
    ids=batch['input_ids'][0]
    labels=torch.full_like(batch['input_ids'],-100)
    positions=torch.tensor(annotation['positions'],dtype=torch.long)
    labels[0,positions]=ids[positions]
    start=processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end=processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    starts=(ids==start).nonzero().flatten().tolist()
    ends=(ids==end).nonzero().flatten().tolist()
    grids=batch.get('image_grid_thw')
    assert len(starts)==len(ends)==len(grids)
    assert all(a<b for a,b in zip(starts,ends))
    full_length=len(ids);rows=[]
    for tokens in [32000,64000,96000,120000]:
        if tokens>full_length:raise ValueError('Source is too short; no padding or repetition')
        if any(a<tokens<=b for a,b in zip(starts,ends)):
            raise ValueError(f'{tokens} cuts inside a vision block; choose another real source')
        images=sum(b<tokens for b in ends)
        patches=int(grids[:images].prod(dim=-1).sum())
        prefix={}
        for key,value in batch.items():
            if key in {'input_ids','attention_mask','mm_token_type_ids'}:
                assert value.shape==batch['input_ids'].shape
                prefix[key]=value[:,:tokens].clone()
            elif key=='image_grid_thw':prefix[key]=value[:images].clone()
            elif key=='pixel_values':prefix[key]=value[:patches].clone()
            else:raise ValueError('Unsupported processor field: '+key)
        prefix['labels']=labels[:,:tokens].clone()
        targets=int((prefix['labels'][:,1:]!=-100).sum())
        if not targets:raise ValueError('Prefix has no assistant targets')
        path=root/f'sample-{tokens}.pt';torch.save(prefix,path)
        row=dict(path=path.name,tokens=tokens,targets=targets,target_fraction=targets/(tokens-1),
            images=images,sha256=file_hash(path),bytes=path.stat().st_size,
            objective='all assistant positions, interleaved with ignored context',
            complete_trajectory=False,purpose='diagnostic capacity prefix only')
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(source_id=sample['id'],source_sha256=file_hash(args.source),
        full_tokens=full_length,full_targets=annotation['target_tokens'],rows=rows,
        encoder_sources={name:file_hash(Path(args.encoder_root)/name)
            for name in ['dataset.py','trajectory_encode.py']},
        script_sha256=file_hash(__file__),production_policy_changed=False)
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'annotation.json').write_text(json.dumps(annotation,indent=2)+'\n')


if __name__=='__main__':main()
