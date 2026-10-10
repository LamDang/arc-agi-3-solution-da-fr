"""Take a fixed suffix of a real encoded request without splitting images.

Preserves every target token. Context is cropped and positions are recomputed
by HF; this is a diagnostic window, not the original full-context request.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def prepare(source, destination, config_path, maximum=16384):
    source, destination = Path(source), Path(destination)
    metadata = json.loads((source/'sample.json').read_text())
    batch = torch.load(source/'sample.pt', map_location='cpu', weights_only=True)
    config = json.loads(Path(config_path).read_text())
    ids = batch['input_ids'][0]
    starts = (ids == config['vision_start_token_id']).nonzero().flatten().tolist()
    ends = (ids == config['vision_end_token_id']).nonzero().flatten().tolist()
    assert len(starts) == len(ends) == len(batch['image_grid_thw'])
    assert all(s < e for s,e in zip(starts,ends))
    cut = max(0, len(ids)-maximum)
    for s,e in zip(starts,ends):
        if s < cut <= e:
            cut = e+1
    prompt = metadata['row']['prompt_tokens']
    assert cut < prompt
    dropped_images = sum(e < cut for e in ends)
    dropped_patches = int(batch['image_grid_thw'][:dropped_images].prod(dim=-1).sum())
    result = {}
    for name, value in batch.items():
        if name in ('input_ids','attention_mask','mm_token_type_ids'):
            result[name] = value[:,cut:].contiguous()
        elif name == 'pixel_values':
            result[name] = value[dropped_patches:].contiguous()
        elif name == 'image_grid_thw':
            result[name] = value[dropped_images:].contiguous()
        else:
            raise ValueError(f'Unhandled multimodal field: {name}')
    merge = config['vision_config']['spatial_merge_size']
    assert (result['input_ids'] == config['image_token_id']).sum().item() == int(result['image_grid_thw'].prod(-1).sum())//merge**2
    assert torch.equal(result['input_ids'][:,prompt-cut:], batch['input_ids'][:,prompt:])
    destination.mkdir(parents=True,exist_ok=False)
    torch.save(result,destination/'sample.pt')
    with (destination/'sample.pt').open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    report = dict(source_sample=metadata['row']['sample_id'], source_metadata=metadata,
                  cropped_context_tokens=cut, dropped_images=dropped_images,
                  total_tokens=result['input_ids'].shape[1],prompt_tokens=prompt-cut,
                  target_tokens=len(ids)-prompt,images=len(result['image_grid_thw']),
                  sample_sha256=digest,diagnostic_only=True)
    (destination/'sample.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_metadata'}),flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--max-tokens',type=int,default=16384)
    a=p.parse_args()
    prepare(a.source,a.out,a.config,a.max_tokens)
