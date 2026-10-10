"""Run identities, immutable source snapshots and numerical comparison."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import shutil
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n')


def retain_raw_gradients(config, mode):
    """Only the reference qualification owns a persistent raw gradient archive."""
    return mode == 'test' and config.architecture == 'reference'


def snapshot(output, config, mode):
    root = Path(__file__).resolve().parents[1]
    destination = Path(output)/'sources'
    destination.mkdir()
    hashes = {}
    for path in sorted(root.rglob('*.py')):
        if any(part in {'results','tests','reports','__pycache__'} for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[str(relative)] = sha(path)
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    inputs = Path(output)/'inputs';inputs.mkdir()
    archived = {}
    for label,original in [(f'sample-{i:04d}.pt',name) for i,name in enumerate(config.samples)]+([('adapter.pt',config.adapter)] if config.adapter else []):
        target = inputs/label
        shutil.copyfile(original,target)
        archived[original] = dict(path=str(target.relative_to(output)),sha256=sha(target))
    provenance = dict(mode=mode,config=config.as_dict(),source_hashes=hashes,archived_inputs=archived,
        script_commit=config.dispatch_commit or (commit.stdout.strip() if commit.returncode == 0 else None),
        source_identity='Actual source SHA256s are authoritative; commit is dispatch provenance.',
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','auto-round')},
        sample_hashes={path:sha(path) for path in config.samples},
        adapter_sha256=sha(config.adapter) if config.adapter else None,
        model_config_sha256=sha(Path(config.model)/'config.json'),python=sys.version,
        optimizer_updates=0 if mode == 'test' else None,pushed_to_remote=False)
    write(Path(output)/'provenance.json',provenance)
    return provenance


def reference_tensors(path):
    """Verify and stream saved reference shards without retaining a second full state."""
    import torch
    path=Path(path)
    seen=set()
    if path.suffix=='.json':
        manifest=json.loads(path.read_text())
        for row in manifest['shards']:
            shard=path.parent/row['path']
            if sha(shard)!=row['sha256']:raise ValueError('Baseline gradient shard hash differs: '+str(shard))
            state=torch.load(shard,map_location='cpu',weights_only=True)
            if seen & state.keys():raise ValueError('Duplicate gradient shard keys')
            seen.update(state)
            yield from state.items()
            del state
        if len(seen)!=manifest['tensors']:raise ValueError('Baseline gradient count differs')
    else:
        yield from torch.load(path,map_location='cpu',weights_only=True).items()


def compare_initial(state, baseline):
    """Exact initialization gate; retain only identity and mismatch statistics."""
    import torch
    path=baseline['initial_adapter'];seen=set();different=[]
    for name,a in reference_tensors(path):
        seen.add(name)
        b=state.get(name)
        if b is None or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):different.append(name)
    if seen != set(state) or not seen:raise ValueError('Initial adapter keys/count differ')
    return dict(passed=not different,raw_tensors=len(seen),bitwise_equal_tensors=len(seen)-len(different),
        different_tensors=different,baseline_initial_sha256=sha(path),raw_initial_state_retained=False)


def compare(gradients, loss, baseline):
    import torch
    rows = {};error = norm = other = dot = 0.
    source=baseline['gradients']
    reference=source.items() if isinstance(source,dict) else reference_tensors(source)
    for name,a in reference:
        if name not in gradients:raise ValueError('Raw gradient keys/count differ')
        b = gradients[name]
        if a.shape != b.shape:
            raise ValueError('Gradient shape differs: '+name)
        x,y = a.double(),b.double()
        aa,bb,ee,ab = [float(t.sum()) for t in (x*x,y*y,(x-y)**2,x*y)]
        error += ee;norm += aa;other += bb;dot += ab
        rows[name] = dict(bitwise_equal=a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),
            reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),
            reference_norm=aa**.5,candidate_norm=bb**.5,
            relative_l2=(ee/aa)**.5 if aa else None,max_absolute_difference=float((x-y).abs().max()),
            reference_zero=aa==0,candidate_zero=bb==0,finite=bool(torch.isfinite(b).all()))
    if set(rows) != set(gradients) or not rows:raise ValueError('Raw gradient keys/count differ')
    expected = float(baseline['loss'])
    loss_exact = struct.pack('<f',loss) == struct.pack('<f',expected)
    exact = sum(row['bitwise_equal'] for row in rows.values())
    return dict(loss=loss,baseline_loss=expected,loss_bitwise_equal=loss_exact,
        loss_relative_change=(loss-expected)/expected,raw_tensors=len(rows),bitwise_equal_tensors=exact,
        global_relative_l2=(error/norm)**.5 if norm else None,cosine=dot/(norm*other)**.5 if norm and other else None,
        reference_norm=norm**.5,candidate_norm=other**.5,error_norm=error**.5,
        nonzero_reference_tensors=sum(not row['reference_zero'] for row in rows.values()),
        bitwise_equal_nonzero_reference_tensors=sum(row['bitwise_equal'] and not row['reference_zero'] for row in rows.values()),
        passed=loss_exact and exact==len(rows),criterion=f'Bitwise loss and all {len(rows)} raw gradients',
        baseline_gradients_sha256=None if isinstance(source,dict) else sha(source),tensors=rows)


def snapshot_imports(output):
    roots = ('/tmp/peft-autoround-compat/',str(Path(__file__).resolve().parents[2]/'dependencies')+'/')
    hashes = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and name.endswith('.py') and name.startswith(roots):
            path = Path(name)
            if path.is_file():hashes[name] = sha(path)
    write(Path(output)/'imported-source-hashes.json',hashes)
