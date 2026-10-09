"""Run identities, immutable source snapshots and numerical comparison."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n')


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
    provenance = dict(mode=mode,config=config.as_dict(),source_hashes=hashes,
        script_commit=config.dispatch_commit or (commit.stdout.strip() if commit.returncode == 0 else None),
        source_identity='Actual source SHA256s are authoritative; commit is dispatch provenance.',
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','auto-round')},
        sample_hashes={path:sha(path) for path in config.samples},
        adapter_sha256=sha(config.adapter) if config.adapter else None,
        model_config_sha256=sha(Path(config.model)/'config.json'),python=sys.version,
        optimizer_updates=0 if mode == 'test' else None,pushed_to_remote=False)
    write(Path(output)/'provenance.json',provenance)
    return provenance


def compare(gradients, loss, baseline):
    import torch
    reference = torch.load(baseline['gradients'],map_location='cpu',weights_only=True)
    if set(reference) != set(gradients) or len(reference) != 744:
        raise ValueError('Raw gradient keys/count differ')
    rows = {};error = norm = other = dot = 0.
    for name,a in reference.items():
        b = gradients[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Gradient metadata differs: '+name)
        x,y = a.double(),b.double()
        aa,bb,ee,ab = [float(t.sum()) for t in (x*x,y*y,(x-y)**2,x*y)]
        error += ee;norm += aa;other += bb;dot += ab
        rows[name] = dict(bitwise_equal=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),
            relative_l2=(ee/aa)**.5 if aa else None,max_absolute_difference=float((x-y).abs().max()),
            reference_zero=aa==0,candidate_zero=bb==0,finite=bool(torch.isfinite(b).all()))
    expected = float(baseline['loss'])
    loss_exact = struct.pack('<f',loss) == struct.pack('<f',expected)
    exact = sum(row['bitwise_equal'] for row in rows.values())
    return dict(loss=loss,baseline_loss=expected,loss_bitwise_equal=loss_exact,
        loss_relative_change=(loss-expected)/expected,raw_tensors=len(rows),bitwise_equal_tensors=exact,
        global_relative_l2=(error/norm)**.5,cosine=dot/(norm*other)**.5,
        passed=loss_exact and exact==744,criterion='Refactor isolation: bitwise loss and all 744 raw gradients',
        baseline_gradients_sha256=sha(baseline['gradients']),tensors=rows)


def snapshot_imports(output):
    roots = ('/tmp/peft-autoround-compat/',str(Path(output).parent/'dependencies')+'/')
    hashes = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and name.endswith('.py') and name.startswith(roots):
            path = Path(name)
            if path.is_file():hashes[name] = sha(path)
    write(Path(output)/'imported-source-hashes.json',hashes)
