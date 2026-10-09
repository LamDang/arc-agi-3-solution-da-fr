"""Independently verify a collected refactor capture without importing Torch.

Usage: python reports/verify_iso.py results/<attempt-id>
Requires NumPy. Reads saved evidence only; no GPU job or network access.
"""
import argparse
import collections
import hashlib
import io
import json
import pickle
from pathlib import Path
import struct
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent/'train/gradient-results/reference-runner-v0'


class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if module=='torch' and name in ('FloatStorage','BFloat16Storage'):
            return {'FloatStorage':'<f4','BFloat16Storage':'<u2'}[name]
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):
            return lambda storage,offset,size,stride,*rest:(storage,offset,size,stride)
        raise ValueError((module,name))
    def persistent_load(self,item):
        kind,dtype,key,device,numel=item
        assert kind=='storage' and device=='cpu'
        return key,numel,dtype


class Tensors:
    def __init__(self,path):
        self.zip=zipfile.ZipFile(path)
        self.prefix=next(n[:-len('data.pkl')] for n in self.zip.namelist() if n.endswith('/data.pkl'))
        assert self.zip.read(self.prefix+'byteorder')==b'little'
        self.state=Reader(io.BytesIO(self.zip.read(self.prefix+'data.pkl'))).load()
    def tensor(self,name):
        (key,stored,dtype),offset,shape,stride=self.state[name]
        dt=np.dtype(dtype);raw=self.zip.read(self.prefix+'data/'+key)
        assert len(raw)==stored*dt.itemsize
        return np.ndarray(shape,dtype=dt,buffer=raw,offset=offset*dt.itemsize,
                          strides=tuple(x*dt.itemsize for x in stride))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    job=Path(args.attempt).resolve();output=job/'output'
    read=lambda path:json.loads(path.read_text())
    config=read(job/'config.json');result=read(output/'result.json');monitor=read(job/'monitor.json')
    assert config['architecture']=='optimized' and config['optimizations']=={}
    assert not monitor['timed_out']
    process_success = monitor['returncode']==0
    known_final_event_failure = False
    if not process_success:
        failure=read(output/'failure.json')
        # Only the preserved initial attempt's post-capture logger bug is allowed.
        known_final_event_failure = (job.name=='20261009213548935-2c90d284'
            and monitor['returncode']==1
            and "multiple values for keyword argument 'elapsed_seconds'" in failure['error']
            and "event('finished',**result)" in failure['traceback'])
        assert known_final_event_failure, failure
    assert result['optimizer_updates']==0 and not result['clipping_applied'] and result['iso_verified']
    assert result['tokens']==16249 and result['targets']==651
    assert read(output/'components.json')==[]
    for name,digest in read(job/'source-hashes.json').items():
        assert sha(job/'source'/name)==digest,name
    provenance=read(output/'provenance.json')
    for name,digest in provenance['source_hashes'].items():
        assert sha(output/'sources'/name)==sha(job/'source'/name)==digest,name
    old_config=read(BASELINE/'resolved-config.json')
    for name in ('sample','adapter','model_config'):
        assert config['expected_sha256'][name]==old_config['expected_sha256'][name]
    verified={}
    for filename in ('initial-adapter.pt','gradients.pt'):
        old=Tensors(BASELINE/filename);new=Tensors(output/filename)
        assert set(old.state)==set(new.state) and len(new.state)==744
        nonzero=0
        for name in old.state:
            a,b=old.tensor(name),new.tensor(name)
            assert a.shape==b.shape and a.dtype==b.dtype==np.dtype('<f4')
            assert a.tobytes()==b.tobytes() and np.isfinite(b).all(),name
            nonzero+=int(np.count_nonzero(b)>0)
        assert nonzero==744
        verified[filename]=dict(sha256=sha(output/filename),bitwise_equal_tensors=744,nonzero_tensors=744)
    expected=read(BASELINE/'result.json')['measured_loss']
    assert struct.pack('<f',result['loss'])==struct.pack('<f',expected)
    comparison=read(output/'comparison.json')
    assert comparison['passed'] and comparison['global_relative_l2']==0.0 and comparison['cosine']>0.999999999999
    resources=read(output/'resources.json')
    assert [row['phase'] for row in resources]==['loading','forward','backward','gradient_export']
    for row in resources:
        assert row['seconds']>0 and row['samples']>0 and row['rss_bytes']>0
        assert row['cuda_peak_allocated_bytes']>0 and row['cuda_peak_reserved_bytes']>=row['cuda_peak_allocated_bytes']
    manifest={}
    for path in sorted(job.rglob('*')):
        if not path.is_file() or path.name=='file-hashes.json':continue
        relative=str(path.relative_to(job))
        manifest[relative]=dict(bytes=path.stat().st_size,sha256=sha(path))
        if path.suffix in ('.py','.mjs','.json','.jsonl','.txt','.log','.md'):
            assert 'https://kkb-production.jupyter-proxy.kaggle.net/k/' not in path.read_text(errors='replace'),relative
    (job/'file-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(passed=True,attempt=job.name,execution_commit=config['dispatch_commit'],
        numerical_iso_verified=True,process_success=process_success,
        known_post_capture_logging_failure=known_final_event_failure,
        architecture='optimized',optimizations='all disabled',loss=result['loss'],loss_bitwise_equal=True,
        raw_gradients=744,bitwise_equal_gradients=744,all_gradients_nonzero_and_finite=True,
        initialization_bitwise_equal=True,global_gradient_relative_l2=0.0,
        optimizer_updates=0,clipping_applied=False,source_hashes_verified=True,
        preserved_files=len(manifest),gradient_archive_sha256=verified['gradients.pt']['sha256'],
        resources=resources,scope='Full 16K native-equivalent clean builder/loop with every optimization disabled. Enabled flags require separate qualification.',
        pushed_to_remote=False)
    (ROOT/'reports/refactor-iso.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/f'reports/refactor-iso-{job.name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='resources'},indent=2))


if __name__=='__main__':main()
