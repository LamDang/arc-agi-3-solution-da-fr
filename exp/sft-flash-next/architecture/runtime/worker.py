"""Detached Jupyter job supervisor. This file never imports Torch or model code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['test','train'],default='test')
    p.add_argument('--timeout',type=int,default=1200)
    args = p.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent/'source'
    job = config_path.parent
    config = json.loads(config_path.read_text())
    if Path(config['output']).exists():raise FileExistsError('Attempt output already exists')
    manifest = json.loads((job/'source-hashes.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Uploaded source hash differs: '+name)
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    if gpu:raise RuntimeError('GPU process already running; no concurrent job allowed')
    packages = ['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao',str(root)]
    # Dependencies are the already verified wheels from the interrupted Opt6 job.
    opt = config.get('optimizations',{})
    if opt.get('head') in ['cce_exact','liger_flce'] or opt.get('liger_rmsnorm') or opt.get('liger_swiglu'):
        old = Path('/kaggle/working/gradient-audit-20261009/liger-opt6-overfit-v9-launch-attempt-20261009204949-5b997b5f')
        if opt.get('head') == 'cce_exact':
            packages.insert(0,str(old/'cce-runtime/ml-cross-entropy-3de376c106a1916bc5e1b619f9c77c87a461ee1c'))
        if opt.get('head') == 'liger_flce' or opt.get('liger_rmsnorm') or opt.get('liger_swiglu'):
            packages.insert(0,str(old/'liger-runtime'))
    bootstrap = job/'bootstrap.py'
    script = root/(args.mode+'.py')
    bootstrap.write_text('import sys,runpy\n'+
        "sys.path=[p for p in sys.path if p!='/usr/local/lib/python3.13/dist-packages']\n"+
        'sys.path[:0]='+repr(packages)+'\n'+
        'sys.argv='+repr([str(script),'--config',str(config_path)])+'\n'+
        "if __name__ == '__main__':\n    import unittest\n    suite=unittest.defaultTestLoader.discover("+repr(str(root/'tests'))+")\n    result=unittest.TextTestRunner(verbosity=2).run(suite)\n    if not result.wasSuccessful(): raise RuntimeError('Component checks failed before model loading')\n    runpy.run_path("+repr(str(script))+",run_name='__main__')\n")
    env = {**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    started = time.monotonic()
    with (job/'process.log').open('w') as log:
        child = subprocess.Popen(['/usr/bin/python3',str(bootstrap)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (job/'child-pid').write_text(str(child.pid))
        timeout = False
        try:code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timeout = True;os.killpg(child.pid,signal.SIGTERM)
            try:code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    (job/'monitor.json').write_text(json.dumps(dict(returncode=code,timed_out=timeout,seconds=time.monotonic()-started,pid=child.pid),indent=2)+'\n')


if __name__ == '__main__':main()
