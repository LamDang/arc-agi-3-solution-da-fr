"""Detached Jupyter job supervisor. This file never imports Torch or model code."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from fla_environment import configure_fla_environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['test','train','benchmark'],default='test')
    p.add_argument('--timeout',type=int,default=1200)
    p.add_argument('--entrypoint',choices=['diagnostics/gdn_backward.py','diagnostics/expert_replay.py','diagnostics/expert_head_replay.py','diagnostics/hyperconnection_replay.py','diagnostics/hyperconnection_focus.py','diagnostics/host_memory.py'])
    args = p.parse_args()
    lock_path = Path('/tmp/flash-next-architecture.lock')
    # flock needs an open descriptor, not write access to an existing lock.
    lock = lock_path.open('r' if lock_path.exists() else 'w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = Path(args.config).resolve()
    root = config_path.parent/'source'
    job = config_path.parent
    config = json.loads(config_path.read_text())
    if args.mode=='train' and config.get('fla_numeric_profile') is not None:
        raise ValueError('FLA numerical profiles are test-only; real training uses native autotuning')
    if Path(config['output']).exists():raise FileExistsError('Attempt output already exists')
    if Path(config['output']) != job/'output':
        Path(config['output']).parent.mkdir(parents=True,exist_ok=True)
        (job/'output').symlink_to(config['output'],target_is_directory=True)
    manifest = json.loads((job/'source-hashes.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Uploaded source hash differs: '+name)
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    if gpu:raise RuntimeError('GPU process already running; no concurrent job allowed')
    packages = ['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao',str(root)]
    # Isolated, pinned dependency archives travel with each attempt.
    opt = config.get('optimizations',{})
    registry = json.loads((root/'configs/dependencies.json').read_text())
    needed = []
    if opt.get('head') == 'cce_exact':needed.append('cce')
    if opt.get('head') == 'liger_flce' or opt.get('liger_rmsnorm') or opt.get('liger_swiglu'):needed.append('liger')
    for name in needed:
        dependency = registry[name]
        archive = job/'dependencies'/dependency['filename']
        if hashlib.sha256(archive.read_bytes()).hexdigest() != dependency['sha256']:
            raise RuntimeError('Dependency archive hash differs: '+name)
        destination = job/'dependencies'/name
        destination.mkdir()
        if name == 'liger':
            import zipfile
            with zipfile.ZipFile(archive) as package:
                if any(Path(member).is_absolute() or '..' in Path(member).parts for member in package.namelist()):
                    raise ValueError('Unsafe dependency archive')
                package.extractall(destination)
            packages.insert(0,str(destination))
        else:
            import tarfile
            with tarfile.open(archive) as package:package.extractall(destination,filter='data')
            packages.insert(0,str(destination/dependency['archive_prefix']))
    bootstrap = job/'bootstrap.py'
    if args.entrypoint and args.mode!='test':raise ValueError('Diagnostics require test mode')
    script = root/(args.entrypoint or (args.mode+'.py'))
    bootstrap.write_text('import sys,runpy\n'+
        "sys.path=[p for p in sys.path if p!='/usr/local/lib/python3.13/dist-packages']\n"+
        'sys.path[:0]='+repr(packages)+'\n'+
        'sys.argv='+repr([str(script),'--config',str(config_path)])+'\n'+
        "if __name__ == '__main__':\n    import unittest\n    suite=unittest.defaultTestLoader.discover("+repr(str(root/'tests'))+")\n    result=unittest.TextTestRunner(verbosity=2).run(suite)\n    if not result.wasSuccessful(): raise RuntimeError('Component checks failed before model loading')\n    runpy.run_path("+repr(str(script))+",run_name='__main__')\n")
    env = {**os.environ,'FLASH_NEXT_CHECK_EXPERT_OFFLOAD':'1' if opt.get('offload_routed_experts') else '0',
           'FLASH_NEXT_EXPERT_CHECK_RESULT':str(job/'expert-offload-check.json'),'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    # Kaggle can stop accepting writes to the root overlay while its working
    # volume remains writable. Keep compiler caches and IPC temporary files on
    # that volume; checkpoint/model sources are still read from their pinned paths.
    cache_root = Path('/kaggle/working/.flash-next-runtime')
    for key, leaf in {'TMPDIR':'tmp','TRITON_CACHE_DIR':'triton','XDG_CACHE_HOME':'cache',
                      'CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'extensions'}.items():
        destination = cache_root/leaf
        destination.mkdir(parents=True,exist_ok=True)
        env[key] = str(destination)
    configure_fla_environment(env,args.mode,config.get('fla_numeric_profile'),root)
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
    if (job/'output').is_symlink() and not (job/'output').exists():(job/'output').unlink()
    (job/'monitor.json').write_text(json.dumps(dict(returncode=code,timed_out=timeout,seconds=time.monotonic()-started,pid=child.pid),indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        # Prelaunch failures must also finish the watch protocol.
        if '--config' in sys.argv:
            job = Path(sys.argv[sys.argv.index('--config')+1]).resolve().parent
            if not (job/'monitor.json').exists():
                (job/'monitor.json').write_text(json.dumps(dict(returncode=1,timed_out=False,
                    prelaunch_failure=True,error_type=type(exc).__name__,error=str(exc)))+'\n')
        raise
