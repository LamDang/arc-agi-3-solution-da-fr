"""Mock model -> bounded copy-back -> real DVC -> crash/resume -> same result."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch

from test_training import Tiny, samples, loss_fn, evaluate
from training import checkpoint
from training.engine import fit


@unittest.skipUnless(os.environ.get('DVC_PYTHON'), 'Set DVC_PYTHON for real DVC end-to-end recovery')
class TrainingDVCEndToEnd(unittest.TestCase):
    def test_train_publish_crash_resume_and_retire_previous_checkpoint(self):
        torch.manual_seed(8)
        control=Tiny();initial=copy.deepcopy(control.state_dict())
        optimizer=torch.optim.AdamW(control.parameters(),lr=1e-4,weight_decay=0,foreach=False)
        data=samples()
        def run(model,opt,save,progress=None):
            return fit(model,opt,list(range(4)),2,6,17,
                lambda order,start:(data[i] for i in order[start:]),lambda s:loss_fn(model,s),
                lambda:evaluate(model),save,lambda *a:None,progress=progress,max_grad_norm=.25)
        expected=run(control,optimizer,lambda p:None)
        python=os.environ['DVC_PYTHON']
        helper=Path(__file__).resolve().parents[1]/'training/dvc_checkpoint.py'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run_root=root/'run';spool=root/'server-spool'
            subprocess.run([python,'-c','from dvc.repo import Repo; import sys; Repo.init(sys.argv[1],no_scm=True).close()',str(root)],
                           check=True,capture_output=True)
            def transfer(name,path,kind):
                incoming=run_root/'incoming'/name;incoming.mkdir(parents=True,exist_ok=True)
                copied=incoming/path.name;shutil.copyfile(path,copied)
                with copied.open('rb') as stream:os.fsync(stream.fileno())
                self.assertEqual(checkpoint.file_hash(path),checkpoint.file_hash(copied))
                if kind=='manifest':
                    result=subprocess.run([python,str(helper),'--repository',str(root),'--run',str(run_root),'--manifest',str(copied)],
                                          capture_output=True,text=True)
                    if result.returncode:raise RuntimeError(result.stderr)
                path.unlink()
            interrupted=Tiny();interrupted.load_state_dict(initial)
            first=torch.optim.AdamW(interrupted.parameters(),lr=1e-4,weight_decay=0,foreach=False)
            def stop(progress):
                checkpoint.save(interrupted,first,spool,progress,{'mock':'fixed'},6000,transfer)
                raise RuntimeError('kernel crashed')
            with self.assertRaisesRegex(RuntimeError,'kernel crashed'):run(interrupted,first,stop)
            latest=json.loads((run_root/'latest.json').read_text())
            resumed=Tiny();second=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=0,foreach=False)
            progress=checkpoint.load(resumed,second,run_root/latest['checkpoint'],{'mock':'fixed'})
            def save(progress):checkpoint.save(resumed,second,spool,progress,{'mock':'fixed'},6000,transfer)
            actual=run(resumed,second,save,progress)
            self.assertEqual(actual,expected)
            for p,q in zip(control.parameters(),resumed.parameters()):self.assertTrue(torch.equal(p,q))
            for a,b in zip(optimizer.state.values(),second.state.values()):
                for key in a:self.assertTrue(torch.equal(a[key],b[key]))
            latest=json.loads((run_root/'latest.json').read_text())
            self.assertEqual(latest['updates'],expected['updates'])
            self.assertEqual(len([p for p in (run_root/'checkpoints').iterdir() if p.is_dir()]),1)
            self.assertTrue((root/latest['dvc_pointer']).exists())


if __name__=='__main__':unittest.main()
