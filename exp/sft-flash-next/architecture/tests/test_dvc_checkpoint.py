"""Real local DVC publication in a temporary repository; no network or tensors."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    from dvc.repo import Repo
except ImportError:
    Repo = None

spec = importlib.util.spec_from_file_location('dvc_checkpoint',Path(__file__).resolve().parents[1]/'training/dvc_checkpoint.py')
publisher = importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)


@unittest.skipUnless(Repo,'Run this integration test with the DVC Python interpreter')
class DVCCheckpointTests(unittest.TestCase):
    def test_publishes_retires_old_versions_and_preserves_shared_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            repository=Path(directory)
            repo=Repo.init(str(repository),no_scm=True);repo.close()
            shared=repository/'.dvc/cache/protected';shared.parent.mkdir(parents=True,exist_ok=True)
            shared.write_text('unrelated cache must survive')
            run=repository/'training-run';run.mkdir()
            previous=None
            for update in (1,2):
                name=f'update-{update:08d}'
                incoming=run/'incoming'/name;incoming.mkdir(parents=True)
                data=f'adapter and AdamW state {update}'.encode()
                (incoming/'00000.pt').write_bytes(data)
                manifest=dict(version=1,checkpoint=name,progress=dict(updates=update),
                    shards=[dict(path='00000.pt',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))])
                manifest['sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                path=incoming/'manifest.json';path.write_text(json.dumps(manifest))
                result=publisher.publish(repository,run,path)
                current=run/result['checkpoint']
                self.assertTrue(current.with_suffix('.dvc').exists())
                self.assertTrue(shared.exists())
                if previous:self.assertFalse(previous.exists());self.assertFalse(previous.with_suffix('.dvc').exists())
                previous=current
                # Crash after publication but before acknowledgment is idempotent.
                self.assertEqual(publisher.publish(repository,run,current/'manifest.json'),result)
            self.assertEqual(json.loads((run/'latest.json').read_text())['updates'],2)
            with Repo(str(repository),config={'cache':{'dir':str(run/'dvc-cache'),'type':['hardlink']}}) as repo:
                self.assertEqual(repo.status(targets=[str(current)]),{})


if __name__=='__main__':unittest.main()
