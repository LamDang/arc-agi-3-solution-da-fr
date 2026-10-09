import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from runtime.evidence import retain_raw_gradients


class RetentionTests(unittest.TestCase):
    def test_only_reference_test_retains_raw_gradients(self):
        for architecture in ('reference','optimized','native'):
            for mode in ('test','train'):
                self.assertEqual(retain_raw_gradients(SimpleNamespace(architecture=architecture),mode),
                                 architecture == 'reference' and mode == 'test')


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch available in pinned runtime')
class EvidenceTests(unittest.TestCase):
    def test_sharded_baseline_and_changed_inventory(self):
        import torch
        from runtime.evidence import compare,compare_initial,sha
        state={str(i):torch.tensor([1.125,.123456789]) for i in range(3)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for index,shard in enumerate([{'0':state['0']},{k:v for k,v in state.items() if k!='0'}]):
                path=root/f'{index}.pt';torch.save(shard,path)
                rows.append(dict(path=path.name,sha256=sha(path)))
            path=root/'manifest.json';path.write_text(json.dumps(dict(tensors=3,shards=rows)))
            baseline=dict(loss=.5,gradients=str(path))
            result=compare(state,.5,baseline)
            self.assertTrue(result['passed']);self.assertEqual(result['bitwise_equal_tensors'],3)
            baseline['initial_adapter']=str(path)
            self.assertTrue(compare_initial(state,baseline)['passed'])
            altered={k:v.clone() for k,v in state.items()};altered['0'][0]+=1
            self.assertFalse(compare_initial(altered,baseline)['passed'])
            changed=compare({k:v.bfloat16() for k,v in state.items()},.5,baseline)
            self.assertFalse(changed['passed']);self.assertGreater(changed['global_relative_l2'],0)
            rows[0]['sha256']='0'*64;path.write_text(json.dumps(dict(tensors=3,shards=rows)))
            with self.assertRaisesRegex(ValueError,'hash differs'):compare(state,.5,baseline)
