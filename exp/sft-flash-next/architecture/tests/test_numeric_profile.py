import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and os.environ.get('FLA_CACHE_MODE')=='strict',
                     'Enabled for pinned FLA profile GPU jobs')
class NumericProfileTests(unittest.TestCase):
    def test_native_config_file_selects_reference_reverse_scan(self):
        import torch
        import fla.ops.utils.cumsum as cumsum
        from runtime.numeric_profile import verify_profile,KEY
        config=SimpleNamespace(fla_numeric_profile='reference-v0')
        auto=cumsum.chunk_local_cumsum_scalar_kernel.fn
        previous=dict(auto.cache)
        try:
            with tempfile.TemporaryDirectory() as out:
                verify_profile(config,Path(out))
                value=torch.zeros((1,65,48),device='cuda',dtype=torch.float32)
                result=cumsum.chunk_local_cumsum(value,chunk_size=64,reverse=True)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(result,torch.zeros_like(result)))
                verify_profile(config,Path(out),required=True)
                self.assertEqual(auto.cache[KEY].num_warps,1)
        finally:
            auto.cache.clear();auto.cache.update(previous)
