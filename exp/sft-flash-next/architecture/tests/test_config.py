import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config, Optimizations


class ConfigTests(unittest.TestCase):
    def config(self, **kwargs):
        return Config(architecture='optimized',model='/model',samples=['/sample.pt'],output='/output',optimizations=Optimizations(),**kwargs)
    def test_reference_rejects_flags(self):
        with self.assertRaises(ValueError):
            Config('reference','/model',['/sample'],'/out',Optimizations(bf16_lora=True)).validate('test')
    def test_training_rejects_diagnostic_adapter(self):
        with self.assertRaises(ValueError):self.config(adapter='/trained-diagnostic.pt').validate('train')
    def test_test_is_one_complete_sample(self):
        with self.assertRaises(ValueError):
            Config('optimized','/model',['/a','/b'],'/out',Optimizations()).validate('test')
    def test_flags_have_explicit_types(self):
        with self.assertRaises(ValueError):Optimizations(disk_ple='false').validate()
    def test_unknown_head_rejected(self):
        with self.assertRaises(ValueError):Optimizations(head='approximate').validate()

if __name__ == '__main__':unittest.main()
