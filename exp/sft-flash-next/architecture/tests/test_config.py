import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config, Optimizations,load_config,reference_options


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
    def test_new_reference_preset_and_adapter_inventory(self):
        c=load_config(Path(__file__).resolve().parents[1]/'configs/reference.json','test')
        self.assertEqual(c.optimizations,reference_options())
        self.assertEqual(c.adapter_tensors,74472)
        self.assertFalse(c.optimizations.bf16_lora)
        self.assertTrue(c.diagnostic_initialization)
        with self.assertRaises(ValueError):c.validate('train')

    def test_benchmark_requires_exact_length_labels_and_initial_pin(self):
        from dataclasses import replace
        c=self.config(benchmark_tokens=32000,baseline={'initial_adapter':'/reference.json'},
                      diagnostic_initialization=True)
        c.validate('benchmark')
        for bad in [replace(c,benchmark_tokens=None),replace(c,prompt_tokens=10),
                    replace(c,baseline=None),replace(c,benchmark_repeats=0)]:
            with self.assertRaises(ValueError):bad.validate('benchmark')
        with self.assertRaises(ValueError):c.validate('train')

if __name__ == '__main__':unittest.main()
