"""CPU-only checks for the observational native capture worker."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaggle_reference_worker import bootstrap_source, native_command, phase_for, summarize


CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/reference-v0.json').read_text())


class WorkerTests(unittest.TestCase):
    def test_opt3_keeps_cce_and_adds_mask_checks_before_capture(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/cce-opt3-mask-v4.json').read_text())
        source = bootstrap_source(config)
        compile(source, '<opt3-bootstrap>', 'exec')
        self.assertIn('from cce_target_loss import install_for_capture', source)
        self.assertIn('from opt3_mask_capture import install_for_capture as install_mask', source)
        self.assertIn('from opt3_mask_operator_check import qualify as qualify_mask', source)
        self.assertIn(config['native_gradients'], source)
        self.assertLess(source.index('qualify_mask('), source.index('runpy.run_path('))
        self.assertNotIn('optimizer.step', source)

    def test_cce_bootstrap_uses_isolated_source_and_saves_gradient_comparison(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/cce-exact-v3.json').read_text())
        source = bootstrap_source(config)
        compile(source, '<cce-bootstrap>', 'exec')
        self.assertIn('from cce_operator_check import qualify', source)
        self.assertIn('from cce_target_loss import compare_gradients', source)
        self.assertIn(config['cce_dependency']['archive_prefix'], source)
        self.assertNotIn('optimizer.step', source)

    def test_liger_bootstrap_records_differences_without_an_optimizer(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/liger-flce-v2.json').read_text())
        source = bootstrap_source(config)
        compile(source, '<liger-bootstrap>', 'exec')
        self.assertIn('from liger_operator_check import qualify', source)
        self.assertIn('from liger_target_loss import compare_gradients', source)
        self.assertIn('liger-runtime', source)
        self.assertNotIn('optimizer.step', source)

    def test_generated_bootstrap_preserves_native_script_and_compiles(self):
        source = bootstrap_source(CONFIG)
        compile(source, '<instrumented-bootstrap>', 'exec')
        self.assertIn("runpy.run_path('/kaggle/working/bootstrap/reference-one-run-bootstrap.py'", source)
        self.assertIn('original_backward(self, *args, **kwargs)', source)
        self.assertIn("allocator_peak('pure_backward')", source)
        command = native_command(CONFIG, '/tmp/instrumented-bootstrap.py')
        self.assertEqual(command.count('--first-pass-only'), 1)
        self.assertNotIn('--max-steps', command)
        self.assertNotIn('--learning-rate', command)
        self.assertEqual(command[-2:], ['--out', CONFIG['remote_output']])

    def test_phase_marks_use_absolute_monotonic_clock(self):
        marks = [{'event': event, 'monotonic_ns': stamp} for event, stamp in [
            ('load_start', 100), ('load_complete', 200), ('forward_start', 220),
            ('loss', 300), ('backward_start', 400), ('backward_call_end', 500),
            ('gradients_saved', 600)]]
        self.assertEqual([phase_for(marks, stamp) for stamp in [50, 150, 210, 250, 350, 450, 550, 650]],
                         ['startup', 'loading', 'forward_setup', 'forward', 'pre_backward_export',
                          'pure_backward', 'gradient_export', 'finished'])

    def test_separate_phase_timings_and_memory(self):
        marks = [{'event': event, 'monotonic_ns': stamp} for event, stamp in [
            ('load_start', 1_000_000_000), ('load_complete', 3_000_000_000),
            ('forward_start', 3_500_000_000), ('loss', 5_000_000_000),
            ('backward_start', 6_000_000_000),
            ('backward_call_start', 6_100_000_000),
            ('backward_call_end', 9_100_000_000),
            ('gradients_saved', 10_000_000_000)]]
        telemetry = [{'phase': 'forward', 'gpu_used_mib': 42,
                      'process_rss_bytes': 100, 'host_used_bytes': 200},
                     {'phase': 'forward', 'gpu_used_mib': 43,
                      'process_rss_bytes': 120, 'host_used_bytes': 210}]
        allocator = [{'phase': 'forward', 'exact_peak_allocated_bytes': 60,
                      'exact_peak_reserved_bytes': 70}]
        result = summarize(marks, telemetry, allocator, 11.0)
        self.assertEqual(result['loading_seconds'], 2)
        self.assertEqual(result['forward_setup_seconds'], .5)
        self.assertEqual(result['forward_seconds'], 1.5)
        self.assertEqual(result['pre_backward_export_seconds'], 1)
        self.assertEqual(result['pure_backward_seconds'], 3)
        self.assertEqual(result['gradient_export_seconds'], .9)
        self.assertEqual(result['phase_resource_samples']['forward']['peak_process_rss_bytes'], 120)
        self.assertEqual(result['phase_resource_samples']['forward']['peak_gpu_used_mib'], 43)
        self.assertEqual(result['phase_resource_samples']['forward']['exact_cuda_peak_reserved_bytes'], 70)


if __name__ == '__main__':
    unittest.main()
