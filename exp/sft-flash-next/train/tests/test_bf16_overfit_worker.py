"""Learning mode must update fresh adapters and retain each step's resource data."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kaggle_overfit_worker import native_command,bootstrap_source,summarize,phase_for
CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/bf16-overfit-v8.json').read_text())

class LearningTests(unittest.TestCase):
    def test_fresh_learning_command_has_updates_and_no_diagnostic_adapter(self):
        command=native_command(CONFIG,'/tmp/bootstrap.py')
        self.assertNotIn('--adapter-state',command)
        self.assertNotIn('--first-pass-only',command)
        self.assertEqual(command[command.index('--max-steps')+1],'20')
        self.assertEqual(command[command.index('--learning-rate')+1],'0.0002')
        self.assertEqual(command[command.index('--target-ratio')+1],'0.05')

    def test_bootstrap_preserves_spawn_guard_and_skips_gradient_equality(self):
        source=bootstrap_source({**CONFIG,'remote_config':'/tmp/config.json'})
        compile(source,'<overfit>','exec')
        namespace={'__name__':'__mp_main__'};exec(source,namespace)
        self.assertNotIn('torch',namespace)
        self.assertIn('training=True',source)
        self.assertNotIn('compare_gradients(',source)
        self.assertIn('finalize_bf16(optimizer_updates=updates)',source)

    def test_detached_worker_has_its_shared_helper_in_launch_directory(self):
        source=(Path(__file__).resolve().parents[1]/'kaggle_overfit_worker.py').read_text()
        self.assertIn("(launch_dir/'kaggle_reference_worker.py').write_bytes(helper.read_bytes())",source)

    def test_repeated_events_keep_each_step_duration_and_resource_peak(self):
        marks=[];telemetry=[];allocator=[]
        for step,start in [(0,10),(1,100)]:
            for name,offset in [('forward_start',0),('loss',5),('backward_call_start',6),('backward_call_end',13),('optimizer_start',15),('optimizer_end',17)]:
                marks.append(dict(step=step,event=name,monotonic_ns=(start+offset)*10**9))
            telemetry.append(dict(step=step,phase='forward',gpu_used_mib=step+1,process_rss_bytes=step+2,host_used_bytes=step+3))
            allocator.append(dict(step=step,phase='forward',exact_peak_allocated_bytes=step+4,exact_peak_reserved_bytes=step+5))
        rows=summarize(marks,telemetry,allocator,120)['steps']
        self.assertEqual([row['forward_seconds'] for row in rows],[5,5])
        self.assertEqual([row['pure_backward_seconds'] for row in rows],[7,7])
        self.assertEqual([row['optimizer_seconds'] for row in rows],[2,2])
        self.assertEqual(rows[0]['phase_resource_samples']['forward']['peak_process_rss_bytes'],2)
        self.assertEqual(rows[1]['phase_resource_samples']['forward']['exact_cuda_peak_allocated_bytes'],5)
        self.assertEqual(phase_for(marks,116*10**9),'optimizer')

if __name__=='__main__':unittest.main()
