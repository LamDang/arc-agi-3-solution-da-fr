import unittest
from runtime.fla_environment import configure_fla_environment


class FLAEnvironmentTests(unittest.TestCase):
    def test_real_training_clears_inherited_test_pin(self):
        env=dict(FLA_CACHE_MODE='strict',FLA_CONFIG_DIR='/old/test-profile',OTHER='preserved')
        configure_fla_environment(env,'train',None,'/source')
        self.assertEqual(env,dict(FLA_CACHE_MODE='disabled',OTHER='preserved'))

    def test_training_profile_rejected_before_environment_changes(self):
        env={}
        with self.assertRaisesRegex(ValueError,'test-only'):
            configure_fla_environment(env,'train','reference-v0','/source')
        self.assertEqual(env,{})

    def test_test_and_capacity_benchmarks_can_pin_reference(self):
        for mode in ['test','benchmark']:
            env={}
            configure_fla_environment(env,mode,'reference-v0','/source')
            self.assertEqual(env,dict(FLA_CACHE_MODE='strict',FLA_CONFIG_DIR='/source/configs/kernels-reference-v0'))
