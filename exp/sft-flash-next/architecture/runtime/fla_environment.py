"""The reference reduction pin is exclusively for numerical testing."""
from pathlib import Path


def configure_fla_environment(env, mode, profile, source):
    if mode == 'train':
        if profile is not None:
            raise ValueError('FLA numerical profiles are test-only; real training uses native autotuning')
        # Do not inherit a diagnostic profile from the launching process.
        env.pop('FLA_CONFIG_DIR', None)
        env['FLA_CACHE_MODE'] = 'disabled'
    elif profile is not None:
        if mode not in {'test', 'benchmark'} or profile != 'reference-v0':
            raise ValueError('Unsupported test-only FLA profile')
        env['FLA_CACHE_MODE'] = 'strict'
        env['FLA_CONFIG_DIR'] = str(Path(source)/'configs/kernels-reference-v0')
