"""Independent checks shared by the collected-run reviewers."""
import hashlib
import json


def numerical_profile(job, config):
    if config.get('fla_numeric_profile') is None:
        return None
    assert config['fla_numeric_profile'] == 'reference-v0'
    profile = json.loads((job/'output/numeric-profile.json').read_text())
    path = job/'source/configs/kernels-reference-v0/chunk_local_cumsum_scalar_kernel.json'
    stored = json.loads(path.read_text())
    key = [1, 48, 64, False, True, 'torch.float32', 'torch.float32']
    selected = dict(num_warps=1, num_ctas=1, num_stages=3)
    assert stored['kernel_name'] == 'chunk_local_cumsum_scalar_kernel'
    assert len(stored['autotune_entries']) == 1
    entry = next(iter(stored['autotune_entries'].values()))
    assert entry == dict(autotune_key=key, config=dict(kwargs={}, **selected))
    assert profile['profile'] == 'reference-v0' and profile['backward_verified']
    assert profile['autotune_key'] == key
    assert profile['observed_configuration'] == profile['required_configuration'] == selected
    assert not profile['package_sources_modified']
    assert profile['cumsum_sha256'] == '0405701c46cee331088bfee395b3bf37f8384829dace0aa3a55a509844bcf4cb'
    assert profile['config_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    return profile
