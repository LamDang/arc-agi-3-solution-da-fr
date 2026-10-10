import json
import hashlib

import pytest
import torch

from make_nonzero_diagnostic_adapter import (
    derive_state, tensor_manifest, tensor_state_sha256, validate_zero_b_anchor,
)
from trajectory_training import validate_clean_initial_adapter


def small_anchor():
    return {f'layer{i}.lora_{kind}.weight':
            (torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1
             if kind == 'A' else torch.zeros(2, 3))
            for i in range(2) for kind in ('A', 'B')}


def test_diagnostic_derivation_replays_exactly_and_keeps_a():
    anchor = small_anchor()
    validate_zero_b_anchor(anchor, expected_tensors=4)
    first = derive_state(anchor, seed=20261010, std=0.003)
    replay = derive_state(anchor, seed=20261010, std=0.003)
    changed_seed = derive_state(anchor, seed=20261011, std=0.003)
    assert tensor_manifest(first) == tensor_manifest(replay)
    assert tensor_state_sha256(first) == tensor_state_sha256(replay)
    assert tensor_state_sha256(first) != tensor_state_sha256(changed_seed)
    assert all(torch.equal(first[name], anchor[name]) for name in anchor if 'lora_A' in name)
    assert all(torch.count_nonzero(first[name]) for name in anchor if 'lora_B' in name)
    assert any(not torch.equal(first[name], changed_seed[name]) for name in anchor if 'lora_B' in name)
    assert all(not torch.count_nonzero(anchor[name]) for name in anchor if 'lora_B' in name)


def test_diagnostic_rejects_modified_anchor():
    anchor = small_anchor()
    anchor['layer0.lora_B.weight'][0, 0] = 1
    with pytest.raises(ValueError, match='not zero'):
        validate_zero_b_anchor(anchor, expected_tensors=4)


def test_production_validator_rejects_diagnostic_provenance_and_nonzero_b(tmp_path):
    anchor = {f'layer{i}.lora_{kind}.weight':
              (torch.ones(2, 3) if kind == 'A' else torch.zeros(2, 3))
              for i in range(372) for kind in ('A', 'B')}
    diagnostic = derive_state(anchor, seed=20261010, std=0.003)
    state_path = tmp_path / 'adapter.pt'
    provenance_path = tmp_path / 'manifest.json'
    torch.save(diagnostic, state_path)
    file_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    provenance = {'purpose': 'diagnostic-gradient-qualification-only',
                  'validation_exposed': True, 'optimizer_updates': 0,
                  'adapter_sha256': file_hash}
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='provenance is not clean'):
        validate_clean_initial_adapter(state_path, provenance_path)

    # Even a forged clean provenance cannot admit the nonzero B matrices.
    provenance['purpose'] = 'clean-production-initialization'
    provenance['validation_exposed'] = False
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='zero B'):
        validate_clean_initial_adapter(state_path, provenance_path)
