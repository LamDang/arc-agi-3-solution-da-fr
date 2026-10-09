"""Independent identities and raw-evidence gates for multi-span production."""
import json
import importlib.metadata
from pathlib import Path
import torch
from dataset import file_hash
from native_gradient_compare import compare, OPERATOR_SOURCES

TRAJECTORY_SOURCES = (*OPERATOR_SOURCES, 'trajectory_head_loss.py',
                      'trajectory_training.py', 'trajectory_checkpoints.py',
                      'trajectory_run.py', 'trajectory_encode.py', 'trajectory_data.py',
                      'trajectory_gradient_gate.py', 'trajectory_gradient_audit.py',
                      'dataset.py', 'run.py', 'native_gradient_compare.py', 'overfit_hf_reference.py',
                      'trajectory_initialize.py')


def source_hashes():
    root = Path(__file__).parent
    return {name:file_hash(root/name) for name in TRAJECTORY_SOURCES}


def verify_gradient_gate(directory, flags, backward_rows):
    """Recheck raw native repeats/candidate, hashes, and nonzero adapter state."""
    root = Path(directory)
    identity = json.loads((root/'identity.json').read_text())
    if identity.get('objective') != 'native-all-assistant-summed-ce-v1':
        raise ValueError('A final-reply gradient audit cannot qualify multi-span production')
    if identity['source_sha256'] != source_hashes():
        raise ValueError('Trajectory operator source identity changed')
    if identity['flags'] != flags or identity['head_backward_rows'] != backward_rows:
        raise ValueError('Different trajectory operator recipe')
    if not identity['deterministic']:
        raise ValueError('Require deterministic native reference')
    current = {n:importlib.metadata.version(n) for n in identity['runtime']}
    if current != identity['runtime'] or identity['cuda'] != torch.version.cuda:
        raise ValueError('Different native trajectory runtime')
    if identity['gpu'] != torch.cuda.get_device_name(0) or any(identity[key] != getattr(torch.backends.cuda.matmul,key)
            for key in ['allow_bf16_reduced_precision_reduction','allow_tf32']):
        raise ValueError('Different GPU/backend math settings')
    for name,checksum in identity['artifacts'].items():
        if Path(name).name != name or file_hash(root/name) != checksum:
            raise ValueError('Changed trajectory gradient evidence: '+name)
    reference = torch.load(root/'native-1.pt', map_location='cpu', weights_only=True)
    for name in ['native-2.pt', 'candidate.pt']:
        comparison = compare(torch.load(root/name,map_location='cpu',weights_only=True), reference)
        if comparison['total_tensors'] != 744 or not comparison['bitwise_equal']:
            raise ValueError('Require bitwise native repeat and all 744 candidate gradients')
    state = torch.load(root/'adapter.pt', map_location='cpu', weights_only=True)
    if len(state) != 744:
        raise ValueError('Incomplete adapter qualification state')
    counts = {key:sum(bool(torch.count_nonzero(value)) for name,value in state.items()
                     if key in name) for key in ['lora_A', 'lora_B']}
    if counts != {'lora_A':372,'lora_B':372}:
        raise ValueError('Production gate requires nonzero A and B adapters')
    result = json.loads((root/'results.json').read_text())
    if any(case['loss'] != result[0]['loss'] for case in result):
        raise ValueError('Multi-span scalar loss differs from native')
    return identity
