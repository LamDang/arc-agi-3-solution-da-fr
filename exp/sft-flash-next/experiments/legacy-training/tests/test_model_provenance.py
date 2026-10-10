import copy
from types import SimpleNamespace

import pytest

from dataset import file_hash, write_json
from run import check_model


def test_source_model_accepts_only_train_clean_expert_map(tmp_path):
    model, data = tmp_path/'model', tmp_path/'data'
    model.mkdir(); data.mkdir(); (data/'processor').mkdir()
    write_json(data/'folds.json', {'folds': [
        {'fold': 0, 'game_ids': ['held-v1']}, {'fold': 1, 'game_ids': ['train-v1']}]})
    config = dict(model_type='qwen4_exp', text_config=dict(num_experts=512, num_hidden_layers=2),
                  quantization_config=dict(bits=4, sym=True, packing_format='auto_round:auto_gptq'))
    write_json(model/'config.json', config)
    write_json(model/'model.safetensors.index.json', {'weight_map': {'layer': 'weights.safetensors'}})
    (model/'weights.safetensors').write_bytes(b'fixture weights')
    keep = dict(validation_exposed=False, calibration_provenance={'run': {'game': 'train'}},
                kept={str(i): list(range(256)) for i in range(2)})
    map_path = tmp_path/'keep.json'; write_json(map_path, keep)
    bundle = SimpleNamespace(root=data, manifest={'validation_fold': 0})
    hashes = check_model(model, bundle, map_path)
    assert hashes['expert_map'] == file_hash(map_path)
    assert hashes['weights.safetensors'] == file_hash(model/'weights.safetensors')
    bad = copy.deepcopy(keep); bad['calibration_provenance']['run']['game'] = 'held'
    write_json(map_path, bad)
    with pytest.raises(ValueError, match='held-out'):
        check_model(model, bundle, map_path)
    bad = copy.deepcopy(keep); bad['validation_exposed'] = True
    write_json(map_path, bad)
    with pytest.raises(ValueError, match='train-only'):
        check_model(model, bundle, map_path)
    config['text_config']['num_experts'] = 256
    write_json(model/'config.json', config)
    write_json(model/'keep.json', keep)
    bad = copy.deepcopy(keep); bad['kept']['0'] = list(range(1,257))
    write_json(map_path, bad)
    with pytest.raises(ValueError, match='own expert map'):
        check_model(model, bundle, map_path)
