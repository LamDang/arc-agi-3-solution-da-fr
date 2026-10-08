import copy
import json

import pytest

from prepare_variant import CODE_ONLY_LINE, audit, source_order
from test_data import sample


def pair():
    original = sample()
    original['game'], original['request_index'] = 'game', 1
    original['messages'][0]['content'] = CODE_ONLY_LINE
    original['tools'][0]['function']['parameters'] = {
        'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}
    generated = copy.deepcopy(original)
    generated['messages'][-1]['reasoning_content'] = 'First inspect, then test the move.'
    generated['thinking_source'] = 'think_gen-refine2'
    return original, generated


def test_variant_accepts_only_changed_final_thinking():
    original, generated = pair()
    assert audit(generated, original) == {'history_calls': 0, 'final_calls': 1, 'python_schemas': 1}
    assert original['messages'][-1]['reasoning_content'] != generated['messages'][-1]['reasoning_content']


@pytest.mark.parametrize('mutation', ['history', 'code', 'image', 'schema', 'system', 'empty_thinking'])
def test_variant_rejects_unaccounted_changes(mutation):
    original, generated = pair()
    if mutation == 'history':
        generated['messages'][1]['reasoning_content'] = 'Changed context'
    elif mutation == 'code':
        generated['messages'][-1]['tool_calls'][0]['function']['arguments']['code'] = 'print(2)'
    elif mutation == 'image':
        generated['messages'][2]['content'].pop()
    elif mutation == 'schema':
        generated['tools'][0]['function']['parameters']['properties']['description'] = {'type': 'string'}
    elif mutation == 'system':
        generated['messages'][0]['content'] = 'Python takes reasoning, description and code.'
    else:
        generated['messages'][-1]['reasoning_content'] = ''
    with pytest.raises(ValueError):
        audit(generated, original)


def test_variant_rejects_sol_fields_even_when_both_panels_have_them():
    original, generated = pair()
    for obj in (original, generated):
        obj['messages'][1]['tool_calls'] = [{'function': {
            'name': 'python', 'arguments': {'code': 'print(1)', 'description': 'Sol-only'}}}]
    with pytest.raises(ValueError, match='extra fields'):
        audit(generated, original)


def test_render_sensitive_tool_key_order_is_restored_without_changing_values():
    original, generated = pair()
    generated = json.loads(json.dumps(generated, sort_keys=True))
    assert json.dumps(original['tools']) != json.dumps(generated['tools'])
    assert audit(generated, original)['final_calls'] == 1
    ordered = source_order(generated, original)
    assert ordered == generated
    assert json.dumps(ordered['tools']) == json.dumps(original['tools'])
    assert ordered['messages'][-1]['reasoning_content'] == generated['messages'][-1]['reasoning_content']
