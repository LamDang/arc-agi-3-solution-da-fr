import copy
from types import SimpleNamespace

import pytest

from inference.utils.openai_compat import responses_input_from_messages
from think_gen import client, progressive as p


def messages(turns=8):
    rows = [{'role': 'system', 'content': 'stable instructions'}]
    for i in range(turns):
        rows += [{'role': 'assistant', 'content': f'thinking {i}', 'tool_calls': [
            {'id': f'c{i}', 'function': {'name': 'python', 'arguments': '{"code":"inspect"}'}}]},
            {'role': 'tool', 'tool_call_id': f'c{i}', 'content': [
                {'type': 'text', 'text': f'result {i}'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGVsbG8='}}]},
            {'role': 'user', 'content': f'board {i}'}]
    return rows + [{'role': 'user', 'content': 'changing judge payload'}]


def markers(items):
    return [(i, j) for i, item in enumerate(items)
            for j, block in enumerate(item.get('content', item.get('output', [])))
            if isinstance(block, dict) and 'prompt_cache_breakpoint' in block]


def strip_markers(items):
    rows = copy.deepcopy(items)
    for item in rows:
        blocks = item.get('content', item.get('output', []))
        if isinstance(blocks, str):
            item['content'] = [{'type': 'input_text', 'text': blocks}]
            continue
        for block in blocks:
            block.pop('prompt_cache_breakpoint', None)
    return rows


def test_boundaries_include_images_exclude_judge_preserve_all_evidence():
    msgs = messages()
    original = copy.deepcopy(msgs)
    items = client.explicit_history_input(msgs)
    indices = markers(items)
    assert len(indices) == 4
    assert all(i < len(items) - 1 for i, _ in indices)
    assert any(items[i].get('output', [{}])[j].get('type') == 'input_image'
               for i, j in indices if 'output' in items[i])
    assert strip_markers(items) == strip_markers(responses_input_from_messages(msgs))
    assert msgs == original


def test_previous_cached_endpoint_is_kept_after_history_grows():
    old = client.explicit_history_input(messages(8))
    new = client.explicit_history_input(messages(9))
    i, j = markers(old)[-1]
    assert (i, j) in markers(new)
    assert strip_markers(old[:i+1]) == strip_markers(new[:i+1])


def test_payload_explicit_mode_and_30m_ttl(monkeypatch):
    seen = {}
    def capture(url, **kwargs):
        seen.update(kwargs['json'])
        raise RuntimeError('captured before sending')
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test')
    monkeypatch.setattr(client.requests, 'post', capture)
    with pytest.raises(RuntimeError, match='captured'):
        client.openai_responses(messages(), cache_policy='explicit-history-v1', cache_key='same-game')
    assert seen['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert seen['prompt_cache_key'] == 'same-game'
    assert 'prompt_cache_retention' not in seen
    assert len(markers(seen['input'])) == 4


def test_write_pricing_and_long_context_tier(tmp_path):
    calls = p.DurableCalls(tmp_path, SimpleNamespace(sol_prices=[2, .1, 10]))
    usage = {'input_tokens': 1000, 'output_tokens': 100,
             'input_tokens_details': {'cached_tokens': 600, 'cache_write_tokens': 100}}
    assert calls.price({'usage': usage}, 'sol') == pytest.approx(.00191)
    usage.update(input_tokens=300000, input_tokens_details={'cached_tokens': 100000, 'cache_write_tokens': 150000})
    assert calls.price({'usage': usage}, 'sol') == pytest.approx(.9715)
    usage['input_tokens_details']['cache_write_tokens'] = 300000
    with pytest.raises(ValueError, match='Invalid cache'):
        calls.price({'usage': usage}, 'sol')


def test_graceful_stop_before_send_has_safe_same_slot_resume(tmp_path, monkeypatch):
    from test_progressive import args, response
    a = args(tmp_path)
    calls = p.DurableCalls(a.out, a)
    original = p.atomic_json
    def stop_after_reservation(path, row):
        original(path, row)
        if path.name.startswith('attempt-') and row['state'] == 'reserved':
            calls.stop.set()
    monkeypatch.setattr(p, 'atomic_json', stop_after_reservation)
    monkeypatch.setattr(p.client, 'openai_responses', lambda *_a, **_k: pytest.fail('Unsent request'))
    folder = a.out/'turns/game/00000'
    with pytest.raises(RuntimeError, match='stopped'):
        calls.call(folder, 'judge', 'sol', [], [], p.validate_thinking)
    attempt = folder/'calls/judge/attempt-00.json'
    assert p.read_json(attempt)['state'] == 'cancelled_unsent'
    monkeypatch.setattr(p, 'atomic_json', original)
    sent = []
    monkeypatch.setattr(p.client, 'openai_responses', lambda *_a, **_k: sent.append(1) or response('valid'))
    resumed = p.DurableCalls(a.out, a)
    assert resumed.call(folder, 'judge', 'sol', [], [], p.validate_thinking) == 'valid'
    assert sent == [1]
    assert len(list(attempt.parent.glob('attempt-*.json'))) == 1
