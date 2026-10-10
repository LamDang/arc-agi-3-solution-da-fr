"""Strict trajectory reconstruction and immutable trajectory dataset access."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from dataset import digest, file_hash, read_json, load_processor
from trajectory_encode import encode_input, vision_hashes


def request_key(row):
    return f"{row['game']}_p0#{row['request_index']}"


def call_ids(message):
    return tuple(call['id'] for call in message.get('tool_calls', []))


def without_thinking(message):
    out = copy.deepcopy(message)
    if out.get('role') == 'assistant':
        out.pop('reasoning_content', None)
    return out


def reconstruct(rows, generated):
    """Require all requests from index zero and prove each history overlap.

    Historical assistant thinking is ignored ONLY while matching identity;
    executable code, call IDs, observations and all other fields must match.
    All actual assistant turns then receive their own generated source thinking.
    A trimmed request must begin with the same system message and a contiguous
    suffix of the preceding transcript. No bridging observations are invented.
    """
    if any(not isinstance(r.get('request_index'), int) or isinstance(r.get('request_index'), bool) or r['request_index'] < 0 for r in rows):
        raise ValueError('request_index must be a nonnegative nonboolean integer')
    rows = sorted(rows, key=lambda r: r['request_index'])
    if not rows:
        raise ValueError('Empty trajectory')
    game = rows[0]['game']
    indices = [r['request_index'] for r in rows]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
        raise ValueError('request_index must be a nonnegative nonboolean integer')
    if not rows[0].get('messages') or rows[0]['messages'][0].get('role') != 'system':
        raise ValueError('Trajectory must start with a system message')
    conditioning = {k: rows[0].get(k) for k in ('tools', 'chat_template_kwargs')}
    if any({k:r.get(k) for k in conditioning} != conditioning for r in rows):
        raise ValueError('Tools or chat_template_kwargs changed across requests')
    if indices != list(range(len(rows))):
        raise ValueError(f'{game}: missing/duplicate request indices; full corpus required')
    transcript, provenance, owned_ids = [], [], set()
    for row in rows:
        if row['game'] != game:
            raise ValueError('Mixed games in trajectory')
        messages = copy.deepcopy(row['messages'])
        if not messages or messages[-1].get('role') != 'assistant':
            raise ValueError('Request must end in its teacher assistant reply')
        source_key = request_key(row)
        thought = generated.get(source_key)
        if not thought or thought.get('status') != 'ok' or not isinstance(thought.get('thinking'), str) or not thought['thinking'].strip():
            raise ValueError(f'Missing successful generated thinking: {source_key}')
        if transcript:
            if messages[0] != transcript[0] or messages[0].get('role') != 'system':
                raise ValueError(f'{source_key}: system context changed')
            history = messages[1:-1]
            old = transcript[1:]
            # The overlap must include the immediately preceding assistant.
            candidates = []
            old_identity = [digest(without_thinking(m)) for m in old]
            history_identity = [digest(without_thinking(m)) for m in history]
            for size in range(min(len(old), len(history)), 0, -1):
                if old_identity[-size:] == history_identity[:size]:
                    candidates.append(size)
            if not candidates:
                raise ValueError(f'{source_key}: no verified chronological history overlap')
            overlap = candidates[0]
            additions = history[overlap:]
            if any(m.get('role') == 'assistant' for m in additions):
                raise ValueError(f'{source_key}: unowned assistant in new context')
            transcript.extend(additions)
        else:
            if any(m.get('role') == 'assistant' for m in messages[:-1]):
                raise ValueError(f'{source_key}: initial history already contains unowned assistant turns')
            transcript.extend(messages[:-1])
        final = messages[-1]
        ids = call_ids(final)
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids) or owned_ids.intersection(ids):
            raise ValueError(f'{source_key}: duplicate/invalid tool call ownership')
        if thought.get('ref') is not None and thought['ref'] not in ids:
            raise ValueError(f'{source_key}: generated thinking tool ref does not match owned reply')
        owned_ids.update(ids)
        final['reasoning_content'] = thought['thinking']
        transcript.append(final)
        provenance.append(dict(source_id=source_key, request_sha256=digest(row),
            generated_sha256=digest(thought), message_index=len(transcript)-1,
            tool_call_ids=list(call_ids(final))))
    out = copy.deepcopy(rows[-1])
    out.update(messages=transcript, trajectory_id=f'{game}_p0',
               thinking_source='generated-per-turn', source_turns=provenance)
    return out


def plan_token_updates(rows, token_budget):
    """Greedy ordered whole-trajectory updates; counts are exact, budgets soft."""
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget <= 0:
        raise ValueError('token_budget must be a positive integer')
    plans, pending, count, seen = [], [], 0, set()
    for row in rows:
        if row['sample_id'] in seen:
            raise ValueError('Duplicate trajectory in update plan')
        seen.add(row['sample_id'])
        n = row['target_tokens']
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError('Every trajectory needs a positive exact target_tokens count')
        pending.append(row['sample_id'])
        count += n
        if count >= token_budget:
            plans.append(dict(sample_ids=pending, target_tokens=count, token_budget=token_budget,
                              overshoot_tokens=max(0, count-token_budget), tail=False))
            pending, count = [], 0
    if pending:
        plans.append(dict(sample_ids=pending, target_tokens=count, token_budget=token_budget,
                          overshoot_tokens=0, tail=True))
    return plans


class Trajectories:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = read_json(self.root / 'manifest.json')
        if self.manifest.get('dataset_kind') != 'true-trajectories':
            raise ValueError('Not a trajectory manifest')
        if digest({k:v for k,v in self.manifest.items() if k != 'sha256'}) != self.manifest['sha256']:
            raise ValueError('Trajectory manifest checksum mismatch')
        if self.manifest.get('source_completeness') != 'complete original logs with verified game completion':
            raise ValueError('Production trajectories require complete original logs with verified game completion')
        for name, checksum in self.manifest['files'].items():
            if file_hash(self.root / name) != checksum:
                raise ValueError(f'Dataset file changed: {name}')
        self.processor = load_processor(self.root / 'processor')
        self.rows = self.manifest['rows']

    def get(self, row):
        with open(self.root / 'trajectories.jsonl', 'rb') as stream:
            stream.seek(row['offset'])
            raw = stream.read(row['length'])
        if hashlib.sha256(raw).hexdigest() != row['line_sha256']:
            raise ValueError('Indexed trajectory checksum mismatch')
        sample = json.loads(raw)
        if sample['trajectory_id'] != row['sample_id']:
            raise ValueError('Indexed trajectory identity mismatch')
        annotation = read_json(self.root / row['annotation_file'])
        enc = encode_input(self.processor, sample)
        ids = enc['input_ids'][0].tolist()
        if digest(ids) != annotation['input_sha256'] or [ids[p] for p in annotation['positions']] != annotation['target_ids']:
            raise ValueError('Trajectory processor input changed since preparation')
        if vision_hashes(enc) != annotation.get('vision_sha256'):
            raise ValueError('Trajectory vision processor output changed since preparation')
        if digest(annotation) != row['annotation_sha256']:
            raise ValueError('Trajectory processor output changed since preparation')
        return enc, annotation

    def split(self, name):
        return [r for r in self.rows if r['split'] == name]
