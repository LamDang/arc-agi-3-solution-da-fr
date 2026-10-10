"""Indexed Sol25 compaction trajectories with explicit, nonduplicated targets."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_processor(path):
    from transformers import AutoProcessor, AutoImageProcessor
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, backend='pil')
    return processor


def messages_for_processor(messages):
    from PIL import Image
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message.get('content'), list):
            continue
        parts = []
        for part in message['content']:
            if part['type'] == 'text':
                parts.append(part)
            elif part['type'] == 'image_url':
                url = part['image_url']['url']
                if not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Expected embedded image')
                with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1], validate=True))) as image:
                    parts.append(dict(type='image', image=image.convert('RGB')))
            else:
                raise ValueError('Unexpected content type')
        message['content'] = parts
    return result


def encode(processor, row, max_tokens=130000):
    """One vision pass, verified text/token prefixes for every owned assistant.

    Image expansion is outside assistant spans. The unchanged processor output
    supplies input tensors; text-render offsets identify its target suffixes.
    Five-token generation prefixes remain context, as in the dataset export.
    """
    if row.get('chat_template_kwargs', {}).get('preserve_thinking') is not True:
        raise ValueError('preserve_thinking must be true')
    messages = messages_for_processor(row['messages'])
    kwargs = dict(tools=row.get('tools') or None, **row['chat_template_kwargs'])
    def render(ms, generation=False):
        return processor.apply_chat_template(ms, tokenize=False, add_generation_prompt=generation, **kwargs)
    full = render(messages)
    enc = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt', processor_kwargs={'add_special_tokens': False}, **kwargs)
    ids = enc['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 < ids.shape[1] <= max_tokens:
        raise ValueError('Complete trajectory exceeds input + output cap; never truncate')
    if ids.shape[1] != row['total_tokens']:
        raise ValueError('Processor length differs from pinned trajectory export')
    tokenized = processor.tokenizer(full, add_special_tokens=False)
    text_ids = tokenized['input_ids']
    image_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    # Prove an order-preserving mapping, including each expanded image run.
    mapping, cursor = [], 0
    for token in text_ids:
        if cursor >= ids.shape[1] or int(ids[0, cursor]) != token:
            raise ValueError('Text/vision processor token mapping differs')
        mapping.append(cursor)
        if token == image_id:
            while cursor < ids.shape[1] and int(ids[0, cursor]) == image_id:
                cursor += 1
        else:
            cursor += 1
    if cursor != ids.shape[1]:
        raise ValueError('Unmapped processor tokens')
    targets = row['loss_target_message_indices']
    if (not targets or targets != sorted(set(targets)) or
            targets != [t['assistant_message_index'] for t in row['turns']]):
        raise ValueError('Target ownership is missing, duplicated or changed')
    labels = torch.full_like(ids, -100)
    previous_end = 0
    for index, turn in zip(targets, row['turns']):
        if not 0 <= index < len(messages) or messages[index]['role'] != 'assistant':
            raise ValueError('Target is not an assistant message')
        prompt, end = render(messages[:index], True), render(messages[:index+1])
        if not end.startswith(prompt) or not full.startswith(end):
            raise ValueError('Owned assistant is not an exact rendered prefix')
        # Tokenizer prefix equality rules out boundary retokenization.
        before = processor.tokenizer(prompt, add_special_tokens=False)['input_ids']
        after = processor.tokenizer(end, add_special_tokens=False)['input_ids']
        if text_ids[:len(before)] != before or text_ids[:len(after)] != after:
            raise ValueError('Assistant token prefix changed in full trajectory')
        start, stop = len(before), len(after)
        if not previous_end <= start < stop or stop-start != turn['output_tokens']:
            raise ValueError('Assistant target count/overlap differs from export')
        positions = mapping[start:stop]
        if 0 in positions or any(text_ids[p] == image_id for p in range(start, stop)):
            raise ValueError('Invalid assistant target positions')
        labels[0, positions] = ids[0, positions]
        previous_end = stop
    if int((labels != -100).sum()) != row['output_tokens']:
        raise ValueError('Supervised token count differs from export')
    return {**enc, 'labels': labels}


class Corpus:
    def __init__(self, root, folds, validation_fold=0, max_tokens=130000):
        self.root = Path(root)
        summary = json.loads((self.root/'summary.json').read_text())
        self.rows = json.loads((self.root/'index.json').read_text())
        self.data_hash = file_hash(self.root/'trajectories.jsonl')
        if self.data_hash != summary['trajectories_sha256']:
            raise ValueError('Trajectory data checksum mismatch')
        fold_data = json.loads(Path(folds).read_text())
        known = {g for f in fold_data['folds'] for g in f['game_ids']}
        validation = next(set(f['game_ids']) for f in fold_data['folds'] if f['fold'] == validation_fold)
        seen, owned, cursor, rows = set(), set(), 0, []
        with (self.root/'trajectories.jsonl').open('rb') as stream:
            for item in self.rows:
                if item['offset'] != cursor or item['trajectory_id'] in seen or item['game'] not in known:
                    raise ValueError('Invalid trajectory index/identity/fold')
                raw = stream.read(item['length']);cursor += len(raw)
                row = json.loads(raw)
                if any(row.get(k) != v for k, v in item.items() if k not in ('line', 'offset', 'length')):
                    raise ValueError('Trajectory index differs from row')
                if not 1 < row['total_tokens'] <= max_tokens or row['output_tokens'] <= 0:
                    raise ValueError('Invalid trajectory lengths')
                if row['loss_target_message_indices'] != [t['assistant_message_index'] for t in row['turns']]:
                    raise ValueError('Target ownership mismatch')
                if row['loss_target_message_indices'] != sorted(set(row['loss_target_message_indices'])):
                    raise ValueError('Duplicate or nonchronological assistant ownership')
                if sum(t['output_tokens'] for t in row['turns']) != row['output_tokens']:
                    raise ValueError('Per-turn target counts differ from trajectory')
                for turn in row['turns']:
                    if turn['id'] in owned:
                        raise ValueError('Assistant target duplicated across trajectories')
                    owned.add(turn['id'])
                seen.add(row['trajectory_id'])
                rows.append({**item, 'sha256': hashlib.sha256(raw).hexdigest(),
                             'split': 'validation' if item['game'] in validation else 'train'})
            if stream.read(1):
                raise ValueError('Index does not cover complete dataset')
        if len(rows) != summary['trajectories'] or len(owned) != summary['supervised_targets']:
            raise ValueError('Incomplete trajectory/turn coverage')
        if sum(r['output_tokens'] for r in rows) != summary['total_output_tokens']:
            raise ValueError('Target totals differ from release')
        self.rows = rows
        self.identity = dict(data_sha256=self.data_hash, index_sha256=file_hash(self.root/'index.json'),
            summary_sha256=file_hash(self.root/'summary.json'), folds_sha256=file_hash(folds))
        if not self.split('train') or not self.split('validation'):
            raise ValueError('Both training and validation trajectories are required')

    def split(self, name):
        return [r for r in self.rows if r['split'] == name]


class TrajectoryDataset(Dataset):
    def __init__(self, corpus, rows, processor):
        self.root, self.rows, self.processor_path = corpus.root, rows, processor
        self.processor = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.rows[index]
        with (self.root/'trajectories.jsonl').open('rb') as stream:
            stream.seek(item['offset']);raw = stream.read(item['length'])
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Trajectory changed after preflight')
        if self.processor is None:
            self.processor = load_processor(self.processor_path)
        row = json.loads(raw)
        tokenizer = Path(self.processor_path)/'tokenizer.json'
        template = Path(self.processor_path)/'chat_template.jinja'
        if file_hash(tokenizer) != row['tokenizer_sha256'] or file_hash(template) != row['chat_template_sha256']:
            raise ValueError('Processor tokenizer/template differs from release')
        return encode(self.processor, row)
