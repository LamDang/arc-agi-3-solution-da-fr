"""Offline end-to-end training/resume tests; no model download or CUDA."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import checkpoint
from training.config import Training, load
from training.data import Corpus, digest, encode
from training.engine import fit
from training.metrics import Metrics, correct_tokens


class Tiny(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 5, dtype=torch.float64)
        self.head = torch.nn.Linear(5, 11, bias=False, dtype=torch.float64)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, ids):
        return self.head(self.dropout(self.embedding(ids)))


def samples():
    result = []
    for index,count in enumerate((2, 5, 3, 1)):
        ids = (torch.arange(12).unsqueeze(0)+index) % 11
        labels = torch.full_like(ids, -100)
        positions = [1, 3, 5, 7, 10][:count]
        labels[0, positions] = ids[0, positions]
        result.append(dict(input_ids=ids, labels=labels))
    return result


def loss_fn(model, sample):
    logits = model(sample['input_ids'])[:, :-1]
    targets = sample['labels'][:, 1:]
    return F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1), ignore_index=-100), int((targets != -100).sum())


def evaluate(model):
    metrics = Metrics()
    for sample in samples():
        logits = model(sample['input_ids'])[:, :-1]
        targets = sample['labels'][:, 1:]
        mask = targets != -100
        metrics.add(F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1), reduction='sum'),
                    ((logits.argmax(-1) == targets) & mask).sum(), mask.sum())
    return metrics.result()


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42);random.seed(42);np.random.seed(42)

    def run_fit(self, model, optimizer, save=lambda p: None, progress=None, logs=None, epochs=2):
        logs = logs if logs is not None else []
        data = samples()
        result = fit(model,optimizer,list(range(len(data))),epochs,6,17,
            lambda order,start: (data[i] for i in order[start:]),
            lambda sample: loss_fn(model,sample),lambda: evaluate(model),save,
            lambda group,step,values: logs.append((group,step,values)),
            progress=progress,max_grad_norm=.25)
        return result,logs

    def test_accumulation_matches_token_weighted_oracle_and_flushes_each_epoch(self):
        model = Tiny();oracle = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False,weight_decay=0)
        control = torch.optim.AdamW(oracle.parameters(),lr=1e-4,foreach=False,weight_decay=0)
        progress,logs = self.run_fit(model,optimizer)
        expected_updates = []
        for epoch in range(2):
            order = list(range(4));random.Random(17+epoch).shuffle(order)
            pending, count = [],0
            for index,i in enumerate(order):
                loss,n = loss_fn(oracle,samples()[i]);pending.append(loss*n);count += n
                if count >= 6 or index == len(order)-1:
                    (sum(pending)/count).backward()
                    norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(),.25)
                    control.step();control.zero_grad(set_to_none=True)
                    expected_updates.append((count,float(norm)))
                    pending,count = [],0
        for p,q in zip(model.parameters(),oracle.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-12)
        updates = [row[2] for row in logs if row[0] == 'train']
        self.assertEqual([x['targets'] for x in updates],[x[0] for x in expected_updates])
        self.assertTrue(any(x['epoch_tail'] for x in updates))
        self.assertEqual(progress['target_tokens'],22)
        self.assertEqual(len([row for row in logs if row[0] == 'validation']),2)
        for actual,(_,norm) in zip(updates,expected_updates):
            self.assertAlmostEqual(actual['gradient_norm'],norm,12)

    def test_sharded_resume_matches_uninterrupted_weights_moments_rng_and_order(self):
        model = Tiny(.2)
        initial = copy.deepcopy(model.state_dict());rng = checkpoint.rng_state()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False,weight_decay=0)
        complete,_ = self.run_fit(model,optimizer)
        expected_rng = torch.rand(4)
        torch.manual_seed(100)
        interrupted = Tiny(.2);interrupted.load_state_dict(initial)
        checkpoint.restore_rng(*rng)
        first_optimizer = torch.optim.AdamW(interrupted.parameters(),lr=1e-4,foreach=False,weight_decay=0)
        identity = {'dataset':'mock','config':'fixed'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def stop(progress):
                checkpoint.save(interrupted,first_optimizer,root,progress,identity,shard_bytes=6000)
                raise RuntimeError('simulated crash after durable update')
            with self.assertRaisesRegex(RuntimeError,'simulated crash'):
                self.run_fit(interrupted,first_optimizer,save=stop)
            resumed = Tiny(.2)
            new_optimizer = torch.optim.AdamW(resumed.parameters(),lr=1e-4,foreach=False,weight_decay=0)
            progress = checkpoint.load(resumed,new_optimizer,root,identity)
            final,_ = self.run_fit(resumed,new_optimizer,progress=progress)
            self.assertEqual(final,complete)
            for p,q in zip(model.parameters(),resumed.parameters()):self.assertTrue(torch.equal(p,q))
            for a,b in zip(optimizer.state.values(),new_optimizer.state.values()):
                for key in a:self.assertTrue(torch.equal(a[key],b[key]))
            self.assertTrue(torch.equal(expected_rng,torch.rand(4)))
            with self.assertRaisesRegex(ValueError,'identity differs'):
                checkpoint.load(resumed,new_optimizer,root,{'dataset':'changed'})
            first = next(root.glob('*.pt'));first.write_bytes(first.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'corrupt'):
                checkpoint.load(resumed,new_optimizer,root,identity)

    def test_crash_during_accumulation_replays_from_last_update(self):
        model = Tiny();initial = copy.deepcopy(model.state_dict());data = samples();saved = []
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        def broken(order,start):
            yield data[order[start]]
            raise RuntimeError('interrupted before threshold')
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            fit(model,optimizer,list(range(4)),1,100,17,broken,lambda s:loss_fn(model,s),
                lambda:evaluate(model),saved.append,lambda *a:None)
        self.assertFalse(saved)
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,initial[key]))

    def test_resume_at_epoch_flush_repeats_validation_without_repeating_update(self):
        model=Tiny();initial=copy.deepcopy(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
        full,_=self.run_fit(model,optimizer)
        interrupted=Tiny();interrupted.load_state_dict(initial)
        first_optimizer=torch.optim.AdamW(interrupted.parameters(),lr=1e-4,weight_decay=0)
        with tempfile.TemporaryDirectory() as directory:
            def stop(progress):
                if progress['cursor']==4:
                    checkpoint.save(interrupted,first_optimizer,directory,progress,{},6000)
                    raise RuntimeError('crash before epoch validation')
            with self.assertRaisesRegex(RuntimeError,'epoch validation'):
                self.run_fit(interrupted,first_optimizer,save=stop)
            resumed=Tiny();new_optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-4)
            progress=checkpoint.load(resumed,new_optimizer,directory,{})
            final,logs=self.run_fit(resumed,new_optimizer,progress=progress)
            self.assertEqual(final,full)
            self.assertEqual(len([r for r in logs if r[0]=='validation']),2)
            for p,q in zip(model.parameters(),resumed.parameters()):self.assertTrue(torch.equal(p,q))

    def test_mailbox_keeps_unacknowledged_shard_on_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'00000.pt';path.write_bytes(b'checkpoint payload')
            mailbox=checkpoint.Mailbox(directory,timeout=1)
            with patch('training.checkpoint.time.monotonic',side_effect=[0,2]):
                with self.assertRaises(TimeoutError):mailbox('update-00000001',path,'shard')
            self.assertTrue(path.exists())
            request=json.loads((Path(directory)/'transfer/request.json').read_text())
            checkpoint.atomic_json(Path(directory)/'transfer/ack.json',
                dict(id=request['id'],sha256=request['sha256']))
            mailbox('update-00000001',path,'shard')
            self.assertFalse(path.exists())

    def test_streaming_save_and_restore_keep_only_one_shard_in_server_spool(self):
        model = Tiny();optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        loss_fn(model,samples()[0])[0].backward();optimizer.step()
        with tempfile.TemporaryDirectory() as server,tempfile.TemporaryDirectory() as desktop:
            server,desktop = Path(server),Path(desktop)
            def collect(name,path,kind):
                self.assertEqual(len(list(server.iterdir())),1)
                (desktop/path.name).write_bytes(path.read_bytes());path.unlink()
            checkpoint.save(model,optimizer,server,dict(updates=1),{},6000,collect)
            fresh = Tiny();new_optimizer = torch.optim.AdamW(fresh.parameters(),lr=1e-4)
            (server/'manifest.json').write_bytes((desktop/'manifest.json').read_bytes())
            def receive(row):
                path=server/row['path'];path.write_bytes((desktop/row['path']).read_bytes())
                self.assertEqual(len(list(server.glob('*.pt'))),1)
                return path
            checkpoint.load(fresh,new_optimizer,server,{},receive=receive)
            for p,q in zip(model.parameters(),fresh.parameters()):self.assertTrue(torch.equal(p,q))
            self.assertFalse(list(server.glob('*.pt')))

    def test_accuracy_chunking_matches_full_head_with_ties_and_interleaved_labels(self):
        hidden = torch.randn(1,12,5,dtype=torch.float64)
        weight = torch.randn(11,5,dtype=torch.float64)
        weight[1] = weight[0]
        labels = samples()[1]['labels']
        reference = ((F.linear(hidden,weight)[:,:-1].argmax(-1) == labels[:,1:]) & (labels[:,1:] != -100)).sum()
        self.assertEqual(correct_tokens(hidden,weight,labels,2,3),int(reference))
        weight.zero_();labels = torch.full_like(labels,-100);labels[:,1::2]=0
        self.assertEqual(correct_tokens(hidden,weight,labels,2,3),6)
        metrics = Metrics();metrics.add(2.,1,2);metrics.add(12.,2,6)
        self.assertEqual(metrics.result(),dict(nll=1.75,accuracy=.375,correct=3,targets=8))

    def test_production_config_and_limits(self):
        path=Path(__file__).resolve().parents[1]/'configs/train-trajectories.json'
        config,settings=load(path)
        self.assertEqual(settings.learning_rate,1e-4)
        self.assertEqual(config.max_tokens,130000)
        self.assertEqual(config.adapter_tensors,74472)
        self.assertIsNone(config.fla_numeric_profile)
        for changes in (dict(epochs=6),dict(lookahead=3),dict(validation_fold=1),dict(target_tokens_per_update=0)):
            with self.assertRaises(ValueError):Training('/data','/processor','/folds',**changes).validate()

    def test_five_epochs_max_and_validation_is_eval(self):
        model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        progress,logs=self.run_fit(model,optimizer,epochs=5)
        self.assertEqual(progress['epoch'],5)
        self.assertEqual(progress['trajectories'],20)
        self.assertFalse(model.training)
        self.assertEqual(len([x for x in logs if x[0]=='validation']),5)

    def test_tensorboard_records_train_validation_and_norm(self):
        try:from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:self.skipTest('TensorBoard is a declared training dependency')
        from training.telemetry import Logger
        with tempfile.TemporaryDirectory() as directory:
            logger=Logger(directory)
            model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            fit(model,optimizer,list(range(4)),1,6,17,
                lambda order,start:(samples()[i] for i in order[start:]),
                lambda s:loss_fn(model,s),lambda:evaluate(model),lambda p:None,logger)
            logger.close()
            events=EventAccumulator(str(Path(directory)/'tensorboard')).Reload()
            self.assertIn('train/gradient_norm',events.Tags()['scalars'])
            self.assertIn('train/loss',events.Tags()['scalars'])
            self.assertIn('validation/accuracy',events.Tags()['scalars'])


class FakeTokenizer:
    def __call__(self,text,**kwargs):
        return dict(input_ids=[ord(c) for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])
    def convert_tokens_to_ids(self,token):return ord('~')


class FakeProcessor:
    tokenizer=FakeTokenizer()
    def apply_chat_template(self,messages,tokenize,add_generation_prompt=False,**kwargs):
        text=''.join(m['role'][0]+'|'+m.get('content','')+';' for m in messages)
        if add_generation_prompt:text+='a|'
        if not tokenize:return text
        ids=[]
        for c in text:ids.extend([ord(c)]*(3 if c=='~' else 1))
        return dict(input_ids=torch.tensor([ids]))


class EncodingTests(unittest.TestCase):
    def test_owned_targets_mask_inherited_assistant_and_image_expansion(self):
        row=dict(messages=[dict(role='system',content='s'),dict(role='assistant',content='old'),
            dict(role='user',content='~'),dict(role='assistant',content='new'),
            dict(role='tool',content='observation'),dict(role='assistant',content='ok')],
            chat_template_kwargs={'preserve_thinking':True},loss_target_message_indices=[3,5],
            turns=[dict(assistant_message_index=3,output_tokens=4),dict(assistant_message_index=5,output_tokens=3)],
            total_tokens=0,output_tokens=7)
        processor=FakeProcessor()
        row['total_tokens']=processor.apply_chat_template(row['messages'],True)['input_ids'].shape[1]
        result=encode(processor,row)
        self.assertEqual(int((result['labels']!=-100).sum()),7)
        supervised=''.join(chr(x) for x in result['labels'][result['labels']!=-100].tolist())
        self.assertEqual(supervised,'new;ok;')
        self.assertEqual(int(result['labels'][0,0]),-100)
        with self.assertRaisesRegex(ValueError,'never truncate'):encode(processor,row,max_tokens=10)
        row['loss_target_message_indices']=[1,3,5]
        with self.assertRaises(ValueError):encode(processor,row)

    def test_corpus_index_hash_coverage_and_fold0_game_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[];raw=[]
            for i,game in enumerate(('held-out','train','held-out')):
                row=dict(trajectory_id=f'{game}-{i}',game=game,total_tokens=10,output_tokens=3,
                    loss_target_message_indices=[1],turns=[dict(id=f'{game}#{i}',assistant_message_index=1,output_tokens=3)])
                line=(json.dumps(row)+'\n').encode()
                rows.append(dict(trajectory_id=row['trajectory_id'],game=game,total_tokens=10,output_tokens=3,
                                 offset=sum(map(len,raw)),length=len(line)))
                raw.append(line)
            (root/'trajectories.jsonl').write_bytes(b''.join(raw))
            (root/'index.json').write_text(json.dumps(rows))
            summary=dict(trajectories_sha256=checkpoint.file_hash(root/'trajectories.jsonl'),
                         trajectories=3,supervised_targets=3,total_output_tokens=9)
            (root/'summary.json').write_text(json.dumps(summary))
            folds=root/'folds.json';folds.write_text(json.dumps(dict(folds=[dict(fold=0,game_ids=['held-out']),dict(fold=1,game_ids=['train'])])))
            corpus=Corpus(root,folds)
            self.assertEqual(len(corpus.split('train')),1)
            self.assertEqual(len(corpus.split('validation')),2)
            self.assertTrue(all(r['game']=='train' for r in corpus.split('train')))
            (root/'trajectories.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):Corpus(root,folds)


if __name__=='__main__':unittest.main()
