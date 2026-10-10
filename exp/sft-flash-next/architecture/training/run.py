"""Final architecture + real compaction trajectories, with acknowledged recovery."""
from contextlib import contextmanager, nullcontext
import importlib.metadata
import random
from pathlib import Path

import numpy as np
import torch

from components.head import cce_target_loss
from components.ple import PreparedPLEDataset
from model import build
from runtime.evidence import model_identity, write
from runtime.loop import prepare
from runtime.resources import Resources
from . import checkpoint
from .config import identity_config
from .data import Corpus, TrajectoryDataset, file_hash
from .engine import fit
from .telemetry import Logger
from .metrics import Metrics, correct_tokens


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json', '.mjs'} and
            not set(p.relative_to(root).parts).intersection({'results', 'reports', '__pycache__'})}


def run(config, settings):
    if config.optimizations.head != 'cce_exact':
        raise ValueError('Production evaluation currently requires frozen-head CCE exact')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    torch.set_num_threads(8)
    # Real training keeps native kernel autotuning. Reproducibility forcing is
    # confined to test/capacity entry points.
    output = Path(config.output);output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(settings.dataset, settings.folds, max_tokens=config.max_tokens)
    identity = dict(dataset=corpus.identity, model=model_identity(config.model),
                    processor={p.name:file_hash(p) for p in Path(settings.processor).iterdir()
                               if p.is_file() and p.suffix in {'.json', '.jinja', '.txt', '.model'}},
                    config=identity_config(config, settings), source=source_identity())
    identity['packages'] = {name:importlib.metadata.version(name) for name in
        ('torch','transformers','peft','auto-round','flash-linear-attention','fla-core','tensorboard')}
    identity['cuda'] = torch.version.cuda
    write(output/'identity.json', identity)
    write(output/'config.json', {**config.as_dict(), 'training': settings.__dict__})
    write(output/'data-summary.json', {split:dict(trajectories=len(corpus.split(split)),
        targets=sum(r['output_tokens'] for r in corpus.split(split)),
        tokens=sum(r['total_tokens'] for r in corpus.split(split))) for split in ('train','validation')})
    logger = Logger(output)
    resources = Resources(output)
    architecture = None
    try:
        with resources.phase('loading'):
            architecture = build(config, output)
        model = architecture.model
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, betas=tuple(settings.betas),
            eps=settings.eps, weight_decay=settings.weight_decay, foreach=False, fused=False)
        progress = checkpoint.load(model, optimizer, settings.resume, identity,
            checkpoint.RestoreMailbox(output, settings.ack_timeout)) if settings.resume else None
        write(output/'components.json', architecture.inventory)
        transfer = checkpoint.Mailbox(output, settings.ack_timeout)

        def batches(rows, start):
            if start == len(rows):
                return
            source = TrajectoryDataset(corpus, rows[start:], settings.processor)
            # A dedicated generator prevents iterator creation/restart from
            # consuming the training RNG restored from the checkpoint.
            generator = torch.Generator().manual_seed(config.seed)
            if config.optimizations.disk_ple:
                source = PreparedPLEDataset(source, config.model, str(output/'ple-events.jsonl'))
            loader = torch.utils.data.DataLoader(source, batch_size=None, num_workers=1,
                multiprocessing_context='spawn', prefetch_factor=settings.lookahead, generator=generator)
            iterator = iter(loader)
            try:
                yield from iterator
            finally:
                if getattr(iterator, '_shutdown_workers', None):
                    iterator._shutdown_workers()

        def activate(sample):
            if not config.optimizations.disk_ple:
                sample = {'batch': sample}
            architecture.activate(sample)
            return prepare(sample['batch'], config, 'train')

        def release():
            if config.optimizations.disk_ple:
                module = model.get_base_model().model.language_model.layers[1].ple.ple_embedding
                module.prepared_payload = module.prepared_input_ids = None

        def loss_fn(sample):
            batch,labels,count = activate(sample)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = architecture.loss(batch, labels)
            return loss,count

        def evaluate():
            metrics = Metrics()
            for sample in batches(corpus.split('validation'), 0):
                batch,labels,count = activate(sample)
                base = model.get_base_model()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = base.model(**batch, use_cache=False).last_hidden_state
                    nll = cce_target_loss(hidden, base.lm_head.weight, labels, reduction='sum')
                    correct = correct_tokens(hidden, base.lm_head.weight, labels,
                        settings.accuracy_tokens, settings.accuracy_vocab)
                metrics.add(float(nll), correct, count)
                del hidden,nll,batch,labels,sample
                release()
            return metrics.result()

        @contextmanager
        def phase(name):
            with resources.phase(name):
                yield
            row = resources.rows[-1]
            # Resource sampling also records cgroup and allocator diagnostics.
            logger('resources_'+name, len(resources.rows),
                   {k:v for k,v in row.items() if isinstance(v, (int, float))})
            write(output/'resources.json', resources.rows)

        def save(progress):
            logger.writer.flush()
            checkpoint.save(model, optimizer, output/'transfer'/'spool', progress, identity,
                settings.shard_bytes, transfer)

        progress = fit(model, optimizer, corpus.split('train'), settings.epochs,
            settings.target_tokens_per_update, config.seed, batches, loss_fn, evaluate, save, logger,
            progress=progress,
            context=(lambda: torch.autograd.graph.save_on_cpu(pin_memory=False)) if config.save_on_cpu else nullcontext,
            max_grad_norm=settings.max_grad_norm, phase=phase, release=release)
        write(output/'result.json', dict(completed=True, mode='train', **progress))
    finally:
        logger.close()
        write(output/'resources.json', resources.rows)
        if architecture and architecture.expert_stager:
            architecture.expert_stager.close()
