"""Token-normalized AdamW updates and restartable epoch boundaries."""
from contextlib import nullcontext
import random
import time

import torch


def initial_progress():
    return dict(epoch=0, cursor=0, updates=0, target_tokens=0, trajectories=0,
                epoch_nll_sum=0., epoch_targets=0)


def fit(model, optimizer, train_rows, epochs, target_budget, seed, batches,
        loss_fn, evaluate, checkpoint, log, progress=None, context=nullcontext,
        max_grad_norm=1., phase=lambda name: nullcontext(), release=lambda: None):
    progress = dict(progress or initial_progress())
    optimizer.zero_grad(set_to_none=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(progress['epoch'], epochs):
        order = list(train_rows)
        random.Random(seed+epoch).shuffle(order)
        start = progress['cursor'] if epoch == progress['epoch'] else 0
        if not 0 <= start <= len(order):
            raise ValueError('Checkpoint cursor exceeds epoch')
        pending = 0; nll_sum = 0.; update_start = time.monotonic()
        model.train()
        for index,sample in enumerate(batches(order, start), start):
            if index >= len(order):
                raise ValueError('Loader yielded too many trajectories')
            with context():
                with phase('forward'):
                    loss, targets = loss_fn(sample)
                if type(targets) is not int or targets <= 0 or not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss or invalid target count')
                scalar = float(loss.detach())
                with phase('backward'):
                    (loss*targets).backward()
            del loss, sample
            release()
            pending += targets; nll_sum += scalar*targets
            progress.update(epoch=epoch, cursor=index+1,
                target_tokens=progress['target_tokens']+targets,
                trajectories=progress['trajectories']+1,
                epoch_nll_sum=progress['epoch_nll_sum']+scalar*targets,
                epoch_targets=progress['epoch_targets']+targets)
            log('trajectory', progress['target_tokens'], dict(loss=scalar, targets=targets,
                epoch=epoch+1, trajectory=index+1))
            if pending >= target_budget or index+1 == len(order):
                for p in parameters:
                    if p.grad is not None:
                        p.grad.div_(pending)
                with phase('optimizer'):
                    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm,
                        error_if_nonfinite=True)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress['updates'] += 1
                log('train', progress['updates'], dict(loss=nll_sum/pending,
                    gradient_norm=float(norm), learning_rate=optimizer.param_groups[0]['lr'],
                    targets=pending, total_targets=progress['target_tokens'], epoch=epoch+1,
                    epoch_tail=pending < target_budget,
                    update_wall_seconds=time.monotonic()-update_start,
                    targets_per_second=pending/(time.monotonic()-update_start)))
                started = time.monotonic()
                checkpoint(dict(progress))  # Blocks until durable local copy + DVC registration.
                log('checkpoint', progress['updates'], dict(seconds=time.monotonic()-started))
                pending = 0; nll_sum = 0.; update_start = time.monotonic()
        if progress['cursor'] != len(order):
            raise ValueError('Loader omitted trajectories')
        model.eval()
        with torch.no_grad(), phase('validation'):
            metrics = evaluate()
        if not metrics['targets'] or not all(torch.isfinite(torch.tensor(metrics[k])) for k in ('nll', 'accuracy')):
            raise RuntimeError('Invalid validation metrics')
        log('validation', epoch+1, {**metrics, 'updates': progress['updates'],
            'train_epoch_nll': progress['epoch_nll_sum']/progress['epoch_targets']})
        progress.update(epoch=epoch+1, cursor=0, epoch_nll_sum=0., epoch_targets=0)
    return progress
