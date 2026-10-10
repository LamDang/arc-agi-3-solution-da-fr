"""CUDA-only native/target-mask checks for interleaved 16K supervision.

Executed as part of the logged candidate process, before model loading. Saves
the synthetic hidden fixture and both raw hidden gradients for mean and summed
CE. This qualifies the operator, independently of the anchor adapter check.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers.loss.loss_utils import ForCausalLMLoss
from target_only_head import target_logits, target_positions, target_cross_entropy


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def qualify(model_path, sample_path, out):
    out = Path(out)
    index = json.loads((Path(model_path) / 'model.safetensors.index.json').read_text())
    with safe_open(str(Path(model_path) / index['weight_map']['lm_head.weight']),
                   framework='pt', device='cpu') as archive:
        weight_cpu = archive.get_tensor('lm_head.weight')
    weight_sha = digest(weight_cpu)
    weight = weight_cpu.cuda()
    del weight_cpu
    batch = torch.load(sample_path, map_location='cpu', weights_only=True)
    tokens = batch['input_ids'].shape[1]
    labels = torch.full_like(batch['input_ids'], -100)
    # Four separated diagnostic spans. No claim that these are actual assistant
    # annotations; their purpose is to qualify arbitrary interleaving.
    spans = [(1, 17), (tokens // 4, tokens // 4 + 128),
             (tokens // 2, tokens // 2 + 256), (tokens - 251, tokens)]
    for start, stop in spans:
        labels[:, start:stop] = batch['input_ids'][:, start:stop]
    del batch
    generator = torch.Generator(device='cpu').manual_seed(20261011)
    original = torch.randn((1, tokens, weight.shape[1]), generator=generator)
    torch.save(original, out / 'operator-hidden.pt')
    torch.save(labels, out / 'operator-labels.pt')
    labels = labels.cuda()
    positions, targets = target_positions(labels)
    report = {'operator_only': True, 'synthetic_hidden_and_supervision': True,
              'context_tokens': tokens, 'target_tokens': positions.numel(),
              'spans': spans, 'head_sha256': weight_sha,
              'hidden_sha256': digest(original), 'cases': [],
              'optimizer_updates': 0}
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    for reduction, denominator, scale in [('mean', None, 1.), ('sum', 1, .125)]:
        expected = None
        expected_loss = None
        for mode in ['native', 'candidate']:
            hidden = original.cuda().requires_grad_()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if mode == 'native':
                        logits = torch.nn.functional.linear(hidden, weight)
                        loss = ForCausalLMLoss(logits, labels, weight.shape[0],
                                               num_items_in_batch=denominator)
                    else:
                        logits, selected = target_logits(hidden, weight, labels)
                        loss = target_cross_entropy(logits, labels, positions,
                                                    selected, denominator)
                del logits
                (loss * scale).backward()
            torch.cuda.synchronize()
            gradient = hidden.grad.detach().cpu().clone()
            torch.save(gradient, out / f'operator-{reduction}-{mode}-dhidden.pt')
            row = {'mode': mode, 'reduction': reduction, 'upstream_scale': scale,
                   'loss': loss.item(), 'gradient_sha256': digest(gradient),
                   'seconds': time.monotonic() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            if mode == 'native':
                expected, expected_loss = gradient, loss.item()
            else:
                row.update(loss_exact=loss.item() == expected_loss,
                           gradient_bytes_exact=digest(gradient) == digest(expected))
            report['cases'].append(row)
            (out / 'operator-check-report.json').write_text(json.dumps(report, indent=2) + '\n')
            del hidden, loss, gradient
            gc.collect()
            torch.cuda.empty_cache()
        del expected
    report['passed'] = all(row['loss_exact'] and row['gradient_bytes_exact']
                           for row in report['cases'] if row['mode'] == 'candidate')
    (out / 'operator-check-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('Interleaved CUDA operator loss/gradient gate failed')
    del original, labels, positions, targets, weight
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
