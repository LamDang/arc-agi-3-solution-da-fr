"""Logged real native indexer and CUDA SDPA mask/gradient checks for Opt3."""
import gc
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType
from types import SimpleNamespace

import torch
from native_mask_storage import LazyCausalMask, indexer_with_direct_bias, language_with_lazy_mask
from opt3_mask_capture import check_native_sources


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def qualify(out):
    out = Path(out)
    native = check_native_sources()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    config = SimpleNamespace(indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
                             indexer_budget=8, indexer_compress_ratio=4, hidden_size=32, rms_norm_eps=1e-6)
    # Fork RNG so qualification does not alter the subsequent model/adapter seed.
    report = {'operator_only': True, 'seed': 20261012, 'cases': [], 'guards': [],
              'native_selection_sets_exact': True, 'optimizer_updates': 0}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(report['seed'])
        for dtype in (torch.bfloat16, torch.float32):
            for tokens in (3, 4, 9, 33):
                indexer = native.Qwen4ExpTextQSAIndexer(config, layer_idx=0).cuda().to(dtype)
                hidden = torch.randn((1, tokens, 32), device='cuda', dtype=dtype)
                angles = torch.randn((1, tokens, 16), device='cuda', dtype=dtype)
                embeddings = (angles.cos(), angles.sin())
                causal = torch.ones((1, 1, tokens, tokens), device='cuda', dtype=torch.bool).tril()
                selected = indexer(hidden, embeddings, causal, None)
                patched = MethodType(indexer_with_direct_bias(indexer.forward), indexer)
                bias = patched(hidden, embeddings, LazyCausalMask(tokens, hidden.device), None)
                allowed = causal & selected
                assert torch.equal(bias == 0, allowed)
                assert torch.isneginf(bias[~allowed]).all()
                assert not bool((selected & ~causal).any())
                assert bias.dtype == hidden.dtype
                # Inspect exact native selected indices, including unused slots, by
                # replacing only the final storage block in a third read-only call.
                source = textwrap.dedent(inspect.getsource(indexer.forward))
                prefix = source.split('    # Create the additive mask to be added to the main causal mask\n')[0]
                namespace = dict(indexer.forward.__func__.__globals__)
                exec(compile(prefix + '    return selected_token_indices\n', '<selection-probe>', 'exec'), namespace)
                indices = MethodType(namespace['forward'], indexer)(hidden, embeddings, causal, None)
                assert bool(((indices < 0) | (indices <= torch.arange(tokens, device='cuda')[None, :, None])).all())
                assert bool((indices == -1).any())
                expected_bias = torch.where(allowed, hidden.new_zeros(()), float('-inf'))
                qkv = [torch.randn((1, 2, tokens, 16), device='cuda', dtype=dtype) for _ in range(3)]
                outputs, gradients = [], []
                for mask in (allowed, bias):
                    q, k, v = [x.detach().clone().requires_grad_() for x in qkv]
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                    grad = torch.autograd.grad(y.float().square().sum(), (q, k, v))
                    outputs.append(y.detach()); gradients.append(grad)
                output_exact = digest(outputs[0]) == digest(outputs[1])
                gradients_exact = all(digest(a) == digest(b) for a, b in zip(*gradients))
                row = {'tokens': tokens, 'dtype': str(dtype), 'selected_indices_sha256': digest(indices),
                       'unused_slots': int((indices == -1).sum()), 'causal': True,
                       'allowed_set_exact': True, 'bias_values_exact': torch.equal(bias, expected_bias),
                       'sdpa_output_bytes_exact': output_exact, 'qkv_gradient_bytes_exact': gradients_exact,
                       'bias_storage_bytes': bias.untyped_storage().nbytes()}
                report['cases'].append(row)
                # Small raw fixtures/outputs/gradients support independent review.
                torch.save({'hidden': hidden.cpu(), 'cos': embeddings[0].cpu(), 'sin': embeddings[1].cpu(),
                            'indices': indices.cpu(), 'allowed': allowed.cpu(), 'bias': bias.cpu(),
                            'q': qkv[0].cpu(), 'k': qkv[1].cpu(), 'v': qkv[2].cpu(),
                            'native_output': outputs[0].cpu(), 'opt3_output': outputs[1].cpu(),
                            **{f'{mode}_d{name}': gradient.cpu() for mode, group in zip(('native', 'opt3'), gradients)
                               for name, gradient in zip(('q', 'k', 'v'), group)}},
                           out / f'mask-operator-{str(dtype).split(".")[-1]}-{tokens}.pt')
                assert output_exact and gradients_exact
                del indexer, hidden, angles, embeddings, causal, selected, bias, qkv, outputs, gradients, indices
    stub = SimpleNamespace(native_mask_forward=lambda **kwargs: kwargs)
    x = torch.zeros((1, 3, 4))
    for name, kwargs in [('padding', {'inputs_embeds': x, 'attention_mask': torch.tensor([[1, 0, 1]])}),
                         ('batch', {'inputs_embeds': x.expand(2, -1, -1)}),
                         ('cache', {'inputs_embeds': x, 'past_key_values': object()}),
                         ('use_cache', {'inputs_embeds': x, 'use_cache': True})]:
        try:
            language_with_lazy_mask(stub, **kwargs)
        except ValueError:
            report['guards'].append(name)
        else:
            raise AssertionError('Missing mask guard: ' + name)
    result = language_with_lazy_mask(stub, inputs_embeds=x, attention_mask=torch.ones(1, 3), use_cache=False)
    assert isinstance(result['attention_mask']['indexed_attention'], LazyCausalMask)
    assert result['attention_mask']['linear_attention'] is None
    report['passed'] = True
    (out / 'mask-operator-check-report.json').write_text(json.dumps(report, indent=2) + '\n')
    gc.collect(); torch.cuda.empty_cache()
    return report
