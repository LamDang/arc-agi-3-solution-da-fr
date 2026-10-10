"""Summarize completed capacity cases without hiding a later OOM or cache effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    cases = json.loads((root/'completed-cases.json').read_text())
    events = [json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    samples = [json.loads(s) for s in (root/'memory.jsonl').read_text().splitlines()]
    results = []
    for case in cases:
        name = case['name']
        start = next(e['seconds'] for e in events if e['event']=='sample_start' and e.get('case')==name)
        end = next(e['seconds'] for e in events if e['event']=='optimizer_complete' and e.get('case')==name)
        points = [e for e in events if e.get('case')==name]
        points += [s for s in samples if start <= s['seconds'] <= end]
        # RssFile excludes RssShmem. Subtract only the former: pinned/shared
        # tensors are part of the process working set, not reclaimable cache.
        host = max(p.get('VmRSS_gib',0)-p.get('RssFile_gib',0) for p in points)
        results.append(dict(name=name, tokens=case['tokens'], targets=case['targets'],
            context_tokens=case['tokens']-case['targets'],
            step_seconds=case['step_seconds'], forward_seconds=case['forward_seconds'],
            backward_seconds=case['backward_seconds'],
            total_tokens_per_second=case['tokens']/case['step_seconds'],
            target_tokens_per_second=case['targets']/case['step_seconds'],
            gpu_allocated_gib=case['memory']['gpu_peak_allocated_gib'],
            gpu_reserved_gib=case['memory']['gpu_peak_reserved_gib'],
            sampled_process_anon_plus_shared_peak_gib=host,
            sampled_cgroup_anon_peak_gib=max(p.get('cgroup_anon_gib',0) for p in points),
            loss=case['loss'], gradient_norm=case['grad_norm'], offload=case['offload']))
    final = json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
    result = dict(cases=results, suite_complete=bool(final), suite_passed=final.get('passed'),
                  suite_error=final.get('error'), host_memory_method='max(VmRSS-RssFile), including RssShmem; sampled, not an allocator peak',
                  timing_scope='One request forward/backward/Adam; excludes load, CPU encoding, sample deserialization and durable production checkpoint writes')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in results:
        print(f"{row['name']:25} {row['step_seconds']:7.2f}s GPU {row['gpu_allocated_gib']:6.2f}/{row['gpu_reserved_gib']:6.2f} GiB RAM {row['sampled_process_anon_plus_shared_peak_gib']:6.2f} GiB")
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_directory')
    summarize(p.parse_args().result_directory)
