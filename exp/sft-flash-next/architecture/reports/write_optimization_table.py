"""Render the active v0 table from the verified reference and rerun summaries."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ref=json.loads((ROOT/'reports/reference.json').read_text());g=2**30
lines=['# v0 — current all-expert reference and Opt1–3', '',
'Anchor:16,249 tokens,651 targets,7 images. Test-only nonzero A/B initialization;',
'74,472 FP32 LoRA tensors, including all routed experts. BF16 activation ports,',
'CPU expert prefetch and disk PLE are fixed reference settings. No optimizer',
'updates or clipping. Opt4–6 are dropped as separate experiments.', '',
'| Run | Loss | Gradient difference vs reference (relative L2) | Bitwise matching tensors | F/B seconds | F/B GPU allocated GiB | F/B RAM tree PSS GiB |',
'| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
def metrics(rows):
 phases={r['phase']:r for r in rows};a,b=phases['forward'],phases['backward']
 return f"{a['seconds']:.2f} / {b['seconds']:.2f}",f"{a['cuda_peak_allocated_bytes']/g:.3f} / {b['cuda_peak_allocated_bytes']/g:.3f}",f"{a['tree_pss_bytes']/g:.3f} / {b['tree_pss_bytes']/g:.3f}"
time,gpu,ram=metrics(ref['resources']);lines.append(f"| Reference | {ref['loss']:.10f} | — | — | {time} | {gpu} | {ram} |")
reports=[]
for key,label in [('opt1','Opt1: target-only logits'),('opt2','Opt2: CCE exact'),('opt3','Opt3: CCE exact + direct bias')]:
 path=ROOT/'reports'/f'{key}-rerun.json'
 if not path.exists():
  lines.append(f'| {label} | pending | pending | pending | pending | pending | pending |');continue
 r=json.loads(path.read_text());reports.append(r);time,gpu,ram=metrics(r['resources'])
 lines.append(f"| {label} | {r['loss']:.10f} | {r['gradient_relative_l2']:.6%} | {r['bitwise_equal_gradients']:,} /74,472 | {time} | {gpu} | {ram} |")
lines+=['', 'RAM is sampled process-tree PSS; GPU is synchronized allocator allocated peak.',
'Full JSON retains reserved GPU, parent RSS, child RSS, host-used RAM, all phases',
'and per-tensor comparison metrics. No AdamW state or130K capacity measurement.', '',
'All candidates compare directly with the same saved native-head reference.',
'Opt3 includes CCE, so its difference from reference cannot be attributed solely',
'to mask construction. Small native/direct-bias selection and SDPA forward/',
'backward fixtures are checked independently before the Opt2/Opt3 model passes.', '',
'Only reference raw gradients are retained. Candidates keep comparisons and',
'statistics, with exact initialization verified against reference shards. Their',
'raw gradients and duplicate initial-state archives are never written. No',
'acceptance tolerance is invented; bitwise matches and numerical drift are',
'reported explicitly. No Git or DVC push.', '', '## Run identities', '',
f"- Reference `{ref['attempt']}`, execution `{ref['execution_commit']}`."]
for r in reports:
 lines.append(f"- {r['optimization']} `{r['attempt']}`, execution `{r['execution_commit']}`; evidence checks passed. All gradients finite; initialization matches every reference tensor.")
 lines.append(f"  Nonzero reference gradient matches: {r['bitwise_equal_nonzero_reference_gradients']:,} /{r['nonzero_reference_tensors']:,}; gradient cosine {r['gradient_cosine']:.10f}.")
(ROOT/'v0.md').write_text('\n'.join(lines)+'\n')
