"""One consolidated table: numerical comparisons and every recorded phase counter."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
ref=read(ROOT/'reports/reference.json');g=2**30
columns=['Run / total wall s','Loss (change vs ref)','Gradient relative L2 / cosine',
    'Bitwise matches: all / nonzero','Chunking vs unchunked control: relative L2 / cosine / gate','Phase','Seconds','GPU allocated / reserved GiB',
    'RAM parent RSS / tree PSS GiB','RAM child RSS / host-used GiB','RAM samples']
lines=['# v0 — current all-expert reference and optimizations','',
'Anchor: 16,249 tokens, 651 targets, 7 images; exact test-only nonzero A/B initialization.',
'All 74,472 LoRA tensors remain FP32, including all routed experts. BF16 activation',
'ports, CPU expert prefetch and disk PLE are fixed reference settings. Every',
'completed run has finite gradients, zero optimizer updates and no clipping.',
'Opt4–6 are dropped as separate experiments.',
'Each run has 1,920,915,456 trainable parameters: FP32 masters and gradients',
'are 7.155968 GiB each. There are 744 CUDA and 73,728 CPU parameter tensors.',
'Expert staging is bounded to two layers; all observed expert inputs are BF16.','',
'**One table contains the run comparisons and every measured phase statistic.**',
'Run-level loss/gradient results appear once on the first row of each experiment.',
'Bitwise match denominators are 74,472 overall and 74,394 nonzero reference tensors.',
'Each paired memory cell follows the order in its column heading.','',
'| '+' | '.join(columns)+' |','| '+' | '.join(['---']+['---:']*4+['---']+['---:']*5)+' |']
reports=[]
cases=[('Reference',ref,ROOT/'results'/ref['attempt']/'monitor.json',True)]
for key,label in [('opt1','Opt1: target-only logits'),('opt2','Opt2: CCE exact'),('opt3','Opt3: CCE exact + direct bias'),('opt7-rejected','Opt7 v1: input chunks (rejected)'),('opt7','Opt7: + expert chunks'),('opt8','Opt8: + QSA query chunks'),('opt9','Opt9: + hyperconnection chunks'),('opt10','Opt10: + PLE windows')]:
 path=ROOT/'reports'/f'{key}-rerun.json'
 if path.exists():
  r=read(path);reports.append(r);cases.append((label,r,ROOT/'results'/r['attempt']/'monitor.json',False))
 elif key!='opt7-rejected':cases.append((label,None,None,False))
for label,r,monitor,is_ref in cases:
 if r is None:
  lines.append('| '+' | '.join([label]+['pending']*10)+' |');continue
 wall=read(monitor)['seconds']
 for index,row in enumerate(r['resources']):
  phase=row['phase'];phase=f'**{phase}**' if phase in ('forward','backward') else phase
  prefix=['','','','','']
  if index==0:
   prefix=[f'{label}<br>wall {wall:.2f}',f"{r['loss']:.10f}"+(' (reference)' if is_ref else f" ({r['loss_relative_change']:+.6%})"),
       '—' if is_ref else f"{r['gradient_relative_l2']:.6%} / {r['gradient_cosine']:.10f}",
       '—' if is_ref else f"{r['bitwise_equal_gradients']:,} / {r['bitwise_equal_nonzero_reference_gradients']:,}",
       '—' if 'chunking_gradient_relative_l2' not in r else f"{r['chunking_gradient_relative_l2']:.6%} / {r['chunking_gradient_cosine']:.10f} / {'PASS' if r['chunking_gradient_gate_passed'] else 'FAIL'}"]
  values=prefix+[phase,f"{row['seconds']:.3f}",
       f"{row['cuda_peak_allocated_bytes']/g:.3f} / {row['cuda_peak_reserved_bytes']/g:.3f}",
       f"{row['rss_bytes']/g:.3f} / {row['tree_pss_bytes']/g:.3f}",
       f"{row['children_rss_bytes']/g:.3f} / {row['host_used_bytes']/g:.3f}",str(row['samples'])]
  assert len(values)==len(columns)
  lines.append('| '+' | '.join(values)+' |')
partial=ROOT/'reports/opt7-interrupted.json'
if partial.exists():
 r=read(partial)
 for index,row in enumerate(r['resources']):
  prefix=['','','','','']
  if index==0:prefix=[f"Opt7 v2: interrupted<br>wall {r['monitor']['seconds']:.2f}",'not completed','not run','—','—']
  phase=row['phase']+(' (interrupted)' if row['phase']=='forward' else '')
  values=prefix+[phase,f"{row['seconds']:.3f}",f"{row['cuda_peak_allocated_bytes']/g:.3f} / {row['cuda_peak_reserved_bytes']/g:.3f}",f"{row['rss_bytes']/g:.3f} / {row['tree_pss_bytes']/g:.3f}",f"{row['children_rss_bytes']/g:.3f} / {row['host_used_bytes']/g:.3f}",str(row['samples'])]
  lines.append('| '+' | '.join(values)+' |')
for attempt,reason in [('20261009233104571-623698fb','dispatch commit corrected'),('20261009233157982-16930101','new 1% gate/control added')]:
 job=ROOT/'results'/attempt
 if (job/'monitor.json').exists():
  wall=read(job/'monitor.json')['seconds']
  values=[f'Opt7 loading stopped: {reason}<br>wall {wall:.2f}','not run','not run','—','—','startup/loading (incomplete)','—','unavailable','unavailable','unavailable','—']
  lines.append('| '+' | '.join(values)+' |')
lines+=['','GPU peaks are synchronized CUDA allocator counters; reserved includes cache.',
'RAM peaks are sampled, not exact allocator peaks. Tree PSS includes the worker;',
'host-used is a distinct psutil system counter with different cache/mapping accounting.',
'Loading, initialization verification, gradient export and gradient comparison are',
'separate from forward/backward. Total wall also includes startup and teardown.',
'Timings are single captures and can include compilation/cache effects. No AdamW',
'state or 130K capacity measurement is included.','',
'All candidates compare directly with the same saved native-head reference.',
'Opt3 includes CCE, so its difference from reference cannot be attributed solely',
'to mask construction. Small native/direct-bias selection and SDPA forward/',
'backward fixtures passed independently before the Opt2/Opt3 model passes.','',
'Only reference raw gradients are retained. Candidates keep comparisons and',
'statistics, with exact initialization verified against reference shards. Raw',
'candidate gradients and duplicate initial-state archives are never written.',
'Opt7–10 use the user-specified gradient gate: bitwise equality or global relative L2 below 1%.',
'The isolated comparison disables all chunking on the same model; Opt1–3 remain enabled.',
'Its control F/B is separately timed and holds candidate raw gradients in RAM (never on disk).',
'Candidate F/B precedes that extra RAM retention. Native-reference differences remain reported.',
'All attempt files are SHA256 inventoried and cached in local DVC. No pushes.','',
'## Run identities','',f"- Reference `{ref['attempt']}`, execution `{ref['execution_commit']}`."]
for r in reports:
 lines.append(f"- {r['optimization']} `{r['attempt']}`, execution `{r['execution_commit']}`; evidence checks passed. See `reports/{r['optimization'].lower()}-rerun.json` and the DVC attempt for complete per-tensor metrics and source hashes.")
(ROOT/'v0.md').write_text('\n'.join(lines)+'\n')
