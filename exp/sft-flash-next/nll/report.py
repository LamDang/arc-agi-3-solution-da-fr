"""CPU-only paired NLL summaries and a local interactive target-token viewer."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import LABELS, atomic_bytes, length_band, read_json, write_json, verify_manifest
from results import load_result


def weighted_summary(rows, values):
    by_game = defaultdict(list)
    loss_sum, tokens, request_sum, weights = 0., 0., 0., 0.
    categories = {label: [0., 0., 0] for label in LABELS}
    for row in rows:
        nll = values[row["sample_id"]].astype(np.float64)
        w = row["weight"]
        mean = float(nll.mean())
        by_game[row["game"]].append((w, mean))
        loss_sum += w * nll.sum()
        tokens += w * len(nll)
        request_sum += w * mean
        weights += w
        cats = np.asarray(row["labels"])
        for i, label in enumerate(LABELS):
            chosen = nll[cats == i]
            categories[label][0] += w * chosen.sum()
            categories[label][1] += w * len(chosen)
            categories[label][2] += int(len(chosen) > 0)
    game_nll = {g: sum(w*x for w, x in pairs)/sum(w for w, x in pairs)
                for g, pairs in by_game.items()}
    return dict(primary=float(np.mean(list(game_nll.values()))) if game_nll else None,
                per_game=game_nll, token_nll=float(loss_sum/tokens) if tokens else None,
                perplexity=math.exp(min(700, loss_sum/tokens)) if tokens else None,
                population_request_nll=request_sum/weights if weights else None,
                categories={k: dict(nll=float(s/t) if t else None, weighted_sum=float(s),
                                    weighted_tokens=float(t), requests=n) for k, (s, t, n) in categories.items()})


def choice(summaries):
    best = min(s["primary"] for s in summaries.values())
    return min(int(c) for c, s in summaries.items() if s["primary"] <= best + .05)


def generate(root):
    root = Path(root)
    manifest, run = read_json(root / "manifest.json"), read_json(root / "run.json")
    verify_manifest(manifest)
    counts, samples = manifest["counts"], manifest["samples"]
    results, timing, status = {}, {}, {}
    for count in counts:
        values, metrics = {}, []
        for row in samples:
            record = load_result(root, run["identity"], count, row)
            if record is not None:
                meta, loss = record
                values[row["sample_id"]] = loss
                metrics.append(meta["metrics"])
        results[count], timing[count], status[count] = values, metrics, len(values)
    common_ids = set.intersection(*(set(results[c]) for c in counts))
    common_rows = [r for r in samples if r["sample_id"] in common_ids]
    complete = len(common_rows) == len(samples)
    summaries = {str(c): weighted_summary(common_rows, results[c]) for c in counts}
    for c in counts:
        entry = summaries[str(c)]
        reference = summaries[str(counts[0])]["primary"]
        entry.update(completed=status[c], common_requests=len(common_rows),
                     delta_vs_full=entry["primary"]-reference if reference is not None else None,
                     scoring_seconds=sum(m.get("seconds", 0) for m in timing[c]),
                     peak_allocated_bytes=max((m.get("peak_allocated_bytes", 0) for m in timing[c]), default=0))
    selection = dict(status="complete" if complete else "incomplete", selected_experts=None,
                     matched_requests=len(common_rows), required_requests=len(samples),
                     manifest_sha256=manifest["manifest_sha256"], identity=run["identity"],
                     training_fit="unverified", calibration="training-only")
    if complete:
        candidate = choice(summaries)
        candidate_summary, full = summaries[str(candidate)], summaries[str(counts[0])]
        flags = []
        a, b = candidate_summary["categories"]["tool_code"]["nll"], full["categories"]["tool_code"]["nll"]
        if a is not None and b is not None and a-b > .1:
            flags.append("tool_code_delta_gt_0.10")
        if any(candidate_summary["per_game"][g]-full["per_game"][g] > .15 for g in full["per_game"]):
            flags.append("game_delta_gt_0.15")
        leave_one_out = {}
        for game in manifest["validation_games"]:
            subset = [r for r in common_rows if r["game"] != game]
            leave_one_out[game] = choice({str(c): weighted_summary(subset, results[c]) for c in counts})
        if any(c != candidate for c in leave_one_out.values()):
            flags.append("leave_one_game_out_choice_changes")
        selection.update(selected_experts=candidate, diagnostic_flags=flags,
                         leave_one_game_out=leave_one_out,
                         status="review_diagnostics" if flags else "nll_candidate_selected")
    write_json(root / "selection.json", selection)
    write_json(root / "comparison.json", dict(complete=complete, common_requests=len(common_rows),
                                              summaries=summaries, selection=selection))
    stream = io.StringIO()
    fields = ["experts", "completed", "common_requests", "primary", "delta_vs_full", "token_nll", "perplexity",
              "scoring_seconds", "peak_allocated_bytes"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for c in counts:
        writer.writerow(dict(experts=c, **{k: summaries[str(c)][k] for k in fields[1:]}))
    atomic_bytes(root / "comparison.csv", stream.getvalue().encode())
    # Granular diagnostics use only paired rows; counts expose sparse coverage.
    slices = []
    for dimension, key in [("game", lambda r: r["game"]),
                           ("game_level", lambda r: f"{r['game']}:{r.get('level')}"),
                           ("context_length", lambda r: length_band(r["prompt_tokens"]))]:
        for value in sorted({key(r) for r in common_rows}):
            subset = [r for r in common_rows if key(r) == value]
            for c in counts:
                s = weighted_summary(subset, results[c])
                slices.append(dict(dimension=dimension, value=value, experts=c, requests=len(subset), **s))
    write_json(root / "slices.json", slices)
    write_json(root / "coverage.json", manifest.get("coverage", []))
    payload = dict(counts=counts, summaries=summaries, selection=selection,
                   coverage=manifest.get("coverage", []),
                   samples=[dict(sample_id=r["sample_id"], game=r["game"], level=r.get("level"),
                                 prompt_tokens=r["prompt_tokens"], target_text=r["target_text"],
                                 offsets=r["offsets"], labels=r["labels"], target_ids=r["target_ids"],
                                 losses={str(c): results[c][r["sample_id"]].tolist()
                                         for c in counts if r["sample_id"] in results[c]}) for r in samples],
                   label_names=LABELS)
    js_data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    html = HTML.replace("__DATA__", js_data)
    atomic_bytes(root / "comparison.html", html.encode())
    return selection


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Sol transcript NLL</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172334}
table{border-collapse:collapse}td,th{padding:8px 16px;border-bottom:1px solid #ccd}pre{white-space:pre-wrap;
line-height:1.9;background:#f7f8fc;padding:16px}select{margin:8px}small{color:#567}</style>
<h1>Sol transcript NLL</h1><p id="status"></p><table id="summary"></table>
<p>Teacher forcing; final reply only. NLL in nats/token. Missing results are not zero.</p>
<details><summary>Game / level coverage</summary><table id="coverage"></table></details>
<label>Request <select id="sample"></select></label><label>Experts <select id="model"></select></label>
<label>Display <select id="mode"><option value="nll">NLL (red, capped at 10)</option>
<option value="delta">Delta versus full (red worse / blue better, capped at ±2)</option></select></label>
<p id="info"></p><pre id="tokens"></pre><small>Hover for token losses and labels. Overlapping character offsets
are grouped for display. Full-precision values remain in the NPZ artifacts. Level means level at request time.</small>
<script>const D=__DATA__; const el=id=>document.getElementById(id);
el('status').textContent=`${D.selection.matched_requests}/${D.selection.required_requests} paired requests; ${D.selection.status}`;
const header=el('summary').insertRow();for(const text of ['Experts','Completed','Paired NLL','Δ full']){const th=document.createElement('th');th.textContent=text;header.appendChild(th)}
for(const r of D.coverage){const tr=el('coverage').insertRow();for(const x of [r.game,`level ${r.level}`,`${r.sampled_requests}/${r.population_requests} requests`,r.status])tr.insertCell().textContent=x}
for(const c of D.counts){const s=D.summaries[c];const row=el('summary').insertRow();for(const x of [c,s.completed,s.primary,s.delta_vs_full])row.insertCell().textContent=x===null?'pending':typeof x==='number'&&!Number.isInteger(x)?x.toFixed(5):x;
el('model').add(new Option(c,c))}
D.samples.forEach((s,i)=>el('sample').add(new Option(`${s.game} level ${s.level} — ${s.sample_id}`,i)));
function show(){const s=D.samples[+el('sample').value],m=el('model').value,delta=el('mode').value==='delta',loss=s.losses[m],ref=s.losses[D.counts[0]],chars=Array.from(s.target_text);
el('tokens').textContent='';el('info').textContent=`${s.prompt_tokens} prompt tokens; ${s.target_ids.length} target tokens`;
if(!loss||(delta&&!ref)){el('tokens').textContent='This result is pending.';return}
let cursor=0;for(let i=0;i<s.offsets.length;i++){let [a,b]=s.offsets[i];let ids=[i];
while(i+1<s.offsets.length&&s.offsets[i+1][0]<b){i++;b=Math.max(b,s.offsets[i][1]);ids.push(i)}
if(a>cursor)el('tokens').append(document.createTextNode(chars.slice(cursor,a).join('')));
const span=document.createElement('span'),values=ids.map(j=>loss[j]-(delta?ref[j]:0)),value=values.reduce((a,b)=>a+b,0)/values.length;
span.textContent=b>a?chars.slice(Math.max(a,cursor),b).join(''):`[token ${s.target_ids[i]}]`;
span.title=ids.map(j=>`token ${j}, ${D.label_names[s.labels[j]]}: NLL ${loss[j].toFixed(5)}${ref?', Δ '+(loss[j]-ref[j]).toFixed(5):''}`).join('\\n');
span.style.background=delta?(value<0?`rgba(30,110,210,${Math.min(.8,-value/2)})`:`rgba(230,80,50,${Math.min(.8,value/2)})`):`rgba(230,80,50,${Math.min(.8,value/10)})`;
el('tokens').appendChild(span);cursor=Math.max(cursor,b)}
el('tokens').append(document.createTextNode(chars.slice(cursor).join('')))}
['sample','model','mode'].forEach(id=>el(id).addEventListener('change',show));show();</script></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    generate(parser.parse_args().out)
