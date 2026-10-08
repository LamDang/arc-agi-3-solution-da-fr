"""Build a private, offline Quarto HTML report from both verified token-loss panels."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nll"))
from common import LABELS, file_hash, read_json, write_json
from compare_panels import compare, loss_metrics
from results import load_result


def verify_offsets(row):
    cursor = 0
    if len(row["offsets"]) != row["target_tokens"]:
        raise ValueError("Offset/token length differs")
    for start, end in row["offsets"]:
        if start != cursor or end <= start or end > len(row["target_text"]):
            raise ValueError("Cannot display exact token spans: overlapping, empty or missing character offsets")
        cursor = end
    if cursor != len(row["target_text"]):
        raise ValueError("Token spans do not reconstruct the complete final reply")


def build(source, variant, out, quarto):
    source, variant, out = map(Path, (source, variant, out))
    out.mkdir(parents=True, exist_ok=True)
    compare(source, variant, out / "comparison", reference_metric="categorywise")
    comparison = read_json(out / "comparison/comparison.json")
    manifests = {"source": read_json(source / "manifest.json"), "genthink": read_json(variant / "manifest.json")}
    runs = {"source": read_json(source / "run.json"), "genthink": read_json(variant / "run.json")}
    roots = {"source": source, "genthink": variant}
    env = dict(os.environ)
    env.setdefault("XDG_CACHE_HOME", str(out.resolve() / ".cache"))
    env.setdefault("DENO_DIR", str(out.resolve() / ".cache/deno"))
    version = subprocess.check_output([quarto, "--version"], text=True, env=env).strip()
    requests = []
    for index, (a, b) in enumerate(zip(manifests["source"]["samples"], manifests["genthink"]["samples"]), 1):
        request = {key: a[key] for key in ("sample_id", "game", "request_index", "prompt_tokens", "level")}
        request["order"] = index
        for panel, row in (("source", a), ("genthink", b)):
            verify_offsets(row)
            values, metrics = {}, {}
            for count in (512, 256):
                result = load_result(roots[panel], runs[panel]["identity"], count, row)
                if result is None:
                    raise ValueError("Missing verified token-loss array")
                values[str(count)] = result[1].tolist()
                metrics[str(count)] = loss_metrics(row, result[1])
            request[panel] = dict(text=row["target_text"], ids=row["target_ids"], labels=row["labels"],
                                  offsets=row["offsets"], losses=values, metrics=metrics)
        requests.append(request)
    payload = dict(requests=requests, labels=list(LABELS), summaries=comparison["summaries"],
                   identities=comparison["run_identities"], decision=comparison["decision"], quarto_version=version)
    # JSON script data is inert; escape '<' so teacher/code text cannot close the script element.
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    (out / "payload.html").write_text('<script type="application/json" id="nll-data">' + encoded +
                                      '</script>\n<script src="report.js"></script>\n')
    here = Path(__file__).parent
    for name in ("report.qmd", "report.css", "report.js"):
        shutil.copyfile(here / name, out / name)
    subprocess.run([quarto, "render", "report.qmd", "--to", "html"], cwd=out, check=True, env=env)
    html = out / "report.html"
    provenance = dict(quarto_version=version, requests=30, completed_evaluations=120,
                      per_token_losses=72932, default_experts=256, default_shared_scale_nats=[0, 6],
                      identical_scale_across="panels, requests, categories and expert counts; adjustable only globally",
                      exact_one_token_per_highlight=True, full_replies_reconstructed=True,
                      run_identities=payload["identities"], report_sha256=file_hash(html), report_bytes=html.stat().st_size,
                      sources={name: file_hash(here/name) for name in ("build.py", "report.qmd", "report.css", "report.js")})
    write_json(out / "provenance.json", provenance)
    print(html)
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "variant", "out"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--quarto", default="quarto")
    args = parser.parse_args()
    build(args.source, args.variant, args.out, args.quarto)
