"""Verify complete same-context panels and compare their different final replies."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from common import LABELS, atomic_bytes, digest, length_band, read_json, verify_manifest, write_json
from protocol import gate
from report import weighted_summary
from results import load_result


def validate_pair(source, variant):
    for manifest in (source, variant):
        verify_manifest(manifest)
        if len(manifest["samples"]) != 30:
            raise ValueError("Comparison requires the complete frozen 30-request panel")
    proof = variant.get("variant", {})
    if proof.get("source_manifest_sha256") != source["manifest_sha256"]:
        raise ValueError("Variant does not derive from this source panel")
    for key in ("processor_files", "maps", "fold_sha256", "validation_games", "training_games"):
        if source[key] != variant[key]:
            raise ValueError(f"Panel provenance differs: {key}")
    audits = {r["sample_id"]: r for r in proof["tool_contract_audit"]}
    source_rows, variant_rows = source["samples"], variant["samples"]
    if len({r["sample_id"] for r in source_rows}) != 30 or set(audits) != {r["sample_id"] for r in source_rows}:
        raise ValueError("Duplicate request or incomplete context audit")
    if [r["sample_id"] for r in source_rows] != [r["sample_id"] for r in variant_rows]:
        raise ValueError("Request identity/order differs")
    for a, b in zip(source_rows, variant_rows):
        for key in ("game", "request_index", "stratum", "weight", "prompt_tokens", "images"):
            if a[key] != b[key]:
                raise ValueError(f"Request context/sampling differs: {a['sample_id']}: {key}")
        if not audits[a["sample_id"]].get("prompts_images_code_unchanged"):
            raise ValueError("Missing processor-verified prompt/image/code equivalence")
        code = LABELS.index("tool_code")
        ids = lambda r: [v for v, label in zip(r["target_ids"], r["labels"]) if label == code]
        if ids(a) != ids(b):
            raise ValueError("Python target token IDs differ")


def loss_metrics(row, losses):
    labels = np.asarray(row["labels"])
    values = np.asarray(losses, dtype=np.float64)
    metrics = dict(mean_nll=float(values.mean()), target_tokens=len(values))
    for i, name in enumerate(LABELS):
        chosen = values[labels == i]
        metrics[name + "_nll"] = float(chosen.mean()) if len(chosen) else None
        metrics[name + "_tokens"] = len(chosen)
    output = values[labels != LABELS.index("thinking")]
    metrics["output_nll"] = float(output.mean()) if len(output) else None
    return metrics


def compare(source_root, variant_root, out, reference_metric="categorywise"):
    roots = {"source": Path(source_root), "genthink": Path(variant_root)}
    manifests = {k: read_json(v / "manifest.json") for k, v in roots.items()}
    runs = {k: read_json(v / "run.json") for k, v in roots.items()}
    validate_pair(manifests["source"], manifests["genthink"])
    for run in runs.values():
        if digest(run["config"]) != run["identity"]:
            raise ValueError("Run configuration checksum mismatch")
    for key in ("model", "scoring_numerics", "chunk", "lm_block", "dtype", "kernel_options",
                "versions", "gpu", "capability", "cuda", "numerical_tolerance"):
        if runs["source"]["config"].get(key) != runs["genthink"]["config"].get(key):
            raise ValueError(f"Scoring configurations differ: {key}")
    out = Path(out)
    summaries, pairs, slices = {}, [], []
    for count in (512, 256):
        losses = {}
        for panel, manifest in manifests.items():
            losses[panel] = {}
            for row in manifest["samples"]:
                result = load_result(roots[panel], runs[panel]["identity"], count, row)
                if result is None:
                    raise ValueError(f"Incomplete panel: {panel}, {count}, {row['sample_id']}")
                losses[panel][row["sample_id"]] = result[1]
            summaries.setdefault(panel, {})[str(count)] = weighted_summary(manifest["samples"], losses[panel])
        for dimension, key in (("game", lambda row: row["game"]),
                               ("context_length", lambda row: length_band(row["prompt_tokens"]))):
            for group in sorted({key(r) for r in manifests["source"]["samples"]}):
                entry = dict(dimension=dimension, value=group, experts=count)
                for panel, manifest in manifests.items():
                    rows = [r for r in manifest["samples"] if key(r) == group]
                    entry[panel] = weighted_summary(rows, losses[panel])
                    entry["requests"] = len(rows)
                slices.append(entry)
        for a, b in zip(manifests["source"]["samples"], manifests["genthink"]["samples"]):
            sid = a["sample_id"]
            x, y = losses["source"][sid], losses["genthink"][sid]
            left, right = loss_metrics(a, x), loss_metrics(b, y)
            deltas = {k: right[k] - left[k] if left[k] is not None and right[k] is not None else None
                      for k in left if k.endswith("nll")}
            pairs.append(dict(sample_id=sid, game=a["game"], experts=count,
                              prompt_tokens=a["prompt_tokens"], weight=a["weight"],
                              source=left, genthink=right, delta_genthink_minus_source=deltas))
            # Reasoning text differs: only the identical Python tokens can be aligned.
            code = LABELS.index("tool_code")
            ca, cb = np.asarray(a["labels"]) == code, np.asarray(b["labels"]) == code
            array = io.BytesIO()
            np.savez_compressed(array, token_ids=np.asarray(a["target_ids"])[ca],
                                source_nll=x[ca], genthink_nll=y[cb],
                                delta=y[cb].astype(np.float64)-x[ca].astype(np.float64))
            atomic_bytes(out / "code-token-diffs" / str(count) / (sid + ".npz"), array.getvalue())
    def value(panel, count):
        summary = summaries[panel][str(count)]
        return summary["primary"] if reference_metric == "primary" else summary["categories"]["thinking"]["nll"]
    if reference_metric == "categorywise":
        baseline = {panel: summaries[panel]["512"]["categories"] for panel in roots}
        reference = "genthink" if all(baseline["genthink"][key]["nll"] < baseline["source"][key]["nll"]
                                      for key in ("thinking", "tool_code")) else "source"
        component_gates = {panel: {key: gate(summaries[panel]["512"]["categories"][key]["nll"],
                                           summaries[panel]["256"]["categories"][key]["nll"])
                                  for key in ("thinking", "tool_code")} for panel in roots}
        def combined(panel):
            components = component_gates[panel]
            return dict(status="scan_required" if any(v["status"] == "scan_required" for v in components.values())
                        else "stop_at_256", components=components, threshold_relative=.05,
                        metric="sampling-weighted thinking and Python-code NLL, independently")
        panel_gates = {panel: combined(panel) for panel in roots}
        delta = {key: baseline["genthink"][key]["nll"]-baseline["source"][key]["nll"]
                 for key in ("thinking", "tool_code")}
    else:
        reference = "genthink" if value("genthink", 512) < value("source", 512) else "source"
        panel_gates = {k: gate(value(k, 512), value(k, 256)) for k in roots}
        delta = value("genthink", 512)-value("source", 512)
    decision = dict(reference_metric=reference_metric, reference_panel=reference,
                    baseline_comparison_at_experts=512,
                    genthink_minus_source=delta,
                    pruning_gate=panel_gates[reference],
                    panel_gates=panel_gates,
                    requests=30, paired_panel_model_results=60,
                    context_equivalence="processor-verified frozen variant audit; matching IDs/weights/maps/code IDs",
                    limitation="Thinking targets differ in content and length; means are paired by request, not by thinking token.")
    decision["selected_experts"] = 256 if decision["pruning_gate"]["status"] == "stop_at_256" else None
    decision["status"] = "accepted_at_256" if decision["selected_experts"] else "intermediate_scan_required"
    decision["paired_wins"] = {
        str(count): {key: sum(r["delta_genthink_minus_source"][key] < 0
                             for r in pairs if r["experts"] == count)
                     for key in ("thinking_nll", "tool_code_nll")} for count in (512, 256)}
    write_json(out / "comparison.json", dict(run_identities={k: v["identity"] for k, v in runs.items()},
                                              summaries=summaries, decision=decision, requests=pairs))
    write_json(out / "decision.json", decision)
    write_json(out / "slices.json", slices)
    stream = io.StringIO()
    fields = ["sample_id", "game", "experts", "prompt_tokens", "weight"]
    fields += [prefix + metric for prefix in ("source_", "genthink_", "delta_")
               for metric in ("mean_nll", "thinking_nll", "tool_code_nll", "output_nll")]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in pairs:
        flat = {k: row[k] for k in fields[:5]}
        for prefix, values in (("source_", row["source"]), ("genthink_", row["genthink"]),
                               ("delta_", row["delta_genthink_minus_source"])):
            flat.update({prefix+k: v for k, v in values.items() if prefix+k in fields})
        writer.writerow(flat)
    atomic_bytes(out / "paired-requests.csv", stream.getvalue().encode())
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-metric", choices=["primary", "thinking", "categorywise"], default="categorywise")
    args = parser.parse_args()
    print(compare(args.source, args.variant, args.out, args.reference_metric))
