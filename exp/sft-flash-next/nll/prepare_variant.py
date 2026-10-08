"""Freeze changed final thinking on the same panel, prompts and train-only maps."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

from common import COUNTS, LABELS, digest, file_hash, read_json, write_json
from protocol import budget
from run import validate_bundle

CODE_ONLY_LINE = "- The only tool is `python`; call it with one ephemeral `code` string."


def audit(sample, original):
    """Fail closed on changed context, code, schema or Sol-only tool fields."""
    expected = copy.deepcopy(original)
    thinking = sample["messages"][-1].get("reasoning_content")
    if not isinstance(thinking, str) or not thinking.strip():
        raise ValueError("Missing generated final thinking")
    expected["messages"][-1]["reasoning_content"] = thinking
    expected["thinking_source"] = "think_gen-refine2"
    if expected != sample:
        raise ValueError("Variant changed something other than final thinking/source")
    python_tools = [t["function"] for t in sample["tools"]
                    if t.get("function", {}).get("name") == "python"]
    if len(python_tools) != 1:
        raise ValueError("Expected exactly one Python tool")
    params = python_tools[0]["parameters"]
    if set(params.get("properties", {})) != {"code"} or params.get("required") != ["code"]:
        raise ValueError("Python schema contains Sol-only fields; fix the dataset first")
    systems = [m["content"] for m in sample["messages"] if m.get("role") == "system"]
    if not systems or any(not isinstance(s, str) or CODE_ONLY_LINE not in s for s in systems):
        raise ValueError("System prompt must instruct code-only Python calls")
    history, final = 0, 0
    for i, message in enumerate(sample["messages"]):
        for call in message.get("tool_calls", []):
            fn = call["function"]
            if fn.get("name") != "python" or set(fn.get("arguments", {})) != {"code"}:
                raise ValueError("Historical/final Python call contains extra fields")
            if not isinstance(fn["arguments"]["code"], str):
                raise ValueError("Python code must be a string")
            if i == len(sample["messages"])-1:
                final += 1
            else:
                history += 1
    if final != 1:
        raise ValueError("Expected one final Python call")
    return dict(history_calls=history, final_calls=final, python_schemas=1)


def source_order(sample, original):
    """Retain logical values but restore render-sensitive dictionary order."""
    ordered = copy.deepcopy(original)
    ordered["messages"][-1]["reasoning_content"] = sample["messages"][-1]["reasoning_content"]
    ordered["thinking_source"] = sample["thinking_source"]
    if ordered != sample:
        raise ValueError("Dictionary-order restoration changed logical values")
    return ordered


def prepare_variant(source_bundle, dataset, out, revision):
    from data import encode, load_processor

    source_bundle, dataset, out = map(Path, (source_bundle, dataset, out))
    base = validate_bundle(source_bundle)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Variant output must be empty; never overwrite a frozen panel")
    payload = dataset / "requests.jsonl"
    pointer = (dataset / "requests.jsonl.dvc").read_text()
    match = re.search(r"\bmd5:\s*([0-9a-f]{32})\b", pointer)
    if not match or file_hash(payload, "md5") != match[1]:
        raise ValueError("Generated-thinking DVC payload checksum mismatch")
    rows = [json.loads(line) for line in payload.read_text().splitlines() if line.strip()]
    index = read_json(dataset / "index.json")
    ids = [f"{r['game']}-r{r['request_index']}" for r in rows]
    if ids != [r["sample_id"] for r in base["samples"]] or len(rows) != 30:
        raise ValueError("Variant must contain the same 30 IDs in frozen evaluation order")
    if [r["sample_id"] for r in index] != ids:
        raise ValueError("Variant index identity/order mismatch")
    raw = payload.read_bytes()
    for row in index:
        line = raw[row["offset"]:row["offset"]+row["length"]]
        if hashlib.sha256(line).hexdigest() != row["line_sha256"]:
            raise ValueError("Variant index byte range/checksum mismatch")
    provenance = read_json(dataset / "provenance.json")
    if provenance["counts"] != {"source": 30, "included": 30, "excluded": 0}:
        raise ValueError("Variant provenance does not account for all 30 requests")
    processor = load_processor(source_bundle / "processor")
    for name in ("processor", "maps"):
        shutil.copytree(source_bundle / name, out / name)
    shutil.copyfile(source_bundle / "folds.json", out / "folds.json")
    records, audits = [], []
    for sample, old_row in zip(rows, base["samples"]):
        original = read_json(source_bundle / old_row["path"])
        checked = audit(sample, original)
        # HF's tool | tojson renderer preserves dictionary insertion order.
        # The repository export sorted keys; restore the source order while
        # preserving every value so only the final thinking changes tokens.
        sample = source_order(sample, original)
        old_enc, old_annotation = encode(processor, original)
        if digest(old_annotation) != digest({k: old_row[k] for k in old_annotation}):
            raise ValueError("Source bundle reprocessing differs from its frozen annotations")
        enc, annotation = encode(processor, sample)
        p = old_row["prompt_tokens"]
        if annotation["prompt_tokens"] != p or not enc["input_ids"][:, :p].equal(old_enc["input_ids"][:, :p]):
            raise ValueError("Variant changed processor-expanded prompt tokens")
        for key in ("pixel_values", "image_grid_thw"):
            a, b = enc.get(key), old_enc.get(key)
            if (a is None) != (b is None) or (a is not None and not a.equal(b)):
                raise ValueError("Variant changed processor-expanded images")
        code_label = LABELS.index("tool_code")
        old_code = [token for token, label in zip(old_annotation["target_ids"], old_annotation["labels"])
                    if label == code_label]
        new_code = [token for token, label in zip(annotation["target_ids"], annotation["labels"])
                    if label == code_label]
        if old_code != new_code:
            raise ValueError("Variant changed scored Python-code token IDs")
        if annotation["total_tokens"] > 139264 or not annotation["category_counts"]["thinking"]:
            raise ValueError("Variant exceeds full context capacity or has no thinking target")
        path = out / old_row["path"]
        write_json(path, sample)
        record = dict(old_row, **annotation, sample_sha256=file_hash(path))
        records.append(record)
        audits.append(dict(sample_id=old_row["sample_id"], **checked,
                           prompt_sha256=digest(enc["input_ids"][0, :p].tolist()),
                           prompts_images_code_unchanged=True))
        write_json(out / "annotations" / f"{record['sample_id']}.json", annotation)
        print(f"prepared {record['sample_id']}: {p} + {annotation['target_tokens']} tokens", flush=True)
    manifest = copy.deepcopy(base)
    manifest.update(samples=records, dataset_revision=revision,
                    dataset_md5={"requests.jsonl": match[1]},
                    variant={"name": "sol-nll-fold0-30-genthink",
                             "source_manifest_sha256": base["manifest_sha256"],
                             "dataset_sha256": file_hash(payload), "provenance": provenance,
                             "index_sha256": file_hash(dataset / "index.json"),
                             "transformation": "final thinking only; source-bundle dictionary order restored for token-identical prompts; no field changes or resampling",
                             "tool_contract_audit": audits})
    manifest["processed_tokens"] = sum(r["total_tokens"] for r in records)*len(COUNTS)
    manifest["estimated_minutes"] = {str(rate): round(manifest["processed_tokens"]/rate/60+30, 1)
                                     for rate in (500, 850, 950)}
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = digest(manifest)
    write_json(out / "manifest.json", manifest)
    validate_bundle(out)
    print(json.dumps(dict(manifest_sha256=manifest["manifest_sha256"], staged_budget=budget(manifest)), indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare_variant(args.source_bundle, args.dataset, args.out, args.revision)
