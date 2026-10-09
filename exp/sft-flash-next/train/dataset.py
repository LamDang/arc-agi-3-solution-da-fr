"""Immutable, offset-indexed full-context requests with game-disjoint splits."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "nll"))
from data import encode, load_processor


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def fold_games(folds, validation_fold):
    validation = next(set(f["game_ids"]) for f in folds["folds"] if f["fold"] == validation_fold)
    known = {g for f in folds["folds"] for g in f["game_ids"]}
    return known, validation


def inject_thinking(sample, generated):
    """Same refinement export as panel30; preserve context and dictionary order."""
    key = f"{sample['game']}_p0#{sample['request_index']}"
    row = generated.get(key)
    if not row or row.get("status") != "ok" or not isinstance(row.get("thinking"), str) or not row["thinking"].strip():
        raise ValueError(f"Missing successful generated thinking: {key}; no rows are silently skipped")
    out = copy.deepcopy(sample)
    out["messages"][-1]["reasoning_content"] = row["thinking"]
    out["thinking_source"] = "think_gen-refine2"
    return out


def validate_sample(sample, known):
    if sample.get("game") not in known:
        raise ValueError(f"Unknown game: {sample.get('game')}")
    ri = sample.get("request_index")
    if not isinstance(ri, int) or isinstance(ri, bool) or ri < 0:
        raise ValueError("request_index must be a nonnegative integer")
    if sample.get("thinking_source") != "think_gen-refine2":
        raise ValueError("Expected thinking_source=think_gen-refine2; supply --generated-dir to join refinement output")
    final = sample.get("messages", [{}])[-1]
    if final.get("role") != "assistant" or not isinstance(final.get("reasoning_content"), str) or not final["reasoning_content"].strip():
        raise ValueError("Final assistant generated thinking is missing")
    if sample.get("chat_template_kwargs", {}).get("preserve_thinking") is not True:
        raise ValueError("preserve_thinking must explicitly be true")
    python = [t["function"] for t in sample.get("tools", []) if t.get("function", {}).get("name") == "python"]
    if (len(python) != 1 or set(python[0]["parameters"].get("properties", {})) != {"code"}
            or python[0]["parameters"].get("required") != ["code"]):
        raise ValueError("Expected code-only Python schema")
    for m in sample["messages"]:
        for call in m.get("tool_calls", []):
            fn = call.get("function", {})
            args = fn.get("arguments")
            if fn.get("name") != "python" or not isinstance(args, dict) or set(args) != {"code"} or not isinstance(args["code"], str):
                raise ValueError("Historical and final Python arguments must be code-only mappings")


class Requests:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        claimed = self.manifest["sha256"]
        if digest({k: v for k, v in self.manifest.items() if k != "sha256"}) != claimed:
            raise ValueError("Dataset manifest checksum mismatch")
        for name, checksum in self.manifest["files"].items():
            if file_hash(self.root / name) != checksum:
                raise ValueError(f"Dataset file changed: {name}")
        self.processor = load_processor(self.root / "processor")
        self.rows = self.manifest["rows"]

    def get(self, row):
        with open(self.root / "requests.jsonl", "rb") as stream:
            stream.seek(row["offset"])
            raw = stream.read(row["length"])
        if hashlib.sha256(raw).hexdigest() != row["line_sha256"]:
            raise ValueError("Indexed request checksum mismatch")
        sample = json.loads(raw)
        if f"{sample['game']}-r{sample['request_index']}" != row["sample_id"]:
            raise ValueError("Indexed request identity mismatch")
        enc, annotation = encode(self.processor, sample)
        if digest(annotation) != row["annotation_sha256"]:
            raise ValueError("Processor output changed since preparation")
        return enc, annotation

    def split(self, name):
        return [r for r in self.rows if r["split"] == name]
