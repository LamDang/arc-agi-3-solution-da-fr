"""Build the generated-thinking variant of the frozen fold-0 panel.

Each sample is the same request as data/sol-nll-fold0-30, with the final
assistant turn's `reasoning_content` replaced by the think_gen pipeline's
generated (refine-2) thinking. The python tool call stays code-only (no
description or rationale), exactly as in the source. Records whose generated
thinking did not pass the generator's length/leak gate (no status=="ok") are
excluded and listed in provenance.json.

    python data/sol-nll-fold0-30-genthink/build.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

SRC = Path("data/sol-nll-fold0-30")
OUT = Path("data/sol-nll-fold0-30-genthink")
GEN = Path("ARC3-Inference/runs/think-sol-nll30/refine2")

gen = {}
for p in sorted(GEN.glob("*.jsonl")):
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("thinking") and r.get("status") == "ok":
                gen[r["key"]] = r

frozen = [json.loads(l) for l in (SRC / "requests.jsonl").read_text().splitlines() if l.strip()]
src_index = {row["sample_id"]: row for row in json.loads((SRC / "index.json").read_text())}

OUT.mkdir(exist_ok=True)
out_lines, index, excluded, offset = [], [], [], 0
for o in frozen:
    game, ri = o["game"], o["request_index"]
    key, sid = f"{game}_p0#{ri}", f"{game}-r{ri}"
    g = gen.get(key)
    if not g:
        excluded.append({"sample_id": sid, "key": key,
                         "reason": "no status==ok generated thinking (rejected by length/leak gate)"})
        continue
    obj = json.loads(json.dumps(o))
    last = max(i for i, m in enumerate(obj["messages"]) if m.get("role") == "assistant")
    old = obj["messages"][last].get("reasoning_content") or ""
    obj["messages"][last]["reasoning_content"] = g["thinking"]
    obj["thinking_source"] = "think_gen-refine2"
    line = json.dumps(obj, ensure_ascii=False)
    b = (line + "\n").encode()
    index.append({
        "sample_id": sid, "game": game, "request_index": ri, "key": key,
        "offset": offset, "length": len(b),
        "orig_reasoning_chars": len(old),
        "generated_reasoning_chars": len(g["thinking"]),
        "generated_words": len(g["thinking"].split()),
        "prompt_tokens_frozen": src_index[sid]["prompt_tokens"],
        "line_sha256": hashlib.sha256(b).hexdigest(),
    })
    out_lines.append(line)
    offset += len(b)

(OUT / "requests.jsonl").write_text("\n".join(out_lines) + "\n")
(OUT / "index.json").write_text(json.dumps(index, indent=1))

head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
src_md5 = ""
for l in (SRC / "requests.jsonl.dvc").read_text().splitlines():
    if "md5:" in l:
        src_md5 = l.split("md5:")[1].strip()
prov = {
    "dataset": "sol-nll-fold0-30-genthink",
    "derived_from": {"dataset": "data/sol-nll-fold0-30", "requests_md5": src_md5},
    "thinking": {
        "source": "think_gen refine-2 (generated, 2-pass judge+refine)",
        "run_dir": str(GEN),
        "manifest": "ARC3-Inference/experiments/teacher-reasoning/evalset/sol-nll-fold0-30.json",
        "pipeline_commit": head,
        "replaced": "final assistant turn's reasoning_content; context and code-only tool call unchanged",
    },
    "counts": {"source": len(frozen), "included": len(index), "excluded": len(excluded)},
    "excluded": excluded,
}
(OUT / "provenance.json").write_text(json.dumps(prov, indent=1))
print(f"wrote {len(index)} samples, excluded {len(excluded)}: {[e['sample_id'] for e in excluded]}")
print("requests.jsonl size KB:", (OUT / 'requests.jsonl').stat().st_size // 1024)
