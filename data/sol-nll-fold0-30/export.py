"""Export or verify the frozen Sol NLL panel using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_MANIFEST = "f75451395a05abaee253543275b901a471b98adbf7112a5b48bac5e6faf0960c"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def verify(out):
    index = json.loads((out / "index.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    payload = (out / "requests.jsonl").read_bytes()
    assert sha(payload) == provenance["payload_sha256"], "Payload checksum mismatch"
    assert sha((out / "index.json").read_bytes()) == provenance["index_sha256"], "Index checksum mismatch"
    assert len(index) == len(payload.splitlines()) == 30
    assert index[-1]["offset"] + index[-1]["length"] == len(payload)
    offset = 0
    for row in index:
        assert row["offset"] == offset
        raw = payload[offset:offset + row["length"]]
        assert raw.endswith(b"\n") and sha(raw) == row["line_sha256"]
        sample = json.loads(raw)
        assert (sample["game"], sample["request_index"]) == (row["game"], row["request_index"])
        assert sample["messages"][-1]["role"] == "assistant"
        offset += row["length"]
    games = provenance["validation_games"]
    assert set(row["game"] for row in index) == set(games)
    for game in games:
        for stratum in range(3):
            assert sum(row["game"] == game and row["stratum"] == stratum for row in index) == 2
    print(f"Verified 30 full requests, five games, payload SHA256 {sha(payload)}")


def export(bundle, out):
    manifest = json.loads((bundle / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert manifest["manifest_sha256"] == sha(encoded(body)) == EXPECTED_MANIFEST
    assert manifest["real_data"] and len(manifest["samples"]) == 30
    out.mkdir(parents=True, exist_ok=True)
    for name in ("requests.jsonl", "index.json", "provenance.json"):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite {out / name}")
    rows, chunks, offset = [], [], 0
    for selected in manifest["samples"]:
        raw = (bundle / selected["path"]).read_bytes()
        assert sha(raw) == selected["sample_sha256"]
        sample = json.loads(raw)
        line = encoded(sample) + b"\n"
        assert json.loads(line) == sample  # Images/history/code remain identical.
        row = {k: selected[k] for k in (
            "sample_id", "game", "request_index", "level", "stratum", "draw",
            "population", "inclusion_probability", "weight", "prompt_tokens",
            "target_tokens", "total_tokens", "images", "input_sha256", "target_sha256")}
        row.update(offset=offset, length=len(line), line_sha256=sha(line),
                   frozen_sample_sha256=selected["sample_sha256"],
                   source_line=selected["line"], source_offset=selected["offset"],
                   source_length=selected["length"])
        rows.append(row)
        chunks.append(line)
        offset += len(line)
    payload = b"".join(chunks)
    (out / "requests.jsonl").write_bytes(payload)
    write_json(out / "index.json", rows)
    write_json(out / "provenance.json", dict(
        schema=1, name="sol-nll-fold0-30", created_utc="2026-10-08",
        purpose="frozen held-out teacher-transcript NLL evaluation; not training data",
        source_repository="LamDang/arc-agi-3-solution-da-fr",
        source_dataset="data/sft-gpt61sol-features-25games",
        source_dataset_revision=manifest["dataset_revision"], source_dataset_md5=manifest["dataset_md5"],
        preparation_commit="d162b9a", frozen_manifest_sha256=EXPECTED_MANIFEST,
        fold=manifest["fold"], fold_sha256=manifest["fold_sha256"],
        seed=manifest["seed"], sampling=manifest["sampling"], strata=manifest["strata"],
        validation_games=manifest["validation_games"], processor_revision=manifest["model_revision"],
        processor_files_sha256=manifest["processor_files"], expert_counts=manifest["counts"],
        transformation="canonical JSON serialization only; every source request object is unchanged",
        row_order="frozen manifest order; interleaved game rounds, draw then stratum",
        scoring="full multimodal context; final assistant reply only; no truncation",
        requests=30, payload_bytes=len(payload), payload_sha256=sha(payload),
        index_sha256=sha((out / "index.json").read_bytes()), export_script_sha256=sha(Path(__file__).read_bytes()),
        prompt_tokens=sum(r["prompt_tokens"] for r in rows), target_tokens=sum(r["target_tokens"] for r in rows),
        processed_tokens_all_candidates=manifest["processed_tokens"],
        coverage=manifest["coverage"], gpu_started=False))
    verify(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Verified frozen panel to export; omit to verify existing dataset")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.bundle:
        export(args.bundle, args.out)
    else:
        verify(args.out)
