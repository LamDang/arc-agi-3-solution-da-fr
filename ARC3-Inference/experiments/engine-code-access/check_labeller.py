"""Check the thinking-topic labeller of scripts/token_breakdown.py.

Run 20261004_135539's analyses/ holds 54 thinking chunks labelled by hand and
labels by Claude Haiku subagents for all 7,856 chunks. This labels the hand
sample plus 346 random chunks with token_breakdown.py's labeller, saves the
labels to data/labeller_check.json, and prints how often the labellers agree. Run
from ARC3-Inference/ after `dvc pull runs/20261004_135539.dvc`:

    uv run --no-sync python experiments/engine-code-access/check_labeller.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ANALYSES = Path("runs/20261004_135539/analyses")
OUT = Path(__file__).resolve().parent / "data" / "labeller_check.json"
sys.path[:0] = [str(ANALYSES), "scripts"]

import thinking_categories  # noqa: E402
from token_breakdown import LABEL_BATCH, _label_batch  # noqa: E402

LETTER = {"mechanics": "M", "planning": "P", "tooling": "C", "other": "O"}


def _chunk_texts() -> dict[str, str]:
    # Regenerates the chunks as thinking_categories.py numbered them.
    texts: dict[str, str] = {}
    for game_run in thinking_categories._responses():
        transcript = thinking_categories.RUN_DIR / "transcripts" / f"{game_run}.txt"
        for block in thinking_categories._thinking_blocks(transcript):
            for chunk in thinking_categories._chunks(block):
                texts[f"c{len(texts):05d}"] = chunk
    return texts


def _tsv(path: Path) -> dict[str, str]:
    return dict(line.split("\t") for line in path.read_text(encoding="utf-8").splitlines())


def _agreement(names: tuple[str, str], first: dict[str, str], second: dict[str, str]) -> None:
    pairs = [(first[c], second[c]) for c in first if c in second]
    same = sum(a == b for a, b in pairs)
    print(
        f"{names[0]} vs {names[1]}: {same}/{len(pairs)} = {same / len(pairs):.0%} agree; "
        f"{names[0]} {dict(Counter(a for a, _ in pairs))}, "
        f"{names[1]} {dict(Counter(b for _, b in pairs))}"
    )


def main() -> None:
    hand = _tsv(ANALYSES / "thinking_labels_check.tsv")
    prior = _tsv(ANALYSES / "thinking_labels.tsv")
    if OUT.exists():
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        sample, labels = saved["sample"], saved["labels"]
    else:
        texts = _chunk_texts()
        random.seed(7)
        sample = sorted(set(hand) | set(random.sample(sorted(prior), 346)))
        items = [(chunk_id, texts[chunk_id]) for chunk_id in sample]
        labels, cost = {}, 0.0
        for start in range(0, len(items), LABEL_BATCH):
            result, spent = _label_batch(
                items[start : start + LABEL_BATCH], os.environ["OPENROUTER_API_KEY"]
            )
            labels.update({chunk_id: LETTER[item["topic"]] for chunk_id, item in result.items()})
            cost += spent
        print(f"labelled {len(labels)} chunks for ${cost:.3f}")
        OUT.write_text(json.dumps({"sample": sample, "labels": labels}, indent=1) + "\n")
    _agreement(("hand", "script"), hand, labels)
    _agreement(("subagents", "script"), {c: prior[c] for c in sample if c in prior}, labels)
    _agreement(("hand", "subagents"), hand, prior)


if __name__ == "__main__":
    main()
