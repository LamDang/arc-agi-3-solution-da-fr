#!/usr/bin/env bash
# Rebuild the stuck-detection tables and score every detector against labels.json.
# Needs the five runs pulled and unpacked (see exp/stuck-detection.md). Usage: run.sh <out_dir>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:?usage: run.sh <out_dir>}"; mkdir -p "$out"; cd "$out"
python3 "$here/load.py" > /dev/null      # games.pkl: per-action boards, levels, time, tokens
python3 "$here/features.py" > /dev/null  # feats.pkl: windowed rates, resets, effort
python3 "$here/near.py" > /dev/null      # near.pkl: counter-tolerant board revisits
python3 "$here/backstart.py" > /dev/null # backstart.pkl: returns to the level's start board
python3 "$here/signals.py" > /dev/null   # table.pkl: everything per action
python3 "$here/evaluate.py" "$here/labels.json"
