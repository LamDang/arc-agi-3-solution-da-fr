#!/usr/bin/env bash
# gpt-6.1-sol on the 25 official games with the four new harness settings
# (exp/gpt61sol-features-25games.md), then up to two reruns of every game that
# scored under 100. Each attempt is its own run directory and is kept:
#   runs/gpt61sol-features-25games          all 25 games
#   runs/gpt61sol-features-25games-retry1   the games under 100 in the first
#   runs/gpt61sol-features-25games-retry2   the games still under 100
# An attempt whose score.json exists is skipped, so the script can be started
# again after an interruption; an attempt directory left without score.json
# stops it (resume that attempt by hand with RESUME_FROM).
#
#   cd ARC3-Inference && bash experiments/gpt61sol-features/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

BASE=runs/gpt61sol-features-25games
# Long games first, so they start in the first group of 10.
GAMES=sk48,lf52,bp35,wa30,dc22,re86,s5i5,su15,g50t,tr87,ls20,sb26,ka59,tu93,sc25,cd82,m0r0,vc33,sp80,lp85,ar25,cn04,r11l,tn36,ft09

# The baseline runs' settings (base-gpt61sol-20games), the 300K output cap,
# and the four settings under test.
COMMON=(
  --make CONFIG_PATH=configs/inference.openai.json --make MODEL=gpt-6.1-sol
  --make CONCURRENT_JOBS=10
  --make MAX_GENERATED_TOKENS_PER_GAME=300000 --make MAX_RUNTIME_MINUTES=240
  --env LOCAL_ANALYZER_MAX_OUTPUT=0 --env OPENAI_REASONING_EFFORT=xhigh
  --env OPENAI_REASONING_SUMMARY=detailed
  --env ARC3_SEND_REASONING_DETAILS=1 --env ARC3_OPENAI_PRICING=2,0.1,2.5,10
  --env ARC3_PYTHON_RATIONALE=1
  --env ARC3_NOTE_COMPACTION_TOKENS=120000 --env ARC3_NOTE_COMPACTION_KEEP_TURNS=10
  --env ARC3_NO_BUDGET_BURN=1
  --env ARC3_REPEAT_HINT=1 --env ARC3_REPEAT_HINT_POSITIONS=3
  --env ARC3_REPEAT_HINT_VISITS=3 --env ARC3_REPEAT_HINT_COOLDOWN=10
)

attempt() {  # run_dir games
  local dir=$1 games=$2
  if [ -f "$dir/score.json" ]; then
    echo "run.sh: $dir already scored, skipped"
    return 0
  fi
  if [ -e "$dir" ]; then
    echo "run.sh: $dir exists without score.json; resume it by hand" >&2
    exit 1
  fi
  echo "run.sh: $(date -u +%FT%TZ) starting $dir with $games"
  uv run --no-sync python scripts/dvc_eval.py --run-dir "$dir" \
    --metrics "$dir.metrics.json" --make GAME="$games" "${COMMON[@]}" || {
    echo "run.sh: $(date -u +%FT%TZ) $dir failed" >&2
    exit 1
  }
  echo "run.sh: $(date -u +%FT%TZ) finished $dir"
}

below_100() {  # run_dir games -> the games of the list without a score of 100
  python3 - "$1" "$2" <<'EOF'
import json, sys
scores = json.load(open(f"{sys.argv[1]}/score.json"))["games"]
best = {gid.split("-")[0]: entry["score"] for gid, entry in scores.items()}
print(",".join(g for g in sys.argv[2].split(",") if best.get(g, 0.0) < 100.0))
EOF
}

attempt "$BASE" "$GAMES"
games=$(below_100 "$BASE" "$GAMES")
for retry in 1 2; do
  if [ -z "$games" ]; then
    echo "run.sh: every game at 100"
    exit 0
  fi
  attempt "$BASE-retry$retry" "$games"
  games=$(below_100 "$BASE-retry$retry" "$games")
done
echo "run.sh: done; still under 100: ${games:-none}"
