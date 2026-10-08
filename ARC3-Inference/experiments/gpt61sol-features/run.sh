#!/usr/bin/env bash
# gpt-6.1-sol on the 25 official games with the four new harness settings
# (exp/gpt61sol-features-25games.md), then up to two reruns of every game that
# scored under 100. Each attempt is its own run directory and is kept:
#   runs/gpt61sol-features-25games          all 25 games
#   runs/gpt61sol-features-25games-retry1   the games under 100 in the first
#   runs/gpt61sol-features-25games-retry2   the games still under 100
# The script can be started again after an interruption: an attempt with a
# score.json, or whose games all ended (benchmark.json has an end_time), is
# not played again; an attempt still playing under another dvc_eval process
# is waited for; an attempt directory left unfinished with nothing playing
# stops the script (resume that attempt by hand with RESUME_FROM).
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
# A rerun has one wave of games, so the inline job's own deadline (240 min
# per wave + 10, minus its 10-minute buffer) would cancel a game at about its
# 240-minute limit; 270 minutes lets the game's own limit end it (gave_up).
RETRY=(--make MAX_EXPERIMENT_RUNTIME_MINUTES=270)

ended() {  # run_dir -> success when every game of the attempt has ended
  python3 - "$1" <<'EOF'
import json, sys
try:
    sys.exit(0 if json.load(open(f"{sys.argv[1]}/benchmark.json")).get("end_time") else 1)
except (OSError, ValueError):
    sys.exit(1)
EOF
}

attempt() {  # run_dir games [extra dvc_eval args...]
  local dir=$1 games=$2
  shift 2
  if [ -f "$dir/score.json" ]; then
    echo "run.sh: $dir already scored, skipped"
    return 0
  fi
  if [ -e "$dir" ]; then
    while pgrep -f "[d]vc_eval.py --run-dir $dir " > /dev/null; do
      sleep 60
    done
    if [ -f "$dir/score.json" ]; then
      echo "run.sh: $(date -u +%FT%TZ) $dir finished under the earlier driver"
      return 0
    fi
    if ended "$dir"; then
      echo "run.sh: $(date -u +%FT%TZ) $dir ended but was not scored; reading benchmark.json" >&2
      return 0
    fi
    echo "run.sh: $dir exists unfinished and nothing is playing it; resume it by hand" >&2
    exit 1
  fi
  echo "run.sh: $(date -u +%FT%TZ) starting $dir with $games"
  if ! uv run --no-sync python scripts/dvc_eval.py --run-dir "$dir" \
    --metrics "$dir.metrics.json" --make GAME="$games" "${COMMON[@]}" "$@"; then
    # A crashed game makes `make score_run` fail although every game ended;
    # the reruns then go on from benchmark.json.
    if ended "$dir"; then
      echo "run.sh: $(date -u +%FT%TZ) $dir ended, but scoring or packing failed; reading benchmark.json" >&2
      return 0
    fi
    echo "run.sh: $(date -u +%FT%TZ) $dir failed before its games ended" >&2
    exit 1
  fi
  echo "run.sh: $(date -u +%FT%TZ) finished $dir"
}

below_100() {  # run_dir games -> the games of the list without a score of 100
  python3 - "$1" "$2" <<'EOF'
import json, os, sys
run, games = sys.argv[1], sys.argv[2].split(",")
if os.path.exists(f"{run}/score.json"):
    scores = json.load(open(f"{run}/score.json"))["games"]
    best = {gid.split("-")[0]: entry["score"] for gid, entry in scores.items()}
else:
    runs = json.load(open(f"{run}/benchmark.json"))["game_runs"]
    best = {r["game_id"].split("-")[0]: (r.get("final_score") or 0.0)
            for r in runs if r.get("state") in ("won", "gave_up")}
print(",".join(g for g in games if best.get(g, 0.0) < 100.0))
EOF
}

attempt "$BASE" "$GAMES"
games=$(below_100 "$BASE" "$GAMES") || { echo "run.sh: cannot read $BASE results" >&2; exit 1; }
for retry in 1 2; do
  if [ -z "$games" ]; then
    echo "run.sh: every game at 100"
    exit 0
  fi
  attempt "$BASE-retry$retry" "$games" "${RETRY[@]}"
  games=$(below_100 "$BASE-retry$retry" "$games") || { echo "run.sh: cannot read $BASE-retry$retry results" >&2; exit 1; }
done
echo "run.sh: done; still under 100: ${games:-none}"
