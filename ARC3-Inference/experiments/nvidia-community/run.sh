#!/usr/bin/env bash
# gpt-6.1-sol on the 25 NVIDIA DreamTeam community games (vendored in the
# community_games/arc3-synthetic-games submodule), with the settings of
# experiments/gpt61sol-features/run.sh: stated reasoning in every tool call,
# handover notes, no budget burn, step-back hints, 300K output cap. One pass,
# no reruns.
#
#   cd ARC3-Inference && bash experiments/nvidia-community/run.sh [run_dir]
#
# Extra arguments after run_dir go to dvc_eval.py (e.g. --make GAME=cc2048
# --make MAX_GENERATED_TOKENS_PER_GAME=20000 for a smoke test).
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN=${1:-runs/gpt61sol-nvidia-25games}
shift || true
GAMES=al7306,cc2048,cg1842,cl0426,df4821,dl4827,fl5273,fw4821,gc4721,hr2048,ll4821,ma4173,mb2741,ml2048,mt4926,ne4172,os1842,ps1842,ps7413,rl2048,rs0427,sf2048,sl4821,ss6041,td4826

# The NVIDIA games import arc_agi_3.game_creator.arcengine_adapter.
export PYTHONPATH="$PWD/community_games/arc3-synthetic-games/third_party/nvidia/runtime_support${PYTHONPATH:+:$PYTHONPATH}"
[ -f environment_files_community/catalog.json ] || uv run --no-sync python scripts/build_community_envs.py

exec uv run --no-sync python scripts/dvc_eval.py --run-dir "$RUN" --metrics "$RUN.metrics.json" \
  --make GAME="$GAMES" --make ENVIRONMENTS_DIR=environment_files_community \
  --make CONFIG_PATH=configs/inference.openai.json --make MODEL=gpt-6.1-sol \
  --make CONCURRENT_JOBS=10 \
  --make MAX_GENERATED_TOKENS_PER_GAME=300000 --make MAX_RUNTIME_MINUTES=240 \
  --env LOCAL_ANALYZER_MAX_OUTPUT=0 --env OPENAI_REASONING_EFFORT=xhigh \
  --env OPENAI_REASONING_SUMMARY=detailed \
  --env ARC3_SEND_REASONING_DETAILS=1 --env ARC3_OPENAI_PRICING=2,0.1,2.5,10 \
  --env ARC3_PYTHON_RATIONALE=1 \
  --env ARC3_NOTE_COMPACTION_TOKENS=120000 --env ARC3_NOTE_COMPACTION_KEEP_TURNS=10 \
  --env ARC3_NO_BUDGET_BURN=1 \
  --env ARC3_REPEAT_HINT=1 --env ARC3_REPEAT_HINT_POSITIONS=3 \
  --env ARC3_REPEAT_HINT_VISITS=3 --env ARC3_REPEAT_HINT_COOLDOWN=10 \
  "$@"
