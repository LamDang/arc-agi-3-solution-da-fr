#!/usr/bin/env bash
# SessionStart hook for Claude Code cloud sessions: makes ARC3-Inference ready
# for OpenRouter runs and scoring. See ARC3-Inference/LOCAL_EVAL.md.
set -euo pipefail

if [[ "${CLAUDE_CODE_REMOTE:-}" != "true" ]]; then
  exit 0
fi

harness_dir="${CLAUDE_PROJECT_DIR}/ARC3-Inference"
export PATH="${HOME}/.local/bin:${PATH}"

# The project pins Python 3.12.12, which older uv releases cannot provision.
if ! uv python install 3.12.12 >/dev/null 2>&1; then
  python3 -m pip install --quiet --user --upgrade uv
  uv python install 3.12.12
fi

cd "${harness_dir}"
# --frozen, not --locked: --locked re-validates the lockfile, which downloads
# the vLLM wheel from wheels.vllm.ai. vLLM is only needed by the `server` extra.
uv sync --frozen --extra dev --quiet

# The harness reads OPENROUTER_API_KEY; this environment provides OR_API_KEY.
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
  echo 'export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-${OR_API_KEY:-}}"' >> "${CLAUDE_ENV_FILE}"
fi

# DVC saves game files and runs to the S3 remote; see LOCAL_EVAL.md. Kept out
# of the project venv so its dependencies cannot move the harness's pins.
if ! uv tool install --quiet 'dvc[s3]==3.67.1' >&2; then
  echo "session-start: dvc not installed; see ARC3-Inference/LOCAL_EVAL.md" >&2
fi

# Best effort: a blocked three.arcprize.org must not fail session startup.
if ! timeout 300 uv run --no-sync python scripts/fetch_games.py >&2; then
  echo "session-start: game files not downloaded; see ARC3-Inference/LOCAL_EVAL.md" >&2
fi

# Best effort: community games (submodules) for runs with
# ENVIRONMENTS_DIR=environment_files_community; see LOCAL_EVAL.md.
if timeout 300 git -C "${CLAUDE_PROJECT_DIR}" submodule update --init ARC3-Inference/community_games >&2; then
  uv run --no-sync python scripts/build_community_envs.py >/dev/null 2>&1 \
    || echo "session-start: community games not built; see ARC3-Inference/LOCAL_EVAL.md" >&2
else
  echo "session-start: community game submodules not cloned; see ARC3-Inference/LOCAL_EVAL.md" >&2
fi
