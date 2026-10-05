"""The stepwise harness (v6): the harness drives, the agent fixes one breaking step at a time.

    uv run --no-sync python -m engine_re.run_experiment --run-dir runs/<harness run> --games lp85 \\
        --out runs/engine-re/<name>            (--mode stepwise is the default)

The loop:
  1. Opening (a new run): engine.py from the template, auto_sprites(0) put into make_level, so that
     level 0's first frame is drawn (agent.EngineAgent.play_opening).
  2. Replay the whole recording through engine.py. If every step passes, the run is done.
  3. At the first step k that fails, open a new conversation (an EngineAgent in step mode): its
     kernel shows only step k (`step`, not S), its tests replay steps 0..k, its first message says
     "Fix the breaking test: step k" with the report. It ends as soon as steps 0..k pass (finish,
     run_tests or the automatic test), or when its turns (`episode_turns`) or a budget run out.
  4. Back to 2: the steps after k that already pass are skipped, and the next breaking step opens the
     next conversation. A step still broken after `attempts` conversations ends the run ("stuck").

Nothing but engine.py (and its versions) carries over from one conversation to the next.

<game_dir>/episode_trace/   steps 0..k: what the current conversation's kernel and tests see
<game_dir>/episodes/        one result file per conversation (epNNN.json)
<game_dir>/episodes.jsonl   one line per conversation: step, attempt, outcome, turns, tokens, cost
<game_dir>/result.json      the run's totals, rewritten every turn ("mode": "stepwise", "episode" and
                            "step" while a conversation runs, "passing_prefix" of the last replay)
The transcript, tests.jsonl, images and engine_versions are shared with the conversations; their
records carry "episode". Running the same command again continues an interrupted run from engine.py.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from engine_re.agent import AgentResult, Budget, EngineAgent, ModelConfig, OpenRouterClient, Usage
from engine_re.tester import replay_test
from engine_re.trace import Trace

EPISODE_TURNS = 20
ATTEMPTS = 2
# Counters of a conversation's AgentResult that add up over the run.
_COUNTS = ("tests_run", "nudges", "auto_tests", "python_paused", "engine_changes", "image_messages", "finish_calls")


def _add_usage(total: Usage, part: Usage) -> Usage:
    out = Usage(**asdict(total))
    for f in fields(Usage):
        if f.name == "max_prompt_tokens":
            out.max_prompt_tokens = max(total.max_prompt_tokens, part.max_prompt_tokens)
        else:
            setattr(out, f.name, getattr(total, f.name) + getattr(part, f.name))
    return out


class StepwiseRun:
    def __init__(
        self,
        game: str,
        game_dir: Path,
        model: ModelConfig,
        budget: Budget,
        client: Any = None,
        images: bool = True,
        match: str = "final",
        episode_turns: int = EPISODE_TURNS,
        attempts: int = ATTEMPTS,
        opening: bool = True,
    ):
        self.game = game
        self.dir = Path(game_dir).resolve()
        self.trace = Trace.load(self.dir / "trace")
        self.model = model
        self.budget = budget
        self.client = client or OpenRouterClient(model)
        self.images = images
        self.match = match
        self.episode_turns = episode_turns
        self.attempts = attempts
        self.opening = opening
        self.engine_path = self.dir / "workspace" / "engine.py"
        self.result = AgentResult(game=game, model=model.model, trace_steps=len(self.trace), match=match, images=images)
        self.episodes: list[dict[str, Any]] = []
        self.prior_minutes = 0.0
        self.started = time.time()
        self.passing_prefix: int | None = None
        self.current: dict[str, Any] = {}  # the running conversation: episode, step

    # --- bookkeeping ---------------------------------------------------------------

    def _minutes(self) -> float:
        return self.prior_minutes + (time.time() - self.started) / 60

    def _write(self, result: AgentResult) -> None:
        result.minutes = round(self._minutes(), 2)
        data = {
            **asdict(result), "mode": "stepwise", "episodes": self.episodes, "passing_prefix": self.passing_prefix,
            "episode_turns": self.episode_turns, "attempts": self.attempts, **self.current,
        }
        (self.dir / "result.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def _save(self) -> None:
        self._write(self.result)

    def _live(self, part: AgentResult) -> None:
        """A conversation's progress, shown in result.json as the run's totals so far."""
        live = AgentResult(**{**asdict(self.result), "usage": _add_usage(self.result.usage, part.usage)})
        live.turns = part.turns
        live.tool_calls = {k: self.result.tool_calls.get(k, 0) + part.tool_calls.get(k, 0)
                           for k in set(self.result.tool_calls) | set(part.tool_calls)}
        for name in _COUNTS:
            setattr(live, name, getattr(self.result, name) + getattr(part, name))
        live.provider_errors = part.provider_errors
        self._write(live)

    def _absorb(self, part: AgentResult) -> None:
        self.result.turns = part.turns
        self.result.usage = _add_usage(self.result.usage, part.usage)
        for k, v in part.tool_calls.items():
            self.result.tool_calls[k] = self.result.tool_calls.get(k, 0) + v
        for name in _COUNTS:
            setattr(self.result, name, getattr(self.result, name) + getattr(part, name))
        self.result.provider_errors = part.provider_errors

    def _restore(self) -> None:
        """Continue an interrupted stepwise run: its totals, episodes and time."""
        path = self.dir / "result.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("mode") != "stepwise" or data.get("status") != "running":
            return
        self.episodes = list(data.get("episodes") or [])
        self.prior_minutes = float(data.get("minutes") or 0.0)
        self.result.turns = max([e.get("last_turn", 0) for e in self.episodes] or [0])
        self.result.usage = Usage(**{f.name: sum(e.get("usage", {}).get(f.name, 0) for e in self.episodes) if f.name != "max_prompt_tokens"
                                     else max([e.get("usage", {}).get(f.name, 0) for e in self.episodes] or [0]) for f in fields(Usage)})
        for e in self.episodes:
            for k, v in (e.get("tool_calls") or {}).items():
                self.result.tool_calls[k] = self.result.tool_calls.get(k, 0) + v
            for name in _COUNTS:
                setattr(self.result, name, getattr(self.result, name) + int(e.get(name) or 0))
        self.result.opening = data.get("opening") or {}
        self.result.resumes = int(data.get("resumes") or 0) + 1

    def _over_budget(self) -> str | None:
        u = self.result.usage
        if self.result.turns >= self.budget.max_turns:
            return "budget_turns"
        if u.completion_tokens >= self.budget.max_output_tokens:
            return "budget_tokens"
        if u.cost_usd >= self.budget.max_cost_usd:
            return "budget_cost"
        if self._minutes() >= self.budget.max_minutes:
            return "budget_time"
        return None

    def _agent(self, **kwargs: Any) -> EngineAgent:
        return EngineAgent(self.game, self.dir, self.model, kwargs.pop("budget", self.budget), client=self.client,
                           match=self.match, images=self.images, **kwargs)

    # --- the loop ------------------------------------------------------------------

    def run(self) -> AgentResult:
        self.started = time.time()
        self._restore()
        if not self.engine_path.exists():
            opener = self._agent(opening=self.opening)
            if self.opening:
                opener.play_opening()
                self.result.opening = opener.result.opening
            else:
                opener.setup()
        last_step, attempt = None, 0
        for e in self.episodes:  # a restored run continues the attempt count of its last step
            last_step, attempt = e["step"], (attempt + 1 if e["step"] == last_step else 1)
        try:
            while True:
                self._save()
                report = replay_test(self.engine_path, self.trace, failures=1, scratch_root=self.dir, match=self.match)
                summary = report.summary()
                self.passing_prefix = summary.get("passing_prefix")
                if not self.result.best or (self.passing_prefix or 0) > (self.result.best.get("passing_prefix") or 0):
                    self.result.best = {"turn": self.result.turns, **summary}
                    shutil.copy(self.engine_path, self.dir / "engine_best.py")
                if report.passed:
                    self.result.status = "passed"
                    self.result.first_pass_turn = self.result.turns
                    break
                reason = self._over_budget()
                if reason:
                    self.result.status = reason
                    break
                k = summary.get("first_fail")
                k = len(self.trace) - 1 if k is None else int(k)  # a contract failure alone: test every step
                attempt = attempt + 1 if k == last_step else 1
                last_step = k
                if attempt > self.attempts:
                    self.result.status = "stuck"
                    break
                if not self._episode(k, attempt):
                    break
        except Exception as exc:  # noqa: BLE001  (record the failure in result.json)
            self.result.status = "error"
            self.result.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.current = {}
            self._final_test()
            self._save()
        return self.result

    def _episode(self, k: int, attempt: int) -> bool:
        """One conversation on step k; False when the run must stop (an error in the conversation)."""
        n = len(self.episodes) + 1
        episode_dir = self.dir / "episode_trace"
        shutil.rmtree(episode_dir, ignore_errors=True)
        Trace(self.trace.game_id, self.trace.steps[: k + 1], {**self.trace.meta, "focus": k}).save(episode_dir)
        offset = self.result.turns
        u = self.result.usage
        budget = Budget(
            max_turns=offset + min(self.episode_turns, self.budget.max_turns - offset),
            max_output_tokens=self.budget.max_output_tokens - u.completion_tokens,
            max_cost_usd=self.budget.max_cost_usd - u.cost_usd,
            max_minutes=self.budget.max_minutes,
            python_quota=self.budget.python_quota,
        )
        self.current = {"episode": n, "step": k}
        agent = self._agent(
            budget=budget, opening=False, trace_dir=episode_dir, focus=k, episode=n, turn_offset=offset,
            prior_minutes=self._minutes(), result_path=self.dir / "episodes" / f"ep{n:03d}.json", on_save=self._live,
        )
        part = agent.run()
        self._absorb(part)
        record = {
            "episode": n, "step": k, "attempt": attempt, "status": part.status, "fixed": part.status == "passed",
            "turns": part.turns - offset, "first_turn": offset + 1, "last_turn": part.turns,
            "usage": asdict(part.usage), "tool_calls": part.tool_calls, "error": part.error,
            **{name: getattr(part, name) for name in _COUNTS},
        }
        self.episodes.append(record)
        with (self.dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if part.status == "error":
            self.result.status = "error"
            self.result.error = part.error
            return False
        return True

    def _final_test(self) -> None:
        try:
            report = replay_test(self.engine_path, self.trace, details=3, scratch_root=self.dir, match=self.match)
            self.result.final = report.summary()
            (self.dir / "final_test.txt").write_text(report.text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.result.final = {"error": f"{type(exc).__name__}: {exc}"}
