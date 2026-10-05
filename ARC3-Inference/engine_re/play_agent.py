"""The play-and-model agent (v10): plays a live game while building engine.py, in one conversation.

    uv run --no-sync python -m engine_re.run_play --games ft09 --out runs/engine-play/<name>

The design is in PLAY_DESIGN.md. In short, the stepwise agent (engine_re.agent, stepwise mode) with the
recording replaced by the game being played (engine_re.live_game):

- PLAN: the model sees every step played so far (`recording`) and the game's current frame; in python,
  state_now() is its engine's state and simulate(actions) plays moves on it. It sends moves with
  commit_moves(actions, note).
- commit_moves first runs the tests on engine.py over every step played so far; if any fails, nothing is
  sent and the model gets the report. Otherwise the batch is predicted with the engine in one sandboxed
  run (tester.run_candidate on the recorded actions plus the batch), and the moves are sent to the real
  game one at a time; each real result is compared with the prediction by the tests' rule
  (tester.check_step). The batch stops at the first difference (the moves after it are not sent), after a
  solved level and when the game ends.
- FIT: after a difference at step k, the message "step k did not go as predicted" with the test report
  (the comparison, with its picture); the model fixes engine.py and submits it with commit_engine, which
  must pass every step so far; then PLAN again. A commit whose engine still fails some other step gets
  that step as in the stepwise harness (advance_message).
- After a game over the harness sends RESET itself (auto_reset), checked like any move. A WIN ends the run.

Per game directory, besides the stepwise agent's files: trace/ (the live trace, saved after every step;
visible_trace/ is what the kernel and the tests see, steps 0..focus) and
artifacts/<game_id>_p0_viewer_data_events.jsonl (the base harness's event sidecar). result.json adds the
play fields (PlayResult). The transcript logs every batch ("batch" records, one "move" record per move
sent), every PLAN message ("plan") and every FIT round ("step_start", as the stepwise harness logs a
breaking step). A run is resumed like a stepwise run, with the real game replayed from trace/.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine_re import diff_report
from engine_re.agent import (
    READ_CHARS_IN_MESSAGES, REPORT_CHARS, AgentResult, Budget, EngineAgent, ModelConfig, OpenRouterClient, _truncate,
)
from engine_re.helpers import PLAY_FUNCTIONS
from engine_re.kernel import KernelClient
from engine_re.live_game import LiveGame
from engine_re.prompts import (
    advance_message, batch_lines, commit_moves_description, kernel_names_text, mismatch_message, plan_message, tools,
)
from engine_re.tester import StepCheck, check_step, run_candidate, trace_meta
from engine_re.trace import Action, Trace, move_label, parse_move

CURRENT_FRAME_NOTE = "The real game's current frame (after step {k}), upscaled 8x:"
# After a game over the harness restarts the level itself, as the base harness does; the RESET is a real step.
AUTO_RESET_NOTE = "[harness] automatic RESET after the game over"


@dataclass
class PlayResult(AgentResult):
    game_id: str = ""
    outcome: str = "playing"  # "won" | "playing" (the run ended with the game where it was)
    actions: int = 0  # actions sent, the opening RESET excluded (automatic RESETs included)
    levels_completed: int = 0
    win_levels: int = 0
    actions_per_level: list = field(default_factory=list)
    baseline_actions: list | None = None
    score: float | None = None  # the official formula (live_game.LiveGame.score)
    batches: int = 0  # commit_moves calls that sent something
    refused_batches: int = 0  # commit_moves calls refused (tests failing, bad actions, ...)
    moves_sent: int = 0
    mismatches: int = 0  # moves whose real result differed from the prediction
    auto_resets: int = 0
    batch_log: list = field(default_factory=list)  # per batch: turn, first step, moves, matched, mismatch, note
    fit_rounds: list = field(default_factory=list)  # per fit round: step, start turn, end turn (None: open), commits
    phase_turns: dict = field(default_factory=lambda: {"plan": 0, "fit": 0})
    step_tokens: list = field(default_factory=list)  # output tokens spent up to each step (cumulative)
    phase: str = "plan"


class PlayAgent(EngineAgent):
    TOOLS = ("python", "run_tests", "commit_engine", "commit_moves")
    BUILTINS = PLAY_FUNCTIONS

    def __init__(
        self,
        game: str,
        game_dir: Path,
        model: ModelConfig,
        budget: Budget,
        environments_dir: Path = Path("environment_files"),
        client: OpenRouterClient | None = None,
        images: bool = True,
        batch_size: int = 10,
        max_actions: int = 500,
        auto_reset: bool = True,
    ):
        game_dir = Path(game_dir).resolve()
        trace_dir = game_dir / "trace"
        if (trace_dir / "trace.json").exists():  # a resumed run: the real game replays what was played
            self.live = LiveGame(game, environments_dir, trace=Trace.load(trace_dir))
        else:
            self.live = LiveGame(game, environments_dir)
            self.live.start()
            self.live.save(trace_dir)
        super().__init__(game, game_dir, model, budget, client=client, images=images, stepwise=True, history=True)
        self.full_trace = self.trace = self.live.trace  # the live trace: it grows as the game is played
        self.mode = "play"
        self.kernel = KernelClient(self.workspace, self.trace_dir, images=images, log=self._log_engine_change, history=True, play=True)
        self.result = PlayResult(
            game=game, model=model.model, trace_steps=len(self.full_trace), match=self.match, interface=self.interface,
            images=images, mode="play", context=model.context, thinking_budget=model.thinking_budget,
            game_id=self.live.game_id, baseline_actions=self.live.baseline_actions,
        )
        self.batch_size = max(1, int(batch_size))
        self.max_actions = int(max_actions)
        self.auto_reset = auto_reset
        self.phase = "plan"
        self.committed_sha: str | None = None
        self.listed_sha: str | None = None  # engine.py as last listed in a PLAN or FIT message
        self.batches_this_turn = 0
        self.pending: dict[str, Any] | None = None  # a batch sent this turn, handled at the end of the turn
        self.fit_round: dict[str, Any] | None = None

    # --- tools -----------------------------------------------------------------------------------

    def _tools(self) -> list[dict[str, Any]]:
        schemas = tools(self.images, "play", True)
        for schema in schemas:
            if schema["function"]["name"] == "commit_moves":
                schema["function"]["description"] = commit_moves_description(self.batch_size)
        return schemas

    def _tool_commit_engine(self, message: Any = None) -> str:
        """As in the stepwise harness: the tests on steps 0..k must pass; the commit is applied after the
        turn (_advance), which replays every step played and asks for the next moves when all pass."""
        text = super()._tool_commit_engine(message)
        if text.startswith("Committed:"):
            return (f"Committed: steps 0-{self.focus} pass. After this turn the harness replays every step played so far "
                    "and, when all pass, asks you to plan the next moves (the next message shows the game's current frame).")
        return text

    def _tool_commit_moves(self, actions: Any = None, note: Any = None) -> str:
        """Send moves to the real game (PLAY_DESIGN.md 3.3): the tests first, nothing sent if they fail;
        then each move predicted, sent and checked, stopping at the first difference."""
        if self.batches_this_turn:
            self.result.refused_batches += 1
            return ("Not sent: one commit_moves call per turn. This turn's batch and its result are above; the next message "
                    "says what to do next. Send the next batch in your next turn.")
        if not isinstance(note, str) or not note.strip():
            self.result.refused_batches += 1
            return "Not sent: commit_moves needs a note (one or two sentences: what the batch is meant to do and what your engine predicts)."
        if not isinstance(actions, list) or not actions:
            self.result.refused_batches += 1
            return 'Not sent: actions must be a non-empty list of moves, e.g. ["UP", "UP", {"click": [12, 40]}].'
        if len(actions) > self.batch_size:
            self.result.refused_batches += 1
            return f"Not sent: at most {self.batch_size} moves per call; you gave {len(actions)}. Send the first {self.batch_size}, look at the result, then the rest."
        try:
            acts = [parse_move(a) for a in actions]
        except ValueError as exc:
            self.result.refused_batches += 1
            return f"Not sent: {exc}."
        allowed = set(self.full_trace[0].available_actions) | {0}
        for a in acts:
            if a.id == 6 and not (0 <= int(a.x) <= 63 and 0 <= int(a.y) <= 63):
                self.result.refused_batches += 1
                return f"Not sent: the click {move_label(a)} is off the 64x64 screen (x and y must be 0-63)."
            if a.id not in allowed:
                self.result.refused_batches += 1
                names = ", ".join(move_label(Action(i, 0, 0)) if i != 6 else "clicks" for i in sorted(allowed))
                return f"Not sent: this game does not accept {move_label(a)}. It accepts: {names}."
        if self.live.won:
            return "Not sent: the game is won; nothing more to play."
        if self.live.game_over and acts[0].id != 0:
            self.result.refused_batches += 1
            return "Not sent: the game is over, so only RESET is accepted now. Start the batch with RESET (it restarts the level)."
        n = len(self.full_trace)
        if self.focus != n - 1:  # leaving a fit round on an earlier step: the claim is that every step passes
            self._focus_on(n - 1)
        report = self._run_tests(None, 1, auto=False)
        if isinstance(report, str):
            self.result.refused_batches += 1
            return "Not sent: the tests could not run. " + report
        if not report.passed:
            self.result.refused_batches += 1
            self.phase = "fit"
            first = report.summary().get("first_fail")
            if first is not None and first < n - 1:
                self._focus_on(int(first))
            self._log({"turn": self.result.turns, "refused_batch": {"reason": "tests fail", "first_fail": first, "note": note}})
            return (f"Not sent: your engine does not reproduce the game so far (steps 0-{n - 1}), so no move was sent. Fix "
                    "engine.py first, then commit_engine(message) and plan again. The report:\n\n" + _truncate(report.text, REPORT_CHARS))
        sha = self._engine_hash()
        if sha != self.committed_sha:  # an engine that passes everything, submitted by the batch itself
            self.committed_sha = sha
            entry = {"turn": self.result.turns, "fixed": self.focus, "next": None, "message": f"(commit_moves) {note}",
                     "engine_sha": sha, "version": self._version_of(sha), "implicit": True}
            self.result.advances.append(entry)
            self._log({"turn": self.result.turns, "commit": {k: v for k, v in entry.items() if k != "turn"}})
        self.batches_this_turn += 1
        self.result.batches += 1
        outcomes, dropped = self._play(acts, note)
        self.pending = {"outcomes": outcomes, "dropped": dropped, "note": note}
        first_step = outcomes[0]["index"] if outcomes else n
        lines = [f"Sent {len(outcomes)} of {len(acts)} move(s) (steps {first_step}-{first_step + len(outcomes) - 1}):"] if outcomes else []
        lines += batch_lines(self.full_trace, first_step, outcomes)
        last = outcomes[-1] if outcomes else None
        if last is not None and not last["ok"]:
            lines.append(f"The batch stopped at step {last['index']}: the real result differs from your engine's prediction"
                         + (f"; {dropped} move(s) not sent" if dropped else "") + ". A fit round opens: the next message shows the comparison.")
        elif last is not None and last["state"] == "WIN":
            lines.append("The game is won.")
        elif last is not None and last["state"] == "GAME_OVER":
            lines.append("The game is over" + (": the harness will RESET the level (checked against your engine too), then ask for the next moves."
                                               if self.auto_reset else ": only RESET is accepted now."))
        elif last is not None and last.get("level_solved"):
            lines.append(f"Level {last['level']} is solved; the batch stops there" + (f" ({dropped} move(s) not sent)" if dropped else "")
                         + ". The next message shows the new level.")
        else:
            lines.append("Every move matched your engine. The next message asks for the next moves.")
        return "\n".join(lines)

    # --- playing ---------------------------------------------------------------------------------

    def _predict(self, acts: list[Action]) -> tuple[dict[str, Any], list[np.ndarray]]:
        """The engine's prediction for `acts` after everything played: one sandboxed run on the recorded
        actions followed by the batch (no contract tests)."""
        actions = [s.action.to_json() for s in self.full_trace.steps] + [a.to_json() for a in acts]
        return run_candidate(self.engine_path, actions, scratch_root=self.dir, meta=trace_meta(self.full_trace), contract=False)

    @staticmethod
    def _verdict(check: StepCheck, real: Any, got: dict[str, Any] | None, error: str | None) -> str:
        if got is None:
            last = (error or "").strip().splitlines()[-1][:160] if error else "no prediction"
            return f"your engine raised an error ({last})"
        parts = []
        if "final frame" in check.problems:
            parts.append("the final frame differs")
        for name, shown in (("state", "outcome"), ("levels_completed", "levels completed"), ("win_levels", "win levels"), ("available_actions", "available actions")):
            if name in check.problems:
                parts.append(f"{shown}: the game says {getattr(real, name)!r}, your engine {got.get(name)!r}")
        return "; ".join(parts) if parts else "; ".join(check.problems)

    def _play(self, acts: list[Action], note: str) -> tuple[list[dict[str, Any]], int]:
        """Send `acts` one at a time, each checked against the engine's prediction; stop at the first
        difference, a solved level or the end of the game. Returns the outcomes and how many were not sent."""
        n = len(self.full_trace)
        prediction, frames = self._predict(acts)
        predicted = prediction.get("steps") or []
        error = prediction.get("error")
        outcomes: list[dict[str, Any]] = []
        for i, act in enumerate(acts):
            pos = n + i
            level_before = self.live.level
            real = self.live.perform(act)
            self.result.step_tokens.append(int(self.result.usage.completion_tokens))
            self.live.save(self.dir / "trace")
            got = predicted[pos] if pos < len(predicted) else None
            got_frames = frames[pos] if pos < len(frames) else None
            if got is None:
                check = StepCheck(real.index, False, False, ["engine error"])
            else:
                check = check_step(real, got, got_frames, self.match)
            outcome = {
                "index": real.index, "label": move_label(act), "ok": check.ok, "warning": check.warning,
                "verdict": "" if check.ok else self._verdict(check, real, got, error),
                "frames": real.n_frames, "level": level_before, "state": real.state,
                "level_solved": real.levels_completed > self.full_trace[pos - 1].levels_completed,
                "levels_completed": real.levels_completed, "auto": note == AUTO_RESET_NOTE,
            }
            outcomes.append(outcome)
            self.result.moves_sent += 1
            if not check.ok:
                self.result.mismatches += 1
            self._log({"turn": self.result.turns, "move": outcome})
            if not check.ok or real.state in ("WIN", "GAME_OVER") or outcome["level_solved"]:
                break
        self._focus_on(len(self.full_trace) - 1)
        self.result.batch_log.append({
            "turn": self.result.turns, "first_step": n, "moves": [move_label(a) for a in acts], "sent": len(outcomes),
            "matched": sum(o["ok"] for o in outcomes), "mismatch": next((o["index"] for o in outcomes if not o["ok"]), None),
            "note": note,
        })
        self._log({"turn": self.result.turns, "batch": self.result.batch_log[-1]})
        return outcomes, len(acts) - len(outcomes)

    # --- phases ----------------------------------------------------------------------------------

    def _budget_line(self) -> str:
        u = self.result.usage
        counts = self.live.actions_per_level()
        per_level = ", ".join(f"level {i}: {c}" for i, c in enumerate(counts) if c)
        return (f"Actions played: {self.live.actions} of at most {self.max_actions} ({per_level or 'none yet'}). Turns: "
                f"{self.result.turns} of {self.budget.max_turns}; minutes: {self._elapsed_minutes():.0f} of {self.budget.max_minutes:.0f}; "
                f"output tokens: {u.completion_tokens:,} of {self.budget.max_output_tokens:,}.")

    def _engine_listing_if_changed(self) -> str:
        sha = self._engine_hash()
        if sha == self.listed_sha:
            return ""
        self.listed_sha = sha
        return self._read_engine(fold=True, max_chars=READ_CHARS_IN_MESSAGES)

    def _frame_part(self) -> list[dict[str, Any]]:
        """The current frame as an image part (with its note), when images are on."""
        if not self.images:
            return []
        k = len(self.full_trace) - 1
        frame = self.full_trace[k].last
        if frame is None:
            return []
        png = diff_report.png_bytes(diff_report.panels_image([frame], [f"ORIGINAL GAME: after step {k}"], [], diff_report.UPSCALE))
        path = self._save_png(png, f"turn{self.result.turns:03d}_frame{k}")
        self._log({"turn": self.result.turns, "images": [str(path.relative_to(self.dir))], "captions": [CURRENT_FRAME_NOTE.format(k=k)]})
        self.result.image_messages += 1
        return [{"type": "text", "text": CURRENT_FRAME_NOTE.format(k=k)}, self._image_part(png, path)]

    def _enter_plan(self, last_batch: str, say: bool = True) -> str | list[dict[str, Any]]:
        """The PLAN message (said to the model unless `say` is False; the content is returned either way)."""
        self.phase = "plan"
        if self.fit_round is not None and self.fit_round.get("end_turn") is None:
            self.fit_round["end_turn"] = self.result.turns
        self.fit_round = None
        n = len(self.full_trace)
        text = plan_message(
            self.game, self.full_trace, last_batch=last_batch, budget_line=self._budget_line(), batch_size=self.batch_size,
            engine_read=self._engine_listing_if_changed(), kernel_names=kernel_names_text(*self.kernel.names()),
            baseline=self.live.baseline_actions,
        )
        self._log({"turn": self.result.turns, "plan": {"step": n - 1, "actions": self.live.actions, "level": self.live.level}})
        if say and not self.condense and self.images:
            self._hide_images(self.messages)
            self._log({"turn": self.result.turns, "hide_images": True})
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}, *self._frame_part()]
        content: str | list[dict[str, Any]] = parts if len(parts) > 1 else text
        if say:
            self._say("user", content)
        return content

    def _new_fit_round(self, k: int, verdict: str) -> None:
        if self.fit_round is not None and self.fit_round.get("end_turn") is None:
            self.fit_round["end_turn"] = self.result.turns
        self.fit_round = {"step": k, "start_turn": self.result.turns, "end_turn": None, "commits": 0, "verdict": verdict}
        self.result.fit_rounds.append(self.fit_round)

    def _enter_fit(self, k: int, verdict: str, dropped: int, say: bool = True) -> str | list[dict[str, Any]]:
        """A move differed from the prediction at step k: the FIT message with the test report on steps 0..k."""
        self.phase = "fit"
        self.passed = False
        self._focus_on(k)
        report = self._step_report("advance")
        self.tested_hash = self._engine_hash()
        self._new_fit_round(k, verdict)
        text = mismatch_message(
            self.full_trace, k, verdict, dropped, report, engine_read=self._engine_listing_if_changed(),
            kernel_names=kernel_names_text(*self.kernel.names()),
        )
        self._log({"turn": self.result.turns, "step_start": {"step": k, "report": report, "verdict": verdict}})
        content = self._opening_content(text)
        if say:
            self._say("user", content)
        return content

    def _after_sync(self, last_batch: str) -> bool:
        """The engine reproduces everything played: a RESET after a game over, then PLAN; False when the game is won."""
        if self.live.won:
            return False
        if self.live.game_over and self.auto_reset:
            outcomes, _ = self._play([Action(0)], AUTO_RESET_NOTE)
            self.result.auto_resets += 1
            o = outcomes[0]
            if not o["ok"]:
                self._enter_fit(o["index"], o["verdict"], 0)
                return True
            last_batch += f" The game was over, so the harness sent a RESET (step {o['index']}, an action): the level restarted as your engine predicted."
        elif self.live.game_over:
            last_batch += " The game is over: only RESET is accepted now."
        self._enter_plan(last_batch)
        return True

    def _advance(self, commit: dict[str, Any]) -> bool:
        """A commit was accepted (steps 0..k pass): replay every step played; another failing step is the
        next fit (advance_message); when all pass, the fix is committed and the model plans again."""
        fixed = self.focus
        k = self._replay_all()
        entry = {"turn": self.result.turns, "fixed": fixed, "next": k, **commit}
        self.result.advances.append(entry)
        self._log({"turn": self.result.turns, "commit": {key: v for key, v in entry.items() if key != "turn"}})
        if self.fit_round is not None:
            self.fit_round["commits"] += 1
        if k is not None:
            self._focus_on(k)
            self.passed = False
            text = self._step_report("advance")
            self.tested_hash = self._engine_hash()
            self._new_fit_round(k, "fails after the commit (an earlier or later step)")
            self._log({"turn": self.result.turns, "advance": {"fixed": fixed, "next": k, "report": text}})
            engine_read = self._engine_listing_if_changed()
            names = kernel_names_text(*self.kernel.names())
            self._say("user", self._opening_content(advance_message(self.full_trace, fixed, k, text, True, engine_read, names)))
            return True
        self.committed_sha = commit["engine_sha"]
        self._focus_on(len(self.full_trace) - 1)
        return self._after_sync(f"Commit accepted: your engine now reproduces steps 0-{len(self.full_trace) - 1}.")

    def _end_of_turn(self) -> bool:
        self.result.phase_turns[self.phase] = self.result.phase_turns.get(self.phase, 0) + 1
        self.batches_this_turn = 0
        pending, self.pending = self.pending, None
        if self.stepwise and self.commit is not None:
            commit, self.commit = self.commit, None
            if not self._advance(commit):
                return self._finish_won()
            pending = None  # the commit's message replaces the batch's (a commit after a batch in one turn)
        if pending is not None:
            outcomes, dropped = pending["outcomes"], pending["dropped"]
            last = outcomes[-1] if outcomes else None
            if last is not None and not last["ok"]:
                self._enter_fit(last["index"], last["verdict"], dropped)
            else:
                sent = len(outcomes)
                what = (f"Your last batch: {sent} move(s) sent, all as your engine predicted"
                        + (f"; level {last['level']} solved, so the batch stopped there" if last and last.get("level_solved") and last["state"] != "WIN" else "")
                        + (f" ({dropped} move(s) not sent)" if dropped else "") + ".")
                if not self._after_sync(what):
                    return self._finish_won()
        if self.live.won:
            return self._finish_won()
        return True

    def _finish_won(self) -> bool:
        self.result.status = "won"
        self.result.outcome = "won"
        return False

    # --- the run ---------------------------------------------------------------------------------

    def _stepwise_start(self) -> str | list[dict[str, Any]] | None:
        """The first message of a new run (or of a resumed one whose conversation could not be rebuilt):
        the opening (level 0's sprite code into make_level) on a fresh engine, then PLAN when every step
        passes, else the first failing step as a fit round."""
        n = len(self.full_trace)
        self._focus_on(n - 1)
        if self.result.turns == 0 and not self.result.opening:
            self._open()
        first = self._replay_all()
        if first is None:
            self.committed_sha = self._engine_hash()
            last = "The game has just started." if n == 1 else f"{self.live.actions} action(s) were played before this session."
            return self._enter_plan(last, say=False)
        return self._enter_fit(first, "the engine does not reproduce this step yet", 0, say=False)

    def _resume_conversation(self) -> bool:
        if not super()._resume_conversation():
            return False
        last_phase = next((("plan" if "plan" in r else "fit") for r in reversed(self.records)
                           if "plan" in r or "step_start" in r or "advance" in r), "plan")
        if last_phase == "plan":
            self.phase = "plan"
            self._focus_on(len(self.full_trace) - 1)
        else:
            self.phase = "fit"
            self.fit_round = {"step": self.focus, "start_turn": self.result.turns, "end_turn": None, "commits": 0, "verdict": "(resumed)"}
            self.result.fit_rounds.append(self.fit_round)
        return True

    def _restore(self) -> bool:
        restored = super()._restore()
        previous = self.dir / "result.json"
        if previous.exists():
            data = json.loads(previous.read_text(encoding="utf-8"))
            for key in ("batches", "refused_batches", "moves_sent", "mismatches", "auto_resets", "batch_log", "fit_rounds", "phase_turns", "step_tokens"):
                if key in data:
                    setattr(self.result, key, data[key])
        return restored

    def _over_budget(self) -> str | None:
        reason = super()._over_budget()
        if reason:
            return reason
        if self.live.actions >= self.max_actions:
            return "budget_actions"
        return None

    def _save_result(self) -> None:
        r = self.result
        r.phase = self.phase
        r.trace_steps = len(self.full_trace)
        r.actions = self.live.actions
        r.levels_completed = self.live.levels_completed
        r.win_levels = int(self.full_trace[0].win_levels)
        r.actions_per_level = self.live.actions_per_level()
        r.score = self.live.score()
        r.outcome = "won" if self.live.won else "playing"
        super()._save_result()

    def run(self) -> AgentResult:
        result = super().run()
        try:
            self.live.save(self.dir / "trace")
            self.live.write_events(self.dir)
        finally:
            self._save_result()
        return result

    def game_run(self) -> dict[str, Any]:
        """The TAAF GameRun record of this game (live_game.LiveGame.game_run), with the output tokens per action."""
        cumulative = self.result.step_tokens
        tokens = [0] * len(self.full_trace)
        for k in range(1, min(len(tokens), len(cumulative) + 1)):
            tokens[k] = int(cumulative[k - 1]) - (int(cumulative[k - 2]) if k >= 2 else 0)
        state = "won" if self.live.won else "gave_up"
        return self.live.game_run(tokens, state=state, note=f"{self.result.status}; output tokens={self.result.usage.completion_tokens}")


def engine_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["PlayAgent", "PlayResult", "engine_sha", "time"]
