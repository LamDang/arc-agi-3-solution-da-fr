"""The play-and-model agent (v10): plays a live game while building engine.py, in one conversation.

    uv run --no-sync python -m engine_re.run_play --games ft09 --out runs/engine-play/<name>

The design is in PLAY_DESIGN.md. In short, the stepwise agent (engine_re.agent, stepwise mode) with the
recording replaced by the game being played (engine_re.live_game):

- PLAN: the model sees every step played so far (`recording`) and the game's current frame; in python,
  state_now() is its replica's state (engine.py's) and it plays moves by calling replica.step on copies of it.
  It sends moves with commit_moves(actions, note), the actions being the Actions it stepped its replica with,
  as python prints them (Action(4), Action(6, x=3, y=4)), or labels.
- commit_moves first runs the tests on engine.py over every step played so far; if any fails, nothing is
  sent and the model gets the report (a fit round opens on the failing step). Otherwise engine.py becomes
  the committed engine (engine_committed.py), the batch is predicted with it in one sandboxed run
  (tester.predict: the played actions plus the batch), and the moves are sent to the real game one at a
  time; each real result is compared with the prediction by the tests' rule (tester.check_step: a
  one-pixel HUD-bar difference is a match, with its warning shown). The batch stops at the first
  difference (the moves after it are not sent), after a solved level and when the game ends. One batch
  per turn; a commit_engine earlier in the same turn is the batch's commit.
- FIT: after a difference at step k, the message "step k did not go as your replica predicted" with the
  test report (the comparison, with its picture); the model fixes engine.py and submits it with
  commit_engine, which must pass every step so far; then PLAN again. A commit whose engine still fails a
  later step gets that step as in the stepwise harness (advance_message).
- Support (PLAY_DESIGN.md 3.11, engine_re.support): the tests and the prediction record which lines of engine.py
  each step ran. The committed engine's support map (per line, how many recorded steps ran it; per and/or, whether
  the steps separated its operands) is saved beside it (engine_committed.support.json); each move of a batch is
  read against it (its weakest line, the untested lines it runs, the conditions it relies on that were never
  separated), computed before anything is sent: in commit_moves' output, in the batch's `support` entry of batch_log, in the
  mismatch message, and the thin rules of the last batch's path and the outcome rules in the PLAN message.
  `cut_untested` (off by default) cuts a batch after the first move that runs untested code.
- After a game over the harness sends RESET itself (auto_reset), checked like any move. A WIN ends the run.
- Nudges: after `plan_turns` turns of a plan round without commit_moves, a reminder to send a short batch
  (PLAN_NUDGE, every as many turns; result.json plan_nudges). The test nudge of the stepwise harness only
  in fit rounds.
- The escape hatch (`fit_turns`, off by default; PLAY_DESIGN.md 3.6): after that many turns in one fit round
  without an accepted commit the model is told (FIT_ESCAPE) that commit_moves now sends moves even though
  the tests fail: the engine is out of step with the game from the failing step k on, moves are sent
  unchecked, and the steps from k on are unexplained (the tests replay them but never fail on them). The
  first later RESET or level change the game makes is a resync point (the runner restarts at that level's
  start); then every step is tested again and the loop goes on as usual. Both are kept in the live trace's
  meta ("ignore", "resync", "out_of_sync"), so the tests, the kernel and a resumed run see them.

Per game directory, besides the stepwise agent's files: trace/ (the live trace, saved after every step;
visible_trace/ is what the kernel and the tests see, steps 0..focus), engine_committed.py (the engine the
predictions come from) and artifacts/<game_id>_p0_events.jsonl (the base harness's event sidecar).
result.json adds the play fields (PlayResult). The transcript logs every batch ("batch" records, one
"move" record per move sent), every PLAN message ("plan"), every FIT round ("step_start", as the stepwise
harness logs a breaking step), the nudges ("plan_nudge"), the escape hatch ("fit_escape", "out_of_sync",
"resync"); every record carries the phase it was logged in. A run is resumed like a stepwise run, with the
real game replayed from trace/; moves played after the last message the model got (an interruption during
a batch) are tested and lead to the right next message.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine_re import diff_report, hashline
from engine_re import animation  # transient cells of a batch's animated moves
from engine_re import support as sup
from engine_re.agent import (
    READ_CHARS_IN_MESSAGES, REPORT_CHARS, AgentResult, Budget, EngineAgent, ModelConfig, OpenRouterClient, _truncate,
)
from engine_re.game_api import fixed_block_lines, sync_points
from engine_re.helpers import PLAY_FUNCTIONS
from engine_re.kernel import KernelClient
from engine_re.live_game import LiveGame
from engine_re.prompts import (
    COMMIT_HINT_PLAY, FIT_ESCAPE, PLAN_NUDGE, accepted_actions_text, advance_message, batch_lines, commit_moves_description,
    kernel_names_text, mismatch_message, move_text, plan_message, tools,
)
from engine_re.prompts import NOTES_FILE, NOTES_TEMPLATE, plan_additions
from engine_re.skeleton import render_skeleton
from engine_re.tester import StepCheck, check_step, predict
from engine_re.trace import Action, Trace, action_code, parse_moves

CURRENT_FRAME_NOTE = "The game's current frame (after step {k}), upscaled 8x:"
# After a game over the harness restarts the level itself, as the base harness does; the RESET is a real step.
AUTO_RESET_NOTE = "[harness] automatic RESET after the game over"
PLAN_TURNS = 6  # turns of a plan round without commit_moves before the reminder (PLAN_NUDGE)
COMMITTED_FILE = "engine_committed.py"  # the engine the predictions come from (the last that reproduced every step)
SUPPORT_FILE = "engine_committed.support.json"  # its support map (engine_re.support), keyed by its sha256
PLAN_SUPPORT_LINES = 4  # thin or unseparated items the PLAN message lists at most
OPENING_TEXT = (
    "The game has just started. Before your first turn the harness put level 0's first frame into make_level "
    "(recording[0].pieces_after.code()); step() does nothing yet, so your replica predicts that no move changes anything. "
    "The first moves you send show what they do: send one or a few early."
)
SKELETON_PLAY = (  # the starting engine.py's docstring, in the play mode's words
    ("reverse-engineered from a recorded run.", "modelled from the game as it is played."),
    ("the recorded actions are replayed; after each one, your final frame and the game\n  state must equal the recording.",
     "every action played so far is replayed; after each one, your final frame and\n  the game state must equal the game's."),
)


@dataclass
class PlayResult(AgentResult):
    game_id: str = ""
    outcome: str = "playing"  # "won" | "playing" (the run ended with the game where it was)
    actions: int = 0  # actions sent, the opening RESET excluded (automatic RESETs included)
    levels_completed: int = 0
    win_levels: int = 0
    actions_per_level: list = field(default_factory=list)
    baseline_actions: list | None = None
    score: float | None = None  # the official formula (live_game.LiveGame.score); None without baselines
    batches: int = 0  # commit_moves calls that sent something
    refused_batches: int = 0  # commit_moves calls refused (tests failing, bad actions, ...)
    moves_sent: int = 0
    mismatches: int = 0  # moves whose real result differed from the prediction
    auto_resets: int = 0
    plan_nudges: int = 0  # PLAN_NUDGE reminders
    # per batch: turn, first step, moves, sent, matched, mismatch (step), diff (one line), note, blind (out of step),
    # support (per planned move: move, step, weakest, untested lines, unseparated conditions; None when not predicted),
    # cut_untested (moves cut by --cut-untested)
    batch_log: list = field(default_factory=list)
    # per fit round: step, start_turn, end_turn (None: open), turns, commits, accepted, end ("accepted", "replaced",
    # "escaped"), verdict, escape_offered (the turn FIT_ESCAPE was given)
    fit_rounds: list = field(default_factory=list)
    phase_turns: dict = field(default_factory=lambda: {"plan": 0, "fit": 0})
    step_tokens: list = field(default_factory=list)  # output tokens spent up to each step after step 0 (cumulative)
    phase: str = "plan"
    committed_sha: str | None = None  # the engine the predictions come from
    fit_turns: int | None = None  # the escape hatch's setting (None: off)
    cut_untested: bool = False  # a batch is cut after its first move that runs untested code
    escapes: int = 0  # fit rounds left out of step through the escape hatch
    out_of_sync: int | None = None  # the step the engine is out of step since (None: in step)
    unexplained: list = field(default_factory=list)  # steps played out of step: replayed by the tests, never compared
    resync: dict = field(default_factory=dict)  # step -> {"level", "score"}: where the engine was put back in step


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
        fit_turns: int | None = None,
        plan_turns: int = PLAN_TURNS,
        cut_untested: bool = False,
    ):
        if model.context not in ("compact", "rebuilt"):
            raise ValueError("the play agent bounds its context by compaction (ModelConfig.context='compact') or rebuilds it "
                             "every request ('rebuilt'); engine_re.condense does not know its plan rounds")
        self.phase = "plan"  # before EngineAgent.__init__: every transcript record carries the phase
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
        self.support_path = self.dir / SUPPORT_FILE
        self.kernel = KernelClient(self.workspace, self.trace_dir, images=images, log=self._log_engine_change, history=True, play=True,
                                   support=self.support_path)
        self.fit_turns = int(fit_turns) if fit_turns else None
        self.plan_turns = max(0, int(plan_turns or 0))
        self.result = PlayResult(
            game=game, model=model.model, trace_steps=len(self.full_trace), match=self.match, interface=self.interface,
            images=images, mode="play", context=model.context, thinking_budget=model.thinking_budget,
            game_id=self.live.game_id, baseline_actions=self.live.baseline_actions, fit_turns=self.fit_turns,
            cut_untested=bool(cut_untested),
        )
        self.cut_untested = bool(cut_untested)
        self.batch_size = max(1, int(batch_size))
        self.max_actions = int(max_actions)
        self.auto_reset = auto_reset
        self.committed_sha: str | None = None
        self.committed_path = self.dir / COMMITTED_FILE
        self.listed_sha: str | None = None  # engine.py as last listed in a PLAN or FIT message
        self.batches_this_turn = 0
        self.moves_called = False  # commit_moves was called in this turn
        self.plan_idle = 0  # turns of the current plan round without commit_moves
        self.pending: dict[str, Any] | None = None  # a batch sent this turn, handled at the end of the turn
        self.fit_round: dict[str, Any] | None = None  # the open fit round (an entry of result.fit_rounds)
        self.restored_status: str | None = None  # result.json's status when the run was restored
        self.supports: dict[str, dict[str, Any]] = {}  # engine sha -> the support map of its last passing full replay
        self.support: dict[str, Any] | None = None  # the committed engine's support map (saved as SUPPORT_FILE)
        self.batch_support: list[str] = []  # the support lines of this turn's batch (commit_moves' output)
        self.batch_cut = 0  # moves of this turn's batch cut by cut_untested
        self.last_path: tuple[list[int], list[list[int]]] | None = None  # lines and conditions the last batch ran
        self.warnings: list[str] = []  # the last batch's prediction warnings (_prediction_warnings)

    # --- the trace's out-of-step bookkeeping (kept in its meta: the tests and the kernel read it there) ---------

    @property
    def out_of_sync(self) -> int | None:
        value = self.live.trace.meta.get("out_of_sync")
        return None if value is None else int(value)

    def _sync_point(self, j: int) -> dict[str, int] | None:
        """Where step j puts an engine back in step with the game: a RESET (the level restarts) or a level change
        that is not the WIN (the next level starts); None otherwise."""
        step, before = self.full_trace[j], self.full_trace[j - 1]
        if step.action.id == 0:
            return {"level": min(before.levels_completed, max(0, step.win_levels - 1)), "score": int(step.levels_completed)}
        if step.levels_completed > before.levels_completed and step.state != "WIN":
            return {"level": int(step.levels_completed), "score": int(step.levels_completed)}
        return None

    def _mark(self, j: int) -> dict[str, int] | None:
        """Step j was played out of step: a resync point (the engine is back in step), or unexplained."""
        meta = self.live.trace.meta
        point = self._sync_point(j)
        if point is not None:
            meta["resync"] = {**(meta.get("resync") or {}), str(j): point}
            meta.pop("out_of_sync", None)
        else:
            meta["ignore"] = sorted(set(meta.get("ignore") or []) | {j})
        return point

    def _go_out_of_sync(self, k: int) -> None:
        """The escape hatch is taken at step k: steps k..n-1 are unexplained (a RESET or level change among them
        still sets the level, as a resync point that is not compared), and the engine stays out of step until the
        next one the game makes."""
        meta = self.live.trace.meta
        n = len(self.full_trace)
        meta["ignore"] = sorted(set(meta.get("ignore") or []) | set(range(k, n)))
        for j in range(k + 1, n):
            point = self._sync_point(j)
            if point is not None:
                meta["resync"] = {**(meta.get("resync") or {}), str(j): point}
        meta["out_of_sync"] = k
        self.live.save(self.dir / "trace")
        self._close_fit_round("escaped")
        self.result.escapes += 1
        self.phase = "plan"
        self._focus_on(n - 1)
        self._log({"turn": self.result.turns, "out_of_sync": {"from": k, "unexplained": list(range(k, n))}})

    # --- tools -----------------------------------------------------------------------------------

    def _tools(self) -> list[dict[str, Any]]:
        schemas = tools(self.images, "play", True)
        for schema in schemas:
            if schema["function"]["name"] == "commit_moves":
                schema["function"]["description"] = commit_moves_description(self.batch_size)
        return schemas

    def _commit_hint(self, report: Any) -> str:
        if isinstance(report, str) or not report.passed or report.level is not None:
            return ""
        skipped = " (the unexplained steps are not compared)" if report.ignored else ""
        return COMMIT_HINT_PLAY["fit" if self.phase == "fit" else "plan"].format(k=self.focus, skipped=skipped)

    def _tool_commit_engine(self, message: Any = None) -> str:
        """As in the stepwise harness: the tests on steps 0..k must pass; the commit is applied after the
        turn (_advance), which replays every step played and asks for the next moves when all pass."""
        text = super()._tool_commit_engine(message)
        if text.startswith("Committed:"):
            return (f"Committed: steps 0-{self.focus} pass. After this turn the harness replays every step played so far "
                    "and, when all pass, asks you to plan the next moves (the next message shows the game's current frame).")
        return text

    def _refuse(self, text: str) -> str:
        self.result.refused_batches += 1
        return text

    def _tool_commit_moves(self, actions: Any = None, note: Any = None) -> str:
        """Send moves to the real game (PLAY_DESIGN.md 3.3): the tests first, nothing sent if they fail;
        then each move predicted, sent and checked, stopping at the first difference."""
        self.moves_called = True
        if self.batches_this_turn:
            return self._refuse("Not sent: one commit_moves call per turn. This turn's batch and its result are above; the next "
                                "message says what to do next. Send the next batch in your next turn.")
        if not isinstance(note, str) or not note.strip():
            return self._refuse("Not sent: commit_moves needs a note (one or two sentences: what the batch is meant to do and "
                                "what your replica predicts).")
        empty = 'Not sent: actions must be a non-empty list of moves, e.g. ["Action(4)", "Action(6, x=12, y=40)"].'
        if not isinstance(actions, (list, str)) or not actions:
            return self._refuse(empty)
        try:  # a list of moves, or a printed list of Actions as one string
            acts = parse_moves(actions)
        except (ValueError, TypeError) as exc:
            return self._refuse(f"Not sent: {exc}.")
        if not acts:
            return self._refuse(empty)
        if len(acts) > self.batch_size:
            return self._refuse(f"Not sent: at most {self.batch_size} moves per call; you gave {len(acts)}. Send the first "
                                f"{self.batch_size}, look at the result, then the rest.")
        allowed = set(self.full_trace[0].available_actions) | {0}
        for a in acts:
            if a.id == 6 and not (0 <= int(a.x) <= 63 and 0 <= int(a.y) <= 63):
                return self._refuse(f"Not sent: the click {action_code(a)} is off the 64x64 screen (x and y must be 0-63).")
            if a.id not in allowed:
                return self._refuse(f"Not sent: this game does not accept {move_text(a)}. "
                                    + accepted_actions_text(self.full_trace[0].available_actions))
        if self.live.won:
            return "Not sent: the game is won; nothing more to play."
        left = self.max_actions - self.live.actions
        if left <= 0:
            return self._refuse(f"Not sent: the run's {self.max_actions} actions are used up.")
        if self.live.game_over and acts[0].id != 0:
            return self._refuse("Not sent: the game is over, so only RESET is accepted now. Start the batch with RESET (it "
                                "restarts the level).")
        cut = max(0, len(acts) - left)
        acts = acts[:left]
        n = len(self.full_trace)
        if self.out_of_sync is not None:
            return self._blind_batch(acts, note, cut)
        if self.focus != n - 1:  # leaving a fit round on an earlier step: the claim is that every step passes
            self._focus_on(n - 1)
        report = self._run_tests(None, 1, auto=False)
        if isinstance(report, str):
            return self._refuse("Not sent: the tests could not run. " + report)
        if not report.passed:
            first = report.first_fail
            if self._escape_open() and first is not None:
                k = int(self.fit_round["step"])
                if first < k:
                    return self._refuse(
                        f"Not sent: playing on out of step is offered from step {k} on, but engine.py now fails step {first}, "
                        "which your committed replica reproduces. Fix it (undo_edit() brings back earlier versions), then send "
                        "again. The report:\n\n" + _truncate(report.text, REPORT_CHARS))
                self._go_out_of_sync(first)
                return self._blind_batch(acts, note, cut)
            self.phase = "fit"
            k = n - 1 if first is None else int(first)
            if k < n - 1:
                self._focus_on(k)
            if self.fit_round is None or self.fit_round.get("end_turn") is not None or self.fit_round["step"] != k:
                self._new_fit_round(k, "commit_moves refused: the tests fail here")
            self._log({"turn": self.result.turns, "refused_batch": {"reason": "tests fail", "first_fail": first, "note": note}})
            return self._refuse(
                f"Not sent: your replica does not reproduce the game so far (steps 0-{n - 1}), so no move was sent. Fix "
                "engine.py first, then commit_engine(message) and plan again. The report:\n\n" + _truncate(report.text, REPORT_CHARS))
        commit, self.commit = self.commit, None  # a commit_engine earlier in this turn: this batch's commit
        sha = self._engine_hash()
        if commit is not None and commit["engine_sha"] == sha:
            self._record_commit(commit["message"])
        elif sha != self.committed_sha:  # an engine that passes everything, submitted by the batch itself
            self._record_commit(f"(commit_moves) {note}", implicit=True)
        self.batches_this_turn += 1
        self.result.batches += 1
        outcomes, dropped = self._play(acts, note)
        self.pending = {"outcomes": outcomes, "dropped": dropped + cut, "note": note}
        first_step = outcomes[0]["index"] if outcomes else n
        lines = [f"Sent {len(outcomes)} of {len(acts) + cut} move(s) (steps {first_step}-{first_step + len(outcomes) - 1}):"] if outcomes else []
        lines += batch_lines(self.full_trace, first_step, outcomes)
        lines += self.batch_support  # what each planned move's prediction rested on, computed before anything was sent
        if self.batch_cut:
            j = len(acts) - self.batch_cut
            lines.append(f"The batch was cut after move {j}, the first to run code no recorded step has run: it is the experiment "
                         f"(the harness's cut-untested rule); the {self.batch_cut} move(s) after it were not sent.")
            dropped -= self.batch_cut
        lines += self._animation_lines(outcomes)
        last = outcomes[-1] if outcomes else None
        if last is not None and not last["ok"]:
            lines.append(f"The batch stopped at step {last['index']}: the game's result differs from your replica's prediction"
                         + (f"; {dropped} move(s) not sent" if dropped else "") + ". A fit round opens: the next message shows the comparison.")
        elif last is not None and last["state"] == "WIN":
            lines.append("The game is won.")
        elif last is not None and last["state"] == "GAME_OVER":
            lines.append("The game is over" + (": the harness will RESET the level (checked against your replica too), then ask for the next moves."
                                               if self.auto_reset else ": only RESET is accepted now."))
        elif last is not None and last.get("level_solved"):
            lines.append(f"Level {last['level']} is solved; the batch stops there" + (f" ({dropped} move(s) not sent)" if dropped else "")
                         + ". The next message shows the new level.")
        else:
            lines.append("Every move matched your replica. The next message asks for the next moves.")
        if cut:
            lines.append(f"The last {cut} move(s) of the batch were cut: the run allows {self.max_actions} actions in all.")
        lines += self.warnings
        return "\n".join(lines)

    def _blind_batch(self, acts: list[Action], note: str, cut: int) -> str:
        """commit_moves while the engine is out of step: the moves are sent unchecked, up to the first resync point."""
        since = self.out_of_sync
        self.batches_this_turn += 1
        self.result.batches += 1
        outcomes = self._play_blind(acts, note)
        dropped = len(acts) - len(outcomes)
        self.pending = {"outcomes": outcomes, "dropped": dropped + cut, "note": note, "blind": True}
        first_step = outcomes[0]["index"]
        lines = [f"Sent {len(outcomes)} of {len(acts) + cut} move(s) (steps {first_step}-{first_step + len(outcomes) - 1}), not "
                 f"checked: your replica is out of step with the game since step {since}."]
        lines += batch_lines(self.full_trace, first_step, outcomes)
        last = outcomes[-1]
        if last.get("resync"):
            lines.append(f"Your replica is back in step with the game at step {last['index']}" + (f" ({dropped} move(s) not sent)" if dropped else "")
                         + ". After this turn the harness replays every step (the unexplained ones are not compared) and, when "
                         "they pass, asks for the next moves.")
        elif last["state"] == "WIN":
            lines.append("The game is won.")
        elif last["state"] == "GAME_OVER":
            lines.append("The game is over: " + ("the harness will RESET the level, which puts your replica back in step." if self.auto_reset
                                                 else "only RESET is accepted now; it puts your replica back in step."))
        else:
            lines.append("Your replica is still out of step. The next message asks for the next moves.")
        if cut:
            lines.append(f"The last {cut} move(s) of the batch were cut: the run allows {self.max_actions} actions in all.")
        return "\n".join(lines)

    # --- playing ---------------------------------------------------------------------------------

    def _record_commit(self, message: str, implicit: bool = False) -> None:
        """engine.py, which reproduces every step played, becomes the committed engine (an `advances` entry and a
        "commit" record, as an accepted commit_engine)."""
        sha = self._engine_hash()
        self._keep_committed(sha)
        entry = {"turn": self.result.turns, "fixed": self.focus, "next": None, "message": message, "engine_sha": sha,
                 "version": self._version_of(sha)}
        if implicit:
            entry["implicit"] = True
        self.result.advances.append(entry)
        self._log({"turn": self.result.turns, "commit": {k: v for k, v in entry.items() if k != "turn"}})
        if self.fit_round is not None and self.fit_round.get("end_turn") is None:
            self.fit_round["commits"] += 1

    def _tested(self, report: Any) -> None:
        """Keep the support map of an engine that passes a full replay of everything played (for its commit)."""
        smap = getattr(report, "support", None)
        if smap and report.passed and report.level is None:
            known = self.supports.get(smap["engine_sha"])
            if known is None or known["steps"] <= smap["steps"]:
                self.supports[smap["engine_sha"]] = smap
                for old in list(self.supports)[:-4]:  # a few recent engines are enough
                    self.supports.pop(old)

    def _save_support(self, smap: dict[str, Any] | None) -> None:
        """The committed engine's support map: kept, and written beside engine_committed.py (removed when unknown)."""
        self.support = smap
        if smap is None:
            self.support_path.unlink(missing_ok=True)
            return
        self.support_path.write_text(json.dumps(smap), encoding="utf-8")

    def _keep_committed(self, sha: str) -> None:
        """The engine with this hash is the one the predictions come from: a copy in engine_committed.py, and its
        support map beside it (SUPPORT_FILE)."""
        source = self.engine_path
        if self._engine_hash() != sha:  # (engine.py changed since: the saved version with that hash)
            version = self._version_of(sha)
            source = self.dir / "engine_versions" / f"v{version or 0:04d}.py"
            if version is None or not source.exists():
                return
        shutil.copy(source, self.committed_path)
        self.committed_sha = self.result.committed_sha = sha
        self._save_support(self.supports.get(sha))

    def _predict(self, acts: list[Action]) -> tuple[dict[str, Any], list[Any]]:
        """The committed engine's prediction for `acts` after everything played (tester.predict)."""
        return predict(self._predicting_engine(), self.full_trace, acts, scratch_root=self.dir)

    def _predicting_engine(self) -> Path:
        return self.committed_path if self.committed_path.exists() else self.engine_path

    def _fold(self, prediction: dict[str, Any], upto: int) -> dict[str, Any] | None:
        """The support map of the prediction's engine over the steps played before position `upto` (the unexplained
        ones left out): every one of them passes, since commit_moves sends nothing otherwise."""
        ignore, _ = sync_points(self.live.trace.meta)
        data = self._predicting_engine().read_bytes()
        positions = {str(i): i for i in range(upto) if i not in ignore}
        return sup.fold_result(prediction, positions, data.decode("utf-8", "replace"), hashlib.sha256(data).hexdigest())

    @staticmethod
    def _move_paths(prediction: dict[str, Any], smap: dict[str, Any] | None, n: int, m: int) -> list[dict[str, Any] | None]:
        """Each planned move's path against the support map (support.path_support); None when it was not predicted."""
        if smap is None:
            return [None] * m
        executed, evaluated = prediction.get("executed") or {}, prediction.get("evaluated") or {}
        out = []
        for j in range(m):
            key = str(n + j)
            out.append(sup.path_support(smap, executed[key], evaluated.get(key)) if key in executed else None)
        return out

    @staticmethod
    def _compact_path(ps: dict[str, Any]) -> dict[str, Any]:
        """A move's path support as batch_log and the move records keep it."""
        return {"weakest": ps["weakest"], "weakest_lines": ps["weakest_lines"][:12], "untested": ps["untested"][:60],
                "thin": ps["thin"][:60], "unseparated": ps["unseparated"][:6]}

    @staticmethod
    def _support_lines(acts: list[Action], paths: list[dict[str, Any] | None], smap: dict[str, Any] | None) -> list[str]:
        """commit_moves' lines on what each move's prediction rests on: one per move whose path is thin, untested or
        relies on a condition never separated, one for the others together."""
        if smap is None or not any(paths):
            return []
        exe = smap.get("lines") or {}
        lines = [f"Support of your replica's predictions (how many of the {smap['steps']} recorded steps ran the code each move "
                 "runs; thin: fewer than 3):"]
        solid = []
        weak: dict[str, list[str]] = {}  # the same words for several moves: one line
        for j, (act, ps) in enumerate(zip(acts, paths), 1):
            if ps is None:
                continue
            if ps["weakest"] is not None and ps["weakest"] >= sup.THIN_SUPPORT and not ps["unseparated"]:
                solid.append((j, ps["weakest"]))
                continue
            weak.setdefault(sup.move_support_text(ps, exe), []).append(f"{j} {action_code(act)}")
        for text, moves in weak.items():
            lines.append(f"  move{'s' if len(moves) > 1 else ''} {', '.join(moves)}: {text}")
        if solid:
            moves = ", ".join(str(j) for j, _ in solid)
            if len(solid) == 1:
                lines.append(f"  move {moves}: its path is supported by at least {solid[0][1]} steps")
            else:
                lines.append(f"  moves {moves}: their paths are supported by at least {min(w for _, w in solid)} steps each")
        return lines

    @staticmethod
    def _verdict(check: StepCheck, real: Any, got: dict[str, Any] | None, error: str | None) -> str:
        if got is None:
            last = (error or "").strip().splitlines()[-1][:160] if error else "no prediction"
            return f"your replica raised an error ({last})"
        parts = []
        if "final frame" in check.problems:
            parts.append("the final frame differs")
        for name, shown in (("state", "outcome"), ("levels_completed", "levels completed"), ("win_levels", "win levels"), ("available_actions", "available actions")):
            if name in check.problems:
                parts.append(f"{shown}: the game says {getattr(real, name)!r}, your replica {got.get(name)!r}")
        return "; ".join(parts) if parts else "; ".join(check.problems)

    def _played(self) -> None:
        """A step was appended to the live trace: count the tokens spent so far and save the trace."""
        self.result.step_tokens.append(int(self.result.usage.completion_tokens))
        self.live.save(self.dir / "trace")

    def _play(self, acts: list[Action], note: str) -> tuple[list[dict[str, Any]], int]:
        """Send `acts` one at a time, each checked against the committed engine's prediction; stop at the first
        difference, a solved level or the end of the game. Returns the outcomes and how many were not sent."""
        n = len(self.full_trace)
        prediction, frames = self._predict(acts)
        predicted = prediction.get("steps") or []
        error = prediction.get("error")
        smap = self._fold(prediction, n)
        paths = self._move_paths(prediction, smap, n, len(acts))
        planned = list(acts)
        self.batch_support = self._support_lines(acts, paths, smap)
        self.batch_cut = 0
        if self.cut_untested:  # the first move that runs untested code is the experiment: the batch ends there
            j = next((j for j, ps in enumerate(paths) if ps is not None and ps["untested"]), None)
            if j is not None and j + 1 < len(acts):
                self.batch_cut = len(acts) - j - 1
                acts = acts[: j + 1]
        self.warnings = self._prediction_warnings(acts, predicted, frames, n) if note != AUTO_RESET_NOTE else []
        outcomes: list[dict[str, Any]] = []
        for i, act in enumerate(acts):
            pos = n + i
            level_before = self.live.level
            real = self.live.perform(act)
            self._played()
            got = predicted[pos] if pos < len(predicted) else None
            got_frames = frames[pos] if pos < len(frames) else None
            if got is None:
                check = StepCheck(real.index, False, False, ["engine error"])
            else:
                check = check_step(real, got, got_frames, self.match)
            outcome = {
                "index": real.index, "label": action_code(act), "ok": check.ok, "warning": check.warning,
                "verdict": "" if check.ok else self._verdict(check, real, got, error),
                "frames": real.n_frames, "level": level_before, "state": real.state,
                "level_solved": real.levels_completed > self.full_trace[pos - 1].levels_completed,
                "levels_completed": real.levels_completed, "auto": note == AUTO_RESET_NOTE,
            }
            if paths[i] is not None:
                outcome["support"] = self._compact_path(paths[i])
            outcomes.append(outcome)
            self.result.moves_sent += 1
            if not check.ok:
                self.result.mismatches += 1
            self._log({"turn": self.result.turns, "move": outcome})
            if not check.ok or real.state in ("WIN", "GAME_OVER") or outcome["level_solved"]:
                break
        self._focus_on(len(self.full_trace) - 1)
        miss = next((o for o in outcomes if not o["ok"]), None)
        entry = {
            "turn": self.result.turns, "first_step": n, "moves": [action_code(a) for a in planned], "sent": len(outcomes),
            "matched": sum(bool(o["ok"]) for o in outcomes), "mismatch": miss["index"] if miss else None,
            "diff": miss["verdict"] if miss else None, "note": note,
            "support": [None if ps is None else {"move": j + 1, "step": n + j, **self._compact_path(ps)} for j, ps in enumerate(paths)],
        }
        if self.batch_cut:
            entry["cut_untested"] = self.batch_cut
        if self.warnings:
            entry["warnings"] = list(self.warnings)
        self.result.batch_log.append(entry)
        self._log({"turn": self.result.turns, "batch": self.result.batch_log[-1]})
        # The moves that matched are passing steps now: the committed engine's map grows with them.
        matched = n + sum(1 for _ in itertools.takewhile(lambda o: o["ok"], outcomes))
        if smap is not None and self.committed_path.exists() and self.committed_sha is not None:
            self._save_support(self._fold(prediction, matched) if matched > n else smap)
        executed, evaluated = prediction.get("executed") or {}, prediction.get("evaluated") or {}
        keys = [str(n + i) for i in range(len(outcomes)) if str(n + i) in executed]
        if note != AUTO_RESET_NOTE:  # (the harness's RESET is not the model's plan)
            self.last_path = (sorted({line for k in keys for line in executed[k]}),
                              [c for k in keys for c in evaluated.get(k) or []]) if keys else None
        return outcomes, len(acts) - len(outcomes) + self.batch_cut

    def _play_blind(self, acts: list[Action], note: str) -> list[dict[str, Any]]:
        """Out of step: send `acts` unchecked, each step unexplained, up to the first resync point (a RESET or a
        level change), a game over or the WIN."""
        n = len(self.full_trace)
        outcomes: list[dict[str, Any]] = []
        for act in acts:
            level_before = self.live.level
            real = self.live.perform(act)
            point = self._mark(real.index)
            self._played()  # saves the trace with its meta
            outcome = {
                "index": real.index, "label": action_code(act), "ok": None, "warning": None, "verdict": "",
                "frames": real.n_frames, "level": level_before, "state": real.state,
                "level_solved": real.levels_completed > self.full_trace[real.index - 1].levels_completed,
                "levels_completed": real.levels_completed, "auto": note == AUTO_RESET_NOTE, "resync": point is not None,
            }
            outcomes.append(outcome)
            self.result.moves_sent += 1
            self._log({"turn": self.result.turns, "move": outcome})
            if point is not None:
                self._log({"turn": self.result.turns, "resync": {"step": real.index, **point}})
            if point is not None or real.state in ("WIN", "GAME_OVER"):
                break
        self._focus_on(len(self.full_trace) - 1)
        self.result.batch_log.append({
            "turn": self.result.turns, "first_step": n, "moves": [action_code(a) for a in acts], "sent": len(outcomes),
            "matched": 0, "mismatch": None, "diff": None, "note": note, "blind": True,
        })
        self._log({"turn": self.result.turns, "batch": self.result.batch_log[-1]})
        return outcomes

    # --- ported from the base harness: warnings from the prediction, notes.md ----------------------------------

    def _prediction_warnings(self, acts: list[Action], predicted: list[Any], frames: list[Any], n: int) -> list[str]:
        """What the committed replica predicts for the batch that is worth a word before it is sent (warnings only:
        the batch is sent as it is, since a deliberate probe is legitimate; the base harness's DEATH_GUARD_ADDENDUM and
        NOOP_GUARD_ADDENDUM, without their refusals): a game over at some move, and runs of moves that change nothing
        in the replica (the same final frame and status as before the move)."""
        out: list[str] = []
        noop: list[int] = []  # 0-based positions in the batch

        def flush() -> None:
            if not noop:
                return
            which = f"move {noop[0] + 1}" if len(noop) == 1 else f"moves {noop[0] + 1}-{noop[-1] + 1}"
            shown = ", ".join(action_code(acts[i]) for i in noop)
            verb = "changes" if len(noop) == 1 else "change"
            out.append(f"[harness] Warning: {which} ({shown}) {verb} nothing in your replica (the same frame and status as "
                       "before): if your replica is right, an action spent for nothing; a deliberate probe of that rule is fine.")
            noop.clear()

        for i, act in enumerate(acts):
            pos = n + i
            got = predicted[pos] if pos < len(predicted) else None
            if got is None:
                break
            prev = predicted[pos - 1] if pos - 1 < len(predicted) else None
            frame = frames[pos][-1] if pos < len(frames) and len(frames[pos]) else None
            prev_frame = frames[pos - 1][-1] if 0 < pos <= len(frames) and len(frames[pos - 1]) else None
            same = (prev is not None and frame is not None and prev_frame is not None and np.array_equal(frame, prev_frame)
                    and all(got.get(k) == prev.get(k) for k in ("state", "levels_completed")))
            if same:
                noop.append(i)
            else:
                flush()
            if got.get("state") == "GAME_OVER":
                after = (f"; the {len(acts) - i - 1} move(s) after it would not be sent" if i < len(acts) - 1 else "")
                out.append(f"[harness] Warning: your replica predicts a game over at move {i + 1} ({action_code(act)}){after}, and "
                           "the harness then RESETs the level (one more action). Sent anyway: a deliberate probe is fine.")
                break
        flush()
        return out

    ANIMATION_LINES = 3  # animated moves of a batch whose transient cells commit_moves' output names

    def _animation_lines(self, outcomes: list[dict[str, Any]]) -> list[str]:
        """One line per move of the batch that animated with transient cells (cells that changed and changed back,
        in no frame the model can otherwise reach; engine_re.animation), at most ANIMATION_LINES: the base harness's
        describe_animation, which is silent when no cell changed back."""
        out: list[str] = []
        for o in outcomes:
            k = int(o["index"])
            step = self.full_trace[k]
            if step.n_frames <= 1:
                continue
            found = animation.digest(self.full_trace[k - 1].last if k else None, step.frames)
            if found is None or not found.transient:
                continue
            if len(out) == self.ANIMATION_LINES:
                out.append("[harness] (more moves of this batch animated; recording[k].animation shows each)")
                break
            out.append(f"[harness] Step {k} ({o['label']}) animated over {found.frames} frames: {found.transient_text()}. "
                       f"recording[{k}].animation has its timeline.")
        return out

    def _notes(self) -> str | None:
        """notes.md in the workspace (the model's goal model, open questions and plan), or None."""
        path = self.workspace / NOTES_FILE
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _init_notes(self) -> None:
        """notes.md with its three headings, when the workspace has none."""
        path = self.workspace / NOTES_FILE
        if not path.exists():
            self.workspace.mkdir(parents=True, exist_ok=True)
            path.write_text(NOTES_TEMPLATE, encoding="utf-8")

    # --- phases ----------------------------------------------------------------------------------

    def _budget_line(self) -> str:
        u = self.result.usage
        counts = self.live.actions_per_level()
        per_level = ", ".join(f"level {i}: {c}" for i, c in enumerate(counts) if c)
        return (f"Actions played: {self.live.actions} of at most {self.max_actions} ({per_level or 'none yet'}). Turns: "
                f"{self.result.turns} of {self.budget.max_turns}; minutes: {self._elapsed_minutes():.0f} of {self.budget.max_minutes:.0f}; "
                f"output tokens: {u.completion_tokens:,} of {self.budget.max_output_tokens:,}.")

    def _read_engine(self, fold: bool, max_chars: int) -> str:
        """engine.py with anchors and, once the committed engine has a support map, each line's support in the margin
        (support.margins: the count, 0 untested, "new" for a line changed since the commit), as read_file() shows it."""
        text = self.engine_path.read_text(encoding="utf-8")
        margin = sup.margins(self.support, text) if self.support else None
        return hashline.render_read(text, max_chars=max_chars, fold=fixed_block_lines(text) if fold else None, margin=margin)

    def _plan_support(self) -> str:
        """The PLAN message's support items (support.plan_items): the thin rules on the last batch's path and the
        outcome rules no step ran or never separated; empty when there are none."""
        if not self.support or not self.committed_path.exists():
            return ""
        items = sup.plan_items(self.support, self.committed_path.read_text(encoding="utf-8"), self.last_path, PLAN_SUPPORT_LINES)
        return "\n".join(items)

    def _engine_listing_if_changed(self) -> str:
        sha = self._engine_hash()
        if sha == self.listed_sha:
            return ""
        self.listed_sha = sha
        return self._read_engine(fold=True, max_chars=READ_CHARS_IN_MESSAGES)

    def _engine_note(self, report: bool = False) -> str:
        """A sentence when engine.py is not the committed engine (it changed after the last commit or batch)."""
        if self.committed_sha is None or self._engine_hash() == self.committed_sha or (self.out_of_sync is not None and not report):
            return ""
        if report:
            return "engine.py has changed since your last commit: the report below is of engine.py as it is now."
        if self.tested_hash == self._engine_hash() and self.passed:
            return ("engine.py has changed since your last commit and still reproduces every step played; commit_moves "
                    "commits it.")
        return ("engine.py has changed since your last commit and does not reproduce every step played (the last test report "
                "shows where): commit_moves sends nothing until it does; undo_edit() brings back earlier versions.")

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

    def _hide_old_images(self) -> None:
        """A phase message carries the latest images: the earlier ones become placeholders (logged, for a resume)."""
        if self.images:
            self._hide_images(self.messages)
            self._log({"turn": self.result.turns, "hide_images": True})

    def _enter_plan(self, last_batch: str, say: bool = True) -> str | list[dict[str, Any]]:
        """The PLAN message (said to the model unless `say` is False; the content is returned either way)."""
        self.phase = "plan"
        self._close_fit_round("accepted")
        self.fit_round = None
        self.plan_idle = 0
        n = len(self.full_trace)
        ignore, _ = sync_points(self.live.trace.meta)
        text = plan_message(
            self.game, self.full_trace, last_batch=last_batch, budget_line=self._budget_line(), batch_size=self.batch_size,
            engine_read=self._engine_listing_if_changed(), kernel_names=kernel_names_text(*self.kernel.names()),
            baseline=self.live.baseline_actions, unexplained=sorted(ignore), out_of_sync=self.out_of_sync,
            engine_note=self._engine_note(), images=self.images,
            support_note=self._plan_support() if self.out_of_sync is None else "",
        )
        text = plan_additions(text, self.full_trace, notes=self._notes(), images=self.images)
        self._log({"turn": self.result.turns, "plan": {"step": n - 1, "steps": n, "actions": self.live.actions, "level": self.live.level,
                                                        "out_of_sync": self.out_of_sync}})
        if say:
            self._hide_old_images()
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}, *self._frame_part()]
        content: str | list[dict[str, Any]] = parts if len(parts) > 1 else text
        if say:
            self._say("user", content, phase="plan")
        return content

    def _close_fit_round(self, how: str) -> None:
        r = self.fit_round
        if r is None or r.get("end_turn") is not None:
            return
        r["end_turn"] = self.result.turns
        r["turns"] = self.result.turns - int(r["start_turn"])
        r["end"] = how
        r["accepted"] = how == "accepted"

    def _new_fit_round(self, k: int, verdict: str) -> None:
        if self.fit_round is not None and self.fit_round.get("end_turn") is None:
            self._close_fit_round("accepted" if self.fit_round["commits"] else "replaced")
        self.fit_round = {"step": k, "start_turn": self.result.turns, "end_turn": None, "turns": None, "commits": 0,
                          "accepted": None, "verdict": verdict}
        self.result.fit_rounds.append(self.fit_round)

    def _enter_fit(self, k: int, verdict: str, dropped: int, say: bool = True, predicted: bool = True,
                   path: dict[str, Any] | None = None) -> str | list[dict[str, Any]]:
        """Step k differs (from the prediction, or, `predicted` False, from what the engine now gives): the FIT
        message with the test report on steps 0..k. `path`: the move's path support (its outcome's "support")."""
        self.phase = "fit"
        self.passed = False
        self._focus_on(k)
        report = self._step_report("advance")
        self.tested_hash = self._engine_hash()
        self._new_fit_round(k, verdict)
        text = mismatch_message(
            self.full_trace, k, verdict, dropped, report, engine_read=self._engine_listing_if_changed(),
            kernel_names=kernel_names_text(*self.kernel.names()), auto_reset=self.auto_reset,
            engine_note=self._engine_note(report=True), predicted=predicted,
            support_note=sup.mismatch_support_text(path, (self.support or {}).get("lines")),
        )
        self._log({"turn": self.result.turns, "step_start": {"step": k, "steps": len(self.full_trace), "report": report, "verdict": verdict}})
        if say:
            self._hide_old_images()
        content = self._opening_content(text)
        if say:
            self._say("user", content, phase="fit")
        return content

    def _opening_phase(self) -> str | None:
        return self.phase  # set by _enter_plan / _enter_fit when the opening message was made (_stepwise_start)

    def _after_sync(self, last_batch: str, say: bool = True) -> str | list[dict[str, Any]] | None:
        """The engine reproduces everything played (or is out of step): a RESET after a game over, then the next
        message (PLAN, or FIT when the RESET differs). Returns its content, or None when the game is won."""
        if self.live.won:
            return None
        if self.live.game_over and self.auto_reset and self.live.actions < self.max_actions:
            self.result.auto_resets += 1
            if self.out_of_sync is not None:
                o = self._play_blind([Action(0)], AUTO_RESET_NOTE)[0]
                return self._after_resync(
                    f"{last_batch} The game was over, so the harness sent a RESET (step {o['index']}, an action), which puts your "
                    "replica back in step.", 0, say)
            outcomes, _ = self._play([Action(0)], AUTO_RESET_NOTE)
            o = outcomes[0]
            if not o["ok"]:
                return self._enter_fit(o["index"], o["verdict"], 0, say=say, path=o.get("support"))
            last_batch += f" The game was over, so the harness sent a RESET (step {o['index']}, an action): the level restarted as your replica predicted."
        elif self.live.game_over:
            last_batch += " The game is over: only RESET is accepted now."
        return self._enter_plan(last_batch, say=say)

    def _after_resync(self, last_batch: str, dropped: int, say: bool = True) -> str | list[dict[str, Any]] | None:
        """The engine was put back in step with the game: every step is tested (the unexplained ones are not
        compared); PLAN when they pass, else FIT on the first that fails."""
        k = self._replay_all()
        if k is not None:
            return self._enter_fit(k, "your replica does not reproduce it now that it is back in step with the game", dropped,
                                   say=say, predicted=False)
        if self._engine_hash() != self.committed_sha:
            self._record_commit("(back in step with the game) engine.py reproduces every step played", implicit=True)
        self._focus_on(len(self.full_trace) - 1)
        return self._after_sync(last_batch + " Every step played passes with your replica again (the unexplained ones are not compared).", say)

    @staticmethod
    def _batch_summary(pending: dict[str, Any]) -> str:
        outcomes, dropped = pending["outcomes"], pending["dropped"]
        last = outcomes[-1] if outcomes else None
        if pending.get("blind"):
            text = f"Your last batch: {len(outcomes)} move(s) sent while your replica was out of step (not checked)"
            if last is not None and last.get("resync"):
                text += f"; at step {last['index']}, {last['label']}, it is back in step"
        elif last is not None and not last["ok"]:  # (a commit in the same turn fixed it)
            text = (f"Your last batch: {len(outcomes)} move(s) sent; step {last['index']}, {last['label']}, differed from your "
                    f"replica's prediction ({last['verdict']})")
        else:
            text = f"Your last batch: {len(outcomes)} move(s) sent, all as your replica predicted"
            if last is not None and last.get("level_solved") and last["state"] != "WIN":
                text += (f"; level {last['level']} solved, and your make_level({last['level'] + 1}) drew the new level's start "
                         "as the game did, so the batch stopped there")
        return text + (f" ({dropped} move(s) not sent)" if dropped else "") + "."

    def _after_batch(self, pending: dict[str, Any]) -> bool:
        """The next message after a turn's batch; False when the game is won."""
        if self.live.won:
            return False
        outcomes, dropped = pending["outcomes"], pending["dropped"]
        last = outcomes[-1] if outcomes else None
        if pending.get("blind"):
            if last is not None and last.get("resync"):
                return self._after_resync(self._batch_summary(pending), dropped) is not None
            return self._after_sync(self._batch_summary(pending)) is not None
        if last is not None and not last["ok"]:
            self._enter_fit(last["index"], last["verdict"], dropped, path=last.get("support"))
            return True
        return self._after_sync(self._batch_summary(pending)) is not None

    def _advance(self, commit: dict[str, Any], pending: dict[str, Any] | None = None) -> bool:
        """A commit was accepted (steps 0..k pass): replay every step played; another failing step is the
        next fit (advance_message); when all pass, the engine is committed and the model plans again. False
        when the game is won."""
        fixed = self.focus
        k = self._replay_all()
        n = len(self.full_trace)
        entry = {"turn": self.result.turns, "fixed": fixed, "next": k, **commit}
        self.result.advances.append(entry)
        self._log({"turn": self.result.turns, "commit": {key: v for key, v in entry.items() if key != "turn"}})
        if self.fit_round is not None and self.fit_round.get("end_turn") is None:
            self.fit_round["commits"] += 1
        if k is not None:
            self._focus_on(k)
            self.passed = False
            text = self._step_report("advance")
            self.tested_hash = self._engine_hash()
            self._new_fit_round(k, "fails after the commit")
            self.phase = "fit"
            self._log({"turn": self.result.turns, "advance": {"fixed": fixed, "next": k, "steps": n, "report": text}})
            engine_read = self._engine_listing_if_changed()
            names = kernel_names_text(*self.kernel.names())
            self._hide_old_images()
            self._say("user", self._opening_content(advance_message(self.full_trace, fixed, k, text, True, engine_read, names)), phase="fit")
            return True
        self._keep_committed(commit["engine_sha"])
        self._focus_on(n - 1)
        done = f"Commit accepted: your replica now reproduces steps 0-{n - 1}" + (
            " (but the unexplained ones, which are not compared)." if sync_points(self.live.trace.meta)[0] else ".")
        if pending is not None:
            done = self._batch_summary(pending) + " " + done
        return self._after_sync(done) is not None

    def _end_of_turn(self) -> bool:
        self.result.phase_turns[self.phase] = self.result.phase_turns.get(self.phase, 0) + 1
        self.batches_this_turn = 0
        pending, self.pending = self.pending, None
        commit, self.commit = self.commit, None
        if commit is not None:  # a commit_engine after this turn's batch (one before it was the batch's commit)
            if pending is not None and not pending.get("blind"):
                last = pending["outcomes"][-1] if pending["outcomes"] else None
                if last is not None and not last["ok"]:  # the batch's difference, fixed by the commit in the same turn
                    self._new_fit_round(last["index"], last["verdict"])
            go_on = self._advance(commit, pending)
        elif pending is not None:
            go_on = self._after_batch(pending)
        else:
            go_on = True
        if not go_on or self.live.won:
            return self._finish_won()
        return True

    def _finish_won(self) -> bool:
        self.result.status = "won"
        self.result.outcome = "won"
        return False

    def _turn_notes(self) -> None:
        """Before the turn's images: the plan nudge, and the escape hatch's offer."""
        if self.phase == "plan" and not self.moves_called and self.commit is None:
            self.plan_idle += 1
            if self.plan_turns and self.plan_idle % self.plan_turns == 0:
                self._add_to_last(PLAN_NUDGE.format(n=self.plan_idle))
                self.result.plan_nudges += 1
                self._log({"turn": self.result.turns, "plan_nudge": self.plan_idle})
        elif self.moves_called:
            self.plan_idle = 0
        self.moves_called = False
        r = self.fit_round
        if (self.fit_turns and self.phase == "fit" and r is not None and r.get("end_turn") is None and r.get("escape_offered") is None
                and self.commit is None and self.out_of_sync is None and self.result.turns - int(r["start_turn"]) >= self.fit_turns):
            r["escape_offered"] = self.result.turns
            n = self.result.turns - int(r["start_turn"])
            self._add_to_last(FIT_ESCAPE.format(n=n, k=r["step"]))
            self._log({"turn": self.result.turns, "fit_escape": {"step": r["step"], "turns": n}})

    def _escape_open(self) -> bool:
        r = self.fit_round
        return r is not None and r.get("end_turn") is None and r.get("escape_offered") is not None

    def _test_nudge_due(self) -> bool:
        return self.phase == "fit" and super()._test_nudge_due()

    def _log(self, record: dict[str, Any]) -> None:
        record.setdefault("phase", getattr(self, "phase", "plan"))
        super()._log(record)

    # --- the run ---------------------------------------------------------------------------------

    def setup(self) -> None:
        if not self.engine_path.exists():  # the starting engine.py, its docstring in the play mode's words
            text = render_skeleton(self.game, self.full_trace[0].available_actions)
            for old, new in SKELETON_PLAY:
                text = text.replace(old, new, 1)
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.engine_path.write_text(text, encoding="utf-8")
        self._init_notes()
        super().setup()

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
            self._keep_committed(self._engine_hash())
            text = OPENING_TEXT if n == 1 else f"{self.live.actions} action(s) were played before this session."
            content = self._after_sync(text, say=False)
            return content if content is not None else "The game is won."
        return self._enter_fit(first, "your replica does not reproduce this step", 0, say=False, predicted=False)

    def _rebuild_conversation(self) -> dict[str, Any] | None:
        state = super()._rebuild_conversation()
        if state is not None and state.get("focus") is None and state.get("messages"):
            state["focus"] = len(self.full_trace) - 1  # the first PLAN message is logged before the system message
        return state

    def _resume_conversation(self) -> bool:
        if not super()._resume_conversation():
            return False
        phases = [r for r in self.records if "plan" in r or "step_start" in r or "advance" in r]
        last = phases[-1] if phases else {"plan": {}}
        self.committed_sha = self.result.committed_sha
        open_round = next((r for r in reversed(self.result.fit_rounds) if r.get("end_turn") is None), None)
        if "plan" in last:
            self.phase, self.fit_round = "plan", open_round
            self._close_fit_round("escaped" if self.out_of_sync is not None else "accepted")  # (ended before the PLAN message)
            self.fit_round = None
            self._focus_on(len(self.full_trace) - 1)
        else:
            self.phase = "fit"
            self.fit_round = open_round
            if self.fit_round is None:
                self._new_fit_round(int(self.focus), "(resumed)")
        seen = (last.get("plan") or last.get("step_start") or last.get("advance") or {}).get("steps")
        n = len(self.full_trace)
        if seen is not None and seen != n:  # moves were played after the last message the model got
            text = (f"The run was interrupted while moves were being played: steps {seen}-{n - 1} were played after the last "
                    "message (the tool output of that batch may be missing above).")
            if self.out_of_sync is not None:
                self._after_sync(text)
                return True
            k = self._replay_all()
            if k is None:
                if self._engine_hash() != self.committed_sha:
                    self._record_commit("(resumed) engine.py reproduces every step played", implicit=True)
                self._focus_on(n - 1)
                self._after_sync(text)
            else:
                self._enter_fit(k, "your replica does not reproduce it (found when the run resumed)", 0, predicted=False)
        return True

    def _restore(self) -> bool:
        restored = super()._restore()
        previous = self.dir / "result.json"
        if previous.exists():
            data = json.loads(previous.read_text(encoding="utf-8"))
            self.restored_status = data.get("status")
            for key in ("batches", "refused_batches", "moves_sent", "mismatches", "auto_resets", "plan_nudges", "batch_log",
                        "fit_rounds", "phase_turns", "step_tokens", "committed_sha", "escapes"):
                if key in data:
                    setattr(self.result, key, data[key])
            self.phase = data.get("phase") or self.phase
        self.committed_sha = self.result.committed_sha
        if self.support_path.exists():  # the committed engine's support map, when it is that engine's
            smap = json.loads(self.support_path.read_text(encoding="utf-8"))
            self.support = smap if smap.get("engine_sha") == self.committed_sha else None
        # One cumulative token count per step after step 0 (a run interrupted mid-batch saved its trace, not result.json).
        need = len(self.full_trace) - 1
        tokens = list(self.result.step_tokens)[:need]
        tokens += [int(self.result.usage.completion_tokens)] * (need - len(tokens))
        self.result.step_tokens = tokens
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
        ignore, resync = sync_points(self.live.trace.meta)
        r.out_of_sync, r.unexplained, r.resync = self.out_of_sync, sorted(ignore), {str(k): v for k, v in sorted(resync.items())}
        super()._save_result()

    def run(self) -> AgentResult:
        if self.live.won:  # the game was won before this session (interrupted right after the winning move)
            self.setup()
            self._restore()
            self._finish_won()
            self._final_test()
        else:
            super().run()
        try:
            self.live.save(self.dir / "trace")
            self.live.write_events(self.dir)
        finally:
            self._save_result()
        return self.result

    def game_run(self) -> dict[str, Any]:
        """The TAAF GameRun record of this game (live_game.LiveGame.game_run), with the output tokens per action."""
        cumulative = [int(c) for c in self.result.step_tokens]
        tokens = [0] * len(self.full_trace)
        for k in range(1, min(len(tokens), len(cumulative) + 1)):
            tokens[k] = cumulative[k - 1] - (cumulative[k - 2] if k >= 2 else 0)
        total = int(self.result.usage.completion_tokens)
        status = self.result.status if self.result.status != "running" else (self.restored_status or "running")
        state = "won" if self.live.won else "gave_up"
        return self.live.game_run(tokens, state=state, note=f"{status}; output tokens={total}",
                                  final_tokens=max(0, total - (cumulative[-1] if cumulative else 0)))


__all__ = ["PlayAgent", "PlayResult"]
