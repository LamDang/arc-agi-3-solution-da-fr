"""The stepwise harness (v6): the harness leads one conversation from one breaking step to the next.

    uv run --no-sync python -m engine_re.run_experiment --run-dir runs/<harness run> --games lp85 \\
        --out runs/engine-re/<name>            (--mode stepwise is the default)

The loop:
  1. Opening (a new run): engine.py from the template, auto_sprites(0) put into make_level, so that
     level 0's first frame is drawn (agent.EngineAgent.play_opening).
  2. The harness replays the whole recording through engine.py and finds the first step k that fails.
     The conversation starts with "Fix the breaking test: step k": the model sees the recording up to
     step k (`recording`, steps 0..k, in python and on disk, and `step_to_fix`, step k; with
     history=False only `step_to_fix`), never a later step, and its tests replay steps 0..k.
  3. As soon as steps 0..k pass (finish, run_tests or the automatic test after an edit), the harness
     replays on. If the whole recording passes, the run is done. Otherwise it adds a user message to the
     same conversation: steps 0..k pass, so many more steps passed without error, step k' fails, with
     its report. The kernel keeps its variables and `recording` grows to step k'. Back to 3.
There is no limit per step: the run ends when the recording passes, or when a budget (turns, output
tokens, cost, time) runs out, the model stops calling tools, or a request fails for good.

Files, besides the usual ones (agent.py): visible_trace/ (the recording up to the step being fixed);
result.json has "mode": "stepwise", "step" (the step being fixed), "passing_prefix" (of the last
replay of the recording) and "advances" (one per step fixed: the turn, the step, the next failing
step). Records in the transcript carry "step". Running the same command again continues an
interrupted run from engine.py, in a new conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine_re.agent import AgentResult, Budget, EngineAgent, ModelConfig


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
        opening: bool = True,
        history: bool = True,
    ):
        self.game, self.dir, self.model, self.budget = game, Path(game_dir).resolve(), model, budget
        self.client, self.images, self.match, self.opening, self.history = client, images, match, opening, history

    def run(self) -> AgentResult:
        opening_result: dict[str, Any] = {}
        if not (self.dir / "workspace" / "engine.py").exists():
            opener = EngineAgent(self.game, self.dir, self.model, self.budget, client=self.client, match=self.match,
                                 images=self.images, opening=self.opening)
            if self.opening:
                opener.play_opening()
                opening_result = opener.result.opening
            else:
                opener.setup()
        agent = EngineAgent(self.game, self.dir, self.model, self.budget, client=self.client, match=self.match,
                            images=self.images, stepwise=True, history=self.history)
        agent.result.opening = opening_result
        return agent.run()
