"""The reverse-engineering agent: an OpenRouter tool-calling loop.

One `EngineAgent` works on one game in its own directory:

    <game_dir>/trace/            the recording (engine_re.trace)
    <game_dir>/workspace/        engine.py and anything the model writes
    <game_dir>/engine_versions/  every version of engine.py (edit_file/undo_edit), out of the model's reach
    <game_dir>/images/           the pictures sent to the model (test reports, show_frames())
    <game_dir>/transcript.jsonl  every model turn, tool call, engine change and image
    <game_dir>/tests.jsonl       every run_tests result
    <game_dir>/engine_best.py    the best engine tested so far (engine_files.BEST_RULE)
    <game_dir>/result.json       outcome, tokens, cost, final test

Tools: python (a kernel with the recording, each step's pieces, and read_file/edit_file/undo_edit/render_state/
show_frames/replay_step; it cannot write engine.py except through edit_file() and undo_edit(), which
the harness applies), run_tests and commit_engine(message), which submits engine.py: it runs the
tests, and the message says what changed and why (result.json keeps it). In the single mode the
session ends when every test passes (by commit_engine, run_tests or the automatic test) or when a
budget (turns, output tokens, cost, wall time) runs out.

The opening: before the first turn of a new session the harness plays the first round itself. In the
kernel, recording[0].pieces_after.code() makes sprite code for level 0's first frame (through the private
helpers._level_code, which also prints a summary) and one edit_file() puts it above make_level, which
then returns level_0_sprites(); then it runs the tests. The first message shows that summary, the
test report (with its picture) and engine.py, and sets the first task: the first failing step,
usually step 1. That edit and test are not counted as the model's
(engine_changes, tests_run); tests.jsonl marks the test "auto": "opening".

Context (ModelConfig.context): both schemes act at the same moments, after a turn whose request went over
compact_prompt_tokens, and leave the conversation alone in between, so the prompt's prefix stays the same
from one request to the next and the provider's prompt cache hits. "compact" shortens the conversation in
place by age (old tool outputs, old reasoning, long old arguments, older engine.py listings; a "compact"
record). "condense" keeps the full conversation and condenses it by iteration (engine_re.condense) into a
view that becomes the prefix of every request until the next firing, the messages added since following it
as they are (a "condense" record with the estimate marks each firing).

Feedback the harness adds on its own: when engine.py changed during a turn and was not tested
since, run_tests runs automatically with its defaults (a full replay, reported up to the first
failure) and its report is appended to the turn's last tool output; after every TEST_NUDGE_TURNS
turns without any test, a reminder to write and test is appended instead.

Images (``images=True``, the default): tool messages stay plain strings, so after the turn's tool
messages one extra user message carries the turn's pictures: what show_frames() made, then the latest
test report's picture (the engine's final frame next to the original's, the differing regions
boxed). When a newer such message is added, the images of the older ones are replaced by a short
placeholder. The PNGs are saved under ``<game_dir>/images/`` and the transcript logs their paths.

Stepwise mode (v6, ``stepwise=True``, see engine_re.stepwise): one conversation that fixes the
recording one breaking step at a time. The harness replays the whole recording; at the first step k
that fails it shows the model the recording up to k (``visible_trace/``: ``recording`` holds steps
0..k with ``history``, and ``step_to_fix`` is step k), its tests replay steps 0..k, and its message
asks to fix step k. Only commit_engine(message) moves on: when steps 0..k pass, the commit is
recorded (transcript "commit" record, result.json ``advances``: the message, the engine's hash and
version) and the harness replays on and adds a user message to the same conversation: how many more
steps passed and the next one that fails, with its report (the kernel keeps its variables;
``recording`` grows to the new step). Tests that pass without a commit (run_tests, the automatic
test) only say that a commit is now possible, so the model can keep refining first. The conversation
ends with status "passed" when a commit makes the whole recording pass, or when a budget runs out;
there is no per-step limit.

Sessions survive interruptions: result.json is rewritten every turn with status "running", and
running a game again whose session did not end continues from its engine.py (and its versions),
carrying over the turns, tokens, cost, time and test history: a stepwise run in its own conversation,
rebuilt from transcript.jsonl, its kernel restarted and the conversation's python cells re-run in
order with edits disabled (KernelClient.replay), the resume note saying which cells raised and what
the kernel keeps; a single-mode run with a fresh conversation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

from engine_re import diff_report, hashline
from inference.agent.tool_agent import _env_float, _env_int, _post_with_retries
from engine_re.engine_files import best_key
from engine_re.game_api import fixed_block_lines
from engine_re.helpers import FUNCTIONS as BUILTIN_FUNCTIONS
from engine_re.kernel import KernelClient
from engine_re.prompts import (
    ENGINE_HEADER, advance_message, elide_engine_listing, episode_message, first_user_message, kernel_names_text,
    resume_user_message, system_prompt, tools,
)
from engine_re.skeleton import render_skeleton
from engine_re.tester import MAX_FAILURES, replay_test
from engine_re.trace import Trace

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"  # public: no key needed
NO_BUDGET_WARNING = (
    "[{model}] warning: this model advertises no thinking-token budget on OpenRouter (its reasoning settings lack "
    "supports_max_tokens), so the thinking budget of {budget} tokens may be ignored or mapped to an effort level."
)
TOOL_OUTPUT_CHARS = 8000
# Turns without a run_tests call after which the harness reminds the model to
# write what it knows into engine.py and test it (and again every as many turns).
TEST_NUDGE_TURNS = 30
NUDGE = (
    "\n\n[harness] {n} turns since your last run_tests (or none yet). Put what you have established into engine.py now "
    "with edit_file(), even if partial, and run run_tests: its report shows which step and pixels to fix next."
)
# When engine.py changed during a turn and the model did not test it, the
# harness runs run_tests() with its defaults (full replay, reported up to the
# first failure) and appends the report to the turn's last output.
AUTO_TEST = "\n\n[harness] engine.py changed, so it was tested automatically (run_tests with its defaults):\n{report}"
# The automatic test found the same failure as the last test (tester.TestReport.signature): one line instead of the report.
AUTO_TEST_SAME = "\n\n[harness] engine.py changed, tested automatically: the same result as the last test ({brief})."
# A python built-in function (helpers.FUNCTIONS) the model called as if it were a tool: run as python.
BUILTIN_AS_TOOL = "[harness] {name} is a python function, not a tool; this call ran as python: {call}\n"
REPORT_CHARS = 9000  # a test report in a tool output
AUTO_TEST_CHARS = 3500
# Stepwise: appended to a test (run_tests or the automatic one) that shows steps 0..k pass.
COMMIT_HINT = (
    "\n\nSteps 0-{k} pass. You can now call commit_engine(message) to submit the fix, or keep refining first; the next "
    "steps are shown only after a commit."
)
COMMIT_DROPPED = (
    "\n\n[harness] engine.py changed after commit_engine in this turn, so the commit was not kept: call commit_engine "
    "again once the tests pass."
)
IMAGE_NOTE = "[harness] The images of this turn, in order:"
TEST_IMAGE_NOTE = (
    "From the latest run_tests report: on the left your engine's final frame, on the right the original game's "
    "(upscaled 8x); the differing regions are boxed and numbered as in the report's text."
)
IMAGE_PLACEHOLDER = "[image omitted; the latest images come later]"
MAX_IMAGES_PER_MESSAGE = 6
# OpenRouter answers that fail at the provider (finish_reason "error", or a read that stalls) are asked
# again this many times. Rate limits, gateway errors and connection failures are retried apart, by the
# main harness's _post_with_retries: without limit by default, as its OpenRouter config does
# (ARC3_HTTP_RETRIES=-1 in params.yaml; ARC3_HTTP_RETRY_BASE_SECONDS / _MAX_SECONDS set the waits).
PROVIDER_ERROR_RETRIES = 20
MAX_TEST_IMAGES = 3
PYTHON_PAUSED = (
    "[harness] Python is paused: {n} python calls since engine.py last changed. Until engine.py changes, only python "
    "calls that change it with edit_file() or undo_edit() run; then python resumes. run_tests shows which step and pixels "
    "to fix next."
)
# A resumed run continues its conversation, rebuilt from transcript.jsonl: every message the model is sent is logged
# there ("message" records; assistant turns and tool outputs as their own records), with the text the harness adds to
# a message ("append"), and the points where old images are hidden ("hide_images") and old turns shortened ("compact").
# The kernel restarts empty and re-runs the conversation's python cells (KernelClient.replay); the note says so.
RESUME_NOTE = (
    "[harness] The run was interrupted here and has now resumed, in this same conversation. engine.py, its versions "
    "(undo_edit) and everything above are kept. {replay}\n{names}"
)
REPLAY_DONE = (
    "The python kernel restarted and re-ran your {n} python cell{s} in order with file edits disabled, so your variables "
    "and functions are back{failed}."
)
REPLAY_FAILED = "; cells that raised when re-run (as before, or because engine.py changed later): turn{s} {turns}"
REPLAY_SKIPPED = " The last {n} cell{s} were not re-run (the replay's time ran out)."
REPLAY_NONE = "The python kernel restarted (there were no python cells to re-run)."


def resume_note(replay: dict[str, Any], names: str) -> str:
    """RESUME_NOTE for a replay result (KernelClient.replay) and the kernel's names line (kernel_names_text)."""
    n = int(replay.get("replayed") or 0)
    if not n and not replay.get("skipped"):
        text = REPLAY_NONE
    else:
        turns = sorted({int(f["turn"]) for f in replay.get("failed") or [] if f.get("turn") is not None})
        failed = REPLAY_FAILED.format(s="s" if len(turns) > 1 else "", turns=", ".join(map(str, turns))) if turns else ""
        text = REPLAY_DONE.format(n=n, s="" if n == 1 else "s", failed=failed)
        if replay.get("skipped"):
            text += REPLAY_SKIPPED.format(n=replay["skipped"], s="" if replay["skipped"] == 1 else "s")
    return RESUME_NOTE.format(replay=text, names=names)
CONTINUE = "Continue by calling a tool (python, run_tests or commit_engine)."
READ_CHARS_IN_MESSAGES = 14000  # engine.py shown in the first message (FIXED block folded)
# The opening, run in the kernel: level 0's first frame as code, recording[0].pieces_after.code(), through the private
# helpers._level_code (its summary printed, not its code), then one edit_file() that puts the code above make_level
# and makes make_level return level_0_sprites(). Filled in by EngineAgent._opening_code with the anchors of the
# starting engine.py.
OPENING_SPLIT = "----- harness: edit -----"
OPENING_CODE = '''\
import contextlib as _harness_contextlib, io as _harness_io
from engine_re.helpers import _level_code as _harness_level_code
_harness_out = _harness_io.StringIO()
with _harness_contextlib.redirect_stdout(_harness_out):
    _harness_code = _harness_level_code(0)
print(_harness_out.getvalue().split("\\n\\n")[0])
print({split!r})
_harness_view = f", view=View(scale={{_harness_code.view}})" if _harness_code.view else ""
edit_file(edits=[
    {{"op": "prepend", "pos": {head!r}, "lines": _harness_code.rstrip("\\n").splitlines() + ["", ""]}},
    {{"op": "replace", "pos": {start!r}, "end": {end!r}, "lines": [
        "    # For now every level starts as level 0: add level n when the tests reach it, from",
        "    # recording[e].pieces_after.code(), e being the step that enters level n.",
        f"    return State(grid={{_harness_code.grid}}, sprites=level_0_sprites(){{_harness_view}})"]}},
])
del _harness_contextlib, _harness_io, _harness_out, _harness_code, _harness_level_code, _harness_view
'''


@dataclass
class Budget:
    max_turns: int = 250
    max_output_tokens: int = 1_000_000
    max_cost_usd: float = 5.0
    max_minutes: float = 150.0
    # If set: after this many python calls without any change to engine.py,
    # the python tool pauses until engine.py changes (an analysis quota).
    python_quota: int | None = None


@dataclass
class ModelConfig:
    model: str = "qwen/qwen3.8-flash"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 32768
    reasoning: bool = True
    # OpenRouter's reasoning.effort ("low", "medium", "high", ...); None leaves the provider's default.
    reasoning_effort: str | None = None
    # OpenRouter's reasoning.max_tokens: at most this many thinking tokens per answer; None: no limit. Not together
    # with reasoning_effort (OpenRouter takes one or the other).
    thinking_budget: int | None = None
    # None: not sent (the provider's default).
    top_k: int | None = None
    # Above this prompt size, old tool outputs are elided from the history.
    compact_prompt_tokens: int = 140_000
    keep_recent_tool_outputs: int = 8
    # The model's reasoning is sent back with its turns (as the main harness
    # does on OpenRouter); compaction trims all but the most recent ones.
    keep_recent_reasoning: int = 10
    old_reasoning_chars: int = 1200
    # OpenRouter providers to use, in order, with no fallback to others (e.g. ["z-ai"]);
    # None lets OpenRouter route each request.
    providers: list[str] | None = None
    # How the conversation is kept within bounds, both applied after a request went over compact_prompt_tokens:
    # "compact" (the settings above, applied in place) or "condense" (engine_re.condense: the full conversation is
    # kept and condensed by iteration into the prefix of every request until the next firing; the turns since
    # follow it as they are). The condenser keeps the reasoning and failed commands of the current iteration's last
    # condense_keep_turns turns; its safety cap estimates tokens with condense_chars_per_token.
    context: str = "compact"
    condense_keep_turns: int = 10
    condense_chars_per_token: float = 3.0

    def __post_init__(self) -> None:
        if self.thinking_budget is not None:
            if self.reasoning_effort:
                raise ValueError("a thinking budget (reasoning.max_tokens) and a reasoning effort cannot be combined: give one")
            if int(self.thinking_budget) <= 0:
                raise ValueError(f"the thinking budget must be a positive number of tokens, got {self.thinking_budget}")


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    max_prompt_tokens: int = 0

    def add(self, usage: dict[str, Any]) -> None:
        self.requests += 1
        prompt = int(usage.get("prompt_tokens") or 0)
        self.prompt_tokens += prompt
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        self.cached_tokens += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self.cost_usd += float(usage.get("cost") or 0.0)


def _truncate(text: str, limit: int = TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return f"{text[:head]}\n...[{len(text) - head - tail} characters truncated]...\n{text[-tail:]}"


def builtin_call_code(name: str, args: Any) -> str:
    """`name(**args)` as python code, for a built-in function the model called as a tool: a string
    argument that parses as JSON (an `edits` list given as text, a number as "240") is parsed first."""
    parts = []
    for key, value in (args.items() if isinstance(args, dict) else []):
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            if not isinstance(parsed, str):
                value = parsed
        parts.append(f"{key}={value!r}")
    return f"{name}({', '.join(parts)})"


def cell_code(call: dict[str, Any]) -> str | None:
    """The python code a logged tool call ran: the python tool's code, or a built-in called as a tool
    (builtin_call_code); None for the other tools or unreadable arguments."""
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    if name == "python":
        return args.get("code") if isinstance(args, dict) and isinstance(args.get("code"), str) else None
    if name in BUILTIN_FUNCTIONS:
        return builtin_call_code(name, args)
    return None


def _elide_arguments(arguments: str) -> str:
    """Shorten a past tool call's long string arguments, keeping the tool's own
    keys: a placeholder key would teach the model a call shape the tools reject."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"code": f"# [{len(arguments)} characters elided to save context]"})
    if not isinstance(args, dict):
        return arguments
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 600:
            args[key] = f"{value[:300]}\n# [... {len(value) - 300} more characters elided to save context]"
    return json.dumps(args)


class OpenRouterClient:
    def __init__(self, config: ModelConfig, api_key: str | None = None):
        self.config = config
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")
        if not self.api_key:
            raise RuntimeError("set OPENROUTER_API_KEY")
        self.session = requests.Session()
        self.provider_errors: list[str] = []  # answers that failed at the provider (or stalled), asked again

    def body(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """The request body sent to OpenRouter."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "reasoning": {"enabled": self.config.reasoning},
            "usage": {"include": True},
        }
        if self.config.reasoning and self.config.thinking_budget is not None:
            payload["reasoning"] = {"max_tokens": int(self.config.thinking_budget)}
        elif self.config.reasoning and self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k
        if self.config.providers:
            payload["provider"] = {"order": list(self.config.providers), "allow_fallbacks": False}
        return payload

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = self.body(messages, tools)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        error = ""
        for attempt in range(PROVIDER_ERROR_RETRIES):
            try:
                # HTTP 429 and gateway errors, and connections that fail, wait and retry in place, as in the
                # main harness (with no limit unless ARC3_HTTP_RETRIES says otherwise).
                resp = _post_with_retries(
                    lambda: self.session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=900),
                    retries=_env_int("ARC3_HTTP_RETRIES", -1),
                    base_seconds=_env_float("ARC3_HTTP_RETRY_BASE_SECONDS", 5.0),
                    max_seconds=_env_float("ARC3_HTTP_RETRY_MAX_SECONDS", 5.0),
                    sleep=lambda seconds: time.sleep(seconds),
                )
            except requests.RequestException as exc:  # a read that stalled: ask again
                error = f"{type(exc).__name__}: {exc}"
                self.provider_errors.append(error)
            else:
                if resp.status_code != 200:  # not retryable, or ARC3_HTTP_RETRIES ran out
                    raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1000]}")
                data = resp.json()
                choices = data.get("choices") or []
                if choices and choices[0].get("finish_reason") != "error":
                    return data
                # The provider failed mid-answer (finish_reason "error"): ask again rather than
                # hand the model an empty turn.
                detail = (choices[0].get("error") if choices else None) or data.get("error") or data
                error = f"{'provider error' if choices else 'no choices'}: {json.dumps(detail)[:500]}"
                self.provider_errors.append(error)
            time.sleep(min(60.0, 2.0 * 2 ** attempt) + random.random())
        raise RuntimeError(f"OpenRouter request failed {PROVIDER_ERROR_RETRIES} times at the provider: {error}")



_BUDGET_CHECKED: dict[tuple[str, int], str | None] = {}  # (model, budget) -> the warning, once per process


def thinking_budget_warning(model: str, budget: int, fetch: Any = None) -> str | None:
    """A one-line warning when OpenRouter's model listing gives `model` reasoning settings without
    supports_max_tokens: true (the budget may then be ignored or mapped to an effort). None when it supports one,
    when the model is not listed or has no reasoning settings, and when the listing cannot be read. `fetch` returns
    the listing's JSON (by default a GET of OPENROUTER_MODELS_URL)."""

    def get() -> Any:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=20)
        resp.raise_for_status()
        return resp.json()

    try:
        listing = (fetch or get)()
        entry = next((m for m in listing.get("data") or [] if m.get("id") == model), None)
    except Exception:  # noqa: BLE001  (only a warning: a failed lookup says nothing)
        return None
    reasoning = (entry or {}).get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("supports_max_tokens") is not True:
        return NO_BUDGET_WARNING.format(model=model, budget=budget)
    return None


@dataclass
class AgentResult:
    game: str
    model: str
    status: str = "running"  # passed | budget_* | stalled | error
    turns: int = 0
    minutes: float = 0.0
    usage: Usage = field(default_factory=Usage)
    tool_calls: dict[str, int] = field(default_factory=dict)
    tests_run: int = 0
    first_pass_turn: int | None = None
    best: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    commit_message: str | None = None  # the message of the last commit_engine call
    commit_calls: int = 0
    error: str | None = None
    trace_steps: int = 0
    resumes: int = 0
    nudges: int = 0
    auto_tests: int = 0
    python_paused: int = 0
    engine_changes: int = 0
    match: str = "final"
    interface: str = "simple"
    images: bool = True
    image_messages: int = 0
    provider_errors: int = 0  # answers that failed at the provider and were asked again
    opening: dict = field(default_factory=dict)  # the harness's first round: exact, first_fail, or error
    mode: str = "single"  # "stepwise": one conversation led from one breaking step to the next
    context: str = "compact"  # "compact" | "condense": ModelConfig.context, how the conversation was bounded
    thinking_budget: int | None = None  # ModelConfig.thinking_budget (reasoning.max_tokens); None: no limit
    step: int | None = None  # stepwise: the step being fixed
    passing_prefix: int | None = None  # stepwise: steps passing in order in the last replay of the recording
    # stepwise: one per accepted commit: {"turn", "fixed", "next", "message", "engine_sha", "version"}
    advances: list = field(default_factory=list)


class EngineAgent:
    def __init__(
        self,
        game: str,
        game_dir: Path,
        model: ModelConfig,
        budget: Budget,
        client: OpenRouterClient | None = None,
        match: str = "final",
        interface: str = "simple",
        images: bool = True,
        opening: bool = True,
        *,
        stepwise: bool = False,
        history: bool = True,
    ):
        if interface != "simple":
            raise ValueError("the agent writes make_level/step engines only (interface='simple')")
        self.game = game
        self.match = match
        self.interface = interface
        self.images = images
        self.stepwise = stepwise
        self.history = history
        self.mode = "step" if stepwise else "single"
        self.focus: int | None = None  # stepwise: the step being fixed
        self.opening = opening and not stepwise
        self._in_opening = False
        self.dir = Path(game_dir).resolve()
        self.full_trace = Trace.load(self.dir / "trace")
        # Stepwise: the model's kernel and tests see visible_trace/, the recording up to the step being fixed.
        self.trace_dir = self.dir / ("visible_trace" if stepwise else "trace")
        self.workspace = self.dir / "workspace"
        self.engine_path = self.workspace / "engine.py"
        self.trace = self.full_trace
        self.model = model
        self.budget = budget
        self.client = client or OpenRouterClient(model)
        self.kernel = KernelClient(self.workspace, self.trace_dir, images=images, log=self._log_engine_change, history=history)
        self.result = AgentResult(
            game=game, model=model.model, trace_steps=len(self.trace), match=match, interface=interface, images=images,
            mode="stepwise" if stepwise else "single", context=model.context, thinking_budget=model.thinking_budget,
        )
        if model.context not in ("compact", "condense"):
            raise ValueError(f"ModelConfig.context must be 'compact' or 'condense', got {model.context!r}")
        self.condense = model.context == "condense"
        # Every message the model is sent, in full. With context "condense" nothing is ever shortened in place: each
        # request gets self._condensed (the condensed view of self.messages[:self._condensed_from], computed the last
        # time the condenser fired; empty before) followed by the messages added since (see _context).
        self.messages: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []  # every transcript record of this run, in order (the condenser's input)
        self._condensed: list[dict[str, Any]] = []
        self._condensed_from = 0
        self.best_key: tuple[int, int] = (-1, -1)
        self.passed = False
        self.prior_minutes = 0.0
        self.prior_notes = ""
        self.started = time.time()
        self.turns_since_test = 0
        self.tested_hash: str | None = None
        self.last_signature: str | None = None  # of the last full test's failure (tester.TestReport.signature)
        self.python_since_change = 0
        self.engine_hash_seen: str | None = None
        # The turn's pictures, sent after its tool messages: (caption, saved PNG path, PNG bytes).
        self.pending_shown: list[tuple[str, Path, bytes]] = []
        self.pending_test: list[tuple[str, Path, bytes]] = []
        self.image_files: dict[str, str] = {}  # an image's data URL -> its saved file, for the transcript
        self.commit: dict[str, Any] | None = None  # stepwise: an accepted commit_engine, applied after the turn

    # --- tools -----------------------------------------------------------------

    def _tool_python(self, code: str) -> str:
        current = self._engine_hash()
        if current != self.engine_hash_seen:
            self.engine_hash_seen = current
            self.python_since_change = 0
        quota = self.budget.python_quota
        if quota is not None and self.python_since_change >= quota and "edit_file(" not in code and "undo_edit(" not in code:
            self.result.python_paused += 1
            return PYTHON_PAUSED.format(n=self.python_since_change)
        self.python_since_change += 1
        output = self.kernel.execute(code)
        self._keep_shown(self.kernel.last_images)
        return _truncate(output)

    def _engine_hash(self) -> str:
        return hashlib.sha256(self.engine_path.read_bytes()).hexdigest()

    def _tool_run_tests(self, level: int | None = None, failures: int = 1) -> str:
        """No level: replay from step 0; level=L: only level L. The report stops after `failures`
        failing steps (clamped to 1..MAX_FAILURES); the counts kept in tests.jsonl are those of the
        whole replay either way."""
        report = self._run_tests(level, failures, auto=False)
        return report if isinstance(report, str) else _truncate(report.text, REPORT_CHARS) + self._commit_hint(report)

    def _commit_hint(self, report: Any) -> str:
        """Stepwise: the sentence that says steps 0..k pass and a commit is now possible."""
        if self.stepwise and not isinstance(report, str) and report.passed and report.level is None:
            return COMMIT_HINT.format(k=self.focus)
        return ""

    def _run_tests(self, level: Any, failures: Any, auto: bool) -> Any:
        """Run and record a test; returns the TestReport, or an error text."""
        try:
            level = None if level is None else int(level)
        except (TypeError, ValueError):
            return f"Error: level must be a level number, got {level!r}."
        try:
            failures = int(failures)
        except (TypeError, ValueError):
            failures = 1
        failures = max(1, min(MAX_FAILURES, failures))
        self.turns_since_test = 0
        tested_hash = self._engine_hash()
        try:
            report = replay_test(
                self.engine_path, self.trace, level=level, failures=failures,
                scratch_root=self.dir, match=self.match, images=self.images,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        full = level is None
        if full:
            self.tested_hash = tested_hash
            self.last_signature = report.signature
        if auto not in ("opening", "episode", "advance"):  # the harness's own tests are not the model's
            self.result.tests_run += 1
        # "from_level" keeps its earlier meaning for older readers of tests.jsonl, the level the engine
        # started at (null: a fresh engine). A full replay has "level" null and "total" = the trace length.
        # "engine_sha" ties the result to a version of engine.py (undo shows it).
        entry = {
            "turn": self.result.turns, "time": time.time(), "from_level": level or None, "auto": auto,
            "engine_sha": tested_hash, **report.summary(),
        }
        if self.stepwise:
            entry["focus"] = self.focus
        with (self.dir / "tests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._keep_test_images(report, auto)
        if full:
            key = best_key(entry)
            if key > self.best_key:
                self.best_key = key
                shutil.copy(self.engine_path, self.dir / "engine_best.py")
                self.result.best = {"turn": self.result.turns, **report.summary()}
            self.passed = report.passed  # the latest full test (stepwise: steps 0..k)
            if report.passed:
                if self.result.first_pass_turn is None:
                    self.result.first_pass_turn = self.result.turns
        return report

    def _tool_commit_engine(self, message: Any = None) -> str:
        """Submit engine.py: run the tests. Single mode: the session ends when every test passes.
        Stepwise: when steps 0..k pass, the commit is kept and the harness moves on after the turn."""
        if not isinstance(message, str) or not message.strip():
            return ("Error: commit_engine needs a message: what you changed and the key analysis behind each rule (what in "
                    "the recording shows it: which steps or frames, what changed), and which guesses remain. Nothing was "
                    "committed.")
        self.result.commit_calls += 1
        self.result.commit_message = message
        report = self._run_tests(None, 1, auto=False)
        if isinstance(report, str):
            return report
        if not report.passed:
            what = f"steps 0-{self.focus} do not all pass yet, so nothing moves on" if self.stepwise else "the tests still fail, so the session goes on"
            return f"Not committed: {what}. The report:\n\n" + _truncate(report.text, REPORT_CHARS)
        if not self.stepwise:
            return "Every test passes. Session finished."
        sha = self._engine_hash()
        self.commit = {"message": message, "engine_sha": sha, "version": self._version_of(sha)}
        return f"Committed: steps 0-{self.focus} pass. The harness now replays the rest of the recording."

    def _version_of(self, sha: str) -> int | None:
        """The number of the latest saved version of engine.py with this sha256, if any."""
        return next((v["version"] for v in reversed(self.kernel.editor.versions()) if v.get("sha") == sha), None)

    # --- images ----------------------------------------------------------------

    def _save_png(self, png: bytes, stem: str) -> Path:
        folder = self.dir / "images"
        folder.mkdir(exist_ok=True)
        path, n = folder / f"{stem}.png", 2
        while path.exists():
            path, n = folder / f"{stem}_{n}.png", n + 1
        path.write_bytes(png)
        return path

    def _keep_shown(self, shown: list[tuple[bytes, str]]) -> None:
        """Save what show_frames() made in this python call and queue it for the turn's image message."""
        paths = []
        for png, caption in shown:
            path = self._save_png(png, f"turn{self.result.turns:03d}_show{len(self.pending_shown) + 1}")
            self.pending_shown.append((caption, path, png))
            paths.append(str(path.relative_to(self.dir)))
        if paths:
            self._log({"turn": self.result.turns, "show_images": paths})

    def _keep_test_images(self, report: Any, auto: bool) -> None:
        """Save a report's images and make them the turn's test images (a later test replaces them)."""
        if not self.images:
            return
        self.pending_test = []
        for image in report.images[:MAX_TEST_IMAGES]:
            suffix = f"_{auto}" if isinstance(auto, str) else "_auto" if auto else ""
            path = self._save_png(image.png, f"turn{self.result.turns:03d}_step{image.step}{suffix}")
            self.pending_test.append((image.caption, path, image.png))

    def _attach_images(self) -> None:
        """Append one user message with the turn's images after its tool messages, and replace the
        images of earlier such messages by a placeholder, so one set stays in the context."""
        pictures = (self.pending_shown + self.pending_test)[:MAX_IMAGES_PER_MESSAGE] if self.images else []
        self.pending_shown, tests, self.pending_test = [], self.pending_test, []
        if not pictures:
            return
        if not self.condense:  # condense: self.messages keeps every image; _context and the condenser hide the older ones
            self._hide_images(self.messages)
            self._log({"turn": self.result.turns, "hide_images": True})
        content: list[dict[str, Any]] = [{"type": "text", "text": IMAGE_NOTE}]
        for caption, path, png in pictures:
            text = (TEST_IMAGE_NOTE + " " + caption) if any(path == t[1] for t in tests) else caption
            content.append({"type": "text", "text": text})
            content.append(self._image_part(png, path))
        self._say("user", content)
        self.result.image_messages += 1
        self._log(
            {
                "turn": self.result.turns,
                "images": [str(path.relative_to(self.dir)) for _, path, _ in pictures],
                "captions": [caption for caption, _, _ in pictures],
            }
        )

    # --- engine.py -------------------------------------------------------------

    def _log_engine_change(self, record: dict[str, Any]) -> None:
        """Called by the harness side of edit_file()/undo_edit() for every change to engine.py. The opening's
        change is logged as the harness's and not counted."""
        if self._in_opening:
            record = {**record, "by": "harness"}
        else:
            self.result.engine_changes += 1
        self._log({"turn": self.result.turns, **record})

    def _read_engine(self, fold: bool, max_chars: int) -> str:
        text = self.engine_path.read_text(encoding="utf-8")
        return hashline.render_read(text, max_chars=max_chars, fold=fixed_block_lines(text) if fold else None)

    def _dispatch(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Error: tool arguments are not valid JSON ({exc}). Send a JSON object."
        if name in BUILTIN_FUNCTIONS:  # a python function called as a tool: run it as python
            call = builtin_call_code(name, args)
            return BUILTIN_AS_TOOL.format(name=name, call=call) + self._tool_python(call)
        if name not in ("python", "run_tests", "commit_engine"):
            return f"Error: unknown tool {name!r}. The tools are python, run_tests and commit_engine."
        handler = getattr(self, f"_tool_{name}")
        try:
            return handler(**args)
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001  (a tool failure is reported to the model, not raised)
            return f"Error while running {name}: {type(exc).__name__}: {exc}"

    # --- loop --------------------------------------------------------------------

    def _elapsed_minutes(self) -> float:
        return self.prior_minutes + (time.time() - self.started) / 60

    def _log(self, record: dict[str, Any]) -> None:
        if self.stepwise and self.focus is not None:
            record.setdefault("step", self.focus)
        record["elapsed_min"] = round(self._elapsed_minutes(), 3)
        self.records.append(record)
        with (self.dir / "transcript.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _context(self) -> list[dict[str, Any]]:
        """The messages of the next request: the conversation as it is, or, with context "condense", the condensed view
        made the last time the condenser fired (nothing before the first firing) followed by the messages added since, as
        they are but for their images: only the latest message with images keeps them (as _hide_images does for
        "compact"). The view itself never changes between two firings."""
        if not self.condense:
            return self.messages
        from engine_re.condense import hide_but_latest  # (condense imports this module's constants)

        return self._condensed + hide_but_latest(self.messages[self._condensed_from :])

    def _condensed_view(self, messages: list[dict[str, Any]], records: list[dict[str, Any]]) -> Any:
        """engine_re.condense over a full conversation and the transcript records logged up to that point; a copy of
        its messages, so nothing done to the conversation later can change the view."""
        from engine_re.condense import condense

        result = condense(messages, records, self.dir / "engine_versions", keep_turns=self.model.condense_keep_turns,
                          chars_per_token=self.model.condense_chars_per_token)
        result.messages = copy.deepcopy(result.messages)
        return result

    def _condense_now(self) -> None:
        """Context "condense": the condenser fires (a request went over compact_prompt_tokens, where "compact" would
        compact). The condensed view of the whole conversation so far becomes the prefix of every request until it fires
        again; the "condense" record logs it (estimated tokens, characters, live images, messages and, when the safety cap
        fired, what it cut) and marks the point a resumed run recomputes it at."""
        from engine_re.condense import measure

        result = self._condensed_view(self.messages, self.records)
        self._condensed, self._condensed_from = result.messages, len(self.messages)
        chars, images = measure(result.messages)
        record = {"tokens": result.estimate, "chars": chars, "images": images, "messages": len(result.messages)}
        if result.cap:
            record["cap"] = result.cap
        self._log({"turn": self.result.turns, "condense": record})

    def _save_result(self) -> None:
        self.result.minutes = round(self._elapsed_minutes(), 2)
        (self.dir / "result.json").write_text(json.dumps(asdict(self.result), indent=2) + "\n", encoding="utf-8")

    def _restore(self) -> bool:
        """Load the turns, tokens, time and tests of an interrupted session."""
        transcript = self.dir / "transcript.jsonl"
        if not transcript.exists():
            return False
        thoughts = []
        commits: list[dict[str, Any]] = []
        advances: list[dict[str, Any]] = []
        self.records = []
        for line in transcript.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            self.records.append(record)
            if "finish_reason" in record:
                self.result.turns = max(self.result.turns, record["turn"])
                self.result.usage.add(record.get("usage") or {})
                text = "\n".join(t for t in (record.get("reasoning"), record.get("content")) if t)
                if text.strip():
                    thoughts.append((record["turn"], text.strip()))
            elif "tool" in record:
                self.result.tool_calls[record["tool"]] = self.result.tool_calls.get(record["tool"], 0) + 1
                if record["tool"] in ("commit_engine", "finish"):  # finish: the tool of earlier runs
                    self.result.commit_calls += 1
            elif "engine_change" in record and record.get("by") != "harness":
                self.result.engine_changes += 1
            elif "commit" in record:
                commits.append({"turn": record["turn"], **record["commit"]})
            elif "advance" in record:  # runs from before commit_engine log only their advances
                advances.append({"turn": record["turn"], "fixed": record["advance"]["fixed"], "next": record["advance"]["next"]})
            self.prior_minutes = max(self.prior_minutes, float(record.get("elapsed_min") or 0.0))
        self.result.advances = commits or advances
        if commits:
            self.result.commit_message = commits[-1].get("message")
            if commits[-1].get("next") is None:  # that commit made the whole recording pass (run() replays to confirm)
                self.result.first_pass_turn = commits[-1]["turn"]
        tests = self.dir / "tests.jsonl"
        if tests.exists():
            for line in tests.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("auto") not in ("opening", "episode", "advance"):
                    self.result.tests_run += 1
                if entry.get("level") is None and entry.get("from_level") in (None, 0):
                    self.last_signature = entry.get("signature")
                    if best_key(entry) > self.best_key:
                        self.best_key = best_key(entry)
                        self.result.best = {k: v for k, v in entry.items() if k not in ("time", "from_level")}
                    if entry.get("passed") and self.result.first_pass_turn is None:
                        self.result.first_pass_turn = entry["turn"]
        # The interrupted session's notes file and last thoughts, so the analysis is not all lost.
        notes_file = self.workspace / "notes.md"
        parts = []
        if notes_file.exists():
            parts.append("Your notes.md:\n" + _truncate(notes_file.read_text(encoding="utf-8", errors="replace"), 6000))
        parts.append("\n\n".join(f"[turn {turn}] ...{text[-1500:]}" for turn, text in thoughts[-5:]))
        self.prior_notes = "\n\n".join(p for p in parts if p)
        previous = self.dir / "result.json"
        if previous.exists():
            self.result.resumes = int(json.loads(previous.read_text(encoding="utf-8")).get("resumes", 0)) + 1
        return self.result.turns > 0

    def _image_part(self, png: bytes, path: Path) -> dict[str, Any]:
        """An image for a message; the transcript logs it by its saved file."""
        url = diff_report.data_url(png)
        self.image_files[url] = str(path.relative_to(self.dir))
        return {"type": "image_url", "image_url": {"url": url}}

    def _log_message(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        if isinstance(content, list):
            content = [
                {"type": "image_file", "path": self.image_files.get(part["image_url"]["url"], "")} if part.get("type") == "image_url" else part
                for part in content
            ]
        self._log({"turn": self.result.turns, "message": {**message, "content": content}})

    def _say(self, role: str, content: Any) -> None:
        """Send the model a system or user message (and log it as sent)."""
        message = {"role": role, "content": content}
        self.messages.append(message)
        self._log_message(message)

    def _add_to_last(self, text: str) -> None:
        """Add the harness's text to the last message (a tool output), and log it."""
        self.messages[-1]["content"] += text
        self._log({"turn": self.result.turns, "append": text})

    def _hide_images(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "text", "text": IMAGE_PLACEHOLDER} if part.get("type") == "image_url" else part
                    for part in message["content"]
                ]

    def _rebuild_conversation(self) -> dict[str, Any] | None:
        """The conversation of the last segment of transcript.jsonl (from its last system message) exactly as the model
        was sent it, and the step it was on; None when there is none. Older transcripts: _rebuild_legacy."""
        transcript = self.dir / "transcript.jsonl"
        if not transcript.exists():
            return None
        records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        self._condensed, self._condensed_from = [], 0
        starts = [i for i, r in enumerate(records) if (r.get("message") or {}).get("role") == "system"]
        if not starts:
            return self._rebuild_legacy(records)
        self.messages = messages = []
        focus = None
        calls: Any = iter(())
        cells: list[dict[str, Any]] = []  # the python cells that ran, in order: {"turn", "code"}
        last_turn = None  # of the last assistant record (its cells go when its turn is left out)
        fired: tuple[int, int] | None = None  # the last "condense" record: (its index, the messages before it)
        for index in range(starts[-1], len(records)):
            r = records[index]
            if "message" in r:
                message = dict(r["message"])
                if isinstance(message.get("content"), list):
                    message["content"] = [
                        self._image_part((self.dir / part["path"]).read_bytes(), self.dir / part["path"])
                        if part.get("type") == "image_file" else part
                        for part in message["content"]
                    ]
                messages.append(message)
            elif "finish_reason" in r:
                assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
                if r.get("reasoning"):
                    assistant["reasoning"] = r["reasoning"]
                if r.get("tool_calls"):
                    assistant["tool_calls"] = r["tool_calls"]
                calls = iter(r.get("tool_calls") or [])
                last_turn = r.get("turn")
                messages.append(assistant)
            elif "tool" in r:
                call = next(calls, None)
                messages.append({"role": "tool", "tool_call_id": r.get("id") or (call["id"] if call else ""), "content": r["output"]})
                code = cell_code(call) if call and r["tool"] == "python" else None
                if code is not None and not str(r["output"]).startswith(PYTHON_PAUSED[:40]):  # a paused call never ran
                    cells.append({"turn": r.get("turn"), "code": code})
            elif "append" in r:
                messages[-1]["content"] += r["append"]
            elif "hide_images" in r:
                self._hide_images(messages)
            elif "compact" in r:
                self._compact()
            elif "condense" in r:
                fired = (index, len(messages))
            elif "step_start" in r:
                focus = r["step_start"]["step"]
            elif "advance" in r:
                focus = r["advance"]["next"]
            elif "resumed" in r:
                focus = r["resumed"]["step"]
        if self._drop_unanswered(messages):
            cells = [c for c in cells if c["turn"] != last_turn]
        if self.condense and fired is not None:
            # The condenser is a function of the full conversation and the records up to its firing, so its last firing
            # alone gives the view a resumed run sends, as the uninterrupted run did. (Transcripts of the earlier
            # per-turn condenser have a "condense" record before every request: the last one is taken as a firing.)
            index, covered = fired
            covered = min(covered, len(messages))
            self._condensed = self._condensed_view(messages[:covered], records[:index]).messages
            self._condensed_from = covered
        return {"messages": messages, "focus": focus, "cells": cells}

    @staticmethod
    def _drop_unanswered(messages: list[dict[str, Any]]) -> bool:
        """A turn cut off before all its tools answered is left out (every call needs an answer); True when one was."""
        assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if assistants:
            last = assistants[-1]
            if sum(m["role"] == "tool" for m in messages[last + 1 :]) < len(messages[last].get("tool_calls") or []):
                del messages[last:]
                return True
        return False

    def _resume_conversation(self) -> bool:
        """Continue an interrupted stepwise run in its own conversation, rebuilt from transcript.jsonl. A transcript
        in the older format is written back in full as "message" records first, so it is exact from then on."""
        state = self._rebuild_conversation()
        if state is None or state["focus"] is None or not state["messages"]:
            return False
        self.messages = state["messages"]
        if state.get("legacy"):
            self._log({"turn": self.result.turns, "rebased": "the conversation rebuilt from the older records, logged in full"})
            for message in self.messages:
                self._log_message(message)
            if state.get("condense"):
                self._condense_now()
        self._focus_on(int(state["focus"]))
        self.passed = False
        self.tested_hash = self.engine_hash_seen = self._engine_hash()
        # The kernel restarted empty: re-run the conversation's python cells (edits disabled), then say what it keeps.
        cells = state.get("cells") or []
        replay = self.kernel.replay(cells)
        self._log({"turn": self.result.turns, "replay": {
            "cells": len(cells), "replayed": replay.get("replayed", 0), "failed": replay.get("failed", []),
            "skipped": replay.get("skipped", 0), "seconds": replay.get("seconds", 0.0), **({"error": replay["error"]} if "error" in replay else {}),
        }})
        self._say("user", resume_note(replay, kernel_names_text(*self.kernel.names())))
        self._log({"turn": self.result.turns, "resumed": {"step": self.focus, "messages": len(self.messages)}})
        return True

    def _rebuild_legacy(self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        """For transcripts from before "message" records: the conversation of the last stepwise segment (from its last
        step_start), rebuilt from what was logged. Approximate where the old records do not keep the text: an automatic
        test's report is its logged (shortened) copy, the commit sentence is inferred from tests.jsonl."""
        if not self.stepwise:
            return None
        starts = [i for i, r in enumerate(records) if "step_start" in r]
        if not starts:
            return None
        begin = starts[-1]
        version = None
        for r in records[:begin]:
            if isinstance(r.get("engine_change"), dict) and r["engine_change"].get("version"):
                version = r["engine_change"]["version"]
        engine_file = self.dir / "engine_versions" / f"v{version or 1:04d}.py"  # no change logged before: the first version
        if not engine_file.exists():
            engine_file = self.engine_path
        engine = engine_file.read_text(encoding="utf-8")
        k = int(records[begin]["step_start"]["step"])
        visible = Trace(self.full_trace.game_id, self.full_trace.steps[: k + 1], {**self.full_trace.meta, "focus": k})
        read = hashline.render_read(engine, max_chars=READ_CHARS_IN_MESSAGES, fold=fixed_block_lines(engine))
        first = episode_message(self.game, visible, k, records[begin]["step_start"]["report"], read, self.history)
        system = system_prompt(self.match, self.interface, self.images, self.mode, self.history)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": first}]
        auto_passed = {}
        tests = self.dir / "tests.jsonl"
        if tests.exists():
            for line in tests.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("auto") is True:
                    auto_passed[entry.get("turn")] = bool(entry.get("passed")) and entry.get("level") is None
        focus, prompt_tokens = k, 0
        last_tool = None
        calls: Any = iter(())
        cells: list[dict[str, Any]] = []
        last_turn = None

        def engine_listing() -> str:  # engine.py as it was at this point of the transcript
            file = self.dir / "engine_versions" / f"v{version or 1:04d}.py"
            text = (file if file.exists() else self.engine_path).read_text(encoding="utf-8")
            return hashline.render_read(text, max_chars=READ_CHARS_IN_MESSAGES, fold=fixed_block_lines(text))

        def pictures(r: dict[str, Any], note: str | None, tests_note: bool) -> list[dict[str, Any]]:
            parts: list[dict[str, Any]] = [{"type": "text", "text": note}] if note else []
            for path, caption in zip(r["images"], r.get("captions") or [""] * len(r["images"])):
                png = (self.dir / path).read_bytes()
                text = (TEST_IMAGE_NOTE + " " + caption) if tests_note and "_show" not in path else caption
                parts.append({"type": "text", "text": text})
                parts.append(self._image_part(png, self.dir / path))
            return parts

        def with_text(message: dict[str, Any]) -> list[dict[str, Any]]:
            content = message["content"]
            return content if isinstance(content, list) else [{"type": "text", "text": content}]

        for r in records[begin + 1 :]:
            if "finish_reason" in r:
                prompt_tokens = max(prompt_tokens, (r.get("usage") or {}).get("prompt_tokens") or 0)  # ever over: compacted
                assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
                if r.get("reasoning"):
                    assistant["reasoning"] = r["reasoning"]
                if r.get("tool_calls"):
                    assistant["tool_calls"] = r["tool_calls"]
                    calls = iter(r["tool_calls"])
                last_turn = r.get("turn")
                messages.append(assistant)
                if not r.get("tool_calls"):
                    messages.append({"role": "user", "content": CONTINUE})
            elif "tool" in r:
                call = next(calls, None)
                last_tool = {"role": "tool", "tool_call_id": call["id"] if call else "", "content": r["output"]}
                messages.append(last_tool)
                code = cell_code(call) if call and r["tool"] == "python" else None
                if code is not None and not str(r["output"]).startswith(PYTHON_PAUSED[:40]):
                    cells.append({"turn": r.get("turn"), "code": code})
            elif "auto_test" in r and last_tool is not None:
                hint = COMMIT_HINT.format(k=focus) if auto_passed.get(r["turn"]) else ""
                last_tool["content"] += AUTO_TEST.format(report=_truncate(r["auto_test"], AUTO_TEST_CHARS)) + hint
            elif "nudge" in r and last_tool is not None:
                last_tool["content"] += NUDGE.format(n=r["nudge"])
            elif "images" in r:
                if any("_episode" in path for path in r["images"]):
                    messages[1]["content"] = with_text(messages[1]) + pictures(r, None, True)
                elif any("_advance" in path for path in r["images"]) and messages[-1]["role"] == "user":
                    messages[-1]["content"] = with_text(messages[-1]) + pictures(r, None, True)
                else:
                    self._hide_images(messages)
                    messages.append({"role": "user", "content": pictures(r, IMAGE_NOTE, True)})
            elif isinstance(r.get("engine_change"), dict) and r["engine_change"].get("version"):
                version = r["engine_change"]["version"]
            elif "advance" in r:
                a = r["advance"]
                focus = a["next"]
                messages.append({"role": "user", "content": advance_message(
                    self.full_trace, a["fixed"], a["next"], a["report"], self.history, engine_listing())})
        if self._drop_unanswered(messages):
            cells = [c for c in cells if c["turn"] != last_turn]
        self.messages = messages
        over = prompt_tokens > self.model.compact_prompt_tokens
        if over and not self.condense:
            self._compact()
        # With context "condense" the condenser fires once the conversation is written back (_resume_conversation),
        # so its record follows the messages it covers.
        return {"messages": self.messages, "focus": focus, "legacy": True, "cells": cells, "condense": over and self.condense}

    @staticmethod
    def _has_engine_listing(message: dict[str, Any]) -> bool:
        content = message.get("content")
        parts = [content] if isinstance(content, str) else [p.get("text", "") for p in content or [] if p.get("type") == "text"]
        return message["role"] == "user" and any(ENGINE_HEADER in part for part in parts)

    def _compact(self) -> None:
        """Elide old tool outputs, old reasoning, large tool-call arguments and the engine.py listings of all
        but the latest step message to bound the prompt."""
        listed = [i for i, m in enumerate(self.messages) if self._has_engine_listing(m)]
        for i in listed[:-1]:
            content = self.messages[i]["content"]
            if isinstance(content, str):
                self.messages[i]["content"] = elide_engine_listing(content)
            else:
                self.messages[i]["content"] = [
                    {**p, "text": elide_engine_listing(p["text"])} if p.get("type") == "text" else p for p in content
                ]
        tool_indices = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        for i in tool_indices[: -self.model.keep_recent_tool_outputs]:
            content = self.messages[i]["content"]
            if isinstance(content, str) and len(content) > 400:
                self.messages[i]["content"] = content[:200] + f"\n[... older output elided to save context ({len(content)} chars)]"
        assistant_indices = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        for i in assistant_indices[: -self.model.keep_recent_reasoning]:
            reasoning = self.messages[i].get("reasoning") or ""
            if len(reasoning) > self.model.old_reasoning_chars + 100:
                self.messages[i]["reasoning"] = "[earlier reasoning trimmed] ..." + reasoning[-self.model.old_reasoning_chars :]
        recent_cut = tool_indices[-self.model.keep_recent_tool_outputs] if len(tool_indices) >= self.model.keep_recent_tool_outputs else 0
        for i, m in enumerate(self.messages[:recent_cut]):
            for call in m.get("tool_calls") or []:
                args = call["function"]["arguments"]
                if len(args) > 1500:
                    call["function"]["arguments"] = _elide_arguments(args)

    def _over_budget(self) -> str | None:
        u = self.result.usage
        if self.result.turns >= self.budget.max_turns:
            return "budget_turns"
        if u.completion_tokens >= self.budget.max_output_tokens:
            return "budget_tokens"
        if u.cost_usd >= self.budget.max_cost_usd:
            return "budget_cost"
        if self._elapsed_minutes() >= self.budget.max_minutes:
            return "budget_time"
        return None

    # --- the opening --------------------------------------------------------------

    def _opening_code(self) -> str | None:
        """OPENING_CODE with the anchors of make_level and its return State(...) block, or None when
        engine.py does not have the starting template's make_level."""
        lines, _ = hashline.split_lines(self.engine_path.read_text(encoding="utf-8"))
        try:
            head = next(i for i, line in enumerate(lines, 1) if line.startswith("def make_level("))
            start = next(i for i in range(head, len(lines) + 1) if lines[i - 1] == "    return State(")
            end = next(i for i in range(start, len(lines) + 1) if lines[i - 1] == "    )")
        except StopIteration:
            return None
        return OPENING_CODE.format(
            head=hashline.anchor(lines, head), start=hashline.anchor(lines, start), end=hashline.anchor(lines, end),
            split=OPENING_SPLIT,
        )

    def _open(self) -> dict[str, Any] | None:
        """Play the first round before the first turn: level 0's sprite code into make_level, then the tests.
        Returns what the first message shows ("sprites", "report", "first_fail"), or None when it could
        not be done (engine.py is then as it was)."""
        code = self._opening_code()
        if code is None:
            self.result.opening = {"error": "engine.py has no template make_level to fill"}
            return None
        self._in_opening = True
        try:
            output = self.kernel.execute(code)
            sprites, _, edited = output.partition(OPENING_SPLIT)
            edited = edited.strip()
            if not edited.startswith("engine.py: ") or "Syntax OK" not in edited:
                if edited.startswith("engine.py: "):  # applied but broken: back to the template
                    self.kernel.editor.undo(1)
                self.result.opening = {"error": output[-1500:]}
                self._log({"turn": 0, "opening_error": output})
                return None
            report = self._run_tests(None, 1, auto="opening")
        finally:
            self._in_opening = False
        if isinstance(report, str):
            self.result.opening = {"error": report}
            self._log({"turn": 0, "opening_error": report})
            return None
        summary = report.summary()
        first_fail = None if report.passed else summary.get("first_fail")
        self.result.opening = {
            "exact": "exactly: yes" in sprites, "first_fail": first_fail, "passing_prefix": summary.get("passing_prefix"),
        }
        self._log({"turn": 0, "opening": {"sprites": sprites.strip(), "edit": edited, "report": report.text}})
        return {"sprites": sprites, "report": _truncate(report.text, REPORT_CHARS), "first_fail": first_fail}

    def _opening_content(self, text: str) -> str | list[dict[str, Any]]:
        """The first message: its text, then the opening test's pictures (when images are on)."""
        pictures, self.pending_test = (self.pending_test if self.images else []), []
        if not pictures:
            return text
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for caption, path, png in pictures:
            content.append({"type": "text", "text": TEST_IMAGE_NOTE + " " + caption})
            content.append(self._image_part(png, path))
        self.result.image_messages += 1
        self._log({"turn": self.result.turns + self.stepwise,
                   "images": [str(path.relative_to(self.dir)) for _, path, _ in pictures],
                   "captions": [caption for caption, _, _ in pictures]})
        return content

    def play_opening(self) -> dict[str, Any] | None:
        """Only the opening, for the stepwise driver: engine.py from the template, level 0's sprite code
        (recording[0].pieces_after.code()) put into make_level, the first test. Returns what _open returns."""
        self.setup()
        self.started = time.time()
        try:
            return self._open()
        finally:
            self.kernel.stop()

    # --- stepwise ----------------------------------------------------------------

    def _replay_all(self) -> int | None:
        """Replay the whole recording; the first failing step, or None when everything passes."""
        report = replay_test(self.engine_path, self.full_trace, failures=1, scratch_root=self.dir, match=self.match)
        summary = report.summary()
        self.result.passing_prefix = summary.get("passing_prefix")
        if report.passed:
            return None
        first = summary.get("first_fail")
        return len(self.full_trace) - 1 if first is None else int(first)  # a contract failure alone: test every step

    def _focus_on(self, k: int) -> None:
        """Show the model the recording up to step k: visible_trace/ holds steps 0..k, its tests replay them."""
        shutil.rmtree(self.trace_dir, ignore_errors=True)
        self.trace = Trace(self.full_trace.game_id, self.full_trace.steps[: k + 1], {**self.full_trace.meta, "focus": k})
        self.trace.save(self.trace_dir)
        self.focus = self.result.step = k
        self.kernel.refocus(k)

    def _step_report(self, auto: str) -> str:
        report = self._run_tests(None, 1, auto=auto)
        return report if isinstance(report, str) else _truncate(report.text, REPORT_CHARS)

    def _stepwise_opening(self) -> str | list[dict[str, Any]]:
        """The first message: fix step k (with the report's picture)."""
        text = self._step_report("episode")
        self._log({"turn": self.result.turns + 1, "step_start": {"step": self.focus, "report": text}})
        return self._opening_content(
            episode_message(self.game, self.trace, self.focus, text, self._read_engine(fold=True, max_chars=READ_CHARS_IN_MESSAGES),
                            self.history)
        )

    def _advance(self, commit: dict[str, Any]) -> bool:
        """A commit was accepted (steps 0..k pass): record it and replay on. Returns False when the whole
        recording passes (the session ends); otherwise moves to the next failing step and adds a user
        message saying so."""
        fixed = self.focus
        k = self._replay_all()
        entry = {"turn": self.result.turns, "fixed": fixed, "next": k, **commit}
        self.result.advances.append(entry)
        self._log({"turn": self.result.turns, "commit": {key: v for key, v in entry.items() if key != "turn"}})
        if k is None:
            self.result.status = "passed"
            self.result.first_pass_turn = self.result.turns
            return False
        self._focus_on(k)
        self.passed = False
        text = self._step_report("advance")
        self.tested_hash = self._engine_hash()
        self._log({"turn": self.result.turns, "advance": {"fixed": fixed, "next": k, "report": text}})
        engine_read = self._read_engine(fold=True, max_chars=READ_CHARS_IN_MESSAGES)
        names = kernel_names_text(*self.kernel.names())
        self._say("user", self._opening_content(advance_message(self.full_trace, fixed, k, text, self.history, engine_read, names)))
        return True

    def setup(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not self.engine_path.exists():
            self.engine_path.write_text(render_skeleton(self.game, self.trace[0].available_actions), encoding="utf-8")
        self.kernel.editor.sync()  # version 1, or a version for a change made while no session ran

    def run(self) -> AgentResult:
        self.setup()
        self.started = time.time()
        resumed = False
        budget = self.model.thinking_budget
        if budget is not None:  # does OpenRouter say the model takes a thinking budget? (once per process)
            key = (self.model.model, int(budget))
            if key not in _BUDGET_CHECKED:
                _BUDGET_CHECKED[key] = thinking_budget_warning(*key)
                if _BUDGET_CHECKED[key]:
                    print(_BUDGET_CHECKED[key], flush=True)
        if self.stepwise:
            if self._restore():  # the totals of an interrupted run; then its conversation, when it can be continued
                self._replay_all()
                resumed = self._resume_conversation()
        if resumed:
            pass
        elif self.stepwise:
            first = self._replay_all()
            if first is None:
                self.result.status = "passed"
                self._final_test()
                self._save_result()
                return self.result
            self._focus_on(first)
            opening = self._stepwise_opening()
        elif self._restore():
            report = replay_test(self.engine_path, self.trace, failures=1, scratch_root=self.dir, match=self.match)
            opening = resume_user_message(
                self.game, self.trace, self.result.turns, _truncate(report.text, 6000),
                self._read_engine(fold=True, max_chars=READ_CHARS_IN_MESSAGES), self.prior_notes,
            )
        else:
            done = self._open() if self.opening else None
            opening = self._opening_content(
                first_user_message(self.game, self.trace, self._read_engine(fold=True, max_chars=READ_CHARS_IN_MESSAGES), done)
            )
        system = system_prompt(self.match, self.interface, self.images, self.mode, self.history)
        if self.budget.python_quota is not None:
            system += (
                f"\n\n# Analysis quota\nThe python tool pauses after {self.budget.python_quota} calls without any change to "
                "engine.py (only calls that change it with edit_file() or undo_edit() run), and resumes as soon as engine.py changes."
            )
        if not resumed:
            self.messages = []
            self._say("system", system)
            self._say("user", opening)
        # engine.py as the session starts counts as tested, so any change to it triggers an automatic test.
        if not resumed:
            self.tested_hash = self._engine_hash()
            self.engine_hash_seen = self.tested_hash
        idle_turns = 0
        try:
            while True:
                self._save_result()
                reason = self._over_budget()
                if reason:
                    self.result.status = reason
                    break
                response = self.client.chat(self._context(), tools(self.images, self.mode, self.history))
                self.result.provider_errors = len(getattr(self.client, "provider_errors", []))
                self.result.turns += 1
                usage = response.get("usage") or {}
                self.result.usage.add(usage)
                choice = response["choices"][0]
                message = choice["message"]
                tool_calls = message.get("tool_calls") or []
                assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
                if message.get("reasoning"):
                    assistant["reasoning"] = message["reasoning"]
                if tool_calls:
                    assistant["tool_calls"] = [
                        {"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
                        for c in tool_calls
                    ]
                self.messages.append(assistant)
                self._log(
                    {
                        "turn": self.result.turns,
                        "finish_reason": choice.get("finish_reason"),
                        "content": message.get("content"),
                        "reasoning": message.get("reasoning"),
                        "tool_calls": assistant.get("tool_calls"),
                        "usage": usage,
                        "provider": response.get("provider"),
                    }
                )
                if not tool_calls:
                    idle_turns += 1
                    if idle_turns >= 4:
                        self.result.status = "stalled"
                        break
                    self._say("user", CONTINUE)
                    continue
                idle_turns = 0
                self.turns_since_test += 1
                for call in assistant["tool_calls"]:
                    name = call["function"]["name"]
                    counted = "python" if name in BUILTIN_FUNCTIONS else name  # a built-in called as a tool runs as python
                    self.result.tool_calls[counted] = self.result.tool_calls.get(counted, 0) + 1
                    t0 = time.time()
                    output = self._dispatch(name, call["function"]["arguments"])
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                    record = {"turn": self.result.turns, "tool": counted, "id": call["id"], "seconds": round(time.time() - t0, 2), "output": output}
                    if counted != name:
                        record["called_as"] = name
                    self._log(record)
                changed = self.tested_hash is not None and self._engine_hash() != self.tested_hash
                if changed and self.commit is not None and self.commit["engine_sha"] != self._engine_hash():
                    self.commit = None  # engine.py changed after the commit in the same turn
                    self._add_to_last(COMMIT_DROPPED)
                if changed and (self.stepwise or not self.passed):
                    previous = self.last_signature
                    tested = self._run_tests(None, 1, auto=True)
                    report = tested if isinstance(tested, str) else tested.text
                    if not isinstance(tested, str) and not tested.passed and tested.signature == previous:
                        note = AUTO_TEST_SAME.format(brief=tested.brief())  # the full report is in tests.jsonl and the log
                    else:
                        note = AUTO_TEST.format(report=_truncate(report, AUTO_TEST_CHARS))
                    self._add_to_last(note + self._commit_hint(tested))
                    self.result.auto_tests += 1
                    self._log({"turn": self.result.turns, "auto_test": report[:AUTO_TEST_CHARS]})
                elif self.turns_since_test and self.turns_since_test % TEST_NUDGE_TURNS == 0:
                    self._add_to_last(NUDGE.format(n=self.turns_since_test))
                    self.result.nudges += 1
                    self._log({"turn": self.result.turns, "nudge": self.turns_since_test})
                # After the tool messages (and the automatic test): the turn's images.
                self._attach_images()
                if self.stepwise and self.commit is not None:  # only a commit moves on
                    commit, self.commit = self.commit, None
                    if not self._advance(commit):
                        break
                elif self.passed and not self.stepwise:
                    self.result.status = "passed"
                    break
                if (usage.get("prompt_tokens") or 0) > self.model.compact_prompt_tokens:
                    if self.condense:
                        self._condense_now()
                    else:
                        self._log({"turn": self.result.turns, "compact": True})
                        self._compact()
        except Exception as exc:  # noqa: BLE001  (record the failure in result.json)
            self.result.status = "error"
            self.result.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.kernel.stop()
            self._final_test()
            self._save_result()
        return self.result

    def _final_test(self) -> None:
        """Authoritative full replay of the final engine.py."""
        try:
            report = replay_test(self.engine_path, self.full_trace, details=3, scratch_root=self.dir, match=self.match)
            self.result.final = report.summary()
            (self.dir / "final_test.txt").write_text(report.text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.result.final = {"error": f"{type(exc).__name__}: {exc}"}
