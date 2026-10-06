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

Context (ModelConfig.context): "compact" and "condense" act at the same moments, after a turn whose request went
over compact_prompt_tokens, and leave the conversation alone in between, so the prompt's prefix stays the same
from one request to the next and the provider's prompt cache hits. "compact" shortens the conversation in
place by age (old tool outputs, old reasoning, long old arguments, older engine.py listings; a "compact"
record). "condense" keeps the full conversation and condenses it by iteration (engine_re.condense) into a
view that becomes the prefix of every request until the next firing, the messages added since following it
as they are (a "condense" record with the estimate marks each firing). "rebuilt" (the play agent's, see
rebuilt_context) never shortens the conversation: every request is rebuilt from it as the system prompt, one
user message with the compacted context (the commit turns, the current phase message, the older turns since
it) and the last rebuilt_keep_turns turns as they are (a "rebuilt" record per request gives the composition).

A turn is one model reply with everything said before the next reply: its tool outputs (with the text the harness
appends to them), its image message and the user messages the harness adds after it (the next phase message, the
"continue" line, the resume note). Turn 0 is the system prompt and the opening message. Every message in
self.messages is a TurnMessage that knows its turn, and the transcript logs the same number on every record.

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

import base64
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
    restart_note, resume_user_message, system_prompt, tools,
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
# A forked run (engine_re.tools.fork_run): the cells' edits to notes.md and the other workspace files were applied.
REPLAY_DONE_FILES = (
    "The python kernel restarted and re-ran your {n} python cell{s} in order with edits to engine.py disabled (its versions "
    "are kept) and the edits to notes.md and your other files applied, so your variables, functions and notes are back{failed}."
)
REPLAY_FAILED = "; cells that raised when re-run (as before, or because engine.py changed later): turn{s} {turns}"
REPLAY_SKIPPED = " The last {n} cell{s} were not re-run (the replay's time ran out)."
REPLAY_NONE = "The python kernel restarted (there were no python cells to re-run)."


def resume_note(replay: dict[str, Any], names: str, files: bool = False) -> str:
    """RESUME_NOTE for a replay result (KernelClient.replay) and the kernel's names line (kernel_names_text); `files`:
    the replay applied the edits to files other than engine.py (a fork)."""
    n = int(replay.get("replayed") or 0)
    if not n and not replay.get("skipped"):
        text = REPLAY_NONE
    else:
        turns = sorted({int(f["turn"]) for f in replay.get("failed") or [] if f.get("turn") is not None})
        failed = REPLAY_FAILED.format(s="s" if len(turns) > 1 else "", turns=", ".join(map(str, turns))) if turns else ""
        text = (REPLAY_DONE_FILES if files else REPLAY_DONE).format(n=n, s="" if n == 1 else "s", failed=failed)
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
    # How the conversation is kept within bounds. "compact" (the settings above, applied in place) and "condense"
    # (engine_re.condense: the full conversation is kept and condensed by iteration into the prefix of every request
    # until the next firing; the turns since follow it as they are) both apply after a request went over
    # compact_prompt_tokens. The condenser keeps the reasoning and failed commands of the current iteration's last
    # condense_keep_turns turns; its safety cap estimates tokens with condense_chars_per_token. "rebuilt" (the play
    # agent) has no threshold: every request is rebuilt from the full conversation (rebuilt_context), the last
    # rebuilt_keep_turns turns sent as they are and the rest compacted into one user message.
    context: str = "compact"
    condense_keep_turns: int = 10
    condense_chars_per_token: float = 3.0
    rebuilt_keep_turns: int = 10
    # "rebuilt": the model's context window and the tokens reserved for its reply (None: max_tokens); the request budget
    # is the window minus the reserve minus REQUEST_SAFETY_TOKENS, and a rebuilt request over it is shrunk (shrink_step).
    context_window: int = 131_072
    reply_reserve: int | None = None

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


class TurnMessage(dict):
    """A message of the conversation that knows the turn it belongs to (see the module docstring: a turn is one
    model reply with its tool outputs and the harness's messages before the next reply; turn 0 the system prompt and
    the opening) and, for a PLAN or FIT message (the stepwise step messages too), which phase it opens. The tags live
    outside the dict, so the request (json) and the transcript ("message" records) see a plain message, and a
    comparison with a plain dict holds."""

    __slots__ = ("turn", "phase")

    def __init__(self, message: dict[str, Any], turn: int | None = None, phase: str | None = None):
        super().__init__(message)
        self.turn = turn
        self.phase = phase


def message_turn(message: dict[str, Any], default: int = 0) -> int:
    turn = getattr(message, "turn", None)
    return default if turn is None else int(turn)


def message_chars(message: dict[str, Any]) -> int:
    """The characters the model reads in a message: its text (every text part), its reasoning and its tool calls'
    arguments; images count for none (the "rebuilt" record counts them apart)."""
    content = message.get("content")
    n = len(content) if isinstance(content, str) else sum(len(p.get("text") or "") for p in content or [] if p.get("type") == "text")
    n += len(message.get("reasoning") or "")
    n += sum(len(c["function"].get("arguments") or "") for c in message.get("tool_calls") or [])
    return n


def message_images(message: dict[str, Any]) -> int:
    content = message.get("content")
    return sum(p.get("type") == "image_url" for p in content) if isinstance(content, list) else 0


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text") or "" for p in content or [] if p.get("type") == "text")


REBUILT_CLOSING = "The context has been compacted. Continue from the context above."
# The token estimate of a request (the base harness's, inference/agent/tool_agent.py): the payload rendered as json
# (ensure_ascii=False) with every image replaced by a placeholder, divided by a characters-per-token figure calibrated
# from each response's prompt_tokens (seed 3, clamped to [1.0, 3.3]: under-counting overflows the context, the
# dangerous direction), plus the images at their vision cost: one token per 32x32 patch plus two sentinels.
CHARS_PER_TOKEN_SEED = 3.0
CHARS_PER_TOKEN_MIN = 1.0
CHARS_PER_TOKEN_MAX = 3.3
VISION_PATCH_PIXELS = 32
VISION_SENTINEL_TOKENS = 2
IMAGE_TOKENS_FALLBACK = 402
REQUEST_SAFETY_TOKENS = 512  # the budget is the window minus the reply reserve minus this
COUNT_MARGIN_PERCENT = 2  # an exact count's margin on top: its residual against the reported prompt_tokens (engine_re.tokens)
REBUILT_MIN_REASONING_TURNS = 3  # the last turns that always keep their reasoning (and are never touched)
REBUILT_MIN_COMMIT_TURNS = 5  # the commit turns the compacted message always keeps
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(data_url: str) -> tuple[int, int] | None:
    """(width, height) from the IHDR of a PNG data URL, decoding only its head; None when it is not a readable PNG."""
    marker = "base64,"
    index = data_url.find(marker)
    if index < 0:
        return None
    head = data_url[index + len(marker): index + len(marker) + 32]
    if len(head) < 32:
        return None
    try:
        raw = base64.b64decode(head, validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) < 24 or not raw.startswith(_PNG_SIGNATURE) or raw[12:16] != b"IHDR":
        return None
    width, height = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return (width, height) if width > 0 and height > 0 else None


def image_part_tokens(part: dict[str, Any]) -> int:
    """The vision tokens of an image part: one per merged 32x32 patch plus the two sentinels (a 536x554 PLAN frame is
    308, a 1060x554 test comparison 614); IMAGE_TOKENS_FALLBACK when the PNG header cannot be read."""
    url = (part.get("image_url") or {}).get("url") or ""
    dims = png_dimensions(url) if url else None
    if dims is None:
        return IMAGE_TOKENS_FALLBACK
    width, height = dims
    return -(-width // VISION_PATCH_PIXELS) * -(-height // VISION_PATCH_PIXELS) + VISION_SENTINEL_TOKENS


def split_for_estimate(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The messages with every image replaced by a short placeholder (the originals untouched), and the images' vision
    tokens."""
    image_tokens = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_tokens += image_part_tokens(part)
                parts.append({"type": "image_url", "image_url": {"url": "<image>"}})
            else:
                parts.append(part)
        out.append({**message, "content": parts})
    return out, image_tokens


def render_request(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[str, int]:
    """The request's text as the estimate and the calibration both count it (the same rendering, so an image's cost
    cancels out between them): the payload's messages (images as placeholders), tools and tool_choice as json without
    \\u escapes; and the images' vision tokens."""
    scrubbed, image_tokens = split_for_estimate(messages)
    payload: dict[str, Any] = {"messages": scrubbed}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), image_tokens


def estimate_request_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, chars_per_token: float) -> dict[str, int]:
    """The estimate of a request: {"tokens", "text_tokens", "image_tokens", "text_chars"}."""
    rendered, image_tokens = render_request(messages, tools)
    text_tokens = -(-len(rendered) // max(0.1, chars_per_token))
    return {"tokens": int(text_tokens) + image_tokens, "text_tokens": int(text_tokens), "image_tokens": image_tokens,
            "text_chars": len(rendered)}


def is_context_length_error(error: str) -> bool:
    """A request the provider rejected as too long (the base harness's _is_context_length_error, with OpenRouter's
    wordings)."""
    text = error.lower().replace("’", "'")
    return any(s in text for s in (
        "context length", "maximum context", "too many tokens", "context_length_exceeded",
        "reduce the length of the input prompt", "parameter=input_tokens", '"param":"input_tokens"',
    ))


def _render_call(call: dict[str, Any]) -> str:
    """A tool call as the compacted context shows it: its name and its arguments as the model wrote them (the python
    code, the commit message, the actions and the note), one argument per line."""
    name = call["function"]["name"]
    raw = call["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        return f"[call {name}] {raw}"
    lines = [f"[call {name}]"]
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value)
        lines.append(f"{key}: {text}" if "\n" not in text else f"{key}:\n{text}")
    return "\n".join(lines)


def render_turn(turn: int, messages: list[dict[str, Any]], calls: tuple[str, ...] | None = None, header: str | None = None) -> str:
    """One turn of the conversation as the compacted context shows it: a header ("Turn 61:", or "Turn 57
    (commit_moves):" when `calls` names the tools it is rendered for), the assistant's text, its tool calls with their
    arguments and their outputs (the harness's appends included, since they are part of the output), and the harness's
    user messages of the turn (image messages left out). Without `calls` every call is rendered; with it only the
    calls to those tools. The reasoning is never rendered."""
    assistant = next((m for m in messages if m["role"] == "assistant"), None)
    tool_calls = (assistant or {}).get("tool_calls") or []
    chosen = [c for c in tool_calls if calls is None or c["function"]["name"] in calls]
    if header is None:
        named = ", ".join(dict.fromkeys(c["function"]["name"] for c in chosen)) if calls is not None else ""
        header = f"Turn {turn} ({named}):" if named else f"Turn {turn}:"
    parts = [header]
    text = (assistant or {}).get("content") or ""
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    outputs = {m.get("tool_call_id"): m for m in messages if m["role"] == "tool"}
    for call in chosen:
        parts.append(_render_call(call))
        answer = outputs.get(call["id"])
        if answer is not None:
            parts.append("[output]\n" + (answer.get("content") or "").rstrip())
    for m in messages:
        if m["role"] == "user" and not getattr(m, "phase", None) and not _is_image_message(m):
            parts.append("[harness] " + _message_text(m).strip())
    return "\n".join(parts)


def _is_image_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and bool(content) and content[0].get("type") == "text" and content[0].get("text") == IMAGE_NOTE


SHRINK_STEPS = ("reasoning", "older", "commits", "phase_image")  # the order the shrink of a rebuilt request takes


def rebuilt_context(messages: list[dict[str, Any]], keep_turns: int = 10,
                    commit_tools: tuple[str, ...] = ("commit_engine", "commit_moves"),
                    shrink: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The messages of a request in the "rebuilt" context mode, from the full conversation (TurnMessage-tagged), and
    the composition for the "rebuilt" record. With T the latest turn and the window the last `keep_turns` turns
    (T - keep_turns < t <= T):

    1. the system prompt;
    2. one user message, the compacted context, holding in order: (a) every turn older than the window and before
       the current phase message (the latest PLAN or FIT message, by its tag) in which the model called a commit
       tool, rendered as its text, those calls and their outputs; (b) the phase message in full (its image parts as
       the conversation holds them) when it is older than the window; (c) the turns after it that are older than the
       window, each rendered with every call and output but no reasoning; then REBUILT_CLOSING. The message is left
       out when it would hold nothing;
    3. the messages of the window as they are (reasoning, tool calls, outputs, images, a phase message at its place).

    Turns older than the window and before the phase message in which nothing was committed are not sent.

    `shrink` (what shrink_step adds, when a request is over budget): "reasoning": the oldest N turns of the window are
    sent without their reasoning (never the last REBUILT_MIN_REASONING_TURNS); "older": the older turns (c) are left
    out; "commits": the oldest N commit turns are left out (never the last REBUILT_MIN_COMMIT_TURNS); "phase_image":
    the phase message's images are replaced by IMAGE_PLACEHOLDER, wherever it is. The phase message's text and the
    last REBUILT_MIN_REASONING_TURNS turns are never touched."""
    shrink = shrink or {}
    if not messages:
        return [], {"commit_turns": 0, "phase_turn": None, "older_turns": 0, "chars": {"commits": 0, "phase": 0, "older": 0, "recent": 0}}
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    latest = max((message_turn(m) for m in messages), default=0)
    cut = latest - keep_turns  # turns <= cut are older than the window
    phase_index = next((i for i in range(len(rest) - 1, -1, -1) if getattr(rest[i], "phase", None)), None)
    if phase_index is None:  # no tagged phase message: the opening message stands for it
        phase_index = next((i for i, m in enumerate(rest) if m["role"] == "user"), 0)
    phase_turn = message_turn(rest[phase_index]) if rest else 0
    phase_kind = (getattr(rest[phase_index], "phase", None) or "phase").upper() if rest else "PHASE"
    # The turns before the phase message, the phase message, the turns after it: (turn, messages) in order.
    before: dict[int, list[dict[str, Any]]] = {}
    after: dict[int, list[dict[str, Any]]] = {}
    for i, m in enumerate(rest):
        if i != phase_index:
            (before if i < phase_index else after).setdefault(message_turn(m), []).append(m)
    commits: list[str] = []
    for turn, group in before.items():
        if turn > cut:
            continue
        assistant = next((m for m in group if m["role"] == "assistant"), None)
        if assistant and any(c["function"]["name"] in commit_tools for c in assistant.get("tool_calls") or []):
            commits.append(render_turn(turn, group, calls=commit_tools))
    commits_dropped = min(int(shrink.get("commits") or 0), max(0, len(commits) - REBUILT_MIN_COMMIT_TURNS))
    commits = commits[commits_dropped:]
    older: list[str] = []
    for turn, group in after.items():
        if turn > cut:
            continue
        older.append(render_turn(turn, group, header=f"Turn {turn} (continued):" if turn == phase_turn else None))
    older_dropped = bool(shrink.get("older")) and bool(older)
    if older_dropped:
        older = []
    phase_in_compacted = phase_turn <= cut
    recent = [m for i, m in enumerate(rest) if message_turn(m) > cut and (i != phase_index or not phase_in_compacted)]
    # The window's oldest turns without their reasoning (copies; the conversation keeps it).
    window_turns = sorted({message_turn(m) for m in recent if m["role"] == "assistant"})
    stripped = set(window_turns[: max(0, min(int(shrink.get("reasoning") or 0), len(window_turns) - REBUILT_MIN_REASONING_TURNS))])
    if stripped:
        recent = [TurnMessage({k: v for k, v in m.items() if k != "reasoning"}, turn=message_turn(m), phase=getattr(m, "phase", None))
                  if m["role"] == "assistant" and message_turn(m) in stripped and m.get("reasoning") else m for m in recent]
    phase_image_dropped = False
    if shrink.get("phase_image") and rest and message_images(rest[phase_index]):
        phase_message = rest[phase_index]
        hidden = TurnMessage({**phase_message, "content": [
            {"type": "text", "text": IMAGE_PLACEHOLDER} if p.get("type") == "image_url" else p for p in phase_message["content"]]},
            turn=phase_turn, phase=getattr(phase_message, "phase", None))
        rest = [hidden if i == phase_index else m for i, m in enumerate(rest)]
        recent = [hidden if m is phase_message else m for m in recent]
        phase_image_dropped = True
    parts: list[dict[str, Any]] = []
    sections: list[str] = []
    if commits:
        sections.append("Earlier turns of this game in which you committed (your text, the call and its result):\n\n" + "\n\n".join(commits))
    phase_chars = 0
    if phase_in_compacted:
        phase_message = rest[phase_index]
        if sections:
            parts.append({"type": "text", "text": "\n\n".join(sections)})
            sections = []
        parts.append({"type": "text", "text": f"Turn {phase_turn} (the current {phase_kind} message):"})
        content = phase_message.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            parts.extend(dict(p) for p in content or [])
        phase_chars = message_chars(phase_message)
    if older:
        sections.append("The turns since that message, before the last ones:\n\n" + "\n\n".join(older))
    if commits or phase_in_compacted or older:
        sections.append(REBUILT_CLOSING)
        parts.append({"type": "text", "text": "\n\n".join(sections)})
    merged: list[dict[str, Any]] = []  # adjacent text parts as one, so the message is one text unless it has images
    for part in parts:
        if part.get("type") == "text" and merged and merged[-1].get("type") == "text":
            merged[-1] = {"type": "text", "text": merged[-1]["text"] + "\n\n" + part["text"]}
        else:
            merged.append(part)
    compacted: list[dict[str, Any]] = []
    if merged:
        content = merged[0]["text"] if len(merged) == 1 else merged
        compacted = [{"role": "user", "content": content}]
    out = list(system[:1]) + compacted + recent
    chars = {
        "commits": sum(len(t) for t in commits), "phase": phase_chars, "older": sum(len(t) for t in older),
        "recent": sum(message_chars(m) for m in recent),
    }
    stats = {
        "turn": latest, "commit_turns": len(commits), "phase_turn": phase_turn, "phase_compacted": phase_in_compacted,
        "older_turns": len(older), "recent_turns": len({message_turn(m) for m in recent}), "chars": chars,
        "images": sum(message_images(m) for m in out), "messages": len(out),
    }
    applied = {"reasoning": len(stripped), "older": older_dropped, "commits": commits_dropped, "phase_image": phase_image_dropped}
    if any(applied.values()):
        stats["shrink"] = applied
    return out, stats


def shrink_step(shrink: dict[str, Any], stats: dict[str, Any], keep_turns: int = 10) -> dict[str, Any] | None:
    """The next shrink state after `shrink` gave the view with `stats`, in the order of SHRINK_STEPS: one more turn of
    the window without reasoning (while more than REBUILT_MIN_REASONING_TURNS turns keep it), then the older turns
    out, then one more commit turn out (while more than REBUILT_MIN_COMMIT_TURNS stay), then the phase message's
    image out; None when every step is taken."""
    applied = stats.get("shrink") or {}
    reasoning = int(shrink.get("reasoning") or 0)
    # (one more only when the last one took effect: rebuilt_context caps the count at the turns the window has)
    if reasoning < keep_turns - REBUILT_MIN_REASONING_TURNS and applied.get("reasoning", 0) == reasoning:
        return {**shrink, "reasoning": reasoning + 1}
    if not shrink.get("older") and stats.get("older_turns"):
        return {**shrink, "older": True}
    commits = int(shrink.get("commits") or 0)
    if applied.get("commits", 0) == commits and int(stats.get("commit_turns") or 0) > REBUILT_MIN_COMMIT_TURNS:
        return {**shrink, "commits": commits + 1}
    if not shrink.get("phase_image") and stats.get("images"):
        return {**shrink, "phase_image": True}
    return None


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
    kernel_restarts: int = 0  # cells that timed out or crashed the kernel; the earlier cells were re-run after each
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
    TOOLS = ("python", "run_tests", "commit_engine")  # the tools _dispatch accepts (a subclass adds its own)
    BUILTINS = BUILTIN_FUNCTIONS  # python built-ins the model may call as tools (run as python)

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
        if model.context not in ("compact", "condense", "rebuilt"):
            raise ValueError(f"ModelConfig.context must be 'compact', 'condense' or 'rebuilt', got {model.context!r}")
        self.condense = model.context == "condense"
        # "rebuilt": the conversation is never shortened; every request is rebuilt from it (rebuilt_context) and kept under
        # the budget by its estimate, whose characters-per-token figure is calibrated from every response (seed 3; the
        # ceiling is lowered for the run when the provider rejects a request as too long).
        self.rebuilt = model.context == "rebuilt"
        self.chars_per_token = CHARS_PER_TOKEN_SEED
        self.chars_ceiling = CHARS_PER_TOKEN_MAX
        self._calibrations = 0
        self.context_overflows = 0  # requests the provider rejected as too long (each retried once after a further shrink)
        # The exact count, when a tokenizer is at hand (engine_re.tokens.TokenCounter: the chat template and the model's
        # tokenizer, or the server's /tokenize); the calibrated estimate otherwise.
        self.counter: Any = None
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
        # The python cells that ran in this run, in order ({"turn", "code"}; a resumed run collects them from the
        # transcript): re-run in the kernel after a restart (_restart_kernel), the cell that killed it left out.
        self.cells: list[dict[str, Any]] = []
        # A resume's kernel replay applies the cells' edits to files other than engine.py (a fork: engine_re.tools.fork_run)
        self.replay_files = False

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
        if self.kernel.restarted:
            return _truncate(output) + "\n\n" + self._restart_kernel(code)
        self.cells.append({"turn": self.result.turns, "code": code})
        return _truncate(output)

    def _restart_kernel(self, code: str) -> str:
        """The kernel was killed on `code` (a timeout, a crash) and restarts empty: re-run the earlier cells of this
        run in it (KernelClient.replay: edits and images off, the cell itself left out) and say what was lost (the
        names the kernel held before the cell), which cells came back and what the kernel keeps now (prompts.restart_note)."""
        reason = self.kernel.restarted or "crash"
        lost = self.kernel.last_names
        replay = self.kernel.replay(list(self.cells))
        self.result.kernel_restarts += 1
        self._log({"turn": self.result.turns, "kernel_restart": {
            "reason": reason, "lost": list(lost[0]), "lost_more": lost[1], "cells": len(self.cells),
            "replayed": replay.get("replayed", 0), "failed": replay.get("failed", []), "skipped": replay.get("skipped", 0),
            "seconds": replay.get("seconds", 0.0), **({"error": replay["error"]} if "error" in replay else {}),
        }})
        head = next((line.strip() for line in code.splitlines() if line.strip()), "")[:80]
        return restart_note(reason, self.kernel.timeout, lost, head, replay, kernel_names_text(*self.kernel.names()))

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
        self._tested(report)
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

    def _tested(self, report: Any) -> None:
        """Called with every full or level test's report (the play agent keeps the support maps)."""

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
        if name in self.BUILTINS:  # a python function called as a tool: run it as python
            call = builtin_call_code(name, args)
            return BUILTIN_AS_TOOL.format(name=name, call=call) + self._tool_python(call)
        if name not in self.TOOLS:
            listed = ", ".join(self.TOOLS[:-1]) + " and " + self.TOOLS[-1]
            return f"Error: unknown tool {name!r}. The tools are {listed}."
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
        if self.rebuilt:
            return self._rebuilt_request()
        if not self.condense:
            return self.messages
        from engine_re.condense import hide_but_latest  # (condense imports this module's constants)

        return self._condensed + hide_but_latest(self.messages[self._condensed_from :])

    def _context_budget(self) -> int:
        """Context "rebuilt": the tokens a request may take, the window minus the reply reserve minus the safety margin."""
        reserve = self.model.max_tokens if self.model.reply_reserve is None else int(self.model.reply_reserve)
        return max(1024, int(self.model.context_window) - reserve - REQUEST_SAFETY_TOKENS)

    def _rebuilt_view(self, messages: list[dict[str, Any]] | None = None, extra_steps: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Context "rebuilt": the messages of the next request (rebuilt_context over the conversation, or over `messages`)
        and the composition record: the counts, the characters of each part (and of the system prompt), the estimate
        (estimate_request_tokens with the calibrated chars_per_token, the images at their vision cost), the budget, and
        the shrink steps taken (shrink_step, in order) while the estimate was over the budget, plus `extra_steps` more
        (after a rejected request). A warning when it is still over after every step; nothing is ever truncated."""
        messages = self.messages if messages is None else messages
        tools = self._tools()
        budget = self._context_budget()
        shrink: dict[str, Any] = {}
        forced = 0
        while True:
            view, stats = rebuilt_context(messages, keep_turns=self.model.rebuilt_keep_turns, commit_tools=self._commit_tools(), shrink=shrink)
            estimate = self._count_request(view, tools)
            if estimate["tokens"] + estimate["margin"] <= budget:
                if forced >= extra_steps:
                    break
                forced += 1
            following = shrink_step(shrink, stats, self.model.rebuilt_keep_turns)
            if following is None:
                break
            shrink = following
        system_chars = sum(message_chars(m) for m in view if m["role"] == "system")
        stats["chars"]["system"] = system_chars
        stats["chars"]["total"] = system_chars + sum(stats["chars"][k] for k in ("commits", "phase", "older", "recent"))
        stats.update({k: v for k, v in estimate.items() if k != "tokens"}, estimated_tokens=estimate["tokens"], budget=budget)
        if estimate["tokens"] + estimate["margin"] > budget:
            stats["warning"] = (f"the rebuilt request is {'counted' if estimate['exact'] else 'estimated'} at {estimate['tokens']:,} "
                                f"tokens (+{estimate['margin']} margin), over the budget of {budget:,}, after every shrink step; sent as it is")
        return view, stats

    def _count_request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """The tokens of a request: counted exactly with self.counter (engine_re.tokens.TokenCounter) when there is one,
        with a margin of COUNT_MARGIN_PERCENT (the counter's residual against the reported prompt_tokens, 1.5-2.3%
        under on the v11 run; REQUEST_SAFETY_TOKENS is in the budget already); otherwise estimated at the calibrated
        chars_per_token (estimate_request_tokens), with no margin beyond the clamp's haircut. "exact" says which."""
        if self.counter is not None:
            try:
                counted = self.counter.count(messages, tools)
            except Exception as exc:  # noqa: BLE001  (a counter that fails falls back to the estimate, once noted)
                print(f"[{self.game}] the tokenizer failed ({type(exc).__name__}: {exc}); the estimate is used from here on", flush=True)
                self._log({"turn": self.result.turns, "tokenizer_error": f"{type(exc).__name__}: {exc}"[:300]})
                self.counter = None
            else:
                return {"tokens": counted["tokens"], "text_tokens": counted["text_tokens"], "image_tokens": counted["image_tokens"],
                        "reasoning_tokens": counted.get("reasoning_tokens", 0),
                        "margin": counted["tokens"] * COUNT_MARGIN_PERCENT // 100, "exact": True}
        estimate = estimate_request_tokens(messages, tools, self.chars_per_token)
        return {"tokens": estimate["tokens"], "text_tokens": estimate["text_tokens"], "image_tokens": estimate["image_tokens"],
                "text_chars": estimate["text_chars"], "chars_per_token": round(self.chars_per_token, 3), "margin": 0, "exact": False}

    def _commit_tools(self) -> tuple[str, ...]:
        """The tools whose turns the rebuilt context keeps from before the current phase message."""
        return tuple(t for t in self.TOOLS if t.startswith("commit_"))

    def _rebuilt_request(self, extra_steps: int = 0) -> list[dict[str, Any]]:
        view, stats = self._rebuilt_view(extra_steps=extra_steps)
        self._log({"turn": self.result.turns, "rebuilt": {k: v for k, v in stats.items() if k != "turn"}})
        if stats.get("warning"):
            print(f"[{self.game}] turn {self.result.turns}: warning: {stats['warning']}", flush=True)
        return view

    def _calibrate_from_usage(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, usage: dict[str, Any] | None) -> None:
        """The characters-per-token figure of the next estimate, from the request just served: its text as the estimate
        renders it (render_request, the images subtracted at the same cost the estimate adds) over its prompt_tokens; the
        last measurement, clamped to [CHARS_PER_TOKEN_MIN, chars_ceiling]. A "token_calibration" record when it moves by
        0.05 or more (or the first time)."""
        try:
            prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            return
        if prompt_tokens <= 0 or not messages:
            return
        rendered, image_tokens = render_request(messages, tools)
        text_tokens = prompt_tokens - image_tokens
        if text_tokens <= 0 or not rendered:
            return
        measured = len(rendered) / text_tokens
        clamped = min(self.chars_ceiling, max(CHARS_PER_TOKEN_MIN, measured))
        previous, self.chars_per_token = self.chars_per_token, clamped
        if abs(clamped - previous) >= 0.05 or self._calibrations == 0:
            self._log({"turn": self.result.turns, "token_calibration": {
                "chars_per_token": round(clamped, 4), "measured": round(measured, 4), "ceiling": self.chars_ceiling,
                "prompt_tokens": prompt_tokens, "image_tokens": image_tokens, "text_chars": len(rendered)}})
        self._calibrations += 1

    def _context_overflow(self, error: str) -> None:
        """The provider rejected the request as too long: the divisor's ceiling comes down for the run (to the figure in
        use less a tenth, so every later estimate is higher) and the request is rebuilt with one more shrink step."""
        self.context_overflows += 1
        self.chars_ceiling = max(CHARS_PER_TOKEN_MIN, min(self.chars_ceiling, self.chars_per_token) * 0.9)
        self.chars_per_token = min(self.chars_per_token, self.chars_ceiling)
        self._log({"turn": self.result.turns, "context_overflow": {
            "error": error[:300], "ceiling": round(self.chars_ceiling, 4), "chars_per_token": round(self.chars_per_token, 4)}})
        print(f"[{self.game}] turn {self.result.turns}: the provider rejected the request as too long; retrying with a further "
              f"shrink step and the chars-per-token ceiling at {self.chars_ceiling:.2f}", flush=True)

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
        calibration = next((r["token_calibration"] for r in reversed(self.records) if "token_calibration" in r), None)
        if calibration:  # the last figure of the interrupted run, so the first estimate is not the seed again
            self.chars_ceiling = float(calibration.get("ceiling") or self.chars_ceiling)
            self.chars_per_token = min(self.chars_ceiling, float(calibration.get("chars_per_token") or self.chars_per_token))
            self._calibrations = 1
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

    def _say(self, role: str, content: Any, phase: str | None = None) -> None:
        """Send the model a system or user message (and log it as sent), tagged with the current turn; `phase` marks
        a PLAN or FIT message ("plan" / "fit"), the one the rebuilt context keeps in full."""
        message = TurnMessage({"role": role, "content": content}, turn=self.result.turns, phase=phase)
        self.messages.append(message)
        self._log_message(message)

    def _opening_phase(self) -> str | None:
        """The phase the opening message opens: stepwise, a step to fix ("fit"); the play agent says which."""
        return "fit" if self.stepwise else None

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
        phase: str | None = None  # a "plan" / "step_start" / "advance" record was logged: the next user message opens it
        for r in reversed(records[: starts[-1]]):  # the opening's phase record comes before the system message
            if "message" in r:
                break
            if "plan" in r or "step_start" in r or "advance" in r:
                phase = "plan" if "plan" in r else "fit"
                break
        killed = False  # a "kernel_restart" record: the next python cell is the one that killed the kernel (not re-run)
        for index in range(starts[-1], len(records)):
            r = records[index]
            turn = int(r.get("turn") or 0)
            if "message" in r:
                message = dict(r["message"])
                if isinstance(message.get("content"), list):
                    message["content"] = [
                        self._image_part((self.dir / part["path"]).read_bytes(), self.dir / part["path"])
                        if part.get("type") == "image_file" else part
                        for part in message["content"]
                    ]
                opens = phase if message.get("role") == "user" else None
                if opens:
                    phase = None
                messages.append(TurnMessage(message, turn=turn, phase=opens))
            elif "finish_reason" in r:
                assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
                if r.get("reasoning"):
                    assistant["reasoning"] = r["reasoning"]
                if r.get("tool_calls"):
                    assistant["tool_calls"] = r["tool_calls"]
                calls = iter(r.get("tool_calls") or [])
                last_turn = r.get("turn")
                messages.append(TurnMessage(assistant, turn=turn))
            elif "tool" in r:
                call = next(calls, None)
                messages.append(TurnMessage(
                    {"role": "tool", "tool_call_id": r.get("id") or (call["id"] if call else ""), "content": r["output"]}, turn=turn))
                code = cell_code(call) if call and r["tool"] == "python" else None
                if code is not None and killed:  # the cell that timed out or crashed the kernel
                    killed = False
                elif code is not None and not str(r["output"]).startswith(PYTHON_PAUSED[:40]):  # a paused call never ran
                    cells.append({"turn": r.get("turn"), "code": code})
            elif "kernel_restart" in r:
                killed = True
            elif "append" in r:
                messages[-1]["content"] += r["append"]
            elif "hide_images" in r:
                self._hide_images(messages)
            elif "compact" in r:
                if not self.rebuilt:  # (rebuilt: the conversation is kept in full, a run that compacted included)
                    self._compact()
            elif "condense" in r:
                fired = (index, len(messages))
            elif "step_start" in r:
                focus = r["step_start"]["step"]
                phase = "fit"
            elif "advance" in r:
                focus = r["advance"]["next"]
                phase = "fit"
            elif "resumed" in r:
                focus = r["resumed"]["step"]
            elif "plan" in r:  # the play-and-model agent (engine_re.play_agent): a PLAN message on the last step played
                focus = r["plan"]["step"]
                phase = "plan"
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
        self.cells = list(cells)
        replay = self.kernel.replay(cells, files=self.replay_files)
        self._log({"turn": self.result.turns, "replay": {
            "cells": len(cells), "replayed": replay.get("replayed", 0), "failed": replay.get("failed", []),
            "skipped": replay.get("skipped", 0), "seconds": replay.get("seconds", 0.0), **({"error": replay["error"]} if "error" in replay else {}),
            **({"files": True} if self.replay_files else {}),
        }})
        self._say("user", resume_note(replay, kernel_names_text(*self.kernel.names()), files=self.replay_files))
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
        messages: list[dict[str, Any]] = [TurnMessage({"role": "system", "content": system}, turn=0),
                                          TurnMessage({"role": "user", "content": first}, turn=0, phase="fit")]
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

        turn = 0
        for r in records[begin + 1 :]:
            turn = int(r.get("turn") or turn)
            if "finish_reason" in r:
                prompt_tokens = max(prompt_tokens, (r.get("usage") or {}).get("prompt_tokens") or 0)  # ever over: compacted
                assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
                if r.get("reasoning"):
                    assistant["reasoning"] = r["reasoning"]
                if r.get("tool_calls"):
                    assistant["tool_calls"] = r["tool_calls"]
                    calls = iter(r["tool_calls"])
                last_turn = r.get("turn")
                messages.append(TurnMessage(assistant, turn=turn))
                if not r.get("tool_calls"):
                    messages.append(TurnMessage({"role": "user", "content": CONTINUE}, turn=turn))
            elif "tool" in r:
                call = next(calls, None)
                last_tool = TurnMessage({"role": "tool", "tool_call_id": call["id"] if call else "", "content": r["output"]}, turn=turn)
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
                    messages.append(TurnMessage({"role": "user", "content": pictures(r, IMAGE_NOTE, True)}, turn=turn))
            elif isinstance(r.get("engine_change"), dict) and r["engine_change"].get("version"):
                version = r["engine_change"]["version"]
            elif "advance" in r:
                a = r["advance"]
                focus = a["next"]
                messages.append(TurnMessage({"role": "user", "content": advance_message(
                    self.full_trace, a["fixed"], a["next"], a["report"], self.history, engine_listing())}, turn=turn, phase="fit"))
        if self._drop_unanswered(messages):
            cells = [c for c in cells if c["turn"] != last_turn]
        self.messages = messages
        over = prompt_tokens > self.model.compact_prompt_tokens
        if over and not self.condense and not self.rebuilt:
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
        self._tested(report)
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
        self._say("user", self._opening_content(advance_message(self.full_trace, fixed, k, text, self.history, engine_read, names)), phase="fit")
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
            opening = self._stepwise_start()
            if opening is None:  # the whole recording passes already
                self.result.status = "passed"
                self._final_test()
                self._save_result()
                return self.result
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
            self._say("user", opening, phase=self._opening_phase())
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
                request, tools = self._context(), self._tools()
                try:
                    response = self.client.chat(request, tools)
                except RuntimeError as exc:
                    # Rejected as too long (rebuilt: the count or the estimate was wrong): once more, shrunk one step further.
                    if not self.rebuilt or not is_context_length_error(str(exc)):
                        raise
                    self._context_overflow(str(exc))
                    request = self._rebuilt_request(extra_steps=1)
                    response = self.client.chat(request, tools)
                self.result.provider_errors = len(getattr(self.client, "provider_errors", []))
                self.result.turns += 1
                usage = response.get("usage") or {}
                self.result.usage.add(usage)
                if self.rebuilt:
                    self._calibrate_from_usage(request, tools, usage)
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
                self.messages.append(TurnMessage(assistant, turn=self.result.turns))
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
                    self._say("user", CONTINUE if self.TOOLS == EngineAgent.TOOLS else
                              f"Continue by calling a tool ({', '.join(self.TOOLS[:-1])} or {self.TOOLS[-1]}).")
                    continue
                idle_turns = 0
                self.turns_since_test += 1
                for call in assistant["tool_calls"]:
                    name = call["function"]["name"]
                    counted = "python" if name in self.BUILTINS else name  # a built-in called as a tool runs as python
                    self.result.tool_calls[counted] = self.result.tool_calls.get(counted, 0) + 1
                    t0 = time.time()
                    output = self._dispatch(name, call["function"]["arguments"])
                    self.messages.append(TurnMessage({"role": "tool", "tool_call_id": call["id"], "content": output}, turn=self.result.turns))
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
                elif self._test_nudge_due():
                    self._add_to_last(NUDGE.format(n=self.turns_since_test))
                    self.result.nudges += 1
                    self._log({"turn": self.result.turns, "nudge": self.turns_since_test})
                self._turn_notes()
                # After the tool messages (and the automatic test): the turn's images.
                self._attach_images()
                if not self._end_of_turn():
                    break
                if (usage.get("prompt_tokens") or 0) > self.model.compact_prompt_tokens and not self.rebuilt:
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

    def _tools(self) -> list[dict[str, Any]]:
        return tools(self.images, self.mode, self.history)

    def _test_nudge_due(self) -> bool:
        """Whether this turn ends with the reminder to write and test (NUDGE)."""
        return bool(self.turns_since_test) and self.turns_since_test % TEST_NUDGE_TURNS == 0

    def _turn_notes(self) -> None:
        """A subclass's notes to the turn's last tool output (with _add_to_last), before its images."""

    def _stepwise_start(self) -> str | list[dict[str, Any]] | None:
        """Stepwise: the first message, on the first step that fails; None when the whole recording passes."""
        first = self._replay_all()
        if first is None:
            return None
        self._focus_on(first)
        return self._stepwise_opening()

    def _end_of_turn(self) -> bool:
        """After a turn's tool calls and images: stepwise, an accepted commit moves on (False ends the run when the
        whole recording passes); single mode ends when the tests passed."""
        if self.stepwise and self.commit is not None:  # only a commit moves on
            commit, self.commit = self.commit, None
            return self._advance(commit)
        if self.passed and not self.stepwise:
            self.result.status = "passed"
            return False
        return True

    def _final_test(self) -> None:
        """Authoritative full replay of the final engine.py."""
        try:
            report = replay_test(self.engine_path, self.full_trace, details=3, scratch_root=self.dir, match=self.match)
            self.result.final = report.summary()
            (self.dir / "final_test.txt").write_text(report.text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.result.final = {"error": f"{type(exc).__name__}: {exc}"}
