"""The reverse-engineering agent: an OpenRouter tool-calling loop.

One `EngineAgent` works on one game in its own directory:

    <game_dir>/trace/            the recording (engine_re.trace)
    <game_dir>/workspace/        engine.py and anything the model writes
    <game_dir>/engine_versions/  every version of engine.py (edit/undo), out of the model's reach
    <game_dir>/images/           the pictures sent to the model (test reports, show())
    <game_dir>/transcript.jsonl  every model turn, tool call, engine change and image
    <game_dir>/tests.jsonl       every run_tests result
    <game_dir>/engine_best.py    the best engine tested so far (engine_files.BEST_RULE)
    <game_dir>/result.json       outcome, tokens, cost, final test

Tools: python (a kernel with the recording and read/edit/undo/render/show/try_step/auto_sprites;
it cannot write engine.py except through edit() and undo(), which the harness applies), run_tests
and finish. finish runs the tests: the session ends when every test passes (by finish, run_tests
or the automatic test) or when a budget (turns, output tokens, cost, wall time) runs out.

The opening: before the first turn of a new session the harness plays the first round itself. In the
kernel, auto_sprites(0) makes sprite code for level 0's first frame and one edit() puts it above
make_level, which then returns level_0_sprites(); then it runs the tests. The first message shows
what auto_sprites printed, the test report (with its picture) and engine.py, and sets the first task:
the first failing step, usually step 1. That edit and test are not counted as the model's
(engine_changes, tests_run); tests.jsonl marks the test "auto": "opening".

Feedback the harness adds on its own: when engine.py changed during a turn and was not tested
since, run_tests runs automatically with its defaults (a full replay, reported up to the first
failure) and its report is appended to the turn's last tool output; after every TEST_NUDGE_TURNS
turns without any test, a reminder to write and test is appended instead.

Images (``images=True``, the default): tool messages stay plain strings, so after the turn's tool
messages one extra user message carries the turn's pictures: what show() made, then the latest
test report's picture (the engine's final frame next to the original's, the differing regions
boxed). When a newer such message is added, the images of the older ones are replaced by a short
placeholder. The PNGs are saved under ``<game_dir>/images/`` and the transcript logs their paths.

Sessions survive interruptions: result.json is rewritten every turn with status "running", and
running a game again whose session did not end continues from its engine.py (and its versions)
with a fresh conversation, carrying over the turns, tokens, cost, time and test history.
"""

from __future__ import annotations

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
from engine_re.engine_files import best_key
from engine_re.game_api import fixed_block_lines
from engine_re.kernel import KernelClient
from engine_re.prompts import first_user_message, resume_user_message, system_prompt, tools
from engine_re.skeleton import render_skeleton
from engine_re.tester import MAX_FAILURES, replay_test
from engine_re.trace import Trace

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TOOL_OUTPUT_CHARS = 8000
# Turns without a run_tests call after which the harness reminds the model to
# write what it knows into engine.py and test it (and again every as many turns).
TEST_NUDGE_TURNS = 30
NUDGE = (
    "\n\n[harness] {n} turns since your last run_tests (or none yet). Put what you have established into engine.py now "
    "with edit(), even if partial, and run run_tests: its report shows which step and pixels to fix next."
)
# When engine.py changed during a turn and the model did not test it, the
# harness runs run_tests() with its defaults (full replay, reported up to the
# first failure) and appends the report to the turn's last output.
AUTO_TEST = "\n\n[harness] engine.py changed, so it was tested automatically (run_tests with its defaults):\n{report}"
REPORT_CHARS = 9000  # a test report in a tool output
AUTO_TEST_CHARS = 3500
IMAGE_NOTE = "[harness] The images of this turn, in order:"
TEST_IMAGE_NOTE = (
    "From the latest run_tests report: on the left your engine's final frame, on the right the original game's "
    "(upscaled 8x); the differing regions are boxed and numbered as in the report's text."
)
IMAGE_PLACEHOLDER = "[image omitted; the latest images come later]"
MAX_IMAGES_PER_MESSAGE = 6
MAX_TEST_IMAGES = 3
PYTHON_PAUSED = (
    "[harness] Python is paused: {n} python calls since engine.py last changed. Until engine.py changes, only python "
    "calls that change it with edit() or undo() run; then python resumes. run_tests shows which step and pixels "
    "to fix next."
)
READ_CHARS_IN_MESSAGES = 14000  # engine.py shown in the first message (FIXED block folded)
# The opening, run in the kernel: auto_sprites(0) (its summary printed, not its code), then one edit that
# puts the code above make_level and makes make_level return level_0_sprites(). Filled in by
# EngineAgent._opening_code with the anchors of the starting engine.py.
OPENING_SPLIT = "----- harness: edit -----"
OPENING_CODE = '''\
import contextlib as _harness_contextlib, io as _harness_io
_harness_out = _harness_io.StringIO()
with _harness_contextlib.redirect_stdout(_harness_out):
    _harness_code = auto_sprites(0)
print(_harness_out.getvalue().split("\\n\\n")[0])
print({split!r})
edit(edits=[
    {{"op": "prepend", "pos": {head!r}, "lines": _harness_code.rstrip("\\n").splitlines() + ["", ""]}},
    {{"op": "replace", "pos": {start!r}, "end": {end!r}, "lines": [
        "    # For now every level starts as level 0: add level n (auto_sprites(n)) when the tests reach it.",
        f"    return State(grid={{_harness_code.grid}}, sprites=level_0_sprites())"]}},
])
del _harness_contextlib, _harness_io, _harness_out, _harness_code
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
        self.provider_errors: list[str] = []  # answers that came back with finish_reason "error", retried

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
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
        if self.config.providers:
            payload["provider"] = {"order": list(self.config.providers), "allow_fallbacks": False}
        delay = 2.0
        for attempt in range(10):
            try:
                resp = self.session.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=900,
                )
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if choices and choices[0].get("finish_reason") != "error":
                        return data
                    # The provider failed mid-answer (finish_reason "error"): ask again rather than
                    # hand the model an empty turn.
                    detail = (choices[0].get("error") if choices else None) or data.get("error") or data
                    error = f"{'provider error' if choices else 'no choices'}: {json.dumps(detail)[:500]}"
                    self.provider_errors.append(error)
                elif resp.status_code in (408, 429) or resp.status_code >= 500:
                    error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                else:
                    raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1000]}")
            if attempt == 9:
                raise RuntimeError(f"OpenRouter request failed after retries: {error}")
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)
        raise AssertionError("unreachable")



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
    finish_summary: str | None = None
    finish_calls: int = 0
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
    ):
        if interface != "simple":
            raise ValueError("the agent writes make_level/step engines only (interface='simple')")
        self.game = game
        self.match = match
        self.interface = interface
        self.images = images
        self.opening = opening
        self._in_opening = False
        self.dir = Path(game_dir).resolve()
        self.trace_dir = self.dir / "trace"
        self.workspace = self.dir / "workspace"
        self.engine_path = self.workspace / "engine.py"
        self.trace = Trace.load(self.trace_dir)
        self.model = model
        self.budget = budget
        self.client = client or OpenRouterClient(model)
        self.kernel = KernelClient(self.workspace, self.trace_dir, images=images, log=self._log_engine_change)
        self.result = AgentResult(
            game=game, model=model.model, trace_steps=len(self.trace), match=match, interface=interface, images=images
        )
        self.messages: list[dict[str, Any]] = []
        self.best_key: tuple[int, int] = (-1, -1)
        self.passed = False
        self.prior_minutes = 0.0
        self.prior_notes = ""
        self.started = time.time()
        self.turns_since_test = 0
        self.tested_hash: str | None = None
        self.python_since_change = 0
        self.engine_hash_seen: str | None = None
        # The turn's pictures, sent after its tool messages: (caption, saved PNG path, PNG bytes).
        self.pending_shown: list[tuple[str, Path, bytes]] = []
        self.pending_test: list[tuple[str, Path, bytes]] = []

    # --- tools -----------------------------------------------------------------

    def _tool_python(self, code: str) -> str:
        current = self._engine_hash()
        if current != self.engine_hash_seen:
            self.engine_hash_seen = current
            self.python_since_change = 0
        quota = self.budget.python_quota
        if quota is not None and self.python_since_change >= quota and "edit(" not in code and "undo(" not in code:
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
        return report if isinstance(report, str) else _truncate(report.text, REPORT_CHARS)

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
        if auto != "opening":
            self.result.tests_run += 1
        # "from_level" keeps its earlier meaning for older readers of tests.jsonl, the level the engine
        # started at (null: a fresh engine). A full replay has "level" null and "total" = the trace length.
        # "engine_sha" ties the result to a version of engine.py (undo shows it).
        entry = {
            "turn": self.result.turns, "time": time.time(), "from_level": level or None, "auto": auto,
            "engine_sha": tested_hash, **report.summary(),
        }
        with (self.dir / "tests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._keep_test_images(report, auto)
        if full:
            key = best_key(entry)
            if key > self.best_key:
                self.best_key = key
                shutil.copy(self.engine_path, self.dir / "engine_best.py")
                self.result.best = {"turn": self.result.turns, **report.summary()}
            if report.passed:
                self.passed = True
                if self.result.first_pass_turn is None:
                    self.result.first_pass_turn = self.result.turns
        return report

    def _tool_finish(self, summary: str) -> str:
        """Run the tests; the session ends only when every test passes."""
        self.result.finish_calls += 1
        self.result.finish_summary = summary
        report = self._run_tests(None, 1, auto=False)
        if isinstance(report, str):
            return report
        if report.passed:
            return "Every test passes. Session finished."
        return "Not finished: the tests still fail, so the session goes on. The report:\n\n" + _truncate(report.text, REPORT_CHARS)

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
        """Save what show() made in this python call and queue it for the turn's image message."""
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
            suffix = "_opening" if auto == "opening" else "_auto" if auto else ""
            path = self._save_png(image.png, f"turn{self.result.turns:03d}_step{image.step}{suffix}")
            self.pending_test.append((image.caption, path, image.png))

    def _attach_images(self) -> None:
        """Append one user message with the turn's images after its tool messages, and replace the
        images of earlier such messages by a placeholder, so one set stays in the context."""
        pictures = (self.pending_shown + self.pending_test)[:MAX_IMAGES_PER_MESSAGE] if self.images else []
        self.pending_shown, tests, self.pending_test = [], self.pending_test, []
        if not pictures:
            return
        for message in self.messages:
            if isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "text", "text": IMAGE_PLACEHOLDER} if part.get("type") == "image_url" else part
                    for part in message["content"]
                ]
        content: list[dict[str, Any]] = [{"type": "text", "text": IMAGE_NOTE}]
        for caption, path, png in pictures:
            text = (TEST_IMAGE_NOTE + " " + caption) if any(path == t[1] for t in tests) else caption
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": diff_report.data_url(png)}})
        self.messages.append({"role": "user", "content": content})
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
        """Called by the harness side of edit()/undo() for every change to engine.py. The opening's
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
        if name not in ("python", "run_tests", "finish"):
            return f"Error: unknown tool {name!r}. The tools are python, run_tests and finish."
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
        record["elapsed_min"] = round(self._elapsed_minutes(), 3)
        with (self.dir / "transcript.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _save_result(self) -> None:
        self.result.minutes = round(self._elapsed_minutes(), 2)
        (self.dir / "result.json").write_text(json.dumps(asdict(self.result), indent=2) + "\n", encoding="utf-8")

    def _restore(self) -> bool:
        """Load the turns, tokens, time and tests of an interrupted session."""
        transcript = self.dir / "transcript.jsonl"
        if not transcript.exists():
            return False
        thoughts = []
        for line in transcript.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "finish_reason" in record:
                self.result.turns = max(self.result.turns, record["turn"])
                self.result.usage.add(record.get("usage") or {})
                text = "\n".join(t for t in (record.get("reasoning"), record.get("content")) if t)
                if text.strip():
                    thoughts.append((record["turn"], text.strip()))
            elif "tool" in record:
                self.result.tool_calls[record["tool"]] = self.result.tool_calls.get(record["tool"], 0) + 1
            elif "engine_change" in record and record.get("by") != "harness":
                self.result.engine_changes += 1
            self.prior_minutes = max(self.prior_minutes, float(record.get("elapsed_min") or 0.0))
        tests = self.dir / "tests.jsonl"
        if tests.exists():
            for line in tests.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("auto") != "opening":
                    self.result.tests_run += 1
                if entry.get("level") is None and entry.get("from_level") in (None, 0):
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

    def _compact(self) -> None:
        """Elide old tool outputs, old reasoning and large tool-call arguments to bound the prompt."""
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
        """Play the first round before the first turn: auto_sprites(0) into make_level, then the tests.
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
        for caption, _, png in pictures:
            content.append({"type": "text", "text": TEST_IMAGE_NOTE + " " + caption})
            content.append({"type": "image_url", "image_url": {"url": diff_report.data_url(png)}})
        self.result.image_messages += 1
        self._log({"turn": 0, "images": [str(path.relative_to(self.dir)) for _, path, _ in pictures],
                   "captions": [caption for caption, _, _ in pictures]})
        return content

    def setup(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not self.engine_path.exists():
            self.engine_path.write_text(render_skeleton(self.game, self.trace[0].available_actions), encoding="utf-8")
        self.kernel.editor.sync()  # version 1, or a version for a change made while no session ran

    def run(self) -> AgentResult:
        self.setup()
        self.started = time.time()
        if self._restore():
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
        system = system_prompt(self.match, self.interface, self.images)
        if self.budget.python_quota is not None:
            system += (
                f"\n\n# Analysis quota\nThe python tool pauses after {self.budget.python_quota} calls without any change to "
                "engine.py (only calls that change it with edit() or undo() run), and resumes as soon as engine.py changes."
            )
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": opening}]
        # engine.py as the session starts counts as tested, so any change to it triggers an automatic test.
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
                response = self.client.chat(self.messages, tools(self.images))
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
                    self.messages.append({"role": "user", "content": "Continue by calling a tool (python, run_tests or finish)."})
                    continue
                idle_turns = 0
                self.turns_since_test += 1
                for call in assistant["tool_calls"]:
                    name = call["function"]["name"]
                    self.result.tool_calls[name] = self.result.tool_calls.get(name, 0) + 1
                    t0 = time.time()
                    output = self._dispatch(name, call["function"]["arguments"])
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                    self._log({"turn": self.result.turns, "tool": name, "seconds": round(time.time() - t0, 2), "output": output})
                if not self.passed and self.tested_hash is not None and self._engine_hash() != self.tested_hash:
                    tested = self._run_tests(None, 1, auto=True)
                    report = tested if isinstance(tested, str) else tested.text
                    self.messages[-1]["content"] += AUTO_TEST.format(report=_truncate(report, AUTO_TEST_CHARS))
                    self.result.auto_tests += 1
                    self._log({"turn": self.result.turns, "auto_test": report[:AUTO_TEST_CHARS]})
                elif self.turns_since_test and self.turns_since_test % TEST_NUDGE_TURNS == 0:
                    self.messages[-1]["content"] += NUDGE.format(n=self.turns_since_test)
                    self.result.nudges += 1
                    self._log({"turn": self.result.turns, "nudge": self.turns_since_test})
                # After the tool messages (and the automatic test): the turn's images.
                self._attach_images()
                if self.passed:
                    self.result.status = "passed"
                    break
                if (usage.get("prompt_tokens") or 0) > self.model.compact_prompt_tokens:
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
            report = replay_test(self.engine_path, self.trace, details=3, scratch_root=self.dir, match=self.match)
            self.result.final = report.summary()
            (self.dir / "final_test.txt").write_text(report.text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.result.final = {"error": f"{type(exc).__name__}: {exc}"}
