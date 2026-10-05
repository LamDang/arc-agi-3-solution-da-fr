"""The reverse-engineering agent: an OpenRouter tool-calling loop.

One `EngineAgent` works on one game in its own directory:

    <game_dir>/trace/            the recording (engine_re.trace)
    <game_dir>/workspace/        engine.py and anything the model writes
    <game_dir>/transcript.jsonl  every model turn and tool call
    <game_dir>/tests.jsonl       every run_tests result
    <game_dir>/engine_best.py    the engine with the most exactly-matching steps so far
    <game_dir>/result.json       outcome, tokens, cost, final test

The session stops when a full replay matches every step, when the model calls
finish twice, or when a budget (turns, output tokens, cost, wall time) runs out.

Feedback the harness adds on its own: when engine.py changed during a turn and
was not tested since, a full replay runs automatically and its summary is
appended to the turn's last tool output; after every TEST_NUDGE_TURNS turns
without any test, a reminder to write and test is appended instead.

Sessions survive interruptions: result.json is rewritten every turn with status
"running", and running a game again whose session did not end continues from
its engine.py with a fresh conversation, carrying over the turns, tokens, cost,
time and test history recorded so far.
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

from engine_re.kernel import KernelClient
from engine_re.prompts import SYSTEM_PROMPT, TOOLS, first_user_message, resume_user_message
from engine_re.skeleton import render_skeleton
from engine_re.tester import replay_test
from engine_re.trace import Trace

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TOOL_OUTPUT_CHARS = 8000
# Turns without a run_tests call after which the harness reminds the model to
# write what it knows into engine.py and test it (and again every as many turns).
TEST_NUDGE_TURNS = 30
NUDGE = (
    "\n\n[harness] {n} turns since your last run_tests (or none yet). Put what you have established into engine.py now, "
    "even if partial, and run run_tests: its report shows exactly which step and pixels to fix next."
)
# When engine.py changed during a turn and the model did not test it, the
# harness runs a full replay and appends this summary to the turn's last output.
AUTO_TEST = "\n\n[harness] engine.py changed, so it was tested automatically (full replay):\n{report}"
AUTO_TEST_CHARS = 2500
PYTHON_PAUSED = (
    "[harness] Python is paused: {n} python calls since engine.py last changed. Write what you have established "
    "into engine.py now with write_engine or edit_engine (even partially); python resumes as soon as engine.py "
    "changes. run_tests then shows exactly which step and pixels to fix next."
)
MAX_ENGINE_BYTES = 3_000_000


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
                    if data.get("choices"):
                        return data
                    error = f"no choices: {json.dumps(data)[:500]}"
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
    status: str = "running"  # passed | finished | budget_* | error
    turns: int = 0
    minutes: float = 0.0
    usage: Usage = field(default_factory=Usage)
    tool_calls: dict[str, int] = field(default_factory=dict)
    tests_run: int = 0
    first_pass_turn: int | None = None
    best: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    finish_summary: str | None = None
    error: str | None = None
    trace_steps: int = 0
    resumes: int = 0
    nudges: int = 0
    auto_tests: int = 0
    python_paused: int = 0


class EngineAgent:
    def __init__(self, game: str, game_dir: Path, model: ModelConfig, budget: Budget, client: OpenRouterClient | None = None):
        self.game = game
        self.dir = Path(game_dir).resolve()
        self.trace_dir = self.dir / "trace"
        self.workspace = self.dir / "workspace"
        self.engine_path = self.workspace / "engine.py"
        self.trace = Trace.load(self.trace_dir)
        self.model = model
        self.budget = budget
        self.client = client or OpenRouterClient(model)
        self.kernel = KernelClient(self.workspace, self.trace_dir)
        self.result = AgentResult(game=game, model=model.model, trace_steps=len(self.trace))
        self.messages: list[dict[str, Any]] = []
        self.finish_requests = 0
        self.best_exact = -1
        self.passed = False
        self.prior_minutes = 0.0
        self.prior_notes = ""
        self.started = time.time()
        self.turns_since_test = 0
        self.tested_hash: str | None = None
        self.python_since_change = 0
        self.engine_hash_seen: str | None = None

    # --- tools -----------------------------------------------------------------

    def _tool_python(self, code: str) -> str:
        current = self._engine_hash()
        if current != self.engine_hash_seen:
            self.engine_hash_seen = current
            self.python_since_change = 0
        quota = self.budget.python_quota
        if quota is not None and self.python_since_change >= quota:
            self.result.python_paused += 1
            return PYTHON_PAUSED.format(n=self.python_since_change)
        self.python_since_change += 1
        return _truncate(self.kernel.execute(code))

    def _tool_view_engine(self, start_line: int | None = None, end_line: int | None = None) -> str:
        lines = self.engine_path.read_text(encoding="utf-8").splitlines()
        start = max(1, int(start_line or 1))
        end = min(len(lines), int(end_line or len(lines)))
        if end - start > 400:
            end = start + 400
            note = f"\n[showing lines {start}-{end} of {len(lines)}; ask for a range to see more]"
        else:
            note = f"\n[{len(lines)} lines total]"
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
        return _truncate(body, 14000) + note

    def _tool_write_engine(self, content: str) -> str:
        if len(content.encode()) > MAX_ENGINE_BYTES:
            return f"Refused: engine.py would be {len(content.encode())} bytes (limit {MAX_ENGINE_BYTES})."
        self.engine_path.write_text(content, encoding="utf-8")
        return f"Wrote engine.py ({len(content.splitlines())} lines). {self._syntax_check()}"

    def _tool_edit_engine(self, old_str: str, new_str: str, replace_all: bool = False) -> str:
        text = self.engine_path.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            return "Error: old_str not found in engine.py (it must match exactly, including indentation)."
        if count > 1 and not replace_all:
            return f"Error: old_str occurs {count} times; add surrounding lines to make it unique, or set replace_all."
        text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
        self.engine_path.write_text(text, encoding="utf-8")
        return f"Edited engine.py ({count if replace_all else 1} replacement(s)). {self._syntax_check()}"

    def _syntax_check(self) -> str:
        try:
            compile(self.engine_path.read_text(encoding="utf-8"), "engine.py", "exec")
        except SyntaxError as exc:
            return f"WARNING: engine.py has a syntax error: line {exc.lineno}: {exc.msg}"
        return "Syntax OK."

    def _engine_hash(self) -> str:
        return hashlib.sha256(self.engine_path.read_bytes()).hexdigest()

    def _tool_run_tests(self, from_level: int | None = None, details: int | None = None, auto: bool = False) -> str:
        details = max(1, min(6, int(details or 2)))
        self.turns_since_test = 0
        tested_hash = self._engine_hash()
        try:
            report = replay_test(self.engine_path, self.trace, from_level=from_level, details=details, scratch_root=self.dir)
        except ValueError as exc:
            return f"Error: {exc}"
        if from_level in (None, 0):
            self.tested_hash = tested_hash
        self.result.tests_run += 1
        entry = {"turn": self.result.turns, "time": time.time(), "from_level": from_level, "auto": auto, **report.summary()}
        with (self.dir / "tests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if from_level in (None, 0):
            if report.exact > self.best_exact:
                self.best_exact = report.exact
                shutil.copy(self.engine_path, self.dir / "engine_best.py")
                self.result.best = {"turn": self.result.turns, **report.summary()}
            if report.passed:
                self.passed = True
                if self.result.first_pass_turn is None:
                    self.result.first_pass_turn = self.result.turns
        return _truncate(report.text, 9000)

    def _tool_finish(self, summary: str) -> str:
        self.finish_requests += 1
        self.result.finish_summary = summary
        if self.finish_requests == 1 and not self.passed:
            return (
                "Not every step matches yet (run run_tests to see the current state). If you can still make progress, "
                "keep going; call finish again to end the session anyway."
            )
        return "Session finished."

    def _dispatch(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Error: tool arguments are not valid JSON ({exc}). Send a JSON object."
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool {name!r}."
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
            self.prior_minutes = max(self.prior_minutes, float(record.get("elapsed_min") or 0.0))
        tests = self.dir / "tests.jsonl"
        if tests.exists():
            for line in tests.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.result.tests_run += 1
                if entry.get("from_level") in (None, 0):
                    if entry["exact"] > self.best_exact:
                        self.best_exact = entry["exact"]
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
            if len(content) > 400:
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

    def setup(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not self.engine_path.exists():
            self.engine_path.write_text(render_skeleton(self.game, self.trace[0].available_actions), encoding="utf-8")

    def run(self) -> AgentResult:
        self.setup()
        self.started = time.time()
        engine = self.engine_path.read_text(encoding="utf-8")
        if self._restore():
            report = replay_test(self.engine_path, self.trace, details=2, scratch_root=self.dir)
            opening = resume_user_message(
                self.game, self.trace, self.result.turns, _truncate(report.text, 6000), len(engine.splitlines()), self.prior_notes
            )
        else:
            opening = first_user_message(self.game, self.trace, engine)
        system = SYSTEM_PROMPT
        if self.budget.python_quota is not None:
            system += (
                f"\n\n# Analysis quota\nThe python tool pauses after {self.budget.python_quota} calls without any change to "
                "engine.py, and resumes as soon as engine.py changes. Write what you learn into engine.py as you go."
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
                response = self.client.chat(self.messages, TOOLS)
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
                    self.messages.append(
                        {"role": "user", "content": "Continue by calling a tool (python, view_engine, write_engine, edit_engine, run_tests, or finish)."}
                    )
                    continue
                idle_turns = 0
                finished = False
                self.turns_since_test += 1
                for call in assistant["tool_calls"]:
                    name = call["function"]["name"]
                    self.result.tool_calls[name] = self.result.tool_calls.get(name, 0) + 1
                    t0 = time.time()
                    output = self._dispatch(name, call["function"]["arguments"])
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                    self._log({"turn": self.result.turns, "tool": name, "seconds": round(time.time() - t0, 2), "output": output})
                    if name == "finish" and output == "Session finished.":
                        finished = True
                if self.tested_hash is not None and self._engine_hash() != self.tested_hash:
                    report = self._tool_run_tests(details=1, auto=True)
                    self.messages[-1]["content"] += AUTO_TEST.format(report=_truncate(report, AUTO_TEST_CHARS))
                    self.result.auto_tests += 1
                    self._log({"turn": self.result.turns, "auto_test": report[:AUTO_TEST_CHARS]})
                elif self.turns_since_test and self.turns_since_test % TEST_NUDGE_TURNS == 0:
                    self.messages[-1]["content"] += NUDGE.format(n=self.turns_since_test)
                    self.result.nudges += 1
                    self._log({"turn": self.result.turns, "nudge": self.turns_since_test})
                if self.passed:
                    self.result.status = "passed"
                    break
                if finished:
                    self.result.status = "finished"
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
            report = replay_test(self.engine_path, self.trace, details=3, scratch_root=self.dir)
            self.result.final = report.summary()
            (self.dir / "final_test.txt").write_text(report.text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.result.final = {"error": f"{type(exc).__name__}: {exc}"}
