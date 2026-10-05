"""An alternative condenser for the stepwise agent's conversation: by iteration, not by message age.

The agent's `_compact` shortens the conversation by age alone: once a request went over
`compact_prompt_tokens`, every tool output but the 8 most recent is cut to 200 characters, the
reasoning of all but the 10 most recent turns to its last 1200, and long old tool-call arguments to
300. This module condenses by the structure of a stepwise run instead:

    iteration       from a failing-step user message (the first message, or the "Commit accepted"
                    message after an accepted commit) up to and including the turn of the commit that
                    fixed it (its tool results and image message). The current iteration is the one
                    after the last accepted commit. In runs from before commit_engine, the "advance"
                    record marks the commit point: the harness moved on when the tests passed.
    turn            one assistant message with its tool results and the image message that follows.
    failed command  a tool call whose result is an error of the call itself: a Traceback; a result
                    starting with "Error" (unknown tool, bad arguments, the reserved-name refusal) or
                    "Not committed"; an edit that applied nothing ([E_...]). A run_tests or
                    replay_step whose report shows failing steps is NOT a failed command.
    net change      the unified diff (2 context lines) between engine.py at the iteration's start (the
                    latest engine_change before its first turn) and at its commit (the latest change up
                    to the commit turn), from engine_versions/; capped at DIFF_LINES, then
                    "... N more lines" and the names of the defs changed further down.

`condense(messages, records, versions_dir)` is a pure function of the FULL conversation (every message
as the model was sent it, nothing shortened), the transcript records (turn numbers, advances, commits,
engine changes) and the engine_versions/ folder. It returns the conversation to send:

    1. the system prompt unchanged;
    2. finished iterations older than the last KEEP_ITERATIONS: one user message each: the failing-step
       message's text (no images; an engine.py listing elided), a note, the net diff and the commit
       message (or that the harness moved on when the tests passed);
    3. the last KEEP_ITERATIONS finished iterations: one user message each (a content list): the
       failing-step message in full with its images (the engine.py listing elided unless it is the most
       recent message that has one), a note, then turn by turn every tool call that is not a failed
       command with its result (the automatic test's text included; per-turn images dropped), edits as
       one-liners (the diff covers them), then the net diff, the commit message and the harness's reply;
    4. the current iteration: its failing-step message as is; the turns older than the last KEEP_TURNS
       as real assistant and tool messages with the reasoning removed and the failed commands removed
       (the call and its result; an assistant turn with no call left keeps its text, or goes with its
       "Continue by calling a tool" message when it has none); the last KEEP_TURNS turns untouched;
    5. images: within the current iteration only the latest set stays live (as the agent's
       _hide_images); the failing-step messages of 3 keep theirs;
    6. a safety cap: when the estimate (CHARS_PER_TOKEN, IMAGE_TOKENS per live image) is still over
       CAP_TOKENS, the tool results of the current iteration's stripped turns are cut to
       CAPPED_RESULT_CHARS, oldest first, until it fits; as a last resort the blocks of 3 are reduced to
       the form of 2, oldest first. The result says when the cap fired.

The live agent can use it in place of `_compact`: keep the full conversation in `self.messages` and
send `condense(self.messages, records, versions_dir).messages` every turn. The same inputs give the
same output, so a turn's prompt never depends on what earlier turns were sent.

Judgement calls (the specification left them open): a failed call whose result carries the harness's
appended text (an automatic test report, a nudge, a dropped commit) is kept in the current iteration's
stripped turns with its own output replaced by its first line, so the harness's text survives; an edit
one-liner ends with the automatic test's verdict line; a resume note inside a finished iteration
becomes one line of the record; the "Continue by calling a tool" message is kept when the assistant
text it answered is kept.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine_re.agent import CONTINUE, IMAGE_NOTE, IMAGE_PLACEHOLDER

KEEP_ITERATIONS = 3
KEEP_TURNS = 5
CAP_TOKENS = 140_000
CHARS_PER_TOKEN = 4
IMAGE_TOKENS = 1000
DIFF_LINES = 200
CAPPED_RESULT_CHARS = 600
LISTING_HEAD = "\n\nengine.py now, as read_file() shows it"
LISTING_ELIDED = "[engine.py as it was then: elided]"
RESUME_HEAD = "[harness] The run was interrupted here and has now resumed"
HARNESS_APPEND = "\n\n[harness] "
OLD_RECORD = (
    "[harness] Condensed record of the turns that fixed step {k} (turns {a}-{b}): the net change to engine.py and the "
    "commit; the intermediate analysis is dropped."
)
RECENT_RECORD = (
    "[harness] Condensed record of the turns that fixed step {k} (turns {a}-{b}): reasoning and failed commands dropped; "
    "each successful tool call with its result; then the net change to engine.py and the commit."
)
NO_COMMIT_MESSAGE = "(no commit message: the harness moved on when the tests passed)"
RESUME_LINE = "turn {n}: [the run was interrupted here and resumed; the python kernel restarted]"
FAILED_KEPT = "[failed call: its output is dropped; the harness's text that followed it is kept]"
CAPPED_NOTE = "\n[... result cut to {n} characters by the context cap ({total} chars)]"
_EDIT_FUNCTIONS = ("edit", "undo")


# --- annotation -------------------------------------------------------------------------------


@dataclass
class Note:
    """The annotation of one message of the full conversation."""

    turn: int  # 0 before the first assistant message; then the number of the assistant message it follows
    iteration: int
    kind: str  # system | start | assistant | tool | images | continue | resume | user
    failed: bool = False  # a tool message: its call was a failed command
    call: dict[str, Any] | None = None  # a tool message: the tool call it answers
    edits: list[dict[str, Any]] = field(default_factory=list)  # a tool message: the engine changes its call made


@dataclass
class Iteration:
    index: int
    start: int  # index of its failing-step message
    end: int  # index after its last message
    first_turn: int | None  # its first and last turn present (None: no turn yet)
    last_turn: int | None
    finished: bool
    step: int | None = None  # the step it fixes
    commit: dict[str, Any] | None = None  # the commit record (message, version, engine_sha) when there is one


@dataclass
class Condensed:
    messages: list[dict[str, Any]]
    estimate: int  # estimated tokens of `messages`
    cap: dict[str, Any] | None = None  # when the safety cap fired: what it did


def message_text(message: dict[str, Any]) -> str:
    """The text of a message: its string content, or its first text part."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return next((p.get("text") or "" for p in content if p.get("type") == "text"), "")
    return ""


def own_output(output: str) -> tuple[str, str]:
    """A tool output split into the tool's own text and what the harness appended (an automatic test, a nudge)."""
    i = output.find(HARNESS_APPEND)
    return (output, "") if i < 0 else (output[:i], output[i:])


def failed_command(output: str) -> bool:
    """Whether a tool result reports an error of the call itself (see the module docstring)."""
    own, _ = own_output(output)
    return (
        own.startswith("Error")
        or own.startswith("Not committed")
        or "Traceback (most recent call last)" in own
        or "[E_" in own
    )


def _edits_per_tool(records: list[dict[str, Any]]) -> dict[int, list[list[dict[str, Any]]]]:
    """Per turn, the engine changes logged before each tool result, in the order of the tool results."""
    per_turn: dict[int, list[list[dict[str, Any]]]] = {}
    pending: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        turn = int(r.get("turn") or 0)
        if isinstance(r.get("engine_change"), dict):
            pending.setdefault(turn, []).append(r["engine_change"])
        elif "tool" in r:
            per_turn.setdefault(turn, []).append(pending.pop(turn, []))
    return per_turn


def annotate(messages: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[Note]:
    """One Note per message: its turn, iteration and kind; tool messages with their call and failed flag."""
    advance_turns = {int(r["turn"]) for r in records if "advance" in r}
    edits = _edits_per_tool(records)
    notes: list[Note] = []
    turn = iteration = tools_in_turn = 0
    started = False
    pending: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            notes.append(Note(turn, iteration, "system"))
        elif role == "assistant":
            turn += 1
            tools_in_turn = 0
            pending = list(m.get("tool_calls") or [])
            notes.append(Note(turn, iteration, "assistant"))
        elif role == "tool":
            wanted = m.get("tool_call_id")
            call = next((c for c in pending if c.get("id") == wanted), None) or (pending[0] if pending else None)
            if call in pending:
                pending.remove(call)
            changes = edits.get(turn, [])
            own = changes[tools_in_turn] if tools_in_turn < len(changes) else []
            tools_in_turn += 1
            notes.append(Note(turn, iteration, "tool", failed=failed_command(m.get("content") or ""), call=call, edits=own))
        else:
            text = message_text(m)
            if text.startswith(IMAGE_NOTE):
                kind = "images"
            elif text == CONTINUE:
                kind = "continue"
            elif text.startswith(RESUME_HEAD):
                kind = "resume"
            elif not started:
                kind, started = "start", True
            elif turn in advance_turns or text.startswith("Commit accepted:"):
                kind = "start"
                iteration += 1
            else:
                kind = "user"
            notes.append(Note(turn, iteration, kind))
    return notes


def iterations(notes: list[Note], records: list[dict[str, Any]]) -> list[Iteration]:
    """The iterations of the conversation, in order; the last one is the current (unfinished) one."""
    starts = [i for i, n in enumerate(notes) if n.kind == "start"]
    advances = {int(r["turn"]): r["advance"] for r in records if "advance" in r}
    commits = {int(r["turn"]): r["commit"] for r in records if "commit" in r}
    step_start = next((r["step_start"].get("step") for r in records if "step_start" in r), None)
    out: list[Iteration] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(notes)
        turns = [n.turn for n in notes[start:end] if n.kind == "assistant"]
        it = Iteration(index, start, end, min(turns) if turns else None, max(turns) if turns else None,
                       finished=index + 1 < len(starts))
        if index == 0:
            it.step = step_start
        else:
            previous = advances.get(notes[start].turn)
            it.step = previous.get("next") if previous else None
        if it.finished:
            advance = advances.get(notes[end].turn)
            if advance and it.step is None:
                it.step = advance.get("fixed")
            it.commit = commits.get(notes[end].turn)
        out.append(it)
    return out


# --- the net change of engine.py -----------------------------------------------------------------


def _versions(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(turn, version) of every engine change logged, in order (the harness's opening change included)."""
    out = []
    for r in records:
        change = r.get("engine_change")
        if isinstance(change, dict) and change.get("version"):
            out.append((int(r.get("turn") or 0), int(change["version"])))
    return out


def _version_at(versions: list[tuple[int, int]], turn: int, before: bool, versions_dir: Path) -> int | None:
    """The version in force before `turn` (before=True) or through it."""
    found = [v for t, v in versions if (t < turn if before else t <= turn)]
    if found:
        return found[-1]
    return 1 if (versions_dir / "v0001.py").exists() else None


def _read_version(versions_dir: Path, version: int | None) -> str | None:
    if version is None:
        return None
    path = versions_dir / f"v{version:04d}.py"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _enclosing_defs(lines: list[str], numbers: list[int]) -> list[str]:
    """The top-level def/class names enclosing the given 1-based line numbers, in order, once each (a line at
    column 0 is module-level code: it has none, unless it is the def line itself)."""
    names: list[str] = []
    for n in numbers:
        i = min(n, len(lines)) - 1
        while i >= 0:
            line = lines[i]
            if line and not line[0].isspace():
                m = re.match(r"(?:def|class)\s+(\w+)", line)
                if m and m.group(1) not in names:
                    names.append(m.group(1))
                break
            i -= 1
    return names


def net_diff(old: str | None, new: str | None, old_label: str, new_label: str, max_lines: int = DIFF_LINES) -> str:
    """A unified diff (2 context lines) capped at `max_lines`, then "... N more lines" and the defs changed further down."""
    if old is None or new is None:
        return "(engine.py versions not available for the diff)"
    if old == new:
        return "(none: engine.py is the same at the start and at the commit)"
    old_lines, new_lines = old.splitlines(), new.splitlines()
    lines = list(difflib.unified_diff(old_lines, new_lines, fromfile=old_label, tofile=new_label, n=2, lineterm=""))
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # The defs touched by the cut part: follow the hunk headers to know the line numbers.
    old_no = new_no = 0
    touched_old: list[int] = []
    touched_new: list[int] = []
    for i, line in enumerate(lines):
        header = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if header:
            old_no, new_no = int(header.group(1)), int(header.group(2))
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("+"):
            if i >= max_lines:
                touched_new.append(new_no)
            new_no += 1
        elif line.startswith("-"):
            if i >= max_lines:
                touched_old.append(old_no)
            old_no += 1
        else:
            old_no += 1
            new_no += 1
    names = _enclosing_defs(new_lines, touched_new)
    names += [n for n in _enclosing_defs(old_lines, touched_old) if n not in names]
    tail = f"... {len(lines) - max_lines} more lines"
    if names:
        tail += f" (changed further down: {', '.join(names)})"
    return "\n".join(lines[:max_lines] + [tail])


def iteration_diff(it: Iteration, records: list[dict[str, Any]], versions_dir: Path) -> str:
    """The net change of engine.py over a finished iteration, as text with a heading."""
    versions = _versions(records)
    first_turn = it.first_turn if it.first_turn is not None else (it.last_turn or 0) + 1
    old = _version_at(versions, first_turn, True, versions_dir)
    new = (it.commit or {}).get("version") or _version_at(versions, it.last_turn or first_turn, False, versions_dir)
    if old == new:
        return f"Net change to engine.py: none (version {old} at the start and at the commit)."
    diff = net_diff(_read_version(versions_dir, old), _read_version(versions_dir, new),
                    f"engine.py (version {old}, at the start)", f"engine.py (version {new}, at the commit)")
    return f"Net change to engine.py (version {old} -> {new}):\n{diff}"


# --- the condensed blocks of finished iterations -------------------------------------------------


def _elide_listing(text: str) -> str:
    i = text.find(LISTING_HEAD)
    j = text.find("\n\nFix step", i) if i >= 0 else -1
    if i < 0 or j < 0:
        return text
    return text[:i] + "\n\n" + LISTING_ELIDED + text[j:]


def _call_arguments(call: dict[str, Any] | None) -> tuple[str, str]:
    """A tool call's name and what to show of its arguments: python's code, commit_engine's message, else the JSON."""
    if not call:
        return "?", ""
    name = (call.get("function") or {}).get("name") or "?"
    raw = (call.get("function") or {}).get("arguments") or ""
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return name, raw
    if isinstance(args, dict):
        if name == "python" and isinstance(args.get("code"), str):
            return name, args["code"]
        if name == "commit_engine" and isinstance(args.get("message"), str):
            return name, args["message"]
    return name, json.dumps(args) if args else ""


def _verdict(appended: str) -> str:
    """The automatic test's verdict in the harness's appended text (the acceptance line), when there is one."""
    m = re.search(r"Acceptance test \(the recording replayed in order\): (.*)", appended)
    if m:
        return m.group(1).strip()
    m = re.search(r"Contract tests: (.*)", appended)
    return f"contract tests: {m.group(1).strip()}" if m else ""


def _is_edit(note: Note, output: str) -> bool:
    """A python call that changed engine.py through edit_file()/undo_edit() (or the older edit()/undo())."""
    if not note.edits:
        return False
    own, _ = own_output(output)
    return "engine.py: " in own


def _is_accepted_commit(note: Note, output: str) -> bool:
    name = ((note.call or {}).get("function") or {}).get("name")
    return name == "commit_engine" and own_output(output)[0].startswith("Committed")


def _error_line(own: str) -> str:
    """What a failed call said, in one line: the exception of a Traceback, else the first line."""
    lines = [line for line in own.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1] if "Traceback (most recent call last)" in own else lines[0]


def _edit_line(note: Note, output: str) -> str:
    kinds = {c.get("op") or "edit" for c in note.edits}
    label = "undo_edit" if kinds == {"undo"} else "edit_file"
    summary = "; ".join(str(c.get("summary") or "") for c in note.edits)
    _, appended = own_output(output)
    verdict = _verdict(appended)
    line = f"turn {note.turn}: {label} ({summary})"
    return line + (f" -> tested automatically: {verdict}" if verdict else "")


def turn_entries(messages: list[dict[str, Any]], notes: list[Note], it: Iteration) -> str:
    """The successful tool calls of an iteration with their results, turn by turn; edits as one-liners."""
    parts: list[str] = []
    for i in range(it.start, it.end):
        note, m = notes[i], messages[i]
        if note.kind == "resume":
            parts.append(RESUME_LINE.format(n=note.turn))
        elif note.kind == "tool" and not note.failed:
            output = m.get("content") or ""
            if _is_accepted_commit(note, output):
                continue  # the commit line at the end of the block covers it
            if _is_edit(note, output):
                parts.append(_edit_line(note, output))
            else:
                name, args = _call_arguments(note.call)
                head = f"turn {note.turn}, {name}:"
                parts.append(f"{head}\n{args}\n->\n{output}" if args else f"{head}\n->\n{output}")
    return "\n\n".join(parts)


def _commit_text(messages: list[dict[str, Any]], notes: list[Note], it: Iteration) -> str:
    """The commit message and the harness's reply (the result of the accepted commit_engine call)."""
    for i in range(it.end - 1, it.start - 1, -1):
        note = notes[i]
        if note.kind == "tool" and not note.failed and (note.call or {}).get("function", {}).get("name") == "commit_engine":
            _, message = _call_arguments(note.call)
            reply, _ = own_output(messages[i].get("content") or "")
            return f"commit_engine (turn {note.turn}): {message}\n-> {reply.strip()}"
    if it.commit and it.commit.get("message"):
        return f"commit (turn {it.last_turn}): {it.commit['message']}"
    return NO_COMMIT_MESSAGE


def _span(it: Iteration) -> tuple[str, str]:
    a = it.first_turn if it.first_turn is not None else "?"
    b = it.last_turn if it.last_turn is not None else "?"
    return str(a), str(b)


def old_block(messages: list[dict[str, Any]], notes: list[Note], it: Iteration, records: list[dict[str, Any]],
              versions_dir: Path) -> dict[str, Any]:
    """Rule 2: the failing-step text, the net diff and the commit, as one user message."""
    a, b = _span(it)
    text = _elide_listing(message_text(messages[it.start]))
    record = OLD_RECORD.format(k=it.step if it.step is not None else "?", a=a, b=b)
    body = "\n\n".join([text, record, iteration_diff(it, records, versions_dir), _commit_text(messages, notes, it)])
    return {"role": "user", "content": body}


def recent_block(messages: list[dict[str, Any]], notes: list[Note], it: Iteration, records: list[dict[str, Any]],
                 versions_dir: Path, keep_listing: bool) -> dict[str, Any]:
    """Rule 3: the failing-step message with its images, then the successful calls, the diff and the commit."""
    a, b = _span(it)
    start = messages[it.start]
    text = message_text(start)
    if not keep_listing:
        text = _elide_listing(text)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if isinstance(start.get("content"), list):
        first_text = True
        for part in start["content"]:
            if part.get("type") == "text" and first_text:
                first_text = False  # the message text, already added
                continue
            content.append(copy.deepcopy(part))
    record = RECENT_RECORD.format(k=it.step if it.step is not None else "?", a=a, b=b)
    entries = turn_entries(messages, notes, it)
    body = "\n\n".join(p for p in [record, entries, iteration_diff(it, records, versions_dir), _commit_text(messages, notes, it)] if p)
    content.append({"type": "text", "text": body})
    return {"role": "user", "content": content}


# --- the current iteration --------------------------------------------------------------------------


def _strip_turn(group: list[tuple[dict[str, Any], Note]]) -> list[tuple[dict[str, Any], Note]]:
    """Rule 4 for one turn older than the last KEEP_TURNS: no reasoning, no failed commands."""
    assistant, note = group[0]
    assistant = copy.deepcopy(assistant)
    assistant.pop("reasoning", None)
    kept: list[tuple[dict[str, Any], Note]] = []
    dropped_ids: set[str] = set()
    for m, n in group[1:]:
        if n.kind != "tool" or not n.failed:
            kept.append((m, n))
            continue
        own, appended = own_output(m.get("content") or "")
        if appended:  # the harness's text (an automatic test, a nudge) survives the failed call
            kept.append(({**m, "content": f"{_error_line(own)}\n{FAILED_KEPT}{appended}"}, n))
        else:
            dropped_ids.add((n.call or {}).get("id") or "")
    if dropped_ids:
        assistant["tool_calls"] = [c for c in assistant.get("tool_calls") or [] if c.get("id") not in dropped_ids]
    if not assistant.get("tool_calls"):
        assistant.pop("tool_calls", None)
        if not (assistant.get("content") or "").strip():
            return [(m, n) for m, n in kept if n.kind != "continue"]
    return [(assistant, note)] + kept


def _hide_but_latest(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule 5: only the last message with images keeps them; the others get the agent's placeholder."""
    with_images = [i for i, m in enumerate(messages) if isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"])]
    for i in with_images[:-1]:
        messages[i] = {
            **messages[i],
            "content": [{"type": "text", "text": IMAGE_PLACEHOLDER} if p.get("type") == "image_url" else p
                        for p in messages[i]["content"]],
        }
    return messages


def current_messages(messages: list[dict[str, Any]], notes: list[Note], it: Iteration, keep_turns: int
                     ) -> list[tuple[dict[str, Any], str]]:
    """Rules 4 and 5 for the current iteration: (message, tag) with tag "stripped-tool" on the results the cap may cut."""
    cut = (it.last_turn or 0) - keep_turns  # turns <= cut are stripped
    out: list[tuple[dict[str, Any], str]] = [(messages[it.start], "current")]
    group: list[tuple[dict[str, Any], Note]] = []

    def flush() -> None:
        if not group:
            return
        turn = group[0][1].turn
        stripped = turn <= cut
        for m, n in (_strip_turn(group) if stripped else group):
            out.append((m, "stripped-tool" if stripped and n.kind == "tool" else "current"))
        group.clear()

    for i in range(it.start + 1, it.end):
        m, n = messages[i], notes[i]
        if n.kind == "assistant":
            flush()
            group.append((m, n))
        elif group:
            group.append((m, n))
        else:
            out.append((m, "current"))
    flush()
    hidden = _hide_but_latest([m for m, _ in out])
    return [(m, tag) for m, (_, tag) in zip(hidden, out)]


# --- the whole ------------------------------------------------------------------------------------


def estimate_tokens(messages: list[dict[str, Any]], chars_per_token: float = CHARS_PER_TOKEN,
                    image_tokens: int = IMAGE_TOKENS) -> int:
    chars, images = measure(messages)
    return int(chars / chars_per_token) + images * image_tokens


def measure(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """(characters of text, reasoning and tool-call arguments; live images)."""
    chars = images = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for p in content:
                if p.get("type") == "image_url":
                    images += 1
                else:
                    chars += len(p.get("text") or "")
        chars += len(m.get("reasoning") or "")
        for call in m.get("tool_calls") or []:
            chars += len(json.dumps(call))
    return chars, images


def condense(
    messages: list[dict[str, Any]],
    records: list[dict[str, Any]],
    versions_dir: Path | str,
    *,
    keep_iterations: int = KEEP_ITERATIONS,
    keep_turns: int = KEEP_TURNS,
    cap_tokens: int = CAP_TOKENS,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> Condensed:
    """The condensed conversation (see the module docstring). `messages` is not changed. `chars_per_token` is the
    cap's estimate (4 by default; the providers' counts of these runs are nearer 3 for code-heavy text)."""
    versions_dir = Path(versions_dir)
    notes = annotate(messages, records)
    its = iterations(notes, records)
    if not its:
        return Condensed(list(messages), estimate_tokens(messages))
    finished, current = its[:-1], its[-1]
    old = finished[: len(finished) - keep_iterations] if keep_iterations else finished
    recent = finished[len(old):]
    listing_at = max((it.start for it in its if LISTING_HEAD in message_text(messages[it.start])), default=-1)
    tagged: list[tuple[dict[str, Any], str]] = [(m, "system") for m, n in zip(messages, notes) if n.kind == "system"]
    for it in old:
        tagged.append((old_block(messages, notes, it, records, versions_dir), "old"))
    for it in recent:
        tagged.append((recent_block(messages, notes, it, records, versions_dir, keep_listing=it.start == listing_at), f"recent:{it.index}"))
    tagged += current_messages(messages, notes, current, keep_turns)
    out = [m for m, _ in tagged]
    estimate = estimate_tokens(out, chars_per_token)
    if estimate <= cap_tokens:
        return Condensed(out, estimate)
    cap: dict[str, Any] = {"before": estimate, "results_cut": 0, "blocks_reduced": 0}
    for i, (m, tag) in enumerate(tagged):
        if estimate <= cap_tokens:
            break
        content = m.get("content")
        if tag == "stripped-tool" and isinstance(content, str) and len(content) > CAPPED_RESULT_CHARS:
            out[i] = {**m, "content": content[:CAPPED_RESULT_CHARS] + CAPPED_NOTE.format(n=CAPPED_RESULT_CHARS, total=len(content))}
            cap["results_cut"] += 1
            estimate = estimate_tokens(out, chars_per_token)
    for i, (m, tag) in enumerate(tagged):
        if estimate <= cap_tokens:
            break
        if tag.startswith("recent:"):
            it = its[int(tag.split(":")[1])]
            out[i] = old_block(messages, notes, it, records, versions_dir)
            cap["blocks_reduced"] += 1
            estimate = estimate_tokens(out, chars_per_token)
    cap["after"] = estimate
    return Condensed(out, estimate, cap)
