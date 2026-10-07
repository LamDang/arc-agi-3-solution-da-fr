"""The repetition hint: positions counted without the border, per level, with a cooldown."""
from __future__ import annotations

import pytest

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent, _repeated_positions


def board(cell: int, bar: int = 0, size: int = 8) -> Frame:
    """A board with one marker inside the play area and a bar value on the border row."""
    grid = [[0] * size for _ in range(size)]
    grid[size // 2][cell] = 1
    grid[0][0] = bar          # border pixel, changes on every move like a step bar
    return tuple(tuple(row) for row in grid)


def history(cells: list[int], level: int = 1) -> list[HistoryEntry]:
    return [HistoryEntry(action="UP", frame=Frame(grid=board(c, bar=i % 7), step=i, level=level))
            for i, c in enumerate(cells)]


def test_border_changes_do_not_make_positions_differ() -> None:
    entries = history([2, 3, 2, 3, 2, 3, 4])
    repeated, current = _repeated_positions(entries, entries[2].frame, visits=3)
    assert repeated == 2          # cells 2 and 3 seen three times each, despite the bar
    assert current


def test_only_the_current_level_counts() -> None:
    entries = history([2, 2, 2], level=1) + history([2, 5], level=2)
    repeated, current = _repeated_positions(entries, entries[-1].frame, visits=3)
    assert (repeated, current) == (0, False)


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> ToolAgent:
    monkeypatch.setenv("ARC3_REPEAT_HINT", "1")
    obj = ToolAgent.__new__(ToolAgent)
    obj._repeat_hint_turns_since = None
    return obj


def test_hint_needs_three_positions_seen_three_times(agent: ToolAgent) -> None:
    entries = history([2, 3, 2, 3, 2, 3])
    assert agent._repeat_hint_lines(entries[-1].frame, entries) == []
    entries = history([2, 3, 4, 2, 3, 4, 2, 3, 4])
    lines = agent._repeat_hint_lines(entries[-1].frame, entries)
    assert lines[0].startswith("Repetition check: The current board position and 2 other position(s)")
    assert lines[-1].startswith("If you are progressing normally")


def test_hint_waits_ten_turns_while_the_repetition_continues(agent: ToolAgent) -> None:
    entries = history([2, 3, 4] * 3)
    shown = [bool(agent._repeat_hint_lines(entries[-1].frame, entries)) for _ in range(21)]
    assert [i for i, s in enumerate(shown) if s] == [0, 10, 20]


def test_hint_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARC3_REPEAT_HINT", raising=False)
    obj = ToolAgent.__new__(ToolAgent)
    obj._repeat_hint_turns_since = None
    entries = history([2, 3, 4] * 3)
    assert obj._repeat_hint_lines(entries[-1].frame, entries) == []
