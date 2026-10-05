"""Write engine_re_level_starts.npz: the start frame of every level of the five verified reference
ports (runs/engine-re/ports/<game>/workspace/engine.py, which reproduce their full recordings) and
each level's true grid, for the auto_sprites and grid-guess tests.

    uv run --no-sync python tests/fixtures/make_engine_re_level_starts.py runs/engine-re/ports

The frame of level n is render(make_level(n)), the same as the recorded start for every level the
recordings reach.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from engine_re import game_api
from engine_re.trace import Trace

GAMES = ("ls20", "ft09", "vc33", "sp80", "lp85")


def main(ports: Path) -> None:
    frames, grids, names = [], [], []
    for game in GAMES:
        module = types.ModuleType(f"port_{game}")
        sys.modules[module.__name__] = module
        source = (ports / game / "workspace" / "engine.py").read_text(encoding="utf-8")
        exec(compile(source, f"{game}/engine.py", "exec", dont_inherit=True), module.__dict__)
        trace = Trace.load(ports / game / "trace")
        starts = trace.level_starts()
        for level in range(trace[0].win_levels):
            state = module.make_level(level)
            frame = game_api.render(state)
            if level in starts:
                assert np.array_equal(frame, trace[starts[level]].last), (game, level)
            scale, _, _ = game_api.geometry(state.grid, state.view.scale)
            frames.append(frame)
            grids.append((state.grid[0], state.grid[1], scale))
            names.append(f"{game}:{level}:{'recorded' if level in starts else 'not recorded'}")
    out = Path(__file__).with_name("engine_re_level_starts.npz")
    np.savez_compressed(out, frames=np.stack(frames).astype(np.int8), grids=np.asarray(grids, np.int16), names=np.asarray(names))
    print(f"wrote {out}: {len(frames)} level starts")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
