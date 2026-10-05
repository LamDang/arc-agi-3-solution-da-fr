# How ARC-AGI-3 games behave

The real games are written with a game library; your engine.py models the same
behaviour with `make_level` and `step` on the `Sprite` / `State` classes of the
fixed interface. These are the facts that hold across the games.

| Primitive | Fact |
| --- | --- |
| Screen | Every frame is 64x64 with 16 colours. A game draws onto a smaller logical grid, which is scaled up by a whole number (x1 to x5) and centred, with a border colour around it. HUD elements (a budget bar, lives, progress) are drawn at screen resolution afterwards, often in the border: make them screen sprites. |
| Time | Nothing moves without an input. The real game sometimes animates an action over several frames; only the last frame, the resulting state, is compared. |
| Actions | RESET (handled by the harness), 1 up, 2 down, 3 left, 4 right, 5 interact, 6 click at (x, y), 7 undo. Each game advertises a fixed subset. |
| Levels | Each level starts from its initial layout, also after a RESET, so a move budget refills then. Completing a level is permanent. After a game over only RESET is accepted. |
| Objects | Pixel sprites with a layer and tags. Pixel -1 is transparent; -2 is invisible but solid. Collision is usually pixel-exact, and a blocked move is undone. Objects that disappear are often only hidden. Rotation, mirroring and scaling change the drawn pixels: use rotate_cw, flip_lr, flip_ud and scale_up. |
| Determinism | The same state and action always give the same result. |

## Colours

0 white, 1 light grey, 2 grey, 3 dark grey, 4 darker grey, 5 black, 6 magenta,
7 pink, 8 red, 9 blue, 10 light blue, 11 yellow, 12 orange, 13 maroon, 14 green,
15 purple.
