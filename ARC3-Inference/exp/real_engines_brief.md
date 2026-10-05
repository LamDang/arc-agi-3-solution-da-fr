# Real ARC-AGI-3 engine mechanics brief (ls20, ft09, vc33, sp80, lp85)

Engine facts that apply to all five games (arcengine/base_game.py, camera.py, sprites.py):
- `perform_action`: RESET -> `handle_reset()` (with ONLY_RESET_LEVELS=true always `level_reset()` unless state is WIN), then `step()` is ALSO called with action id RESET (one frame, unless the game does something on RESET). If state is GAME_OVER/WIN, any non-RESET action returns 0 frames and changes nothing.
- One frame is rendered after every `step()` call until `complete_action()`. `next_level()` adds exactly one extra frame (the new level, rendered after `on_set_level`); on the last level `next_level()` calls `win()` instead (no extra frame). `levels_completed` (score) increments in `next_level()`.
- `level_reset()` replaces the current level with a fresh clone of the clean level and calls `set_level` -> `on_set_level`. Game-object attributes (not level sprites) persist unless `on_set_level` resets them.
- Render: sprites sorted by layer (stable for ties, list order), pixels < 0 are transparent (both -1 and -2). Collision masks treat only -1 as empty, so -2 pixels are invisible but collidable. Frame = grid scaled by `min(64//w, 64//h)` and centred with letterbox colour, then each RenderableUserDisplay draws on the 64x64 frame. Click mapping: `display_to_grid`: gx = (X - pad_x)//scale, gy = (Y - pad_y)//scale, None if in the letterbox; then + camera.x/y.

---

## ls20 (environment_files/ls20/9607627b/ls20.py)

**1. Camera.** `Camera(width=16, height=16, background=3, letter_box=3, interfaces=[HUD])` (L1780-1786), but every level has `grid_size=(64,64)`, so `set_level` resizes it to 64x64: scale 1, no letterbox, camera never moves. Screen pixel == grid cell.

**2. Levels.** 7 levels (L604-1467). All: StepCounter=42. Differences are layout plus Level.data:
| lvl | goal shape/colour/rot | start shape/colour/rot | StepsDecrement | Fog |
|---|---|---|---|---|
| 0 | 5/9/0 | 5/9/270 | 1 | no |
| 1 | 5/9/270 | 5/9/0 | 2 (default) | no |
| 2 | 5/9/180 | 5/12/0 | 2 | no |
| 3 | 5/9/0 | 4/14/0 | 1 | no |
| 4 | 0/8/180 | 4/12/0 | 2 | no |
| 5 | two goals: [5/9/90, 0/8/180] | 0/14/0 | 1 | no |
| 6 | 0/8/180 | 1/12/0 | 2 | yes |
Shapes (index -> 3x3 sprite, L1769-1774): 0 gngifvjddu, 1 fywfjzkxlm, 2 mkfbgalsbe, 3 nnjhdcanjk, 4 grcpfuizfp, 5 ubspnhafvq. Colour cycle list `[12, 9, 14, 8]`, rotation list `[0, 90, 180, 270]` (L1767-1768).

**3. Actions** `[1,2,3,4]` (L1787). 1=up, 2=down, 3=left, 4=right; the player (tag sfqyzhzkij, 5x5, rows colour 12/12/9/9/9) moves exactly one 5-px cell. Any other action id: no effect, 1 frame. Movement order inside `step` (L1912-2014):
  a. Every track-mover advances one cell (see 7) BEFORE the player's move is checked (L1958).
  b. Target cell = player + 5*dir. `txnfzvzetn` (L1872-1910) scans every level sprite whose TOP-LEFT (x,y) lies in [tx,tx+5)x[ty,ty+5) (position-based, not pixel collision), in sprite-list order:
     - tag `ihdgageizm` (walls, colour 4; several variants with a -2 edge column/row): blocked, stop scanning.
     - tag `rjlbuycveu` (goal, 5x5 colour 5): if key != this goal's requirement -> blocked, HUD box (eqatonpohu at (1,53), 10x10) recoloured all-0, start 5-tick flash. A matching key is NOT blocked.
     - tag `npxgalaybz` (3x3 colour-11 ring, refill): step budget refilled to max, sprite removed (remembered for death-restore), and this move costs no budget.
     - tag `ttfwljgohq` (shape changer): shape=(shape+1)%6, key pixels replaced, 0 recoloured to current colour, rotation kept.
     - tag `soyhouuebz` (colour changer): colour index +1 mod 4, key recoloured.
     - tag `rhsxkxzdjz` (rotation changer): rotation index +1 mod 4 (90 deg clockwise as rendered).
     Changers are passable, trigger on EVERY entry into their cell (not while standing), and are never consumed.
  c. If not blocked, player moves; if blocked, every track-mover undoes this action's advance (L1966-1969).
  d. If the flash counter was set (wrong goal, or level-0 match hint) -> return now: NO budget decrement, no push, no win check.
  e. Budget: unless a refill was taken, `current -= decrement`; exhausted iff the result < 0 (L1509-1512, L1972). So 42 moves with decrement 1 (death on the 43rd) or 21 moves with decrement 2 (death on the 22nd). BLOCKED moves still cost budget.
  f. If not exhausted: first pusher whose bounding box overlaps the player starts a push (see 4) and the action continues as an animation.
  g. Win check `pbznecvnfr` (L2042-2060) even when the budget is exhausted: player.xy == goal.xy and key matches -> goal + its indicator removed, flagged done; a `vfkkzdgxzx` frame (level 5's njpewhmtfd) is hidden. All goals done -> `next_level()`.
  h. If exhausted (and not won): death (see 6).
- Key match (L2039): shape index, colour index and rotation index all equal the goal's. The goal's 3x3 indicator (kvynsvxbpi at goal+(1,1)) shows the required shape in the required colour and rotation.

**4. Multi-frame actions.** Normal move: 1 frame. Winning move: 2 frames (old level, then new level). Flash (wrong goal; level-0 key-matches-goal hint): 6 frames (move frame + 5 ticks; box/hint restored on the 6th). Death: 6 frames, frames 1-5 entirely colour 11 (1x1 sprite scaled 64, layer 3; key sprite hidden; HUD suppressed), frame 6 normal at the start position. Push: 17 frames = move frame + 8 push ticks + 1 snap tick + 6 return ticks + 1 final snap; the player is carried during the 8 push ticks only. Easing (L1608-1647): t=k/dur, e=4t^3 if t<0.5 else 1-(-2t+2)^3/2; per tick d=from+(to-from)*e - sprite.pos + carry; move by int(d) (truncate toward 0); carry=d-int(d) (carry is double-counted; copy it exactly). Observed _r pusher (8,5)->(33,5): x per frame 8,8,9,14,20,28,31,33,33,33,33,29,21,12,8,8,8.
- Pusher (tag gbvqrjtaqo, 5x5 with colour-1 bar on the side opposite its push direction, rest -2; direction from name suffix `_t` up, `_b` down, `_r` right, `_l` left; L1575-1582). It sits 1 px off the cell grid, next to a wall. Distance n: scan cells (pusher.xy + dir) + k*5*dir for k=1..11 for the top-left of a wall OR goal (set built at level start; removed goals still count); n=k-1; no push if n<=0 or none found within 11 (L1595-1606). Pusher and player both move n cells; the player ends in the cell before the obstacle. After the animation `txnfzvzetn` runs on the landing cell (changers/refills apply, blocked flag ignored), with NO win check and NO budget change. If that landing scan triggers the wrong-goal flash, the flash plays during the NEXT action, which then does nothing else.

**5. HUD** (class hbuhvkxlhc, L1492-1552), drawn on the 64x64 frame, skipped entirely during the death flash:
- Fog (only if level Fog): every pixel with euclid((row,col),(player.y+1.5, player.x+1.5)) > 20.0 set to 5; then the key sprite is redrawn at rows 55-60, cols 3-8.
- Budget bar: rows 61-62, x=13..54 (42 cols); col 13+i is 11 if 41-i < current_steps else 3, so it empties from the LEFT. Decrement 2 removes 2 columns per move.
- Lives: rows 61-62, 2x2 squares at x=56,59,62; square b is 8 if lives > b else 3 (the rightmost goes dark first).
- In-level HUD sprites: box eqatonpohu (1,53) 10x10 colour 5; key sprite wgmbtyhvbc at (3,55) scale 2 (6x6), layer 10; hint frame ghizzeqtoh (1,53) colour 0, layer 5, normally invisible.

**6. Win/lose/reset.** Win: all goals done -> next level; level 6 -> WIN. Death when budget exhausted: lives-1 (start 3). If lives reach 0 -> `lose()` (GAME_OVER, 1 frame, no flash, the player stays where it moved). Otherwise the death restores the player to start, key to Start values, done-goal flags cleared, removed refills/goals/indicators re-added, budget full, track-movers to start (dir=down), hint/highlights hidden, all goal frames shown. RESET/level change: `on_set_level` re-clones the level and resets lives=3, full budget and all counters (L1800-1865). Nothing carries across levels.

**7. Hidden state.** Lives counter (shown only as HUD squares); budget count; key indices (shape is visible but the index isn't); done-goal list; flash counters. Track-movers (dboxixicic, L1674-1762): changers whose pixels overlap an invisible `xfmluydglp` track (colour-14 path: lvl4 11x1 at (14,35); lvl5 21x1 at (14,10) and (14,40) plus an 11x11 ring at (19,20); lvl6 1x29 at (54,5)). Each movement action moves them 5 px: try current dir, then dir-1, dir+1, dir+2 (dir 0=down,1=right,2=up,3=left; initial 0); new top-left must be inside the track bbox on a pixel >= 0. Net effect: bounce along a line, circulate around a ring. Walls and goals carry -2 pixels (collidable, invisible). Level 0 only (`vqfjzzkhid`, L2024-2037): a changer that makes the key match an unfinished goal shows the colour-0 hint frame on the HUD box and the colour-0 hoswmpiqkw highlight round the goal for the 6-frame flash, with no budget cost.

**8. Likely RE mistakes.** Blocked moves still cost budget; wrong-goal bumps and refills don't. Winning on the move that exhausts the budget still wins. Pushes run only if the move didn't exhaust the budget, and never trigger the win check. Track-movers advance before the player's collision test, and the advance is undone on a blocked move. The walkable test is top-left-in-cell, not pixel overlap. A RESET or level change restores 3 lives. Decrement is 2 where `StepsDecrement` is absent. The fog centre is offset +1.5, not +2.

---

## ft09 (environment_files/ft09/0d8bbf25/ft09.py)

**1. Camera.** `Camera(0, 0, 16, 16, background=4, letter_box=4, [budget HUD])` (L2306), resized by every level's `grid_size=(32,32)`: scale 2, no letterbox, so each grid cell is 2x2 screen px. Click mapping: grid = (X//2, Y//2). No camera movement.

**2. Levels.** 6 levels (L2040-2266), names THR, hxv, Fmh, oea, INW, DFx. Level.data: `kCv` = click budget, `cwU` = colour cycle, `elp` = click pattern for plain tiles (always the centre-only pattern [[0,0,0],[0,1,0],[0,0,0]]).
| lvl | kCv | cwU (cycle) | tiles |
|---|---|---|---|
| 0 | 32 | [9,8] | 8 Hkx around 1 constraint, plus a left-hand tutorial picture (untagged) and a blinking 15x15 colour-2 frame `UEq` (tag Ycb) at (16,16) |
| 1 | 32 | [9,12] | 13 Hkx, 2 constraints |
| 2 | 96 | [8,12] | 23 Hkx, 4 constraints |
| 3 | 96 | [9,8,12] (3-cycle) | 18 Hkx, 3 constraints |
| 4 | 128 | [14,15] | 27 Hkx + 3 NTi (plus pattern), 8 constraints |
| 5 | 128 | [11,14] | 22 NTi (pattern = self + tile above), 4 constraints (INTANGIBLE) |
Tiles are 3x3 on a 4-cell pitch (1-cell gaps). Colour swatches in the top-right corner (2x2 sprites at x=27..31, y=0..5) show the cycle; they are decoration.
`on_set_level` (L2314-2348): every Hkx tile is recoloured so its colour becomes cwU[0]; every NTi tile has all non-6 pixels set to cwU[0] (the 6s form the pattern dots).

**3. Actions** `[6]` only (L2306). Click step (L2350-2434):
- Grid point = display_to_grid(x,y). Tile lookup: `get_sprite_at(gx,gy,"Hkx")`, else `get_sprite_at(gx,gy,"NTi")`. Any pixel of the 3x3 tile counts. Gaps, constraints and decorations are not tiles.
- Tile click: build pattern P. For Hkx P = `elp` (centre only). For NTi P = centre + every position where the clicked NTi's pixel == 6 (L2399-2406). For each P[j][i]==1, the tile whose area contains (tile.x+(i-1)*4, tile.y+(j-1)*4) (Hkx first, then NTi) advances its colour one step: idx = cwU.index(pixels[1][1]); `color_remap(old, cwU[(idx+1)%len])`. That remaps every pixel of the old colour, so NTi 6-dots are kept. Missing neighbours are ignored.
- Then win check `cgj` (see 6). If solved -> `next_level()`, no budget cost. Else budget `dzy -= 1` (if >0), and `dzy == 0` -> `lose()` (L2428-2434).
- Non-tile click inside the grid: no budget cost, no state change, 1 frame. Exception: level 0, if the click is not on the constraint sprite -> blink animation (see 4).
- Any non-6, non-RESET action id (not offered) skips the click code but still runs the win check and DOES cost 1 budget (L2390-2434).

**4. Multi-frame.** Tile click: 1 frame (2 when it completes a level, 1 on the final WIN). Level-0 empty-space click: 5 frames. `our=4` is set and frame 1 is unchanged. Then each tick does our-=1 and sets every non-negative pixel of UEq to 0 if our is odd, else 2. Over frames 1-5 the frame colour reads 2,0,2,0,2 (L2354-2362). No budget cost, no tile change.

**5. HUD** (class sve, L2272-2298): screen row 63 only (bottom half of grid row 31). Pixel x is 12 if x < round(64*dzy/kCv) else 11. Python banker's rounding: on 128-budget levels dzy=1 gives 0 lit pixels and 127 gives 64. It starts full (64 px) and shrinks from the right by 64/kCv px per costed click.

**6. Win / lose / reset.** Win (`cgj`, L2436-2520): for EVERY constraint sprite (tag bsT, 3x3: centre colour C, 8 border pixels): for each border position (dx,dy), find the tile containing (bsT.x+4dx, bsT.y+4dy) (Hkx then NTi). If no tile, no constraint. If the border pixel == 0 the tile centre must == C; if it is any other value (2, 3, ...) the centre must != C. All satisfied -> next level (level 5 -> WIN). Lose: the click that takes the budget to 0 (the kCv-th costed click) applies its toggle, then GAME_OVER. RESET -> level re-cloned, colours re-normalised, budget full (`olv`), blink cleared. Nothing persists across levels.

**7. Hidden state.** Budget counter (only the bar shows it, quantised). Blink counter `our`. Colour cycles live in level data. In 2-colour levels "!= C" means "the other colour", but in the 3-colour level 3 it means either of the other two. Constraint pixel 3 sits where there is no tile (no effect). Level-5 constraint sprites are INTANGIBLE (irrelevant: `cgj` looks up tiles, not constraints). NTi pattern dots (colour 6) never change colour.

**8. Likely RE mistakes.** The winning click costs no budget. Empty clicks are free, and on level 0 they take 5 frames. Hkx tiles toggle only themselves. Only NTi tiles toggle neighbours, and their pattern comes from their own 6-pixels (level 4: plus shape; level 5: self + above). The neighbour pitch is 4 grid cells. The constraint check uses the tile centre pixel [1][1]. A 3-colour cycle in level 3. Bar rounding is banker's. Clicks map at scale 2 (32x32 grid).

---

## vc33 (environment_files/vc33/5430563c/vc33.py)

**1. Camera.** `Camera(background=3, letter_box=4, interfaces=[HUD])` (L1837-1841), default 64x64, then resized per level: L0/L1 32x32 (scale 2, no letterbox); L2 52x52 (scale 1, 6-px letterbox of colour 4 on all sides); L3-L5 64x64 (scale 1); L6 48x48 (scale 1, 8-px letterbox). Click grid = ((X-off)//scale, (Y-off)//scale); a click in the letterbox maps to None. No camera movement.

**2. Levels.** 7 levels (L1555-1742). Level.data: `StepCounter` (budget) and `Gravity` (list [gx,gy]).
| lvl | grid | budget | Gravity | sprite rotation | content |
|---|---|---|---|---|---|
| 0 | 32 | 50 | [2,0] right | 270 | 2 containers, 2 buttons, 1 object |
| 1 | 32 | 50 | [-2,0] left | 90 | 3 containers, 4 buttons, 1 object |
| 2 | 52 | 75 | [0,2] down | 0 | 5 containers, 8 buttons, 3 objects |
| 3 | 64 | 50 | [0,3] down | 0 | 5 containers, 6 buttons, 2 gates, 1 object (6x6) |
| 4 | 64 | 200 | [3,0] right | 270 | 4 containers, 6 buttons, 3 gates, 2 objects |
| 5 | 64 | 50 | [-3,0] left | 90 | 3 containers, 4 buttons, 2 gates, 1 floor, 1 object |
| 6 | 48 | 200 | [0,-2] up | 180 | 5 containers, 8 buttons, 3 gates, 2 floors, 3 objects |
Sprite roles (by tag): containers `0043…` (solid colour-0 "liquid columns", layer 0); objects `0016…` (3x3 or 6x6 boats, colour 4 plus a stripe of 11/14/15) sitting on a container surface; markers `0010…` (colour stripe plus -2 body) mounted on walls `0025…` (colour 5); buttons `0022…` (colour 9, 2x2 or 3x3, layer 1); gates `0004…` (colour 1, 3x12 or 2x8, layer 3); floors/ceilings `0001…` (colour 5); gate indicators `0007…` (added at runtime).
Terms (gravity axis = "axial", other axis = "lateral"): base = end toward gravity, surface = end away from gravity (`zfcrfmorna`/`hpakcxndwy`, L1876-1886). Positive gravity = gx>0 or gy>0.

**3. Actions** `[6]` only. Every ACTION6 first costs 1 budget (`tuvumryrbp`, L2101), whatever is clicked. The click target is `get_sprite_at(gx,gy)` with no tag: the topmost-layer collidable sprite whose pixel there is not -1 (L2104-2107). Only that one sprite is considered:
- Button (L2108-2109): a fixed (src, dst) pair computed once in `on_set_level` (L1894-1935). A button whose lateral start equals a container's start pumps INTO that container FROM the container ending at button.lat - button.size. A button strictly inside a container's lateral span pumps INTO that container FROM the container starting at button.lat + 2*button.size. In practice buttons come in pairs at the bottom (or side) between two containers, and each button fills the container it sits in. Pump (`xowbpvmzbd`, L2030-2052) happens only if src's axial size > 0 AND dst's surface has not yet reached its limit (`ysoqxdegud`, L2014-2028):
  - If a floor/ceiling sprite (0001) has the same lateral start as dst and lies on its surface side, the limit is the ceiling's base, minus 4 if dst carries an object (minus 6 when gx==-3, level 5).
  - Otherwise the limit is the nearer surface of the walls touching dst's lateral sides (the lower wall top for down gravity).
  Pump effect: src surface moves |g| toward gravity (it shrinks), dst surface moves |g| away from gravity (it grows). Objects whose base is on src's surface (and within its lateral span) move +g; those on dst move -g. Implementation: src loses |g| pixel rows, dst gains |g| colour-0 rows, and positions shift only for non-negative components (L2041-2052). Then gates are recomputed (`wpcgsoumbr`).
- Gate (L2110-2116): if active (both lateral neighbour containers' surfaces == gate base, `ezbubuphlm` L1957-1964) -> starts the swap animation and hides the indicators. Inactive gate: nothing.
- Anything else (container, wall, object, marker, empty): nothing, but the budget is still spent.
- After every non-animating step, and at the end of an animation: win check, then `if current_steps == 0: lose()` (L2119-2124). Non-6 actions cost no budget and run only the checks.

**4. Multi-frame.** Button/empty click: 1 frame (2 if it completes a level). Gate swap (`mwsdltsaxd`, L2058-2092): a queue of movers run one at a time. Each mover moves 1 px per frame (along the axis with the larger remaining |delta|, ties to x), and its arrival costs one extra no-move frame. Queue: [gate slides by its own size: -width if gx>0, +width if gx<0, -height (UP) for vertical gravity of either sign] + [each object on the left container -> centred on the right container: lateral = other.lat + other.latsize//2 - obj.latsize//2, axial unchanged] + [each object on the right -> left] + [gate back]. Frames = 1 + sum(d_i + 1). Observed L3: 1 + 13 + 16 + 13 = 43. No pumping during it.

**5. HUD** (xclqwacrmx, L1792-1826): screen row 0 (drawn over the letterbox in L2/L6). Pixel x is 7 if x < round(64*cur/budget) else 4 (banker's rounding). Full at level start and shrinks from the right with every click. Gate visuals (`vktvbwfvhu`, L1966-1984): after each pump and each swap, every gate is recoloured: 12 if active (and a 7x12 or 6x8 indicator with colour-0 side bars, rotated like the gate, is added at gate.x-2 (rot 0/180) or gate.y-2 (rot 90/270)), else 1. NOT run at level start: initial gates are 1 with no indicator.

**6. Win/lose/reset.** Win (`ielczunthe`, L1937-1955): every object is satisfied. An object is satisfied when some marker contains the object's colour (pixels[-1,-1] of the raw array), the marker's axial top-left coordinate equals the object's axial top-left coordinate, and the wall the marker overlaps touches a lateral side of the container the object sits on. Visually: the boat's colour stripe lines up with the same-colour notch on a wall bordering its column. Win -> next level (L6 -> WIN). Lose: the click that brings the budget to 0 (the budget-th click), unless that same click (including a gate animation it starts) wins. RESET: level re-cloned, budget refilled, animation cleared, mapping rebuilt. Nothing persists.

**7. Hidden state.** The budget counter (the bar is quantised). The button->(src,dst) map is fixed at level start and computed from the initial geometry. Markers' -2 bodies are invisible but they are how the marker->wall link is found (`collides_with`). The gate-active status is only refreshed after pumps and swaps. The animation queue lives in `bnnqyrupir`.

**8. Likely RE mistakes.** Every click costs budget, including no-op ones and clicks on inactive gates. A pump is refused once dst reaches its limit (the click is still spent). Levels are rotated copies of the same mechanics with different gravity (sign and magnitude 2 or 3 change the step size). The ceiling margin is -4 normally but -6 in level 5. For vertical gravity the gate always slides UP during a swap. Each mover's arrival costs an extra frame. Objects keep their axial position when swapped. Initial gate colour ignores activity. Topmost-sprite click resolution: a button drawn over a container wins.

---

## sp80 (environment_files/sp80/589a99af/sp80.py)

**1. Camera.** `Camera(background=12, letter_box=1, interfaces=[budget bar, rotator])` (L527-531), resized per level: L0-L2 16x16 (scale 4, no letterbox), L3-L5 20x20 (scale 3, 2-px letterbox of colour 1). No camera movement. The interfaces run in order: the bar is drawn on row 0, then `ntashmkaxf` returns `np.rot90(frame, k)` with k = dojfslwbg//90 (L439-451). k=2 (180 deg) on L1, L2 and L4, so the WHOLE screen, bar included, is upside down there (bar on row 63, filling from the right).

**2. Levels.** 6 (L274-409). data `steps` (budget) and `dojfslwbg` (screen rotation):
| lvl | grid | steps | rot | sources | cups | other |
|---|---|---|---|---|---|---|
| 0 | 16 | 30 | 0 | 1 | 2 | one 5-wide platform |
| 1 | 16 | 45 | 180 | 1 | 3 | platforms 3,3,5 |
| 2 | 16 | 100 | 180 | 3 | 3 | platforms 4,5,6,6 |
| 3 | 20 | 120 | 0 | 1 + a 7-wide platform with an embedded source pixel | 4 | platforms 4,4,5,5 |
| 4 | 20 | 100 | 180 | 2 | 3 down + 1 rotated 270 on the right | deflector (8,5), floor also on the right wall (19,0) and at x=-1 |
| 5 | 20 | 120 | 0 | 1 + a 5-wide platform with a source pixel | 1 down + 3 sideways | 2 deflectors, a vertical 4-tall platform, floors on both side walls |
Sprites: source `sowlljgtjvn` (1x1 colour 4, at y=0) with a static drop below it; drop `liolfvkveqg` (1x1 colour 6); platforms `plzwjbfyfli*` (colour 8 bars, 9 when selected); deflectors `tuvkdkhdokr-*` (2x2 L, colour 15: `riwynidseun` [[15,-1],[15,15]] opens top-right, `lexuhyxqrsm` [[-1,15],[15,15]] opens top-left); cups `repwkzbkhxl` (3x2 U [[11,-1,11],[11,11,11]], 13 when filled); floor/kill line `waoewejnqzc` (32x1 colour 1); `bodekplurlf16/20` = invisible-interior frame with colour-1 ring just OUTSIDE the grid (at -1 and N). Colours are normalised in `on_set_level` (platforms 8, cups 11, floors 1; L538-566).

**3. Actions** `[1,2,3,4,5,6]`. Under rotation k the inputs are remapped first (`krehtwyvlu` L845-859): for k=2, 1<->2 and 3<->4, and click (x,y) -> (63-x, 63-y). Controls therefore match the screen as seen. In "change" mode (L674-726), EVERY non-RESET action first costs 1 step (`ytqycovhld`, L861-868); at 0 -> `lose()` + complete, and the rest of the step still runs.
- ACTION6: grid point; the first platform-or-deflector (list order: platforms, then deflectors) whose BOUNDING BOX contains it becomes selected. The previously selected one goes back to 8 (or 15), the new one has its non-4 pixels set to 9 and moves to layer 1. A click elsewhere does nothing (still costs a step).
- ACTION1-4 (up/down/left/right, grid space after remap): moves the selected piece 1 cell if (a) new y >= 3; (b) its bbox, expanded by 1 cell on every side, does not touch any cup's bbox (`husluhmboo` L586-598); (c) pixel collision with every other collidable sprite is empty, OR all colliders are platforms/deflectors (overlap with those is allowed; drops, sources, floors, cups and the border frame block). With nothing selected: no-op.
- ACTION5: if 4 failed spills already happened this level -> `lose()` (the 5th spill is fatal). Otherwise -> "spill" mode (`vdwhttyyfq` L631-653): deselect (selection = None); the frontier = all existing drops, each with dir (0,+1); every pixel==4 of a `sowlljgtjvn`-tagged sprite (including source pixels embedded in platforms) spawns a drop below itself if that cell is empty.

**4. Multi-frame (spill).** The ACTION5 frame, then one frame per propagation tick (L752-821). For every frontier drop (pos p, dir d), with s = topmost collidable sprite with a non -1 pixel at p+d:
- None: new drop at p+d, same dir.
- drop: that drop rejoins the frontier with dir d.
- platform: spawn drops at p +/- perpendicular(d) (sideways, same row) where empty, keeping dir d. The flow spreads along the platform's top and falls off both ends.
- cup: if both perpendicular neighbours of p are this cup (the drop sits in the U notch), the cup turns 13 and counts as filled, and the drop ends. Else spread sideways like a platform.
- deflector: if the left/perp- neighbour is the deflector and the right one is empty -> new drop at (p.x+d.y, p.y-d.x) with dir (d.y,-d.x) (falling into the notch of the riwynidseun shape exits RIGHT). If the right one is the deflector and the left empty -> new drop at (p.x-d.y, p.y+d.x) with dir (-d.y, d.x) (exits LEFT). Else spread sideways.
- floor `waoewejnqzc`: it turns 14, the fail flag is set, and the drop ends. Anything else (border frame, source): the drop ends.
When the frontier is empty: one more frame sets "done". Next step: if all cups are filled and no floor was hit -> `next_level()` (+1 frame). Else 6 blink frames (hit floors alternate 1/14 starting at 1; unfilled cups alternate 0/11 for 5 frames, starting at 0), then a reset frame (`lpqbikobah` L655-672): spawned drops removed, colours restored, failed-spill counter +1, and the selection jumps to the piece nearest the origin (min x^2+y^2). Then `lose()` if steps <= 0. Observed L0: a straight fall onto the floor = 22 frames; a 2-cup win = 20 frames.

**5. HUD** (sklgkadoxw L415-436): unrotated row 0. Pixel x is 14 if x < round(64*steps_left/steps) else 0 (banker's rounding). Then the whole frame is rotated by k. Selected piece shown as colour 9. On level start/reset the piece nearest the origin is auto-selected (`gsvsxaspkc` L580-584).

**6. Win/lose/reset.** Win: a spill whose flow fills every cup and touches no floor line. Lose: (a) the action that brings steps to 0 (the steps-th costed action, always fatal even if it is a would-be-winning ACTION5, since the action ends right there); (b) a 5th ACTION5 after 4 failed spills; (c) steps <= 0 after a failed spill's reset. RESET / new level: everything is re-initialised (selection, colours, budget, failed-spill counter). Nothing persists.

**7. Hidden state.** Failed-spill counter (max 4 tolerated, not displayed); selection identity; spawned drops (temporary); mode change/spill. The border frame sprite sits just outside the visible grid, so drops leaving the grid stop silently (no fail).

**8. Likely RE mistakes.** Every action costs a step, including clicks that select nothing, blocked moves and ACTION5. Platforms may overlap each other but must keep a 1-cell gap from cups and stay at y >= 3. Selection is by bounding box. After a failed spill the selection resets to the nearest-to-origin piece, not the previous one. The screen and controls are rotated 180 deg on L1, L2 and L4. Spill length depends on flow geometry (1 frame per cell of propagation front). The fail animation is exactly 6 + 1 frames. Pieces with an embedded source pixel move their source with them.

---

## lp85 (environment_files/lp85/305b61c3/lp85.py)

**1. Camera.** `Camera(width=16, height=16, background=4, letter_box=3, interfaces=[HUD])` (L21346-21352), resized per level `grid_size` (w,h). Scale = min(64//w, 64//h); offsets centre the grid. No movement.
| lvl | grid (w,h) | scale | offset (x,y) | budget | rings (map key: length) | goals |
|---|---|---|---|---|---|---|
| 0 | 32x19 | 2 | (0,13) | 13 | A:20 | 1 goal |
| 1 | 41x41 | 1 | (11,11) | 60 | A:26, B:10, C:10 (4 shared cells) | 2 goal |
| 2 | 39x31 | 1 | (12,16) | 80 | A:16, B:16 | 1 goal + 1 goal-o |
| 3 | 57x57 | 1 | (3,3) | 150 | A:20, B:20 | 1 goal + 1 goal-o |
| 4 | 27x32 | 2 | (5,0) | 80 | A:21, B:5 | 2 goal |
| 5 | 60x64 | 1 | (2,0) | 80 | A-I: 8 each; 1-24: 3 each; 25-27: 2 each (only R buttons, stacked 3 or 8 per spot) | 3 goal |
| 6 | 48x36 | 1 | (8,14) | 80 | A:8, B:3, C:4 (no C buttons), D:4; A and D buttons stacked | 2 goal |
| 7 | 63x63 | 1 | (0,0) | 80 | A:2, B:6, C:7, D:14, E:16, F:15; D/E/F buttons stacked | 3 goal |
Click mapping: grid = ((X-offx)//scale, (Y-offy)//scale); letterbox -> None.

**2. Levels / data.** 8 levels (L1091-1582). Level.data: `StepCounter`, `level_name`. Ring geometry lives in the module dict `izutyjcpih[level_name][key]` (L1583-21205): a small grid whose cells hold 1..N (the cycle order), -1 elsewhere. Cell (cx,cy) is the grid position (3*cx, 3*cy) (`crxpafuiwp=3`). Sprites: tiles (2x2, colours 9/1/10/2/15, tag `tile`), goal (2x2 colour 11, tag `goal`), goal-o (2x2 colour 12, tag `goal-o`), all sitting on ring cells (L7 also has static off-ring tiles); target frames `bghvgbtwcb` (4x4, colour-11 corner brackets) and `fdgmtkfrxl` (4x4, colour-12 corners) at (cell-1, cell-1); buttons (3x4 arrow sprites; colour 14 pointing right = `button_<key>_R`, colour 8 pointing left = `button_<key>_L`; some rotated 270 so 4x3).

**3. Actions** `[6]` only. Step (L21394-21440):
- `pubeyzotzr` collects EVERY sprite whose BOUNDING BOX contains the grid point (transparent -1 pixels count), in level list order. For each one whose tags[0] contains "button": parse `button_<key>_<R|L>` and rotate ring <key>. For every ring cell n at p, the target is n+1 (R; N wraps to 1) or n-1 (L; 1 wraps to N) (`chmfaflqhy` L21265-21286). The sprite moved is the FIRST sprite in the level list whose top-left == (3p.x, 3p.y) exactly (`ttawusezqc`), whatever its type. All (sprite, target) pairs for that ring are collected first, then applied: a simultaneous cyclic shift of everything on the ring by one cell. Stacked buttons (same bbox) fire one after another in list order, each ring computed on the positions left by the previous one, so shared cells cascade.
- If no button was hit: no state change, NO budget cost, no win check, 1 frame.
- If a button was hit: win check first; win -> `next_level()` (no budget cost). Else `current -= 1`; if it reaches 0 -> `lose()` (L21431-21440).

**4. Multi-frame.** None. Every click is 1 frame; a level-completing click is 2 frames (WIN on level 7 is 1).

**5. HUD** (fonypcnqmf, L21289-21336), drawn on the 64x64 screen over whatever is there (usually letterbox; it overlays the grid on row 1 in L4/L5/L7 and on column 0 in L7):
- Budget column: screen column x=0, rows 0..63. Row i is 5 if i < round(64*(max-cur)/max) else 14 (banker's rounding). It starts all 14 and blackens from the top as clicks are spent.
- Level progress: row y=1, eight 4-px segments at x=10+5i..13+5i (i=0..7). Segment i is 14 if i <= current_level_index-1, else 5. So the number of lit segments = levels completed (none on level 0).

**6. Win/lose/reset.** Win (`khartslnwa` L21442-21451): for every `bghvgbtwcb` frame at (fx,fy), some `goal` sprite must cover (fx+1,fy+1); for every `fdgmtkfrxl` frame, some `goal-o` must cover (fx+1,fy+1). In practice each colour-11 / colour-12 token must sit inside a same-coloured bracket frame. Tiles are irrelevant to winning. Lose: the costed (button) click that takes the budget to 0 without winning; with budget B you get B-1 non-winning rotations, and the B-th must win. RESET: level re-cloned (all sprites back on their initial cells), budget full. Nothing persists. The progress segments depend only on the level index.

**7. Hidden state.** Budget counter (the column is quantised: 64 rows over budgets like 13 give uneven steps). Ring topology is invisible: it lives in the side table, not on screen, and tiles merely reveal the occupied cells. Ring cells can be empty, and rings can share cells. `on_set_level` strips duplicate `sys_click` tags from co-located buttons (this affects only valid-action enumeration, not play).

**8. Likely RE mistakes.** A click anywhere in a button's bbox fires it, and overlapping (stacked) buttons fire together, which is how level 5's compound buttons drive 3-8 rings at once. Clicks that miss buttons are free. The winning click is free. Only the first sprite at an exact cell top-left moves. The goal check is "covers (frame+1, frame+1)", not exact equality. Ring direction: R moves pieces toward the next number, L toward the previous. L6 ring C has no button (static).
