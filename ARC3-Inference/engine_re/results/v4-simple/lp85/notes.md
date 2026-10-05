# lp85 findings

## Screen model
- 64x64, border sprite colour 3 (dark grey) full screen; playfield = colour 4 rect (the grid background), centred,
  size varies per level. Base fill 5 shows at col0 rows eaten by the bar.
- HUD: green(14) column at screen col 0, 64 px tall = budget bar, depletes from the TOP (rows become 5/black).
- HUD: row 1, 8 ticks of 4 px at cols 10-13,15-18,...,45-48 (period 5). colour 5 = not done, 14 = level completed
  (left to right). win_levels=8.
- No rotation/mirroring of the view (state.view default).

## Levels (bg-4 bbox, scale from block size: blocks are 2x2 CELLS)
L0: px cols 0-63 rows 13-50 -> grid 32x19 scale2
L1: px cols 11-51 rows 11-51 -> grid 41x41 scale1
L2: px cols 12-50 rows 16-46 -> grid 39x31 scale1
L3: px cols 3-59  rows 3-59  -> grid 57x57 scale1
L4: px cols 5-58 rows 0-63   -> grid 27x32 scale2
L5: px cols 2-61 rows 0-63   -> grid 30x32 scale?
L6: px cols 8-55 rows 14-49  -> grid 48x36 scale1
L7: px cols 1-62 rows 0-62   -> grid 62x63 scale1
L8: same bbox as L7 (level after the 8th solve; game is WIN at step 119)
Level start steps: {0:0,1:8,2:17,3:36,4:53,5:65,6:96,7:105,8:119}

## Gameplay
- Blocks: 2x2-cell sprites of colours from {1,2,9,10,11,12,15} (not 8/14 which are arrows, not 3/4/5).
- Arrows: red(8) and green(14) chevron sprites. green = rotate ring CW / shift right; red = CCW / shift left.
  Shapes (scale1): left  4rows x 3cols: .XX / XXX / XXX / .XX ; right: mirrored; up: 3rows x 4cols wide-at-bottom;
  down: wide-at-top. At scale2 they are 2x bigger (8x? size 40 px).
- Brackets: 4 small marks (2x2 px at scale2, 1x1 px at scale1) of colour 11 or 12 around one lattice cell =
  GOAL: that cell must hold a block of the bracket's colour. Level solved when all goals satisfied.
- Rings: cycles of cells. Clicking an arrow rotates its whole ring by 1 (new[i]=old[i-1] for CW).
  Rings can overlap (share cells). Ring types seen: rectangle perimeter loops, and plain single-row cycles.
* L0: one ring, perimeter of 7x5 lattice (20 cells). green ▶ at right, red ◀ at left. cost 5 px/click?!
* L1: ring1 = perimeter rows17-44 x cols23-35 (26 cells); ring2 = row 26 (10 cells, cyclic); ring3 = row 35 (10).
* L2: two ovals of 16 cells sharing 2 cells (30 blocks). arrows: 2 pairs at bottom (rows 40-43).
* L3: 4 crosses (9 blocks each = 36). rings: col15+col45 (20 cells) and row15+row45 (20 cells) x2 groups.
  16 arrows (4 per cross: ^v<>).
* L4: scale2, brackets at (6,17) and (6,41) both colour 11.
