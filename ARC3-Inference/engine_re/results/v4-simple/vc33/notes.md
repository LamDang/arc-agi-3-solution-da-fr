# vc33 findings

## Screen / geometry
- Screen 64x64, 16 colours.
- Row 0 of the screen is a HUD bar: 64 px wide, colour 7 (pink) for the unused part and
  colour 4 (darker grey) for the used part. It resets to all-pink at the start of every level.
- Levels 0,1,2 (and the static parts of 5,6) are a 32x32 logical grid at scale 2, offset (0,0).
- Level 3 is NOT 2x2 aligned (3-px bars) -> modelled at 64x64 scale 1 in the current engine
  (its exact first frame is embedded).
- Level 4 is a mixture: 3-px aligned maze + 2-px moving pieces -> 64x64 scale 1.

## Core mechanic (levels 0 and 1 confirmed, same family)
- A stack of horizontal "sheets" (colour 3) separated by 2-row "bands" (colour 5, black) that
  carry a static marker (11 yellow in L0, 14 green in L1).
- Each sheet is anchored to one side; its only state is the column of its free edge.
- Blue 2x2 buttons (colour 9) sit at the free edge of a sheet next to a band.
  Clicking a button shrinks that sheet by `step` cells and grows the sheet on the other side of
  the band by `step` cells. If either edge would leave [0,31] the whole move is rejected.
  L0: step=2, sheets A(anchored left, edge 15) rows 0-13, B(edge 25) rows 16-31,
     buttons at grid (30,12)->shrink A/grow B and (30,16)->shrink B/grow A.
  L1: step=2, sheets T(rows 0-9, edge 26) A(rows 12-19, edge 6) B(rows 22-31, edge 4), all
     anchored right; buttons at grid col 0, rows 8,12,18,22 pairing (T,A) and (A,B).
- An arrow sprite is attached to one sheet, its head pixel sitting on the sheet's edge column
  (3x3 cells: head column = edge, body of colour 4 tapering inward).
- Level solved when the arrow head column equals the band marker column.
  L0 goal col 19, L1 goal col 14.
- Animation frames show intermediate positions (arrow slides 1 cell at a time) - only the final
  frame matters.

## The HUD bar (open problem)
- Per click it goes down by 0, 1 or 2 pixels; monotone within a level; resets per level.
- Not predictable from the game state: e.g. L4 134 clicks -> 43 px, L6 85 -> 27 px,
  L0 6 -> 8 px, L3 29 -> 37 px, L5 36 -> 46 px, L2 23 -> 20 px, L1 11 -> 14 px.
  Rates differ ~4x between levels; clicks that do change the board sometimes cost 0 px and
  clicks that change nothing sometimes cost 2 px. No linear model dark=floor(k*64/B) fits.
  => it is a real-time (wall-clock/frame) countdown; the recording cannot reproduce it from the
  action sequence alone. The engine implements "1 pixel per action", the frame-based equivalent.

## Levels not yet modelled
- L2: vertical corridors of grey separated by black walls, blue buttons along grid row 28;
  clicking drains one row of grey from a corridor and moves a 3-cell block into the next one.
- L3: vertical 3-px bars, a 6x6 px "4"-bodied marker with a yellow/green head that slides
  1 px per click; two arrow sprites (yellow/green heads) that change rows.
- L4: 4 horizontal sheets (rows 1-13, 17-31, 35-48, 52-63) + 3 bands (14-16, 32-34, 49-51),
  6 blue buttons at cols 61-63; step = 3 px; two arrows (yellow & green heads) whose rows change.
- L5, L6: not analysed.

## Current match status
- Contract tests pass. L0 and L1 game logic reproduce the recording exactly (only row 0 differs).
- All levels' first frames reproduce exactly.


## Evidence that the HUD bar (and the game) is real-time
- Step 21 of the recording replays the exact same click as step 20 (grid cell (6,28), a button)
  and does NOTHING, while step 20 transferred grey between two tubes. Same state, same action,
  different result -> the outcome depends on something outside the action sequence (an in-progress
  animation / a clock). Also steps 13,14 (L1) and 74,96 (L4) repeat earlier clicks with no effect.
- The bar goes down by 0/1/2 px per click with no relation to the game state, at ~1.3 px/click in
  L0/L1/L3/L5, ~0.9 in L2 and ~0.32 in L4/L6 (the player clicked ~4x faster in those levels).
  No floor(k*64/B) budget model fits (best fit still off by 5-29 steps).
=> the bar is a wall-clock/frame countdown drawn in screen row 0 (pink 7 -> dark grey 4). The
engine implements the frame-based equivalent: one pixel per action, and game_over when it runs out.

## Final status
- Contract tests 5/5. Acceptance 5/332.
- Level 0 and level 1 game logic reproduces the recording exactly (0 differing pixels in the whole
  63x64 play area for every step of those levels); only row 0 (the clock) differs.
- Every level's first frame is reproduced exactly (embedded pixel data); levels 2-6 use that static
  picture because their mechanics (tube/fluid maze in L2, 1-px sliding markers in L3/L4, etc.) are
  not yet modelled.
