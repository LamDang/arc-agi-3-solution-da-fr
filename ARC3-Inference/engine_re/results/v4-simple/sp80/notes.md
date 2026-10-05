
# sp80 findings
- screen 64x64, logical grid 16x16, scale 4 (no offset). Field colour 12 (orange).
- HUD: a bar on internal screen row 0 (drawn over grid row 0): green 14 for remaining budget,
  black 0 for spent. width = round(64*budget/max). Level0 max=30, level1 max=45.
  All actions cost 1 move. budget 0 -> GAME_OVER.
- A pointer at the source column: screen sprite 4 wide x 7 tall at (src*4, 1):
  rows 0-2 colour 4 (grey), rows 3-6 colour 6 (magenta). Level0 src col 9, level1 src col 5.
  It is not collidable; it never moves within a level.
- Floor: grid row 15, colour 1 (light grey), solid.
- Cups: yellow 11 U-shapes 3 wide x 2 tall at rows 13-14. Level0: x=4 and x=10.
  Level1: x=2, x=6, x=10. interior column = x+1. Solid.
- Boards: 1-row sprites. Active = blue 9, inactive = red 8.
  Level0: one board (row 4, cols 3-7, len 5).
  Level1: (row 6, cols 6-10, len5), (row 9, cols 6-8, len3), (row 11, cols 11-13, len3).
- Level 1 is shown rotated 180 (state.view.rotation=180). Input directions are rotated too:
  ACTION1(up on screen) = internal +y in level1, -y in level0.
- Invisible walls confine the boards to rows 3..11 (walls on grid rows 2 and 12).
  Level 0 also has a wall in column 0 (rows 2..12); level 1 does not (a board reaches col 0 at rows 9-11).
- ACTION6 click: if the clicked grid cell holds a board, that board becomes the active one
  (colours swap). Costs a move either way.
- ACTION5 = pour paint from the source column:
  * paint falls from (src,1) down; solid for the paint = boards, cup pixels, floor (walls ignored)
  * when blocked at (x,yy) by sprite b: paint spreads on row yy-1 over [b.x-1, b.x+b.width] (clamped)
    and drips start at (b.x-1, yy) and (b.x+b.width, yy)
  * blocked by the floor -> "wasted"
  * blocked by a cup bottom with x == cup interior column -> that cup is FILLED, branch ends
  * a drip whose start cell is already solid -> JAM -> GAME_OVER
  * after the pour, the ACTIVE board becomes the topmost board touched by the paint
    (a board is touched if a paint cell is orthogonally adjacent to one of its cells)
  * WIN when every cup is filled and nothing wasted.
- Pour is REFUSED (no effect, still costs a move) when the remaining budget < number of boards:
  step 107 (budget 2 after, 3 boards) did nothing although geometry identical to step 102 (budget 6).
- Final frames of pours look unchanged (the paint/fill animation is transient).
