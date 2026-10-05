# ft09 notes

## Screen / geometry
- 64x64 screen, logical grid 32x32, scale 2 (default), offset 0,0. No rotation/mirror.
- Border colour 5 (invisible, grid fills screen). Background per level: L0 = 5 (black), L1..L5 = 4 (darker grey).
- HUD: row 63 (screen) is a budget bar: 64 px wide, colour 12 (orange); each click consumes 2 px from the RIGHT
  changed to colour 1 (light grey). 32 clicks budget. Refills on level start.
- Legend: stack of 2x2 grid-cell blocks near top-right corner showing the cycle colours of clickable tiles.
  L0: none. L1: [(30,0)=9,(30,2)=12]. L2: [(30,0)=8,(30,2)=12]. L3: [(30,0)=9,(30,2)=8,(30,4)=12].
  L4: [(27,0)=14,(27,2)=15]. L5: [(30,0)=11,(30,2)=14].

## Tiles
- Tiles are 3x3 grid-cell sprites on a lattice (col x0, row y0; neighbours 4 apart).
- "Plain" tile: 3x3 all one colour (clickable). Click cycles colour through legend cycle list.
- "Pattern" tile: center = its colour C (not clickable; clicking it does nothing),
  the 8 outer cells are flags: 0 = neighbour must be C, 2 = neighbour must be the OTHER colour,
  3 (and maybe 6) = ? (level 4/5 use extra symbols).
- WIN condition: every pattern tile's 8 neighbour constraints satisfied (only for neighbour cells that
  contain an actual tile).

## Levels
level starts (step -> level): {0:0,1:4,2:11,3:28,4:52,5:79,6(win):100}; 6 playable levels 0..5.
Clicks per level: L0 4, L1 7, L2 16 (one no-op on pattern tile), L3 23, L4 26, L5 21.

## Level 5 special: clicking a tile toggles it AND the tile directly above it (verified steps 81,84,88,97).
Level 4 steps 67,69,70 change >9 cells -> need to check (probably toggles a row/line).
