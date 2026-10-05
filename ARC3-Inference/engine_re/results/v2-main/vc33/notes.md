
# vc33 reverse engineering notes

## Global
- Frames are 64x64. Row 0 = HUD budget bar: cols 0..(64-n-1)=7 (pink), remaining = 4 (dark grey), n = round(64*k/T).
  T per level: L0=50/51, L1=50, L2=75, L3=96, L4=196, L5=142, L6=223, L7=197. k = actions taken in this level (0-based).
  n4 = number of 4s at the RIGHT end of row 0.
- Level geometry (screen px): L0 32x32@2, L1 32x32@2, L2 32x32@2 (balls 3px -> treat as 64x64@1), L3 ~21x21@3,
  L4 64x64@1, L5 21x21@3, L6 32x32@2, L7 32x32@2.
  DECISION: implement all levels on a 64x64 logical grid @ scale 1 (render everything in screen px).
- Colors: 0 white bg, 3 darkgrey "material/bands", 5 black "bars", 4 = ball body / border, 9 blue buttons,
  1 lightgrey piston, 12 orange piston(active), 11 yellow, 14 green, 15 purple markers/holes.
- available_actions [6] only (click). 7 levels, win at score 7.

## Model (deduced)
Bands = rectangles of colour 3 anchored to a side; their "edge" is a scalar. Bars separate adjacent bands.
Buttons (blue squares) inside a band next to a bar: clicking transfers +/-step from that band to the neighbour across that bar.
Holes = coloured 2-3px marks inside a bar. Balls = arrow sprites sitting at a band's edge; ball tip must align with hole.
Win when every hole has its coloured ball in an adjacent band at the hole's column/row.
Pistons: bar segment colour 1 normally, 12 when the two adjacent band edges are EQUAL; clicking the bar then pumps a ball across (long animation).

## Level 0 (steps 0-6, 7 actions; completes at action k=7 -> step 7)
- band A rows 0-31 (drawn under bar), left-anchored, edge EA; band B rows 28-63 left-anchored edge EB.
- bar rows 28-31: black cols 20-63, yellow hole cols 38-39.
- buttons: A rows 24-27 cols 60-63; B rows 32-35 cols 60-63.
- ball: yellow arrow rows 44-49, tip cols EB-1..EB, body 4s to left.
- EA+EB = 82, step 4, EA in [?]. start EA=31 EB=51. click A btn: EA-4 EB+4. click B btn: EA+4 EB-4.
- win: EB == 39 (ball tip at 38-39).
## Level 1 (steps 7-18)
- right-anchored bands: A rows 0-23 edge LA (3 from LA..31), C rows 24-39, E rows 40-63.
  bars rows 20-23 (bar1) and 40-43 (bar2)?? actual: bar1 screen rows 20-23, bar2 rows 40-43 with green hole cols 28-29.
- buttons at cols 0-3: A rows 16-19, C rows 24-27, C rows 36-39, E rows 44-47.
- step 2 px. LA+LC = 60? observed...
## Level 2 (steps 19-42) - vertical strips
## Level 4 (steps 73-207) - 4 bands A(1-13) B(17-31) C(35-48) D(52-63), bars 14-16,32-34,49-51
- edges in screen px, step 3. A[15,54] B[24,63] C[24,63] D[18,51]. sum=159.
- buttons cols 61-63 rows 11-13(A),17-19(B top),29-31(B bottom),35-37(C top),46-48(C bottom),52-54(D top),61-63(D bottom)
- balls: green rest rows A:4-9 (initial 5-10), B:21-26, C:39-44, D:55-60. yellow D 55-60.
- bar1 rows14-16 black 16-63, yellow hole 20-21, piston 28-39 (12 wide)
- bar2 rows32-34 black 25-63, no hole, piston 40-51
- bar3 rows49-51 black 10-63, green hole 14-15, piston 25-36
- piston colour 12 iff adjacent edges equal; then white "slots" at rows bar_top-2 & bar_bottom+2, cols piston_left+2..+9
- pump animation (44 frames when ball travels 16): f0 slots filled; f1..12 piston -1/frame; f13 hold;
  f14..(13+d) ball +1/frame; then hold; piston returns +1/frame; last frame slots unfilled.


## Level 3 (steps 43-72), screen px
- 5 vertical bands top-anchored row 1: A cols0-11 bottom48, B 15-26 bottom60, C 30-41 bottom54, D 45-53 bottom57, E 57-62 bottom45
- walls 3px wide cols 12-14,27-29,42-44,54-56; black tops 31,22,25,34 -> row 63
- pistons color1: W1 rows43-54, W2 rows34-45
- hole yellow rows29-30 cols42-44 (in W3)
- ball rows43-48 cols3-8 (4+11) in band A
- buttons rows61-63 blue at cols 9-11,15-17,39-41,45-47,51-53,57-59 (W2 has none)
- bg 3
