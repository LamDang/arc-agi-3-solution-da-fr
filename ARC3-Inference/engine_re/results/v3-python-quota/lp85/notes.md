
# lp85 notes
- Screen 64x64, scale 1, grid 64x64 (letter_box=3 dark grey, background=3). Col0 = budget bar (green 14 top-down, black 5 from top growing). Row1 = level indicator: 8 blocks of 4px at cols 10-13,15-18,...,45-48; done levels -> 14, current -> 14+5 pattern, todo -> 5.
  Actual: at step0 all 5; step1 first block 14; step2 first two 14; step3+ first three 14 (levels_completed=2).
- Cell colours used: 1,2,9,10,11,12,15 (+8 red arrows, 14 green arrows, 11 brackets)
- Arrows: red(8)=shift "backward", green(14)=shift "forward" along the cycle.
- Win when every bracketed target cell holds a token of the bracket colour.
- Budget bar: black px = round(used*64/B), B per level = [13,60,80,150,80,80,80,80]; completing action does not charge.
