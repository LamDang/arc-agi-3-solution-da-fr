# gpt-6.1-sol after history compaction

## Question

With dfranzen's settings the harness drops the oldest half of a game's conversation when the
prompt passes about 115K tokens (`ARC3_CONTEXT_DRAIN_TOKENS=59392` and the history settings in
`params.yaml`), with no summary in its place. Does gpt-6.1-sol struggle to find its rules again
afterwards?

## Data

The two gpt-6.1-sol runs (`runs/base-gpt61sol-dfranzen`, `runs/base-gpt61sol-20games`) have 43
compactions, in 10 of the 25 games (sk48 10, lf52 9, bp35 8, wa30 7, dc22 4, one each in ls20,
g50t, ka59, s5i5, su15). Each takes the prompt from 111-128K tokens to 59-68K: the system
prompt and the most recent turns (sk48 step 56: 138 messages covering game steps 96-224 became
69 covering steps 177-228). Functions the model wrote stay available for the game
(`ARC3_PERSISTENT_FUNCTIONS_SCOPE=game`), and its Python tool still sees the whole game record
(`history`, `transitions`).

Three readers each took about 15 compactions and read the 10 responses before and after,
with earlier turns where needed, and gave each a verdict. Counts over the same windows come
from the request and event logs.

## Results

| verdict | compactions |
| --- | --- |
| no visible effect | 29 |
| recovers what it needs in Python, no wasted game actions | 11 |
| spends game actions re-testing what it knew | 2 (sk48 steps 56 and 74) |
| unclear | 1 (sk48 step 36) |

| 10 responses | before | after |
| --- | --- | --- |
| game actions | 1,208 | 1,216 |
| actions with no effect | 32 | 31 |
| game overs | 1 | 2 |
| levels solved | 15 | 8 |
| output tokens | 356,675 | 330,317 |

- The model's knowledge lives mostly in its retained functions (`route`, `waypoint_search`,
  `dual_move`, `play9`), so the dropped turns hold little it still needs. When it needs an
  older fact it queries `history`/`transitions` (bp35 step 95: it counts which clicked colours
  changed the board over levels 1-7) or prints its own functions' code objects, whose source
  was in the dropped turns (sk48 step 180).
- In 15 compactions a level had just been solved, so the dropped turns were the previous
  level's; none lost a mechanic.
- Levels solved fall from 15 to 8 in the windows, but both are below the whole-game rate of
  0.56 per 10 responses (compaction comes during long levels), and the turn-by-turn reading
  does not attribute the gap to lost rules.
- The two re-probe cases are sk48 on level 5, already stuck before the cut. At step 56 the
  model, judging the red cars unreachable, spent about 120 UP/DOWN actions to run out the
  budget and force a reset, then re-clicked a cell that its own lookup of past clicks showed
  as a no-op. What compaction loses there is the list of probes already tried.

The page with every verdict and four compactions in full (the prompt after compaction and
the 10 responses after): https://claude.ai/artifact/Evv6ogYwgQoqF6TpSPVmiZ

## Conclusions

- Compaction to about 60K tokens does not make gpt-6.1-sol lose its rules in these runs.
  Its code carries the mechanics, and it reads older facts back from the game record.
- The one visible cost is when it is stuck: the record of probes already tried is gone from
  the prompt. A kept note of established rules and tried probes would target that.
