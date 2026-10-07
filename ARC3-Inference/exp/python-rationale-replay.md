# The python tool with a description and a reasoning field

## Question

The reasoning summaries OpenAI returns are short and missing for about two thirds of the
responses that reason, whatever the summary setting (30 replayed requests: `auto` 10/30,
`detailed` 9/30, same length). Can the model state its intent in the tool call itself:
`python(description, reasoning, code)`, where `description` says what the code does and
`reasoning` why it helps solve the game? And does it keep its turns coherent when asked to?

## Setup

`ARC3_PYTHON_RATIONALE=1` puts `description` and `reasoning` before `code` in the `python`
tool schema, all required, adds a sentence to the tool description, and changes the system
prompt's tool line to name them. The 30 requests of the summary replay (seed 1: responses
with at least 200 reasoning tokens and prompts under 50K tokens, from
`runs/base-gpt61sol-dfranzen` and `runs/base-gpt61sol-20games`) were sent again with the new
system prompt and tool, through the same adapter, effort `xhigh`, summary `auto`.

## Results

| | schema not strict | `strict: true` |
| --- | --- | --- |
| calls with both fields | 4/30 | 30/30 |
| median description / reasoning | - | 94 / 154 characters |
| median reasoning tokens (logged: 508) | 408 | 402 |
| responses with a reasoning summary | 10/30 | 9/30 |
| cost | $1.57 | $0.34 |

- **Without strict mode the fields are skipped.** The four calls that filled them were three
  step-1 requests and one at step 4. Every replayed request carries the game's earlier,
  code-only calls in its history, and the model follows them. In a real run the history would
  hold the fields from the first call. `strict: true` (OpenAI enforces the schema) filled them
  on all 30.
- **The turns are coherent.** Read one by one, all 30 descriptions match the code, and the
  reasoning names a mechanic or observation from the game state and what the call should
  show (for example bp35 step 12: "The budget still had plenty remaining, so the purple
  ceiling hazard killed the avatar. Stitching the scrolled rooms will let us find a globally
  safe path"). One description promises a check the code does not make (re86 step 10: "stopping
  when its upper arm first reaches it" for a fixed batch of 13 moves).
- **The code is mostly the same.** In most of the 30 the code makes the same calls or the
  same moves as the logged reply, with small differences. Writing the two fields first did
  not change what the model does.
- **It costs little.** About 250 characters per call, some 60 output tokens, against a median
  of 400 reasoning tokens per response.

## Detailed reasoning

The `reasoning` field first asked "why running it helps solve the game now". It now asks for
detailed reasoning: "Include, when relevant: the observations it builds on, what you deduce
from them, the assumptions you are making, and the decision you take." The tool sentence and
the system prompt line say the same. Same 30 requests, strict schema:

| | short reasoning | detailed reasoning |
| --- | --- | --- |
| calls with both fields | 30/30 | 30/30 |
| median description / reasoning | 94 / 154 characters | 98 / 368 characters |
| median reasoning tokens | 402 | 406 |
| responses with a reasoning summary | 9/30 | 12/30 |
| cost | $0.34 | $1.68 |

- The reasoning is 2.4 times longer, and the model's hidden reasoning does not grow.
- Most of the 30 go from an observation to a deduction to a decision (sp80 step 11: "The
  dark center is a pass-through hole, not an anchor: the animation showed fluid emerging
  directly beneath it while the bar moved as a unit ... so a broader search can seek a
  leak-free layout without spending game actions"). About a third name an assumption (sp80
  step 9: "Treating those segments as permanently anchored is only a hypothesis").
- The re86 step 10 mismatch is gone: the description now says it checks whether touching the
  swatch recolors the shape, which the code does after the moves.
- The cost difference is mostly prompt caching between the replays, not the longer field.

The review page with the 30 requests, both versions and the original calls:
https://claude.ai/artifact/WGQxkRPMzZ381jQo4qhHrU

## Conclusions

- With `strict: true`, the fields give a readable statement of intent on every call, where
  the reasoning summary covers about a third of responses.
- The next test is a whole run with `ARC3_PYTHON_RATIONALE=1`, to see whether the fields stay
  useful over a game and whether they change the score or the tokens.
