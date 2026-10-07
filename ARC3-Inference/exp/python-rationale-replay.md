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

## Conclusions

- With `strict: true`, the fields give a readable statement of intent on every call, where
  the reasoning summary covers about a third of responses.
- The next test is a whole run with `ARC3_PYTHON_RATIONALE=1`, to see whether the fields stay
  useful over a game and whether they change the score or the tokens.
