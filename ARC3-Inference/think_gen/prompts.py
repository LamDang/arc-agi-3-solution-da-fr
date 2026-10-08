"""Prompts for thinking generation, synthetic summaries and the calibration
judge. Kept in one place so a prompt change shows up as one diff, and each
output records the prompt version it was made with."""

PROMPT_VERSION = "b1"  # default reconstruct prompt; see RECONSTRUCT_PROMPTS

# Approach B (reconstructor): appended as a user message after the agent's
# context. Flash writes the thinking as its visible answer.
RECONSTRUCT = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

The conversation above is a game-playing agent's run, up to the moment it produced its next output. {history_note}Its thinking for this step is missing: write it.

The agent's next output was:

{call}

{summary_block}

How to write the thinking:
- Write it as your own private thinking at this moment, in first person, in the way you naturally think when you work on this game: plain text, exploratory, with the checks, doubts and corrections a real solver has.
- Start from the newest information above (the latest tool output, frame or message) and get to the decision: what it shows, how that fits or breaks the current understanding of the game, what the options are, which one to take and why, and what the code needs to do.
- Use only what is visible above. The result of this output is not known yet: do not predict it as if it were seen.
- End with the decision that leads to exactly this output. Describe what the code will do; do not paste the code.
- Do not mention a summary, this note, being given the output, or reconstructing anything. Do not refer to "the agent": it is you.
- Answer with the thinking text only: no title, no {open} tags, no preface.
"""

SUMMARY_BLOCK = """\
A summary of the thinking you had at this step (it lists what you thought about; use it, and fill in the reasoning between its points):
<summary>
{summary}
</summary>"""

NO_SUMMARY_BLOCK = """\
There is no summary for this step: the thinking here was short. Keep it brief, a few sentences to a short paragraph, focused on what the newest information shows and why this output is the next step."""


# b2: thinking as working, not as an explanation written afterwards. On the
# qwen3.8-max calibration b1 read as a polished account of the reasoning,
# stating results the real thinking derived step by step.
RECONSTRUCT_B2 = RECONSTRUCT.replace(
    """- Write it as your own private thinking at this moment, in first person, in the way you naturally think when you work on this game: plain text, exploratory, with the checks, doubts and corrections a real solver has.
""",
    """- Write it as your own private thinking at this moment, in first person, the way you think while working, not the way you would explain it afterwards. Work things out in the text: list the cells, positions or values that matter and compute from them, check them against what the frame and tool output show, and say so ("wait", "hmm") when something does not fit, then correct it. Numbers and coordinates in the code should be derived in the thinking, not just stated. Short notes and fragments are fine; no headings, no polished prose, no closing "Decision:" line.
""")

RECONSTRUCT_PROMPTS = {"b1": RECONSTRUCT, "b2": RECONSTRUCT_B2}


# b3: for teachers whose python calls state `reasoning` and `description`
# (ARC3_PYTHON_RATIONALE, runs/gpt61sol-features-25games). The thinking works
# through the stated reasoning step by step and ends planning the described
# move; its length follows the teacher's billed reasoning tokens.
RECONSTRUCT_B3 = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

The conversation above is a game-playing agent's run, up to the moment it produced its next output. Its thinking for this step is missing: write it.

The next output was a `python` call. With the code, the agent stated why it took this step and what the code does:

<reasoning>
{reasoning}
</reasoning>

<description>
{description}
</description>

<code>
{code}
</code>

{length}

How to write the thinking:
- Write it as your own private thinking at this moment, in first person, the way you think while working, not the way you would explain it afterwards. Plain text, no headings.
- Start from the content of the reasoning above and work it out step by step: take its points in order, and for each one go back to what is visible in the conversation (the newest tool output, frame or message first), state the cells, positions, counts or values that matter, and derive the point from them. Where the reasoning states a conclusion, reach it; where it states a rule or hypothesis about the game, check it against the evidence and say what supports it; where something does not fit, say so and correct it.
- Do not add facts, rules, coordinates or numbers that neither the conversation nor the reasoning supports. If a detail cannot be read from the conversation, leave it out rather than guess.
- End by planning the next move, as the description says: what the code will do, which actions it will send, and what you expect to learn from its result. Describe the code; do not paste it.
- The result of this output is not known yet: do not predict it as if it were seen.
- Do not mention the reasoning, the description, this note, or being given the output. Do not refer to "the agent": it is you.
- Answer with the thinking text only: no title, no {open} tags, no preface.
"""

# 0 reasoning tokens: the teacher answered without thinking; the stated
# reasoning still has to be reached, so the thinking is short, not empty.
LENGTH_NONE = "At this step the thinking was very short: write a few sentences, about as long as the reasoning above."
LENGTH_TOKENS = "At this step the thinking was about {words} words long: match that length; a short step stays short, a long one is worked out in full."

RECONSTRUCT_PROMPTS["b3"] = RECONSTRUCT_B3


# b4: the thinking is the full working behind the stated reasoning, which is
# its short version: each point traced back through the earlier turns that
# established it. On the b3 dev20 run, short steps came out about as long as
# the stated reasoning and mostly restated it.
RECONSTRUCT_B4 = RECONSTRUCT_B3.replace(
    """- Start from the content of the reasoning above and work it out step by step: take its points in order, and for each one go back to what is visible in the conversation (the newest tool output, frame or message first), state the cells, positions, counts or values that matter, and derive the point from them. Where the reasoning states a conclusion, reach it; where it states a rule or hypothesis about the game, check it against the evidence and say what supports it; where something does not fit, say so and correct it.
""",
    """- The reasoning above is the short version of this thinking; the thinking is the full working behind it, so it is much more detailed. Start from the content of the reasoning and work it out step by step, taking its points in order.
- For each point, trace it back through the conversation: find the earlier tool outputs, frames, images and your own earlier calls that established it, and walk through that evidence in order up to the newest output. State the cells, positions, counts or values that matter, and derive the point from them, as you would at the time. Where the reasoning states a conclusion, reach it; where it states a rule or hypothesis about the game, check it against that evidence and say what supports it and what would contradict it; where something does not fit, say so and correct it.
""").replace("{length}", "{length} Never write less than {min_words} words: the reasoning above alone is about {reason_words}.")

RECONSTRUCT_PROMPTS["b4"] = RECONSTRUCT_B4
MIN_LENGTH_FACTOR = 3  # b4: thinking at least this many times the stated reasoning


def reconstruct_prompt_b4(reply: dict, reasoning_tokens: int) -> tuple[str, int]:
    """The b4 prompt and its target length in words: the larger of the
    teacher's reasoning (0.75 words per token) and MIN_LENGTH_FACTOR times
    the stated reasoning."""
    from .context import THINK_OPEN
    args = python_args(reply)
    reasoning = (args.get("reasoning") or "").strip()
    reason_words = len(reasoning.split())
    min_words = int(round(MIN_LENGTH_FACTOR * reason_words, -1))
    words = max(min_words, int(round(reasoning_tokens * 0.75, -1)))
    prompt = RECONSTRUCT_B4.format(
        reasoning=reasoning, description=(args.get("description") or "").strip(),
        code=(args.get("code") or "").rstrip(), length=f"Write about {words} words.",
        min_words=min_words, reason_words=reason_words, open=THINK_OPEN)
    return prompt, words


def python_args(reply: dict) -> dict:
    """The first python call's arguments, {} when there is none."""
    import json
    for c in reply.get("tool_calls") or []:
        fn = c.get("function") or {}
        if fn.get("name") == "python":
            try:
                return json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                return {}
    return {}


def reconstruct_prompt_b3(reply: dict, reasoning_tokens: int) -> str:
    from .context import THINK_OPEN
    args = python_args(reply)
    # ~0.75 words per token, rounded to tens
    length = (LENGTH_TOKENS.format(words=max(10, int(round(reasoning_tokens * 0.75, -1))))
              if reasoning_tokens else LENGTH_NONE)
    return RECONSTRUCT_B3.format(
        reasoning=(args.get("reasoning") or "").strip(),
        description=(args.get("description") or "").strip(),
        code=(args.get("code") or "").rstrip(), length=length, open=THINK_OPEN)


def reconstruct_prompt(call: str, summary: str, version: str = PROMPT_VERSION,
                       history_mode: str = "native") -> str:
    from .context import THINK_CLOSE, THINK_OPEN
    block = SUMMARY_BLOCK.format(summary=summary.strip()) if summary.strip() else NO_SUMMARY_BLOCK
    note = (f"Its thinking for earlier turns is shown between {THINK_OPEN} and {THINK_CLOSE} lines. "
            if history_mode == "inline" else "")
    return RECONSTRUCT_PROMPTS[version].format(
        open=THINK_OPEN, close=THINK_CLOSE, call=call, summary_block=block, history_note=note)


# Calibration: turn a teacher's real thinking into a summary in the style
# of the OpenAI reasoning summaries the real teacher returns.
SYNTH_SUMMARY = """\
Below is the private thinking of an agent playing a grid puzzle game. Write the short summary of it that a reasoning-summary model would show the user, in exactly the style of these real examples:

{examples}

Rules: one section (rarely two): a bold title line (**Like this**) followed by a short paragraph in first person present continuous ("I'm examining...", "I'm considering..."). About {words} words in total, however long the thinking is. Summarize what was observed, considered and decided at a high level; no code, few exact coordinates. Answer with the summary only.

The thinking:
<thinking>
{thinking}
</thinking>"""


# Real gpt-6.1-sol summaries are one section of about 450 characters (75
# words) whatever the reasoning length (correlation 0.08 over 61 summaries
# of runs/base-gpt61sol-dfranzen); the first calibration's summaries, asked
# for 1-3 sections of 90 words, came out at a median of 922 characters.
def synth_summary_prompt(thinking: str, examples: list[str], words: int = 75) -> str:
    ex = "\n\n".join(f"<example>\n{e.strip()}\n</example>" for e in examples)
    return SYNTH_SUMMARY.format(examples=ex, words=words, thinking=thinking.strip())


# Calibration judge: real thinking (R) against generated thinking (G).
JUDGE = """\
You compare two versions of an agent's private thinking at one step of a grid puzzle game. REAL is what the agent actually thought. GENERATED was written afterwards by another model that saw the same context and the agent's output, but not REAL. Both end in this output:

<output>
{call}
</output>

<real>
{real}
</real>

<generated>
{generated}
</generated>

1. List the key points of REAL (observations about the board or tool output, hypotheses about the game's rules or goal, plans and decisions), at most 10, most important first.
2. For each, say whether GENERATED contains it (same content, any wording).
3. List claims in GENERATED that contradict REAL or the output (wrong observations, wrong rules, a different reason for the decision). Omit claims REAL simply does not mention.
4. Does GENERATED lead to this exact output?
5. Does GENERATED leak that it was written afterwards (mentions a summary, a given output, "the agent", or knows the output's result)?

Answer with JSON only:
{{"real_points": ["..."], "covered": [true, false], "contradictions": ["..."], "leads_to_output": true, "leak": false, "notes": "one sentence on the main difference"}}"""


def judge_prompt(call: str, real: str, generated: str) -> str:
    return JUDGE.format(call=call, real=real.strip(), generated=generated.strip())


# Sol judge: four checks on generated thinking where the teacher's real
# thinking is hidden (gpt-6.1-sol), so there is nothing to diff against.
# Each check uses something that IS known about the step:
#   1 code     the python code the teacher ran;
#   2 words    the teacher's own stated reasoning, description and summary;
#   3 fact     the real game context and images (checked by gpt-6.1-sol);
#   4 call     the teacher's actual call (regenerate one from the thinking
#              and compare the next move).

# 1. Code consistency: does the thinking's plan match the code the teacher ran?
JUDGE_CODE = """\
You check an agent's private thinking at one step of a grid puzzle game against the code it then ran. The thinking should lead to exactly this code: the move or moves it decides on, the targets and values it computes, and what the code inspects or does should all follow from it. You see only the thinking and the code, not the game.

<thinking>
{thinking}
</thinking>

The code that followed:
{call}

1. List what the code actually does: every real game action it sends (the arguments of each `action(...)`, in order), and the main things it computes, inspects or asserts.
2. For each item, say whether the thinking decides on it or accounts for it (same content, any wording).
3. List the disagreements: a move, target or value in the code that the thinking decides differently, a plan in the thinking the code does not carry out, or a step the code takes that the thinking never mentions.

Answer with JSON only:
{{"code_does": ["..."], "accounted_for": [true, false], "disagreements": ["..."], "leads_to_call": true, "notes": "one sentence"}}"""


# 2. Coverage of the teacher's own words (stated reasoning, description, summary).
JUDGE_WORDS = """\
You compare an agent's private thinking at one step of a grid puzzle game with the agent's own brief account of that same step. The account is ground truth for what the agent thought; the thinking was written separately and should contain the same points, worked out in more detail.

The agent's account of the step:
{account}

<thinking>
{thinking}
</thinking>

1. List the distinct points of the account (observations, hypotheses about the rules or goal, the plan, the decision), at most 10, most important first.
2. For each, say whether the thinking contains it (same content, any wording).
3. List claims in the thinking that contradict the account.

Answer with JSON only:
{{"points": ["..."], "covered": [true, false], "contradictions": ["..."], "notes": "one sentence"}}"""


# 3. Fact-check by gpt-6.1-sol, which is given the real context and images and
# checks the thinking against the state actually shown. Appended as a user
# message after the agent's own context (which the Responses call keeps whole,
# images included).
JUDGE_FACT = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

Above is the full context an agent had at one step of a game it was playing, up to the moment of its next move. Below is a written-out version of the private thinking for that step, produced afterwards by another model. Using the real state visible in the context above (the frames, tool outputs and images), check the thinking for statements that are factually wrong: coordinates, counts, colours, directions, positions, sizes, or claimed game mechanics that do not match what the context actually shows.

<thinking>
{thinking}
</thinking>

List only clear factual errors, each as the wrong claim and what the context actually shows. Do not list matters of style, points the thinking merely leaves out, or things that are genuinely uncertain from the context. Judge only against the context above, never against what a later turn would reveal.

Answer with JSON only:
{{"errors": [{{"claim": "...", "actual": "..."}}], "grounded": true, "notes": "one sentence"}}"""


# 4. Equivalence of a regenerated call with the teacher's. Flash is first given
# the thinking and the code-only request (REGEN_NOTE) and made to produce a
# python call; this judge then compares the two calls' next move.
REGEN_NOTE = """\
That is your private thinking for this step. Now make the `python` tool call it leads to: call `python` with the `code` that carries out the move you decided on. Do not add any other text."""

JUDGE_CALL = """\
You compare two Python snippets an agent could run at one step of a grid puzzle game. Only the real environment actions they send matter, that is the calls to `action(...)`, not how the code computes them or what it prints. Snippet A is what the agent actually ran. Snippet B was produced by another model from a reconstruction of the agent's thinking, without seeing A.

Snippet A (actual):
```python
{sol_code}
```

Snippet B (regenerated):
```python
{regen_code}
```

1. List the game actions A sends, in order (e.g. "MOUSE(row=39,col=45)", "LEFT"); write "(none: inspection only)" if it sends no action.
2. Do the same for B.
3. Do they make the same next move? Yes if the first action matches, or the first batch matches in order, with MOUSE targets the same or within a cell or two; a snippet that only inspects matches another that only inspects.

Answer with JSON only:
{{"a_actions": ["..."], "b_actions": ["..."], "same_next_move": true, "notes": "one sentence"}}"""


def judge_code_prompt(call: str, generated: str) -> str:
    return JUDGE_CODE.format(call=call.strip(), thinking=generated.strip())


def judge_words_prompt(reasoning: str, description: str, summary: str, generated: str) -> str:
    blocks = []
    if (reasoning or "").strip():
        blocks.append(f"<reasoning>\n{reasoning.strip()}\n</reasoning>")
    if (description or "").strip():
        blocks.append(f"<description>\n{description.strip()}\n</description>")
    if (summary or "").strip():
        blocks.append(f"<summary>\n{summary.strip()}\n</summary>")
    return JUDGE_WORDS.format(account="\n".join(blocks), thinking=generated.strip())


def judge_fact_prompt(generated: str) -> str:
    return JUDGE_FACT.format(thinking=generated.strip())


def judge_call_prompt(sol_code: str, regen_code: str) -> str:
    return JUDGE_CALL.format(sol_code=sol_code.rstrip(), regen_code=regen_code.rstrip())
