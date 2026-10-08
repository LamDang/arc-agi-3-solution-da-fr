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
# thinking is hidden (gpt-6.1-sol), so there is nothing to diff against. All
# four run on gpt-6.1-sol with the teacher's own context (frames and images)
# in front of them, and each uses something else that IS known about the step:
#   1 code     the python code the teacher ran;
#   2 words    the teacher's own stated reasoning, description and summary;
#   3 fact     the real game state (claims checked against it);
#   4 call     the teacher's actual call (regenerate one from the thinking and
#              judge whether it does the same thing).
# Each prompt is appended after the context, as a user turn.

# 1. Code consistency: does the thinking's plan match the code the teacher ran?
# Lenient on code that does MORE than the thinking: extra inspection, prints,
# asserts and implicit computation are expected, not disagreements.
JUDGE_CODE = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

Above is the context you had at one step of a game you were playing, up to the moment of your next move. Below is a written-out version of your private thinking for that step, produced afterwards by another model, and the code you then ran. Using the context above to follow what the code does, check that the thinking leads to this code: the move or moves it decides on, and the targets and values it commits to, should match what the code actually does.

<thinking>
{thinking}
</thinking>

The code you ran:
{call}

Judge only genuine conflicts. The code may inspect, print, assert or sanity-check more than the thinking mentions, and may compute details the thinking leaves implicit; that is expected and is NOT a disagreement, so do not list it. List a disagreement only where the code's actual decision departs from the thinking: a different game action, a different target or value, or a step the thinking commits to that the code does not carry out. Set leads_to_call true when the code carries out the move the thinking decided on, even if the code also does more.

Answer with JSON only:
{{"disagreements": ["..."], "leads_to_call": true, "notes": "one sentence"}}"""


# 2. Coverage of the teacher's own words (stated reasoning, description, summary),
# with the context in front so the thinking can be understood against the state.
JUDGE_WORDS = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

Above is the context you had at one step of a game you were playing. Below is a written-out version of your private thinking for that step, produced afterwards by another model, and your own brief account of the step (the reasoning, description and summary you gave with your move). The account is the ground truth for what you thought; the thinking should contain the same points, worked out in more detail. Use the context above to understand the thinking.

Your account of the step:
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

The thinking is exploratory working, not a finished statement: it often puts forward a figure or reading, tests it, and corrects it a few lines later. Do not flag a claim that the thinking itself later revises, retracts or corrects. Judge only the claims the thinking still stands on by the end — the observations and conclusions its final plan rests on. A wrong value that the thinking catches and fixes is the process working, not an error.

When the thinking contains no such standing errors, this is the expected result: return an empty list `[]` for `errors` and set `grounded` to true. Only set `grounded` to false when you list at least one error.

Answer with JSON only:
{{"errors": [{{"claim": "...", "actual": "..."}}], "grounded": true, "notes": "one sentence"}}"""


# 4. Functional equivalence of a regenerated call with the teacher's. Flash is
# first given the thinking and the code-only request (REGEN_NOTE) and made to
# produce a python call; this judge then, with the context in front, decides
# whether the two snippets would do the same thing to the game, not whether
# their text or printed output matches.
REGEN_NOTE = """\
That is your private thinking for this step. Now make the `python` tool call it leads to: call `python` with the `code` that carries out the move you decided on. Do not add any other text."""

JUDGE_CALL = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

Above is the context you had at one step of a game you were playing. Below are two Python snippets for that step. Snippet A is the code you actually ran. Snippet B was produced by another model from a reconstruction of your thinking, without seeing A. Using the context above, judge whether the two snippets are functionally the same: run on this state, would they carry out the same operation and have the same effect on the game, the same move or moves on the same targets, regardless of how the code is written, what it prints, or how it computes the result?

Snippet A (yours):
```python
{sol_code}
```

Snippet B (regenerated):
```python
{regen_code}
```

Ignore differences in inspection, printing, variable names and style; weigh only what each snippet would actually do to the game. List the functional differences that would change what happens in the game, if any.

Answer with JSON only:
{{"differences": ["..."], "functionally_same": true, "notes": "one sentence"}}"""


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


# Refine pass: the student revises its own draft from the sol-judge feedback,
# keeping what the feedback endorses and fixing what it raises. Appended after
# the agent's context, like the b4 reconstruct prompt.
REFINE = """\
[Note outside the game: this is not a turn of the game, and you must not call any tool.]

The conversation above is a game-playing agent's run, up to the moment it produced its next output. Its thinking for this step was missing and you wrote a first draft of it. A reviewer who could see the full game state checked that draft. Rewrite the thinking, keeping everything the draft got right and fixing only what the reviewer raises.

The next output was a `python` call. With the code, the agent stated:

<reasoning>
{reasoning}
</reasoning>

<description>
{description}
</description>

<code>
{code}
</code>

Your first draft of the thinking:
<draft>
{draft}
</draft>

Reviewer feedback:
{feedback}

Rewrite the thinking so that it keeps the draft's correct observations, derivations and plan, and fixes the points above: correct any wrong coordinate, count or mechanic; work in the points from your own account that the draft missed; and make the plan match the code. Keep the draft's first-person voice, its working style with the checks and corrections of real work, and about its length. Fold the corrected facts in as if you had them right from the start: do not mention the draft, the feedback, the reviewer, or being corrected.

Answer with the thinking text only: no title, no {open} tags, no preface.
"""


def refine_feedback(judge: dict) -> str:
    """The sol-judge verdicts for one record as reviewer feedback: a general
    line per check (its `notes`) and the specific points to fix."""
    w = judge.get("words") or {}
    c = judge.get("code") or {}
    f = judge.get("fact") or {}
    general, specific = [], []
    if w.get("notes"):
        general.append(f"- Coverage of your stated reasoning: {w['notes']}")
    if c.get("notes"):
        general.append(f"- Match with the code you ran: {c['notes']}")
    if f.get("notes"):
        general.append(f"- Factual grounding against the board: {f['notes']}")
    for p, ok in zip(w.get("points") or [], w.get("covered") or []):
        if not ok:
            specific.append(f"- Missing from the draft (a point of your own reasoning): {p}")
    for con in w.get("contradictions") or []:
        specific.append(f"- Contradicts your reasoning: {con}")
    for d in c.get("disagreements") or []:
        specific.append(f"- Plan does not match the code: {d}")
    for e in f.get("errors") or []:
        if isinstance(e, dict):
            specific.append(f'- Factual error: the draft says "{e.get("claim", "")}", but actually {e.get("actual", "")}')
        else:
            specific.append(f"- Factual error: {e}")
    out = []
    if general:
        out.append("General:\n" + "\n".join(general))
    out.append("Specific points to fix:\n" + ("\n".join(specific) if specific else "- (nothing specific; tighten the draft)"))
    return "\n\n".join(out)


def refine_prompt(reply: dict, draft: str, feedback: str) -> str:
    from .context import THINK_OPEN
    a = python_args(reply)
    return REFINE.format(
        reasoning=(a.get("reasoning") or "").strip(), description=(a.get("description") or "").strip(),
        code=(a.get("code") or "").rstrip(), draft=draft.strip(), feedback=feedback.strip(),
        open=THINK_OPEN)
