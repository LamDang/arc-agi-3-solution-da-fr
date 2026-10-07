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
