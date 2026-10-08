import json

from think_gen import assemble, checks, context, logs, prompts

SYS = {"role": "system", "content": "You are a coding agent."}


def call(i, code="print(1)"):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})}}


def write_log(path, turns):
    """turns: list of (messages, reply, usage)."""
    with open(path, "w") as f:
        for i, (messages, reply, usage) in enumerate(turns):
            f.write(json.dumps({"event": "request", "messages": messages, "tools": [],
                                "analysis_step": i + 1, "request_index_within_turn": 1}) + "\n")
            f.write(json.dumps({"event": "response", "reply": reply, "usage": usage,
                                "analysis_step": i + 1, "request_index_within_turn": 1}) + "\n")


def game_log(tmp_path):
    u1 = {"role": "user", "content": [{"type": "text", "text": "frame 1"}]}
    r1 = {"role": "assistant", "content": "", "tool_calls": [call(1)],
          "reasoning": "summary text",
          "reasoning_details": [{"type": "reasoning.encrypted", "data": "x",
                                 "summary": [{"text": "**Looking**\n\nI'm looking."}]}]}
    t1 = {"role": "tool", "tool_call_id": "c1", "content": "1"}
    u2 = {"role": "user", "content": "frame 2", "_arc3_control": "nudge"}
    r2 = {"role": "assistant", "content": None, "tool_calls": [call(2, "action('UP')")],
          "reasoning_details": [{"type": "reasoning.encrypted", "data": "y", "summary": []}]}
    t2 = {"role": "tool", "tool_call_id": "c2", "content": "2"}
    u3 = {"role": "user", "content": "frame 3"}
    r3 = {"role": "assistant", "content": None, "tool_calls": [call(3)]}
    usage = lambda n: {"completion_tokens_details": {"reasoning_tokens": n}}
    turns = [
        ([SYS, u1], r1, usage(50)),
        ([SYS, u1, r1, t1, u2], r2, usage(0)),
        ([SYS, u3], r3, usage(30)),  # history trimmed: a new stretch
    ]
    path = tmp_path / "ab12-0d8bbf25_p0_requests.jsonl"
    write_log(path, turns)
    return path


def test_read_log_summary_and_hidden_reasoning(tmp_path):
    recs = logs.read_log(game_log(tmp_path))
    assert [r.key for r in recs] == ["ab12-0d8bbf25_p0#0", "ab12-0d8bbf25_p0#1", "ab12-0d8bbf25_p0#2"]
    assert recs[0].summary == "**Looking**\n\nI'm looking."
    assert recs[0].real_reasoning == ""  # encrypted: `reasoning` holds the summary, not the thinking
    assert recs[0].reasoning_tokens == 50 and recs[1].summary == ""


def test_history_inlines_thinking_and_drops_private_keys(tmp_path):
    recs = logs.read_log(game_log(tmp_path))
    thinking = {"c1": "First line.\n\nSecond line."}
    asst = context.history(recs[1].messages, thinking)[2]
    assert asst["reasoning"] == "First line.\nSecond line." and asst["content"] == ""
    assert "reasoning_details" not in asst
    msgs = context.history(recs[1].messages, thinking, mode="inline")
    asst = msgs[2]
    assert "reasoning" not in asst and "reasoning_details" not in asst
    assert asst["content"] == "[thinking]\nFirst line.\nSecond line.\n[/thinking]"
    assert all("_arc3_control" not in m for m in msgs)
    assert recs[1].messages[2].get("reasoning") == "summary text"  # input left untouched


def test_call_text_and_prompt(tmp_path):
    recs = logs.read_log(game_log(tmp_path))
    text = context.call_text(recs[1].reply)
    assert "```python\naction('UP')\n```" in text
    assert "<summary>" not in prompts.reconstruct_prompt(text, "")
    assert "[thinking] and [/thinking] lines" not in prompts.reconstruct_prompt(text, "")
    assert "[thinking] and [/thinking] lines" in prompts.reconstruct_prompt(text, "", history_mode="inline")
    assert "I'm looking." in prompts.reconstruct_prompt(text, recs[0].summary)


def test_checks():
    reply = {"tool_calls": [call(1, "for n in current_frame.segmentation['nodes']: print(n)")]}
    assert checks.check("I look at the board and print the nodes.", reply)["ok"]
    bad = checks.check("The summary says I should print nodes.", reply)
    assert bad["leaks"] == ["summary"] and not bad["ok"]
    pasted = checks.check("for n in current_frame.segmentation['nodes']: print(n)", reply)
    assert pasted["pasted_code"] == 1.0 and not pasted["ok"]
    assert checks.clean("[thinking]\nabc\n[/thinking]") == "abc"


def test_assemble_stretches_and_statuses(tmp_path):
    recs = logs.read_log(game_log(tmp_path))
    assert assemble.stretches(recs) == [[0, 1], [2]]
    rows = {
        recs[0].key: {"ref": "c1", "thinking": "A.\n\nB.", "status": "ok"},
        recs[1].key: {"ref": "c2", "thinking": "", "status": "teacher_empty"},
    }
    s = assemble.sample(recs, [0, 1], rows, raw_final=False)
    asst = [m for m in s["messages"] if m["role"] == "assistant"]
    assert [m["reasoning_content"] for m in asst] == ["A.\nB.", ""]
    assert asst[1]["content"] == "" and "reasoning_details" not in asst[0]
    assert s["turn_status"] == ["ok", "teacher_empty"]
    assert all("_arc3_control" not in m for m in s["messages"])
    s2 = assemble.sample(recs, [2], rows, raw_final=False)
    assert s2["turn_status"] == ["missing"]


def test_b3_prompt_and_token_bins():
    import json
    from think_gen import evalset, prompts
    reply = {"tool_calls": [{"id": "c1", "function": {"name": "python", "arguments": json.dumps(
        {"description": "Press UP once.", "reasoning": "The door is above.", "code": "action(['UP'])"})}}]}
    p = prompts.reconstruct_prompt_b3(reply, 400)
    assert "The door is above." in p and "Press UP once." in p and "action(['UP'])" in p
    assert "about 300 words" in p
    assert prompts.LENGTH_NONE in prompts.reconstruct_prompt_b3(reply, 0)
    assert checks.leaks("As the description says, I press UP.") == ["stated_fields"]
    edges = [10, 20, 30]
    assert [evalset.token_bin(t, edges) for t in (0, 5, 10, 25, 99)] == [0, 1, 2, 3, 4]


def test_leak_ignores_words_the_context_uses():
    assert checks.leaks("The agent moves left.") == ["the_agent"]
    assert checks.leaks("The agent moves left.", "the agent sprite is at (3, 4)") == []


def test_b4_prompt_length_floor():
    import json
    reasoning = " ".join(["word"] * 60)
    reply = {"tool_calls": [{"id": "c1", "function": {"name": "python", "arguments": json.dumps(
        {"description": "Press UP once.", "reasoning": reasoning, "code": "action(['UP'])"})}}]}
    p, words = prompts.reconstruct_prompt_b4(reply, 0)
    assert words == 180 and "Write about 180 words." in p and "less than 180 words" in p
    assert "trace it back through the conversation" in p and "{" not in p.replace("{'", "")
    assert prompts.reconstruct_prompt_b4(reply, 2000)[1] == 1500


def test_copy_of_stated_reasoning_is_rejected():
    import json
    reasoning = "The right arrow shifted the ten row tiles cyclically. I will search the two yellow tiles."
    reply = {"tool_calls": [{"id": "c1", "function": {"name": "python", "arguments": json.dumps(
        {"description": "Search.", "reasoning": reasoning, "code": "action(['UP'])"})}}]}
    c = checks.check(reasoning, reply, min_words=100)
    assert c["copied_reasoning"] == 1.0 and c["too_short"] and not c["ok"]
    assert checks.check(reasoning, reply)["ok"]
