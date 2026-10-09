"""Offline behavioral tests: no API credentials or model requests needed."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from think_gen import context, logs, progressive as p


def verdict(issue=False, call=False):
    v = {"words": {"points": ["move"], "covered": [not issue], "contradictions": [], "notes": "coverage"},
         "code": {"leads_to_call": True, "disagreements": [], "notes": "code"},
         "fact": {"grounded": not issue, "errors": [{"claim": "x", "actual": "y"}] if issue else [], "notes": "facts"}}
    if call:
        v["call"] = {"functionally_same": True, "differences": [], "notes": "same"}
    return v


def args(tmp_path):
    return SimpleNamespace(out=tmp_path / "out", run=tmp_path, limit=0, input_limit=120000,
        model="flash", provider="Alibaba", judge_model="sol", max_tokens=8192,
        sol_max_tokens=16000, sol_prices=[2., .1, 10.], cache_retention="default",
        budget=10., max_attempts=3, retry_uncertain=False, monitor_per_game=15)


def response(text):
    return {"content": text, "finish_reason": "stop", "status": "completed",
            "usage": {"input_tokens": 1000, "output_tokens": 100}}


def test_strict_judge_and_truncation():
    assert p.validate_judge(response(json.dumps(verdict(True))))["fact"]["errors"]
    for bad in [{}, {"words": {"error": "timeout"}}, {**verdict(), "code": {"leads_to_call": "true"}}]:
        with pytest.raises(ValueError):
            p.validate_judge(response(json.dumps(bad)))
    wrong = verdict()
    wrong["fact"]["errors"] = [{"claim": "bad", "actual": "good"}]
    with pytest.raises(ValueError):
        p.validate_judge(response(json.dumps(wrong)))
    with pytest.raises(ValueError):
        p.validate_thinking({**response("unfinished"), "finish_reason": "length"})


def test_resumes_response_committed_before_stage_checkpoint(tmp_path, monkeypatch):
    a = args(tmp_path)
    calls = p.DurableCalls(a.out, a)
    count = []
    monkeypatch.setattr(p.client, "openai_responses", lambda *a, **k: count.append(1) or response("valid"))
    original = p.atomic_json
    def interrupt(path, value):
        if path.name == "complete.json":
            raise RuntimeError("power loss before stage commit")
        return original(path, value)
    monkeypatch.setattr(p, "atomic_json", interrupt)
    folder = a.out / "turns/game/00000"
    with pytest.raises(RuntimeError, match="power loss"):
        calls.call(folder, "draft", "sol", [], [], p.validate_thinking)
    monkeypatch.setattr(p, "atomic_json", original)
    resumed = p.DurableCalls(a.out, a)
    assert resumed.call(folder, "draft", "sol", [], [], p.validate_thinking) == "valid"
    assert len(count) == 1
    assert resumed.cost() == pytest.approx(.003)
    # A changed request cannot reuse a previous completion.
    with pytest.raises(ValueError, match="Changed request"):
        resumed.call(folder, "draft", "sol", [{"role": "user", "content": "changed"}], [], p.validate_thinking)


def test_invalid_output_billed_and_retried(tmp_path, monkeypatch):
    a = args(tmp_path)
    calls = p.DurableCalls(a.out, a)
    replies = iter([response(""), response("valid")])
    monkeypatch.setattr(p.client, "openai_responses", lambda *a, **k: next(replies))
    monkeypatch.setattr(p.time, "sleep", lambda _: None)
    assert calls.call(a.out / "turns/game/00000", "judge", "sol", [], [], p.validate_thinking) == "valid"
    assert calls.cost() == pytest.approx(.006)
    assert calls.new_calls == 2


def test_http_rejection_has_zero_charge_and_retries(tmp_path, monkeypatch):
    a = args(tmp_path)
    calls = p.DurableCalls(a.out, a)
    replies = iter([p.client.HTTPFailure(503, "HTTP 503: upstream connect failure"), response("valid")])
    def send(*args, **kwargs):
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(p.client, "openai_responses", send)
    monkeypatch.setattr(p.time, "sleep", lambda _: None)
    folder = a.out / "turns/game/00000"
    assert calls.call(folder, "judge", "sol", [], [], p.validate_thinking) == "valid"
    failed = p.read_json(folder / "calls/judge/attempt-00.json")
    assert failed["state"] == "http_rejected" and failed["charged_usd"] == 0
    assert calls.cost() == pytest.approx(.003)


def test_unknown_call_outcome_keeps_reservation(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.max_attempts = 1
    calls = p.DurableCalls(a.out, a)
    monkeypatch.setattr(p.client, "openai_responses",
                        lambda *args, **kwargs: (_ for _ in ()).throw(p.client.CallError("connection lost")))
    folder = a.out / "turns/game/00000"
    with pytest.raises(RuntimeError, match="failed after 1 attempts"):
        calls.call(folder, "judge", "sol", [], [], p.validate_thinking)
    failed = p.read_json(folder / "calls/judge/attempt-00.json")
    assert failed["state"] == "error_uncertain"
    assert calls.cost() == failed["charged_usd"] > 0


def test_historical_http_adjustment_preserves_journal(tmp_path):
    a = args(tmp_path)
    attempt = a.out / "turns/game/00000/calls/judge/attempt-00.json"
    p.atomic_json(attempt, {"state": "error_uncertain", "charged_usd": 1.25,
                            "error": "gave up after 1 attempts: HTTP 503: upstream failure"})
    original_hash = p.file_hash(attempt)
    item = {"path": str(attempt.relative_to(a.out)), "sha256": original_hash,
            "http_status": 503, "original_reservation_usd": 1.25,
            "effective_charge_usd": 0.0}
    sidecar = a.out / p.HTTP_BILLING_ADJUSTMENTS
    p.atomic_json(sidecar, {"version": 1, "adjustments": [item]})
    assert p.DurableCalls(a.out, a).cost() == 0
    assert p.file_hash(attempt) == original_hash
    item["sha256"] = "0" * 64
    p.atomic_json(sidecar, {"version": 1, "adjustments": [item]})
    with pytest.raises(ValueError, match="does not match original journal"):
        p.DurableCalls(a.out, a)


def test_client_reports_explicit_http_failure(monkeypatch):
    class Rejected:
        status_code = 503
        text = "upstream connect failure"
    monkeypatch.setenv("OR_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(p.client.requests, "post", lambda *args, **kwargs: Rejected())
    for send in (lambda: p.client.chat([], retries=0),
                 lambda: p.client.openai_responses([], retries=0)):
        with pytest.raises(p.client.HTTPFailure) as exc:
            send()
        assert exc.value.status_code == 503


def test_monitoring_only_stage_can_exhaust_retries_without_stopping_run(tmp_path, monkeypatch):
    a = args(tmp_path)
    calls = p.DurableCalls(a.out, a)
    sent = []
    def unavailable(*args, **kwargs):
        sent.append(1)
        raise p.client.CallError("HTTP 503: upstream connect failure")
    monkeypatch.setattr(p.client, "openai_responses", unavailable)
    monkeypatch.setattr(p.time, "sleep", lambda _: None)
    folder = a.out / "turns/game/00000"
    result = calls.call(folder, "final_audit", "sol", [], [], p.validate_thinking,
                       monitoring_optional=True)
    assert result is None
    assert len(sent) == a.max_attempts
    assert not calls.stop.is_set()
    complete = p.read_json(folder / "calls/final_audit/complete.json")
    assert complete["value"] is None
    assert complete["attempt"] == "attempt-02.json"
    assert "HTTP 503" in complete["monitoring_error"]
    resumed = p.DurableCalls(a.out, a)
    assert resumed.call(folder, "final_audit", "sol", [], [], p.validate_thinking,
                        monitoring_optional=True) is None
    assert len(sent) == a.max_attempts


def test_uncertain_monitoring_request_is_terminalized_without_resend(tmp_path, monkeypatch):
    a = args(tmp_path)
    folder = a.out / "turns/game/00000"
    first = p.DurableCalls(a.out, a)
    def interrupt(*args, **kwargs):
        raise RuntimeError("process interrupted during network request")
    monkeypatch.setattr(p.client, "openai_responses", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        first.call(folder, "final_audit", "sol", [], [], p.validate_thinking,
                   monitoring_optional=True)
    attempt_path = folder / "calls/final_audit/attempt-00.json"
    assert p.read_json(attempt_path)["state"] == "pending"
    resumed = p.DurableCalls(a.out, a)
    monkeypatch.setattr(p.client, "openai_responses", lambda *a, **k: pytest.fail("uncertain request resent"))
    assert resumed.call(folder, "final_audit", "sol", [], [], p.validate_thinking,
                        monitoring_optional=True) is None
    assert p.read_json(attempt_path)["state"] == "monitoring_uncertain"
    assert p.read_json(folder / "calls/final_audit/complete.json")["attempt"] == "attempt-00.json"
    assert resumed.cost() > 0  # retain the conservative uncertain-call reservation


def test_restart_repairs_monitoring_completion_after_uncertain_state_write(tmp_path, monkeypatch):
    a = args(tmp_path)
    folder = a.out / "turns/game/00000"
    first = p.DurableCalls(a.out, a)
    monkeypatch.setattr(p.client, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        first.call(folder, "regen", "flash", [], [], lambda _: "code",
                   monitoring_optional=True)
    attempt = folder / "calls/regen/attempt-00.json"
    saved = p.read_json(attempt)
    saved.update(state="monitoring_uncertain", error="crash after terminal state write")
    p.atomic_json(attempt, saved)
    resumed = p.DurableCalls(a.out, a)
    monkeypatch.setattr(p.client, "chat", lambda *a, **k: pytest.fail("uncertain request resent"))
    assert resumed.call(folder, "regen", "flash", [], [], lambda _: "code",
                        monitoring_optional=True) is None
    assert p.read_json(folder / "calls/regen/complete.json")["attempt"] == "attempt-00.json"


def test_budget_reserves_before_send_and_pending_requires_opt_in(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.budget = .01
    calls = p.DurableCalls(a.out, a)
    monkeypatch.setattr(p.client, "openai_responses", lambda *a, **k: pytest.fail("must not send"))
    with pytest.raises(p.BudgetExceeded):
        calls.call(a.out / "turns/game/00000", "judge", "sol", [], [], p.validate_thinking)
    a.budget = 10
    calls = p.DurableCalls(a.out, a)
    def crash(*a, **k):
        raise RuntimeError("process interrupted during network request")
    monkeypatch.setattr(p.client, "openai_responses", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        calls.call(a.out / "turns/game/00000", "judge", "sol", [], [], p.validate_thinking)
    resumed = p.DurableCalls(a.out, a)
    assert resumed.cost() > 0
    with pytest.raises(RuntimeError, match="Uncertain API billing"):
        resumed.call(a.out / "turns/game/00000", "judge", "sol", [], [], p.validate_thinking)


def source(tmp_path):
    system = {"role": "system", "content": "system"}
    user = {"role": "user", "content": "initial board"}
    def reply(i):
        return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
            "name": "python", "arguments": json.dumps({"code": "print(1)", "reasoning": "look", "description": "inspect"})}}]}
    r0, r1, r2 = [reply(i) for i in range(3)]
    tool0 = {"role": "tool", "tool_call_id": "c0", "content": "board 0"}
    tool1 = {"role": "tool", "tool_call_id": "c1", "content": "board 1"}
    contexts = [[system, user], [system, user, r0, tool0, user],
                [system, {"role": "user", "content": "compaction note"}, r1, tool1, user]]
    path = tmp_path / "aa00-test_p0_requests.jsonl"
    with path.open("w") as f:
        for i, (msgs, reply) in enumerate(zip(contexts, [r0, r1, r2])):
            f.write(json.dumps({"event": "request", "messages": msgs, "tools": []}) + "\n")
            f.write(json.dumps({"event": "response", "reply": reply, "usage": {}}) + "\n")
    return path


class FakeStudent:
    def messages(self, rec, history, final=None):
        msgs = copy.deepcopy(rec.messages)
        for m in msgs:
            if m["role"] == "assistant":
                m["reasoning_content"] = history[context.message_ref(m)]
        if final is not None:
            msgs.append({**copy.deepcopy(rec.reply), "reasoning_content": final})
        return msgs, []

    def count(self, messages, tools):
        # Turn 1 overflows, but turn 2 fits after compaction.
        return 120001 if any(m.get("tool_call_id") == "c0" for m in messages) else 120000


class FakeCalls:
    def __init__(self):
        self.seen = []
        self.break_at = None
        self.monitoring_unavailable = set()

    def cost(self):
        return 0

    def call(self, folder, stage, api, messages, tools, validate, **kwargs):
        if (folder.name, stage) == self.break_at:
            raise RuntimeError("interrupted")
        self.seen.append((folder.name, stage, copy.deepcopy(messages)))
        if stage in self.monitoring_unavailable and kwargs.get("monitoring_optional"):
            p.atomic_json(folder / "calls" / stage / "complete.json", {
                "request_hash": "fake-request", "value": None,
                "monitoring_error": f"{stage} monitoring unavailable after retries (HTTP 503)."})
            return None
        if api == "sol":
            return verdict(issue=stage != "final_audit", call=stage == "final_audit")
        if stage == "regen":
            return "print(1)"
        return f"{stage} final rationale for {folder.name}"


def test_final_history_crosses_compaction_and_overflow_is_only_target_filter(tmp_path):
    path = source(tmp_path)
    a, student, calls = args(tmp_path), FakeStudent(), FakeCalls()
    rows = p.generate_game(path, a, student, calls)
    assert [r["chunk"] for r in rows] == [0, 0, 1]
    assert [r["training_eligible"] for r in rows] == [True, False, True]
    # Every call for later turns carries REFINE2 output, never the initial draft.
    for idx, stage, messages in calls.seen:
        if idx == "00000":
            continue
        expected = "00000" if idx == "00001" else "00001"
        history = [m for m in messages[:-1] if m["role"] == "assistant"]
        assert len(history) == 1
        assert f"refine2 final rationale for {expected}" in json.dumps(history)
    p.export_game(path, a, student)
    samples = [json.loads(l) for l in next((a.out / "sft").glob("*.jsonl")).read_text().splitlines()]
    assert [s["id"].split("#")[-1] for s in samples] == ["0", "2"]
    assert "refine2 final rationale for 00001" in json.dumps(samples[-1])
    assert samples[-1]["loss_target_message_index"] == len(samples[-1]["messages"]) - 1
    calls.seen.clear()
    assert p.generate_game(path, a, student, calls) == rows
    assert calls.seen == []


def test_game_restart_loads_finalized_previous_turn_and_stops_on_failure(tmp_path):
    path = source(tmp_path)
    a, student, calls = args(tmp_path), FakeStudent(), FakeCalls()
    calls.break_at = ("00001", "draft")
    with pytest.raises(RuntimeError, match="interrupted"):
        p.generate_game(path, a, student, calls)
    assert (a.out / "turns/aa00-test_p0/00000/final.json").exists()
    assert not (a.out / "turns/aa00-test_p0/00001/final.json").exists()
    calls.break_at = None
    calls.seen.clear()
    p.generate_game(path, a, student, calls)
    assert all(idx != "00000" for idx, _, _ in calls.seen)


def test_sol_history_is_visible_and_has_no_teacher_reasoning():
    msg = {"role": "assistant", "content": "", "reasoning_content": "final generated",
           "tool_calls": [{"function": {"name": "python", "arguments": {"code": "print(1)"}}}]}
    result = p.api_history([msg], "sol")[0]
    assert "final generated" in result["content"]
    assert "reasoning_content" not in result
    assert isinstance(result["tool_calls"][0]["function"]["arguments"], str)


def test_monitoring_panel_is_fixed_and_evenly_distributed():
    panel = p.monitoring_indices(147, 15)
    assert len(panel) == 15 and min(panel) == 0 and max(panel) == 146
    gaps = [b-a for a,b in zip(sorted(panel),sorted(panel)[1:])]
    assert max(gaps)-min(gaps) <= 1
    assert p.monitoring_indices(14, 15) == set(range(14))


def test_unsampled_turns_skip_regeneration_and_final_judge(tmp_path):
    path = source(tmp_path)
    a, student, calls = args(tmp_path), FakeStudent(), FakeCalls()
    a.monitor_per_game = 2  # small synthetic game: first + last, skip middle
    rows = p.generate_game(path, a, student, calls)
    assert [r['monitoring_sample'] for r in rows] == [True, False, True]
    assert rows[1]['final_judge'] is None and rows[1]['regenerated_code'] is None
    assert rows[1]['last_judge_applies_to_final'] is False  # refine2 was not re-judged
    middle_stages = [stage for idx,stage,_ in calls.seen if idx == '00001']
    assert middle_stages == ['draft', 'judge1', 'refine1', 'judge2', 'refine2']
    # Unsampled finalized thinking is still needed by the next compacted chunk.
    assert rows[2]['history_refs']['c1'] == p.digest(rows[1]['thinking'])


def test_unavailable_sampled_monitoring_does_not_break_final_thinking_history(tmp_path):
    path = source(tmp_path)
    a, student, calls = args(tmp_path), FakeStudent(), FakeCalls()
    a.monitor_per_game = 2
    calls.monitoring_unavailable = {'regen', 'final_audit'}
    rows = p.generate_game(path, a, student, calls)
    assert rows[0]['training_eligible']
    assert rows[0]['final_judge'] is None
    assert rows[0]['regenerated_code'] is None
    assert len(rows[0]['monitoring_exception']) == 2
    assert rows[1]['history_refs']['c0'] == p.digest(rows[0]['thinking'])
    assert rows[2]['history_refs']['c1'] == p.digest(rows[1]['thinking'])


def test_rate_limits_retry_same_attempt_without_multiplying_reservation(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.max_attempts = 1
    calls = p.DurableCalls(a.out, a)
    waits, sent = [], []
    def wait_without_charge(delay):
        assert calls.cost() == 0
        waits.append(delay)
        return False
    monkeypatch.setattr(calls.stop, 'wait', wait_without_charge)
    def send(messages, **kwargs):
        sent.append(copy.deepcopy(messages))
        if len(sent) <= 12:
            raise p.client.RateLimited('HTTP 429: busy', retry_after='1')
        return response('valid')
    monkeypatch.setattr(p.client, 'openai_responses', send)
    folder = a.out / 'turns/game/00000'
    assert calls.call(folder, 'judge', 'sol', [{'role':'user','content':'same request'}], [], p.validate_thinking) == 'valid'
    assert len(waits) == 12 and len(sent) == 13
    assert all(m == sent[0] for m in sent)
    attempts = list((folder/'calls/judge').glob('attempt-*.json'))
    assert len(attempts) == 1
    assert p.read_json(attempts[0])['rate_limit_retries'] == 12
    assert calls.cost() == pytest.approx(.003)
    assert all(60 <= d <= 61 for d in waits[5:])


def test_resume_during_known_rate_limit_wait_reuses_slot(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.max_attempts = 1
    folder = a.out / 'turns/game/00000'
    calls = p.DurableCalls(a.out, a)
    def busy(*args, **kwargs):
        raise p.client.RateLimited('HTTP 429: busy')
    def interrupt_wait(delay):
        raise RuntimeError('process killed during known 429 wait')
    monkeypatch.setattr(p.client, 'openai_responses', busy)
    monkeypatch.setattr(calls.stop, 'wait', interrupt_wait)
    with pytest.raises(RuntimeError, match='known 429'):
        calls.call(folder, 'judge', 'sol', [], [], p.validate_thinking)
    saved = p.read_json(folder/'calls/judge/attempt-00.json')
    assert saved['state'] == 'rate_limited'
    resumed = p.DurableCalls(a.out, a)
    waits = []
    monkeypatch.setattr(resumed.stop, 'wait', lambda delay: waits.append(delay) or False)
    monkeypatch.setattr(p.client, 'openai_responses', lambda *a, **k: response('valid'))
    assert resumed.call(folder, 'judge', 'sol', [], [], p.validate_thinking) == 'valid'
    assert len(list((folder/'calls/judge').glob('attempt-*.json'))) == 1
    assert resumed.cost() == pytest.approx(.003)
    assert len(waits) == 1 and waits[0] > 0


def test_rate_limit_retry_after_is_floor_not_shortening_backoff(monkeypatch):
    monkeypatch.setattr(p.random, 'uniform', lambda *args: 0)
    assert p.rate_limit_delay(0, '30') == 30
    assert p.rate_limit_delay(0, '120') == 120
    assert p.rate_limit_delay(5, '1') == 60
    assert p.rate_limit_delay(100000, 'invalid') == 60
