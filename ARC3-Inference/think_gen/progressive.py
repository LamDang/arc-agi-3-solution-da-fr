"""Resumable turn-by-turn rationalization. Games parallel; all turns/chunks serial.

Use --dry-run for a no-inference preflight. Each paid attempt is reserved on
 disk before sending, and its raw response is committed before validation.
Completed stages and turns are immutable and reused on restart. Training
exports have one final-turn target, filtered at 120,000 student INPUT tokens.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import signal
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

from . import checks, client, context, logs, prompts
from .refine import has_issue

VERSION = 2
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/sft-gpt61sol-features-25games"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(Path(path).read_text())


HTTP_BILLING_ADJUSTMENTS = "http-failure-billing-adjustments.json"
_EXPLICIT_HTTP_FAILURE = re.compile(r"^(?:gave up after 1 attempts: )?HTTP ([45]\d{2}):")


def explicit_http_failure_status(error):
    """Recognize the client's recorded rejection, not a transport exception."""
    match = _EXPLICIT_HTTP_FAILURE.match(str(error))
    return int(match.group(1)) if match else None


def load_billing_adjustments(out):
    """Apply audited zero charges while leaving historical API journals intact."""
    out = Path(out)
    path = out / HTTP_BILLING_ADJUSTMENTS
    if not path.exists():
        return {}
    data = read_json(path)
    if data.get("version") != 1 or not isinstance(data.get("adjustments"), list):
        raise ValueError("Invalid HTTP billing-adjustment file")
    corrections = {}
    for item in data["adjustments"]:
        relative = Path(item["path"])
        if (relative.is_absolute() or ".." in relative.parts or
                relative.parts[:1] != ("turns",) or not relative.name.startswith("attempt-")):
            raise ValueError(f"Invalid billing-adjustment path: {relative}")
        attempt = out / relative
        row = read_json(attempt)
        if (file_hash(attempt) != item["sha256"] or row.get("state") != "error_uncertain" or
                row.get("response") is not None or
                row.get("charged_usd") != item["original_reservation_usd"] or
                explicit_http_failure_status(row.get("error")) is None or
                explicit_http_failure_status(row.get("error")) != item["http_status"] or
                item.get("effective_charge_usd") != 0):
            raise ValueError(f"Billing adjustment does not match original journal: {relative}")
        key = str(attempt)
        if key in corrections:
            raise ValueError(f"Duplicate billing adjustment: {relative}")
        corrections[key] = 0.0
    return corrections


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudentFormat:
    """Same pinned template, code-only rewrite and image accounting as SFT data.

    Source board images must be 640x640: this model uses 400 vision tokens each.
    Unknown image shapes fail closed, rather than silently undercounting them.
    """
    def __init__(self, tokenizer, template):
        from tokenizers import Tokenizer
        self.convert = load_module("sft_convert", DATA / "convert.py")
        self.index = load_module("sft_index", DATA / "build_index.py")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.template = self.index.compile_template(Path(template).read_text())
        self.checked_images = set()
        self.lock = threading.Lock()

    def messages(self, rec, history, final=None):
        out = []
        source = rec.messages if final is None else [*rec.messages, rec.reply]
        for i, original in enumerate(source):
            m = {k: copy.deepcopy(v) for k, v in original.items() if not k.startswith("_")}
            if m["role"] == "system":
                m["content"] = self.convert.rationale_off_system_prompt(m["content"])
            elif m["role"] == "assistant":
                ref = context.message_ref(m)
                text = final if final is not None and i == len(source) - 1 else history[ref]
                m = self.convert.rationale_off_assistant(m)
                m["reasoning_content"] = context.normalize(text)
            out.append(m)
        return out, self.convert.rationale_off_tools(rec.tools)

    def count(self, messages, tools):
        import base64
        import io
        from PIL import Image
        for m in messages:
            content = m.get("content")
            for p in content if isinstance(content, list) else []:
                if p.get("type") != "image_url":
                    continue
                url = p["image_url"]["url"]
                sha = hashlib.sha256(url.encode()).hexdigest()
                with self.lock:
                    if sha in self.checked_images:
                        continue
                    if not url.startswith("data:image/"):
                        raise ValueError("Token counting requires embedded source images")
                    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
                        if im.size != (640, 640):
                            raise ValueError(f"Unsupported image size {im.size}; need processor accounting")
                    self.checked_images.add(sha)
        prompt = self.index.render(self.template, messages, tools,
                                   {"enable_thinking": True, "preserve_thinking": True}, True)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids) + \
            self.index.count_images(messages) * 399


def api_history(messages, mode):
    """Keep the same finalized evidence for both providers; no encrypted teacher history."""
    out = copy.deepcopy(messages)
    for m in out:
        if m["role"] != "assistant":
            continue
        text = m.pop("reasoning_content", "")
        if text:
            if mode == "flash":
                m["reasoning"] = text
            else:
                # Responses conversion ignores `reasoning` fields; use visible
                # assistant text so the Sol judge actually sees generated history.
                m["content"] = context.with_thinking(m.get("content"), text)
        for c in m.get("tool_calls") or []:
            args = c["function"].get("arguments")
            if isinstance(args, dict):
                c["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
    return out


def combined_prompt(rec, thinking, regenerated=None):
    target = prompts.python_args(rec.reply)
    account = {k: target.get(k, "") for k in ("reasoning", "description")}
    account["summary"] = rec.summary
    instruction = """[Evaluation outside the game. Do not call tools. Return one JSON object.]
Judge the reconstructed thinking against the visible context and the teacher's
recorded next output. Prior generated thinking is fallible, not ground truth:
ground claims in the actual frames, tool outputs and observations.
1. words: list at most 10 important points from the stated account; give one
boolean covered per point, contradictions and a one-sentence notes string.
2. code: leads_to_call boolean, disagreements list, notes. Flag only actual
conflicting decisions/actions/targets. Extra inspection, printing, assertions
or implicit computation in the teacher code are not disagreements. For a
text-only output, check consistency with that text instead.
3. fact: grounded boolean, errors list of {claim, actual}, notes. Flag only
clear unsupported standing factual claims. Ignore claims corrected later in
the thinking itself. grounded must equal whether errors is empty.
Return keys words, code, fact with exactly those fields.
JSON shape: {"words":{"points":["..."],"covered":[true],"contradictions":[],"notes":"..."},
"code":{"leads_to_call":true,"disagreements":[],"notes":"..."},
"fact":{"grounded":true,"errors":[],"notes":"..."}}
"""
    if regenerated is not None:
        instruction += """4. call: functionally_same boolean, differences list, notes. Compare teacher
code with regenerated code for effects on the game: same actions and targets.
Ignore printing, naming, inspection and implementation differences. Include
call in the SAME JSON object. Do not execute either snippet.
"""
    payload = {"account": account, "teacher_output": context.call_text(rec.reply), "thinking": thinking}
    if regenerated is not None:
        payload["regenerated_code"] = regenerated
    return instruction + "\n" + json.dumps(payload, ensure_ascii=False)


def validate_judge(response, with_call=False, require_points=False):
    reject_incomplete(response)
    try:
        v = json.loads(response["content"])
    except (KeyError, ValueError) as e:
        raise ValueError("Judge did not return a JSON object") from e
    if not isinstance(v, dict):
        raise ValueError("Judge verdict is not an object")
    for kind in ["words", "code", "fact"] + (["call"] if with_call else []):
        if not isinstance(v.get(kind), dict) or not isinstance(v[kind].get("notes"), str):
            raise ValueError(f"Invalid judge section {kind}")
    w, c, f = v["words"], v["code"], v["fact"]
    def list_of(value, typ):
        return isinstance(value, list) and all(type(x) is typ for x in value)
    if not (list_of(w.get("points"), str) and list_of(w.get("covered"), bool)
            and len(w["points"]) == len(w["covered"]) and list_of(w.get("contradictions"), str)):
        raise ValueError("Invalid coverage verdict")
    if require_points and not w["points"]:
        raise ValueError("Empty coverage verdict for a nonempty teacher account")
    if type(c.get("leads_to_call")) is not bool or not list_of(c.get("disagreements"), str):
        raise ValueError("Invalid code verdict")
    if type(f.get("grounded")) is not bool or not isinstance(f.get("errors"), list):
        raise ValueError("Invalid fact verdict")
    if any(not isinstance(e, dict) or not all(isinstance(e.get(k), str) for k in ("claim", "actual")) for e in f["errors"]):
        raise ValueError("Invalid fact errors")
    if f["grounded"] != (not f["errors"]):
        raise ValueError("Inconsistent fact verdict")
    if with_call:
        call = v["call"]
        if type(call.get("functionally_same")) is not bool or not list_of(call.get("differences"), str):
            raise ValueError("Invalid equivalence verdict")
    return v


def reject_incomplete(response):
    if response.get("finish_reason") in ("length", "content_filter") or response.get("status") in ("incomplete", "failed"):
        raise ValueError("Incomplete model response")


def validate_thinking(response):
    reject_incomplete(response)
    text = checks.clean(response.get("content") or "")
    if not text.strip():
        raise ValueError("Empty thinking")
    return text


def validate_code(response):
    reject_incomplete(response)
    calls = response.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "python":
        raise ValueError("Expected exactly one regenerated python call")
    args = calls[0]["function"].get("arguments")
    args = json.loads(args) if isinstance(args, str) else args
    if not isinstance(args, dict) or not isinstance(args.get("code"), str) or not args["code"].strip():
        raise ValueError("Missing regenerated code")
    return args["code"]


def issue(verdict, advisory):
    return has_issue(verdict, advisory) or not verdict["code"]["leads_to_call"] or bool(verdict["fact"]["errors"])


def monitoring_indices(count, per_game):
    """Fixed, evenly spaced production-monitoring panel, independent of --limit.

    Include both ends of each full game. Short games use all responses. Selecting
    on full source length keeps the panel identical when a pilot is extended.
    """
    n = min(count, per_game)
    if n < 2:
        return set(range(n))
    return {i * (count - 1) // (n - 1) for i in range(n)}


class BudgetExceeded(RuntimeError):
    pass


def rate_limit_delay(attempt, retry_after=None):
    """Harness-style exponential backoff + jitter; short Retry-After cannot lower it."""
    delay = min(60., 2. * 2 ** min(attempt, 6)) + random.uniform(0., 1.)
    if retry_after:
        try:
            requested = float(retry_after)
        except (ValueError, TypeError):
            try:
                requested = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                requested = 0
        if math.isfinite(requested):
            delay = max(delay, requested)
    return delay


class DurableCalls:
    """A single-process, multi-worker durable API journal and cost reservation.

    A crash after an API accepted a request but before its response was committed
    cannot be made exactly-once by a local checkpoint. Such pending attempts keep
    their reservation charged and require --retry-uncertain to resend on resume.
    """
    def __init__(self, out, args):
        self.out, self.args = Path(out), args
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.costs = {}
        for p in self.out.glob("turns/*/*/calls/*/attempt-*.json"):
            self.costs[str(p)] = read_json(p)["charged_usd"]
        for path, effective_charge in load_billing_adjustments(self.out).items():
            if path not in self.costs:
                raise ValueError(f"Adjusted journal is missing: {path}")
            self.costs[path] = effective_charge
        self.new_calls = 0

    def cost(self):
        with self.lock:
            return sum(self.costs.values())

    def price(self, response, api):
        u = response.get("usage") or {}
        if api == "flash" and isinstance(u.get("cost"), (float, int)):
            return u["cost"]
        inp = u.get("input_tokens", u.get("prompt_tokens"))
        output = u.get("output_tokens", u.get("completion_tokens"))
        if inp is None or output is None:
            raise ValueError("Missing usage; cannot reconcile budget")
        cached = (u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if api == "flash":
            rates = (.15, .016, .47)
            written = 0
        else:
            rates = (4, .2, 15) if inp > 272000 else tuple(self.args.sol_prices)
            written = (u.get("input_tokens_details") or {}).get("cache_write_tokens", 0)
            if written < 0 or cached < 0 or written + cached > inp:
                raise ValueError("Invalid cache read/write token usage")
        return ((inp - cached) * rates[0] + written * rates[0] * .25 + cached * rates[1] + output * rates[2]) / 1e6

    def call(self, folder, stage, api, messages, tools, validate, *, reasoning=False, regen=False,
             cache_key=None, monitoring_optional=False):
        if monitoring_optional and stage not in ("regen", "final_audit"):
            raise ValueError("Only sampled code regeneration and final audit may be optional")
        args = self.args
        cap = args.sol_max_tokens if api == "sol" else args.max_tokens
        request = {"api": api, "messages": messages, "tools": tools, "cap": cap,
                   "model": args.judge_model if api == "sol" else args.model,
                   "reasoning": reasoning, "regen": regen, "cache_key": cache_key}
        # This is semantic identity. Cache metadata doesn't alter model evidence;
        # manifest/code fingerprints govern its separately audited migration.
        signature = digest(request)
        cache_policy = "explicit-history-v1" if api == "sol" else None
        transport_hash = digest({"semantic_request_hash": signature, "cache_policy": cache_policy})
        base = folder / "calls" / stage
        complete = base / "complete.json"
        if complete.exists():
            saved = read_json(complete)
            if saved["request_hash"] != signature:
                raise ValueError(f"Changed request on resume: {base}")
            return saved["value"]
        # Conservative local reservation: UTF-8 text bytes + 4096 per image,
        # higher long-context Sol input/output rates, never estimated cache savings.
        slim = client._slim(messages)
        image_count = sum(1 for m in messages for p in (m.get("content") if isinstance(m.get("content"), list) else []) if p.get("type") == "image_url")
        input_bound = len(json.dumps([slim, tools], ensure_ascii=False).encode()) + image_count * 4096
        reserve = (input_bound * (max(4, args.sol_prices[0]) * 1.25 if api == "sol" else .2)
                   + cap * (max(15, args.sol_prices[2]) if api == "sol" else .47)) / 1e6
        prior = sorted(base.glob("attempt-*.json"))

        def finish_monitoring_unavailable(reason, current_path=None):
            """Journal a terminally unavailable sample without stopping training."""
            if "data_inspection_failed" in reason or "inappropriate content" in reason.lower():
                safe_reason = "Provider content inspection rejected optional monitoring code regeneration."
            else:
                import re
                status = re.search(r"HTTP\s+(\d{3})", reason)
                suffix = f"HTTP {status.group(1)}" if status else "provider or validation error"
                safe_reason = f"Optional {stage} monitoring unavailable after retries ({suffix})."
            value = {"request_hash": signature, "value": None,
                     "attempt": current_path.name if current_path else (prior[-1].name if prior else None),
                     "monitoring_error": safe_reason}
            atomic_json(complete, value)
            return None

        # Recover response saved before a crash between validation and completion.
        for p in prior:
            saved = read_json(p)
            if saved["request_hash"] != signature:
                raise ValueError(f"Changed request on resume: {p}")
            if saved["state"] == "monitoring_uncertain" and monitoring_optional:
                return finish_monitoring_unavailable(
                    saved.get("error", "Optional monitoring request outcome is uncertain."), p)
            if saved["state"] == "pending" and not args.retry_uncertain:
                if monitoring_optional:
                    # The request may have been billed, so retain its reservation
                    # and mark the outcome terminal/uncertain instead of resending.
                    saved.update(state="monitoring_uncertain",
                                 error="Optional monitoring request was pending at restart; not resent automatically.",
                                 finished=time.time())
                    atomic_json(p, saved)
                    return finish_monitoring_unavailable(
                        "Prior monitoring request outcome is uncertain; it was not resent automatically.", p)
                raise RuntimeError(f"Uncertain API billing at {p}; inspect then use --retry-uncertain")
            if "response" in saved:
                try:
                    value = validate(saved["response"])
                except (ValueError, TypeError, KeyError):
                    continue
                atomic_json(complete, {"request_hash": signature, "value": value, "attempt": p.name})
                return value
        # A crash during a known 429 wait is safe to resume in the SAME slot.
        # A crash while the next HTTP request is in flight remains uncertain.
        reusable_unsent = bool(prior and read_json(prior[-1])["state"] in
                               ("rate_limited", "reserved", "cancelled_unsent"))
        start_attempt = len(prior) - 1 if reusable_unsent else len(prior)
        for attempt in range(start_attempt, args.max_attempts):
            p = base / f"attempt-{attempt:02d}.json"
            with self.lock:
                if self.stop.is_set():
                    raise RuntimeError("Run stopped after another worker failed")
                previous = read_json(p) if p.exists() else {}
                resuming_rate_limit = previous.get("state") == "rate_limited"
                if not resuming_rate_limit and sum(self.costs.values()) - self.costs.get(str(p), 0) + reserve > args.budget:
                    self.stop.set()
                    raise BudgetExceeded("Budget reservation would exceed limit; resume with an explicitly increased budget")
                saved = {"state": "rate_limited" if resuming_rate_limit else "reserved",
                         "request_hash": signature, "charged_usd": 0.0 if resuming_rate_limit else reserve,
                         "pricing_version": 2,
                         "rate_limit_retries": previous.get("rate_limit_retries", 0),
                         "last_rate_limit": previous.get("last_rate_limit"),
                         "api": api, "stage": stage, "started": time.time(),
                         "cache_policy": cache_policy, "transport_hash": transport_hash,
                         "request": {k: v for k, v in request.items() if k not in ("messages", "tools")},
                         "context": client._digest(messages)}
                atomic_json(p, saved)
                self.costs[str(p)] = saved["charged_usd"]
                self.new_calls += 1
            try:
                if saved["state"] == "rate_limited":
                    last = saved.get("last_rate_limit") or {}
                    remaining = max(0., last.get("at", 0) + last.get("delay", 0) - time.time())
                    if remaining and self.stop.wait(remaining):
                        raise RuntimeError("Run stopped while resuming provider backoff")
                while True:
                    if self.stop.is_set():
                        if saved['state'] == 'reserved':
                            saved.update(state='cancelled_unsent', charged_usd=0.0)
                            with self.lock:
                                atomic_json(p, saved)
                                self.costs[str(p)] = 0.0
                        raise RuntimeError("Run stopped while waiting for provider capacity")
                    if saved["state"] == "rate_limited":
                        # A known HTTP 429 costs nothing while waiting. Reserve
                        # again atomically before the next network send.
                        with self.lock:
                            if sum(self.costs.values()) - self.costs.get(str(p), 0) + reserve > args.budget:
                                self.stop.set()
                                raise BudgetExceeded("Budget reservation would exceed limit; resume with an explicitly increased budget")
                            saved.update(state="reserved", charged_usd=reserve)
                            atomic_json(p, saved)
                            self.costs[str(p)] = reserve
                    saved["state"] = "pending"
                    atomic_json(p, saved)
                    try:
                        if api == "sol":
                            response = client.openai_responses(messages, model=args.judge_model, effort="xhigh",
                                json_mode=True, max_output_tokens=cap, retries=0, preserve_empty_response=True,
                                cache_retention=args.cache_retention if args.cache_retention != "default" else None,
                                cache_policy=cache_policy,
                                cache_key=cache_key)
                        else:
                            response = client.chat(messages, model=args.model, provider=args.provider, tools=tools,
                                reasoning=reasoning, max_tokens=cap, temperature=.4 if regen else .7, retries=0, preserve_empty_response=True,
                                tool_choice={"type": "function", "function": {"name": "python"}} if regen else "none")
                        break
                    except client.RateLimited as e:
                        count = saved["rate_limit_retries"]
                        delay = rate_limit_delay(count, e.retry_after)
                        saved.update(state="rate_limited", charged_usd=0.0,
                                     rate_limit_retries=count + 1,
                                     last_rate_limit={"at": time.time(), "error": str(e), "delay": delay})
                        with self.lock:
                            atomic_json(p, saved)
                            self.costs[str(p)] = 0.0
                        print(f"{folder.parent.name}#{int(folder.name)} {stage}: HTTP 429, "
                              f"retry {count + 1}/unlimited in {delay:.1f}s", flush=True)
                        if self.stop.wait(delay):
                            raise RuntimeError("Run stopped while waiting for provider capacity")
                charged = self.price(response, api)
                saved.update(state="response", response=response, charged_usd=charged, finished=time.time())
                # Commit usage even when validation subsequently rejects the answer.
                with self.lock:
                    atomic_json(p, saved)
                    self.costs[str(p)] = charged
                value = validate(response)
            except (client.CallError, ValueError, TypeError, KeyError) as e:
                saved.update(error=str(e)[:1000])
                if saved["state"] == "pending":
                    if isinstance(e, client.HTTPFailure) and 400 <= e.status_code <= 599:
                        saved.update(state="http_rejected", charged_usd=0.0, finished=time.time())
                        with self.lock:
                            atomic_json(p, saved)
                            self.costs[str(p)] = 0.0
                    else:
                        saved.update(state="error_uncertain", finished=time.time())
                        atomic_json(p, saved)
                else:
                    atomic_json(p, saved)
                if attempt + 1 == args.max_attempts:
                    if monitoring_optional:
                        return finish_monitoring_unavailable(str(e), p)
                    self.stop.set()
                    raise RuntimeError(f"{stage} failed after {args.max_attempts} attempts: {e}") from e
                time.sleep(min(30, 2 ** (attempt + 1)))
                continue
            atomic_json(complete, {"request_hash": signature, "value": value, "attempt": p.name})
            return value
        if monitoring_optional:
            last_error = read_json(prior[-1]).get("error", "Monitoring retry limit reached") if prior else "No monitoring attempts made"
            return finish_monitoring_unavailable(last_error)
        raise RuntimeError(f"Attempt limit reached at {base}; inspect failure before raising --max-attempts")


def generate_game(path, args, student, calls):
    records = logs.read_log(path)
    monitor = monitoring_indices(len(records), args.monitor_per_game)
    if args.limit:
        records = records[:args.limit]
    history = {}
    chunk = -1
    last_context = []
    results = []
    for rec in records:
        current = [digest(m) for m in context.history(rec.messages, {})]
        if not last_context or current[:len(last_context)] != last_context:
            chunk += 1
        last_context = current
        # Missing retained thinking is an error, never silently replaced by teacher text.
        refs = [context.message_ref(m) for m in rec.messages if m["role"] == "assistant"]
        prior = {ref: digest(history[ref]) for ref in refs}
        history_hash = digest(prior)
        folder = args.out / "turns" / rec.game / f"{rec.index:05d}"
        final_path = folder / "final.json"
        if final_path.exists():
            row = read_json(final_path)
            if row["key"] != rec.key or row["history_hash"] != history_hash:
                raise ValueError(f"Finalized history changed on resume: {rec.key}")
        else:
            messages, tools = student.messages(rec, history)
            input_tokens = student.count(messages, tools)
            flash = api_history(messages, "flash")
            sol = api_history(messages, "sol")
            cache_key = f"think-progressive-{rec.game}-{chunk}"
            if rec.reply.get("tool_calls"):
                prompt, target_words = prompts.reconstruct_prompt_b4(rec.reply, rec.reasoning_tokens)
            else:
                target_words = 0
                prompt = ("Write a concise first-person rationale for the following recorded visible response, "
                          "using only the preceding observations. Do not invent a tool call or refer to this instruction.\n"
                          + context.call_text(rec.reply))
            def flash_call(stage, prompt, validator=validate_thinking, **kwargs):
                return calls.call(folder, stage, "flash", flash + [{"role": "user", "content": prompt}], tools, validator, **kwargs)
            def judge(stage, thinking, code=None):
                account = prompts.python_args(rec.reply)
                require_points = bool(account.get("reasoning") or account.get("description") or rec.summary)
                return calls.call(folder, stage, "sol", sol + [{"role": "user", "content": combined_prompt(rec, thinking, code)}],
                                  [], lambda r: validate_judge(r, code is not None, require_points), cache_key=cache_key,
                                  monitoring_optional=(stage == "final_audit"))
            def advisory(text):
                return checks.check(text, rec.reply, json.dumps(rec.messages, ensure_ascii=False), target_words // 2)
            thinking = flash_call("draft", prompt)
            verdict = judge("judge1", thinking)
            refined = 0
            for round_no in (1, 2):
                chk = advisory(thinking)
                if not issue(verdict, chk):
                    break
                feedback = prompts.refine_feedback(verdict, chk)
                if rec.reply.get("tool_calls"):
                    refine_prompt = prompts.refine_prompt(rec.reply, thinking, feedback)
                else:
                    refine_prompt = f"Revise this first-person rationale for the recorded visible response. Keep correct points.\nResponse: {context.call_text(rec.reply)}\nDraft: {thinking}\nFeedback: {feedback}\nReturn only the revised rationale."
                thinking = flash_call(f"refine{round_no}", refine_prompt)
                refined = round_no
                if round_no == 1:
                    verdict = judge("judge2", thinking)
            teacher_code = prompts.python_args(rec.reply).get("code")
            code = None
            final_verdict = None
            monitoring_errors = []
            if rec.index in monitor:
                if teacher_code:
                    code = flash_call("regen", thinking + "\n\n" + prompts.REGEN_NOTE, validate_code,
                                      reasoning=True, regen=True, monitoring_optional=True)
                    if code is None:
                        regen_result = read_json(folder / "calls/regen/complete.json")
                        monitoring_errors.append(regen_result.get("monitoring_error") or
                            "Code equivalence unavailable after a provider content-filter rejection.")
                final_verdict = judge("final_audit", thinking, code)
                if final_verdict is None:
                    audit_result = read_json(folder / "calls/final_audit/complete.json")
                    monitoring_errors.append(audit_result.get("monitoring_error") or
                        "Sampled final audit unavailable after provider or validation errors.")
            # Only sampled turns get final auditing/equivalence. No third refine.
            row = {"key": rec.key, "game": rec.game, "index": rec.index, "chunk": chunk,
                   "ref": context.message_ref(rec.reply), "thinking": thinking, "status": "ok",
                   "history_hash": history_hash, "history_refs": prior, "refinements": refined,
                   "student_input_tokens": input_tokens, "training_eligible": input_tokens <= args.input_limit,
                   "exclusion": None if input_tokens <= args.input_limit else "student_input_over_limit",
                   "checks": advisory(thinking), "monitoring_sample": rec.index in monitor,
                   "last_refinement_judge": verdict, "last_judge_applies_to_final": refined < 2,
                   "final_judge": final_verdict, "regenerated_code": code}
            if monitoring_errors:
                row["monitoring_exception"] = monitoring_errors
            atomic_json(final_path, row)
            print(f"{rec.key} chunk={chunk} finalized refinements={refined} input={input_tokens} "
                  f"train={row['training_eligible']} cost/reserved=${calls.cost():.3f}", flush=True)
        history[row["ref"]] = row["thinking"]
        results.append(row)
    return results


def export_game(path, args, student):
    """Only final turn has loss; overflow turns may remain as masked history."""
    rows, history = [], {}
    output = args.out / "sft" / f"{path.name.split('_requests')[0]}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    with tmp.open("w") as f:
        for rec in logs.read_log(path):
            if args.limit and rec.index >= args.limit:
                break
            p = args.out / "turns" / rec.game / f"{rec.index:05d}" / "final.json"
            if not p.exists():
                break
            row = read_json(p)
            if row["training_eligible"]:
                msgs, tools = student.messages(rec, history, row["thinking"])
                sample = {"id": rec.key, "game": rec.game.rsplit("_p", 1)[0], "request_index": rec.index,
                          "messages": msgs, "tools": tools,
                          "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
                          "loss_target_message_index": len(msgs) - 1,
                          "student_input_tokens": row["student_input_tokens"]}
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            history[row["ref"]] = row["thinking"]
            rows.append(row)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)
    return rows


def build_manifest(args, paths):
    code_paths = list(Path(__file__).parent.glob("*.py")) + [DATA / "convert.py", DATA / "build_index.py",
                 ROOT / "ARC3-Inference/inference/utils/openai_compat.py"]
    return {"version": VERSION, "source": {p.name: file_hash(p) for p in paths},
            "code": {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths},
            "tokenizer": file_hash(args.tokenizer), "template": file_hash(args.template),
            "settings": {k: getattr(args, k) for k in ("model", "provider", "judge_model", "max_tokens",
                         "sol_max_tokens", "input_limit", "cache_retention", "sol_prices", "monitor_per_game")},
            "cache_policy": "explicit-history-v1"}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="First N consecutive turns per game; 0=all. May increase on resume.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--model", default=client.MODEL)
    ap.add_argument("--provider", default=client.PROVIDER)
    ap.add_argument("--judge-model", default="gpt-6.1-sol")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--sol-max-tokens", type=int, default=16000)
    ap.add_argument("--input-limit", type=int, default=120000)
    ap.add_argument("--monitor-per-game", type=int, choices=range(10, 21), default=15,
                    help="Evenly spaced final audits + call-equivalence checks per full game (10–20)")
    ap.add_argument("--cache-retention", choices=["default", "in-memory", "24h"], default="default",
                    help="Responses retention options; 1h is not a supported API value. Validate 24h support separately.")
    ap.add_argument("--sol-prices", nargs=3, type=float, default=[2., .1, 10.], metavar=("INPUT", "CACHED", "OUTPUT"))
    ap.add_argument("--budget", type=float, required=True, help="USD estimated spending + uncertain-call reservations")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--retry-uncertain", action="store_true", help="Allow resending interrupted calls; keep their worst-case reservation")
    ap.add_argument("--dry-run", action="store_true", help="Inspect sources/settings only; no model calls")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.workers < 1 or args.limit < 0 or args.budget <= 0 or args.input_limit < 1 or args.max_attempts < 1:
        raise ValueError("Invalid run limits")
    paths = logs.request_logs(args.run, args.games)
    if not paths:
        raise ValueError("No matching game logs")
    # A pilot's game/turn selection can expand without invalidating its results.
    manifest = build_manifest(args, logs.request_logs(args.run))
    if args.dry_run:
        games = []
        for p in paths:
            records = logs.read_log(p)
            games.append({"game": records[0].game, "records": min(len(records), args.limit) if args.limit else len(records)})
        print(json.dumps({"manifest": manifest, "games": games, "budget": args.budget}, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = args.out / "manifest.json"
        if settings.exists():
            if read_json(settings) != manifest:
                raise ValueError("Resume rejected: source/code/model/tokenizer/settings changed; use a new output directory")
        else:
            atomic_json(settings, manifest)
        student = StudentFormat(args.tokenizer, args.template)
        calls = DurableCalls(args.out, args)
        # Finish and journal an active HTTP response, then prevent new calls.
        # A later resume reuses all completed stages; a hard kill is unnecessary.
        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, lambda *_: calls.stop.set())
        failures = []
        started = time.time()
        try:
            with cf.ThreadPoolExecutor(args.workers) as pool:
                jobs = {pool.submit(generate_game, p, args, student, calls): p for p in paths}
                for future in cf.as_completed(jobs):
                    try:
                        future.result()
                    except Exception as e:
                        calls.stop.set()
                        failures.append({"game": jobs[future].name, "error": str(e)})
        finally:
            rows = [row for p in paths for row in export_game(p, args, student)]
            summary = {"completed": len(rows), "eligible": sum(r["training_eligible"] for r in rows),
                       "excluded": [r["key"] for r in rows if not r["training_eligible"]],
                       "refinement1": sum(r["refinements"] >= 1 for r in rows),
                       "refinement2": sum(r["refinements"] >= 2 for r in rows),
                       "monitored": sum(r["monitoring_sample"] for r in rows),
                       "monitoring_unavailable": sum(bool(r.get("monitoring_exception")) for r in rows),
                       "monitoring_exceptions": [{"key": r["key"], "detail": r["monitoring_exception"]}
                                                for r in rows if r.get("monitoring_exception")],
                       "estimated_cost_including_uncertain": calls.cost(), "new_calls": calls.new_calls,
                       "invocation_seconds": time.time() - started, "failures": failures}
            atomic_json(args.out / "summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
