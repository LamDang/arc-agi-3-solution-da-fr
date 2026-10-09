"""Finish the one sampled turn whose optional code regeneration was filtered.

This narrowly scoped recovery preserves the three original provider-error
attempts, reuses finalized thinking/history, runs the sampled thinking audit
without a regenerated snippet, and records that code equivalence was unavailable.
"""
from __future__ import annotations

import json
import fcntl
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ARC3-Inference"))

from think_gen import checks, context, logs, prompts  # noqa: E402
from think_gen.progressive import (  # noqa: E402
    DurableCalls,
    StudentFormat,
    api_history,
    atomic_json,
    combined_prompt,
    digest,
    build_manifest,
    issue,
    monitoring_indices,
    read_json,
    validate_judge,
)
from think_gen import client  # noqa: E402


OUT = ROOT / "ARC3-Inference/runs/think-progressive-sampled-pilot"
RUN = ROOT / "ARC3-Inference/runs/gpt61sol-features-25games"
TOKENIZER = ROOT / "data/sft-gpt61sol-features-25games/qwen/tokenizer.json"
TEMPLATE = ROOT / "data/sft-gpt61sol-features-25games/qwen/chat_template.jinja"
GAME_PREFIX = "sk48"
TURN_INDEX = 41


def main() -> None:
    lock = (OUT / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("A generation process holds the run lock") from exc
    if read_json(OUT / "driver_status.json").get("phase") != "stopped":
        raise RuntimeError("Recovery requires a stopped driver")
    manifest = read_json(OUT / "manifest.json")
    settings = manifest["settings"]
    expected_settings = {
        "model": "qwen/qwen3.8-flash",
        "provider": "Alibaba",
        "judge_model": "gpt-6.1-sol",
        "max_tokens": 8192,
        "sol_max_tokens": 16000,
        "input_limit": 120000,
        "cache_retention": "default",
        "sol_prices": [2.0, 0.1, 10.0],
        "monitor_per_game": 15,
    }
    if settings != expected_settings:
        raise RuntimeError(f"Unexpected run settings: {settings}")
    manifest_args = SimpleNamespace(run=RUN, games=None, tokenizer=TOKENIZER, template=TEMPLATE, **settings)
    expected_manifest = build_manifest(manifest_args, logs.request_logs(RUN))
    if manifest != expected_manifest:
        raise RuntimeError("Current source/code/tokenizer/settings do not match the saved manifest")
    paths = logs.request_logs(RUN, [GAME_PREFIX])
    if len(paths) != 1:
        raise RuntimeError(f"Expected one source game log, found {len(paths)}")
    records = logs.read_log(paths[0])
    rec = records[TURN_INDEX]
    if rec.index != TURN_INDEX or rec.game != "sk48-d8078629_p0":
        raise RuntimeError(f"Unexpected sampled turn identity: {rec.key}")
    folder = OUT / "turns" / rec.game / f"{rec.index:05d}"
    final_path = folder / "final.json"
    if final_path.exists():
        existing = read_json(final_path)
        if existing.get("key") == rec.key and existing.get("monitoring_exception"):
            print(json.dumps({"recovered": rec.key, "already_finalized": True}))
            return
        raise RuntimeError("Turn is already finalized with an unexpected row; refusing to overwrite it")
    if TURN_INDEX not in monitoring_indices(len(records), settings["monitor_per_game"]):
        raise RuntimeError("Turn is not in the configured production-monitoring panel")

    regen = folder / "calls/regen"
    attempts = sorted(regen.glob("attempt-*.json"))
    if [p.name for p in attempts] != [f"attempt-{i:02d}.json" for i in range(3)]:
        raise RuntimeError("Expected exactly the three original regen attempts")
    for path in attempts:
        attempt = read_json(path)
        if attempt.get("state") != "error_uncertain" or "data_inspection_failed" not in attempt.get("error", ""):
            raise RuntimeError(f"Unexpected regen attempt state at {path}")
    saved_regen_complete = regen / "complete.json"

    # Rebuild the exact in-game finalized history and verify every reference
    # against the checkpoint before making the single audit call.
    history: dict[str, str] = {}
    chunk = -1
    last_context: list[str] = []
    for earlier in records[:TURN_INDEX]:
        current = [digest(m) for m in context.history(earlier.messages, {})]
        if not last_context or current[:len(last_context)] != last_context:
            chunk += 1
        last_context = current
        p = OUT / "turns" / earlier.game / f"{earlier.index:05d}" / "final.json"
        if not p.exists():
            raise RuntimeError(f"Missing prior finalized turn required for history: {earlier.key}")
        prior_row = read_json(p)
        if prior_row.get("key") != earlier.key:
            raise RuntimeError(f"Prior turn key mismatch: {earlier.key}")
        history[prior_row["ref"]] = prior_row["thinking"]
    current = [digest(m) for m in context.history(rec.messages, {})]
    if not last_context or current[:len(last_context)] != last_context:
        chunk += 1
    refs = [context.message_ref(m) for m in rec.messages if m["role"] == "assistant"]
    prior = {ref: digest(history[ref]) for ref in refs}
    history_hash = digest(prior)

    refine1 = read_json(folder / "calls/refine1/complete.json")["value"]
    verdict = read_json(folder / "calls/judge2/complete.json")["value"]
    if (folder / "calls/refine2/complete.json").exists():
        raise RuntimeError("Unexpected refine2 completion; manual review required")
    prompt, target_words = prompts.reconstruct_prompt_b4(rec.reply, rec.reasoning_tokens)
    advisory = checks.check(refine1, rec.reply, json.dumps(rec.messages, ensure_ascii=False), target_words // 2)
    account = prompts.python_args(rec.reply)
    require_points = bool(account.get("reasoning") or account.get("description") or rec.summary)
    verdict = validate_judge({"content": json.dumps(verdict)}, False, require_points)
    if issue(verdict, advisory):
        raise RuntimeError("The saved judge/checks require a second refinement; refusing to finalize")

    manifest = read_json(OUT / "manifest.json")
    settings = manifest["settings"]
    args = SimpleNamespace(
        out=OUT,
        max_tokens=settings["max_tokens"],
        sol_max_tokens=settings["sol_max_tokens"],
        judge_model=settings["judge_model"],
        model=settings["model"],
        provider=settings["provider"],
        sol_prices=settings["sol_prices"],
        budget=300.0,
        cache_retention=settings["cache_retention"],
        max_attempts=15,
        retry_uncertain=False,
    )
    student = StudentFormat(TOKENIZER, TEMPLATE)
    messages, tools = student.messages(rec, history)
    input_tokens = student.count(messages, tools)
    if input_tokens > 120000:
        training_eligible = False
    else:
        training_eligible = True
    flash = api_history(messages, "flash")
    regen_request = {
        "api": "flash",
        "messages": flash + [{"role": "user", "content": refine1 + "\n\n" + prompts.REGEN_NOTE}],
        "tools": tools,
        "cap": args.max_tokens,
        "model": args.model,
        "reasoning": True,
        "regen": True,
        "cache_key": None,
    }
    regen_hash = digest(regen_request)
    if any(read_json(p).get("request_hash") != regen_hash for p in attempts):
        raise RuntimeError("Original filtered attempts do not match reconstructed request")
    # DurableCalls treats this validated terminal outcome as an empty optional
    # code result, allowing the normal sampled audit to proceed without claiming
    # that code equivalence was checked. The raw 400 responses remain preserved.
    if saved_regen_complete.exists():
        saved = read_json(saved_regen_complete)
        if (saved.get("request_hash") != regen_hash or saved.get("value", "not-null") is not None
                or saved.get("terminal_outcome") != "provider_content_filter"):
            raise RuntimeError("Existing regen completion does not match the validated terminal outcome")
    else:
        atomic_json(saved_regen_complete, {
            "request_hash": regen_hash,
            "value": None,
            "attempt": attempts[-1].name,
            "terminal_outcome": "provider_content_filter",
        })
    sol = api_history(messages, "sol")
    calls = DurableCalls(OUT, args)
    audit = calls.call(
        folder,
        "final_audit",
        "sol",
        sol + [{"role": "user", "content": combined_prompt(rec, refine1, None)}],
        [],
        lambda response: validate_judge(response, False, require_points),
        cache_key=f"think-progressive-{rec.game}-{chunk}",
    )
    exception = (
        "Regenerated code-equivalence check unavailable: Alibaba content inspection rejected code "
        "regeneration after three HTTP 400 responses. The sampled final audit evaluated the thinking "
        "against the teacher record without regenerated code."
    )
    row = {
        "key": rec.key,
        "game": rec.game,
        "index": rec.index,
        "chunk": chunk,
        "ref": context.message_ref(rec.reply),
        "thinking": refine1,
        "status": "ok",
        "history_hash": history_hash,
        "history_refs": prior,
        "refinements": 1,
        "student_input_tokens": input_tokens,
        "training_eligible": training_eligible,
        "exclusion": None if training_eligible else "student_input_over_limit",
        "checks": advisory,
        "monitoring_sample": True,
        "last_refinement_judge": verdict,
        "last_judge_applies_to_final": True,
        "final_judge": audit,
        "regenerated_code": None,
        "monitoring_exception": exception,
    }
    atomic_json(final_path, row)
    print(json.dumps({"recovered": rec.key, "chunk": chunk, "input_tokens": input_tokens,
                      "training_eligible": training_eligible, "audit_sections": sorted(audit),
                      "equivalence_check": "unavailable after three provider content-filter rejections",
                      "new_audit_cost_including_reservations_usd": calls.cost()}, indent=2))


if __name__ == "__main__":
    main()

