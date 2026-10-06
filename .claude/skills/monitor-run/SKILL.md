---
name: monitor-run
description: Launch, watch and wrap up a long experiment under ARC3-Inference from this cloud session. Use it for an engine_re play run (engine_re.run_play) or history run, or any OpenRouter eval, that is started here and checked about every 30 minutes until it finishes. It covers launching so the run survives a VM reclaim, the 30-minute check-in loop (process check, a read-only monitor subagent, rebuilding the transcript page, re-arming), the monitor brief, two status scripts, and the wrap-up (write-up, dvc add/push, commit).
---

# Monitoring a long run

Paths are relative to `ARC3-Inference/` unless they start with `../`. Setup, limits and DVC for
local evals: `LOCAL_EVAL.md`. Known harness defects (do not re-report): `exp/v11-followups.md`.

## 1. Launch so the run survives

The VM is reclaimed when the session idles. It comes back with the disk but no processes. A run
launched detached (`setsid`, `nohup ... & disown`) died four times in one morning. Launch it as a
harness-tracked background Bash task instead:

- `run_in_background: true`, `timeout: 7200000` (the 2-hour maximum), `dangerouslyDisableSandbox: true`
- command:
  `cd ARC3-Inference && exec uv run --no-sync python -m engine_re.run_play --games sp80,ls20 --out runs/engine-play/<run> --model <model> --max-turns 300 --max-minutes 240 --max-cost 6 --max-actions 500 --label <run> >> runs/engine-play/<run>/run.log 2>&1 < /dev/null`
- The task re-invokes the session when it exits (finished, killed at the 2-hour timeout, or
  crashed). Relaunch then with the same command: the engine_re runners resume each game from
  `result.json` and `transcript.jsonl`.

To stop a run, never `pkill -f <out dir>`: the pattern matches the shell running your own
command, and that kills everything. Find the PIDs with
`ps -eo pid,etime,cmd | grep engine_re.run_play | grep -v grep`, then kill by PID: the python
process first, then uv, then any leftover `engine_re.kernel` children.

## 2. The check-in loop (every ~30 minutes)

Use a one-shot `send_later` (or `update_trigger` with `run_once_at` 30 minutes ahead), re-armed at
each firing. Its stored prompt must carry everything verbatim, because the session may have been
reclaimed and compacted:
- the out dir and the comparison run;
- the restart command from section 1;
- the transcript-page rebuild command with its full `--title` and `--lede`;
- the artifact URL.

Each firing:
1. `bash ../.claude/skills/monitor-run/alive.sh runs/engine-play/<run>`. An uptime under 30 min
   means the VM was swapped. If no runner process is listed and summary.md is missing, relaunch
   (section 1).
2. Spawn the monitor subagent (Opus, in the background) with the brief in section 3.
3. Rebuild and republish the transcript page:
   `uv run --no-sync python engine_re/tools/play_transcripts/build.py runs/engine-play/<run> <out.html> --compare runs/engine-play/<previous run> --title "..." --lede "..."`,
   then call the Artifact tool with the same file path and `url`.
4. Re-arm the one-shot.

Stop when `runs/engine-play/<run>/summary.md` exists. Then:
- do a final page rebuild;
- write `exp/<run>.md` and add rows to `exp/README.md`;
- run `uv run --no-sync dvc add runs/engine-play/<run>` and `uv run --no-sync dvc push runs/engine-play/<run>.dvc`;
- commit and push.

The stop hook refuses an idle turn with uncommitted changes, so commit small things as they come
(follow-up notes, the .dvc pointer).

## 3. Monitor subagent brief (template)

Fill in the `<...>` fields. Keep the known-defects list current: copy it from `exp/<run>-followups.md`
or `exp/v11-followups.md`.

```
You are monitoring a running experiment. Repo: /home/user/arc-agi-3-solution-da-fr, code under
ARC3-Inference/ (run commands from there). Run: runs/engine-play/<run> (games <games>, model <model>,
harness engine_re/run_play.py + engine_re/play_agent.py), running since <start> UTC (PIDs <pids>).
Last check (<time>): <per game: turn, actions (per level), levels, score, cost, phase>.
Limits: <turns> turns, <minutes> min, $<cost>, <actions> actions. Comparison run: runs/engine-play/<prev>.

Do read-only checks and report. Do not edit anything under runs/. Never run the OpenRouter model yourself.
Start with: uv run --no-sync python ../.claude/skills/monitor-run/status.py runs/engine-play/<run>
  --compare runs/engine-play/<prev> --last 10 --since-turn <game>=<turn at last check>,...
1. Process: `ps -eo pid,etime,cmd | grep engine_re.run_play | grep -v grep`. If it is dead, read the
   tail of run.log for a traceback. Report it; do NOT restart it; never `pkill -f` the out dir.
2. Per game: result.json (status, turns, actions per level, levels, score, batches, mismatches,
   phase, usage.cost_usd, completion tokens, max prompt tokens). In transcript.jsonl, look for:
   - harness errors: tracebacks from harness code (engine_re/...), not from the model's code or
     engine.py;
   - repeated identical turns;
   - nudges and plan_nudges;
   - compaction events;
   - kernel timeouts ("Timed out after 120s").
   run.log tail: the "analyzer endpoint unreachable" proxy retries are normal.
3. Story per game: progress since the last check and compared with <prev>. For each level solve,
   say how the route was found (a search over replica.step/pour or by hand) and which mechanic
   was missing. Check whether the replica predicted the solve: the move record (ok, verdict,
   level_solved, support.weakest) and the support lines. For the first PLAN of a new level, give
   its unfamiliar elements. End with what the model did in the last <N> turns, one line each.
   <game-specific questions>
4. Feature deltas since the last check, one line each:
   - support lines in commit_moves output and in move records;
   - animation notes;
   - notes.md in PLAN messages and its updates;
   - prediction warnings;
   - replica.step / replica.pour / make_level use;
   - traced() / support() use;
   - edit_file-as-tool failures;
   - kernel timeouts;
   - max prompt tokens.
5. Known defects, do not re-report: <list>. Report new ones only.
Only for a harness bug that blocks the run: fix it, run
`uv run --no-sync python -m pytest tests/test_play.py tests/test_engine_re.py -q`, commit on
<branch> with a message ending in the session's attribution trailer lines (no model identifiers
elsewhere), run `git push -u origin <branch>`, and say so, so the run can be restarted by PID.
Report in under 30-35 lines: a status table per game, a short story per game (with the last N
turns), feature deltas, anything wrong.
```

## 4. Scripts

Both are read-only. Run them from `ARC3-Inference`; `--help` shows the options.

- `uv run --no-sync python ../.claude/skills/monitor-run/status.py <run dir> [--compare <run dir>] [--last N] [--since-turn g=T,...]`
  prints:
  - the per-game status table, with max prompt tokens taken from the usage records;
  - the plan/fit split in turns and output tokens;
  - feature and event counts (`--since-turn` limits them to recent turns);
  - error counts by kind, splitting model/engine-code tracebacks from possible harness ones;
  - for each level solve: whether the replica predicted it, and the turns and output tokens until
    the next commit and the next batch;
  - the last N model turns, one line each.
- `bash ../.claude/skills/monitor-run/alive.sh <run dir>` prints uptime, the engine_re processes,
  whether summary.md exists, and when each result.json last changed.

Sample (`status.py runs/engine-play/qwen38flash-v11-a --compare runs/engine-play/qwen38flash-v10 --last 3 --width 80`, abridged):

```
== runs/engine-play/qwen38flash-v11-a
game  status        turns acts per-level              lvls score batch mism phase cost$  out_tok  maxprompt min
ls20  budget_turns    300  231 13 65 153 0 0 0 0           2  10.7    36   19 fit    1.94  468,632   296,762   136
sp80  budget_turns    300  122 11 19 21 71 0 0             4  47.6    26   21 fit    1.96  612,788   289,211   166
summary.md: exists (run finished)
== runs/engine-play/qwen38flash-v10
ft09  won             156   77 4 7 14 16 23 13             6 100.0    16   11 plan   0.60  289,302   147,354    70
...
--- sp80
phase split: plan 177 turns / 341,156 out, fit 123 turns / 271,632 out
counts: ... edit_file called as a tool 55, kernel timeouts 3, make_level( 33, plan_nudge 19, replica.pour( 63, replica.step( 94, support blocks 26 ...
errors: traceback (model or engine code) 24, TypeError 9, NameError 7, IndexError 3, ...
level solve T210 step 51 (replica predicted the solve: True): next commit T217 (+4,490 out), next batch T248 (+55,078 out)
last 3 turns:
  T298 plan s111 commit_moves   Level 3 part 3: nudge the 4x1 to (9,9) so the joined piece's left run drops into
  T299 plan s121 commit_moves   The pour: my replica says this arrangement (diverter at (5,4), joined 3+cap+3 pi
  T300 fit  s122 python         GridGuess(width=20, height=20, scale=3, x_offset=2, y_offset=2, border=1, frames
```

`alive.sh runs/engine-play/qwen38flash-v11-a` (after the run finished):

```
uptime:  09:56:08 up  1:59,  0 user,  load average: 0.07, 0.13, 0.17
processes: none
summary.md: exists (run finished)
ls20: result.json modified 09:14:18Z
sp80: result.json modified 09:44:28Z
```

Script caveats:
- A "harness? read it" traceback count is a lead, not a verdict: read the output.
- The `ok` field is False at every level solve, because the next level's frame is not drawn yet.
  "Replica predicted the solve" therefore checks that the verdict has no "levels completed: the
  game says n, your replica m".

## 5. Lessons

- **VM reclaim:** an idle session loses every process. Only a harness-tracked background task
  wakes the session when the run dies (section 1).
- **`pkill -f <out dir>`** kills your own shell along with the run. Kill by PID.
- **120 s kernel timeout:** the model's python kernel restarts and all its variables are lost.
  Count `Timed out after` in tool outputs; long searches over deep copies of the replica hit it.
- **Resume:** it replays the kernel cells, and some fail (NameError, IndexError) when engine.py
  changed since. Harmless: the `replay` record lists them.
- **Misleading verdict on a predicted level solve:** when the replica predicts a solve and the game
  does not, `make_level(n+1)` raises IndexError and the message says "your replica raised an
  error". This, and the other known defects, are in [exp/v11-followups.md](../../../ARC3-Inference/exp/v11-followups.md).
