#!/usr/bin/env bash
# Is the run still going?  Prints uptime, the engine_re runner PIDs (with elapsed time), and whether
# <run dir>/summary.md exists (the runner writes it when every game has finished).
#   from ARC3-Inference: bash ../.claude/skills/monitor-run/alive.sh runs/engine-play/<run>
# An uptime below ~30 min means the VM was reclaimed and restarted: no process survives that.
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ -z "$1" ]; then
  sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0
fi
run="${1%/}"
echo "uptime: $(uptime)"
procs=$(ps -eo pid,etime,cmd | grep -E 'engine_re\.(run_|kernel)' | grep -v grep)
if [ -n "$procs" ]; then
  echo "processes (pid, elapsed, cmd):"; echo "$procs" | cut -c1-200
  echo "$procs" | grep -q -- "$(basename "$run")" || echo "WARNING: no runner process names $(basename "$run")"
else
  echo "processes: none"
fi
if [ -f "$run/summary.md" ]; then echo "summary.md: exists (run finished)"; else echo "summary.md: missing (run unfinished)"; fi
for g in "$run"/*/result.json; do
  [ -f "$g" ] && echo "$(basename "$(dirname "$g")"): result.json modified $(date -u -r "$g" +%H:%M:%SZ)"
done
echo "now: $(date -u +%H:%M:%SZ)"
