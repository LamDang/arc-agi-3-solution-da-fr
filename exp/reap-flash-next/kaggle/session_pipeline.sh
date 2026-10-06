#!/bin/bash
# Overnight chain: calibration -> pruned server -> stream benchmark -> games -> wait for results to be fetched -> stop.
cd /kaggle/working
echo "PIPELINE start $(date +%H:%M)"
while kill -0 "$(cat calib_run.pid)" 2>/dev/null && ! grep -q CALIB_DONE calib_run.log; do sleep 30; done
grep -q CALIB_DONE calib_run.log || { echo "PIPELINE calibration did not finish; stopping"; exit 1; }
echo "PIPELINE calibration done $(date +%H:%M)"
/usr/bin/python3 -u serve_pruned.py > serve_pruned.log 2>&1
grep -q SERVER_READY serve_pruned.log || { echo "PIPELINE server failed; see serve_pruned.log"; tail -20 serve_pruned.log; exit 1; }
echo "PIPELINE server ready $(date +%H:%M)"
/usr/bin/python3 -u bench_streams.py > bench_streams.log 2>&1
echo "PIPELINE bench done $(date +%H:%M): $(grep -c '\[batch\]' bench_streams.log) rows"
/usr/bin/python3 -u pick_streams.py
echo "PIPELINE games start $(date +%H:%M) with $(cat streams.txt) streams"
/usr/bin/python3 -u games.py > games.log 2>&1
echo "GAMES_STATUS $? $(date +%H:%M)"
# results are copied off the notebook from outside; wait up to 2 h for that, then free the GPU and end the session
for i in $(seq 240); do [ -f fetched.flag ] && break; sleep 30; done
echo "PIPELINE stopping $(date +%H:%M) (fetched: $([ -f fetched.flag ] && echo yes || echo no))"
pkill -f 'sglang serve'; sleep 30
pkill -f jupyter-server
