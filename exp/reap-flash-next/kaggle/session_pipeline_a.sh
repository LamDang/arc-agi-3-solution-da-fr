#!/bin/bash
# Run A after the stream benchmark crashed the server at 28 streams: restart the pruned server, play at 20 streams.
cd /kaggle/working
echo "PIPELINE_A start $(date +%H:%M)"
/usr/bin/python3 -u serve_pruned.py > serve_pruned.log 2>&1
grep -q SERVER_READY serve_pruned.log || { echo "PIPELINE_A server failed; see serve_pruned.log"; tail -20 serve_pruned.log; exit 1; }
echo "PIPELINE_A server ready $(date +%H:%M)"
echo "PIPELINE_A games start $(date +%H:%M) with 20 streams"
GAMES_STREAMS=20 /usr/bin/python3 -u games.py > games.log 2>&1
echo "GAMES_STATUS $? $(date +%H:%M)"
# no automatic shutdown: the session and the SGLang server stay up until stopped by hand
echo "PIPELINE_A finished $(date +%H:%M); server left running"
