#!/bin/bash
# Follow-up runs after pipeline.sh, same 7 games x 4 passes:
#   B: 256 experts with 10 streams (same server), C: full model with 10 streams (dfranzen's settings, mem 0.93).
cd /kaggle/working
echo "PIPELINE2 start $(date +%H:%M)"
while ! grep -q "PIPELINE finished\|PIPELINE .*failed\|did not finish" pipeline.log 2>/dev/null; do sleep 60; done
grep -q "PIPELINE finished" pipeline.log || { echo "PIPELINE2 first pipeline failed; nothing to do"; exit 1; }
grep -q SERVER_READY serve_pruned.log || { echo "PIPELINE2 no pruned server"; exit 1; }
echo "PIPELINE2 B start $(date +%H:%M): 256 experts, 10 streams"
GAMES_STREAMS=10 GAMES_DIR=games_256_s10 /usr/bin/python3 -u games.py > games_B.log 2>&1
echo "GAMES_B_STATUS $? $(date +%H:%M)"
pkill -f 'sglang serve'
for i in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 3000 ] && ! (echo > /dev/tcp/127.0.0.1/8001) 2>/dev/null && break
  sleep 5
done
echo "PIPELINE2 pruned server stopped $(date +%H:%M); starting the full model"
/usr/bin/python3 -u serve_full.py > serve_full.log 2>&1
grep -q SERVER_READY serve_full.log || { echo "PIPELINE2 full server failed; see serve_full.log"; tail -20 serve_full.log; exit 1; }
echo "PIPELINE2 C start $(date +%H:%M): full model, 10 streams"
GAMES_STREAMS=10 GAMES_DIR=games_full_s10 GAMES_KEEP=512 /usr/bin/python3 -u games.py > games_C.log 2>&1
echo "GAMES_C_STATUS $? $(date +%H:%M)"
echo "PIPELINE2 finished $(date +%H:%M); server left running"
