#!/usr/bin/env bash
# One model at a time: start, wait until healthy, score, stop.
#
# Three servers at once got killed mid-run last time (no crash in the logs, just gone —
# most likely memory pressure from three RoBERTa + spaCy processes). score_human_eval
# swallows per-request exceptions and keeps going, so a dead server does not fail the
# run, it silently produces "never fired" for every turn. Sequential is slower and
# cannot lie in that particular way.
set -u
cd "$(dirname "$0")"
SP="$1"        # scratchpad dir for logs
declare -A DIRS=( [v5]=conv_model [v8]=conv_model_v8 [v9]=conv_model_v9 )

for M in v5 v8 v9; do
  echo "=== $M (${DIRS[$M]}) ==="
  MODEL_DIR="$PWD/saved_model" PAYCHAT_CONV_MODEL="$PWD/${DIRS[$M]}" \
    PAYCHAT_CONV_CLASSIFIER=1 \
    nohup python -m uvicorn app:app --host 127.0.0.1 --port 8030 > "$SP/he_$M.log" 2>&1 &
  for i in $(seq 1 120); do
    sleep 2
    curl -s -m 3 http://127.0.0.1:8030/health >/dev/null 2>&1 && break
  done
  ACT=$(curl -s -m 5 -X POST http://127.0.0.1:8030/classify \
        -H "Content-Type: application/json" \
        -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
        | python -c "import json,sys;print((json.load(sys.stdin).get('model_version') or {}).get('conv_dir'))" 2>/dev/null)
  echo "  serving conv_dir=$ACT"
  if [ "$ACT" != "${DIRS[$M]}" ]; then
    echo "  WRONG MODEL — refusing to score"; else
    python score_human_eval.py --url http://127.0.0.1:8030/detect --label "$M" --save "HE2_$M.json"
  fi
  PID=$(netstat -ano | grep ":8030 " | grep LISTENING | awk '{print $5}' | head -1)
  [ -n "$PID" ] && taskkill //F //PID "$PID" //T >/dev/null 2>&1
  sleep 3
done
echo "[all done]"
