#!/usr/bin/env bash
# One model at a time: start, verify identity, record decisions, stop.
set -u
cd "$(dirname "$0")"
SP="$1"
declare -A DIRS=( [v5]=conv_model [v8]=conv_model_v8 [v9]=conv_model_v9 )
for M in v5 v8 v9; do
  echo "=== $M (${DIRS[$M]}) ==="
  MODEL_DIR="$PWD/saved_model" PAYCHAT_CONV_MODEL="$PWD/${DIRS[$M]}" \
    PAYCHAT_CONV_CLASSIFIER=1 \
    nohup python -m uvicorn app:app --host 127.0.0.1 --port 8050 > "$SP/rec_$M.log" 2>&1 &
  for i in $(seq 1 150); do
    sleep 2
    curl -s -m 3 http://127.0.0.1:8050/health >/dev/null 2>&1 && break
  done
  ACT=$(curl -s -m 5 -X POST http://127.0.0.1:8050/classify \
        -H "Content-Type: application/json" \
        -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
        | python -c "import json,sys;print((json.load(sys.stdin).get('model_version') or {}).get('conv_dir'))" 2>/dev/null)
  if [ "$ACT" != "${DIRS[$M]}" ]; then
    echo "  WRONG MODEL (serving $ACT) — skipping"
  else
    echo "  serving $ACT"
    python record_decisions.py --url http://127.0.0.1:8050/detect --label "$M"
  fi
  PID=$(netstat -ano | grep ":8050 " | grep LISTENING | awk '{print $5}' | head -1)
  [ -n "$PID" ] && taskkill //F //PID "$PID" //T >/dev/null 2>&1
  sleep 3
done
echo "[recording done]"
