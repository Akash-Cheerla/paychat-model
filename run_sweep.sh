#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
DIR="$1"; SP="$2"; LIMIT="${3:-400}"
FIRST=1
for T in 0.53 0.40 0.30 0.20 0.12 0.07 0.03; do
  MODEL_DIR="$PWD/saved_model" PAYCHAT_CONV_MODEL="$PWD/$DIR" \
    PAYCHAT_CONV_CLASSIFIER=1 PAYCHAT_CONV_THRESHOLDS="money=$T,ride=$T" \
    nohup python -m uvicorn app:app --host 127.0.0.1 --port 8070 > "$SP/sweep_$T.log" 2>&1 &
  for i in $(seq 1 150); do sleep 2; curl -s -m 3 http://127.0.0.1:8070/health >/dev/null 2>&1 && break; done
  if [ $FIRST -eq 1 ]; then
    python sweep_thresholds.py --url http://127.0.0.1:8070/detect --limit "$LIMIT" --tag "$T" --header
    FIRST=0
  else
    python sweep_thresholds.py --url http://127.0.0.1:8070/detect --limit "$LIMIT" --tag "$T"
  fi
  PID=$(netstat -ano | grep ":8070 " | grep LISTENING | awk '{print $5}' | head -1)
  [ -n "$PID" ] && taskkill //F //PID "$PID" //T >/dev/null 2>&1
  sleep 2
done
echo "[sweep done]"
