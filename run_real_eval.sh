#!/usr/bin/env bash
# Both models through the same real-data suite, sequentially.
#
# Sequential on purpose: two servers at once ran 117 turns/min each against 285 for one
# alone. A leftover scorer competing for the machine already cost an hour today.
set -u
cd "$(dirname "$0")"
R="$PWD"
SP="$1"
CAP="${2:-600}"

kill_port () {
  local P
  P=$(netstat -ano | grep ":$1 " | grep LISTENING | awk '{print $5}' | head -1)
  [ -n "$P" ] && taskkill //F //PID "$P" //T >/dev/null 2>&1
  sleep 2
}

run_one () {
  local NAME="$1" DIR="$2" PORT="$3" OUT="$4"
  kill_port "$PORT"
  MODEL_DIR="$R/saved_model" PAYCHAT_CONV_MODEL="$R/$DIR" PAYCHAT_CONV_CLASSIFIER=1 \
    nohup python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" \
    > "$SP/rev_$NAME.log" 2>&1 &
  for i in $(seq 1 200); do
    sleep 2
    curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  done
  # Identity check: a stale server answered a whole round once and voided the table.
  local ID
  ID=$(curl -s -m 20 -X POST "http://127.0.0.1:$PORT/classify" \
       -H "Content-Type: application/json" \
       -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
       | python -c "import json,sys;print((json.load(sys.stdin).get('model_version') or {}).get('conv_dir'))" 2>/dev/null)
  if [ "$ID" != "$DIR" ]; then
    echo "  ABORT $NAME - wrong model on :$PORT (got $ID, wanted $DIR)"
    return 1
  fi
  python real_eval.py --url "http://127.0.0.1:$PORT/detect" --label "$NAME" \
                      --cap "$CAP" --out "$OUT"
  kill_port "$PORT"
}

run_one v5  conv_model     8701 data/eval/REALEVAL_v5.json
run_one v11 conv_model_v11 8702 data/eval/REALEVAL_v11.json
echo "[real eval done]"
