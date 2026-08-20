#!/usr/bin/env bash
# One threshold, one port, one server. Run several at once - the sweep is CPU-bound and
# serially it took 48 min per point (6.5 h for eight points).
#
# The fire decision is NOT re-thresholdable offline: on a fire the pipeline calls
# request_meta.mark_fired() and take(), consuming the open request that the
# self-acknowledgement guard reads on later turns. Change the threshold and the state
# machine takes a different path, so every point needs a real server.
set -u
cd "$(dirname "$0")"
T="$1"; PORT="$2"; SP="$3"; LIMIT="${4:-500}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
MODEL_DIR="$PWD/saved_model" PAYCHAT_CONV_MODEL="$PWD/conv_model_v11" \
  PAYCHAT_CONV_CLASSIFIER=1 PAYCHAT_CONV_THRESHOLDS="money=$T,ride=$T" \
  python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "$SP/arm_$T.log" 2>&1 &
for i in $(seq 1 200); do
  sleep 2
  curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
done
# Identity check. A stale server on a port answered a whole eval round once and the
# entire table was void; never score without confirming which model replied. The
# version lives on /classify, not /health.
ID=$(curl -s -m 20 -X POST "http://127.0.0.1:$PORT/classify" \
     -H "Content-Type: application/json" \
     -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
     | python -c "import json,sys;v=(json.load(sys.stdin).get('model_version') or {});print(v.get('conv_dir','?'),v.get('conv_thresholds','?'))" 2>/dev/null)
echo "  [$T] serving $ID"
case "$ID" in
  conv_model_v11*) ;;
  *) echo "  [$T] ABORT - wrong model on :$PORT"
     PID=$(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | head -1)
     [ -n "$PID" ] && taskkill //F //PID "$PID" //T >/dev/null 2>&1
     exit 1;;
esac
python sweep_thresholds.py --url "http://127.0.0.1:$PORT/detect" --limit "$LIMIT" --tag "$T"
PID=$(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //F //PID "$PID" //T >/dev/null 2>&1
echo "  [$T] arm done"
