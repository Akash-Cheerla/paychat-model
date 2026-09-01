#!/usr/bin/env bash
# Everything that must still hold before the link goes to strangers.
#
# Changed today: ride request patterns (_REQUEST_OF_OTHER, which feeds _resolve_payer and
# therefore the duplicate-fire key), the root redirect, DEMO_DIR, and the logging
# counters. Only the patterns can move firing, but the full set runs anyway - the whole
# point of a preflight is not deciding in advance which change was harmless.
#
# One server, nothing else running. A leftover scorer competing for the machine already
# cost an hour today and made the timings meaningless.
set -u
cd "$(dirname "$0")/.."
R="$PWD"
SP="$1"
PORT=8900

P=$(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$P" ] && taskkill //F //PID "$P" //T >/dev/null 2>&1
sleep 2

MODEL_DIR="$R/saved_model" PAYCHAT_CONV_MODEL="$R/conv_model" PAYCHAT_CONV_CLASSIFIER=1 \
  nohup python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "$SP/pre.log" 2>&1 &
for i in $(seq 1 200); do
  sleep 2
  curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
done

ID=$(curl -s -m 20 -X POST "http://127.0.0.1:$PORT/classify" \
     -H "Content-Type: application/json" \
     -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
     | python -c "import json,sys;print((json.load(sys.stdin).get('model_version') or {}).get('conv_dir'))" 2>/dev/null)
echo "=== serving $ID ==="
[ "$ID" != "conv_model" ] && { echo "WRONG MODEL - aborting"; exit 1; }

echo
echo "--- routes ---"
for p in / /chat /demo /health /log-status; do
  printf "  %-12s %s\n" "$p" "$(curl -s -o /dev/null -w '%{http_code}' -m 8 "http://127.0.0.1:$PORT$p")"
done

echo
echo "--- Andril's misfire report ---"
python tests/test_gowtham_ride_misfire.py --url "http://127.0.0.1:$PORT/detect" 2>&1 | tail -3

echo
echo "--- the four real dogfood false fires ---"
python tests/test_dogfood_false_fires.py --url "http://127.0.0.1:$PORT/detect" 2>&1 | tail -6

echo
echo "--- non-acceptance battery ---"
python tests/test_nonacceptance.py --url "http://127.0.0.1:$PORT/detect" 2>&1 | tail -3

echo
echo "--- who gets the prompt ---"
python tests/test_who_gets_prompt.py --url "http://127.0.0.1:$PORT/detect" 2>&1 | grep -E "FIRES|show_to" | head -6

echo
echo "--- full frozen set, 1315 conversations ---"
python tests/score_human_eval.py --url "http://127.0.0.1:$PORT/detect" --label preflight \
  --save FULL_preflight.json 2>&1 | grep -E "FIRES|STAYS|overall|false prompts"

P=$(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$P" ] && taskkill //F //PID "$P" //T >/dev/null 2>&1
echo
echo "[preflight done]"
