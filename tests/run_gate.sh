#!/usr/bin/env bash
# Every criterion a candidate has to clear, in one command, with the verdict at the end.
#
#     bash tests/run_gate.sh ./saved_model_v26 /tmp/scratch          # a base candidate
#     bash tests/run_gate.sh ./saved_model ./conv_model_v12 /tmp/scratch   # a conv one
#
# Run it against the CANDIDATE. saved_model/ and conv_model/ are what production loads
# and must not be replaced until this passes. Pointing it AT them read-only is fine and
# is how the baseline is measured.
#
# The criteria are not equal and are not scored together:
#
#   1  base battery      >= 72/75 with ZERO miss. A miss means a request never registers
#                        and the whole conversation is lost; noise is masked by the conv
#                        classifier declining it. Zero-miss is the hard half.
#   2  basics battery    >= 75/82. The base may improve, but end-to-end behaviour must
#                        not regress while it does.
#   3  room replay       no room, in any of the four real logs, loses all its prompts.
#                        Real traffic, and the measurement that survived contact every
#                        time the synthetic ones did not.
#   4  Switchboard       <= 0.54 prompts per 1,000 utterances of 1990s telephone speech
#                        between strangers. The only check here whose data nobody in this
#                        repo wrote. v26 sets that 0.54; a candidate may not raise it.
#
set -u
cd "$(dirname "$0")/.."
R="$PWD"
MODEL="${1:?usage: run_v26_gate.sh <base_model_dir> [conv_model_dir] <scratch_dir>}"
if [ "$#" -ge 3 ]; then CONV="$2"; SP="$3"; else CONV="$R/conv_model"; SP="$2"; fi
PORT=8960
[ -d "$CONV" ] || { echo "no such conv dir: $CONV"; exit 2; }

[ -d "$MODEL" ] || { echo "no such model dir: $MODEL"; exit 2; }
# Pointing this at saved_model/ is fine and useful - it is read-only, and running the
# baseline through the identical harness is the only way the comparison means anything.
# What must not happen is copying a candidate INTO saved_model/ before this passes.
case "$(cd "$MODEL" && pwd)" in
  "$R/saved_model") echo "NOTE: measuring the LIVE model - this is the v25 baseline, not a candidate."; echo;;
esac

P=$(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$P" ] && taskkill //F //PID "$P" //T >/dev/null 2>&1
sleep 2

echo "=== serving $MODEL on :$PORT ==="
MODEL_DIR="$MODEL" PAYCHAT_CONV_MODEL="$CONV" PAYCHAT_CONV_CLASSIFIER=1 \
  nohup python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "$SP/v26.log" 2>&1 &
for i in $(seq 1 200); do
  sleep 2
  curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
done
curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "server never came up - see $SP/v26.log"; exit 2; }

# the model actually loaded must be the one named, or every number below is about
# something else. This check exists because a whole eval round was once voided by it.
ID=$(curl -s -m 20 -X POST "http://127.0.0.1:$PORT/classify" -H "Content-Type: application/json" \
     -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
     | python -c "import json,sys;v=(json.load(sys.stdin).get('model_version') or {});print(v.get('base_dir'),'|',v.get('conv_dir'))" 2>/dev/null)
echo "  serving: $ID"
echo

URL="http://127.0.0.1:$PORT/detect"
FAIL=0

echo "=== 1. base model  (need >= 72/75, ZERO miss) ==="
OUT=$(python tests/test_base_model.py --url "$URL" 2>&1)
echo "$OUT" | grep -E "^  TOTAL|MISS - |^    (ride|money) " | head -14
SCORE=$(echo "$OUT" | grep -oE "TOTAL +[0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | head -1)
MISSES=$(echo "$OUT" | grep -cE "^    (ride|money) +[0-9.]+ < ")
NUM=${SCORE%%/*}
echo "  -> $SCORE, misses=$MISSES"
[ "${NUM:-0}" -ge 72 ] || { echo "  FAILED: below 72"; FAIL=1; }
[ "$MISSES" -eq 0 ] || { echo "  FAILED: $MISSES miss(es) - a miss loses a whole conversation"; FAIL=1; }
echo

echo "=== 2. basics, end to end  (need >= 75/82) ==="
OUT=$(python tests/test_basics.py --url "$URL" 2>&1)
echo "$OUT" | grep -E "^  TOTAL|^  [a-z].*[0-9]+/[0-9]+ +[0-9]+%" | tail -8
SCORE=$(echo "$OUT" | grep -oE "TOTAL +[0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | head -1)
NUM=${SCORE%%/*}
echo "  -> $SCORE"
[ "${NUM:-0}" -ge 75 ] || { echo "  FAILED: below v25's 75/82"; FAIL=1; }
echo

echo "=== 3. real traffic, per room  (no room may go to zero) ==="
for LOG in dogfood_2026-08-11 dogfood_2026-08-21 dogfood_2026-08-30 dogfood_2026-08-31; do
  F="data/eval/$LOG.jsonl"
  [ -f "$F" ] || { echo "  skipping $LOG (not present)"; continue; }
  python tests/replay_rooms.py --log "$F" --url "$URL" 2>&1 | grep -E "rooms |ZERO|REGRESSION|==="
  [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
done
echo

echo "=== 4. conversation nobody here wrote  (need <= 0.54 per 1,000) ==="
# Switchboard: 1990s telephone calls between strangers, no money and no cabs in it.
# The only external evidence we have of what makes us fire wrongly is short
# acknowledgements in unrelated talk - and a conv batch full of short acceptances
# labelled fire pulls the same way. This is the criterion that catches that trade.
SWDA="${PAYCHAT_SWDA:-$SP/swda.zip}"
if [ -f "$SWDA" ]; then
  OUT=$(python tests/test_external_switchboard.py --swda "$SWDA" --url "$URL"         --conversations 40 2>&1)
  echo "$OUT" | grep -E "messages scored|prompts fired|rate "
  RATE=$(echo "$OUT" | grep -oE "rate +: [0-9.]+" | grep -oE "[0-9.]+$")
  WORSE=$(python -c "print(1 if float('${RATE:-99}') > 0.54 else 0)" 2>/dev/null)
  [ "${WORSE:-1}" -eq 0 ] || { echo "  FAILED: $RATE per 1,000, worse than v26's 0.54"; FAIL=1; }
else
  echo "  SKIPPED - no swda.zip. Fetch it:"
  echo "    curl -sL -o \"$SWDA\" https://github.com/cgpotts/swda/raw/master/swda.zip"
  echo "  This criterion is the only external check we have; do not ship without it."
  FAIL=1
fi
echo

P=$(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$P" ] && taskkill //F //PID "$P" //T >/dev/null 2>&1

echo "================================================"
if [ "$FAIL" -eq 0 ]; then
  echo "  PASSED all four. Safe to move into saved_model/."
  echo "  Export ONNX before deploying - latency is 1900ms without it, 90ms with."
else
  echo "  FAILED. Do not ship it."
  echo "  This is the system working: the failure is visible now rather than in"
  echo "  someone's chat next week."
fi
exit $FAIL
