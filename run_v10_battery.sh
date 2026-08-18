#!/usr/bin/env bash
# Full battery on ONE model. Verifies which model is serving before scoring anything.
set -u
cd "$(dirname "$0")"
M="$1"; DIR="$2"; SP="$3"
MODEL_DIR="$PWD/saved_model" PAYCHAT_CONV_MODEL="$PWD/$DIR" PAYCHAT_CONV_CLASSIFIER=1 \
  nohup python -m uvicorn app:app --host 127.0.0.1 --port 8060 > "$SP/bat_$M.log" 2>&1 &
for i in $(seq 1 150); do sleep 2; curl -s -m 3 http://127.0.0.1:8060/health >/dev/null 2>&1 && break; done
ACT=$(curl -s -m 5 -X POST http://127.0.0.1:8060/classify -H "Content-Type: application/json" \
      -d '{"text":"hi","room_id":"idchk","sender":"1","message_id":"x"}' \
      | python -c "import json,sys;print((json.load(sys.stdin).get('model_version') or {}).get('conv_dir'))" 2>/dev/null)
echo "=== $M : serving $ACT ==="
[ "$ACT" != "$DIR" ] && { echo "WRONG MODEL — aborting"; exit 1; }

echo "--- human_eval (1365 convs) ---"
python score_human_eval.py --url http://127.0.0.1:8060/detect --label "$M" --save "HE2_$M.json" 2>&1 | grep -E "FIRES|STAYS|overall|false prompts"
echo "--- test_screenshots (real dogfood) ---"
python test_screenshots.py --url http://127.0.0.1:8060/detect 2>&1 | grep -E "^  (PASS|FAIL)|CHECK"
echo "--- conversation evals ---"
T=$(date +%m%d%H%M%S)
for S in behaviour_eval ship_eval group_heldout mixed_eval; do
  printf "  %-16s " "$S"
  python run_eval_server.py --file $S.json --label "B${T}_${M}_$S" \
    --url http://127.0.0.1:8060/detect 2>&1 | grep -E "^  (conversations|turns)" | tr -s ' ' | tr '\n' ' '; echo
done
echo "--- gate_coverage ---"
python gate_coverage.py --url http://127.0.0.1:8060/detect 2>&1 | grep -Ei "overall" | tail -1
PID=$(netstat -ano | grep ":8060 " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //F //PID "$PID" //T >/dev/null 2>&1
echo "[$M battery done]"
