#!/usr/bin/env bash
# v5 vs v4 on group_eval_v6 — 787 held-out group conversations, 3-6 speakers.
#
# Everything in one script and one background task: a previous attempt lost its servers
# and its eval when the session tore down, because they were three separate tasks and
# nothing owned the sequence. Here the servers are children of this script, each model
# runs to completion before the next starts, and both servers are killed at the end.
set -u
cd "$(dirname "$0")"
OUT="${1:?usage: run_groupv6_eval.sh <output-dir>}"
mkdir -p "$OUT"

wait_for_port () {
  for _ in $(seq 1 100); do
    curl -s -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    python -c "import time;time.sleep(5)"
  done
  echo "server on :$1 never came up" >&2; return 1
}

run_model () {  # $1=conv model dir  $2=port  $3=label  $4=save file
  echo "[$(date +%H:%M)] starting $1 on :$2"
  python serve_conv_port.py --model "$1" --port "$2" > "$OUT/server_$3.log" 2>&1 &
  local pid=$!
  wait_for_port "$2" || { kill $pid 2>/dev/null; return 1; }
  echo "[$(date +%H:%M)] scoring $3"
  python run_eval_server.py --file group_eval_v6.json --label "$3" \
      --url "http://127.0.0.1:$2/detect" --save "$4" > "$OUT/$3.txt" 2>&1
  tail -8 "$OUT/$3.txt"
  kill $pid 2>/dev/null
  python -c "import time;time.sleep(3)"
}

run_model conv_model    8002 v5_groupv6 RUN_groupv6_v5.json
run_model conv_model_v4 8003 v4_groupv6 RUN_groupv6_v4.json
echo "=== both done $(date +%H:%M) ==="
