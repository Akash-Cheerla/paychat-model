#!/usr/bin/env bash
# The pre-push gate: the SAME model (v5) on HEAD code vs the working tree.
#
# Anything that differs is caused by today's app.py changes and nothing else — the
# conversation model, the base model and the thresholds are identical on both ports.
# That is the only comparison that answers "does this break anything in production",
# and it is the one I kept failing to set up today: comparing against numbers measured
# on a server running different code proves nothing.
#
# Labels are unique per run. run_eval_server.py derives room ids from the label, and
# room state lives on the server for 4 hours, so re-using a label replays into rooms
# that still hold the previous run's fires and silently measures leftovers.
#
#   :8007  HEAD (git worktree)     = baseline
#   :8002  working tree            = candidate
#
# Run:  bash run_ship_gate.sh <tag>
set -u
cd "$(dirname "$0")"
TAG="${1:?usage: run_ship_gate.sh <tag>}"
OUT="$TEMP/ship_gate_$TAG"
mkdir -p "$OUT"

run () {   # $1=port $2=arm $3=evalfile
  local label="gate_${TAG}_$2_${3%.json}"
  python run_eval_server.py --file "$3" --label "$label" \
      --url "http://127.0.0.1:$1/detect" --save "RUN_${label}.json" \
      > "$OUT/$2_${3%.json}.txt" 2>&1
  printf '  %-10s %-22s %s\n' "$2" "${3%.json}" \
      "$(grep -E '^  (conversations|turns)' "$OUT/$2_${3%.json}.txt" | tr -s ' ' | tr '\n' ' ')"
}

for SET in behaviour_eval.json group_heldout.json ship_eval.json mixed_eval.json; do
  echo "[$(date +%H:%M)] $SET"
  run 8007 baseline  "$SET"
  run 8002 candidate "$SET"
done
echo "[$(date +%H:%M)] gate complete"
