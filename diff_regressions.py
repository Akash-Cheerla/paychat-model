"""Per-turn replay of the conversations v5 fires and v11 does not.

The conversation-level diff says v11 loses 84 should-fire conversations. It does not say
WHICH MESSAGE it fails on, and the eval's `fire` label cannot answer that either - the
label sits on the last turn of the conversation, not on the commitment. Clustering by the
labelled text put 57 of 84 into "other" and showed things like "anyway what time we
meeting saturday" as the message that should have fired.

So replay both models turn by turn over just those conversations and record where each
one fires. The message v5 fires on IS the commitment, by definition of it being right.

    python diff_regressions.py --v5 http://127.0.0.1:8300/detect --out data/eval/REG_v5.json
"""
import argparse, json, sys, io, time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
MR = ("money", "ride")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="data/eval/REG_IDS.json")
    a = ap.parse_args()

    convs = json.loads((ROOT / "data/eval/human_eval.json").read_text(encoding="utf-8"))
    ids = json.loads((ROOT / a.ids).read_text(encoding="utf-8"))
    tag = f"rg_{int(time.time())}"
    out = {}
    for k, ci in enumerate(ids):
        c = convs[ci]
        room = f"{tag}_{ci}"
        turns = []
        for ti, t in enumerate(c["turns"]):
            d = requests.post(a.url, timeout=60, json={
                "text": t["text"], "room_id": room, "sender": str(t["sender"]),
                "message_id": f"{tag}_{ci}_{ti}"}).json()
            got = [i for i in (d.get("intents") or []) if i in MR]
            sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
            turns.append({
                "ti": ti, "sender": t["sender"], "text": t["text"],
                "fired": got,
                "money": round(float(sc.get("money", 0) or 0), 3),
                "ride": round(float(sc.get("ride", 0) or 0), 3),
            })
        out[str(ci)] = {"scenario": c["scenario"], "turns": turns}
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(ids)}")
    (ROOT / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"  wrote {a.out}  ({len(out)} conversations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
