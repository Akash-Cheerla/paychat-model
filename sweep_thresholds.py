"""Score one model at whatever thresholds its server was started with.

Every model since v5 picks its operating point by plateau-centring an F-beta curve that
is nearly flat, so the number it lands on moves between runs on the same data and is not
obviously right. v5 runs at money 0.970 / ride 0.825; v8, v9 and v10 all sit near
0.52-0.56. Nobody has checked which end of that range is better, and three rounds have
been judged — possibly unfairly — at whatever the plateau happened to pick.

It matters most for v10, whose only real problem is recall: 97.9% quiet, 66.1% firing.
Threshold is exactly the knob that trades those. If lowering it restores the firing rate
without giving back the precision, v10 is shippable now instead of after another round.

One line of output per run, so a shell loop can restart the server per setting and
collect a curve. Subset by default — the point is the shape, not a final number.

    PAYCHAT_CONV_THRESHOLDS="money=0.30,ride=0.30" uvicorn app:app ... &
    python sweep_thresholds.py --url http://127.0.0.1:8070/detect --limit 400 --tag 0.30
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
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--header", action="store_true")
    a = ap.parse_args()

    convs = json.loads((ROOT / "data/eval/human_eval.json").read_text(encoding="utf-8"))
    convs = convs[:a.limit] if a.limit else convs

    if a.header:
        fires = sum(1 for c in convs if c["expects_fire"])
        print(f"  {len(convs)} conversations ({fires} fire / {len(convs)-fires} quiet)")
        print(f"  {'thresholds':16} {'fires':>8} {'quiet':>8} {'overall':>9}")

    tag = f"{a.tag}_{int(time.time())}"
    fo = fn = qo = qn = dead = 0
    for ci, c in enumerate(convs):
        room = f"sw_{tag}_{ci}"
        want = next((t["fire"][0] for t in c["turns"] if t.get("fire")), None)
        fired_any = fired_right = False
        for ti, t in enumerate(c["turns"]):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": t["text"], "room_id": room,
                    "sender": str(t.get("sender", "A")),
                    "message_id": f"{tag}_{ci}_{ti}"}).json()
            except Exception:
                dead += 1
                continue
            got = [i for i in (d.get("intents") or []) if i in MR]
            if got:
                fired_any = True
                if want and want in got:
                    fired_right = True
        if c["expects_fire"]:
            fn += 1; fo += fired_right
        else:
            qn += 1; qo += (not fired_any)

    # A failed request looks exactly like "the model stayed silent", which would read as
    # brilliant precision and terrible recall. Say so rather than print a plausible lie.
    if dead:
        print(f"  {a.tag:16} {dead} FAILED REQUESTS — result discarded")
        return 1

    print(f"  {a.tag:16} {fo/max(fn,1):7.1%} {qo/max(qn,1):8.1%} "
          f"{(fo+qo)/max(fn+qn,1):8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
