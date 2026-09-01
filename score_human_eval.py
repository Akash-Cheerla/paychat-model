"""Score a model on human_eval.json as TWO numbers, not one.

A single accuracy figure hides the failure that matters. A model that never fires scores
54% here (all the quiet conversations correct) and is useless; a model that fires on
everything scores 46% and is worse than useless. Both directions have to be reported.

  FIRES WHEN IT SHOULD   of the conversations that contain a real commitment,
                         how many produce a prompt for the right intent
  STAYS QUIET            of the conversations where nobody sends money or books a cab,
                         how many produce no prompt at all

The second is the one nobody has been measuring, and it is what "works flawlessly with
human conversation" means to a user: the app does not interrupt an ordinary chat.

  python score_human_eval.py --url http://127.0.0.1:8020/detect --label v5
"""
import argparse, json, sys, io, time
from collections import defaultdict
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
MR = ("money", "ride")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020/detect")
    ap.add_argument("--label", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    convs = json.loads((ROOT / "data/eval/human_eval.json").read_text(encoding="utf-8"))
    if a.limit:
        convs = convs[:a.limit]

    tag = f"{a.label}_{int(time.time())}"
    by_scen = defaultdict(lambda: [0, 0])
    fire_ok = fire_n = quiet_ok = quiet_n = 0
    noise_turns = noise_total = quiet_turns = 0
    misses, noise = [], []
    decisions = []

    for ci, c in enumerate(convs):
        room = f"he_{tag}_{ci}"
        fired_any, fired_right = False, False
        want_intent = None
        # The intent this conversation is built around, known before replaying it.
        want_expected = next((t["fire"][0] for t in c["turns"] if t.get("fire")), None)
        for ti, t in enumerate(c["turns"]):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": t["text"], "room_id": room,
                    "sender": str(t.get("sender", "A")),
                    "message_id": f"{tag}_{ci}_{ti}"}).json()
            except Exception:
                continue
            got = [i for i in (d.get("intents") or []) if i in MR]
            noise_total += 1
            if t.get("fire"):
                want_intent = t["fire"][0]
            if got:
                fired_any = True
                if want_expected and want_expected in got:
                    fired_right = True
                elif not c["expects_fire"]:
                    # Only a QUIET conversation can produce noise. In a fire
                    # conversation the commitment lands wherever the writer put it -
                    # usually mid-thread, with chat continuing after - so demanding it
                    # on one exact turn scored correct fires as both a miss AND a false
                    # prompt. That was a scoring bug, not model behaviour.
                    noise_turns += 1
                    noise.append((c["scenario"], got, t["text"][:58]))

        if not c["expects_fire"]:
            quiet_turns += len(c["turns"])
        if c["expects_fire"]:
            fire_n += 1
            ok = fired_right
            fire_ok += ok
            if not ok:
                misses.append((c["scenario"], want_intent,
                               c["turns"][-1]["text"][:58]))
        else:
            quiet_n += 1
            ok = not fired_any
            quiet_ok += ok
        # Every conversation, pass or fail. Diffing two models needs the passes too:
        # a conversation both models miss is a data problem, one only v11 misses is a
        # regression, and the saved file could not tell them apart.
        decisions.append({
            "i": ci,
            "scenario": c["scenario"],
            "expects_fire": bool(c["expects_fire"]),
            "want": want_expected,
            "fired": bool(fired_any),
            "correct": bool(ok),
            "n_turns": len(c["turns"]),
            "fire_text": next((t["text"] for t in c["turns"] if t.get("fire")), None),
            "last_text": c["turns"][-1]["text"],
        })
        by_scen[c["scenario"]][0] += ok
        by_scen[c["scenario"]][1] += 1

    print(f"\n{'='*68}\n  {a.label}   {len(convs)} conversations, {noise_total} turns\n{'='*68}")
    print(f"  FIRES WHEN IT SHOULD   {fire_ok:>4}/{fire_n:<4} = {fire_ok/max(fire_n,1):6.1%}")
    print(f"  STAYS QUIET            {quiet_ok:>4}/{quiet_n:<4} = {quiet_ok/max(quiet_n,1):6.1%}")
    print(f"  overall                {(fire_ok+quiet_ok):>4}/{len(convs):<4} = "
          f"{(fire_ok+quiet_ok)/max(len(convs),1):6.1%}")
    print(f"  false prompts          {noise_turns} in quiet conversations = "
          f"{noise_turns/max(quiet_turns,1)*1000:.1f} per 1000 quiet turns")

    print("\n  by scenario")
    for k in sorted(by_scen, key=lambda x: by_scen[x][0] / max(by_scen[x][1], 1)):
        ok, n = by_scen[k]
        print(f"    {k:26} {ok:>3}/{n:<3} {ok/max(n,1):6.1%}")

    if noise:
        print("\n  --- fired during a quiet conversation ---")
        for scen, got, txt in noise[:15]:
            print(f"    {scen:24} {str(got):10} {txt}")
    if misses:
        print("\n  --- should have fired, stayed silent ---")
        for scen, want, txt in misses[:15]:
            print(f"    {scen:24} want={str(want):8} {txt}")

    if a.save:
        (ROOT / "data/eval" / a.save).write_text(json.dumps(
            {"label": a.label, "fire": [fire_ok, fire_n], "quiet": [quiet_ok, quiet_n],
             "by_scenario": {k: v for k, v in by_scen.items()},
             "noise": noise, "misses": misses, "decisions": decisions}, indent=1, ensure_ascii=False),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
