"""Every conv-layer defect, from every real source, scored against the live model.

This exists because six retrains were built from lists nobody re-measured. On 2026-09-01
five separate "defects" dissolved when checked properly:

    11 training targets      -> 5 of 9 already worked
    ride requests w/o "cab"  -> 0 occurrences in 19,000 real messages
    offer flow 30/30 dead    -> 1/5 and 15/30 with realistic context
    "I'll be sending..."     -> 0.077 alone, 0.996 in its real room
    ride_past needs contrast -> model gets it 12/12

Every one came from scoring a phrase with no conversation in front of it. So this pulls
candidates ONLY from places a real user or a real test produced them:

  dogfood   messages the classifier scored >= 0.9 on that produced no prompt, and
            prompts that fired where the log says nothing should have. Real traffic,
            four days of it.
  batteries the conv-layer failures in tests/test_basics.py, which are conversations
  offers    the 5x6 offer matrix, run padded

and then RE-SCORES each one against the running model, in context, so a candidate that
the model already handles never reaches the generator. Anything it cannot reproduce is
dropped with the reason recorded, not carried forward on the strength of a memory.

    python data_gen/inventory_conv_v12.py --url http://127.0.0.1:8960/detect
"""
import argparse, collections, glob, io, json, re, sys, time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
MR = ("money", "ride")
TESTP = ("g15_", "df_", "si_", "rv_", "pre_", "gate_", "echo_", "rq_", "tri_", "cy_",
         "cz_", "rc_", "dbg", "rep_", "rr_", "idchk", "ov_", "oa_", "of_", "basic_",
         "bm_", "mp_", "snd_", "doc_", "dc_", "nat_", "po_", "pa_", "bv_", "vc_",
         "t85_", "tc", "L", "padchk_", "rpl", "dm_verify", "dm_fix", "dm_selfack",
         # WhatsApp exports replayed into the log for measurement. These are Akash's
         # own private chats with third parties - Anupam, the Everts roommates - given
         # to understand how people really text, NOT app traffic. They are mostly
         # Telugu written in English letters, and training data has been English-only
         # by rule. They must never reach a generator. Room ids look like
         # nw_1787713538_Anupam_5 and rw_1787714247_30.
         "nw_", "rw_", "wa_", "anupam", "everts", "rakesh")

# Same reason, from the other direction: dogfood_live.jsonl is where those replays were
# written. The dated logs are real app traffic; this one is not.
SKIP_LOGS = ("dogfood_live",)

# Romanised Telugu/Hindi, which the dated logs also contain a little of. Matching on
# function words rather than trying to detect a language - these appear in almost every
# such message and in almost no English one.
ROMANISED = re.compile(r"\b(cheskuni|cheskovachu|chesam|chestha|chesi|ostad|ostunna|"
                       r"annaru|antha|anta|enti|ledu|kuda|vachi|vasta|repu|eroj|"
                       r"paisal|matladuthunav|karo|kiya|nahi|hai|hoga|bhej|kar)\b", re.I)

# conversation that is about nothing, to fill the ten-message window the way a real
# room does. Without it every candidate is scored as though it were the whole chat.
PAD = [("A", "hey"), ("B", "hey whats up"), ("A", "not much just got back"),
       ("B", "nice how was it"), ("A", "pretty good tbh, long day though"),
       ("B", "same here honestly")]

# app chatter - the team is its own user base, so a lot of the log is people discussing
# the product. Those are not user defects and must not become training data.
CHATTER = re.compile(r"\bintent|apk|wifi|mobile data|works now|notification|build|"
                     r"\blog\b|test|bug|deploy|screen|app\b|samyak|andril|gowtham", re.I)


def send(url, room, turns, tag):
    last = {}
    for i, (spk, txt) in enumerate(turns):
        try:
            last = requests.post(url, timeout=60, json={
                "text": txt, "room_id": room, "sender": spk,
                "message_id": f"{tag}_{i}"}).json()
        except Exception:
            return {}
    return last


def score(url, turns, intent, tag, pad=True):
    room = f"inv_{tag}"
    seq = (PAD + turns) if pad else turns
    d = send(url, room, seq, tag)
    conv = float(((d.get("conversation_state") or {}).get("scores") or {})
                 .get(intent, 0) or 0)
    fired = intent in [x for x in (d.get("intents") or []) if x in MR]
    return conv, fired


def from_dogfood():
    """Rooms from the real logs, with the message index that looks wrong."""
    out = []
    for lg in sorted(glob.glob(str(ROOT / "data/eval/dogfood*.jsonl"))):
        if any(s in Path(lg).stem for s in SKIP_LOGS):
            print(f"    skipping {Path(lg).name} - replayed WhatsApp exports, not app traffic")
            continue
        rooms = collections.OrderedDict()
        for line in open(lg, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            rid = str(r.get("room") or r.get("room_id") or "")
            if rid and not rid.startswith(TESTP):
                rooms.setdefault(rid, []).append(r)
        for rid, msgs in rooms.items():
            for i, m in enumerate(msgs):
                txt = (m.get("text") or "").strip()
                if not txt or len(txt) > 120:
                    continue
                # Chatter has to be judged on the WINDOW, not the message. "Oh I got it
                # now" is innocuous alone and is app-debugging talk when it follows
                # "But payment intent does appear" - and the window is what the model
                # scores, so the window is what decides.
                window = " ".join((k.get("text") or "") for k in msgs[max(0, i - 5):i + 1])
                if CHATTER.search(window) or ROMANISED.search(window):
                    continue
                conv = m.get("conv") or {}
                fired = [x for x in (m.get("fired") or []) if x in MR]
                for intent in MR:
                    s = float(conv.get(intent, 0) or 0)
                    if s >= 0.90 and not fired:
                        out.append(("dogfood/silent", Path(lg).stem[-5:], rid,
                                    msgs[max(0, i - 5):i + 1], intent, True))
                    elif intent in fired and s < 0.50:
                        out.append(("dogfood/lowfire", Path(lg).stem[-5:], rid,
                                    msgs[max(0, i - 5):i + 1], intent, False))
    return out


def from_batteries():
    """The conv-layer failures in tests/, and the offer matrix.

    Real traffic yields only a couple of conv defects - the deployed model mostly works,
    and half the team's own log is product discussion. The batteries are the other
    legitimate source: every case in them is a conversation, they were written from
    rulings and screenshots rather than invented at a prompt, and they are the thing
    v12 will be judged against. They are NOT a substitute for real data; they are the
    curated half of it.
    """
    out = []
    # offer matrix - 5 offers x 6 acceptances, the shape that measured 1/5 and 15/30
    OFFERS = [("shall i transfer it now?", "money"),
              ("want me to send you the money?", "money"),
              ("let me send you 500", "money"),
              ("should i book you a cab?", "ride"),
              ("let me book you a cab", "ride")]
    ACCS = ["yes please", "yes please send it", "yeah do it", "sure", "ok please", "yes"]
    for off, intent in OFFERS:
        # the offer alone must NOT fire
        out.append(("battery/offer-waits", "spec3c", "-",
                    [{"sender": "A", "text": off}], intent, False))
        # each acceptance of it MUST fire
        for acc in ACCS:
            out.append(("battery/offer-accept", "spec3c", "-",
                        [{"sender": "A", "text": off}, {"sender": "B", "text": acc}],
                        intent, True))
    # acceptance vocabulary that the basics battery still fails on
    for acc in ("\U0001F44D", "ok will send", "yes please send"):
        out.append(("battery/acceptance", "basics", "-",
                    [{"sender": "B", "text": "can you send me 500"},
                     {"sender": "A", "text": acc}], "money", True))
    # modality: deliberation must not open a ride
    for txt in ("we should book a cab next time", "maybe ill take a cab later",
                "thinking of calling an uber after work"):
        out.append(("battery/modality", "basics", "-",
                    [{"sender": "A", "text": txt}, {"sender": "B", "text": "yeah"}],
                    "ride", False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="data_gen/conv_v12/INVENTORY.json")
    a = ap.parse_args()
    tag = int(time.time())
    verified, dropped = [], []

    print("  === source 1: real dogfood traffic ===")
    cands = from_dogfood()
    print(f"  candidates before re-scoring: {len(cands)}")
    bat = from_batteries()
    print("")
    print("  === source 2: batteries and the offer matrix ===")
    print(f"  candidates before re-scoring: {len(bat)}")
    cands = cands + bat
    seen = set()
    for kind, day, rid, msgs, intent, want_fire in cands:
        turns = [(str(m.get("sender")), m.get("text") or "") for m in msgs]
        key = (turns[-1][1].lower(), intent)
        if key in seen:
            continue
        seen.add(key)
        conv, fired = score(a.url, turns, intent, f"{tag}_{len(seen)}")

        # "scored high, did not fire" is NOT automatically a defect. If a prompt already
        # fired in this exchange, the suppression is the echo window doing its job, and
        # generating from it would teach the model to fire on duplicates.
        #
        # Real examples this catches:
        #   'You need the payment right sending now'  after  'Sending now'
        #   "Okay, I'll arrange it for 8:30pm"        after  the booking already fired
        #
        # Replay the same turns and see whether anything fired BEFORE the last message.
        prior_fire = False
        if want_fire and not fired and len(turns) > 1:
            rm = f"pri_{tag}_{len(seen)}"
            for j, (spk, txt) in enumerate(PAD + turns[:-1]):
                try:
                    d = requests.post(a.url, timeout=60, json={
                        "text": txt, "room_id": rm, "sender": spk,
                        "message_id": f"{tag}p_{len(seen)}_{j}"}).json()
                except Exception:
                    continue
                if [x for x in (d.get("intents") or []) if x in MR]:
                    prior_fire = True
        # a defect only counts if it still reproduces on the CURRENT model, in context,
        # and is not simply a duplicate of a prompt that already appeared
        still_wrong = ((want_fire and not fired and not prior_fire)
                       or ((not want_fire) and fired))
        rec = {"source": kind, "day": day, "room": rid, "intent": intent,
               "want": "fire" if want_fire else "quiet",
               "turns": turns[-4:], "conv": round(conv, 3), "fired": fired,
               "prior_fire": prior_fire}
        (verified if still_wrong else dropped).append(rec)
    print(f"  reproduce on the live model : {len(verified)}")
    print(f"  no longer reproduce, dropped: {len(dropped)}")

    print("\n  === verified defects, by shape ===")
    byshape = collections.Counter((v["source"], v["intent"], v["want"]) for v in verified)
    for (src, intent, want), n in byshape.most_common():
        print(f"    {src:18} {intent:5} want={want:5}  {n}")

    print("\n  === the actual messages ===")
    for v in verified[:24]:
        last = v["turns"][-1][1][:58]
        prev = v["turns"][-2][1][:34] if len(v["turns"]) > 1 else ""
        print(f"    [{v['day']}/{v['room']}] {v['intent']} want={v['want']:5} "
              f"conv={v['conv']:.3f}  {last!r}")
        if prev:
            print(f"        after: {prev!r}")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"verified": verified, "dropped": dropped},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    print(f"  {len(verified)} verified defects -> these, and only these, get generated for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
